//! The C-ABI boundary (`kortex.h`). **All `unsafe` in this crate lives here.**
//!
//! Design recap (locked in `specs/STAGE_6.md` section 3):
//! - `kx_engine*` is an opaque handle; callers never see Rust types.
//! - `kx_search` writes into a **caller-allocated** `out_ids[k]` buffer and
//!   reports the filled length via `*out_len` — no engine-allocated memory to
//!   free, so the ownership story is "the caller owns everything except the
//!   `kx_engine*`, which `kx_free_engine` destroys."
//! - Status codes: `0` = OK, negative = error; `kx_last_error()` returns a
//!   thread-local, NUL-terminated description of the most recent error on the
//!   calling thread.
//!
//! ## Panic safety
//!
//! Every `extern "C"` entry point wraps its body in
//! [`std::panic::catch_unwind`] and converts a caught panic into
//! `KX_ERR_PANIC`. This is the "wrap every body" option from the spec (the
//! alternative, `panic = "abort"`, would kill the whole host process on a
//! Rust-side bug — too blunt for a library embedded in a larger app). All the
//! engine code below this boundary is safe Rust and `Result`-based, so a panic
//! here would only come from an actual bug (e.g. an unwrap on attacker-free,
//! self-generated data) or, conceivably, an allocator abort; `catch_unwind`
//! cannot save us from the latter, but it does turn the former into a normal
//! error return instead of unwinding into C, which is undefined behavior.
//!
//! ## `unsafe` inventory (for R1 review)
//!
//! 1. `kx_open`: dereferences `out_engine` (write) and reads `index_dir` as a
//!    C string. Both are raw-pointer reads/writes whose validity is the
//!    caller's contract per `kortex.h`; we null-check before use and bail out
//!    with an error code (not UB) if either is null.
//! 2. `kx_search`: dereferences `engine` (read-only borrow of the boxed
//!    `Engine`), reads `query` as a `&[f32]` slice of length `dim`, and writes
//!    up to `k` `u32`s into `out_ids` plus one `usize` into `out_len`. All
//!    pointer/length pairs come straight from the caller per the header
//!    contract; we null-check pointers and treat `k == 0` / `dim == 0` as
//!    trivially valid (zero-length slices), never forming a slice from a null
//!    pointer with nonzero length.
//! 3. `kx_free_engine`: reconstructs the `Box<Engine>` via `Box::from_raw` and
//!    drops it. Sound iff `engine` is either null (no-op) or a pointer
//!    previously returned by `kx_open` and not yet freed — this is the
//!    standard "one `Box::into_raw`/`Box::from_raw` pair" pattern; `kx_open` is
//!    the only function that produces these pointers (via `Box::into_raw`).
//! 4. `kx_last_error` / `kx_version`: build a `CString` (safe) and hand out its
//!    raw pointer. The `CString` for `kx_last_error` is owned by thread-local
//!    storage (see `error.rs`) so the pointer stays valid until the next error
//!    on this thread or thread exit, matching the documented "thread-local"
//!    contract. `kx_version`'s `CString` is `'static` (leaked once via
//!    `Box::leak`, see below) so its pointer is valid for the program's
//!    lifetime, matching a `const char*` with no free function in the header.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

use crate::engine::Engine;
use crate::error::{clear_last_error, set_last_error, with_last_error};

/// Opaque handle type named in `kortex.h` (`typedef struct kx_engine
/// kx_engine`). The C side only ever sees `kx_engine*`; the real type is
/// `Box<Engine>` reinterpreted via `Box::into_raw`/`Box::from_raw`. The
/// `snake_case` name matches the C struct tag in the header exactly.
#[allow(non_camel_case_types)]
pub struct kx_engine {
    inner: Engine,
}

// Status codes — keep in sync with the comments in `include/kortex.h`.
pub const KX_OK: i32 = 0;
pub const KX_ERR_NULL_PTR: i32 = -1;
pub const KX_ERR_INVALID_UTF8: i32 = -2;
pub const KX_ERR_OPEN_FAILED: i32 = -3;
pub const KX_ERR_SEARCH_FAILED: i32 = -4;
pub const KX_ERR_PANIC: i32 = -5;

/// Turn a caught panic payload into a human-readable string for
/// `kx_last_error`. Panic payloads are typically `&str` or `String`.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "kortex-ffi: panic with non-string payload".to_string()
    }
}

/// Open a Stage 1 two-stage index from `index_dir` (expects `rabitq.idx` and
/// `base.f32` in that directory). On success, `*out_engine` is set to a
/// freshly-allocated handle that must later be passed to
/// [`kx_free_engine`]. Returns `KX_OK` (0) on success, or a negative
/// `KX_ERR_*` code; on error, `kx_last_error()` describes the failure and
/// `*out_engine` is left untouched.
///
/// # Safety
/// `index_dir` must be a valid pointer to a NUL-terminated C string readable
/// for its full length, and `out_engine` must be a valid, non-null,
/// writable `kx_engine**`. Both are caller-supplied per the `kortex.h`
/// contract; see the module-level "unsafe inventory" item 1.
#[no_mangle]
pub unsafe extern "C" fn kx_open(index_dir: *const c_char, out_engine: *mut *mut kx_engine) -> i32 {
    let result = std::panic::catch_unwind(|| {
        clear_last_error();

        if index_dir.is_null() || out_engine.is_null() {
            set_last_error("kx_open: index_dir or out_engine is null");
            return KX_ERR_NULL_PTR;
        }

        // SAFETY: `index_dir` is a non-null pointer to a NUL-terminated C
        // string per the function contract; `CStr::from_ptr` reads only up to
        // the first NUL byte.
        let c_str = unsafe { CStr::from_ptr(index_dir) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("kx_open: index_dir is not valid UTF-8");
                return KX_ERR_INVALID_UTF8;
            }
        };

        match Engine::open(Path::new(path_str)) {
            Ok(engine) => {
                let boxed = Box::new(kx_engine { inner: engine });
                // SAFETY: `out_engine` is a non-null, writable `kx_engine**`
                // per the function contract. `Box::into_raw` hands ownership
                // of the boxed `Engine` to the caller; it is reclaimed exactly
                // once, by `kx_free_engine`, via the matching `Box::from_raw`.
                unsafe {
                    *out_engine = Box::into_raw(boxed);
                }
                KX_OK
            }
            Err(e) => {
                set_last_error(format!("kx_open: {e}"));
                KX_ERR_OPEN_FAILED
            }
        }
    });

    result.unwrap_or_else(|payload| {
        set_last_error(format!("kx_open: panic: {}", panic_message(payload)));
        KX_ERR_PANIC
    })
}

/// Run a two-stage search on `engine` for `query` (a `dim`-length float
/// vector), writing up to `k` result ids into the caller-allocated `out_ids`
/// buffer (which must have room for at least `k` entries) and the number of
/// ids actually written into `*out_len`. Returns `KX_OK` (0) on success or a
/// negative `KX_ERR_*` code; `kx_last_error()` describes any failure.
///
/// # Safety
/// `engine` must be a live pointer returned by [`kx_open`] and not yet passed
/// to [`kx_free_engine`]. `query` must point to `dim` readable `f32`s.
/// `out_ids` must point to at least `k` writable `u32`s, and `out_len` must be
/// a valid, non-null, writable `size_t`. See module-level "unsafe inventory"
/// item 2.
#[no_mangle]
pub unsafe extern "C" fn kx_search(
    engine: *const kx_engine,
    query: *const f32,
    dim: usize,
    k: usize,
    out_ids: *mut u32,
    out_len: *mut usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        clear_last_error();

        if engine.is_null() || query.is_null() || out_ids.is_null() || out_len.is_null() {
            set_last_error("kx_search: a required pointer argument is null");
            return KX_ERR_NULL_PTR;
        }

        // SAFETY: `engine` is a live handle from `kx_open` per the contract;
        // we only take a shared reference (no aliasing with the boxed data,
        // which nothing else mutates).
        let engine = unsafe { &*engine };

        // SAFETY: `query` points to `dim` readable `f32`s per the contract.
        // `dim == 0` yields an empty slice from a non-null pointer, which is
        // always valid.
        let query_slice = unsafe { std::slice::from_raw_parts(query, dim) };

        // SAFETY: `out_ids` points to at least `k` writable `u32`s per the
        // contract; same zero-length reasoning as above for `k == 0`.
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ids, k) };

        match engine.inner.search(query_slice, k, out_slice) {
            Ok(written) => {
                // SAFETY: `out_len` is a valid, non-null, writable `size_t`
                // pointer per the contract.
                unsafe {
                    *out_len = written;
                }
                KX_OK
            }
            Err(e) => {
                set_last_error(format!("kx_search: {e}"));
                KX_ERR_SEARCH_FAILED
            }
        }
    });

    result.unwrap_or_else(|payload| {
        set_last_error(format!("kx_search: panic: {}", panic_message(payload)));
        KX_ERR_PANIC
    })
}

/// Free an engine handle previously returned by [`kx_open`]. `engine` may be
/// null (no-op). After this call, `engine` must not be used again.
///
/// # Safety
/// `engine` must be either null or a pointer previously returned by
/// [`kx_open`] that has not already been freed. See module-level "unsafe
/// inventory" item 3.
#[no_mangle]
pub unsafe extern "C" fn kx_free_engine(engine: *mut kx_engine) {
    // Freeing is infallible by contract (no status code in the header), but a
    // malformed pointer could still make `Box::from_raw`'s internal bookkeeping
    // panic/abort; catch_unwind keeps that from unwinding across the boundary.
    let _ = std::panic::catch_unwind(|| {
        if engine.is_null() {
            return;
        }
        // SAFETY: `engine` is either null (checked above) or a pointer
        // previously produced by `Box::into_raw` in `kx_open` and not yet
        // freed, per the function contract. Reconstructing and dropping the
        // `Box` is exactly the matching `from_raw` for that `into_raw`.
        unsafe {
            drop(Box::from_raw(engine));
        }
    });
}

/// Return this thread's most recent error message as a NUL-terminated C
/// string, or an empty string if no error has been recorded since the last
/// call into this library on this thread. The returned pointer is valid until
/// the next `kx_*` call on this thread (or thread exit); callers must not free
/// it.
///
/// # Safety
/// This function takes no pointer arguments and performs no caller-supplied
/// pointer dereferences; it is safe to call from any thread. It is `unsafe`
/// only because it is `extern "C"` and returns a raw pointer (module-level
/// "unsafe inventory" item 4) — the pointer's validity window is documented
/// above.
#[no_mangle]
pub extern "C" fn kx_last_error() -> *const c_char {
    // `with_last_error` borrows the thread-local CString; we copy out a raw
    // pointer to its buffer (or to a static empty string), which stays valid
    // as long as no *new* error is recorded on this thread (the documented
    // contract). No `unsafe` block is needed here: `CStr::as_ptr` is a safe
    // method, and we never construct or dereference a raw pointer ourselves.
    with_last_error(|e| match e {
        Some(c) => c.as_ptr(),
        None => c"".as_ptr(),
    })
}

/// Return the crate version as a NUL-terminated, `'static` C string (e.g.
/// `"0.1.0"`). Callers must not free the returned pointer.
#[no_mangle]
pub extern "C" fn kx_version() -> *const c_char {
    // Leaked once into a 'static CString; this is a fixed, tiny, one-time
    // allocation for the process lifetime, matching a `const char*` with no
    // corresponding free function in the header (module-level "unsafe
    // inventory" item 4). No raw-pointer construction here either.
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| {
            CString::new(env!("CARGO_PKG_VERSION")).expect("CARGO_PKG_VERSION has no interior NUL")
        })
        .as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kortex_vector::quant::RaBitQBuilder;
    use kortex_vector::Rng;
    use std::ffi::CString;

    fn build_fixture(dir: &Path, dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
        std::fs::create_dir_all(dir).unwrap();
        let mut rng = Rng::new(seed);
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.unit() * 2.0 - 1.0).collect())
            .collect();

        let mut centroid = vec![0.0f32; dim];
        for row in &rows {
            for (c, x) in centroid.iter_mut().zip(row.iter()) {
                *c += x / n as f32;
            }
        }

        let mut builder = RaBitQBuilder::new(dim, 1, seed, centroid);
        for row in &rows {
            builder.push(row);
        }
        builder.finish().save(dir).unwrap();

        let mut buf = Vec::with_capacity(n * dim * 4);
        for row in &rows {
            for x in row {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(dir.join("base.f32"), buf).unwrap();

        rows
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("kortex_ffi_abi_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn version_is_nonempty() {
        let v = unsafe { CStr::from_ptr(kx_version()) };
        assert!(!v.to_str().unwrap().is_empty());
    }

    #[test]
    fn open_search_free_roundtrip() {
        let dim = 64;
        let n = 32;
        let dir = temp_dir("roundtrip");
        let rows = build_fixture(&dir, dim, n, 123);

        let c_path = CString::new(dir.to_str().unwrap()).unwrap();
        let mut engine: *mut kx_engine = std::ptr::null_mut();
        let rc = unsafe { kx_open(c_path.as_ptr(), &mut engine) };
        assert_eq!(rc, KX_OK);
        assert!(!engine.is_null());

        let k = 5;
        let mut out_ids = vec![0u32; k];
        let mut out_len = 0usize;
        let rc = unsafe {
            kx_search(
                engine,
                rows[0].as_ptr(),
                dim,
                k,
                out_ids.as_mut_ptr(),
                &mut out_len,
            )
        };
        assert_eq!(rc, KX_OK);
        assert_eq!(out_len, k);
        assert_eq!(out_ids[0], 0);

        unsafe { kx_free_engine(engine) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_null_args_errors() {
        clear_last_error();
        let mut engine: *mut kx_engine = std::ptr::null_mut();
        let rc = unsafe { kx_open(std::ptr::null(), &mut engine) };
        assert_eq!(rc, KX_ERR_NULL_PTR);
        let msg = unsafe { CStr::from_ptr(kx_last_error()) };
        assert!(msg.to_str().unwrap().contains("null"));
    }

    #[test]
    fn open_missing_dir_errors() {
        let dir = temp_dir("missing");
        let c_path = CString::new(dir.to_str().unwrap()).unwrap();
        let mut engine: *mut kx_engine = std::ptr::null_mut();
        let rc = unsafe { kx_open(c_path.as_ptr(), &mut engine) };
        assert_eq!(rc, KX_ERR_OPEN_FAILED);
        assert!(engine.is_null());
    }

    #[test]
    fn free_null_is_noop() {
        unsafe { kx_free_engine(std::ptr::null_mut()) };
    }
}
