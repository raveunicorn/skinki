//! Symbol-existence test (specs/STAGE_6.md invariant: "the header in
//! `include/kortex.h` exactly matches the exported symbols").
//!
//! This crate's `rlib`/`cdylib`/`staticlib` are all built from the same
//! `src/ffi.rs`, so the simplest deterministic check is: parse every `kx_*(`
//! function name out of the header, and assert each one is reachable as a
//! public item of `kortex_ffi::ffi` (i.e. the symbol the linker will export
//! from the cdylib/staticlib via `#[no_mangle]`/`extern "C"`). A fixed,
//! hand-maintained list keeps this independent of any external tool
//! (`nm`/`cbindgen`), matching the "no new deps" constraint.

use std::path::Path;

/// Every symbol declared in `include/kortex.h`. Kept in sync by hand; the
/// regex-extraction below cross-checks it against the header so a forgotten
/// header update (or a forgotten Rust export) fails this test.
const EXPECTED_SYMBOLS: &[&str] = &[
    "kx_open",
    "kx_search",
    "kx_free_engine",
    "kx_last_error",
    "kx_version",
];

/// Extract `kx_<name>(` identifiers from the header text (a tiny hand-rolled
/// scan is enough; full C parsing would be overkill and pull in a dep).
fn symbols_in_header(text: &str) -> Vec<String> {
    // Operate on raw bytes throughout (including the final `String::from_utf8`)
    // so we never slice into the middle of a multi-byte UTF-8 character
    // (the header's comments may contain non-ASCII punctuation).
    let mut found = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"kx_") {
            let start = i;
            let mut j = i;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            // Only count it as a declared symbol if immediately followed by
            // `(` (a function), skipping whitespace - this excludes the
            // `kx_engine` typedef and the KX_ERR_* macro names (which don't
            // start with lowercase `kx_`).
            let mut k = j;
            while k < bytes.len() && bytes[k] == b' ' {
                k += 1;
            }
            if k < bytes.len() && bytes[k] == b'(' {
                found.push(String::from_utf8(bytes[start..j].to_vec()).unwrap());
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    found.sort();
    found.dedup();
    found
}

#[test]
fn header_symbols_match_expected_list() {
    let header_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("include/kortex.h");
    let text = std::fs::read_to_string(&header_path).expect("read kortex.h");
    let header_syms = symbols_in_header(&text);

    let mut expected: Vec<String> = EXPECTED_SYMBOLS.iter().map(|s| s.to_string()).collect();
    expected.sort();

    assert_eq!(
        header_syms, expected,
        "kortex.h declared symbols != EXPECTED_SYMBOLS (update one or the other)"
    );
}

/// Assert each expected symbol is a real, callable item exported by
/// `kortex_ffi::ffi` — i.e. it resolves at compile time, which for
/// `#[no_mangle] extern "C"` functions means it will also be present in the
/// built `cdylib`/`staticlib` under that exact name.
#[test]
fn expected_symbols_are_exported_from_ffi_module() {
    use kortex_ffi::ffi::{kx_free_engine, kx_last_error, kx_open, kx_search, kx_version};

    // Referencing each function as a value proves it exists with the right
    // path/visibility; the `as usize` cast is just to use the value so the
    // compiler doesn't warn about unused imports.
    let ptrs: [usize; 5] = [
        kx_open as *const () as usize,
        kx_search as *const () as usize,
        kx_free_engine as *const () as usize,
        kx_last_error as *const () as usize,
        kx_version as *const () as usize,
    ];
    assert_eq!(ptrs.len(), EXPECTED_SYMBOLS.len());
    assert!(ptrs.iter().all(|&p| p != 0));
}
