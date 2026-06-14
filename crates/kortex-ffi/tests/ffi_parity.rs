//! Integration test: builds a tiny on-disk index, then verifies that the
//! `extern "C"` FFI path (`kx_open`/`kx_search`, called directly from Rust —
//! they are ordinary `unsafe extern "C" fn`s and callable in-process) returns
//! exactly the same ids as the pure-Rust `two_stage_search` on the same
//! index + query, per specs/STAGE_6.md's "cross-language equality" gate.

use std::ffi::CString;

use kortex_ffi::ffi::{kx_engine, kx_free_engine, kx_open, kx_search, KX_OK};
use kortex_vector::quant::{RaBitQ, RaBitQBuilder};
use kortex_vector::search::two_stage_search;
use kortex_vector::store::FloatMmapStore;
use kortex_vector::Rng;

/// Mirrors `kortex-harness::run_scale_bench`'s flat-index build: stream
/// deterministic rows into a 1-bit `RaBitQBuilder` against the dataset
/// centroid, `finish().save(dir)`, and separately dump the same rows as raw
/// little-endian f32 to `dir/base.f32` for the rerank stage.
fn build_index(dir: &std::path::Path, dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
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

#[test]
fn ffi_search_matches_two_stage_search() {
    let dim = 64;
    let n = 64;
    let seed = 2026;
    let k = 8;
    let refine = kortex_ffi::engine::DEFAULT_REFINE;

    let dir = std::env::temp_dir().join(format!("kortex_ffi_parity_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let rows = build_index(&dir, dim, n, seed);
    let query = rows[3].clone();

    // --- Rust reference path ---
    let rabitq = RaBitQ::load(&dir).unwrap();
    let floatstore = FloatMmapStore::open(&dir.join("base.f32"), dim).unwrap();
    let expected = two_stage_search(&rabitq, &floatstore, &query, k, refine);

    // --- FFI path (extern "C" functions, called directly since we're already
    // in Rust; cross-language callers go through the identical entry points
    // via the cdylib/staticlib) ---
    let c_path = CString::new(dir.to_str().unwrap()).unwrap();
    let mut engine: *mut kx_engine = std::ptr::null_mut();
    let rc = unsafe { kx_open(c_path.as_ptr(), &mut engine) };
    assert_eq!(rc, KX_OK, "kx_open failed");

    let mut out_ids = vec![0u32; k];
    let mut out_len = 0usize;
    let rc = unsafe {
        kx_search(
            engine,
            query.as_ptr(),
            dim,
            k,
            out_ids.as_mut_ptr(),
            &mut out_len,
        )
    };
    assert_eq!(rc, KX_OK, "kx_search failed");
    assert_eq!(out_len, expected.len());
    assert_eq!(&out_ids[..out_len], expected.as_slice());

    unsafe { kx_free_engine(engine) };
    let _ = std::fs::remove_dir_all(&dir);
}
