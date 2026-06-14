//! Safe inner engine: loads a Stage 1 two-stage index and runs searches.
//!
//! This module is the "real" implementation behind the C-ABI in `ffi.rs`. It
//! is plain safe Rust, unit-testable on its own, and knows nothing about the
//! FFI boundary (no raw pointers, no `extern "C"`). `ffi.rs` is a thin,
//! `unsafe`-only adapter on top of this.

use std::path::Path;

use skinki_vector::quant::RaBitQ;
use skinki_vector::search::two_stage_search;
use skinki_vector::store::FloatMmapStore;

/// Default `refine` (shortlist size for the coarse stage) when none is given
/// by the caller. Matches the values used in `compress-bench`/`scale-bench`
/// for small-to-medium indexes; large indexes can still pass a larger value
/// via a future FFI extension, but v0's `sk_search` has no such parameter.
pub const DEFAULT_REFINE: usize = 256;

/// A loaded Stage 1 two-stage index, ready to search.
///
/// Wraps the coarse 1-bit RaBitQ index (`rabitq.idx`) and the full-precision
/// rerank store (`base.f32`), both loaded from the same `index_dir`.
pub struct Engine {
    coarse: RaBitQ,
    precise: FloatMmapStore,
    dim: usize,
    refine: usize,
}

// `RaBitQ`/`FloatMmapStore` don't derive `Debug`; a minimal hand-written impl
// (just the scalar fields) is enough for `unwrap_err`/`assert!` in tests and
// any future logging, without reaching into upstream crates.
impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("dim", &self.dim)
            .field("refine", &self.refine)
            .finish_non_exhaustive()
    }
}

/// Errors `Engine::open`/`Engine::search` can produce. `ffi.rs` maps these to
/// the integer status codes documented in `skinki.h`.
#[derive(Debug)]
pub enum EngineError {
    /// `index_dir/rabitq.idx` could not be loaded (missing/corrupt).
    LoadIndex(std::io::Error),
    /// `index_dir/base.f32` could not be mmap'd, or its size doesn't match
    /// `dim` (which is read off the loaded RaBitQ index).
    LoadFloatStore(std::io::Error),
    /// The query vector's length didn't match the index dimensionality.
    DimMismatch { expected: usize, got: usize },
    /// The caller-provided output buffer (`out_ids[k]`) is too small for `k`.
    BufferTooSmall { need: usize, have: usize },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::LoadIndex(e) => write!(f, "failed to load rabitq.idx: {e}"),
            EngineError::LoadFloatStore(e) => write!(f, "failed to load base.f32: {e}"),
            EngineError::DimMismatch { expected, got } => {
                write!(f, "query dim {got} does not match index dim {expected}")
            }
            EngineError::BufferTooSmall { need, have } => {
                write!(f, "output buffer has room for {have} ids, need {need}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

impl Engine {
    /// Load the two-stage index from `index_dir` (expects `rabitq.idx` and
    /// `base.f32` produced by the same build, e.g. via `RaBitQBuilder::save`
    /// and a raw little-endian f32 row dump). Uses [`DEFAULT_REFINE`].
    pub fn open(index_dir: &Path) -> Result<Self, EngineError> {
        Self::open_with_refine(index_dir, DEFAULT_REFINE)
    }

    /// Like [`Engine::open`] but with an explicit default `refine` (shortlist
    /// size) for [`Engine::search`].
    pub fn open_with_refine(index_dir: &Path, refine: usize) -> Result<Self, EngineError> {
        let coarse = RaBitQ::load(index_dir).map_err(EngineError::LoadIndex)?;
        let dim = coarse.dim();
        let precise = FloatMmapStore::open(&index_dir.join("base.f32"), dim)
            .map_err(EngineError::LoadFloatStore)?;
        Ok(Engine {
            coarse,
            precise,
            dim,
            refine,
        })
    }

    /// The query/index dimensionality this engine was loaded for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Two-stage search: returns up to `k` ids, best match first. `out` must
    /// have room for at least `k` entries; on success its first `k.min(n)`
    /// entries are filled and the filled length is returned.
    ///
    /// Mirrors `skinki_vector::search::two_stage_search(&rabitq, &floatstore,
    /// query, k, refine)` exactly (same inputs -> same ids).
    pub fn search(&self, query: &[f32], k: usize, out: &mut [u32]) -> Result<usize, EngineError> {
        if query.len() != self.dim {
            return Err(EngineError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if out.len() < k {
            return Err(EngineError::BufferTooSmall {
                need: k,
                have: out.len(),
            });
        }
        let ids = two_stage_search(&self.coarse, &self.precise, query, k, self.refine);
        out[..ids.len()].copy_from_slice(&ids);
        Ok(ids.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skinki_vector::quant::RaBitQBuilder;
    use skinki_vector::Rng;

    /// Build a tiny on-disk index (rabitq.idx + base.f32) in a fresh temp dir,
    /// matching the format `RaBitQBuilder::save` + a raw f32 dump produce
    /// (the same pattern `skinki-harness::run_scale_bench` uses).
    fn build_fixture(dir: &Path, dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
        std::fs::create_dir_all(dir).unwrap();
        let mut rng = Rng::new(seed);
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.unit() * 2.0 - 1.0).collect())
            .collect();

        // Centroid for residual centering, as RaBitQ expects.
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

        // Raw little-endian f32 dump, row-major.
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
        let dir = std::env::temp_dir().join(format!(
            "skinki_ffi_engine_{tag}_{}_{}",
            std::process::id(),
            tag.len() // tiny extra entropy to avoid cross-test collisions
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn open_and_search_returns_expected_ids() {
        let dim = 64;
        let n = 32;
        let dir = temp_dir("open_search");
        let rows = build_fixture(&dir, dim, n, 42);

        let engine = Engine::open(&dir).unwrap();
        assert_eq!(engine.dim(), dim);

        let mut out = vec![0u32; 5];
        let len = engine.search(&rows[0], 5, &mut out).unwrap();
        assert_eq!(len, 5);
        // Searching with the exact stored vector 0 should rank it first.
        assert_eq!(out[0], 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_rejects_dim_mismatch() {
        let dim = 64;
        let dir = temp_dir("dim_mismatch");
        build_fixture(&dir, dim, 16, 7);
        let engine = Engine::open(&dir).unwrap();

        let bad_query = vec![0.0f32; dim + 1];
        let mut out = vec![0u32; 4];
        let err = engine.search(&bad_query, 4, &mut out).unwrap_err();
        assert!(matches!(err, EngineError::DimMismatch { .. }));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_rejects_buffer_too_small() {
        let dim = 64;
        let dir = temp_dir("buf_small");
        let rows = build_fixture(&dir, dim, 16, 9);
        let engine = Engine::open(&dir).unwrap();

        let mut out = vec![0u32; 2];
        let err = engine.search(&rows[0], 4, &mut out).unwrap_err();
        assert!(matches!(
            err,
            EngineError::BufferTooSmall { need: 4, have: 2 }
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_missing_dir_errors() {
        let dir = temp_dir("missing");
        let err = Engine::open(&dir).unwrap_err();
        assert!(matches!(err, EngineError::LoadIndex(_)));
    }
}
