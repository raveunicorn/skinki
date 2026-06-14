//! Gate-only helper: build a tiny deterministic Stage 1 two-stage index
//! (`rabitq.idx` + `base.f32`) and print a JSON dump of `{dim, k, query,
//! expected_ids}` to stdout, where `expected_ids` is the pure-Rust
//! `two_stage_search` result on that index/query.
//!
//! `scripts/ffi-gate.sh` uses this to build a fixture both the Rust
//! integration test's pattern and the Python `ctypes` binding can search,
//! and to get a ground-truth id list for the Python parity check -- without
//! re-implementing index construction (or the deterministic data
//! generation) in Python.
//!
//! Usage: `ffi-fixture <out_dir>`

use skinki_ffi::engine::DEFAULT_REFINE;
use skinki_vector::quant::{RaBitQ, RaBitQBuilder};
use skinki_vector::search::two_stage_search;
use skinki_vector::store::FloatMmapStore;
use skinki_vector::Rng;

const DIM: usize = 64;
const N: usize = 64;
const SEED: u64 = 2026;
const K: usize = 8;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: ffi-fixture <out_dir>");
    let dir = std::path::PathBuf::from(out_dir);
    std::fs::create_dir_all(&dir).expect("create out_dir");

    // Same deterministic construction as the Rust integration test
    // (tests/ffi_parity.rs): stream rows into a 1-bit RaBitQ builder against
    // the dataset centroid, save it, and separately dump the rows as raw
    // little-endian f32 for the rerank stage.
    let mut rng = Rng::new(SEED);
    let rows: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..DIM).map(|_| rng.unit() * 2.0 - 1.0).collect())
        .collect();

    let mut centroid = vec![0.0f32; DIM];
    for row in &rows {
        for (c, x) in centroid.iter_mut().zip(row.iter()) {
            *c += x / N as f32;
        }
    }

    let mut builder = RaBitQBuilder::new(DIM, 1, SEED, centroid);
    for row in &rows {
        builder.push(row);
    }
    builder.finish().save(&dir).expect("save rabitq.idx");

    let mut buf = Vec::with_capacity(N * DIM * 4);
    for row in &rows {
        for x in row {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(dir.join("base.f32"), buf).expect("write base.f32");

    let query = rows[3].clone();

    let rabitq = RaBitQ::load(&dir).expect("load rabitq.idx");
    let floatstore = FloatMmapStore::open(&dir.join("base.f32"), DIM).expect("open base.f32");
    let expected_ids = two_stage_search(&rabitq, &floatstore, &query, K, DEFAULT_REFINE);

    let out = serde_json::json!({
        "dim": DIM,
        "k": K,
        "query": query,
        "expected_ids": expected_ids,
    });
    println!("{out}");
}
