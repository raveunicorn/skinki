//! Dev utility: print the FNV-64 hash of the entries JSON per difficulty.
//! The V1 values pin the back-compat golden test in lib.rs; rerun this after
//! generator changes to confirm V1 is untouched (hashes must not move).

use kortex_corpus::{generate, Difficulty, GenConfig};

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}

fn main() {
    for difficulty in [Difficulty::V1, Difficulty::V2] {
        for (seed, years) in [(42u64, 2u32), (7, 3)] {
            let c = generate(&GenConfig {
                seed,
                years,
                entries_per_day: 2,
                difficulty,
            });
            let s = serde_json::to_string(&c.entries).unwrap();
            println!(
                "{difficulty:?} seed={seed} years={years} entries={} json_len={} fnv64={:#018x}",
                c.entries.len(),
                s.len(),
                fnv1a64(s.as_bytes())
            );
        }
    }
}
