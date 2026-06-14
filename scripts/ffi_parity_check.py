#!/usr/bin/env python3
"""Python `ctypes` parity check for scripts/ffi-gate.sh.

Loads the skinki_ffi cdylib (path via the SKINKI_FFI_LIB env var, set by
ffi-gate.sh), opens the fixture index built by the `ffi-fixture` helper
binary, runs `skinkiEngine.search`, and asserts the result matches the
`expected_ids` from the fixture's JSON dump (computed in Rust via
`two_stage_search` on the same index/query).

Usage: ffi_parity_check.py <index_dir> <fixture_json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))

from skinki import skinkiEngine  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ffi_parity_check.py <index_dir> <fixture_json>", file=sys.stderr)
        return 2

    index_dir, fixture_path = sys.argv[1], sys.argv[2]
    with open(fixture_path) as f:
        fixture = json.load(f)

    dim = fixture["dim"]
    k = fixture["k"]
    query = fixture["query"]
    expected_ids = fixture["expected_ids"]

    engine = skinkiEngine.open(index_dir, dim)
    try:
        got_ids = engine.search(query, k)
    finally:
        engine.close()

    if got_ids != expected_ids:
        print(f"MISMATCH: python ids {got_ids} != rust ids {expected_ids}", file=sys.stderr)
        return 1

    print(f"OK: python ids == rust ids ({got_ids})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
