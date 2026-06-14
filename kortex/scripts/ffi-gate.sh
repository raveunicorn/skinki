#!/usr/bin/env bash
# Stage 6 FFI gate: build the kortex_ffi cdylib/staticlib, run the Rust
# C/Rust parity test, and (if python3 is available) run a Python `ctypes`
# parity check against the same fixture. Non-zero exit on any mismatch.
#
# Run from anywhere; paths are resolved relative to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KORTEX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$KORTEX_DIR"

echo "== ffi-gate: build kortex-ffi (cdylib + staticlib) =="
cargo build --release -p kortex-ffi

echo "== ffi-gate: Rust C-ABI / two_stage_search parity test =="
cargo test --release -p kortex-ffi --test ffi_parity
cargo test --release -p kortex-ffi --test symbol_existence

# Locate the built cdylib (name/extension is platform-dependent).
TARGET_DIR="$KORTEX_DIR/target/release"
case "$(uname -s)" in
    Darwin) LIB_NAME="libkortex_ffi.dylib" ;;
    MINGW*|MSYS*|CYGWIN*) LIB_NAME="kortex_ffi.dll" ;;
    *) LIB_NAME="libkortex_ffi.so" ;;
esac
LIB_PATH="$TARGET_DIR/$LIB_NAME"

if [[ ! -f "$LIB_PATH" ]]; then
    echo "ffi-gate: expected cdylib at $LIB_PATH, not found" >&2
    exit 1
fi
echo "ffi-gate: cdylib at $LIB_PATH"

if command -v python3 >/dev/null 2>&1; then
    echo "== ffi-gate: Python ctypes parity check =="

    FIXTURE_DIR="$(mktemp -d)"
    trap 'rm -rf "$FIXTURE_DIR"' EXIT

    # Build the fixture index + a JSON dump of {dim, k, query, expected_ids}
    # via the ffi-fixture helper binary (same construction as ffi_parity.rs).
    cargo run --release -p kortex-ffi --bin ffi-fixture -- "$FIXTURE_DIR/index" \
        > "$FIXTURE_DIR/fixture.json"

    KORTEX_FFI_LIB="$LIB_PATH" python3 "$SCRIPT_DIR/ffi_parity_check.py" \
        "$FIXTURE_DIR/index" "$FIXTURE_DIR/fixture.json"
else
    echo "ffi-gate: python3 not found, skipping Python parity check (Rust/C parity already verified)"
fi

echo "== ffi-gate: PASS =="
