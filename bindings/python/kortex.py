"""Pure-`ctypes` Python binding for the kortex Stage 6 C-ABI (`kortex.h`).

No PyO3, no build step: this module just `ctypes.CDLL`s the `kortex_ffi`
cdylib (built via `cargo build --release -p kortex-ffi`, producing
`libkortex_ffi.{so,dylib,dll}`) and wraps the five C functions declared in
`crates/kortex-ffi/include/kortex.h`:

    int  kx_open(const char* index_dir, kx_engine** out_engine);
    int  kx_search(kx_engine* engine, const float* query, size_t dim,
                    size_t k, uint32_t* out_ids, size_t* out_len);
    void kx_free_engine(kx_engine* engine);
    const char* kx_last_error(void);
    const char* kx_version(void);

Status codes: 0 = OK, negative = error (see KX_ERR_* below, mirroring the
header). On error, `kx_last_error()` gives a human-readable message.
"""

from __future__ import annotations

import ctypes
import os
import platform
from typing import List, Optional

KX_OK = 0
KX_ERR_NULL_PTR = -1
KX_ERR_INVALID_UTF8 = -2
KX_ERR_OPEN_FAILED = -3
KX_ERR_SEARCH_FAILED = -4
KX_ERR_PANIC = -5


def _default_lib_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libkortex_ffi.dylib"
    if system == "Windows":
        return "kortex_ffi.dll"
    return "libkortex_ffi.so"


def _load_library(lib_path: Optional[str]) -> ctypes.CDLL:
    """Load the kortex_ffi cdylib.

    `lib_path` may be an explicit path to the shared library, a directory
    containing it (using the platform-default file name), or `None` to use
    the `KORTEX_FFI_LIB` environment variable (falling back to the
    platform-default name on the default search path).
    """
    if lib_path is None:
        lib_path = os.environ.get("KORTEX_FFI_LIB")
    if lib_path is None:
        lib_path = _default_lib_name()
    elif os.path.isdir(lib_path):
        lib_path = os.path.join(lib_path, _default_lib_name())
    return ctypes.CDLL(lib_path)


class KortexError(RuntimeError):
    """Raised when a `kx_*` call returns a negative status code."""


class KortexEngine:
    """A loaded Stage 1 two-stage index, ready to search.

    Mirrors the Rust `Engine`/C `kx_engine*` lifecycle: `open()` allocates a
    native handle, `search()` runs `two_stage_search` against it, and
    `close()` (or `__exit__`/`__del__`) frees it. Results are written into a
    caller-allocated buffer on the C side and copied into a Python list here
    -- there is no engine-allocated memory for the caller to manage.
    """

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p, dim: int):
        self._lib = lib
        self._handle = handle
        self._dim = dim

    @classmethod
    def open(cls, index_dir: str, dim: int, lib_path: Optional[str] = None) -> "KortexEngine":
        """Open the two-stage index at `index_dir`. `dim` is the query/index
        dimensionality (the caller must know it; v0's `kx_open` does not
        return it -- it is read from the on-disk `rabitq.idx` on the Rust
        side, but not surfaced over the ABI yet)."""
        lib = _load_library(lib_path)
        _declare_signatures(lib)

        out_engine = ctypes.c_void_p()
        rc = lib.kx_open(index_dir.encode("utf-8"), ctypes.byref(out_engine))
        if rc != KX_OK:
            raise KortexError(f"kx_open failed ({rc}): {_last_error(lib)}")
        return cls(lib, out_engine, dim)

    def search(self, query: List[float], k: int) -> List[int]:
        """Two-stage search: returns up to `k` ids, best match first."""
        if len(query) != self._dim:
            raise ValueError(f"query has {len(query)} dims, engine expects {self._dim}")
        if self._handle is None:
            raise KortexError("search on a closed KortexEngine")

        query_arr = (ctypes.c_float * len(query))(*query)
        out_ids = (ctypes.c_uint32 * k)()
        out_len = ctypes.c_size_t()

        rc = self._lib.kx_search(
            self._handle,
            query_arr,
            ctypes.c_size_t(len(query)),
            ctypes.c_size_t(k),
            out_ids,
            ctypes.byref(out_len),
        )
        if rc != KX_OK:
            raise KortexError(f"kx_search failed ({rc}): {_last_error(self._lib)}")
        return list(out_ids[: out_len.value])

    def close(self) -> None:
        if self._handle is not None:
            self._lib.kx_free_engine(self._handle)
            self._handle = None

    def __enter__(self) -> "KortexEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; `close()` is idempotent.
        try:
            self.close()
        except Exception:
            pass


def kortex_version(lib_path: Optional[str] = None) -> str:
    lib = _load_library(lib_path)
    _declare_signatures(lib)
    return lib.kx_version().decode("utf-8")


def _last_error(lib: ctypes.CDLL) -> str:
    ptr = lib.kx_last_error()
    return ptr.decode("utf-8") if ptr else ""


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Set `argtypes`/`restype` for the C functions we call.

    Without this, `ctypes` assumes `c_int` return types and untyped pointer
    arguments, which happens to work for these simple signatures on most
    platforms but is not guaranteed (e.g. pointer truncation on 64-bit
    Windows). Declaring them explicitly matches `kortex.h` exactly.
    """
    lib.kx_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.kx_open.restype = ctypes.c_int

    lib.kx_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.kx_search.restype = ctypes.c_int

    lib.kx_free_engine.argtypes = [ctypes.c_void_p]
    lib.kx_free_engine.restype = None

    lib.kx_last_error.argtypes = []
    lib.kx_last_error.restype = ctypes.c_char_p

    lib.kx_version.argtypes = []
    lib.kx_version.restype = ctypes.c_char_p
