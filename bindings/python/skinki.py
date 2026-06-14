"""Pure-`ctypes` Python binding for the skinki Stage 6 C-ABI (`skinki.h`).

No PyO3, no build step: this module just `ctypes.CDLL`s the `skinki_ffi`
cdylib (built via `cargo build --release -p skinki-ffi`, producing
`libskinki_ffi.{so,dylib,dll}`) and wraps the five C functions declared in
`crates/skinki-ffi/include/skinki.h`:

    int  sk_open(const char* index_dir, sk_engine** out_engine);
    int  sk_search(sk_engine* engine, const float* query, size_t dim,
                    size_t k, uint32_t* out_ids, size_t* out_len);
    void sk_free_engine(sk_engine* engine);
    const char* sk_last_error(void);
    const char* sk_version(void);

Status codes: 0 = OK, negative = error (see SK_ERR_* below, mirroring the
header). On error, `sk_last_error()` gives a human-readable message.
"""

from __future__ import annotations

import ctypes
import os
import platform
from typing import List, Optional

SK_OK = 0
SK_ERR_NULL_PTR = -1
SK_ERR_INVALID_UTF8 = -2
SK_ERR_OPEN_FAILED = -3
SK_ERR_SEARCH_FAILED = -4
SK_ERR_PANIC = -5


def _default_lib_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libskinki_ffi.dylib"
    if system == "Windows":
        return "skinki_ffi.dll"
    return "libskinki_ffi.so"


def _load_library(lib_path: Optional[str]) -> ctypes.CDLL:
    """Load the skinki_ffi cdylib.

    `lib_path` may be an explicit path to the shared library, a directory
    containing it (using the platform-default file name), or `None` to use
    the `SKINKI_FFI_LIB` environment variable (falling back to the
    platform-default name on the default search path).
    """
    if lib_path is None:
        lib_path = os.environ.get("SKINKI_FFI_LIB")
    if lib_path is None:
        lib_path = _default_lib_name()
    elif os.path.isdir(lib_path):
        lib_path = os.path.join(lib_path, _default_lib_name())
    return ctypes.CDLL(lib_path)


class skinkiError(RuntimeError):
    """Raised when a `sk_*` call returns a negative status code."""


class skinkiEngine:
    """A loaded Stage 1 two-stage index, ready to search.

    Mirrors the Rust `Engine`/C `sk_engine*` lifecycle: `open()` allocates a
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
    def open(cls, index_dir: str, dim: int, lib_path: Optional[str] = None) -> "skinkiEngine":
        """Open the two-stage index at `index_dir`. `dim` is the query/index
        dimensionality (the caller must know it; v0's `sk_open` does not
        return it -- it is read from the on-disk `rabitq.idx` on the Rust
        side, but not surfaced over the ABI yet)."""
        lib = _load_library(lib_path)
        _declare_signatures(lib)

        out_engine = ctypes.c_void_p()
        rc = lib.sk_open(index_dir.encode("utf-8"), ctypes.byref(out_engine))
        if rc != SK_OK:
            raise skinkiError(f"sk_open failed ({rc}): {_last_error(lib)}")
        return cls(lib, out_engine, dim)

    def search(self, query: List[float], k: int) -> List[int]:
        """Two-stage search: returns up to `k` ids, best match first."""
        if len(query) != self._dim:
            raise ValueError(f"query has {len(query)} dims, engine expects {self._dim}")
        if self._handle is None:
            raise skinkiError("search on a closed skinkiEngine")

        query_arr = (ctypes.c_float * len(query))(*query)
        out_ids = (ctypes.c_uint32 * k)()
        out_len = ctypes.c_size_t()

        rc = self._lib.sk_search(
            self._handle,
            query_arr,
            ctypes.c_size_t(len(query)),
            ctypes.c_size_t(k),
            out_ids,
            ctypes.byref(out_len),
        )
        if rc != SK_OK:
            raise skinkiError(f"sk_search failed ({rc}): {_last_error(self._lib)}")
        return list(out_ids[: out_len.value])

    def close(self) -> None:
        if self._handle is not None:
            self._lib.sk_free_engine(self._handle)
            self._handle = None

    def __enter__(self) -> "skinkiEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; `close()` is idempotent.
        try:
            self.close()
        except Exception:
            pass


def skinki_version(lib_path: Optional[str] = None) -> str:
    lib = _load_library(lib_path)
    _declare_signatures(lib)
    return lib.sk_version().decode("utf-8")


def _last_error(lib: ctypes.CDLL) -> str:
    ptr = lib.sk_last_error()
    return ptr.decode("utf-8") if ptr else ""


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Set `argtypes`/`restype` for the C functions we call.

    Without this, `ctypes` assumes `c_int` return types and untyped pointer
    arguments, which happens to work for these simple signatures on most
    platforms but is not guaranteed (e.g. pointer truncation on 64-bit
    Windows). Declaring them explicitly matches `skinki.h` exactly.
    """
    lib.sk_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.sk_open.restype = ctypes.c_int

    lib.sk_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.sk_search.restype = ctypes.c_int

    lib.sk_free_engine.argtypes = [ctypes.c_void_p]
    lib.sk_free_engine.restype = None

    lib.sk_last_error.argtypes = []
    lib.sk_last_error.restype = ctypes.c_char_p

    lib.sk_version.argtypes = []
    lib.sk_version.restype = ctypes.c_char_p
