/*
 * kortex.h - Stage 6 v0 C-ABI for the kortex engine.
 *
 * v0 surface: load a prebuilt Stage 1 two-stage index (a directory containing
 * `rabitq.idx` + `base.f32`, as produced by `RaBitQBuilder::save` + a raw
 * little-endian f32 row dump) and run searches against it.
 *
 * Memory ownership / error model (locked, see specs/STAGE_6.md section 3):
 *   - `kx_engine*` is an opaque handle; never dereference its fields.
 *   - `kx_search` writes results into a CALLER-ALLOCATED `out_ids[k]` buffer
 *     and reports how many ids were written via `*out_len`. There is no
 *     engine-allocated result buffer to free.
 *   - Status codes: 0 = OK, negative = error. On error, `kx_last_error()`
 *     returns a thread-local, NUL-terminated description of the failure.
 *   - `kx_last_error()` and `kx_version()` return pointers owned by the
 *     library; callers must NOT free them.
 */

#ifndef KORTEX_H
#define KORTEX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle. */
typedef struct kx_engine kx_engine;

/* Status codes returned by kx_open / kx_search. */
#define KX_OK               0
#define KX_ERR_NULL_PTR    -1
#define KX_ERR_INVALID_UTF8 -2
#define KX_ERR_OPEN_FAILED -3
#define KX_ERR_SEARCH_FAILED -4
#define KX_ERR_PANIC       -5

/*
 * Open a Stage 1 two-stage index from `index_dir` (a directory containing
 * `rabitq.idx` and `base.f32`). On success, `*out_engine` is set to a freshly
 * allocated handle that must later be passed to `kx_free_engine`.
 *
 * Returns KX_OK (0) on success, or a negative KX_ERR_* code on failure (see
 * kx_last_error() for details). `*out_engine` is left untouched on failure.
 */
int kx_open(const char* index_dir, kx_engine** out_engine);

/*
 * Run a two-stage search on `engine` for the `dim`-length float vector
 * `query`, writing up to `k` result ids (best match first) into the
 * caller-allocated `out_ids` buffer (which must have room for at least `k`
 * entries), and the number of ids actually written into `*out_len`.
 *
 * Returns KX_OK (0) on success, or a negative KX_ERR_* code on failure (see
 * kx_last_error() for details).
 */
int kx_search(kx_engine* engine,
               const float* query, size_t dim,
               size_t k,
               uint32_t* out_ids, size_t* out_len);

/*
 * Free an engine handle previously returned by kx_open. `engine` may be NULL
 * (no-op). After this call, `engine` must not be used again.
 */
void kx_free_engine(kx_engine* engine);

/*
 * Return this thread's most recent error message as a NUL-terminated C
 * string, or an empty string if no error has been recorded since the last
 * call into this library on this thread. The returned pointer is owned by
 * the library and valid until the next kx_* call on this thread (or thread
 * exit); callers must not free it.
 */
const char* kx_last_error(void);

/*
 * Return the crate version (e.g. "0.1.0") as a NUL-terminated, statically
 * allocated C string. Callers must not free the returned pointer.
 */
const char* kx_version(void);

#ifdef __cplusplus
}
#endif

#endif /* KORTEX_H */
