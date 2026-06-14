//! Code storage: in-RAM versus memory-mapped.
//!
//! The idle-RAM budget (< 250 MB at ~5M vectors) only closes if the bulk of the
//! index lives on disk and is demand-paged via `mmap`, so resident memory is the
//! working set, not the whole index. This module provides a read-only mmap of a
//! byte buffer (written to a temp file) so the bench can exercise — and prove out
//! — the cold/mmap path, not just an in-RAM `Vec<u8>`.

/// A read-only view over some bytes, either owned in RAM or memory-mapped.
pub enum CodeStore {
    Ram(Vec<u8>),
    #[cfg(unix)]
    Mmap(MmapBytes),
}

impl CodeStore {
    pub fn ram(bytes: Vec<u8>) -> Self {
        CodeStore::Ram(bytes)
    }

    /// Persist `bytes` to `path` and mmap it read-only. Falls back to RAM on
    /// non-unix targets.
    #[cfg(unix)]
    pub fn mmap_from(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<Self> {
        std::fs::write(path, bytes)?;
        Ok(CodeStore::Mmap(MmapBytes::open(path)?))
    }

    #[cfg(not(unix))]
    pub fn mmap_from(_path: &std::path::Path, bytes: &[u8]) -> std::io::Result<Self> {
        Ok(CodeStore::Ram(bytes.to_vec()))
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            CodeStore::Ram(v) => v,
            #[cfg(unix)]
            CodeStore::Mmap(m) => m.as_slice(),
        }
    }

    pub fn is_mmap(&self) -> bool {
        match self {
            CodeStore::Ram(_) => false,
            #[cfg(unix)]
            CodeStore::Mmap(_) => true,
        }
    }
}

#[cfg(unix)]
pub struct MmapBytes {
    ptr: *mut libc::c_void,
    len: usize,
}

#[cfg(unix)]
impl MmapBytes {
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        use std::os::unix::ffi::OsStrExt;
        let len = std::fs::metadata(path)?.len() as usize;
        if len == 0 {
            // mmap of length 0 is invalid; represent as an empty, valid view.
            return Ok(MmapBytes {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            });
        }
        let mut cpath: Vec<u8> = path.as_os_str().as_bytes().to_vec();
        cpath.push(0);
        // SAFETY: standard open/mmap/close sequence; we check all return codes
        // and only expose the mapping as an immutable byte slice for `len` bytes.
        unsafe {
            let fd = libc::open(cpath.as_ptr() as *const libc::c_char, libc::O_RDONLY);
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            );
            libc::close(fd);
            if ptr == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }
            Ok(MmapBytes { ptr, len })
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` is a valid read-only mapping of exactly `len` bytes,
        // alive for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

#[cfg(unix)]
impl Drop for MmapBytes {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: unmapping the same region we mapped.
            unsafe {
                libc::munmap(self.ptr, self.len);
            }
        }
    }
}

// The mapping is read-only and never mutated through the pointer, so sharing
// across threads is sound.
#[cfg(unix)]
unsafe impl Send for MmapBytes {}
#[cfg(unix)]
unsafe impl Sync for MmapBytes {}

// ---------------------------------------------------------------------------
// FloatMmapStore — full-precision vectors served from disk for the rerank stage
// ---------------------------------------------------------------------------

/// Read-only float32 vectors backed by a memory-mapped file (little-endian,
/// row-major, no header). This is the "precise" stage of the two-stage
/// pipeline at scale: only the shortlisted candidates' pages are touched, so
/// resident memory stays a tiny working set while the full-precision set —
/// gigabytes at 5M vectors — lives on disk.
pub struct FloatMmapStore {
    view: CodeStore,
    dim: usize,
    count: usize,
}

impl FloatMmapStore {
    /// mmap an existing raw f32 file. The file length must be a whole number
    /// of `dim`-sized rows.
    pub fn open(path: &std::path::Path, dim: usize) -> std::io::Result<Self> {
        #[cfg(unix)]
        let view = CodeStore::Mmap(MmapBytes::open(path)?);
        #[cfg(not(unix))]
        let view = CodeStore::Ram(std::fs::read(path)?);
        let len = view.as_slice().len();
        let row = dim * 4;
        if dim == 0 || !len.is_multiple_of(row) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("float file length {len} is not a multiple of row size {row}"),
            ));
        }
        let count = len / row;
        Ok(FloatMmapStore { view, dim, count })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Inner product of `query` with stored vector `id`, decoded on the fly
    /// from the mapped bytes (no copy, no alignment assumptions).
    pub fn dot_with(&self, id: usize, query: &[f32]) -> f32 {
        debug_assert_eq!(query.len(), self.dim);
        let bytes = self.view.as_slice();
        let base = id * self.dim * 4;
        let mut acc = 0.0f32;
        for (d, q) in query.iter().enumerate() {
            let p = base + d * 4;
            acc += q * f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        }
        acc
    }
}

impl crate::quant::Quantizer for FloatMmapStore {
    fn name(&self) -> String {
        "float32-mmap".into()
    }
    fn count(&self) -> usize {
        self.count
    }
    fn bytes_per_vector(&self) -> f64 {
        // On-disk footprint; resident is only the demand-paged working set.
        (self.dim * 4) as f64
    }
    fn scores(&self, query: &[f32]) -> Vec<f32> {
        (0..self.count).map(|i| self.dot_with(i, query)).collect()
    }
    fn scores_subset(&self, query: &[f32], ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .map(|&i| self.dot_with(i as usize, query))
            .collect()
    }
}

/// Best-effort free disk space at `path` (used by the scale bench to refuse
/// writing a multi-GB vector file onto a nearly-full disk).
#[cfg(unix)]
pub fn available_disk_bytes(path: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let mut cpath: Vec<u8> = path.as_os_str().as_bytes().to_vec();
    cpath.push(0);
    // SAFETY: statvfs writes into the zeroed struct we own; we check the rc.
    unsafe {
        let mut st: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(cpath.as_ptr() as *const libc::c_char, &mut st) == 0 {
            // The `as u64` casts are load-bearing on 32-bit targets, where the
            // statvfs fields are `c_ulong` (u32); newer clippy only sees the
            // 64-bit case where `c_ulong == u64` and flags them as redundant.
            #[allow(clippy::unnecessary_cast)]
            Some(st.f_bavail as u64 * st.f_frsize as u64)
        } else {
            None
        }
    }
}

#[cfg(not(unix))]
pub fn available_disk_bytes(_path: &std::path::Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_roundtrip() {
        let s = CodeStore::ram(vec![1, 2, 3, 4]);
        assert_eq!(s.as_slice(), &[1, 2, 3, 4]);
        assert!(!s.is_mmap());
    }

    #[cfg(unix)]
    #[test]
    fn mmap_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("skinki_mmap_test_{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let s = CodeStore::mmap_from(&path, &bytes).unwrap();
        assert!(s.is_mmap());
        assert_eq!(s.as_slice(), bytes.as_slice());
        drop(s);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn float_mmap_store_matches_in_ram_dots() {
        use crate::quant::Quantizer;
        let dim = 8;
        let rows: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.25).collect())
            .collect();
        let mut bytes = Vec::new();
        for r in &rows {
            for v in r {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let path = std::env::temp_dir().join(format!("skinki_fmm_{}.f32", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let store = FloatMmapStore::open(&path, dim).unwrap();
        assert_eq!(store.count(), 5);
        let query: Vec<f32> = (0..dim).map(|d| 1.0 - d as f32 * 0.1).collect();
        for (i, r) in rows.iter().enumerate() {
            let expect: f32 = r.iter().zip(&query).map(|(a, b)| a * b).sum();
            assert!((store.dot_with(i, &query) - expect).abs() < 1e-4);
        }
        let sub = store.scores_subset(&query, &[4, 0]);
        assert_eq!(sub.len(), 2);
        drop(store);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn float_mmap_store_rejects_ragged_files() {
        let path = std::env::temp_dir().join(format!("skinki_fmm_bad_{}.f32", std::process::id()));
        std::fs::write(&path, [0u8; 10]).unwrap();
        assert!(FloatMmapStore::open(&path, 4).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn disk_space_probe_returns_something_on_unix() {
        let avail = available_disk_bytes(&std::env::temp_dir());
        #[cfg(unix)]
        assert!(avail.unwrap_or(0) > 0);
        #[cfg(not(unix))]
        assert!(avail.is_none());
    }
}
