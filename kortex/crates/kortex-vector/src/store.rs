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
        let path = dir.join(format!("kortex_mmap_test_{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let s = CodeStore::mmap_from(&path, &bytes).unwrap();
        assert!(s.is_mmap());
        assert_eq!(s.as_slice(), bytes.as_slice());
        drop(s);
        let _ = std::fs::remove_file(&path);
    }
}
