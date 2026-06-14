//! Thread-local last-error slot.
//!
//! `sk_last_error` is documented as thread-local NUL-terminated string, so
//! errors from concurrent callers on different threads never clobber each
//! other. Everything here is safe Rust: the only "unsafe-adjacent" part is
//! that the returned pointer (in `ffi.rs`) must outlive the caller's read,
//! which we guarantee by storing the `CString` in thread-local storage for
//! the lifetime of the thread (replaced, never freed early).

use std::cell::RefCell;
use std::ffi::CString;

thread_local! {
    /// Holds the most recent error message for this thread, if any. `ffi.rs`
    /// hands out a pointer into the `CString`'s buffer; storing it here (and
    /// only ever *replacing* it, never dropping it out from under a live
    /// pointer mid-call) keeps that pointer valid until the next error on
    /// this thread or thread exit.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Record `msg` as this thread's last error. Embedded NUL bytes are stripped
/// (they can't occur in a NUL-terminated C string); this can never panic.
pub fn set_last_error(msg: impl Into<String>) {
    let mut s = msg.into();
    s.retain(|c| c != '\0');
    // CString::new only fails on interior NULs, which we just removed.
    let c = CString::new(s).unwrap_or_else(|_| CString::new("skinki-ffi: error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

/// Clear this thread's last-error slot (called at the start of every
/// `extern "C"` entry point so a stale message from a previous call is never
/// mistaken for the current one).
pub fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Run `f` against the current thread's last-error `CString`, or `None` if
/// no error has been recorded. Used by `ffi.rs` to hand out a raw pointer.
pub fn with_last_error<R>(f: impl FnOnce(Option<&CString>) -> R) -> R {
    LAST_ERROR.with(|slot| f(slot.borrow().as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        clear_last_error();
        with_last_error(|e| assert!(e.is_none()));
    }

    #[test]
    fn set_then_read() {
        set_last_error("boom");
        with_last_error(|e| assert_eq!(e.unwrap().to_str().unwrap(), "boom"));
    }

    #[test]
    fn strips_interior_nul() {
        set_last_error("a\0b");
        with_last_error(|e| assert_eq!(e.unwrap().to_str().unwrap(), "ab"));
    }

    #[test]
    fn clear_resets() {
        set_last_error("boom");
        clear_last_error();
        with_last_error(|e| assert!(e.is_none()));
    }
}
