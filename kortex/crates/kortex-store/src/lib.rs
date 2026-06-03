#![cfg_attr(not(unix), forbid(unsafe_code))]
//! Storage substrate: append-only L0 log + content-addressed unit store.
//!
//! Stage 2 of the Kortex engine. A pure-Rust, mmap-backed store for raw capture
//! events and derived memory units, with lossless provenance and deduplication.
//!
//! ## Encoding (manual, compact, lossless)
//!
//! Event record: created_utc_secs (i64 LE) | source (u8) | text bytes (UTF-8)
//!   text_len = outer_len - 9 (derived from length prefix, no cap)
//! Unit record:   event_id (u64 LE) | byte_start (u32 LE) | byte_end (u32 LE)

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type EventId = u64;
pub type UnitId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    Voice,
    Text,
    Import,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawEvent {
    pub source: Source,
    pub created_utc_secs: i64,
    pub text: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Unit {
    pub event: EventId,
    pub byte_start: u32,
    pub byte_end: u32,
}

// ---------------------------------------------------------------------------
// Compact encoding — lossless, no text_len in payload
// ---------------------------------------------------------------------------

/// event: created_utc_secs(i64 LE) | source(u8) | text bytes
pub fn encode_event(ev: &RawEvent) -> Vec<u8> {
    let text_bytes = ev.text.as_bytes();
    let total = 8 + 1 + text_bytes.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&ev.created_utc_secs.to_le_bytes());
    let src: u8 = match ev.source {
        Source::Voice => 0,
        Source::Text => 1,
        Source::Import => 2,
    };
    buf.push(src);
    buf.extend_from_slice(text_bytes);
    buf
}

pub fn decode_event(bytes: &[u8]) -> Option<RawEvent> {
    if bytes.len() < 9 {
        return None;
    }
    let created_utc_secs = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let source = match bytes[8] {
        0 => Source::Voice,
        1 => Source::Text,
        2 => Source::Import,
        _ => return None,
    };
    // text_len = bytes.len() - 9 (the caller owns the framing)
    let text = String::from_utf8(bytes[9..].to_vec()).ok()?;
    Some(RawEvent {
        source,
        created_utc_secs,
        text,
    })
}

/// unit: event(u64 LE) | byte_start(u32 LE) | byte_end(u32 LE)
pub fn encode_unit(u: &Unit) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&u.event.to_le_bytes());
    buf.extend_from_slice(&u.byte_start.to_le_bytes());
    buf.extend_from_slice(&u.byte_end.to_le_bytes());
    buf
}

pub fn decode_unit(bytes: &[u8]) -> Option<Unit> {
    if bytes.len() < 16 {
        return None;
    }
    let event = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let byte_start = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let byte_end = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    Some(Unit {
        event,
        byte_start,
        byte_end,
    })
}

// ---------------------------------------------------------------------------
// mmap
// ---------------------------------------------------------------------------

enum SegView {
    #[cfg(not(unix))]
    Ram(Vec<u8>),
    #[cfg(unix)]
    Mmap(MmapBytes),
}

impl SegView {
    #[cfg(not(unix))]
    #[allow(dead_code)]
    fn ram(_bytes: Vec<u8>) -> Self {
        SegView::Ram(_bytes)
    }

    #[cfg(unix)]
    fn mmap(path: &Path, _n: usize) -> anyhow::Result<Self> {
        Ok(SegView::Mmap(MmapBytes::open(path)?))
    }

    #[cfg(not(unix))]
    fn mmap(path: &Path, _n: usize) -> anyhow::Result<Self> {
        Ok(SegView::Ram(std::fs::read(path)?))
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(unix))]
            SegView::Ram(v) => v,
            #[cfg(unix)]
            SegView::Mmap(m) => m.as_slice(),
        }
    }
}

#[cfg(unix)]
struct MmapBytes {
    ptr: *mut libc::c_void,
    len: usize,
}

#[cfg(unix)]
impl MmapBytes {
    fn open(path: &Path) -> anyhow::Result<Self> {
        use std::os::unix::ffi::OsStrExt;
        let len = std::fs::metadata(path)?.len() as usize;
        if len == 0 {
            return Ok(MmapBytes {
                ptr: std::ptr::NonNull::<libc::c_void>::dangling().as_ptr(),
                len: 0,
            });
        }
        let mut cpath: Vec<u8> = path.as_os_str().as_bytes().to_vec();
        cpath.push(0);
        // SAFETY: standard open/mmap/close sequence with checked return codes.
        unsafe {
            let fd = libc::open(cpath.as_ptr() as *const libc::c_char, libc::O_RDONLY);
            if fd < 0 {
                bail!("open failed: {}", std::io::Error::last_os_error());
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
                bail!("mmap failed: {}", std::io::Error::last_os_error());
            }
            Ok(MmapBytes { ptr, len })
        }
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is a valid read-only mapping of exactly len bytes.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

#[cfg(unix)]
impl Drop for MmapBytes {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                libc::munmap(self.ptr, self.len);
            }
        }
    }
}

#[cfg(unix)]
unsafe impl Send for MmapBytes {}
#[cfg(unix)]
unsafe impl Sync for MmapBytes {}

// ---------------------------------------------------------------------------
// Content hash
// ---------------------------------------------------------------------------

fn content_hash_128(bytes: &[u8]) -> u128 {
    let hi = fnv1a_64(bytes, 0x012B_9B0A_BE15_D09D);
    let lo = fnv1a_64(bytes, 0xCBF2_9CE4_8422_2325);
    ((hi as u128) << 64) | (lo as u128)
}

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01B3;

fn fnv1a_64(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Sentence splitter
// ---------------------------------------------------------------------------

fn sentence_spans(text: &str) -> Vec<(u32, u32)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut start: u32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let is_terminal = b == b'.' || b == b'!' || b == b'?';
        let is_end = i + 1 == bytes.len();
        let next_is_boundary = is_end
            || bytes
                .get(i + 1)
                .map(|&c| c.is_ascii_whitespace())
                .unwrap_or(true);
        if is_terminal && next_is_boundary {
            let end = (i + 1) as u32;
            if end > start && std::str::from_utf8(&bytes[start as usize..end as usize]).is_ok() {
                spans.push((start, end));
            }
            start = end;
        }
    }
    if (start as usize) < bytes.len() {
        let end = bytes.len() as u32;
        if end > start && std::str::from_utf8(&bytes[start as usize..end as usize]).is_ok() {
            spans.push((start, end));
        }
    }
    spans
}

pub fn derive_units(event: EventId, text: &str) -> Vec<Unit> {
    sentence_spans(text)
        .into_iter()
        .map(|(byte_start, byte_end)| Unit {
            event,
            byte_start,
            byte_end,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

pub struct Store {
    dir: PathBuf,
    content_index: HashMap<u128, EventId>,
    event_seg: Vec<u8>,
    unit_seg: Vec<u8>,
    event_seg_num: u32,
    unit_seg_num: u32,
    event_views: Vec<(u32, SegView)>,
    unit_views: Vec<(u32, SegView)>,
    stored_raw_bytes: u64,
    stored_event_bytes: u64,
    stored_unit_bytes: u64,
}

impl Store {
    pub fn open(dir: &Path) -> anyhow::Result<Store> {
        std::fs::create_dir_all(dir).context("creating store directory")?;
        let (event_seg_num, unit_seg_num) = discover_max_seg(dir);
        let mut store = Store {
            dir: dir.to_path_buf(),
            content_index: HashMap::new(),
            event_seg: Vec::new(),
            unit_seg: Vec::new(),
            event_seg_num,
            unit_seg_num,
            event_views: Vec::new(),
            unit_views: Vec::new(),
            stored_raw_bytes: 0,
            stored_event_bytes: 0,
            stored_unit_bytes: 0,
        };
        store.load_existing()?;
        Ok(store)
    }

    pub fn append_event(&mut self, ev: &RawEvent) -> anyhow::Result<EventId> {
        let encoded = encode_event(ev);
        let hash = content_hash_128(&encoded);
        if let Some(&existing) = self.content_index.get(&hash) {
            return Ok(existing);
        }
        let offset = self.event_seg.len() as u64;
        let len = encoded.len() as u32;
        self.event_seg.extend_from_slice(&len.to_le_bytes());
        self.event_seg.extend_from_slice(&encoded);
        let id = (self.event_seg_num as u64) << 32 | offset;
        self.content_index.insert(hash, id);
        self.stored_raw_bytes += ev.text.len() as u64;
        self.stored_event_bytes += (4 + encoded.len()) as u64;
        Ok(id)
    }

    pub fn append_unit(&mut self, u: &Unit) -> anyhow::Result<UnitId> {
        let event_text = self
            .event_text(u.event)
            .with_context(|| format!("unit references unknown event {}", u.event))?;
        if u.byte_end > event_text.len() as u32 || u.byte_start >= u.byte_end {
            bail!(
                "unit span [{}, {}) out of range for event {} (len {})",
                u.byte_start,
                u.byte_end,
                u.event,
                event_text.len()
            );
        }
        let _span = &event_text[u.byte_start as usize..u.byte_end as usize];
        let encoded = encode_unit(u);
        let offset = self.unit_seg.len() as u64;
        let len = encoded.len() as u32;
        self.unit_seg.extend_from_slice(&len.to_le_bytes());
        self.unit_seg.extend_from_slice(&encoded);
        let id = (self.unit_seg_num as u64) << 32 | offset;
        self.stored_unit_bytes += (4 + encoded.len()) as u64;
        Ok(id)
    }

    /// Zero-copy read of the raw event text (FIX 7: no allocation).
    pub fn event_text(&self, id: EventId) -> anyhow::Result<&str> {
        let (seg_num, offset) = id_parts(id);
        let data = self.find_event_seg(seg_num)?;
        read_event_text_from_seg(data, offset as usize).context("event text read")
    }

    /// The exact source slice a unit points at.
    pub fn unit_text(&self, id: UnitId) -> anyhow::Result<&str> {
        let (seg_num, offset) = id_parts(id);
        let data = self.find_unit_seg(seg_num)?;
        let unit = read_unit_record(data, offset as usize).context("reading unit record")?;
        let event_text = self
            .event_text(unit.event)
            .context("unit references unknown event")?;
        Ok(&event_text[unit.byte_start as usize..unit.byte_end as usize])
    }

    pub fn event_count(&self) -> usize {
        self.content_index.len()
    }

    /// FIX 6: unit_count includes pending buffer, not only flushed segments.
    pub fn unit_count(&self) -> usize {
        let flushed: usize = self
            .unit_views
            .iter()
            .map(|(_, v)| count_records(v.as_slice(), 16))
            .sum();
        let pending: usize = count_records(&self.unit_seg, 16);
        flushed + pending
    }

    pub fn units(&self) -> impl Iterator<Item = (UnitId, Unit)> + '_ {
        let mut iter: Box<dyn Iterator<Item = (UnitId, Unit)>> = Box::new(std::iter::empty());
        for (seg_num, view) in &self.unit_views {
            let data = view.as_slice();
            let seg = *seg_num as u64;
            iter = Box::new(iter.chain(iter_units_seg(seg, data)));
        }
        let seg = self.unit_seg_num as u64;
        let pending: Vec<(UnitId, Unit)> = iter_units_seg(seg, &self.unit_seg).collect();
        iter = Box::new(iter.chain(pending));
        iter
    }

    pub fn sync(&mut self) -> anyhow::Result<()> {
        self.flush_events()?;
        self.flush_units()?;
        Ok(())
    }

    pub fn raw_text_bytes(&self) -> u64 {
        self.stored_raw_bytes
    }

    pub fn event_store_bytes(&self) -> u64 {
        self.stored_event_bytes
    }

    pub fn unit_store_bytes(&self) -> u64 {
        self.stored_unit_bytes
    }

    pub fn store_bytes(&self) -> u64 {
        self.stored_event_bytes + self.stored_unit_bytes
    }

    // -- internals --

    fn flush_events(&mut self) -> anyhow::Result<()> {
        if self.event_seg.is_empty() {
            return Ok(());
        }
        let path = seg_path(&self.dir, "events", self.event_seg_num);
        std::fs::write(&path, &self.event_seg)
            .with_context(|| format!("writing {}", path.display()))?;
        #[cfg(unix)]
        let view = SegView::mmap(&path, self.event_seg.len())?;
        #[cfg(not(unix))]
        let view = SegView::mmap(&path, self.event_seg.len())?;
        self.event_views.push((self.event_seg_num, view));
        self.event_seg.clear();
        self.event_seg_num += 1;
        Ok(())
    }

    fn flush_units(&mut self) -> anyhow::Result<()> {
        if self.unit_seg.is_empty() {
            return Ok(());
        }
        let path = seg_path(&self.dir, "units", self.unit_seg_num);
        std::fs::write(&path, &self.unit_seg)
            .with_context(|| format!("writing {}", path.display()))?;
        #[cfg(unix)]
        let view = SegView::mmap(&path, self.unit_seg.len())?;
        #[cfg(not(unix))]
        let view = SegView::mmap(&path, self.unit_seg.len())?;
        self.unit_views.push((self.unit_seg_num, view));
        self.unit_seg.clear();
        self.unit_seg_num += 1;
        Ok(())
    }

    fn load_existing(&mut self) -> anyhow::Result<()> {
        for seg_num in 0..self.event_seg_num {
            let path = seg_path(&self.dir, "events", seg_num);
            if !path.exists() {
                continue;
            }
            #[cfg(unix)]
            let view = SegView::mmap(&path, 0)?;
            #[cfg(not(unix))]
            let view = SegView::mmap(&path, 0)?;
            index_events_from_seg(&mut self.content_index, view.as_slice(), seg_num);
            self.event_views.push((seg_num, view));
        }
        for seg_num in 0..self.unit_seg_num {
            let path = seg_path(&self.dir, "units", seg_num);
            if !path.exists() {
                continue;
            }
            #[cfg(unix)]
            let view = SegView::mmap(&path, 0)?;
            #[cfg(not(unix))]
            let view = SegView::mmap(&path, 0)?;
            self.unit_views.push((seg_num, view));
        }
        for (_, v) in &self.event_views {
            self.stored_event_bytes += v.as_slice().len() as u64;
        }
        for (_, v) in &self.unit_views {
            self.stored_unit_bytes += v.as_slice().len() as u64;
        }
        Ok(())
    }

    fn find_event_seg(&self, seg_num: u32) -> anyhow::Result<&[u8]> {
        if seg_num == self.event_seg_num && !self.event_seg.is_empty() {
            return Ok(&self.event_seg);
        }
        for (sn, view) in &self.event_views {
            if *sn == seg_num {
                return Ok(view.as_slice());
            }
        }
        bail!("event segment {seg_num} not found")
    }

    fn find_unit_seg(&self, seg_num: u32) -> anyhow::Result<&[u8]> {
        if seg_num == self.unit_seg_num && !self.unit_seg.is_empty() {
            return Ok(&self.unit_seg);
        }
        for (sn, view) in &self.unit_views {
            if *sn == seg_num {
                return Ok(view.as_slice());
            }
        }
        bail!("unit segment {seg_num} not found")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn discover_max_seg(dir: &Path) -> (u32, u32) {
    let mut max_ev = 0u32;
    let mut max_un = 0u32;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name
                .strip_prefix("events-")
                .and_then(|s| s.strip_suffix(".seg"))
            {
                if let Ok(n) = rest.parse::<u32>() {
                    max_ev = max_ev.max(n + 1);
                }
            }
            if let Some(rest) = name
                .strip_prefix("units-")
                .and_then(|s| s.strip_suffix(".idx"))
            {
                if let Ok(n) = rest.parse::<u32>() {
                    max_un = max_un.max(n + 1);
                }
            }
        }
    }
    (max_ev, max_un)
}

fn seg_path(dir: &Path, prefix: &str, num: u32) -> PathBuf {
    let ext = match prefix {
        "events" => "seg",
        "units" => "idx",
        _ => "dat",
    };
    dir.join(format!("{prefix}-{num:03}.{ext}"))
}

fn id_parts(id: u64) -> (u32, u64) {
    ((id >> 32) as u32, id & 0xFFFF_FFFF)
}

/// FIX 7: single-pass event text extraction — no allocation of RawEvent.
fn read_event_text_from_seg(data: &[u8], offset: usize) -> Option<&str> {
    // outer len prefix
    if offset + 4 > data.len() {
        return None;
    }
    let rec_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    let rec_start = offset + 4;
    if rec_start + rec_len > data.len() {
        return None;
    }
    let rec = &data[rec_start..rec_start + rec_len];
    // header: 8 (i64) + 1 (source) = 9 bytes
    if rec.len() < 9 {
        return None;
    }
    let text_len = rec.len() - 9;
    std::str::from_utf8(&rec[9..9 + text_len]).ok()
}

fn read_unit_record(data: &[u8], offset: usize) -> Option<Unit> {
    if offset + 4 > data.len() {
        return None;
    }
    let len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    let rec_start = offset + 4;
    if rec_start + len > data.len() {
        return None;
    }
    decode_unit(&data[rec_start..rec_start + len])
}

fn count_records(data: &[u8], min_record_bytes: usize) -> usize {
    let mut count = 0;
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let rec_start = pos + 4;
        if rec_start + len > data.len() || len < min_record_bytes {
            break;
        }
        count += 1;
        pos = rec_start + len;
    }
    count
}

fn iter_units_seg(seg: u64, data: &[u8]) -> impl Iterator<Item = (UnitId, Unit)> + '_ {
    let mut pos = 0;
    std::iter::from_fn(move || {
        if pos + 4 > data.len() {
            return None;
        }
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let rec_start = pos + 4;
        if rec_start + len > data.len() {
            return None;
        }
        let unit = decode_unit(&data[rec_start..rec_start + len])?;
        let id = seg << 32 | pos as u64;
        pos = rec_start + len;
        Some((id, unit))
    })
}

fn index_events_from_seg(index: &mut HashMap<u128, EventId>, data: &[u8], seg_num: u32) {
    let seg = seg_num as u64;
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let rec_start = pos + 4;
        if rec_start + len > data.len() {
            break;
        }
        let id = seg << 32 | pos as u64;
        let rec = &data[rec_start..rec_start + len];
        let hash = content_hash_128(rec);
        index.entry(hash).or_insert(id);
        pos = rec_start + len;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FIX 2/3: lossless encode/decode ---

    #[test]
    fn event_encode_decode_roundtrip() {
        let ev = RawEvent {
            source: Source::Text,
            created_utc_secs: 1_700_000_123,
            text: "Hello, world! This is a test.".into(),
        };
        let encoded = encode_event(&ev);
        let decoded = decode_event(&encoded).unwrap();
        assert_eq!(decoded.source, Source::Text);
        assert_eq!(decoded.created_utc_secs, 1_700_000_123);
        assert_eq!(decoded.text, "Hello, world! This is a test.");
    }

    #[test]
    fn event_encode_decode_preserves_time_of_day() {
        // FIX 2: time-of-day must survive round-trip exactly.
        for ts in [0i64, 1_700_000_123, -86400, i64::MAX / 2] {
            let ev = RawEvent {
                source: Source::Text,
                created_utc_secs: ts,
                text: format!("ts={ts}"),
            };
            let encoded = encode_event(&ev);
            let decoded = decode_event(&encoded).unwrap();
            assert_eq!(decoded.created_utc_secs, ts, "time lost for {ts}");
        }
    }

    #[test]
    fn event_long_text_no_panic() {
        // FIX 3: text > 64 KB must not panic.
        let long = "A".repeat(200_000);
        let ev = RawEvent {
            source: Source::Voice,
            created_utc_secs: 0,
            text: long.clone(),
        };
        let encoded = encode_event(&ev);
        assert!(encoded.len() > 200_000);
        let decoded = decode_event(&encoded).unwrap();
        assert_eq!(decoded.text, long);
    }

    #[test]
    fn event_encode_decode_voice() {
        let ev = RawEvent {
            source: Source::Voice,
            created_utc_secs: 86400,
            text: "Voice note".into(),
        };
        let encoded = encode_event(&ev);
        let decoded = decode_event(&encoded).unwrap();
        assert_eq!(decoded.source, Source::Voice);
        assert_eq!(decoded.created_utc_secs, 86400);
        assert_eq!(decoded.text, "Voice note");
    }

    #[test]
    fn event_encode_decode_import() {
        let ev = RawEvent {
            source: Source::Import,
            created_utc_secs: 0,
            text: String::new(),
        };
        let encoded = encode_event(&ev);
        let decoded = decode_event(&encoded).unwrap();
        assert_eq!(decoded.source, Source::Import);
        assert_eq!(decoded.text, "");
    }

    #[test]
    fn unit_encode_decode_roundtrip() {
        let u = Unit {
            event: 0xABCD_0000_0001,
            byte_start: 10,
            byte_end: 42,
        };
        let encoded = encode_unit(&u);
        let decoded = decode_unit(&encoded).unwrap();
        assert_eq!(decoded.event, 0xABCD_0000_0001);
        assert_eq!(decoded.byte_start, 10);
        assert_eq!(decoded.byte_end, 42);
    }

    #[test]
    fn decode_event_truncated() {
        assert!(decode_event(&[0; 1]).is_none());
        assert!(decode_event(&[0; 8]).is_none());
    }

    #[test]
    fn decode_unit_truncated() {
        assert!(decode_unit(&[0; 1]).is_none());
        assert!(decode_unit(&[0; 15]).is_none());
    }

    #[test]
    fn event_encoding_size() {
        let ev = RawEvent {
            source: Source::Text,
            created_utc_secs: 1_700_000_000,
            text: "Hi".into(),
        };
        let encoded = encode_event(&ev);
        // 8 (i64) + 1 (source) + 2 (text) = 11
        assert_eq!(encoded.len(), 11);
    }

    #[test]
    fn unit_encoding_is_fixed() {
        let encoded = encode_unit(&Unit {
            event: 0,
            byte_start: 0,
            byte_end: 0,
        });
        assert_eq!(encoded.len(), 16);
    }

    // --- T2: append-only log ---

    #[test]
    fn append_and_read_event() {
        let dir = temp_dir("append_read_event");
        let mut store = Store::open(&dir).unwrap();
        let ev = RawEvent {
            source: Source::Text,
            created_utc_secs: 1_700_000_000,
            text: "Hello, world!".into(),
        };
        let id = store.append_event(&ev).unwrap();
        store.sync().unwrap();
        assert_eq!(store.event_text(id).unwrap(), "Hello, world!");
    }

    #[test]
    fn append_and_read_unit() {
        let dir = temp_dir("append_read_unit");
        let mut store = Store::open(&dir).unwrap();
        let ev = RawEvent {
            source: Source::Text,
            created_utc_secs: 1_700_000_000,
            text: "Hello, world!".into(),
        };
        let event_id = store.append_event(&ev).unwrap();
        store.sync().unwrap();
        let u = Unit {
            event: event_id,
            byte_start: 0,
            byte_end: 5,
        };
        let unit_id = store.append_unit(&u).unwrap();
        store.sync().unwrap();
        assert_eq!(store.unit_text(unit_id).unwrap(), "Hello");
    }

    #[test]
    fn reopen_keeps_ids() {
        let dir = temp_dir("reopen_ids");
        let event_id;
        {
            let mut store = Store::open(&dir).unwrap();
            let ev = RawEvent {
                source: Source::Text,
                created_utc_secs: 1_700_000_000,
                text: "Persistent!".into(),
            };
            event_id = store.append_event(&ev).unwrap();
            store.sync().unwrap();
        }
        {
            let store = Store::open(&dir).unwrap();
            assert_eq!(store.event_text(event_id).unwrap(), "Persistent!");
        }
    }

    #[test]
    fn reopen_and_continue_appending() {
        let dir = temp_dir("reopen_continue");
        let id1;
        {
            let mut store = Store::open(&dir).unwrap();
            id1 = store
                .append_event(&RawEvent {
                    source: Source::Text,
                    created_utc_secs: 86400,
                    text: "First".into(),
                })
                .unwrap();
            store.sync().unwrap();
        }
        let id2;
        {
            let mut store = Store::open(&dir).unwrap();
            assert_eq!(store.event_text(id1).unwrap(), "First");
            id2 = store
                .append_event(&RawEvent {
                    source: Source::Text,
                    created_utc_secs: 2 * 86400,
                    text: "Second".into(),
                })
                .unwrap();
            store.sync().unwrap();
        }
        {
            let store = Store::open(&dir).unwrap();
            assert_eq!(store.event_text(id1).unwrap(), "First");
            assert_eq!(store.event_text(id2).unwrap(), "Second");
        }
    }

    #[test]
    fn event_count_tracks() {
        let dir = temp_dir("event_count");
        let mut store = Store::open(&dir).unwrap();
        assert_eq!(store.event_count(), 0);
        store
            .append_event(&RawEvent {
                source: Source::Text,
                created_utc_secs: 86400,
                text: "A".into(),
            })
            .unwrap();
        store
            .append_event(&RawEvent {
                source: Source::Text,
                created_utc_secs: 2 * 86400,
                text: "B".into(),
            })
            .unwrap();
        assert_eq!(store.event_count(), 2);
    }

    // --- T3: dedup ---

    #[test]
    fn dedup_same_event_returns_same_id() {
        let dir = temp_dir("dedup");
        let mut store = Store::open(&dir).unwrap();
        let ev = RawEvent {
            source: Source::Text,
            created_utc_secs: 1_700_000_000,
            text: "Duplicate me".into(),
        };
        let id1 = store.append_event(&ev).unwrap();
        let id2 = store.append_event(&ev).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.event_count(), 1);
    }

    #[test]
    fn hash_different_events_different() {
        let h1 = content_hash_128(b"hello");
        let h2 = content_hash_128(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_empty() {
        let h = content_hash_128(b"");
        assert_ne!(h, 0);
    }

    // --- T4: derive_units ---

    #[test]
    fn sentence_split_simple() {
        let units = derive_units(0, "Hello. World!");
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn sentence_split_multi_boundaries() {
        let text = "One. Two! Three? Four.";
        let units = derive_units(0, text);
        assert_eq!(units.len(), 4);
    }

    #[test]
    fn sentence_split_no_trailing_boundary() {
        let text = "No punctuation here";
        let units = derive_units(0, text);
        assert_eq!(units.len(), 1);
        assert_eq!(
            &text[units[0].byte_start as usize..units[0].byte_end as usize],
            text
        );
    }

    #[test]
    fn sentence_split_empty_text() {
        assert_eq!(derive_units(0, "").len(), 0);
    }

    #[test]
    fn sentence_split_preserves_spans() {
        let text = "Hello. World!";
        let units = derive_units(42, text);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].event, 42);
        assert_eq!(
            &text[units[0].byte_start as usize..units[0].byte_end as usize],
            "Hello."
        );
        assert_eq!(units[1].event, 42);
        assert_eq!(
            &text[units[1].byte_start as usize..units[1].byte_end as usize],
            " World!"
        );
    }

    #[test]
    fn sentence_split_unicode() {
        let text = "Привет. Мир!";
        let units = derive_units(0, text);
        assert_eq!(units.len(), 2);
        assert_eq!(
            &text[units[0].byte_start as usize..units[0].byte_end as usize],
            "Привет."
        );
    }

    // --- T5: provenance ---

    #[test]
    fn provenance_roundtrip_100_percent() {
        let dir = temp_dir("provenance");
        let mut store = Store::open(&dir).unwrap();
        let text = "First sentence. Second one! Third? End.";
        let event_id = store
            .append_event(&RawEvent {
                source: Source::Text,
                created_utc_secs: 86400,
                text: text.into(),
            })
            .unwrap();
        store.sync().unwrap();
        let units = derive_units(event_id, text);
        for u in &units {
            let unit_id = store.append_unit(u).unwrap();
            store.sync().unwrap();
            let stored_text = store.unit_text(unit_id).unwrap();
            let expected = &text[u.byte_start as usize..u.byte_end as usize];
            assert_eq!(
                stored_text, expected,
                "provenance mismatch for unit {unit_id}"
            );
        }
    }

    #[test]
    fn unit_referencing_unknown_event_fails() {
        let dir = temp_dir("bad_unit_ref");
        let mut store = Store::open(&dir).unwrap();
        let u = Unit {
            event: 999_999,
            byte_start: 0,
            byte_end: 1,
        };
        assert!(store.append_unit(&u).is_err());
    }

    #[test]
    fn unit_bad_span_fails() {
        let dir = temp_dir("bad_unit_span");
        let mut store = Store::open(&dir).unwrap();
        let event_id = store
            .append_event(&RawEvent {
                source: Source::Text,
                created_utc_secs: 86400,
                text: "Hi".into(),
            })
            .unwrap();
        store.sync().unwrap();
        assert!(store
            .append_unit(&Unit {
                event: event_id,
                byte_start: 0,
                byte_end: 100,
            })
            .is_err());
        assert!(store
            .append_unit(&Unit {
                event: event_id,
                byte_start: 2,
                byte_end: 1,
            })
            .is_err());
    }

    // --- T6: unit iter ---

    #[test]
    fn unit_iter_produces_all_units() {
        let dir = temp_dir("unit_iter");
        let mut store = Store::open(&dir).unwrap();
        let text = "A. B. C.";
        let event_id = store
            .append_event(&RawEvent {
                source: Source::Text,
                created_utc_secs: 86400,
                text: text.into(),
            })
            .unwrap();
        store.sync().unwrap();
        let derived = derive_units(event_id, text);
        for u in &derived {
            store.append_unit(u).unwrap();
        }
        store.sync().unwrap();
        let collected: Vec<_> = store.units().collect();
        assert_eq!(collected.len(), derived.len());
    }

    /// FIX 6: unit_count must match units().count() even before sync.
    #[test]
    fn unit_count_matches_units_even_before_sync() {
        let dir = temp_dir("unit_count_pending");
        let mut store = Store::open(&dir).unwrap();
        let event_id = store
            .append_event(&RawEvent {
                source: Source::Text,
                created_utc_secs: 86400,
                text: "A. B.".into(),
            })
            .unwrap();
        store.sync().unwrap();
        let derived = derive_units(event_id, "A. B.");
        for u in &derived {
            store.append_unit(u).unwrap();
        }
        // No sync! Still must agree.
        assert_eq!(
            store.unit_count(),
            store.units().count(),
            "unit_count must include pending buffer"
        );
    }

    // --- Golden: overhead ---

    #[test]
    fn storage_overhead_within_budget() {
        let dir = temp_dir("overhead");
        let mut store = Store::open(&dir).unwrap();
        let texts: Vec<String> = (0..100)
            .map(|i| format!(
                "Entry number {}. This is a longer sentence with more text to make the overhead realistic. \
                 The fixed record header amortizes well over typical journal entries of this length.",
                 i))
            .collect();
        let mut total_raw = 0u64;
        for t in &texts {
            total_raw += t.len() as u64;
            store
                .append_event(&RawEvent {
                    source: Source::Text,
                    created_utc_secs: 1_700_000_000 + (total_raw as i64),
                    text: t.clone(),
                })
                .unwrap();
        }
        store.sync().unwrap();

        let content_overhead =
            store.event_store_bytes() as f64 / store.raw_text_bytes().max(1) as f64;
        assert!(
            content_overhead <= 1.25,
            "content overhead {:.3}x exceeds budget 1.25x",
            content_overhead
        );
    }

    // --- FIX 4: byte-level determinism ---

    #[test]
    fn deterministic_store_files_byte_identical() {
        fn store_bytes(dir_name: &str) -> Vec<(String, Vec<u8>)> {
            let dir = temp_dir(dir_name);
            let mut store = Store::open(&dir).unwrap();
            let texts = ["Alpha. Beta.", "Gamma. Delta.", "Epsilon."];
            for t in &texts {
                store
                    .append_event(&RawEvent {
                        source: Source::Text,
                        created_utc_secs: 1_700_000_000,
                        text: t.to_string(),
                    })
                    .unwrap();
            }
            store.sync().unwrap();
            let mut files = Vec::new();
            for entry in std::fs::read_dir(&dir).unwrap() {
                let e = entry.unwrap();
                let path = e.path();
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                if name.ends_with(".seg") || name.ends_with(".idx") {
                    files.push((name, std::fs::read(&path).unwrap()));
                }
            }
            files.sort_by(|a, b| a.0.cmp(&b.0));
            files
        }
        let a = store_bytes("det_byte_a");
        let b = store_bytes("det_byte_b");
        assert_eq!(a, b, "segment file bytes must be identical");
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kortex_store_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
