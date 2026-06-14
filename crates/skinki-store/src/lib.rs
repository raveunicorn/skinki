#![cfg_attr(not(unix), forbid(unsafe_code))]
//! Storage substrate: append-only L0 log + content-addressed unit store.
//!
//! Stage 2/2B of the skinki engine. A pure-Rust, mmap-backed store for raw
//! capture events and derived memory units, with lossless provenance,
//! deduplication, and crash-safe durability.
//!
//! ## Encoding (manual, compact, lossless)
//!
//! Event record: created_utc_secs (i64 LE) | source (u8) | text bytes (UTF-8)
//!   text_len = outer_len - 9 (derived from framing, no cap)
//! Unit record:   event_id (u64 LE) | byte_start (u32 LE) | byte_end (u32 LE)
//!
//! ## Durability model (Stage 2B)
//!
//! - **Write-through append.** Records are written to the current segment file
//!   as they arrive (buffered); `sync()` flushes and fsyncs. The window of loss
//!   after a crash is exactly "appends since the last `sync()`".
//! - **Size-based segment rotation** (default 64 MiB) instead of a segment per
//!   sync: `open()` never validates more than one segment tail, and the
//!   in-RAM tail mirror stays bounded.
//! - **Torn-tail recovery.** On `open()`, the last segment of each stream is
//!   framing-validated; a torn final record (crash mid-write) is physically
//!   truncated away. Committed bytes are never rewritten.
//! - **Persistent dedup index.** The content-hash index is persisted as sorted
//!   runs (`dedup-NNNN.run`, written at event-segment rotation, compacted when
//!   there are too many). Lookups binary-search the mmap'd runs; only the
//!   since-last-rotation delta lives in a RAM map. `open()` therefore scans at
//!   most one segment of events instead of the whole history — this is what
//!   keeps cold start fast and idle RAM flat as the log grows to years.

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
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
    #[cfg(unix)]
    fn open(path: &Path) -> anyhow::Result<Self> {
        Ok(SegView::Mmap(MmapBytes::open(path)?))
    }

    #[cfg(not(unix))]
    fn open(path: &Path) -> anyhow::Result<Self> {
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
// Options & constants
// ---------------------------------------------------------------------------

/// Default segment rotation threshold. Chosen so `open()` validates at most
/// this much tail data and the per-stream RAM mirror stays bounded.
pub const DEFAULT_SEGMENT_TARGET_BYTES: u64 = 64 * 1024 * 1024;

/// Compact dedup runs once more than this many accumulate.
const MAX_DEDUP_RUNS: usize = 8;

const DEDUP_RUN_MAGIC: &[u8; 8] = b"KXDDRUN1";
const DEDUP_RUN_HEADER: usize = 8 + 8 + 8; // magic | count u64 | covered_max u64
const DEDUP_RUN_ENTRY: usize = 16 + 8; // hash u128 | event id u64

const COUNTS_META_FILE: &str = "counts.meta";

const MIN_EVENT_RECORD: usize = 9;
const MIN_UNIT_RECORD: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct StoreOptions {
    /// Rotate a segment once it reaches this many bytes.
    pub segment_target_bytes: u64,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions {
            segment_target_bytes: DEFAULT_SEGMENT_TARGET_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// AppendStream — one segmented, write-through, crash-recovering record log
// ---------------------------------------------------------------------------

/// A segmented append-only record log. Records are length-prefixed
/// (`len: u32 LE` + payload). Ids are `(segment << 32) | byte_offset`, stable
/// across reopen. Finished segments are mmap'd read-only; the current segment
/// is served from an mmap of its validated-at-open prefix plus an in-RAM
/// mirror of bytes appended since open (records never straddle the boundary).
struct AppendStream {
    dir: PathBuf,
    prefix: &'static str,
    ext: &'static str,
    target_bytes: u64,
    seg_num: u32,
    writer: std::io::BufWriter<std::fs::File>,
    base: SegView,
    base_len: u64,
    tail: Vec<u8>,
    /// Finished segments, ascending by segment number.
    finished: Vec<(u32, SegView)>,
    total_bytes: u64,
    /// Torn bytes discarded from the last segment at open (crash recovery).
    recovered_truncated_bytes: u64,
}

/// Length of the valid framing prefix of `data`: the scan stops at the first
/// record that is incomplete (torn write) or implausibly small.
fn validated_prefix_len(data: &[u8], min_record_bytes: usize) -> usize {
    let mut pos = 0usize;
    while pos + 4 <= data.len() {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        if len < min_record_bytes {
            break;
        }
        let Some(end) = (pos + 4).checked_add(len) else {
            break;
        };
        if end > data.len() {
            break;
        }
        pos = end;
    }
    pos
}

fn seg_file_path(dir: &Path, prefix: &str, num: u32, ext: &str) -> PathBuf {
    dir.join(format!("{prefix}-{num:03}.{ext}"))
}

/// Best-effort directory fsync so renames/creates survive power loss.
fn fsync_dir(dir: &Path) {
    if let Ok(f) = std::fs::File::open(dir) {
        let _ = f.sync_all();
    }
}

impl AppendStream {
    fn open(
        dir: &Path,
        prefix: &'static str,
        ext: &'static str,
        min_record_bytes: usize,
        target_bytes: u64,
    ) -> anyhow::Result<AppendStream> {
        // Discover existing segments, ascending.
        let mut nums: Vec<u32> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(rest) = name
                    .strip_prefix(&format!("{prefix}-"))
                    .and_then(|s| s.strip_suffix(&format!(".{ext}")))
                {
                    if let Ok(n) = rest.parse::<u32>() {
                        nums.push(n);
                    }
                }
            }
        }
        nums.sort_unstable();

        let mut finished = Vec::with_capacity(nums.len().saturating_sub(1));
        let mut total_bytes = 0u64;
        let (seg_num, base, base_len, recovered) = if let Some(&last) = nums.last() {
            for &n in &nums[..nums.len() - 1] {
                let view = SegView::open(&seg_file_path(dir, prefix, n, ext))?;
                total_bytes += view.as_slice().len() as u64;
                finished.push((n, view));
            }
            let path = seg_file_path(dir, prefix, last, ext);
            let view = SegView::open(&path)?;
            let raw_len = view.as_slice().len();
            let valid = validated_prefix_len(view.as_slice(), min_record_bytes);
            let recovered = (raw_len - valid) as u64;
            let view = if recovered > 0 {
                // Crash recovery: physically truncate the torn tail. Committed
                // records are untouched; only garbage past the last complete
                // record is discarded.
                drop(view);
                let f = std::fs::OpenOptions::new().write(true).open(&path)?;
                f.set_len(valid as u64)?;
                f.sync_all()?;
                SegView::open(&path)?
            } else {
                view
            };
            total_bytes += valid as u64;
            (last, view, valid as u64, recovered)
        } else {
            let path = seg_file_path(dir, prefix, 0, ext);
            std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
            fsync_dir(dir);
            (0, SegView::open(&path)?, 0u64, 0u64)
        };

        let path = seg_file_path(dir, prefix, seg_num, ext);
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {} for append", path.display()))?;
        Ok(AppendStream {
            dir: dir.to_path_buf(),
            prefix,
            ext,
            target_bytes,
            seg_num,
            writer: std::io::BufWriter::with_capacity(1 << 16, file),
            base,
            base_len,
            tail: Vec::new(),
            finished,
            total_bytes,
            recovered_truncated_bytes: recovered,
        })
    }

    fn current_size(&self) -> u64 {
        self.base_len + self.tail.len() as u64
    }

    /// Append one record. Returns `(id, rotated)`; `rotated` is true when a
    /// segment was finished right before this record (the caller may want to
    /// persist per-rotation metadata — dedup runs, counters).
    fn append(&mut self, payload: &[u8]) -> anyhow::Result<(u64, bool)> {
        let rec_len = 4 + payload.len() as u64;
        let mut rotated = false;
        if self.current_size() > 0 && self.current_size() + rec_len > self.target_bytes {
            self.rotate()?;
            rotated = true;
        }
        let offset = self.current_size();
        anyhow::ensure!(
            offset + rec_len <= u32::MAX as u64,
            "record would overflow the 4 GiB segment offset space"
        );
        let len = payload.len() as u32;
        self.tail.extend_from_slice(&len.to_le_bytes());
        self.tail.extend_from_slice(payload);
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(payload)?;
        self.total_bytes += rec_len;
        Ok((((self.seg_num as u64) << 32) | offset, rotated))
    }

    /// Make everything appended so far durable (flush + fsync).
    fn sync(&mut self) -> anyhow::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Finish the current segment (flush, fsync, re-mmap read-only) and start
    /// the next one.
    fn rotate(&mut self) -> anyhow::Result<()> {
        self.sync()?;
        let old_path = seg_file_path(&self.dir, self.prefix, self.seg_num, self.ext);
        let view = SegView::open(&old_path)?;
        self.finished.push((self.seg_num, view));
        self.seg_num += 1;
        let new_path = seg_file_path(&self.dir, self.prefix, self.seg_num, self.ext);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_path)
            .with_context(|| format!("creating {}", new_path.display()))?;
        self.writer = std::io::BufWriter::with_capacity(1 << 16, file);
        self.base = SegView::open(&new_path)?;
        self.base_len = 0;
        self.tail.clear();
        fsync_dir(&self.dir);
        Ok(())
    }

    /// Slice from a record's length prefix to the end of its storage region.
    /// Records never straddle region boundaries, so the whole record is
    /// contained in the returned slice.
    fn slice_at(&self, id: u64) -> Option<&[u8]> {
        let seg = (id >> 32) as u32;
        let off = (id & 0xFFFF_FFFF) as usize;
        if seg == self.seg_num {
            if (off as u64) < self.base_len {
                return Some(&self.base.as_slice()[off..]);
            }
            let rel = off - self.base_len as usize;
            if rel < self.tail.len() {
                return Some(&self.tail[rel..]);
            }
            return None;
        }
        let idx = self.finished.binary_search_by_key(&seg, |(n, _)| *n).ok()?;
        let data = self.finished[idx].1.as_slice();
        if off < data.len() {
            Some(&data[off..])
        } else {
            None
        }
    }

    /// Storage regions in id order: finished segments, then the current
    /// segment's validated base, then its in-RAM tail. Each region is
    /// `(first_record_id_base, bytes)`.
    fn regions(&self) -> Vec<(u64, &[u8])> {
        let mut out: Vec<(u64, &[u8])> = Vec::with_capacity(self.finished.len() + 2);
        for (n, view) in &self.finished {
            out.push(((*n as u64) << 32, view.as_slice()));
        }
        let cur = (self.seg_num as u64) << 32;
        out.push((cur, &self.base.as_slice()[..self.base_len as usize]));
        out.push((cur | self.base_len, &self.tail));
        out
    }

    /// Visit `(id, payload)` of every record with `id > floor` (all records
    /// when `floor` is None). Regions fully covered by `floor` are skipped
    /// without being walked — this is what makes reopen O(one segment).
    fn for_each_record(&self, floor: Option<u64>, mut f: impl FnMut(u64, &[u8])) {
        for (base_id, data) in self.regions() {
            if let Some(fl) = floor {
                // All ids in this region are < base_id + data.len().
                if data.is_empty() || base_id + data.len() as u64 - 1 <= fl {
                    continue;
                }
            }
            let mut pos = 0usize;
            while pos + 4 <= data.len() {
                let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                let start = pos + 4;
                if start + len > data.len() {
                    break;
                }
                let id = base_id + pos as u64;
                if floor.is_none_or(|fl| id > fl) {
                    f(id, &data[start..start + len]);
                }
                pos = start + len;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dedup runs — the persistent content-hash index
// ---------------------------------------------------------------------------

/// One immutable sorted run of `(content_hash, event_id)` entries, mmap'd.
/// `covered_max` is the largest event id whose hash is guaranteed to be in
/// the runs (collectively): everything newer is rebuilt into the RAM delta at
/// open by scanning only the events past that watermark.
struct DedupRun {
    view: SegView,
    count: usize,
    covered_max: u64,
}

impl DedupRun {
    fn open(path: &Path) -> Option<DedupRun> {
        let view = SegView::open(path).ok()?;
        let data = view.as_slice();
        if data.len() < DEDUP_RUN_HEADER || &data[0..8] != DEDUP_RUN_MAGIC {
            return None;
        }
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let covered_max = u64::from_le_bytes(data[16..24].try_into().unwrap());
        if data.len() != DEDUP_RUN_HEADER + count * DEDUP_RUN_ENTRY {
            return None;
        }
        Some(DedupRun {
            view,
            count,
            covered_max,
        })
    }

    fn entry(&self, i: usize) -> (u128, EventId) {
        let off = DEDUP_RUN_HEADER + i * DEDUP_RUN_ENTRY;
        let data = self.view.as_slice();
        let hash = u128::from_le_bytes(data[off..off + 16].try_into().unwrap());
        let id = u64::from_le_bytes(data[off + 16..off + 24].try_into().unwrap());
        (hash, id)
    }

    fn lookup(&self, hash: u128) -> Option<EventId> {
        let (mut lo, mut hi) = (0usize, self.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (h, id) = self.entry(mid);
            match h.cmp(&hash) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(id),
            }
        }
        None
    }
}

fn dedup_run_path(dir: &Path, num: u32) -> PathBuf {
    dir.join(format!("dedup-{num:04}.run"))
}

/// Write a sorted run atomically (tmp + fsync + rename + dir fsync).
fn write_dedup_run(
    dir: &Path,
    num: u32,
    entries: &[(u128, EventId)],
    covered_max: u64,
) -> anyhow::Result<DedupRun> {
    let mut buf = Vec::with_capacity(DEDUP_RUN_HEADER + entries.len() * DEDUP_RUN_ENTRY);
    buf.extend_from_slice(DEDUP_RUN_MAGIC);
    buf.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    buf.extend_from_slice(&covered_max.to_le_bytes());
    for (hash, id) in entries {
        buf.extend_from_slice(&hash.to_le_bytes());
        buf.extend_from_slice(&id.to_le_bytes());
    }
    let path = dedup_run_path(dir, num);
    let tmp = path.with_extension("run.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    fsync_dir(dir);
    DedupRun::open(&path).context("re-opening freshly written dedup run")
}

// ---------------------------------------------------------------------------
// Unit-count metadata (so reopen doesn't walk the whole unit store)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Default)]
struct CountsMeta {
    version: u32,
    units_counted: u64,
    /// All unit records with id <= this watermark are included in
    /// `units_counted`; None means nothing is counted (full scan on open).
    units_high_water: Option<u64>,
}

fn read_counts_meta(dir: &Path) -> CountsMeta {
    let path = dir.join(COUNTS_META_FILE);
    std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<CountsMeta>(&bytes).ok())
        .filter(|m| m.version == 1)
        .unwrap_or_default()
}

fn write_counts_meta(dir: &Path, meta: &CountsMeta) -> anyhow::Result<()> {
    let path = dir.join(COUNTS_META_FILE);
    let tmp = path.with_extension("meta.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&serde_json::to_vec(meta)?)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    fsync_dir(dir);
    Ok(())
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

pub struct Store {
    dir: PathBuf,
    events: AppendStream,
    units: AppendStream,
    runs: Vec<(u32, DedupRun)>,
    next_run_num: u32,
    /// Unique-event hashes appended since the last persisted run.
    delta: HashMap<u128, EventId>,
    event_unique: usize,
    unit_count: usize,
    /// Raw text bytes appended *this session* (bench accounting; not
    /// recomputed on reopen — recomputing would require decoding the full
    /// history, defeating fast open).
    stored_raw_bytes: u64,
}

impl Store {
    pub fn open(dir: &Path) -> anyhow::Result<Store> {
        Store::open_with(dir, StoreOptions::default())
    }

    pub fn open_with(dir: &Path, opts: StoreOptions) -> anyhow::Result<Store> {
        std::fs::create_dir_all(dir).context("creating store directory")?;
        let events = AppendStream::open(
            dir,
            "events",
            "seg",
            MIN_EVENT_RECORD,
            opts.segment_target_bytes,
        )?;
        let units = AppendStream::open(
            dir,
            "units",
            "idx",
            MIN_UNIT_RECORD,
            opts.segment_target_bytes,
        )?;

        // Load dedup runs; any unreadable/corrupt run invalidates them all and
        // falls back to a full rebuild scan (correct, just slower).
        let mut run_nums: Vec<u32> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(rest) = name
                    .strip_prefix("dedup-")
                    .and_then(|s| s.strip_suffix(".run"))
                {
                    if let Ok(n) = rest.parse::<u32>() {
                        run_nums.push(n);
                    }
                }
            }
        }
        run_nums.sort_unstable();
        let next_run_num = run_nums.last().map_or(0, |n| n + 1);
        let mut runs: Vec<(u32, DedupRun)> = Vec::with_capacity(run_nums.len());
        let mut runs_ok = true;
        for n in &run_nums {
            match DedupRun::open(&dedup_run_path(dir, *n)) {
                Some(r) => runs.push((*n, r)),
                None => {
                    runs_ok = false;
                    break;
                }
            }
        }
        if !runs_ok {
            runs.clear();
        }
        let covered = runs.iter().map(|(_, r)| r.covered_max).max();

        // Rebuild the RAM delta from events newer than the run watermark.
        // Segments contain only unique events (duplicates are never appended),
        // so this cannot double-insert a hash that lives in a run.
        let mut delta: HashMap<u128, EventId> = HashMap::new();
        events.for_each_record(covered, |id, payload| {
            delta.entry(content_hash_128(payload)).or_insert(id);
        });
        let event_unique = runs.iter().map(|(_, r)| r.count).sum::<usize>() + delta.len();

        // Unit count: persisted watermark + a scan of only what's newer.
        let meta = read_counts_meta(dir);
        let mut unit_count = meta.units_counted as usize;
        units.for_each_record(meta.units_high_water, |_, _| unit_count += 1);

        Ok(Store {
            dir: dir.to_path_buf(),
            events,
            units,
            runs,
            next_run_num,
            delta,
            event_unique,
            unit_count,
            stored_raw_bytes: 0,
        })
    }

    pub fn append_event(&mut self, ev: &RawEvent) -> anyhow::Result<EventId> {
        let encoded = encode_event(ev);
        let hash = content_hash_128(&encoded);
        if let Some(&existing) = self.delta.get(&hash) {
            return Ok(existing);
        }
        for (_, run) in &self.runs {
            if let Some(existing) = run.lookup(hash) {
                return Ok(existing);
            }
        }
        let (id, rotated) = self.events.append(&encoded)?;
        if rotated {
            // The delta's ids all precede the new segment; persist them so
            // reopen only ever scans the current segment.
            self.persist_dedup_run()?;
        }
        self.delta.insert(hash, id);
        self.event_unique += 1;
        self.stored_raw_bytes += ev.text.len() as u64;
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
        let encoded = encode_unit(u);
        let (id, rotated) = self.units.append(&encoded)?;
        if rotated {
            // unit_count currently equals exactly the records in finished
            // segments (the new record isn't counted yet) — the watermark
            // covers everything below the fresh segment.
            write_counts_meta(
                &self.dir,
                &CountsMeta {
                    version: 1,
                    units_counted: self.unit_count as u64,
                    units_high_water: Some(((self.units.seg_num as u64) << 32) - 1),
                },
            )?;
        }
        self.unit_count += 1;
        Ok(id)
    }

    fn persist_dedup_run(&mut self) -> anyhow::Result<()> {
        if self.delta.is_empty() {
            return Ok(());
        }
        // Everything in segments below the (already-incremented) current one.
        let covered_max = ((self.events.seg_num as u64) << 32) - 1;
        let mut entries: Vec<(u128, EventId)> = self.delta.drain().collect();
        entries.sort_unstable_by_key(|e| e.0);
        let run = write_dedup_run(&self.dir, self.next_run_num, &entries, covered_max)?;
        self.runs.push((self.next_run_num, run));
        self.next_run_num += 1;
        if self.runs.len() > MAX_DEDUP_RUNS {
            self.compact_runs()?;
        }
        Ok(())
    }

    fn compact_runs(&mut self) -> anyhow::Result<()> {
        let mut all: Vec<(u128, EventId)> = Vec::new();
        let mut covered_max = 0u64;
        for (_, run) in &self.runs {
            covered_max = covered_max.max(run.covered_max);
            for i in 0..run.count {
                all.push(run.entry(i));
            }
        }
        all.sort_unstable_by_key(|e| e.0);
        let merged = write_dedup_run(&self.dir, self.next_run_num, &all, covered_max)?;
        let old: Vec<u32> = self.runs.iter().map(|(n, _)| *n).collect();
        self.runs = vec![(self.next_run_num, merged)];
        self.next_run_num += 1;
        for n in old {
            let _ = std::fs::remove_file(dedup_run_path(&self.dir, n));
        }
        fsync_dir(&self.dir);
        Ok(())
    }

    /// Zero-copy read of the raw event text.
    pub fn event_text(&self, id: EventId) -> anyhow::Result<&str> {
        let data = self
            .events
            .slice_at(id)
            .with_context(|| format!("event {id} not found"))?;
        read_event_text_at(data).context("event text read")
    }

    /// The exact source slice a unit points at.
    pub fn unit_text(&self, id: UnitId) -> anyhow::Result<&str> {
        let data = self
            .units
            .slice_at(id)
            .with_context(|| format!("unit {id} not found"))?;
        let unit = read_record_at(data)
            .and_then(decode_unit)
            .context("reading unit record")?;
        let event_text = self
            .event_text(unit.event)
            .context("unit references unknown event")?;
        Ok(&event_text[unit.byte_start as usize..unit.byte_end as usize])
    }

    pub fn event_count(&self) -> usize {
        self.event_unique
    }

    pub fn unit_count(&self) -> usize {
        self.unit_count
    }

    pub fn units(&self) -> impl Iterator<Item = (UnitId, Unit)> + '_ {
        self.units
            .regions()
            .into_iter()
            .flat_map(|(base_id, data)| iter_units_region(base_id, data))
    }

    /// Flush and fsync both streams. After this returns, everything appended
    /// so far survives a crash or power loss.
    pub fn sync(&mut self) -> anyhow::Result<()> {
        self.events.sync()?;
        self.units.sync()?;
        Ok(())
    }

    /// Torn bytes discarded during crash recovery at open (0 = clean open).
    pub fn recovered_truncated_bytes(&self) -> u64 {
        self.events.recovered_truncated_bytes + self.units.recovered_truncated_bytes
    }

    pub fn raw_text_bytes(&self) -> u64 {
        self.stored_raw_bytes
    }

    pub fn event_store_bytes(&self) -> u64 {
        self.events.total_bytes
    }

    pub fn unit_store_bytes(&self) -> u64 {
        self.units.total_bytes
    }

    pub fn store_bytes(&self) -> u64 {
        self.event_store_bytes() + self.unit_store_bytes()
    }
}

// ---------------------------------------------------------------------------
// Record helpers
// ---------------------------------------------------------------------------

/// `data` starts at a record's length prefix; return the payload slice.
fn read_record_at(data: &[u8]) -> Option<&[u8]> {
    if data.len() < 4 {
        return None;
    }
    let len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if 4 + len > data.len() {
        return None;
    }
    Some(&data[4..4 + len])
}

/// Single-pass event text extraction — no allocation of RawEvent.
fn read_event_text_at(data: &[u8]) -> Option<&str> {
    let rec = read_record_at(data)?;
    if rec.len() < MIN_EVENT_RECORD {
        return None;
    }
    std::str::from_utf8(&rec[9..]).ok()
}

fn iter_units_region(base_id: u64, data: &[u8]) -> impl Iterator<Item = (UnitId, Unit)> + '_ {
    let mut pos = 0usize;
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
        let id = base_id + pos as u64;
        pos = rec_start + len;
        Some((id, unit))
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- lossless encode/decode ---

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

    // --- append-only log ---

    fn ev(ts: i64, text: &str) -> RawEvent {
        RawEvent {
            source: Source::Text,
            created_utc_secs: ts,
            text: text.into(),
        }
    }

    #[test]
    fn append_and_read_event() {
        let dir = temp_dir("append_read_event");
        let mut store = Store::open(&dir).unwrap();
        let id = store
            .append_event(&ev(1_700_000_000, "Hello, world!"))
            .unwrap();
        store.sync().unwrap();
        assert_eq!(store.event_text(id).unwrap(), "Hello, world!");
    }

    #[test]
    fn read_works_before_sync() {
        let dir = temp_dir("read_before_sync");
        let mut store = Store::open(&dir).unwrap();
        let id = store.append_event(&ev(1, "unsynced")).unwrap();
        assert_eq!(store.event_text(id).unwrap(), "unsynced");
    }

    #[test]
    fn append_and_read_unit() {
        let dir = temp_dir("append_read_unit");
        let mut store = Store::open(&dir).unwrap();
        let event_id = store
            .append_event(&ev(1_700_000_000, "Hello, world!"))
            .unwrap();
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
            event_id = store
                .append_event(&ev(1_700_000_000, "Persistent!"))
                .unwrap();
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
            id1 = store.append_event(&ev(86400, "First")).unwrap();
            store.sync().unwrap();
        }
        let id2;
        {
            let mut store = Store::open(&dir).unwrap();
            assert_eq!(store.event_text(id1).unwrap(), "First");
            id2 = store.append_event(&ev(2 * 86400, "Second")).unwrap();
            store.sync().unwrap();
        }
        {
            let store = Store::open(&dir).unwrap();
            assert_eq!(store.event_text(id1).unwrap(), "First");
            assert_eq!(store.event_text(id2).unwrap(), "Second");
        }
    }

    #[test]
    fn reopen_continues_same_segment_until_target() {
        let dir = temp_dir("reopen_same_segment");
        let id1;
        {
            let mut store = Store::open(&dir).unwrap();
            id1 = store.append_event(&ev(1, "one")).unwrap();
            store.sync().unwrap();
        }
        {
            let mut store = Store::open(&dir).unwrap();
            let id2 = store.append_event(&ev(2, "two")).unwrap();
            store.sync().unwrap();
            // Same segment (high 32 bits), later offset — no per-open segment churn.
            assert_eq!(id1 >> 32, id2 >> 32, "reopen must continue the segment");
            assert!(id2 > id1);
        }
        // Exactly one event segment file exists.
        let segs = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".seg"))
            .count();
        assert_eq!(segs, 1);
    }

    #[test]
    fn event_count_tracks() {
        let dir = temp_dir("event_count");
        let mut store = Store::open(&dir).unwrap();
        assert_eq!(store.event_count(), 0);
        store.append_event(&ev(86400, "A")).unwrap();
        store.append_event(&ev(2 * 86400, "B")).unwrap();
        assert_eq!(store.event_count(), 2);
    }

    // --- rotation ---

    fn tiny_opts() -> StoreOptions {
        StoreOptions {
            segment_target_bytes: 256,
        }
    }

    #[test]
    fn rotation_creates_segments_and_ids_stay_readable() {
        let dir = temp_dir("rotation_basic");
        let mut store = Store::open_with(&dir, tiny_opts()).unwrap();
        let mut ids = Vec::new();
        for i in 0..50 {
            ids.push(
                store
                    .append_event(&ev(i, &format!("event number {i} with some padding text")))
                    .unwrap(),
            );
        }
        store.sync().unwrap();
        let max_seg = ids.iter().map(|id| id >> 32).max().unwrap();
        assert!(max_seg >= 2, "expected multiple segments, got {max_seg}");
        for (i, id) in ids.iter().enumerate() {
            assert!(store
                .event_text(*id)
                .unwrap()
                .contains(&format!("number {i} ")));
        }
        // And across a reopen.
        let store = Store::open_with(&dir, tiny_opts()).unwrap();
        for (i, id) in ids.iter().enumerate() {
            assert!(store
                .event_text(*id)
                .unwrap()
                .contains(&format!("number {i} ")));
        }
        assert_eq!(store.event_count(), 50);
    }

    #[test]
    fn dedup_works_across_rotation_and_reopen() {
        let dir = temp_dir("dedup_rotation");
        let first_id;
        {
            let mut store = Store::open_with(&dir, tiny_opts()).unwrap();
            first_id = store
                .append_event(&ev(7, "the original event, long enough to matter"))
                .unwrap();
            for i in 0..50 {
                store
                    .append_event(&ev(1000 + i, &format!("filler event {i} aaaaaaaaaaaaaaaa")))
                    .unwrap();
            }
            // Original is now in a rotated-away segment, covered by a run.
            let dup = store
                .append_event(&ev(7, "the original event, long enough to matter"))
                .unwrap();
            assert_eq!(dup, first_id, "dedup must hit the persisted run");
            store.sync().unwrap();
        }
        {
            let mut store = Store::open_with(&dir, tiny_opts()).unwrap();
            let dup = store
                .append_event(&ev(7, "the original event, long enough to matter"))
                .unwrap();
            assert_eq!(dup, first_id, "dedup must survive reopen via runs");
        }
    }

    #[test]
    fn dedup_falls_back_to_rebuild_when_runs_deleted() {
        let dir = temp_dir("dedup_runs_deleted");
        let first_id;
        {
            let mut store = Store::open_with(&dir, tiny_opts()).unwrap();
            first_id = store
                .append_event(&ev(7, "needle event with enough text to rotate"))
                .unwrap();
            for i in 0..50 {
                store
                    .append_event(&ev(1000 + i, &format!("filler event {i} bbbbbbbbbbbbbbbb")))
                    .unwrap();
            }
            store.sync().unwrap();
        }
        // Sabotage: delete every dedup run.
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            if entry.file_name().to_string_lossy().ends_with(".run") {
                std::fs::remove_file(entry.path()).unwrap();
            }
        }
        let mut store = Store::open_with(&dir, tiny_opts()).unwrap();
        let dup = store
            .append_event(&ev(7, "needle event with enough text to rotate"))
            .unwrap();
        assert_eq!(dup, first_id, "full-rebuild fallback must still dedup");
        assert_eq!(store.event_count(), 51);
    }

    #[test]
    fn runs_get_compacted() {
        let dir = temp_dir("run_compaction");
        let mut store = Store::open_with(&dir, tiny_opts()).unwrap();
        // Enough rotations to exceed MAX_DEDUP_RUNS and trigger compaction.
        for i in 0..400 {
            store
                .append_event(&ev(i, &format!("event {i} cccccccccccccccccccccccc")))
                .unwrap();
        }
        store.sync().unwrap();
        let runs = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".run"))
            .count();
        assert!(
            runs <= MAX_DEDUP_RUNS,
            "compaction should bound run count, got {runs}"
        );
        // Dedup still resolves an early event.
        let dup = store
            .append_event(&ev(0, "event 0 cccccccccccccccccccccccc"))
            .unwrap();
        assert_eq!(dup >> 32, 0, "event 0 lives in segment 0");
        assert_eq!(store.event_count(), 400);
    }

    // --- crash recovery ---

    #[test]
    fn torn_event_tail_is_recovered() {
        let dir = temp_dir("torn_event_tail");
        let (id1, id2);
        {
            let mut store = Store::open(&dir).unwrap();
            id1 = store.append_event(&ev(1, "alpha")).unwrap();
            id2 = store.append_event(&ev(2, "beta")).unwrap();
            store.sync().unwrap();
        }
        // Simulate a crash mid-append: a length prefix promising more bytes
        // than exist, plus some garbage.
        {
            use std::io::Write;
            let path = dir.join("events-000.seg");
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&999u32.to_le_bytes()).unwrap();
            f.write_all(b"torn").unwrap();
        }
        let mut store = Store::open(&dir).unwrap();
        assert!(store.recovered_truncated_bytes() > 0);
        assert_eq!(store.event_text(id1).unwrap(), "alpha");
        assert_eq!(store.event_text(id2).unwrap(), "beta");
        assert_eq!(store.event_count(), 2);
        // Appending after recovery lands cleanly where the garbage was.
        let id3 = store.append_event(&ev(3, "gamma")).unwrap();
        store.sync().unwrap();
        assert_eq!(store.event_text(id3).unwrap(), "gamma");
        let store = Store::open(&dir).unwrap();
        assert_eq!(store.event_text(id3).unwrap(), "gamma");
        assert_eq!(store.event_count(), 3);
    }

    #[test]
    fn torn_unit_tail_is_recovered() {
        let dir = temp_dir("torn_unit_tail");
        let unit_id;
        {
            let mut store = Store::open(&dir).unwrap();
            let event_id = store.append_event(&ev(1, "Hello there.")).unwrap();
            unit_id = store
                .append_unit(&Unit {
                    event: event_id,
                    byte_start: 0,
                    byte_end: 5,
                })
                .unwrap();
            store.sync().unwrap();
        }
        {
            use std::io::Write;
            let path = dir.join("units-000.idx");
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&16u32.to_le_bytes()).unwrap();
            f.write_all(b"short").unwrap(); // promised 16, delivered 5
        }
        let store = Store::open(&dir).unwrap();
        assert!(store.recovered_truncated_bytes() > 0);
        assert_eq!(store.unit_text(unit_id).unwrap(), "Hello");
        assert_eq!(store.unit_count(), 1);
        assert_eq!(store.units().count(), 1);
    }

    // --- dedup basics ---

    #[test]
    fn dedup_same_event_returns_same_id() {
        let dir = temp_dir("dedup");
        let mut store = Store::open(&dir).unwrap();
        let e = ev(1_700_000_000, "Duplicate me");
        let id1 = store.append_event(&e).unwrap();
        let id2 = store.append_event(&e).unwrap();
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

    // --- derive_units ---

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

    // --- provenance ---

    #[test]
    fn provenance_roundtrip_100_percent() {
        let dir = temp_dir("provenance");
        let mut store = Store::open(&dir).unwrap();
        let text = "First sentence. Second one! Third? End.";
        let event_id = store.append_event(&ev(86400, text)).unwrap();
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
        let event_id = store.append_event(&ev(86400, "Hi")).unwrap();
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

    // --- unit iter & counts ---

    #[test]
    fn unit_iter_produces_all_units() {
        let dir = temp_dir("unit_iter");
        let mut store = Store::open(&dir).unwrap();
        let text = "A. B. C.";
        let event_id = store.append_event(&ev(86400, text)).unwrap();
        store.sync().unwrap();
        let derived = derive_units(event_id, text);
        for u in &derived {
            store.append_unit(u).unwrap();
        }
        store.sync().unwrap();
        let collected: Vec<_> = store.units().collect();
        assert_eq!(collected.len(), derived.len());
    }

    #[test]
    fn unit_count_matches_units_even_before_sync() {
        let dir = temp_dir("unit_count_pending");
        let mut store = Store::open(&dir).unwrap();
        let event_id = store.append_event(&ev(86400, "A. B.")).unwrap();
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

    #[test]
    fn unit_count_survives_reopen_with_and_without_meta() {
        let dir = temp_dir("unit_count_reopen");
        {
            let mut store = Store::open_with(&dir, tiny_opts()).unwrap();
            let event_id = store
                .append_event(&ev(1, "Some text. More text. Even more."))
                .unwrap();
            for u in derive_units(event_id, "Some text. More text. Even more.") {
                store.append_unit(&u).unwrap();
            }
            // Many units to force unit-segment rotation (meta gets written).
            for i in 0..100 {
                store
                    .append_unit(&Unit {
                        event: event_id,
                        byte_start: 0,
                        byte_end: 4 + (i % 3),
                    })
                    .unwrap();
            }
            store.sync().unwrap();
            assert_eq!(store.unit_count(), 103);
        }
        {
            let store = Store::open_with(&dir, tiny_opts()).unwrap();
            assert_eq!(store.unit_count(), 103, "count via meta + tail scan");
            assert_eq!(store.units().count(), 103);
        }
        // Without meta: full-scan fallback must agree.
        std::fs::remove_file(dir.join(COUNTS_META_FILE)).unwrap();
        let store = Store::open_with(&dir, tiny_opts()).unwrap();
        assert_eq!(store.unit_count(), 103, "full-scan fallback");
    }

    // --- overhead & determinism ---

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
                .append_event(&ev(1_700_000_000 + total_raw as i64, t))
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

    #[test]
    fn deterministic_store_files_byte_identical() {
        fn store_bytes(dir_name: &str) -> Vec<(String, Vec<u8>)> {
            let dir = temp_dir(dir_name);
            let mut store = Store::open(&dir).unwrap();
            let texts = ["Alpha. Beta.", "Gamma. Delta.", "Epsilon."];
            for t in &texts {
                store.append_event(&ev(1_700_000_000, t)).unwrap();
            }
            store.sync().unwrap();
            let mut files = Vec::new();
            for entry in std::fs::read_dir(&dir).unwrap() {
                let e = entry.unwrap();
                let path = e.path();
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                if name.ends_with(".seg") || name.ends_with(".idx") || name.ends_with(".run") {
                    files.push((name, std::fs::read(&path).unwrap()));
                }
            }
            files.sort_by(|a, b| a.0.cmp(&b.0));
            files
        }
        let a = store_bytes("det_byte_a");
        let b = store_bytes("det_byte_b");
        assert_eq!(a, b, "store file bytes must be identical");
    }

    #[test]
    fn validated_prefix_handles_edge_cases() {
        // Empty, short, exact record, torn length, torn payload, tiny record.
        assert_eq!(validated_prefix_len(&[], 9), 0);
        assert_eq!(validated_prefix_len(&[1, 0], 9), 0);
        let mut good = Vec::new();
        good.extend_from_slice(&10u32.to_le_bytes());
        good.extend_from_slice(&[7u8; 10]);
        assert_eq!(validated_prefix_len(&good, 9), 14);
        let mut torn = good.clone();
        torn.extend_from_slice(&100u32.to_le_bytes());
        torn.extend_from_slice(&[1u8; 5]);
        assert_eq!(validated_prefix_len(&torn, 9), 14);
        let mut tiny = good.clone();
        tiny.extend_from_slice(&3u32.to_le_bytes()); // < min record size
        tiny.extend_from_slice(&[1u8; 3]);
        assert_eq!(validated_prefix_len(&tiny, 9), 14);
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("skinki_store_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
