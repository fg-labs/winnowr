//! Reading text files that may be gzip- or BGZF-compressed, and writing tab-separated files.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// The empty block that ends a complete BGZF file (SAM/BAM specification, section 4.1.2).
pub const BGZF_EOF: [u8; 28] = [
    0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
    0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Appends the BGZF blocks of a complete BGZF file to `out`, without its end-of-file block and
/// its first `skip` bytes (which must equal `expected_prefix`, e.g. a header written as its own
/// blocks). The blocks are copied through a fixed buffer, not read into memory.
///
/// # Errors
///
/// Returns an error if the file cannot be read, lacks the end-of-file block, or does not start
/// with `expected_prefix`.
pub fn append_bgzf_blocks<W: std::io::Write>(
    out: &mut W,
    path: &Path,
    expected_prefix: &[u8],
) -> Result<()> {
    let context = || format!("failed to read {}", path.display());
    let mut file = File::open(path).with_context(context)?;
    let length = file.metadata().with_context(context)?.len();
    let (prefix_length, eof_length) = (expected_prefix.len() as u64, BGZF_EOF.len() as u64);
    if length < eof_length {
        anyhow::bail!("{} is not a complete BGZF file", path.display());
    }
    file.seek(SeekFrom::Start(length - eof_length)).with_context(context)?;
    let mut tail = [0u8; BGZF_EOF.len()];
    file.read_exact(&mut tail).with_context(context)?;
    if tail != BGZF_EOF {
        anyhow::bail!("{} is not a complete BGZF file", path.display());
    }
    let mut prefix = vec![0u8; expected_prefix.len()];
    file.seek(SeekFrom::Start(0)).with_context(context)?;
    if length - eof_length < prefix_length
        || file.read_exact(&mut prefix).is_err()
        || prefix != expected_prefix
    {
        anyhow::bail!("{} does not start with the expected BGZF blocks", path.display());
    }
    let mut body = BufReader::new(file).take(length - eof_length - prefix_length);
    std::io::copy(&mut body, out).with_context(|| "failed to write BGZF blocks".to_string())?;
    Ok(())
}

/// Opens a text file for buffered reading, decompressing it if it is gzip- or BGZF-compressed
/// (whatever its extension), with htslib.
///
/// A BGZF file without the end-of-file block was probably truncated at a block boundary, which
/// otherwise decompresses without error; htslib warns about it on stderr.
///
/// # Errors
///
/// Returns an error if the file cannot be opened.
pub fn open_text<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>> {
    let path = path.as_ref();
    let reader = rust_htslib::bgzf::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    Ok(Box::new(BufReader::new(reader)))
}

/// The device and inode of an opened regular input file, independent of its later pathname.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FileIdentity {
    device: u64,
    inode: u64,
}

impl FileIdentity {
    /// Identifies a regular file from its handle's metadata; streams have no file identity.
    fn from_metadata(metadata: &std::fs::Metadata) -> Option<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            metadata.is_file().then(|| Self { device: metadata.dev(), inode: metadata.ino() })
        }
        #[cfg(not(unix))]
        {
            let _ = metadata;
            None
        }
    }

    /// True if `path` currently names this regular file, following symlinks.
    pub(crate) fn matches_path(&self, path: &Path) -> bool {
        std::fs::metadata(path)
            .ok()
            .and_then(|metadata| Self::from_metadata(&metadata))
            .is_some_and(|identity| identity == *self)
    }
}

/// Opens text and returns the identity of the regular file supplying its bytes, if any.
///
/// Like [`open_text`], accepts plain text, gzip, BGZF (with or without its EOF block), and pipes.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, its metadata cannot be read, or its compression
/// prefix cannot be read. Further read or decompression errors are reported by the reader.
pub fn open_text_with_identity<P: AsRef<Path>>(
    path: P,
) -> Result<(Box<dyn BufRead>, Option<FileIdentity>)> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    text_reader_with_identity(file, path)
}

/// Opens a bundled text input without waiting for a stream producer, rejecting nonregular
/// handles before reading even the compression prefix. Explicit streams use
/// [`open_text_with_identity`] instead.
///
/// # Errors
///
/// Returns an error if opening, metadata inspection, or prefix reading fails, or the opened
/// handle is not a regular file.
pub fn open_regular_text_with_identity<P: AsRef<Path>>(
    path: P,
) -> Result<(Box<dyn BufRead>, Option<FileIdentity>)> {
    let path = path.as_ref();
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path).with_context(|| format!("failed to open {}", path.display()))?;
    anyhow::ensure!(
        file.metadata().with_context(|| format!("failed to inspect {}", path.display()))?.is_file(),
        "bundled region BED {} is not a regular file",
        path.display()
    );
    text_reader_with_identity(file, path)
}

/// Non-opening equality for existing physical inputs (including FIFOs) and repeated missing paths.
pub fn same_input_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(a), Ok(b)) = (std::fs::metadata(a), std::fs::metadata(b)) {
            return (a.dev(), a.ino()) == (b.dev(), b.ino());
        }
    }
    matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

/// Identifies and decompresses the same opened handle without reopening or seeking it.
/// `path` is used only for error messages.
fn text_reader_with_identity(
    mut file: File,
    path: &Path,
) -> Result<(Box<dyn BufRead>, Option<FileIdentity>)> {
    let context = || format!("failed to read {}", path.display());
    let identity = FileIdentity::from_metadata(&file.metadata().with_context(context)?);
    // A pipe may return only one byte per read. Collect both magic bytes, then replay them so
    // neither plain nor compressed readers lose their prefix, even on an empty or one-byte input.
    let mut prefix = Vec::with_capacity(2);
    Read::by_ref(&mut file).take(2).read_to_end(&mut prefix).with_context(context)?;
    let gzip = prefix == [0x1f, 0x8b];
    let reader = std::io::Cursor::new(prefix).chain(file);
    let reader: Box<dyn BufRead> = if gzip {
        // BGZF and concatenated gzip are both sequences of gzip members. Unlike winnowr's own
        // tables, external BEDs remain readable when their producer omits the BGZF EOF marker.
        Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(reader)))
    } else {
        Box::new(BufReader::new(reader))
    };
    Ok((reader, identity))
}

/// Opens a table that winnowr wrote, as [`open_text`] does, but rejects a BGZF file without the
/// end-of-file block: winnowr always writes it, so the table was truncated (e.g. a killed run).
///
/// # Errors
///
/// Returns an error if the file cannot be opened, changes from regular to nonregular while being
/// opened, has a malformed gzip extra field, or is BGZF-compressed (whatever subfields precede
/// `BC`) without the end-of-file block.
pub fn open_table<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>> {
    open_table_with(path.as_ref(), || {}, || {})
}

/// [`open_table`], calling `before_open` between classifying the path and opening it, and
/// `after_check` between checking the opened file and reading it (so tests can replace the path
/// in between).
///
/// A regular file is opened once: its compression and end-of-file block are checked on that
/// handle, and its rows are read from the same file, so a file replaced after the check is never
/// read in its place. Anything else (e.g. a pipe) is read by path with [`open_text`], once, as
/// its start cannot be checked without consuming it.
fn open_table_with<F: FnOnce(), G: FnOnce()>(
    path: &Path,
    before_open: F,
    after_check: G,
) -> Result<Box<dyn BufRead>> {
    // Only a regular file can be checked: reading the start of a pipe (e.g. process substitution)
    // would consume it.
    if !is_regular_file(path) {
        return open_text(path);
    }
    before_open();
    let mut file = open_regular(path)?;
    let gzip = check_compressed(&mut file, path)?;
    after_check();
    // The rows are read from the handle checked, never by path: not even its descriptor's
    // `/dev/fd/N`, which macOS can transiently fail to find while the descriptor is open. BGZF is
    // a sequence of gzip members.
    Ok(if gzip {
        Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    })
}

/// Opens `path`, classified as a regular file, without waiting for a stream's producer, and
/// fails if the opened handle is not a regular file (the path was replaced, e.g. by a pipe).
///
/// # Errors
///
/// Returns an error if the file cannot be opened or inspected, or is not a regular file.
fn open_regular(path: &Path) -> Result<File> {
    let context = || format!("failed to open {}", path.display());
    let file = nonblocking_read().open(path).with_context(context)?;
    check_still_regular(&file.metadata().with_context(context)?, path)?;
    Ok(file)
}

/// Options to read a path classified as a regular file: it can become a FIFO after it was
/// classified, so never wait for a producer.
fn nonblocking_read() -> std::fs::OpenOptions {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    options
}

/// Fails if the handle opened for `path`, classified as a regular file, is not one (the path was
/// replaced, e.g. by a pipe): a raced FIFO may have released its producer, so callers must not
/// continue by reopening it.
///
/// # Errors
///
/// Returns an error if `metadata` is not that of a regular file.
fn check_still_regular(metadata: &std::fs::Metadata, path: &Path) -> Result<()> {
    if !metadata.is_file() {
        anyhow::bail!(
            "{} changed type while being opened: expected a regular file",
            path.display()
        );
    }
    Ok(())
}

/// Opens a table for metadata-only inspection (e.g. its provenance prelude) if, and only if, it
/// is a regular file; `None` for anything else (e.g. a pipe, which reading would consume).
///
/// The regular-file decision is made on the opened handle, and every byte returned is read from
/// that same handle. If a regular path becomes nonregular before opening, inspection fails
/// instead of continuing to a second open for rows. The nonblocking open can still release a
/// FIFO producer on that failed run; it does not preserve the stream for a retry.
///
/// Prelude inspection needs only the first lines, so decompression here uses flate2; tables are
/// still read in full with [`open_table`] (htslib).
///
/// # Errors
///
/// Returns an error if the file cannot be read, changes from regular to nonregular while being
/// opened, or is BGZF-compressed without the end-of-file block.
pub fn open_regular_table(path: &Path) -> Result<Option<Box<dyn BufRead>>> {
    open_regular_table_with(path, || {})
}

/// [`open_regular_table`], calling `before_open` between classifying the path and opening it (so
/// tests can replace the path in between).
fn open_regular_table_with<F: FnOnce()>(
    path: &Path,
    before_open: F,
) -> Result<Option<Box<dyn BufRead>>> {
    // Skip known streams. A raced FIFO may still release its producer, so detecting one on the
    // opened handle must fail the run rather than retrying the stream for rows.
    if !is_regular_file(path) {
        return Ok(None);
    }
    before_open();
    decompressing_reader(open_regular(path)?, path).map(Some)
}

/// A buffered reader of an opened regular file, decompressing it if it is gzip- or
/// BGZF-compressed (whatever its extension). A BGZF file must end with the end-of-file block,
/// which is checked by seeking on `file` itself. `path` only names the file in errors.
///
/// # Errors
///
/// Returns an error if the file cannot be read, or is BGZF-compressed without the end-of-file
/// block.
fn decompressing_reader(mut file: File, path: &Path) -> Result<Box<dyn BufRead>> {
    Ok(if check_compressed(&mut file, path)? {
        // BGZF is a series of gzip members, as is concatenated gzip.
        Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    })
}

/// Checks an opened regular file and rewinds it: true if it is gzip-compressed, and an error if
/// it is BGZF (its first block has a `BC` subfield, wherever among its extra subfields) without
/// the end-of-file block. `path` only names the file in errors.
///
/// # Errors
///
/// Returns an error if the file cannot be read, has a malformed gzip extra field, or is
/// BGZF-compressed without the end-of-file block.
fn check_compressed(file: &mut File, path: &Path) -> Result<bool> {
    let context = || format!("failed to read {}", path.display());
    let length = file.metadata().with_context(context)?.len();
    let mut prefix = [0u8; 12];
    let prefix_length = usize::try_from(length.min(prefix.len() as u64))?;
    file.read_exact(&mut prefix[..prefix_length]).with_context(context)?;
    let gzip = prefix[..2] == [0x1f, 0x8b];
    let bgzf = if prefix_length == prefix.len() && gzip && prefix[2] == 8 && prefix[3] & 4 != 0 {
        let mut extra = vec![0; usize::from(u16::from_le_bytes([prefix[10], prefix[11]]))];
        file.read_exact(&mut extra).with_context(context)?;
        has_bgzf_subfield(&extra)
            .with_context(|| format!("{} has a malformed gzip extra field", path.display()))?
    } else {
        false
    };
    if bgzf {
        let eof_length = BGZF_EOF.len() as u64;
        let mut tail = [0u8; BGZF_EOF.len()];
        let complete = length >= eof_length && {
            file.seek(SeekFrom::Start(length - eof_length)).with_context(context)?;
            file.read_exact(&mut tail).with_context(context)?;
            tail == BGZF_EOF
        };
        if !complete {
            anyhow::bail!(
                "{} is truncated (no BGZF end-of-file block): write it again",
                path.display()
            );
        }
    }
    file.rewind().with_context(context)?;
    Ok(gzip)
}

/// True if a gzip extra field (`extra`, its `XLEN` bytes) has a complete BGZF `BC` subfield,
/// walking each subfield, as other subfields may precede `BC` (SAM/BAM spec 4.1).
///
/// # Errors
///
/// Returns an error if a subfield overruns the field, or the field ends inside a subfield's
/// four-byte header.
pub fn has_bgzf_subfield(extra: &[u8]) -> Result<bool> {
    let mut bgzf = false;
    let mut rest = extra;
    while rest.len() >= 4 {
        let size = usize::from(u16::from_le_bytes([rest[2], rest[3]]));
        if size > rest.len() - 4 {
            anyhow::bail!("a gzip extra subfield overruns the field");
        }
        bgzf |= rest[..2] == *b"BC" && size == 2;
        rest = &rest[4 + size..];
    }
    if !rest.is_empty() {
        anyhow::bail!("the gzip extra field ends inside a subfield header");
    }
    Ok(bgzf)
}

/// True if `path` is a regular file: not, e.g., a pipe (process substitution), which reading
/// would consume, or a missing file.
pub fn is_regular_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
}

/// True if a regular file does not end with the BGZF end-of-file block. Files that cannot be
/// checked (e.g. pipes, missing files, or unreadable files) are assumed complete.
///
/// # Errors
///
/// Returns an error if a path classified as regular is nonregular when opened. A raced FIFO
/// may have released its producer, so callers must not continue by reopening it for rows.
pub fn lacks_bgzf_eof(path: &Path) -> Result<bool> {
    lacks_bgzf_eof_with(path, || {})
}

/// Checks the EOF after calling `before_open` between classification and opening the path.
fn lacks_bgzf_eof_with<F: FnOnce()>(path: &Path, before_open: F) -> Result<bool> {
    if !is_regular_file(path) {
        return Ok(false);
    }
    before_open();
    let Ok(mut file) = nonblocking_read().open(path) else { return Ok(false) };
    let Ok(metadata) = file.metadata() else { return Ok(false) };
    check_still_regular(&metadata, path)?;
    let mut check = || -> std::io::Result<bool> {
        if metadata.len() < BGZF_EOF.len() as u64 {
            return Ok(true);
        }
        file.seek(SeekFrom::End(-(BGZF_EOF.len() as i64)))?;
        let mut tail = [0u8; BGZF_EOF.len()];
        file.read_exact(&mut tail)?;
        Ok(tail != BGZF_EOF)
    };
    Ok(check().unwrap_or(false))
}

/// Writes a TSV without overwriting any of the regular BED files actually consumed.
///
/// # Errors
///
/// Returns an error if the opened output is a protected BED or cannot be written.
pub fn write_tsv_protecting<I: Iterator<Item = Vec<String>>>(
    path: &Path,
    header: &[&str],
    rows: I,
    protected_beds: &[FileIdentity],
) -> Result<()> {
    let file = create_output(path, protected_beds)?;
    let mut out = std::io::BufWriter::new(file);
    writeln!(out, "{}", header.join("\t"))?;
    for row in rows {
        debug_assert_eq!(row.len(), header.len());
        writeln!(out, "{}", row.join("\t"))?;
    }
    out.flush().with_context(|| format!("failed to write {}", path.display()))
}

/// Opens an output without truncation, checks its handle against consumed BEDs, then truncates
/// that same handle. Path changes cannot redirect the subsequent write to a protected BED.
///
/// # Errors
///
/// Returns an error before truncation if the opened file is a protected BED, or if it cannot be
/// opened, inspected or truncated. Without protected BEDs, preserves ordinary file creation.
pub fn create_output(path: &Path, protected_beds: &[FileIdentity]) -> Result<File> {
    let context = || format!("failed to create {}", path.display());
    if protected_beds.is_empty() {
        return File::create(path).with_context(context);
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .with_context(context)?;
    let metadata = file.metadata().with_context(context)?;
    if FileIdentity::from_metadata(&metadata)
        .is_some_and(|identity| protected_beds.contains(&identity))
    {
        anyhow::bail!(
            "the output {} is the input region BED; write to another path",
            path.display()
        );
    }
    // Streams have no previous contents to truncate and remain valid output destinations.
    if metadata.is_file() {
        file.set_len(0).with_context(context)?;
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use tempfile::TempDir;

    use super::*;

    #[cfg(unix)]
    #[test]
    fn guarded_output_writes_the_checked_handle_after_its_path_changes() {
        let dir = TempDir::new().unwrap();
        let bed = dir.path().join("input.bed");
        let output = dir.path().join("report.tsv");
        let moved = dir.path().join("moved.tsv");
        let original = b"chr1\t0\t100\n";
        std::fs::write(&bed, original).unwrap();
        let (_, identity) = open_text_with_identity(&bed).unwrap();
        let protected = [identity.unwrap()];
        std::fs::write(&output, b"old report contents").unwrap();
        let mut file = create_output(&output, &protected).unwrap();
        std::fs::rename(&output, &moved).unwrap();
        std::os::unix::fs::symlink(&bed, &output).unwrap();
        file.write_all(b"new report\n").unwrap();
        file.flush().unwrap();
        assert_eq!(std::fs::read(&bed).unwrap(), original);
        assert_eq!(std::fs::read(&moved).unwrap(), b"new report\n");
        let error = create_output(&output, &protected).unwrap_err().to_string();
        assert!(error.contains("is the input region BED"), "{error}");
        assert_eq!(std::fs::read(&bed).unwrap(), original);
    }

    /// The BGZF block(s) of `text`, without the end-of-file block, as htslib writes them.
    fn bgzf_block(text: &[u8]) -> Vec<u8> {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("block.gz");
        let mut writer = rust_htslib::bgzf::Writer::from_path(&path).unwrap();
        writer.write_all(text).unwrap();
        drop(writer);
        let bytes = std::fs::read(&path).unwrap();
        bytes.strip_suffix(&BGZF_EOF).unwrap().to_vec()
    }

    fn gzip(text: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(text).unwrap();
        encoder.finish().unwrap()
    }

    fn read_text(dir: &TempDir, bytes: &[u8]) -> Vec<u8> {
        let path = dir.path().join("in.txt");
        std::fs::write(&path, bytes).unwrap();
        let mut text = Vec::new();
        open_text(&path).unwrap().read_to_end(&mut text).unwrap();
        text
    }

    #[test]
    fn reads_plain_gzip_and_multi_member_text() {
        let dir = TempDir::new().unwrap();
        assert_eq!(read_text(&dir, b"a\nb\n"), b"a\nb\n");
        assert_eq!(read_text(&dir, b""), b"");
        assert_eq!(read_text(&dir, &gzip(b"a\nb\n")), b"a\nb\n");
        let members = [bgzf_block(b"a\n"), bgzf_block(b"b\n"), BGZF_EOF.to_vec()].concat();
        assert_eq!(read_text(&dir, &members), b"a\nb\n");
    }

    #[test]
    fn text_identity_reader_preserves_empty_short_and_long_inputs() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.bed");
        let long = b"chr1\t0\t5\n".repeat(10_000);
        for text in [b"".as_slice(), b"c", &[0x1f], long.as_slice()] {
            for bytes in [text.to_vec(), gzip(text)] {
                std::fs::write(&path, bytes).unwrap();
                let (mut reader, _) = open_text_with_identity(&path).unwrap();
                let mut actual = Vec::new();
                reader.read_to_end(&mut actual).unwrap();
                assert_eq!(actual, text);
            }
        }
        std::fs::write(&path, BGZF_EOF).unwrap();
        let (mut reader, _) = open_text_with_identity(&path).unwrap();
        let mut text = Vec::new();
        reader.read_to_end(&mut text).unwrap();
        assert!(text.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn text_identity_and_bytes_come_from_the_same_opened_handle() {
        let dir = TempDir::new().unwrap();
        let original = dir.path().join("original.bed");
        let other = dir.path().join("other.bed");
        let path = dir.path().join("input.bed");
        let alias = dir.path().join("alias.bed");
        for bytes in [
            b"chr1\t0\t5\n".to_vec(),
            gzip(b"chr1\t0\t5\n"),
            [gzip(b"chr1\t"), gzip(b"0\t5\n")].concat(),
            [bgzf_block(b"chr1\t0\t5\n"), BGZF_EOF.to_vec()].concat(),
            bgzf_block(b"chr1\t0\t5\n"),
        ] {
            std::fs::write(&original, bytes).unwrap();
            std::fs::write(&other, b"chr2\t0\t9\n").unwrap();
            std::os::unix::fs::symlink(&original, &path).unwrap();
            std::fs::hard_link(&original, &alias).unwrap();
            let (mut reader, identity) = open_text_with_identity(&path).unwrap();
            let opened = File::open(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            std::os::unix::fs::symlink(&other, &path).unwrap();
            // Construct another reader after retargeting, so even its metadata and magic-byte
            // detection must use the supplied handle, not a second observation of the path.
            let (mut raced_reader, raced_identity) =
                text_reader_with_identity(opened, &path).unwrap();
            assert_eq!(raced_identity, identity);
            for reader in [&mut reader, &mut raced_reader] {
                let mut text = String::new();
                reader.read_to_string(&mut text).unwrap();
                assert_eq!(text, "chr1\t0\t5\n");
            }
            let identity = identity.expect("a regular BED has an identity");
            assert!(identity.matches_path(&alias));
            assert!(!identity.matches_path(&path));
            std::fs::remove_file(&path).unwrap();
            std::fs::remove_file(&alias).unwrap();
        }
    }

    #[test]
    fn tables_without_the_bgzf_eof_block_are_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.tsv.gz");
        std::fs::write(&path, [bgzf_block(b"a\n"), BGZF_EOF.to_vec()].concat()).unwrap();
        let mut text = String::new();
        open_table(&path).unwrap().read_to_string(&mut text).unwrap();
        assert_eq!(text, "a\n");
        std::fs::write(&path, bgzf_block(b"a\n")).unwrap();
        let message = open_table(&path).err().unwrap().to_string();
        assert!(
            message
                .ends_with("table.tsv.gz is truncated (no BGZF end-of-file block): write it again"),
            "{message}"
        );
        // Plain text and plain gzip have no end-of-file block to check.
        std::fs::write(&path, gzip(b"a\n")).unwrap();
        assert!(open_table(&path).is_ok());
        let plain = dir.path().join("table.tsv");
        std::fs::write(&plain, b"a\n").unwrap();
        assert!(open_table(&plain).is_ok());
    }

    /// A table read from a pipe (e.g. `--strlen <(...)`) is read once, from its start. The reader
    /// and writer are separate processes under one deadline, so a missed read fails, not hangs.
    #[cfg(unix)]
    #[test]
    fn tables_are_read_from_a_pipe() {
        let dir = TempDir::new().unwrap();
        let fifo = dir.path().join("table.fifo");
        let status = std::process::Command::new("mkfifo").arg(&fifo).status().unwrap();
        assert!(status.success());
        let mut processes = crate::testing::TestProcesses::default();
        for role in ["reader", "writer"] {
            let envs = [
                ("WINNOWR_IO_FIFO_ROLE", std::ffi::OsStr::new(role)),
                ("WINNOWR_IO_FIFO", fifo.as_os_str()),
            ];
            // A log each: the two run at once.
            let log = dir.path().join(format!("{role}.log"));
            processes.spawn_test("io::tests::fifo_table_child", envs, &log);
        }
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    /// Child of [`tables_are_read_from_a_pipe`]: one end of its pipe.
    #[cfg(unix)]
    #[test]
    fn fifo_table_child() {
        let Some(role) = std::env::var_os("WINNOWR_IO_FIFO_ROLE") else { return };
        let fifo = std::env::var_os("WINNOWR_IO_FIFO").unwrap();
        if role == "writer" {
            let data = [bgzf_block(b"a\nb\n"), BGZF_EOF.to_vec()].concat();
            std::fs::write(fifo, data).unwrap();
        } else {
            let mut text = String::new();
            open_table(fifo).unwrap().read_to_string(&mut text).unwrap();
            assert_eq!(text, "a\nb\n");
        }
        crate::testing::child_body_ran();
    }

    fn read_regular(path: &Path) -> Result<Option<Vec<u8>>> {
        let Some(mut reader) = open_regular_table(path)? else { return Ok(None) };
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(Some(bytes))
    }

    #[test]
    fn regular_tables_are_decompressed_from_their_start() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.tsv.gz");
        let mut long = b"#a long first line, longer than the prefix used to detect gzip\n".to_vec();
        long.extend_from_slice(b"b\n");
        let bgzf = [bgzf_block(b"a\n"), bgzf_block(b"b\n"), BGZF_EOF.to_vec()].concat();
        for (bytes, expected) in [
            (Vec::new(), b"".to_vec()),
            (b"a".to_vec(), b"a".to_vec()),
            (vec![0x1f], vec![0x1f]),
            (long.clone(), long.clone()),
            (gzip(&long), long.clone()),
            ([gzip(b"a\n"), gzip(b"b\n")].concat(), b"a\nb\n".to_vec()),
            (bgzf, b"a\nb\n".to_vec()),
            (BGZF_EOF.to_vec(), Vec::new()),
        ] {
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(read_regular(&path).unwrap(), Some(expected.clone()), "{bytes:?}");
            // The rows of a table are read with htslib, from the handle checked.
            let mut rows = Vec::new();
            open_table(&path).unwrap().read_to_end(&mut rows).unwrap();
            assert_eq!(rows, expected, "{bytes:?}");
        }
    }

    #[rstest::rstest]
    #[case::none(&[], Some(false))]
    #[case::bc(&[b'B', b'C', 2, 0, 0x1b, 0], Some(true))]
    #[case::xy_then_bc(&[b'X', b'Y', 0, 0, b'B', b'C', 2, 0, 0x1b, 0], Some(true))]
    #[case::other(&[b'X', b'Y', 2, 0, 1, 2], Some(false))]
    #[case::bc_of_another_length(&[b'B', b'C', 1, 0, 0x1b], Some(false))]
    #[case::overrun(&[b'B', b'C', 4, 0, 0x1b, 0], None)]
    #[case::partial_header(&[b'B', b'C', 2, 0, 0x1b, 0, b'X'], None)]
    fn bgzf_subfields_are_walked_within_the_field(
        #[case] extra: &[u8],
        #[case] expected: Option<bool>,
    ) {
        assert_eq!(has_bgzf_subfield(extra).ok(), expected);
    }

    /// A gzip extra field whose subfield overruns it is malformed, named with the table.
    #[test]
    fn a_malformed_gzip_extra_field_is_an_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.tsv.gz");
        let mut bytes = vec![0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0xff, 6, 0];
        bytes.extend_from_slice(&[b'B', b'C', 9, 0, 0, 0]);
        bytes.extend_from_slice(&[0; 32]);
        std::fs::write(&path, bytes).unwrap();
        for error in [open_table(&path).err().unwrap(), read_regular(&path).unwrap_err()] {
            let message = format!("{error:#}");
            assert!(message.contains("malformed gzip extra field"), "{message}");
            assert!(message.contains(&path.display().to_string()), "{message}");
        }
    }

    #[test]
    fn review_bgzf_with_an_extra_subfield_before_bc_checks_eof() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("extra.bgz");
        let mut block = bgzf_block(b"a\n");
        // Insert a legal, zero-length gzip extra subfield before BC; update XLEN and BSIZE.
        block.splice(12..12, [b'X', b'Y', 0, 0]);
        block[10..12].copy_from_slice(&10u16.to_le_bytes());
        let bsize = u16::try_from(block.len() - 1).unwrap();
        block[20..22].copy_from_slice(&bsize.to_le_bytes());
        std::fs::write(&path, [&block[..], &BGZF_EOF].concat()).unwrap();
        assert_eq!(read_regular(&path).unwrap().unwrap(), b"a\n");
        std::fs::write(&path, block).unwrap();
        let error = read_regular(&path).unwrap_err().to_string();
        assert!(error.contains("no BGZF end-of-file block"), "{error}");
    }

    /// Rows are read with the same BGZF detection as the preflight: a block whose `BC` follows
    /// another subfield is BGZF, so its missing end-of-file block is an error here too.
    #[test]
    fn open_table_detects_bgzf_with_an_extra_subfield_before_bc() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("extra.bgz");
        let mut block = bgzf_block(b"a\n");
        block.splice(12..12, [b'X', b'Y', 0, 0]);
        block[10..12].copy_from_slice(&10u16.to_le_bytes());
        let bsize = u16::try_from(block.len() - 1).unwrap();
        block[20..22].copy_from_slice(&bsize.to_le_bytes());
        std::fs::write(&path, [&block[..], &BGZF_EOF].concat()).unwrap();
        let mut text = String::new();
        open_table(&path).unwrap().read_to_string(&mut text).unwrap();
        assert_eq!(text, "a\n");
        std::fs::write(&path, block).unwrap();
        let error = open_table(&path).err().expect("a BGZF table without its EOF block");
        assert!(error.to_string().contains("no BGZF end-of-file block"), "{error:#}");
    }

    /// The rows come from the file whose end-of-file block was checked, even if the path names
    /// another (here a BGZF file without the block) by the time they are read.
    #[test]
    fn open_table_reads_the_file_it_checked() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.tsv.gz");
        let moved = dir.path().join("moved.tsv.gz");
        std::fs::write(&path, [bgzf_block(b"checked\n"), BGZF_EOF.to_vec()].concat()).unwrap();
        let replace = || {
            std::fs::rename(&path, &moved).unwrap();
            std::fs::write(&path, bgzf_block(b"replacement\n")).unwrap();
        };
        let mut text = String::new();
        open_table_with(&path, || {}, replace).unwrap().read_to_string(&mut text).unwrap();
        assert_eq!(text, "checked\n");
    }

    #[test]
    fn regular_tables_without_the_bgzf_eof_block_are_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.tsv.gz");
        for bytes in [bgzf_block(b"a\n"), [bgzf_block(b"a\n"), BGZF_EOF[..27].to_vec()].concat()] {
            std::fs::write(&path, bytes).unwrap();
            let message = read_regular(&path).unwrap_err().to_string();
            assert!(message.ends_with("is truncated (no BGZF end-of-file block): write it again"));
        }
        // A missing file is not a regular file; scoring reports it when it reads the table.
        assert!(read_regular(&dir.path().join("missing")).unwrap().is_none());
    }

    /// A pipe is never opened, which would consume it (or release and then break its writer).
    /// Run in a child, so a regression that waits for a writer fails, not hangs.
    #[cfg(unix)]
    #[test]
    fn pipes_are_not_regular_tables() {
        let dir = TempDir::new().unwrap();
        let fifo = dir.path().join("table.fifo");
        assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        let log = dir.path().join("child.log");
        let mut processes = crate::testing::TestProcesses::default();
        let envs = [("WINNOWR_IO_UNREAD_FIFO", &fifo)];
        processes.spawn_test("io::tests::unread_pipe_child", envs, &log);
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    /// Child of [`pipes_are_not_regular_tables`].
    #[cfg(unix)]
    #[test]
    fn unread_pipe_child() {
        let Some(fifo) = std::env::var_os("WINNOWR_IO_UNREAD_FIFO") else { return };
        let fifo = Path::new(&fifo);
        // A producer's open waits for a reader's: if either helper opened the FIFO (even without
        // blocking), the producer would be released.
        let producer = {
            let fifo = fifo.to_path_buf();
            std::thread::spawn(move || std::fs::OpenOptions::new().write(true).open(fifo))
        };
        // Let the producer reach its blocking open first, so an open by a helper releases it.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(!producer.is_finished());
        assert!(read_regular(fifo).unwrap().is_none());
        assert!(!lacks_bgzf_eof(fifo).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(!producer.is_finished(), "a helper opened the FIFO");
        // Release the producer.
        let _reader = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(fifo)
        }
        .unwrap();
        producer.join().unwrap().unwrap();
        crate::testing::child_body_ran();
    }

    /// A path replaced by a FIFO after classification fails closed. Run in a child, so a
    /// regression that waits for a writer fails, not hangs.
    #[cfg(unix)]
    #[test]
    fn a_path_replaced_by_a_pipe_before_opening_is_not_a_regular_table() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.tsv");
        std::fs::write(&path, b"a\n").unwrap();
        let log = dir.path().join("child.log");
        let mut processes = crate::testing::TestProcesses::default();
        let envs = [("WINNOWR_IO_RACED_PATH", &path)];
        processes.spawn_test("io::tests::raced_regular_table_child", envs, &log);
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    /// Child of [`a_path_replaced_by_a_pipe_before_opening_is_not_a_regular_table`].
    #[cfg(unix)]
    #[test]
    fn raced_regular_table_child() {
        let Some(path) = std::env::var_os("WINNOWR_IO_RACED_PATH") else { return };
        let path = Path::new(&path);
        let replace = || {
            std::fs::remove_file(path).unwrap();
            assert!(std::process::Command::new("mkfifo").arg(path).status().unwrap().success());
        };
        let error = open_regular_table_with(path, replace)
            .err()
            .expect("a regular table replaced by a FIFO must fail closed");
        assert!(error.to_string().contains("changed type while being opened"), "{error:#}");
        crate::testing::child_body_ran();
    }

    /// Bytes come from the opened handle, even once the path names a pipe. Run in a child, so a
    /// regression that reads the path again (and waits for a writer) fails, not hangs.
    #[cfg(unix)]
    #[test]
    fn an_opened_regular_table_is_read_after_its_path_is_replaced_by_a_pipe() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.tsv.gz");
        std::fs::write(&path, [bgzf_block(b"a\n"), BGZF_EOF.to_vec()].concat()).unwrap();
        let log = dir.path().join("child.log");
        let mut processes = crate::testing::TestProcesses::default();
        let envs = [("WINNOWR_IO_OPENED_PATH", &path)];
        processes.spawn_test("io::tests::opened_regular_table_child", envs, &log);
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    /// Child of [`an_opened_regular_table_is_read_after_its_path_is_replaced_by_a_pipe`].
    #[cfg(unix)]
    #[test]
    fn opened_regular_table_child() {
        let Some(path) = std::env::var_os("WINNOWR_IO_OPENED_PATH") else { return };
        let path = Path::new(&path);
        let mut reader = open_regular_table(path).unwrap().unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(std::process::Command::new("mkfifo").arg(path).status().unwrap().success());
        let mut text = String::new();
        reader.read_to_string(&mut text).unwrap();
        assert_eq!(text, "a\n");
        crate::testing::child_body_ran();
    }

    #[test]
    fn appends_the_blocks_of_a_bgzf_file() {
        let dir = TempDir::new().unwrap();
        let (header, body) = (bgzf_block(b"#h\n"), bgzf_block(b"a\n"));
        let path = dir.path().join("shard.gz");
        std::fs::write(&path, [header.clone(), body.clone(), BGZF_EOF.to_vec()].concat()).unwrap();
        let mut out = Vec::new();
        append_bgzf_blocks(&mut out, &path, &header).unwrap();
        assert_eq!(out, body);
        let message = append_bgzf_blocks(&mut Vec::new(), &path, &body).unwrap_err().to_string();
        assert!(message.ends_with("does not start with the expected BGZF blocks"), "{message}");
        std::fs::write(&path, [header.clone(), body].concat()).unwrap();
        let message = append_bgzf_blocks(&mut Vec::new(), &path, &header).unwrap_err().to_string();
        assert!(message.ends_with("is not a complete BGZF file"), "{message}");
        std::fs::write(&path, &BGZF_EOF[..4]).unwrap();
        let message = append_bgzf_blocks(&mut Vec::new(), &path, &[]).unwrap_err().to_string();
        assert!(message.ends_with("is not a complete BGZF file"), "{message}");
        // The prefix may not overlap the end-of-file block.
        std::fs::write(&path, BGZF_EOF).unwrap();
        let message =
            append_bgzf_blocks(&mut Vec::new(), &path, &BGZF_EOF).unwrap_err().to_string();
        assert!(message.ends_with("does not start with the expected BGZF blocks"), "{message}");
    }

    #[test]
    fn detects_a_missing_bgzf_eof_block() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("in.bed.gz");
        std::fs::write(&path, [bgzf_block(b"a\n"), BGZF_EOF.to_vec()].concat()).unwrap();
        assert!(!lacks_bgzf_eof(&path).unwrap());
        std::fs::write(&path, bgzf_block(b"a\n")).unwrap();
        assert!(lacks_bgzf_eof(&path).unwrap());
        std::fs::write(&path, b"").unwrap();
        assert!(lacks_bgzf_eof(&path).unwrap());
        assert!(!lacks_bgzf_eof(&dir.path().join("missing.gz")).unwrap());
    }

    #[test]
    fn eof_check_still_skips_a_file_that_cannot_be_opened() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("removed.gz");
        std::fs::write(&path, BGZF_EOF).unwrap();
        assert!(!lacks_bgzf_eof_with(&path, || std::fs::remove_file(&path).unwrap()).unwrap());
    }

    /// A failed EOF inspection must reach the caller before a second open for rows. Both cases
    /// run under a deadline: no producer to wait for, or a one-shot producer that may be released.
    #[cfg(unix)]
    #[rstest::rstest]
    fn review_bgzf_eof_check_does_not_wait_for_a_replaced_fifo(
        #[values(false, true)] producer: bool,
    ) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("table.gz");
        std::fs::write(&path, BGZF_EOF).unwrap();
        let mut processes = crate::testing::TestProcesses::default();
        let envs = [
            ("WINNOWR_IO_EOF_RACE", path.into_os_string()),
            ("WINNOWR_IO_EOF_PRODUCER", producer.to_string().into()),
        ];
        processes.spawn_test(
            "io::tests::review_bgzf_replaced_fifo_child",
            envs,
            &dir.path().join("child.log"),
        );
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    #[cfg(unix)]
    #[test]
    fn review_bgzf_replaced_fifo_child() {
        let Some(path) = std::env::var_os("WINNOWR_IO_EOF_RACE") else { return };
        let path = Path::new(&path);
        let mut processes = crate::testing::TestProcesses::default();
        let error = open_table_with(
            path,
            || {
                std::fs::remove_file(path).unwrap();
                assert!(std::process::Command::new("mkfifo").arg(path).status().unwrap().success());
                if std::env::var("WINNOWR_IO_EOF_PRODUCER").unwrap() == "true" {
                    // The producer bounds its own open (SIGALRM ends it), so it cannot outlive this
                    // child if the outer deadline kills the child before dropping `processes`.
                    let open = "alarm 10; open my $fifo, '>', $ARGV[0] or die $!";
                    processes
                        .spawn(std::process::Command::new("perl").args(["-e", open]).arg(path));
                }
            },
            || {},
        )
        .err()
        .expect("a failed EOF check must not reopen the replaced FIFO for rows");
        assert!(error.to_string().contains("changed type while being opened"), "{error:#}");
        crate::testing::child_body_ran();
    }

    #[test]
    fn missing_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let message = format!("{:#}", open_text(dir.path().join("no.bed")).err().unwrap());
        assert!(message.starts_with("failed to open") && message.contains("no.bed"), "{message}");
    }
}
