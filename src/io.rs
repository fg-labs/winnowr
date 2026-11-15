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

/// Opens a table that winnowr wrote, as [`open_text`] does, but rejects a BGZF file without the
/// end-of-file block: winnowr always writes it, so the table was truncated (e.g. a killed run).
///
/// # Errors
///
/// Returns an error if the file cannot be opened, or is BGZF-compressed without the end-of-file
/// block.
pub fn open_table<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>> {
    let path = path.as_ref();
    // Only a regular file can be checked: reading the start of a pipe (e.g. process substitution)
    // would consume it.
    let regular = std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file());
    if regular
        && rust_htslib::bgzf::is_bgzip(path)
            .with_context(|| format!("failed to open {}", path.display()))?
        && lacks_bgzf_eof(path)
    {
        anyhow::bail!(
            "{} is truncated (no BGZF end-of-file block): write it again",
            path.display()
        );
    }
    open_text(path)
}

/// True if a regular file does not end with the BGZF end-of-file block. Files that cannot be
/// checked (e.g. pipes) are assumed complete.
pub fn lacks_bgzf_eof(path: &Path) -> bool {
    let check = || -> std::io::Result<bool> {
        let mut file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Ok(false);
        }
        if metadata.len() < BGZF_EOF.len() as u64 {
            return Ok(true);
        }
        file.seek(SeekFrom::End(-(BGZF_EOF.len() as i64)))?;
        let mut tail = [0u8; BGZF_EOF.len()];
        file.read_exact(&mut tail)?;
        Ok(tail != BGZF_EOF)
    };
    check().unwrap_or(false)
}

/// Writes a tab-separated file with a header line.
pub fn write_tsv<I: Iterator<Item = Vec<String>>>(
    path: &Path,
    header: &[&str],
    rows: I,
) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut out = std::io::BufWriter::new(file);
    writeln!(out, "{}", header.join("\t"))?;
    for row in rows {
        debug_assert_eq!(row.len(), header.len());
        writeln!(out, "{}", row.join("\t"))?;
    }
    out.flush().with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use tempfile::TempDir;

    use super::*;

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

    /// A table read from a pipe (e.g. `--strlen <(...)`) is read once, from its start.
    #[cfg(unix)]
    #[test]
    fn tables_are_read_from_a_pipe() {
        let dir = TempDir::new().unwrap();
        let fifo = dir.path().join("table.fifo");
        let status = std::process::Command::new("mkfifo").arg(&fifo).status().unwrap();
        assert!(status.success());
        let data = [bgzf_block(b"a\nb\n"), BGZF_EOF.to_vec()].concat();
        let writer = {
            let fifo = fifo.clone();
            std::thread::spawn(move || std::fs::write(fifo, data).unwrap())
        };
        let mut text = String::new();
        open_table(&fifo).unwrap().read_to_string(&mut text).unwrap();
        writer.join().unwrap();
        assert_eq!(text, "a\nb\n");
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
        assert!(!lacks_bgzf_eof(&path));
        std::fs::write(&path, bgzf_block(b"a\n")).unwrap();
        assert!(lacks_bgzf_eof(&path));
        std::fs::write(&path, b"").unwrap();
        assert!(lacks_bgzf_eof(&path));
        assert!(!lacks_bgzf_eof(&dir.path().join("missing.gz")));
    }

    #[test]
    fn missing_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let message = format!("{:#}", open_text(dir.path().join("no.bed")).err().unwrap());
        assert!(message.starts_with("failed to open") && message.contains("no.bed"), "{message}");
    }
}
