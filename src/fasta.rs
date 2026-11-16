//! Reading sequence from an indexed (`.fai`), uncompressed FASTA.
//!
//! winnowr reads the reference itself rather than through rust-htslib's `faidx`, whose
//! `fetch_seq` hands memory allocated by htslib to Rust's allocator to free.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// One sequence's entry in a FASTA index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaiEntry {
    /// The sequence length.
    length: u64,
    /// The byte offset of the sequence's first base.
    offset: u64,
    /// The bases per line.
    line_bases: u64,
    /// The bytes per line, including the line terminator.
    line_width: u64,
}

/// An indexed FASTA file.
#[derive(Debug)]
pub struct IndexedFasta {
    path: PathBuf,
    file: File,
    /// The FASTA's size in bytes.
    size: u64,
    entries: HashMap<String, FaiEntry>,
}

impl IndexedFasta {
    /// Opens a FASTA and reads its `<path>.fai` index.
    ///
    /// # Errors
    ///
    /// Returns an error if the FASTA or its index cannot be read, the index is malformed, or the
    /// FASTA is compressed.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut magic = [0u8; 2];
        if file.read(&mut magic).ok() == Some(2) && magic == [0x1f, 0x8b] {
            bail!("{} is compressed; winnowr needs an uncompressed, indexed FASTA", path.display());
        }
        let mut fai = path.as_os_str().to_owned();
        fai.push(".fai");
        let fai = PathBuf::from(fai);
        let reader = BufReader::new(File::open(&fai).with_context(|| {
            format!("failed to open the FASTA index {} (index with samtools faidx)", fai.display())
        })?);
        let mut entries = HashMap::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("failed to read {}", fai.display()))?;
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            let number = |i: usize| -> Option<u64> { fields.get(i)?.parse().ok() };
            let entry = (|| {
                Some(FaiEntry {
                    length: number(1)?,
                    offset: number(2)?,
                    line_bases: number(3)?,
                    line_width: number(4)?,
                })
            })();
            match entry {
                Some(entry) if entry.line_bases > 0 && entry.line_width > entry.line_bases => {
                    entries.insert(fields[0].to_string(), entry);
                }
                _ => bail!("{}:{}: malformed FASTA index line: {line:?}", fai.display(), index + 1),
            }
        }
        file.rewind()?;
        let size =
            file.metadata().with_context(|| format!("failed to read {}", path.display()))?.len();
        Ok(IndexedFasta { path: path.to_path_buf(), file, size, entries })
    }

    /// The length of a sequence, or `None` if the index does not have it.
    #[must_use]
    pub fn length(&self, name: &str) -> Option<u64> {
        self.entries.get(name).map(|entry| entry.length)
    }

    /// The length of a sequence, checked against the FASTA's size, or `None` if the index does
    /// not have it.
    ///
    /// # Errors
    ///
    /// Returns an error if the index places the sequence's end past the end of the file.
    pub fn checked_length(&self, name: &str) -> Result<Option<u64>> {
        let Some(entry) = self.entries.get(name) else { return Ok(None) };
        if entry.length > 0 {
            self.span(name, entry, 0, entry.length)?;
        }
        Ok(Some(entry.length))
    }

    /// The bytes of the file holding `[start, end)` of a sequence (0-based; `start < end`), as
    /// `(first, last)` (half-open).
    ///
    /// # Errors
    ///
    /// Returns an error if they are not inside the file.
    fn span(&self, name: &str, entry: &FaiEntry, start: u64, end: u64) -> Result<(u64, u64)> {
        let byte = |pos: u64| {
            (pos / entry.line_bases)
                .checked_mul(entry.line_width)?
                .checked_add(entry.offset)?
                .checked_add(pos % entry.line_bases)
        };
        // A corrupt index can name a sequence far longer than the file.
        match byte(start).zip(byte(end - 1).and_then(|byte| byte.checked_add(1))) {
            Some((first, last)) if last <= self.size => Ok((first, last)),
            _ => Err(self.mismatch(name)),
        }
    }

    /// The error for an index that does not match the FASTA.
    fn mismatch(&self, name: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "the FASTA index of {} does not match its sequence {name}",
            self.path.display()
        )
    }

    /// The bases of `[start, end)` of a sequence (0-based), as stored (case preserved); `end` is
    /// clamped to the sequence length.
    ///
    /// # Errors
    ///
    /// Returns an error if the index does not have the sequence, or the file cannot be read.
    pub fn fetch(&mut self, name: &str, start: u64, end: u64) -> Result<Vec<u8>> {
        let entry = *self.entries.get(name).with_context(|| {
            format!("contig {name} is not in the FASTA {}", self.path.display())
        })?;
        let end = end.min(entry.length);
        if start >= end {
            return Ok(Vec::new());
        }
        let (first, last) = self.span(name, &entry, start, end)?;
        let mut raw = vec![0u8; usize::try_from(last - first)?];
        self.file.seek(SeekFrom::Start(first))?;
        self.file
            .read_exact(&mut raw)
            .with_context(|| format!("failed to read {name} from {}", self.path.display()))?;
        raw.retain(|&b| b != b'\n' && b != b'\r');
        if raw.len() as u64 != end - start {
            return Err(self.mismatch(name));
        }
        Ok(raw)
    }

    /// The whole sequence (see [`IndexedFasta::fetch`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the index does not have the sequence, or the file cannot be read.
    pub fn fetch_all(&mut self, name: &str) -> Result<Vec<u8>> {
        let length = self.length(name).unwrap_or(0);
        self.fetch(name, 0, length)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::testing::write_fasta;

    #[test]
    fn fetches_across_lines_and_clamps_the_end() {
        let dir = TempDir::new().unwrap();
        let path = write_fasta(dir.path(), &[("chr1", "ACGTacgtAACCGG"), ("chr2", "TTT")]);
        let mut fasta = IndexedFasta::from_path(&path).unwrap();
        assert_eq!(fasta.length("chr1"), Some(14));
        assert_eq!(fasta.fetch("chr1", 0, 4).unwrap(), b"ACGT");
        assert_eq!(fasta.fetch("chr1", 3, 9).unwrap(), b"TacgtA");
        assert_eq!(fasta.fetch("chr1", 12, 100).unwrap(), b"GG");
        assert_eq!(fasta.fetch("chr1", 20, 30).unwrap(), b"");
        assert_eq!(fasta.fetch_all("chr2").unwrap(), b"TTT");
        let message = fasta.fetch("chr9", 0, 1).unwrap_err().to_string();
        assert!(message.starts_with("contig chr9 is not in the FASTA"), "{message}");
    }

    #[test]
    fn an_index_longer_than_the_file_is_an_error_before_allocating() {
        let dir = TempDir::new().unwrap();
        let path = write_fasta(dir.path(), &[("chr1", "ACGT")]);
        let fai = dir.path().join("ref.fa.fai");
        assert!(fai.exists(), "write_fasta names its index {}", fai.display());
        let expected =
            format!("the FASTA index of {} does not match its sequence chr1", path.display());
        // Far past the file, and so long that its byte offsets overflow.
        for length in [1_000_000_000_000_000, u64::MAX] {
            std::fs::write(&fai, format!("chr1\t{length}\t6\t60\t61\n")).unwrap();
            let mut fasta = IndexedFasta::from_path(&path).unwrap();
            assert_eq!(fasta.fetch_all("chr1").unwrap_err().to_string(), expected, "{length}");
            assert_eq!(fasta.checked_length("chr1").unwrap_err().to_string(), expected);
            assert_eq!(fasta.length("chr1"), Some(length));
        }
        // A sequence that fits passes the check.
        std::fs::write(&fai, "chr1\t4\t6\t60\t61\n").unwrap();
        let fasta = IndexedFasta::from_path(&path).unwrap();
        assert_eq!(fasta.checked_length("chr1").unwrap(), Some(4));
        assert_eq!(fasta.checked_length("chr9").unwrap(), None);
    }

    #[test]
    fn rejects_compressed_or_unindexed_fastas() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("plain.fa");
        std::fs::write(&path, ">chr1\nACGT\n").unwrap();
        let message = format!("{:#}", IndexedFasta::from_path(&path).unwrap_err());
        assert!(message.contains("samtools faidx"), "{message}");
        let gz = dir.path().join("ref.fa.gz");
        std::fs::write(&gz, [0x1f, 0x8b, 0, 0]).unwrap();
        let message = IndexedFasta::from_path(&gz).unwrap_err().to_string();
        assert!(message.contains("is compressed"), "{message}");
    }
}
