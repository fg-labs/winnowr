//! Read-evidence tables: the tab-separated tables `winnowr strlen` and `winnowr pileup` write,
//! each starting with a line that records the settings its evidence was measured with.

use std::fmt;
use std::hash::BuildHasher;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};

use crate::io::open_table;

/// A kind of read evidence, as the settings it is measured with (e.g.
/// [`crate::strlen::StrlenOptions`]), which are written as text (see [`fmt::Display`] and
/// [`FromStr`]) in its tables, feature tables and models.
pub trait Evidence:
    Copy + Eq + fmt::Debug + fmt::Display + FromStr<Err = anyhow::Error> + Send + Sync
{
    /// One row of the evidence's tables.
    type Row: fmt::Debug + Clone + PartialEq + Send + Sync;

    /// The evidence's name in messages, and its `winnowr` subcommand: `strlen` or `pileup`.
    const NAME: &'static str;
    /// What its settings are called in messages (e.g. `read settings`).
    const SETTINGS: &'static str;
    /// The option of the subcommand that sets the settings (e.g. `--reads`).
    const OPTION: &'static str;
    /// The columns of its tables, in file order.
    const COLUMNS: &'static [&'static str];
    /// The prefix of the line recording a table's settings, before the column header.
    const SETTINGS_PREFIX: &'static str;
    /// The settings of a table without a settings line (`None`: such a table is an error).
    const UNRECORDED: Option<Self>;

    /// Parses one table line.
    ///
    /// # Errors
    ///
    /// Returns an error if the line is malformed.
    fn parse_row(line: &str) -> Result<Self::Row>;

    /// A row's record: `(chrom, pos, ref, alt)`, with `alt` comma-joined.
    fn row_key(row: &Self::Row) -> (&str, u64, &str, &str);
}

/// The rows of one or more evidence tables, with the settings they were measured with.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceTables<E: Evidence> {
    /// The settings the tables were measured with.
    pub options: E,
    /// The rows.
    pub rows: Vec<E::Row>,
}

/// Reads and concatenates evidence tables (e.g. one per shard), keeping the first row of each
/// `(chrom, pos, ref, alt)`.
///
/// # Errors
///
/// Returns an error if no table is given, or a table cannot be read, is truncated (BGZF without
/// the end-of-file block), lacks the [`Evidence::COLUMNS`] header, has malformed settings, no
/// settings line (unless [`Evidence::UNRECORDED`] gives them) or a malformed row, or the tables
/// were measured with different settings.
pub fn read_tables<E: Evidence>(paths: &[impl AsRef<Path>]) -> Result<EvidenceTables<E>> {
    if paths.is_empty() {
        bail!("no {} tables given", E::NAME);
    }
    let mut rows = Vec::new();
    // The rows kept, by the hash of their key: the hash (so the table grows without hashing keys
    // again) and the index into `rows`, whose key is compared, so no key is copied.
    let hasher = std::hash::RandomState::new();
    let mut seen: hashbrown::HashTable<(u64, usize)> = hashbrown::HashTable::new();
    let mut options: Option<(E, &Path)> = None;
    for path in paths {
        let path = path.as_ref();
        let mut lines = std::io::BufRead::lines(open_table(path)?);
        let mut line_number = 0;
        let mut table_options = None;
        let header = loop {
            line_number += 1;
            let line = lines
                .next()
                .transpose()
                .with_context(|| format!("failed to read {}", path.display()))?
                .unwrap_or_default();
            if let Some(settings) = line.trim_end_matches('\r').strip_prefix(E::SETTINGS_PREFIX) {
                table_options = Some(settings.parse::<E>().with_context(|| {
                    format!("{}:{line_number} has malformed {} settings", path.display(), E::NAME)
                })?);
            } else if !line.starts_with("##") {
                break line;
            }
        };
        if header.trim_end_matches('\r').split('\t').ne(E::COLUMNS.iter().copied()) {
            bail!("{} is not a {} table: its header is {header:?}", path.display(), E::NAME);
        }
        let Some(table_options) = table_options.or(E::UNRECORDED) else {
            bail!("{} has no {} line", path.display(), E::SETTINGS_PREFIX);
        };
        match options {
            None => options = Some((table_options, path)),
            Some((first, first_path)) if first != table_options => bail!(
                "the {} tables were measured with different {}: {} with {first}, {} with \
                 {table_options}",
                E::NAME,
                E::SETTINGS,
                first_path.display(),
                path.display()
            ),
            Some(_) => {}
        }
        for (index, line) in lines.enumerate() {
            let line_number = line_number + index + 1;
            let line =
                line.with_context(|| format!("failed to read {}:{line_number}", path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            let row =
                E::parse_row(&line).with_context(|| format!("{}:{line_number}", path.display()))?;
            let key = E::row_key(&row);
            let hash = hasher.hash_one(key);
            let same = |&(h, i): &(u64, usize)| h == hash && E::row_key(&rows[i]) == key;
            if let hashbrown::hash_table::Entry::Vacant(entry) = seen.entry(hash, same, |&(h, _)| h)
            {
                entry.insert((hash, rows.len()));
                rows.push(row);
            }
        }
    }
    let (options, _) = options.expect("at least one table was read");
    Ok(EvidenceTables { options, rows })
}

/// Writes lines of evidence measured with `options` as a tab-separated table (`.gz` or `.bgz` is
/// BGZF-compressed): a [`Evidence::SETTINGS_PREFIX`] line with the settings, the header line, and
/// the lines. Returns the number of lines written.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_table<E: Evidence>(
    path: &Path,
    options: E,
    lines: impl IntoIterator<Item = String>,
) -> Result<usize> {
    use std::io::Write as _;
    let compressed = crate::common::is_bgzf_path(path);
    let mut out: Box<dyn std::io::Write> = if compressed {
        Box::new(
            rust_htslib::bgzf::Writer::from_path(path)
                .with_context(|| format!("failed to create {}", path.display()))?,
        )
    } else {
        Box::new(std::io::BufWriter::new(
            std::fs::File::create(path)
                .with_context(|| format!("failed to create {}", path.display()))?,
        ))
    };
    writeln!(out, "{}{options}", E::SETTINGS_PREFIX)?;
    writeln!(out, "{}", E::COLUMNS.join("\t"))?;
    let mut n = 0;
    for line in lines {
        writeln!(out, "{line}")?;
        n += 1;
    }
    out.flush().with_context(|| format!("failed to write {}", path.display()))?;
    // The BGZF writer closes the file when dropped and cannot report a failure, so check that
    // the end-of-file block was written.
    drop(out);
    if compressed && crate::io::lacks_bgzf_eof(path) {
        bail!("failed to finish writing {}", path.display());
    }
    Ok(n)
}
