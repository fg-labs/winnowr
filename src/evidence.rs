//! Read-evidence tables: the tab-separated tables `winnowr strlen` and `winnowr pileup` write,
//! each starting with a line that records the settings its evidence was measured with and, when
//! known, a line that records how the reads were aligned (see
//! [`crate::alignment_provenance`]).

use std::fmt;
use std::hash::BuildHasher;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};

use crate::alignment_provenance::{
    AlignmentProvenance, AlignmentTags, LINE_PREFIX, LazySource, resolve_reads,
};
use crate::io::{open_regular_table, open_table};

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
    /// How the reads were aligned, as the tables record it (`None` if none does).
    pub alignment: Option<AlignmentProvenance>,
}

impl<E: Evidence> EvidenceTables<E> {
    /// Tables of rows measured with `options` in memory, of reads whose alignment is not known
    /// here.
    #[must_use]
    pub fn new(options: E, rows: Vec<E::Row>) -> Self {
        EvidenceTables { options, rows, alignment: None }
    }
}

/// The lines of a table before its rows: what they record, and the column header.
struct Prelude<S> {
    /// The settings, parsed as soon as their declaration is read.
    settings: Option<S>,
    /// The `##alignment=` line's provenance.
    alignment: Option<AlignmentProvenance>,
    /// The column header (empty at the end of the file).
    header: String,
    /// The column header's line number.
    line_number: usize,
}

/// Reads the `##` lines of a table (`path`, from `reader`), up to and including the column header:
/// the settings line (of `settings_prefix`, if given), validated immediately by `parse_settings`,
/// the `##alignment=` line, and other `##` lines, which are skipped. With no settings prefix,
/// settings are not parsed. Without `read_alignment`, the `##alignment=` line is skipped before
/// its payload is decoded, so it need not be UTF-8.
///
/// # Errors
///
/// Returns an error if a line cannot be read or is not UTF-8 (other than a skipped `##alignment=`
/// line), settings are malformed or repeated, or, when `read_alignment`, the `##alignment=` line
/// is repeated or not a JSON object of strings with unique keys.
fn read_prelude<S>(
    path: &Path,
    reader: &mut impl std::io::BufRead,
    settings_prefix: Option<&str>,
    read_alignment: bool,
    mut parse_settings: impl FnMut(&str, usize) -> Result<S>,
) -> Result<Prelude<S>> {
    let mut alignment_seen = false;
    let mut prelude =
        Prelude { settings: None, alignment: None, header: String::new(), line_number: 0 };
    let mut bytes = Vec::new();
    loop {
        prelude.line_number += 1;
        bytes.clear();
        reader
            .read_until(b'\n', &mut bytes)
            .with_context(|| format!("failed to read {}", path.display()))?;
        // As `BufRead::lines` does, drop the line ending (`\n` or `\r\n`).
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
        }
        // Without provenance (`--alignment-tags none`, legacy scoring), the line is not read.
        if !read_alignment && bytes.starts_with(LINE_PREFIX.as_bytes()) {
            continue;
        }
        let line = String::from_utf8(std::mem::take(&mut bytes)).with_context(|| {
            format!("failed to read {}:{}", path.display(), prelude.line_number)
        })?;
        let trimmed = line.trim_end_matches('\r');
        if let Some(settings) = settings_prefix.and_then(|prefix| trimmed.strip_prefix(prefix)) {
            if prelude.settings.is_some() {
                bail!(
                    "{}:{} has a second {} declaration",
                    path.display(),
                    prelude.line_number,
                    settings_prefix.expect("matched settings prefix")
                );
            }
            prelude.settings = Some(parse_settings(settings, prelude.line_number)?);
        } else if let Some(json) = trimmed.strip_prefix(LINE_PREFIX) {
            if alignment_seen {
                bail!(
                    "{}:{} has a second {LINE_PREFIX} declaration",
                    path.display(),
                    prelude.line_number
                );
            }
            alignment_seen = true;
            prelude.alignment = Some(parse_alignment(path, json)?);
        } else if !line.starts_with("##") {
            prelude.header = line;
            return Ok(prelude);
        }
    }
}

/// Reads and concatenates evidence tables (e.g. one per shard), keeping the first row of each
/// `(chrom, pos, ref, alt)`.
///
/// # Errors
///
/// Returns an error if no table is given, or a table cannot be read, is truncated (BGZF without
/// the end-of-file block), lacks the [`Evidence::COLUMNS`] header, has malformed settings, no
/// settings line (unless [`Evidence::UNRECORDED`] gives them), a malformed `##alignment=` line or
/// row, or the tables were measured with different settings or record alignment provenance that
/// differs (see [`crate::alignment_provenance::agreed`]).
#[cfg(test)]
pub fn read_tables<E: Evidence>(paths: &[impl AsRef<Path>]) -> Result<EvidenceTables<E>> {
    read_tables_with_tags(paths, None)
}

/// Reads evidence rows, applying tag overrides to each file before shard agreement.
/// `none` skips provenance, but still validates settings, column headers and rows.
///
/// # Errors
///
/// Returns an error for malformed settings, headers, provenance or rows, or shard disagreement.
pub fn read_tables_with_tags<E: Evidence>(
    paths: &[impl AsRef<Path>],
    tags: Option<&AlignmentTags>,
) -> Result<EvidenceTables<E>> {
    read_tables_checked(paths, tags, |_, _| Ok(()))
}

/// Reads rows and checks each file's effective provenance against its preflight snapshot.
///
/// # Errors
///
/// Returns table validation errors or an error from `check`.
pub fn read_tables_checked<E: Evidence>(
    paths: &[impl AsRef<Path>],
    tags: Option<&AlignmentTags>,
    mut check: impl FnMut(&Path, Option<&AlignmentProvenance>) -> Result<()>,
) -> Result<EvidenceTables<E>> {
    if paths.is_empty() {
        bail!("no {} tables given", E::NAME);
    }
    let mut rows = Vec::new();
    // The rows kept, by the hash of their key: the hash (so the table grows without hashing keys
    // again) and the index into `rows`, whose key is compared, so no key is copied.
    let hasher = std::hash::RandomState::new();
    let mut seen: hashbrown::HashTable<(u64, usize)> = hashbrown::HashTable::new();
    let mut options: Option<(E, &Path)> = None;
    let mut alignments = Vec::new();
    for path in paths {
        let path = path.as_ref();
        let mut reader = open_table(path)?;
        let Prelude { settings, alignment, header, line_number } = read_prelude(
            path,
            &mut reader,
            Some(E::SETTINGS_PREFIX),
            !matches!(tags, Some(AlignmentTags::None)),
            |settings, line_number| {
                settings.parse::<E>().with_context(|| {
                    format!("{}:{line_number} has malformed {} settings", path.display(), E::NAME)
                })
            },
        )?;
        if header.trim_end_matches('\r').split('\t').ne(E::COLUMNS.iter().copied()) {
            bail!("{} is not a {} table: its header is {header:?}", path.display(), E::NAME);
        }
        let name = path.display().to_string();
        let effective = resolve_table_alignment(vec![(name.clone(), alignment.clone())], tags)?;
        check(path, effective.as_ref())?;
        alignments.push((name, alignment));
        let Some(table_options) = settings.or(E::UNRECORDED) else {
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
        for (index, line) in std::io::BufRead::lines(reader).enumerate() {
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
    let alignment = resolve_table_alignment(alignments, tags)?;
    Ok(EvidenceTables { options, rows, alignment })
}

/// The provenance of a table's `##alignment=` line (`json`, after the prefix).
///
/// # Errors
///
/// Returns an error naming the table if it is not a JSON object of strings.
fn parse_alignment(path: &Path, json: &str) -> Result<AlignmentProvenance> {
    AlignmentProvenance::from_json(json)
        .with_context(|| format!("{} has a malformed {LINE_PREFIX} line", path.display()))
}

/// The alignment provenance evidence tables record (each table's `##alignment=` line, before its
/// column header), which tables given together must share (see [`crate::alignment_provenance::agreed`]); `None` if none
/// records one. Only the lines before the column header are read, and only of regular files: a
/// table that is not one (e.g. a pipe, which reading would consume) is skipped, with a warning.
///
/// # Errors
///
/// Returns an error if a table cannot be read, its line is not a JSON object of strings, or two
/// tables record different provenance.
#[cfg(test)]
pub fn read_alignment(paths: &[impl AsRef<Path>]) -> Result<Option<AlignmentProvenance>> {
    read_alignment_with_tags(paths, None)
}

/// What scoring preflight observed of one evidence table's alignment provenance (see
/// [`observe_alignment_with_tags`]).
#[derive(Debug, PartialEq)]
pub enum AlignmentObservation {
    /// Nothing was read: the table is not a regular file (e.g. a pipe, read only once, with its
    /// rows), or provenance is disabled (`--alignment-tags none`).
    Skipped,
    /// The effective provenance (after tag overrides) of a regular file, read from the same
    /// opened handle that showed it is one; `None` if it records none.
    Regular(Option<AlignmentProvenance>),
}

/// Observes the alignment provenance a table records, before its column header, applying tag
/// overrides. Only a regular file is read (see [`crate::io::open_regular_table`]); inputs already
/// known to be nonregular are skipped, with a warning, without being opened. If an initially
/// regular path opens as nonregular, inspection fails immediately. Opening a raced FIFO may still
/// release its producer, but the error prevents a later row read from waiting for that producer.
///
/// # Errors
///
/// Returns an error if the table cannot be read, changes to a nonregular type while being opened,
/// or its provenance or tags are malformed.
pub fn observe_alignment_with_tags(
    path: &Path,
    tags: Option<&AlignmentTags>,
) -> Result<AlignmentObservation> {
    observe_alignment_with(path, tags, open_regular_table)
}

/// [`observe_alignment_with_tags`], opening the table with `open` (as
/// [`crate::io::open_regular_table`] does), so tests can act on the path once it is opened.
fn observe_alignment_with<F>(
    path: &Path,
    tags: Option<&AlignmentTags>,
    open: F,
) -> Result<AlignmentObservation>
where
    F: FnOnce(&Path) -> Result<Option<Box<dyn std::io::BufRead>>>,
{
    if matches!(tags, Some(AlignmentTags::None)) {
        return Ok(AlignmentObservation::Skipped);
    }
    let Some(mut reader) = open(path)? else {
        log::warn!(
            "{} is not a regular file, so the alignment provenance it records is not read \
             before scoring",
            path.display()
        );
        return Ok(AlignmentObservation::Skipped);
    };
    let prelude = read_prelude(path, &mut reader, None, true, |_, _| Ok(()))?;
    let sources = vec![(path.display().to_string(), prelude.alignment)];
    Ok(AlignmentObservation::Regular(resolve_table_alignment(sources, tags)?))
}

/// The alignment provenance the regular files among `paths` record, as
/// [`observe_alignment_with_tags`] observes each, which must agree once `tags` are applied to
/// each. With `none`, no files are opened.
///
/// # Errors
///
/// Returns an error if a table cannot be read, its provenance or the tags are malformed, or the
/// tables record different provenance.
#[cfg(test)]
pub fn read_alignment_with_tags(
    paths: &[impl AsRef<Path>],
    tags: Option<&AlignmentTags>,
) -> Result<Option<AlignmentProvenance>> {
    if matches!(tags, Some(AlignmentTags::None)) {
        return Ok(None);
    }
    let mut sources = Vec::new();
    for path in paths {
        let path = path.as_ref();
        if let AlignmentObservation::Regular(alignment) = observe_alignment_with_tags(path, None)? {
            sources.push((path.display().to_string(), alignment));
        }
    }
    resolve_table_alignment(sources, tags)
}

/// Applies overrides before any sources are merged, so a repaired shard cannot fail prematurely.
fn resolve_table_alignment(
    sources: Vec<(String, Option<AlignmentProvenance>)>,
    tags: Option<&AlignmentTags>,
) -> Result<Option<AlignmentProvenance>> {
    let sources: Vec<(String, LazySource<'_>)> = sources
        .into_iter()
        .map(|(name, provenance)| (name, Box::new(|| Ok(provenance)) as LazySource<'_>))
        .collect();
    Ok(resolve_reads(tags, None, sources)?.provenance)
}

/// Writes lines of evidence measured with `options` as a tab-separated table (`.gz` or `.bgz` is
/// BGZF-compressed): a [`Evidence::SETTINGS_PREFIX`] line with the settings, an `##alignment=`
/// line with the reads' provenance when known, the header line, and the lines. Returns the
/// number of lines written.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_table<E: Evidence>(
    path: &Path,
    options: E,
    alignment: Option<&AlignmentProvenance>,
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
    if let Some(alignment) = alignment {
        writeln!(out, "{LINE_PREFIX}{}", alignment.to_json())?;
    }
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
    if compressed && crate::io::lacks_bgzf_eof(path)? {
        bail!("failed to finish writing {}", path.display());
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pileup::PileupOptions;

    #[test]
    fn malformed_settings_fail_before_reading_the_next_line() {
        struct FirstLine<'a>(&'a [u8]);

        impl std::io::Read for FirstLine<'_> {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                panic!("use buffered reads");
            }
        }

        impl std::io::BufRead for FirstLine<'_> {
            fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
                assert!(!self.0.is_empty(), "read after malformed settings would stall a FIFO");
                Ok(self.0)
            }

            fn consume(&mut self, amount: usize) {
                self.0 = &self.0[amount..];
            }
        }

        fn check<E: Evidence>() {
            let line = format!("{}not-settings\n", E::SETTINGS_PREFIX);
            for read_alignment in [true, false] {
                let mut reader = FirstLine(line.as_bytes());
                assert!(
                    read_prelude(
                        Path::new("stalled.tsv"),
                        &mut reader,
                        Some(E::SETTINGS_PREFIX),
                        read_alignment,
                        |settings, _| settings.parse::<E>(),
                    )
                    .is_err()
                );
            }
        }
        check::<PileupOptions>();
        check::<crate::strlen::StrlenOptions>();
    }

    #[test]
    fn provenance_observation_does_not_validate_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("malformed-settings.tsv");
        let expected = crate::testing::bwa_provenance("primary");
        std::fs::write(
            &path,
            format!(
                "{}not-settings\n{}also-invalid\n{LINE_PREFIX}{}\nnot-a-header\nnot-a-row\n",
                PileupOptions::SETTINGS_PREFIX,
                PileupOptions::SETTINGS_PREFIX,
                expected.to_json()
            ),
        )
        .unwrap();
        assert_eq!(
            observe_alignment_with_tags(&path, None).unwrap(),
            AlignmentObservation::Regular(Some(expected))
        );
    }

    /// Run in a child, so a regression that reads the path again (and waits for a writer) fails,
    /// not hangs.
    #[cfg(unix)]
    #[test]
    fn regular_observation_survives_path_replacement_by_a_fifo() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("regular.tsv");
        write_table::<PileupOptions>(
            &path,
            PileupOptions::default(),
            Some(&crate::testing::bwa_provenance("primary")),
            std::iter::empty(),
        )
        .unwrap();
        let log = dir.path().join("child.log");
        let mut processes = crate::testing::TestProcesses::default();
        let envs = [("WINNOWR_EVIDENCE_REPLACED_PATH", &path)];
        processes.spawn_test("evidence::tests::replaced_observation_child", envs, &log);
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    /// Child of [`regular_observation_survives_path_replacement_by_a_fifo`].
    #[cfg(unix)]
    #[test]
    fn replaced_observation_child() {
        let Some(path) = std::env::var_os("WINNOWR_EVIDENCE_REPLACED_PATH") else { return };
        let expected = crate::testing::bwa_provenance("primary");
        let observation = observe_alignment_with(Path::new(&path), None, |path| {
            let reader = open_regular_table(path)?;
            std::fs::remove_file(path)?;
            assert!(std::process::Command::new("mkfifo").arg(path).status()?.success());
            Ok(reader)
        })
        .unwrap();
        assert_eq!(observation, AlignmentObservation::Regular(Some(expected)));
        crate::testing::child_body_ran();
    }

    #[test]
    fn unknown_regular_observation_is_not_a_skipped_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unknown.tsv");
        std::fs::write(&path, "header\n").unwrap();
        assert_eq!(
            observe_alignment_with_tags(&path, None).unwrap(),
            AlignmentObservation::Regular(None)
        );
        assert_eq!(
            observe_alignment_with_tags(&path, Some(&AlignmentTags::None)).unwrap(),
            AlignmentObservation::Skipped
        );
    }

    #[test]
    fn none_skips_provenance_but_validates_required_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("required.tsv");
        let tags = Some(&AlignmentTags::None);
        let write = |settings: &str, rows: &str| {
            std::fs::write(
                &path,
                format!(
                    "{}{settings}\n{LINE_PREFIX}not-json\n{}\n{rows}",
                    PileupOptions::SETTINGS_PREFIX,
                    PileupOptions::COLUMNS.join("\t")
                ),
            )
            .unwrap()
        };
        write("default", "");
        let table = read_tables_with_tags::<PileupOptions>(&[&path], tags).unwrap();
        assert!(table.alignment.is_none());
        write("not-settings", "");
        assert!(read_tables_with_tags::<PileupOptions>(&[&path], tags).is_err());
        write("default", "malformed-row\n");
        let error = read_tables_with_tags::<PileupOptions>(&[&path], tags).unwrap_err();
        assert!(format!("{error:#}").contains(&format!("{}:4", path.display())));
    }

    #[test]
    fn none_skips_alignment_lines_that_are_not_utf8() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-utf8.tsv");
        let mut bytes =
            format!("{}default\r\n{LINE_PREFIX}", PileupOptions::SETTINGS_PREFIX).into_bytes();
        bytes.extend_from_slice(b"{\"aligner\":\"\xff\"}\r\n");
        bytes.extend_from_slice(format!("{}\r\n", PileupOptions::COLUMNS.join("\t")).as_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let tables =
            read_tables_with_tags::<PileupOptions>(&[&path], Some(&AlignmentTags::None)).unwrap();
        assert!(tables.alignment.is_none());
        assert_eq!(tables.options, PileupOptions::default());
        assert!(tables.rows.is_empty());
        // With provenance, the line must still be valid UTF-8.
        let at_line = |line: usize| format!("failed to read {}:{line}", path.display());
        for error in [
            read_tables::<PileupOptions>(&[&path]).unwrap_err(),
            observe_alignment_with_tags(&path, None).unwrap_err(),
        ] {
            let error = format!("{error:#}");
            assert!(error.contains(&at_line(2)), "{error}");
        }

        // Other prelude lines are still decoded without provenance.
        let mut bytes = PileupOptions::SETTINGS_PREFIX.as_bytes().to_vec();
        bytes.extend_from_slice(b"\xff\n");
        bytes.extend_from_slice(format!("{}\n", PileupOptions::COLUMNS.join("\t")).as_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let error = read_tables_with_tags::<PileupOptions>(&[&path], Some(&AlignmentTags::None))
            .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains(&at_line(1)), "{error}");
    }

    #[test]
    fn review_repeated_settings_are_rejected_even_without_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("repeated.tsv");
        for first in ["not-settings", "default"] {
            std::fs::write(
                &path,
                format!(
                    "{}{}\n{}default\n{}\n",
                    PileupOptions::SETTINGS_PREFIX,
                    first,
                    PileupOptions::SETTINGS_PREFIX,
                    PileupOptions::COLUMNS.join("\t")
                ),
            )
            .unwrap();
            for tags in [None, Some(&AlignmentTags::None)] {
                let error = read_tables_with_tags::<PileupOptions>(&[&path], tags).unwrap_err();
                let message = format!("{error:#}");
                let expected =
                    if first == "not-settings" { ":1 has malformed" } else { ":2 has a second" };
                assert!(message.contains(expected), "{message}");
            }
        }
    }

    #[test]
    fn duplicate_alignment_json_keys_are_rejected_unless_provenance_is_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duplicate-keys.tsv");
        for json in [
            r#"{"aligner":"bwa","aligner":"other"}"#,
            r#"{"aligner":"bwa","aligner":"bwa"}"#,
            r#"{"aligner":"bwa","\u0061ligner":"bwa"}"#,
        ] {
            std::fs::write(
                &path,
                format!(
                    "{}default\n{LINE_PREFIX}{json}\n{}\n",
                    PileupOptions::SETTINGS_PREFIX,
                    PileupOptions::COLUMNS.join("\t")
                ),
            )
            .unwrap();
            for result in [
                read_tables::<PileupOptions>(&[&path]).map(|table| table.alignment),
                read_alignment(&[&path]),
            ] {
                let error = format!("{:#}", result.unwrap_err());
                assert!(error.contains("duplicate"), "{error}");
                assert!(error.contains(&path.display().to_string()), "{error}");
            }
            let tables =
                read_tables_with_tags::<PileupOptions>(&[&path], Some(&AlignmentTags::None))
                    .unwrap();
            assert!(tables.alignment.is_none());
            assert_eq!(tables.options, PileupOptions::default());
        }
    }

    #[test]
    fn duplicate_alignment_declarations_are_rejected_with_path_and_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duplicate.tsv");
        for second in [r#"{"aligner":"bwa"}"#, r#"{"aligner":"other"}"#] {
            std::fs::write(
                &path,
                format!(
                    "{}default\n{LINE_PREFIX}{{\"aligner\":\"bwa\"}}\n{LINE_PREFIX}{second}\n{}\n",
                    PileupOptions::SETTINGS_PREFIX,
                    PileupOptions::COLUMNS.join("\t")
                ),
            )
            .unwrap();
            for result in [
                read_alignment(&[&path]),
                read_tables::<PileupOptions>(&[&path]).map(|t| t.alignment),
            ] {
                let error = format!("{:#}", result.unwrap_err());
                assert!(error.contains(&format!("{}:3", path.display())), "{error}");
                assert!(error.contains("alignment"), "{error}");
            }
            // `none` processes no provenance, as legacy scoring does, but still reads the table.
            let tables =
                read_tables_with_tags::<PileupOptions>(&[&path], Some(&AlignmentTags::None))
                    .unwrap();
            assert!(tables.alignment.is_none());
            assert_eq!(tables.options, PileupOptions::default());
        }
    }
}
