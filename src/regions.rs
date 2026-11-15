//! Membership lookups against named BED region sets (e.g. GIAB stratifications).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail};

use crate::io::open_text;

/// The merged, sorted, non-overlapping, half-open (0-based) intervals of one contig.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContigIntervals {
    starts: Vec<u64>,
    ends: Vec<u64>,
}

impl ContigIntervals {
    /// Sorts and merges overlapping or adjacent non-empty `(start, end)` spans.
    fn from_spans(mut spans: Vec<(u64, u64)>) -> Self {
        spans.sort_unstable();
        let mut starts: Vec<u64> = Vec::with_capacity(spans.len());
        let mut ends: Vec<u64> = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            match ends.last_mut() {
                Some(last_end) if start <= *last_end => *last_end = (*last_end).max(end),
                _ => {
                    starts.push(start);
                    ends.push(end);
                }
            }
        }
        ContigIntervals { starts, ends }
    }

    /// True if the half-open span `[start, end)` shares at least one base with an interval.
    ///
    /// An empty span (`end <= start`) contains no base, so overlaps nothing.
    #[must_use]
    pub fn overlaps(&self, start: u64, end: u64) -> bool {
        // The last interval starting before the span ends. Merged intervals have increasing ends,
        // so it is also the one reaching furthest right among those.
        let before_end = self.starts.partition_point(|&s| s < end);
        end > start && before_end > 0 && self.ends[before_end - 1] > start
    }

    /// True if the half-open span `[start, end)` lies entirely inside one interval.
    ///
    /// An empty span (`end <= start`) contains no base, so is never inside an interval.
    #[must_use]
    pub fn contains(&self, start: u64, end: u64) -> bool {
        let at_or_before_start = self.starts.partition_point(|&s| s <= start);
        end > start && at_or_before_start > 0 && end <= self.ends[at_or_before_start - 1]
    }
}

/// A named set of merged, sorted, half-open (0-based) genomic intervals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionSet {
    name: String,
    intervals: HashMap<String, ContigIntervals>,
    /// The BED file the set was read from, if any, for error messages.
    path: Option<PathBuf>,
}

impl RegionSet {
    /// Reads a BED file into a region set. Gzipped and BGZF files are detected and decompressed.
    ///
    /// Header lines (starting with `#`, or `track`/`browser` lines that are not intervals) and
    /// blank lines are skipped, as are empty (`start == end`) intervals, which contain no base.
    /// Whitespace around the coordinates is ignored. Overlapping and adjacent intervals are
    /// merged.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not UTF-8, or a line has fewer than three
    /// tab-separated columns, an empty contig name, coordinates that are not integers,
    /// `start < 0` or `end < start`. The error names the file and line.
    pub fn from_bed<N: Into<String>, P: AsRef<Path>>(name: N, path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut reader = open_text(path)?;
        let mut by_contig: HashMap<String, Vec<(u64, u64)>> = HashMap::new();
        let mut line = String::new();
        let mut line_number = 0usize;
        loop {
            line.clear();
            line_number += 1;
            let read = reader
                .read_line(&mut line)
                .with_context(|| format!("failed to read {}:{line_number}", path.display()))?;
            if read == 0 {
                break;
            }
            let interval = parse_bed_line(&line).with_context(|| {
                let shown = line.trim_end_matches(['\r', '\n']);
                format!("{}:{line_number}: {shown:?}", path.display())
            })?;
            if let Some((contig, start, end)) = interval {
                push_span(&mut by_contig, contig, start, end);
            }
        }
        Ok(Self::from_grouped(name, by_contig, Some(path.to_path_buf())))
    }

    /// Builds a region set from `(contig, start, end)` half-open spans, in any order.
    ///
    /// Overlapping and adjacent spans are merged, and empty spans (`end <= start`) are ignored.
    /// Contigs without a non-empty span are left out.
    pub fn from_spans<N, C, I>(name: N, spans: I) -> Self
    where
        N: Into<String>,
        C: AsRef<str>,
        I: IntoIterator<Item = (C, u64, u64)>,
    {
        let mut by_contig: HashMap<String, Vec<(u64, u64)>> = HashMap::new();
        for (contig, start, end) in spans {
            if end > start {
                push_span(&mut by_contig, contig.as_ref(), start, end);
            }
        }
        Self::from_grouped(name, by_contig, None)
    }

    /// Builds a region set from non-empty spans grouped by contig.
    fn from_grouped<N: Into<String>>(
        name: N,
        by_contig: HashMap<String, Vec<(u64, u64)>>,
        path: Option<PathBuf>,
    ) -> Self {
        let intervals = by_contig
            .into_iter()
            .map(|(contig, spans)| (contig, ContigIntervals::from_spans(spans)))
            .collect();
        RegionSet { name: name.into(), intervals, path }
    }

    /// The name of the region set, which names the derived feature.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The intervals on a contig, or `None` if the contig has none.
    #[must_use]
    pub fn contig(&self, contig: &str) -> Option<&ContigIntervals> {
        self.intervals.get(contig)
    }

    /// The contigs with at least one interval, in no particular order.
    pub fn contigs(&self) -> impl Iterator<Item = &str> {
        self.intervals.keys().map(String::as_str)
    }

    /// True if the half-open span `[start, end)` on `contig` shares a base with an interval (see
    /// [`ContigIntervals::overlaps`]).
    #[must_use]
    pub fn overlaps(&self, contig: &str, start: u64, end: u64) -> bool {
        self.contig(contig).is_some_and(|intervals| intervals.overlaps(start, end))
    }

    /// True if the half-open span `[start, end)` on `contig` lies entirely inside one interval
    /// (see [`ContigIntervals::contains`]).
    #[must_use]
    pub fn contains(&self, contig: &str, start: u64, end: u64) -> bool {
        self.contig(contig).is_some_and(|intervals| intervals.contains(start, end))
    }

    /// Checks that the region set has intervals on at least one of `contigs`.
    ///
    /// A set may lack some contigs, but one sharing none of them usually means the BED names its
    /// contigs differently (e.g. `1` vs `chr1`), which would silently flag nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if none of `contigs` has an interval in the set.
    pub fn check_contigs<S: AsRef<str>>(&self, contigs: &[S]) -> Result<()> {
        if contigs.iter().any(|contig| self.intervals.contains_key(contig.as_ref())) {
            return Ok(());
        }
        let mut bed_contigs: Vec<&str> = self.contigs().collect();
        bed_contigs.sort_unstable();
        bed_contigs.truncate(5);
        let bed_contigs =
            if bed_contigs.is_empty() { "none".into() } else { bed_contigs.join(", ") };
        let source =
            self.path.as_ref().map_or_else(|| self.name.clone(), |p| p.display().to_string());
        let contigs: Vec<&str> = contigs.iter().map(AsRef::as_ref).collect();
        bail!(
            "region BED {source} has no intervals on the extracted contigs ({}); its contigs \
             include: {bed_contigs}. Its contig names must match the variant calls' (e.g. chr1, \
             not 1)",
            contigs.join(", ")
        )
    }
}

/// Appends a span to its contig's list, allocating the contig name only on first sight.
fn push_span(by_contig: &mut HashMap<String, Vec<(u64, u64)>>, contig: &str, start: u64, end: u64) {
    match by_contig.get_mut(contig) {
        Some(spans) => spans.push((start, end)),
        None => {
            by_contig.insert(contig.to_string(), vec![(start, end)]);
        }
    }
}

/// Parses a BED line into `(contig, start, end)`, or `None` for a header, blank line or empty
/// interval.
///
/// A `track` or `browser` line is a header unless it parses as an interval, so a contig named
/// `track` still works.
fn parse_bed_line(line: &str) -> Result<Option<(&str, u64, u64)>> {
    let line = line.trim_end_matches(['\r', '\n']);
    if line.trim().is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let is_header = matches!(line.split_whitespace().next(), Some("track" | "browser"));
    let mut fields = line.split('\t');
    let (contig, start, end) = match (fields.next(), fields.next(), fields.next()) {
        (Some(contig), Some(start), Some(end)) => (contig, start, end),
        _ if is_header => return Ok(None),
        _ => bail!("expected at least 3 tab-separated columns"),
    };
    let (start, end) = match (start.trim().parse::<i64>(), end.trim().parse::<i64>()) {
        (Ok(start), Ok(end)) => (start, end),
        _ if is_header => return Ok(None),
        (Err(e), _) => bail!("start {start:?} is not an integer: {e}"),
        (_, Err(e)) => bail!("end {end:?} is not an integer: {e}"),
    };
    if contig.is_empty() {
        bail!("empty contig name");
    }
    if start < 0 || end < start {
        bail!("invalid interval: start must be >= 0 and end >= start");
    }
    if start == end {
        return Ok(None);
    }
    // Both coordinates are non-negative, so the casts are lossless.
    Ok(Some((contig, start as u64, end as u64)))
}

/// The 0-based, half-open span of a record's REF allele, from its 1-based `pos` and the length of
/// its REF: at least one base, so that an empty REF still has a position. A `pos` below 1 (a
/// telomere record) starts at 0.
#[must_use]
pub(crate) fn ref_span(pos: i64, ref_len: usize) -> (u64, u64) {
    let start = u64::try_from(pos).unwrap_or(0).saturating_sub(1);
    (start, start + ref_len.max(1) as u64)
}

/// A named file, given on the command line as `NAME=PATH`, with any non-empty name and path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedPath {
    /// The name.
    pub name: String,
    /// The path.
    pub path: PathBuf,
}

impl FromStr for NamedPath {
    type Err = anyhow::Error;

    /// Parses `NAME=PATH`, splitting at the first `=`.
    fn from_str(spec: &str) -> Result<Self> {
        match spec.split_once('=') {
            Some((name, path)) if !name.is_empty() && !path.is_empty() => {
                Ok(NamedPath { name: name.to_string(), path: PathBuf::from(path) })
            }
            _ => bail!("expected NAME=PATH, got {spec:?}"),
        }
    }
}

/// A named region BED, given on the command line as `name=path`: a [`NamedPath`] whose name is
/// also a feature-name part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedBed {
    /// The name of the region set: one or more ASCII letters, digits and underscores.
    pub name: String,
    /// The path to the BED file.
    pub path: PathBuf,
}

impl FromStr for NamedBed {
    type Err = anyhow::Error;

    /// Parses `name=path`, e.g. `segdups=/refs/segdups.bed.gz`.
    fn from_str(spec: &str) -> Result<Self> {
        let valid_name = |name: &str| name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
        match spec.parse::<NamedPath>() {
            Ok(NamedPath { name, path }) if valid_name(&name) => Ok(NamedBed { name, path }),
            _ => bail!("region BEDs must be given as name=path (name: [A-Za-z0-9_]+), got: {spec}"),
        }
    }
}

impl fmt::Display for NamedBed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.path.display())
    }
}

/// Reads the region sets named on the command line, in order.
///
/// # Errors
///
/// Returns an error if a name is repeated, or a BED cannot be read.
pub fn read_region_sets(specs: &[NamedBed]) -> Result<Vec<RegionSet>> {
    let repeated = crate::common::repeated(specs.iter().map(|spec| spec.name.as_str()));
    if !repeated.is_empty() {
        bail!("region names must be unique; repeated: {}", repeated.join(", "));
    }
    specs.iter().map(|spec| RegionSet::from_bed(&spec.name, &spec.path)).collect()
}

/// Reads a hap.py `--stratification` manifest: one `NAME<TAB>PATH` line per region, where a
/// relative path is relative to the manifest's directory. Blank lines and lines starting with
/// `#` are skipped.
///
/// # Errors
///
/// Returns an error if the file cannot be read or a line does not have two non-empty
/// tab-separated fields.
pub fn read_stratification_manifest(path: &Path) -> Result<Vec<NamedPath>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let dir = path.parent().unwrap_or_else(|| Path::new(""));
    let mut regions = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split('\t');
        match (fields.next(), fields.next()) {
            (Some(name), Some(file)) if !name.is_empty() && !file.is_empty() => {
                regions.push(NamedPath { name: name.to_string(), path: dir.join(file) });
            }
            _ => bail!("{}:{}: expected NAME<TAB>PATH, got {line:?}", path.display(), i + 1),
        }
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use rstest::rstest;
    use tempfile::TempDir;

    use super::*;

    /// How a test BED file is written.
    #[derive(Debug, Clone, Copy)]
    enum Encoding {
        Plain,
        Gzip,
        /// Gzip with one member per line, as in a BGZF file.
        MultiMember,
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap()
    }

    fn write_bed(dir: &TempDir, file_name: &str, lines: &[&str], encoding: Encoding) -> PathBuf {
        let path = dir.path().join(file_name);
        let lines: Vec<String> = lines.iter().map(|line| format!("{line}\n")).collect();
        let bytes = match encoding {
            Encoding::Plain => lines.concat().into_bytes(),
            Encoding::Gzip => gzip(lines.concat().as_bytes()),
            Encoding::MultiMember => lines.iter().flat_map(|line| gzip(line.as_bytes())).collect(),
        };
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn read_bed(lines: &[&str]) -> RegionSet {
        let dir = TempDir::new().unwrap();
        RegionSet::from_bed("r", write_bed(&dir, "r.bed", lines, Encoding::Plain)).unwrap()
    }

    fn bed_error(dir: &TempDir, bytes: &[u8]) -> String {
        let path = dir.path().join("bad.bed");
        std::fs::write(&path, bytes).unwrap();
        format!("{:#}", RegionSet::from_bed("bad", path).unwrap_err())
    }

    fn spans(regions: &RegionSet, contig: &str) -> Vec<(u64, u64)> {
        let intervals = regions.contig(contig).unwrap();
        intervals.starts.iter().copied().zip(intervals.ends.iter().copied()).collect()
    }

    #[rstest]
    fn overlaps_respects_half_open_bed_coordinates(
        #[values(Encoding::Plain, Encoding::Gzip, Encoding::MultiMember)] encoding: Encoding,
    ) {
        let dir = TempDir::new().unwrap();
        let lines =
            ["track name=x", "browser position chr1", "chr1\t10\t20", "chr1\t15\t30", "chr2\t0\t1"];
        let regions = RegionSet::from_bed("r", write_bed(&dir, "r.bed", &lines, encoding)).unwrap();
        // BED [10, 30) on chr1 covers 0-based bases 10 to 29.
        let queries = [(9, 10), (10, 11), (29, 30), (30, 31), (5, 10), (28, 40)];
        let got: Vec<bool> = queries.iter().map(|&(s, e)| regions.overlaps("chr1", s, e)).collect();
        assert_eq!(got, [false, true, true, false, false, true]);
        // A deletion anchored just before the region overlaps it through its deleted bases.
        assert!(regions.overlaps("chr1", 8, 12));
        assert!(regions.overlaps("chr2", 0, 1));
        assert!(!regions.overlaps("chr2", 1, 2));
        assert!(!regions.overlaps("chr3", 0, 1));
    }

    #[test]
    fn overlaps_finds_a_long_interval_starting_well_before_the_span() {
        let regions = read_bed(&["chr1\t0\t1000", "chr1\t1100\t1200"]);
        assert!(regions.overlaps("chr1", 500, 501));
        assert!(!regions.overlaps("chr1", 1000, 1100));
    }

    #[rstest]
    #[case::inside(15, 15)]
    #[case::at_interval_start(10, 10)]
    #[case::at_interval_end(20, 20)]
    #[case::reversed(18, 12)]
    fn empty_spans_overlap_and_are_contained_by_nothing(#[case] start: u64, #[case] end: u64) {
        let regions = read_bed(&["chr1\t10\t20"]);
        assert!(!regions.overlaps("chr1", start, end));
        assert!(!regions.contains("chr1", start, end));
    }

    #[test]
    fn track_and_browser_lines_are_headers_unless_they_are_intervals() {
        let regions = read_bed(&[
            "track1\t0\t5",
            "browser_x\t0\t5",
            "track\tname=y",
            "track name=z\tdescription=a b",
            "browser position chr1:1-100",
            "track\t7\t9",
        ]);
        let mut contigs: Vec<&str> = regions.contigs().collect();
        contigs.sort_unstable();
        assert_eq!(contigs, ["browser_x", "track", "track1"]);
        assert_eq!(spans(&regions, "track"), [(7, 9)]);
    }

    #[test]
    fn comments_blank_lines_crlf_whitespace_and_extra_columns_are_handled() {
        let regions = read_bed(&[
            "# comment",
            "",
            "  ",
            "chr1\t0\t5\r",
            "chr1\t10\t20\tname\t0\t+",
            "chr1\t 30\t40 ",
        ]);
        assert_eq!(spans(&regions, "chr1"), [(0, 5), (10, 20), (30, 40)]);
    }

    #[test]
    fn adjacent_intervals_are_merged() {
        let regions = read_bed(&["chr1\t0\t5", "chr1\t5\t9"]);
        assert_eq!(spans(&regions, "chr1"), [(0, 9)]);
    }

    #[rstest]
    #[case::too_few_columns("chr1\t10", "expected at least 3 tab-separated columns")]
    #[case::space_separated("chr1 0 5", "expected at least 3 tab-separated columns")]
    #[case::non_integer_start("chr1\tten\t20", "start \"ten\" is not an integer")]
    #[case::non_integer_end("chr1\t10\t2.5", "end \"2.5\" is not an integer")]
    #[case::overflow("chr1\t0\t99999999999999999999", "number too large")]
    #[case::end_before_start("chr1\t20\t10", "invalid interval")]
    #[case::negative_start("chr1\t-1\t10", "invalid interval")]
    #[case::empty_contig("\t0\t5", "empty contig name")]
    fn bad_bed_lines_are_rejected_with_their_location(#[case] line: &str, #[case] reason: &str) {
        let dir = TempDir::new().unwrap();
        let text = format!("# header\n\nchr1\t0\t5\n{line}\n");
        let message = bed_error(&dir, text.as_bytes());
        assert!(message.contains(&format!("bad.bed:4: {line:?}: ")), "{message}");
        assert!(message.contains(reason), "{message}");
    }

    #[test]
    fn error_shows_the_whitespace_of_the_bad_line() {
        let dir = TempDir::new().unwrap();
        let message = bed_error(&dir, b"chr1\t0\tfive \r\n");
        assert!(message.contains("bad.bed:1: \"chr1\\t0\\tfive \": "), "{message}");
    }

    #[test]
    fn non_utf8_bed_is_an_error() {
        let dir = TempDir::new().unwrap();
        let message = bed_error(&dir, b"chr1\t0\t5\nchr\xff\t0\t5\n");
        assert!(message.contains("failed to read") && message.contains("bad.bed:2"), "{message}");
    }

    #[test]
    fn truncated_gzip_bed_is_an_error() {
        let dir = TempDir::new().unwrap();
        let compressed = gzip("chr1\t0\t5\n".repeat(1000).as_bytes());
        let message = bed_error(&dir, &compressed[..compressed.len() / 2]);
        assert!(message.contains("failed to read") && message.contains("bad.bed:"), "{message}");
    }

    #[test]
    fn missing_bed_is_an_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("no.bed");
        let message = format!("{:#}", RegionSet::from_bed("r", path).unwrap_err());
        assert!(message.contains("failed to open") && message.contains("no.bed"), "{message}");
    }

    #[test]
    fn empty_bed_has_no_contigs() {
        let regions = read_bed(&[]);
        assert_eq!(regions.contigs().count(), 0);
        assert!(!regions.overlaps("chr1", 0, 1));
    }

    #[rstest]
    #[case::named("seg_dups=/a/b.bed", "seg_dups", "/a/b.bed")]
    #[case::equals_in_path("x=/a/b=c.bed", "x", "/a/b=c.bed")]
    fn parses_named_beds(#[case] spec: &str, #[case] name: &str, #[case] path: &str) {
        let named: NamedBed = spec.parse().unwrap();
        assert_eq!(named, NamedBed { name: name.to_string(), path: PathBuf::from(path) });
        assert_eq!(named.to_string(), spec);
    }

    #[rstest]
    #[case::no_name("/a/b.bed")]
    #[case::empty_name("=x.bed")]
    #[case::empty_path("a=")]
    #[case::hyphen("a-b=x.bed")]
    #[case::non_ascii("café=x.bed")]
    fn rejects_bad_named_beds(#[case] spec: &str) {
        let message = spec.parse::<NamedBed>().unwrap_err().to_string();
        let expected =
            format!("region BEDs must be given as name=path (name: [A-Za-z0-9_]+), got: {spec}");
        assert_eq!(message, expected);
    }

    #[test]
    fn check_contigs_needs_a_shared_contig() {
        let dir = TempDir::new().unwrap();
        let lines = ["1\t0\t5", "2\t0\t5"];
        let path = write_bed(&dir, "numbered.bed", &lines, Encoding::Plain);
        let regions = RegionSet::from_bed("r", &path).unwrap();
        regions.check_contigs(&["chr1", "2"]).unwrap();
        let message = regions.check_contigs(&["chr1", "chr2"]).unwrap_err().to_string();
        let expected = format!(
            "region BED {} has no intervals on the extracted contigs (chr1, chr2); its contigs \
             include: 1, 2. Its contig names must match the variant calls' (e.g. chr1, not 1)",
            path.display()
        );
        assert_eq!(message, expected);
    }

    #[test]
    fn check_contigs_lists_the_first_five_contigs_in_sorted_order() {
        let regions = read_bed(&[
            "c7\t0\t5", "c3\t0\t5", "c1\t0\t5", "c6\t0\t5", "c2\t0\t5", "c5\t0\t5", "c4\t0\t5",
        ]);
        let message = regions.check_contigs(&["chr1"]).unwrap_err().to_string();
        assert!(message.contains("include: c1, c2, c3, c4, c5. "), "{message}");
    }

    #[test]
    fn check_contigs_names_a_set_without_a_path() {
        let regions = RegionSet::from_spans("calls", [("chr1", 5, 5)]);
        let message = regions.check_contigs(&["chr1"]).unwrap_err().to_string();
        assert!(message.starts_with("region BED calls has"), "{message}");
        assert!(message.contains("include: none. "), "{message}");
    }

    #[test]
    fn contains_requires_one_interval() {
        let regions = read_bed(&["chr1\t10\t20", "chr1\t20\t25", "chr1\t30\t40"]);
        let queries = [(10, 12), (15, 25), (29, 31), (35, 41), (5, 9), (40, 41)];
        let got: Vec<bool> = queries.iter().map(|&(s, e)| regions.contains("chr1", s, e)).collect();
        assert_eq!(got, [true, true, false, false, false, false]);
        assert!(!regions.contains("chr9", 0, 1));
    }

    #[test]
    fn empty_bed_intervals_are_skipped() {
        let regions = read_bed(&["chr1\t100\t100", "chr2\t5\t5", "chr2\t10\t20"]);
        // An empty interval contains no base, so a span around it does not overlap it.
        assert!(!regions.overlaps("chr1", 98, 101));
        assert!(regions.contig("chr1").is_none());
        assert!(!regions.overlaps("chr2", 4, 6));
        assert!(!regions.overlaps("chr2", 9, 10));
    }

    #[test]
    fn from_spans_merges_overlapping_and_adjacent_spans() {
        let input = [
            ("chr1", 30, 40),
            ("chr1", 10, 20),
            ("chr1", 15, 30),
            ("chr1", 50, 50),
            ("chr1", 60, 70),
            ("chr1", 12, 14),
            ("chr2", 5, 5),
        ];
        let regions = RegionSet::from_spans("s", input);
        // [10, 20) + [15, 30) + [30, 40) merge, [12, 14) is inside them, and the empty [50, 50)
        // is dropped.
        assert_eq!(spans(&regions, "chr1"), [(10, 40), (60, 70)]);
        assert!(regions.contig("chr2").is_none());
        assert_eq!(regions.name(), "s");
        assert!(regions.path.is_none());
    }

    #[test]
    fn region_arguments_and_manifests_parse() {
        // Report regions take any non-empty name, unlike feature regions.
        assert_eq!(
            "GRCh38_MHC-v2.1=/data/MHC.bed.gz".parse::<NamedPath>().unwrap(),
            NamedPath { name: "GRCh38_MHC-v2.1".into(), path: PathBuf::from("/data/MHC.bed.gz") }
        );
        for bad in ["MHC", "=x.bed", "MHC="] {
            let message = bad.parse::<NamedPath>().unwrap_err().to_string();
            assert_eq!(message, format!("expected NAME=PATH, got {bad:?}"));
        }
        let dir = TempDir::new().unwrap();
        let manifest = dir.path().join("strat.tsv");
        std::fs::write(&manifest, "# comment\nMHC\tMHC.bed.gz\n\nsegdups\t/abs/segdups.bed\n")
            .unwrap();
        let got = read_stratification_manifest(&manifest).unwrap();
        assert_eq!(
            got,
            [
                NamedPath { name: "MHC".into(), path: dir.path().join("MHC.bed.gz") },
                NamedPath { name: "segdups".into(), path: PathBuf::from("/abs/segdups.bed") },
            ]
        );
        std::fs::write(&manifest, "MHC MHC.bed.gz\n").unwrap();
        let message = read_stratification_manifest(&manifest).unwrap_err().to_string();
        assert!(message.contains("strat.tsv:1"), "{message}");
    }
}
