//! Per-read repeat-tract lengths at indel calls ("strlen" evidence): the repeat tract, the length
//! histogram, and the strlen tables `winnowr strlen` writes.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::evidence::{Evidence, EvidenceTables, write_table};
use crate::read_pass::{
    ReadInputs, Shard, SweepSites, anchored_length, run_shards, sweep_clusters,
};

/// Reads per length difference (bp, observed minus reference tract length), in increasing order
/// of difference.
pub type Histogram = BTreeMap<i64, u64>;

/// The columns of a strlen table, in file order.
pub const STRLEN_COLUMNS: [&str; 14] = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "period",
    "tract_len",
    "unit",
    "tract_start",
    "n_overlap",
    "n_spanning",
    "length_histogram",
    "mean_mapq",
    "frac_mapq_lt20",
    "mean_tract_bq",
];

/// A repeat tract in the reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatTract {
    /// The 0-based start.
    pub start: u64,
    /// The 0-based exclusive end (equal to `start` for an empty tract).
    pub end: u64,
    /// The repeat period.
    pub period: u64,
    /// The first `period` bases of the tract (the inserted unit for an empty tract).
    pub unit: String,
}

impl RepeatTract {
    /// The tract length in bp.
    #[must_use]
    pub fn length(&self) -> u64 {
        self.end - self.start
    }

    /// The first `length` bases of the tract's unit repeated: a periodic sequence in the tract's
    /// phase (the tract itself at its own length). Empty if the unit is.
    #[must_use]
    pub fn sequence(&self, length: u64) -> String {
        self.unit
            .chars()
            .cycle()
            .take(if self.unit.is_empty() { 0 } else { length as usize })
            .collect()
    }
}

/// Formats a histogram as `delta:count;...` (`.` when empty).
#[must_use]
pub fn format_histogram(histogram: &Histogram) -> String {
    if histogram.is_empty() {
        return ".".to_string();
    }
    let items: Vec<String> =
        histogram.iter().map(|(delta, count)| format!("{delta}:{count}")).collect();
    items.join(";")
}

/// Parses a histogram written by [`format_histogram`] (`.` or empty is an empty histogram).
///
/// # Errors
///
/// Returns an error if an item is not `delta:count` with integer values.
pub fn parse_histogram(text: &str) -> Result<Histogram> {
    let mut histogram = Histogram::new();
    if text.is_empty() || text == "." {
        return Ok(histogram);
    }
    for item in text.split(';') {
        let parsed = item.split_once(':').and_then(|(delta, count)| {
            Some((delta.parse::<i64>().ok()?, count.parse::<u64>().ok()?))
        });
        let Some((delta, count)) = parsed else {
            bail!("invalid length histogram item '{item}' in '{text}'");
        };
        histogram.insert(delta, count);
    }
    Ok(histogram)
}

/// One record of a strlen table: an indel record's tract and read evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct StrlenRow {
    /// The contig.
    pub chrom: String,
    /// The record's 1-based position.
    pub pos: u64,
    /// The reference allele.
    pub reference: String,
    /// The comma-joined alternate alleles.
    pub alt: String,
    /// The tract's repeat period.
    pub period: u64,
    /// The reference tract length.
    pub tract_len: u64,
    /// The repeat unit.
    pub unit: String,
    /// The tract's 1-based start.
    pub tract_start: u64,
    /// The reads overlapping the tract.
    pub n_overlap: u64,
    /// The spanning reads measured: MAPQ at least the settings' minimum, and one read per
    /// fragment when mates are collapsed.
    pub n_spanning: u64,
    /// The measured reads' length differences, formatted (see [`format_histogram`]).
    pub length_histogram: String,
    /// The mean MAPQ over every spanning read, measured or not (NaN if none).
    pub mean_mapq: f64,
    /// The fraction of every spanning read, measured or not, with MAPQ below 20 (NaN if none).
    pub frac_mapq_lt20: f64,
    /// The mean base quality over the tract of the measured reads (NaN if none).
    pub mean_tract_bq: f64,
}

/// The rows of one or more strlen tables, with the read settings they were measured with.
pub type StrlenTables = EvidenceTables<StrlenOptions>;

/// The prefix of the strlen table line that records its read settings, before the column header
/// (e.g. `##strlen_settings=mapq20,flank5`). Tables without it were measured with
/// [`StrlenOptions::LEGACY`].
pub const SETTINGS_PREFIX: &str = "##strlen_settings=";

/// Parses one strlen table row.
fn parse_row(line: &str) -> Result<StrlenRow> {
    let fields: Vec<&str> = line.trim_end_matches('\r').split('\t').collect();
    if fields.len() != STRLEN_COLUMNS.len() {
        bail!("expected {} columns, got {}", STRLEN_COLUMNS.len(), fields.len());
    }
    let integer = |index: usize| -> Result<u64> {
        fields[index].parse().with_context(|| {
            format!("{} is not an integer: {:?}", STRLEN_COLUMNS[index], fields[index])
        })
    };
    let float = |index: usize| -> Result<f64> {
        let value = fields[index];
        if value == "nan" {
            return Ok(f64::NAN);
        }
        value
            .parse()
            .with_context(|| format!("{} is not a number: {value:?}", STRLEN_COLUMNS[index]))
    };
    // Validate the histogram now, so a malformed one fails here with its location.
    parse_histogram(fields[10])?;
    let (period, tract_start) = (integer(4)?, integer(7)?);
    if period == 0 {
        bail!("period must be at least 1");
    }
    if tract_start == 0 {
        bail!("tract_start is 1-based and must be at least 1");
    }
    Ok(StrlenRow {
        chrom: fields[0].to_string(),
        pos: integer(1)?,
        reference: fields[2].to_string(),
        alt: fields[3].to_string(),
        period,
        tract_len: integer(5)?,
        unit: fields[6].to_string(),
        tract_start,
        n_overlap: integer(8)?,
        n_spanning: integer(9)?,
        length_histogram: fields[10].to_string(),
        mean_mapq: float(11)?,
        frac_mapq_lt20: float(12)?,
        mean_tract_bq: float(13)?,
    })
}

/// Reference bases a read must align on outside each end of the tract to span it (by default).
pub const FLANK: u64 = 5;

/// The MAPQ a read needs to be measured (by default).
pub const MIN_MAPQ: u8 = 20;

/// The largest flank (bp) of the strlen and pileup settings: far beyond any read's useful
/// anchor, and small enough that tract windows cannot overflow.
pub const MAX_FLANK: u64 = 10_000;

/// Which reads measure a tract, and how far outside it they must align.
///
/// Written as comma-separated settings (see [`fmt::Display`] and [`std::str::FromStr`]), e.g.
/// `mapq20,flank5`, in strlen tables, feature tables and models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrlenOptions {
    /// Measure only reads with at least this MAPQ (all spanning reads still give the MAPQ
    /// summaries).
    pub min_mapq: u8,
    /// Count a read pair (fragment) once: a read is not measured when its mate already was.
    pub collapse_mates: bool,
    /// The aligned bases a read needs outside each end of the tract.
    pub flank: u64,
    /// Use at least one repeat unit (the tract's period) of flank.
    pub flank_unit: bool,
}

impl Default for StrlenOptions {
    /// MAPQ at least [`MIN_MAPQ`] and [`FLANK`] flank bases.
    fn default() -> Self {
        StrlenOptions { min_mapq: MIN_MAPQ, collapse_mates: false, flank: FLANK, flank_unit: false }
    }
}

impl StrlenOptions {
    /// The settings of strlen tables, feature tables and models that do not record theirs (made
    /// by winnowr before the settings were recorded): any MAPQ and two flank bases.
    pub const LEGACY: StrlenOptions =
        StrlenOptions { min_mapq: 0, collapse_mates: false, flank: 2, flank_unit: false };

    /// The flank of a tract.
    #[must_use]
    pub fn flank_of(&self, tract: &RepeatTract) -> u64 {
        if self.flank_unit { self.flank.max(tract.period) } else { self.flank }
    }
}

impl fmt::Display for StrlenOptions {
    /// Every setting that [`std::str::FromStr`] parses back, e.g. `mapq20,flank5` or
    /// `mapq0,mates,flank2,unit`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mapq{}", self.min_mapq)?;
        if self.collapse_mates {
            f.write_str(",mates")?;
        }
        write!(f, ",flank{}", self.flank)?;
        if self.flank_unit {
            f.write_str(",unit")?;
        }
        Ok(())
    }
}

impl std::str::FromStr for StrlenOptions {
    type Err = anyhow::Error;

    /// Parses comma-separated settings over the defaults: `mapqN`, `mates`, `flankN` and `unit`
    /// (e.g. `mapq20,mates,flank5`; `default` alone is the defaults, and `legacy` starts from
    /// [`StrlenOptions::LEGACY`] instead).
    fn from_str(text: &str) -> Result<Self> {
        let mut options = StrlenOptions::default();
        let items: Vec<&str> = text.split(',').map(str::trim).collect();
        if items.contains(&"legacy") {
            options = StrlenOptions::LEGACY;
        }
        for &item in items.iter().filter(|item| !matches!(**item, "" | "default" | "legacy")) {
            if let Some(value) = item.strip_prefix("mapq") {
                options.min_mapq = value.parse().with_context(|| format!("bad MAPQ in {item}"))?;
            } else if let Some(value) = item.strip_prefix("flank") {
                options.flank = value.parse().with_context(|| format!("bad flank in {item}"))?;
            } else if item == "mates" {
                options.collapse_mates = true;
            } else if item == "unit" {
                options.flank_unit = true;
            } else {
                bail!(
                    "unknown strlen setting '{item}' (expected mapqN, mates, flankN, unit, \
                     default or legacy)"
                );
            }
        }
        check_flank(options.flank)?;
        Ok(options)
    }
}

impl Evidence for StrlenOptions {
    type Row = StrlenRow;

    const NAME: &'static str = "strlen";
    const SETTINGS: &'static str = "read settings";
    const OPTION: &'static str = "--reads";
    const COLUMNS: &'static [&'static str] = &STRLEN_COLUMNS;
    const SETTINGS_PREFIX: &'static str = SETTINGS_PREFIX;
    /// Tables without a settings line were measured with the legacy settings.
    const UNRECORDED: Option<Self> = Some(StrlenOptions::LEGACY);

    fn parse_row(line: &str) -> Result<StrlenRow> {
        parse_row(line)
    }

    fn row_key(row: &StrlenRow) -> (&str, u64, &str, &str) {
        (&row.chrom, row.pos, &row.reference, &row.alt)
    }
}

impl serde::Serialize for StrlenOptions {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for StrlenOptions {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Reference bases fetched on each side of a record to find its tract.
pub const CONTEXT: u64 = 600;

/// Stop reading a site after this many overlapping reads.
pub const MAX_READS: u64 = 600;

/// The smallest `k` such that `seq` is periodic with period `k` (`seq.len()` if aperiodic).
#[must_use]
pub fn min_period(seq: &[u8]) -> usize {
    let n = seq.len();
    (1..=n).find(|&k| (0..n - k).all(|j| seq[j] == seq[j + k])).unwrap_or(n)
}

/// The repeat tract of a pure insertion or deletion in an (upper-case) reference context, in
/// context coordinates, or `None` if the event is not a pure indel.
///
/// A deletion seeds the tract on the deleted bases; an insertion on the `k` reference bases next
/// to the event that match the inserted sequence (or, if neither side matches, the tract is empty
/// at the event). The tract is then the maximal period-`k` run of the reference around the seed.
///
/// # Arguments
///
/// * `context` - the reference sequence.
/// * `event` - the index in `context` of the first deleted base / the insertion point.
/// * `deleted` - the deleted bases.
/// * `inserted` - the inserted bases.
#[must_use]
pub fn find_tract(
    context: &[u8],
    event: usize,
    deleted: &[u8],
    inserted: &[u8],
) -> Option<RepeatTract> {
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
    let (seq, period, seed_start, seed_end) = if !deleted.is_empty() && inserted.is_empty() {
        let period = min_period(deleted);
        (deleted, period, event, event + deleted.len())
    } else if !inserted.is_empty() && deleted.is_empty() {
        let period = min_period(inserted);
        let unit = &inserted[..period];
        if context.get(event..event + period) == Some(unit) {
            (inserted, period, event, event + period)
        } else if event >= period
            && context.get(event - period..event) == Some(&inserted[inserted.len() - period..])
        {
            (inserted, period, event - period, event)
        } else {
            let at = event as u64;
            return Some(RepeatTract {
                start: at,
                end: at,
                period: period as u64,
                unit: text(unit),
            });
        }
    } else {
        return None;
    };
    let (mut start, mut end) = (seed_start, seed_end);
    while end < context.len() && context[end] == context[end - period] {
        end += 1;
    }
    while start >= 1 && start - 1 + period < context.len() {
        if context[start - 1] != context[start - 1 + period] {
            break;
        }
        start -= 1;
    }
    // Sliced as Python does: a tract cut off at the context's end gives a shorter unit.
    let unit = if end > start {
        &context[start..(start + period).min(context.len())]
    } else {
        &seq[..period]
    };
    Some(RepeatTract {
        start: start as u64,
        end: end as u64,
        period: period as u64,
        unit: text(unit),
    })
}

/// The repeat tract of an indel record, from its first pure insertion or deletion allele, in
/// 0-based contig coordinates, or `None` if no allele is a pure indel.
///
/// # Arguments
///
/// * `contig` - the contig's sequence.
/// * `pos` - the record's 1-based position.
/// * `reference` - the reference allele.
/// * `alts` - the alternate alleles (`*`, `.` and symbolic alleles are skipped).
#[must_use]
pub fn tract_of_record<S: AsRef<[u8]>>(
    contig: &[u8],
    pos: u64,
    reference: &[u8],
    alts: &[S],
) -> Option<RepeatTract> {
    let p0 = pos.checked_sub(1)?;
    let lo = p0.saturating_sub(CONTEXT);
    let hi = (p0 + reference.len() as u64 + CONTEXT).min(contig.len() as u64);
    let context = contig.get(lo as usize..hi as usize)?.to_ascii_uppercase();
    for alt in alts.iter().map(AsRef::as_ref) {
        if alt == b"*" || alt == b"." || alt.first() == Some(&b'<') {
            continue;
        }
        // VCF bases may be in either case; match them against the upper-case context.
        let (offset, deleted, inserted) = crate::alleles::trim_alleles(reference, alt);
        let (deleted, inserted) = (deleted.to_ascii_uppercase(), inserted.to_ascii_uppercase());
        if let Some(tract) = find_tract(&context, (p0 - lo) as usize + offset, &deleted, &inserted)
        {
            return Some(RepeatTract { start: lo + tract.start, end: lo + tract.end, ..tract });
        }
    }
    None
}

/// Read evidence for the length of one repeat tract.
#[derive(Debug, Clone, PartialEq)]
pub struct TractEvidence {
    /// The reads overlapping the anchored tract (capped just above [`MAX_READS`]).
    pub n_overlap: u64,
    /// The spanning reads measured: MAPQ at least the settings' minimum, and one read per
    /// fragment when mates are collapsed.
    pub n_spanning: u64,
    /// Measured reads per observed-minus-reference tract length (bp).
    pub histogram: Histogram,
    /// The mean MAPQ over every spanning read, measured or not.
    pub mean_mapq: Option<f64>,
    /// The fraction of every spanning read, measured or not, with MAPQ below 20.
    pub frac_mapq_lt20: Option<f64>,
    /// The mean base quality of the tract bases in the measured reads.
    pub mean_tract_bq: Option<f64>,
}

/// Checks a flank of the strlen or pileup settings is between 1 and [`MAX_FLANK`].
///
/// # Errors
///
/// Returns an error if it is not.
pub(crate) fn check_flank(flank: u64) -> Result<()> {
    if !(1..=MAX_FLANK).contains(&flank) {
        bail!("the flank must be between 1 and {MAX_FLANK}, got {flank}");
    }
    Ok(())
}

/// The 0-based, half-open span a read must overlap to count for a tract: the tract with `flank`
/// bases on each side. Returns `(first, last)` (the anchors, inclusive; `first` may be negative)
/// and the fetch span `(beg, end)`.
pub(crate) fn tract_window(tract: &RepeatTract, flank: u64) -> ((i64, i64), (i64, i64)) {
    let first = tract.start as i64 - flank as i64;
    let last = (tract.end + flank - 1) as i64;
    ((first, last), (first.max(0), last + 1))
}

/// The read evidence for one tract, accumulated over the reads overlapping it, in order.
struct TractAccumulator {
    first: i64,
    last: i64,
    flank: u64,
    length: i64,
    min_mapq: u8,
    /// The names of the measured reads, when mates are collapsed.
    measured: Option<HashSet<Vec<u8>>>,
    n_overlap: u64,
    n_spanning: u64,
    histogram: Histogram,
    n_mapq: u64,
    mapq_sum: u64,
    mapq_lt20: u64,
    bq_sum: u64,
    bq_n: u64,
}

impl TractAccumulator {
    fn new(tract: &RepeatTract, options: &StrlenOptions) -> TractAccumulator {
        let flank = options.flank_of(tract);
        let ((first, last), _) = tract_window(tract, flank);
        TractAccumulator {
            first,
            last,
            flank,
            length: tract.length() as i64,
            min_mapq: options.min_mapq,
            measured: options.collapse_mates.then(HashSet::new),
            n_overlap: 0,
            n_spanning: 0,
            histogram: Histogram::new(),
            n_mapq: 0,
            mapq_sum: 0,
            mapq_lt20: 0,
            bq_sum: 0,
            bq_n: 0,
        }
    }

    /// True once [`MAX_READS`] reads have been counted: later reads are not read.
    fn is_full(&self) -> bool {
        self.n_overlap > MAX_READS
    }

    /// Counts the next overlapping read (not one with [`crate::read_pass::SKIP_FLAGS`]),
    /// measuring it if it spans the tract. `cigar` caches the read's decoded CIGAR across the
    /// tracts and options it is added to: it is decoded here the first time it is needed.
    fn add(
        &mut self,
        record: &rust_htslib::bam::Record,
        cigar: &mut Option<rust_htslib::bam::record::CigarStringView>,
    ) {
        self.n_overlap += 1;
        if self.is_full() {
            return;
        }
        let start = record.pos();
        if start > self.first || record.cigar_len() == 0 {
            return;
        }
        let cigar = cigar.get_or_insert_with(|| record.cigar());
        if cigar.end_pos() <= self.last {
            return;
        }
        let Some((first_query, last_query, observed)) =
            anchored_length(cigar, start, self.first, self.last, self.flank)
        else {
            return;
        };
        let flank = self.flank;
        let mapq = u64::from(record.mapq());
        self.n_mapq += 1;
        self.mapq_sum += mapq;
        self.mapq_lt20 += u64::from(mapq < 20);
        if record.mapq() < self.min_mapq {
            return;
        }
        if let Some(measured) = &mut self.measured
            && !measured.insert(record.qname().to_vec())
        {
            return;
        }
        *self.histogram.entry(observed - self.length).or_insert(0) += 1;
        self.n_spanning += 1;
        let qualities = record.qual();
        // A missing quality string is stored as 0xff.
        if observed > 0 && qualities.first() != Some(&0xff) {
            let tract_qualities = qualities
                .get((first_query + flank) as usize..=(last_query - flank) as usize)
                .unwrap_or_default();
            self.bq_sum += tract_qualities.iter().map(|&q| u64::from(q)).sum::<u64>();
            self.bq_n += tract_qualities.len() as u64;
        }
    }

    fn finish(self) -> TractEvidence {
        let n_mapq = self.n_mapq;
        let per_spanning = |sum: u64| (n_mapq > 0).then(|| sum as f64 / n_mapq as f64);
        TractEvidence {
            n_overlap: self.n_overlap,
            n_spanning: self.n_spanning,
            histogram: self.histogram,
            mean_mapq: per_spanning(self.mapq_sum),
            frac_mapq_lt20: per_spanning(self.mapq_lt20),
            mean_tract_bq: (self.bq_n > 0).then(|| self.bq_sum as f64 / self.bq_n as f64),
        }
    }
}

/// Measures a tract's length in each read spanning it, with the default [`StrlenOptions`]: a
/// primary, non-duplicate, QC-passing, mapped read aligned (M/=/X) at both anchors, [`FLANK`]
/// bases outside each end of the tract; reads with MAPQ below [`MIN_MAPQ`] are counted in the
/// MAPQ summaries but not measured. Its observed tract length is the number of query bases
/// between the anchors.
///
/// The reads are those a fetch of the tract and its flanks returns, in order, up to
/// [`MAX_READS`]. [`measure_tracts`] measures many tracts with the same result, faster.
///
/// # Errors
///
/// Returns an error if the reads cannot be fetched or read.
#[cfg(test)]
pub fn measure_tract(
    reads: &mut rust_htslib::bam::IndexedReader,
    tid: u32,
    tract: &RepeatTract,
) -> Result<TractEvidence> {
    measure_tract_with(reads, tid, tract, &StrlenOptions::default())
}

/// Measures a tract as [`measure_tract`] does, with other [`StrlenOptions`].
///
/// # Errors
///
/// Returns an error if the reads cannot be fetched or read.
#[cfg(test)]
pub fn measure_tract_with(
    reads: &mut rust_htslib::bam::IndexedReader,
    tid: u32,
    tract: &RepeatTract,
    options: &StrlenOptions,
) -> Result<TractEvidence> {
    use rust_htslib::bam::Read as _;
    let (_, (beg, end)) = tract_window(tract, options.flank_of(tract));
    reads.fetch((tid, beg, end)).context("failed to fetch reads")?;
    let mut accumulator = TractAccumulator::new(tract, options);
    let mut record = rust_htslib::bam::Record::new();
    while let Some(read) = reads.read(&mut record) {
        read.context("failed to read a read")?;
        if record.flags() & crate::read_pass::SKIP_FLAGS != 0 {
            continue;
        }
        accumulator.add(&record, &mut None);
        if accumulator.is_full() {
            break;
        }
    }
    Ok(accumulator.finish())
}

/// Measures tracts on one contig as `measure_tract_with` does for each of `options`, with the
/// same results, but reads each read once per cluster of nearby tracts rather than once per tract
/// and option (see [`sweep_clusters`]): each read is counted for every tract (and option) whose
/// fetch would have returned it (it overlaps the tract's fetch span), in the same order. Returns
/// the evidence per option, then per tract.
///
/// # Errors
///
/// Returns an error if the reads cannot be fetched or read.
pub fn measure_tracts(
    reads: &mut rust_htslib::bam::IndexedReader,
    tid: u32,
    tracts: &[RepeatTract],
    options: &[StrlenOptions],
) -> Result<Vec<Vec<TractEvidence>>> {
    // Per option, then tract: the fetch spans and accumulators.
    let spans: Vec<Vec<(i64, i64)>> = options
        .iter()
        .map(|o| tracts.iter().map(|tract| tract_window(tract, o.flank_of(tract)).1).collect())
        .collect();
    let accumulators: Vec<Vec<TractAccumulator>> = options
        .iter()
        .map(|o| tracts.iter().map(|tract| TractAccumulator::new(tract, o)).collect())
        .collect();
    // The union of each tract's spans over the options.
    let union: Vec<(i64, i64)> = (0..tracts.len())
        .map(|i| {
            let beg = spans.iter().map(|s| s[i].0).min().unwrap_or(0);
            let end = spans.iter().map(|s| s[i].1).max().unwrap_or(0);
            (beg, end)
        })
        .collect();
    let mut measuring = Measuring { spans: &spans, accumulators };
    sweep_clusters(reads, tid, &union, &mut measuring)?;
    Ok(measuring
        .accumulators
        .into_iter()
        .map(|per_tract| per_tract.into_iter().map(TractAccumulator::finish).collect())
        .collect())
}

/// The tracts [`measure_tracts`] measures: per option, then tract, the fetch spans and the
/// evidence so far.
struct Measuring<'a> {
    spans: &'a [Vec<(i64, i64)>],
    accumulators: Vec<Vec<TractAccumulator>>,
}

impl SweepSites for Measuring<'_> {
    /// A tract is full once it is full for every option.
    fn is_full(&self, i: usize) -> bool {
        self.accumulators.iter().all(|per_tract| per_tract[i].is_full())
    }

    /// Adds the read for each option whose fetch span of the tract it overlaps.
    fn add(
        &mut self,
        i: usize,
        record: &rust_htslib::bam::Record,
        start: i64,
        end: i64,
        cigar: &mut Option<rust_htslib::bam::record::CigarStringView>,
    ) {
        for (option_spans, per_tract) in self.spans.iter().zip(self.accumulators.iter_mut()) {
            let (beg, fetch_end) = option_spans[i];
            if start < fetch_end && end > beg && !per_tract[i].is_full() {
                per_tract[i].add(record, cigar);
            }
        }
    }
}

/// The tract and read evidence of one indel record.
#[derive(Debug, Clone, PartialEq)]
pub struct StrlenSite {
    /// The contig.
    pub chrom: String,
    /// The record's 1-based position.
    pub pos: u64,
    /// The reference allele.
    pub reference: String,
    /// The comma-joined alternate alleles.
    pub alt: String,
    /// The repeat tract.
    pub tract: RepeatTract,
    /// The read evidence.
    pub evidence: TractEvidence,
}

/// A measured value as written to a strlen table: `digits` decimal places, or `nan`.
fn format_value(value: Option<f64>, digits: usize) -> String {
    value.map_or_else(|| "nan".to_string(), |v| format!("{v:.digits$}"))
}

/// A measured value as reading it back from a strlen table gives: rounded to `digits` decimal
/// places exactly as [`format_value`] writes it (NaN if none).
fn written_value(value: Option<f64>, digits: usize) -> f64 {
    // Formatting rounds the exact decimal value, which arithmetic rounding does not reproduce.
    value.map_or(f64::NAN, |v| format!("{v:.digits$}").parse().unwrap_or(f64::NAN))
}

impl StrlenSite {
    /// The [`STRLEN_COLUMNS`] values, as written to a strlen table.
    #[must_use]
    pub fn to_row(&self) -> String {
        let ev = &self.evidence;
        let float = format_value;
        [
            self.chrom.clone(),
            self.pos.to_string(),
            self.reference.clone(),
            self.alt.clone(),
            self.tract.period.to_string(),
            self.tract.length().to_string(),
            self.tract.unit.clone(),
            (self.tract.start + 1).to_string(),
            ev.n_overlap.to_string(),
            ev.n_spanning.to_string(),
            format_histogram(&ev.histogram),
            float(ev.mean_mapq, 1),
            float(ev.frac_mapq_lt20, 3),
            float(ev.mean_tract_bq, 1),
        ]
        .join("\t")
    }
}

/// Measures the repeat tract of every indel record (any FILTER) whose POS lies in each region,
/// in region order, then VCF order, with the default [`StrlenOptions`]. Records without a pure insertion or deletion allele are
/// skipped.
///
/// Regions are split into shards measured in parallel on the rayon pool, each with its own
/// readers.
///
/// # Errors
///
/// Returns an error if a region is malformed or its contig is not in the VCF header or the
/// reference, an indel record extends past the end of its contig in the reference, or a file
/// cannot be read.
#[cfg(test)]
pub fn measure_regions<S: AsRef<str>>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
) -> Result<Vec<StrlenSite>> {
    let mut per_option = measure_regions_with(inputs, regions, &[StrlenOptions::default()])?;
    Ok(per_option.pop().unwrap_or_default())
}

/// Measures the records of regions as `measure_regions` does, once for each of `options`, in
/// one pass over the reads. Returns the sites per option.
///
/// # Errors
///
/// As `measure_regions`.
pub fn measure_regions_with<S: AsRef<str>>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
    options: &[StrlenOptions],
) -> Result<Vec<Vec<StrlenSite>>> {
    // Ignored for BAM; for CRAM, skips decoding the fields the tract pass does not read.
    let names = if options.iter().any(|o| o.collapse_mates) { CRAM_READ_NAME } else { 0 };
    let per_shard = run_shards(inputs, regions, CRAM_REQUIRED_FIELDS | names, |shard| {
        measure_shard(shard, inputs, options)
    })?;
    let skipped: usize = per_shard.iter().map(|(_, skipped)| skipped).sum();
    if skipped > 0 {
        log::info!("Skipped {skipped} indel records without a pure insertion or deletion allele");
    }
    let mut per_option: Vec<Vec<StrlenSite>> = vec![Vec::new(); options.len()];
    for (sites, _) in per_shard {
        for (all, shard) in per_option.iter_mut().zip(sites) {
            all.extend(shard);
        }
    }
    Ok(per_option)
}

/// The htslib CRAM fields the tract pass reads (FLAG, RNAME, POS, MAPQ, CIGAR and QUAL), so CRAM
/// decoding can skip read names, bases and tags.
const CRAM_REQUIRED_FIELDS: u32 = 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x400;

/// The htslib CRAM field of read names, decoded only to collapse mates.
const CRAM_READ_NAME: u32 = 0x1;

/// Measures the indel records whose POS is in the shard, returning the sites and the number of
/// records without a pure-indel allele.
fn measure_shard(
    shard: &mut Shard<'_>,
    inputs: &ReadInputs<'_>,
    options: &[StrlenOptions],
) -> Result<(Vec<Vec<StrlenSite>>, usize)> {
    use rust_htslib::bcf::Read as _;
    let (chrom, start, end) = (shard.chrom, shard.start, shard.end);
    // Indel records and their tracts, measured together once all are read; a region without
    // indels needs no reads on its contig.
    let mut pending: Vec<(u64, String, String, RepeatTract)> = Vec::new();
    let mut sites: Vec<Vec<StrlenSite>> = vec![Vec::new(); options.len()];
    let mut skipped = 0;
    if !shard.fetch_records(inputs.vcf)? {
        return Ok((sites, skipped));
    }
    let mut record = shard.vcf.empty_record();
    while let Some(read) = shard.vcf.read(&mut record) {
        read.with_context(|| format!("failed to read {chrom} from {}", inputs.vcf.display()))?;
        let pos = (record.pos() + 1) as u64;
        if pos < start || pos > end {
            continue;
        }
        let alleles = record.alleles();
        let Some((reference, alts)) = alleles.split_first() else { continue };
        if crate::alleles::variant_type_of(reference, alts)
            != Some(crate::common::VariantType::Indel)
        {
            continue;
        }
        shard.check_record_end(pos, reference, inputs.reference)?;
        let Some(tract) = tract_of_record(shard.contig, pos, reference, alts) else {
            skipped += 1;
            continue;
        };
        let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
        let alts: Vec<String> = alts.iter().map(|alt| text(alt)).collect();
        pending.push((pos, text(reference), alts.join(","), tract));
    }
    if pending.is_empty() {
        return Ok((sites, skipped));
    }
    let tid = shard.tid(inputs.alignments)?;
    let tracts: Vec<RepeatTract> = pending.iter().map(|(.., tract)| tract.clone()).collect();
    let evidence = measure_tracts(shard.reads, tid, &tracts, options)
        .with_context(|| format!("failed to measure {chrom}:{start}-{end}"))?;
    for (option_sites, option_evidence) in sites.iter_mut().zip(evidence) {
        for ((pos, reference, alt, tract), evidence) in pending.iter().zip(option_evidence) {
            option_sites.push(StrlenSite {
                chrom: chrom.to_string(),
                pos: *pos,
                reference: reference.clone(),
                alt: alt.clone(),
                tract: tract.clone(),
                evidence,
            });
        }
    }
    Ok((sites, skipped))
}

/// The strlen table rows of measured sites, exactly as writing them to a table and reading it
/// back gives (values are rounded as written), so that measuring in memory gives the same features
/// as `winnowr strlen` followed by `--strlen`.
#[must_use]
pub fn sites_to_rows(sites: &[StrlenSite]) -> Vec<StrlenRow> {
    use rayon::prelude::*;
    sites
        .par_iter()
        .map(|site| {
            let ev = &site.evidence;
            StrlenRow {
                chrom: site.chrom.clone(),
                pos: site.pos,
                reference: site.reference.clone(),
                alt: site.alt.clone(),
                period: site.tract.period,
                tract_len: site.tract.length(),
                unit: site.tract.unit.clone(),
                tract_start: site.tract.start + 1,
                n_overlap: ev.n_overlap,
                n_spanning: ev.n_spanning,
                length_histogram: format_histogram(&ev.histogram),
                mean_mapq: written_value(ev.mean_mapq, 1),
                frac_mapq_lt20: written_value(ev.frac_mapq_lt20, 3),
                mean_tract_bq: written_value(ev.mean_tract_bq, 1),
            }
        })
        .collect()
}

/// Writes sites measured with `options` as a strlen table (see [`write_table`]). Returns the
/// number of sites written.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_strlen_table(
    path: &Path,
    sites: &[StrlenSite],
    options: StrlenOptions,
) -> Result<usize> {
    write_table(path, options, sites.iter().map(StrlenSite::to_row))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tempfile::TempDir;

    use super::*;
    use crate::evidence::{check_options, read_tables};

    #[rstest]
    #[case::empty(".", &[])]
    #[case::one("0:6", &[(0, 6)])]
    #[case::several("-5:1;-2:11;0:173;4:5", &[(-5, 1), (-2, 11), (0, 173), (4, 5)])]
    fn histograms_round_trip(#[case] text: &str, #[case] expected: &[(i64, u64)]) {
        let histogram = parse_histogram(text).unwrap();
        assert_eq!(histogram.into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(format_histogram(&parse_histogram(text).unwrap()), text);
        assert!(parse_histogram("").unwrap().is_empty());
    }

    #[test]
    fn values_are_rounded_as_written() {
        // Formatting rounds the exact decimal value: 0.25 is 0.25 and rounds to even, 0.35 is
        // just below 0.35.
        for (value, digits) in [(0.25, 1), (0.35, 1), (42.25, 1), (0.0625, 3), (1.0 / 3.0, 3)] {
            let written: f64 = format_value(Some(value), digits).parse().unwrap();
            assert_eq!(written_value(Some(value), digits).to_bits(), written.to_bits(), "{value}");
        }
        assert_eq!(written_value(Some(0.25), 1), 0.2);
        assert!(written_value(None, 1).is_nan());
        assert_eq!(format_value(None, 1), "nan");
    }

    #[rstest]
    #[case("1")]
    #[case("a:1")]
    #[case("1:-1")]
    #[case("1:2;")]
    fn bad_histograms_are_rejected(#[case] text: &str) {
        assert!(parse_histogram(text).is_err(), "{text}");
    }

    #[test]
    fn tract_sequences_follow_the_unit_phase() {
        let tract = RepeatTract { start: 10, end: 15, period: 2, unit: "CA".to_string() };
        assert_eq!(tract.length(), 5);
        assert_eq!(tract.sequence(5), "CACAC");
        assert_eq!(tract.sequence(0), "");
        let empty = RepeatTract { start: 3, end: 3, period: 1, unit: String::new() };
        assert_eq!(empty.sequence(4), "");
    }

    const HEADER: &str = "chrom\tpos\tref\talt\tperiod\ttract_len\tunit\ttract_start\tn_overlap\tn_spanning\tlength_histogram\tmean_mapq\tfrac_mapq_lt20\tmean_tract_bq";

    #[test]
    fn reads_and_deduplicates_tables() {
        let dir = TempDir::new().unwrap();
        let first = dir.path().join("a.tsv");
        std::fs::write(
            &first,
            format!("{HEADER}\nchr1\t10146\tAC\tA\t1\t4\tC\t10147\t344\t235\t-1:36;0:173\t8.3\t0.864\t32.3\n"),
        )
        .unwrap();
        let second = dir.path().join("b.tsv");
        std::fs::write(
            &second,
            format!(
                "{HEADER}\nchr1\t10146\tAC\tA\t1\t4\tC\t10147\t1\t1\t0:1\t1\t0\t1\nchr2\t5\tG\tGC\t1\t2\tC\t6\t0\t0\t.\tnan\tnan\tnan\n"
            ),
        )
        .unwrap();
        let tables = read_tables::<StrlenOptions>(&[first, second]).unwrap();
        // Tables without a settings line were measured with the legacy settings.
        assert_eq!(tables.options, StrlenOptions::LEGACY);
        let rows = tables.rows;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].n_overlap, 344);
        assert_eq!(rows[0].length_histogram, "-1:36;0:173");
        assert_eq!((rows[1].chrom.as_str(), rows[1].pos, rows[1].alt.as_str()), ("chr2", 5, "GC"));
        assert!(rows[1].mean_mapq.is_nan());
    }

    #[test]
    fn reads_the_settings_of_tables_and_rejects_mixed_settings() {
        let dir = TempDir::new().unwrap();
        let row = "chr1\t10146\tAC\tA\t1\t4\tC\t10147\t1\t1\t0:1\t1\t0\t1";
        let write = |name: &str, preamble: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, format!("{preamble}{HEADER}\n{row}\n")).unwrap();
            path
        };
        let new = write("new.tsv", "##strlen_settings=mapq20,flank5\n##other=x\n");
        let also_new = write("also_new.tsv", "##strlen_settings=mapq20,flank5\n");
        let old = write("old.tsv", "");
        let tables = read_tables::<StrlenOptions>(&[&new, &also_new]).unwrap();
        assert_eq!((tables.options, tables.rows.len()), (StrlenOptions::default(), 1));
        let message = read_tables::<StrlenOptions>(&[&new, &old]).unwrap_err().to_string();
        assert!(
            message.contains("different read settings")
                && message.contains("mapq20,flank5")
                && message.contains("mapq0,flank2"),
            "{message}"
        );
        let bad = write("bad.tsv", "##strlen_settings=flank0\n");
        let message = format!("{:#}", read_tables::<StrlenOptions>(&[&bad]).unwrap_err());
        assert!(message.contains("bad.tsv:1 has malformed strlen settings"), "{message}");
    }

    #[test]
    fn mismatched_strlen_settings_say_how_to_measure_again() {
        let model = Path::new("m.winnowr.json");
        let (new, old) = (StrlenOptions::default(), StrlenOptions::LEGACY);
        assert!(check_options(new, "strlen tables", new, model).is_ok());
        let message = check_options(old, "strlen tables", new, model).unwrap_err();
        let message = message.to_string();
        assert!(
            message.contains("strlen tables was measured with strlen read settings mapq0,flank2")
                && message.contains("--reads mapq20,flank5"),
            "{message}"
        );
    }

    #[test]
    fn rejects_bad_tables() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.tsv");
        std::fs::write(&path, "chrom\tpos\n").unwrap();
        let message = read_tables::<StrlenOptions>(&[&path]).unwrap_err().to_string();
        assert!(message.contains("bad.tsv is not a strlen table"), "{message}");
        std::fs::write(
            &path,
            format!("{HEADER}\nchr1\tx\tA\tAC\t1\t1\tC\t1\t0\t0\t.\tnan\tnan\tnan\n"),
        )
        .unwrap();
        let message = format!("{:#}", read_tables::<StrlenOptions>(&[&path]).unwrap_err());
        assert!(
            message.contains("bad.tsv:2") && message.contains("pos is not an integer"),
            "{message}"
        );
        let none: [&Path; 0] = [];
        assert!(read_tables::<StrlenOptions>(&none).is_err());
    }

    /// 80 bp of left flank ending in C, an A10 tract, and 80 bp of right flank.
    fn reference() -> (String, usize) {
        let left = "CGTC".repeat(20);
        (format!("{left}{}{}", "A".repeat(10), "GCTG".repeat(20)), left.len())
    }

    fn a_tract(start: usize) -> RepeatTract {
        RepeatTract {
            start: start as u64,
            end: start as u64 + 10,
            period: 1,
            unit: "A".to_string(),
        }
    }

    #[rstest]
    #[case("A", 1)]
    #[case("AAAA", 1)]
    #[case("TGT", 2)]
    #[case("ACGACG", 3)]
    #[case("ACGT", 4)]
    fn min_periods(#[case] seq: &str, #[case] expected: usize) {
        assert_eq!(min_period(seq.as_bytes()), expected);
    }

    #[test]
    fn deletion_tract_extends_over_the_run() {
        let (reference, start) = reference();
        assert_eq!(find_tract(reference.as_bytes(), start, b"A", b""), Some(a_tract(start)));
    }

    #[test]
    fn insertion_tract_seeds_on_the_matching_side() {
        let (reference, start) = reference();
        assert_eq!(find_tract(reference.as_bytes(), start, b"", b"AA"), Some(a_tract(start)));
        assert_eq!(find_tract(reference.as_bytes(), start + 10, b"", b"A"), Some(a_tract(start)));
        let empty = find_tract(reference.as_bytes(), start + 10, b"", b"TT").unwrap();
        let at = start as u64 + 10;
        assert_eq!(empty, RepeatTract { start: at, end: at, period: 1, unit: "T".to_string() });
        assert!(find_tract(reference.as_bytes(), start, b"A", b"G").is_none());
    }

    #[test]
    fn tract_of_record_skips_symbolic_and_star_alleles() {
        let (reference, start) = reference();
        let pos = start as u64; // the C before the tract, 1-based
        let tract = tract_of_record(reference.as_bytes(), pos, b"CA", &["*", "<NON_REF>", "C"]);
        assert_eq!(tract, Some(a_tract(start)));
        assert!(tract_of_record(reference.as_bytes(), pos, b"CA", &["*"]).is_none());
        // Lowercase (soft-masked) reference bases are matched as uppercase.
        let lower = reference.to_ascii_lowercase();
        assert_eq!(tract_of_record(lower.as_bytes(), pos, b"CA", &["C"]), Some(a_tract(start)));
        // So are lowercase allele bases, inserted or deleted.
        let upper = tract_of_record(reference.as_bytes(), pos, b"C", &["CAA"]);
        assert_eq!(upper, Some(a_tract(start)));
        assert_eq!(tract_of_record(reference.as_bytes(), pos, b"c", &["caa"]), upper);
        assert_eq!(tract_of_record(reference.as_bytes(), pos, b"C", &["CaA"]), upper);
        assert_eq!(tract_of_record(reference.as_bytes(), pos, b"ca", &["c"]), Some(a_tract(start)));
    }

    #[test]
    fn a_deletion_cut_off_at_the_contig_end_gives_a_shorter_unit() {
        // The deletion's REF runs one base past the contig, so its tract ends at the context's
        // end; Python's slicing gives a one-base unit rather than failing.
        let contig = b"ACGTACGTAC";
        let tract = tract_of_record(contig, 9, b"ACG", &["A"]).unwrap();
        assert_eq!((tract.start, tract.end, tract.period), (9, 11, 2)); // as in Python
        assert_eq!(tract.unit, "C");
        assert!(tract_of_record(contig, 0, b"AC", &["A"]).is_none());
    }

    fn read<'a>(name: &'a str, pos: i64, cigar: &'a str) -> crate::testing::TestRead<'a> {
        crate::testing::TestRead { name, pos, cigar, flag: 0, mapq: 60, qual: 30, bases: None }
    }

    #[test]
    fn measures_spanning_reads() {
        let dir = TempDir::new().unwrap();
        let tract = a_tract(80); // [80, 90); the anchors are 78 and 91
        let mut reads = vec![
            read("ref1", 50, "70M"),
            read("ref2", 50, "70M"),
            read("del", 50, "32M2D36M"),
            read("ins", 50, "33M1I37M"),
            read("short_left", 79, "40M"), // starts after the left anchor
            read("short_right", 40, "50M"), // ends before the right anchor
            read("anchor_deleted", 50, "28M1D41M"), // the left anchor (78) is deleted
            crate::testing::TestRead { flag: 0x400, ..read("duplicate", 50, "70M") },
            crate::testing::TestRead { mapq: 5, qual: 10, ..read("low_mapq", 50, "70M") },
        ];
        reads.push(crate::testing::TestRead { flag: 0x100, ..read("secondary", 50, "70M") });
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", 200, &reads);
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        // The legacy settings: any MAPQ and two flank bases (the anchors).
        let evidence = measure_tract_with(&mut reader, 0, &tract, &StrlenOptions::LEGACY).unwrap();
        // Overlapping, non-skipped reads: all but the duplicate and the secondary.
        assert_eq!(evidence.n_overlap, 8);
        assert_eq!(evidence.n_spanning, 5);
        assert_eq!(evidence.histogram.into_iter().collect::<Vec<_>>(), [(-2, 1), (0, 3), (1, 1)]);
        assert_eq!(evidence.mean_mapq, Some((60.0 * 4.0 + 5.0) / 5.0));
        assert_eq!(evidence.frac_mapq_lt20, Some(0.2));
        // Tract bases: 10 + 10 + 8 + 11 at quality 30, and 10 at quality 10.
        assert_eq!(evidence.mean_tract_bq, Some((39.0 * 30.0 + 10.0 * 10.0) / 49.0));
    }

    #[test]
    fn measurement_without_reads_is_missing() {
        let dir = TempDir::new().unwrap();
        let bam =
            crate::testing::write_bam(dir.path(), "reads.bam", 200, &[read("far", 150, "40M")]);
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let evidence = measure_tract(&mut reader, 0, &a_tract(80)).unwrap();
        assert_eq!((evidence.n_overlap, evidence.n_spanning), (0, 0));
        assert!(evidence.mean_mapq.is_none() && evidence.mean_tract_bq.is_none());
        let site = StrlenSite {
            chrom: "chr1".to_string(),
            pos: 80,
            reference: "CA".to_string(),
            alt: "C".to_string(),
            tract: a_tract(80),
            evidence,
        };
        assert_eq!(site.to_row(), "chr1\t80\tCA\tC\t1\t10\tA\t81\t0\t0\t.\tnan\tnan\tnan");
    }

    #[test]
    fn stops_after_max_reads() {
        let dir = TempDir::new().unwrap();
        let names: Vec<String> = (0..=MAX_READS + 5).map(|i| format!("r{i}")).collect();
        let reads: Vec<crate::testing::TestRead<'_>> =
            names.iter().map(|name| read(name, 50, "70M")).collect();
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", 200, &reads);
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let evidence = measure_tract(&mut reader, 0, &a_tract(80)).unwrap();
        assert_eq!(evidence.n_overlap, MAX_READS + 1);
        assert_eq!(evidence.n_spanning, MAX_READS);
    }

    #[test]
    fn measuring_tracts_together_matches_measuring_each() {
        // Reads with assorted CIGARs, flags and qualities over a 60 kb contig (a simple LCG keeps
        // them deterministic), a pile-up deeper than MAX_READS, and tracts that overlap, sit at
        // the contig start, lie in gaps without reads, or are more than CLUSTER_GAP apart.
        let mut state = 12345u64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n
        };
        let cigars = ["100M", "40M3D60M", "50M2I48M", "10S90M", "90M10S", "30M5D30M5I35M", "1M"];
        let flags = [0u16, 0, 0, 0, 0x10, 0x400, 0x100, 0x800];
        let mut specs: Vec<(String, i64, &str, u16, u8, u8)> = Vec::new();
        for i in 0..4000 {
            let pos = next(20_000) as i64 + if i % 2 == 0 { 0 } else { 35_000 };
            let cigar = cigars[next(cigars.len() as u64) as usize];
            let flag = flags[next(flags.len() as u64) as usize];
            specs.push((format!("r{i}"), pos, cigar, flag, next(61) as u8, 10 + next(30) as u8));
        }
        for i in 0..=MAX_READS + 20 {
            specs.push((format!("pile{i}"), 5_000, "100M", 0, 60, 30));
        }
        let reads: Vec<crate::testing::TestRead<'_>> = specs
            .iter()
            .map(|(name, pos, cigar, flag, mapq, qual)| crate::testing::TestRead {
                name,
                pos: *pos,
                cigar,
                flag: *flag,
                mapq: *mapq,
                qual: *qual,
                bases: None,
            })
            .collect();
        let dir = TempDir::new().unwrap();
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", 60_000, &reads);
        let mut tracts: Vec<RepeatTract> = vec![a_tract(0), a_tract(1), a_tract(5_040)];
        tracts.extend((0..300).map(|_| a_tract(next(59_900) as usize)));
        tracts.push(RepeatTract { end: 5_200, ..a_tract(5_000) }); // longer than the reads
        tracts.push(a_tract(5_040)); // a repeat of a tract
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let together = measure_tracts(&mut reader, 0, &tracts, &[StrlenOptions::default()])
            .unwrap()
            .pop()
            .unwrap();
        let each: Vec<TractEvidence> =
            tracts.iter().map(|tract| measure_tract(&mut reader, 0, tract).unwrap()).collect();
        assert_eq!(together, each);
        assert!(each.iter().any(|e| e.n_overlap == MAX_READS + 1));
        assert!(each.iter().any(|e| e.n_overlap == 0));
        assert!(each.iter().filter(|e| e.n_spanning > 0).count() > 100);
    }

    #[test]
    fn read_settings_parse() {
        assert_eq!("default".parse::<StrlenOptions>().unwrap(), StrlenOptions::default());
        assert_eq!(StrlenOptions::default().to_string(), "mapq20,flank5");
        assert_eq!("legacy".parse::<StrlenOptions>().unwrap(), StrlenOptions::LEGACY);
        assert_eq!(StrlenOptions::LEGACY.to_string(), "mapq0,flank2");
        assert_eq!("mapq0,flank2".parse::<StrlenOptions>().unwrap(), StrlenOptions::LEGACY);
        assert_eq!(
            "legacy,flank5".parse::<StrlenOptions>().unwrap(),
            StrlenOptions { flank: 5, ..StrlenOptions::LEGACY }
        );
        // Every setting prints as it parses back, and serializes as that text.
        let every = StrlenOptions { min_mapq: 7, collapse_mates: true, flank: 3, flank_unit: true };
        assert_eq!(every.to_string().parse::<StrlenOptions>().unwrap(), every);
        assert_eq!(serde_json::to_string(&every).unwrap(), "\"mapq7,mates,flank3,unit\"");
        assert_eq!(
            serde_json::from_str::<StrlenOptions>("\"mapq7,mates,flank3,unit\"").unwrap(),
            every
        );
        assert!(serde_json::from_str::<StrlenOptions>("\"flank0\"").is_err());
        let options: StrlenOptions = "mapq20,mates,flank5,unit".parse().unwrap();
        assert_eq!(
            options,
            StrlenOptions { min_mapq: 20, collapse_mates: true, flank: 5, flank_unit: true }
        );
        assert_eq!(options.flank_of(&RepeatTract { period: 7, ..a_tract(80) }), 7);
        assert_eq!(options.flank_of(&a_tract(80)), 5);
        assert!("mapq".parse::<StrlenOptions>().is_err());
        assert!("flank0".parse::<StrlenOptions>().is_err());
        assert_eq!("flank10000".parse::<StrlenOptions>().unwrap().flank, MAX_FLANK);
        let error = "flank18446744073709551615".parse::<StrlenOptions>().unwrap_err();
        assert_eq!(
            error.to_string(),
            "the flank must be between 1 and 10000, got 18446744073709551615"
        );
        assert!("other".parse::<StrlenOptions>().is_err());
    }

    #[test]
    fn read_settings_filter_mapq_collapse_mates_and_widen_the_flank() {
        let dir = TempDir::new().unwrap();
        let reads = vec![
            read("pair", 50, "70M"),
            read("pair", 52, "68M"), // its mate: counted once when mates are collapsed
            read("del", 50, "32M2D36M"),
            crate::testing::TestRead { mapq: 5, ..read("low_mapq", 50, "70M") },
            read("near", 77, "40M"), // spans with 2 flank bases, not with 5
        ];
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", 200, &reads);
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let tract = a_tract(80);
        let measure = |reader: &mut rust_htslib::bam::IndexedReader, settings: &str| {
            measure_tract_with(reader, 0, &tract, &settings.parse().unwrap()).unwrap()
        };
        let all = measure(&mut reader, "legacy");
        assert_eq!(all.n_spanning, 5);
        let high_mapq = measure(&mut reader, "legacy,mapq20");
        assert_eq!(high_mapq.n_spanning, 4);
        // The MAPQ summaries still cover every spanning read.
        assert_eq!(high_mapq.frac_mapq_lt20, Some(0.2));
        assert_eq!(measure(&mut reader, "legacy,mates").n_spanning, 4);
        let wide = measure(&mut reader, "legacy,flank5");
        assert_eq!(wide.n_spanning, 4);
        assert_eq!(wide.histogram.into_iter().collect::<Vec<_>>(), [(-2, 1), (0, 3)]);
        // The default: MAPQ 20 and 5 flank bases.
        let default = measure(&mut reader, "default");
        assert_eq!((default.n_spanning, default.frac_mapq_lt20), (3, Some(0.25)));
        // Measured together in one pass, each setting gives what it gives alone.
        let options: Vec<StrlenOptions> =
            ["legacy", "legacy,mapq20", "mates", "default", "legacy,mapq20,mates"]
                .iter()
                .map(|s| s.parse().unwrap())
                .collect();
        let together =
            measure_tracts(&mut reader, 0, std::slice::from_ref(&tract), &options).unwrap();
        for (option, evidence) in options.iter().zip(together) {
            let alone = measure_tract_with(&mut reader, 0, &tract, option).unwrap();
            assert_eq!(evidence, vec![alone], "{option:?}");
        }
    }

    #[test]
    fn measured_sites_become_the_rows_a_table_reads_back() {
        let dir = TempDir::new().unwrap();
        let reads: Vec<crate::testing::TestRead<'_>> = vec![
            crate::testing::TestRead { mapq: 7, qual: 33, ..read("a", 50, "70M") },
            read("b", 50, "32M2D36M"),
            read("c", 50, "33M1I37M"),
        ];
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", 200, &reads);
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let evidence = measure_tract(&mut reader, 0, &a_tract(80)).unwrap();
        let site = StrlenSite {
            chrom: "chr1".to_string(),
            pos: 80,
            reference: "CA".to_string(),
            alt: "C".to_string(),
            tract: a_tract(80),
            evidence,
        };
        let path = dir.path().join("sites.tsv");
        let options = StrlenOptions::default();
        write_strlen_table(&path, std::slice::from_ref(&site), options).unwrap();
        let read_back = read_tables::<StrlenOptions>(&[&path]).unwrap();
        let rows = sites_to_rows(&[site]);
        assert_eq!(read_back, StrlenTables { options, rows: rows.clone() });
        // Rounded as written: the mean MAPQ (7 + 60 + 60) / 3 to one decimal.
        assert_eq!(rows[0].mean_mapq, 42.3);
    }

    #[test]
    fn measures_regions_end_to_end() {
        let dir = TempDir::new().unwrap();
        let (reference, start) = reference();
        let fasta =
            crate::testing::write_fasta(dir.path(), &[("chr1", &reference), ("chr2", "ACGT")]);
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let records = [
            format!("chr1\t{start}\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1"),
            format!("chr1\t{}\t.\tG\tT\t50\tPASS\t.\tGT\t0/1", start + 12),
            format!("chr1\t{}\t.\tG\tGT\t50\tLowQual\t.\tGT\t0/1", start + 20),
        ];
        let records: Vec<&str> = records.iter().map(String::as_str).collect();
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &records);
        let bam = crate::testing::write_bam(
            dir.path(),
            "reads.bam",
            reference.len() as u64,
            &[read("ref", 50, "70M")],
        );
        let inputs = ReadInputs { vcf: &vcf, alignments: &bam, reference: &fasta };
        let sites = measure_regions(&inputs, &["chr1", "chr2"]).unwrap();
        // The SNP is not measured; the insertion has an empty tract (no T next to it).
        assert_eq!(
            sites.iter().map(|s| s.pos).collect::<Vec<_>>(),
            [start as u64, start as u64 + 20]
        );
        assert_eq!(sites[0].evidence.histogram.get(&0), Some(&1));
        let only_snp =
            measure_regions(&inputs, &[format!("chr1:{}-{}", start + 1, start + 15)]).unwrap();
        assert!(only_snp.is_empty());

        let table = dir.path().join("strlen.tsv.gz");
        assert_eq!(write_strlen_table(&table, &sites, StrlenOptions::default()).unwrap(), 2);
        let rows = read_tables::<StrlenOptions>(&[&table]).unwrap().rows;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].length_histogram, "0:1");
        // A region past the contig end is capped to it.
        let capped = measure_regions(&inputs, &["chr1:1-18446744073709551000"]).unwrap();
        assert_eq!(capped.len(), 2);
        let message = format!("{:#}", measure_regions(&inputs, &["chrX"]).unwrap_err());
        assert!(
            message.contains("contig chrX of region chrX is not in the VCF header"),
            "{message}"
        );
        // Repeated or overlapping regions would measure records twice.
        for (regions, expected) in [
            (["chr1", "chr1"], "regions chr1 and chr1 overlap"),
            (["chr1:10-20", "chr1:20-30"], "regions chr1:10-20 and chr1:20-30 overlap"),
            (["chr2", "chr1:100-200"], ""),
        ] {
            let measured = measure_regions(&inputs, &regions);
            if expected.is_empty() {
                assert!(measured.is_ok());
            } else {
                let message = format!("{:#}", measured.unwrap_err());
                assert!(
                    message.ends_with(&format!(
                        "{expected}, so their records would be measured twice"
                    )),
                    "{message}"
                );
            }
        }
    }

    #[rstest]
    #[case::overhanging("chr2\t3\t.\tGTA\tG\t50\tPASS\t.\tGT\t0/1")]
    #[case::past_the_end("chr2\t50\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1")]
    fn records_past_the_reference_contig_end_are_an_error(#[case] record: &str) {
        let dir = TempDir::new().unwrap();
        let (reference, _) = reference();
        let fasta =
            crate::testing::write_fasta(dir.path(), &[("chr1", &reference), ("chr2", "ACGT")]);
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &[record]);
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", reference.len() as u64, &[]);
        let inputs = ReadInputs { vcf: &vcf, alignments: &bam, reference: &fasta };
        let message = format!("{:#}", measure_regions(&inputs, &["chr2"]).unwrap_err());
        assert!(message.contains("extends past the end of chr2 (4 bp)"), "{message}");
    }
}
