//! Allele read counts by mapping quality ("pileup" evidence): the reads supporting each call's
//! reference and alternate alleles at MAPQ 0, at low MAPQ and at high MAPQ, counted from the
//! alignments at any MAPQ, and the pileup tables `winnowr pileup` writes. The model features
//! derived from them are in [`crate::pileup_features`].
//!
//! HaplotypeCaller ignores reads below its minimum MAPQ (1 in DRAGEN mode, 20 otherwise), so its
//! annotations cannot see reads that an aligner placed at a site with MAPQ 0. When those reads
//! carry one allele (e.g. reads from a paralogous copy, or reads ALT-projected onto the primary
//! assembly), a call made from the high-MAPQ reads alone can have the wrong zygosity. The counts
//! here include every MAPQ, split into bins, so a model can weigh them.
//!
//! A SNP record's allele in a read is the read's base at the SNP (bases below a minimum base
//! quality are not counted; a padded record whose alternates substitute different bases is read
//! at all of them). An indel record's allele is read as `winnowr strlen` reads repeat lengths:
//! the read must align a few flank bases outside each end of the record's repeat tract, and its
//! length change across the tract is compared with each allele's (0 for the reference), so an
//! indel placed anywhere in the tract counts. When an indel record also has an alternate as long
//! as the reference (e.g. `A>T,AT`), a read without a length change is read by its bases where
//! that alternate substitutes them. Reads with neither allele count as "other".
//!
//! Every read is counted, so the two reads of a fragment whose mates overlap a site count twice
//! (unlike HaplotypeCaller, which counts a fragment once).

use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::alleles::{length_change, variant_type_of};
use crate::common::VariantType;
use crate::evidence::{Evidence, EvidenceTables, write_table};
use crate::read_pass::{
    MAX_READS, ReadInputs, SweepSites, anchored_length, query_position_at, run_shards,
    sweep_clusters,
};
use crate::strlen::{tract_of_record, tract_window};

/// The columns of a pileup table, in file order.
pub const PILEUP_COLUMNS: [&str; 14] = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "n_reads",
    "ref_mq0",
    "ref_mqlow",
    "ref_mqhigh",
    "alt_mq0",
    "alt_mqlow",
    "alt_mqhigh",
    "other_mq0",
    "other_mqlow",
    "other_mqhigh",
];

/// The prefix of the pileup table line that records its read settings, before the column header
/// (e.g. `##pileup_settings=mapq20,bq10,flank5`).
pub const SETTINGS_PREFIX: &str = "##pileup_settings=";

/// The MAPQ from which a read is in the high-MAPQ bin (by default).
pub const HIGH_MAPQ: u8 = 20;

/// The base quality a SNP base needs to be counted (by default), as HaplotypeCaller's default
/// minimum base quality.
pub const MIN_BASE_QUALITY: u8 = 10;

/// Reference bases an indel read must align on outside each end of the repeat tract (by
/// default): `winnowr strlen`'s.
pub const FLANK: u64 = crate::strlen::FLANK;

/// How reads are binned and which of their bases count.
///
/// Written as comma-separated settings (see [`fmt::Display`] and [`std::str::FromStr`]), e.g.
/// `mapq20,bq10,flank5`, in pileup tables, feature tables and models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PileupOptions {
    /// Reads with MAPQ at least this are in the high bin; reads with MAPQ from 1 to one less are
    /// in the low bin, and reads with MAPQ 0 in their own.
    pub high_mapq: u8,
    /// The base quality a SNP base needs to be counted.
    pub min_base_quality: u8,
    /// The aligned bases an indel read needs outside each end of the repeat tract.
    pub flank: u64,
}

impl Default for PileupOptions {
    /// The high bin from MAPQ [`HIGH_MAPQ`], base quality [`MIN_BASE_QUALITY`] and [`FLANK`]
    /// flank bases.
    fn default() -> Self {
        PileupOptions { high_mapq: HIGH_MAPQ, min_base_quality: MIN_BASE_QUALITY, flank: FLANK }
    }
}

impl PileupOptions {
    /// The MAPQ bin of a read: 0 (MAPQ 0), 1 (low) or 2 (high).
    #[must_use]
    pub fn bin(&self, mapq: u8) -> usize {
        match mapq {
            0 => 0,
            q if q < self.high_mapq => 1,
            _ => 2,
        }
    }
}

impl fmt::Display for PileupOptions {
    /// Every setting, as [`std::str::FromStr`] parses it back: e.g. `mapq20,bq10,flank5`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mapq{},bq{},flank{}", self.high_mapq, self.min_base_quality, self.flank)
    }
}

impl std::str::FromStr for PileupOptions {
    type Err = anyhow::Error;

    /// Parses comma-separated settings over the defaults: `mapqN` (the high bin's lowest MAPQ),
    /// `bqN` and `flankN` (`default` alone is the defaults).
    fn from_str(text: &str) -> Result<Self> {
        let mut options = PileupOptions::default();
        for item in text.split(',').map(str::trim).filter(|item| !matches!(*item, "" | "default")) {
            if let Some(value) = item.strip_prefix("mapq") {
                options.high_mapq = value.parse().with_context(|| format!("bad MAPQ in {item}"))?;
            } else if let Some(value) = item.strip_prefix("bq") {
                options.min_base_quality =
                    value.parse().with_context(|| format!("bad base quality in {item}"))?;
            } else if let Some(value) = item.strip_prefix("flank") {
                options.flank = value.parse().with_context(|| format!("bad flank in {item}"))?;
            } else {
                bail!("unknown pileup setting '{item}' (expected mapqN, bqN, flankN or default)");
            }
        }
        if options.high_mapq < 2 {
            bail!("the high MAPQ bin must start at 2 or more, so that the low bin is not empty");
        }
        crate::strlen::check_flank(options.flank)?;
        Ok(options)
    }
}

impl Evidence for PileupOptions {
    type Row = PileupRow;

    const NAME: &'static str = "pileup";
    const SETTINGS: &'static str = "settings";
    const OPTION: &'static str = "--settings";
    const COLUMNS: &'static [&'static str] = &PILEUP_COLUMNS;
    const SETTINGS_PREFIX: &'static str = SETTINGS_PREFIX;
    /// A table must record its settings.
    const UNRECORDED: Option<Self> = None;

    fn parse_row(line: &str) -> Result<PileupRow> {
        PileupRow::parse(line)
    }

    fn row_key(row: &PileupRow) -> (&str, u64, &str, &str) {
        (&row.chrom, row.pos, &row.reference, &row.alt)
    }
}

impl serde::Serialize for PileupOptions {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for PileupOptions {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Reads per allele (reference, alternate, other) and MAPQ bin (0, low, high).
pub type AlleleCounts = [[u64; 3]; 3];

/// The allele of a read: the index into [`AlleleCounts`].
pub(crate) const REF: usize = 0;
/// See [`REF`].
pub(crate) const ALT: usize = 1;
/// See [`REF`].
pub(crate) const OTHER: usize = 2;

/// One record of a pileup table: a SNP or indel record's allele read counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PileupRow {
    /// The contig.
    pub chrom: String,
    /// The record's 1-based position.
    pub pos: u64,
    /// The reference allele.
    pub reference: String,
    /// The comma-joined alternate alleles.
    pub alt: String,
    /// The reads overlapping the site (any MAPQ; capped just above [`MAX_READS`]).
    pub n_reads: u64,
    /// The reads per allele and MAPQ bin.
    pub counts: AlleleCounts,
}

impl PileupRow {
    /// The [`PILEUP_COLUMNS`] values, as written to a pileup table.
    #[must_use]
    pub fn to_line(&self) -> String {
        let mut fields = vec![
            self.chrom.clone(),
            self.pos.to_string(),
            self.reference.clone(),
            self.alt.clone(),
        ];
        fields.push(self.n_reads.to_string());
        fields.extend(self.counts.iter().flatten().map(ToString::to_string));
        fields.join("\t")
    }

    /// Parses one pileup table line.
    fn parse(line: &str) -> Result<Self> {
        let fields: Vec<&str> = line.trim_end_matches('\r').split('\t').collect();
        if fields.len() != PILEUP_COLUMNS.len() {
            bail!("expected {} columns, got {}", PILEUP_COLUMNS.len(), fields.len());
        }
        let integer = |index: usize| -> Result<u64> {
            fields[index].parse().with_context(|| {
                format!("{} is not an integer: {:?}", PILEUP_COLUMNS[index], fields[index])
            })
        };
        let mut counts = [[0u64; 3]; 3];
        for (k, count) in counts.iter_mut().flatten().enumerate() {
            *count = integer(5 + k)?;
        }
        Ok(PileupRow {
            chrom: fields[0].to_string(),
            pos: integer(1)?,
            reference: fields[2].to_string(),
            alt: fields[3].to_string(),
            n_reads: integer(4)?,
            counts,
        })
    }
}

/// The rows of one or more pileup tables, with the settings they were measured with.
pub type PileupTables = EvidenceTables<PileupOptions>;

/// Writes rows measured with `options` as a pileup table (see [`write_table`]). Returns the
/// number of rows written.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_pileup_table(
    path: &Path,
    rows: &[PileupRow],
    options: PileupOptions,
) -> Result<usize> {
    write_table(path, options, rows.iter().map(PileupRow::to_line))
}

/// The bases that tell a record's same-length alleles apart: the 0-based positions at which
/// any of them differs from the reference, and each allele's bases there.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Substitutions {
    /// The 0-based positions, in increasing order.
    positions: Vec<i64>,
    /// The reference bases at the positions (upper case).
    reference: Vec<u8>,
    /// Each alternate allele's bases at the positions (upper case).
    alts: Vec<Vec<u8>>,
}

impl Substitutions {
    /// The substitutions of `alts` (each as long as `reference`) in a record at 1-based `pos`,
    /// or `None` if no alternate differs from the reference (ignoring case).
    fn of<'a>(
        pos: u64,
        reference: &[u8],
        alts: impl Iterator<Item = &'a [u8]> + Clone,
    ) -> Option<Substitutions> {
        let differs = |offset: usize| {
            alts.clone().any(|alt| !alt[offset].eq_ignore_ascii_case(&reference[offset]))
        };
        let offsets: Vec<usize> = (0..reference.len()).filter(|&offset| differs(offset)).collect();
        if offsets.is_empty() {
            return None;
        }
        let bases = |allele: &[u8]| -> Vec<u8> {
            offsets.iter().map(|&offset| allele[offset].to_ascii_uppercase()).collect()
        };
        let start = (pos - 1) as i64;
        Some(Substitutions {
            positions: offsets.iter().map(|&offset| start + offset as i64).collect(),
            reference: bases(reference),
            alts: alts.map(bases).collect(),
        })
    }

    /// The 0-based, half-open span of the positions.
    fn span(&self) -> (i64, i64) {
        (self.positions[0], self.positions[self.positions.len() - 1] + 1)
    }

    /// The allele of a read's bases at the positions: [`REF`] or [`ALT`] if they are an allele's
    /// bases, [`OTHER`] otherwise, or `None` if a position is deleted, not aligned or below the
    /// base quality.
    fn allele_of(
        &self,
        record: &rust_htslib::bam::Record,
        cigar: &[rust_htslib::bam::record::Cigar],
        options: &PileupOptions,
    ) -> Option<usize> {
        let start = u64::try_from(record.pos()).ok()?;
        let (qualities, sequence) = (record.qual(), record.seq());
        let mut bases = Vec::with_capacity(self.positions.len());
        for &at in &self.positions {
            let query = query_position_at(cigar, start, u64::try_from(at).ok()?)? as usize;
            let quality = *qualities.get(query)?;
            // A missing quality string is stored as 0xff, and does not exclude the base.
            if quality < options.min_base_quality && quality != 0xff {
                return None;
            }
            bases.push(sequence[query].to_ascii_uppercase());
        }
        Some(if bases == self.reference {
            REF
        } else if self.alts.contains(&bases) {
            ALT
        } else {
            OTHER
        })
    }
}

/// How a site reads the allele of a read.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SiteKind {
    /// A SNP: the read's bases where the alternate alleles substitute a base (one position,
    /// unless a padded record's alternates substitute different bases).
    Snp(Substitutions),
    /// An indel: the read's length change across a repeat tract, with flank bases outside it.
    Indel {
        /// The 0-based first anchor (the first flank base; may be negative).
        first: i64,
        /// The 0-based last anchor (the last flank base).
        last: i64,
        /// The flank bases on each side.
        flank: u64,
        /// The reference tract length.
        length: i64,
        /// The alternate alleles' length changes (none is 0).
        alt_deltas: Vec<i64>,
        /// The substitutions of the alternate alleles as long as the reference (e.g. `T` in
        /// `A>T,AT`), which tell apart the reads without a length change.
        same_length: Option<Substitutions>,
    },
}

/// A record to count reads at: how it reads alleles, and the 0-based, half-open span a read must
/// overlap to be counted.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Site {
    kind: SiteKind,
    span: (i64, i64),
}

impl Site {
    /// The site of a SNP or indel record, or `None` for other records and indel records without a
    /// pure insertion or deletion allele.
    ///
    /// A SNP's alleles are its bases where any alternate allele substitutes one. An indel's are
    /// its length changes; when the record also has an alternate allele as long as the reference
    /// (e.g. `A>T,AT`), a read without a length change is that allele, the reference or other
    /// by its bases where that allele substitutes them.
    ///
    /// # Arguments
    ///
    /// * `contig` - the contig's sequence (for indel repeat tracts).
    /// * `pos` - the record's 1-based position.
    /// * `reference` - the reference allele.
    /// * `alts` - the alternate alleles.
    /// * `options` - the settings (the indel flank).
    fn of_record<S: AsRef<[u8]>>(
        contig: &[u8],
        pos: u64,
        reference: &[u8],
        alts: &[S],
        options: &PileupOptions,
    ) -> Option<Site> {
        let usable = || {
            alts.iter()
                .map(AsRef::as_ref)
                .filter(|alt| *alt != b"*" && *alt != b"." && alt.first() != Some(&b'<'))
        };
        match variant_type_of(reference, alts)? {
            VariantType::Snp => {
                // Every usable alternate is as long as the reference and differs at one base.
                let substitutions = Substitutions::of(pos, reference, usable())?;
                let span = substitutions.span();
                Some(Site { kind: SiteKind::Snp(substitutions), span })
            }
            VariantType::Indel => {
                let tract = tract_of_record(contig, pos, reference, alts)?;
                let ((first, last), span) = tract_window(&tract, options.flank);
                let mut alt_deltas: Vec<i64> = usable()
                    .filter_map(|alt| length_change(reference, alt))
                    .filter(|&delta| delta != 0)
                    .collect();
                alt_deltas.sort_unstable();
                alt_deltas.dedup();
                let same_length = Substitutions::of(
                    pos,
                    reference,
                    usable().filter(|alt| alt.len() == reference.len()),
                );
                // A read must also overlap the substituted bases, wherever they are.
                let span = same_length.as_ref().map_or(span, |substitutions| {
                    let (beg, end) = substitutions.span();
                    (span.0.min(beg), span.1.max(end))
                });
                Some(Site {
                    kind: SiteKind::Indel {
                        first,
                        last,
                        flank: options.flank,
                        length: tract.length() as i64,
                        alt_deltas,
                        same_length,
                    },
                    span,
                })
            }
        }
    }
}

/// The reads counted at one site so far.
#[derive(Debug, Clone, Default)]
struct Accumulator {
    n_reads: u64,
    counts: AlleleCounts,
}

impl Accumulator {
    /// True once more than [`MAX_READS`] reads have been counted (the last is not counted at
    /// its alleles): later reads are not read.
    fn is_full(&self) -> bool {
        self.n_reads > MAX_READS
    }
}

/// The allele a read supports at a site, or `None` if it does not cover the site (a substituted
/// base deleted, not aligned or below the base quality; an indel tract the read does not span
/// with its flanks).
fn allele_of(
    site: &Site,
    record: &rust_htslib::bam::Record,
    cigar: &[rust_htslib::bam::record::Cigar],
    options: &PileupOptions,
) -> Option<usize> {
    match &site.kind {
        SiteKind::Snp(substitutions) => substitutions.allele_of(record, cigar, options),
        SiteKind::Indel { first, last, flank, length, alt_deltas, same_length } => {
            let (_, _, observed) = anchored_length(cigar, record.pos(), *first, *last, *flank)?;
            let delta = observed - length;
            if delta == 0 {
                return match same_length {
                    Some(substitutions) => substitutions.allele_of(record, cigar, options),
                    None => Some(REF),
                };
            }
            Some(if alt_deltas.contains(&delta) { ALT } else { OTHER })
        }
    }
}

/// Counts reads at sites on one contig: each read (see [`sweep_clusters`]) is counted at every
/// site whose span it overlaps, until the site is full: at most [`MAX_READS`] reads are counted
/// at a site's alleles. Overlapping mates are both counted.
fn count_sites(
    reads: &mut rust_htslib::bam::IndexedReader,
    tid: u32,
    sites: &[Site],
    options: &PileupOptions,
) -> Result<Vec<Accumulator>> {
    let spans: Vec<(i64, i64)> = sites.iter().map(|site| site.span).collect();
    let mut counting =
        Counting { sites, options, accumulators: vec![Accumulator::default(); sites.len()] };
    sweep_clusters(reads, tid, &spans, &mut counting)?;
    Ok(counting.accumulators)
}

/// The sites [`count_sites`] counts reads at, with their counts so far.
struct Counting<'a> {
    sites: &'a [Site],
    options: &'a PileupOptions,
    accumulators: Vec<Accumulator>,
}

impl SweepSites for Counting<'_> {
    fn is_full(&self, i: usize) -> bool {
        self.accumulators[i].is_full()
    }

    fn add(
        &mut self,
        i: usize,
        record: &rust_htslib::bam::Record,
        _start: i64,
        _end: i64,
        cigar: &mut Option<rust_htslib::bam::record::CigarStringView>,
    ) {
        let accumulator = &mut self.accumulators[i];
        accumulator.n_reads += 1;
        if accumulator.is_full() || record.cigar_len() == 0 {
            return;
        }
        let cigar = cigar.get_or_insert_with(|| record.cigar());
        if let Some(allele) = allele_of(&self.sites[i], record, cigar, self.options) {
            accumulator.counts[allele][self.options.bin(record.mapq())] += 1;
        }
    }
}

/// The htslib CRAM fields the pileup reads (FLAG, RNAME, POS, MAPQ, CIGAR, SEQ and QUAL), so CRAM
/// decoding can skip read names, mate fields and tags.
const CRAM_REQUIRED_FIELDS: u32 = 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x200 | 0x400;

/// Counts the allele reads of every SNP and indel record (any FILTER) whose POS lies in each
/// region, in region order, then VCF order. Indel records without a pure insertion or deletion
/// allele, and records of other types, are skipped.
///
/// Reads are those that are mapped, primary, not duplicates, not supplementary and pass QC, at
/// any MAPQ. Regions are split into shards measured in parallel on the rayon pool, as `winnowr
/// strlen` measures them.
///
/// # Errors
///
/// Returns an error if a region is malformed or its contig is not in the VCF header or the
/// reference, a record extends past the end of its contig in the reference, or a file cannot be
/// read.
pub fn measure_pileup<S: AsRef<str>>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
    options: &PileupOptions,
) -> Result<Vec<PileupRow>> {
    use rust_htslib::bcf::Read as _;
    let per_shard = run_shards(inputs, regions, CRAM_REQUIRED_FIELDS, |shard| {
        let mut pending: Vec<(PileupRow, Site)> = Vec::new();
        if !shard.fetch_records(inputs.vcf)? {
            return Ok(Vec::new());
        }
        let mut record = shard.vcf.empty_record();
        while let Some(read) = shard.vcf.read(&mut record) {
            read.with_context(|| {
                format!("failed to read {} from {}", shard.chrom, inputs.vcf.display())
            })?;
            let pos = (record.pos() + 1) as u64;
            if pos < shard.start || pos > shard.end {
                continue;
            }
            let alleles = record.alleles();
            let Some((reference, alts)) = alleles.split_first() else { continue };
            if variant_type_of(reference, alts).is_none() {
                continue;
            }
            shard.check_record_end(pos, reference, inputs.reference)?;
            let Some(site) = Site::of_record(shard.contig, pos, reference, alts, options) else {
                continue;
            };
            let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
            let alt: Vec<String> = alts.iter().map(|alt| text(alt)).collect();
            let row = PileupRow {
                chrom: shard.chrom.to_string(),
                pos,
                reference: text(reference),
                alt: alt.join(","),
                n_reads: 0,
                counts: [[0; 3]; 3],
            };
            pending.push((row, site));
        }
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let tid = shard.tid(inputs.alignments)?;
        let (rows, sites): (Vec<PileupRow>, Vec<Site>) = pending.into_iter().unzip();
        let counted = count_sites(shard.reads, tid, &sites, options).with_context(|| {
            format!("failed to count reads on {}:{}-{}", shard.chrom, shard.start, shard.end)
        })?;
        Ok(rows
            .into_iter()
            .zip(counted)
            .map(|(row, accumulator)| PileupRow {
                n_reads: accumulator.n_reads,
                counts: accumulator.counts,
                ..row
            })
            .collect())
    })?;
    Ok(per_shard.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::evidence::read_tables;
    use crate::model_evidence::check_options;
    use crate::testing::{TestRead, write_bam};

    #[test]
    fn settings_parse_print_and_serialize() {
        let default = PileupOptions::default();
        assert_eq!(default.to_string(), "mapq20,bq10,flank5");
        assert_eq!("default".parse::<PileupOptions>().unwrap(), default);
        assert_eq!("".parse::<PileupOptions>().unwrap(), default);
        let other: PileupOptions = "mapq30,bq0,flank2".parse().unwrap();
        assert_eq!(other, PileupOptions { high_mapq: 30, min_base_quality: 0, flank: 2 });
        assert_eq!(other.to_string().parse::<PileupOptions>().unwrap(), other);
        assert_eq!(serde_json::to_string(&other).unwrap(), "\"mapq30,bq0,flank2\"");
        assert_eq!(serde_json::from_str::<PileupOptions>("\"mapq30,bq0,flank2\"").unwrap(), other);
        for bad in ["mapq1", "flank0", "bq", "other", "mapqx"] {
            assert!(bad.parse::<PileupOptions>().is_err(), "{bad}");
        }
        let error = "flank10001".parse::<PileupOptions>().unwrap_err();
        assert_eq!(error.to_string(), "the flank must be between 1 and 10000, got 10001");
        assert_eq!(
            (default.bin(0), default.bin(1), default.bin(19), default.bin(20)),
            (0, 1, 1, 2)
        );
    }

    const HEADER: &str = "chrom\tpos\tref\talt\tn_reads\tref_mq0\tref_mqlow\tref_mqhigh\talt_mq0\talt_mqlow\talt_mqhigh\tother_mq0\tother_mqlow\tother_mqhigh";

    #[test]
    fn tables_round_trip_and_reject_mixed_or_missing_settings() {
        let dir = TempDir::new().unwrap();
        let rows = vec![
            PileupRow {
                chrom: "chr1".into(),
                pos: 10,
                reference: "A".into(),
                alt: "G".into(),
                n_reads: 40,
                counts: [[21, 1, 0], [0, 2, 16], [0, 0, 0]],
            },
            PileupRow {
                chrom: "chr1".into(),
                pos: 20,
                reference: "AC".into(),
                alt: "A,*".into(),
                n_reads: 0,
                counts: [[0; 3]; 3],
            },
        ];
        let options = PileupOptions::default();
        let path = dir.path().join("p.tsv.gz");
        assert_eq!(write_pileup_table(&path, &rows, options).unwrap(), 2);
        let other = dir.path().join("q.tsv");
        write_pileup_table(&other, &rows[..1], options).unwrap();
        let tables = read_tables::<PileupOptions>(&[&path, &other]).unwrap();
        assert_eq!(tables, PileupTables { options, rows: rows.clone() });

        let mixed = dir.path().join("mixed.tsv");
        write_pileup_table(&mixed, &rows, "mapq30".parse().unwrap()).unwrap();
        let message = read_tables::<PileupOptions>(&[&path, &mixed]).unwrap_err().to_string();
        assert!(message.contains("different settings"), "{message}");
        let bare = dir.path().join("bare.tsv");
        std::fs::write(&bare, format!("{HEADER}\n")).unwrap();
        let message = read_tables::<PileupOptions>(&[&bare]).unwrap_err().to_string();
        assert!(message.contains("has no ##pileup_settings= line"), "{message}");
        let bad = dir.path().join("bad.tsv");
        std::fs::write(&bad, format!("{SETTINGS_PREFIX}default\n{HEADER}\nchr1\t1\tA\tG\t1\n"))
            .unwrap();
        let message = format!("{:#}", read_tables::<PileupOptions>(&[&bad]).unwrap_err());
        assert!(message.contains("bad.tsv:3") && message.contains("expected 14"), "{message}");
        let none: [&Path; 0] = [];
        assert!(read_tables::<PileupOptions>(&none).is_err());
    }

    #[test]
    fn mismatched_settings_say_how_to_measure_again() {
        let model = Path::new("m.winnowr.json");
        let (new, other) = (PileupOptions::default(), "mapq30".parse().unwrap());
        assert!(check_options(new, "pileup tables", new, model).is_ok());
        let message = check_options(other, "pileup tables", new, model).unwrap_err();
        assert!(message.to_string().contains("--settings mapq20,bq10,flank5"), "{message}");
    }

    /// 80 bp of left flank ending in C, an A10 tract, and 80 bp of right flank (as in strlen's
    /// tests); the SNP below is at 0-based 122, a G.
    fn reference() -> String {
        format!("{}{}{}", "CGTC".repeat(20), "A".repeat(10), "GCTG".repeat(20))
    }

    fn read<'a>(name: &'a str, pos: i64, cigar: &'a str, mapq: u8) -> TestRead<'a> {
        TestRead { name, pos, cigar, flag: 0, mapq, qual: 30, bases: None }
    }

    /// The bases of a 70-base read from 0-based 90 with base `base` at 0-based 122.
    fn snp_bases(base: char) -> String {
        let mut bases: Vec<u8> = reference().as_bytes()[90..160].to_vec();
        bases[32] = base as u8;
        String::from_utf8(bases).unwrap()
    }

    #[test]
    fn counts_snp_alleles_by_mapq() {
        let dir = TempDir::new().unwrap();
        let reference = reference();
        let options = PileupOptions::default();
        assert_eq!(&reference[122..123], "G");
        // Reference reads at MAPQ 0 (an ALT-projected or paralogous copy), alternate reads at
        // MAPQ 60, and reads that do not count.
        let specs: Vec<(String, String, u8, u16, u8)> = vec![
            ("r0a", 'G', 0, 0, 30),
            ("r0b", 'G', 0, 0, 30),
            ("r0c", 'G', 0, 0x10, 30),
            ("rlow", 'G', 7, 0, 30),
            ("a60a", 'T', 60, 0, 30),
            ("a60b", 'T', 60, 0, 30),
            ("a5", 'T', 5, 0, 30),
            ("other", 'C', 60, 0, 30),
            ("lowbq", 'T', 60, 0, 5),
            ("dup", 'T', 60, 0x400, 30),
            ("secondary", 'G', 0, 0x100, 30),
            ("supplementary", 'G', 0, 0x800, 30),
        ]
        .into_iter()
        .map(|(name, base, mapq, flag, qual)| (name.to_string(), snp_bases(base), mapq, flag, qual))
        .collect();
        let mut reads: Vec<TestRead<'_>> = specs
            .iter()
            .map(|(name, bases, mapq, flag, qual)| TestRead {
                flag: *flag,
                qual: *qual,
                bases: Some(bases),
                ..read(name, 90, "70M", *mapq)
            })
            .collect();
        // A read with the SNP base deleted does not count, but overlaps.
        let deleted: String = reference[90..122].to_string() + &reference[123..161];
        reads.push(TestRead { bases: Some(&deleted), ..read("deleted", 90, "32M1D38M", 60) });
        let bam = write_bam(dir.path(), "reads.bam", reference.len() as u64, &reads);
        let site = Site::of_record(reference.as_bytes(), 123, b"G", &["T"], &options).unwrap();
        assert_eq!(site.span, (122, 123));
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let counted = count_sites(&mut reader, 0, &[site], &options).unwrap();
        // All but the duplicate, secondary and supplementary overlap.
        assert_eq!(counted[0].n_reads, 10);
        assert_eq!(counted[0].counts, [[3, 1, 0], [0, 1, 2], [0, 0, 1]]);
    }

    #[test]
    fn counts_indel_alleles_across_the_tract() {
        let dir = TempDir::new().unwrap();
        let reference = reference();
        let options = PileupOptions::default();
        // A 1 bp deletion of the A10 tract ([80, 90)): the anchors are 75 and 94.
        let site = Site::of_record(reference.as_bytes(), 80, b"CA", &["C"], &options).unwrap();
        assert_eq!(site.span, (75, 95));
        let reads = vec![
            read("ref0", 50, "70M", 0),
            read("ref60", 50, "70M", 60),
            read("del0_left", 50, "31M1D38M", 0), // placed at the start of the tract
            read("del0_right", 50, "38M1D31M", 0), // placed at its end: the same allele
            read("del60", 50, "35M1D34M", 60),
            read("del2", 50, "35M2D33M", 25), // another length
            read("short", 78, "40M", 60),     // starts after the left anchor
        ];
        let bam = write_bam(dir.path(), "reads.bam", reference.len() as u64, &reads);
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let counted = count_sites(&mut reader, 0, &[site], &options).unwrap();
        assert_eq!(counted[0].n_reads, 7);
        assert_eq!(counted[0].counts, [[1, 0, 1], [2, 0, 1], [0, 0, 1]]);
    }

    /// The reference bases of a read from 0-based `start` to `end`, with `changes` (0-based
    /// position, base) substituted.
    fn bases_with(start: usize, end: usize, changes: &[(usize, char)]) -> String {
        let mut bases: Vec<u8> = reference().as_bytes()[start..end].to_vec();
        for &(at, base) in changes {
            bases[at - start] = base as u8;
        }
        String::from_utf8(bases).unwrap()
    }

    #[test]
    fn counts_padded_snp_alleles_at_every_substituted_base() {
        let dir = TempDir::new().unwrap();
        let options = PileupOptions::default();
        // GC>TC,GA at 0-based 122: the alternates substitute different bases.
        assert_eq!(&reference()[122..124], "GC");
        let reads: Vec<(&str, String)> = vec![
            ("ref", bases_with(90, 160, &[])),
            ("tc_a", bases_with(90, 160, &[(122, 'T')])),
            ("tc_b", bases_with(90, 160, &[(122, 'T')])),
            ("ga_a", bases_with(90, 160, &[(123, 'A')])),
            ("ga_b", bases_with(90, 160, &[(123, 'A')])),
            ("ta", bases_with(90, 160, &[(122, 'T'), (123, 'A')])),
        ];
        let mut reads: Vec<TestRead<'_>> = reads
            .iter()
            .map(|(name, bases)| TestRead { bases: Some(bases), ..read(name, 90, "70M", 60) })
            .collect();
        // Ends on the first substituted base, so cannot tell GC from GA: overlaps, not counted.
        let short = bases_with(60, 123, &[]);
        reads.push(TestRead { bases: Some(&short), ..read("short", 60, "63M", 60) });
        let bam = write_bam(dir.path(), "reads.bam", reference().len() as u64, &reads);
        let site =
            Site::of_record(reference().as_bytes(), 123, b"GC", &["TC", "GA"], &options).unwrap();
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let counted = count_sites(&mut reader, 0, &[site], &options).unwrap();
        assert_eq!(counted[0].n_reads, 7);
        assert_eq!(counted[0].counts, [[0, 0, 1], [0, 0, 4], [0, 0, 1]]);
    }

    #[test]
    fn counts_same_length_alleles_of_indel_records_by_their_bases() {
        let dir = TempDir::new().unwrap();
        let reference = reference();
        let options = PileupOptions::default();
        // CA>C,TA at 0-based 79: a deletion of the A10 tract, and a C>T substitution.
        assert_eq!(&reference[79..80], "C");
        let deleted = bases_with(50, 85, &[]) + &bases_with(86, 120, &[]);
        let reads: Vec<(&str, String, &str, u8)> = vec![
            ("ref", bases_with(50, 120, &[]), "70M", 60),
            ("snp_a", bases_with(50, 120, &[(79, 'T')]), "70M", 60),
            ("snp_b", bases_with(50, 120, &[(79, 'T')]), "70M", 0),
            ("other", bases_with(50, 120, &[(79, 'G')]), "70M", 60),
            ("del", deleted, "35M1D34M", 60),
        ];
        let mut reads: Vec<TestRead<'_>> = reads
            .iter()
            .map(|(name, bases, cigar, mapq)| TestRead {
                bases: Some(bases),
                ..read(name, 50, cigar, *mapq)
            })
            .collect();
        // A low-quality substituted base is not counted.
        let low = bases_with(50, 120, &[(79, 'T')]);
        reads.push(TestRead { qual: 5, bases: Some(&low), ..read("lowbq", 50, "70M", 60) });
        let bam = write_bam(dir.path(), "reads.bam", reference.len() as u64, &reads);
        let site =
            Site::of_record(reference.as_bytes(), 80, b"CA", &["C", "TA"], &options).unwrap();
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let counted = count_sites(&mut reader, 0, &[site], &options).unwrap();
        assert_eq!(counted[0].n_reads, 6);
        assert_eq!(counted[0].counts, [[0, 0, 1], [1, 0, 2], [0, 0, 1]]);
    }

    #[test]
    fn sites_of_records() {
        let reference = reference();
        let options = PileupOptions::default();
        let contig = reference.as_bytes();
        let substitutions = |positions: &[i64], reference: &str, alts: &[&str]| Substitutions {
            positions: positions.to_vec(),
            reference: reference.as_bytes().to_vec(),
            alts: alts.iter().map(|alt| alt.as_bytes().to_vec()).collect(),
        };
        // A multi-allelic SNP; an MNP-like record whose alternate differs at its second base.
        let site = Site::of_record(contig, 121, b"G", &["T", "c", "*"], &options).unwrap();
        assert_eq!(site.kind, SiteKind::Snp(substitutions(&[120], "G", &["T", "C"])));
        assert_eq!(site.span, (120, 121));
        let site = Site::of_record(contig, 120, b"CG", &["CT"], &options).unwrap();
        assert_eq!(site.kind, SiteKind::Snp(substitutions(&[120], "G", &["T"])));
        // A padded SNP whose alternates substitute different bases is read at both.
        let site = Site::of_record(contig, 120, b"CG", &["CT", "AG"], &options).unwrap();
        assert_eq!(site.kind, SiteKind::Snp(substitutions(&[119, 120], "CG", &["CT", "AG"])));
        assert_eq!(site.span, (119, 121));
        // A deletion and insertion with their deltas; a complex record is skipped.
        let Some(Site { kind: SiteKind::Indel { alt_deltas, length, same_length, .. }, .. }) =
            Site::of_record(contig, 80, b"CAA", &["C", "CA", "CAAA"], &options)
        else {
            panic!("not an indel")
        };
        assert_eq!((alt_deltas, length, same_length), (vec![-2, -1, 1], 10, None));
        // An indel record with a same-length alternate (as HaplotypeCaller writes at 1/2 sites).
        for (reference, alts) in [(&b"CA"[..], ["C", "TA"]), (b"C", ["T", "CA"])] {
            let Some(Site { kind: SiteKind::Indel { alt_deltas, same_length, .. }, span }) =
                Site::of_record(contig, 80, reference, &alts, &options)
            else {
                panic!("not an indel")
            };
            let delta = if reference.len() == 2 { -1 } else { 1 };
            assert_eq!(alt_deltas, vec![delta]);
            assert_eq!(same_length, Some(substitutions(&[79], "C", &["T"])));
            assert_eq!(span, (75, 95));
        }
        assert!(Site::of_record(contig, 80, b"CA", &["GT"], &options).is_none());
        assert!(Site::of_record(contig, 80, b"C", &["<NON_REF>"], &options).is_none());
    }

    #[test]
    fn measures_records_of_regions_end_to_end() {
        let dir = TempDir::new().unwrap();
        let reference = reference();
        let fasta =
            crate::testing::write_fasta(dir.path(), &[("chr1", &reference), ("chr2", "ACGT")]);
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let records = [
            "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1",
            "chr1\t123\t.\tG\tT\t50\tPASS\t.\tGT\t1/1",
            "chr1\t125\t.\tGCT\tAGA\t50\tLowQual\t.\tGT\t0/1", // an MNP: skipped
            "chr1\t130\t.\tG\tC\t10\tLowQual\t.\tGT\t0/1",
        ];
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &records);
        let (r0, a60) = (snp_bases('G'), snp_bases('T'));
        let reads = vec![
            TestRead { bases: Some(&r0), ..read("r0", 90, "70M", 0) },
            TestRead { bases: Some(&a60), ..read("a60", 90, "70M", 60) },
        ];
        let bam = write_bam(dir.path(), "reads.bam", reference.len() as u64, &reads);
        let inputs = ReadInputs { vcf: &vcf, alignments: &bam, reference: &fasta };
        let options = PileupOptions::default();
        let rows = measure_pileup(&inputs, &["chr1", "chr2"], &options).unwrap();
        let summary: Vec<(u64, u64, AlleleCounts)> =
            rows.iter().map(|r| (r.pos, r.n_reads, r.counts)).collect();
        assert_eq!(
            summary,
            [
                // Overlapping the deletion's tract, but not spanning it with flanks.
                (80, 2, [[0; 3]; 3]),
                (123, 2, [[1, 0, 0], [0, 0, 1], [0, 0, 0]]),
                (130, 2, [[1, 0, 1], [0; 3], [0; 3]]),
            ]
        );
        let path = dir.path().join("pileup.tsv.gz");
        write_pileup_table(&path, &rows, options).unwrap();
        assert_eq!(read_tables::<PileupOptions>(&[&path]).unwrap(), PileupTables { options, rows });
    }
}
