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
//! that alternate substitutes them. Reads with neither allele count as "other". The read
//! annotations' allele sums (see [`MapqSums`]) read alleles alike, but a read of an indel
//! allele's length must also have that allele's bases across the tract.
//!
//! Every read is counted, so the two reads of a fragment whose mates overlap a site count twice
//! (as in HaplotypeCaller's mapping-quality sums, which count both mates).

use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, bail};
use smallvec::SmallVec;

use crate::alignment_provenance::AlignmentProvenance;
use crate::alleles::{Genotype, length_change, trim_alleles, variant_type_of};
use crate::common::{PhaseTimer, VariantType};
use crate::evidence::{Evidence, EvidenceTables, write_table};
use crate::read_annotations::{
    LEGACY_MAX_READS, MapqSums, ReadAnnotationOptions, UNAVAILABLE_MAPQ,
};
use crate::read_pass::{
    MAX_READS, ReadInputs, SweepSites, anchored_length, query_position_at, run_shards,
    sweep_clusters,
};
use crate::read_settings::ReadSettings;
use crate::strlen::{RepeatTract, tract_of_record, tract_window};
use crate::vcf_source::{CalledAlts, called_alts, counts_allele};

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

pub use crate::read_settings::{FLANK, MIN_BASE_QUALITY};

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

impl PileupOptions {
    /// The settings as their shared text form holds them.
    fn settings(&self) -> ReadSettings {
        ReadSettings {
            mapq: self.high_mapq,
            min_base_quality: self.min_base_quality,
            flank: self.flank,
        }
    }

    /// The pileup settings that read alleles as read annotations do (with the default MAPQ bins,
    /// which the annotations do not use).
    #[must_use]
    pub fn reading_alleles_as(annotations: &ReadAnnotationOptions) -> Self {
        let (min_base_quality, flank) = annotations.settings.alleles_key();
        PileupOptions { min_base_quality, flank, ..PileupOptions::default() }
    }

    /// True if the pileup reads alleles as read annotations do (the same base quality and
    /// flank), so that one pass over the reads can measure both.
    #[must_use]
    pub fn reads_alleles_as(&self, annotations: &ReadAnnotationOptions) -> bool {
        self.settings().alleles_key() == annotations.settings.alleles_key()
    }
}

impl fmt::Display for PileupOptions {
    /// Every setting, as [`std::str::FromStr`] parses it back: e.g. `mapq20,bq10,flank5`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.settings().fmt(f)
    }
}

impl std::str::FromStr for PileupOptions {
    type Err = anyhow::Error;

    /// Parses comma-separated settings over the defaults: `mapqN` (the high bin's lowest MAPQ),
    /// `bqN` and `flankN` (`default` alone is the defaults).
    fn from_str(text: &str) -> Result<Self> {
        let settings = PileupOptions::default().settings().parse_over(text, "pileup")?;
        if settings.mapq < 2 {
            bail!("the high MAPQ bin must start at 2 or more, so that the low bin is not empty");
        }
        Ok(PileupOptions {
            high_mapq: settings.mapq,
            min_base_quality: settings.min_base_quality,
            flank: settings.flank,
        })
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
pub const REF: usize = 0;
/// See [`REF`].
pub const ALT: usize = 1;
/// See [`REF`].
pub const OTHER: usize = 2;

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

/// Writes rows measured with `options`, of reads aligned as `alignment` records (when known), as a
/// pileup table (see [`write_table`]). Returns the number of rows written.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_pileup_table(
    path: &Path,
    rows: &[PileupRow],
    options: PileupOptions,
    alignment: Option<&AlignmentProvenance>,
) -> Result<usize> {
    write_table(path, options, alignment, rows.iter().map(PileupRow::to_line))
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
    /// with the bases of the `called` ones among them as its alternate alleles, or `None` if no
    /// alternate differs from the reference (ignoring case). The positions are those where any
    /// of `alts` differs, so that a read of an alternate not `called` is neither the reference
    /// nor a called allele (see [`Substitutions::allele_of`]).
    fn of<'a>(
        pos: u64,
        reference: &[u8],
        alts: impl Iterator<Item = &'a [u8]> + Clone,
        called: impl Iterator<Item = &'a [u8]>,
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
            alts: called.map(bases).collect(),
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
        // The read's bases, on the stack for the usual few positions: this runs for every read at
        // every site.
        let mut bases: SmallVec<[u8; 8]> = SmallVec::with_capacity(self.positions.len());
        for &at in &self.positions {
            let query = query_position_at(cigar, start, u64::try_from(at).ok()?)? as usize;
            let quality = *qualities.get(query)?;
            // A missing quality string is stored as 0xff, and does not exclude the base.
            if quality < options.min_base_quality && quality != 0xff {
                return None;
            }
            bases.push(sequence[query].to_ascii_uppercase());
        }
        let bases = bases.as_slice();
        Some(if bases == self.reference.as_slice() {
            REF
        } else if self.alts.iter().any(|alt| alt.as_slice() == bases) {
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
        /// The alternate alleles with a length change: their length changes, which the pileup
        /// counts read, and their bases across the tract, which the read annotations' sums also
        /// check.
        alts: IndelAlleles,
        /// The substitutions of the alternate alleles as long as the reference (e.g. `T` in
        /// `A>T,AT`), which tell apart the reads without a length change.
        same_length: Option<Substitutions>,
    },
}

/// The called alternate alleles of an indel record with a length change, pure or complex: each
/// one's length change, which a read's length change across the repeat tract is compared with,
/// and, when the read annotations' sums are measured, the alleles' bases (see [`IndelBases`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct IndelAlleles {
    /// Each called alternate allele's length change (not 0).
    deltas: Vec<i64>,
    /// The bases the sums check, or `None` if the sums do not read the site (boxed: a site the
    /// pileup counts alone reads holds none; see [`Target::of_record`]).
    bases: Option<Box<IndelBases>>,
}

impl IndelAlleles {
    /// True if a called allele has length change `delta`.
    fn has_delta(&self, delta: i64) -> bool {
        self.deltas.contains(&delta)
    }
}

/// The bases of an indel record's alleles across its repeat tract, which a read must also have
/// to be in the read annotations' sums (the pileup counts read the length alone): a read of a
/// called alternate allele's length must have that allele's bases, and not also those of an
/// uncalled alternate of that length (in `A>T,AT` called `0/2`, a read inserting `A` has `AT`'s
/// length, but not its bases; in `A>AT,AG` called `0/1`, a read inserting a low-quality base has
/// both `AT`'s and `AG`'s); a read of the reference's length must have the reference's bases,
/// with no indel among them; a read of a called alternate allele as long as the reference (by its
/// substituted bases) must have that allele's bases, with no indel among them.
///
/// Each allele's bases are its haplotype over the tract, the record's REF and every alternate
/// allele's equivalent change placements (see [`comparison_span`]): the reference from the start
/// of any of them up to the record, the allele, and the reference after the REF up to their end. A read with
/// an allele's change placed elsewhere in its tract (e.g. the last GT of a run deleted, where the
/// allele deletes the first) has the same bases. Where that span reaches past the anchors (e.g.
/// a long REF of a record with several tracts), it is cut to them, and the allele's bases beyond the anchors are not checked
/// (the cut assumes the allele's change lies within the anchors: a record with an allele whose
/// change lies past them has no allele sums; see [`Site::reads_every_alt`]).
///
/// The flank bases between the span and the anchors are not compared, but a read must have no
/// net length change there: its length change must lie in the span, as the allele's does (see
/// [`IndelBases::window`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct IndelBases {
    /// The 0-based, half-open reference span the bases are over, within the anchors.
    span: (i64, i64),
    /// The reference's bases over the span (upper case).
    reference: Vec<u8>,
    /// Each called alternate allele's length change (not 0) and its bases over the span.
    called: Vec<(i64, Vec<u8>)>,
    /// Each uncalled alternate allele's length change (not 0) and its bases over the span.
    uncalled: Vec<(i64, Vec<u8>)>,
    /// Each called alternate allele as long as the reference: its bases over the span.
    same_length: Vec<Vec<u8>>,
}

impl IndelBases {
    /// The bases of a record at 1-based `pos` with `reference`, and of its `called` and
    /// `uncalled` alternates (pure or complex, and those as long as the reference), across
    /// `tract`, the REF and every alternate's equivalent placements (see [`comparison_span`]),
    /// within the anchors `first` and `last`, or `None` if the contig does not hold them (as for every
    /// record that is measured: one past the contig's end is an error).
    fn of<'a>(
        contig: &[u8],
        pos: u64,
        reference: &[u8],
        (called, uncalled): (impl Iterator<Item = &'a [u8]>, impl Iterator<Item = &'a [u8]>),
        tract: &RepeatTract,
        (first, last): (i64, i64),
    ) -> Option<IndelBases> {
        let (called, uncalled): (Vec<&[u8]>, Vec<&[u8]>) = (called.collect(), uncalled.collect());
        let start = pos.checked_sub(1)? as i64;
        let ref_end = start + reference.len() as i64;
        // Every alternate's change, so that a read with an equivalent change elsewhere in its
        // tract (e.g. the last GT of a run deleted, not the first) is compared across it; those
        // past the anchors are cut to them below (the record then has no allele sums).
        let changes = called.iter().chain(&uncalled).filter_map(|&alt| {
            length_change(reference, alt).filter(|&delta| delta != 0)?;
            comparison_span(contig, pos, reference, alt, (first, last))
        });
        let (core_start, core_end) = changes.fold(
            (start.min(tract.start as i64), ref_end.max(tract.end as i64)),
            |(beg, end), (change_start, change_end)| {
                (beg.min(change_start as i64), end.max(change_end as i64))
            },
        );
        let before = contig.get(usize::try_from(core_start).ok()?..usize::try_from(start).ok()?)?;
        let after = contig.get(usize::try_from(ref_end).ok()?..usize::try_from(core_end).ok()?)?;
        // The bases outside the anchors, which a read's bases between them do not cover.
        let (cut_start, cut_end) =
            ((first - core_start).max(0) as usize, (core_end - (last + 1)).max(0) as usize);
        let bases_of = |allele: &[u8]| -> Vec<u8> {
            let bases: Vec<u8> =
                before.iter().chain(allele).chain(after).map(u8::to_ascii_uppercase).collect();
            let kept = bases.len().saturating_sub(cut_end);
            bases.get(cut_start..kept).unwrap_or_default().to_vec()
        };
        let with_length_change = |alts: &[&[u8]]| {
            alts.iter()
                .filter_map(|&alt| {
                    let delta = length_change(reference, alt).filter(|&delta| delta != 0)?;
                    Some((delta, bases_of(alt)))
                })
                .collect()
        };
        let same_length = called
            .iter()
            .filter(|&&alt| length_change(reference, alt) == Some(0))
            .map(|&alt| bases_of(alt))
            .collect();
        Some(IndelBases {
            span: (core_start.max(first), core_end.min(last + 1)),
            reference: bases_of(reference),
            called: with_length_change(&called),
            uncalled: with_length_change(&uncalled),
            same_length,
        })
    }

    /// The query positions of a read's bases over the span, given its anchors at query positions
    /// `first_query` and `last_query`, or `None` if they cannot be compared.
    ///
    /// They are the bases after the read's base at the reference position just before the span
    /// and before its base at the reference position just after it (or from or to its anchor,
    /// where the span reaches one), so that bases inserted at either edge are in them. Both of
    /// those bases must be aligned (not deleted), and the read must have no net length change
    /// between each of them and its anchor, even where an insertion in one flank and a deletion
    /// in the other cancel out: otherwise its indel is outside the span, where the allele's is
    /// not. A read without bases (SEQ `*`) has none.
    fn window(
        &self,
        record: &rust_htslib::bam::Record,
        cigar: &[rust_htslib::bam::record::Cigar],
        (first, last): (i64, i64),
        (first_query, last_query): (u64, u64),
    ) -> Option<std::ops::Range<usize>> {
        let (beg, end) = self.span;
        let start = u64::try_from(record.pos()).ok()?;
        // The query position aligned at a reference position, if it is as far from the anchor's
        // as on the reference (no length change between them).
        let aligned = |at: i64, anchor: i64, anchor_query: u64| -> Option<u64> {
            let query = query_position_at(cigar, start, u64::try_from(at).ok()?)?;
            (query.abs_diff(anchor_query) == at.abs_diff(anchor)).then_some(query)
        };
        let from =
            if beg > first { aligned(beg - 1, first, first_query)? + 1 } else { first_query };
        let to = if end <= last { aligned(end, last, last_query)? } else { last_query + 1 };
        let window = from as usize..(to as usize).max(from as usize);
        (window.end <= record.seq_len()).then_some(window)
    }

    /// True if the read's bases at the query positions `window` are `bases`: a base below the
    /// base quality, or `N`, matches any.
    fn has_bases(
        record: &rust_htslib::bam::Record,
        window: &std::ops::Range<usize>,
        bases: &[u8],
        options: &PileupOptions,
    ) -> bool {
        let (qualities, sequence) = (record.qual(), record.seq());
        bases.len() == window.len()
            && window.clone().zip(bases).all(|(query, &base)| {
                let quality = qualities.get(query).copied().unwrap_or(0xff);
                let read_base = sequence[query].to_ascii_uppercase();
                read_base == base
                    || read_base == b'N'
                    || (quality < options.min_base_quality && quality != 0xff)
            })
    }

    /// The allele in the sums of a read whose anchors are at query positions `first_query` and
    /// `last_query`, with length change `delta` between them: [`ALT`] if it has the bases of a
    /// called alternate allele of that length change and of no uncalled one, [`REF`] if it has
    /// no length change and the reference's bases with no indel among them, else [`OTHER`] (as
    /// for a read that skips reference, `N`, between its anchors: its length change is not an
    /// indel).
    fn summed(
        &self,
        record: &rust_htslib::bam::Record,
        cigar: &[rust_htslib::bam::record::Cigar],
        anchors: (i64, i64),
        queries: (u64, u64),
        delta: i64,
        options: &PileupOptions,
    ) -> usize {
        if has_ref_skip_between(cigar, record.pos(), anchors) {
            return OTHER;
        }
        let Some(window) = self.window(record, cigar, anchors, queries) else {
            return OTHER;
        };
        let has = |bases: &[u8]| Self::has_bases(record, &window, bases, options);
        if delta == 0 {
            let clean = !self.has_indel(record, cigar, anchors);
            return if clean && has(&self.reference) { REF } else { OTHER };
        }
        let any = |alleles: &[(i64, Vec<u8>)]| {
            alleles.iter().any(|(alt_delta, bases)| *alt_delta == delta && has(bases))
        };
        if any(&self.called) && !any(&self.uncalled) { ALT } else { OTHER }
    }

    /// The allele in the sums of a read with no length change whose substituted bases are those
    /// of a called alternate allele as long as the reference (see [`SiteKind::Indel`]): [`ALT`]
    /// if it has that allele's bases across the span too, with no indel among them, else
    /// [`OTHER`] (e.g. in `CA>C,GA` called `1/2`, a read with the `G` of `GA` and another `G` in
    /// the A tract), or that skips reference (`N`) between its anchors.
    fn summed_same_length(
        &self,
        record: &rust_htslib::bam::Record,
        cigar: &[rust_htslib::bam::record::Cigar],
        anchors: (i64, i64),
        queries: (u64, u64),
        options: &PileupOptions,
    ) -> usize {
        if has_ref_skip_between(cigar, record.pos(), anchors) {
            return OTHER;
        }
        let Some(window) = self.window(record, cigar, anchors, queries) else {
            return OTHER;
        };
        let has = |bases: &Vec<u8>| Self::has_bases(record, &window, bases, options);
        let clean = !self.has_indel(record, cigar, anchors);
        if clean && self.same_length.iter().any(has) { ALT } else { OTHER }
    }

    /// True if a read has an insertion or deletion in the span, or an insertion at either edge
    /// of it within the anchors (see [`has_indel_within`]).
    fn has_indel(
        &self,
        record: &rust_htslib::bam::Record,
        cigar: &[rust_htslib::bam::record::Cigar],
        anchors: (i64, i64),
    ) -> bool {
        let start = u64::try_from(record.pos()).unwrap_or_default();
        has_indel_within(cigar, start, self.span, anchors)
    }
}

/// True if a read aligned from 0-based `start` with `cigar` skips reference (`N`) between the
/// anchors `first` and `last` (exclusive): a skip, unlike a deletion, is no allele.
fn has_ref_skip_between(
    cigar: &[rust_htslib::bam::record::Cigar],
    start: i64,
    (first, last): (i64, i64),
) -> bool {
    use rust_htslib::bam::record::Cigar;
    let mut at = start;
    for op in cigar {
        match *op {
            Cigar::Match(len) | Cigar::Equal(len) | Cigar::Diff(len) | Cigar::Del(len) => {
                at += i64::from(len);
            }
            Cigar::RefSkip(len) => {
                if at < last && at + i64::from(len) > first + 1 {
                    return true;
                }
                at += i64::from(len);
            }
            Cigar::Ins(_) | Cigar::SoftClip(_) | Cigar::HardClip(_) | Cigar::Pad(_) => {}
        }
        if at >= last {
            break;
        }
    }
    false
}

/// True if consecutive skipped reference bases cover the entire half-open annotation span.
/// A deletion still contributes depth, as does a read aligned to either flank of the span.
fn skips_span(
    cigar: &[rust_htslib::bam::record::Cigar],
    start: i64,
    (beg, end): (i64, i64),
) -> bool {
    use rust_htslib::bam::record::Cigar;
    let (mut at, mut skipped_from) = (start, start);
    for op in cigar {
        match *op {
            Cigar::RefSkip(len) => {
                at += i64::from(len);
                if skipped_from <= beg && at >= end {
                    return true;
                }
            }
            Cigar::Match(len) | Cigar::Equal(len) | Cigar::Diff(len) | Cigar::Del(len) => {
                at += i64::from(len);
                skipped_from = at;
            }
            // Operations consuming no reference do not interrupt consecutive skipped bases.
            Cigar::Ins(_) | Cigar::SoftClip(_) | Cigar::HardClip(_) | Cigar::Pad(_) => {}
        }
        if at >= end {
            break;
        }
    }
    false
}

/// True if a read aligned from 0-based `start` with `cigar` has an insertion or deletion within
/// the 0-based, half-open reference span `(beg, end)`, or an insertion at either edge of it
/// between the anchors `first` and `last` (inclusive: an insertion before `first` or after
/// `last` is outside them, and changes no allele).
fn has_indel_within(
    cigar: &[rust_htslib::bam::record::Cigar],
    start: u64,
    (beg, end): (i64, i64),
    (first, last): (i64, i64),
) -> bool {
    use rust_htslib::bam::record::Cigar;
    let mut at = start as i64;
    for op in cigar {
        match *op {
            Cigar::Match(len) | Cigar::Equal(len) | Cigar::Diff(len) => at += i64::from(len),
            Cigar::Del(len) | Cigar::RefSkip(len) => {
                if at < end && at + i64::from(len) > beg {
                    return true;
                }
                at += i64::from(len);
            }
            Cigar::Ins(_) => {
                // Inserted before the reference base `at`.
                if (beg.max(first + 1)..=end.min(last)).contains(&at) {
                    return true;
                }
            }
            Cigar::SoftClip(_) | Cigar::HardClip(_) | Cigar::Pad(_) => {}
        }
        if at > end {
            break;
        }
    }
    false
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
        Site::of_called(contig, pos, reference, alts, |_| true, options, None)
    }

    /// The site of a record as [`Site::of_record`] reads it, but whose alternate alleles are
    /// only those `called` (by 1-based index): a read is classified against every alternate (the
    /// record's variant type, substituted positions and span are its own), and a read of an
    /// alternate not called is [`OTHER`], neither the reference nor an alternate. An indel's
    /// repeat tract is that of the called alternates, or the record's when they have none.
    ///
    /// As the substituted positions include those only an uncalled alternate substitutes, a read
    /// whose base there is deleted, not aligned or below the base quality has no allele (and so
    /// is in neither allele sum), even if its bases at the called alternates' positions are read.
    ///
    /// With `annotation_tract`, uses that independently selected tract and holds the alleles'
    /// bases for annotation sums (see [`IndelBases`]); pileup-only sites use the original tract
    /// selection and do not build base templates.
    fn of_called<S: AsRef<[u8]>>(
        contig: &[u8],
        pos: u64,
        reference: &[u8],
        alts: &[S],
        called: impl Fn(u32) -> bool,
        options: &PileupOptions,
        annotation_tract: Option<&RepeatTract>,
    ) -> Option<Site> {
        let is_usable = |alt: &&[u8]| *alt != b"*" && *alt != b"." && alt.first() != Some(&b'<');
        let usable = || alts.iter().map(AsRef::as_ref).filter(is_usable);
        let called = &called;
        let usable_called_or_not = move |is_called: bool| {
            (1..)
                .zip(alts)
                .filter(move |&(index, _)| called(index) == is_called)
                .map(|(_, alt)| alt.as_ref())
                .filter(is_usable)
        };
        let usable_called = move || usable_called_or_not(true);
        match variant_type_of(reference, alts)? {
            VariantType::Snp => {
                // Every usable alternate is as long as the reference and differs at one base.
                let substitutions = Substitutions::of(pos, reference, usable(), usable_called())?;
                let span = substitutions.span();
                Some(Site { kind: SiteKind::Snp(substitutions), span })
            }
            VariantType::Indel => {
                // The called alternates' tract, so that a read of a called allele of another
                // tract than the record's first is read across its own (the sums are read only
                // if its anchors take in every alternate's change: see `reads_every_alt`).
                let called_alts: Vec<&[u8]> = usable_called().collect();
                let tract = annotation_tract
                    .cloned()
                    .or_else(|| tract_of_record(contig, pos, reference, &called_alts))
                    .or_else(|| tract_of_record(contig, pos, reference, alts))?;
                let ((first, last), span) = tract_window(&tract, options.flank);
                let deltas = usable_called()
                    .filter_map(|alt| length_change(reference, alt))
                    .filter(|&delta| delta != 0)
                    .collect();
                let bases = if annotation_tract.is_some() {
                    Some(Box::new(IndelBases::of(
                        contig,
                        pos,
                        reference,
                        (usable_called(), usable_called_or_not(false)),
                        &tract,
                        (first, last),
                    )?))
                } else {
                    None
                };
                let alts = IndelAlleles { deltas, bases };
                // Without a called alternate as long as the reference, a read of one that is not
                // called is still told apart from the reference, as other.
                let is_same_length = |alt: &&[u8]| alt.len() == reference.len();
                let same_length = Substitutions::of(
                    pos,
                    reference,
                    usable().filter(is_same_length),
                    usable_called().filter(is_same_length),
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
                        alts,
                        same_length,
                    },
                    span,
                })
            }
        }
    }

    /// True if the site's anchors take in the change of every alternate allele with a length
    /// change, called or not (as an SNP site always does): its own repeat tract for a pure
    /// insertion or deletion (including equivalent rotations), or its deleted bases for a complex allele, must lie
    /// after the first anchor and up to the last. Otherwise a read of an alternate whose change
    /// lies outside the anchors has no length change between them, and would be read as the
    /// reference (e.g. a GT deleted past the anchors of the called A deletion in
    /// `CAAAAGT>CAAAGT,CAAAA` called `0/1`).
    fn reads_every_alt<S: AsRef<[u8]>>(
        &self,
        contig: &[u8],
        pos: u64,
        reference: &[u8],
        alts: &[S],
    ) -> bool {
        let SiteKind::Indel { first, last, .. } = self.kind else {
            return true;
        };
        changes_within_anchors(contig, pos, reference, alts, (first, last))
    }
}

/// Whether every length-changing ALT, called or not, fits inside these annotation anchors.
fn changes_within_anchors<S: AsRef<[u8]>>(
    contig: &[u8],
    pos: u64,
    reference: &[u8],
    alts: &[S],
    (first, last): (i64, i64),
) -> bool {
    alts.iter().map(AsRef::as_ref).all(|alt| {
        if length_change(reference, alt).is_none_or(|delta| delta == 0) {
            return true;
        }
        comparison_span(contig, pos, reference, alt, (first, last))
            .is_some_and(|(start, end)| start as i64 > first && end as i64 <= last)
    })
}

/// Chooses the shortest existing ALT tract window covering every length change, then the
/// leftmost on a tie. This is independent of ALT order and never enlarges the pileup window.
fn annotation_tract<S: AsRef<[u8]>>(
    contig: &[u8],
    pos: u64,
    reference: &[u8],
    alts: &[S],
    options: &PileupOptions,
) -> Option<RepeatTract> {
    alts.iter()
        .filter_map(|alt| tract_of_record(contig, pos, reference, std::slice::from_ref(alt)))
        .filter(|tract| {
            changes_within_anchors(
                contig,
                pos,
                reference,
                alts,
                tract_window(tract, options.flank).0,
            )
        })
        .min_by_key(|tract| (tract.length(), tract.start, tract.end))
}

/// The 0-based, half-open reference span of the change of an alternate allele `alt` with a
/// length change, of a record at 1-based `pos` with `reference`: its own repeat tract for a
/// pure insertion or deletion (perhaps empty), its deleted bases for a complex allele; or `None`
/// if `pos` is 0.
fn change_span(contig: &[u8], pos: u64, reference: &[u8], alt: &[u8]) -> Option<(u64, u64)> {
    let p0 = pos.checked_sub(1)?;
    Some(match tract_of_record(contig, pos, reference, &[alt]) {
        Some(tract) => (tract.start, tract.end),
        None => {
            let (offset, deleted, _) = trim_alleles(reference, alt);
            let start = p0 + offset as u64;
            (start, start + deleted.len() as u64)
        }
    })
}

/// The annotation comparison span, including indel-component rotations up to the first placement
/// outside either anchor. Those out-of-anchor bounds let `reads_every_alt` reject ambiguity
/// before the comparison templates are clipped, without scanning an arbitrarily long repeat.
/// Equivalent rotations can lie outside the tract span, including when an insertion cannot
/// seed a complete motif.
/// This extends only the bases compared, not the anchors or the pileup's eligibility span.
fn comparison_span(
    contig: &[u8],
    pos: u64,
    reference: &[u8],
    alt: &[u8],
    (first, last): (i64, i64),
) -> Option<(u64, u64)> {
    let (beg, end) = change_span(contig, pos, reference, alt)?;
    let (offset, deleted, inserted) = trim_alleles(reference, alt);
    let length = deleted.len().abs_diff(inserted.len());
    if length == 0 {
        return Some((beg, end));
    }
    // A complex allele can place its length-changing component at either edge of its
    // substitutions. Beyond the trimmed span, its prefix/suffix gives the rotating bases.
    // The complete allele sequence is still checked, not just this component's length.
    let longer = if inserted.len() > deleted.len() { inserted } else { deleted };
    let (left_unit, right_unit) = (&longer[..length], &longer[longer.len() - length..]);
    let at = pos - 1 + offset as u64;
    let mut left = at;
    while left as i64 > first && left > 0 {
        let phase = length - 1 - ((at - left) % length as u64) as usize;
        if !contig
            .get((left - 1) as usize)
            .is_some_and(|b| b.eq_ignore_ascii_case(&left_unit[phase]))
        {
            break;
        }
        left -= 1;
    }
    let deleted_end = at + deleted.len() as u64;
    let mut right = deleted_end;
    while right as i64 <= last && right < contig.len() as u64 {
        let phase = ((right - deleted_end) % length as u64) as usize;
        if !contig.get(right as usize).is_some_and(|b| b.eq_ignore_ascii_case(&right_unit[phase])) {
            break;
        }
        right += 1;
    }
    Some((beg.min(left), end.max(right)))
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

/// The allele a read supports at a site, as the pileup counts it and as the read annotations sum
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadAllele {
    /// The allele the pileup counts ([`REF`], [`ALT`] or [`OTHER`]): an indel's by its length.
    counted: usize,
    /// The allele the read annotations sum, when [`allele_of`] is asked for it (else
    /// `counted`): `counted`, but [`OTHER`] for an indel read without the bases of the allele of
    /// its length (or, without a length change, of the same-length allele its substituted bases
    /// are) across the tract (see [`IndelBases`]), for an indel read that skips reference (`N`)
    /// between its anchors, and for a read without bases (SEQ `*`), whose allele is not known.
    summed: usize,
}

impl ReadAllele {
    /// The same allele for the counts and the sums.
    fn both(allele: usize) -> ReadAllele {
        ReadAllele { counted: allele, summed: allele }
    }
}

/// The allele a read supports at a site, or `None` if it does not cover the site (a substituted
/// base deleted, not aligned or below the base quality; an indel tract the read does not span
/// with its flanks). With `summed`, its allele in the read annotations' sums too (see
/// [`ReadAllele::summed`]); without, the indel's bases are not compared. `resolved`, when given,
/// has the read's '=' bases resolved against the reference, for the sums only.
fn allele_of(
    site: &Site,
    record: &rust_htslib::bam::Record,
    cigar: &[rust_htslib::bam::record::Cigar],
    options: &PileupOptions,
    summed: bool,
    resolved: Option<&rust_htslib::bam::Record>,
) -> Option<ReadAllele> {
    if let Some(resolved) = resolved {
        let counted = allele_of_sequence(site, record, cigar, options, false);
        let annotated = allele_of_sequence(site, resolved, cigar, options, true);
        // A SNP read of '=' was OTHER in the pileup; resolving it must change only the sums.
        return counted.map(|allele| ReadAllele {
            counted: allele.counted,
            summed: annotated.map_or(OTHER, |allele| allele.summed),
        });
    }
    allele_of_sequence(site, record, cigar, options, summed)
}

/// Resolves BAM SEQ '=' only where a query base is aligned to a FASTA base. Inserted or
/// soft-clipped '=' has no aligned reference base and remains a literal mismatch, not a wildcard.
fn resolve_reference_bases(
    record: &rust_htslib::bam::Record,
    cigar: &[rust_htslib::bam::record::Cigar],
    contig: &[u8],
) -> rust_htslib::bam::Record {
    use rust_htslib::bam::record::{Cigar, CigarString};
    let mut bases = record.seq().as_bytes();
    let (mut reference, mut query) = (record.pos(), 0usize);
    for op in cigar {
        match *op {
            Cigar::Match(len) | Cigar::Equal(len) | Cigar::Diff(len) => {
                for offset in 0..len as usize {
                    if bases.get(query + offset) == Some(&b'=')
                        && let Some(base) = usize::try_from(reference + offset as i64)
                            .ok()
                            .and_then(|at| contig.get(at))
                    {
                        bases[query + offset] = base.to_ascii_uppercase();
                    }
                }
                reference += i64::from(len);
                query += len as usize;
            }
            Cigar::Del(len) | Cigar::RefSkip(len) => reference += i64::from(len),
            Cigar::Ins(len) | Cigar::SoftClip(len) => query += len as usize,
            Cigar::HardClip(_) | Cigar::Pad(_) => {}
        }
    }
    let mut resolved = record.clone();
    resolved.set(record.qname(), Some(&CigarString(cigar.to_vec())), &bases, record.qual());
    resolved
}

/// Classifies a read using its literal SEQ bases. Annotation callers resolve '=' first, while
/// standalone pileup retains its original classification.
fn allele_of_sequence(
    site: &Site,
    record: &rust_htslib::bam::Record,
    cigar: &[rust_htslib::bam::record::Cigar],
    options: &PileupOptions,
    summed: bool,
) -> Option<ReadAllele> {
    let allele = match &site.kind {
        SiteKind::Snp(substitutions) => {
            substitutions.allele_of(record, cigar, options).map(ReadAllele::both)
        }
        SiteKind::Indel { first, last, flank, length, alts, same_length } => {
            let (first_query, last_query, observed) =
                anchored_length(cigar, record.pos(), *first, *last, *flank)?;
            let delta = observed - length;
            // The sums' allele, by the bases, of a read the counts read by its length.
            debug_assert!(!summed || alts.bases.is_some(), "the sums' site holds no bases");
            let by_bases = |counted: usize| -> ReadAllele {
                let summed = match alts.bases.as_ref().filter(|_| summed) {
                    Some(bases) => bases.summed(
                        record,
                        cigar,
                        (*first, *last),
                        (first_query, last_query),
                        delta,
                        options,
                    ),
                    None => counted,
                };
                ReadAllele { counted, summed }
            };
            if delta == 0 {
                match same_length.as_ref().map(|s| s.allele_of(record, cigar, options)) {
                    // A reference read has the reference's bases across the tract too (they hold
                    // the substituted bases).
                    Some(Some(REF)) | None => Some(by_bases(REF)),
                    // A same-length alternate's read has its bases across the tract too, with no
                    // indel among them.
                    Some(Some(ALT)) => {
                        let summed = match alts.bases.as_ref().filter(|_| summed) {
                            Some(bases) => bases.summed_same_length(
                                record,
                                cigar,
                                (*first, *last),
                                (first_query, last_query),
                                options,
                            ),
                            None => ALT,
                        };
                        Some(ReadAllele { counted: ALT, summed })
                    }
                    Some(allele) => allele.map(ReadAllele::both),
                }
            } else if !alts.has_delta(delta) {
                Some(ReadAllele::both(OTHER))
            } else {
                Some(by_bases(ALT))
            }
        }
    }?;
    // A read without bases has a length change, but no allele that its bases would tell.
    let unknown = summed && record.seq_len() == 0;
    Some(if unknown { ReadAllele { summed: OTHER, ..allele } } else { allele })
}

/// A record to measure reads at: its pileup site, its read annotations, or both.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Target {
    /// The site reads are counted at by allele, or `None` if the record's alleles cannot be read
    /// (see [`Site::of_record`]).
    site: Option<Site>,
    /// The read annotations to measure, or `None` if they are not measured.
    annotated: Option<Annotated>,
}

/// The read annotations of a record: which reads are counted, and how their alleles are read.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Annotated {
    /// The 0-based, half-open span a read must overlap to be counted: the record's REF with the
    /// settings' flank on each side, clipped to the physical contig.
    span: (i64, i64),
    /// How the alleles of the counted reads are read.
    alleles: AnnotatedAlleles,
}

/// How the allele sums of a record's read annotations read a read's allele. The sums are of the
/// reference and of the alternate alleles the genotype calls (see [`counts_allele`]: ReblockGVCF
/// keeps `AS_RAW_MQ` for these alleles only), where the pileup counts every alternate allele. A
/// read of an alternate allele not called is in neither sum (GATK gives it to that allele, which
/// ReblockGVCF drops), even where it has the reference's bases at the called alleles' positions.
/// A record without a pileup site (an indel record without a pure insertion or deletion allele)
/// has no sums, as a read of an alternate it cannot read could not be told apart from a called
/// one; so has a record for which no annotation window covers every alternate allele's change
/// (see [`annotation_tract`] and [`Site::reads_every_alt`]). The sums also check an indel read's bases across the tract where
/// the pileup counts read its length alone (see [`IndelAlleles`]).
#[derive(Debug, Clone, PartialEq, Eq)]
enum AnnotatedAlleles {
    /// Reuses the pileup site: every ALT is called and the selected annotation window matches.
    AsCounted,
    /// Uses an independently selected site with only called alternates (see [`Site::of_called`]);
    /// a read of an uncalled alternate is other. This may use a different window from the pileup.
    Called(Site),
    /// Not at all: the alleles cannot be read from a read.
    Unread,
}

impl Annotated {
    /// The read annotations of a record at 1-based `pos`, whose pileup site is `site`.
    fn of_record<S: AsRef<[u8]>>(
        contig: &[u8],
        pos: u64,
        reference: &[u8],
        alts: &[S],
        called: CalledAlts,
        site: Option<&Site>,
        options: &PileupOptions,
    ) -> Annotated {
        let (start, flank) = ((pos - 1) as i64, options.flank as i64);
        let span = (
            (start - flank).max(0),
            (start + reference.len() as i64 + flank).min(contig.len() as i64),
        );
        let n_alts = u32::try_from(alts.len()).unwrap_or(u32::MAX);
        let all_called = (1..=n_alts).all(|index| counts_allele(called, index));
        // Preserve the existing admission rule: both the record and its called alleles must
        // have a usable site. A window chosen from another ALT does not admit a new record type.
        let readable = site.is_some()
            && (all_called || {
                let counted: Vec<&[u8]> = (1..)
                    .zip(alts)
                    .filter(|&(index, _)| counts_allele(called, index))
                    .map(|(_, alt)| alt.as_ref())
                    .collect();
                Site::of_record(contig, pos, reference, &counted, options).is_some()
            });
        let by_site = |tract: Option<&RepeatTract>| {
            Site::of_called(
                contig,
                pos,
                reference,
                alts,
                |index| counts_allele(called, index),
                options,
                tract,
            )
            .filter(|site| site.reads_every_alt(contig, pos, reference, alts))
            .map_or(AnnotatedAlleles::Unread, AnnotatedAlleles::Called)
        };
        let alleles = match site.filter(|_| readable) {
            None => AnnotatedAlleles::Unread,
            Some(Site { kind: SiteKind::Indel { first, last, .. }, .. }) => {
                match annotation_tract(contig, pos, reference, alts, options) {
                    Some(tract)
                        if all_called
                            && tract_window(&tract, options.flank).0 == (*first, *last) =>
                    {
                        AnnotatedAlleles::AsCounted
                    }
                    Some(tract) => by_site(Some(&tract)),
                    None => AnnotatedAlleles::Unread,
                }
            }
            Some(_) if all_called => AnnotatedAlleles::AsCounted,
            Some(_) => by_site(None),
        };
        Annotated { span, alleles }
    }
}

impl Target {
    /// The target of a record at 1-based `pos`: its pileup site and, with the alternate alleles
    /// its genotype calls (`called`), its read annotations; or `None` if it has neither.
    ///
    /// The alleles' bases (see [`IndelBases`]) are held only by the site the annotations' sums
    /// read: the pileup site when all ALTs are called and its window is selected
    /// ([`AnnotatedAlleles::AsCounted`]), else the independent annotation site
    /// ([`AnnotatedAlleles::Called`]); the pileup counts read
    /// an indel read's length alone.
    fn of_record<S: AsRef<[u8]>>(
        contig: &[u8],
        pos: u64,
        reference: &[u8],
        alts: &[S],
        called: Option<CalledAlts>,
        options: &PileupOptions,
    ) -> Option<Target> {
        let site = Site::of_record(contig, pos, reference, alts, options);
        let Some(called) = called else {
            return site.map(|site| Target { site: Some(site), annotated: None });
        };
        let mut annotated =
            Annotated::of_record(contig, pos, reference, alts, called, site.as_ref(), options);
        let site = match site {
            Some(site) if annotated.alleles == AnnotatedAlleles::AsCounted => {
                let tract = tract_of_record(contig, pos, reference, alts);
                match Site::of_called(
                    contig,
                    pos,
                    reference,
                    alts,
                    |_| true,
                    options,
                    tract.as_ref(),
                ) {
                    Some(with_bases) => Some(with_bases),
                    None => {
                        annotated.alleles = AnnotatedAlleles::Unread;
                        Some(site)
                    }
                }
            }
            site => site,
        };
        Some(Target { site, annotated: Some(annotated) })
    }

    /// The 0-based, half-open span holding every read measured at the target.
    fn span(&self) -> (i64, i64) {
        match (self.site.as_ref().map(|site| site.span), self.annotated.as_ref().map(|a| a.span)) {
            (Some((beg, end)), Some((annotated_beg, annotated_end))) => {
                (beg.min(annotated_beg), end.max(annotated_end))
            }
            (Some(span), None) | (None, Some(span)) => span,
            (None, None) => unreachable!("a target has a site or read annotations"),
        }
    }
}

/// Measures reads at targets on one contig, reading targets at most `cluster_gap` apart from one
/// fetch: each read (see [`sweep_clusters`]) is counted at every site whose span it overlaps,
/// until the site is full (at most [`MAX_READS`] reads are counted at a site's alleles), and, with
/// `annotations`, in the read annotations of every target whose annotated span it overlaps (see
/// [`MapqSums`]). Overlapping mates are both counted.
///
/// A site's counts do not depend on whether the read annotations are measured.
fn measure_targets(
    reads: &mut rust_htslib::bam::IndexedReader,
    tid: u32,
    targets: &[Target],
    options: &PileupOptions,
    annotations: Option<&ReadAnnotationOptions>,
    contig: &[u8],
    cluster_gap: i64,
) -> Result<Vec<(Accumulator, MapqSums)>> {
    let spans: Vec<(i64, i64)> = targets.iter().map(Target::span).collect();
    let sums = |target: &Target| MapqSums {
        alleles_read: target
            .annotated
            .as_ref()
            .is_some_and(|annotated| annotated.alleles != AnnotatedAlleles::Unread),
        ..MapqSums::default()
    };
    let mut counting = Counting {
        targets,
        options,
        contig,
        min_mapq: annotations.map_or(0, ReadAnnotationOptions::min_mapq),
        max_annotation_reads: annotations
            .map_or(LEGACY_MAX_READS, ReadAnnotationOptions::max_reads),
        measured: targets.iter().map(|target| (Accumulator::default(), sums(target))).collect(),
    };
    sweep_clusters(reads, tid, &spans, cluster_gap, &mut counting)?;
    Ok(counting.measured)
}

/// The targets [`measure_targets`] measures reads at, with their counts and sums so far.
struct Counting<'a> {
    targets: &'a [Target],
    options: &'a PileupOptions,
    contig: &'a [u8],
    /// The MAPQ a read needs to count in the read annotations.
    min_mapq: u8,
    /// The independently configured annotation cap; pileup still uses its legacy limit.
    max_annotation_reads: u64,
    measured: Vec<(Accumulator, MapqSums)>,
}

/// Lazily decoded data for one read, reused at every overlapping target.
#[derive(Default)]
struct AlleleReadCache {
    cigar: Option<rust_htslib::bam::record::CigarStringView>,
    /// Uninitialized until annotations need the sequence; `None` means it contains no '='.
    resolved: std::cell::OnceCell<Option<rust_htslib::bam::Record>>,
}

impl SweepSites for Counting<'_> {
    type ReadCache = AlleleReadCache;

    /// A target stays open until its site and its sums are both full. Where most reads are below
    /// the annotations' MAPQ the sums fill slowly, and the target is offered every overlapping
    /// read; a read the full site and the sums both pass over costs no more than the checks at
    /// the top of [`Counting::add`] (no CIGAR is decoded and no allele read).
    fn is_full(&self, i: usize) -> bool {
        let (target, (accumulator, sums)) = (&self.targets[i], &self.measured[i]);
        (target.site.is_none() || accumulator.is_full())
            && (target.annotated.is_none() || sums.is_full(self.max_annotation_reads))
    }

    fn add(
        &mut self,
        i: usize,
        record: &rust_htslib::bam::Record,
        (start, end): (i64, i64),
        cache: &mut Self::ReadCache,
    ) {
        let cigar = &mut cache.cigar;
        let target = &self.targets[i];
        let (accumulator, sums) = &mut self.measured[i];
        let mapq = record.mapq();
        let overlaps = |(beg, span_end): (i64, i64)| start < span_end && end > beg;
        // The read overlaps the target's span, which is the site's own unless the read
        // annotations are measured too: each of the two spans is then checked, so that the site
        // counts the reads it would alone.
        let site =
            target.site.as_ref().filter(|site| target.annotated.is_none() || overlaps(site.span));
        let annotated = target.annotated.as_ref().filter(|annotated| {
            !sums.is_full(self.max_annotation_reads)
                && mapq >= self.min_mapq
                && mapq != UNAVAILABLE_MAPQ
                && overlaps(annotated.span)
                && record.cigar_len() > 0
                && !skips_span(
                    cigar.get_or_insert_with(|| record.cigar()),
                    start,
                    (start.max(annotated.span.0), end.min(annotated.span.1)),
                )
        });
        // The site counts the read at its alleles unless it is full, or this read fills it.
        let counted = site.is_some() && !accumulator.is_full() && {
            accumulator.n_reads += 1;
            !accumulator.is_full()
        };
        let sum_alleles = annotated.map(|annotated| {
            sums.add(mapq);
            &annotated.alleles
        });
        if record.cigar_len() == 0 {
            return;
        }
        let mut allele_at = |site: &Site, summed: bool| {
            let cigar = cigar.get_or_insert_with(|| record.cigar());
            let resolved = summed
                .then(|| {
                    cache.resolved.get_or_init(|| {
                        (0..record.seq_len())
                            .any(|query| record.seq()[query] == b'=')
                            .then(|| resolve_reference_bases(record, cigar, self.contig))
                    })
                })
                .and_then(Option::as_ref);
            allele_of(site, record, cigar, self.options, summed, resolved)
        };
        // The site's allele is read once, for the counts and (when every allele is called) the
        // sums.
        let as_counted = matches!(sum_alleles, Some(AnnotatedAlleles::AsCounted));
        let allele =
            site.filter(|_| counted || as_counted).and_then(|site| allele_at(site, as_counted));
        if let (true, Some(allele)) = (counted, allele) {
            accumulator.counts[allele.counted][self.options.bin(mapq)] += 1;
        }
        let sum_allele = match sum_alleles {
            Some(AnnotatedAlleles::AsCounted) => allele,
            Some(AnnotatedAlleles::Called(called)) => allele_at(called, true),
            Some(AnnotatedAlleles::Unread) | None => None,
        }
        .map(|allele| allele.summed);
        match sum_allele {
            Some(REF) => sums.add_allele(mapq, false),
            Some(ALT) => sums.add_allele(mapq, true),
            _ => {}
        }
    }
}

/// The htslib CRAM fields the pileup reads (FLAG, RNAME, POS, MAPQ, CIGAR, SEQ and QUAL), so CRAM
/// decoding can skip read names, mate fields and tags.
const CRAM_REQUIRED_FIELDS: u32 = 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x200 | 0x400;

/// The reads measured at one SNP or indel record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredSite {
    /// The record and its allele read counts (all 0 unless `counted`).
    pub row: PileupRow,
    /// True if the record's alleles can be read, so the row belongs in a pileup table: false for
    /// indel records without a pure insertion or deletion allele, which are measured only for
    /// their read annotations.
    pub counted: bool,
    /// The read annotations, when they are measured.
    pub mapq: Option<MapqSums>,
}

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
    // Without read annotations, every site measured is counted.
    let sites = measure_sites(inputs, regions, options, None, "counting the pileup")?;
    let mut timer = PhaseTimer::start();
    let rows = sites.into_iter().map(|site| site.row).collect();
    timer.lap_in("counting the pileup", "building the rows");
    Ok(rows)
}

/// Measures the read annotations (see [`MapqSums`]) of every SNP and indel record (any FILTER)
/// whose POS lies in each region, in region order, then VCF order. The VCF must have one sample,
/// whose genotype says which alternate alleles the allele sums are of.
///
/// Every SNP and indel record has a site. The read annotations count the reads
/// [`measure_pileup`] reads (mapped, primary, not duplicates, not supplementary, passing QC) that
/// have at least the annotations' MAPQ (and an available one, i.e. not 255) and overlap the
/// record's REF with the annotations' flank on each side.
///
/// # Errors
///
/// Returns an error if the reads cannot be measured (see [`measure_pileup`]).
pub fn measure_annotations<S: AsRef<str>>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
    annotations: &ReadAnnotationOptions,
) -> Result<Vec<MeasuredSite>> {
    let options = PileupOptions::reading_alleles_as(annotations);
    measure_sites(inputs, regions, &options, Some(annotations), "measuring the read annotations")
}

/// Measures the read annotations of every SNP and indel record in the regions (see
/// [`measure_annotations`]) and, with `pileup`, the pileup evidence of those settings (the rows
/// [`measure_pileup`] returns).
///
/// Both come from one pass over the reads when the pileup reads alleles as the annotations do
/// (see [`PileupOptions::reads_alleles_as`]): the rows are then those of the counted sites.
/// Otherwise the reads are read once for each.
///
/// # Errors
///
/// Returns an error if the reads cannot be measured (see [`measure_pileup`]).
pub fn measure_annotations_and_pileup<S: AsRef<str>>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
    annotations: &ReadAnnotationOptions,
    pileup: Option<PileupOptions>,
) -> Result<(Vec<MeasuredSite>, Option<PileupTables>)> {
    let Some(options) = pileup else {
        return Ok((measure_annotations(inputs, regions, annotations)?, None));
    };
    if options.reads_alleles_as(annotations) {
        let what = "measuring the read annotations and the pileup";
        let sites = measure_sites(inputs, regions, &options, Some(annotations), what)?;
        // These cloned rows are retained too; filtering otherwise loses the exact size hint.
        let mut rows = Vec::with_capacity(sites.iter().filter(|site| site.counted).count());
        rows.extend(sites.iter().filter(|site| site.counted).map(|site| site.row.clone()));
        return Ok((sites, Some(PileupTables::new(options, rows))));
    }
    let sites = measure_annotations(inputs, regions, annotations)?;
    // score reports this at startup (see `crate::model_set`).
    log::debug!(
        "The pileup settings {options} read alleles with another base quality or flank than the \
         alignment settings {annotations}: reading the reads again for the pileup evidence"
    );
    let rows = measure_pileup(inputs, regions, &options)?;
    Ok((sites, Some(PileupTables::new(options, rows))))
}

/// Measures the reads at every SNP and indel record (any FILTER) whose POS lies in each region,
/// in region order, then VCF order: the allele read counts [`measure_pileup`] returns and, with
/// `annotations`, the read annotations, from one pass over the reads.
///
/// Without `annotations`, the sites are those of [`measure_pileup`]. With them, every SNP and
/// indel record has a site: an indel record without a pure insertion or deletion allele has only
/// its read annotations (its row's counts are 0 and it is not `counted`). The rows of the counted
/// sites are the same either way.
///
/// `annotations` must read alleles as `options` do (the callers in this module see to it): the
/// reads' alleles are read with `options`.
///
/// The pass is timed at debug level as `what` (what the caller measures with it).
fn measure_sites<S: AsRef<str>>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
    options: &PileupOptions,
    annotations: Option<&ReadAnnotationOptions>,
    what: &str,
) -> Result<Vec<MeasuredSite>> {
    use rust_htslib::bcf::Read as _;
    debug_assert!(
        annotations.is_none_or(|annotations| options.reads_alleles_as(annotations)),
        "the alignment settings {annotations:?} do not read alleles as {options} does"
    );
    let mut timer = PhaseTimer::start();
    let per_shard = run_shards(inputs, regions, CRAM_REQUIRED_FIELDS, |shard| {
        let mut pending: Vec<(PileupRow, Target)> = Vec::new();
        if !shard.fetch_records(inputs.vcf)? {
            return Ok(Vec::new());
        }
        let mut record = shard.vcf.empty_record();
        let mut fields = crate::vcf::FieldReader::new();
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
            // A single alternate allele is counted whatever the genotype.
            let called = annotations.map(|_| {
                called_alts(if alts.len() > 1 {
                    fields.genotype(&record)
                } else {
                    Genotype::MISSING
                })
            });
            let Some(target) =
                Target::of_record(shard.contig, pos, reference, alts, called, options)
            else {
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
            pending.push((row, target));
        }
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let tid = shard.tid(inputs.alignments)?;
        let (rows, targets): (Vec<PileupRow>, Vec<Target>) = pending.into_iter().unzip();
        let measured = measure_targets(
            shard.reads,
            tid,
            &targets,
            options,
            annotations,
            shard.contig,
            shard.cluster_gap,
        )
        .with_context(|| {
            format!("failed to count reads on {}:{}-{}", shard.chrom, shard.start, shard.end)
        })?;
        Ok(rows
            .into_iter()
            .zip(targets)
            .zip(measured)
            .map(|((row, target), (accumulator, sums))| MeasuredSite {
                row: PileupRow { n_reads: accumulator.n_reads, counts: accumulator.counts, ..row },
                counted: target.site.is_some(),
                mapq: annotations.map(|_| sums),
            })
            .collect())
    })?;
    timer.lap_in(what, "the shards");
    // These sites stay live alongside the calls and feature table. Flattening cannot hint at
    // every shard's length, so preallocate instead of retaining geometric spare capacity.
    let mut sites = Vec::with_capacity(per_shard.iter().map(Vec::len).sum());
    for shard in per_shard {
        sites.extend(shard);
    }
    timer.lap_in(what, "collecting the sites");
    Ok(sites)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::evidence::read_tables;
    use crate::model_evidence::check_options;
    use crate::read_pass::CLUSTER_GAP;
    use crate::testing::{ReadFiles, TestRead, write_bam};

    /// Retained measurements must not carry geometric spare capacity alongside the feature
    /// table. Allow modest slack rather than requiring an allocator's exact capacity.
    #[rstest::rstest]
    #[case::all_counted(false)]
    #[case::mixed(true)]
    fn measured_vectors_do_not_retain_geometric_spare_capacity(#[case] mixed: bool) {
        let dir = TempDir::new().unwrap();
        let mut sequence = vec![b'A'; 200];
        let mut records = Vec::new();
        for pos in (10..=110).step_by(10) {
            records.push(format!("chr1\t{pos}\t.\tA\tC\t50\tPASS\t.\tGT\t0/1"));
            if mixed {
                // Complex indels retain annotations but cannot contribute a pileup row.
                sequence[pos + 3..pos + 5].copy_from_slice(b"CT");
                records.push(format!("chr1\t{}\t.\tCT\tGTT\t50\tPASS\t.\tGT\t0/1", pos + 4));
            }
        }
        let sequence = String::from_utf8(sequence).unwrap();
        let lines: Vec<&str> = records.iter().map(String::as_str).collect();
        let fixture = ReadFiles::write(dir.path(), &[("chr1", &sequence)], &lines, &[]);
        let annotations = ReadAnnotationOptions::default();
        let regions = ["chr1:1-40", "chr1:41-80", "chr1:81-200"];
        let (sites, tables) = measure_annotations_and_pileup(
            &fixture.inputs(),
            &regions,
            &annotations,
            Some(PileupOptions::default()),
        )
        .unwrap();
        let expected_positions: &[u64] = if mixed {
            &[
                10, 14, 20, 24, 30, 34, 40, 44, 50, 54, 60, 64, 70, 74, 80, 84, 90, 94, 100, 104,
                110, 114,
            ]
        } else {
            &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        };
        let positions: Vec<u64> = sites.iter().map(|site| site.row.pos).collect();
        assert_eq!(positions, expected_positions);
        for site in &sites {
            let counted = site.row.pos % 10 == 0;
            assert_eq!(site.counted, counted);
            assert_eq!(site.row.chrom, "chr1");
            assert_eq!(site.row.reference, if counted { "A" } else { "CT" });
            assert_eq!(site.row.alt, if counted { "C" } else { "GTT" });
        }
        let rows = tables.unwrap().rows;
        let expected_rows: Vec<PileupRow> = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
            .into_iter()
            .map(|pos| PileupRow {
                chrom: "chr1".into(),
                pos,
                reference: "A".into(),
                alt: "C".into(),
                n_reads: 0,
                counts: [[0; 3]; 3],
            })
            .collect();
        assert_eq!(rows, expected_rows);
        // A quarter of the live rows allows slack while catching geometric growth and, in the
        // mixed case, allocating the filtered rows for all sites rather than just counted sites.
        assert!(sites.capacity() <= sites.len() * 5 / 4, "{} site slots", sites.capacity());
        assert!(rows.capacity() <= rows.len() * 5 / 4, "{} row slots", rows.capacity());
    }

    /// The reads counted at sites alone, without read annotations.
    fn count_sites(
        reads: &mut rust_htslib::bam::IndexedReader,
        tid: u32,
        sites: &[Site],
        options: &PileupOptions,
    ) -> Result<Vec<Accumulator>> {
        let targets: Vec<Target> =
            sites.iter().map(|site| Target { site: Some(site.clone()), annotated: None }).collect();
        let measured = measure_targets(reads, tid, &targets, options, None, &[], CLUSTER_GAP)?;
        Ok(measured.into_iter().map(|(accumulator, _)| accumulator).collect())
    }

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
        assert_eq!(write_pileup_table(&path, &rows, options, None).unwrap(), 2);
        let other = dir.path().join("q.tsv");
        write_pileup_table(&other, &rows[..1], options, None).unwrap();
        let tables = read_tables::<PileupOptions>(&[&path, &other]).unwrap();
        assert_eq!(tables, PileupTables::new(options, rows.clone()));

        let mixed = dir.path().join("mixed.tsv");
        write_pileup_table(&mixed, &rows, "mapq30".parse().unwrap(), None).unwrap();
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

    /// The reads' alignment provenance is recorded in its own line, which the rows' reading skips
    /// and older tables lack; the tables given together must agree on it.
    #[test]
    fn the_alignment_provenance_is_recorded() {
        use crate::evidence::read_alignment;
        let dir = TempDir::new().unwrap();
        let rows = [PileupRow {
            chrom: "chr1".into(),
            pos: 10,
            reference: "A".into(),
            alt: "G".into(),
            n_reads: 1,
            counts: [[1, 0, 0], [0; 3], [0; 3]],
        }];
        let options = PileupOptions::default();
        let provenance = crate::testing::bwa_provenance;
        let primary = provenance("primary");
        let path = dir.path().join("p.tsv.gz");
        write_pileup_table(&path, &rows, options, Some(&primary)).unwrap();
        let unknown = dir.path().join("u.tsv");
        write_pileup_table(&unknown, &rows, options, None).unwrap();
        let tables = read_tables::<PileupOptions>(&[&path, &unknown]).unwrap();
        assert_eq!(tables.rows, rows);
        assert_eq!(tables.alignment.as_ref(), Some(&primary));
        assert_eq!(read_alignment(&[&path, &unknown]).unwrap(), Some(primary));
        assert_eq!(read_alignment(&[&unknown]).unwrap(), None);
        assert_eq!(read_tables::<PileupOptions>(&[&unknown]).unwrap().alignment, None);
        let other = dir.path().join("o.tsv");
        write_pileup_table(&other, &rows, options, Some(&provenance("keep-alt"))).unwrap();
        for message in [
            read_alignment(&[&path, &other]).unwrap_err().to_string(),
            read_tables::<PileupOptions>(&[&path, &other]).unwrap_err().to_string(),
        ] {
            assert!(message.contains("were aligned differently: alt-project: "), "{message}");
            assert!(message.contains("o.tsv keep-alt. "), "{message}");
        }
        // A malformed line names the table.
        let bad = dir.path().join("bad.tsv");
        std::fs::write(&bad, format!("{SETTINGS_PREFIX}default\n##alignment=[1]\n{HEADER}\n"))
            .unwrap();
        for error in [
            read_alignment(&[&bad]).unwrap_err(),
            read_tables::<PileupOptions>(&[&bad]).unwrap_err(),
        ] {
            let message = error.to_string();
            assert!(message.contains("bad.tsv has a malformed ##alignment= line"), "{message}");
        }
    }

    /// A table that is not a regular file (e.g. a pipe) is not read for its provenance before
    /// scoring, so that it is not consumed. Opening this pipe, which has no writer, would block,
    /// so the read runs in a child under a deadline: a regression fails rather than hangs.
    #[cfg(unix)]
    #[test]
    fn a_pipe_is_not_read_for_its_provenance() {
        let dir = TempDir::new().unwrap();
        let fifo = dir.path().join("pipe.tsv");
        let status = std::process::Command::new("mkfifo").arg(&fifo).status().unwrap();
        assert!(status.success());
        let log = dir.path().join("child.log");
        let mut processes = crate::testing::TestProcesses::default();
        let envs = [("WINNOWR_PILEUP_FIFO", &fifo)];
        processes.spawn_test("pileup::tests::fifo_provenance_child", envs, &log);
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    /// Child of [`a_pipe_is_not_read_for_its_provenance`].
    #[cfg(unix)]
    #[test]
    fn fifo_provenance_child() {
        let Some(fifo) = std::env::var_os("WINNOWR_PILEUP_FIFO") else { return };
        assert_eq!(crate::evidence::read_alignment(&[Path::new(&fifo)]).unwrap(), None);
        crate::testing::child_body_ran();
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
        TestRead { name, pos, cigar, flag: 0, mapq, qual: 30, quals: None, bases: None }
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
    fn counts_alleles_substituted_at_more_bases_than_fit_on_the_stack() {
        let dir = TempDir::new().unwrap();
        let options = PileupOptions::default();
        // Ten substituted bases at 0-based [122, 132), more than the 8 a read's bases hold on the
        // stack: the alternate changes every one.
        let positions: Vec<usize> = (122..132).collect();
        let change = |base: u8| match base {
            b'A' => 'C',
            b'C' => 'G',
            b'G' => 'T',
            _ => 'A',
        };
        let alt: Vec<(usize, char)> =
            positions.iter().map(|&at| (at, change(reference().as_bytes()[at]))).collect();
        let reads: Vec<(&str, String)> = vec![
            ("ref", bases_with(90, 160, &[])),
            ("alt", bases_with(90, 160, &alt)),
            // The alternate but for its last base.
            ("other", bases_with(90, 160, &alt[..alt.len() - 1])),
        ];
        let reads: Vec<TestRead<'_>> = reads
            .iter()
            .map(|(name, bases)| TestRead { bases: Some(bases), ..read(name, 90, "70M", 60) })
            .collect();
        let bam = write_bam(dir.path(), "reads.bam", reference().len() as u64, &reads);
        let substitutions = Substitutions {
            positions: positions.iter().map(|&at| at as i64).collect(),
            reference: reference().as_bytes()[122..132].to_vec(),
            alts: vec![alt.iter().map(|&(_, base)| base as u8).collect()],
        };
        let span = substitutions.span();
        let site = Site { kind: SiteKind::Snp(substitutions), span };
        let mut reader = rust_htslib::bam::IndexedReader::from_path(&bam).unwrap();
        let counted = count_sites(&mut reader, 0, &[site], &options).unwrap();
        assert_eq!(counted[0].n_reads, 3);
        assert_eq!(counted[0].counts, [[0, 0, 1], [0, 0, 1], [0, 0, 1]]);
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
        let deltas = |alts: &IndelAlleles| -> Vec<i64> {
            // A pileup-only site holds no bases.
            assert_eq!(alts.bases, None);
            alts.deltas.clone()
        };
        let Some(Site { kind: SiteKind::Indel { alts, length, same_length, .. }, .. }) =
            Site::of_record(contig, 80, b"CAA", &["C", "CA", "CAAA"], &options)
        else {
            panic!("not an indel")
        };
        assert_eq!((deltas(&alts), length, same_length), (vec![-2, -1, 1], 10, None));
        // An indel record with a same-length alternate (as HaplotypeCaller writes at 1/2 sites).
        for (reference, alts) in [(&b"CA"[..], ["C", "TA"]), (b"C", ["T", "CA"])] {
            let Some(Site { kind: SiteKind::Indel { alts, same_length, .. }, span }) =
                Site::of_record(contig, 80, reference, &alts, &options)
            else {
                panic!("not an indel")
            };
            let delta = if reference.len() == 2 { -1 } else { 1 };
            assert_eq!(deltas(&alts), vec![delta]);
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
        let inputs = ReadInputs::new(&vcf, &bam, &fasta);
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
        write_pileup_table(&path, &rows, options, None).unwrap();
        assert_eq!(
            read_tables::<PileupOptions>(&[&path]).unwrap(),
            PileupTables::new(options, rows)
        );
    }

    /// Writes the read files of a VCF of `records` on [`reference`] and the reads.
    fn fixture(dir: &Path, records: &[&str], reads: &[TestRead<'_>]) -> ReadFiles {
        fixture_on(dir, &reference(), records, reads)
    }

    /// [`fixture`] on chr1 of `reference`.
    fn fixture_on(
        dir: &Path,
        reference: &str,
        records: &[&str],
        reads: &[TestRead<'_>],
    ) -> ReadFiles {
        ReadFiles::write(dir, &[("chr1", reference)], records, reads)
    }

    /// A deletion in the A10 tract, an indel record without a pure insertion or deletion allele,
    /// and the SNP at 0-based 122.
    const RECORDS: [&str; 3] = [
        "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1",
        "chr1\t100\t.\tCT\tGTT\t50\tPASS\t.\tGT\t0/1",
        "chr1\t123\t.\tG\tT\t50\tPASS\t.\tGT\t0/1",
    ];

    /// A read over the SNP at 0-based 122: its name, base there, MAPQ, flag and base quality.
    type SnpRead = (&'static str, char, u8, u16, u8);

    /// Reads over the SNP of [`RECORDS`].
    const SNP_READS: [SnpRead; 11] = [
        ("r0", 'G', 0, 0, 30),
        ("r10", 'G', 10, 0, 30),
        ("r60", 'G', 60, 0, 30),
        ("a60a", 'T', 60, 0, 30),
        ("a60b", 'T', 60, 0x10, 30),
        ("a20", 'T', 20, 0, 30),
        ("other", 'C', 30, 0, 30),
        ("lowbq", 'T', 60, 0, 5),
        ("duplicate", 'T', 60, 0x400, 30),
        ("supplementary", 'G', 60, 0x800, 30),
        ("unavailable", 'G', 255, 0, 30),
    ];

    /// The bases of each of `reads` (see [`snp_bases`]), which [`snp_reads`] borrows.
    fn bases_of(reads: &[SnpRead]) -> Vec<String> {
        reads.iter().map(|read| snp_bases(read.1)).collect()
    }

    /// `reads` as 70-base reads from 0-based 90, with their `bases` (see [`bases_of`]).
    fn snp_reads<'a>(reads: &[SnpRead], bases: &'a [String]) -> Vec<TestRead<'a>> {
        reads
            .iter()
            .zip(bases)
            .map(|(&(name, _, mapq, flag, qual), bases)| TestRead {
                flag,
                qual,
                bases: Some(bases),
                ..read(name, 90, "70M", mapq)
            })
            .collect()
    }

    /// The position and sums of each site.
    fn sums_of(sites: &[MeasuredSite]) -> Vec<(u64, Option<MapqSums>)> {
        sites.iter().map(|site| (site.row.pos, site.mapq)).collect()
    }

    /// The rows of the counted sites.
    fn counted_rows(sites: &[MeasuredSite]) -> Vec<PileupRow> {
        sites.iter().filter(|site| site.counted).map(|site| site.row.clone()).collect()
    }

    /// Measures [`RECORDS`] with [`SNP_READS`] and reads beside the SNP, with read annotations
    /// in the pileup's pass; returns the sites and the rows of the pileup alone.
    fn measure_annotated(
        options: &PileupOptions,
        annotations: &ReadAnnotationOptions,
    ) -> (Vec<MeasuredSite>, Vec<PileupRow>) {
        let dir = TempDir::new().unwrap();
        let bases = bases_of(&SNP_READS);
        let mut reads = snp_reads(&SNP_READS, &bases);
        // Reads beside the SNP at 0-based 122: their last (or first) base is 5 or 6 bases away.
        // They have the reference's bases.
        let contig = reference();
        let beside: Vec<(&str, i64, String, String, u8)> = [
            ("left6", 57, 60, 60),
            ("left5", 58, 60, 40),
            ("right5", 127, 30, 40),
            ("right6", 128, 30, 60),
        ]
        .into_iter()
        .map(|(name, pos, len, mapq): (&str, usize, usize, u8)| {
            (name, pos as i64, format!("{len}M"), contig[pos..pos + len].to_string(), mapq)
        })
        .collect();
        reads.extend(beside.iter().map(|(name, pos, cigar, bases, mapq)| TestRead {
            bases: Some(bases),
            ..read(name, *pos, cigar, *mapq)
        }));
        let fixture = fixture(dir.path(), &RECORDS, &reads);
        let sites = measure_sites(&fixture.inputs(), &["chr1"], options, Some(annotations), "test")
            .unwrap();
        let rows = measure_pileup(&fixture.inputs(), &["chr1"], options).unwrap();
        (sites, rows)
    }

    #[test]
    fn measures_mapq_sums_of_the_reads_over_each_record() {
        let (options, annotations) = (PileupOptions::default(), ReadAnnotationOptions::default());
        let contig = reference();
        assert!(Site::of_record(contig.as_bytes(), 100, b"CT", &["GTT"], &options).is_none());
        let (sites, _) = measure_annotated(&options, &annotations);
        let summary: Vec<(u64, bool, Option<MapqSums>)> =
            sites.iter().map(|site| (site.row.pos, site.counted, site.mapq)).collect();
        let sq = |mapq: u64| mapq * mapq;
        assert_eq!(
            summary,
            [
                // Only the reads left of the SNP reach the deletion, and span its tract.
                (
                    80,
                    true,
                    Some(MapqSums {
                        n_reads: 2,
                        sum_sq: sq(60) + sq(40),
                        ref_sum_sq: sq(60) + sq(40),
                        alt_sum_sq: 0,
                        alleles_read: true,
                    })
                ),
                // Totals only: the reads over the SNP with MAPQ from 20, and those left of it.
                (
                    100,
                    false,
                    Some(MapqSums {
                        n_reads: 8,
                        sum_sq: 4 * sq(60) + sq(20) + sq(30) + sq(60) + sq(40),
                        ref_sum_sq: 0,
                        alt_sum_sq: 0,
                        alleles_read: false,
                    })
                ),
                // Not the reads below MAPQ 20, the duplicate, the supplementary alignment, the
                // read without a MAPQ, or the reads ending 6 bases away. The read with another
                // base and the one with a low-quality base are in neither allele's sum.
                (
                    123,
                    true,
                    Some(MapqSums {
                        n_reads: 8,
                        sum_sq: 4 * sq(60) + sq(20) + sq(30) + 2 * sq(40),
                        ref_sum_sq: sq(60),
                        alt_sum_sq: 2 * sq(60) + sq(20),
                        alleles_read: true,
                    })
                ),
            ]
        );
        // The record whose alleles cannot be read has no counts; the SNP's are the pileup's, at
        // any MAPQ and over the SNP alone.
        assert_eq!((sites[1].row.n_reads, sites[1].row.counts), (0, [[0; 3]; 3]));
        assert_eq!(sites[2].row.n_reads, 9);
        assert_eq!(sites[2].row.counts, [[1, 1, 2], [0, 0, 3], [0, 0, 1]]);
    }

    #[rstest::rstest]
    #[case::defaults("default", "default")]
    #[case::any_mapq("default", "mapq0")]
    #[case::other_bins_and_mapq("mapq30", "mapq1")]
    #[case::narrow_flank("flank2", "flank2")]
    #[case::wide_flank_and_any_base("bq0,flank40", "mapq10,bq0,flank40")]
    fn read_annotations_leave_the_pileup_rows_as_they_are(
        #[case] options: PileupOptions,
        #[case] annotations: ReadAnnotationOptions,
    ) {
        let (sites, rows) = measure_annotated(&options, &annotations);
        assert_eq!(rows.iter().map(|row| row.pos).collect::<Vec<_>>(), [80, 123]);
        assert_eq!(counted_rows(&sites), rows);
        assert_eq!(sites.len(), 3);
        assert!(sites.iter().all(|site| site.mapq.is_some()));
    }

    #[test]
    fn sites_without_read_annotations_are_the_pileup_rows() {
        let dir = TempDir::new().unwrap();
        let fixture = fixture(dir.path(), &RECORDS, &[read("r", 90, "70M", 60)]);
        let options = PileupOptions::default();
        let sites = measure_sites(&fixture.inputs(), &["chr1"], &options, None, "test").unwrap();
        let summary: Vec<(u64, bool, Option<MapqSums>)> =
            sites.iter().map(|site| (site.row.pos, site.counted, site.mapq)).collect();
        assert_eq!(summary, [(80, true, None), (123, true, None)]);
    }

    #[test]
    fn pileup_settings_say_whether_they_read_alleles_as_annotations_do() {
        let annotations: ReadAnnotationOptions = "mapq1,bq7,flank3".parse().unwrap();
        let pileup = PileupOptions::reading_alleles_as(&annotations);
        assert_eq!(pileup, "mapq20,bq7,flank3".parse().unwrap());
        assert!(pileup.reads_alleles_as(&annotations));
        // The MAPQs do not matter (they mean different things); the base quality and flank do.
        let reads_alike = |settings: &str| {
            settings.parse::<PileupOptions>().unwrap().reads_alleles_as(&annotations)
        };
        assert!(reads_alike("mapq30,bq7,flank3"));
        assert!(!reads_alike("bq10,flank3"));
        assert!(!reads_alike("bq7,flank5"));
    }

    /// The sums' alternate allele is the one the genotype calls, where the pileup counts a read
    /// with any alternate allele of the record.
    #[rstest::rstest]
    #[case::first_alt_called("0/1", [1, 1, 0])]
    #[case::second_alt_called("0/2", [1, 0, 1])]
    #[case::hom_alt("2/2", [1, 0, 1])]
    #[case::both_called("1/2", [1, 1, 1])]
    #[case::hom_ref_counts_every_alt("0/0", [1, 1, 1])]
    #[case::missing_counts_every_alt("./.", [1, 1, 1])]
    fn allele_sums_are_of_the_called_alternate_alleles(
        #[case] genotype: &str,
        #[case] in_sums: [u64; 3],
    ) {
        let dir = TempDir::new().unwrap();
        // The reference base, each alternate allele's, and another, at MAPQs that tell them
        // apart in the sums.
        let reads: [SnpRead; 4] = [
            ("ref", 'G', 20, 0, 30),
            ("alt1", 'T', 30, 0, 30),
            ("alt2", 'C', 40, 0, 30),
            ("other", 'A', 50, 0, 30),
        ];
        let bases = bases_of(&reads);
        let record = format!("chr1\t123\t.\tG\tT,C\t50\tPASS\t.\tGT\t{genotype}");
        let fixture = fixture(dir.path(), &[&record], &snp_reads(&reads, &bases));
        let annotations = ReadAnnotationOptions::default();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &annotations).unwrap();
        let [site] = sites.as_slice() else { panic!("{sites:?}") };
        // The pileup counts both alternate alleles, whatever the genotype.
        assert_eq!(site.row.counts, [[0, 0, 1], [0, 0, 2], [0, 0, 1]]);
        let [in_ref, in_alt1, in_alt2] = in_sums;
        assert_eq!(
            site.mapq,
            Some(MapqSums {
                n_reads: 4,
                sum_sq: 400 + 900 + 1600 + 2500,
                ref_sum_sq: in_ref * 400,
                alt_sum_sq: in_alt1 * 900 + in_alt2 * 1600,
                alleles_read: true,
            })
        );
    }

    /// The allele sums of one record measured with `reads`, each its name, 0-based position,
    /// CIGAR, bases and MAPQ.
    fn sums_at(record: &str, reads: &[(&str, i64, &str, &str, u8)]) -> (MapqSums, AlleleCounts) {
        sums_on(&reference(), record, reads)
    }

    /// [`sums_at`] on the `reference` sequence: the sums, and the pileup row's counts.
    fn sums_on(
        reference: &str,
        record: &str,
        reads: &[(&str, i64, &str, &str, u8)],
    ) -> (MapqSums, AlleleCounts) {
        let reads: Vec<TestRead<'_>> = reads
            .iter()
            .map(|&(name, pos, cigar, bases, mapq)| TestRead {
                bases: Some(bases),
                ..read(name, pos, cigar, mapq)
            })
            .collect();
        sums_with(reference, record, &reads)
    }

    /// [`sums_on`] with the reads as given.
    fn sums_with(
        reference: &str,
        record: &str,
        reads: &[TestRead<'_>],
    ) -> (MapqSums, AlleleCounts) {
        let dir = TempDir::new().unwrap();
        let fixture = fixture_on(dir.path(), reference, &[record], reads);
        let annotations = ReadAnnotationOptions::default();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &annotations).unwrap();
        let [site] = sites.as_slice() else { panic!("{sites:?}") };
        (site.mapq.unwrap(), site.row.counts)
    }

    /// The reference from 0-based `start` to `end`, with `edits` (0-based position, base)
    /// substituted.
    fn edited(start: usize, end: usize, edits: &[(usize, u8)]) -> String {
        let mut bases = reference().as_bytes()[start..end].to_vec();
        for &(at, base) in edits {
            bases[at - start] = base;
        }
        String::from_utf8(bases).unwrap()
    }

    /// A read with an uncalled alternate allele is in neither allele's sum, though it shares the
    /// reference's bases where the called alleles differ: in `GC>TC,GA` called `0/1`, a `GA`
    /// read has the reference's `G` (GATK gives it to `GA`, which ReblockGVCF drops).
    #[test]
    fn reads_of_an_uncalled_alternate_are_in_neither_sum() {
        let record = "chr1\t123\t.\tGC\tTC,GA\t50\tPASS\t.\tGT\t0/1";
        let (gc, tc, ga) = (
            edited(90, 160, &[]),
            edited(90, 160, &[(122, b'T')]),
            edited(90, 160, &[(123, b'A')]),
        );
        let (sums, counts) = sums_at(
            record,
            &[("gc", 90, "70M", &gc, 20), ("tc", 90, "70M", &tc, 30), ("ga", 90, "70M", &ga, 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (3, 400, 900));
        assert!(sums.alleles_read);
        // The pileup counts every alternate allele: `GA` too.
        assert_eq!(counts, [[0, 0, 1], [0, 0, 2], [0; 3]]);
    }

    /// The alleles of a record whose called alternate is of another kind than the record's are
    /// read as the record's are: in `A>T,AT` at the A10 tract, a read inserting `T` is not the
    /// reference when only `T` is called, and a `T` read is not the reference when only `AT` is.
    #[rstest::rstest]
    #[case::snp_called("0/1", 400, 900)]
    #[case::insertion_called("0/2", 400, 1600)]
    fn called_alternates_of_another_kind_than_the_record(
        #[case] genotype: &str,
        #[case] ref_sum_sq: u64,
        #[case] alt_sum_sq: u64,
    ) {
        let record = format!("chr1\t81\t.\tA\tT,AT\t50\tPASS\t.\tGT\t{genotype}");
        let (same, t) = (edited(50, 120, &[]), edited(50, 120, &[(80, b'T')]));
        let insertion = format!("{}T{}", edited(50, 81, &[]), edited(81, 120, &[]));
        let (sums, counts) = sums_at(
            &record,
            &[
                ("ref", 50, "70M", &same, 20),
                ("t", 50, "70M", &t, 30),
                ("ins", 50, "31M1I39M", &insertion, 40),
            ],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (3, ref_sum_sq, alt_sum_sq));
        assert!(sums.alleles_read);
        // The pileup counts both alternate alleles, whatever the genotype.
        assert_eq!(counts, [[0, 0, 1], [0, 0, 2], [0; 3]]);
    }

    /// The reference from 0-based 50 to 120 with `inserted` inserted before 0-based `at`.
    fn inserted(at: usize, inserted: &str) -> String {
        format!("{}{inserted}{}", edited(50, at, &[]), edited(at, 120, &[]))
    }

    /// A read of a called insertion's length is in the ALT sum only if it has the inserted bases
    /// across the tract, where the pileup counts it by its length: in `A>T,AT` called `0/2`, and
    /// in `A>AT` called `0/1`, a read inserting `A` (`AA`, not `AT`), or a `T` further along the
    /// A10 tract, counts towards the depth only.
    #[rstest::rstest]
    #[case::with_a_snp("chr1\t81\t.\tA\tT,AT\t50\tPASS\t.\tGT\t0/2")]
    #[case::alone("chr1\t81\t.\tA\tAT\t50\tPASS\t.\tGT\t0/1")]
    fn a_called_insertion_is_matched_by_its_bases(#[case] record: &str) {
        let same = edited(50, 120, &[]);
        let (t, a, t_later) = (inserted(81, "T"), inserted(81, "A"), inserted(84, "T"));
        let (sums, counts) = sums_at(
            record,
            &[
                ("ref", 50, "70M", &same, 20),
                ("t", 50, "31M1I39M", &t, 30),
                ("a", 50, "31M1I39M", &a, 40),
                ("t_later", 50, "34M1I36M", &t_later, 50),
            ],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (4, 400, 900));
        assert!(sums.alleles_read);
        // The pileup counts every read of the insertion's length.
        assert_eq!(counts, [[0, 0, 1], [0, 0, 3], [0; 3]]);
    }

    /// A read's bases below the base quality, and its `N`s, match any of the allele's; its other
    /// bases are compared. In `A>T,AT` called `0/2`, the inserted base alone is low quality or
    /// `N`, and a read whose high-quality base before it is not the allele's is in neither sum.
    #[test]
    fn low_quality_and_n_bases_match_any_allele_base() {
        let record = "chr1\t81\t.\tA\tT,AT\t50\tPASS\t.\tGT\t0/2";
        let (a, n) = (inserted(81, "A"), inserted(81, "N"));
        // The padding base (0-based 80) is not the allele's `A`.
        let mismatched = format!("{}A{}", edited(50, 81, &[(80, b'G')]), edited(81, 120, &[]));
        // Only the inserted base (query position 31) is below the base quality.
        let mut quals = vec![30; a.len()];
        quals[31] = 5;
        let reads = [
            TestRead {
                bases: Some(&a),
                quals: Some(&quals),
                ..read("low_quality", 50, "31M1I39M", 30)
            },
            TestRead { bases: Some(&n), ..read("n", 50, "31M1I39M", 40) },
            TestRead { bases: Some(&a), ..read("a", 50, "31M1I39M", 50) },
            TestRead {
                bases: Some(&mismatched),
                quals: Some(&quals),
                ..read("mismatched", 50, "31M1I39M", 60)
            },
        ];
        let (sums, counts) = sums_with(&reference(), record, &reads);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (4, 0, 900 + 1600));
        assert_eq!(counts, [[0; 3], [0, 0, 4], [0; 3]]);
    }

    /// A called allele's bases are cut to the anchors where the record's REF reaches past them:
    /// in `C G7 A4 > C G7 A3` the REF starts more than the flank before the A4 tract, and in
    /// `C A4 G10 > C A3 G10` it ends more than the flank after it. A read of the called deletion
    /// is in the ALT sum; a read of its length with another base inside the anchors is in
    /// neither.
    #[rstest::rstest]
    #[case::ref_before_the_tract(
        format!("{}CGGGGGGGAAAA{}", "CGTC".repeat(20), "CTCG".repeat(20)),
        "CGGGGGGGAAAA",
        "CGGGGGGGAAA",
        88,
        86
    )]
    #[case::ref_after_the_tract(
        format!("{}CAAAAGGGGGGGGGG{}", "CGTC".repeat(20), "CATG".repeat(20)),
        "CAAAAGGGGGGGGGG",
        "CAAAGGGGGGGGGG",
        81,
        87
    )]
    fn called_bases_are_cut_to_the_anchors(
        #[case] reference: String,
        #[case] ref_allele: &str,
        #[case] alt_allele: &str,
        #[case] deleted: usize,
        #[case] mismatch: usize,
    ) {
        let record = format!("chr1\t81\t.\t{ref_allele}\t{alt_allele}\t50\tPASS\t.\tGT\t0/1");
        let with_deletion = |edit: Option<usize>| -> String {
            let mut bases = reference.as_bytes()[50..120].to_vec();
            if let Some(at) = edit {
                bases[at - 50] = b'T';
            }
            bases.remove(deleted - 50);
            String::from_utf8(bases).unwrap()
        };
        let (called, other) = (with_deletion(None), with_deletion(Some(mismatch)));
        let cigar = format!("{}M1D{}M", deleted - 50, 120 - deleted - 1);
        let (sums, counts) = sums_on(
            &reference,
            &record,
            &[("called", 50, &cigar, &called, 30), ("other", 50, &cigar, &other, 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900));
        assert_eq!(counts, [[0; 3], [0, 0, 2], [0; 3]]);
    }

    /// A complex allele with a length change is matched by its bases too: in `CA>C,GTT` called
    /// `1/2`, a read of `GTT` is in the ALT sum, and a read inserting another base is in neither.
    #[test]
    fn a_complex_allele_is_matched_by_its_bases() {
        let record = "chr1\t80\t.\tCA\tC,GTT\t50\tPASS\t.\tGT\t1/2";
        let gtt = format!("{}GTT{}", edited(50, 79, &[]), edited(81, 120, &[]));
        let (sums, counts) = sums_at(
            record,
            &[("gtt", 50, "31M1I39M", &gtt, 30), ("c", 50, "31M1I39M", &inserted(81, "C"), 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900));
        assert_eq!(counts, [[0; 3], [0, 0, 2], [0; 3]]);
    }

    /// A read's length change must lie in the span, flank by flank: in `CA>C` called `0/1` at
    /// the A10 tract, a read inserting a G in the left flank, deleting an A of the tract and
    /// deleting a base of the right flank has the called length change and the called bases over
    /// the span, but is in neither sum.
    #[test]
    fn indels_that_cancel_out_across_the_flanks_are_in_neither_sum() {
        let record = "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1";
        let r = reference();
        let bases = format!("{}G{}{}{}", &r[50..77], &r[77..80], &r[81..92], &r[93..120]);
        let (sums, counts) = sums_at(record, &[("both", 50, "27M1I3M1D11M1D27M", &bases, 30)]);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (1, 0, 0));
        assert_eq!(counts, [[0; 3], [0, 0, 1], [0; 3]]);
    }

    /// The span's left edge is the read's base just before it: where the record's REF starts
    /// inside the tract (`AA>A` and `A>AA` at 0-based 84 of the A10 tract [80, 90)), a read
    /// deleting the tract's first base, or inserting before it, has the called bases.
    #[test]
    fn an_indel_at_the_left_edge_of_the_span_is_read() {
        let r = reference();
        let deleted = format!("{}{}", &r[50..80], &r[81..120]);
        let (sums, _) = sums_at(
            "chr1\t85\t.\tAA\tA\t50\tPASS\t.\tGT\t0/1",
            &[("left", 50, "30M1D39M", &deleted, 30), ("right", 50, "39M1D30M", &deleted, 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900 + 1600));
        let (a, g) = (inserted(80, "A"), inserted(85, "G"));
        let (sums, counts) = sums_at(
            "chr1\t85\t.\tA\tAA\t50\tPASS\t.\tGT\t0/1",
            &[
                ("left", 50, "30M1I40M", &a, 30),
                ("right", 50, "40M1I30M", &a, 40),
                ("middle", 50, "35M1I35M", &a, 50),
                ("g", 50, "35M1I35M", &g, 60),
            ],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (4, 0, 900 + 1600 + 2500));
        assert_eq!(counts, [[0; 3], [0, 0, 4], [0; 3]]);
    }

    /// An insertion is read at either end of the tract: in `C>CA` at the A10 tract, before its
    /// first base and after its last.
    #[test]
    fn an_insertion_at_either_end_of_the_tract_is_read() {
        let a = inserted(80, "A");
        let (sums, counts) = sums_at(
            "chr1\t80\t.\tC\tCA\t50\tPASS\t.\tGT\t0/1",
            &[("left", 50, "30M1I40M", &a, 30), ("right", 50, "40M1I30M", &a, 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900 + 1600));
        assert_eq!(counts, [[0; 3], [0, 0, 2], [0; 3]]);
    }

    /// A read without bases (SEQ `*`) is counted by its length in the pileup, and is in neither
    /// sum, whether of the reference's length or the called allele's; at a SNP it has no allele.
    #[test]
    fn a_read_without_bases_is_in_neither_sum() {
        let record = "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1";
        let (sums, counts) = sums_at(
            record,
            &[("ref_length", 50, "70M", "*", 30), ("deletion", 50, "30M1D39M", "*", 30)],
        );
        assert_eq!((sums.n_reads, sums.sum_sq, sums.ref_sum_sq, sums.alt_sum_sq), (2, 1800, 0, 0));
        assert!(sums.alleles_read);
        assert_eq!(counts, [[0, 0, 1], [0, 0, 1], [0; 3]]);
        let snp = "chr1\t123\t.\tG\tT\t50\tPASS\t.\tGT\t0/1";
        let (sums, counts) = sums_at(snp, &[("snp", 90, "70M", "*", 30)]);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (1, 0, 0));
        assert_eq!(counts, [[0; 3]; 3]);
    }

    /// A read's indel must lie in the span the bases are compared over, not between it and an
    /// anchor: in `CA>C` called `0/1` on A5 T A4 (the A5 tract and its anchors reach into the
    /// A4), a read deleting the T, or an A of the A4, has the called length, but not its bases.
    #[test]
    fn an_indel_between_the_span_and_an_anchor_is_in_neither_sum() {
        let reference = format!("{}AAAAATAAAA{}", "CGTC".repeat(20), "GCTG".repeat(20));
        let record = "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1";
        let deleted = |at: usize| format!("{}{}", &reference[50..at], &reference[at + 1..120]);
        let (a, t, a4) = (deleted(80), deleted(85), deleted(87));
        let (sums, counts) = sums_on(
            &reference,
            record,
            &[
                ("a", 50, "30M1D39M", &a, 30),
                ("t", 50, "35M1D34M", &t, 40),
                ("a4", 50, "37M1D32M", &a4, 50),
            ],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (3, 0, 900));
        // The pileup counts each of them by its length.
        assert_eq!(counts, [[0; 3], [0, 0, 3], [0; 3]]);
    }

    /// A deletion of the called length elsewhere in an impure repeat is another allele: in
    /// `CAAAAATAAAA>CAAAATAAAA` (an A deleted from the A5 run of A5 T A4) called `0/1`, a read
    /// with the T deleted is in neither sum.
    #[test]
    fn a_deletion_of_other_bases_in_an_impure_repeat_is_in_neither_sum() {
        let reference = format!("{}AAAAATAAAA{}", "CGTC".repeat(20), "GCTG".repeat(20));
        let record = "chr1\t80\t.\tCAAAAATAAAA\tCAAAATAAAA\t50\tPASS\t.\tGT\t0/1";
        let deleted = |at: usize| format!("{}{}", &reference[50..at], &reference[at + 1..120]);
        let (a, t) = (deleted(80), deleted(85));
        let (sums, counts) = sums_on(
            &reference,
            record,
            &[("a", 50, "30M1D39M", &a, 30), ("t", 50, "35M1D34M", &t, 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900));
        assert_eq!(counts, [[0; 3], [0, 0, 2], [0; 3]]);
    }

    /// A deletion read has the called allele's bases wherever the deletion is in a perfect
    /// repeat tract: in `CA>C` at the A10 tract, at either end of it.
    #[test]
    fn a_called_deletion_is_matched_at_either_end_of_its_tract() {
        let record = "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1";
        let deleted = |at: usize| format!("{}{}", edited(50, at, &[]), edited(at + 1, 120, &[]));
        let (left, right) = (deleted(80), deleted(89));
        let (sums, counts) = sums_at(
            record,
            &[("left", 50, "30M1D39M", &left, 30), ("right", 50, "39M1D30M", &right, 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900 + 1600));
        assert_eq!(counts, [[0; 3], [0, 0, 2], [0; 3]]);
    }

    /// A reference skip (`N`) is not a deletion in the sums: in `CA>C` at the A10 tract, a read
    /// skipping an A has the called length change and bases, but is in neither sum, while the
    /// pileup counts it by its length.
    #[test]
    fn a_reference_skip_is_in_neither_sum() {
        let record = "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1";
        let deleted = format!("{}{}", edited(50, 80, &[]), edited(81, 120, &[]));
        let (sums, counts) = sums_at(
            record,
            &[("skip", 50, "30M1N39M", &deleted, 30), ("deletion", 50, "30M1D39M", &deleted, 40)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 1600));
        assert_eq!(counts, [[0; 3], [0, 0, 2], [0; 3]]);
    }

    #[rstest::rstest]
    #[case::snp("chr1\t71\t.\tT\tG\t50\tPASS\t.\tGT\t0/1", "70M", 0, OTHER)]
    #[case::indel_reference("chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1", "70M", 0, REF)]
    #[case::deletion("chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1", "30M1D39M", 900, ALT)]
    fn reference_equal_bases_are_resolved_for_annotations(
        #[case] record: &str,
        #[case] cigar: &str,
        #[case] alt_sum: u64,
        #[case] encoded_counted: usize,
    ) {
        let literal = if alt_sum == 0 {
            edited(50, 120, &[])
        } else {
            format!("{}{}", edited(50, 80, &[]), edited(81, 120, &[]))
        };
        for (encoded, bases) in
            [literal.as_str(), &"=".repeat(literal.len())].into_iter().enumerate()
        {
            let dir = TempDir::new().unwrap();
            let fixture = fixture(
                dir.path(),
                &[record],
                &[TestRead { bases: Some(bases), ..read("read", 50, cigar, 30) }],
            );
            let annotations = ReadAnnotationOptions::default();
            let sites = measure_annotations(&fixture.inputs(), &["chr1"], &annotations).unwrap();
            let sums = sites[0].mapq.unwrap();
            assert_eq!(
                (sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq),
                (1, 900 - alt_sum, alt_sum)
            );
            let counted = if encoded == 1 {
                encoded_counted
            } else if alt_sum > 0 {
                ALT
            } else {
                REF
            };
            let mut counts = [[0; 3]; 3];
            counts[counted][2] = 1;
            assert_eq!(sites[0].row.counts, counts);
            let pileup = measure_pileup(
                &fixture.inputs(),
                &["chr1"],
                &PileupOptions::reading_alleles_as(&annotations),
            )
            .unwrap();
            assert_eq!(sites[0].row, pileup[0]);
        }
    }

    #[test]
    fn reference_equal_is_not_an_inserted_base_wildcard() {
        let record = "chr1\t80\t.\tC\tCT\t50\tPASS\t.\tGT\t0/1";
        let (sums, _) = sums_at(record, &[("read", 50, "30M1I40M", &"=".repeat(71), 30)]);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (1, 0, 0));
    }

    #[rstest::rstest]
    #[case::insertion_right(false, "GA", 1, 5, 2500)]
    #[case::insertion_left(false, "GCC", -2, 5, 2500)]
    #[case::deletion_right(true, "GA", 1, 5, 2500)]
    #[case::deletion_left(true, "GCC", -2, 5, 2500)]
    #[case::indistinguishable_rotation(false, "GA", 1, 1, 0)]
    #[case::just_wide_enough(false, "GA", 1, 2, 2500)]
    #[case::left_indistinguishable(false, "GCC", -2, 1, 0)]
    #[case::left_wide_enough(false, "GCC", -2, 2, 2500)]
    fn rotated_indel_placements_have_the_same_annotation(
        #[case] deletion: bool,
        #[case] motif: &str,
        #[case] shift: i64,
        #[case] flank: u64,
        #[case] alt_sum: u64,
    ) {
        let dir = TempDir::new().unwrap();
        let mut reference = reference();
        if deletion {
            reference.replace_range(61..61 + motif.len(), motif);
            if shift > 0 {
                reference.replace_range(61 + motif.len()..62 + motif.len(), &motif[..1]);
            }
        }
        let (ref_allele, alt) = if deletion {
            (format!("C{motif}"), "C".to_string())
        } else {
            ("C".to_string(), format!("C{motif}"))
        };
        let record = format!("chr1\t61\t.\t{ref_allele}\t{alt}\t50\tPASS\t.\tGT\t0/1");
        let bases = if deletion {
            format!("{}{}", &reference[50..61], &reference[61 + motif.len()..120])
        } else {
            format!("{}{motif}{}", &reference[50..61], &reference[61..120])
        };
        let deleted = if deletion { motif.len() as i64 } else { 0 };
        let cigar = |shift| {
            format!(
                "{}M{}{}{}M",
                11 + shift,
                motif.len(),
                if deletion { 'D' } else { 'I' },
                59 - shift - deleted
            )
        };
        let (original, shifted) = (cigar(0), cigar(shift));
        let fixture = fixture_on(
            dir.path(),
            &reference,
            &[&record],
            &[
                TestRead { bases: Some(&bases), ..read("original", 50, &original, 30) },
                TestRead { bases: Some(&bases), ..read("shifted", 50, &shifted, 40) },
            ],
        );
        let options: ReadAnnotationOptions = format!("flank{flank}").parse().unwrap();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
        assert_eq!(sites.len(), 1);
        let sums = sites[0].mapq.unwrap();
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, alt_sum));
        assert_eq!(sums.alleles_read, alt_sum > 0);
        if flank > 1 {
            assert_eq!(sites[0].row.counts, [[0; 3], [0, 0, 2], [0; 3]]);
        }
        let pileup = measure_pileup(
            &fixture.inputs(),
            &["chr1"],
            &PileupOptions::reading_alleles_as(&options),
        )
        .unwrap();
        assert_eq!(pileup.len(), 1);
        assert_eq!(sites[0].row, pileup[0]);
    }

    #[test]
    fn an_indistinguishable_uncalled_rotation_disables_allele_sums() {
        let dir = TempDir::new().unwrap();
        let bases = inserted(61, "T");
        let fixture = fixture(
            dir.path(),
            &["chr1\t61\t.\tC\tCT,CGA\t50\tPASS\t.\tGT\t0/1"],
            &[TestRead { bases: Some(&bases), ..read("called", 50, "11M1I59M", 40) }],
        );
        let options: ReadAnnotationOptions = "flank1".parse().unwrap();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
        let sums = sites[0].mapq.unwrap();
        assert_eq!((sums.n_reads, sums.sum_sq, sums.ref_sum_sq, sums.alt_sum_sq), (1, 1600, 0, 0));
        assert!(!sums.alleles_read);
        let pileup = measure_pileup(
            &fixture.inputs(),
            &["chr1"],
            &PileupOptions::reading_alleles_as(&options),
        )
        .unwrap();
        assert_eq!(sites[0].row, pileup[0]);
    }

    #[rstest::rstest]
    #[case::partial_right(71, 70, "6N20M", 0)]
    #[case::partial_right_aligned(71, 70, "1M5N20M", 1)]
    #[case::partial_left(71, 44, "20M6N", 0)]
    #[case::partial_left_aligned(71, 44, "20M5N1M", 1)]
    #[case::split_right(71, 70, "3N3N20M", 0)]
    #[case::split_right_insertion(71, 70, "3N1I3N20M", 0)]
    #[case::split_left(71, 44, "20M3N3N", 0)]
    #[case::split_left_insertion(71, 44, "20M3N1I3N", 0)]
    #[case::split_with_deletion(71, 70, "3N1D2N20M", 1)]
    #[case::left_skipped(1, 0, "6N20M", 0)]
    #[case::left_aligned(1, 0, "1M5N20M", 1)]
    #[case::right_skipped(170, 144, "20M6N", 0)]
    #[case::right_aligned(170, 144, "20M5N1M", 1)]
    fn annotation_windows_are_clipped_to_the_contig(
        #[case] pos: u64,
        #[case] start: i64,
        #[case] cigar: &str,
        #[case] expected: u64,
    ) {
        let dir = TempDir::new().unwrap();
        let base = reference().as_bytes()[(pos - 1) as usize] as char;
        let record = format!("chr1\t{pos}\t.\t{base}\tA\t50\tPASS\t.\tGT\t0/1");
        let fixture = fixture(dir.path(), &[&record], &[read("read", start, cigar, 60)]);
        let options = ReadAnnotationOptions::default();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
        assert_eq!(sites.len(), 1);
        let sums = sites[0].mapq.unwrap();
        assert_eq!((sums.n_reads, sums.sum_sq), (expected, 3600 * expected));
        let pileup = measure_pileup(
            &fixture.inputs(),
            &["chr1"],
            &PileupOptions::reading_alleles_as(&options),
        )
        .unwrap();
        assert_eq!(pileup.len(), 1);
        assert_eq!(sites[0].row, pileup[0]);
    }

    #[test]
    fn reference_equal_cache_preserves_multiple_reads_and_sites() {
        let dir = TempDir::new().unwrap();
        let positions = [70, 79, 100];
        let reference = reference();
        let edits: Vec<(usize, u8)> = positions
            .iter()
            .map(|&at| (at, if reference.as_bytes()[at] == b'A' { b'G' } else { b'A' }))
            .collect();
        let records: Vec<String> = edits
            .iter()
            .map(|&(at, alt)| {
                format!(
                    "chr1\t{}\t.\t{}\t{}\t50\tPASS\t.\tGT\t0/1",
                    at + 1,
                    reference.as_bytes()[at] as char,
                    alt as char
                )
            })
            .collect();
        let (equal, alt, literal) = ("=".repeat(70), edited(50, 120, &edits), edited(60, 120, &[]));
        let fixture = fixture(
            dir.path(),
            &records.iter().map(String::as_str).collect::<Vec<_>>(),
            &[
                TestRead { bases: Some(&equal), ..read("cache_equal1", 50, "70M", 30) },
                TestRead { bases: Some(&alt), ..read("cache_alt", 50, "70M", 40) },
                TestRead { bases: Some(&equal[..60]), ..read("cache_equal2", 60, "60M", 50) },
                TestRead { bases: Some(&literal), ..read("cache_literal", 60, "60M", 60) },
            ],
        );
        let options = ReadAnnotationOptions::default();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
        let pileup = measure_pileup(
            &fixture.inputs(),
            &["chr1"],
            &PileupOptions::reading_alleles_as(&options),
        )
        .unwrap();
        assert_eq!(sites.len(), 3);
        assert_eq!(pileup.len(), sites.len());
        for (site, row) in sites.iter().zip(pileup) {
            let sums = site.mapq.unwrap();
            assert_eq!(
                (sums.n_reads, sums.sum_sq, sums.ref_sum_sq, sums.alt_sum_sq),
                (4, 8600, 7000, 1600)
            );
            assert_eq!(site.row.counts, [[0, 0, 1], [0, 0, 1], [0, 0, 2]]);
            assert_eq!(site.row, row);
        }
    }

    #[test]
    fn a_read_without_a_cigar_does_not_contribute_annotations() {
        let dir = TempDir::new().unwrap();
        let fixture = fixture(
            dir.path(),
            &["chr1\t71\t.\tT\tG\t50\tPASS\t.\tGT\t0/1"],
            &[read("aligned", 50, "70M", 30), read("no_cigar", 70, "", 60)],
        );
        let options = ReadAnnotationOptions::default();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
        let sums = sites[0].mapq.unwrap();
        assert_eq!((sums.n_reads, sums.sum_sq), (1, 900));
        let pileup = measure_pileup(
            &fixture.inputs(),
            &["chr1"],
            &PileupOptions::reading_alleles_as(&options),
        )
        .unwrap();
        assert_eq!(sites[0].row, pileup[0]);
    }

    #[rstest::rstest]
    #[case::skip_all("20M100N20M", 0)]
    #[case::split_skip("20M50N50N20M", 0)]
    #[case::split_skip_with_insertion("20M50N1I50N20M", 0)]
    #[case::split_skip_with_deletion("20M50N1D49N20M", 1)]
    #[case::exact_window("65M11N64M", 0)]
    #[case::left_flank("66M54N20M", 1)]
    #[case::right_flank("20M55N65M", 1)]
    #[case::deletion("20M100D20M", 1)]
    fn only_skipping_the_entire_annotation_window_excludes_depth(
        #[case] cigar: &str,
        #[case] expected: u64,
    ) {
        let dir = TempDir::new().unwrap();
        let fixture = fixture(
            dir.path(),
            &["chr1\t71\t.\tC\tG\t50\tPASS\t.\tGT\t0/1"],
            &[read("read", 0, cigar, 60)],
        );
        let options = ReadAnnotationOptions::default();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
        let sums = sites[0].mapq.unwrap();
        assert_eq!((sums.n_reads, sums.sum_sq), (expected, 3600 * expected));
        let pileup = measure_pileup(
            &fixture.inputs(),
            &["chr1"],
            &PileupOptions::reading_alleles_as(&options),
        )
        .unwrap();
        assert_eq!(sites[0].row, pileup[0]);
    }

    /// An indel record without a pure insertion or deletion allele has no pileup row, and no
    /// allele sums, though its called alleles could be read: in `CA>GTT,GA` called `0/2`, a `GTT`
    /// read has the `G` of `GA`, and read by `GA`'s bases alone would be in the ALT sum.
    #[test]
    fn a_record_with_an_alternate_that_cannot_be_read_has_no_allele_sums() {
        let record = "chr1\t80\t.\tCA\tGTT,GA\t50\tPASS\t.\tGT\t0/2";
        let (same, ga) = (edited(50, 120, &[]), edited(50, 120, &[(79, b'G')]));
        let gtt = format!("{}GTT{}", edited(50, 79, &[]), edited(81, 120, &[]));
        let (sums, counts) = sums_at(
            record,
            &[
                ("ref", 50, "70M", &same, 20),
                ("ga", 50, "70M", &ga, 30),
                ("gtt", 50, "31M1I39M", &gtt, 40),
            ],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (3, 0, 0));
        assert!(!sums.alleles_read);
        assert_eq!(counts, [[0; 3]; 3]);
    }

    /// A called indel allele is read across its own repeat tract, not the record's first: in
    /// `CAAAAGT>CAAAGT,CAAAA` (an A deleted from an A4 run, and a GT from the GT6 run after it)
    /// called `0/2`, a read with the last GT deleted, beyond the A run's flank, has the called
    /// allele, as does a read with a TG deleted inside the run (the same bases), and a read with
    /// an A deleted has neither.
    #[test]
    fn a_called_indel_is_read_across_its_own_tract() {
        let reference = format!("{}AAAA{}{}", "CGTC".repeat(20), "GT".repeat(6), "CATG".repeat(20));
        let record = "chr1\t80\t.\tCAAAAGT\tCAAAGT,CAAAA\t50\tPASS\t.\tGT\t0/2";
        let bases = |ranges: &[(usize, usize)]| -> String {
            ranges.iter().map(|&(start, end)| &reference[start..end]).collect()
        };
        let (same, gt, a) =
            (bases(&[(40, 140)]), bases(&[(40, 94), (96, 140)]), bases(&[(40, 80), (81, 140)]));
        let tg = bases(&[(40, 85), (87, 140)]);
        assert_eq!(tg, gt);
        let (sums, _) = sums_on(
            &reference,
            record,
            &[
                ("ref", 40, "100M", &same, 20),
                ("gt", 40, "54M2D44M", &gt, 30),
                ("a", 40, "40M1D59M", &a, 40),
                ("tg", 40, "45M2D53M", &tg, 50),
            ],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (4, 400, 900 + 2500));
        assert!(sums.alleles_read);
    }

    /// A read that has a called allele's bases, and also an uncalled allele's of the same length
    /// change, is in neither sum: in `A>AT,AG` called `0/1`, a read inserting a low-quality base
    /// or an `N` could be either. A read inserting `G` is `AG`, uncalled.
    #[test]
    fn a_read_matching_an_uncalled_allele_too_is_in_neither_sum() {
        let record = "chr1\t81\t.\tA\tAT,AG\t50\tPASS\t.\tGT\t0/1";
        let (t, g, n) = (inserted(81, "T"), inserted(81, "G"), inserted(81, "N"));
        // Only the inserted base (query position 31) is below the base quality.
        let mut quals = vec![30; t.len()];
        quals[31] = 5;
        let reads = [
            TestRead { bases: Some(&t), ..read("t", 50, "31M1I39M", 20) },
            TestRead { bases: Some(&g), quals: Some(&quals), ..read("low_g", 50, "31M1I39M", 30) },
            TestRead { bases: Some(&n), ..read("n", 50, "31M1I39M", 40) },
            TestRead { bases: Some(&g), ..read("g", 50, "31M1I39M", 50) },
        ];
        let (sums, counts) = sums_with(&reference(), record, &reads);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (4, 0, 400));
        assert_eq!(sums.sum_sq, 400 + 900 + 1600 + 2500);
        assert_eq!(counts, [[0; 3], [0, 0, 4], [0; 3]]);
    }

    /// A read of the reference's length is in the REF sum only if it has the reference's bases
    /// across the tract, with no indel among them: in `CA>C` at the A10 tract, a read deleting
    /// the first A and inserting a T (or an A) further along is in neither sum. A low-quality
    /// base matches any. The pileup counts all of them as the reference.
    #[test]
    fn a_read_of_the_reference_length_is_matched_by_its_bases() {
        let record = "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1";
        let r = reference();
        let same = edited(50, 120, &[]);
        let indels = |base: &str| format!("{}{}{base}{}", &r[50..80], &r[81..84], &r[84..120]);
        let (with_t, with_a) = (indels("T"), indels("A"));
        let mismatched = edited(50, 120, &[(85, b'G')]);
        let mut quals = vec![30; mismatched.len()];
        quals[85 - 50] = 5;
        let high = edited(50, 120, &[(85, b'G')]);
        let reads = [
            TestRead { bases: Some(&same), ..read("ref", 50, "70M", 20) },
            TestRead { bases: Some(&with_t), ..read("t", 50, "30M1D3M1I36M", 30) },
            TestRead { bases: Some(&with_a), ..read("a", 50, "30M1D3M1I36M", 40) },
            TestRead {
                bases: Some(&mismatched),
                quals: Some(&quals),
                ..read("low_quality", 50, "70M", 50)
            },
            TestRead { bases: Some(&high), ..read("high_quality", 50, "70M", 60) },
        ];
        let (sums, counts) = sums_with(&reference(), record, &reads);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (5, 400 + 2500, 0));
        assert_eq!(counts, [[0, 0, 5], [0; 3], [0; 3]]);
        // With a same-length alternate (`CA>C,GA` called `1/2`), the reference's and the
        // alternate's reads read by their substituted base need no indel in the tract too.
        let record = "chr1\t80\t.\tCA\tC,GA\t50\tPASS\t.\tGT\t1/2";
        let ga = edited(50, 120, &[(79, b'G')]);
        let pair = "32M1D1M1I36M";
        let reads = [
            TestRead { bases: Some(&same), ..read("ref", 50, "70M", 20) },
            TestRead { bases: Some(&same), ..read("ref_pair", 50, pair, 30) },
            TestRead { bases: Some(&ga), ..read("ga", 50, "70M", 40) },
            TestRead { bases: Some(&ga), ..read("ga_pair", 50, pair, 50) },
        ];
        let (sums, counts) = sums_with(&reference(), record, &reads);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (4, 400, 1600));
        assert_eq!(counts, [[0, 0, 2], [0, 0, 2], [0; 3]]);
    }

    /// A read of a same-length alternate's substituted bases is in the ALT sum only if it has
    /// that allele's bases across the tract too: in `CA>C,GA` called `1/2`, a read with the `G`
    /// of `GA` and a high-quality `G` in the A10 tract is neither allele, though the pileup
    /// counts it as `GA`. A low-quality `G` there matches any base.
    #[test]
    fn a_same_length_alternate_is_matched_by_its_bases_across_the_tract() {
        let record = "chr1\t80\t.\tCA\tC,GA\t50\tPASS\t.\tGT\t1/2";
        let ga = edited(50, 120, &[(79, b'G')]);
        let with_g = edited(50, 120, &[(79, b'G'), (85, b'G')]);
        let mut quals = vec![30; with_g.len()];
        quals[85 - 50] = 5;
        let reads = [
            TestRead { bases: Some(&ga), ..read("ga", 50, "70M", 20) },
            TestRead { bases: Some(&with_g), ..read("high_g", 50, "70M", 30) },
            TestRead { bases: Some(&with_g), quals: Some(&quals), ..read("low_g", 50, "70M", 40) },
        ];
        let (sums, counts) = sums_with(&reference(), record, &reads);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (3, 0, 400 + 1600));
        assert_eq!(counts, [[0; 3], [0, 0, 3], [0; 3]]);
    }

    /// The bases of an indel record's alleles are compared across every alternate's tract
    /// within the anchors, not only the REF and the called alleles' tract: in
    /// `CAAAAGT>CAAAGT,CAAAA` called `1/2` with a flank of 13, the anchors of the A deletion take
    /// in the GT6 run, and a read deleting its last GT has `CAAAA`'s bases, as does one deleting
    /// its first.
    #[test]
    fn indel_bases_are_compared_across_every_alternates_tract() {
        let reference = two_tracts();
        let record = "chr1\t80\t.\tCAAAAGT\tCAAAGT,CAAAA\t50\tPASS\t.\tGT\t1/2";
        let bases = |ranges: &[(usize, usize)]| -> String {
            ranges.iter().map(|&(start, end)| &reference[start..end]).collect()
        };
        let (same, last_gt, first_gt, a) = (
            bases(&[(40, 140)]),
            bases(&[(40, 94), (96, 140)]),
            bases(&[(40, 84), (86, 140)]),
            bases(&[(40, 80), (81, 140)]),
        );
        let reads: Vec<TestRead<'_>> = [
            ("ref", "100M", same.as_str(), 20),
            ("last_gt", "54M2D44M", last_gt.as_str(), 30),
            ("first_gt", "44M2D54M", first_gt.as_str(), 40),
            ("a", "40M1D59M", a.as_str(), 50),
        ]
        .into_iter()
        .map(|(name, cigar, bases, mapq)| TestRead {
            bases: Some(bases),
            ..read(name, 40, cigar, mapq)
        })
        .collect();
        let dir = TempDir::new().unwrap();
        let fixture = fixture_on(dir.path(), &reference, &[record], &reads);
        let annotations: ReadAnnotationOptions = "flank13".parse().unwrap();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &annotations).unwrap();
        let [site] = sites.as_slice() else { panic!("{sites:?}") };
        let sums = site.mapq.unwrap();
        assert!(sums.alleles_read);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (4, 400, 900 + 1600 + 2500));
    }

    /// An insertion outside the anchors changes no allele: with a flank of 1, in `CA>C` at the
    /// A10 tract, a read inserting a G before the padding base (the first anchor) is the
    /// reference, in the pileup and in the sums.
    #[test]
    fn an_insertion_outside_the_anchors_is_not_an_indel_of_the_tract() {
        let r = reference();
        let bases = format!("{}G{}", &r[50..79], &r[79..119]);
        let reads = [TestRead { bases: Some(&bases), ..read("ins", 50, "29M1I40M", 30) }];
        let dir = TempDir::new().unwrap();
        let record = "chr1\t80\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1";
        let fixture = fixture_on(dir.path(), &r, &[record], &reads);
        let annotations: ReadAnnotationOptions = "flank1".parse().unwrap();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &annotations).unwrap();
        let [site] = sites.as_slice() else { panic!("{sites:?}") };
        let sums = site.mapq.unwrap();
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (1, 900, 0));
        assert_eq!(site.row.counts, [[0, 0, 1], [0; 3], [0; 3]]);
    }

    #[rstest::rstest]
    #[case::insertion("CA", "C,GAT", "GAT", "31M1I39M", "32M1I38M", true, ("CATTT", 5))]
    #[case::wrong_insertion("CA", "C,GAT", "GAT", "31M1I39M", "32M1I38M", false, ("CATTT", 5))]
    #[case::deletion("CATT", "CAT,GA", "GA", "31M2D37M", "32M2D36M", true, ("CATTT", 5))]
    #[case::wrong_deletion("CATT", "CAT,GA", "GA", "31M2D37M", "32M2D36M", false, ("CATTT", 5))]
    #[case::left_insertion("CA", "C,AGTGA", "AGTGA", "29M3I41M", "27M3I43M", true, ("CATTT", 5))]
    #[case::wrong_left_insertion("CA", "C,AGTGA", "AGTGA", "29M3I41M", "27M3I43M", false, ("CATTT", 5))]
    #[case::left_deletion("AGTCA", "AGTC,GA", "GA", "29M3D38M", "27M3D40M", true, ("AGTCA", 13))]
    #[case::wrong_left_deletion("AGTCA", "AGTC,GA", "GA", "29M3D38M", "27M3D40M", false, ("AGTCA", 13))]
    fn complex_indel_placements_have_identical_annotation_sums(
        #[case] ref_allele: &str,
        #[case] alts: &str,
        #[case] complex: &str,
        #[case] original: &str,
        #[case] shifted: &str,
        #[case] correct: bool,
        #[case] context: (&str, u64),
    ) {
        let mut reference = reference();
        reference.replace_range(79..84, context.0);
        let allele = if correct { complex.to_string() } else { format!("C{}", &complex[1..]) };
        let bases =
            format!("{}{}{}", &reference[50..79], allele, &reference[79 + ref_allele.len()..120]);
        let record = format!("chr1\t80\t.\t{ref_allele}\t{alts}\t50\tPASS\t.\tGT\t1/2");
        for cigar in [original, shifted] {
            let dir = TempDir::new().unwrap();
            let fixture = fixture_on(
                dir.path(),
                &reference,
                &[&record],
                &[TestRead { bases: Some(&bases), ..read("read", 50, cigar, 30) }],
            );
            let options: ReadAnnotationOptions = format!("flank{}", context.1).parse().unwrap();
            let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
            assert_eq!(sites.len(), 1);
            let sums = sites[0].mapq.unwrap();
            assert_eq!(
                (sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq),
                (1, 0, if correct { 900 } else { 0 }),
                "{cigar}"
            );
            let rows = measure_pileup(
                &fixture.inputs(),
                &["chr1"],
                &PileupOptions::reading_alleles_as(&options),
            )
            .unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(sites[0].row, rows[0]);
        }
    }

    #[rstest::rstest]
    #[case::all(5, "1/2", "1/2", 2500)]
    #[case::only_a(5, "0/1", "0/2", 900)]
    #[case::only_gt(5, "0/2", "0/1", 1600)]
    #[case::no_covering_window(1, "1/2", "1/2", 0)]
    #[case::stable_shortest(13, "1/2", "1/2", 2500)]
    fn annotation_windows_do_not_depend_on_alt_order(
        #[case] flank: u64,
        #[case] gt: &str,
        #[case] swapped_gt: &str,
        #[case] alt_sum: u64,
    ) {
        let reference = two_tracts();
        let start = if flank == 13 { 67 } else { 50 };
        let end = if flank == 13 { 97 } else { 120 };
        let a = format!("{}{}", &reference[start..80], &reference[81..end]);
        let a_cigar = format!("{}M1D{}M", 80 - start, end - 81);
        let gt_bases = format!("{}{}", &reference[50..94], &reference[96..120]);
        let options: ReadAnnotationOptions = format!("flank{flank}").parse().unwrap();
        let mut measured = Vec::new();
        for (alts, called) in [("CAAAGT,CAAAA", gt), ("CAAAA,CAAAGT", swapped_gt)] {
            let dir = TempDir::new().unwrap();
            let record = format!("chr1\t80\t.\tCAAAAGT\t{alts}\t50\tPASS\t.\tGT\t{called}");
            let fixture = fixture_on(
                dir.path(),
                &reference,
                &[&record],
                &[
                    TestRead { bases: Some(&a), ..read("a", start as i64, &a_cigar, 30) },
                    TestRead { bases: Some(&gt_bases), ..read("gt", 50, "44M2D24M", 40) },
                ],
            );
            let sites = measure_annotations(&fixture.inputs(), &["chr1"], &options).unwrap();
            assert_eq!(sites.len(), 1);
            let sums = sites[0].mapq.unwrap();
            assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, alt_sum), "{alts}");
            assert_eq!(sums.alleles_read, alt_sum > 0);
            let rows = measure_pileup(
                &fixture.inputs(),
                &["chr1"],
                &PileupOptions::reading_alleles_as(&options),
            )
            .unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(sites[0].row, rows[0]);
            measured.push(sums);
        }
        assert_eq!(measured[0], measured[1]);
    }

    /// The `CAAAAGT>CAAAGT,CAAAA` reference: an A4 run, then a GT6 run.
    fn two_tracts() -> String {
        format!("{}AAAA{}{}", "CGTC".repeat(20), "GT".repeat(6), "CATG".repeat(20))
    }

    /// The reference of a REF starting more than the flank before an A4 tract: `C G7 A4`.
    fn g7_a4() -> String {
        format!("{}CGGGGGGGAAAA{}", "CGTC".repeat(20), "CTCG".repeat(20))
    }

    /// The annotation window covers an uncalled GT deletion outside the pileup's A-tract
    /// anchors, so it is no longer mistaken for REF. The original pileup still counts it by
    /// the length between its own unchanged anchors.
    #[test]
    fn an_annotation_window_covers_an_uncalled_indel_outside_the_pileup_anchors() {
        let reference = two_tracts();
        let record = "chr1\t80\t.\tCAAAAGT\tCAAAGT,CAAAA\t50\tPASS\t.\tGT\t0/1";
        let (same, gt) = (
            reference[40..140].to_string(),
            format!("{}{}", &reference[40..94], &reference[96..140]),
        );
        let (sums, counts) = sums_on(
            &reference,
            record,
            &[("ref", 40, "100M", &same, 20), ("gt", 40, "54M2D44M", &gt, 30)],
        );
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 400, 0));
        assert!(sums.alleles_read);
        assert_eq!(counts, [[0, 0, 2], [0; 3], [0; 3]]);
    }

    /// With every alternate called, the independently selected annotation anchors cover every
    /// change; the pileup rows still use the original first-ALT window.
    #[test]
    fn every_called_alternate_fits_the_selected_annotation_anchors() {
        let reference = two_tracts();
        let record = "chr1\t80\t.\tCAAAAGT\tCAAAGT,CAAAA\t50\tPASS\t.\tGT\t1/2";
        let (same, gt) = (
            reference[40..140].to_string(),
            format!("{}{}", &reference[40..94], &reference[96..140]),
        );
        let reads =
            [("ref", 40, "100M", same.as_str(), 20), ("gt", 40, "54M2D44M", gt.as_str(), 30)];
        let (sums, counts) = sums_on(&reference, record, &reads);
        assert!(sums.alleles_read);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 400, 900));
        let dir = TempDir::new().unwrap();
        let reads: Vec<TestRead<'_>> = reads
            .iter()
            .map(|&(name, pos, cigar, bases, mapq)| TestRead {
                bases: Some(bases),
                ..read(name, pos, cigar, mapq)
            })
            .collect();
        let fixture = fixture_on(dir.path(), &reference, &[record], &reads);
        let rows = measure_pileup(&fixture.inputs(), &["chr1"], &PileupOptions::default()).unwrap();
        let [row] = rows.as_slice() else { panic!("{rows:?}") };
        assert_eq!(counts, row.counts);
    }

    /// Records whose anchors take in every alternate's change are read: in `CAA>C,CA` called
    /// `0/2`, a read deleting one A is the called allele and one deleting two is in neither sum;
    /// in `CA>C,GTT` called `0/1`, a `GTT` read is in neither sum.
    #[test]
    fn alternates_within_the_anchors_are_read() {
        let r = reference();
        let deleted = |len: usize| format!("{}{}", &r[50..80], &r[80 + len..120]);
        let (one, two) = (deleted(1), deleted(2));
        let (sums, _) = sums_at(
            "chr1\t80\t.\tCAA\tC,CA\t50\tPASS\t.\tGT\t0/2",
            &[("one", 50, "30M1D39M", &one, 30), ("two", 50, "30M2D38M", &two, 40)],
        );
        assert!(sums.alleles_read);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900));
        let gtt = format!("{}GTT{}", &r[50..79], &r[81..120]);
        let (sums, _) = sums_at(
            "chr1\t80\t.\tCA\tC,GTT\t50\tPASS\t.\tGT\t0/1",
            &[("c", 50, "30M1D39M", &one, 30), ("gtt", 50, "31M1I39M", &gtt, 40)],
        );
        assert!(sums.alleles_read);
        assert_eq!((sums.n_reads, sums.ref_sum_sq, sums.alt_sum_sq), (2, 0, 900));
    }

    /// A complex alternate whose deleted bases lie before the called allele's first anchor: in
    /// `CGGGGGGGAAAA>CGGGGGGGAAA,TGGGGGGAAAA` called `0/1`, the `TG` replacing `CGG` is outside
    /// the anchors of the A deletion.
    #[test]
    fn a_complex_alternate_outside_the_anchors_has_no_allele_sums() {
        let reference = g7_a4();
        let record = "chr1\t81\t.\tCGGGGGGGAAAA\tCGGGGGGGAAA,TGGGGGGAAAA\t50\tPASS\t.\tGT\t0/1";
        let same = reference[50..120].to_string();
        let (sums, _) = sums_on(&reference, record, &[("ref", 50, "70M", &same, 20)]);
        assert!(!sums.alleles_read);
    }

    /// Only the site the annotations' sums read holds the alleles' bases: the pileup site when
    /// all ALTs are called and its window is selected, else the annotation site; neither when the sums are
    /// not read, and none without annotations.
    #[test]
    fn only_the_site_the_sums_read_holds_the_alleles_bases() {
        let options = PileupOptions::default();
        let has_bases = |site: &Site| match &site.kind {
            SiteKind::Indel { alts, .. } => alts.bases.is_some(),
            SiteKind::Snp(_) => panic!("not an indel"),
        };
        // What holds bases: the pileup site, and the annotations' own site if any.
        let target = |reference: &str, pos: u64, alleles: &str, genotype: Option<&str>| {
            let (ref_allele, alts) = alleles.split_once('>').unwrap();
            let alts: Vec<&str> = alts.split(',').collect();
            let called = genotype.map(|gt| called_alts(gt.parse().unwrap()));
            let target = Target::of_record(
                reference.as_bytes(),
                pos,
                ref_allele.as_bytes(),
                &alts,
                called,
                &options,
            )
            .unwrap();
            let own = target.annotated.as_ref().map(|annotated| match &annotated.alleles {
                AnnotatedAlleles::AsCounted => "as counted",
                AnnotatedAlleles::Called(site) if has_bases(site) => "called with bases",
                AnnotatedAlleles::Called(_) => "called without bases",
                AnnotatedAlleles::Unread => "unread",
            });
            (has_bases(target.site.as_ref().unwrap()), own)
        };
        let (a10, two) = (reference(), two_tracts());
        assert_eq!(target(&a10, 80, "CA>C", None), (false, None));
        assert_eq!(target(&a10, 80, "CA>C", Some("0/1")), (true, Some("as counted")));
        assert_eq!(target(&a10, 80, "CAA>C,CA", Some("0/2")), (false, Some("called with bases")));
        assert_eq!(
            target(&two, 80, "CAAAAGT>CAAAGT,CAAAA", Some("1/2")),
            (false, Some("called with bases"))
        );
    }

    /// Which sites' anchors take in every alternate's change.
    #[test]
    fn reads_every_alt_checks_each_alternates_change() {
        let options = PileupOptions::default();
        let reads_every_alt = |reference: &str, pos: u64, alleles: &str, called: &[u32]| {
            let (ref_allele, alts) = alleles.split_once('>').unwrap();
            let alts: Vec<&str> = alts.split(',').collect();
            let site = Site::of_called(
                reference.as_bytes(),
                pos,
                ref_allele.as_bytes(),
                &alts,
                |index| called.contains(&index),
                &options,
                None,
            )
            .unwrap();
            site.reads_every_alt(reference.as_bytes(), pos, ref_allele.as_bytes(), &alts)
        };
        let (two, a10, g7) = (two_tracts(), reference(), g7_a4());
        assert!(!reads_every_alt(&two, 80, "CAAAAGT>CAAAGT,CAAAA", &[1]));
        assert!(reads_every_alt(&two, 80, "CAAAAGT>CAAAGT,CAAAA", &[2]));
        assert!(!reads_every_alt(&two, 80, "CAAAAGT>CAAAGT,CAAAA", &[1, 2]));
        assert!(reads_every_alt(&a10, 80, "CAA>C,CA", &[2]));
        assert!(reads_every_alt(&a10, 80, "CA>C,GTT", &[1]));
        assert!(!reads_every_alt(&g7, 81, "CGGGGGGGAAAA>CGGGGGGGAAA,TGGGGGGAAAA", &[1]));
        assert!(reads_every_alt(&a10, 80, "CA>C", &[1]));
        // An insertion of T into the A10 tract: an empty tract.
        assert!(reads_every_alt(&a10, 81, "A>AT", &[1]));
        // A SNP site.
        assert!(reads_every_alt(&a10, 123, "G>T", &[1]));
    }

    /// A record whose called alternate allele cannot be read from a read has no allele sums,
    /// though its other allele gives it a pileup row.
    #[test]
    fn allele_sums_need_a_called_allele_that_can_be_read() {
        let dir = TempDir::new().unwrap();
        // The deletion of an A of the tract, and a complex allele, which is the one called.
        let record = "chr1\t80\t.\tCA\tC,GTT\t50\tPASS\t.\tGT\t0/2";
        let reads = [read("ref", 50, "70M", 60), read("del", 50, "35M1D34M", 60)];
        let fixture = fixture(dir.path(), &[record], &reads);
        let annotations = ReadAnnotationOptions::default();
        let sites = measure_annotations(&fixture.inputs(), &["chr1"], &annotations).unwrap();
        let [site] = sites.as_slice() else { panic!("{sites:?}") };
        assert!(site.counted);
        assert_eq!(site.row.counts, [[0, 0, 1], [0, 0, 1], [0; 3]]);
        assert_eq!(site.mapq, Some(MapqSums { n_reads: 2, sum_sq: 7200, ..MapqSums::default() }));
    }

    /// One pass measures both when the pileup reads alleles as the annotations do; otherwise the
    /// pileup rows are measured apart, with their own settings.
    #[rstest::rstest]
    #[case::no_pileup("mapq20,bq10,flank5", None, false)]
    #[case::one_pass("mapq1,bq10,flank5", Some("mapq30,bq10,flank5"), false)]
    #[case::a_pass_each_for_another_base_quality("mapq20,bq10,flank5", Some("bq20"), true)]
    #[case::a_pass_each_for_another_flank("mapq20,bq10,flank5", Some("flank3"), true)]
    fn annotations_and_pileup_are_those_measured_alone(
        #[case] annotations: ReadAnnotationOptions,
        #[case] pileup: Option<&str>,
        #[case] separate: bool,
    ) {
        let pileup: Option<PileupOptions> = pileup.map(|settings| settings.parse().unwrap());
        let dir = TempDir::new().unwrap();
        // Reads that the settings count differently: a SNP base of quality 15 (counted from
        // bq10, not from bq20), and a read with 4 bases left of the tract (spanning it with a
        // flank of 3, not of 5).
        let extra: [SnpRead; 1] = [("bq15", 'T', 60, 0, 15)];
        let bases = bases_of(&[&SNP_READS[..], &extra[..]].concat());
        let mut reads = snp_reads(&[&SNP_READS[..], &extra[..]].concat(), &bases);
        reads.extend([read("left", 50, "70M", 60), read("near", 76, "50M", 60)]);
        let fixture = fixture(dir.path(), &RECORDS, &reads);
        let inputs = fixture.inputs();
        let (sites, measured) =
            measure_annotations_and_pileup(&inputs, &["chr1"], &annotations, pileup).unwrap();
        let alone = measure_annotations(&inputs, &["chr1"], &annotations).unwrap();
        assert_eq!(sums_of(&sites), sums_of(&alone));
        assert_eq!(sites.len(), 3);
        let rows_of =
            |options: &PileupOptions| measure_pileup(&inputs, &["chr1"], options).unwrap();
        match (pileup, measured) {
            (None, None) => {}
            (Some(options), Some(tables)) => {
                let rows = rows_of(&options);
                assert_eq!(tables, PileupTables::new(options, rows.clone()));
                if separate {
                    // The rows are not those the annotations' pass counts with its settings.
                    assert_ne!(rows, counted_rows(&sites));
                    assert_ne!(rows, rows_of(&PileupOptions::reading_alleles_as(&annotations)));
                } else {
                    assert_eq!(rows, counted_rows(&sites));
                }
            }
            other => panic!("{other:?}"),
        }
    }

    /// Measures the SNP of [`RECORDS`] with `reads` over it, as (base at the SNP, MAPQ, number
    /// of reads), in order; returns its site and the pileup's row.
    fn measure_deep_snp(reads: &[(char, u8, usize)]) -> (MeasuredSite, PileupRow) {
        measure_deep_snp_with(reads, &ReadAnnotationOptions::default())
    }

    /// Measures the deep SNP with independent annotation settings, preserving the pileup oracle.
    fn measure_deep_snp_with(
        reads: &[(char, u8, usize)],
        annotations: &ReadAnnotationOptions,
    ) -> (MeasuredSite, PileupRow) {
        let (mut sites, mut rows) = measure_deep_records(reads, annotations, &RECORDS[2..]);
        assert_eq!((sites.len(), rows.len()), (1, 1));
        (sites.remove(0), rows.remove(0))
    }

    /// Measures selected fixture records at high depth with an independently measured pileup.
    fn measure_deep_records(
        reads: &[(char, u8, usize)],
        annotations: &ReadAnnotationOptions,
        records: &[&str],
    ) -> (Vec<MeasuredSite>, Vec<PileupRow>) {
        let dir = TempDir::new().unwrap();
        let specs: Vec<(String, String, u8)> = reads
            .iter()
            .enumerate()
            .flat_map(|(group, &(base, mapq, count))| {
                (0..count).map(move |i| (format!("g{group}_{i}"), snp_bases(base), mapq))
            })
            .collect();
        let reads: Vec<TestRead<'_>> = specs
            .iter()
            .map(|(name, bases, mapq)| TestRead {
                bases: Some(bases),
                ..read(name, 90, "70M", *mapq)
            })
            .collect();
        let fixture = fixture(dir.path(), records, &reads);
        let options = PileupOptions::default();
        let sites =
            measure_sites(&fixture.inputs(), &["chr1"], &options, Some(annotations), "test")
                .unwrap();
        let rows = measure_pileup(&fixture.inputs(), &["chr1"], &options).unwrap();
        (sites, rows)
    }

    #[rstest::rstest]
    #[case::tiny(1)]
    #[case::two(2)]
    #[case::below_legacy(599)]
    #[case::legacy(600)]
    #[case::past_legacy(601)]
    #[case::deep(1200)]
    #[case::above_depth(2000)]
    fn maxreads_annotations_and_pileup_have_independent_caps(#[case] cap: u64) {
        let annotations: ReadAnnotationOptions = format!("maxreads{cap}").parse().unwrap();
        let (site, row) = measure_deep_snp_with(&[('G', 60, 1500)], &annotations);
        assert_eq!(site.row, row);
        assert_eq!((row.n_reads, row.counts), (601, [[0, 0, 600], [0; 3], [0; 3]]));
        let counted = cap.min(1500);
        assert_eq!(
            site.mapq.unwrap(),
            MapqSums {
                n_reads: counted,
                sum_sq: counted * 3600,
                ref_sum_sq: counted * 3600,
                alt_sum_sq: 0,
                alleles_read: true,
            }
        );
    }

    /// The cap is the annotations' alone: the pileup's settings have none.
    #[test]
    fn maxreads_is_not_a_pileup_setting() {
        let error = "maxreads1200".parse::<PileupOptions>().unwrap_err().to_string();
        assert_eq!(
            error,
            "unknown pileup setting 'maxreads1200' (expected mapqN, bqN, flankN or default)"
        );
    }

    #[rstest::rstest]
    #[case::tiny(1)]
    #[case::boundary(601)]
    #[case::deep(1200)]
    #[case::above_depth(2000)]
    fn maxreads_caps_counted_and_annotation_only_records(#[case] cap: u64) {
        let annotations: ReadAnnotationOptions = format!("maxreads{cap}").parse().unwrap();
        let (sites, rows) = measure_deep_records(&[('G', 60, 1300)], &annotations, &RECORDS[1..]);
        let identities: Vec<_> = sites.iter().map(|site| (site.row.pos, site.counted)).collect();
        assert_eq!(identities, [(100, false), (123, true)]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pos, 123);
        assert_eq!(sites[1].row, rows[0]);
        assert_eq!(rows[0].n_reads, 601);
        assert_eq!(rows[0].counts, [[0, 0, 600], [0; 3], [0; 3]]);
        assert_eq!((sites[0].row.n_reads, sites[0].row.counts), (0, [[0; 3]; 3]));
        let counted = cap.min(1300);
        let complex = sites[0].mapq.unwrap();
        assert_eq!(complex.n_reads, counted);
        assert_eq!(complex.sum_sq, counted * 3600);
        assert!(!complex.alleles_read);
        assert_eq!((complex.ref_sum_sq, complex.alt_sum_sq), (0, 0));
        let snp = sites[1].mapq.unwrap();
        assert_eq!(snp.n_reads, counted);
        assert_eq!(snp.ref_sum_sq, counted * 3600);
    }

    #[test]
    fn maxreads_tiny_cap_still_reaches_reads_after_a_low_mapq_pileup() {
        let annotations: ReadAnnotationOptions = "maxreads2".parse().unwrap();
        let (site, row) =
            measure_deep_snp_with(&[('G', 0, 650), ('T', 60, 100), ('G', 30, 20)], &annotations);
        assert_eq!(site.row, row);
        assert_eq!((row.n_reads, row.counts), (601, [[600, 0, 0], [0; 3], [0; 3]]));
        let sums = site.mapq.unwrap();
        assert_eq!(sums.n_reads, 2);
        assert_eq!((sums.sum_sq, sums.ref_sum_sq, sums.alt_sum_sq), (7200, 0, 7200));
    }

    #[test]
    fn maxreads_excludes_unavailable_mapq_and_sums_the_highest_available_mapq() {
        let annotations: ReadAnnotationOptions = "maxreads1200".parse().unwrap();
        let (site, row) = measure_deep_snp_with(&[('G', 255, 10), ('T', 254, 1300)], &annotations);
        assert_eq!(site.row, row);
        assert_eq!(row.n_reads, 601);
        let sums = site.mapq.unwrap();
        assert_eq!(sums.n_reads, 1200);
        assert_eq!(sums.sum_sq, 1200 * 254 * 254);
        assert_eq!(sums.ref_sum_sq, 0);
        assert_eq!(sums.alt_sum_sq, sums.sum_sq);
    }

    #[test]
    fn deep_sites_are_capped() {
        // More reads than are measured: the pileup counts one read past the cap (but not at its
        // allele), and the sums stop at the cap.
        let (site, row) = measure_deep_snp(&[('G', 60, 700)]);
        assert_eq!(site.row, row);
        assert_eq!((row.n_reads, row.counts), (MAX_READS + 1, [[0, 0, MAX_READS], [0; 3], [0; 3]]));
        let sum_sq = MAX_READS * 3600;
        let sums = site.mapq.unwrap();
        assert_eq!(
            sums,
            MapqSums {
                n_reads: MAX_READS,
                sum_sq,
                ref_sum_sq: sum_sq,
                alt_sum_sq: 0,
                alleles_read: true
            }
        );
        assert!(sums.is_full(LEGACY_MAX_READS));
    }

    #[test]
    fn sums_are_read_past_a_pileup_filled_by_low_mapq_reads() {
        // The MAPQ-0 reads fill the pileup, and are not in the sums: the reads after them are
        // still read at their alleles for the sums.
        let (site, row) = measure_deep_snp(&[('G', 0, 650), ('T', 60, 100), ('G', 30, 20)]);
        assert_eq!(site.row, row);
        assert_eq!((row.n_reads, row.counts), (MAX_READS + 1, [[MAX_READS, 0, 0], [0; 3], [0; 3]]));
        let sums = site.mapq.unwrap();
        assert_eq!(
            sums,
            MapqSums {
                n_reads: 120,
                sum_sq: 100 * 3600 + 20 * 900,
                ref_sum_sq: 20 * 900,
                alt_sum_sq: 100 * 3600,
                alleles_read: true
            }
        );
        assert!(!sums.is_full(LEGACY_MAX_READS));
    }
}
