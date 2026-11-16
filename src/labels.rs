//! Training labels from hap.py: the classes that genotype and promote models predict, and the
//! reading of hap.py output VCFs into per-call labels and truth FN counts.
//!
//! - [`Label`], [`PromoteLabel`] and [`UnfixableFlips`]: the classes and how flip labels a flip
//!   cannot fix are trained.
//! - [`read_happy_records`], [`read_happy_labels`] and [`attach_labels`]: hap.py records, their
//!   labels by `(chrom, pos, variant type)`, and the join of those labels onto a feature table's
//!   calls (with [`labelled_mask`] selecting the labelled PASS calls).
//!
//! Promotion labels, which come from the truth VCF, are made in [`crate::promote_labels`].

/// The genotype-model classes, in model output order.
///
/// `Regenotype` is only used by models trained with the re-genotype action: the call has the
/// wrong genotype or alleles (e.g. a wrong repeat length, which hap.py reports as a local match),
/// a het <-> hom flip does not fix it, but the repeat-length caller's genotype does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Label {
    /// The call is wrong: drop it (make it `0/0`).
    Drop = 0,
    /// The call is right: keep it.
    Keep = 1,
    /// The call has the right alleles but the wrong zygosity: flip het <-> hom.
    Flip = 2,
    /// The repeat-length caller's genotype is right.
    Regenotype = 3,
}

impl Label {
    /// Every class, in model output order.
    pub const ALL: [Label; 4] = [Label::Drop, Label::Keep, Label::Flip, Label::Regenotype];

    /// The class's column in a model's predicted probabilities.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The class stored in a label column (e.g. `2.0`), if it is one.
    #[must_use]
    pub fn from_value(value: f64) -> Option<Label> {
        Label::ALL.into_iter().find(|label| label.index() as f64 == value)
    }
}

/// What training does with an unfixable flip: a [`Label::Flip`] call (an allele-match FP) that
/// scoring's het <-> hom flip cannot fix, because the call is not `0/1` or `1/1` (e.g. a `1/2`
/// call whose truth is homozygous) or flipping it does not give the truth genotype (e.g. the
/// truth is het-alt, so the call lacks an allele).
///
/// Only the training labels change: evaluation and the cross-validated threshold search score
/// every call against its hap.py label, and an unfixable flip is an FP whatever it is labelled.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    serde::Serialize,
    serde::Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum UnfixableFlips {
    /// Train it as [`Label::Flip`], as hap.py labels it.
    #[value(help = None::<&str>)]
    #[default]
    Flip,
    /// Train it as [`Label::Drop`]: dropping it is the only action that removes the FP.
    #[value(help = None::<&str>)]
    Drop,
    /// Leave it out of the fits.
    #[value(help = None::<&str>)]
    Exclude,
}

impl UnfixableFlips {
    /// The lowercase name, as written in model files and on the command line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            UnfixableFlips::Flip => "flip",
            UnfixableFlips::Drop => "drop",
            UnfixableFlips::Exclude => "exclude",
        }
    }

    /// True for the default, [`UnfixableFlips::Flip`].
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == UnfixableFlips::default()
    }
}

impl std::fmt::Display for UnfixableFlips {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per call, whether it is an unfixable flip (see [`UnfixableFlips`]): labelled
/// [`Label::Flip`], and not both flippable and a correct flip.
#[must_use]
pub fn unfixable_flips(labels: &[Label], flip_ok: &[bool], flippable: &[bool]) -> Vec<bool> {
    labels
        .iter()
        .zip(flip_ok.iter().zip(flippable))
        .map(|(&label, (&ok, &can_flip))| label == Label::Flip && !(ok && can_flip))
        .collect()
}

/// The promote-model classes, in model output order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PromoteLabel {
    /// Leave the call filtered.
    Stay = 0,
    /// Promote the call to PASS.
    Promote = 1,
    /// Promote the call to PASS and flip its genotype het <-> hom.
    PromoteFlip = 2,
}

impl PromoteLabel {
    /// The number of promote classes.
    pub const COUNT: usize = 3;

    /// Every class, in model output order.
    pub const ALL: [PromoteLabel; 3] =
        [PromoteLabel::Stay, PromoteLabel::Promote, PromoteLabel::PromoteFlip];

    /// The class's column in a model's predicted probabilities.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The class stored in a promote label column (e.g. `1.0`), if it is one.
    #[must_use]
    pub fn from_value(value: f64) -> Option<PromoteLabel> {
        PromoteLabel::ALL.into_iter().find(|label| label.index() as f64 == value)
    }
}

// ------------------------------------------------------------------------------------------
// Training labels from hap.py (vcfeval engine) output.
//
// Each hap.py record has TRUTH and QUERY samples with FORMAT `BD` (decision: TP/FP/FN/UNK/N),
// `BK` (match kind: `gm` genotype match, `am` allele match, `lm` local match, `.` none) and
// `BVT` (SNP/INDEL/NOCALL). For each query call: TP is `Keep`; FP with BK `.` or `lm` is `Drop`;
// FP with BK `am` is `Flip` (right allele, wrong genotype), and the flip is correct (`flip_ok`)
// only if switching the query genotype between `0/1` and `1/1` gives the truth genotype. The
// truth FNs (TRUTH BD = FN) are counted, since under vcfeval semantics F1 needs them. Every call
// also gets the length changes of its two truth alleles from the first hap.py record at its
// position with a diploid TRUTH genotype and a TRUTH BVT of the call's variant type.
//
// hap.py splits a multi-allelic query call into one record per ALT allele and trims each record on
// its own, so the record of a call's second allele can be at another position (`CTATA>CTA,CTATATA`
// gives `CTA>C` at the call's POS and `T>TAT` three bases on), or at the call's POS with the other
// variant type (the indel call `A>T,AT` gives a SNP record and an indel record). Such a record
// matches no call at its own key, and labels the multi-allelic call it was split from. A call is
// `Keep` only if every record that labels it is TP.
// ------------------------------------------------------------------------------------------

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use log::{info, warn};
use rayon::prelude::*;
use rust_htslib::bcf::header::HeaderView;
use rust_htslib::bcf::{Read as _, Reader};

use crate::alleles::{Genotype, allele_deltas, length_change, trim_alleles};
use crate::common::{PASS, VariantType};
use crate::table::FeatureTable;
use crate::vcf::{has_index, open_indexed};

/// The truth alleles' length changes (bp, sorted) at a call's position, from the hap.py labels.
pub const TRUTH_DELTA_COLUMNS: [&str; 2] = ["truth_delta_lo", "truth_delta_hi"];

/// More hap.py labels than this fraction matching no call means hap.py ran on another VCF.
pub const MAX_UNMATCHED_LABEL_FRACTION: f64 = 0.5;

/// With [`HappyPassOnly::Auto`], hap.py is taken to have run without `--pass-only` when at least
/// this fraction of the non-PASS calls it could have scored (with a non-reference allele in their
/// genotype, on a contig with a hap.py query record) have a hap.py query record that no PASS call
/// explains (see [`attach_labels`]). Run without `--pass-only`, hap.py writes a query record (TP,
/// FP or UNK) for each of those calls; run with it, only for PASS calls, so such a record is then
/// a rare artefact of hap.py representing a PASS call differently. Half sits far from both.
///
/// The genotype is the feature table's normalized diploid one (see [`Genotype`]), where a haploid
/// or partly missing genotype is `./.`, so non-PASS calls with one are not counted. A hap.py run
/// restricted to regions (`-T`, `-R`, or `-l` with a range such as `chr20:1-10000000`) scores no
/// call outside them. Either pulls the fraction down: on haploid contigs (e.g. a male chrX), or
/// for such a run, `--happy-pass-only` then says how hap.py was run.
pub const MIN_NON_PASS_RECORD_FRACTION: f64 = 0.5;

/// A call's `(chrom, pos, variant type)`, the key labels join on.
pub type LabelKey = (String, u64, VariantType);

/// The unscored query records of one contig (see [`HappyLabels::unscored`]): per
/// `(pos, variant type)`, the sorted, distinct length changes of their query alleles.
pub type UnscoredRecords = HashMap<(u64, VariantType), Box<[i64]>>;

/// One call's label (NaN if none), `flip_ok`, truth allele length changes (NaN if none), the
/// position its label came from, and whether that was a padded call's trimmed position.
type CallLabel = (f64, bool, (f64, f64), u64, bool);

/// A borrowed [`LabelKey`], to look keys up without allocating. It hashes as a [`LabelKey`] does
/// (a tuple and a tuple struct hash their fields in order, and `&str` hashes as `String`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct KeyRef<'a>(&'a str, u64, VariantType);

impl indexmap::Equivalent<LabelKey> for KeyRef<'_> {
    fn equivalent(&self, key: &LabelKey) -> bool {
        self.0 == key.0 && self.1 == key.1 && self.2 == key.2
    }
}

/// The fields of one hap.py output record that labelling needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HappyRecord {
    /// The contig.
    pub chrom: String,
    /// The 1-based position.
    pub pos: u64,
    /// The TRUTH genotype.
    pub truth_gt: Genotype,
    /// The TRUTH BD.
    pub truth_bd: String,
    /// The TRUTH BVT.
    pub truth_bvt: String,
    /// The QUERY genotype.
    pub query_gt: Genotype,
    /// The QUERY BD.
    pub query_bd: String,
    /// The QUERY BK.
    pub query_bk: String,
    /// The QUERY BVT.
    pub query_bvt: String,
    /// The record's REF and ALT alleles.
    pub alleles: Vec<String>,
}

/// The label of the TP and FP query records at one `(chrom, pos, variant type)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryLabel {
    /// `Keep` if every record is TP, otherwise the label of the first that is not.
    pub label: Label,
    /// Whether the label is `Flip` and some record's het <-> hom flip gives the truth genotype.
    pub flip_ok: bool,
    /// The sorted, distinct length changes (ALT minus REF) of the records' query alleles, other
    /// than the reference and `*`: they identify which allele of a multi-allelic call a record
    /// was split from.
    pub alt_deltas: Vec<i64>,
}

impl QueryLabel {
    /// A label with no allele length changes, for tests.
    #[cfg(test)]
    fn of(label: Label, flip_ok: bool) -> QueryLabel {
        QueryLabel { label, flip_ok, alt_deltas: Vec::new() }
    }

    /// Adds a later record's label: it replaces a `Keep`, so that `Keep` means every record was
    /// TP; `flip_ok` is true if either is a correct flip.
    fn add(&mut self, later: QueryLabel) {
        if self.label == Label::Keep {
            self.label = later.label;
        }
        self.flip_ok |= later.flip_ok;
        self.alt_deltas.extend(later.alt_deltas);
        self.alt_deltas.sort_unstable();
        self.alt_deltas.dedup();
    }
}

/// Labels for the query calls of one sample, and its truth FN counts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HappyLabels {
    /// The label of the TP and FP query records at each `(chrom, pos, variant type)`, in record
    /// order.
    pub labels: IndexMap<LabelKey, QueryLabel>,
    /// The number of truth FN records per variant type.
    pub fn_counts: IndexMap<VariantType, u64>,
    /// The `(chrom, pos, variant type)` of each truth FN record.
    pub fn_positions: HashSet<LabelKey>,
    /// Per `(chrom, pos, variant type)` of the truth records: the truth alleles' length changes
    /// (`None` when an allele is not a sequence (see [`crate::alleles::allele_deltas`]): symbolic,
    /// a breakend or `*`).
    pub truth_deltas: IndexMap<LabelKey, Option<(i64, i64)>>,
    /// The query records that hap.py did not score TP or FP (e.g. `UNK`, outside the confident
    /// regions), per contig. They label no call, but show which calls hap.py was given (see
    /// [`attach_labels`]). Keyed by contig first, so that the records (about a million for a
    /// whole-genome sample) do not each hold a copy of its name.
    pub unscored: HashMap<String, UnscoredRecords>,
}

impl HappyLabels {
    /// No labels, and zero FN counts for SNPs and indels.
    fn empty() -> HappyLabels {
        HappyLabels {
            fn_counts: [(VariantType::Snp, 0), (VariantType::Indel, 0)].into_iter().collect(),
            ..HappyLabels::default()
        }
    }

    /// The length changes of the unscored query records at `(chrom, pos, variant type)`, if any
    /// (see [`HappyLabels::unscored`]).
    #[must_use]
    pub fn unscored_deltas(
        &self,
        chrom: &str,
        pos: u64,
        variant_type: VariantType,
    ) -> Option<&[i64]> {
        self.unscored.get(chrom)?.get(&(pos, variant_type)).map(AsRef::as_ref)
    }

    /// Every query record, by `(chrom, pos, variant type)`, with the length changes of its query
    /// alleles: the TP and FP records (in record order), then the unscored ones.
    fn query_records(&self) -> impl Iterator<Item = (KeyRef<'_>, &[i64])> {
        let scored = self.labels.iter().map(|((chrom, pos, variant_type), query_label)| {
            (KeyRef(chrom, *pos, *variant_type), query_label.alt_deltas.as_slice())
        });
        let unscored = self.unscored.iter().flat_map(|(chrom, records)| {
            records.iter().map(|(&(pos, variant_type), deltas)| {
                (KeyRef(chrom, pos, variant_type), deltas.as_ref())
            })
        });
        scored.chain(unscored)
    }

    /// The contigs with a query record (TP, FP or unscored): those hap.py scored calls on.
    fn query_contigs(&self) -> HashSet<&str> {
        let mut contigs: HashSet<&str> = self.unscored.keys().map(String::as_str).collect();
        let mut last = None;
        for (chrom, _, _) in self.labels.keys() {
            if last != Some(chrom) {
                contigs.insert(chrom);
                last = Some(chrom);
            }
        }
        contigs
    }

    /// Adds the labels of other contigs (keys are disjoint, as they include the contig).
    fn merge(&mut self, other: HappyLabels) {
        self.labels.extend(other.labels);
        for (variant_type, n) in other.fn_counts {
            *self.fn_counts.entry(variant_type).or_default() += n;
        }
        self.fn_positions.extend(other.fn_positions);
        self.truth_deltas.extend(other.truth_deltas);
        self.unscored.extend(other.unscored);
    }
}

/// The variant type of a hap.py BVT, if it is SNP or INDEL.
pub fn variant_type_of_bvt(bvt: &str) -> Option<VariantType> {
    match bvt {
        "SNP" => Some(VariantType::Snp),
        "INDEL" => Some(VariantType::Indel),
        _ => None,
    }
}

/// Derives query-call labels and truth FN counts from hap.py records on `contigs`.
///
/// When several records share `(chrom, pos)` and query variant type, the label is `Keep` if all
/// of them are TP and otherwise the label of the first that is not, and `flip_ok` is true if any
/// of them is a correct flip.
///
/// # Errors
///
/// Returns an error if no record is on the contigs (e.g. `1` vs `chr1` naming).
pub fn labels_from_happy_records<I, S>(records: I, contigs: &[S]) -> Result<HappyLabels>
where
    I: IntoIterator<Item = Result<HappyRecord>>,
    S: AsRef<str>,
{
    let wanted: HashSet<&str> = contigs.iter().map(AsRef::as_ref).collect();
    let mut labels = HappyLabels::empty();
    let n_on_contigs = add_happy_records(&mut labels, records, &wanted)?;
    finish_happy_labels(labels, n_on_contigs, &wanted)
}

/// Adds the labels of hap.py records on the `wanted` contigs to `labels`, in record order, and
/// returns how many records were on those contigs.
fn add_happy_records<I>(
    labels: &mut HappyLabels,
    records: I,
    wanted: &HashSet<&str>,
) -> Result<usize>
where
    I: IntoIterator<Item = Result<HappyRecord>>,
{
    let mut n_on_contigs = 0usize;
    for record in records {
        let record = record?;
        if !wanted.contains(record.chrom.as_str()) {
            continue;
        }
        n_on_contigs += 1;
        let truth_type = variant_type_of_bvt(&record.truth_bvt);
        if let Some(truth_type) = truth_type
            && !record.truth_gt.is_missing()
        {
            labels
                .truth_deltas
                .entry((record.chrom.clone(), record.pos, truth_type))
                .or_insert_with(|| allele_deltas(&record.alleles, record.truth_gt));
        }
        if record.truth_bd == "FN"
            && let Some(truth_type) = truth_type
        {
            *labels.fn_counts.entry(truth_type).or_default() += 1;
            labels.fn_positions.insert((record.chrom.clone(), record.pos, truth_type));
        }
        let Some(query_type) = variant_type_of_bvt(&record.query_bvt) else { continue };
        let alt_deltas = query_alt_deltas(&record);
        if record.query_bd != "TP" && record.query_bd != "FP" {
            let records = labels.unscored.entry(record.chrom).or_default();
            match records.entry((record.pos, query_type)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(alt_deltas.into_boxed_slice());
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let mut deltas = entry.get().to_vec();
                    deltas.extend(alt_deltas);
                    deltas.sort_unstable();
                    deltas.dedup();
                    entry.insert(deltas.into_boxed_slice());
                }
            }
            continue;
        }
        let flip_ok = record.query_bd == "FP"
            && record.query_bk == "am"
            && record.query_gt.flipped() == Some(record.truth_gt);
        let label = if record.query_bd == "TP" {
            Label::Keep
        } else if record.query_bk == "am" {
            Label::Flip
        } else {
            Label::Drop
        };
        let query_label = QueryLabel { label, flip_ok, alt_deltas };
        match labels.labels.entry((record.chrom, record.pos, query_type)) {
            indexmap::map::Entry::Occupied(mut entry) => entry.get_mut().add(query_label),
            indexmap::map::Entry::Vacant(entry) => {
                entry.insert(query_label);
            }
        }
    }
    Ok(n_on_contigs)
}

/// The sorted, distinct length changes of a hap.py record's query alleles, other than the
/// reference and `*`.
fn query_alt_deltas(record: &HappyRecord) -> Vec<i64> {
    let mut alt_deltas: Vec<i64> = match record.query_gt.alleles() {
        Some((first, second)) => [first, second]
            .into_iter()
            .filter(|&index| index > 0)
            .filter_map(|index| record.alleles.get(index as usize))
            .filter_map(|allele| length_change(record.alleles[0].as_bytes(), allele))
            .collect(),
        None => Vec::new(),
    };
    alt_deltas.sort_unstable();
    alt_deltas.dedup();
    alt_deltas
}

/// Checks that some hap.py records were on the contigs, and keeps `flip_ok` only for flips.
fn finish_happy_labels(
    mut labels: HappyLabels,
    n_on_contigs: usize,
    wanted: &HashSet<&str>,
) -> Result<HappyLabels> {
    if n_on_contigs == 0 {
        let mut sorted: Vec<&str> = wanted.iter().copied().collect();
        sorted.sort_unstable();
        bail!("no hap.py records on the extracted contigs: {}", sorted.join(", "));
    }
    // Only a flip-labelled call can be a correct flip.
    for query_label in labels.labels.values_mut() {
        query_label.flip_ok &= query_label.label == Label::Flip;
    }
    Ok(labels)
}

/// A truth FN record of a hap.py output VCF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HappyFn {
    /// The contig.
    pub chrom: String,
    /// The variant type (from the TRUTH BVT).
    pub variant_type: VariantType,
    /// The 0-based, half-open span of its REF allele (see [`crate::regions::ref_span`]).
    pub span: (u64, u64),
}

/// The truth FN records of a hap.py output VCF that are SNPs or indels, on the contigs `on`
/// accepts.
///
/// # Errors
///
/// Returns an error if the VCF cannot be read or is not hap.py output.
pub fn read_happy_fns<F: Fn(&str) -> bool>(
    path: &Path,
    on: F,
) -> Result<impl Iterator<Item = Result<HappyFn>>> {
    Ok(read_happy_records(path)?.filter_map(move |record| {
        let record = match record {
            Ok(record) => record,
            Err(e) => return Some(Err(e)),
        };
        if record.truth_bd != "FN" || !on(&record.chrom) {
            return None;
        }
        let variant_type = variant_type_of_bvt(&record.truth_bvt)?;
        let ref_len = record.alleles.first().map_or(1, String::len);
        let pos = match i64::try_from(record.pos) {
            Ok(pos) => pos,
            Err(e) => return Some(Err(e.into())),
        };
        let span = crate::regions::ref_span(pos, ref_len);
        Some(Ok(HappyFn { chrom: record.chrom, variant_type, span }))
    }))
}

/// Reads the records of a hap.py output VCF (with TRUTH and QUERY samples).
///
/// # Errors
///
/// Returns an error if the VCF cannot be read, or does not have TRUTH and QUERY samples or the
/// GT, BD, BK and BVT FORMAT fields.
pub fn read_happy_records(path: &Path) -> Result<impl Iterator<Item = Result<HappyRecord>>> {
    let mut reader =
        Reader::from_path(path).with_context(|| format!("failed to open {}", path.display()))?;
    let (truth, query) = happy_samples(reader.header(), path)?;
    let path = path.to_path_buf();
    let mut record = reader.empty_record();
    let mut failed = false;
    Ok(std::iter::from_fn(move || {
        if failed {
            return None;
        }
        match reader.read(&mut record)? {
            Ok(()) => Some(happy_record(&record, truth, query)),
            Err(e) => {
                failed = true;
                Some(Err(
                    anyhow::Error::new(e).context(format!("failed to read {}", path.display()))
                ))
            }
        }
    }))
}

/// The indices of the TRUTH and QUERY samples of a hap.py output VCF.
///
/// # Errors
///
/// Returns an error if the header lacks the TRUTH and QUERY samples or the GT, BD, BK and BVT
/// FORMAT fields.
fn happy_samples(header: &HeaderView, path: &Path) -> Result<(usize, usize)> {
    let samples = crate::vcf::sample_names(header);
    let index_of = |name: &str| samples.iter().position(|s| s == name);
    let missing_formats: Vec<&str> = ["GT", "BD", "BK", "BVT"]
        .into_iter()
        .filter(|tag| !crate::vcf::has_format(header, tag.as_bytes()))
        .collect();
    let (Some(truth), Some(query)) = (index_of("TRUTH"), index_of("QUERY")) else {
        bail!(
            "{} does not look like hap.py output: it lacks TRUTH and QUERY samples",
            path.display()
        );
    };
    if !missing_formats.is_empty() {
        bail!(
            "{} does not look like hap.py output: it lacks the FORMAT fields {}",
            path.display(),
            missing_formats.join(", ")
        );
    }
    Ok((truth, query))
}

/// The labelling fields of one hap.py record.
fn happy_record(
    record: &rust_htslib::bcf::Record,
    truth: usize,
    query: usize,
) -> Result<HappyRecord> {
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
    // Each FORMAT field is decoded once, for both samples.
    let string_fields = |tag: &[u8], wanted: [usize; 2]| -> [String; 2] {
        let values = record.format(tag).string();
        wanted.map(|sample| match &values {
            Ok(values) => values
                .get(sample)
                .map(|v| text(v))
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| ".".to_string()),
            Err(_) => ".".to_string(),
        })
    };
    let genotypes = record.genotypes();
    let genotype = |sample: usize| match &genotypes {
        Ok(genotypes) => Genotype::from_alleles(
            genotypes.get(sample).iter().map(|allele| allele.index()),
            record.allele_count() as usize,
        ),
        Err(_) => Genotype::MISSING,
    };
    let header = record.header();
    let chrom = record
        .rid()
        .and_then(|rid| header.rid2name(rid).ok())
        .map(text)
        .context("a hap.py record has no contig")?;
    let [truth_bd, query_bd] = string_fields(b"BD", [truth, query]);
    let [truth_bvt, query_bvt] = string_fields(b"BVT", [truth, query]);
    let [query_bk, _] = string_fields(b"BK", [query, query]);
    Ok(HappyRecord {
        chrom,
        pos: u64::try_from(record.pos() + 1)?,
        truth_gt: genotype(truth),
        truth_bd,
        truth_bvt,
        query_gt: genotype(query),
        query_bd,
        query_bk,
        query_bvt,
        alleles: record.alleles().into_iter().map(text).collect(),
    })
}

/// Reads query-call labels and truth FN counts from a hap.py output VCF.
///
/// # Errors
///
/// Returns an error if the VCF is not hap.py output, or has no record on the contigs.
///
/// An indexed VCF is read one contig per task, in parallel on the rayon pool; otherwise it is
/// streamed. Both give the same labels: they depend only on the order of records within a contig.
pub fn read_happy_labels<S: AsRef<str> + Sync>(path: &Path, contigs: &[S]) -> Result<HappyLabels> {
    let labels = if has_index(path) {
        read_happy_labels_by_contig(path, contigs)
    } else {
        labels_from_happy_records(read_happy_records(path)?, contigs)
    };
    labels.with_context(|| format!("{}", path.display()))
}

/// [`read_happy_labels`] of an indexed VCF: each contig is fetched and labelled by its own task,
/// and the per-contig labels (whose keys are disjoint) are merged.
fn read_happy_labels_by_contig<S: AsRef<str> + Sync>(
    path: &Path,
    contigs: &[S],
) -> Result<HappyLabels> {
    let (truth, query) = happy_samples(open_indexed(path)?.header(), path)?;
    let wanted: HashSet<&str> = contigs.iter().map(AsRef::as_ref).collect();
    let mut unique: Vec<&str> = Vec::with_capacity(contigs.len());
    for contig in contigs {
        if !unique.contains(&contig.as_ref()) {
            unique.push(contig.as_ref());
        }
    }
    let partials = unique
        .par_iter()
        .map(|contig| -> Result<(HappyLabels, usize)> {
            let mut labels = HappyLabels::empty();
            let mut reader = open_indexed(path)?;
            // A contig missing from the header (or the index) has no records.
            let Ok(rid) = reader.header().name2rid(contig.as_bytes()) else {
                return Ok((labels, 0));
            };
            if reader.fetch(rid, 0, None).is_err() {
                return Ok((labels, 0));
            }
            let mut record = reader.empty_record();
            let records = std::iter::from_fn(|| {
                reader.read(&mut record).map(|read| {
                    read.with_context(|| format!("failed to read {}", path.display()))
                        .and_then(|()| happy_record(&record, truth, query))
                })
            });
            let n = add_happy_records(&mut labels, records, &wanted)?;
            Ok((labels, n))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut labels = HappyLabels::empty();
    let mut n_on_contigs = 0;
    for (partial, n) in partials {
        labels.merge(partial);
        n_on_contigs += n;
    }
    finish_happy_labels(labels, n_on_contigs, &wanted)
}

/// Whether hap.py was run with `--pass-only`, which decides which calls its labels can belong to
/// (see [`attach_labels`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, clap::ValueEnum)]
pub enum HappyPassOnly {
    /// Infer it from the labels.
    #[value(help = None::<&str>)]
    #[default]
    Auto,
    /// hap.py was run with `--pass-only`.
    #[value(help = None::<&str>)]
    Yes,
    /// hap.py was run without `--pass-only`.
    #[value(help = None::<&str>)]
    No,
}

impl HappyPassOnly {
    /// The lowercase name, as written on the command line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            HappyPassOnly::Auto => "auto",
            HappyPassOnly::Yes => "yes",
            HappyPassOnly::No => "no",
        }
    }
}

impl std::fmt::Display for HappyPassOnly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the hap.py query labels joined to a feature table's calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LabelJoin {
    /// Whether the labels were joined as hap.py run with `--pass-only`.
    pub pass_only: bool,
    /// How that was decided: given ([`HappyPassOnly::Yes`] or [`HappyPassOnly::No`]) or inferred
    /// ([`HappyPassOnly::Auto`]).
    pub happy_pass_only: HappyPassOnly,
    /// hap.py query records scored TP or FP (one per `(chrom, pos, variant type)`).
    pub n_labels: usize,
    /// Those that labelled a PASS call, or that a PASS call matched without taking the label
    /// (as its label at its key was left out; see [`attach_labels`]).
    pub n_pass: usize,
    /// Of `n_pass`, those found at the differing base of a padded SNP.
    pub n_padded_snp: usize,
    /// Of `n_pass`, those found at the trimmed position of a padded biallelic indel.
    pub n_padded_indel: usize,
    /// Of `n_pass`, those that hap.py split from a multi-allelic call.
    pub n_multiallelic: usize,
    /// Those that labelled no PASS call but are a non-PASS call's: at its key, at the position a
    /// non-PASS padded call was trimmed to, or split from a non-PASS multi-allelic call (as for
    /// [`LabelJoin::n_non_pass_unexplained`]). They label no call.
    pub n_non_pass: usize,
    /// Those at no call.
    pub n_unmatched: usize,
    /// The non-PASS calls that hap.py could have scored, one per `(chrom, pos, variant type)`
    /// (a PASS call may be at it too): those with a non-reference allele in their genotype, on a contig
    /// with a hap.py query record (TP, FP or unscored). A haploid or partly missing genotype is
    /// `./.` in the table, so those calls are not counted.
    pub n_non_pass_calls: usize,
    /// Of those, the ones that hap.py evidently scored: with a query record (TP, FP or unscored) at
    /// their key that no PASS call explains, or the call that such a record at no call's key (or at
    /// a PASS call's, with a length change that no PASS indel call covering the key has) belongs
    /// to: the non-PASS padded call trimmed to it, with no record at its own POS, or else the
    /// non-PASS multi-allelic call that [`attach_labels`] would take as its source, each chosen
    /// among the non-PASS calls that hap.py could have scored. With
    /// [`HappyPassOnly::Auto`], hap.py was run without `--pass-only` if they are at least
    /// [`MIN_NON_PASS_RECORD_FRACTION`] of `n_non_pass_calls`.
    pub n_non_pass_unexplained: usize,
}

/// Whether `n_unexplained` of the `n_calls` non-PASS calls that hap.py could have scored show
/// that it was run with `--pass-only`: fewer than [`MIN_NON_PASS_RECORD_FRACTION`] of them (see
/// [`LabelJoin::n_non_pass_unexplained`]). With no such calls, it was.
fn looks_pass_only(n_unexplained: usize, n_calls: usize) -> bool {
    n_calls == 0 || (n_unexplained as f64) < MIN_NON_PASS_RECORD_FRACTION * n_calls as f64
}

/// The length changes of a merged query record's alleles, one at a time (`None` if it has none).
fn each_delta(deltas: &[i64]) -> impl Iterator<Item = Option<i64>> + '_ {
    deltas.is_empty().then_some(None).into_iter().chain(deltas.iter().copied().map(Some))
}

/// Whether a genotype from a feature table's `gt` column has a non-reference allele. The column
/// holds the normalized diploid genotype (see [`Genotype`]: e.g. `0/1` or `1/2`), where a haploid
/// or partly missing genotype is `./.`, so such a call does not have one, nor does `0/0`. No other
/// column tells a haploid ALT call from a no-call reliably.
fn has_alt_allele(gt: &str) -> bool {
    gt.parse::<Genotype>().ok().and_then(Genotype::alleles).is_some_and(|(_, high)| high > 0)
}

/// The offset of the differing base of a SNP record (0 if its alternates differ at other bases).
fn snp_offset(reference: &str, alt: &str) -> usize {
    let offsets: HashSet<usize> = alt
        .split(',')
        .filter(|allele| *allele != "*" && allele.len() == reference.len() && *allele != reference)
        .filter_map(|allele| reference.bytes().zip(allele.bytes()).position(|(r, a)| r != a))
        .collect();
    if offsets.len() == 1 { offsets.into_iter().next().unwrap_or(0) } else { 0 }
}

/// A call with more than one ALT allele (`*` included).
#[derive(Debug, Clone, PartialEq, Eq)]
struct MultiAllelicCall {
    /// The call's POS.
    start: u64,
    /// The position of the last base of its REF allele.
    end: u64,
    /// Its row in the feature table.
    row: usize,
    /// The length changes of its called ALT alleles other than `*` (all of them, if its genotype
    /// is missing).
    alt_deltas: Vec<i64>,
}

/// The multi-allelic calls of a feature table, to find the call that hap.py split a record from.
#[derive(Debug, Default)]
struct MultiAllelicCalls<'a> {
    /// Per contig: its multi-allelic calls, sorted by POS (then row), and the longest REF allele.
    by_contig: HashMap<&'a str, (Vec<MultiAllelicCall>, u64)>,
}

impl<'a> MultiAllelicCalls<'a> {
    /// Indexes the calls with more than one ALT allele, by the ALT alleles their genotype
    /// (`gts`, normalized) calls: hap.py writes no record for an uncalled allele. A missing
    /// genotype (`./.`, which a haploid or partly missing one is too) may call any of them, so all
    /// are indexed.
    fn new(
        chroms: &[&'a str],
        positions: &[i64],
        refs: &[&str],
        alts: &[&str],
        gts: &[&str],
    ) -> Self {
        let mut index = MultiAllelicCalls::default();
        for row in (0..chroms.len()).filter(|&row| alts[row].contains(',')) {
            let reference = refs[row].as_bytes();
            let start = positions[row] as u64;
            let ref_len = reference.len().max(1) as u64;
            let called = gts[row].parse::<Genotype>().ok().and_then(Genotype::alleles);
            let is_called = |allele: usize| {
                called.is_none_or(|(first, second)| [first, second].contains(&(allele as u32 + 1)))
            };
            let alt_deltas: Vec<i64> = alts[row]
                .split(',')
                .enumerate()
                .filter(|&(allele, _)| is_called(allele))
                .filter_map(|(_, alt)| length_change(reference, alt))
                .collect();
            if alt_deltas.is_empty() {
                continue;
            }
            let (calls, longest) = index.by_contig.entry(chroms[row]).or_default();
            calls.push(MultiAllelicCall { start, end: start + ref_len - 1, row, alt_deltas });
            *longest = (*longest).max(ref_len);
        }
        for (calls, _) in index.by_contig.values_mut() {
            calls.sort_unstable_by_key(|call| (call.start, call.row));
        }
        index
    }

    /// The row of the call that hap.py split a record at `pos` from: the multi-allelic call whose
    /// REF allele covers `pos`, that calls ALT alleles with every length change of the record's
    /// query alleles, and whose row is `eligible`. Of several, the one with the greatest POS, then
    /// the first row.
    fn owner(
        &self,
        chrom: &str,
        pos: u64,
        alt_deltas: &[i64],
        eligible: impl Fn(usize) -> bool,
    ) -> Option<usize> {
        if alt_deltas.is_empty() {
            return None;
        }
        let (calls, longest) = self.by_contig.get(chrom)?;
        let earliest = pos.saturating_sub(*longest);
        let after = calls.partition_point(|call| call.start <= pos);
        calls[..after]
            .iter()
            .rev()
            .take_while(|call| call.start >= earliest)
            .filter(|call| {
                call.end >= pos
                    && alt_deltas.iter().all(|delta| call.alt_deltas.contains(delta))
                    && eligible(call.row)
            })
            .max_by_key(|call| (call.start, std::cmp::Reverse(call.row)))
            .map(|call| call.row)
    }
}

/// Adds `label` (NaN where unlabelled), `flip_ok` and the truth allele length changes to a
/// feature table.
///
/// Labels attach to PASS calls, and truth allele length changes to every call, by `(chrom, pos,
/// variant type)`. A PASS indel call takes the label at its key only if every length change of the
/// records there is that of an ALT allele of a PASS indel call whose REF allele covers the key
/// (itself included): hap.py may write one record that joins its allele with one of an overlapping
/// PASS call, but a length change of no such call is another call's record, and the merged label is
/// then not its own, so it is left unlabelled (and a record split from it does not label it
/// either). SNP records all have a length change of 0, so this cannot tell them apart. A padded
/// call with no record at its POS (TP, FP or unscored) takes the label at the position hap.py may
/// have trimmed it to, unless a PASS call of its variant type is at that position (or any call of
/// its variant type, when hap.py was run without `--pass-only`; see below): a padded SNP whose
/// differing base is not its first (e.g. `TTAGA>TTCGA`) the label at that base, and a padded
/// biallelic indel (e.g. `ACGT>ACT`, trimmed to `CG>C` one base on) the label at the base before
/// its trimmed alleles, if the records there have its length change only (with another, they
/// include another call's record, and their label is not its own).
///
/// A label that matches no PASS call (of its variant type, at its position) labels the PASS
/// multi-allelic call that hap.py split it from, if there is one: a PASS call with more than one
/// ALT allele whose REF allele covers the label's position, and whose called ALT alleles (all of
/// them, if its genotype is missing) have the length changes of the label's query alleles (of
/// several, the one with the greatest POS). Such a label
/// makes a call unlabelled at its own key take its label, and a `Keep` call take a label other
/// than `Keep` (a call is `Keep` only if every allele is TP). It does not set `flip_ok`, which it
/// computed for the split record's genotype rather than the call's.
///
/// hap.py run with `--pass-only` labels only PASS calls, so a label at a non-PASS call's key is
/// then a PASS call's: a padded call's, or a multi-allelic call's (whose source is chosen among
/// PASS calls only). Without `--pass-only`, a label at a non-PASS call's key is that call's, and
/// a multi-allelic source is chosen among all calls, labelling it only if it is PASS. hap.py's
/// output does not say which way it was run, so `happy_pass_only` does: [`HappyPassOnly::Yes`]
/// or [`HappyPassOnly::No`], or [`HappyPassOnly::Auto`] to infer it from how many of the
/// non-PASS calls that hap.py could have scored it evidently did. It could have scored those
/// with a non-reference allele in their genotype, on a contig with a hap.py query record (one
/// per key, though a PASS call may be at it too). It evidently scored those with a query record at their key
/// (TP, FP or unscored) that no PASS call explains, the non-PASS multi-allelic call that a
/// record at no call's key was most likely split from (of several, the one with the greatest
/// POS, as for labels), and the non-PASS padded call with a record at the position hap.py trims
/// it to (as for a PASS one, with no call there). A PASS call explains a record at its key with
/// a length change that it, or a PASS indel call covering the key, has (any record, for a
/// SNP), split from it as a multi-allelic call, or at the position it was trimmed to as a padded call with no record at
/// its POS (as for labels, but counting unscored records at the trimmed position). Records at one
/// key are merged, so each length change of their query alleles is explained, and attributed to
/// a non-PASS call, on its own. hap.py was run without `--pass-only` if they are at least
/// [`MIN_NON_PASS_RECORD_FRACTION`] of the calls it could have scored, and with it otherwise, as
/// one is then most likely a PASS call that hap.py represented differently. Run without
/// `--pass-only` and inferred to be run with it, a non-PASS call's label may go to a PASS call:
/// hap.py should be run with `--pass-only`.
///
/// # Errors
///
/// Returns an error if the table lacks the key columns.
pub fn attach_labels(
    table: FeatureTable,
    labels: &HappyLabels,
    happy_pass_only: HappyPassOnly,
) -> Result<(FeatureTable, LabelJoin)> {
    let n = table.num_rows();
    let (chroms, positions) = (table.str_column("chrom")?, table.i64_column("pos")?);
    let (types, filters) = (table.str_column("variant_type")?, table.str_column("filter")?);
    let (refs, alts, gts) =
        (table.str_column("ref")?, table.str_column("alt")?, table.str_column("gt")?);
    let variant_types: Vec<Option<VariantType>> =
        types.par_iter().map(|t| t.parse().ok()).collect();
    let key = |i: usize, pos: u64| variant_types[i].map(|vt| KeyRef(chroms[i], pos, vt));

    let on_pass_call: HashSet<KeyRef<'_>> = (0..n)
        .filter(|&i| filters[i] == PASS)
        .filter_map(|i| key(i, positions[i] as u64))
        .collect();
    let on_call: HashSet<KeyRef<'_>> = (0..n).filter_map(|i| key(i, positions[i] as u64)).collect();
    // The position that hap.py may have trimmed a PASS padded call to, and the length change a
    // record there must have (for an indel): for a padded SNP (`TTAGA>TTCGA`), its differing
    // base; for a padded biallelic indel (`ACGT>ACT`), the base before its trimmed alleles.
    let trimmed = |i: usize| -> Option<(KeyRef<'_>, Option<i64>)> {
        let variant_type = variant_types[i]?;
        let (offset, delta) = match variant_type {
            VariantType::Snp => (snp_offset(refs[i], alts[i]), None),
            VariantType::Indel if !alts[i].contains(',') => {
                let delta = length_change(refs[i].as_bytes(), alts[i])?;
                let (first_differing, _, _) = trim_alleles(refs[i].as_bytes(), alts[i].as_bytes());
                (first_differing.saturating_sub(1), Some(delta))
            }
            VariantType::Indel => return None,
        };
        let shifted = KeyRef(chroms[i], positions[i] as u64 + offset as u64, variant_type);
        (offset > 0).then_some((shifted, delta))
    };
    // The same for a PASS padded call, if no other PASS call of its type is there.
    let padded_shift = |i: usize| {
        trimmed(i).filter(|(shifted, _)| filters[i] == PASS && !on_pass_call.contains(shifted))
    };
    // A padded call has its record at the position hap.py trims it to only if it has none at its
    // POS (TP, FP or unscored).
    let has_record_at_pos = |i: usize, variant_type: VariantType| {
        let (chrom, pos) = (chroms[i], positions[i] as u64);
        labels.labels.contains_key(&KeyRef(chrom, pos, variant_type))
            || labels.unscored_deltas(chrom, pos, variant_type).is_some()
    };
    // The position whose label a PASS padded call with no record at its POS takes, if a label
    // there may be its: no other PASS call of its type is there (a non-PASS one may be, see
    // `pass_only`), and its records have the indel's length change only (with another, the
    // merged label includes another call's record).
    let padded_source = |i: usize| -> Option<u64> {
        let (shifted, delta) = padded_shift(i)?;
        (!has_record_at_pos(i, shifted.2)
            && labels
                .labels
                .get(&shifted)
                .is_some_and(|label| delta.is_none_or(|delta| label.alt_deltas == [delta])))
        .then_some(shifted.1)
    };
    let multiallelic = MultiAllelicCalls::new(&chroms, positions, &refs, &alts, &gts);
    let is_pass = |row: usize| filters[row] == PASS;
    // The length changes of the ALT alleles of the PASS indel calls whose REF allele covers each
    // PASS indel call's key (hap.py may write an overlapping PASS call's allele there).
    let pass_indels: Vec<usize> = (0..n)
        .filter(|&i| filters[i] == PASS && variant_types[i] == Some(VariantType::Indel))
        .collect();
    // Per contig: the PASS indel calls' `(POS, end of REF, row)`, sorted, and the longest REF.
    type Spans = (Vec<(u64, u64, usize)>, u64);
    let mut spans: HashMap<&str, Spans> = HashMap::new();
    for &i in &pass_indels {
        let (start, len) = (positions[i] as u64, refs[i].len().max(1) as u64);
        let (calls, longest) = spans.entry(chroms[i]).or_default();
        calls.push((start, start + len - 1, i));
        *longest = (*longest).max(len);
    }
    for (calls, _) in spans.values_mut() {
        calls.sort_unstable();
    }
    let mut pass_deltas: HashMap<KeyRef<'_>, Vec<i64>> = HashMap::new();
    for &i in &pass_indels {
        let Some(call_key) = key(i, positions[i] as u64) else { continue };
        let (calls, longest) = &spans[chroms[i]];
        let pos = call_key.1;
        let after = calls.partition_point(|&(start, _, _)| start <= pos);
        let deltas = pass_deltas.entry(call_key).or_default();
        for &(_, end, j) in calls[..after].iter().rev().take_while(|c| c.0 + longest > pos) {
            if end >= pos {
                deltas.extend(
                    alts[j].split(',').filter_map(|alt| length_change(refs[j].as_bytes(), alt)),
                );
            }
        }
    }
    // Whether a record at a PASS call's key, with a length change (`None` if not known), can be
    // that call's: any SNP record can, and an indel record if a PASS indel call covering the key
    // (it included) has the length change.
    let pass_call_has = |record_key: &KeyRef<'_>, delta: Option<i64>| {
        on_pass_call.contains(record_key)
            && (record_key.2 == VariantType::Snp
                || delta
                    .is_none_or(|d| pass_deltas.get(record_key).is_some_and(|ds| ds.contains(&d))))
    };

    // hap.py run with --pass-only labels only PASS calls, so a label at a non-PASS call's key is
    // then a PASS call's: a padded call's, or that of a PASS multi-allelic call it was split from.
    // Without --pass-only, it is the non-PASS call's own. hap.py's output does not say which, so
    // unless told, it is inferred from how many of the non-PASS calls that hap.py could have
    // scored (with a non-reference allele, on a contig with a query record) it evidently did:
    // those that hap.py wrote a query record (TP, FP or unscored) for that no PASS call explains.
    // Each counts once, by its key. The counts are logged whichever way the labels are joined.
    let query_contigs = labels.query_contigs();
    let non_pass_keys: HashSet<KeyRef<'_>> = (0..n)
        .filter(|&i| filters[i] != PASS && has_alt_allele(gts[i]))
        .filter(|&i| query_contigs.contains(chroms[i]))
        .filter_map(|i| key(i, positions[i] as u64))
        .collect();
    // Each query record is taken by its key and one length change of its query alleles (records
    // at one key are merged, so a match on one length change must not hide another).
    let records = labels
        .query_records()
        .flat_map(|(record_key, deltas)| each_delta(deltas).map(move |delta| (record_key, delta)));
    // The record (key and length change, `None` for a SNP) at the position each padded call was
    // trimmed to, with its row (a PASS one's, if both a PASS and a non-PASS call are trimmed to
    // it).
    let scorable = |row: usize| {
        key(row, positions[row] as u64).is_some_and(|call_key| non_pass_keys.contains(&call_key))
    };
    let padded_rank = |row: usize| (is_pass(row), scorable(row));
    let mut padded_records: HashMap<(KeyRef<'_>, Option<i64>), usize> = HashMap::new();
    for (i, &filter) in filters.iter().enumerate() {
        let shifted = if filter == PASS { padded_shift(i) } else { trimmed(i) };
        if let Some((shifted, delta)) = shifted
            && !has_record_at_pos(i, shifted.2)
        {
            // A PASS call first, then a non-PASS call that hap.py could have scored.
            let row = padded_records.entry((shifted, delta)).or_insert(i);
            if padded_rank(i) > padded_rank(*row) {
                *row = i;
            }
        }
    }
    let padded_row = |record_key: KeyRef<'_>, delta: Option<i64>| {
        let snp = padded_records.get(&(record_key, None));
        snp.or_else(|| delta.and_then(|d| padded_records.get(&(record_key, Some(d))))).copied()
    };
    // Whether a PASS call explains a record: it is at a PASS call's key with a length change of
    // its ALT alleles, at the position a PASS padded call was trimmed to, or split from a PASS
    // multi-allelic call.
    let explained_by_pass = |record_key: KeyRef<'_>, delta: Option<i64>| {
        let KeyRef(chrom, pos, _) = record_key;
        pass_call_has(&record_key, delta)
            || padded_row(record_key, delta).is_some_and(is_pass)
            || multiallelic.owner(chrom, pos, delta.as_slice(), is_pass).is_some()
    };
    // The non-PASS call a record that no PASS call explains is from: the non-PASS call at its
    // key, or (with none there) the non-PASS padded call trimmed to it, or the non-PASS
    // multi-allelic call it was split from. The evidence and the label counts both use it.
    let on_non_pass_call: HashSet<KeyRef<'_>> = (0..n)
        .filter(|&i| filters[i] != PASS)
        .filter_map(|i| key(i, positions[i] as u64))
        .collect();
    // With `only_scorable`, the padded and multi-allelic sources are chosen among the non-PASS
    // calls that hap.py could have scored, so that one it could not have (e.g. `./.`) does not
    // hide one it could.
    let non_pass_source = |record_key, delta: Option<i64>, only_scorable: bool| {
        let KeyRef(chrom, pos, _) = record_key;
        if on_non_pass_call.contains(&record_key) {
            return Some(record_key);
        }
        let eligible = |row: usize| !is_pass(row) && (!only_scorable || scorable(row));
        let row = padded_row(record_key, delta)
            .filter(|&row| eligible(row))
            .or_else(|| multiallelic.owner(chrom, pos, delta.as_slice(), eligible))?;
        key(row, positions[row] as u64)
    };
    // It counts as scored if hap.py could have scored it.
    let scored_call = |record_key, delta: Option<i64>| {
        non_pass_source(record_key, delta, true).filter(|call_key| non_pass_keys.contains(call_key))
    };
    let unexplained: HashSet<KeyRef<'_>> = records
        .filter(|&(record_key, delta)| !explained_by_pass(record_key, delta))
        .filter_map(|(record_key, delta)| scored_call(record_key, delta))
        .collect();
    let (n_non_pass_calls, n_non_pass_unexplained) = (non_pass_keys.len(), unexplained.len());
    let pass_only = match happy_pass_only {
        HappyPassOnly::Yes => true,
        HappyPassOnly::No => false,
        HappyPassOnly::Auto => looks_pass_only(n_non_pass_unexplained, n_non_pass_calls),
    };

    // Per call: its label, flip_ok, truth deltas, the position its label came from, and whether
    // that was a padded SNP's differing base. Calls are labelled independently, in parallel.
    let mut rows: Vec<CallLabel> = (0..n)
        .into_par_iter()
        .map(|i| {
            let pos = positions[i] as u64;
            let mut row = (f64::NAN, false, (f64::NAN, f64::NAN), pos, false);
            let Some(call_key) = key(i, pos) else { return row };
            if let Some(Some((lo, hi))) = labels.truth_deltas.get(&call_key) {
                row.2 = (*lo as f64, *hi as f64);
            }
            if filters[i] != PASS {
                return row;
            }
            if let Some(query_label) = labels.labels.get(&call_key) {
                // A length change that no PASS indel call covering the key has is another call's
                // record, and the merged label is then not this call's own.
                if query_label.alt_deltas.iter().all(|&d| pass_call_has(&call_key, Some(d))) {
                    (row.0, row.1) = (query_label.label.index() as f64, query_label.flip_ok);
                }
            } else if let Some(shifted) = padded_source(i)
                // Without --pass-only, a label at a non-PASS call's key is that call's.
                && (pass_only || !on_call.contains(&KeyRef(chroms[i], shifted, call_key.2)))
                && let Some(query_label) = labels.labels.get(&KeyRef(chroms[i], shifted, call_key.2))
            {
                row = (query_label.label.index() as f64, query_label.flip_ok, row.2, shifted, true);
            }
            row
        })
        .collect();

    // The keys whose label labelled a PASS call (at its key or as a padded call's), and those at a
    // PASS call's key whose label it did not take, as it includes another call's record.
    let labelled_pass = |i: usize| filters[i] == PASS && !rows[i].0.is_nan();
    let on_pass: HashSet<KeyRef<'_>> =
        (0..n).filter(|&i| labelled_pass(i)).filter_map(|i| key(i, rows[i].3)).collect();
    let refused_rows: Vec<bool> = (0..n)
        .map(|i| {
            filters[i] == PASS
                && rows[i].0.is_nan()
                && key(i, positions[i] as u64).is_some_and(|k| labels.labels.contains_key(&k))
        })
        .collect();
    let refused: HashSet<KeyRef<'_>> =
        (0..n).filter(|&i| refused_rows[i]).filter_map(|i| key(i, positions[i] as u64)).collect();
    let (mut n_pass, mut n_multiallelic, mut n_non_pass, mut n_unmatched) = (0, 0, 0, 0);
    // In record order, so that a call takes the first label other than `Keep`.
    for ((chrom, pos, variant_type), query_label) in &labels.labels {
        let label_key = KeyRef(chrom, *pos, *variant_type);
        let deltas = &query_label.alt_deltas;
        let any_owner = || multiallelic.owner(chrom, *pos, deltas, |_| true);
        // With --pass-only, the source is among PASS calls. Without, a label at a non-PASS call's
        // key is its own, and the source is among all calls, labelling it only if it is PASS.
        let pass_owner = || {
            if pass_only {
                multiallelic.owner(chrom, *pos, deltas, is_pass)
            } else if on_call.contains(&label_key) {
                None
            } else {
                any_owner().filter(|&row| is_pass(row))
            }
        };
        if on_pass.contains(&label_key) {
            n_pass += 1;
        } else if let Some(row) = pass_owner().filter(|_| !refused.contains(&label_key)) {
            n_pass += 1;
            n_multiallelic += 1;
            // A call whose label at its key was left out stays unlabelled: its own records there
            // may include an FP that the merged label no longer shows.
            let call = &mut rows[row];
            let is_keep = call.0 == Label::Keep.index() as f64;
            if !refused_rows[row]
                && (call.0.is_nan() || (is_keep && query_label.label != Label::Keep))
            {
                (call.0, call.1) = (query_label.label.index() as f64, false);
            }
        } else if each_delta(deltas).any(|delta| non_pass_source(label_key, delta, false).is_some())
        {
            n_non_pass += 1;
        } else {
            n_unmatched += 1;
        }
    }
    let n_padded = |variant_type| {
        (0..n).filter(|&i| rows[i].4 && variant_types[i] == Some(variant_type)).count()
    };
    let join = LabelJoin {
        pass_only,
        happy_pass_only,
        n_labels: labels.labels.len(),
        n_pass,
        n_padded_snp: n_padded(VariantType::Snp),
        n_padded_indel: n_padded(VariantType::Indel),
        n_multiallelic,
        n_non_pass,
        n_unmatched,
        n_non_pass_calls,
        n_non_pass_unexplained,
    };

    let mut label = Vec::with_capacity(n);
    let mut flip_ok = Vec::with_capacity(n);
    let mut truth_lo = Vec::with_capacity(n);
    let mut truth_hi = Vec::with_capacity(n);
    for (l, ok, (lo, hi), _, _) in rows {
        label.push(l);
        flip_ok.push(ok);
        truth_lo.push(lo);
        truth_hi.push(hi);
    }
    let table = table
        .with_f64_column("label", label)?
        .with_bool_column("flip_ok", flip_ok)?
        .with_f64_column(TRUTH_DELTA_COLUMNS[0], truth_lo)?
        .with_f64_column(TRUTH_DELTA_COLUMNS[1], truth_hi)?;
    Ok((table, join))
}

/// The evidence for how hap.py was run: how many of the non-PASS calls it could have scored have
/// a query record that no PASS call explains.
fn pass_only_evidence(join: &LabelJoin) -> String {
    format!(
        "{} of {} non-PASS calls that hap.py could have scored (with a non-reference allele, on a \
         contig with hap.py query records) have a query record that no PASS call explains",
        join.n_non_pass_unexplained, join.n_non_pass_calls
    )
}

/// Whether the evidence reaches [`MIN_NON_PASS_RECORD_FRACTION`], in words.
fn threshold_side(join: &LabelJoin) -> String {
    let side = if looks_pass_only(join.n_non_pass_unexplained, join.n_non_pass_calls) {
        "under"
    } else {
        "at least"
    };
    format!(
        "{side} the {:.0}% that shows a run without --pass-only",
        100.0 * MIN_NON_PASS_RECORD_FRACTION
    )
}

/// How the labels were joined, as to hap.py's `--pass-only`, and why.
fn pass_only_mode(join: &LabelJoin) -> String {
    let mode = if join.pass_only { "with" } else { "without" };
    let evidence = pass_only_evidence(join);
    match join.happy_pass_only {
        HappyPassOnly::Yes | HappyPassOnly::No => format!(
            "joined as hap.py run {mode} --pass-only, as set by --happy-pass-only {} ({evidence})",
            join.happy_pass_only
        ),
        HappyPassOnly::Auto => format!(
            "joined as hap.py run {mode} --pass-only, inferred from the labels: {evidence} ({})",
            threshold_side(join)
        ),
    }
}

/// The warning, if any, that the labels may have been joined the wrong way: inferred as hap.py
/// run with `--pass-only` although some non-PASS calls have a query record that no PASS call
/// explains, set so by `--happy-pass-only yes` although enough of them do to show a run without
/// it, or set by `--happy-pass-only no` although too few of the non-PASS calls that hap.py could
/// have scored (if any) do.
fn pass_only_warning(join: &LabelJoin) -> Option<String> {
    let evidence = pass_only_evidence(join);
    match join.happy_pass_only {
        HappyPassOnly::Auto if join.pass_only && join.n_non_pass_unexplained > 0 => Some(format!(
            "{evidence} (each most likely a PASS call's record that hap.py placed at a non-PASS \
             call): {}, so the labels were joined as hap.py run with it. If hap.py was run \
             without --pass-only, pass --happy-pass-only no; if with it, pass --happy-pass-only \
             yes",
            threshold_side(join)
        )),
        HappyPassOnly::Yes
            if !looks_pass_only(join.n_non_pass_unexplained, join.n_non_pass_calls) =>
        {
            Some(format!(
                "{evidence} ({}): the labels contradict --happy-pass-only yes",
                threshold_side(join)
            ))
        }
        HappyPassOnly::No
            if join.n_non_pass_calls > 0
                && looks_pass_only(join.n_non_pass_unexplained, join.n_non_pass_calls) =>
        {
            Some(format!(
                "{evidence} ({}): the labels contradict --happy-pass-only no (with it, a PASS \
                 call's record that hap.py placed at a non-PASS call does not label the PASS call)",
                threshold_side(join)
            ))
        }
        _ => None,
    }
}

/// The note, when the labels were joined as hap.py run with `--pass-only`, that the TP/FP query
/// records at non-PASS calls only label no call.
fn left_out_note(join: &LabelJoin) -> Option<String> {
    (join.pass_only && join.n_non_pass > 0).then(|| {
        format!(
            "{} TP/FP query records at non-PASS calls only are left out of the labels, as they \
             were joined as hap.py run with --pass-only",
            join.n_non_pass
        )
    })
}

/// Logs how the hap.py labels joined to the calls, and fails when most of them matched no call.
///
/// # Errors
///
/// Returns an error if more than [`MAX_UNMATCHED_LABEL_FRACTION`] of the labels matched no call.
pub fn check_label_join(join: &LabelJoin, source: &str) -> Result<()> {
    info!(
        "{source}: {} TP/FP query records, {} labelling PASS calls ({} at a padded SNP's \
         differing base, {} at a padded indel's trimmed position, {} split from a multi-allelic \
         call)",
        join.n_labels, join.n_pass, join.n_padded_snp, join.n_padded_indel, join.n_multiallelic
    );
    info!("{source}: {}", pass_only_mode(join));
    if let Some(warning) = pass_only_warning(join) {
        warn!("{source}: {warning}");
    }
    if let Some(note) = left_out_note(join) {
        info!("{source}: {note}");
    }
    if !join.pass_only && join.n_non_pass > 0 {
        warn!(
            "{source} scores {} non-PASS calls: hap.py must be run with --pass-only, or a truth \
             variant matched only by a non-PASS call is not counted as an FN (and cannot be a \
             promote label)",
            join.n_non_pass
        );
    }
    if join.n_labels > 0
        && join.n_unmatched as f64 / join.n_labels as f64 > MAX_UNMATCHED_LABEL_FRACTION
    {
        bail!(
            "{} of {} TP/FP query records in {source} match no call: was hap.py run on this VCF?",
            join.n_unmatched,
            join.n_labels
        );
    }
    if join.n_unmatched > 0 {
        warn!(
            "{} TP/FP query records in {source} match no call, by position and variant type or as \
             an allele of a multi-allelic call (hap.py may have represented them differently); \
             they are left out of the labels",
            join.n_unmatched
        );
    }
    Ok(())
}

/// The rows of a labelled feature table that hap.py scored for one variant type: labelled PASS
/// calls, one per `(chrom, pos)` (the first).
///
/// # Errors
///
/// Returns an error if the table lacks the label or key columns.
pub fn labelled_mask(table: &FeatureTable, variant_type: VariantType) -> Result<Vec<bool>> {
    let (chroms, positions) = (table.str_column("chrom")?, table.i64_column("pos")?);
    let (types, filters) = (table.str_column("variant_type")?, table.str_column("filter")?);
    let labels = table.f64_column("label")?;
    let mut seen: HashSet<(&str, i64)> = HashSet::new();
    Ok((0..table.num_rows())
        .map(|i| {
            types[i] == variant_type.as_str()
                && filters[i] == PASS
                && !labels[i].is_nan()
                && seen.insert((chroms[i], positions[i]))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::testing::call_table;

    /// Counts the labels of each class, for logging.
    #[must_use]
    fn label_counts(labels: &[f64]) -> HashMap<i64, usize> {
        let mut counts = HashMap::new();
        for &label in labels.iter().filter(|l| !l.is_nan()) {
            *counts.entry(label as i64).or_default() += 1;
        }
        counts
    }

    fn gt(text: &str) -> Genotype {
        if text == "./." { Genotype::MISSING } else { text.parse().unwrap() }
    }

    /// A hap.py record: `truth` and `query` are `GT:BD:BK:BVT` (BK is ignored for TRUTH).
    fn record(chrom: &str, pos: u64, alleles: &str, truth: &str, query: &str) -> HappyRecord {
        let t: Vec<&str> = truth.split(':').collect();
        let q: Vec<&str> = query.split(':').collect();
        HappyRecord {
            chrom: chrom.to_string(),
            pos,
            truth_gt: gt(t[0]),
            truth_bd: t[1].to_string(),
            truth_bvt: t[3].to_string(),
            query_gt: gt(q[0]),
            query_bd: q[1].to_string(),
            query_bk: q[2].to_string(),
            query_bvt: q[3].to_string(),
            alleles: alleles.split(',').map(str::to_string).collect(),
        }
    }

    fn key(pos: u64, variant_type: VariantType) -> LabelKey {
        ("chr1".to_string(), pos, variant_type)
    }

    #[test]
    fn labels_query_calls_and_counts_truth_fns() {
        let records = [
            record("chr1", 100, "A,C", "0/1:TP:gm:SNP", "0/1:TP:gm:SNP"),
            record("chr1", 200, "AT,A", "0/1:FN:.:INDEL", "1/1:FP:lm:INDEL"),
            record("chr1", 300, "A,C", "1/1:FN:.:SNP", "0/1:FP:am:SNP"),
            record("chr1", 400, "A,C,G", "1/2:FN:.:SNP", "0/1:FP:am:SNP"),
            // Two records at one key: the first label, and flip_ok if any flip is correct.
            record("chr1", 500, "A,C,G", "1/2:FN:.:SNP", "0/1:FP:am:SNP"),
            record("chr1", 500, "A,C", "1/1:FN:.:SNP", "0/1:FP:am:SNP"),
            // A correct flip after a drop label is not a flip.
            record("chr1", 600, "A,C", "./.:.:.:NOCALL", "0/1:FP:lm:SNP"),
            record("chr1", 600, "A,C", "1/1:FN:.:SNP", "0/1:FP:am:SNP"),
            // A TP then an FP at one key: not a keep, as one record is wrong.
            record("chr1", 650, "AT,A", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 650, "AT,ATT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            // An FP then a TP: the FP's label.
            record("chr1", 660, "AT,A", "./.:.:.:NOCALL", "0/1:FP:.:INDEL"),
            record("chr1", 660, "AT,ATT", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 700, "ATT,A", "1/1:FN:.:INDEL", "./.:.:.:NOCALL"),
            record("chr1", 800, "A,C", "./.:.:.:NOCALL", "0/1:UNK:.:SNP"),
            record("chr3", 100, "A,C", "0/1:TP:gm:SNP", "0/1:TP:gm:SNP"),
        ];
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
        let expected: Vec<(LabelKey, (Label, bool))> = vec![
            (key(100, VariantType::Snp), (Label::Keep, false)),
            (key(200, VariantType::Indel), (Label::Drop, false)),
            (key(300, VariantType::Snp), (Label::Flip, true)),
            (key(400, VariantType::Snp), (Label::Flip, false)),
            (key(500, VariantType::Snp), (Label::Flip, true)),
            (key(600, VariantType::Snp), (Label::Drop, false)),
            (key(650, VariantType::Indel), (Label::Drop, false)),
            (key(660, VariantType::Indel), (Label::Drop, false)),
        ];
        let actual: Vec<(LabelKey, (Label, bool))> =
            labels.labels.iter().map(|(k, l)| (k.clone(), (l.label, l.flip_ok))).collect();
        assert_eq!(actual, expected);
        // The length changes of the query alleles, over the records at a key.
        assert_eq!(labels.labels[&key(200, VariantType::Indel)].alt_deltas, [-1]);
        assert_eq!(labels.labels[&key(650, VariantType::Indel)].alt_deltas, [-1, 1]);
        assert_eq!(labels.labels[&key(100, VariantType::Snp)].alt_deltas, [0]);
        assert_eq!(labels.fn_counts[&VariantType::Snp], 5);
        assert_eq!(labels.fn_counts[&VariantType::Indel], 2);
        assert!(labels.fn_positions.contains(&key(700, VariantType::Indel)));
        assert!(!labels.fn_positions.contains(&key(100, VariantType::Snp)));
        assert_eq!(labels.truth_deltas[&key(200, VariantType::Indel)], Some((-1, 0)));
        assert_eq!(labels.truth_deltas[&key(700, VariantType::Indel)], Some((-2, -2)));
        // The first truth record at a key gives its deltas.
        assert_eq!(labels.truth_deltas[&key(500, VariantType::Snp)], Some((0, 0)));
        assert!(!labels.truth_deltas.contains_key(&key(800, VariantType::Snp)));
        // A query record that hap.py did not score TP or FP, with its query alleles' changes.
        assert_eq!(labels.unscored.len(), 1);
        assert_eq!(labels.unscored["chr1"].len(), 1);
        assert_eq!(labels.unscored_deltas("chr1", 800, VariantType::Snp), Some(&[0][..]));
    }

    #[test]
    fn records_on_no_extracted_contig_are_an_error() {
        let records = [record("1", 100, "A,C", "0/1:TP:gm:SNP", "0/1:TP:gm:SNP")];
        let message = labels_from_happy_records(records.into_iter().map(Ok), &["chr2", "chr1"])
            .unwrap_err()
            .to_string();
        assert_eq!(message, "no hap.py records on the extracted contigs: chr1, chr2");
    }

    const HAPPY_HEADER: &str = "##fileformat=VCFv4.2
##contig=<ID=chr1,length=100000>
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=BD,Number=1,Type=String,Description=\"Decision\">
##FORMAT=<ID=BK,Number=1,Type=String,Description=\"Match kind\">
##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"Variant type\">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTRUTH\tQUERY
";

    fn write_happy(dir: &TempDir, header: &str, records: &[&str]) -> PathBuf {
        let path = dir.path().join("happy.vcf");
        let mut text = header.to_string();
        for line in records {
            text.push_str(&line.replace(' ', "\t"));
            text.push('\n');
        }
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn reads_happy_output() {
        let dir = TempDir::new().unwrap();
        let path = write_happy(
            &dir,
            HAPPY_HEADER,
            &[
                "chr1 100 . A C . . . GT:BD:BK:BVT 0/1:TP:gm:SNP 0|1:TP:gm:SNP",
                "chr1 200 . AT A . . . GT:BD:BK:BVT 1/1:FN:.:INDEL .:.:.:NOCALL",
                "chr1 300 . A C . . . GT:BD:BVT 0/1:FN:SNP 0/1:FP:SNP",
                // An allele index past the ALTs is missing, as pysam reads it.
                "chr1 400 . A C . . . GT:BD:BK:BVT 0/2:FN:.:SNP 0/1:FP:am:SNP",
            ],
        );
        let records: Vec<HappyRecord> =
            read_happy_records(&path).unwrap().collect::<Result<_>>().unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!((records[3].truth_gt, records[3].query_gt), (Genotype::MISSING, Genotype::HET));
        assert_eq!(records[0], record("chr1", 100, "A,C", "0/1:TP:.:SNP", "0/1:TP:gm:SNP"));
        assert_eq!(records[1], record("chr1", 200, "AT,A", "1/1:FN:.:INDEL", "./.:.:.:NOCALL"));
        assert_eq!(records[2].query_bk, ".");
        let labels = read_happy_labels(&path, &["chr1"]).unwrap();
        assert_eq!(labels.labels.len(), 3);
    }

    #[test]
    fn indexed_and_streamed_happy_output_give_the_same_labels() {
        let header = HAPPY_HEADER
            .replace("##contig=<ID=chr1,length=100000>\n", "##contig=<ID=chr1,length=100000>\n##contig=<ID=chr2,length=100000>\n##contig=<ID=chr3,length=100000>\n");
        let records = [
            "chr1 100 . A C . . . GT:BD:BK:BVT 0/1:TP:gm:SNP 0/1:TP:gm:SNP",
            "chr1 150 . A C . . . GT:BD:BK:BVT .:.:.:NOCALL 0/1:UNK:.:SNP",
            // Two records at one (chrom, pos, type): the first label wins, flip_ok is OR'd.
            "chr1 200 . A G . . . GT:BD:BK:BVT 1/1:FN:am:SNP 0/1:FP:am:SNP",
            "chr1 200 . A T . . . GT:BD:BK:BVT 0/1:FN:am:SNP 0/1:FP:lm:SNP",
            "chr2 300 . AT A . . . GT:BD:BK:BVT 1/1:FN:.:INDEL .:.:.:NOCALL",
            "chr2 400 . C A . . . GT:BD:BK:BVT 0/1:TP:gm:SNP 0/1:TP:gm:SNP",
            "chr2 450 . CA C . . . GT:BD:BK:BVT .:.:.:NOCALL 1/1:UNK:.:INDEL",
            // Not an extracted contig.
            "chr3 500 . C A . . . GT:BD:BK:BVT 0/1:FN:.:SNP .:.:.:NOCALL",
        ];
        let lines: Vec<String> = records.iter().map(|r| r.replace(' ', "\t")).collect();
        let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
        let dir = TempDir::new().unwrap();
        let indexed =
            crate::testing::write_indexed_vcf(dir.path(), "happy.vcf.gz", &header, &lines);
        let streamed = write_happy(&dir, &header, &records);
        let contigs = ["chr1", "chr2", "chr1"];
        let from_index = read_happy_labels(&indexed, &contigs).unwrap();
        assert_eq!(from_index, read_happy_labels(&streamed, &contigs).unwrap());
        assert_eq!(from_index.labels.len(), 3);
        assert_eq!(
            from_index.labels[&("chr1".to_string(), 200, VariantType::Snp)],
            QueryLabel { label: Label::Flip, flip_ok: true, alt_deltas: vec![0] }
        );
        assert_eq!(from_index.fn_counts[&VariantType::Snp], 2);
        assert_eq!(from_index.fn_counts[&VariantType::Indel], 1);
        // The unscored records of each contig are merged.
        assert_eq!(from_index.unscored.len(), 2);
        assert_eq!(from_index.unscored_deltas("chr1", 150, VariantType::Snp), Some(&[0][..]));
        assert_eq!(from_index.unscored_deltas("chr2", 450, VariantType::Indel), Some(&[-1][..]));
        let message = read_happy_labels(&indexed, &["chr9"]).err().unwrap();
        assert!(
            format!("{message:#}").contains("no hap.py records on the extracted contigs: chr9")
        );
    }

    #[rstest::rstest]
    #[case::samples("TRUTH\tQUERY", "TRUTH\tCALLS", "lacks TRUTH and QUERY samples")]
    #[case::formats("##FORMAT=<ID=BK,", "##FORMAT=<ID=XX,", "lacks the FORMAT fields BK")]
    fn rejects_vcfs_that_are_not_happy_output(
        #[case] from: &str,
        #[case] to: &str,
        #[case] expected: &str,
    ) {
        let dir = TempDir::new().unwrap();
        let path = write_happy(&dir, &HAPPY_HEADER.replace(from, to), &[]);
        let message = read_happy_records(&path).err().unwrap().to_string();
        assert!(message.contains(expected), "{message}");
    }

    #[test]
    fn attaches_labels_to_pass_calls_and_counts_the_join() {
        let table = call_table(&[
            ("chr1", 100, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 200, "AT", "A", "LowQual", "indel", "0/1", false),
            // A padded SNP differing at its third base, unlabelled at its POS.
            ("chr1", 300, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false),
            // As above, but another SNP call is at the differing base.
            ("chr1", 400, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false),
            ("chr1", 402, "A", "C", "LowQual", "snp", "0/1", false),
        ]);
        let mut labels = HappyLabels::default();
        labels.labels.insert(key(100, VariantType::Snp), QueryLabel::of(Label::Keep, false));
        labels.labels.insert(key(200, VariantType::Indel), QueryLabel::of(Label::Drop, false));
        labels.labels.insert(key(302, VariantType::Snp), QueryLabel::of(Label::Flip, true));
        labels.labels.insert(key(402, VariantType::Snp), QueryLabel::of(Label::Keep, false));
        labels.labels.insert(key(900, VariantType::Snp), QueryLabel::of(Label::Keep, false));
        labels.truth_deltas.insert(key(200, VariantType::Indel), Some((-1, 0)));
        labels.truth_deltas.insert(key(100, VariantType::Snp), None);

        let (table, join) = attach_labels(table, &labels, HappyPassOnly::Auto).unwrap();
        let label = table.f64_column("label").unwrap();
        let shown: Vec<String> = label.iter().map(|l| format!("{l}")).collect();
        assert_eq!(shown, ["1", "NaN", "2", "NaN", "NaN"]);
        assert_eq!(table.bool_column("flip_ok").unwrap(), [false, false, true, false, false]);
        let lo = table.f64_column(TRUTH_DELTA_COLUMNS[0]).unwrap();
        let hi = table.f64_column(TRUTH_DELTA_COLUMNS[1]).unwrap();
        assert!(lo[0].is_nan() && hi[0].is_nan());
        assert_eq!((lo[1], hi[1]), (-1.0, 0.0));
        assert_eq!(
            join,
            LabelJoin {
                pass_only: false,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 5,
                n_pass: 2,
                n_padded_snp: 1,
                n_padded_indel: 0,
                n_multiallelic: 0,
                n_non_pass: 2,
                n_unmatched: 1,
                n_non_pass_calls: 2,
                n_non_pass_unexplained: 1,
            }
        );
        check_label_join(&join, "happy.vcf").unwrap();
    }

    #[test]
    fn records_split_from_a_multiallelic_call_label_it() {
        let table = call_table(&[
            // hap.py gives `CTA>C` at 100 and `T>TAT` at 103: both TP, so a keep.
            ("chr1", 100, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            // A SNP record and an indel record at the call's POS: one FP makes it a drop.
            ("chr1", 200, "A", "T,AT", "PASS", "indel", "1/2", false),
            // Unlabelled at its POS (hap.py drops `*` and trims the insertion to 301).
            ("chr1", 300, "GTAAA", "*,GTAAATAAA", "PASS", "indel", "1/2", false),
            // A flip at its POS stays a flip (with its flip_ok) after a split-off drop.
            ("chr1", 400, "AACAC", "A,AAC", "PASS", "indel", "1/2", false),
            // A record at 500 whose length change is not one of the call's: unmatched.
            ("chr1", 498, "ATTT", "A,AT", "PASS", "indel", "1/2", false),
            // A biallelic call is not split by hap.py: a record at 602 is unmatched.
            ("chr1", 600, "ATTT", "A", "PASS", "indel", "0/1", false),
            // A non-PASS multi-allelic call is not labelled.
            ("chr1", 700, "CTATA", "CTA,CTATATA", "LowQual", "indel", "1/2", false),
            // Two multi-allelic calls cover 804 with a -1 allele: the one with the greater POS.
            ("chr1", 800, "CAAAAA", "C,CAAAA", "PASS", "indel", "1/2", false),
            ("chr1", 803, "AAA", "AA,AAAA", "PASS", "indel", "1/2", false),
            // A split record's label goes to the PASS call, not a non-PASS call at its key.
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
            // Nor does a non-PASS multi-allelic call with a greater POS.
            ("chr1", 1000, "CAAAAA", "C,CAAAA", "PASS", "indel", "1/2", false),
            ("chr1", 1003, "AAA", "AA,AAAA", "LowQual", "indel", "1/2", false),
        ]);
        let records = [
            record("chr1", 100, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 103, "T,TAT", "./.:.:.:NOCALL", "0/1:TP:gm:INDEL"),
            record("chr1", 200, "A,T,AT", "1/2:TP:gm:INDEL", "0/1:FP:lm:SNP"),
            record("chr1", 200, "A,AT", "./.:.:.:NOCALL", "0/1:TP:gm:INDEL"),
            record("chr1", 301, "T,TAAAT", "./.:.:.:NOCALL", "0/1:TP:gm:INDEL"),
            record("chr1", 400, "AAC,A", "1/1:FN:.:INDEL", "0/1:FP:am:INDEL"),
            record("chr1", 402, "CAC,C", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            record("chr1", 500, "T,TTTT", "./.:.:.:NOCALL", "0/1:TP:gm:INDEL"),
            record("chr1", 602, "TT,T", "./.:.:.:NOCALL", "0/1:TP:gm:INDEL"),
            record("chr1", 703, "T,TAT", "./.:.:.:NOCALL", "0/1:TP:gm:INDEL"),
            record("chr1", 804, "AA,A", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            record("chr1", 1004, "AA,A", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
        ];
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();

        let (table, join) = attach_labels(table, &labels, HappyPassOnly::Auto).unwrap();
        let label = table.f64_column("label").unwrap();
        let shown: Vec<String> = label.iter().map(|l| format!("{l}")).collect();
        assert_eq!(
            shown,
            ["1", "0", "1", "2", "NaN", "NaN", "NaN", "NaN", "0", "0", "NaN", "0", "NaN"]
        );
        let flip_ok = table.bool_column("flip_ok").unwrap();
        let mut expected = [false; 13];
        expected[3] = true;
        assert_eq!(flip_ok, expected);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 14,
                n_pass: 11,
                n_padded_snp: 0,
                n_padded_indel: 0,
                n_multiallelic: 7,
                n_non_pass: 1,
                n_unmatched: 2,
                // The 703 record can only have been split from the LowQual call at 700.
                n_non_pass_calls: 3,
                n_non_pass_unexplained: 1,
            }
        );
    }

    /// Without --pass-only, hap.py labels non-PASS calls: a label at a non-PASS call's key is
    /// that call's, even where a PASS multi-allelic call could have been split into it.
    #[test]
    fn without_pass_only_a_non_pass_call_keeps_its_label() {
        let rows = [
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
            ("chr1", 2000, "G", "GA", "LowQual", "indel", "0/1", false),
        ];
        let attach = |n: usize, happy_pass_only: HappyPassOnly| {
            let records = [
                record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
                record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
                record("chr1", 2000, "G,GA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            ];
            let records = records.into_iter().take(n).map(Ok);
            let labels = labels_from_happy_records(records, &["chr1"]).unwrap();
            let (table, join) = attach_labels(call_table(&rows), &labels, happy_pass_only).unwrap();
            let shown: Vec<String> =
                table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
            (shown, join.pass_only, join.n_multiallelic, join.n_non_pass)
        };
        // With --pass-only: the 903 label is the PASS call's.
        let with = (vec!["0".into(), "NaN".into(), "NaN".into()], true, 1, 0);
        assert_eq!(attach(2, HappyPassOnly::Yes), with);
        assert_eq!(attach(2, HappyPassOnly::Auto), with);
        // Without: the 903 label is the LowQual call's.
        let without = (vec!["1".into(), "NaN".into(), "NaN".into()], false, 0, 2);
        assert_eq!(attach(3, HappyPassOnly::No), without);
    }

    /// A label at a non-PASS SNP call's key that a padded PASS SNP was trimmed to does not show
    /// that hap.py labelled non-PASS calls: with --pass-only, it is the padded SNP's.
    #[test]
    fn a_padded_pass_snp_label_does_not_show_non_pass_calls_were_labelled() {
        let rows = [
            ("chr1", 400, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false),
            ("chr1", 402, "A", "C", "LowQual", "snp", "0/1", false),
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
            ("chr1", 2000, "G", "GA", "LowQual", "indel", "0/1", false),
        ];
        let attach = |n: usize, happy_pass_only: HappyPassOnly| {
            let records = [
                record("chr1", 402, "A,C", "0/1:TP:gm:SNP", "0/1:TP:gm:SNP"),
                record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
                record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
                record("chr1", 2000, "G,GA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            ];
            let records = records.into_iter().take(n).map(Ok);
            let labels = labels_from_happy_records(records, &["chr1"]).unwrap();
            let (table, join) = attach_labels(call_table(&rows), &labels, happy_pass_only).unwrap();
            let shown: Vec<String> =
                table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
            (shown, join.pass_only, join.n_padded_snp, join.n_multiallelic, join.n_non_pass)
        };
        let shown = |labels: [&str; 5]| labels.map(String::from).to_vec();
        // With --pass-only: the 402 label is the padded SNP's, and the 903 label the PASS
        // multi-allelic call's. Neither shows that hap.py labelled non-PASS calls.
        let with = (shown(["1", "NaN", "0", "NaN", "NaN"]), true, 1, 1, 0);
        assert_eq!(attach(3, HappyPassOnly::Yes), with);
        assert_eq!(attach(3, HappyPassOnly::Auto), with);
        // Without: the 402 and 903 labels are the LowQual calls'.
        let without = (shown(["NaN", "NaN", "1", "NaN", "NaN"]), false, 0, 0, 3);
        assert_eq!(attach(4, HappyPassOnly::No), without);
    }

    /// A label at a non-PASS indel call's key that hap.py trimmed a PASS biallelic indel to does
    /// not show that hap.py labelled non-PASS calls: with --pass-only, it is the trimmed call's.
    #[test]
    fn a_trimmed_pass_indel_label_does_not_show_non_pass_calls_were_labelled() {
        let rows = [
            // hap.py trims `ACGT>ACT` to `CG>C` at 101.
            ("chr1", 100, "ACGT", "ACT", "PASS", "indel", "0/1", false),
            ("chr1", 101, "CG", "C", "LowQual", "indel", "0/1", false),
            // As above, with no call at 301.
            ("chr1", 300, "ACGT", "ACT", "PASS", "indel", "1/1", false),
            // A label at its trimmed position with another length change is not its.
            ("chr1", 500, "ACGT", "ACT", "PASS", "indel", "0/1", false),
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
            ("chr1", 2000, "G", "GA", "LowQual", "indel", "0/1", false),
        ];
        let attach = |n: usize, happy_pass_only: HappyPassOnly| {
            let records = [
                record("chr1", 101, "CG,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
                record("chr1", 301, "CG,C", "0/1:FN:am:INDEL", "1/1:FP:am:INDEL"),
                record("chr1", 501, "C,CA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
                record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
                record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
                record("chr1", 2000, "G,GA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            ];
            let records = records.into_iter().take(n).map(Ok);
            let labels = labels_from_happy_records(records, &["chr1"]).unwrap();
            let (table, join) = attach_labels(call_table(&rows), &labels, happy_pass_only).unwrap();
            let shown: Vec<String> =
                table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
            let flip_ok = table.bool_column("flip_ok").unwrap().to_vec();
            (
                shown,
                flip_ok,
                join.pass_only,
                join.n_padded_indel,
                join.n_multiallelic,
                join.n_non_pass,
                join.n_unmatched,
            )
        };
        let shown = |labels: [&str; 7]| labels.map(String::from).to_vec();
        let flip_ok = |at_300: bool| vec![false, false, at_300, false, false, false, false];
        // With --pass-only: the 101 and 301 labels are the trimmed calls' (301 keeps its
        // flip_ok), and the 903 label the PASS multi-allelic call's. None shows that hap.py
        // labelled non-PASS calls.
        let with =
            (shown(["1", "NaN", "2", "NaN", "0", "NaN", "NaN"]), flip_ok(true), true, 2, 1, 0, 1);
        assert_eq!(attach(5, HappyPassOnly::Yes), with);
        assert_eq!(attach(5, HappyPassOnly::Auto), with);
        // Without: the 101 and 903 labels are the LowQual calls'. The 301 label is still the
        // trimmed call's, as no call is at 301.
        assert_eq!(
            attach(6, HappyPassOnly::No),
            (
                shown(["NaN", "NaN", "2", "NaN", "1", "NaN", "NaN"]),
                flip_ok(true),
                false,
                1,
                0,
                3,
                1
            )
        );
    }

    /// Without --pass-only, a label that hap.py split from a non-PASS multi-allelic call is that
    /// call's, not that of a PASS multi-allelic call with a smaller POS that also covers it.
    #[test]
    fn without_pass_only_a_label_split_from_a_non_pass_call_is_its_own() {
        let rows = [
            ("chr1", 1000, "CAAAAA", "C,CAAAA", "PASS", "indel", "1/2", false),
            ("chr1", 1003, "AAA", "AA,AAAA", "LowQual", "indel", "1/2", false),
            ("chr1", 2000, "G", "GA", "LowQual", "indel", "0/1", false),
        ];
        let attach = |n: usize, happy_pass_only: HappyPassOnly| {
            let records = [
                record("chr1", 1000, "CAAAAA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
                record("chr1", 1004, "AA,A", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
                record("chr1", 2000, "G,GA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            ];
            let records = records.into_iter().take(n).map(Ok);
            let labels = labels_from_happy_records(records, &["chr1"]).unwrap();
            let (table, join) = attach_labels(call_table(&rows), &labels, happy_pass_only).unwrap();
            let shown: Vec<String> =
                table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
            (shown, join.pass_only, join.n_multiallelic, join.n_non_pass)
        };
        // With --pass-only, only the PASS call can have been split into the 1004 label.
        let with = (vec!["0".into(), "NaN".into(), "NaN".into()], true, 1, 0);
        assert_eq!(attach(2, HappyPassOnly::Yes), with);
        assert_eq!(attach(2, HappyPassOnly::Auto), with);
        // Without, the LowQual call with the greater POS is its source, as for a PASS call.
        let without = (vec!["1".into(), "NaN".into(), "NaN".into()], false, 0, 2);
        assert_eq!(attach(3, HappyPassOnly::No), without);
    }

    /// `--happy-pass-only yes` joins the labels as hap.py run with `--pass-only`, whatever
    /// labels are at non-PASS calls' keys: here every non-PASS call has one that no PASS call
    /// explains.
    #[test]
    fn happy_pass_only_yes_joins_as_run_with_pass_only() {
        let rows = [
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
            ("chr1", 2000, "G", "GA", "LowQual", "indel", "0/1", false),
            ("chr1", 3000, "G", "GA", "LowQual", "indel", "0/1", false),
        ];
        let records = [
            record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            record("chr1", 2000, "G,GA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            record("chr1", 3000, "G,GA", "./.:.:.:NOCALL", "0/1:TP:gm:INDEL"),
        ];
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
        let (table, join) = attach_labels(call_table(&rows), &labels, HappyPassOnly::Yes).unwrap();
        let shown: Vec<String> =
            table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
        // The 903 label is the PASS multi-allelic call's.
        assert_eq!(shown, ["0", "NaN", "NaN", "NaN"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Yes,
                n_labels: 4,
                n_pass: 2,
                n_multiallelic: 1,
                n_non_pass: 2,
                n_non_pass_calls: 3,
                n_non_pass_unexplained: 2,
                ..LabelJoin::default()
            }
        );
        assert!(
            pass_only_mode(&join).contains("with --pass-only, as set by --happy-pass-only yes")
        );
        // 2 of 3 is enough to show a run without --pass-only.
        assert_eq!(
            pass_only_warning(&join).unwrap(),
            "2 of 3 non-PASS calls that hap.py could have scored (with a non-reference allele, on \
             a contig with hap.py query records) have a query record that no PASS call explains \
             (at least the 50% that shows a run without --pass-only): the labels contradict \
             --happy-pass-only yes"
        );
        assert_eq!(
            left_out_note(&join).unwrap(),
            "2 TP/FP query records at non-PASS calls only are left out of the labels, as they \
             were joined as hap.py run with --pass-only"
        );
    }

    /// `--happy-pass-only no` joins the labels as hap.py run without `--pass-only`, though no
    /// label is at a non-PASS call's key that a PASS call cannot explain.
    #[test]
    fn happy_pass_only_no_joins_as_run_without_pass_only() {
        let rows = [
            ("chr1", 400, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false),
            ("chr1", 402, "A", "C", "LowQual", "snp", "0/1", false),
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
        ];
        let records = [
            record("chr1", 402, "A,C", "0/1:TP:gm:SNP", "0/1:TP:gm:SNP"),
            record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
        ];
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
        let (table, join) = attach_labels(call_table(&rows), &labels, HappyPassOnly::No).unwrap();
        let shown: Vec<String> =
            table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
        // The 402 and 903 labels are the LowQual calls'.
        assert_eq!(shown, ["NaN", "NaN", "1", "NaN"]);
        let no = LabelJoin {
            pass_only: false,
            happy_pass_only: HappyPassOnly::No,
            n_labels: 3,
            n_pass: 1,
            n_non_pass: 2,
            n_non_pass_calls: 2,
            ..LabelJoin::default()
        };
        assert_eq!(join, no);
        assert!(
            pass_only_mode(&join).contains("without --pass-only, as set by --happy-pass-only no")
        );
        // Which the labels contradict.
        assert!(
            pass_only_warning(&join)
                .unwrap()
                .starts_with("0 of 2 non-PASS calls that hap.py could have scored")
        );
        assert_eq!(left_out_note(&join), None);
        // Without the option, the labels show nothing of non-PASS calls: joined as with it.
        let (_, join) = attach_labels(call_table(&rows), &labels, HappyPassOnly::Auto).unwrap();
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Auto,
                n_pass: 3,
                n_padded_snp: 1,
                n_multiallelic: 1,
                n_non_pass: 0,
                ..no
            }
        );
        assert!(pass_only_mode(&join).contains("with --pass-only, inferred from the labels"));
    }

    /// The calls of the `auto` tests: a padded PASS SNP trimmed to a LowQual SNP call's position,
    /// a PASS multi-allelic call split into a LowQual call's position, and `n` more LowQual calls.
    fn auto_rows(n: usize) -> Vec<crate::testing::CallRow<'static>> {
        let mut rows = vec![
            ("chr1", 400, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false),
            ("chr1", 402, "A", "C", "LowQual", "snp", "0/1", false),
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
        ];
        rows.extend(
            (0..n).map(|i| {
                ("chr1", 2000 + 10 * i as i64, "G", "GA", "LowQual", "indel", "0/1", false)
            }),
        );
        rows
    }

    /// The records of the PASS calls of [`auto_rows`], and `bd` records at the first `n` of its
    /// other LowQual calls.
    fn auto_records(n: usize, bd: &str) -> Vec<HappyRecord> {
        let mut records = vec![
            record("chr1", 402, "A,C", "0/1:TP:gm:SNP", "0/1:TP:gm:SNP"),
            record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
        ];
        let query = format!("0/1:{bd}:.:INDEL");
        records.extend(
            (0..n).map(|i| record("chr1", 2000 + 10 * i as u64, "G,GA", "./.:.:.:NOCALL", &query)),
        );
        records
    }

    fn attach_auto(
        rows: &[crate::testing::CallRow<'_>],
        records: Vec<HappyRecord>,
    ) -> (Vec<String>, LabelJoin) {
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
        let (table, join) = attach_labels(call_table(rows), &labels, HappyPassOnly::Auto).unwrap();
        let shown = table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
        (shown, join)
    }

    /// One label at a non-PASS call that no PASS call explains (e.g. a PASS call that hap.py
    /// represented differently), among many non-PASS calls, does not show that hap.py was run
    /// without --pass-only: the padded SNP and the multi-allelic call keep their labels, with a
    /// warning.
    #[test]
    fn auto_one_stray_label_among_many_non_pass_calls_joins_as_run_with_pass_only() {
        let (shown, join) = attach_auto(&auto_rows(20), auto_records(1, "FP"));
        assert_eq!(&shown[..4], ["1", "NaN", "0", "NaN"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 4,
                n_pass: 3,
                n_padded_snp: 1,
                n_multiallelic: 1,
                n_non_pass: 1,
                n_non_pass_calls: 22,
                n_non_pass_unexplained: 1,
                ..LabelJoin::default()
            }
        );
        assert_eq!(
            pass_only_mode(&join),
            "joined as hap.py run with --pass-only, inferred from the labels: 1 of 22 non-PASS \
             calls that hap.py could have scored (with a non-reference allele, on a contig with \
             hap.py query records) have a query record that no PASS call explains (under the 50% \
             that shows a run without --pass-only)"
        );
        assert_eq!(
            pass_only_warning(&join).unwrap(),
            "1 of 22 non-PASS calls that hap.py could have scored (with a non-reference allele, on \
             a contig with hap.py query records) have a query record that no PASS call explains \
             (each most likely a PASS call's record that hap.py placed at a non-PASS call): under \
             the 50% that shows a run without --pass-only, so the labels were joined as hap.py \
             run with it. If hap.py was run without --pass-only, pass --happy-pass-only no; if \
             with it, pass --happy-pass-only yes"
        );
        assert_eq!(
            left_out_note(&join).unwrap(),
            "1 TP/FP query records at non-PASS calls only are left out of the labels, as they \
             were joined as hap.py run with --pass-only"
        );
    }

    /// Records (TP, FP or UNK) at most non-PASS calls, that no PASS call explains, show that
    /// hap.py was run without --pass-only: labels at non-PASS calls' positions are theirs.
    #[rstest::rstest]
    #[case::fp("FP")]
    #[case::unk("UNK")]
    fn auto_records_at_most_non_pass_calls_join_as_run_without_pass_only(#[case] bd: &str) {
        let (shown, join) = attach_auto(&auto_rows(4), auto_records(3, bd));
        assert_eq!(&shown[..4], ["NaN", "NaN", "1", "NaN"]);
        // The 402 and 903 labels are explained by PASS calls: 3 of 6 is half.
        let without = LabelJoin {
            pass_only: false,
            happy_pass_only: HappyPassOnly::Auto,
            n_labels: 3,
            n_pass: 1,
            n_non_pass: 2,
            n_non_pass_calls: 6,
            n_non_pass_unexplained: 3,
            ..LabelJoin::default()
        };
        let fp = usize::from(bd == "FP");
        assert_eq!(join, LabelJoin { n_labels: 3 + 3 * fp, n_non_pass: 2 + 3 * fp, ..without });
        assert_eq!(pass_only_warning(&join), None);
        assert_eq!(
            pass_only_mode(&join),
            "joined as hap.py run without --pass-only, inferred from the labels: 3 of 6 non-PASS \
             calls that hap.py could have scored (with a non-reference allele, on a contig with \
             hap.py query records) have a query record that no PASS call explains (at least the \
             50% that shows a run without --pass-only)"
        );
        // One record fewer is under half.
        let (_, join) = attach_auto(&auto_rows(4), auto_records(2, bd));
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 3 + 2 * fp,
                n_pass: 3,
                n_padded_snp: 1,
                n_multiallelic: 1,
                n_non_pass: 2 * fp,
                n_non_pass_calls: 6,
                n_non_pass_unexplained: 2,
                ..LabelJoin::default()
            }
        );
    }

    /// A record at no call's position that only a non-PASS multi-allelic call can have been split
    /// into shows that hap.py looked at that call: it counts once, as that call.
    #[test]
    fn auto_a_record_split_only_from_a_non_pass_call_counts_as_that_call() {
        let rows = [
            ("chr1", 700, "CTATA", "CTA,CTATATA", "LowQual", "indel", "1/2", false),
            ("chr1", 2000, "G", "GA", "LowQual", "indel", "0/1", false),
        ];
        let split = |alleles: &str| {
            vec![
                record("chr1", 700, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
                record("chr1", 703, alleles, "./.:.:.:NOCALL", "0/1:UNK:.:INDEL"),
            ]
        };
        let joined = |pass_only: bool, n_labels: usize, n_non_pass_unexplained: usize| LabelJoin {
            pass_only,
            happy_pass_only: HappyPassOnly::Auto,
            n_labels,
            n_non_pass: n_labels,
            n_non_pass_calls: 2,
            n_non_pass_unexplained,
            ..LabelJoin::default()
        };
        let (_, join) = attach_auto(&rows, split("T,TAT"));
        assert_eq!(join, joined(false, 1, 1));
        // Only the 703 record: still the call at 700.
        let (_, join) = attach_auto(&rows, split("T,TAT")[1..].to_vec());
        assert_eq!(join, joined(false, 0, 1));
        // A record whose length change is none of the call's alleles' is not the call's.
        let (_, join) = attach_auto(&rows, split("T,TA")[1..].to_vec());
        assert_eq!(join, joined(true, 0, 0));
        // A PASS multi-allelic call that covers 703 explains it.
        let mut with_pass = rows.to_vec();
        with_pass
            .insert(0, ("chr1", 699, "CCTATA", "CCTA,CCTATATA", "PASS", "indel", "1/2", false));
        let (_, join) = attach_auto(&with_pass, split("T,TAT")[1..].to_vec());
        assert_eq!(join, joined(true, 0, 0));
    }

    /// Non-PASS calls on a contig with no hap.py query record (e.g. hap.py was run on chr1 only)
    /// do not count, so extracting another contig does not change the labels.
    #[test]
    fn auto_counts_only_non_pass_calls_on_contigs_with_happy_records() {
        let expected = LabelJoin {
            pass_only: false,
            happy_pass_only: HappyPassOnly::Auto,
            n_labels: 3,
            n_pass: 1,
            n_non_pass: 2,
            n_non_pass_calls: 6,
            n_non_pass_unexplained: 3,
            ..LabelJoin::default()
        };
        let chr1 = auto_rows(4);
        let (chr1_labels, join) = attach_auto(&chr1, auto_records(3, "UNK"));
        assert_eq!(join, expected);
        let mut both = chr1.clone();
        both.push(("chr2", 50, "A", "C", "PASS", "snp", "0/1", false));
        both.extend(
            (0..10).map(|i| ("chr2", 100 + 10 * i, "G", "GA", "LowQual", "indel", "0/1", false)),
        );
        let (labels, join) = attach_auto(&both, auto_records(3, "UNK"));
        assert_eq!(join, expected);
        assert_eq!(labels[..chr1.len()], chr1_labels);
    }

    /// Non-PASS calls with no non-reference allele in their genotype (e.g. `0/0` or `./.`
    /// `RefCall`s) do not count: hap.py writes no query record for them.
    #[test]
    fn auto_counts_only_non_pass_calls_with_a_non_reference_genotype() {
        let mut rows = auto_rows(4);
        rows.extend((0..10).map(|i| {
            let gt = if i % 2 == 0 { "0/0" } else { "./." };
            ("chr1", 3000 + 10 * i, "G", "GA", "RefCall", "indel", gt, false)
        }));
        let (_, join) = attach_auto(&rows, auto_records(3, "UNK"));
        assert_eq!(
            join,
            LabelJoin {
                pass_only: false,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 3,
                n_pass: 1,
                n_non_pass: 2,
                n_non_pass_calls: 6,
                n_non_pass_unexplained: 3,
                ..LabelJoin::default()
            }
        );
    }

    /// An unscored (`UNK`) record at the position hap.py trimmed a padded PASS SNP or indel to is
    /// that call's, unless the padded call has a record at its own POS, or the record's length
    /// change is not the indel's.
    #[rstest::rstest]
    #[case::trimmed(&[], 0)]
    #[case::snp_recorded_at_its_pos(&[("chr1", 400, "TTAGA,TTCGA", "SNP")], 1)]
    #[case::indel_recorded_at_its_pos(&[("chr1", 100, "ACGT,ACT", "INDEL")], 1)]
    #[case::other_length_change(&[("chr1", 101, "C,CA", "INDEL")], 1)]
    fn an_unscored_record_at_a_padded_pass_calls_trimmed_position_is_its(
        #[case] more: &[(&str, u64, &str, &str)],
        #[case] n_unexplained: usize,
    ) {
        let rows = [
            // hap.py trims `ACGT>ACT` to `CG>C` at 101, and `TTAGA>TTCGA` to `A>C` at 402.
            ("chr1", 100, "ACGT", "ACT", "PASS", "indel", "0/1", false),
            ("chr1", 101, "CG", "C", "LowQual", "indel", "0/1", false),
            ("chr1", 400, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false),
            ("chr1", 402, "A", "C", "LowQual", "snp", "0/1", false),
        ];
        let unk = |(chrom, pos, alleles, bvt): (&str, u64, &str, &str)| {
            record(chrom, pos, alleles, "./.:.:.:NOCALL", &format!("0/1:UNK:.:{bvt}"))
        };
        let mut records =
            vec![unk(("chr1", 101, "CG,C", "INDEL")), unk(("chr1", 402, "A,C", "SNP"))];
        // A record with another length change at 101 replaces the trimmed one.
        if more.first().is_some_and(|m| m.1 == 101) {
            records.remove(0);
        }
        records.extend(more.iter().copied().map(unk));
        let (shown, join) = attach_auto(&rows, records);
        assert_eq!(shown, ["NaN"; 4]);
        assert_eq!(
            join,
            LabelJoin {
                // 1 of 2 is half.
                pass_only: n_unexplained == 0,
                happy_pass_only: HappyPassOnly::Auto,
                n_non_pass_calls: 2,
                n_non_pass_unexplained: n_unexplained,
                ..LabelJoin::default()
            }
        );
    }

    /// A record at a padded PASS indel's trimmed position with its length change does not hide
    /// another record there with another length change: that one is the non-PASS call's. Nor
    /// does the padded call take the label of the two records together, which is not its own.
    #[test]
    fn a_padded_pass_indels_record_does_not_hide_another_at_its_trimmed_position() {
        let rows = [
            // hap.py trims `ACGT>ACT` to `CG>C` at 101, where a LowQual insertion is.
            ("chr1", 100, "ACGT", "ACT", "PASS", "indel", "0/1", false),
            ("chr1", 101, "C", "CAA", "LowQual", "indel", "0/1", false),
        ];
        let records = |both: bool| {
            let mut records =
                vec![record("chr1", 101, "CG,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL")];
            if both {
                records.push(record("chr1", 101, "C,CAA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"));
            }
            labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap()
        };
        let attach = |both: bool, happy_pass_only: HappyPassOnly| {
            let (table, join) =
                attach_labels(call_table(&rows), &records(both), happy_pass_only).unwrap();
            let shown: Vec<String> =
                table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
            (shown, join)
        };
        let joined =
            |pass_only: bool, happy_pass_only: HappyPassOnly, n_unexplained: usize| LabelJoin {
                pass_only,
                happy_pass_only,
                n_labels: 1,
                n_non_pass: 1,
                n_non_pass_calls: 1,
                n_non_pass_unexplained: n_unexplained,
                ..LabelJoin::default()
            };
        // The insertion's FP record shows that hap.py scored the LowQual call.
        let (shown, join) = attach(true, HappyPassOnly::Auto);
        assert_eq!(shown, ["NaN", "NaN"]);
        assert_eq!(join, joined(false, HappyPassOnly::Auto, 1));
        // Joined as with --pass-only, the PASS deletion does not take the merged drop label.
        let (shown, join) = attach(true, HappyPassOnly::Yes);
        assert_eq!(shown, ["NaN", "NaN"]);
        assert_eq!(join, joined(true, HappyPassOnly::Yes, 1));
        // With its own record only, it takes its keep label.
        let (shown, join) = attach(false, HappyPassOnly::Yes);
        assert_eq!(shown, ["1", "NaN"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Yes,
                n_labels: 1,
                n_pass: 1,
                n_padded_indel: 1,
                n_non_pass_calls: 1,
                ..LabelJoin::default()
            }
        );
    }

    /// A record at a PASS indel call's key with a length change that no PASS indel call covering
    /// the key has is not that call's: here hap.py trimmed a LowQual padded deletion to it. Nor does the PASS
    /// call take the label of the two records together, which is not its own.
    #[test]
    fn a_pass_indel_does_not_explain_another_length_change_at_its_key() {
        let rows = [
            // hap.py trims `ACGT>ACT` to `CG>C` at 101, where a PASS insertion is.
            ("chr1", 100, "ACGT", "ACT", "LowQual", "indel", "0/1", false),
            ("chr1", 101, "C", "CAA", "PASS", "indel", "0/1", false),
        ];
        let attach = |both: bool, happy_pass_only: HappyPassOnly| {
            let mut records =
                vec![record("chr1", 101, "C,CAA", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL")];
            if both {
                records.push(record("chr1", 101, "CG,C", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"));
            }
            let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
            let (table, join) = attach_labels(call_table(&rows), &labels, happy_pass_only).unwrap();
            let shown: Vec<String> =
                table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
            (shown, join)
        };
        let joined = |pass_only: bool, happy_pass_only: HappyPassOnly| LabelJoin {
            pass_only,
            happy_pass_only,
            n_labels: 1,
            n_non_pass: 1,
            n_non_pass_calls: 1,
            n_non_pass_unexplained: 1,
            ..LabelJoin::default()
        };
        // The deletion's FP record shows that hap.py scored the LowQual call.
        assert_eq!(
            attach(true, HappyPassOnly::Auto),
            (vec!["NaN".into(), "NaN".into()], joined(false, HappyPassOnly::Auto))
        );
        // Joined as with --pass-only, the PASS insertion does not take the merged drop label.
        assert_eq!(
            attach(true, HappyPassOnly::Yes),
            (vec!["NaN".into(), "NaN".into()], joined(true, HappyPassOnly::Yes))
        );
        // With its own record only, it takes its keep label.
        assert_eq!(
            attach(false, HappyPassOnly::Yes),
            (
                vec!["NaN".into(), "1".into()],
                LabelJoin {
                    pass_only: true,
                    happy_pass_only: HappyPassOnly::Yes,
                    n_labels: 1,
                    n_pass: 1,
                    n_non_pass_calls: 1,
                    ..LabelJoin::default()
                }
            )
        );
    }

    /// A record at a PASS indel call's key may join its allele with one of an overlapping PASS
    /// call (as hap.py writes a GATK het-alt call split into two records): the call takes its
    /// label, though the record has a length change that none of its own ALT alleles has.
    #[test]
    fn a_pass_indel_takes_a_record_joining_its_allele_with_an_overlapping_pass_calls() {
        let rows = [
            ("chr1", 136, "TAAA", "T,TAAAA", "PASS", "indel", "1/2", false),
            ("chr1", 139, "A", "ATT", "PASS", "indel", "0/1", false),
        ];
        let records = vec![
            record("chr1", 136, "TAAA,T", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            // `TAAA>TAAAA` at 136 written as `A>AA` at 139, with the call at 139's `A>ATT`.
            record("chr1", 139, "A,AA,ATT", "1/2:TP:gm:INDEL", "1/2:TP:gm:INDEL"),
        ];
        let (shown, join) = attach_auto(&rows, records);
        assert_eq!(shown, ["1", "1"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 2,
                n_pass: 2,
                ..LabelJoin::default()
            }
        );
    }

    /// As `a_pass_indel_does_not_explain_another_length_change_at_its_key`, but the padded
    /// deletion is PASS and hap.py writes one record joining both calls' alleles at 101: the
    /// insertion takes its label.
    #[test]
    fn a_pass_indel_takes_a_record_joining_its_allele_with_a_pass_padded_calls() {
        let rows = [
            ("chr1", 100, "ACGT", "ACT", "PASS", "indel", "0/1", false),
            ("chr1", 101, "C", "CAA", "PASS", "indel", "0/1", false),
        ];
        // `CG>C` is the deletion trimmed to 101, and `CG>CAAG` the insertion.
        let records = vec![record("chr1", 101, "CG,CAAG,C", "./.:.:.:NOCALL", "1/2:FP:lm:INDEL")];
        let (shown, join) = attach_auto(&rows, records);
        assert_eq!(shown, ["NaN", "0"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 1,
                n_pass: 1,
                ..LabelJoin::default()
            }
        );
    }

    /// A padded PASS call with a record at its POS, even an unscored one, does not take the label
    /// at its trimmed position: hap.py writes one record per call.
    #[test]
    fn a_padded_pass_call_with_an_unscored_record_at_its_pos_takes_no_trimmed_label() {
        let rows = [("chr1", 400, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false)];
        let records = [
            record("chr1", 400, "TTAGA,TTCGA", "./.:.:.:NOCALL", "0/1:UNK:.:SNP"),
            record("chr1", 402, "A,C", "0/1:TP:gm:SNP", "0/1:TP:gm:SNP"),
        ];
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
        let (table, join) = attach_labels(call_table(&rows), &labels, HappyPassOnly::Yes).unwrap();
        assert!(table.f64_column("label").unwrap()[0].is_nan());
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Yes,
                n_labels: 1,
                n_unmatched: 1,
                ..LabelJoin::default()
            }
        );
    }

    /// A record at the position a padded PASS SNP was trimmed to is the PASS call's, though a
    /// non-PASS multi-allelic call covering it could also have been split into it.
    #[rstest::rstest]
    #[case::with_the_pass_call(true, 0)]
    #[case::without(false, 1)]
    fn a_padded_pass_snp_explains_a_record_a_non_pass_multiallelic_call_could_own(
        #[case] with_pass: bool,
        #[case] n_unexplained: usize,
    ) {
        let mut rows = vec![("chr1", 401, "TAGA", "TCGA,TGGA", "LowQual", "snp", "1/2", false)];
        if with_pass {
            rows.insert(0, ("chr1", 400, "TTAGA", "TTCGA", "PASS", "snp", "0/1", false));
        }
        let records = vec![record("chr1", 402, "A,C", "./.:.:.:NOCALL", "0/1:UNK:.:SNP")];
        let (_, join) = attach_auto(&rows, records);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: n_unexplained == 0,
                happy_pass_only: HappyPassOnly::Auto,
                n_non_pass_calls: 1,
                n_non_pass_unexplained: n_unexplained,
                ..LabelJoin::default()
            }
        );
    }

    /// A record at the position a non-PASS padded indel is trimmed to is a PASS multi-allelic
    /// call's, if one can have been split into it.
    #[rstest::rstest]
    #[case::with_the_pass_call(true, 0)]
    #[case::without(false, 1)]
    fn a_pass_multiallelic_call_explains_a_record_at_a_non_pass_padded_calls_trimmed_position(
        #[case] with_pass: bool,
        #[case] n_unexplained: usize,
    ) {
        // hap.py trims `ACGT>ACT` at 102 to `CG>C` at 103, which the PASS call at 100 covers.
        let mut rows = vec![("chr1", 102, "ACGT", "ACT", "LowQual", "indel", "0/1", false)];
        if with_pass {
            rows.insert(
                0,
                ("chr1", 100, "GTACGT", "GTACT,GTACGTGT", "PASS", "indel", "1/2", false),
            );
        }
        let records = vec![record("chr1", 103, "CG,C", "./.:.:.:NOCALL", "0/1:UNK:.:INDEL")];
        let (_, join) = attach_auto(&rows, records);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: n_unexplained == 0,
                happy_pass_only: HappyPassOnly::Auto,
                n_non_pass_calls: 1,
                n_non_pass_unexplained: n_unexplained,
                ..LabelJoin::default()
            }
        );
    }

    /// With no non-PASS calls, the labels are joined as hap.py run with --pass-only.
    #[test]
    fn auto_with_no_non_pass_calls_joins_as_run_with_pass_only() {
        let (shown, join) = attach_auto(&auto_rows(0)[..1], auto_records(0, "FP"));
        assert_eq!(shown, ["1"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 3,
                n_pass: 1,
                n_padded_snp: 1,
                n_unmatched: 2,
                ..LabelJoin::default()
            }
        );
        assert_eq!(pass_only_warning(&join), None);
    }

    /// A forced mode logs the evidence, and warns only when it contradicts
    /// `--happy-pass-only yes` (see `happy_pass_only_yes_joins_as_run_with_pass_only`).
    #[rstest::rstest]
    #[case::yes(HappyPassOnly::Yes)]
    #[case::no(HappyPassOnly::No)]
    fn a_forced_mode_warns_only_when_the_labels_contradict_it(
        #[case] happy_pass_only: HappyPassOnly,
    ) {
        let attach = |rows: usize, records: usize| {
            let records = auto_records(records, "FP").into_iter().map(Ok);
            let labels = labels_from_happy_records(records, &["chr1"]).unwrap();
            attach_labels(call_table(&auto_rows(rows)), &labels, happy_pass_only).unwrap().1
        };
        let is_yes = happy_pass_only == HappyPassOnly::Yes;
        // 1 of 22 is under the threshold: it contradicts `no`.
        let join = attach(20, 1);
        assert_eq!((join.n_non_pass_calls, join.n_non_pass_unexplained), (22, 1));
        assert!(pass_only_mode(&join).contains(&format!(
            "--happy-pass-only {happy_pass_only} (1 of 22 non-PASS calls that hap.py could"
        )));
        let evidence = |n: usize, of: usize, side: &str| {
            format!(
                "{n} of {of} non-PASS calls that hap.py could have scored (with a non-reference \
                 allele, on a contig with hap.py query records) have a query record that no PASS \
                 call explains ({side} the 50% that shows a run without --pass-only)"
            )
        };
        let no = |n: usize, of: usize| {
            format!(
                "{}: the labels contradict --happy-pass-only no (with it, a PASS call's record that \
                 hap.py placed at a non-PASS call does not label the PASS call)",
                evidence(n, of, "under")
            )
        };
        assert_eq!(pass_only_warning(&join), (!is_yes).then(|| no(1, 22)));
        // 0 of 22 contradicts it too.
        let join = attach(20, 0);
        assert_eq!(pass_only_warning(&join), (!is_yes).then(|| no(0, 22)));
        // 3 of 6 reaches the threshold: it contradicts `yes`.
        let join = attach(4, 3);
        assert_eq!((join.n_non_pass_calls, join.n_non_pass_unexplained), (6, 3));
        let yes =
            format!("{}: the labels contradict --happy-pass-only yes", evidence(3, 6, "at least"));
        assert_eq!(pass_only_warning(&join), is_yes.then_some(yes));
        // With no non-PASS calls that hap.py could have scored, nothing contradicts either.
        let records = auto_records(0, "FP").into_iter().map(Ok);
        let labels = labels_from_happy_records(records, &["chr1"]).unwrap();
        let rows = &auto_rows(0)[..1];
        let join = attach_labels(call_table(rows), &labels, happy_pass_only).unwrap().1;
        assert_eq!((join.n_non_pass_calls, pass_only_warning(&join)), (0, None));
    }

    /// A multi-allelic call is the source of records only of the ALT alleles its genotype calls:
    /// a `0/1` PASS `CTATA>CTA,CTATATA` explains no `+2` record, which is the LowQual call's.
    #[test]
    fn a_multiallelic_call_is_the_source_of_its_called_alleles_only() {
        let rows = [
            ("chr1", 900, "CTATA", "CTA,CTATATA", "PASS", "indel", "0/1", false),
            ("chr1", 903, "T", "TAT", "LowQual", "indel", "0/1", false),
        ];
        let records = vec![
            record("chr1", 900, "CTA,C", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
            record("chr1", 903, "T,TAT", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
        ];
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
        let (table, join) = attach_labels(call_table(&rows), &labels, HappyPassOnly::Yes).unwrap();
        let shown: Vec<String> =
            table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
        // The PASS call keeps its own keep label: the FP is not split from it.
        assert_eq!(shown, ["1", "NaN"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Yes,
                n_labels: 2,
                n_pass: 1,
                n_non_pass: 1,
                n_non_pass_calls: 1,
                n_non_pass_unexplained: 1,
                ..LabelJoin::default()
            }
        );
    }

    /// A call that hap.py could not have scored (`0/0` or `./.`) does not hide the source it
    /// shadows: a multi-allelic call with a greater POS, or a padded call trimmed to the same
    /// position.
    #[rstest::rstest]
    #[case::hom_ref_multiallelic("0/0", false)]
    #[case::no_call_multiallelic("./.", false)]
    #[case::no_call_padded("./.", true)]
    fn auto_an_ineligible_call_does_not_hide_an_eligible_source(
        #[case] gt: &'static str,
        #[case] padded: bool,
    ) {
        let (rows, records) = if padded {
            // Both are trimmed to `CG>C` at 101; the ineligible one comes first.
            let rows = vec![
                ("chr1", 99, "TACGT", "TACT", "RefCall", "indel", gt, false),
                ("chr1", 100, "ACGT", "ACT", "LowQual", "indel", "0/1", false),
            ];
            (rows, vec![record("chr1", 101, "CG,C", "./.:.:.:NOCALL", "0/1:UNK:.:INDEL")])
        } else {
            // Both cover 703 with a `+2` allele; the ineligible one has the greater POS.
            let rows = vec![
                ("chr1", 700, "CTATA", "CTA,CTATATA", "LowQual", "indel", "1/2", false),
                ("chr1", 701, "TATA", "TA,TATATA", "RefCall", "indel", gt, false),
            ];
            (rows, vec![record("chr1", 703, "T,TAT", "./.:.:.:NOCALL", "0/1:UNK:.:INDEL")])
        };
        let (_, join) = attach_auto(&rows, records);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: false,
                happy_pass_only: HappyPassOnly::Auto,
                n_non_pass_calls: 1,
                n_non_pass_unexplained: 1,
                ..LabelJoin::default()
            }
        );
    }

    /// A TP or FP record at the position hap.py trimmed a non-PASS padded call to is that call's
    /// in the label counts too, not one that matches no call.
    #[test]
    fn a_scored_record_at_a_non_pass_padded_calls_trimmed_position_is_its() {
        let rows = [("chr1", 100, "ACGT", "ACT", "LowQual", "indel", "0/1", false)];
        let records = vec![record("chr1", 101, "CG,C", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL")];
        let (shown, join) = attach_auto(&rows, records);
        assert_eq!(shown, ["NaN"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: false,
                happy_pass_only: HappyPassOnly::Auto,
                n_labels: 1,
                n_non_pass: 1,
                n_non_pass_calls: 1,
                n_non_pass_unexplained: 1,
                ..LabelJoin::default()
            }
        );
        check_label_join(&join, "happy.vcf").unwrap();
    }

    /// A PASS multi-allelic call whose label at its key was left out (a record there has a
    /// length change of another call's) is not labelled by a record split from it either: its
    /// own records at its key may include an FP.
    #[test]
    fn a_split_record_does_not_label_a_call_whose_label_was_left_out() {
        let rows = [
            ("chr1", 100, "CTATA", "CTA,CTATATA", "PASS", "indel", "1/2", false),
            ("chr1", 100, "C", "CA", "LowQual", "indel", "0/1", false),
        ];
        let records = [
            record("chr1", 100, "CTA,C", "0/1:FN:.:INDEL", "0/1:FP:lm:INDEL"),
            record("chr1", 100, "C,CA", "./.:.:.:NOCALL", "0/1:FP:lm:INDEL"),
            record("chr1", 103, "T,TAT", "0/1:TP:gm:INDEL", "0/1:TP:gm:INDEL"),
        ];
        let labels = labels_from_happy_records(records.into_iter().map(Ok), &["chr1"]).unwrap();
        let (table, join) = attach_labels(call_table(&rows), &labels, HappyPassOnly::Yes).unwrap();
        let shown: Vec<String> =
            table.f64_column("label").unwrap().iter().map(|l| format!("{l}")).collect();
        assert_eq!(shown, ["NaN", "NaN"]);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: true,
                happy_pass_only: HappyPassOnly::Yes,
                n_labels: 2,
                n_pass: 1,
                n_multiallelic: 1,
                n_non_pass: 1,
                n_non_pass_calls: 1,
                n_non_pass_unexplained: 1,
                ..LabelJoin::default()
            }
        );
    }

    /// A non-PASS call at a PASS call's key counts among the calls hap.py could have scored, and
    /// a record there with a length change that no PASS call covering the key has is its own.
    #[rstest::rstest]
    #[case::no_record(&[], 1, 3)]
    #[case::its_record(&[(2000, "G,GTT")], 2, 3)]
    fn auto_counts_a_non_pass_call_at_a_pass_calls_key(
        #[case] more: &[(u64, &str)],
        #[case] n_unexplained: usize,
        #[case] n_calls: usize,
    ) {
        let rows = [
            ("chr1", 2000, "G", "GA", "PASS", "indel", "0/1", false),
            ("chr1", 2000, "G", "GTT", "LowQual", "indel", "0/1", false),
            ("chr1", 3000, "G", "GA", "LowQual", "indel", "0/1", false),
            ("chr1", 4000, "G", "GA", "LowQual", "indel", "0/1", false),
        ];
        let unk = |pos: u64, alleles: &str| {
            record("chr1", pos, alleles, "./.:.:.:NOCALL", "0/1:UNK:.:INDEL")
        };
        let mut records = vec![unk(3000, "G,GA")];
        records.extend(more.iter().map(|&(pos, alleles)| unk(pos, alleles)));
        let (_, join) = attach_auto(&rows, records);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: looks_pass_only(n_unexplained, n_calls),
                happy_pass_only: HappyPassOnly::Auto,
                n_non_pass_calls: n_calls,
                n_non_pass_unexplained: n_unexplained,
                ..LabelJoin::default()
            }
        );
    }

    /// hap.py writes the record of a non-PASS padded call, as of a PASS one, at the position it
    /// trims the call to: it shows that hap.py scored the call, unless another call is there or
    /// the record has another length change.
    #[rstest::rstest]
    #[case::trimmed("CG,C", &[], 2)]
    #[case::other_length_change("C,CA", &[], 1)]
    #[case::a_call_at_the_trimmed_position("CG,C", &[("chr1", 402, "A", "C", "LowQual", "snp", "0/1", false)], 2)]
    fn auto_a_non_pass_padded_call_has_the_record_at_its_trimmed_position(
        #[case] indel_record: &str,
        #[case] more: &[crate::testing::CallRow<'static>],
        #[case] n_unexplained: usize,
    ) {
        let mut rows = vec![
            // hap.py trims `ACGT>ACT` to `CG>C` at 101, and `TTAGA>TTCGA` to `A>C` at 402.
            ("chr1", 100, "ACGT", "ACT", "LowQual", "indel", "0/1", false),
            ("chr1", 400, "TTAGA", "TTCGA", "LowQual", "snp", "0/1", false),
        ];
        rows.extend_from_slice(more);
        let records = vec![
            record("chr1", 101, indel_record, "./.:.:.:NOCALL", "0/1:UNK:.:INDEL"),
            record("chr1", 402, "A,C", "./.:.:.:NOCALL", "0/1:UNK:.:SNP"),
        ];
        let (_, join) = attach_auto(&rows, records);
        assert_eq!(
            join,
            LabelJoin {
                pass_only: false,
                happy_pass_only: HappyPassOnly::Auto,
                n_non_pass_calls: rows.len(),
                n_non_pass_unexplained: n_unexplained,
                ..LabelJoin::default()
            }
        );
    }

    #[rstest::rstest]
    #[case::het("0/1", true)]
    #[case::hom_alt("1/1", true)]
    #[case::multi_allelic("1/2", true)]
    #[case::hom_ref("0/0", false)]
    #[case::missing("./.", false)]
    fn genotypes_with_a_non_reference_allele(#[case] gt: &str, #[case] expected: bool) {
        assert_eq!(has_alt_allele(gt), expected);
    }

    #[test]
    fn most_labels_matching_no_call_is_an_error() {
        let join = LabelJoin { n_labels: 5, n_pass: 2, n_unmatched: 3, ..LabelJoin::default() };
        let message = check_label_join(&join, "happy.vcf").unwrap_err().to_string();
        assert_eq!(
            message,
            "3 of 5 TP/FP query records in happy.vcf match no call: was hap.py run on this VCF?"
        );
    }

    #[test]
    fn labelled_mask_keeps_the_first_labelled_pass_call_per_position() {
        let table = call_table(&[
            ("chr1", 100, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 100, "A", "G", "PASS", "snp", "0/1", false),
            ("chr1", 200, "A", "C", "LowQual", "snp", "0/1", false),
            ("chr1", 300, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 400, "AT", "A", "PASS", "indel", "0/1", false),
        ])
        .with_f64_column("label", vec![1.0, 0.0, 1.0, f64::NAN, 0.0])
        .unwrap();
        assert_eq!(
            labelled_mask(&table, VariantType::Snp).unwrap(),
            [true, false, false, false, false]
        );
        assert_eq!(
            labelled_mask(&table, VariantType::Indel).unwrap(),
            [false, false, false, false, true]
        );
        let counts = label_counts(&[1.0, 0.0, 1.0, f64::NAN]);
        assert_eq!(counts, HashMap::from([(0, 1), (1, 2)]));
    }

    #[test]
    fn unfixable_flip_policies_are_named_alike_everywhere() {
        use clap::ValueEnum;
        let policies = [UnfixableFlips::Flip, UnfixableFlips::Drop, UnfixableFlips::Exclude];
        assert_eq!(UnfixableFlips::value_variants(), policies);
        for policy in policies {
            // The command line (clap), the model file (serde) and messages (Display) agree.
            assert_eq!(policy.to_possible_value().unwrap().get_name(), policy.as_str());
            assert_eq!(policy.to_string(), policy.as_str());
            assert_eq!(serde_json::to_string(&policy).unwrap(), format!("\"{policy}\""));
        }
        assert_eq!(UnfixableFlips::from_str("DROP", true).unwrap(), UnfixableFlips::Drop);
        assert!(UnfixableFlips::from_str("keep", true).is_err());
        assert!(UnfixableFlips::default().is_default());
        assert!(!UnfixableFlips::Exclude.is_default());
    }

    #[test]
    fn happy_pass_only_modes_are_named_alike_everywhere() {
        use clap::ValueEnum;
        let modes = [HappyPassOnly::Auto, HappyPassOnly::Yes, HappyPassOnly::No];
        assert_eq!(HappyPassOnly::value_variants(), modes);
        for mode in modes {
            // The command line (clap) and messages (Display) agree.
            assert_eq!(mode.to_possible_value().unwrap().get_name(), mode.as_str());
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert_eq!(HappyPassOnly::from_str("No", true).unwrap(), HappyPassOnly::No);
        assert!(HappyPassOnly::from_str("maybe", true).is_err());
        assert_eq!(HappyPassOnly::default(), HappyPassOnly::Auto);
    }

    #[test]
    fn unfixable_flips_are_flip_labels_a_flip_cannot_fix() {
        let labels = [Label::Flip, Label::Flip, Label::Flip, Label::Keep, Label::Drop];
        let flip_ok = [true, false, true, false, false];
        let flippable = [true, true, false, true, true];
        assert_eq!(
            unfixable_flips(&labels, &flip_ok, &flippable),
            [false, true, true, false, false]
        );
    }
}
