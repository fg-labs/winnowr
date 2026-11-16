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

/// A call's `(chrom, pos, variant type)`, the key labels join on.
pub type LabelKey = (String, u64, VariantType);

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
}

impl HappyLabels {
    /// No labels, and zero FN counts for SNPs and indels.
    fn empty() -> HappyLabels {
        HappyLabels {
            fn_counts: [(VariantType::Snp, 0), (VariantType::Indel, 0)].into_iter().collect(),
            ..HappyLabels::default()
        }
    }

    /// Adds the labels of other contigs (keys are disjoint, as they include the contig).
    fn merge(&mut self, other: HappyLabels) {
        self.labels.extend(other.labels);
        for (variant_type, n) in other.fn_counts {
            *self.fn_counts.entry(variant_type).or_default() += n;
        }
        self.fn_positions.extend(other.fn_positions);
        self.truth_deltas.extend(other.truth_deltas);
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
        if record.query_bd != "TP" && record.query_bd != "FP" {
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
    /// Those that labelled a PASS call.
    pub n_pass: usize,
    /// Of `n_pass`, those found at the differing base of a padded SNP.
    pub n_padded_snp: usize,
    /// Of `n_pass`, those found at the trimmed position of a padded biallelic indel.
    pub n_padded_indel: usize,
    /// Of `n_pass`, those that hap.py split from a multi-allelic call.
    pub n_multiallelic: usize,
    /// Those at a non-PASS call only (hap.py was run without `--pass-only`).
    pub n_non_pass: usize,
    /// Those at no call.
    pub n_unmatched: usize,
    /// Those at a non-PASS call's key that no PASS call can explain. With
    /// [`HappyPassOnly::Auto`], any one of them means hap.py was run without `--pass-only`.
    pub n_without_pass_only: usize,
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
    /// The length changes of its ALT alleles other than `*`.
    alt_deltas: Vec<i64>,
}

/// The multi-allelic calls of a feature table, to find the call that hap.py split a record from.
#[derive(Debug, Default)]
struct MultiAllelicCalls<'a> {
    /// Per contig: its multi-allelic calls, sorted by POS (then row), and the longest REF allele.
    by_contig: HashMap<&'a str, (Vec<MultiAllelicCall>, u64)>,
}

impl<'a> MultiAllelicCalls<'a> {
    /// Indexes the calls with more than one ALT allele.
    fn new(chroms: &[&'a str], positions: &[i64], refs: &[&str], alts: &[&str]) -> Self {
        let mut index = MultiAllelicCalls::default();
        for row in (0..chroms.len()).filter(|&row| alts[row].contains(',')) {
            let reference = refs[row].as_bytes();
            let start = positions[row] as u64;
            let ref_len = reference.len().max(1) as u64;
            let alt_deltas =
                alts[row].split(',').filter_map(|alt| length_change(reference, alt)).collect();
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
    /// REF allele covers `pos`, that has ALT alleles with every length change of the record's
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
/// Labels attach to PASS calls, and truth allele length changes to every call, by
/// `(chrom, pos, variant type)`. A padded call with no label at its POS takes the label at the
/// position hap.py may have trimmed it to, unless a PASS call of its variant type is at that
/// position (or any call of its variant type, when hap.py was run without `--pass-only`; see
/// below): a padded SNP whose differing base is not its first (e.g. `TTAGA>TTCGA`) the label at
/// that base, and a padded biallelic indel (e.g. `ACGT>ACT`, trimmed to `CG>C` one base on) the
/// label at the base before its trimmed alleles, if that label has its length change.
///
/// A label that matches no PASS call (of its variant type, at its position) labels the PASS
/// multi-allelic call that hap.py split it from, if there is one: a PASS call with more than one
/// ALT allele whose REF allele covers the label's position, and whose ALT alleles have the length
/// changes of the label's query alleles (of several, the one with the greatest POS). Such a label
/// makes a call unlabelled at its own key take its label, and a `Keep` call take a label other
/// than `Keep` (a call is `Keep` only if every allele is TP). It does not set `flip_ok`, which it
/// computed for the split record's genotype rather than the call's.
///
/// hap.py run with `--pass-only` labels only PASS calls, so a label at a non-PASS call's key is
/// then a PASS call's: a padded call's, or a multi-allelic call's (whose source is chosen among
/// PASS calls only). Without `--pass-only`, a label at a non-PASS call's key is that call's, and
/// a multi-allelic source is chosen among all calls, labelling it only if it is PASS. hap.py's
/// output does not say which way it was run, so `happy_pass_only` does: [`HappyPassOnly::Yes`]
/// or [`HappyPassOnly::No`], or [`HappyPassOnly::Auto`] to infer it. Inferred, it was run
/// without `--pass-only` if a label at a non-PASS call's key can come from no PASS call (neither
/// a padded call nor a multi-allelic call). Run without `--pass-only` and every such label
/// explained by a PASS call, the labels are inferred to be joined as if run with it, and a
/// non-PASS call's label may go to a PASS call: hap.py should be run with `--pass-only`.
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
    let (refs, alts) = (table.str_column("ref")?, table.str_column("alt")?);
    let variant_types: Vec<Option<VariantType>> =
        types.par_iter().map(|t| t.parse().ok()).collect();
    let key = |i: usize, pos: u64| variant_types[i].map(|vt| KeyRef(chroms[i], pos, vt));

    let on_pass_call: HashSet<KeyRef<'_>> = (0..n)
        .filter(|&i| filters[i] == PASS)
        .filter_map(|i| key(i, positions[i] as u64))
        .collect();
    let on_call: HashSet<KeyRef<'_>> = (0..n).filter_map(|i| key(i, positions[i] as u64)).collect();
    // The position that hap.py may have trimmed a PASS padded call unlabelled at its POS to, if a
    // label there may be its: no other PASS call of its type is there (a non-PASS one may be, see
    // `pass_only`). For a padded SNP (`TTAGA>TTCGA`), its differing base; for a padded biallelic
    // indel (`ACGT>ACT`), the base before its trimmed alleles, with a label of its length change.
    let padded_source = |i: usize| -> Option<u64> {
        let variant_type = variant_types[i]?;
        if filters[i] != PASS {
            return None;
        }
        let (chrom, pos) = (chroms[i], positions[i] as u64);
        let (offset, delta) = match variant_type {
            VariantType::Snp => (snp_offset(refs[i], alts[i]), None),
            VariantType::Indel if !alts[i].contains(',') => {
                let delta = length_change(refs[i].as_bytes(), alts[i])?;
                let (first_differing, _, _) = trim_alleles(refs[i].as_bytes(), alts[i].as_bytes());
                (first_differing.saturating_sub(1), Some(delta))
            }
            VariantType::Indel => return None,
        };
        let shifted = pos + offset as u64;
        let shifted_key = KeyRef(chrom, shifted, variant_type);
        (offset > 0
            && !labels.labels.contains_key(&KeyRef(chrom, pos, variant_type))
            && !on_pass_call.contains(&shifted_key)
            && labels.labels.get(&shifted_key).is_some_and(|query_label| {
                delta.is_none_or(|delta| query_label.alt_deltas.contains(&delta))
            }))
        .then_some(shifted)
    };
    let padded_labels: HashSet<KeyRef<'_>> = (0..n)
        .filter_map(|i| Some(KeyRef(chroms[i], padded_source(i)?, variant_types[i]?)))
        .collect();
    let multiallelic = MultiAllelicCalls::new(&chroms, positions, &refs, &alts);
    let is_pass = |row: usize| filters[row] == PASS;
    // hap.py run with --pass-only labels only PASS calls, so a label at a non-PASS call's key is
    // then a PASS call's: a padded call's, or that of a PASS multi-allelic call it was split from.
    // Without --pass-only, it is the non-PASS call's own. hap.py's output does not say which, so
    // it is told from the run: a label at a non-PASS call's key that no PASS call can explain
    // only comes without --pass-only.
    let n_without_pass_only = labels
        .labels
        .iter()
        .filter(|((chrom, pos, variant_type), query_label)| {
            let label_key = KeyRef(chrom, *pos, *variant_type);
            !on_pass_call.contains(&label_key)
                && on_call.contains(&label_key)
                && !padded_labels.contains(&label_key)
                && multiallelic.owner(chrom, *pos, &query_label.alt_deltas, is_pass).is_none()
        })
        .count();
    let pass_only = match happy_pass_only {
        HappyPassOnly::Yes => true,
        HappyPassOnly::No => false,
        HappyPassOnly::Auto => n_without_pass_only == 0,
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
                (row.0, row.1) = (query_label.label.index() as f64, query_label.flip_ok);
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

    let on_pass: HashSet<KeyRef<'_>> =
        (0..n).filter(|&i| filters[i] == PASS).filter_map(|i| key(i, rows[i].3)).collect();
    let mut join = LabelJoin {
        pass_only,
        happy_pass_only,
        n_labels: labels.labels.len(),
        n_padded_snp: (0..n)
            .filter(|&i| rows[i].4 && variant_types[i] == Some(VariantType::Snp))
            .count(),
        n_padded_indel: (0..n)
            .filter(|&i| rows[i].4 && variant_types[i] == Some(VariantType::Indel))
            .count(),
        n_without_pass_only,
        ..LabelJoin::default()
    };
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
            join.n_pass += 1;
        } else if let Some(row) = pass_owner() {
            join.n_pass += 1;
            join.n_multiallelic += 1;
            let call = &mut rows[row];
            let is_keep = call.0 == Label::Keep.index() as f64;
            if call.0.is_nan() || (is_keep && query_label.label != Label::Keep) {
                (call.0, call.1) = (query_label.label.index() as f64, false);
            }
        } else if on_call.contains(&label_key) || any_owner().is_some() {
            join.n_non_pass += 1;
        } else {
            join.n_unmatched += 1;
        }
    }

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

/// How the labels were joined, as to hap.py's `--pass-only`, and why.
fn pass_only_mode(join: &LabelJoin) -> String {
    let mode = if join.pass_only { "with" } else { "without" };
    let n = join.n_without_pass_only;
    match join.happy_pass_only {
        HappyPassOnly::Yes | HappyPassOnly::No => format!(
            "joined as hap.py run {mode} --pass-only, as set by --happy-pass-only {} ({n} TP/FP \
             query records are at a non-PASS call that no PASS call explains)",
            if join.pass_only { "yes" } else { "no" }
        ),
        HappyPassOnly::Auto => format!(
            "joined as hap.py run {mode} --pass-only, inferred from the labels: {n} TP/FP query \
             records are at a non-PASS call that no PASS call explains"
        ),
    }
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
            // Two records at one (chrom, pos, type): the first label wins, flip_ok is OR'd.
            "chr1 200 . A G . . . GT:BD:BK:BVT 1/1:FN:am:SNP 0/1:FP:am:SNP",
            "chr1 200 . A T . . . GT:BD:BK:BVT 0/1:FN:am:SNP 0/1:FP:lm:SNP",
            "chr2 300 . AT A . . . GT:BD:BK:BVT 1/1:FN:.:INDEL .:.:.:NOCALL",
            "chr2 400 . C A . . . GT:BD:BK:BVT 0/1:TP:gm:SNP 0/1:TP:gm:SNP",
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
                n_without_pass_only: 1
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
                n_without_pass_only: 0
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
            ("chr1", 1000, "CAAAAA", "C,CAAAA", "PASS", "indel", "0/1", false),
            ("chr1", 1003, "AAA", "AA,AAAA", "LowQual", "indel", "0/1", false),
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
        assert!(join.pass_only);
        assert_eq!(join.happy_pass_only, HappyPassOnly::Yes);
        assert_eq!(join.n_multiallelic, 1);
        assert!(
            pass_only_mode(&join).contains("with --pass-only, as set by --happy-pass-only yes")
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
        assert!(!join.pass_only);
        assert_eq!(join.happy_pass_only, HappyPassOnly::No);
        assert_eq!((join.n_padded_snp, join.n_multiallelic, join.n_non_pass), (0, 0, 2));
        assert!(
            pass_only_mode(&join).contains("without --pass-only, as set by --happy-pass-only no")
        );
        // Without the option, the labels show nothing of non-PASS calls: joined as with it.
        let (_, join) = attach_labels(call_table(&rows), &labels, HappyPassOnly::Auto).unwrap();
        assert!(join.pass_only);
        assert!(pass_only_mode(&join).contains("with --pass-only, inferred from the labels"));
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
