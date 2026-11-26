//! The checks of a set of models before a run: that they can be scored together, and against the
//! inputs given (region names, the reference's contig lengths). `score` runs them at startup, and
//! `winnowr check-models` runs them with [`check_region_beds`], which reads the region BEDs (see
//! [`crate::check_models`]). Every check runs, and every problem is reported, with a stable
//! [`Check`] id.
//!
//! The checks read each model's information only (see [`crate::model::ModelInfo::read`]): a
//! model's booster is not checked here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use rayon::prelude::*;
use serde::{Serialize, Serializer};

use crate::alignment_provenance::{
    AlignmentProvenance, MergeConflict, ProvenanceDifference, UnknownReason, compare_with_models,
    conflicts, describe_differences, incomplete_options_reason, indeterminate_tags, merge,
};
use crate::common::named_enum;
use crate::feature_source::{FeatureSource, shared_source};
use crate::model::{ModelInfo, ReferenceContigs, check_reference};
use crate::model_evidence::{ModelEvidence, models_options};
use crate::pileup::PileupOptions;
use crate::policy::{
    NOT_DIFFICULT_REGIONS, duplicate_kinds, is_indel_genotype, model_kinds_message,
};
use crate::regions::{NamedBed, RegionSet};
use crate::strlen::StrlenOptions;

/// What a [`Problem`] is about: a stable id, written in snake case (e.g. `region_missing`). New
/// ids may be added; callers should ignore ids they do not know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Check {
    /// A model file cannot be read: its name, its text, its fields, or its information's checks.
    ModelUnreadable,
    /// A model file is of a format version this winnowr does not read.
    FormatVersion,
    /// A model file is given twice.
    DuplicateModel,
    /// More than one model of a task and variant type.
    ModelKinds,
    /// The models were trained on different contigs.
    ContigsDiffer,
    /// The models record different lengths of a contig.
    ReferenceDiffers,
    /// A model's contigs are missing from, or of other lengths in, the reference or VCF given.
    ReferenceMismatch,
    /// The models' features come from different feature sources (read settings included).
    FeatureSourceDiffers,
    /// The models need repeat-length evidence measured with different settings.
    StrlenSettingsDiffer,
    /// The models need pileup evidence measured with different settings.
    PileupSettingsDiffer,
    /// Models of the alignments source whose pileup reads alleles otherwise than their alignment
    /// settings: `score` reads the reads twice.
    PileupAlignmentPasses,
    /// A region a model uses has no region BED.
    RegionMissing,
    /// A region name is given twice.
    RegionRepeated,
    /// A region BED cannot be read: an error if a model may use it, else a warning.
    RegionUnreadable,
    /// A region BED has no interval on the models' contigs: an error if a model may use it, else
    /// a warning.
    RegionContigs,
    /// A region BED no model uses.
    RegionUnused,
    /// An indel genotype model, and no not-difficult regions.
    NotDifficultMissing,
    /// A model was trained by another winnowr version.
    OtherWinnowrVersion,
    /// A model records no reference contigs.
    NoReferenceContigs,
    /// No reference or VCF was given to check the models' reference against.
    ReferenceNotChecked,
    /// A reference index or VCF given to check the models against cannot be read.
    InputUnreadable,
    /// The models record different alignment provenance: a warning, or an error when strict.
    AlignmentProvenanceDiffers,
    /// The models' alignment provenance differs from the reads': a warning, or an error when
    /// strict.
    AlignmentProvenanceMismatch,
    /// A model records no alignment provenance.
    AlignmentProvenanceUnknown,
    /// The models record alignment provenance, and the reads' is not known: a warning, or an
    /// error when strict.
    AlignmentProvenanceNotChecked,
    /// The reads' header has aligner records that normalize differently.
    AlignmentHeaderMixed,
    /// Equal mixed markers do not establish that the models and reads were aligned alike.
    AlignmentProvenanceIndeterminate,
    /// Alignment settings could not be inferred reliably from a lossy recorded command line.
    AlignmentProvenanceIncomplete,
}

named_enum!(Check, "check", {
    ModelUnreadable => "model_unreadable",
    FormatVersion => "format_version",
    DuplicateModel => "duplicate_model",
    ModelKinds => "model_kinds",
    ContigsDiffer => "contigs_differ",
    ReferenceDiffers => "reference_differs",
    ReferenceMismatch => "reference_mismatch",
    FeatureSourceDiffers => "feature_source_differs",
    StrlenSettingsDiffer => "strlen_settings_differ",
    PileupSettingsDiffer => "pileup_settings_differ",
    PileupAlignmentPasses => "pileup_alignment_passes",
    RegionMissing => "region_missing",
    RegionRepeated => "region_repeated",
    RegionUnreadable => "region_unreadable",
    RegionContigs => "region_contigs",
    RegionUnused => "region_unused",
    NotDifficultMissing => "not_difficult_missing",
    OtherWinnowrVersion => "other_winnowr_version",
    NoReferenceContigs => "no_reference_contigs",
    ReferenceNotChecked => "reference_not_checked",
    InputUnreadable => "input_unreadable",
    AlignmentProvenanceDiffers => "alignment_provenance_differs",
    AlignmentProvenanceMismatch => "alignment_provenance_mismatch",
    AlignmentProvenanceUnknown => "alignment_provenance_unknown",
    AlignmentProvenanceNotChecked => "alignment_provenance_not_checked",
    AlignmentHeaderMixed => "alignment_header_mixed",
    AlignmentProvenanceIndeterminate => "alignment_provenance_indeterminate",
    AlignmentProvenanceIncomplete => "alignment_provenance_incomplete",
});

/// How bad a [`Problem`] is: an error stops a run, a warning does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// The models cannot be run as given.
    Error,
    /// The run may not be what was meant.
    Warning,
}

named_enum!(Severity, "severity", { Error => "error", Warning => "warning" });

/// Written as its id ([`Check::as_str`]).
impl Serialize for Check {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// Written as its name ([`Severity::as_str`]).
impl Serialize for Severity {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// One problem a check found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Problem {
    /// The check.
    pub check: Check,
    /// How bad it is.
    pub severity: Severity,
    /// The model it is about, or `None` for a problem of the set.
    #[serde(serialize_with = "lossy_path::option")]
    pub model: Option<PathBuf>,
    /// What is wrong, and what to do.
    pub message: String,
    /// The tags on which the models' and the reads' alignment provenance differ, for
    /// [`Check::AlignmentProvenanceMismatch`] (not written when empty).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub differences: Vec<ProvenanceDifference>,
}

impl Problem {
    /// An error.
    pub fn error<M: Into<String>>(check: Check, model: Option<&Path>, message: M) -> Self {
        Problem::new(check, Severity::Error, model, message)
    }

    /// A warning.
    pub fn warning<M: Into<String>>(check: Check, model: Option<&Path>, message: M) -> Self {
        Problem::new(check, Severity::Warning, model, message)
    }

    /// A problem of a severity.
    fn new<M: Into<String>>(
        check: Check,
        severity: Severity,
        model: Option<&Path>,
        message: M,
    ) -> Self {
        Problem {
            check,
            severity,
            model: model.map(Path::to_path_buf),
            message: message.into(),
            differences: Vec::new(),
        }
    }
}

/// Serializers of paths as strings, any part that is not UTF-8 replaced (serde fails on such a
/// path), for `serialize_with`.
pub mod lossy_path {
    use std::path::{Path, PathBuf};

    use serde::Serializer;

    /// A path.
    pub fn one<S: Serializer>(path: &Path, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&path.to_string_lossy())
    }

    /// A path, or `null`.
    pub fn option<S: Serializer>(path: &Option<PathBuf>, serializer: S) -> Result<S::Ok, S::Error> {
        match path {
            Some(path) => serializer.serialize_some(&path.to_string_lossy()),
            None => serializer.serialize_none(),
        }
    }

    /// A list of paths.
    pub fn many<S: Serializer>(paths: &[PathBuf], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(paths.iter().map(|path| path.to_string_lossy()))
    }
}

/// The models given to a command whose information could be read: their paths and information,
/// in the order given.
#[derive(Debug, Clone, Copy)]
pub struct ModelSet<'a> {
    /// The models' paths.
    pub paths: &'a [PathBuf],
    /// The models' information, one for each of `paths`.
    pub infos: &'a [&'a ModelInfo],
}

/// What a [`ModelSet`] is checked against; every part is optional.
#[derive(Debug, Clone, Default)]
pub struct SetInputs<'a> {
    /// The region BEDs given, by name.
    pub regions: &'a [NamedBed],
    /// The contig lengths of the reference the calls will be on, each with where they came from
    /// (e.g. `VCF calls.vcf.gz`).
    pub contigs: Vec<(String, &'a ReferenceContigs)>,
    /// True if some models given could not be read, and so are not in the set: whether a region
    /// BED is used is then not known.
    pub unreadable_models: bool,
    /// What the models' alignment provenance is checked against.
    pub reads: ReadsCheck<'a>,
}

/// How the reads to be scored were aligned, and how strictly the models' alignment provenance is
/// checked against it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadsCheck<'a> {
    /// The reads' alignment provenance, when known (from their header, their evidence tables or
    /// `--alignment-tags`).
    pub alignment: Option<&'a AlignmentProvenance>,
    /// Why it is not known, when it is not.
    pub unknown: UnknownReason,
    /// Whether an alignment provenance problem is an error (`--strict-provenance`).
    pub strict: bool,
}

/// Runs every check of a model set against its inputs, and returns every problem found, in a
/// stable order: by check, then in the order found. The check of the inputs alone (the region
/// names) runs even when the set is empty. The region BEDs are not read (see
/// [`check_region_beds`]).
#[must_use]
pub fn check_model_set(set: &ModelSet<'_>, inputs: &SetInputs<'_>) -> Vec<Problem> {
    let mut problems: Vec<Problem> = region_names(inputs).into_iter().collect();
    if !set.infos.is_empty() {
        // A file given twice is one problem: the other checks see it once.
        let repeats = repeated_files(set.paths);
        problems.extend(duplicate_models(set.paths, &repeats));
        let distinct: Vec<&ModelInfo> = set
            .infos
            .iter()
            .zip(&repeats)
            .filter(|(_, repeat)| !**repeat)
            .map(|(info, _)| *info)
            .collect();
        problems.extend(model_kinds(&distinct));
        problems.extend(same_contigs(set));
        problems.extend(reference_differs(set));
        problems.extend(reference_mismatch(set, inputs));
        problems.extend(feature_source(set));
        problems.extend(evidence_differs::<StrlenOptions>(set, Check::StrlenSettingsDiffer));
        problems.extend(evidence_differs::<PileupOptions>(set, Check::PileupSettingsDiffer));
        problems.extend(pileup_alignment_passes(set));
        problems.extend(model_regions(set, inputs));
        problems.extend(model_warnings(set));
        problems.extend(alignment_provenance_problems(set, inputs.reads));
    }
    problems.sort_by_key(|problem| problem.check);
    problems
}

/// For each of `paths`, whether it is a file given earlier (compared by canonical path, so that
/// two spellings of one file are the same).
fn repeated_files(paths: &[PathBuf]) -> Vec<bool> {
    let mut seen = BTreeSet::new();
    paths
        .iter()
        .map(|path| !seen.insert(path.canonicalize().unwrap_or_else(|_| path.clone())))
        .collect()
}

/// [`Check::DuplicateModel`]: the same file given twice (`repeats`, by [`repeated_files`]).
fn duplicate_models(paths: &[PathBuf], repeats: &[bool]) -> Vec<Problem> {
    paths
        .iter()
        .zip(repeats)
        .filter(|(_, repeat)| **repeat)
        .map(|(path, _)| {
            let message = format!("--models lists {} more than once", path.display());
            Problem::error(Check::DuplicateModel, Some(path), message)
        })
        .collect()
}

/// [`Check::ModelKinds`]: at most one model per task and variant type; one problem per task and
/// variant type with more (see [`duplicate_kinds`]).
fn model_kinds(infos: &[&ModelInfo]) -> Vec<Problem> {
    duplicate_kinds(infos.iter().copied())
        .into_iter()
        .map(|(task, variant_type)| {
            Problem::error(Check::ModelKinds, None, model_kinds_message(task, variant_type))
        })
        .collect()
}

/// [`Check::ContigsDiffer`]: the models were trained on the same contigs, in any order (a
/// model's contig order is only the order its training merged them in).
fn same_contigs(set: &ModelSet<'_>) -> Option<Problem> {
    let contig_set =
        |info: &ModelInfo| -> BTreeSet<String> { info.contigs.iter().cloned().collect() };
    let first = contig_set(set.infos.first()?);
    if set.infos.iter().all(|info| contig_set(info) == first) {
        return None;
    }
    let described: Vec<String> = set
        .paths
        .iter()
        .zip(set.infos)
        .map(|(path, info)| format!("{}: {}", path.display(), info.contigs.join(",")))
        .collect();
    let message = format!("--models were trained on different contigs: {}", described.join("; "));
    Some(Problem::error(Check::ContigsDiffer, None, message))
}

/// Two models' lengths of a contig that differ.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LengthConflict<'a> {
    /// The contig.
    contig: &'a str,
    /// The model with the other length, and its length.
    model: (&'a Path, u64),
    /// The first model to record a length, and its length.
    first: (&'a Path, u64),
}

/// The reference contig lengths the models record, merged in model order (models that record
/// none skipped): each contig's first known length, or `None` if no model knows it.
///
/// # Errors
///
/// Returns every contig a model records another length of than the first model to record one.
fn merge_reference_contigs<'a>(
    set: &ModelSet<'a>,
) -> Result<ReferenceContigs, Vec<LengthConflict<'a>>> {
    let mut merged: IndexMap<&'a str, Option<(u64, &'a Path)>> = IndexMap::new();
    let mut conflicts = Vec::new();
    for (path, info) in set.paths.iter().zip(set.infos) {
        let Some(contigs) = &info.reference_contigs else { continue };
        for (contig, &length) in contigs {
            let entry = merged.entry(contig.as_str()).or_insert(None);
            match (*entry, length) {
                (Some((first, first_path)), Some(length)) if first != length => {
                    conflicts.push(LengthConflict {
                        contig,
                        model: (path, length),
                        first: (first_path, first),
                    });
                }
                (None, Some(length)) => *entry = Some((length, path)),
                _ => {}
            }
        }
    }
    if !conflicts.is_empty() {
        return Err(conflicts);
    }
    Ok(merged
        .into_iter()
        .map(|(contig, known)| (contig.to_string(), known.map(|(length, _)| length)))
        .collect())
}

/// The models' reference contig lengths, merged in model order (see
/// [`merge_reference_contigs`]): `None` if any model records none, or two models record different
/// lengths of a contig.
#[must_use]
pub fn merged_reference_contigs(set: &ModelSet<'_>) -> Option<ReferenceContigs> {
    if set.infos.iter().any(|info| info.reference_contigs.is_none()) {
        return None;
    }
    merge_reference_contigs(set).ok()
}

/// [`Check::ReferenceDiffers`]: the models record the same length of each contig they share.
fn reference_differs(set: &ModelSet<'_>) -> Vec<Problem> {
    let conflicts = merge_reference_contigs(set).err().unwrap_or_default();
    conflicts
        .into_iter()
        .map(|LengthConflict { contig, model: (path, length), first: (first_path, first) }| {
            Problem::error(
                Check::ReferenceDiffers,
                Some(path),
                format!(
                    "--models were trained on different references: {} records {contig} as \
                     {length} bp, {} as {first} bp",
                    path.display(),
                    first_path.display()
                ),
            )
        })
        .collect()
}

/// [`Check::ReferenceMismatch`]: each model's contigs are in each given set of contig lengths,
/// with the same lengths.
fn reference_mismatch(set: &ModelSet<'_>, inputs: &SetInputs<'_>) -> Vec<Problem> {
    let mut problems = Vec::new();
    for (source, contigs) in &inputs.contigs {
        for (path, info) in set.paths.iter().zip(set.infos) {
            if let Err(error) = check_reference(path, info, contigs, source, true) {
                problems.push(Problem::error(
                    Check::ReferenceMismatch,
                    Some(path),
                    error.to_string(),
                ));
            }
        }
    }
    problems
}

/// The feature source of the models, if they share one (`None` with no models).
#[must_use]
pub fn models_feature_source(infos: &[&ModelInfo]) -> Option<FeatureSource> {
    shared_source(infos, |info| info.feature_source).ok().flatten()
}

/// [`Check::FeatureSourceDiffers`]: the models share one feature source.
fn feature_source(set: &ModelSet<'_>) -> Option<Problem> {
    if models_feature_source(set.infos).is_some() {
        return None;
    }
    let described: Vec<String> = set
        .paths
        .iter()
        .zip(set.infos)
        .map(|(path, info)| format!("{}: {}", path.display(), info.feature_source))
        .collect();
    let message = format!(
        "--models were trained on features from different sources ({}): score with models \
         trained on tables of one --feature-source (and, with --alignments, the same \
         --alignment-settings)",
        described.join("; ")
    );
    Some(Problem::error(Check::FeatureSourceDiffers, None, message))
}

/// The settings the models need evidence `E` measured with: `None` if no model uses it.
///
/// # Errors
///
/// Returns an error if two models need it measured differently (see [`models_options`]).
pub fn evidence_settings<E: ModelEvidence>(set: &ModelSet<'_>) -> anyhow::Result<Option<E>> {
    models_options::<E>(set.infos, set.paths)
}

/// [`Check::StrlenSettingsDiffer`] and [`Check::PileupSettingsDiffer`]: the models that use the
/// evidence need it measured one way.
fn evidence_differs<E: ModelEvidence>(set: &ModelSet<'_>, check: Check) -> Option<Problem> {
    evidence_settings::<E>(set).err().map(|error| Problem::error(check, None, error.to_string()))
}

/// [`Check::PileupAlignmentPasses`]: the pileup of a model of the alignments source reads alleles
/// as its alignment settings do, so that one pass over the reads measures both.
fn pileup_alignment_passes(set: &ModelSet<'_>) -> Vec<Problem> {
    set.paths
        .iter()
        .zip(set.infos)
        .filter_map(|(path, info)| {
            let FeatureSource::VcfAlignments(options) = info.feature_source else { return None };
            let pileup = info.pileup_settings?;
            (!pileup.reads_alleles_as(&options)).then(|| {
                Problem::warning(
                    Check::PileupAlignmentPasses,
                    Some(path),
                    format!(
                        "model {} counts its pileup with settings {pileup}, which read alleles with \
                         another base quality or flank than its alignment settings {options}: \
                         score reads the reads twice",
                        path.display()
                    ),
                )
            })
        })
        .collect()
}

/// True if `score` may read the region BED `name` for the set: a model's features use it, or it
/// is the not-difficult regions and an indel genotype model will score (which the default policy
/// decides with them), or `unreadable_models` (some models could not be read, so which BEDs they
/// use is not known).
#[must_use]
fn region_needed(set: &ModelSet<'_>, name: &str, unreadable_models: bool) -> bool {
    unreadable_models
        || set.infos.iter().any(|info| info.uses_region(name))
        || (name == NOT_DIFFICULT_REGIONS && set.infos.iter().any(|info| is_indel_genotype(info)))
}

/// [`Check::RegionRepeated`]: each region name is given once.
fn region_names(inputs: &SetInputs<'_>) -> Option<Problem> {
    let repeated = crate::common::repeated(inputs.regions.iter().map(|r| r.name.as_str()));
    (!repeated.is_empty()).then(|| {
        let message = format!("region names must be unique; repeated: {}", repeated.join(", "));
        Problem::error(Check::RegionRepeated, None, message)
    })
}

/// Reads the region BEDs of `inputs` (in parallel) and checks them: [`Check::RegionUnreadable`]
/// and (with a model) [`Check::RegionContigs`], in the order of the BEDs. Each is an error for a
/// BED `score` may read (see [`region_needed`]), and a warning, saying why, for one it does not.
#[must_use]
pub fn check_region_beds(set: &ModelSet<'_>, inputs: &SetInputs<'_>) -> Vec<Problem> {
    // The models' contigs are the same (or Check::ContigsDiffer says otherwise).
    let contigs = set.infos.first().map(|info| info.contigs.as_slice());
    // Each BED is checked as it is read, so that only its problem, not its intervals, is kept.
    inputs
        .regions
        .par_iter()
        .filter_map(|bed| {
            let message = match RegionSet::from_bed(&bed.name, &bed.path) {
                Err(error) => Some((
                    Check::RegionUnreadable,
                    format!("region BED {}={}: {error:#}", bed.name, bed.path.display()),
                )),
                Ok(regions) => contigs
                    .and_then(|contigs| regions.check_contigs(contigs).err())
                    .map(|error| (Check::RegionContigs, error.to_string())),
            };
            let (check, message) = message?;
            Some(if region_needed(set, &bed.name, inputs.unreadable_models) {
                Problem::new(check, Severity::Error, None, message)
            } else {
                let why = if bed.name == NOT_DIFFICULT_REGIONS {
                    "no indel genotype model uses it, so score reads it only with --abstain-actions"
                } else {
                    "no model uses it, so score does not read it"
                };
                Problem::new(check, Severity::Warning, None, format!("{message}; {why}"))
            })
        })
        .collect()
}

/// The region checks of the models: [`Check::RegionMissing`] and [`Check::RegionUnused`] (not
/// when a model could not be read, as what it uses is not known).
fn model_regions(set: &ModelSet<'_>, inputs: &SetInputs<'_>) -> Vec<Problem> {
    let mut problems = Vec::new();
    let names: Vec<&str> = inputs.regions.iter().map(|r| r.name.as_str()).collect();
    for (path, info) in set.paths.iter().zip(set.infos) {
        let mut missing: Vec<&str> = info
            .region_names
            .iter()
            .map(String::as_str)
            .filter(|name| !names.contains(name))
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            let message = format!(
                "model {} ({} {}) needs region BEDs (--regions NAME=PATH) for: {}",
                path.display(),
                info.task,
                info.variant_type,
                missing.join(", ")
            );
            problems.push(Problem::error(Check::RegionMissing, Some(path), message));
        }
    }
    // The not-difficult regions are not called unused: score reads them with --abstain-actions.
    let mut unused: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| {
            *name != NOT_DIFFICULT_REGIONS && !region_needed(set, name, inputs.unreadable_models)
        })
        .collect();
    if !unused.is_empty() {
        unused.sort_unstable();
        let message =
            format!("no model uses the --regions {}; they are ignored", unused.join(", "));
        problems.push(Problem::warning(Check::RegionUnused, None, message));
    }
    problems
}

/// [`Check::OtherWinnowrVersion`] and [`Check::NoReferenceContigs`]: the models' own warnings
/// (see [`ModelInfo::warnings`]).
fn model_warnings(set: &ModelSet<'_>) -> Vec<Problem> {
    let mut problems = Vec::new();
    for (path, info) in set.paths.iter().zip(set.infos) {
        let warnings = info.warnings(path);
        if let Some(message) = warnings.other_version {
            problems.push(Problem::warning(Check::OtherWinnowrVersion, Some(path), message));
        }
        if let Some(message) = warnings.no_reference_contigs {
            problems.push(Problem::warning(Check::NoReferenceContigs, Some(path), message));
        }
    }
    problems
}

/// The alignment provenance the models share: the [`merge`] of every model's that records one, if
/// they agree, else `None`.
#[must_use]
pub fn models_alignment(infos: &[&ModelInfo]) -> Option<AlignmentProvenance> {
    let known: Vec<&AlignmentProvenance> =
        infos.iter().filter_map(|info| info.alignment.as_ref()).collect();
    merge(&known).ok().flatten()
}

/// The alignment provenance checks: [`Check::AlignmentProvenanceDiffers`],
/// [`Check::AlignmentProvenanceMismatch`], [`Check::AlignmentProvenanceNotChecked`],
/// [`Check::AlignmentProvenanceIndeterminate`] and [`Check::AlignmentProvenanceIncomplete`], each
/// an error when strict, else a warning. Models that record no provenance are not checked. Models
/// that agree are compared with the reads as one (their [`merge`]), retaining complete contributors'
/// known absent settings even with incomplete peers; otherwise each is (see [`compare_with_models`]).
/// Scoring repeats these checks after required evidence is loaded.
pub fn alignment_provenance_problems(set: &ModelSet<'_>, check: ReadsCheck<'_>) -> Vec<Problem> {
    let (paths, known): (Vec<&Path>, Vec<&AlignmentProvenance>) = set
        .paths
        .iter()
        .zip(set.infos)
        .filter_map(|(path, info)| Some((path.as_path(), info.alignment.as_ref()?)))
        .unzip();
    if known.is_empty() {
        return Vec::new();
    }
    let severity = if check.strict { Severity::Error } else { Severity::Warning };
    let advice = if check.strict {
        "Pass --alignment-tags if the reads' recorded alignment is wrong."
    } else {
        "Pass --strict-provenance to make this an error, or --alignment-tags if the reads' \
         recorded alignment is wrong."
    };
    let conflicts = conflicts(&known);
    let mut problems: Vec<Problem> = conflicts
        .iter()
        .map(|MergeConflict { first, second, differences }| {
            let (first, second) = (paths[*first], paths[*second]);
            let message = format!(
                "--models were trained on reads aligned differently: {}. Score with models \
                 trained on reads aligned alike",
                describe_differences(
                    differences,
                    &first.display().to_string(),
                    &second.display().to_string()
                )
            );
            Problem::new(Check::AlignmentProvenanceDiffers, severity, Some(second), message)
        })
        .collect();
    for (path, model) in paths.iter().zip(&known) {
        if let Some(reason) = incomplete_options_reason(model) {
            problems.push(Problem::new(
                Check::AlignmentProvenanceIncomplete,
                severity,
                Some(path),
                format!(
                    "model {} records incomplete alignment settings: {reason}. Regenerate its \
                     training feature tables with extract --alignment-tags options-inference=explicit \
                     and the intended settings, then retrain from those tables",
                    path.display()
                ),
            ));
        }
    }
    let Some(reads) = check.alignment else {
        if check.unknown == UnknownReason::Unreadable {
            return problems;
        }
        let recorded = match merge(&known) {
            Ok(Some(merged)) => merged.to_string(),
            _ => known[0].to_string(),
        };
        let message = format!(
            "the models record how their training reads were aligned ({recorded}), but the \
             reads' alignment is not known, so it is not checked: {}",
            check.unknown.explanation()
        );
        problems.push(Problem::new(Check::AlignmentProvenanceNotChecked, severity, None, message));
        return problems;
    };
    if let Some(reason) = incomplete_options_reason(reads) {
        problems.push(Problem::new(
            Check::AlignmentProvenanceIncomplete,
            severity,
            None,
            format!(
                "the reads record incomplete alignment settings: {reason}. Supply the intended \
                 settings as --alignment-tags with an options-inference=explicit declaration"
            ),
        ));
    }
    let compared = merge(&known).ok().flatten();
    let indeterminate: BTreeSet<String> = if let Some(merged) = &compared {
        indeterminate_tags(merged, reads)
    } else {
        known.iter().flat_map(|model| indeterminate_tags(model, reads)).collect()
    };
    if !indeterminate.is_empty() {
        problems.push(Problem::new(
            Check::AlignmentProvenanceIndeterminate,
            severity,
            None,
            format!(
                "alignment provenance is indeterminate on {}: both the models and reads record \
                 mixed, which does not establish agreement. {advice}",
                indeterminate.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
    }
    for (index, differences) in compare_with_models(&known, reads) {
        let path = index.map(|index| paths[index]);
        let whose = path.map_or_else(
            || "the models'".to_string(),
            |path| format!("model {}'s", path.display()),
        );
        let message = format!(
            "the reads were not aligned as {whose} training reads were: {}. Calls may be scored \
             worse (see the user guide: Alignment provenance). {advice}",
            describe_differences(&differences, "models", "reads")
        );
        let mut problem = Problem::new(Check::AlignmentProvenanceMismatch, severity, path, message);
        problem.differences = differences;
        problems.push(problem);
    }
    problems
}

/// The message of the errors among `problems` for a command to fail with: the one error's message
/// alone, or `N problems with the models and their inputs:` and each, one per line (`None`
/// without errors).
#[must_use]
pub fn error_message(problems: &[Problem]) -> Option<String> {
    let errors: Vec<&Problem> =
        problems.iter().filter(|problem| problem.severity == Severity::Error).collect();
    match errors.as_slice() {
        [] => None,
        [error] => Some(error.message.clone()),
        errors => {
            let lines: Vec<String> = errors
                .iter()
                .map(|error| format!("  {}: {}", error.check, error.message))
                .collect();
            Some(format!(
                "{} problems with the models and their inputs:\n{}",
                errors.len(),
                lines.join("\n")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tempfile::TempDir;

    use super::*;
    use crate::common::{Task, VariantType};
    use crate::read_annotations::ReadAnnotationOptions;
    use crate::testing::provenance;

    /// A genotype model of `variant_type` on chr1 (of 1000 bp).
    fn info(variant_type: VariantType) -> ModelInfo {
        let mut info = crate::testing::separable_model(Task::Genotype, variant_type).info;
        info.reference_contigs = Some([("chr1".to_string(), Some(1000))].into_iter().collect());
        info
    }

    /// The paths of `n` models: `a.winnowr.json`, `b.winnowr.json`, ...
    fn paths(n: usize) -> Vec<PathBuf> {
        (0..n)
            .map(|i| PathBuf::from(format!("{}.winnowr.json", char::from(b'a' + i as u8))))
            .collect()
    }

    /// The problems of `infos` (as [`paths`]) against `inputs`.
    fn check(infos: &[ModelInfo], inputs: &SetInputs<'_>) -> Vec<Problem> {
        let paths = paths(infos.len());
        let refs: Vec<&ModelInfo> = infos.iter().collect();
        check_model_set(&ModelSet { paths: &paths, infos: &refs }, inputs)
    }

    /// The problems of `infos` against `inputs`, with those of reading the region BEDs, by check.
    fn check_with_beds(infos: &[ModelInfo], inputs: &SetInputs<'_>) -> Vec<Problem> {
        let paths = paths(infos.len());
        let refs: Vec<&ModelInfo> = infos.iter().collect();
        let set = ModelSet { paths: &paths, infos: &refs };
        let mut problems = check_model_set(&set, inputs);
        problems.extend(check_region_beds(&set, inputs));
        problems.sort_by_key(|problem| problem.check);
        problems
    }

    /// The checks of the problems, in order.
    fn checks(problems: &[Problem]) -> Vec<Check> {
        problems.iter().map(|problem| problem.check).collect()
    }

    /// The severities of the problems, in order.
    fn severities(problems: &[Problem]) -> Vec<Severity> {
        problems.iter().map(|problem| problem.severity).collect()
    }

    /// Contig lengths of chr1.
    fn chr1(length: u64) -> ReferenceContigs {
        [("chr1".to_string(), Some(length))].into_iter().collect()
    }

    /// `lengths`, as of the reference index `ref.fa.fai`.
    fn of_index(lengths: &ReferenceContigs) -> Vec<(String, &ReferenceContigs)> {
        vec![("reference index ref.fa.fai".to_string(), lengths)]
    }

    /// Each id is written in JSON as `as_str` writes it, and [`Check::ALL`] lists every check
    /// (`as_str` and `ALL` come from one list, so a check missing from it does not compile).
    #[test]
    fn ids_are_written_alike_in_text_and_json() {
        for &check in Check::ALL {
            let json = serde_json::to_value(check).unwrap();
            assert_eq!(json, serde_json::Value::String(check.as_str().into()));
            assert_eq!(check.as_str().parse::<Check>().unwrap(), check);
        }
        for &severity in Severity::ALL {
            let json = serde_json::to_value(severity).unwrap();
            assert_eq!(json, serde_json::Value::String(severity.as_str().into()));
        }
        assert_eq!(Check::ALL.len(), Check::AlignmentProvenanceIncomplete as usize + 1);
        assert_eq!(Severity::ALL, [Severity::Error, Severity::Warning]);
    }

    #[test]
    fn a_good_set_has_no_problems() {
        let lengths = chr1(1000);
        let inputs = SetInputs { contigs: of_index(&lengths), ..SetInputs::default() };
        let problems = check(&[info(VariantType::Snp), info(VariantType::Indel)], &inputs);
        assert!(problems.is_empty(), "{problems:?}");
    }

    /// Every task and variant type with more than one model is a problem, once.
    #[test]
    fn two_models_of_a_kind_are_rejected() {
        let snp = info(VariantType::Snp);
        let problems = check(&[snp.clone(), snp.clone()], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::ModelKinds]);
        assert_eq!(problems[0].severity, Severity::Error);
        assert_eq!(
            problems[0].message,
            "expected at most one model per task and variant type, got two genotype snp models"
        );
        let indel = info(VariantType::Indel);
        let set = [indel.clone(), snp.clone(), snp.clone(), indel, snp];
        let problems = check(&set, &SetInputs::default());
        assert_eq!(checks(&problems), [Check::ModelKinds, Check::ModelKinds]);
        assert!(problems[0].message.ends_with("got two genotype indel models"), "{problems:?}");
        assert!(problems[1].message.ends_with("got two genotype snp models"), "{problems:?}");
    }

    /// One file given twice, under two spellings, is one problem: the other checks see it once.
    #[test]
    fn the_same_file_twice_is_a_duplicate() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let file = dir.path().join("a.winnowr.json");
        std::fs::write(&file, "{}").unwrap();
        let other_spelling = dir.path().join("sub").join("..").join("a.winnowr.json");
        assert_ne!(file, other_spelling);
        let paths = [file, other_spelling.clone()];
        let snp = info(VariantType::Snp);
        let problems = check_model_set(
            &ModelSet { paths: &paths, infos: &[&snp, &snp] },
            &SetInputs::default(),
        );
        assert_eq!(checks(&problems), [Check::DuplicateModel]);
        assert_eq!(problems[0].model.as_deref(), Some(other_spelling.as_path()));
    }

    #[test]
    fn models_of_other_contigs_or_references_are_rejected() {
        let snp = info(VariantType::Snp);
        let mut indel = info(VariantType::Indel);
        indel.contigs = vec!["chr2".to_string()];
        indel.reference_contigs = Some([("chr1".to_string(), Some(999))].into_iter().collect());
        let problems = check(&[snp, indel], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::ContigsDiffer, Check::ReferenceDiffers]);
        assert_eq!(
            problems[0].message,
            "--models were trained on different contigs: a.winnowr.json: chr1; b.winnowr.json: chr2"
        );
        assert_eq!(
            problems[1].message,
            "--models were trained on different references: b.winnowr.json records chr1 as 999 \
             bp, a.winnowr.json as 1000 bp"
        );
    }

    /// The models' contigs may be in any order, but must be the same.
    #[test]
    fn models_of_the_same_contigs_in_any_order_score_together() {
        let contigs = |names: &[&str]| names.iter().map(ToString::to_string).collect();
        let mut snp = info(VariantType::Snp);
        let mut indel = info(VariantType::Indel);
        snp.contigs = contigs(&["chr1", "chr2"]);
        indel.contigs = contigs(&["chr2", "chr1"]);
        assert!(check(&[snp.clone(), indel.clone()], &SetInputs::default()).is_empty());
        indel.contigs = contigs(&["chr1"]);
        let problems = check(&[snp, indel], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::ContigsDiffer]);
        assert_eq!(
            problems[0].message,
            "--models were trained on different contigs: a.winnowr.json: chr1,chr2; \
             b.winnowr.json: chr1"
        );
    }

    #[test]
    fn the_reference_is_checked_when_given() {
        let snp = info(VariantType::Snp);
        let lengths = chr1(5);
        let problems = check(
            std::slice::from_ref(&snp),
            &SetInputs { contigs: of_index(&lengths), ..SetInputs::default() },
        );
        assert_eq!(checks(&problems), [Check::ReferenceMismatch]);
        assert_eq!(
            problems[0].message,
            "reference index ref.fa.fai is not on the reference model a.winnowr.json was trained \
             on: other lengths: chr1 (5 bp, model 1000 bp)"
        );
        assert!(check(&[snp], &SetInputs::default()).is_empty());
    }

    /// Feature sources that differ only in their read settings differ.
    #[test]
    fn models_of_different_feature_sources_are_rejected() {
        let snp = info(VariantType::Snp);
        let mut indel = info(VariantType::Indel);
        let dragen: ReadAnnotationOptions = "mapq1".parse().unwrap();
        let mut snp_reads = snp.clone();
        snp_reads.feature_source = FeatureSource::VcfAlignments(ReadAnnotationOptions::default());
        indel.feature_source = FeatureSource::VcfAlignments(dragen);
        let problems = check(&[snp_reads, indel], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::FeatureSourceDiffers]);
        assert!(
            problems[0].message.starts_with(
                "--models were trained on features from different sources (a.winnowr.json: \
                 alignments:mapq20,bq10,flank5; b.winnowr.json: alignments:mapq1,bq10,flank5)"
            ),
            "{}",
            problems[0].message
        );
        assert_eq!(models_feature_source(&[&snp]), Some(FeatureSource::Gvcf));
        assert_eq!(models_feature_source(&[]), None);
    }

    /// The pileup settings must agree.
    #[test]
    fn pileup_measured_with_different_settings_is_rejected() {
        let mut a = info(VariantType::Snp);
        let mut b = info(VariantType::Indel);
        a.pileup_settings = Some(PileupOptions::default());
        b.pileup_settings = Some("bq20".parse().unwrap());
        let problems = check(&[a, b], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::PileupSettingsDiffer]);
        assert!(problems[0].message.starts_with("--models need pileup evidence"), "{problems:?}");
    }

    /// The strlen settings must agree; a model with repeat-length features that records no strlen
    /// settings needs the legacy ones.
    #[test]
    fn strlen_measured_with_different_settings_is_rejected() {
        let strlen = |settings: Option<StrlenOptions>| {
            let mut info = info(VariantType::Indel);
            info.stutter = Some(crate::testing::fit_stutter_model(&[]));
            info.strlen_settings = settings;
            info
        };
        let mut legacy = strlen(None);
        legacy.task = Task::Promote;
        let problems =
            check(&[legacy.clone(), strlen(Some(StrlenOptions::LEGACY))], &SetInputs::default());
        assert!(problems.is_empty(), "{problems:?}");
        let problems =
            check(&[legacy, strlen(Some(StrlenOptions::default()))], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::StrlenSettingsDiffer]);
        assert!(problems[0].message.starts_with("--models need strlen evidence"), "{problems:?}");
    }

    /// A model of the alignments source whose pileup reads alleles otherwise needs two passes.
    #[test]
    fn a_pileup_read_otherwise_than_the_alignments_needs_two_passes() {
        let mut snp = info(VariantType::Snp);
        snp.feature_source = FeatureSource::VcfAlignments(ReadAnnotationOptions::default());
        snp.pileup_settings = Some(PileupOptions::default());
        assert!(check(&[snp.clone()], &SetInputs::default()).is_empty());
        snp.pileup_settings = Some("bq20".parse().unwrap());
        let problems = check(&[snp], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::PileupAlignmentPasses]);
        assert_eq!(problems[0].severity, Severity::Warning);
    }

    fn bed(dir: &TempDir, name: &str, text: &str) -> NamedBed {
        let path = dir.path().join(format!("{name}.bed"));
        std::fs::write(&path, text).unwrap();
        NamedBed { name: name.to_string(), path }
    }

    fn absent(dir: &TempDir, name: &str) -> NamedBed {
        NamedBed { name: name.to_string(), path: dir.path().join(format!("absent-{name}.bed")) }
    }

    /// A BED a model uses (the not-difficult regions with an indel genotype model) must be
    /// readable and on the models' contigs; one score does not read only warns.
    #[test]
    fn regions_are_checked() {
        let dir = TempDir::new().unwrap();
        let mut snp = info(VariantType::Snp);
        snp.region_names = vec!["segdups".to_string(), "lowmap".to_string()];
        let indel = info(VariantType::Indel);
        let regions = [
            bed(&dir, "segdups", "chr9\t0\t10\n"),
            bed(&dir, "other", "chr1\t0\t10\n"),
            bed(&dir, "other", "chr1\t0\t10\n"),
            absent(&dir, "broken"),
            absent(&dir, NOT_DIFFICULT_REGIONS),
        ];
        let inputs = SetInputs { regions: &regions, ..SetInputs::default() };
        // score does not read the BEDs.
        let problems = check(&[snp.clone(), indel.clone()], &inputs);
        assert_eq!(
            checks(&problems),
            [Check::RegionMissing, Check::RegionRepeated, Check::RegionUnused]
        );
        let problems = check_with_beds(&[snp.clone(), indel], &inputs);
        assert_eq!(
            checks(&problems),
            [
                Check::RegionMissing,
                Check::RegionRepeated,
                Check::RegionUnreadable,
                Check::RegionUnreadable,
                Check::RegionContigs,
                Check::RegionUnused,
            ]
        );
        assert_eq!(
            severities(&problems),
            [
                Severity::Error,
                Severity::Error,
                Severity::Warning,
                Severity::Error,
                Severity::Error,
                Severity::Warning,
            ]
        );
        assert_eq!(
            problems[0].message,
            "model a.winnowr.json (genotype snp) needs region BEDs (--regions NAME=PATH) for: lowmap"
        );
        assert_eq!(problems[1].message, "region names must be unique; repeated: other");
        assert!(problems[2].message.starts_with("region BED broken="), "{}", problems[2].message);
        assert!(
            problems[2].message.ends_with("; no model uses it, so score does not read it"),
            "{}",
            problems[2].message
        );
        let not_difficult = format!("region BED {NOT_DIFFICULT_REGIONS}=");
        assert!(problems[3].message.starts_with(&not_difficult), "{}", problems[3].message);
        assert!(
            problems[4].message.contains("has no intervals on the contigs (chr1)"),
            "{}",
            problems[4].message
        );
        assert_eq!(
            problems[5].message,
            "no model uses the --regions broken, other, other; they are ignored"
        );

        // Without an indel genotype model, score reads the not-difficult regions only with
        // --abstain-actions.
        let problems = check_with_beds(&[snp], &inputs);
        let not_difficult: Vec<&Problem> =
            problems.iter().filter(|p| p.message.starts_with("region BED notinall")).collect();
        assert_eq!(not_difficult.len(), 1, "{problems:?}");
        assert_eq!(not_difficult[0].severity, Severity::Warning);
        assert!(
            not_difficult[0].message.ends_with("so score reads it only with --abstain-actions")
        );
    }

    /// When a model could not be read, a BED may be used: its problems are errors, and none is
    /// called unused. The checks of the BEDs alone run with no model at all.
    #[test]
    fn regions_of_models_that_could_not_be_read() {
        let dir = TempDir::new().unwrap();
        let regions = [absent(&dir, "broken"), bed(&dir, "x", "chr9\t0\t10\n"), absent(&dir, "x")];
        let inputs =
            SetInputs { regions: &regions, unreadable_models: true, ..SetInputs::default() };
        let problems = check_with_beds(&[info(VariantType::Snp)], &inputs);
        assert_eq!(
            checks(&problems),
            [
                Check::RegionRepeated,
                Check::RegionUnreadable,
                Check::RegionUnreadable,
                Check::RegionContigs
            ]
        );
        assert!(severities(&problems).iter().all(|s| *s == Severity::Error), "{problems:?}");
        // No model: no contigs to check the BEDs against.
        let problems = check_with_beds(&[], &inputs);
        assert_eq!(
            checks(&problems),
            [Check::RegionRepeated, Check::RegionUnreadable, Check::RegionUnreadable]
        );
        assert!(severities(&problems).iter().all(|s| *s == Severity::Error), "{problems:?}");
    }

    /// Contigs and their lengths, as a model records them.
    type Lengths<'a> = &'a [(&'a str, Option<u64>)];

    /// Reference contigs of the given lengths.
    fn lengths(contigs: Lengths<'_>) -> ReferenceContigs {
        contigs.iter().map(|&(name, length)| (name.to_string(), length)).collect()
    }

    /// The merged reference contigs of models recording `contigs` (`None`: recording none).
    fn merged(contigs: &[Option<Lengths<'_>>]) -> Option<ReferenceContigs> {
        let infos: Vec<ModelInfo> = contigs
            .iter()
            .map(|contigs| {
                let mut info = info(VariantType::Snp);
                info.reference_contigs = contigs.map(lengths);
                info
            })
            .collect();
        let refs: Vec<&ModelInfo> = infos.iter().collect();
        merged_reference_contigs(&ModelSet { paths: &paths(infos.len()), infos: &refs })
    }

    #[test]
    fn reference_contigs_are_merged_where_the_models_agree() {
        let a: Lengths<'_> = &[("chr1", Some(1000)), ("chr2", None)];
        let b: Lengths<'_> = &[("chr2", Some(500)), ("chr1", Some(1000)), ("chr3", Some(9))];
        // Agreeing, with an unknown length filled by a later model.
        assert_eq!(
            merged(&[Some(a), Some(b)]),
            Some(lengths(&[("chr1", Some(1000)), ("chr2", Some(500)), ("chr3", Some(9))]))
        );
        // Disagreeing.
        assert_eq!(merged(&[Some(a), Some(&[("chr1", Some(999))])]), None);
        // A model recording none.
        assert_eq!(merged(&[Some(a), None]), None);
        assert_eq!(merged(&[]), Some(ReferenceContigs::new()));
    }

    #[rstest]
    #[case::other_version("0.0.1", true, Check::OtherWinnowrVersion)]
    #[case::no_reference_contigs(env!("CARGO_PKG_VERSION"), false, Check::NoReferenceContigs)]
    fn model_warnings_are_reported(
        #[case] version: &str,
        #[case] reference: bool,
        #[case] expected: Check,
    ) {
        let mut snp = info(VariantType::Snp);
        snp.winnowr_version = version.to_string();
        if !reference {
            snp.reference_contigs = None;
        }
        let problems = check(&[snp], &SetInputs::default());
        assert_eq!(checks(&problems), [expected]);
        assert_eq!(problems[0].severity, Severity::Warning);
    }

    /// Genotype snp and indel models trained on reads aligned as `snp` and `indel` record.
    fn aligned(snp: AlignmentProvenance, indel: AlignmentProvenance) -> [ModelInfo; 2] {
        let (mut a, mut b) = (info(VariantType::Snp), info(VariantType::Indel));
        a.alignment = Some(snp);
        b.alignment = Some(indel);
        [a, b]
    }

    /// The alignment problems of `infos` against reads aligned as `reads`.
    fn alignment_checks(
        infos: &[ModelInfo],
        reads: Option<&AlignmentProvenance>,
        unknown: UnknownReason,
        strict: bool,
    ) -> Vec<(Check, Severity)> {
        let inputs = SetInputs {
            reads: ReadsCheck { alignment: reads, unknown, strict },
            ..SetInputs::default()
        };
        check(infos, &inputs).iter().map(|p| (p.check, p.severity)).collect()
    }

    /// The models are merged before they are compared with the reads, whatever their order: a
    /// model of tags alone does not hide another's difference.
    #[test]
    fn the_models_are_merged_before_the_reads_are_compared() {
        let tags_only = provenance(&[("aligner", "bwa-mem3")], true);
        let header = provenance(&[("aligner", "bwa-mem3"), ("alt-project", "primary")], false);
        let reads = provenance(&[("aligner", "bwa-mem3"), ("alt-project", "keep-alt")], false);
        let mismatch = [(Check::AlignmentProvenanceMismatch, Severity::Warning)];
        for infos in
            [aligned(tags_only.clone(), header.clone()), aligned(header.clone(), tags_only.clone())]
        {
            let problems = check(
                &infos,
                &SetInputs {
                    reads: ReadsCheck { alignment: Some(&reads), ..ReadsCheck::default() },
                    ..SetInputs::default()
                },
            );
            assert_eq!(
                problems.iter().map(|p| (p.check, p.severity)).collect::<Vec<_>>(),
                mismatch
            );
            assert_eq!(problems[0].model, None);
        }
        // Two models of tags alone that record different tags are compared as their union.
        let a = provenance(&[("alt-project", "primary")], true);
        let b = provenance(&[("Y", "")], true);
        let reads = provenance(&[("alt-project", "keep-alt"), ("Y", "")], false);
        assert_eq!(
            alignment_checks(&aligned(a, b), Some(&reads), UnknownReason::NotGiven, false),
            mismatch
        );
    }

    /// Models that differ are a problem of their own (an error when strict), and each is then
    /// compared with the reads.
    #[test]
    fn models_aligned_differently_are_compared_one_by_one() {
        let primary = provenance(&[("alt-project", "primary")], false);
        let keep = provenance(&[("alt-project", "keep-alt")], false);
        let infos = aligned(primary.clone(), keep);
        assert_eq!(
            alignment_checks(&infos, Some(&primary), UnknownReason::NotGiven, true),
            [
                (Check::AlignmentProvenanceDiffers, Severity::Error),
                (Check::AlignmentProvenanceMismatch, Severity::Error)
            ]
        );
        let problems = check(
            &infos,
            &SetInputs {
                reads: ReadsCheck { alignment: Some(&primary), ..ReadsCheck::default() },
                ..SetInputs::default()
            },
        );
        assert!(problems[0].message.ends_with("Score with models trained on reads aligned alike"));
        assert_eq!(problems[1].model.as_deref(), Some(Path::new("b.winnowr.json")));
    }

    #[rstest]
    #[case::normal(false, Severity::Warning)]
    #[case::strict(true, Severity::Error)]
    fn equal_mixed_markers_are_indeterminate(#[case] strict: bool, #[case] severity: Severity) {
        let mixed = provenance(&[("alt-project", "mixed")], false);
        let infos = aligned(mixed.clone(), mixed.clone());
        let problems = check(
            &infos,
            &SetInputs {
                reads: ReadsCheck { alignment: Some(&mixed), strict, ..ReadsCheck::default() },
                ..SetInputs::default()
            },
        );
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert_eq!(problems[0].check.as_str(), "alignment_provenance_indeterminate");
        assert_eq!(problems[0].severity, severity);
        assert!(problems[0].message.contains("alt-project"));
        assert!(problems[0].message.contains("mixed"));
    }

    #[rstest]
    #[case::reads_only(false, true)]
    #[case::models_only(true, false)]
    #[case::both(true, true)]
    fn scoring_requires_complete_options_on_both_sides(
        #[case] model_incomplete: bool,
        #[case] reads_incomplete: bool,
        #[values(false, true)] strict: bool,
    ) {
        let provenance = |incomplete| {
            let mut provenance = provenance(&[("aligner", "bwa")], false);
            if incomplete {
                provenance.0.insert("options-inference".to_string(), "incomplete".to_string());
            }
            provenance
        };
        let model = provenance(model_incomplete);
        let reads = provenance(reads_incomplete);
        let infos = aligned(model.clone(), model);
        let problems = check(
            &infos,
            &SetInputs {
                reads: ReadsCheck { alignment: Some(&reads), strict, ..ReadsCheck::default() },
                ..SetInputs::default()
            },
        );
        let incomplete: Vec<_> = problems
            .iter()
            .filter(|problem| problem.check.as_str() == "alignment_provenance_incomplete")
            .collect();
        assert_eq!(
            incomplete.len(),
            usize::from(model_incomplete) * 2 + usize::from(reads_incomplete),
            "{problems:?}"
        );
        assert!(
            incomplete.iter().all(|problem| problem.severity
                == if strict { Severity::Error } else { Severity::Warning })
        );
        assert!(
            incomplete.iter().all(|problem| problem.message.contains("options-inference=explicit"))
        );
        if model_incomplete {
            let model_problems: Vec<_> =
                incomplete.iter().filter(|problem| problem.message.starts_with("model ")).collect();
            assert_eq!(model_problems.len(), 2, "{problems:?}");
            assert!(
                model_problems.iter().all(|problem| {
                    problem.message.contains("extract --alignment-tags options-inference=explicit")
                        && problem.message.contains("then retrain")
                }),
                "{problems:?}"
            );
        }
        assert!(
            !problems.iter().any(|problem| problem.check == Check::AlignmentProvenanceMismatch),
            "{problems:?}"
        );
    }

    #[rstest]
    fn constraints_from_complete_models_survive_an_incomplete_peer(
        #[values(false, true)] strict: bool,
        #[values(false, true)] incomplete_first: bool,
    ) {
        let complete = provenance(&[("aligner", "bwa")], false);
        let incomplete =
            provenance(&[("aligner", "bwa"), ("options-inference", "incomplete")], false);
        let reads = provenance(&[("aligner", "bwa"), ("p", "")], false);
        let infos = if incomplete_first {
            aligned(incomplete, complete)
        } else {
            aligned(complete, incomplete)
        };
        let problems = check(
            &infos,
            &SetInputs {
                reads: ReadsCheck { alignment: Some(&reads), strict, ..ReadsCheck::default() },
                ..SetInputs::default()
            },
        );
        let severity = if strict { Severity::Error } else { Severity::Warning };
        assert_eq!(
            problems.iter().map(|p| (p.check, p.severity)).collect::<Vec<_>>(),
            [
                (Check::AlignmentProvenanceMismatch, severity),
                (Check::AlignmentProvenanceIncomplete, severity),
            ],
            "{problems:?}"
        );
        assert_eq!(
            problems[0].differences,
            [ProvenanceDifference {
                tag: "p".to_string(),
                models: None,
                reads: Some(String::new()),
            }]
        );
    }

    #[test]
    fn scoring_legacy_models_ignore_incomplete_reads_provenance() {
        let reads = provenance(&[("options-inference", "incomplete")], false);
        let infos = [ModelInfo { alignment: None, ..info(VariantType::Snp) }];
        assert!(alignment_checks(&infos, Some(&reads), UnknownReason::NotGiven, true).is_empty());
    }

    #[test]
    fn ignored_or_uncompared_mixed_tags_are_not_indeterminate() {
        let header = provenance(
            &[("aligner", "bwa"), ("alt-project", "mixed"), ("version", "mixed")],
            false,
        );
        let partial = provenance(&[("aligner", "bwa")], true);
        let infos = aligned(partial.clone(), partial);
        assert!(alignment_checks(&infos, Some(&header), UnknownReason::NotGiven, true).is_empty());
        let version = provenance(&[("version", "mixed")], false);
        assert!(
            alignment_checks(
                &aligned(version.clone(), version.clone()),
                Some(&version),
                UnknownReason::NotGiven,
                true
            )
            .is_empty()
        );
    }

    /// Why the reads' alignment is not known is said, and an unreadable header is not also "not
    /// checked".
    #[rstest]
    #[case::not_given(
        UnknownReason::NotGiven,
        "neither --alignments nor --alignment-tags was given"
    )]
    #[case::not_recorded(UnknownReason::NotRecorded, "the reads' header identifies no aligner")]
    #[case::tags_none(UnknownReason::TagsNone, "--alignment-tags none was given")]
    fn reads_of_unknown_alignment_are_not_checked(
        #[case] unknown: UnknownReason,
        #[case] why: &str,
    ) {
        let primary = provenance(&[("alt-project", "primary")], false);
        let infos = aligned(primary.clone(), primary);
        let reads = ReadsCheck { unknown, ..ReadsCheck::default() };
        let inputs = SetInputs { reads, ..SetInputs::default() };
        let problems = check(&infos, &inputs);
        assert_eq!(checks(&problems), [Check::AlignmentProvenanceNotChecked]);
        assert!(problems[0].message.contains(why), "{}", problems[0].message);
        let unreadable = ReadsCheck { unknown: UnknownReason::Unreadable, ..reads };
        let unreadable = SetInputs { reads: unreadable, ..inputs };
        assert!(check(&infos, &unreadable).is_empty());
    }

    #[test]
    fn errors_make_one_message() {
        assert_eq!(error_message(&[]), None);
        let one = Problem::error(Check::ModelKinds, None, "two snp models");
        let warning = Problem::warning(Check::RegionUnused, None, "unused");
        assert_eq!(error_message(&[one.clone(), warning.clone()]).unwrap(), "two snp models");
        let other = Problem::error(Check::RegionMissing, None, "needs segdups");
        assert_eq!(
            error_message(&[one, warning, other]).unwrap(),
            "2 problems with the models and their inputs:\n  model_kinds: two snp models\n  \
             region_missing: needs segdups"
        );
    }

    /// A path that is not UTF-8 is written in JSON with its other bytes replaced.
    #[cfg(unix)]
    #[test]
    fn paths_that_are_not_utf8_are_written_lossily() {
        use std::os::unix::ffi::OsStrExt;
        let path = Path::new(std::ffi::OsStr::from_bytes(b"a\xffb.winnowr.json"));
        let problem = Problem::error(Check::ModelUnreadable, Some(path), "unreadable");
        let json = serde_json::to_value(&problem).unwrap();
        assert_eq!(json["model"], "a\u{fffd}b.winnowr.json");
    }
}
