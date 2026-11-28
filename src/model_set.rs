//! The checks of a set of models before a run: that they can be scored together, and against the
//! inputs given (region names, the reference's contig lengths). `score` runs them at startup, and
//! `winnowr check-models` runs them with [`check_region_beds`], which reads the region BEDs (see
//! [`crate::check_models`]). Every check runs, and every problem is reported, with a stable
//! [`Check`] id.
//!
//! The checks read each model's information only (see [`crate::model::ModelInfo::read_file`]): a
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
use crate::io::FileIdentity;
use crate::model::{ModelInfo, ReferenceContigs, check_reference};
use crate::model_evidence::{ModelEvidence, models_options};
use crate::pileup::PileupOptions;
use crate::policy::{
    NOT_DIFFICULT_REGIONS, duplicate_kinds, is_indel_genotype, model_kinds_message,
};
use crate::regions::{NamedBed, RecordedDigests, RegionDigest, RegionSet};
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
    /// Several region BEDs of a name beside the models, with different intervals.
    RegionAmbiguous,
    /// A region BED's digest is not the one the models record: an error, or a warning with
    /// `--allow-region-mismatch`.
    RegionChecksum,
    /// The models record different digests of a region BED.
    RegionDigestsDiffer,
    /// A region BED a model uses whose digest no model records, so it is used unchecked.
    RegionUnverified,
    /// An indel genotype model, and no model records the not-difficult regions' policy digest.
    PolicyRegionUnrecorded,
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
    RegionAmbiguous => "region_ambiguous",
    RegionChecksum => "region_checksum",
    RegionDigestsDiffer => "region_digests_differ",
    RegionUnverified => "region_unverified",
    PolicyRegionUnrecorded => "policy_region_unrecorded",
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
    /// The models' paths, as given.
    pub paths: &'a [PathBuf],
    /// The file each of `paths` resolved to when its information was read (see
    /// [`crate::model::ModelFile`]): the region BEDs are looked for beside it, not beside where the
    /// path resolves to now.
    pub resolved: &'a [PathBuf],
    /// The directory of each of `paths` as given, resolved when its information was read (see
    /// [`crate::model::ModelFile::given_dir`]): the region BEDs are looked for in it, not in
    /// wherever the directory resolves to now.
    pub given_dirs: &'a [PathBuf],
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
/// names) runs even when the set is empty. The region BEDs are neither read (see
/// [`check_region_beds`]) nor looked for ([`Check::RegionMissing`] is [`RegionPlan::new`]'s).
#[must_use]
pub fn check_model_set(set: &ModelSet<'_>, inputs: &SetInputs<'_>) -> Vec<Problem> {
    let mut problems: Vec<Problem> = region_names(inputs).into_iter().collect();
    if !set.infos.is_empty() {
        // A file given twice is one problem: the other checks see it once.
        let repeats = repeated_files(set.resolved);
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
        problems.extend(unused_regions(set, inputs));
        problems.extend(model_warnings(set));
        problems.extend(alignment_provenance_problems(set, inputs.reads));
    }
    problems.sort_by_key(|problem| problem.check);
    problems
}

/// For each of the files the models were read from (`resolved`, see [`ModelSet::resolved`]),
/// whether it is one read earlier. They are compared as resolved when read, so that two spellings
/// of one file are the same, and a symlink retargeted since does not change which file was read.
fn repeated_files(resolved: &[PathBuf]) -> Vec<bool> {
    let mut seen = BTreeSet::new();
    resolved.iter().map(|file| !seen.insert(file)).collect()
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

/// True if an indel genotype model is among the models, which the default policy decides with
/// the not-difficult regions (see [`is_indel_genotype`]).
#[must_use]
pub fn has_indel_genotype(infos: &[&ModelInfo]) -> bool {
    infos.iter().any(|info| is_indel_genotype(info))
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

/// True if `score` may read the region BED `name` for the set: it is one of the
/// [`needed_region_names`] of the set (with the not-difficult regions when an indel genotype model
/// will score, as the default policy decides with them), or `unreadable_models` (some models could
/// not be read, so which BEDs they use is not known).
#[must_use]
fn region_needed(set: &ModelSet<'_>, name: &str, unreadable_models: bool) -> bool {
    unreadable_models || needed_region_names(set, has_indel_genotype(set.infos)).contains(&name)
}

/// [`Check::RegionRepeated`]: each region name is given once.
fn region_names(inputs: &SetInputs<'_>) -> Option<Problem> {
    let repeated = crate::common::repeated(inputs.regions.iter().map(|r| r.name.as_str()));
    (!repeated.is_empty()).then(|| {
        let message = format!("region names must be unique; repeated: {}", repeated.join(", "));
        Problem::error(Check::RegionRepeated, None, message)
    })
}

/// Reads the region BED `name` at `path`, and checks it names some of `contigs` (when given): the
/// intervals with the [`Check::RegionContigs`] error if it names none, or the
/// [`Check::RegionUnreadable`] error if it cannot be read.
fn read_checked_bed<S: AsRef<str>>(
    name: &str,
    path: &Path,
    contigs: Option<&[S]>,
) -> Result<(RegionSet, Option<Problem>), Problem> {
    read_checked_bed_with_identity(name, path, path, contigs, false)
        .map(|bed| (bed.regions, bed.contigs))
}

/// Reads the selected target while naming its original alias in diagnostics, retaining the
/// identity captured from the opened BED handle for output guards.
fn read_checked_bed_with_identity<S: AsRef<str>>(
    name: &str,
    path: &Path,
    alias: &Path,
    contigs: Option<&[S]>,
    regular_only: bool,
) -> Result<ReadBed, Problem> {
    let unreadable = |error: anyhow::Error| {
        let message = format!("region BED {name}={}: {error:#}", alias.display());
        Problem::error(Check::RegionUnreadable, None, message)
    };
    let (regions, identity) = if regular_only {
        RegionSet::from_regular_bed_with_identity(name, path)
    } else {
        RegionSet::from_bed_with_identity(name, path)
    }
    .map_err(unreadable)?;
    let digest = regions.digest().map_err(unreadable)?;
    let contigs = contigs.and_then(|contigs| regions.check_contigs(contigs).err()).map(|error| {
        Problem::error(Check::RegionContigs, None, format!("{}: {error}", alias.display()))
    });
    Ok(ReadBed { regions, digest, contigs, identity })
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
            let mut problem = match read_checked_bed(&bed.name, &bed.path, contigs) {
                Ok((_, problem)) => problem?,
                Err(problem) => problem,
            };
            if !region_needed(set, &bed.name, inputs.unreadable_models) {
                let why = if bed.name == NOT_DIFFICULT_REGIONS {
                    "no indel genotype model uses it, so score reads it only with --abstain-actions"
                } else {
                    "no model uses it, so score does not read it"
                };
                problem.severity = Severity::Warning;
                problem.message = format!("{}; {why}", problem.message);
            }
            Some(problem)
        })
        .collect()
}

/// [`Check::RegionUnused`]: a region BED given that no model uses (not when a model could not be
/// read, as what it uses is not known). [`Check::RegionMissing`] is [`RegionPlan::new`]'s.
fn unused_regions(set: &ModelSet<'_>, inputs: &SetInputs<'_>) -> Option<Problem> {
    // The not-difficult regions are not called unused: score reads them with --abstain-actions.
    let mut unused: Vec<&str> = inputs
        .regions
        .iter()
        .map(|r| r.name.as_str())
        .filter(|name| {
            *name != NOT_DIFFICULT_REGIONS && !region_needed(set, name, inputs.unreadable_models)
        })
        .collect();
    (!unused.is_empty()).then(|| {
        unused.sort_unstable();
        let message =
            format!("no model uses the --regions {}; they are ignored", unused.join(", "));
        Problem::warning(Check::RegionUnused, None, message)
    })
}

/// The suffixes of a region BED beside the models (`regions/NAME` and the suffix), in the order
/// they are looked for.
const BUNDLED_SUFFIXES: [&str; 3] = [".bed.gz", ".bed.bgz", ".bed"];

/// The files a region BED of `name` is looked for as beside the models, for messages:
/// `regions/NAME.bed.gz, regions/NAME.bed.bgz, regions/NAME.bed`.
#[must_use]
pub fn bundled_bed_files(name: &str) -> String {
    let files: Vec<String> =
        BUNDLED_SUFFIXES.iter().map(|suffix| format!("regions/{name}{suffix}")).collect();
    files.join(", ")
}

/// True if a region name is one file name (not empty, `.` or `..`, and without a path
/// separator): only such names are looked for beside the models, as model files name them.
fn is_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\'])
}

/// The `regions` directories the region BEDs are looked for in beside the models of `set`: in
/// the directory of each model as given, then beside the file it resolved to (pipelines stage
/// models as symlinks), both as resolved when the model was read. Neither is resolved again: a
/// symlink (to the file or to its directory) retargeted since would pair the model with another
/// bundle's BEDs.
fn bundled_dirs(set: &ModelSet<'_>) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for (given_dir, resolved) in set.given_dirs.iter().zip(set.resolved) {
        let beside = resolved.parent().unwrap_or_else(|| Path::new(""));
        for dir in [given_dir.join("regions"), beside.join("regions")] {
            if !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    dirs
}

/// Existing candidates among the discovery `paths`, each canonical target once. Broken links
/// and paths that cannot be inspected are retained so reading reports them as unreadable.
fn bundled_beds(paths: &[PathBuf]) -> Vec<BedCandidate> {
    let mut found = Vec::new();
    let mut seen = BTreeSet::new();
    for alias in paths {
        let present = match alias.symlink_metadata() {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                // Keep broken or uninspectable links for diagnostics, but never discover streams
                // or directories through links when their direct entries would be ignored.
                alias.metadata().map_or(true, |target| target.is_file())
            }
            Ok(metadata) => metadata.is_file(),
            Err(error) => error.kind() != std::io::ErrorKind::NotFound,
        };
        if present {
            let (path, resolution_error) = match alias.canonicalize() {
                Ok(path) => (path, None),
                // Never follow the alias later: it may be retargeted to another bundle's BED.
                Err(error) => (alias.clone(), Some(format!("cannot resolve it: {error}"))),
            };
            if seen.insert(path.clone()) {
                found.push(BedCandidate { alias: alias.clone(), path, resolution_error });
            }
        }
    }
    found
}

/// All discovery aliases for one automatic region, including files not yet present.
fn bundled_paths(dirs: &[PathBuf], name: &str) -> Vec<PathBuf> {
    if !is_file_name(name) {
        return Vec::new();
    }
    dirs.iter()
        .flat_map(|dir| BUNDLED_SUFFIXES.map(|suffix| dir.join(format!("{name}{suffix}"))))
        .collect()
}

/// One BED discovery alias and the pinned canonical target to read, when resolvable.
#[derive(Debug, Clone)]
struct BedCandidate {
    alias: PathBuf,
    path: PathBuf,
    /// Why a bundled alias could not be resolved: it is reported unreadable, never reopened.
    resolution_error: Option<String>,
}

impl BedCandidate {
    /// Reads and checks the BED (see [`read_checked_bed_with_identity`]), or reports it unreadable
    /// without opening it if it could not be resolved when found.
    fn read<S: AsRef<str>>(
        &self,
        name: &str,
        contigs: Option<&[S]>,
        regular_only: bool,
    ) -> Result<ReadBed, Problem> {
        if let Some(error) = &self.resolution_error {
            let message = format!("region BED {name}={}: {error}", self.alias.display());
            return Err(Problem::error(Check::RegionUnreadable, None, message));
        }
        read_checked_bed_with_identity(name, &self.path, &self.alias, contigs, regular_only)
    }
}

/// The region names a run reads, each once: those of the models' features and, with `policy`
/// (when the decision policy reads them), the not-difficult regions.
#[must_use]
pub fn needed_region_names<'a>(set: &ModelSet<'a>, policy: bool) -> Vec<&'a str> {
    let mut names: Vec<&str> = Vec::new();
    for info in set.infos {
        for name in &info.region_names {
            if !names.contains(&name.as_str()) {
                names.push(name);
            }
        }
    }
    if policy && !names.contains(&NOT_DIFFICULT_REGIONS) {
        names.push(NOT_DIFFICULT_REGIONS);
    }
    names
}

/// A role a region BED serves in a run, for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionRole {
    /// A feature's region of the model ([`ModelInfo::region_digests`]).
    Feature,
    /// The decision policy's regions, which the model was trained to be scored with
    /// ([`ModelInfo::policy_region_digests`]).
    Policy,
}

/// A model that uses a region BED in a role (recording its digest there, or not), and the role.
pub type DigestSource<'a> = (&'a Path, RegionRole);

/// Writes a model and the role it records a region's digest in, for messages: the model's path,
/// and ` (policy)` for the policy's regions.
fn digest_source(source: DigestSource<'_>) -> String {
    match source {
        (path, RegionRole::Feature) => path.display().to_string(),
        (path, RegionRole::Policy) => format!("{} (policy)", path.display()),
    }
}

/// The digests the models record for the region `name` in the roles the BED serves in a run,
/// each with the models (and roles) that record it: the [`ModelInfo::region_digests`] entry of
/// each model that uses it as a feature when the run reads features, and, when the policy reads
/// it, every model's [`ModelInfo::policy_region_digests`] entry. A table-based run reads only the
/// policy BED; a BED serving both roles must match both records.
#[must_use]
pub fn recorded_digests<'a>(
    set: &ModelSet<'a>,
    name: &str,
    run: RunRegions,
) -> RecordedDigests<'a, DigestSource<'a>> {
    let features = matches!(run, RunRegions::Features { .. });
    let policy = matches!(run, RunRegions::Features { policy: true } | RunRegions::Policy(true))
        && name == NOT_DIFFICULT_REGIONS;
    let mut sources = Vec::new();
    for (path, info) in set.paths.iter().zip(set.infos) {
        if features && info.uses_region(name) {
            sources.push(((path.as_path(), RegionRole::Feature), info.region_digests.get(name)));
        }
        if policy {
            let digest = info.policy_region_digests.get(name);
            // A model that records none for the policy says nothing of it.
            if digest.is_some() {
                sources.push(((path.as_path(), RegionRole::Policy), digest));
            }
        }
    }
    RecordedDigests::group(sources)
}

/// [`Check::RegionDigestsDiffer`], of each region a run reads (see [`needed_region_names`], with
/// `policy` when the policy reads the not-difficult regions): the models record different digests
/// of it, in the roles the BED serves (see [`recorded_digests`]). An error whatever the BEDs
/// given, and even with `--allow-region-mismatch`: no BED is the one all of them record.
#[must_use]
pub fn region_digests_differ(set: &ModelSet<'_>, policy: bool) -> Vec<Problem> {
    needed_region_names(set, policy)
        .into_iter()
        .filter_map(|name| {
            digests_differ_problem(
                name,
                &recorded_digests(set, name, RunRegions::Features { policy }),
            )
        })
        .collect()
}

/// How a region BED that is not the one the models record is reported, and what to do: an error,
/// advising `fix` or `--allow-region-mismatch`, or with `allow_mismatch` a warning that it is used
/// anyway.
#[must_use]
pub fn mismatch_severity(allow_mismatch: bool, fix: &str) -> (Severity, String) {
    if allow_mismatch {
        (Severity::Warning, "it is used anyway (--allow-region-mismatch)".to_string())
    } else {
        (Severity::Error, format!("{fix}, or --allow-region-mismatch to use it anyway"))
    }
}

/// Where a region BED a run uses came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionFrom {
    /// `--regions NAME=PATH`.
    Given,
    /// `regions/NAME.bed.gz`, `.bed.bgz` or `.bed` beside the models.
    Bundled,
}

impl RegionFrom {
    /// `given` or `bundled`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RegionFrom::Given => "given",
            RegionFrom::Bundled => "bundled",
        }
    }
}

/// A region BED a run uses, read, with the digest the models record for it. Its intervals are
/// returned beside it (see [`RegionPlan::read`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRegion {
    /// Its name.
    pub name: String,
    /// Its file.
    pub path: PathBuf,
    /// The index of its `--regions` BED among those given, or `None` if found beside the models.
    pub given: Option<usize>,
    /// Its digest.
    pub digest: RegionDigest,
    /// The digest the models record for it (`None` if none does, or they differ).
    pub recorded: Option<RegionDigest>,
    /// The opened regular-file identities of every candidate read for this region, not just the
    /// candidate selected for its intervals. Streams have no file identity.
    pub(crate) input_identities: Vec<FileIdentity>,
}

impl ResolvedRegion {
    /// Where it came from.
    #[must_use]
    pub fn origin(&self) -> RegionFrom {
        if self.given.is_some() { RegionFrom::Given } else { RegionFrom::Bundled }
    }

    /// Whether it is the BED the models record: `None` if they record none.
    #[must_use]
    pub fn verified(&self) -> Option<bool> {
        self.recorded.as_ref().map(|recorded| *recorded == self.digest)
    }
}

/// Which region BEDs a run reads (see [`RegionPlan`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunRegions {
    /// `score` and `check-models`: those of the models' features (each checked to name some of the
    /// models' contigs, and [`Check::RegionMissing`] if neither given nor found) and, with
    /// `policy` (when the policy reads them), the not-difficult regions.
    Features {
        /// Whether the policy reads the not-difficult regions.
        policy: bool,
    },
    /// `evaluate` and `stratify`, whose features come from a table: the not-difficult regions,
    /// when the policy reads them (`true`).
    Policy(bool),
}

impl RunRegions {
    /// The names of the region BEDs the run reads, each once.
    #[must_use]
    pub fn names<'a>(self, set: &ModelSet<'a>) -> Vec<&'a str> {
        match self {
            RunRegions::Features { policy } => needed_region_names(set, policy),
            RunRegions::Policy(true) => vec![NOT_DIFFICULT_REGIONS],
            RunRegions::Policy(false) => Vec::new(),
        }
    }
}

/// A region BED a run reads, found but not read.
#[derive(Debug, Clone)]
struct Located<'a> {
    /// Its name.
    name: &'a str,
    /// The index of its `--regions` BED, or `None` if looked for beside the models.
    given: Option<usize>,
    /// Its `--regions` BED, or those found beside the models (none, one, or several of other
    /// suffixes or directories).
    candidates: Vec<BedCandidate>,
    /// All automatic discovery aliases, including absent siblings. Empty for explicit overrides.
    prospective: Vec<PathBuf>,
}

/// The region BEDs a run reads, found (each the `--regions` BED of the name, else those beside the
/// models, see [`bundled_dirs`]) but not yet read: [`RegionPlan::read`] retains intervals for
/// scoring, while [`RegionPlan::check`] retains only report metadata and diagnostics.
#[derive(Debug)]
pub struct RegionPlan<'a> {
    /// The models.
    set: ModelSet<'a>,
    /// The BEDs, in the order of [`RunRegions::names`].
    located: Vec<Located<'a>>,
    /// The models' contigs, each BED is checked to name some of (for a run of the features).
    contigs: Option<&'a [String]>,
    /// The active roles of the BEDs: feature and/or policy.
    run: RunRegions,
}

impl<'a> RegionPlan<'a> {
    /// Finds the region BEDs of a `run` of the models `set`, with the `--regions` BEDs `given`,
    /// with the [`Check::RegionMissing`] errors of a run of the features: a region a model uses
    /// that is neither given nor found (the policy's regions may be absent).
    #[must_use]
    pub fn new(set: &ModelSet<'a>, given: &[NamedBed], run: RunRegions) -> (Self, Vec<Problem>) {
        let dirs = bundled_dirs(set);
        let located: Vec<Located<'a>> = run
            .names(set)
            .into_iter()
            .map(|name| match given.iter().position(|bed| bed.name == name) {
                Some(index) => Located {
                    name,
                    given: Some(index),
                    candidates: vec![BedCandidate {
                        alias: given[index].path.clone(),
                        path: given[index].path.clone(),
                        resolution_error: None,
                    }],
                    prospective: Vec::new(),
                },
                None => {
                    let prospective = bundled_paths(&dirs, name);
                    let candidates = bundled_beds(&prospective);
                    Located { name, given: None, candidates, prospective }
                }
            })
            .collect();
        let features = matches!(run, RunRegions::Features { .. });
        let problems = if features { missing_regions(set, &located, &dirs) } else { Vec::new() };
        let contigs = set.infos.first().map(|info| info.contigs.as_slice()).filter(|_| features);
        (RegionPlan { set: *set, located, contigs, run }, problems)
    }

    /// Every BED path the plan reads, including equivalent bundled candidates that resolve to
    /// one region. Used to protect consumed inputs before creating outputs.
    pub fn paths(&self) -> impl Iterator<Item = &Path> {
        self.located.iter().flat_map(|located| {
            located.candidates.iter().flat_map(|candidate| {
                std::iter::once(candidate.alias.as_path())
                    .chain((candidate.path != candidate.alias).then_some(candidate.path.as_path()))
            })
        })
    }

    /// All paths automatically discovered regions can use, including absent BED siblings.
    /// Explicit overrides bypass bundled discovery and contribute no paths here.
    pub fn prospective_paths(&self) -> impl Iterator<Item = &Path> {
        self.located.iter().flat_map(|located| located.prospective.iter().map(PathBuf::as_path))
    }

    /// The selected explicit input indices, even when their later reads fail.
    pub fn given_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.located.iter().filter_map(|located| located.given)
    }

    /// Whether a candidate was given or found for `name`, even if it cannot be read or selected.
    #[must_use]
    pub fn has_candidate(&self, name: &str) -> bool {
        self.located.iter().any(|located| located.name == name && !located.candidates.is_empty())
    }

    /// [`Check::RegionDigestsDiffer`] of each region the plan reads, in the roles the run reads
    /// it: known from the models alone, so a caller can fail before any BED is opened (a stream
    /// without a producer would otherwise block first). [`Self::read`] reports them too.
    #[must_use]
    pub fn digest_conflicts(&self) -> Vec<Problem> {
        self.located
            .iter()
            .filter_map(|located| {
                digests_differ_problem(
                    located.name,
                    &recorded_digests(&self.set, located.name, self.run),
                )
            })
            .collect()
    }

    /// Identifies conflicting streams before either consumer opens any BED, so only those
    /// candidates are skipped. Regular files can be read independently; streams may have one producer.
    fn shared_stream_conflicts(&self) -> (BTreeSet<PathBuf>, Vec<Problem>) {
        let mut seen: Vec<(&str, &BedCandidate)> = Vec::new();
        let mut blocked = BTreeSet::new();
        let mut problems = Vec::new();
        for located in &self.located {
            for candidate in &located.candidates {
                // An unresolved candidate is never opened, so it shares no stream.
                if candidate.resolution_error.is_some()
                    || std::fs::metadata(&candidate.path).is_ok_and(|metadata| metadata.is_file())
                {
                    continue;
                }
                for (name, previous) in seen.iter().filter(|(name, previous)| {
                    *name != located.name
                        && crate::io::same_input_path(&previous.path, &candidate.path)
                }) {
                    blocked.insert(previous.path.clone());
                    blocked.insert(candidate.path.clone());
                    let message = format!(
                        "region BEDs {name}={} and {}={} share the same nonregular input; \
                         give distinct streams or use a regular BED file for multiple names",
                        previous.alias.display(),
                        located.name,
                        candidate.alias.display()
                    );
                    problems.push(Problem::error(Check::RegionUnreadable, None, message));
                }
                seen.push((located.name, candidate));
            }
        }
        (blocked, problems)
    }

    /// Checks the BEDs as [`Self::read`] does (every file in parallel, so distinct streams one
    /// producer feeds in turn are all opened), each worker dropping its intervals as soon as it
    /// has their digest. Returns only the report metadata and diagnostics, with the same checks
    /// and ordering as [`Self::read`].
    #[must_use]
    pub fn check(self, allow_mismatch: bool) -> (Vec<ResolvedRegion>, Vec<Problem>) {
        let (resolved, problems) = self.read_keeping(allow_mismatch, drop);
        (resolved.into_iter().map(|(region, ())| region).collect(), problems)
    }

    /// Reads the BEDs (every file in parallel) and checks each against the digest the models
    /// record for it. Returns each BED read, with its intervals, and the problems, by check:
    /// [`Check::RegionUnreadable`], [`Check::RegionContigs`], [`Check::RegionAmbiguous`] (several
    /// BEDs beside the models with different intervals: none is used),
    /// [`Check::RegionDigestsDiffer`], [`Check::RegionChecksum`] (an error, or a warning when
    /// `allow_mismatch`) and [`Check::RegionUnverified`] (for a region a model's features use; the policy's regions
    /// recorded by no model are used without a problem).
    #[must_use]
    pub fn read(self, allow_mismatch: bool) -> (Vec<(ResolvedRegion, RegionSet)>, Vec<Problem>) {
        self.read_keeping(allow_mismatch, |regions| regions)
    }

    /// [`Self::read`], keeping of each BED's intervals what `keep` returns, in the worker that
    /// read them.
    fn read_keeping<R: Send>(
        self,
        allow_mismatch: bool,
        keep: impl Fn(RegionSet) -> R + Sync,
    ) -> (Vec<(ResolvedRegion, R)>, Vec<Problem>) {
        let (blocked, mut problems) = self.shared_stream_conflicts();
        let jobs: Vec<(usize, &BedCandidate)> = self
            .located
            .iter()
            .enumerate()
            .flat_map(|(i, located)| located.candidates.iter().map(move |candidate| (i, candidate)))
            .filter(|(_, candidate)| !blocked.contains(&candidate.path))
            .collect();
        let read: Vec<CandidateRead<'_, R>> = jobs
            .into_par_iter()
            .map(|(i, candidate)| {
                let located = &self.located[i];
                let read = candidate.read(located.name, self.contigs, located.given.is_none()).map(
                    |ReadBed { regions, digest, contigs, identity }| ReadBed {
                        regions: keep(regions),
                        digest,
                        contigs,
                        identity,
                    },
                );
                (i, candidate.alias.as_path(), read)
            })
            .collect();
        let mut by_region: Vec<Vec<(PathBuf, ReadBed<R>)>> =
            self.located.iter().map(|_| Vec::new()).collect();
        for (i, path, read) in read {
            match read {
                Ok(read) => by_region[i].push((path.to_path_buf(), read)),
                Err(problem) => problems.push(problem),
            }
        }
        let mut resolved = Vec::new();
        for (located, read) in self.located.iter().zip(by_region) {
            resolved.extend(self.resolve(located, read, allow_mismatch, &mut problems));
        }
        problems.sort_by_key(|problem| problem.check);
        (resolved, problems)
    }

    /// The region BED of `located`, of the files read (`read`), checked against the digest the
    /// models record, adding the problems: `None` if none was read, or several beside the models
    /// differ.
    fn resolve<R>(
        &self,
        located: &Located<'_>,
        read: Vec<(PathBuf, ReadBed<R>)>,
        allow_mismatch: bool,
        problems: &mut Vec<Problem>,
    ) -> Option<(ResolvedRegion, R)> {
        let name = located.name;
        let recorded = recorded_digests(&self.set, name, self.run);
        problems.extend(digests_differ_problem(name, &recorded));
        if read.iter().any(|(_, bed)| bed.digest != read[0].1.digest) {
            let paths: Vec<String> =
                read.iter().map(|(path, _)| path.display().to_string()).collect();
            let message = format!(
                "the {name} region BEDs beside the models differ: {}; keep one, or pass --regions \
                 {name}=PATH",
                paths.join(", ")
            );
            problems.extend(read.into_iter().filter_map(|(_, bed)| bed.contigs));
            problems.push(Problem::error(Check::RegionAmbiguous, None, message));
            return None;
        }
        let input_identities = read.iter().filter_map(|(_, bed)| bed.identity).collect();
        let (path, ReadBed { regions, digest, contigs, identity: _ }) = read.into_iter().next()?;
        problems.extend(contigs);
        let region = ResolvedRegion {
            name: name.to_string(),
            path,
            given: located.given,
            digest,
            recorded: recorded.agreed().cloned(),
            input_identities,
        };
        let unrecorded = matches!(self.run, RunRegions::Features { .. })
            && recorded.digests.is_empty()
            && self.set.infos.iter().any(|info| info.uses_region(name));
        problems.extend(verification_problem(&region, unrecorded, allow_mismatch));
        Some((region, regions))
    }
}

/// A candidate read by [`RegionPlan::read_keeping`]: the index of its region, its alias, and the
/// BED read or the problem reading it.
type CandidateRead<'a, R> = (usize, &'a Path, Result<ReadBed<R>, Problem>);

/// A region BED read, with its digest and the [`Check::RegionContigs`] error if it names none of
/// the models' contigs.
struct ReadBed<R = RegionSet> {
    /// Its intervals when needed by the caller, or `()` for a report-only check.
    regions: R,
    /// Their digest.
    digest: RegionDigest,
    /// Its [`Check::RegionContigs`] error.
    contigs: Option<Problem>,
    /// The identity captured from the same handle as its intervals, if a regular file.
    identity: Option<FileIdentity>,
}

/// [`Check::RegionMissing`]: of each model, the regions it uses (among the `located`) neither given
/// nor found beside the models (in the `dirs`).
fn missing_regions(set: &ModelSet<'_>, located: &[Located<'_>], dirs: &[PathBuf]) -> Vec<Problem> {
    let missing: Vec<&str> =
        located.iter().filter(|l| l.candidates.is_empty()).map(|l| l.name).collect();
    if missing.is_empty() {
        return Vec::new();
    }
    let looked_in: Vec<String> = dirs.iter().map(|dir| dir.display().to_string()).collect();
    let mut problems = Vec::new();
    for (path, info) in set.paths.iter().zip(set.infos) {
        let mut names: Vec<&str> = info
            .region_names
            .iter()
            .map(String::as_str)
            .filter(|name| missing.contains(name))
            .collect();
        if names.is_empty() {
            continue;
        }
        names.sort_unstable();
        let not_files: Vec<&str> =
            names.iter().copied().filter(|name| !is_file_name(name)).collect();
        let not_files = if not_files.is_empty() {
            String::new()
        } else {
            format!(
                "; {} cannot be looked for beside the models, not being file names",
                not_files.join(", ")
            )
        };
        let message = format!(
            "model {} ({} {}) needs region BEDs for: {}; pass --regions NAME=PATH, or put each \
             beside the models as {} (looked in: {}){not_files}",
            path.display(),
            info.task,
            info.variant_type,
            names.join(", "),
            bundled_bed_files("NAME"),
            looked_in.join(", ")
        );
        problems.push(Problem::error(Check::RegionMissing, Some(path), message));
    }
    problems
}

/// [`Check::RegionDigestsDiffer`]: the models record different digests of the region `name`, in
/// the roles of `recorded`.
fn digests_differ_problem(
    name: &str,
    recorded: &RecordedDigests<'_, DigestSource<'_>>,
) -> Option<Problem> {
    let ((first_source, first), (other_source, other)) = recorded.conflict()?;
    let message = format!(
        "--models were trained with different {name} region BEDs: {} records {first}, {} \
         {other}; use models trained with the same BED",
        digest_source(first_source),
        digest_source(other_source)
    );
    Some(Problem::error(Check::RegionDigestsDiffer, Some(other_source.0), message))
}

/// [`Check::RegionChecksum`] (a BED that is not the one the models record: an error, or a warning
/// when `allow_mismatch`) and [`Check::RegionUnverified`] (a BED of a feature's region whose
/// digest no model records: `unrecorded`).
fn verification_problem(
    region: &ResolvedRegion,
    unrecorded: bool,
    allow_mismatch: bool,
) -> Option<Problem> {
    let name = &region.name;
    match &region.recorded {
        Some(recorded) if *recorded != region.digest => {
            let (severity, advice) =
                mismatch_severity(allow_mismatch, "pass the BED the models were trained with");
            let message = format!(
                "the {name} region BED {} ({}) is not the one the models were trained with: it \
                 has {}, the models' {recorded}; {advice}",
                region.path.display(),
                region.origin().as_str(),
                region.digest
            );
            Some(Problem::new(Check::RegionChecksum, severity, None, message))
        }
        None if unrecorded => {
            let message = format!(
                "no model records the digest of the {name} region BED, so {} is used unchecked",
                region.path.display()
            );
            Some(Problem::warning(Check::RegionUnverified, None, message))
        }
        _ => None,
    }
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
    use std::collections::BTreeMap;

    use rstest::rstest;
    use tempfile::TempDir;

    use super::*;
    use crate::common::{Task, VariantType};
    use crate::read_annotations::ReadAnnotationOptions;
    use crate::testing::provenance;

    /// Finds, reads and checks a run's BEDs, including errors from both planning and reading.
    fn resolve_run_regions(
        set: &ModelSet<'_>,
        given: &[NamedBed],
        run: RunRegions,
        allow_mismatch: bool,
    ) -> (Vec<(ResolvedRegion, RegionSet)>, Vec<Problem>) {
        let (plan, mut problems) = RegionPlan::new(set, given, run);
        let (resolved, read_problems) = plan.read(allow_mismatch);
        problems.extend(read_problems);
        problems.sort_by_key(|problem| problem.check);
        (resolved, problems)
    }

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
        check_model_set(
            &ModelSet {
                paths: &paths,
                resolved: &paths,
                given_dirs: &crate::testing::given_dirs(&paths),
                infos: &refs,
            },
            inputs,
        )
    }

    /// The problems of `infos` against `inputs`, with those of finding (as `check-models` does)
    /// and reading the region BEDs, by check.
    fn check_with_beds(infos: &[ModelInfo], inputs: &SetInputs<'_>) -> Vec<Problem> {
        let paths = paths(infos.len());
        let refs: Vec<&ModelInfo> = infos.iter().collect();
        let set = ModelSet {
            paths: &paths,
            resolved: &paths,
            given_dirs: &crate::testing::given_dirs(&paths),
            infos: &refs,
        };
        let mut problems = check_model_set(&set, inputs);
        let run = RunRegions::Features { policy: has_indel_genotype(&refs) };
        problems.extend(RegionPlan::new(&set, inputs.regions, run).1);
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
        assert_eq!(Check::ALL.len(), Check::PolicyRegionUnrecorded as usize + 1);
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
        let files = crate::model::ModelFile::resolve_all(&paths);
        let resolved: Vec<PathBuf> = files.iter().map(|file| file.resolved.clone()).collect();
        let given_dirs: Vec<PathBuf> = files.iter().map(|file| file.given_dir.clone()).collect();
        let snp = info(VariantType::Snp);
        let set = ModelSet {
            paths: &paths,
            resolved: &resolved,
            given_dirs: &given_dirs,
            infos: &[&snp, &snp],
        };
        let problems = check_model_set(&set, &SetInputs::default());
        assert_eq!(checks(&problems), [Check::DuplicateModel]);
        assert_eq!(problems[0].model.as_deref(), Some(other_spelling.as_path()));
    }

    /// Duplicates are of the files the models were read from: two staged symlinks to different
    /// files are not duplicates even if one is retargeted to the other's file after they were
    /// read, and are if they were read from one file whatever they resolve to since.
    #[cfg(unix)]
    #[test]
    fn duplicates_are_of_the_files_the_models_were_read_from() {
        let dir = TempDir::new().unwrap();
        let targets = [dir.path().join("x.winnowr.json"), dir.path().join("y.winnowr.json")];
        for target in &targets {
            std::fs::write(target, "{}").unwrap();
        }
        let links = [dir.path().join("a.winnowr.json"), dir.path().join("b.winnowr.json")];
        let link = |link: &Path, target: &Path| {
            let _ = std::fs::remove_file(link);
            std::os::unix::fs::symlink(target, link).unwrap();
        };
        let (snp, indel) = (info(VariantType::Snp), info(VariantType::Indel));
        let duplicates = |resolved: &[PathBuf]| {
            let given_dirs = crate::testing::given_dirs(&links);
            let set = ModelSet {
                paths: &links,
                resolved,
                given_dirs: &given_dirs,
                infos: &[&snp, &indel],
            };
            checks(&check_model_set(&set, &SetInputs::default()))
        };
        // Read from different files; b retargeted to a's since.
        link(&links[0], &targets[0]);
        link(&links[1], &targets[1]);
        let read_from: Vec<PathBuf> =
            crate::model::ModelFile::resolve_all(&links).into_iter().map(|f| f.resolved).collect();
        link(&links[1], &targets[0]);
        assert_eq!(duplicates(&read_from), Vec::<Check>::new());
        // Read from one file; b retargeted to another since.
        let read_from: Vec<PathBuf> =
            crate::model::ModelFile::resolve_all(&links).into_iter().map(|f| f.resolved).collect();
        link(&links[1], &targets[1]);
        assert_eq!(duplicates(&read_from), [Check::DuplicateModel]);
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

    #[rstest::rstest]
    #[case::tiny(2)]
    #[case::past_legacy(601)]
    #[case::deep(1200)]
    fn maxreads_model_sources_match_legacy_or_reject_different_caps(#[case] cap: u64) {
        let mut snp = info(VariantType::Snp);
        let mut indel = info(VariantType::Indel);
        snp.feature_source = "alignments:mapq20,bq10,flank5".parse().unwrap();
        indel.feature_source = "alignments:maxreads600".parse().unwrap();
        assert!(check(&[snp.clone(), indel.clone()], &SetInputs::default()).is_empty());
        let deeper = format!("alignments:maxreads{cap}").parse().unwrap();
        indel.feature_source = deeper;
        let problems = check(&[snp.clone(), indel.clone()], &SetInputs::default());
        assert_eq!(checks(&problems), [Check::FeatureSourceDiffers]);
        assert_eq!(models_feature_source(&[&snp, &indel]), None);
        snp.feature_source = deeper;
        assert!(check(&[snp.clone(), indel.clone()], &SetInputs::default()).is_empty());
        assert_eq!(models_feature_source(&[&snp, &indel]), Some(deeper));
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
        // The checks of the set neither read nor look for the BEDs.
        let problems = check(&[snp.clone(), indel.clone()], &inputs);
        assert_eq!(checks(&problems), [Check::RegionRepeated, Check::RegionUnused]);
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
            "model a.winnowr.json (genotype snp) needs region BEDs for: lowmap; pass --regions \
             NAME=PATH, or put each beside the models as regions/NAME.bed.gz, \
             regions/NAME.bed.bgz, regions/NAME.bed (looked in: regions)"
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
        let paths = paths(infos.len());
        merged_reference_contigs(&ModelSet {
            paths: &paths,
            resolved: &paths,
            given_dirs: &crate::testing::given_dirs(&paths),
            infos: &refs,
        })
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

    /// A model directory: model files (`snp.winnowr.json`, which uses `segdups`) and a `regions`
    /// directory.
    struct Bundle {
        dir: TempDir,
        paths: Vec<PathBuf>,
    }

    impl Bundle {
        fn new() -> Self {
            let dir = TempDir::new().unwrap();
            std::fs::create_dir(dir.path().join("regions")).unwrap();
            let paths = vec![dir.path().join("snp.winnowr.json")];
            std::fs::write(&paths[0], "{}").unwrap();
            Bundle { dir, paths }
        }

        /// The path of `regions/<file>`.
        fn path(&self, file: &str) -> PathBuf {
            self.dir.path().join("regions").join(file)
        }

        /// Writes `regions/<file>`, of `lines`.
        fn write(&self, file: &str, lines: &str) -> PathBuf {
            let path = self.path(file);
            std::fs::write(&path, lines).unwrap();
            path
        }

        /// Resolves the region BEDs of a `run` of models of `infos` (as [`Bundle::paths`]).
        fn resolve(
            &self,
            infos: &[ModelInfo],
            given: &[NamedBed],
            run: RunRegions,
            allow: bool,
        ) -> (Vec<ResolvedRegion>, Vec<Problem>) {
            let refs: Vec<&ModelInfo> = infos.iter().collect();
            let set = ModelSet {
                paths: &self.paths,
                resolved: &self.paths,
                given_dirs: &crate::testing::given_dirs(&self.paths),
                infos: &refs,
            };
            let (resolved, problems) = resolve_run_regions(&set, given, run, allow);
            let resolved: Vec<_> = resolved.into_iter().map(|(region, _)| region).collect();
            // The report-only reader must preserve the scoring reader's metadata and diagnostics
            // across every bundled, explicit, corrupt, ambiguous and mismatched BED fixture.
            let (plan, mut report_problems) = RegionPlan::new(&set, given, run);
            let (reported, read_problems) = plan.check(allow);
            report_problems.extend(read_problems);
            report_problems.sort_by_key(|problem| problem.check);
            assert_eq!(reported, resolved);
            assert_eq!(report_problems, problems);
            (resolved, problems)
        }

        /// Resolves the region BEDs of `model`'s features, with `given`.
        fn features(
            &self,
            model: &ModelInfo,
            given: &[NamedBed],
            allow: bool,
        ) -> (Vec<ResolvedRegion>, Vec<Problem>) {
            let run = RunRegions::Features { policy: false };
            self.resolve(std::slice::from_ref(model), given, run, allow)
        }

        /// Resolves the not-difficult regions of a policy that reads them, for `model`.
        fn policy(&self, model: &ModelInfo) -> (Vec<ResolvedRegion>, Vec<Problem>) {
            self.resolve(std::slice::from_ref(model), &[], RunRegions::Policy(true), false)
        }
    }

    /// A genotype snp model using `segdups`, recording `digest` for it if given.
    fn segdups_model(digest: Option<RegionDigest>) -> ModelInfo {
        let mut model = info(VariantType::Snp);
        model.region_names = vec!["segdups".to_string()];
        model.region_digests = digest.into_iter().map(|d| ("segdups".to_string(), d)).collect();
        model
    }

    /// The digest of a BED of `lines` (`contig\tstart\tend` each), without writing it.
    fn digest_of(lines: &str) -> RegionDigest {
        let spans = lines.lines().map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            (fields[0], fields[1].parse().unwrap(), fields[2].parse().unwrap())
        });
        RegionSet::from_spans("x", spans).digest().unwrap()
    }

    const SEGDUPS: &str = "chr1\t0\t10\n";
    const OTHER: &str = "chr1\t0\t20\n";

    /// Neither consumer may wait for a producer when selected streams alias, or a bundled
    /// regular file is replaced after planning. The deadline bounds regressions that block.
    #[cfg(unix)]
    #[rstest]
    fn region_input_safety_rejects_without_a_producer(
        #[values("same", "symlink", "hardlink", "replacement", "replacement_connected")] case: &str,
        #[values(false, true)] report: bool,
    ) {
        let dir = TempDir::new().unwrap();
        let mut processes = crate::testing::TestProcesses::default();
        processes.spawn_test(
            "model_set::tests::region_input_safety_child",
            [
                ("WINNOWR_REGION_SAFETY_CASE", case.to_string()),
                ("WINNOWR_REGION_SAFETY_REPORT", report.to_string()),
            ],
            &dir.path().join("child.log"),
        );
        processes.wait_for_success(std::time::Duration::from_secs(3));
    }

    #[cfg(unix)]
    #[rstest]
    fn shared_streams_preserve_independent_inputs(
        #[values("same", "symlink", "hardlink")] alias: &str,
        #[values(false, true)] report: bool,
        #[values("valid", "malformed")] unrelated: &str,
    ) {
        let dir = TempDir::new().unwrap();
        let mut processes = crate::testing::TestProcesses::default();
        processes.spawn_test(
            "model_set::tests::region_input_safety_child",
            [
                ("WINNOWR_REGION_SAFETY_CASE", alias.to_string()),
                ("WINNOWR_REGION_SAFETY_REPORT", report.to_string()),
                ("WINNOWR_REGION_SAFETY_UNRELATED", unrelated.to_string()),
            ],
            &dir.path().join("child.log"),
        );
        processes.wait_for_success(std::time::Duration::from_secs(3));
    }

    /// Distinct explicit streams one producer opens in turn before writing either are all opened
    /// by the report-only check, as by the read: reading them in turn would wait forever. The
    /// producer and the checker run under one deadline.
    #[cfg(unix)]
    #[rstest]
    fn distinct_streams_are_read_concurrently(#[values(false, true)] report: bool) {
        let dir = TempDir::new().unwrap();
        let fifos = [dir.path().join("segdups.fifo"), dir.path().join("lowmap.fifo")];
        for fifo in &fifos {
            assert!(std::process::Command::new("mkfifo").arg(fifo).status().unwrap().success());
        }
        let mut processes = crate::testing::TestProcesses::default();
        processes.spawn_test(
            "model_set::tests::distinct_streams_child",
            [
                ("WINNOWR_DISTINCT_STREAMS", fifos[0].as_os_str().to_owned()),
                ("WINNOWR_DISTINCT_STREAMS_OTHER", fifos[1].as_os_str().to_owned()),
                ("WINNOWR_DISTINCT_STREAMS_REPORT", report.to_string().into()),
            ],
            &dir.path().join("child.log"),
        );
        // Opens both before writing either, as one producer of several streams may.
        let script = concat!(
            "exec 3>\"$1\" 4>\"$2\"; ",
            "printf 'chr1\\t0\\t10\\n' >&3; exec 3>&-; ",
            "printf 'chr1\\t0\\t20\\n' >&4"
        );
        processes
            .spawn(std::process::Command::new("sh").args(["-c", script, "producer"]).args(&fifos));
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    /// Child of [`distinct_streams_are_read_concurrently`].
    #[cfg(unix)]
    #[test]
    fn distinct_streams_child() {
        let Some(first) = std::env::var_os("WINNOWR_DISTINCT_STREAMS") else { return };
        let other = std::env::var_os("WINNOWR_DISTINCT_STREAMS_OTHER").unwrap();
        let report = std::env::var("WINNOWR_DISTINCT_STREAMS_REPORT").unwrap() == "true";
        let bundle = Bundle::new();
        let mut model = segdups_model(None);
        model.region_names.push("lowmap".to_string());
        let given = [
            NamedBed { name: "segdups".to_string(), path: first.into() },
            NamedBed { name: "lowmap".to_string(), path: other.into() },
        ];
        let given_dirs = crate::testing::given_dirs(&bundle.paths);
        let set = ModelSet {
            paths: &bundle.paths,
            resolved: &bundle.paths,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let (plan, problems) =
            RegionPlan::new(&set, &given, RunRegions::Features { policy: false });
        assert!(problems.is_empty(), "{problems:?}");
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let resolved: Vec<ResolvedRegion> = pool.install(|| {
            if report {
                plan.check(false).0
            } else {
                plan.read(false).0.into_iter().map(|(region, _)| region).collect()
            }
        });
        let bases: Vec<(&str, u64)> =
            resolved.iter().map(|region| (region.name.as_str(), region.digest.bases)).collect();
        assert_eq!(bases, [("segdups", 10), ("lowmap", 20)]);
        crate::testing::child_body_ran();
    }

    #[cfg(unix)]
    #[test]
    fn region_input_safety_child() {
        let Ok(case) = std::env::var("WINNOWR_REGION_SAFETY_CASE") else { return };
        let report = std::env::var("WINNOWR_REGION_SAFETY_REPORT").unwrap() == "true";
        let bundle = Bundle::new();
        let fifo = bundle.path("segdups.bed");
        let mut model = segdups_model(None);
        let replacement = case.starts_with("replacement");
        let mut given = if replacement {
            std::fs::write(&fifo, SEGDUPS).unwrap();
            Vec::new()
        } else {
            assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
            let alias = if case == "same" {
                fifo.clone()
            } else {
                let alias = bundle.path("alias.bed");
                if case == "symlink" {
                    std::os::unix::fs::symlink(&fifo, &alias).unwrap();
                } else {
                    std::fs::hard_link(&fifo, &alias).unwrap();
                }
                alias
            };
            model.region_names.push("lowmap".to_string());
            vec![
                NamedBed { name: "segdups".to_string(), path: fifo.clone() },
                NamedBed { name: "lowmap".to_string(), path: alias },
            ]
        };
        let unrelated = std::env::var("WINNOWR_REGION_SAFETY_UNRELATED").ok();
        let tandem = unrelated.as_deref().map(|kind| {
            let path =
                bundle.write("tandem.bed", if kind == "valid" { SEGDUPS } else { "chr1\t10\t5\n" });
            model.region_names.push("tandem".to_string());
            given.push(NamedBed { name: "tandem".to_string(), path: path.clone() });
            path
        });
        let set = ModelSet {
            paths: &bundle.paths,
            resolved: &bundle.paths,
            given_dirs: &crate::testing::given_dirs(&bundle.paths),
            infos: &[&model],
        };
        let (plan, problems) =
            RegionPlan::new(&set, &given, RunRegions::Features { policy: false });
        assert!(problems.is_empty(), "{problems:?}");
        if replacement {
            std::fs::remove_file(&fifo).unwrap();
            assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        }
        // A connected but idle producer makes a premature prefix read return WouldBlock (or
        // hang if the open was blocking), rather than letting it see an immediate FIFO EOF.
        let _connected = (case == "replacement_connected")
            .then(|| std::fs::OpenOptions::new().read(true).write(true).open(&fifo).unwrap());
        let (resolved, problems) = if report {
            plan.check(false)
        } else {
            let (resolved, problems) = plan.read(false);
            (resolved.into_iter().map(|(region, _)| region).collect(), problems)
        };
        if unrelated.as_deref() == Some("valid") {
            assert_eq!(resolved.len(), 1, "independent BED metadata was lost: {problems:?}");
            assert_eq!(resolved[0].name, "tandem");
            assert_eq!(resolved[0].path, tandem.as_ref().unwrap().as_path());
            assert_eq!(resolved[0].digest.bases, 10);
            assert_eq!(resolved[0].input_identities.len(), 1);
        } else {
            assert!(resolved.is_empty());
        }
        let unreadable: Vec<_> =
            problems.iter().filter(|problem| problem.check == Check::RegionUnreadable).collect();
        assert_eq!(
            unreadable.len(),
            if unrelated.as_deref() == Some("malformed") { 2 } else { 1 },
            "{problems:?}"
        );
        if unrelated.as_deref() == Some("malformed") {
            let path = tandem.as_ref().unwrap().display().to_string();
            assert!(
                unreadable
                    .iter()
                    .any(|problem| problem.message.contains("tandem")
                        && problem.message.contains(&path)),
                "{problems:?}"
            );
        }
        let message = &unreadable[0].message;
        assert!(
            message.contains("segdups") && message.contains(&fifo.display().to_string()),
            "{message}"
        );
        if replacement {
            assert!(message.contains("regular"), "{message}");
        } else {
            assert!(message.contains("lowmap") && message.contains("nonregular"), "{message}");
        }
        crate::testing::child_body_ran();
    }

    #[cfg(unix)]
    #[rstest]
    fn distinct_names_can_share_a_regular_bed(
        #[values("same", "symlink", "hardlink")] alias: &str,
        #[values(false, true)] compressed: bool,
    ) {
        let bundle = Bundle::new();
        let path = bundle.write("input.bed", SEGDUPS);
        if compressed {
            crate::testing::write_gzip(&path, SEGDUPS.as_bytes());
        }
        let other = if alias == "same" {
            path.clone()
        } else {
            let other = bundle.path("alias.bed");
            if alias == "symlink" {
                std::os::unix::fs::symlink(&path, &other).unwrap();
            } else {
                std::fs::hard_link(&path, &other).unwrap();
            }
            other
        };
        let mut model = segdups_model(None);
        model.region_names.push("lowmap".to_string());
        let given = [
            NamedBed { name: "segdups".to_string(), path },
            NamedBed { name: "lowmap".to_string(), path: other },
        ];
        let (resolved, problems) = bundle.features(&model, &given, false);
        assert_eq!(resolved.len(), 2);
        assert_eq!(checks(&problems), [Check::RegionUnverified, Check::RegionUnverified]);
        assert!(resolved.iter().all(|region| region.digest.bases == 10));
    }

    #[rstest]
    fn overflowed_bed_digest_is_an_unreadable_region(#[values(false, true)] report: bool) {
        let bundle = Bundle::new();
        let path = bundle
            .write("segdups.bed", &format!("chr1\t0\t{0}\nchr2\t0\t{0}\nchr3\t0\t{0}\n", i64::MAX));
        let model = segdups_model(None);
        let set = ModelSet {
            paths: &bundle.paths,
            resolved: &bundle.paths,
            given_dirs: &crate::testing::given_dirs(&bundle.paths),
            infos: &[&model],
        };
        let (plan, _) = RegionPlan::new(&set, &[], RunRegions::Features { policy: false });
        let (resolved, problems) = if report {
            plan.check(false)
        } else {
            let (resolved, problems) = plan.read(false);
            (resolved.into_iter().map(|(region, _)| region).collect(), problems)
        };
        assert!(resolved.is_empty());
        assert_eq!(checks(&problems), [Check::RegionUnreadable]);
        assert!(problems[0].message.contains(&path.display().to_string()));
        assert!(problems[0].message.contains("u64"));
    }

    /// The BEDs are found beside the models, as `.bed.gz` or `.bed`, and checked against the
    /// models' digests; a given BED is used in place of one found.
    #[test]
    fn region_beds_are_found_beside_the_models_and_checked() {
        let bundle = Bundle::new();
        let found = bundle.write("segdups.bed", SEGDUPS);
        let model = segdups_model(Some(digest_of(SEGDUPS)));
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!((resolved[0].path.clone(), resolved[0].origin()), (found, RegionFrom::Bundled));
        assert_eq!(resolved[0].verified(), Some(true));

        // Given: used in place of the one beside the models, and checked the same way.
        let dir = TempDir::new().unwrap();
        let given = [bed(&dir, "segdups", OTHER)];
        let (resolved, problems) = bundle.features(&model, &given, false);
        assert_eq!((resolved[0].origin(), resolved[0].given), (RegionFrom::Given, Some(0)));
        assert_eq!(checks(&problems), [Check::RegionChecksum]);
        assert_eq!(problems[0].severity, Severity::Error);
        let message = &problems[0].message;
        assert!(message.starts_with("the segdups region BED "), "{message}");
        assert!(message.contains("(given) is not the one the models were trained with: it has 1 intervals covering 20 bases"), "{message}");
        let (_, problems) = bundle.features(&model, &given, true);
        assert_eq!(severities(&problems), [Severity::Warning]);

        // Recorded by no model: used unchecked, with a warning.
        let (resolved, problems) = bundle.features(&segdups_model(None), &[], false);
        assert_eq!(resolved[0].verified(), None);
        assert_eq!(checks(&problems), [Check::RegionUnverified]);
    }

    /// A gzipped BED is found too, and several BEDs of a name are one if their intervals are.
    #[test]
    fn several_region_beds_of_a_name() {
        let bundle = Bundle::new();
        let gz = bundle.path("segdups.bed.gz");
        crate::testing::write_gzip(&gz, SEGDUPS.as_bytes());
        let model = segdups_model(Some(digest_of(SEGDUPS)));
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(resolved[0].path, gz);
        bundle.write("segdups.bed", SEGDUPS);
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(resolved[0].path, gz);
        bundle.write("segdups.bed", OTHER);
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert!(resolved.is_empty());
        assert_eq!(checks(&problems), [Check::RegionAmbiguous]);
        // A .bed.bgz is found too, and read as BGZF.
        std::fs::remove_file(&gz).unwrap();
        std::fs::remove_file(bundle.path("segdups.bed")).unwrap();
        let bgz = bundle.path("segdups.bed.bgz");
        crate::testing::write_bgzf(&bgz, SEGDUPS.as_bytes());
        assert!(crate::testing::is_bgzf(&bgz), "the fixture is not BGZF");
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(resolved[0].path, bgz);
    }

    #[rstest]
    #[case::invalid_first("chr2\t0\t10\n", SEGDUPS, &[Check::RegionContigs, Check::RegionAmbiguous])]
    #[case::invalid_second(SEGDUPS, "chr2\t0\t10\n", &[Check::RegionContigs, Check::RegionAmbiguous])]
    #[case::both_invalid("chr2\t0\t10\n", "chr3\t0\t10\n", &[Check::RegionContigs, Check::RegionContigs, Check::RegionAmbiguous])]
    #[case::equivalent_invalid("chr2\t0\t10\n", "chr2\t0\t10\n", &[Check::RegionContigs])]
    #[case::equivalent_valid(SEGDUPS, SEGDUPS, &[])]
    #[case::differing_valid(SEGDUPS, OTHER, &[Check::RegionAmbiguous])]
    fn bundled_candidate_contig_errors_survive_ambiguity(
        #[case] first: &str,
        #[case] second: &str,
        #[case] expected: &[Check],
        #[values(false, true)] policy: bool,
        #[values(false, true)] allow: bool,
    ) {
        let bundle = Bundle::new();
        let compressed = bundle.path(&format!("{NOT_DIFFICULT_REGIONS}.bed.gz"));
        crate::testing::write_gzip(&compressed, first.as_bytes());
        let plain = bundle.write(&format!("{NOT_DIFFICULT_REGIONS}.bed"), second);
        let mut model = info(VariantType::Indel);
        model.region_names = vec![NOT_DIFFICULT_REGIONS.to_string()];
        model.region_digests =
            [(NOT_DIFFICULT_REGIONS.to_string(), digest_of(first))].into_iter().collect();
        model.policy_region_digests = model.region_digests.clone();
        let run = RunRegions::Features { policy };
        // Bundle::resolve asserts exact read/check metadata and diagnostic parity.
        let (resolved, problems) = bundle.resolve(&[model], &[], run, allow);
        assert_eq!(checks(&problems), expected);
        assert!(problems.iter().all(|problem| problem.severity == Severity::Error));
        if expected.contains(&Check::RegionAmbiguous) {
            assert!(resolved.is_empty());
            let ambiguity = problems.iter().find(|p| p.check == Check::RegionAmbiguous).unwrap();
            for path in [&compressed, &plain] {
                assert!(ambiguity.message.contains(&path.display().to_string()));
            }
        } else {
            assert_eq!(resolved.len(), 1);
            assert_eq!(resolved[0].verified(), Some(true));
        }
        let contig_errors: Vec<_> =
            problems.iter().filter(|p| p.check == Check::RegionContigs).collect();
        if first.starts_with("chr2") {
            assert!(contig_errors[0].message.contains(&compressed.display().to_string()));
        } else if !contig_errors.is_empty() {
            assert!(contig_errors[0].message.contains(&plain.display().to_string()));
        }
        if contig_errors.len() == 2 {
            assert!(contig_errors[1].message.contains(&plain.display().to_string()));
        }
    }

    #[rstest]
    #[case::features(RunRegions::Features { policy: false })]
    #[case::features_and_policy(RunRegions::Features { policy: true })]
    #[case::policy(RunRegions::Policy(true))]
    #[case::no_policy(RunRegions::Policy(false))]
    fn prospective_region_paths_cover_only_automatic_roles(
        #[case] run: RunRegions,
        #[values(false, true)] explicit: bool,
    ) {
        let bundle = Bundle::new();
        let compressed = bundle.path("segdups.bed.gz");
        crate::testing::write_gzip(&compressed, SEGDUPS.as_bytes());
        let model = segdups_model(None);
        let given_dirs = crate::testing::given_dirs(&bundle.paths);
        let set = ModelSet {
            paths: &bundle.paths,
            resolved: &bundle.paths,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let given = [NamedBed {
            name: "segdups".to_string(),
            path: bundle.dir.path().join("explicit.bed"),
        }];
        let (plan, _) = RegionPlan::new(&set, if explicit { &given } else { &[] }, run);
        let mut expected = Vec::new();
        if matches!(run, RunRegions::Features { .. }) && !explicit {
            expected.extend(
                ["segdups.bed.gz", "segdups.bed.bgz", "segdups.bed"].map(|file| bundle.path(file)),
            );
        }
        if matches!(run, RunRegions::Features { policy: true } | RunRegions::Policy(true)) {
            expected.extend(
                [".bed.gz", ".bed.bgz", ".bed"]
                    .map(|suffix| bundle.path(&format!("{NOT_DIFFICULT_REGIONS}{suffix}"))),
            );
        }
        assert_eq!(
            plan.prospective_paths().collect::<Vec<_>>(),
            expected.iter().map(PathBuf::as_path).collect::<Vec<_>>()
        );
        let actual: Vec<_> = plan.paths().collect();
        if matches!(run, RunRegions::Features { .. }) {
            if explicit {
                assert_eq!(actual, [given[0].path.as_path()]);
            } else {
                assert!(actual.contains(&compressed.as_path()));
                assert!(!actual.contains(&bundle.path("segdups.bed").as_path()));
            }
        } else {
            assert!(actual.is_empty());
        }
    }

    #[test]
    fn selected_region_indices_include_failed_reads_but_not_duplicate_names() {
        let bundle = Bundle::new();
        let model = segdups_model(None);
        let given_dirs = crate::testing::given_dirs(&bundle.paths);
        let set = ModelSet {
            paths: &bundle.paths,
            resolved: &bundle.paths,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let given = [
            absent(&bundle.dir, "unused"),
            absent(&bundle.dir, "segdups"),
            absent(&bundle.dir, "segdups"),
        ];
        let (plan, _) = RegionPlan::new(&set, &given, RunRegions::Features { policy: false });
        assert_eq!(plan.given_indices().collect::<Vec<_>>(), [1]);
        let (read, problems) = plan.read(false);
        assert!(read.is_empty());
        assert_eq!(checks(&problems), [Check::RegionUnreadable]);
    }

    #[test]
    fn every_equivalent_candidate_retains_its_opened_identity() {
        let bundle = Bundle::new();
        let gzip = bundle.path("segdups.bed.gz");
        let bgzf = bundle.path("segdups.bed.bgz");
        let plain = bundle.write("segdups.bed", SEGDUPS);
        crate::testing::write_gzip(&gzip, SEGDUPS.as_bytes());
        crate::testing::write_bgzf(&bgzf, SEGDUPS.as_bytes());
        let model = segdups_model(None);
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert_eq!(checks(&problems), [Check::RegionUnverified]);
        let identities = &resolved[0].input_identities;
        assert_eq!(identities.len(), 3);
        for path in [&gzip, &bgzf, &plain] {
            assert!(
                identities.iter().any(|identity| identity.matches_path(path)),
                "identity missing for {}",
                path.display()
            );
        }
    }

    /// Idle FIFOs must never be opened by automatic discovery, even through a symlink. Keep
    /// the reader in a bounded child so a regression fails rather than hanging the test suite.
    #[cfg(unix)]
    #[rstest]
    fn bundled_symlinks_to_nonregular_targets_are_ignored(
        #[values("fifo", "directory")] target: &str,
        #[values(false, true)] policy: bool,
        #[values(false, true)] usable: bool,
    ) {
        let dir = TempDir::new().unwrap();
        let mut processes = crate::testing::TestProcesses::default();
        processes.spawn_test(
            "model_set::tests::bundled_nonregular_symlink_child",
            [
                ("WINNOWR_BUNDLED_NONREGULAR", target.to_string()),
                ("WINNOWR_BUNDLED_POLICY", policy.to_string()),
                ("WINNOWR_BUNDLED_USABLE", usable.to_string()),
            ],
            &dir.path().join("child.log"),
        );
        processes.wait_for_success(std::time::Duration::from_secs(5));
    }

    #[cfg(unix)]
    #[test]
    fn bundled_nonregular_symlink_child() {
        let Some(target_kind) = std::env::var_os("WINNOWR_BUNDLED_NONREGULAR") else { return };
        let policy = std::env::var("WINNOWR_BUNDLED_POLICY").unwrap() == "true";
        let usable = std::env::var("WINNOWR_BUNDLED_USABLE").unwrap() == "true";
        let bundle = Bundle::new();
        let name = if policy { NOT_DIFFICULT_REGIONS } else { "segdups" };
        let target = bundle.dir.path().join("nonregular");
        if target_kind == "fifo" {
            assert!(std::process::Command::new("mkfifo").arg(&target).status().unwrap().success());
        } else {
            std::fs::create_dir(&target).unwrap();
        }
        std::os::unix::fs::symlink(&target, bundle.path(&format!("{name}.bed.gz"))).unwrap();
        if usable {
            bundle.write(&format!("{name}.bed"), SEGDUPS);
        }
        let (resolved, problems) = if policy {
            bundle.policy(&info(VariantType::Indel))
        } else {
            bundle.features(&segdups_model(None), &[], false)
        };
        let expected = match (policy, usable) {
            (true, _) => Vec::new(),
            (false, true) => vec![Check::RegionUnverified],
            (false, false) => vec![Check::RegionMissing],
        };
        assert_eq!(checks(&problems), expected, "{problems:?}");
        assert_eq!(resolved.len(), usize::from(usable));
        crate::testing::child_body_ran();
    }

    #[cfg(unix)]
    #[test]
    fn bundled_bed_deduplication_pins_the_read_target() {
        let bundle = Bundle::new();
        let target = bundle.dir.path().join("original.bed");
        let other = bundle.dir.path().join("other.bed");
        std::fs::write(&target, SEGDUPS).unwrap();
        std::fs::write(&other, OTHER).unwrap();
        let alias = bundle.path("segdups.bed.gz");
        std::os::unix::fs::symlink(&target, &alias).unwrap();
        std::os::unix::fs::symlink(&target, bundle.path("segdups.bed")).unwrap();
        let model = segdups_model(None);
        let given_dirs = crate::testing::given_dirs(&bundle.paths);
        let set = ModelSet {
            paths: &bundle.paths,
            resolved: &bundle.paths,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let (plan, problems) = RegionPlan::new(&set, &[], RunRegions::Features { policy: false });
        assert!(problems.is_empty());
        std::fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink(&other, &alias).unwrap();
        let (resolved, problems) = plan.read(false);
        assert_eq!(checks(&problems), [Check::RegionUnverified]);
        assert_eq!(resolved[0].0.digest, digest_of(SEGDUPS));
        assert_eq!(resolved[0].0.path, alias, "diagnostics should retain the discovery alias");
    }

    /// A bundled link dangling when found is unreadable, even if retargeted to a regular BED
    /// before the read: the plan never follows an alias it could not resolve.
    #[cfg(unix)]
    #[rstest]
    fn a_dangling_bundled_bed_is_not_followed_after_retargeting(
        #[values(false, true)] check: bool,
    ) {
        let bundle = Bundle::new();
        let other = bundle.dir.path().join("other.bed");
        std::fs::write(&other, OTHER).unwrap();
        let alias = bundle.path("segdups.bed.gz");
        std::os::unix::fs::symlink(bundle.dir.path().join("absent.bed"), &alias).unwrap();
        let model = segdups_model(None);
        let given_dirs = crate::testing::given_dirs(&bundle.paths);
        let set = ModelSet {
            paths: &bundle.paths,
            resolved: &bundle.paths,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let (plan, problems) = RegionPlan::new(&set, &[], RunRegions::Features { policy: false });
        assert!(problems.is_empty(), "{problems:?}");
        std::fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink(&other, &alias).unwrap();
        let (resolved, problems) = if check {
            plan.check(false)
        } else {
            let (resolved, problems) = plan.read(false);
            (resolved.into_iter().map(|(region, _)| region).collect(), problems)
        };
        assert!(resolved.is_empty(), "{resolved:?}");
        assert_eq!(checks(&problems), [Check::RegionUnreadable], "{problems:?}");
        assert!(problems[0].message.contains("cannot resolve it"), "{}", problems[0].message);
    }

    /// The BEDs are looked for beside the file a model's symlink resolves to.
    #[cfg(unix)]
    #[test]
    fn region_beds_are_found_beside_a_symlinks_target() {
        let bundle = Bundle::new();
        let found = bundle.write("segdups.bed", SEGDUPS);
        let staged = TempDir::new().unwrap();
        let link = staged.path().join("snp.winnowr.json");
        std::os::unix::fs::symlink(&bundle.paths[0], &link).unwrap();
        let paths = [link];
        let file = crate::model::ModelFile::resolve(&paths[0]);
        let (resolved, given_dirs) = ([file.resolved], [file.given_dir]);
        let model = segdups_model(None);
        let set = ModelSet {
            paths: &paths,
            resolved: &resolved,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let dirs = bundled_dirs(&set);
        let found_beds = bundled_beds(&bundled_paths(&dirs, "segdups"));
        assert_eq!(found_beds.len(), 1);
        assert_eq!(found_beds[0].path, found.canonicalize().unwrap());
        // The staged directory, resolved (e.g. macOS's /var is /private/var).
        assert_eq!(dirs[0], staged.path().canonicalize().unwrap().join("regions"));
    }

    /// The BEDs are looked for beside the file a model was read from: a staged symlink retargeted
    /// to another bundle after the model was read does not pair it with that bundle's BEDs.
    #[cfg(unix)]
    #[test]
    fn region_beds_are_found_beside_the_file_the_model_was_read_from() {
        let (read_from, retargeted) = (Bundle::new(), Bundle::new());
        let found = read_from.write("segdups.bed", SEGDUPS);
        retargeted.write("segdups.bed", OTHER);
        let staged = TempDir::new().unwrap();
        let link = staged.path().join("snp.winnowr.json");
        std::os::unix::fs::symlink(&read_from.paths[0], &link).unwrap();
        let files = crate::model::ModelFile::resolve_all(std::slice::from_ref(&link));
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(&retargeted.paths[0], &link).unwrap();
        let paths = [link];
        let (resolved, given_dirs) = ([files[0].resolved.clone()], [files[0].given_dir.clone()]);
        let model = segdups_model(Some(digest_of(SEGDUPS)));
        let set = ModelSet {
            paths: &paths,
            resolved: &resolved,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let run = RunRegions::Features { policy: false };
        let (regions, problems) = resolve_run_regions(&set, &[], run, false);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(regions[0].0.path.canonicalize().unwrap(), found.canonicalize().unwrap());
    }

    /// The BEDs are looked for in the directory a model's path was in when it was read: a staged
    /// directory symlink retargeted to another bundle after the model was read does not pair it
    /// with that bundle's BEDs.
    #[cfg(unix)]
    #[test]
    fn region_beds_are_found_in_the_directory_the_model_was_read_from() {
        let (read_from, retargeted) = (Bundle::new(), Bundle::new());
        let found = read_from.write("segdups.bed", SEGDUPS);
        retargeted.write("segdups.bed", OTHER);
        let staged = TempDir::new().unwrap();
        let link = staged.path().join("bundle");
        std::os::unix::fs::symlink(read_from.dir.path(), &link).unwrap();
        let paths = [link.join("snp.winnowr.json")];
        let files = crate::model::ModelFile::resolve_all(&paths);
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(retargeted.dir.path(), &link).unwrap();
        let resolved: Vec<PathBuf> = files.iter().map(|file| file.resolved.clone()).collect();
        let given_dirs: Vec<PathBuf> = files.iter().map(|file| file.given_dir.clone()).collect();
        let model = segdups_model(Some(digest_of(SEGDUPS)));
        let set = ModelSet {
            paths: &paths,
            resolved: &resolved,
            given_dirs: &given_dirs,
            infos: &[&model],
        };
        let run = RunRegions::Features { policy: false };
        let (regions, problems) = resolve_run_regions(&set, &[], run, false);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(regions[0].0.path.canonicalize().unwrap(), found.canonicalize().unwrap());
    }

    /// The digests a BED is checked against are those of the roles it serves in the run: a
    /// model's feature always, and the policy's only when the policy reads it. One BED serving
    /// both must be the one both record.
    #[rstest]
    // A feature and the policy, recording different BEDs: no BED is both.
    #[case::feature_and_policy_differ(Some(SEGDUPS), Some(OTHER), true, SEGDUPS, &[Check::RegionDigestsDiffer])]
    #[case::feature_and_policy_differ_policy_bed(Some(SEGDUPS), Some(OTHER), true, OTHER, &[Check::RegionDigestsDiffer])]
    #[case::feature_and_policy_agree(Some(SEGDUPS), Some(SEGDUPS), true, SEGDUPS, &[])]
    #[case::feature_and_policy_agree_other_bed(Some(SEGDUPS), Some(SEGDUPS), true, OTHER, &[Check::RegionChecksum])]
    // The policy does not read it: only the feature's digest counts.
    #[case::inactive_policy(Some(SEGDUPS), Some(OTHER), false, SEGDUPS, &[])]
    #[case::inactive_policy_other_bed(Some(SEGDUPS), Some(OTHER), false, OTHER, &[Check::RegionChecksum])]
    // The policy alone records one.
    #[case::policy_only(None, Some(OTHER), true, OTHER, &[])]
    #[case::policy_only_other_bed(None, Some(OTHER), true, SEGDUPS, &[Check::RegionChecksum])]
    fn region_digests_are_of_the_roles_a_bed_serves(
        #[case] feature: Option<&str>,
        #[case] policy_digest: Option<&str>,
        #[case] policy: bool,
        #[case] bed: &str,
        #[case] expected: &[Check],
    ) {
        let bundle = Bundle::new();
        bundle.write(&format!("{NOT_DIFFICULT_REGIONS}.bed"), bed);
        let mut model = info(VariantType::Indel);
        if let Some(lines) = feature {
            model.region_names = vec![NOT_DIFFICULT_REGIONS.to_string()];
            model.region_digests =
                BTreeMap::from([(NOT_DIFFICULT_REGIONS.into(), digest_of(lines))]);
        }
        model.policy_region_digests = policy_digest
            .map(|lines| BTreeMap::from([(NOT_DIFFICULT_REGIONS.to_string(), digest_of(lines))]))
            .unwrap_or_default();
        let run = RunRegions::Features { policy };
        let (_, problems) = bundle.resolve(std::slice::from_ref(&model), &[], run, false);
        assert_eq!(checks(&problems), expected, "{problems:?}");
    }

    /// Table-based commands already checked the feature BED via the table's metadata. Their
    /// live policy BED must not be compared with that independent feature digest.
    #[rstest]
    #[case::strict(false)]
    #[case::allow_mismatch(true)]
    fn a_policy_only_bed_ignores_feature_digests(#[case] allow: bool) {
        let bundle = Bundle::new();
        bundle.write(&format!("{NOT_DIFFICULT_REGIONS}.bed"), OTHER);
        let mut model = info(VariantType::Indel);
        model.region_names = vec![NOT_DIFFICULT_REGIONS.to_string()];
        model.region_digests =
            BTreeMap::from([(NOT_DIFFICULT_REGIONS.to_string(), digest_of(SEGDUPS))]);
        model.policy_region_digests =
            BTreeMap::from([(NOT_DIFFICULT_REGIONS.to_string(), digest_of(OTHER))]);
        let (resolved, problems) =
            bundle.resolve(std::slice::from_ref(&model), &[], RunRegions::Policy(true), allow);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(resolved[0].verified(), Some(true));
        // A policy-only legacy model with no policy digest must not check a feature digest.
        model.policy_region_digests.clear();
        let (resolved, problems) =
            bundle.resolve(std::slice::from_ref(&model), &[], RunRegions::Policy(true), allow);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(resolved[0].verified(), None);
    }

    /// A policy digest of another model does not constrain a BED that serves only as a feature:
    /// the policy does not read it.
    #[test]
    fn an_inactive_policy_does_not_constrain_a_feature_bed() {
        let bundle = Bundle::new();
        bundle.write(&format!("{NOT_DIFFICULT_REGIONS}.bed"), SEGDUPS);
        let mut snp = info(VariantType::Snp);
        snp.region_names = vec![NOT_DIFFICULT_REGIONS.to_string()];
        snp.region_digests = BTreeMap::from([(NOT_DIFFICULT_REGIONS.into(), digest_of(SEGDUPS))]);
        let mut indel = info(VariantType::Indel);
        indel.policy_region_digests =
            BTreeMap::from([(NOT_DIFFICULT_REGIONS.to_string(), digest_of(OTHER))]);
        let mut paths = bundle.paths.clone();
        paths.push(bundle.dir.path().join("indel.winnowr.json"));
        let set = ModelSet {
            paths: &paths,
            resolved: &paths,
            given_dirs: &crate::testing::given_dirs(&paths),
            infos: &[&snp, &indel],
        };
        let run = RunRegions::Features { policy: false };
        let (resolved, problems) = resolve_run_regions(&set, &[], run, false);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(resolved[0].0.verified(), Some(true));
        // With the policy reading it, the two models' digests conflict.
        let run = RunRegions::Features { policy: true };
        let (_, problems) = resolve_run_regions(&set, &[], run, false);
        assert_eq!(checks(&problems), [Check::RegionDigestsDiffer]);
        let message = &problems[0].message;
        assert!(message.contains("indel.winnowr.json (policy) "), "{message}");
    }

    /// A region a model uses that is neither given nor found is missing; models that record
    /// different digests of a region cannot be scored together.
    #[test]
    fn missing_regions_and_models_that_record_different_beds() {
        let bundle = Bundle::new();
        let model = segdups_model(None);
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert!(resolved.is_empty());
        assert_eq!(checks(&problems), [Check::RegionMissing]);
        assert_eq!(problems[0].model.as_deref(), Some(bundle.paths[0].as_path()));
        assert!(problems[0].message.contains("(looked in: "), "{}", problems[0].message);
        // The policy's regions are not missing: score runs without them.
        let (resolved, problems) = bundle.policy(&model);
        assert!(resolved.is_empty() && problems.is_empty(), "{problems:?}");
        bundle.write("segdups.bed", SEGDUPS);
        assert_eq!(checks(&bundle.features(&model, &[], false).1), [Check::RegionUnverified]);

        let mut paths = bundle.paths.clone();
        paths.push(bundle.dir.path().join("indel.winnowr.json"));
        let mut indel = segdups_model(Some(digest_of(OTHER)));
        indel.variant_type = VariantType::Indel;
        let infos = [segdups_model(Some(digest_of(SEGDUPS))), indel];
        let refs: Vec<&ModelInfo> = infos.iter().collect();
        let set = ModelSet {
            paths: &paths,
            resolved: &paths,
            given_dirs: &crate::testing::given_dirs(&paths),
            infos: &refs,
        };
        let run = RunRegions::Features { policy: false };
        let (_, problems) = resolve_run_regions(&set, &[], run, false);
        assert_eq!(checks(&problems), [Check::RegionDigestsDiffer]);
        assert_eq!(problems[0].severity, Severity::Error);
        assert_eq!(problems[0].model.as_deref(), Some(paths[1].as_path()));
        // --allow-region-mismatch does not cover it.
        let (_, problems) = resolve_run_regions(&set, &[], run, true);
        assert_eq!(severities(&problems), [Severity::Error]);
    }

    /// A model's region name that is not a file name is not looked for beside the models.
    #[test]
    fn region_names_that_are_not_file_names_are_not_looked_for() {
        let bundle = Bundle::new();
        std::fs::write(bundle.dir.path().join("escape.bed"), SEGDUPS).unwrap();
        for name in ["../escape", "", ".", "..", "a\\b"] {
            assert!(!is_file_name(name), "{name:?}");
            let model = segdups_model(None);
            let set = ModelSet {
                paths: &bundle.paths,
                resolved: &bundle.paths,
                given_dirs: &crate::testing::given_dirs(&bundle.paths),
                infos: &[&model],
            };
            assert!(bundled_beds(&bundled_paths(&bundled_dirs(&set), name)).is_empty(), "{name:?}");
        }
        assert!(is_file_name("segdups"));
        let mut model = info(VariantType::Snp);
        model.region_names = vec!["../escape".to_string()];
        let (_, problems) = bundle.features(&model, &[], false);
        assert_eq!(checks(&problems), [Check::RegionMissing]);
        let message = &problems[0].message;
        assert!(
            message.ends_with(
                "; ../escape cannot be looked for beside the models, not being file names"
            ),
            "{message}"
        );
    }

    /// A BED that cannot be read (given and missing, or found and corrupt) is an error, and is
    /// not used; one with no interval on the models' contigs is an error.
    #[test]
    fn unreadable_region_beds_and_beds_of_other_contigs() {
        let bundle = Bundle::new();
        let model = segdups_model(None);
        let dir = TempDir::new().unwrap();
        let (resolved, problems) = bundle.features(&model, &[absent(&dir, "segdups")], false);
        assert!(resolved.is_empty());
        assert_eq!(checks(&problems), [Check::RegionUnreadable]);
        assert_eq!(problems[0].severity, Severity::Error);
        std::fs::write(bundle.path("segdups.bed.gz"), [0x1f, 0x8b, 0, 1, 2]).unwrap();
        let (resolved, problems) = bundle.features(&model, &[], false);
        assert!(resolved.is_empty());
        assert_eq!(checks(&problems), [Check::RegionUnreadable]);
        std::fs::remove_file(bundle.path("segdups.bed.gz")).unwrap();
        bundle.write("segdups.bed", "chr2\t0\t10\n");
        let (_, problems) = bundle.features(&model, &[], false);
        assert_eq!(checks(&problems), [Check::RegionContigs, Check::RegionUnverified]);
    }

    /// The policy's regions are checked against the digests models record for them, and used
    /// without a problem when none does; absent, they are left out.
    #[test]
    fn the_policy_regions_are_found_and_checked() {
        let bundle = Bundle::new();
        let mut indel = info(VariantType::Indel);
        let (resolved, problems) = bundle.policy(&indel);
        assert!(resolved.is_empty() && problems.is_empty(), "{problems:?}");
        bundle.write(&format!("{NOT_DIFFICULT_REGIONS}.bed"), SEGDUPS);
        let (resolved, problems) = bundle.policy(&indel);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(resolved[0].verified(), None);
        indel.policy_region_digests =
            [(NOT_DIFFICULT_REGIONS.to_string(), digest_of(SEGDUPS))].into_iter().collect();
        assert_eq!(bundle.policy(&indel).0[0].verified(), Some(true));
        indel.policy_region_digests =
            [(NOT_DIFFICULT_REGIONS.to_string(), digest_of(OTHER))].into_iter().collect();
        assert_eq!(checks(&bundle.policy(&indel).1), [Check::RegionChecksum]);
        // A policy that does not read them reads nothing.
        let run = RunRegions::Policy(false);
        let (resolved, problems) = bundle.resolve(&[indel], &[], run, false);
        assert!(resolved.is_empty() && problems.is_empty(), "{problems:?}");
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
