//! `winnowr check-models`: checks a set of models before a run, without loading their boosters,
//! and reports what they need and every problem found (see [`crate::model_set`]), as text or as
//! a JSON summary of schema version [`SCHEMA_VERSION`].

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::Serialize;

use crate::alignment_provenance::{
    AlignmentProvenance, AlignmentTags, ProvenanceFrom, ReadsAlignment, UnknownReason,
    resolve_reads,
};
use crate::feature_source::FeatureSource;
use crate::io::same_input_path;
use crate::model::{ModelFile, ModelInfo, ReferenceContigs, UnsupportedFormatVersion};
use crate::model_evidence::ModelEvidence;
use crate::model_set::{
    Check, ModelSet, Problem, ReadsCheck, RegionPlan, ResolvedRegion, RunRegions, SetInputs,
    Severity, bundled_bed_files, check_model_set, check_region_beds, evidence_settings,
    has_indel_genotype, lossy_path, merged_reference_contigs, models_alignment,
    models_feature_source, recorded_digests,
};
use crate::pileup::PileupOptions;
use crate::policy::NOT_DIFFICULT_REGIONS;
use crate::regions::{NamedBed, RegionDigest};
use crate::strlen::StrlenOptions;

/// The version of the JSON summary's schema. It changes only when a key is removed or renamed,
/// or changes meaning: new keys, check ids and values do not change it, and callers should ignore
/// those they do not know.
pub const SCHEMA_VERSION: u32 = 1;

/// What `check-models` is given.
#[derive(Debug, Clone, Copy)]
pub struct CheckInputs<'a> {
    /// The model files.
    pub models: &'a [PathBuf],
    /// The region BEDs, by name.
    pub regions: &'a [NamedBed],
    /// A FASTA with its `.fai` beside it, or the `.fai`: its index's contig lengths.
    pub reference: Option<&'a Path>,
    /// The VCF to be scored: its header's contig lengths.
    pub vcf: Option<&'a Path>,
    /// The BAM or CRAM of the reads to be scored: its header's alignment provenance.
    pub alignments: Option<&'a Path>,
    /// `--alignment-tags`: the reads' alignment provenance, or tags over the header's.
    pub alignment_tags: Option<&'a AlignmentTags>,
    /// Whether alignment provenance problems are errors.
    pub strict_provenance: bool,
    /// Whether a region BED whose digest is not the models' is a warning, not an error.
    pub allow_region_mismatch: bool,
}

/// The JSON summary of a model set (schema version [`SCHEMA_VERSION`]).
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// The winnowr version that checked the models.
    pub winnowr_version: String,
    /// True if no problem is an error.
    pub ok: bool,
    /// Whether the models' boosters were checked: never, as only their information is read.
    pub booster_checked: bool,
    /// Each model, in the order given.
    pub models: Vec<ModelSummary>,
    /// What the models share.
    pub set: SetSummary,
    /// The inputs the models were checked against.
    pub inputs: InputsSummary,
    /// Every problem found, by check.
    pub problems: Vec<Problem>,
}

/// One model of a [`Report`]: its path, and, when it can be read, its summary.
#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    /// The model file.
    #[serde(serialize_with = "lossy_path::one")]
    pub path: PathBuf,
    /// Whether its information could be read.
    pub readable: bool,
    /// The summary of a readable model.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub details: Option<ModelDetails>,
}

/// The summary of a readable model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelDetails {
    /// The model format version.
    pub format_version: u32,
    /// `genotype` or `promote`.
    pub task: String,
    /// `snp` or `indel`.
    pub variant_type: String,
    /// Whether the model has the re-genotype class.
    pub regenotype: bool,
    /// The number of features.
    pub n_features: usize,
    /// The contigs the training calls came from.
    pub contigs: Vec<String>,
    /// The lengths of those contigs in the training reference, if recorded.
    pub reference_contigs: Option<ReferenceContigs>,
    /// The region sets the features use.
    pub region_names: Vec<String>,
    /// The training samples.
    pub training_samples: Vec<String>,
    /// The feature source (`gvcf`, `vcf`, or `alignments:` and its read settings).
    pub feature_source: String,
    /// The read settings of the alignments source.
    pub alignment_settings: Option<String>,
    /// The read settings the repeat-length evidence must be measured with (the legacy ones for a
    /// model that records none).
    pub strlen_settings: Option<String>,
    /// The settings the pileup evidence must be measured with.
    pub pileup_settings: Option<String>,
    /// What scoring with the model needs.
    pub needs: Needs,
    /// The versions that trained the model.
    pub trained_with: TrainedWith,
    /// How its training reads were aligned (`null` if unknown).
    pub alignment: Option<AlignmentProvenance>,
    /// The digest of each region BED its features use, by name.
    pub region_digests: BTreeMap<String, RegionDigest>,
    /// The digest of each region BED of the policy it was trained to be scored with, by name.
    pub policy_region_digests: BTreeMap<String, RegionDigest>,
}

/// What scoring with models needs besides the VCF.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Needs {
    /// Repeat-length evidence (`--strlen` tables, or `--alignments`).
    pub strlen: bool,
    /// Pileup evidence (`--pileup` tables, or `--alignments`).
    pub pileup: bool,
    /// The reads (`--alignments` and `--reference`), for the alignments feature source.
    pub alignments: bool,
    /// The raw annotations of the gVCF source, in the VCF or a gVCF (`--gvcf`).
    pub gvcf: bool,
}

impl Needs {
    /// What a model needs.
    fn of(info: &ModelInfo) -> Self {
        Needs {
            strlen: StrlenOptions::of_model(info).is_some(),
            pileup: PileupOptions::of_model(info).is_some(),
            alignments: matches!(info.feature_source, FeatureSource::VcfAlignments(_)),
            gvcf: info.feature_source == FeatureSource::Gvcf,
        }
    }

    /// What either needs.
    fn or(self, other: Needs) -> Self {
        Needs {
            strlen: self.strlen || other.strlen,
            pileup: self.pileup || other.pileup,
            alignments: self.alignments || other.alignments,
            gvcf: self.gvcf || other.gvcf,
        }
    }

    /// The names of what is needed, in the order of the fields.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        [
            ("strlen", self.strlen),
            ("pileup", self.pileup),
            ("alignments", self.alignments),
            ("gvcf", self.gvcf),
        ]
        .into_iter()
        .filter_map(|(name, needed)| needed.then_some(name))
        .collect()
    }
}

/// The versions that trained a model.
#[derive(Debug, Clone, Serialize)]
pub struct TrainedWith {
    /// winnowr's.
    pub winnowr: String,
    /// XGBoost's.
    pub xgboost: String,
}

/// What the readable models share; a value is `null` when they disagree (a problem says how).
#[derive(Debug, Clone, Default, Serialize)]
pub struct SetSummary {
    /// The first model's contigs.
    pub contigs: Vec<String>,
    /// The models' reference contig lengths, merged (`null` if any model records none, or two
    /// models record different lengths of a contig).
    pub reference_contigs: Option<ReferenceContigs>,
    /// The region sets any model uses, sorted.
    pub region_names: Vec<String>,
    /// The shared feature source.
    pub feature_source: Option<String>,
    /// The shared alignment settings, for the alignments source.
    pub alignment_settings: Option<String>,
    /// The repeat-length settings the models need.
    pub strlen_settings: Option<String>,
    /// The pileup settings the models need.
    pub pileup_settings: Option<String>,
    /// What scoring with the models needs.
    pub needs: Needs,
    /// The not-difficult regions of `score`'s default policy.
    pub not_difficult_regions: NotDifficultRegions,
    /// How the models' training reads were aligned: `null` if a model records none, or two
    /// record different provenance.
    pub alignment: Option<AlignmentProvenance>,
}

/// Whether `score`'s default policy uses the not-difficult regions, and whether they were given.
#[derive(Debug, Clone, Default, Serialize)]
pub struct NotDifficultRegions {
    /// Their `--regions` name.
    pub name: String,
    /// True if a model (an indel genotype model) uses them.
    pub used: bool,
    /// True if `--regions` gives them.
    pub given: bool,
    /// The digest the models record for them, in the roles score's default policy reads them in
    /// (a model's feature, and the policy's when `used`; see
    /// [`crate::model_set::recorded_digests`]): `null` if none does, or they differ.
    pub recorded: Option<RegionDigest>,
    /// The BED used: given, or found beside the models (`null` if neither).
    #[serde(serialize_with = "lossy_path::option")]
    pub path: Option<PathBuf>,
}

/// The inputs the models were checked against.
#[derive(Debug, Clone, Default, Serialize)]
pub struct InputsSummary {
    /// `--reference`.
    #[serde(serialize_with = "lossy_path::option")]
    pub reference: Option<PathBuf>,
    /// `--vcf`.
    #[serde(serialize_with = "lossy_path::option")]
    pub vcf: Option<PathBuf>,
    /// `--regions`, in order.
    pub regions: Vec<RegionInput>,
    /// The region BEDs found beside the models (`regions/NAME.bed.gz`, `.bed.bgz` or `.bed`) and
    /// read, as no `--regions` BED of their name was given.
    pub bundled_regions: Vec<RegionInput>,
    /// `--alignments`.
    #[serde(serialize_with = "lossy_path::option")]
    pub alignments: Option<PathBuf>,
    /// The reads' alignment provenance, if known.
    pub alignment: Option<AlignmentProvenance>,
    /// Where it came from: `header`, `tags` or `header+tags`.
    pub alignment_from: Option<&'static str>,
}

/// A region BED, given or found beside the models.
#[derive(Debug, Clone, Serialize)]
pub struct RegionInput {
    /// Its name.
    pub name: String,
    /// Its path.
    #[serde(serialize_with = "lossy_path::one")]
    pub path: PathBuf,
    /// The models that use it.
    #[serde(serialize_with = "lossy_path::many")]
    pub used_by: Vec<PathBuf>,
    /// Its digest (as `models[].region_digests` holds them), if it was read as a BED a run uses.
    pub digest: Option<RegionDigest>,
    /// Whether it is the BED the models record (`null` if they record none, or it was not read).
    pub verified: Option<bool>,
}

/// Reads model files for [`check_model_set`]: each file's information, or the problem that
/// stopped it being read ([`Check::FormatVersion`] or [`Check::ModelUnreadable`]).
fn read_models(files: &[ModelFile]) -> Vec<Result<ModelInfo, Problem>> {
    files
        .par_iter()
        .map(|file| {
            ModelInfo::read_file(file).map_err(|error| {
                let check = if error.chain().any(|e| e.is::<UnsupportedFormatVersion>()) {
                    Check::FormatVersion
                } else {
                    Check::ModelUnreadable
                };
                Problem::error(check, Some(&file.path), format!("{error:#}"))
            })
        })
        .collect()
}

/// Checks a model set against its inputs: reads each model's information (never its booster),
/// the reference's and the VCF's contig lengths and the region BEDs, and runs every check (see
/// [`check_model_set`]), with the warnings `score` gives elsewhere or not at all: that the models'
/// reference is not checked ([`Check::ReferenceNotChecked`]), and that an indel genotype model
/// has no not-difficult regions ([`Check::NotDifficultMissing`]).
#[must_use]
pub fn check_models(inputs: &CheckInputs<'_>) -> Report {
    // The models, the reference index and the VCF header are read in parallel (the reads'
    // header, a few KB, is read after).
    // Each model is resolved once: its information is read from, and its region BEDs looked for
    // beside, that one file.
    let files = ModelFile::resolve_all(inputs.models);
    let (read, (reference, vcf)) = rayon::join(
        || read_models(&files),
        || {
            rayon::join(
                || inputs.reference.map(|path| (path, crate::fasta::index_lengths(path))),
                || {
                    inputs.vcf.map(|path| {
                        (
                            path,
                            crate::vcf::VcfSummary::read_header(path).map(|vcf| vcf.contig_lengths),
                        )
                    })
                },
            )
        },
    );
    let mut problems: Vec<Problem> =
        read.iter().filter_map(|info| info.as_ref().err().cloned()).collect();
    let (files, infos): (Vec<&ModelFile>, Vec<&ModelInfo>) = files
        .iter()
        .zip(&read)
        .filter_map(|(file, info)| Some((file, info.as_ref().ok()?)))
        .unzip();
    let paths: Vec<PathBuf> = files.iter().map(|file| file.path.clone()).collect();
    let resolved_paths: Vec<PathBuf> = files.iter().map(|file| file.resolved.clone()).collect();
    let given_dirs: Vec<PathBuf> = files.iter().map(|file| file.given_dir.clone()).collect();
    let mut contigs = Vec::new();
    for (option, source, read) in [("--reference", "reference", &reference), ("--vcf", "VCF", &vcf)]
    {
        match read {
            Some((path, Ok(lengths))) => {
                contigs.push((format!("{source} {}", path.display()), lengths))
            }
            Some((path, Err(error))) => problems.push(input_unreadable(option, path, error)),
            None => {}
        }
    }
    // Given but unreadable is an InputUnreadable error, not this.
    if inputs.reference.is_none() && inputs.vcf.is_none() {
        let message = "neither --reference nor --vcf was given, so the models' reference is not \
                       checked";
        problems.push(Problem::warning(Check::ReferenceNotChecked, None, message));
    }
    // The region BEDs score would read: the features', and the not-difficult regions of the
    // default policy of an indel genotype model, given or beside the models.
    let set = ModelSet {
        paths: &paths,
        resolved: &resolved_paths,
        given_dirs: &given_dirs,
        infos: &infos,
    };
    let indel_genotype = has_indel_genotype(&infos);
    let run = RunRegions::Features { policy: indel_genotype };
    let (plan, region_problems) = RegionPlan::new(&set, inputs.regions, run);
    let not_difficult_present = plan.has_candidate(NOT_DIFFICULT_REGIONS);
    let selected_indices: Vec<usize> = plan.given_indices().collect();
    // Decide which physical inputs remain before consuming any streams.
    let unread = unselected_region_beds(inputs.regions, &selected_indices);
    problems.extend(region_problems);
    // The selected and unselected BEDs are read at once, as one producer may open every stream
    // before writing any: reading one group first would wait on a stream of the other.
    let unreadable_models = infos.len() < inputs.models.len();
    let unread_inputs = SetInputs {
        regions: &unread,
        contigs: Vec::new(),
        unreadable_models,
        reads: ReadsCheck::default(),
    };
    let ((resolved, region_problems), unread_problems) = rayon::join(
        || plan.check(inputs.allow_region_mismatch),
        || check_region_beds(&set, &unread_inputs),
    );
    problems.extend(region_problems);
    let resolved_not_difficult = resolved.iter().find(|r| r.name == NOT_DIFFICULT_REGIONS);
    // A BED given but not read (unused, or unreadable) is still the one score would be given.
    let given_not_difficult = inputs.regions.iter().find(|bed| bed.name == NOT_DIFFICULT_REGIONS);
    let recorded_not_difficult = recorded_digests(&set, NOT_DIFFICULT_REGIONS, run);
    let not_difficult = NotDifficultRegions {
        name: NOT_DIFFICULT_REGIONS.to_string(),
        used: indel_genotype,
        given: given_not_difficult.is_some(),
        recorded: recorded_not_difficult.agreed().cloned(),
        path: resolved_not_difficult
            .map(|region| region.path.clone())
            .or_else(|| given_not_difficult.map(|bed| bed.path.clone())),
    };
    let recorded_policy = recorded_digests(&set, NOT_DIFFICULT_REGIONS, RunRegions::Policy(true));
    if indel_genotype && recorded_policy.digests.is_empty() {
        let message = format!(
            "no model records the policy digest of the {NOT_DIFFICULT_REGIONS} regions (train \
             --policy-regions), so their policy provenance is not checked, even if a feature \
             digest is checked"
        );
        problems.push(Problem::warning(Check::PolicyRegionUnrecorded, None, message));
    }
    let not_difficult_feature = infos.iter().any(|info| info.uses_region(NOT_DIFFICULT_REGIONS));
    // Models recording different digests are rejected (RegionDigestsDiffer) whatever BED is
    // given, so score has no fallback to advertise.
    let not_difficult_conflict = recorded_not_difficult.conflict().is_some();
    if not_difficult.used
        && !not_difficult_present
        && !not_difficult_feature
        && !not_difficult_conflict
    {
        let message = format!(
            "the not-difficult regions are neither given (--regions {NOT_DIFFICULT_REGIONS}=PATH) \
             nor beside the models ({}), so score's default policy drops indel calls at the \
             model's threshold everywhere; give them (GIAB's GRCh38 genome stratification \
             {NOT_DIFFICULT_REGIONS}.bed.gz) as score will have them",
            bundled_bed_files(NOT_DIFFICULT_REGIONS)
        );
        problems.push(Problem::warning(Check::NotDifficultMissing, None, message));
    }
    // With --alignment-tags none, the header is not read.
    let reads = resolve_reads(inputs.alignment_tags, inputs.alignments, Vec::new()).unwrap_or_else(
        |error| {
            let message = format!("{error:#}");
            problems.push(Problem::error(Check::InputUnreadable, None, message));
            ReadsAlignment { unknown: UnknownReason::Unreadable, ..ReadsAlignment::default() }
        },
    );
    if let (Some(path), Some(header)) = (inputs.alignments, &reads.header) {
        match header {
            Ok(header) => {
                if let Some(warning) = header.unprofiled_warning(path) {
                    log::warn!("{warning}");
                }
                if let Some(message) = header.mixed_warning(path) {
                    problems.push(Problem::warning(Check::AlignmentHeaderMixed, None, message));
                }
            }
            Err(error) => {
                let message = format!("--alignments {} cannot be read: {error}", path.display());
                problems.push(Problem::error(Check::InputUnreadable, None, message));
            }
        }
    }
    let ReadsAlignment { provenance: reads_alignment, from: alignment_from, unknown, .. } = reads;
    for (path, info) in paths.iter().zip(&infos) {
        if info.alignment.is_none() {
            let message = format!(
                "{} records no alignment provenance, so the reads' alignment cannot be checked \
                 against it",
                path.display()
            );
            problems.push(Problem::warning(Check::AlignmentProvenanceUnknown, Some(path), message));
        }
    }
    let set_inputs = SetInputs {
        regions: inputs.regions,
        contigs,
        unreadable_models,
        reads: ReadsCheck {
            alignment: reads_alignment.as_ref(),
            unknown,
            strict: inputs.strict_provenance,
        },
    };
    problems.extend(check_model_set(&set, &set_inputs));
    // Distinct unselected inputs, including duplicate names but not aliases of a BED already
    // selected or checked, even when its read failed.
    problems.extend(unread_problems);
    problems.sort_by_key(|problem| problem.check);
    let models = inputs
        .models
        .iter()
        .zip(&read)
        .map(|(path, info)| ModelSummary {
            path: path.clone(),
            readable: info.is_ok(),
            details: info.as_ref().ok().map(model_details),
        })
        .collect();
    Report {
        schema_version: SCHEMA_VERSION,
        winnowr_version: crate::version::VERSION.clone(),
        ok: problems.iter().all(|problem| problem.severity != Severity::Error),
        booster_checked: false,
        models,
        set: set_summary(&set, not_difficult),
        inputs: InputsSummary {
            reference: inputs.reference.map(Path::to_path_buf),
            vcf: inputs.vcf.map(Path::to_path_buf),
            alignments: inputs.alignments.map(Path::to_path_buf),
            alignment: reads_alignment.clone(),
            alignment_from: alignment_from.map(ProvenanceFrom::as_str),
            regions: given_region_inputs(&paths, &infos, inputs.regions, &resolved),
            bundled_regions: bundled_region_inputs(&paths, &infos, &resolved),
        },
        problems,
    }
}

/// The explicit BEDs not selected by the plan, each physical input once. Comparisons do not
/// open files, so aliases of a consumed FIFO never wait for a second producer.
fn unselected_region_beds(given: &[NamedBed], selected: &[usize]) -> Vec<NamedBed> {
    let mut checked: Vec<&Path> =
        selected.iter().map(|&index| given[index].path.as_path()).collect();
    let mut unread = Vec::new();
    for bed in given {
        if checked.iter().any(|path| same_input_path(path, &bed.path)) {
            continue;
        }
        checked.push(&bed.path);
        unread.push(bed.clone());
    }
    unread
}

/// The region BEDs given, in order, each with the models that use it and, if it was read as a
/// BED a run uses, its digest and whether it is the models'.
fn given_region_inputs(
    paths: &[PathBuf],
    infos: &[&ModelInfo],
    given: &[NamedBed],
    resolved: &[ResolvedRegion],
) -> Vec<RegionInput> {
    let mut read: Vec<Option<&ResolvedRegion>> = vec![None; given.len()];
    for region in resolved {
        if let Some(index) = region.given {
            read[index] = Some(region);
        }
    }
    given
        .iter()
        .zip(read)
        .map(|(bed, read)| region_input(paths, infos, &bed.name, &bed.path, read))
        .collect()
}

/// The region BEDs found beside the models and read, as [`given_region_inputs`].
fn bundled_region_inputs(
    paths: &[PathBuf],
    infos: &[&ModelInfo],
    resolved: &[ResolvedRegion],
) -> Vec<RegionInput> {
    resolved
        .iter()
        .filter(|region| region.given.is_none())
        .map(|region| region_input(paths, infos, &region.name, &region.path, Some(region)))
        .collect()
}

/// The summary of the region BED `name` at `path`, with the models (of `paths`) that use it and,
/// if it was `read`, its digest and whether it is the models'.
fn region_input(
    paths: &[PathBuf],
    infos: &[&ModelInfo],
    name: &str,
    path: &Path,
    read: Option<&ResolvedRegion>,
) -> RegionInput {
    RegionInput {
        name: name.to_string(),
        path: path.to_path_buf(),
        used_by: paths
            .iter()
            .zip(infos)
            .filter(|(_, info)| info.uses_region(name))
            .map(|(path, _)| path.clone())
            .collect(),
        digest: read.map(|r| r.digest.clone()),
        verified: read.and_then(ResolvedRegion::verified),
    }
}

/// The [`Check::InputUnreadable`] problem of an input that cannot be read.
fn input_unreadable(option: &str, path: &Path, error: &anyhow::Error) -> Problem {
    let message = format!("{option} {} cannot be read: {error:#}", path.display());
    Problem::error(Check::InputUnreadable, None, message)
}

/// The read settings of the alignments source.
fn alignment_settings(source: FeatureSource) -> Option<String> {
    match source {
        FeatureSource::VcfAlignments(options) => Some(options.to_string()),
        FeatureSource::Gvcf | FeatureSource::Vcf => None,
    }
}

/// The summary of a readable model.
fn model_details(info: &ModelInfo) -> ModelDetails {
    ModelDetails {
        format_version: info.format_version,
        task: info.task.to_string(),
        variant_type: info.variant_type.to_string(),
        regenotype: info.regenotype,
        n_features: info.features.len(),
        contigs: info.contigs.clone(),
        reference_contigs: info.reference_contigs.clone(),
        region_names: info.region_names.clone(),
        training_samples: info.training_samples.clone(),
        feature_source: info.feature_source.to_string(),
        alignment_settings: alignment_settings(info.feature_source),
        strlen_settings: info.strlen_options().map(|options| options.to_string()),
        pileup_settings: info.pileup_settings.map(|options| options.to_string()),
        needs: Needs::of(info),
        trained_with: TrainedWith {
            winnowr: info.winnowr_version.clone(),
            xgboost: info.xgboost_version.clone(),
        },
        alignment: info.alignment.clone(),
        region_digests: info.region_digests.clone(),
        policy_region_digests: info.policy_region_digests.clone(),
    }
}

/// What the readable models of a set share, with what is known of the not-difficult regions.
fn set_summary(set: &ModelSet<'_>, not_difficult: NotDifficultRegions) -> SetSummary {
    let infos = set.infos;
    let Some(first) = infos.first() else {
        return SetSummary { not_difficult_regions: not_difficult, ..SetSummary::default() };
    };
    let mut region_names: Vec<String> =
        infos.iter().flat_map(|info| info.region_names.iter().cloned()).collect();
    region_names.sort_unstable();
    region_names.dedup();
    let feature_source = models_feature_source(infos);
    SetSummary {
        contigs: first.contigs.clone(),
        reference_contigs: merged_reference_contigs(set),
        region_names,
        feature_source: feature_source.map(|source| source.to_string()),
        alignment_settings: feature_source.and_then(alignment_settings),
        strlen_settings: settings::<StrlenOptions>(set),
        pileup_settings: settings::<PileupOptions>(set),
        needs: infos.iter().fold(Needs::default(), |needs, info| needs.or(Needs::of(info))),
        not_difficult_regions: not_difficult,
        alignment: infos
            .iter()
            .all(|info| info.alignment.is_some())
            .then(|| models_alignment(infos))
            .flatten(),
    }
}

/// The settings the models need evidence `E` measured with, if they agree on them.
fn settings<E: ModelEvidence>(set: &ModelSet<'_>) -> Option<String> {
    evidence_settings::<E>(set).ok().flatten().map(|options| options.to_string())
}

/// Writes the report as JSON, then a newline. The JSON is made whole before any of it is written,
/// so a report that cannot be made writes nothing.
///
/// # Errors
///
/// Returns an error if it cannot be made or written.
pub fn write_json<W: Write>(report: &Report, mut out: W) -> Result<()> {
    let mut json = serde_json::to_vec_pretty(report).context("failed to make the JSON summary")?;
    json.push(b'\n');
    out.write_all(&json).context("failed to write the JSON summary")
}

/// Writes the report as text: one line per model (`out`), then each problem (`problems`).
///
/// # Errors
///
/// Returns an error if it cannot be written.
pub fn write_text<W: Write, E: Write>(report: &Report, mut out: W, mut problems: E) -> Result<()> {
    for model in &report.models {
        let line = match &model.details {
            None => format!("{}: not readable", model.path.display()),
            Some(details) => {
                let needs = details.needs.names();
                let regions = if details.region_names.is_empty() {
                    "none".to_string()
                } else {
                    details.region_names.join(",")
                };
                format!(
                    "{}: {} {}, feature source {}, needs {}, regions {}",
                    model.path.display(),
                    details.task,
                    details.variant_type,
                    details.feature_source,
                    if needs.is_empty() { "the VCF only".to_string() } else { needs.join(",") },
                    regions
                )
            }
        };
        writeln!(out, "{line}")?;
    }
    for problem in &report.problems {
        writeln!(problems, "{}: [{}] {}", problem.severity, problem.check, problem.message)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use tempfile::TempDir;

    use super::*;
    use crate::alignment_provenance::MIXED;
    use crate::common::{Task, VariantType};
    use crate::testing::bwa_provenance as provenance;

    /// `--alignment-tags` matching the test models' provenance, so that the tests of other
    /// checks see no alignment provenance problem.
    static MATCHING_TAGS: std::sync::LazyLock<AlignmentTags> = std::sync::LazyLock::new(|| {
        AlignmentTags::parse(&["aligner=bwa-mem3", "alt-project=primary"]).unwrap().unwrap()
    });

    /// Saves a genotype model of `variant_type` on chr1 (of 1000 bp), trained on reads aligned
    /// with `alt-project=primary`, as `name`.
    fn save(dir: &TempDir, name: &str, variant_type: VariantType) -> PathBuf {
        save_bundle_model(dir, name, variant_type, &[], &[])
    }

    /// Saves a genotype model `name` of `variant_type` (as [`save`]) whose features use the
    /// regions `region_names`, recording the `digests` of regions: of the not-difficult regions
    /// as the policy's, of others as a feature's.
    fn save_bundle_model(
        dir: &TempDir,
        name: &str,
        variant_type: VariantType,
        region_names: &[&str],
        digests: &[(&str, &RegionDigest)],
    ) -> PathBuf {
        let mut model = crate::testing::separable_model(Task::Genotype, variant_type);
        model.info.reference_contigs =
            Some([("chr1".to_string(), Some(1000))].into_iter().collect());
        model.info.alignment = Some(provenance("primary"));
        model.info.region_names = region_names.iter().map(|name| (*name).to_string()).collect();
        for &(region, digest) in digests {
            let recorded = if region == NOT_DIFFICULT_REGIONS {
                &mut model.info.policy_region_digests
            } else {
                &mut model.info.region_digests
            };
            recorded.insert(region.to_string(), digest.clone());
        }
        let path = dir.path().join(name);
        model.save(&path).unwrap();
        path
    }

    /// The report's first problem of `check`.
    fn problem_of(report: &Report, check: Check) -> &Problem {
        report.problems.iter().find(|problem| problem.check == check).unwrap()
    }

    fn inputs<'a>(models: &'a [PathBuf]) -> CheckInputs<'a> {
        CheckInputs {
            models,
            regions: &[],
            reference: None,
            vcf: None,
            alignments: None,
            alignment_tags: Some(&MATCHING_TAGS),
            strict_provenance: false,
            allow_region_mismatch: false,
        }
    }

    /// The checks of a report's problems, in order.
    fn checks(report: &Report) -> Vec<Check> {
        report.problems.iter().map(|problem| problem.check).collect()
    }

    #[test]
    fn a_good_set_is_ok() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        std::fs::write(dir.path().join("ref.fa.fai"), "chr1\t1000\t6\t60\t61\n").unwrap();
        let reference = dir.path().join("ref.fa");
        let report = check_models(&CheckInputs { reference: Some(&reference), ..inputs(&models) });
        assert!(report.ok, "{:?}", report.problems);
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert!(!report.booster_checked);
        let details = report.models[0].details.as_ref().unwrap();
        assert_eq!((details.task.as_str(), details.variant_type.as_str()), ("genotype", "snp"));
        assert_eq!(details.feature_source, "gvcf");
        assert_eq!(details.needs, Needs { gvcf: true, ..Needs::default() });
        assert_eq!(report.set.feature_source.as_deref(), Some("gvcf"));
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["models"][0]["readable"], true);
        assert_eq!(json["set"]["reference_contigs"]["chr1"], 1000);
    }

    /// The booster is scanned past, never parsed: a model whose booster XGBoost cannot load is
    /// still checked.
    #[test]
    fn the_booster_is_not_parsed() {
        let dir = TempDir::new().unwrap();
        let path = save(&dir, "snp.winnowr.json", VariantType::Snp);
        let text = std::fs::read_to_string(&path).unwrap();
        let start = text.find("\"booster\"").unwrap();
        let broken = format!("{}\"booster\":{{\"x\":[NaN, \"}}\"]}}}}", &text[..start]);
        std::fs::write(&path, &broken).unwrap();
        let threads = NonZeroUsize::new(1).unwrap();
        assert!(crate::model::WinnowrModel::load(&path, threads).is_err());
        let report = check_models(&inputs(std::slice::from_ref(&path)));
        assert!(report.models[0].readable, "{:?}", report.problems);
    }

    /// Every problem is reported at once, the JSON of an unreadable model has its path alone,
    /// and inputs that cannot be read are problems too.
    #[test]
    fn a_bad_set_reports_every_problem() {
        let dir = TempDir::new().unwrap();
        let unreadable = dir.path().join("cut.winnowr.json");
        std::fs::write(&unreadable, r#"{"format_version": 3, "booster": {"#).unwrap();
        let models = [
            save(&dir, "a.winnowr.json", VariantType::Snp),
            save(&dir, "b.winnowr.json", VariantType::Snp),
            unreadable,
        ];
        let reference = dir.path().join("absent.fa");
        let report = check_models(&CheckInputs { reference: Some(&reference), ..inputs(&models) });
        assert!(!report.ok);
        let checks: Vec<Check> = report.problems.iter().map(|problem| problem.check).collect();
        assert_eq!(checks, [Check::ModelUnreadable, Check::ModelKinds, Check::InputUnreadable]);
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["models"][2], serde_json::json!({"path": models[2], "readable": false}));
        assert_eq!(json["problems"][1]["check"], "model_kinds");
        assert_eq!(json["problems"][1]["model"], serde_json::Value::Null);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        write_text(&report, &mut out, &mut err).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(
            out.contains("a.winnowr.json: genotype snp, feature source gvcf, needs gvcf"),
            "{out}"
        );
        let err = String::from_utf8(err).unwrap();
        assert!(err.contains("error: [model_kinds] expected at most one model"), "{err}");
    }

    /// The reference index and the VCF are each compared with the models' reference.
    #[test]
    fn the_reference_index_and_the_vcf_are_checked() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        let fai = dir.path().join("ref.fa.fai");
        // The test VCF's chr1 is 100000 bp, the model's 1000 bp.
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &[]);
        let checked = || {
            check_models(&CheckInputs { reference: Some(&fai), vcf: Some(&vcf), ..inputs(&models) })
        };
        std::fs::write(&fai, "chr1\t1000\t6\t60\t61\n").unwrap();
        let report = checked();
        assert_eq!(checks(&report), [Check::ReferenceMismatch]);
        let message = &report.problems[0].message;
        assert!(message.starts_with(&format!("VCF {} is not on the reference", vcf.display())));
        assert!(message.ends_with("other lengths: chr1 (100000 bp, model 1000 bp)"), "{message}");

        std::fs::write(&fai, "chr1\t999\t6\t60\t61\n").unwrap();
        let report = checked();
        assert_eq!(checks(&report), [Check::ReferenceMismatch, Check::ReferenceMismatch]);
        let messages: Vec<&str> = report.problems.iter().map(|p| p.message.as_str()).collect();
        assert!(messages[0].starts_with(&format!("reference {}", fai.display())), "{messages:?}");
        assert!(messages[0].ends_with("chr1 (999 bp, model 1000 bp)"), "{messages:?}");
        assert!(messages[1].starts_with(&format!("VCF {}", vcf.display())), "{messages:?}");

        let report = check_models(&inputs(&models));
        assert_eq!(checks(&report), [Check::ReferenceNotChecked]);
        assert!(report.ok);
    }

    /// Only the VCF's header is read: a bgzipped VCF or a BCF without an index, or an
    /// uncompressed VCF, is checked as an indexed one is.
    #[test]
    fn the_vcf_needs_no_index() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        // The test VCF's chr1 is 100000 bp, the model's 1000 bp.
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let bgzipped = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &[]);
        std::fs::remove_file(dir.path().join("in.vcf.gz.tbi")).unwrap();
        let plain = dir.path().join("in.vcf");
        std::fs::write(&plain, &header).unwrap();
        let bcf = dir.path().join("in.bcf");
        {
            use rust_htslib::bcf::{Format, Header, Read as _, Reader, Writer};
            let reader = Reader::from_path(&plain).unwrap();
            let header = Header::from_template(reader.header());
            Writer::from_path(&bcf, &header, false, Format::Bcf).unwrap();
        }
        for vcf in [&bgzipped, &plain, &bcf] {
            let report = check_models(&CheckInputs { vcf: Some(vcf), ..inputs(&models) });
            assert_eq!(checks(&report), [Check::ReferenceMismatch], "{:?}", report.problems);
            let message = &report.problems[0].message;
            assert!(message.starts_with(&format!("VCF {} is not on the reference", vcf.display())));
        }
    }

    /// An input given but unreadable is an error, and not also "not given".
    #[test]
    fn an_unreadable_input_is_not_also_unchecked() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        let vcf = dir.path().join("absent.vcf.gz");
        let report = check_models(&CheckInputs { vcf: Some(&vcf), ..inputs(&models) });
        assert_eq!(checks(&report), [Check::InputUnreadable]);
    }

    /// An indel genotype model warns without the not-difficult regions.
    #[test]
    fn an_indel_genotype_model_needs_the_not_difficult_regions() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "indel.winnowr.json", VariantType::Indel)];
        let report = check_models(&inputs(&models));
        assert_eq!(
            checks(&report),
            [Check::NotDifficultMissing, Check::ReferenceNotChecked, Check::PolicyRegionUnrecorded]
        );
        assert!(report.ok);
    }

    #[test]
    fn a_feature_digest_does_not_record_policy_provenance() {
        let dir = TempDir::new().unwrap();
        let bed = dir.path().join("not-difficult.bed");
        std::fs::write(&bed, "chr1\t0\t10\n").unwrap();
        let digest = crate::regions::RegionSet::from_bed("nd", &bed).unwrap().digest().unwrap();
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Indel);
        model.info.region_names = vec![NOT_DIFFICULT_REGIONS.to_string()];
        model.info.region_digests.insert(NOT_DIFFICULT_REGIONS.to_string(), digest);
        let path = dir.path().join("indel.winnowr.json");
        model.save(&path).unwrap();
        let models = [path];
        let regions = [NamedBed { name: NOT_DIFFICULT_REGIONS.to_string(), path: bed }];
        let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
        assert_eq!(report.inputs.regions[0].verified, Some(true));
        let warning = problem_of(&report, Check::PolicyRegionUnrecorded);
        assert!(warning.message.contains("policy digest"), "{}", warning.message);
        assert!(!warning.message.contains("BED score uses for them is not checked"));
    }

    #[test]
    fn a_missing_required_feature_has_no_policy_fallback() {
        let dir = TempDir::new().unwrap();
        let models = [save_bundle_model(
            &dir,
            "indel.winnowr.json",
            VariantType::Indel,
            &[NOT_DIFFICULT_REGIONS],
            &[],
        )];
        let report = check_models(&inputs(&models));
        assert!(checks(&report).contains(&Check::RegionMissing), "{:?}", report.problems);
        assert!(!checks(&report).contains(&Check::NotDifficultMissing), "{:?}", report.problems);
    }

    /// Models recording different policy digests are rejected whatever BED is given, so no
    /// threshold-everywhere fallback is advertised.
    #[test]
    fn conflicting_policy_digests_advertise_no_fallback() {
        let dir = TempDir::new().unwrap();
        let digest = |text: &str| {
            let bed = dir.path().join(format!("{text}.bed"));
            std::fs::write(&bed, format!("chr1\t0\t{text}\n")).unwrap();
            crate::regions::RegionSet::from_bed("nd", &bed).unwrap().digest().unwrap()
        };
        let (first, second) = (digest("10"), digest("20"));
        let models = [
            save_bundle_model(
                &dir,
                "a.winnowr.json",
                VariantType::Indel,
                &[],
                &[(NOT_DIFFICULT_REGIONS, &first)],
            ),
            save_bundle_model(
                &dir,
                "b.winnowr.json",
                VariantType::Indel,
                &[],
                &[(NOT_DIFFICULT_REGIONS, &second)],
            ),
        ];
        let report = check_models(&inputs(&models));
        assert!(checks(&report).contains(&Check::RegionDigestsDiffer), "{:?}", report.problems);
        assert!(!checks(&report).contains(&Check::NotDifficultMissing), "{:?}", report.problems);
    }

    #[test]
    fn duplicate_explicit_beds_are_checked_by_index() {
        let dir = TempDir::new().unwrap();
        let models =
            [save_bundle_model(&dir, "snp.winnowr.json", VariantType::Snp, &["segdups"], &[])];
        let readable = dir.path().join("readable.bed");
        std::fs::write(&readable, "chr1\t0\t10\n").unwrap();
        for selected_readable in [true, false] {
            let selected = if selected_readable {
                readable.clone()
            } else {
                dir.path().join("first-missing.bed")
            };
            let missing = dir.path().join("second-missing.bed");
            let regions = [
                NamedBed { name: "segdups".to_string(), path: selected },
                NamedBed { name: "segdups".to_string(), path: missing.clone() },
            ];
            let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
            let unreadable: Vec<_> =
                report.problems.iter().filter(|p| p.check == Check::RegionUnreadable).collect();
            assert_eq!(
                unreadable.len(),
                if selected_readable { 1 } else { 2 },
                "{:?}",
                report.problems
            );
            assert!(unreadable.iter().any(|p| p.message.contains(&missing.display().to_string())));
            assert!(unreadable.iter().all(|p| p.severity == Severity::Error));
            assert!(checks(&report).contains(&Check::RegionRepeated));
        }
    }

    #[cfg(unix)]
    #[rstest::rstest]
    fn unselected_regular_bed_aliases_are_checked_once(
        #[values("same", "symlink", "hardlink")] alias: &str,
    ) {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        let path = dir.path().join("malformed.bed");
        std::fs::write(&path, "not a BED\n").unwrap();
        let duplicate = if alias == "same" {
            path.clone()
        } else {
            let duplicate = dir.path().join("alias.bed");
            if alias == "symlink" {
                std::os::unix::fs::symlink(&path, &duplicate).unwrap();
            } else {
                std::fs::hard_link(&path, &duplicate).unwrap();
            }
            duplicate
        };
        let regions = [
            NamedBed { name: "segdups".to_string(), path },
            NamedBed { name: "segdups".to_string(), path: duplicate },
        ];
        let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
        assert_eq!(
            checks(&report).iter().filter(|&&check| check == Check::RegionUnreadable).count(),
            1,
            "{:?}",
            report.problems
        );
    }

    #[cfg(unix)]
    #[rstest::rstest]
    fn distinct_selected_names_reject_only_shared_streams(
        #[values("same", "symlink", "hardlink")] alias: &str,
        #[values(false, true)] stream: bool,
    ) {
        let dir = TempDir::new().unwrap();
        let mut processes = crate::testing::TestProcesses::default();
        processes.spawn_test(
            "check_models::tests::distinct_selected_names_child",
            [
                ("WINNOWR_SELECTED_NAMES_ALIAS", alias.to_string()),
                ("WINNOWR_SELECTED_NAMES_STREAM", stream.to_string()),
            ],
            &dir.path().join("reader.log"),
        );
        processes.wait_for_success(std::time::Duration::from_secs(3));
    }

    #[cfg(unix)]
    #[test]
    fn distinct_selected_names_child() {
        let Ok(alias) = std::env::var("WINNOWR_SELECTED_NAMES_ALIAS") else { return };
        let stream = std::env::var("WINNOWR_SELECTED_NAMES_STREAM").unwrap() == "true";
        let dir = TempDir::new().unwrap();
        let models = [save_bundle_model(
            &dir,
            "snp.winnowr.json",
            VariantType::Snp,
            &["segdups", "lowmap"],
            &[],
        )];
        let path = dir.path().join("input.bed");
        if stream {
            assert!(std::process::Command::new("mkfifo").arg(&path).status().unwrap().success());
        } else {
            std::fs::write(&path, "chr1\t0\t10\n").unwrap();
        }
        let other = if alias == "same" {
            path.clone()
        } else {
            let other = dir.path().join("alias.bed");
            if alias == "symlink" {
                std::os::unix::fs::symlink(&path, &other).unwrap();
            } else {
                std::fs::hard_link(&path, &other).unwrap();
            }
            other
        };
        let regions = [
            NamedBed { name: "segdups".to_string(), path },
            NamedBed { name: "lowmap".to_string(), path: other },
        ];
        let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
        let unreadable: Vec<_> = report
            .problems
            .iter()
            .filter(|problem| problem.check == Check::RegionUnreadable)
            .collect();
        assert_eq!(unreadable.len(), usize::from(stream), "{:?}", report.problems);
        if stream {
            assert!(unreadable[0].message.contains("nonregular"));
        } else {
            assert_eq!(report.inputs.regions.len(), 2);
            assert!(report.inputs.regions.iter().all(|region| region.digest.is_some()));
        }
        crate::testing::child_body_ran();
    }

    /// Each physical stream is read once even when duplicate explicit inputs alias it.
    #[cfg(unix)]
    #[rstest::rstest]
    fn duplicate_explicit_stream_aliases_are_read_once(
        #[values("same", "symlink", "hardlink")] alias: &str,
        #[values(false, true)] selected: bool,
        #[values(false, true)] malformed: bool,
    ) {
        let dir = TempDir::new().unwrap();
        let names: &[&str] = if selected { &["segdups"] } else { &[] };
        let model = save_bundle_model(&dir, "snp.winnowr.json", VariantType::Snp, names, &[]);
        let fifo = dir.path().join("input.fifo");
        assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        let duplicate = if alias == "same" {
            fifo.clone()
        } else {
            let duplicate = dir.path().join("alias.fifo");
            if alias == "symlink" {
                std::os::unix::fs::symlink(&fifo, &duplicate).unwrap();
            } else {
                std::fs::hard_link(&fifo, &duplicate).unwrap();
            }
            duplicate
        };
        let mut processes = crate::testing::TestProcesses::default();
        for role in ["reader", "writer"] {
            processes.spawn_test(
                "check_models::tests::duplicate_explicit_stream_alias_child",
                [
                    ("WINNOWR_CHECK_ALIAS_ROLE", std::ffi::OsString::from(role)),
                    ("WINNOWR_CHECK_ALIAS_FIFO", fifo.clone().into_os_string()),
                    ("WINNOWR_CHECK_ALIAS_PATH", duplicate.clone().into_os_string()),
                    ("WINNOWR_CHECK_ALIAS_MODEL", model.clone().into_os_string()),
                    ("WINNOWR_CHECK_ALIAS_MALFORMED", malformed.to_string().into()),
                ],
                &dir.path().join(format!("{role}.log")),
            );
        }
        processes.wait_for_success(std::time::Duration::from_secs(5));
    }

    #[cfg(unix)]
    #[test]
    fn duplicate_explicit_stream_alias_child() {
        let Some(role) = std::env::var_os("WINNOWR_CHECK_ALIAS_ROLE") else { return };
        let fifo = PathBuf::from(std::env::var_os("WINNOWR_CHECK_ALIAS_FIFO").unwrap());
        let malformed = std::env::var("WINNOWR_CHECK_ALIAS_MALFORMED").unwrap() == "true";
        if role == "writer" {
            std::fs::write(&fifo, if malformed { "not a BED\n" } else { "chr1\t0\t10\n" }).unwrap();
        } else {
            let models = [PathBuf::from(std::env::var_os("WINNOWR_CHECK_ALIAS_MODEL").unwrap())];
            let duplicate = PathBuf::from(std::env::var_os("WINNOWR_CHECK_ALIAS_PATH").unwrap());
            let regions = [
                NamedBed { name: "segdups".to_string(), path: fifo },
                NamedBed { name: "segdups".to_string(), path: duplicate },
            ];
            let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
            assert_eq!(
                checks(&report).iter().filter(|&&check| check == Check::RegionUnreadable).count(),
                usize::from(malformed),
                "{:?}",
                report.problems
            );
            assert!(checks(&report).contains(&Check::RegionRepeated));
        }
        crate::testing::child_body_ran();
    }

    /// The selected stream is checked once even when parsing fails; a regression that opens it
    /// again is killed by the bounded child-process harness instead of hanging the suite.
    #[cfg(unix)]
    #[test]
    fn a_selected_failed_bed_stream_is_not_reopened() {
        let dir = TempDir::new().unwrap();
        let model =
            save_bundle_model(&dir, "snp.winnowr.json", VariantType::Snp, &["segdups"], &[]);
        let fifo = dir.path().join("selected.fifo");
        assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        let mut processes = crate::testing::TestProcesses::default();
        for role in ["reader", "writer"] {
            processes.spawn_test(
                "check_models::tests::selected_failed_bed_stream_child",
                [
                    ("WINNOWR_CHECK_BED_ROLE", std::ffi::OsString::from(role)),
                    ("WINNOWR_CHECK_BED_FIFO", fifo.clone().into_os_string()),
                    ("WINNOWR_CHECK_BED_MODEL", model.clone().into_os_string()),
                ],
                &dir.path().join(format!("{role}.log")),
            );
        }
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    #[cfg(unix)]
    #[test]
    fn selected_failed_bed_stream_child() {
        let Some(role) = std::env::var_os("WINNOWR_CHECK_BED_ROLE") else { return };
        let fifo = PathBuf::from(std::env::var_os("WINNOWR_CHECK_BED_FIFO").unwrap());
        if role == "writer" {
            std::fs::write(&fifo, "not a BED\n").unwrap();
        } else {
            let models = [PathBuf::from(std::env::var_os("WINNOWR_CHECK_BED_MODEL").unwrap())];
            let regions = [
                NamedBed { name: "segdups".to_string(), path: fifo.clone() },
                NamedBed { name: "segdups".to_string(), path: fifo.with_file_name("missing.bed") },
            ];
            let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
            assert_eq!(
                checks(&report).iter().filter(|&&check| check == Check::RegionUnreadable).count(),
                2,
                "{:?}",
                report.problems
            );
            assert!(checks(&report).contains(&Check::RegionRepeated));
        }
        crate::testing::child_body_ran();
    }

    /// A selected and an unselected stream are read at once: a producer that opens both before
    /// writing either finishes, where reading one group before the other would wait forever.
    #[cfg(unix)]
    #[test]
    fn selected_and_unselected_streams_are_read_together() {
        let dir = TempDir::new().unwrap();
        let model =
            save_bundle_model(&dir, "snp.winnowr.json", VariantType::Snp, &["segdups"], &[]);
        let selected = dir.path().join("segdups.fifo");
        let unselected = dir.path().join("lowmap.fifo");
        for fifo in [&selected, &unselected] {
            assert!(std::process::Command::new("mkfifo").arg(fifo).status().unwrap().success());
        }
        let mut processes = crate::testing::TestProcesses::default();
        for role in ["reader", "writer"] {
            processes.spawn_test(
                "check_models::tests::selected_and_unselected_streams_child",
                [
                    ("WINNOWR_CHECK_TOGETHER_ROLE", std::ffi::OsString::from(role)),
                    ("WINNOWR_CHECK_TOGETHER_SELECTED", selected.clone().into_os_string()),
                    ("WINNOWR_CHECK_TOGETHER_UNSELECTED", unselected.clone().into_os_string()),
                    ("WINNOWR_CHECK_TOGETHER_MODEL", model.clone().into_os_string()),
                ],
                &dir.path().join(format!("{role}.log")),
            );
        }
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }

    #[cfg(unix)]
    #[test]
    fn selected_and_unselected_streams_child() {
        let Some(role) = std::env::var_os("WINNOWR_CHECK_TOGETHER_ROLE") else { return };
        let selected = PathBuf::from(std::env::var_os("WINNOWR_CHECK_TOGETHER_SELECTED").unwrap());
        let unselected =
            PathBuf::from(std::env::var_os("WINNOWR_CHECK_TOGETHER_UNSELECTED").unwrap());
        if role == "writer" {
            // Opening a FIFO to write waits for its reader, so both are open before either is
            // written.
            let open = |path: &Path| std::fs::OpenOptions::new().write(true).open(path).unwrap();
            let mut writers = [open(&selected), open(&unselected)];
            for writer in &mut writers {
                writer.write_all(b"chr1\t0\t10\n").unwrap();
            }
        } else {
            let models = [PathBuf::from(std::env::var_os("WINNOWR_CHECK_TOGETHER_MODEL").unwrap())];
            let regions = [
                NamedBed { name: "segdups".to_string(), path: selected },
                NamedBed { name: "lowmap".to_string(), path: unselected },
            ];
            let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
            assert!(!checks(&report).contains(&Check::RegionUnreadable), "{:?}", report.problems);
            assert!(checks(&report).contains(&Check::RegionUnused), "{:?}", report.problems);
        }
        crate::testing::child_body_ran();
    }

    #[cfg(unix)]
    #[test]
    fn a_broken_bundled_policy_link_is_unreadable_not_missing() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "indel.winnowr.json", VariantType::Indel)];
        let regions = dir.path().join("regions");
        std::fs::create_dir(&regions).unwrap();
        let bed = regions.join(format!("{NOT_DIFFICULT_REGIONS}.bed"));
        std::os::unix::fs::symlink(dir.path().join("absent.bed"), &bed).unwrap();
        let report = check_models(&inputs(&models));
        assert!(checks(&report).contains(&Check::RegionUnreadable), "{:?}", report.problems);
        assert!(!checks(&report).contains(&Check::NotDifficultMissing), "{:?}", report.problems);
    }

    /// A bundled BED that exists but cannot be read is not a missing optional policy BED.
    #[test]
    fn an_unreadable_bundled_policy_bed_is_not_missing() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "indel.winnowr.json", VariantType::Indel)];
        let regions = dir.path().join("regions");
        std::fs::create_dir(&regions).unwrap();
        let bed = regions.join(format!("{NOT_DIFFICULT_REGIONS}.bed"));
        std::fs::write(&bed, "not a BED\n").unwrap();
        let report = check_models(&inputs(&models));
        assert!(!report.ok);
        assert!(checks(&report).contains(&Check::RegionUnreadable), "{:?}", report.problems);
        assert!(!checks(&report).contains(&Check::NotDifficultMissing), "{:?}", report.problems);
    }

    /// Ambiguous bundled policy BEDs are present too, even though neither can be selected.
    #[test]
    fn ambiguous_bundled_policy_beds_are_not_missing() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "indel.winnowr.json", VariantType::Indel)];
        let regions = dir.path().join("regions");
        std::fs::create_dir(&regions).unwrap();
        std::fs::write(regions.join(format!("{NOT_DIFFICULT_REGIONS}.bed")), "chr1\t0\t10\n")
            .unwrap();
        crate::testing::write_bgzf(
            &regions.join(format!("{NOT_DIFFICULT_REGIONS}.bed.gz")),
            b"chr1\t20\t30\n",
        );
        let report = check_models(&inputs(&models));
        assert!(!report.ok);
        assert!(checks(&report).contains(&Check::RegionAmbiguous), "{:?}", report.problems);
        assert!(!checks(&report).contains(&Check::NotDifficultMissing), "{:?}", report.problems);
    }

    /// The BEDs are checked even when no model can be read, and a BED is not called unused while
    /// a model that may use it cannot be read.
    #[test]
    fn regions_are_checked_with_models_that_cannot_be_read() {
        let dir = TempDir::new().unwrap();
        let unreadable = dir.path().join("cut.winnowr.json");
        std::fs::write(&unreadable, r#"{"format_version": 3, "booster": {"#).unwrap();
        let regions = [NamedBed { name: "segdups".to_string(), path: dir.path().join("x.bed") }];
        let models = [unreadable.clone()];
        let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
        assert_eq!(
            checks(&report),
            [Check::ModelUnreadable, Check::RegionUnreadable, Check::ReferenceNotChecked]
        );
        assert_eq!(report.problems[1].severity, Severity::Error);

        let bed = dir.path().join("segdups.bed");
        std::fs::write(&bed, "chr1\t0\t10\n").unwrap();
        let regions = [NamedBed { name: "segdups".to_string(), path: bed }];
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp), unreadable];
        let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
        assert_eq!(checks(&report), [Check::ModelUnreadable, Check::ReferenceNotChecked]);
    }

    /// A model that cannot be read, or is of another format version, is a problem of its own.
    #[test]
    fn unreadable_models_are_problems() {
        let dir = TempDir::new().unwrap();
        let old = dir.path().join("old.winnowr.json");
        std::fs::write(&old, r#"{"format_version": 1, "booster": {}}"#).unwrap();
        let cut = dir.path().join("cut.winnowr.json");
        std::fs::write(&cut, r#"{"format_version": 3, "booster": {"x": [1, 2"#).unwrap();
        let read = read_models(&ModelFile::resolve_all(&[old, cut, dir.path().join("a.json")]));
        let problems: Vec<Check> = read.iter().map(|r| r.as_ref().unwrap_err().check).collect();
        assert_eq!(
            problems,
            [Check::FormatVersion, Check::ModelUnreadable, Check::ModelUnreadable]
        );
        let message = &read[0].as_ref().unwrap_err().message;
        assert!(message.contains("unsupported model format version 1"), "{message}");
    }

    /// The keys of a readable model and of the set, as schema version 1 has them.
    #[test]
    fn the_json_of_a_readable_model_has_the_schema_keys() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        let json = serde_json::to_value(check_models(&inputs(&models))).unwrap();
        let keys = |value: &serde_json::Value| -> Vec<String> {
            let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
            keys.sort_unstable();
            keys
        };
        assert_eq!(
            keys(&json),
            [
                "booster_checked",
                "inputs",
                "models",
                "ok",
                "problems",
                "schema_version",
                "set",
                "winnowr_version"
            ]
        );
        assert_eq!(
            keys(&json["models"][0]),
            [
                "alignment",
                "alignment_settings",
                "contigs",
                "feature_source",
                "format_version",
                "n_features",
                "needs",
                "path",
                "pileup_settings",
                "policy_region_digests",
                "readable",
                "reference_contigs",
                "regenotype",
                "region_digests",
                "region_names",
                "strlen_settings",
                "task",
                "trained_with",
                "training_samples",
                "variant_type"
            ]
        );
        assert_eq!(
            keys(&json["set"]),
            [
                "alignment",
                "alignment_settings",
                "contigs",
                "feature_source",
                "needs",
                "not_difficult_regions",
                "pileup_settings",
                "reference_contigs",
                "region_names",
                "strlen_settings"
            ]
        );
        assert_eq!(keys(&json["models"][0]["needs"]), ["alignments", "gvcf", "pileup", "strlen"]);
        assert_eq!(keys(&json["models"][0]["trained_with"]), ["winnowr", "xgboost"]);
        assert_eq!(
            keys(&json["set"]["not_difficult_regions"]),
            ["given", "name", "path", "recorded", "used"]
        );
        assert_eq!(
            keys(&json["inputs"]),
            [
                "alignment",
                "alignment_from",
                "alignments",
                "bundled_regions",
                "reference",
                "regions",
                "vcf"
            ]
        );
        assert_eq!(json["models"][0]["alignment"]["alt-project"], "primary");
        assert_eq!(json["set"]["alignment"]["alt-project"], "primary");
        assert_eq!(json["inputs"]["alignment_from"], "tags");
        assert_eq!(keys(&json["problems"][0]), ["check", "message", "model", "severity"]);
    }

    /// Reference contigs of the given lengths.
    fn lengths(contigs: &[(&str, Option<u64>)]) -> ReferenceContigs {
        contigs.iter().map(|&(name, length)| (name.to_string(), length)).collect()
    }

    /// A model recording `contigs` as its reference contigs.
    fn recording(contigs: Option<&[(&str, Option<u64>)]>) -> ModelInfo {
        let mut info = crate::testing::separable_model(Task::Genotype, VariantType::Snp).info;
        info.reference_contigs = contigs.map(lengths);
        info
    }

    /// The not-difficult regions, neither used nor given.
    fn not_difficult() -> NotDifficultRegions {
        NotDifficultRegions {
            name: NOT_DIFFICULT_REGIONS.to_string(),
            used: false,
            given: false,
            recorded: None,
            path: None,
        }
    }

    #[test]
    fn the_set_summary_is_null_where_the_models_disagree() {
        let mut a = recording(None);
        a.feature_source = FeatureSource::Vcf;
        let mut b = recording(None);
        b.task = Task::Promote;
        b.pileup_settings = Some(PileupOptions::default());
        let paths = [PathBuf::from("a.winnowr.json"), PathBuf::from("b.winnowr.json")];
        let summary = set_summary(
            &ModelSet {
                paths: &paths,
                resolved: &paths,
                given_dirs: &crate::testing::given_dirs(&paths),
                infos: &[&a, &b],
            },
            not_difficult(),
        );
        assert_eq!(summary.feature_source, None);
        assert_eq!(summary.alignment_settings, None);
        assert_eq!(summary.reference_contigs, None);
        assert_eq!(summary.pileup_settings, Some(PileupOptions::default().to_string()));
        assert_eq!(summary.needs, Needs { pileup: true, gvcf: true, ..Needs::default() });
        assert_eq!(summary.needs.names(), ["pileup", "gvcf"]);

        let summary = set_summary(
            &ModelSet { paths: &[], resolved: &[], given_dirs: &[], infos: &[] },
            not_difficult(),
        );
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["contigs"], serde_json::json!([]));
        assert_eq!(json["feature_source"], serde_json::Value::Null);
        assert_eq!(json["not_difficult_regions"]["used"], false);
    }

    /// Each region BED names the models that use it; the not-difficult regions are used by an
    /// indel genotype model.
    #[test]
    fn regions_name_the_models_that_use_them() {
        let dir = TempDir::new().unwrap();
        let snp_path =
            save_bundle_model(&dir, "snp.winnowr.json", VariantType::Snp, &["segdups"], &[]);
        let models = [snp_path.clone(), save(&dir, "indel.winnowr.json", VariantType::Indel)];
        let bed = |name: &str| {
            let path = dir.path().join(format!("{name}.bed"));
            std::fs::write(&path, "chr1\t0\t10\n").unwrap();
            NamedBed { name: name.to_string(), path }
        };
        let regions = [bed("segdups"), bed(NOT_DIFFICULT_REGIONS)];
        let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
        // The models record no digests, so the BEDs are used unchecked.
        assert_eq!(
            checks(&report),
            [Check::ReferenceNotChecked, Check::RegionUnverified, Check::PolicyRegionUnrecorded]
        );
        assert_eq!(report.inputs.regions[0].used_by, [snp_path]);
        assert!(report.inputs.bundled_regions.is_empty());
        assert_eq!(report.inputs.regions[0].digest.as_ref().map(|d| d.intervals), Some(1));
        assert_eq!(report.inputs.regions[0].verified, None);
        assert!(report.inputs.regions[1].used_by.is_empty());
        let not_difficult = &report.set.not_difficult_regions;
        assert_eq!(not_difficult.name, NOT_DIFFICULT_REGIONS);
        assert!(not_difficult.used && not_difficult.given);
        assert_eq!(report.set.region_names, ["segdups"]);
    }

    /// The checks of a report's alignment provenance problems, with their severities.
    fn alignment_problems(report: &Report) -> Vec<(Check, Severity)> {
        report
            .problems
            .iter()
            .filter(|problem| problem.check.as_str().starts_with("alignment_"))
            .map(|problem| (problem.check, problem.severity))
            .collect()
    }

    /// Models that record no provenance are each a warning; models that record it are checked
    /// against the reads', or say that they are not.
    #[test]
    fn the_alignment_provenance_is_checked() {
        let dir = TempDir::new().unwrap();
        let primary = save(&dir, "snp.winnowr.json", VariantType::Snp);
        let mut unknown = crate::testing::separable_model(Task::Genotype, VariantType::Indel);
        unknown.info.reference_contigs = Some(lengths(&[("chr1", Some(1000))]));
        let unknown_path = dir.path().join("indel.winnowr.json");
        unknown.save(&unknown_path).unwrap();
        let models = [primary.clone(), unknown_path.clone()];
        let report = check_models(&inputs(&models));
        assert_eq!(
            alignment_problems(&report),
            [(Check::AlignmentProvenanceUnknown, Severity::Warning)]
        );
        let unknown = problem_of(&report, Check::AlignmentProvenanceUnknown);
        assert_eq!(unknown.model.as_deref(), Some(unknown_path.as_path()));
        assert_eq!(report.set.alignment, None);

        // Not checked: a warning, an error when strict.
        let models = [primary];
        let unchecked = CheckInputs { alignment_tags: None, ..inputs(&models) };
        let report = check_models(&unchecked);
        assert_eq!(
            alignment_problems(&report),
            [(Check::AlignmentProvenanceNotChecked, Severity::Warning)]
        );
        assert!(report.ok);
        let report = check_models(&CheckInputs { strict_provenance: true, ..unchecked });
        assert_eq!(
            alignment_problems(&report),
            [(Check::AlignmentProvenanceNotChecked, Severity::Error)]
        );
        assert!(!report.ok);

        // A mismatch names the differing tags, also in the JSON.
        let keep_alt =
            AlignmentTags::parse(&["aligner=bwa-mem3", "alt-project=keep-alt"]).unwrap().unwrap();
        let mismatched = CheckInputs { alignment_tags: Some(&keep_alt), ..inputs(&models) };
        let report = check_models(&mismatched);
        assert_eq!(
            alignment_problems(&report),
            [(Check::AlignmentProvenanceMismatch, Severity::Warning)]
        );
        let problem = problem_of(&report, Check::AlignmentProvenanceMismatch);
        assert!(
            problem.message.starts_with(
                "the reads were not aligned as the models' training reads were: alt-project: \
                 models primary, reads keep-alt."
            ),
            "{}",
            problem.message
        );
        let json = serde_json::to_value(&report).unwrap();
        let mismatch = json["problems"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["check"] == "alignment_provenance_mismatch")
            .unwrap();
        assert_eq!(
            mismatch["differences"],
            serde_json::json!([{"tag": "alt-project", "models": "primary", "reads": "keep-alt"}])
        );
        let report = check_models(&CheckInputs { strict_provenance: true, ..mismatched });
        assert!(!report.ok);
    }

    /// Models that record different provenance are a problem of their own, and each is compared
    /// with the reads.
    #[test]
    fn models_aligned_differently_are_a_problem() {
        let dir = TempDir::new().unwrap();
        let snp = save(&dir, "snp.winnowr.json", VariantType::Snp);
        let mut indel = crate::testing::separable_model(Task::Genotype, VariantType::Indel);
        indel.info.reference_contigs = Some(lengths(&[("chr1", Some(1000))]));
        indel.info.alignment = Some(provenance("keep-alt"));
        let indel_path = dir.path().join("indel.winnowr.json");
        indel.save(&indel_path).unwrap();
        let models = [snp, indel_path.clone()];
        let report = check_models(&inputs(&models));
        assert_eq!(
            alignment_problems(&report),
            [
                (Check::AlignmentProvenanceDiffers, Severity::Warning),
                (Check::AlignmentProvenanceMismatch, Severity::Warning)
            ]
        );
        let mismatch = problem_of(&report, Check::AlignmentProvenanceMismatch);
        assert_eq!(mismatch.model.as_deref(), Some(indel_path.as_path()));
        assert_eq!(report.set.alignment, None);
    }

    /// The reads' header is read for their provenance (no reference needed); aligner records
    /// that disagree are a warning.
    #[test]
    fn the_reads_header_is_read() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        let bwa = "@PG\tID:bwa-mem3\tPN:bwa-mem3\tVN:0.16.0\tCL:bwa-mem3 mem --alt-project primary ref.fa a.fq";
        let bam = crate::testing::write_bam_with_header(dir.path(), "a.bam", 1000, &[], &[bwa]);
        let header =
            CheckInputs { alignments: Some(&bam), alignment_tags: None, ..inputs(&models) };
        let report = check_models(&header);
        assert_eq!(
            alignment_problems(&report),
            [
                (Check::AlignmentProvenanceMismatch, Severity::Warning),
                (Check::AlignmentProvenanceIncomplete, Severity::Warning),
            ]
        );
        let mismatch = report
            .problems
            .iter()
            .find(|problem| problem.check == Check::AlignmentProvenanceMismatch)
            .unwrap();
        assert_eq!(
            mismatch
                .differences
                .iter()
                .map(|difference| difference.tag.as_str())
                .collect::<Vec<_>>(),
            ["sq-alt"]
        );
        assert_eq!(report.inputs.alignment_from, Some("header"));
        let alignment = report.inputs.alignment.as_ref().unwrap();
        assert_eq!(alignment.0["aligner"], "bwa-mem3");
        assert_eq!(alignment.0["version"], "0.16.0");
        assert_eq!(alignment.0["options-inference"], "incomplete");
        assert!(!alignment.0.contains_key("alt-project"));
        let strict = check_models(&CheckInputs { strict_provenance: true, ..header });
        assert_eq!(
            alignment_problems(&strict),
            [
                (Check::AlignmentProvenanceMismatch, Severity::Error),
                (Check::AlignmentProvenanceIncomplete, Severity::Error),
            ]
        );
        let declared =
            AlignmentTags::parse(&["options-inference=explicit", "alt-project=primary", "sq-alt!"])
                .unwrap()
                .unwrap();
        let explicit = check_models(&CheckInputs {
            alignment_tags: Some(&declared),
            strict_provenance: true,
            ..header
        });
        assert!(alignment_problems(&explicit).is_empty(), "{:?}", explicit.problems);

        // PN provides a reliable difference even though neither command line proves its options.
        let other = "@PG\tID:bwa-other\tPN:bwa\tCL:bwa mem ref.fa b.fq";
        let mixed =
            crate::testing::write_bam_with_header(dir.path(), "m.bam", 1000, &[], &[bwa, other]);
        let report = check_models(&CheckInputs { alignments: Some(&mixed), ..header });
        let problems = alignment_problems(&report);
        assert_eq!(problems.len(), 3, "{problems:?}");
        for expected in [
            (Check::AlignmentProvenanceMismatch, Severity::Warning),
            (Check::AlignmentHeaderMixed, Severity::Warning),
            (Check::AlignmentProvenanceIncomplete, Severity::Warning),
        ] {
            assert!(problems.contains(&expected), "{problems:?}");
        }
        assert_eq!(report.inputs.alignment.as_ref().unwrap().0["aligner"], MIXED);
    }

    /// With --alignment-tags none, the reads' header is not read: an unreadable one is no
    /// problem.
    #[test]
    fn the_header_is_not_read_with_alignment_tags_none() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        let absent = dir.path().join("absent.bam");
        let report = check_models(&CheckInputs {
            alignments: Some(&absent),
            alignment_tags: None,
            ..inputs(&models)
        });
        assert!(checks(&report).contains(&Check::InputUnreadable), "{:?}", report.problems);
        // The unreadable header is the problem: the reads' alignment is not also "not checked".
        assert!(alignment_problems(&report).is_empty(), "{:?}", report.problems);
        let none = AlignmentTags::None;
        let report = check_models(&CheckInputs {
            alignments: Some(&absent),
            alignment_tags: Some(&none),
            ..inputs(&models)
        });
        assert!(!checks(&report).contains(&Check::InputUnreadable), "{:?}", report.problems);
        assert_eq!(
            alignment_problems(&report),
            [(Check::AlignmentProvenanceNotChecked, Severity::Warning)]
        );
        assert_eq!(report.inputs.alignment, None);
    }

    /// --alignment-tags alone is compared on the tags given only.
    #[test]
    fn tags_alone_are_compared_on_the_tags_given() {
        let dir = TempDir::new().unwrap();
        let models = [save(&dir, "snp.winnowr.json", VariantType::Snp)];
        let given = AlignmentTags::parse(&["alt-project=primary"]).unwrap().unwrap();
        let report = check_models(&CheckInputs { alignment_tags: Some(&given), ..inputs(&models) });
        assert!(alignment_problems(&report).is_empty(), "{:?}", report.problems);
        let other = AlignmentTags::parse(&["alt-project=keep-alt"]).unwrap().unwrap();
        let report = check_models(&CheckInputs { alignment_tags: Some(&other), ..inputs(&models) });
        assert_eq!(
            alignment_problems(&report),
            [(Check::AlignmentProvenanceMismatch, Severity::Warning)]
        );
    }

    /// The region BEDs found beside the models are listed, with their digests and whether they
    /// are the models' own; a BED that is not is an error, unless allowed.
    #[test]
    fn region_beds_beside_the_models_are_summarized() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("regions")).unwrap();
        let bed = dir.path().join("regions/segdups.bed");
        std::fs::write(&bed, "chr1\t0\t10\n").unwrap();
        let digest =
            crate::regions::RegionSet::from_bed("segdups", &bed).unwrap().digest().unwrap();
        let path = save_bundle_model(
            &dir,
            "snp.winnowr.json",
            VariantType::Snp,
            &["segdups"],
            &[("segdups", &digest)],
        );
        let models = [path];
        let report = check_models(&inputs(&models));
        assert_eq!(checks(&report), [Check::ReferenceNotChecked]);
        let json = serde_json::to_value(&report).unwrap();
        // A BED found beside the models is not one given.
        assert_eq!(json["inputs"]["regions"], serde_json::json!([]));
        let region = &json["inputs"]["bundled_regions"][0];
        assert!(region["path"].as_str().unwrap().ends_with("regions/segdups.bed"), "{region}");
        assert_eq!(region["verified"], true);
        assert_eq!(region["digest"]["intervals"], 1);
        assert_eq!(json["models"][0]["region_digests"]["segdups"]["bases"], 10);

        std::fs::write(&bed, "chr1\t0\t20\n").unwrap();
        let report = check_models(&inputs(&models));
        assert_eq!(checks(&report), [Check::ReferenceNotChecked, Check::RegionChecksum]);
        assert!(!report.ok);
        assert_eq!(report.inputs.bundled_regions[0].verified, Some(false));
        let report = check_models(&CheckInputs { allow_region_mismatch: true, ..inputs(&models) });
        assert!(report.ok);
    }

    /// The not-difficult regions found beside the models are used, with the digest the models
    /// record; models that record different ones are an error, with none recorded for the set.
    /// A region BED a run uses that cannot be read is an error.
    #[test]
    fn the_not_difficult_regions_beside_the_models() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("regions")).unwrap();
        let bed = dir.path().join(format!("regions/{NOT_DIFFICULT_REGIONS}.bed.gz"));
        std::fs::write(&bed, "chr1\t0\t10\n").unwrap();
        let digest = crate::regions::RegionSet::from_bed("nd", &bed).unwrap().digest().unwrap();
        let save_with = |name: &str, variant_type: VariantType, digest: &RegionDigest| {
            save_bundle_model(&dir, name, variant_type, &[], &[(NOT_DIFFICULT_REGIONS, digest)])
        };
        let indel = save_with("indel.winnowr.json", VariantType::Indel, &digest);
        let models = [indel.clone()];
        let report = check_models(&inputs(&models));
        assert_eq!(checks(&report), [Check::ReferenceNotChecked]);
        let not_difficult = &report.set.not_difficult_regions;
        // Found in the models' directory as resolved (e.g. macOS's /var is /private/var).
        assert_eq!(not_difficult.path.as_deref(), Some(bed.canonicalize().unwrap().as_path()));
        assert_eq!(not_difficult.recorded.as_ref(), Some(&digest));

        let other = RegionDigest { bases: 20, ..digest.clone() };
        let snp = save_with("snp.winnowr.json", VariantType::Snp, &other);
        let models = [indel, snp];
        let report = check_models(&inputs(&models));
        assert_eq!(checks(&report), [Check::ReferenceNotChecked, Check::RegionDigestsDiffer]);
        assert_eq!(report.set.not_difficult_regions.recorded, None);

        // A used BED that cannot be read.
        let segdups = dir.path().join("regions/segdups.bed");
        std::fs::write(&segdups, "not a bed\n").unwrap();
        let path =
            save_bundle_model(&dir, "segdups.winnowr.json", VariantType::Snp, &["segdups"], &[]);
        let report = check_models(&inputs(&[path]));
        assert_eq!(checks(&report), [Check::RegionUnreadable, Check::ReferenceNotChecked]);
        assert!(!report.ok);
    }

    /// Paths that are not UTF-8 are written lossily, and the JSON is whole.
    #[cfg(unix)]
    #[test]
    fn paths_that_are_not_utf8_make_whole_json() {
        use std::os::unix::ffi::OsStrExt;
        let model = PathBuf::from(std::ffi::OsStr::from_bytes(b"a\xff.winnowr.json"));
        let vcf = PathBuf::from(std::ffi::OsStr::from_bytes(b"b\xff.vcf.gz"));
        let regions = [NamedBed { name: "r".to_string(), path: vcf.clone() }];
        let models = [model];
        let report =
            check_models(&CheckInputs { regions: &regions, vcf: Some(&vcf), ..inputs(&models) });
        let mut out = Vec::new();
        write_json(&report, &mut out).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(json["models"][0]["path"], "a\u{fffd}.winnowr.json");
        assert_eq!(json["inputs"]["vcf"], "b\u{fffd}.vcf.gz");
        assert_eq!(json["inputs"]["regions"][0]["path"], "b\u{fffd}.vcf.gz");
        assert_eq!(json["problems"][0]["model"], "a\u{fffd}.winnowr.json");
    }
}
