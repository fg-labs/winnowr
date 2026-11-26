//! `winnowr check-models`: checks a set of models before a run, without loading their boosters,
//! and reports what they need and every problem found (see [`crate::model_set`]), as text or as
//! a JSON summary of schema version [`SCHEMA_VERSION`].

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
use crate::model::{ModelInfo, ReferenceContigs, UnsupportedFormatVersion};
use crate::model_evidence::ModelEvidence;
use crate::model_set::{
    Check, ModelSet, Problem, ReadsCheck, SetInputs, Severity, check_model_set, check_region_beds,
    evidence_settings, lossy_path, merged_reference_contigs, models_alignment,
    models_feature_source,
};
use crate::pileup::PileupOptions;
use crate::policy::{NOT_DIFFICULT_REGIONS, is_indel_genotype};
use crate::regions::NamedBed;
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
    /// `--regions`.
    pub regions: Vec<RegionInput>,
    /// `--alignments`.
    #[serde(serialize_with = "lossy_path::option")]
    pub alignments: Option<PathBuf>,
    /// The reads' alignment provenance, if known.
    pub alignment: Option<AlignmentProvenance>,
    /// Where it came from: `header`, `tags` or `header+tags`.
    pub alignment_from: Option<&'static str>,
}

/// A region BED given.
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
}

/// Reads model files for [`check_model_set`]: each file's information, or the problem that
/// stopped it being read ([`Check::FormatVersion`] or [`Check::ModelUnreadable`]).
fn read_models(paths: &[PathBuf]) -> Vec<Result<ModelInfo, Problem>> {
    paths
        .par_iter()
        .map(|path| {
            ModelInfo::read(path).map_err(|error| {
                let check = if error.chain().any(|e| e.is::<UnsupportedFormatVersion>()) {
                    Check::FormatVersion
                } else {
                    Check::ModelUnreadable
                };
                Problem::error(check, Some(path), format!("{error:#}"))
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
    let (read, (reference, vcf)) = rayon::join(
        || read_models(inputs.models),
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
    let (paths, infos): (Vec<PathBuf>, Vec<&ModelInfo>) = inputs
        .models
        .iter()
        .zip(&read)
        .filter_map(|(path, info)| Some((path.clone(), info.as_ref().ok()?)))
        .unzip();
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
    let not_difficult = NotDifficultRegions {
        name: NOT_DIFFICULT_REGIONS.to_string(),
        used: infos.iter().any(|info| is_indel_genotype(info)),
        given: inputs.regions.iter().any(|bed| bed.name == NOT_DIFFICULT_REGIONS),
    };
    if not_difficult.used && !not_difficult.given {
        let message = format!(
            "--regions has no {NOT_DIFFICULT_REGIONS}=PATH, so score's default policy drops indel \
             calls at the model's threshold everywhere; pass it (GIAB's GRCh38 genome \
             stratification {NOT_DIFFICULT_REGIONS}.bed.gz) as score will be given it"
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
        unreadable_models: infos.len() < inputs.models.len(),
        reads: ReadsCheck {
            alignment: reads_alignment.as_ref(),
            unknown,
            strict: inputs.strict_provenance,
        },
    };
    let set = ModelSet { paths: &paths, infos: &infos };
    problems.extend(check_model_set(&set, &set_inputs));
    problems.extend(check_region_beds(&set, &set_inputs));
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
            regions: inputs
                .regions
                .iter()
                .map(|bed| RegionInput {
                    name: bed.name.clone(),
                    path: bed.path.clone(),
                    used_by: paths
                        .iter()
                        .zip(&infos)
                        .filter(|(_, info)| info.uses_region(&bed.name))
                        .map(|(path, _)| path.clone())
                        .collect(),
                })
                .collect(),
        },
        problems,
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
        let mut model = crate::testing::separable_model(Task::Genotype, variant_type);
        model.info.reference_contigs =
            Some([("chr1".to_string(), Some(1000))].into_iter().collect());
        model.info.alignment = Some(provenance("primary"));
        let path = dir.path().join(name);
        model.save(&path).unwrap();
        path
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
        assert_eq!(checks(&report), [Check::NotDifficultMissing, Check::ReferenceNotChecked]);
        assert!(report.ok);
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
        let read = read_models(&[old, cut, dir.path().join("a.json")]);
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
                "readable",
                "reference_contigs",
                "regenotype",
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
        assert_eq!(keys(&json["set"]["not_difficult_regions"]), ["given", "name", "used"]);
        assert_eq!(
            keys(&json["inputs"]),
            ["alignment", "alignment_from", "alignments", "reference", "regions", "vcf"]
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
        NotDifficultRegions { name: NOT_DIFFICULT_REGIONS.to_string(), used: false, given: false }
    }

    #[test]
    fn the_set_summary_is_null_where_the_models_disagree() {
        let mut a = recording(None);
        a.feature_source = FeatureSource::Vcf;
        let mut b = recording(None);
        b.task = Task::Promote;
        b.pileup_settings = Some(PileupOptions::default());
        let paths = [PathBuf::from("a.winnowr.json"), PathBuf::from("b.winnowr.json")];
        let summary = set_summary(&ModelSet { paths: &paths, infos: &[&a, &b] }, not_difficult());
        assert_eq!(summary.feature_source, None);
        assert_eq!(summary.alignment_settings, None);
        assert_eq!(summary.reference_contigs, None);
        assert_eq!(summary.pileup_settings, Some(PileupOptions::default().to_string()));
        assert_eq!(summary.needs, Needs { pileup: true, gvcf: true, ..Needs::default() });
        assert_eq!(summary.needs.names(), ["pileup", "gvcf"]);

        let summary = set_summary(&ModelSet { paths: &[], infos: &[] }, not_difficult());
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
        let mut snp = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        snp.info.reference_contigs = Some(lengths(&[("chr1", Some(1000))]));
        snp.info.region_names = vec!["segdups".to_string()];
        snp.info.alignment = Some(provenance("primary"));
        let snp_path = dir.path().join("snp.winnowr.json");
        snp.save(&snp_path).unwrap();
        let models = [snp_path.clone(), save(&dir, "indel.winnowr.json", VariantType::Indel)];
        let bed = |name: &str| {
            let path = dir.path().join(format!("{name}.bed"));
            std::fs::write(&path, "chr1\t0\t10\n").unwrap();
            NamedBed { name: name.to_string(), path }
        };
        let regions = [bed("segdups"), bed(NOT_DIFFICULT_REGIONS)];
        let report = check_models(&CheckInputs { regions: &regions, ..inputs(&models) });
        assert_eq!(checks(&report), [Check::ReferenceNotChecked]);
        assert_eq!(report.inputs.regions[0].used_by, [snp_path]);
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
        assert_eq!(report.problems.last().unwrap().model.as_deref(), Some(unknown_path.as_path()));
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
        let problem =
            report.problems.iter().find(|p| p.check == Check::AlignmentProvenanceMismatch).unwrap();
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
        let mismatch = &report.problems[report.problems.len() - 1];
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
