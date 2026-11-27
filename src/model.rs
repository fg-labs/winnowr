//! A trained model: an XGBoost multi-class booster bundled with winnowr's model information in one
//! `<name>.winnowr.json` file.
//!
//! The file is a JSON object holding [`ModelInfo`]'s fields (the ordered feature names, the variant
//! type and task, the cross-validated thresholds, the training samples, the reference contigs, the
//! stutter model of models with repeat-length features, and the versions that trained it) plus
//! `booster`, the XGBoost booster in its JSON format. Format version 3 is this format, and the
//! only one read.

use std::collections::BTreeMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use log::warn;
use serde::{Deserialize, Serialize};
use winnowr_xgboost::{Booster, DMatrix, SharedPredictor};

use crate::alignment_provenance::AlignmentProvenance;
use crate::common::{Task, VariantType};
use crate::feature_source::FeatureSource;
use crate::labels::{Label, PromoteLabel, UnfixableFlips};
use crate::pileup::PileupOptions;
use crate::pileup_features::PILEUP_FEATURES;
use crate::regions::RegionDigest;
use crate::rescore::{Mode, Probabilities};
use crate::strlen::StrlenOptions;
use crate::stutter::StutterModel;
use crate::table::FeatureTable;
use crate::thresholds::{PromoteThresholds, Thresholds};

/// The suffix of a model file.
pub const MODEL_SUFFIX: &str = ".winnowr.json";

/// The reference contigs a model or table records: each contig's length (`None` where unknown),
/// in order.
pub type ReferenceContigs = IndexMap<String, Option<u64>>;

/// The model format version this winnowr reads and writes.
pub const MODEL_FORMAT_VERSION: u32 = 3;

/// The key of the XGBoost booster in a model file.
const BOOSTER_KEY: &str = "booster";

/// Whether `train` refits the final model ([`TrainingParams::refit_final`]) unless told not to.
pub const DEFAULT_REFIT_FINAL: bool = true;

/// XGBoost training parameters. The defaults are deliberately regularized.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TrainingParams {
    /// Shrinkage per boosting round (`eta`).
    pub learning_rate: f64,
    /// The maximum leaves per tree (trees grow loss-guided).
    pub max_leaves: u32,
    /// The maximum tree depth (0 = unlimited, bounded by `max_leaves`).
    pub max_depth: u32,
    /// The minimum hessian sum per leaf.
    pub min_child_weight: f64,
    /// L2 regularization of leaf weights.
    pub reg_lambda: f64,
    /// The maximum boosting rounds.
    pub max_rounds: u32,
    /// Stop when the validation log-loss has not improved for this many rounds.
    pub early_stopping_rounds: u32,
    /// About the fraction of training calls held out for early stopping (0 disables early
    /// stopping), in whole genomic blocks (see
    /// [`validation_mask`](crate::fitting::validation_mask)).
    pub validation_fraction: f64,
    /// The size (bp) of the genomic blocks held out together for early stopping.
    pub validation_block_size: u64,
    /// The random seed (validation split and XGBoost).
    pub seed: u64,
    /// The XGBoost threads the model was trained with (0 = all cores).
    pub threads: u32,
    /// What training did with the flip labels a het <-> hom flip cannot fix (written only when
    /// not the default, so default models are unchanged).
    #[serde(skip_serializing_if = "UnfixableFlips::is_default")]
    pub unfixable_flips: UnfixableFlips,
    /// Whether the final model was refitted on all its training calls, those held out for early
    /// stopping included, for the rounds its early stopping chose. Written only when set, so a
    /// model file without it was not refitted: `train` refits by default
    /// ([`DEFAULT_REFIT_FINAL`]), but the default here stays false, which is what model files
    /// from before refitting existed mean.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub refit_final: bool,
    /// How the cross-validation that chose the thresholds formed its folds (written only when not
    /// the default, so default models are unchanged).
    #[serde(skip_serializing_if = "CvFolds::is_default")]
    pub cv_folds: CvFolds,
}

/// How cross-validation forms the folds whose held-out predictions choose the thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CvFolds {
    /// Each sample group is held out in turn (see [`crate::training::group_samples`]): a fold
    /// fits the other groups' calls.
    #[default]
    Samples,
    /// Groups of whole contigs are held out in turn, from every sample at once (see
    /// [`crate::training::contig_groups`]): a fold fits every sample's calls on the other contigs.
    Chromosomes {
        /// The number of folds (contig groups).
        folds: u32,
    },
}

impl CvFolds {
    /// True for the default, [`CvFolds::Samples`].
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == CvFolds::default()
    }
}

impl std::fmt::Display for CvFolds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CvFolds::Samples => f.write_str("samples"),
            CvFolds::Chromosomes { folds } => write!(f, "{folds} chromosome folds"),
        }
    }
}

impl Default for TrainingParams {
    fn default() -> Self {
        TrainingParams {
            learning_rate: 0.05,
            max_leaves: 31,
            max_depth: 0,
            min_child_weight: 10.0,
            reg_lambda: 1.0,
            max_rounds: 3000,
            early_stopping_rounds: 20,
            validation_fraction: 0.1,
            validation_block_size: 10_000,
            seed: 1,
            threads: 0,
            unfixable_flips: UnfixableFlips::Flip,
            refit_final: false,
            cv_folds: CvFolds::Samples,
        }
    }
}

/// The number of classes of a model: 4 for re-genotype models, 3 for other genotype models, or
/// the number of promote classes.
#[must_use]
pub fn num_classes(task: Task, regenotype: bool) -> usize {
    match task {
        Task::Promote => PromoteLabel::COUNT,
        Task::Genotype if regenotype => Label::Regenotype.index() + 1,
        Task::Genotype => Label::Regenotype.index(),
    }
}

/// What winnowr records about a model besides its booster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInfo {
    /// The model format version.
    pub format_version: u32,
    /// The variant type the model scores.
    pub variant_type: VariantType,
    /// What the model decides: `genotype` (scores PASS calls) or `promote` (non-PASS calls).
    pub task: Task,
    /// The ordered feature names.
    pub features: Vec<String>,
    /// The region sets the features use (`region_<name>`), which must be supplied under the
    /// same names when scoring.
    pub region_names: Vec<String>,
    /// The cross-validated thresholds per mode of a genotype model (empty for promote models).
    pub thresholds: BTreeMap<Mode, Thresholds>,
    /// The cross-validated thresholds of a promote model.
    pub promote_thresholds: Option<PromoteThresholds>,
    /// The training sample names.
    pub training_samples: Vec<String>,
    /// The contigs the training calls came from.
    pub contigs: Vec<String>,
    /// The lengths of `contigs` in the reference the training calls were called on (`None` when
    /// unknown).
    pub reference_contigs: Option<ReferenceContigs>,
    /// The training parameters.
    pub params: TrainingParams,
    /// The stutter model of a model with repeat-length features (see [`crate::stutter`]); its
    /// features need strlen evidence in the feature table.
    pub stutter: Option<StutterModel>,
    /// Whether the model has the re-genotype class ([`Label::Regenotype`]).
    pub regenotype: bool,
    /// The read settings the strlen evidence of the training tables was measured with (`None`
    /// without a stutter model, or for models that record none: see
    /// [`ModelInfo::strlen_options`]). The strlen evidence a model scores must be measured the same
    /// way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strlen_settings: Option<StrlenOptions>,
    /// The settings the allele read counts by MAPQ of the training tables were measured with, for
    /// a model with pileup features (see [`crate::pileup`]); `None` for a model without them.
    /// The pileup evidence a model scores must be measured the same way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pileup_settings: Option<PileupOptions>,
    /// Where the features of gVCF-mode annotations of the training tables came from (see
    /// [`FeatureSource`]); `score` derives the calls' features from the same source. Not written
    /// for the gVCF source, so that its models are as before, and models that record none are of
    /// the gVCF source.
    #[serde(default, skip_serializing_if = "FeatureSource::is_default")]
    pub feature_source: FeatureSource,
    /// How the training tables' reads were aligned (see [`crate::alignment_provenance`]), which
    /// `score` compares with the reads it scores. `None` when unknown, not written then, so that
    /// such models are as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<AlignmentProvenance>,
    /// The digest of each region BED the features use, by name (of `region_names`), as the
    /// training tables record it: `score` checks the BEDs it is given or finds against them.
    /// Empty when unknown, not written then, so that such models are as before.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub region_digests: BTreeMap<String, RegionDigest>,
    /// The digest of each region BED of the decision policy the model was trained to be scored
    /// with (`train --policy-regions`, today only the not-difficult regions), by name. Empty when
    /// none was given, not written then.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub policy_region_digests: BTreeMap<String, RegionDigest>,
    /// The winnowr version that trained the model.
    pub winnowr_version: String,
    /// The XGBoost version that trained the model.
    pub xgboost_version: String,
}

impl ModelInfo {
    /// The read settings of the strlen evidence a model with repeat-length features needs: those
    /// recorded, or [`StrlenOptions::LEGACY`] for models that record none.
    #[must_use]
    pub fn strlen_options(&self) -> Option<StrlenOptions> {
        self.stutter.as_ref().map(|_| self.strlen_settings.unwrap_or(StrlenOptions::LEGACY))
    }

    /// The number of classes the model predicts (see [`num_classes`]).
    #[must_use]
    pub fn num_classes(&self) -> usize {
        num_classes(self.task, self.regenotype)
    }

    /// Checks that the thresholds match the task.
    ///
    /// # Errors
    ///
    /// Returns an error if a genotype model lacks thresholds for a mode or has promote
    /// thresholds, a promote model lacks promote thresholds or has genotype thresholds, or a
    /// re-genotype model is not a genotype model with a stutter model.
    pub fn validate(&self) -> Result<()> {
        if self.regenotype && (self.task != Task::Genotype || self.stutter.is_none()) {
            bail!("a re-genotype model must be a genotype model with a stutter model");
        }
        if self.strlen_settings.is_some() && self.stutter.is_none() {
            bail!("only a model with a stutter model has strlen read settings");
        }
        let has_pileup_features =
            self.features.iter().any(|feature| PILEUP_FEATURES.contains(&feature.as_str()));
        if has_pileup_features != self.pileup_settings.is_some() {
            bail!("a model has pileup settings if and only if it has pileup features");
        }
        match self.task {
            Task::Genotype => {
                let missing: Vec<&str> = Mode::ALL
                    .iter()
                    .filter(|mode| !self.thresholds.contains_key(mode))
                    .map(|mode| mode.as_str())
                    .collect();
                if !missing.is_empty() || self.promote_thresholds.is_some() {
                    bail!(
                        "a genotype model needs thresholds for every mode (missing: {}) and no \
                         promote thresholds",
                        missing.join(", ")
                    );
                }
            }
            Task::Promote => {
                if self.promote_thresholds.is_none() || !self.thresholds.is_empty() {
                    bail!("a promote model needs promote thresholds and no genotype thresholds");
                }
            }
        }
        Ok(())
    }
}

/// The error of a model file of a format version this winnowr does not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedFormatVersion {
    /// The file's format version, if it records one.
    pub format_version: Option<u64>,
    /// The winnowr version that wrote it, if it records one.
    pub winnowr_version: Option<String>,
}

impl std::fmt::Display for UnsupportedFormatVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported model format version {} (written by winnowr {}); this winnowr reads \
             version {MODEL_FORMAT_VERSION}: train new models with this winnowr",
            self.format_version.map_or_else(|| "none".to_string(), |v| v.to_string()),
            self.winnowr_version.as_deref().unwrap_or("unknown")
        )
    }
}

impl std::error::Error for UnsupportedFormatVersion {}

impl ModelInfo {
    /// The information of a model file's text, and the raw text of its booster, which is scanned
    /// past but not parsed (see [`split_member`]): the booster's members may come in any order,
    /// and a file cut short is an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is not a JSON object with a booster, is not of
    /// [`MODEL_FORMAT_VERSION`] (an [`UnsupportedFormatVersion`]), has unknown or invalid fields,
    /// or fails [`ModelInfo::validate`].
    pub fn from_model_json(text: &str) -> Result<(ModelInfo, &str)> {
        // The booster is handed to XGBoost as written: it may hold tokens such as `NaN` that are
        // not standard JSON, and parsing a large model into a JSON tree would be wasteful.
        let (info_text, booster_json) = split_member(text, BOOSTER_KEY)
            .context("the model is not a JSON object with a booster")?;
        /// The fields read before the rest, to explain an unsupported version.
        #[derive(Deserialize)]
        struct Versions {
            format_version: Option<u64>,
            winnowr_version: Option<String>,
        }
        let versions: Versions =
            serde_json::from_str(&info_text).context("invalid model information")?;
        if versions.format_version != Some(u64::from(MODEL_FORMAT_VERSION)) {
            return Err(UnsupportedFormatVersion {
                format_version: versions.format_version,
                winnowr_version: versions.winnowr_version,
            }
            .into());
        }
        let info: ModelInfo =
            serde_json::from_str(&info_text).context("invalid model information")?;
        info.validate()?;
        Ok((info, booster_json))
    }

    /// Reads the information of the model file `path` (see [`ModelInfo::read_file`]).
    ///
    /// # Errors
    ///
    /// See [`ModelInfo::read_file`].
    #[cfg(test)]
    pub fn read<P: AsRef<Path>>(path: P) -> Result<ModelInfo> {
        ModelInfo::read_file(&ModelFile::resolve(path.as_ref()))
    }

    /// Reads the information of a model file, from the file it resolved to. Its booster is scanned
    /// past, never parsed or loaded (see [`ModelInfo::from_model_json`]), so it is not checked.
    ///
    /// # Errors
    ///
    /// Returns an error if the path given does not end in `.winnowr.json`, or the file cannot be
    /// read or fails [`ModelInfo::from_model_json`].
    pub fn read_file(file: &ModelFile) -> Result<ModelInfo> {
        let text = file.read_text()?;
        let (info, _) = ModelInfo::from_model_json(&text)
            .with_context(|| format!("failed to load model {}", file.path.display()))?;
        Ok(info)
    }

    /// True if the model's features use the region set `name` (its `--regions` BED).
    #[must_use]
    pub fn uses_region(&self, name: &str) -> bool {
        self.region_names.iter().any(|region| region == name)
    }

    /// The warnings about a model file whose information this is: it was trained by another
    /// winnowr version, or records no reference contigs (so the reference of the calls cannot be
    /// checked against it).
    #[must_use]
    pub fn warnings(&self, path: &Path) -> ModelWarnings {
        ModelWarnings {
            other_version: other_version_warning(path, &self.winnowr_version),
            no_reference_contigs: self.reference_contigs.is_none().then(|| {
                format!(
                    "{} records no reference contigs, so the reference of the calls cannot be \
                     checked against it",
                    path.display()
                )
            }),
        }
    }
}

/// The warnings about a model file (see [`ModelInfo::warnings`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelWarnings {
    /// It was trained by another winnowr version.
    pub other_version: Option<String>,
    /// It records no reference contigs.
    pub no_reference_contigs: Option<String>,
}

impl ModelWarnings {
    /// Logs the warnings.
    pub fn log(&self) {
        for warning in [&self.other_version, &self.no_reference_contigs].into_iter().flatten() {
            warn!("{warning}");
        }
    }
}

/// A model file: the path given, and the file and the directory it resolved to (with their
/// symlinks followed), resolved once. Pipelines stage models as symlinks, which may be retargeted
/// while a command runs; reading the model from, and looking for its region BEDs beside, the one
/// resolution keeps the model and its BEDs of one bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    /// The path given, which messages name.
    pub path: PathBuf,
    /// The file it resolved to: its canonical path, or the unresolved path for diagnostics only
    /// if resolution failed (reading then fails without following that path later).
    pub resolved: PathBuf,
    /// The directory of the path given, resolved with it: its canonical path. If it cannot be
    /// resolved, this holds the given directory for diagnostics and reading the model fails.
    pub given_dir: PathBuf,
    /// A failure to resolve the parent or filename: never retry through unresolved paths.
    resolution_error: Option<String>,
}

impl ModelFile {
    /// Resolves the model file `path`.
    #[must_use]
    pub fn resolve(path: &Path) -> Self {
        let dir = path.parent().unwrap_or_else(|| Path::new(""));
        // The directory of a bare file name is the working directory, which `""` does not name.
        let named = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
        let given_dir = match named.canonicalize() {
            Ok(dir) => dir,
            Err(error) => {
                return ModelFile {
                    path: path.to_path_buf(),
                    resolved: path.to_path_buf(),
                    given_dir: dir.to_path_buf(),
                    resolution_error: Some(format!(
                        "cannot resolve model parent directory {}: {error}",
                        named.display()
                    )),
                };
            }
        };
        // Resolve the filename through the captured parent so retargeting a staging directory
        // cannot pair a model from one bundle with the BEDs of another.
        let captured =
            path.file_name().map_or_else(|| path.to_path_buf(), |name| given_dir.join(name));
        let (resolved, resolution_error) = match captured.canonicalize() {
            Ok(resolved) => (resolved, None),
            Err(error) => {
                let reason = format!("cannot resolve model file {}: {error}", captured.display());
                (captured, Some(reason))
            }
        };
        ModelFile { path: path.to_path_buf(), resolved, given_dir, resolution_error }
    }

    /// Resolves each of the model files `paths`, in order.
    #[must_use]
    pub fn resolve_all(paths: &[PathBuf]) -> Vec<Self> {
        paths.iter().map(|path| ModelFile::resolve(path)).collect()
    }

    /// The text of the model file, read from the file it resolved to, after checking the path given
    /// ends in [`MODEL_SUFFIX`] (the file a symlink names need not).
    ///
    /// # Errors
    ///
    /// Returns an error naming the path given if it does not end in `.winnowr.json`, or the file
    /// cannot be read.
    pub fn read_text(&self) -> Result<String> {
        self.read_text_with_metadata().map(|(text, _)| text)
    }

    /// The text and metadata of the model file, from the same open file, so a later path
    /// replacement cannot hide the identity of the model that was read.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ModelFile::read_text`], or an error reading the opened file's
    /// metadata.
    pub fn read_text_with_metadata(&self) -> Result<(String, fs::Metadata)> {
        use std::io::Read as _;
        let path = &self.path;
        if !path.to_string_lossy().ends_with(MODEL_SUFFIX) {
            bail!("a model file must end in {MODEL_SUFFIX}: {}", path.display());
        }
        if let Some(error) = &self.resolution_error {
            bail!("failed to read model {}: {error}", path.display());
        }
        let context = || format!("failed to read model {}", path.display());
        let mut file = fs::File::open(&self.resolved).with_context(context)?;
        let metadata = file.metadata().with_context(context)?;
        let mut text = String::new();
        file.read_to_string(&mut text).with_context(context)?;
        Ok((text, metadata))
    }
}

/// A trained booster with its model information.
#[derive(Debug)]
pub struct WinnowrModel {
    /// The XGBoost booster.
    pub booster: Booster,
    /// The model information.
    pub info: ModelInfo,
}

impl WinnowrModel {
    /// Loads the model file `path` (see [`WinnowrModel::load_file`]).
    ///
    /// # Errors
    ///
    /// See [`WinnowrModel::load_file`].
    #[cfg(test)]
    pub fn load<P: AsRef<Path>>(path: P, threads: NonZeroUsize) -> Result<Self> {
        Self::load_file(&ModelFile::resolve(path.as_ref()), threads)
    }

    /// Loads a model file, from the file it resolved to, predicting with at most `threads`
    /// threads. Its information is read as [`ModelInfo::read_file`] reads it;
    /// [`ModelInfo::warnings`] are not logged.
    ///
    /// # Errors
    ///
    /// Returns an error if the path given does not end in `.winnowr.json`, the file cannot be
    /// read, is not of a supported format version, has unknown or invalid fields, or its booster
    /// does not load or does not have the model's features and classes.
    pub fn load_file(file: &ModelFile, threads: NonZeroUsize) -> Result<Self> {
        let text = file.read_text()?;
        Self::from_json(&text, threads)
            .with_context(|| format!("failed to load model {}", file.path.display()))
    }

    /// A model from the text of a model file.
    ///
    /// # Errors
    ///
    /// See [`WinnowrModel::load_file`].
    pub fn from_json(text: &str, threads: NonZeroUsize) -> Result<Self> {
        let (info, booster_json) = ModelInfo::from_model_json(text)?;
        WinnowrModel::from_parts(info, booster_json, threads)
    }

    /// A model from its information and the raw text of its booster (as
    /// [`ModelInfo::from_model_json`] splits a model file's text), predicting with at most
    /// `threads` threads.
    ///
    /// # Errors
    ///
    /// Returns an error if the booster does not load or does not have the model's features and
    /// classes.
    pub fn from_parts(info: ModelInfo, booster_json: &str, threads: NonZeroUsize) -> Result<Self> {
        let booster = Booster::from_json(booster_json.as_bytes(), threads)
            .context("invalid XGBoost booster")?;
        let model = WinnowrModel { booster, info };
        model.check_booster()?;
        Ok(model)
    }

    /// The model file's text: the model information, then the booster as XGBoost wrote it.
    ///
    /// # Errors
    ///
    /// Returns an error if the booster cannot be serialized.
    pub fn to_json(&self) -> Result<String> {
        let info = serde_json::to_string(&self.info)?;
        let booster =
            String::from_utf8(self.booster.to_json()?).context("XGBoost wrote non-UTF-8 JSON")?;
        let fields =
            info.strip_suffix('}').context("the model information is not a JSON object")?;
        Ok(format!("{fields},\"{BOOSTER_KEY}\":{booster}}}"))
    }

    /// Writes the model file: to a temporary file beside it, then renamed over it, so that a
    /// failed write (e.g. a full disk) leaves any earlier file at the path whole.
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not end in `.winnowr.json`, or the file cannot be
    /// written.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.to_string_lossy().ends_with(MODEL_SUFFIX) {
            bail!("a model file must end in {MODEL_SUFFIX}: {}", path.display());
        }
        let json = self.to_json()?;
        let dir = path.parent().filter(|dir| !dir.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let write = || -> std::io::Result<()> {
            use std::io::Write;
            use std::os::unix::fs::PermissionsExt;
            // Created as fs::write creates a file (read-write for all, less the umask), not
            // with a temporary file's owner-only permissions.
            let mut file = tempfile::Builder::new()
                .prefix(".winnowr-model-")
                .suffix(".tmp")
                .permissions(fs::Permissions::from_mode(0o666))
                .tempfile_in(dir)?;
            file.write_all(json.as_bytes())?;
            file.as_file().sync_all()?;
            file.persist(path).map_err(|e| e.error)?;
            Ok(())
        };
        write().with_context(|| format!("failed to write model {}", path.display()))
    }

    /// Checks that the booster has the model's features, in order, and its number of classes.
    fn check_booster(&self) -> Result<()> {
        let names = self.booster.feature_names()?;
        if names != self.info.features {
            bail!(
                "the booster's features ({}) are not the model's ({})",
                names.join(", "),
                self.info.features.join(", ")
            );
        }
        // Read from the booster's config: predicting an empty matrix would report it too, but
        // XGBoost then warns "Empty dataset".
        let classes = self.booster.outputs_per_row()?;
        if classes != self.info.num_classes() {
            bail!(
                "the booster predicts {classes} classes, but a {} model has {}",
                self.info.task,
                self.info.num_classes()
            );
        }
        Ok(())
    }

    /// Predicts class probabilities for a feature table's calls, in one XGBoost call with the
    /// booster's threads ([`Predicting::Whole`]).
    ///
    /// The probabilities are in [`Label`] order (genotype models, with the re-genotype column for
    /// re-genotype models) or [`PromoteLabel`] order (promote models).
    ///
    /// # Errors
    ///
    /// Returns an error if the table lacks a model feature, or XGBoost fails.
    pub fn predict(&self, table: &FeatureTable) -> Result<Probabilities> {
        self.predict_with(table, Predicting::Whole)
    }

    /// Predicts class probabilities for a feature table's calls as [`WinnowrModel::predict`]
    /// does, `predicting` them in one XGBoost call or in chunks on the current rayon pool. The
    /// probabilities are the same either way.
    ///
    /// # Errors
    ///
    /// Returns an error if the table lacks a model feature, XGBoost fails, or
    /// [`Predicting::Chunked`] is asked of a booster of more than one thread.
    pub fn predict_with(
        &self,
        table: &FeatureTable,
        predicting: Predicting,
    ) -> Result<Probabilities> {
        let classes = self.info.num_classes();
        if table.num_rows() == 0 {
            return Ok(Probabilities::new(Vec::new(), classes));
        }
        let values = to_matrix(table, &self.info.features)?;
        let (rows, features) = (table.num_rows(), &self.info.features);
        // Only a tree booster predicts from several threads at once; another predicts in one
        // call, with its one thread.
        let shared = match predicting {
            Predicting::Whole => None,
            Predicting::Chunked => {
                anyhow::ensure!(
                    self.booster.threads() == NonZeroUsize::MIN,
                    "chunked prediction needs a booster of one thread, not {}",
                    self.booster.threads()
                );
                self.booster.shared_predictor()?
            }
        };
        match shared {
            Some(shared) => predict_probabilities_chunked(
                shared,
                &values,
                rows,
                features,
                classes,
                PREDICT_CHUNK_ROWS,
            ),
            None => predict_probabilities(&self.booster, &values, rows, features, classes),
        }
    }
}

/// The warning that a model at `path` was trained by another winnowr version, if it was.
///
/// Only the package versions are compared, without the commit a git build adds (see
/// [`crate::version::package_version`]), so a release built from a checkout and from its source
/// tarball read each other's models without a warning.
fn other_version_warning(path: &Path, trained_with: &str) -> Option<String> {
    use crate::version::package_version;
    let this_version = crate::version::VERSION.as_str();
    (package_version(trained_with) != package_version(this_version)).then(|| {
        format!(
            "{} was trained with winnowr {trained_with}; this is winnowr {this_version}",
            path.display()
        )
    })
}

/// Splits a JSON object's text into the text without its top-level member `key`, and the raw
/// text of that member's value. Values are scanned, not parsed, so they may hold non-standard
/// tokens (e.g. XGBoost's `NaN`). Returns `None` if the text is not an object or has no such
/// member.
fn split_member<'a>(text: &'a str, key: &str) -> Option<(String, &'a str)> {
    let bytes = text.as_bytes();
    let skip_ws = |mut i: usize| {
        while bytes.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        i
    };
    let mut i = skip_ws(0);
    if bytes.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    loop {
        let member_start = i;
        i = skip_ws(i);
        if bytes.get(i) != Some(&b'"') {
            return None;
        }
        let key_end = end_of_string(bytes, i)?;
        let name: String = serde_json::from_str(&text[i..key_end]).ok()?;
        i = skip_ws(key_end);
        if bytes.get(i) != Some(&b':') {
            return None;
        }
        let value_start = skip_ws(i + 1);
        let value_end = end_of_value(bytes, value_start)?;
        i = skip_ws(value_end);
        let delimiter = *bytes.get(i)?;
        if name == key {
            let value = &text[value_start..value_end];
            let rest = if delimiter == b',' {
                format!("{}{}", &text[..member_start], &text[i + 1..])
            } else {
                // The last member: drop the comma before it, if any.
                let before = text[..member_start].trim_end();
                format!("{}{}", before.strip_suffix(',').unwrap_or(before), &text[i..])
            };
            return Some((rest, value));
        }
        if delimiter != b',' {
            return None;
        }
        i += 1;
    }
}

/// The index just past the JSON string starting at `start` (a `"`).
fn end_of_string(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// The index just past the JSON value starting at `start`.
fn end_of_value(bytes: &[u8], start: usize) -> Option<usize> {
    match bytes.get(start)? {
        b'"' => end_of_string(bytes, start),
        b'{' | b'[' => {
            let mut depth = 0usize;
            let mut i = start;
            while i < bytes.len() {
                match bytes[i] {
                    b'"' => {
                        i = end_of_string(bytes, i)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            let length = bytes[start..]
                .iter()
                .position(|b| matches!(b, b',' | b'}' | b']') || b.is_ascii_whitespace())
                .unwrap_or(bytes.len() - start);
            (length > 0).then_some(start + length)
        }
    }
}

/// Selects features as a row-major `f32` matrix; infinities become NaN (missing).
///
/// # Errors
///
/// Returns an error naming the features the table lacks.
pub fn to_matrix<S: AsRef<str>>(table: &FeatureTable, features: &[S]) -> Result<Vec<f32>> {
    let missing: Vec<&str> =
        features.iter().map(AsRef::as_ref).filter(|name| !table.has_column(name)).collect();
    if !missing.is_empty() {
        bail!("the feature table lacks model features: {}", missing.join(", "));
    }
    let width = features.len();
    let rows = table.num_rows();
    let mut matrix = vec![f32::NAN; rows * width];
    // Filled a few columns at a time (a column with nulls is copied, so not all at once), a block
    // of rows at a time, so that the writes stay in the block's cache lines rather than striding
    // the whole matrix once per column.
    const GROUP_COLUMNS: usize = 8;
    const BLOCK_ROWS: usize = 4096;
    for (group, names) in features.chunks(GROUP_COLUMNS).enumerate() {
        let columns: Vec<_> =
            names.iter().map(|name| table.f64_column(name.as_ref())).collect::<Result<_>>()?;
        for first in (0..rows).step_by(BLOCK_ROWS) {
            let last = (first + BLOCK_ROWS).min(rows);
            for (offset, values) in columns.iter().enumerate() {
                let column = group * GROUP_COLUMNS + offset;
                for (row, &value) in (first..last).zip(&values[first..last]) {
                    let value = value as f32;
                    matrix[row * width + column] =
                        if value.is_infinite() { f32::NAN } else { value };
                }
            }
        }
    }
    Ok(matrix)
}

/// A dense matrix of `rows` rows of the features, with their names, copied with `threads` threads.
pub fn named_matrix<S: AsRef<str>>(
    values: &[f32],
    rows: usize,
    features: &[S],
    threads: NonZeroUsize,
) -> Result<DMatrix> {
    let mut matrix =
        DMatrix::from_dense_with_threads(values, rows, features.len(), f32::NAN, threads)?;
    matrix.set_feature_names(features)?;
    Ok(matrix)
}

/// How a model predicts its calls' probabilities: the probabilities are the same either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predicting {
    /// In one XGBoost call, with the booster's threads (OpenMP).
    Whole,
    /// In chunks of [`PREDICT_CHUNK_ROWS`] calls, each a task on the current rayon pool calling
    /// XGBoost with the booster's threads, which must then be one: idle threads of the pool,
    /// e.g. those of other work sharing it, take up chunks as they free up. A booster that is not
    /// a tree booster (e.g. `gblinear`), which cannot predict from several threads at once,
    /// predicts in one call with its one thread.
    Chunked,
}

/// The calls in a chunk of [`Predicting::Chunked`] prediction: a fraction of a second of work.
pub const PREDICT_CHUNK_ROWS: usize = 8192;

/// Predicts the class probabilities of `rows` calls with a booster.
///
/// # Errors
///
/// Returns an error if XGBoost fails or predicts another shape.
pub fn predict_probabilities<S: AsRef<str>>(
    booster: &Booster,
    values: &[f32],
    rows: usize,
    features: &[S],
    num_class: usize,
) -> Result<Probabilities> {
    if rows == 0 {
        return Ok(Probabilities::new(Vec::new(), num_class));
    }
    let mut timer = crate::common::PhaseTimer::start();
    let matrix = named_matrix(values, rows, features, booster.threads())?;
    timer.lap("building the XGBoost matrix");
    let prediction = booster.predict(&matrix)?;
    timer.lap("running the XGBoost prediction");
    if prediction.shape != [rows, num_class] {
        bail!("XGBoost predicted {:?} values for {rows} calls", prediction.shape);
    }
    Ok(Probabilities::new(prediction.values.into_iter().map(f64::from).collect(), num_class))
}

/// [`predict_probabilities`] in chunks of `chunk_rows` calls (see [`Predicting::Chunked`]),
/// predicted in parallel on the current rayon pool and put together in row order.
///
/// # Errors
///
/// Returns an error if the booster has more than one thread, there are no features or the values
/// are not `rows` of them, or XGBoost fails or predicts another shape for a chunk.
fn predict_probabilities_chunked<S: AsRef<str> + Sync>(
    shared: SharedPredictor<'_>,
    values: &[f32],
    rows: usize,
    features: &[S],
    num_class: usize,
    chunk_rows: usize,
) -> Result<Probabilities> {
    use rayon::prelude::*;
    // Each of the pool's threads may predict a chunk at once.
    anyhow::ensure!(
        shared.threads() == NonZeroUsize::MIN,
        "chunked prediction needs a booster of one thread, not {}",
        shared.threads()
    );
    if rows == 0 {
        return Ok(Probabilities::new(Vec::new(), num_class));
    }
    let width = features.len();
    anyhow::ensure!(width > 0, "a model of no features cannot predict");
    anyhow::ensure!(
        values.len() == rows * width,
        "{} values are not {rows} calls of {width} features",
        values.len()
    );
    let mut timer = crate::common::PhaseTimer::start();
    let chunks: Vec<Vec<f32>> = values
        .par_chunks(chunk_rows * width)
        // Each chunk is a job of its own, which any idle thread of the pool can take.
        .with_max_len(1)
        .map(|chunk| -> Result<Vec<f32>> {
            #[cfg(test)]
            let _running =
                crate::model_evidence::work_observer::prediction(features, shared.threads().get());
            let chunk_rows = chunk.len() / width;
            let matrix = named_matrix(chunk, chunk_rows, features, NonZeroUsize::MIN)?;
            let prediction = shared.predict(&matrix)?;
            if prediction.shape != [chunk_rows, num_class] {
                bail!("XGBoost predicted {:?} values for {chunk_rows} calls", prediction.shape);
            }
            Ok(prediction.values)
        })
        .collect::<Result<_>>()?;
    timer.lap("running the XGBoost prediction in chunks");
    let values = chunks.into_iter().flatten().map(f64::from).collect();
    Ok(Probabilities::new(values, num_class))
}

/// Checks that calls come from the reference a model was trained on (its contig names and
/// lengths).
///
/// A model without reference contigs is not checked; lengths are compared only where both are
/// known.
///
/// # Arguments
///
/// * `model` - the model file (for the message).
/// * `info` - the model's information.
/// * `contigs` - the contig lengths of the calls' reference (e.g. from their VCF header).
/// * `source` - where `contigs` came from (for the message).
/// * `require_all` - whether every model contig must be in `contigs`.
///
/// # Errors
///
/// Returns an error if a model contig has another length in `contigs`, or (with `require_all`) is
/// missing from it.
pub fn check_reference(
    model: &Path,
    info: &ModelInfo,
    contigs: &ReferenceContigs,
    source: &str,
    require_all: bool,
) -> Result<()> {
    let Some(reference) = &info.reference_contigs else {
        return Ok(());
    };
    let missing: Vec<&str> = reference
        .keys()
        .filter(|name| require_all && !contigs.contains_key(*name))
        .map(String::as_str)
        .collect();
    let differ: Vec<String> = reference
        .iter()
        .filter_map(|(name, &length)| {
            let (Some(length), Some(Some(theirs))) = (length, contigs.get(name)) else {
                return None;
            };
            (*theirs != length).then(|| format!("{name} ({theirs} bp, model {length} bp)"))
        })
        .collect();
    if missing.is_empty() && differ.is_empty() {
        return Ok(());
    }
    let mut problems = Vec::new();
    if !missing.is_empty() {
        problems.push(format!("missing contigs {}", missing.join(", ")));
    }
    if !differ.is_empty() {
        problems.push(format!("other lengths: {}", differ.join(", ")));
    }
    bail!(
        "{source} is not on the reference model {} was trained on: {}",
        model.display(),
        problems.join("; ")
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use rstest::rstest;

    use super::*;
    use crate::common::NEVER;
    use crate::table::FeatureTableInfo;

    const THREADS: NonZeroUsize = NonZeroUsize::new(1).expect("1 is non-zero");

    /// Even a missing filename stays anchored to the captured staging directory, rather than
    /// becoming readable from a different bundle after that directory symlink is retargeted.
    #[cfg(unix)]
    #[test]
    fn an_unresolved_model_filename_keeps_its_captured_parent() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        fs::create_dir(&a).unwrap();
        fs::create_dir(&b).unwrap();
        let staged = dir.path().join("staged");
        std::os::unix::fs::symlink(&a, &staged).unwrap();
        let model = ModelFile::resolve(&staged.join("model.winnowr.json"));
        fs::write(b.join("model.winnowr.json"), "another bundle").unwrap();
        fs::remove_file(&staged).unwrap();
        std::os::unix::fs::symlink(&b, &staged).unwrap();
        assert!(model.read_text().is_err(), "missing model was read from the new bundle");
        assert_eq!(model.resolved, a.canonicalize().unwrap().join("model.winnowr.json"));
    }

    #[cfg(unix)]
    #[test]
    fn an_unresolved_model_parent_never_becomes_readable_later() {
        let dir = tempfile::TempDir::new().unwrap();
        let missing = dir.path().join("missing");
        let staged = dir.path().join("staged");
        std::os::unix::fs::symlink(&missing, &staged).unwrap();
        let file = ModelFile::resolve(&staged.join("model.winnowr.json"));
        fs::create_dir(&missing).unwrap();
        fs::write(missing.join("model.winnowr.json"), "new bundle").unwrap();
        assert!(file.read_text().is_err(), "an unresolved parent was followed later");
    }

    /// A failed filename resolution stays a failure even if the target is published later.
    #[cfg(unix)]
    #[rstest]
    #[case::published_later(false)]
    #[case::retargeted(true)]
    fn review_unresolved_model_filename_cannot_be_followed_later(#[case] retargeted: bool) {
        let dir = tempfile::TempDir::new().unwrap();
        let staged = dir.path().join("stage");
        fs::create_dir(&staged).unwrap();
        let missing = dir.path().join("unpublished.json");
        let path = staged.join("model.winnowr.json");
        std::os::unix::fs::symlink(&missing, &path).unwrap();
        let file = ModelFile::resolve(&path);
        if retargeted {
            let other = dir.path().join("another-bundle.json");
            fs::write(&other, "other model").unwrap();
            fs::remove_file(&path).unwrap();
            std::os::unix::fs::symlink(other, &path).unwrap();
        } else {
            fs::write(&missing, "newly published model").unwrap();
        }
        let error = file.read_text().unwrap_err().to_string();
        assert!(error.contains(&path.display().to_string()), "{error}");
        assert!(error.contains("cannot resolve model file"), "{error}");
    }

    /// A captured directory still follows a symlink on the model filename itself.
    #[cfg(unix)]
    #[test]
    fn resolving_a_model_follows_its_filename_symlink() {
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("target.json");
        fs::write(&target, "model text").unwrap();
        let staged = dir.path().join("staged");
        fs::create_dir(&staged).unwrap();
        let path = staged.join("model.winnowr.json");
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let file = ModelFile::resolve(&path);
        assert_eq!(file.given_dir, staged.canonicalize().unwrap());
        assert_eq!(file.resolved, target.canonicalize().unwrap());
        assert_eq!(file.read_text().unwrap(), "model text");
    }

    /// A table with Float64 columns.
    fn table(columns: &[(&str, Vec<f64>)]) -> FeatureTable {
        let fields: Vec<Field> =
            columns.iter().map(|(name, _)| Field::new(*name, DataType::Float64, true)).collect();
        let arrays = columns
            .iter()
            .map(|(_, values)| {
                Arc::new(Float64Array::from(values.clone())) as arrow_array::ArrayRef
            })
            .collect();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let info = FeatureTableInfo { sample: "S".to_string(), ..FeatureTableInfo::default() };
        FeatureTable::new(batch, info)
    }

    /// Trains a tiny three-class booster on features `a` and `b`.
    fn booster(features: &[&str], classes: usize) -> Booster {
        booster_of("gbtree", features, classes)
    }

    /// A booster of kind `kind` (e.g. `gbtree`, `gblinear`) trained on 30 rows of `features`.
    fn booster_of(kind: &str, features: &[&str], classes: usize) -> Booster {
        let rows = 30;
        let values: Vec<f32> = (0..rows * features.len()).map(|i| (i % 7) as f32).collect();
        let labels: Vec<f32> = (0..rows).map(|i| (i % classes) as f32).collect();
        let mut train = DMatrix::from_dense(&values, rows, features.len(), f32::NAN).unwrap();
        train.set_labels(&labels).unwrap();
        train.set_feature_names(features).unwrap();
        let mut booster = Booster::new(&[&train], THREADS).unwrap();
        let classes_value = classes.to_string();
        booster
            .set_params(&[
                ("booster", kind),
                ("objective", "multi:softprob"),
                ("num_class", &classes_value),
            ])
            .unwrap();
        for iteration in 0..3 {
            booster.update(iteration, &train).unwrap();
        }
        booster
    }

    fn genotype_info(features: &[&str]) -> ModelInfo {
        let thresholds = Thresholds { t_drop: 0.5, t_flip: 0.6, gain: 0.01, t_regenotype: NEVER };
        ModelInfo {
            format_version: MODEL_FORMAT_VERSION,
            variant_type: VariantType::Snp,
            task: Task::Genotype,
            features: features.iter().map(ToString::to_string).collect(),
            region_names: vec![],
            thresholds: Mode::ALL.iter().map(|&mode| (mode, thresholds)).collect(),
            promote_thresholds: None,
            training_samples: vec!["HG001".to_string()],
            contigs: vec!["chr1".to_string()],
            reference_contigs: Some(IndexMap::from([("chr1".to_string(), Some(1000))])),
            params: TrainingParams::default(),
            stutter: None,
            regenotype: false,
            strlen_settings: None,
            pileup_settings: None,
            feature_source: FeatureSource::Gvcf,
            alignment: None,
            region_digests: BTreeMap::new(),
            policy_region_digests: BTreeMap::new(),
            winnowr_version: crate::version::VERSION.clone(),
            xgboost_version: "3.4.2".to_string(),
        }
    }

    /// The matrix is the features' values row by row, in the features' order, whatever the
    /// blocks and column groups it is filled in.
    #[test]
    fn to_matrix_is_the_features_row_by_row() {
        let (rows, width) = (4097, 10);
        let names: Vec<String> = (0..width).map(|c| format!("f{c}")).collect();
        let value = |row: usize, column: usize| match (row + column) % 7 {
            0 => f64::INFINITY,
            1 => f64::NEG_INFINITY,
            2 => f64::NAN,
            _ => (row * 31 + column) as f64 / 3.0,
        };
        let columns: Vec<(&str, Vec<f64>)> = names
            .iter()
            .enumerate()
            .map(|(c, name)| (name.as_str(), (0..rows).map(|r| value(r, c)).collect()))
            .collect();
        let table = table(&columns);
        // The features in another order than the table's columns.
        let order: Vec<usize> = (0..width).rev().collect();
        let features: Vec<&str> = order.iter().map(|&c| names[c].as_str()).collect();
        let matrix = to_matrix(&table, &features).unwrap();
        assert_eq!(matrix.len(), rows * width);
        for row in 0..rows {
            for (i, &column) in order.iter().enumerate() {
                let expected = value(row, column) as f32;
                let expected = if expected.is_infinite() { f32::NAN } else { expected };
                let got = matrix[row * width + i];
                assert_eq!(got.to_bits(), expected.to_bits(), "row {row}, feature {i}");
            }
        }
    }

    #[test]
    fn class_counts() {
        assert_eq!(num_classes(Task::Genotype, false), 3);
        assert_eq!(num_classes(Task::Genotype, true), 4);
        assert_eq!(num_classes(Task::Promote, false), 3);
    }

    #[test]
    fn saves_loads_and_predicts() {
        let model =
            WinnowrModel { booster: booster(&["a", "b"], 3), info: genotype_info(&["a", "b"]) };
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("snp.winnowr.json");
        model.save(&path).unwrap();
        let loaded = WinnowrModel::load(&path, THREADS).unwrap();
        assert_eq!(loaded.info, model.info);

        // Columns are selected by name, so the table's column order does not matter.
        let calls =
            table(&[("b", vec![1.0, f64::INFINITY]), ("a", vec![2.0, 3.0]), ("c", vec![0.0, 0.0])]);
        let got = loaded.predict(&calls).unwrap();
        let expected = model.predict(&calls).unwrap();
        assert_eq!(got, expected);
        assert_eq!((got.len(), got.classes()), (2, 3));
        assert!((got.row(0).iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(loaded.predict(&calls.filter(&[false, false]).unwrap()).unwrap().is_empty());

        let message = loaded.predict(&table(&[("a", vec![1.0])])).unwrap_err().to_string();
        assert_eq!(message, "the feature table lacks model features: b");
        // The suffix is checked before reading: snp.json does not exist either.
        let wrong = dir.path().join("snp.json");
        let message = WinnowrModel::load(&wrong, THREADS).unwrap_err().to_string();
        assert_eq!(
            message,
            format!("a model file must end in {MODEL_SUFFIX}: {}", wrong.display())
        );
    }

    /// The calls of a table of the features `a` and `b`, `rows` of them.
    fn two_feature_calls(rows: usize) -> FeatureTable {
        let column = |step: usize| (0..rows).map(|i| ((i * step) % 11) as f64).collect();
        table(&[("a", column(3)), ("b", column(5))])
    }

    /// The bits of each probability, row by row, to compare probabilities exactly.
    fn bits(probabilities: &Probabilities) -> Vec<u64> {
        (0..probabilities.len())
            .flat_map(|row| probabilities.row(row).iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            .collect()
    }

    /// Chunked prediction gives exactly the probabilities of whole prediction, in row order,
    /// whatever the row count and the pool, for genotype and promote models.
    #[rstest::rstest]
    fn chunked_prediction_is_whole_prediction(
        #[values(Task::Genotype, Task::Promote)] task: Task,
        #[values(1, 4)] threads: usize,
    ) {
        let chunk = 16;
        let features = ["a", "b"];
        let mut booster = booster(&features, 3);
        booster.set_threads(NonZeroUsize::MIN).unwrap();
        let mut model =
            WinnowrModel { booster, info: ModelInfo { task, ..genotype_info(&features) } };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        for rows in [0, 1, chunk - 1, chunk, chunk + 1, 5 * chunk + 3] {
            let calls = two_feature_calls(rows);
            let whole = model.predict_with(&calls, Predicting::Whole).unwrap();
            let values = to_matrix(&calls, &model.info.features).unwrap();
            let shared = model.booster.shared_predictor().unwrap().unwrap();
            let chunked = pool
                .install(|| {
                    predict_probabilities_chunked(shared, &values, rows, &features, 3, chunk)
                })
                .unwrap();
            assert_eq!(whole.len(), rows);
            assert_eq!(bits(&chunked), bits(&whole), "{rows} rows");
            let moved = &mut model;
            let calls = &calls;
            let default =
                pool.install(move || moved.predict_with(calls, Predicting::Chunked)).unwrap();
            assert_eq!(bits(&default), bits(&whole), "{rows} rows");
        }
    }

    /// A chunk that cannot be predicted fails the prediction, as do a booster of more than one
    /// thread, no features, and values that are not the calls' features.
    #[test]
    fn a_failing_chunk_fails_the_prediction() {
        let mut booster = booster(&["a", "b"], 3);
        booster.set_threads(NonZeroUsize::new(2).unwrap()).unwrap();
        let values = vec![0.0; 2 * 40];
        let chunked = |booster: &Booster, values: &[f32], features: &[&str]| {
            let shared = booster.shared_predictor().unwrap().unwrap();
            let error =
                predict_probabilities_chunked(shared, values, 40, features, 3, 16).unwrap_err();
            format!("{error:#}")
        };
        let message = chunked(&booster, &values, &["a", "b"]);
        assert!(message.contains("needs a booster of one thread, not 2"), "{message}");
        booster.set_threads(NonZeroUsize::MIN).unwrap();
        // Three features for a model of two.
        let message = chunked(&booster, &[0.0; 3 * 40], &["a", "b", "c"]);
        assert!(message.contains("2 features but the matrix has 3 columns"), "{message}");
        let message = chunked(&booster, &[], &[]);
        assert_eq!(message, "a model of no features cannot predict");
        let message = chunked(&booster, &values[1..], &["a", "b"]);
        assert_eq!(message, "79 values are not 40 calls of 2 features");
    }

    /// Nonempty chunked prediction needs one thread even for a non-tree booster; whole and
    /// empty predictions still accept a booster of more than one thread.
    #[rstest::rstest]
    fn chunked_prediction_rejects_multiple_threads(#[values("gbtree", "gblinear")] kind: &str) {
        let features = ["a", "b"];
        let mut booster = booster_of(kind, &features, 3);
        booster.set_threads(NonZeroUsize::new(2).unwrap()).unwrap();
        let model = WinnowrModel { booster, info: genotype_info(&features) };
        let empty = two_feature_calls(0);
        for predicting in [Predicting::Whole, Predicting::Chunked] {
            let prediction = model.predict_with(&empty, predicting).unwrap();
            assert!(prediction.is_empty());
            assert_eq!(prediction.classes(), 3);
        }
        let calls = two_feature_calls(40);
        let whole = model.predict_with(&calls, Predicting::Whole).unwrap();
        assert_eq!((whole.len(), whole.classes()), (40, 3));
        let message = model.predict_with(&calls, Predicting::Chunked).unwrap_err().to_string();
        assert!(message.contains("needs a booster of one thread, not 2"), "{message}");
    }

    /// A model whose booster is not a tree booster, which cannot predict from several threads at
    /// once, predicts in one call when asked to predict in chunks, with the same probabilities.
    #[test]
    fn a_linear_booster_predicts_whole_when_asked_for_chunks() {
        let features = ["a", "b"];
        let booster = booster_of("gblinear", &features, 3);
        assert!(booster.shared_predictor().unwrap().is_none());
        let model = WinnowrModel { booster, info: genotype_info(&features) };
        let calls = two_feature_calls(40);
        let whole = model.predict_with(&calls, Predicting::Whole).unwrap();
        let chunked = model.predict_with(&calls, Predicting::Chunked).unwrap();
        assert_eq!(chunked.len(), 40);
        assert_eq!(bits(&chunked), bits(&whole));
    }

    #[test]
    fn models_record_strlen_settings_and_unrecorded_ones_need_the_legacy_settings() {
        use crate::stutter::{LENGTH_BINS, N_ERRORS, N_PERIOD_CLASSES, StutterModel};
        let mut info = genotype_info(&["a", "b"]);
        assert_eq!(info.strlen_options(), None);
        info.strlen_settings = Some(StrlenOptions::default());
        let message = info.validate().unwrap_err().to_string();
        assert_eq!(message, "only a model with a stutter model has strlen read settings");
        let errors = [1.0 / N_ERRORS as f64; N_ERRORS];
        let stutter = StutterModel::new(
            vec![vec![errors; LENGTH_BINS.len()]; N_PERIOD_CLASSES],
            vec![vec![0; LENGTH_BINS.len()]; N_PERIOD_CLASSES],
        )
        .unwrap();
        info.stutter = Some(stutter);
        info.validate().unwrap();
        assert_eq!(info.strlen_options(), Some(StrlenOptions::default()));
        let model = WinnowrModel { booster: booster(&["a", "b"], 3), info };
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("indel.winnowr.json");
        model.save(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains(r#""strlen_settings":"mapq20,flank5""#), "{text}");
        assert_eq!(WinnowrModel::load(&path, THREADS).unwrap().info, model.info);
        // A model that records no settings needs the legacy settings.
        fs::write(&path, text.replace(r#""strlen_settings":"mapq20,flank5","#, "")).unwrap();
        let unrecorded = WinnowrModel::load(&path, THREADS).unwrap();
        assert_eq!(unrecorded.info.strlen_settings, None);
        assert_eq!(unrecorded.info.strlen_options(), Some(StrlenOptions::LEGACY));
    }

    #[test]
    fn models_with_pileup_features_record_their_settings() {
        let features = ["a", "pu_ref_frac"];
        let mut info = genotype_info(&features);
        let message = info.validate().unwrap_err().to_string();
        assert_eq!(message, "a model has pileup settings if and only if it has pileup features");
        info.pileup_settings = Some(PileupOptions::default());
        info.validate().unwrap();
        let model = WinnowrModel { booster: booster(&features, 3), info };
        let text = model.to_json().unwrap();
        assert!(text.contains(r#""pileup_settings":"mapq20,bq10,flank5""#), "{text}");
        assert_eq!(WinnowrModel::from_json(&text, THREADS).unwrap().info, model.info);
        // Settings without pileup features are rejected too.
        let mut info = genotype_info(&["a", "b"]);
        info.pileup_settings = Some(PileupOptions::default());
        let message = info.validate().unwrap_err().to_string();
        assert_eq!(message, "a model has pileup settings if and only if it has pileup features");
        // A model without pileup features is written as before, without the field.
        let plain =
            WinnowrModel { booster: booster(&["a", "b"], 3), info: genotype_info(&["a", "b"]) };
        assert!(!plain.to_json().unwrap().contains("pileup_settings"));
    }

    /// A model of the gVCF feature source is written without the field, as before sources were
    /// recorded, so a model written before then loads as of the gVCF source; another source round
    /// trips.
    #[test]
    fn models_record_their_feature_source_unless_it_is_gvcf() {
        let features = ["a", "b"];
        let gvcf = WinnowrModel { booster: booster(&features, 3), info: genotype_info(&features) };
        let text = gvcf.to_json().unwrap();
        assert!(!text.contains("feature_source"), "{text}");
        let loaded = WinnowrModel::from_json(&text, THREADS).unwrap();
        assert_eq!(loaded.info.feature_source, FeatureSource::Gvcf);
        assert_eq!(loaded.info, gvcf.info);

        let mut info = genotype_info(&features);
        info.feature_source = FeatureSource::Vcf;
        let vcf = WinnowrModel { booster: booster(&features, 3), info };
        let text = vcf.to_json().unwrap();
        assert!(text.contains(r#""feature_source":"vcf""#), "{text}");
        assert_eq!(WinnowrModel::from_json(&text, THREADS).unwrap().info, vcf.info);
        let unknown = text.replace(r#""feature_source":"vcf""#, r#""feature_source":"bam""#);
        let error = format!("{:#}", WinnowrModel::from_json(&unknown, THREADS).unwrap_err());
        assert!(error.contains("unknown feature source 'bam'"), "{error}");
    }

    /// A model without region digests is written as before they were recorded; with them (of
    /// its features' regions and of the policy's), it round trips; a malformed digest is not
    /// read.
    #[test]
    fn models_record_their_region_digests_when_known() {
        let features = ["a", "b"];
        let plain = WinnowrModel { booster: booster(&features, 3), info: genotype_info(&features) };
        let text = plain.to_json().unwrap();
        assert!(!text.contains("region_digests"), "{text}");
        let mut info = genotype_info(&features);
        let digest = RegionDigest { sha256: "ab".repeat(32), intervals: 2, bases: 30 };
        info.region_digests = [("segdups".to_string(), digest.clone())].into_iter().collect();
        info.policy_region_digests =
            [("notinalldifficultregions".to_string(), digest)].into_iter().collect();
        let known = WinnowrModel { booster: booster(&features, 3), info };
        let text = known.to_json().unwrap();
        assert!(text.contains(r#""region_digests":{"segdups":"#), "{text}");
        assert!(text.contains(r#""policy_region_digests":{"notinalldifficultregions":"#));
        assert_eq!(WinnowrModel::from_json(&text, THREADS).unwrap().info, known.info);
        let bad = text.replace(&"ab".repeat(32), "not-a-digest");
        let error = format!("{:#}", WinnowrModel::from_json(&bad, THREADS).unwrap_err());
        assert!(error.contains("sha256 must be 64 lower-case hexadecimal digits"), "{error}");
    }

    /// A model of unknown alignment provenance is written without the field, as before it was
    /// recorded, so such a model loads with it unknown; a known one round trips.
    #[test]
    fn models_record_their_alignment_provenance_when_known() {
        let features = ["a", "b"];
        let unknown =
            WinnowrModel { booster: booster(&features, 3), info: genotype_info(&features) };
        let text = unknown.to_json().unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(json.get("alignment").is_none(), "{text}");
        assert_eq!(WinnowrModel::from_json(&text, THREADS).unwrap().info.alignment, None);

        let mut info = genotype_info(&features);
        info.alignment = Some(crate::testing::bwa_provenance("primary"));
        let known = WinnowrModel { booster: booster(&features, 3), info };
        let text = known.to_json().unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            json["alignment"],
            serde_json::json!({
                "aligner": "bwa-mem3",
                "alt-project": "primary",
                "options-inference": "explicit",
            })
        );
        assert_eq!(WinnowrModel::from_json(&text, THREADS).unwrap().info, known.info);
    }

    /// A model's information is read without its booster, which may come first; a file cut short
    /// is an error.
    #[test]
    fn model_information_is_read_without_the_booster() {
        let dir = tempfile::TempDir::new().unwrap();
        let model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        let path = dir.path().join("m.winnowr.json");
        model.save(&path).unwrap();
        assert_eq!(ModelInfo::read(&path).unwrap(), model.info);
        assert_eq!(
            ModelInfo::read(&path).unwrap(),
            WinnowrModel::load(&path, THREADS).unwrap().info
        );
        let text = fs::read_to_string(&path).unwrap();
        let (info_text, booster) = split_member(&text, "booster").unwrap();
        let first = format!(
            "{{\"booster\":{booster},{}",
            info_text.trim_start().strip_prefix('{').unwrap()
        );
        fs::write(&path, &first).unwrap();
        assert_eq!(ModelInfo::read(&path).unwrap(), model.info);
        fs::write(&path, &text[..text.len() / 2]).unwrap();
        let message = format!("{:#}", ModelInfo::read(&path).unwrap_err());
        assert!(message.contains("not a JSON object with a booster"), "{message}");
    }

    #[test]
    fn splits_the_booster_member_without_parsing_it() {
        let text =
            r#"{ "a": 1, "booster": {"x": [NaN, "}", {"y": "\"q"}], "z": -Infinity} , "b": "c" }"#;
        let (rest, booster) = split_member(text, "booster").unwrap();
        assert_eq!(booster, r#"{"x": [NaN, "}", {"y": "\"q"}], "z": -Infinity}"#);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&rest).unwrap(),
            serde_json::json!({"a": 1, "b": "c"})
        );
        let (rest, value) = split_member(r#"{"a": 1, "booster": 2.5}"#, "booster").unwrap();
        assert_eq!((rest.as_str(), value), (r#"{"a": 1}"#, "2.5"));
        assert!(split_member(r#"{"a": 1}"#, "booster").is_none());
        assert!(split_member("[1]", "booster").is_none());
    }

    #[test]
    fn keeps_the_reference_contig_order() {
        let mut info = genotype_info(&["a", "b"]);
        info.reference_contigs =
            Some(["chr1", "chr2", "chr10"].iter().map(|c| (c.to_string(), Some(5))).collect());
        let model = WinnowrModel { booster: booster(&["a", "b"], 3), info };
        let loaded = WinnowrModel::from_json(&model.to_json().unwrap(), THREADS).unwrap();
        let order: Vec<&String> = loaded.info.reference_contigs.as_ref().unwrap().keys().collect();
        assert_eq!(order, ["chr1", "chr2", "chr10"]);
    }

    #[test]
    fn rejects_bad_model_files() {
        let model =
            WinnowrModel { booster: booster(&["a", "b"], 3), info: genotype_info(&["a", "b"]) };
        let json: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        let load = |value: &serde_json::Value| {
            WinnowrModel::from_json(&value.to_string(), THREADS).err().map(|e| format!("{e:#}"))
        };
        assert_eq!(load(&json), None);

        // Any format version but 3, or none, is refused.
        let mut other_format = json.clone();
        other_format["winnowr_version"] = "0.0.1".into();
        for (version, shown) in [(Some(2), "2"), (Some(4), "4"), (None, "none")] {
            match version {
                Some(version) => other_format["format_version"] = version.into(),
                None => {
                    other_format.as_object_mut().unwrap().remove("format_version");
                }
            }
            assert_eq!(
                load(&other_format).unwrap(),
                format!(
                    "unsupported model format version {shown} (written by winnowr 0.0.1); this \
                     winnowr reads version 3: train new models with this winnowr"
                )
            );
        }

        let mut unknown = json.clone();
        unknown["extra"] = true.into();
        assert!(load(&unknown).unwrap().contains("unknown field `extra`"));

        let mut swapped = json.clone();
        swapped["features"] = serde_json::json!(["b", "a"]);
        assert!(
            load(&swapped)
                .unwrap()
                .contains("the booster's features (a, b) are not the model's (b, a)")
        );

        let mut regenotype = json.clone();
        regenotype["regenotype"] = true.into();
        assert!(
            load(&regenotype)
                .unwrap()
                .contains("a re-genotype model must be a genotype model with a stutter model")
        );

        let mut promote = json.clone();
        promote["task"] = "promote".into();
        assert!(load(&promote).unwrap().contains("a promote model needs promote thresholds"));

        let mut no_veto = json.clone();
        no_veto["thresholds"].as_object_mut().unwrap().remove("veto");
        assert!(load(&no_veto).unwrap().contains("missing: veto"));

        let mut no_booster = json;
        no_booster.as_object_mut().unwrap().remove("booster");
        assert!(
            load(&no_booster).unwrap().contains("the model is not a JSON object with a booster")
        );
    }

    #[test]
    fn saving_replaces_a_model_whole_with_the_usual_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let model =
            WinnowrModel { booster: booster(&["a", "b"], 3), info: genotype_info(&["a", "b"]) };
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("snp.winnowr.json");
        fs::write(&path, "an older model").unwrap();
        let plain = dir.path().join("plain");
        fs::write(&plain, "").unwrap();
        model.save(&path).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), model.to_json().unwrap());
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&path), mode(&plain));
        let mut names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["plain", "snp.winnowr.json"]);
        // A file at a path that is not a directory fails before anything is written.
        let message = format!("{:#}", model.save(plain.join("x.winnowr.json")).unwrap_err());
        assert!(message.starts_with("failed to write model"), "{message}");
    }

    #[test]
    fn warns_only_for_models_of_another_version() {
        let path = Path::new("m.winnowr.json");
        // What training records, with the commit in a git build.
        assert_eq!(other_version_warning(path, crate::version::VERSION.as_str()), None);
        // The same release built without git (a source tarball or crates.io), or from another
        // commit.
        let package = env!("CARGO_PKG_VERSION");
        assert_eq!(other_version_warning(path, package), None);
        assert_eq!(other_version_warning(path, &format!("{package}-0123456789ab-dirty")), None);
        let this = crate::version::VERSION.as_str();
        assert_eq!(
            other_version_warning(path, "0.0.1"),
            Some(format!("m.winnowr.json was trained with winnowr 0.0.1; this is winnowr {this}"))
        );
    }

    /// A booster in XGBoost's binary format (UBJSON, whose objects also start with `{`) never
    /// reaches XGBoost, whose binary parser does not bounds-check its input.
    #[rstest]
    #[case::letter(r"{Lzzzz}")]
    #[case::space_then_letter("{ \nLzzzz}")]
    fn rejects_a_binary_booster(#[case] binary: &str) {
        let model =
            WinnowrModel { booster: booster(&["a", "b"], 3), info: genotype_info(&["a", "b"]) };
        let text = model.to_json().unwrap();
        let at = text.find("\"booster\":").unwrap();
        let text = format!("{}\"booster\":{binary}}}", &text[..at]);
        let message = format!("{:#}", WinnowrModel::from_json(&text, THREADS).unwrap_err());
        assert_eq!(message, "invalid XGBoost booster: XGBoost: the model is not a JSON object");
    }

    #[test]
    fn rejects_a_booster_with_another_class_count() {
        let mut info = genotype_info(&["a", "b"]);
        info.task = Task::Promote;
        info.thresholds.clear();
        info.promote_thresholds =
            Some(PromoteThresholds { t_promote: 0.5, t_flip: 0.5, gain: 0.0 });
        let model = WinnowrModel { booster: booster(&["a", "b"], 4), info };
        let message = format!(
            "{:#}",
            WinnowrModel::from_json(&model.to_json().unwrap(), THREADS).unwrap_err()
        );
        assert!(
            message.contains("the booster predicts 4 classes, but a promote model has 3"),
            "{message}"
        );
    }

    #[test]
    fn checks_the_reference() {
        let info = genotype_info(&["a"]);
        let path = Path::new("m.winnowr.json");
        let same = IndexMap::from([("chr1".to_string(), Some(1000)), ("chr2".to_string(), None)]);
        check_reference(path, &info, &same, "VCF x", true).unwrap();
        let unknown_length = IndexMap::from([("chr1".to_string(), None)]);
        check_reference(path, &info, &unknown_length, "VCF x", true).unwrap();
        let other = IndexMap::from([("chr1".to_string(), Some(999))]);
        let message = check_reference(path, &info, &other, "VCF x", true).unwrap_err().to_string();
        assert_eq!(
            message,
            "VCF x is not on the reference model m.winnowr.json was trained on: other lengths: chr1 (999 bp, model 1000 bp)"
        );
        let numbered = IndexMap::from([("1".to_string(), Some(1000))]);
        let message =
            check_reference(path, &info, &numbered, "VCF x", true).unwrap_err().to_string();
        assert!(message.ends_with("missing contigs chr1"), "{message}");
        check_reference(path, &info, &numbered, "VCF x", false).unwrap();
        let mut unchecked = info;
        unchecked.reference_contigs = None;
        check_reference(path, &unchecked, &numbered, "VCF x", true).unwrap();
    }

    #[test]
    fn the_unfixable_flip_policy_is_written_only_when_not_the_default() {
        let default = TrainingParams::default();
        let text = serde_json::to_string(&default).unwrap();
        assert!(!text.contains("unfixable_flips"), "{text}");
        assert_eq!(serde_json::from_str::<TrainingParams>(&text).unwrap(), default);
        let dropped = TrainingParams { unfixable_flips: UnfixableFlips::Drop, ..default };
        let text = serde_json::to_string(&dropped).unwrap();
        assert!(text.contains("\"unfixable_flips\":\"drop\""), "{text}");
        assert_eq!(serde_json::from_str::<TrainingParams>(&text).unwrap(), dropped);
    }

    #[test]
    fn the_final_refit_and_cv_folds_are_written_only_when_not_the_default() {
        let default = TrainingParams::default();
        let text = serde_json::to_string(&default).unwrap();
        assert!(!text.contains("refit_final") && !text.contains("cv_folds"), "{text}");
        let changed = TrainingParams {
            refit_final: true,
            cv_folds: CvFolds::Chromosomes { folds: 5 },
            ..default
        };
        let text = serde_json::to_string(&changed).unwrap();
        assert!(text.contains("\"refit_final\":true"), "{text}");
        assert!(text.contains("\"cv_folds\":{\"chromosomes\":{\"folds\":5}}"), "{text}");
        assert_eq!(serde_json::from_str::<TrainingParams>(&text).unwrap(), changed);
        let samples = TrainingParams { cv_folds: CvFolds::Samples, ..default };
        assert_eq!(
            serde_json::to_string(&samples).unwrap(),
            serde_json::to_string(&default).unwrap()
        );
    }
}
