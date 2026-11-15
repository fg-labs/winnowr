//! A trained model: an XGBoost multi-class booster bundled with winnowr's model information in one
//! `<name>.winnowr.json` file.
//!
//! The file is a JSON object holding [`ModelInfo`]'s fields (the ordered feature names, the variant
//! type and task, the cross-validated thresholds, the training samples, the reference contigs, the
//! stutter model of models with repeat-length features, and the versions that trained it) plus
//! `booster`, the XGBoost booster in its JSON format. Format version 3 is this bundled format; the
//! Python version's two-file models (formats 1 and 2) are not read, and cannot be converted.

use std::collections::BTreeMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;
use log::warn;
use serde::{Deserialize, Serialize};
use winnowr_xgboost::{Booster, DMatrix};

use crate::common::{Task, VariantType};
use crate::labels::{Label, PromoteLabel, UnfixableFlips};
use crate::pileup::PileupOptions;
use crate::pileup_features::PILEUP_FEATURES;
use crate::rescore::{Mode, Probabilities};
use crate::strlen::StrlenOptions;
use crate::stutter::StutterModel;
use crate::table::FeatureTable;
use crate::thresholds::{PromoteThresholds, Thresholds};

/// The suffix of a model file.
pub const MODEL_SUFFIX: &str = ".winnowr.json";

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
    pub reference_contigs: Option<IndexMap<String, Option<u64>>>,
    /// The training parameters.
    pub params: TrainingParams,
    /// The stutter model of a model with repeat-length features (see [`crate::stutter`]); its
    /// features need strlen evidence in the feature table.
    pub stutter: Option<StutterModel>,
    /// Whether the model has the re-genotype class ([`Label::Regenotype`]).
    pub regenotype: bool,
    /// The read settings the strlen evidence of the training tables was measured with (`None`
    /// without a stutter model, or for models trained before the settings were recorded: see
    /// [`ModelInfo::strlen_options`]). The strlen evidence a model scores must be measured the
    /// same way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strlen_settings: Option<StrlenOptions>,
    /// The settings the allele read counts by MAPQ of the training tables were measured with, for
    /// a model with pileup features (see [`crate::pileup`]); `None` for a model without them
    /// (every model trained before they existed). The pileup evidence a model scores must be
    /// measured the same way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pileup_settings: Option<PileupOptions>,
    /// The winnowr version that trained the model.
    pub winnowr_version: String,
    /// The XGBoost version that trained the model.
    pub xgboost_version: String,
}

impl ModelInfo {
    /// The read settings of the strlen evidence a model with repeat-length features needs: those
    /// recorded, or [`StrlenOptions::LEGACY`] for models trained before the settings were
    /// recorded.
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

/// A trained booster with its model information.
#[derive(Debug)]
pub struct WinnowrModel {
    /// The XGBoost booster.
    pub booster: Booster,
    /// The model information.
    pub info: ModelInfo,
}

impl WinnowrModel {
    /// Loads a model file, predicting with at most `threads` threads.
    ///
    /// Logs a warning when the model was trained by another winnowr version, or records no
    /// reference contigs (so the reference of the calls cannot be checked).
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not end in `.winnowr.json`, the file cannot be read, is
    /// not of a supported format version, has unknown or invalid fields, or its booster does not
    /// load or does not have the model's features and classes.
    pub fn load<P: AsRef<Path>>(path: P, threads: NonZeroUsize) -> Result<Self> {
        let path = path.as_ref();
        if !path.to_string_lossy().ends_with(MODEL_SUFFIX) {
            bail!("a model file must end in {MODEL_SUFFIX}: {}", path.display());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read model {}", path.display()))?;
        let model = Self::from_json(&text, threads)
            .with_context(|| format!("failed to load model {}", path.display()))?;
        if let Some(warning) = other_version_warning(path, &model.info.winnowr_version) {
            warn!("{warning}");
        }
        if model.info.reference_contigs.is_none() {
            warn!(
                "{} records no reference contigs, so the reference of the calls cannot be checked \
                 against it",
                path.display()
            );
        }
        Ok(model)
    }

    /// A model from the text of a model file.
    ///
    /// # Errors
    ///
    /// See [`WinnowrModel::load`].
    pub fn from_json(text: &str, threads: NonZeroUsize) -> Result<Self> {
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
            bail!(
                "unsupported model format version {} (written by winnowr {}); this winnowr reads \
                 version {MODEL_FORMAT_VERSION}: train new models with this winnowr (the Python version's \
                 models cannot be converted)",
                versions.format_version.map_or_else(|| "none".to_string(), |v| v.to_string()),
                versions.winnowr_version.as_deref().unwrap_or("unknown")
            );
        }
        let info: ModelInfo =
            serde_json::from_str(&info_text).context("invalid model information")?;
        info.validate()?;
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

    /// Predicts class probabilities for a feature table's calls.
    ///
    /// The probabilities are in [`Label`] order (genotype models, with the re-genotype column for
    /// re-genotype models) or [`PromoteLabel`] order (promote models).
    ///
    /// # Errors
    ///
    /// Returns an error if the table lacks a model feature, or XGBoost fails.
    pub fn predict(&self, table: &FeatureTable) -> Result<Probabilities> {
        let classes = self.info.num_classes();
        if table.num_rows() == 0 {
            return Ok(Probabilities::new(Vec::new(), classes));
        }
        let values = to_matrix(table, &self.info.features)?;
        predict_probabilities(
            &self.booster,
            &values,
            table.num_rows(),
            &self.info.features,
            classes,
        )
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
    let rows = table.num_rows();
    let mut matrix = vec![f32::NAN; rows * features.len()];
    for (column, name) in features.iter().enumerate() {
        let values = table.f64_column(name.as_ref())?;
        for (row, &value) in values.iter().enumerate() {
            let value = value as f32;
            matrix[row * features.len() + column] =
                if value.is_infinite() { f32::NAN } else { value };
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
    contigs: &IndexMap<String, Option<u64>>,
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
        let info = FeatureTableInfo {
            sample: "S".to_string(),
            contigs: vec![],
            region_names: vec![],
            fn_counts: None,
            promote_labelled: false,
            strlen: false,
            strlen_settings: None,
            pileup_settings: None,
            reference_contigs: None,
        };
        FeatureTable::new(batch, info)
    }

    /// Trains a tiny three-class booster on features `a` and `b`.
    fn booster(features: &[&str], classes: usize) -> Booster {
        let rows = 30;
        let values: Vec<f32> = (0..rows * features.len()).map(|i| (i % 7) as f32).collect();
        let labels: Vec<f32> = (0..rows).map(|i| (i % classes) as f32).collect();
        let mut train = DMatrix::from_dense(&values, rows, features.len(), f32::NAN).unwrap();
        train.set_labels(&labels).unwrap();
        train.set_feature_names(features).unwrap();
        let mut booster = Booster::new(&[&train], THREADS).unwrap();
        let classes_value = classes.to_string();
        booster
            .set_params(&[("objective", "multi:softprob"), ("num_class", &classes_value)])
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
            winnowr_version: crate::version::VERSION.clone(),
            xgboost_version: "3.4.2".to_string(),
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

    #[test]
    fn models_record_strlen_settings_and_old_models_need_the_legacy_settings() {
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
        // A model saved before the settings were recorded needs the legacy settings.
        fs::write(&path, text.replace(r#""strlen_settings":"mapq20,flank5","#, "")).unwrap();
        let old = WinnowrModel::load(&path, THREADS).unwrap();
        assert_eq!(old.info.strlen_settings, None);
        assert_eq!(old.info.strlen_options(), Some(StrlenOptions::LEGACY));
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

        let mut python_format = json.clone();
        python_format["format_version"] = 2.into();
        assert!(load(&python_format).unwrap().contains("unsupported model format version 2"));

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
