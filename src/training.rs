//! Train a model and choose its thresholds by leave-one-group-out cross-validation.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroUsize;

use anyhow::{Result, bail};
use indexmap::IndexMap;
use log::info;
use winnowr_xgboost::Booster;

use crate::common::{Task, VariantType};
use crate::fitting::{fit_booster, validation_mask};
use crate::model::{
    CvFolds, MODEL_FORMAT_VERSION, ModelInfo, TrainingParams, WinnowrModel, num_classes,
    predict_probabilities, to_matrix,
};
use crate::pileup::PileupOptions;
use crate::rescore::{Mode, Probabilities};
use crate::samples::{PromoteSampleCalls, SampleCalls};
use crate::strlen::StrlenOptions;
use crate::stutter::{StutterFit, StutterModel, fit_stutter_from_feature_tables};
use crate::table::{FeatureTable, FeatureTableInfo};
use crate::thresholds::{
    FoldPredictions, PromoteFold, Thresholds, search_promote_thresholds, search_thresholds,
};
use crate::version::{VERSION, XGBOOST_VERSION};

/// The columns training and evaluation read besides the model features: the call keys, labels,
/// the columns selecting calls, and the strlen evidence of length calls.
pub const SUPPORT_COLUMNS: [&str; 20] = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "filter",
    "variant_type",
    "gt",
    "label",
    "flip_ok",
    "promote_label",
    "overlaps_pass",
    "flippable",
    "truth_delta_lo",
    "truth_delta_hi",
    "sl_n_spanning",
    "sl_period",
    "sl_tract_len",
    "sl_tract_start",
    "sl_histogram",
    "sl_unit",
];

/// The columns to read from a feature table to train or evaluate models with these features.
#[must_use]
pub fn table_columns<S: AsRef<str>>(features: &[S]) -> Vec<String> {
    let mut columns: Vec<String> = SUPPORT_COLUMNS.iter().map(ToString::to_string).collect();
    for feature in features.iter().map(AsRef::as_ref) {
        if !columns.iter().any(|c| c == feature) {
            columns.push(feature.to_string());
        }
    }
    columns
}

/// The calls of one training sample ([`SampleCalls`] or [`PromoteSampleCalls`]).
pub trait CvSample {
    /// The held-out predictions of the sample's calls.
    type Fold;

    /// The sample name.
    fn sample(&self) -> &str;

    /// The labelled training calls.
    fn calls(&self) -> &FeatureTable;

    /// The class of each call.
    fn classes(&self) -> Vec<usize>;

    /// Per call, whether it is left out of the fits (`None` if none is). Every call is still
    /// predicted when the sample is held out.
    fn excluded(&self) -> Option<&[bool]> {
        None
    }

    /// Packages held-out predictions for the calls as a cross-validation fold.
    ///
    /// # Errors
    ///
    /// Returns an error if the calls lack a column of the fold.
    fn to_fold(&self, probabilities: Probabilities) -> Result<Self::Fold>;

    /// Combines folds of the sample's calls row by row: call `i` from `parts[which[i]]`.
    ///
    /// # Errors
    ///
    /// Returns an error if a part has other calls or `which` names a missing part.
    fn merge_folds(parts: Vec<Self::Fold>, which: &[usize]) -> Result<Self::Fold>;
}

impl CvSample for SampleCalls {
    type Fold = FoldPredictions;

    fn sample(&self) -> &str {
        &self.sample
    }

    fn calls(&self) -> &FeatureTable {
        &self.calls
    }

    fn classes(&self) -> Vec<usize> {
        self.labels.iter().map(|label| label.index()).collect()
    }

    fn excluded(&self) -> Option<&[bool]> {
        (!self.excluded.is_empty()).then_some(self.excluded.as_slice())
    }

    fn to_fold(&self, probabilities: Probabilities) -> Result<FoldPredictions> {
        SampleCalls::to_fold(self, probabilities)
    }

    fn merge_folds(parts: Vec<FoldPredictions>, which: &[usize]) -> Result<FoldPredictions> {
        FoldPredictions::merge_rows(&parts, which)
    }
}

impl CvSample for PromoteSampleCalls {
    type Fold = PromoteFold;

    fn sample(&self) -> &str {
        &self.sample
    }

    fn calls(&self) -> &FeatureTable {
        &self.calls
    }

    fn classes(&self) -> Vec<usize> {
        self.labels.iter().map(|label| label.index()).collect()
    }

    fn to_fold(&self, probabilities: Probabilities) -> Result<PromoteFold> {
        PromoteSampleCalls::to_fold(self, probabilities)
    }

    fn merge_folds(parts: Vec<PromoteFold>, which: &[usize]) -> Result<PromoteFold> {
        PromoteFold::merge_rows(&parts, which)
    }
}

/// Fits a booster on the pooled calls of the samples, less their excluded calls, early-stopping
/// on the calls in held-out genomic blocks (see [`validation_mask`]).
///
/// Each sample comes with the calls to fit (`None`: every call).
fn fit<S: CvSample>(
    samples: &[(&S, Option<&[bool]>)],
    features: &[String],
    params: &TrainingParams,
    num_class: usize,
    threads: NonZeroUsize,
) -> Result<Booster> {
    fit_holding_out(samples, features, params, num_class, threads).map(|(booster, _)| booster)
}

/// Fits a booster as [`fit`] does, and returns the number of calls held out for early stopping.
fn fit_holding_out<S: CvSample>(
    samples: &[(&S, Option<&[bool]>)],
    features: &[String],
    params: &TrainingParams,
    num_class: usize,
    threads: NonZeroUsize,
) -> Result<(Booster, usize)> {
    let mut values = Vec::new();
    let mut classes = Vec::new();
    let mut is_validation = Vec::new();
    for &(sample, rows) in samples {
        let calls = sample.calls();
        let matrix = to_matrix(calls, features)?;
        let sample_classes = sample.classes();
        let chroms = calls.str_column("chrom")?;
        let mask = validation_mask(&chroms, calls.i64_column("pos")?, params)?;
        match (rows, sample.excluded()) {
            (None, None) => {
                values.extend(matrix);
                classes.extend(sample_classes);
                is_validation.extend(mask);
            }
            (rows, excluded) => {
                let width = features.len();
                for (row, class) in sample_classes.into_iter().enumerate() {
                    let fitted = rows.is_none_or(|rows| rows[row]);
                    if fitted && !excluded.is_some_and(|excluded| excluded[row]) {
                        values.extend_from_slice(&matrix[row * width..(row + 1) * width]);
                        classes.push(class);
                        is_validation.push(mask[row]);
                    }
                }
            }
        }
    }
    let held_out = is_validation.iter().filter(|&&v| v).count();
    let booster =
        fit_booster(&values, &classes, &is_validation, features, params, num_class, threads)?;
    Ok((booster, held_out))
}

/// Whether the final model is refit (see [`fit_final`]): only when asked and early stopping held
/// calls out, as otherwise the refit would repeat the fit on the same calls.
fn refits(params: &TrainingParams, held_out: usize) -> bool {
    params.refit_final && params.validation_fraction > 0.0 && held_out > 0
}

/// Fits the final model on every call of the samples: early-stopping as [`fit`] does, then, with
/// [`TrainingParams::refit_final`], refitting on every call, those held out for early stopping
/// included, for the rounds early stopping chose.
///
/// The final fit's own early stopping sets the rounds because it fits the same samples as the
/// refit, less only the held-out blocks; cross-validation folds fit fewer calls (a held-out sample
/// group or contig group fewer) and so stop earlier.
fn fit_final<S: CvSample>(
    samples: &[S],
    features: &[String],
    params: &TrainingParams,
    num_class: usize,
    threads: NonZeroUsize,
) -> Result<Booster> {
    let all: Vec<(&S, Option<&[bool]>)> = samples.iter().map(|s| (s, None)).collect();
    let (booster, held_out) = fit_holding_out(&all, features, params, num_class, threads)?;
    if !refits(params, held_out) {
        return Ok(booster);
    }
    let rounds = booster.boosted_rounds()?;
    info!("Refitting the final model on all its training calls for {rounds} rounds");
    let refit =
        TrainingParams { max_rounds: u32::try_from(rounds)?, validation_fraction: 0.0, ..*params };
    fit(&all, features, &refit, num_class, threads)
}

/// Partitions samples into cross-validation groups that are held out together.
///
/// A sample not named in any group is a group of its own, so with no groups every sample is held
/// out alone (leave-one-sample-out). Use groups for related samples (e.g. a trio), which would
/// otherwise leak into each other's training folds.
///
/// Returns the groups ordered by their first sample in `names`, each listing its samples in
/// `names` order.
///
/// # Errors
///
/// Returns an error if a sample name is repeated, a group is empty or names an unknown sample,
/// or a sample is in more than one group.
pub fn group_samples<S: AsRef<str>, G: AsRef<str>>(
    names: &[S],
    groups: &[Vec<G>],
) -> Result<Vec<Vec<String>>> {
    let names: Vec<&str> = names.iter().map(AsRef::as_ref).collect();
    let repeated = crate::common::repeated(names.iter().copied());
    if !repeated.is_empty() {
        bail!("each training sample must be given once; repeated: {}", repeated.join(", "));
    }
    let known: BTreeSet<&str> = names.iter().copied().collect();
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    for (index, group) in groups.iter().enumerate() {
        if group.is_empty() {
            bail!("sample groups must not be empty");
        }
        let unknown: BTreeSet<&str> =
            group.iter().map(AsRef::as_ref).filter(|name| !known.contains(name)).collect();
        if !unknown.is_empty() {
            let unknown: Vec<&str> = unknown.into_iter().collect();
            let known: Vec<&str> = known.iter().copied().collect();
            bail!(
                "a sample group names unknown samples: {}; known: {}",
                unknown.join(", "),
                known.join(", ")
            );
        }
        for name in group.iter().map(AsRef::as_ref) {
            if *group_of.entry(name).or_insert(index) != index {
                bail!("sample {name} is in more than one sample group");
            }
        }
    }
    let mut keyed: IndexMap<String, Vec<String>> = IndexMap::new();
    for name in names {
        let key = group_of.get(name).map_or_else(|| format!("sample:{name}"), |g| format!("{g}"));
        keyed.entry(key).or_default().push(name.to_string());
    }
    Ok(keyed.into_values().collect())
}

/// Splits contigs into `folds` groups of about equal calls, to hold out together in
/// cross-validation.
///
/// Contigs are taken from the most calls to the fewest (ties in input order), each into the
/// group with the fewest calls so far (ties to the earlier group). Every call at a site, from
/// every sample, and all its neighbours are in the same group, so related samples (e.g. a trio,
/// sharing most sites) and neighbourhood features do not leak between a fold's fitted and scored
/// calls.
///
/// Returns the groups, each listing its contigs in input order.
///
/// # Arguments
///
/// * `calls` - the contigs, with their calls pooled over every sample.
/// * `folds` - the number of groups.
///
/// # Errors
///
/// Returns an error if `folds` is below two or above the number of contigs.
pub fn contig_groups<S: AsRef<str>>(
    calls: &[(S, usize)],
    folds: usize,
) -> Result<Vec<Vec<String>>> {
    if folds < 2 {
        bail!("chromosome cross-validation needs at least two folds, got {folds}");
    }
    if calls.len() < folds {
        bail!(
            "chromosome cross-validation with {folds} folds needs at least {folds} contigs with \
             calls, got {}",
            calls.len()
        );
    }
    let mut order: Vec<usize> = (0..calls.len()).collect();
    order.sort_by(|&a, &b| calls[b].1.cmp(&calls[a].1).then(a.cmp(&b)));
    let mut totals = vec![0usize; folds];
    let mut group_of = vec![0usize; calls.len()];
    for index in order {
        let group = (0..folds).min_by_key(|&g| (totals[g], g)).expect("at least two folds");
        totals[group] += calls[index].1;
        group_of[index] = group;
    }
    let mut groups = vec![Vec::new(); folds];
    for (index, (contig, _)) in calls.iter().enumerate() {
        groups[group_of[index]].push(contig.as_ref().to_string());
    }
    Ok(groups)
}

/// The calls a cross-validation fold fits: those of `samples` on `contigs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldTraining {
    /// The samples whose calls the fold fits.
    pub samples: BTreeSet<String>,
    /// The contigs whose calls the fold fits (`None`: every contig).
    pub contigs: Option<BTreeSet<String>>,
}

/// Rebuilds every sample, in input order, for one cross-validation fold from the calls it fits
/// (e.g. with a stutter model fitted on them only).
pub type Refit<'a, S> = &'a (dyn Fn(&FoldTraining) -> Result<Vec<S>> + Sync);

/// Fits the stutter model of a cross-validation fold on the calls it fits (see
/// [`FoldTraining`]): those of its samples' tables, and of those only the calls on its contigs
/// when it has them, so that the truth of held-out calls is not fitted.
///
/// # Errors
///
/// Returns an error if a table lacks a column, or the stutter model cannot be fitted (see
/// [`fit_stutter_from_feature_tables`]).
pub fn fit_fold_stutter(
    tables: &[(&FeatureTable, &FeatureTableInfo)],
    training: &FoldTraining,
    fit: StutterFit,
) -> Result<StutterModel> {
    let chosen = tables.iter().copied().filter(|(_, info)| training.samples.contains(&info.sample));
    let Some(contigs) = &training.contigs else {
        return fit_stutter_from_feature_tables(&chosen.collect::<Vec<_>>(), fit);
    };
    let tables: Vec<FeatureTable> = chosen
        .map(|(table, _)| {
            let chroms = table.str_column("chrom")?;
            table.filter(&chroms.iter().map(|c| contigs.contains(*c)).collect::<Vec<_>>())
        })
        .collect::<Result<_>>()?;
    let fold: Vec<(&FeatureTable, &FeatureTableInfo)> =
        tables.iter().map(|table| (table, table.info())).collect();
    fit_stutter_from_feature_tables(&fold, fit)
}

/// What a model is trained on and how, shared by genotype and promote models.
#[derive(Debug, Clone)]
pub struct TrainingSpec<'a> {
    /// The variant type.
    pub variant_type: VariantType,
    /// The model features, in order.
    pub features: &'a [String],
    /// The region set names the features use.
    pub region_names: &'a [String],
    /// The contigs the calls came from.
    pub contigs: &'a [String],
    /// The lengths of `contigs` in the training samples' reference.
    pub reference_contigs: Option<IndexMap<String, Option<u64>>>,
    /// The training parameters, including how cross-validation forms its folds
    /// ([`TrainingParams::cv_folds`]).
    pub params: TrainingParams,
    /// Sample groups to hold out together in sample cross-validation (see [`group_samples`]).
    pub groups: &'a [Vec<String>],
    /// The XGBoost threads.
    pub threads: NonZeroUsize,
    /// The read settings of the training tables' repeat-length evidence, if the model uses it.
    pub strlen_settings: Option<StrlenOptions>,
    /// The settings of the training tables' pileup evidence, if the model uses it.
    pub pileup_settings: Option<PileupOptions>,
}

/// What a cross-validation fold holds out.
#[derive(Debug, Clone)]
enum HeldOut {
    /// Every call of these samples.
    Samples(Vec<String>),
    /// Every sample's calls on these contigs.
    Contigs(BTreeSet<String>),
}

impl HeldOut {
    /// Which calls of the sample the fold holds out (`None`: none), as a mask unless all are.
    fn rows<S: CvSample>(&self, sample: &S) -> Result<Option<Rows>> {
        match self {
            HeldOut::Samples(held_out) => {
                Ok(held_out.iter().any(|h| h == sample.sample()).then_some(Rows::All))
            }
            HeldOut::Contigs(contigs) => {
                let mask: Vec<bool> = sample
                    .calls()
                    .str_column("chrom")?
                    .iter()
                    .map(|chrom| contigs.contains(*chrom))
                    .collect();
                Ok(mask.iter().any(|&h| h).then_some(Rows::Mask(mask)))
            }
        }
    }

    /// The calls the fold fits.
    fn training(&self, names: &[String], all_contigs: &BTreeSet<String>) -> FoldTraining {
        match self {
            HeldOut::Samples(held_out) => FoldTraining {
                samples: names.iter().filter(|n| !held_out.contains(n)).cloned().collect(),
                contigs: None,
            },
            HeldOut::Contigs(contigs) => FoldTraining {
                samples: names.iter().cloned().collect(),
                contigs: Some(all_contigs.difference(contigs).cloned().collect()),
            },
        }
    }

    /// Describes what is held out, for logging.
    fn describe(&self) -> String {
        match self {
            HeldOut::Samples(samples) => samples.join(", "),
            HeldOut::Contigs(contigs) => {
                format!("every sample on {}", contigs.iter().cloned().collect::<Vec<_>>().join(","))
            }
        }
    }
}

/// Some calls of a sample.
#[derive(Debug, Clone)]
enum Rows {
    /// Every call.
    All,
    /// The calls whose flag is set.
    Mask(Vec<bool>),
}

/// The held-out predictions of one cross-validation fold: per held-out sample, its index, its
/// fold (predicting every call; only the held-out ones are meaningful) and the held-out calls
/// (`None`: all).
type FoldResult<F> = Result<Vec<(usize, F, Option<Vec<bool>>)>>;

/// The number of fits, one after the other, that make the final model: two when it is refitted
/// after early stopping (see [`fit_final`]).
fn final_fits(params: &TrainingParams) -> usize {
    if params.refit_final && params.validation_fraction > 0.0 { 2 } else { 1 }
}

/// The fewest threads per fit for fits to run at the same time: with fewer, XGBoost uses its
/// threads well enough that running the fits one after another is faster.
const MIN_THREADS_PER_CONCURRENT_FIT: usize = 2;

/// Splits `threads` between concurrent fits in proportion to their training calls (largest
/// remainder, ties to the earlier fit), giving each at least one thread.
///
/// Returns `None` if there are no fits or fewer than [`MIN_THREADS_PER_CONCURRENT_FIT`] threads
/// per fit; the fits then run one after another, each with all the threads.
fn split_threads(threads: NonZeroUsize, calls: &[usize]) -> Option<Vec<NonZeroUsize>> {
    let total_threads = threads.get();
    if calls.is_empty() || total_threads < MIN_THREADS_PER_CONCURRENT_FIT * calls.len() {
        return None;
    }
    // Every fit's first thread is given up front; the rest are shared by training calls.
    let spare = total_threads - calls.len();
    let total_calls: usize = calls.iter().sum::<usize>().max(1);
    let mut shares: Vec<usize> = calls.iter().map(|&n| spare * n / total_calls).collect();
    let mut remainders: Vec<(usize, usize)> =
        calls.iter().enumerate().map(|(i, &n)| (spare * n % total_calls, i)).collect();
    // Largest remainder first; on ties, the earlier fit.
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let left = spare - shares.iter().sum::<usize>();
    for &(_, i) in remainders.iter().take(left) {
        shares[i] += 1;
    }
    Some(shares.into_iter().map(|share| NonZeroUsize::MIN.saturating_add(share)).collect())
}

/// The folds of cross-validation (see [`TrainingParams::cv_folds`]) and every contig with calls.
fn cv_folds<S: CvSample>(
    samples: &[S],
    names: &[String],
    spec: &TrainingSpec<'_>,
) -> Result<(Vec<HeldOut>, BTreeSet<String>)> {
    let partition = group_samples(names, spec.groups)?;
    let mut calls: IndexMap<String, usize> = IndexMap::new();
    for sample in samples {
        for chrom in sample.calls().str_column("chrom")? {
            match calls.get_mut(chrom) {
                Some(n) => *n += 1,
                None => {
                    calls.insert(chrom.to_string(), 1);
                }
            }
        }
    }
    let all_contigs: BTreeSet<String> = calls.keys().cloned().collect();
    let folds = match spec.params.cv_folds {
        CvFolds::Samples => {
            if partition.len() < 2 {
                let described: Vec<String> = partition.iter().map(|g| g.join(",")).collect();
                bail!(
                    "leave-one-group-out cross-validation needs at least two samples in at least \
                     two groups; groups: {}",
                    described.join("; ")
                );
            }
            partition.into_iter().map(HeldOut::Samples).collect()
        }
        CvFolds::Chromosomes { folds } => {
            let calls: Vec<(String, usize)> = calls.into_iter().collect();
            let groups = contig_groups(&calls, usize::try_from(folds)?)?;
            let count: HashMap<&str, usize> = calls.iter().map(|(c, n)| (c.as_str(), *n)).collect();
            for (index, group) in groups.iter().enumerate() {
                let n: usize = group.iter().map(|c| count[c.as_str()]).sum();
                info!("CV fold {}: {} ({n} calls)", index + 1, group.join(","));
            }
            groups.into_iter().map(|group| HeldOut::Contigs(group.into_iter().collect())).collect()
        }
    };
    Ok((folds, all_contigs))
}

/// A sample's held-out predictions from one fold, and the calls it held out (`None`: all).
type FoldPart<F> = (F, Option<Vec<bool>>);

/// Combines a sample's held-out predictions from the folds that held out its calls.
fn merge_sample_folds<S: CvSample>(
    sample: &S,
    mut parts: Vec<FoldPart<S::Fold>>,
    num_class: usize,
) -> Result<S::Fold> {
    let rows = sample.calls().num_rows();
    if parts.is_empty() {
        if rows == 0 {
            return sample.to_fold(Probabilities::new(Vec::new(), num_class));
        }
        bail!("no cross-validation fold held out sample {}", sample.sample());
    }
    if parts.len() == 1 && parts[0].1.is_none() {
        return Ok(parts.pop().expect("one part").0);
    }
    let mut which = vec![usize::MAX; rows];
    for (part, (_, held_out)) in parts.iter().enumerate() {
        let Some(held_out) = held_out else {
            bail!(
                "sample {} is held out whole by one fold and in part by another",
                sample.sample()
            );
        };
        if held_out.len() != rows {
            bail!("a cross-validation fold rebuilt sample {} with other calls", sample.sample());
        }
        for (row, _) in held_out.iter().enumerate().filter(|(_, h)| **h) {
            if which[row] != usize::MAX {
                bail!("two cross-validation folds held out a call of sample {}", sample.sample());
            }
            which[row] = part;
        }
    }
    if which.contains(&usize::MAX) {
        bail!("no cross-validation fold held out some calls of sample {}", sample.sample());
    }
    S::merge_folds(parts.into_iter().map(|(fold, _)| fold).collect(), &which)
}

/// Cross-validation and the final fit on all samples.
///
/// By default ([`CvFolds::Samples`]) cross-validation is leave-one-group-out: each fold fits on
/// all other sample groups and predicts its held-out samples. Samples are grouped by
/// [`group_samples`]; with no groups this is leave-one-sample-out. With [`CvFolds::Chromosomes`],
/// each fold holds out a group of contigs (see [`contig_groups`]) from every sample, fits every
/// sample's calls on the other contigs, and predicts the held-out calls; each sample's predictions
/// are then assembled from the folds.
///
/// The final fit uses every sample (see [`TrainingParams::refit_final`]). The fits do not depend
/// on each other, and XGBoost's fits stop gaining from threads well before using a large machine
/// (each tree is grown one node at a time, in many short parallel steps), so with at least as many
/// threads as fits, the fits run at the same time, sharing the threads in proportion to their
/// training calls. The thread split depends only on the thread count and the samples, so the
/// models still depend only on them.
///
/// Returns one fold per sample, in input order, and the final booster.
///
/// # Arguments
///
/// * `samples` - the training samples (with distinct names; in at least two groups for sample
///   cross-validation).
/// * `spec` - the features, parameters, groups and threads.
/// * `num_class` - the number of classes.
/// * `refit` - rebuilds the samples for each fold (see [`Refit`]), when their calls depend on
///   something fitted to the training calls; otherwise `samples` are used in every fold.
///
/// # Errors
///
/// Returns an error if a sample name is repeated, the groups are invalid, there are too few
/// groups or contigs for the folds, `refit` returns other samples, or fitting fails.
pub fn cross_validate_and_fit<S>(
    samples: &[S],
    spec: &TrainingSpec<'_>,
    num_class: usize,
    refit: Option<Refit<'_, S>>,
) -> Result<(Vec<S::Fold>, Booster)>
where
    S: CvSample + Sync,
    S::Fold: Send,
{
    let names: Vec<String> = samples.iter().map(|s| s.sample().to_string()).collect();
    let (partition, all_contigs) = cv_folds(samples, &names, spec)?;
    // Each fold's samples, rebuilt before any fitting, so that fitting is all that runs at once.
    let mut timer = crate::common::PhaseTimer::start();
    let refitted: Vec<Option<Vec<S>>> = match refit {
        Some(refit) => partition
            .iter()
            .map(|held_out| -> Result<Option<Vec<S>>> {
                let rebuilt = refit(&held_out.training(&names, &all_contigs))?;
                if rebuilt.iter().map(CvSample::sample).ne(names.iter().map(String::as_str)) {
                    bail!("refit must return the samples {} in order", names.join(", "));
                }
                Ok(Some(rebuilt))
            })
            .collect::<Result<_>>()?,
        None => partition.iter().map(|_| None).collect(),
    };
    timer.lap("CV: rebuilding the folds' samples");
    let fold_samples: Vec<&[S]> =
        refitted.iter().map(|rebuilt| rebuilt.as_deref().unwrap_or(samples)).collect();
    // Per fold and sample, the held-out calls, and the calls fitted (`None` for every call) of
    // the samples the fold fits.
    let mut held_out_rows: Vec<Vec<Option<Rows>>> = Vec::new();
    let mut fitted_rows: Vec<Vec<Option<Option<Vec<bool>>>>> = Vec::new();
    for (held_out, fold) in partition.iter().zip(&fold_samples) {
        let rows: Vec<Option<Rows>> =
            fold.iter().map(|sample| held_out.rows(sample)).collect::<Result<_>>()?;
        fitted_rows.push(
            rows.iter()
                .map(|rows| match rows {
                    None => Some(None),
                    Some(Rows::All) => None,
                    Some(Rows::Mask(mask)) => Some(Some(mask.iter().map(|h| !h).collect())),
                })
                .collect(),
        );
        held_out_rows.push(rows);
    }
    let fitted_of = |index: usize| -> Vec<(&S, Option<&[bool]>)> {
        fold_samples[index]
            .iter()
            .zip(&fitted_rows[index])
            .filter_map(|(sample, rows)| rows.as_ref().map(|rows| (sample, rows.as_deref())))
            .collect()
    };

    // Fits the fold's training calls and predicts its held-out calls.
    let run_fold = |index: usize, threads: NonZeroUsize| -> FoldResult<S::Fold> {
        let fitted = fitted_of(index);
        let booster = fit(&fitted, spec.features, &spec.params, num_class, threads)?;
        let mut folds = Vec::new();
        for (i, (sample, rows)) in fold_samples[index].iter().zip(&held_out_rows[index]).enumerate()
        {
            let Some(rows) = rows else { continue };
            let calls = sample.calls();
            let predict = |calls: &FeatureTable| -> Result<Probabilities> {
                let values = to_matrix(calls, spec.features)?;
                predict_probabilities(&booster, &values, calls.num_rows(), spec.features, num_class)
            };
            let (probabilities, held_out) = match rows {
                Rows::All => (predict(calls)?, None),
                Rows::Mask(mask) => {
                    // Only the held-out calls are predicted; the others get zeros.
                    let predicted = predict(&calls.filter(mask)?)?;
                    let mut values = vec![0.0; calls.num_rows() * num_class];
                    let rows = mask.iter().enumerate().filter(|(_, h)| **h).map(|(row, _)| row);
                    for (k, row) in rows.enumerate() {
                        values[row * num_class..(row + 1) * num_class]
                            .copy_from_slice(predicted.row(k));
                    }
                    (Probabilities::new(values, num_class), Some(mask.clone()))
                }
            };
            folds.push((i, sample.to_fold(probabilities)?, held_out));
        }
        match &partition[index] {
            HeldOut::Samples(held_out) => {
                let fitted: Vec<&str> = fitted.iter().map(|(s, _)| s.sample()).collect();
                info!("CV: fitted {} -> scored {}", fitted.join(", "), held_out.join(", "));
            }
            held_out @ HeldOut::Contigs(_) => {
                info!(
                    "CV: fitted every sample off the held-out contigs -> scored {}",
                    held_out.describe()
                );
            }
        }
        Ok(folds)
    };
    let run_final =
        |threads: NonZeroUsize| fit_final(samples, spec.features, &spec.params, num_class, threads);

    // The final fit, then the folds: their training calls set their share of the threads. A
    // refitted final model is fitted twice, one fit after the other, so it counts its calls twice.
    let final_calls = samples.iter().map(|s| s.calls().num_rows()).sum::<usize>();
    let mut calls = vec![final_calls * final_fits(&spec.params)];
    for index in 0..partition.len() {
        let rows = fitted_of(index)
            .into_iter()
            .map(|(sample, rows)| {
                rows.map_or_else(
                    || sample.calls().num_rows(),
                    |rows| rows.iter().filter(|&&r| r).count(),
                )
            })
            .sum();
        calls.push(rows);
    }
    let (fold_results, booster) = match split_threads(spec.threads, &calls) {
        Some(threads) => {
            let described: Vec<String> = threads.iter().map(ToString::to_string).collect();
            info!(
                "Running the final fit and {} cross-validation fits at once, with {} threads",
                partition.len(),
                described.join(", ")
            );
            std::thread::scope(|scope| {
                let final_fit = scope.spawn(|| run_final(threads[0]));
                let folds: Vec<_> = (0..partition.len())
                    .map(|index| {
                        let fold_threads = threads[index + 1];
                        let run_fold = &run_fold;
                        scope.spawn(move || run_fold(index, fold_threads))
                    })
                    .collect();
                let folds: Vec<FoldResult<S::Fold>> = folds
                    .into_iter()
                    .map(|handle| handle.join().expect("a cross-validation fit panicked"))
                    .collect();
                (folds, final_fit.join().expect("the final fit panicked"))
            })
        }
        None => {
            let folds: Vec<FoldResult<S::Fold>> =
                (0..partition.len()).map(|index| run_fold(index, spec.threads)).collect();
            (folds, run_final(spec.threads))
        }
    };
    timer.lap("fitting the cross-validation folds and the final model");
    let mut parts: Vec<Vec<FoldPart<S::Fold>>> = samples.iter().map(|_| Vec::new()).collect();
    for result in fold_results {
        for (index, fold, held_out) in result? {
            parts[index].push((fold, held_out));
        }
    }
    let booster = booster?;
    let folds = samples
        .iter()
        .zip(parts)
        .map(|(sample, parts)| merge_sample_folds(sample, parts, num_class))
        .collect::<Result<_>>()?;
    Ok((folds, booster))
}

/// The model information of a trained model.
fn model_info(
    spec: &TrainingSpec<'_>,
    training_samples: Vec<String>,
    task: Task,
    thresholds: BTreeMap<Mode, Thresholds>,
    promote_thresholds: Option<crate::thresholds::PromoteThresholds>,
    stutter: Option<StutterModel>,
    regenotype: bool,
) -> ModelInfo {
    ModelInfo {
        format_version: MODEL_FORMAT_VERSION,
        variant_type: spec.variant_type,
        task,
        features: spec.features.to_vec(),
        region_names: spec.region_names.to_vec(),
        thresholds,
        promote_thresholds,
        training_samples,
        contigs: spec.contigs.to_vec(),
        reference_contigs: spec.reference_contigs.clone(),
        params: spec.params,
        stutter,
        regenotype,
        strlen_settings: spec.strlen_settings,
        pileup_settings: spec.pileup_settings,
        winnowr_version: VERSION.clone(),
        xgboost_version: XGBOOST_VERSION.clone(),
    }
}

/// Trains a genotype model: thresholds for every mode by leave-one-group-out cross-validation,
/// then the final fit on all samples.
///
/// # Arguments
///
/// * `samples` - the training samples (at least two groups).
/// * `spec` - the features, parameters, groups and threads.
/// * `grid` - candidate thresholds.
/// * `stutter` - the stutter model the samples' length-call features were computed with (fitted
///   on all the samples), saved with the model.
/// * `regenotype` - train the re-genotype class (the samples must be relabelled with it, see
///   [`SampleCalls::from_table`]); its threshold is searched with the genotyper's.
/// * `refit` - rebuilds the samples for each fold (see [`cross_validate_and_fit`]), so held-out samples'
///   features and labels do not depend on their own truth.
///
/// # Errors
///
/// Returns an error if cross-validation or fitting fails.
pub fn train_model(
    samples: &[SampleCalls],
    spec: &TrainingSpec<'_>,
    grid: &[f64],
    stutter: Option<StutterModel>,
    regenotype: bool,
    refit: Option<Refit<'_, SampleCalls>>,
) -> Result<WinnowrModel> {
    let num_class = num_classes(Task::Genotype, regenotype);
    let (folds, booster) = cross_validate_and_fit(samples, spec, num_class, refit)?;
    let mut timer = crate::common::PhaseTimer::start();
    let mut thresholds = BTreeMap::new();
    for mode in Mode::ALL {
        let chosen = search_thresholds(&folds, mode, grid, regenotype)?;
        let regenotype_text = if regenotype && mode == Mode::Genotyper {
            format!(" t_regenotype={}", chosen.t_regenotype)
        } else {
            String::new()
        };
        info!(
            "CV-chosen {mode}: t_drop={} t_flip={}{regenotype_text} (summed CV F1 gain {:+.5})",
            chosen.t_drop, chosen.t_flip, chosen.gain
        );
        thresholds.insert(mode, chosen);
    }
    timer.lap("searching the thresholds");
    let names = samples.iter().map(|s| s.sample.clone()).collect();
    let info = model_info(spec, names, Task::Genotype, thresholds, None, stutter, regenotype);
    Ok(WinnowrModel { booster, info })
}

/// Trains a promote model: thresholds by leave-one-group-out cross-validation, then the final
/// fit on all samples.
///
/// # Errors
///
/// Returns an error if cross-validation or fitting fails.
pub fn train_promote_model(
    samples: &[PromoteSampleCalls],
    spec: &TrainingSpec<'_>,
    promote_grid: &[f64],
    flip_grid: &[f64],
) -> Result<WinnowrModel> {
    let num_class = num_classes(Task::Promote, false);
    let (folds, booster) = cross_validate_and_fit(samples, spec, num_class, None)?;
    let mut timer = crate::common::PhaseTimer::start();
    let chosen = search_promote_thresholds(&folds, promote_grid, flip_grid)?;
    timer.lap("searching the thresholds");
    info!(
        "CV-chosen promote: t_promote={} t_flip={} (summed CV F1 gain {:+.5})",
        chosen.t_promote, chosen.t_flip, chosen.gain
    );
    let names = samples.iter().map(|s| s.sample.clone()).collect();
    let info = model_info(spec, names, Task::Promote, BTreeMap::new(), Some(chosen), None, false);
    Ok(WinnowrModel { booster, info })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use rstest::rstest;

    use super::*;

    fn threads(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    #[rstest]
    #[case::too_few_threads(5, &[10, 10, 10], None)]
    #[case::two_each(6, &[10, 10, 10], Some(vec![2, 2, 2]))]
    #[case::by_calls(30, &[1_557_019, 1_136_844, 420_231], Some(vec![14, 11, 5]))]
    #[case::at_least_one(8, &[1000, 1, 1], Some(vec![6, 1, 1]))]
    #[case::ties_to_the_earlier_fit(7, &[1, 1, 1], Some(vec![3, 2, 2]))]
    #[case::no_calls(4, &[0, 0], Some(vec![2, 2]))]
    #[case::no_fits(4, &[], None)]
    fn splits_threads_by_training_calls(
        #[case] total: usize,
        #[case] calls: &[usize],
        #[case] expected: Option<Vec<usize>>,
    ) {
        let split = split_threads(threads(total), calls);
        let split: Option<Vec<usize>> =
            split.map(|shares| shares.into_iter().map(NonZeroUsize::get).collect());
        if let Some(shares) = &split {
            assert_eq!(shares.iter().sum::<usize>(), total);
        }
        assert_eq!(split, expected);
    }

    /// A small deterministic generator (SplitMix64) for the toy samples.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A sample of noisy calls of two features, whose fold is its probabilities.
    struct ToySample {
        name: String,
        calls: FeatureTable,
        classes: Vec<usize>,
        excluded: Option<Vec<bool>>,
    }

    impl ToySample {
        fn new(name: &str, seed: u64, rows: usize) -> Self {
            Self::on_contigs(name, seed, rows, &["chr1"])
        }

        /// Calls on the contigs in turn, in equal runs.
        fn on_contigs(name: &str, seed: u64, rows: usize, contigs: &[&str]) -> Self {
            let mut rng = SplitMix64(seed);
            let (mut a, mut b, mut classes) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..rows {
                let (x, y) = (rng.next_f64(), rng.next_f64());
                a.push(x);
                b.push(y);
                let noisy = rng.next_f64() < 0.3;
                classes.push(if noisy {
                    (rng.next_u64() % 3) as usize
                } else {
                    (x * 3.0) as usize
                });
            }
            // Calls 997 bp apart, so early stopping holds out whole blocks of them.
            let chroms: Vec<&str> = (0..rows).map(|i| contigs[i * contigs.len() / rows]).collect();
            let positions: Vec<i64> = (0..rows).map(|i| 1 + 997 * i as i64).collect();
            let fields = vec![
                Field::new("chrom", DataType::Utf8, false),
                Field::new("pos", DataType::Int64, false),
                Field::new("a", DataType::Float64, true),
                Field::new("b", DataType::Float64, true),
            ];
            let arrays: Vec<arrow_array::ArrayRef> = vec![
                Arc::new(arrow_array::StringArray::from(chroms)),
                Arc::new(arrow_array::Int64Array::from(positions)),
                Arc::new(Float64Array::from(a)),
                Arc::new(Float64Array::from(b)),
            ];
            let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
            let info = FeatureTableInfo {
                sample: name.to_string(),
                contigs: vec![],
                region_names: vec![],
                fn_counts: None,
                promote_labelled: false,
                strlen: false,
                strlen_settings: None,
                pileup_settings: None,
                reference_contigs: None,
            };
            ToySample {
                name: name.to_string(),
                calls: FeatureTable::new(batch, info),
                classes,
                excluded: None,
            }
        }
    }

    impl CvSample for ToySample {
        type Fold = Vec<f64>;

        fn sample(&self) -> &str {
            &self.name
        }

        fn calls(&self) -> &FeatureTable {
            &self.calls
        }

        fn classes(&self) -> Vec<usize> {
            self.classes.clone()
        }

        fn excluded(&self) -> Option<&[bool]> {
            self.excluded.as_deref()
        }

        fn to_fold(&self, probabilities: Probabilities) -> Result<Vec<f64>> {
            Ok((0..self.calls.num_rows()).flat_map(|row| probabilities.row(row).to_vec()).collect())
        }

        fn merge_folds(parts: Vec<Vec<f64>>, which: &[usize]) -> Result<Vec<f64>> {
            let width = parts[0].len() / which.len().max(1);
            Ok(which
                .iter()
                .enumerate()
                .flat_map(|(row, &part)| parts[part][row * width..(row + 1) * width].to_vec())
                .collect())
        }
    }

    fn toy_spec<'a>(features: &'a [String], threads: NonZeroUsize) -> TrainingSpec<'a> {
        TrainingSpec {
            variant_type: VariantType::Snp,
            features,
            region_names: &[],
            contigs: &[],
            reference_contigs: None,
            params: TrainingParams { max_rounds: 30, ..TrainingParams::default() },
            groups: &[],
            threads,
            strlen_settings: None,
            pileup_settings: None,
        }
    }

    /// Each fold fitted on its own, with the given threads.
    fn fold_alone(samples: &[ToySample], held_out: usize, spec: &TrainingSpec<'_>) -> Vec<f64> {
        let others: Vec<&ToySample> =
            samples.iter().enumerate().filter(|(i, _)| *i != held_out).map(|(_, s)| s).collect();
        let others: Vec<(&ToySample, Option<&[bool]>)> =
            others.into_iter().map(|s| (s, None)).collect();
        let booster = fit(&others, spec.features, &spec.params, 3, spec.threads).unwrap();
        let sample = &samples[held_out];
        let values = to_matrix(sample.calls(), spec.features).unwrap();
        let rows = sample.calls().num_rows();
        sample
            .to_fold(predict_probabilities(&booster, &values, rows, spec.features, 3).unwrap())
            .unwrap()
    }

    /// A fit with enough threads to build its 3 classes' trees at once (9: 3 each), with calls held
    /// out for early stopping, is the fit 3 threads build in turn, and the same every time.
    #[test]
    fn a_fit_building_its_class_trees_at_once_is_that_of_its_share_of_the_threads() {
        let samples: Vec<ToySample> = [("S1", 1, 300), ("S2", 2, 200), ("S3", 3, 100)]
            .iter()
            .map(|&(name, seed, rows)| ToySample::new(name, seed, rows))
            .collect();
        let features = vec!["a".to_string(), "b".to_string()];
        let mut spec = toy_spec(&features, threads(9));
        spec.params.validation_fraction = 0.3;
        let fit_with = |n: usize| {
            fit_final(&samples, &features, &spec.params, 3, threads(n)).unwrap().to_json().unwrap()
        };
        let at_once = fit_with(9);
        assert_eq!(at_once, fit_with(9));
        assert_eq!(at_once, fit_with(3));
    }

    #[rstest]
    #[case::one_after_another(1, false)]
    #[case::at_once(8, false)]
    #[case::at_once_refitted(8, true)]
    fn concurrent_fits_match_fits_alone_with_their_threads(
        #[case] total: usize,
        #[case] refit: bool,
    ) {
        let samples: Vec<ToySample> = [("S1", 1, 300), ("S2", 2, 200), ("S3", 3, 100)]
            .iter()
            .map(|&(name, seed, rows)| ToySample::new(name, seed, rows))
            .collect();
        let features = vec!["a".to_string(), "b".to_string()];
        let mut spec = toy_spec(&features, threads(total));
        spec.params.refit_final = refit;
        let (folds, booster) = cross_validate_and_fit(&samples, &spec, 3, None).unwrap();
        // The final fit and the folds' training calls: 600 (twice over when refitted), 300, 400,
        // 500.
        let final_calls = if refit { 1200 } else { 600 };
        let split: Vec<NonZeroUsize> = split_threads(threads(total), &[final_calls, 300, 400, 500])
            .unwrap_or_else(|| vec![threads(total); 4]);
        if total == 8 {
            let expected = if refit { [3, 1, 2, 2] } else { [2, 2, 2, 2] };
            assert_eq!(split.iter().map(|t| t.get()).collect::<Vec<_>>(), expected);
        }
        let all: Vec<&ToySample> = samples.iter().collect();
        let all: Vec<(&ToySample, Option<&[bool]>)> = all.into_iter().map(|s| (s, None)).collect();
        let alone = fit_final(&samples, &features, &spec.params, 3, split[0]).unwrap();
        if !refit {
            let fitted = fit(&all, &features, &spec.params, 3, split[0]).unwrap();
            assert_eq!(fitted.to_json().unwrap(), alone.to_json().unwrap());
        }
        assert_eq!(booster.to_json().unwrap(), alone.to_json().unwrap());
        assert_eq!(folds.len(), samples.len());
        for (held_out, fold) in folds.iter().enumerate() {
            let fold_spec = toy_spec(&features, split[held_out + 1]);
            assert_eq!(*fold, fold_alone(&samples, held_out, &fold_spec), "fold {held_out}");
        }
    }

    #[test]
    fn models_record_the_evidence_settings_of_their_spec() {
        let features =
            vec!["a".to_string(), crate::pileup_features::PILEUP_FEATURES[0].to_string()];
        let strlen = Some(StrlenOptions::default());
        let pileup = Some(PileupOptions::default());
        let contigs = ["chr1".to_string()];
        let spec = TrainingSpec {
            contigs: &contigs,
            strlen_settings: strlen,
            pileup_settings: pileup,
            ..toy_spec(&features, threads(1))
        };
        let thresholds =
            Thresholds { t_drop: 0.5, t_flip: 0.6, gain: 0.0, t_regenotype: crate::common::NEVER };
        let thresholds = Mode::ALL.iter().map(|&mode| (mode, thresholds)).collect();
        let stutter = Some(crate::testing::fit_stutter_model(&[]));
        let samples = vec!["S1".to_string()];
        let info =
            model_info(&spec, samples.clone(), Task::Genotype, thresholds, None, stutter, false);
        // A model as training returns it is complete: it records its spec, the settings and the
        // versions, and validates.
        assert_eq!((info.format_version, info.task), (MODEL_FORMAT_VERSION, Task::Genotype));
        assert_eq!((info.variant_type, info.params), (spec.variant_type, spec.params));
        assert_eq!((&info.features[..], &info.contigs[..]), (&features[..], &contigs[..]));
        assert_eq!(info.training_samples, samples);
        assert_eq!((info.strlen_settings, info.pileup_settings), (strlen, pileup));
        assert_eq!(info.winnowr_version, *VERSION);
        assert_eq!(info.xgboost_version, *XGBOOST_VERSION);
        assert!(info.stutter.is_some() && !info.regenotype);
        info.validate().unwrap();
    }

    #[test]
    fn folds_fit_and_score_the_samples_refit_for_them() {
        let specs = [("S1", 1, 300), ("S2", 2, 200), ("S3", 3, 100)];
        let samples: Vec<ToySample> =
            specs.iter().map(|&(name, seed, rows)| ToySample::new(name, seed, rows)).collect();
        // Every sample rebuilt from seeds that depend on the fold's training samples, as a fold's
        // stutter model depends on its training calls.
        let rebuilt_for = |training: &BTreeSet<String>| -> Vec<ToySample> {
            let offset: u64 = training.iter().map(|name| name[1..].parse::<u64>().unwrap()).sum();
            specs
                .iter()
                .map(|&(name, seed, rows)| ToySample::new(name, 100 * offset + seed, rows))
                .collect()
        };
        let seen = std::sync::Mutex::new(Vec::new());
        let refit = |training: &FoldTraining| -> Result<Vec<ToySample>> {
            assert_eq!(training.contigs, None);
            seen.lock().unwrap().push(training.samples.iter().cloned().collect::<Vec<_>>());
            Ok(rebuilt_for(&training.samples))
        };
        let features = vec!["a".to_string(), "b".to_string()];
        let spec = toy_spec(&features, threads(1));
        let (folds, booster) = cross_validate_and_fit(&samples, &spec, 3, Some(&refit)).unwrap();
        let mut seen = seen.into_inner().unwrap();
        seen.sort();
        assert_eq!(seen, [vec!["S1", "S2"], vec!["S1", "S3"], vec!["S2", "S3"]]);
        for (held_out, fold) in folds.iter().enumerate() {
            let training: BTreeSet<String> = specs
                .iter()
                .filter(|(name, _, _)| *name != specs[held_out].0)
                .map(|(name, _, _)| name.to_string())
                .collect();
            let rebuilt = rebuilt_for(&training);
            assert_eq!(*fold, fold_alone(&rebuilt, held_out, &spec), "fold {held_out}");
            assert_ne!(*fold, fold_alone(&samples, held_out, &spec), "fold {held_out}");
        }
        // The final model fits the samples as given.
        let all: Vec<(&ToySample, Option<&[bool]>)> = samples.iter().map(|s| (s, None)).collect();
        let alone = fit(&all, &features, &spec.params, 3, threads(1)).unwrap();
        assert_eq!(booster.to_json().unwrap(), alone.to_json().unwrap());

        // A refit must return every sample, in order.
        let reordered = |training: &FoldTraining| -> Result<Vec<ToySample>> {
            let mut rebuilt = rebuilt_for(&training.samples);
            rebuilt.reverse();
            Ok(rebuilt)
        };
        let message =
            cross_validate_and_fit(&samples, &spec, 3, Some(&reordered)).err().unwrap().to_string();
        assert_eq!(message, "refit must return the samples S1, S2, S3 in order");
    }

    #[rstest]
    #[case::largest_first(&[("chr1", 100), ("chr2", 80), ("chr3", 60), ("chr4", 40)], 2,
                          &[&["chr1", "chr4"][..], &["chr2", "chr3"]])]
    #[case::ties_in_input_order(&[("a", 10), ("b", 10), ("c", 10)], 3, &[&["a"][..], &["b"], &["c"]])]
    #[case::small_contigs_fill_in(&[("a", 1), ("b", 50), ("c", 30), ("d", 25)], 2,
                                  &[&["a", "b"][..], &["c", "d"]])]
    fn groups_contigs_by_calls(
        #[case] calls: &[(&str, usize)],
        #[case] folds: usize,
        #[case] expected: &[&[&str]],
    ) {
        assert_eq!(contig_groups(calls, folds).unwrap(), expected);
    }

    #[test]
    fn contig_groups_need_two_folds_and_enough_contigs() {
        let calls = [("chr1", 10), ("chr2", 10)];
        let error = |folds: usize| contig_groups(&calls, folds).unwrap_err().to_string();
        assert_eq!(error(1), "chromosome cross-validation needs at least two folds, got 1");
        assert_eq!(
            error(3),
            "chromosome cross-validation with 3 folds needs at least 3 contigs with calls, got 2"
        );
        assert!(contig_groups(&calls, 2).is_ok());
    }

    #[test]
    fn chromosome_folds_hold_out_contigs_from_every_sample() {
        let contigs = ["chr1", "chr2", "chr3", "chr4"];
        let samples: Vec<ToySample> = [("S1", 1, 300), ("S2", 2, 200), ("S3", 3, 100)]
            .iter()
            .map(|&(name, seed, rows)| ToySample::on_contigs(name, seed, rows, &contigs))
            .collect();
        let features = vec!["a".to_string(), "b".to_string()];
        let mut spec = toy_spec(&features, threads(1));
        spec.params.cv_folds = CvFolds::Chromosomes { folds: 2 };
        // Sample groups are not used: every fold fits every sample.
        let groups = vec![vec!["S1".to_string(), "S2".to_string(), "S3".to_string()]];
        spec.groups = &groups;
        let (folds, _) = cross_validate_and_fit(&samples, &spec, 3, None).unwrap();
        // Each contig has 150 calls, so the groups alternate.
        for held_out in [["chr1", "chr3"], ["chr2", "chr4"]] {
            let masks: Vec<Vec<bool>> = samples
                .iter()
                .map(|s| {
                    let chroms = s.calls.str_column("chrom").unwrap();
                    chroms.iter().map(|c| held_out.contains(c)).collect()
                })
                .collect();
            let kept: Vec<Vec<bool>> =
                masks.iter().map(|m| m.iter().map(|h| !h).collect()).collect();
            let fitted: Vec<(&ToySample, Option<&[bool]>)> =
                samples.iter().zip(&kept).map(|(s, k)| (s, Some(k.as_slice()))).collect();
            let booster = fit(&fitted, &features, &spec.params, 3, spec.threads).unwrap();
            for ((sample, mask), fold) in samples.iter().zip(&masks).zip(&folds) {
                let values = to_matrix(sample.calls(), &features).unwrap();
                let rows = sample.calls().num_rows();
                let all = predict_probabilities(&booster, &values, rows, &features, 3).unwrap();
                for row in (0..rows).filter(|&row| mask[row]) {
                    assert_eq!(
                        &fold[row * 3..(row + 1) * 3],
                        all.row(row),
                        "{} {row}",
                        sample.name
                    );
                }
            }
        }
    }

    #[test]
    fn the_final_refit_fits_every_call_for_the_early_stopped_rounds() {
        let samples: Vec<ToySample> = [("S1", 1, 300), ("S2", 2, 200)]
            .iter()
            .map(|&(n, s, r)| ToySample::new(n, s, r))
            .collect();
        let features = vec!["a".to_string(), "b".to_string()];
        let params = TrainingParams {
            max_rounds: 500,
            early_stopping_rounds: 5,
            ..TrainingParams::default()
        };
        let stopped = fit_final(&samples, &features, &params, 3, threads(1)).unwrap();
        let rounds = stopped.boosted_rounds().unwrap();
        assert!(rounds < 500, "early stopping chose {rounds} rounds");
        let refit = TrainingParams { refit_final: true, ..params };
        let refitted = fit_final(&samples, &features, &refit, 3, threads(1)).unwrap();
        assert_eq!(refitted.boosted_rounds().unwrap(), rounds);
        let all: Vec<(&ToySample, Option<&[bool]>)> = samples.iter().map(|s| (s, None)).collect();
        let every_call = TrainingParams {
            max_rounds: u32::try_from(rounds).unwrap(),
            validation_fraction: 0.0,
            ..params
        };
        let expected = fit(&all, &features, &every_call, 3, threads(1)).unwrap();
        assert_eq!(refitted.to_json().unwrap(), expected.to_json().unwrap());
        assert_ne!(refitted.to_json().unwrap(), stopped.to_json().unwrap());
    }

    #[test]
    fn the_final_model_is_refit_only_when_calls_were_held_out() {
        let params = TrainingParams { refit_final: true, ..TrainingParams::default() };
        assert!(params.validation_fraction > 0.0);
        assert!(refits(&params, 1));
        // A validation fraction that held no block out stopped nothing early.
        assert!(!refits(&params, 0));
        assert!(!refits(&TrainingParams { validation_fraction: 0.0, ..params }, 1));
        assert!(!refits(&TrainingParams { refit_final: false, ..params }, 1));
    }

    #[test]
    fn excluded_calls_are_left_out_of_the_fit() {
        let features = vec!["a".to_string(), "b".to_string()];
        let spec = toy_spec(&features, threads(1));
        let mut sample = ToySample::new("S1", 1, 300);
        let excluded: Vec<bool> = (0..300).map(|row| row % 3 == 0).collect();
        let kept: Vec<bool> = excluded.iter().map(|&e| !e).collect();
        let reduced = ToySample {
            name: "S1".to_string(),
            calls: sample.calls.filter(&kept).unwrap(),
            classes: sample
                .classes
                .iter()
                .zip(&kept)
                .filter(|(_, k)| **k)
                .map(|(c, _)| *c)
                .collect(),
            excluded: None,
        };
        sample.excluded = Some(excluded);
        let with_mask = fit(&[(&sample, None)], &features, &spec.params, 3, spec.threads).unwrap();
        let without = fit(&[(&reduced, None)], &features, &spec.params, 3, spec.threads).unwrap();
        assert_eq!(with_mask.to_json().unwrap(), without.to_json().unwrap());
    }

    #[test]
    fn a_folds_stutter_model_is_fitted_on_its_samples_calls_on_its_contigs() {
        use crate::labels::TRUTH_DELTA_COLUMNS;
        // A labelled table with strlen evidence, for calls at (chrom, pos) with a number of
        // spanning reads, all of the reference length, at truth-homozygous sites.
        let table = |sample: &str, rows: &[(&str, i64, u64)]| {
            let calls: Vec<crate::testing::CallRow<'_>> =
                rows.iter().map(|r| (r.0, r.1, "AT", "A", "PASS", "indel", "0/1", false)).collect();
            let n = rows.len();
            let histograms: Vec<String> = rows.iter().map(|r| format!("0:{}", r.2)).collect();
            let mut table = crate::testing::call_table(&calls)
                .with_f64_column("sl_period", vec![1.0; n])
                .unwrap()
                .with_f64_column("sl_tract_len", vec![10.0; n])
                .unwrap()
                .with_f64_column("sl_n_spanning", rows.iter().map(|r| r.2 as f64).collect())
                .unwrap()
                .with_str_column("sl_histogram", &histograms)
                .unwrap()
                .with_f64_column(TRUTH_DELTA_COLUMNS[0], vec![0.0; n])
                .unwrap()
                .with_f64_column(TRUTH_DELTA_COLUMNS[1], vec![0.0; n])
                .unwrap();
            let info = table.info_mut();
            info.sample = sample.to_string();
            info.strlen = true;
            info.fn_counts = Some([("snp".to_string(), 0), ("indel".to_string(), 0)].into());
            table
        };
        let a = table("A", &[("chr1", 100, 10), ("chr2", 100, 20)]);
        let b = table("B", &[("chr1", 200, 40)]);
        let tables = [(&a, a.info()), (&b, b.info())];
        let reads = |samples: &[&str], contigs: Option<&[&str]>| -> u64 {
            let training = FoldTraining {
                samples: samples.iter().map(ToString::to_string).collect(),
                contigs: contigs.map(|c| c.iter().map(ToString::to_string).collect()),
            };
            let stutter = fit_fold_stutter(&tables, &training, StutterFit::default()).unwrap();
            stutter.n_reads().iter().flatten().sum()
        };
        assert_eq!(reads(&["A", "B"], None), 70);
        assert_eq!(reads(&["B"], None), 40);
        // Held-out chromosomes' calls are left out of the fit.
        assert_eq!(reads(&["A", "B"], Some(&["chr1"])), 50);
        assert_eq!(reads(&["A", "B"], Some(&["chr2"])), 20);
    }
}
