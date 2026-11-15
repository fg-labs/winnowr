//! A repeat-length stutter model, a stutter-aware diploid length caller, and its features.
//!
//! The stutter model gives `P(d | t)`: the probability that a read spanning a repeat tract shows
//! a length difference `d` (bp, observed minus reference tract) when the true allele's length
//! change is `t`. The error `e = d - t` is modelled as an empirical distribution over `e / k` in
//! `-MAX_UNITS..=MAX_UNITS` (`k` = repeat period) plus an "other" bucket (not a whole number of
//! units, or more than `MAX_UNITS` of them; spread evenly over `N_OTHER` values), per stratum of
//! period class (1 to 4, 5+) and allele tract length bin (`LENGTH_BINS`).
//!
//! The length caller considers the candidate allele length changes `{0}`, the called alleles,
//! and the observed differences seen in at least two reads (the five most common), and picks the
//! unordered pair `(a1, a2)` maximising `sum_d n_d ln(P(d | a1) / 2 + P(d | a2) / 2)`. Its call
//! and log-likelihood margins ([`LENGTH_CALL_FEATURES`]) are model features, and it proposes the
//! genotype of the re-genotype action.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::labels::TRUTH_DELTA_COLUMNS;
use crate::numeric::{numpy_sum, python_sum};
use crate::repeat_features::{called_deltas_of, ranked_lengths};
use crate::strlen::{Histogram, parse_histogram};
use crate::table::{FeatureTable, FeatureTableInfo};

/// The lower bounds (bp) of the allele tract length bins.
pub const LENGTH_BINS: [i64; 10] = [0, 5, 8, 11, 14, 17, 20, 25, 30, 40];

/// Errors of up to this many repeat units each way have their own probability.
pub const MAX_UNITS: i64 = 3;

/// The "other" error bucket is spread over this many values.
pub const N_OTHER: f64 = 20.0;

/// Period classes 1, 2, 3, 4 and 5+.
pub const N_PERIOD_CLASSES: usize = 5;

/// The pseudo-reads each stratum gets, distributed as its period class's pooled errors.
pub const STRATUM_PSEUDOCOUNT: f64 = 200.0;

/// At most this many observed length differences are candidate alleles.
pub const MAX_OBSERVED_CANDIDATES: usize = 5;

/// The error buckets: `-MAX_UNITS..=MAX_UNITS` units, then "other".
pub const N_ERRORS: usize = 2 * MAX_UNITS as usize + 2;

/// The model features from the length caller.
pub const LENGTH_CALL_FEATURES: [&str; 7] = [
    "lc_margin_called",
    "lc_flip_vs_called",
    "lc_margin_2nd",
    "lc_het",
    "lc_is_called",
    "lc_n_new",
    "lc_multiallelic",
];

/// The length caller's genotype: the sorted length changes of its two alleles (not features).
pub const LENGTH_CALL_COLUMNS: [&str; 2] = ["lc_delta_lo", "lc_delta_hi"];

/// The stutter period class of a repeat period: the period capped at [`N_PERIOD_CLASSES`].
#[must_use]
pub fn period_class(period: u64) -> usize {
    (period as usize).clamp(1, N_PERIOD_CLASSES)
}

/// The index of the [`LENGTH_BINS`] bin of an allele tract length (negative lengths fall in the
/// first bin).
#[must_use]
pub fn length_bin(length: i64) -> usize {
    LENGTH_BINS[1..].iter().position(|&upper| length < upper).unwrap_or(LENGTH_BINS.len() - 1)
}

/// The error bucket of a read's length error: `error / period + MAX_UNITS` for whole units within
/// [`MAX_UNITS`], else the "other" bucket.
#[must_use]
pub fn error_bucket(error: i64, period: u64) -> usize {
    if period == 0 {
        return N_ERRORS - 1;
    }
    let period = period as i64;
    // Python's floor division and modulo, which the Python version used.
    let (units, remainder) = (error.div_euclid(period), error.rem_euclid(period));
    if remainder == 0 && units.abs() <= MAX_UNITS {
        (units + MAX_UNITS) as usize
    } else {
        N_ERRORS - 1
    }
}

/// How a stutter model is fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StutterFit {
    /// Key errors by direction: an alternate allele's error is signed towards (negative) or away
    /// from (positive) the reference length, so that reference bias in its reads is not applied
    /// to the reference allele, whose errors are taken as symmetric.
    pub direction: bool,
    /// Fit only sites whose true allele changes the tract by whole repeat units.
    pub whole_unit: bool,
}

impl std::str::FromStr for StutterFit {
    type Err = anyhow::Error;

    /// Parses comma-separated settings: `direction` and `whole-unit` (`default` alone is none).
    fn from_str(text: &str) -> Result<Self> {
        let mut fit = StutterFit::default();
        for item in text.split(',').filter(|item| !item.is_empty() && *item != "default") {
            match item {
                "direction" => fit.direction = true,
                "whole-unit" => fit.whole_unit = true,
                _ => bail!("unknown stutter fit setting '{item}' (expected direction, whole-unit)"),
            }
        }
        Ok(fit)
    }
}

/// The sign that turns a read error at an allele into its direction-aware error: negative is
/// towards the reference length.
fn direction_sign(allele: i64) -> i64 {
    if allele < 0 { -1 } else { 1 }
}

/// Read length-error probabilities per period class and allele tract length bin.
#[derive(Debug, Clone, PartialEq)]
pub struct StutterModel {
    /// `[period class - 1][length bin]` -> the probability of each error bucket.
    probabilities: Vec<Vec<[f64; N_ERRORS]>>,
    /// The training reads per `[period class - 1][length bin]`.
    n_reads: Vec<Vec<u64>>,
    /// Errors are direction-aware (see [`StutterFit::direction`]).
    direction: bool,
}

impl StutterModel {
    /// A model from its tables.
    ///
    /// # Errors
    ///
    /// Returns an error if a table does not have a row per period class and length bin.
    pub fn new(probabilities: Vec<Vec<[f64; N_ERRORS]>>, n_reads: Vec<Vec<u64>>) -> Result<Self> {
        let shape_ok = probabilities.len() == N_PERIOD_CLASSES
            && probabilities.iter().all(|row| row.len() == LENGTH_BINS.len())
            && n_reads.len() == N_PERIOD_CLASSES
            && n_reads.iter().all(|row| row.len() == LENGTH_BINS.len());
        if !shape_ok {
            bail!("stutter model tables have the wrong shape");
        }
        Ok(StutterModel { probabilities, n_reads, direction: false })
    }

    /// The training reads per `[period class - 1][length bin]`.
    #[must_use]
    #[cfg(test)]
    pub fn n_reads(&self) -> &[Vec<u64>] {
        &self.n_reads
    }

    /// The error-bucket probabilities of an allele with this tract length.
    #[must_use]
    pub fn error_probabilities(&self, period: u64, allele_tract_len: i64) -> &[f64; N_ERRORS] {
        &self.probabilities[period_class(period) - 1][length_bin(allele_tract_len.max(0))]
    }

    /// `P(d | t)`: the probability of a read's observed length difference given the allele.
    ///
    /// # Arguments
    ///
    /// * `observed` - the read's length difference `d` (bp, versus the reference tract).
    /// * `allele` - the allele's length change `t` (bp).
    /// * `period` - the repeat period.
    /// * `tract_len` - the reference tract length.
    #[must_use]
    pub fn read_probability(&self, observed: i64, allele: i64, period: u64, tract_len: i64) -> f64 {
        let probabilities = self.error_probabilities(period, tract_len + allele);
        let error = observed - allele;
        let of = |bucket: usize| {
            if bucket == N_ERRORS - 1 {
                probabilities[bucket] / N_OTHER
            } else {
                probabilities[bucket]
            }
        };
        if !self.direction {
            return of(error_bucket(error, period));
        }
        if allele == 0 {
            // The reference allele: errors either way, as likely.
            return 0.5 * (of(error_bucket(error, period)) + of(error_bucket(-error, period)));
        }
        of(error_bucket(error * direction_sign(allele), period))
    }
}

/// One site of homozygous truth that a stutter model is fitted from.
#[derive(Debug, Clone, PartialEq)]
pub struct StutterSite {
    /// The repeat period.
    pub period: u64,
    /// The reference tract length.
    pub tract_len: i64,
    /// The true allele's length change.
    pub allele: i64,
    /// The reads per observed length difference.
    pub histogram: Histogram,
}

/// Fits a stutter model from reads at sites with one true allele (homozygous truth).
///
/// Each stratum's counts get [`STRATUM_PSEUDOCOUNT`] pseudo-reads distributed as its period class
/// (all length bins pooled, plus one pseudo-read per bucket). Sums are taken in NumPy's order, so
/// the model is the Python version's bit for bit.
#[must_use]
#[cfg(test)]
pub fn fit_stutter_model(sites: &[StutterSite]) -> StutterModel {
    fit_stutter_model_with(sites, StutterFit::default())
}

/// Fits a stutter model as `fit_stutter_model` does, with other [`StutterFit`] settings.
#[must_use]
pub fn fit_stutter_model_with(sites: &[StutterSite], fit: StutterFit) -> StutterModel {
    let mut counts = vec![vec![[0.0f64; N_ERRORS]; LENGTH_BINS.len()]; N_PERIOD_CLASSES];
    for site in sites {
        if fit.whole_unit && site.period > 0 && site.allele % site.period as i64 != 0 {
            continue;
        }
        let stratum =
            &mut counts[period_class(site.period) - 1][length_bin(site.tract_len + site.allele)];
        for (&observed, &n) in &site.histogram {
            let error = observed - site.allele;
            if fit.direction && site.allele == 0 {
                // No direction at the reference allele: half each way.
                stratum[error_bucket(error, site.period)] += 0.5 * n as f64;
                stratum[error_bucket(-error, site.period)] += 0.5 * n as f64;
            } else if fit.direction {
                stratum[error_bucket(error * direction_sign(site.allele), site.period)] += n as f64;
            } else {
                stratum[error_bucket(error, site.period)] += n as f64;
            }
        }
    }
    let normalized = |values: [f64; N_ERRORS]| -> [f64; N_ERRORS] {
        let total = numpy_sum(&values);
        values.map(|value| value / total)
    };
    let mut probabilities = Vec::with_capacity(N_PERIOD_CLASSES);
    let mut n_reads = Vec::with_capacity(N_PERIOD_CLASSES);
    for class_counts in &counts {
        // Whole-number counts, so the pooled sums are exact in any order.
        let mut pooled = [1.0f64; N_ERRORS];
        for stratum in class_counts {
            for (total, count) in pooled.iter_mut().zip(stratum) {
                *total += count;
            }
        }
        let base = normalized(pooled);
        probabilities.push(
            class_counts
                .iter()
                .map(|stratum| {
                    normalized(std::array::from_fn(|e| stratum[e] + STRATUM_PSEUDOCOUNT * base[e]))
                })
                .collect(),
        );
        n_reads.push(class_counts.iter().map(|stratum| numpy_sum(stratum) as u64).collect());
    }
    StutterModel { probabilities, n_reads, direction: fit.direction }
}

/// The truth-homozygous sites of labelled feature tables with strlen evidence: one per
/// `(chrom, pos)` among calls (any FILTER) with spanning reads whose truth allele length changes
/// are equal.
///
/// # Errors
///
/// Returns an error if a table lacks strlen evidence or truth allele length changes.
pub fn stutter_sites(tables: &[&FeatureTable]) -> Result<Vec<StutterSite>> {
    let mut sites = Vec::new();
    for table in tables {
        let mut missing: Vec<&str> = std::iter::once("sl_histogram")
            .chain(TRUTH_DELTA_COLUMNS)
            .filter(|column| !table.has_column(column))
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            bail!("the feature table lacks columns to fit stutter rates: {}", missing.join(", "));
        }
        let (chroms, positions) = (table.str_column("chrom")?, table.i64_column("pos")?);
        let n_spanning = table.f64_column("sl_n_spanning")?;
        let (periods, tract_lens) =
            (table.f64_column("sl_period")?, table.f64_column("sl_tract_len")?);
        let (lo, hi) =
            (table.f64_column(TRUTH_DELTA_COLUMNS[0])?, table.f64_column(TRUTH_DELTA_COLUMNS[1])?);
        let histograms = table.str_column("sl_histogram")?;
        let mut seen = BTreeSet::new();
        for row in 0..table.num_rows() {
            let usable = n_spanning[row] > 0.0 && !lo[row].is_nan() && lo[row] == hi[row];
            if !usable || !seen.insert((chroms[row], positions[row])) {
                continue;
            }
            sites.push(StutterSite {
                period: periods[row] as u64,
                tract_len: tract_lens[row] as i64,
                allele: lo[row] as i64,
                histogram: parse_histogram(histograms[row])?,
            });
        }
    }
    Ok(sites)
}

/// Fits the stutter model of a model with repeat-length features from its training tables.
///
/// # Errors
///
/// Returns an error if a table is unlabelled or has no strlen evidence, or no read is usable.
pub fn fit_stutter_from_feature_tables(
    tables: &[(&FeatureTable, &FeatureTableInfo)],
    fit: StutterFit,
) -> Result<StutterModel> {
    let lacking: Vec<&str> = tables
        .iter()
        .filter(|(_, info)| !info.strlen || !info.is_labelled())
        .map(|(_, info)| info.sample.as_str())
        .collect();
    if !lacking.is_empty() {
        bail!(
            "repeat-length features need labelled tables with strlen evidence (extract --happy \
             --strlen): {}",
            lacking.join(", ")
        );
    }
    let frames: Vec<&FeatureTable> = tables.iter().map(|(table, _)| *table).collect();
    let stutter = fit_stutter_model_with(&stutter_sites(&frames)?, fit);
    let n_reads: u64 = stutter.n_reads.iter().flatten().sum();
    if n_reads == 0 {
        let samples: Vec<&str> = tables.iter().map(|(_, info)| info.sample.as_str()).collect();
        bail!(
            "no reads to fit the stutter model: no indel call with spanning strlen reads has a \
             homozygous truth genotype in {}",
            samples.join(", ")
        );
    }
    log::info!("Fitted the stutter model on {n_reads} reads at truth-homozygous calls");
    Ok(stutter)
}

/// The JSON form of a stutter model, with its bins and error range so a model fitted with other
/// ones is rejected.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StutterModelJson {
    length_bins: Vec<i64>,
    max_units: i64,
    probabilities: Vec<Vec<Vec<f64>>>,
    n_reads: Vec<Vec<u64>>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    direction: bool,
}

impl Serialize for StutterModel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        StutterModelJson {
            length_bins: LENGTH_BINS.to_vec(),
            max_units: MAX_UNITS,
            probabilities: self
                .probabilities
                .iter()
                .map(|row| row.iter().map(|p| p.to_vec()).collect())
                .collect(),
            n_reads: self.n_reads.clone(),
            direction: self.direction,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StutterModel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let raw = StutterModelJson::deserialize(deserializer)?;
        if raw.length_bins != LENGTH_BINS || raw.max_units != MAX_UNITS {
            return Err(D::Error::custom(
                "the stutter model was fitted with other length bins or error range",
            ));
        }
        let probabilities = raw
            .probabilities
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|p| <[f64; N_ERRORS]>::try_from(p.as_slice()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| D::Error::custom("stutter model tables have the wrong shape"))?;
        let mut model = StutterModel::new(probabilities, raw.n_reads).map_err(D::Error::custom)?;
        model.direction = raw.direction;
        Ok(model)
    }
}

/// The length caller's result at one site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthCall {
    /// The best genotype's sorted allele length changes.
    pub alleles: (i64, i64),
    /// Its log-likelihood.
    pub log_likelihood: f64,
    /// The called genotype's (NaN if unknown or not a candidate).
    pub called_log_likelihood: f64,
    /// The het <-> hom flip of the called genotype's (NaN if none).
    pub flipped_log_likelihood: f64,
    /// The second-best genotype's (NaN if there is only one).
    pub second_log_likelihood: f64,
}

/// The het <-> hom flip of sorted length changes (`0/x` <-> `x/x`, sorted), if any.
fn flipped(called: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let (low, high) = called?;
    if low == high {
        return (low != 0).then(|| (low.min(0), low.max(0)));
    }
    if low == 0 || high == 0 {
        let alt = if low == 0 { high } else { low };
        return Some((alt, alt));
    }
    None
}

/// The most likely diploid genotype of a repeat tract given its reads.
///
/// Ties go to the genotype with the smaller length changes.
///
/// # Arguments
///
/// * `histogram` - reads per observed length difference (non-empty).
/// * `period` - the repeat period.
/// * `tract_len` - the reference tract length.
/// * `called` - the called genotype's sorted allele length changes, if known.
/// * `model` - the stutter model.
#[must_use]
pub fn call_length_genotype(
    histogram: &Histogram,
    period: u64,
    tract_len: i64,
    called: Option<(i64, i64)>,
    model: &StutterModel,
) -> LengthCall {
    let mut candidates: BTreeSet<i64> = BTreeSet::from([0]);
    if let Some((low, high)) = called {
        candidates.extend([low, high]);
    }
    let ranked = ranked_lengths(histogram);
    candidates.extend(
        ranked.iter().take(MAX_OBSERVED_CANDIDATES).filter(|&&(_, n)| n >= 2).map(|&(d, _)| d),
    );
    let candidates: Vec<i64> = candidates.into_iter().collect();
    let read_probabilities: BTreeMap<i64, Vec<f64>> = candidates
        .iter()
        .map(|&allele| {
            let row = histogram
                .keys()
                .map(|&d| model.read_probability(d, allele, period, tract_len))
                .collect();
            (allele, row)
        })
        .collect();
    let mut log_likelihoods: BTreeMap<(i64, i64), f64> = BTreeMap::new();
    for (i, &first) in candidates.iter().enumerate() {
        for &second in &candidates[i..] {
            let (p_first, p_second) = (&read_probabilities[&first], &read_probabilities[&second]);
            let terms = histogram
                .values()
                .enumerate()
                .map(|(k, &n)| n as f64 * (0.5 * p_first[k] + 0.5 * p_second[k]).ln());
            // Summed as the Python version did, so exact ties between genotypes stay exact.
            let log_likelihood = python_sum(terms);
            log_likelihoods.insert((first, second), log_likelihood);
        }
    }
    let mut ranked_genotypes: Vec<((i64, i64), f64)> =
        log_likelihoods.iter().map(|(&g, &l)| (g, l)).collect();
    // By decreasing log-likelihood, then increasing genotype (stable over the sorted map).
    ranked_genotypes.sort_by(|a, b| b.1.total_cmp(&a.1));
    let flip = flipped(called);
    LengthCall {
        alleles: ranked_genotypes[0].0,
        log_likelihood: ranked_genotypes[0].1,
        called_log_likelihood: called.map_or(f64::NAN, |c| log_likelihoods[&c]),
        flipped_log_likelihood: flip
            .and_then(|f| log_likelihoods.get(&f).copied())
            .unwrap_or(f64::NAN),
        second_log_likelihood: ranked_genotypes.get(1).map_or(f64::NAN, |&(_, l)| l),
    }
}

/// The length-caller genotype and features of one site: [`LENGTH_CALL_COLUMNS`] then
/// [`LENGTH_CALL_FEATURES`], or `None` for an empty histogram.
#[must_use]
pub fn length_call_features(
    histogram: &Histogram,
    period: u64,
    tract_len: i64,
    called: Option<(i64, i64)>,
    model: &StutterModel,
) -> Option<[f64; 9]> {
    if histogram.is_empty() {
        return None;
    }
    let call = call_length_genotype(histogram, period, tract_len, called, model);
    let (low, high) = call.alleles;
    let flag = |value: bool| f64::from(u8::from(value));
    let n_new = called.map_or(f64::NAN, |(c_low, c_high)| {
        let called_set: BTreeSet<i64> = [c_low, c_high].into();
        let new: BTreeSet<i64> =
            [low, high].into_iter().filter(|a| !called_set.contains(a)).collect();
        new.len() as f64
    });
    Some([
        low as f64,
        high as f64,
        call.log_likelihood - call.called_log_likelihood,
        call.flipped_log_likelihood - call.called_log_likelihood,
        call.log_likelihood - call.second_log_likelihood,
        flag(low != high),
        flag(called == Some(call.alleles)),
        n_new,
        flag(low != 0 && high != 0 && low != high),
    ])
}

/// Adds the length caller's genotype and features to calls with strlen evidence (missing where
/// there are no spanning reads).
///
/// # Errors
///
/// Returns an error if the table has no strlen evidence, or a histogram is malformed.
pub fn add_length_call_features(table: FeatureTable, model: &StutterModel) -> Result<FeatureTable> {
    if !table.has_column("sl_histogram") {
        bail!("the feature table has no strlen evidence (extract with --strlen)");
    }
    let n_spanning = table.f64_column("sl_n_spanning")?;
    let periods = table.f64_column("sl_period")?;
    let tract_lens = table.f64_column("sl_tract_len")?;
    let histograms = table.str_column("sl_histogram")?;
    let called = called_deltas_of(&table)?;
    // Calls are measured in parallel, then split into columns.
    let rows: Vec<Option<[f64; 9]>> = (0..table.num_rows())
        .into_par_iter()
        .map(|row| {
            // No spanning reads (zero, or no strlen evidence at all).
            if n_spanning[row].is_nan() || n_spanning[row] <= 0.0 {
                return Ok(None);
            }
            let histogram = parse_histogram(histograms[row])?;
            Ok(length_call_features(
                &histogram,
                periods[row] as u64,
                tract_lens[row] as i64,
                called[row],
                model,
            ))
        })
        .collect::<Result<_>>()?;
    let mut columns: Vec<Vec<f64>> = (0..9).map(|_| Vec::with_capacity(rows.len())).collect();
    for features in rows {
        for (column, value) in columns.iter_mut().zip(features.unwrap_or([f64::NAN; 9])) {
            column.push(value);
        }
    }
    let names = LENGTH_CALL_COLUMNS.iter().chain(&LENGTH_CALL_FEATURES);
    let mut table = table;
    for (name, values) in names.zip(columns) {
        table = table.with_f64_column(name, values)?;
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// A model whose error distribution is the same in every stratum.
    fn uniform_model(probabilities: [f64; N_ERRORS]) -> StutterModel {
        let rows = vec![vec![probabilities; LENGTH_BINS.len()]; N_PERIOD_CLASSES];
        StutterModel::new(rows, vec![vec![0; LENGTH_BINS.len()]; N_PERIOD_CLASSES]).unwrap()
    }

    /// Mostly exact reads, some one-unit stutter.
    fn model() -> StutterModel {
        uniform_model([0.001, 0.009, 0.05, 0.85, 0.05, 0.009, 0.001, 0.03])
    }

    #[rstest]
    #[case(1, 1)]
    #[case(4, 4)]
    #[case(7, 5)]
    fn period_classes(#[case] period: u64, #[case] expected: usize) {
        assert_eq!(period_class(period), expected);
    }

    #[rstest]
    #[case(-3, 0)]
    #[case(0, 0)]
    #[case(4, 0)]
    #[case(5, 1)]
    #[case(39, 8)]
    #[case(40, 9)]
    #[case(1000, 9)]
    fn length_bins(#[case] length: i64, #[case] expected: usize) {
        assert_eq!(length_bin(length), expected);
    }

    #[rstest]
    #[case(0, 2, 3)]
    #[case(-2, 2, 2)]
    #[case(6, 2, 6)]
    #[case(-6, 2, 0)]
    #[case(8, 2, 7)]
    #[case(-1, 2, 7)]
    #[case(1, 2, 7)]
    fn error_buckets(#[case] error: i64, #[case] period: u64, #[case] expected: usize) {
        assert_eq!(error_bucket(error, period), expected);
    }

    #[test]
    fn direction_aware_errors_are_signed_towards_the_reference() {
        let site = |allele: i64, histogram: &[(i64, u64)]| StutterSite {
            period: 1,
            tract_len: 10,
            allele,
            histogram: histogram.iter().copied().collect(),
        };
        // Insertions read one unit short, deletions one unit long: both towards the reference.
        let sites = vec![site(2, &[(2, 80), (1, 20)]), site(-2, &[(-2, 80), (-1, 20)])];
        let fit = StutterFit { direction: true, whole_unit: false };
        let model = fit_stutter_model_with(&sites, fit);
        let plain = fit_stutter_model(&sites);
        let p = |m: &StutterModel, observed: i64, allele: i64| {
            m.read_probability(observed, allele, 1, 10)
        };
        // Towards the reference is more likely than away, for either allele direction.
        assert!(p(&model, 1, 2) > p(&model, 3, 2));
        assert!(p(&model, -1, -2) > p(&model, -3, -2));
        // The reference allele's errors are symmetric.
        assert_eq!(p(&model, 1, 0), p(&model, -1, 0));
        // Without direction, errors keep their sign: an insertion's -1 and a deletion's +1 pool.
        assert_eq!(p(&plain, 1, 2), p(&plain, 1, 0));
        // The setting survives a round trip through JSON.
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(serde_json::from_str::<StutterModel>(&json).unwrap(), model);
        assert!(!serde_json::to_string(&plain).unwrap().contains("direction"));
    }

    #[test]
    fn model_tables_must_have_a_row_per_period_class_and_length_bin() {
        let errors = [1.0 / N_ERRORS as f64; N_ERRORS];
        let probabilities = vec![vec![errors; LENGTH_BINS.len()]; N_PERIOD_CLASSES];
        let n_reads = vec![vec![0; LENGTH_BINS.len()]; N_PERIOD_CLASSES];
        StutterModel::new(probabilities.clone(), n_reads.clone()).unwrap();
        let mut short_reads = n_reads.clone();
        short_reads[1].pop();
        let mut short_probabilities = probabilities.clone();
        short_probabilities[1].pop();
        for (p, n) in [(probabilities.clone(), short_reads), (short_probabilities, n_reads)] {
            let message = StutterModel::new(p, n).unwrap_err().to_string();
            assert_eq!(message, "stutter model tables have the wrong shape");
        }
    }

    #[test]
    fn whole_unit_fits_skip_partial_unit_alleles() {
        let site = |allele: i64| StutterSite {
            period: 2,
            tract_len: 10,
            allele,
            histogram: [(allele, 10)].into_iter().collect(),
        };
        let fit = StutterFit { direction: false, whole_unit: true };
        // Different read counts tell the sites apart; both are of period class 2 and in the
        // [11, 14) length bin.
        let mut partial = site(3);
        partial.histogram = [(3, 7)].into_iter().collect();
        let model = fit_stutter_model_with(&[site(2), partial], fit);
        let bin = length_bin(12);
        assert_eq!(bin, length_bin(13));
        let mut expected = vec![vec![0; LENGTH_BINS.len()]; N_PERIOD_CLASSES];
        expected[period_class(2) - 1][bin] = 10;
        assert_eq!(model.n_reads(), expected);
        assert_eq!(
            "direction,whole-unit".parse::<StutterFit>().unwrap(),
            StutterFit { direction: true, whole_unit: true }
        );
        assert!("other".parse::<StutterFit>().is_err());
    }

    #[test]
    fn read_probability_spreads_other_errors() {
        let model = model();
        assert_eq!(model.read_probability(2, 2, 1, 10), 0.85);
        assert_eq!(model.read_probability(1, 2, 1, 10), 0.05);
        assert_eq!(model.read_probability(10, 2, 1, 10), 0.03 / N_OTHER);
    }

    #[test]
    fn json_round_trips_and_checks_its_bins() {
        let model = model();
        let json = serde_json::to_string(&model).unwrap();
        assert!(
            json.starts_with(r#"{"length_bins":[0,5,8,11,14,17,20,25,30,40],"max_units":3,"#),
            "{json}"
        );
        assert_eq!(serde_json::from_str::<StutterModel>(&json).unwrap(), model);
        let other_bins = json.replace("[0,5,8,", "[0,4,8,");
        let message = serde_json::from_str::<StutterModel>(&other_bins).unwrap_err().to_string();
        assert!(message.contains("other length bins"), "{message}");
        let short = json.replacen(",0.001,0.03]", "]", 1);
        assert!(serde_json::from_str::<StutterModel>(&short).is_err());
    }

    #[test]
    fn calls_a_het_deletion_from_its_reads() {
        let histogram: Histogram = [(-2, 9), (-1, 1), (0, 10)].into();
        let call = call_length_genotype(&histogram, 1, 10, Some((0, 0)), &model());
        assert_eq!(call.alleles, (-2, 0));
        assert!(call.log_likelihood > call.called_log_likelihood);
        assert!(call.flipped_log_likelihood.is_nan(), "0/0 has no flip");
        assert!(call.second_log_likelihood < call.log_likelihood);

        let features = length_call_features(&histogram, 1, 10, Some((-2, -2)), &model()).unwrap();
        assert_eq!(features[..2], [-2.0, 0.0]);
        // The flip of -2/-2 is -2/0, the best call, so it is as far from the call as the best.
        assert_eq!(features[3], features[2]);
        assert_eq!(features[5], 1.0, "het");
        assert_eq!(features[6], 0.0, "not the called genotype");
        assert_eq!(features[7], 1.0, "one new allele (0)");
        assert_eq!(features[8], 0.0, "not multi-allelic");
        assert!(length_call_features(&Histogram::new(), 1, 10, None, &model()).is_none());
    }

    #[rstest]
    #[case(Some((0, 2)), Some((2, 2)))]
    #[case(Some((-2, 0)), Some((-2, -2)))]
    #[case(Some((2, 2)), Some((0, 2)))]
    #[case(Some((-2, -2)), Some((-2, 0)))]
    #[case(Some((0, 0)), None)]
    #[case(Some((1, 2)), None)]
    #[case(None, None)]
    fn flips(#[case] called: Option<(i64, i64)>, #[case] expected: Option<(i64, i64)>) {
        assert_eq!(flipped(called), expected);
    }

    #[test]
    fn ties_go_to_the_smaller_genotype() {
        // Every read is "other" for every candidate, so every genotype ties.
        let model = uniform_model([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let histogram: Histogram = [(7, 1)].into();
        let call = call_length_genotype(&histogram, 3, 10, Some((0, 3)), &model);
        assert_eq!(call.alleles, (0, 0));
    }

    /// A model fitted to homopolymer reads with 90% exact lengths and 5% one unit either way.
    fn fitted() -> StutterModel {
        let site = StutterSite {
            period: 1,
            tract_len: 10,
            allele: 0,
            histogram: Histogram::from([(-1, 5), (0, 90), (1, 5)]),
        };
        fit_stutter_model(&vec![site; 50])
    }

    #[test]
    fn fitting_learns_the_error_rates() {
        let model = fitted();
        let probabilities = model.error_probabilities(1, 10);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // 5000 reads: 4500 exact, 250 each way, smoothed by 200 pseudo-reads.
        let exact = MAX_UNITS as usize;
        assert!((probabilities[exact] - 0.9).abs() < 0.01);
        assert!((probabilities[exact - 1] - 0.05).abs() < 0.005);
        assert_eq!(model.n_reads()[0][length_bin(10)], 5000);
        // Strata without reads fall back to their period class; other classes are uniform.
        assert!((model.error_probabilities(1, 35)[exact] - 0.9).abs() < 0.01);
        for p in model.error_probabilities(3, 10) {
            assert!((p - 1.0 / N_ERRORS as f64).abs() < 1e-12);
        }
        // An allele's tract length picks the bin: +6 of a 10 bp tract is in the 14-17 bin.
        assert_eq!(model.read_probability(6, 6, 1, 10), model.error_probabilities(1, 16)[exact]);
        assert_eq!(fit_stutter_model(&[]).n_reads().iter().flatten().sum::<u64>(), 0);
    }

    /// A labelled table with strlen evidence: (pos, n_spanning, histogram, truth lo, truth hi).
    fn strlen_table(rows: &[(i64, f64, &str, f64, f64)]) -> FeatureTable {
        let calls: Vec<crate::testing::CallRow<'_>> =
            rows.iter().map(|r| ("chr1", r.0, "AT", "A", "PASS", "indel", "0/1", false)).collect();
        crate::testing::call_table(&calls)
            .with_f64_column("sl_period", vec![1.0; rows.len()])
            .unwrap()
            .with_f64_column("sl_tract_len", vec![10.0; rows.len()])
            .unwrap()
            .with_f64_column("sl_n_spanning", rows.iter().map(|r| r.1).collect())
            .unwrap()
            .with_str_column("sl_histogram", &rows.iter().map(|r| r.2).collect::<Vec<_>>())
            .unwrap()
            .with_f64_column(TRUTH_DELTA_COLUMNS[0], rows.iter().map(|r| r.3).collect())
            .unwrap()
            .with_f64_column(TRUTH_DELTA_COLUMNS[1], rows.iter().map(|r| r.4).collect())
            .unwrap()
    }

    #[test]
    fn fits_from_truth_homozygous_sites_with_spanning_reads() {
        let table = strlen_table(&[
            (100, 20.0, "0:20", -1.0, 0.0), // heterozygous truth
            (200, 20.0, "0:18;-1:2", 0.0, 0.0),
            (200, 20.0, "0:18;-1:2", 0.0, 0.0), // the same position again: used once
            (300, 0.0, ".", 0.0, 0.0),          // no spanning reads
            (400, 20.0, "0:20", f64::NAN, f64::NAN),
        ]);
        let sites = stutter_sites(&[&table]).unwrap();
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].histogram, Histogram::from([(-1, 2), (0, 18)]));
        let mut info = table.info().clone();
        info.strlen = true;
        info.fn_counts = Some([("snp".to_string(), 0), ("indel".to_string(), 0)].into());
        let model =
            fit_stutter_from_feature_tables(&[(&table, &info)], StutterFit::default()).unwrap();
        assert_eq!(model.n_reads().iter().flatten().sum::<u64>(), 20);

        let unusable = strlen_table(&[(100, 20.0, "0:20", -1.0, 0.0)]);
        let message = fit_stutter_from_feature_tables(
            &[(&unusable, &info), (&unusable, &info)],
            StutterFit::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(message.starts_with("no reads to fit the stutter model"), "{message}");
        assert!(message.ends_with("in S1, S1"), "{message}");
        let mut unlabelled = info.clone();
        unlabelled.fn_counts = None;
        unlabelled.sample = "b".to_string();
        let message = fit_stutter_from_feature_tables(
            &[(&table, &info), (&table, &unlabelled)],
            StutterFit::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(message.ends_with("--strlen): b"), "{message}");
        let message = stutter_sites(&[&crate::testing::call_table(&[])]).unwrap_err().to_string();
        assert_eq!(
            message,
            "the feature table lacks columns to fit stutter rates: sl_histogram, truth_delta_hi, \
             truth_delta_lo"
        );
    }
}
