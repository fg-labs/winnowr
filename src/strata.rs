//! Decisions, outcomes, F1 and P(keep) calibration stratified by benchmark-region membership.
//!
//! Given two truth sets for one sample (e.g. GIAB v4.2.1 and the T2T Q100 benchmark), each call
//! falls in one stratum by whether its REF span lies inside each truth set's confident regions:
//! inside both (`both`), inside only the second (`<second>-only`), inside only the first
//! (`<first>-only`), or inside neither (`neither`). Within a stratum:
//!
//! - the actions on every scored call are counted, with no truth needed (see [`ActionRates`]);
//! - where a truth set covers the stratum, the actions on its labelled calls are scored with the
//!   vcfeval arithmetic of [`crate::rescore`] (see [`Outcome`]), against the truth FNs that fall in
//!   the stratum, giving a stratum-local F1;
//! - the genotype model's P(keep) is binned against the observed fraction of `Keep` labels (see
//!   [`Calibration`]).
//!
//! Summed over strata, a truth set's outcome is a row of `winnowr evaluate` for its table: the
//! `--mode` row (`genotyper` or `veto`), `genotyper+promote` with a genotype and a promote model in
//! genotyper mode, or `promote` with only a promote model. `evaluate` has no row for a genotype and
//! a promote model in veto mode, so those outcomes match none.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::common::{PhaseTimer, Task, VariantType};
use crate::io::write_tsv;
use crate::labels::{
    HappyFn, Label, PromoteLabel, TRUTH_DELTA_COLUMNS, labelled_mask, read_happy_fns,
};
use crate::model::WinnowrModel;
use crate::numeric::metric_float;
use crate::policy::{ModelRowDecisions, Policy, check_model_kinds, decide_model_rows};
use crate::promote_labels::promote_mask;
use crate::regions::{RegionSet, ref_span};
use crate::rescore::{Counts, Mode};
use crate::samples::SampleCalls;
use crate::table::FeatureTable;

/// The label columns a truth set gives a feature table.
pub const TRUTH_LABEL_COLUMNS: [&str; 5] =
    ["label", "flip_ok", TRUTH_DELTA_COLUMNS[0], TRUTH_DELTA_COLUMNS[1], "promote_label"];

/// The columns that identify a call, which two tables of one sample's calls must share.
pub const CALL_KEY_COLUMNS: [&str; 6] = ["chrom", "pos", "ref", "alt", "filter", "variant_type"];

/// The number of P(keep) calibration bins (of equal width on `[0, 1]`).
pub const CALIBRATION_BINS: usize = 10;

/// Where a call lies relative to the confident regions of two truth sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stratum {
    /// Inside both truth sets' regions.
    Both,
    /// Inside the second truth set's regions only.
    SecondOnly,
    /// Inside the first truth set's regions only.
    FirstOnly,
    /// Inside neither truth set's regions.
    Neither,
}

impl Stratum {
    /// Every stratum, in report order.
    pub const ALL: [Stratum; 4] =
        [Stratum::Both, Stratum::SecondOnly, Stratum::FirstOnly, Stratum::Neither];

    /// The stratum of a call inside (or not) each truth set's regions.
    #[must_use]
    pub const fn of(in_first: bool, in_second: bool) -> Stratum {
        match (in_first, in_second) {
            (true, true) => Stratum::Both,
            (false, true) => Stratum::SecondOnly,
            (true, false) => Stratum::FirstOnly,
            (false, false) => Stratum::Neither,
        }
    }

    /// True if the truth set at `truth` (0 or 1) covers the stratum.
    #[must_use]
    pub const fn has_truth(self, truth: usize) -> bool {
        match self {
            Stratum::Both => true,
            Stratum::SecondOnly => truth == 1,
            Stratum::FirstOnly => truth == 0,
            Stratum::Neither => false,
        }
    }

    /// The stratum's name given the truth sets' names: `both`, `<second>-only`, `<first>-only`
    /// or `neither`.
    #[must_use]
    pub fn name(self, names: &[String; 2]) -> String {
        match self {
            Stratum::Both => "both".to_string(),
            Stratum::SecondOnly => format!("{}-only", names[1]),
            Stratum::FirstOnly => format!("{}-only", names[0]),
            Stratum::Neither => "neither".to_string(),
        }
    }
}

/// The stratum of each call of a table: whether its REF span lies inside each truth set's
/// confident regions (as promotion candidates are chosen; see
/// [`crate::promote_labels::promote_candidates`]).
///
/// # Errors
///
/// Returns an error if the table lacks `chrom`, `pos` or `ref`.
pub fn table_strata(table: &FeatureTable, beds: [&RegionSet; 2]) -> Result<Vec<Stratum>> {
    let (chroms, positions, refs) =
        (table.str_column("chrom")?, table.i64_column("pos")?, table.str_column("ref")?);
    Ok((0..table.num_rows())
        .map(|i| {
            let (start, end) = ref_span(positions[i], refs[i].len());
            Stratum::of(
                beds[0].contains(chroms[i], start, end),
                beds[1].contains(chroms[i], start, end),
            )
        })
        .collect())
}

/// Truth FN counts by variant type and stratum.
pub type StratumFnCounts = BTreeMap<(VariantType, Stratum), i64>;

/// Truth FN counts by variant type and region (an index into a list of region sets).
pub type RegionFnCounts = BTreeMap<(VariantType, usize), i64>;

/// The truth FN records of a hap.py output VCF on `contigs`, counted by variant type and stratum
/// (as `fn_strata`) and by variant type and each region of `regions` that the FN's REF span
/// overlaps (as hap.py's `--stratification` does).
///
/// # Errors
///
/// Returns an error if the VCF cannot be read or is not hap.py output.
pub fn fn_counts<S: AsRef<str>>(
    happy: &Path,
    contigs: &[S],
    beds: [&RegionSet; 2],
    regions: &[RegionSet],
) -> Result<(StratumFnCounts, RegionFnCounts)> {
    let mut counts = BTreeMap::new();
    let mut region_counts = BTreeMap::new();
    for record in read_happy_fns(happy, |chrom| contigs.iter().any(|c| c.as_ref() == chrom))? {
        let HappyFn { chrom, variant_type, span: (start, end) } = record?;
        let stratum =
            Stratum::of(beds[0].contains(&chrom, start, end), beds[1].contains(&chrom, start, end));
        *counts.entry((variant_type, stratum)).or_insert(0) += 1;
        for (r, region) in regions.iter().enumerate() {
            if region.overlaps(&chrom, start, end) {
                *region_counts.entry((variant_type, r)).or_insert(0) += 1;
            }
        }
    }
    Ok((counts, region_counts))
}

/// Which of a list of region sets (e.g. GIAB stratifications) each call of a table overlaps.
///
/// A call is in a region set if its REF span shares at least one base with it, as hap.py
/// assigns records to `--stratification` regions; region sets may overlap one another, so a call
/// may be in several.
#[derive(Debug, Clone)]
pub struct RegionMembership {
    /// The number of region sets.
    regions: usize,
    /// 64-bit words per call.
    words: usize,
    /// Per call, `words` words whose bit `r` is set if the call is in region set `r`.
    bits: Vec<u64>,
}

impl RegionMembership {
    /// The membership of each call of a table in each region set.
    ///
    /// # Errors
    ///
    /// Returns an error if the table lacks `chrom`, `pos` or `ref`.
    pub fn of_table(table: &FeatureTable, regions: &[RegionSet]) -> Result<Self> {
        let (chroms, positions, refs) =
            (table.str_column("chrom")?, table.i64_column("pos")?, table.str_column("ref")?);
        let words = regions.len().div_ceil(64);
        let mut bits = vec![0u64; table.num_rows() * words];
        for row in 0..table.num_rows() {
            let (start, end) = ref_span(positions[row], refs[row].len());
            for (r, region) in regions.iter().enumerate() {
                if region.overlaps(chroms[row], start, end) {
                    bits[row * words + r / 64] |= 1 << (r % 64);
                }
            }
        }
        Ok(RegionMembership { regions: regions.len(), words, bits })
    }

    /// The indices of the region sets a call is in, in order.
    pub fn of(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        let words = &self.bits[row * self.words..(row + 1) * self.words];
        (0..self.regions).filter(move |&r| words[r / 64] & (1 << (r % 64)) != 0)
    }
}

/// The table with the label columns and label metadata of another table of the same calls (e.g.
/// the same VCF labelled against another truth set).
///
/// # Errors
///
/// Returns an error if the tables have different numbers of rows, differ in any call key, or
/// `labels` is not labelled or lacks a hap.py label column `table` has. If `labels` lacks only
/// the promote labels, the result has none (all NaN, and not `promote_labelled`).
pub fn with_truth_labels(table: &FeatureTable, labels: &FeatureTable) -> Result<FeatureTable> {
    if table.num_rows() != labels.num_rows() {
        bail!(
            "the feature tables have {} and {} calls: they must be extracted from the same VCF",
            table.num_rows(),
            labels.num_rows()
        );
    }
    if !labels.info().is_labelled() {
        bail!("the feature table for {} has no hap.py labels", labels.info().sample);
    }
    let (left, right) = (table.i64_column("pos")?, labels.i64_column("pos")?);
    if let Some(row) = (0..left.len()).find(|&i| left[i] != right[i]) {
        bail!("the feature tables differ at call {row}: they must be extracted from the same VCF");
    }
    for name in CALL_KEY_COLUMNS.into_iter().filter(|name| *name != "pos") {
        let (left, right) = (table.str_column(name)?, labels.str_column(name)?);
        if let Some(row) = (0..left.len()).find(|&i| left[i] != right[i]) {
            bail!(
                "the feature tables differ in {name} at call {row}: they must be extracted from \
                 the same VCF"
            );
        }
    }
    let mut out = table.clone();
    for name in TRUTH_LABEL_COLUMNS {
        if let Some(column) = labels.batch().column_by_name(name) {
            out = out.with_column(name, column.clone())?;
        } else if name == "promote_label" && out.has_column(name) {
            // Promote labels come only from extracting with --truth-vcf, and are needed only to
            // score a promote model, which checks `promote_labelled` (copied below).
            out = out.with_f64_column(name, vec![f64::NAN; labels.num_rows()])?;
        } else if out.has_column(name) {
            bail!("the feature table for {} lacks {name}", labels.info().sample);
        }
    }
    let info = out.info_mut();
    info.fn_counts.clone_from(&labels.info().fn_counts);
    info.promote_labelled = labels.info().promote_labelled;
    Ok(out)
}

/// A genotype model's action on a PASS call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallAction {
    /// No action.
    Keep,
    /// Dropped (including re-genotyped to `0/0`).
    Drop,
    /// Flipped het <-> hom.
    Flip,
    /// Re-genotyped to the length caller's genotype.
    Regenotype,
}

impl CallAction {
    /// The action given drop, flip and re-genotype flags (the first set wins).
    #[must_use]
    pub const fn of(drop: bool, flip: bool, regenotype: bool) -> CallAction {
        if drop {
            CallAction::Drop
        } else if flip {
            CallAction::Flip
        } else if regenotype {
            CallAction::Regenotype
        } else {
            CallAction::Keep
        }
    }
}

/// The actions on one stratum's labelled calls and their outcomes against one truth set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Outcome {
    /// Labelled PASS calls.
    pub labelled: i64,
    /// PASS calls dropped.
    pub dropped: i64,
    /// True-positive PASS calls dropped.
    pub dropped_tp: i64,
    /// PASS calls flipped.
    pub flipped: i64,
    /// Flips that produced the truth genotype.
    pub flips_fixed: i64,
    /// Flips of true-positive calls.
    pub flips_broke_tp: i64,
    /// PASS calls re-genotyped.
    pub regenotyped: i64,
    /// Re-genotypes that produced the truth genotype.
    pub regenotypes_fixed: i64,
    /// Re-genotypes of true-positive calls.
    pub regenotypes_broke_tp: i64,
    /// Labelled promotion candidates (non-PASS calls).
    pub candidates: i64,
    /// Candidates promoted.
    pub promoted: i64,
    /// Promotions that turned a truth FN into a TP.
    pub promoted_correct: i64,
    /// Promotions whose genotype was flipped.
    pub promoted_flipped: i64,
    /// The counts before any action (FNs from the truth FNs in the stratum).
    pub before: Counts,
    /// The change in counts from the actions.
    pub delta: Counts,
}

impl Outcome {
    /// Adds a labelled PASS call and the action taken on it.
    pub fn add_call(
        &mut self,
        label: Label,
        flip_ok: bool,
        regenotype_ok: bool,
        action: CallAction,
    ) {
        let is_tp = label == Label::Keep;
        self.labelled += 1;
        if is_tp {
            self.before.tp += 1;
        } else {
            self.before.fp += 1;
        }
        let d = &mut self.delta;
        match action {
            CallAction::Keep => {}
            CallAction::Drop => {
                self.dropped += 1;
                if is_tp {
                    self.dropped_tp += 1;
                    d.tp -= 1;
                    d.fn_ += 1;
                } else {
                    d.fp -= 1;
                }
            }
            CallAction::Flip | CallAction::Regenotype => {
                let ok = if action == CallAction::Flip {
                    self.flipped += 1;
                    flip_ok
                } else {
                    self.regenotyped += 1;
                    regenotype_ok
                };
                let (fixed, broke) = if action == CallAction::Flip {
                    (&mut self.flips_fixed, &mut self.flips_broke_tp)
                } else {
                    (&mut self.regenotypes_fixed, &mut self.regenotypes_broke_tp)
                };
                if is_tp {
                    *broke += 1;
                    d.tp -= 1;
                    d.fp += 1;
                    d.fn_ += 1;
                } else if ok {
                    *fixed += 1;
                    d.tp += 1;
                    d.fp -= 1;
                    d.fn_ -= 1;
                }
            }
        }
    }

    /// Adds a labelled promotion candidate and whether it was promoted (and flipped).
    pub fn add_candidate(&mut self, label: PromoteLabel, promote: bool, flip: bool) {
        self.candidates += 1;
        if !promote {
            return;
        }
        self.promoted += 1;
        if flip {
            self.promoted_flipped += 1;
        }
        let right = (label == PromoteLabel::Promote && !flip)
            || (label == PromoteLabel::PromoteFlip && flip);
        if right {
            self.promoted_correct += 1;
            self.delta.tp += 1;
            self.delta.fn_ -= 1;
        } else {
            self.delta.fp += 1;
        }
    }

    /// The counts after the actions.
    #[must_use]
    pub fn after(&self) -> Counts {
        Counts {
            tp: self.before.tp + self.delta.tp,
            fp: self.before.fp + self.delta.fp,
            fn_: self.before.fn_ + self.delta.fn_,
        }
    }

    /// Adds another outcome's counts.
    pub fn add(&mut self, other: &Outcome) {
        self.labelled += other.labelled;
        self.dropped += other.dropped;
        self.dropped_tp += other.dropped_tp;
        self.flipped += other.flipped;
        self.flips_fixed += other.flips_fixed;
        self.flips_broke_tp += other.flips_broke_tp;
        self.regenotyped += other.regenotyped;
        self.regenotypes_fixed += other.regenotypes_fixed;
        self.regenotypes_broke_tp += other.regenotypes_broke_tp;
        self.candidates += other.candidates;
        self.promoted += other.promoted;
        self.promoted_correct += other.promoted_correct;
        self.promoted_flipped += other.promoted_flipped;
        for (sum, part) in [(&mut self.before, other.before), (&mut self.delta, other.delta)] {
            sum.tp += part.tp;
            sum.fp += part.fp;
            sum.fn_ += part.fn_;
        }
    }
}

/// Predicted P(keep) against the observed fraction of `Keep` labels, in equal-width bins.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Calibration {
    /// Calls per bin.
    pub calls: [i64; CALIBRATION_BINS],
    /// The sum of P(keep) per bin.
    pub sum_p: [f64; CALIBRATION_BINS],
    /// `Keep`-labelled calls per bin.
    pub keep: [i64; CALIBRATION_BINS],
}

impl Calibration {
    /// The bin of a probability (`1.0` falls in the last bin; NaN in the first).
    #[must_use]
    pub fn bin(p: f64) -> usize {
        if p.is_nan() {
            return 0;
        }
        ((p * CALIBRATION_BINS as f64) as usize).min(CALIBRATION_BINS - 1)
    }

    /// Adds a call's P(keep) and whether it is labelled `Keep`.
    pub fn add_call(&mut self, p_keep: f64, keep: bool) {
        let bin = Self::bin(p_keep);
        self.calls[bin] += 1;
        self.sum_p[bin] += p_keep;
        self.keep[bin] += i64::from(keep);
    }

    /// Adds another calibration's counts.
    pub fn add(&mut self, other: &Calibration) {
        for bin in 0..CALIBRATION_BINS {
            self.calls[bin] += other.calls[bin];
            self.sum_p[bin] += other.sum_p[bin];
            self.keep[bin] += other.keep[bin];
        }
    }

    /// The expected calibration error: the call-weighted mean over bins of
    /// |mean P(keep) - observed keep fraction| (0 with no calls).
    #[must_use]
    pub fn expected_calibration_error(&self) -> f64 {
        let total: i64 = self.calls.iter().sum();
        if total == 0 {
            return 0.0;
        }
        (0..CALIBRATION_BINS)
            .filter(|&bin| self.calls[bin] > 0)
            .map(|bin| (self.sum_p[bin] - self.keep[bin] as f64).abs())
            .sum::<f64>()
            / total as f64
    }
}

/// The actions on every scored call of one stratum, chromosome and variant type (no truth
/// needed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ActionRates {
    /// PASS calls (scored by a genotype model).
    pub pass_calls: i64,
    /// PASS calls dropped.
    pub dropped: i64,
    /// PASS calls flipped.
    pub flipped: i64,
    /// PASS calls re-genotyped.
    pub regenotyped: i64,
    /// Non-PASS calls that do not overlap a PASS call (scored by a promote model).
    pub eligible: i64,
    /// Non-PASS calls promoted.
    pub promoted: i64,
    /// Promoted calls whose genotype was flipped.
    pub promoted_flipped: i64,
}

impl ActionRates {
    /// Adds a PASS call and the action taken on it.
    pub fn add_call(&mut self, action: CallAction) {
        self.pass_calls += 1;
        match action {
            CallAction::Keep => {}
            CallAction::Drop => self.dropped += 1,
            CallAction::Flip => self.flipped += 1,
            CallAction::Regenotype => self.regenotyped += 1,
        }
    }

    /// Adds a non-PASS call and whether it was promoted (and flipped).
    pub fn add_candidate(&mut self, promote: bool, flip: bool) {
        self.eligible += 1;
        self.promoted += i64::from(promote);
        self.promoted_flipped += i64::from(promote && flip);
    }

    /// Adds another stratum's counts.
    pub fn add(&mut self, other: &ActionRates) {
        self.pass_calls += other.pass_calls;
        self.dropped += other.dropped;
        self.flipped += other.flipped;
        self.regenotyped += other.regenotyped;
        self.eligible += other.eligible;
        self.promoted += other.promoted;
        self.promoted_flipped += other.promoted_flipped;
    }
}

/// `numerator / denominator`, or 0 when the denominator is 0.
fn ratio(numerator: i64, denominator: i64) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

/// The name of the summary over all strata or chromosomes.
pub const ALL: &str = "all";

/// Stratified decisions, outcomes and calibration for one sample.
#[derive(Debug, Clone)]
pub struct StratifiedReport {
    /// The sample name.
    pub sample: String,
    /// The names of the two truth sets.
    pub names: [String; 2],
    /// The decision mode of the genotype models.
    pub mode: Mode,
    /// Outcomes by truth set (0 or 1), variant type and stratum.
    pub outcomes: BTreeMap<(usize, VariantType, Stratum), Outcome>,
    /// P(keep) calibration by truth set, variant type and stratum.
    pub calibration: BTreeMap<(usize, VariantType, Stratum), Calibration>,
    /// Action counts by variant type, chromosome (index into `chroms`) and stratum.
    pub rates: BTreeMap<(VariantType, usize, Stratum), ActionRates>,
    /// The chromosomes, in the order first seen.
    pub chroms: Vec<String>,
    /// The index in `chroms` of the chromosome [`StratifiedReport::rates`] last looked up: calls
    /// come grouped by chromosome, so this saves a scan of `chroms` per call.
    last_chrom: Option<usize>,
}

impl StratifiedReport {
    /// An empty report.
    #[must_use]
    pub fn new(sample: &str, names: [String; 2], mode: Mode) -> Self {
        StratifiedReport {
            sample: sample.to_string(),
            names,
            mode,
            outcomes: BTreeMap::new(),
            calibration: BTreeMap::new(),
            rates: BTreeMap::new(),
            chroms: Vec::new(),
            last_chrom: None,
        }
    }

    /// The outcome of a truth set, variant type and stratum, created empty if missing.
    pub fn outcome(&mut self, truth: usize, vt: VariantType, stratum: Stratum) -> &mut Outcome {
        self.outcomes.entry((truth, vt, stratum)).or_default()
    }

    /// The calibration of a truth set, variant type and stratum, created empty if missing.
    pub fn calibration(
        &mut self,
        truth: usize,
        vt: VariantType,
        stratum: Stratum,
    ) -> &mut Calibration {
        self.calibration.entry((truth, vt, stratum)).or_default()
    }

    /// The action counts of a variant type, chromosome and stratum, created empty if missing.
    pub fn rates(&mut self, vt: VariantType, chrom: &str, stratum: Stratum) -> &mut ActionRates {
        let last = self.last_chrom.filter(|&i| self.chroms.get(i).is_some_and(|c| c == chrom));
        let index = match last.or_else(|| self.chroms.iter().position(|c| c == chrom)) {
            Some(index) => index,
            None => {
                self.chroms.push(chrom.to_string());
                self.chroms.len() - 1
            }
        };
        self.last_chrom = Some(index);
        self.rates.entry((vt, index, stratum)).or_default()
    }

    /// The variant types with any outcome or action counts, in order.
    fn variant_types(&self) -> Vec<VariantType> {
        let mut types: Vec<VariantType> =
            self.outcomes.keys().map(|k| k.1).chain(self.rates.keys().map(|k| k.0)).collect();
        types.sort_unstable();
        types.dedup();
        types
    }

    /// The outcome rows: per truth set and variant type, each stratum the truth set covers or
    /// has labelled calls in, then `all`. Each row is `(truth, variant type, stratum, outcome)`.
    #[must_use]
    pub fn outcome_rows(&self) -> Vec<(usize, VariantType, String, Outcome)> {
        let mut rows = Vec::new();
        for truth in 0..2 {
            for vt in self.variant_types() {
                let outcomes: Vec<(Stratum, Outcome)> = Stratum::ALL
                    .into_iter()
                    .map(|s| (s, self.outcomes.get(&(truth, vt, s)).copied().unwrap_or_default()))
                    .collect();
                let mut total = Outcome::default();
                for (_, outcome) in &outcomes {
                    total.add(outcome);
                }
                if total == Outcome::default() {
                    continue;
                }
                for (stratum, outcome) in outcomes {
                    if stratum.has_truth(truth) || outcome != Outcome::default() {
                        rows.push((truth, vt, stratum.name(&self.names), outcome));
                    }
                }
                rows.push((truth, vt, ALL.to_string(), total));
            }
        }
        rows
    }

    /// Writes the outcomes (see [`OUTCOME_COLUMNS`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_outcomes(&self, path: &Path) -> Result<()> {
        let lines = self.outcome_rows().into_iter().map(|(truth, vt, stratum, o)| {
            outcome_fields(&self.sample, &self.names[truth], stratum, vt, self.mode, &o)
        });
        write_tsv(path, &OUTCOME_COLUMNS, lines)
    }

    /// Writes the P(keep) calibration (see [`CALIBRATION_COLUMNS`]): per truth set, variant type
    /// and stratum (and `all`), one row per non-empty bin and a `[0, 1]` summary row whose
    /// last column is the expected calibration error.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_calibration(&self, path: &Path) -> Result<()> {
        let mut lines = Vec::new();
        for truth in 0..2 {
            for vt in self.variant_types() {
                let mut total = Calibration::default();
                let mut groups = Vec::new();
                for stratum in Stratum::ALL {
                    if let Some(c) = self.calibration.get(&(truth, vt, stratum)) {
                        total.add(c);
                        groups.push((stratum.name(&self.names), *c));
                    }
                }
                if groups.is_empty() {
                    continue;
                }
                groups.push((ALL.to_string(), total));
                for (stratum, c) in groups {
                    calibration_lines(
                        &mut lines,
                        [&self.sample, &self.names[truth], &stratum],
                        vt,
                        &c,
                    );
                }
            }
        }
        write_tsv(path, &CALIBRATION_COLUMNS, lines.into_iter())
    }

    /// Writes the action counts and rates of every scored call (see [`RATE_COLUMNS`]): per
    /// stratum (and `all`), variant type and chromosome (and `all`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_rates(&self, path: &Path) -> Result<()> {
        let mut lines = Vec::new();
        let strata: Vec<Option<Stratum>> =
            Stratum::ALL.into_iter().map(Some).chain(std::iter::once(None)).collect();
        for stratum in &strata {
            for vt in self.variant_types() {
                let chroms: Vec<Option<usize>> =
                    std::iter::once(None).chain((0..self.chroms.len()).map(Some)).collect();
                for chrom in chroms {
                    let mut r = ActionRates::default();
                    for ((v, c, s), rates) in &self.rates {
                        if *v == vt
                            && chrom.is_none_or(|chrom| chrom == *c)
                            && stratum.is_none_or(|stratum| stratum == *s)
                        {
                            r.add(rates);
                        }
                    }
                    if r == ActionRates::default() && chrom.is_some() {
                        continue;
                    }
                    let mut fields = vec![
                        self.sample.clone(),
                        stratum.map_or_else(|| ALL.to_string(), |s| s.name(&self.names)),
                        vt.to_string(),
                        chrom.map_or_else(|| ALL.to_string(), |c| self.chroms[c].clone()),
                        self.mode.to_string(),
                    ];
                    fields.extend(rate_fields(&r));
                    lines.push(fields);
                }
            }
        }
        write_tsv(path, &RATE_COLUMNS, lines.into_iter())
    }

    /// One log line per outcome row.
    #[must_use]
    pub fn summary_lines(&self) -> Vec<String> {
        self.outcome_rows()
            .into_iter()
            .map(|(truth, vt, stratum, o)| {
                summary_line(&self.sample, &self.names[truth], &stratum, 12, vt, &o, false)
            })
            .collect()
    }
}

/// Decisions, outcomes and calibration within named, possibly overlapping regions (e.g. GIAB
/// stratifications), as hap.py's `--stratification` reports them.
///
/// For each truth set, a region's outcome counts the truth set's labelled calls and truth FNs
/// whose REF span overlaps the region: hap.py labels calls and reports FNs only inside the truth
/// set's own confident regions, so each truth set is scored within those. The `all` region is
/// every labelled call and FN (the `all` rows of [`StratifiedReport`]). Action rates count every
/// scored call by region and benchmark [`Stratum`], so the parts of a region outside the truth
/// sets have rates too.
#[derive(Debug, Clone)]
pub struct RegionReport {
    /// The sample name.
    pub sample: String,
    /// The names of the two truth sets.
    pub names: [String; 2],
    /// The decision mode of the genotype models.
    pub mode: Mode,
    /// The regions' names, in report order.
    pub regions: Vec<String>,
    /// Outcomes by truth set, variant type and region (index into `regions`).
    pub outcomes: BTreeMap<(usize, VariantType, usize), Outcome>,
    /// P(keep) calibration by truth set, variant type and region.
    pub calibration: BTreeMap<(usize, VariantType, usize), Calibration>,
    /// Action counts by variant type, region and benchmark stratum.
    pub rates: BTreeMap<(VariantType, usize, Stratum), ActionRates>,
    /// Outcomes over every region and none, by truth set and variant type (the `all` rows).
    pub totals: BTreeMap<(usize, VariantType), Outcome>,
    /// Calibration over every labelled call, by truth set and variant type.
    pub total_calibration: BTreeMap<(usize, VariantType), Calibration>,
}

impl RegionReport {
    /// An empty report over regions named `regions`.
    #[must_use]
    pub fn new(sample: &str, names: [String; 2], mode: Mode, regions: Vec<String>) -> Self {
        RegionReport {
            sample: sample.to_string(),
            names,
            mode,
            regions,
            outcomes: BTreeMap::new(),
            calibration: BTreeMap::new(),
            rates: BTreeMap::new(),
            totals: BTreeMap::new(),
            total_calibration: BTreeMap::new(),
        }
    }

    /// True if the report has no regions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The variant types with any outcome or action counts, in order.
    fn variant_types(&self) -> Vec<VariantType> {
        let mut types: Vec<VariantType> =
            self.totals.keys().map(|k| k.1).chain(self.rates.keys().map(|k| k.0)).collect();
        types.sort_unstable();
        types.dedup();
        types
    }

    /// The groups of a truth set and variant type: each region, then `all`.
    fn groups(&self) -> impl Iterator<Item = (Option<usize>, &str)> {
        self.regions
            .iter()
            .enumerate()
            .map(|(r, name)| (Some(r), name.as_str()))
            .chain(std::iter::once((None, ALL)))
    }

    /// The outcome rows: per truth set and variant type with labelled calls or FNs, each region
    /// (empty or not), then `all`. Each row is `(truth, variant type, region, outcome)`.
    #[must_use]
    pub fn outcome_rows(&self) -> Vec<(usize, VariantType, String, Outcome)> {
        let mut rows = Vec::new();
        for truth in 0..2 {
            for vt in self.variant_types() {
                let Some(total) = self.totals.get(&(truth, vt)) else { continue };
                for (r, name) in self.groups() {
                    let outcome = match r {
                        Some(r) => self.outcomes.get(&(truth, vt, r)).copied().unwrap_or_default(),
                        None => *total,
                    };
                    rows.push((truth, vt, name.to_string(), outcome));
                }
            }
        }
        rows
    }

    /// Writes the outcomes (see [`REGION_OUTCOME_COLUMNS`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_outcomes(&self, path: &Path) -> Result<()> {
        let lines = self.outcome_rows().into_iter().map(|(truth, vt, region, o)| {
            outcome_fields(&self.sample, &self.names[truth], region, vt, self.mode, &o)
        });
        write_tsv(path, &REGION_OUTCOME_COLUMNS, lines)
    }

    /// Writes the P(keep) calibration (see [`REGION_CALIBRATION_COLUMNS`]): per truth set,
    /// variant type and region with calls (and `all`), as [`StratifiedReport::write_calibration`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_calibration(&self, path: &Path) -> Result<()> {
        let mut lines = Vec::new();
        for truth in 0..2 {
            for vt in self.variant_types() {
                let Some(total) = self.total_calibration.get(&(truth, vt)) else { continue };
                for (r, name) in self.groups() {
                    let c = match r {
                        Some(r) => match self.calibration.get(&(truth, vt, r)) {
                            Some(c) => c,
                            None => continue,
                        },
                        None => total,
                    };
                    calibration_lines(&mut lines, [&self.sample, &self.names[truth], name], vt, c);
                }
            }
        }
        write_tsv(path, &REGION_CALIBRATION_COLUMNS, lines.into_iter())
    }

    /// Writes the action counts and rates of every scored call (see [`REGION_RATE_COLUMNS`]):
    /// per region, benchmark stratum (and `all`) and variant type.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_rates(&self, path: &Path) -> Result<()> {
        let mut lines = Vec::new();
        for (r, name) in self.regions.iter().enumerate() {
            for stratum in Stratum::ALL.into_iter().map(Some).chain(std::iter::once(None)) {
                for vt in self.variant_types() {
                    let mut rates = ActionRates::default();
                    for ((v, region, s), part) in &self.rates {
                        if *v == vt && *region == r && stratum.is_none_or(|stratum| stratum == *s) {
                            rates.add(part);
                        }
                    }
                    let mut fields = vec![
                        self.sample.clone(),
                        name.clone(),
                        stratum.map_or_else(|| ALL.to_string(), |s| s.name(&self.names)),
                        vt.to_string(),
                        self.mode.to_string(),
                    ];
                    fields.extend(rate_fields(&rates));
                    lines.push(fields);
                }
            }
        }
        write_tsv(path, &REGION_RATE_COLUMNS, lines.into_iter())
    }

    /// One log line per outcome row.
    #[must_use]
    pub fn summary_lines(&self) -> Vec<String> {
        self.outcome_rows()
            .into_iter()
            .map(|(truth, vt, region, o)| {
                summary_line(&self.sample, &self.names[truth], &region, 38, vt, &o, true)
            })
            .collect()
    }
}

/// A log line of an outcome row: the sample, truth set, group (stratum or region, padded to
/// `width`) and variant type, then the actions, their outcomes and the F1 change, with the truth
/// FNs when `with_fn`.
fn summary_line(
    sample: &str,
    truth: &str,
    group: &dyn std::fmt::Display,
    width: usize,
    vt: VariantType,
    o: &Outcome,
    with_fn: bool,
) -> String {
    let (before, after) = (o.before, o.after());
    let fns = if with_fn { format!("FN={} ", before.fn_) } else { String::new() };
    format!(
        "{sample} {truth:5} {group:width$} {vt} labelled={} drop={} (TP {}) flip={} (fixed {}, \
         broke {}) regenotype={} (fixed {}, broke {}) promote={} (correct {}) {fns}F1 {:.5} -> \
         {:.5} ({:+.5})",
        o.labelled,
        o.dropped,
        o.dropped_tp,
        o.flipped,
        o.flips_fixed,
        o.flips_broke_tp,
        o.regenotyped,
        o.regenotypes_fixed,
        o.regenotypes_broke_tp,
        o.promoted,
        o.promoted_correct,
        before.f1(),
        after.f1(),
        after.f1() - before.f1()
    )
}

/// An outcome row: the sample, truth set, group (stratum or region), variant type and mode, then
/// the counts and F1 of [`OUTCOME_COLUMNS`].
fn outcome_fields(
    sample: &str,
    truth: &str,
    group: String,
    vt: VariantType,
    mode: Mode,
    o: &Outcome,
) -> Vec<String> {
    let (before, after) = (o.before, o.after());
    let counts = [
        o.labelled,
        o.dropped,
        o.dropped_tp,
        o.flipped,
        o.flips_fixed,
        o.flips_broke_tp,
        o.regenotyped,
        o.regenotypes_fixed,
        o.regenotypes_broke_tp,
        o.candidates,
        o.promoted,
        o.promoted_correct,
        o.promoted_flipped,
        before.tp,
        before.fp,
        before.fn_,
    ];
    let mut fields =
        vec![sample.to_string(), truth.to_string(), group, vt.to_string(), mode.to_string()];
    fields.extend(counts.iter().map(ToString::to_string));
    fields.push(metric_float(before.f1()));
    fields.extend([after.tp, after.fp, after.fn_].iter().map(ToString::to_string));
    fields.push(metric_float(after.f1()));
    fields.push(metric_float(after.f1() - before.f1()));
    fields
}

/// Appends the calibration rows of one group (see [`CALIBRATION_COLUMNS`]): one per non-empty
/// bin and a `[0, 1]` summary row whose last column is the expected calibration error. `keys`
/// are the sample, truth set and group (stratum or region).
fn calibration_lines(
    lines: &mut Vec<Vec<String>>,
    keys: [&str; 3],
    vt: VariantType,
    c: &Calibration,
) {
    let row = |lo: f64, hi: f64, n: i64, sum_p: f64, keep: i64, ece: f64| {
        let mut fields: Vec<String> = keys.iter().map(ToString::to_string).collect();
        fields.extend([
            vt.to_string(),
            metric_float(lo),
            metric_float(hi),
            n.to_string(),
            metric_float(if n == 0 { 0.0 } else { sum_p / n as f64 }),
            metric_float(ratio(keep, n)),
            metric_float(ece),
        ]);
        fields
    };
    let width = 1.0 / CALIBRATION_BINS as f64;
    for bin in (0..CALIBRATION_BINS).filter(|&bin| c.calls[bin] > 0) {
        let lo = bin as f64 * width;
        let gap = (c.sum_p[bin] / c.calls[bin] as f64 - ratio(c.keep[bin], c.calls[bin])).abs();
        lines.push(row(lo, lo + width, c.calls[bin], c.sum_p[bin], c.keep[bin], gap));
    }
    let n: i64 = c.calls.iter().sum();
    let keep: i64 = c.keep.iter().sum();
    lines.push(row(0.0, 1.0, n, c.sum_p.iter().sum(), keep, c.expected_calibration_error()));
}

/// The counts and rates of [`RATE_COLUMNS`] from `pass_calls` on.
fn rate_fields(r: &ActionRates) -> Vec<String> {
    let mut fields: Vec<String> = [r.pass_calls, r.dropped, r.flipped, r.regenotyped]
        .iter()
        .map(ToString::to_string)
        .collect();
    fields.extend(
        [r.dropped, r.flipped, r.regenotyped].iter().map(|&n| metric_float(ratio(n, r.pass_calls))),
    );
    fields.extend([r.eligible, r.promoted, r.promoted_flipped].iter().map(ToString::to_string));
    fields.push(metric_float(ratio(r.promoted, r.eligible)));
    fields.push(metric_float(ratio(r.promoted, r.pass_calls)));
    fields
}

/// Checks that region names are unique and none is `all` (the name of the summary row).
///
/// # Errors
///
/// Returns an error naming the reserved name, or the first repeated name (in sorted order).
pub fn check_region_names<S: AsRef<str>>(names: &[S]) -> Result<()> {
    if names.iter().any(|name| name.as_ref() == ALL) {
        bail!("a region cannot be named {ALL:?}: it names the summary over all calls");
    }
    if let Some(name) = crate::common::repeated(names.iter().map(AsRef::as_ref)).first() {
        bail!("the region name {name:?} is given more than once");
    }
    Ok(())
}

/// The inputs of [`stratify`] for the two truth sets.
#[derive(Debug, Clone, Copy)]
pub struct TruthSets<'a> {
    /// The truth sets' names (e.g. `v421` and `v5q`).
    pub names: &'a [String; 2],
    /// Their confident regions.
    pub beds: [&'a RegionSet; 2],
    /// Their hap.py output VCFs for the sample, for the truth FNs.
    pub happy: [&'a Path; 2],
}

/// One model's decisions for the table rows it scores, with each row's index into them.
struct RowIndex {
    decisions: ModelRowDecisions,
    /// Per table row, its index in `decisions` (`u32::MAX` if not scored).
    index: Vec<u32>,
}

impl RowIndex {
    fn new(decisions: ModelRowDecisions, rows: usize) -> Result<Self> {
        let mut index = vec![u32::MAX; rows];
        for (i, &row) in decisions.rows.iter().enumerate() {
            index[row] = u32::try_from(i)?;
        }
        Ok(RowIndex { decisions, index })
    }

    /// The index of a table row in the decisions, if the model scored it.
    fn get(&self, row: usize) -> Option<usize> {
        let i = self.index[row];
        (i != u32::MAX).then_some(i as usize)
    }

    /// The genotype action on a scored row.
    fn action(&self, i: usize) -> CallAction {
        let d = &self.decisions;
        CallAction::of(d.drop[i], d.flip[i], d.regenotype[i])
    }
}

/// Optional per-call outputs of [`stratify`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CallOutputs<'a> {
    /// A gzipped TSV to write each scored call's stratum, labels, probabilities and action to.
    pub calls: Option<&'a Path>,
    /// A directory to write, per truth set and variant type, the labelled PASS calls as the
    /// genotype model sees them (features including the length-call features, re-genotype
    /// columns, labels, and the call's `row` in the feature table) to
    /// `<sample>.<truth>.<variant type>.parquet`.
    pub labelled_dir: Option<&'a Path>,
}

/// Scores a sample's calls with the models as `score` does, and reports the decisions, outcomes
/// and calibration by stratum (see the module docs).
///
/// # Arguments
///
/// * `table` - the sample's feature table labelled against the first truth set, with the columns
///   the models need.
/// * `second` - the same calls labelled against the second truth set (only the key and label
///   columns are needed).
/// * `models` - at most one genotype and one promote model per variant type.
/// * `policy` - how the probabilities become actions.
/// * `truths` - the truth sets' names, regions and hap.py VCFs.
/// * `regions` - named regions (e.g. GIAB stratifications) to also report within (see
///   [`RegionReport`]); may be empty.
/// * `outputs` - optional per-call outputs (see [`CallOutputs`]).
///
/// # Errors
///
/// Returns an error if the tables are not of the same calls, lack columns the models need, or a
/// truth set's FN count disagrees with its table's.
pub fn stratify(
    table: &FeatureTable,
    second: &FeatureTable,
    models: &[WinnowrModel],
    policy: &Policy,
    truths: TruthSets<'_>,
    regions: &[RegionSet],
    outputs: CallOutputs<'_>,
) -> Result<(StratifiedReport, RegionReport)> {
    check_model_kinds(models)?;
    let mut timer = PhaseTimer::start();
    let views = [table.clone(), with_truth_labels(table, second)?];
    let strata = table_strata(table, truths.beds)?;
    let chroms = table.str_column("chrom")?;
    let membership = RegionMembership::of_table(table, regions)?;
    timer.lap("assigning strata");
    let mut report = StratifiedReport::new(&table.info().sample, truths.names.clone(), policy.mode);
    let region_names = regions.iter().map(|r| r.name().to_string()).collect();
    let mut by_region =
        RegionReport::new(&table.info().sample, truths.names.clone(), policy.mode, region_names);
    report.chroms.clone_from(&table.info().contigs);
    let mut genotype: BTreeMap<VariantType, (&WinnowrModel, RowIndex)> = BTreeMap::new();
    let mut promote: BTreeMap<VariantType, RowIndex> = BTreeMap::new();
    for model in models {
        let vt = model.info.variant_type;
        let decisions = decide_model_rows(table, model, policy, &mut timer)?;
        let index = RowIndex::new(decisions, table.num_rows())?;
        let d = &index.decisions;
        for (i, &row) in d.rows.iter().enumerate() {
            let add = |rates: &mut ActionRates| match model.info.task {
                Task::Genotype => rates.add_call(index.action(i)),
                Task::Promote => rates.add_candidate(d.promote[i], d.flip[i]),
            };
            add(report.rates(vt, chroms[row], strata[row]));
            for r in membership.of(row) {
                add(by_region.rates.entry((vt, r, strata[row])).or_default());
            }
        }
        match model.info.task {
            Task::Genotype => {
                genotype.insert(vt, (model, index));
            }
            Task::Promote => {
                promote.insert(vt, index);
            }
        }
    }
    let variant_types: Vec<VariantType> = {
        let mut types: Vec<VariantType> = models.iter().map(|m| m.info.variant_type).collect();
        types.sort_unstable();
        types.dedup();
        types
    };
    for (truth, view) in views.iter().enumerate() {
        let (fns, region_fns) =
            fn_counts(truths.happy[truth], &table.info().contigs, truths.beds, regions)?;
        for &vt in &variant_types {
            let expected = i64::try_from(view.info().fn_count(vt)?)?;
            let found: i64 = fns.iter().filter(|((v, _), _)| *v == vt).map(|(_, n)| n).sum();
            if found != expected {
                bail!(
                    "{} has {found} {vt} FNs on the table's contigs, but the {} table records \
                     {expected}: is it the hap.py VCF the table was labelled from?",
                    truths.happy[truth].display(),
                    truths.names[truth]
                );
            }
            for stratum in Stratum::ALL {
                let n = fns.get(&(vt, stratum)).copied().unwrap_or(0);
                if n > 0 || stratum.has_truth(truth) {
                    report.outcome(truth, vt, stratum).before.fn_ += n;
                }
            }
            by_region.totals.entry((truth, vt)).or_default().before.fn_ += found;
            for r in 0..regions.len() {
                let n = region_fns.get(&(vt, r)).copied().unwrap_or(0);
                by_region.outcomes.entry((truth, vt, r)).or_default().before.fn_ += n;
            }
            let model = genotype.get(&vt);
            let stutter = model.and_then(|(m, _)| m.info.stutter.as_ref());
            let regenotype = model.is_some_and(|(m, _)| m.info.regenotype);
            let calls = SampleCalls::from_table(view, vt, stutter, regenotype)?;
            let mask = labelled_mask(view, vt)?;
            let rows: Vec<usize> = (0..mask.len()).filter(|&row| mask[row]).collect();
            if let Some(dir) = outputs.labelled_dir {
                let path = dir.join(format!(
                    "{}.{}.{vt}.parquet",
                    table.info().sample,
                    truths.names[truth]
                ));
                let row_values: Vec<f64> = rows.iter().map(|&row| row as f64).collect();
                calls.calls.clone().with_f64_column("row", row_values)?.write_parquet(&path)?;
            }
            let flip_ok = calls.calls.bool_column("flip_ok")?;
            let regenotype_ok = calls.calls.bool_column("regenotype_ok")?;
            for (k, &row) in rows.iter().enumerate() {
                let (action, p_keep) = match model {
                    Some((_, index)) => {
                        let i = index.get(row).context("a labelled PASS call was not scored")?;
                        let p = index.decisions.probabilities.get(i, Label::Keep.index());
                        (index.action(i), Some(p))
                    }
                    None => (CallAction::Keep, None),
                };
                let label = calls.labels[k];
                let add = |outcome: &mut Outcome| {
                    outcome.add_call(label, flip_ok[k], regenotype_ok[k], action);
                };
                add(report.outcome(truth, vt, strata[row]));
                add(by_region.totals.entry((truth, vt)).or_default());
                for r in membership.of(row) {
                    add(by_region.outcomes.entry((truth, vt, r)).or_default());
                }
                if let Some(p) = p_keep {
                    let keep = label == Label::Keep;
                    report.calibration(truth, vt, strata[row]).add_call(p, keep);
                    by_region.total_calibration.entry((truth, vt)).or_default().add_call(p, keep);
                    for r in membership.of(row) {
                        by_region.calibration.entry((truth, vt, r)).or_default().add_call(p, keep);
                    }
                }
            }
            if let Some(index) = promote.get(&vt) {
                if !view.info().promote_labelled {
                    bail!(
                        "the {} feature table has no promote labels (extract with --truth-vcf)",
                        truths.names[truth]
                    );
                }
                let mask = promote_mask(view, vt)?;
                let labels = view.f64_column("promote_label")?;
                for row in (0..mask.len()).filter(|&row| mask[row]) {
                    let label = PromoteLabel::from_value(labels[row])
                        .with_context(|| format!("promote_label {} is not a class", labels[row]))?;
                    let i = index.get(row).context("a promotion candidate was not scored")?;
                    let d = &index.decisions;
                    let add = |outcome: &mut Outcome| {
                        outcome.add_candidate(label, d.promote[i], d.flip[i])
                    };
                    add(report.outcome(truth, vt, strata[row]));
                    add(by_region.totals.entry((truth, vt)).or_default());
                    for r in membership.of(row) {
                        add(by_region.outcomes.entry((truth, vt, r)).or_default());
                    }
                }
            }
        }
    }
    timer.lap("scoring outcomes");
    if let Some(path) = outputs.calls {
        let indices: Vec<&RowIndex> =
            genotype.values().map(|(_, index)| index).chain(promote.values()).collect();
        write_calls(path, &views, &strata, &indices, truths.names)?;
        timer.lap("writing the calls");
    }
    Ok((report, by_region))
}

/// A label for a TSV: its integer value, or `.` if missing.
fn label_text(value: f64) -> String {
    if value.is_nan() { ".".to_string() } else { format!("{}", value as i64) }
}

/// Writes each scored call's row, key, stratum, labels from both truth sets, probabilities and
/// action to a gzipped TSV.
fn write_calls(
    path: &Path,
    views: &[FeatureTable; 2],
    strata: &[Stratum],
    indices: &[&RowIndex],
    names: &[String; 2],
) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut out =
        std::io::BufWriter::new(flate2::write::GzEncoder::new(file, flate2::Compression::fast()));
    let table = &views[0];
    let (chroms, positions) = (table.str_column("chrom")?, table.i64_column("pos")?);
    let (refs, alts) = (table.str_column("ref")?, table.str_column("alt")?);
    let (filters, types, gts) =
        (table.str_column("filter")?, table.str_column("variant_type")?, table.str_column("gt")?);
    let labels = [views[0].f64_column("label")?, views[1].f64_column("label")?];
    let nan = vec![f64::NAN; table.num_rows()];
    let promote_label = |view: &FeatureTable| -> Result<Vec<f64>> {
        if view.has_column("promote_label") {
            Ok(view.f64_column("promote_label")?.into_owned())
        } else {
            Ok(nan.clone())
        }
    };
    let promote_labels = [promote_label(&views[0])?, promote_label(&views[1])?];
    writeln!(
        out,
        "row\tchrom\tpos\tref\talt\tfilter\tvariant_type\tgt\tstratum\tlabel_{a}\tlabel_{b}\t\
         promote_label_{a}\tpromote_label_{b}\tp0\tp1\tp2\tp3\taction",
        a = names[0],
        b = names[1]
    )?;
    let mut rows: Vec<(usize, &RowIndex, usize)> = indices
        .iter()
        .flat_map(|index| {
            index.decisions.rows.iter().enumerate().map(move |(i, &row)| (row, *index, i))
        })
        .collect();
    rows.sort_unstable_by_key(|&(row, _, _)| row);
    for (row, index, i) in rows {
        let d = &index.decisions;
        let p: Vec<String> = (0..4)
            .map(|class| {
                if class < d.probabilities.classes() {
                    format!("{:.4}", d.probabilities.get(i, class))
                } else {
                    ".".to_string()
                }
            })
            .collect();
        let action = if d.promote[i] {
            if d.flip[i] { "promote_flip" } else { "promote" }
        } else if filters[row] != crate::common::PASS {
            "stay"
        } else {
            match index.action(i) {
                CallAction::Keep => "keep",
                CallAction::Drop => "drop",
                CallAction::Flip => "flip",
                CallAction::Regenotype => "regenotype",
            }
        };
        writeln!(
            out,
            "{row}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{action}",
            chroms[row],
            positions[row],
            refs[row],
            alts[row],
            filters[row],
            types[row],
            gts[row],
            strata[row].name(names),
            label_text(labels[0][row]),
            label_text(labels[1][row]),
            label_text(promote_labels[0][row]),
            label_text(promote_labels[1][row]),
            p.join("\t"),
        )?;
    }
    out.into_inner()
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", path.display()))?
        .finish()
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// The columns of the outcomes file. `labelled` to `promoted_flipped` count calls; `*_before`
/// are the counts and F1 with no action, `tp`/`fp`/`fn`/`f1` after all actions.
pub const OUTCOME_COLUMNS: [&str; 27] = [
    "sample",
    "truth",
    "stratum",
    "variant_type",
    "mode",
    "labelled",
    "dropped",
    "dropped_tp",
    "flipped",
    "flips_fixed",
    "flips_broke_tp",
    "regenotyped",
    "regenotypes_fixed",
    "regenotypes_broke_tp",
    "candidates",
    "promoted",
    "promoted_correct",
    "promoted_flipped",
    "tp_before",
    "fp_before",
    "fn_before",
    "f1_before",
    "tp",
    "fp",
    "fn",
    "f1",
    "f1_gain",
];

/// The columns of the calibration file. `gap` is |mean_p_keep - observed_keep| for a bin, and
/// the expected calibration error for the `[0, 1]` summary row.
pub const CALIBRATION_COLUMNS: [&str; 10] = [
    "sample",
    "truth",
    "stratum",
    "variant_type",
    "bin_lo",
    "bin_hi",
    "calls",
    "mean_p_keep",
    "observed_keep",
    "gap",
];

/// The columns of the action-rates file. Rates are fractions of `pass_calls`, except
/// `promote_rate` (of `eligible`) and `promoted_per_pass` (promotions per PASS call).
pub const RATE_COLUMNS: [&str; 17] = [
    "sample",
    "stratum",
    "variant_type",
    "chrom",
    "mode",
    "pass_calls",
    "dropped",
    "flipped",
    "regenotyped",
    "drop_rate",
    "flip_rate",
    "regenotype_rate",
    "eligible",
    "promoted",
    "promoted_flipped",
    "promote_rate",
    "promoted_per_pass",
];

/// The columns of the region outcomes file: [`OUTCOME_COLUMNS`] with `region` for `stratum`.
pub const REGION_OUTCOME_COLUMNS: [&str; 27] = {
    let mut columns = OUTCOME_COLUMNS;
    columns[2] = "region";
    columns
};

/// The columns of the region calibration file: [`CALIBRATION_COLUMNS`] with `region` for
/// `stratum`.
pub const REGION_CALIBRATION_COLUMNS: [&str; 10] = {
    let mut columns = CALIBRATION_COLUMNS;
    columns[2] = "region";
    columns
};

/// The columns of the region action-rates file: [`RATE_COLUMNS`] with `region` and the benchmark
/// `stratum` in place of `stratum` and `chrom`.
pub const REGION_RATE_COLUMNS: [&str; 17] = [
    "sample",
    "region",
    "stratum",
    "variant_type",
    "mode",
    "pass_calls",
    "dropped",
    "flipped",
    "regenotyped",
    "drop_rate",
    "flip_rate",
    "regenotype_rate",
    "eligible",
    "promoted",
    "promoted_flipped",
    "promote_rate",
    "promoted_per_pass",
];

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;
    use tempfile::TempDir;

    use super::*;
    use crate::testing::{call_table, vcf_header, write_indexed_vcf};

    /// The truth FN records of a hap.py output VCF on `contigs`, counted by variant type and stratum.
    ///
    /// An FN's stratum is decided by its REF span as a call's is (see [`table_strata`]): it is inside
    /// a truth set's regions only if its whole span is. hap.py reports the FNs of its own truth set
    /// that overlap that truth set's regions, so an FN whose span crosses their edge falls in a
    /// stratum without that truth set, as a labelled call with the same span does; it still counts in
    /// the truth set's `all` row.
    ///
    /// # Errors
    ///
    /// Returns an error if the VCF cannot be read or is not hap.py output.
    fn fn_strata<S: AsRef<str>>(
        happy: &Path,
        contigs: &[S],
        beds: [&RegionSet; 2],
    ) -> Result<StratumFnCounts> {
        Ok(fn_counts(happy, contigs, beds, &[])?.0)
    }

    fn names() -> [String; 2] {
        ["v421".to_string(), "v5q".to_string()]
    }

    #[rstest]
    #[case(true, true, Stratum::Both, "both", [true, true])]
    #[case(false, true, Stratum::SecondOnly, "v5q-only", [false, true])]
    #[case(true, false, Stratum::FirstOnly, "v421-only", [true, false])]
    #[case(false, false, Stratum::Neither, "neither", [false, false])]
    fn strata_are_named_and_covered(
        #[case] first: bool,
        #[case] second: bool,
        #[case] expected: Stratum,
        #[case] name: &str,
        #[case] covered: [bool; 2],
    ) {
        let stratum = Stratum::of(first, second);
        assert_eq!(stratum, expected);
        assert_eq!(stratum.name(&names()), name);
        assert_eq!([stratum.has_truth(0), stratum.has_truth(1)], covered);
    }

    #[test]
    fn summary_lines_pad_the_group_and_add_the_fns_on_request() {
        let outcome = Outcome {
            labelled: 4,
            dropped: 1,
            before: Counts { tp: 3, fp: 1, fn_: 2 },
            delta: Counts { tp: 0, fp: -1, fn_: 0 },
            ..Outcome::default()
        };
        assert_eq!(
            summary_line("S", "v5", &"both", 6, VariantType::Snp, &outcome, false),
            "S v5    both   snp labelled=4 drop=1 (TP 0) flip=0 (fixed 0, broke 0) regenotype=0 \
             (fixed 0, broke 0) promote=0 (correct 0) F1 0.66667 -> 0.75000 (+0.08333)"
        );
        assert_eq!(
            summary_line("S", "v5", &"MHC", 5, VariantType::Indel, &outcome, true),
            "S v5    MHC   indel labelled=4 drop=1 (TP 0) flip=0 (fixed 0, broke 0) regenotype=0 \
             (fixed 0, broke 0) promote=0 (correct 0) FN=2 F1 0.66667 -> 0.75000 (+0.08333)"
        );
    }
    #[test]
    fn table_strata_use_the_ref_span() {
        // v421 covers chr1 [0, 100); v5q covers chr1 [50, 200).
        let first = RegionSet::from_spans("v421", [("chr1", 0, 100)]);
        let second = RegionSet::from_spans("v5q", [("chr1", 50, 200)]);
        let table = call_table(&[
            ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 60, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 99, "AT", "A", "PASS", "indel", "0/1", false), // spans [98, 100)
            ("chr1", 100, "AT", "A", "PASS", "indel", "0/1", false), // spans [99, 101)
            ("chr1", 150, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 250, "A", "C", "PASS", "snp", "0/1", false),
            ("chr2", 10, "A", "C", "PASS", "snp", "0/1", false),
        ]);
        let got = table_strata(&table, [&first, &second]).unwrap();
        use Stratum::*;
        assert_eq!(got, [FirstOnly, Both, Both, SecondOnly, SecondOnly, Neither, Neither]);
    }

    /// Every label, flip_ok, regenotype_ok and action: the outcomes summed over calls are what
    /// `evaluate`'s rescoring gives the same calls and actions.
    #[test]
    fn outcomes_sum_to_rescore() {
        use crate::rescore::{Actions, Promotions, baseline_counts, rescore, rescore_promote};
        let labels = [Label::Drop, Label::Keep, Label::Flip, Label::Regenotype];
        let actions =
            [CallAction::Keep, CallAction::Drop, CallAction::Flip, CallAction::Regenotype];
        let mut outcome = Outcome::default();
        let (mut all_labels, mut flip_oks, mut regenotype_oks) = (vec![], vec![], vec![]);
        let mut masks = Actions { drop: vec![], flip: vec![], regenotype: vec![] };
        for label in labels {
            for flip_ok in [false, true] {
                for regenotype_ok in [false, true] {
                    for action in actions {
                        outcome.add_call(label, flip_ok, regenotype_ok, action);
                        all_labels.push(label);
                        flip_oks.push(flip_ok);
                        regenotype_oks.push(regenotype_ok);
                        masks.drop.push(action == CallAction::Drop);
                        masks.flip.push(action == CallAction::Flip);
                        masks.regenotype.push(action == CallAction::Regenotype);
                    }
                }
            }
        }
        outcome.before.fn_ = 100;
        let baseline = baseline_counts(&all_labels, 100);
        assert_eq!(outcome.before, baseline);
        let rescored = rescore(&all_labels, &flip_oks, &masks, baseline, &regenotype_oks);
        assert_eq!(outcome.after(), rescored);
        // Promotions on top.
        let promote_labels = [PromoteLabel::Stay, PromoteLabel::Promote, PromoteLabel::PromoteFlip];
        let mut candidates = Vec::new();
        let mut promotions = Promotions { promote: vec![], flip: vec![] };
        for label in promote_labels {
            for (promote, flip) in [(false, false), (true, false), (true, true)] {
                outcome.add_candidate(label, promote, flip);
                candidates.push(label);
                promotions.promote.push(promote);
                promotions.flip.push(flip);
            }
        }
        assert_eq!(outcome.after(), rescore_promote(&candidates, &promotions, rescored));
    }

    mod end_to_end {
        use super::*;
        use crate::common::Task;
        use crate::testing::separable_model;

        /// The calls of HG002 labelled against one truth set: feature `x` is the class a
        /// separable model gives them (drop, keep, flip, keep).
        fn table(labels: [f64; 4], flip_ok: [bool; 4], fns: u64) -> FeatureTable {
            let mut table = call_table(&[
                ("chr1", 100, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 200, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 300, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 600, "A", "C", "PASS", "snp", "0/1", false),
            ]);
            table.info_mut().sample = "HG002".to_string();
            table.info_mut().fn_counts =
                Some([("snp".to_string(), fns), ("indel".to_string(), 0)].into_iter().collect());
            table
                .with_f64_column("x", vec![0.0, 1.0, 2.0, 1.0])
                .unwrap()
                .with_bool_column("flippable", vec![true; 4])
                .unwrap()
                .with_f64_column("label", labels.to_vec())
                .unwrap()
                .with_bool_column("flip_ok", flip_ok.to_vec())
                .unwrap()
                .with_f64_column("truth_delta_lo", vec![f64::NAN; 4])
                .unwrap()
                .with_f64_column("truth_delta_hi", vec![f64::NAN; 4])
                .unwrap()
        }

        /// A hap.py VCF with SNP FNs at `positions` on chr1.
        fn happy(dir: &Path, name: &str, positions: &[u64]) -> PathBuf {
            let fields = "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n\
                ##FORMAT=<ID=BD,Number=1,Type=String,Description=\"BD\">\n\
                ##FORMAT=<ID=BK,Number=1,Type=String,Description=\"BK\">\n\
                ##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"BVT\">\n";
            let header = vcf_header(&[fields], &["TRUTH", "QUERY"]);
            let records: Vec<String> = positions
                .iter()
                .map(|pos| {
                    format!(
                        "chr1\t{pos}\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL"
                    )
                })
                .collect();
            let records: Vec<&str> = records.iter().map(String::as_str).collect();
            write_indexed_vcf(dir, name, &header, &records)
        }

        #[test]
        fn stratifies_decisions_outcomes_and_calls() {
            let dir = TempDir::new().unwrap();
            let nan = f64::NAN;
            // v421: TP, TP, fixable flip, and an unlabelled call outside its regions; one FN.
            let first = table([1.0, 1.0, 2.0, nan], [false, false, true, false], 1);
            // v5q: FP, TP, TP, TP; two FNs, one inside v421's regions too.
            let second = table([0.0, 1.0, 1.0, 1.0], [false; 4], 2);
            let happy =
                [happy(dir.path(), "a.vcf.gz", &[50]), happy(dir.path(), "b.vcf.gz", &[60, 700])];
            let beds = [
                RegionSet::from_spans("v421", [("chr1", 0, 500)]),
                RegionSet::from_spans("v5q", [("chr1", 0, 1000)]),
            ];
            let names = names();
            let truths = TruthSets {
                names: &names,
                beds: [&beds[0], &beds[1]],
                happy: [&happy[0], &happy[1]],
            };
            let regions = [RegionSet::from_spans("easy", [("chr1", 0, 250)])];
            let models = [separable_model(Task::Genotype, VariantType::Snp)];
            let policy = Policy::new(Mode::Genotyper);
            let calls = dir.path().join("calls.tsv.gz");
            let outputs = CallOutputs { calls: Some(&calls), labelled_dir: Some(dir.path()) };
            let (report, by_region) =
                stratify(&first, &second, &models, &policy, truths, &regions, outputs).unwrap();

            let snp = VariantType::Snp;
            let after =
                |truth: usize, stratum: Stratum| report.outcomes[&(truth, snp, stratum)].after();
            let before =
                |truth: usize, stratum: Stratum| report.outcomes[&(truth, snp, stratum)].before;
            let counts = |tp: i64, fp: i64, fn_: i64| Counts { tp, fp, fn_ };
            // v421, inside both: a dropped TP and a fixed flip.
            assert_eq!(before(0, Stratum::Both), counts(2, 1, 1));
            assert_eq!(after(0, Stratum::Both), counts(2, 0, 1));
            let outcome = report.outcomes[&(0, snp, Stratum::Both)];
            let actions = [outcome.labelled, outcome.dropped, outcome.dropped_tp, outcome.flipped];
            assert_eq!(actions, [3, 1, 1, 1]);
            assert_eq!((outcome.flips_fixed, outcome.flips_broke_tp), (1, 0));
            // v5q: a dropped FP and a broken TP inside both; a TP and an FN inside v5q only.
            assert_eq!(before(1, Stratum::Both), counts(2, 1, 1));
            assert_eq!(after(1, Stratum::Both), counts(1, 1, 2));
            assert_eq!(before(1, Stratum::SecondOnly), counts(1, 0, 1));
            assert_eq!(after(1, Stratum::SecondOnly), counts(1, 0, 1));
            assert!(!report.outcomes.contains_key(&(0, snp, Stratum::SecondOnly)));
            // Actions on every scored call, by stratum, with no truth needed.
            let rates = &report.rates[&(snp, 0, Stratum::Both)];
            assert_eq!((rates.pass_calls, rates.dropped, rates.flipped), (3, 1, 1));
            assert_eq!(report.rates[&(snp, 0, Stratum::SecondOnly)].pass_calls, 1);
            // Within `easy` ([0, 250)): the dropped call and the kept one, and each truth set's
            // FN there.
            let easy = by_region.outcomes[&(0, snp, 0)];
            assert_eq!((easy.labelled, easy.dropped, easy.before.fn_), (2, 1, 1));
            assert_eq!(by_region.outcomes[&(1, snp, 0)].before.fn_, 1);
            assert_eq!(by_region.totals[&(1, snp)].after(), counts(2, 1, 3));

            // The calls file: one row per scored call, with both truth sets' labels.
            let mut text = String::new();
            std::io::Read::read_to_string(
                &mut flate2::read::GzDecoder::new(std::fs::File::open(&calls).unwrap()),
                &mut text,
            )
            .unwrap();
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(lines.len(), 5);
            assert!(lines[0].starts_with("row\tchrom\tpos\tref\talt\tfilter\tvariant_type\tgt\tstratum\tlabel_v421\tlabel_v5q\t"));
            let fields = |line: &str| -> Vec<String> {
                let f: Vec<&str> = line.split('\t').collect();
                [&f[..11], &f[f.len() - 1..]].concat().iter().map(ToString::to_string).collect()
            };
            assert_eq!(
                fields(lines[1]),
                ["0", "chr1", "100", "A", "C", "PASS", "snp", "0/1", "both", "1", "0", "drop"]
            );
            assert_eq!(
                fields(lines[3]),
                ["2", "chr1", "300", "A", "C", "PASS", "snp", "0/1", "both", "2", "1", "flip"]
            );
            assert_eq!(
                fields(lines[4]),
                ["3", "chr1", "600", "A", "C", "PASS", "snp", "0/1", "v5q-only", ".", "1", "keep"]
            );
            // The labelled calls as the model saw them, with their table rows.
            let labelled = |truth: &str| {
                let path = dir.path().join(format!("HG002.{truth}.snp.parquet"));
                FeatureTable::read_parquet(path).unwrap().f64_column("row").unwrap().into_owned()
            };
            assert_eq!(labelled("v421"), [0.0, 1.0, 2.0]);
            assert_eq!(labelled("v5q"), [0.0, 1.0, 2.0, 3.0]);

            // A hap.py VCF whose FNs disagree with its table's count is refused.
            let swapped = TruthSets { happy: [&happy[1], &happy[0]], ..truths };
            let message =
                stratify(&first, &second, &models, &policy, swapped, &[], CallOutputs::default())
                    .unwrap_err()
                    .to_string();
            assert!(
                message.ends_with("has 2 snp FNs on the table's contigs, but the v421 table records 1: is it the hap.py VCF the table was labelled from?"),
                "{message}"
            );
        }
    }

    #[test]
    fn outcomes_follow_the_vcfeval_arithmetic() {
        let mut o = Outcome { before: Counts { tp: 0, fp: 0, fn_: 10 }, ..Outcome::default() };
        o.add_call(Label::Keep, false, false, CallAction::Drop); // TP-1 FN+1
        o.add_call(Label::Drop, false, false, CallAction::Drop); // FP-1
        o.add_call(Label::Keep, false, false, CallAction::Flip); // TP-1 FP+1 FN+1
        o.add_call(Label::Flip, true, false, CallAction::Flip); // TP+1 FP-1 FN-1
        o.add_call(Label::Flip, false, false, CallAction::Flip); // none
        o.add_call(Label::Regenotype, false, true, CallAction::Regenotype); // fixed
        o.add_call(Label::Keep, false, false, CallAction::Keep);
        assert_eq!(o.before, Counts { tp: 3, fp: 4, fn_: 10 });
        // As in rescore's test: the same calls give TP 3, FP 2, FN 10.
        assert_eq!(o.after(), Counts { tp: 3, fp: 2, fn_: 10 });
        assert_eq!((o.labelled, o.dropped, o.dropped_tp), (7, 2, 1));
        assert_eq!((o.flipped, o.flips_fixed, o.flips_broke_tp), (3, 1, 1));
        assert_eq!((o.regenotyped, o.regenotypes_fixed, o.regenotypes_broke_tp), (1, 1, 0));
        o.add_candidate(PromoteLabel::Promote, true, false); // FN -> TP
        o.add_candidate(PromoteLabel::PromoteFlip, true, true); // FN -> TP
        o.add_candidate(PromoteLabel::PromoteFlip, true, false); // new FP
        o.add_candidate(PromoteLabel::Stay, false, false);
        assert_eq!(
            (o.candidates, o.promoted, o.promoted_correct, o.promoted_flipped),
            (4, 3, 2, 1)
        );
        assert_eq!(o.after(), Counts { tp: 5, fp: 3, fn_: 8 });
        let mut sum = Outcome::default();
        sum.add(&o);
        sum.add(&o);
        assert_eq!(sum.after(), Counts { tp: 10, fp: 6, fn_: 16 });
    }

    #[test]
    fn calibration_bins_and_error() {
        let mut c = Calibration::default();
        c.add_call(0.05, false);
        c.add_call(0.95, true);
        c.add_call(1.0, false);
        c.add_call(f64::NAN, false);
        assert_eq!(c.calls[0], 2);
        assert_eq!(c.calls[9], 2);
        assert_eq!(c.keep[9], 1);
        // Bin 9: |1.95 - 1| = 0.95; bin 0 has a NaN sum, so a NaN error.
        assert!(c.expected_calibration_error().is_nan());
        let mut clean = Calibration::default();
        clean.add_call(0.05, false);
        clean.add_call(0.95, true);
        clean.add_call(0.85, false);
        let expected = (0.05 + 0.05 + 0.85) / 3.0;
        assert!((clean.expected_calibration_error() - expected).abs() < 1e-12);
        assert_eq!(Calibration::default().expected_calibration_error(), 0.0);
    }

    #[test]
    fn rates_count_actions() {
        let mut r = ActionRates::default();
        for action in [CallAction::Keep, CallAction::Drop, CallAction::Flip, CallAction::Regenotype]
        {
            r.add_call(action);
        }
        r.add_candidate(true, true);
        r.add_candidate(true, false);
        r.add_candidate(false, false);
        let expected = ActionRates {
            pass_calls: 4,
            dropped: 1,
            flipped: 1,
            regenotyped: 1,
            eligible: 3,
            promoted: 2,
            promoted_flipped: 1,
        };
        assert_eq!(r, expected);
        assert_eq!(CallAction::of(true, true, true), CallAction::Drop);
        assert_eq!(CallAction::of(false, true, true), CallAction::Flip);
    }

    fn labelled(label: f64) -> FeatureTable {
        let mut table = call_table(&[
            ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 20, "A", "C", "LowQual", "snp", "0/1", false),
        ]);
        table.info_mut().fn_counts =
            Some([("snp".to_string(), 3), ("indel".to_string(), 0)].into_iter().collect());
        table
            .with_f64_column("label", vec![label, f64::NAN])
            .unwrap()
            .with_bool_column("flip_ok", vec![false, false])
            .unwrap()
            .with_f64_column("truth_delta_lo", vec![f64::NAN; 2])
            .unwrap()
            .with_f64_column("truth_delta_hi", vec![f64::NAN; 2])
            .unwrap()
    }

    #[test]
    fn truth_labels_replace_the_label_columns() {
        let first = labelled(1.0);
        let mut second = labelled(0.0);
        second.info_mut().fn_counts =
            Some([("snp".to_string(), 7), ("indel".to_string(), 1)].into_iter().collect());
        let got = with_truth_labels(&first, &second).unwrap();
        assert_eq!(got.f64_column("label").unwrap()[0], 0.0);
        assert_eq!(got.info().fn_count(VariantType::Snp).unwrap(), 7);
        let error =
            |labels: &FeatureTable| with_truth_labels(&first, labels).unwrap_err().to_string();
        let other = call_table(&[("chr1", 11, "A", "C", "PASS", "snp", "0/1", false)]);
        assert_eq!(
            error(&other),
            "the feature tables have 2 and 1 calls: they must be extracted from the same VCF"
        );
        // As many calls, but another POS.
        let mut shifted = call_table(&[
            ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 21, "A", "C", "LowQual", "snp", "0/1", false),
        ]);
        shifted.info_mut().fn_counts.clone_from(&second.info().fn_counts);
        let shifted = shifted
            .with_f64_column("label", vec![0.0, f64::NAN])
            .unwrap()
            .with_bool_column("flip_ok", vec![false; 2])
            .unwrap();
        assert_eq!(
            error(&shifted),
            "the feature tables differ at call 1: they must be extracted from the same VCF"
        );
        let mut unlabelled = labelled(0.0);
        unlabelled.info_mut().fn_counts = None;
        assert_eq!(error(&unlabelled), "the feature table for S1 has no hap.py labels");
        let mut moved = labelled(0.0);
        moved = moved.with_str_column("alt", &["G", "C"]).unwrap();
        let message = error(&moved);
        assert!(message.contains("differ in alt"), "{message}");
        // A hap.py label column the first table has must be in the second.
        let without_deltas = {
            let mut t = call_table(&[
                ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 20, "A", "C", "LowQual", "snp", "0/1", false),
            ]);
            t.info_mut().fn_counts.clone_from(&second.info().fn_counts);
            t.with_f64_column("label", vec![0.0, f64::NAN])
                .unwrap()
                .with_bool_column("flip_ok", vec![false; 2])
                .unwrap()
        };
        assert_eq!(error(&without_deltas), "the feature table for S1 lacks truth_delta_lo");
    }

    #[test]
    fn promote_labels_are_dropped_when_the_second_table_has_none() {
        let mut first =
            labelled(1.0).with_f64_column("promote_label", vec![f64::NAN, 1.0]).unwrap();
        first.info_mut().promote_labelled = true;
        let second = labelled(0.0);
        let got = with_truth_labels(&first, &second).unwrap();
        assert!(!got.info().promote_labelled);
        assert!(got.f64_column("promote_label").unwrap().iter().all(|v| v.is_nan()));
        // The reverse: the second's promote labels are taken.
        let mut promote =
            labelled(0.0).with_f64_column("promote_label", vec![f64::NAN, 2.0]).unwrap();
        promote.info_mut().promote_labelled = true;
        let got = with_truth_labels(&labelled(1.0), &promote).unwrap();
        assert!(got.info().promote_labelled);
        assert_eq!(got.f64_column("promote_label").unwrap()[1], 2.0);
    }

    #[test]
    fn fns_are_counted_by_stratum() {
        let dir = TempDir::new().unwrap();
        let fields = "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n\
            ##FORMAT=<ID=BD,Number=1,Type=String,Description=\"BD\">\n\
            ##FORMAT=<ID=BK,Number=1,Type=String,Description=\"BK\">\n\
            ##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"BVT\">\n";
        let header = vcf_header(&[fields], &["TRUTH", "QUERY"]);
        let records = [
            "chr1\t10\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
            "chr1\t60\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
            "chr1\t70\t.\tAT\tA\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:INDEL\t./.:.:.:NOCALL",
            "chr1\t80\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:TP:gm:SNP\t0/1:TP:gm:SNP",
            // [98, 101) crosses the first truth set's edge, so it is not inside it, as a call with
            // that span would not be.
            "chr1\t99\t.\tATT\tA\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:INDEL\t./.:.:.:NOCALL",
            "chr2\t10\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
        ];
        let happy = write_indexed_vcf(dir.path(), "happy.vcf.gz", &header, &records);
        let first = RegionSet::from_spans("v421", [("chr1", 0, 100)]);
        let second = RegionSet::from_spans("v5q", [("chr1", 50, 200)]);
        let got = fn_strata(&happy, &["chr1"], [&first, &second]).unwrap();
        let expected: BTreeMap<(VariantType, Stratum), i64> = [
            ((VariantType::Snp, Stratum::FirstOnly), 1),
            ((VariantType::Snp, Stratum::Both), 1),
            ((VariantType::Indel, Stratum::Both), 1),
            ((VariantType::Indel, Stratum::SecondOnly), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(got, expected);
        // The stratum of the same calls as table_strata gives them.
        let calls = call_table(&[
            ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 99, "ATT", "A", "PASS", "indel", "0/1", false),
        ]);
        let strata = table_strata(&calls, [&first, &second]).unwrap();
        assert_eq!(strata, [Stratum::FirstOnly, Stratum::SecondOnly]);
        // chr2 is outside both truth sets.
        let got = fn_strata(&happy, &["chr1", "chr2"], [&first, &second]).unwrap();
        assert_eq!(got[&(VariantType::Snp, Stratum::Neither)], 1);
        assert_eq!(got.values().sum::<i64>(), 5);
    }

    #[test]
    fn reports_write_rows_per_truth_and_stratum() {
        let dir = TempDir::new().unwrap();
        let mut report = StratifiedReport::new("HG002", names(), Mode::Genotyper);
        report.outcome(0, VariantType::Snp, Stratum::Both).add_call(
            Label::Keep,
            false,
            false,
            CallAction::Drop,
        );
        report.outcome(1, VariantType::Snp, Stratum::SecondOnly).add_call(
            Label::Drop,
            false,
            false,
            CallAction::Drop,
        );
        report.calibration(1, VariantType::Snp, Stratum::SecondOnly).add_call(0.15, false);
        report.rates(VariantType::Snp, "chr1", Stratum::Neither).add_call(CallAction::Drop);
        report.rates(VariantType::Snp, "chr1", Stratum::Neither).add_call(CallAction::Keep);
        report.rates(VariantType::Snp, "chr2", Stratum::Neither).add_candidate(true, false);
        let rows: Vec<(usize, String)> =
            report.outcome_rows().into_iter().map(|(t, _, s, _)| (t, s)).collect();
        let expected: Vec<(usize, String)> =
            [(0, "both"), (0, "v421-only"), (0, "all"), (1, "both"), (1, "v5q-only"), (1, "all")]
                .into_iter()
                .map(|(t, s)| (t, s.to_string()))
                .collect();
        assert_eq!(rows, expected);

        let prefix = dir.path().join("out");
        let path = |suffix: &str| PathBuf::from(format!("{}.{suffix}", prefix.display()));
        report.write_outcomes(&path("strata.tsv")).unwrap();
        report.write_rates(&path("rates.tsv")).unwrap();
        report.write_calibration(&path("calibration.tsv")).unwrap();
        let read = |suffix: &str| std::fs::read_to_string(path(suffix)).unwrap();
        let strata = read("strata.tsv");
        let lines: Vec<&str> = strata.lines().collect();
        assert_eq!(lines[0], OUTCOME_COLUMNS.join("\t"));
        assert_eq!(
            lines[1],
            "HG002\tv421\tboth\tsnp\tgenotyper\t1\t1\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t1.0\t0\t0\t1\t0.0\t-1.0"
        );
        let rates = read("rates.tsv");
        let neither_chr1 = rates.lines().find(|l| l.contains("\tneither\tsnp\tchr1\t")).unwrap();
        assert_eq!(
            neither_chr1,
            "HG002\tneither\tsnp\tchr1\tgenotyper\t2\t1\t0\t0\t0.5\t0.0\t0.0\t0\t0\t0\t0.0\t0.0"
        );
        let all = rates.lines().find(|l| l.starts_with("HG002\tall\tsnp\tall\t")).unwrap();
        assert!(all.ends_with("\t1\t1\t0\t1.0\t0.5"), "{all}");
        // Strata with no calls get no per-chromosome rows.
        assert!(!rates.contains("\tboth\tsnp\tchr1\t"));
        let calibration = read("calibration.tsv");
        assert!(calibration.contains("HG002\tv5q\tv5q-only\tsnp\t0.1\t0.2\t1\t0.15\t0.0\t0.15"));
        assert!(calibration.contains("HG002\tv5q\tall\tsnp\t0.0\t1.0\t1\t0.15\t0.0\t0.15"));
    }

    #[test]
    fn region_membership_uses_overlap_and_allows_overlapping_regions() {
        // Regions overlap one another; a call is in every region its REF span touches.
        let regions = [
            RegionSet::from_spans("easy", [("chr1", 0, 100)]),
            RegionSet::from_spans("hard", [("chr1", 99, 200)]),
            RegionSet::from_spans("mhc", [("chr6", 0, 10)]),
        ];
        let table = call_table(&[
            ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false), // [9, 10): easy
            ("chr1", 99, "AT", "A", "PASS", "indel", "0/1", false), // [98, 100): easy and hard
            ("chr1", 101, "A", "C", "PASS", "snp", "0/1", false), // [100, 101): hard
            ("chr1", 300, "A", "C", "PASS", "snp", "0/1", false), // none
            ("chr2", 10, "A", "C", "PASS", "snp", "0/1", false), // no such contig
        ]);
        let membership = RegionMembership::of_table(&table, &regions).unwrap();
        let got: Vec<Vec<usize>> = (0..5).map(|row| membership.of(row).collect()).collect();
        assert_eq!(got, [vec![0], vec![0, 1], vec![1], vec![], vec![]]);
    }

    #[test]
    fn region_membership_spans_several_words() {
        // 70 regions need two 64-bit words per call.
        let regions: Vec<RegionSet> = (0..70u64)
            .map(|r| RegionSet::from_spans(format!("r{r}"), [("chr1", r * 10, r * 10 + 10)]))
            .collect();
        let table = call_table(&[
            ("chr1", 1, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 655, "AT", "A", "PASS", "indel", "0/1", false), // [654, 656): r65
            ("chr1", 700, "AT", "A", "PASS", "indel", "0/1", false), // [699, 701): r69
        ]);
        let membership = RegionMembership::of_table(&table, &regions).unwrap();
        assert_eq!(membership.of(0).collect::<Vec<_>>(), [0]);
        assert_eq!(membership.of(1).collect::<Vec<_>>(), [65]);
        assert_eq!(membership.of(2).collect::<Vec<_>>(), [69]);
    }

    #[test]
    fn fns_are_counted_by_region() {
        let dir = TempDir::new().unwrap();
        let fields = "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n\
            ##FORMAT=<ID=BD,Number=1,Type=String,Description=\"BD\">\n\
            ##FORMAT=<ID=BK,Number=1,Type=String,Description=\"BK\">\n\
            ##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"BVT\">\n";
        let header = vcf_header(&[fields], &["TRUTH", "QUERY"]);
        let records = [
            "chr1\t10\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
            "chr1\t60\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
            "chr1\t80\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:TP:gm:SNP\t0/1:TP:gm:SNP",
            // [99, 101) overlaps both regions.
            "chr1\t100\t.\tAT\tA\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:INDEL\t./.:.:.:NOCALL",
        ];
        let happy = write_indexed_vcf(dir.path(), "happy.vcf.gz", &header, &records);
        let first = RegionSet::from_spans("v421", [("chr1", 0, 1000)]);
        let second = RegionSet::from_spans("v5q", [("chr1", 0, 1000)]);
        let regions = [
            RegionSet::from_spans("easy", [("chr1", 0, 50), ("chr1", 90, 100)]),
            RegionSet::from_spans("hard", [("chr1", 50, 90), ("chr1", 100, 200)]),
        ];
        let (strata, by_region) =
            fn_counts(&happy, &["chr1"], [&first, &second], &regions).unwrap();
        assert_eq!(strata[&(VariantType::Snp, Stratum::Both)], 2);
        let expected: BTreeMap<(VariantType, usize), i64> = [
            ((VariantType::Snp, 0), 1),
            ((VariantType::Snp, 1), 1),
            ((VariantType::Indel, 0), 1),
            ((VariantType::Indel, 1), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(by_region, expected);
        // With no regions, the stratum counts are fn_strata's.
        let plain = fn_strata(&happy, &["chr1"], [&first, &second]).unwrap();
        assert_eq!(plain, strata);
    }

    #[test]
    fn region_reports_write_rows_per_truth_and_region() {
        let dir = TempDir::new().unwrap();
        let regions = vec!["easy".to_string(), "MHC".to_string()];
        let mut report = RegionReport::new("HG002", names(), Mode::Genotyper, regions);
        // v421: one dropped TP in `easy` and one FN outside any region.
        let mut easy = Outcome::default();
        easy.add_call(Label::Keep, false, false, CallAction::Drop);
        let mut total = easy;
        total.before.fn_ += 1;
        report.outcomes.insert((0, VariantType::Snp, 0), easy);
        report.totals.insert((0, VariantType::Snp), total);
        report.totals.insert((1, VariantType::Snp), Outcome::default());
        let mut calibration = Calibration::default();
        calibration.add_call(0.15, false);
        report.calibration.insert((0, VariantType::Snp, 0), calibration);
        report.total_calibration.insert((0, VariantType::Snp), calibration);
        let mut rates = ActionRates::default();
        rates.add_call(CallAction::Drop);
        report.rates.insert((VariantType::Snp, 0, Stratum::Neither), rates);
        rates.add_call(CallAction::Keep);
        report.rates.insert((VariantType::Snp, 0, Stratum::Both), rates);

        let rows: Vec<(usize, String)> =
            report.outcome_rows().into_iter().map(|(t, _, r, _)| (t, r)).collect();
        let expected: Vec<(usize, String)> =
            [(0, "easy"), (0, "MHC"), (0, "all"), (1, "easy"), (1, "MHC"), (1, "all")]
                .into_iter()
                .map(|(t, r)| (t, r.to_string()))
                .collect();
        assert_eq!(rows, expected);

        let path = |suffix: &str| dir.path().join(format!("out.regions.{suffix}"));
        report.write_outcomes(&path("strata.tsv")).unwrap();
        report.write_rates(&path("rates.tsv")).unwrap();
        report.write_calibration(&path("calibration.tsv")).unwrap();
        let read = |suffix: &str| std::fs::read_to_string(path(suffix)).unwrap();
        let strata = read("strata.tsv");
        let lines: Vec<&str> = strata.lines().collect();
        assert_eq!(lines[0], REGION_OUTCOME_COLUMNS.join("\t"));
        assert!(lines[0].starts_with("sample\ttruth\tregion\tvariant_type\t"));
        assert_eq!(
            lines[1],
            "HG002\tv421\teasy\tsnp\tgenotyper\t1\t1\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t1.0\t0\t0\t1\t0.0\t-1.0"
        );
        // An empty region still gets a row; `all` includes the FN outside every region.
        assert!(lines[2].starts_with("HG002\tv421\tMHC\tsnp\tgenotyper\t0\t"), "{}", lines[2]);
        assert!(lines[3].starts_with("HG002\tv421\tall\tsnp\tgenotyper\t1\t"), "{}", lines[3]);
        assert!(lines[3].ends_with("\t1\t0\t1\t0.66667\t0\t0\t2\t0.0\t-0.66667"), "{}", lines[3]);

        let rates = read("rates.tsv");
        let lines: Vec<&str> = rates.lines().collect();
        assert_eq!(lines[0], REGION_RATE_COLUMNS.join("\t"));
        // Per region: the four benchmark strata and `all`, empty or not.
        assert_eq!(lines.len(), 1 + 2 * 5);
        assert!(rates.contains("HG002\teasy\tneither\tsnp\tgenotyper\t1\t1\t0\t0\t1.0\t"));
        assert!(rates.contains("HG002\teasy\tboth\tsnp\tgenotyper\t2\t1\t0\t0\t0.5\t"));
        assert!(rates.contains("HG002\teasy\tall\tsnp\tgenotyper\t3\t2\t0\t0\t0.66667\t"));
        assert!(rates.contains("HG002\tMHC\tall\tsnp\tgenotyper\t0\t0\t0\t0\t0.0\t"));

        let calibration = read("calibration.tsv");
        assert_eq!(calibration.lines().next().unwrap(), REGION_CALIBRATION_COLUMNS.join("\t"));
        assert!(calibration.contains("HG002\tv421\teasy\tsnp\t0.1\t0.2\t1\t0.15\t0.0\t0.15"));
        assert!(calibration.contains("HG002\tv421\tall\tsnp\t0.0\t1.0\t1\t0.15\t0.0\t0.15"));
        // No calls in MHC, so no calibration rows for it.
        assert!(!calibration.contains("\tMHC\t"));
    }

    #[test]
    fn region_names_are_unique_and_not_all() {
        assert!(check_region_names(&["MHC", "segdups"]).is_ok());
        let message = check_region_names(&["b", "MHC", "b", "MHC"]).unwrap_err().to_string();
        assert_eq!(message, "the region name \"MHC\" is given more than once");
        let message = check_region_names(&["MHC", "all"]).unwrap_err().to_string();
        assert!(message.starts_with("a region cannot be named \"all\""), "{message}");
    }
}
