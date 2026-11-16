//! Turn class probabilities into drop/flip/re-genotype/promote decisions, and rescore F1 with
//! vcfeval arithmetic.
//!
//! Under vcfeval semantics a genotype mismatch is a query FP *and* a truth FN. Starting from
//! baseline counts (TP, FP, FN), each action changes them as follows:
//!
//! | action                     | change           |
//! |----------------------------|------------------|
//! | drop a TP                  | TP-1, FN+1       |
//! | drop an FP                 | FP-1             |
//! | flip a TP                  | TP-1, FP+1, FN+1 |
//! | flip an FP, flip correct   | TP+1, FP-1, FN-1 |
//! | flip an FP, not correct    | none             |
//! | re-genotype a TP           | TP-1, FP+1, FN+1 |
//! | re-genotype an FP, correct | TP+1, FP-1, FN-1 |
//! | re-genotype an FP, not     | none             |
//!
//! A "correct" flip is an allele-match FP whose het <-> hom flip gives the truth genotype. A
//! correct re-genotype is an FP whose length-caller genotype is the truth; re-genotyping to `0/0`
//! is a drop.

use std::fmt;
use std::str::FromStr;

use anyhow::bail;
use serde::{Deserialize, Serialize};

use crate::labels::{Label, PromoteLabel};

/// How a genotype model's probabilities are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Drop calls with P(drop) >= t_drop; flip het <-> hom calls with P(flip) >= t_flip; for
    /// re-genotype models, re-genotype the remaining calls the length caller disagrees with when
    /// P(regenotype) >= t_regenotype.
    Genotyper,
    /// Drop calls with P(drop) + P(flip) (+ P(regenotype)) >= t_drop (the call as made is wrong);
    /// never flip or re-genotype.
    Veto,
}

impl Mode {
    /// Every mode.
    pub const ALL: [Mode; 2] = [Mode::Genotyper, Mode::Veto];

    /// The lowercase name, as written in model files and on the command line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Mode::Genotyper => "genotyper",
            Mode::Veto => "veto",
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Mode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "genotyper" => Ok(Mode::Genotyper),
            "veto" => Ok(Mode::Veto),
            _ => bail!("unknown mode '{s}' (expected one of: genotyper, veto)"),
        }
    }
}

/// Class probabilities of calls: one row per call, one column per class.
#[derive(Debug, Clone, PartialEq)]
pub struct Probabilities {
    values: Vec<f64>,
    classes: usize,
}

impl Probabilities {
    /// Probabilities from row-major `values` with `classes` columns.
    ///
    /// # Panics
    ///
    /// Panics if `classes` is zero or does not divide the number of values.
    #[must_use]
    pub fn new(values: Vec<f64>, classes: usize) -> Self {
        assert!(
            classes > 0 && values.len().is_multiple_of(classes),
            "values do not fill whole rows"
        );
        Probabilities { values, classes }
    }

    /// The number of rows (calls).
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len() / self.classes
    }

    /// True if there are no rows.
    #[must_use]
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The number of classes.
    #[must_use]
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// The probability of class `class` for row `row`.
    #[must_use]
    pub fn get(&self, row: usize, class: usize) -> f64 {
        self.values[row * self.classes + class]
    }

    /// The probabilities of one row.
    #[must_use]
    pub fn row(&self, row: usize) -> &[f64] {
        &self.values[row * self.classes..(row + 1) * self.classes]
    }
}

/// vcfeval-style counts. Signed, as the arithmetic of actions on inconsistent inputs (e.g. more
/// promotions matching truth FNs than there are FNs) must not wrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counts {
    /// True positives: query calls matching the truth genotype.
    pub tp: i64,
    /// False positives: query calls not matching the truth.
    pub fp: i64,
    /// False negatives: truth variants not matched.
    pub fn_: i64,
}

impl Counts {
    /// The F1 score, `2TP / (2TP + FP + FN)` (0 when undefined).
    #[must_use]
    pub fn f1(&self) -> f64 {
        let denominator = 2 * self.tp + self.fp + self.fn_;
        if denominator == 0 { 0.0 } else { (2 * self.tp) as f64 / denominator as f64 }
    }

    /// These counts changed by `delta`.
    #[must_use]
    pub fn plus(self, delta: Counts) -> Counts {
        Counts { tp: self.tp + delta.tp, fp: self.fp + delta.fp, fn_: self.fn_ + delta.fn_ }
    }
}

/// The counts before any action: calls labelled [`Label::Keep`] are TPs, all others FPs.
#[must_use]
pub fn baseline_counts(labels: &[Label], fn_: i64) -> Counts {
    let tp = labels.iter().filter(|&&label| label == Label::Keep).count() as i64;
    Counts { tp, fp: labels.len() as i64 - tp, fn_ }
}

/// The actions decided for each call; a call gets at most one.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Actions {
    /// Calls to drop (including those re-genotyped to `0/0`).
    pub drop: Vec<bool>,
    /// Calls to flip het <-> hom.
    pub flip: Vec<bool>,
    /// Calls to re-genotype (to a non-reference genotype).
    pub regenotype: Vec<bool>,
}

/// The decision thresholds of a genotype model in one mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionThresholds {
    /// The drop threshold.
    pub t_drop: f64,
    /// The flip threshold (ignored in veto mode).
    pub t_flip: f64,
    /// The re-genotype threshold (used only by re-genotype models in genotyper mode).
    pub t_regenotype: f64,
}

/// How a genotype model's probabilities and thresholds become actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Rule {
    /// The thresholds alone (see [`decide`]).
    #[default]
    Thresholds,
    /// The thresholds, then also drop calls left alone whose P(keep) is below one half.
    KeepFallback,
    /// The action of greatest expected F1 gain under the vcfeval arithmetic (for F1 near 1):
    /// flip or re-genotype when its probability is above one half, else drop when P(keep) is
    /// below one half. The model's thresholds are not used in genotyper mode (veto mode drops
    /// when P(keep) is below one half, or as its threshold says). A call's own drop threshold
    /// ([`CallLimits::t_drop`]) is used in both modes.
    ExpectedF1,
}

/// The probability above which [`Rule::ExpectedF1`] fixes a call, and below which a P(keep)
/// drops it ([`Rule::KeepFallback`] and [`Rule::ExpectedF1`]).
pub const ONE_HALF: f64 = 0.5;

/// Limits on the actions taken on one call (e.g. outside a trusted region set).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CallLimits {
    /// Never flip the call.
    pub no_flip: bool,
    /// Never re-genotype the call (nor drop it as re-genotyped to `0/0`).
    pub no_regenotype: bool,
    /// The drop threshold of the call, in place of the model's: P(drop) (P(drop) + P(flip)
    /// (+ P(regenotype)) in veto mode) must be at least this. Under [`Rule::ExpectedF1`] in
    /// genotyper mode, which has no model threshold, the call is dropped at it before a flip or
    /// re-genotype. Drops for a P(keep) below one half ([`Rule::KeepFallback`],
    /// [`Rule::ExpectedF1`]) and re-genotypes to `0/0` are not affected.
    pub t_drop: Option<f64>,
}

/// Decides which calls to drop, flip and re-genotype.
///
/// In genotyper mode the actions take precedence in that order: drop, then flip, then
/// re-genotype. A call re-genotyped to `0/0` is dropped instead.
///
/// # Arguments
///
/// * `probabilities` - class probabilities in [`Label`] order (with a re-genotype column for
///   re-genotype models).
/// * `flippable` - per call, whether the genotype is `0/1` or `1/1`.
/// * `thresholds` - the thresholds.
/// * `mode` - the decision mode.
/// * `regenotypable` - per call, whether re-genotyping can change it.
/// * `regenotypes_to_ref` - per call, whether the length caller's genotype is `0/0`.
#[must_use]
pub fn decide(
    probabilities: &Probabilities,
    flippable: &[bool],
    thresholds: DecisionThresholds,
    mode: Mode,
    regenotypable: &[bool],
    regenotypes_to_ref: &[bool],
) -> Actions {
    let calls = CallFlags { flippable, regenotypable, regenotypes_to_ref, limits: None };
    decide_with_rule(probabilities, &calls, thresholds, mode, Rule::Thresholds)
}

/// Per-call flags that [`decide_with_rule`] needs.
#[derive(Debug, Clone, Copy)]
pub struct CallFlags<'a> {
    /// Per call, whether the genotype is `0/1` or `1/1`.
    pub flippable: &'a [bool],
    /// Per call, whether re-genotyping can change it.
    pub regenotypable: &'a [bool],
    /// Per call, whether the length caller's genotype is `0/0`.
    pub regenotypes_to_ref: &'a [bool],
    /// Per call, limits on its actions (none if `None`).
    pub limits: Option<&'a [CallLimits]>,
}

/// Decides which calls to drop, flip and re-genotype, as [`decide`] does, with a [`Rule`] and
/// per-call [`CallLimits`].
///
/// A call re-genotyped to `0/0` is dropped instead. A call that may not be flipped or
/// re-genotyped (see [`CallLimits`]) falls through to the next action the rule allows.
#[must_use]
pub fn decide_with_rule(
    probabilities: &Probabilities,
    calls: &CallFlags<'_>,
    thresholds: DecisionThresholds,
    mode: Mode,
    rule: Rule,
) -> Actions {
    let n = probabilities.len();
    let has_regenotype = probabilities.classes() > Label::Regenotype.index();
    let p = |row: usize, label: Label| probabilities.get(row, label.index());
    let mut actions =
        Actions { drop: vec![false; n], flip: vec![false; n], regenotype: vec![false; n] };
    for row in 0..n {
        let limits = calls.limits.map_or_else(CallLimits::default, |limits| limits[row]);
        let t_drop = limits.t_drop.unwrap_or(thresholds.t_drop);
        let low_keep = rule != Rule::Thresholds && p(row, Label::Keep) < ONE_HALF;
        if mode == Mode::Veto {
            let mut p_wrong = p(row, Label::Drop) + p(row, Label::Flip);
            if has_regenotype {
                p_wrong += p(row, Label::Regenotype);
            }
            actions.drop[row] = p_wrong >= t_drop || low_keep;
            continue;
        }
        let can_flip = calls.flippable[row] && !limits.no_flip;
        let can_regenotype = has_regenotype && calls.regenotypable[row] && !limits.no_regenotype;
        let (mut drop, mut flip, mut regenotype) = (false, false, false);
        if rule == Rule::ExpectedF1 {
            if limits.t_drop.is_some_and(|t_drop| p(row, Label::Drop) >= t_drop) {
                drop = true;
            } else if can_flip && p(row, Label::Flip) > ONE_HALF {
                flip = true;
            } else if can_regenotype && p(row, Label::Regenotype) > ONE_HALF {
                regenotype = true;
            } else {
                drop = low_keep;
            }
        } else if p(row, Label::Drop) >= t_drop {
            drop = true;
        } else if can_flip && p(row, Label::Flip) >= thresholds.t_flip {
            flip = true;
        } else if can_regenotype && p(row, Label::Regenotype) >= thresholds.t_regenotype {
            regenotype = true;
        } else {
            drop = low_keep;
        }
        if regenotype && calls.regenotypes_to_ref[row] {
            (drop, regenotype) = (true, false);
        }
        actions.drop[row] = drop;
        actions.flip[row] = flip;
        actions.regenotype[row] = regenotype;
    }
    actions
}

/// Applies actions to baseline counts (see the module docs).
///
/// # Arguments
///
/// * `labels` - the [`Label`] of each scored call.
/// * `flip_ok` - per call, whether flipping it gives the truth genotype.
/// * `actions` - the actions (a call in more than one mask takes the first: drop, flip,
///   re-genotype).
/// * `baseline` - the counts before any action.
/// * `regenotype_ok` - per call, whether re-genotyping it gives the truth genotype.
#[must_use]
pub fn rescore(
    labels: &[Label],
    flip_ok: &[bool],
    actions: &Actions,
    baseline: Counts,
    regenotype_ok: &[bool],
) -> Counts {
    // The threshold search rescores every candidate thresholds' actions, so this is a lean loop
    // of four counters rather than an [`Outcome`] (which counts the same way, 30% slower here);
    // `outcomes_sum_to_rescore` checks that the two agree for every label, action and fix.
    let (mut dropped_tp, mut dropped_fp, mut broke, mut fixed) = (0i64, 0i64, 0i64, 0i64);
    for (i, &label) in labels.iter().enumerate() {
        let is_tp = label == Label::Keep;
        if actions.drop[i] {
            if is_tp { dropped_tp += 1 } else { dropped_fp += 1 }
        } else if actions.flip[i] || actions.regenotype[i] {
            let ok = if actions.flip[i] { flip_ok[i] } else { regenotype_ok[i] };
            if is_tp {
                broke += 1;
            } else if ok {
                fixed += 1;
            }
        }
    }
    Counts {
        tp: baseline.tp - dropped_tp - broke + fixed,
        fp: baseline.fp - dropped_fp + broke - fixed,
        fn_: baseline.fn_ + dropped_tp + broke - fixed,
    }
}

/// The promote and flip decisions for non-PASS calls; flipped calls are always promoted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Promotions {
    /// Calls to promote to PASS.
    pub promote: Vec<bool>,
    /// Promoted calls whose genotype is flipped het <-> hom.
    pub flip: Vec<bool>,
}

/// Decides which non-PASS calls to promote, and which promoted calls to flip.
///
/// A call is promoted when P(promote) + P(promote and flip) >= `t_promote`, and flipped when also
/// P(promote and flip) >= `t_flip`, P(promote and flip) > P(promote) and its genotype is `0/1` or
/// `1/1`.
#[must_use]
pub fn decide_promote(
    probabilities: &Probabilities,
    flippable: &[bool],
    t_promote: f64,
    t_flip: f64,
) -> Promotions {
    let n = probabilities.len();
    let mut promotions = Promotions { promote: vec![false; n], flip: vec![false; n] };
    for row in 0..n {
        let p_promote = probabilities.get(row, PromoteLabel::Promote.index());
        let p_flip = probabilities.get(row, PromoteLabel::PromoteFlip.index());
        let promote = p_promote + p_flip >= t_promote;
        promotions.promote[row] = promote;
        promotions.flip[row] = promote && p_flip >= t_flip && p_flip > p_promote && flippable[row];
    }
    promotions
}

/// Applies promotions to baseline counts.
///
/// A promoted call with the right genotype ([`PromoteLabel::Promote`] not flipped, or
/// [`PromoteLabel::PromoteFlip`] flipped) turns a truth FN into a TP; any other promoted call is a
/// new FP.
#[must_use]
pub fn rescore_promote(
    labels: &[PromoteLabel],
    promotions: &Promotions,
    baseline: Counts,
) -> Counts {
    let mut delta = Counts { tp: 0, fp: 0, fn_: 0 };
    for (i, &label) in labels.iter().enumerate() {
        if promotions.promote[i] {
            delta = delta.plus(promotion_delta(label, promotions.flip[i]));
        }
    }
    baseline.plus(delta)
}

/// How promoting a labelled candidate (and flipping it, with `flip`) changes the counts: one with
/// the right genotype ([`PromoteLabel::Promote`] not flipped, or [`PromoteLabel::PromoteFlip`]
/// flipped) turns a truth FN into a TP; any other is a new FP.
const fn promotion_delta(label: PromoteLabel, flip: bool) -> Counts {
    if promotion_is_right(label, flip) {
        Counts { tp: 1, fp: 0, fn_: -1 }
    } else {
        Counts { tp: 0, fp: 1, fn_: 0 }
    }
}

/// Whether a promoted candidate gets the right genotype (see [`promotion_delta`]).
const fn promotion_is_right(label: PromoteLabel, flip: bool) -> bool {
    matches!((label, flip), (PromoteLabel::Promote, false) | (PromoteLabel::PromoteFlip, true))
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

/// The actions on labelled calls (a stratum's in `stratify`, a sample's in `evaluate`) and their
/// outcomes against one truth set.
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
    /// The counts before any action (FNs from the truth FNs of the calls' stratum or sample).
    pub before: Counts,
    /// The change in counts from the actions.
    pub delta: Counts,
}

impl Outcome {
    /// The outcome of actions on labelled PASS calls, with no FNs before them.
    ///
    /// # Arguments
    ///
    /// * `labels` - the [`Label`] of each call.
    /// * `flip_ok` - per call, whether flipping it gives the truth genotype.
    /// * `regenotype_ok` - per call, whether re-genotyping it gives the truth genotype.
    /// * `actions` - the actions (a call in more than one mask takes the first: drop, flip,
    ///   re-genotype).
    #[must_use]
    pub fn of_calls(
        labels: &[Label],
        flip_ok: &[bool],
        regenotype_ok: &[bool],
        actions: &Actions,
    ) -> Outcome {
        let mut outcome = Outcome::default();
        for (i, &label) in labels.iter().enumerate() {
            let action = CallAction::of(actions.drop[i], actions.flip[i], actions.regenotype[i]);
            outcome.add_call(label, flip_ok[i], regenotype_ok[i], action);
        }
        outcome
    }

    /// The outcome of promotions of labelled candidates, with no counts before them.
    #[must_use]
    pub fn of_candidates(labels: &[PromoteLabel], promotions: &Promotions) -> Outcome {
        let mut outcome = Outcome::default();
        for (i, &label) in labels.iter().enumerate() {
            outcome.add_candidate(label, promotions.promote[i], promotions.flip[i]);
        }
        outcome
    }

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
                let (count, fixed, broke, ok) = if action == CallAction::Flip {
                    (&mut self.flipped, &mut self.flips_fixed, &mut self.flips_broke_tp, flip_ok)
                } else {
                    (
                        &mut self.regenotyped,
                        &mut self.regenotypes_fixed,
                        &mut self.regenotypes_broke_tp,
                        regenotype_ok,
                    )
                };
                *count += 1;
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
        if promotion_is_right(label, flip) {
            self.promoted_correct += 1;
        }
        self.delta = self.delta.plus(promotion_delta(label, flip));
    }

    /// The counts after the actions.
    #[must_use]
    pub fn after(&self) -> Counts {
        self.before.plus(self.delta)
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
        self.before = self.before.plus(other.before);
        self.delta = self.delta.plus(other.delta);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn probabilities(rows: &[&[f64]]) -> Probabilities {
        Probabilities::new(rows.concat(), rows.first().map_or(3, |row| row.len()))
    }

    const NEVER: f64 = crate::common::NEVER;

    #[test]
    fn f1_is_zero_when_undefined() {
        assert_eq!(Counts::default().f1(), 0.0);
        assert_eq!(Counts { tp: 2, fp: 1, fn_: 1 }.f1(), 4.0 / 6.0);
    }

    #[test]
    fn promotions_beyond_the_fn_count_do_not_wrap() {
        let promotions = Promotions { promote: vec![true], flip: vec![false] };
        let got =
            rescore_promote(&[PromoteLabel::Promote], &promotions, Counts { tp: 0, fp: 0, fn_: 0 });
        assert_eq!(got, Counts { tp: 1, fp: 0, fn_: -1 });
    }

    #[test]
    fn baseline_counts_keeps_as_tps() {
        let labels = [Label::Keep, Label::Drop, Label::Flip, Label::Keep];
        assert_eq!(baseline_counts(&labels, 7), Counts { tp: 2, fp: 2, fn_: 7 });
    }

    #[test]
    fn genotyper_mode_drops_then_flips() {
        let p = probabilities(&[
            &[0.6, 0.1, 0.3], // drop
            &[0.1, 0.2, 0.7], // flip
            &[0.1, 0.2, 0.7], // would flip, but not flippable
            &[0.3, 0.6, 0.1], // keep
        ]);
        let thresholds = DecisionThresholds { t_drop: 0.5, t_flip: 0.5, t_regenotype: NEVER };
        let flippable = [true, true, false, true];
        let no = [false; 4];
        let actions = decide(&p, &flippable, thresholds, Mode::Genotyper, &no, &no);
        assert_eq!(actions.drop, [true, false, false, false]);
        assert_eq!(actions.flip, [false, true, false, false]);
        assert_eq!(actions.regenotype, no);
    }

    #[test]
    fn veto_mode_drops_wrong_calls_and_never_flips() {
        let p = probabilities(&[&[0.3, 0.4, 0.3], &[0.1, 0.8, 0.1]]);
        let thresholds = DecisionThresholds { t_drop: 0.5, t_flip: 0.0, t_regenotype: 0.0 };
        let actions = decide(&p, &[true, true], thresholds, Mode::Veto, &[true; 2], &[false; 2]);
        assert_eq!(actions.drop, [true, false]);
        assert_eq!(actions.flip, [false, false]);
        // With a re-genotype class, its probability counts as wrong too.
        let p = probabilities(&[&[0.2, 0.45, 0.1, 0.25]]);
        let actions = decide(&p, &[true], thresholds, Mode::Veto, &[true], &[false]);
        assert_eq!(actions.drop, [true]);
    }

    #[rstest]
    #[case::regenotyped(true, false, (false, false, true))]
    #[case::to_ref_is_a_drop(true, true, (true, false, false))]
    #[case::not_regenotypable(false, false, (false, false, false))]
    fn regenotype_decisions(
        #[case] regenotypable: bool,
        #[case] to_ref: bool,
        #[case] expected: (bool, bool, bool),
    ) {
        let p = probabilities(&[&[0.1, 0.2, 0.1, 0.6]]);
        let thresholds = DecisionThresholds { t_drop: 0.5, t_flip: 0.5, t_regenotype: 0.5 };
        let actions = decide(&p, &[true], thresholds, Mode::Genotyper, &[regenotypable], &[to_ref]);
        assert_eq!((actions.drop[0], actions.flip[0], actions.regenotype[0]), expected);
    }

    fn flags<'a>(
        flippable: &'a [bool],
        regenotypable: &'a [bool],
        to_ref: &'a [bool],
        limits: Option<&'a [CallLimits]>,
    ) -> CallFlags<'a> {
        CallFlags { flippable, regenotypable, regenotypes_to_ref: to_ref, limits }
    }

    #[test]
    fn keep_fallback_drops_low_keep_calls_left_alone() {
        let p = probabilities(&[
            &[0.35, 0.45, 0.2], // no threshold met; P(keep) < 0.5: dropped
            &[0.3, 0.55, 0.15], // P(keep) >= 0.5: kept
            &[0.1, 0.2, 0.7],   // flipped by the thresholds first
        ]);
        let thresholds = DecisionThresholds { t_drop: 0.4, t_flip: 0.6, t_regenotype: NEVER };
        let no = [false; 3];
        let calls = flags(&[true; 3], &no, &no, None);
        let actions = decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::KeepFallback);
        assert_eq!(actions.drop, [true, false, false]);
        assert_eq!(actions.flip, [false, false, true]);
        let thresholds_only =
            decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::Thresholds);
        assert_eq!(thresholds_only.drop, [false, false, false]);
    }

    #[test]
    fn expected_f1_fixes_above_one_half_else_drops_below_one_half() {
        let p = probabilities(&[
            &[0.45, 0.0, 0.55, 0.0], // flip beats the drop threshold
            &[0.2, 0.3, 0.0, 0.5],   // P(regenotype) not above 0.5; P(keep) < 0.5: drop
            &[0.1, 0.3, 0.0, 0.6],   // re-genotype
            &[0.2, 0.3, 0.5, 0.0],   // not flippable: drop
            &[0.3, 0.5, 0.2, 0.0],   // keep: P(keep) is not below 0.5
        ]);
        let thresholds = DecisionThresholds { t_drop: 0.4, t_flip: 0.7, t_regenotype: 0.4 };
        let no = [false; 5];
        let calls = flags(&[true, true, true, false, true], &[true; 5], &no, None);
        let actions = decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::ExpectedF1);
        assert_eq!(actions.flip, [true, false, false, false, false]);
        assert_eq!(actions.regenotype, [false, false, true, false, false]);
        assert_eq!(actions.drop, [false, true, false, true, false]);
        // Veto mode drops at its threshold or when P(keep) < 0.5.
        let veto = DecisionThresholds { t_drop: 0.9, t_flip: NEVER, t_regenotype: NEVER };
        let actions = decide_with_rule(&p, &calls, veto, Mode::Veto, Rule::ExpectedF1);
        assert_eq!(actions.drop, [true, true, true, true, false]);
        assert_eq!(actions.flip, no);
    }

    #[test]
    fn limits_withhold_fixes_and_raise_the_drop_threshold() {
        let p = probabilities(&[
            &[0.1, 0.2, 0.7, 0.0], // would flip: withheld, so kept
            &[0.1, 0.2, 0.0, 0.7], // would re-genotype: withheld, so kept
            &[0.5, 0.4, 0.1, 0.0], // drop below the call's t_drop: kept
            &[0.8, 0.1, 0.1, 0.0], // drop at or above it
            &[0.1, 0.2, 0.7, 0.0], // no limits: flipped
            &[0.5, 0.3, 0.2, 0.0], // below the call's t_drop, but P(keep) < 0.5
        ]);
        let thresholds = DecisionThresholds { t_drop: 0.4, t_flip: 0.6, t_regenotype: 0.4 };
        let limited = CallLimits { no_flip: true, no_regenotype: true, t_drop: Some(0.7) };
        let limits = [limited, limited, limited, limited, CallLimits::default(), limited];
        let no = [false; 6];
        let calls = flags(&[true; 6], &[true; 6], &no, Some(&limits));
        let actions = decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::Thresholds);
        assert_eq!(actions.drop, [false, false, false, true, false, false]);
        assert_eq!(actions.flip, [false, false, false, false, true, false]);
        assert_eq!(actions.regenotype, no);
        // The P(keep) fallback is not held to the call's drop threshold: the calls left alone
        // with P(keep) < 0.5 are dropped.
        let actions = decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::KeepFallback);
        assert_eq!(actions.drop, [true, true, true, true, false, true]);
    }

    #[test]
    fn expected_f1_uses_the_calls_limits() {
        let p = probabilities(&[
            &[0.35, 0.55, 0.1, 0.0], // P(drop) >= the call's 0.3: dropped, as in veto mode
            &[0.2, 0.3, 0.0, 0.5],   // below it, P(keep) < 0.5: dropped
            &[0.1, 0.3, 0.6, 0.0],   // flip withheld: falls through to the P(keep) drop
            &[0.1, 0.3, 0.0, 0.6],   // re-genotype withheld: falls through to the P(keep) drop
            &[0.0, 0.6, 0.4, 0.0],   // flip withheld, P(keep) >= 0.5: kept
            &[0.35, 0.55, 0.1, 0.0], // no limits: kept (the model's thresholds are not used)
            &[0.4, 0.0, 0.0, 0.6],   // re-genotype withheld; P(drop) >= the call's 0.3: dropped
        ]);
        let thresholds = DecisionThresholds { t_drop: 0.9, t_flip: 0.9, t_regenotype: 0.9 };
        let limited = CallLimits { no_flip: true, no_regenotype: true, t_drop: Some(0.3) };
        let mut limits = [limited; 7];
        limits[5] = CallLimits::default();
        let no = [false; 7];
        let calls = flags(&[true; 7], &[true; 7], &no, Some(&limits));
        let actions = decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::ExpectedF1);
        assert_eq!(actions.drop, [true, true, true, true, false, false, true]);
        assert_eq!((actions.flip, actions.regenotype), (no.to_vec(), no.to_vec()));
        // Veto mode drops the first call at the call's threshold too (P(drop) + P(flip) = 0.45).
        let veto = decide_with_rule(&p, &calls, thresholds, Mode::Veto, Rule::ExpectedF1);
        assert!(veto.drop[0]);
        // Fixes allowed: the call's drop threshold comes before a flip or re-genotype.
        let fixable = [CallLimits { t_drop: Some(0.3), ..CallLimits::default() }; 2];
        let p = probabilities(&[&[0.35, 0.0, 0.65, 0.0], &[0.2, 0.0, 0.0, 0.8]]);
        let calls = flags(&[true; 2], &[true; 2], &[false; 2], Some(&fixable));
        let actions = decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::ExpectedF1);
        assert_eq!(actions.drop, [true, false]);
        assert_eq!(actions.regenotype, [false, true]);
    }

    /// The call's drop threshold replaces the model's rather than only raising it: below a high
    /// (or never-dropping) model threshold, it drops calls the model alone would keep.
    #[test]
    fn a_call_drop_threshold_below_the_models_lowers_it() {
        let p = probabilities(&[
            &[0.7, 0.3, 0.0], // dropped at the call's 0.6, kept at the model's threshold
            &[0.5, 0.5, 0.0], // below both
        ]);
        let limits = [CallLimits { t_drop: Some(0.6), ..CallLimits::default() }; 2];
        let no = [false; 2];
        for model_t_drop in [0.9, NEVER] {
            let thresholds =
                DecisionThresholds { t_drop: model_t_drop, t_flip: NEVER, t_regenotype: NEVER };
            let unlimited = flags(&[true; 2], &[true; 2], &no, None);
            let actions =
                decide_with_rule(&p, &unlimited, thresholds, Mode::Genotyper, Rule::Thresholds);
            assert_eq!(actions.drop, no);
            let calls = flags(&[true; 2], &[true; 2], &no, Some(&limits));
            let actions =
                decide_with_rule(&p, &calls, thresholds, Mode::Genotyper, Rule::Thresholds);
            assert_eq!(actions.drop, [true, false]);
        }
    }

    #[test]
    fn veto_drops_at_the_calls_drop_threshold() {
        let p = probabilities(&[
            &[0.3, 0.45, 0.25], // P(drop) + P(flip) = 0.55: above the model's 0.4, below the call's
            &[0.6, 0.1, 0.3],   // 0.9: above both
            &[0.3, 0.45, 0.25], // no limits: the model's threshold
            &[0.1, 0.3, 0.6],   // 0.7 >= the call's, though P(drop) alone is low
        ]);
        let thresholds = DecisionThresholds { t_drop: 0.4, t_flip: 0.6, t_regenotype: NEVER };
        let limited = CallLimits { no_flip: true, no_regenotype: true, t_drop: Some(0.7) };
        let limits = [limited, limited, CallLimits::default(), limited];
        let no = [false; 4];
        let calls = flags(&[true; 4], &no, &no, Some(&limits));
        let veto = |rule| decide_with_rule(&p, &calls, thresholds, Mode::Veto, rule);
        let actions = veto(Rule::Thresholds);
        assert_eq!(actions.drop, [false, true, true, true]);
        assert_eq!((actions.flip, actions.regenotype), (no.to_vec(), no.to_vec()));
        // P(keep) < 0.5 drops whatever the call's threshold.
        assert_eq!(veto(Rule::KeepFallback).drop, [true, true, true, true]);
    }

    #[test]
    fn rescore_applies_the_vcfeval_arithmetic() {
        let labels = [
            Label::Keep,
            Label::Drop,
            Label::Keep,
            Label::Flip,
            Label::Flip,
            Label::Regenotype,
            Label::Keep,
        ];
        let actions = Actions {
            drop: vec![true, true, false, false, false, false, false],
            flip: vec![false, false, true, true, true, false, false],
            regenotype: vec![false, false, false, false, false, true, false],
        };
        let flip_ok = [false, false, false, true, false, false, false];
        let regenotype_ok = [false, false, false, false, false, true, false];
        let baseline = baseline_counts(&labels, 10);
        assert_eq!(baseline, Counts { tp: 3, fp: 4, fn_: 10 });
        let got = rescore(&labels, &flip_ok, &actions, baseline, &regenotype_ok);
        // Dropped TP: TP-1 FN+1. Dropped FP: FP-1. Flipped TP: TP-1 FP+1 FN+1. Correct flip and
        // correct re-genotype: TP+1 FP-1 FN-1 each. Incorrect flip: nothing.
        assert_eq!(got, Counts { tp: 3, fp: 2, fn_: 10 });
    }

    #[test]
    fn promotions() {
        let p = probabilities(&[
            &[0.3, 0.5, 0.2],   // promote
            &[0.2, 0.3, 0.5],   // promote and flip
            &[0.2, 0.3, 0.5],   // promote; not flippable
            &[0.6, 0.2, 0.2],   // stay
            &[0.1, 0.45, 0.45], // promote; a tie does not flip
        ]);
        let flippable = [true, true, false, true, true];
        let promotions = decide_promote(&p, &flippable, 0.6, 0.4);
        assert_eq!(promotions.promote, [true, true, true, false, true]);
        assert_eq!(promotions.flip, [false, true, false, false, false]);
        let labels = [
            PromoteLabel::Promote,
            PromoteLabel::PromoteFlip,
            PromoteLabel::PromoteFlip,
            PromoteLabel::Stay,
            PromoteLabel::Stay,
        ];
        let got = rescore_promote(&labels, &promotions, Counts { tp: 5, fp: 1, fn_: 4 });
        assert_eq!(got, Counts { tp: 7, fp: 3, fn_: 2 });
    }

    #[test]
    fn modes_parse_and_print() {
        assert_eq!("GENOTYPER".parse::<Mode>().unwrap(), Mode::Genotyper);
        assert_eq!(Mode::Veto.to_string(), "veto");
        assert!("other".parse::<Mode>().is_err());
        assert_eq!(serde_json::to_string(&Mode::Genotyper).unwrap(), "\"genotyper\"");
    }

    /// Every label, flip_ok, regenotype_ok and action: the outcomes summed over calls are what
    /// `evaluate`'s rescoring gives the same calls and actions.
    #[test]
    fn outcomes_sum_to_rescore() {
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
}
