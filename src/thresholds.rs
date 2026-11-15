//! Decision thresholds chosen by maximising the summed F1 gain over cross-validation folds.
//!
//! Every search also considers [`NEVER`] (take no action), whose gain is exactly zero, so a chosen
//! threshold never has a negative summed gain. Among candidates with equal gain the most
//! conservative wins: the higher first threshold (`t_drop` / `t_promote`), then the higher
//! `t_flip`, then the higher `t_regenotype`. So when the folds give no evidence for acting (every
//! gain is zero), the search returns [`NEVER`].

use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::NEVER;
use crate::labels::{Label, PromoteLabel};
use crate::numeric::python_sum;
use crate::rescore::{
    Actions, CallFlags, CallLimits, Counts, DecisionThresholds, Mode, Probabilities, Rule, decide,
    decide_promote, decide_with_rule, rescore, rescore_promote,
};

/// Candidate thresholds for `t_drop`, `t_flip` and `t_regenotype`.
///
/// Regularized models are well calibrated, so cross-validation often picks thresholds near 0.5;
/// the grid extends below 0.5 (a class need not be the argmax to be acted on) and up to
/// near-certainty.
pub const DEFAULT_GRID: [f64; 17] = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.99,
    0.999,
    0.9999,
    1.0 - 1e-5,
    1.0 - 1e-6,
    1.0 - 1e-7,
    1.0 - 1e-8,
    1.0 - 1e-10,
    NEVER,
];

/// Candidate promotion thresholds.
pub const DEFAULT_PROMOTE_GRID: [f64; 16] =
    [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, NEVER];

/// Candidate flip thresholds for promoted calls.
pub const DEFAULT_PROMOTE_FLIP_GRID: [f64; 8] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, NEVER];

/// The grid plus [`NEVER`], de-duplicated, from most to least conservative.
fn candidates(grid: &[f64]) -> Vec<f64> {
    // A NaN or infinite threshold would be written to the model as JSON null (`train --grid`
    // rejects them).
    debug_assert!(grid.iter().all(|v| v.is_finite()), "non-finite threshold in {grid:?}");
    let mut values: Vec<f64> = grid.iter().copied().chain([NEVER]).collect();
    values.sort_by(|a, b| b.total_cmp(a));
    values.dedup();
    values
}

/// The threshold combination with the highest gain (see the module docs).
///
/// Combinations are visited in the order of the Cartesian product of the candidate lists (the
/// first grid varies slowest); a later combination wins only with a strictly higher gain, so ties
/// go to the most conservative, comparing the thresholds in grid order. Gains are computed in
/// parallel.
fn grid_argmax<F>(gain_of: F, grids: &[&[f64]]) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let lists: Vec<Vec<f64>> = grids.iter().map(|grid| candidates(grid)).collect();
    let total: usize = lists.iter().map(Vec::len).product();
    let combination = |mut index: usize| -> Vec<f64> {
        let mut values = vec![0.0; lists.len()];
        for (k, list) in lists.iter().enumerate().rev() {
            values[k] = list[index % list.len()];
            index /= list.len();
        }
        values
    };
    let gains: Vec<f64> = (0..total).into_par_iter().map(|i| gain_of(&combination(i))).collect();
    let mut best = 0;
    for (i, &gain) in gains.iter().enumerate().skip(1) {
        if gain > gains[best] {
            best = i;
        }
    }
    (combination(best), gains[best])
}

/// Held-out predictions for one sample's labelled calls.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldPredictions {
    /// Class probabilities in [`Label`] order (with a re-genotype column for re-genotype models).
    pub probabilities: Probabilities,
    /// The label of each call.
    pub labels: Vec<Label>,
    /// Per call, whether flipping it gives the truth genotype.
    pub flip_ok: Vec<bool>,
    /// Per call, whether the genotype is `0/1` or `1/1`.
    pub flippable: Vec<bool>,
    /// The sample's counts before any action.
    pub baseline: Counts,
    /// Per call, whether re-genotyping can change it (all false without length calls).
    pub regenotypable: Vec<bool>,
    /// Per call, whether the length caller's genotype is `0/0`.
    pub regenotypes_to_ref: Vec<bool>,
    /// Per call, whether re-genotyping it gives the truth genotype.
    pub regenotype_ok: Vec<bool>,
}

impl FoldPredictions {
    /// The actions the thresholds take on this fold (see [`decide`]).
    #[must_use]
    pub fn decide(&self, thresholds: DecisionThresholds, mode: Mode) -> Actions {
        decide(
            &self.probabilities,
            &self.flippable,
            thresholds,
            mode,
            &self.regenotypable,
            &self.regenotypes_to_ref,
        )
    }

    /// The actions with a [`Rule`] and per-call limits (see [`decide_with_rule`]).
    #[must_use]
    pub fn decide_with_rule(
        &self,
        thresholds: DecisionThresholds,
        mode: Mode,
        rule: Rule,
        limits: Option<&[CallLimits]>,
    ) -> Actions {
        let calls = CallFlags {
            flippable: &self.flippable,
            regenotypable: &self.regenotypable,
            regenotypes_to_ref: &self.regenotypes_to_ref,
            limits,
        };
        decide_with_rule(&self.probabilities, &calls, thresholds, mode, rule)
    }

    /// The counts after actions on this fold (see [`rescore`]).
    #[must_use]
    pub fn rescore(&self, actions: &Actions) -> Counts {
        rescore(&self.labels, &self.flip_ok, actions, self.baseline, &self.regenotype_ok)
    }

    /// The F1 after applying the thresholds to this fold minus the baseline F1.
    #[must_use]
    pub fn f1_gain(&self, thresholds: DecisionThresholds, mode: Mode) -> f64 {
        self.rescore(&self.decide(thresholds, mode)).f1() - self.baseline.f1()
    }
}

/// Picks each row's value from the part `which` names for it: row `i` from `parts[which[i]]`.
fn pick_rows<T: Clone>(parts: &[&[T]], which: &[usize]) -> Result<Vec<T>> {
    if let Some(part) = parts.iter().find(|part| part.len() != which.len()) {
        bail!("a fold of {} calls does not match the {} calls to merge", part.len(), which.len());
    }
    which
        .iter()
        .enumerate()
        .map(|(row, &part)| match parts.get(part) {
            Some(values) => Ok(values[row].clone()),
            None => bail!("call {row} names fold {part} of {}", parts.len()),
        })
        .collect()
}

/// Merges the probabilities of folds of the same calls row by row (see [`pick_rows`]).
fn merge_probabilities(parts: &[&Probabilities], which: &[usize]) -> Result<Probabilities> {
    let classes = parts.first().map_or(1, |p| p.classes());
    if let Some(part) = parts.iter().find(|p| p.classes() != classes || p.len() != which.len()) {
        bail!(
            "fold probabilities of {} calls x {} classes do not match {} calls x {classes}",
            part.len(),
            part.classes(),
            which.len()
        );
    }
    let rows: Vec<&[f64]> = which
        .iter()
        .enumerate()
        .map(|(row, &part)| match parts.get(part) {
            Some(probabilities) => Ok(probabilities.row(row)),
            None => bail!("call {row} names fold {part} of {}", parts.len()),
        })
        .collect::<Result<_>>()?;
    Ok(Probabilities::new(rows.concat(), classes))
}

/// The baseline counts shared by folds of the same calls.
fn shared_baseline(baselines: impl IntoIterator<Item = Counts>) -> Result<Counts> {
    let mut baselines = baselines.into_iter();
    let Some(first) = baselines.next() else { bail!("no folds to merge") };
    if baselines.any(|b| b != first) {
        bail!("folds of the same calls have different baseline counts");
    }
    Ok(first)
}

impl FoldPredictions {
    /// Combines folds predicting the same calls row by row: call `i` from `parts[which[i]]`
    /// (e.g. each call from the fold that held it out).
    ///
    /// # Errors
    ///
    /// Returns an error if there are no parts, the parts have other calls or baselines, or
    /// `which` names a missing part.
    pub fn merge_rows(parts: &[FoldPredictions], which: &[usize]) -> Result<FoldPredictions> {
        let baseline = shared_baseline(parts.iter().map(|p| p.baseline))?;
        let probabilities: Vec<&Probabilities> = parts.iter().map(|p| &p.probabilities).collect();
        let flags = |of: fn(&FoldPredictions) -> &[bool]| -> Result<Vec<bool>> {
            pick_rows(&parts.iter().map(of).collect::<Vec<_>>(), which)
        };
        Ok(FoldPredictions {
            probabilities: merge_probabilities(&probabilities, which)?,
            labels: pick_rows(
                &parts.iter().map(|p| p.labels.as_slice()).collect::<Vec<_>>(),
                which,
            )?,
            flip_ok: flags(|p| &p.flip_ok)?,
            flippable: flags(|p| &p.flippable)?,
            baseline,
            regenotypable: flags(|p| &p.regenotypable)?,
            regenotypes_to_ref: flags(|p| &p.regenotypes_to_ref)?,
            regenotype_ok: flags(|p| &p.regenotype_ok)?,
        })
    }
}

/// Grid-searches the thresholds that maximise the summed F1 gain over folds.
///
/// [`NEVER`] is always a candidate, and ties go to the higher `t_drop`, then the higher `t_flip`,
/// then the higher `t_regenotype` (see the module docs). Fold gains are summed as Python's `sum()`
/// does, so ties resolve as in the Python version.
///
/// # Arguments
///
/// * `folds` - the held-out predictions.
/// * `mode` - the decision mode; in veto mode `t_flip` and `t_regenotype` are [`NEVER`].
/// * `grid` - the candidate thresholds for `t_drop`, `t_flip` and `t_regenotype`.
/// * `regenotype` - also search `t_regenotype` (re-genotype models in genotyper mode).
///
/// # Errors
///
/// Returns an error if there are no folds.
pub fn search_thresholds(
    folds: &[FoldPredictions],
    mode: Mode,
    grid: &[f64],
    regenotype: bool,
) -> Result<Thresholds> {
    if folds.is_empty() {
        bail!("threshold search needs at least one fold");
    }
    let veto = mode == Mode::Veto;
    let never: &[f64] = &[NEVER];
    let flip_grid = if veto { never } else { grid };
    let regenotype_grid = if regenotype && !veto { grid } else { never };
    let (best, gain) = grid_argmax(
        |t: &[f64]| {
            let thresholds = DecisionThresholds { t_drop: t[0], t_flip: t[1], t_regenotype: t[2] };
            python_sum(folds.iter().map(|fold| fold.f1_gain(thresholds, mode)))
        },
        &[grid, flip_grid, regenotype_grid],
    );
    Ok(Thresholds { t_drop: best[0], t_flip: best[1], gain, t_regenotype: best[2] })
}

/// Held-out promote-model predictions for one sample's promotion candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct PromoteFold {
    /// Class probabilities in [`PromoteLabel`] order.
    pub probabilities: Probabilities,
    /// The label of each candidate.
    pub labels: Vec<PromoteLabel>,
    /// Per candidate, whether the genotype is `0/1` or `1/1`.
    pub flippable: Vec<bool>,
    /// The sample's PASS-call counts before any action.
    pub baseline: Counts,
}

impl PromoteFold {
    /// Combines folds predicting the same candidates row by row: candidate `i` from
    /// `parts[which[i]]` (see [`FoldPredictions::merge_rows`]).
    ///
    /// # Errors
    ///
    /// Returns an error if there are no parts, the parts have other candidates or baselines, or
    /// `which` names a missing part.
    pub fn merge_rows(parts: &[PromoteFold], which: &[usize]) -> Result<PromoteFold> {
        let baseline = shared_baseline(parts.iter().map(|p| p.baseline))?;
        let probabilities: Vec<&Probabilities> = parts.iter().map(|p| &p.probabilities).collect();
        Ok(PromoteFold {
            probabilities: merge_probabilities(&probabilities, which)?,
            labels: pick_rows(
                &parts.iter().map(|p| p.labels.as_slice()).collect::<Vec<_>>(),
                which,
            )?,
            flippable: pick_rows(
                &parts.iter().map(|p| p.flippable.as_slice()).collect::<Vec<_>>(),
                which,
            )?,
            baseline,
        })
    }

    /// The F1 after promoting with the thresholds minus the baseline F1.
    #[must_use]
    pub fn f1_gain(&self, t_promote: f64, t_flip: f64) -> f64 {
        let promotions = decide_promote(&self.probabilities, &self.flippable, t_promote, t_flip);
        rescore_promote(&self.labels, &promotions, self.baseline).f1() - self.baseline.f1()
    }
}

/// Grid-searches the promote thresholds that maximise the summed F1 gain over folds.
///
/// [`NEVER`] is always a candidate, and ties go to the higher `t_promote`, then the higher
/// `t_flip` (see the module docs).
///
/// # Errors
///
/// Returns an error if there are no folds.
pub fn search_promote_thresholds(
    folds: &[PromoteFold],
    promote_grid: &[f64],
    flip_grid: &[f64],
) -> Result<PromoteThresholds> {
    if folds.is_empty() {
        bail!("threshold search needs at least one fold");
    }
    let (best, gain) = grid_argmax(
        |t: &[f64]| python_sum(folds.iter().map(|fold| fold.f1_gain(t[0], t[1]))),
        &[promote_grid, flip_grid],
    );
    Ok(PromoteThresholds { t_promote: best[0], t_flip: best[1], gain })
}

/// A genotype model's thresholds in one mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    /// The drop threshold.
    pub t_drop: f64,
    /// The flip threshold ([`NEVER`] in veto mode).
    pub t_flip: f64,
    /// The summed F1 gain over the folds they were chosen on.
    pub gain: f64,
    /// The re-genotype threshold ([`NEVER`] unless the model has the re-genotype class and the
    /// mode is genotyper).
    #[serde(default = "never")]
    pub t_regenotype: f64,
}

impl Thresholds {
    /// The thresholds used to decide actions.
    #[must_use]
    pub fn decision(&self) -> DecisionThresholds {
        DecisionThresholds {
            t_drop: self.t_drop,
            t_flip: self.t_flip,
            t_regenotype: self.t_regenotype,
        }
    }
}

/// A promote model's thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromoteThresholds {
    /// The promotion threshold.
    pub t_promote: f64,
    /// The flip threshold.
    pub t_flip: f64,
    /// The summed F1 gain over the folds they were chosen on.
    pub gain: f64,
}

/// [`NEVER`], the default re-genotype threshold.
fn never() -> f64 {
    NEVER
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(tp: i64, fp: i64, fn_: i64) -> Counts {
        Counts { tp, fp, fn_ }
    }

    fn fold(
        rows: &[&[f64]],
        labels: &[Label],
        flip_ok: &[bool],
        baseline: Counts,
    ) -> FoldPredictions {
        let n = rows.len();
        FoldPredictions {
            probabilities: Probabilities::new(rows.concat(), rows.first().map_or(3, |r| r.len())),
            labels: labels.to_vec(),
            flip_ok: flip_ok.to_vec(),
            flippable: vec![true; n],
            baseline,
            regenotypable: vec![false; n],
            regenotypes_to_ref: vec![false; n],
            regenotype_ok: vec![false; n],
        }
    }

    /// A confident 0/0 FP, a less confident 0/0 FP, a correct-flip FP and a TP that looks flippy.
    fn example() -> FoldPredictions {
        fold(
            &[&[0.999, 0.001, 0.0], &[0.95, 0.05, 0.0], &[0.0, 0.05, 0.95], &[0.0, 0.4, 0.6]],
            &[Label::Drop, Label::Drop, Label::Flip, Label::Keep],
            &[false, false, true, false],
            counts(10, 3, 2),
        )
    }

    fn promote_fold(rows: &[&[f64]], labels: &[PromoteLabel], baseline: Counts) -> PromoteFold {
        PromoteFold {
            probabilities: Probabilities::new(rows.concat(), 3),
            labels: labels.to_vec(),
            flippable: vec![true; rows.len()],
            baseline,
        }
    }

    #[test]
    fn merges_folds_row_by_row() {
        let a = fold(
            &[&[0.9, 0.1, 0.0], &[0.8, 0.2, 0.0], &[0.7, 0.3, 0.0]],
            &[Label::Drop, Label::Keep, Label::Flip],
            &[false, false, true],
            counts(10, 3, 2),
        );
        let mut b = fold(
            &[&[0.1, 0.9, 0.0], &[0.2, 0.8, 0.0], &[0.3, 0.7, 0.0]],
            &[Label::Drop, Label::Regenotype, Label::Flip],
            &[false, true, true],
            counts(10, 3, 2),
        );
        b.regenotype_ok = vec![false, true, false];
        let merged = FoldPredictions::merge_rows(&[a.clone(), b.clone()], &[0, 1, 0]).unwrap();
        let mut expected = fold(
            &[&[0.9, 0.1, 0.0], &[0.2, 0.8, 0.0], &[0.7, 0.3, 0.0]],
            &[Label::Drop, Label::Regenotype, Label::Flip],
            &[false, true, true],
            counts(10, 3, 2),
        );
        expected.regenotype_ok = vec![false, true, false];
        assert_eq!(merged, expected);
        let error = |which: &[usize]| {
            FoldPredictions::merge_rows(&[a.clone(), b.clone()], which).unwrap_err().to_string()
        };
        assert_eq!(error(&[0, 2, 0]), "call 1 names fold 2 of 2");
        assert_eq!(
            error(&[0, 1]),
            "fold probabilities of 3 calls x 3 classes do not match 2 calls x 3"
        );
        b.baseline = counts(10, 3, 3);
        let message = FoldPredictions::merge_rows(&[a, b], &[0, 1, 0]).unwrap_err().to_string();
        assert_eq!(message, "folds of the same calls have different baseline counts");
        let p = promote_fold(
            &[&[0.9, 0.1, 0.0], &[0.8, 0.2, 0.0]],
            &[PromoteLabel::Stay; 2],
            counts(1, 1, 1),
        );
        let q = promote_fold(
            &[&[0.1, 0.9, 0.0], &[0.2, 0.8, 0.0]],
            &[PromoteLabel::Promote; 2],
            counts(1, 1, 1),
        );
        let merged = PromoteFold::merge_rows(&[p, q], &[1, 0]).unwrap();
        let expected = promote_fold(
            &[&[0.1, 0.9, 0.0], &[0.8, 0.2, 0.0]],
            &[PromoteLabel::Promote, PromoteLabel::Stay],
            counts(1, 1, 1),
        );
        assert_eq!(merged, expected);
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn genotyper_search_picks_the_best_thresholds() {
        let best =
            search_thresholds(&[example()], Mode::Genotyper, &[0.5, 0.9, NEVER], false).unwrap();
        // Dropping both 0/0 FPs ties at 0.5 and 0.9 (the more conservative wins); only the
        // confident flip is flipped.
        assert_eq!((best.t_drop, best.t_flip, best.t_regenotype), (0.9, 0.9, NEVER));
        assert_close(best.gain, counts(11, 0, 1).f1() - counts(10, 3, 2).f1());
    }

    #[test]
    fn veto_search_never_flips_and_sums_folds() {
        let folds = [example(), example()];
        let best = search_thresholds(&folds, Mode::Veto, &[0.5, 0.99, NEVER], false).unwrap();
        assert_eq!((best.t_drop, best.t_flip), (0.5, NEVER));
        assert_close(best.gain, 2.0 * (counts(9, 0, 3).f1() - counts(10, 3, 2).f1()));
    }

    #[test]
    fn no_evidence_means_never() {
        let fold = promote_fold(&[&[0.7, 0.25, 0.05]], &[PromoteLabel::Stay], counts(10, 0, 5));
        let best = search_promote_thresholds(&[fold], &[0.5, 0.9], &[0.5]).unwrap();
        assert_eq!((best.t_promote, best.t_flip, best.gain), (NEVER, NEVER, 0.0));
        let no_calls = fold_without_calls();
        let best = search_thresholds(&[no_calls], Mode::Genotyper, &DEFAULT_GRID, false).unwrap();
        assert_eq!((best.t_drop, best.t_flip, best.gain), (NEVER, NEVER, 0.0));
    }

    #[test]
    fn a_fold_decides_with_a_rule_and_limits() {
        let fold = example();
        // The TP that looks flippy has P(keep) 0.4: the keep fallback drops it when the flip
        // threshold leaves it alone.
        let thresholds = DecisionThresholds { t_drop: 0.9, t_flip: 0.99, t_regenotype: NEVER };
        let plain = fold.decide_with_rule(thresholds, Mode::Genotyper, Rule::Thresholds, None);
        assert_eq!(plain, fold.decide(thresholds, Mode::Genotyper));
        let fallback = fold.decide_with_rule(thresholds, Mode::Genotyper, Rule::KeepFallback, None);
        assert_eq!(fallback.drop, [true, true, true, true]);
        // A stricter drop threshold for the first two calls keeps the second (P(drop) 0.95).
        let strict = CallLimits { t_drop: Some(0.99), ..CallLimits::default() };
        let limits = [strict, strict, CallLimits::default(), CallLimits::default()];
        let limited =
            fold.decide_with_rule(thresholds, Mode::Genotyper, Rule::Thresholds, Some(&limits));
        assert_eq!(limited.drop, [true, false, false, false]);
    }

    fn fold_without_calls() -> FoldPredictions {
        fold(&[], &[], &[], counts(10, 3, 2))
    }

    #[test]
    fn never_is_always_a_candidate() {
        let mut harmful = example();
        harmful.labels = vec![Label::Keep; 4];
        harmful.flip_ok = vec![false; 4];
        harmful.baseline = counts(4, 0, 0);
        let best = search_thresholds(&[harmful], Mode::Genotyper, &[0.5, 0.9], false).unwrap();
        assert_eq!((best.t_drop, best.t_flip, best.gain), (NEVER, NEVER, 0.0));
        let promote = promote_fold(&[&[0.1, 0.8, 0.1]], &[PromoteLabel::Stay], counts(10, 0, 5));
        let best = search_promote_thresholds(&[promote], &[0.5], &[0.5]).unwrap();
        assert_eq!((best.t_promote, best.gain), (NEVER, 0.0));
    }

    #[test]
    fn searches_require_folds() {
        let message = search_thresholds(&[], Mode::Genotyper, &DEFAULT_GRID, false).unwrap_err();
        assert_eq!(message.to_string(), "threshold search needs at least one fold");
        let message =
            search_promote_thresholds(&[], &DEFAULT_PROMOTE_GRID, &DEFAULT_PROMOTE_FLIP_GRID)
                .unwrap_err();
        assert_eq!(message.to_string(), "threshold search needs at least one fold");
    }

    #[test]
    fn the_default_grid_can_act_below_one_half() {
        // Every FP has P(drop) 0.35 and every TP 0.05: only a threshold below 0.5 helps.
        let fp: &[f64] = &[0.35, 0.33, 0.32];
        let tp: &[f64] = &[0.05, 0.9, 0.05];
        let labels = [[Label::Drop; 3], [Label::Keep; 3]].concat();
        let fold = fold(&[fp, fp, fp, tp, tp, tp], &labels, &[false; 6], counts(3, 3, 0));
        let best = search_thresholds(&[fold], Mode::Genotyper, &DEFAULT_GRID, false).unwrap();
        assert_eq!(best.t_drop, 0.3);
        assert!(best.gain > 0.0);
    }

    #[test]
    fn promote_search() {
        // Two correct promotions (one needs a flip) and one wrong candidate with lower confidence.
        let fold = promote_fold(
            &[&[0.1, 0.8, 0.1], &[0.2, 0.2, 0.6], &[0.5, 0.4, 0.1]],
            &[PromoteLabel::Promote, PromoteLabel::PromoteFlip, PromoteLabel::Stay],
            counts(10, 0, 5),
        );
        let best = search_promote_thresholds(&[fold], &[0.5, 0.7, NEVER], &[0.5, NEVER]).unwrap();
        assert_eq!((best.t_promote, best.t_flip), (0.7, 0.5));
        assert_close(best.gain, counts(12, 0, 3).f1() - counts(10, 0, 5).f1());
    }

    #[test]
    fn regenotype_search() {
        // A confident correct re-genotype, and a less confident one that breaks a TP.
        let mut fold = fold(
            &[&[0.0, 0.1, 0.0, 0.9], &[0.0, 0.4, 0.0, 0.6]],
            &[Label::Regenotype, Label::Keep],
            &[false, false],
            counts(10, 1, 1),
        );
        fold.regenotypable = vec![true, true];
        fold.regenotype_ok = vec![true, false];
        let folds = [fold];
        let best = search_thresholds(&folds, Mode::Genotyper, &[0.5, 0.8], true).unwrap();
        assert_eq!((best.t_drop, best.t_flip, best.t_regenotype), (NEVER, NEVER, 0.8));
        assert_close(best.gain, counts(11, 0, 0).f1() - counts(10, 1, 1).f1());
        let without = search_thresholds(&folds, Mode::Genotyper, &[0.5, 0.8], false).unwrap();
        assert_eq!(without.t_regenotype, NEVER);
        let veto = search_thresholds(&folds, Mode::Veto, &DEFAULT_GRID, true).unwrap();
        assert_eq!(veto.t_regenotype, NEVER);
    }

    #[test]
    fn candidates_are_deduplicated_from_most_to_least_conservative() {
        assert_eq!(candidates(&[0.5, NEVER, 0.9, 0.5]), [NEVER, 0.9, 0.5]);
    }
}
