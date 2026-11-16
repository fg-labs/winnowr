//! F1 reports for a model's decisions on a labelled sample.

use std::path::Path;

use anyhow::{Context, Result};

use crate::common::NEVER;
use crate::io::write_tsv;
use crate::labels::Label;
use crate::numeric::metric_float;
use crate::policy::withhold_promotions;
use crate::rescore::{
    CallLimits, Counts, DecisionThresholds, Mode, Outcome, Probabilities, Rule, decide_promote,
};
use crate::samples::{PromoteSampleCalls, SampleCalls};
use crate::thresholds::{
    DEFAULT_GRID, DEFAULT_PROMOTE_FLIP_GRID, DEFAULT_PROMOTE_GRID, search_promote_thresholds,
    search_thresholds,
};

/// Thresholds that take no action.
const NO_ACTION: DecisionThresholds =
    DecisionThresholds { t_drop: NEVER, t_flip: NEVER, t_regenotype: NEVER };

/// How a [`crate::policy::Policy`] changes a genotype model's policies that `evaluate` reports,
/// beyond their thresholds, so that they decide as `score` does.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvaluationPolicy<'a> {
    /// The genotype model's rule.
    pub rule: Rule,
    /// Per labelled call, the limits on its actions (none if `None`).
    pub limits: Option<&'a [CallLimits]>,
}

/// The outcome of applying one decision policy to a labelled sample.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyMetric {
    /// The sample name.
    pub sample: String,
    /// The variant type.
    pub variant_type: String,
    /// The policy name (`baseline`, `veto`, `genotyper`, `genotyper_no_regenotype`, `flip_only`,
    /// `promote`, `promote_no_flip`, `genotyper+promote`, or `*_tuned` for thresholds tuned on
    /// this sample, which are optimistic).
    pub policy: String,
    /// The decision mode of the PASS-call policy.
    pub mode: Mode,
    /// The drop threshold of the PASS-call policy.
    pub t_drop: f64,
    /// The flip threshold of the PASS-call policy.
    pub t_flip: f64,
    /// PASS calls dropped.
    pub dropped: i64,
    /// True-positive PASS calls dropped.
    pub dropped_tp: i64,
    /// PASS calls flipped.
    pub flipped: i64,
    /// PASS-call flips that produced the truth genotype.
    pub flips_fixed: i64,
    /// Flips of true-positive PASS calls.
    pub flips_broke_tp: i64,
    /// The counts after all actions.
    pub counts: Counts,
    /// The F1 change versus the baseline.
    pub f1_gain: f64,
    /// The promotion threshold (promote policies).
    pub t_promote: f64,
    /// The flip threshold for promoted calls (promote policies).
    pub t_promote_flip: f64,
    /// Non-PASS calls promoted.
    pub promoted: i64,
    /// Promoted calls that turned a truth FN into a TP.
    pub promoted_correct: i64,
    /// Promoted calls whose genotype was flipped.
    pub promoted_flipped: i64,
    /// The re-genotype threshold of the PASS-call policy (re-genotype models).
    pub t_regenotype: f64,
    /// PASS calls re-genotyped (not counting those re-genotyped to 0/0, which are dropped).
    pub regenotyped: i64,
    /// Re-genotyped PASS calls that got the truth genotype.
    pub regenotypes_fixed: i64,
    /// Re-genotyped true-positive PASS calls.
    pub regenotypes_broke_tp: i64,
}

impl PolicyMetric {
    /// The metric's values as written, in [`POLICY_METRIC_COLUMNS`] order.
    #[must_use]
    pub fn formatted_values(&self) -> Vec<String> {
        let f = metric_float;
        vec![
            self.sample.clone(),
            self.variant_type.clone(),
            self.policy.clone(),
            self.mode.to_string(),
            f(self.t_drop),
            f(self.t_flip),
            self.dropped.to_string(),
            self.dropped_tp.to_string(),
            self.flipped.to_string(),
            self.flips_fixed.to_string(),
            self.flips_broke_tp.to_string(),
            self.counts.tp.to_string(),
            self.counts.fp.to_string(),
            self.counts.fn_.to_string(),
            f(self.counts.f1()),
            f(self.f1_gain),
            f(self.t_promote),
            f(self.t_promote_flip),
            self.promoted.to_string(),
            self.promoted_correct.to_string(),
            self.promoted_flipped.to_string(),
            f(self.t_regenotype),
            self.regenotyped.to_string(),
            self.regenotypes_fixed.to_string(),
            self.regenotypes_broke_tp.to_string(),
        ]
    }
}

/// Applies one policy to a sample's labelled calls and rescores.
///
/// # Errors
///
/// Returns an error if the calls lack a column of their fold.
pub fn evaluate_policy(
    calls: &SampleCalls,
    probabilities: &Probabilities,
    variant_type: &str,
    policy: &str,
    mode: Mode,
    thresholds: DecisionThresholds,
    acting: EvaluationPolicy<'_>,
) -> Result<PolicyMetric> {
    let fold = calls.to_fold(probabilities.clone())?;
    let actions = fold.decide_with_rule(thresholds, mode, acting.rule, acting.limits);
    let outcome = Outcome::of_calls(&fold.labels, &fold.flip_ok, &fold.regenotype_ok, &actions);
    let counts = fold.baseline.plus(outcome.delta);
    Ok(PolicyMetric {
        sample: calls.sample.clone(),
        variant_type: variant_type.to_string(),
        policy: policy.to_string(),
        mode,
        t_drop: thresholds.t_drop,
        t_flip: thresholds.t_flip,
        dropped: outcome.dropped,
        dropped_tp: outcome.dropped_tp,
        flipped: outcome.flipped,
        flips_fixed: outcome.flips_fixed,
        flips_broke_tp: outcome.flips_broke_tp,
        counts,
        f1_gain: counts.f1() - fold.baseline.f1(),
        t_promote: NEVER,
        t_promote_flip: NEVER,
        promoted: 0,
        promoted_correct: 0,
        promoted_flipped: 0,
        t_regenotype: thresholds.t_regenotype,
        regenotyped: outcome.regenotyped,
        regenotypes_fixed: outcome.regenotypes_fixed,
        regenotypes_broke_tp: outcome.regenotypes_broke_tp,
    })
}

/// The outcome with no actions (the `baseline` policy).
///
/// # Errors
///
/// Returns an error if the calls lack a column of their fold.
pub fn baseline_metric(calls: &SampleCalls, variant_type: &str) -> Result<PolicyMetric> {
    let none = Probabilities::new(vec![0.0; calls.labels.len() * 3], 3);
    let plain = EvaluationPolicy::default();
    evaluate_policy(calls, &none, variant_type, "baseline", Mode::Genotyper, NO_ACTION, plain)
}

/// Reports the baseline, each mode at the given thresholds, flips alone, and optionally
/// thresholds tuned on this sample (optimistic: they see the test data).
///
/// For a re-genotype model (4 class probabilities) also reports `genotyper_no_regenotype`: the
/// genotyper thresholds with the re-genotype action off.
///
/// # Arguments
///
/// * `thresholds` - a function giving the model's thresholds for a mode.
/// * `acting` - how the mode policies (veto, genotyper, genotyper_no_regenotype and the tuned
///   ones) act beyond their thresholds; `baseline` and `flip_only` use the thresholds alone. Tuned
///   thresholds are searched with the thresholds alone.
///
/// # Errors
///
/// Returns an error if the calls lack a column of their fold.
pub fn evaluate_model(
    calls: &SampleCalls,
    probabilities: &Probabilities,
    variant_type: &str,
    thresholds: impl Fn(Mode) -> DecisionThresholds,
    tuned: bool,
    acting: EvaluationPolicy<'_>,
) -> Result<Vec<PolicyMetric>> {
    let mut metrics = vec![baseline_metric(calls, variant_type)?];
    for mode in [Mode::Veto, Mode::Genotyper] {
        metrics.push(evaluate_policy(
            calls,
            probabilities,
            variant_type,
            mode.as_str(),
            mode,
            thresholds(mode),
            acting,
        )?);
    }
    let genotyper = thresholds(Mode::Genotyper);
    let regenotype = probabilities.classes() > Label::Regenotype.index();
    if regenotype {
        metrics.push(evaluate_policy(
            calls,
            probabilities,
            variant_type,
            "genotyper_no_regenotype",
            Mode::Genotyper,
            DecisionThresholds { t_regenotype: NEVER, ..genotyper },
            acting,
        )?);
    }
    metrics.push(evaluate_policy(
        calls,
        probabilities,
        variant_type,
        "flip_only",
        Mode::Genotyper,
        DecisionThresholds { t_flip: genotyper.t_flip, ..NO_ACTION },
        EvaluationPolicy::default(),
    )?);
    if tuned {
        let folds = [calls.to_fold(probabilities.clone())?];
        for mode in [Mode::Veto, Mode::Genotyper] {
            let best = search_thresholds(&folds, mode, &DEFAULT_GRID, regenotype)?;
            metrics.push(evaluate_policy(
                calls,
                probabilities,
                variant_type,
                &format!("{mode}_tuned"),
                mode,
                best.decision(),
                acting,
            )?);
        }
    }
    Ok(metrics)
}

/// Applies promotions on top of a PASS-call outcome and rescores.
///
/// The PASS-call columns come from `pass_metric`, the promote columns from the promotions, and
/// the counts and F1 gain (relative to the PASS-call baseline) include both.
///
/// # Errors
///
/// Returns an error if the candidates lack `flippable`.
pub fn evaluate_promote_policy(
    candidates: &PromoteSampleCalls,
    probabilities: &Probabilities,
    policy: &str,
    t_promote: f64,
    t_flip: f64,
    pass_metric: &PolicyMetric,
    withheld: Option<&[bool]>,
) -> Result<PolicyMetric> {
    let fold = candidates.to_fold(probabilities.clone())?;
    let mut promotions = decide_promote(probabilities, &fold.flippable, t_promote, t_flip);
    if let Some(withheld) = withheld {
        withhold_promotions(&mut promotions, withheld);
    }
    let outcome = Outcome::of_candidates(&fold.labels, &promotions);
    let counts = pass_metric.counts.plus(outcome.delta);
    Ok(PolicyMetric {
        policy: policy.to_string(),
        t_promote,
        t_promote_flip: t_flip,
        promoted: outcome.promoted,
        promoted_correct: outcome.promoted_correct,
        promoted_flipped: outcome.promoted_flipped,
        counts,
        f1_gain: counts.f1() - candidates.baseline.f1(),
        ..pass_metric.clone()
    })
}

/// Reports promotions on top of the PASS baseline and, if present, the genotyper's outcome.
///
/// # Arguments
///
/// * `pass_metrics` - PASS-call outcomes to add promotions to; must include `baseline`.
/// * `tuned` - also report thresholds tuned on this sample (optimistic).
/// * `withheld` - per candidate, whether promoting it is withheld (none if `None`).
///
/// # Errors
///
/// Returns an error if `pass_metrics` has no baseline or the candidates lack `flippable`.
pub fn evaluate_promote_model(
    candidates: &PromoteSampleCalls,
    probabilities: &Probabilities,
    t_promote: f64,
    t_flip: f64,
    pass_metrics: &[PolicyMetric],
    tuned: bool,
    withheld: Option<&[bool]>,
) -> Result<Vec<PolicyMetric>> {
    let by_policy = |name: &str| pass_metrics.iter().find(|m| m.policy == name);
    let baseline = by_policy("baseline").context("the PASS-call outcomes have no baseline")?;
    let promote = |policy: &str, t_promote: f64, t_flip: f64, pass: &PolicyMetric| {
        evaluate_promote_policy(
            candidates,
            probabilities,
            policy,
            t_promote,
            t_flip,
            pass,
            withheld,
        )
    };
    let mut metrics = vec![
        promote("promote", t_promote, t_flip, baseline)?,
        promote("promote_no_flip", t_promote, NEVER, baseline)?,
    ];
    if let Some(genotyper) = by_policy("genotyper") {
        metrics.push(promote("genotyper+promote", t_promote, t_flip, genotyper)?);
    }
    if tuned {
        let folds = [candidates.to_fold(probabilities.clone())?];
        let best =
            search_promote_thresholds(&folds, &DEFAULT_PROMOTE_GRID, &DEFAULT_PROMOTE_FLIP_GRID)?;
        metrics.push(promote("promote_tuned", best.t_promote, best.t_flip, baseline)?);
    }
    Ok(metrics)
}

/// The columns of a metrics file, in order.
pub const POLICY_METRIC_COLUMNS: [&str; 25] = [
    "sample",
    "variant_type",
    "policy",
    "mode",
    "t_drop",
    "t_flip",
    "dropped",
    "dropped_tp",
    "flipped",
    "flips_fixed",
    "flips_broke_tp",
    "tp",
    "fp",
    "fn",
    "f1",
    "f1_gain",
    "t_promote",
    "t_promote_flip",
    "promoted",
    "promoted_correct",
    "promoted_flipped",
    "t_regenotype",
    "regenotyped",
    "regenotypes_fixed",
    "regenotypes_broke_tp",
];

/// Writes metrics as a tab-separated file with a header line.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_policy_metrics(path: &Path, metrics: &[PolicyMetric]) -> Result<()> {
    write_tsv(path, &POLICY_METRIC_COLUMNS, metrics.iter().map(PolicyMetric::formatted_values))
}

#[cfg(test)]
mod tests {
    use super::*;

    mod policies {
        use super::*;
        use crate::common::VariantType;
        use crate::labels::PromoteLabel;
        use crate::table::FeatureTable;
        use crate::testing::call_table;

        /// Five labelled PASS SNPs (TP, FP, fixable flip, TP, TP), two promotion candidates and
        /// a promote-flip candidate, with 2 truth FNs.
        fn table() -> FeatureTable {
            let mut table = call_table(&[
                ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 20, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 30, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 40, "A", "C", "PASS", "snp", "1/1", false),
                ("chr1", 50, "A", "C", "PASS", "snp", "0/1", false),
                ("chr1", 60, "A", "C", "LowQual", "snp", "0/1", false),
                ("chr1", 70, "A", "C", "LowQual", "snp", "0/1", false),
                ("chr1", 80, "A", "C", "LowQual", "snp", "0/1", false),
            ]);
            table.info_mut().fn_counts =
                Some([("snp".to_string(), 2), ("indel".to_string(), 0)].into_iter().collect());
            table.info_mut().promote_labelled = true;
            let nan = f64::NAN;
            table
                .with_f64_column("label", vec![1.0, 0.0, 2.0, 1.0, 1.0, nan, nan, nan])
                .unwrap()
                .with_bool_column(
                    "flip_ok",
                    vec![false, false, true, false, false, false, false, false],
                )
                .unwrap()
                .with_bool_column("flippable", vec![true; 8])
                .unwrap()
                .with_f64_column("truth_delta_lo", vec![nan; 8])
                .unwrap()
                .with_f64_column("truth_delta_hi", vec![nan; 8])
                .unwrap()
                .with_f64_column("promote_label", vec![nan, nan, nan, nan, nan, 1.0, 0.0, 2.0])
                .unwrap()
        }

        fn probabilities(rows: &[&[f64]]) -> Probabilities {
            Probabilities::new(rows.concat(), rows[0].len())
        }

        /// Drops the TP and the FP, flips the fixable flip and the hom TP, keeps the last TP.
        fn genotype_probabilities() -> Probabilities {
            probabilities(&[
                &[0.8, 0.2, 0.0],
                &[0.9, 0.1, 0.0],
                &[0.1, 0.2, 0.7],
                &[0.1, 0.3, 0.6],
                &[0.1, 0.9, 0.0],
            ])
        }

        const THRESHOLDS: DecisionThresholds =
            DecisionThresholds { t_drop: 0.5, t_flip: 0.5, t_regenotype: NEVER };

        fn counts(tp: i64, fp: i64, fn_: i64) -> Counts {
            Counts { tp, fp, fn_ }
        }

        /// A metric's policy, action counts and TP/FP/FN.
        fn summary(m: &PolicyMetric) -> (&str, [i64; 5], Counts) {
            let actions = [m.dropped, m.dropped_tp, m.flipped, m.flips_fixed, m.flips_broke_tp];
            (m.policy.as_str(), actions, m.counts)
        }

        #[test]
        fn each_policy_is_rescored() {
            let calls = SampleCalls::from_table(&table(), VariantType::Snp, None, false).unwrap();
            assert_eq!(calls.baseline, counts(3, 2, 2));
            // The keep fallback acts in the mode policies, but not in flip_only.
            let acting = EvaluationPolicy { rule: Rule::KeepFallback, limits: None };
            let metrics = evaluate_model(
                &calls,
                &genotype_probabilities(),
                "snp",
                |_| THRESHOLDS,
                false,
                acting,
            )
            .unwrap();
            let got: Vec<_> = metrics.iter().map(summary).collect();
            let expected = [
                ("baseline", [0, 0, 0, 0, 0], counts(3, 2, 2)),
                // P(drop) + P(flip) >= 0.5 drops all but the last.
                ("veto", [4, 2, 0, 0, 0], counts(1, 0, 4)),
                ("genotyper", [2, 1, 2, 1, 1], counts(2, 1, 3)),
                // The thresholds alone, so the TP with P(keep) 0.2 is not dropped.
                ("flip_only", [0, 0, 2, 1, 1], counts(3, 2, 2)),
            ];
            assert_eq!(got, expected);
            let genotyper = &metrics[2];
            assert_eq!((genotyper.t_drop, genotyper.t_flip), (0.5, 0.5));
            // The F1 gain is against the baseline: 2*2/(4+1+3) - 6/10.
            assert!((genotyper.f1_gain - (0.5 - 0.6)).abs() < 1e-12);
            assert_eq!(metrics[1].mode, Mode::Veto);
            assert_eq!(metrics[3].t_drop, NEVER);
            // With the thresholds alone, the genotyper policy is the same here.
            let plain = evaluate_model(
                &calls,
                &genotype_probabilities(),
                "snp",
                |_| THRESHOLDS,
                false,
                EvaluationPolicy::default(),
            )
            .unwrap();
            assert_eq!(summary(&plain[2]), summary(&metrics[2]));
        }

        #[test]
        fn the_keep_fallback_drops_calls_the_thresholds_leave() {
            let calls = SampleCalls::from_table(&table(), VariantType::Snp, None, false).unwrap();
            // The last TP now has P(keep) 0.4: left alone by the thresholds.
            let p = probabilities(&[
                &[0.8, 0.2, 0.0],
                &[0.9, 0.1, 0.0],
                &[0.1, 0.2, 0.7],
                &[0.1, 0.3, 0.6],
                &[0.3, 0.4, 0.3],
            ]);
            let genotyper = |rule| {
                let acting = EvaluationPolicy { rule, limits: None };
                let policy =
                    evaluate_policy(&calls, &p, "snp", "g", Mode::Genotyper, THRESHOLDS, acting);
                summary(&policy.unwrap()).1
            };
            assert_eq!(genotyper(Rule::Thresholds), [2, 1, 2, 1, 1]);
            assert_eq!(genotyper(Rule::KeepFallback), [3, 2, 2, 1, 1]);
        }

        #[test]
        fn regenotype_models_report_regenotypes() {
            let mut calls =
                SampleCalls::from_table(&table(), VariantType::Snp, None, false).unwrap();
            // A TP and a re-genotype FP, both re-genotypable and fixed by it (the TP too: it is
            // not counted as fixed).
            calls.labels[1] = Label::Regenotype;
            let n = calls.labels.len();
            let regenotypable = vec![true, true, false, false, false];
            calls.calls = calls
                .calls
                .clone()
                .with_bool_column("regenotypable", regenotypable.clone())
                .unwrap()
                .with_bool_column("regenotype_ok", regenotypable)
                .unwrap()
                .with_bool_column("regenotypes_to_ref", vec![false; n])
                .unwrap();
            let p = probabilities(&[
                &[0.0, 0.2, 0.0, 0.8],
                &[0.0, 0.2, 0.0, 0.8],
                &[0.0, 1.0, 0.0, 0.0],
                &[0.0, 1.0, 0.0, 0.0],
                &[0.0, 1.0, 0.0, 0.0],
            ]);
            let thresholds = DecisionThresholds { t_regenotype: 0.5, ..THRESHOLDS };
            let metrics = evaluate_model(
                &calls,
                &p,
                "snp",
                |_| thresholds,
                false,
                EvaluationPolicy::default(),
            )
            .unwrap();
            let names: Vec<&str> = metrics.iter().map(|m| m.policy.as_str()).collect();
            assert_eq!(
                names,
                ["baseline", "veto", "genotyper", "genotyper_no_regenotype", "flip_only"]
            );
            let regenotypes =
                |m: &PolicyMetric| (m.regenotyped, m.regenotypes_fixed, m.regenotypes_broke_tp);
            assert_eq!(regenotypes(&metrics[2]), (2, 1, 1));
            assert_eq!(metrics[2].t_regenotype, 0.5);
            assert_eq!(regenotypes(&metrics[3]), (0, 0, 0));
            assert_eq!(metrics[3].t_regenotype, NEVER);
        }

        #[test]
        fn promotions_add_to_the_pass_outcomes() {
            let table = table();
            let calls = SampleCalls::from_table(&table, VariantType::Snp, None, false).unwrap();
            let candidates = PromoteSampleCalls::from_table(&table, VariantType::Snp).unwrap();
            assert_eq!(
                candidates.labels,
                [PromoteLabel::Promote, PromoteLabel::Stay, PromoteLabel::PromoteFlip]
            );
            let pass = evaluate_model(
                &calls,
                &genotype_probabilities(),
                "snp",
                |_| THRESHOLDS,
                false,
                EvaluationPolicy::default(),
            )
            .unwrap();
            // Promoted right; promoted and flipped, though it should stay; not promoted.
            let p = probabilities(&[&[0.1, 0.8, 0.1], &[0.3, 0.1, 0.6], &[0.9, 0.05, 0.05]]);
            let promoted = |withheld: Option<&[bool]>| {
                evaluate_promote_model(&candidates, &p, 0.5, 0.5, &pass, false, withheld).unwrap()
            };
            let metrics = promoted(None);
            let got: Vec<_> = metrics
                .iter()
                .map(|m| {
                    let actions = [m.promoted, m.promoted_correct, m.promoted_flipped];
                    (m.policy.as_str(), actions, m.counts)
                })
                .collect();
            let expected = [
                ("promote", [2, 1, 1], counts(4, 3, 1)),
                ("promote_no_flip", [2, 1, 0], counts(4, 3, 1)),
                ("genotyper+promote", [2, 1, 1], counts(3, 2, 2)),
            ];
            assert_eq!(got, expected);
            // Against the PASS baseline's F1 (0.6), not the genotyper's.
            assert!((metrics[0].f1_gain - (8.0 / 12.0 - 0.6)).abs() < 1e-12);
            assert!(metrics[2].f1_gain.abs() < 1e-12);
            // The genotyper's PASS columns are kept.
            assert_eq!((metrics[2].dropped, metrics[2].flipped), (2, 2));
            assert_eq!((metrics[1].t_promote, metrics[1].t_promote_flip), (0.5, NEVER));
            // A withheld promotion is not made.
            let withheld = promoted(Some(&[false, true, false]));
            assert_eq!((withheld[0].promoted, withheld[0].counts), (1, counts(4, 2, 1)));
            // Without a baseline there is nothing to add promotions to.
            let message =
                evaluate_promote_model(&candidates, &p, 0.5, 0.5, &pass[1..], false, None)
                    .unwrap_err()
                    .to_string();
            assert_eq!(message, "the PASS-call outcomes have no baseline");
        }

        #[test]
        fn metrics_are_written_as_tsv() {
            let calls = SampleCalls::from_table(&table(), VariantType::Snp, None, false).unwrap();
            let metrics = evaluate_model(
                &calls,
                &genotype_probabilities(),
                "snp",
                |_| THRESHOLDS,
                false,
                EvaluationPolicy::default(),
            )
            .unwrap();
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir.path().join("metrics.tsv");
            write_policy_metrics(&path, &metrics).unwrap();
            let text = std::fs::read_to_string(&path).unwrap();
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(lines.len(), 1 + metrics.len());
            assert_eq!(lines[0], POLICY_METRIC_COLUMNS.join("\t"));
            assert_eq!(
                lines[3],
                "S1\tsnp\tgenotyper\tgenotyper\t0.5\t0.5\t2\t1\t2\t1\t1\t2\t1\t3\t0.5\t-0.1\t\
                 1.01\t1.01\t0\t0\t0\t1.01\t0\t0\t0"
            );
        }
    }
}
