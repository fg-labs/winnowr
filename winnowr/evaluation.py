"""F1 reports for a model's decisions on a labelled sample."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from fgpyo.util.metric import Metric

from winnowr.common import NEVER
from winnowr.common import FloatArray
from winnowr.labels import Label
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.rescore import decide_promote
from winnowr.rescore import has_repeat_info
from winnowr.rescore import rescore_promote
from winnowr.samples import PromoteSampleCalls
from winnowr.samples import SampleCalls
from winnowr.thresholds import FoldPredictions
from winnowr.thresholds import PromoteThresholds
from winnowr.thresholds import Thresholds
from winnowr.thresholds import search_promote_thresholds
from winnowr.thresholds import search_repeat_flip
from winnowr.thresholds import search_thresholds

logger = logging.getLogger(__name__)

_NO_ACTION = Thresholds(t_drop=NEVER, t_flip=NEVER, gain=0.0)
"""Thresholds that take no action."""


@dataclass(frozen=True)
class PolicyMetric(Metric["PolicyMetric"]):
    """
    The outcome of applying one decision policy to a labelled sample.

    Attributes:
        sample: the sample name.
        variant_type: the variant type.
        policy: the policy name (`baseline`, `veto`, `genotyper`, `genotyper_no_regenotype`,
            `genotyper_no_repeat_flip`, `flip_only`, `promote`, `promote_no_flip`,
            `genotyper+promote`, or `*_tuned` for thresholds tuned on this sample, which are
            optimistic).
        mode: the decision mode of the PASS-call policy.
        t_drop: the drop threshold of the PASS-call policy.
        t_flip: the flip threshold of the PASS-call policy.
        dropped: PASS calls dropped.
        dropped_tp: true-positive PASS calls dropped.
        flipped: PASS calls flipped.
        flips_fixed: PASS-call flips that produced the truth genotype.
        flips_broke_tp: flips of true-positive PASS calls.
        tp: TP after all actions.
        fp: FP after all actions.
        fn: FN after all actions.
        f1: F1 after all actions.
        f1_gain: F1 change versus baseline.
        t_promote: the promotion threshold (promote policies).
        t_promote_flip: the flip threshold for promoted calls (promote policies).
        promoted: non-PASS calls promoted.
        promoted_correct: promoted calls that turned a truth FN into a TP.
        promoted_flipped: promoted calls whose genotype was flipped.
        t_regenotype: the re-genotype threshold of the PASS-call policy (re-genotype models).
        regenotyped: PASS calls re-genotyped (not counting those re-genotyped to 0/0, which are
            dropped).
        regenotypes_fixed: re-genotyped PASS calls that got the truth genotype.
        regenotypes_broke_tp: re-genotyped true-positive PASS calls.
        repeat_min_period: the minimum repeat period of the long-tandem-repeat flip stratum (0
            without one).
        repeat_min_units: the minimum repeat count of the long-tandem-repeat flip stratum (0
            without one).
        t_flip_repeat: the flip threshold of the long-tandem-repeat stratum (`NEVER` without
            one, when `t_flip` applies to every call; a stratum can also have `NEVER`, so
            `repeat_min_period` = 0 is what marks "no stratum").
    """

    sample: str
    variant_type: str
    policy: str
    mode: str
    t_drop: float
    t_flip: float
    dropped: int
    dropped_tp: int
    flipped: int
    flips_fixed: int
    flips_broke_tp: int
    tp: int
    fp: int
    fn: int
    f1: float
    f1_gain: float
    t_promote: float = NEVER
    t_promote_flip: float = NEVER
    promoted: int = 0
    promoted_correct: int = 0
    promoted_flipped: int = 0
    t_regenotype: float = NEVER
    regenotyped: int = 0
    regenotypes_fixed: int = 0
    regenotypes_broke_tp: int = 0
    repeat_min_period: int = 0
    repeat_min_units: int = 0
    t_flip_repeat: float = NEVER


def evaluate_policy(
    calls: SampleCalls,
    probabilities: FloatArray,
    variant_type: str,
    policy: str,
    mode: Mode,
    thresholds: Thresholds,
) -> PolicyMetric:
    """
    Apply one policy to a sample's labelled calls and rescore.

    Args:
        calls: the sample's labelled calls.
        probabilities: the model's `(n, 3)` (or `(n, 4)`) probabilities for those calls.
        variant_type: the variant type (for the report).
        policy: the policy name (for the report).
        mode: the decision mode.
        thresholds: the thresholds (`gain` is ignored).

    Returns:
        The policy outcome.
    """
    repeat_flip = thresholds.repeat_flip
    fold = calls.to_fold(probabilities)
    drop, flip, regenotype = fold.decide_thresholds(thresholds, mode)
    counts = fold.rescore(drop, flip, regenotype)
    is_tp = fold.labels == Label.KEEP
    return PolicyMetric(
        sample=calls.sample,
        variant_type=variant_type,
        policy=policy,
        mode=mode.value,
        t_drop=thresholds.t_drop,
        t_flip=thresholds.t_flip,
        dropped=int(drop.sum()),
        dropped_tp=int((drop & is_tp).sum()),
        flipped=int(flip.sum()),
        flips_fixed=int((flip & fold.flip_ok).sum()),
        flips_broke_tp=int((flip & is_tp).sum()),
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        f1=counts.f1,
        f1_gain=counts.f1 - fold.baseline.f1,
        t_regenotype=thresholds.t_regenotype,
        regenotyped=int(regenotype.sum()),
        regenotypes_fixed=int((regenotype & ~is_tp & fold.regenotype_ok).sum()),
        regenotypes_broke_tp=int((regenotype & is_tp).sum()),
        repeat_min_period=0 if repeat_flip is None else repeat_flip.min_period,
        repeat_min_units=0 if repeat_flip is None else repeat_flip.min_units,
        t_flip_repeat=NEVER if repeat_flip is None else repeat_flip.t_flip,
    )


def baseline_metric(calls: SampleCalls, variant_type: str) -> PolicyMetric:
    """
    The outcome with no actions.

    Args:
        calls: the sample's labelled calls.
        variant_type: the variant type (for the report).

    Returns:
        The `baseline` policy outcome.
    """
    no_predictions = np.zeros((len(calls.calls), 3), dtype=np.float64)
    return evaluate_policy(
        calls, no_predictions, variant_type, "baseline", Mode.GENOTYPER, _NO_ACTION
    )


def evaluate_model(
    calls: SampleCalls,
    probabilities: FloatArray,
    variant_type: str,
    thresholds: dict[Mode, Thresholds],
    tuned: bool = False,
    tune_repeat_flip: bool = False,
) -> list[PolicyMetric]:
    """
    Report baseline, each mode at the given thresholds, flips alone, and optionally tuned.

    For a re-genotype model (4 class probabilities) also reports `genotyper_no_regenotype`: the
    genotyper thresholds with the re-genotype action off. When the genotyper has a
    long-tandem-repeat flip threshold, also reports `genotyper_no_repeat_flip`: the genotyper
    thresholds with `t_flip` for every call. If it has one, or `tune_repeat_flip` is set (the
    model was trained with `--repeat-flip`, whether or not it kept a stratum), `genotyper_tuned`
    also tunes the stratum and its threshold, after the other thresholds as in training.

    Args:
        calls: the sample's labelled calls.
        probabilities: the model's probabilities for those calls.
        variant_type: the variant type (for the report).
        thresholds: the thresholds per mode.
        tuned: also report thresholds tuned on this sample (optimistic: they see the test data).
        tune_repeat_flip: with `tuned`, also tune a long-tandem-repeat flip threshold, even if the
            genotyper has none.

    Returns:
        One metric per policy.
    """
    repeat_info = has_repeat_info(calls.calls.get("str_period"), calls.calls.get("str_repeats"))
    has_stratum = thresholds[Mode.GENOTYPER].repeat_flip is not None
    tune_stratum = tuned and (tune_repeat_flip or has_stratum)
    if (tune_stratum or has_stratum) and not repeat_info:
        logger.warning(
            f"{calls.sample}: no call has a DRAGstr repeat period and count (DRAGstrInfo), so the "
            "long-tandem-repeat flip threshold selects no call"
        )
    metrics = [baseline_metric(calls, variant_type)]
    for mode in (Mode.VETO, Mode.GENOTYPER):
        metrics.append(
            evaluate_policy(calls, probabilities, variant_type, mode.value, mode, thresholds[mode])
        )
    genotyper = thresholds[Mode.GENOTYPER]
    regenotype = probabilities.shape[1] > Label.REGENOTYPE
    if regenotype:
        metrics.append(
            evaluate_policy(
                calls,
                probabilities,
                variant_type,
                "genotyper_no_regenotype",
                Mode.GENOTYPER,
                replace(genotyper, t_regenotype=NEVER),
            )
        )
    if genotyper.repeat_flip is not None:
        metrics.append(
            evaluate_policy(
                calls,
                probabilities,
                variant_type,
                "genotyper_no_repeat_flip",
                Mode.GENOTYPER,
                replace(genotyper, repeat_flip=None),
            )
        )
    metrics.append(
        evaluate_policy(
            calls,
            probabilities,
            variant_type,
            "flip_only",
            Mode.GENOTYPER,
            replace(_NO_ACTION, t_flip=genotyper.t_flip, repeat_flip=genotyper.repeat_flip),
        )
    )
    if tuned:
        fold: FoldPredictions = calls.to_fold(probabilities)
        for mode in (Mode.VETO, Mode.GENOTYPER):
            best = search_thresholds([fold], mode, regenotype=regenotype)
            if mode is Mode.GENOTYPER and tune_stratum and repeat_info:
                # tune the stratum too (after the other thresholds, as in training); on one sample
                # there is no nested check, which is fine for an optimistic bound
                best = search_repeat_flip([fold], best, nested=False)
            metrics.append(
                evaluate_policy(
                    calls, probabilities, variant_type, f"{mode.value}_tuned", mode, best
                )
            )
    return metrics


def evaluate_promote_policy(
    candidates: PromoteSampleCalls,
    probabilities: FloatArray,
    policy: str,
    t_promote: float,
    t_flip: float,
    pass_metric: PolicyMetric,
) -> PolicyMetric:
    """
    Apply promotions on top of a PASS-call outcome and rescore.

    Args:
        candidates: the sample's labelled promotion candidates.
        probabilities: the promote model's probabilities for them.
        policy: the policy name (for the report).
        t_promote: the promotion threshold.
        t_flip: the flip threshold for promoted calls.
        pass_metric: the outcome on PASS calls the promotions add to (e.g. the baseline).

    Returns:
        The combined outcome: the PASS-call columns come from `pass_metric`, the promote columns
        from the promotions, and the counts and F1 gain (relative to the PASS-call baseline)
        include both.
    """
    fold = candidates.to_fold(probabilities)
    promote, flip = decide_promote(probabilities, fold.flippable, t_promote, t_flip)
    start = Counts(pass_metric.tp, pass_metric.fp, pass_metric.fn)
    counts = rescore_promote(fold.labels, promote, flip, start)
    return replace(
        pass_metric,
        policy=policy,
        t_promote=t_promote,
        t_promote_flip=t_flip,
        promoted=int(promote.sum()),
        promoted_correct=counts.tp - start.tp,
        promoted_flipped=int(flip.sum()),
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        f1=counts.f1,
        f1_gain=counts.f1 - candidates.baseline.f1,
    )


def evaluate_promote_model(
    candidates: PromoteSampleCalls,
    probabilities: FloatArray,
    thresholds: PromoteThresholds,
    pass_metrics: Sequence[PolicyMetric],
    tuned: bool = False,
) -> list[PolicyMetric]:
    """
    Report promotions on top of the PASS baseline and any other PASS-call outcomes.

    Args:
        candidates: the sample's labelled promotion candidates.
        probabilities: the promote model's probabilities for them.
        thresholds: the promote model's thresholds.
        pass_metrics: PASS-call outcomes to add promotions to; must include `baseline`.
        tuned: also report thresholds tuned on this sample (optimistic).

    Returns:
        One metric per policy.
    """
    by_policy = {m.policy: m for m in pass_metrics}
    baseline = by_policy["baseline"]
    t_promote, t_flip = thresholds.t_promote, thresholds.t_flip
    metrics = [
        evaluate_promote_policy(candidates, probabilities, "promote", t_promote, t_flip, baseline),
        evaluate_promote_policy(
            candidates, probabilities, "promote_no_flip", t_promote, NEVER, baseline
        ),
    ]
    if "genotyper" in by_policy:
        metrics.append(
            evaluate_promote_policy(
                candidates,
                probabilities,
                "genotyper+promote",
                t_promote,
                t_flip,
                by_policy["genotyper"],
            )
        )
    if tuned:
        best = search_promote_thresholds([candidates.to_fold(probabilities)])
        metrics.append(
            evaluate_promote_policy(
                candidates,
                probabilities,
                "promote_tuned",
                best.t_promote,
                best.t_flip,
                baseline,
            )
        )
    return metrics
