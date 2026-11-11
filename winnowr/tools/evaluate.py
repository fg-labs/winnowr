import logging
from pathlib import Path

from winnowr.common import NEVER
from winnowr.common import Task
from winnowr.evaluation import PolicyMetric
from winnowr.evaluation import baseline_metric
from winnowr.evaluation import evaluate_model
from winnowr.evaluation import evaluate_promote_model
from winnowr.model import WinnowrModel
from winnowr.model import check_reference
from winnowr.samples import PromoteSampleCalls
from winnowr.samples import SampleCalls
from winnowr.table import read_feature_table

logger = logging.getLogger(__name__)


def evaluate(
    *,
    models: list[Path],
    table: Path,
    output: Path,
    tuned: bool = False,
) -> None:
    """
    Report F1 for models' decisions on a labelled sample.

    Rescoring uses vcfeval arithmetic on the sample's hap.py labels. For a genotype model, writes
    one row per policy: baseline, veto and genotyper at the model's CV-chosen thresholds, and
    flip_only. A promote model adds promote, promote_no_flip and (with a genotype model)
    genotyper+promote. Promote rows report the promote thresholds and flips in their own columns
    (t_promote, t_promote_flip, promoted_flipped). A re-genotype model adds
    genotyper_no_regenotype and reports its re-genotypes (t_regenotype, regenotyped, ...). A model
    with a long-tandem-repeat flip threshold (train --repeat-flip) adds genotyper_no_repeat_flip
    (t_flip for every call) and reports the stratum and its threshold (repeat_min_period,
    repeat_min_units, t_flip_repeat; repeat_min_period is 0 without one).

    Args:
        models: Model sidecars (<prefix>.winnowr.json) of one variant type: a genotype model, a
            promote model, or one of each.
        table: A labelled feature table (from `extract --happy`, plus --truth-vcf for promote
            models) for a held-out sample (a training sample of the models gives optimistic
            metrics, with a warning).
        output: Output metrics file (TSV).
        tuned: Also report thresholds tuned on this sample (optimistic; they peek at the labels).
            For a model trained with --repeat-flip, genotyper_tuned also tunes the long-repeat
            stratum and its flip threshold, after the other thresholds as in training.
    """
    if not models:
        raise ValueError("--models needs at least one model sidecar")
    loaded = [WinnowrModel.load(path) for path in models]
    genotype = [m for m in loaded if m.info.task is Task.GENOTYPE]
    promote = [m for m in loaded if m.info.task is Task.PROMOTE]
    variant_types = {m.info.variant_type for m in loaded}
    if len(genotype) > 1 or len(promote) > 1 or len(variant_types) != 1:
        described = ", ".join(
            f"{path} ({m.info.task.value} {m.info.variant_type})"
            for path, m in zip(models, loaded, strict=True)
        )
        raise ValueError(
            "--models takes at most one genotype and one promote model of the same variant "
            f"type, got: {described}"
        )
    variant_type = loaded[0].info.variant_type
    frame, info = read_feature_table(table)
    if info.reference_contigs is None:
        logger.warning(
            f"{table} does not record its reference contigs (extracted by an older winnowr), so "
            "it cannot be checked against the models' reference"
        )
    else:
        for path, model in zip(models, loaded, strict=True):
            check_reference(
                path,
                model.info,
                info.reference_contigs,
                f"Feature table {table}",
                require_all=False,
            )
    trained_on = [
        str(p)
        for p, m in zip(models, loaded, strict=True)
        if info.sample in m.info.training_samples
    ]
    if trained_on:
        logger.warning(
            f"{info.sample} is a training sample of {', '.join(trained_on)}: its metrics are "
            "in-sample and optimistic, not held-out"
        )
    stutter = genotype[0].info.stutter if genotype else None
    regenotype = genotype[0].info.regenotype if genotype else False
    calls = SampleCalls.from_table(frame, info, variant_type, stutter, regenotype)
    if genotype:
        probabilities = genotype[0].predict(calls.calls)
        metrics = evaluate_model(
            calls,
            probabilities,
            variant_type.value,
            genotype[0].info.thresholds,
            tuned,
            tune_repeat_flip=genotype[0].info.repeat_flip_searched,
        )
    else:
        metrics = [baseline_metric(calls, variant_type.value)]
    if promote:
        chosen = promote[0].info.promote_thresholds
        assert chosen is not None  # ModelInfo guarantees promote thresholds for promote models
        candidates = PromoteSampleCalls.from_table(frame, info, variant_type)
        metrics.extend(
            evaluate_promote_model(
                candidates, promote[0].predict(candidates.calls), chosen, metrics, tuned
            )
        )
    for m in metrics:
        promoted = (
            f" t_promote={m.t_promote:<6.4g} t_promote_flip={m.t_promote_flip:<6.4g} "
            f"promoted={m.promoted} (correct {m.promoted_correct}, flipped {m.promoted_flipped})"
            if m.t_promote < NEVER
            else ""
        )
        regenotyped = (
            f" t_regenotype={m.t_regenotype:<6.4g} regenotyped={m.regenotyped} "
            f"(fixed {m.regenotypes_fixed}, broke {m.regenotypes_broke_tp})"
            if regenotype
            else ""
        )
        repeat = (
            f" (repeat >= {m.repeat_min_period} bp x {m.repeat_min_units}: "
            f"t_flip={m.t_flip_repeat:.8g})"
            if m.repeat_min_period
            else ""
        )
        logger.info(
            f"{m.sample} {m.variant_type} {m.policy:24s} t_drop={m.t_drop:<10.8g} "
            f"t_flip={m.t_flip:<10.8g}{repeat} drop={m.dropped} (TP {m.dropped_tp}) "
            f"flip={m.flipped} (fixed {m.flips_fixed}, broke {m.flips_broke_tp}){regenotyped}"
            f"{promoted} TP={m.tp} FP={m.fp} FN={m.fn} F1={m.f1:.5f} ({m.f1_gain:+.5f})"
        )
    PolicyMetric.write(output, *metrics)
