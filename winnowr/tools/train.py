import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.features import feature_names
from winnowr.model import ContigLengths
from winnowr.model import TrainingParams
from winnowr.repeat_features import STRLEN_FEATURES
from winnowr.samples import PromoteSampleCalls
from winnowr.samples import SampleCalls
from winnowr.stutter import LENGTH_CALL_FEATURES
from winnowr.stutter import StutterModel
from winnowr.stutter import fit_stutter_from_feature_tables
from winnowr.table import FeatureTableInfo
from winnowr.table import read_feature_table
from winnowr.thresholds import DEFAULT_GRID
from winnowr.thresholds import DEFAULT_PROMOTE_FLIP_GRID
from winnowr.thresholds import DEFAULT_PROMOTE_GRID
from winnowr.training import group_samples
from winnowr.training import train_model
from winnowr.training import train_promote_model

logger = logging.getLogger(__name__)


def _reference_contigs(infos: Sequence[FeatureTableInfo]) -> ContigLengths | None:
    """
    The reference contigs the training tables share, to record in the sidecar.

    Args:
        infos: the training tables' metadata.

    Returns:
        The shared contig lengths, or None (with a warning) if a table does not record them.

    Raises:
        ValueError: if two tables record different contig lengths (different references).
    """
    recorded = [info for info in infos if info.reference_contigs is not None]
    if len(recorded) < len(infos):
        logger.warning(
            "Some --tables do not record their reference contigs (extracted by an older winnowr); "
            "the model will not record them either, so score cannot check the reference"
        )
        return None
    first = recorded[0]
    for info in recorded[1:]:
        if info.reference_contigs != first.reference_contigs:
            raise ValueError(
                f"--tables come from different references: {first.sample} has contig lengths "
                f"{first.reference_contigs}, {info.sample} has {info.reference_contigs}"
            )
    return first.reference_contigs


def _read_training_tables(
    tables: Sequence[Path],
) -> tuple[list[tuple[pd.DataFrame, FeatureTableInfo]], tuple[str, ...], tuple[str, ...]]:
    """
    Read the training feature tables and check that they can be trained on together.

    Args:
        tables: the `--tables` paths.

    Returns:
        The tables with their metadata, and their shared region names and contigs.

    Raises:
        ValueError: if there are fewer than two tables, a file is listed twice, or the tables were
            extracted with different regions or contigs.
    """
    if len(tables) < 2:
        raise ValueError("--tables needs at least two feature tables (one per training sample)")
    resolved = [path.resolve() for path in tables]
    repeated = sorted({str(path) for path in resolved if resolved.count(path) > 1})
    if repeated:
        raise ValueError(f"--tables lists the same file more than once: {repeated}")
    loaded = [read_feature_table(path) for path in tables]
    region_names = {info.region_names for _, info in loaded}
    if len(region_names) != 1:
        raise ValueError(f"--tables were extracted with different regions: {region_names}")
    contigs = {info.contigs for _, info in loaded}
    if len(contigs) != 1:
        raise ValueError(f"--tables were extracted from different contigs: {contigs}")
    return loaded, next(iter(region_names)), next(iter(contigs))


def train(
    *,
    tables: list[Path],
    variant_type: VariantType,
    output: Path,
    task: Task = Task.GENOTYPE,
    neighbourhood: bool = True,
    repeat_length: bool = False,
    regenotype: bool = False,
    repeat_flip: bool = False,
    sample_groups: list[str] | None = None,
    grid: list[float] | None = None,
    learning_rate: float = TrainingParams.learning_rate,
    max_leaves: int = TrainingParams.max_leaves,
    max_depth: int = TrainingParams.max_depth,
    min_child_weight: float = TrainingParams.min_child_weight,
    reg_lambda: float = TrainingParams.reg_lambda,
    max_rounds: int = TrainingParams.max_rounds,
    early_stopping_rounds: int = TrainingParams.early_stopping_rounds,
    validation_fraction: float = TrainingParams.validation_fraction,
    seed: int = TrainingParams.seed,
    threads: int = TrainingParams.threads,
) -> None:
    """
    Train a model for one variant type; thresholds are chosen by leave-one-group-out CV.

    Writes <output>.xgb.json (XGBoost model) and <output>.winnowr.json (sidecar).

    Args:
        tables: Labelled feature tables (from `extract --happy`), one per training sample; at
            least two, of distinct samples.
        variant_type: The variant type to train for.
        output: Output path prefix.
        task: GENOTYPE trains on PASS calls (drop/keep/flip, needs hap.py labels); PROMOTE trains on
            non-PASS calls (stay/promote/promote+flip, needs tables extracted with --truth-vcf).
        neighbourhood: Use the multi-site neighbourhood features (--no-neighbourhood drops them).
        repeat_length: Use the repeat-length features (INDEL GENOTYPE models; tables extracted
            with `extract --strlen`): fits a stutter model on the training samples'
            truth-homozygous calls, saved in the sidecar, and adds read-length histogram and
            length-caller features. Each CV fold refits the stutter model on its training samples
            only.
        regenotype: Add the re-genotype class (implies --repeat-length): calls whose genotype a
            het<->hom flip does not fix but the length caller's genotype does; its threshold is
            chosen by CV with the genotyper's.
        repeat_flip: Experimental and opt-in. Also choose by CV a separate genotyper flip threshold
            (higher or lower than t_flip) for calls in long tandem repeats (INDEL GENOTYPE models;
            needs DRAGstrInfo in the VCFs): the stratum (minimum DRAGstr repeat period and count)
            and its threshold are searched after the other thresholds, and kept only if nested
            selection (chosen on the other folds, scored on each held-out fold) also raises F1.
            Opt-in because it adds a CV-chosen parameter whose benefit has been small; models
            trained with it are written as sidecar format 3, others as format 2.
        sample_groups: Samples to hold out together in CV, one comma-separated group per value
            (e.g. a trio: HG005,HG006,HG007). Samples not in a group are held out alone, so by
            default each sample is its own fold (leave-one-sample-out).
        grid: Candidate thresholds: t_drop and t_flip for GENOTYPE, t_promote and t_flip for
            PROMOTE. "Never act" is always added (default: built-in grids from 0.2 up).
        learning_rate: XGBoost learning rate (eta).
        max_leaves: Maximum leaves per tree.
        max_depth: Maximum tree depth (0 = unlimited, bounded by --max-leaves).
        min_child_weight: Minimum hessian sum per leaf.
        reg_lambda: L2 regularization of leaf weights.
        max_rounds: Maximum boosting rounds.
        early_stopping_rounds: Early-stopping patience (rounds) on the validation split.
        validation_fraction: Fraction of training calls held out for early stopping.
        seed: Random seed.
        threads: XGBoost threads (0 = all cores).
    """
    loaded, regions, contigs = _read_training_tables(tables)
    reference_contigs = _reference_contigs([info for _, info in loaded])
    groups = [group.split(",") for group in sample_groups or []]
    group_samples([info.sample for _, info in loaded], groups)  # fail before any fitting
    use_strlen = repeat_length or regenotype
    if use_strlen and (variant_type is not VariantType.INDEL or task is not Task.GENOTYPE):
        raise ValueError(
            "Repeat-length features (--repeat-length, or --regenotype) need --variant-type INDEL "
            f"and --task GENOTYPE, got --variant-type {variant_type.name} --task {task.name}"
        )
    if repeat_flip and (variant_type is not VariantType.INDEL or task is not Task.GENOTYPE):
        raise ValueError(
            "--repeat-flip needs --variant-type INDEL and --task GENOTYPE, got --variant-type "
            f"{variant_type.name} --task {task.name}"
        )
    stutter = fit_stutter_from_feature_tables(loaded) if use_strlen else None
    repeat_features = (*STRLEN_FEATURES, *LENGTH_CALL_FEATURES) if use_strlen else ()
    features = feature_names(variant_type, regions, neighbourhood, task, repeat_features)
    params = TrainingParams(
        learning_rate=learning_rate,
        max_leaves=max_leaves,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=validation_fraction,
        seed=seed,
        threads=threads,
    )
    if task is Task.PROMOTE:
        promote_samples = [
            PromoteSampleCalls.from_table(frame, info, variant_type) for frame, info in loaded
        ]
        for p in promote_samples:
            counts = p.calls["promote_label"].value_counts().sort_index().to_dict()
            logger.info(f"{p.sample} {variant_type}: {len(p.calls):,} candidates, labels {counts}")
        model = train_promote_model(
            promote_samples,
            variant_type,
            features,
            regions,
            contigs,
            params,
            promote_grid=DEFAULT_PROMOTE_GRID if grid is None else grid,
            flip_grid=DEFAULT_PROMOTE_FLIP_GRID if grid is None else grid,
            groups=groups,
            reference_contigs=reference_contigs,
        )
    else:

        def build(stutter_model: StutterModel | None) -> list[SampleCalls]:
            return [
                SampleCalls.from_table(frame, info, variant_type, stutter_model, regenotype)
                for frame, info in loaded
            ]

        def refit(training: frozenset[str]) -> list[SampleCalls]:
            fold_tables = [(frame, info) for frame, info in loaded if info.sample in training]
            return build(fit_stutter_from_feature_tables(fold_tables))

        samples = build(stutter)
        for s in samples:
            logger.info(
                f"{s.sample} {variant_type}: {len(s.calls):,} calls TP={s.baseline.tp} "
                f"FP={s.baseline.fp} FN={s.baseline.fn} F1={s.baseline.f1:.5f}"
            )
            if regenotype:
                counts = s.calls["label"].value_counts().sort_index().to_dict()
                logger.info(f"{s.sample} {variant_type}: labels {counts}")
        model = train_model(
            samples,
            variant_type,
            features,
            regions,
            contigs,
            params,
            grid=DEFAULT_GRID if grid is None else grid,
            stutter=stutter,
            regenotype=regenotype,
            groups=groups,
            refit=refit if use_strlen else None,
            reference_contigs=reference_contigs,
            repeat_flip=repeat_flip,
        )
    sidecar = model.save(output)
    logger.info(f"Wrote model sidecar {sidecar}")
