"""Train a model and choose its thresholds by leave-one-group-out cross-validation."""

import logging
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Final
from typing import Protocol

import pandas as pd
import xgboost as xgb

from winnowr.common import FloatArray
from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.model import ContigLengths
from winnowr.model import ModelInfo
from winnowr.model import TrainingParams
from winnowr.model import WinnowrModel
from winnowr.model import fit_booster
from winnowr.model import num_classes
from winnowr.model import predict_probabilities
from winnowr.rescore import Mode
from winnowr.rescore import RepeatFlip
from winnowr.rescore import has_repeat_info
from winnowr.samples import PromoteSampleCalls
from winnowr.samples import SampleCalls
from winnowr.stutter import StutterModel
from winnowr.thresholds import DEFAULT_GRID
from winnowr.thresholds import DEFAULT_PROMOTE_FLIP_GRID
from winnowr.thresholds import DEFAULT_PROMOTE_GRID
from winnowr.thresholds import search_promote_thresholds
from winnowr.thresholds import search_repeat_flip
from winnowr.thresholds import search_thresholds

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = TrainingParams()


LABEL_COLUMNS: Final[dict[Task, str]] = {Task.GENOTYPE: "label", Task.PROMOTE: "promote_label"}
"""The label column of each task's training calls."""


class CvSample[FoldT](Protocol):
    """The calls of one training sample (`SampleCalls` or `PromoteSampleCalls`)."""

    @property
    def sample(self) -> str:
        """The sample name."""
        ...

    @property
    def calls(self) -> pd.DataFrame:
        """The labelled training calls."""
        ...

    def to_fold(self, probabilities: FloatArray) -> FoldT:
        """
        Package held-out predictions for the calls as a cross-validation fold.

        Args:
            probabilities: the class probabilities of the calls.

        Returns:
            The fold.
        """
        ...


def _fit(
    samples: Sequence[CvSample[object]],
    features: Sequence[str],
    params: TrainingParams,
    task: Task,
    regenotype: bool = False,
) -> xgb.Booster:
    """Fit a booster on the pooled calls of the samples."""
    calls = pd.concat([s.calls for s in samples], ignore_index=True)
    num_class = num_classes(task, regenotype)
    return fit_booster(calls, calls[LABEL_COLUMNS[task]], features, params, num_class)


def group_samples(
    names: Sequence[str], groups: Iterable[Iterable[str]] = ()
) -> list[tuple[str, ...]]:
    """
    Partition samples into cross-validation groups that are held out together.

    A sample not named in any group is a group of its own, so with no groups every sample is held
    out alone (leave-one-sample-out). Use groups for related samples (e.g. a trio), which would
    otherwise leak into each other's training folds.

    Args:
        names: the training sample names.
        groups: the sample groups, each a collection of sample names.

    Returns:
        The groups, ordered by their first sample in `names`, each listing its samples in `names`
        order.

    Raises:
        ValueError: if a sample name is repeated, a group is empty or names an unknown sample, or a
            sample is in more than one group.
    """
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Each training sample must be given once; repeated: {repeated}")
    known = set(names)
    group_of: dict[str, int] = {}
    for index, group in enumerate(groups):
        members = list(group)
        if not members:
            raise ValueError("Sample groups must not be empty")
        unknown = sorted(set(members) - known)
        if unknown:
            raise ValueError(
                f"Sample group names unknown samples: {unknown}; known: {sorted(known)}"
            )
        for name in members:
            if group_of.setdefault(name, index) != index:
                raise ValueError(f"Sample {name!r} is in more than one sample group")
    keyed: dict[str | int, list[str]] = {}
    for name in names:
        keyed.setdefault(group_of.get(name, name), []).append(name)
    return [tuple(members) for members in keyed.values()]


def cross_validate[FoldT](
    samples: Sequence[CvSample[FoldT]],
    features: Sequence[str],
    params: TrainingParams,
    task: Task = Task.GENOTYPE,
    regenotype: bool = False,
    groups: Iterable[Iterable[str]] = (),
    refit: Callable[[frozenset[str]], Sequence[CvSample[FoldT]]] | None = None,
) -> list[FoldT]:
    """
    Leave-one-group-out predictions: fit on all other groups, predict each held-out sample.

    Samples are grouped by `group_samples`; with no groups this is leave-one-sample-out. Samples
    are told apart by name, so a sample given twice is an error rather than leaking into its own
    training fold. (Related samples that are not grouped still leak.)

    Args:
        samples: the training samples (with distinct names, in at least two groups).
        features: the model features.
        params: the training parameters.
        task: the task, which picks the label column and the number of classes.
        regenotype: whether a genotype model has the re-genotype class.
        groups: sample groups to hold out together (see `group_samples`).
        refit: rebuilds every sample, in `samples` order, for one fold from the names of the
            fold's training samples, when the calls depend on something fitted to the training
            samples (e.g. the stutter model); otherwise `samples` are used in every fold.

    Returns:
        One fold per sample, in input order.

    Raises:
        ValueError: if a sample name is repeated, the groups are invalid, there are fewer than
            two groups, or `refit` returns other samples.
    """
    names = [s.sample for s in samples]
    partition = group_samples(names, groups)
    if len(partition) < 2:
        raise ValueError(
            "Leave-one-group-out cross-validation needs at least two samples in at least two "
            f"groups; groups: {partition}"
        )
    num_class = num_classes(task, regenotype)
    fold_of: dict[str, FoldT] = {}
    for held_out in partition:
        fold_samples = samples
        if refit is not None:
            fold_samples = refit(frozenset(names) - set(held_out))
            if [s.sample for s in fold_samples] != names:
                raise ValueError(f"refit must return the samples {names} in order")
        others = [s for s in fold_samples if s.sample not in held_out]
        booster = _fit(others, features, params, task, regenotype)
        for s in fold_samples:
            if s.sample in held_out:
                probabilities = predict_probabilities(booster, s.calls, features, num_class)
                fold_of[s.sample] = s.to_fold(probabilities)
        logger.info(f"CV: fitted {[s.sample for s in others]} -> scored {list(held_out)}")
    return [fold_of[name] for name in names]


def train_model(
    samples: Sequence[SampleCalls],
    variant_type: VariantType,
    features: Sequence[str],
    region_names: Sequence[str],
    contigs: Sequence[str],
    params: TrainingParams = DEFAULT_PARAMS,
    grid: Sequence[float] = DEFAULT_GRID,
    stutter: StutterModel | None = None,
    regenotype: bool = False,
    groups: Iterable[Iterable[str]] = (),
    refit: Callable[[frozenset[str]], Sequence[SampleCalls]] | None = None,
    reference_contigs: ContigLengths | None = None,
    repeat_flip: bool = False,
) -> WinnowrModel:
    """
    Choose thresholds by leave-one-group-out CV, then fit the final model on all samples.

    Args:
        samples: the training samples (at least two).
        variant_type: the variant type.
        features: the model features.
        region_names: the region set names used by the features.
        contigs: the contigs the calls came from.
        params: the training parameters.
        grid: candidate thresholds.
        stutter: the stutter model the samples' length-call features were computed with (fitted on
            all the samples), saved with the model.
        regenotype: train the re-genotype class (the samples must be relabelled with it, see
            `SampleCalls.from_table`); its threshold is searched with the genotyper's.
        groups: sample groups to hold out together in CV (see `group_samples`).
        refit: rebuilds the samples for each CV fold (see `cross_validate`), e.g. with a stutter
            model fitted on the fold's training samples only, so that held-out samples' features
            and labels do not depend on their own truth.
        reference_contigs: the lengths of `contigs` in the training samples' reference.
        repeat_flip: also choose by CV a separate genotyper flip threshold for calls in long
            tandem repeats (see `search_repeat_flip`).

    Returns:
        The trained model with CV-chosen thresholds for every mode.

    Raises:
        ValueError: with `repeat_flip`, if no training call has a repeat period and count (e.g. the
            VCFs lack DRAGstrInfo), so no stratum could select any call.
    """
    if repeat_flip and not any(
        has_repeat_info(s.calls.get("str_period"), s.calls.get("str_repeats")) for s in samples
    ):
        raise ValueError(
            "--repeat-flip needs calls with a DRAGstr repeat period and count (the str_period and "
            "str_repeats columns, from the VCFs' DRAGstrInfo), but no training call has them"
        )
    folds = cross_validate(
        samples, features, params, regenotype=regenotype, groups=groups, refit=refit
    )
    thresholds = {
        mode: search_thresholds(folds, mode, grid, regenotype=regenotype) for mode in Mode
    }
    if repeat_flip:
        thresholds[Mode.GENOTYPER] = search_repeat_flip(folds, thresholds[Mode.GENOTYPER], grid)
    for mode, chosen in thresholds.items():
        regenotype_text = (
            f" t_regenotype={chosen.t_regenotype:.10g}"
            if regenotype and mode is Mode.GENOTYPER
            else ""
        )
        repeat_text = _repeat_flip_text(chosen.repeat_flip, repeat_flip and mode is Mode.GENOTYPER)
        logger.info(
            f"CV-chosen {mode.value}: t_drop={chosen.t_drop:.10g} t_flip={chosen.t_flip:.10g}"
            f"{regenotype_text}{repeat_text} "
            f"(summed CV F1 gain {chosen.gain:+.5f})"
        )
    booster = _fit(samples, features, params, Task.GENOTYPE, regenotype)
    info = ModelInfo(
        variant_type=variant_type,
        features=tuple(features),
        region_names=tuple(region_names),
        thresholds=thresholds,
        training_samples=tuple(s.sample for s in samples),
        contigs=tuple(contigs),
        params=params,
        stutter=stutter,
        regenotype=regenotype,
        reference_contigs=reference_contigs,
        repeat_flip_searched=repeat_flip,
    )
    return WinnowrModel(booster=booster, info=info)


def _repeat_flip_text(chosen: RepeatFlip | None, searched: bool) -> str:
    """The log text of a CV-chosen repeat-stratum flip threshold."""
    if chosen is not None:
        return (
            f" t_flip_repeat={chosen.t_flip:.10g} (repeat period >= {chosen.min_period} bp, "
            f">= {chosen.min_units} units)"
        )
    return " (no repeat-stratum flip threshold kept)" if searched else ""


def train_promote_model(
    samples: Sequence[PromoteSampleCalls],
    variant_type: VariantType,
    features: Sequence[str],
    region_names: Sequence[str],
    contigs: Sequence[str],
    params: TrainingParams = DEFAULT_PARAMS,
    promote_grid: Sequence[float] = DEFAULT_PROMOTE_GRID,
    flip_grid: Sequence[float] = DEFAULT_PROMOTE_FLIP_GRID,
    groups: Iterable[Iterable[str]] = (),
    reference_contigs: ContigLengths | None = None,
) -> WinnowrModel:
    """
    Train a promote model: thresholds by leave-one-group-out CV, then a final fit.

    Args:
        samples: the training samples (at least two).
        variant_type: the variant type.
        features: the model features.
        region_names: the region set names used by the features.
        contigs: the contigs the calls came from.
        params: the training parameters.
        promote_grid: candidate promotion thresholds.
        flip_grid: candidate flip thresholds.
        groups: sample groups to hold out together in CV (see `group_samples`).
        reference_contigs: the lengths of `contigs` in the training samples' reference.

    Returns:
        The trained promote model.

    Raises:
        ValueError: if a sample name is repeated, the groups are invalid, or there are fewer than
            two groups.
    """
    folds = cross_validate(samples, features, params, Task.PROMOTE, groups=groups)
    chosen = search_promote_thresholds(folds, promote_grid, flip_grid)
    logger.info(
        f"CV-chosen promote: t_promote={chosen.t_promote:.10g} t_flip={chosen.t_flip:.10g} "
        f"(summed CV F1 gain {chosen.gain:+.5f})"
    )
    info = ModelInfo(
        variant_type=variant_type,
        features=tuple(features),
        region_names=tuple(region_names),
        thresholds={},
        training_samples=tuple(s.sample for s in samples),
        contigs=tuple(contigs),
        task=Task.PROMOTE,
        promote_thresholds=chosen,
        params=params,
        reference_contigs=reference_contigs,
    )
    booster = _fit(samples, features, params, Task.PROMOTE)
    return WinnowrModel(booster=booster, info=info)
