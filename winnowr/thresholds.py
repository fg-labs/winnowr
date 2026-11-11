"""
Choose decision thresholds by maximising the summed F1 gain over cross-validation folds.

Every search also considers `NEVER` (take no action), whose gain is exactly zero, so a chosen
threshold never has a negative summed gain. Among candidates with equal gain the most conservative
wins: the higher first threshold (`t_drop` / `t_promote`), then the higher `t_flip`. So when the
folds give no evidence for acting (every gain is zero), the search returns `NEVER`.
"""

import itertools
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from winnowr.common import NEVER
from winnowr.common import BoolArray
from winnowr.common import FloatArray
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.rescore import decide
from winnowr.rescore import decide_promote
from winnowr.rescore import rescore
from winnowr.rescore import rescore_promote

DEFAULT_GRID: Final[tuple[float, ...]] = (
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
    1 - 1e-5,
    1 - 1e-6,
    1 - 1e-7,
    1 - 1e-8,
    1 - 1e-10,
    NEVER,
)
"""Candidate thresholds for t_drop and t_flip.

Regularized models are well calibrated, so CV often picks thresholds near 0.5; the grid extends
below 0.5 (a class need not be the argmax to be acted on) and up to near-certainty.
"""


def _candidates(grid: Iterable[float]) -> list[float]:
    """The grid plus `NEVER`, de-duplicated, from most to least conservative."""
    return sorted({*grid, NEVER}, reverse=True)


def _grid_argmax(
    gain_of: Callable[..., float], *grids: Iterable[float]
) -> tuple[tuple[float, ...], float]:
    """
    The threshold combination with the highest gain (see the module docstring).

    Args:
        gain_of: the summed gain of a combination (one threshold per grid, in grid order).
        grids: candidates for each threshold; `NEVER` is always added to each.

    Returns:
        The best combination and its gain; ties go to the most conservative, comparing the
        thresholds in grid order.
    """
    best: tuple[tuple[float, ...], float] | None = None
    for combination in itertools.product(*(_candidates(grid) for grid in grids)):
        gain = gain_of(*combination)
        if best is None or gain > best[1]:
            best = (combination, gain)
    assert best is not None  # the candidates always include NEVER
    return best


@dataclass(frozen=True)
class FoldPredictions:
    """
    Held-out predictions for one sample.

    Attributes:
        probabilities: `(n, 3)` class probabilities of the sample's labelled calls (`(n, 4)` for
            re-genotype models).
        labels: the `Label` of each call.
        flip_ok: per call, whether flipping it gives the truth genotype.
        flippable: per call, whether the genotype is `0/1` or `1/1`.
        baseline: the sample's counts before any action.
        regenotypable: per call, whether the re-genotype action can change it (all false without
            length calls).
        regenotypes_to_ref: per call, whether the length caller's genotype is `0/0`.
        regenotype_ok: per call, whether re-genotyping it gives the truth genotype.
    """

    probabilities: FloatArray
    labels: npt.NDArray[np.int64]
    flip_ok: BoolArray
    flippable: BoolArray
    baseline: Counts
    regenotypable: BoolArray
    regenotypes_to_ref: BoolArray
    regenotype_ok: BoolArray

    def decide(
        self, t_drop: float, t_flip: float, mode: Mode, t_regenotype: float = NEVER
    ) -> tuple[BoolArray, BoolArray, BoolArray]:
        """
        The actions the thresholds take on this fold (see `winnowr.rescore.decide`).

        Args:
            t_drop: the drop threshold.
            t_flip: the flip threshold.
            mode: the decision mode.
            t_regenotype: the re-genotype threshold.

        Returns:
            The drop, flip and re-genotype masks.
        """
        return decide(
            self.probabilities,
            self.flippable,
            t_drop,
            t_flip,
            mode,
            t_regenotype,
            self.regenotypable,
            self.regenotypes_to_ref,
        )

    def rescore(self, drop: BoolArray, flip: BoolArray, regenotype: BoolArray) -> Counts:
        """
        The counts after actions on this fold (see `winnowr.rescore.rescore`).

        Args:
            drop: calls to drop.
            flip: calls to flip.
            regenotype: calls to re-genotype.

        Returns:
            The counts after the actions.
        """
        return rescore(
            self.labels, self.flip_ok, drop, flip, self.baseline, regenotype, self.regenotype_ok
        )

    def f1_gain(
        self, t_drop: float, t_flip: float, mode: Mode, t_regenotype: float = NEVER
    ) -> float:
        """
        The F1 change from applying the thresholds to this fold.

        Args:
            t_drop: the drop threshold.
            t_flip: the flip threshold.
            mode: the decision mode.
            t_regenotype: the re-genotype threshold.

        Returns:
            The F1 after the actions minus the baseline F1.
        """
        actions = self.decide(t_drop, t_flip, mode, t_regenotype)
        return self.rescore(*actions).f1 - self.baseline.f1


@dataclass(frozen=True)
class Thresholds:
    """
    Decision thresholds.

    Attributes:
        t_drop: the drop threshold.
        t_flip: the flip threshold (`NEVER` in veto mode).
        gain: the summed F1 gain over the folds they were chosen on.
        t_regenotype: the re-genotype threshold (`NEVER` unless the model has the re-genotype
            class and the mode is genotyper).
    """

    t_drop: float
    t_flip: float
    gain: float
    t_regenotype: float = NEVER


def search_thresholds(
    folds: Sequence[FoldPredictions],
    mode: Mode,
    grid: Sequence[float] = DEFAULT_GRID,
    regenotype: bool = False,
) -> Thresholds:
    """
    Grid-search the thresholds that maximise the summed F1 gain over folds.

    `NEVER` is always a candidate, and ties go to the higher t_drop, then the higher t_flip, then
    the higher t_regenotype (see the module docstring).

    Args:
        folds: the held-out predictions.
        mode: the decision mode; in veto mode t_flip and t_regenotype are fixed to `NEVER`.
        grid: the candidate thresholds for t_drop, t_flip and t_regenotype.
        regenotype: also search t_regenotype (re-genotype models in genotyper mode).

    Returns:
        The best thresholds.

    Raises:
        ValueError: if there are no folds.
    """
    if not folds:
        raise ValueError("Threshold search needs at least one fold")
    veto = mode is Mode.VETO
    (t_drop, t_flip, t_regenotype), gain = _grid_argmax(
        lambda t_drop, t_flip, t_regenotype: sum(
            fold.f1_gain(t_drop, t_flip, mode, t_regenotype) for fold in folds
        ),
        grid,
        [NEVER] if veto else grid,
        grid if regenotype and not veto else [NEVER],
    )
    return Thresholds(t_drop=t_drop, t_flip=t_flip, gain=gain, t_regenotype=t_regenotype)


DEFAULT_PROMOTE_GRID: Final[tuple[float, ...]] = (
    0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, NEVER,
)  # fmt: skip
"""Candidate promotion thresholds."""

DEFAULT_PROMOTE_FLIP_GRID: Final[tuple[float, ...]] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, NEVER)
"""Candidate flip thresholds for promoted calls."""


@dataclass(frozen=True)
class PromoteFold:
    """
    Held-out promote-model predictions for one sample.

    Attributes:
        probabilities: `(n, 3)` class probabilities of the sample's promotion candidates.
        labels: the `PromoteLabel` of each candidate.
        flippable: per candidate, whether the genotype is `0/1` or `1/1`.
        baseline: the sample's PASS-call counts before any action.
    """

    probabilities: FloatArray
    labels: npt.NDArray[np.int64]
    flippable: BoolArray
    baseline: Counts

    def f1_gain(self, t_promote: float, t_flip: float) -> float:
        """
        The F1 change from promoting with the thresholds.

        Args:
            t_promote: the promotion threshold.
            t_flip: the flip threshold.

        Returns:
            The F1 after the promotions minus the baseline F1.
        """
        promote, flip = decide_promote(self.probabilities, self.flippable, t_promote, t_flip)
        return rescore_promote(self.labels, promote, flip, self.baseline).f1 - self.baseline.f1


@dataclass(frozen=True)
class PromoteThresholds:
    """
    Promote-model thresholds.

    Attributes:
        t_promote: the promotion threshold on P(promote) + P(promote and flip).
        t_flip: the flip threshold on P(promote and flip).
        gain: the summed F1 gain over the folds they were chosen on.
    """

    t_promote: float
    t_flip: float
    gain: float


def search_promote_thresholds(
    folds: Sequence[PromoteFold],
    promote_grid: Sequence[float] = DEFAULT_PROMOTE_GRID,
    flip_grid: Sequence[float] = DEFAULT_PROMOTE_FLIP_GRID,
) -> PromoteThresholds:
    """
    Grid-search the promote thresholds that maximise the summed F1 gain over folds.

    `NEVER` is always a candidate, and ties go to the higher t_promote, then the higher t_flip (see
    the module docstring).

    Args:
        folds: the held-out predictions.
        promote_grid: candidate promotion thresholds.
        flip_grid: candidate flip thresholds.

    Returns:
        The best thresholds.

    Raises:
        ValueError: if there are no folds.
    """
    if not folds:
        raise ValueError("Threshold search needs at least one fold")
    (t_promote, t_flip), gain = _grid_argmax(
        lambda t_promote, t_flip: sum(fold.f1_gain(t_promote, t_flip) for fold in folds),
        promote_grid,
        flip_grid,
    )
    return PromoteThresholds(t_promote=t_promote, t_flip=t_flip, gain=gain)
