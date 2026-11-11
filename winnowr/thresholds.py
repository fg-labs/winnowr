"""
Choose decision thresholds by maximising the summed F1 gain over cross-validation folds.

Every search also considers `NEVER` (take no action), whose gain is exactly zero, so a chosen
threshold never has a negative summed gain. Among candidates with equal gain the most conservative
wins: the higher first threshold (`t_drop` / `t_promote`), then the higher `t_flip`. So when the
folds give no evidence for acting (every gain is zero), the search returns `NEVER`.

An optional second search (`search_repeat_flip`) gives calls in long tandem repeats their own flip
threshold (`RepeatFlip`), keeping the other thresholds. Because it picks among many strata on the
same folds, its in-sample gain is optimistic; it is kept only if a nested check (choose on the other
folds, score on the held-out one) also gains.
"""

import itertools
import logging
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Final

import numpy as np
import numpy.typing as npt

from winnowr.common import NEVER
from winnowr.common import BoolArray
from winnowr.common import FloatArray
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.rescore import RepeatFlip
from winnowr.rescore import decide
from winnowr.rescore import decide_promote
from winnowr.rescore import flip_thresholds
from winnowr.rescore import rescore
from winnowr.rescore import rescore_promote

logger = logging.getLogger(__name__)

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


def _argmax(
    gain_of: Callable[..., float],
    candidates: Sequence[Sequence[float]],
    floor: float = -np.inf,
) -> tuple[tuple[float, ...], float] | None:
    """
    The combination of candidates with the highest gain, if it beats `floor`.

    Args:
        gain_of: the summed gain of a combination (one value per candidate list, in order).
        candidates: the candidates for each value, each from most to least conservative.
        floor: the gain a combination must exceed.

    Returns:
        The best combination and its gain, or None if none exceeds `floor`; ties go to the first
        combination in `itertools.product` order, i.e. the most conservative.
    """
    best: tuple[tuple[float, ...], float] | None = None
    for combination in itertools.product(*candidates):
        gain = gain_of(*combination)
        if gain > (floor if best is None else best[1]):
            best = (combination, gain)
    return best


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
    best = _argmax(gain_of, [_candidates(grid) for grid in grids])
    assert best is not None  # with no floor, the first combination is always kept
    return best


DEFAULT_REPEAT_PERIODS: Final[tuple[int, ...]] = (1, 2, 3)
"""Candidate minimum repeat periods (bp) of the long-tandem-repeat flip stratum."""

DEFAULT_REPEAT_UNITS: Final[tuple[int, ...]] = (6, 8, 11, 15, 20)
"""Candidate minimum repeat counts (units) of the long-tandem-repeat flip stratum."""


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
        str_period: per call, the DRAGstr repeat period (NaN when unknown); None if not known.
        str_repeats: per call, the DRAGstr repeat count (NaN when unknown); None if not known.
    """

    probabilities: FloatArray
    labels: npt.NDArray[np.int64]
    flip_ok: BoolArray
    flippable: BoolArray
    baseline: Counts
    regenotypable: BoolArray
    regenotypes_to_ref: BoolArray
    regenotype_ok: BoolArray
    str_period: FloatArray | None = None
    str_repeats: FloatArray | None = None

    def decide(
        self,
        t_drop: float,
        t_flip: float,
        mode: Mode,
        t_regenotype: float = NEVER,
        repeat_flip: RepeatFlip | None = None,
    ) -> tuple[BoolArray, BoolArray, BoolArray]:
        """
        The actions the thresholds take on this fold (see `winnowr.rescore.decide`).

        Args:
            t_drop: the drop threshold.
            t_flip: the flip threshold.
            mode: the decision mode.
            t_regenotype: the re-genotype threshold.
            repeat_flip: the flip threshold of calls in long tandem repeats, if any.

        Returns:
            The drop, flip and re-genotype masks.
        """
        return decide(
            self.probabilities,
            self.flippable,
            t_drop,
            flip_thresholds(t_flip, repeat_flip, self.str_period, self.str_repeats),
            mode,
            t_regenotype,
            self.regenotypable,
            self.regenotypes_to_ref,
        )

    def decide_thresholds(
        self, thresholds: "Thresholds", mode: Mode
    ) -> tuple[BoolArray, BoolArray, BoolArray]:
        """
        The actions a set of thresholds takes on this fold, including its `repeat_flip`.

        Args:
            thresholds: the thresholds.
            mode: the decision mode.

        Returns:
            The drop, flip and re-genotype masks.
        """
        return self.decide(
            thresholds.t_drop,
            thresholds.t_flip,
            mode,
            thresholds.t_regenotype,
            thresholds.repeat_flip,
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
        self,
        t_drop: float,
        t_flip: float,
        mode: Mode,
        t_regenotype: float = NEVER,
        repeat_flip: RepeatFlip | None = None,
    ) -> float:
        """
        The F1 change from applying the thresholds to this fold.

        Args:
            t_drop: the drop threshold.
            t_flip: the flip threshold.
            mode: the decision mode.
            t_regenotype: the re-genotype threshold.
            repeat_flip: the flip threshold of calls in long tandem repeats, if any.

        Returns:
            The F1 after the actions minus the baseline F1.
        """
        actions = self.decide(t_drop, t_flip, mode, t_regenotype, repeat_flip)
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
        repeat_flip: a separate flip threshold for calls in long tandem repeats (genotyper mode
            only; see `search_repeat_flip`), or None to use `t_flip` for every call.
    """

    t_drop: float
    t_flip: float
    gain: float
    t_regenotype: float = NEVER
    repeat_flip: RepeatFlip | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Thresholds":
        """
        Build from the dictionary `dataclasses.asdict` gives.

        Args:
            raw: the thresholds' fields (`repeat_flip` may be missing, e.g. in older sidecars).

        Returns:
            The thresholds.

        Raises:
            TypeError: if a field is missing or unexpected (`ModelInfo.from_json` reports it as a
                malformed sidecar).
        """
        fields = dict(raw)
        repeat_flip = fields.pop("repeat_flip", None)
        return cls(**fields, repeat_flip=None if repeat_flip is None else RepeatFlip(**repeat_flip))


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


def _best_repeat_flip(
    folds: Sequence[FoldPredictions],
    chosen: Thresholds,
    grid: Sequence[float],
    periods: Sequence[int],
    units: Sequence[int],
) -> tuple[RepeatFlip, float] | None:
    """
    The stratum and threshold with the highest summed gain on `folds`, if it beats no stratum.

    Args:
        folds: the predictions to score candidates on.
        chosen: the other genotyper thresholds, kept fixed.
        grid: candidate flip thresholds for the stratum (`NEVER` is added).
        periods: candidate minimum repeat periods.
        units: candidate minimum repeat counts.

    Returns:
        The best `RepeatFlip` and its summed gain, or None if none beats `chosen` without one.
    """

    def gain_of(repeat_flip: RepeatFlip | None) -> float:
        return sum(
            fold.f1_gain(
                chosen.t_drop, chosen.t_flip, Mode.GENOTYPER, chosen.t_regenotype, repeat_flip
            )
            for fold in folds
        )

    best = _argmax(
        lambda min_units, min_period, t_flip: gain_of(
            RepeatFlip(min_period=int(min_period), min_units=int(min_units), t_flip=t_flip)
        ),
        [sorted(set(units), reverse=True), sorted(set(periods), reverse=True), _candidates(grid)],
        floor=gain_of(None),
    )
    if best is None:
        return None
    (min_units, min_period, t_flip), gain = best
    return RepeatFlip(min_period=int(min_period), min_units=int(min_units), t_flip=t_flip), gain


def search_repeat_flip(
    folds: Sequence[FoldPredictions],
    chosen: Thresholds,
    grid: Sequence[float] = DEFAULT_GRID,
    periods: Sequence[int] = DEFAULT_REPEAT_PERIODS,
    units: Sequence[int] = DEFAULT_REPEAT_UNITS,
    nested: bool = True,
) -> Thresholds:
    """
    Grid-search a separate flip threshold for calls in long tandem repeats.

    Keeping the genotyper thresholds `chosen` (from `search_thresholds`), each stratum (minimum
    repeat period x minimum repeat count) and flip threshold is scored by the summed F1 gain over
    the folds; ties go to the narrowest stratum (the higher minimum count, then the higher minimum
    period), then the higher threshold. The best candidate must beat no stratum.

    With many candidates scored on the same folds, that in-sample gain is optimistic: on data where
    repeats behave like other calls, some candidate beats no stratum by chance. So with `nested`,
    the candidate is also checked by nested selection: for each fold, the best candidate on the
    other folds (keeping the other thresholds at `chosen`, which all folds chose) is scored on it,
    and the stratum is kept only if those held-out gains are positive on more than half of the folds
    and sum to more than zero. On simulated folds where repeats behave like other calls, this keeps
    a stratum in about 1 in 8 trials (3,000 calls per fold) or 1 in 10 (50,000), against 9 in 10 for
    the in-sample gain alone. Without `nested` (e.g. to tune on a single sample), the in-sample gain
    decides.

    Args:
        folds: the held-out predictions, with the calls' repeat periods and counts.
        chosen: the genotyper thresholds, without a `repeat_flip`.
        grid: candidate flip thresholds for the stratum.
        periods: candidate minimum repeat periods.
        units: candidate minimum repeat counts.
        nested: keep the stratum only if nested selection also gains (needs two or more folds).

    Returns:
        `chosen`, with `repeat_flip` and its (in-sample) `gain` set if a stratum is kept.

    Raises:
        ValueError: if there are no folds (or fewer than two with `nested`), `chosen` already has a
            `repeat_flip`, or a fold lacks the calls' repeat periods or counts.
    """
    if not folds:
        raise ValueError("Threshold search needs at least one fold")
    if nested and len(folds) < 2:
        raise ValueError("Nested selection of a long-tandem-repeat flip threshold needs 2+ folds")
    if chosen.repeat_flip is not None:
        raise ValueError(f"Expected thresholds without a repeat_flip, got {chosen.repeat_flip}")
    best = _best_repeat_flip(folds, chosen, grid, periods, units)
    if best is None:
        return chosen
    repeat_flip, gain = best
    if nested:
        held_out = np.zeros(len(folds))
        for i, fold in enumerate(folds):
            inner = _best_repeat_flip([*folds[:i], *folds[i + 1 :]], chosen, grid, periods, units)
            if inner is not None:
                held_out[i] = fold.f1_gain(
                    chosen.t_drop, chosen.t_flip, Mode.GENOTYPER, chosen.t_regenotype, inner[0]
                ) - fold.f1_gain(chosen.t_drop, chosen.t_flip, Mode.GENOTYPER, chosen.t_regenotype)
        kept = held_out.sum() > 0 and 2 * int((held_out > 0).sum()) > len(folds)
        logger.info(
            f"Long-tandem-repeat flip threshold {repeat_flip}: summed CV F1 gain {gain:+.6f} "
            f"in-sample; nested held-out gains {', '.join(f'{g:+.6f}' for g in held_out)} "
            f"({'kept' if kept else 'not kept'})"
        )
        if not kept:
            return chosen
    return replace(chosen, repeat_flip=repeat_flip, gain=gain)


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
