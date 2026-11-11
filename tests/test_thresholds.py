from dataclasses import replace

import numpy as np
import pytest

from winnowr.common import NEVER
from winnowr.labels import Label
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.rescore import RepeatFlip
from winnowr.rescore import flip_thresholds
from winnowr.rescore import has_repeat_info
from winnowr.thresholds import FoldPredictions
from winnowr.thresholds import PromoteFold
from winnowr.thresholds import search_promote_thresholds
from winnowr.thresholds import search_repeat_flip
from winnowr.thresholds import search_thresholds


def _fold() -> FoldPredictions:
    # a confident 0/0 FP, a less confident 0/0 FP, a correct-flip FP, and a TP that looks flippy
    probabilities = np.array([
        [0.999, 0.001, 0.0],
        [0.95, 0.05, 0.0],
        [0.0, 0.05, 0.95],
        [0.0, 0.4, 0.6],
    ])
    labels = np.array([Label.DROP, Label.DROP, Label.FLIP, Label.KEEP])
    return FoldPredictions(
        probabilities=probabilities,
        labels=labels,
        flip_ok=np.array([False, False, True, False]),
        flippable=np.ones(4, dtype=np.bool_),
        baseline=Counts(tp=10, fp=3, fn=2),
        regenotypable=np.zeros(4, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(4, dtype=np.bool_),
        regenotype_ok=np.zeros(4, dtype=np.bool_),
    )


def test_genotyper_search_picks_the_best_thresholds() -> None:
    fold = _fold()
    best = search_thresholds([fold], Mode.GENOTYPER, grid=[0.5, 0.9, NEVER])
    # drop both 0/0 FPs (t_drop 0.5 ties 0.9; the more conservative wins), flip only the
    # confident flip (0.9)
    assert (best.t_drop, best.t_flip) == (0.9, 0.9)
    assert best.gain == pytest.approx(Counts(11, 0, 1).f1 - Counts(10, 3, 2).f1)


def test_veto_search_never_flips_and_sums_folds() -> None:
    fold = _fold()
    best = search_thresholds([fold, fold], Mode.VETO, grid=[0.5, 0.99, NEVER])
    assert best.t_flip == NEVER
    # t_drop=0.5 drops all four (incl. the TP); 0.99 drops only the most confident FP
    assert best.t_drop == 0.5
    one_fold_gain = Counts(9, 0, 3).f1 - Counts(10, 3, 2).f1
    assert best.gain == pytest.approx(2 * one_fold_gain)


def test_no_evidence_means_never() -> None:
    # no candidate threshold promotes the call (P(promote) + P(flip) = 0.3), so every gain is 0
    fold = PromoteFold(
        probabilities=np.array([[0.7, 0.25, 0.05]]),
        labels=np.array([0]),
        flippable=np.ones(1, dtype=np.bool_),
        baseline=Counts(tp=10, fp=0, fn=5),
    )
    best = search_promote_thresholds([fold], promote_grid=[0.5, 0.9], flip_grid=[0.5])
    assert (best.t_promote, best.t_flip, best.gain) == (NEVER, NEVER, 0.0)
    no_calls = FoldPredictions(
        probabilities=np.zeros((0, 3)),
        labels=np.zeros(0, dtype=np.int64),
        flip_ok=np.zeros(0, dtype=np.bool_),
        flippable=np.zeros(0, dtype=np.bool_),
        baseline=Counts(tp=10, fp=3, fn=2),
        regenotypable=np.zeros(0, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(0, dtype=np.bool_),
        regenotype_ok=np.zeros(0, dtype=np.bool_),
    )
    best_genotype = search_thresholds([no_calls], Mode.GENOTYPER)
    assert (best_genotype.t_drop, best_genotype.t_flip, best_genotype.gain) == (NEVER, NEVER, 0.0)


def test_never_is_always_a_candidate() -> None:
    # every action hurts: a grid without NEVER must still be able to choose "no action"
    fold = _fold()
    harmful = FoldPredictions(
        probabilities=fold.probabilities,
        labels=np.full(4, Label.KEEP),
        flip_ok=np.zeros(4, dtype=np.bool_),
        flippable=fold.flippable,
        baseline=Counts(tp=4, fp=0, fn=0),
        regenotypable=np.zeros(4, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(4, dtype=np.bool_),
        regenotype_ok=np.zeros(4, dtype=np.bool_),
    )
    best = search_thresholds([harmful], Mode.GENOTYPER, grid=[0.5, 0.9])
    assert (best.t_drop, best.t_flip, best.gain) == (NEVER, NEVER, 0.0)
    promote = PromoteFold(
        probabilities=np.array([[0.1, 0.8, 0.1]]),
        labels=np.array([0]),
        flippable=np.ones(1, dtype=np.bool_),
        baseline=Counts(tp=10, fp=0, fn=5),
    )
    best_promote = search_promote_thresholds([promote], promote_grid=[0.5], flip_grid=[0.5])
    assert (best_promote.t_promote, best_promote.gain) == (NEVER, 0.0)


def test_search_requires_folds() -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        search_thresholds([], Mode.GENOTYPER)


def test_default_grid_can_act_below_one_half() -> None:
    # every FP has P(drop) 0.35 and every TP 0.05: only a threshold below 0.5 helps
    labels = np.array([Label.DROP] * 3 + [Label.KEEP] * 3)
    probabilities = np.array([[0.35, 0.33, 0.32]] * 3 + [[0.05, 0.9, 0.05]] * 3)
    fold = FoldPredictions(
        probabilities=probabilities,
        labels=labels,
        flip_ok=np.zeros(6, dtype=np.bool_),
        flippable=np.ones(6, dtype=np.bool_),
        baseline=Counts(tp=3, fp=3, fn=0),
        regenotypable=np.zeros(6, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(6, dtype=np.bool_),
        regenotype_ok=np.zeros(6, dtype=np.bool_),
    )
    best = search_thresholds([fold], Mode.GENOTYPER)
    assert best.t_drop == 0.3 and best.gain > 0


def test_promote_search() -> None:
    # two correct promotions (one needs a flip) and one wrong candidate with lower confidence
    fold = PromoteFold(
        probabilities=np.array([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]]),
        labels=np.array([1, 2, 0]),
        flippable=np.ones(3, dtype=np.bool_),
        baseline=Counts(tp=10, fp=0, fn=5),
    )
    best = search_promote_thresholds([fold], promote_grid=[0.5, 0.7, NEVER], flip_grid=[0.5, NEVER])
    assert (best.t_promote, best.t_flip) == (0.7, 0.5)
    assert best.gain == pytest.approx(Counts(12, 0, 3).f1 - Counts(10, 0, 5).f1)
    with pytest.raises(ValueError, match="at least one fold"):
        search_promote_thresholds([])


def test_regenotype_search() -> None:
    # a confident correct re-genotype, a less confident one that breaks a TP
    probabilities = np.array([[0.0, 0.1, 0.0, 0.9], [0.0, 0.4, 0.0, 0.6]])
    fold = FoldPredictions(
        probabilities=probabilities,
        labels=np.array([Label.REGENOTYPE, Label.KEEP]),
        flip_ok=np.zeros(2, dtype=np.bool_),
        flippable=np.ones(2, dtype=np.bool_),
        baseline=Counts(tp=10, fp=1, fn=1),
        regenotypable=np.ones(2, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(2, dtype=np.bool_),
        regenotype_ok=np.array([True, False]),
    )
    best = search_thresholds([fold], Mode.GENOTYPER, grid=[0.5, 0.8], regenotype=True)
    assert (best.t_drop, best.t_flip, best.t_regenotype) == (NEVER, NEVER, 0.8)
    assert best.gain == pytest.approx(Counts(11, 0, 0).f1 - Counts(10, 1, 1).f1)
    # without the re-genotype search, or in veto mode, t_regenotype stays NEVER
    assert search_thresholds([fold], Mode.GENOTYPER, grid=[0.5, 0.8]).t_regenotype == NEVER
    assert search_thresholds([fold], Mode.VETO, regenotype=True).t_regenotype == NEVER


def _repeat_fold(str_period: np.ndarray | None, str_repeats: np.ndarray | None) -> FoldPredictions:
    # four flippy calls: two correct flips outside long repeats at P(flip) 0.55, and in a long
    # dinucleotide repeat a correct flip at 0.95 and a TP at 0.6; t_flip 0.5 is best for all calls,
    # but a stricter threshold in the repeat leaves its TP alone
    probabilities = np.array([
        [0.0, 0.45, 0.55],
        [0.0, 0.45, 0.55],
        [0.0, 0.05, 0.95],
        [0.0, 0.4, 0.6],
    ])
    return FoldPredictions(
        probabilities=probabilities,
        labels=np.array([Label.FLIP, Label.FLIP, Label.FLIP, Label.KEEP]),
        flip_ok=np.array([True, True, True, False]),
        flippable=np.ones(4, dtype=np.bool_),
        baseline=Counts(tp=10, fp=3, fn=3),
        regenotypable=np.zeros(4, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(4, dtype=np.bool_),
        regenotype_ok=np.zeros(4, dtype=np.bool_),
        str_period=str_period,
        str_repeats=str_repeats,
    )


def test_repeat_flip_selects_long_repeats() -> None:
    repeat_flip = RepeatFlip(min_period=2, min_units=8, t_flip=0.9)
    period = np.array([1.0, 2.0, 2.0, 3.0, np.nan])
    units = np.array([20.0, 7.0, 8.0, 12.0, np.nan])
    assert repeat_flip.selects(period, units).tolist() == [False, False, True, True, False]
    assert flip_thresholds(0.5, None, None, None) == 0.5
    per_call = flip_thresholds(0.5, repeat_flip, period, units)
    assert isinstance(per_call, np.ndarray)
    assert per_call.tolist() == [0.5, 0.5, 0.9, 0.9, 0.5]
    with pytest.raises(ValueError, match="repeat period and count"):
        flip_thresholds(0.5, repeat_flip, None, units)
    with pytest.raises(ValueError, match="repeat period and count"):
        flip_thresholds(0.5, repeat_flip, period, None)
    assert has_repeat_info(period, units) and not has_repeat_info(period, None)
    assert not has_repeat_info(np.array([1.0, np.nan]), np.array([np.nan, 8.0]))


def test_repeat_flip_search() -> None:
    fold = _repeat_fold(np.array([np.nan, 1.0, 2.0, 2.0]), np.array([np.nan, 5.0, 12.0, 12.0]))
    chosen = search_thresholds([fold], Mode.GENOTYPER, grid=[0.5, 0.7, 0.9])
    assert (chosen.t_drop, chosen.t_flip, chosen.repeat_flip) == (NEVER, 0.5, None)
    best = search_repeat_flip(
        [fold], chosen, grid=[0.5, 0.7, 0.9], periods=[1, 2], units=[6, 11], nested=False
    )
    # the narrowest stratum and the highest threshold that skip only the repeat TP
    assert best.repeat_flip == RepeatFlip(min_period=2, min_units=11, t_flip=0.9)
    assert (best.t_drop, best.t_flip) == (chosen.t_drop, chosen.t_flip)
    assert best.gain == pytest.approx(Counts(13, 0, 0).f1 - Counts(10, 3, 3).f1)
    assert best.gain > chosen.gain
    drop, flip, _ = fold.decide(best.t_drop, best.t_flip, Mode.GENOTYPER, NEVER, best.repeat_flip)
    assert flip.tolist() == [True, True, True, False] and not drop.any()


def test_repeat_flip_search_keeps_the_regenotype_threshold() -> None:
    # the repeat TP of `_repeat_fold` gets a re-genotype class: re-genotyping it would break it,
    # and a fifth call outside repeats is fixed only by re-genotyping
    fold = _repeat_fold(np.array([np.nan, 1.0, 2.0, 2.0]), np.array([np.nan, 5.0, 12.0, 12.0]))
    probabilities = np.zeros((5, 4))
    probabilities[:4, :3] = fold.probabilities
    probabilities[4] = [0.0, 0.1, 0.0, 0.9]
    fold = replace(
        fold,
        probabilities=probabilities,
        labels=np.array([*fold.labels, Label.REGENOTYPE]),
        flip_ok=np.array([*fold.flip_ok, False]),
        flippable=np.ones(5, dtype=np.bool_),
        regenotypable=np.ones(5, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(5, dtype=np.bool_),
        regenotype_ok=np.array([False, False, False, False, True]),
        str_period=np.array([np.nan, 1.0, 2.0, 2.0, np.nan]),
        str_repeats=np.array([np.nan, 5.0, 12.0, 12.0, np.nan]),
    )
    grid = [0.5, 0.7, 0.9]
    chosen = search_thresholds([fold], Mode.GENOTYPER, grid=grid, regenotype=True)
    assert chosen.t_regenotype < NEVER
    best = search_repeat_flip([fold], chosen, grid=grid, periods=[2], units=[11], nested=False)
    assert best.repeat_flip == RepeatFlip(min_period=2, min_units=11, t_flip=0.9)
    assert best.t_regenotype == chosen.t_regenotype
    assert best.gain == pytest.approx(
        fold.f1_gain(best.t_drop, best.t_flip, Mode.GENOTYPER, best.t_regenotype, best.repeat_flip)
    )
    assert best.gain > fold.f1_gain(
        best.t_drop, best.t_flip, Mode.GENOTYPER, NEVER, best.repeat_flip
    )


def test_repeat_flip_search_keeps_the_thresholds_without_evidence() -> None:
    # no call is in a repeat: no stratum threshold changes anything
    fold = _repeat_fold(np.full(4, np.nan), np.full(4, np.nan))
    chosen = search_thresholds([fold], Mode.GENOTYPER, grid=[0.5, 0.7])
    assert search_repeat_flip([fold], chosen, grid=[0.5, 0.7], nested=False) == chosen
    assert search_repeat_flip([fold, fold], chosen, grid=[0.5, 0.7]) == chosen
    with pytest.raises(ValueError, match="at least one fold"):
        search_repeat_flip([], chosen)
    with pytest.raises(ValueError, match="needs 2\\+ folds"):
        search_repeat_flip([fold], chosen)
    with pytest.raises(ValueError, match="repeat period and count"):
        search_repeat_flip([_repeat_fold(None, None)], chosen, nested=False)
    with_stratum = replace(chosen, repeat_flip=RepeatFlip(min_period=1, min_units=6, t_flip=0.9))
    with pytest.raises(ValueError, match="without a repeat_flip"):
        search_repeat_flip([fold], with_stratum, nested=False)


def _flip_fold(calls: list[tuple[float, bool, tuple[float, float] | None]]) -> FoldPredictions:
    """A fold of flippable calls: (P(flip), whether a flip fixes it, (period, units) or None)."""
    n = len(calls)
    p_flip = np.array([p for p, _, _ in calls])
    fixes = np.array([fix for _, fix, _ in calls])
    period = np.array([np.nan if r is None else r[0] for _, _, r in calls])
    units = np.array([np.nan if r is None else r[1] for _, _, r in calls])
    return FoldPredictions(
        probabilities=np.stack([np.zeros(n), 1 - p_flip, p_flip], axis=1),
        labels=np.where(fixes, Label.FLIP, Label.KEEP),
        flip_ok=fixes,
        flippable=np.ones(n, dtype=np.bool_),
        baseline=Counts(tp=20, fp=10, fn=10),
        regenotypable=np.zeros(n, dtype=np.bool_),
        regenotypes_to_ref=np.zeros(n, dtype=np.bool_),
        regenotype_ok=np.zeros(n, dtype=np.bool_),
        str_period=period,
        str_repeats=units,
    )


LONG: tuple[float, float] = (2.0, 12.0)


def test_repeat_flip_search_can_choose_a_lower_threshold() -> None:
    # outside repeats a TP sits at P(flip) 0.6, so t_flip stays high; in the repeat a fixable call
    # sits at 0.6 too, which only a lower stratum threshold flips
    fold = _flip_fold([(0.95, True, None), (0.6, False, None), (0.6, True, LONG)])
    grid = [0.5, 0.7, 0.9]
    chosen = search_thresholds([fold], Mode.GENOTYPER, grid=grid)
    assert chosen.t_flip == 0.9
    best = search_repeat_flip([fold], chosen, grid=grid, periods=[2], units=[11], nested=False)
    assert best.repeat_flip == RepeatFlip(min_period=2, min_units=11, t_flip=0.5)
    assert best.repeat_flip.t_flip < chosen.t_flip and best.gain > chosen.gain


def test_repeat_flip_search_breaks_ties_by_count_then_period() -> None:
    # two strata that skip different repeat TPs gain the same: (period >= 3, units >= 6) skips the
    # first, (period >= 1, units >= 11) the second; the stratum that covers both also covers two
    # fixable calls, so it gains nothing; the higher minimum count wins the tie
    fold = _flip_fold([
        *[(0.95, True, None)] * 5,
        (0.6, False, (3.0, 6.0)),
        (0.6, False, (1.0, 11.0)),
        (0.6, True, (2.0, 8.0)),
        (0.6, True, (2.0, 8.0)),
    ])
    chosen = search_thresholds([fold], Mode.GENOTYPER, grid=[0.5])
    assert chosen.t_flip == 0.5
    best = search_repeat_flip(
        [fold], chosen, grid=[0.5, 0.9], periods=[1, 2, 3], units=[6, 8, 11], nested=False
    )
    assert best.repeat_flip == RepeatFlip(min_period=1, min_units=11, t_flip=NEVER)
    skip_first = RepeatFlip(min_period=3, min_units=6, t_flip=NEVER)
    assert fold.f1_gain(NEVER, 0.5, Mode.GENOTYPER, NEVER, skip_first) == best.gain


def test_repeat_flip_search_nested_selection() -> None:
    # every fold has two repeat TPs that a stricter stratum threshold leaves alone: kept
    effect = _flip_fold([*[(0.95, True, None)] * 3, (0.6, False, LONG), (0.6, False, LONG)])
    chosen = search_thresholds([effect] * 3, Mode.GENOTYPER, grid=[0.5])
    assert chosen.t_flip == 0.5
    best = search_repeat_flip([effect] * 3, chosen, grid=[0.5], periods=[2], units=[11])
    assert best.repeat_flip == RepeatFlip(min_period=2, min_units=11, t_flip=NEVER)
    # one fold has three such TPs, two folds a fixable repeat call each: the stratum gains
    # in-sample, but chosen without the first fold it never gains on another fold, so it is dropped
    big = _flip_fold([(0.95, True, None), *[(0.6, False, LONG)] * 3])
    small = _flip_fold([(0.95, True, None), (0.6, True, LONG)])
    folds = [big, small, small]
    chosen = search_thresholds(folds, Mode.GENOTYPER, grid=[0.5])
    assert chosen.t_flip == 0.5
    in_sample = search_repeat_flip(folds, chosen, grid=[0.5], periods=[2], units=[11], nested=False)
    assert in_sample.repeat_flip is not None and in_sample.gain > chosen.gain
    assert search_repeat_flip(folds, chosen, grid=[0.5], periods=[2], units=[11]) == chosen
