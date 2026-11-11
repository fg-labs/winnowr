import numpy as np
import pytest

from winnowr.common import NEVER
from winnowr.labels import Label
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.thresholds import FoldPredictions
from winnowr.thresholds import PromoteFold
from winnowr.thresholds import search_promote_thresholds
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
