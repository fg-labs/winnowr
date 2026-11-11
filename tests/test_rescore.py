import numpy as np
import pytest

from winnowr.common import NEVER
from winnowr.labels import Label
from winnowr.promote_labels import PromoteLabel
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.rescore import baseline_counts
from winnowr.rescore import decide
from winnowr.rescore import decide_promote
from winnowr.rescore import rescore
from winnowr.rescore import rescore_promote

BASE = Counts(tp=100, fp=10, fn=20)


def _one(label: Label, flip_ok: bool, drop: bool, flip: bool) -> Counts:
    no = np.array([False])
    return rescore([label], [flip_ok], np.array([drop]), np.array([flip]), BASE, no, no)


@pytest.mark.parametrize(
    "label, flip_ok, drop, flip, expected",
    [
        (Label.KEEP, False, True, False, Counts(99, 10, 21)),  # drop TP
        (Label.DROP, False, True, False, Counts(100, 9, 20)),  # drop FP
        (Label.FLIP, True, True, False, Counts(100, 9, 20)),  # drop allele-match FP
        (Label.KEEP, False, False, True, Counts(99, 11, 21)),  # flip TP
        (Label.FLIP, True, False, True, Counts(101, 9, 19)),  # flip FP, correct
        (Label.FLIP, False, False, True, Counts(100, 10, 20)),  # flip FP, still wrong
        (Label.DROP, False, False, True, Counts(100, 10, 20)),  # flip a 0/0 FP
        (Label.FLIP, True, True, True, Counts(100, 9, 20)),  # drop wins over flip
        (Label.KEEP, False, False, False, BASE),  # no action
    ],
)
def test_rescore_each_case(
    label: Label, flip_ok: bool, drop: bool, flip: bool, expected: Counts
) -> None:
    assert _one(label, flip_ok, drop, flip) == expected


def test_f1_and_baseline() -> None:
    assert Counts(tp=8, fp=2, fn=2).f1 == pytest.approx(0.8)
    assert Counts(0, 0, 0).f1 == 0.0
    assert baseline_counts([Label.KEEP, Label.DROP, Label.FLIP, Label.KEEP], fn=5) == Counts(
        2, 2, 5
    )


def test_decide_genotyper_and_veto() -> None:
    probabilities = np.array([
        [0.6, 0.1, 0.3],  # drop
        [0.1, 0.3, 0.6],  # flip
        [0.1, 0.3, 0.6],  # flip, but not flippable
        [0.3, 0.4, 0.3],  # keep (genotyper); veto drops (0.6)
    ])
    flippable = [True, True, False, True]
    no = [False] * 4
    drop, flip, regenotype = decide(
        probabilities, flippable, 0.5, 0.5, Mode.GENOTYPER, NEVER, no, no
    )
    assert not regenotype.any()
    assert drop.tolist() == [True, False, False, False]
    assert flip.tolist() == [False, True, False, False]
    drop, flip, _ = decide(probabilities, flippable, 0.6, 0.5, Mode.VETO, NEVER, no, no)
    assert drop.tolist() == [True, True, True, True]
    assert not flip.any()


def test_decide_and_rescore_promote() -> None:
    probabilities = np.array([
        [0.3, 0.6, 0.1],  # promote
        [0.3, 0.2, 0.5],  # promote + flip
        [0.3, 0.2, 0.5],  # promote, not flippable so no flip
        [0.8, 0.1, 0.1],  # stay
        [0.2, 0.5, 0.3],  # promote (p_flip not > p_promote)
    ])
    promote, flip = decide_promote(probabilities, [True, True, False, True, True], 0.6, 0.3)
    assert promote.tolist() == [True, True, True, False, True]
    assert flip.tolist() == [False, True, False, False, False]
    labels = [1, 2, 2, 1, 0]
    # correct: rows 0 and 1; wrong: row 2 (needed a flip) and row 4 (label 0)
    assert rescore_promote(labels, promote, flip, BASE) == Counts(102, 12, 18)


def test_decide_regenotype() -> None:
    probabilities = np.array([
        [0.1, 0.1, 0.1, 0.7],  # re-genotype
        [0.1, 0.1, 0.1, 0.7],  # re-genotype to 0/0: dropped instead
        [0.1, 0.1, 0.1, 0.7],  # the length caller agrees with the call: no action
        [0.1, 0.1, 0.6, 0.2],  # flip wins (and p_regenotype is low)
        [0.6, 0.3, 0.0, 0.1],  # drop (veto: 0.7 < 0.8)
    ])
    regenotypable = [True, True, False, True, True]
    to_ref = [False, True, False, False, False]
    drop, flip, regenotype = decide(
        probabilities, [True] * 5, 0.5, 0.5, Mode.GENOTYPER, 0.5, regenotypable, to_ref
    )
    assert drop.tolist() == [False, True, False, False, True]
    assert flip.tolist() == [False, False, False, True, False]
    assert regenotype.tolist() == [True, False, False, False, False]
    # without length calls, or with a 3-class model, nothing is re-genotyped
    none = [False] * 5
    assert not decide(probabilities, [True] * 5, 0.5, 0.5, Mode.GENOTYPER, 0.5, none, none)[2].any()
    three_class = probabilities[:, :3]
    args = (0.5, 0.5, Mode.GENOTYPER, 0.5, regenotypable, to_ref)
    assert not decide(three_class, [True] * 5, *args)[2].any()
    # veto counts P(regenotype) as "the call is wrong"
    drop, _, regenotype = decide(probabilities, [True] * 5, 0.8, 0.5, Mode.VETO, 0.5, none, none)
    assert drop.tolist() == [True, True, True, True, False] and not regenotype.any()


@pytest.mark.parametrize(
    "label, regenotype_ok, expected",
    [
        (Label.KEEP, False, Counts(99, 11, 21)),  # re-genotype a TP
        (Label.REGENOTYPE, True, Counts(101, 9, 19)),  # re-genotype an FP, correct
        (Label.FLIP, False, Counts(100, 10, 20)),  # re-genotype an FP, still wrong
    ],
)
def test_rescore_regenotype(label: Label, regenotype_ok: bool, expected: Counts) -> None:
    no = np.array([False])
    got = rescore([label], [False], no, no, BASE, np.array([True]), [regenotype_ok])
    assert got == expected
    # a call also flipped is only flipped
    flipped = rescore([label], [False], no, np.array([True]), BASE, np.array([True]), [True])
    assert flipped == _one(label, False, False, True)


def test_a_wrongly_flipped_promotion_is_a_new_fp() -> None:
    # a PROMOTE call (right genotype as called) that the thresholds also flip
    promote, flip = decide_promote(np.array([[0.1, 0.35, 0.55]]), [True], 0.5, 0.3)
    assert promote.tolist() == [True] and flip.tolist() == [True]
    assert rescore_promote([PromoteLabel.PROMOTE], promote, flip, BASE) == Counts(100, 11, 20)
