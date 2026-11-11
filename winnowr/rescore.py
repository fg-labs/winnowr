"""
Turn class probabilities into drop/flip decisions and rescore F1 with vcfeval arithmetic.

Under vcfeval semantics a genotype mismatch is a query FP *and* a truth FN. Starting from baseline
counts (TP, FP, FN), each action changes them as follows:

==========================  =================
action                      change
==========================  =================
drop a TP                   TP-1, FN+1
drop an FP                  FP-1
flip a TP                   TP-1, FP+1, FN+1
flip an FP, flip correct    TP+1, FP-1, FN-1
flip an FP, not correct     none
re-genotype a TP            TP-1, FP+1, FN+1
re-genotype an FP, correct  TP+1, FP-1, FN-1
re-genotype an FP, not      none
==========================  =================

A "correct" flip is an allele-match FP whose het<->hom flip gives the truth genotype. Models
trained with the re-genotype action (`Label.REGENOTYPE`) can also replace a call's genotype with
the repeat-length caller's (see `winnowr.regenotype`); that is correct for an FP whose
length-caller genotype equals the truth (an allele match, or a local match such as a wrong
repeat length), and re-genotyping to `0/0` is a drop.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from winnowr.common import BoolArray
from winnowr.common import FloatArray
from winnowr.labels import Label
from winnowr.promote_labels import PromoteLabel


class Mode(Enum):
    """
    How model probabilities are used.

    Attributes:
        GENOTYPER: drop calls with P(drop) >= t_drop; flip het<->hom calls with P(flip) >= t_flip;
            for re-genotype models, re-genotype the remaining calls the length caller disagrees
            with when P(regenotype) >= t_regenotype.
        VETO: drop calls with P(drop) + P(flip) (+ P(regenotype)) >= t_drop (the call as made is
            wrong); never flip or re-genotype.
    """

    GENOTYPER = "genotyper"
    VETO = "veto"


@dataclass(frozen=True)
class Counts:
    """
    vcfeval-style counts.

    Attributes:
        tp: true positives (query calls matching truth genotype).
        fp: false positives (query calls not matching truth).
        fn: false negatives (truth variants not matched).
    """

    tp: int
    fp: int
    fn: int

    @property
    def f1(self) -> float:
        """The F1 score, 2TP / (2TP + FP + FN) (0 when undefined)."""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0


def baseline_counts(labels: npt.ArrayLike, fn: int) -> Counts:
    """
    The counts before any action.

    Args:
        labels: the `Label` of each scored query call.
        fn: the number of truth FNs.

    Returns:
        TP = calls labelled KEEP, FP = all other calls, and FN.
    """
    labels_array = np.asarray(labels)
    is_tp = labels_array == Label.KEEP
    return Counts(tp=int(is_tp.sum()), fp=int((~is_tp).sum()), fn=fn)


def decide(
    probabilities: FloatArray,
    flippable: npt.ArrayLike,
    t_drop: float,
    t_flip: float,
    mode: Mode,
    t_regenotype: float,
    regenotypable: npt.ArrayLike,
    regenotypes_to_ref: npt.ArrayLike,
) -> tuple[BoolArray, BoolArray, BoolArray]:
    """
    Decide which calls to drop, flip and re-genotype.

    In genotyper mode the actions take precedence in that order: drop, then flip, then
    re-genotype. A call re-genotyped to `0/0` is dropped instead.

    Args:
        probabilities: an `(n, 3)` array of class probabilities in `Label` order, or `(n, 4)` for
            re-genotype models.
        flippable: per call, whether the genotype is `0/1` or `1/1`.
        t_drop: the drop threshold.
        t_flip: the flip threshold (ignored in veto mode).
        mode: the decision mode.
        t_regenotype: the re-genotype threshold (used only with 4 classes in genotyper mode).
        regenotypable: per call, whether the re-genotype action can change it (all false without
            length calls; see `winnowr.regenotype.with_regenotype_columns`).
        regenotypes_to_ref: per call, whether the length caller's genotype is `0/0`.

    Returns:
        The drop, flip and re-genotype masks; a call gets at most one action.
    """
    n = len(probabilities)
    no_action = np.zeros(n, dtype=np.bool_)
    p_drop, p_flip = probabilities[:, Label.DROP], probabilities[:, Label.FLIP]
    has_regenotype = probabilities.shape[1] > Label.REGENOTYPE
    if mode is Mode.VETO:
        p_wrong = p_drop + p_flip
        if has_regenotype:
            p_wrong = p_wrong + probabilities[:, Label.REGENOTYPE]
        return p_wrong >= t_drop, no_action, no_action
    drop: BoolArray = p_drop >= t_drop
    flip: BoolArray = (p_flip >= t_flip) & np.asarray(flippable, dtype=np.bool_) & ~drop
    if not has_regenotype:
        return drop, flip, no_action
    regenotype: BoolArray = (
        (probabilities[:, Label.REGENOTYPE] >= t_regenotype)
        & np.asarray(regenotypable, dtype=np.bool_)
        & ~drop
        & ~flip
    )
    to_ref = regenotype & np.asarray(regenotypes_to_ref, dtype=np.bool_)
    return drop | to_ref, flip, regenotype & ~to_ref


def rescore(
    labels: npt.ArrayLike,
    flip_ok: npt.ArrayLike,
    drop: BoolArray,
    flip: BoolArray,
    baseline: Counts,
    regenotype: BoolArray,
    regenotype_ok: npt.ArrayLike,
) -> Counts:
    """
    Apply drop/flip/re-genotype actions to baseline counts (see the module docstring).

    Args:
        labels: the `Label` of each scored query call.
        flip_ok: per call, whether flipping it gives the truth genotype.
        drop: calls to drop.
        flip: calls to flip (calls also in `drop` are only dropped).
        baseline: the counts before any action.
        regenotype: calls to re-genotype (calls also in `drop` or `flip` only get that action).
        regenotype_ok: per call, whether re-genotyping it gives the truth genotype.

    Returns:
        The counts after the actions.
    """
    labels_array = np.asarray(labels)
    is_tp = labels_array == Label.KEEP
    flip_only = flip & ~drop
    dropped_tp = int((drop & is_tp).sum())
    dropped_fp = int((drop & ~is_tp).sum())
    broke = int((flip_only & is_tp).sum())
    fixed = int((flip_only & ~is_tp & np.asarray(flip_ok, dtype=np.bool_)).sum())
    regenotype_only = regenotype & ~drop & ~flip
    broke += int((regenotype_only & is_tp).sum())
    fixed += int((regenotype_only & ~is_tp & np.asarray(regenotype_ok, dtype=np.bool_)).sum())
    return Counts(
        tp=baseline.tp - dropped_tp - broke + fixed,
        fp=baseline.fp - dropped_fp + broke - fixed,
        fn=baseline.fn + dropped_tp + broke - fixed,
    )


def decide_promote(
    probabilities: FloatArray, flippable: npt.ArrayLike, t_promote: float, t_flip: float
) -> tuple[BoolArray, BoolArray]:
    """
    Decide which non-PASS calls to promote, and which promoted calls to flip.

    A call is promoted when P(promote) + P(promote and flip) >= `t_promote`, and flipped when also
    P(promote and flip) >= `t_flip`, P(promote and flip) > P(promote) and its genotype is `0/1` or
    `1/1`.

    Args:
        probabilities: an `(n, 3)` array of class probabilities in `PromoteLabel` order.
        flippable: per call, whether the genotype is `0/1` or `1/1`.
        t_promote: the promotion threshold.
        t_flip: the flip threshold.

    Returns:
        The promote and flip masks; flipped calls are always promoted.
    """
    p_promote = probabilities[:, PromoteLabel.PROMOTE]
    p_flip = probabilities[:, PromoteLabel.PROMOTE_FLIP]
    promote: BoolArray = (p_promote + p_flip) >= t_promote
    flip: BoolArray = (
        promote & (p_flip >= t_flip) & (p_flip > p_promote) & np.asarray(flippable, dtype=np.bool_)
    )
    return promote, flip


def rescore_promote(
    labels: npt.ArrayLike, promote: BoolArray, flip: BoolArray, baseline: Counts
) -> Counts:
    """
    Apply promotions to baseline counts.

    A promoted call with the right genotype (`PROMOTE` not flipped, or `PROMOTE_FLIP` flipped)
    turns a truth FN into a TP; any other promoted call is a new FP.

    Args:
        labels: the `PromoteLabel` of each candidate.
        promote: candidates to promote.
        flip: promoted candidates whose genotype is flipped.
        baseline: the counts before promotion.

    Returns:
        The counts after promotion.
    """
    labels_array = np.asarray(labels)
    correct = promote & (
        ((labels_array == PromoteLabel.PROMOTE) & ~flip)
        | ((labels_array == PromoteLabel.PROMOTE_FLIP) & flip)
    )
    n_correct, n_wrong = int(correct.sum()), int((promote & ~correct).sum())
    return Counts(tp=baseline.tp + n_correct, fp=baseline.fp + n_wrong, fn=baseline.fn - n_correct)
