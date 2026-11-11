import numpy as np
import pandas as pd
import pytest

from winnowr.common import NEVER
from winnowr.evaluation import PolicyMetric
from winnowr.evaluation import evaluate_model
from winnowr.evaluation import evaluate_promote_model
from winnowr.labels import Label
from winnowr.promote_labels import PromoteLabel
from winnowr.regenotype import without_regenotype
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.rescore import RepeatFlip
from winnowr.samples import PromoteSampleCalls
from winnowr.samples import SampleCalls
from winnowr.thresholds import PromoteThresholds
from winnowr.thresholds import Thresholds
from winnowr.thresholds import search_promote_thresholds

BASELINE = Counts(tp=3, fp=2, fn=5)


def _calls() -> tuple[SampleCalls, np.ndarray]:
    """A dropped FP, a dropped TP, a fixed flip, a flipped TP and an untouched TP."""
    labels = [Label.DROP, Label.KEEP, Label.FLIP, Label.KEEP, Label.KEEP]
    calls = pd.DataFrame({
        "label": np.array(labels, dtype=np.float64),
        "flip_ok": [False, False, True, False, False],
        "flippable": True,
    })
    probabilities = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.2, 0.8],
        [0.1, 0.3, 0.6],
        [0.0, 1.0, 0.0],
    ])
    return SampleCalls(
        sample="s", calls=without_regenotype(calls), baseline=BASELINE
    ), probabilities


def _candidates() -> tuple[PromoteSampleCalls, np.ndarray]:
    """A correct promotion, a correct promotion with a flip, and a wrong promotion."""
    labels = [PromoteLabel.PROMOTE, PromoteLabel.PROMOTE_FLIP, PromoteLabel.STAY]
    calls = pd.DataFrame({"promote_label": np.array(labels, dtype=np.float64), "flippable": True})
    probabilities = np.array([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3]])
    return PromoteSampleCalls(sample="s", calls=calls, baseline=BASELINE), probabilities


def _metrics() -> dict[str, PolicyMetric]:
    calls, probabilities = _calls()
    thresholds = {Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0), Mode.VETO: Thresholds(0.5, NEVER, 0.0)}
    pass_metrics = evaluate_model(calls, probabilities, "snp", thresholds)
    candidates, promote_probabilities = _candidates()
    promote = evaluate_promote_model(
        candidates, promote_probabilities, PromoteThresholds(0.6, 0.5, 0.0), pass_metrics
    )
    return {m.policy: m for m in [*pass_metrics, *promote]}


def test_pass_call_policies() -> None:
    metrics = _metrics()
    assert metrics["baseline"] == PolicyMetric(
        "s", "snp", "baseline", "genotyper", NEVER, NEVER, 0, 0, 0, 0, 0, 3, 2, 5, BASELINE.f1, 0.0
    )
    genotyper = Counts(tp=2, fp=1, fn=6)
    assert metrics["genotyper"] == PolicyMetric(
        "s", "snp", "genotyper", "genotyper", 0.5, 0.5, 2, 1, 2, 1, 1, 2, 1, 6,
        genotyper.f1, genotyper.f1 - BASELINE.f1,
    )  # fmt: skip
    veto = metrics["veto"]
    assert (veto.t_flip, veto.dropped, veto.dropped_tp, veto.flipped) == (NEVER, 4, 2, 0)
    assert (veto.tp, veto.fp, veto.fn) == (1, 0, 7)
    flip_only = metrics["flip_only"]
    assert (flip_only.t_drop, flip_only.dropped, flip_only.flipped) == (NEVER, 0, 2)
    assert (flip_only.tp, flip_only.fp, flip_only.fn) == (3, 2, 5)


def test_repeat_flip_policies() -> None:
    calls, probabilities = _calls()
    # the flipped TP (P(flip) 0.6) is in a long repeat, where the threshold is 0.7
    repeats = calls.calls.assign(
        str_period=[np.nan, np.nan, np.nan, 2.0, np.nan],
        str_repeats=[np.nan, np.nan, np.nan, 12.0, np.nan],
    )
    calls = SampleCalls(sample="s", calls=repeats, baseline=BASELINE)
    repeat_flip = RepeatFlip(min_period=2, min_units=11, t_flip=0.7)
    thresholds = {
        Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0, repeat_flip=repeat_flip),
        Mode.VETO: Thresholds(0.5, NEVER, 0.0),
    }
    metrics = {m.policy: m for m in evaluate_model(calls, probabilities, "indel", thresholds)}
    genotyper = metrics["genotyper"]
    assert (genotyper.flipped, genotyper.flips_fixed, genotyper.flips_broke_tp) == (1, 1, 0)
    assert (genotyper.repeat_min_period, genotyper.repeat_min_units) == (2, 11)
    assert genotyper.t_flip_repeat == 0.7
    unstratified = metrics["genotyper_no_repeat_flip"]
    assert (unstratified.flipped, unstratified.flips_broke_tp) == (2, 1)
    assert unstratified.t_flip_repeat == NEVER and unstratified.repeat_min_period == 0
    assert genotyper.f1 > unstratified.f1
    assert metrics["flip_only"].flipped == 1
    # without a stratum threshold there is no genotyper_no_repeat_flip policy
    assert "genotyper_no_repeat_flip" not in _metrics()


def test_tuned_genotyper_tunes_the_repeat_flip_threshold() -> None:
    # two fixed flips outside repeats at P(flip) 0.6, and in a long repeat a TP at 0.8 and a fixed
    # flip at 0.95: only a separate threshold in the repeat gets all of the fixes and no TP
    labels = [Label.FLIP, Label.FLIP, Label.KEEP, Label.FLIP, Label.KEEP]
    frame = pd.DataFrame({
        "label": np.array(labels, dtype=np.float64),
        "flip_ok": [True, True, False, True, False],
        "flippable": True,
        "str_period": [np.nan, np.nan, 2.0, 2.0, np.nan],
        "str_repeats": [np.nan, np.nan, 12.0, 12.0, np.nan],
    })
    calls = SampleCalls(sample="s", calls=without_regenotype(frame), baseline=BASELINE)
    probabilities = np.array([
        [0, 0.4, 0.6],
        [0, 0.4, 0.6],
        [0, 0.2, 0.8],
        [0, 0.05, 0.95],
        [0, 1, 0],
    ])
    thresholds = {
        Mode.GENOTYPER: Thresholds(NEVER, 0.6, 0.0, repeat_flip=RepeatFlip(2, 11, 0.9)),
        Mode.VETO: Thresholds(NEVER, NEVER, 0.0),
    }
    metrics = {
        m.policy: m for m in evaluate_model(calls, probabilities, "indel", thresholds, tuned=True)
    }
    genotyper, tuned = metrics["genotyper"], metrics["genotyper_tuned"]
    assert (genotyper.flipped, genotyper.flips_broke_tp) == (3, 0)
    assert tuned.f1 == genotyper.f1 > metrics["genotyper_no_repeat_flip"].f1
    assert tuned.flips_broke_tp == 0
    assert tuned.repeat_min_period > 0 and 0.8 < tuned.t_flip_repeat <= 0.95
    # a --repeat-flip model whose search kept no stratum still gets one tuned, if asked
    plain = {**thresholds, Mode.GENOTYPER: Thresholds(NEVER, 0.6, 0.0)}
    for tune_repeat_flip, stratum in ((False, False), (True, True)):
        metrics = {
            m.policy: m
            for m in evaluate_model(
                calls, probabilities, "indel", plain, tuned=True, tune_repeat_flip=tune_repeat_flip
            )
        }
        assert (metrics["genotyper_tuned"].repeat_min_period > 0) is stratum
        assert "genotyper_no_repeat_flip" not in metrics


def test_repeat_flip_without_repeat_info_warns(caplog: pytest.LogCaptureFixture) -> None:
    calls, probabilities = _calls()
    no_repeats = calls.calls.assign(str_period=np.nan, str_repeats=np.nan)
    calls = SampleCalls(sample="s", calls=no_repeats, baseline=BASELINE)
    thresholds = {
        Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0, repeat_flip=RepeatFlip(2, 11, NEVER)),
        Mode.VETO: Thresholds(0.5, NEVER, 0.0),
    }
    metrics = {
        m.policy: m for m in evaluate_model(calls, probabilities, "indel", thresholds, tuned=True)
    }
    assert "no call has a DRAGstr repeat period and count" in caplog.text
    assert metrics["genotyper"].f1 == metrics["genotyper_no_repeat_flip"].f1
    assert metrics["genotyper_tuned"].repeat_min_period == 0


def test_promote_policies_report_promote_flips_separately() -> None:
    metrics = _metrics()
    promote = metrics["promote"]
    assert (promote.t_flip, promote.flipped, promote.t_promote, promote.t_promote_flip) == (
        NEVER, 0, 0.6, 0.5,
    )  # fmt: skip
    assert (promote.promoted, promote.promoted_correct, promote.promoted_flipped) == (3, 2, 1)
    assert (promote.tp, promote.fp, promote.fn) == (5, 3, 3)
    no_flip = metrics["promote_no_flip"]
    assert (no_flip.t_promote_flip, no_flip.promoted_correct, no_flip.promoted_flipped) == (
        NEVER, 1, 0,
    )  # fmt: skip

    both = metrics["genotyper+promote"]
    genotyper = metrics["genotyper"]
    pass_columns = ("mode", "t_drop", "t_flip", "dropped", "dropped_tp", "flipped", "flips_fixed")
    assert all(getattr(both, c) == getattr(genotyper, c) for c in (*pass_columns, "flips_broke_tp"))
    assert (both.t_promote, both.t_promote_flip) == (0.6, 0.5)
    assert (both.promoted, both.promoted_correct, both.promoted_flipped) == (3, 2, 1)
    assert (both.tp, both.fp, both.fn) == (4, 2, 4)
    assert both.f1_gain == pytest.approx(Counts(4, 2, 4).f1 - BASELINE.f1)


def test_tuned_policies() -> None:
    calls, probabilities = _calls()
    thresholds = {
        Mode.GENOTYPER: Thresholds(NEVER, NEVER, 0.0),
        Mode.VETO: Thresholds(NEVER, NEVER, 0.0),
    }
    metrics = {
        m.policy: m for m in evaluate_model(calls, probabilities, "snp", thresholds, tuned=True)
    }
    # tuned on the sample itself: drop only the confident FP (0.9) and flip only the fixed call
    tuned = metrics["genotyper_tuned"]
    assert (tuned.t_drop, tuned.t_flip, tuned.dropped_tp, tuned.flips_broke_tp) == (0.9, 0.8, 0, 0)
    # veto tuned: dropping at 0.9 removes the FP only; at 0.8 it would also drop a TP
    veto = metrics["veto_tuned"]
    assert (veto.t_drop, veto.dropped, veto.dropped_tp, veto.flipped) == (0.9, 1, 0, 0)
    assert veto.f1_gain == pytest.approx(Counts(3, 1, 5).f1 - BASELINE.f1)

    candidates, promote_probabilities = _candidates()
    promote = evaluate_promote_model(
        candidates,
        promote_probabilities,
        PromoteThresholds(NEVER, NEVER, 0.0),
        list(metrics.values()),
        tuned=True,
    )
    tuned_promote = {m.policy: m for m in promote}["promote_tuned"]
    best = search_promote_thresholds([candidates.to_fold(promote_probabilities)])
    assert (tuned_promote.t_promote, tuned_promote.t_promote_flip) == (best.t_promote, best.t_flip)
    assert tuned_promote.t_promote < NEVER and tuned_promote.f1_gain > 0
