from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

import winnowr.training
from tests.test_model import FAST
from tests.test_model import synthetic_calls
from winnowr.common import NEVER
from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.labels import Label
from winnowr.regenotype import without_regenotype
from winnowr.rescore import Counts
from winnowr.rescore import Mode
from winnowr.rescore import baseline_counts
from winnowr.samples import PromoteSampleCalls
from winnowr.samples import SampleCalls
from winnowr.stutter import fit_stutter_model
from winnowr.table import FeatureTableInfo
from winnowr.training import cross_validate
from winnowr.training import group_samples
from winnowr.training import train_model
from winnowr.training import train_promote_model


def _sample(name: str, seed: int) -> SampleCalls:
    calls = synthetic_calls(2000, seed)
    calls = without_regenotype(calls)
    return SampleCalls(sample=name, calls=calls, baseline=baseline_counts(calls["label"], fn=50))


def test_train_model_chooses_thresholds_that_help() -> None:
    samples = [_sample("a", 1), _sample("b", 2), _sample("c", 3)]
    model = train_model(
        samples, VariantType.SNP, ["x", "y"], region_names=[], contigs=["chr1"], params=FAST
    )
    assert model.info.training_samples == ("a", "b", "c")
    genotyper, veto = model.info.thresholds[Mode.GENOTYPER], model.info.thresholds[Mode.VETO]
    assert genotyper.gain > veto.gain > 0
    assert genotyper.t_flip < NEVER and veto.t_flip == NEVER


def test_cross_validate_needs_two_distinct_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        cross_validate([_sample("a", 1)], ["x"], FAST)
    with pytest.raises(ValueError, match=r"given once; repeated: \['a'\]"):
        cross_validate([_sample("a", 1), _sample("b", 2), _sample("a", 1)], ["x"], FAST)


def test_cross_validate_holds_out_each_sample(mocker: MockerFixture) -> None:
    samples = [
        SampleCalls(
            sample=name,
            calls=without_regenotype(calls),
            baseline=baseline_counts(calls["label"], fn=0),
        )
        for name, calls in (
            ("a", synthetic_calls(300, 1)),
            ("b", synthetic_calls(400, 2)),
            ("c", synthetic_calls(500, 3)),
        )
    ]
    fit = mocker.spy(winnowr.training, "fit_booster")
    folds = cross_validate(samples, ["x", "y"], FAST)
    # each fold is fitted on exactly the other samples' calls and scores the held-out sample
    assert [len(call.args[0]) for call in fit.call_args_list] == [900, 800, 700]
    assert [len(fold.labels) for fold in folds] == [300, 400, 500]


def test_group_samples_defaults_to_singletons() -> None:
    assert group_samples(["a", "b", "c"]) == [("a",), ("b",), ("c",)]


def test_group_samples_holds_out_groups_together() -> None:
    # groups follow the input order of their first sample; ungrouped samples stay singletons
    names = ["a", "b", "c", "d", "e"]
    assert group_samples(names, [["e", "b"], ["c"]]) == [("a",), ("b", "e"), ("c",), ("d",)]


def test_group_samples_checks_groups() -> None:
    with pytest.raises(ValueError, match=r"unknown samples: \['x'\]"):
        group_samples(["a", "b"], [["a", "x"]])
    with pytest.raises(ValueError, match="'a' is in more than one"):
        group_samples(["a", "b", "c"], [["a", "b"], ["a", "c"]])
    with pytest.raises(ValueError, match="must not be empty"):
        group_samples(["a", "b"], [[]])


def test_cross_validate_holds_out_each_group(mocker: MockerFixture) -> None:
    samples = [
        SampleCalls(
            sample=name,
            calls=without_regenotype(calls),
            baseline=baseline_counts(calls["label"], fn=0),
        )
        for name, calls in (
            ("a", synthetic_calls(300, 1)),
            ("b", synthetic_calls(400, 2)),
            ("c", synthetic_calls(500, 3)),
        )
    ]
    fit = mocker.spy(winnowr.training, "fit_booster")
    folds = cross_validate(samples, ["x", "y"], FAST, groups=[["a", "c"]])
    # {a, c} is held out together (fitted on b), then b (fitted on a + c); one fold per sample
    assert [len(call.args[0]) for call in fit.call_args_list] == [400, 800]
    assert [len(fold.labels) for fold in folds] == [300, 400, 500]
    with pytest.raises(ValueError, match="at least two groups"):
        cross_validate(samples, ["x"], FAST, groups=[["a", "b", "c"]])
    with pytest.raises(ValueError, match="unknown samples"):
        cross_validate(samples, ["x"], FAST, groups=[["a", "z"]])


def test_cross_validate_rebuilds_the_samples_of_each_fold() -> None:
    samples = [_sample("a", 1), _sample("b", 2), _sample("c", 3)]
    seen: list[frozenset[str]] = []
    marked = Counts(tp=1, fp=2, fn=3)

    def refit(training: frozenset[str]) -> list[SampleCalls]:
        seen.append(training)
        return [s if s.sample in training else replace(s, baseline=marked) for s in samples]

    folds = cross_validate(samples, ["x", "y"], FAST, refit=refit)
    assert seen == [frozenset("bc"), frozenset("ac"), frozenset("ab")]
    # each held-out sample is scored as the fold rebuilt it
    assert [fold.baseline for fold in folds] == [marked] * 3
    with pytest.raises(ValueError, match="refit must return the samples"):
        cross_validate(samples, ["x", "y"], FAST, refit=lambda _: samples[::-1])


def _promote_sample(name: str, seed: int) -> PromoteSampleCalls:
    calls = synthetic_calls(2000, seed).rename(columns={"label": "promote_label"})
    # stay (0) for low x, promote (1) for mid x, promote + flip (2) for high x
    return PromoteSampleCalls(sample=name, calls=calls, baseline=Counts(tp=10_000, fp=100, fn=2000))


def test_train_promote_model() -> None:
    samples = [_promote_sample("a", 1), _promote_sample("b", 2)]
    model = train_promote_model(
        samples, VariantType.SNP, ["x", "y"], region_names=[], contigs=["chr1"], params=FAST
    )
    assert model.info.task is Task.PROMOTE and model.info.thresholds == {}
    chosen = model.info.promote_thresholds
    assert chosen is not None and chosen.gain > 0 and chosen.t_flip < NEVER
    with pytest.raises(ValueError, match="at least two samples"):
        train_promote_model(samples[:1], VariantType.SNP, ["x"], [], ["chr1"], FAST)


def test_samples_need_happy_labels() -> None:
    with pytest.raises(ValueError, match="no hap.py labels"):
        SampleCalls.from_table(
            pd.DataFrame(), FeatureTableInfo("s", ("chr1",), ()), VariantType.SNP
        )


def test_promote_samples_need_promote_labels() -> None:
    with pytest.raises(ValueError, match="no promote labels"):
        PromoteSampleCalls.from_table(
            pd.DataFrame(), FeatureTableInfo("s", ("chr1",), (), {"snp": 0}), VariantType.SNP
        )


def _indel_table() -> pd.DataFrame:
    """Labelled het deletions: a TP, a 0/0 FP, a 1/1 truth (flip) and a 1/2 truth (regenotype)."""
    return pd.DataFrame({
        "chrom": ["chr1"] * 4,
        "pos": [100, 200, 300, 400],
        "ref": ["CA"] * 4,
        "alt": ["C"] * 4,
        "gt": ["0/1"] * 4,
        "filter": ["PASS"] * 4,
        "variant_type": ["indel"] * 4,
        "label": [float(Label.KEEP), float(Label.DROP), float(Label.FLIP), float(Label.FLIP)],
        "flip_ok": [False, False, True, False],
        "flippable": True,
        "sl_period": 1.0,
        "sl_tract_len": 5.0,
        "sl_tract_start": [101.0, 201.0, 301.0, 401.0],
        "sl_unit": "A",
        "sl_n_spanning": 20.0,
        "sl_histogram": ["-1:10;0:10", "0:20", "-1:20", "-1:10;1:10"],
        "truth_delta_lo": [-1.0, np.nan, -1.0, -1.0],
        "truth_delta_hi": [0.0, np.nan, -1.0, 1.0],
    })


def test_sample_calls_with_length_calls_and_regenotype_labels() -> None:
    stutter = fit_stutter_model([1], [5], [0], [{-1: 5, 0: 90, 1: 5}])
    info = FeatureTableInfo("s", ("chr1",), (), {"snp": 0, "indel": 3}, strlen=True)
    calls = SampleCalls.from_table(
        _indel_table(), info, VariantType.INDEL, stutter, regenotype=True
    )
    assert calls.calls["label"].tolist() == [1, 0, 2, 3]
    # pos 300 is fixed by a re-genotype too, but a flip fixes it so it keeps the flip label
    assert calls.calls["regenotype_ok"].tolist() == [False, False, True, True]
    assert calls.baseline == Counts(tp=1, fp=3, fn=3)
    fold = calls.to_fold(np.zeros((4, 4)))
    assert fold.regenotype_ok is not None and fold.regenotype_ok.tolist()[3]
    # without the re-genotype class the labels stay 3-class
    plain = SampleCalls.from_table(_indel_table(), info, VariantType.INDEL, stutter)
    assert plain.calls["label"].tolist() == [1, 0, 2, 2]
    with pytest.raises(ValueError, match="need a stutter model"):
        SampleCalls.from_table(_indel_table(), info, VariantType.INDEL, regenotype=True)
    no_strlen = FeatureTableInfo("s", ("chr1",), (), {"snp": 0, "indel": 3})
    with pytest.raises(ValueError, match="no strlen evidence"):
        SampleCalls.from_table(_indel_table(), no_strlen, VariantType.INDEL, stutter)


def test_wrong_repeat_length_is_relabelled_regenotype() -> None:
    stutter = fit_stutter_model([1], [5], [0], [{-1: 5, 0: 90, 1: 5}])
    info = FeatureTableInfo("s", ("chr1",), (), {"snp": 0, "indel": 2}, strlen=True)
    # a -1 deletion where the truth is a +1 insertion (hap.py: FP, local match -> DROP), and a 0/0
    # FP the length caller also calls 0/1 +1 but with no truth record at its position
    table = (
        _indel_table()
        .iloc[[1, 1]]
        .assign(
            pos=[500, 600],
            sl_tract_start=[501.0, 601.0],
            sl_histogram=["0:10;1:10", "0:10;1:10"],
            truth_delta_lo=[0.0, np.nan],
            truth_delta_hi=[1.0, np.nan],
        )
    )
    calls = SampleCalls.from_table(table, info, VariantType.INDEL, stutter, regenotype=True)
    assert calls.calls["regenotype_ok"].tolist() == [True, False]
    assert calls.calls["label"].tolist() == [Label.REGENOTYPE, Label.DROP]
