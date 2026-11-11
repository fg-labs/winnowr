"""Tests of the stutter model, the length caller and its features."""

import math

import numpy as np
import pandas as pd
import pytest

from winnowr.labels import Label
from winnowr.regenotype import length_call_matches_truth
from winnowr.regenotype import regenotypable
from winnowr.regenotype import regenotypes_to_ref
from winnowr.regenotype import with_regenotype_columns
from winnowr.stutter import LENGTH_BINS
from winnowr.stutter import LENGTH_CALL_COLUMNS
from winnowr.stutter import LENGTH_CALL_FEATURES
from winnowr.stutter import MAX_UNITS
from winnowr.stutter import StutterModel
from winnowr.stutter import add_length_call_features
from winnowr.stutter import call_length_genotype
from winnowr.stutter import error_bucket
from winnowr.stutter import fit_stutter_from_feature_tables
from winnowr.stutter import fit_stutter_from_tables
from winnowr.stutter import fit_stutter_model
from winnowr.stutter import length_bin
from winnowr.stutter import length_call_features
from winnowr.stutter import period_class
from winnowr.table import FeatureTableInfo

OTHER = 2 * MAX_UNITS + 1


def _model() -> StutterModel:
    """A model fitted to homopolymer reads with 90% exact lengths and 5% one unit either way."""
    histograms = [{-1: 5, 0: 90, 1: 5}] * 50
    return fit_stutter_model([1] * 50, [10] * 50, [0] * 50, histograms)


def test_period_class_and_length_bin() -> None:
    assert [period_class(k) for k in (1, 2, 4, 5, 9)] == [1, 2, 4, 5, 5]
    assert [length_bin(n) for n in (-3, 0, 4, 5, 39, 40, 1000)] == [0, 0, 0, 1, 8, 9, 9]
    assert len(LENGTH_BINS) == 10


def test_error_bucket() -> None:
    assert error_bucket(0, 1) == MAX_UNITS
    assert error_bucket(-2, 2) == MAX_UNITS - 1
    assert error_bucket(6, 2) == 2 * MAX_UNITS
    assert error_bucket(8, 2) == OTHER  # four units
    assert error_bucket(1, 2) == OTHER  # not a whole unit
    assert error_bucket(-7, 2) == OTHER


def test_fit_stutter_model_learns_the_error_rates() -> None:
    model = _model()
    probabilities = model.error_probabilities(1, 10)
    assert sum(probabilities) == pytest.approx(1.0)
    # 5000 reads: 4500 exact, 250 each way, smoothed by 200 pseudo-reads
    assert probabilities[MAX_UNITS] == pytest.approx(0.9, abs=0.01)
    assert probabilities[MAX_UNITS - 1] == pytest.approx(0.05, abs=0.005)
    assert model.n_reads[0][length_bin(10)] == 5000
    # strata without reads fall back to their period class, and other classes are uniform-ish
    assert model.error_probabilities(1, 35)[MAX_UNITS] == pytest.approx(0.9, abs=0.01)
    assert model.error_probabilities(3, 10) == pytest.approx([1 / (OTHER + 1)] * (OTHER + 1))
    # an "other" error is spread over N_OTHER values
    assert model.read_probability(5, 0, 1, 10) == pytest.approx(probabilities[OTHER] / 20)
    # the allele's tract length picks the bin: a +6 allele of a 10 bp tract is in the 14-17 bin
    assert model.read_probability(6, 6, 1, 10) == model.error_probabilities(1, 16)[MAX_UNITS]


def test_stutter_model_round_trip_and_checks() -> None:
    model = _model()
    assert StutterModel.from_dict(model.to_dict()) == model
    raw = model.to_dict() | {"length_bins": [0, 10]}
    with pytest.raises(ValueError, match="other length bins"):
        StutterModel.from_dict(raw)
    with pytest.raises(ValueError, match="wrong shape"):
        StutterModel(probabilities=(), n_reads=())


def test_call_length_genotype() -> None:
    model = _model()
    het = call_length_genotype({-2: 10, -1: 1, 0: 10}, 1, 10, (0, 0), model)
    assert het.alleles == (-2, 0)
    assert het.log_likelihood > het.called_log_likelihood
    assert math.isnan(het.flipped_log_likelihood)  # 0/0 has no flip
    assert het.second_log_likelihood < het.log_likelihood
    hom = call_length_genotype({2: 20, 1: 1}, 1, 10, (0, 2), model)
    assert hom.alleles == (2, 2)
    assert hom.flipped_log_likelihood == hom.log_likelihood
    # one read at +3 is not enough to make it a candidate
    noisy = call_length_genotype({0: 20, 3: 1}, 1, 10, None, model)
    assert noisy.alleles == (0, 0)
    assert math.isnan(noisy.called_log_likelihood)


def test_length_call_features() -> None:
    model = _model()
    assert length_call_features({}, 1, 10, (0, 1), model) == {}
    features = length_call_features({-2: 10, 2: 10}, 1, 10, (-2, 0), model)
    assert set(features) == {*LENGTH_CALL_FEATURES, *LENGTH_CALL_COLUMNS}
    assert (features["lc_delta_lo"], features["lc_delta_hi"]) == (-2, 2)
    assert features["lc_het"] == 1
    assert features["lc_multiallelic"] == 1
    assert features["lc_is_called"] == 0
    assert features["lc_n_new"] == 1
    assert features["lc_margin_called"] > 0
    assert features["lc_margin_2nd"] > 0
    agreeing = length_call_features({0: 10, 1: 10}, 1, 10, (0, 1), model)
    assert (agreeing["lc_is_called"], agreeing["lc_n_new"], agreeing["lc_margin_called"]) == (
        1,
        0,
        0,
    )


def _calls() -> pd.DataFrame:
    return pd.DataFrame({
        "chrom": ["chr1"] * 4,
        "pos": [100, 200, 300, 400],
        "ref": ["CAA", "C", "CA", "A"],
        "alt": ["C", "CA", "C", "G"],
        "gt": ["0/1", "0/1", "1/1", "0/1"],
        "sl_period": [1.0, 1.0, 1.0, np.nan],
        "sl_tract_len": [10.0, 10.0, 10.0, np.nan],
        "sl_tract_start": [101.0, 201.0, 301.0, np.nan],
        "sl_unit": ["A", "A", "A", None],
        "sl_n_spanning": [20.0, 20.0, 0.0, np.nan],
        "sl_histogram": ["-2:10;0:10", "0:20", ".", None],
        "truth_delta_lo": [-2.0, 0.0, np.nan, 0.0],
        "truth_delta_hi": [0.0, 0.0, np.nan, 1.0],
    })


def test_add_length_call_features_and_regenotype_helpers() -> None:
    out = add_length_call_features(_calls(), _model())
    assert out["lc_delta_lo"].isna().tolist() == [False, False, True, True]
    assert out["lc_delta_lo"].tolist()[:2] == [-2, 0]
    assert out["lc_delta_hi"].tolist()[:2] == [0, 0]
    assert regenotypable(out).tolist() == [False, True, False, False]
    assert regenotypes_to_ref(out).tolist() == [False, True, False, False]
    assert length_call_matches_truth(out).tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="no strlen evidence"):
        add_length_call_features(_calls().drop(columns="sl_histogram"), _model())


def test_fit_stutter_from_tables_uses_truth_homozygous_sites() -> None:
    calls = _calls()
    # duplicated position (e.g. a SNP and an indel record) is used once
    frame = pd.concat([calls, calls.iloc[[1]]], ignore_index=True)
    model = fit_stutter_from_tables([frame])
    # only pos 200 qualifies: spanning reads and a homozygous (0/0) truth
    assert model.n_reads[0][length_bin(10)] == 20
    assert sum(sum(row) for row in model.n_reads) == 20
    with pytest.raises(ValueError, match="lacks columns"):
        fit_stutter_from_tables([calls.drop(columns="truth_delta_lo")])


def test_with_regenotype_columns() -> None:
    calls = add_length_call_features(_calls(), _model())
    # pos 100: called 0/1 deletion, length call agrees; pos 200: length call 0/0 (to ref)
    calls["label"] = [Label.FLIP, Label.FLIP, Label.KEEP, Label.DROP]
    out = with_regenotype_columns(calls)
    assert out["regenotypable"].tolist() == [False, True, False, False]
    assert out["regenotypes_to_ref"].tolist() == [False, True, False, False]
    assert not out["regenotype_ok"].any()
    # a 1/2 truth that the length caller gets right and the call does not
    multi = pd.DataFrame({
        "pos": [100],
        "ref": ["CAA"],
        "alt": ["C"],
        "gt": ["0/1"],
        "sl_period": [1.0],
        "sl_tract_len": [10.0],
        "sl_tract_start": [101.0],
        "sl_unit": ["A"],
        "sl_n_spanning": [20.0],
        "sl_histogram": ["-2:10;1:10"],
        "truth_delta_lo": [-2.0],
        "truth_delta_hi": [1.0],
        "label": [float(Label.FLIP)],
    })
    fixable = with_regenotype_columns(add_length_call_features(multi, _model()))
    assert fixable[["regenotypable", "regenotype_ok"]].to_numpy().tolist() == [[True, True]]
    # a length call that score cannot write (the tract starts before the record) is not acted on
    unbuildable = with_regenotype_columns(
        add_length_call_features(multi.assign(sl_tract_start=99.0), _model())
    )
    assert unbuildable[["regenotypable", "regenotype_ok"]].to_numpy().tolist() == [[False, False]]
    # unlabelled calls are never "ok"
    unlabelled = with_regenotype_columns(add_length_call_features(_calls(), _model()))
    assert not unlabelled["regenotype_ok"].any()


def test_flip_log_likelihood_of_deletions() -> None:
    model = _model()
    # a het deletion's flip is the hom deletion, and a hom deletion's flip the het deletion
    het = call_length_genotype({-2: 20}, 1, 10, (-2, 0), model)
    assert het.flipped_log_likelihood == het.log_likelihood
    hom = call_length_genotype({-2: 10, 0: 10}, 1, 10, (-2, -2), model)
    assert hom.flipped_log_likelihood == hom.log_likelihood
    assert math.isnan(call_length_genotype({0: 9}, 1, 10, (-1, 1), model).flipped_log_likelihood)


def test_call_length_genotype_ties_go_to_the_smaller_length_changes() -> None:
    # an untrained model gives every error within 3 units the same probability, so with reads at
    # -1 and +1 all six genotypes of {-1, 0, 1} tie
    uniform = fit_stutter_model([], [], [], [])
    call = call_length_genotype({1: 10, -1: 10}, 1, 10, None, uniform)
    assert call.alleles == (-1, -1)
    assert call.log_likelihood == call.second_log_likelihood


def test_fit_stutter_needs_usable_reads() -> None:
    # labelled indel tables with strlen evidence, but no truth-homozygous call with spanning reads
    frame = pd.DataFrame({
        "chrom": ["chr1"],
        "pos": [100],
        "sl_period": [1.0],
        "sl_tract_len": [5.0],
        "sl_n_spanning": [20.0],
        "sl_histogram": ["0:20"],
        "truth_delta_lo": [0.0],
        "truth_delta_hi": [1.0],
    })
    info = FeatureTableInfo("a", ("chr1",), (), {"snp": 0, "indel": 0}, strlen=True)
    with pytest.raises(ValueError, match=r"No reads to fit the stutter model.*\['a', 'a'\]"):
        fit_stutter_from_feature_tables([(frame, info), (frame, info)])
    unlabelled = FeatureTableInfo("b", ("chr1",), (), strlen=True)
    with pytest.raises(ValueError, match=r"labelled tables with strlen evidence.*\['b'\]"):
        fit_stutter_from_feature_tables([(frame, info), (frame, unlabelled)])
