"""Tests of the repeat-length features joined from strlen tables."""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from winnowr.repeat_features import STRLEN_FEATURES
from winnowr.repeat_features import add_strlen_features
from winnowr.repeat_features import called_deltas
from winnowr.repeat_features import histogram_features
from winnowr.repeat_features import ranked_lengths
from winnowr.strlen import STRLEN_COLUMNS


def test_called_deltas() -> None:
    assert called_deltas("CAA", "C", "0/1") == (-2, 0)
    assert called_deltas("C", "CA,CAA", "1/2") == (1, 2)
    assert called_deltas("C", "*,CA", "1/2") is None
    assert called_deltas("C", "CA", "./.") is None


def test_histogram_features_of_a_het_deletion() -> None:
    # 6 reads at the reference length, 3 two bases shorter, 1 one base longer
    features = histogram_features({-2: 3, 0: 6, 1: 1}, period=2, called=(-2, 0))
    assert features["sl_top1_delta"] == 0
    assert features["sl_top1_frac"] == pytest.approx(0.6)
    assert features["sl_top2_delta"] == -2
    assert features["sl_top2_frac"] == pytest.approx(0.3)
    assert features["sl_top3_frac"] == pytest.approx(0.1)
    assert features["sl_n_lengths"] == 3
    assert features["sl_frac_ref"] == pytest.approx(0.6)
    fractions = np.array([0.6, 0.3, 0.1])
    assert features["sl_entropy"] == pytest.approx(-(fractions * np.log2(fractions)).sum())
    mean = 0 * 0.6 - 2 * 0.3 + 1 * 0.1
    sd = math.sqrt(0.6 * mean**2 + 0.3 * (-2 - mean) ** 2 + 0.1 * (1 - mean) ** 2)
    assert features["sl_sd"] == pytest.approx(sd)
    assert features["sl_frac_alt1"] == pytest.approx(0.3)
    assert math.isnan(features["sl_frac_alt2"])
    assert features["sl_frac_called"] == pytest.approx(0.9)
    assert features["sl_top1_vs_alt"] == pytest.approx(1.0)  # (0 - -2) / period 2
    assert features["sl_top2_vs_alt"] == pytest.approx(0.0)


def test_histogram_features_edge_cases() -> None:
    assert histogram_features({}, period=1, called=(0, 1)) == {}
    single = histogram_features({1: 4}, period=1, called=None)
    assert math.isnan(single["sl_top2_delta"])
    assert (single["sl_top2_frac"], single["sl_top3_frac"], single["sl_entropy"]) == (0, 0, 0)
    assert "sl_frac_alt1" not in single
    # a 0/0 call has no alternate allele
    ref_call = histogram_features({0: 4}, period=1, called=(0, 0))
    assert math.isnan(ref_call["sl_frac_alt1"])
    assert ref_call["sl_frac_called"] == 1


def _strlen_table() -> pd.DataFrame:
    rows = [
        ("chr1", 100, "CA", "C", 1, 10, "A", 101, 20, 10, "-1:4;0:6", 60.0, 0.0, 35.0),
        ("chr1", 200, "C", "CTT", 2, 0, "TT", 201, 5, 0, ".", math.nan, math.nan, math.nan),
    ]
    return pd.DataFrame(rows, columns=list(STRLEN_COLUMNS))


def test_add_strlen_features_joins_by_record() -> None:
    frame = pd.DataFrame({
        "chrom": ["chr1"] * 4,
        "pos": [100, 100, 200, 300],
        "ref": ["CA", "C", "C", "A"],
        "alt": ["C", "T", "CTT", "G"],
        "gt": ["0/1", "0/1", "1/1", "0/1"],
        "variant_type": ["indel", "snp", "indel", "snp"],
    })
    out = add_strlen_features(frame, _strlen_table())
    assert set(STRLEN_FEATURES) <= set(out.columns)
    assert out["sl_period"].isna().tolist() == [False, True, False, True]
    first = out.iloc[0]
    assert (first["sl_tract_len"], first["sl_tract_units"]) == (10, 10)
    assert first["sl_frac_not_spanning"] == pytest.approx(0.5)
    assert first["sl_frac_alt1"] == pytest.approx(0.4)
    assert first["sl_histogram"] == "-1:4;0:6"
    # no spanning reads: table values only, histogram features missing
    empty = out.iloc[2]
    assert empty["sl_frac_not_spanning"] == 1
    assert empty["sl_tract_units"] == 0
    assert math.isnan(empty["sl_top1_frac"])
    # joining twice replaces rather than duplicates the columns
    again = add_strlen_features(out, _strlen_table())
    assert list(again.columns) == list(out.columns)


def test_add_strlen_features_reports_coverage(caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame({
        "chrom": ["chr1", "chr2", "chr2"],
        "pos": [100, 100, 200],
        "ref": ["CA", "CA", "C"],
        "alt": ["C", "C", "CA"],
        "gt": "0/1",
        "variant_type": "indel",
    })
    with caplog.at_level(logging.INFO):
        add_strlen_features(frame, _strlen_table())
    assert "strlen evidence for 1 of 3 indel calls" in caplog.text
    assert "on 1 contig(s), e.g. a missing shard: chr2" in caplog.text
    # strlen tables from another VCF match no indel call
    with pytest.raises(ValueError, match="None of the 2 indel calls"):
        add_strlen_features(frame.iloc[1:], _strlen_table())
    # SNP-only tables need no evidence
    snps = frame.assign(variant_type="snp", ref="A", alt="G")
    assert add_strlen_features(snps, _strlen_table())["sl_period"].isna().all()


def test_ranked_lengths_breaks_ties_by_increasing_difference() -> None:
    assert ranked_lengths({0: 5, 2: 7, -1: 5}) == [(2, 7), (-1, 5), (0, 5)]
    features = histogram_features({0: 5, -1: 5}, 1, (-1, 0))
    assert (features["sl_top1_delta"], features["sl_top2_delta"]) == (-1, 0)
