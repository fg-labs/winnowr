import numpy as np
import pandas as pd
import pytest

from winnowr.neighbourhood import LOW_AB
from winnowr.neighbourhood import NEIGHBOURHOOD_FEATURES
from winnowr.neighbourhood import neighbourhood_features


def _records(rows: list[tuple[str, int, bool, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["chrom", "pos", "het_any", "ab", "filtered"])


def test_window_boundaries_are_inclusive() -> None:
    # neighbours of 1000 at exactly -500 and +500 count; at +501 they do not (for the 500 window)
    records = _records([
        ("chr1", 500, True, 0.2, False),
        ("chr1", 1000, True, 0.5, False),
        ("chr1", 1500, False, 1.0, True),
        ("chr1", 3001, True, 0.4, False),
    ])
    nb = neighbourhood_features(records).set_index("pos")
    row = nb.loc[1000]
    assert row["nb_calls_500"] == 2
    assert row["nb_het_500"] == 1
    assert row["nb_lowab_500"] == 1
    assert row["nb_filt_500"] == 1
    assert row["nb_ab_500"] == pytest.approx(0.2)
    assert row["nb_calls_2k"] == 2  # 3001 is 2001 away
    assert row["nb_dist"] == 500
    far = nb.loc[3001]
    assert far["nb_calls_500"] == 0 and np.isnan(nb["nb_ab_500"].to_numpy()[-1])
    assert far["nb_calls_2k"] == 1  # 1500 is 1501 away
    assert far["nb_dist"] == 1501


def test_mean_ab_ignores_hets_without_ab_and_own_site() -> None:
    records = _records([
        ("chr1", 100, True, 0.9, False),
        ("chr1", 200, True, float("nan"), False),
        ("chr1", 300, True, 0.1, False),
    ])
    nb = neighbourhood_features(records).set_index("pos")
    assert nb.loc[200, "nb_ab_500"] == pytest.approx(0.5)
    assert nb.loc[100, "nb_het_500"] == 2
    assert nb.loc[100, "nb_ab_500"] == pytest.approx(0.1)


def test_contigs_are_independent_and_duplicates_collapsed() -> None:
    records = _records([
        ("chr1", 100, True, 0.5, False),
        ("chr1", 100, False, 1.0, True),
        ("chr2", 120, True, 0.5, False),
    ])
    nb = neighbourhood_features(records)
    assert list(nb.columns) == ["chrom", "pos", *NEIGHBOURHOOD_FEATURES]
    assert len(nb) == 2
    assert nb["nb_calls_2k"].tolist() == [0, 0]
    assert nb["nb_dist"].isna().all()


def test_empty_input() -> None:
    assert neighbourhood_features(_records([])).empty


def test_low_ab_is_strictly_below_the_cutoff() -> None:
    records = _records([("chr1", 1000, True, 0.5, False), ("chr1", 1200, True, LOW_AB, False)])
    row = neighbourhood_features(records).set_index("pos").loc[1000]
    assert (row["nb_het_500"], row["nb_lowab_500"]) == (1, 0)
