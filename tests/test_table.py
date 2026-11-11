from pathlib import Path

import pandas as pd
import pytest

from winnowr.common import VariantType
from winnowr.table import FeatureTableInfo
from winnowr.table import read_feature_table
from winnowr.table import write_feature_table


def test_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"chrom": ["chr1"], "pos": [5], "qual": [1.5]})
    info = FeatureTableInfo("HG001", ("chr1",), ("seg",), {"snp": 3, "indel": 1}, strlen=True)
    write_feature_table(tmp_path / "t.parquet", frame, info)
    got_frame, got_info = read_feature_table(tmp_path / "t.parquet")
    pd.testing.assert_frame_equal(got_frame, frame)
    assert got_info == info
    assert got_info.is_labelled and got_info.fn_count(VariantType.INDEL) == 1


def test_unlabelled_table_has_no_fn_counts() -> None:
    info = FeatureTableInfo("s", ("chr1",), ())
    assert not info.is_labelled
    with pytest.raises(ValueError, match="no hap.py labels"):
        info.fn_count(VariantType.SNP)


def test_reading_a_foreign_parquet_is_an_error(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="not a winnowr feature table"):
        read_feature_table(tmp_path / "x.parquet")
