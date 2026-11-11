"""Per-sample feature tables stored as Parquet, with winnowr metadata in the schema."""

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from winnowr.common import VariantType

_METADATA_KEY: Final[bytes] = b"winnowr"


@dataclass(frozen=True)
class FeatureTableInfo:
    """
    Metadata stored alongside a feature table.

    Attributes:
        sample: the sample name.
        contigs: the contigs features were extracted from.
        region_names: the names of the region sets flagged (`region_<name>` columns).
        fn_counts: truth FN counts per variant type (`snp`, `indel`), when labelled from hap.py.
        promote_labelled: whether non-PASS calls carry promote labels (`promote_label`).
        strlen: whether indel calls carry repeat-length evidence (see
            `winnowr.repeat_features`).
        reference_contigs: the lengths of `contigs` in the VCF header (None for tables written
            before they were recorded).
    """

    sample: str
    contigs: tuple[str, ...]
    region_names: tuple[str, ...]
    fn_counts: dict[str, int] | None = None
    promote_labelled: bool = False
    strlen: bool = False
    reference_contigs: dict[str, int | None] | None = None

    @property
    def is_labelled(self) -> bool:
        """True if the table carries hap.py labels."""
        return self.fn_counts is not None

    def fn_count(self, variant_type: VariantType) -> int:
        """
        The truth FN count for a variant type.

        Args:
            variant_type: the variant type.

        Returns:
            The number of truth FNs of that type.

        Raises:
            ValueError: if the table is not labelled.
        """
        if self.fn_counts is None:
            raise ValueError(f"Feature table for {self.sample} has no hap.py labels")
        return self.fn_counts[variant_type.value]


def write_feature_table(path: Path, frame: pd.DataFrame, info: FeatureTableInfo) -> None:
    """
    Write a feature table to Parquet.

    Args:
        path: the output path.
        frame: the feature table.
        info: the metadata to store.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[_METADATA_KEY] = json.dumps(asdict(info)).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def read_feature_table(path: Path) -> tuple[pd.DataFrame, FeatureTableInfo]:
    """
    Read a feature table written by `write_feature_table()`.

    Args:
        path: the Parquet file.

    Returns:
        The feature table and its metadata.

    Raises:
        ValueError: if the file has no winnowr metadata.
    """
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if _METADATA_KEY not in metadata:
        raise ValueError(f"{path} is not a winnowr feature table (no winnowr metadata)")
    raw = json.loads(metadata[_METADATA_KEY])
    info = FeatureTableInfo(
        sample=raw["sample"],
        contigs=tuple(raw["contigs"]),
        region_names=tuple(raw["region_names"]),
        fn_counts=raw["fn_counts"],
        promote_labelled=raw["promote_labelled"],
        strlen=raw.get("strlen", False),
        reference_contigs=raw.get("reference_contigs"),
    )
    frame: pd.DataFrame = table.to_pandas()
    return frame, info
