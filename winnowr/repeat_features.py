"""
Repeat-length features: the strlen read evidence joined to the calls, and histogram summaries.

`add_strlen_features` joins a strlen table (see `winnowr.strlen`) to a feature table by
`(chrom, pos, ref, alt)` and derives `STRLEN_FEATURES` from each call's length histogram and its
called genotype. Calls without strlen evidence (SNPs, or indels outside the measured shards) get
missing values. Features that need a stutter model live in `winnowr.stutter`.
"""

import logging
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from winnowr.alleles import allele_deltas
from winnowr.common import VariantType
from winnowr.strlen import parse_histogram

logger = logging.getLogger(__name__)

STRLEN_COLUMN_NAMES: Final[dict[str, str]] = {
    "period": "sl_period",
    "tract_len": "sl_tract_len",
    "unit": "sl_unit",
    "tract_start": "sl_tract_start",
    "n_overlap": "sl_n_overlap",
    "n_spanning": "sl_n_spanning",
    "length_histogram": "sl_histogram",
    "mean_mapq": "sl_mean_mapq",
    "frac_mapq_lt20": "sl_frac_mapq_lt20",
    "mean_tract_bq": "sl_mean_tract_bq",
}
"""Feature-table names of the strlen table columns joined to each call."""

_TABLE_FEATURES: Final[tuple[str, ...]] = (
    "sl_period",
    "sl_tract_len",
    "sl_tract_units",
    "sl_n_overlap",
    "sl_n_spanning",
    "sl_frac_not_spanning",
    "sl_mean_mapq",
    "sl_frac_mapq_lt20",
    "sl_mean_tract_bq",
)
"""The features taken or derived directly from the strlen table columns."""

_HISTOGRAM_FEATURES: Final[tuple[str, ...]] = (
    "sl_top1_delta",
    "sl_top1_frac",
    "sl_top2_delta",
    "sl_top2_frac",
    "sl_top3_frac",
    "sl_n_lengths",
    "sl_frac_ref",
    "sl_entropy",
    "sl_sd",
    "sl_frac_alt1",
    "sl_frac_alt2",
    "sl_frac_called",
    "sl_top1_vs_alt",
    "sl_top2_vs_alt",
)
"""The features computed from each call's length histogram (see `histogram_features`)."""

STRLEN_FEATURES: Final[tuple[str, ...]] = (*_TABLE_FEATURES, *_HISTOGRAM_FEATURES)
"""Model features from the strlen evidence (no stutter model needed), in model order."""


def called_deltas(ref: str, alt: str, gt: str) -> tuple[int, int] | None:
    """
    The sorted length changes of a call's genotype alleles.

    Args:
        ref: the reference allele.
        alt: the comma-joined alternate alleles.
        gt: the normalized genotype.

    Returns:
        The two length changes, or None (see `allele_deltas`).
    """
    return allele_deltas((ref, *alt.split(",")), gt)


def called_deltas_of(frame: pd.DataFrame) -> list[tuple[int, int] | None]:
    """
    The `called_deltas` of every call in a feature table.

    Args:
        frame: a feature table with `ref`, `alt` and `gt`.

    Returns:
        One entry per row.
    """
    return [
        called_deltas(ref, alt, gt)
        for ref, alt, gt in zip(frame["ref"], frame["alt"], frame["gt"], strict=True)
    ]


def ranked_lengths(histogram: dict[int, int]) -> list[tuple[int, int]]:
    """
    The observed length differences by decreasing read count (ties by increasing difference).

    Args:
        histogram: reads per length difference.

    Returns:
        `(difference, count)` pairs.
    """
    return sorted(histogram.items(), key=lambda item: (-item[1], item[0]))


def histogram_features(
    histogram: dict[int, int], period: int, called: tuple[int, int] | None
) -> dict[str, float]:
    """
    Summaries of one call's length histogram.

    Args:
        histogram: reads per length difference (bp, observed minus reference tract).
        period: the tract's repeat period.
        called: the called genotype's length changes, if known.

    Returns:
        The `_HISTOGRAM_FEATURES` that are defined for the call (none for an empty histogram).
    """
    n = sum(histogram.values())
    if n == 0:
        return {}
    ranked = ranked_lengths(histogram)
    features = {
        "sl_top1_delta": float(ranked[0][0]),
        "sl_top1_frac": ranked[0][1] / n,
        "sl_top2_delta": float(ranked[1][0]) if len(ranked) > 1 else math.nan,
        "sl_top2_frac": ranked[1][1] / n if len(ranked) > 1 else 0.0,
        "sl_top3_frac": ranked[2][1] / n if len(ranked) > 2 else 0.0,
        "sl_n_lengths": float(len(ranked)),
        "sl_frac_ref": histogram.get(0, 0) / n,
    }
    fractions = np.array([count / n for _, count in ranked])
    features["sl_entropy"] = float(-(fractions * np.log2(fractions)).sum())
    deltas = np.array([delta for delta, _ in ranked], dtype=np.float64)
    mean = (deltas * fractions).sum()
    features["sl_sd"] = float(math.sqrt(max(((deltas - mean) ** 2 * fractions).sum(), 0)))
    if called is not None:
        alts = sorted({delta for delta in called if delta != 0})
        features["sl_frac_alt1"] = histogram.get(alts[0], 0) / n if alts else math.nan
        features["sl_frac_alt2"] = histogram.get(alts[1], 0) / n if len(alts) > 1 else math.nan
        features["sl_frac_called"] = sum(histogram.get(d, 0) for d in set(called)) / n
        features["sl_top1_vs_alt"] = (ranked[0][0] - alts[0]) / period if alts else math.nan
        features["sl_top2_vs_alt"] = (
            (ranked[1][0] - alts[0]) / period if alts and len(ranked) > 1 else math.nan
        )
    return features


def add_strlen_features(frame: pd.DataFrame, strlen: pd.DataFrame) -> pd.DataFrame:
    """
    Join strlen evidence to the calls and derive `STRLEN_FEATURES`.

    Logs how many indel calls have evidence, and warns about contigs whose indel calls have none
    (e.g. a missing shard). Some indel calls legitimately have none (no pure insertion or deletion
    allele).

    Args:
        frame: a feature table.
        strlen: a strlen table (see `winnowr.strlen.read_strlen_tables`), one row per record.

    Returns:
        A new feature table with the `STRLEN_COLUMN_NAMES` columns and `STRLEN_FEATURES`.

    Raises:
        ValueError: if the table has indel calls and none of them has evidence (the strlen tables
            come from another VCF).
    """
    joined = strlen[["chrom", "pos", "ref", "alt", *STRLEN_COLUMN_NAMES]].rename(
        columns=STRLEN_COLUMN_NAMES
    )
    out = frame.drop(columns=[*STRLEN_COLUMN_NAMES.values(), *STRLEN_FEATURES], errors="ignore")
    out = out.merge(joined, on=["chrom", "pos", "ref", "alt"], how="left", validate="many_to_one")
    numeric = [c for c in STRLEN_COLUMN_NAMES.values() if c not in ("sl_unit", "sl_histogram")]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float64)
    n_overlap, n_spanning = out["sl_n_overlap"], out["sl_n_spanning"]
    out["sl_frac_not_spanning"] = 1 - n_spanning / n_overlap.where(n_overlap > 0)
    out["sl_tract_units"] = out["sl_tract_len"] / out["sl_period"].where(out["sl_period"] > 0)
    _check_coverage(out)
    rows = _histogram_rows(out)
    for name in _HISTOGRAM_FEATURES:
        out[name] = np.asarray([row.get(name, math.nan) for row in rows], dtype=np.float64)
    return out


def _check_coverage(frame: pd.DataFrame) -> None:
    """Log the indel calls with strlen evidence; fail if there are indel calls but none has any."""
    is_indel = frame["variant_type"] == VariantType.INDEL.value
    n_indels = int(is_indel.sum())
    if n_indels == 0:
        return
    has_evidence = frame["sl_period"].notna() & is_indel
    n_with = int(has_evidence.sum())
    if n_with == 0:
        raise ValueError(
            f"None of the {n_indels:,} indel calls has a row in the strlen tables: were they made "
            "from this VCF?"
        )
    logger.info(
        f"strlen evidence for {n_with:,} of {n_indels:,} indel calls ({n_with / n_indels:.1%})"
    )
    per_contig = has_evidence[is_indel].groupby(frame["chrom"][is_indel], sort=False).sum()
    missing = [str(contig) for contig, n in per_contig.items() if n == 0]
    if missing:
        logger.warning(
            f"No strlen evidence for the indel calls on {len(missing)} contig(s), e.g. a missing "
            f"shard: {', '.join(missing)}"
        )


def _histogram_rows(frame: pd.DataFrame) -> Sequence[dict[str, float]]:
    """The `histogram_features` of each row (empty where there is no strlen evidence)."""
    called = called_deltas_of(frame)
    return [
        histogram_features(parse_histogram(text), int(period), call) if period == period else {}
        for text, period, call in zip(
            frame["sl_histogram"], frame["sl_period"], called, strict=True
        )
    ]
