"""
Multi-site "neighbourhood" features computed from all calls near each call.

Collapsed or mismapped paralogous sequence shows up as clusters of calls, often low-allele-balance
hets, so the calls around a site carry information about whether the site itself is real.
"""

from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from winnowr.common import FloatArray

WINDOWS: Final[tuple[tuple[int, str], ...]] = ((500, "500"), (2000, "2k"))
"""Half-window sizes (bp) and the suffixes used in feature names."""

LOW_AB: Final[float] = 0.3
"""Hets with allele balance below this are counted as low-AB hets."""

NEIGHBOURHOOD_FEATURES: Final[tuple[str, ...]] = (
    *(f"nb_{k}_{tag}" for _, tag in WINDOWS for k in ("calls", "het", "lowab", "filt", "ab")),
    "nb_dist",
)
"""Feature names, in model order."""


def neighbourhood_features(records: pd.DataFrame) -> pd.DataFrame:
    """
    Compute neighbourhood features for every distinct call position.

    For each position and each window `±W` bp (inclusive), counting only *other* positions:
    `nb_calls_W` calls, `nb_het_W` het calls (GT `0/x`, x > 0), `nb_lowab_W` hets with AB below
    `LOW_AB`, `nb_filt_W` non-PASS calls, and `nb_ab_W` the mean AB of hets with a defined AB
    (NaN if none). `nb_dist` is the distance to the nearest other call on the contig (NaN if none).
    Records sharing a position are collapsed to the first one.

    Args:
        records: all calls (any FILTER, any variant type) with columns `chrom`, `pos`,
            `het_any` (bool), `ab` (float, NaN if undefined) and `filtered` (bool).

    Returns:
        One row per distinct `(chrom, pos)` with columns `chrom`, `pos` and
        `NEIGHBOURHOOD_FEATURES`.
    """
    distinct = records.drop_duplicates(["chrom", "pos"], keep="first")
    frames = [
        _contig_features(contig, group.sort_values("pos", kind="stable"))
        for contig, group in distinct.groupby("chrom", sort=False)
    ]
    if not frames:
        empty = pd.DataFrame(columns=["chrom", "pos", *NEIGHBOURHOOD_FEATURES])
        return empty.astype(
            {"chrom": object, "pos": np.int64} | dict.fromkeys(NEIGHBOURHOOD_FEATURES, np.float64)
        )
    return pd.concat(frames, ignore_index=True)


def _contig_features(contig: object, group: pd.DataFrame) -> pd.DataFrame:
    """Compute neighbourhood features for the sorted, distinct positions of one contig."""
    pos = group["pos"].to_numpy(dtype=np.int64)
    het = group["het_any"].to_numpy(dtype=np.float64)
    ab = group["ab"].to_numpy(dtype=np.float64)
    has_ab = het * ~np.isnan(ab)
    own = {
        "het": het,
        "lowab": het * (np.nan_to_num(ab, nan=1.0) < LOW_AB),
        "filt": group["filtered"].to_numpy(dtype=np.float64),
        "ab_n": has_ab,
        "ab_sum": np.where(has_ab > 0, ab, 0.0),
    }
    cumulative = {key: np.concatenate([[0.0], np.cumsum(values)]) for key, values in own.items()}
    out = pd.DataFrame({"chrom": contig, "pos": pos})
    for half_window, tag in WINDOWS:
        lo = np.searchsorted(pos, pos - half_window, side="left")
        hi = np.searchsorted(pos, pos + half_window, side="right")

        def others(
            key: str, lo: npt.NDArray[np.intp] = lo, hi: npt.NDArray[np.intp] = hi
        ) -> FloatArray:
            windowed: FloatArray = cumulative[key][hi] - cumulative[key][lo] - own[key]
            return windowed

        out[f"nb_calls_{tag}"] = (hi - lo - 1).astype(np.float64)
        for key in ("het", "lowab", "filt"):
            out[f"nb_{key}_{tag}"] = others(key)
        n_ab = others("ab_n")
        out[f"nb_ab_{tag}"] = np.where(n_ab > 0, others("ab_sum") / np.maximum(n_ab, 1), np.nan)
    gaps = np.diff(pos).astype(np.float64)
    left = np.concatenate([[np.inf], gaps])
    right = np.concatenate([gaps, [np.inf]])
    nearest = np.minimum(left, right)
    out["nb_dist"] = np.where(np.isinf(nearest), np.nan, nearest)
    return out[["chrom", "pos", *NEIGHBOURHOOD_FEATURES]]
