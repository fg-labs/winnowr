"""
A repeat-length stutter model, a stutter-aware diploid length caller, and its features.

The stutter model gives `P(d | t)`: the probability that a read spanning a repeat tract shows a
length difference `d` (bp, observed minus reference tract) when the true allele's length change
is `t`. The error `e = d - t` is modelled as an empirical distribution over `e / k` in
`-MAX_UNITS..MAX_UNITS` (`k` = repeat period) plus an "other" bucket (not a whole number of units,
or more than `MAX_UNITS` of them; spread evenly over `N_OTHER` values), per stratum of period
class (1..4, 5+) and allele tract length bin (`LENGTH_BINS`). It is fitted from the reads at
truth-homozygous calls of the training samples (one true allele), with each stratum smoothed
towards its period class.

The length caller considers the candidate allele length changes `{0}`, the called alleles, and
the observed differences seen in at least two reads (the five most common), and picks the
unordered pair `(a1, a2)` maximising `sum_d n_d log(P(d | a1) / 2 + P(d | a2) / 2)`. Its call and
log-likelihood margins (`LENGTH_CALL_FEATURES`) are model features, and it proposes the genotype
of the re-genotype action (`Label.REGENOTYPE`).
"""

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from winnowr.common import FloatArray
from winnowr.labels import TRUTH_DELTA_COLUMNS
from winnowr.repeat_features import called_deltas_of
from winnowr.repeat_features import ranked_lengths
from winnowr.strlen import parse_histogram
from winnowr.table import FeatureTableInfo

logger = logging.getLogger(__name__)

LENGTH_BINS: Final[tuple[int, ...]] = (0, 5, 8, 11, 14, 17, 20, 25, 30, 40)
"""Lower bounds (bp) of the allele tract length bins."""

MAX_UNITS: Final[int] = 3
"""Errors of up to this many repeat units each way have their own probability."""

N_OTHER: Final[float] = 20.0
"""The "other" error bucket is spread over this many values."""

N_PERIOD_CLASSES: Final[int] = 5
"""Period classes 1, 2, 3, 4 and 5+."""

STRATUM_PSEUDOCOUNT: Final[float] = 200.0
"""Pseudo-reads of the period-class distribution added to each stratum."""

MAX_OBSERVED_CANDIDATES: Final[int] = 5
"""At most this many observed length differences are candidate alleles."""

LENGTH_CALL_FEATURES: Final[tuple[str, ...]] = (
    "lc_margin_called",
    "lc_flip_vs_called",
    "lc_margin_2nd",
    "lc_het",
    "lc_is_called",
    "lc_n_new",
    "lc_multiallelic",
)
"""Model features from the length caller."""

LENGTH_CALL_COLUMNS: Final[tuple[str, str]] = ("lc_delta_lo", "lc_delta_hi")
"""The length caller's genotype: the sorted length changes of its two alleles (not features)."""

_N_ERRORS: Final[int] = 2 * MAX_UNITS + 2
"""Error buckets: `-MAX_UNITS..MAX_UNITS` units, then "other"."""


def period_class(period: int) -> int:
    """
    The stutter period class of a repeat period.

    Args:
        period: the repeat period.

    Returns:
        `period` capped at `N_PERIOD_CLASSES`.
    """
    return min(int(period), N_PERIOD_CLASSES)


def length_bin(length: int) -> int:
    """
    The index of the `LENGTH_BINS` bin of an allele tract length.

    Args:
        length: the tract length in bp (negative lengths fall in the first bin).

    Returns:
        The bin index.
    """
    for index in range(len(LENGTH_BINS) - 1):
        if length < LENGTH_BINS[index + 1]:
            return index
    return len(LENGTH_BINS) - 1


def error_bucket(error: int, period: int) -> int:
    """
    The error bucket of a read's length error.

    Args:
        error: the observed minus true length difference (bp).
        period: the repeat period.

    Returns:
        `error / period + MAX_UNITS` for whole units within `MAX_UNITS`, else the "other" bucket.
    """
    if error % period == 0 and abs(error // period) <= MAX_UNITS:
        return error // period + MAX_UNITS
    return _N_ERRORS - 1


@dataclass(frozen=True)
class StutterModel:
    """
    Read length-error probabilities per period class and allele tract length bin.

    Attributes:
        probabilities: `[period class - 1][length bin]` -> the probability of each error bucket
            (`-MAX_UNITS..MAX_UNITS` units, then "other").
        n_reads: the training reads per `[period class - 1][length bin]`.
    """

    probabilities: tuple[tuple[tuple[float, ...], ...], ...]
    n_reads: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """
        Check the table shapes.

        Raises:
            ValueError: if a table does not have a row per period class and length bin.
        """
        shape_ok = len(self.probabilities) == N_PERIOD_CLASSES and all(
            len(row) == len(LENGTH_BINS) and all(len(p) == _N_ERRORS for p in row)
            for row in self.probabilities
        )
        if not shape_ok or len(self.n_reads) != N_PERIOD_CLASSES:
            raise ValueError("Stutter model tables have the wrong shape")

    def error_probabilities(self, period: int, allele_tract_len: int) -> tuple[float, ...]:
        """
        The error-bucket probabilities of an allele.

        Args:
            period: the repeat period.
            allele_tract_len: the allele's tract length (reference tract plus length change).

        Returns:
            The probability of each error bucket.
        """
        return self.probabilities[period_class(period) - 1][length_bin(max(allele_tract_len, 0))]

    def read_probability(self, observed: int, allele: int, period: int, tract_len: int) -> float:
        """
        `P(d | t)`: the probability of a read's observed length difference given the allele.

        Args:
            observed: the read's length difference `d` (bp, versus the reference tract).
            allele: the allele's length change `t` (bp).
            period: the repeat period.
            tract_len: the reference tract length.

        Returns:
            The probability (an "other" error's bucket probability over `N_OTHER`).
        """
        probabilities = self.error_probabilities(period, tract_len + allele)
        bucket = error_bucket(observed - allele, period)
        if bucket == _N_ERRORS - 1:
            return probabilities[bucket] / N_OTHER
        return probabilities[bucket]

    def to_dict(self) -> dict[str, object]:
        """
        The model as JSON-serializable data.

        Returns:
            The bins, the error range and the tables.
        """
        return {
            "length_bins": list(LENGTH_BINS),
            "max_units": MAX_UNITS,
            "probabilities": [[list(p) for p in row] for row in self.probabilities],
            "n_reads": [list(row) for row in self.n_reads],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "StutterModel":
        """
        Rebuild a model from `to_dict()` data.

        Args:
            raw: the data.

        Returns:
            The model.

        Raises:
            ValueError: if the data was written with other length bins or error range.
        """
        if raw.get("length_bins") != list(LENGTH_BINS) or raw.get("max_units") != MAX_UNITS:
            raise ValueError("Stutter model was fitted with other length bins or error range")
        probabilities = raw["probabilities"]
        n_reads = raw["n_reads"]
        assert isinstance(probabilities, list) and isinstance(n_reads, list)
        return cls(
            probabilities=tuple(
                tuple(tuple(float(x) for x in p) for p in row) for row in probabilities
            ),
            n_reads=tuple(tuple(int(x) for x in row) for row in n_reads),
        )


def fit_stutter_model(
    periods: Sequence[int],
    tract_lens: Sequence[int],
    true_alleles: Sequence[int],
    histograms: Sequence[dict[int, int]],
) -> StutterModel:
    """
    Fit a stutter model from reads at sites with one true allele (homozygous truth).

    Each stratum's counts get `STRATUM_PSEUDOCOUNT` pseudo-reads distributed as its period class
    (all length bins pooled, plus one pseudo-read per bucket).

    Args:
        periods: the repeat period of each site.
        tract_lens: the reference tract length of each site.
        true_alleles: the true allele's length change at each site.
        histograms: the reads per observed length difference at each site.

    Returns:
        The fitted model.
    """
    counts: defaultdict[tuple[int, int], FloatArray] = defaultdict(lambda: np.zeros(_N_ERRORS))
    for period, tract_len, allele, histogram in zip(
        periods, tract_lens, true_alleles, histograms, strict=True
    ):
        stratum = counts[(period_class(period), length_bin(tract_len + allele))]
        for observed, n in histogram.items():
            stratum[error_bucket(observed - allele, period)] += n
    probabilities = []
    n_reads = []
    for kc in range(1, N_PERIOD_CLASSES + 1):
        pooled = sum((v for (c, _), v in counts.items() if c == kc), np.zeros(_N_ERRORS)) + 1.0
        base = pooled / pooled.sum()
        rows = []
        for lb in range(len(LENGTH_BINS)):
            smoothed = counts.get((kc, lb), np.zeros(_N_ERRORS)) + STRATUM_PSEUDOCOUNT * base
            rows.append(tuple(float(x) for x in smoothed / smoothed.sum()))
        probabilities.append(tuple(rows))
        n_reads.append(
            tuple(int(counts.get((kc, lb), np.zeros(1)).sum()) for lb in range(len(LENGTH_BINS)))
        )
    return StutterModel(probabilities=tuple(probabilities), n_reads=tuple(n_reads))


def fit_stutter_from_tables(frames: Sequence[pd.DataFrame]) -> StutterModel:
    """
    Fit a stutter model from labelled feature tables with strlen evidence.

    Uses one row per `(chrom, pos)` among calls (any FILTER) with spanning reads whose truth
    genotype is homozygous (`TRUTH_DELTA_COLUMNS` equal).

    Args:
        frames: labelled feature tables with strlen evidence (see `winnowr.repeat_features`).

    Returns:
        The fitted model.

    Raises:
        ValueError: if a table lacks strlen evidence or truth allele length changes.
    """
    periods: list[int] = []
    tract_lens: list[int] = []
    alleles: list[int] = []
    histograms: list[dict[int, int]] = []
    for frame in frames:
        missing = {"sl_histogram", *TRUTH_DELTA_COLUMNS} - set(frame.columns)
        if missing:
            raise ValueError(f"Feature table lacks columns to fit stutter rates: {sorted(missing)}")
        lo, hi = (frame[c] for c in TRUTH_DELTA_COLUMNS)
        usable = frame[(frame["sl_n_spanning"] > 0) & lo.notna() & (lo == hi)]
        usable = usable.drop_duplicates(["chrom", "pos"])
        periods.extend(usable["sl_period"].astype(int))
        tract_lens.extend(usable["sl_tract_len"].astype(int))
        alleles.extend(usable[TRUTH_DELTA_COLUMNS[0]].astype(int))
        histograms.extend(parse_histogram(text) for text in usable["sl_histogram"])
    return fit_stutter_model(periods, tract_lens, alleles, histograms)


def fit_stutter_from_feature_tables(
    tables: Sequence[tuple[pd.DataFrame, FeatureTableInfo]],
) -> StutterModel:
    """
    Fit the stutter model of a model with repeat-length features from its training tables.

    Args:
        tables: the training feature tables and their metadata.

    Returns:
        The stutter model fitted on the tables' truth-homozygous calls with strlen evidence (see
        `fit_stutter_from_tables`).

    Raises:
        ValueError: if a table is unlabelled or has no strlen evidence, or no read is usable.
    """
    lacking = [info.sample for _, info in tables if not info.strlen or not info.is_labelled]
    if lacking:
        raise ValueError(
            "Repeat-length features need labelled tables with strlen evidence (extract --happy "
            f"--strlen): {lacking}"
        )
    stutter = fit_stutter_from_tables([frame for frame, _ in tables])
    n_reads = sum(sum(row) for row in stutter.n_reads)
    if n_reads == 0:
        raise ValueError(
            "No reads to fit the stutter model: no indel call with spanning strlen reads has a "
            f"homozygous truth genotype in {[info.sample for _, info in tables]}"
        )
    logger.info(f"Fitted the stutter model on {n_reads:,} reads at truth-homozygous calls")
    return stutter


@dataclass(frozen=True)
class LengthCall:
    """
    The length caller's result at one site.

    Attributes:
        alleles: the best genotype's sorted allele length changes.
        log_likelihood: its log-likelihood.
        called_log_likelihood: the called genotype's (NaN if unknown or not a candidate).
        flipped_log_likelihood: the het<->hom flip of the called genotype's (NaN if none).
        second_log_likelihood: the second-best genotype's (NaN if there is only one).
    """

    alleles: tuple[int, int]
    log_likelihood: float
    called_log_likelihood: float
    flipped_log_likelihood: float
    second_log_likelihood: float


def _flipped(called: tuple[int, int] | None) -> tuple[int, int] | None:
    """The het<->hom flip of sorted length changes (`0/x` <-> `x/x`, sorted), if any."""
    if called is None:
        return None
    low, high = called
    if low == high:
        return None if low == 0 else (min(0, low), max(0, low))
    if 0 in called:
        alt = high if low == 0 else low
        return alt, alt
    return None


def call_length_genotype(
    histogram: dict[int, int],
    period: int,
    tract_len: int,
    called: tuple[int, int] | None,
    model: StutterModel,
) -> LengthCall:
    """
    The most likely diploid genotype of a repeat tract given its reads.

    Args:
        histogram: reads per observed length difference (non-empty).
        period: the repeat period.
        tract_len: the reference tract length.
        called: the called genotype's sorted allele length changes, if known.
        model: the stutter model.

    Returns:
        The best genotype and the log-likelihoods of it, the called genotype, its flip and the
        second-best genotype. Ties go to the genotype with the smaller length changes.
    """
    observed = [d for d, n in ranked_lengths(histogram)[:MAX_OBSERVED_CANDIDATES] if n >= 2]
    candidates = sorted({0, *(called or ()), *observed})
    read_probabilities = {
        allele: {d: model.read_probability(d, allele, period, tract_len) for d in histogram}
        for allele in candidates
    }
    log_likelihoods: dict[tuple[int, int], float] = {}
    for i, first in enumerate(candidates):
        for second in candidates[i:]:
            p_first, p_second = read_probabilities[first], read_probabilities[second]
            log_likelihoods[(first, second)] = sum(
                n * math.log(0.5 * p_first[d] + 0.5 * p_second[d]) for d, n in histogram.items()
            )
    ranked = sorted(log_likelihoods.items(), key=lambda item: (-item[1], item[0]))
    flipped = _flipped(called)
    return LengthCall(
        alleles=ranked[0][0],
        log_likelihood=ranked[0][1],
        called_log_likelihood=math.nan if called is None else log_likelihoods[called],
        flipped_log_likelihood=math.nan
        if flipped is None
        else log_likelihoods.get(flipped, math.nan),
        second_log_likelihood=ranked[1][1] if len(ranked) > 1 else math.nan,
    )


def length_call_features(
    histogram: dict[int, int],
    period: int,
    tract_len: int,
    called: tuple[int, int] | None,
    model: StutterModel,
) -> dict[str, float]:
    """
    The length-caller features and genotype of one site.

    Args:
        histogram: reads per observed length difference.
        period: the repeat period.
        tract_len: the reference tract length.
        called: the called genotype's sorted allele length changes, if known.
        model: the stutter model.

    Returns:
        `LENGTH_CALL_FEATURES` and `LENGTH_CALL_COLUMNS` (empty for an empty histogram).
    """
    if not histogram:
        return {}
    call = call_length_genotype(histogram, period, tract_len, called, model)
    low, high = call.alleles
    return {
        "lc_delta_lo": float(low),
        "lc_delta_hi": float(high),
        "lc_margin_called": call.log_likelihood - call.called_log_likelihood,
        "lc_flip_vs_called": call.flipped_log_likelihood - call.called_log_likelihood,
        "lc_margin_2nd": call.log_likelihood - call.second_log_likelihood,
        "lc_het": float(low != high),
        "lc_is_called": float(called is not None and call.alleles == called),
        "lc_n_new": math.nan if called is None else float(len(set(call.alleles) - set(called))),
        "lc_multiallelic": float(low != 0 and high != 0 and low != high),
    }


def add_length_call_features(frame: pd.DataFrame, model: StutterModel) -> pd.DataFrame:
    """
    Add the length caller's features and genotype to calls with strlen evidence.

    Args:
        frame: a feature table with strlen evidence (see `winnowr.repeat_features`).
        model: the stutter model.

    Returns:
        A new feature table with `LENGTH_CALL_FEATURES` and `LENGTH_CALL_COLUMNS` (missing where
        there are no spanning reads).

    Raises:
        ValueError: if the table has no strlen evidence.
    """
    if "sl_histogram" not in frame.columns:
        raise ValueError("Feature table has no strlen evidence (extract with --strlen)")
    out = frame.copy()
    columns = (*LENGTH_CALL_COLUMNS, *LENGTH_CALL_FEATURES)
    values = {name: np.full(len(out), np.nan) for name in columns}
    has_reads = (out["sl_n_spanning"] > 0).to_numpy()
    rows = np.flatnonzero(has_reads)
    subset = out.iloc[rows]
    called = called_deltas_of(subset)
    for row, text, period, tract_len, call in zip(
        rows,
        subset["sl_histogram"],
        subset["sl_period"],
        subset["sl_tract_len"],
        called,
        strict=True,
    ):
        features = length_call_features(
            parse_histogram(text), int(period), int(tract_len), call, model
        )
        for name, value in features.items():
            values[name][row] = value
    for name in columns:
        out[name] = values[name]
    return out
