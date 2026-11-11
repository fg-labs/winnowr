"""
The re-genotype action: which calls it can change, whether the change is right, and the new record.

A re-genotype replaces a call's genotype with the repeat-length caller's (see `winnowr.stutter`).
`with_regenotype_columns` flags the calls it applies to and, for labelled calls, whether it gives
the truth genotype; `regenotyped_alleles` builds the alleles and genotype written by `score`.
"""

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from winnowr.common import BoolArray
from winnowr.labels import TRUTH_DELTA_COLUMNS
from winnowr.labels import Label
from winnowr.strlen import RepeatTract


def regenotypable(frame: pd.DataFrame) -> BoolArray:
    """
    Calls the re-genotype action can change: the length caller disagrees with the call.

    Args:
        frame: calls with length-call features.

    Returns:
        Per call, whether the length caller made a call that differs from the called genotype.
    """
    return (frame["lc_delta_lo"].notna() & (frame["lc_is_called"] == 0)).to_numpy(dtype=np.bool_)


def regenotypes_to_ref(frame: pd.DataFrame) -> BoolArray:
    """
    Calls whose length-caller genotype is homozygous reference (re-genotyping them drops them).

    Args:
        frame: calls with length-call features.

    Returns:
        Per call, whether the length caller called `0/0`.
    """
    return ((frame["lc_delta_lo"] == 0) & (frame["lc_delta_hi"] == 0)).to_numpy(dtype=np.bool_)


def length_call_matches_truth(frame: pd.DataFrame) -> BoolArray:
    """
    Calls whose length-caller genotype has the truth alleles' length changes.

    A match by length change counts as the truth genotype; in a pure repeat tract alleles of the
    same length have the same sequence.

    Args:
        frame: labelled calls with length-call features.

    Returns:
        Per call, whether the length caller's genotype equals the truth's length changes.
    """
    lo, hi = TRUTH_DELTA_COLUMNS
    matches = (frame["lc_delta_lo"] == frame[lo]) & (frame["lc_delta_hi"] == frame[hi])
    return matches.to_numpy(dtype=np.bool_)


REGENOTYPE_COLUMNS: Final[tuple[str, str, str]] = (
    "regenotypable",
    "regenotypes_to_ref",
    "regenotype_ok",
)
"""Per-call re-genotype columns added by `with_regenotype_columns` (not features)."""


def regenotype_buildable(frame: pd.DataFrame) -> BoolArray:
    """
    Calls whose length-caller genotype can be written to the VCF (see `regenotyped_alleles`).

    Args:
        frame: calls with length-call features and the strlen tract columns (`sl_tract_start`,
            `sl_tract_len`, `sl_unit`).

    Returns:
        Per call, whether a length call exists and its alleles can be found or built.
    """
    buildable = np.zeros(len(frame), dtype=np.bool_)
    has_call = frame["lc_delta_lo"].notna().to_numpy()
    rows = np.flatnonzero(has_call)
    subset = frame.iloc[rows]
    for row, values in zip(
        rows,
        zip(
            subset["pos"],
            subset["ref"],
            subset["alt"],
            subset["lc_delta_lo"],
            subset["lc_delta_hi"],
            subset["sl_tract_start"],
            subset["sl_tract_len"],
            subset["sl_unit"],
            strict=True,
        ),
        strict=True,
    ):
        pos, ref, alt, lo, hi, tract_start, tract_len, unit = values
        target = regenotype_target(pos, ref, alt, (lo, hi), tract_start, tract_len, unit)
        buildable[row] = target is not None
    return buildable


def without_regenotype(calls: pd.DataFrame) -> pd.DataFrame:
    """
    Add the re-genotype columns, all false, to calls of a model without the re-genotype action.

    Args:
        calls: the calls.

    Returns:
        A copy of the calls with `REGENOTYPE_COLUMNS`, all false.
    """
    return calls.assign(**dict.fromkeys(REGENOTYPE_COLUMNS, False))


def with_regenotype_columns(calls: pd.DataFrame) -> pd.DataFrame:
    """
    Add the re-genotype columns to calls with length-call features.

    `regenotypable` marks the calls the length caller disagrees with (see `regenotypable`) whose
    new genotype `score` can write (see `regenotype_buildable`), so training, evaluation and
    scoring act on the same calls; `regenotypes_to_ref` comes from `regenotypes_to_ref`.
    `regenotype_ok` (labelled calls only; otherwise false) marks FPs (`Label.FLIP`, an allele
    match, or `Label.DROP`, e.g. a wrong repeat length that hap.py reports as a local match) whose
    length-caller genotype differs from the call, is not `0/0`, and equals the truth (see
    `length_call_matches_truth`); a `DROP` call only qualifies with a truth record of its type at
    its position.

    Args:
        calls: calls with length-call features (and, for `regenotype_ok`, labels and
            `TRUTH_DELTA_COLUMNS`).

    Returns:
        A copy of the calls with `REGENOTYPE_COLUMNS`.
    """
    out = calls.copy()
    to_ref = regenotypes_to_ref(out)
    changes = regenotypable(out) & regenotype_buildable(out)
    out["regenotypable"], out["regenotypes_to_ref"] = changes, to_ref
    labelled = "label" in out.columns and set(TRUTH_DELTA_COLUMNS) <= set(out.columns)
    out["regenotype_ok"] = (
        length_call_matches_truth(out)
        & out["label"].isin([Label.FLIP, Label.DROP]).to_numpy()
        & changes
        & ~to_ref
        if labelled
        else np.zeros(len(out), dtype=np.bool_)
    )
    return out


def regenotyped_alleles(
    pos: int, alleles: Sequence[str], deltas: tuple[int, int], tract: RepeatTract | None
) -> tuple[tuple[str, ...], tuple[int, int]] | None:
    """
    The alleles and genotype of a record re-genotyped to the given allele length changes.

    A length change of 0 is the REF. Any other is the repeat tract resized by that change in its
    phase: an ALT with exactly that sequence is reused, otherwise it is added as a new ALT (an ALT
    of the same length but another sequence, e.g. a different inserted base, is not used). That
    needs a tract starting after the record's first base and no later than the end of its REF;
    when a new ALT is added and the tract extends past the REF, the REF (and every other
    non-symbolic allele) is extended to the end of the tract. Without such a tract, a length change
    can only use the one ALT of that length, if there is exactly one.

    Args:
        pos: the record's 1-based position.
        alleles: the record's REF and ALT alleles.
        deltas: the new genotype's sorted allele length changes.
        tract: the record's repeat tract (from the strlen evidence), if known.

    Returns:
        The (possibly extended) alleles and the new genotype's sorted allele indices, or None if
        an allele cannot be found or built.
    """
    ref = alleles[0]
    start, end = pos - 1, pos - 1 + len(ref)
    index_of_delta: dict[int, int] = {0: 0}
    if tract is None or not start < tract.start <= end:
        for delta in set(deltas) - {0}:
            same_length = [
                index
                for index, allele in enumerate(alleles[1:], start=1)
                if not _is_symbolic(allele) and len(allele) - len(ref) == delta
            ]
            if len(same_length) != 1:
                return None
            index_of_delta[delta] = same_length[0]
        first, second = sorted(index_of_delta[delta] for delta in deltas)
        return tuple(alleles), (first, second)
    if any(tract.length + delta < 0 for delta in deltas):
        return None
    suffix = tract.sequence(tract.length)[end - tract.start :] if tract.end > end else ""
    extended = tuple(a if _is_symbolic(a) else a + suffix for a in alleles)
    head, tail = extended[0][: tract.start - start], extended[0][tract.end - start :]
    new_alleles = extended
    for delta in sorted(set(deltas) - {0}):
        resized = head + tract.sequence(tract.length + delta) + tail
        index = next(
            (
                i
                for i, allele in enumerate(new_alleles)
                if i > 0 and allele.upper() == resized.upper()
            ),
            None,
        )
        if index is None:
            index = len(new_alleles)
            new_alleles += (resized,)
        index_of_delta[delta] = index
    first, second = sorted(index_of_delta[delta] for delta in deltas)
    added = len(new_alleles) > len(alleles)
    return new_alleles if added else tuple(alleles), (first, second)


def _is_symbolic(allele: str) -> bool:
    """True for a spanning deletion (`*`) or a symbolic allele (e.g. `<NON_REF>`)."""
    return allele == "*" or allele.startswith("<")


def regenotype_target(
    pos: int,
    ref: str,
    alt: str,
    deltas: tuple[float, float],
    tract_start: float,
    tract_len: float,
    unit: object,
) -> tuple[tuple[str, ...], tuple[int, int]] | None:
    """
    The new alleles and genotype of a re-genotyped call, from its feature-table values.

    Args:
        pos: the call's 1-based position.
        ref: the reference allele.
        alt: the comma-joined alternate alleles.
        deltas: the length caller's genotype (`lc_delta_lo`, `lc_delta_hi`).
        tract_start: the repeat tract's 1-based start (`sl_tract_start`).
        tract_len: the tract length (`sl_tract_len`).
        unit: the tract's repeat unit (`sl_unit`).

    Returns:
        The alleles and genotype (see `regenotyped_alleles`), or None if a value is missing or the
        new allele cannot be built.
    """
    if any(math.isnan(v) for v in (*deltas, tract_start, tract_len)) or not isinstance(unit, str):
        return None
    start = int(tract_start) - 1
    tract = RepeatTract(start=start, end=start + int(tract_len), period=len(unit), unit=unit)
    low, high = (int(d) for d in deltas)
    return regenotyped_alleles(pos, (ref, *alt.split(",")), (low, high), tract)
