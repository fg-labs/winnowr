"""
Training labels from hap.py (vcfeval engine) output.

Each hap.py record has TRUTH and QUERY samples with FORMAT `BD` (decision: TP/FP/FN/UNK/N),
`BK` (match kind: `gm` genotype match, `am` allele match, `lm` local match, `.` none) and `BVT`
(SNP/INDEL/NOCALL). For each query call:

- TP -> `KEEP`
- FP with BK `.` (no truth variant nearby) or `lm` (a truth variant nearby, but not with these
  alleles) -> `DROP`: of the actions besides re-genotyping, dropping is the one that helps F1
- FP with BK `am` -> `FLIP` (right allele, wrong genotype). The flip is *correct* (`flip_ok`) only
  if switching the query genotype between `0/1` and `1/1` yields the truth genotype.

Under vcfeval semantics a genotype mismatch is both a query FP and a truth FN, so the number of
truth FNs is needed to compute F1; it is counted from the TRUTH sample (BD = FN).

Every call (any FILTER) also gets the length changes of its two truth alleles
(`TRUTH_DELTA_COLUMNS`) from the first hap.py record at its position with a diploid TRUTH
genotype and a TRUTH BVT of the call's variant type (so a truth SNP never gives an indel call
its length changes); the repeat-length model uses them to fit stutter rates and to label
re-genotyping (see `winnowr.stutter`).
"""

import logging
import math
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import pysam

from winnowr.alleles import FLIPPED_GT
from winnowr.alleles import allele_deltas
from winnowr.alleles import normalize_gt
from winnowr.common import AUTOSOMES
from winnowr.common import PASS
from winnowr.common import VariantType

logger = logging.getLogger(__name__)


class Label(IntEnum):
    """
    The genotype-model classes.

    `REGENOTYPE` is only used by models trained with the re-genotype action (see
    `winnowr.regenotype`): the call has the wrong genotype or alleles (e.g. a wrong repeat length,
    which hap.py reports as a local match), a het<->hom flip does not fix it, but the repeat-length
    caller's genotype does.
    """

    DROP = 0
    KEEP = 1
    FLIP = 2
    REGENOTYPE = 3


TRUTH_DELTA_COLUMNS: Final[tuple[str, str]] = ("truth_delta_lo", "truth_delta_hi")
"""Length changes (bp, ALT minus REF, sorted) of the two truth alleles at a call's position."""


_BVT_TO_TYPE: Final[dict[str, VariantType]] = {"SNP": VariantType.SNP, "INDEL": VariantType.INDEL}


@dataclass(frozen=True)
class HappyRecord:
    """
    The fields of one hap.py output record that labelling needs.

    Attributes:
        chrom: the contig.
        pos: the 1-based position.
        truth_gt: the normalized TRUTH genotype (see `normalize_gt`).
        truth_bd: the TRUTH BD.
        truth_bvt: the TRUTH BVT.
        query_gt: the normalized QUERY genotype.
        query_bd: the QUERY BD.
        query_bk: the QUERY BK.
        query_bvt: the QUERY BVT.
        alleles: the record's REF and ALT alleles (shared by TRUTH and QUERY).
    """

    chrom: str
    pos: int
    truth_gt: str
    truth_bd: str
    truth_bvt: str
    query_gt: str
    query_bd: str
    query_bk: str
    query_bvt: str
    alleles: tuple[str, ...]


@dataclass(frozen=True)
class HappyLabels:
    """
    Labels for the query calls of one sample, and its truth FN counts.

    Attributes:
        labels: one row per `(chrom, pos, variant_type)` with columns `label` (a `Label` value)
            and `flip_ok` (true for `FLIP` calls whose het<->hom flip gives the truth genotype).
        fn_counts: the number of truth FN records per variant type.
        fn_positions: the `(chrom, pos, variant type)` of each truth FN record.
        truth_deltas: one row per `(chrom, pos, variant_type)` of the truth records, with
            `TRUTH_DELTA_COLUMNS` (NaN when an allele is symbolic or `*`).
    """

    labels: pd.DataFrame
    fn_counts: dict[VariantType, int]
    fn_positions: frozenset[tuple[str, int, VariantType]]
    truth_deltas: pd.DataFrame


def labels_from_happy_records(
    records: Iterable[HappyRecord],
    contigs: Sequence[str] = AUTOSOMES,
) -> HappyLabels:
    """
    Derive query-call labels and truth FN counts from hap.py records.

    When several records share `(chrom, pos)` and query variant type, the label comes from the
    first and `flip_ok` is true if any of them is a correct flip.

    Args:
        records: the hap.py records.
        contigs: only records on these contigs are used.

    Returns:
        The labels and FN counts.

    Raises:
        ValueError: if no record is on the contigs (e.g. `1` vs `chr1` naming).
    """
    wanted = set(contigs)
    n_on_contigs = 0
    labels: dict[tuple[str, int, VariantType], tuple[Label, bool]] = {}
    fn_counts = dict.fromkeys(VariantType, 0)
    fn_positions: set[tuple[str, int, VariantType]] = set()
    truth_deltas: dict[tuple[str, int, VariantType], tuple[int, int] | None] = {}
    for rec in records:
        if rec.chrom not in wanted:
            continue
        n_on_contigs += 1
        truth_type = _BVT_TO_TYPE.get(rec.truth_bvt)
        if rec.truth_gt != "./." and truth_type is not None:
            truth_key = (rec.chrom, rec.pos, truth_type)
            if truth_key not in truth_deltas:
                truth_deltas[truth_key] = allele_deltas(rec.alleles, rec.truth_gt)
        if rec.truth_bd == "FN" and truth_type is not None:
            fn_counts[truth_type] += 1
            fn_positions.add((rec.chrom, rec.pos, truth_type))
        query_type = _BVT_TO_TYPE.get(rec.query_bvt)
        if query_type is None or rec.query_bd not in ("TP", "FP"):
            continue
        key = (rec.chrom, rec.pos, query_type)
        flip_ok = (
            rec.query_bd == "FP"
            and rec.query_bk == "am"
            and FLIPPED_GT.get(rec.query_gt) == rec.truth_gt
        )
        if key in labels:
            label, any_flip_ok = labels[key]
            labels[key] = (label, any_flip_ok or flip_ok)
        else:
            labels[key] = (_label_of(rec.query_bd, rec.query_bk), flip_ok)
    if n_on_contigs == 0:
        raise ValueError(f"No hap.py records on the extracted contigs: {sorted(wanted)}")
    frame = pd.DataFrame(
        [
            (chrom, pos, variant_type.value, int(label), ok)
            for (chrom, pos, variant_type), (label, ok) in labels.items()
        ],
        columns=["chrom", "pos", "variant_type", "label", "flip_ok"],
    )
    frame["flip_ok"] = frame["flip_ok"] & (frame["label"] == Label.FLIP)
    deltas = pd.DataFrame(
        [
            (chrom, pos, variant_type.value, *(pair if pair is not None else (math.nan, math.nan)))
            for (chrom, pos, variant_type), pair in truth_deltas.items()
        ],
        columns=["chrom", "pos", "variant_type", *TRUTH_DELTA_COLUMNS],
    )
    return HappyLabels(
        labels=frame,
        fn_counts=fn_counts,
        fn_positions=frozenset(fn_positions),
        truth_deltas=deltas.astype(dict.fromkeys(TRUTH_DELTA_COLUMNS, "float64")),
    )


def _label_of(query_bd: str, query_bk: str) -> Label:
    """The label of a query call with BD TP or FP."""
    if query_bd == "TP":
        return Label.KEEP
    return Label.FLIP if query_bk == "am" else Label.DROP


def read_happy_records(path: Path) -> Iterable[HappyRecord]:
    """
    Read the records of a hap.py output VCF.

    Args:
        path: the hap.py VCF (with TRUTH and QUERY samples).

    Yields:
        The hap.py records.

    Raises:
        ValueError: if the VCF does not have TRUTH and QUERY samples or the BD/BK/BVT fields.
    """
    with pysam.VariantFile(str(path)) as vcf:
        samples = set(vcf.header.samples)
        missing = {"TRUTH", "QUERY"} - samples
        missing_formats = {"GT", "BD", "BK", "BVT"} - set(vcf.header.formats)
        if missing or missing_formats:
            raise ValueError(
                f"{path} does not look like hap.py output: missing samples {sorted(missing)} "
                f"/ FORMAT fields {sorted(missing_formats)}"
            )
        for rec in vcf:
            truth, query = rec.samples["TRUTH"], rec.samples["QUERY"]
            yield HappyRecord(
                chrom=rec.chrom,
                pos=rec.pos,
                truth_gt=normalize_gt(truth.get("GT")),
                truth_bd=truth.get("BD") or ".",
                truth_bvt=truth.get("BVT") or ".",
                query_gt=normalize_gt(query.get("GT")),
                query_bd=query.get("BD") or ".",
                query_bk=query.get("BK") or ".",
                query_bvt=query.get("BVT") or ".",
                alleles=tuple(rec.alleles or ()),
            )


def read_happy_labels(path: Path, contigs: Sequence[str] = AUTOSOMES) -> HappyLabels:
    """
    Read query-call labels and truth FN counts from a hap.py output VCF.

    Args:
        path: the hap.py VCF.
        contigs: only records on these contigs are used.

    Returns:
        The labels and FN counts.

    Raises:
        ValueError: if the VCF is not hap.py output or has no record on the contigs.
    """
    try:
        return labels_from_happy_records(read_happy_records(path), contigs)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


@dataclass(frozen=True)
class LabelJoin:
    """
    How the hap.py query labels joined to a feature table's calls.

    Attributes:
        n_labels: hap.py query records scored TP or FP (one per `(chrom, pos, variant type)`).
        n_pass: those that labelled a PASS call.
        n_padded_snp: of `n_pass`, those found at the differing base of a padded SNP (see
            `attach_labels`).
        n_non_pass: those at a non-PASS call only (hap.py scored non-PASS calls: it was run
            without `--pass-only`).
        n_unmatched: those at no call of their variant type (e.g. hap.py ran on another VCF, or
            represented the call differently); they count in neither the labels nor TP/FP.
    """

    n_labels: int
    n_pass: int
    n_padded_snp: int
    n_non_pass: int
    n_unmatched: int


def attach_labels(features: pd.DataFrame, labels: HappyLabels) -> tuple[pd.DataFrame, LabelJoin]:
    """
    Add `label` (float, NaN for unlabelled), `flip_ok` and `TRUTH_DELTA_COLUMNS` to a feature table.

    Labels attach to PASS calls, and truth allele length changes to every call, by
    `(chrom, pos, variant_type)`. A padded SNP whose differing base is not its first (e.g.
    `TTAGA>TTCGA`, written so by the caller when an overlapping allele was dropped) that has no
    label at its POS takes the label at its differing base, in case hap.py trimmed it there,
    unless another SNP call is at that position.

    Args:
        features: a feature table (see `winnowr.features`).
        labels: the hap.py labels.

    Returns:
        A new feature table with the label columns, and how the labels joined to the calls.
    """
    keys = ["chrom", "pos", "variant_type"]
    merged = (
        features
        .drop(columns=["label", "flip_ok", *TRUTH_DELTA_COLUMNS], errors="ignore")
        .merge(labels.labels, on=keys, how="left", validate="many_to_one")
        .merge(labels.truth_deltas, on=keys, how="left", validate="many_to_one")
    )
    for column in TRUTH_DELTA_COLUMNS:
        merged[column] = merged[column].astype("float64")
    not_pass = merged["filter"] != PASS
    label_pos = merged["pos"].to_numpy(dtype=np.int64).copy()
    padded = _padded_snp_label_positions(merged, labels.labels, not_pass.to_numpy())
    for row, (pos, label, flip_ok) in padded.items():
        label_pos[row] = pos
        merged.loc[row, ["label", "flip_ok"]] = (label, flip_ok)
    merged["label"] = merged["label"].astype("float64").where(~not_pass)
    merged["flip_ok"] = merged["flip_ok"].astype("boolean").fillna(False).astype(bool) & ~not_pass
    join = _label_join(merged, labels.labels, label_pos, len(padded))
    return merged, join


MAX_UNMATCHED_LABEL_FRACTION: Final[float] = 0.5
"""More hap.py labels than this fraction matching no call means hap.py ran on another VCF."""


def check_label_join(join: LabelJoin, source: str = "the hap.py VCF") -> None:
    """
    Log how the hap.py labels joined to the calls, and fail when most of them matched no call.

    Args:
        join: the label join counts (see `attach_labels`).
        source: where the labels came from (for the messages).

    Raises:
        ValueError: if more than `MAX_UNMATCHED_LABEL_FRACTION` of the labels matched no call.
    """
    logger.info(
        f"{source}: {join.n_labels:,} TP/FP query records, {join.n_pass:,} labelling PASS calls "
        f"({join.n_padded_snp:,} at a padded SNP's differing base)"
    )
    if join.n_non_pass:
        logger.warning(
            f"{source} scores {join.n_non_pass:,} non-PASS calls: hap.py must be run with "
            "--pass-only, or a truth variant matched only by a non-PASS call is not counted as "
            "an FN (and cannot be a promote label)"
        )
    if join.n_labels and join.n_unmatched / join.n_labels > MAX_UNMATCHED_LABEL_FRACTION:
        raise ValueError(
            f"{join.n_unmatched:,} of {join.n_labels:,} TP/FP query records in {source} match no "
            "call of their variant type by position: was hap.py run on this VCF?"
        )
    if join.n_unmatched:
        logger.warning(
            f"{join.n_unmatched:,} TP/FP query records in {source} match no call of their variant "
            "type by position (hap.py may have represented them differently); they are left out "
            "of the labels and the TP/FP counts"
        )


def _snp_offset(ref: str, alt: str) -> int:
    """The offset of the differing base of a SNP record (0 if alternates differ at other bases)."""
    offsets = {
        next(i for i, (r, a) in enumerate(zip(ref, allele, strict=True)) if r != a)
        for allele in alt.split(",")
        if allele != "*" and len(allele) == len(ref) and allele != ref
    }
    return offsets.pop() if len(offsets) == 1 else 0


def _padded_snp_label_positions(
    merged: pd.DataFrame, labels: pd.DataFrame, not_pass: npt.NDArray[np.bool_]
) -> dict[int, tuple[int, float, bool]]:
    """The unlabelled PASS padded SNPs with a label at their differing base: row -> pos, label."""
    is_snp = (merged["variant_type"] == VariantType.SNP.value).to_numpy()
    unlabelled = np.flatnonzero(is_snp & ~not_pass & merged["label"].isna().to_numpy())
    if len(unlabelled) == 0:
        return {}
    snp_labels = labels[labels["variant_type"] == VariantType.SNP.value]
    by_key = {
        (chrom, pos): (label, flip_ok)
        for chrom, pos, label, flip_ok in zip(
            snp_labels["chrom"],
            snp_labels["pos"],
            snp_labels["label"],
            snp_labels["flip_ok"],
            strict=True,
        )
    }
    occupied = set(zip(merged["chrom"][is_snp], merged["pos"][is_snp], strict=True))
    found: dict[int, tuple[int, float, bool]] = {}
    subset = merged.iloc[unlabelled]
    for row, chrom, pos, ref, alt in zip(
        unlabelled.tolist(),
        subset["chrom"].tolist(),
        subset["pos"].tolist(),
        subset["ref"].tolist(),
        subset["alt"].tolist(),
        strict=True,
    ):
        offset = _snp_offset(ref, alt)
        hit = by_key.get((chrom, pos + offset)) if offset else None
        if hit is not None and (chrom, pos + offset) not in occupied:
            found[row] = (pos + offset, float(hit[0]), bool(hit[1]))
    return found


def _label_join(
    merged: pd.DataFrame, labels: pd.DataFrame, label_pos: npt.NDArray[np.int64], n_padded: int
) -> LabelJoin:
    """Count where the hap.py labels went (see `LabelJoin`)."""
    is_pass = (merged["filter"] == PASS).to_numpy()
    on_pass = set(
        zip(
            merged["chrom"][is_pass],
            label_pos[is_pass],
            merged["variant_type"][is_pass],
            strict=True,
        )
    )
    on_call = set(zip(merged["chrom"], merged["pos"], merged["variant_type"], strict=True))
    n_pass = n_non_pass = n_unmatched = 0
    for key in zip(labels["chrom"], labels["pos"], labels["variant_type"], strict=True):
        if key in on_pass:
            n_pass += 1
        elif key in on_call:
            n_non_pass += 1
        else:
            n_unmatched += 1
    return LabelJoin(
        n_labels=len(labels),
        n_pass=n_pass,
        n_padded_snp=n_padded,
        n_non_pass=n_non_pass,
        n_unmatched=n_unmatched,
    )


def labelled_calls(table: pd.DataFrame, variant_type: VariantType) -> pd.DataFrame:
    """
    Select the labelled PASS calls of one variant type, one per `(chrom, pos)`.

    These are the calls hap.py scored (TP or FP), i.e. the rows used for training and F1.

    Args:
        table: a labelled feature table.
        variant_type: the variant type.

    Returns:
        The selected rows (a copy, with a fresh index).
    """
    selected = table[
        (table["variant_type"] == variant_type.value)
        & (table["filter"] == PASS)
        & table["label"].notna()
    ]
    return selected.drop_duplicates(["chrom", "pos"]).reset_index(drop=True)
