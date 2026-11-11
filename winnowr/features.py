"""
Per-call feature extraction from a HaplotypeCaller VCF (and, for older VCFs, its gVCF).

Extraction happens in two steps: `extract_raw_records()` reads raw values (one row per VCF record)
and `add_derived_features()` turns them into model features, including region flags and
neighbourhood features.
"""

import logging
import math
import re
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import astuple
from dataclasses import fields
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pysam

from winnowr.alleles import FLIPPED_GT
from winnowr.alleles import first_alt_index
from winnowr.alleles import normalize_gt
from winnowr.alleles import variant_type_of
from winnowr.common import AUTOSOMES
from winnowr.common import PASS
from winnowr.common import BoolArray
from winnowr.common import IntArray
from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.gvcf import GvcfAnnotation
from winnowr.gvcf import check_gvcf_header
from winnowr.gvcf import has_raw_annotations
from winnowr.gvcf import missing_raw_annotations
from winnowr.gvcf import raw_annotation
from winnowr.gvcf import read_gvcf_annotations
from winnowr.neighbourhood import NEIGHBOURHOOD_FEATURES
from winnowr.neighbourhood import neighbourhood_features
from winnowr.regions import RegionSet

logger = logging.getLogger(__name__)

RAW_COLUMNS: Final[tuple[str, ...]] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "filter",
    "variant_type",
    "gt",
    "qual",
    "dp",
    "mq_rank_sum",
    "read_pos_rank_sum",
    "base_q_rank_sum",
    "str_period",
    "str_repeats",
    "gq",
    "ad_ref",
    "ad_alt",
    "sb_ref_fwd",
    "sb_ref_rev",
    "sb_alt_fwd",
    "sb_alt_rev",
    "excess_het",
    "pl_ref",
    "pl_het",
    "pl_hom",
    "ad_alt_total",
    "raw_mq",
    "raw_dp",
    "qual_approx",
    "var_dp",
    "as_mq_ref",
    "as_mq_alt",
    "n_alt",
)
"""Columns produced by `extract_raw_records()`; `gt` is the normalized genotype, e.g. `0/1`."""

COMMON_FEATURES: Final[tuple[str, ...]] = (
    "qual",
    "dp",
    "gq",
    "mq_rank_sum",
    "read_pos_rank_sum",
    "base_q_rank_sum",
    "ab",
    "sor",
    "het",
    "mq",
    "qd",
    "as_alt_frac",
    "dp_norm",
    "n_alt",
)
"""Features used for every variant type."""

INDEL_FEATURES: Final[tuple[str, ...]] = ("indel_len", "indel_abs_len", "str_period", "str_repeats")
"""Additional features used for indels."""


PROMOTE_FEATURES: Final[tuple[str, ...]] = (
    "excess_het",
    "pl_ref",
    "pl_het",
    "pl_hom",
    "ad_ref",
    "ad_alt_total",
    "n_alt_rec",
)
"""Additional features used by promote models."""


def region_feature(name: str) -> str:
    """
    The feature name for membership in the named region set.

    Args:
        name: the region set name.

    Returns:
        `region_<name>`.
    """
    return f"region_{name}"


def feature_names(
    variant_type: VariantType,
    region_names: Iterable[str],
    neighbourhood: bool = True,
    task: Task = Task.GENOTYPE,
    repeat_length: Sequence[str] = (),
) -> list[str]:
    """
    The ordered model features for a variant type.

    Args:
        variant_type: the variant type.
        region_names: names of the region sets, in order.
        neighbourhood: whether to include the neighbourhood features.
        task: the model's task; promote models add `PROMOTE_FEATURES`.
        repeat_length: repeat-length features to append (see `winnowr.repeat_features` and
            `winnowr.stutter`).

    Returns:
        The feature column names.
    """
    names = list(COMMON_FEATURES)
    if task is Task.PROMOTE:
        names.extend(PROMOTE_FEATURES)
    if variant_type is VariantType.INDEL:
        names.extend(INDEL_FEATURES)
    names.extend(region_feature(name) for name in region_names)
    if neighbourhood:
        names.extend(NEIGHBOURHOOD_FEATURES)
    names.extend(repeat_length)
    return names


def extract_raw_records(
    vcf_path: Path, gvcf_path: Path | None = None, contigs: Sequence[str] = AUTOSOMES
) -> pd.DataFrame:
    """
    Read the raw per-record values of a single-sample VCF, with its raw annotations.

    The raw annotations (`RAW_ANNOTATION_INFO`: RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ) come from
    the VCF's own records when its header declares all of them, and otherwise from the gVCF, joined
    by `(chrom, pos)`. Records without them (no RAW_MQandDP, or no gVCF match) get NaN. Records that
    are neither SNPs nor indels (see `variant_type_of`) are skipped.

    Args:
        vcf_path: the indexed single-sample VCF.
        gvcf_path: the indexed single-sample gVCF it was genotyped from; read only when the VCF
            lacks the raw annotations.
        contigs: the contigs to read, in order.

    Returns:
        A data frame with `RAW_COLUMNS`, in VCF order within each contig.

    Raises:
        ValueError: if the VCF does not have exactly one sample, a contig is not in the VCF or gVCF
            header, or neither the VCF nor a gVCF supplies the raw annotations.
    """
    with pysam.VariantFile(str(vcf_path)) as vcf:
        _check_vcf(vcf, vcf_path, contigs)
        if has_raw_annotations(vcf.header):
            if gvcf_path is not None:
                logger.warning(
                    f"VCF {vcf_path} has the raw annotations; the gVCF {gvcf_path} is not read"
                )
            return _read_records(vcf, contigs, None)
        if gvcf_path is None:
            raise ValueError(
                f"VCF {vcf_path} lacks the INFO annotations {missing_raw_annotations(vcf.header)} "
                "and no gVCF was given: pass the gVCF the VCF was genotyped from (--gvcf), or a "
                "VCF that keeps the raw annotations of HaplotypeCaller's gVCF mode"
            )
        with pysam.VariantFile(str(gvcf_path)) as gvcf:
            check_gvcf_header(gvcf.header, str(gvcf_path), contigs)
            logger.info(f"Reading the raw annotations from the gVCF {gvcf_path}")
            return _read_records(vcf, contigs, gvcf)


def _read_records(
    vcf: pysam.VariantFile, contigs: Sequence[str], gvcf: pysam.VariantFile | None
) -> pd.DataFrame:
    """Read the VCF's records, with raw annotations from `gvcf`, or from the VCF if it is None."""
    vcf_rows: list[tuple[object, ...]] = []
    raw_rows: list[tuple[float, ...]] = []
    n_missing = n_skipped = 0
    info_keys = frozenset(vcf.header.info.keys())
    for contig in contigs:
        positions: list[int] = []
        annotations: list[GvcfAnnotation | None] = []
        for rec in vcf.fetch(contig):
            variant_type = variant_type_of(rec.ref or "", rec.alts or ())
            if variant_type is None:
                n_skipped += 1
                continue
            positions.append(rec.pos)
            vcf_rows.append(_raw_values(rec, variant_type, info_keys))
            if gvcf is None:
                annotations.append(raw_annotation(rec))
        if gvcf is not None:
            # only parse the gVCF sites that the VCF has calls at
            by_position = read_gvcf_annotations(gvcf, contig, set(positions))
            annotations = [by_position.get(pos) for pos in positions]
        for annotation in annotations:
            n_missing += annotation is None
            raw_rows.append(_MISSING_RAW if annotation is None else astuple(annotation))
        logger.info(f"Read {len(vcf_rows):,} records through {contig}")
    if n_skipped:
        logger.info(
            f"Skipped {n_skipped:,} records that are neither SNPs nor indels (MNPs, symbolic or "
            "only `*` alleles)"
        )
    if n_missing:
        source = "had no gVCF annotations" if gvcf is not None else "lack RAW_MQandDP"
        logger.warning(f"{n_missing:,} records {source} (set to missing)")
    vcf_columns = zip(*vcf_rows, strict=True) if vcf_rows else ((),) * len(_VCF_COLUMNS)
    raw_columns = zip(*raw_rows, strict=True) if raw_rows else ((),) * len(_RAW_ANNOTATION_COLUMNS)
    columns = dict(zip(_VCF_COLUMNS, vcf_columns, strict=True))
    columns |= dict(zip(_RAW_ANNOTATION_COLUMNS, raw_columns, strict=True))
    return as_raw_frame(columns)


def _check_vcf(vcf: pysam.VariantFile, path: Path, contigs: Sequence[str]) -> None:
    """Validate the sample count, contigs and genotype field of a VCF."""
    if len(vcf.header.samples) != 1:
        raise ValueError(f"VCF {path} must have exactly one sample, has {len(vcf.header.samples)}")
    unknown = [contig for contig in contigs if contig not in vcf.header.contigs]
    if unknown:
        raise ValueError(f"Contig(s) not in the header of {path}: {', '.join(unknown)}")
    if "GT" not in vcf.header.formats:
        raise ValueError(f"VCF {path} has no FORMAT/GT")


_RAW_ANNOTATION_COLUMNS: Final[tuple[str, ...]] = tuple(f.name for f in fields(GvcfAnnotation))
"""The `RAW_COLUMNS` from the raw annotations, in `GvcfAnnotation` field order."""

_VCF_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c in RAW_COLUMNS if c not in _RAW_ANNOTATION_COLUMNS
)
"""The `RAW_COLUMNS` read from the VCF, in `_raw_values` order."""

_MISSING_RAW: Final[tuple[float, ...]] = (math.nan,) * len(_RAW_ANNOTATION_COLUMNS)
"""The raw-annotation values of a record without them."""


def _raw_values(
    rec: pysam.VariantRecord, variant_type: VariantType, info_keys: frozenset[str]
) -> tuple[object, ...]:
    """
    Read the raw VCF values of one record, in `_VCF_COLUMNS` order.

    `info_keys` are the INFO keys the header declares.
    """
    sample = rec.samples[0]
    alts = rec.alts or ()
    ad = sample.get("AD")
    sb = sample.get("SB")
    pl = sample.get("PL")
    first_alt = first_alt_index(alts)
    alt_depths = [v for v in ad[1:] if v is not None] if isinstance(ad, tuple) else []
    dragstr = _info(rec, "DRAGstrInfo", info_keys)
    filters = [str(name) for name in rec.filter.keys()]
    return (
        rec.chrom,
        rec.pos,
        rec.ref,
        ",".join(alts),
        ";".join(filters) if filters else PASS,
        variant_type.value,
        normalize_gt(sample.get("GT")),
        rec.qual,
        _info(rec, "DP", info_keys),
        _info(rec, "MQRankSum", info_keys),
        _info(rec, "ReadPosRankSum", info_keys),
        _info(rec, "BaseQRankSum", info_keys),
        _element(dragstr, 0),
        _element(dragstr, 1),
        sample.get("GQ"),
        _element(ad, 0),
        _element(ad, first_alt + 1) if first_alt is not None else None,
        _element(sb, 0),
        _element(sb, 1),
        _element(sb, 2),
        _element(sb, 3),
        _info(rec, "ExcessHet", info_keys),
        _element(pl, 0),
        _element(pl, 1),
        _element(pl, 2),
        sum(alt_depths) if alt_depths else None,
    )


def _info(rec: pysam.VariantRecord, key: str, info_keys: frozenset[str]) -> object:
    """An INFO value, or None when the key is not declared (not in `info_keys`) or not set."""
    return rec.info.get(key) if key in info_keys else None


def _element(values: object, index: int) -> object:
    """The `index`-th element of a tuple-valued field, or None."""
    if isinstance(values, tuple) and len(values) > index:
        return values[index]
    return None


STRING_RAW_COLUMNS: Final[frozenset[str]] = frozenset({
    "chrom", "ref", "alt", "filter", "variant_type", "gt",
})  # fmt: skip
"""The string-valued `RAW_COLUMNS`; `pos` is int64 and every other column float64."""


def as_raw_frame(columns: Mapping[str, Sequence[object] | pd.Series]) -> pd.DataFrame:
    """
    Build a raw record frame with fixed dtypes, coercing numeric columns to float (None -> NaN).

    The string columns are given an explicit dtype so that an empty frame still supports `.str`.

    Args:
        columns: the values of each of `RAW_COLUMNS`, one entry per record.

    Returns:
        A data frame with `RAW_COLUMNS`: `pos` int64, `STRING_RAW_COLUMNS` object, the rest float64.
    """
    frame = pd.DataFrame(columns, columns=list(RAW_COLUMNS))
    for column in RAW_COLUMNS:
        if column in STRING_RAW_COLUMNS:
            frame[column] = frame[column].astype(object)
        elif column != "pos":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
    frame["pos"] = frame["pos"].astype(np.int64)
    return frame


def strand_odds_ratio(
    ref_fwd: pd.Series, ref_rev: pd.Series, alt_fwd: pd.Series, alt_rev: pd.Series
) -> pd.Series:
    """
    GATK's StrandOddsRatio (SOR) from the per-strand allele depths of FORMAT/SB.

    As in GATK 4 (`StrandOddsRatio.calculateSOR`), each count gets a pseudocount of one, then
    with `R = (rf * ar) / (rr * af)`:
    `SOR = ln(R + 1/R) + ln(min(rf, rr) / max(rf, rr)) - ln(min(af, ar) / max(af, ar))`.
    FORMAT/SB already sums the alternate alleles of multi-allelic records.

    Args:
        ref_fwd: reference-allele reads on the forward strand (SB[0]).
        ref_rev: reference-allele reads on the reverse strand (SB[1]).
        alt_fwd: alternate-allele reads on the forward strand (SB[2]).
        alt_rev: alternate-allele reads on the reverse strand (SB[3]).

    Returns:
        The SOR of each record (NaN where any count is missing).
    """
    rf, rr, af, ar = (counts + 1 for counts in (ref_fwd, ref_rev, alt_fwd, alt_rev))
    ratio = (rf * ar) / (rr * af)
    ref_ratio = np.minimum(rf, rr) / np.maximum(rf, rr)
    alt_ratio = np.minimum(af, ar) / np.maximum(af, ar)
    sor = np.log(ratio + 1 / ratio) + np.log(ref_ratio) - np.log(alt_ratio)
    return pd.Series(sor, index=ref_fwd.index, dtype=np.float64)


_HET_ANY = re.compile(r"^0/[1-9]\d*$")


def add_derived_features(raw: pd.DataFrame, regions: Sequence[RegionSet]) -> pd.DataFrame:
    """
    Derive the model features from raw record values.

    Derived features:

    - `ab`: AD[alt] / (AD[0] + AD[alt]) for the first non-`*` alternate allele (`ad_alt`).
    - `sor`: GATK's StrandOddsRatio from FORMAT/SB (see `strand_odds_ratio`).
    - `het`: GT is `0/1`. `flippable`: GT is `0/1` or `1/1`.
    - `mq`: sqrt(RAW_MQ / DP) from RAW_MQandDP.
    - `qd`: QUALapprox / VarDP.
    - `as_alt_frac`: the alternate alleles' share of AS_RAW_MQ (an MQ-weighted allele balance).
    - `dp_norm`: RAW_MQandDP depth over the sample median of that depth across all records.
    - `indel_len`, `indel_abs_len`: length change of the first non-`*` alternate allele.
    - `region_<name>`: 1 if the REF span (POS and, for deletions, the deleted bases) overlaps the
      named region set.
    - `n_alt_rec`: the number of alternate alleles in the record.
    - `overlaps_pass` (not a feature): the record's REF span overlaps a PASS record's.
    - neighbourhood features over all records (see `neighbourhood_features`).

    Args:
        raw: the output of `extract_raw_records()` (or an equivalent frame with `RAW_COLUMNS`).
        regions: the region sets to flag.

    Returns:
        A new data frame with the raw columns plus the derived feature columns.
    """
    frame = raw.copy()
    ad_total = frame["ad_ref"] + frame["ad_alt"]
    frame["ab"] = frame["ad_alt"] / ad_total.where(ad_total > 0)
    frame["sor"] = strand_odds_ratio(
        *(frame[c] for c in ("sb_ref_fwd", "sb_ref_rev", "sb_alt_fwd", "sb_alt_rev"))
    )
    frame["het"] = (frame["gt"] == "0/1").astype(np.float64)
    frame["flippable"] = frame["gt"].isin(FLIPPED_GT)
    frame["mq"] = np.sqrt(frame["raw_mq"] / frame["raw_dp"].where(frame["raw_dp"] > 0))
    frame["qd"] = frame["qual_approx"] / frame["var_dp"].where(frame["var_dp"] > 0)
    as_total = frame["as_mq_ref"] + frame["as_mq_alt"]
    frame["as_alt_frac"] = frame["as_mq_alt"] / as_total.where(as_total > 0)
    median_dp = frame["raw_dp"].median()
    frame["dp_norm"] = frame["raw_dp"] / median_dp if median_dp > 0 else np.nan
    first_alt_len = [_first_alt_length(alt) for alt in frame["alt"]]
    frame["indel_len"] = np.asarray(first_alt_len, dtype=np.float64) - [
        len(ref) for ref in frame["ref"]
    ]
    frame["indel_abs_len"] = frame["indel_len"].abs()
    starts, ends = _ref_spans(frame)
    contig_indices = frame.groupby("chrom", sort=False).indices.items()
    for region_set in regions:
        flags = np.zeros(len(frame), dtype=np.float64)
        for contig, index in contig_indices:
            flags[index] = region_set.overlaps_span(str(contig), starts[index], ends[index])
        frame[region_feature(region_set.name)] = flags
    frame["n_alt_rec"] = (frame["alt"].str.count(",") + 1).astype(np.float64)
    frame["overlaps_pass"] = _overlaps_pass(frame)
    neighbours = neighbourhood_features(
        pd.DataFrame({
            "chrom": frame["chrom"],
            "pos": frame["pos"],
            "het_any": frame["gt"].str.match(_HET_ANY),
            "ab": frame["ab"],
            "filtered": frame["filter"] != PASS,
        })
    )
    return frame.merge(neighbours, on=["chrom", "pos"], how="left", validate="many_to_one")


def _first_alt_length(alt: str) -> float:
    """The length of the first non-`*` allele of a comma-joined ALT (NaN if there is none)."""
    alts = alt.split(",")
    index = first_alt_index(alts)
    return math.nan if index is None else float(len(alts[index]))


def _ref_spans(frame: pd.DataFrame) -> tuple[IntArray, IntArray]:
    """The 0-based, half-open REF span `[pos - 1, pos - 1 + len(ref))` of each record."""
    starts = frame["pos"].to_numpy(dtype=np.int64) - 1
    ends = starts + np.asarray([len(ref) for ref in frame["ref"]], dtype=np.int64)
    return starts, ends


def _overlaps_pass(frame: pd.DataFrame) -> BoolArray:
    """Flag records whose REF span overlaps the REF span of any PASS record (itself included)."""
    out = np.zeros(len(frame), dtype=np.bool_)
    starts, ends = _ref_spans(frame)
    is_pass = (frame["filter"] == PASS).to_numpy()
    contig_indices = {
        str(contig): np.asarray(index, dtype=np.int64)
        for contig, index in frame.groupby("chrom", sort=False).indices.items()
    }
    pass_spans = RegionSet.from_spans(
        "pass",
        {
            contig: (starts[index[is_pass[index]]], ends[index[is_pass[index]]])
            for contig, index in contig_indices.items()
        },
    )
    for contig, index in contig_indices.items():
        out[index] = pass_spans.overlaps_span(contig, starts[index], ends[index])
    return out
