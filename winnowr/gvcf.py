"""
Read the raw HaplotypeCaller annotations that gVCF-mode calling produces.

In reblocked `--dragen-mode` gVCFs, variant sites carry `RAW_MQandDP` (sum of squared mapping
qualities and the read depth), `QUALapprox`, `VarDP` and `AS_RAW_MQ` (per-allele sum of squared
mapping qualities, `|`-separated, with the `<NON_REF>` allele last). Reference blocks do not, and
are skipped. A VCF extracted from the gVCF may keep the same annotations, set for its own alleles
(with `AS_RAW_MQ` one value per allele and no `<NON_REF>`), in which case they are read from the
VCF itself (see `has_raw_annotations` and `raw_annotation`).
"""

import math
from collections.abc import Collection
from collections.abc import Sequence
from dataclasses import dataclass

import pysam

RAW_ANNOTATION_INFO: tuple[str, ...] = ("RAW_MQandDP", "QUALapprox", "VarDP", "AS_RAW_MQ")
"""The raw INFO annotations read, which a gVCF (or a VCF that keeps them) must declare."""

NON_REF: str = "<NON_REF>"


@dataclass(frozen=True)
class GvcfAnnotation:
    """
    The raw annotations at one variant site, from the gVCF or the VCF.

    Attributes:
        raw_mq: sum of squared mapping qualities (RAW_MQandDP[0]).
        raw_dp: read depth used for RAW_MQandDP (RAW_MQandDP[1]).
        qual_approx: QUALapprox.
        var_dp: VarDP.
        as_mq_ref: AS_RAW_MQ of the reference allele.
        as_mq_alt: AS_RAW_MQ summed over the alternate alleles (excluding `<NON_REF>`).
        n_alt: number of alternate alleles (excluding `<NON_REF>`).
    """

    raw_mq: float
    raw_dp: float
    qual_approx: float
    var_dp: float
    as_mq_ref: float
    as_mq_alt: float
    n_alt: float


def check_gvcf_header(
    header: pysam.VariantHeader, path: str, contigs: Collection[str] = ()
) -> None:
    """
    Check that a gVCF header declares the required INFO fields and contigs.

    Args:
        header: the gVCF header.
        path: the gVCF path (for the error message).
        contigs: contigs that must be in the header.

    Raises:
        ValueError: if any of `RAW_ANNOTATION_INFO` is not declared, or a contig is not in the
            header (e.g. `1` vs `chr1` naming).
    """
    unknown = [contig for contig in contigs if contig not in header.contigs]
    if unknown:
        raise ValueError(f"Contig(s) not in the header of gVCF {path}: {', '.join(unknown)}")
    missing = missing_raw_annotations(header)
    if missing:
        raise ValueError(
            f"gVCF {path} lacks required INFO annotations {missing}; it must come from "
            "HaplotypeCaller in gVCF mode (e.g. --dragen-mode)"
        )


def missing_raw_annotations(header: pysam.VariantHeader) -> list[str]:
    """
    The raw annotations a VCF or gVCF header does not declare.

    Args:
        header: the header.

    Returns:
        The keys of `RAW_ANNOTATION_INFO` that the header does not declare, in that order.
    """
    return [key for key in RAW_ANNOTATION_INFO if key not in header.info]


def has_raw_annotations(header: pysam.VariantHeader) -> bool:
    """
    Whether a VCF header declares every raw annotation, so its records can supply them.

    Args:
        header: the VCF header.

    Returns:
        True if every key of `RAW_ANNOTATION_INFO` is declared.
    """
    return not missing_raw_annotations(header)


def raw_annotation(rec: pysam.VariantRecord) -> GvcfAnnotation | None:
    """
    Read the raw annotations of one record (a gVCF variant site or a VCF call).

    Args:
        rec: the record; its header must declare `RAW_ANNOTATION_INFO`.

    Returns:
        The annotations, or None if the record has no RAW_MQandDP (e.g. a reference block).
    """
    raw_mq_and_dp = rec.info.get("RAW_MQandDP")
    if raw_mq_and_dp is None:
        return None
    as_mq_ref, as_mq_alt, n_alt = parse_as_raw_mq(rec.info.get("AS_RAW_MQ"), rec.alleles or ())
    return GvcfAnnotation(
        raw_mq=_to_float(raw_mq_and_dp[0]),
        raw_dp=_to_float(raw_mq_and_dp[1]),
        qual_approx=_to_float(rec.info.get("QUALapprox")),
        var_dp=_to_float(rec.info.get("VarDP")),
        as_mq_ref=as_mq_ref,
        as_mq_alt=as_mq_alt,
        n_alt=n_alt,
    )


def parse_as_raw_mq(
    value: str | Sequence[float | None] | None, alleles: Sequence[str]
) -> tuple[float, float, float]:
    """
    Split an AS_RAW_MQ value into reference and alternate shares.

    Args:
        value: the AS_RAW_MQ value; a `|`-separated string or a per-allele sequence.
        alleles: the record's alleles (reference first), used to skip `<NON_REF>`.

    Returns:
        The reference share, the summed alternate share and the number of alternate alleles.
        All are NaN when the value is missing or does not have one entry per allele.
    """
    nan = (math.nan, math.nan, math.nan)
    if value is None:
        return nan
    raw = value.split("|") if isinstance(value, str) else list(value)
    if len(raw) != len(alleles):
        return nan
    values = [_to_float(v) for v in raw]
    alt_values = [v for v, a in zip(values[1:], alleles[1:], strict=True) if a != NON_REF]
    as_mq_ref = 0.0 if math.isnan(values[0]) else values[0]
    as_mq_alt = sum(0.0 if math.isnan(v) else v for v in alt_values)
    return as_mq_ref, as_mq_alt, float(len(alt_values))


def read_gvcf_annotations(
    gvcf: pysam.VariantFile, contig: str, positions: Collection[int] | None = None
) -> dict[int, GvcfAnnotation]:
    """
    Read the annotations of the gVCF variant sites on one contig.

    Args:
        gvcf: an indexed gVCF opened with pysam.
        contig: the contig to read.
        positions: only read sites at these 1-based positions (default: all sites).

    Returns:
        A map from 1-based position to annotation; at a position, the first record with
        RAW_MQandDP wins.
    """
    annotations: dict[int, GvcfAnnotation] = {}
    if contig not in gvcf.header.contigs:
        return annotations
    for rec in gvcf.fetch(contig):
        if positions is not None and rec.pos not in positions:
            continue
        if rec.pos in annotations:
            continue
        annotation = raw_annotation(rec)
        if annotation is not None:
            annotations[rec.pos] = annotation
    return annotations


def _to_float(value: str | float | None) -> float:
    """Convert a VCF value to float, mapping missing/unparseable values to NaN."""
    if value is None or value == ".":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan
