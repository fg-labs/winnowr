"""Helpers to build small HaplotypeCaller-like VCFs, gVCFs and hap.py VCFs for tests."""

from collections.abc import Mapping
from pathlib import Path

from fgpyo.vcf.builder import VariantBuilder

SAMPLE = "NA00001"


def vcf_builder(raw_annotations: bool = False) -> VariantBuilder:
    """
    A single-sample VCF builder with the HaplotypeCaller annotations winnowr reads.

    With `raw_annotations`, it also declares the raw annotations that a VCF extracted from a gVCF
    can keep (see `raw_annotation_info`).
    """
    builder = VariantBuilder(sample_ids=[SAMPLE])
    if raw_annotations:
        _add_raw_annotation_headers(builder)
    for key in ("MQRankSum", "ReadPosRankSum", "BaseQRankSum"):
        _add_header(builder, "INFO", key, "1", "Float")
    _add_header(builder, "INFO", "DRAGstrInfo", "2", "Integer")
    _add_header(builder, "FORMAT", "GQ", "1", "Integer")
    _add_header(builder, "FORMAT", "SB", "4", "Integer")
    _add_header(builder, "FORMAT", "PL", "G", "Integer")
    _add_header(builder, "INFO", "ExcessHet", "1", "Float")
    builder.header.add_line('##FILTER=<ID=LowQual,Description="Low quality">')
    return builder


def _add_header(builder: VariantBuilder, kind: str, key: str, number: str, field_type: str) -> None:
    """Declare an INFO or FORMAT field (fgpyo's header helpers need enum arguments)."""
    builder.header.add_line(
        f'##{kind}=<ID={key},Number={number},Type={field_type},Description="{key}">'
    )


def add_call(
    builder: VariantBuilder,
    pos: int,
    gt: tuple[int | None, int | None] = (0, 1),
    contig: str = "chr1",
    ref: str = "A",
    alt: str = "C",
    ad: tuple[int | None, ...] = (10, 10),
    filter: str | None = None,  # noqa: A002  # mirrors VariantBuilder.add
    qual: int = 50,
    phased: bool = False,
    raw: Mapping[str, object] | None = None,
) -> None:
    """Add a call with plausible annotations (and `raw` INFO values) to a `vcf_builder()`."""
    builder.add(
        contig=contig,
        pos=pos,
        ref=ref,
        alts=alt.split(","),
        qual=qual,
        filter=filter,
        info={
            "DP": sum(d for d in ad if d is not None),
            "MQRankSum": 0.5,
            "ReadPosRankSum": -0.5,
            "BaseQRankSum": 1.0,
            "DRAGstrInfo": (1, 4),
            "ExcessHet": 3.0,
            **(raw or {}),
        },
        samples={
            SAMPLE: {
                "GT": gt,
                "AD": ad,
                "GQ": 40,
                "SB": (5, 5, 4, 6),
                "PL": tuple(range(0, 10 * (len(ad) * (len(ad) + 1) // 2), 10)),
            }
        },
    )
    if phased:
        builder.records[-1].samples[SAMPLE].phased = True


def gvcf_builder() -> VariantBuilder:
    """A single-sample gVCF builder declaring the gVCF-only annotations."""
    builder = VariantBuilder(sample_ids=[SAMPLE])
    _add_raw_annotation_headers(builder)
    _add_header(builder, "INFO", "END", "1", "Integer")
    return builder


def _add_raw_annotation_headers(builder: VariantBuilder) -> None:
    """Declare RAW_MQandDP, QUALapprox, VarDP and AS_RAW_MQ as HaplotypeCaller does."""
    _add_header(builder, "INFO", "RAW_MQandDP", "2", "Integer")
    _add_header(builder, "INFO", "QUALapprox", "1", "Integer")
    _add_header(builder, "INFO", "VarDP", "1", "Integer")
    _add_header(builder, "INFO", "AS_RAW_MQ", "1", "String")


def raw_annotation_info(
    raw_mq: int = 36000,
    raw_dp: int = 10,
    qual_approx: int = 100,
    var_dp: int = 20,
    as_raw_mq: str = "18000.00|18000.00",
) -> dict[str, object]:
    """
    The raw annotations of a VCF call, for `add_call`'s `raw`.

    They are as a VCF extracted from a gVCF keeps them: AS_RAW_MQ has one value per allele and no
    `<NON_REF>`. The defaults match `add_gvcf_site`'s.
    """
    return {
        "RAW_MQandDP": (raw_mq, raw_dp),
        "QUALapprox": qual_approx,
        "VarDP": var_dp,
        "AS_RAW_MQ": as_raw_mq,
    }


def add_gvcf_site(
    builder: VariantBuilder,
    pos: int,
    contig: str = "chr1",
    ref: str = "A",
    alt: str = "C",
    raw_mq: int = 36000,
    raw_dp: int = 10,
    qual_approx: int = 100,
    var_dp: int = 20,
    as_raw_mq: str = "18000.00|18000.00|0.00",
) -> None:
    """Add a variant site (with a `<NON_REF>` allele) to a `gvcf_builder()`."""
    builder.add(
        contig=contig,
        pos=pos,
        ref=ref,
        alts=[*alt.split(","), "<NON_REF>"],
        info={
            "RAW_MQandDP": (raw_mq, raw_dp),
            "QUALapprox": qual_approx,
            "VarDP": var_dp,
            "AS_RAW_MQ": as_raw_mq,
        },
    )


def add_ref_block(builder: VariantBuilder, pos: int, end: int, contig: str = "chr1") -> None:
    """Add a reference block to a `gvcf_builder()`."""
    builder.add(contig=contig, pos=pos, ref="A", alts=["<NON_REF>"], info={"END": end})


def happy_builder() -> VariantBuilder:
    """A hap.py-style output VCF builder with TRUTH and QUERY samples."""
    builder = VariantBuilder(sample_ids=["TRUTH", "QUERY"])
    for key in ("BD", "BK", "BVT"):
        _add_header(builder, "FORMAT", key, "1", "String")
    return builder


def add_happy_record(
    builder: VariantBuilder,
    pos: int,
    query: tuple[tuple[int, int] | None, str, str, str],
    truth: tuple[tuple[int, int] | None, str, str, str] = (None, ".", ".", "NOCALL"),
    contig: str = "chr1",
    ref: str = "A",
    alt: str = "C",
) -> None:
    """Add a hap.py record; `query`/`truth` are `(GT, BD, BK, BVT)` (GT None for missing)."""
    samples = {}
    for name, (gt, bd, bk, bvt) in (("TRUTH", truth), ("QUERY", query)):
        samples[name] = {
            "GT": gt if gt is not None else (None, None),
            "BD": bd,
            "BK": bk,
            "BVT": bvt,
        }
    builder.add(contig=contig, pos=pos, ref=ref, alts=alt.split(","), samples=samples)


def write(builder: VariantBuilder, path: Path) -> Path:
    """Write a builder to a bgzipped, tabix-indexed VCF."""
    return builder.to_path(path)
