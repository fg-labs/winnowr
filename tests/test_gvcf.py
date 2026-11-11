import math
from pathlib import Path

import pysam
import pytest
from fgpyo.vcf.builder import VariantBuilder

from tests.util import add_call
from tests.util import add_gvcf_site
from tests.util import add_ref_block
from tests.util import gvcf_builder
from tests.util import raw_annotation_info
from tests.util import vcf_builder
from tests.util import write
from winnowr.gvcf import check_gvcf_header
from winnowr.gvcf import has_raw_annotations
from winnowr.gvcf import missing_raw_annotations
from winnowr.gvcf import parse_as_raw_mq
from winnowr.gvcf import raw_annotation
from winnowr.gvcf import read_gvcf_annotations


def test_read_gvcf_annotations_skips_ref_blocks_and_keeps_first(tmp_path: Path) -> None:
    builder = gvcf_builder()
    add_ref_block(builder, pos=1, end=99)
    add_gvcf_site(builder, pos=100, raw_mq=3600, raw_dp=4, as_raw_mq="900.0|2700.0|0.0")
    add_gvcf_site(builder, pos=100, raw_mq=1, raw_dp=1)
    add_gvcf_site(builder, pos=200, contig="chr2")
    with pysam.VariantFile(str(write(builder, tmp_path / "g.vcf.gz"))) as gvcf:
        annotations = read_gvcf_annotations(gvcf, "chr1")
        assert read_gvcf_annotations(gvcf, "chrNotThere") == {}
        assert read_gvcf_annotations(gvcf, "chr1", positions={50}) == {}
    assert list(annotations) == [100]
    site = annotations[100]
    assert (site.raw_mq, site.raw_dp, site.qual_approx, site.var_dp) == (3600, 4, 100, 20)
    assert (site.as_mq_ref, site.as_mq_alt, site.n_alt) == (900.0, 2700.0, 1.0)


def test_parse_as_raw_mq_multiallelic() -> None:
    alleles = ("A", "C", "G", "<NON_REF>")
    assert parse_as_raw_mq("10|20|30|0", alleles) == (10.0, 50.0, 2.0)
    assert parse_as_raw_mq(".|20|30|0", alleles) == (0.0, 50.0, 2.0)
    assert all(math.isnan(v) for v in parse_as_raw_mq("10|20", alleles))
    assert all(math.isnan(v) for v in parse_as_raw_mq(None, alleles))


def test_check_gvcf_header_reports_missing_annotations() -> None:
    check_gvcf_header(gvcf_builder().header, "ok.g.vcf.gz")
    with pytest.raises(ValueError, match=r"RAW_MQandDP.*QUALapprox.*VarDP.*AS_RAW_MQ"):
        check_gvcf_header(VariantBuilder(sample_ids=["s"]).header, "plain.vcf.gz")


def test_has_raw_annotations() -> None:
    assert has_raw_annotations(vcf_builder(raw_annotations=True).header)
    assert has_raw_annotations(gvcf_builder().header)
    plain = vcf_builder().header
    assert not has_raw_annotations(plain)
    assert missing_raw_annotations(plain) == ["RAW_MQandDP", "QUALapprox", "VarDP", "AS_RAW_MQ"]


def test_raw_annotation_of_a_vcf_call(tmp_path: Path) -> None:
    builder = vcf_builder(raw_annotations=True)
    add_call(
        builder, pos=100, ref="A", alt="C,G", ad=(2, 3, 4),
        raw=raw_annotation_info(3600, 4, 50, 10, "10|20|30"),
    )  # fmt: skip
    add_call(builder, pos=200)  # no raw annotations
    with pysam.VariantFile(str(write(builder, tmp_path / "v.vcf.gz"))) as vcf:
        first, second = list(vcf)
        annotation = raw_annotation(first)
        assert raw_annotation(second) is None
    assert annotation is not None
    assert (annotation.raw_mq, annotation.raw_dp) == (3600, 4)
    assert (annotation.qual_approx, annotation.var_dp) == (50, 10)
    assert (annotation.as_mq_ref, annotation.as_mq_alt, annotation.n_alt) == (10.0, 50.0, 2.0)
