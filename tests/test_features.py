import logging
import math
from pathlib import Path

import pandas as pd
import pytest
from fgpyo.vcf.builder import VariantBuilder

from tests.util import add_call
from tests.util import add_gvcf_site
from tests.util import gvcf_builder
from tests.util import raw_annotation_info
from tests.util import vcf_builder
from tests.util import write
from winnowr.alleles import normalize_gt
from winnowr.alleles import variant_type_of
from winnowr.common import VariantType
from winnowr.features import RAW_COLUMNS
from winnowr.features import add_derived_features
from winnowr.features import as_raw_frame
from winnowr.features import extract_raw_records
from winnowr.features import feature_names
from winnowr.features import strand_odds_ratio
from winnowr.regions import RegionSet


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    vcf = vcf_builder()
    add_call(vcf, pos=100, gt=(0, 1), ad=(6, 2))
    add_call(vcf, pos=200, gt=(1, 1), ref="AT", alt="A", ad=(0, 9), filter="LowQual")
    add_call(vcf, pos=300, gt=(1, 0), ad=(5, 5), phased=True)
    add_call(vcf, pos=400, contig="chr2", ref="A", alt="<DEL>")  # unclassified: skipped
    add_call(vcf, pos=500, contig="chrX")  # not in the extracted contigs
    # spanning deletion listed first; AD of `*` is 7, of the insertion 3; AD of alts missing
    add_call(vcf, pos=6000, gt=(0, 2), ref="A", alt="*,AT", ad=(5, 7, 3), filter="LowQual")
    add_call(vcf, pos=7000, gt=(0, 1), ad=(4, None), filter="LowQual")
    gvcf = gvcf_builder()
    add_gvcf_site(gvcf, pos=100, raw_mq=3600, raw_dp=4, qual_approx=50, var_dp=10)
    add_gvcf_site(gvcf, pos=200, ref="AT", alt="A", raw_dp=12, as_raw_mq="100.0|300.0|0.0")
    # no gVCF site at 300
    return write(vcf, tmp_path / "s.vcf.gz"), write(gvcf, tmp_path / "s.g.vcf.gz")


def test_extract_and_derive_features(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    vcf, gvcf = _inputs(tmp_path)
    with caplog.at_level(logging.INFO):
        raw = extract_raw_records(vcf, gvcf, contigs=["chr1", "chr2"])
    assert "Skipped 1 records that are neither SNPs nor indels" in caplog.text
    assert raw["pos"].tolist() == [100, 200, 300, 6000, 7000]
    assert raw["variant_type"].tolist() == ["snp", "indel", "snp", "indel", "snp"]
    assert raw["filter"].tolist() == ["PASS", "LowQual", "PASS", "LowQual", "LowQual"]
    assert raw["gt"].tolist() == ["0/1", "1/1", "0/1", "0/2", "0/1"]

    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t199\t200\n")
    df = add_derived_features(raw, [RegionSet.from_bed("seg", bed)]).set_index("pos")
    snp, indel = df.loc[100], df.loc[200]
    assert snp["mq"] == pytest.approx(30.0)  # sqrt(3600 / 4)
    assert snp["qd"] == pytest.approx(5.0)  # 50 / 10
    assert snp["ab"] == pytest.approx(0.25)
    assert snp["as_alt_frac"] == pytest.approx(0.5)
    # SB (5, 5, 4, 6) + 1 -> rf=6 rr=6 af=5 ar=7: R = 42/30
    expected_sor = math.log(42 / 30 + 30 / 42) + math.log(6 / 6) - math.log(5 / 7)
    assert snp["sor"] == pytest.approx(expected_sor) and snp["sor"] > 0.1
    # dp_norm: raw DP / median raw DP over records with a gVCF match (4 and 12 -> median 8)
    assert snp["dp_norm"] == pytest.approx(0.5)
    assert indel["dp_norm"] == pytest.approx(1.5)
    assert (snp["het"], snp["flippable"], indel["het"], indel["flippable"]) == (1, True, 0, True)
    assert (indel["indel_len"], indel["indel_abs_len"]) == (-1.0, 1.0)
    assert (indel["str_period"], indel["str_repeats"]) == (1.0, 4.0)
    assert indel["as_alt_frac"] == pytest.approx(0.75)
    assert (snp["region_seg"], indel["region_seg"]) == (0.0, 1.0)
    assert indel["nb_filt_500"] == 0 and snp["nb_filt_500"] == 1
    assert snp["nb_calls_500"] == 2 and snp["nb_dist"] == 100
    assert (snp["pl_ref"], snp["pl_het"], snp["pl_hom"]) == (0.0, 10.0, 20.0)
    assert (snp["excess_het"], snp["ad_alt_total"], snp["n_alt_rec"]) == (3.0, 2.0, 1.0)
    assert snp["overlaps_pass"] and not indel["overlaps_pass"]
    spanning = df.loc[6000]
    assert (spanning["ad_alt"], spanning["ad_alt_total"]) == (3.0, 10.0)
    assert (spanning["ab"], spanning["indel_len"]) == (3 / 8, 1.0)
    no_alt_depth = df.loc[7000]
    assert all(pd.isna(no_alt_depth[c]) for c in ("ad_alt", "ad_alt_total", "ab"))
    missing = df.loc[300]
    assert all(pd.isna(missing[c]) for c in ("mq", "qd", "as_alt_frac", "dp_norm", "n_alt"))
    assert set(feature_names(VariantType.INDEL, ["seg"])) <= set(df.columns)


def test_contigs_without_records_give_an_empty_table(tmp_path: Path) -> None:
    vcf, gvcf = _inputs(tmp_path)
    raw = extract_raw_records(vcf, gvcf, contigs=["chr3"])
    df = add_derived_features(raw, [])
    assert df.empty and set(feature_names(VariantType.INDEL, [])) <= set(df.columns)
    assert df["pos"].dtype == "int64"


def test_unknown_contig_is_an_error(tmp_path: Path) -> None:
    vcf, gvcf = _inputs(tmp_path)
    with pytest.raises(ValueError, match="chrUn"):
        extract_raw_records(vcf, gvcf, contigs=["chr1", "chrUn"])


def test_contig_missing_from_the_gvcf_is_an_error(tmp_path: Path) -> None:
    vcf, _ = _inputs(tmp_path)
    builder = VariantBuilder(sample_ids=["s"], sd={"1": {"ID": "1", "length": 10_000}})
    for key in ("RAW_MQandDP", "QUALapprox", "VarDP", "AS_RAW_MQ"):
        builder.header.add_line(f'##INFO=<ID={key},Number=.,Type=String,Description="x">')
    numbered = write(builder, tmp_path / "numbered.g.vcf.gz")
    with pytest.raises(ValueError, match=r"gVCF .*numbered.g.vcf.gz: chr1"):
        extract_raw_records(vcf, numbered, contigs=["chr1"])


def test_gvcf_without_annotations_is_an_error(tmp_path: Path) -> None:
    vcf, _ = _inputs(tmp_path)
    plain = VariantBuilder(sample_ids=["s"])
    plain.add(pos=100)
    with pytest.raises(ValueError, match="RAW_MQandDP"):
        extract_raw_records(vcf, write(plain, tmp_path / "p.vcf.gz"), contigs=["chr1"])


def test_multi_sample_vcf_is_an_error(tmp_path: Path) -> None:
    _, gvcf = _inputs(tmp_path)
    two = VariantBuilder(sample_ids=["a", "b"])
    two.add(pos=100, alts="C")
    with pytest.raises(ValueError, match="exactly one sample"):
        extract_raw_records(write(two, tmp_path / "two.vcf.gz"), gvcf, contigs=["chr1"])


def _annotated_vcf(tmp_path: Path) -> Path:
    """`_inputs`' VCF, keeping the raw annotations `_inputs`' gVCF has (none at 300)."""
    vcf = vcf_builder(raw_annotations=True)
    add_call(vcf, pos=100, gt=(0, 1), ad=(6, 2), raw=raw_annotation_info(3600, 4, 50, 10))
    add_call(
        vcf, pos=200, gt=(1, 1), ref="AT", alt="A", ad=(0, 9), filter="LowQual",
        raw=raw_annotation_info(raw_dp=12, as_raw_mq="100.0|300.0"),
    )  # fmt: skip
    add_call(vcf, pos=300, gt=(1, 0), ad=(5, 5), phased=True)
    add_call(vcf, pos=400, contig="chr2", ref="A", alt="<DEL>")
    add_call(vcf, pos=500, contig="chrX")
    add_call(vcf, pos=6000, gt=(0, 2), ref="A", alt="*,AT", ad=(5, 7, 3), filter="LowQual")
    add_call(vcf, pos=7000, gt=(0, 1), ad=(4, None), filter="LowQual")
    return write(vcf, tmp_path / "annotated.vcf.gz")


def test_raw_annotations_from_the_vcf_match_the_gvcf(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    vcf, gvcf = _inputs(tmp_path)
    from_gvcf = extract_raw_records(vcf, gvcf, contigs=["chr1", "chr2"])
    annotated = _annotated_vcf(tmp_path)
    with caplog.at_level(logging.INFO):
        from_vcf = extract_raw_records(annotated, contigs=["chr1", "chr2"])
    assert "3 records lack RAW_MQandDP (set to missing)" in caplog.text
    pd.testing.assert_frame_equal(from_vcf, from_gvcf)
    pd.testing.assert_frame_equal(
        add_derived_features(from_vcf, []), add_derived_features(from_gvcf, [])
    )


def test_vcf_raw_annotations_win_over_a_gvcf(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    vcf = vcf_builder(raw_annotations=True)
    add_call(vcf, pos=100, raw=raw_annotation_info(3600, 4, 50, 10, "900.0|2700.0"))
    gvcf = gvcf_builder()
    add_gvcf_site(gvcf, pos=100, raw_mq=1, raw_dp=1, qual_approx=1, var_dp=1)
    with caplog.at_level(logging.WARNING):
        raw = extract_raw_records(
            write(vcf, tmp_path / "v.vcf.gz"), write(gvcf, tmp_path / "g.vcf.gz"), ["chr1"]
        )
    assert "the gVCF" in caplog.text and "is not read" in caplog.text
    row = raw.iloc[0]
    assert (row["raw_mq"], row["raw_dp"], row["qual_approx"], row["var_dp"]) == (3600, 4, 50, 10)
    assert (row["as_mq_ref"], row["as_mq_alt"], row["n_alt"]) == (900.0, 2700.0, 1.0)


def test_vcf_as_raw_mq_without_one_value_per_allele_is_missing(tmp_path: Path) -> None:
    vcf = vcf_builder(raw_annotations=True)
    # an untrimmed AS_RAW_MQ (still with a <NON_REF> value) is not one value per allele
    add_call(vcf, pos=100, raw=raw_annotation_info(as_raw_mq="18000.00|18000.00|0.00"))
    raw = extract_raw_records(write(vcf, tmp_path / "v.vcf.gz"), contigs=["chr1"]).iloc[0]
    assert all(math.isnan(raw[c]) for c in ("as_mq_ref", "as_mq_alt", "n_alt"))
    assert (raw["raw_mq"], raw["qual_approx"]) == (36000, 100)


def test_vcf_without_raw_annotations_needs_a_gvcf(tmp_path: Path) -> None:
    vcf, _ = _inputs(tmp_path)
    with pytest.raises(
        ValueError,
        match=r"lacks the INFO annotations \['RAW_MQandDP', 'QUALapprox', 'VarDP', 'AS_RAW_MQ'\] "
        r"and no gVCF was given",
    ):
        extract_raw_records(vcf, contigs=["chr1"])
    # a VCF declaring only some of them still needs the gVCF
    partial = vcf_builder()
    partial.header.add_line('##INFO=<ID=RAW_MQandDP,Number=2,Type=Integer,Description="x">')
    add_call(partial, pos=100)
    with pytest.raises(ValueError, match=r"\['QUALapprox', 'VarDP', 'AS_RAW_MQ'\]"):
        extract_raw_records(write(partial, tmp_path / "partial.vcf.gz"), contigs=["chr1"])


@pytest.mark.parametrize(
    "ref, alts, expected",
    [
        ("A", ["C"], VariantType.SNP),
        ("A", ["C", "*"], VariantType.SNP),
        ("A", ["AT"], VariantType.INDEL),
        ("A", ["C", "AT"], VariantType.INDEL),
        ("AC", ["GT"], None),
        ("TTAGA", ["CTAGA"], VariantType.SNP),
        ("TTAGA", ["CTAGA", "TTAGT"], VariantType.SNP),
        ("TTAGA", ["CTAGT"], None),
        ("TTAGA", ["CTAGA", "T"], VariantType.INDEL),
        ("A", ["<DEL>"], None),
        ("A", ["*"], None),
    ],
)
def test_variant_type_of(ref: str, alts: list[str], expected: VariantType | None) -> None:
    assert variant_type_of(ref, alts) is expected


def test_normalize_gt() -> None:
    assert normalize_gt((1, 0)) == "0/1"
    assert normalize_gt((10, 2)) == "2/10"
    assert normalize_gt((None, 1)) == "./."
    assert normalize_gt((1,)) == "./."
    assert normalize_gt(None) == "./."


def test_strand_odds_ratio_matches_gatk() -> None:
    # the worked example in GATK's StrandOddsRatio docs (SB_TABLE=1450,345,160,212 -> SOR=0.592),
    # a balanced table, and a table with a missing count
    sor = strand_odds_ratio(
        pd.Series([1450.0, 10.0, 5.0]),
        pd.Series([345.0, 10.0, 5.0]),
        pd.Series([160.0, 10.0, None]),
        pd.Series([212.0, 10.0, 6.0]),
    )
    assert sor[0] == pytest.approx(0.592, abs=5e-4)
    assert sor[1] == pytest.approx(math.log(2.0))
    assert math.isnan(sor[2])


def _raw(rows: list[tuple[str, int, str, str, str]]) -> pd.DataFrame:
    """A raw frame with only `(chrom, pos, ref, alt, filter)` set."""
    columns: dict[str, list[object]] = {column: [] for column in RAW_COLUMNS}
    for chrom, pos, ref, alt, filter_ in rows:
        values = {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "filter": filter_}
        values |= {"variant_type": "snp", "gt": "0/1"}
        for column in RAW_COLUMNS:
            columns[column].append(values.get(column))
    return as_raw_frame(columns)


def test_overlaps_pass() -> None:
    raw = _raw([
        ("chr1", 100, "ACGT", "A", "PASS"),  # PASS deletion over [99, 103)
        ("chr1", 102, "G", "C", "LowQual"),  # inside the PASS deletion's REF span
        ("chr1", 104, "A", "C", "LowQual"),  # adjacent to it, not overlapping
        ("chr1", 200, "ACGTA", "A", "LowQual"),  # deletion over [199, 204) ...
        ("chr1", 203, "T", "C", "PASS"),  # ... covering a later PASS SNP
        ("chr2", 102, "G", "C", "LowQual"),  # same position as the second call, other contig
    ])
    got = add_derived_features(raw, []).set_index(["chrom", "pos"])["overlaps_pass"]
    assert got.tolist() == [True, True, False, True, True, False]
