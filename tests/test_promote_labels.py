from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from fgpyo.vcf.builder import VariantBuilder

from tests.util import write
from winnowr.common import VariantType
from winnowr.features import RAW_COLUMNS
from winnowr.features import add_derived_features
from winnowr.features import as_raw_frame
from winnowr.promote_labels import PromoteLabel
from winnowr.promote_labels import attach_promote_labels
from winnowr.promote_labels import promote_calls
from winnowr.promote_labels import read_truth_variants
from winnowr.regions import RegionSet
from winnowr.table import FeatureTableInfo
from winnowr.table import read_feature_table
from winnowr.table import write_feature_table


def _reference(tmp_path: Path) -> Path:
    bases = list("ACGT" * 300)  # 1200 bp, no long repeats
    bases[799:807] = list("GCACACAT")  # 1-based 800..807
    path = tmp_path / "ref.fa"
    path.write_text(">chr1\n" + "".join(bases) + "\n")
    pysam.faidx(str(path))
    return path


def _truth(tmp_path: Path) -> Path:
    builder = VariantBuilder(sample_ids=["truth"])
    for pos, ref, alt, gt in [
        (100, "A", "C", (0, 1)),
        (200, "A", "C", (1, 1)),
        (300, "A", "C", (0, 1)),
        (700, "A", "C,G", (1, 2)),  # multi-allelic FN
        (800, "GCA", "G", (0, 1)),  # left-aligned deletion
        (900, "A", "C", (0, 1)),  # an FN only of the other variant type
    ]:
        builder.add(pos=pos, ref=ref, alts=alt, samples={"truth": {"GT": gt}})
    return write(builder, tmp_path / "truth.vcf.gz")


def _table() -> pd.DataFrame:
    rows = [
        # pos, ref, alt, filter, gt, variant type, overlaps PASS
        (100, "A", "C", "LowQual", "0/1", "snp", False),  # promote
        (200, "A", "C", "LowQual", "0/1", "snp", False),  # promote + flip
        (300, "A", "C", "LowQual", "0/1", "snp", False),  # truth not a hap.py FN -> stay
        (400, "A", "C", "LowQual", "0/1", "snp", False),  # not in truth -> stay
        (804, "ACA", "A", "LowQual", "0/1", "indel", False),  # matches after left-alignment
        (700, "A", "C,G", "LowQual", "0/1", "snp", False),  # truth 1/2: neither same nor flip
        (900, "A", "C", "LowQual", "0/1", "snp", False),  # truth is an indel FN here -> stay
        (1100, "A", "C", "LowQual", "0/1", "snp", False),  # outside the confident BED
        (500, "A", "C", "LowQual", "0/1", "snp", True),  # overlaps a PASS call
        (600, "A", "C", "PASS", "0/1", "snp", True),  # PASS: not a candidate
    ]
    frame = pd.DataFrame(
        rows, columns=["pos", "ref", "alt", "filter", "gt", "variant_type", "overlaps_pass"]
    )
    return frame.assign(chrom="chr1")


def test_attach_promote_labels(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    fns = {
        ("chr1", 100, VariantType.SNP),
        ("chr1", 200, VariantType.SNP),
        ("chr1", 700, VariantType.SNP),
        ("chr1", 800, VariantType.INDEL),
        ("chr1", 900, VariantType.INDEL),
    }
    truth = read_truth_variants(_truth(tmp_path), fns, reference)
    assert ("chr1", 800, "GCA", "G") in truth and ("chr1", 300, "A", "C") not in truth
    bed = tmp_path / "conf.bed"
    bed.write_text("chr1\t0\t1000\n")
    labelled = attach_promote_labels(
        _table(), truth, fns, RegionSet.from_bed("conf", bed), reference
    )
    expected = [
        PromoteLabel.PROMOTE,
        PromoteLabel.PROMOTE_FLIP,
        PromoteLabel.STAY,
        PromoteLabel.STAY,
        PromoteLabel.PROMOTE,
        PromoteLabel.STAY,
        PromoteLabel.STAY,
        np.nan,
        np.nan,
        np.nan,
    ]
    np.testing.assert_array_equal(labelled["promote_label"].to_numpy(), expected)
    snp_calls = promote_calls(labelled, VariantType.SNP)["pos"].tolist()
    assert snp_calls == [100, 200, 300, 400, 700, 900]


def test_a_truth_fn_is_promoted_at_most_once(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    fns = {("chr1", 800, VariantType.INDEL)}
    truth = read_truth_variants(_truth(tmp_path), fns, reference)
    bed = tmp_path / "conf.bed"
    bed.write_text("chr1\t0\t1000\n")
    # three representations of the GCA>G deletion at 800: one without a genotype (does not use up
    # the truth record), then two that would each turn the same FN into a TP
    table = pd.DataFrame({
        "chrom": "chr1",
        "pos": [802, 804, 802],
        "ref": "ACA",
        "alt": "A",
        "filter": "LowQual",
        "gt": ["./.", "0/1", "0/1"],
        "variant_type": "indel",
        "overlaps_pass": False,
    })
    labelled = attach_promote_labels(table, truth, fns, RegionSet.from_bed("conf", bed), reference)
    assert labelled["promote_label"].tolist() == [
        PromoteLabel.STAY,
        PromoteLabel.PROMOTE,
        PromoteLabel.STAY,
    ]


def test_fn_contig_missing_from_the_truth_vcf_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Contig 1 .* truth VCF .*truth.vcf.gz"):
        read_truth_variants(_truth(tmp_path), {("1", 100, VariantType.SNP)})


def test_without_reference_indels_only_match_when_already_normalized(tmp_path: Path) -> None:
    fns = {("chr1", 800, VariantType.INDEL)}
    truth = read_truth_variants(_truth(tmp_path), fns)
    bed = tmp_path / "conf.bed"
    bed.write_text("chr1\t0\t1000\n")
    labelled = attach_promote_labels(_table(), truth, fns, RegionSet.from_bed("conf", bed))
    assert labelled["promote_label"].iloc[4] == PromoteLabel.STAY


def test_empty_table_round_tripped_through_parquet(tmp_path: Path) -> None:
    raw = as_raw_frame({column: [] for column in RAW_COLUMNS})
    path = tmp_path / "empty.parquet"
    write_feature_table(path, add_derived_features(raw, []), FeatureTableInfo("s", ("chr1",), ()))
    table, _ = read_feature_table(path)
    bed = tmp_path / "conf.bed"
    bed.write_text("chr1\t0\t1000\n")
    labelled = attach_promote_labels(table, {}, set(), RegionSet.from_bed("conf", bed))
    assert labelled.empty and "promote_label" in labelled.columns


def test_candidates_must_lie_inside_the_confident_regions(tmp_path: Path) -> None:
    bed = tmp_path / "conf.bed"
    bed.write_text("chr1\t0\t1000\n")
    # a deletion starting inside [0, 1000) whose REF ends past it is not a candidate
    table = pd.DataFrame({
        "chrom": "chr1",
        "pos": [998, 999],
        "ref": ["ACG", "ACG"],
        "alt": ["A", "A"],
        "filter": "LowQual",
        "gt": "0/1",
        "variant_type": "indel",
        "overlaps_pass": False,
    })
    labelled = attach_promote_labels(table, {}, set(), RegionSet.from_bed("conf", bed))
    assert labelled["promote_label"].isna().tolist() == [False, True]


def test_the_first_truth_record_of_a_key_wins(tmp_path: Path) -> None:
    builder = VariantBuilder(sample_ids=["truth"])
    for gt in ((0, 1), (1, 1)):
        builder.add(pos=100, ref="A", alts="C", samples={"truth": {"GT": gt}})
    truth = read_truth_variants(
        write(builder, tmp_path / "dup.vcf.gz"), {("chr1", 100, VariantType.SNP)}
    )
    assert truth[("chr1", 100, "A", "C")].gt == "0/1"
