from pathlib import Path

import pandas as pd
import pytest
from fgpyo.vcf.builder import VariantBuilder

from tests.util import add_happy_record
from tests.util import happy_builder
from tests.util import write
from winnowr.alleles import allele_deltas
from winnowr.common import VariantType
from winnowr.labels import TRUTH_DELTA_COLUMNS
from winnowr.labels import Label
from winnowr.labels import LabelJoin
from winnowr.labels import attach_labels
from winnowr.labels import check_label_join
from winnowr.labels import labelled_calls
from winnowr.labels import read_happy_labels


def _happy(tmp_path: Path) -> Path:
    builder = happy_builder()
    # TP
    add_happy_record(
        builder, 100, query=((0, 1), "TP", "gm", "SNP"), truth=((0, 1), "TP", "gm", "SNP")
    )
    # FP, no truth variant
    add_happy_record(builder, 200, query=((0, 1), "FP", ".", "SNP"))
    # FP local match
    add_happy_record(builder, 300, query=((0, 1), "FP", "lm", "INDEL"), ref="AT", alt="A")
    # FP allele match, flip gives truth: query 0/1, truth 1/1 (also a truth FN)
    add_happy_record(
        builder, 400, query=((0, 1), "FP", "am", "SNP"), truth=((1, 1), "FN", "am", "SNP")
    )
    # FP allele match, flip does NOT give truth: query 1/1, truth 1/2
    add_happy_record(
        builder,
        500,
        alt="C,G",
        query=((1, 1), "FP", "am", "SNP"),
        truth=((1, 2), "FN", "am", "SNP"),
    )
    # truth-only FN (indel) and an UNK query call
    add_happy_record(
        builder, 600, query=(None, ".", ".", "NOCALL"), truth=((0, 1), "FN", ".", "INDEL")
    )
    add_happy_record(builder, 700, query=((0, 1), "UNK", ".", "SNP"))
    # not in the requested contigs
    add_happy_record(
        builder,
        800,
        contig="chrX",
        query=((0, 1), "FP", ".", "SNP"),
        truth=((0, 1), "FN", ".", "SNP"),
    )
    return write(builder, tmp_path / "happy.vcf.gz")


def test_read_happy_labels(tmp_path: Path) -> None:
    labels = read_happy_labels(_happy(tmp_path), contigs=["chr1"])
    got = labels.labels.set_index("pos")
    assert got.index.tolist() == [100, 200, 300, 400, 500]
    assert got["label"].tolist() == [Label.KEEP, Label.DROP, Label.DROP, Label.FLIP, Label.FLIP]
    assert got["variant_type"].tolist() == ["snp", "snp", "indel", "snp", "snp"]
    assert got["flip_ok"].tolist() == [False, False, False, True, False]
    assert labels.fn_counts == {VariantType.SNP: 2, VariantType.INDEL: 1}
    assert labels.fn_positions == {
        ("chr1", 400, VariantType.SNP),
        ("chr1", 500, VariantType.SNP),
        ("chr1", 600, VariantType.INDEL),
    }


def test_attach_labels_only_labels_pass_calls_of_the_matching_type(tmp_path: Path) -> None:
    labels = read_happy_labels(_happy(tmp_path), contigs=["chr1"])
    features = pd.DataFrame({
        "chrom": ["chr1"] * 5,
        "pos": [100, 200, 300, 400, 999],
        "ref": "A",
        "alt": "C",
        "variant_type": ["snp", "snp", "snp", "snp", "snp"],
        "filter": ["PASS", "LowQual", "PASS", "PASS", "PASS"],
    })
    labelled, _ = attach_labels(features, labels)
    assert labelled["label"].tolist()[0] == Label.KEEP
    assert labelled["label"].isna().tolist() == [False, True, True, False, True]
    assert labelled["flip_ok"].tolist() == [False, False, False, True, False]
    calls = labelled_calls(labelled, VariantType.SNP)
    assert calls["pos"].tolist() == [100, 400]


def test_duplicate_happy_records_and_calls_are_collapsed(tmp_path: Path) -> None:
    builder = happy_builder()
    # two records for one call: the label comes from the first, flip_ok from any of them
    for truth_gt in ((1, 2), (1, 1)):
        add_happy_record(
            builder,
            100,
            alt="C,G",
            query=((0, 1), "FP", "am", "SNP"),
            truth=(truth_gt, "FN", "am", "SNP"),
        )
    add_happy_record(builder, 200, query=((0, 1), "FP", ".", "SNP"))
    labels = read_happy_labels(write(builder, tmp_path / "dup.vcf.gz"), contigs=["chr1"])
    got = labels.labels.set_index("pos")
    assert got.index.tolist() == [100, 200]
    assert (got.loc[100, "label"], got.loc[100, "flip_ok"]) == (Label.FLIP, True)

    features = pd.DataFrame({
        "chrom": ["chr1"] * 3,
        "pos": [100, 100, 200],
        "variant_type": ["snp"] * 3,
        "filter": ["PASS"] * 3,
    })
    calls = labelled_calls(attach_labels(features, labels)[0], VariantType.SNP)
    assert calls["pos"].tolist() == [100, 200]


def test_happy_vcf_without_records_on_the_contigs_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"happy.vcf.gz: No hap.py records on the extracted"):
        read_happy_labels(_happy(tmp_path), contigs=["1", "2"])


def test_non_happy_vcf_is_an_error(tmp_path: Path) -> None:
    plain = VariantBuilder(sample_ids=["s"])
    plain.add(pos=100)
    with pytest.raises(ValueError, match="hap.py"):
        read_happy_labels(write(plain, tmp_path / "p.vcf.gz"))


def test_allele_deltas() -> None:
    assert allele_deltas(("CA", "C"), "0/1") == (-1, 0)
    assert allele_deltas(("C", "CAA", "CA"), "1/2") == (1, 2)
    assert allele_deltas(("C", "CAA"), "1/1") == (2, 2)
    assert allele_deltas(("C", "CAA"), "./.") is None
    assert allele_deltas(("C", "*", "CA"), "1/2") is None
    assert allele_deltas(("C", "<NON_REF>"), "0/1") is None
    assert allele_deltas(("C", "CA"), "0/2") is None
    assert allele_deltas((), "0/1") is None


def test_truth_deltas_attach_to_every_call(tmp_path: Path) -> None:
    labels = read_happy_labels(_happy(tmp_path), contigs=["chr1"])
    deltas = labels.truth_deltas.set_index("pos")
    # positions without a truth genotype (200, 300, 700) have no row; 500 is truth 1/2 of C,G
    assert deltas.index.tolist() == [100, 400, 500, 600]
    assert [deltas.loc[600, c] for c in TRUTH_DELTA_COLUMNS] == [0.0, 0.0]
    features = pd.DataFrame({
        "chrom": ["chr1"] * 3,
        "pos": [100, 600, 999],
        "ref": ["A", "A", "A"],
        "alt": ["C", "C", "C"],
        "variant_type": ["snp", "indel", "snp"],
        "filter": ["PASS", "LowQual", "PASS"],
    })
    labelled, _ = attach_labels(features, labels)
    assert labelled["truth_delta_lo"].tolist()[:2] == [0.0, 0.0]
    assert labelled["truth_delta_hi"].isna().tolist() == [False, False, True]


def test_truth_deltas_of_an_indel_record(tmp_path: Path) -> None:
    builder = happy_builder()
    add_happy_record(
        builder,
        100,
        ref="CAA",
        alt="C,CAAA",
        query=((0, 1), "FP", "am", "INDEL"),
        truth=((1, 2), "FN", "am", "INDEL"),
    )
    add_happy_record(
        builder,
        100,
        ref="CAA",
        alt="CA",
        query=((0, 1), "FP", "am", "INDEL"),
        truth=((1, 1), "FN", "am", "INDEL"),
    )
    labels = read_happy_labels(write(builder, tmp_path / "indel.vcf.gz"), contigs=["chr1"])
    # the first record with a truth genotype wins
    assert labels.truth_deltas[list(TRUTH_DELTA_COLUMNS)].to_numpy().tolist() == [[-2.0, 1.0]]


@pytest.mark.parametrize("snp_first", [True, False])
def test_truth_deltas_are_keyed_by_variant_type(tmp_path: Path, snp_first: bool) -> None:
    builder = happy_builder()
    for is_snp in (True, False) if snp_first else (False, True):
        add_happy_record(
            builder,
            100,
            alt="G" if is_snp else "AT",
            query=((1, 1), "TP", "gm", "SNP") if is_snp else ((0, 1), "FP", "am", "INDEL"),
            truth=((1, 1), "TP", "gm", "SNP") if is_snp else ((0, 1), "FN", "am", "INDEL"),
        )
    labels = read_happy_labels(write(builder, tmp_path / "mixed.vcf.gz"), contigs=["chr1"])
    features = pd.DataFrame({
        "chrom": ["chr1"] * 2,
        "pos": [100, 100],
        "variant_type": ["snp", "indel"],
        "filter": ["PASS"] * 2,
    })
    labelled, _ = attach_labels(features, labels)
    # whatever the record order, the indel call gets the truth indel's length changes
    assert labelled[list(TRUTH_DELTA_COLUMNS)].to_numpy().tolist() == [[0.0, 0.0], [0.0, 1.0]]


def test_attach_labels_counts_where_the_labels_went(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    labels = read_happy_labels(_happy(tmp_path), contigs=["chr1"])
    # labels at 100-500: 100 and 400 label PASS calls, 200 is at a non-PASS call (hap.py was run
    # without --pass-only), 300 (an indel) and 500 match no call
    features = pd.DataFrame({
        "chrom": ["chr1"] * 4,
        "pos": [100, 200, 300, 400],
        "ref": "A",
        "alt": "C",
        "variant_type": "snp",
        "filter": ["PASS", "LowQual", "PASS", "PASS"],
    })
    _, join = attach_labels(features, labels)
    assert join == LabelJoin(n_labels=5, n_pass=2, n_padded_snp=0, n_non_pass=1, n_unmatched=2)
    check_label_join(join)
    assert "--pass-only" in caplog.text and "2 TP/FP query records" in caplog.text
    with pytest.raises(ValueError, match="was hap.py run on this VCF"):
        check_label_join(
            LabelJoin(n_labels=4, n_pass=1, n_padded_snp=0, n_non_pass=0, n_unmatched=3)
        )


def test_padded_snp_takes_the_label_at_its_differing_base(tmp_path: Path) -> None:
    builder = happy_builder()
    # hap.py trimmed the caller's TTAGA>TTCGA,* at 100 to A>C at 102
    add_happy_record(builder, 102, query=((0, 1), "TP", "gm", "SNP"))
    # and TTAGA>TTCGA at 200 to A>C at 202, where the caller also has a SNP of its own
    add_happy_record(builder, 202, query=((0, 1), "FP", ".", "SNP"))
    labels = read_happy_labels(write(builder, tmp_path / "padded.vcf.gz"), contigs=["chr1"])
    features = pd.DataFrame({
        "chrom": ["chr1"] * 3,
        "pos": [100, 200, 202],
        "ref": ["TTAGA", "TTAGA", "A"],
        "alt": ["TTCGA,*", "TTCGA", "G"],
        "variant_type": "snp",
        "filter": "PASS",
    })
    labelled, join = attach_labels(features, labels)
    assert labelled["label"].tolist()[0] == Label.KEEP
    assert labelled["label"].isna().tolist() == [False, True, False]
    assert (join.n_pass, join.n_padded_snp, join.n_unmatched) == (2, 1, 0)


def test_the_first_hap_py_record_of_a_call_gives_its_label(tmp_path: Path) -> None:
    builder = happy_builder()
    # a TP then a flip-ok FP for one call: the label is the first's, and a KEEP is never flip_ok
    add_happy_record(
        builder, 100, query=((0, 1), "TP", "gm", "SNP"), truth=((0, 1), "TP", "gm", "SNP")
    )
    add_happy_record(
        builder, 100, query=((0, 1), "FP", "am", "SNP"), truth=((1, 1), "FN", "am", "SNP")
    )
    # a FLIP that is not ok, then one that is: still FLIP, and flip_ok from either
    for truth_gt in ((1, 2), (1, 1)):
        add_happy_record(
            builder,
            200,
            alt="C,G",
            query=((0, 1), "FP", "am", "SNP"),
            truth=(truth_gt, "FN", "am", "SNP"),
        )
    labels = read_happy_labels(write(builder, tmp_path / "first.vcf.gz"), contigs=["chr1"])
    got = labels.labels.set_index("pos")
    assert (got.loc[100, "label"], got.loc[100, "flip_ok"]) == (Label.KEEP, False)
    assert (got.loc[200, "label"], got.loc[200, "flip_ok"]) == (Label.FLIP, True)


def test_non_pass_calls_are_never_flip_ok(tmp_path: Path) -> None:
    labels = read_happy_labels(_happy(tmp_path), contigs=["chr1"])
    # 400 is a flip-ok FLIP in hap.py, but this call is not PASS
    features = pd.DataFrame({
        "chrom": ["chr1"],
        "pos": [400],
        "ref": "A",
        "alt": "C",
        "variant_type": "snp",
        "filter": "LowQual",
    })
    labelled, _ = attach_labels(features, labels)
    assert labelled["label"].isna().all() and not labelled["flip_ok"].any()
