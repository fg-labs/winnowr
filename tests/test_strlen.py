"""Tests of the per-read repeat-tract length pass."""

import logging
from pathlib import Path

import pysam
import pytest
from fgpyo.sam.builder import SamBuilder
from fgpyo.vcf.builder import VariantBuilder
from pytest import MonkeyPatch

import winnowr.strlen
from tests.util import add_call
from tests.util import vcf_builder
from tests.util import write
from winnowr.alleles import trim_alleles
from winnowr.strlen import STRLEN_COLUMNS
from winnowr.strlen import RepeatTract
from winnowr.strlen import find_tract
from winnowr.strlen import format_histogram
from winnowr.strlen import indel_records
from winnowr.strlen import measure_records
from winnowr.strlen import measure_tract
from winnowr.strlen import min_period
from winnowr.strlen import parse_histogram
from winnowr.strlen import parse_region
from winnowr.strlen import read_strlen_tables
from winnowr.strlen import tract_of_record
from winnowr.strlen import write_strlen_table
from winnowr.tools.strlen import strlen

LEFT = "CGTC" * 20
"""80 bp of left flank, ending in C so it does not extend an A run."""
TRACT = "A" * 10
RIGHT = "GCTG" * 20
REFERENCE = LEFT + TRACT + RIGHT
TRACT_START = len(LEFT)


def write_fasta(path: Path, sequence: str = REFERENCE, contig: str = "chr1") -> Path:
    """Write and index a one-contig FASTA."""
    path.write_text(f">{contig}\n{sequence}\n")
    pysam.faidx(str(path))
    return path


def test_trim_alleles() -> None:
    assert trim_alleles("CA", "C") == (1, "A", "")
    assert trim_alleles("C", "CAA") == (1, "", "AA")
    assert trim_alleles("CAT", "CGT") == (1, "A", "G")
    assert trim_alleles("CAAA", "CA") == (1, "AA", "")


def test_min_period() -> None:
    assert min_period("A") == 1
    assert min_period("AAAA") == 1
    assert min_period("TGT") == 2
    assert min_period("ACGACG") == 3
    assert min_period("ACGT") == 4


def test_find_tract_deletion_extends_over_the_run() -> None:
    tract = find_tract(REFERENCE, TRACT_START, "A", "")
    assert tract == RepeatTract(start=TRACT_START, end=TRACT_START + 10, period=1, unit="A")
    assert tract.length == 10


def test_find_tract_insertion_seeds_on_matching_side() -> None:
    right = find_tract(REFERENCE, TRACT_START, "", "AA")
    assert right == RepeatTract(start=TRACT_START, end=TRACT_START + 10, period=1, unit="A")
    left = find_tract(REFERENCE, TRACT_START + 10, "", "A")
    assert left == RepeatTract(start=TRACT_START, end=TRACT_START + 10, period=1, unit="A")


def test_find_tract_unmatched_insertion_is_empty() -> None:
    tract = find_tract(REFERENCE, TRACT_START + 10, "", "TT")
    assert tract == RepeatTract(start=TRACT_START + 10, end=TRACT_START + 10, period=1, unit="T")
    assert tract.length == 0


def test_find_tract_complex_is_none() -> None:
    assert find_tract(REFERENCE, TRACT_START, "A", "G") is None


def test_repeat_tract_sequence_keeps_phase() -> None:
    tract = RepeatTract(start=0, end=6, period=2, unit="CA")
    assert tract.sequence(6) == "CACACA"
    assert tract.sequence(3) == "CAC"
    assert tract.sequence(0) == ""


def test_tract_of_record_skips_symbolic_and_star(tmp_path: Path) -> None:
    fasta = pysam.FastaFile(str(write_fasta(tmp_path / "ref.fa")))
    tract = tract_of_record(fasta, "chr1", TRACT_START, "CA", ["*", "<NON_REF>", "C"])
    assert tract == RepeatTract(start=TRACT_START, end=TRACT_START + 10, period=1, unit="A")
    assert tract_of_record(fasta, "chr1", TRACT_START, "CA", ["*"]) is None


def test_histogram_round_trip() -> None:
    histogram = {-2: 3, 0: 10, 1: 1}
    assert format_histogram(histogram) == "-2:3;0:10;1:1"
    assert parse_histogram(format_histogram(histogram)) == histogram
    assert format_histogram({}) == "."
    assert parse_histogram(".") == {}
    assert parse_histogram(float("nan")) == {}


def _alignments(tmp_path: Path) -> pysam.AlignmentFile:
    """
    Reads over the A10 tract.

    Three have the reference length, two a 2-bp deletion and one a 1-bp insertion; the rest must
    not count.
    """
    builder = SamBuilder(r1_len=68, sd=[{"SN": "chr1", "LN": len(REFERENCE)}])
    start = TRACT_START - 30
    for i in range(3):
        builder.add_single(
            name=f"ref{i}", chrom="chr1", start=start, bases=REFERENCE[start : start + 70]
        )
    for i in range(2):
        bases = REFERENCE[start:TRACT_START] + "A" * 8 + REFERENCE[TRACT_START + 10 : start + 70]
        builder.add_single(name=f"del{i}", chrom="chr1", start=start, bases=bases, cigar="38M2D30M")
    ins = REFERENCE[start:TRACT_START] + "A" * 11 + REFERENCE[TRACT_START + 10 : start + 70]
    builder.add_single(name="ins", chrom="chr1", start=start, bases=ins, cigar="41M1I29M")
    # ends inside the right anchor: overlaps but does not span
    builder.add_single(
        name="short", chrom="chr1", start=start, bases=REFERENCE[start : TRACT_START + 11]
    )
    # duplicates and secondary alignments are skipped altogether
    dup = builder.add_single(
        name="dup", chrom="chr1", start=start, bases=REFERENCE[start : start + 70]
    )
    dup.is_duplicate = True
    builder.add_single(
        name="sec", chrom="chr1", start=start, bases=REFERENCE[start : start + 70], secondary=True
    )
    return pysam.AlignmentFile(str(builder.to_path(tmp_path / "reads.bam")))


def test_measure_tract_histogram(tmp_path: Path) -> None:
    alignments = _alignments(tmp_path)
    tract = RepeatTract(start=TRACT_START, end=TRACT_START + 10, period=1, unit="A")
    evidence = measure_tract(alignments, "chr1", tract)
    assert evidence.histogram == {-2: 2, 0: 3, 1: 1}
    assert evidence.n_spanning == 6
    assert evidence.n_overlap == 7
    assert evidence.mean_mapq == 60
    assert evidence.frac_mapq_lt20 == 0
    assert evidence.mean_tract_bq == 30


def _quality_reads(tmp_path: Path) -> pysam.AlignmentFile:
    """
    Reads over the A10 tract with known MAPQ, base qualities and anchors.

    Two reads span the tract, with MAPQ 10 and 50 and base quality 40 in the tract and 20 outside
    it; a third read has its left anchor base deleted, so it does not span the tract.
    """
    builder = SamBuilder(r1_len=70, sd=[{"SN": "chr1", "LN": len(REFERENCE)}])
    for offset, mapq in ((0, 10), (1, 50)):
        start = TRACT_START - 30 + offset
        tract_offset = TRACT_START - start
        quals = [20] * tract_offset + [40] * len(TRACT) + [20] * (70 - tract_offset - len(TRACT))
        builder.add_single(
            name=f"q{mapq}",
            chrom="chr1",
            start=start,
            bases=REFERENCE[start : start + 70],
            quals=quals,
            mapq=mapq,
        )
    start = TRACT_START - 28
    anchor = TRACT_START - 2 - start  # the left anchor (FLANK = 2) is deleted
    builder.add_single(
        name="deleted_anchor",
        chrom="chr1",
        start=start,
        bases=REFERENCE[start : start + anchor] + REFERENCE[start + anchor + 1 : start + 70],
        cigar=f"{anchor}M1D{69 - anchor}M",
    )
    return pysam.AlignmentFile(str(builder.to_path(tmp_path / "quality.bam")))


def test_measure_tract_mapq_base_quality_and_anchors(tmp_path: Path) -> None:
    tract = RepeatTract(start=TRACT_START, end=TRACT_START + 10, period=1, unit="A")
    evidence = measure_tract(_quality_reads(tmp_path), "chr1", tract)
    assert (evidence.n_overlap, evidence.n_spanning, evidence.histogram) == (3, 2, {0: 2})
    assert evidence.mean_mapq == 30 and evidence.frac_mapq_lt20 == 0.5
    assert evidence.mean_tract_bq == 40  # only the tract's bases, not the anchors or flanks


def test_measure_tract_stops_after_max_reads(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(winnowr.strlen, "MAX_READS", 1)
    tract = RepeatTract(start=TRACT_START, end=TRACT_START + 10, period=1, unit="A")
    evidence = measure_tract(_quality_reads(tmp_path), "chr1", tract)
    # the second read is counted as overlapping, then reading stops: only the first is measured
    assert (evidence.n_overlap, evidence.n_spanning, evidence.mean_mapq) == (2, 1, 10)


def test_measure_tract_without_reads(tmp_path: Path) -> None:
    alignments = _alignments(tmp_path)
    tract = RepeatTract(start=2, end=4, period=1, unit="G")
    evidence = measure_tract(alignments, "chr1", tract)
    assert evidence.n_spanning == 0
    assert evidence.histogram == {}
    assert evidence.mean_mapq is None


def test_write_and_read_strlen_table(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fasta = pysam.FastaFile(str(write_fasta(tmp_path / "ref.fa")))
    alignments = _alignments(tmp_path)
    records = [
        ("chr1", TRACT_START, "CA", ["C"]),
        ("chr1", TRACT_START, "CA", ["C"]),  # duplicate record: kept once on read
        ("chr1", TRACT_START + 1, "A", ["G"]),  # not a pure indel: skipped
        ("chr1", 10, "T", ["TAA"]),
    ]
    with caplog.at_level(logging.INFO):
        sites = list(measure_records(records, alignments, fasta))
    assert len(sites) == 3
    assert "Skipped 1 indel records without a pure insertion/deletion allele" in caplog.text
    path = tmp_path / "out.strlen.tsv.gz"
    assert write_strlen_table(path, sites) == 3
    table = read_strlen_tables([path])
    assert tuple(table.columns) == STRLEN_COLUMNS
    assert len(table) == 2
    first = table.iloc[0]
    assert (first["period"], first["tract_len"], first["unit"]) == (1, 10, "A")
    assert first["tract_start"] == TRACT_START + 1
    assert parse_histogram(first["length_histogram"]) == {-2: 2, 0: 3, 1: 1}
    second = table.iloc[1]
    assert second["length_histogram"] == "."
    assert second["mean_mapq"] != second["mean_mapq"]  # NaN without spanning reads


def test_read_strlen_tables_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("a\tb\n1\t2\n")
    with pytest.raises(ValueError, match="not a strlen table"):
        read_strlen_tables([path])
    with pytest.raises(ValueError, match="No strlen tables"):
        read_strlen_tables([])


def test_parse_region() -> None:
    assert parse_region("chr1") == ("chr1", None, None)
    assert parse_region("chr1:100-2,000") == ("chr1", 100, 2000)
    assert parse_region("HLA-A*01:01:5-10") == ("HLA-A*01:01", 5, 10)
    for bad in ("chr1:100", "chr1:5-1", ":1-2", "chr1:a-b", "chr1:0-10"):
        with pytest.raises(ValueError, match="Malformed region"):
            parse_region(bad)


def _calls_vcf(tmp_path: Path) -> Path:
    """Calls at the tract (a deletion), a SNP, and an insertion past the first 100 bp."""
    builder = vcf_builder()
    builder.header.add_line(f"##contig=<ID=chr1,length={len(REFERENCE)}>")
    add_call(builder, TRACT_START, ref="CA", alt="C")
    add_call(builder, TRACT_START + 20, ref="G", alt="T")
    add_call(builder, 150, ref="C", alt="CGG", filter="LowQual")
    return write(builder, tmp_path / "calls.vcf.gz")


def test_indel_records_selects_by_pos(tmp_path: Path) -> None:
    with pysam.VariantFile(str(_calls_vcf(tmp_path))) as vcf:
        everything = [(c, p) for c, p, _, _ in indel_records(vcf, ["chr1"])]
        assert everything == [("chr1", TRACT_START), ("chr1", 150)]
        # a deletion starting before the region overlaps it but is not selected
        shard = [(c, p) for c, p, _, _ in indel_records(vcf, [f"chr1:{TRACT_START + 1}-200"])]
        assert shard == [("chr1", 150)]
        with pytest.raises(ValueError, match="not in the VCF header"):
            indel_records(vcf, ["chr1", "chrZ"])


def test_indel_records_takes_contig_names_with_colons(tmp_path: Path) -> None:
    contig = "HLA-A*01:01:01:01"
    builder = VariantBuilder(sample_ids=["s"], sd={contig: {"ID": contig, "length": 1000}})
    builder.add(contig=contig, pos=50, ref="CA", alts=["C"])
    with pysam.VariantFile(str(write(builder, tmp_path / "hla.vcf.gz"))) as vcf:
        assert [(c, p) for c, p, _, _ in indel_records(vcf, [contig])] == [(contig, 50)]
        assert [p for _, p, _, _ in indel_records(vcf, [f"{contig}:40-60"])] == [50]


def test_strlen_tool(tmp_path: Path) -> None:
    reference = write_fasta(tmp_path / "ref.fa")
    bam = tmp_path / "reads.bam"
    _alignments(tmp_path).close()
    output = tmp_path / "out.strlen.tsv"
    strlen(
        vcf=_calls_vcf(tmp_path),
        alignments=bam,
        reference=reference,
        output=output,
        regions=["chr1"],
    )
    table = read_strlen_tables([output])
    assert table["pos"].tolist() == [TRACT_START, 150]
    assert parse_histogram(table["length_histogram"].iloc[0]) == {-2: 2, 0: 3, 1: 1}
    with pytest.raises(ValueError, match=r"Contig chrZ .* choose the regions with --regions"):
        strlen(
            vcf=_calls_vcf(tmp_path),
            alignments=bam,
            reference=reference,
            output=output,
            regions=["chrZ"],
        )
