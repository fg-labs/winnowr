import gzip
from pathlib import Path

import numpy as np
import pytest

from winnowr.regions import RegionSet
from winnowr.regions import parse_named_bed


def _write_bed(path: Path, lines: list[str], compress: bool = False) -> Path:
    text = "".join(f"{line}\n" for line in lines)
    if compress:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


@pytest.mark.parametrize("compress", [False, True])
def test_overlaps_span_respects_half_open_bed_coordinates(tmp_path: Path, compress: bool) -> None:
    bed = _write_bed(
        tmp_path / ("r.bed.gz" if compress else "r.bed"),
        ["track name=x", "browser position chr1", "chr1\t10\t20", "chr1\t15\t30", "chr2\t0\t1"],
        compress,
    )
    regions = RegionSet.from_bed("r", bed)
    # BED [10, 30) on chr1 covers 0-based bases 10..29
    got = regions.overlaps_span("chr1", [9, 10, 29, 30, 5, 28], [10, 11, 30, 31, 10, 40])
    np.testing.assert_array_equal(got, [False, True, True, False, False, True])
    # a deletion anchored just before the region overlaps it through its deleted bases
    np.testing.assert_array_equal(regions.overlaps_span("chr1", [8], [12]), [True])
    np.testing.assert_array_equal(regions.overlaps_span("chr2", [0, 1], [1, 2]), [True, False])
    np.testing.assert_array_equal(regions.overlaps_span("chr3", [0], [1]), [False])


def test_contigs_starting_with_track_or_browser_are_not_headers(tmp_path: Path) -> None:
    bed = _write_bed(tmp_path / "r.bed", ["track1\t0\t5", "browser_x\t0\t5", "track\tname=y"])
    assert sorted(RegionSet.from_bed("r", bed).intervals) == ["browser_x", "track1"]


def test_adjacent_intervals_are_merged(tmp_path: Path) -> None:
    bed = _write_bed(tmp_path / "r.bed", ["chr1\t0\t5", "chr1\t5\t9"])
    starts, ends = RegionSet.from_bed("r", bed).intervals["chr1"]
    assert starts.tolist() == [0] and ends.tolist() == [9]


@pytest.mark.parametrize(
    "line", ["chr1\t10", "chr1\tten\t20", "chr1\t20\t10", "chr1\t-1\t10"], ids=str
)
def test_bad_bed_lines_raise(tmp_path: Path, line: str) -> None:
    bed = _write_bed(tmp_path / "bad.bed", ["chr1\t0\t5", line])
    with pytest.raises(ValueError, match=r"bad.bed:2"):
        RegionSet.from_bed("bad", bed)


def test_parse_named_bed() -> None:
    assert parse_named_bed("seg_dups=/a/b.bed") == ("seg_dups", Path("/a/b.bed"))
    for bad in ["/a/b.bed", "=x.bed", "a=", "a-b=x.bed"]:
        with pytest.raises(ValueError, match="name=path"):
            parse_named_bed(bad)


def test_check_contigs_needs_a_shared_contig(tmp_path: Path) -> None:
    bed = _write_bed(tmp_path / "numbered.bed", ["1\t0\t5", "2\t0\t5"])
    regions = RegionSet.from_bed("r", bed)
    regions.check_contigs(["chr1", "2"])
    with pytest.raises(ValueError, match=r"numbered.bed has no intervals.*chr1, chr2.*1, 2"):
        regions.check_contigs(["chr1", "chr2"])


def test_contains_span(tmp_path: Path) -> None:
    bed = _write_bed(tmp_path / "r.bed", ["chr1\t10\t20", "chr1\t20\t25", "chr1\t30\t40"])
    regions = RegionSet.from_bed("r", bed)
    got = regions.contains_span("chr1", [10, 15, 29, 35, 5], [12, 25, 31, 41, 9])
    np.testing.assert_array_equal(got, [True, True, False, False, False])
    assert not regions.contains_span("chr9", [0], [1]).any()


def test_empty_bed_intervals_are_skipped(tmp_path: Path) -> None:
    bed = _write_bed(tmp_path / "r.bed", ["chr1\t100\t100", "chr2\t5\t5", "chr2\t10\t20"])
    regions = RegionSet.from_bed("r", bed)
    # an empty interval contains no base, so a span around it does not overlap it
    assert not regions.overlaps_span("chr1", [98], [101]).any()
    assert "chr1" not in regions.intervals
    assert regions.overlaps_span("chr2", [4, 9], [6, 10]).tolist() == [False, False]


def test_from_spans_merges_overlapping_and_adjacent_spans() -> None:
    spans = {"chr1": ([30, 10, 15, 50, 60], [40, 20, 30, 50, 70]), "chr2": ([5], [5])}
    regions = RegionSet.from_spans("s", spans)
    starts, ends = regions.intervals["chr1"]
    # [10, 20) + [15, 30) + [30, 40) merge; the empty [50, 50) is dropped
    assert (starts.tolist(), ends.tolist()) == ([10, 60], [40, 70])
    assert "chr2" not in regions.intervals
