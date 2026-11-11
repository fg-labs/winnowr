"""
Per-read repeat-tract lengths at indel calls (a lightweight homopolymer/STR "stutter profile").

For each indel record the repeat tract around the event is located in the reference
(`tract_of_record`): the first alternate allele that is a pure insertion or deletion (after
trimming the alleles' shared suffix, then prefix) gives the inserted/deleted sequence, whose
minimal period `k` defines the repeat unit; the tract is the maximal period-`k` run of the
reference around the event. A deletion seeds the run on the deleted bases; an insertion on the
`k` reference bases next to the event that match the inserted sequence (or, if neither side
matches, the tract is empty at the event).

Each primary, non-duplicate, QC-passing, mapped read (any MAPQ) that is aligned (M/=/X) at both
anchors, `FLANK` bases outside each end of the tract, "spans" it; its observed tract length is
the number of query bases between the anchors. The per-site output is the histogram of observed
minus reference tract length (bp), plus spanning-read MAPQ and tract base-quality summaries
(`TractEvidence`).
"""

import logging
from collections import Counter
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import pysam
from fgpyo.io import to_writer
from fgpyo.sam import CigarOp

from winnowr.alleles import trim_alleles
from winnowr.alleles import variant_type_of
from winnowr.common import VariantType

logger = logging.getLogger(__name__)

FLANK: Final[int] = 2
"""Reference bases a read must align on outside each end of the tract to span it."""

CONTEXT: Final[int] = 600
"""Reference bases fetched on each side of a record to find its tract."""

MAX_READS: Final[int] = 600
"""Stop reading a site after this many overlapping reads."""

_SKIP_FLAGS: Final[int] = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
"""Unmapped, secondary, QC-fail, duplicate and supplementary reads are skipped."""

CRAM_REQUIRED_FIELDS: Final[int] = 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x400
"""htslib SAM_FLAG, SAM_RNAME, SAM_POS, SAM_MAPQ, SAM_CIGAR and SAM_QUAL: the only fields read, so
CRAM decoding can skip read names, bases and tags (about 2.4x faster)."""


def open_alignments(path: Path, reference: Path) -> pysam.AlignmentFile:
    """
    Open a BAM/CRAM for the tract pass, decoding only the fields it needs from CRAM.

    Args:
        path: the indexed BAM/CRAM.
        reference: the reference FASTA (to decode CRAM).

    Returns:
        The open alignment file.
    """
    return pysam.AlignmentFile(
        str(path),
        reference_filename=str(reference),
        format_options=[f"required_fields={CRAM_REQUIRED_FIELDS}"],
    )


STRLEN_COLUMNS: Final[tuple[str, ...]] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "period",
    "tract_len",
    "unit",
    "tract_start",
    "n_overlap",
    "n_spanning",
    "length_histogram",
    "mean_mapq",
    "frac_mapq_lt20",
    "mean_tract_bq",
)
"""The columns of a strlen table (see `StrlenSite`), in file order."""


@dataclass(frozen=True)
class RepeatTract:
    """
    A repeat tract in the reference.

    Attributes:
        start: the 0-based start of the tract.
        end: the 0-based exclusive end (equal to `start` for an empty tract).
        period: the repeat period `k`.
        unit: the first `k` bases of the tract (the inserted unit for an empty tract).
    """

    start: int
    end: int
    period: int
    unit: str

    @property
    def length(self) -> int:
        """The tract length in bp."""
        return self.end - self.start

    def sequence(self, length: int) -> str:
        """
        A periodic sequence in the tract's phase, e.g. the tract itself at `length == self.length`.

        Args:
            length: the sequence length (>= 0).

        Returns:
            The first `length` bases of the tract's unit repeated.
        """
        copies = length // len(self.unit) + 1 if self.unit else 0
        return (self.unit * copies)[:length]


def min_period(seq: str) -> int:
    """
    The smallest `k` such that `seq` is periodic with period `k`.

    Args:
        seq: a non-empty sequence.

    Returns:
        The minimal period (`len(seq)` if aperiodic).
    """
    n = len(seq)
    for k in range(1, n + 1):
        if all(seq[j] == seq[j + k] for j in range(n - k)):
            return k
    return n


def find_tract(context: str, event: int, deleted: str, inserted: str) -> RepeatTract | None:
    """
    Find the repeat tract of a pure insertion or deletion in a reference context.

    Args:
        context: the (upper-case) reference sequence.
        event: the index in `context` of the first deleted base / the insertion point.
        deleted: the deleted bases.
        inserted: the inserted bases.

    Returns:
        The tract, in `context` coordinates, or None if the event is not a pure indel.
    """
    if deleted and not inserted:
        seq = deleted
        period = min_period(seq)
        seed_start, seed_end = event, event + len(seq)
    elif inserted and not deleted:
        seq = inserted
        period = min_period(seq)
        if context[event : event + period] == seq[:period]:
            seed_start, seed_end = event, event + period
        elif context[event - period : event] == seq[-period:]:
            seed_start, seed_end = event - period, event
        else:
            return RepeatTract(start=event, end=event, period=period, unit=seq[:period])
    else:
        return None
    start, end = seed_start, seed_end
    while end < len(context) and context[end] == context[end - period]:
        end += 1
    while start - 1 >= 0 and start - 1 + period < len(context):
        if context[start - 1] != context[start - 1 + period]:
            break
        start -= 1
    unit = context[start : start + period] if end > start else seq[:period]
    return RepeatTract(start=start, end=end, period=period, unit=unit)


def tract_of_record(
    fasta: pysam.FastaFile, chrom: str, pos: int, ref: str, alts: Sequence[str]
) -> RepeatTract | None:
    """
    The repeat tract of an indel record, from its first pure insertion/deletion allele.

    Args:
        fasta: the reference.
        chrom: the contig.
        pos: the 1-based record position.
        ref: the reference allele.
        alts: the alternate alleles (`*`, `.` and symbolic alleles are skipped).

    Returns:
        The tract in genomic 0-based coordinates, or None if no allele is a pure indel.
    """
    p0 = pos - 1
    lo = max(0, p0 - CONTEXT)
    context = fasta.fetch(chrom, lo, p0 + len(ref) + CONTEXT).upper()
    for alt in alts:
        if alt in ("*", ".") or alt.startswith("<"):
            continue
        offset, deleted, inserted = trim_alleles(ref, alt)
        tract = find_tract(context, p0 - lo + offset, deleted, inserted)
        if tract is not None:
            return RepeatTract(
                start=lo + tract.start, end=lo + tract.end, period=tract.period, unit=tract.unit
            )
    return None


_CONSUMES_REFERENCE: Final[frozenset[int]] = frozenset(
    op.code for op in CigarOp if op.consumes_reference
)
"""Codes of the CIGAR ops that consume reference bases (M, D, N, =, X), for fast lookup per read."""

_CONSUMES_QUERY: Final[frozenset[int]] = frozenset(op.code for op in CigarOp if op.consumes_query)
"""Codes of the CIGAR ops that consume query bases (M, I, S, =, X)."""


def _query_position_at(
    cigar: Sequence[tuple[int, int]], reference_start: int, target: int
) -> int | None:
    """The query position aligned (M/=/X) at a reference position, or None (deleted/unaligned)."""
    ref_pos, query_pos = reference_start, 0
    for op, length in cigar:
        consumes_reference = op in _CONSUMES_REFERENCE
        consumes_query = op in _CONSUMES_QUERY
        if consumes_reference and ref_pos <= target < ref_pos + length:
            return query_pos + target - ref_pos if consumes_query else None
        ref_pos += length if consumes_reference else 0
        query_pos += length if consumes_query else 0
    return None


def _anchor_query_positions(
    cigar: Sequence[tuple[int, int]], reference_start: int, first: int, last: int
) -> tuple[int, int] | None:
    """The query positions aligned (M/=/X) at reference positions `first` and `last`, or None."""
    first_query = _query_position_at(cigar, reference_start, first)
    if first_query is None:
        return None
    last_query = _query_position_at(cigar, reference_start, last)
    if last_query is None:
        return None
    return first_query, last_query


@dataclass(frozen=True)
class TractEvidence:
    """
    Read evidence for the length of one repeat tract.

    Attributes:
        n_overlap: reads overlapping the anchored tract (capped just above `MAX_READS`).
        n_spanning: reads spanning it.
        histogram: spanning reads per observed-minus-reference tract length (bp).
        mean_mapq: the mean MAPQ of spanning reads (None without any).
        frac_mapq_lt20: the fraction of spanning reads with MAPQ < 20 (None without any).
        mean_tract_bq: the mean base quality of the tract bases in spanning reads (None without
            any).
    """

    n_overlap: int
    n_spanning: int
    histogram: dict[int, int]
    mean_mapq: float | None
    frac_mapq_lt20: float | None
    mean_tract_bq: float | None


def measure_tract(alignments: pysam.AlignmentFile, chrom: str, tract: RepeatTract) -> TractEvidence:
    """
    Measure the tract length in each read spanning it.

    Args:
        alignments: the indexed BAM/CRAM.
        chrom: the contig.
        tract: the tract.

    Returns:
        The read evidence.
    """
    first, last = tract.start - FLANK, tract.end + FLANK - 1
    histogram: Counter[int] = Counter()
    n_overlap = n_spanning = 0
    mapq_sum = mapq_lt20 = 0
    bq_sum = bq_n = 0
    for read in alignments.fetch(chrom, max(0, first), last + 1):
        if read.flag & _SKIP_FLAGS:
            continue
        n_overlap += 1
        if n_overlap > MAX_READS:
            break
        if read.reference_start > first or read.reference_end is None:
            continue
        if read.reference_end <= last or read.cigartuples is None:
            continue
        anchors = _anchor_query_positions(read.cigartuples, read.reference_start, first, last)
        if anchors is None:
            continue
        first_query, last_query = anchors
        observed = last_query - first_query + 1 - 2 * FLANK
        histogram[observed - tract.length] += 1
        n_spanning += 1
        mapq_sum += read.mapping_quality
        mapq_lt20 += read.mapping_quality < 20
        qualities = read.query_qualities
        if observed > 0 and qualities is not None:
            tract_qualities = qualities[first_query + FLANK : last_query - FLANK + 1]
            bq_sum += sum(tract_qualities)
            bq_n += len(tract_qualities)
    return TractEvidence(
        n_overlap=n_overlap,
        n_spanning=n_spanning,
        histogram=dict(sorted(histogram.items())),
        mean_mapq=mapq_sum / n_spanning if n_spanning else None,
        frac_mapq_lt20=mapq_lt20 / n_spanning if n_spanning else None,
        mean_tract_bq=bq_sum / bq_n if bq_n else None,
    )


def format_histogram(histogram: dict[int, int]) -> str:
    """
    Format a length histogram as `delta:count;...` (`.` when empty).

    Args:
        histogram: reads per length difference.

    Returns:
        The formatted histogram.
    """
    return ";".join(f"{d}:{c}" for d, c in sorted(histogram.items())) or "."


def parse_histogram(text: object) -> dict[int, int]:
    """
    Parse a histogram written by `format_histogram` (anything not a non-empty string is empty).

    Args:
        text: the formatted histogram.

    Returns:
        Reads per length difference.
    """
    if not isinstance(text, str) or text in ("", "."):
        return {}
    histogram: dict[int, int] = {}
    for item in text.split(";"):
        delta, count = item.split(":")
        histogram[int(delta)] = int(count)
    return histogram


@dataclass(frozen=True)
class StrlenSite:
    """
    The tract and read evidence of one indel record.

    Attributes:
        chrom: the contig.
        pos: the 1-based record position.
        ref: the reference allele.
        alt: the comma-joined alternate alleles.
        tract: the record's repeat tract.
        evidence: the read evidence.
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    tract: RepeatTract
    evidence: TractEvidence

    def to_row(self) -> list[str]:
        """
        The site as a strlen table row.

        Returns:
            The `STRLEN_COLUMNS` values as strings.
        """
        ev = self.evidence
        return [
            self.chrom,
            str(self.pos),
            self.ref,
            self.alt,
            str(self.tract.period),
            str(self.tract.length),
            self.tract.unit,
            str(self.tract.start + 1),
            str(ev.n_overlap),
            str(ev.n_spanning),
            format_histogram(ev.histogram),
            "nan" if ev.mean_mapq is None else f"{ev.mean_mapq:.1f}",
            "nan" if ev.frac_mapq_lt20 is None else f"{ev.frac_mapq_lt20:.3f}",
            "nan" if ev.mean_tract_bq is None else f"{ev.mean_tract_bq:.1f}",
        ]


def measure_records(
    records: Iterable[tuple[str, int, str, Sequence[str]]],
    alignments: pysam.AlignmentFile,
    fasta: pysam.FastaFile,
) -> Iterator[StrlenSite]:
    """
    Measure the tract of each record; records without a pure-indel allele are skipped.

    Args:
        records: `(chrom, pos, ref, alts)` of indel records.
        alignments: the indexed BAM/CRAM.
        fasta: the reference.

    Yields:
        One site per record with a tract.
    """
    n_skipped = 0
    for chrom, pos, ref, alts in records:
        tract = tract_of_record(fasta, chrom, pos, ref, alts)
        if tract is None:
            n_skipped += 1
            continue
        yield StrlenSite(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=",".join(alts),
            tract=tract,
            evidence=measure_tract(alignments, chrom, tract),
        )
    if n_skipped:
        logger.info(f"Skipped {n_skipped:,} indel records without a pure insertion/deletion allele")


def parse_region(region: str) -> tuple[str, int | None, int | None]:
    """
    Parse a `chrom` or `chrom:start-end` region (1-based, inclusive).

    The span is split off at the last `:`, so a contig name may contain colons when a span is
    given; a whole contig whose name contains a colon must be matched against the contig names
    first (see `indel_records`).

    Args:
        region: the region.

    Returns:
        The contig and the start and end (None for a whole contig).

    Raises:
        ValueError: if the region is malformed or its start is below 1.
    """
    chrom, sep, span = region.rpartition(":")
    if not sep:
        return region, None, None
    start, dash, end = span.replace(",", "").partition("-")
    if (
        not chrom
        or not dash
        or not start.isdigit()
        or not end.isdigit()
        or not 1 <= int(start) <= int(end)
    ):
        raise ValueError(f"Malformed region (expected chrom or chrom:start-end): {region}")
    return chrom, int(start), int(end)


def indel_records(
    vcf: pysam.VariantFile, regions: Iterable[str]
) -> Iterator[tuple[str, int, str, Sequence[str]]]:
    """
    The indel records (any FILTER) whose POS lies in each region, in region order.

    A record is selected by its POS so that adjacent shards never measure a record twice. A region
    that is a contig name of the VCF is that whole contig, even if the name contains a colon (e.g.
    `HLA-A*01:01:01:01`). The regions are checked before any record is read.

    Args:
        vcf: the indexed VCF.
        regions: `chrom` or `chrom:start-end` regions (1-based, inclusive).

    Returns:
        `(chrom, pos, ref, alts)` of each indel record (see `variant_type_of`).

    Raises:
        ValueError: if a region is malformed or its contig is not in the VCF header.
    """
    parsed = []
    for region in regions:
        chrom, start, end = (
            (region, None, None) if region in vcf.header.contigs else parse_region(region)
        )
        if chrom not in vcf.header.contigs:
            raise ValueError(f"Contig {chrom} of region {region} is not in the VCF header")
        parsed.append((chrom, start, end))
    return _indel_records(vcf, parsed)


def _indel_records(
    vcf: pysam.VariantFile, regions: Iterable[tuple[str, int | None, int | None]]
) -> Iterator[tuple[str, int, str, Sequence[str]]]:
    """The indel records in each parsed region (see `indel_records`)."""
    for chrom, start, end in regions:
        fetch_start = None if start is None else start - 1
        for rec in vcf.fetch(chrom, fetch_start, end):
            if start is not None and end is not None and not start <= rec.pos <= end:
                continue
            alts = rec.alts or ()
            if variant_type_of(rec.ref or "", alts) is VariantType.INDEL:
                yield rec.chrom, rec.pos, rec.ref or "", alts


def write_strlen_table(path: Path, sites: Iterable[StrlenSite]) -> int:
    """
    Write sites as a tab-separated strlen table with a header line.

    Args:
        path: the output path (`.gz` or `.bgz` is gzip-compressed).
        sites: the sites.

    Returns:
        The number of sites written.
    """
    n = 0
    with to_writer(path) as out:
        out.write("\t".join(STRLEN_COLUMNS) + "\n")
        for site in sites:
            out.write("\t".join(site.to_row()) + "\n")
            n += 1
    return n


def read_strlen_tables(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Read and concatenate strlen tables (e.g. one per shard), keeping one row per record.

    Args:
        paths: the tables written by `write_strlen_table`.

    Returns:
        A data frame with `STRLEN_COLUMNS`; `pos` and the counts are int64, `length_histogram` and
        the alleles strings, and the rest float64.

    Raises:
        ValueError: if a table lacks a `STRLEN_COLUMNS` header.
    """
    frames = []
    for path in paths:
        frame = pd.read_csv(
            path,
            sep="\t",
            dtype={"chrom": str, "ref": str, "alt": str, "unit": str, "length_histogram": str},
            keep_default_na=False,
            na_values={c: ["nan"] for c in ("mean_mapq", "frac_mapq_lt20", "mean_tract_bq")},
        )
        if tuple(frame.columns) != STRLEN_COLUMNS:
            raise ValueError(f"{path} is not a strlen table: columns {list(frame.columns)}")
        frames.append(frame)
    if not frames:
        raise ValueError("No strlen tables given")
    table = pd.concat(frames, ignore_index=True)
    return table.drop_duplicates(["chrom", "pos", "ref", "alt"]).reset_index(drop=True)
