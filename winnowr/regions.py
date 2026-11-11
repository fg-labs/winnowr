"""Membership lookups against named BED region sets (e.g. GIAB stratifications)."""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from fgpyo.io import to_reader

from winnowr.common import BoolArray
from winnowr.common import IntArray


@dataclass(frozen=True)
class RegionSet:
    """
    A named set of merged, sorted, half-open (0-based) genomic intervals.

    Attributes:
        name: the name of the region set (used to name the derived feature).
        intervals: per-contig `(starts, ends)` arrays of merged, sorted, non-overlapping intervals.
        path: the BED file the set was read from, if any (for error messages).
    """

    name: str
    intervals: dict[str, tuple[IntArray, IntArray]]
    path: Path | None = None

    @classmethod
    def from_bed(cls, name: str, path: Path) -> "RegionSet":
        """
        Read a BED file (gzipped or bgzipped if it ends in `.gz` or `.bgz`) into a region set.

        Header (`#`, and lines whose first field is `track` or `browser`) and blank lines are
        skipped, as are empty (zero-length, `start == end`) intervals, which contain no base;
        overlapping and adjacent intervals are merged.

        Args:
            name: the name of the region set.
            path: path to the BED file.

        Returns:
            The region set.

        Raises:
            ValueError: if a line does not have at least three columns, has non-integer
                coordinates, or has `end < start` or `start < 0`.
        """
        raw: dict[str, list[tuple[int, int]]] = {}
        with to_reader(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith("#") or _is_track_or_browser(line):
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 3:
                    raise ValueError(f"{path}:{line_number}: expected >= 3 columns, got: {line!r}")
                try:
                    start, end = int(fields[1]), int(fields[2])
                except ValueError:
                    raise ValueError(
                        f"{path}:{line_number}: non-integer coordinates: {line!r}"
                    ) from None
                if start < 0 or end < start:
                    raise ValueError(f"{path}:{line_number}: invalid interval: {line!r}")
                if end == start:
                    continue
                raw.setdefault(fields[0], []).append((start, end))
        spans = {
            contig: (np.array([s for s, _ in ivs], dtype=np.int64), np.array([e for _, e in ivs]))
            for contig, ivs in raw.items()
        }
        return cls.from_spans(name, spans, path)

    @classmethod
    def from_spans(
        cls,
        name: str,
        spans: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
        path: Path | None = None,
    ) -> "RegionSet":
        """
        Build a region set from per-contig half-open spans, merging overlapping and adjacent ones.

        Args:
            name: the name of the region set.
            spans: per contig, the 0-based starts and exclusive ends of the spans (in any order);
                empty spans (`end <= start`) are ignored.
            path: the file the spans came from, if any.

        Returns:
            The region set (contigs without a non-empty span are left out).
        """
        intervals = {}
        for contig, (starts, ends) in spans.items():
            start0 = np.asarray(starts, dtype=np.int64)
            end0 = np.asarray(ends, dtype=np.int64)
            keep = end0 > start0
            if keep.any():
                intervals[contig] = _merge(start0[keep], end0[keep])
        return cls(name=name, intervals=intervals, path=path)

    def check_contigs(self, contigs: Sequence[str]) -> None:
        """
        Check that the region set has intervals on at least one of the contigs.

        A set may lack some contigs, but one sharing none of them usually means the BED names its
        contigs differently (e.g. `1` vs `chr1`), which would silently flag nothing.

        Args:
            contigs: the contigs being extracted.

        Raises:
            ValueError: if none of the contigs has an interval in the set.
        """
        if not any(contig in self.intervals for contig in contigs):
            bed_contigs = ", ".join(sorted(self.intervals)[:5]) or "none"
            raise ValueError(
                f"Region BED {self.path or self.name} has no intervals on the extracted contigs "
                f"({', '.join(contigs)}); its contigs include: {bed_contigs}"
            )

    def overlaps_span(self, contig: str, starts: npt.ArrayLike, ends: npt.ArrayLike) -> BoolArray:
        """
        Test which half-open, 0-based spans overlap the region set by at least one base.

        Args:
            contig: the contig name.
            starts: 0-based span starts.
            ends: 0-based exclusive span ends (greater than the starts).

        Returns:
            A boolean array, true where `[start, end)` shares a base with an interval.
        """
        start0 = np.asarray(starts, dtype=np.int64)
        end0 = np.asarray(ends, dtype=np.int64)
        if contig not in self.intervals:
            return np.zeros(start0.shape, dtype=np.bool_)
        interval_starts, interval_ends = self.intervals[contig]
        # the last interval starting before each span ends; merged intervals have sorted ends,
        # so it is also the one reaching furthest right among those
        idx = np.searchsorted(interval_starts, end0, side="left") - 1
        overlaps: BoolArray = (idx >= 0) & (interval_ends[np.maximum(idx, 0)] > start0)
        return overlaps

    def contains_span(self, contig: str, starts: npt.ArrayLike, ends: npt.ArrayLike) -> BoolArray:
        """
        Test which half-open, 0-based spans lie entirely inside one interval.

        Args:
            contig: the contig name.
            starts: 0-based span starts.
            ends: 0-based exclusive span ends.

        Returns:
            A boolean array, true where `[start, end)` is inside a (merged) interval.
        """
        start0 = np.asarray(starts, dtype=np.int64)
        end0 = np.asarray(ends, dtype=np.int64)
        if contig not in self.intervals:
            return np.zeros(start0.shape, dtype=np.bool_)
        interval_starts, interval_ends = self.intervals[contig]
        idx = np.searchsorted(interval_starts, start0, side="right") - 1
        inside: BoolArray = (idx >= 0) & (end0 <= interval_ends[np.maximum(idx, 0)])
        return inside


def parse_named_bed(spec: str) -> tuple[str, Path]:
    """
    Parse a `name=path` region BED specification.

    Args:
        spec: the specification, e.g. `segdups=/refs/segdups.bed.gz`.

    Returns:
        The name and path.

    Raises:
        ValueError: if the specification is not of the form `name=path` with a non-empty name
            made of letters, digits and underscores.
    """
    name, sep, path = spec.partition("=")
    if not sep or not path or not name or not name.replace("_", "").isalnum():
        raise ValueError(
            f"Region BEDs must be given as name=path (name: [A-Za-z0-9_]), got: {spec}"
        )
    return name, Path(path)


def _is_track_or_browser(line: str) -> bool:
    """True for a BED `track` or `browser` header line (not a contig named e.g. `track1`)."""
    return line.split(maxsplit=1)[0] in ("track", "browser")


def _merge(starts: IntArray, ends: IntArray) -> tuple[IntArray, IntArray]:
    """Sort and merge overlapping or adjacent half-open intervals (at least one)."""
    order = np.argsort(starts, kind="stable")
    sorted_starts, sorted_ends = starts[order], ends[order]
    # an interval starts a new merged interval when it starts after every earlier one ends
    reach = np.maximum.accumulate(sorted_ends)
    first = np.concatenate([[True], sorted_starts[1:] > reach[:-1]])
    group_starts = np.flatnonzero(first)
    merged_ends: IntArray = np.maximum.reduceat(sorted_ends, group_starts)
    return sorted_starts[first], merged_ends
