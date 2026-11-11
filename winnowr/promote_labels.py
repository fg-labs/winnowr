"""
Labels for non-PASS calls, from the truth VCF and its confident regions.

hap.py, run with `--pass-only`, only evaluates PASS calls, so non-PASS calls are labelled by exact
matching against the truth VCF. A non-PASS call is a promotion *candidate* when it is on the
extracted contigs, of a known variant type, its REF span lies inside the truth confident regions
and does not overlap any PASS call. Calls and truth records are left-aligned and trimmed
(`normalize_alleles`) and matched on position, REF and the ALT list. A candidate is labelled:

- `PROMOTE` (1): matches a truth record with the same genotype,
- `PROMOTE_FLIP` (2): matches a truth record and switching its genotype between `0/1` and `1/1`
  gives the truth genotype,
- `STAY` (0): anything else.

A match only counts if the truth record is a hap.py FN of the same variant type (i.e. it is not
already matched by a PASS call), and only for the first candidate that matches it with a
`PROMOTE` or `PROMOTE_FLIP` label (later ones, e.g. another representation of the same variant,
are `STAY`), so a promotion that matches is exactly one FN turned into a TP.
"""

from collections.abc import Collection
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from winnowr.alleles import FLIPPED_GT
from winnowr.alleles import BaseFetcher
from winnowr.alleles import normalize_alleles
from winnowr.alleles import normalize_gt
from winnowr.common import PASS
from winnowr.common import BoolArray
from winnowr.common import VariantType
from winnowr.regions import RegionSet

TruthKey = tuple[str, int, str, str]
"""A normalized `(chrom, pos, ref, comma-joined alts)`."""

FnPosition = tuple[str, int, VariantType]
"""A hap.py truth FN: `(chrom, pos, variant type)`."""


class PromoteLabel(IntEnum):
    """The promote-model classes."""

    STAY = 0
    PROMOTE = 1
    PROMOTE_FLIP = 2


@dataclass(frozen=True)
class TruthVariant:
    """
    A truth record that a candidate can match.

    Attributes:
        pos: the record's original (un-normalized) position, as hap.py reports it.
        gt: the normalized truth genotype.
    """

    pos: int
    gt: str


def _fetcher(fasta: pysam.FastaFile | None, contig: str) -> BaseFetcher | None:
    """A base fetcher for one contig, or None without a reference."""
    if fasta is None:
        return None
    return lambda pos: fasta.fetch(contig, pos - 1, pos)


def _key(
    chrom: str, pos: int, ref: str, alts: Sequence[str], fasta: pysam.FastaFile | None
) -> TruthKey:
    """The normalized match key of a record."""
    norm_pos, alleles = normalize_alleles(pos, (ref, *alts), _fetcher(fasta, chrom))
    return chrom, norm_pos, alleles[0], ",".join(alleles[1:])


def read_truth_variants(
    truth_vcf: Path,
    fn_positions: Collection[FnPosition],
    reference: Path | None = None,
) -> dict[TruthKey, TruthVariant]:
    """
    Read the truth records at hap.py FN positions, keyed by their normalized alleles.

    Args:
        truth_vcf: the indexed truth VCF (single sample).
        fn_positions: the hap.py truth FNs; only records at these positions are read.
        reference: the reference FASTA for left-alignment (without it alleles are only trimmed).

    Returns:
        The truth records by normalized key (the first record wins on collisions).

    Raises:
        ValueError: if an FN contig is not in the truth VCF header (e.g. `1` vs `chr1` naming).
    """
    wanted = {(chrom, pos) for chrom, pos, _ in fn_positions}
    by_contig: dict[str, list[int]] = {}
    for chrom, pos in wanted:
        by_contig.setdefault(chrom, []).append(pos)
    truth: dict[TruthKey, TruthVariant] = {}
    fasta = pysam.FastaFile(str(reference)) if reference is not None else None
    try:
        with pysam.VariantFile(str(truth_vcf)) as vcf:
            for contig, positions in by_contig.items():
                if contig not in vcf.header.contigs:
                    raise ValueError(
                        f"Contig {contig} has hap.py FNs but is not in the header of truth VCF "
                        f"{truth_vcf}"
                    )
                # one indexed fetch per FN position: FNs are sparse (~30k per GIAB sample), and
                # this was ~7x faster than streaming each contig of a GIAB truth VCF
                for pos in sorted(positions):
                    for rec in vcf.fetch(contig, pos - 1, pos):
                        if rec.pos != pos or not rec.alts:
                            continue
                        key = _key(contig, rec.pos, rec.ref or "", rec.alts, fasta)
                        gt = normalize_gt(rec.samples[0].get("GT"))
                        truth.setdefault(key, TruthVariant(pos=rec.pos, gt=gt))
    finally:
        if fasta is not None:
            fasta.close()
    return truth


def promote_eligible(table: pd.DataFrame) -> BoolArray:
    """
    Flag the calls a promote model scores: non-PASS calls that do not overlap a PASS call.

    This is the label-independent part of the candidate rule, shared by training (see
    `promote_candidates`) and scoring.

    Args:
        table: a feature table (with `overlaps_pass`).

    Returns:
        A boolean mask over the table rows.
    """
    eligible: BoolArray = (
        (table["filter"] != PASS) & ~table["overlaps_pass"].astype(bool)
    ).to_numpy(dtype=np.bool_)
    return eligible


def promote_candidates(table: pd.DataFrame, confident: RegionSet) -> BoolArray:
    """
    Flag the non-PASS calls that are promotion candidates (see the module docstring).

    Args:
        table: a feature table (with `overlaps_pass`).
        confident: the truth confident regions.

    Returns:
        A boolean mask over the table rows: `promote_eligible` calls inside the confident regions.
    """
    mask = promote_eligible(table)
    in_bed = np.zeros(len(table), dtype=np.bool_)
    starts = table["pos"].to_numpy(dtype=np.int64) - 1
    ends = starts + table["ref"].str.len().to_numpy(dtype=np.int64)
    for contig, group_index in table.groupby("chrom", sort=False).indices.items():
        index = np.asarray(group_index, dtype=np.int64)
        in_bed[index] = confident.contains_span(str(contig), starts[index], ends[index])
    candidates: BoolArray = mask & in_bed
    return candidates


def attach_promote_labels(
    table: pd.DataFrame,
    truth: dict[TruthKey, TruthVariant],
    fn_positions: Collection[FnPosition],
    confident: RegionSet,
    reference: Path | None = None,
) -> pd.DataFrame:
    """
    Add a `promote_label` column (float; NaN for non-candidates).

    Args:
        table: a feature table.
        truth: truth records from `read_truth_variants()`.
        fn_positions: the hap.py truth FNs.
        confident: the truth confident regions.
        reference: the reference FASTA used for `truth` (must match).

    Returns:
        A new table with the `promote_label` column.
    """
    fns = set(fn_positions)
    used: set[TruthKey] = set()
    candidates = promote_candidates(table, confident)
    labels = np.full(len(table), np.nan)
    fasta = pysam.FastaFile(str(reference)) if reference is not None else None
    try:
        rows = table.loc[candidates]
        for index, chrom, pos, ref, alt, gt, variant_type in zip(
            np.flatnonzero(candidates),
            rows["chrom"].tolist(),
            rows["pos"].to_numpy(dtype=np.int64).tolist(),
            rows["ref"].tolist(),
            rows["alt"].tolist(),
            rows["gt"].tolist(),
            rows["variant_type"].tolist(),
            strict=True,
        ):
            key = _key(chrom, pos, ref, alt.split(","), fasta)
            hit = truth.get(key)
            label = PromoteLabel.STAY
            if hit is not None and key not in used:
                if (chrom, hit.pos, VariantType(variant_type)) in fns:
                    if hit.gt == gt:
                        label = PromoteLabel.PROMOTE
                    elif FLIPPED_GT.get(gt) == hit.gt:
                        label = PromoteLabel.PROMOTE_FLIP
            if label is not PromoteLabel.STAY:
                used.add(key)
            labels[index] = label
    finally:
        if fasta is not None:
            fasta.close()
    return table.assign(promote_label=labels)


def promote_calls(table: pd.DataFrame, variant_type: VariantType) -> pd.DataFrame:
    """
    Select the labelled promotion candidates of one variant type.

    Args:
        table: a promote-labelled feature table.
        variant_type: the variant type.

    Returns:
        The selected rows (a copy, with a fresh index).
    """
    selected = table[(table["variant_type"] == variant_type.value) & table["promote_label"].notna()]
    return selected.reset_index(drop=True)
