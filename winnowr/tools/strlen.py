import logging
import time
from pathlib import Path

import pysam

from winnowr.common import AUTOSOMES
from winnowr.strlen import indel_records
from winnowr.strlen import measure_records
from winnowr.strlen import open_alignments
from winnowr.strlen import write_strlen_table

logger = logging.getLogger(__name__)


def strlen(
    *,
    vcf: Path,
    alignments: Path,
    reference: Path,
    output: Path,
    regions: list[str] | None = None,
) -> None:
    """
    Measure repeat-tract lengths in reads at indel calls, for the repeat-length features.

    For every indel record (any FILTER) whose POS is in the regions, finds the repeat tract of its
    first pure insertion/deletion allele in the reference and histograms, over primary,
    non-duplicate reads (any MAPQ) that span the tract with two aligned flank bases each side,
    the observed minus reference tract length. Writes one row per record (tab-separated, with a
    header). Pass the output(s) to `extract --strlen`. Run one shard per region set and give all
    shard outputs to `extract`.

    Args:
        vcf: Indexed single-sample VCF of the calls (the VCF given to `extract`).
        alignments: Indexed BAM/CRAM of the sample's reads.
        reference: Indexed reference FASTA (also used to decode CRAM).
        output: Output table (.gz is gzip-compressed).
        regions: Regions to process, as chrom or chrom:start-end (1-based, inclusive); default
            chr1-chr22.
    """
    t0 = time.monotonic()
    with (
        pysam.VariantFile(str(vcf)) as variants,
        open_alignments(alignments, reference) as reads,
        pysam.FastaFile(str(reference)) as fasta,
    ):
        try:
            records = indel_records(variants, regions or list(AUTOSOMES))
        except ValueError as error:
            raise ValueError(
                f"{error}; choose the regions with --regions (default chr1-chr22)"
            ) from error
        n = write_strlen_table(output, measure_records(records, reads, fasta))
    seconds = time.monotonic() - t0
    logger.info(
        f"Wrote {n:,} sites to {output} in {seconds:.1f} s "
        f"({1000 * seconds / max(n, 1):.2f} ms/site)"
    )
