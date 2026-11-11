import logging
from pathlib import Path

from winnowr.common import AUTOSOMES
from winnowr.extraction import build_feature_table
from winnowr.extraction import check_truth_inputs
from winnowr.extraction import read_region_sets
from winnowr.extraction import vcf_sample_name
from winnowr.strlen import read_strlen_tables
from winnowr.table import write_feature_table

logger = logging.getLogger(__name__)


def extract(
    *,
    vcf: Path,
    output: Path,
    gvcf: Path | None = None,
    happy: Path | None = None,
    regions: list[str] | None = None,
    contigs: list[str] | None = None,
    sample: str | None = None,
    truth_vcf: Path | None = None,
    truth_bed: Path | None = None,
    reference: Path | None = None,
    strlen: list[Path] | None = None,
) -> None:
    """
    Extract per-call features (and optionally hap.py labels) to a Parquet feature table.

    Args:
        vcf: Indexed single-sample VCF from HaplotypeCaller (--dragen-mode, gVCF mode).
        output: Output feature table (.parquet).
        gvcf: Indexed gVCF the VCF was genotyped from, which supplies the raw annotations
            (RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ) when the VCF lacks them; ignored when the
            VCF has them.
        happy: hap.py output VCF for this sample (run with --engine vcfeval --pass-only);
            required for training and evaluation tables.
        regions: Region BEDs as name=path (e.g. segdups=segdups.bed.gz); must match the names a
            model was trained with.
        contigs: Contigs to extract (default chr1-chr22).
        sample: Sample name to record (default: the VCF's sample).
        truth_vcf: Indexed truth VCF (e.g. GIAB benchmark); with --truth-bed and --happy, labels
            non-PASS calls for training promote models.
        truth_bed: Truth confident-regions BED.
        reference: Reference FASTA (indexed) used to left-align indels when matching non-PASS
            calls to the truth; without it alleles are only trimmed.
        strlen: Repeat-tract read-length tables from `winnowr strlen` (e.g. one per shard) to
            join as repeat-length features of indel calls; needed to train or apply models that
            use them.
    """
    check_truth_inputs(happy, truth_vcf, truth_bed, reference)
    contig_list = contigs or list(AUTOSOMES)
    region_sets = read_region_sets(regions or [])
    if sample is None:
        sample = vcf_sample_name(vcf)
    strlen_table = None if not strlen else read_strlen_tables(strlen)
    frame, info = build_feature_table(
        sample,
        vcf,
        gvcf,
        region_sets,
        contig_list,
        happy,
        truth_vcf,
        truth_bed,
        reference,
        strlen_table,
    )
    write_feature_table(output, frame, info)
    logger.info(f"Wrote {len(frame):,} calls for {sample} to {output}")
