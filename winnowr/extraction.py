"""Build a (optionally labelled) feature table for one sample from its VCF, gVCF and BEDs."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pysam

from winnowr.features import add_derived_features
from winnowr.features import extract_raw_records
from winnowr.labels import attach_labels
from winnowr.labels import check_label_join
from winnowr.labels import read_happy_labels
from winnowr.promote_labels import attach_promote_labels
from winnowr.promote_labels import read_truth_variants
from winnowr.regions import RegionSet
from winnowr.regions import parse_named_bed
from winnowr.repeat_features import add_strlen_features
from winnowr.table import FeatureTableInfo


def read_region_sets(specs: Sequence[str]) -> list[RegionSet]:
    """
    Read region sets from `name=path` specifications.

    Args:
        specs: the specifications, e.g. `segdups=segdups.bed.gz`.

    Returns:
        One region set per specification, in order.

    Raises:
        ValueError: if a specification is malformed or a name is repeated.
    """
    named = [parse_named_bed(spec) for spec in specs]
    names = [name for name, _ in named]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Region names (--regions name=path) must be unique; repeated: {repeated}")
    return [RegionSet.from_bed(name, path) for name, path in named]


def vcf_sample_name(vcf: Path) -> str:
    """
    The name of a VCF's first sample (the file name if it has none).

    Args:
        vcf: the VCF.

    Returns:
        The sample name.
    """
    with pysam.VariantFile(str(vcf)) as handle:
        return str(next(iter(handle.header.samples), vcf.name))


def vcf_contig_lengths(vcf: Path) -> dict[str, int | None]:
    """
    The contig lengths declared in a VCF header.

    Args:
        vcf: the VCF.

    Returns:
        The length of each header contig (None where the header gives none).
    """
    with pysam.VariantFile(str(vcf)) as handle:
        return {str(name): contig.length for name, contig in handle.header.contigs.items()}


def check_truth_inputs(
    happy: Path | None, truth_vcf: Path | None, truth_bed: Path | None, reference: Path | None
) -> None:
    """
    Check that the promote-label inputs are given together.

    Args:
        happy: the hap.py output VCF, if any.
        truth_vcf: the truth VCF, if any.
        truth_bed: the truth confident regions, if any.
        reference: the reference FASTA, if any.

    Raises:
        ValueError: if `truth_vcf`, `truth_bed` or `reference` is given without all of `truth_vcf`,
            `truth_bed` and `happy`.
    """
    if any(i is not None for i in (truth_vcf, truth_bed, reference)) and (
        truth_vcf is None or truth_bed is None or happy is None
    ):
        raise ValueError(
            "Promote labels need --truth-vcf, --truth-bed and --happy together "
            "(--reference is optional and only used with them)"
        )


def build_feature_table(
    sample: str,
    vcf: Path,
    gvcf: Path | None,
    regions: Sequence[RegionSet],
    contigs: Sequence[str],
    happy: Path | None = None,
    truth_vcf: Path | None = None,
    truth_bed: Path | None = None,
    reference: Path | None = None,
    strlen: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FeatureTableInfo]:
    """
    Extract features for every call and, given a hap.py VCF, attach labels.

    Args:
        sample: the sample name to record.
        vcf: the indexed single-sample VCF.
        gvcf: the indexed gVCF the VCF was genotyped from, or None; read only when the VCF lacks
            the raw annotations (see `extract_raw_records`).
        regions: the region sets to flag.
        contigs: the contigs to extract.
        happy: optional hap.py output VCF (vcfeval engine) for labels.
        truth_vcf: optional truth VCF; with `truth_bed` and `happy`, non-PASS calls get promote
            labels (see `winnowr.promote_labels`).
        truth_bed: the truth confident regions.
        reference: the reference FASTA used to left-align indels for truth matching.
        strlen: optional repeat-tract read evidence (see `winnowr.strlen.read_strlen_tables`) to
            join as repeat-length features.

    Returns:
        The feature table and its metadata.

    Raises:
        ValueError: if the promote-label inputs are incomplete (see `check_truth_inputs`; checked
            before any input is read), a region BED has no intervals on the contigs, neither the
            VCF nor the gVCF has the raw annotations, or most hap.py labels match no call.
    """
    check_truth_inputs(happy, truth_vcf, truth_bed, reference)
    for region_set in regions:
        region_set.check_contigs(contigs)
    raw = extract_raw_records(vcf, gvcf, contigs)
    lengths = vcf_contig_lengths(vcf)
    labels = None if happy is None else read_happy_labels(happy, contigs)
    frame = add_derived_features(raw, regions)
    if strlen is not None:
        frame = add_strlen_features(frame, strlen)
    fn_counts: dict[str, int] | None = None
    if labels is not None:
        frame, join = attach_labels(frame, labels)
        check_label_join(join, str(happy))
        fn_counts = {variant_type.value: n for variant_type, n in labels.fn_counts.items()}
        if truth_vcf is not None and truth_bed is not None:
            truth = read_truth_variants(truth_vcf, labels.fn_positions, reference)
            confident = RegionSet.from_bed("confident", truth_bed)
            confident.check_contigs(contigs)
            frame = attach_promote_labels(frame, truth, labels.fn_positions, confident, reference)
    info = FeatureTableInfo(
        sample=sample,
        contigs=tuple(contigs),
        region_names=tuple(r.name for r in regions),
        fn_counts=fn_counts,
        promote_labelled=truth_vcf is not None,
        strlen=strlen is not None,
        reference_contigs={name: lengths.get(name) for name in contigs},
    )
    return frame, info
