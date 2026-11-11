import logging
from pathlib import Path

from winnowr.extraction import build_feature_table
from winnowr.extraction import read_region_sets
from winnowr.extraction import vcf_contig_lengths
from winnowr.extraction import vcf_sample_name
from winnowr.model import WinnowrModel
from winnowr.model import check_reference
from winnowr.regions import parse_named_bed
from winnowr.rescore import Mode
from winnowr.scoring import decide_calls
from winnowr.scoring import write_scored_vcf
from winnowr.strlen import read_strlen_tables

logger = logging.getLogger(__name__)


def score(
    *,
    vcf: Path,
    models: list[Path],
    output: Path,
    gvcf: Path | None = None,
    regions: list[str] | None = None,
    mode: Mode = Mode.GENOTYPER,
    strlen: list[Path] | None = None,
) -> None:
    """
    Score a sample's calls and write a VCF with drops filtered and genotypes flipped.

    Genotype models score PASS calls and promote models score non-PASS calls (not overlapping a
    PASS call) of their variant type on the models' training contigs. Dropped calls get FILTER
    WinnowrDrop; promoted calls get FILTER PASS (original in INFO WINNOWR_OFILTER); flipped calls
    switch GT between 0/1 and 1/1 (original in FORMAT WINNOWR_OGT, GQ set from the model, PL
    unchanged); re-genotyped calls (models trained with --regenotype) get the repeat-length
    caller's genotype, adding an ALT allele when needed (INFO WINNOWR_REGENOTYPE). INFO WINNOWR_P /
    WINNOWR_PROMOTE_P / WINNOWR_REGENOTYPE_P hold the model probabilities.

    Args:
        vcf: Indexed single-sample VCF from HaplotypeCaller (--dragen-mode, gVCF mode).
        models: Model sidecars (<prefix>.winnowr.json), at most one genotype and one promote model
            per variant type.
        output: Output VCF (.vcf.gz is bgzipped and indexed, .bcf is indexed BCF).
        gvcf: Indexed gVCF the VCF was genotyped from; needed only when the VCF lacks the raw
            annotations (RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ), and ignored when it has them.
        regions: Region BEDs as name=path, with the names the models were trained with.
        mode: GENOTYPER (drop, flip and re-genotype) or VETO (drop only).
        strlen: Repeat-tract read-length tables from `winnowr strlen` (e.g. one per shard); needed
            by models trained with --repeat-length or --regenotype.
    """
    if not models:
        raise ValueError("--models needs at least one model sidecar")
    loaded = [WinnowrModel.load(path) for path in models]
    supplied = {name for name, _ in map(parse_named_bed, regions or [])}
    for path, model in zip(models, loaded, strict=True):
        missing = set(model.info.region_names) - supplied
        if missing:
            raise ValueError(
                f"Model {path} ({model.info.task.value} {model.info.variant_type}) needs region "
                f"BEDs (--regions name=path) for: {sorted(missing)}"
            )
    unused = supplied - {name for model in loaded for name in model.info.region_names}
    if unused:
        logger.warning(f"No model uses the --regions {sorted(unused)}; they are ignored")
    needs_strlen = [str(p) for p, m in zip(models, loaded, strict=True) if m.info.stutter]
    if needs_strlen and not strlen:
        raise ValueError(f"Models use repeat-length features; pass --strlen: {needs_strlen}")
    if strlen and not needs_strlen:
        logger.warning("No model uses repeat-length features; --strlen is ignored")
        strlen = None
    contigs = {model.info.contigs for model in loaded}
    if len(contigs) != 1:
        described = "; ".join(
            f"{path}: {','.join(m.info.contigs)}" for path, m in zip(models, loaded, strict=True)
        )
        raise ValueError(f"--models were trained on different contigs: {described}")
    lengths = vcf_contig_lengths(vcf)
    for path, model in zip(models, loaded, strict=True):
        check_reference(path, model.info, lengths, f"VCF {vcf}")
    region_sets = read_region_sets(regions or [])
    sample = vcf_sample_name(vcf)
    strlen_table = read_strlen_tables(strlen) if strlen else None
    table, _ = build_feature_table(
        sample, vcf, gvcf, region_sets, next(iter(contigs)), strlen=strlen_table
    )
    decisions = decide_calls(table, loaded, mode)
    write_scored_vcf(vcf, output, decisions)
