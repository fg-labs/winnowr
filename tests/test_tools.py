"""Argument checks of the CLI tools that fail before any heavy work."""

from pathlib import Path

import pandas as pd
import pysam
import pytest

from tests.test_model import FAST
from tests.test_model import synthetic_calls
from tests.util import gvcf_builder
from tests.util import write
from winnowr.common import NEVER
from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.extraction import read_region_sets
from winnowr.features import extract_raw_records
from winnowr.model import ModelInfo
from winnowr.model import WinnowrModel
from winnowr.model import fit_booster
from winnowr.model import num_classes
from winnowr.rescore import Mode
from winnowr.table import FeatureTableInfo
from winnowr.table import write_feature_table
from winnowr.thresholds import PromoteThresholds
from winnowr.thresholds import Thresholds
from winnowr.tools.evaluate import evaluate
from winnowr.tools.extract import extract
from winnowr.tools.score import score
from winnowr.tools.train import _reference_contigs
from winnowr.tools.train import train


def _sidecar(
    path: Path,
    variant_type: VariantType = VariantType.SNP,
    task: Task = Task.GENOTYPE,
    region_names: tuple[str, ...] = (),
    contigs: tuple[str, ...] = ("chr1",),
) -> Path:
    calls = synthetic_calls(200, seed=1)
    booster = fit_booster(calls, calls["label"], ["x", "y"], FAST, num_classes(Task.GENOTYPE))
    genotype = task is Task.GENOTYPE
    info = ModelInfo(
        variant_type=variant_type,
        features=("x", "y"),
        region_names=region_names,
        thresholds=(
            {Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0), Mode.VETO: Thresholds(0.5, NEVER, 0.0)}
            if genotype
            else {}
        ),
        training_samples=("a",),
        contigs=contigs,
        task=task,
        promote_thresholds=None if genotype else PromoteThresholds(0.5, 0.5, 0.0),
    )
    return WinnowrModel(booster, info).save(path)


def test_evaluate_model_checks(tmp_path: Path) -> None:
    table, output = tmp_path / "t.parquet", tmp_path / "out.tsv"
    snp = _sidecar(tmp_path / "snp")
    with pytest.raises(ValueError, match="--models needs at least one"):
        evaluate(models=[], table=table, output=output)
    with pytest.raises(
        ValueError, match=r"at most one genotype.*snp.winnowr.json \(genotype snp\)"
    ):
        evaluate(models=[snp, snp], table=table, output=output)
    indel = _sidecar(tmp_path / "indel-promote", VariantType.INDEL, Task.PROMOTE)
    with pytest.raises(ValueError, match=r"same variant type.*\(promote indel\)"):
        evaluate(models=[snp, indel], table=table, output=output)


def test_score_model_checks(tmp_path: Path) -> None:
    vcf, gvcf, output = tmp_path / "v.vcf.gz", tmp_path / "g.vcf.gz", tmp_path / "out.vcf.gz"
    with pytest.raises(ValueError, match="--models needs at least one"):
        score(vcf=vcf, gvcf=gvcf, models=[], output=output)
    seg = _sidecar(tmp_path / "seg", region_names=("seg",))
    with pytest.raises(ValueError, match=r"seg.winnowr.json \(genotype snp\) needs region BEDs"):
        score(vcf=vcf, gvcf=gvcf, models=[seg], output=output)
    chr1 = _sidecar(tmp_path / "chr1")
    chr2 = _sidecar(tmp_path / "chr2", VariantType.INDEL, contigs=("chr2",))
    with pytest.raises(ValueError, match=r"different contigs: .*chr1.winnowr.json: chr1"):
        score(vcf=vcf, gvcf=gvcf, models=[chr1, chr2], output=output)


def test_train_table_checks(tmp_path: Path) -> None:
    table = tmp_path / "a.parquet"
    with pytest.raises(ValueError, match="at least two feature tables"):
        train(tables=[table], variant_type=VariantType.SNP, output=tmp_path / "m")
    with pytest.raises(ValueError, match="same file more than once"):
        train(
            tables=[table, tmp_path / "." / "a.parquet"],
            variant_type=VariantType.SNP,
            output=tmp_path / "m",
        )


@pytest.mark.parametrize("given", ["truth_bed", "reference"])
def test_extract_truth_inputs_need_truth_vcf(tmp_path: Path, given: str) -> None:
    extra = tmp_path / "x"
    with pytest.raises(ValueError, match="--truth-vcf, --truth-bed and --happy together"):
        extract(
            vcf=tmp_path / "missing.vcf.gz",
            gvcf=tmp_path / "missing.g.vcf.gz",
            output=tmp_path / "t.parquet",
            happy=tmp_path / "missing.happy.vcf.gz",
            truth_bed=extra if given == "truth_bed" else None,
            reference=extra if given == "reference" else None,
        )


def _table(
    path: Path,
    sample: str,
    regions: tuple[str, ...],
    contigs: tuple[str, ...],
    reference_contigs: dict[str, int | None] | None = None,
) -> Path:
    info = FeatureTableInfo(
        sample, contigs, regions, {"snp": 0, "indel": 0}, reference_contigs=reference_contigs
    )
    write_feature_table(path, pd.DataFrame({"pos": [1]}), info)
    return path


def test_train_tables_must_share_regions_and_contigs(tmp_path: Path) -> None:
    a = _table(tmp_path / "a.parquet", "a", ("seg",), ("chr1",))
    other_regions = _table(tmp_path / "b.parquet", "b", ("lowmap",), ("chr1",))
    other_contigs = _table(tmp_path / "c.parquet", "c", ("seg",), ("chr2",))
    with pytest.raises(ValueError, match="extracted with different regions"):
        train(tables=[a, other_regions], variant_type=VariantType.SNP, output=tmp_path / "m")
    with pytest.raises(ValueError, match="extracted from different contigs"):
        train(tables=[a, other_contigs], variant_type=VariantType.SNP, output=tmp_path / "m")


def test_region_names_must_be_unique(tmp_path: Path) -> None:
    specs = [f"seg={tmp_path / 'x.bed'}", f"low={tmp_path / 'y.bed'}", f"seg={tmp_path / 'z.bed'}"]
    with pytest.raises(
        ValueError, match=r"--regions name=path\) must be unique; repeated: \['seg'\]$"
    ):
        read_region_sets(specs)


def test_vcf_without_genotypes_is_an_error(tmp_path: Path) -> None:
    text = tmp_path / "nogt.vcf"
    text.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
    )
    vcf = Path(pysam.tabix_index(str(text), preset="vcf", force=True))
    gvcf = write(gvcf_builder(), tmp_path / "g.vcf.gz")
    with pytest.raises(ValueError, match="has no FORMAT/GT"):
        extract_raw_records(vcf, gvcf, ["chr1"])


def test_train_tables_must_be_of_distinct_samples(tmp_path: Path) -> None:
    a = _table(tmp_path / "a.parquet", "a", (), ("chr1",))
    again = _table(tmp_path / "a2.parquet", "a", (), ("chr1",))
    with pytest.raises(ValueError, match=r"given once; repeated: \['a'\]"):
        train(tables=[a, again], variant_type=VariantType.SNP, output=tmp_path / "m")


def test_train_tables_must_share_a_reference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    a = _table(tmp_path / "a.parquet", "a", (), ("chr1",), {"chr1": 1000})
    b = _table(tmp_path / "b.parquet", "b", (), ("chr1",), {"chr1": 999})
    with pytest.raises(ValueError, match="--tables come from different references"):
        train(tables=[a, b], variant_type=VariantType.SNP, output=tmp_path / "m")
    # a table without reference contigs (an older winnowr): the model records none, with a warning
    infos = [
        FeatureTableInfo("a", ("chr1",), (), reference_contigs={"chr1": 1000}),
        FeatureTableInfo("old", ("chr1",), ()),
    ]
    assert _reference_contigs(infos) is None
    assert "do not record their reference contigs" in caplog.text
    assert _reference_contigs(infos[:1] * 2) == {"chr1": 1000}
