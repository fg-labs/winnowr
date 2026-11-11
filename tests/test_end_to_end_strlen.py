"""Extract -> train -> evaluate -> score an indel model with repeat-length features."""

from pathlib import Path

import pysam
import pytest
from pytest_mock import MockerFixture

import winnowr.tools.train
from tests.util import add_call
from tests.util import add_gvcf_site
from tests.util import add_happy_record
from tests.util import gvcf_builder
from tests.util import happy_builder
from tests.util import vcf_builder
from tests.util import write
from winnowr.common import VariantType
from winnowr.evaluation import PolicyMetric
from winnowr.scoring import ORIGINAL_GT_FORMAT
from winnowr.scoring import REGENOTYPE_INFO
from winnowr.strlen import STRLEN_COLUMNS
from winnowr.tools.evaluate import evaluate
from winnowr.tools.extract import extract
from winnowr.tools.score import score
from winnowr.tools.train import train

N_CALLS = 400
TRACT_LEN = 5

# per kind: called GT, AD, hap.py (query BD, BK), truth GT and truth ALT, and the read lengths
KINDS = {
    "tp": ((0, 1), (10, 10), ("TP", "gm"), (0, 1), "C", "-1:10;0:10"),
    "absent": ((0, 1), (12, 8), ("FP", "."), None, "C", "0:20"),
    "hom": ((0, 1), (1, 19), ("FP", "am"), (1, 1), "C", "-1:20"),
    "multi": ((0, 1), (10, 10), ("FP", "am"), (1, 2), "C,CAA", "-1:10;1:10"),
}


def _kind(i: int) -> str:
    return ("absent", "hom", "multi")[i % 10] if i % 10 < 3 else "tp"


def _sample(tmp_path: Path, name: str, offset: int) -> dict[str, Path]:
    """Het deletions `CA>C` in A5 tracts every 3 kb; see `KINDS` and `_kind`."""
    vcf, gvcf, happy = vcf_builder(), gvcf_builder(), happy_builder()
    rows = ["\t".join(STRLEN_COLUMNS)]
    for i in range(N_CALLS):
        pos = 1000 + 3000 * i + offset
        gt, ad, (bd, bk), truth_gt, truth_alt, histogram = KINDS[_kind(i)]
        add_call(vcf, pos, gt=gt, ref="CA", alt="C", ad=ad)
        add_gvcf_site(gvcf, pos, ref="CA", alt="C", raw_mq=3600 * 20, raw_dp=20)
        truth = (truth_gt, "FN" if bd == "FP" else "TP", bk, "INDEL")
        add_happy_record(
            happy,
            pos,
            ref="CA",
            alt=truth_alt,
            query=(gt, bd, bk, "INDEL"),
            truth=truth if truth_gt is not None else (None, ".", ".", "NOCALL"),
        )
        n_spanning = sum(int(item.split(":")[1]) for item in histogram.split(";"))
        rows.append(
            f"chr1\t{pos}\tCA\tC\t1\t{TRACT_LEN}\tA\t{pos + 1}\t{n_spanning}\t{n_spanning}\t"
            f"{histogram}\t60.0\t0.000\t35.0"
        )
    strlen = tmp_path / f"{name}.strlen.tsv"
    strlen.write_text("\n".join(rows) + "\n")
    return {
        "vcf": write(vcf, tmp_path / f"{name}.vcf.gz"),
        "gvcf": write(gvcf, tmp_path / f"{name}.g.vcf.gz"),
        "happy": write(happy, tmp_path / f"{name}.happy.vcf.gz"),
        "strlen": strlen,
    }


def test_regenotype_model(tmp_path: Path, mocker: MockerFixture) -> None:
    tables, inputs = [], []
    for i, name in enumerate(["a", "b", "c"]):
        paths = _sample(tmp_path, name, offset=i)
        table = tmp_path / f"{name}.parquet"
        extract(
            vcf=paths["vcf"],
            gvcf=paths["gvcf"],
            happy=paths["happy"],
            output=table,
            contigs=["chr1"],
            sample=name,
            strlen=[paths["strlen"]],
        )
        tables.append(table)
        inputs.append(paths)
    prefix = tmp_path / "indel"
    fit = mocker.spy(winnowr.tools.train, "fit_stutter_from_feature_tables")
    train(
        tables=tables[:2],
        variant_type=VariantType.INDEL,
        output=prefix,
        regenotype=True,
        max_rounds=30,
        min_child_weight=0.5,
        validation_fraction=0.0,
        threads=1,
    )
    sidecar = Path(f"{prefix}.winnowr.json")
    # the final model's stutter model is fitted on both samples, each CV fold's on the other only
    fitted_on = [[info.sample for _, info in call.args[0]] for call in fit.call_args_list]
    assert fitted_on == [["a", "b"], ["b"], ["a"]]

    metrics_path = tmp_path / "c.tsv"
    evaluate(models=[sidecar], table=tables[2], output=metrics_path)
    metrics = {m.policy: m for m in PolicyMetric.read(metrics_path)}
    genotyper = metrics["genotyper"]
    n_multi = sum(_kind(i) == "multi" for i in range(N_CALLS))
    assert genotyper.regenotyped == n_multi and genotyper.regenotypes_fixed == n_multi
    assert genotyper.regenotypes_broke_tp == 0 and genotyper.flips_fixed == n_multi
    assert genotyper.fp == 0
    assert metrics["genotyper_no_regenotype"].fp == n_multi

    out = tmp_path / "c.scored.vcf"
    paths = inputs[2]
    score(
        vcf=paths["vcf"], gvcf=paths["gvcf"], models=[sidecar], output=out, strlen=[paths["strlen"]]
    )
    with pysam.VariantFile(str(out)) as scored:
        regenotyped = [r for r in scored if REGENOTYPE_INFO in r.info]
    assert len(regenotyped) == n_multi
    for rec in regenotyped:
        # the +1 allele is added and the REF extended over the A5 tract
        assert rec.alleles == ("CAAAAA", "CAAAA", "CAAAAAA")
        assert (rec.samples[0]["GT"], rec.samples[0][ORIGINAL_GT_FORMAT]) == ((1, 2), "0/1")

    with pytest.raises(ValueError, match="pass --strlen"):
        score(vcf=paths["vcf"], gvcf=paths["gvcf"], models=[sidecar], output=out)
    with pytest.raises(ValueError, match="need --variant-type INDEL and --task GENOTYPE, got"):
        train(tables=tables[:2], variant_type=VariantType.SNP, output=prefix, repeat_length=True)
