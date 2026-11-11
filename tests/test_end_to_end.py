"""Train -> evaluate -> score on a tiny synthetic dataset, through the CLI tool functions."""

from pathlib import Path
from typing import Any

import pysam
import pytest
from fgpyo.vcf.builder import VariantBuilder
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

import winnowr.training
from tests.util import SAMPLE
from tests.util import add_call
from tests.util import add_gvcf_site
from tests.util import add_happy_record
from tests.util import gvcf_builder
from tests.util import happy_builder
from tests.util import raw_annotation_info
from tests.util import vcf_builder
from tests.util import write
from winnowr import main
from winnowr.common import NEVER
from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.evaluation import PolicyMetric
from winnowr.extraction import build_feature_table
from winnowr.labels import Label
from winnowr.model import WinnowrModel
from winnowr.regions import RegionSet
from winnowr.scoring import DROP_FILTER
from winnowr.scoring import FLIP_INFO
from winnowr.scoring import ORIGINAL_FILTER_INFO
from winnowr.scoring import ORIGINAL_GT_FORMAT
from winnowr.scoring import PROBABILITIES_INFO
from winnowr.scoring import PROMOTE_PROBABILITIES_INFO
from winnowr.scoring import flip_gq
from winnowr.table import read_feature_table
from winnowr.tools.evaluate import evaluate
from winnowr.tools.extract import extract
from winnowr.tools.score import score
from winnowr.tools.train import train

N_CALLS = 300


def _sample(tmp_path: Path, name: str, offset: int) -> dict[str, Path]:
    """
    PASS calls every 3 kb on chr1, plus non-PASS calls between them.

    PASS: every 10th is a low-MQ 0/0 FP, every 10th+1 a het that is truly 1/1, the rest TPs.
    Non-PASS (missed by the caller's filter, so truth FNs): every 3rd is a real het, every 3rd+1
    a low-MQ artefact absent from the truth, every 3rd+2 a het that is truly 1/1.
    """
    vcf, gvcf, happy, truth = vcf_builder(), gvcf_builder(), happy_builder(), VariantBuilder(["T"])
    # the same calls, keeping the gVCF's raw annotations as CALL's gVCF-mode VCF does
    annotated = vcf_builder(raw_annotations=True)

    def call(pos: int, raw_mq: int, raw_dp: int, **kwargs: Any) -> None:
        add_call(vcf, pos, **kwargs)
        add_call(annotated, pos, raw=raw_annotation_info(raw_mq, raw_dp), **kwargs)
        add_gvcf_site(gvcf, pos, raw_mq=raw_mq, raw_dp=raw_dp)

    for i in range(N_CALLS):
        pos = 1000 + 3000 * i + offset
        kind = i % 10
        if kind == 0:  # false positive: low MQ
            call(pos, 100 * 16, 16, gt=(0, 1), ad=(10, 6))
            add_happy_record(happy, pos, query=((0, 1), "FP", ".", "SNP"))
        elif kind == 1:  # genotype error: called het, almost all reads alt, truth 1/1
            call(pos, 3600 * 20, 20, gt=(0, 1), ad=(1, 19))
            add_happy_record(
                happy, pos, query=((0, 1), "FP", "am", "SNP"), truth=((1, 1), "FN", "am", "SNP")
            )
            truth.add(pos=pos, ref="A", alts="C", samples={"T": {"GT": (1, 1)}})
        else:  # true positive het
            call(pos, 3600 * 20, 20, gt=(0, 1), ad=(10, 10))
            add_happy_record(
                happy, pos, query=((0, 1), "TP", "gm", "SNP"), truth=((0, 1), "TP", "gm", "SNP")
            )
            truth.add(pos=pos, ref="A", alts="C", samples={"T": {"GT": (0, 1)}})

        filtered = pos + 1500
        if i % 3 == 1:  # artefact: stays filtered
            call(filtered, 100 * 16, 16, gt=(0, 1), ad=(10, 6), filter="LowQual")
            continue
        truth_gt = (0, 1) if i % 3 == 0 else (1, 1)
        call(
            filtered,
            3600 * 20,
            20,
            gt=(0, 1),
            ad=(10, 10) if i % 3 == 0 else (1, 19),
            filter="LowQual",
        )
        add_happy_record(
            happy, filtered, query=(None, ".", ".", "NOCALL"), truth=(truth_gt, "FN", ".", "SNP")
        )
        truth.add(pos=filtered, ref="A", alts="C", samples={"T": {"GT": truth_gt}})
    add_call(vcf, 5, contig="chr3", filter="LowQual")  # not on the extracted contigs
    add_call(annotated, 5, contig="chr3", filter="LowQual")
    confident = tmp_path / f"{name}.confident.bed"
    confident.write_text("chr1\t0\t1000000\n")
    return {
        "vcf": write(vcf, tmp_path / f"{name}.vcf.gz"),
        "annotated_vcf": write(annotated, tmp_path / f"{name}.annotated.vcf.gz"),
        "gvcf": write(gvcf, tmp_path / f"{name}.g.vcf.gz"),
        "happy": write(happy, tmp_path / f"{name}.happy.vcf.gz"),
        "truth_vcf": write(truth, tmp_path / f"{name}.truth.vcf.gz"),
        "truth_bed": confident,
    }


def _scored(rec: pysam.VariantRecord) -> tuple[object, ...]:
    """What scoring decides for a record: its position, FILTER, GT and winnowr's INFO."""
    winnowr_info = tuple(sorted((k, v) for k, v in rec.info.items() if k.startswith("WINNOWR_")))
    return rec.pos, tuple(rec.filter.keys()), rec.samples[0]["GT"], winnowr_info


def _run_cli(monkeypatch: MonkeyPatch, *args: str) -> None:
    """Run a winnowr command line."""
    monkeypatch.setattr("sys.argv", ["winnowr", *args])
    main.run()


def test_extract_train_evaluate_score(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t0\t100000\n")
    regions = [f"early={bed}"]
    contigs = ["chr1", "chr2"]
    tables = []
    for i, name in enumerate(["a", "b", "c"]):
        table = tmp_path / f"{name}.parquet"
        paths = _sample(tmp_path, name, offset=i)
        extract(
            vcf=paths["vcf"],
            gvcf=paths["gvcf"],
            happy=paths["happy"],
            truth_vcf=paths["truth_vcf"],
            truth_bed=paths["truth_bed"],
            output=table,
            regions=regions,
            contigs=contigs,
            sample=name,
        )
        tables.append(table)

    prefix = tmp_path / "model" / "snp"
    prefix.parent.mkdir()
    # through the CLI, so defopt parsing of defaults (e.g. --task) is exercised
    _run_cli(
        monkeypatch,
        "train", "--tables", str(tables[0]), str(tables[1]), "--variant-type", "SNP",
        "--output", str(prefix), "--max-rounds", "30", "--min-child-weight", "0.5",
        "--validation-fraction", "0", "--threads", "1",
    )  # fmt: skip
    sidecar = Path(f"{prefix}.winnowr.json")
    promote_prefix = tmp_path / "model" / "snp-promote"
    train(
        tables=tables[:2],
        variant_type=VariantType.SNP,
        task=Task.PROMOTE,
        output=promote_prefix,
        max_rounds=30,
        min_child_weight=0.5,
        validation_fraction=0.0,
        threads=1,
    )
    promote_sidecar = Path(f"{promote_prefix}.winnowr.json")

    metrics_path = tmp_path / "c.metrics.tsv"
    evaluate(
        models=[sidecar, promote_sidecar],
        table=tables[2],
        output=metrics_path,
        tuned=True,
    )
    metrics = {m.policy: m for m in PolicyMetric.read(metrics_path)}
    assert "training sample" not in caplog.text
    evaluate(models=[sidecar], table=tables[0], output=tmp_path / "a.metrics.tsv")
    assert "a is a training sample of" in caplog.text and "in-sample" in caplog.text
    baseline, genotyper = metrics["baseline"], metrics["genotyper"]
    assert (baseline.tp, baseline.fp, baseline.fn) == (240, 60, 230)
    assert genotyper.dropped == 30 and genotyper.flips_fixed == 30
    assert (genotyper.tp, genotyper.fp, genotyper.fn) == (270, 0, 200)
    assert metrics["veto"].f1_gain > 0 and "genotyper_tuned" in metrics
    both = metrics["genotyper+promote"]
    assert both.promoted == 200 and both.promoted_correct == 200
    assert (both.tp, both.fp, both.fn) == (470, 0, 0)
    # PASS-call columns come from the genotyper; promote flips have their own columns
    assert (both.t_flip, both.flipped, both.flips_fixed) == (genotyper.t_flip, 30, 30)
    assert both.promoted_flipped == 100
    assert both.t_promote_flip == metrics["promote"].t_promote_flip < NEVER
    assert metrics["promote"].f1_gain > metrics["promote_no_flip"].f1_gain > 0

    out = tmp_path / "c.scored.vcf.gz"
    vcf_c = tmp_path / "c.vcf.gz"
    models = [sidecar, promote_sidecar]
    _run_cli(
        monkeypatch,
        "score", "--vcf", str(vcf_c), "--gvcf", str(tmp_path / "c.g.vcf.gz"),
        "--models", *map(str, models), "--output", str(out), "--regions", *regions,
    )  # fmt: skip
    assert "ignored" not in caplog.text
    with pysam.VariantFile(str(out)) as scored:
        records = list(scored)
    # a VCF that keeps the raw annotations scores the same without --gvcf
    annotated_out = tmp_path / "c.annotated.scored.vcf.gz"
    _run_cli(
        monkeypatch,
        "score", "--vcf", str(tmp_path / "c.annotated.vcf.gz"),
        "--models", *map(str, models), "--output", str(annotated_out), "--regions", *regions,
    )  # fmt: skip
    with pysam.VariantFile(str(annotated_out)) as scored:
        annotated_records = list(scored)
    assert [_scored(r) for r in annotated_records] == [_scored(r) for r in records]
    assert len(records) == 2 * N_CALLS + 1
    # sample c's PASS calls are at 1000 + 3000 i + 2 and its non-PASS calls 1500 bp later (see
    # _sample): i % 10 == 0 are 0/0 FPs, i % 10 == 1 hets that are truly 1/1; non-PASS i % 3 == 0
    # are real hets and i % 3 == 2 hets that are truly 1/1
    pass_calls = {i: 1002 + 3000 * i for i in range(N_CALLS)}
    filtered_calls = {i: 2502 + 3000 * i for i in range(N_CALLS)}
    dropped = {r.pos for r in records if DROP_FILTER in r.filter}
    assert dropped == {pos for i, pos in pass_calls.items() if i % 10 == 0}
    promoted = [r for r in records if ORIGINAL_FILTER_INFO in r.info]
    assert {r.pos for r in promoted} == {pos for i, pos in filtered_calls.items() if i % 3 != 1}
    assert all(
        "PASS" in r.filter and r.info[ORIGINAL_FILTER_INFO] == ("LowQual",) for r in promoted
    )
    flipped = [r for r in records if r.info.get(FLIP_INFO)]
    assert {r.pos for r in flipped} == {pos for i, pos in pass_calls.items() if i % 10 == 1} | {
        pos for i, pos in filtered_calls.items() if i % 3 == 2
    }
    for r in flipped:
        assert (r.samples[0]["GT"], r.samples[0][ORIGINAL_GT_FORMAT]) == ((1, 1), "0/1")
        # the GQ is the model's confidence in the flip, from the probabilities of its own task
        key = PROBABILITIES_INFO if PROBABILITIES_INFO in r.info else PROMOTE_PROBABILITIES_INFO
        assert abs(r.samples[0]["GQ"] - flip_gq(r.info[key][2])) <= 1
    stays = [r for r in records if r.chrom == "chr1" and "LowQual" in r.filter]
    assert len(stays) == 100 and all(PROMOTE_PROBABILITIES_INFO in r.info for r in stays)
    unscored = records[-1]
    assert unscored.chrom == "chr3" and PROMOTE_PROBABILITIES_INFO not in unscored.info

    # nothing on the models' contigs: the VCF is written with its records unchanged
    lonely = vcf_builder()
    add_call(lonely, 5, contig="chr3", filter="LowQual")
    lonely_vcf = write(lonely, tmp_path / "lonely.vcf.gz")
    lonely_out = tmp_path / "lonely.scored.vcf"
    # extra --regions and --strlen tables that no model uses are ignored, with a warning
    score(
        vcf=lonely_vcf,
        gvcf=write(gvcf_builder(), tmp_path / "lonely.g.vcf.gz"),
        models=models,
        output=lonely_out,
        regions=[*regions, f"extra={bed}"],
        strlen=[tmp_path / "missing.strlen.tsv"],
    )
    assert "No model uses the --regions ['extra']" in caplog.text
    assert "--strlen is ignored" in caplog.text
    with pysam.VariantFile(str(lonely_vcf)) as before, pysam.VariantFile(str(lonely_out)) as after:
        assert [str(r) for r in after] == [str(r) for r in before]

    with pytest.raises(ValueError, match="needs region BEDs"):
        score(vcf=vcf_c, gvcf=tmp_path / "c.g.vcf.gz", models=[sidecar], output=out)

    # the sidecar records the training reference; calls on another reference are rejected
    info = WinnowrModel.load(sidecar).info
    assert info.reference_contigs == {"chr1": 249250621, "chr2": 243199373}
    other = Path(f"{tmp_path / 'other'}.winnowr.json")
    other.write_text(sidecar.read_text().replace("249250621", "248956422"))
    Path(f"{tmp_path / 'other'}.xgb.json").write_bytes(Path(f"{prefix}.xgb.json").read_bytes())
    with pytest.raises(ValueError, match=r"not on the reference .* chr1 \(249,250,621 bp, model"):
        score(vcf=vcf_c, gvcf=tmp_path / "c.g.vcf.gz", models=[other], output=out, regions=regions)
    with pytest.raises(ValueError, match=r"Feature table .* is not on the reference"):
        evaluate(models=[other], table=tables[2], output=tmp_path / "other.tsv")


def test_promote_labels_need_truth_bed_and_happy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--truth-vcf, --truth-bed and --happy"):
        build_feature_table(
            "s", tmp_path / "v.vcf.gz", tmp_path / "g.vcf.gz", [], ["chr1"],
            happy=None, truth_vcf=tmp_path / "t.vcf.gz",
        )  # fmt: skip


def test_region_beds_must_share_a_contig_with_the_extraction(tmp_path: Path) -> None:
    bed = tmp_path / "numbered.bed"
    bed.write_text("1\t0\t100\n")
    with pytest.raises(ValueError, match="numbered.bed has no intervals"):
        build_feature_table(
            "s", tmp_path / "v.vcf.gz", tmp_path / "g.vcf.gz",
            [RegionSet.from_bed("seg", bed)], ["chr1"],
        )  # fmt: skip


def test_spanning_deletion_and_mixed_calls_are_labelled(tmp_path: Path) -> None:
    """PASS `1/2` calls with a `*` allele or mixed SNP and indel alleles are trained on too."""
    vcf, gvcf, happy = vcf_builder(), gvcf_builder(), happy_builder()
    add_call(vcf, 100, gt=(1, 2), alt="*,C", ad=(1, 8, 9))  # SNP next to a deletion
    add_call(vcf, 200, gt=(1, 2), alt="C,AT", ad=(1, 8, 9))  # mixed SNP and insertion: indel
    add_call(vcf, 300)
    for pos, alt in ((100, "*,C"), (200, "C,AT"), (300, "C")):
        add_gvcf_site(gvcf, pos, alt=alt)
    # hap.py drops the `*` allele and scores each allele of the mixed call on its own
    add_happy_record(happy, 100, query=((0, 1), "TP", "gm", "SNP"))
    add_happy_record(happy, 200, query=((0, 1), "TP", "gm", "SNP"))
    add_happy_record(happy, 200, alt="AT", query=((0, 1), "FP", ".", "INDEL"))
    add_happy_record(happy, 300, query=((0, 1), "TP", "gm", "SNP"))
    table, _ = build_feature_table(
        "s",
        write(vcf, tmp_path / "s.vcf.gz"),
        write(gvcf, tmp_path / "s.g.vcf.gz"),
        [],
        ["chr1"],
        happy=write(happy, tmp_path / "s.happy.vcf.gz"),
    )
    rows = table.set_index("pos")
    assert rows["variant_type"].to_dict() == {100: "snp", 200: "indel", 300: "snp"}
    assert rows["label"].to_dict() == {100: Label.KEEP, 200: Label.DROP, 300: Label.KEEP}
    assert rows.loc[300, "nb_calls_500"] == 2


def _extract_tables(tmp_path: Path, names: list[str], sample: bool = True) -> list[Path]:
    """Extract promote-labelled feature tables of `_sample` data for the named samples."""
    tables = []
    for i, name in enumerate(names):
        paths = _sample(tmp_path, name, offset=i)
        table = tmp_path / f"{name}.parquet"
        extract(
            vcf=paths["vcf"],
            gvcf=paths["gvcf"],
            happy=paths["happy"],
            truth_vcf=paths["truth_vcf"],
            truth_bed=paths["truth_bed"],
            output=table,
            contigs=["chr1"],
            sample=name if sample else None,
        )
        tables.append(table)
    return tables


@pytest.mark.parametrize("task", [Task.GENOTYPE, Task.PROMOTE])
def test_train_holds_out_sample_groups_together(
    tmp_path: Path, mocker: MockerFixture, task: Task
) -> None:
    tables = _extract_tables(tmp_path, ["a", "b", "c"])
    fit = mocker.spy(winnowr.training, "fit_booster")

    def train_grouped(groups: list[str]) -> None:
        train(
            tables=tables,
            variant_type=VariantType.SNP,
            output=tmp_path / "m",
            task=task,
            sample_groups=groups,
            neighbourhood=False,
            max_rounds=5,
            validation_fraction=0.0,
            threads=1,
        )

    with pytest.raises(ValueError, match=r"unknown samples: \['z'\]"):
        train_grouped(["a,z"])
    assert fit.call_count == 0
    train_grouped(["a,b"])
    # {a, b} is held out together (fitted on c), then c (fitted on a + b), then the final fit
    sizes = [len(call.args[0]) for call in fit.call_args_list]
    assert len(sizes) == 3 and sizes[0] + sizes[1] == sizes[2] and sizes[1] == 2 * sizes[0]
    # --no-neighbourhood leaves out the neighbourhood features
    features = WinnowrModel.load(tmp_path / "m.winnowr.json").info.features
    assert features and not any(name.startswith("nb_") for name in features)


def test_extract_takes_the_sample_name_from_the_vcf(tmp_path: Path) -> None:
    (table,) = _extract_tables(tmp_path, ["a"], sample=False)
    assert read_feature_table(table)[1].sample == SAMPLE
