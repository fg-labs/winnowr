from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from tests.test_model import FAST
from tests.test_model import synthetic_calls
from tests.util import add_call
from tests.util import vcf_builder
from tests.util import write
from winnowr.common import NEVER
from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.labels import Label
from winnowr.model import ModelInfo
from winnowr.model import WinnowrModel
from winnowr.model import fit_booster
from winnowr.model import num_classes
from winnowr.regenotype import regenotype_target
from winnowr.regenotype import regenotyped_alleles
from winnowr.rescore import Mode
from winnowr.rescore import RepeatFlip
from winnowr.scoring import DROP_FILTER
from winnowr.scoring import FLIP_INFO
from winnowr.scoring import MAX_GQ
from winnowr.scoring import ORIGINAL_FILTER_INFO
from winnowr.scoring import ORIGINAL_GT_FORMAT
from winnowr.scoring import PROBABILITIES_INFO
from winnowr.scoring import PROMOTE_PROBABILITIES_INFO
from winnowr.scoring import REGENOTYPE_INFO
from winnowr.scoring import REGENOTYPE_PROBABILITY_INFO
from winnowr.scoring import Decision
from winnowr.scoring import Key
from winnowr.scoring import _regenotype_targets
from winnowr.scoring import decide_calls
from winnowr.scoring import flip_gq
from winnowr.scoring import write_scored_vcf
from winnowr.strlen import RepeatTract
from winnowr.thresholds import PromoteThresholds
from winnowr.thresholds import Thresholds


def test_flip_gq() -> None:
    assert flip_gq(0.9) == 10
    assert flip_gq(0.999) == 30
    assert flip_gq(1.0) == MAX_GQ
    assert flip_gq(1 - 1e-12) == MAX_GQ


def test_decide_calls() -> None:
    calls = synthetic_calls(500, seed=3)
    booster = fit_booster(calls, calls["label"], ["x", "y"], FAST, num_classes(Task.GENOTYPE))
    genotype = WinnowrModel(
        booster,
        ModelInfo(
            variant_type=VariantType.SNP,
            thresholds={
                Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0),
                Mode.VETO: Thresholds(0.3, NEVER, 0.0),
            },
            features=("x", "y"),
            region_names=(),
            training_samples=("a",),
            contigs=("chr1",),
        ),
    )
    promote = WinnowrModel(
        booster,
        ModelInfo(
            variant_type=VariantType.SNP,
            thresholds={},
            contigs=("chr1",),
            task=Task.PROMOTE,
            promote_thresholds=PromoteThresholds(0.5, 0.5, 0.0),
            features=("x", "y"),
            region_names=(),
            training_samples=("a",),
        ),
    )
    table = calls.assign(
        chrom="chr1",
        pos=range(len(calls)),
        ref="A",
        alt="C",
        variant_type=["snp"] * 250 + ["indel"] * 250,
        filter=["PASS", "LowQual"] * 250,
        overlaps_pass=[False, False, False, True] * 125,
    )

    genotyper = decide_calls(table, [genotype], Mode.GENOTYPER)
    assert [key for key in genotyper] == [
        ("chr1", pos, "A", "C", "PASS") for pos in range(0, 250, 2)
    ]
    p = genotype.predict(table.iloc[:250:2])
    drops = np.array([d.drop for d in genotyper.values()])
    flips = np.array([d.flip for d in genotyper.values()])
    np.testing.assert_array_equal(drops, p[:, 0] >= 0.5)
    assert flips.any() and not (flips & drops).any()
    assert next(iter(genotyper.values())).probabilities == pytest.approx(tuple(p[0]))

    veto = decide_calls(table, [genotype], Mode.VETO)
    np.testing.assert_array_equal([d.drop for d in veto.values()], p[:, 0] + p[:, 2] >= 0.3)
    assert not any(d.flip for d in veto.values())

    # promote models score non-PASS calls that do not overlap a PASS call
    both = decide_calls(table, [genotype, promote], Mode.GENOTYPER)
    promoted = {key: d for key, d in both.items() if d.task is Task.PROMOTE}
    assert [key[1] for key in promoted] == list(range(1, 250, 4))
    assert any(d.promote for d in promoted.values())
    assert not any(d.drop for d in promoted.values())

    assert decide_calls(table, [], Mode.VETO) == {}
    with pytest.raises(ValueError, match="one model per task and variant type"):
        decide_calls(pd.DataFrame(), [genotype, genotype], Mode.GENOTYPER)


def _decision(
    pos: int,
    task: Task,
    p: tuple[float, ...],
    ref: str = "A",
    alt: str = "C",
    filter: str = "PASS",  # noqa: A002  # the VCF column
    regenotype: tuple[tuple[str, ...], tuple[int, int]] | None = None,
    **actions: bool,
) -> tuple[Key, Decision]:
    return ("chr1", pos, ref, alt, filter), Decision(task, p, regenotype=regenotype, **actions)


def test_decide_calls_with_a_repeat_flip_threshold(caplog: pytest.LogCaptureFixture) -> None:
    calls = synthetic_calls(500, seed=3)
    booster = fit_booster(calls, calls["label"], ["x", "y"], FAST, num_classes(Task.GENOTYPE))
    repeat_flip = RepeatFlip(min_period=2, min_units=11, t_flip=NEVER)
    info = ModelInfo(
        variant_type=VariantType.INDEL,
        thresholds={
            Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0, repeat_flip=repeat_flip),
            Mode.VETO: Thresholds(0.3, NEVER, 0.0),
        },
        features=("x", "y"),
        region_names=(),
        training_samples=("a",),
        contigs=("chr1",),
    )
    # every other call is in a long dinucleotide repeat, where flips are off
    table = calls.assign(
        chrom="chr1",
        pos=range(len(calls)),
        ref="AC",
        alt="A",
        variant_type="indel",
        filter="PASS",
        str_period=[2.0, np.nan] * 250,
        str_repeats=[12.0, np.nan] * 250,
    )
    model = WinnowrModel(booster, info)
    flips = np.array([d.flip for d in decide_calls(table, [model], Mode.GENOTYPER).values()])
    p = model.predict(table)
    expected = (p[:, Label.FLIP] >= 0.5) & (p[:, Label.DROP] < 0.5)
    expected[::2] = False
    np.testing.assert_array_equal(flips, expected)
    assert expected[1::2].any()
    # without repeat information the stratum selects nothing, with a warning
    no_repeats = table.assign(str_period=np.nan, str_repeats=np.nan)
    flips = np.array([d.flip for d in decide_calls(no_repeats, [model], Mode.GENOTYPER).values()])
    np.testing.assert_array_equal(flips, (p[:, Label.FLIP] >= 0.5) & (p[:, Label.DROP] < 0.5))
    assert "The indel genotype model has a long-tandem-repeat flip threshold, but no" in caplog.text
    with pytest.raises(ValueError, match="needs the calls' str_period and str_repeats columns"):
        decide_calls(table.drop(columns="str_repeats"), [model], Mode.GENOTYPER)


@pytest.mark.parametrize("suffix", [".vcf", ".vcf.gz", ".bcf"])
def test_write_scored_vcf(tmp_path: Path, suffix: str) -> None:
    builder = vcf_builder()
    for key, number, kind in (("AC", "A", "Integer"), ("AF", "A", "Float")):
        for name in (key, f"MLE{key}"):
            builder.header.add_line(
                f'##INFO=<ID={name},Number={number},Type={kind},Description="{name}">'
            )
    add_call(builder, 100, gt=(1, 1), phased=True)  # flipped by the genotype model
    for key, value in (("AC", 2), ("AF", 1.0), ("MLEAC", 2), ("MLEAF", 1.0)):
        builder.records[-1].info[key] = (value,)
    add_call(builder, 200)  # dropped
    add_call(builder, 300, filter="LowQual")  # promoted and flipped
    add_call(builder, 400, gt=(0, 2), alt="C,G", ad=(5, 5, 5))  # flip decided but not 0/1 or 1/1
    add_call(builder, 500)  # not scored
    # re-genotyped from 0/1 to -1/+1 in the A4 tract at 600-603: adds the +1 allele
    add_call(builder, 600, ref="CA", alt="C", ad=(5, 5))
    for key, value in (("AC", 1), ("AF", 0.5), ("MLEAC", 1), ("MLEAF", 0.5)):
        builder.records[-1].info[key] = (value,)
    add_call(builder, 700, ref="C", alt="CA,CAA", ad=(5, 5, 0))  # re-genotyped to 1/2
    add_call(builder, 800, ref="CA", alt="C")  # re-genotype to a new allele without a tract
    vcf = write(builder, tmp_path / "in.vcf.gz")
    decisions = dict([
        _decision(100, Task.GENOTYPE, (0.01, 0.09, 0.9), flip=True),
        _decision(200, Task.GENOTYPE, (0.8, 0.1, 0.1), drop=True),
        _decision(
            300, Task.PROMOTE, (0.0, 0.001, 0.999), filter="LowQual", promote=True, flip=True
        ),
        _decision(400, Task.GENOTYPE, (0.0, 0.1, 0.9), alt="C,G", flip=True),
        _decision(
            600, Task.GENOTYPE, (0.0, 0.1, 0.0, 0.9), ref="CA",
            regenotype=regenotype_target(600, "CA", "C", (-1.0, 1.0), 601.0, 4.0, "A"),
        ),
        _decision(
            700, Task.GENOTYPE, (0.0, 0.2, 0.0, 0.8), ref="C", alt="CA,CAA",
            regenotype=regenotype_target(700, "C", "CA,CAA", (1.0, 2.0), 701.0, 0.0, "A"),
        ),
        # a re-genotype model's call that is not re-genotyped
        _decision(800, Task.GENOTYPE, (0.0, 0.2, 0.0, 0.8), ref="CA"),
    ])  # fmt: skip
    output = tmp_path / f"out{suffix}"
    write_scored_vcf(vcf, output, decisions)
    if suffix != ".vcf":
        index = Path(f"{output}.{'csi' if suffix == '.bcf' else 'tbi'}")
        assert index.is_file()
    with pysam.VariantFile(str(output)) as scored:
        records = {rec.pos: rec for rec in scored}

    flipped = records[100]
    sample = flipped.samples[0]
    assert (sample["GT"], sample.phased, sample[ORIGINAL_GT_FORMAT]) == ((0, 1), False, "1/1")
    assert sample["GQ"] == flip_gq(0.9) and flipped.info[FLIP_INFO]
    assert flipped.info[PROBABILITIES_INFO] == pytest.approx((0.01, 0.09, 0.9))
    assert (flipped.info["AC"], flipped.info["MLEAC"]) == ((1,), (1,))
    assert (flipped.info["AF"], flipped.info["MLEAF"]) == ((0.5,), (0.5,))
    assert DROP_FILTER not in flipped.filter

    assert list(records[200].filter) == [DROP_FILTER] and FLIP_INFO not in records[200].info

    promoted = records[300]
    assert list(promoted.filter) == ["PASS"]
    assert promoted.info[ORIGINAL_FILTER_INFO] == ("LowQual",)
    assert PROBABILITIES_INFO not in promoted.info
    assert promoted.info[PROMOTE_PROBABILITIES_INFO] == pytest.approx((0.0, 0.001, 0.999))
    assert promoted.samples[0]["GT"] == (1, 1) and promoted.samples[0]["GQ"] == flip_gq(0.999)

    skipped = records[400]
    assert skipped.samples[0]["GT"] == (0, 2) and FLIP_INFO not in skipped.info
    assert PROBABILITIES_INFO not in records[500].info

    extended = records[600]
    assert extended.alleles == ("CAAAA", "CAAA", "CAAAAA")
    sample = extended.samples[0]
    assert (sample["GT"], sample[ORIGINAL_GT_FORMAT], sample["GQ"]) == ((1, 2), "0/1", 10)
    assert sample["AD"] == (5, 5, None)
    assert sample["PL"] == (0, 10, 20, None, None, None)
    assert (extended.info["AC"], extended.info["AF"]) == ((1, 1), (0.5, 0.5))
    assert extended.info[REGENOTYPE_INFO] and FLIP_INFO not in extended.info
    assert extended.info[PROBABILITIES_INFO] == pytest.approx((0.0, 0.1, 0.0))
    assert extended.info[REGENOTYPE_PROBABILITY_INFO] == pytest.approx(0.9)

    existing = records[700]
    assert existing.alleles == ("C", "CA", "CAA") and existing.samples[0]["GT"] == (1, 2)
    assert existing.info[REGENOTYPE_INFO]

    not_regenotyped = records[800]
    assert (
        not_regenotyped.samples[0]["GT"] == (0, 1) and REGENOTYPE_INFO not in not_regenotyped.info
    )
    assert not_regenotyped.info[REGENOTYPE_PROBABILITY_INFO] == pytest.approx(0.8)


def test_write_scored_vcf_duplicate_keys(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    builder = vcf_builder()
    add_call(builder, 100, gt=(1, 1))  # scored and flipped
    add_call(builder, 100, gt=(None, None), filter="LowQual")  # same alleles, not scored
    add_call(builder, 100, gt=(0, 1), filter="LowQual")  # same alleles, not scored
    add_call(builder, 100, gt=(1, 1))  # an exact repeat of the first: left unchanged
    add_call(builder, 200, gt=(0, None))  # a flip of a genotype with a missing allele: skipped
    vcf = write(builder, tmp_path / "in.vcf.gz")
    decisions = dict([
        _decision(100, Task.GENOTYPE, (0.0, 0.1, 0.9), flip=True),
        _decision(200, Task.GENOTYPE, (0.0, 0.1, 0.9), flip=True),
    ])
    output = tmp_path / "out.vcf"
    write_scored_vcf(vcf, output, decisions)
    with pysam.VariantFile(str(output)) as scored:
        records = list(scored)
    flipped, missing, het, repeat, partial = records
    assert flipped.samples[0]["GT"] == (0, 1) and flipped.info[FLIP_INFO]
    for unchanged, gt in ((missing, (None, None)), (het, (0, 1)), (repeat, (1, 1))):
        assert unchanged.samples[0]["GT"] == gt
        assert PROBABILITIES_INFO not in unchanged.info and FLIP_INFO not in unchanged.info
    assert partial.samples[0]["GT"] == (0, None) and FLIP_INFO not in partial.info
    assert "Left 1 records unchanged" in caplog.text


def _written_formats(vcf: Path) -> dict[int, dict[str, str]]:
    """The FORMAT values of each record of a plain-text VCF, by position, as written."""
    formats = {}
    for line in vcf.read_text().splitlines():
        if not line.startswith("#"):
            fields = line.split("\t")
            keys, values = fields[8].split(":"), fields[9].split(":")
            formats[int(fields[1])] = dict(zip(keys, values, strict=True))
    return formats


def test_write_scored_vcf_clears_phasing(tmp_path: Path) -> None:
    builder = vcf_builder()
    for key, kind in (("PGT", "String"), ("PID", "String"), ("PS", "Integer")):
        builder.header.add_line(f'##FORMAT=<ID={key},Number=1,Type={kind},Description="{key}">')
    for pos, ref, alt in ((100, "A", "C"), (200, "CA", "C"), (300, "G", "T")):
        add_call(builder, pos, ref=ref, alt=alt, phased=True)
        sample = builder.records[-1].samples[0]
        sample["PGT"], sample["PID"], sample["PS"] = "0|1", f"{pos}_A_C", 100
    vcf = write(builder, tmp_path / "in.vcf.gz")
    decisions = dict([
        _decision(100, Task.GENOTYPE, (0.0, 0.1, 0.9), flip=True),
        _decision(
            200, Task.GENOTYPE, (0.0, 0.1, 0.0, 0.9), ref="CA",
            regenotype=regenotype_target(200, "CA", "C", (-1.0, 1.0), 201.0, 4.0, "A"),
        ),
        _decision(300, Task.GENOTYPE, (0.0, 0.9, 0.1), ref="G", alt="T"),  # phasing untouched
    ])  # fmt: skip
    output = tmp_path / "out.vcf"
    write_scored_vcf(vcf, output, decisions)
    by_pos = _written_formats(output)
    for pos in (100, 200):
        assert (by_pos[pos]["PGT"], by_pos[pos]["PID"], by_pos[pos]["PS"]) == (".", ".", ".")
        assert "|" not in by_pos[pos]["GT"]
    assert (by_pos[300]["GT"], by_pos[300]["PGT"], by_pos[300]["PS"]) == ("0|1", "0|1", "100")


def test_regenotyped_alleles() -> None:
    tract = RepeatTract(start=100, end=104, period=1, unit="A")  # REF CA at pos 100 (0-based 99)
    assert regenotyped_alleles(100, ("CA", "C"), (-1, 0), tract) == (("CA", "C"), (0, 1))
    assert regenotyped_alleles(100, ("CA", "C"), (1, 1), tract) == (
        ("CAAAA", "CAAA", "CAAAAA"),
        (2, 2),
    )
    # deleting more than the tract, no tract, or a tract without an anchor base: not buildable
    assert regenotyped_alleles(100, ("CA", "C"), (-5, 0), tract) is None
    assert regenotyped_alleles(100, ("CA", "C"), (1, 1), None) is None
    assert regenotyped_alleles(101, ("AA", "A"), (2, 2), tract) is None
    # an existing ALT is used only if it is the tract resized: in a T tract (0-based 100-105) +1
    # is AT, not AG, and -1 is the T deletion, not the complex GT
    t_tract = RepeatTract(start=100, end=105, period=1, unit="T")
    assert regenotyped_alleles(100, ("A", "AG", "AT"), (1, 1), t_tract) == (
        ("A", "AG", "AT"),
        (2, 2),
    )
    assert regenotyped_alleles(100, ("A", "AG"), (0, 1), t_tract) == (
        ("ATTTTT", "AGTTTTT", "ATTTTTT"),
        (0, 2),
    )
    assert regenotyped_alleles(100, ("ATT", "A", "GT"), (-2, -1), t_tract) == (
        ("ATTTTT", "ATTT", "GTTTT", "ATTTT"),
        (1, 3),
    )
    # without a usable tract, only an unambiguous ALT of the right length is used
    assert regenotyped_alleles(101, ("AA", "A"), (-1, 0), tract) == (("AA", "A"), (0, 1))
    assert regenotyped_alleles(101, ("A", "AG", "AT"), (0, 1), tract) is None
    assert regenotyped_alleles(101, ("A", "AG"), (0, 0), None) == (("A", "AG"), (0, 0))
    # a dinucleotide tract keeps its phase; symbolic alleles are kept as they are
    dinucleotide = RepeatTract(start=10, end=16, period=2, unit="CA")
    assert regenotyped_alleles(10, ("GCA", "G", "<NON_REF>"), (-2, 4), dinucleotide) == (
        ("GCACACA", "GCACA", "<NON_REF>", "GCACACACACA"),
        (1, 3),
    )


def test_unbuildable_regenotypes_are_skipped(caplog: pytest.LogCaptureFixture) -> None:
    calls = pd.DataFrame({
        "pos": [100, 200],
        "ref": ["CA", "CA"],
        "alt": ["C", "C"],
        "lc_delta_lo": [1.0, -1.0],
        "lc_delta_hi": [1.0, 1.0],
        "sl_tract_start": [np.nan, 201.0],  # no tract for the first: its +1 allele cannot be built
        "sl_tract_len": [np.nan, 4.0],
        "sl_unit": [None, "A"],
    })
    targets = _regenotype_targets(calls, np.array([True, True]))
    assert list(targets) == [1] and "Skipped 1 re-genotypes" in caplog.text
