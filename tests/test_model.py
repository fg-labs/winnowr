import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.labels import Label
from winnowr.model import ModelInfo
from winnowr.model import TrainingParams
from winnowr.model import WinnowrModel
from winnowr.model import check_reference
from winnowr.model import fit_booster
from winnowr.model import num_classes
from winnowr.model import to_matrix
from winnowr.rescore import Mode
from winnowr.stutter import fit_stutter_model
from winnowr.thresholds import PromoteThresholds
from winnowr.thresholds import Thresholds

FAST = TrainingParams(max_rounds=30, min_child_weight=1.0, threads=1)


def synthetic_calls(n: int, seed: int) -> pd.DataFrame:
    """Calls whose label is determined by feature `x` (with a little noise in `y`)."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    label = np.where(x < 0.15, Label.DROP, np.where(x > 0.85, Label.FLIP, Label.KEEP))
    return pd.DataFrame({
        "x": x,
        "y": rng.normal(size=n),
        "label": label.astype(np.float64),
        "flip_ok": label == Label.FLIP,
        "flippable": True,
    })


def test_to_matrix_maps_infinity_to_missing_and_checks_columns() -> None:
    frame = pd.DataFrame({"a": [1.0, np.inf], "b": [None, 2.0]})
    matrix = to_matrix(frame, ["b", "a"])
    assert matrix.dtype == np.float32
    assert np.isnan(matrix[0, 0]) and np.isnan(matrix[1, 1]) and matrix[0, 1] == 1.0
    with pytest.raises(ValueError, match="lacks model features"):
        to_matrix(frame, ["a", "c"])


def test_fit_predict_save_load_round_trip(tmp_path: Path) -> None:
    calls = synthetic_calls(3000, seed=1)
    booster = fit_booster(calls, calls["label"], ["x", "y"], FAST, num_classes(Task.GENOTYPE))
    info = ModelInfo(
        variant_type=VariantType.SNP,
        features=("x", "y"),
        region_names=(),
        thresholds={
            Mode.GENOTYPER: Thresholds(0.5, 0.9, 0.01),
            Mode.VETO: Thresholds(0.9, 1.01, 0),
        },
        training_samples=("a",),
        contigs=("chr1",),
        params=FAST,
    )
    model = WinnowrModel(booster=booster, info=info)
    probabilities = model.predict(calls)
    assert probabilities.shape == (3000, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
    accuracy = (probabilities.argmax(axis=1) == calls["label"].to_numpy()).mean()
    assert accuracy > 0.95

    sidecar = model.save(tmp_path / "snp")
    assert sidecar.name == "snp.winnowr.json" and (tmp_path / "snp.xgb.json").is_file()
    loaded = WinnowrModel.load(sidecar)
    assert loaded.info == info
    np.testing.assert_allclose(loaded.predict(calls), probabilities)
    assert model.predict(calls.iloc[:0]).shape == (0, 3)


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in .winnowr.json"):
        WinnowrModel.load(tmp_path / "x.json")
    with pytest.raises(ValueError, match=r"format version 99 \(written by winnowr 9.0\)"):
        ModelInfo.from_json('{"format_version": 99, "winnowr_version": "9.0"}')


def _info(
    thresholds: dict[Mode, Thresholds],
    task: Task = Task.GENOTYPE,
    promote_thresholds: PromoteThresholds | None = None,
) -> ModelInfo:
    return ModelInfo(
        variant_type=VariantType.SNP,
        features=("x",),
        region_names=(),
        thresholds=thresholds,
        training_samples=("a",),
        contigs=("chr1",),
        task=task,
        promote_thresholds=promote_thresholds,
    )


def test_model_info_thresholds_must_match_the_task() -> None:
    genotype = {Mode.GENOTYPER: Thresholds(0.5, 0.9, 0.0), Mode.VETO: Thresholds(0.9, 1.01, 0.0)}
    promote = PromoteThresholds(0.5, 0.5, 0.0)
    with pytest.raises(ValueError, match=r"missing: \['veto'\]"):
        _info({Mode.GENOTYPER: Thresholds(0.5, 0.9, 0.0)})
    with pytest.raises(ValueError, match="genotype model"):
        _info(genotype, promote_thresholds=promote)
    with pytest.raises(ValueError, match="promote model"):
        _info({}, Task.PROMOTE)
    with pytest.raises(ValueError, match="promote model"):
        _info(genotype, Task.PROMOTE, promote)
    assert _info({}, Task.PROMOTE, promote).promote_thresholds == promote


def test_promote_sidecar_round_trip() -> None:
    info = ModelInfo(
        variant_type=VariantType.INDEL,
        features=("x",),
        region_names=("seg",),
        thresholds={},
        training_samples=("a", "b"),
        contigs=("chr1",),
        task=Task.PROMOTE,
        promote_thresholds=PromoteThresholds(0.6, 0.5, 0.001),
    )
    assert ModelInfo.from_json(info.to_json()) == info


def test_regenotype_sidecar_round_trip_and_checks() -> None:
    stutter = fit_stutter_model([1], [10], [0], [{0: 9, 1: 1}])
    genotype = {Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0, 0.6), Mode.VETO: Thresholds(0.5, 1.01, 0)}
    info = replace(
        _info(genotype), variant_type=VariantType.INDEL, stutter=stutter, regenotype=True
    )
    loaded = ModelInfo.from_json(info.to_json())
    assert loaded == info and loaded.thresholds[Mode.GENOTYPER].t_regenotype == 0.6
    assert num_classes(Task.GENOTYPE, regenotype=True) == 4
    assert num_classes(Task.GENOTYPE) == 3 and num_classes(Task.PROMOTE) == 3
    with pytest.raises(ValueError, match="re-genotype model"):
        replace(info, stutter=None)
    # sidecars written before the repeat-length features load without them
    raw = json.loads(_info(genotype).to_json())
    del raw["stutter"], raw["regenotype"], raw["thresholds"]["genotyper"]["t_regenotype"]
    old = ModelInfo.from_json(json.dumps(raw))
    assert old.stutter is None and not old.regenotype
    assert old.thresholds[Mode.GENOTYPER].t_regenotype == 1.01


def test_validation_fraction_must_leave_training_calls() -> None:
    for fraction in (-0.1, 1.0):
        with pytest.raises(ValueError, match="at least 0 and below 1"):
            TrainingParams(validation_fraction=fraction)
    calls = synthetic_calls(3, seed=1)
    with pytest.raises(ValueError, match="No training calls left"):
        fit_booster(
            calls, calls["label"], ["x"], replace(FAST, validation_fraction=0.999), num_class=3
        )


def _old_sidecar(path: Path) -> Path:
    """A model saved with a format-1 sidecar, as winnowr 0.1.0 wrote before recording more."""
    calls = synthetic_calls(200, seed=1)
    booster = fit_booster(calls, calls["label"], ["x", "y"], FAST, num_classes(Task.GENOTYPE))
    genotype = {Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0), Mode.VETO: Thresholds(0.5, 1.01, 0.0)}
    sidecar = WinnowrModel(booster, replace(_info(genotype), features=("x", "y"))).save(path)
    raw = json.loads(sidecar.read_text())
    for key in ("stutter", "regenotype", "reference_contigs"):
        del raw[key]
    for thresholds in raw["thresholds"].values():
        del thresholds["t_regenotype"]
    raw["format_version"] = 1
    sidecar.write_text(json.dumps(raw))
    return sidecar


def test_old_format_sidecars_load_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = WinnowrModel.load(_old_sidecar(tmp_path / "old"))
    assert model.info.format_version == 1 and model.info.reference_contigs is None
    assert model.info.stutter is None and not model.info.regenotype
    assert model.predict(synthetic_calls(5, seed=2)).shape == (5, 3)
    assert "sidecar format 1) records no reference contigs" in caplog.text
    # not checked against any reference
    check_reference(tmp_path / "old.winnowr.json", model.info, {}, "VCF v.vcf.gz")


def test_sidecar_records_versions_and_reference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    genotype = {Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0), Mode.VETO: Thresholds(0.5, 1.01, 0.0)}
    info = replace(_info(genotype), reference_contigs={"chr1": 1000}, winnowr_version="0.0.1")
    raw = json.loads(info.to_json())
    assert (raw["format_version"], raw["reference_contigs"]) == (2, {"chr1": 1000})
    assert ModelInfo.from_json(info.to_json()) == info
    calls = synthetic_calls(50, seed=1)
    booster = fit_booster(calls, calls["label"], ["x"], FAST, num_classes(Task.GENOTYPE))
    WinnowrModel.load(WinnowrModel(booster, info).save(tmp_path / "m"))
    assert "trained with winnowr 0.0.1" in caplog.text and "no reference contigs" not in caplog.text


def test_check_reference() -> None:
    genotype = {Mode.GENOTYPER: Thresholds(0.5, 0.5, 0.0), Mode.VETO: Thresholds(0.5, 1.01, 0.0)}
    info = replace(_info(genotype), contigs=("chr1", "chr2"))
    info = replace(info, reference_contigs={"chr1": 1000, "chr2": 500})
    model = Path("m.winnowr.json")
    check_reference(model, info, {"chr1": 1000, "chr2": 500, "chrX": 9}, "VCF v")
    check_reference(model, info, {"chr1": None, "chr2": 500}, "VCF v")  # unknown length
    with pytest.raises(ValueError, match=r"VCF v is not on .* chr1 \(999 bp, model 1,000 bp\)"):
        check_reference(model, info, {"chr1": 999, "chr2": 500}, "VCF v")
    with pytest.raises(ValueError, match=r"missing contigs \['chr2'\]"):
        check_reference(model, info, {"chr1": 1000}, "VCF v")
    check_reference(model, info, {"chr1": 1000}, "table t", require_all=False)
