"""
The XGBoost 3-class model and its JSON sidecar.

A trained model is saved as two files sharing a prefix:

- `<prefix>.xgb.json`: the XGBoost booster in its portable JSON format.
- `<prefix>.winnowr.json`: the sidecar (`ModelInfo`) with the ordered feature names, the variant
  type, the cross-validated thresholds per mode, the training samples, the reference contigs
  (names and lengths) the training calls came from, the winnowr and XGBoost versions and the
  sidecar format version, and for models with repeat-length features the stutter model fitted on
  the training samples.

Sidecar format versions: 1 (winnowr 0.1.0 before the reference contigs were recorded), and 2
(adds `reference_contigs`). Older formats load, with a warning for what cannot be checked; a
format newer than this winnowr knows is an error.
"""

import json
import logging
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

from winnowr import __version__
from winnowr.common import FloatArray
from winnowr.common import Task
from winnowr.common import VariantType
from winnowr.labels import Label
from winnowr.promote_labels import PromoteLabel
from winnowr.rescore import Mode
from winnowr.stutter import StutterModel
from winnowr.thresholds import PromoteThresholds
from winnowr.thresholds import Thresholds

logger = logging.getLogger(__name__)

MODEL_SUFFIX: Final[str] = ".xgb.json"
SIDECAR_SUFFIX: Final[str] = ".winnowr.json"
SIDECAR_FORMAT_VERSION: Final[int] = 2
"""The sidecar format (schema) version this winnowr writes; it reads every version up to it."""

ContigLengths = dict[str, int | None]
"""Contig names and lengths (None when the length is unknown)."""


@dataclass(frozen=True)
class TrainingParams:
    """
    XGBoost training parameters. The defaults are deliberately regularized.

    Attributes:
        learning_rate: shrinkage per boosting round (`eta`).
        max_leaves: maximum leaves per tree (trees grow loss-guided).
        max_depth: maximum tree depth (0 = unlimited, bounded by `max_leaves`).
        min_child_weight: minimum hessian sum per leaf.
        reg_lambda: L2 regularization of leaf weights.
        max_rounds: maximum boosting rounds.
        early_stopping_rounds: stop when the validation log-loss has not improved for this many
            rounds.
        validation_fraction: fraction of training calls held out for early stopping (0 disables
            early stopping).
        seed: random seed (validation split and XGBoost).
        threads: XGBoost threads (0 = all cores).
    """

    learning_rate: float = 0.05
    max_leaves: int = 31
    max_depth: int = 0
    min_child_weight: float = 10.0
    reg_lambda: float = 1.0
    max_rounds: int = 3000
    early_stopping_rounds: int = 20
    validation_fraction: float = 0.1
    seed: int = 1
    threads: int = 0

    def __post_init__(self) -> None:
        """
        Check the validation fraction.

        Raises:
            ValueError: if `validation_fraction` is not in [0, 1).
        """
        if not 0 <= self.validation_fraction < 1:
            raise ValueError(
                "The validation fraction must be at least 0 and below 1, got "
                f"{self.validation_fraction}"
            )

    def to_xgboost(self, num_class: int) -> dict[str, str | float | int]:
        """
        The XGBoost booster parameters.

        Args:
            num_class: the number of model classes (see `num_classes`).

        Returns:
            The parameters for `xgboost.train`.
        """
        return {
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "eta": self.learning_rate,
            "max_leaves": self.max_leaves,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "lambda": self.reg_lambda,
            "seed": self.seed,
            "nthread": self.threads,
        }


def num_classes(task: Task, regenotype: bool = False) -> int:
    """
    The number of classes of a model.

    Args:
        task: the model's task.
        regenotype: whether a genotype model has the re-genotype class.

    Returns:
        4 for re-genotype models, 3 for other genotype models (`Label` without `REGENOTYPE`), or
        the size of `PromoteLabel` (promote models).
    """
    if task is Task.PROMOTE:
        return len(PromoteLabel)
    return len(Label) if regenotype else Label.REGENOTYPE.value


def to_matrix(frame: pd.DataFrame, features: Sequence[str]) -> npt.NDArray[np.float32]:
    """
    Select the features as a float32 matrix; infinities become NaN (missing).

    Args:
        frame: the calls.
        features: the feature columns, in model order.

    Returns:
        An `(n, len(features))` float32 matrix.

    Raises:
        ValueError: if a feature column is missing.
    """
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Feature table lacks model features: {missing}")
    matrix = frame[list(features)].to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
    matrix[np.isinf(matrix)] = np.nan
    return matrix


def fit_booster(
    frame: pd.DataFrame,
    labels: npt.ArrayLike,
    features: Sequence[str],
    params: TrainingParams,
    num_class: int,
) -> xgb.Booster:
    """
    Fit a multi-class booster, early-stopping on a random validation split.

    Args:
        frame: the training calls.
        labels: the class of each call (`Label` or `PromoteLabel`).
        features: the feature columns, in model order.
        params: the training parameters.
        num_class: the number of classes (see `num_classes`).

    Returns:
        The booster (when early stopping, trained up to the best round).

    Raises:
        ValueError: if no call is left for training after the validation split.
    """
    matrix = to_matrix(frame, features)
    label_array = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(params.seed)
    is_validation = rng.random(len(label_array)) < params.validation_fraction
    if is_validation.all():
        raise ValueError(
            f"No training calls left after holding out {params.validation_fraction:g} of "
            f"{len(label_array):,} for early stopping"
        )
    train = xgb.DMatrix(
        matrix[~is_validation], label=label_array[~is_validation], feature_names=list(features)
    )
    evals: list[tuple[xgb.DMatrix, str]] = []
    early_stopping: int | None = None
    if is_validation.any():
        validation = xgb.DMatrix(
            matrix[is_validation], label=label_array[is_validation], feature_names=list(features)
        )
        evals = [(validation, "validation")]
        early_stopping = params.early_stopping_rounds
    booster = xgb.train(
        params.to_xgboost(num_class),
        train,
        num_boost_round=params.max_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    if early_stopping is not None:
        booster = booster[: booster.best_iteration + 1]
    logger.info(
        f"Fitted {booster.num_boosted_rounds()} rounds on {int((~is_validation).sum()):,} calls"
    )
    return booster


def predict_probabilities(
    booster: xgb.Booster, frame: pd.DataFrame, features: Sequence[str], num_class: int
) -> FloatArray:
    """
    Predict class probabilities.

    Args:
        booster: the fitted booster.
        frame: the calls to score.
        features: the feature columns, in model order.
        num_class: the number of classes the booster was fitted with.

    Returns:
        An `(n, num_class)` array of probabilities in class order.
    """
    if frame.empty:
        return np.zeros((0, num_class), dtype=np.float64)
    matrix = xgb.DMatrix(to_matrix(frame, features), feature_names=list(features))
    probabilities: FloatArray = booster.predict(matrix).astype(np.float64).reshape(-1, num_class)
    return probabilities


@dataclass(frozen=True)
class ModelInfo:
    """
    The model sidecar.

    Attributes:
        variant_type: the variant type the model scores.
        features: the ordered feature names.
        region_names: the region sets the features use (`region_<name>`), which must be supplied
            under the same names when scoring.
        thresholds: CV-chosen thresholds per mode (`genotyper`, `veto`); empty for promote models.
        training_samples: the training sample names.
        contigs: the contigs the training calls came from.
        task: `genotype` (scores PASS calls) or `promote` (scores non-PASS calls).
        promote_thresholds: CV-chosen thresholds of a promote model.
        params: the training parameters.
        stutter: the stutter model of a model with repeat-length features (see
            `winnowr.stutter`); its features need strlen evidence in the feature table.
        regenotype: whether the model has the re-genotype class (`Label.REGENOTYPE`).
        winnowr_version: the winnowr version that trained the model.
        xgboost_version: the XGBoost version that trained the model.
        format_version: the sidecar format version.
        reference_contigs: the lengths of `contigs` in the reference the training calls were
            called on (None for sidecars of format 1, or tables extracted without them).
    """

    variant_type: VariantType
    features: tuple[str, ...]
    region_names: tuple[str, ...]
    thresholds: dict[Mode, Thresholds]
    training_samples: tuple[str, ...]
    contigs: tuple[str, ...]
    task: Task = Task.GENOTYPE
    promote_thresholds: PromoteThresholds | None = None
    params: TrainingParams = field(default_factory=TrainingParams)
    stutter: StutterModel | None = None
    regenotype: bool = False
    winnowr_version: str = __version__
    xgboost_version: str = xgb.__version__
    format_version: int = SIDECAR_FORMAT_VERSION
    reference_contigs: ContigLengths | None = None

    def __post_init__(self) -> None:
        """
        Check that the thresholds match the task.

        Raises:
            ValueError: if a genotype model lacks thresholds for a mode or has promote thresholds,
                a promote model lacks promote thresholds or has genotype thresholds, or a
                re-genotype model is not a genotype model with a stutter model.
        """
        if self.regenotype and (self.task is not Task.GENOTYPE or self.stutter is None):
            raise ValueError("A re-genotype model must be a genotype model with a stutter model")
        if self.task is Task.GENOTYPE:
            missing = [mode.value for mode in Mode if mode not in self.thresholds]
            if missing or self.promote_thresholds is not None:
                raise ValueError(
                    f"A genotype model needs thresholds for every mode (missing: {missing}) and no "
                    "promote thresholds"
                )
        elif self.promote_thresholds is None or self.thresholds:
            raise ValueError("A promote model needs promote thresholds and no genotype thresholds")

    def to_json(self) -> str:
        """
        Serialize to JSON.

        Returns:
            The sidecar JSON text.
        """
        raw = asdict(self)
        raw["variant_type"] = self.variant_type.value
        raw["task"] = self.task.value
        raw["thresholds"] = {mode.value: asdict(t) for mode, t in self.thresholds.items()}
        raw["stutter"] = None if self.stutter is None else self.stutter.to_dict()
        return json.dumps(raw, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ModelInfo":
        """
        Deserialize from JSON.

        Args:
            text: the sidecar JSON text (see `to_json`).

        Returns:
            The sidecar.

        Raises:
            ValueError: if the sidecar format version is not supported (e.g. it was written by a
                newer winnowr) or the thresholds do not match the task.
        """
        raw = json.loads(text)
        version = raw.get("format_version")
        if not isinstance(version, int) or not 1 <= version <= SIDECAR_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported sidecar format version {version} (written by winnowr "
                f"{raw.get('winnowr_version', 'unknown')}); winnowr {__version__} reads versions "
                f"1 to {SIDECAR_FORMAT_VERSION}"
            )
        return cls(
            variant_type=VariantType(raw["variant_type"]),
            features=tuple(raw["features"]),
            region_names=tuple(raw["region_names"]),
            thresholds={Mode(m): Thresholds(**t) for m, t in raw["thresholds"].items()},
            training_samples=tuple(raw["training_samples"]),
            contigs=tuple(raw["contigs"]),
            task=Task(raw["task"]),
            promote_thresholds=(
                None
                if raw["promote_thresholds"] is None
                else PromoteThresholds(**raw["promote_thresholds"])
            ),
            params=TrainingParams(**raw["params"]),
            stutter=(
                None if raw.get("stutter") is None else StutterModel.from_dict(raw["stutter"])
            ),
            regenotype=raw.get("regenotype", False),
            winnowr_version=raw["winnowr_version"],
            xgboost_version=raw["xgboost_version"],
            format_version=version,
            reference_contigs=raw.get("reference_contigs"),
        )


@dataclass(frozen=True)
class WinnowrModel:
    """
    A trained booster with its sidecar.

    Attributes:
        booster: the XGBoost booster.
        info: the sidecar.
    """

    booster: xgb.Booster
    info: ModelInfo

    @property
    def num_classes(self) -> int:
        """The number of classes the model predicts (see `num_classes`)."""
        return num_classes(self.info.task, self.info.regenotype)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        """
        Predict class probabilities for a feature table.

        Args:
            frame: the calls to score (with the model's feature columns; for a model with a
                stutter model, see `winnowr.samples.SampleCalls.from_table` or
                `winnowr.stutter.add_length_call_features`).

        Returns:
            An `(n, 3)` array in `Label` (genotype) or `PromoteLabel` (promote) order, or `(n, 4)`
            for re-genotype models.
        """
        return predict_probabilities(self.booster, frame, self.info.features, self.num_classes)

    def save(self, prefix: Path) -> Path:
        """
        Write `<prefix>.xgb.json` and `<prefix>.winnowr.json`.

        Args:
            prefix: the output path prefix.

        Returns:
            The sidecar path.
        """
        self.booster.save_model(str(prefix) + MODEL_SUFFIX)
        sidecar = Path(str(prefix) + SIDECAR_SUFFIX)
        sidecar.write_text(self.info.to_json())
        return sidecar

    @classmethod
    def load(cls, sidecar: Path) -> "WinnowrModel":
        """
        Load a model from its sidecar path (the booster is read from beside it).

        Args:
            sidecar: the `<prefix>.winnowr.json` path.

        Returns:
            The model.

        Logs a warning when the model was trained by another winnowr version, or its sidecar
        records no reference contigs (format 1), so the reference cannot be checked.

        Raises:
            ValueError: if the path does not end in `.winnowr.json`, or the sidecar format is not
                supported.
        """
        name = str(sidecar)
        if not name.endswith(SIDECAR_SUFFIX):
            raise ValueError(f"Model sidecar must end in {SIDECAR_SUFFIX}: {sidecar}")
        info = ModelInfo.from_json(sidecar.read_text())
        booster = xgb.Booster()
        booster.load_model(name.removesuffix(SIDECAR_SUFFIX) + MODEL_SUFFIX)
        if info.winnowr_version != __version__:
            logger.warning(
                f"{sidecar} was trained with winnowr {info.winnowr_version}; this is winnowr "
                f"{__version__}"
            )
        if info.reference_contigs is None:
            logger.warning(
                f"{sidecar} (sidecar format {info.format_version}) records no reference contigs, "
                "so the reference of the calls cannot be checked against it"
            )
        return cls(booster=booster, info=info)


def check_reference(
    model: Path,
    info: ModelInfo,
    contigs: Mapping[str, int | None],
    source: str,
    require_all: bool = True,
) -> None:
    """
    Check that calls come from the reference a model was trained on (its contig names and lengths).

    A model without `reference_contigs` (an old sidecar) is not checked; lengths are compared only
    where both are known.

    Args:
        model: the model sidecar (for the message).
        info: the model's sidecar.
        contigs: the contig lengths of the calls' reference (e.g. from their VCF header).
        source: where `contigs` came from (for the message).
        require_all: whether every model contig must be in `contigs`.

    Raises:
        ValueError: if a model contig has another length in `contigs`, or (with `require_all`) is
            missing from it.
    """
    if info.reference_contigs is None:
        return
    missing = [name for name in info.reference_contigs if require_all and name not in contigs]
    differ = [
        f"{name} ({contigs[name]:,} bp, model {length:,} bp)"
        for name, length in info.reference_contigs.items()
        if name in contigs
        and length is not None
        and contigs[name] is not None
        and contigs[name] != length
    ]
    if missing or differ:
        raise ValueError(
            f"{source} is not on the reference model {model} was trained on: "
            + "; ".join(
                ([f"missing contigs {missing}"] if missing else [])
                + ([f"other lengths: {', '.join(differ)}"] if differ else [])
            )
        )
