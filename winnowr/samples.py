"""The labelled calls of one sample, as used for training and evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from winnowr.common import FloatArray
from winnowr.common import VariantType
from winnowr.labels import Label
from winnowr.labels import labelled_calls
from winnowr.promote_labels import promote_calls
from winnowr.regenotype import with_regenotype_columns
from winnowr.regenotype import without_regenotype
from winnowr.rescore import Counts
from winnowr.rescore import baseline_counts
from winnowr.stutter import StutterModel
from winnowr.stutter import add_length_call_features
from winnowr.table import FeatureTableInfo
from winnowr.thresholds import FoldPredictions
from winnowr.thresholds import PromoteFold


@dataclass(frozen=True)
class SampleCalls:
    """
    The labelled calls of one sample and one variant type.

    Attributes:
        sample: the sample name.
        calls: the labelled PASS calls (see `labelled_calls`).
        baseline: the sample's TP/FP/FN before any action.
    """

    sample: str
    calls: pd.DataFrame
    baseline: Counts

    @classmethod
    def from_table(
        cls,
        frame: pd.DataFrame,
        info: FeatureTableInfo,
        variant_type: VariantType,
        stutter: StutterModel | None = None,
        regenotype: bool = False,
    ) -> "SampleCalls":
        """
        Select the labelled calls of a variant type from a labelled feature table.

        The calls get the re-genotype columns (`REGENOTYPE_COLUMNS`); without a stutter model they
        are all false. With one the calls also get the length caller's features and genotype.

        Args:
            frame: the feature table.
            info: its metadata.
            variant_type: the variant type to select.
            stutter: the stutter model for the length-call features, if the model uses them.
            regenotype: relabel as `Label.REGENOTYPE` the FPs (`FLIP` or `DROP`) that a flip does
                not fix but re-genotyping to the length caller's genotype does (needs `stutter`).

        Returns:
            The sample's labelled calls and baseline counts.

        Raises:
            ValueError: if the table has no hap.py labels, or length calls are requested from a
                table without strlen evidence.
        """
        if not info.is_labelled:
            raise ValueError(
                f"Feature table for {info.sample} has no hap.py labels (extract with --happy)"
            )
        if regenotype and stutter is None:
            raise ValueError("Re-genotype labels need a stutter model")
        calls = without_regenotype(labelled_calls(frame, variant_type))
        if stutter is not None:
            if not info.strlen:
                raise ValueError(
                    f"Feature table for {info.sample} has no strlen evidence "
                    "(extract with --strlen)"
                )
            calls = with_regenotype_columns(add_length_call_features(calls, stutter))
            if regenotype:
                fixed_by_flip = (calls["label"] == Label.FLIP) & calls["flip_ok"]
                relabel = calls["regenotype_ok"] & ~fixed_by_flip
                calls.loc[relabel, "label"] = float(Label.REGENOTYPE)
        baseline = baseline_counts(calls["label"], info.fn_count(variant_type))
        return cls(sample=info.sample, calls=calls, baseline=baseline)

    def to_fold(self, probabilities: FloatArray) -> FoldPredictions:
        """
        Package predictions for these calls as a cross-validation fold.

        Args:
            probabilities: the `(n, 3)` (or `(n, 4)`) class probabilities of the calls, in `Label`
                order.

        Returns:
            The fold.
        """
        return FoldPredictions(
            probabilities=probabilities,
            labels=self.calls["label"].to_numpy(dtype=np.int64),
            flip_ok=self.calls["flip_ok"].to_numpy(dtype=np.bool_),
            flippable=self.calls["flippable"].to_numpy(dtype=np.bool_),
            baseline=self.baseline,
            regenotypable=self.calls["regenotypable"].to_numpy(dtype=np.bool_),
            regenotypes_to_ref=self.calls["regenotypes_to_ref"].to_numpy(dtype=np.bool_),
            regenotype_ok=self.calls["regenotype_ok"].to_numpy(dtype=np.bool_),
        )


@dataclass(frozen=True)
class PromoteSampleCalls:
    """
    The promotion candidates of one sample and one variant type.

    Attributes:
        sample: the sample name.
        calls: the labelled non-PASS candidates (see `promote_calls`).
        baseline: the sample's PASS-call TP/FP/FN, which promotions add to.
    """

    sample: str
    calls: pd.DataFrame
    baseline: Counts

    @classmethod
    def from_table(
        cls, frame: pd.DataFrame, info: FeatureTableInfo, variant_type: VariantType
    ) -> "PromoteSampleCalls":
        """
        Select the promotion candidates of a variant type from a promote-labelled table.

        Args:
            frame: the feature table.
            info: its metadata.
            variant_type: the variant type to select.

        Returns:
            The sample's labelled candidates and PASS-call baseline counts.

        Raises:
            ValueError: if the table has no promote labels.
        """
        if not info.promote_labelled:
            raise ValueError(
                f"Feature table for {info.sample} has no promote labels (extract with --truth-vcf)"
            )
        baseline = SampleCalls.from_table(frame, info, variant_type).baseline
        return cls(sample=info.sample, calls=promote_calls(frame, variant_type), baseline=baseline)

    def to_fold(self, probabilities: FloatArray) -> PromoteFold:
        """
        Package predictions for these candidates as a cross-validation fold.

        Args:
            probabilities: the `(n, 3)` class probabilities of the candidates, in `PromoteLabel`
                order.

        Returns:
            The fold.
        """
        return PromoteFold(
            probabilities=probabilities,
            labels=self.calls["promote_label"].to_numpy(dtype=np.int64),
            flippable=self.calls["flippable"].to_numpy(dtype=np.bool_),
            baseline=self.baseline,
        )
