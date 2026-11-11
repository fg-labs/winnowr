"""
Apply trained models to a sample's calls and write the re-genotyped, filtered VCF.

Genotype models score PASS calls of their variant type; promote models score non-PASS calls of
their variant type that do not overlap a PASS call. Only calls on the models' contigs are scored;
all other records are written unchanged. For scored records:

- INFO `WINNOWR_P` holds a genotype model's probabilities `p_drop,p_keep,p_flip`, or INFO
  `WINNOWR_PROMOTE_P` a promote model's `p_stay,p_promote,p_promote_flip`.
- A dropped call gets FILTER `WinnowrDrop` (replacing PASS).
- A promoted call gets FILTER PASS; its original FILTER is kept in INFO `WINNOWR_OFILTER`.
- A flipped call has its GT switched between `0/1` and `1/1` (unphased), INFO flag `WINNOWR_FLIP`,
  FORMAT `WINNOWR_OGT` with the original genotype, and GQ set to the phred-scaled model confidence
  in the new genotype, `-10 log10(1 - p_flip)` capped at 99. INFO AC/AF and MLEAC/MLEAF, when
  present, are recomputed from the new genotype so the record stays self-consistent (in a
  single-sample VCF MLEAC/MLEAF then equal AC/AF), and the physical-phasing FORMAT fields PGT, PID
  and PS, when present, are set to missing. PL is left as emitted by the caller, so it reflects the
  caller's likelihoods, not winnowr's genotype.
- Re-genotype models (see `winnowr.stutter`) also write the re-genotype probability to INFO
  `WINNOWR_REGENOTYPE_P`. A re-genotyped call gets the repeat-length caller's genotype, INFO flag
  `WINNOWR_REGENOTYPE`, FORMAT `WINNOWR_OGT` and GQ `-10 log10(1 - p_regenotype)`. Each new
  allele is the repeat tract resized to its length in the tract's own phase: an ALT with exactly
  that sequence is reused, otherwise it is added as a new ALT (the REF is extended to the end of
  the tract when needed, with the same bases appended to every allele; see
  `winnowr.regenotype.regenotyped_alleles`). Number=A/R/G fields get missing values for the new
  alleles. Calls whose new alleles cannot be built (e.g. no anchor base before the tract) are not
  re-genotyped (see `winnowr.regenotype.regenotype_buildable`).
"""

import logging
import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import pysam

from winnowr.alleles import FLIPPED_GT
from winnowr.alleles import normalize_gt
from winnowr.common import PASS
from winnowr.common import Task
from winnowr.labels import Label
from winnowr.model import WinnowrModel
from winnowr.promote_labels import promote_eligible
from winnowr.regenotype import regenotype_target
from winnowr.regenotype import with_regenotype_columns
from winnowr.regenotype import without_regenotype
from winnowr.rescore import Mode
from winnowr.rescore import decide
from winnowr.rescore import decide_promote
from winnowr.rescore import flip_thresholds
from winnowr.rescore import has_repeat_info
from winnowr.stutter import add_length_call_features

logger = logging.getLogger(__name__)

DROP_FILTER: Final[str] = "WinnowrDrop"
PROBABILITIES_INFO: Final[str] = "WINNOWR_P"
PROMOTE_PROBABILITIES_INFO: Final[str] = "WINNOWR_PROMOTE_P"
ORIGINAL_FILTER_INFO: Final[str] = "WINNOWR_OFILTER"
FLIP_INFO: Final[str] = "WINNOWR_FLIP"
REGENOTYPE_INFO: Final[str] = "WINNOWR_REGENOTYPE"
REGENOTYPE_PROBABILITY_INFO: Final[str] = "WINNOWR_REGENOTYPE_P"
ORIGINAL_GT_FORMAT: Final[str] = "WINNOWR_OGT"
MAX_GQ: Final[int] = 99

Key = tuple[str, int, str, str, str]
"""A call's `(chrom, pos, ref, alt, filter)`."""


@dataclass(frozen=True)
class Decision:
    """
    A model's decision for one call.

    Attributes:
        task: the task of the model that scored the call.
        probabilities: the model's class probabilities (`Label` or `PromoteLabel` order; four for
            re-genotype models).
        drop: drop the call.
        promote: promote the call to PASS.
        flip: switch the call's genotype between `0/1` and `1/1`.
        regenotype: the re-genotyped call's alleles and genotype (see
            `winnowr.regenotype.regenotyped_alleles`), if it is re-genotyped.
    """

    task: Task
    probabilities: tuple[float, ...]
    drop: bool = False
    promote: bool = False
    flip: bool = False
    regenotype: tuple[tuple[str, ...], tuple[int, int]] | None = None


def _scored_calls(table: pd.DataFrame, model: WinnowrModel) -> pd.DataFrame:
    """The calls a model scores: PASS calls (genotype) or `promote_eligible` calls (promote)."""
    of_type = (table["variant_type"] == model.info.variant_type.value).to_numpy(dtype=np.bool_)
    if model.info.task is Task.PROMOTE:
        return table[of_type & promote_eligible(table)]
    return table[of_type & (table["filter"] == PASS).to_numpy(dtype=np.bool_)]


def decide_calls(
    table: pd.DataFrame, models: Sequence[WinnowrModel], mode: Mode
) -> dict[Key, Decision]:
    """
    Score each model's calls and decide drops, promotions, flips and re-genotypes.

    Args:
        table: a feature table (see `winnowr.extraction.build_feature_table`; with strlen
            evidence for models with a stutter model).
        models: at most one genotype and one promote model per variant type.
        mode: the decision mode for genotype models (their CV-chosen thresholds for it are used).

    Returns:
        The decision for each scored call, in table order (the first of calls sharing a key).

    Raises:
        ValueError: if two models share a task and variant type, or a model with a stutter model
            is given a table without strlen evidence.
    """
    kinds = [(m.info.task, m.info.variant_type) for m in models]
    if len(set(kinds)) != len(kinds):
        raise ValueError(f"Expected at most one model per task and variant type, got {kinds}")
    decisions: dict[Key, Decision] = {}
    for model in models:
        for key, decision in _decide_model_calls(table, model, mode).items():
            decisions.setdefault(key, decision)
    return decisions


def _decide_model_calls(
    table: pd.DataFrame, model: WinnowrModel, mode: Mode
) -> dict[Key, Decision]:
    """The `decide_calls()` decisions of one model."""
    calls = _scored_calls(table, model)
    if model.info.stutter is None:
        calls = without_regenotype(calls)
    else:
        calls = with_regenotype_columns(add_length_call_features(calls, model.info.stutter))
    probabilities = model.predict(calls)
    no_action = np.zeros(len(calls), dtype=np.bool_)
    regenotype = no_action
    if model.info.task is Task.PROMOTE:
        chosen = model.info.promote_thresholds
        assert chosen is not None  # ModelInfo guarantees promote thresholds for promote models
        promote, flip = decide_promote(
            probabilities, calls["flippable"], chosen.t_promote, chosen.t_flip
        )
        drop = no_action
    else:
        thresholds = model.info.thresholds[mode]
        period, units = calls.get("str_period"), calls.get("str_repeats")
        if thresholds.repeat_flip is not None:
            described = f"The {model.info.variant_type} {model.info.task.value} model"
            if period is None or units is None:
                raise ValueError(
                    f"{described} has a long-tandem-repeat flip threshold, which needs the calls' "
                    "str_period and str_repeats columns; the feature table lacks them"
                )
            if not has_repeat_info(period, units):
                logger.warning(
                    f"{described} has a long-tandem-repeat flip threshold, but no call has a "
                    "DRAGstr repeat period and count (DRAGstrInfo), so it applies t_flip to every "
                    "call"
                )
        t_flip = flip_thresholds(thresholds.t_flip, thresholds.repeat_flip, period, units)
        drop, flip, regenotype = decide(
            probabilities,
            calls["flippable"],
            thresholds.t_drop,
            t_flip,
            mode,
            thresholds.t_regenotype,
            calls["regenotypable"],
            calls["regenotypes_to_ref"],
        )
        promote = no_action
    targets = _regenotype_targets(calls, regenotype)
    keys = zip(
        calls["chrom"], calls["pos"], calls["ref"], calls["alt"], calls["filter"], strict=True
    )
    decisions: dict[Key, Decision] = {}
    for row, key in enumerate(keys):
        decisions.setdefault(
            key,
            Decision(
                task=model.info.task,
                probabilities=tuple(float(p) for p in probabilities[row]),
                drop=bool(drop[row]),
                promote=bool(promote[row]),
                flip=bool(flip[row]),
                regenotype=targets.get(row),
            ),
        )
    logger.info(
        f"{model.info.task.value} {model.info.variant_type}: scored {len(calls):,} calls, "
        f"drop {int(drop.sum()):,}, promote {int(promote.sum()):,}, flip {int(flip.sum()):,}, "
        f"re-genotype {len(targets):,}"
    )
    return decisions


def _regenotype_targets(
    calls: pd.DataFrame, regenotype: npt.NDArray[np.bool_]
) -> dict[int, tuple[tuple[str, ...], tuple[int, int]]]:
    """The new alleles and genotype of each re-genotyped call, by row (unbuildable ones skipped)."""
    targets: dict[int, tuple[tuple[str, ...], tuple[int, int]]] = {}
    rows = np.flatnonzero(regenotype)
    if len(rows) == 0:
        return targets
    subset = calls.iloc[rows]
    for row, pos, ref, alt, lo, hi, tract_start, tract_len, unit in zip(
        rows.tolist(),
        subset["pos"],
        subset["ref"],
        subset["alt"],
        subset["lc_delta_lo"],
        subset["lc_delta_hi"],
        subset["sl_tract_start"],
        subset["sl_tract_len"],
        subset["sl_unit"],
        strict=True,
    ):
        target = regenotype_target(pos, ref, alt, (lo, hi), tract_start, tract_len, unit)
        if target is not None:
            targets[row] = target
    if len(targets) < len(rows):
        logger.warning(
            f"Skipped {len(rows) - len(targets):,} re-genotypes whose new allele cannot be built"
        )
    return targets


def flip_gq(p_flip: float) -> int:
    """
    The GQ of a flipped genotype: phred-scaled `1 - p_flip`, capped at `MAX_GQ`.

    Args:
        p_flip: the model's probability of the flipped genotype.

    Returns:
        The GQ.
    """
    error = 1.0 - p_flip
    if error <= 0:
        return MAX_GQ
    return min(MAX_GQ, round(-10 * math.log10(error)))


def _add_header_lines(header: pysam.VariantHeader) -> None:
    """Declare winnowr's FILTER, INFO and FORMAT fields."""
    header.add_line(f'##FILTER=<ID={DROP_FILTER},Description="winnowr: call is likely 0/0">')
    header.add_line(
        f"##INFO=<ID={PROBABILITIES_INFO},Number=3,Type=Float,"
        'Description="winnowr class probabilities: drop (0/0), keep, flip het<->hom">'
    )
    header.add_line(
        f"##INFO=<ID={PROMOTE_PROBABILITIES_INFO},Number=3,Type=Float,"
        'Description="winnowr promote probabilities: stay filtered, promote, promote and flip">'
    )
    header.add_line(
        f"##INFO=<ID={ORIGINAL_FILTER_INFO},Number=.,Type=String,"
        'Description="FILTER before winnowr promoted the call to PASS">'
    )
    header.add_line(
        f'##INFO=<ID={FLIP_INFO},Number=0,Type=Flag,Description="winnowr flipped het<->hom">'
    )
    header.add_line(
        f"##FORMAT=<ID={ORIGINAL_GT_FORMAT},Number=1,Type=String,"
        'Description="Genotype before winnowr flipped or re-genotyped it">'
    )
    header.add_line(
        f"##INFO=<ID={REGENOTYPE_PROBABILITY_INFO},Number=1,Type=Float,"
        'Description="winnowr probability that the repeat-length genotype is right">'
    )
    header.add_line(
        f"##INFO=<ID={REGENOTYPE_INFO},Number=0,Type=Flag,"
        'Description="winnowr set the genotype to the repeat-length caller\'s">'
    )


def write_scored_vcf(vcf: Path, output: Path, decisions: Mapping[Key, Decision]) -> None:
    """
    Write a copy of the VCF with the decisions applied (see the module docstring).

    Logs the number of drops, promotions and flips applied. A flip is counted only when the
    record's genotype could be flipped (it is `0/1` or `1/1`); other flips are skipped with a
    warning. A decision applies to the first record with its `(chrom, pos, ref, alt, filter)`;
    later records with the same key are written unchanged, with a warning.

    Args:
        vcf: the input VCF.
        output: the output path: `.bcf` writes indexed BCF, `.gz` bgzipped and tabix-indexed VCF,
            anything else plain-text VCF.
        decisions: the decisions by call (see `decide_calls()`).
    """
    by_key = dict(decisions)
    name = str(output)
    write_mode = "wb" if name.endswith(".bcf") else "wz" if name.endswith(".gz") else "w"
    applied = dict.fromkeys(("drop", "promote", "flip", "skipped flip", "regenotype"), 0)
    n_duplicates = 0
    applied_keys: set[Key] = set()
    with pysam.VariantFile(str(vcf)) as reader:
        header = reader.header.copy()
        _add_header_lines(header)
        with pysam.VariantFile(name, write_mode, header=header) as writer:
            for rec in reader:
                rec.translate(header)
                key = (rec.chrom, rec.pos, rec.ref or "", ",".join(rec.alts or ()), _filter(rec))
                decision = by_key.pop(key, None)
                if decision is None:
                    n_duplicates += key in applied_keys
                else:
                    applied_keys.add(key)
                    _apply(rec, decision)
                    applied["drop"] += decision.drop
                    applied["promote"] += decision.promote
                    flipped = FLIP_INFO in rec.info
                    applied["flip"] += flipped
                    applied["skipped flip"] += decision.flip and not decision.drop and not flipped
                    applied["regenotype"] += REGENOTYPE_INFO in rec.info
                writer.write(rec)
    if n_duplicates:
        logger.warning(
            f"Left {n_duplicates:,} records unchanged that repeat an earlier record's "
            "CHROM, POS, REF, ALT and FILTER (only the first was scored)"
        )
    if write_mode != "w":
        pysam.tabix_index(name, preset="vcf", force=True, csi=write_mode == "wb")
    if applied["skipped flip"]:
        logger.warning(f"Skipped {applied['skipped flip']:,} flips of calls not 0/1 or 1/1")
    logger.info(
        f"Wrote {output}: {applied['drop']:,} calls dropped, {applied['promote']:,} promoted, "
        f"{applied['flip']:,} flipped, {applied['regenotype']:,} re-genotyped"
    )


def _apply(rec: pysam.VariantRecord, decision: Decision) -> None:
    """Annotate one record in place with its decision."""
    if decision.task is Task.PROMOTE:
        rec.info[PROMOTE_PROBABILITIES_INFO] = decision.probabilities
    else:
        rec.info[PROBABILITIES_INFO] = decision.probabilities[:3]
        if len(decision.probabilities) > 3:
            rec.info[REGENOTYPE_PROBABILITY_INFO] = decision.probabilities[3]
    if decision.drop:
        rec.filter.clear()
        rec.filter.add(DROP_FILTER)
        return
    if decision.promote:
        rec.info[ORIGINAL_FILTER_INFO] = tuple(str(f) for f in rec.filter.keys())
        rec.filter.clear()
        rec.filter.add(PASS)
    if decision.flip:
        _flip(rec, decision.probabilities[2])
    elif decision.regenotype is not None:
        _regenotype(rec, *decision.regenotype, decision.probabilities[Label.REGENOTYPE])


def _filter(rec: pysam.VariantRecord) -> str:
    """The record's FILTER as the feature table holds it (`;`-joined, PASS when missing)."""
    filters = [str(name) for name in rec.filter.keys()]
    return ";".join(filters) if filters else PASS


def _flip(rec: pysam.VariantRecord, p_flip: float) -> None:
    """Switch the record's genotype between 0/1 and 1/1 (other genotypes are left unchanged)."""
    original = normalize_gt(rec.samples[0].get("GT"))
    flipped = FLIPPED_GT.get(original)
    if flipped is None:
        return
    first, second = (int(a) for a in flipped.split("/"))
    _set_genotype(rec, (first, second), original, p_flip, FLIP_INFO)


def _regenotype(
    rec: pysam.VariantRecord, alleles: tuple[str, ...], gt: tuple[int, int], p: float
) -> None:
    """Set the record's alleles (when extended) and genotype to a re-genotype's."""
    original = normalize_gt(rec.samples[0].get("GT"))
    if alleles != tuple(rec.alleles or ()):
        _set_alleles(rec, alleles)
    _set_genotype(rec, gt, original, p, REGENOTYPE_INFO)


def _set_genotype(
    rec: pysam.VariantRecord, gt: tuple[int, int], original: str, p: float, flag: str
) -> None:
    """
    Write a new, unphased genotype with its model GQ, and keep the record self-consistent.

    Args:
        rec: the record.
        gt: the new genotype's allele indices.
        original: the normalized genotype before the change (for `ORIGINAL_GT_FORMAT`).
        p: the model's probability of the new genotype (for GQ).
        flag: the INFO flag marking the change (`FLIP_INFO` or `REGENOTYPE_INFO`).
    """
    sample = rec.samples[0]
    sample["GT"] = gt
    sample.phased = False
    sample[ORIGINAL_GT_FORMAT] = original
    if "GQ" in rec.header.formats:
        sample["GQ"] = flip_gq(p)
    _clear_phasing(rec)
    _update_allele_counts(rec, gt)
    rec.info[flag] = True


PHASING_FORMATS: Final[tuple[str, ...]] = ("PGT", "PID", "PS")
"""HaplotypeCaller's physical-phasing FORMAT fields, cleared when winnowr changes a genotype."""


def _clear_phasing(rec: pysam.VariantRecord) -> None:
    """
    Set the record's physical-phasing FORMAT fields to missing (`.`), since GT is now unphased.

    String fields are given an explicit `.`: pysam writes an empty value for None there.
    """
    sample = rec.samples[0]
    present = set(rec.format.keys())
    for key in PHASING_FORMATS:
        if key in present:
            sample[key] = "." if rec.header.formats[key].type == "String" else None


def _per_allele_length(number: object, n_alleles: int) -> int | None:
    """The value count of a Number=A/R/G field for `n_alleles` alleles (None for other Numbers)."""
    if number == "A":
        return n_alleles - 1
    if number == "R":
        return n_alleles
    if number == "G":
        return n_alleles * (n_alleles + 1) // 2
    return None


def _set_alleles(rec: pysam.VariantRecord, alleles: tuple[str, ...]) -> None:
    """
    Replace the record's alleles with an extended set, padding Number=A/R/G fields.

    The existing alleles keep their order and any new ones are appended, so existing per-allele
    and (diploid) per-genotype values keep their positions and the new entries are missing.
    """
    sample = rec.samples[0]
    info = {key: rec.info[key] for key in rec.info}
    formats = {key: sample[key] for key in sample if key != "GT"}
    rec.alleles = alleles
    sample = rec.samples[0]
    for key, value in info.items():
        size = _per_allele_length(rec.header.info[key].number, len(alleles))
        if size is not None and isinstance(value, tuple):
            rec.info[key] = (*value, *([None] * (size - len(value))))
    for key, value in formats.items():
        size = _per_allele_length(rec.header.formats[key].number, len(alleles))
        if size is not None and isinstance(value, tuple):
            sample[key] = (*value, *([None] * (size - len(value))))


_ALLELE_COUNT_INFO: Final[tuple[tuple[str, str], ...]] = (("AC", "AF"), ("MLEAC", "MLEAF"))
"""INFO allele count and frequency pairs recomputed after a flip."""


def _update_allele_counts(rec: pysam.VariantRecord, gt: Sequence[int]) -> None:
    """Recompute the record's INFO AC/AF and MLEAC/MLEAF (when present) from its sample's GT."""
    n_alts = len(rec.alts or ())
    counts = tuple(sum(allele == index for allele in gt) for index in range(1, n_alts + 1))
    for count_key, frequency_key in _ALLELE_COUNT_INFO:
        if count_key in rec.header.info and count_key in rec.info:
            rec.info[count_key] = counts
        if frequency_key in rec.header.info and frequency_key in rec.info:
            rec.info[frequency_key] = tuple(count / len(gt) for count in counts)
