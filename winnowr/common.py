"""Basic types and constants shared across winnowr modules."""

from enum import Enum
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

AUTOSOMES: Final[tuple[str, ...]] = tuple(f"chr{i}" for i in range(1, 23))
"""Default contigs to extract (GRCh38 autosomes)."""

NEVER: Final[float] = 1.01
"""A threshold no probability reaches (the action is disabled)."""

PASS: Final[str] = "PASS"
"""The FILTER value of a passing call (a missing FILTER is read as PASS)."""

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class Task(Enum):
    """
    What a model decides.

    Attributes:
        GENOTYPE: for PASS calls: drop (0/0), keep, or flip het<->hom.
        PROMOTE: for non-PASS calls: leave filtered, promote to PASS, or promote and flip.
    """

    GENOTYPE = "genotype"
    PROMOTE = "promote"


class VariantType(StrEnum):
    """The variant type a model is trained for."""

    SNP = "snp"
    INDEL = "indel"
