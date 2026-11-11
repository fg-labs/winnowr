"""
Pure helpers for alleles and genotypes.

Variant typing, genotype normalization, allele length changes, and trimming and left-alignment
(so calls and truth records can be matched exactly).
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Final

from winnowr.common import VariantType

BaseFetcher = Callable[[int], str]
"""Returns the reference base at a 1-based position of the record's contig."""


def normalize_alleles(
    pos: int, alleles: Sequence[str], fetch_base: BaseFetcher | None = None
) -> tuple[int, tuple[str, ...]]:
    """
    Left-align and trim a record's alleles jointly (as `bcftools norm` without splitting).

    Shared trailing bases are trimmed; when an allele becomes empty and `fetch_base` is given the
    alleles are extended one reference base to the left, which shifts indels left through repeats.
    Finally shared leading bases are trimmed while every allele keeps at least one base. Without
    `fetch_base`, or at position 1 where there is no base to the left, the alleles are only trimmed
    and keep at least one base each. Records with symbolic or `*` alleles, or whose alleles are all
    the same (ignoring case), are returned unchanged.

    Args:
        pos: the 1-based position of the first base of the alleles.
        alleles: the reference allele followed by the alternate alleles.
        fetch_base: the reference base at a 1-based position, or None.

    Returns:
        The normalized position and alleles.
    """
    current = [allele.upper() for allele in alleles]
    if (
        len(current) < 2
        or any(a.startswith("<") or a == "*" or not a for a in current)
        or all(a == current[0] for a in current)
    ):
        return pos, tuple(alleles)
    while True:
        can_extend = fetch_base is not None and pos > 1
        if all(a[-1] == current[0][-1] for a in current) and (
            can_extend or all(len(a) > 1 for a in current)
        ):
            current = [a[:-1] for a in current]
            if any(not a for a in current) and fetch_base is not None:
                pos -= 1
                base = fetch_base(pos).upper()
                current = [base + a for a in current]
            continue
        break
    while all(len(a) > 1 for a in current) and all(a[0] == current[0][0] for a in current):
        current = [a[1:] for a in current]
        pos += 1
    return pos, tuple(current)


FLIPPED_GT: Final[dict[str, str]] = {"0/1": "1/1", "1/1": "0/1"}
"""The het<->hom flip of a normalized genotype; only these genotypes can be flipped."""


def normalize_gt(alleles: Sequence[int | None] | None) -> str:
    """
    Normalize a genotype to unphased, numerically sorted form, e.g. `(1, 0)` -> `0/1`.

    Args:
        alleles: the allele indices (None for missing).

    Returns:
        The normalized genotype, `./.` when any allele is missing or the genotype is not diploid.
    """
    if alleles is None or len(alleles) != 2 or any(a is None for a in alleles):
        return "./."
    first, second = sorted(a for a in alleles if a is not None)
    return f"{first}/{second}"


def variant_type_of(ref: str, alts: Sequence[str]) -> VariantType | None:
    """
    Classify a record as SNP or indel.

    Spanning-deletion (`*`) alleles are ignored. A record is an indel when any alternate allele
    differs in length from the reference. Otherwise it is a SNP when every alternate differs from
    the reference at exactly one base, wherever that base is (HaplotypeCaller writes some SNPs with
    padding, e.g. `TTAGA>CTAGA`), and unclassified (None) when there is no usable alternate, a
    symbolic allele, or an MNP.

    Args:
        ref: the reference allele.
        alts: the alternate alleles.

    Returns:
        The variant type, or None.
    """
    usable = [alt for alt in alts if alt != "*"]
    if not usable or any(alt.startswith("<") or "[" in alt or "]" in alt for alt in usable):
        return None
    if any(len(alt) != len(ref) for alt in usable):
        return VariantType.INDEL
    if all(sum(r != a for r, a in zip(ref, alt, strict=True)) == 1 for alt in usable):
        return VariantType.SNP
    return None


def first_alt_index(alts: Sequence[str]) -> int | None:
    """
    The index of the first alternate allele that is not a spanning deletion (`*`).

    Args:
        alts: the alternate alleles.

    Returns:
        The 0-based index into `alts`, or None if every alternate is `*`.
    """
    return next((i for i, alt in enumerate(alts) if alt != "*"), None)


def allele_deltas(alleles: Sequence[str], gt: str) -> tuple[int, int] | None:
    """
    The sorted length changes (ALT minus REF, bp) of a normalized diploid genotype's alleles.

    Args:
        alleles: the record's REF and ALT alleles.
        gt: the normalized genotype (see `normalize_gt`), e.g. `0/1`.

    Returns:
        The two length changes, or None if the genotype is missing or an allele is unknown,
        symbolic or `*`.
    """
    if gt == "./." or not alleles:
        return None
    deltas = []
    for index in (int(a) for a in gt.split("/")):
        if index >= len(alleles) or alleles[index] == "*" or alleles[index].startswith("<"):
            return None
        deltas.append(len(alleles[index]) - len(alleles[0]))
    low, high = sorted(deltas)
    return low, high


def trim_alleles(ref: str, alt: str) -> tuple[int, str, str]:
    """
    Trim the shared suffix, then the shared prefix, of two alleles.

    Args:
        ref: the reference allele.
        alt: the alternate allele.

    Returns:
        The offset in `ref` of the first differing base, the deleted bases and the inserted bases.
    """
    r, a = ref, alt
    while r and a and r[-1] == a[-1]:
        r, a = r[:-1], a[:-1]
    i = 0
    while i < len(r) and i < len(a) and r[i] == a[i]:
        i += 1
    return i, r[i:], a[i:]
