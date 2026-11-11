import pytest

from winnowr.alleles import normalize_alleles

#         1-based: 123456789
REFERENCE = "GCACACATG"


def _fetch(pos: int) -> str:
    return REFERENCE[pos - 1]


@pytest.mark.parametrize(
    "pos, alleles, expected",
    [
        (1, ("G", "T"), (1, ("G", "T"))),  # already normal SNP
        (3, ("ACAGA", "CCAGA"), (3, ("A", "C"))),  # padded SNP
        (5, ("ACA", "A"), (1, ("GCA", "G"))),  # deletion of CA shifts left through the repeat
        (6, ("CAT", "CT"), (6, ("CA", "C"))),  # an A deletion here has no equivalent to the left
        (7, ("A", "ACA"), (1, ("G", "GCA"))),  # insertion shifts left
        (2, ("CAC", "C", "CACAC"), (1, ("GCA", "G", "GCACA"))),  # multi-allelic shifts jointly
        (1, ("G", "<DEL>"), (1, ("G", "<DEL>"))),  # symbolic: unchanged
        (3, ("ACACA", "ACACG"), (7, ("A", "G"))),  # leading-padded SNP: leading bases trimmed
        (1, ("GG", "G"), (1, ("GG", "G"))),  # deletion at position 1 cannot shift further left
        (2, ("CA", "CACA"), (1, ("G", "GCA"))),  # shifts left to position 1, keeping its anchor
        (4, ("A", "a"), (4, ("A", "a"))),  # identical alleles (ignoring case): unchanged
        (4, ("CA", "CA", "CA"), (4, ("CA", "CA", "CA"))),
    ],
)
def test_normalize_with_reference(
    pos: int, alleles: tuple[str, ...], expected: tuple[int, tuple[str, ...]]
) -> None:
    assert normalize_alleles(pos, alleles, _fetch) == expected


def test_normalize_without_reference_only_trims() -> None:
    assert normalize_alleles(3, ("ACAGA", "CCAGA")) == (3, ("A", "C"))
    assert normalize_alleles(6, ("CAT", "CT")) == (6, ("CA", "C"))
    assert normalize_alleles(1, ("AA", "A")) == (1, ("AA", "A"))
