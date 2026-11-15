//! Pure helpers for alleles and genotypes.
//!
//! Variant typing, genotype normalization, allele length changes, and trimming and left-alignment
//! (so that calls and truth records can be matched exactly). Alleles are byte strings, as read
//! from a VCF.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;

use anyhow::bail;

use crate::common::VariantType;

/// The spanning-deletion allele.
const SPANNING_DELETION: &[u8] = b"*";

/// True for an allele that is not a sequence of bases (other than `*`): a symbolic allele (e.g.
/// `<DEL>`), a breakend (e.g. `G[chr2:10[`, or the single breakends `.G` and `G.`), or a missing
/// ALT (`.`).
fn is_non_sequence(allele: &[u8]) -> bool {
    allele.first() == Some(&b'<')
        || allele.iter().any(|&b| b == b'[' || b == b']')
        || allele.first() == Some(&b'.')
        || allele.last() == Some(&b'.')
}

/// A record's normalized position and alleles; see [`normalize_alleles`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedAlleles {
    /// The 1-based position of the first base of the alleles.
    pub pos: u64,
    /// The reference allele followed by the alternate alleles.
    pub alleles: Vec<Vec<u8>>,
}

/// Left-aligns and trims a record's alleles jointly, as `bcftools norm` does without splitting.
///
/// Shared trailing bases are trimmed. When an allele becomes empty, all alleles are extended one
/// reference base to the left, which shifts indels left through repeats. Finally shared leading
/// bases are trimmed while every allele keeps at least one base. At position 1, where there is no
/// base to the left, the alleles are only trimmed and keep at least one base each. Bases are
/// compared ignoring case, and normalized alleles are uppercase.
///
/// Records with fewer than two alleles, an empty, `*`, symbolic, breakend or missing (`.`) allele,
/// or whose alleles are all the same (ignoring case) are returned unchanged, in their original
/// case.
///
/// # Arguments
///
/// * `pos` - the 1-based position of the first base of the alleles.
/// * `alleles` - the reference allele followed by the alternate alleles.
/// * `fetch_base` - returns the reference base at a 1-based position on the record's contig.
///
/// # Errors
///
/// Returns the first error from `fetch_base`.
pub fn normalize_alleles<S, F, E>(
    pos: u64,
    alleles: &[S],
    fetch_base: F,
) -> Result<NormalizedAlleles, E>
where
    S: AsRef<[u8]>,
    F: FnMut(u64) -> Result<u8, E>,
{
    normalize(pos, alleles, Some(fetch_base))
}

/// Trims a record's alleles jointly, without a reference to left-align against.
///
/// The same as [`normalize_alleles`], except that alleles are never extended to the left, so
/// every allele keeps at least one base and indels are not shifted through repeats.
#[must_use]
pub fn trim_shared_bases<S: AsRef<[u8]>>(pos: u64, alleles: &[S]) -> NormalizedAlleles {
    let no_reference: Option<fn(u64) -> Result<u8, Infallible>> = None;
    let Ok(normalized) = normalize(pos, alleles, no_reference);
    normalized
}

/// [`normalize_alleles`], left-aligning only when `fetch_base` is given.
fn normalize<S, F, E>(
    pos: u64,
    alleles: &[S],
    mut fetch_base: Option<F>,
) -> Result<NormalizedAlleles, E>
where
    S: AsRef<[u8]>,
    F: FnMut(u64) -> Result<u8, E>,
{
    let unnormalizable = alleles.len() < 2
        || alleles.iter().map(AsRef::as_ref).any(|allele| {
            allele.is_empty() || allele == SPANNING_DELETION || is_non_sequence(allele)
        })
        || alleles.iter().all(|allele| allele.as_ref().eq_ignore_ascii_case(alleles[0].as_ref()));
    if unnormalizable {
        let alleles = alleles.iter().map(|allele| allele.as_ref().to_vec()).collect();
        return Ok(NormalizedAlleles { pos, alleles });
    }

    // Deques, so that trimming a trailing base and prepending a reference base are both O(1).
    let mut current: Vec<VecDeque<u8>> = alleles
        .iter()
        .map(|allele| allele.as_ref().iter().map(u8::to_ascii_uppercase).collect())
        .collect();

    // Trim shared trailing bases, extending every allele one base to the left whenever one
    // becomes empty. Every allele is non-empty at the top of the loop: an allele is only emptied
    // when a reference base can be prepended. This terminates because the alleles differ, so
    // each extension moves left until they no longer share a last base, or position 1 is reached.
    let mut pos = pos;
    loop {
        let can_extend = fetch_base.is_some() && pos > 1;
        let last = current[0].back().copied();
        let shared_last = current.iter().all(|a| a.back().copied() == last);
        if !shared_last || !(can_extend || current.iter().all(|a| a.len() > 1)) {
            break;
        }
        for allele in &mut current {
            allele.pop_back();
        }
        if let Some(fetch) = fetch_base.as_mut().filter(|_| current.iter().any(VecDeque::is_empty))
        {
            pos -= 1;
            let base = fetch(pos)?.to_ascii_uppercase();
            for allele in &mut current {
                allele.push_front(base);
            }
        }
    }

    // Trim shared leading bases, keeping at least one base in every allele.
    let min_len = current.iter().map(VecDeque::len).min().unwrap_or(0);
    let shared_leading = (0..min_len.saturating_sub(1))
        .take_while(|&i| current.iter().all(|a| a[i] == current[0][i]))
        .count();
    let alleles = current
        .into_iter()
        .map(|mut allele| {
            allele.drain(..shared_leading);
            Vec::from(allele)
        })
        .collect();
    Ok(NormalizedAlleles { pos: pos + shared_leading as u64, alleles })
}

/// A diploid genotype in unphased, numerically sorted form (e.g. `0/1`), or missing (`./.`).
///
/// A genotype with a missing allele, or that is not diploid, is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Genotype(Option<(u32, u32)>);

impl Genotype {
    /// The missing genotype, `./.`.
    pub const MISSING: Genotype = Genotype(None);
    /// The heterozygous genotype `0/1`.
    pub const HET: Genotype = Genotype(Some((0, 1)));
    /// The homozygous-alternate genotype `1/1`.
    pub const HOM_ALT: Genotype = Genotype(Some((1, 1)));

    /// The genotype with the given two allele indices, in either order.
    #[must_use]
    pub fn diploid(first: u32, second: u32) -> Self {
        Genotype(Some((first.min(second), first.max(second))))
    }

    /// Normalizes a sample's allele indices (`None` for a missing allele), ignoring phasing.
    ///
    /// For example `[Some(1), Some(0)]` is `0/1`. Any missing allele, an allele index that is not
    /// one of the record's `n_alleles` alleles (as pysam reads it), or a number of alleles other
    /// than two, gives the missing genotype.
    #[must_use]
    pub fn from_alleles<I: IntoIterator<Item = Option<u32>>>(alleles: I, n_alleles: usize) -> Self {
        let mut alleles = alleles.into_iter();
        match (alleles.next(), alleles.next(), alleles.next()) {
            (Some(Some(first)), Some(Some(second)), None)
                if (first.max(second) as usize) < n_alleles =>
            {
                Genotype::diploid(first, second)
            }
            _ => Genotype::MISSING,
        }
    }

    /// The sorted allele indices, or `None` if the genotype is missing.
    #[must_use]
    pub fn alleles(self) -> Option<(u32, u32)> {
        self.0
    }

    /// True for the missing genotype.
    #[must_use]
    pub fn is_missing(self) -> bool {
        self.0.is_none()
    }

    /// The het <-> hom flip: `0/1` <-> `1/1`. Other genotypes cannot be flipped.
    #[must_use]
    pub fn flipped(self) -> Option<Genotype> {
        match self {
            Genotype::HET => Some(Genotype::HOM_ALT),
            Genotype::HOM_ALT => Some(Genotype::HET),
            _ => None,
        }
    }

    /// True if the genotype can be flipped (see [`Genotype::flipped`]).
    #[must_use]
    pub fn is_flippable(self) -> bool {
        self.flipped().is_some()
    }
}

impl fmt::Display for Genotype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some((low, high)) => write!(f, "{low}/{high}"),
            None => f.write_str("./."),
        }
    }
}

impl FromStr for Genotype {
    type Err = anyhow::Error;

    /// Parses a normalized genotype exactly as written by `Display`: `./.`, or `low/high` with
    /// `low <= high` in plain decimal (no sign, leading zeros or whitespace).
    fn from_str(s: &str) -> anyhow::Result<Self> {
        let parsed = match s.split_once('/') {
            _ if s == "./." => Some(Genotype::MISSING),
            Some((low, high)) => match (low.parse::<u32>(), high.parse::<u32>()) {
                (Ok(low), Ok(high)) if low <= high => Some(Genotype(Some((low, high)))),
                _ => None,
            },
            None => None,
        };
        // Reject anything Display would not write, e.g. `+0/1` or `00/1`.
        match parsed {
            Some(genotype) if genotype.to_string() == s => Ok(genotype),
            _ => bail!("invalid normalized genotype '{s}' (expected e.g. 0/1, 1/2 or ./.)"),
        }
    }
}

/// Classifies a record as a SNP or an indel.
///
/// Spanning-deletion (`*`) alleles are ignored. A record is an indel when any alternate allele
/// differs in length from the reference. Otherwise it is a SNP when every alternate differs from
/// the reference at exactly one base (ignoring case), wherever that base is (HaplotypeCaller
/// writes some SNPs with padding, e.g. `TTAGA>CTAGA`). It is unclassified (`None`) when there is
/// no alternate other than `*`, when an alternate is symbolic, a breakend or missing (`.`), or
/// when it is an MNP.
#[must_use]
pub fn variant_type_of<S: AsRef<[u8]>>(reference: &[u8], alts: &[S]) -> Option<VariantType> {
    let usable = || alts.iter().map(AsRef::as_ref).filter(|&alt| alt != SPANNING_DELETION);
    if usable().next().is_none() || usable().any(is_non_sequence) {
        return None;
    }
    if usable().any(|alt| alt.len() != reference.len()) {
        return Some(VariantType::Indel);
    }
    let differs_at_one_base = |alt: &[u8]| {
        reference.iter().zip(alt).filter(|(r, a)| !r.eq_ignore_ascii_case(a)).count() == 1
    };
    usable().all(differs_at_one_base).then_some(VariantType::Snp)
}

/// The index of the first alternate allele that is not a spanning deletion (`*`), or `None` if
/// every alternate is `*`.
#[must_use]
pub fn first_alt_index<S: AsRef<[u8]>>(alts: &[S]) -> Option<usize> {
    alts.iter().position(|alt| alt.as_ref() != SPANNING_DELETION)
}

/// The sorted length changes (ALT minus REF, in bp) of a genotype's two alleles, or `None` if
/// the genotype is missing, or one of its alleles is out of range, `*`, symbolic, a breakend or
/// missing (`.`).
///
/// # Arguments
///
/// * `alleles` - the record's REF and ALT alleles.
/// * `genotype` - the genotype.
#[must_use]
pub fn allele_deltas<S: AsRef<[u8]>>(alleles: &[S], genotype: Genotype) -> Option<(i64, i64)> {
    let (first, second) = genotype.alleles()?;
    let reference = alleles.first()?.as_ref();
    let delta = |index: u32| length_change(reference, alleles.get(usize::try_from(index).ok()?)?);
    let (first, second) = (delta(first)?, delta(second)?);
    Some((first.min(second), first.max(second)))
}

/// The length change (in bp) of an allele relative to the reference allele, or `None` if the
/// allele is `*`, symbolic, a breakend or missing (`.`).
#[must_use]
pub fn length_change<S: AsRef<[u8]>>(reference: &[u8], allele: S) -> Option<i64> {
    let allele = allele.as_ref();
    if allele == SPANNING_DELETION || is_non_sequence(allele) {
        return None;
    }
    Some(allele.len() as i64 - reference.len() as i64)
}

/// Trims the shared suffix, then the shared prefix, of two alleles, comparing bases ignoring
/// case.
///
/// Returns the offset in `reference` of the first differing base, the deleted bases and the
/// inserted bases (as slices of the inputs, in their original case).
#[must_use]
pub fn trim_alleles<'a>(reference: &'a [u8], alt: &'a [u8]) -> (usize, &'a [u8], &'a [u8]) {
    let same = |(r, a): &(&u8, &u8)| r.eq_ignore_ascii_case(a);
    let suffix = reference.iter().rev().zip(alt.iter().rev()).take_while(same).count();
    let (reference, alt) = (&reference[..reference.len() - suffix], &alt[..alt.len() - suffix]);
    let prefix = reference.iter().zip(alt).take_while(same).count();
    (prefix, &reference[prefix..], &alt[prefix..])
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// The reference the normalization tests left-align against; 1-based positions 1 to 9.
    const REFERENCE: &[u8] = b"GCACACATG";

    fn fetch(pos: u64) -> Result<u8, String> {
        let index = usize::try_from(pos).unwrap().checked_sub(1).ok_or("position 0")?;
        REFERENCE.get(index).copied().ok_or_else(|| format!("position {pos} is off the end"))
    }

    fn normalized(pos: u64, alleles: &[&str]) -> NormalizedAlleles {
        NormalizedAlleles { pos, alleles: alleles.iter().map(|a| a.as_bytes().to_vec()).collect() }
    }

    #[rstest]
    #[case::normal_snp(1, &["G", "T"], 1, &["G", "T"])]
    #[case::padded_snp(3, &["ACAGA", "CCAGA"], 3, &["A", "C"])]
    #[case::deletion_shifts_left_through_repeat(5, &["ACA", "A"], 1, &["GCA", "G"])]
    #[case::deletion_with_no_left_equivalent(6, &["CAT", "CT"], 6, &["CA", "C"])]
    #[case::insertion_shifts_left(7, &["A", "ACA"], 1, &["G", "GCA"])]
    #[case::multi_allelic_shifts_jointly(2, &["CAC", "C", "CACAC"], 1, &["GCA", "G", "GCACA"])]
    #[case::symbolic_unchanged(1, &["G", "<DEL>"], 1, &["G", "<DEL>"])]
    #[case::spanning_deletion_unchanged(4, &["CA", "*"], 4, &["CA", "*"])]
    #[case::leading_padded_snp(3, &["ACACA", "ACACG"], 7, &["A", "G"])]
    #[case::deletion_at_position_one(1, &["GG", "G"], 1, &["GG", "G"])]
    #[case::shifts_to_position_one(2, &["CA", "CACA"], 1, &["G", "GCA"])]
    #[case::identical_ignoring_case(4, &["A", "a"], 4, &["A", "a"])]
    #[case::all_identical(4, &["CA", "CA", "CA"], 4, &["CA", "CA", "CA"])]
    #[case::single_allele(4, &["CA"], 4, &["CA"])]
    #[case::lowercase_uppercased(3, &["acaga", "ccaga"], 3, &["A", "C"])]
    #[case::empty_ref_unchanged(4, &["", "A"], 4, &["", "A"])]
    #[case::empty_alt_unchanged(4, &["A", ""], 4, &["A", ""])]
    #[case::missing_alt_unchanged(4, &["CA", "."], 4, &["CA", "."])]
    #[case::breakend_unchanged(5, &["A", "]2:5]A", "]3:7]A"], 5, &["A", "]2:5]A", "]3:7]A"])]
    #[case::single_breakend_unchanged(5, &["CA", "CA."], 5, &["CA", "CA."])]
    #[case::unchanged_keeps_case(1, &["g", "<DEL>"], 1, &["g", "<DEL>"])]
    fn normalize_with_reference(
        #[case] pos: u64,
        #[case] alleles: &[&str],
        #[case] expected_pos: u64,
        #[case] expected: &[&str],
    ) {
        let got = normalize_alleles(pos, alleles, fetch).unwrap();
        assert_eq!(got, normalized(expected_pos, expected));
    }

    #[test]
    fn normalize_uppercases_lowercase_fetched_bases() {
        let lowercase = |pos: u64| fetch(pos).map(|base| base.to_ascii_lowercase());
        let got = normalize_alleles(5, &["ACA", "A"], lowercase).unwrap();
        assert_eq!(got, normalized(1, &["GCA", "G"]));
    }

    #[test]
    fn normalize_propagates_fetch_errors() {
        let failing = |pos: u64| Err::<u8, _>(format!("no base at {pos}"));
        assert_eq!(normalize_alleles(5, &["ACA", "A"], failing), Err("no base at 4".to_string()));
    }

    #[rstest]
    #[case::padded_snp(3, &["ACAGA", "CCAGA"], 3, &["A", "C"])]
    #[case::deletion_not_shifted(6, &["CAT", "CT"], 6, &["CA", "C"])]
    #[case::keeps_one_base(1, &["AA", "A"], 1, &["AA", "A"])]
    #[case::repeat_deletion_not_shifted(5, &["ACA", "A"], 5, &["ACA", "A"])]
    #[case::empty_ref_unchanged(4, &["", "A"], 4, &["", "A"])]
    fn trim_shared_bases_never_extends(
        #[case] pos: u64,
        #[case] alleles: &[&str],
        #[case] expected_pos: u64,
        #[case] expected: &[&str],
    ) {
        assert_eq!(trim_shared_bases(pos, alleles), normalized(expected_pos, expected));
    }

    #[rstest]
    #[case::het(&[Some(0), Some(1)], "0/1")]
    #[case::unsorted(&[Some(1), Some(0)], "0/1")]
    #[case::hom_alt(&[Some(1), Some(1)], "1/1")]
    #[case::multi_allelic(&[Some(2), Some(1)], "1/2")]
    #[case::hom_ref(&[Some(0), Some(0)], "0/0")]
    #[case::one_missing(&[Some(0), None], "./.")]
    #[case::haploid(&[Some(1)], "./.")]
    #[case::triploid(&[Some(0), Some(1), Some(1)], "./.")]
    #[case::empty(&[], "./.")]
    #[case::allele_past_the_alts(&[Some(0), Some(3)], "./.")]
    fn genotype_from_alleles(#[case] alleles: &[Option<u32>], #[case] expected: &str) {
        let genotype = Genotype::from_alleles(alleles.iter().copied(), 3);
        assert_eq!(genotype.to_string(), expected);
        assert_eq!(expected.parse::<Genotype>().unwrap(), genotype);
    }

    #[rstest]
    #[case::unsorted("1/0")]
    #[case::phased("0|1")]
    #[case::half_missing("./1")]
    #[case::haploid("1")]
    #[case::empty("")]
    #[case::triploid("0/1/1")]
    #[case::plus_sign("+0/1")]
    #[case::leading_zero("00/1")]
    #[case::leading_zero_high("0/01")]
    #[case::whitespace(" 0/1")]
    #[case::missing_one_dot(".")]
    fn unnormalized_genotypes_do_not_parse(#[case] text: &str) {
        let message = text.parse::<Genotype>().unwrap_err().to_string();
        assert_eq!(
            message,
            format!("invalid normalized genotype '{text}' (expected e.g. 0/1, 1/2 or ./.)")
        );
    }

    #[rstest]
    #[case(Genotype::HET, Some(Genotype::HOM_ALT))]
    #[case(Genotype::HOM_ALT, Some(Genotype::HET))]
    #[case(Genotype::diploid(0, 0), None)]
    #[case(Genotype::diploid(1, 2), None)]
    #[case(Genotype::diploid(2, 2), None)]
    #[case(Genotype::MISSING, None)]
    fn flipped(#[case] genotype: Genotype, #[case] expected: Option<Genotype>) {
        assert_eq!(genotype.flipped(), expected);
        assert_eq!(genotype.is_flippable(), expected.is_some());
    }

    #[rstest]
    #[case::snp("A", &["C"], Some(VariantType::Snp))]
    #[case::padded_snp("TTAGA", &["CTAGA"], Some(VariantType::Snp))]
    #[case::multi_allelic_snp("A", &["C", "G"], Some(VariantType::Snp))]
    #[case::snp_with_spanning_deletion("A", &["*", "C"], Some(VariantType::Snp))]
    #[case::deletion("AC", &["A"], Some(VariantType::Indel))]
    #[case::insertion("A", &["AC"], Some(VariantType::Indel))]
    #[case::snp_and_indel("A", &["C", "AC"], Some(VariantType::Indel))]
    #[case::mnp("AC", &["GT"], None)]
    #[case::mixed_snp_and_mnp("AC", &["AT", "GT"], None)]
    #[case::identical("A", &["A"], None)]
    #[case::only_spanning_deletion("A", &["*"], None)]
    #[case::no_alts("A", &[], None)]
    #[case::symbolic("A", &["<DEL>"], None)]
    #[case::symbolic_with_indel("A", &["AC", "<NON_REF>"], None)]
    #[case::breakend("A", &["A[chr2:10["], None)]
    #[case::breakend_reverse("A", &["]chr2:10]A"], None)]
    #[case::single_breakend_after("A", &["A."], None)]
    #[case::single_breakend_before("A", &[".A"], None)]
    #[case::missing_alt("A", &["."], None)]
    #[case::soft_masked_identical("a", &["A"], None)]
    #[case::soft_masked_snp("ac", &["AT"], Some(VariantType::Snp))]
    fn variant_types(
        #[case] reference: &str,
        #[case] alts: &[&str],
        #[case] expected: Option<VariantType>,
    ) {
        assert_eq!(variant_type_of(reference.as_bytes(), alts), expected);
    }

    #[rstest]
    #[case(&["C"], Some(0))]
    #[case(&["*", "C"], Some(1))]
    #[case(&["*", "*"], None)]
    #[case(&[], None)]
    fn first_alt(#[case] alts: &[&str], #[case] expected: Option<usize>) {
        assert_eq!(first_alt_index(alts), expected);
    }

    #[rstest]
    #[case::het_deletion(&["ACC", "A"], Genotype::HET, Some((-2, 0)))]
    #[case::hom_insertion(&["A", "ACC"], Genotype::HOM_ALT, Some((2, 2)))]
    #[case::sorted_by_delta(&["AC", "ACCC", "A"], Genotype::diploid(1, 2), Some((-1, 2)))]
    #[case::missing(&["A", "AC"], Genotype::MISSING, None)]
    #[case::out_of_range(&["A", "AC"], Genotype::diploid(0, 2), None)]
    #[case::spanning_deletion(&["A", "*"], Genotype::HET, None)]
    #[case::symbolic(&["A", "<DEL>"], Genotype::HET, None)]
    #[case::breakend(&["A", "A[chr2:5["], Genotype::HET, None)]
    #[case::single_breakend(&["A", "A."], Genotype::HET, None)]
    #[case::missing_alt(&["A", "."], Genotype::HET, None)]
    #[case::no_alleles(&[], Genotype::HET, None)]
    fn deltas(
        #[case] alleles: &[&str],
        #[case] genotype: Genotype,
        #[case] expected: Option<(i64, i64)>,
    ) {
        assert_eq!(allele_deltas(alleles, genotype), expected);
    }

    #[rstest]
    #[case::snp("A", "C", Some(0))]
    #[case::insertion("A", "ACG", Some(2))]
    #[case::deletion("ACG", "A", Some(-2))]
    #[case::spanning_deletion("A", "*", None)]
    #[case::symbolic("A", "<DEL>", None)]
    #[case::missing("A", ".", None)]
    fn length_changes(
        #[case] reference: &str,
        #[case] allele: &str,
        #[case] expected: Option<i64>,
    ) {
        assert_eq!(length_change(reference.as_bytes(), allele), expected);
    }

    #[rstest]
    #[case::deletion("GCA", "G", 1, "CA", "")]
    #[case::insertion("G", "GCA", 1, "", "CA")]
    #[case::repeat_deletion("ACAG", "AG", 0, "AC", "")]
    #[case::snp("ACG", "ATG", 1, "C", "T")]
    #[case::suffix_trimmed_before_prefix("AA", "A", 0, "A", "")]
    #[case::identical("AC", "AC", 0, "", "")]
    #[case::empty("", "", 0, "", "")]
    #[case::ignores_case("AC", "ac", 0, "", "")]
    #[case::keeps_input_case("ac", "aGc", 1, "", "G")]
    fn trims(
        #[case] reference: &str,
        #[case] alt: &str,
        #[case] offset: usize,
        #[case] deleted: &str,
        #[case] inserted: &str,
    ) {
        let got = trim_alleles(reference.as_bytes(), alt.as_bytes());
        assert_eq!(got, (offset, deleted.as_bytes(), inserted.as_bytes()));
    }
}
