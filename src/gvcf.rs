//! The raw HaplotypeCaller annotations that gVCF-mode calling produces.
//!
//! In reblocked `--dragen-mode` gVCFs, variant sites carry `RAW_MQandDP` (sum of squared mapping
//! qualities and the read depth), `QUALapprox`, `VarDP` and `AS_RAW_MQ` (per-allele sum of squared
//! mapping qualities, `|`-separated, with the `<NON_REF>` allele last). Reference blocks do not,
//! and are skipped. A VCF extracted from the gVCF may keep the same annotations, set for its own
//! alleles (with `AS_RAW_MQ` one value per allele and no `<NON_REF>`), in which case they are read
//! from the VCF itself.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use rust_htslib::bcf::header::HeaderView;
use rust_htslib::bcf::{IndexedReader, Read};

use crate::vcf::{FieldReader, check_contigs, fetch_span, has_info};

/// The raw INFO annotations read, which a gVCF (or a VCF that keeps them) must declare.
pub const RAW_ANNOTATION_INFO: [&str; 4] = ["RAW_MQandDP", "QUALapprox", "VarDP", "AS_RAW_MQ"];

/// The gVCF allele standing for any unobserved alternate allele.
pub const NON_REF: &[u8] = b"<NON_REF>";

/// The raw annotations at one variant site, from the gVCF or the VCF. Missing values are NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawAnnotation {
    /// The sum of squared mapping qualities (`RAW_MQandDP[0]`).
    pub raw_mq: f64,
    /// The read depth used for `RAW_MQandDP` (`RAW_MQandDP[1]`).
    pub raw_dp: f64,
    /// `QUALapprox`.
    pub qual_approx: f64,
    /// `VarDP`.
    pub var_dp: f64,
    /// The reference allele's `AS_RAW_MQ`.
    pub as_mq_ref: f64,
    /// `AS_RAW_MQ` summed over the alternate alleles (excluding `<NON_REF>`).
    pub as_mq_alt: f64,
    /// The number of alternate alleles (excluding `<NON_REF>`).
    pub n_alt: f64,
}

impl RawAnnotation {
    /// The annotations of a record that has none.
    pub const MISSING: RawAnnotation = RawAnnotation {
        raw_mq: f64::NAN,
        raw_dp: f64::NAN,
        qual_approx: f64::NAN,
        var_dp: f64::NAN,
        as_mq_ref: f64::NAN,
        as_mq_alt: f64::NAN,
        n_alt: f64::NAN,
    };
}

/// The raw annotations a VCF or gVCF header does not declare, in [`RAW_ANNOTATION_INFO`] order.
#[must_use]
pub fn missing_raw_annotations(header: &HeaderView) -> Vec<&'static str> {
    RAW_ANNOTATION_INFO.into_iter().filter(|key| !has_info(header, key.as_bytes())).collect()
}

/// True if a VCF header declares every raw annotation, so its records can supply them.
#[must_use]
pub fn has_raw_annotations(header: &HeaderView) -> bool {
    missing_raw_annotations(header).is_empty()
}

/// Checks that a gVCF header declares the raw annotations and `contigs`.
///
/// # Errors
///
/// Returns an error if a contig is not in the header (e.g. `1` vs `chr1` naming), or any of
/// [`RAW_ANNOTATION_INFO`] is not declared.
pub fn check_gvcf_header<S: AsRef<str>>(
    header: &HeaderView,
    path: &Path,
    contigs: &[S],
) -> Result<()> {
    check_contigs(header, contigs, path)?;
    let missing = missing_raw_annotations(header);
    if !missing.is_empty() {
        bail!(
            "gVCF {} lacks the INFO annotations {}; it must come from HaplotypeCaller in gVCF mode \
             (e.g. --dragen-mode)",
            path.display(),
            missing.join(", ")
        );
    }
    Ok(())
}

/// Reads the raw annotations of one record (a gVCF variant site or a VCF call), or `None` if it
/// has no `RAW_MQandDP` (e.g. a reference block).
///
/// # Errors
///
/// Returns an error if htslib cannot decode an annotation.
pub fn raw_annotation(
    fields: &mut FieldReader,
    record: &rust_htslib::bcf::Record,
    alleles: &[&[u8]],
) -> Result<Option<RawAnnotation>> {
    let Some(raw_mq_and_dp) = fields.info(record, b"RAW_MQandDP")? else {
        return Ok(None);
    };
    let element = |values: &[f64], index: usize| values.get(index).copied().unwrap_or(f64::NAN);
    let (raw_mq, raw_dp) = (element(raw_mq_and_dp, 0), element(raw_mq_and_dp, 1));
    let qual_approx = fields.info_value(record, b"QUALapprox", 0)?;
    let var_dp = fields.info_value(record, b"VarDP", 0)?;
    let (as_mq_ref, as_mq_alt, n_alt) = match fields.info(record, b"AS_RAW_MQ")? {
        Some(values) => split_as_raw_mq(values, alleles),
        None => (f64::NAN, f64::NAN, f64::NAN),
    };
    Ok(Some(RawAnnotation { raw_mq, raw_dp, qual_approx, var_dp, as_mq_ref, as_mq_alt, n_alt }))
}

/// Splits per-allele `AS_RAW_MQ` values into the reference share, the summed alternate share
/// (excluding `<NON_REF>`) and the number of alternate alleles (excluding `<NON_REF>`).
///
/// Missing (NaN) values count as zero. All three are NaN when there is not one value per allele.
#[must_use]
pub fn split_as_raw_mq<S: AsRef<[u8]>>(values: &[f64], alleles: &[S]) -> (f64, f64, f64) {
    if values.len() != alleles.len() || values.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let or_zero = |value: f64| if value.is_nan() { 0.0 } else { value };
    let alt_values: Vec<f64> = values[1..]
        .iter()
        .zip(&alleles[1..])
        .filter(|(_, allele)| allele.as_ref() != NON_REF)
        .map(|(&value, _)| value)
        .collect();
    let as_mq_alt = alt_values.iter().copied().map(or_zero).sum();
    (or_zero(values[0]), as_mq_alt, alt_values.len() as f64)
}

/// A gVCF variant site with raw annotations.
#[derive(Debug, Clone, PartialEq)]
pub struct GvcfSite {
    /// The reference allele.
    pub reference: Vec<u8>,
    /// The alternate alleles, excluding `<NON_REF>`.
    pub alts: Vec<Vec<u8>>,
    /// The raw annotations.
    pub annotation: RawAnnotation,
}

impl GvcfSite {
    /// True if the site has the call's reference allele and every one of its alternate alleles
    /// other than `*` (ignoring case).
    pub fn has_alleles<S: AsRef<[u8]>>(&self, reference: &[u8], alts: &[S]) -> bool {
        self.reference.eq_ignore_ascii_case(reference)
            && alts
                .iter()
                .map(AsRef::as_ref)
                .filter(|&alt| alt != b"*")
                .all(|alt| self.alts.iter().any(|site_alt| site_alt.eq_ignore_ascii_case(alt)))
    }
}

/// The gVCF site whose annotations a call takes: the first site at the call's position with the
/// call's alleles (see [`GvcfSite::has_alleles`]), or else the first site at the position.
///
/// Matching alleles keeps a call from taking the annotations of a different variant that the
/// gVCF reports at the same position. The fallback to the first site keeps calls whose alleles
/// the gVCF represents differently; callers count and report it.
pub fn matching_site<'a, S: AsRef<[u8]>>(
    sites: &'a [GvcfSite],
    reference: &[u8],
    alts: &[S],
) -> Option<&'a GvcfSite> {
    sites.iter().find(|site| site.has_alleles(reference, alts)).or_else(|| sites.first())
}

/// Reads the gVCF variant sites that have raw annotations at `positions` (1-based) on one
/// contig, in gVCF order at each position. Reading stops after the last of `positions`.
///
/// # Errors
///
/// Returns an error if the gVCF header does not declare the contig, or the gVCF cannot be read.
pub fn read_gvcf_sites(
    gvcf: &mut IndexedReader,
    path: &Path,
    contig: &str,
    positions: &HashSet<u64>,
) -> Result<HashMap<u64, Vec<GvcfSite>>> {
    let mut sites: HashMap<u64, Vec<GvcfSite>> = HashMap::with_capacity(positions.len());
    let (Some(&first), Some(&last)) = (positions.iter().min(), positions.iter().max()) else {
        return Ok(sites);
    };
    // From the first position: the records before it are not needed.
    if !fetch_span(gvcf, contig, first.saturating_sub(1), Some(last), path)? {
        return Ok(sites);
    }
    let mut record = gvcf.empty_record();
    let mut fields = FieldReader::new();
    while let Some(read) = gvcf.read(&mut record) {
        read.with_context(|| {
            format!("failed to read a record on {contig} from {}", path.display())
        })?;
        // Check the position first: most gVCF records are reference blocks at no call.
        let pos = u64::try_from(record.pos() + 1).unwrap_or(0);
        if pos > last {
            break;
        }
        if !positions.contains(&pos) {
            continue;
        }
        let alleles = record.alleles();
        if let Some(annotation) = raw_annotation(&mut fields, &record, &alleles)? {
            let (reference, alts) =
                alleles.split_first().map_or((&b""[..], &[][..]), |(r, a)| (*r, a));
            let alts = alts.iter().filter(|&&alt| alt != NON_REF).map(|alt| alt.to_vec()).collect();
            sites.entry(pos).or_default().push(GvcfSite {
                reference: reference.to_vec(),
                alts,
                annotation,
            });
        }
    }
    Ok(sites)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const NAN3: (f64, f64, f64) = (f64::NAN, f64::NAN, f64::NAN);

    fn same(got: (f64, f64, f64), expected: (f64, f64, f64)) -> bool {
        let eq = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b;
        eq(got.0, expected.0) && eq(got.1, expected.1) && eq(got.2, expected.2)
    }

    fn site(reference: &str, alts: &[&str], raw_mq: f64) -> GvcfSite {
        GvcfSite {
            reference: reference.as_bytes().to_vec(),
            alts: alts.iter().map(|alt| alt.as_bytes().to_vec()).collect(),
            annotation: RawAnnotation { raw_mq, ..RawAnnotation::MISSING },
        }
    }

    #[rstest]
    #[case::exact(&["AT"], Some(2.0))]
    #[case::subset_of_site_alts(&["C"], Some(1.0))]
    #[case::ignores_spanning_deletion(&["*", "AT"], Some(2.0))]
    #[case::case_insensitive(&["at"], Some(2.0))]
    #[case::no_match_takes_the_first(&["G"], Some(1.0))]
    fn matches_sites_by_alleles(#[case] alts: &[&str], #[case] expected: Option<f64>) {
        let sites = [site("A", &["C", "T"], 1.0), site("A", &["AT"], 2.0)];
        let got = matching_site(&sites, b"A", alts).map(|s| s.annotation.raw_mq);
        assert_eq!(got, expected);
    }

    #[test]
    fn matching_requires_the_reference_allele() {
        let sites = [site("A", &["C"], 1.0), site("AT", &["A"], 2.0)];
        assert_eq!(matching_site(&sites, b"AT", &["A"]).map(|s| s.annotation.raw_mq), Some(2.0));
        assert!(matching_site(&[], b"A", &["C"]).is_none());
    }

    #[rstest]
    #[case::gvcf_with_non_ref(&[10.0, 20.0, 0.0], &["A", "C", "<NON_REF>"], (10.0, 20.0, 1.0))]
    #[case::vcf_multi_allelic(&[10.0, 20.0, 5.0], &["A", "C", "G"], (10.0, 25.0, 2.0))]
    #[case::missing_values_count_as_zero(&[f64::NAN, 20.0, f64::NAN], &["A", "C", "G"], (0.0, 20.0, 2.0))]
    #[case::wrong_count(&[10.0, 20.0], &["A", "C", "<NON_REF>"], NAN3)]
    #[case::empty(&[], &[] as &[&str], NAN3)]
    fn splits_as_raw_mq(
        #[case] values: &[f64],
        #[case] alleles: &[&str],
        #[case] expected: (f64, f64, f64),
    ) {
        let got = split_as_raw_mq(values, alleles);
        assert!(same(got, expected), "{got:?} != {expected:?}");
    }
}
