//! The VCF feature source ([`crate::feature_source::FeatureSource::Vcf`]): the raw annotations of
//! HaplotypeCaller's gVCF mode, derived from a standard VCF's annotations instead.
//!
//! A reblocked gVCF's variant sites carry `RAW_MQandDP`, `QUALapprox`, `VarDP` and `AS_RAW_MQ`
//! (see [`crate::gvcf`]). A VCF without its gVCF (e.g. from GenotypeGVCFs, or HaplotypeCaller run
//! without `-ERC`) has INFO/MQ, INFO/DP, FORMAT/AD, FORMAT/PL and FORMAT/SB or INFO/SOR, from which
//! [`derive_annotation`] computes stand-ins that the features are derived from as for the gVCF
//! source.

use std::path::Path;

use anyhow::{Result, bail};
use log::info;
use rust_htslib::bcf::Record;
use rust_htslib::bcf::header::HeaderView;

use crate::alleles::Genotype;
use crate::gvcf::{RawAnnotation, is_concrete_alt};
use crate::raw_records::RawRecord;
use crate::vcf::{FieldReader, field_name, has_field};

/// A VCF header field: its ID, and whether it is INFO (else FORMAT).
type HeaderField = (&'static str, bool);

/// The fields that the VCF source needs a VCF header to declare, besides one of
/// [`SOR_FIELDS`].
const REQUIRED_FIELDS: [HeaderField; 4] =
    [("MQ", true), ("DP", true), ("AD", false), ("PL", false)];

/// The fields that the VCF source needs a VCF header to declare at least one of: `sor` comes from
/// FORMAT/SB, or else INFO/SOR.
const SOR_FIELDS: [HeaderField; 2] = [("SB", false), ("SOR", true)];

/// True if a header declares a field.
fn declares(header: &HeaderView, &(field, is_info): &HeaderField) -> bool {
    has_field(header, field.as_bytes(), is_info)
}

/// The [`REQUIRED_FIELDS`] a VCF header does not declare, as `INFO/MQ` etc., and
/// `FORMAT/SB or INFO/SOR` when it declares neither of [`SOR_FIELDS`].
fn missing_fields(header: &HeaderView) -> Vec<String> {
    let name = |&(field, is_info): &HeaderField| field_name(field, is_info);
    let mut missing: Vec<String> =
        REQUIRED_FIELDS.iter().filter(|f| !declares(header, f)).map(name).collect();
    if !SOR_FIELDS.iter().any(|f| declares(header, f)) {
        missing.push(SOR_FIELDS.iter().map(name).collect::<Vec<_>>().join(" or "));
    }
    missing
}

/// Checks that a VCF header declares the fields that the VCF source derives its features from:
/// INFO/MQ, INFO/DP, FORMAT/AD, FORMAT/PL, and FORMAT/SB or INFO/SOR.
///
/// The VCF source is chosen by `extract --feature-source vcf`, or by `score` for models trained on
/// such tables, so the error names the remedies of both.
///
/// # Errors
///
/// Returns an error naming every field the header does not declare.
pub fn check_header(header: &HeaderView, vcf: &Path) -> Result<()> {
    let missing = missing_fields(header);
    if !missing.is_empty() {
        bail!(
            "VCF {} does not declare {}, which the vcf feature source (of extract --feature-source \
             vcf, and of models trained on its tables) derives the features of HaplotypeCaller's \
             gVCF-mode annotations from: give a VCF that declares them, or use the gvcf feature \
             source (extract --feature-source gvcf, and models trained on its tables) with the \
             gVCF the VCF was genotyped from",
            vcf.display(),
            missing.join(", ")
        );
    }
    info!(
        "Deriving the features of gVCF-mode annotations from the standard annotations of VCF {} \
         (feature source vcf)",
        vcf.display()
    );
    Ok(())
}

/// The distinct alternate alleles a genotype calls, by allele index: none for a genotype that is
/// not a variant (`0/0`, or missing, which includes every haploid genotype), one for `0/1` or
/// `1/1`, two for `1/2`.
pub type CalledAlts = [Option<u32>; 2];

/// The [`CalledAlts`] of a genotype.
#[must_use]
pub fn called_alts(genotype: Genotype) -> CalledAlts {
    match genotype.alleles() {
        Some((0, 0)) | None => [None, None],
        Some((first, second)) if first == second || first == 0 => [Some(second), None],
        Some((first, second)) => [Some(first), Some(second)],
    }
}

/// True if a genotype is a variant (het or hom-alt): it calls an alternate allele.
fn is_variant(called: CalledAlts) -> bool {
    called[0].is_some()
}

/// The FORMAT/AD of the reference and the called alternate alleles, summed, with missing depths
/// left out (`None` if every one is missing), as ReblockGVCF keeps them: it drops the alternate
/// alleles the genotype does not call, with their depths.
#[must_use]
pub fn called_depth(ad: &[f64], called: CalledAlts) -> Option<f64> {
    std::iter::once(Some(0))
        .chain(called)
        .flatten()
        .filter_map(|index| ad.get(index as usize).copied())
        .filter(|depth| !depth.is_nan())
        .reduce(|a, b| a + b)
}

/// True if [`variant_depth`] falls back to FORMAT/DP: a variant genotype whose
/// [`called_depth`] is zero or missing.
fn needs_format_dp(called: CalledAlts, called_depth: Option<f64>) -> bool {
    is_variant(called) && !called_depth.is_some_and(|depth| depth != 0.0)
}

/// Reads the VCF source's raw annotations of a record (see [`derive_annotation`]), whose other
/// raw values are `raw` and whose [`called_depth`] over [`called_alts`] is `called_depth`. It
/// reads INFO/MQ, INFO/SOR only when the record has no FORMAT/SB, and FORMAT/DP only when VarDP
/// falls back to it.
///
/// # Errors
///
/// Returns an error if htslib cannot decode a field.
pub fn read_annotation(
    fields: &mut FieldReader,
    record: &Record,
    raw: &RawRecord,
    called: CalledAlts,
    called_depth: Option<f64>,
) -> Result<RawAnnotation> {
    let info_mq = fields.info_value(record, b"MQ", 0)?;
    let sb = [raw.sb_ref_fwd, raw.sb_ref_rev, raw.sb_alt_fwd, raw.sb_alt_rev];
    let info_sor = if sb.iter().any(|count| count.is_nan()) {
        fields.info_value(record, b"SOR", 0)?
    } else {
        f64::NAN
    };
    let format_dp = if needs_format_dp(called, called_depth) {
        fields.format(record, b"DP")?.and_then(|dp| dp.first().copied()).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    Ok(derive_annotation(raw, called, called_depth, info_mq, info_sor, format_dp))
}

/// The raw annotations that the VCF source derives from a record's standard annotations, standing
/// in for those of a reblocked gVCF, from its raw values `raw`, its [`called_alts`] and their
/// [`called_depth`], INFO/MQ, INFO/SOR and FORMAT/DP:
///
/// * `raw_mq` is INFO/MQ² × INFO/DP and `raw_dp` is INFO/DP, so `mq` is INFO/MQ where INFO/DP is
///   positive (and missing elsewhere, as the gVCF source's is at a zero depth), and `dp_norm`
///   divides INFO/DP.
/// * `qual_approx` is FORMAT/PL\[0\], as ReblockGVCF's QUALapprox is (INFO/QD is not read: GATK
///   divides QUAL rather than PL\[0\], and replaces values of 35 or more with a random draw).
/// * `var_dp` is ReblockGVCF's VarDP (see [`variant_depth`]).
/// * `n_alt` is, for a variant genotype, the number of distinct alternate alleles it calls, as
///   ReblockGVCF drops the others; for other genotypes, the number of alternate alleles. Either
///   excludes `<NON_REF>`.
/// * `as_mq_ref` and `as_mq_alt` are missing (a VCF has no per-allele mapping qualities), so
///   `as_alt_frac` is.
/// * `sor` is INFO/SOR, which the `sor` feature takes only when FORMAT/SB is missing.
#[must_use]
fn derive_annotation(
    raw: &RawRecord,
    called: CalledAlts,
    called_depth: Option<f64>,
    info_mq: f64,
    info_sor: f64,
    format_dp: f64,
) -> RawAnnotation {
    let counted = |index: u32| !is_variant(called) || called.contains(&Some(index));
    let n_alt =
        (1..).zip(&raw.alts).filter(|&(index, alt)| counted(index) && is_concrete_alt(alt)).count();
    RawAnnotation {
        raw_mq: info_mq * info_mq * raw.dp,
        raw_dp: raw.dp,
        qual_approx: raw.pl_ref,
        var_dp: variant_depth(called, called_depth, format_dp, raw.dp),
        as_mq_ref: f64::NAN,
        as_mq_alt: f64::NAN,
        n_alt: n_alt as f64,
        sor: info_sor,
    }
}

/// ReblockGVCF's VarDP of a single-sample call. ReblockGVCF first drops the alternate alleles the
/// genotype does not call (with their FORMAT/AD), then GATK's `QualByDepth.getDepth` (without read
/// likelihoods) takes:
///
/// * for a het or hom-alt genotype, its [`called_depth`] if not zero, else FORMAT/DP (zero if
///   missing);
/// * for any other genotype (hom-ref or missing), zero.
///
/// A zero depth is then replaced with INFO/DP, or with 1 when the record has no INFO/DP.
///
/// winnowr reads a haploid genotype (e.g. `1` on chrX) as missing, so a haploid call takes
/// INFO/DP, where GATK takes the AD sum of a haploid alternate call.
fn variant_depth(
    called: CalledAlts,
    called_depth: Option<f64>,
    format_dp: f64,
    info_dp: f64,
) -> f64 {
    let depth = match called_depth {
        _ if !is_variant(called) => 0.0,
        _ if !needs_format_dp(called, called_depth) => called_depth.unwrap_or(0.0),
        _ if format_dp.is_nan() => 0.0,
        _ => format_dp,
    };
    if depth != 0.0 {
        depth
    } else if info_dp.is_nan() {
        1.0
    } else {
        info_dp
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const NAN: f64 = f64::NAN;

    /// [`variant_depth`] of a genotype and its FORMAT/AD.
    fn depth(genotype: Genotype, ad: Option<&[f64]>, format_dp: f64, info_dp: f64) -> f64 {
        let called = called_alts(genotype);
        variant_depth(called, ad.and_then(|ad| called_depth(ad, called)), format_dp, info_dp)
    }

    /// ReblockGVCF's VarDP: the depths of the reference and called alleles of a variant genotype,
    /// else its FORMAT/DP, and INFO/DP (or 1) in place of a zero depth.
    #[rstest]
    #[case::het(Genotype::HET, Some(&[5.0, 7.0][..]), 20.0, 30.0, 12.0)]
    #[case::hom_alt(Genotype::HOM_ALT, Some(&[1.0, 6.0][..]), NAN, 30.0, 7.0)]
    #[case::two_alts(Genotype::diploid(1, 2), Some(&[2.0, 3.0, 4.0][..]), 20.0, 30.0, 9.0)]
    #[case::uncalled_alt_is_dropped(Genotype::HET, Some(&[10.0, 8.0, 3.0][..]), 21.0, 21.0, 18.0)]
    #[case::second_alt_called(Genotype::diploid(0, 2), Some(&[10.0, 8.0, 3.0][..]), NAN, 21.0, 13.0)]
    #[case::missing_depth_left_out(Genotype::HET, Some(&[NAN, 7.0][..]), 20.0, 30.0, 7.0)]
    #[case::hom_ref_takes_info_dp(Genotype::diploid(0, 0), Some(&[12.0, 0.0][..]), 20.0, 30.0, 30.0)]
    #[case::no_call_takes_info_dp(Genotype::MISSING, Some(&[6.0, 6.0][..]), 20.0, 30.0, 30.0)]
    #[case::zero_ad_takes_format_dp(Genotype::HET, Some(&[0.0, 0.0][..]), 20.0, 30.0, 20.0)]
    #[case::all_missing_ad_takes_format_dp(Genotype::HET, Some(&[NAN, NAN][..]), 20.0, 30.0, 20.0)]
    #[case::no_ad_takes_format_dp(Genotype::HET, None, 20.0, 30.0, 20.0)]
    #[case::no_format_dp_takes_info_dp(Genotype::HET, Some(&[0.0, 0.0][..]), NAN, 30.0, 30.0)]
    #[case::zero_format_dp_takes_info_dp(Genotype::HET, None, 0.0, 30.0, 30.0)]
    #[case::nothing_is_one(Genotype::HET, None, NAN, NAN, 1.0)]
    #[case::hom_ref_without_info_dp_is_one(Genotype::diploid(0, 0), None, 5.0, NAN, 1.0)]
    #[case::zero_info_dp_is_kept(Genotype::MISSING, None, NAN, 0.0, 0.0)]
    fn variant_depth_follows_reblock_gvcf(
        #[case] genotype: Genotype,
        #[case] ad: Option<&[f64]>,
        #[case] format_dp: f64,
        #[case] info_dp: f64,
        #[case] expected: f64,
    ) {
        assert_eq!(depth(genotype, ad, format_dp, info_dp), expected);
    }

    /// A haploid genotype reads as missing, so a haploid call takes INFO/DP.
    #[test]
    fn a_haploid_call_takes_info_dp() {
        let haploid = Genotype::from_alleles([Some(1)], 2);
        assert_eq!(depth(haploid, Some(&[2.0, 10.0]), 12.0, 15.0), 15.0);
    }

    /// FORMAT/DP is read only for a variant genotype without a depth from FORMAT/AD.
    #[rstest]
    #[case::het_with_ad(Genotype::HET, Some(7.0), false)]
    #[case::het_with_zero_ad(Genotype::HET, Some(0.0), true)]
    #[case::het_without_ad(Genotype::HET, None, true)]
    #[case::hom_ref(Genotype::diploid(0, 0), None, false)]
    #[case::missing(Genotype::MISSING, None, false)]
    fn format_dp_is_needed_only_without_a_called_depth(
        #[case] genotype: Genotype,
        #[case] called_depth: Option<f64>,
        #[case] expected: bool,
    ) {
        assert_eq!(needs_format_dp(called_alts(genotype), called_depth), expected);
    }

    /// A record of `genotype` and `alts` with INFO/DP 9 and FORMAT/PL\[0\] 45.
    fn record(genotype: Genotype, alts: &[&str]) -> RawRecord {
        let mut record = crate::testing::raw_record_with_missing_values("chr1", 100, "A", alts);
        (record.genotype, record.dp, record.pl_ref) = (genotype, 9.0, 45.0);
        record
    }

    /// [`derive_annotation`] with INFO/MQ 30, INFO/SOR 0.5 and FORMAT/DP 8.
    fn derive(record: &RawRecord, called_depth: Option<f64>) -> RawAnnotation {
        derive_annotation(record, called_alts(record.genotype), called_depth, 30.0, 0.5, 8.0)
    }

    #[test]
    fn derives_the_raw_annotations() {
        let raw = derive(&record(Genotype::HET, &["C"]), Some(7.0));
        assert_eq!(
            [raw.raw_mq, raw.raw_dp, raw.qual_approx, raw.var_dp, raw.n_alt, raw.sor],
            [900.0 * 9.0, 9.0, 45.0, 7.0, 1.0, 0.5]
        );
        assert!(raw.as_mq_ref.is_nan() && raw.as_mq_alt.is_nan());
    }

    /// Without FORMAT/AD, VarDP is FORMAT/DP, else INFO/DP.
    #[test]
    fn without_ad_var_dp_falls_back_to_format_dp_then_info_dp() {
        let het = record(Genotype::HET, &["C"]);
        assert_eq!(derive(&het, None).var_dp, 8.0);
        let without_format_dp =
            derive_annotation(&het, called_alts(het.genotype), None, 30.0, 0.5, NAN);
        assert_eq!(without_format_dp.var_dp, 9.0);
    }

    /// n_alt counts the distinct called alternates of a variant genotype (as ReblockGVCF keeps),
    /// else every alternate, and never `<NON_REF>`.
    #[rstest]
    #[case::uncalled_alt_dropped(Genotype::HET, &["C", "G"], 1.0)]
    #[case::two_called(Genotype::diploid(1, 2), &["C", "G"], 2.0)]
    #[case::hom_alt_second(Genotype::diploid(2, 2), &["C", "G", "T"], 1.0)]
    #[case::called_star(Genotype::diploid(1, 2), &["*", "C"], 2.0)]
    #[case::hom_ref_counts_all(Genotype::diploid(0, 0), &["C", "G"], 2.0)]
    #[case::missing_counts_all(Genotype::MISSING, &["C", "G", "T"], 3.0)]
    #[case::non_ref_not_counted(Genotype::diploid(0, 0), &["C", "<NON_REF>"], 1.0)]
    #[case::called_non_ref_not_counted(Genotype::diploid(1, 2), &["C", "<NON_REF>"], 1.0)]
    fn n_alt_counts_the_alleles_reblocking_keeps(
        #[case] genotype: Genotype,
        #[case] alts: &[&str],
        #[case] expected: f64,
    ) {
        assert_eq!(derive(&record(genotype, alts), None).n_alt, expected);
    }
}
