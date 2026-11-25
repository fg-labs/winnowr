//! Model features: their names, and the features derived from raw record values.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::alleles::{Genotype, first_alt_index};
use crate::common::{PASS, Task, VariantType};
use crate::neighbourhood::{Call, N_FEATURES, NEIGHBOURHOOD_FEATURES, neighbourhood_features};
use crate::raw_records::RawRecord;
use crate::regions::RegionSet;

/// The smallest depth `dp_norm` may divide by in place of the calls' median depth (see
/// [`derive_features`]): a read depth of at least one read. A smaller divisor makes `dp_norm` of
/// ordinary depths huge, or infinite, which a model reads as missing.
pub const MIN_DP_MEDIAN: f64 = 1.0;

/// Checks a depth for `dp_norm` to divide by in place of the calls' median depth: finite and at
/// least [`MIN_DP_MEDIAN`].
///
/// # Errors
///
/// Returns an error naming the value if it is not finite or is below [`MIN_DP_MEDIAN`].
pub fn check_dp_median(value: f64) -> anyhow::Result<f64> {
    if !(value.is_finite() && value >= MIN_DP_MEDIAN) {
        anyhow::bail!(
            "the depth median must be a finite depth of at least {MIN_DP_MEDIAN}, got {value:?}"
        );
    }
    Ok(value)
}

/// The features used for every variant type.
pub const COMMON_FEATURES: [&str; 14] = [
    "qual",
    "dp",
    "gq",
    "mq_rank_sum",
    "read_pos_rank_sum",
    "base_q_rank_sum",
    "ab",
    "sor",
    "het",
    "mq",
    "qd",
    "as_alt_frac",
    "dp_norm",
    "n_alt",
];

/// The additional features used for indels.
pub const INDEL_FEATURES: [&str; 4] = ["indel_len", "indel_abs_len", "str_period", "str_repeats"];

/// The additional features used by promote models.
pub const PROMOTE_FEATURES: [&str; 7] =
    ["excess_het", "pl_ref", "pl_het", "pl_hom", "ad_ref", "ad_alt_total", "n_alt_rec"];

/// The feature name for membership in the named region set: `region_<name>`.
#[must_use]
pub fn region_feature(name: &str) -> String {
    format!("region_{name}")
}

/// The ordered model features for a variant type.
///
/// # Arguments
///
/// * `variant_type` - the variant type; indels add [`INDEL_FEATURES`].
/// * `region_names` - the names of the region sets, in order.
/// * `neighbourhood` - whether to include the neighbourhood features.
/// * `task` - the model's task; promote models add [`PROMOTE_FEATURES`].
/// * `repeat_length` - repeat-length features to append.
#[must_use]
pub fn feature_names<R: AsRef<str>, L: AsRef<str>>(
    variant_type: VariantType,
    region_names: &[R],
    neighbourhood: bool,
    task: Task,
    repeat_length: &[L],
) -> Vec<String> {
    let mut names: Vec<String> = COMMON_FEATURES.iter().map(ToString::to_string).collect();
    if task == Task::Promote {
        names.extend(PROMOTE_FEATURES.iter().map(ToString::to_string));
    }
    if variant_type == VariantType::Indel {
        names.extend(INDEL_FEATURES.iter().map(ToString::to_string));
    }
    names.extend(region_names.iter().map(|name| region_feature(name.as_ref())));
    if neighbourhood {
        names.extend(NEIGHBOURHOOD_FEATURES.iter().map(ToString::to_string));
    }
    names.extend(repeat_length.iter().map(|name| name.as_ref().to_string()));
    names
}

/// GATK's StrandOddsRatio (SOR) from the per-strand allele depths of FORMAT/SB.
///
/// As in GATK 4 (`StrandOddsRatio.calculateSOR`), each count gets a pseudocount of one, then with
/// `R = (rf * ar) / (rr * af)`:
/// `SOR = ln(R + 1/R) + ln(min(rf, rr) / max(rf, rr)) - ln(min(af, ar) / max(af, ar))`.
/// FORMAT/SB already sums the alternate alleles of multi-allelic records. NaN if any count is.
#[must_use]
pub fn strand_odds_ratio(ref_fwd: f64, ref_rev: f64, alt_fwd: f64, alt_rev: f64) -> f64 {
    // f64::min and f64::max ignore NaN, so a missing count is handled up front.
    if [ref_fwd, ref_rev, alt_fwd, alt_rev].iter().any(|count| count.is_nan()) {
        return f64::NAN;
    }
    let (rf, rr, af, ar) = (ref_fwd + 1.0, ref_rev + 1.0, alt_fwd + 1.0, alt_rev + 1.0);
    let ratio = (rf * ar) / (rr * af);
    let ref_ratio = rf.min(rr) / rf.max(rr);
    let alt_ratio = af.min(ar) / af.max(ar);
    (ratio + 1.0 / ratio).ln() + ref_ratio.ln() - alt_ratio.ln()
}

/// The depth `dp_norm` divides each call's depth by (see [`derive_features`]), as a feature table
/// records it for provenance.
///
/// Equality compares `depth` bit for bit (the type is `Eq` so that the table metadata holding it
/// can be); recorded depths are finite, so this is ordinary equality on them.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DpNormDivisor {
    /// The depth divided by; `None` when it is the calls' median and that median is not finite
    /// (no call has a depth, so every call's `dp_norm` is missing). JSON has no NaN, hence the
    /// option. The default (`None`, not given) is the calls' median of no calls.
    pub depth: Option<f64>,
    /// True if the depth was given (`--dp-median`), false if it is the calls' median depth.
    pub given: bool,
}

impl DpNormDivisor {
    /// The divisor of a depth, `None` if the depth is not finite (e.g. the NaN median of no
    /// depths).
    #[must_use]
    pub fn from_depth(depth: f64, given: bool) -> Self {
        Self { depth: depth.is_finite().then_some(depth), given }
    }
}

impl PartialEq for DpNormDivisor {
    fn eq(&self, other: &Self) -> bool {
        self.given == other.given && self.depth.map(f64::to_bits) == other.depth.map(f64::to_bits)
    }
}

impl Eq for DpNormDivisor {}

/// The features derived from raw record values, one value per record in each column.
#[derive(Debug, Clone, Default)]
pub struct DerivedFeatures {
    /// AD of the first non-`*` alternate over AD\[0\] plus it (NaN if the total is not positive).
    pub ab: Vec<f64>,
    /// GATK's StrandOddsRatio of FORMAT/SB (see [`strand_odds_ratio`]), or else the record's
    /// [`crate::gvcf::RawAnnotation::sor`] (INFO/SOR, for the VCF feature source only).
    pub sor: Vec<f64>,
    /// 1 if the genotype is `0/1`, else 0.
    pub het: Vec<f64>,
    /// True if the genotype can be flipped (`0/1` or `1/1`).
    pub flippable: Vec<bool>,
    /// `sqrt(RAW_MQ / DP)` from RAW_MQandDP (NaN if the depth is not positive).
    pub mq: Vec<f64>,
    /// `QUALapprox / VarDP` (NaN if VarDP is not positive).
    pub qd: Vec<f64>,
    /// The alternate alleles' share of AS_RAW_MQ (NaN if the total is not positive).
    pub as_alt_frac: Vec<f64>,
    /// The RAW_MQandDP depth over the median of that depth across the records, or over a given
    /// depth (see [`derive_features`]); NaN if the divisor is not positive.
    pub dp_norm: Vec<f64>,
    /// The length change of the first non-`*` alternate allele.
    pub indel_len: Vec<f64>,
    /// The absolute length change of the first non-`*` alternate allele.
    pub indel_abs_len: Vec<f64>,
    /// Per region set, `(name, flags)`: 1 if the record's REF span overlaps the set, else 0.
    pub regions: Vec<(String, Vec<f64>)>,
    /// The number of alternate alleles in the record.
    pub n_alt_rec: Vec<f64>,
    /// True if the record's REF span overlaps the REF span of a PASS record (itself included).
    /// Not a model feature.
    pub overlaps_pass: Vec<bool>,
    /// The neighbourhood features, in [`NEIGHBOURHOOD_FEATURES`] order.
    pub neighbourhood: Vec<[f64; N_FEATURES]>,
    /// The depth `dp_norm` divides by. Not a column: a feature table records it in its metadata.
    pub dp_norm_divisor: DpNormDivisor,
}

/// Derives the model features of raw records.
///
/// The raw annotations (RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ) are those of the records'
/// feature source: a reblocked gVCF's, or the stand-ins that [`crate::vcf_source`] derives from a
/// standard VCF (where the RAW_MQandDP depth is INFO/DP, and `mq` is INFO/MQ), with, for the
/// alignments source, the mapping quality sums and depth measured from the reads instead (see
/// [`crate::read_annotations::MapqSums`]: the depth is the reads counted over the record, and
/// `mq` their root mean square MAPQ). The features of every source are computed alike.
///
/// The REF span of a record is `[pos - 1, pos - 1 + len(REF))` (0-based, half-open), so a
/// deletion overlaps a region through its deleted bases.
///
/// `dp_median` is the depth `dp_norm` divides by in place of the median RAW_MQandDP depth of
/// `records` (e.g. for calls of only a target footprint, whose median is not the genome-wide one). It must pass [`check_dp_median`] (callers check it;
/// debug builds assert it). The divisor, and with a given one the records' own median, is logged,
/// and returned as [`DerivedFeatures::dp_norm_divisor`].
#[must_use]
pub fn derive_features(
    records: &[RawRecord],
    regions: &[RegionSet],
    dp_median: Option<f64>,
) -> DerivedFeatures {
    let positive_or_nan = |value: f64| if value > 0.0 { value } else { f64::NAN };
    // Each column is computed in parallel over the records, keeping their order.
    let column = |f: &(dyn Fn(&RawRecord) -> f64 + Sync)| -> Vec<f64> {
        records.par_iter().map(f).collect()
    };
    let ab = column(&|r| r.ad_alt / positive_or_nan(r.ad_ref + r.ad_alt));
    let indel_len = column(&|r| {
        first_alt_index(&r.alts)
            .map_or(f64::NAN, |index| r.alts[index].len() as f64 - r.reference.len() as f64)
    });
    debug_assert!(
        dp_median.is_none_or(|given| check_dp_median(given).is_ok()),
        "unchecked dp_median {dp_median:?}"
    );
    let calls_median = || median(records.iter().map(|record| record.raw.raw_dp));
    let median_dp = match dp_median {
        None => {
            let median_dp = calls_median();
            log::info!("dp_norm divides each call's depth by the calls' median depth, {median_dp}");
            median_dp
        }
        Some(given) => {
            // The calls' own median only for the log, as a check on the given depth.
            if log::log_enabled!(log::Level::Info) {
                log::info!(
                    "dp_norm divides each call's depth by the given depth median, {given}, not the \
                     calls' median depth, {}",
                    calls_median()
                );
            }
            given
        }
    };
    let mut derived = DerivedFeatures {
        sor: column(&|r| {
            let from_sb = strand_odds_ratio(r.sb_ref_fwd, r.sb_ref_rev, r.sb_alt_fwd, r.sb_alt_rev);
            // Only the VCF and alignments feature sources set a fallback (INFO/SOR); FORMAT/SB is
            // preferred.
            if from_sb.is_nan() { r.raw.sor } else { from_sb }
        }),
        het: column(&|r| f64::from(u8::from(r.genotype == Genotype::HET))),
        flippable: records.par_iter().map(|r| r.genotype.is_flippable()).collect(),
        mq: column(&|r| (r.raw.raw_mq / positive_or_nan(r.raw.raw_dp)).sqrt()),
        qd: column(&|r| r.raw.qual_approx / positive_or_nan(r.raw.var_dp)),
        as_alt_frac: column(&|r| {
            r.raw.as_mq_alt / positive_or_nan(r.raw.as_mq_ref + r.raw.as_mq_alt)
        }),
        indel_abs_len: indel_len.iter().map(|len| len.abs()).collect(),
        indel_len,
        n_alt_rec: column(&|r| r.alts.len() as f64),
        dp_norm: column(&|r| if median_dp > 0.0 { r.raw.raw_dp / median_dp } else { f64::NAN }),
        dp_norm_divisor: DpNormDivisor::from_depth(median_dp, dp_median.is_some()),
        ab,
        ..DerivedFeatures::default()
    };

    derived.regions = regions
        .iter()
        .map(|region_set| {
            let flags = records
                .par_iter()
                .map(|record| {
                    let (start, end) = ref_span(record);
                    f64::from(u8::from(region_set.overlaps(&record.chrom, start, end)))
                })
                .collect();
            (region_set.name().to_string(), flags)
        })
        .collect();
    derived.overlaps_pass = overlaps_pass(records);

    let calls: Vec<Call<'_>> = records
        .iter()
        .zip(&derived.ab)
        .map(|(record, &ab)| Call {
            chrom: &record.chrom,
            pos: record.pos,
            het: matches!(record.genotype.alleles(), Some((0, alt)) if alt > 0),
            ab,
            filtered: record.filter != PASS,
        })
        .collect();
    derived.neighbourhood = neighbourhood_features(&calls);
    derived
}

/// The 0-based, half-open REF span of a record (see [`crate::regions::ref_span`]).
fn ref_span(record: &RawRecord) -> (u64, u64) {
    let pos = i64::try_from(record.pos).unwrap_or(i64::MAX);
    crate::regions::ref_span(pos, record.reference.len())
}

/// Flags the records whose REF span overlaps the REF span of any PASS record (itself included).
fn overlaps_pass(records: &[RawRecord]) -> Vec<bool> {
    let pass_spans = records.iter().filter(|record| record.filter == PASS).map(|record| {
        let (start, end) = ref_span(record);
        (record.chrom.as_ref(), start, end)
    });
    let pass_set = RegionSet::from_spans("pass", pass_spans);
    records
        .par_iter()
        .map(|record| {
            let (start, end) = ref_span(record);
            pass_set.overlaps(&record.chrom, start, end)
        })
        .collect()
}

/// The median of the non-NaN values (the mean of the middle two for an even count), or NaN if
/// there are none.
fn median<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut values: Vec<f64> = values.into_iter().filter(|value| !value.is_nan()).collect();
    if values.is_empty() {
        return f64::NAN;
    }
    // A selection, not a sort: the values below the middle one are only partitioned.
    let odd = values.len() % 2 == 1;
    let middle = values.len() / 2;
    let (below, &mut upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if odd {
        return upper;
    }
    let lower = below.iter().copied().max_by(f64::total_cmp).expect("an even count of at least 2");
    (lower + upper) / 2.0
}

#[cfg(test)]
mod tests {

    use rstest::rstest;

    use super::*;
    use crate::gvcf::RawAnnotation;

    /// A record with every numeric value missing, and the given keys.
    fn record(chrom: &str, pos: u64, reference: &str, alts: &[&str], filter: &str) -> RawRecord {
        RawRecord {
            filter: filter.to_string(),
            ..crate::testing::raw_record_with_missing_values(chrom, pos, reference, alts)
        }
    }

    fn snp(pos: u64) -> RawRecord {
        record("chr1", pos, "A", &["C"], PASS)
    }

    #[test]
    fn feature_names_by_type_and_task() {
        let none: [&str; 0] = [];
        let snp = feature_names(VariantType::Snp, &none, false, Task::Genotype, &none);
        assert_eq!(snp, COMMON_FEATURES.to_vec());
        let indel =
            feature_names(VariantType::Indel, &["segdups"], true, Task::Promote, &["len_a"]);
        let mut expected: Vec<&str> = COMMON_FEATURES.to_vec();
        expected.extend(PROMOTE_FEATURES);
        expected.extend(INDEL_FEATURES);
        expected.push("region_segdups");
        expected.extend(NEIGHBOURHOOD_FEATURES);
        expected.push("len_a");
        assert_eq!(indel, expected);
    }

    #[test]
    fn strand_odds_ratio_matches_gatk() {
        // The worked example in GATK's StrandOddsRatio docs: SB=1450,345,160,212 gives 0.592.
        assert!((strand_odds_ratio(1450.0, 345.0, 160.0, 212.0) - 0.592).abs() < 5e-4);
        assert!((strand_odds_ratio(10.0, 10.0, 10.0, 10.0) - 2f64.ln()).abs() < 1e-12);
        assert!(strand_odds_ratio(5.0, 5.0, f64::NAN, 6.0).is_nan());
    }

    #[rstest]
    #[case::odd(&[3.0, 1.0, 2.0], 2.0)]
    #[case::even(&[4.0, 1.0, 3.0, 2.0], 2.5)]
    #[case::ignores_nan(&[f64::NAN, 5.0], 5.0)]
    #[case::two(&[2.0, 1.0], 1.5)]
    #[case::repeated(&[7.0, 1.0, 7.0, 1.0, 9.0, 1.0], 4.0)]
    fn medians(#[case] values: &[f64], #[case] expected: f64) {
        assert_eq!(median(values.iter().copied()), expected);
    }

    #[test]
    fn median_of_nothing_is_nan() {
        assert!(median([f64::NAN]).is_nan());
        assert!(median([]).is_nan());
    }

    #[test]
    fn derives_per_record_features() {
        let mut first = snp(100);
        (first.ad_ref, first.ad_alt) = (6.0, 2.0);
        (first.sb_ref_fwd, first.sb_ref_rev, first.sb_alt_fwd, first.sb_alt_rev) =
            (10.0, 10.0, 10.0, 10.0);
        first.raw = RawAnnotation {
            raw_mq: 3600.0,
            raw_dp: 4.0,
            qual_approx: 30.0,
            var_dp: 2.0,
            as_mq_ref: 30.0,
            as_mq_alt: 10.0,
            n_alt: 1.0,
            sor: f64::NAN,
        };
        let mut deletion = record("chr1", 200, "ACG", &["*", "A"], "LowQual");
        deletion.genotype = Genotype::HOM_ALT;
        (deletion.ad_ref, deletion.ad_alt) = (0.0, 0.0);
        deletion.raw.raw_dp = 2.0;
        deletion.raw.var_dp = 0.0;
        let mut missing_gt = snp(300);
        missing_gt.genotype = Genotype::MISSING;
        missing_gt.raw.raw_dp = 0.0;

        let derived = derive_features(&[first, deletion, missing_gt], &[], None);
        assert_eq!(derived.ab[0], 0.25);
        assert!(derived.ab[1].is_nan(), "zero total depth");
        assert!((derived.sor[0] - 2f64.ln()).abs() < 1e-12);
        assert_eq!(derived.het, [1.0, 0.0, 0.0]);
        assert_eq!(derived.flippable, [true, true, false]);
        assert_eq!(derived.mq[0], 30.0);
        assert!(derived.mq[2].is_nan(), "zero depth");
        assert_eq!(derived.qd[0], 15.0);
        assert!(derived.qd[1].is_nan(), "zero VarDP");
        assert_eq!(derived.as_alt_frac[0], 0.25);
        // The median depth of 4, 2 and 0 is 2.
        assert_eq!(derived.dp_norm, [2.0, 1.0, 0.0]);
        assert_eq!(derived.indel_len[..2], [0.0, -2.0]);
        assert_eq!(derived.indel_abs_len[1], 2.0);
        assert_eq!(derived.n_alt_rec, [1.0, 2.0, 1.0]);
        assert_eq!(derived.neighbourhood.len(), 3);
    }

    #[test]
    fn dp_norm_is_missing_without_a_positive_median() {
        let derived = derive_features(&[snp(1), snp(2)], &[], None);
        assert!(derived.dp_norm.iter().all(|value| value.is_nan()));
        assert!(derive_features(&[], &[], None).dp_norm.is_empty());
    }

    /// When more than half the calls have a depth of 0 (e.g. no read counted), their median is 0,
    /// and every call's `dp_norm` is missing, the calls with a depth too; a given median divides
    /// them.
    #[test]
    fn dp_norm_is_missing_for_every_call_when_the_median_depth_is_0() {
        let records: Vec<RawRecord> = [0.0, 0.0, 5.0]
            .iter()
            .zip(1..)
            .map(|(&depth, pos)| {
                let mut call = snp(pos);
                call.raw.raw_dp = depth;
                call
            })
            .collect();
        let derived = derive_features(&records, &[], None);
        assert!(derived.dp_norm.iter().all(|value| value.is_nan()), "{:?}", derived.dp_norm);
        assert_eq!(derived.dp_norm_divisor, DpNormDivisor::from_depth(0.0, false));
        assert_eq!(derive_features(&records, &[], Some(10.0)).dp_norm, [0.0, 0.0, 0.5]);
    }

    /// With an even count, the median averages the middle two depths: when exactly half the
    /// calls have a depth of 0, the median is positive and `dp_norm` is present.
    #[test]
    fn dp_norm_is_present_when_exactly_half_the_depths_are_0() {
        let records: Vec<RawRecord> = [0.0, 0.0, 5.0, 5.0]
            .iter()
            .zip(1..)
            .map(|(&depth, pos)| {
                let mut call = snp(pos);
                call.raw.raw_dp = depth;
                call
            })
            .collect();
        let derived = derive_features(&records, &[], None);
        // The median of 0, 0, 5 and 5 is 2.5.
        assert_eq!(derived.dp_norm, [0.0, 0.0, 2.0, 2.0]);
        assert_eq!(derived.dp_norm_divisor, DpNormDivisor::from_depth(2.5, false));
    }

    #[test]
    fn dp_norm_divides_by_a_given_median() {
        let depths = [4.0, 2.0, 0.0, f64::NAN];
        let records: Vec<RawRecord> = depths
            .iter()
            .zip(1..)
            .map(|(&depth, pos)| {
                let mut call = snp(pos);
                call.raw.raw_dp = depth;
                call
            })
            .collect();
        let given = derive_features(&records, &[], Some(8.0)).dp_norm;
        assert_eq!(given[..3], [0.5, 0.25, 0.0]);
        assert!(given[3].is_nan(), "a missing depth stays missing");
        // Without one, the median of the depths (4, 2 and 0: 2), as before.
        assert_eq!(derive_features(&records, &[], None).dp_norm[..3], [2.0, 1.0, 0.0]);
        // A given median applies even where the calls' own median is not positive.
        let zero = derive_features(&records[2..], &[], Some(10.0)).dp_norm;
        assert_eq!(zero[0], 0.0);
    }

    #[test]
    fn returns_the_dp_norm_divisor() {
        let mut deep = snp(1);
        deep.raw.raw_dp = 6.0;
        let mut shallow = snp(2);
        shallow.raw.raw_dp = 0.0;
        let calls = [deep, shallow]; // depths 6 and 0: median 3
        let divisor =
            |records: &[RawRecord], given| derive_features(records, &[], given).dp_norm_divisor;
        assert_eq!(divisor(&calls, None), DpNormDivisor { depth: Some(3.0), given: false });
        assert_eq!(divisor(&calls, Some(8.0)), DpNormDivisor { depth: Some(8.0), given: true });
        // No call has a depth: the median is NaN, recorded as no depth.
        assert_eq!(divisor(&[snp(3)], None), DpNormDivisor { depth: None, given: false });
        assert_eq!(divisor(&[], None), DpNormDivisor::default());
    }

    #[test]
    fn region_flags_use_the_ref_span() {
        let regions = RegionSet::from_spans("segdups", [("chr1", 100, 110)]);
        let records = [
            record("chr1", 100, "AC", &["A"], PASS), // anchored before, deletes base 100
            record("chr1", 100, "A", &["C"], PASS),  // base 99, just before the region
            record("chr1", 110, "A", &["C"], PASS),  // base 109, the region's last
            record("chr2", 105, "A", &["C"], PASS),  // another contig
        ];
        let derived = derive_features(&records, &[regions], None);
        assert_eq!(derived.regions, [("segdups".to_string(), vec![1.0, 0.0, 1.0, 0.0])]);
    }

    #[test]
    fn overlaps_pass_uses_ref_spans() {
        let records = [
            record("chr1", 100, "ACGT", &["A"], PASS), // PASS deletion over [99, 103)
            record("chr1", 102, "G", &["C"], "LowQual"), // inside the PASS deletion's span
            record("chr1", 104, "A", &["C"], "LowQual"), // adjacent to it, not overlapping
            record("chr1", 200, "ACGTA", &["A"], "LowQual"), // deletion over [199, 204) ...
            record("chr1", 203, "T", &["C"], PASS),    // ... covering a later PASS SNP
            record("chr2", 102, "G", &["C"], "LowQual"), // same position, other contig
        ];
        let derived = derive_features(&records, &[], None);
        assert_eq!(derived.overlaps_pass, [true, true, false, true, true, false]);
    }

    #[test]
    fn neighbourhood_counts_het_any_and_filtered_calls() {
        let mut multi = snp(100);
        multi.genotype = Genotype::diploid(0, 2);
        let mut hom = snp(200);
        hom.genotype = Genotype::HOM_ALT;
        let filtered = record("chr1", 300, "A", &["C"], "LowQual");
        let derived = derive_features(&[multi, hom, filtered], &[], None);
        let names = NEIGHBOURHOOD_FEATURES;
        let column = |name: &str| names.iter().position(|n| *n == name).unwrap();
        // From 200: the 0/2 call is a het; the 1/1 call itself is not counted.
        assert_eq!(derived.neighbourhood[1][column("nb_het_500")], 2.0);
        assert_eq!(derived.neighbourhood[1][column("nb_filt_500")], 1.0);
        assert_eq!(derived.neighbourhood[0][column("nb_het_500")], 1.0);
    }
}
