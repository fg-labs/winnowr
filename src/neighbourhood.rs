//! Multi-site "neighbourhood" features computed from all calls near each call.
//!
//! Collapsed or mismapped paralogous sequence shows up as clusters of calls, often low allele
//! balance hets, so the calls around a site carry information about whether the site itself is
//! real.

use std::collections::HashMap;

use rayon::prelude::*;

/// The half-window sizes (bp) and the suffixes used in feature names.
pub const WINDOWS: [(u64, &str); 2] = [(500, "500"), (2000, "2k")];

/// Hets with an allele balance below this are counted as low-AB hets.
pub const LOW_AB: f64 = 0.3;

/// The number of neighbourhood features.
pub const N_FEATURES: usize = 11;

/// The feature names, in model order: per window, then the distance to the nearest call.
pub const NEIGHBOURHOOD_FEATURES: [&str; N_FEATURES] = [
    "nb_calls_500",
    "nb_het_500",
    "nb_lowab_500",
    "nb_filt_500",
    "nb_ab_500",
    "nb_calls_2k",
    "nb_het_2k",
    "nb_lowab_2k",
    "nb_filt_2k",
    "nb_ab_2k",
    "nb_dist",
];

/// What the neighbourhood features need to know about one call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Call<'a> {
    /// The contig.
    pub chrom: &'a str,
    /// The position.
    pub pos: u64,
    /// True for a het genotype `0/x` with `x > 0`.
    pub het: bool,
    /// The allele balance (NaN if undefined).
    pub ab: f64,
    /// True for a non-PASS call.
    pub filtered: bool,
}

/// Computes the neighbourhood features of every call, in [`NEIGHBOURHOOD_FEATURES`] order.
///
/// For each call and each window `±W` bp (inclusive), counting only *other* positions:
/// `nb_calls` calls, `nb_het` het calls, `nb_lowab` hets with AB below [`LOW_AB`], `nb_filt`
/// non-PASS calls, and `nb_ab` the mean AB of hets with a defined AB (NaN if none). `nb_dist` is
/// the distance to the nearest other call position on the contig (NaN if none). Calls sharing a
/// position are counted once, as the first of them, and all get that position's features.
#[must_use]
pub fn neighbourhood_features(calls: &[Call<'_>]) -> Vec<[f64; N_FEATURES]> {
    // Contigs are independent: compute them in parallel, then place each call's row.
    let per_contig: Vec<(Vec<usize>, Vec<[f64; N_FEATURES]>)> = contig_groups(calls)
        .into_par_iter()
        .map(|indices| {
            let rows = contig_features(calls, &indices);
            (indices, rows)
        })
        .collect();
    let mut features = vec![[f64::NAN; N_FEATURES]; calls.len()];
    for (indices, rows) in per_contig {
        for (index, row) in indices.into_iter().zip(rows) {
            features[index] = row;
        }
    }
    features
}

/// The indices of the calls on each contig, in order of first appearance.
fn contig_groups(calls: &[Call<'_>]) -> Vec<Vec<usize>> {
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (index, call) in calls.iter().enumerate() {
        let group = *group_of.entry(call.chrom).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(index);
    }
    groups
}

/// Computes the features of one contig's calls (`indices`), one row per index.
fn contig_features(calls: &[Call<'_>], indices: &[usize]) -> Vec<[f64; N_FEATURES]> {
    // Distinct positions in increasing order, each represented by its first call.
    let mut sorted = indices.to_vec();
    sorted.sort_by_key(|&index| calls[index].pos); // stable: keeps the first call first
    sorted.dedup_by_key(|&mut index| calls[index].pos);
    let pos: Vec<u64> = sorted.iter().map(|&index| calls[index].pos).collect();

    // Each position's own contribution to the windowed sums, and their prefix sums.
    let own =
        |key: Own| -> Vec<f64> { sorted.iter().map(|&index| key.value(&calls[index])).collect() };
    let owns: [Vec<f64>; 5] = Own::ALL.map(own);
    let prefix: [Vec<f64>; 5] = std::array::from_fn(|k| prefix_sums(&owns[k]));

    let mut distinct = vec![[f64::NAN; N_FEATURES]; pos.len()];
    for (i, row) in distinct.iter_mut().enumerate() {
        for (w, &(half_window, _)) in WINDOWS.iter().enumerate() {
            let lo = pos.partition_point(|&p| p < pos[i].saturating_sub(half_window));
            let hi = pos.partition_point(|&p| p <= pos[i] + half_window);
            let others = |k: usize| prefix[k][hi] - prefix[k][lo] - owns[k][i];
            let n_ab = others(Own::AbCount as usize);
            let column = w * 5;
            row[column] = (hi - lo - 1) as f64;
            row[column + 1] = others(Own::Het as usize);
            row[column + 2] = others(Own::LowAb as usize);
            row[column + 3] = others(Own::Filtered as usize);
            row[column + 4] =
                if n_ab > 0.0 { others(Own::AbSum as usize) / n_ab } else { f64::NAN };
        }
        let left = i.checked_sub(1).map(|j| (pos[i] - pos[j]) as f64);
        let right = pos.get(i + 1).map(|&next| (next - pos[i]) as f64);
        row[N_FEATURES - 1] = match (left, right) {
            (Some(l), Some(r)) => l.min(r),
            (Some(d), None) | (None, Some(d)) => d,
            (None, None) => f64::NAN,
        };
    }

    // Every call takes the features of its position (`pos` is sorted and distinct).
    indices
        .iter()
        .map(|&index| {
            let row = pos.binary_search(&calls[index].pos).expect("every position is in `pos`");
            distinct[row]
        })
        .collect()
}

/// `[0, v0, v0 + v1, ...]`, summed left to right.
fn prefix_sums(values: &[f64]) -> Vec<f64> {
    let mut sums = Vec::with_capacity(values.len() + 1);
    let mut total = 0.0;
    sums.push(total);
    for &value in values {
        total += value;
        sums.push(total);
    }
    sums
}

/// The per-position values summed over each window.
#[derive(Debug, Clone, Copy)]
enum Own {
    /// 1 for a het.
    Het,
    /// 1 for a het with AB below [`LOW_AB`].
    LowAb,
    /// 1 for a non-PASS call.
    Filtered,
    /// 1 for a het with a defined AB.
    AbCount,
    /// The AB of a het with a defined AB, else 0.
    AbSum,
}

impl Own {
    const ALL: [Own; 5] = [Own::Het, Own::LowAb, Own::Filtered, Own::AbCount, Own::AbSum];

    fn value(self, call: &Call<'_>) -> f64 {
        let het = f64::from(u8::from(call.het));
        let has_ab = call.het && !call.ab.is_nan();
        match self {
            Own::Het => het,
            Own::LowAb => {
                // A het without an AB is not a low-AB het.
                let ab = if call.ab.is_nan() { 1.0 } else { call.ab };
                het * f64::from(u8::from(ab < LOW_AB))
            }
            Own::Filtered => f64::from(u8::from(call.filtered)),
            Own::AbCount => f64::from(u8::from(has_ab)),
            Own::AbSum => {
                if has_ab {
                    call.ab
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(chrom: &str, pos: u64, het: bool, ab: f64, filtered: bool) -> Call<'_> {
        Call { chrom, pos, het, ab, filtered }
    }

    fn feature(row: &[f64; N_FEATURES], name: &str) -> f64 {
        row[NEIGHBOURHOOD_FEATURES.iter().position(|n| *n == name).unwrap()]
    }

    #[test]
    fn feature_names_follow_the_windows() {
        let mut expected: Vec<String> = Vec::new();
        for (_, tag) in WINDOWS {
            for key in ["calls", "het", "lowab", "filt", "ab"] {
                expected.push(format!("nb_{key}_{tag}"));
            }
        }
        expected.push("nb_dist".to_string());
        assert_eq!(NEIGHBOURHOOD_FEATURES.to_vec(), expected);
    }

    #[test]
    fn window_boundaries_are_inclusive() {
        // Neighbours of 1000 at exactly -500 and +500 count; at +501 they do not (500 window).
        let calls = [
            call("chr1", 500, true, 0.2, false),
            call("chr1", 1000, true, 0.5, false),
            call("chr1", 1500, false, 1.0, true),
            call("chr1", 3001, true, 0.4, false),
        ];
        let features = neighbourhood_features(&calls);
        let row = &features[1];
        assert_eq!(feature(row, "nb_calls_500"), 2.0);
        assert_eq!(feature(row, "nb_het_500"), 1.0);
        assert_eq!(feature(row, "nb_lowab_500"), 1.0);
        assert_eq!(feature(row, "nb_filt_500"), 1.0);
        assert!((feature(row, "nb_ab_500") - 0.2).abs() < 1e-12);
        assert_eq!(feature(row, "nb_calls_2k"), 2.0); // 3001 is 2001 away
        assert_eq!(feature(row, "nb_dist"), 500.0);
        let far = &features[3];
        assert_eq!(feature(far, "nb_calls_500"), 0.0);
        assert!(feature(far, "nb_ab_500").is_nan());
        assert_eq!(feature(far, "nb_calls_2k"), 1.0); // 1500 is 1501 away
        assert_eq!(feature(far, "nb_dist"), 1501.0);
    }

    #[test]
    fn mean_ab_ignores_hets_without_ab_and_the_own_site() {
        let calls = [
            call("chr1", 100, true, 0.9, false),
            call("chr1", 200, true, f64::NAN, false),
            call("chr1", 300, true, 0.1, false),
        ];
        let features = neighbourhood_features(&calls);
        assert!((feature(&features[1], "nb_ab_500") - 0.5).abs() < 1e-12);
        assert_eq!(feature(&features[0], "nb_het_500"), 2.0);
        assert!((feature(&features[0], "nb_ab_500") - 0.1).abs() < 1e-12);
        // A het without AB is not a low-AB het.
        assert_eq!(feature(&features[0], "nb_lowab_500"), 1.0);
    }

    #[test]
    fn contigs_are_independent_and_duplicate_positions_collapsed() {
        let calls = [
            call("chr1", 100, true, 0.5, false),
            call("chr1", 100, false, 1.0, true),
            call("chr2", 120, true, 0.5, false),
            call("chr1", 150, false, f64::NAN, false),
        ];
        let features = neighbourhood_features(&calls);
        // The second call at chr1:100 takes the features of the first.
        assert_eq!(features[0].map(f64::to_bits), features[1].map(f64::to_bits));
        // From 150, the one distinct position at 100 counts once, as the first (unfiltered) call.
        assert_eq!(feature(&features[3], "nb_calls_500"), 1.0);
        assert_eq!(feature(&features[3], "nb_filt_500"), 0.0);
        assert_eq!(feature(&features[3], "nb_het_500"), 1.0);
        assert_eq!(feature(&features[0], "nb_dist"), 50.0);
        assert_eq!(feature(&features[2], "nb_calls_2k"), 0.0);
        assert!(feature(&features[2], "nb_dist").is_nan());
    }

    #[test]
    fn unsorted_calls_are_handled() {
        let calls =
            [call("chr1", 300, false, f64::NAN, false), call("chr1", 100, false, f64::NAN, false)];
        let features = neighbourhood_features(&calls);
        assert_eq!(feature(&features[0], "nb_dist"), 200.0);
        assert_eq!(feature(&features[1], "nb_calls_500"), 1.0);
        assert!(neighbourhood_features(&[]).is_empty());
    }
}
