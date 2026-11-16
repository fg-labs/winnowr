//! Repeat-length features: the strlen read evidence joined to the calls, and histogram summaries.
//!
//! [`add_strlen_features`] joins strlen tables (see [`crate::strlen`]) to a feature table by
//! `(chrom, pos, ref, alt)` and derives [`STRLEN_FEATURES`] from each call's length histogram and
//! its called genotype. Calls without strlen evidence (SNPs, or indels outside the measured
//! shards) get missing values. Features that need a stutter model live in [`crate::stutter`].

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail};
use arrow_array::{ArrayRef, StringArray};
use log::{info, warn};
use rayon::prelude::*;

use crate::alleles::{Genotype, allele_deltas};
use crate::common::VariantType;
use crate::evidence::Evidence as _;
use crate::numeric::pairwise_sum;
use crate::strlen::{Histogram, StrlenOptions, StrlenRow, StrlenTables, parse_histogram};
use crate::table::FeatureTable;

/// The features taken or derived directly from the strlen table columns.
const TABLE_FEATURES: [&str; 9] = [
    "sl_period",
    "sl_tract_len",
    "sl_tract_units",
    "sl_n_overlap",
    "sl_n_spanning",
    "sl_frac_not_spanning",
    "sl_mean_mapq",
    "sl_frac_mapq_lt20",
    "sl_mean_tract_bq",
];

/// The features computed from each call's length histogram (see [`histogram_features`]).
pub const HISTOGRAM_FEATURES: [&str; 14] = [
    "sl_top1_delta",
    "sl_top1_frac",
    "sl_top2_delta",
    "sl_top2_frac",
    "sl_top3_frac",
    "sl_n_lengths",
    "sl_frac_ref",
    "sl_entropy",
    "sl_sd",
    "sl_frac_alt1",
    "sl_frac_alt2",
    "sl_frac_called",
    "sl_top1_vs_alt",
    "sl_top2_vs_alt",
];

/// The model features from the strlen evidence (no stutter model needed), in model order.
pub const STRLEN_FEATURES: [&str; 23] = [
    TABLE_FEATURES[0],
    TABLE_FEATURES[1],
    TABLE_FEATURES[2],
    TABLE_FEATURES[3],
    TABLE_FEATURES[4],
    TABLE_FEATURES[5],
    TABLE_FEATURES[6],
    TABLE_FEATURES[7],
    TABLE_FEATURES[8],
    HISTOGRAM_FEATURES[0],
    HISTOGRAM_FEATURES[1],
    HISTOGRAM_FEATURES[2],
    HISTOGRAM_FEATURES[3],
    HISTOGRAM_FEATURES[4],
    HISTOGRAM_FEATURES[5],
    HISTOGRAM_FEATURES[6],
    HISTOGRAM_FEATURES[7],
    HISTOGRAM_FEATURES[8],
    HISTOGRAM_FEATURES[9],
    HISTOGRAM_FEATURES[10],
    HISTOGRAM_FEATURES[11],
    HISTOGRAM_FEATURES[12],
    HISTOGRAM_FEATURES[13],
];

/// The sorted length changes of a call's genotype alleles, from its feature-table values.
///
/// These are length genotypes: an alternate allele as long as the reference (e.g. `T` in
/// `A>T,AT`) has length change 0, as the reference has.
///
/// # Arguments
///
/// * `reference` - the reference allele.
/// * `alt` - the comma-joined alternate alleles.
/// * `gt` - the normalized genotype.
///
/// Returns `None` if the genotype is missing or does not parse, or an allele is unusable (see
/// [`allele_deltas`]).
#[must_use]
pub fn called_deltas(reference: &str, alt: &str, gt: &str) -> Option<(i64, i64)> {
    let genotype: Genotype = gt.parse().ok()?;
    let alleles: Vec<&str> = std::iter::once(reference).chain(alt.split(',')).collect();
    allele_deltas(&alleles, genotype)
}

/// The `called_deltas` of every call in a feature table.
///
/// # Errors
///
/// Returns an error if the table lacks `ref`, `alt` or `gt`.
pub fn called_deltas_of(table: &FeatureTable) -> Result<Vec<Option<(i64, i64)>>> {
    let (refs, alts, gts) =
        (table.str_column("ref")?, table.str_column("alt")?, table.str_column("gt")?);
    Ok((0..table.num_rows()).map(|i| called_deltas(refs[i], alts[i], gts[i])).collect())
}

/// The observed length differences by decreasing read count (ties by increasing difference).
#[must_use]
pub fn ranked_lengths(histogram: &Histogram) -> Vec<(i64, u64)> {
    let mut ranked: Vec<(i64, u64)> = histogram.iter().map(|(&d, &n)| (d, n)).collect();
    ranked.sort_by_key(|&(delta, count)| (std::cmp::Reverse(count), delta));
    ranked
}

/// Summaries of one call's length histogram, in [`HISTOGRAM_FEATURES`] order (all NaN for an
/// empty histogram).
///
/// # Arguments
///
/// * `histogram` - reads per length difference (bp, observed minus reference tract).
/// * `period` - the tract's repeat period.
/// * `called` - the called genotype's length changes, if known; without them the features about
///   the called alleles are NaN.
#[must_use]
pub fn histogram_features(
    histogram: &Histogram,
    period: u64,
    called: Option<(i64, i64)>,
) -> [f64; HISTOGRAM_FEATURES.len()] {
    let mut features = [f64::NAN; HISTOGRAM_FEATURES.len()];
    let n: u64 = histogram.values().sum();
    if n == 0 {
        return features;
    }
    let n_f = n as f64;
    let frac = |count: u64| count as f64 / n_f;
    let count_of = |delta: i64| histogram.get(&delta).copied().unwrap_or(0);
    let ranked = ranked_lengths(histogram);
    features[0] = ranked[0].0 as f64;
    features[1] = frac(ranked[0].1);
    features[2] = ranked.get(1).map_or(f64::NAN, |&(delta, _)| delta as f64);
    features[3] = ranked.get(1).map_or(0.0, |&(_, count)| frac(count));
    features[4] = ranked.get(2).map_or(0.0, |&(_, count)| frac(count));
    features[5] = ranked.len() as f64;
    features[6] = frac(count_of(0));
    let fractions: Vec<f64> = ranked.iter().map(|&(_, count)| frac(count)).collect();
    let entropy_terms: Vec<f64> = fractions.iter().map(|&f| f * f.log2()).collect();
    features[7] = -pairwise_sum(&entropy_terms);
    let deltas: Vec<f64> = ranked.iter().map(|&(delta, _)| delta as f64).collect();
    let weighted: Vec<f64> = deltas.iter().zip(&fractions).map(|(d, f)| d * f).collect();
    let mean = pairwise_sum(&weighted);
    let squares: Vec<f64> =
        deltas.iter().zip(&fractions).map(|(d, f)| (d - mean).powi(2) * f).collect();
    features[8] = pairwise_sum(&squares).max(0.0).sqrt();
    if let Some((low, high)) = called {
        let mut alts: Vec<i64> = [low, high].into_iter().filter(|&d| d != 0).collect();
        alts.sort_unstable();
        alts.dedup();
        let period = period as f64;
        features[9] = alts.first().map_or(f64::NAN, |&alt| frac(count_of(alt)));
        features[10] = alts.get(1).map_or(f64::NAN, |&alt| frac(count_of(alt)));
        let called_alleles: HashSet<i64> = [low, high].into_iter().collect();
        features[11] = frac(called_alleles.iter().map(|&d| count_of(d)).sum());
        features[12] = alts.first().map_or(f64::NAN, |&alt| (ranked[0].0 - alt) as f64 / period);
        features[13] = match (alts.first(), ranked.get(1)) {
            (Some(&alt), Some(&(second, _))) => (second - alt) as f64 / period,
            _ => f64::NAN,
        };
    }
    features
}

/// Joins strlen evidence to a feature table's calls by `(chrom, pos, ref, alt)` and derives
/// [`STRLEN_FEATURES`], marking the table as having strlen evidence measured with its settings.
///
/// Adds the strlen columns (`sl_period`, `sl_tract_len`, `sl_unit`, `sl_tract_start`,
/// `sl_n_overlap`, `sl_n_spanning`, `sl_histogram`, `sl_mean_mapq`, `sl_frac_mapq_lt20`,
/// `sl_mean_tract_bq`), then the derived features. Logs how many indel calls have evidence, and
/// warns about contigs whose indel calls have none (e.g. a missing shard).
///
/// # Errors
///
/// Returns an error if the table has indel calls and none of them has evidence (the strlen
/// tables come from another VCF), or a histogram is malformed.
pub fn add_strlen_features(table: FeatureTable, strlen: &StrlenTables) -> Result<FeatureTable> {
    let joined = table.join_calls(&strlen.rows, StrlenOptions::row_key)?;
    check_coverage(&table, &joined)?;
    let called = called_deltas_of(&table)?;

    let number = |f: fn(&StrlenRow) -> f64| -> Vec<f64> {
        joined.iter().map(|row| row.map_or(f64::NAN, f)).collect()
    };
    let text = |f: fn(&StrlenRow) -> &str| -> ArrayRef {
        Arc::new(joined.iter().map(|row| row.map(f)).collect::<StringArray>())
    };
    let period = number(|r| r.period as f64);
    let tract_len = number(|r| r.tract_len as f64);
    let n_overlap = number(|r| r.n_overlap as f64);
    let n_spanning = number(|r| r.n_spanning as f64);
    let positive_or_nan = |value: f64| if value > 0.0 { value } else { f64::NAN };
    // n_spanning counts only the measured reads, so spanning reads the read settings exclude
    // (below the minimum MAPQ, or a mate already measured) count as not spanning.
    let frac_not_spanning: Vec<f64> =
        n_overlap.iter().zip(&n_spanning).map(|(&o, &s)| 1.0 - s / positive_or_nan(o)).collect();
    let tract_units: Vec<f64> =
        tract_len.iter().zip(&period).map(|(&l, &p)| l / positive_or_nan(p)).collect();

    // Histograms are parsed and summarized in parallel, then split into columns.
    let rows: Vec<[f64; HISTOGRAM_FEATURES.len()]> = joined
        .par_iter()
        .zip(&called)
        .map(|(row, call)| match row {
            Some(row) => {
                Ok(histogram_features(&parse_histogram(&row.length_histogram)?, row.period, *call))
            }
            None => Ok([f64::NAN; HISTOGRAM_FEATURES.len()]),
        })
        .collect::<Result<_>>()?;
    let mut histogram_columns: Vec<Vec<f64>> =
        HISTOGRAM_FEATURES.iter().map(|_| Vec::with_capacity(joined.len())).collect();
    for features in rows {
        for (column, value) in histogram_columns.iter_mut().zip(features) {
            column.push(value);
        }
    }

    let mut table = table
        .with_f64_column("sl_period", period)?
        .with_f64_column("sl_tract_len", tract_len)?
        .with_column("sl_unit", text(|r| r.unit.as_str()))?
        .with_f64_column("sl_tract_start", number(|r| r.tract_start as f64))?
        .with_f64_column("sl_n_overlap", n_overlap)?
        .with_f64_column("sl_n_spanning", n_spanning)?
        .with_column("sl_histogram", text(|r| r.length_histogram.as_str()))?
        .with_f64_column("sl_mean_mapq", number(|r| r.mean_mapq))?
        .with_f64_column("sl_frac_mapq_lt20", number(|r| r.frac_mapq_lt20))?
        .with_f64_column("sl_mean_tract_bq", number(|r| r.mean_tract_bq))?
        .with_f64_column("sl_frac_not_spanning", frac_not_spanning)?
        .with_f64_column("sl_tract_units", tract_units)?;
    for (name, values) in HISTOGRAM_FEATURES.iter().zip(histogram_columns) {
        table = table.with_f64_column(name, values)?;
    }
    table.info_mut().strlen = true;
    table.info_mut().strlen_settings = Some(strlen.options);
    Ok(table)
}

/// Logs the indel calls with strlen evidence; fails if there are indel calls but none has any.
fn check_coverage(table: &FeatureTable, joined: &[Option<&StrlenRow>]) -> Result<()> {
    let (types, chroms) = (table.str_column("variant_type")?, table.str_column("chrom")?);
    let indel = VariantType::Indel.as_str();
    let n_indels = types.iter().filter(|&&t| t == indel).count();
    if n_indels == 0 {
        return Ok(());
    }
    let mut per_contig: Vec<(&str, usize)> = Vec::new();
    let mut n_with = 0;
    for i in (0..types.len()).filter(|&i| types[i] == indel) {
        let has = usize::from(joined[i].is_some());
        n_with += has;
        match per_contig.iter_mut().find(|(contig, _)| *contig == chroms[i]) {
            Some((_, count)) => *count += has,
            None => per_contig.push((chroms[i], has)),
        }
    }
    if n_with == 0 {
        bail!(
            "none of the {n_indels} indel calls has a row in the strlen tables: were they made \
             from this VCF?"
        );
    }
    info!(
        "strlen evidence for {n_with} of {n_indels} indel calls ({:.1}%)",
        100.0 * n_with as f64 / n_indels as f64
    );
    let missing: Vec<&str> =
        per_contig.iter().filter(|&&(_, count)| count == 0).map(|&(contig, _)| contig).collect();
    if !missing.is_empty() {
        warn!(
            "no strlen evidence for the indel calls on {} contig(s), e.g. a missing shard: {}",
            missing.len(),
            missing.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn histogram(items: &[(i64, u64)]) -> Histogram {
        items.iter().copied().collect()
    }

    #[test]
    fn strlen_features_are_the_table_then_histogram_features() {
        let expected: Vec<&str> =
            TABLE_FEATURES.iter().chain(&HISTOGRAM_FEATURES).copied().collect();
        assert_eq!(STRLEN_FEATURES.to_vec(), expected);
    }

    #[rstest]
    #[case::het_deletion("ACA", "A", "0/1", Some((-2, 0)))]
    #[case::multi_allelic("A", "AC,ACC", "1/2", Some((1, 2)))]
    #[case::missing("A", "AC", "./.", None)]
    #[case::unparseable("A", "AC", "0|1", None)]
    #[case::spanning_deletion("A", "*,AC", "1/2", None)]
    fn called_deltas_of_calls(
        #[case] reference: &str,
        #[case] alt: &str,
        #[case] gt: &str,
        #[case] expected: Option<(i64, i64)>,
    ) {
        assert_eq!(called_deltas(reference, alt, gt), expected);
    }

    #[test]
    fn ranks_lengths_by_count_then_difference() {
        let ranked = ranked_lengths(&histogram(&[(-1, 5), (0, 10), (1, 5), (2, 1)]));
        assert_eq!(ranked, [(0, 10), (-1, 5), (1, 5), (2, 1)]);
    }

    #[test]
    fn histogram_features_of_a_het_deletion() {
        // 10 reads at 0, 8 at -2, 2 at -1; called 0/-2 (a het 2 bp deletion), period 2.
        let features =
            histogram_features(&histogram(&[(-2, 8), (-1, 2), (0, 10)]), 2, Some((-2, 0)));
        let get =
            |name: &str| features[HISTOGRAM_FEATURES.iter().position(|n| *n == name).unwrap()];
        assert_eq!(get("sl_top1_delta"), 0.0);
        assert_eq!(get("sl_top1_frac"), 0.5);
        assert_eq!(get("sl_top2_delta"), -2.0);
        assert_eq!(get("sl_top2_frac"), 0.4);
        assert_eq!(get("sl_top3_frac"), 0.1);
        assert_eq!(get("sl_n_lengths"), 3.0);
        assert_eq!(get("sl_frac_ref"), 0.5);
        let entropy = -(0.5 * 0.5f64.log2() + 0.4 * 0.4f64.log2() + 0.1 * 0.1f64.log2());
        assert!((get("sl_entropy") - entropy).abs() < 1e-12);
        let mean: f64 = -2.0 * 0.4 - 0.1;
        let variance =
            0.5 * mean * mean + 0.4 * (-2.0 - mean).powi(2) + 0.1 * (-1.0 - mean).powi(2);
        assert!((get("sl_sd") - variance.sqrt()).abs() < 1e-12);
        assert_eq!(get("sl_frac_alt1"), 0.4);
        assert!(get("sl_frac_alt2").is_nan());
        assert_eq!(get("sl_frac_called"), 0.9);
        assert_eq!(get("sl_top1_vs_alt"), 1.0);
        assert_eq!(get("sl_top2_vs_alt"), 0.0);
    }

    #[test]
    fn histogram_features_without_a_called_genotype_or_reads() {
        let features = histogram_features(&histogram(&[(1, 3)]), 1, None);
        assert_eq!(features[0], 1.0);
        assert_eq!(features[3], 0.0, "no second length");
        assert!(features[2].is_nan());
        assert!(features[9..].iter().all(|v| v.is_nan()));
        assert!(histogram_features(&Histogram::new(), 1, Some((0, 1))).iter().all(|v| v.is_nan()));
    }
}
