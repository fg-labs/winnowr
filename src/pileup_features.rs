//! Pileup features: the allele read counts of pileup tables (see [`crate::pileup`]) joined to
//! the calls, and the features derived from them.

use anyhow::{Result, bail};
use log::{info, warn};
use rayon::prelude::*;

use crate::evidence::Evidence as _;
use crate::pileup::{ALT, AlleleCounts, OTHER, PileupOptions, PileupTables, REF};
use crate::table::FeatureTable;

/// The feature columns joined from the pileup counts (see [`add_pileup_features`]), in model
/// order.
pub const PILEUP_FEATURES: [&str; 12] = [
    "pu_ref_mq0",
    "pu_ref_mqlow",
    "pu_ref_mqhigh",
    "pu_alt_mq0",
    "pu_alt_mqlow",
    "pu_alt_mqhigh",
    "pu_other",
    "pu_ref_frac",
    "pu_ref_frac_mqhigh",
    "pu_ref_frac_mq0",
    "pu_frac_mq0",
    "pu_log_ref_mq0_alt_mqhigh",
];

/// The [`PILEUP_FEATURES`] of one row of counts.
///
/// The fractions are NaN when their denominator is 0: `pu_ref_frac` is the reference reads over
/// the reference and alternate reads at any MAPQ, `pu_ref_frac_mqhigh` the same in the high bin,
/// `pu_ref_frac_mq0` the reference reads at MAPQ 0 over all reference reads, and `pu_frac_mq0`
/// the MAPQ-0 reads over all counted reads. `pu_log_ref_mq0_alt_mqhigh` is
/// `ln((ref MAPQ 0 + 1) / (alt high + 1))`.
#[must_use]
pub fn pileup_features(counts: &AlleleCounts) -> [f64; PILEUP_FEATURES.len()] {
    let ratio = |numerator: u64, denominator: u64| {
        if denominator > 0 { numerator as f64 / denominator as f64 } else { f64::NAN }
    };
    let total = |allele: usize| counts[allele].iter().sum::<u64>();
    let (ref_all, alt_all, other_all) = (total(REF), total(ALT), total(OTHER));
    let mq0_all: u64 = counts.iter().map(|bins| bins[0]).sum();
    let [ref_bins, alt_bins, _] = counts;
    [
        ref_bins[0] as f64,
        ref_bins[1] as f64,
        ref_bins[2] as f64,
        alt_bins[0] as f64,
        alt_bins[1] as f64,
        alt_bins[2] as f64,
        other_all as f64,
        ratio(ref_all, ref_all + alt_all),
        ratio(ref_bins[2], ref_bins[2] + alt_bins[2]),
        ratio(ref_bins[0], ref_all),
        ratio(mq0_all, ref_all + alt_all + other_all),
        ((ref_bins[0] as f64 + 1.0) / (alt_bins[2] as f64 + 1.0)).ln(),
    ]
}

/// Joins pileup evidence to a feature table's calls by `(chrom, pos, ref, alt)` and adds
/// [`PILEUP_FEATURES`] (NaN for calls without a row), marking the table as having pileup evidence
/// measured with its settings.
///
/// # Errors
///
/// Returns an error if the table has SNP or indel calls and none of them has a row (the pileup
/// tables come from another VCF).
pub fn add_pileup_features(table: FeatureTable, pileup: &PileupTables) -> Result<FeatureTable> {
    let features: Vec<Option<[f64; PILEUP_FEATURES.len()]>> = table
        .join_calls(&pileup.rows, PileupOptions::row_key)?
        .into_par_iter()
        .map(|row| row.map(|row| pileup_features(&row.counts)))
        .collect();
    let n_with = features.iter().filter(|f| f.is_some()).count();
    let n_calls = table.num_rows();
    if n_calls > 0 && n_with == 0 {
        bail!(
            "none of the {n_calls} calls has a row in the pileup tables: were they made from \
             this VCF?"
        );
    }
    if n_calls > 0 {
        info!(
            "pileup evidence for {n_with} of {n_calls} calls ({:.1}%)",
            100.0 * n_with as f64 / n_calls as f64
        );
        warn_missing_contigs(&table, &features)?;
    }
    let mut table = table;
    for (k, name) in PILEUP_FEATURES.iter().enumerate() {
        let column = features.iter().map(|f| f.map_or(f64::NAN, |f| f[k])).collect();
        table = table.with_f64_column(name, column)?;
    }
    table.info_mut().pileup_settings = Some(pileup.options);
    Ok(table)
}

/// Warns about contigs none of whose calls has pileup evidence (e.g. a missing shard).
fn warn_missing_contigs<T>(table: &FeatureTable, joined: &[Option<T>]) -> Result<()> {
    let chroms = table.str_column("chrom")?;
    let mut per_contig: Vec<(&str, bool)> = Vec::new();
    for (chrom, row) in chroms.iter().zip(joined) {
        match per_contig.last_mut() {
            Some((contig, has)) if contig == chrom => *has |= row.is_some(),
            _ => per_contig.push((chrom, row.is_some())),
        }
    }
    let mut missing: Vec<&str> =
        per_contig.iter().filter(|(_, has)| !has).map(|(contig, _)| *contig).collect();
    missing.dedup();
    if !missing.is_empty() {
        warn!(
            "no pileup evidence for the calls on {} contig(s), e.g. a missing shard: {}",
            missing.len(),
            missing.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pileup::PileupRow;

    #[test]
    fn features_of_counts() {
        // ref: 21 at MAPQ 0, 1 low, 0 high; alt: 0, 2, 18; other: 1 high.
        let counts = [[21, 1, 0], [0, 2, 18], [0, 0, 1]];
        let f = pileup_features(&counts);
        let get = |name: &str| f[PILEUP_FEATURES.iter().position(|n| *n == name).unwrap()];
        assert_eq!(get("pu_ref_mq0"), 21.0);
        assert_eq!(get("pu_alt_mqhigh"), 18.0);
        assert_eq!(get("pu_other"), 1.0);
        assert_eq!(get("pu_ref_frac"), 22.0 / 42.0);
        assert_eq!(get("pu_ref_frac_mqhigh"), 0.0);
        assert_eq!(get("pu_ref_frac_mq0"), 21.0 / 22.0);
        assert_eq!(get("pu_frac_mq0"), 21.0 / 43.0);
        assert_eq!(get("pu_log_ref_mq0_alt_mqhigh"), (22.0f64 / 19.0).ln());
        let none = pileup_features(&[[0; 3]; 3]);
        assert!(none[7..=10].iter().all(|v| v.is_nan()));
        assert_eq!(none[11], 0.0);
    }

    #[test]
    fn joins_features_to_calls_and_marks_the_table() {
        let table = crate::testing::call_table(&[
            ("chr1", 10, "A", "G", "PASS", "snp", "1/1", true),
            ("chr1", 20, "AC", "A", "PASS", "indel", "0/1", true),
            ("chr2", 5, "C", "T", "PASS", "snp", "0/1", true),
        ]);
        let row = |pos: u64, reference: &str, alt: &str, counts: AlleleCounts| PileupRow {
            chrom: "chr1".into(),
            pos,
            reference: reference.into(),
            alt: alt.into(),
            n_reads: 30,
            counts,
        };
        let options: PileupOptions = "mapq30".parse().unwrap();
        let pileup = PileupTables {
            options,
            rows: vec![
                row(10, "A", "G", [[21, 1, 0], [0, 2, 18], [0, 0, 1]]),
                row(20, "AC", "A", [[0, 0, 10], [0, 0, 9], [0; 3]]),
                row(20, "AC", "AT", [[9; 3]; 3]), // another record at the position
            ],
        };
        let joined = add_pileup_features(table.clone(), &pileup).unwrap();
        assert_eq!(joined.info().pileup_settings, Some(options));
        assert_eq!(*joined.f64_column("pu_ref_mq0").unwrap().first().unwrap(), 21.0);
        let frac = joined.f64_column("pu_ref_frac").unwrap();
        assert_eq!((frac[0], frac[1]), (22.0 / 42.0, 10.0 / 19.0));
        assert!(frac[2].is_nan(), "a call without a row is missing");
        for name in PILEUP_FEATURES {
            assert!(joined.has_column(name), "{name}");
        }
        let other = PileupTables { options, rows: vec![row(99, "A", "G", [[0; 3]; 3])] };
        let message = add_pileup_features(table, &other).unwrap_err().to_string();
        assert!(message.contains("none of the 3 calls has a row"), "{message}");
    }
}
