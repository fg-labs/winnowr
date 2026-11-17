//! Restrict a labelled feature table to some of its contigs, or away from some regions, with its
//! truth FN counts recounted to match.
//!
//! A held-out evaluation within one sample (e.g. training on its even chromosomes and evaluating
//! on its odd ones) needs a training table whose labels come only from the training part: its
//! calls on the other contigs dropped, and its FN counts, which enter the F1 that thresholds are
//! chosen by, counted on the kept contigs only. Features are not recomputed: per-sample features
//! (e.g. the depth normalization) still come from all of the sample's calls, as no truth enters
//! them.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, bail};

use crate::common::VariantType;
use crate::labels::{HappyFn, read_happy_fns};
use crate::regions::{RegionSet, ref_span};
use crate::table::FeatureTable;

/// Truth FN counts by variant type.
type FnCounts = BTreeMap<VariantType, u64>;

/// The truth FN records of a hap.py output VCF, counted by variant type: on `all` contigs, and on
/// the `kept` contigs with a REF span that does not overlap `exclude`.
fn count_fns(
    happy: &Path,
    all: &[String],
    kept: &[String],
    exclude: Option<&RegionSet>,
) -> Result<(FnCounts, FnCounts)> {
    let (mut total, mut retained) = (FnCounts::new(), FnCounts::new());
    for record in read_happy_fns(happy, |chrom| all.iter().any(|c| c == chrom))? {
        let HappyFn { chrom, variant_type, span: (start, end) } = record?;
        *total.entry(variant_type).or_default() += 1;
        let excluded = exclude.is_some_and(|regions| regions.overlaps(&chrom, start, end));
        if kept.contains(&chrom) && !excluded {
            *retained.entry(variant_type).or_default() += 1;
        }
    }
    Ok((total, retained))
}

/// The calls of a labelled feature table on `contigs` whose REF span does not overlap `exclude`,
/// with the truth FN counts of the same part of the genome, from the hap.py VCF it was labelled
/// from.
///
/// The table's metadata is otherwise unchanged; in particular it still lists every contig it was
/// extracted from, as its per-sample features come from all of them.
///
/// # Arguments
///
/// * `table` - the labelled feature table.
/// * `happy` - the hap.py output VCF the table was labelled from.
/// * `contigs` - the contigs to keep: each one of the table's contigs.
/// * `exclude` - regions to drop calls and FNs in (by REF span overlap), if any.
///
/// # Errors
///
/// Returns an error if the table is not labelled, a contig is not one of the table's, the hap.py
/// VCF cannot be read, or its FN counts on the table's contigs differ from the table's (it is not
/// the VCF the table was labelled from).
pub fn restrict_table(
    table: &FeatureTable,
    happy: &Path,
    contigs: &[String],
    exclude: Option<&RegionSet>,
) -> Result<FeatureTable> {
    let info = table.info();
    if !info.is_labelled() {
        bail!("the feature table for {} has no hap.py labels (extract with --happy)", info.sample);
    }
    let unknown: Vec<&str> =
        contigs.iter().filter(|c| !info.contigs.contains(c)).map(String::as_str).collect();
    if !unknown.is_empty() {
        bail!(
            "the feature table for {} was not extracted from contigs {}; it has [{}]",
            info.sample,
            unknown.join(", "),
            info.contigs.join(", ")
        );
    }
    let (total, retained) = count_fns(happy, &info.contigs, contigs, exclude)?;
    for vt in [VariantType::Snp, VariantType::Indel] {
        let recorded = info.fn_count(vt)?;
        let found = total.get(&vt).copied().unwrap_or(0);
        if found != recorded {
            bail!(
                "{} has {found} {vt} FNs on the table's contigs, but the table for {} records \
                 {recorded}: is it the hap.py VCF the table was labelled from?",
                happy.display(),
                info.sample
            );
        }
    }
    let (chroms, positions, refs) =
        (table.str_column("chrom")?, table.i64_column("pos")?, table.str_column("ref")?);
    let mask: Vec<bool> = (0..table.num_rows())
        .map(|row| {
            let (start, end) = ref_span(positions[row], refs[row].len());
            contigs.iter().any(|c| c == chroms[row])
                && !exclude.is_some_and(|regions| regions.overlaps(chroms[row], start, end))
        })
        .collect();
    let mut restricted = table.filter(&mask)?;
    restricted.info_mut().fn_counts = Some(
        [VariantType::Snp, VariantType::Indel]
            .into_iter()
            .map(|vt| (vt.to_string(), retained.get(&vt).copied().unwrap_or(0)))
            .collect(),
    );
    Ok(restricted)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::features::DpNormDivisor;
    use crate::testing::{call_table, vcf_header, write_indexed_vcf};

    /// A hap.py VCF of FNs: SNPs at chr1:10, chr1:60, chr2:10 and an indel at chr2:100.
    fn happy(dir: &Path) -> std::path::PathBuf {
        let fields = "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n\
            ##FORMAT=<ID=BD,Number=1,Type=String,Description=\"BD\">\n\
            ##FORMAT=<ID=BK,Number=1,Type=String,Description=\"BK\">\n\
            ##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"BVT\">\n";
        let header = vcf_header(&[fields], &["TRUTH", "QUERY"]);
        let records = [
            "chr1\t10\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
            "chr1\t60\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
            "chr1\t80\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:TP:gm:SNP\t0/1:TP:gm:SNP",
            "chr2\t10\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t./.:.:.:NOCALL",
            "chr2\t100\t.\tAT\tA\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:INDEL\t./.:.:.:NOCALL",
        ];
        write_indexed_vcf(dir, "happy.vcf.gz", &header, &records)
    }

    /// A labelled table of calls on chr1 and chr2, with FN counts of the VCF of [`happy`].
    fn table() -> FeatureTable {
        let mut table = call_table(&[
            ("chr1", 20, "A", "C", "PASS", "snp", "0/1", true),
            ("chr1", 55, "AT", "A", "PASS", "indel", "0/1", true),
            ("chr1", 80, "A", "C", "PASS", "snp", "0/1", true),
            ("chr2", 30, "A", "C", "PASS", "snp", "0/1", true),
        ]);
        let info = table.info_mut();
        info.contigs = vec!["chr1".to_string(), "chr2".to_string()];
        info.fn_counts =
            Some([("snp".to_string(), 3), ("indel".to_string(), 1)].into_iter().collect());
        info.dp_norm_divisor = Some(DpNormDivisor { depth: Some(30.0), given: true });
        table
    }

    fn fn_counts(table: &FeatureTable) -> (u64, u64) {
        let info = table.info();
        (info.fn_count(VariantType::Snp).unwrap(), info.fn_count(VariantType::Indel).unwrap())
    }

    #[test]
    fn keeps_the_calls_and_fns_of_the_contigs() {
        let dir = TempDir::new().unwrap();
        let happy = happy(dir.path());
        let got = restrict_table(&table(), &happy, &["chr2".to_string()], None).unwrap();
        assert_eq!(got.str_column("chrom").unwrap(), ["chr2"]);
        assert_eq!(fn_counts(&got), (1, 1));
        assert_eq!(got.info().contigs, ["chr1", "chr2"]);
        assert_eq!(got.info().dp_norm_divisor, table().info().dp_norm_divisor);
        let all = ["chr1".to_string(), "chr2".to_string()];
        let unchanged = restrict_table(&table(), &happy, &all, None).unwrap();
        assert_eq!(unchanged.num_rows(), 4);
        assert_eq!(fn_counts(&unchanged), (3, 1));
    }

    #[test]
    fn drops_the_calls_and_fns_in_excluded_regions() {
        let dir = TempDir::new().unwrap();
        let happy = happy(dir.path());
        // [50, 60) overlaps the indel at chr1:55 (REF span [54, 56)) and the FN at chr1:60
        // (REF span [59, 60)); [99, 100) overlaps the indel FN at chr2:100.
        let exclude = RegionSet::from_spans("hp", [("chr1", 50, 60), ("chr2", 99, 100)]);
        let all = ["chr1".to_string(), "chr2".to_string()];
        let got = restrict_table(&table(), &happy, &all, Some(&exclude)).unwrap();
        assert_eq!(got.i64_column("pos").unwrap(), [20, 80, 30]);
        assert_eq!(fn_counts(&got), (2, 0));
    }

    #[test]
    fn rejects_unknown_contigs_and_another_vcf() {
        let dir = TempDir::new().unwrap();
        let happy = happy(dir.path());
        let message =
            restrict_table(&table(), &happy, &["chr3".to_string()], None).unwrap_err().to_string();
        assert!(message.contains("not extracted from contigs chr3"), "{message}");
        let mut other = table();
        other.info_mut().fn_counts =
            Some([("snp".to_string(), 2), ("indel".to_string(), 1)].into_iter().collect());
        let message =
            restrict_table(&other, &happy, &["chr1".to_string()], None).unwrap_err().to_string();
        assert!(message.contains("has 3 snp FNs"), "{message}");
        let mut unlabelled = table();
        unlabelled.info_mut().fn_counts = None;
        let message = restrict_table(&unlabelled, &happy, &["chr1".to_string()], None)
            .unwrap_err()
            .to_string();
        assert!(message.contains("has no hap.py labels (extract with --happy)"), "{message}");
    }
}
