//! Opt-in, data-free validation helpers. See `scripts/benchmark-table-reads.sh` and
//! `scripts/validate-native-info.sh`.
//!
//! Keep this module compatible with the pre-provenance reader so that the same harness can be
//! copied into historical builds without changing their production read or write paths.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use arrow_array::{Array, Float64Array, Int64Array};
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::common::VariantType;
use crate::features::derive_features;
use crate::table::{FeatureTable, FeatureTableInfo, TableSelection};

// Libtest does not include main.rs: use the binary's allocator for production-reader timings.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Generates bounded batches of deterministic, varied calls with synthetic truth labels.
/// These labels exercise table consumers; they are not analytical accuracy evidence.
fn generate(path: &Path, rows: usize) -> Result<()> {
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut writer = None;
    for start in (0..rows).step_by(65_536) {
        let end = (start + 65_536).min(rows);
        let records: Vec<_> = (start..end)
            .map(|index| {
                let indel = index % 5 == 0;
                let mut record = crate::testing::raw_record_with_missing_values(
                    "chr1",
                    index as u64 * 7 + 1,
                    "A",
                    &[if indel { "AT" } else { "C" }],
                );
                record.variant_type = if indel { VariantType::Indel } else { VariantType::Snp };
                record.dp = (index % 71 + 1) as f64;
                record.qual = (index % 997) as f64 / 3.0;
                record.gq = (index % 100) as f64;
                record.ad_ref = (index % 29) as f64;
                record.ad_alt = (index % 31) as f64;
                record.ad_alt_total = record.ad_alt;
                record.raw.raw_dp = record.dp;
                record.raw.raw_mq = record.dp * ((index % 61) as f64).powi(2);
                record.raw.qual_approx = record.qual;
                record.raw.var_dp = record.dp;
                record.raw.n_alt = 1.0;
                record.sb_ref_fwd = (index % 17) as f64;
                record.sb_ref_rev = (index % 19) as f64;
                record.sb_alt_fwd = (index % 23) as f64;
                record.sb_alt_rev = (index % 29) as f64;
                record
            })
            .collect();
        let derived = derive_features(&records, &[], Some(36.0));
        let info = FeatureTableInfo {
            sample: "synthetic".to_owned(),
            contigs: vec!["chr1".to_owned()],
            fn_counts: Some([("snp".to_owned(), 11), ("indel".to_owned(), 7)].into()),
            ..FeatureTableInfo::default()
        };
        let table = FeatureTable::from_records(&records, derived, info)?
            .with_f64_column("label", (start..end).map(|index| (index % 3) as f64).collect())?
            .with_bool_column("flip_ok", (start..end).map(|index| index % 3 == 2).collect())?;
        if writer.is_none() {
            let metadata = std::collections::HashMap::from([(
                "winnowr".to_owned(),
                serde_json::to_string(table.info())?,
            )]);
            let schema = Arc::new(table.batch().schema().as_ref().clone().with_metadata(metadata));
            writer = Some(ArrowWriter::try_new(
                std::fs::File::create(path)?,
                schema,
                Some(properties.clone()),
            )?);
        }
        writer.as_mut().unwrap().write(table.batch())?;
    }
    writer.context("row count must be positive")?.close()?;
    Ok(())
}

/// Decodes using stock Parquet/Arrow only, independently of the reader being benchmarked.
fn read_independent(path: &Path) -> Result<FeatureTable> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path)?)?;
    let info = serde_json::from_str(
        builder.schema().metadata().get("winnowr").context("no winnowr metadata")?,
    )?;
    let schema = Arc::new(Schema::new(builder.schema().fields().clone()));
    let batches = builder.build()?.collect::<std::result::Result<Vec<_>, _>>()?;
    let batch = arrow_select::concat::concat_batches(&schema, &batches)?;
    Ok(FeatureTable::new(batch, info))
}

/// The benchmark's projected columns, in their on-disk order.
const SNP_COLUMNS: [&str; 8] = ["chrom", "pos", "variant_type", "gt", "dp", "mq", "qd", "label"];

/// Validates generated row identities and the projection after the read timer has stopped.
fn check_generated_read(table: &FeatureTable, rows: usize, snps_only: bool) -> Result<()> {
    if snps_only {
        assert_eq!(table.column_names(), SNP_COLUMNS);
        assert_eq!(table.num_rows(), rows - rows.div_ceil(5));
    } else {
        assert_eq!(table.column_names().len(), 58);
        assert_eq!(table.num_rows(), rows);
    }
    let positions = table.i64_column("pos")?;
    let types = table.str_column("variant_type")?;
    let chroms = table.str_column("chrom")?;
    let genotypes = table.str_column("gt")?;
    let expected = (0..rows).filter(|index| !snps_only || index % 5 != 0);
    for (row, index) in expected.enumerate() {
        assert_eq!(positions[row], (index * 7 + 1) as i64);
        assert_eq!(types[row], if index % 5 == 0 { "indel" } else { "snp" });
        assert_eq!(chroms[row], "chr1");
        assert_eq!(genotypes[row], "0/1");
    }
    Ok(())
}

/// Checks keys, metadata, every decoded column, and missingness (not compressed-file hashes).
fn compare(left: &FeatureTable, right: &FeatureTable) {
    assert_eq!(left.info(), right.info());
    assert_eq!(left.batch().schema(), right.batch().schema());
    assert_eq!(left.num_rows(), right.num_rows());
    for (a, b) in left.batch().columns().iter().zip(right.batch().columns()) {
        if let (Some(a), Some(b)) =
            (a.as_any().downcast_ref::<Float64Array>(), b.as_any().downcast_ref::<Float64Array>())
        {
            assert!(
                a.iter()
                    .map(|value| value.map(f64::to_bits))
                    .eq(b.iter().map(|value| value.map(f64::to_bits)))
            );
        } else {
            assert_eq!(a, b);
        }
    }
}

/// A production-reader benchmark, or decoded-output comparison, selected by environment.
/// Generation and optional decoded-output writing are outside the measured read region.
#[test]
#[ignore = "manual validation: set WINNOWR_TABLE_ACTION and WINNOWR_TABLE_PATH"]
fn validate_feature_tables() -> Result<()> {
    let action = std::env::var("WINNOWR_TABLE_ACTION")?;
    let path = std::env::var_os("WINNOWR_TABLE_PATH").context("WINNOWR_TABLE_PATH is required")?;
    let path = Path::new(&path);
    match action.as_str() {
        "generate" => {
            let rows = std::env::var("WINNOWR_TABLE_ROWS")?.parse()?;
            generate(path, rows)?;
        }
        "read" => {
            let selection = match std::env::var("WINNOWR_TABLE_SELECTION").as_deref() {
                Ok("snp") => TableSelection {
                    columns: Some(
                        ["chrom", "pos", "variant_type", "gt", "label", "dp", "mq", "qd"]
                            .map(str::to_owned)
                            .into(),
                    ),
                    variant_type: Some(VariantType::Snp),
                },
                Ok("full") => TableSelection::default(),
                _ => bail!("WINNOWR_TABLE_SELECTION must be full or snp"),
            };
            let rows: usize = std::env::var("WINNOWR_TABLE_ROWS")?.parse()?;
            // Match the CLI's explicit pool setup; parity runs cannot warm another process's pool.
            rayon::ThreadPoolBuilder::new().build_global()?;
            rayon::broadcast(|_| ());
            let start = Instant::now();
            let table = FeatureTable::read_parquet_selected(path, &selection)?;
            let seconds = start.elapsed().as_secs_f64();
            assert!(table.info().is_labelled());
            check_generated_read(&table, rows, selection.variant_type.is_some())?;
            std::hint::black_box(&table);
            println!(
                "READ seconds={seconds:.6} rows={} columns={}",
                table.num_rows(),
                table.column_names().len()
            );
            if let Some(output) = std::env::var_os("WINNOWR_TABLE_OUTPUT") {
                table.write_parquet(output)?;
            }
        }
        "compare" => {
            let other = std::env::var_os("WINNOWR_TABLE_OTHER")
                .context("WINNOWR_TABLE_OTHER is required")?;
            compare(&read_independent(path)?, &read_independent(Path::new(&other))?);
            println!("All decoded values, missingness, schema and metadata match");
        }
        _ => bail!("unknown WINNOWR_TABLE_ACTION: {action}"),
    }
    Ok(())
}

/// Rejects empty comparisons, mismatched keys, and MQ/SOR without finite paired evidence.
fn check_native_evidence(baseline: &FeatureTable, native: &FeatureTable) -> Result<()> {
    if baseline.num_rows() == 0 || baseline.num_rows() != native.num_rows() {
        bail!("native feature comparison requires nonempty tables of the same length");
    }
    for name in ["chrom", "pos", "ref", "alt", "gt"] {
        let column = |table: &FeatureTable| {
            table
                .batch()
                .column_by_name(name)
                .cloned()
                .with_context(|| format!("missing call key {name}"))
        };
        if column(baseline)? != column(native)? {
            bail!("call keys differ: {name}");
        }
    }
    for name in ["mq", "sor"] {
        if !baseline
            .f64_column(name)?
            .iter()
            .zip(native.f64_column(name)?.iter())
            .any(|(a, b)| a.is_finite() && b.is_finite())
        {
            bail!("no finite paired {name} observations for native INFO validation");
        }
    }
    Ok(())
}

/// Native INFO fallback evidence: report paired feature differences only after exact key checks.
#[test]
#[ignore = "manual validation: set WINNOWR_NATIVE_BASELINE and WINNOWR_NATIVE_VCF"]
fn compare_native_features() -> Result<()> {
    let baseline = std::env::var_os("WINNOWR_NATIVE_BASELINE")
        .context("WINNOWR_NATIVE_BASELINE is required")?;
    let native =
        std::env::var_os("WINNOWR_NATIVE_VCF").context("WINNOWR_NATIVE_VCF is required")?;
    let baseline = read_independent(Path::new(&baseline))?;
    let native = read_independent(Path::new(&native))?;
    check_native_evidence(&baseline, &native)?;
    for name in ["mq", "dp_norm", "qd", "sor", "n_alt"] {
        let left = baseline.f64_column(name)?;
        let right = native.f64_column(name)?;
        let mut differences = Vec::new();
        let (mut left_missing, mut right_missing, mut equal, mut low_depth, mut above_rounding) =
            (0, 0, 0, 0, 0);
        let strand_counts = ["sb_ref_fwd", "sb_ref_rev", "sb_alt_fwd", "sb_alt_rev"]
            .map(|column| baseline.f64_column(column))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        for (index, (&a, &b)) in left.iter().zip(right.iter()).enumerate() {
            left_missing += usize::from(!a.is_finite());
            right_missing += usize::from(!b.is_finite());
            if a.is_finite() && b.is_finite() {
                differences.push((a - b).abs());
                equal += usize::from(a == b);
                let depth: f64 = strand_counts.iter().map(|counts| counts[index]).sum();
                let tolerance = match name {
                    "mq" => Some(0.00501),
                    "sor" => Some(0.00051),
                    _ => None,
                };
                if let Some(tolerance) = tolerance {
                    let outside_rounding = (a - b).abs() > tolerance;
                    low_depth += usize::from(depth <= 2.0 && outside_rounding);
                    above_rounding += usize::from(outside_rounding);
                }
            }
        }
        differences.sort_by(f64::total_cmp);
        let quantile = |q: f64| {
            differences
                .get(((differences.len().saturating_sub(1)) as f64 * q).round() as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        let rounding = if matches!(name, "mq" | "sor") {
            format!(" outside_rounding={above_rounding} low_sb_depth_outside_rounding={low_depth}")
        } else {
            String::new()
        };
        println!(
            "{name}: pairs={} equal={equal} baseline_missing={left_missing} native_missing={right_missing} median={} p99={} max={}{rounding}",
            differences.len(),
            quantile(0.5),
            quantile(0.99),
            quantile(1.0)
        );
    }
    Ok(())
}

#[test]
fn native_comparisons_require_nonempty_paired_mq_and_sor() -> Result<()> {
    let empty = crate::testing::call_table(&[]);
    assert!(check_native_evidence(&empty, &empty).unwrap_err().to_string().contains("nonempty"));
    let keys = crate::testing::call_table(&[
        ("chr1", 1, "A", "C", "PASS", "snp", "0/1", false),
        ("chr1", 2, "A", "G", "PASS", "snp", "0/1", false),
    ]);
    let baseline = keys
        .clone()
        .with_f64_column("mq", vec![51.0, f64::NAN])?
        .with_f64_column("sor", vec![0.693, f64::NAN])?;
    check_native_evidence(&baseline, &baseline)?;
    for name in ["mq", "sor"] {
        for missing in [
            vec![f64::NAN; 2],
            vec![f64::NAN, 1.0],
            vec![f64::INFINITY; 2],
            vec![f64::NEG_INFINITY; 2],
        ] {
            let unusable = baseline.clone().with_f64_column(name, missing)?;
            let expected = format!("no finite paired {name}");
            for (left, right) in [(&baseline, &unusable), (&unusable, &baseline)] {
                assert!(
                    check_native_evidence(left, right).unwrap_err().to_string().contains(&expected)
                );
            }
        }
    }
    let changed = baseline.clone().with_f64_column("mq", vec![10.0, 20.0])?;
    check_native_evidence(&baseline, &changed)?;
    Ok(())
}

#[test]
fn native_comparisons_reject_mismatched_or_missing_keys_and_lengths() -> Result<()> {
    let rows = [
        ("chr1", 1, "A", "C", "PASS", "snp", "0/1", false),
        ("chr1", 2, "A", "G", "PASS", "snp", "0/1", false),
    ];
    let baseline = crate::testing::call_table(&rows)
        .with_f64_column("mq", vec![51.0; 2])?
        .with_f64_column("sor", vec![0.693; 2])?;
    let rejected = |other: &FeatureTable, expected: &str| {
        for (left, right) in [(&baseline, other), (other, &baseline)] {
            assert!(check_native_evidence(left, right).unwrap_err().to_string().contains(expected));
        }
    };
    for (name, values) in [
        ("chrom", ["chr2", "chr1"]),
        ("ref", ["T", "A"]),
        ("alt", ["T", "G"]),
        ("gt", ["1/1", "0/1"]),
    ] {
        let changed = baseline.clone().with_str_column(name, &values)?;
        rejected(&changed, &format!("call keys differ: {name}"));
    }
    let changed = baseline.clone().with_column("pos", Arc::new(Int64Array::from(vec![3, 2])))?;
    rejected(&changed, "call keys differ: pos");
    rejected(&crate::testing::call_table(&[rows[1], rows[0]]), "call keys differ: pos");
    let short = FeatureTable::new(baseline.batch().slice(0, 1), baseline.info().clone());
    rejected(&short, "same length");
    for key in ["chrom", "pos", "ref", "alt", "gt"] {
        let indices: Vec<_> = baseline
            .column_names()
            .iter()
            .enumerate()
            .filter(|(_, name)| **name != key)
            .map(|(index, _)| index)
            .collect();
        let missing =
            FeatureTable::new(baseline.batch().project(&indices)?, baseline.info().clone());
        rejected(&missing, &format!("missing call key {key}"));
    }
    Ok(())
}

#[test]
fn generated_labelled_tables_round_trip_all_decoded_values() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("labelled.parquet");
    generate(&path, 131_073)?;
    let independent = read_independent(&path)?;
    let table = FeatureTable::read_parquet(&path)?;
    compare(&independent, &table);
    check_generated_read(&table, 131_073, false)?;
    assert_eq!(
        table.info(),
        &FeatureTableInfo {
            sample: "synthetic".to_owned(),
            contigs: vec!["chr1".to_owned()],
            fn_counts: Some([("snp".to_owned(), 11), ("indel".to_owned(), 7)].into()),
            ..FeatureTableInfo::default()
        }
    );
    let labels = table.f64_column("label")?;
    let flips = table.bool_column("flip_ok")?;
    for (index, value) in labels.iter().enumerate() {
        assert_eq!(*value, (index % 3) as f64);
        assert_eq!(flips[index], index % 3 == 2);
    }
    assert_eq!(&table.f64_column("dp")?[..3], &[1.0, 2.0, 3.0]);
    assert_eq!(table.f64_column("dp")?[131_072], 7.0);
    assert_eq!(table.f64_column("qual")?[131_072], 155.0);
    assert_eq!(table.f64_column("mq")?[131_072], 44.0);
    assert_eq!(table.f64_column("qd")?[131_072], 155.0 / 7.0);
    assert_eq!(&table.str_column("ref")?[..3], &["A", "A", "A"]);
    assert_eq!(&table.str_column("alt")?[..3], &["AT", "C", "C"]);
    for name in ["base_q_rank_sum", "as_alt_frac"] {
        assert!(table.f64_column(name)?.iter().all(|value| value.is_nan()));
        assert_eq!(table.batch().column_by_name(name).unwrap().null_count(), 0);
    }
    let output = dir.path().join("roundtrip.parquet");
    table.write_parquet(&output)?;
    compare(&independent, &read_independent(&output)?);
    Ok(())
}

#[test]
fn generated_snp_projection_has_exact_columns_and_call_identities() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("selection.parquet");
    generate(&path, 31)?;
    let selection = TableSelection {
        columns: Some(SNP_COLUMNS.map(str::to_owned).into()),
        variant_type: Some(VariantType::Snp),
    };
    let table = FeatureTable::read_parquet_selected(&path, &selection)?;
    check_generated_read(&table, 31, true)?;
    let positions = table.i64_column("pos")?;
    assert_eq!(&positions[..5], &[8, 15, 22, 29, 43]);
    assert_eq!(positions.last(), Some(&204));
    Ok(())
}
