//! Builds a sample's feature table from its VCF, gVCF and region BEDs.

use std::path::Path;

use anyhow::{Result, bail};

use crate::alignment_provenance::AlignmentProvenance;
use crate::common::PhaseTimer;
use crate::features::derive_features;
use crate::labels::{HappyPassOnly, attach_labels, check_label_join, read_happy_labels};
use crate::pileup::PileupTables;
use crate::pileup_features::add_pileup_features;
use crate::promote_labels::{attach_promote_labels, read_truth_variants};
use crate::raw_records::{AnnotationInputs, extract_raw_records};
use crate::regions::RegionSet;
use crate::repeat_features::add_strlen_features;
use crate::strlen::StrlenTables;
use crate::table::{FeatureTable, FeatureTableInfo};
use crate::vcf::VcfSummary;

/// The truth inputs that label a feature table.
#[derive(Debug, Clone, Copy, Default)]
pub struct TruthInputs<'a> {
    /// The hap.py output VCF (vcfeval engine, preferably `--pass-only`; see `happy_pass_only`),
    /// which labels PASS calls.
    pub happy: Option<&'a Path>,
    /// Whether hap.py was run with `--pass-only` (see [`attach_labels`]).
    pub happy_pass_only: HappyPassOnly,
    /// The truth VCF; with `truth_bed` and `happy`, labels non-PASS calls for promote models.
    pub truth_vcf: Option<&'a Path>,
    /// The truth confident regions.
    pub truth_bed: Option<&'a Path>,
    /// The reference FASTA used to left-align indels when matching non-PASS calls to the truth.
    pub reference: Option<&'a Path>,
}

impl TruthInputs<'_> {
    /// Checks that the promote-label inputs are given together.
    ///
    /// # Errors
    ///
    /// Returns an error if the truth VCF, BED or reference is given without all of the truth
    /// VCF, the truth BED and the hap.py VCF.
    pub fn check(&self) -> Result<()> {
        let any = self.truth_vcf.is_some() || self.truth_bed.is_some() || self.reference.is_some();
        let all = self.truth_vcf.is_some() && self.truth_bed.is_some() && self.happy.is_some();
        if any && !all {
            bail!(
                "promote labels need --truth-vcf, --truth-bed and --happy together (--reference \
                 is optional and only used with them)"
            );
        }
        Ok(())
    }
}

/// The inputs of a feature table besides the calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct TableInputs<'a> {
    /// Where the raw annotations come from (the feature source the table records): by default
    /// the VCF's own, with no gVCF.
    pub annotations: AnnotationInputs<'a>,
    /// Outputs to protect against a gVCF and its indexes only when that gVCF is opened.
    pub outputs: &'a [std::path::PathBuf],
    /// The region sets to flag.
    pub regions: &'a [RegionSet],
    /// Repeat-tract evidence (see [`crate::strlen`]) to join as repeat-length features.
    pub strlen: Option<&'a StrlenTables>,
    /// Allele read counts by MAPQ (see [`crate::pileup`]) to join as pileup features.
    pub pileup: Option<&'a PileupTables>,
    /// The truth inputs that label the table.
    pub truth: TruthInputs<'a>,
    /// The depth `dp_norm` divides by, in place of the calls' median depth (see
    /// [`derive_features`]); checked with [`crate::features::check_dp_median`].
    pub dp_median: Option<f64>,
    /// How the reads were aligned, recorded in the table, when known.
    pub alignment: Option<&'a AlignmentProvenance>,
}

/// Checks the inputs of a feature table that can be checked before any is read:
/// [`build_feature_table`] runs it first, and a command runs it before a pass over the reads
/// that the table needs, so that a mistake fails before the pass rather than after it.
///
/// # Errors
///
/// Returns an error if the promote-label inputs are incomplete, the given depth median is not a
/// depth (see [`crate::features::check_dp_median`]), a contig is repeated, or a region set has no
/// intervals on `contigs`.
pub fn check_table_inputs<S: AsRef<str>>(
    contigs: &[S],
    regions: &[RegionSet],
    truth: &TruthInputs<'_>,
    dp_median: Option<f64>,
) -> Result<()> {
    truth.check()?;
    if let Some(dp_median) = dp_median {
        crate::features::check_dp_median(dp_median)?;
    }
    crate::raw_records::check_unique_contigs(contigs)?;
    for region_set in regions {
        region_set.check_contigs(contigs)?;
    }
    Ok(())
}

/// Extracts the features of every SNP and indel call of a sample and, given a hap.py VCF,
/// attaches labels.
///
/// # Arguments
///
/// * `sample` - the sample name to record.
/// * `vcf` - the indexed single-sample VCF.
/// * `contigs` - the contigs to extract, in order.
/// * `summary` - the VCF's header summary, for the contig lengths recorded in the metadata.
/// * `inputs` - the raw annotations' inputs, region sets, strlen and pileup evidence, and truth
///   inputs.
///
/// # Errors
///
/// Returns an error if the promote-label inputs are incomplete, the given depth median is not a
/// depth (see [`crate::features::check_dp_median`]) or a contig is repeated (checked before any
/// input is read), a region set has no intervals on `contigs` or its covered-base total cannot fit
/// in `u64`, the calls cannot be read (see [`extract_raw_records`]), or most hap.py labels match no
/// call.
pub fn build_feature_table<S: AsRef<str> + Sync>(
    sample: &str,
    vcf: &Path,
    contigs: &[S],
    summary: &VcfSummary,
    inputs: &TableInputs<'_>,
) -> Result<FeatureTable> {
    check_table_inputs(contigs, inputs.regions, &inputs.truth, inputs.dp_median)?;
    let mut timer = PhaseTimer::start();
    let before_gvcf = |gvcf: &Path| {
        let indexes = crate::vcf::index_candidates(gvcf);
        for output in inputs.outputs {
            crate::scoring::check_output_not_input(output, [("gVCF", gvcf)])?;
            crate::scoring::check_output_not_indexes(output, "gVCF index", &indexes)?;
        }
        Ok(())
    };
    let records = extract_raw_records(vcf, inputs.annotations, contigs, &before_gvcf)?;
    timer.lap("reading the calls");
    let labels = inputs.truth.happy.map(|happy| read_happy_labels(happy, contigs)).transpose()?;
    if labels.is_some() {
        timer.lap("reading the hap.py labels");
    }
    let derived = derive_features(&records, inputs.regions, inputs.dp_median);
    timer.lap_total("deriving features");
    let info = FeatureTableInfo {
        sample: sample.to_string(),
        contigs: contigs.iter().map(|contig| contig.as_ref().to_string()).collect(),
        region_names: inputs.regions.iter().map(|r| r.name().to_string()).collect(),
        fn_counts: None,
        promote_labelled: inputs.truth.truth_vcf.is_some(),
        strlen: false,
        strlen_settings: None,
        pileup_settings: None,
        reference_contigs: Some(summary.lengths_of(contigs)),
        dp_norm_divisor: Some(derived.dp_norm_divisor),
        restricted_to: None,
        feature_source: inputs.annotations.feature_source(),
        alignment: inputs.alignment.cloned(),
        region_digests: inputs
            .regions
            .iter()
            .map(|r| Ok((r.name().to_string(), r.digest()?)))
            .collect::<Result<_>>()?,
    };
    let mut table = FeatureTable::from_records(&records, derived, info)?;
    timer.lap_total("building the table");
    drop(records);
    timer.lap("dropping the records");
    if let Some(strlen) = inputs.strlen {
        table = add_strlen_features(table, strlen)?;
        timer.lap_total("joining the repeat-length features");
    }
    if let Some(pileup) = inputs.pileup {
        table = add_pileup_features(table, pileup)?;
        timer.lap("joining the pileup features");
    }
    if let (Some(labels), Some(happy)) = (labels, inputs.truth.happy) {
        let (labelled, join) = attach_labels(table, &labels, inputs.truth.happy_pass_only)?;
        check_label_join(&join, &happy.display().to_string())?;
        table = labelled;
        timer.lap("attaching the hap.py labels");
        table.info_mut().fn_counts = Some(
            labels
                .fn_counts
                .iter()
                .map(|(variant_type, &n)| (variant_type.to_string(), n))
                .collect(),
        );
        if let (Some(truth_vcf), Some(truth_bed)) = (inputs.truth.truth_vcf, inputs.truth.truth_bed)
        {
            let truth =
                read_truth_variants(truth_vcf, &labels.fn_positions, inputs.truth.reference)?;
            timer.lap("reading the truth variants at FNs");
            let confident = RegionSet::from_bed("confident", truth_bed)?;
            confident.check_contigs(contigs)?;
            timer.lap("reading the confident regions");
            table = attach_promote_labels(
                table,
                &truth,
                &labels.fn_positions,
                &confident,
                inputs.truth.reference,
            )?;
            timer.lap("attaching the promote labels");
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::features::DpNormDivisor;
    use crate::regions::{NamedBed, read_region_sets};
    use crate::testing::{CALL_FIELDS, RAW_FIELDS, vcf_header, write_indexed_vcf};

    fn write_vcf(dir: &TempDir, samples: &[&str], records: &[&str]) -> PathBuf {
        let lines: Vec<String> = records.iter().map(|r| r.replace(' ', "\t")).collect();
        let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
        let header = vcf_header(&[CALL_FIELDS, RAW_FIELDS], samples);
        write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &lines)
    }

    fn bed(dir: &TempDir, name: &str, text: &str) -> NamedBed {
        let path = dir.path().join(format!("{name}.bed"));
        std::fs::write(&path, text).unwrap();
        NamedBed { name: name.to_string(), path }
    }

    const RECORDS: [&str; 3] = [
        "chr1 100 . A C 50 PASS RAW_MQandDP=3600,4;QUALapprox=30;VarDP=2;AS_RAW_MQ=30|10 GT:AD 0/1:6,2",
        "chr1 120 . AT A 20 LowQual RAW_MQandDP=100,2 GT:AD 1/1:0,3",
        "chr2 50 . G T 40 PASS RAW_MQandDP=400,1 GT 0/1",
    ];

    #[test]
    fn builds_a_feature_table_and_round_trips_it_through_parquet() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, &["NA12878"], &RECORDS);
        let regions = read_region_sets(&[bed(&dir, "segdups", "chr1\t110\t130\n")]).unwrap();
        let summary = VcfSummary::read(&vcf).unwrap();
        let sample = summary.sample_name(&vcf);
        let table = build_feature_table(
            &sample,
            &vcf,
            &["chr1", "chr2"],
            &summary,
            &TableInputs { regions: &regions, ..TableInputs::default() },
        )
        .unwrap();
        assert_eq!(table.num_rows(), 3);
        assert_eq!(table.str_column("chrom").unwrap(), ["chr1", "chr1", "chr2"]);
        assert_eq!(table.i64_column("pos").unwrap(), [100, 120, 50]);
        assert_eq!(table.str_column("alt").unwrap(), ["C", "A", "T"]);
        assert_eq!(table.str_column("gt").unwrap(), ["0/1", "1/1", "0/1"]);
        assert_eq!(*table.f64_column("region_segdups").unwrap(), [0.0, 1.0, 0.0]);
        assert_eq!(table.f64_column("ab").unwrap()[0], 0.25);
        assert_eq!(table.f64_column("mq").unwrap()[0], 30.0);
        assert_eq!(*table.f64_column("dp_norm").unwrap(), [2.0, 1.0, 0.5]);
        assert_eq!(table.bool_column("overlaps_pass").unwrap(), [true, false, true]);
        assert_eq!(table.f64_column("nb_dist").unwrap()[0], 20.0);

        let info = table.info();
        assert_eq!(info.sample, "NA12878");
        assert_eq!(info.region_names, ["segdups"]);
        assert!(!info.is_labelled());
        let lengths = info.reference_contigs.as_ref().unwrap();
        // In the given order, not sorted.
        assert_eq!(lengths.keys().collect::<Vec<_>>(), ["chr1", "chr2"]);
        assert_eq!(lengths.get("chr1"), Some(&Some(100_000)));
        // The calls' median depth (of 4, 2 and 1), not given.
        assert_eq!(info.dp_norm_divisor, Some(DpNormDivisor { depth: Some(2.0), given: false }));

        let path = dir.path().join("table.parquet");
        table.write_parquet(&path).unwrap();
        let read = FeatureTable::read_parquet(&path).unwrap();
        assert_eq!(read.info(), table.info());
        assert_eq!(read.column_names(), table.column_names());
        assert_eq!(read.batch().num_rows(), 3);
        for name in table.column_names() {
            let (a, b) = (read.batch().column_by_name(name), table.batch().column_by_name(name));
            assert_eq!(a.map(|c| c.to_data()), b.map(|c| c.to_data()), "{name}");
        }
    }

    #[test]
    fn a_given_depth_median_normalizes_the_depth() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, &["S1"], &RECORDS);
        let summary = VcfSummary::read(&vcf).unwrap();
        let inputs = TableInputs { dp_median: Some(8.0), ..TableInputs::default() };
        let table = build_feature_table("S1", &vcf, &["chr1", "chr2"], &summary, &inputs).unwrap();
        // Depths 4, 2 and 1 over 8, not over their median of 2.
        assert_eq!(*table.f64_column("dp_norm").unwrap(), [0.5, 0.25, 0.125]);
        let recorded = Some(DpNormDivisor { depth: Some(8.0), given: true });
        assert_eq!(table.info().dp_norm_divisor, recorded);
        let path = dir.path().join("table.parquet");
        table.write_parquet(&path).unwrap();
        assert_eq!(FeatureTable::read_info(&path).unwrap().dp_norm_divisor, recorded);
    }

    #[test]
    fn table_columns_have_the_expected_names_and_order() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, &["S1"], &RECORDS);
        let regions = read_region_sets(&[bed(&dir, "segdups", "chr1\t110\t130\n")]).unwrap();
        let table = build_feature_table(
            "S1",
            &vcf,
            &["chr1"],
            &VcfSummary::read(&vcf).unwrap(),
            &TableInputs { regions: &regions, ..TableInputs::default() },
        )
        .unwrap();
        let mut expected = vec![
            "chrom",
            "pos",
            "ref",
            "alt",
            "filter",
            "variant_type",
            "gt",
            "qual",
            "dp",
            "mq_rank_sum",
            "read_pos_rank_sum",
            "base_q_rank_sum",
            "str_period",
            "str_repeats",
            "gq",
            "ad_ref",
            "ad_alt",
            "sb_ref_fwd",
            "sb_ref_rev",
            "sb_alt_fwd",
            "sb_alt_rev",
            "excess_het",
            "pl_ref",
            "pl_het",
            "pl_hom",
            "ad_alt_total",
            "raw_mq",
            "raw_dp",
            "qual_approx",
            "var_dp",
            "as_mq_ref",
            "as_mq_alt",
            "n_alt",
            "ab",
            "sor",
            "het",
            "flippable",
            "mq",
            "qd",
            "as_alt_frac",
            "dp_norm",
            "indel_len",
            "indel_abs_len",
            "region_segdups",
            "n_alt_rec",
            "overlaps_pass",
        ];
        expected.extend(crate::neighbourhood::NEIGHBOURHOOD_FEATURES);
        assert_eq!(table.column_names(), expected);
    }

    #[test]
    fn reads_selected_columns_and_rows() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, &["S1"], &RECORDS);
        let summary = VcfSummary::read(&vcf).unwrap();
        let table =
            build_feature_table("S1", &vcf, &["chr1", "chr2"], &summary, &TableInputs::default())
                .unwrap();
        let path = dir.path().join("t.parquet");
        table.write_parquet(&path).unwrap();
        assert_eq!(FeatureTable::read_info(&path).unwrap(), *table.info());

        let selection = crate::table::TableSelection {
            columns: Some(vec!["qual".into(), "pos".into(), "no_such_column".into()]),
            variant_type: Some(crate::common::VariantType::Snp),
        };
        let selected = FeatureTable::read_parquet_selected(&path, &selection).unwrap();
        // File order; variant_type is read to select the rows.
        assert_eq!(selected.column_names(), ["pos", "variant_type", "qual"]);
        assert_eq!(selected.i64_column("pos").unwrap(), [100, 50]);
        assert_eq!(selected.info(), table.info());
        let indels = FeatureTable::read_parquet_selected(
            &path,
            &crate::table::TableSelection {
                columns: None,
                variant_type: Some(crate::common::VariantType::Indel),
            },
        )
        .unwrap();
        assert_eq!(indels.column_names(), table.column_names());
        assert_eq!(indels.i64_column("pos").unwrap(), [120]);
    }

    #[test]
    fn an_empty_table_is_written_and_read() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, &["S1"], &RECORDS[..1]);
        let table = build_feature_table(
            "S1",
            &vcf,
            &["chr2"],
            &VcfSummary::read(&vcf).unwrap(),
            &TableInputs::default(),
        )
        .unwrap();
        assert_eq!(table.num_rows(), 0);
        let path = dir.path().join("empty.parquet");
        table.write_parquet(&path).unwrap();
        assert_eq!(FeatureTable::read_parquet(&path).unwrap().num_rows(), 0);
    }

    #[test]
    fn rejects_repeated_region_names_and_beds_without_the_contigs() {
        let dir = TempDir::new().unwrap();
        let specs = [
            bed(&dir, "a", "chr1\t0\t5\n"),
            bed(&dir, "b", "chr1\t0\t5\n"),
            bed(&dir, "a", "chr1\t0\t5\n"),
        ];
        let message = read_region_sets(&specs).unwrap_err().to_string();
        assert_eq!(message, "region names must be unique; repeated: a");

        let vcf = write_vcf(&dir, &["S1"], &RECORDS);
        let regions = read_region_sets(&[bed(&dir, "numbered", "1\t0\t5\n")]).unwrap();
        let message = build_feature_table(
            "S1",
            &vcf,
            &["chr1"],
            &VcfSummary::read(&vcf).unwrap(),
            &TableInputs { regions: &regions, ..TableInputs::default() },
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("numbered.bed has no intervals"), "{message}");
    }

    #[test]
    fn labels_calls_with_happy_and_the_truth() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, &["S1"], &RECORDS);
        let happy = dir.path().join("happy.vcf");
        let happy_text = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=chr1,length=100000>",
            "##contig=<ID=chr2,length=100000>",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
            "##FORMAT=<ID=BD,Number=1,Type=String,Description=\"Decision\">",
            "##FORMAT=<ID=BK,Number=1,Type=String,Description=\"Match kind\">",
            "##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"Variant type\">",
            "#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT TRUTH QUERY",
            "chr1 100 . A C . . . GT:BD:BK:BVT 0/1:TP:gm:SNP 0/1:TP:gm:SNP",
            "chr1 120 . AT A . . . GT:BD:BK:BVT 0/1:FN:.:INDEL .:.:.:NOCALL",
            "chr2 50 . G T . . . GT:BD:BK:BVT .:.:.:NOCALL 0/1:FP:.:SNP",
        ]
        .map(|line| line.replace(' ', "\t") + "\n")
        .concat();
        std::fs::write(&happy, happy_text).unwrap();
        let truth_header = vcf_header(&[CALL_FIELDS], &["TRUTH"]);
        let truth_vcf = write_indexed_vcf(
            dir.path(),
            "truth.vcf.gz",
            &truth_header,
            &["chr1\t120\t.\tAT\tA\t.\tPASS\t.\tGT\t0/1"],
        );
        let truth_bed = dir.path().join("confident.bed");
        std::fs::write(&truth_bed, "chr1\t0\t1000\nchr2\t0\t1000\n").unwrap();

        let summary = VcfSummary::read(&vcf).unwrap();
        let truth = TruthInputs {
            happy: Some(&happy),
            truth_vcf: Some(&truth_vcf),
            truth_bed: Some(&truth_bed),
            reference: None,
            ..TruthInputs::default()
        };
        let inputs = TableInputs { truth, ..TableInputs::default() };
        let table = build_feature_table("S1", &vcf, &["chr1", "chr2"], &summary, &inputs).unwrap();
        let names = table.column_names();
        let tail = &names[names.len() - 5..];
        assert_eq!(tail, ["label", "flip_ok", "truth_delta_lo", "truth_delta_hi", "promote_label"]);
        let shown = |name: &str| -> Vec<String> {
            table.f64_column(name).unwrap().iter().map(|v| format!("{v}")).collect()
        };
        assert_eq!(shown("label"), ["1", "NaN", "0"]);
        assert_eq!(shown("truth_delta_lo"), ["0", "-1", "NaN"]);
        // The 1/1 deletion is the 0/1 truth FN: promote and flip.
        assert_eq!(shown("promote_label"), ["NaN", "2", "NaN"]);
        let info = table.info();
        assert!(info.promote_labelled);
        let fn_counts: Vec<(&str, u64)> =
            info.fn_counts.as_ref().unwrap().iter().map(|(k, &v)| (k.as_str(), v)).collect();
        assert_eq!(fn_counts, [("snp", 0), ("indel", 1)]);

        let incomplete = TableInputs {
            truth: TruthInputs { truth_vcf: Some(&truth_vcf), ..TruthInputs::default() },
            ..TableInputs::default()
        };
        let message = build_feature_table("S1", &vcf, &["chr1"], &summary, &incomplete)
            .unwrap_err()
            .to_string();
        assert!(message.starts_with("promote labels need --truth-vcf, --truth-bed and --happy"));
    }

    /// `happy_pass_only` reaches the labels: a label at a LowQual SNP call's position that a
    /// padded PASS SNP was trimmed to is the PASS call's with `--pass-only`, and the LowQual
    /// call's without.
    #[rstest::rstest]
    #[case::yes(HappyPassOnly::Yes, ["1", "NaN"])]
    #[case::no(HappyPassOnly::No, ["NaN", "NaN"])]
    fn happy_pass_only_decides_whose_label_it_is(
        #[case] happy_pass_only: HappyPassOnly,
        #[case] expected: [&str; 2],
    ) {
        let dir = TempDir::new().unwrap();
        let records = [
            "chr1 300 . TTAGA TTCGA 50 PASS RAW_MQandDP=400,1 GT 0/1",
            "chr1 302 . A C 20 LowQual RAW_MQandDP=400,1 GT 0/1",
        ];
        let vcf = write_vcf(&dir, &["S1"], &records);
        let happy = dir.path().join("happy.vcf");
        let happy_text = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=chr1,length=100000>",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
            "##FORMAT=<ID=BD,Number=1,Type=String,Description=\"Decision\">",
            "##FORMAT=<ID=BK,Number=1,Type=String,Description=\"Match kind\">",
            "##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"Variant type\">",
            "#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT TRUTH QUERY",
            "chr1 302 . A C . . . GT:BD:BK:BVT 0/1:TP:gm:SNP 0/1:TP:gm:SNP",
        ]
        .map(|line| line.replace(' ', "\t") + "\n")
        .concat();
        std::fs::write(&happy, happy_text).unwrap();
        let summary = VcfSummary::read(&vcf).unwrap();
        let truth = TruthInputs { happy: Some(&happy), happy_pass_only, ..TruthInputs::default() };
        let inputs = TableInputs { truth, ..TableInputs::default() };
        let table = build_feature_table("S1", &vcf, &["chr1"], &summary, &inputs).unwrap();
        let shown: Vec<String> =
            table.f64_column("label").unwrap().iter().map(|v| format!("{v}")).collect();
        assert_eq!(shown, expected);
    }

    #[test]
    fn reading_a_foreign_parquet_is_an_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("foreign.parquet");
        let schema =
            std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "x",
                arrow_schema::DataType::Int64,
                false,
            )]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![std::sync::Arc::new(arrow_array::Int64Array::from(vec![1]))],
        )
        .unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            schema,
            None,
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let message = FeatureTable::read_parquet(&path).unwrap_err().to_string();
        assert!(
            message.ends_with("is not a winnowr feature table (no winnowr metadata)"),
            "{message}"
        );
        let not_parquet = dir.path().join("table.txt");
        std::fs::write(&not_parquet, "not parquet").unwrap();
        let message = FeatureTable::read_parquet(&not_parquet).unwrap_err().to_string();
        assert!(message.ends_with("table.txt is not a Parquet file"), "{message}");
    }

    /// Tables from other Parquet writers may be Snappy-compressed (a common default), with `string`
    /// or `large_string` text columns.
    #[test]
    fn snappy_tables_with_large_strings_are_read() {
        use std::sync::Arc;

        use arrow_array::{ArrayRef, Float64Array, Int64Array, LargeStringArray, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::basic::Compression;
        use parquet::file::properties::WriterProperties;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("other-writer.parquet");
        let fields = vec![
            Field::new("chrom", DataType::Utf8, true),
            Field::new("pos", DataType::Int64, true),
            Field::new("variant_type", DataType::LargeUtf8, true),
            Field::new("sl_unit", DataType::LargeUtf8, true),
            Field::new("qual", DataType::Float64, true),
        ];
        let metadata = std::collections::HashMap::from([(
            "winnowr".to_string(),
            r#"{"sample": "S", "contigs": ["chr1"], "region_names": [], "fn_counts": null, "promote_labelled": false}"#
                .to_string(),
        )]);
        let schema = Arc::new(Schema::new(fields).with_metadata(metadata));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["chr1", "chr1", "chr1"])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(LargeStringArray::from(vec!["snp", "indel", "snp"])),
            Arc::new(LargeStringArray::from(vec![None, Some("CA"), None])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ];
        let batch = arrow_array::RecordBatch::try_new(schema.clone(), columns).unwrap();
        let properties = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
        let file = std::fs::File::create(&path).unwrap();
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(file, schema, Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        // The file really is Snappy-compressed.
        let reader =
            parquet::file::reader::SerializedFileReader::new(std::fs::File::open(&path).unwrap())
                .unwrap();
        let codec = parquet::file::reader::FileReader::metadata(&reader)
            .row_group(0)
            .column(0)
            .compression();
        assert_eq!(codec, Compression::SNAPPY);

        let table = FeatureTable::read_parquet(&path).unwrap();
        assert_eq!(table.str_column("variant_type").unwrap(), ["snp", "indel", "snp"]);
        assert_eq!(table.str_column("sl_unit").unwrap(), ["", "CA", ""]);
        let snps = FeatureTable::read_parquet_selected(
            &path,
            &crate::table::TableSelection {
                columns: Some(vec!["pos".into(), "qual".into()]),
                variant_type: Some(crate::common::VariantType::Snp),
            },
        )
        .unwrap();
        assert_eq!(snps.i64_column("pos").unwrap(), [10, 30]);
        assert_eq!(*snps.f64_column("qual").unwrap(), [1.0, 3.0]);
    }

    #[test]
    fn nulls_in_other_writers_tables_read_as_nan() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("other-writer.parquet");
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("mq", arrow_schema::DataType::Float64, true),
            arrow_schema::Field::new("pos", arrow_schema::DataType::Int64, true),
        ])
        .with_metadata(std::collections::HashMap::from([(
            "winnowr".to_string(),
            r#"{"sample": "S", "contigs": ["chr1"], "region_names": [], "fn_counts": {"snp": 1}, "promote_labelled": false}"#.to_string(),
        )]));
        let schema = std::sync::Arc::new(schema);
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(arrow_array::Float64Array::from(vec![Some(1.5), None])),
                std::sync::Arc::new(arrow_array::Int64Array::from(vec![Some(1), None])),
            ],
        )
        .unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            schema,
            None,
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let table = FeatureTable::read_parquet(&path).unwrap();
        let mq = table.f64_column("mq").unwrap();
        assert_eq!(mq[0], 1.5);
        assert!(mq[1].is_nan());
        let message = table.i64_column("pos").unwrap_err().to_string();
        assert_eq!(message, "column pos has 1 missing values");
        // Metadata without strlen or reference contigs reads with defaults.
        assert!(!table.info().strlen && table.info().reference_contigs.is_none());
        assert_eq!(table.info().strlen_options(), None);
        // Strlen evidence without recorded settings is read as measured with the legacy settings.
        let mut info = table.info().clone();
        info.strlen = true;
        assert_eq!(info.strlen_options(), Some(crate::strlen::StrlenOptions::LEGACY));
        info.strlen_settings = Some(crate::strlen::StrlenOptions::default());
        assert_eq!(info.strlen_options(), Some(crate::strlen::StrlenOptions::default()));
        assert_eq!(table.info().fn_count(crate::common::VariantType::Snp).unwrap(), 1);
        let message =
            table.info().fn_count(crate::common::VariantType::Indel).unwrap_err().to_string();
        assert_eq!(message, "the feature table for S has no indel FN count");
    }
}
