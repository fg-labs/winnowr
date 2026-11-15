//! Helpers for building test inputs (compiled only for tests).

use std::io::Write;
use std::path::{Path, PathBuf};

use rust_htslib::bcf::index::{self, Type};
use rust_htslib::bgzf;

/// The INFO and FORMAT declarations of a HaplotypeCaller VCF, without the raw annotations.
pub(crate) const CALL_FIELDS: &str = "\
##FILTER=<ID=PASS,Description=\"All filters passed\">
##FILTER=<ID=LowQual,Description=\"Low quality\">
##FILTER=<ID=RefCall,Description=\"Reference call\">
##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">
##INFO=<ID=MQRankSum,Number=1,Type=Float,Description=\"MQ rank sum\">
##INFO=<ID=ReadPosRankSum,Number=1,Type=Float,Description=\"Read position rank sum\">
##INFO=<ID=BaseQRankSum,Number=1,Type=Float,Description=\"Base quality rank sum\">
##INFO=<ID=DRAGstrInfo,Number=2,Type=Integer,Description=\"Period and repeats\">
##INFO=<ID=ExcessHet,Number=1,Type=Float,Description=\"Excess het\">
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality\">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allele depths\">
##FORMAT=<ID=SB,Number=4,Type=Integer,Description=\"Strand bias\">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred likelihoods\">
";

/// The INFO declarations of the raw annotations, as in a reblocked gVCF.
pub(crate) const RAW_FIELDS: &str = "\
##INFO=<ID=RAW_MQandDP,Number=2,Type=Integer,Description=\"Raw MQ and DP\">
##INFO=<ID=QUALapprox,Number=1,Type=Integer,Description=\"Approximate QUAL\">
##INFO=<ID=VarDP,Number=1,Type=Integer,Description=\"Variant depth\">
##INFO=<ID=AS_RAW_MQ,Number=1,Type=String,Description=\"Allele-specific raw MQ\">
";

/// A complete VCF header with contigs `chr1` and `chr2`, the given field declarations, and the
/// given samples.
pub(crate) fn vcf_header(fields: &[&str], samples: &[&str]) -> String {
    let mut header = String::from(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=100000>\n##contig=<ID=chr2,length=100000>\n",
    );
    for declarations in fields {
        header.push_str(declarations);
    }
    header.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO");
    if !samples.is_empty() {
        header.push_str("\tFORMAT");
        for sample in samples {
            header.push('\t');
            header.push_str(sample);
        }
    }
    header.push('\n');
    header
}

/// Writes a bgzipped, tabix-indexed VCF of `header` and `records` (one tab-separated line each,
/// sorted) to `dir/name`.
pub(crate) fn write_indexed_vcf(dir: &Path, name: &str, header: &str, records: &[&str]) -> PathBuf {
    let path = dir.join(name);
    let mut writer = bgzf::Writer::from_path(&path).unwrap();
    writer.write_all(header.as_bytes()).unwrap();
    for record in records {
        writer.write_all(record.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
    }
    drop(writer);
    index::build(&path, None, 1, Type::Tbx).unwrap();
    path
}

/// Writes `dir/ref.fa`, with 4 bases per line, and its `.fai` index.
pub(crate) fn write_fasta(dir: &Path, sequences: &[(&str, &str)]) -> PathBuf {
    let path = dir.join("ref.fa");
    let (mut text, mut fai) = (String::new(), String::new());
    for (name, sequence) in sequences {
        text.push_str(&format!(">{name} description\n"));
        let offset = text.len();
        for chunk in sequence.as_bytes().chunks(4) {
            text.push_str(std::str::from_utf8(chunk).unwrap());
            text.push('\n');
        }
        fai.push_str(&format!("{name}\t{}\t{offset}\t4\t5\n", sequence.len()));
    }
    std::fs::write(&path, text).unwrap();
    std::fs::write(dir.join("ref.fa.fai"), fai).unwrap();
    path
}

/// A read to write to a test BAM.
pub(crate) struct TestRead<'a> {
    /// The read name.
    pub name: &'a str,
    /// The 0-based alignment start.
    pub pos: i64,
    /// The CIGAR, e.g. `10M2D10M`.
    pub cigar: &'a str,
    /// The SAM flag.
    pub flag: u16,
    /// The mapping quality.
    pub mapq: u8,
    /// The base quality of every base.
    pub qual: u8,
    /// The bases (`None`: all `A`).
    pub bases: Option<&'a str>,
}

/// Writes a coordinate-sorted, indexed BAM of reads on `chr1` (of `length` bases) to `dir/name`.
pub(crate) fn write_bam(dir: &Path, name: &str, length: u64, reads: &[TestRead<'_>]) -> PathBuf {
    use rust_htslib::bam::{self, header::HeaderRecord, record::CigarString};
    let path = dir.join(name);
    let mut header = bam::Header::new();
    let mut sq = HeaderRecord::new(b"SQ");
    sq.push_tag(b"SN", "chr1").push_tag(b"LN", length);
    header.push_record(&sq);
    let mut writer = bam::Writer::from_path(&path, &header, bam::Format::Bam).unwrap();
    let mut sorted: Vec<&TestRead<'_>> = reads.iter().collect();
    sorted.sort_by_key(|read| read.pos);
    for read in sorted {
        let cigar = CigarString::try_from(read.cigar).unwrap();
        let query_len = cigar
            .iter()
            .filter(|op| matches!(op.char(), 'M' | 'I' | 'S' | '=' | 'X'))
            .map(|op| op.len() as usize)
            .sum::<usize>();
        let bases = read.bases.map_or_else(|| vec![b'A'; query_len], |b| b.as_bytes().to_vec());
        assert_eq!(bases.len(), query_len, "the bases of {} do not match its CIGAR", read.name);
        let mut record = bam::Record::new();
        record.set(read.name.as_bytes(), Some(&cigar), &bases, &vec![read.qual; query_len]);
        record.set_tid(0);
        record.set_pos(read.pos);
        record.set_mapq(read.mapq);
        record.set_flags(read.flag);
        record.set_mtid(-1);
        record.set_mpos(-1);
        writer.write(&record).unwrap();
    }
    drop(writer);
    bam::index::build(&path, None, bam::index::Type::Bai, 1).unwrap();
    path
}

/// One call of a [`call_table`]: chrom, pos, ref, alt, filter, variant type, genotype and
/// whether it overlaps a PASS call.
pub(crate) type CallRow<'a> = (&'a str, i64, &'a str, &'a str, &'a str, &'a str, &'a str, bool);

/// A feature table with only the key columns of calls (enough for labelling).
pub(crate) fn call_table(rows: &[CallRow<'_>]) -> crate::table::FeatureTable {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    let strings = |f: for<'r> fn(&'r CallRow<'_>) -> &'r str| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(rows.iter().map(f)))
    };
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("chrom", strings(|r| r.0)),
        ("pos", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1)))),
        ("ref", strings(|r| r.2)),
        ("alt", strings(|r| r.3)),
        ("filter", strings(|r| r.4)),
        ("variant_type", strings(|r| r.5)),
        ("gt", strings(|r| r.6)),
        ("overlaps_pass", Arc::new(BooleanArray::from_iter(rows.iter().map(|r| Some(r.7))))),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| {
            let data_type: DataType = array.data_type().clone();
            Field::new(*name, data_type, true)
        })
        .collect();
    let arrays = columns.into_iter().map(|(_, array)| array).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let info = crate::table::FeatureTableInfo {
        sample: "S1".to_string(),
        contigs: vec!["chr1".to_string()],
        region_names: vec![],
        fn_counts: None,
        promote_labelled: false,
        strlen: false,
        strlen_settings: None,
        pileup_settings: None,
        reference_contigs: None,
    };
    crate::table::FeatureTable::new(batch, info)
}

/// The probability a [`separable_model`] gives the class its feature `x` names, at least.
pub(crate) const SEPARABLE_P: f64 = 0.9;

/// A three-class model of one feature, `x`, trained so that a call with `x` = k (0, 1 or 2) gets
/// class k with a probability of at least [`SEPARABLE_P`], with thresholds of 0.5 (t_drop and
/// t_flip of a genotype model in every mode; t_promote and t_flip of a promote model).
pub(crate) fn separable_model(
    task: crate::common::Task,
    variant_type: crate::common::VariantType,
) -> crate::model::WinnowrModel {
    use std::num::NonZeroUsize;

    use winnowr_xgboost::{Booster, DMatrix};

    use crate::common::{NEVER, Task};
    use crate::model::{MODEL_FORMAT_VERSION, ModelInfo, TrainingParams, WinnowrModel};
    use crate::rescore::Mode;
    use crate::thresholds::{PromoteThresholds, Thresholds};

    let rows = 300;
    let values: Vec<f32> = (0..rows).map(|i| (i % 3) as f32).collect();
    let mut train = DMatrix::from_dense(&values, rows, 1, f32::NAN).unwrap();
    train.set_labels(&values).unwrap();
    train.set_feature_names(&["x"]).unwrap();
    let threads = NonZeroUsize::new(1).unwrap();
    let mut booster = Booster::new(&[&train], threads).unwrap();
    booster
        .set_params(&[("objective", "multi:softprob"), ("num_class", "3"), ("eta", "0.5")])
        .unwrap();
    for iteration in 0..20 {
        booster.update(iteration, &train).unwrap();
    }
    let (thresholds, promote_thresholds) = match task {
        Task::Genotype => {
            let t = Thresholds { t_drop: 0.5, t_flip: 0.5, gain: 0.0, t_regenotype: NEVER };
            (Mode::ALL.iter().map(|&mode| (mode, t)).collect(), None)
        }
        Task::Promote => (
            std::collections::BTreeMap::new(),
            Some(PromoteThresholds { t_promote: 0.5, t_flip: 0.5, gain: 0.0 }),
        ),
    };
    let info = ModelInfo {
        format_version: MODEL_FORMAT_VERSION,
        variant_type,
        task,
        features: vec!["x".to_string()],
        region_names: vec![],
        thresholds,
        promote_thresholds,
        training_samples: vec!["TRAIN".to_string()],
        contigs: vec!["chr1".to_string()],
        reference_contigs: None,
        params: TrainingParams::default(),
        stutter: None,
        regenotype: false,
        strlen_settings: None,
        pileup_settings: None,
        winnowr_version: env!("CARGO_PKG_VERSION").to_string(),
        xgboost_version: "test".to_string(),
    };
    info.validate().unwrap();
    WinnowrModel { booster, info }
}
