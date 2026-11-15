//! Raw per-record values of a HaplotypeCaller VCF, with the raw annotations from the VCF itself or
//! from the gVCF it was genotyped from.
//!
//! Contigs are read in parallel (on the rayon pool), each with its own readers, and the records
//! are returned in contig order, then VCF order within each contig.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use log::{info, warn};
use rayon::prelude::*;
use rust_htslib::bcf::{IndexedReader, Read, Record};

use crate::alleles::{Genotype, first_alt_index, variant_type_of};
use crate::common::VariantType;
use crate::gvcf::{
    RawAnnotation, check_gvcf_header, has_raw_annotations, matching_site, missing_raw_annotations,
    raw_annotation, read_gvcf_sites,
};
use crate::shards::{ContigShard, ReaderPool, contig_shards};
use crate::vcf::{
    FieldReader, check_contigs, fetch_span, filter_string, has_format, has_info, open_indexed,
};

/// The raw values of one VCF record. Missing numeric values are NaN.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRecord {
    /// The contig.
    pub chrom: Arc<str>,
    /// The 1-based position.
    pub pos: u64,
    /// The reference allele.
    pub reference: Vec<u8>,
    /// The alternate alleles.
    pub alts: Vec<Vec<u8>>,
    /// The FILTER names joined with `;`, or `PASS` when there are none.
    pub filter: String,
    /// The variant type (records that are neither SNPs nor indels are not extracted).
    pub variant_type: VariantType,
    /// The sample's normalized genotype.
    pub genotype: Genotype,
    /// QUAL.
    pub qual: f64,
    /// INFO/DP.
    pub dp: f64,
    /// INFO/MQRankSum.
    pub mq_rank_sum: f64,
    /// INFO/ReadPosRankSum.
    pub read_pos_rank_sum: f64,
    /// INFO/BaseQRankSum.
    pub base_q_rank_sum: f64,
    /// The repeat period from INFO/DRAGstrInfo.
    pub str_period: f64,
    /// The number of repeats from INFO/DRAGstrInfo.
    pub str_repeats: f64,
    /// FORMAT/GQ.
    pub gq: f64,
    /// The reference allele depth, FORMAT/AD\[0\].
    pub ad_ref: f64,
    /// The depth of the first non-`*` alternate allele.
    pub ad_alt: f64,
    /// Reference-allele forward-strand reads, FORMAT/SB\[0\].
    pub sb_ref_fwd: f64,
    /// Reference-allele reverse-strand reads, FORMAT/SB\[1\].
    pub sb_ref_rev: f64,
    /// Alternate-allele forward-strand reads, FORMAT/SB\[2\].
    pub sb_alt_fwd: f64,
    /// Alternate-allele reverse-strand reads, FORMAT/SB\[3\].
    pub sb_alt_rev: f64,
    /// INFO/ExcessHet.
    pub excess_het: f64,
    /// The `0/0` likelihood, FORMAT/PL\[0\].
    pub pl_ref: f64,
    /// The `0/1` likelihood, FORMAT/PL\[1\].
    pub pl_het: f64,
    /// The `1/1` likelihood, FORMAT/PL\[2\].
    pub pl_hom: f64,
    /// The summed depth of every alternate allele (NaN if none is known).
    pub ad_alt_total: f64,
    /// The raw annotations ([`RawAnnotation::MISSING`] when there are none).
    pub raw: RawAnnotation,
}

/// Reads the raw values of every SNP and indel record of a single-sample VCF on `contigs`.
///
/// The raw annotations come from the VCF's own records when its header declares all of them
/// (any `gvcf` is then not read, unless no record has them), and otherwise from the gVCF, joined
/// by position. Records without them (no `RAW_MQandDP`, or no gVCF site) get
/// [`RawAnnotation::MISSING`]. Records that are neither SNPs nor indels (see
/// [`variant_type_of`]) are skipped.
///
/// # Arguments
///
/// * `vcf` - the indexed single-sample VCF.
/// * `gvcf` - the indexed gVCF the VCF was genotyped from, if any.
/// * `contigs` - the contigs to read, in order.
///
/// # Errors
///
/// Returns an error if a contig is repeated, the VCF does not have exactly one sample or
/// FORMAT/GT, a contig is not in the VCF or gVCF header, neither the VCF nor a gVCF supplies the
/// raw annotations (including a VCF whose header declares them but none of whose records has
/// them, without a gVCF), or a file cannot be read.
pub fn extract_raw_records<P, S>(
    vcf: P,
    gvcf: Option<&Path>,
    contigs: &[S],
) -> Result<Vec<RawRecord>>
where
    P: AsRef<Path>,
    S: AsRef<str> + Sync,
{
    let vcf = vcf.as_ref();
    check_unique_contigs(contigs)?;
    let mut annotation_source = {
        let reader = open_indexed(vcf)?;
        let header = reader.header();
        check_vcf_header(header, vcf, contigs)?;
        warn_undeclared_fields(header, vcf);
        match (has_raw_annotations(header), gvcf) {
            (true, gvcf) => {
                if let Some(gvcf) = gvcf {
                    warn!(
                        "VCF {} declares the raw annotations, so they are read from its records; \
                         the gVCF {} is read only if none of them has them",
                        vcf.display(),
                        gvcf.display()
                    );
                }
                AnnotationSource::Vcf
            }
            (false, Some(gvcf)) => {
                check_gvcf_header(open_indexed(gvcf)?.header(), gvcf, contigs)?;
                info!("Reading the raw annotations from the gVCF {}", gvcf.display());
                AnnotationSource::Gvcf(gvcf)
            }
            (false, None) => bail!(
                "VCF {} lacks the INFO annotations {} and no gVCF was given: pass the gVCF the \
                 VCF was genotyped from, or a VCF that keeps the raw annotations of \
                 HaplotypeCaller's gVCF mode",
                vcf.display(),
                missing_raw_annotations(header).join(", ")
            ),
        }
    };

    // Contigs are read in shards of POS spans, in parallel; a contig's last shard is open-ended,
    // so a record past the declared length is still read.
    let summary = crate::vcf::VcfSummary::read(vcf)?;
    let shards = contig_shards(&summary, contigs, RECORD_SHARD_SPAN);
    let mut per_shard = read_shards(vcf, annotation_source, &shards)?;
    let n_records: usize = per_shard.iter().map(|shard| shard.records.len()).sum();
    let n_missing: usize = per_shard.iter().map(|shard| shard.missing_annotations).sum();
    // A header can declare the raw annotations that no record has (e.g. after a tool dropped
    // them): every raw feature would be missing.
    if matches!(annotation_source, AnnotationSource::Vcf) && n_records > 0 && n_missing == n_records
    {
        let Some(gvcf) = gvcf else {
            bail!(
                "VCF {} declares the raw annotations, but none of its {n_records} records has \
                 RAW_MQandDP and no gVCF was given: pass the gVCF the VCF was genotyped from",
                vcf.display()
            );
        };
        warn!(
            "VCF {} declares the raw annotations, but none of its {n_records} records has \
             RAW_MQandDP: reading them from the gVCF {} instead",
            vcf.display(),
            gvcf.display()
        );
        check_gvcf_header(open_indexed(gvcf)?.header(), gvcf, contigs)?;
        annotation_source = AnnotationSource::Gvcf(gvcf);
        per_shard = read_shards(vcf, annotation_source, &shards)?;
    }

    let mut per_contig = vec![0usize; contigs.len()];
    for (shard, records) in shards.iter().zip(&per_shard) {
        per_contig[shard.index] += records.records.len();
    }
    for (contig, n) in contigs.iter().zip(per_contig) {
        info!("Read {n} records on {}", contig.as_ref());
    }

    let (mut skipped, mut missing, mut other_alleles) = (0usize, 0usize, 0usize);
    let mut records = Vec::with_capacity(per_shard.iter().map(|c| c.records.len()).sum());
    for shard in per_shard {
        skipped += shard.skipped;
        missing += shard.missing_annotations;
        other_alleles += shard.other_alleles;
        records.extend(shard.records);
    }
    info!("Read {} records on {} contig(s)", records.len(), contigs.len());
    if skipped > 0 {
        info!(
            "Skipped {skipped} records that are neither SNPs nor indels (MNPs, symbolic, \
             breakend or only `*` alleles)"
        );
    }
    if missing > 0 {
        let source = match annotation_source {
            AnnotationSource::Vcf => "lack RAW_MQandDP",
            AnnotationSource::Gvcf(_) => "had no gVCF annotations",
        };
        warn!("{missing} records {source} (set to missing)");
    }
    if other_alleles > 0 {
        warn!(
            "{other_alleles} records took the raw annotations of the first gVCF site at their \
             position, which has other alleles (no site there has theirs)"
        );
    }
    Ok(records)
}

/// Checks that no contig is listed twice, as its records would be read twice.
///
/// # Errors
///
/// Returns an error naming the repeated contigs.
pub(crate) fn check_unique_contigs<S: AsRef<str>>(contigs: &[S]) -> Result<()> {
    let mut seen = HashSet::new();
    let mut repeated: Vec<&str> =
        contigs.iter().map(AsRef::as_ref).filter(|contig| !seen.insert(*contig)).collect();
    if !repeated.is_empty() {
        repeated.dedup();
        bail!("contigs must be unique; repeated: {}", repeated.join(", "));
    }
    Ok(())
}

/// Reads `shards` in parallel, on the rayon pool.
fn read_shards(
    vcf: &Path,
    annotation_source: AnnotationSource<'_>,
    shards: &[ContigShard<'_>],
) -> Result<Vec<ContigRecords>> {
    let pool: ReaderPool<Readers> = ReaderPool::new();
    shards
        .par_iter()
        .map(|shard| {
            let mut readers = pool.take().unwrap_or_default();
            let read = read_shard(
                &mut readers,
                vcf,
                annotation_source,
                shard.contig,
                shard.start,
                shard.end,
            );
            pool.put(readers);
            read
        })
        .collect()
}

/// The INFO and FORMAT fields whose features are missing (NaN) for every call when a VCF header
/// does not declare them: the core fields of every HaplotypeCaller VCF, and those that some
/// modes or versions do not write.
const FEATURE_FIELDS: [(&str, bool, bool); 10] = [
    // (field, is INFO, is core)
    ("DP", true, true),
    ("AD", false, true),
    ("GQ", false, true),
    ("PL", false, true),
    ("SB", false, false),
    ("MQRankSum", true, false),
    ("ReadPosRankSum", true, false),
    ("BaseQRankSum", true, false),
    ("ExcessHet", true, false),
    ("DRAGstrInfo", true, false),
];

/// Logs the [`FEATURE_FIELDS`] a VCF header does not declare: a warning for core fields, a note
/// for the others.
fn warn_undeclared_fields(header: &rust_htslib::bcf::header::HeaderView, path: &Path) {
    let undeclared = |core: bool| -> Vec<String> {
        FEATURE_FIELDS
            .iter()
            .filter(|&&(_, _, is_core)| is_core == core)
            .filter(|&&(field, is_info, _)| {
                let tag = field.as_bytes();
                !if is_info { has_info(header, tag) } else { has_format(header, tag) }
            })
            .map(|&(field, is_info, _)| {
                format!("{}/{field}", if is_info { "INFO" } else { "FORMAT" })
            })
            .collect()
    };
    let (core, other) = (undeclared(true), undeclared(false));
    if !core.is_empty() {
        warn!(
            "VCF {} does not declare {}: the features read from them are missing for every call",
            path.display(),
            core.join(", ")
        );
    }
    if !other.is_empty() {
        info!(
            "VCF {} does not declare {}: the features read from them are missing for every call",
            path.display(),
            other.join(", ")
        );
    }
}

/// Where the raw annotations are read from.
#[derive(Debug, Clone, Copy)]
enum AnnotationSource<'a> {
    /// The VCF's own records.
    Vcf,
    /// The gVCF at this path.
    Gvcf(&'a Path),
}

/// The records read from one shard of a contig, and counts for logging.
struct ContigRecords {
    records: Vec<RawRecord>,
    skipped: usize,
    missing_annotations: usize,
    /// Records that took a gVCF site with other alleles (see [`matching_site`]).
    other_alleles: usize,
}

/// Validates the sample count, contigs and genotype field of a VCF header.
fn check_vcf_header<S: AsRef<str>>(
    header: &rust_htslib::bcf::header::HeaderView,
    path: &Path,
    contigs: &[S],
) -> Result<()> {
    let samples = header.sample_count();
    if samples != 1 {
        bail!("VCF {} must have exactly one sample, has {samples}", path.display());
    }
    check_contigs(header, contigs, path)?;
    if !has_format(header, b"GT") {
        bail!("VCF {} has no FORMAT/GT", path.display());
    }
    Ok(())
}

/// Lazily opened readers, reused across the shards a worker reads.
#[derive(Default)]
struct Readers {
    vcf: Option<IndexedReader>,
    gvcf: Option<IndexedReader>,
}

/// Returns the reader in `slot`, opening `path` into it first if needed.
fn reader<'a>(slot: &'a mut Option<IndexedReader>, path: &Path) -> Result<&'a mut IndexedReader> {
    if slot.is_none() {
        *slot = Some(open_indexed(path)?);
    }
    Ok(slot.as_mut().expect("the reader was just opened"))
}

/// The span of the shards that a contig's records are read in, in parallel.
const RECORD_SHARD_SPAN: u64 = 5_000_000;

/// Reads the records of one contig whose 0-based POS is in `[start, end)` (to the contig's end
/// if `end` is `None`), with their raw annotations.
fn read_shard(
    readers: &mut Readers,
    vcf_path: &Path,
    source: AnnotationSource<'_>,
    contig: &str,
    start: u64,
    end: Option<u64>,
) -> Result<ContigRecords> {
    let vcf = reader(&mut readers.vcf, vcf_path)?;
    let has_records = fetch_span(vcf, contig, start, end, vcf_path)?;
    let chrom: Arc<str> = Arc::from(contig);
    let mut record = vcf.empty_record();
    let mut fields = FieldReader::new();
    let mut records = Vec::new();
    let (mut skipped, mut missing_annotations, mut other_alleles) = (0, 0, 0);
    while has_records && let Some(read) = vcf.read(&mut record) {
        read.with_context(|| {
            format!("failed to read a record on {contig} from {}", vcf_path.display())
        })?;
        // A fetch also returns records that start before the shard but overlap it.
        let pos = u64::try_from(record.pos()).unwrap_or(0);
        if pos < start || end.is_some_and(|end| pos >= end) {
            continue;
        }
        let alleles = record.alleles();
        let Some(mut raw) = raw_record(&mut fields, &record, &alleles, &chrom)? else {
            skipped += 1;
            continue;
        };
        if let AnnotationSource::Vcf = source {
            match raw_annotation(&mut fields, &record, &alleles)? {
                Some(annotation) => raw.raw = annotation,
                None => missing_annotations += 1,
            }
        }
        records.push(raw);
    }
    if let AnnotationSource::Gvcf(gvcf_path) = source {
        let gvcf = reader(&mut readers.gvcf, gvcf_path)?;
        (missing_annotations, other_alleles) =
            join_gvcf_annotations(&mut records, gvcf, gvcf_path, contig)?;
    }
    Ok(ContigRecords { records, skipped, missing_annotations, other_alleles })
}

/// Sets each record's raw annotations from the gVCF site at its position with its alleles (see
/// [`matching_site`]), returning the number of records without a site, and the number that took
/// a site with other alleles.
fn join_gvcf_annotations(
    records: &mut [RawRecord],
    gvcf: &mut IndexedReader,
    gvcf_path: &Path,
    contig: &str,
) -> Result<(usize, usize)> {
    // Only parse the gVCF sites that the VCF has calls at.
    let positions: HashSet<u64> = records.iter().map(|r| r.pos).collect();
    let sites = read_gvcf_sites(gvcf, gvcf_path, contig, &positions)?;
    let (mut missing, mut other_alleles) = (0, 0);
    for record in records {
        let site = sites
            .get(&record.pos)
            .and_then(|sites| matching_site(sites, &record.reference, &record.alts));
        match site {
            Some(site) => {
                if !site.has_alleles(&record.reference, &record.alts) {
                    other_alleles += 1;
                }
                record.raw = site.annotation;
            }
            None => missing += 1,
        }
    }
    Ok((missing, other_alleles))
}

/// Reads one record's raw values, or `None` if it is neither a SNP nor an indel. The raw
/// annotations are left missing.
fn raw_record(
    fields: &mut FieldReader,
    record: &Record,
    alleles: &[&[u8]],
    chrom: &Arc<str>,
) -> Result<Option<RawRecord>> {
    let (reference, alts) = alleles.split_first().map_or((&b""[..], &[][..]), |(r, a)| (*r, a));
    let Some(variant_type) = variant_type_of(reference, alts) else {
        return Ok(None);
    };

    let first_alt = first_alt_index(alts);
    let element = |values: Option<&[f64]>, index: usize| {
        values.and_then(|v| v.get(index).copied()).unwrap_or(f64::NAN)
    };
    let (ad_ref, ad_alt, ad_alt_total) = {
        let ad = fields.format(record, b"AD")?;
        let known_alt_depths = ad.map(|ad| ad.iter().skip(1).copied().filter(|d| !d.is_nan()));
        let ad_alt_total = match known_alt_depths {
            Some(depths) => depths.reduce(|a, b| a + b).unwrap_or(f64::NAN),
            None => f64::NAN,
        };
        let ad_alt = first_alt.map_or(f64::NAN, |index| element(ad, index + 1));
        (element(ad, 0), ad_alt, ad_alt_total)
    };
    let sb: [f64; 4] = {
        let sb = fields.format(record, b"SB")?;
        std::array::from_fn(|i| element(sb, i))
    };
    let pl: [f64; 3] = {
        let pl = fields.format(record, b"PL")?;
        std::array::from_fn(|i| element(pl, i))
    };
    let gq = element(fields.format(record, b"GQ")?, 0);
    let (str_period, str_repeats) = {
        let dragstr = fields.info(record, b"DRAGstrInfo")?;
        (element(dragstr, 0), element(dragstr, 1))
    };
    Ok(Some(RawRecord {
        chrom: Arc::clone(chrom),
        pos: u64::try_from(record.pos() + 1).context("negative record position")?,
        reference: reference.to_vec(),
        alts: alts.iter().map(|alt| alt.to_vec()).collect(),
        filter: filter_string(record),
        variant_type,
        genotype: fields.genotype(record),
        qual: crate::vcf::qual(record),
        dp: fields.info_value(record, b"DP", 0)?,
        mq_rank_sum: fields.info_value(record, b"MQRankSum", 0)?,
        read_pos_rank_sum: fields.info_value(record, b"ReadPosRankSum", 0)?,
        base_q_rank_sum: fields.info_value(record, b"BaseQRankSum", 0)?,
        str_period,
        str_repeats,
        gq,
        ad_ref,
        ad_alt,
        sb_ref_fwd: sb[0],
        sb_ref_rev: sb[1],
        sb_alt_fwd: sb[2],
        sb_alt_rev: sb[3],
        excess_het: fields.info_value(record, b"ExcessHet", 0)?,
        pl_ref: pl[0],
        pl_het: pl[1],
        pl_hom: pl[2],
        ad_alt_total,
        raw: RawAnnotation::MISSING,
    }))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::testing::{CALL_FIELDS, RAW_FIELDS, vcf_header, write_indexed_vcf};

    /// Turns a space-separated record into a VCF line.
    fn tabs(line: &str) -> String {
        line.split(' ').collect::<Vec<_>>().join("\t")
    }

    fn write_vcf(
        dir: &TempDir,
        name: &str,
        fields: &[&str],
        samples: &[&str],
        records: &[&str],
    ) -> PathBuf {
        let lines: Vec<String> = records.iter().map(|r| tabs(r)).collect();
        let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
        write_indexed_vcf(dir.path(), name, &vcf_header(fields, samples), &lines)
    }

    fn assert_nan(value: f64, what: &str) {
        assert!(value.is_nan(), "{what}: expected NaN, got {value}");
    }

    const GVCF_FIELDS: &str = "\
##ALT=<ID=NON_REF,Description=\"Any other allele\">
##INFO=<ID=END,Number=1,Type=Integer,Description=\"End of the block\">
";

    const VCF_WITH_RAW: [&str; 6] = [
        "chr1 100 . A C 50.5 PASS DP=30;MQRankSum=0.5;ReadPosRankSum=-1.5;BaseQRankSum=2;RAW_MQandDP=54000,30;QUALapprox=60;VarDP=15;AS_RAW_MQ=27000|27000;DRAGstrInfo=1,5;ExcessHet=3.01 GT:GQ:AD:SB:PL 0|1:99:15,15:7,8,9,6:60,0,70",
        "chr1 200 . AT A . LowQual;RefCall DP=10 GT:AD 1/1:.,4",
        "chr1 300 . AC GT 10 PASS . GT 0/1",
        "chr1 400 . G *,T 20 . . GT:AD:PL 1/2:3,2,4:0,1,2,3,4,5",
        "chr1 500 . G . 5 . . GT 0/0",
        "chr2 50 . C CA 30 PASS RAW_MQandDP=3600,1 GT ./.",
    ];

    fn extract(vcf: &Path, gvcf: Option<&Path>, contigs: &[&str]) -> Result<Vec<RawRecord>> {
        extract_raw_records(vcf, gvcf, contigs)
    }

    #[test]
    fn reads_records_across_shard_boundaries_once_and_in_order() {
        // chr1 spans three shards: a deletion straddles the first boundary, and a record lies past
        // the declared length; the gVCF has a reference block across the boundary too.
        let header = |fields: &[&str]| {
            vcf_header(fields, &["S1"])
                .replace("##contig=<ID=chr1,length=100000>", "##contig=<ID=chr1,length=12000000>")
        };
        let calls = [
            "chr1 4999998 . ACGTA A 50 PASS . GT 0/1",
            "chr1 5000001 . G T 50 PASS . GT 0/1",
            "chr1 10000000 . C A 50 PASS . GT 1/1",
            "chr1 12000005 . T G 50 PASS . GT 0/1",
            "chr2 50 . C CA 30 PASS . GT 0/1",
        ];
        let sites = [
            "chr1 4999990 . A <NON_REF> . . END=4999997 GT 0/0",
            "chr1 4999998 . ACGTA A,<NON_REF> 50 . RAW_MQandDP=36000,10;QUALapprox=50;VarDP=10 GT 0/1",
            "chr1 4999999 . C <NON_REF> . . END=5000000 GT 0/0",
            "chr1 5000001 . G T,<NON_REF> 50 . RAW_MQandDP=36000,10;QUALapprox=51;VarDP=10 GT 0/1",
            "chr1 10000000 . C A,<NON_REF> 50 . RAW_MQandDP=36000,10;QUALapprox=52;VarDP=10 GT 1/1",
            "chr1 12000005 . T G,<NON_REF> 50 . RAW_MQandDP=36000,10;QUALapprox=53;VarDP=10 GT 0/1",
            "chr2 50 . C CA,<NON_REF> 30 . RAW_MQandDP=36000,10;QUALapprox=54;VarDP=10 GT 0/1",
        ];
        let dir = TempDir::new().unwrap();
        let write = |name: &str, fields: &[&str], records: &[&str]| {
            let lines: Vec<String> = records.iter().map(|r| tabs(r)).collect();
            let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
            write_indexed_vcf(dir.path(), name, &header(fields), &lines)
        };
        let vcf = write("in.vcf.gz", &[CALL_FIELDS], &calls);
        let gvcf = write("in.g.vcf.gz", &[CALL_FIELDS, RAW_FIELDS, GVCF_FIELDS], &sites);
        let records = extract(&vcf, Some(&gvcf), &["chr1", "chr2"]).unwrap();
        let keys: Vec<(&str, u64)> = records.iter().map(|r| (&*r.chrom, r.pos)).collect();
        assert_eq!(
            keys,
            [
                ("chr1", 4_999_998),
                ("chr1", 5_000_001),
                ("chr1", 10_000_000),
                ("chr1", 12_000_005),
                ("chr2", 50)
            ]
        );
        let qual_approx: Vec<f64> = records.iter().map(|r| r.raw.qual_approx).collect();
        assert_eq!(qual_approx, [50.0, 51.0, 52.0, 53.0, 54.0]);
    }

    #[test]
    fn reads_raw_values_and_annotations_from_the_vcf() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &VCF_WITH_RAW);
        let records = extract(&vcf, None, &["chr1", "chr2"]).unwrap();
        let keys: Vec<(&str, u64)> = records.iter().map(|r| (&*r.chrom, r.pos)).collect();
        // The MNP at 300 and the ALT-less record at 500 are skipped.
        assert_eq!(keys, [("chr1", 100), ("chr1", 200), ("chr1", 400), ("chr2", 50)]);

        let snp = &records[0];
        assert_eq!((snp.reference.as_slice(), snp.alts.clone()), (&b"A"[..], vec![b"C".to_vec()]));
        assert_eq!(snp.filter, "PASS");
        assert_eq!(snp.variant_type, VariantType::Snp);
        assert_eq!(snp.genotype, Genotype::HET);
        let values = [
            snp.qual,
            snp.dp,
            snp.mq_rank_sum,
            snp.read_pos_rank_sum,
            snp.base_q_rank_sum,
            snp.str_period,
            snp.str_repeats,
            snp.gq,
            snp.ad_ref,
            snp.ad_alt,
            snp.sb_ref_fwd,
            snp.sb_ref_rev,
            snp.sb_alt_fwd,
            snp.sb_alt_rev,
            snp.excess_het,
            snp.pl_ref,
            snp.pl_het,
            snp.pl_hom,
            snp.ad_alt_total,
        ];
        let expected = [
            50.5,
            30.0,
            0.5,
            -1.5,
            2.0,
            1.0,
            5.0,
            99.0,
            15.0,
            15.0,
            7.0,
            8.0,
            9.0,
            6.0,
            f64::from(3.01f32),
            60.0,
            0.0,
            70.0,
            15.0,
        ];
        assert_eq!(values, expected);
        let raw = snp.raw;
        assert_eq!(
            [
                raw.raw_mq,
                raw.raw_dp,
                raw.qual_approx,
                raw.var_dp,
                raw.as_mq_ref,
                raw.as_mq_alt,
                raw.n_alt
            ],
            [54000.0, 30.0, 60.0, 15.0, 27000.0, 27000.0, 1.0]
        );

        // Missing values are NaN: QUAL, a missing AD element, unset fields and annotations.
        let deletion = &records[1];
        assert_eq!(deletion.filter, "LowQual;RefCall");
        assert_eq!(deletion.variant_type, VariantType::Indel);
        assert_eq!(deletion.genotype, Genotype::HOM_ALT);
        assert_eq!((deletion.dp, deletion.ad_alt, deletion.ad_alt_total), (10.0, 4.0, 4.0));
        for (value, what) in [
            (deletion.qual, "qual"),
            (deletion.ad_ref, "ad_ref"),
            (deletion.gq, "gq"),
            (deletion.sb_ref_fwd, "sb"),
            (deletion.pl_ref, "pl"),
            (deletion.str_period, "str_period"),
            (deletion.raw.raw_mq, "raw_mq"),
            (deletion.raw.n_alt, "n_alt"),
        ] {
            assert_nan(value, what);
        }

        // AD of the first non-`*` alternate, and every alternate's depth summed.
        let star = &records[2];
        assert_eq!(star.variant_type, VariantType::Snp);
        assert_eq!(star.filter, "PASS");
        assert_eq!(star.genotype, Genotype::diploid(1, 2));
        assert_eq!((star.ad_ref, star.ad_alt, star.ad_alt_total), (3.0, 4.0, 6.0));
        assert_eq!((star.pl_ref, star.pl_het, star.pl_hom), (0.0, 1.0, 2.0));

        let insertion = &records[3];
        assert_eq!(insertion.genotype, Genotype::MISSING);
        assert_eq!((insertion.raw.raw_mq, insertion.raw.raw_dp), (3600.0, 1.0));
        assert_nan(insertion.raw.qual_approx, "qual_approx");
        assert_nan(insertion.raw.as_mq_ref, "as_mq_ref");
    }

    #[test]
    fn records_follow_the_requested_contig_order() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &VCF_WITH_RAW);
        let records = extract(&vcf, None, &["chr2", "chr1"]).unwrap();
        let chroms: Vec<&str> = records.iter().map(|r| &*r.chrom).collect();
        assert_eq!(chroms, ["chr2", "chr1", "chr1", "chr1"]);
        assert!(extract(&vcf, None, &[] as &[&str]).unwrap().is_empty());
    }

    #[test]
    fn reads_float_declared_annotations() {
        let dir = TempDir::new().unwrap();
        let fields = RAW_FIELDS
            .replace("ID=RAW_MQandDP,Number=2,Type=Integer", "ID=RAW_MQandDP,Number=2,Type=Float")
            .replace("ID=AS_RAW_MQ,Number=1,Type=String", "ID=AS_RAW_MQ,Number=R,Type=Float");
        let records = ["chr1 100 . A C,G 50 PASS RAW_MQandDP=10.5,2;AS_RAW_MQ=1,.,3 GT 1/2"];
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, &fields], &["S1"], &records);
        let raw = extract(&vcf, None, &["chr1"]).unwrap()[0].raw;
        assert_eq!((raw.raw_mq, raw.raw_dp), (10.5, 2.0));
        assert_eq!((raw.as_mq_ref, raw.as_mq_alt, raw.n_alt), (1.0, 3.0, 2.0));
    }

    #[test]
    fn joins_annotations_from_the_gvcf() {
        let dir = TempDir::new().unwrap();
        let vcf_records = [
            "chr1 100 . A C 50 PASS . GT 0/1",
            "chr1 200 . A G 50 PASS . GT 0/1",
            "chr1 300 . A G 50 PASS . GT 1/1",
        ];
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS], &["S1"], &vcf_records);
        let gvcf_records = [
            "chr1 1 . A <NON_REF> . . END=99 GT 0/0",
            "chr1 100 . A C,<NON_REF> 50 . RAW_MQandDP=100,4;QUALapprox=50;VarDP=4;AS_RAW_MQ=40|60|0 GT 0/1",
            "chr1 100 . A AT,<NON_REF> 50 . RAW_MQandDP=999,9;QUALapprox=9;VarDP=9;AS_RAW_MQ=9|9|9 GT 0/1",
            "chr1 200 . A <NON_REF> . . END=250 GT 0/0",
            "chr1 200 . A G,<NON_REF> 40 . RAW_MQandDP=200,8;QUALapprox=40;VarDP=8;AS_RAW_MQ=100|100|0 GT 0/1",
            "chr1 301 . A <NON_REF> . . END=400 GT 0/0",
        ];
        let gvcf = write_vcf(
            &dir,
            "in.g.vcf.gz",
            &[CALL_FIELDS, RAW_FIELDS, GVCF_FIELDS],
            &["S1"],
            &gvcf_records,
        );
        let records = extract(&vcf, Some(&gvcf), &["chr1", "chr2"]).unwrap();
        assert_eq!(records.len(), 3);
        // The first variant site at a position wins, and <NON_REF> is not an alternate.
        let first = records[0].raw;
        assert_eq!(
            [
                first.raw_mq,
                first.raw_dp,
                first.qual_approx,
                first.var_dp,
                first.as_mq_ref,
                first.as_mq_alt,
                first.n_alt
            ],
            [100.0, 4.0, 50.0, 4.0, 40.0, 60.0, 1.0]
        );
        // A reference block at a call's position is skipped.
        assert_eq!(records[1].raw.raw_mq, 200.0);
        // No gVCF variant site at 300.
        assert_nan(records[2].raw.raw_mq, "raw_mq without a gVCF site");
    }

    #[test]
    fn a_vcf_with_annotations_ignores_the_gvcf() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &VCF_WITH_RAW);
        // The gVCF is not read, so it need not exist.
        let records = extract(&vcf, Some(&dir.path().join("missing.g.vcf.gz")), &["chr1"]).unwrap();
        assert_eq!(records[0].raw.raw_mq, 54000.0);
    }

    fn error(vcf: &Path, gvcf: Option<&Path>, contigs: &[&str]) -> String {
        format!("{:#}", extract(vcf, gvcf, contigs).unwrap_err())
    }

    #[test]
    fn rejects_bad_inputs() {
        let dir = TempDir::new().unwrap();
        let record = ["chr1 100 . A C 50 PASS . GT 0/1"];
        let two_samples = write_vcf(
            &dir,
            "two.vcf.gz",
            &[CALL_FIELDS, RAW_FIELDS],
            &["S1", "S2"],
            &["chr1 100 . A C 50 PASS . GT 0/1 0/1"],
        );
        assert!(
            error(&two_samples, None, &["chr1"]).ends_with("must have exactly one sample, has 2")
        );

        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &record);
        let message = error(&vcf, None, &["chr1", "1", "chrX"]);
        assert!(
            message.starts_with("contig(s) not in the header of ")
                && message.ends_with(": 1, chrX"),
            "{message}"
        );

        let no_gt_fields = CALL_FIELDS
            .replace("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n", "");
        let no_gt = write_vcf(
            &dir,
            "nogt.vcf.gz",
            &[&no_gt_fields, RAW_FIELDS],
            &["S1"],
            &["chr1 100 . A C 50 PASS . GQ 5"],
        );
        assert!(error(&no_gt, None, &["chr1"]).ends_with("has no FORMAT/GT"));

        let no_raw = write_vcf(&dir, "noraw.vcf.gz", &[CALL_FIELDS], &["S1"], &record);
        let message = error(&no_raw, None, &["chr1"]);
        assert!(message.contains("lacks the INFO annotations RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ and no gVCF was given"), "{message}");

        let bad_gvcf = write_vcf(&dir, "bad.g.vcf.gz", &[CALL_FIELDS], &["S1"], &record);
        let message = error(&no_raw, Some(&bad_gvcf), &["chr1"]);
        assert!(
            message.contains("bad.g.vcf.gz lacks the INFO annotations RAW_MQandDP"),
            "{message}"
        );
        let message = error(&no_raw, Some(&bad_gvcf), &["chr1", "chr9"]);
        // The VCF's contigs are checked before the gVCF is opened.
        assert!(message.ends_with("noraw.vcf.gz: chr9"), "{message}");

        let unindexed = dir.path().join("plain.vcf");
        std::fs::write(&unindexed, vcf_header(&[CALL_FIELDS], &["S1"])).unwrap();
        let message = error(&unindexed, None, &["chr1"]);
        assert!(
            message.ends_with("plain.vcf is not bgzipped: compress it with bgzip and index it"),
            "{message}"
        );
        let message = error(&dir.path().join("absent.vcf.gz"), None, &["chr1"]);
        assert!(message.ends_with("absent.vcf.gz does not exist or is not a file"), "{message}");
    }

    #[test]
    fn a_call_takes_the_gvcf_site_with_its_alleles() {
        let dir = TempDir::new().unwrap();
        let vcf_records = ["chr1 100 . A AT 50 PASS . GT 0/1", "chr1 200 . A G 50 PASS . GT 0/1"];
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS], &["S1"], &vcf_records);
        let gvcf_records = [
            "chr1 100 . A C,<NON_REF> 50 . RAW_MQandDP=100,4;QUALapprox=50;VarDP=4;AS_RAW_MQ=40|60|0 GT 0/1",
            "chr1 100 . A AT,<NON_REF> 50 . RAW_MQandDP=900,9;QUALapprox=9;VarDP=9;AS_RAW_MQ=9|9|0 GT 0/1",
            "chr1 200 . A C,<NON_REF> 50 . RAW_MQandDP=300,3;QUALapprox=3;VarDP=3;AS_RAW_MQ=3|3|0 GT 0/1",
        ];
        let gvcf = write_vcf(
            &dir,
            "in.g.vcf.gz",
            &[CALL_FIELDS, RAW_FIELDS, GVCF_FIELDS],
            &["S1"],
            &gvcf_records,
        );
        let records = extract(&vcf, Some(&gvcf), &["chr1"]).unwrap();
        // The indel takes the indel site, not the SNP site before it at the same position.
        assert_eq!(records[0].raw.raw_mq, 900.0);
        // Without a site with its alleles, a call takes the first site at its position.
        assert_eq!(records[1].raw.raw_mq, 300.0);
    }

    #[test]
    fn calls_taking_a_gvcf_site_with_other_alleles_are_counted() {
        let dir = TempDir::new().unwrap();
        let vcf_records = ["chr1 100 . A AT 50 PASS . GT 0/1", "chr1 200 . A G 50 PASS . GT 0/1"];
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS], &["S1"], &vcf_records);
        let gvcf_records = [
            "chr1 100 . A AT,<NON_REF> 50 . RAW_MQandDP=900,9;QUALapprox=9;VarDP=9;AS_RAW_MQ=9|9|0 GT 0/1",
            "chr1 200 . A C,<NON_REF> 50 . RAW_MQandDP=300,3;QUALapprox=3;VarDP=3;AS_RAW_MQ=3|3|0 GT 0/1",
        ];
        let gvcf = write_vcf(
            &dir,
            "in.g.vcf.gz",
            &[CALL_FIELDS, RAW_FIELDS, GVCF_FIELDS],
            &["S1"],
            &gvcf_records,
        );
        let read = read_shards(
            &vcf,
            AnnotationSource::Gvcf(&gvcf),
            &[ContigShard { index: 0, contig: "chr1", start: 0, end: None }],
        )
        .unwrap()
        .pop()
        .unwrap();
        assert_eq!((read.records.len(), read.missing_annotations, read.other_alleles), (2, 0, 1));
    }

    #[test]
    fn a_vcf_declaring_annotations_none_of_its_records_has_falls_back_to_the_gvcf() {
        let dir = TempDir::new().unwrap();
        let records = ["chr1 100 . A C 50 PASS DP=10 GT 0/1", "chr1 200 . A G 50 PASS . GT 1/1"];
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &records);
        let message = error(&vcf, None, &["chr1"]);
        assert!(
            message.ends_with(
                "declares the raw annotations, but none of its 2 records has RAW_MQandDP and no \
                 gVCF was given: pass the gVCF the VCF was genotyped from"
            ),
            "{message}"
        );
        let gvcf_records = [
            "chr1 100 . A C,<NON_REF> 50 . RAW_MQandDP=100,4;QUALapprox=50;VarDP=4;AS_RAW_MQ=40|60|0 GT 0/1",
            "chr1 200 . A G,<NON_REF> 50 . RAW_MQandDP=200,8;QUALapprox=40;VarDP=8;AS_RAW_MQ=100|100|0 GT 1/1",
        ];
        let gvcf = write_vcf(
            &dir,
            "in.g.vcf.gz",
            &[CALL_FIELDS, RAW_FIELDS, GVCF_FIELDS],
            &["S1"],
            &gvcf_records,
        );
        let records = extract(&vcf, Some(&gvcf), &["chr1"]).unwrap();
        let raw_mq: Vec<f64> = records.iter().map(|r| r.raw.raw_mq).collect();
        assert_eq!(raw_mq, [100.0, 200.0]);
        // Some records with the annotations: the VCF's are kept, and the others are missing.
        let records =
            ["chr1 100 . A C 50 PASS RAW_MQandDP=7,1 GT 0/1", "chr1 200 . A G 50 PASS . GT 1/1"];
        let some = write_vcf(&dir, "some.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &records);
        let records = extract(&some, Some(&gvcf), &["chr1"]).unwrap();
        assert_eq!(records[0].raw.raw_mq, 7.0);
        assert_nan(records[1].raw.raw_mq, "raw_mq of a record without it");
    }

    #[test]
    fn rejects_repeated_contigs() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &VCF_WITH_RAW);
        let message = error(&vcf, None, &["chr1", "chr2", "chr1"]);
        assert_eq!(message, "contigs must be unique; repeated: chr1");
    }

    #[test]
    fn a_header_line_that_is_not_utf8_is_an_error() {
        let dir = TempDir::new().unwrap();
        let latin1 = "##INFO=<ID=XT,Number=1,Type=Float,Description=\"temperature in \u{b0}C\">\n";
        let header = vcf_header(&[CALL_FIELDS, RAW_FIELDS, latin1], &["S1"]);
        // The degree sign as one Latin-1 byte, not UTF-8.
        let header = header.replace('\u{b0}', "\u{0}");
        let mut bytes = header.into_bytes();
        let at = bytes.iter().position(|&b| b == 0).unwrap();
        bytes[at] = 0xb0;
        let path = dir.path().join("latin1.vcf.gz");
        let mut writer = rust_htslib::bgzf::Writer::from_path(&path).unwrap();
        std::io::Write::write_all(&mut writer, &bytes).unwrap();
        std::io::Write::write_all(&mut writer, tabs(VCF_WITH_RAW[0]).as_bytes()).unwrap();
        std::io::Write::write_all(&mut writer, b"\n").unwrap();
        drop(writer);
        rust_htslib::bcf::index::build(&path, None, 1, rust_htslib::bcf::index::Type::Tbx).unwrap();
        let message = error(&path, None, &["chr1"]);
        assert!(message.contains("latin1.vcf.gz has a line that is not UTF-8"), "{message}");
    }

    #[test]
    fn string_annotations_split_on_commas_and_bars() {
        let dir = TempDir::new().unwrap();
        let records = ["chr1 100 . A C,G 50 PASS RAW_MQandDP=10,2;AS_RAW_MQ=1,2|3 GT 1/2"];
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &records);
        let raw = extract(&vcf, None, &["chr1"]).unwrap()[0].raw;
        assert_eq!((raw.as_mq_ref, raw.as_mq_alt, raw.n_alt), (1.0, 5.0, 2.0));
    }

    #[test]
    fn finds_indexes_as_htslib_does() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &VCF_WITH_RAW);
        let index = dir.path().join("in.vcf.gz.tbi");
        // The index with the data file's extension replaced.
        let replaced = dir.path().join("in.vcf.tbi");
        std::fs::rename(&index, &replaced).unwrap();
        assert_eq!(extract(&vcf, None, &["chr1"]).unwrap().len(), 3);
        std::fs::rename(&replaced, dir.path().join("elsewhere.tbi")).unwrap();
        let message = error(&vcf, None, &["chr1"]);
        assert!(message.contains("has no index"), "{message}");
    }

    #[test]
    fn rejects_inputs_that_crash_htslib_rather_than_crashing() {
        let dir = TempDir::new().unwrap();
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &VCF_WITH_RAW);
        let index = std::fs::read(dir.path().join("in.vcf.gz.tbi")).unwrap();

        // Plain gzip (not BGZF) with a stray index.
        let text = format!(
            "{}chr1\t100\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n",
            vcf_header(&[CALL_FIELDS], &["S1"])
        );
        let gzipped = dir.path().join("gz.vcf.gz");
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, text.as_bytes()).unwrap();
        std::fs::write(&gzipped, encoder.finish().unwrap()).unwrap();
        std::fs::write(dir.path().join("gz.vcf.gz.tbi"), &index).unwrap();
        let message = error(&gzipped, None, &["chr1"]);
        assert!(
            message.ends_with("gz.vcf.gz is not bgzipped: compress it with bgzip and index it"),
            "{message}"
        );

        // A corrupt index.
        let corrupt =
            write_vcf(&dir, "corrupt.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &VCF_WITH_RAW);
        std::fs::write(dir.path().join("corrupt.vcf.gz.tbi"), b"not an index").unwrap();
        let message = error(&corrupt, None, &["chr1"]);
        assert!(message.ends_with("corrupt.vcf.gz.tbi is not a tabix or CSI index"), "{message}");

        // A file that is not a VCF.
        let not_vcf = dir.path().join("reads.txt.gz");
        let mut writer = rust_htslib::bgzf::Writer::from_path(&not_vcf).unwrap();
        std::io::Write::write_all(&mut writer, b"just some text\n").unwrap();
        drop(writer);
        std::fs::write(dir.path().join("reads.txt.gz.tbi"), &index).unwrap();
        let message = error(&not_vcf, None, &["chr1"]);
        assert!(message.contains("as a VCF or BCF"), "{message}");

        let message = error(Path::new("s3://bucket/in.vcf.gz"), None, &["chr1"]);
        assert_eq!(
            message,
            "s3://bucket/in.vcf.gz is remote; only local VCF/BCF files are supported"
        );
        drop(vcf);
    }
}
