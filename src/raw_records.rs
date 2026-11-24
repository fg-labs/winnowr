//! Raw per-record values of a HaplotypeCaller VCF, with the raw annotations from the VCF itself or
//! from the gVCF it was genotyped from, or derived from the VCF's standard annotations (see
//! [`AnnotationInputs`]).
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
use crate::feature_source::FeatureSource;
use crate::gvcf::{
    RawAnnotation, check_gvcf_header, has_raw_annotations, matching_site, missing_raw_annotations,
    raw_annotation, read_gvcf_sites,
};
use crate::shards::{ContigShard, ReaderPool, contig_shards};
use crate::vcf::{
    DeclaredIds, FieldReader, check_contigs, fetch_span, field_name, filter_string, has_field,
    has_format, has_info, open_indexed,
};
use crate::vcf_source;

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
    ///
    /// The PL features are those of allele 1 even when it is `*` (e.g. `0/*` for `G>*,T`), unlike
    /// [`RawRecord::ad_alt`], which skips `*`. Models are trained on these values, so they are
    /// kept: taking the PLs of the first non-`*` allele would change the features existing models
    /// see.
    pub pl_het: f64,
    /// The `1/1` likelihood, FORMAT/PL\[2\] (allele 1's, as for [`RawRecord::pl_het`]).
    pub pl_hom: f64,
    /// The summed depth of every alternate allele (NaN if none is known).
    pub ad_alt_total: f64,
    /// The raw annotations ([`RawAnnotation::MISSING`] when there are none).
    pub raw: RawAnnotation,
}

/// Where the raw annotations of the records come from: which [`FeatureSource`] their features
/// are of, and its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationInputs<'a> {
    /// [`FeatureSource::Gvcf`]: the raw annotations of the VCF's own records when its header
    /// declares them, or else of this indexed gVCF the VCF was genotyped from.
    Gvcf(Option<&'a Path>),
    /// [`FeatureSource::Vcf`]: derived from the VCF's standard annotations (see
    /// [`crate::vcf_source`]); the header must declare INFO/MQ, INFO/DP, FORMAT/AD, FORMAT/PL, and
    /// FORMAT/SB or INFO/SOR.
    Vcf,
}

impl AnnotationInputs<'_> {
    /// The feature source of the annotations.
    #[must_use]
    pub fn feature_source(&self) -> FeatureSource {
        match self {
            AnnotationInputs::Gvcf(_) => FeatureSource::Gvcf,
            AnnotationInputs::Vcf => FeatureSource::Vcf,
        }
    }

    /// The gVCF the gVCF source may read, if any (`None` for the VCF source).
    #[must_use]
    pub fn gvcf(&self) -> Option<&Path> {
        match self {
            AnnotationInputs::Gvcf(gvcf) => *gvcf,
            AnnotationInputs::Vcf => None,
        }
    }
}

impl Default for AnnotationInputs<'_> {
    /// The gVCF source, with no gVCF (the VCF must keep the raw annotations).
    fn default() -> Self {
        AnnotationInputs::Gvcf(None)
    }
}

/// Reads the raw values of every SNP and indel record of a single-sample VCF on `contigs`.
///
/// With [`AnnotationInputs::Gvcf`], the raw annotations come from the VCF's own records when its
/// header declares all of them (any gVCF is then not read, unless no record has them), and
/// otherwise from the gVCF, joined by position. Records without them (no `RAW_MQandDP`, or no
/// gVCF site) get [`RawAnnotation::MISSING`]. With [`AnnotationInputs::Vcf`], they are derived
/// from each record's standard annotations (see [`crate::vcf_source::derive_annotation`]). Records that are neither SNPs
/// nor indels (see [`variant_type_of`]) are skipped.
///
/// # Arguments
///
/// * `vcf` - the indexed single-sample VCF.
/// * `annotations` - where the raw annotations come from.
/// * `contigs` - the contigs to read, in order.
///
/// # Errors
///
/// Returns an error if a contig is repeated, the VCF does not have exactly one sample or
/// FORMAT/GT, a contig is not in the VCF or gVCF header, neither the VCF nor a gVCF supplies the
/// raw annotations (including a VCF whose header declares them but none of whose records has
/// them, without a gVCF), the VCF source's VCF does not declare the fields it needs (see
/// [`crate::vcf_source::check_header`]) or none of its records has INFO/MQ and a positive
/// INFO/DP, or a file cannot be read.
pub fn extract_raw_records<P, S>(
    vcf: P,
    annotations: AnnotationInputs<'_>,
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
        // The source's own checks first, so that the VCF source's missing fields are an error,
        // not first logged as features missing for every call.
        let source = match annotations {
            AnnotationInputs::Vcf => {
                vcf_source::check_header(header, vcf)?;
                AnnotationSource::DerivedFromVcf
            }
            AnnotationInputs::Gvcf(gvcf) => gvcf_annotation_source(header, vcf, gvcf, contigs)?,
        };
        // INFO/SOR stands in for FORMAT/SB for the VCF source.
        let skip_sb = annotations == AnnotationInputs::Vcf && has_info(header, b"SOR");
        warn_undeclared_fields(header, vcf, skip_sb);
        source
    };
    let gvcf = annotations.gvcf();

    // Contigs are read in shards of POS spans, in parallel; a contig's last shard is open-ended,
    // so a record past the declared length is still read.
    let summary = crate::vcf::VcfSummary::read(vcf)?;
    let shards = contig_shards(&summary, contigs, RECORD_SHARD_SPAN);
    let mut per_shard = read_shards(vcf, annotation_source, &shards)?;
    let n_records: usize = per_shard.iter().map(|shard| shard.records.len()).sum();
    let n_missing: usize = per_shard.iter().map(|shard| shard.missing_annotations).sum();
    let all_missing = n_records > 0 && n_missing == n_records;
    if all_missing && matches!(annotation_source, AnnotationSource::DerivedFromVcf) {
        bail!(
            "VCF {} declares INFO/MQ and INFO/DP, but none of its {n_records} records has both \
             (with a positive INFO/DP), so the vcf feature source has no mapping quality or depth \
             for any call",
            vcf.display()
        );
    }
    // A header can declare the raw annotations that no record has (e.g. after a tool dropped
    // them): every raw feature would be missing.
    if all_missing && matches!(annotation_source, AnnotationSource::RawFromVcf) {
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
        annotation_source = AnnotationSource::RawFromGvcf(gvcf);
        per_shard = read_shards(vcf, annotation_source, &shards)?;
    }

    // A read by region skips records at POS 0 (telomeres): say so rather than drop them silently.
    match crate::vcf::tabix_scan(vcf, Some(contigs)).and_then(|scan| scan.pos0) {
        Some(found) if found.is_empty() => {}
        Some(found) => warn!(
            "VCF {} has records at POS 0 on {}; they are not read, and so not scored (htslib's \
             warning about the -0 option is about them, and harmless)",
            vcf.display(),
            found.join(", ")
        ),
        None => info!(
            "Could not check VCF {} for records at POS 0 (e.g. it is a BCF); any are not read",
            vcf.display()
        ),
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
        let what = match annotation_source {
            AnnotationSource::RawFromVcf => "lack RAW_MQandDP (set to missing)",
            AnnotationSource::RawFromGvcf(_) => "had no gVCF annotations (set to missing)",
            AnnotationSource::DerivedFromVcf => {
                "lack INFO/MQ, or a positive INFO/DP (their mq, and without INFO/DP their \
                 dp_norm, is missing)"
            }
        };
        warn!("{missing} records {what}");
    }
    if matches!(annotation_source, AnnotationSource::DerivedFromVcf) {
        warn_missing_vcf_source_values(&records);
    }
    if other_alleles > 0 {
        warn!(
            "{other_alleles} records took the raw annotations of the first gVCF site at their \
             position, which has other alleles (no site there has theirs)"
        );
    }
    Ok(records)
}

/// Warns about the VCF source's records that lack a field the header declares: FORMAT/PL (their
/// `qd` is missing), or both FORMAT/SB and INFO/SOR (their `sor` is).
fn warn_missing_vcf_source_values(records: &[RawRecord]) {
    let no_pl = records.iter().filter(|record| record.raw.qual_approx.is_nan()).count();
    if no_pl > 0 {
        warn!("{no_pl} records lack FORMAT/PL (their qd is missing)");
    }
    let no_sor = records
        .iter()
        .filter(|r| {
            let sb = [r.sb_ref_fwd, r.sb_ref_rev, r.sb_alt_fwd, r.sb_alt_rev];
            sb.iter().any(|count| count.is_nan()) && r.raw.sor.is_nan()
        })
        .count();
    if no_sor > 0 {
        warn!("{no_sor} records lack both FORMAT/SB and INFO/SOR (their sor is missing)");
    }
}

/// Where the gVCF source reads the raw annotations from: the VCF's records when its header
/// declares them, or else the gVCF.
///
/// # Errors
///
/// Returns an error if the VCF lacks the raw annotations and no gVCF is given, or the gVCF's
/// header is not one (see [`check_gvcf_header`]).
fn gvcf_annotation_source<'a, S: AsRef<str>>(
    header: &rust_htslib::bcf::header::HeaderView,
    vcf: &Path,
    gvcf: Option<&'a Path>,
    contigs: &[S],
) -> Result<AnnotationSource<'a>> {
    Ok(match (has_raw_annotations(header), gvcf) {
        (true, gvcf) => {
            if let Some(gvcf) = gvcf {
                warn!(
                    "VCF {} declares the raw annotations, so they are read from its records; \
                     the gVCF {} is read only if none of them has them",
                    vcf.display(),
                    gvcf.display()
                );
            }
            AnnotationSource::RawFromVcf
        }
        (false, Some(gvcf)) => {
            check_gvcf_header(open_indexed(gvcf)?.header(), gvcf, contigs)?;
            info!("Reading the raw annotations from the gVCF {}", gvcf.display());
            AnnotationSource::RawFromGvcf(gvcf)
        }
        (false, None) => bail!(
            "VCF {} lacks the INFO annotations {} and no gVCF was given: pass the gVCF the \
             VCF was genotyped from (--gvcf), or a VCF that keeps the raw annotations of \
             HaplotypeCaller's gVCF mode. To use VCFs without them, extract every training and \
             evaluation table with --feature-source vcf, and score with the models trained on \
             those tables",
            vcf.display(),
            missing_raw_annotations(header).join(", ")
        ),
    })
}

/// Checks that no contig is listed twice, as its records would be read twice.
///
/// # Errors
///
/// Returns an error naming the repeated contigs.
pub fn check_unique_contigs<S: AsRef<str>>(contigs: &[S]) -> Result<()> {
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
    (SB, false, false),
    ("MQRankSum", true, false),
    ("ReadPosRankSum", true, false),
    ("BaseQRankSum", true, false),
    ("ExcessHet", true, false),
    ("DRAGstrInfo", true, false),
];

/// Logs the [`FEATURE_FIELDS`] a VCF header does not declare: a warning for core fields, a note
/// for the others. `skip_sb` leaves out FORMAT/SB, for the VCF source when INFO/SOR stands in for
/// it.
fn warn_undeclared_fields(
    header: &rust_htslib::bcf::header::HeaderView,
    path: &Path,
    skip_sb: bool,
) {
    let undeclared = |core: bool| -> Vec<String> {
        FEATURE_FIELDS
            .iter()
            .filter(|&&(field, _, is_core)| is_core == core && !(skip_sb && field == SB))
            .filter(|&&(field, is_info, _)| !has_field(header, field.as_bytes(), is_info))
            .map(|&(field, is_info, _)| field_name(field, is_info))
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
    /// The raw annotations of the VCF's own records (the gVCF feature source).
    RawFromVcf,
    /// The raw annotations of the gVCF at this path (the gVCF feature source).
    RawFromGvcf(&'a Path),
    /// Derived from the VCF's standard annotations (the VCF feature source; see
    /// [`crate::vcf_source`]).
    DerivedFromVcf,
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
    /// The VCF reader, and the ids its header declares.
    vcf: Option<(IndexedReader, DeclaredIds)>,
    gvcf: Option<IndexedReader>,
}

/// Returns the reader in `slot`, opening `path` into it first if needed.
fn reader<'a>(slot: &'a mut Option<IndexedReader>, path: &Path) -> Result<&'a mut IndexedReader> {
    if slot.is_none() {
        *slot = Some(open_indexed(path)?);
    }
    Ok(slot.as_mut().expect("the reader was just opened"))
}

/// The FORMAT field of the per-strand allele depths.
const SB: &str = "SB";

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
    if readers.vcf.is_none() {
        let reader = open_indexed(vcf_path)?;
        let declared = DeclaredIds::new(reader.header(), vcf_path)?;
        readers.vcf = Some((reader, declared));
    }
    let Some((vcf, declared)) = readers.vcf.as_mut() else { unreachable!("opened above") };
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
        declared
            .check(&record)
            .with_context(|| format!("failed to read {}", vcf_path.display()))?;
        // A fetch also returns records that start before the shard but overlap it.
        let pos = u64::try_from(record.pos()).unwrap_or(0);
        if pos < start || end.is_some_and(|end| pos >= end) {
            continue;
        }
        let alleles = record.alleles();
        let derive = matches!(source, AnnotationSource::DerivedFromVcf);
        let Some(mut raw) = raw_record(&mut fields, &record, &alleles, &chrom, derive)? else {
            skipped += 1;
            continue;
        };
        match source {
            AnnotationSource::RawFromVcf => match raw_annotation(&mut fields, &record, &alleles)? {
                Some(annotation) => raw.raw = annotation,
                None => missing_annotations += 1,
            },
            AnnotationSource::DerivedFromVcf => {
                // No INFO/MQ or INFO/DP, or a zero depth: mq is missing.
                if raw.raw.raw_mq.is_nan() || raw.raw.raw_dp.is_nan() || raw.raw.raw_dp <= 0.0 {
                    missing_annotations += 1;
                }
            }
            AnnotationSource::RawFromGvcf(_) => {}
        }
        records.push(raw);
    }
    if let AnnotationSource::RawFromGvcf(gvcf_path) = source {
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

/// Reads one record's raw values, or `None` if it is neither a SNP nor an indel. With `derive`
/// (the VCF source), the raw annotations are derived from the record (see
/// [`vcf_source::read_annotation`]); otherwise they are left missing.
fn raw_record(
    fields: &mut FieldReader,
    record: &Record,
    alleles: &[&[u8]],
    chrom: &Arc<str>,
    derive: bool,
) -> Result<Option<RawRecord>> {
    let (reference, alts) = alleles.split_first().map_or((&b""[..], &[][..]), |(r, a)| (*r, a));
    let Some(variant_type) = variant_type_of(reference, alts) else {
        return Ok(None);
    };

    let first_alt = first_alt_index(alts);
    let element = |values: Option<&[f64]>, index: usize| {
        values.and_then(|v| v.get(index).copied()).unwrap_or(f64::NAN)
    };
    let genotype = fields.genotype(record);
    let called = derive.then(|| vcf_source::called_alts(genotype));
    let (ad_ref, ad_alt, ad_alt_total, called_depth) = {
        let ad = fields.format(record, b"AD")?;
        // Only the VCF source needs it, from this one decoding of FORMAT/AD.
        let called_depth =
            called.zip(ad).and_then(|(called, ad)| vcf_source::called_depth(ad, called));
        let known_alt_depths = ad.map(|ad| ad.iter().skip(1).copied().filter(|d| !d.is_nan()));
        let ad_alt_total = match known_alt_depths {
            Some(depths) => depths.reduce(|a, b| a + b).unwrap_or(f64::NAN),
            None => f64::NAN,
        };
        let ad_alt = first_alt.map_or(f64::NAN, |index| element(ad, index + 1));
        (element(ad, 0), ad_alt, ad_alt_total, called_depth)
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
    let mut raw = RawRecord {
        chrom: Arc::clone(chrom),
        pos: u64::try_from(record.pos() + 1).context("negative record position")?,
        reference: reference.to_vec(),
        alts: alts.iter().map(|alt| alt.to_vec()).collect(),
        filter: filter_string(record),
        variant_type,
        genotype,
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
    };
    if let Some(called) = called {
        raw.raw = vcf_source::read_annotation(fields, record, &raw, called, called_depth)?;
    }
    Ok(Some(raw))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::testing::{
        CALL_FIELDS, DEFAULT_MODE_FIELDS, RAW_FIELDS, vcf_header, write_indexed_vcf,
    };

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
        extract_raw_records(vcf, AnnotationInputs::Gvcf(gvcf), contigs)
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

    /// htslib's indexed reader names an undeclared FILTER through a header that lacks it, which
    /// crashed the process; it is an error instead.
    #[rstest::rstest]
    #[case::alone("Undeclared")]
    #[case::with_a_declared_one("LowQual;Undeclared")]
    fn an_undeclared_filter_is_an_error(#[case] filter: &str) {
        let dir = TempDir::new().unwrap();
        let record =
            format!("chr1 100 . A C 50 {filter} RAW_MQandDP=54000,30;QUALapprox=60 GT 0/1");
        let fields = [CALL_FIELDS, RAW_FIELDS];
        let vcf = write_vcf(&dir, "in.vcf.gz", &fields, &["S1"], &[record.as_str()]);
        let error = format!("{:#}", extract(&vcf, None, &["chr1"]).unwrap_err());
        assert!(
            error.contains(
                "a FILTER of the record at chr1:100 is not declared with a ##FILTER line in the \
                 VCF header; add one for it"
            ),
            "{error}"
        );
        // Declared, it is read.
        let declared = "##FILTER=<ID=Undeclared,Description=\"Now declared\">\n";
        let vcf = write_vcf(
            &dir,
            "ok.vcf.gz",
            &[CALL_FIELDS, RAW_FIELDS, declared],
            &["S1"],
            &[record.as_str()],
        );
        assert_eq!(extract(&vcf, None, &["chr1"]).unwrap()[0].filter, filter);
    }

    /// A read by region skips records at POS 0, in a VCF (where they are reported) or a BCF
    /// (where they cannot be); the other records are read.
    #[test]
    fn records_at_pos_0_are_skipped() {
        let dir = TempDir::new().unwrap();
        let records = [
            "chr1 0 . N A 50 PASS RAW_MQandDP=54000,30;QUALapprox=60 GT 0/1",
            "chr1 100 . A C 50 PASS RAW_MQandDP=54000,30;QUALapprox=60 GT 0/1",
        ];
        let vcf = write_vcf(&dir, "in.vcf.gz", &[CALL_FIELDS, RAW_FIELDS], &["S1"], &records);
        let scan = crate::vcf::tabix_scan(&vcf, Some(&["chr1"][..])).unwrap();
        assert_eq!(scan.pos0, Some(vec!["chr1".to_string()]));
        let positions = |vcf: &Path| -> Vec<u64> {
            extract(vcf, None, &["chr1"]).unwrap().iter().map(|r| r.pos).collect()
        };
        assert_eq!(positions(&vcf), [100]);
        let bcf = dir.path().join("in.bcf");
        {
            use rust_htslib::bcf::{Format, Header, Read as _, Reader, Writer};
            let mut reader = Reader::from_path(&vcf).unwrap();
            let header = Header::from_template(reader.header());
            let mut writer = Writer::from_path(&bcf, &header, false, Format::Bcf).unwrap();
            for record in reader.records() {
                writer.write(&record.unwrap()).unwrap();
            }
        }
        rust_htslib::bcf::index::build(&bcf, None, 1, rust_htslib::bcf::index::Type::Csi(14))
            .unwrap();
        assert_eq!(crate::vcf::tabix_scan::<&str>(&bcf, None), None);
        assert_eq!(positions(&bcf), [100]);
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
        // The PLs stay allele 1's (`*`): 0/0, 0/* and */*, as models were trained on.
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
            AnnotationSource::RawFromGvcf(&gvcf),
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

    /// The fields of a VCF the VCF feature source reads.
    const VCF_SOURCE: [&str; 3] = [CALL_FIELDS, DEFAULT_MODE_FIELDS, ""];

    fn extract_vcf_source(vcf: &Path, contigs: &[&str]) -> Result<Vec<RawRecord>> {
        extract_raw_records(vcf, AnnotationInputs::Vcf, contigs)
    }

    #[test]
    fn the_vcf_source_derives_the_annotations_from_the_vcf() {
        let dir = TempDir::new().unwrap();
        let records = [
            // No SB: SOR is the fallback.
            "chr1 100 . A C 50 PASS DP=4;MQ=30;SOR=0.693 GT:AD:DP:PL 0/1:1,2:5:40,0,50",
            // A hom-ref call: VarDP is INFO/DP; every alternate counts, and SB is read.
            "chr1 200 . A C,G 20 PASS DP=9;MQ=60;SOR=3 GT:AD:SB:PL 0/0:9,0,0:2,3,4,5:0,30,60,30,60,60",
            // No MQ: raw_mq is missing, but the rest is derived.
            "chr1 300 . AT A 30 PASS DP=6 GT:AD:PL 1/1:0,6:90,12,0",
        ];
        let vcf = write_vcf(&dir, "in.vcf.gz", &VCF_SOURCE, &["S1"], &records);
        let records = extract_vcf_source(&vcf, &["chr1"]).unwrap();
        let raw: Vec<RawAnnotation> = records.iter().map(|r| r.raw).collect();
        let [het, hom_ref, deletion] = raw.as_slice() else { panic!("{raw:?}") };
        assert_eq!(
            [het.raw_mq, het.raw_dp, het.qual_approx, het.var_dp, het.n_alt],
            [3600.0, 4.0, 40.0, 3.0, 1.0]
        );
        assert_eq!(het.sor, f64::from(0.693f32));
        assert_nan(het.as_mq_ref, "as_mq_ref");
        assert_nan(het.as_mq_alt, "as_mq_alt");
        assert_eq!([hom_ref.raw_mq, hom_ref.var_dp, hom_ref.n_alt], [3600.0 * 9.0, 9.0, 2.0]);
        assert_eq!((records[1].sb_ref_fwd, records[1].sb_alt_rev), (2.0, 5.0));
        assert_nan(deletion.raw_mq, "raw_mq without MQ");
        assert_eq!([deletion.raw_dp, deletion.qual_approx, deletion.var_dp], [6.0, 90.0, 6.0]);

        // The derived features: mq is INFO/MQ, qd PL[0] over VarDP, and SB is preferred over SOR.
        let derived = crate::features::derive_features(&records, &[], None);
        assert_eq!(derived.mq[..2], [30.0, 60.0]);
        assert_nan(derived.mq[2], "mq without MQ");
        assert_eq!(derived.qd, [40.0 / 3.0, 0.0, 15.0]);
        assert_eq!(derived.sor[0], f64::from(0.693f32));
        let from_sb = crate::features::strand_odds_ratio(2.0, 3.0, 4.0, 5.0);
        assert_eq!(derived.sor[1], from_sb);
        assert_ne!(from_sb, 3.0);
        // The median INFO/DP of 4, 9 and 6 is 6.
        assert_eq!(derived.dp_norm, [4.0 / 6.0, 1.5, 1.0]);
        assert!(derived.as_alt_frac.iter().all(|value| value.is_nan()));
    }

    /// As ReblockGVCF drops the alternates a genotype does not call, VarDP sums the depths of the
    /// reference and the called alternates only, and n_alt counts the called ones.
    #[test]
    fn the_vcf_source_leaves_out_uncalled_alternates() {
        let dir = TempDir::new().unwrap();
        let records = [
            "chr1 100 . A C,G 50 PASS DP=21;MQ=60;SOR=0.7 GT:AD:DP:PL 0/1:10,8,3:21:40,0,50,60,70,80",
            "chr1 200 . A C,G,T 50 PASS DP=30;MQ=60;SOR=0.7 GT:AD:DP:PL 1/3:2,9,4,11:30:90,0,50,60,70,80,1,2,3,4",
        ];
        let vcf = write_vcf(&dir, "in.vcf.gz", &VCF_SOURCE, &["S1"], &records);
        let records = extract_vcf_source(&vcf, &["chr1"]).unwrap();
        let derived: Vec<[f64; 2]> = records.iter().map(|r| [r.raw.var_dp, r.raw.n_alt]).collect();
        assert_eq!(derived, [[18.0, 1.0], [22.0, 2.0]]);
    }

    /// A record without FORMAT/PL has no qd, but keeps the rest of its derived annotations.
    #[test]
    fn the_vcf_source_leaves_qd_missing_without_pl() {
        let dir = TempDir::new().unwrap();
        let records = ["chr1 100 . A C 50 PASS DP=4;MQ=30 GT:AD 0/1:1,2"];
        let vcf = write_vcf(&dir, "in.vcf.gz", &VCF_SOURCE, &["S1"], &records);
        let raw = extract_vcf_source(&vcf, &["chr1"]).unwrap()[0].raw;
        assert_nan(raw.qual_approx, "qual_approx without PL");
        assert_eq!([raw.raw_mq, raw.raw_dp, raw.var_dp], [3600.0, 4.0, 3.0]);
    }

    /// The VCF source ignores the raw annotations of a VCF that has them, and the gVCF source
    /// ignores INFO/SOR, taking `sor` from FORMAT/SB only.
    #[test]
    fn each_source_ignores_the_other_sources_annotations() {
        let dir = TempDir::new().unwrap();
        let records = [
            "chr1 100 . A C 50 PASS DP=4;MQ=30;SOR=3;RAW_MQandDP=999,9;QUALapprox=7;VarDP=7 GT:AD:PL 0/1:1,2:40,0,50",
        ];
        let fields = [CALL_FIELDS, DEFAULT_MODE_FIELDS, RAW_FIELDS];
        let vcf = write_vcf(&dir, "in.vcf.gz", &fields, &["S1"], &records);
        let vcf_source = extract_vcf_source(&vcf, &["chr1"]).unwrap();
        let raw = vcf_source[0].raw;
        assert_eq!([raw.raw_mq, raw.raw_dp, raw.qual_approx, raw.var_dp], [3600.0, 4.0, 40.0, 3.0]);
        // The record has no FORMAT/SB, so the VCF source's sor is INFO/SOR.
        assert_eq!(crate::features::derive_features(&vcf_source, &[], None).sor, [3.0]);
        // The gVCF source reads the raw annotations, and leaves sor missing.
        let gvcf_source = extract(&vcf, None, &["chr1"]).unwrap();
        let raw = gvcf_source[0].raw;
        assert_eq!([raw.raw_mq, raw.qual_approx], [999.0, 7.0]);
        assert_nan(raw.sor, "the gVCF source's sor fallback");
        let sor = crate::features::derive_features(&gvcf_source, &[], None).sor;
        assert_nan(sor[0], "the gVCF source's sor without FORMAT/SB");
    }

    #[test]
    fn the_vcf_source_needs_its_fields_declared() {
        let dir = TempDir::new().unwrap();
        let record = ["chr1 100 . A C 50 PASS DP=4 GT 0/1"];
        let error = |fields: &[&str], name: &str| {
            let vcf = write_vcf(&dir, name, fields, &["S1"], &record);
            format!("{:#}", extract_vcf_source(&vcf, &["chr1"]).unwrap_err())
        };
        // CALL_FIELDS declares FORMAT/SB, so SOR is not needed.
        let message = error(&[CALL_FIELDS, RAW_FIELDS], "no_mq.vcf.gz");
        assert!(
            message.contains(
                "no_mq.vcf.gz does not declare INFO/MQ, which the vcf feature source (of extract \
                 --feature-source vcf, and of models trained on its tables) derives"
            ),
            "{message}"
        );
        let bare = CALL_FIELDS
            .lines()
            .filter(|line| !["ID=SB,", "ID=PL,", "ID=DP,"].iter().any(|id| line.contains(id)))
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        let message = error(&[&bare], "bare.vcf.gz");
        assert!(
            message.contains(
                "does not declare INFO/MQ, INFO/DP, FORMAT/PL, FORMAT/SB or INFO/SOR, which"
            ),
            "{message}"
        );
        // INFO/SOR stands in for FORMAT/SB.
        let with_sor = format!(
            "{bare}{DEFAULT_MODE_FIELDS}##INFO=<ID=DP,Number=1,Type=Integer,Description=\"D\">\n\
             ##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"PL\">\n"
        );
        let with_mq = ["chr1 100 . A C 50 PASS DP=4;MQ=60 GT 0/1"];
        let vcf = write_vcf(&dir, "sor.vcf.gz", &[&with_sor], &["S1"], &with_mq);
        assert_eq!(extract_vcf_source(&vcf, &["chr1"]).unwrap().len(), 1);
    }

    #[test]
    fn the_vcf_source_needs_records_with_mq_and_dp() {
        let dir = TempDir::new().unwrap();
        // A zero INFO/DP counts as missing too: mq cannot be derived from it.
        let records = [
            "chr1 100 . A C 50 PASS DP=4 GT 0/1",
            "chr1 200 . A C 50 PASS MQ=60 GT 0/1",
            "chr1 300 . A C 50 PASS DP=0;MQ=60 GT 0/1",
        ];
        let vcf = write_vcf(&dir, "in.vcf.gz", &VCF_SOURCE, &["S1"], &records);
        let message = format!("{:#}", extract_vcf_source(&vcf, &["chr1"]).unwrap_err());
        assert!(
            message.ends_with(
                "declares INFO/MQ and INFO/DP, but none of its 3 records has both (with a \
                 positive INFO/DP), so the vcf feature source has no mapping quality or depth for \
                 any call"
            ),
            "{message}"
        );
        // With a zero depth, mq is missing though INFO/MQ is not.
        let records = [
            "chr1 100 . A C 50 PASS DP=0;MQ=60 GT 0/1",
            "chr1 200 . A C 50 PASS DP=4;MQ=60 GT 0/1",
        ];
        let vcf = write_vcf(&dir, "zero.vcf.gz", &VCF_SOURCE, &["S1"], &records);
        let records = extract_vcf_source(&vcf, &["chr1"]).unwrap();
        let mq = crate::features::derive_features(&records, &[], None).mq;
        assert_nan(mq[0], "mq at a zero depth");
        assert_eq!(mq[1], 60.0);
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
