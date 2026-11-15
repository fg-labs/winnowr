//! Write a sample's re-genotyped, filtered VCF from the models' decisions (see [`crate::policy`]).
//!
//! Genotype models score PASS calls of their variant type; promote models score non-PASS calls of
//! their variant type that do not overlap a PASS call. All other records are written unchanged.
//! For scored records:
//!
//! - INFO `WINNOWR_P` holds a genotype model's probabilities `p_drop,p_keep,p_flip`, or INFO
//!   `WINNOWR_PROMOTE_P` a promote model's `p_stay,p_promote,p_promote_flip`; re-genotype models
//!   also write INFO `WINNOWR_REGENOTYPE_P`.
//! - A dropped call gets FILTER `WinnowrDrop` (replacing PASS).
//! - A promoted call gets FILTER PASS; its original FILTER is kept in INFO `WINNOWR_OFILTER`.
//! - A flipped call has its GT switched between `0/1` and `1/1` (unphased), INFO flag
//!   `WINNOWR_FLIP`, FORMAT `WINNOWR_OGT` with the original genotype, and GQ set to the
//!   phred-scaled model confidence in the new genotype, `-10 log10(1 - p)` capped at 99. INFO
//!   AC/AF and MLEAC/MLEAF, when present, are recomputed from the new genotype, and the
//!   physical-phasing FORMAT fields PGT, PID and PS, when present, are set to missing. PL is left
//!   as the caller emitted it.
//! - A re-genotyped call gets the repeat-length caller's genotype the same way, with INFO flag
//!   `WINNOWR_REGENOTYPE`, adding an ALT allele when needed (see
//!   [`crate::regenotype::regenotyped_alleles`]); Number=A/R/G fields get missing values for new
//!   alleles.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context, Result, bail};
use log::{info, warn};
use rust_htslib::bcf::header::{HeaderView, TagLength, TagType};
use rust_htslib::bcf::index::{self, Type as IndexType};
use rust_htslib::bcf::record::{GenotypeAllele, Numeric, Record};
use rust_htslib::bcf::{Format, Header, IndexedReader, Read, Reader, Writer};

use crate::alleles::Genotype;
use crate::common::{PASS, Task, has_suffix, is_bgzf_path};
use crate::labels::Label;
use crate::policy::{CallKey, Decision};
use crate::shards::{ReaderPool, contig_shards};
use crate::vcf::{FieldReader, filter_string};

/// The FILTER of a dropped call.
pub const DROP_FILTER: &str = "WinnowrDrop";
/// The INFO field with a genotype model's probabilities.
pub const PROBABILITIES_INFO: &str = "WINNOWR_P";
/// The INFO field with a promote model's probabilities.
pub const PROMOTE_PROBABILITIES_INFO: &str = "WINNOWR_PROMOTE_P";
/// The INFO field with a promoted call's original FILTER.
pub const ORIGINAL_FILTER_INFO: &str = "WINNOWR_OFILTER";
/// The INFO flag of a flipped call.
pub const FLIP_INFO: &str = "WINNOWR_FLIP";
/// The INFO flag of a re-genotyped call.
pub const REGENOTYPE_INFO: &str = "WINNOWR_REGENOTYPE";
/// The INFO field with a re-genotype model's re-genotype probability.
pub const REGENOTYPE_PROBABILITY_INFO: &str = "WINNOWR_REGENOTYPE_P";
/// The FORMAT field with the genotype before a flip or re-genotype.
pub const ORIGINAL_GT_FORMAT: &str = "WINNOWR_OGT";
/// The highest GQ written.
pub const MAX_GQ: i32 = 99;

/// HaplotypeCaller's physical-phasing FORMAT fields, cleared when winnowr changes a genotype.
const PHASING_FORMATS: [&str; 3] = ["PGT", "PID", "PS"];

/// The INFO allele count and frequency pairs recomputed after a genotype change.
const ALLELE_COUNT_INFO: [(&str, &str); 2] = [("AC", "AF"), ("MLEAC", "MLEAF")];

/// The GQ of a changed genotype: phred-scaled `1 - p`, rounded half to even, capped at
/// [`MAX_GQ`], or `None` (missing) if `p` is NaN.
#[must_use]
pub fn flip_gq(p: f64) -> Option<i32> {
    if p.is_nan() {
        return None;
    }
    let error = 1.0 - p;
    if error <= 0.0 {
        return Some(MAX_GQ);
    }
    Some((-10.0 * error.log10()).round_ties_even().min(f64::from(MAX_GQ)) as i32)
}

/// The header lines declaring winnowr's FILTER, INFO and FORMAT fields.
const HEADER_LINES: [&str; 8] = [
    "##FILTER=<ID=WinnowrDrop,Description=\"winnowr: call is likely 0/0\">",
    "##INFO=<ID=WINNOWR_P,Number=3,Type=Float,Description=\"winnowr class probabilities: drop (0/0), keep, flip het<->hom\">",
    "##INFO=<ID=WINNOWR_PROMOTE_P,Number=3,Type=Float,Description=\"winnowr promote probabilities: stay filtered, promote, promote and flip\">",
    "##INFO=<ID=WINNOWR_OFILTER,Number=.,Type=String,Description=\"FILTER before winnowr promoted the call to PASS\">",
    "##INFO=<ID=WINNOWR_FLIP,Number=0,Type=Flag,Description=\"winnowr flipped het<->hom\">",
    "##FORMAT=<ID=WINNOWR_OGT,Number=1,Type=String,Description=\"Genotype before winnowr flipped or re-genotyped it\">",
    "##INFO=<ID=WINNOWR_REGENOTYPE_P,Number=1,Type=Float,Description=\"winnowr probability that the repeat-length genotype is right\">",
    "##INFO=<ID=WINNOWR_REGENOTYPE,Number=0,Type=Flag,Description=\"winnowr set the genotype to the repeat-length caller's\">",
];

/// Counts of the actions applied while writing.
#[derive(Debug, Default)]
struct Applied {
    drop: usize,
    promote: usize,
    flip: usize,
    skipped_flip: usize,
    regenotype: usize,
    duplicates: usize,
}

impl Applied {
    /// Adds another shard's counts.
    fn add(&mut self, other: &Applied) {
        self.drop += other.drop;
        self.promote += other.promote;
        self.flip += other.flip;
        self.skipped_flip += other.skipped_flip;
        self.regenotype += other.regenotype;
        self.duplicates += other.duplicates;
    }
}

/// Writes a copy of the VCF with the decisions applied (see the module docs).
///
/// A decision applies to the first record with its key; later records with the same key are
/// written unchanged, with a warning. A flip of a genotype other than `0/1` or `1/1` is skipped.
///
/// # Arguments
///
/// * `vcf` - the input VCF.
/// * `output` - the output: `.bcf` writes CSI-indexed BCF, `.gz` or `.bgz` bgzipped,
///   tabix-indexed VCF, anything else plain-text VCF (with a warning unless it ends in `.vcf`);
///   suffixes match in any case.
/// * `decisions` - the decisions by call (see [`crate::policy::decide_calls`]).
/// * `threads` - the extra htslib threads for decompressing the input and compressing the output.
///
/// # Errors
///
/// Returns an error if a file cannot be read or written, or a record cannot be updated.
pub fn write_scored_vcf(
    vcf: &Path,
    output: &Path,
    decisions: HashMap<CallKey, Decision>,
    threads: NonZeroUsize,
) -> Result<()> {
    if same_file(vcf, output) {
        bail!("the output {} is the input VCF; write to another path", output.display());
    }
    let (format, uncompressed, index_type) = if has_suffix(output, ".bcf") {
        (Format::Bcf, false, Some(IndexType::Csi(14)))
    } else if is_bgzf_path(output) {
        (Format::Vcf, false, Some(IndexType::Tbx))
    } else {
        if !has_suffix(output, ".vcf") {
            warn!(
                "{} does not end in .vcf, .vcf.gz, .vcf.bgz or .bcf: writing uncompressed VCF",
                output.display()
            );
        }
        (Format::Vcf, true, None)
    };
    let sharded = if format == Format::Vcf && !uncompressed { tabix_contigs(vcf) } else { None };
    let index = DecisionIndex::new(&decisions);
    let (applied, n_applied) = match sharded {
        Some(contigs) => write_sharded(vcf, output, &index, &contigs)?,
        None => write_serial(vcf, output, &index, threads, format, uncompressed)?,
    };
    let unmatched = decisions.len() - n_applied;
    if unmatched > 0 {
        warn!(
            "{unmatched} decisions matched no record of {} by CHROM, POS, REF, ALT and FILTER and \
             were not applied",
            vcf.display()
        );
    }
    if let Some(index_type) = index_type {
        index::build(output, None, 1, index_type)
            .map_err(|e| anyhow::anyhow!("failed to index {}: {e}", output.display()))?;
    }
    if applied.duplicates > 0 {
        warn!(
            "left {} records unchanged that repeat an earlier record's CHROM, POS, REF, ALT and \
             FILTER (only the first was scored)",
            applied.duplicates
        );
    }
    if applied.skipped_flip > 0 {
        warn!("skipped {} flips of calls not 0/1 or 1/1", applied.skipped_flip);
    }
    info!(
        "Wrote {}: {} calls dropped, {} promoted, {} flipped, {} re-genotyped",
        output.display(),
        applied.drop,
        applied.promote,
        applied.flip,
        applied.regenotype
    );
    Ok(())
}

/// The header of the scored VCF: the input's, with winnowr's INFO, FILTER and FORMAT lines.
fn scored_header(input: &HeaderView) -> Header {
    let mut header = Header::from_template(input);
    for line in HEADER_LINES {
        header.push_record(line.as_bytes());
    }
    header
}

/// The decisions by call key, and the positions of the decided calls by contig, so that records
/// without a decision (most records of a gVCF-derived VCF, or of contigs no model scores) are
/// passed over without building their key.
struct DecisionIndex<'a> {
    by_key: &'a HashMap<CallKey, Decision>,
    positions: HashMap<&'a str, HashSet<u64>>,
}

impl<'a> DecisionIndex<'a> {
    fn new(by_key: &'a HashMap<CallKey, Decision>) -> Self {
        let mut positions: HashMap<&str, HashSet<u64>> = HashMap::new();
        for key in by_key.keys() {
            positions.entry(key.0.as_str()).or_default().insert(key.1);
        }
        DecisionIndex { by_key, positions }
    }

    /// False if no decision has the record's contig and position (so none has its key). The
    /// contig is read as [`record_key`] reads it, without allocating for a UTF-8 name.
    fn may_have(&self, record: &Record) -> bool {
        let header = record.header();
        let name = record.rid().and_then(|rid| header.rid2name(rid).ok()).unwrap_or_default();
        let chrom = String::from_utf8_lossy(name);
        let pos = (record.pos() + 1) as u64;
        self.positions.get(chrom.as_ref()).is_some_and(|positions| positions.contains(&pos))
    }
}

/// Applies the decisions to one record (read with the input header): the first record with a
/// decision's key gets it, and later records with that key are counted as duplicates.
fn apply_decision(
    record: &mut Record,
    decisions: &DecisionIndex<'_>,
    applied_keys: &mut HashSet<CallKey>,
    tags: &AlleleTags,
    fields: &mut FieldReader,
    applied: &mut Applied,
) -> Result<()> {
    if !decisions.may_have(record) {
        return Ok(());
    }
    let key = record_key(record);
    if applied_keys.contains(&key) {
        applied.duplicates += 1;
    } else if let Some(decision) = decisions.by_key.get(&key) {
        apply(record, decision, tags, fields, applied)
            .with_context(|| format!("failed to update {}:{}", key.0, key.1))?;
        applied_keys.insert(key);
    }
    Ok(())
}

/// Writes the scored VCF one record at a time (BCF or uncompressed output, or an input without
/// a tabix index), returning the counts and the number of decisions applied.
fn write_serial(
    vcf: &Path,
    output: &Path,
    decisions: &DecisionIndex<'_>,
    threads: NonZeroUsize,
    format: Format,
    uncompressed: bool,
) -> Result<(Applied, usize)> {
    let mut reader =
        Reader::from_path(vcf).with_context(|| format!("failed to open {}", vcf.display()))?;
    reader.set_threads(threads.get()).context("failed to set the htslib threads")?;
    let header = scored_header(reader.header());
    let mut writer = Writer::from_path(output, &header, uncompressed, format)
        .with_context(|| format!("failed to create {}", output.display()))?;
    if !uncompressed {
        writer.set_threads(threads.get()).context("failed to set the htslib threads")?;
    }
    let tags = AlleleTags::from_header(writer.header(), vcf)?;
    let mut applied_keys: HashSet<CallKey> = HashSet::new();
    let mut applied = Applied::default();
    let mut fields = FieldReader::new();
    let mut record = reader.empty_record();
    while let Some(read) = reader.read(&mut record) {
        read.with_context(|| format!("failed to read {}", vcf.display()))?;
        writer.translate(&mut record);
        apply_decision(
            &mut record,
            decisions,
            &mut applied_keys,
            &tags,
            &mut fields,
            &mut applied,
        )?;
        writer.write(&record).with_context(|| format!("failed to write {}", output.display()))?;
    }
    Ok((applied, applied_keys.len()))
}

/// The span of the shards a scored VCF is written in, in parallel.
const WRITE_SHARD_SPAN: u64 = 10_000_000;

/// The contigs with records in a bgzipped VCF's tabix (or CSI) index, in file order, or `None` if
/// it is not a tabix-indexed VCF.
fn tabix_contigs(vcf: &Path) -> Option<Vec<String>> {
    use std::io::Read as _;
    let mut magic = [0u8; 3];
    let mut text = rust_htslib::bgzf::Reader::from_path(vcf).ok()?;
    text.read_exact(&mut magic).ok()?;
    if &magic == b"BCF" {
        return None;
    }
    rust_htslib::tbx::Reader::from_path(vcf).ok().map(|index| index.seqnames())
}

/// An input reader and the scored VCF header, reused across the shards a worker writes.
struct ShardReaders {
    reader: IndexedReader,
    header: Header,
}

/// Writes the scored VCF in shards of [`WRITE_SHARD_SPAN`] (by POS) in parallel on the rayon
/// pool, each to its own bgzipped VCF, then concatenates their BGZF blocks after the header's.
/// htslib writes (and flushes) a VCF header as its own blocks, so every shard file starts with
/// the same header blocks, which are dropped. Returns the counts and the number of decisions
/// applied.
fn write_sharded(
    vcf: &Path,
    output: &Path,
    decisions: &DecisionIndex<'_>,
    contigs: &[String],
) -> Result<(Applied, usize)> {
    use std::io::Write as _;

    use rayon::prelude::*;

    let summary = crate::vcf::VcfSummary::read(vcf)?;
    let dir = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let tmp = tempfile::Builder::new()
        .prefix(".winnowr-score-")
        .tempdir_in(dir)
        .with_context(|| format!("failed to create a temporary directory in {}", dir.display()))?;
    let header_file = tmp.path().join("header.vcf.gz");
    {
        let reader = crate::vcf::open_indexed(vcf)?;
        Writer::from_path(&header_file, &scored_header(reader.header()), false, Format::Vcf)
            .with_context(|| format!("failed to create {}", header_file.display()))?;
    }
    let header_blocks = std::fs::read(&header_file)
        .with_context(|| format!("failed to read {}", header_file.display()))?;
    let Some(header_blocks) = header_blocks.strip_suffix(&crate::io::BGZF_EOF) else {
        bail!("failed to write {}", header_file.display());
    };

    // A contig's last shard is open-ended, so a record past its declared length is still
    // written.
    let shards = contig_shards(&summary, contigs, WRITE_SHARD_SPAN);
    let pool: ReaderPool<ShardReaders> = ReaderPool::new();
    let written: Vec<(Option<std::path::PathBuf>, Applied, usize)> = shards
        .par_iter()
        .enumerate()
        .map(|(i, shard)| {
            let mut readers = match pool.take() {
                Some(readers) => readers,
                None => {
                    let reader = crate::vcf::open_indexed(vcf)?;
                    let header = scored_header(reader.header());
                    ShardReaders { reader, header }
                }
            };
            let path = tmp.path().join(format!("shard-{i:07}.vcf.gz"));
            let (contig, start, end) = (shard.contig, shard.start, shard.end);
            let result = write_shard(&mut readers, vcf, &path, decisions, contig, start, end);
            pool.put(readers);
            result.map(|(wrote, applied, n_applied)| (wrote.then_some(path), applied, n_applied))
        })
        .collect::<Result<_>>()?;

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(output)
            .with_context(|| format!("failed to create {}", output.display()))?,
    );
    out.write_all(header_blocks)
        .with_context(|| format!("failed to write {}", output.display()))?;
    let mut applied = Applied::default();
    let mut n_applied = 0;
    for (path, shard_applied, shard_n_applied) in written {
        if let Some(path) = path {
            crate::io::append_bgzf_blocks(&mut out, &path, header_blocks)
                .with_context(|| format!("failed to write {}", output.display()))?;
            std::fs::remove_file(&path).ok();
        }
        applied.add(&shard_applied);
        n_applied += shard_n_applied;
    }
    out.write_all(&crate::io::BGZF_EOF)
        .and_then(|()| out.flush())
        .with_context(|| format!("failed to write {}", output.display()))?;
    Ok((applied, n_applied))
}

/// Writes the scored records of one shard (records whose 0-based POS is in `[start, end)`) to a
/// bgzipped VCF at `path`, returning whether it had records, the counts, and the number of
/// decisions applied.
fn write_shard(
    readers: &mut ShardReaders,
    vcf: &Path,
    path: &Path,
    decisions: &DecisionIndex<'_>,
    contig: &str,
    start: u64,
    end: Option<u64>,
) -> Result<(bool, Applied, usize)> {
    let reader = &mut readers.reader;
    let rid = reader.header().name2rid(contig.as_bytes()).with_context(|| {
        format!("contig {contig} is indexed but not in the header of {}", vcf.display())
    })?;
    reader
        .fetch(rid, start, end.map(|end| end - 1))
        .with_context(|| format!("failed to read {contig} from {}", vcf.display()))?;
    let mut record = reader.empty_record();
    let mut writer: Option<(Writer, AlleleTags)> = None;
    let mut applied_keys: HashSet<CallKey> = HashSet::new();
    let mut applied = Applied::default();
    let mut fields = FieldReader::new();
    while let Some(read) = reader.read(&mut record) {
        read.with_context(|| format!("failed to read {}", vcf.display()))?;
        // A fetch also returns records that start before the shard but overlap it.
        let pos = u64::try_from(record.pos()).unwrap_or(0);
        if pos < start || end.is_some_and(|end| pos >= end) {
            continue;
        }
        if writer.is_none() {
            let created = Writer::from_path(path, &readers.header, false, Format::Vcf)
                .with_context(|| format!("failed to create {}", path.display()))?;
            let tags = AlleleTags::from_header(created.header(), vcf)?;
            writer = Some((created, tags));
        }
        let Some((writer, tags)) = writer.as_mut() else { unreachable!() };
        writer.translate(&mut record);
        apply_decision(&mut record, decisions, &mut applied_keys, tags, &mut fields, &mut applied)?;
        writer.write(&record).with_context(|| format!("failed to write {}", path.display()))?;
    }
    let wrote = writer.is_some();
    Ok((wrote, applied, applied_keys.len()))
}

/// True if two paths name the same existing file.
fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// A record's key, as the feature table holds it.
fn record_key(record: &Record) -> CallKey {
    let header = record.header();
    let chrom = record
        .rid()
        .and_then(|rid| header.rid2name(rid).ok())
        .map(|name| String::from_utf8_lossy(name).into_owned())
        .unwrap_or_default();
    let alleles = record.alleles();
    let text = |allele: &[u8]| String::from_utf8_lossy(allele).into_owned();
    let reference = alleles.first().map(|a| text(a)).unwrap_or_default();
    let alts: Vec<String> = alleles.iter().skip(1).map(|a| text(a)).collect();
    (chrom, (record.pos() + 1) as u64, reference, alts.join(","), filter_string(record))
}

/// Annotates one record in place with its decision.
fn apply(
    record: &mut Record,
    decision: &Decision,
    tags: &AlleleTags,
    fields: &mut FieldReader,
    applied: &mut Applied,
) -> Result<()> {
    let probabilities: Vec<f32> = decision.probabilities.iter().map(|&p| p as f32).collect();
    if decision.task == Task::Promote {
        record.push_info_float(PROMOTE_PROBABILITIES_INFO.as_bytes(), &probabilities)?;
    } else {
        record.push_info_float(PROBABILITIES_INFO.as_bytes(), &probabilities[..3])?;
        if let Some(&p) = probabilities.get(Label::Regenotype.index()) {
            record.push_info_float(REGENOTYPE_PROBABILITY_INFO.as_bytes(), &[p])?;
        }
    }
    if decision.drop {
        record.set_filters(&[DROP_FILTER.as_bytes()])?;
        applied.drop += 1;
        return Ok(());
    }
    if decision.promote {
        let header = record.header();
        let original: Vec<Vec<u8>> = record.filters().map(|id| header.id_to_name(id)).collect();
        let original: Vec<&[u8]> = original.iter().map(Vec::as_slice).collect();
        record.push_info_string(ORIGINAL_FILTER_INFO.as_bytes(), &original)?;
        record.set_filters(&[PASS.as_bytes()])?;
        applied.promote += 1;
    }
    let original = fields.genotype(record);
    if decision.flip {
        match original.flipped().and_then(Genotype::alleles) {
            Some((first, second)) => {
                let p_flip = decision.probabilities[2];
                set_genotype(
                    record,
                    (first as usize, second as usize),
                    original,
                    p_flip,
                    FLIP_INFO,
                    fields,
                )?;
                applied.flip += 1;
            }
            None => applied.skipped_flip += 1,
        }
    } else if let Some(target) = &decision.regenotype {
        if target.alleles.len() != record.allele_count() as usize
            || !record.alleles().iter().zip(&target.alleles).all(|(a, b)| *a == b.as_bytes())
        {
            set_alleles(record, &target.alleles, tags)?;
        }
        let p = decision.probabilities[Label::Regenotype.index()];
        set_genotype(record, target.genotype, original, p, REGENOTYPE_INFO, fields)?;
        applied.regenotype += 1;
    }
    Ok(())
}

/// Writes a new, unphased genotype with its model GQ, and keeps the record self-consistent.
fn set_genotype(
    record: &mut Record,
    genotype: (usize, usize),
    original: Genotype,
    p: f64,
    flag: &str,
    fields: &mut FieldReader,
) -> Result<()> {
    let header = record.header();
    let has_gq = header.format_type(b"GQ").is_ok();
    let mut phasing: Vec<(&str, TagType)> = Vec::new();
    for tag in PHASING_FORMATS {
        if let Ok((tag_type, _)) = header.format_type(tag.as_bytes())
            && read_format(record, tag, tag_type)?.is_some()
        {
            phasing.push((tag, tag_type));
        }
    }
    let alleles =
        [GenotypeAllele::Unphased(genotype.0 as i32), GenotypeAllele::Unphased(genotype.1 as i32)];
    record.push_genotypes(&alleles)?;
    let original = original.to_string();
    record.push_format_string(ORIGINAL_GT_FORMAT.as_bytes(), &[original.as_bytes()])?;
    if has_gq {
        record.push_format_integer(b"GQ", &[flip_gq(p).unwrap_or_else(i32::missing)])?;
    }
    for (tag, tag_type) in phasing {
        match tag_type {
            TagType::String => record.push_format_string(tag.as_bytes(), &[&b"."[..]])?,
            TagType::Integer => record.push_format_integer(tag.as_bytes(), &[i32::missing()])?,
            TagType::Float => record.push_format_float(tag.as_bytes(), &[f32::missing()])?,
            TagType::Flag => {}
        }
    }
    update_allele_counts(record, genotype, fields)?;
    record.push_info_flag(flag.as_bytes())?;
    Ok(())
}

/// Recomputes the record's INFO AC/AF and MLEAC/MLEAF (when present) from the new genotype.
fn update_allele_counts(
    record: &mut Record,
    genotype: (usize, usize),
    fields: &mut FieldReader,
) -> Result<()> {
    let n_alts = record.allele_count().saturating_sub(1) as usize;
    let counts: Vec<i32> = (1..=n_alts)
        .map(|index| i32::from(genotype.0 == index) + i32::from(genotype.1 == index))
        .collect();
    let frequencies: Vec<f32> = counts.iter().map(|&count| count as f32 / 2.0).collect();
    for (count_key, frequency_key) in ALLELE_COUNT_INFO {
        if fields.info(record, count_key.as_bytes())?.is_some() {
            record.push_info_integer(count_key.as_bytes(), &counts)?;
        }
        if fields.info(record, frequency_key.as_bytes())?.is_some() {
            record.push_info_float(frequency_key.as_bytes(), &frequencies)?;
        }
    }
    Ok(())
}

/// The value count of a Number=A/R/G field for `n_alleles` alleles (`None` for other Numbers).
fn per_allele_length(length: TagLength, n_alleles: usize) -> Option<usize> {
    match length {
        TagLength::AltAlleles => Some(n_alleles - 1),
        TagLength::Alleles => Some(n_alleles),
        TagLength::Genotypes => Some(n_alleles * (n_alleles + 1) / 2),
        TagLength::Fixed(_) | TagLength::Variable => None,
    }
}

/// A field's values, read so they can be padded and written back.
enum Values {
    Integers(Vec<i32>),
    Floats(Vec<f32>),
    Strings(Vec<u8>),
}

/// The header's per-allele fields, which change when alleles are added.
#[derive(Debug, Default)]
struct AlleleTags {
    /// INFO fields declared Number=A, R or G, with their types and lengths.
    infos: Vec<(String, TagType, TagLength)>,
    /// FORMAT fields (other than GT) declared Number=A, R or G, with their types and lengths.
    formats: Vec<(String, TagType, TagLength)>,
    /// Other string INFO fields, which may hold `|`-separated per-allele values (e.g. GATK's
    /// allele-specific `AS_RAW_MQ`, declared Number=1).
    strings: Vec<String>,
}

impl AlleleTags {
    /// The per-allele fields a header (of the VCF at `path`) declares.
    ///
    /// # Errors
    ///
    /// Returns an error if a header line is not UTF-8.
    fn from_header(header: &HeaderView, path: &Path) -> Result<Self> {
        use rust_htslib::bcf::header::HeaderRecord;
        let mut tags = AlleleTags::default();
        for record in crate::vcf::header_records(header, path)? {
            let (values, is_info) = match &record {
                HeaderRecord::Info { values, .. } => (values, true),
                HeaderRecord::Format { values, .. } => (values, false),
                _ => continue,
            };
            let Some(key) = values.get("ID").filter(|key| *key != "GT").cloned() else { continue };
            let declared = if is_info {
                header.info_type(key.as_bytes())
            } else {
                header.format_type(key.as_bytes())
            };
            let Ok((tag_type, length)) = declared else { continue };
            match (is_info, per_allele_length(length, 2).is_some()) {
                (true, true) => tags.infos.push((key, tag_type, length)),
                (false, true) => tags.formats.push((key, tag_type, length)),
                (true, false) if tag_type == TagType::String => tags.strings.push(key),
                _ => {}
            }
        }
        Ok(tags)
    }
}

/// Replaces the record's alleles with an extended set, padding per-allele fields.
///
/// The existing alleles keep their order and new ones are appended, so existing per-allele and
/// per-genotype values of Number=A/R/G fields keep their positions and the new entries are
/// missing. Allele-specific string annotations with one `|`-separated value per allele or ALT
/// (such as GATK's `AS_RAW_MQ`) cannot be filled in for the new alleles, so they are removed.
fn set_alleles(record: &mut Record, alleles: &[String], tags: &AlleleTags) -> Result<()> {
    let old_alleles = record.allele_count() as usize;
    let mut info_values = Vec::new();
    for (tag, tag_type, length) in &tags.infos {
        if let Some(values) = read_info(record, tag, *tag_type)? {
            info_values.push((tag, *length, values));
        }
    }
    let mut format_values = Vec::new();
    for (tag, tag_type, length) in &tags.formats {
        if let Some(values) = read_format(record, tag, *tag_type)? {
            format_values.push((tag, *length, values));
        }
    }
    let mut stale = Vec::new();
    for tag in &tags.strings {
        if let Some(Values::Strings(value)) = read_info(record, tag, TagType::String)? {
            let parts = value.split(|&b| b == b'|').count();
            if value.contains(&b'|') && (parts == old_alleles || parts + 1 == old_alleles) {
                stale.push(tag);
            }
        }
    }
    let allele_bytes: Vec<&[u8]> = alleles.iter().map(String::as_bytes).collect();
    record.set_alleles(&allele_bytes)?;
    for (tag, length, values) in info_values {
        let Some(size) = per_allele_length(length, alleles.len()) else { continue };
        match pad(values, size) {
            Values::Integers(v) => record.push_info_integer(tag.as_bytes(), &v)?,
            Values::Floats(v) => record.push_info_float(tag.as_bytes(), &v)?,
            Values::Strings(v) => record.push_info_string(tag.as_bytes(), &[v.as_slice()])?,
        }
    }
    for (tag, length, values) in format_values {
        let Some(size) = per_allele_length(length, alleles.len()) else { continue };
        match pad(values, size) {
            Values::Integers(v) => record.push_format_integer(tag.as_bytes(), &v)?,
            Values::Floats(v) => record.push_format_float(tag.as_bytes(), &v)?,
            Values::Strings(v) => record.push_format_string(tag.as_bytes(), &[v.as_slice()])?,
        }
    }
    for tag in stale {
        record.clear_info_string(tag.as_bytes())?;
    }
    Ok(())
}

/// An INFO field's raw values, or `None` if the record does not have it.
fn read_info(record: &Record, tag: &str, tag_type: TagType) -> Result<Option<Values>> {
    let info = record.info(tag.as_bytes());
    Ok(match tag_type {
        TagType::Integer => info.integer()?.map(|v| Values::Integers(v.to_vec())),
        TagType::Float => info.float()?.map(|v| Values::Floats(v.to_vec())),
        TagType::String => info.string()?.map(|v| Values::Strings(v.join(&b','))),
        TagType::Flag => None,
    })
}

/// A FORMAT field's raw values for the first sample, or `None` if the record does not have it.
fn read_format(record: &Record, tag: &str, tag_type: TagType) -> Result<Option<Values>> {
    use rust_htslib::errors::Error as HtslibError;
    let format = record.format(tag.as_bytes());
    let values = match tag_type {
        TagType::Integer => format
            .integer()
            .map(|v| Values::Integers(v.first().map(|s| s.to_vec()).unwrap_or_default())),
        TagType::Float => format
            .float()
            .map(|v| Values::Floats(v.first().map(|s| s.to_vec()).unwrap_or_default())),
        TagType::String => format
            .string()
            .map(|v| Values::Strings(v.first().map(|s| s.to_vec()).unwrap_or_default())),
        TagType::Flag => return Ok(None),
    };
    match values {
        Ok(values) => Ok(Some(values)),
        Err(HtslibError::BcfMissingTag { .. }) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Pads values with missing ones up to `size`.
fn pad(values: Values, size: usize) -> Values {
    match values {
        Values::Integers(mut v) => {
            v.resize(size.max(v.len()), i32::missing());
            Values::Integers(v)
        }
        Values::Floats(mut v) => {
            v.resize(size.max(v.len()), f32::missing());
            Values::Floats(v)
        }
        Values::Strings(mut v) => {
            let present = v.split(|&b| b == b',').count();
            for _ in present..size {
                v.extend_from_slice(b",.");
            }
            Values::Strings(v)
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const THREADS: NonZeroUsize = NonZeroUsize::new(2).expect("2 is non-zero");

    #[rstest]
    #[case(0.5, 3)]
    #[case(0.99, 20)]
    #[case(0.9999999999, 99)]
    #[case(1.0, 99)]
    #[case(0.0, 0)]
    #[case::half_even(1.0 - 10f64.powf(-2.5), 25)]
    fn gq_values(#[case] p: f64, #[case] expected: i32) {
        assert_eq!(flip_gq(p), Some(expected));
        assert_eq!(flip_gq(f64::NAN), None);
    }

    use rust_htslib::bcf::Reader as VcfReader;
    use tempfile::TempDir;

    use crate::regenotype::RegenotypeTarget;
    use crate::testing::{vcf_header, write_indexed_vcf};

    const FIELDS: &str = "\
##FILTER=<ID=PASS,Description=\"All filters passed\">
##FILTER=<ID=LowQual,Description=\"Low quality\">
##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count\">
##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">
##INFO=<ID=AS_RAW_MQ,Number=1,Type=String,Description=\"Allele-specific raw MQ\">
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality\">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allele depths\">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred likelihoods\">
##FORMAT=<ID=PGT,Number=1,Type=String,Description=\"Phasing haplotype\">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description=\"Phase set\">
";

    fn key(pos: u64, reference: &str, alt: &str, filter: &str) -> CallKey {
        ("chr1".to_string(), pos, reference.to_string(), alt.to_string(), filter.to_string())
    }

    fn genotype_decision(probabilities: [f64; 4]) -> Decision {
        Decision {
            task: Task::Genotype,
            probabilities: probabilities.to_vec(),
            drop: false,
            promote: false,
            flip: false,
            regenotype: None,
        }
    }

    /// Writes the input VCF and returns its path.
    fn input(dir: &TempDir) -> std::path::PathBuf {
        let lines = [
            "chr1\t100\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD:PL:PGT:PS\t0|1:30:5,5:30,0,40:0|1:100",
            "chr1\t200\t.\tG\tT\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t300\t.\tC\tA\t20\tLowQual\tAC=2;AF=1\tGT:GQ:AD\t1/1:10:0,4",
            "chr1\t400\t.\tGA\tG\t50\tPASS\tAC=1;AF=0.5;AS_RAW_MQ=100|200\tGT:GQ:AD:PL\t0/1:20:6,4:20,0,30",
            "chr1\t400\t.\tGA\tG\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:20:6,4",
        ];
        write_indexed_vcf(dir.path(), "in.vcf.gz", &vcf_header(&[FIELDS], &["S1"]), &lines)
    }

    #[test]
    fn sharded_and_serial_writes_give_the_same_vcf() {
        // chr2 precedes chr1 in the file (not header order); a chr1 record straddles the first
        // shard boundary; a chr2 record lies past its declared length; chr3 has no length.
        let header = vcf_header(&[FIELDS], &["S1"]).replace(
            "##contig=<ID=chr1,length=100000>\n##contig=<ID=chr2,length=100000>\n",
            "##contig=<ID=chr1,length=25000000>\n##contig=<ID=chr2,length=100>\n##contig=<ID=chr3>\n",
        );
        let lines = [
            "chr2\t150\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t100\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t9999998\t.\tACGTA\tA\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t10000001\t.\tG\tT\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t20000001\t.\tC\tA\t20\tLowQual\tAC=2;AF=1\tGT:GQ:AD\t1/1:10:0,4",
            "chr3\t5\t.\tT\tG\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
        ];
        let dir = TempDir::new().unwrap();
        let vcf = write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &lines);
        assert_eq!(tabix_contigs(&vcf).unwrap(), ["chr2", "chr1", "chr3"]);
        let decisions = || {
            let mut decisions = HashMap::new();
            let on = |chrom: &str, k: CallKey| (chrom.to_string(), k.1, k.2, k.3, k.4);
            let flip = Decision { flip: true, ..genotype_decision([0.05, 0.15, 0.8, 0.0]) };
            decisions.insert(on("chr2", key(150, "A", "C", PASS)), flip);
            let drop = Decision { drop: true, ..genotype_decision([0.9, 0.05, 0.05, 0.0]) };
            decisions.insert(key(9_999_998, "ACGTA", "A", PASS), drop.clone());
            decisions.insert(key(10_000_001, "G", "T", PASS), drop.clone());
            decisions.insert(on("chr3", key(5, "T", "G", PASS)), drop);
            decisions.insert(key(999, "A", "T", PASS), genotype_decision([0.1, 0.8, 0.1, 0.0]));
            decisions
        };
        let serial = dir.path().join("serial.vcf");
        write_scored_vcf(&vcf, &serial, decisions(), THREADS).unwrap();
        let sharded = dir.path().join("sharded.vcf.gz");
        write_scored_vcf(&vcf, &sharded, decisions(), THREADS).unwrap();

        let mut text = String::new();
        std::io::Read::read_to_string(
            &mut flate2::read::MultiGzDecoder::new(std::fs::File::open(&sharded).unwrap()),
            &mut text,
        )
        .unwrap();
        assert_eq!(text, std::fs::read_to_string(&serial).unwrap());
        assert!(!crate::io::lacks_bgzf_eof(&sharded));
        assert_eq!(text.lines().filter(|l| !l.starts_with('#')).count(), lines.len());
        assert_eq!(text.matches(DROP_FILTER).count(), 4, "3 drops plus the header line");
        // The shard files were removed with their directory.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with(".winnowr-score-"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn writes_decisions_into_the_vcf() {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        let mut decisions = HashMap::new();
        let flip = Decision { flip: true, ..genotype_decision([0.05, 0.15, 0.8, 0.0]) };
        decisions.insert(key(100, "A", "C", PASS), flip);
        let drop = Decision { drop: true, ..genotype_decision([0.9, 0.05, 0.05, 0.0]) };
        decisions.insert(key(200, "G", "T", PASS), drop);
        let promote = Decision {
            task: Task::Promote,
            probabilities: vec![0.1, 0.8, 0.1],
            promote: true,
            ..genotype_decision([0.0; 4])
        };
        decisions.insert(key(300, "C", "A", "LowQual"), promote);
        let regenotype = Decision {
            regenotype: Some(RegenotypeTarget {
                alleles: vec!["GAA".to_string(), "GA".to_string(), "GAAA".to_string()],
                genotype: (1, 2),
            }),
            ..genotype_decision([0.1, 0.1, 0.1, 0.7])
        };
        decisions.insert(key(400, "GA", "G", PASS), regenotype);
        decisions.insert(key(999, "A", "T", PASS), genotype_decision([0.1, 0.8, 0.1, 0.0]));
        let output = dir.path().join("out.vcf.gz");
        write_scored_vcf(&vcf, &output, decisions, THREADS).unwrap();
        assert!(dir.path().join("out.vcf.gz.tbi").exists());

        let mut reader = VcfReader::from_path(&output).unwrap();
        let records: Vec<Record> = reader.records().map(Result::unwrap).collect();
        assert_eq!(records.len(), 5);
        let filters = |r: &Record| filter_string(r);
        let info_f32 = |r: &Record, tag: &[u8]| r.info(tag).float().unwrap().map(|v| v.to_vec());
        let info_i32 = |r: &Record, tag: &[u8]| r.info(tag).integer().unwrap().map(|v| v.to_vec());
        let fmt_i32 = |r: &Record, tag: &[u8]| r.format(tag).integer().unwrap()[0].to_vec();
        let fmt_str = |r: &Record, tag: &[u8]| {
            String::from_utf8(r.format(tag).string().unwrap()[0].to_vec()).unwrap()
        };
        let mut fields = FieldReader::new();

        // The flip: 0|1 -> unphased 1/1, the original GT kept, GQ from P(flip), phasing cleared,
        // AC/AF recomputed, PL unchanged.
        let flipped = &records[0];
        assert_eq!(fields.genotype(flipped), Genotype::HOM_ALT);
        assert!(
            flipped
                .genotypes()
                .unwrap()
                .get(0)
                .iter()
                .all(|a| matches!(a, GenotypeAllele::Unphased(_)))
        );
        assert_eq!(fmt_str(flipped, b"WINNOWR_OGT"), "0/1");
        assert_eq!(fmt_i32(flipped, b"GQ"), [7]);
        assert_eq!(fmt_str(flipped, b"PGT"), ".");
        assert!(fmt_i32(flipped, b"PS")[0].is_missing());
        assert_eq!(info_i32(flipped, b"AC"), Some(vec![2]));
        assert_eq!(info_f32(flipped, b"AF"), Some(vec![1.0]));
        assert_eq!(fmt_i32(flipped, b"PL"), [30, 0, 40]);
        assert!(flipped.info(b"WINNOWR_FLIP").flag().unwrap());
        assert_eq!(info_f32(flipped, b"WINNOWR_P").unwrap().len(), 3);
        assert_eq!(info_f32(flipped, b"WINNOWR_REGENOTYPE_P"), Some(vec![0.0]));

        // The drop.
        assert_eq!(filters(&records[1]), DROP_FILTER);
        assert_eq!(fields.genotype(&records[1]), Genotype::HET);

        // The promotion keeps its original FILTER in INFO.
        let promoted = &records[2];
        assert_eq!(filters(promoted), PASS);
        let original = promoted.info(b"WINNOWR_OFILTER").string().unwrap().unwrap()[0].to_vec();
        assert_eq!(original, b"LowQual");
        assert_eq!(info_f32(promoted, b"WINNOWR_PROMOTE_P").unwrap().len(), 3);

        // The re-genotype adds an ALT: AD and PL are padded, AC/AF follow the new alleles, and
        // the stale allele-specific annotation is removed.
        let regenotyped = &records[3];
        assert_eq!(regenotyped.alleles(), [&b"GAA"[..], b"GA", b"GAAA"]);
        assert_eq!(fields.genotype(regenotyped), Genotype::diploid(1, 2));
        let ad = fmt_i32(regenotyped, b"AD");
        assert_eq!(ad[..2], [6, 4]);
        assert!(ad[2].is_missing());
        let pl = fmt_i32(regenotyped, b"PL");
        assert_eq!(pl.len(), 6);
        assert!(pl[3..].iter().all(|v| v.is_missing()));
        assert_eq!(info_i32(regenotyped, b"AC"), Some(vec![1, 1]));
        assert!(regenotyped.info(b"AS_RAW_MQ").string().unwrap().is_none());
        assert!(regenotyped.info(b"WINNOWR_REGENOTYPE").flag().unwrap());
        assert_eq!(fmt_i32(regenotyped, b"GQ"), [5]);

        // The duplicate of the re-genotyped record is left unchanged.
        let duplicate = &records[4];
        assert_eq!(duplicate.alleles(), [&b"GA"[..], b"G"]);
        assert!(duplicate.info(b"WINNOWR_P").float().unwrap().is_none());
    }

    #[test]
    fn refuses_to_overwrite_the_input() {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        let message =
            write_scored_vcf(&vcf, &vcf, HashMap::new(), THREADS).unwrap_err().to_string();
        assert!(message.ends_with("is the input VCF; write to another path"), "{message}");
        // Plain-text and BCF outputs are written too.
        let plain = dir.path().join("out.vcf");
        write_scored_vcf(&vcf, &plain, HashMap::new(), THREADS).unwrap();
        assert!(std::fs::read_to_string(&plain).unwrap().contains("##FILTER=<ID=WinnowrDrop"));
        let bcf = dir.path().join("out.bcf");
        write_scored_vcf(&vcf, &bcf, HashMap::new(), THREADS).unwrap();
        assert!(dir.path().join("out.bcf.csi").exists());
    }
}
