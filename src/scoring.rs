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
use std::path::{Path, PathBuf};

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
use crate::vcf::{DeclaredIds, FieldReader, filter_string};

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
    check_output_not_input(output, [("VCF", vcf)])?;
    let indexes = crate::vcf::index_candidates(vcf);
    check_output_not_indexes(output, "VCF index", &indexes)?;
    if let Some(output_index) = scored_vcf_index(output) {
        check_output_not_input(&output_index, [("VCF", vcf)])?;
        check_output_not_indexes(&output_index, "VCF index", &indexes)?;
        check_output_not_input(&output_index, [("scored VCF", output)])?;
    }
    let (format, uncompressed, index_type) = output_format(output);
    if uncompressed && !has_suffix(output, ".vcf") {
        warn!(
            "{} does not end in .vcf, .vcf.gz, .vcf.bgz or .bcf: writing uncompressed VCF",
            output.display()
        );
    }
    // Shards are read by region, which skips records at POS 0: inputs that have them, or may
    // have them, are written in one pass.
    let scan = if format == Format::Vcf && !uncompressed {
        crate::vcf::tabix_scan::<&str>(vcf, None)
    } else {
        None
    };
    let sharded = scan.and_then(|scan| match scan.pos0 {
        Some(found) if found.is_empty() => Some(scan.contigs),
        Some(found) => {
            info!(
                "{} has records at POS 0 (on {}), which a read by region skips, so it is written \
                 in one pass",
                vcf.display(),
                found.join(", ")
            );
            None
        }
        None => {
            warn!(
                "could not check {} for records at POS 0 (its tabix index could not be read), so \
                 it is written in one pass rather than in parallel",
                vcf.display()
            );
            None
        }
    });
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
        // A dangling sidecar symlink may resolve only after the output has been written.
        if let Some(output_index) = scored_vcf_index(output) {
            check_output_not_input(&output_index, [("scored VCF", output)])?;
        }
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
    let declared = DeclaredIds::new(reader.header(), vcf)?;
    let mut record = reader.empty_record();
    while let Some(read) = reader.read(&mut record) {
        read.with_context(|| format!("failed to read {}", vcf.display()))?;
        declared.check(&record).with_context(|| format!("failed to read {}", vcf.display()))?;
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

/// An input reader and the scored VCF header, reused across the shards a worker writes.
struct ShardReaders {
    reader: IndexedReader,
    header: Header,
    declared: DeclaredIds,
    /// A writer to nowhere with the input's header, that checks a record can be formatted, and
    /// whose header also declares the names for [`DeclaredIds::check_tags`].
    check: Writer,
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
                    let declared = DeclaredIds::new(reader.header(), vcf)?;
                    let mut input = Header::from_template(reader.header());
                    for line in declared.tag_probe_lines() {
                        input.push_record(line.as_bytes());
                    }
                    let check = Writer::from_path("/dev/null", &input, true, Format::Vcf)
                        .context("failed to open /dev/null")?;
                    ShardReaders { reader, header, declared, check }
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
        readers
            .declared
            .check(&record)
            .with_context(|| format!("failed to read {}", vcf.display()))?;
        // A record read by region is not flagged for an undeclared INFO or FORMAT tag, whose id
        // lies past the header's dictionary; translating it would give it the id of one of
        // winnowr's header lines. Formatting it with its own header rejects such an id. It is
        // written (to nowhere) rather than formatted to a string, which must be UTF-8.
        readers.check.write(&record).with_context(|| {
            format!(
                "the record at {} has an INFO or FORMAT tag not declared in the VCF header; \
                 declare it",
                crate::vcf::position(&record)
            )
        })?;
        // Nor for an INFO or FORMAT tag named like a field declared only as another kind, which
        // has an id, and would be written with no header line of its kind.
        readers.check.translate(&mut record);
        readers
            .declared
            .check_tags(&record)
            .with_context(|| format!("failed to read {}", vcf.display()))?;
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

/// How [`write_scored_vcf`] writes `output`, by its suffix: its format, whether it is
/// uncompressed, and the index built beside it (CSI for BCF, tabix for bgzipped VCF, none for
/// plain-text VCF).
fn output_format(output: &Path) -> (Format, bool, Option<IndexType>) {
    if has_suffix(output, ".bcf") {
        (Format::Bcf, false, Some(IndexType::Csi(14)))
    } else if is_bgzf_path(output) {
        (Format::Vcf, false, Some(IndexType::Tbx))
    } else {
        (Format::Vcf, true, None)
    }
}

/// The index [`write_scored_vcf`] writes beside `output`, named as htslib names it: `<output>.csi`
/// for BCF, `<output>.tbi` for bgzipped VCF, or `None` for plain-text VCF, which is not indexed.
#[must_use]
pub fn scored_vcf_index(output: &Path) -> Option<PathBuf> {
    let extension = match output_format(output).2? {
        IndexType::Csi(_) => "csi",
        IndexType::Tbx => "tbi",
    };
    let mut index = output.as_os_str().to_owned();
    index.push(".");
    index.push(extension);
    Some(PathBuf::from(index))
}

/// Checks that `output` is none of the `inputs`, each with what it is (e.g. `VCF`): the same
/// file, through a link too (see [`same_file`]), which writing would destroy.
///
/// # Errors
///
/// Returns an error naming the input the output is.
pub fn check_output_not_input<'a>(
    output: &Path,
    inputs: impl IntoIterator<Item = (&'a str, &'a Path)>,
) -> Result<()> {
    for (what, input) in inputs {
        if same_file(input, output) {
            bail!("the output {} is the input {what}; write to another path", output.display());
        }
    }
    Ok(())
}

/// Checks an output against an input's identity captured from the handle that read it, even
/// if the input's original path no longer names that file.
///
/// # Errors
///
/// Returns an error if the output names the captured input (including through a link).
pub fn check_output_not_loaded_input(
    output: &Path,
    what: &str,
    input: &std::fs::Metadata,
) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    if let Ok(actual) = std::fs::metadata(output)
        && (actual.dev(), actual.ino()) == (input.dev(), input.ino())
    {
        bail!("the output {} is the input {what}; write to another path", output.display());
    }
    Ok(())
}

/// Protects existing index files and absent candidates that precede the index currently selected
/// by the reader. Creating a later candidate does not change selection and remains allowed.
///
/// # Errors
///
/// Returns an error if writing `output` would overwrite or mask an input index.
pub fn check_output_not_indexes(output: &Path, what: &str, candidates: &[PathBuf]) -> Result<()> {
    check_output_not_input(output, candidates.iter().map(|path| (what, path.as_path())))?;
    let Some(selected) = candidates.iter().position(|path| path.exists()) else {
        return Ok(());
    };
    let Some(destination) = future_destination(output) else {
        return Ok(());
    };
    for candidate in &candidates[..selected] {
        if let Some(candidate) = future_destination(candidate)
            && same_future_destination(&destination, &candidate)?
        {
            bail!(
                "the output {} would mask the input {what}; write to another path",
                output.display()
            );
        }
    }
    Ok(())
}

/// Compares absent leaf names using their filesystem's case and normalization rules. Ambiguous
/// names are probed inside a unique temporary directory on the destination filesystem, never
/// at either requested path. Ordinary distinct ASCII names need no probe.
fn same_future_destination(a: &Path, b: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;
    if a == b {
        return Ok(true);
    }
    if a.exists() || b.exists() {
        return Ok(same_file(a, b));
    }
    let (Some(parent), Some(other_parent), Some(name), Some(other_name)) =
        (a.parent(), b.parent(), a.file_name(), b.file_name())
    else {
        return Ok(false);
    };
    if !same_file(parent, other_parent) {
        return Ok(false);
    }
    if name == other_name {
        return Ok(true);
    }
    let (bytes, other_bytes) = (name.as_encoded_bytes(), other_name.as_encoded_bytes());
    if bytes.is_ascii() && other_bytes.is_ascii() && !bytes.eq_ignore_ascii_case(other_bytes) {
        return Ok(false);
    }
    let probe =
        tempfile::Builder::new().prefix(".winnowr-path-check-").tempdir_in(parent).with_context(
            || format!("failed to check output filename equivalence in {}", parent.display()),
        )?;
    let file = std::fs::File::create_new(probe.path().join(name))?;
    let ours = file.metadata()?;
    let equal = match std::fs::metadata(probe.path().join(other_name)) {
        Ok(other) => ours.dev() == other.dev() && ours.ino() == other.ino(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error).context("failed to compare output filename aliases"),
    };
    drop(file);
    probe.close().context("failed to remove output filename comparison probe")?;
    Ok(equal)
}

/// Resolves an output's physical parent and any final symlink, even when the final target does
/// not exist. Missing parents and symlink loops cannot be written and have no prospective file.
fn future_destination(path: &Path) -> Option<PathBuf> {
    let mut path = path.to_path_buf();
    for _ in 0..40 {
        match std::fs::read_link(&path) {
            Ok(target) => {
                path = if target.is_absolute() { target } else { path.parent()?.join(target) };
            }
            Err(_) => {
                let parent = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                return Some(std::fs::canonicalize(parent).ok()?.join(path.file_name()?));
            }
        }
    }
    None
}

/// True if two paths name the same existing file (the same device and inode, so through a
/// symbolic or hard link too).
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(a), Ok(b)) => (a.dev(), a.ino()) == (b.dev(), b.ino()),
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
        assert_eq!(
            crate::vcf::tabix_scan::<&str>(&vcf, None).unwrap().contigs,
            ["chr2", "chr1", "chr3"]
        );
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
        assert!(!crate::io::lacks_bgzf_eof(&sharded).unwrap());
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

    /// Whether `write_scored_vcf` writes `vcf` to `name` in parallel shards.
    fn writes_sharded(vcf: &Path, name: &str) -> bool {
        is_bgzf_path(Path::new(name))
            && crate::vcf::tabix_scan::<&str>(vcf, None)
                .is_some_and(|scan| scan.pos0 == Some(vec![]))
    }

    #[rstest]
    #[case::sharded("out.vcf.gz", true)]
    #[case::serial("out.bcf", false)]
    fn an_undeclared_filter_is_an_error(#[case] name: &str, #[case] sharded: bool) {
        let dir = TempDir::new().unwrap();
        let lines = [
            "chr1\t100\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t200\t.\tG\tT\t50\tLowQual;Undeclared\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
        ];
        let vcf =
            write_indexed_vcf(dir.path(), "in.vcf.gz", &vcf_header(&[FIELDS], &["S1"]), &lines);
        assert_eq!(writes_sharded(&vcf, name), sharded);
        let output = dir.path().join(name);
        let error = write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap_err();
        assert!(
            format!("{error:#}").contains(
                "a FILTER of the record at chr1:200 is not declared with a ##FILTER line in the \
                 VCF header"
            ),
            "{error:#}"
        );
    }

    #[rstest]
    #[case::sharded("out.vcf.gz")]
    #[case::serial("out.bcf")]
    fn a_filter_named_like_a_declared_info_field_is_written(#[case] name: &str) {
        // htslib gives it the INFO field's id (FILTER, INFO and FORMAT share one dictionary).
        let dir = TempDir::new().unwrap();
        let lines = ["chr1\t100\t.\tA\tC\t50\tAC\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5"];
        let vcf =
            write_indexed_vcf(dir.path(), "in.vcf.gz", &vcf_header(&[FIELDS], &["S1"]), &lines);
        let output = dir.path().join(name);
        write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap();
        let filters: Vec<String> = VcfReader::from_path(&output)
            .unwrap()
            .records()
            .map(|r| filter_string(&r.unwrap()))
            .collect();
        assert_eq!(filters, ["AC"]);
    }

    /// An undeclared INFO tag made htslib exit the process when the one-pass writer translated
    /// the record; it is an error instead (the sharded writer already failed to write it).
    #[rstest]
    #[case::info("UNDECLARED=1;AC=1", "GT")]
    #[case::format("AC=1", "GT:UNDECLARED")]
    // Declared only as a FILTER, which shares htslib's dictionary with INFO and FORMAT names.
    #[case::info_declared_as_a_filter("LowQual=1;AC=1", "GT")]
    #[case::format_declared_as_a_filter("AC=1", "GT:LowQual")]
    #[case::info_declared_as_a_format_field("GQ=1;AC=1", "GT")]
    #[case::format_declared_as_an_info_field("AC=1", "GT:AF")]
    fn an_undeclared_tag_is_an_error(#[case] info: &str, #[case] format: &str) {
        let dir = TempDir::new().unwrap();
        let sample = if format == "GT" { "0/1" } else { "0/1:3" };
        let line = format!("chr1\t100\t.\tA\tC\t50\tPASS\t{info}\t{format}\t{sample}");
        let vcf =
            write_indexed_vcf(dir.path(), "in.vcf.gz", &vcf_header(&[FIELDS], &["S1"]), &[&line]);
        let error = write_scored_vcf(&vcf, &dir.path().join("out.bcf"), HashMap::new(), THREADS)
            .unwrap_err();
        assert!(
            format!("{error:#}").contains(
                "the record at chr1:100 has a FILTER, INFO or FORMAT tag not declared in the VCF \
                 header"
            ),
            "{error:#}"
        );
        // Read by region, the tag is not flagged, and was written under one of winnowr's names.
        assert!(writes_sharded(&vcf, "out.vcf.gz"));
        let error = write_scored_vcf(&vcf, &dir.path().join("out.vcf.gz"), HashMap::new(), THREADS)
            .unwrap_err();
        assert!(
            format!("{error:#}").contains(
                "the record at chr1:100 has an INFO or FORMAT tag not declared in the VCF header"
            ),
            "{error:#}"
        );
    }

    #[test]
    fn records_at_pos_0_are_written() {
        let dir = TempDir::new().unwrap();
        let lines = [
            "chr1\t0\t.\tN\tA\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t5\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr2\t1\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
        ];
        let header = vcf_header(&[FIELDS], &["S1"]);
        let vcf = write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &lines);
        let contigs = ["chr1", "chr2", "chr3"];
        let scan = crate::vcf::tabix_scan(&vcf, Some(&contigs[..])).unwrap();
        assert_eq!(scan.contigs, ["chr1", "chr2"]);
        assert_eq!(scan.pos0, Some(vec!["chr1".to_string()]));
        assert!(!writes_sharded(&vcf, "out.vcf.gz"));
        let output = dir.path().join("out.vcf.gz");
        write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap();
        assert_eq!(positions(&output), [(0, -1), (0, 4), (1, 0)]);
        // Without them, nothing is found.
        let vcf = write_indexed_vcf(dir.path(), "none.vcf.gz", &header, &lines[1..]);
        let scan = crate::vcf::tabix_scan(&vcf, Some(&contigs[..])).unwrap();
        assert_eq!(scan.pos0, Some(vec![]));
    }

    /// htslib reads a POS of `00` or `+0` as 0 too.
    #[rstest]
    #[case::leading_zero("00")]
    #[case::plus_sign("+0")]
    fn records_at_pos_0_written_otherwise_are_written(#[case] pos: &str) {
        let dir = TempDir::new().unwrap();
        let first = format!("chr1\t{pos}\t.\tN\tA\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5");
        let lines =
            [first.as_str(), "chr1\t5\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5"];
        let header = vcf_header(&[FIELDS], &["S1"]);
        let vcf = write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &lines);
        let scan = crate::vcf::tabix_scan(&vcf, Some(&["chr1"][..])).unwrap();
        assert_eq!(scan.pos0, Some(vec!["chr1".to_string()]));
        assert!(!writes_sharded(&vcf, "out.vcf.gz"));
        let output = dir.path().join("out.vcf.gz");
        write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap();
        assert_eq!(positions(&output), [(0, -1), (0, 4)]);
    }

    /// Each record's contig index and 0-based position.
    fn positions(vcf: &Path) -> Vec<(u32, i64)> {
        VcfReader::from_path(vcf)
            .unwrap()
            .records()
            .map(|r| r.map(|r| (r.rid().unwrap(), r.pos())).unwrap())
            .collect()
    }

    #[test]
    fn inputs_the_tabix_scan_cannot_read_are_written_in_one_pass() {
        let dir = TempDir::new().unwrap();
        let header = vcf_header(&[FIELDS], &["S1"]);
        let lines = [
            "chr1\t0\t.\tN\tA\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
            "chr1\t5\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT:GQ:AD\t0/1:30:5,5",
        ];
        let expected = [(0, -1), (0, 4)];
        let write = |vcf: &Path| {
            let output = dir.path().join("out.vcf.gz");
            write_scored_vcf(vcf, &output, HashMap::new(), THREADS).unwrap();
            positions(&output)
        };
        // A BCF: no tabix index.
        let bcf = dir.path().join("in.bcf");
        let text = write_indexed_vcf(dir.path(), "text.vcf.gz", &header, &lines);
        {
            let mut reader = VcfReader::from_path(&text).unwrap();
            let out = Header::from_template(reader.header());
            let mut writer = Writer::from_path(&bcf, &out, false, Format::Bcf).unwrap();
            for record in reader.records() {
                writer.write(&record.unwrap()).unwrap();
            }
        }
        index::build(&bcf, None, 1, IndexType::Csi(14)).unwrap();
        assert_eq!(crate::vcf::tabix_scan::<&str>(&bcf, None), None);
        assert_eq!(write(&bcf), expected);
        // A Latin-1 sample name, which rust-htslib's tabix reader panics on.
        let latin1 = header.replace("\tS1", "\tJos\u{e9}");
        let mut bytes = latin1.into_bytes();
        let at = bytes.windows(2).position(|w| w == [0xc3, 0xa9]).unwrap();
        bytes.splice(at..at + 2, [0xe9]);
        let path = dir.path().join("latin1.vcf.gz");
        {
            use std::io::Write as _;
            let mut writer = rust_htslib::bgzf::Writer::from_path(&path).unwrap();
            writer.write_all(&bytes).unwrap();
            for line in lines {
                writer.write_all(line.as_bytes()).unwrap();
                writer.write_all(b"\n").unwrap();
            }
        }
        index::build(&path, None, 1, IndexType::Tbx).unwrap();
        assert_eq!(crate::vcf::tabix_scan::<&str>(&path, None), None);
        assert_eq!(write(&path), expected);
        // A record line at POS 1 that is not UTF-8: the contigs are known, POS 0 is not.
        let raw = [
            "chr1\t1\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5;AS_RAW_MQ=caf\u{e9}\tGT\t0/1".to_string(),
            "chr1\t5\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT\t0/1".to_string(),
        ];
        let path = dir.path().join("record.vcf.gz");
        {
            use std::io::Write as _;
            let mut writer = rust_htslib::bgzf::Writer::from_path(&path).unwrap();
            writer.write_all(header.as_bytes()).unwrap();
            for line in &raw {
                let mut line = line.clone().into_bytes();
                if let Some(at) = line.windows(2).position(|w| w == [0xc3, 0xa9]) {
                    line.splice(at..at + 2, [0xe9]);
                }
                writer.write_all(&line).unwrap();
                writer.write_all(b"\n").unwrap();
            }
        }
        index::build(&path, None, 1, IndexType::Tbx).unwrap();
        let scan = crate::vcf::tabix_scan::<&str>(&path, None).unwrap();
        assert_eq!((scan.contigs, scan.pos0), (vec!["chr1".to_string()], None));
        assert_eq!(write(&path), [(0, 0), (0, 4)]);
    }

    /// A record past the first that is not UTF-8 is not seen by the tabix scan, so it is written
    /// in shards, which check it without reading it as UTF-8.
    #[test]
    fn a_record_that_is_not_utf8_is_written_in_shards() {
        let dir = TempDir::new().unwrap();
        let header = vcf_header(&[FIELDS], &["S1"]);
        let path = dir.path().join("in.vcf.gz");
        {
            use std::io::Write as _;
            let mut writer = rust_htslib::bgzf::Writer::from_path(&path).unwrap();
            writer.write_all(header.as_bytes()).unwrap();
            writer.write_all(b"chr1\t1\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5\tGT\t0/1\n").unwrap();
            writer
                .write_all(
                    b"chr1\t100\t.\tA\tC\t50\tPASS\tAC=1;AF=0.5;AS_RAW_MQ=caf\xe9\tGT\t0/1\n",
                )
                .unwrap();
        }
        index::build(&path, None, 1, IndexType::Tbx).unwrap();
        assert!(writes_sharded(&path, "out.vcf.gz"));
        let output = dir.path().join("out.vcf.gz");
        write_scored_vcf(&path, &output, HashMap::new(), THREADS).unwrap();
        assert_eq!(positions(&output), [(0, 0), (0, 99)]);
        let mut reader = VcfReader::from_path(&output).unwrap();
        let record = reader.records().nth(1).unwrap().unwrap();
        let value = record.info(b"AS_RAW_MQ").string().unwrap().unwrap();
        assert_eq!(value[0], b"caf\xe9");
    }

    #[rstest::rstest]
    #[case::appended("appended")]
    #[case::replaced("replaced")]
    #[case::symlink_parent("parent")]
    #[case::dangling_output("output_link")]
    #[case::dangling_candidate("candidate_link")]
    #[case::generated_sidecar("sidecar")]
    fn a_new_output_cannot_shadow_the_selected_index(#[case] kind: &str) {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        let csi = dir.path().join("in.vcf.gz.csi");
        let tbi = dir.path().join("in.vcf.gz.tbi");
        let original = std::fs::read(&vcf).unwrap();
        let index = std::fs::read(&tbi).unwrap();
        let output = match kind {
            "appended" => csi.clone(),
            "replaced" => dir.path().join("in.vcf.csi"),
            "parent" => {
                std::os::unix::fs::symlink(dir.path(), dir.path().join("alias")).unwrap();
                std::fs::create_dir(dir.path().join("child")).unwrap();
                dir.path().join("alias/child/../in.vcf.gz.csi")
            }
            "output_link" => {
                let output = dir.path().join("output.vcf");
                std::os::unix::fs::symlink("in.vcf.gz.csi", &output).unwrap();
                output
            }
            "candidate_link" | "sidecar" => {
                let output =
                    dir.path().join(if kind == "sidecar" { "output.vcf.gz" } else { "output.vcf" });
                let target = if kind == "sidecar" {
                    scored_vcf_index(&output).unwrap()
                } else {
                    output.clone()
                };
                std::os::unix::fs::symlink(target, &csi).unwrap();
                output
            }
            _ => unreachable!(),
        };
        let error =
            write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap_err().to_string();
        assert!(error.contains("would mask the input VCF index"), "{error}");
        assert_eq!(std::fs::read(&vcf).unwrap(), original);
        assert_eq!(std::fs::read(&tbi).unwrap(), index);
        assert!(!output.exists());
        assert!(!csi.exists());
    }

    #[rstest::rstest]
    #[case::tabix("in.vcf.gz.tbi", true)]
    #[case::later_csi("in.vcf.csi", true)]
    #[case::later_tabix_with_absent_csi("in.vcf.tbi", false)]
    #[case::aliased_later_tabix("alias/in.vcf.tbi", false)]
    fn absent_later_index_candidates_remain_usable_outputs(#[case] name: &str, #[case] csi: bool) {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        if csi {
            index::build(&vcf, None, 1, IndexType::Csi(14)).unwrap();
            std::fs::remove_file(dir.path().join("in.vcf.gz.tbi")).unwrap();
        }
        std::os::unix::fs::symlink(dir.path(), dir.path().join("alias")).unwrap();
        let selected = dir.path().join(if csi { "in.vcf.gz.csi" } else { "in.vcf.gz.tbi" });
        let before = std::fs::read(&selected).unwrap();
        write_scored_vcf(&vcf, &dir.path().join(name), HashMap::new(), THREADS).unwrap();
        assert_eq!(std::fs::read(&selected).unwrap(), before);
        let mut reader = crate::vcf::open_indexed(&vcf).unwrap();
        reader.fetch(0, 0, None).unwrap();
        assert_eq!(reader.records().count(), 5);
    }

    #[rstest::rstest]
    #[case::direct(false)]
    #[case::generated_sidecar(true)]
    fn prospective_index_names_follow_filesystem_case_rules(#[case] sidecar: bool) {
        let dir = TempDir::new().unwrap();
        let probe = dir.path().join("case-probe");
        std::fs::write(&probe, b"probe").unwrap();
        let insensitive = same_file(&probe, &dir.path().join("CASE-PROBE"));
        std::fs::remove_file(probe).unwrap();
        eprintln!("case_insensitive={insensitive} root={}", dir.path().display());
        let vcf = input(&dir);
        let tbi = dir.path().join("in.vcf.gz.tbi");
        let before = std::fs::read(&tbi).unwrap();
        let output = if sidecar {
            std::os::unix::fs::symlink("OUTPUT.VCF.GZ.TBI", dir.path().join("in.vcf.gz.csi"))
                .unwrap();
            dir.path().join("output.vcf.gz")
        } else {
            dir.path().join("in.vcf.gz.CSI")
        };
        let result = write_scored_vcf(&vcf, &output, HashMap::new(), THREADS);
        if insensitive {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("would mask the input VCF index"), "{error}");
            assert!(!output.exists());
        } else {
            result.unwrap();
            let mut reader = crate::vcf::open_indexed(&vcf).unwrap();
            reader.fetch(0, 0, None).unwrap();
            assert_eq!(reader.records().count(), 5);
        }
        assert_eq!(std::fs::read(&tbi).unwrap(), before);
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|entry| {
            !entry.unwrap().file_name().to_string_lossy().starts_with(".winnowr-path-check-")
        }));
    }

    #[test]
    fn an_existing_output_needs_no_parent_write_permission() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        let output = dir.path().join("r\u{e9}sultat.vcf");
        std::fs::write(&output, b"replace this writable file").unwrap();
        let permissions = std::fs::metadata(dir.path()).unwrap().permissions();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
        let probe = dir.path().join("permission-check");
        match std::fs::File::create_new(&probe) {
            Ok(file) => {
                drop(file);
                std::fs::set_permissions(dir.path(), permissions).unwrap();
                eprintln!("privileged permission bypass; parent-write regression not exercised");
                return;
            }
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {}
            Err(error) => {
                std::fs::set_permissions(dir.path(), permissions).unwrap();
                panic!("unexpected permission probe error: {error}");
            }
        }
        let result = write_scored_vcf(&vcf, &output, HashMap::new(), THREADS);
        std::fs::set_permissions(dir.path(), permissions).unwrap();
        result.unwrap();
        assert_eq!(VcfReader::from_path(&output).unwrap().records().count(), 5);
    }

    #[test]
    fn symlink_parents_are_resolved_before_dotdot() {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        std::fs::create_dir_all(dir.path().join("one/two")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("one/two"), dir.path().join("alias")).unwrap();
        let output = dir.path().join("alias/../in.vcf.gz.csi");
        let tbi = dir.path().join("in.vcf.gz.tbi");
        let before = std::fs::read(&tbi).unwrap();
        write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap();
        assert!(dir.path().join("one/in.vcf.gz.csi").exists());
        assert!(!dir.path().join("in.vcf.gz.csi").exists());
        assert_eq!(std::fs::read(&tbi).unwrap(), before);
    }

    #[test]
    fn an_unindexed_plain_input_does_not_reserve_index_names() {
        let dir = TempDir::new().unwrap();
        let indexed = input(&dir);
        let plain = dir.path().join("plain.vcf");
        write_scored_vcf(&indexed, &plain, HashMap::new(), THREADS).unwrap();
        let before = std::fs::read(&plain).unwrap();
        write_scored_vcf(&plain, &dir.path().join("plain.vcf.csi"), HashMap::new(), THREADS)
            .unwrap();
        assert_eq!(std::fs::read(&plain).unwrap(), before);
    }

    #[test]
    fn refuses_to_overwrite_the_input() {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        let message =
            write_scored_vcf(&vcf, &vcf, HashMap::new(), THREADS).unwrap_err().to_string();
        assert!(message.ends_with("is the input VCF; write to another path"), "{message}");
        // So is a hard link to it, which has another canonical path.
        let linked = dir.path().join("linked.vcf.gz");
        std::fs::hard_link(&vcf, &linked).unwrap();
        let message =
            write_scored_vcf(&vcf, &linked, HashMap::new(), THREADS).unwrap_err().to_string();
        assert!(message.ends_with("is the input VCF; write to another path"), "{message}");
        // Plain-text and BCF outputs are written too.
        let plain = dir.path().join("out.vcf");
        write_scored_vcf(&vcf, &plain, HashMap::new(), THREADS).unwrap();
        assert!(std::fs::read_to_string(&plain).unwrap().contains("##FILTER=<ID=WinnowrDrop"));
        let bcf = dir.path().join("out.bcf");
        write_scored_vcf(&vcf, &bcf, HashMap::new(), THREADS).unwrap();
        assert!(dir.path().join("out.bcf.csi").exists());
        // Nor is the index written beside the output the input's index (a .tbi beside bgzipped
        // VCF, a .csi beside BCF), through a link to it.
        let index = dir.path().join("in.vcf.gz.tbi");
        let before = std::fs::read(&index).unwrap();
        for (output, sidecar) in [("s.vcf.gz", "s.vcf.gz.tbi"), ("s.bcf", "s.bcf.csi")] {
            std::os::unix::fs::symlink(&index, dir.path().join(sidecar)).unwrap();
            let output = dir.path().join(output);
            let message =
                write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap_err().to_string();
            assert!(
                message.ends_with("is the input VCF index; write to another path"),
                "{message}"
            );
        }
        assert!(std::fs::read(&index).unwrap() == before);
    }

    #[test]
    fn output_sidecars_may_use_tabix_names_beside_bcf() {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        let bcf = dir.path().join("input.bcf");
        write_scored_vcf(&vcf, &bcf, HashMap::new(), THREADS).unwrap();
        let csi = dir.path().join("input.bcf.csi");
        let before = std::fs::read(&csi).unwrap();
        for name in ["input.bcf.tbi", "input.tbi"] {
            let unused = dir.path().join(name);
            std::fs::write(&unused, b"unused tabix path").unwrap();
            let output = dir.path().join(format!("{name}.vcf.gz"));
            let sidecar = scored_vcf_index(&output).unwrap();
            std::os::unix::fs::symlink(&unused, sidecar).unwrap();
            write_scored_vcf(&bcf, &output, HashMap::new(), THREADS).unwrap();
            assert!(crate::vcf::has_index(&output));
        }
        let output = dir.path().join("protected.vcf.gz");
        std::os::unix::fs::symlink(&csi, scored_vcf_index(&output).unwrap()).unwrap();
        let error =
            write_scored_vcf(&bcf, &output, HashMap::new(), THREADS).unwrap_err().to_string();
        assert!(error.ends_with("is the input VCF index; write to another path"), "{error}");
        assert_eq!(std::fs::read(&csi).unwrap(), before);
    }

    #[rstest]
    #[case::gzip("scored.vcf.gz")]
    #[case::bgzf("scored.vcf.bgz")]
    #[case::bcf("scored.bcf")]
    fn an_index_alias_cannot_replace_an_existing_scored_output(
        #[case] name: &str,
        #[values(false, true)] hard_link: bool,
    ) {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        let output = dir.path().join(name);
        let sidecar = scored_vcf_index(&output).unwrap();
        let before = b"preserve the existing scored output";
        std::fs::write(&output, before).unwrap();
        if hard_link {
            std::fs::hard_link(&output, &sidecar).unwrap();
        } else {
            std::os::unix::fs::symlink(name, &sidecar).unwrap();
        }
        let error =
            write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap_err().to_string();
        assert!(error.ends_with("is the input scored VCF; write to another path"), "{error}");
        assert_eq!(std::fs::read(&output).unwrap(), before);
        assert_eq!(std::fs::read(&sidecar).unwrap(), before);
    }

    #[test]
    fn a_dangling_index_link_cannot_overwrite_the_scored_output() {
        let dir = TempDir::new().unwrap();
        let vcf = input(&dir);
        for name in ["scored.vcf.gz", "scored.vcf.bgz", "scored.bcf"] {
            let output = dir.path().join(name);
            let sidecar = scored_vcf_index(&output).unwrap();
            std::os::unix::fs::symlink(name, &sidecar).unwrap();
            let result = write_scored_vcf(&vcf, &output, HashMap::new(), THREADS);
            assert!(result.is_err(), "index overwrote {name}");
            let mut reader = rust_htslib::bcf::Reader::from_path(&output).unwrap();
            assert!(reader.records().next().unwrap().is_ok());
            std::fs::remove_file(sidecar).unwrap();
            write_scored_vcf(&vcf, &output, HashMap::new(), THREADS).unwrap();
            assert!(scored_vcf_index(&output).unwrap().exists());
        }
    }

    #[test]
    fn names_the_index_written_beside_the_output() {
        let index = |output: &str| scored_vcf_index(Path::new(output));
        assert_eq!(index("out.BCF"), Some("out.BCF.csi".into()));
        assert_eq!(index("out.vcf.gz"), Some("out.vcf.gz.tbi".into()));
        assert_eq!(index("out.vcf.bgz"), Some("out.vcf.bgz.tbi".into()));
        assert_eq!(index("out.vcf"), None);
    }
}
