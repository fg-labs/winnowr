//! The sharded pass over a sample's calls, reads and reference that measures read evidence at
//! the calls (`winnowr strlen` and `winnowr pileup`): the regions measured, their shards, the
//! readers each shard uses, and helpers for reading the reads.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use anyhow::{Context, Result, bail};

use crate::shards::ReaderPool;

/// Unmapped, secondary, QC-fail, duplicate and supplementary reads are skipped.
pub const SKIP_FLAGS: u16 = 0x4 | 0x100 | 0x200 | 0x400 | 0x800;

/// The query position aligned (M/=/X) at a reference position, or `None` (deleted or
/// unaligned).
pub fn query_position_at(
    cigar: &[rust_htslib::bam::record::Cigar],
    reference_start: u64,
    target: u64,
) -> Option<u64> {
    use rust_htslib::bam::record::Cigar;
    let (mut ref_pos, mut query_pos) = (reference_start, 0u64);
    for op in cigar {
        let length = u64::from(op.len());
        let consumes_reference = matches!(
            op,
            Cigar::Match(_) | Cigar::Del(_) | Cigar::RefSkip(_) | Cigar::Equal(_) | Cigar::Diff(_)
        );
        let consumes_query = matches!(
            op,
            Cigar::Match(_) | Cigar::Ins(_) | Cigar::SoftClip(_) | Cigar::Equal(_) | Cigar::Diff(_)
        );
        if consumes_reference && ref_pos <= target && target < ref_pos + length {
            return consumes_query.then(|| query_pos + target - ref_pos);
        }
        if consumes_reference {
            ref_pos += length;
        }
        if consumes_query {
            query_pos += length;
        }
    }
    None
}

/// The end of a read's alignment as htslib computes it to decide whether a read overlaps a
/// fetched region: its start plus the reference length of its CIGAR, and at least one base.
pub fn alignment_end(record: &rust_htslib::bam::Record) -> i64 {
    // CIGAR operations consuming the reference: M, D, N, = and X.
    let reference_length: i64 = record
        .raw_cigar()
        .iter()
        .filter(|&&op| matches!(op & 0xf, 0 | 2 | 3 | 7 | 8))
        .map(|&op| i64::from(op >> 4))
        .sum();
    record.pos() + reference_length.max(1)
}

/// A read's length across a repeat tract: the query positions aligned at the tract's anchors
/// `first` and `last` (0-based, inclusive: the flank bases outside each end of the tract, with
/// `flank` bases on each side) and the query bases between the flanks (the observed tract
/// length), or `None` if the read starts after `first` or is not aligned at both anchors.
pub fn anchored_length(
    cigar: &[rust_htslib::bam::record::Cigar],
    start: i64,
    first: i64,
    last: i64,
    flank: u64,
) -> Option<(u64, u64, i64)> {
    if start > first {
        return None;
    }
    let first_query = query_position_at(cigar, start as u64, first as u64)?;
    let last_query = query_position_at(cigar, start as u64, last as u64)?;
    let observed = last_query as i64 - first_query as i64 + 1 - 2 * flank as i64;
    Some((first_query, last_query, observed))
}

/// The most reads measured at a site (a strlen tract or a pileup call): a site is read until
/// more than this many reads overlap it, so its count of overlapping reads is capped at
/// `MAX_READS + 1`.
///
/// Neither strlen nor pileup tables record it, so changing it changes the evidence that models
/// were trained on without changing their recorded settings.
pub const MAX_READS: u64 = 600;

/// Sites whose spans are at most this far apart are read from one fetch.
pub const CLUSTER_GAP: i64 = 10_000;

/// The sites of one contig that [`sweep_clusters`] adds reads to, with the state they accumulate.
pub trait SweepSites {
    /// Lazily computed read data shared across sites; reset before each read.
    type ReadCache: Default;

    /// True once site `i` takes no more reads.
    fn is_full(&self, i: usize) -> bool;

    /// Adds a read that overlaps site `i`'s span. `alignment` is the read's 0-based, half-open
    /// span on the reference (its start, and its [`alignment_end`]), computed once per read.
    /// `cache` holds any read data decoded by a previous site, so later sites can reuse it.
    fn add(
        &mut self,
        i: usize,
        record: &rust_htslib::bam::Record,
        alignment: (i64, i64),
        cache: &mut Self::ReadCache,
    );
}

/// Reads the reads at sites on one contig, given each site's 0-based, half-open span: sites whose
/// spans are at most [`CLUSTER_GAP`] apart are read from one fetch, and each read (not one with
/// [`SKIP_FLAGS`]) is added, in start order, to every site whose span it overlaps, until the site
/// is full. A site gets the reads a fetch of its own span would return, in the same order.
///
/// # Errors
///
/// Returns an error if the reads cannot be fetched or read.
pub fn sweep_clusters<S: SweepSites>(
    reads: &mut rust_htslib::bam::IndexedReader,
    tid: u32,
    spans: &[(i64, i64)],
    sites: &mut S,
) -> Result<()> {
    use rust_htslib::bam::Read as _;
    let mut by_start: Vec<usize> = (0..spans.len()).collect();
    by_start.sort_by_key(|&i| spans[i].0);
    let mut record = rust_htslib::bam::Record::new();
    let mut cluster_start = 0;
    while cluster_start < by_start.len() {
        let mut cluster_end = cluster_start + 1;
        let mut fetch_end = spans[by_start[cluster_start]].1;
        while cluster_end < by_start.len()
            && spans[by_start[cluster_end]].0 <= fetch_end + CLUSTER_GAP
        {
            fetch_end = fetch_end.max(spans[by_start[cluster_end]].1);
            cluster_end += 1;
        }
        let cluster = &by_start[cluster_start..cluster_end];
        cluster_start = cluster_end;
        let fetch_start = spans[cluster[0]].0.max(0);
        reads.fetch((tid, fetch_start, fetch_end)).context("failed to fetch reads")?;
        // Sites are opened once a read may overlap them, and closed once no later read can (reads
        // come in start order) or they are full.
        let (mut next, mut open) = (0, Vec::<usize>::new());
        while let Some(read) = reads.read(&mut record) {
            read.context("failed to read a read")?;
            if record.flags() & SKIP_FLAGS != 0 {
                continue;
            }
            let (start, end) = (record.pos(), alignment_end(&record));
            while next < cluster.len() && spans[cluster[next]].0 < end {
                open.push(cluster[next]);
                next += 1;
            }
            open.retain(|&i| spans[i].1 > start && !sites.is_full(i));
            if open.is_empty() && next == cluster.len() {
                break;
            }
            // Decoded data are shared across sites, but never across reads.
            let mut cache = S::ReadCache::default();
            for &i in &open {
                let (beg, site_end) = spans[i];
                if start < site_end && end > beg {
                    sites.add(i, &record, (start, end), &mut cache);
                }
            }
        }
    }
    Ok(())
}

/// A region of records to measure: a contig, or the 1-based inclusive span of one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// The contig.
    pub chrom: String,
    /// The 1-based inclusive start and end (`None` for the whole contig).
    pub span: Option<(u64, u64)>,
}

impl Region {
    /// Parses `chrom` or `chrom:start-end` (1-based, inclusive; commas allowed in numbers).
    ///
    /// The span is split off at the last `:`, so a contig name may contain colons when a span is
    /// given; a region that is exactly a contig name in `contigs` is that whole contig even if
    /// its name contains a colon (e.g. `HLA-A*01:01:01:01`).
    ///
    /// # Errors
    ///
    /// Returns an error if the region is malformed or its start is below 1.
    pub fn parse<S: AsRef<str>>(region: &str, contigs: &[S]) -> Result<Self> {
        if contigs.iter().any(|contig| contig.as_ref() == region) {
            return Ok(Region { chrom: region.to_string(), span: None });
        }
        let Some((chrom, span)) = region.rsplit_once(':') else {
            return Ok(Region { chrom: region.to_string(), span: None });
        };
        let span = span.replace(',', "");
        let bounds = span.split_once('-').and_then(|(start, end)| {
            let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
            if !digits(start) || !digits(end) {
                return None;
            }
            Some((start.parse::<u64>().ok()?, end.parse::<u64>().ok()?))
        });
        match bounds {
            Some((start, end)) if !chrom.is_empty() && 1 <= start && start <= end => {
                Ok(Region { chrom: chrom.to_string(), span: Some((start, end)) })
            }
            _ => bail!("malformed region (expected chrom or chrom:start-end): {region}"),
        }
    }

    /// The region as written.
    fn as_ref_str(&self) -> String {
        match self.span {
            Some((start, end)) => format!("{}:{start}-{end}", self.chrom),
            None => self.chrom.clone(),
        }
    }
}

/// The header provenance checked before a read pass and the policy applied to it.
#[derive(Debug, Clone, Copy)]
pub struct AlignmentExpectation<'a> {
    /// The header's raw provenance (`None` when its header records none).
    pub provenance: Option<&'a crate::alignment_provenance::AlignmentProvenance>,
    /// Original @PG/@SQ records, compared without interpreting command-line arguments.
    pub header_records: &'a [String],
    /// The same user overrides applied during preflight.
    pub tags: Option<&'a crate::alignment_provenance::AlignmentTags>,
    /// Whether a changed provenance is an error rather than a warning.
    pub strict: bool,
}

impl AlignmentExpectation<'_> {
    /// Checks provenance from the same reader that supplies the measured alignments.
    fn check(&self, header: &rust_htslib::bam::HeaderView, path: &Path) -> Result<()> {
        use crate::alignment_provenance::{
            AlignmentProvenance, AlignmentTags, HeaderProvenance, resolve_reads,
        };
        if matches!(self.tags, Some(AlignmentTags::None)) {
            return Ok(());
        }
        let observed =
            HeaderProvenance::from_header_text(&String::from_utf8_lossy(header.as_bytes()));
        let effective =
            |provenance: Option<AlignmentProvenance>| -> Result<Option<AlignmentProvenance>> {
                let source =
                    Box::new(move || Ok(provenance)) as crate::alignment_provenance::LazySource<'_>;
                Ok(resolve_reads(self.tags, None, vec![(path.display().to_string(), source)])?
                    .provenance)
            };
        let raw_changed = observed.header_records != self.header_records;
        let expected = effective(self.provenance.cloned())?;
        let actual = effective(observed.provenance)?;
        if raw_changed || actual != expected {
            let describe = |provenance: &Option<AlignmentProvenance>| {
                provenance.as_ref().map_or_else(|| "unknown".to_string(), ToString::to_string)
            };
            let detail = if raw_changed {
                "; raw @PG/@SQ records changed (declarations do not verify a replacement header)"
            } else {
                ""
            };
            let message = format!(
                "the alignment provenance of {} changed between preflight and reading its alignments: \
                 expected {}, observed {}{detail}",
                path.display(),
                describe(&expected),
                describe(&actual)
            );
            if self.strict {
                bail!("{message}");
            }
            log::warn!("{message}");
        }
        Ok(())
    }
}

/// Index searches successfully opened by measurement readers, shared across passes and workers.
#[derive(Debug, Clone, Default)]
pub struct ReadIndexObservation(Arc<Mutex<Vec<Vec<std::path::PathBuf>>>>);

impl ReadIndexObservation {
    /// The distinct searches consumed so far. BAM and CRAM searches remain independent.
    pub fn searches(&self) -> Vec<Vec<std::path::PathBuf>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Retains a successful reader's search without duplicating it for each shard.
    fn record(&self, paths: Vec<std::path::PathBuf>) {
        let mut searches = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if !searches.contains(&paths) {
            searches.push(paths);
        }
    }
}

/// The index search for the file's detected format, using htslib's local search order. The
/// nonblocking open cannot hang on a replaced FIFO. Missing or unreadable files retain the BAM
/// search for early output checks; a measurement reader still fails to open those inputs.
pub fn alignment_index_paths(path: &Path) -> Vec<std::path::PathBuf> {
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    let mut magic = [0; 4];
    let cram = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .ok()
        .is_some_and(|file| {
            file.metadata().is_ok_and(|metadata| metadata.is_file())
                && file.read_exact_at(&mut magic, 0).is_ok()
                && &magic == b"CRAM"
        });
    if cram {
        crate::vcf::local_index_paths(path, &["csi", "crai"])
    } else {
        crate::vcf::local_index_paths(path, &["csi", "bai"])
    }
}

/// The inputs of a read pass (`winnowr strlen` and `winnowr pileup`).
#[derive(Debug, Clone)]
pub struct ReadInputs<'a> {
    /// The indexed VCF of the calls.
    pub vcf: &'a Path,
    /// The indexed BAM/CRAM of the sample's reads.
    pub alignments: &'a Path,
    /// The indexed reference FASTA (also used to decode CRAM).
    pub reference: &'a Path,
    /// Checks each measurement reader against the preflight header before consuming reads.
    pub expected_alignment: Option<AlignmentExpectation<'a>>,
    /// Retains the searches used by successful measurement readers for a later output audit.
    pub observed_indexes: Option<ReadIndexObservation>,
}

impl ReadInputs<'_> {
    /// Opens an indexed measurement reader, retaining the format-specific index search. An
    /// explicit index binds that search to the reader: if the file changes format between
    /// detection and opening, its incompatible index fails rather than recording a stale search.
    ///
    /// # Errors
    ///
    /// Returns an error if the alignments or their selected index cannot be opened.
    pub fn open_alignments(&self) -> Result<rust_htslib::bam::IndexedReader> {
        let open_error =
            || format!("failed to open {} (it must be indexed)", self.alignments.display());
        let Some(observed) = &self.observed_indexes else {
            return rust_htslib::bam::IndexedReader::from_path(self.alignments)
                .with_context(open_error);
        };
        let paths = alignment_index_paths(self.alignments);
        let index = paths.iter().find(|path| path.exists()).with_context(open_error)?;
        let reads =
            rust_htslib::bam::IndexedReader::from_path_and_index(self.alignments, index.as_path())
                .with_context(open_error)?;
        observed.record(paths);
        Ok(reads)
    }
}

/// The span length that shards of a region are split into, so that large contigs are measured
/// in parallel.
const SHARD_SPAN: u64 = 5_000_000;

/// The shards of regions: each region split into spans of at most [`SHARD_SPAN`] bp, as
/// `(contig, start, end)` (1-based, inclusive), in region order.
///
/// # Errors
///
/// Returns an error if a region is malformed, its contig is not in the VCF header or the
/// reference, it starts past the contig's end, or it overlaps another region (its records would
/// be measured twice).
fn plan_shards<S: AsRef<str>>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
    summary: &crate::vcf::VcfSummary,
) -> Result<Vec<(String, u64, u64)>> {
    let contigs: Vec<&str> = summary.contig_lengths.keys().map(String::as_str).collect();
    let fasta = crate::fasta::IndexedFasta::from_path(inputs.reference)?;
    let mut shards = Vec::new();
    // The regions planned so far, as (contig, start, end, as written).
    let mut planned: Vec<(String, u64, u64, String)> = Vec::new();
    for region in regions {
        let region = Region::parse(region.as_ref(), &contigs)?;
        if !contigs.contains(&region.chrom.as_str()) {
            bail!(
                "contig {} of region {} is not in the VCF header: measure only contigs the VCF \
                 has (`strlen` and `pileup` take them from --regions, by default chr1-chr22)",
                region.chrom,
                region.as_ref_str()
            );
        }
        let length = fasta.checked_length(&region.chrom)?.with_context(|| {
            format!(
                "contig {} is not in the reference {}",
                region.chrom,
                inputs.reference.display()
            )
        })?;
        // The contig's extent: records past the reference's end are an error (see
        // `Shard::check_record_end`), not silently dropped, so whole contigs span the VCF's length
        // too.
        let vcf_length = summary.contig_lengths.get(&region.chrom).copied().flatten();
        let extent = length.max(vcf_length.unwrap_or(0)).max(1);
        let (start, end) = region.span.unwrap_or((1, extent));
        // A span past the contig's end (e.g. of another build) would give no shards and an empty
        // table; one that only runs past it is clamped.
        if start > extent {
            bail!(
                "region {} starts past the end of contig {} ({extent} bp)",
                region.as_ref_str(),
                region.chrom
            );
        }
        let end = end.min(extent);
        let overlapped = planned.iter().find(|(chrom, other_start, other_end, _)| {
            *chrom == region.chrom && start <= *other_end && *other_start <= end
        });
        if let Some((.., other)) = overlapped {
            bail!(
                "regions {other} and {} overlap, so their records would be measured twice",
                region.as_ref_str()
            );
        }
        planned.push((region.chrom.clone(), start, end, region.as_ref_str()));
        // Shards split the span up to the reference's (checked) length; any rest, past the
        // reference (where records are an error) and sized only by the VCF header, is the last
        // shard's, so a corrupt header length cannot make countless shards.
        let split_end = end.min(length.max(start));
        let mut shard_start = start;
        while shard_start <= split_end {
            let shard_end = (shard_start + SHARD_SPAN - 1).min(split_end);
            shards.push((region.chrom.clone(), shard_start, shard_end));
            shard_start = shard_end + 1;
        }
        if let Some(last) = shards.last_mut() {
            last.2 = end;
        }
    }
    Ok(shards)
}

/// One shard of a region being measured, with readers open on its contig.
pub struct Shard<'r> {
    /// The contig.
    pub chrom: &'r str,
    /// The 1-based, inclusive start of the records to measure.
    pub start: u64,
    /// The 1-based, inclusive end of the records to measure.
    pub end: u64,
    /// The VCF reader.
    pub vcf: &'r mut rust_htslib::bcf::IndexedReader,
    /// The BAM/CRAM reader.
    pub reads: &'r mut rust_htslib::bam::IndexedReader,
    /// The contig's reference sequence.
    pub contig: &'r [u8],
}

impl Shard<'_> {
    /// Fetches the shard's records from the VCF; returns false if the contig has no records in
    /// the VCF's index.
    ///
    /// # Errors
    ///
    /// Returns an error if the fetch fails otherwise.
    pub fn fetch_records(&mut self, vcf_path: &Path) -> Result<bool> {
        use rust_htslib::bcf::Read as _;
        let rid = self.vcf.header().name2rid(self.chrom.as_bytes())?;
        match self.vcf.fetch(rid, self.start - 1, Some(self.end - 1)) {
            Ok(()) => Ok(true),
            Err(rust_htslib::errors::Error::GenomicSeek { .. }) => Ok(false),
            Err(e) => Err(e).with_context(|| {
                format!(
                    "failed to fetch {}:{}-{} from {}",
                    self.chrom,
                    self.start,
                    self.end,
                    vcf_path.display()
                )
            }),
        }
    }

    /// Checks that a record's REF lies on the contig.
    ///
    /// # Errors
    ///
    /// Returns an error, suggesting the VCF was called against another reference, if not.
    pub fn check_record_end(&self, pos: u64, reference: &[u8], fasta: &Path) -> Result<()> {
        if pos - 1 + reference.len() as u64 > self.contig.len() as u64 {
            bail!(
                "the record at {chrom}:{pos} extends past the end of {chrom} ({} bp) in {}: was \
                 the VCF called against this reference?",
                self.contig.len(),
                fasta.display(),
                chrom = self.chrom,
            );
        }
        Ok(())
    }

    /// The target ID of the shard's contig in the reads.
    ///
    /// # Errors
    ///
    /// Returns an error if the contig is not in the reads' header.
    pub fn tid(&self, alignments: &Path) -> Result<u32> {
        use rust_htslib::bam::Read as _;
        self.reads.header().tid(self.chrom.as_bytes()).with_context(|| {
            format!("contig {} is not in the header of {}", self.chrom, alignments.display())
        })
    }
}

/// Splits regions into shards (see [`plan_shards`]) and runs `measure` on each, in parallel on the
/// rayon pool, with readers opened on the inputs (decoding the CRAM fields `cram_fields`) and
/// reused across shards. Returns the results in shard order.
///
/// # Errors
///
/// Returns an error if the regions are malformed (see [`plan_shards`]), a file cannot be opened,
/// or `measure` fails.
pub fn run_shards<S, T, F>(
    inputs: &ReadInputs<'_>,
    regions: &[S],
    cram_fields: u32,
    measure: F,
) -> Result<Vec<T>>
where
    S: AsRef<str>,
    T: Send,
    F: Fn(&mut Shard<'_>) -> Result<T> + Sync,
{
    use rayon::prelude::*;
    let summary = crate::vcf::VcfSummary::read(inputs.vcf)?;
    let shards = plan_shards(inputs, regions, &summary)?;
    // Reader sets are pooled, so at most one per worker thread is ever opened, and a shard takes
    // a set already holding its contig's sequence when there is one.
    let pool: ReaderPool<ShardReaders> = ReaderPool::new();
    let contigs = SharedContigs::default();
    shards
        .par_iter()
        .map(|(chrom, start, end)| {
            let mut readers =
                pool.take_preferring(|readers| readers.holds(chrom)).unwrap_or_default();
            let measured = readers.open(inputs, chrom, cram_fields, &contigs).and_then(
                |(vcf, reads, contig)| {
                    measure(&mut Shard { chrom, start: *start, end: *end, vcf, reads, contig })
                },
            );
            pool.put(readers);
            measured
        })
        .collect()
}

/// Contig sequences shared by the reader sets of a pass: sets on the same contig hold one copy of
/// its sequence, read once while any set holds it, rather than one copy each (a whole contig,
/// e.g. ~250 MB for chr1, per worker thread).
#[derive(Default)]
struct SharedContigs(Mutex<HashMap<String, Arc<ContigSlot>>>);

/// A contig's sequence while some reader set holds it; locked while it is read.
type ContigSlot = Mutex<Weak<Vec<u8>>>;

impl SharedContigs {
    /// `chrom`'s sequence: the copy a reader set holds, else read from `fasta`. A sequence is
    /// freed when no set holds it.
    fn get(&self, chrom: &str, fasta: &mut crate::fasta::IndexedFasta) -> Result<Arc<Vec<u8>>> {
        let slot = {
            let mut slots = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(slots.entry(chrom.to_string()).or_default())
        };
        // Sets that want the contig while it is being read wait for it rather than read it too.
        let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(sequence) = slot.upgrade() {
            return Ok(sequence);
        }
        let sequence = Arc::new(fasta.fetch_all(chrom)?);
        *slot = Arc::downgrade(&sequence);
        Ok(sequence)
    }
}

/// Readers and the last contig's sequence, reused across the shards measured with them.
#[derive(Default)]
struct ShardReaders {
    vcf: Option<rust_htslib::bcf::IndexedReader>,
    reads: Option<rust_htslib::bam::IndexedReader>,
    fasta: Option<crate::fasta::IndexedFasta>,
    contig: Option<(String, Arc<Vec<u8>>)>,
}

impl ShardReaders {
    /// True if the readers hold `chrom`'s sequence.
    fn holds(&self, chrom: &str) -> bool {
        self.contig.as_ref().is_some_and(|(name, _)| name == chrom)
    }

    /// Opens the readers not yet open (the reads decoding only the CRAM fields `cram_fields`) and
    /// takes `chrom`'s sequence from `contigs` if it is not held.
    fn open(
        &mut self,
        inputs: &ReadInputs<'_>,
        chrom: &str,
        cram_fields: u32,
        contigs: &SharedContigs,
    ) -> Result<(&mut rust_htslib::bcf::IndexedReader, &mut rust_htslib::bam::IndexedReader, &[u8])>
    {
        use rust_htslib::bam::Read as _;
        if self.vcf.is_none() {
            self.vcf = Some(crate::vcf::open_indexed(inputs.vcf)?);
        }
        if self.reads.is_none() {
            let mut reads = inputs.open_alignments()?;
            if let Some(expected) = inputs.expected_alignment {
                expected.check(reads.header(), inputs.alignments)?;
            }
            reads.set_reference(inputs.reference)?;
            // Ignored for BAM; for CRAM, skips decoding the fields the pass does not read.
            let _ = reads.set_cram_options(
                rust_htslib::htslib::hts_fmt_option_CRAM_OPT_REQUIRED_FIELDS,
                cram_fields,
            );
            self.reads = Some(reads);
        }
        if self.fasta.is_none() {
            self.fasta = Some(crate::fasta::IndexedFasta::from_path(inputs.reference)?);
        }
        if !self.holds(chrom) {
            // The last contig is let go first, so it is freed if no other set holds it.
            self.contig = None;
            let sequence = contigs.get(chrom, self.fasta.as_mut().expect("opened above"))?;
            self.contig = Some((chrom.to_string(), sequence));
        }
        let (Some(vcf), Some(reads), Some((_, contig))) =
            (self.vcf.as_mut(), self.reads.as_mut(), self.contig.as_ref())
        else {
            unreachable!("the readers were opened above");
        };
        Ok((vcf, reads, contig.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Converts the generated BAM fixture to indexed CRAM using its local reference.
    fn to_cram(path: &Path, reference: &Path) -> std::path::PathBuf {
        use rust_htslib::bam::{self, Read as _};
        let mut source = bam::Reader::from_path(path).unwrap();
        let text = String::from_utf8_lossy(source.header().as_bytes()).replace(
            "\tLN:100",
            &format!("\tLN:100\tM5:8adc5937e635f6c9af646f0b23560fae\tUR:{}", reference.display()),
        );
        let header = bam::Header::from_template(&bam::HeaderView::from_bytes(text.as_bytes()));
        let output = path.with_extension("cram");
        let mut writer = bam::Writer::from_path(&output, &header, bam::Format::Cram).unwrap();
        writer.set_reference(reference).unwrap();
        for record in source.records() {
            writer.write(&record.unwrap()).unwrap();
        }
        drop(writer);
        bam::index::build(&output, None, bam::index::Type::Bai, 1).unwrap();
        output
    }

    #[rstest]
    fn explicit_alignment_indexes_match_reader_format(#[values(false, true)] cram: bool) {
        use rust_htslib::bam::{self, Read as _};

        let dir = tempfile::TempDir::new().unwrap();
        let reference = crate::testing::write_fasta(dir.path(), &[("chr1", &"A".repeat(100))]);
        let read = crate::testing::TestRead {
            name: "r1",
            pos: 10,
            cigar: "20M",
            bases: None,
            qual: 30,
            quals: None,
            mapq: 60,
            flag: 0,
        };
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", 100, &[read]);
        let cram_path = to_cram(&bam, &reference);
        let path = if cram { &cram_path } else { &bam };
        let incompatible = if cram {
            bam.with_extension("bam.bai")
        } else {
            cram_path.with_extension("cram.crai")
        };
        assert!(bam::IndexedReader::from_path_and_index(path, &incompatible).is_err());
        let candidates = if cram {
            crate::vcf::local_index_paths(path, &["csi", "crai"])
        } else {
            crate::vcf::local_index_paths(path, &["csi", "bai"])
        };
        let observed = ReadIndexObservation::default();
        let inputs = ReadInputs {
            vcf: Path::new("unused.vcf.gz"),
            alignments: path,
            reference: &reference,
            expected_alignment: None,
            observed_indexes: Some(observed.clone()),
        };
        let mut reads = inputs.open_alignments().unwrap();
        assert_eq!(observed.searches(), vec![candidates]);
        reads.set_reference(&reference).unwrap();
        reads.fetch((0, 0, 100)).unwrap();
        let records: Vec<_> = reads.records().map(Result::unwrap).collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].qname(), b"r1");
        assert_eq!(records[0].pos(), 10);
    }

    #[test]
    fn index_observation_retains_each_measured_format_search() {
        let dir = tempfile::TempDir::new().unwrap();
        let reference = crate::testing::write_fasta(dir.path(), &[("chr1", &"A".repeat(100))]);
        let bam = crate::testing::write_bam(dir.path(), "reads.bam", 100, &[]);
        let cram = to_cram(&bam, &reference);
        let observed = ReadIndexObservation::default();
        let inputs = ReadInputs {
            vcf: Path::new("unused.vcf.gz"),
            alignments: &bam,
            reference: &reference,
            expected_alignment: None,
            observed_indexes: Some(observed.clone()),
        };
        assert!(observed.searches().is_empty());
        drop(inputs.open_alignments().unwrap());
        drop(inputs.clone().open_alignments().unwrap());
        let bam_paths = crate::vcf::local_index_paths(&bam, &["csi", "bai"]);
        assert_eq!(observed.searches(), vec![bam_paths.clone()]);
        std::fs::copy(&cram, &bam).unwrap();
        std::fs::copy(cram.with_extension("cram.crai"), bam.with_extension("bam.crai")).unwrap();
        drop(inputs.open_alignments().unwrap());
        assert_eq!(
            observed.searches(),
            vec![bam_paths, crate::vcf::local_index_paths(&bam, &["csi", "crai"])]
        );
    }

    #[rstest]
    #[case::different(true, true, &[], true, false)]
    #[case::newly_known(false, true, &[], true, false)]
    #[case::newly_unknown(true, false, &[], true, false)]
    #[case::overridden(true, true, &["aligner=bwa"], true, false)]
    #[case::removed(true, true, &["aligner!"], true, false)]
    #[case::disabled(true, true, &["none"], true, true)]
    #[case::warning(true, true, &[], false, true)]
    fn replacement_header_is_checked_before_measuring(
        #[case] original_known: bool,
        #[case] replacement_known: bool,
        #[case] overrides: &[&str],
        #[case] strict: bool,
        #[case] accepted: bool,
        #[values(false, true)] cram: bool,
    ) {
        let dir = tempfile::TempDir::new().unwrap();
        let fasta = crate::testing::write_fasta(dir.path(), &[("chr1", &"A".repeat(100))]);
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "calls.vcf.gz", &header, &[]);
        // Identity comes from PN, independent of the options that BWA's lossy CL cannot prove.
        let original_program = "@PG\tID:bwa\tPN:bwa\tCL:bwa mem ref.fa r.fq";
        let replacement_program = "@PG\tID:bwa-mem3\tPN:bwa-mem3\tCL:bwa-mem3 mem ref.fa r.fq";
        let original_lines = [original_program];
        let replacement_lines = [replacement_program];
        let bam = crate::testing::write_bam_with_header(
            dir.path(),
            "reads.bam",
            100,
            &[],
            if original_known { &original_lines } else { &[] },
        );
        let bam = if cram { to_cram(&bam, &fasta) } else { bam };
        let expected = crate::alignment_provenance::HeaderProvenance::read(&bam).unwrap();
        let replacement = crate::testing::write_bam_with_header(
            dir.path(),
            "replacement.bam",
            100,
            &[],
            if replacement_known { &replacement_lines } else { &[] },
        );
        let replacement = if cram { to_cram(&replacement, &fasta) } else { replacement };
        std::fs::rename(&replacement, &bam).unwrap();
        let index = if cram { "cram.crai" } else { "bam.bai" };
        std::fs::rename(replacement.with_extension(index), bam.with_extension(index)).unwrap();
        let tags = crate::alignment_provenance::AlignmentTags::parse(overrides).unwrap();
        let inputs = ReadInputs {
            observed_indexes: None,
            vcf: &vcf,
            alignments: &bam,
            reference: &fasta,
            expected_alignment: Some(AlignmentExpectation {
                provenance: expected.provenance.as_ref(),
                header_records: &expected.header_records,
                tags: tags.as_ref(),
                strict,
            }),
        };
        let mut readers = ShardReaders::default();
        let contigs = SharedContigs::default();
        let result = readers.open(&inputs, "chr1", 0, &contigs);
        assert_eq!(result.is_ok(), accepted, "unexpected reader result: {:?}", result.err());
    }

    /// A declaration must not hide a changed lossy CL when the worker opens its own reader.
    #[rstest]
    #[case::bwa("bwa", "bwa mem -p ref.fa reads.fq", "bwa mem ref.fa reads.fq", &["p=", "options-inference=explicit"], true, false)]
    #[case::minimap2("minimap2", "minimap2 -a ref.fa reads.fq", "minimap2 -a -x sr ref.fa reads.fq", &["a=", "options-inference=explicit"], true, false)]
    #[case::unchanged("bwa", "bwa mem -p ref.fa reads.fq", "bwa mem -p ref.fa reads.fq", &["p=", "options-inference=explicit"], true, true)]
    #[case::warn("bwa", "bwa mem -p ref.fa reads.fq", "bwa mem ref.fa reads.fq", &["p=", "options-inference=explicit"], false, true)]
    #[case::disabled("bwa", "bwa mem -p ref.fa reads.fq", "bwa mem ref.fa reads.fq", &["none"], true, true)]
    fn review_declaration_does_not_hide_a_changed_command_line(
        #[case] aligner: &str,
        #[case] original: &str,
        #[case] replacement: &str,
        #[case] overrides: &[&str],
        #[case] strict: bool,
        #[case] accepted: bool,
        #[values(false, true)] cram: bool,
    ) {
        let dir = tempfile::TempDir::new().unwrap();
        let fasta = crate::testing::write_fasta(dir.path(), &[("chr1", &"A".repeat(100))]);
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "calls.vcf.gz", &header, &[]);
        let program = |cl: &str| format!("@PG\tID:mapper\tPN:{aligner}\tVN:1\tCL:{cl}");
        let bam = crate::testing::write_bam_with_header(
            dir.path(),
            "reads.bam",
            100,
            &[],
            &[&program(original)],
        );
        let bam = if cram { to_cram(&bam, &fasta) } else { bam };
        let expected = crate::alignment_provenance::HeaderProvenance::read(&bam).unwrap();
        let replacement = crate::testing::write_bam_with_header(
            dir.path(),
            "replacement.bam",
            100,
            &[],
            &[&program(replacement)],
        );
        let replacement = if cram { to_cram(&replacement, &fasta) } else { replacement };
        std::fs::rename(&replacement, &bam).unwrap();
        let index = if cram { "cram.crai" } else { "bam.bai" };
        std::fs::rename(replacement.with_extension(index), bam.with_extension(index)).unwrap();
        let tags = crate::alignment_provenance::AlignmentTags::parse(overrides).unwrap();
        let inputs = ReadInputs {
            observed_indexes: None,
            vcf: &vcf,
            alignments: &bam,
            reference: &fasta,
            expected_alignment: Some(AlignmentExpectation {
                provenance: expected.provenance.as_ref(),
                header_records: &expected.header_records,
                tags: tags.as_ref(),
                strict,
            }),
        };
        let mut readers = ShardReaders::default();
        let contigs = SharedContigs::default();
        let result = readers.open(&inputs, "chr1", 0, &contigs);
        assert_eq!(result.is_ok(), accepted, "unexpected reader result: {:?}", result.err());
    }

    #[rstest]
    #[case::contig("chr1", "chr1", None)]
    #[case::span("chr1:100-200", "chr1", Some((100, 200)))]
    #[case::commas("chr1:1,000-2,000", "chr1", Some((1000, 2000)))]
    #[case::colon_in_contig("HLA-A*01:01", "HLA-A*01:01", None)]
    #[case::colon_contig_with_span("HLA-A*01:01:5-10", "HLA-A*01:01", Some((5, 10)))]
    fn parses_regions(#[case] text: &str, #[case] chrom: &str, #[case] span: Option<(u64, u64)>) {
        let region = Region::parse(text, &["chr1", "HLA-A*01:01"]).unwrap();
        assert_eq!(region, Region { chrom: chrom.to_string(), span });
    }

    #[test]
    fn plans_shards_clamping_spans_to_the_contig() {
        let dir = tempfile::TempDir::new().unwrap();
        let reference = crate::testing::write_fasta(dir.path(), &[("chr1", &"A".repeat(1000))]);
        let inputs = ReadInputs {
            observed_indexes: None,
            vcf: Path::new("unused.vcf.gz"),
            alignments: Path::new("unused.bam"),
            reference: &reference,
            expected_alignment: None,
        };
        let summary = crate::vcf::VcfSummary {
            samples: vec!["S1".to_string()],
            contig_lengths: [("chr1".to_string(), Some(1000))].into_iter().collect(),
        };
        let shards = plan_shards(&inputs, &["chr1:900-5000"], &summary).unwrap();
        assert_eq!(shards, [("chr1".to_string(), 900, 1000)]);
        // A VCF header length past the reference's is one more span of the last shard, however
        // long.
        let long = crate::vcf::VcfSummary {
            contig_lengths: [("chr1".to_string(), Some(u64::MAX))].into_iter().collect(),
            ..summary.clone()
        };
        let shards = plan_shards(&inputs, &["chr1"], &long).unwrap();
        assert_eq!(shards, [("chr1".to_string(), 1, u64::MAX)]);
        let error = plan_shards(&inputs, &["chr1:5000-6000"], &summary).unwrap_err();
        assert_eq!(
            error.to_string(),
            "region chr1:5000-6000 starts past the end of contig chr1 (1000 bp)"
        );
    }

    #[test]
    fn reader_sets_share_a_contig_while_any_holds_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let path =
            crate::testing::write_fasta(dir.path(), &[("chr1", "ACGTACGTAC"), ("chr2", "GG")]);
        let mut fasta = crate::fasta::IndexedFasta::from_path(&path).unwrap();
        let contigs = SharedContigs::default();
        let first = contigs.get("chr1", &mut fasta).unwrap();
        let second = contigs.get("chr1", &mut fasta).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_slice(), b"ACGTACGTAC");
        assert_eq!(contigs.get("chr2", &mut fasta).unwrap().as_slice(), b"GG");
        // Freed when no set holds it, and read again when next wanted.
        let weak = Arc::downgrade(&first);
        drop((first, second));
        assert!(weak.upgrade().is_none());
        assert_eq!(contigs.get("chr1", &mut fasta).unwrap().as_slice(), b"ACGTACGTAC");
        assert!(contigs.get("chr3", &mut fasta).is_err());
    }

    #[rstest]
    #[case("chr1:0-10")]
    #[case("chr1:20-10")]
    #[case("chr1:10")]
    #[case(":1-2")]
    #[case("chr1:a-b")]
    fn rejects_bad_regions(#[case] text: &str) {
        let message = Region::parse(text, &["chr1"]).unwrap_err().to_string();
        assert!(message.starts_with("malformed region"), "{message}");
    }
}
