//! Typed access to VCF records read with rust-htslib.
//!
//! INFO and FORMAT values are read as `f64` whatever their declared numeric type, with missing
//! values as NaN, so that callers need not care whether a field is an Integer or a Float.

use std::io::Read as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use flate2::read::MultiGzDecoder;
use indexmap::IndexMap;
use rust_htslib::bcf::header::{HeaderRecord, HeaderView, TagType};
use rust_htslib::bcf::record::{Buffer, Numeric, Record};
use rust_htslib::bcf::{IndexedReader, Read};
use rust_htslib::errors::Error as HtslibError;

use crate::alleles::Genotype;

/// The index file extensions htslib looks for next to a VCF or BCF.
const INDEX_EXTENSIONS: [&str; 2] = ["tbi", "csi"];

/// The magic bytes that start a (decompressed) tabix or CSI index.
const INDEX_MAGIC: [&[u8; 4]; 2] = [b"TBI\x01", b"CSI\x01"];

/// Opens a local, bgzipped, indexed VCF or BCF.
///
/// The index is found as htslib does: `<path>.tbi` or `<path>.csi`, or the same with the file's
/// extension replaced (e.g. `calls.csi` for `calls.bcf`). An explicit `<path>##idx##<index>` is
/// not supported, because rust-htslib requires the whole path to be an existing file.
///
/// rust-htslib's `IndexedReader` crashes, rather than failing, when htslib cannot open the file
/// or its index, so this first checks that the file opens as a VCF/BCF, is bgzipped, and has an
/// index that starts like one. A file that passes these checks but whose index is corrupt past
/// its first bytes can still crash the process.
///
/// # Errors
///
/// Returns an error if the path is remote, the file does not exist, is not a bgzipped VCF or BCF,
/// or has no readable index.
pub(crate) fn open_indexed<P: AsRef<Path>>(path: P) -> Result<IndexedReader> {
    let path = path.as_ref();
    let text = path.to_string_lossy();
    if text.contains("://") {
        bail!("{text} is remote; only local VCF/BCF files are supported");
    }
    if !path.is_file() {
        bail!("{} does not exist or is not a file", path.display());
    }
    rust_htslib::bcf::Reader::from_path(path)
        .with_context(|| format!("failed to open {} as a VCF or BCF", path.display()))?;
    let bgzipped = rust_htslib::bgzf::is_bgzip(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if !bgzipped {
        bail!("{} is not bgzipped: compress it with bgzip and index it", path.display());
    }
    let Some(index) = index_candidates(path).into_iter().find(|index| index.is_file()) else {
        bail!(
            "{} has no index ({}.tbi or .csi): it must be bgzipped and indexed (e.g. with \
             bcftools index)",
            path.display(),
            path.display()
        );
    };
    if !starts_like_an_index(&index) {
        bail!("{} is not a tabix or CSI index", index.display());
    }
    IndexedReader::from_path(path).with_context(|| format!("failed to open {}", path.display()))
}

/// True if a local, bgzipped VCF or BCF has a tabix or CSI index next to it, so it can be read
/// by region.
pub(crate) fn has_index(path: &Path) -> bool {
    rust_htslib::bgzf::is_bgzip(path).unwrap_or(false)
        && index_candidates(path)
            .into_iter()
            .any(|index| index.is_file() && starts_like_an_index(&index))
}

/// The index paths htslib tries for a data file, in order.
fn index_candidates(data: &Path) -> Vec<PathBuf> {
    let appended = INDEX_EXTENSIONS.iter().map(|extension| {
        let mut index = data.as_os_str().to_owned();
        index.push(".");
        index.push(extension);
        PathBuf::from(index)
    });
    let replaced = INDEX_EXTENSIONS.iter().map(|extension| data.with_extension(extension));
    appended.chain(replaced).collect()
}

/// True if a file decompresses to start with a tabix or CSI index's magic bytes.
fn starts_like_an_index(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|file| MultiGzDecoder::new(file).read_exact(&mut magic))
        .is_ok_and(|()| INDEX_MAGIC.iter().any(|expected| **expected == magic))
}

/// Fetches the records overlapping the 0-based, half-open span `[start, end)` of a contig (to
/// its end if `end` is `None`), returning whether the index has records on the contig.
///
/// # Errors
///
/// Returns an error if the contig is not in the header, or the fetch fails otherwise.
pub(crate) fn fetch_span(
    reader: &mut IndexedReader,
    contig: &str,
    start: u64,
    end: Option<u64>,
    path: &Path,
) -> Result<bool> {
    let rid = reader
        .header()
        .name2rid(contig.as_bytes())
        .with_context(|| format!("contig {contig} is not in the header of {}", path.display()))?;
    match reader.fetch(rid, start, end.map(|end| end.saturating_sub(1))) {
        Ok(()) => Ok(true),
        Err(HtslibError::GenomicSeek { .. }) => Ok(false),
        Err(e) => {
            Err(e).with_context(|| format!("failed to fetch {contig} from {}", path.display()))
        }
    }
}

/// The contigs that a header does not declare, in the given order.
pub(crate) fn missing_contigs<'a, S: AsRef<str>>(
    header: &HeaderView,
    contigs: &'a [S],
) -> Vec<&'a str> {
    contigs
        .iter()
        .map(AsRef::as_ref)
        .filter(|contig| header.name2rid(contig.as_bytes()).is_err())
        .collect()
}

/// Checks that a header declares every one of `contigs`.
///
/// # Errors
///
/// Returns an error naming the undeclared contigs (e.g. from `1` vs `chr1` naming).
pub(crate) fn check_contigs<S: AsRef<str>>(
    header: &HeaderView,
    contigs: &[S],
    path: &Path,
) -> Result<()> {
    let missing = missing_contigs(header, contigs);
    if !missing.is_empty() {
        bail!("contig(s) not in the header of {}: {}", path.display(), missing.join(", "));
    }
    Ok(())
}

/// True if the header declares the INFO field `tag`.
pub(crate) fn has_info(header: &HeaderView, tag: &[u8]) -> bool {
    header.info_type(tag).is_ok()
}

/// True if the header declares the FORMAT field `tag`.
pub(crate) fn has_format(header: &HeaderView, tag: &[u8]) -> bool {
    header.format_type(tag).is_ok()
}

/// The structured lines of a VCF header (see [`HeaderView::header_records`]).
///
/// rust-htslib panics on a header line that is not UTF-8 (e.g. a Latin-1 character in a
/// Description, which some VCF 4.2 files have); this returns an error instead.
///
/// # Errors
///
/// Returns an error, naming `path`, if a header line is not UTF-8.
pub(crate) fn header_records(
    header: &HeaderView,
    path: &Path,
) -> Result<Vec<rust_htslib::bcf::header::HeaderRecord>> {
    let records = quietly(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| header.header_records()))
    });
    records.map_err(|_| {
        anyhow::anyhow!(
            "the header of {} has a line that is not UTF-8 (e.g. a Latin-1 character in a \
                 Description): convert it to UTF-8 (e.g. with iconv and bcftools reheader)",
            path.display()
        )
    })
}

thread_local! {
    /// Whether the panic hook stays quiet for this thread's panics (see [`quietly`]).
    static QUIET_PANICS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` without printing the message of a panic it raises, which the caller catches and
/// reports as an error. Other threads' panics print as usual.
fn quietly<T>(f: impl FnOnce() -> T) -> T {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !QUIET_PANICS.with(std::cell::Cell::get) {
                default(info);
            }
        }));
    });
    let previous = QUIET_PANICS.with(|quiet| quiet.replace(true));
    let result = f();
    QUIET_PANICS.with(|quiet| quiet.set(previous));
    result
}

/// The name of each sample in a header.
pub(crate) fn sample_names(header: &HeaderView) -> Vec<String> {
    header.samples().iter().map(|name| String::from_utf8_lossy(name).into_owned()).collect()
}

/// What winnowr needs from a VCF header besides its records: its samples and contig lengths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VcfSummary {
    /// The sample names.
    pub samples: Vec<String>,
    /// The declared contigs' lengths, in header order (`None` where the header gives none).
    pub contig_lengths: IndexMap<String, Option<u64>>,
}

impl VcfSummary {
    /// Reads the summary of a VCF's header.
    ///
    /// # Errors
    ///
    /// Returns an error if the VCF cannot be opened, or a header line is not UTF-8.
    pub fn read<P: AsRef<Path>>(vcf: P) -> Result<Self> {
        let reader = open_indexed(&vcf)?;
        let header = reader.header();
        let samples =
            header.samples().iter().map(|s| String::from_utf8_lossy(s).into_owned()).collect();
        let mut contig_lengths = IndexMap::new();
        for record in header_records(header, vcf.as_ref())? {
            if let HeaderRecord::Contig { values, .. } = record
                && let Some(id) = values.get("ID")
            {
                let length = values.get("length").and_then(|length| length.parse().ok());
                contig_lengths.insert(id.clone(), length);
            }
        }
        Ok(VcfSummary { samples, contig_lengths })
    }

    /// The name of the first sample, or the VCF's file name if it has none.
    #[must_use]
    pub fn sample_name(&self, vcf: &Path) -> String {
        match self.samples.first() {
            Some(sample) => sample.clone(),
            None => vcf.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
        }
    }

    /// The lengths of `contigs`, in that order (`None` where the header gives none).
    #[must_use]
    pub fn lengths_of<S: AsRef<str>>(&self, contigs: &[S]) -> IndexMap<String, Option<u64>> {
        contigs
            .iter()
            .map(|contig| {
                let contig = contig.as_ref();
                (contig.to_string(), self.contig_lengths.get(contig).copied().flatten())
            })
            .collect()
    }
}

/// Reads INFO and FORMAT values of records as `f64`, reusing its buffers between calls.
///
/// There is one htslib buffer per value type: htslib records a buffer's capacity in elements of
/// the type last read into it, so reading integers into a buffer last used for (byte) strings
/// would overflow it.
#[derive(Debug)]
pub(crate) struct FieldReader {
    integers: Buffer,
    floats: Buffer,
    strings: Buffer,
    values: Vec<f64>,
}

impl FieldReader {
    pub(crate) fn new() -> Self {
        FieldReader {
            integers: Buffer::new(),
            floats: Buffer::new(),
            strings: Buffer::new(),
            values: Vec::new(),
        }
    }

    /// The first sample's genotype, normalized; missing when the record has no GT.
    pub(crate) fn genotype(&mut self, record: &Record) -> Genotype {
        // GT is decoded as integers, so it shares the integer buffer.
        let n_alleles = record.allele_count() as usize;
        match record.genotypes_shared_buffer(&mut self.integers) {
            Ok(genotypes) => Genotype::from_alleles(
                genotypes.get(0).iter().map(|allele| allele.index()),
                n_alleles,
            ),
            Err(_) => Genotype::MISSING,
        }
    }

    /// The buffer for values of `tag_type`.
    fn buffer(&mut self, tag_type: TagType) -> &mut Buffer {
        match tag_type {
            TagType::Integer => &mut self.integers,
            TagType::Float => &mut self.floats,
            TagType::String | TagType::Flag => &mut self.strings,
        }
    }

    /// The values of an INFO field, or `None` if the header does not declare it, it is a flag, or
    /// the record does not set it.
    ///
    /// Integer and Float values are converted, with missing values as NaN. String values are split
    /// on `,` and `|`, and each part parsed as a float (NaN for `.` or anything unparseable).
    ///
    /// # Errors
    ///
    /// Returns an error if htslib cannot decode the value.
    pub(crate) fn info(&mut self, record: &Record, tag: &[u8]) -> Result<Option<&[f64]>> {
        let Ok((tag_type, _)) = record.header().info_type(tag) else {
            return Ok(None);
        };
        let mut values = std::mem::take(&mut self.values);
        values.clear();
        let info = record.info_shared_buffer(tag, self.buffer(tag_type));
        let found = match tag_type {
            TagType::Integer => info.integer().map(|found| {
                found.map(|found| values.extend(found.iter().map(|&v| int_to_f64(v))))
            }),
            TagType::Float => info.float().map(|found| {
                found.map(|found| values.extend(found.iter().map(|&v| float_to_f64(v))))
            }),
            TagType::String => info.string().map(|found| {
                found.map(|found| {
                    values.extend(found.iter().flat_map(|v| v.split(|&b| b == b'|')).map(parse_f64))
                })
            }),
            TagType::Flag => Ok(None),
        };
        self.values = values;
        let found = found.with_context(|| {
            format!("failed to read INFO/{} at {}", lossy(tag), position(record))
        })?;
        Ok(found.map(|()| self.values.as_slice()))
    }

    /// The `index`-th value of an INFO field, or NaN if it is absent (see [`FieldReader::info`]).
    ///
    /// # Errors
    ///
    /// Returns an error if htslib cannot decode the value.
    pub(crate) fn info_value(&mut self, record: &Record, tag: &[u8], index: usize) -> Result<f64> {
        Ok(self
            .info(record, tag)?
            .and_then(|values| values.get(index).copied())
            .unwrap_or(f64::NAN))
    }

    /// The values of a numeric FORMAT field for the first sample, or `None` if the header does not
    /// declare it as Integer or Float, or the record does not have it. Missing values are NaN.
    ///
    /// # Errors
    ///
    /// Returns an error if htslib cannot decode the value.
    pub(crate) fn format(&mut self, record: &Record, tag: &[u8]) -> Result<Option<&[f64]>> {
        let Ok((tag_type, _)) = record.header().format_type(tag) else {
            return Ok(None);
        };
        if matches!(tag_type, TagType::String | TagType::Flag) {
            return Ok(None);
        }
        let mut values = std::mem::take(&mut self.values);
        values.clear();
        let format = record.format_shared_buffer(tag, self.buffer(tag_type));
        let read = match tag_type {
            TagType::Integer => format.integer().map(|samples| {
                if let Some(found) = samples.first() {
                    values.extend(found.iter().map(|&v| int_to_f64(v)));
                }
            }),
            _ => format.float().map(|samples| {
                if let Some(found) = samples.first() {
                    values.extend(found.iter().map(|&v| float_to_f64(v)));
                }
            }),
        };
        self.values = values;
        match read {
            Ok(()) => Ok(Some(self.values.as_slice())),
            // htslib reports a FORMAT field the record does not have as an error.
            Err(HtslibError::BcfMissingTag { .. }) => Ok(None),
            Err(e) => Err(e).with_context(|| {
                format!("failed to read FORMAT/{} at {}", lossy(tag), position(record))
            }),
        }
    }
}

/// The record's FILTER names joined with `;`, or `PASS` when it has none.
pub(crate) fn filter_string(record: &Record) -> String {
    let header = record.header();
    let names: Vec<String> = record
        .filters()
        .map(|id| String::from_utf8_lossy(&header.id_to_name(id)).into_owned())
        .collect();
    if names.is_empty() { crate::common::PASS.to_string() } else { names.join(";") }
}

/// The record's QUAL, or NaN if missing.
pub(crate) fn qual(record: &Record) -> f64 {
    float_to_f64(record.qual())
}

/// The record's contig name and 1-based position, for messages.
pub(crate) fn position(record: &Record) -> String {
    let contig = record
        .rid()
        .and_then(|rid| record.header().rid2name(rid).ok())
        .map_or_else(|| "?".to_string(), |name| String::from_utf8_lossy(name).into_owned());
    format!("{contig}:{}", record.pos() + 1)
}

fn int_to_f64(value: i32) -> f64 {
    if value.is_missing() { f64::NAN } else { f64::from(value) }
}

fn float_to_f64(value: f32) -> f64 {
    if value.is_missing() { f64::NAN } else { f64::from(value) }
}

/// Parses a VCF string value as a float: NaN for `.` or anything unparseable.
fn parse_f64(value: &[u8]) -> f64 {
    std::str::from_utf8(value).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(f64::NAN)
}

fn lossy(tag: &[u8]) -> String {
    String::from_utf8_lossy(tag).into_owned()
}
