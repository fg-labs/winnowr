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
pub fn open_indexed<P: AsRef<Path>>(path: P) -> Result<IndexedReader> {
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
pub fn has_index(path: &Path) -> bool {
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

/// What the tabix index of a bgzipped VCF says about it (see [`tabix_scan`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabixScan {
    /// The contigs with records in the index, in file order.
    pub contigs: Vec<String>,
    /// The contigs scanned on which the VCF has records at POS 0 (a telomere), or `None` if that
    /// is not known (a record line at POS 0 or 1 is not UTF-8, or the index cannot be read).
    pub pos0: Option<Vec<String>>,
}

/// Opens a VCF's tabix index and finds, of the given contigs (every indexed contig if `None`),
/// those with records at POS 0. Returns `None` if the VCF is not bgzipped text with a tabix (or
/// CSI) index, e.g. a BCF, or rust-htslib cannot read its header lines as UTF-8.
///
/// An [`IndexedReader`] never returns records at POS 0 (htslib reads POS 0 as 0-based position
/// -1, before any region), so a reader that fetches by region must account for them. The tabix
/// index keeps them, at 0-based position 0. Reading them makes htslib warn "Coordinate <= 0
/// detected. Did you forget to use the -0 option?", which is expected here.
///
/// rust-htslib panics, rather than failing, on a header or record line that is not UTF-8; such a
/// panic is caught, without its message, and makes the answer unknown.
pub fn tabix_scan<S: AsRef<str>>(vcf: &Path, contigs: Option<&[S]>) -> Option<TabixScan> {
    use rust_htslib::tbx::Read as _;
    let mut magic = [0u8; 3];
    rust_htslib::bgzf::Reader::from_path(vcf).ok()?.read_exact(&mut magic).ok()?;
    if &magic == b"BCF" {
        return None;
    }
    let caught = |f: &mut dyn FnMut() -> Option<Vec<String>>| {
        quietly(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))).ok().flatten()
    };
    let mut tabix = None;
    let indexed = caught(&mut || {
        let reader = rust_htslib::tbx::Reader::from_path(vcf).ok()?;
        let names = reader.seqnames();
        tabix = Some(reader);
        Some(names)
    })?;
    let mut tabix = tabix?;
    let wanted: Vec<String> = match contigs {
        Some(contigs) => contigs.iter().map(|c| c.as_ref().to_string()).collect(),
        None => indexed.clone(),
    };
    let pos0 = caught(&mut || {
        let mut found = Vec::new();
        let mut line = Vec::new();
        for contig in &wanted {
            // A contig without records is not in the index.
            let Ok(tid) = tabix.tid(contig) else { continue };
            // The records overlapping the first base: those at POS 0 and 1.
            tabix.fetch(tid, 0, 1).ok()?;
            while tabix.read(&mut line).ok()? {
                if line.split(|&b| b == b'\t').nth(1).is_some_and(is_zero) {
                    found.push(contig.clone());
                    break;
                }
            }
        }
        Some(found)
    });
    Some(TabixScan { contigs: indexed, pos0 })
}

/// Whether a POS field is a number htslib reads as 0: zeros, after an optional `+`.
fn is_zero(pos: &[u8]) -> bool {
    let digits = pos.strip_prefix(b"+").unwrap_or(pos);
    !digits.is_empty() && digits.iter().all(|&b| b == b'0')
}

/// Fetches the records overlapping the 0-based, half-open span `[start, end)` of a contig (to
/// its end if `end` is `None`), returning whether the index has records on the contig.
///
/// # Errors
///
/// Returns an error if the contig is not in the header, or the fetch fails otherwise.
pub fn fetch_span(
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
pub fn missing_contigs<'a, S: AsRef<str>>(header: &HeaderView, contigs: &'a [S]) -> Vec<&'a str> {
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
pub fn check_contigs<S: AsRef<str>>(header: &HeaderView, contigs: &[S], path: &Path) -> Result<()> {
    let missing = missing_contigs(header, contigs);
    if !missing.is_empty() {
        bail!("contig(s) not in the header of {}: {}", path.display(), missing.join(", "));
    }
    Ok(())
}

/// True if the header declares the INFO field `tag`.
pub fn has_info(header: &HeaderView, tag: &[u8]) -> bool {
    header.info_type(tag).is_ok()
}

/// True if the header declares the FORMAT field `tag`.
pub fn has_format(header: &HeaderView, tag: &[u8]) -> bool {
    header.format_type(tag).is_ok()
}

/// True if the header declares `tag` as an INFO field (`is_info`) or else a FORMAT field.
pub fn has_field(header: &HeaderView, tag: &[u8], is_info: bool) -> bool {
    if is_info { has_info(header, tag) } else { has_format(header, tag) }
}

/// A field's name, as `INFO/MQ` or `FORMAT/AD`.
#[must_use]
pub fn field_name(tag: &str, is_info: bool) -> String {
    format!("{}/{tag}", if is_info { "INFO" } else { "FORMAT" })
}

/// The structured lines of a VCF header (see [`HeaderView::header_records`]).
///
/// rust-htslib panics on a header line that is not UTF-8 (e.g. a Latin-1 character in a
/// Description, which some VCF 4.2 files have); this returns an error instead.
///
/// # Errors
///
/// Returns an error, naming `path`, if a header line is not UTF-8.
pub fn header_records(
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
pub fn sample_names(header: &HeaderView) -> Vec<String> {
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
pub struct FieldReader {
    integers: Buffer,
    floats: Buffer,
    strings: Buffer,
    values: Vec<f64>,
}

impl FieldReader {
    pub fn new() -> Self {
        FieldReader {
            integers: Buffer::new(),
            floats: Buffer::new(),
            strings: Buffer::new(),
            values: Vec::new(),
        }
    }

    /// The first sample's genotype, normalized; missing when the record has no GT.
    pub fn genotype(&mut self, record: &Record) -> Genotype {
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
    pub fn info(&mut self, record: &Record, tag: &[u8]) -> Result<Option<&[f64]>> {
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
    pub fn info_value(&mut self, record: &Record, tag: &[u8], index: usize) -> Result<f64> {
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
    pub fn format(&mut self, record: &Record, tag: &[u8]) -> Result<Option<&[f64]>> {
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

/// The ids of the FILTER, INFO and FORMAT names a VCF header declares (they share htslib's ID
/// dictionary).
///
/// htslib reads a FILTER the header does not declare with a warning, adding it to its own copy
/// of the header. Through an [`IndexedReader`] that is not the copy a record's header refers to,
/// so naming the record's FILTERs (or translating the record to another header) reads past the
/// header's dictionary and crashes the process. Records are checked with
/// [`DeclaredIds::check`] right after they are read, before either. A FILTER named like a
/// declared INFO or FORMAT field has that field's id, which names it safely, so it is accepted,
/// as htslib accepts it.
///
/// An INFO or FORMAT tag named like a field declared only as another kind (e.g. INFO `LowQual`
/// with only `##FILTER=<ID=LowQual>`) has an id too, so it is formatted and translated as if
/// declared. A plain reader flags it, but a record read by region is not flagged: such records are
/// checked with [`DeclaredIds::check_tags`].
#[derive(Debug, Clone)]
pub struct DeclaredIds {
    /// The ids of the FILTER, INFO and FORMAT names.
    ids: std::collections::HashSet<u32>,
    /// The names declared, but not as INFO fields.
    not_info: Vec<Vec<u8>>,
    /// The names declared, but not as FORMAT fields.
    not_format: Vec<Vec<u8>>,
}

impl DeclaredIds {
    /// The ids declared in the header of the VCF at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if a header line is not UTF-8.
    pub fn new(header: &HeaderView, path: &Path) -> Result<Self> {
        let mut names: IndexMap<String, (bool, bool)> = IndexMap::new();
        for record in header_records(header, path)? {
            let (values, is_info, is_format) = match record {
                HeaderRecord::Filter { values, .. } => (values, false, false),
                HeaderRecord::Info { values, .. } => (values, true, false),
                HeaderRecord::Format { values, .. } => (values, false, true),
                _ => continue,
            };
            let Some(name) = values.get("ID") else { continue };
            let kinds = names.entry(name.clone()).or_default();
            kinds.0 |= is_info;
            kinds.1 |= is_format;
        }
        let ids = names
            .keys()
            .filter_map(|name| header.name_to_id(name.as_bytes()).ok().map(|id| *id))
            .collect();
        let not = |kind: fn(&(bool, bool)) -> bool| -> Vec<Vec<u8>> {
            names
                .iter()
                .filter(|(_, kinds)| !kind(kinds))
                .map(|(name, _)| name.as_bytes().to_vec())
                .collect()
        };
        let (not_info, not_format) = (not(|kinds| kinds.0), not(|kinds| kinds.1));
        Ok(DeclaredIds { ids, not_info, not_format })
    }

    /// Header lines that declare each name declared only as other kinds as an INFO and a FORMAT
    /// field, for [`DeclaredIds::check_tags`]. Added to a copy of the header, they keep every
    /// name's id.
    pub fn tag_probe_lines(&self) -> Vec<String> {
        let description = "Declared by winnowr to find tags declared only as other kinds";
        let info = self.not_info.iter().map(|name| {
            let name = String::from_utf8_lossy(name);
            format!("##INFO=<ID={name},Number=0,Type=Flag,Description=\"{description}\">")
        });
        let format = self.not_format.iter().map(|name| {
            let name = String::from_utf8_lossy(name);
            format!("##FORMAT=<ID={name},Number=.,Type=String,Description=\"{description}\">")
        });
        info.chain(format).collect()
    }

    /// Checks a record has no INFO tag named like a field not declared as an INFO field, and no
    /// FORMAT tag named like a field not declared as a FORMAT field. The record must have been
    /// translated to a copy of its header with the [`DeclaredIds::tag_probe_lines`] added.
    ///
    /// # Errors
    ///
    /// Returns an error naming the record's position and the tag if it has one.
    pub fn check_tags(&self, record: &Record) -> Result<()> {
        let info = self
            .not_info
            .iter()
            .find(|name| record.info(name).flag().is_ok_and(|present| present))
            .map(|name| ("INFO", name));
        let format = || {
            self.not_format
                .iter()
                .find(|name| {
                    !matches!(record.format(name).string(), Err(HtslibError::BcfMissingTag { .. }))
                })
                .map(|name| ("FORMAT", name))
        };
        if let Some((kind, name)) = info.or_else(format) {
            bail!(
                "the record at {} has an INFO or FORMAT tag not declared in the VCF header; \
                 declare it ({kind}/{} is declared only as another kind)",
                position(record),
                String::from_utf8_lossy(name)
            );
        }
        Ok(())
    }

    /// Checks a record (read with the header these were taken from) can be named and written:
    /// its FILTERs are declared, and htslib found nothing wrong with it (e.g. an undeclared INFO
    /// or FORMAT tag, which a plain reader flags and on which htslib exits the process when the
    /// record is translated to another header).
    ///
    /// # Errors
    ///
    /// Returns an error naming the record's position if either is not so.
    pub fn check(&self, record: &Record) -> Result<()> {
        if record.filters().any(|id| !self.ids.contains(&id)) {
            bail!(
                "a FILTER of the record at {} is not declared with a ##FILTER line in the VCF \
                 header; add one for it",
                position(record)
            );
        }
        let errcode = record.inner().errcode;
        if errcode != 0 {
            let what = if errcode & 2 != 0 {
                "has a FILTER, INFO or FORMAT tag not declared in the VCF header; declare it"
            } else if errcode & 1 != 0 {
                "is on a contig not declared in the VCF header; declare it"
            } else {
                "is malformed"
            };
            bail!("the record at {} {what} (htslib error code {errcode})", position(record));
        }
        Ok(())
    }
}

/// The record's FILTER names joined with `;`, or `PASS` when it has none.
///
/// The record's FILTERs must be declared in its header (see [`DeclaredIds`]).
pub fn filter_string(record: &Record) -> String {
    let header = record.header();
    let names: Vec<String> = record
        .filters()
        .map(|id| String::from_utf8_lossy(&header.id_to_name(id)).into_owned())
        .collect();
    if names.is_empty() { crate::common::PASS.to_string() } else { names.join(";") }
}

/// The record's QUAL, or NaN if missing.
pub fn qual(record: &Record) -> f64 {
    float_to_f64(record.qual())
}

/// The record's contig name and 1-based position, for messages.
pub fn position(record: &Record) -> String {
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
