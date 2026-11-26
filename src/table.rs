//! Per-sample feature tables: columnar (Arrow) in memory, Parquet on disk, with winnowr metadata
//! in the schema.
//!
//! Columns have fixed names and types: strings (the keys `chrom`, `ref`, `alt`, `filter`,
//! `variant_type` and `gt`, and the repeat-length evidence's `sl_unit` and `sl_histogram`), `pos`
//! as Int64, flags as Boolean, and every other column as Float64 with NaN for missing values.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use arrow_array::builder::StringBuilder;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch,
    RecordBatchOptions, StringArray,
};
use arrow_schema::{Field, Schema};
use arrow_select::filter::filter_record_batch;
use indexmap::IndexMap;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::arrow_writer::{ArrowColumnChunk, compute_leaves};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::FOOTER_SIZE;
use parquet::file::metadata::{FooterTail, ParquetMetaDataReader};
use parquet::file::properties::{DEFAULT_MAX_ROW_GROUP_ROW_COUNT, WriterProperties};
use parquet::file::reader::ChunkReader as _;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::alignment_provenance::AlignmentProvenance;
use crate::common::VariantType;
use crate::feature_source::FeatureSource;
use crate::features::{DerivedFeatures, DpNormDivisor, region_feature};
use crate::neighbourhood::NEIGHBOURHOOD_FEATURES;
use crate::pileup::PileupOptions;
use crate::raw_records::RawRecord;
use crate::strlen::StrlenOptions;

/// A raw numeric column: its name, and its value in a record.
type RawNumber = (&'static str, fn(&RawRecord) -> f64);

/// The schema metadata key holding the [`FeatureTableInfo`] JSON.
const METADATA_KEY: &str = "winnowr";

/// One opened Parquet file shared by decoders without sharing a seek position. Cloning a `File`
/// alone shares its offset, so Parquet's seek-based `File` implementation cannot be used here.
#[derive(Clone)]
struct PositionedFile {
    file: Arc<File>,
    length: u64,
    /// The modification time when opened, where the platform records one.
    modified: Option<std::time::SystemTime>,
}

impl PositionedFile {
    fn new(file: File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        Ok(Self {
            file: Arc::new(file),
            length: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }

    /// Whether the opened file's current length and modification time still are those it had
    /// when opened: a cheap indication of a change (a coarse clock can miss one, so the footer is
    /// compared too, see [`FeatureTable::read_parquet_selected`]).
    fn unchanged(&self) -> io::Result<bool> {
        let metadata = self.file.metadata()?;
        Ok(metadata.len() == self.length && metadata.modified().ok() == self.modified)
    }
}

/// A sequential cursor using positioned reads from the retained file, independent of other cursors.
struct PositionedRead {
    file: Arc<File>,
    offset: u64,
}

impl Read for PositionedRead {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.file.read_at(buffer, self.offset)?;
        self.offset = self.offset.checked_add(read as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Parquet read offset overflow")
        })?;
        Ok(read)
    }
}

impl parquet::file::reader::Length for PositionedFile {
    fn len(&self) -> u64 {
        self.length
    }
}

impl parquet::file::reader::ChunkReader for PositionedFile {
    type T = BufReader<PositionedRead>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        if start > self.length {
            return Err(
                io::Error::new(io::ErrorKind::UnexpectedEof, "Parquet offset past EOF").into()
            );
        }
        Ok(BufReader::new(PositionedRead { file: Arc::clone(&self.file), offset: start }))
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<bytes::Bytes> {
        let end = start.checked_add(length as u64).filter(|end| *end <= self.length);
        if end.is_none() {
            return Err(
                io::Error::new(io::ErrorKind::UnexpectedEof, "Parquet range past EOF").into()
            );
        }
        let mut buffer = vec![0; length];
        let mut reader = PositionedRead { file: Arc::clone(&self.file), offset: start };
        reader.read_exact(&mut buffer)?;
        Ok(buffer.into())
    }
}

/// Metadata stored alongside a feature table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// Default only for tests: a table's metadata always names its sample.
#[cfg_attr(test, derive(Default))]
pub struct FeatureTableInfo {
    /// The sample name.
    pub sample: String,
    /// The contigs features were extracted from.
    pub contigs: Vec<String>,
    /// The names of the region sets flagged (`region_<name>` columns).
    pub region_names: Vec<String>,
    /// Truth FN counts per variant type (`snp`, `indel`), when labelled from hap.py.
    pub fn_counts: Option<IndexMap<String, u64>>,
    /// Whether non-PASS calls carry promote labels (`promote_label`).
    pub promote_labelled: bool,
    /// Whether indel calls carry repeat-length evidence.
    #[serde(default)]
    pub strlen: bool,
    /// The read settings the repeat-length evidence was measured with (`None` without evidence,
    /// or for tables that record none: see
    /// [`FeatureTableInfo::strlen_options`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strlen_settings: Option<StrlenOptions>,
    /// The settings the allele read counts by MAPQ (`pu_*` features) were measured with
    /// (`None` without pileup evidence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pileup_settings: Option<PileupOptions>,
    /// The lengths of `contigs` in the VCF header (`None` where the header gives none).
    #[serde(default)]
    pub reference_contigs: Option<IndexMap<String, Option<u64>>>,
    /// The depth `dp_norm` divided each call's depth by, and whether it was given
    /// (`--dp-median`) or is the calls' median, for provenance only: nothing checks it. `None`
    /// for tables not built from calls, and for tables written before it was recorded (unknown).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_norm_divisor: Option<DpNormDivisor>,
    /// The part of the genome the calls and truth FN counts are of, when `winnowr subset`
    /// restricted them (see [`crate::subset::restrict_table`]); `None` for a table as extracted,
    /// whose calls and FNs are of all of `contigs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restricted_to: Option<Restriction>,
    /// Where the features of gVCF-mode annotations (`mq`, `dp_norm`, `qd`, `as_alt_frac`, `sor`,
    /// `n_alt`) come from. Not written for the gVCF source, so that its tables are as before, and
    /// tables that record none are of the gVCF source.
    #[serde(default, skip_serializing_if = "FeatureSource::is_default")]
    pub feature_source: FeatureSource,
    /// How the sample's reads were aligned (see [`crate::alignment_provenance`]): from its
    /// evidence tables, its reads' header or `--alignment-tags`. `None` when unknown, not written
    /// then, so that such tables are as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<AlignmentProvenance>,
}

/// The part of the genome a restricted table's calls and truth FN counts are of.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Restriction {
    /// The contigs kept.
    pub contigs: Vec<String>,
    /// The BED whose regions were left out, if any (as given).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excluded: Option<String>,
}

impl FeatureTableInfo {
    /// The read settings of the table's repeat-length evidence, if it has any: those recorded, or
    /// [`StrlenOptions::LEGACY`] for tables that record none.
    #[must_use]
    pub fn strlen_options(&self) -> Option<StrlenOptions> {
        self.strlen.then(|| self.strlen_settings.unwrap_or(StrlenOptions::LEGACY))
    }

    /// True if the table carries hap.py labels.
    #[must_use]
    pub fn is_labelled(&self) -> bool {
        self.fn_counts.is_some()
    }

    /// The truth FN count for a variant type.
    ///
    /// # Errors
    ///
    /// Returns an error if the table is not labelled, or has no count for the variant type.
    pub fn fn_count(&self, variant_type: VariantType) -> Result<u64> {
        let Some(counts) = &self.fn_counts else {
            bail!("the feature table for {} has no hap.py labels", self.sample);
        };
        counts.get(variant_type.as_str()).copied().with_context(|| {
            format!("the feature table for {} has no {variant_type} FN count", self.sample)
        })
    }
}

/// A per-sample feature table: one row per call, with its metadata.
#[derive(Debug, Clone)]
pub struct FeatureTable {
    batch: RecordBatch,
    info: FeatureTableInfo,
}

impl FeatureTable {
    /// Builds a table from raw records and their derived features, with the raw columns first,
    /// then the derived ones, in a fixed column order.
    ///
    /// # Errors
    ///
    /// Returns an error if a derived column does not have one value per record.
    pub fn from_records(
        records: &[RawRecord],
        derived: DerivedFeatures,
        info: FeatureTableInfo,
    ) -> Result<Self> {
        type Column<'a> = (&'static str, Box<dyn Fn() -> ArrayRef + Send + Sync + 'a>);
        let number = |name: &'static str, f: fn(&RawRecord) -> f64| -> Column<'_> {
            (
                name,
                Box::new(move || Arc::new(Float64Array::from_iter_values(records.iter().map(f)))),
            )
        };
        let mut raw: Vec<Column<'_>> = vec![
            ("chrom", Box::new(|| text(records, |r, out| out.push_str(&r.chrom)))),
            (
                "pos",
                Box::new(|| {
                    Arc::new(Int64Array::from_iter_values(records.iter().map(|r| r.pos as i64)))
                }),
            ),
            ("ref", Box::new(|| text(records, |r, out| push_allele(out, &r.reference)))),
            (
                "alt",
                Box::new(|| {
                    text(records, |r, out| {
                        for (i, alt) in r.alts.iter().enumerate() {
                            if i > 0 {
                                out.push(',');
                            }
                            push_allele(out, alt);
                        }
                    })
                }),
            ),
            ("filter", Box::new(|| text(records, |r, out| out.push_str(&r.filter)))),
            (
                "variant_type",
                Box::new(|| text(records, |r, out| out.push_str(r.variant_type.as_str()))),
            ),
            (
                "gt",
                Box::new(|| {
                    text(records, |r, out| {
                        let _ = write!(out, "{}", r.genotype);
                    })
                }),
            ),
        ];
        let raw_numbers: [RawNumber; 26] = [
            ("qual", |r| r.qual),
            ("dp", |r| r.dp),
            ("mq_rank_sum", |r| r.mq_rank_sum),
            ("read_pos_rank_sum", |r| r.read_pos_rank_sum),
            ("base_q_rank_sum", |r| r.base_q_rank_sum),
            ("str_period", |r| r.str_period),
            ("str_repeats", |r| r.str_repeats),
            ("gq", |r| r.gq),
            ("ad_ref", |r| r.ad_ref),
            ("ad_alt", |r| r.ad_alt),
            ("sb_ref_fwd", |r| r.sb_ref_fwd),
            ("sb_ref_rev", |r| r.sb_ref_rev),
            ("sb_alt_fwd", |r| r.sb_alt_fwd),
            ("sb_alt_rev", |r| r.sb_alt_rev),
            ("excess_het", |r| r.excess_het),
            ("pl_ref", |r| r.pl_ref),
            ("pl_het", |r| r.pl_het),
            ("pl_hom", |r| r.pl_hom),
            ("ad_alt_total", |r| r.ad_alt_total),
            ("raw_mq", |r| r.raw.raw_mq),
            ("raw_dp", |r| r.raw.raw_dp),
            ("qual_approx", |r| r.raw.qual_approx),
            ("var_dp", |r| r.raw.var_dp),
            ("as_mq_ref", |r| r.raw.as_mq_ref),
            ("as_mq_alt", |r| r.raw.as_mq_alt),
            ("n_alt", |r| r.raw.n_alt),
        ];
        raw.extend(raw_numbers.into_iter().map(|(name, value)| number(name, value)));
        // The raw columns are independent: build them in parallel, keeping their order.
        let arrays: Vec<(&str, ArrayRef)> =
            raw.par_iter().map(|(name, build)| (*name, build())).collect();
        let mut columns = ColumnsBuilder::default();
        for (name, array) in arrays {
            columns.push(name, array);
        }

        let floats = |values: Vec<f64>| -> ArrayRef { Arc::new(Float64Array::from(values)) };
        let flags = |values: Vec<bool>| -> ArrayRef { Arc::new(BooleanArray::from(values)) };
        let DerivedFeatures {
            ab,
            sor,
            het,
            flippable,
            mq,
            qd,
            as_alt_frac,
            dp_norm,
            indel_len,
            indel_abs_len,
            regions,
            n_alt_rec,
            overlaps_pass,
            neighbourhood,
            dp_norm_divisor: _, // recorded in the metadata by the caller
        } = derived;
        columns.push("ab", floats(ab));
        columns.push("sor", floats(sor));
        columns.push("het", floats(het));
        columns.push("flippable", flags(flippable));
        columns.push("mq", floats(mq));
        columns.push("qd", floats(qd));
        columns.push("as_alt_frac", floats(as_alt_frac));
        columns.push("dp_norm", floats(dp_norm));
        columns.push("indel_len", floats(indel_len));
        columns.push("indel_abs_len", floats(indel_abs_len));
        for (name, values) in regions {
            columns.push(&region_feature(&name), floats(values));
        }
        columns.push("n_alt_rec", floats(n_alt_rec));
        columns.push("overlaps_pass", flags(overlaps_pass));
        for (k, name) in NEIGHBOURHOOD_FEATURES.iter().enumerate() {
            columns.push(name, floats(neighbourhood.iter().map(|row| row[k]).collect()));
        }
        Ok(FeatureTable { batch: columns.finish(records.len())?, info })
    }

    /// A table from columns and metadata.
    #[must_use]
    #[cfg(test)]
    pub fn new(batch: RecordBatch, info: FeatureTableInfo) -> Self {
        FeatureTable { batch, info }
    }

    /// The table's metadata, for updating.
    pub fn info_mut(&mut self) -> &mut FeatureTableInfo {
        &mut self.info
    }

    /// The rows where `mask` is true, in order.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask` does not have one value per row.
    pub fn filter(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.num_rows() {
            bail!("a mask of {} values cannot select from {} rows", mask.len(), self.num_rows());
        }
        let batch = filter_record_batch(&self.batch, &BooleanArray::from(mask.to_vec()))?;
        Ok(FeatureTable { batch, info: self.info.clone() })
    }

    /// Each call's row among `rows`, joined by `(chrom, pos, ref, alt)` (`key` gives a row's), or
    /// `None` if it has none (of rows sharing a key, the last).
    ///
    /// # Errors
    ///
    /// Returns an error if the table lacks `chrom`, `pos`, `ref` or `alt`.
    pub fn join_calls<'r, R: Sync>(
        &self,
        rows: &'r [R],
        key: impl Fn(&'r R) -> (&'r str, u64, &'r str, &'r str),
    ) -> Result<Vec<Option<&'r R>>> {
        let by_key: HashMap<(&str, u64, &str, &str), &R> =
            rows.iter().map(|row| (key(row), row)).collect();
        let (chroms, positions) = (self.str_column("chrom")?, self.i64_column("pos")?);
        let (refs, alts) = (self.str_column("ref")?, self.str_column("alt")?);
        Ok((0..self.num_rows())
            .into_par_iter()
            .map(|i| {
                let pos = u64::try_from(positions[i]).ok()?;
                by_key.get(&(chroms[i], pos, refs[i], alts[i])).copied()
            })
            .collect())
    }

    /// True if the table has a column named `name`.
    #[must_use]
    pub fn has_column(&self, name: &str) -> bool {
        self.batch.column_by_name(name).is_some()
    }

    /// The table with a column set: replaced in place if the table has it, else appended.
    ///
    /// # Errors
    ///
    /// Returns an error if the column does not have one value per row.
    pub fn with_column(self, name: &str, array: ArrayRef) -> Result<Self> {
        if array.len() != self.num_rows() {
            bail!("column {name} has {} values for {} rows", array.len(), self.num_rows());
        }
        let schema = self.batch.schema();
        let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut arrays = self.batch.columns().to_vec();
        let field = Field::new(name, array.data_type().clone(), NULLABLE);
        match fields.iter().position(|existing| existing.name() == name) {
            Some(index) => {
                fields[index] = field;
                arrays[index] = array;
            }
            None => {
                fields.push(field);
                arrays.push(array);
            }
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        Ok(FeatureTable { batch, info: self.info })
    }

    /// The table with a Float64 column set (see [`FeatureTable::with_column`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the column does not have one value per row.
    pub fn with_f64_column(self, name: &str, values: Vec<f64>) -> Result<Self> {
        self.with_column(name, Arc::new(Float64Array::from(values)))
    }

    /// The table with a string column set (see [`FeatureTable::with_column`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the column does not have one value per row.
    #[cfg(test)]
    pub fn with_str_column<S: AsRef<str>>(self, name: &str, values: &[S]) -> Result<Self> {
        let array = StringArray::from_iter_values(values.iter().map(AsRef::as_ref));
        self.with_column(name, Arc::new(array))
    }

    /// The table with a Boolean column set (see [`FeatureTable::with_column`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the column does not have one value per row.
    pub fn with_bool_column(self, name: &str, values: Vec<bool>) -> Result<Self> {
        self.with_column(name, Arc::new(BooleanArray::from(values)))
    }

    /// The table's metadata.
    #[must_use]
    pub fn info(&self) -> &FeatureTableInfo {
        &self.info
    }

    /// The table's columns.
    #[must_use]
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// The number of rows (calls).
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// The column names, in order.
    #[must_use]
    #[cfg(test)]
    pub fn column_names(&self) -> Vec<&str> {
        self.batch.schema_ref().fields().iter().map(|field| field.name().as_str()).collect()
    }

    /// A column, or an error naming it if the table does not have it.
    fn column(&self, name: &str) -> Result<&ArrayRef> {
        self.batch.column_by_name(name).with_context(|| {
            format!("the feature table for {} has no column {name}", self.info.sample)
        })
    }

    /// The values of a Float64 column, with nulls (as other Parquet writers may use for missing
    /// values) read as NaN.
    ///
    /// # Errors
    ///
    /// Returns an error if the table has no such column, or it is not Float64.
    pub fn f64_column(&self, name: &str) -> Result<Cow<'_, [f64]>> {
        let column = self.column(name)?;
        let values = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .with_context(|| format!("column {name} is {}, not Float64", column.data_type()))?;
        if values.null_count() == 0 {
            return Ok(Cow::Borrowed(values.values()));
        }
        Ok(Cow::Owned(values.iter().map(|value| value.unwrap_or(f64::NAN)).collect()))
    }

    /// The values of an Int64 column.
    ///
    /// # Errors
    ///
    /// Returns an error if the table has no such column, it is not Int64, or it has nulls.
    pub fn i64_column(&self, name: &str) -> Result<&[i64]> {
        let column = self.column(name)?;
        let values = column
            .as_any()
            .downcast_ref::<Int64Array>()
            .with_context(|| format!("column {name} is {}, not Int64", column.data_type()))?;
        if values.null_count() > 0 {
            bail!("column {name} has {} missing values", values.null_count());
        }
        Ok(values.values())
    }

    /// The values of a Boolean column (a null is false).
    ///
    /// # Errors
    ///
    /// Returns an error if the table has no such column, or it is not Boolean.
    pub fn bool_column(&self, name: &str) -> Result<Vec<bool>> {
        let column = self.column(name)?;
        let values = column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .with_context(|| format!("column {name} is {}, not Boolean", column.data_type()))?;
        Ok(values.iter().map(Option::unwrap_or_default).collect())
    }

    /// The values of a string column (a null is the empty string).
    ///
    /// # Errors
    ///
    /// Returns an error if the table has no such column, or it is not a string column.
    pub fn str_column(&self, name: &str) -> Result<Vec<&str>> {
        let column = self.column(name)?;
        if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
            return Ok(values.iter().map(Option::unwrap_or_default).collect());
        }
        if let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() {
            return Ok(values.iter().map(Option::unwrap_or_default).collect());
        }
        bail!("column {name} is {}, not a string column", column.data_type())
    }

    /// Writes the table to a Parquet file (zstd-compressed), with its metadata in the schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_parquet<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let metadata = serde_json::to_string(&self.info).context("failed to encode metadata")?;
        let schema = self
            .batch
            .schema_ref()
            .as_ref()
            .clone()
            .with_metadata(HashMap::from([(METADATA_KEY.to_string(), metadata)]));
        let batch = self.batch.clone().with_schema(Arc::new(schema))?;
        let file =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        let properties = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();
        let failed = || format!("failed to write {}", path.display());
        // Column chunks are encoded in parallel (every column of every row group), then appended
        // in order, as ArrowWriter would write them on one thread.
        let (mut writer, factory) = ArrowWriter::try_new(file, batch.schema(), Some(properties))?
            .into_serialized_writer()?;
        let rows = batch.num_rows();
        let starts: Vec<usize> = (0..rows).step_by(DEFAULT_MAX_ROW_GROUP_ROW_COUNT).collect();
        let schema = batch.schema();
        let row_groups: Vec<Vec<ArrowColumnChunk>> = starts
            .par_iter()
            .enumerate()
            .map(|(index, &start)| -> Result<Vec<ArrowColumnChunk>> {
                let slice = batch.slice(start, DEFAULT_MAX_ROW_GROUP_ROW_COUNT.min(rows - start));
                let mut leaves = Vec::new();
                for (field, column) in schema.fields().iter().zip(slice.columns()) {
                    leaves.extend(compute_leaves(field, column)?);
                }
                let writers = factory.create_column_writers(index)?;
                writers
                    .into_par_iter()
                    .zip(leaves)
                    .map(|(mut column_writer, leaf)| {
                        column_writer.write(&leaf)?;
                        Ok(column_writer.close()?)
                    })
                    .collect()
            })
            .collect::<Result<_>>()
            .with_context(failed)?;
        for chunks in row_groups {
            let mut row_group = writer.next_row_group().with_context(failed)?;
            for chunk in chunks {
                chunk.append_to_row_group(&mut row_group).with_context(failed)?;
            }
            row_group.close().with_context(failed)?;
        }
        writer.close().with_context(failed)?;
        Ok(())
    }

    /// Reads a table written by [`FeatureTable::write_parquet`], or by another Parquet writer
    /// with the same columns and metadata, uncompressed or compressed with Snappy or Zstandard.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, or has no winnowr metadata.
    pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::read_parquet_selected(path, &TableSelection::default())
    }

    /// Reads only a table's metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, or has no winnowr metadata.
    pub fn read_info<P: AsRef<Path>>(path: P) -> Result<FeatureTableInfo> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("{} is not a Parquet file", path.display()))?;
        let Some(metadata) = builder.schema().metadata().get(METADATA_KEY) else {
            bail!("{} is not a winnowr feature table (no winnowr metadata)", path.display());
        };
        serde_json::from_str(metadata)
            .with_context(|| format!("{} has invalid winnowr metadata", path.display()))
    }

    /// Reads some columns and rows of a table (see [`TableSelection`]), filtering rows batch by
    /// batch so that the whole table is never in memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, has no winnowr metadata, changes its length or
    /// footer (metadata, schema, or row groups) before or during row decoding, or (when selecting
    /// by variant type) has no string `variant_type` column. Changes are detected only at the
    /// observation points: a rewrite restored before the final check, or preserving the observed
    /// length, modification time and footer bytes, can escape detection.
    pub fn read_parquet_selected<P: AsRef<Path>>(
        path: P,
        selection: &TableSelection,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let file = PositionedFile::new(file)?;
        let not_parquet = || format!("{} is not a Parquet file", path.display());
        // The footer is decoded from these bytes, which every row-group decoder compares with its
        // own read of the footer: the decoded structs cannot be compared, as an all-NaN float
        // column records NaN statistics, which never equal themselves.
        let footer = footer_metadata(&file).with_context(not_parquet)?;
        let opening_metadata = ParquetMetaDataReader::decode_metadata(&footer)
            .and_then(|metadata| {
                ArrowReaderMetadata::try_new(Arc::new(metadata), ArrowReaderOptions::default())
            })
            .with_context(not_parquet)?;
        let mut builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
            file.clone(),
            opening_metadata.clone(),
        );
        let Some(metadata) = builder.schema().metadata().get(METADATA_KEY) else {
            bail!("{} is not a winnowr feature table (no winnowr metadata)", path.display());
        };
        let info: FeatureTableInfo = serde_json::from_str(metadata)
            .with_context(|| format!("{} has invalid winnowr metadata", path.display()))?;
        let projection = selection.columns.as_ref().map(|columns| {
            let wanted = |name: &str| {
                columns.iter().any(|c| c == name)
                    || (selection.variant_type.is_some() && name == "variant_type")
            };
            let indices: Vec<usize> = builder
                .schema()
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, field)| wanted(field.name()))
                .map(|(index, _)| index)
                .collect();
            ProjectionMask::roots(builder.parquet_schema(), indices)
        });
        if let Some(mask) = &projection {
            builder = builder.with_projection(mask.clone());
        }
        let row_groups = builder.metadata().num_row_groups();
        // Drop the schema metadata (e.g. a writer's own): it is carried in `info`.
        let reader = builder.build()?;
        let schema =
            Arc::new(Schema::new(arrow_array::RecordBatchReader::schema(&reader).fields().clone()));
        drop(reader);
        // Row groups use independent cursors on the same opened file, so replacement of its path
        // cannot pair one file's metadata with another's rows. They are decoded in parallel and
        // kept in order, as one batch rather than the reader's default 1024 rows: on a 4.8M-call
        // table that read in about 70% of the time, with a lower peak memory.
        #[cfg(test)]
        tests::after_metadata();
        // The whole footer, row-group statistics included, is compared before any row group is
        // decoded and again once all are, so a rewrite that changes it before or during decoding
        // is rejected; the footer ends the file, so a change of length is rejected first. Each
        // row group checks only the length and modification time, so the footer is not read
        // once per row group (its size grows with their number). A rewrite that leaves the
        // length and footer bytes identical is not detected.
        let changed = || format!("{} changed while it was read", path.display());
        let reread = || {
            format!(
                "failed to read the footer of {} again: it may have changed while it was read",
                path.display()
            )
        };
        let check_footer = || -> Result<()> {
            if !file.unchanged().with_context(reread)?
                || footer_metadata(&file).with_context(reread)? != footer
            {
                bail!(changed());
            }
            Ok(())
        };
        check_footer()?;
        let per_group: Vec<Vec<RecordBatch>> = (0..row_groups)
            .into_par_iter()
            .map(|group| -> Result<Vec<RecordBatch>> {
                if !file.unchanged().with_context(reread)? {
                    bail!(changed());
                }
                #[cfg(test)]
                tests::during_decoding(path);
                let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
                    file.clone(),
                    opening_metadata.clone(),
                );
                let rows = usize::try_from(builder.metadata().row_group(group).num_rows())?;
                let mut builder = builder.with_row_groups(vec![group]).with_batch_size(rows.max(1));
                if let Some(mask) = &projection {
                    builder = builder.with_projection(mask.clone());
                }
                let mut batches = Vec::new();
                for batch in builder.build()? {
                    let batch =
                        batch.with_context(|| format!("failed to read {}", path.display()))?;
                    let batch = match selection.variant_type {
                        Some(variant_type) => {
                            let mask = string_equals(&batch, "variant_type", variant_type.as_str())
                                .with_context(|| format!("failed to read {}", path.display()))?;
                            filter_record_batch(&batch, &mask)?
                        }
                        None => batch,
                    };
                    batches.push(batch);
                }
                Ok(batches)
            })
            .collect::<Result<_>>()?;
        check_footer()?;
        let batches: Vec<RecordBatch> = per_group.into_iter().flatten().collect();
        // Columns are concatenated in parallel: copying a whole-genome table on one thread took
        // longer than decoding it.
        let rows = batches.iter().map(RecordBatch::num_rows).sum();
        let columns: Vec<ArrayRef> = (0..schema.fields().len())
            .into_par_iter()
            .map(|column| -> Result<ArrayRef> {
                let parts: Vec<&dyn Array> =
                    batches.iter().map(|batch| batch.column(column).as_ref()).collect();
                if parts.is_empty() {
                    return Ok(arrow_array::new_empty_array(schema.field(column).data_type()));
                }
                Ok(arrow_select::concat::concat(&parts)?)
            })
            .collect::<Result<_>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        let batch = RecordBatch::try_new_with_options(schema, columns, &options)?;
        Ok(FeatureTable { batch, info })
    }
}

/// The Thrift-encoded metadata that ends a Parquet file, read from the opened file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or does not end as a Parquet file does.
fn footer_metadata(file: &PositionedFile) -> Result<bytes::Bytes> {
    let tail_start =
        file.length.checked_sub(FOOTER_SIZE as u64).context("too short to be a Parquet file")?;
    let tail: [u8; FOOTER_SIZE] = file.get_bytes(tail_start, FOOTER_SIZE)?.as_ref().try_into()?;
    let length = FooterTail::try_new(&tail)?.metadata_length();
    let start = tail_start
        .checked_sub(length as u64)
        .context("the Parquet metadata is longer than the file")?;
    Ok(file.get_bytes(start, length)?)
}

/// Which columns and rows of a feature table to read (see
/// [`FeatureTable::read_parquet_selected`]).
#[derive(Debug, Clone, Default)]
pub struct TableSelection {
    /// The columns to read, in the file's order (names the file lacks are skipped); `None` reads
    /// every column.
    pub columns: Option<Vec<String>>,
    /// Only the rows of this variant type.
    pub variant_type: Option<VariantType>,
}

/// Where a batch's string column equals `value`.
fn string_equals(batch: &RecordBatch, name: &str, value: &str) -> Result<BooleanArray> {
    let column = batch.column_by_name(name).with_context(|| format!("no {name} column"))?;
    if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(values.iter().map(|v| Some(v == Some(value))).collect());
    }
    if let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.iter().map(|v| Some(v == Some(value))).collect());
    }
    bail!("column {name} is {}, not a string column", column.data_type())
}

/// Columns are declared nullable, as most Parquet writers declare them, so the schemas of tables
/// from other writers match; winnowr itself writes NaN, not nulls, for missing numbers.
const NULLABLE: bool = true;

/// A string column with one value per record, written by `write_value`.
fn text<F: Fn(&RawRecord, &mut String)>(records: &[RawRecord], write_value: F) -> ArrayRef {
    let mut builder = StringBuilder::with_capacity(records.len(), records.len() * 4);
    let mut value = String::new();
    for record in records {
        value.clear();
        write_value(record, &mut value);
        builder.append_value(&value);
    }
    Arc::new(builder.finish())
}

/// Appends an allele's bases (ASCII in a valid VCF) to `out`.
fn push_allele(out: &mut String, allele: &[u8]) {
    out.push_str(&String::from_utf8_lossy(allele));
}

/// Accumulates named columns for a [`RecordBatch`].
#[derive(Default)]
struct ColumnsBuilder {
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
}

impl ColumnsBuilder {
    fn push(&mut self, name: &str, array: ArrayRef) {
        self.fields.push(Field::new(name, array.data_type().clone(), NULLABLE));
        self.arrays.push(array);
    }

    fn finish(self, rows: usize) -> Result<RecordBatch> {
        if let Some((field, array)) =
            self.fields.iter().zip(&self.arrays).find(|(_, array)| array.len() != rows)
        {
            bail!("column {} has {} values for {rows} rows", field.name(), array.len());
        }
        Ok(RecordBatch::try_new(Arc::new(Schema::new(self.fields)), self.arrays)?)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use tempfile::TempDir;

    use super::*;

    thread_local! {
        static AFTER_METADATA: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
            const { std::cell::RefCell::new(None) };
    }

    pub(super) fn after_metadata() {
        AFTER_METADATA.with(|hook| {
            if let Some(action) = hook.borrow_mut().take() {
                action();
            }
        });
    }

    /// An action on a path, run once by a row-group worker reading that path, before decoding.
    type DecodingHook = (std::path::PathBuf, Box<dyn FnOnce() + Send>);

    /// Shared, as the row groups are decoded on other threads; keyed by path, as tests run in
    /// parallel.
    static DURING_DECODING: std::sync::Mutex<Vec<DecodingHook>> = std::sync::Mutex::new(Vec::new());

    pub(super) fn during_decoding(path: &Path) {
        let action = {
            let mut hooks = DURING_DECODING.lock().unwrap();
            let index = hooks.iter().position(|(hooked, _)| hooked == path);
            index.map(|index| hooks.remove(index).1)
        };
        if let Some(action) = action {
            action();
        }
    }

    #[test]
    fn positioned_file_reads_have_independent_offsets_and_checked_bounds() {
        use parquet::file::reader::ChunkReader as _;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bytes");
        std::fs::write(&path, b"abcdefgh").unwrap();
        let source = PositionedFile::new(File::open(&path).unwrap()).unwrap();
        let mut first = source.get_read(1).unwrap();
        let mut second = source.get_read(5).unwrap();
        let mut pair = [0; 2];
        first.read_exact(&mut pair).unwrap();
        assert_eq!(&pair, b"bc");
        second.read_exact(&mut pair).unwrap();
        assert_eq!(&pair, b"fg");
        assert_eq!(source.get_bytes(0, 4).unwrap().as_ref(), b"abcd");
        first.read_exact(&mut pair).unwrap();
        assert_eq!(&pair, b"de");
        let owned = source.get_bytes(4, 4).unwrap();
        assert!(source.get_bytes(8, 0).unwrap().is_empty());
        assert!(source.get_bytes(8, 1).is_err());
        assert!(source.get_bytes(u64::MAX, 2).is_err());
        assert!(source.get_read(9).is_err());
        assert_eq!(source.get_read(8).unwrap().read(&mut pair).unwrap(), 0);
        (0..128).into_par_iter().for_each(|i| {
            let offset = i % 6;
            assert_eq!(
                source.get_bytes(offset as u64, 2).unwrap().as_ref(),
                &b"abcdefgh"[offset..offset + 2]
            );
        });
        File::options().write(true).open(&path).unwrap().set_len(2).unwrap();
        assert!(source.get_bytes(0, 8).is_err());
        assert_eq!(owned.as_ref(), b"efgh");
    }

    #[test]
    fn replacement_after_metadata_cannot_change_the_rows() {
        let dir = TempDir::new().unwrap();
        let mut original =
            crate::testing::call_table(&[("chr1", 10, "A", "C", "PASS", "snp", "0/1", false)]);
        original.info.alignment =
            Some(crate::testing::provenance(&[("alt-project", "primary")], false));
        let mut replacement =
            crate::testing::call_table(&[("chr1", 20, "G", "T", "PASS", "snp", "1/1", false)]);
        replacement.info.alignment =
            Some(crate::testing::provenance(&[("alt-project", "keep-alt")], false));
        let path = dir.path().join("input.parquet");
        let next = dir.path().join("replacement.parquet");
        original.write_parquet(&path).unwrap();
        replacement.write_parquet(&next).unwrap();
        let destination = path.clone();
        AFTER_METADATA.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                std::fs::rename(next, destination).unwrap();
            }));
        });
        let loaded = FeatureTable::read_parquet(&path).unwrap();
        assert_eq!(loaded.i64_column("pos").unwrap(), &[10]);
        assert_eq!(loaded.info.alignment, original.info.alignment);
        assert_eq!(FeatureTable::read_parquet(&path).unwrap().i64_column("pos").unwrap(), &[20]);
    }

    #[test]
    fn in_place_metadata_rewrite_cannot_change_the_loaded_provenance() {
        use std::io::Write as _;

        let dir = TempDir::new().unwrap();
        let mut original =
            crate::testing::call_table(&[("chr1", 10, "A", "C", "PASS", "snp", "0/1", false)]);
        original.info.alignment =
            Some(crate::testing::provenance(&[("alt-project", "primary")], false));
        let mut replacement =
            crate::testing::call_table(&[("chr1", 20, "A", "C", "PASS", "snp", "0/1", false)]);
        replacement.info.alignment =
            Some(crate::testing::provenance(&[("alt-project", "otherxx")], false));
        let path = dir.path().join("input.parquet");
        let next = dir.path().join("replacement.parquet");
        original.write_parquet(&path).unwrap();
        replacement.write_parquet(&next).unwrap();
        let bytes = std::fs::read(&next).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes.len() as u64);
        let destination = path.clone();
        AFTER_METADATA.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                let mut file =
                    File::options().write(true).truncate(true).open(destination).unwrap();
                file.write_all(&bytes).unwrap();
            }));
        });
        let error = FeatureTable::read_parquet(&path).unwrap_err();
        assert!(error.to_string().contains("changed while it was read"), "{error:#}");
        assert_eq!(FeatureTable::read_parquet(&path).unwrap().i64_column("pos").unwrap(), &[20]);
    }

    /// Encodes small row groups without allocating the production row-group size.
    fn rewrite_fixture(path: &Path, batch: &RecordBatch, rows_per_group: usize) {
        let metadata = serde_json::to_string(&FeatureTableInfo::default()).unwrap();
        let schema = batch
            .schema()
            .as_ref()
            .clone()
            .with_metadata(HashMap::from([(METADATA_KEY.to_string(), metadata)]));
        let batch = batch.clone().with_schema(Arc::new(schema)).unwrap();
        let mut writer =
            ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), None).unwrap();
        for start in (0..batch.num_rows()).step_by(rows_per_group) {
            writer
                .write(&batch.slice(start, rows_per_group.min(batch.num_rows() - start)))
                .unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
    }

    /// Pads only the space before the footer, leaving column offsets and schemas unchanged.
    fn pad_parquet_to(path: &Path, length: usize) -> Vec<u8> {
        let mut bytes = std::fs::read(path).unwrap();
        let footer_length =
            u32::from_le_bytes(bytes[bytes.len() - 8..bytes.len() - 4].try_into().unwrap())
                as usize;
        let start = bytes.len() - 8 - footer_length;
        let padding = length - bytes.len();
        bytes.splice(start..start, std::iter::repeat_n(0, padding));
        std::fs::write(path, &bytes).unwrap();
        bytes
    }

    /// Rewrites the same inode after the opening footer has been read.
    fn rewrite_after_metadata(path: &Path, bytes: Vec<u8>) {
        use std::io::Write as _;
        let destination = path.to_path_buf();
        AFTER_METADATA.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                let mut file =
                    File::options().write(true).truncate(true).open(destination).unwrap();
                file.write_all(&bytes).unwrap();
            }));
        });
    }

    #[test]
    fn in_place_row_group_removal_returns_an_error_instead_of_panicking() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.parquet");
        let next = dir.path().join("replacement.parquet");
        let batch = RecordBatch::try_from_iter([(
            "qual",
            Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef,
        )])
        .unwrap();
        rewrite_fixture(&path, &batch, 1);
        rewrite_fixture(&next, &batch, 2);
        let length =
            std::fs::metadata(&path).unwrap().len().max(std::fs::metadata(&next).unwrap().len())
                as usize;
        pad_parquet_to(&path, length);
        let bytes = pad_parquet_to(&next, length);
        assert_eq!(FeatureTable::read_parquet(&next).unwrap().num_rows(), 2);
        rewrite_after_metadata(&path, bytes);
        let error = FeatureTable::read_parquet(&path).unwrap_err();
        assert!(error.to_string().contains("changed while it was read"), "{error:#}");
    }

    #[test]
    fn in_place_row_rewrite_with_the_same_schema_and_metadata_is_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.parquet");
        let next = dir.path().join("replacement.parquet");
        let values = |values: Vec<f64>| {
            RecordBatch::try_from_iter([("qual", Arc::new(Float64Array::from(values)) as ArrayRef)])
                .unwrap()
        };
        rewrite_fixture(&path, &values(vec![10.0, 20.0]), 1);
        rewrite_fixture(&next, &values(vec![30.0, 40.0]), 1);
        let length =
            std::fs::metadata(&path).unwrap().len().max(std::fs::metadata(&next).unwrap().len())
                as usize;
        pad_parquet_to(&path, length);
        let bytes = pad_parquet_to(&next, length);
        // Only the rows differ: the length, schema, winnowr metadata and row-group count match.
        assert_eq!(
            FeatureTable::read_info(&path).unwrap(),
            FeatureTable::read_info(&next).unwrap()
        );
        let (original, replacement) = (
            FeatureTable::read_parquet(&path).unwrap(),
            FeatureTable::read_parquet(&next).unwrap(),
        );
        assert_eq!(original.batch.schema(), replacement.batch.schema());
        assert_ne!(original.batch, replacement.batch);
        rewrite_after_metadata(&path, bytes);
        let error = FeatureTable::read_parquet(&path).unwrap_err();
        assert!(error.to_string().contains("changed while it was read"), "{error:#}");
    }

    /// An update that keeps the old footer bytes and writes a new footer farther on must not be
    /// read at the length the file had when it was opened.
    #[test]
    fn pr38_an_in_place_update_that_extends_the_file_is_rejected() {
        use std::io::Write as _;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.parquet");
        let next = dir.path().join("replacement.parquet");
        let values = |values: Vec<f64>| {
            RecordBatch::try_from_iter([("qual", Arc::new(Float64Array::from(values)) as ArrayRef)])
                .unwrap()
        };
        rewrite_fixture(&path, &values(vec![10.0, 20.0]), 1);
        rewrite_fixture(&next, &values(vec![30.0, 40.0, 50.0]), 1);
        let appended = std::fs::read(&next).unwrap();
        let destination = path.clone();
        AFTER_METADATA.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                let mut file = File::options().append(true).open(destination).unwrap();
                file.write_all(&appended).unwrap();
            }));
        });
        let error = FeatureTable::read_parquet(&path).unwrap_err();
        assert!(error.to_string().contains("changed while it was read"), "{error:#}");
    }

    /// A rewrite while the rows are decoded (here of a table's only row group) is found once
    /// they all are.
    #[test]
    fn pr38_a_rewrite_during_decoding_is_rejected() {
        use std::io::Write as _;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.parquet");
        let next = dir.path().join("replacement.parquet");
        let values = |values: Vec<f64>| {
            RecordBatch::try_from_iter([("qual", Arc::new(Float64Array::from(values)) as ArrayRef)])
                .unwrap()
        };
        rewrite_fixture(&path, &values(vec![10.0, 20.0]), 2);
        rewrite_fixture(&next, &values(vec![30.0, 40.0]), 2);
        let length =
            std::fs::metadata(&path).unwrap().len().max(std::fs::metadata(&next).unwrap().len())
                as usize;
        pad_parquet_to(&path, length);
        let bytes = pad_parquet_to(&next, length);
        let destination = path.clone();
        DURING_DECODING.lock().unwrap().push((
            path.clone(),
            Box::new(move || {
                let mut file =
                    File::options().write(true).truncate(true).open(destination).unwrap();
                file.write_all(&bytes).unwrap();
            }),
        ));
        let error = FeatureTable::read_parquet(&path).unwrap_err();
        assert!(error.to_string().contains("changed while it was read"), "{error:#}");
    }

    /// An all-NaN column records NaN statistics, which a decoded-footer comparison would never
    /// find equal to themselves, failing every read of an unchanged file.
    #[test]
    fn a_table_with_an_all_nan_column_is_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.parquet");
        let batch = RecordBatch::try_from_iter([(
            "qual",
            Arc::new(Float64Array::from(vec![f64::NAN, f64::NAN])) as ArrayRef,
        )])
        .unwrap();
        rewrite_fixture(&path, &batch, 1);
        assert_eq!(FeatureTable::read_parquet(&path).unwrap().num_rows(), 2);
    }

    #[test]
    fn in_place_rewrite_to_another_length_names_the_path() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.parquet");
        let batch = RecordBatch::try_from_iter([(
            "qual",
            Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef,
        )])
        .unwrap();
        rewrite_fixture(&path, &batch, 1);
        rewrite_after_metadata(&path, b"not a parquet file".to_vec());
        let error = FeatureTable::read_parquet(&path).unwrap_err();
        // The length change is found before the footer is read again.
        let expected = format!("{} changed while it was read", path.display());
        assert_eq!(error.to_string(), expected, "{error:#}");
    }

    #[test]
    fn in_place_schema_rewrite_is_rejected_before_decoding() {
        for projected in [false, true] {
            for change_type in [false, true] {
                let dir = TempDir::new().unwrap();
                let path = dir.path().join("input.parquet");
                let next = dir.path().join("replacement.parquet");
                let qual = Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef;
                let depth = Arc::new(Float64Array::from(vec![30.0, 40.0])) as ArrayRef;
                let original =
                    RecordBatch::try_from_iter([("qual", qual.clone()), ("dp", depth.clone())])
                        .unwrap();
                let replacement = if change_type {
                    RecordBatch::try_from_iter([
                        ("qual", Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef),
                        ("dp", depth),
                    ])
                    .unwrap()
                } else {
                    RecordBatch::try_from_iter([("dp", depth), ("qual", qual)]).unwrap()
                };
                rewrite_fixture(&path, &original, 2);
                rewrite_fixture(&next, &replacement, 2);
                let length = std::fs::metadata(&path)
                    .unwrap()
                    .len()
                    .max(std::fs::metadata(&next).unwrap().len())
                    as usize;
                pad_parquet_to(&path, length);
                let bytes = pad_parquet_to(&next, length);
                assert_eq!(FeatureTable::read_parquet(&next).unwrap().num_rows(), 2);
                rewrite_after_metadata(&path, bytes);
                let selection = TableSelection {
                    columns: projected.then(|| vec!["qual".to_string()]),
                    ..TableSelection::default()
                };
                let error = FeatureTable::read_parquet_selected(&path, &selection).unwrap_err();
                assert!(error.to_string().contains("changed while it was read"), "{error:#}");
            }
        }
    }

    #[test]
    fn reads_selected_columns_and_rows_across_row_groups() {
        // More rows than fit in one row group, so the file has two.
        let rows = DEFAULT_MAX_ROW_GROUP_ROW_COUNT + 1000;
        let types: Vec<&str> = (0..rows)
            .map(|i| if i % 3 == 0 { VariantType::Indel } else { VariantType::Snp }.as_str())
            .collect();
        let fields = vec![
            Field::new("pos", DataType::Int64, true),
            Field::new("variant_type", DataType::Utf8, true),
            Field::new("qual", DataType::Float64, true),
        ];
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values((0..rows).map(|i| i as i64))),
            Arc::new(StringArray::from(types)),
            Arc::new(Float64Array::from_iter_values((0..rows).map(|i| i as f64 / 2.0))),
        ];
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let info = FeatureTableInfo { sample: "S1".to_string(), ..FeatureTableInfo::default() };
        let table = FeatureTable::new(batch, info);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("t.parquet");
        table.write_parquet(&path).unwrap();

        let whole = FeatureTable::read_parquet(&path).unwrap();
        assert_eq!(whole.column_names(), ["pos", "variant_type", "qual"]);
        assert_eq!(whole.num_rows(), rows);
        assert!(whole.i64_column("pos").unwrap().iter().copied().eq(0..rows as i64));

        let selection = TableSelection {
            columns: Some(vec!["qual".into()]),
            variant_type: Some(VariantType::Indel),
        };
        let indels = FeatureTable::read_parquet_selected(&path, &selection).unwrap();
        assert_eq!(indels.column_names(), ["variant_type", "qual"]);
        let expected: Vec<f64> = (0..rows).step_by(3).map(|i| i as f64 / 2.0).collect();
        let qual = indels.f64_column("qual").unwrap();
        assert_eq!(qual.len(), expected.len());
        assert!(qual.iter().eq(expected.iter()), "the selected rows' values differ");
        assert_eq!(indels.info(), table.info());
    }

    /// Table metadata with the given record of the depth `dp_norm` divided by.
    fn info_with_divisor(dp_norm_divisor: Option<DpNormDivisor>) -> FeatureTableInfo {
        FeatureTableInfo {
            sample: "S1".to_string(),
            contigs: vec!["chr1".to_string()],
            dp_norm_divisor,
            ..FeatureTableInfo::default()
        }
    }

    /// A table of the gVCF source is written as before sources were recorded, and one written
    /// before then reads as of the gVCF source; another source round trips.
    #[test]
    fn the_feature_source_is_recorded_unless_it_is_gvcf() {
        let gvcf = info_with_divisor(None);
        let json = serde_json::to_string(&gvcf).unwrap();
        assert!(!json.contains("feature_source"), "{json}");
        assert_eq!(serde_json::from_str::<FeatureTableInfo>(&json).unwrap(), gvcf);
        let vcf = FeatureTableInfo { feature_source: FeatureSource::Vcf, ..gvcf };
        let json = serde_json::to_string(&vcf).unwrap();
        assert!(json.contains(r#""feature_source":"vcf""#), "{json}");
        assert_eq!(serde_json::from_str::<FeatureTableInfo>(&json).unwrap(), vcf);
        let unknown = json.replace(r#""feature_source":"vcf""#, r#""feature_source":"bam""#);
        let error = serde_json::from_str::<FeatureTableInfo>(&unknown).unwrap_err().to_string();
        assert!(error.contains("unknown feature source 'bam'"), "{error}");
    }

    /// A table of unknown alignment provenance is written as before it was recorded; a known
    /// one round trips.
    #[test]
    fn the_alignment_provenance_is_recorded_when_known() {
        let unknown = info_with_divisor(None);
        let json = serde_json::to_string(&unknown).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("alignment").is_none(), "{json}");
        assert_eq!(serde_json::from_str::<FeatureTableInfo>(&json).unwrap(), unknown);
        let provenance = crate::testing::provenance(&[("aligner", "bwa-mem3")], false);
        let known = FeatureTableInfo { alignment: Some(provenance), ..unknown };
        let json = serde_json::to_string(&known).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value["alignment"],
            serde_json::json!({"aligner": "bwa-mem3", "options-inference": "explicit"})
        );
        assert_eq!(serde_json::from_str::<FeatureTableInfo>(&json).unwrap(), known);
    }

    #[test]
    fn the_dp_norm_divisor_round_trips_through_json() {
        for divisor in [
            DpNormDivisor { depth: Some(31.5), given: false },
            DpNormDivisor { depth: Some(8.0), given: true },
            DpNormDivisor { depth: None, given: false },
        ] {
            let info = info_with_divisor(Some(divisor));
            let json = serde_json::to_string(&info).unwrap();
            assert!(json.contains("\"dp_norm_divisor\""), "{json}");
            let read: FeatureTableInfo = serde_json::from_str(&json).unwrap();
            assert_eq!(read, info);
        }
        let json = serde_json::to_string(&info_with_divisor(None)).unwrap();
        assert!(!json.contains("dp_norm_divisor"), "the key is left out when unknown: {json}");
    }

    #[test]
    fn tables_without_a_dp_norm_divisor_read_as_unknown() {
        // The metadata of a table written before the divisor was recorded.
        let json = r#"{"sample":"S1","contigs":["chr1"],"region_names":[],"fn_counts":null,
            "promote_labelled":false,"strlen":false,"reference_contigs":null}"#;
        let info: FeatureTableInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info, info_with_divisor(None));
    }
}
