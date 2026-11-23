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
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{ArrowColumnChunk, compute_leaves};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{DEFAULT_MAX_ROW_GROUP_ROW_COUNT, WriterProperties};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
    /// Returns an error if the file cannot be read, has no winnowr metadata, or (when selecting
    /// by variant type) has no string `variant_type` column.
    pub fn read_parquet_selected<P: AsRef<Path>>(
        path: P,
        selection: &TableSelection,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("{} is not a Parquet file", path.display()))?;
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
        // Row groups are decoded in parallel, each by its own reader, and kept in order. Each is
        // decoded as one batch rather than the reader's default 1024 rows: on a 4.8M-call table
        // that read in about 70% of the time, with a lower peak memory.
        let per_group: Vec<Vec<RecordBatch>> = (0..row_groups)
            .into_par_iter()
            .map(|group| -> Result<Vec<RecordBatch>> {
                let file = File::open(path)
                    .with_context(|| format!("failed to open {}", path.display()))?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
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
