//! The labelled calls of one sample, as used for training and evaluation.

use anyhow::{Context, Result, bail};

use crate::common::VariantType;
use crate::labels::{Label, PromoteLabel, UnfixableFlips, labelled_mask, unfixable_flips};
use crate::promote_labels::promote_mask;
use crate::regenotype::{with_regenotype_columns, without_regenotype};
use crate::rescore::{Counts, Probabilities, baseline_counts};
use crate::stutter::{StutterModel, add_length_call_features};
use crate::table::FeatureTable;
use crate::thresholds::{FoldPredictions, PromoteFold};

/// The classes in a label column.
///
/// # Errors
///
/// Returns an error if the table lacks the column or a value is not a class.
fn classes<T>(table: &FeatureTable, column: &str, of: fn(f64) -> Option<T>) -> Result<Vec<T>> {
    table
        .f64_column(column)?
        .iter()
        .map(|&value| of(value).with_context(|| format!("{column} {value} is not a class")))
        .collect()
}

/// The labelled calls of one sample and one variant type.
#[derive(Debug, Clone)]
pub struct SampleCalls {
    /// The sample name.
    pub sample: String,
    /// The labelled PASS calls (see [`labelled_mask`]), with the re-genotype columns.
    pub calls: FeatureTable,
    /// The label of each call.
    pub labels: Vec<Label>,
    /// The sample's TP/FP/FN before any action.
    pub baseline: Counts,
    /// Per call, whether it is left out of the fits (empty if none is); it is still scored in
    /// cross-validation.
    pub excluded: Vec<bool>,
}

impl SampleCalls {
    /// Selects the labelled calls of a variant type from a labelled feature table.
    ///
    /// The calls get the re-genotype columns; without a stutter model they are all false. With
    /// one the calls also get the length caller's features and genotype.
    ///
    /// # Arguments
    ///
    /// * `table` - the labelled feature table.
    /// * `variant_type` - the variant type to select.
    /// * `stutter` - the stutter model for the length-call features, if the model uses them.
    /// * `regenotype` - relabel as [`Label::Regenotype`] the FPs that a flip does not fix but
    ///   re-genotyping to the length caller's genotype does (needs `stutter`).
    ///
    /// # Errors
    ///
    /// Returns an error if the table has no hap.py labels, or length calls are requested from a
    /// table without strlen evidence.
    pub fn from_table(
        table: &FeatureTable,
        variant_type: VariantType,
        stutter: Option<&StutterModel>,
        regenotype: bool,
    ) -> Result<Self> {
        let info = table.info();
        if !info.is_labelled() {
            bail!(
                "the feature table for {} has no hap.py labels (extract with --happy)",
                info.sample
            );
        }
        if regenotype && stutter.is_none() {
            bail!("re-genotype labels need a stutter model");
        }
        let mut calls = without_regenotype(table.filter(&labelled_mask(table, variant_type)?)?)?;
        let mut labels = classes(&calls, "label", Label::from_value)?;
        if let Some(stutter) = stutter {
            if !info.strlen {
                bail!(
                    "the feature table for {} has no strlen evidence (extract with --strlen)",
                    info.sample
                );
            }
            calls = with_regenotype_columns(add_length_call_features(calls, stutter)?)?;
            if regenotype {
                let regenotype_ok = calls.bool_column("regenotype_ok")?;
                let flip_ok = calls.bool_column("flip_ok")?;
                for (i, label) in labels.iter_mut().enumerate() {
                    let fixed_by_flip = *label == Label::Flip && flip_ok[i];
                    if regenotype_ok[i] && !fixed_by_flip {
                        *label = Label::Regenotype;
                    }
                }
                let values = labels.iter().map(|label| label.index() as f64).collect();
                calls = calls.with_f64_column("label", values)?;
            }
        }
        let baseline = baseline_counts(&labels, i64::try_from(info.fn_count(variant_type)?)?);
        Ok(SampleCalls {
            sample: info.sample.clone(),
            calls,
            labels,
            baseline,
            excluded: Vec::new(),
        })
    }

    /// Applies an unfixable-flip policy to the training labels (see [`UnfixableFlips`]), and
    /// returns how many calls it applied to.
    ///
    /// [`UnfixableFlips::Drop`] relabels them [`Label::Drop`] (in `labels` and the `label`
    /// column); [`UnfixableFlips::Exclude`] marks them `excluded`. The baseline counts do not
    /// change: an unfixable flip is an FP either way.
    ///
    /// # Errors
    ///
    /// Returns an error if the calls lack the `flip_ok` or `flippable` columns.
    pub fn apply_unfixable_flips(&mut self, policy: UnfixableFlips) -> Result<usize> {
        if policy == UnfixableFlips::Flip {
            return Ok(0);
        }
        let flip_ok = self.calls.bool_column("flip_ok")?;
        let flippable = self.calls.bool_column("flippable")?;
        let unfixable = unfixable_flips(&self.labels, &flip_ok, &flippable);
        let n = unfixable.iter().filter(|&&u| u).count();
        match policy {
            UnfixableFlips::Flip => {}
            UnfixableFlips::Drop => {
                for (label, _) in self.labels.iter_mut().zip(&unfixable).filter(|(_, u)| **u) {
                    *label = Label::Drop;
                }
                let values = self.labels.iter().map(|label| label.index() as f64).collect();
                self.calls = self.calls.clone().with_f64_column("label", values)?;
            }
            UnfixableFlips::Exclude => self.excluded = unfixable,
        }
        Ok(n)
    }

    /// Packages predictions for these calls as a cross-validation fold.
    ///
    /// # Errors
    ///
    /// Returns an error if the calls lack a column of the fold.
    pub fn to_fold(&self, probabilities: Probabilities) -> Result<FoldPredictions> {
        let flags = |name: &str| self.calls.bool_column(name);
        Ok(FoldPredictions {
            probabilities,
            labels: self.labels.clone(),
            flip_ok: flags("flip_ok")?,
            flippable: flags("flippable")?,
            baseline: self.baseline,
            regenotypable: flags("regenotypable")?,
            regenotypes_to_ref: flags("regenotypes_to_ref")?,
            regenotype_ok: flags("regenotype_ok")?,
        })
    }
}

/// The promotion candidates of one sample and one variant type.
#[derive(Debug, Clone)]
pub struct PromoteSampleCalls {
    /// The sample name.
    pub sample: String,
    /// The labelled non-PASS candidates (see [`promote_mask`]).
    pub calls: FeatureTable,
    /// The promote label of each candidate.
    pub labels: Vec<PromoteLabel>,
    /// The sample's PASS-call TP/FP/FN, which promotions add to.
    pub baseline: Counts,
}

impl PromoteSampleCalls {
    /// Selects the promotion candidates of a variant type from a promote-labelled table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table has no promote labels.
    pub fn from_table(table: &FeatureTable, variant_type: VariantType) -> Result<Self> {
        let info = table.info();
        if !info.promote_labelled {
            bail!(
                "the feature table for {} has no promote labels (extract with --truth-vcf)",
                info.sample
            );
        }
        let baseline = SampleCalls::from_table(table, variant_type, None, false)?.baseline;
        let calls = table.filter(&promote_mask(table, variant_type)?)?;
        let labels = classes(&calls, "promote_label", PromoteLabel::from_value)?;
        Ok(PromoteSampleCalls { sample: info.sample.clone(), calls, labels, baseline })
    }

    /// Packages predictions for these candidates as a cross-validation fold.
    ///
    /// # Errors
    ///
    /// Returns an error if the candidates lack `flippable`.
    pub fn to_fold(&self, probabilities: Probabilities) -> Result<PromoteFold> {
        Ok(PromoteFold {
            probabilities,
            labels: self.labels.clone(),
            flippable: self.calls.bool_column("flippable")?,
            baseline: self.baseline,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::call_table;

    /// Four labelled PASS SNPs: a fixable flip, a flip whose flip is not the truth, a `1/2` flip
    /// (never flipped at scoring, though its split record's flip was correct) and a TP.
    fn sample() -> SampleCalls {
        let mut table = call_table(&[
            ("chr1", 10, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 20, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 30, "A", "C,G", "PASS", "snp", "1/2", false),
            ("chr1", 40, "A", "C", "PASS", "snp", "1/1", false),
        ]);
        table.info_mut().fn_counts =
            Some([("snp".to_string(), 3), ("indel".to_string(), 0)].into_iter().collect());
        let table = table
            .with_f64_column("label", vec![2.0, 2.0, 2.0, 1.0])
            .unwrap()
            .with_bool_column("flip_ok", vec![true, false, true, false])
            .unwrap()
            .with_bool_column("flippable", vec![true, true, false, true])
            .unwrap()
            .with_f64_column("truth_delta_lo", vec![f64::NAN; 4])
            .unwrap()
            .with_f64_column("truth_delta_hi", vec![f64::NAN; 4])
            .unwrap();
        SampleCalls::from_table(&table, VariantType::Snp, None, false).unwrap()
    }

    /// The strlen evidence, length calls, re-genotype columns and labels of indel calls, from
    /// strlen rows through `SampleCalls::from_table` with re-genotype labels.
    #[test]
    fn strlen_evidence_becomes_features_and_regenotype_labels() {
        use crate::repeat_features::add_strlen_features;
        use crate::strlen::{StrlenOptions, StrlenRow, StrlenTables};
        use crate::stutter::{LENGTH_BINS, N_ERRORS, N_PERIOD_CLASSES, StutterModel};

        let nan = f64::NAN;
        let mut table = call_table(&[
            ("chr1", 100, "GA", "G", "PASS", "indel", "0/1", false),
            ("chr1", 200, "GA", "G", "PASS", "indel", "0/1", false),
            ("chr1", 300, "GAT", "G", "PASS", "indel", "0/1", false),
            ("chr1", 400, "GA", "G", "PASS", "indel", "0/1", false),
            ("chr1", 500, "GA", "G", "PASS", "indel", "0/1", false),
        ]);
        table.info_mut().fn_counts =
            Some([("snp".to_string(), 0), ("indel".to_string(), 4)].into_iter().collect());
        let table = table
            // A flip a flip fixes, a drop re-genotyping fixes, a drop to 0/0, and two drops.
            .with_f64_column("label", vec![2.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap()
            .with_bool_column("flip_ok", vec![true, false, false, false, false])
            .unwrap()
            .with_bool_column("flippable", vec![true; 5])
            .unwrap()
            .with_f64_column("truth_delta_lo", vec![-1.0, -1.0, 0.0, nan, nan])
            .unwrap()
            .with_f64_column("truth_delta_hi", vec![-1.0, -1.0, 0.0, nan, nan])
            .unwrap();
        let row = |pos: u64, reference: &str, alt: &str, unit: &str, tract_len: u64| StrlenRow {
            chrom: "chr1".to_string(),
            pos,
            reference: reference.to_string(),
            alt: alt.to_string(),
            period: unit.len() as u64,
            tract_len,
            unit: unit.to_string(),
            tract_start: pos + 1,
            n_overlap: 40,
            n_spanning: 30,
            length_histogram: "-1:30".to_string(),
            mean_mapq: 60.0,
            frac_mapq_lt20: 0.0,
            mean_tract_bq: 30.0,
        };
        let rows = vec![
            // Every read one base short: a hom deletion.
            row(100, "GA", "G", "A", 5),
            row(200, "GA", "G", "A", 5),
            // Every read of the reference length, in an AT repeat.
            StrlenRow { length_histogram: "0:30".to_string(), ..row(300, "GAT", "G", "AT", 6) },
            // Reads overlap but none spans: no length call, even with lengths recorded.
            StrlenRow {
                n_spanning: 0,
                length_histogram: "-1:5".to_string(),
                ..row(400, "GA", "G", "A", 5)
            },
            // Another ALT at the same position and REF: not this call's evidence.
            row(500, "GA", "GAA", "A", 5),
        ];
        let strlen = StrlenTables { options: StrlenOptions::default(), rows };
        let table = add_strlen_features(table, &strlen).unwrap();
        assert!(table.info().strlen);
        let shown = |name: &str| -> Vec<String> {
            table.f64_column(name).unwrap().iter().map(|v| format!("{v}")).collect()
        };
        assert_eq!(shown("sl_period"), ["1", "1", "2", "1", "NaN"]);
        // Tract length in repeat units: 6 bp of AT is 3 units.
        assert_eq!(shown("sl_tract_units"), ["5", "5", "3", "5", "NaN"]);
        assert_eq!(shown("sl_frac_not_spanning"), ["0.25", "0.25", "0.25", "1", "NaN"]);
        assert_eq!(shown("sl_top1_delta"), ["-1", "-1", "0", "-1", "NaN"]);

        let errors = [0.001, 0.009, 0.05, 0.85, 0.05, 0.009, 0.001, 0.03];
        assert_eq!(errors.len(), N_ERRORS);
        let stutter = StutterModel::new(
            vec![vec![errors; LENGTH_BINS.len()]; N_PERIOD_CLASSES],
            vec![vec![0; LENGTH_BINS.len()]; N_PERIOD_CLASSES],
        )
        .unwrap();
        let calls =
            SampleCalls::from_table(&table, VariantType::Indel, Some(&stutter), true).unwrap();
        let columns = |name: &str| -> Vec<String> {
            calls.calls.f64_column(name).unwrap().iter().map(|v| format!("{v}")).collect()
        };
        assert_eq!(columns("lc_delta_lo"), ["-1", "-1", "0", "NaN", "NaN"]);
        assert_eq!(columns("lc_delta_hi"), ["-1", "-1", "0", "NaN", "NaN"]);
        assert_eq!(columns("lc_is_called"), ["0", "0", "0", "NaN", "NaN"]);
        let flags = |name: &str| calls.calls.bool_column(name).unwrap();
        assert_eq!(flags("regenotypable"), [true, true, true, false, false]);
        assert_eq!(flags("regenotypes_to_ref"), [false, false, true, false, false]);
        // FPs whose length call has the truth lengths and is not 0/0.
        assert_eq!(flags("regenotype_ok"), [true, true, false, false, false]);
        // Only the drop that a flip does not fix becomes a re-genotype.
        let expected = [Label::Flip, Label::Regenotype, Label::Drop, Label::Drop, Label::Drop];
        assert_eq!(calls.labels, expected);
        assert_eq!(*calls.calls.f64_column("label").unwrap(), [2.0, 3.0, 0.0, 0.0, 0.0]);
        // Without re-genotype labels, the labels stay.
        let plain =
            SampleCalls::from_table(&table, VariantType::Indel, Some(&stutter), false).unwrap();
        assert_eq!(plain.labels[1], Label::Drop);
        let message = SampleCalls::from_table(&table, VariantType::Indel, None, true)
            .unwrap_err()
            .to_string();
        assert_eq!(message, "re-genotype labels need a stutter model");
    }

    #[test]
    fn unfixable_flips_are_kept_as_flips_by_default() {
        let mut calls = sample();
        let before = calls.clone();
        assert_eq!(calls.apply_unfixable_flips(UnfixableFlips::Flip).unwrap(), 0);
        assert_eq!(calls.labels, before.labels);
        assert!(calls.excluded.is_empty());
    }

    #[test]
    fn unfixable_flips_can_be_dropped() {
        let mut calls = sample();
        let baseline = calls.baseline;
        assert_eq!(calls.apply_unfixable_flips(UnfixableFlips::Drop).unwrap(), 2);
        let expected = [Label::Flip, Label::Drop, Label::Drop, Label::Keep];
        assert_eq!(calls.labels, expected);
        assert_eq!(*calls.calls.f64_column("label").unwrap(), [2.0, 0.0, 0.0, 1.0]);
        assert_eq!(calls.baseline, baseline);
        assert!(calls.excluded.is_empty());
    }

    #[test]
    fn unfixable_flips_can_be_excluded() {
        let mut calls = sample();
        let before = calls.clone();
        assert_eq!(calls.apply_unfixable_flips(UnfixableFlips::Exclude).unwrap(), 2);
        assert_eq!(calls.labels, before.labels);
        assert_eq!(calls.excluded, [false, true, true, false]);
        assert_eq!(calls.baseline, before.baseline);
    }
}
