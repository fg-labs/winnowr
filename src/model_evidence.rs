//! The read evidence models use (repeat-length and pileup evidence): the settings training
//! tables share, the checks that evidence was measured as models need, and measuring it from the
//! reads while scoring.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use log::info;

use crate::common::PhaseTimer;
use crate::evidence::{Evidence, EvidenceTables, check_options, read_tables};
use crate::model::{ModelInfo, WinnowrModel};
use crate::pileup::{PileupOptions, PileupTables, measure_pileup};
use crate::pileup_features::add_pileup_features;
use crate::policy::{
    CallKey, Decision, Policy, check_model_kinds, decide_model_calls, merge_decisions,
};
use crate::read_pass::ReadInputs;
use crate::repeat_features::add_strlen_features;
use crate::strlen::{StrlenOptions, StrlenRow, StrlenTables, measure_regions_with, sites_to_rows};
use crate::table::{FeatureTable, FeatureTableInfo};

/// A kind of read evidence as models use it: which models and feature tables have it, and with
/// what settings.
pub(crate) trait ModelEvidence: Evidence {
    /// The features that use the evidence, in messages (e.g. `repeat-length features`).
    const FEATURES: &'static str;
    /// What training with the features needs, in messages (e.g. `repeat-length features need`).
    const TRAINING_NEEDS: &'static str;
    /// Whether [`check_table`] fails a feature table without the evidence a model needs (else it
    /// fails later, when the model reads its features).
    const TABLE_MUST_HAVE: bool;

    /// The settings a feature table's evidence was measured with, if it has the evidence.
    fn of_table(info: &FeatureTableInfo) -> Option<Self>;

    /// The settings the evidence a model uses must be measured with, if it uses the evidence.
    fn of_model(info: &ModelInfo) -> Option<Self>;
}

impl ModelEvidence for StrlenOptions {
    const FEATURES: &'static str = "repeat-length features";
    const TRAINING_NEEDS: &'static str = "repeat-length features need";
    const TABLE_MUST_HAVE: bool = false;

    fn of_table(info: &FeatureTableInfo) -> Option<Self> {
        info.strlen_options()
    }

    fn of_model(info: &ModelInfo) -> Option<Self> {
        info.strlen_options()
    }
}

impl ModelEvidence for PileupOptions {
    const FEATURES: &'static str = "pileup features";
    const TRAINING_NEEDS: &'static str = "--pileup-features needs";
    const TABLE_MUST_HAVE: bool = true;

    fn of_table(info: &FeatureTableInfo) -> Option<Self> {
        info.pileup_settings
    }

    fn of_model(info: &ModelInfo) -> Option<Self> {
        info.pileup_settings
    }
}

/// The settings the training tables' evidence was measured with, recorded in models with the
/// evidence's features.
///
/// # Errors
///
/// Returns an error if a table has no evidence, or two were measured differently.
pub(crate) fn shared_options<E: ModelEvidence>(infos: &[&FeatureTableInfo]) -> Result<E> {
    let mut shared: Option<(E, &str)> = None;
    for info in infos {
        let Some(options) = E::of_table(info) else {
            bail!(
                "{} tables extracted with --{name}; {}'s has no {name} evidence",
                E::TRAINING_NEEDS,
                info.sample,
                name = E::NAME
            );
        };
        match shared {
            None => shared = Some((options, &info.sample)),
            Some((first, sample)) if first != options => bail!(
                "--tables' {} evidence was measured with different {}: {sample}'s with {first}, \
                 {}'s with {options}",
                E::NAME,
                E::SETTINGS,
                info.sample
            ),
            Some(_) => {}
        }
    }
    let (options, _) = shared.context("no training tables")?;
    info!("The training tables' {} evidence was measured with {} {options}", E::NAME, E::SETTINGS);
    Ok(options)
}

/// Checks that a feature table's evidence was measured as each model with the evidence's features
/// needs.
///
/// # Errors
///
/// Returns an error if the settings differ, or the table lacks evidence a model needs and
/// [`ModelEvidence::TABLE_MUST_HAVE`].
pub(crate) fn check_table<E: ModelEvidence>(
    table: &Path,
    info: &FeatureTableInfo,
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Result<()> {
    let source = format!("the {} evidence of feature table {}", E::NAME, table.display());
    for (path, model) in paths.iter().zip(models) {
        let Some(trained) = E::of_model(&model.info) else { continue };
        match E::of_table(info) {
            Some(measured) => check_options(measured, &source, trained, path)?,
            None if E::TABLE_MUST_HAVE => bail!(
                "model {} uses {}, but feature table {} has no {name} evidence: extract it with \
                 --{name} (tables from `winnowr {name} {} {trained}`)",
                path.display(),
                E::FEATURES,
                table.display(),
                E::OPTION,
                name = E::NAME
            ),
            None => {}
        }
    }
    Ok(())
}

/// The models (by path) that use the evidence.
pub(crate) fn models_using<E: ModelEvidence>(
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Vec<String> {
    paths
        .iter()
        .zip(models)
        .filter(|(_, model)| E::of_model(&model.info).is_some())
        .map(|(path, _)| path.display().to_string())
        .collect()
}

/// Checks that `score` is given the evidence the models use (`users`, see [`models_using`]): its
/// `tables`, or reads to measure it from, and warns about tables no model uses.
///
/// # Errors
///
/// Returns an error if models use the evidence and neither tables nor reads are given.
pub(crate) fn check_supplied<E: ModelEvidence>(
    users: &[String],
    tables: &[PathBuf],
    alignments: bool,
) -> Result<()> {
    if !users.is_empty() && tables.is_empty() && !alignments {
        bail!(
            "models use {}; pass --{name} (tables from `winnowr {name}`), or --alignments and \
             --reference: {}",
            E::FEATURES,
            users.join(", "),
            name = E::NAME
        );
    }
    if users.is_empty() && !tables.is_empty() {
        log::warn!("no model uses {}; --{} is ignored", E::FEATURES, E::NAME);
    }
    Ok(())
}

/// The settings the evidence the models use must be measured with (`None` if no model uses it).
///
/// # Errors
///
/// Returns an error if two models need it measured differently.
pub(crate) fn models_options<E: ModelEvidence>(
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Result<Option<E>> {
    let mut options: Option<E> = None;
    for (path, model) in paths.iter().zip(models) {
        match (options, E::of_model(&model.info)) {
            (None, trained) => options = trained,
            (Some(first), Some(trained)) if first != trained => bail!(
                "--models need {} evidence measured with different {}: {} needs {trained}, \
                 another model {first}",
                E::NAME,
                E::SETTINGS,
                path.display()
            ),
            _ => {}
        }
    }
    Ok(options)
}

/// Reads the evidence `tables` given to `score` for the models that use the evidence, checking
/// they were measured as each model needs (`None` if none is given or no model uses it).
///
/// # Errors
///
/// Returns an error if a table cannot be read, or its settings differ from a model's.
pub(crate) fn read_for_models<E: ModelEvidence>(
    tables: &[PathBuf],
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Result<Option<EvidenceTables<E>>> {
    if tables.is_empty() || models.iter().all(|model| E::of_model(&model.info).is_none()) {
        return Ok(None);
    }
    let read = read_tables::<E>(tables)?;
    let source = format!("--{} {}", E::NAME, tables[0].display());
    for (path, model) in paths.iter().zip(models) {
        if let Some(trained) = E::of_model(&model.info) {
            check_options(read.options, &source, trained, path)?;
        }
    }
    Ok(Some(read))
}

/// What `score --alignments` measures: the reads, the settings and the contigs.
pub(crate) struct Measuring<'a> {
    /// The calls, reads and reference.
    pub inputs: &'a ReadInputs<'a>,
    /// The read settings of the repeat-length evidence the models need, if any.
    pub strlen: Option<StrlenOptions>,
    /// The settings of the pileup evidence the models need, if any.
    pub pileup: Option<PileupOptions>,
    /// The contigs to measure.
    pub contigs: &'a [String],
}

impl Measuring<'_> {
    /// Whether a model uses evidence that is being measured.
    fn is_needed_by(&self, model: &WinnowrModel) -> bool {
        (self.strlen.is_some() && StrlenOptions::of_model(&model.info).is_some())
            || (self.pileup.is_some() && PileupOptions::of_model(&model.info).is_some())
    }
}

/// Scores calls with evidence measured from the reads: the evidence is measured while the models
/// that do not use it score the calls, then joined for the models that do. Decisions are merged in
/// model order, as [`crate::policy::decide_calls`] does.
///
/// The measurement runs on its own thread, which is not waited for when scoring fails: the error
/// is returned at once (and the process exits), rather than after the whole pass over the reads.
/// Likewise a failed measurement is reported before the next model scores.
///
/// # Errors
///
/// Returns an error if the reads cannot be measured or a model cannot score the calls.
pub(crate) fn measure_and_decide(
    table: FeatureTable,
    models: &[WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    threads: NonZeroUsize,
    timer: &mut PhaseTimer,
) -> Result<HashMap<CallKey, Decision>> {
    type Measured = (Option<Vec<StrlenRow>>, Option<PileupTables>);
    check_model_kinds(models)?;
    let mut per_model: Vec<Option<Vec<(CallKey, Decision)>>> = vec![None; models.len()];
    // The measurement gets its own pool, so that neither it nor the scoring waits for the other's
    // queued work. Both have -t threads, so up to twice -t run at once (as the --alignments help
    // says).
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.get())
        .build()
        .context("could not set up the thread pool")?;
    let (vcf, alignments, reference) = (
        measure.inputs.vcf.to_path_buf(),
        measure.inputs.alignments.to_path_buf(),
        measure.inputs.reference.to_path_buf(),
    );
    let contigs = measure.contigs.to_vec();
    let (strlen, pileup) = (measure.strlen, measure.pileup);
    let mut measuring = Some(std::thread::spawn(move || {
        pool.install(|| -> Result<Measured> {
            let inputs = ReadInputs { vcf: &vcf, alignments: &alignments, reference: &reference };
            let strlen_rows = strlen
                .map(|options| {
                    let mut sites = measure_regions_with(&inputs, &contigs, &[options])?;
                    Ok::<_, anyhow::Error>(sites_to_rows(&sites.pop().unwrap_or_default()))
                })
                .transpose()?;
            let pileup = pileup
                .map(|options| {
                    let rows = measure_pileup(&inputs, &contigs, &options)?;
                    Ok::<_, anyhow::Error>(PileupTables { options, rows })
                })
                .transpose()?;
            Ok((strlen_rows, pileup))
        })
    }));
    let join = |handle: std::thread::JoinHandle<Result<Measured>>| -> Result<Measured> {
        handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    };
    let mut measured: Option<Measured> = None;
    for (slot, model) in per_model.iter_mut().zip(models) {
        if measure.is_needed_by(model) {
            continue;
        }
        if measuring.as_ref().is_some_and(std::thread::JoinHandle::is_finished) {
            measured = Some(join(measuring.take().expect("checked above"))?);
        }
        *slot = Some(decide_model_calls(&table, model, policy)?);
    }
    let (strlen_rows, pileup) = match (measured, measuring) {
        (Some(measured), _) => measured,
        (None, Some(handle)) => join(handle)?,
        (None, None) => unreachable!("the measurement is joined once"),
    };
    timer.lap("measuring the evidence from the reads (while scoring other models)");
    let mut table = table;
    if let (Some(rows), Some(options)) = (strlen_rows, measure.strlen) {
        table = add_strlen_features(table, &StrlenTables { options, rows })?;
    }
    if let Some(pileup) = pileup {
        table = add_pileup_features(table, &pileup)?;
    }
    for (slot, model) in per_model.iter_mut().zip(models) {
        if measure.is_needed_by(model) {
            *slot = Some(decide_model_calls(&table, model, policy)?);
        }
    }
    Ok(merge_decisions(per_model.into_iter().flatten().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_tables_must_share_their_strlen_settings() {
        let info = |sample: &str, strlen: bool, settings: Option<StrlenOptions>| FeatureTableInfo {
            sample: sample.to_string(),
            contigs: vec![],
            region_names: vec![],
            fn_counts: None,
            promote_labelled: false,
            strlen,
            strlen_settings: settings,
            pileup_settings: None,
            reference_contigs: None,
        };
        let new = Some(StrlenOptions::default());
        let (a, b) = (info("A", true, new), info("B", true, new));
        assert_eq!(shared_options::<StrlenOptions>(&[&a, &b]).unwrap(), StrlenOptions::default());
        // Tables without recorded settings were measured with the legacy settings.
        let (old, older) = (info("C", true, None), info("D", true, None));
        assert_eq!(
            shared_options::<StrlenOptions>(&[&old, &older]).unwrap(),
            StrlenOptions::LEGACY
        );
        let message = shared_options::<StrlenOptions>(&[&a, &old]).unwrap_err().to_string();
        assert!(message.contains("A's with mapq20,flank5, C's with mapq0,flank2"), "{message}");
        let none = info("E", false, None);
        let message = shared_options::<StrlenOptions>(&[&a, &none]).unwrap_err().to_string();
        assert!(message.contains("E's has no strlen evidence"), "{message}");
    }

    #[test]
    fn training_tables_must_share_their_pileup_settings() {
        let info = |sample: &str, settings: Option<&str>| FeatureTableInfo {
            sample: sample.to_string(),
            contigs: vec![],
            region_names: vec![],
            fn_counts: None,
            promote_labelled: false,
            strlen: false,
            strlen_settings: None,
            pileup_settings: settings.map(|s| s.parse().unwrap()),
            reference_contigs: None,
        };
        let (a, b) = (info("A", Some("default")), info("B", Some("default")));
        assert_eq!(shared_options::<PileupOptions>(&[&a, &b]).unwrap(), PileupOptions::default());
        let other = info("C", Some("mapq30"));
        let message = shared_options::<PileupOptions>(&[&a, &other]).unwrap_err().to_string();
        assert!(message.contains("A's with mapq20,bq10,flank5, C's with mapq30"), "{message}");
        let none = info("D", None);
        let message = shared_options::<PileupOptions>(&[&a, &none]).unwrap_err().to_string();
        assert!(message.contains("D's has no pileup evidence"), "{message}");
    }

    /// A genotype model using strlen evidence measured with `strlen` (a stutter model with no
    /// reads) and pileup evidence measured with `pileup`, if given.
    fn model(strlen: Option<StrlenOptions>, pileup: Option<&str>) -> WinnowrModel {
        use crate::common::{Task, VariantType};
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Indel);
        model.info.stutter = strlen.map(|_| crate::stutter::fit_stutter_model(&[]));
        model.info.strlen_settings = strlen;
        model.info.pileup_settings = pileup.map(|s| s.parse().unwrap());
        model
    }

    fn table_info(strlen: Option<StrlenOptions>, pileup: Option<&str>) -> FeatureTableInfo {
        FeatureTableInfo {
            sample: "S".to_string(),
            contigs: vec![],
            region_names: vec![],
            fn_counts: None,
            promote_labelled: false,
            strlen: strlen.is_some(),
            strlen_settings: strlen,
            pileup_settings: pileup.map(|s| s.parse().unwrap()),
            reference_contigs: None,
        }
    }

    #[test]
    fn tables_are_checked_against_each_models_settings() {
        let new = Some(StrlenOptions::default());
        let models = [model(new, None), model(None, Some("default"))];
        let paths = [PathBuf::from("a.winnowr.json"), PathBuf::from("b.winnowr.json")];
        let table = Path::new("t.parquet");
        let measured = table_info(new, Some("default"));
        assert!(check_table::<StrlenOptions>(table, &measured, &models, &paths).is_ok());
        assert!(check_table::<PileupOptions>(table, &measured, &models, &paths).is_ok());

        let other = table_info(Some(StrlenOptions::LEGACY), Some("mapq30"));
        let message =
            check_table::<StrlenOptions>(table, &other, &models, &paths).unwrap_err().to_string();
        assert!(message.contains("--reads mapq20,flank5"), "{message}");
        let message =
            check_table::<PileupOptions>(table, &other, &models, &paths).unwrap_err().to_string();
        assert!(message.contains("--settings mapq20,bq10,flank5"), "{message}");

        // A table without strlen evidence fails later, when a model reads its features; one
        // without the pileup evidence a model uses fails here.
        let bare = table_info(None, None);
        assert!(check_table::<StrlenOptions>(table, &bare, &models, &paths).is_ok());
        let message =
            check_table::<PileupOptions>(table, &bare, &models, &paths).unwrap_err().to_string();
        assert!(
            message.contains("model b.winnowr.json uses pileup features, but feature table"),
            "{message}"
        );
    }

    #[test]
    fn score_needs_the_evidence_models_use_measured_alike() {
        let new = Some(StrlenOptions::default());
        let models = [model(new, Some("default")), model(None, None), model(new, Some("mapq30"))];
        let paths: Vec<PathBuf> =
            ["a", "b", "c"].iter().map(|name| PathBuf::from(format!("{name}.json"))).collect();
        assert_eq!(models_using::<StrlenOptions>(&models, &paths), ["a.json", "c.json"]);
        assert_eq!(models_options::<StrlenOptions>(&models, &paths).unwrap(), new);
        assert_eq!(
            models_options::<PileupOptions>(&models[..2], &paths).unwrap(),
            Some(PileupOptions::default())
        );
        let message = models_options::<PileupOptions>(&models, &paths).unwrap_err().to_string();
        assert!(
            message.contains("c.json needs mapq30,bq10,flank5, another model mapq20,bq10,flank5"),
            "{message}"
        );
        assert_eq!(models_options::<PileupOptions>(&models[1..2], &paths).unwrap(), None);

        let users = models_using::<PileupOptions>(&models, &paths);
        let tables = [PathBuf::from("p.tsv")];
        assert!(check_supplied::<PileupOptions>(&users, &tables, false).is_ok());
        assert!(check_supplied::<PileupOptions>(&users, &[], true).is_ok());
        assert!(check_supplied::<PileupOptions>(&[], &tables, false).is_ok());
        let message = check_supplied::<PileupOptions>(&users, &[], false).unwrap_err().to_string();
        assert!(message.starts_with("models use pileup features; pass --pileup"), "{message}");
        assert!(message.ends_with("a.json, c.json"), "{message}");
        // No tables, or no model using them: nothing is read.
        assert_eq!(read_for_models::<PileupOptions>(&[], &models, &paths).unwrap(), None);
        assert_eq!(read_for_models::<PileupOptions>(&tables, &models[1..2], &paths).unwrap(), None);
    }
}
