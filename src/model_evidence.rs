//! The read evidence models use (repeat-length and pileup evidence): the settings training
//! tables share, the checks that evidence was measured as models need, and measuring it from the
//! reads while scoring.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use log::info;

use crate::alignment_provenance::{AlignmentProvenance, AlignmentTags};
use crate::common::PhaseTimer;
use crate::evidence::{Evidence, EvidenceTables, read_tables_checked};
use crate::model::{ModelInfo, Predicting, WinnowrModel};
use crate::pileup::{PileupOptions, PileupTables, measure_pileup};
use crate::pileup_features::add_pileup_features;
use crate::policy::{
    CallKey, Decision, Policy, check_model_kinds, decide_model_calls, merge_decisions,
};
use crate::read_pass::ReadInputs;
use crate::repeat_features::add_strlen_features;
use crate::strlen::{StrlenOptions, StrlenRow, StrlenTables, measure_regions, sites_to_rows};
use crate::table::{FeatureTable, FeatureTableInfo};

/// Checks that evidence measured with `measured` (from `source`) can feed a model trained on
/// evidence measured with `trained`.
///
/// # Errors
///
/// Returns an error, saying how to measure the evidence again, if the settings differ.
pub fn check_options<E: Evidence>(
    measured: E,
    source: &str,
    trained: E,
    model: &Path,
) -> Result<()> {
    if measured != trained {
        bail!(
            "{source} was measured with {name} {settings} {measured}, but model {} was trained on \
             evidence measured with {trained}: measure again with `winnowr {name} {option} \
             {trained}` (or let `score --alignments` measure with the model's settings)",
            model.display(),
            name = E::NAME,
            settings = E::SETTINGS,
            option = E::OPTION,
        );
    }
    Ok(())
}

/// A kind of read evidence as models use it: which models and feature tables have it, and with
/// what settings.
pub trait ModelEvidence: Evidence {
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
pub fn shared_options<E: ModelEvidence>(infos: &[&FeatureTableInfo]) -> Result<E> {
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
pub fn check_table<E: ModelEvidence>(
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
pub fn models_using<E: ModelEvidence>(models: &[&ModelInfo], paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .zip(models)
        .filter(|(_, info)| E::of_model(info).is_some())
        .map(|(path, _)| path.display().to_string())
        .collect()
}

/// Checks that `score` is given the evidence the models use (`users`, see [`models_using`]): its
/// `tables`, or reads to measure it from, and warns about tables no model uses.
///
/// # Errors
///
/// Returns an error if models use the evidence and neither tables nor reads are given.
pub fn check_supplied<E: ModelEvidence>(
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
        log::warn!(
            "no model uses {}; --{} is ignored for features (it may still be read for alignment provenance)",
            E::FEATURES,
            E::NAME
        );
    }
    Ok(())
}

/// The settings the evidence the models use must be measured with (`None` if no model uses it).
///
/// # Errors
///
/// Returns an error if two models need it measured differently.
pub fn models_options<E: ModelEvidence>(
    models: &[&ModelInfo],
    paths: &[PathBuf],
) -> Result<Option<E>> {
    let mut options: Option<E> = None;
    for (path, info) in paths.iter().zip(models) {
        match (options, E::of_model(info)) {
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
pub fn read_for_models<E: ModelEvidence>(
    tables: &[PathBuf],
    models: &[WinnowrModel],
    paths: &[PathBuf],
    tags: Option<&AlignmentTags>,
    check: impl FnMut(&Path, Option<&AlignmentProvenance>) -> Result<()>,
) -> Result<Option<EvidenceTables<E>>> {
    if tables.is_empty() || models.iter().all(|model| E::of_model(&model.info).is_none()) {
        return Ok(None);
    }
    let read = read_tables_checked::<E>(tables, tags, check)?;
    let source = format!("--{} {}", E::NAME, tables[0].display());
    for (path, model) in paths.iter().zip(models) {
        if let Some(trained) = E::of_model(&model.info) {
            check_options(read.options, &source, trained, path)?;
        }
    }
    Ok(Some(read))
}

/// What `score --alignments` measures: the reads, the settings and the contigs.
pub struct Measuring<'a> {
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

/// The evidence measured from the reads: the repeat-length rows and the pileup tables, of the
/// settings asked for.
type Measured = (Option<Vec<StrlenRow>>, Option<PileupTables>);

/// Measures the evidence of `strlen` and `pileup` settings from the reads, over `contigs`, on the
/// current rayon pool.
///
/// # Errors
///
/// Returns an error if the reads cannot be measured.
fn measure_evidence(
    inputs: &ReadInputs<'_>,
    contigs: &[String],
    strlen: Option<StrlenOptions>,
    pileup: Option<PileupOptions>,
) -> Result<Measured> {
    let strlen_rows = strlen
        .map(|options| {
            Ok::<_, anyhow::Error>(sites_to_rows(&measure_regions(inputs, contigs, &options)?))
        })
        .transpose()?;
    let pileup = pileup
        .map(|options| {
            let rows = measure_pileup(inputs, contigs, &options)?;
            Ok::<_, anyhow::Error>(PileupTables::new(options, rows))
        })
        .transpose()?;
    Ok((strlen_rows, pileup))
}

/// Scores calls with evidence measured from the reads, on the threads of the current rayon pool
/// (`score`'s `-t`): the evidence is measured while the models that do not
/// use it score the calls, then joined for the models that do. While both run, they share the
/// pool: the shards of the reads and chunks of the models' calls are jobs any idle thread takes
/// up, so neither side leaves threads idle while the other has work. With one thread, or when
/// every model uses the evidence, the evidence is measured first, then every model scores.
/// Decisions are merged in model order, as [`crate::policy::decide_calls`] does. The models that
/// score while the reads are measured predict in chunks with one thread each (see
/// [`Predicting::Chunked`]), then their boosters get their own threads back; the others predict
/// with their boosters' threads (as `score` loads them, `-t`).
///
/// The two sides are the arms of one `rayon::join`, scoring first: the pool thread that takes up
/// the join scores, while the measurement waits in that thread's queue for an idle thread to
/// take it up (called from outside the pool, as `score` calls it, the join is first handed to the
/// pool). When scoring fails, it stops the measurement before its next shard of the reads, waits
/// for it (so that no read is being read as the process exits), then returns the error: within
/// about a shard's time, rather than after the whole pass over the reads. Likewise a failed
/// measurement is reported before the next model scores. How long each side took is logged at
/// info.
///
/// # Errors
///
/// Returns an error if the reads cannot be measured or a model cannot score the calls (that
/// error first, when both fail), or a booster cannot be given its threads.
pub fn measure_and_decide(
    table: FeatureTable,
    models: &mut [WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    timer: &mut PhaseTimer,
) -> Result<HashMap<CallKey, Decision>> {
    let (decisions, _) =
        measure_and_decide_with(table, models, policy, measure, timer, Hooks::default())?;
    Ok(decisions)
}

/// Calls made as [`measure_and_decide`] runs, while it measures the reads as models score, for
/// tests to watch or hold it (none otherwise).
#[derive(Default)]
struct Hooks {
    /// Called on the measurement's thread before it measures, with the flag that stops it.
    before_measuring: Option<BeforeMeasuring>,
    /// Called on the measurement's thread once it has measured, told whether it measured the
    /// evidence.
    after_measuring: Option<Box<dyn FnOnce(bool) + Send>>,
    /// Called on the scoring thread before each model that does not need the evidence scores.
    before_scoring: Option<BeforeScoring>,
}

/// A [`Hooks::before_measuring`] call.
type BeforeMeasuring = Box<dyn FnOnce(&AtomicBool) + Send>;

/// A [`Hooks::before_scoring`] call.
type BeforeScoring = Box<dyn Fn(&WinnowrModel) + Send + Sync>;

/// [`measure_and_decide`], calling `hooks`, and returning whether it measured the reads while
/// models scored (with more than one thread and a model that does not need the evidence), else
/// measured them first.
fn measure_and_decide_with(
    table: FeatureTable,
    models: &mut [WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    timer: &mut PhaseTimer,
    hooks: Hooks,
) -> Result<(HashMap<CallKey, Decision>, bool)> {
    check_model_kinds(models.iter().map(|model| &model.info))?;
    // The pool the work runs on (called from outside a pool, the global one).
    let threads = rayon::current_num_threads();
    let shared = threads > 1 && models.iter().any(|model| !measure.is_needed_by(model));
    let (mut per_model, table) = if shared {
        info!(
            "Measuring the evidence from the reads while scoring the models that do not need it, \
             sharing -t {threads}"
        );
        measure_while_scoring(table, models, policy, measure, timer, hooks)?
    } else {
        info!("Measuring the evidence from the reads on -t {threads}, then scoring");
        let started = Instant::now();
        let measured =
            measure_evidence(measure.inputs, measure.contigs, measure.strlen, measure.pileup)?;
        info!("The reads were measured in {:.1?}", started.elapsed());
        timer.lap("measuring the evidence from the reads");
        (vec![None; models.len()], join_evidence(table, measure, measured)?)
    };
    for (slot, model) in per_model.iter_mut().zip(models.iter()) {
        if slot.is_none() {
            *slot = Some(decide_model_calls(&table, model, policy, Predicting::Whole)?);
        }
    }
    Ok((merge_decisions(per_model.into_iter().flatten().collect()), shared))
}

/// Each model's decisions, `None` for those yet to score.
type PerModel = Vec<Option<Vec<(CallKey, Decision)>>>;

/// The evidence joined to the `table`, for the models that use it.
///
/// # Errors
///
/// Returns an error if the evidence cannot be joined.
fn join_evidence(
    table: FeatureTable,
    measure: &Measuring<'_>,
    measured: Measured,
) -> Result<FeatureTable> {
    let (strlen_rows, pileup) = measured;
    let mut table = table;
    if let (Some(rows), Some(options)) = (strlen_rows, measure.strlen) {
        table = add_strlen_features(table, &StrlenTables::new(options, rows))?;
    }
    if let Some(pileup) = pileup {
        table = add_pileup_features(table, &pileup)?;
    }
    Ok(table)
}

/// Measures the evidence while the models that do not use it score, with one thread each (see
/// [`measure_and_score`]), then gives their boosters their own threads back, and logs how long
/// each side took. Returns the decisions of the models that do not use the evidence, and the table
/// with the evidence.
///
/// # Errors
///
/// Returns an error if a model cannot score the calls, else if the reads cannot be measured, else
/// if a booster cannot be given its threads.
fn measure_while_scoring(
    table: FeatureTable,
    models: &mut [WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    timer: &mut PhaseTimer,
    hooks: Hooks,
) -> Result<(PerModel, FeatureTable)> {
    measure_while_scoring_with_threads(
        table,
        models,
        policy,
        measure,
        timer,
        hooks,
        winnowr_xgboost::Booster::set_threads,
    )
}

/// The shared work with a supplied thread setter, for exercising real setter failures in cleanup.
fn measure_while_scoring_with_threads<F>(
    table: FeatureTable,
    models: &mut [WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    timer: &mut PhaseTimer,
    hooks: Hooks,
    mut set_threads: F,
) -> Result<(PerModel, FeatureTable)>
where
    F: FnMut(&mut winnowr_xgboost::Booster, NonZeroUsize) -> winnowr_xgboost::Result<()>,
{
    // The models given one thread, by index, with the threads their boosters had.
    let mut own: Vec<(usize, NonZeroUsize)> = Vec::new();
    // Catch only to restore settings, then resume the original panic after both arms have joined.
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let one_each = models
            .iter_mut()
            .enumerate()
            .filter(|(_, model)| !measure.is_needed_by(model))
            .filter(|(_, model)| model.booster.threads() != NonZeroUsize::MIN)
            .try_for_each(|(index, model)| {
                let threads = model.booster.threads();
                set_threads(&mut model.booster, NonZeroUsize::MIN)?;
                own.push((index, threads));
                Ok::<_, winnowr_xgboost::Error>(())
            });
        // Nothing runs if a booster cannot be given one thread.
        one_each.map(|()| measure_and_score(&table, models, policy, measure, hooks))
    }));
    // Every booster given one thread gets its own back, the first failure kept.
    let given_back = own
        .iter()
        .map(|&(index, threads)| set_threads(&mut models[index].booster, threads))
        .fold(Ok(()), Result::and);
    let ran = match ran {
        Ok(ran) => ran,
        Err(panic) => std::panic::resume_unwind(panic),
    };
    let (scored, (measured, measured_for)) = ran?;
    let (per_model, scored_for) = scored?;
    let measured = measured?;
    given_back?;
    info!(
        "The reads were measured in {:.1?}, while the models that do not need them scored in \
         {:.1?}",
        measured_for, scored_for
    );
    timer.lap("measuring the evidence from the reads (while scoring other models)");
    Ok((per_model, join_evidence(table, measure, measured)?))
}

/// What [`measure_and_score`] returns: the decisions of the models that do not use the evidence
/// and how long they took to score, and the evidence and how long it took to measure.
type MeasuredAndScored = (Result<(PerModel, Duration)>, (Result<Measured>, Duration));

/// Measures the evidence while the models that do not use it score, as the two arms of one
/// `rayon::join` on the current pool (see [`measure_and_decide`]), calling `hooks`. Each side
/// raises a flag when it ends early, on an error or a panic: scoring, the flag that stops the
/// measurement before its next shard; the measurement, the flag that stops scoring before its
/// next model. The join returns once both are done (and re-raises a panic of either).
fn measure_and_score(
    table: &FeatureTable,
    models: &mut [WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    hooks: Hooks,
) -> MeasuredAndScored {
    measure_and_score_timed(table, models, policy, measure, hooks, Instant::now)
}

/// The joined work with a supplied clock, so timing can be checked without wall-clock delays.
fn measure_and_score_timed<C: Fn() -> Instant + Sync>(
    table: &FeatureTable,
    models: &mut [WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    hooks: Hooks,
    now: C,
) -> MeasuredAndScored {
    let now = &now;
    let Hooks { before_measuring, after_measuring, before_scoring } = hooks;
    let (stop, failed) = (AtomicBool::new(false), AtomicBool::new(false));
    let (stop, failed) = (&stop, &failed);
    let inputs = measure.inputs.clone().stopped_by_scoring(stop);
    let before_scoring = before_scoring.as_deref();
    // A booster is `Send`, not `Sync`: the models move to the scoring arm (mutably borrowed,
    // which is `Send`).
    let scoring: &mut [WinnowrModel] = models;
    // Each arm is timed from its own start: the measurement may start well after scoring.
    rayon::join(
        // Scoring is the first arm, run at once by the thread that takes up the join.
        move || {
            let started = now();
            let stopping = RaiseUnlessDone(stop);
            let scored = score_meanwhile(table, scoring, policy, measure, failed, before_scoring);
            if scored.is_ok() {
                stopping.done();
            }
            scored.map(|per_model| (per_model, now().duration_since(started)))
        },
        // The measurement is the second, which an idle thread takes up.
        move || {
            let started = now();
            // (`failed` is raised, unless done, before `after_measuring` is called.)
            let measured = {
                let failing = RaiseUnlessDone(failed);
                if let Some(hook) = before_measuring {
                    hook(stop);
                }
                let measured =
                    measure_evidence(&inputs, measure.contigs, measure.strlen, measure.pileup);
                if measured.is_ok() {
                    failing.done();
                }
                measured
            };
            if let Some(hook) = after_measuring {
                hook(measured.is_ok());
            }
            (measured, now().duration_since(started))
        },
    )
}

/// Scores the calls with the models that do not use the evidence being measured, predicting in
/// chunks, until a model fails or the measurement has `failed` (a failed measurement is reported
/// before the next model scores), calling `before_scoring` before each.
///
/// # Errors
///
/// Returns an error if a model cannot score the calls.
fn score_meanwhile(
    table: &FeatureTable,
    models: &[WinnowrModel],
    policy: &Policy,
    measure: &Measuring<'_>,
    failed: &AtomicBool,
    before_scoring: Option<&(dyn Fn(&WinnowrModel) + Send + Sync)>,
) -> Result<PerModel> {
    let mut per_model: PerModel = vec![None; models.len()];
    for (slot, model) in per_model.iter_mut().zip(models) {
        if measure.is_needed_by(model) {
            continue;
        }
        if failed.load(Ordering::Relaxed) {
            break;
        }
        if let Some(hook) = before_scoring {
            hook(model);
        }
        *slot = Some(decide_model_calls(table, model, policy, Predicting::Chunked)?);
    }
    Ok(per_model)
}

/// Raises a flag when dropped before [`RaiseUnlessDone::done`] is called (on an error or a
/// panic): tells the other side of a join to stop.
struct RaiseUnlessDone<'a>(&'a AtomicBool);

impl RaiseUnlessDone<'_> {
    /// Done: the flag is not raised.
    fn done(self) {
        std::mem::forget(self);
    }
}

impl Drop for RaiseUnlessDone<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Observes enclosing real jobs, including synchronous native work, for registered test operations.
/// This is not native-body instrumentation: extra native pools must remain disabled by the
/// one-thread prediction path and the readers' default single-threaded htslib configuration.
#[cfg(test)]
pub mod work_observer {
    use std::collections::{HashMap, HashSet};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
    use std::thread::ThreadId;
    use std::time::{Duration, Instant};

    #[derive(Clone, PartialEq, Eq, Hash)]
    enum Key {
        Reads(PathBuf),
        Feature(String),
    }

    fn registry() -> &'static Mutex<HashMap<Key, Weak<Observer>>> {
        static REGISTRY: OnceLock<Mutex<HashMap<Key, Weak<Observer>>>> = OnceLock::new();
        REGISTRY.get_or_init(Mutex::default)
    }

    /// Removes only this operation's keys, even if its test unwinds.
    pub struct Registration(Vec<Key>);

    impl Drop for Registration {
        fn drop(&mut self) {
            let mut registry = registry().lock().unwrap();
            for key in &self.0 {
                registry.remove(key);
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Work {
        ReadShard,
        Prediction(usize),
    }

    #[derive(Default)]
    pub struct Seen {
        pub active: HashMap<ThreadId, usize>,
        reads: usize,
        predictions: usize,
        pub peak: usize,
        pub overlapped: bool,
        pub read_jobs: usize,
        pub prediction_jobs: usize,
        pub prediction_threads: Vec<usize>,
        pub workers: HashSet<ThreadId>,
        pub timed_out: bool,
        released: bool,
    }

    pub struct Observer {
        pub seen: Mutex<Seen>,
        changed: Condvar,
        budget: usize,
    }

    impl Observer {
        pub fn new(budget: usize) -> Arc<Self> {
            Arc::new(Self { seen: Mutex::new(Seen::default()), changed: Condvar::new(), budget })
        }

        /// Follows the unique fixture's reads and model, not the pool expected to run them.
        pub fn register(self: &Arc<Self>, vcf: &Path, feature: &str) -> Registration {
            let keys = vec![Key::Reads(vcf.to_path_buf()), Key::Feature(feature.to_string())];
            let mut registry = registry().lock().unwrap();
            for key in &keys {
                assert!(registry.insert(key.clone(), Arc::downgrade(self)).is_none());
            }
            Registration(keys)
        }
    }

    pub struct Running {
        observer: Arc<Observer>,
        id: ThreadId,
        work: Work,
    }

    pub fn read_shard(vcf: &Path) -> Option<Running> {
        enter(Key::Reads(vcf.to_path_buf()), Work::ReadShard)
    }

    pub fn prediction<S: AsRef<str>>(features: &[S], threads: usize) -> Option<Running> {
        // The budget fixture trains on a unique, single named feature.
        enter(Key::Feature(features.first()?.as_ref().to_string()), Work::Prediction(threads))
    }

    /// Enters an actual job; the first wave waits, with a deadline, for both kinds to fill the pool.
    fn enter(key: Key, work: Work) -> Option<Running> {
        let observer = registry().lock().unwrap().get(&key)?.upgrade()?;
        let id = std::thread::current().id();
        let mut seen = observer.seen.lock().unwrap();
        seen.workers.insert(id);
        *seen.active.entry(id).or_default() += 1;
        match work {
            Work::ReadShard => {
                seen.reads += 1;
                seen.read_jobs += 1;
            }
            Work::Prediction(threads) => {
                seen.predictions += 1;
                seen.prediction_jobs += 1;
                seen.prediction_threads.push(threads);
            }
        }
        seen.peak = seen.peak.max(seen.active.len());
        seen.overlapped |= seen.reads > 0 && seen.predictions > 0;
        if seen.active.len() >= observer.budget && seen.overlapped {
            seen.released = true;
            observer.changed.notify_all();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while !seen.released {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                seen.timed_out = true;
                seen.released = true;
                observer.changed.notify_all();
                break;
            }
            seen = observer.changed.wait_timeout(seen, remaining).unwrap().0;
        }
        drop(seen);
        Some(Running { observer, id, work })
    }

    impl Drop for Running {
        fn drop(&mut self) {
            let mut seen = self.observer.seen.lock().unwrap();
            let count = seen.active.get_mut(&self.id).unwrap();
            *count -= 1;
            if *count == 0 {
                seen.active.remove(&self.id);
            }
            match self.work {
                Work::ReadShard => seen.reads -= 1,
                Work::Prediction(_) => seen.predictions -= 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::common::{Task, VariantType};

    #[test]
    fn training_tables_must_share_their_strlen_settings() {
        let info = |sample: &str, strlen: bool, settings: Option<StrlenOptions>| FeatureTableInfo {
            sample: sample.to_string(),
            strlen,
            strlen_settings: settings,
            ..FeatureTableInfo::default()
        };
        let new = Some(StrlenOptions::default());
        let (a, b) = (info("A", true, new), info("B", true, new));
        assert_eq!(shared_options::<StrlenOptions>(&[&a, &b]).unwrap(), StrlenOptions::default());
        // Tables without recorded settings are read as measured with the legacy settings.
        let (c, d) = (info("C", true, None), info("D", true, None));
        assert_eq!(shared_options::<StrlenOptions>(&[&c, &d]).unwrap(), StrlenOptions::LEGACY);
        let message = shared_options::<StrlenOptions>(&[&a, &c]).unwrap_err().to_string();
        assert!(message.contains("A's with mapq20,flank5, C's with mapq0,flank2"), "{message}");
        let none = info("E", false, None);
        let message = shared_options::<StrlenOptions>(&[&a, &none]).unwrap_err().to_string();
        assert!(message.contains("E's has no strlen evidence"), "{message}");
    }

    #[test]
    fn training_tables_must_share_their_pileup_settings() {
        let info = |sample: &str, settings: Option<&str>| FeatureTableInfo {
            sample: sample.to_string(),
            pileup_settings: settings.map(|s| s.parse().unwrap()),
            ..FeatureTableInfo::default()
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
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Indel);
        model.info.stutter = strlen.map(|_| crate::testing::fit_stutter_model(&[]));
        model.info.strlen_settings = strlen;
        model.info.pileup_settings = pileup.map(|s| s.parse().unwrap());
        model
    }

    fn table_info(strlen: Option<StrlenOptions>, pileup: Option<&str>) -> FeatureTableInfo {
        FeatureTableInfo {
            sample: "S".to_string(),
            strlen: strlen.is_some(),
            strlen_settings: strlen,
            pileup_settings: pileup.map(|s| s.parse().unwrap()),
            ..FeatureTableInfo::default()
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
        let infos: Vec<&ModelInfo> = models.iter().map(|model| &model.info).collect();
        assert_eq!(models_using::<StrlenOptions>(&infos, &paths), ["a.json", "c.json"]);
        assert_eq!(models_options::<StrlenOptions>(&infos, &paths).unwrap(), new);
        assert_eq!(
            models_options::<PileupOptions>(&infos[..2], &paths).unwrap(),
            Some(PileupOptions::default())
        );
        let message = models_options::<PileupOptions>(&infos, &paths).unwrap_err().to_string();
        assert!(
            message.contains("c.json needs mapq30,bq10,flank5, another model mapq20,bq10,flank5"),
            "{message}"
        );
        assert_eq!(models_options::<PileupOptions>(&infos[1..2], &paths).unwrap(), None);

        let users = models_using::<PileupOptions>(&infos, &paths);
        let tables = [PathBuf::from("p.tsv")];
        assert!(check_supplied::<PileupOptions>(&users, &tables, false).is_ok());
        assert!(check_supplied::<PileupOptions>(&users, &[], true).is_ok());
        assert!(check_supplied::<PileupOptions>(&[], &tables, false).is_ok());
        let message = check_supplied::<PileupOptions>(&users, &[], false).unwrap_err().to_string();
        assert!(message.starts_with("models use pileup features; pass --pileup"), "{message}");
        assert!(message.ends_with("a.json, c.json"), "{message}");
        // No tables, or no model using them: nothing is read.
        assert_eq!(
            read_for_models::<PileupOptions>(&[], &models, &paths, None, |_, _| Ok(())).unwrap(),
            None
        );
        assert_eq!(
            read_for_models::<PileupOptions>(&tables, &models[1..2], &paths, None, |_, _| Ok(()))
                .unwrap(),
            None
        );
    }

    /// The reads, calls and models of the `measure_and_decide` tests.
    struct Fixture {
        _dir: tempfile::TempDir,
        files: crate::testing::ReadFiles,
    }

    /// A SNP model of the pileup feature `pu_alt_mqhigh`, whose class is the number of alternate
    /// reads at MAPQ 20 or more (drop, keep, flip).
    fn pileup_snp_model() -> WinnowrModel {
        crate::testing::separable_model_of(Task::Genotype, VariantType::Snp, "pu_alt_mqhigh")
    }

    /// The pileup SNP model (which needs the evidence), and an indel model of `feature` (which
    /// does not).
    fn snp_and_indel_of(feature: &str) -> [WinnowrModel; 2] {
        let indel = crate::testing::separable_model_of(Task::Genotype, VariantType::Indel, feature);
        [pileup_snp_model(), indel]
    }

    /// The pileup SNP model, and an indel model that scores the calls.
    fn snp_and_indel() -> [WinnowrModel; 2] {
        snp_and_indel_of("x")
    }

    impl Fixture {
        /// An A10 tract in 80 bp flanks; SNP calls at 123, 131 and 139, with one alternate read at
        /// 123, two at 131 and none at 139 (all at MAPQ 60).
        fn new() -> Self {
            use crate::testing::{ReadFiles, TestRead};
            let dir = tempfile::TempDir::new().unwrap();
            // 80 bp of flank, an A10 tract and 80 bp of flank; 0-based 122, 130 and 138 are Gs.
            let reference = format!("{}{}{}", "CGTC".repeat(20), "A".repeat(10), "GCTG".repeat(20));
            let records = [
                "chr1\t123\t.\tG\tT\t50\tPASS\t.\tGT\t0/1",
                "chr1\t131\t.\tG\tT\t50\tPASS\t.\tGT\t0/1",
                "chr1\t139\t.\tG\tT\t50\tPASS\t.\tGT\t0/1",
            ];
            let with_alts = |positions: &[usize]| {
                let mut bases = reference.as_bytes()[90..160].to_vec();
                for &pos in positions {
                    bases[pos - 90] = b'T';
                }
                String::from_utf8(bases).unwrap()
            };
            let (both, one) = (with_alts(&[122, 130]), with_alts(&[130]));
            let read = |name, bases| TestRead {
                name,
                pos: 90,
                cigar: "70M",
                flag: 0,
                mapq: 60,
                qual: 30,
                quals: None,
                bases: Some(bases),
            };
            let reads = [read("both", &both), read("one", &one)];
            let sequences = [("chr1", reference.as_str()), ("chr2", "ACGT")];
            let mut files = ReadFiles::write(dir.path(), &sequences, &records, &reads);
            files.bam = crate::testing::write_bam_with_header(
                dir.path(),
                "reads.bam",
                reference.len() as u64,
                &reads,
                &["@PG\tID:minimap2\tPN:minimap2\tVN:2.28\tCL:minimap2 -ax map-hifi reads.fq"],
            );
            Fixture { _dir: dir, files }
        }

        fn inputs(&self) -> ReadInputs<'_> {
            self.files.inputs()
        }

        /// The calls: the three SNPs, and an indel sharing its key with the SNP at 131, so that
        /// both a SNP and an indel model decide it.
        fn table() -> FeatureTable {
            let calls = [
                ("chr1", 123, "G", "T", "PASS", "snp", "0/1", false),
                ("chr1", 131, "G", "T", "PASS", "snp", "0/1", false),
                ("chr1", 139, "G", "T", "PASS", "snp", "0/1", false),
                ("chr1", 131, "G", "T", "PASS", "indel", "0/1", false),
            ];
            crate::testing::call_table(&calls)
                .with_f64_column("x", vec![0.0; calls.len()])
                .unwrap()
                .with_bool_column("flippable", vec![true; calls.len()])
                .unwrap()
        }

        /// The decisions of `models` on the calls with the pileup evidence measured beforehand.
        fn expected(&self, models: &[WinnowrModel]) -> HashMap<CallKey, Decision> {
            let options = PileupOptions::default();
            let rows = measure_pileup(&self.inputs(), &["chr1"], &options).unwrap();
            let table =
                add_pileup_features(Self::table(), &PileupTables::new(options, rows)).unwrap();
            let policy = Policy::new(crate::rescore::Mode::Genotyper);
            crate::policy::decide_calls(&table, models, &policy).unwrap()
        }

        /// Measures the pileup evidence while scoring with `models` on a pool of `threads`
        /// threads, calling `hooks`.
        fn measure_and_decide(
            &self,
            models: &mut [WinnowrModel],
            threads: usize,
            hooks: Hooks,
        ) -> Result<(HashMap<CallKey, Decision>, bool)> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| self.measure_and_decide_here(models, hooks))
        }

        /// [`Fixture::measure_and_decide`] on the current pool (the global pool, called from
        /// outside a pool).
        fn measure_and_decide_here(
            &self,
            models: &mut [WinnowrModel],
            hooks: Hooks,
        ) -> Result<(HashMap<CallKey, Decision>, bool)> {
            let inputs = self.inputs();
            let contigs = ["chr1".to_string()];
            let measuring = Measuring {
                inputs: &inputs,
                strlen: None,
                pileup: Some(PileupOptions::default()),
                contigs: &contigs,
            };
            let policy = Policy::new(crate::rescore::Mode::Genotyper);
            measure_and_decide_with(
                Self::table(),
                models,
                &policy,
                &measuring,
                &mut PhaseTimer::start(),
                hooks,
            )
        }

        /// [`measure_and_score`] with `models` on a pool of one thread, calling `hooks`: the
        /// models score, then the reads are measured.
        fn measure_and_score_on_one_thread<C: Fn() -> Instant + Sync + Send>(
            &self,
            models: &mut [WinnowrModel],
            hooks: Hooks,
            now: C,
        ) -> MeasuredAndScored {
            let inputs = self.inputs();
            let contigs = ["chr1".to_string()];
            let measuring = Measuring {
                inputs: &inputs,
                strlen: None,
                pileup: Some(PileupOptions::default()),
                contigs: &contigs,
            };
            let policy = Policy::new(crate::rescore::Mode::Genotyper);
            let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
            pool.install(|| {
                measure_and_score_timed(&Self::table(), models, &policy, &measuring, hooks, now)
            })
        }
    }

    /// Each arm starts its own clock, even when measurement takes longer than scoring.
    #[test]
    fn measuring_and_scoring_are_each_timed_from_their_own_start() {
        use std::sync::atomic::AtomicU64;
        let fixture = Fixture::new();
        let mut models = snp_and_indel();
        let seconds = Arc::new(AtomicU64::new(0));
        let scoring = Arc::clone(&seconds);
        let measuring = Arc::clone(&seconds);
        let measured = Arc::clone(&seconds);
        let hooks = Hooks {
            before_scoring: Some(Box::new(move |_| {
                scoring.store(10, Ordering::Relaxed);
            })),
            before_measuring: Some(Box::new(move |_| {
                measuring.store(100, Ordering::Relaxed);
            })),
            after_measuring: Some(Box::new(move |_| {
                measured.store(200, Ordering::Relaxed);
            })),
        };
        let epoch = Instant::now();
        let now = || epoch + Duration::from_secs(seconds.load(Ordering::Relaxed));
        let (scored, (measured, measured_for)) =
            fixture.measure_and_score_on_one_thread(&mut models, hooks, now);
        let (_, scored_for) = scored.unwrap();
        measured.unwrap();
        assert_eq!(scored_for, Duration::from_secs(10));
        // The second arm starts at 10, not at the first arm's 0; no filesystem speed assumption.
        assert_eq!(measured_for, Duration::from_secs(190));
    }

    /// The threads of the models' boosters.
    fn booster_threads(models: &[WinnowrModel]) -> Vec<usize> {
        models.iter().map(|model| model.booster.threads().get()).collect()
    }

    /// Gives the models' boosters 4 threads, as `score -t 4` loads them.
    fn with_four_threads(models: &mut [WinnowrModel]) {
        for model in models {
            model.booster.set_threads(NonZeroUsize::new(4).unwrap()).unwrap();
        }
    }

    /// How long a test waits for something before it fails, rather than hanging.
    const PATIENCE: Duration = Duration::from_secs(10);

    /// Waits until `done` holds, for at most [`PATIENCE`]; returns whether it held.
    fn wait_until(done: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + PATIENCE;
        while !done() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        done()
    }

    /// `score --alignments` decides as `score --pileup` does with the same evidence measured
    /// beforehand, including which model's decision a call scored by two models keeps: the model
    /// that uses the evidence is first, though it scores after the one that does not. So it does
    /// with every thread count, measuring while scoring with more than one, and the boosters get
    /// their threads back.
    #[rstest::rstest]
    fn measuring_while_scoring_decides_as_scoring_measured_evidence(
        #[values(1, 2, 3, 8)] threads: usize,
    ) {
        let fixture = Fixture::new();
        let mut models = snp_and_indel();
        assert_eq!(models[0].info.pileup_settings, Some(PileupOptions::default()));
        let expected = fixture.expected(&models);
        with_four_threads(&mut models);
        let (got, shared) =
            fixture.measure_and_decide(&mut models, threads, Hooks::default()).unwrap();
        assert_eq!(got, expected);
        assert_eq!(booster_threads(&models), [4, 4]);
        assert_eq!(shared, threads > 1);

        // A matching strict expectation must succeed through the same worker path.
        let inputs = fixture.inputs();
        let checked_header =
            crate::alignment_provenance::HeaderProvenance::read(inputs.alignments).unwrap();
        let checked = checked_header.provenance.as_ref().unwrap();
        let matching_inputs = ReadInputs {
            expected_alignment: Some(crate::read_pass::AlignmentExpectation {
                provenance: Some(checked),
                header_records: &checked_header.header_records,
                tags: None,
                strict: true,
            }),
            ..inputs
        };
        let contigs = ["chr1".to_string()];
        let matching_measure = Measuring {
            inputs: &matching_inputs,
            strlen: None,
            pileup: Some(PileupOptions::default()),
            contigs: &contigs,
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let matching = pool
            .install(|| {
                measure_and_decide(
                    Fixture::table(),
                    &mut models,
                    &Policy::new(crate::rescore::Mode::Genotyper),
                    &matching_measure,
                    &mut PhaseTimer::start(),
                )
            })
            .unwrap();
        assert_eq!(matching, expected);

        // The measurement thread must retain the provenance checked before scoring.
        let wrong = crate::testing::bwa_provenance("primary");
        let inputs = fixture.inputs();
        let changed_inputs = ReadInputs {
            expected_alignment: Some(crate::read_pass::AlignmentExpectation {
                provenance: Some(&wrong),
                header_records: &checked_header.header_records,
                tags: None,
                strict: true,
            }),
            ..inputs
        };
        let contigs = ["chr1".to_string()];
        let changed_measure = Measuring {
            inputs: &changed_inputs,
            strlen: None,
            pileup: Some(PileupOptions::default()),
            contigs: &contigs,
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let changed = pool.install(|| {
            measure_and_decide(
                Fixture::table(),
                &mut models,
                &Policy::new(crate::rescore::Mode::Genotyper),
                &changed_measure,
                &mut PhaseTimer::start(),
            )
        });
        let error = format!("{:#}", changed.unwrap_err());
        assert!(error.contains("changed between preflight and reading its alignments"), "{error}");
        assert!(error.contains(&fixture.files.bam.display().to_string()), "{error}");
        assert!(error.contains(&format!("expected {wrong}, observed {checked}")), "{error}");

        let mut actions: Vec<(u64, Task, bool, bool)> =
            got.iter().map(|(key, d)| (key.1, d.task, d.drop, d.flip)).collect();
        actions.sort_unstable_by_key(|action| action.0);
        // 131 is flipped by the SNP model, not dropped by the indel model.
        assert_eq!(
            actions,
            [
                (123, Task::Genotype, false, false),
                (131, Task::Genotype, false, true),
                (139, Task::Genotype, true, false),
            ]
        );
    }

    /// When every model needs the evidence, it is measured first, then they score.
    #[rstest::rstest]
    fn every_model_needing_the_evidence_measures_first(#[values(1, 2, 8)] threads: usize) {
        let fixture = Fixture::new();
        let mut models = [pileup_snp_model()];
        let expected = fixture.expected(&models);
        let (got, shared) =
            fixture.measure_and_decide(&mut models, threads, Hooks::default()).unwrap();
        assert_eq!(got, expected);
        assert!(!shared);
    }

    /// A failing measurement and a failing model each surface, measuring first or while scoring,
    /// and the boosters get their threads back.
    #[rstest::rstest]
    fn measuring_and_scoring_errors_surface(#[values(1, 4)] threads: usize) {
        let mut fixture = Fixture::new();
        // A model of a feature the table lacks cannot score.
        let mut models = snp_and_indel_of("absent");
        with_four_threads(&mut models);
        let error = fixture.measure_and_decide(&mut models, threads, Hooks::default()).unwrap_err();
        assert!(format!("{error:#}").contains("absent"), "{error:#}");
        assert_eq!(booster_threads(&models), [4, 4]);
        // Reads that cannot be read cannot be measured.
        fixture.files.bam = fixture.files.bam.with_file_name("missing.bam");
        let mut models = snp_and_indel();
        with_four_threads(&mut models);
        let error = fixture.measure_and_decide(&mut models, threads, Hooks::default()).unwrap_err();
        assert!(format!("{error:#}").contains("missing.bam"), "{error:#}");
        assert_eq!(booster_threads(&models), [4, 4]);
    }

    /// A caught panic leaves every temporarily capped booster reusable with its original budget.
    #[rstest::rstest]
    fn a_join_panic_restores_every_boosters_threads(#[values(false, true)] measuring_panics: bool) {
        let fixture = Fixture::new();
        let [snp, indel] = snp_and_indel();
        let mut models =
            [snp, indel, crate::testing::separable_model(Task::Promote, VariantType::Snp)];
        with_four_threads(&mut models);
        let hooks = if measuring_panics {
            Hooks {
                before_measuring: Some(Box::new(|_| panic!("measurement panic"))),
                ..Hooks::default()
            }
        } else {
            Hooks {
                before_scoring: Some(Box::new(|_| panic!("scoring panic"))),
                ..Hooks::default()
            }
        };
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fixture.measure_and_decide(&mut models, 2, hooks)
        }));
        assert!(panicked.is_err());
        assert_eq!(booster_threads(&models), [4, 4, 4]);
        assert!(fixture.measure_and_decide(&mut models, 2, Hooks::default()).is_ok());
    }

    /// A real setter rejection during restoration surfaces, but does not suppress work errors or
    /// prevent restoration of later boosters.
    #[rstest::rstest]
    #[case::restore(false, false, "nthread")]
    #[case::measurement(false, true, "missing.bam")]
    #[case::scoring(true, false, "absent")]
    fn restoration_errors_preserve_precedence_and_attempt_every_booster(
        #[case] scoring_fails: bool,
        #[case] measurement_fails: bool,
        #[case] expected: &str,
    ) {
        let mut fixture = Fixture::new();
        if measurement_fails {
            fixture.files.bam = fixture.files.bam.with_file_name("missing.bam");
        }
        let [snp, indel] = snp_and_indel_of(if scoring_fails { "absent" } else { "x" });
        let mut models =
            [snp, indel, crate::testing::separable_model(Task::Promote, VariantType::Snp)];
        with_four_threads(&mut models);
        let inputs = fixture.inputs();
        let contigs = ["chr1".to_string()];
        let measure = Measuring {
            inputs: &inputs,
            strlen: None,
            pileup: Some(PileupOptions::default()),
            contigs: &contigs,
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let mut restorations = 0;
        let error = pool
            .install(|| {
                measure_while_scoring_with_threads(
                    Fixture::table(),
                    &mut models,
                    &Policy::new(crate::rescore::Mode::Genotyper),
                    &measure,
                    &mut PhaseTimer::start(),
                    Hooks::default(),
                    |booster, threads| {
                        if threads.get() != 1 {
                            restorations += 1;
                            if restorations == 1 {
                                // Exercise an actual setter failure rather than fabricating an error value.
                                return booster.set_threads(NonZeroUsize::new(usize::MAX).unwrap());
                            }
                        }
                        booster.set_threads(threads)
                    },
                )
            })
            .unwrap_err();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
        assert_eq!(restorations, 2, "cleanup stopped at its first error");
        assert_eq!(booster_threads(&models), [4, 1, 4]);
    }

    /// A caller's cancellation remains effective when shared scoring adds its own stop flag.
    #[rstest::rstest]
    fn shared_scoring_preserves_caller_cancellation(#[values(1, 2)] threads: usize) {
        let fixture = Fixture::new();
        let mut models = snp_and_indel();
        let cancelled = AtomicBool::new(true);
        let inputs = ReadInputs { stop: Some(&cancelled), ..fixture.inputs() };
        let contigs = ["chr1".to_string()];
        let measuring = Measuring {
            inputs: &inputs,
            strlen: None,
            pileup: Some(PileupOptions::default()),
            contigs: &contigs,
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let error = pool
            .install(|| {
                measure_and_decide(
                    Fixture::table(),
                    &mut models,
                    &Policy::new(crate::rescore::Mode::Genotyper),
                    &measuring,
                    &mut PhaseTimer::start(),
                )
            })
            .unwrap_err();
        assert!(error.to_string().contains("the pass over the reads was stopped"), "{error:#}");
    }

    /// A model that fails while the reads are being measured stops the measurement, and fails
    /// the run once the measurement has stopped: none runs on as the error ends the process.
    #[test]
    fn a_scoring_error_stops_and_waits_for_the_measurement() {
        use std::sync::mpsc::channel;
        let fixture = Fixture::new();
        let mut models = snp_and_indel_of("absent");
        let (held, was_held) = channel::<bool>();
        let (finished, measured) = channel::<bool>();
        let hooks = Hooks {
            // The measurement starts once scoring has failed and stopped it (or, should it not,
            // after a while, measuring every shard).
            before_measuring: Some(Box::new(move |stop: &AtomicBool| {
                _ = held.send(wait_until(|| stop.load(Ordering::Relaxed)));
            })),
            after_measuring: Some(Box::new(move |ok| _ = finished.send(ok))),
            ..Hooks::default()
        };
        let error = fixture.measure_and_decide(&mut models, 3, hooks).unwrap_err();
        assert!(format!("{error:#}").contains("absent"), "{error:#}");
        // Stopped before measuring, the measurement had failed before the error was returned.
        assert_eq!(was_held.try_recv(), Ok(true), "scoring did not stop the measurement");
        assert_eq!(measured.try_recv(), Ok(false));
    }

    /// A measurement that fails while the models that do not need the evidence score fails the
    /// run before the next of them scores.
    #[test]
    fn a_failed_measurement_is_reported_before_the_next_model_scores() {
        use std::sync::mpsc::{Receiver, channel};
        let mut fixture = Fixture::new();
        fixture.files.bam = fixture.files.bam.with_file_name("missing.bam");
        let [snp, indel] = snp_and_indel();
        let mut models =
            [snp, indel, crate::testing::separable_model(Task::Promote, VariantType::Snp)];
        let (finished, measured) = channel::<bool>();
        let measured: Mutex<Receiver<bool>> = Mutex::new(measured);
        let (started, scoring_started) = channel::<()>();
        let scored: Arc<Mutex<Vec<Task>>> = Arc::default();
        let scoring = Arc::clone(&scored);
        let hooks = Hooks {
            // The measurement starts (and fails) once the first model has been let score: it
            // cannot fail before scoring has checked for a failure.
            before_measuring: Some(Box::new(move |_: &AtomicBool| {
                _ = scoring_started.recv_timeout(PATIENCE);
            })),
            after_measuring: Some(Box::new(move |ok| _ = finished.send(ok))),
            // The first model scores once the measurement has failed.
            before_scoring: Some(Box::new(move |model: &WinnowrModel| {
                let mut scored = scoring.lock().unwrap();
                if scored.is_empty() {
                    _ = started.send(());
                    let failed = measured.lock().unwrap().recv_timeout(PATIENCE);
                    assert_eq!(failed, Ok(false));
                }
                scored.push(model.info.task);
            })),
        };
        let error = fixture.measure_and_decide(&mut models, 3, hooks).unwrap_err();
        assert!(format!("{error:#}").contains("missing.bam"), "{error:#}");
        assert_eq!(*scored.lock().unwrap(), [Task::Genotype]);
    }

    /// One side of a two-way handoff between two threads (see [`handoff`]).
    struct Handoff {
        /// Tells the other side this one has arrived.
        arrived: std::sync::mpsc::Sender<()>,
        /// Hears that the other side has arrived.
        other: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl Handoff {
        /// Arrives, then waits for the other side to, for at most [`PATIENCE`]; returns whether
        /// it did.
        fn meet(&self) -> bool {
            _ = self.arrived.send(());
            self.other.lock().unwrap().recv_timeout(PATIENCE).is_ok()
        }
    }

    /// The two sides of a handoff: neither leaves [`Handoff::meet`] (but on running out of
    /// patience) before the other has arrived, so both are running at once.
    fn handoff() -> (Handoff, Handoff) {
        use std::sync::mpsc::channel;
        let (first_arrived, first) = channel();
        let (second_arrived, second) = channel();
        (
            Handoff { arrived: first_arrived, other: Mutex::new(second) },
            Handoff { arrived: second_arrived, other: Mutex::new(first) },
        )
    }

    /// The work runs on the pool's threads, the measurement while the models score (each side
    /// waits for the other to start), during which their boosters have one thread.
    #[test]
    fn the_work_runs_on_the_pool() {
        use std::collections::HashSet;
        use std::thread::ThreadId;
        let fixture = Fixture::new();
        let pool_threads: Arc<Mutex<HashSet<ThreadId>>> = Arc::default();
        let recorded = Arc::clone(&pool_threads);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .start_handler(move |_| {
                recorded.lock().unwrap().insert(std::thread::current().id());
            })
            .build()
            .unwrap();
        // Every thread has started once a job runs on each of them.
        let barrier = std::sync::Barrier::new(3);
        pool.broadcast(|_| _ = barrier.wait());
        let seen: Arc<Mutex<Vec<(ThreadId, usize)>>> = Arc::default();
        let (measured_on, scored_on) = (Arc::clone(&seen), Arc::clone(&seen));
        // Whether each side (the measurement and the one model's scoring, in either order) met the
        // other.
        let met: Arc<Mutex<Vec<bool>>> = Arc::default();
        let (measuring_met, scoring_met) = (Arc::clone(&met), Arc::clone(&met));
        let (measuring, scoring) = handoff();
        let hooks = Hooks {
            before_measuring: Some(Box::new(move |_: &AtomicBool| {
                measured_on.lock().unwrap().push((std::thread::current().id(), 0));
                let met = measuring.meet();
                measuring_met.lock().unwrap().push(met);
            })),
            before_scoring: Some(Box::new(move |model: &WinnowrModel| {
                let threads = model.booster.threads().get();
                scored_on.lock().unwrap().push((std::thread::current().id(), threads));
                let met = scoring.meet();
                scoring_met.lock().unwrap().push(met);
            })),
            ..Hooks::default()
        };
        let mut models = snp_and_indel();
        with_four_threads(&mut models);
        pool.install(|| fixture.measure_and_decide_here(&mut models, hooks)).unwrap();
        let pool_threads = pool_threads.lock().unwrap();
        let seen = seen.lock().unwrap();
        // The measurement (no threads recorded), and the indel model with one thread.
        let mut threads: Vec<usize> = seen.iter().map(|&(_, threads)| threads).collect();
        threads.sort_unstable();
        assert_eq!(threads, [0, 1]);
        assert!(seen.iter().all(|(id, _)| pool_threads.contains(id)), "{seen:?}");
        assert_eq!(
            *met.lock().unwrap(),
            [true, true],
            "the measurement and scoring did not overlap"
        );
    }

    /// Counts real, nested prediction/read jobs across the operation, including unexpected pools.
    /// Native matrix/prediction and htslib work run synchronously inside these scopes; this does
    /// not instrument native bodies or promise detection of arbitrary upstream worker creation.
    #[rstest::rstest]
    fn prediction_and_read_shards_share_one_active_worker_budget(#[values(2, 4)] budget: usize) {
        use std::collections::HashSet;
        let fixture = Fixture::new();
        let feature = format!("budget-{}", fixture._dir.path().display());
        let mut models = snp_and_indel_of(&feature);
        with_four_threads(&mut models);
        let calls = vec![
            ("chr1", 131, "G", "T", "PASS", "indel", "0/1", false);
            crate::model::PREDICT_CHUNK_ROWS * 8
        ];
        let table = crate::testing::call_table(&calls)
            .with_f64_column(&feature, vec![0.0; calls.len()])
            .unwrap()
            .with_bool_column("flippable", vec![true; calls.len()])
            .unwrap();
        let inputs = fixture.inputs();
        // Forty actual nested read-shard jobs without a large reference or synthetic read work.
        let regions: Vec<String> =
            (0..40).map(|i| format!("chr1:{}-{}", i * 4 + 1, i * 4 + 4)).collect();
        let measure = Measuring {
            inputs: &inputs,
            strlen: None,
            pileup: Some(PileupOptions::default()),
            contigs: &regions,
        };
        let observer = work_observer::Observer::new(budget);
        let _registered = observer.register(inputs.vcf, &feature);
        let expected_workers: Arc<Mutex<HashSet<std::thread::ThreadId>>> = Arc::default();
        let workers = Arc::clone(&expected_workers);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(budget)
            .start_handler(move |_| {
                workers.lock().unwrap().insert(std::thread::current().id());
            })
            .build()
            .unwrap();
        // Start both arms before either fans out: neither class of jobs can monopolize the gate.
        let (measuring, scoring) = handoff();
        let hooks = Hooks {
            before_measuring: Some(Box::new(move |_| assert!(measuring.meet()))),
            before_scoring: Some(Box::new(move |model| {
                assert_eq!(model.booster.threads().get(), 1);
                assert!(scoring.meet());
            })),
            ..Hooks::default()
        };
        let (decisions, shared) = pool
            .install(|| {
                measure_and_decide_with(
                    table,
                    &mut models,
                    &Policy::new(crate::rescore::Mode::Genotyper),
                    &measure,
                    &mut PhaseTimer::start(),
                    hooks,
                )
            })
            .unwrap();
        assert!(shared);
        assert_eq!(decisions.len(), 1);
        assert!(decisions.values().next().unwrap().drop);
        assert_eq!(booster_threads(&models), [4, 4]);
        let seen = observer.seen.lock().unwrap();
        assert!(!seen.timed_out, "jobs did not fill the shared budget before their deadline");
        assert!(seen.overlapped, "prediction and read jobs never overlapped");
        assert_eq!(seen.peak, budget, "must saturate, but never exceed, the combined budget");
        assert_eq!(seen.read_jobs, 40);
        assert_eq!(seen.prediction_jobs, 8);
        assert!(seen.prediction_threads.iter().all(|&threads| threads == 1));
        // Do not filter the counter by these IDs: an unexpected pool must also be observed.
        assert!(seen.workers.is_subset(&expected_workers.lock().unwrap()));
        assert!(seen.active.is_empty());
    }

    /// What the hooks of `called_from_outside_a_pool_the_work_runs_on_the_global_pool` saw.
    #[derive(Default)]
    struct Seen {
        /// The measurement ran on a pool thread.
        measured_on_the_pool: AtomicBool,
        /// A model scored on a pool thread.
        scored_on_the_pool: AtomicBool,
    }

    /// Called from a thread outside any pool, as `score` calls it, the work runs on the global
    /// pool's threads (measuring while a model scores when the pool has more than one thread;
    /// that the two overlap is `the_work_runs_on_the_pool`'s to check).
    #[test]
    fn called_from_outside_a_pool_the_work_runs_on_the_global_pool() {
        let fixture = Fixture::new();
        let mut models = snp_and_indel();
        let expected = fixture.expected(&models);
        assert_eq!(rayon::current_thread_index(), None);
        let seen: Arc<Seen> = Arc::default();
        let (measuring, scoring) = (Arc::clone(&seen), Arc::clone(&seen));
        let on_the_pool = || rayon::current_thread_index().is_some();
        let hooks = Hooks {
            before_measuring: Some(Box::new(move |_: &AtomicBool| {
                measuring.measured_on_the_pool.store(on_the_pool(), Ordering::Relaxed);
            })),
            before_scoring: Some(Box::new(move |_: &WinnowrModel| {
                scoring.scored_on_the_pool.store(on_the_pool(), Ordering::Relaxed);
            })),
            ..Hooks::default()
        };
        let (got, shared) = fixture.measure_and_decide_here(&mut models, hooks).unwrap();
        assert_eq!(got, expected);
        assert_eq!(shared, rayon::current_num_threads() > 1);
        if shared {
            assert!(seen.measured_on_the_pool.load(Ordering::Relaxed));
            assert!(seen.scored_on_the_pool.load(Ordering::Relaxed));
        }
    }
}
