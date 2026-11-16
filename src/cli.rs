//! The command-line interface: global options, the subcommands, and dispatch.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use log::info;
use winnowr::common::{AUTOSOMES, PhaseTimer, Task, VariantType, repeated};
use winnowr::evidence::read_tables;
use winnowr::extraction::{TableInputs, TruthInputs, build_feature_table};
use winnowr::features::feature_names;
use winnowr::labels::UnfixableFlips;
use winnowr::model::{CvFolds, MODEL_SUFFIX, TrainingParams, WinnowrModel, check_reference};
use winnowr::model_evidence::{
    Measuring, check_supplied, check_table, measure_and_decide, models_options, models_using,
    read_for_models, shared_options,
};
use winnowr::pileup::{PileupOptions, measure_pileup, write_pileup_table};
use winnowr::pileup_features::PILEUP_FEATURES;
use winnowr::policy::{
    Abstention, CONFIDENT_REGIONS, DEFAULT_KEEP_FALLBACK, DEFAULT_OUTSIDE_INDEL_T_DROP, Policy,
    decide_calls,
};
use winnowr::read_pass::ReadInputs;
use winnowr::regions::{NamedBed, NamedPath, RegionSet, read_region_sets};
use winnowr::repeat_features::STRLEN_FEATURES;
use winnowr::rescore::{Mode, Rule};
use winnowr::samples::{PromoteSampleCalls, SampleCalls};
use winnowr::scoring::write_scored_vcf;
use winnowr::strlen::{StrlenOptions, measure_regions_with, write_strlen_table};
use winnowr::stutter::{
    LENGTH_CALL_FEATURES, StutterFit, StutterModel, fit_stutter_from_feature_tables,
};
use winnowr::subset::restrict_table;
use winnowr::table::{FeatureTable, FeatureTableInfo, TableSelection};
use winnowr::thresholds::{DEFAULT_GRID, DEFAULT_PROMOTE_FLIP_GRID, DEFAULT_PROMOTE_GRID};
use winnowr::training::{
    FoldTraining, TrainingSpec, fit_fold_stutter, group_samples, table_columns, train_model,
    train_promote_model,
};
use winnowr::vcf::VcfSummary;
use winnowr::version::{LONG_VERSION, VERSION};

/// The default number of threads, when neither `-t` nor `OMP_NUM_THREADS` gives one: enough to
/// overlap I/O with compute, while staying predictable on shared hosts and under schedulers.
pub(crate) const DEFAULT_THREADS: NonZeroUsize = NonZeroUsize::new(4).expect("4 is non-zero");

/// The environment variable that gives the number of threads when `-t` is not given, as it gives
/// XGBoost's own default (and as the Python version of winnowr used it).
pub(crate) const THREADS_VARIABLE: &str = "OMP_NUM_THREADS";

/// Machine-learned genotyping and filtering of germline small-variant calls.
#[derive(Debug, Parser)]
#[command(
    name = "winnowr",
    version = VERSION.as_str(),
    long_version = LONG_VERSION.as_str(),
    max_term_width = 100
)]
pub struct Cli {
    /// The options shared by every subcommand.
    #[command(flatten)]
    pub(crate) global: GlobalOptions,

    /// The subcommand to run.
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// Options shared by every subcommand.
#[derive(Debug, Args)]
pub(crate) struct GlobalOptions {
    /// The number of worker threads to use [default: OMP_NUM_THREADS when set, else 4].
    #[arg(long, short = 't', global = true)]
    pub(crate) threads: Option<NonZeroUsize>,
}

/// Where the number of threads came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThreadsFrom {
    /// The `-t`/`--threads` option.
    Option,
    /// The `OMP_NUM_THREADS` environment variable.
    Environment,
    /// [`DEFAULT_THREADS`].
    Default,
}

impl GlobalOptions {
    /// The number of threads to use and where it came from: `-t`, else the first value of
    /// `OMP_NUM_THREADS` (which may list one per nesting level, e.g. `8,2`), else
    /// [`DEFAULT_THREADS`]. An `OMP_NUM_THREADS` that is not a positive number is ignored, with a
    /// warning.
    ///
    /// `var` looks up an environment variable (`std::env::var` outside tests).
    pub(crate) fn threads<F: Fn(&str) -> Option<String>>(
        &self,
        var: F,
    ) -> (NonZeroUsize, ThreadsFrom) {
        if let Some(threads) = self.threads {
            return (threads, ThreadsFrom::Option);
        }
        let Some(value) = var(THREADS_VARIABLE).filter(|value| !value.trim().is_empty()) else {
            return (DEFAULT_THREADS, ThreadsFrom::Default);
        };
        let first = value.split(',').next().unwrap_or_default().trim();
        match first.parse::<NonZeroUsize>() {
            Ok(threads) => (threads, ThreadsFrom::Environment),
            Err(_) => {
                log::warn!(
                    "{THREADS_VARIABLE}={value} is not a positive number of threads; using \
                     {DEFAULT_THREADS} threads (set -t to choose)"
                );
                (DEFAULT_THREADS, ThreadsFrom::Default)
            }
        }
    }
}

/// The winnowr subcommands.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Extract a feature table for a sample's calls, optionally labelled with truth.
    Extract(ExtractArgs),
    /// Train a genotype or promote model from labelled feature tables.
    ///
    /// The model depends on the number of threads (-t, else OMP_NUM_THREADS), as XGBoost's
    /// histogram sums do: training again with the same tables, options and threads gives the same
    /// model.
    Train(TrainArgs),
    /// Report F1 for models' decisions on a labelled sample.
    Evaluate(EvaluateArgs),
    /// Report models' decisions, outcomes, F1 and P(keep) calibration by benchmark stratum.
    ///
    /// Each call is in one stratum by whether its REF span is inside each of two truth sets'
    /// confident regions: both, <second>-only, <first>-only or neither. Calls are scored as
    /// `score` does. Writes <output>.strata.tsv (actions, outcomes and F1 of the labelled calls
    /// per truth set and stratum; the `all` rows match `evaluate`), <output>.rates.tsv (action
    /// rates of every scored call per stratum, variant type and chromosome, with no truth needed)
    /// and <output>.calibration.tsv (binned P(keep) against the observed keep rate).
    ///
    /// With --stratification or --report-region (e.g. GIAB stratification BEDs), also writes the same
    /// metrics within each named region, as hap.py's --stratification does: a call or truth FN
    /// is in a region if its REF span overlaps it, and regions may overlap. Each truth set is
    /// scored within its own confident regions. Writes <output>.regions.strata.tsv,
    /// <output>.regions.calibration.tsv and <output>.regions.rates.tsv (action rates per region
    /// and benchmark stratum, including the parts outside the truth sets).
    Stratify(StratifyArgs),
    /// Score a sample's calls and write a VCF with drops, promotions, flips and re-genotypes.
    ///
    /// Each model acts at its cross-validated thresholds. Indel genotype models also drop the
    /// calls their thresholds leave alone when P(keep) is below 0.5 (--keep-fallback). Given the
    /// confident regions (--regions notinalldifficultregions=PATH, GIAB's GRCh38 genome
    /// stratification), indel calls outside them are dropped at P(drop) >= 0.6 (in veto mode,
    /// P(drop) + P(flip) (+ P(re-genotype)) >= 0.6; --outside-indel-t-drop) in place of the
    /// model's drop threshold, whether that is lower or higher, and the keep fallback still drops
    /// them when P(keep) is below 0.5. Without the regions, `score` warns and uses the model's
    /// threshold everywhere.
    Score(ScoreArgs),
    /// Measure read lengths across repeat tracts at indel calls (repeat-length evidence).
    Strlen(StrlenArgs),
    /// Count the reads of each call's alleles by MAPQ, at any MAPQ (pileup evidence).
    Pileup(PileupArgs),
    /// Restrict a labelled feature table to some of its contigs, or away from some regions.
    ///
    /// Keeps the calls on --contigs whose REF span does not overlap --exclude-bed, and recounts
    /// the truth FNs from the hap.py VCF on the same terms, so that training on the table (e.g.
    /// a sample's even chromosomes, to evaluate on its odd ones) sees no label from elsewhere.
    /// Features are kept as extracted: per-sample features (e.g. the depth normalization) still
    /// come from all of the sample's calls, and the table still lists all its contigs.
    Subset(SubsetArgs),
}

impl Command {
    /// The subcommand's name on the command line.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Extract(_) => "extract",
            Self::Train(_) => "train",
            Self::Evaluate(_) => "evaluate",
            Self::Stratify(_) => "stratify",
            Self::Score(_) => "score",
            Self::Strlen(_) => "strlen",
            Self::Pileup(_) => "pileup",
            Self::Subset(_) => "subset",
        }
    }
}

/// The options of `winnowr extract`.
#[derive(Debug, Args)]
pub(crate) struct ExtractArgs {
    /// The indexed, single-sample VCF from HaplotypeCaller (gVCF mode, e.g. --dragen-mode).
    #[arg(long)]
    pub(crate) vcf: PathBuf,

    /// The feature table to write (Parquet).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// The indexed gVCF the VCF was genotyped from. It supplies the raw annotations
    /// (RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ) when the VCF lacks them, and is not read
    /// otherwise.
    #[arg(long)]
    pub(crate) gvcf: Option<PathBuf>,

    /// Region BEDs to flag calls in, as NAME=PATH (e.g. segdups=segdups.bed.gz). The names must
    /// match those a model was trained with.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,

    /// The contigs to extract.
    #[arg(long, num_args = 1.., default_values = AUTOSOMES)]
    pub(crate) contigs: Vec<String>,

    /// The sample name to record [default: the VCF's sample].
    #[arg(long)]
    pub(crate) sample: Option<String>,

    /// Repeat-tract read-length tables from `winnowr strlen` (e.g. one per shard), joined as
    /// repeat-length features of indel calls; needed to train or apply models that use them.
    #[arg(long, num_args = 1..)]
    pub(crate) strlen: Vec<PathBuf>,

    /// Allele read count tables from `winnowr pileup` (e.g. one per shard), joined as pileup
    /// features of SNP and indel calls; needed to train or apply models that use them.
    #[arg(long, num_args = 1..)]
    pub(crate) pileup: Vec<PathBuf>,

    /// The hap.py output VCF for this sample (run with --engine vcfeval --pass-only), which
    /// labels PASS calls; needed for training and evaluation tables.
    #[arg(long)]
    pub(crate) happy: Option<PathBuf>,

    /// The indexed truth VCF (e.g. a GIAB benchmark); with --truth-bed and --happy, labels
    /// non-PASS calls for training promote models.
    #[arg(long)]
    pub(crate) truth_vcf: Option<PathBuf>,

    /// The truth confident-regions BED.
    #[arg(long)]
    pub(crate) truth_bed: Option<PathBuf>,

    /// The indexed reference FASTA used to left-align indels when matching non-PASS calls to the
    /// truth; without it, alleles are only trimmed.
    #[arg(long)]
    pub(crate) reference: Option<PathBuf>,
}

/// Parses a value ignoring case (e.g. `SNP` or `snp`).
fn parse_ignoring_case<T: std::str::FromStr<Err = anyhow::Error>>(text: &str) -> Result<T, String> {
    text.to_ascii_lowercase().parse().map_err(|e: anyhow::Error| e.to_string())
}

/// The options of `winnowr train`.
#[derive(Debug, Args)]
pub(crate) struct TrainArgs {
    /// Labelled feature tables (from `extract --happy`), one per training sample: at least two,
    /// of distinct samples.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) tables: Vec<PathBuf>,

    /// The variant type to train for (snp or indel).
    #[arg(long, value_parser = parse_ignoring_case::<VariantType>)]
    pub(crate) variant_type: VariantType,

    /// The model file to write (`.winnowr.json` is appended if missing).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// genotype trains on PASS calls (drop/keep/flip; needs hap.py labels); promote trains on
    /// non-PASS calls (stay/promote/promote+flip; needs tables extracted with --truth-vcf).
    #[arg(long, default_value = "genotype", value_parser = parse_ignoring_case::<Task>)]
    pub(crate) task: Task,

    /// Leave out the multi-site neighbourhood features.
    #[arg(long)]
    pub(crate) no_neighbourhood: bool,

    /// Use the repeat-length features (indel genotype models; tables extracted with
    /// `extract --strlen`): fits a stutter model on the training samples' truth-homozygous
    /// calls, saved with the model, and adds read-length histogram and length-caller features.
    /// Each cross-validation fold refits the stutter model on its training samples only.
    #[arg(long)]
    pub(crate) repeat_length: bool,

    /// Add the re-genotype class (implies --repeat-length): calls whose genotype a het <-> hom
    /// flip does not fix but the length caller's genotype does; its threshold is chosen by
    /// cross-validation with the genotyper's.
    #[arg(long)]
    pub(crate) regenotype: bool,

    /// Experimental: how the stutter model is fitted, as comma-separated settings: direction
    /// (errors signed towards or away from the reference length) and whole-unit (fit only truth
    /// alleles of whole repeat units).
    #[arg(long, default_value = "default", hide = true)]
    pub(crate) stutter_fit: StutterFit,

    /// Use the pileup features (tables extracted with `extract --pileup`): the reads of each
    /// call's reference and alternate alleles at MAPQ 0, at low MAPQ and at high MAPQ, counted
    /// at any MAPQ, and fractions of them. The pileup settings are recorded in the model, which
    /// then needs pileup evidence measured the same way to score.
    #[arg(long)]
    pub(crate) pileup_features: bool,

    /// What genotype training does with flip labels that a het <-> hom flip cannot fix (an
    /// allele-match FP that is not `0/1` or `1/1`, or whose flip is not the truth genotype, e.g.
    /// a call missing the second allele of a het-alt truth): flip trains them as flips, as
    /// hap.py labels them; drop trains them as drops; exclude leaves them out of the fits. They
    /// are scored as FPs in cross-validation either way. Recorded in the model.
    #[arg(long, default_value = "flip", value_parser = parse_ignoring_case::<UnfixableFlips>)]
    pub(crate) unfixable_flips: UnfixableFlips,

    /// Samples to hold out together in cross-validation, one comma-separated group per value
    /// (e.g. a trio: HG005,HG006,HG007). Samples not in a group are held out alone. Unused with
    /// --cv-folds chromosomes.
    #[arg(long, num_args = 1..)]
    pub(crate) sample_groups: Vec<String>,

    /// How the cross-validation that chooses the thresholds holds out calls: samples holds out
    /// each sample group (--sample-groups) in turn; chromosomes holds out groups of whole
    /// chromosomes (--chromosome-folds groups of about equal calls) from every sample at once, so
    /// every fold fits every sample, and related samples' calls at the same site are held out
    /// together. Recorded in the model.
    #[arg(long, default_value = "samples", ignore_case = true,
          value_parser = ["samples", "chromosomes"])]
    pub(crate) cv_folds: String,

    /// The number of chromosome groups (folds) with --cv-folds chromosomes.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(2..))]
    pub(crate) chromosome_folds: u32,

    /// Refit the final model on all its training calls, those held out for early stopping
    /// included, for the rounds its early stopping chose. Has no effect without early stopping
    /// (a zero --validation-fraction). Recorded in the model.
    #[arg(long, default_value_t = winnowr::model::DEFAULT_REFIT_FINAL, num_args = 0..=1,
          default_missing_value = "true", action = clap::ArgAction::Set)]
    pub(crate) refit_final: bool,

    /// Candidate thresholds: t_drop and t_flip for genotype models, t_promote and t_flip for
    /// promote models, each a probability in [0, 1]. "Never act" is always added [default:
    /// built-in grids from 0.2 up].
    #[arg(long, num_args = 1.., value_parser = parse_probability)]
    pub(crate) grid: Vec<f64>,

    /// The XGBoost learning rate (eta).
    #[arg(long, default_value_t = TrainingParams::default().learning_rate)]
    pub(crate) learning_rate: f64,

    /// The maximum leaves per tree.
    #[arg(long, default_value_t = TrainingParams::default().max_leaves)]
    pub(crate) max_leaves: u32,

    /// The maximum tree depth (0 = unlimited, bounded by --max-leaves).
    #[arg(long, default_value_t = TrainingParams::default().max_depth)]
    pub(crate) max_depth: u32,

    /// The minimum hessian sum per leaf.
    #[arg(long, default_value_t = TrainingParams::default().min_child_weight)]
    pub(crate) min_child_weight: f64,

    /// L2 regularization of leaf weights.
    #[arg(long, default_value_t = TrainingParams::default().reg_lambda)]
    pub(crate) reg_lambda: f64,

    /// The maximum boosting rounds.
    #[arg(long, default_value_t = TrainingParams::default().max_rounds,
          value_parser = clap::value_parser!(u32).range(1..))]
    pub(crate) max_rounds: u32,

    /// Early-stopping patience (rounds) on the validation split.
    #[arg(long, default_value_t = TrainingParams::default().early_stopping_rounds,
          value_parser = clap::value_parser!(u32).range(1..))]
    pub(crate) early_stopping_rounds: u32,

    /// About the fraction of training calls held out for early stopping (0 disables it). Whole
    /// genomic blocks (--validation-block-size) are held out, from every sample, so that calls at
    /// the same site in other samples, and neighbouring calls, do not leak into training.
    #[arg(long, default_value_t = TrainingParams::default().validation_fraction)]
    pub(crate) validation_fraction: f64,

    /// The size (bp) of the genomic blocks held out together for early stopping.
    #[arg(long, default_value_t = TrainingParams::default().validation_block_size,
          value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) validation_block_size: u64,

    /// The random seed (validation split and XGBoost; at most 2^63 - 1, as XGBoost takes it).
    #[arg(long, default_value_t = TrainingParams::default().seed,
          value_parser = clap::value_parser!(u64).range(..=i64::MAX as u64))]
    pub(crate) seed: u64,
}

/// The options of `winnowr evaluate`.
#[derive(Debug, Args)]
pub(crate) struct EvaluateArgs {
    /// The model files of one variant type: a genotype model, a promote model, or one of each.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) models: Vec<PathBuf>,

    /// A labelled feature table (from `extract --happy`, plus --truth-vcf for promote models) of
    /// a held-out sample (a training sample of the models gives optimistic metrics, with a
    /// warning).
    #[arg(long)]
    pub(crate) table: PathBuf,

    /// The metrics file to write (TSV).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// Also report thresholds tuned on this sample (optimistic: they peek at the labels).
    #[arg(long)]
    pub(crate) tuned: bool,

    /// Options that change how models act (in the veto, genotyper and tuned policies).
    #[command(flatten)]
    pub(crate) policy: PolicyArgs,

    /// The region BEDs `score` would get, as NAME=PATH: calls are decided as `score` decides them
    /// with these. Only the confident regions (notinalldifficultregions) are read; the models'
    /// region features come from the table.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,
}

/// The drop threshold of indel calls outside the confident regions: a P(drop), or `none` for the
/// model's threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OutsideTDrop(Option<f64>);

/// Parses a probability: a finite number in `[0, 1]`.
///
/// # Errors
///
/// Returns an error naming the value if it is not a number, not finite, or outside `[0, 1]`.
fn parse_probability(text: &str) -> Result<f64> {
    let value: f64 = text.parse().with_context(|| format!("not a number: {text}"))?;
    // A NaN fails the range check too.
    if !(0.0..=1.0).contains(&value) {
        bail!("must be a probability between 0 and 1, got {text}");
    }
    Ok(value)
}

impl std::fmt::Display for OutsideTDrop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(value) => write!(f, "{value}"),
            None => f.write_str("none"),
        }
    }
}

impl std::str::FromStr for OutsideTDrop {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        if text.eq_ignore_ascii_case("none") {
            return Ok(OutsideTDrop(None));
        }
        let value: f64 = text.parse().with_context(|| format!("not a number or none: {text}"))?;
        if !(0.0..=1.0).contains(&value) {
            bail!("must be between 0 and 1, got {value}");
        }
        Ok(OutsideTDrop(Some(value)))
    }
}

/// Options that change how genotype and promote models act, beyond their thresholds.
///
/// By default ([`DEFAULT_KEEP_FALLBACK`] and [`DEFAULT_OUTSIDE_INDEL_T_DROP`]), indel genotype
/// models also drop calls their thresholds leave alone when P(keep) is below 0.5, and, given the
/// confident regions (`--regions notinalldifficultregions=PATH`), drop indel calls outside them at
/// P(drop) >= 0.6 in place of the model's drop threshold (whether that is lower or higher); the
/// keep fallback still drops those calls when P(keep) is below 0.5.
#[derive(Debug, Args)]
pub(crate) struct PolicyArgs {
    /// Variant types whose genotype model also drops the calls its thresholds leave alone when
    /// P(keep) is below 0.5 (none, alone: no variant type).
    #[arg(long, num_args = 1.., default_values_t = DEFAULT_KEEP_FALLBACK.map(|vt| vt.to_string()),
          ignore_case = true, value_parser = ["snp", "indel", "none"])]
    pub(crate) keep_fallback: Vec<String>,

    /// Experimental: variant types whose genotype model takes the action of greatest expected F1
    /// gain instead of using its thresholds (and in place of --keep-fallback): flip or
    /// re-genotype when its probability is above 0.5, else drop when P(keep) is below 0.5.
    #[arg(long, num_args = 1.., value_parser = parse_ignoring_case::<VariantType>)]
    pub(crate) expected_f1: Vec<VariantType>,

    /// Experimental: actions withheld from calls outside the confident regions (--regions
    /// notinalldifficultregions=PATH; calls whose REF span is not inside it are "outside"); none,
    /// alone: no action.
    #[arg(long, num_args = 1.., default_values = ["none"], ignore_case = true,
          value_parser = ["flip", "regenotype", "promote", "none"])]
    pub(crate) abstain_actions: Vec<String>,

    /// The drop threshold of indel calls outside the confident regions: they are dropped when
    /// P(drop) is at least this (in veto mode, P(drop) + P(flip) (+ P(re-genotype)), as the
    /// model's veto threshold is compared with), in place of the model's threshold, even when that
    /// is higher (none: the model's threshold). The keep fallback (--keep-fallback) still drops
    /// them when P(keep) is below 0.5.
    #[arg(long, default_value_t = OutsideTDrop(Some(DEFAULT_OUTSIDE_INDEL_T_DROP)),
          value_name = "P|none")]
    pub(crate) outside_indel_t_drop: OutsideTDrop,
}

impl PolicyArgs {
    /// The policy in a mode, checking the confident regions name some of `contigs`.
    ///
    /// The confident regions are the `regions` named [`CONFIDENT_REGIONS`]. Without them, the
    /// outside indel drop threshold is not applied (with a warning when `indel_genotype`, i.e. an
    /// indel genotype model will score).
    ///
    /// # Errors
    ///
    /// Returns an error if `none` is given with other values of --keep-fallback or
    /// --abstain-actions, actions are withheld without the confident regions, or they cannot be
    /// read or name none of `contigs`.
    fn policy<S: AsRef<str>>(
        &self,
        mode: Mode,
        contigs: &[S],
        regions: &[NamedBed],
        indel_genotype: bool,
    ) -> Result<Policy> {
        let keep_fallback = without_none("--keep-fallback", &self.keep_fallback)?;
        let abstain_actions = without_none("--abstain-actions", &self.abstain_actions)?;
        let mut policy = Policy::new(mode);
        for vt in &keep_fallback {
            policy.rules.insert(vt.parse()?, Rule::KeepFallback);
        }
        for &vt in &self.expected_f1 {
            policy.rules.insert(vt, Rule::ExpectedF1);
        }
        let has = |action: &str| abstain_actions.iter().any(|a| a == action);
        let (flip, regenotype, promote) = (has("flip"), has("regenotype"), has("promote"));
        let indel_t_drop = self.outside_indel_t_drop.0;
        let path = regions.iter().find(|r| r.name == CONFIDENT_REGIONS).map(|r| r.path.as_path());
        match path {
            Some(path) if flip || regenotype || promote || indel_t_drop.is_some() => {
                let regions = RegionSet::from_bed(CONFIDENT_REGIONS, path)?;
                regions.check_contigs(contigs)?;
                info!("Confident regions (acting fully inside): {}", path.display());
                policy.abstain =
                    Some(Abstention { regions, flip, regenotype, promote, indel_t_drop });
            }
            Some(_) => {}
            None if flip || regenotype || promote => bail!(
                "--abstain-actions needs the confident regions: --regions \
                 {CONFIDENT_REGIONS}=PATH"
            ),
            None => {
                if let (Some(t_drop), true) = (indel_t_drop, indel_genotype) {
                    log::warn!(
                        "no confident regions given, so indel calls are dropped at the model's \
                         threshold everywhere; pass --regions {CONFIDENT_REGIONS}=PATH (GIAB's \
                         GRCh38 genome stratification {CONFIDENT_REGIONS}.bed.gz) to drop indel \
                         calls outside it at P(drop) >= {t_drop} instead"
                    );
                }
            }
        }
        for (vt, rule) in &policy.rules {
            info!("{vt} genotype models decide with {rule:?}");
        }
        Ok(policy)
    }
}

/// The values of a multi-valued option, lower-cased, without `none`, which means no value.
///
/// # Errors
///
/// Returns an error if `none` is given with other values.
fn without_none(option: &str, values: &[String]) -> Result<Vec<String>> {
    let values: Vec<String> = values.iter().map(|v| v.to_ascii_lowercase()).collect();
    if values.len() > 1 && values.iter().any(|v| v == "none") {
        bail!("{option} none cannot be given with other values, got: {}", values.join(" "));
    }
    Ok(values.into_iter().filter(|v| v != "none").collect())
}

/// Whether any of the models is an indel genotype model.
fn has_indel_genotype(models: &[WinnowrModel]) -> bool {
    models
        .iter()
        .any(|m| m.info.task == Task::Genotype && m.info.variant_type == VariantType::Indel)
}

/// The options of `winnowr stratify`.
#[derive(Debug, Args)]
pub(crate) struct StratifyArgs {
    /// The model files: at most one genotype and one promote model per variant type.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) models: Vec<PathBuf>,

    /// The names of the two truth sets (e.g. v421 v5q), in the order of the options below.
    #[arg(long, required = true, num_args = 2, value_names = ["FIRST", "SECOND"])]
    pub(crate) truth_names: Vec<String>,

    /// The sample's labelled feature tables (from `extract --happy`, plus --truth-vcf for
    /// promote models), one per truth set, extracted from the same VCF.
    #[arg(long, required = true, num_args = 2, value_names = ["FIRST", "SECOND"])]
    pub(crate) tables: Vec<PathBuf>,

    /// The truth sets' confident-regions BEDs.
    #[arg(long, required = true, num_args = 2, value_names = ["FIRST", "SECOND"])]
    pub(crate) truth_beds: Vec<PathBuf>,

    /// The hap.py output VCFs the tables were labelled from (for the truth FNs per stratum).
    #[arg(long, required = true, num_args = 2, value_names = ["FIRST", "SECOND"])]
    pub(crate) happy: Vec<PathBuf>,

    /// How genotype models act: genotyper (drop, flip and re-genotype) or veto (drop only).
    #[arg(long, default_value = "genotyper")]
    pub(crate) mode: Mode,

    /// Options that change how models act.
    #[command(flatten)]
    pub(crate) policy: PolicyArgs,

    /// The region BEDs `score` would get, as NAME=PATH: calls are decided as `score` decides them
    /// with these. Only the confident regions (notinalldifficultregions) are read; the models'
    /// region features come from the table. Not reported within: see --report-region.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,

    /// The prefix of the reports to write (<output>.strata.tsv, <output>.rates.tsv and
    /// <output>.calibration.tsv).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// Also write each scored call's stratum, labels, probabilities and action (gzipped TSV).
    #[arg(long)]
    pub(crate) calls: Option<PathBuf>,

    /// Also write the labelled PASS calls as the genotype models see them (features, length-call
    /// features, re-genotype columns and labels) to <dir>/<sample>.<truth>.<vt>.parquet.
    #[arg(long, value_name = "DIR")]
    pub(crate) labelled_calls: Option<PathBuf>,

    /// A hap.py stratification manifest of regions to also report within: NAME<TAB>PATH lines,
    /// with paths relative to the manifest's directory.
    #[arg(long)]
    pub(crate) stratification: Option<PathBuf>,

    /// Regions to also report within, as NAME=PATH (BED, optionally gzipped), after those of
    /// --stratification.
    #[arg(long, num_args = 1.., value_name = "NAME=PATH")]
    pub(crate) report_region: Vec<NamedPath>,

    /// Report only the calls and truth FNs on these contigs (e.g. the chromosomes held out from
    /// a model's training) [default: all the table's contigs].
    #[arg(long, num_args = 1..)]
    pub(crate) contigs: Vec<String>,
}

/// The options of `winnowr subset`.
#[derive(Debug, Args)]
pub(crate) struct SubsetArgs {
    /// The labelled feature table (from `extract --happy`).
    #[arg(long)]
    pub(crate) table: PathBuf,

    /// The hap.py output VCF the table was labelled from (to recount its truth FNs).
    #[arg(long)]
    pub(crate) happy: PathBuf,

    /// The contigs to keep [default: all the table's contigs].
    #[arg(long, num_args = 1..)]
    pub(crate) contigs: Vec<String>,

    /// A BED of regions to drop calls and truth FNs in (those whose REF span overlaps it).
    #[arg(long)]
    pub(crate) exclude_bed: Option<PathBuf>,

    /// The feature table to write (Parquet).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,
}

/// The options of `winnowr score`.
#[derive(Debug, Args)]
pub(crate) struct ScoreArgs {
    /// The indexed, single-sample VCF from HaplotypeCaller (gVCF mode, e.g. --dragen-mode).
    #[arg(long)]
    pub(crate) vcf: PathBuf,

    /// The model files (`<name>.winnowr.json`): at most one genotype and one promote model per
    /// variant type.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) models: Vec<PathBuf>,

    /// The VCF to write: `.vcf.gz` (or `.vcf.bgz`) is bgzipped and tabix-indexed, `.bcf` is
    /// CSI-indexed BCF, anything else plain-text VCF (suffixes in any case).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// The indexed gVCF the VCF was genotyped from; needed only when the VCF lacks the raw
    /// annotations (RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ).
    #[arg(long)]
    pub(crate) gvcf: Option<PathBuf>,

    /// Region BEDs as NAME=PATH, with the names the models were trained with, plus the confident
    /// regions as notinalldifficultregions=PATH (GIAB's GRCh38 genome stratification; outside
    /// them, indel calls are dropped at --outside-indel-t-drop in place of the model's
    /// threshold).
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,

    /// How genotype models act: genotyper (drop, flip and re-genotype) or veto (drop only).
    #[arg(long, default_value = "genotyper")]
    pub(crate) mode: Mode,

    /// Options that change how models act.
    #[command(flatten)]
    pub(crate) policy: PolicyArgs,

    /// Repeat-tract read-length tables from `winnowr strlen` (e.g. one per shard); needed by
    /// models with repeat-length features, unless --alignments is given. They must be measured
    /// with the read settings the models' training tables were (`strlen --reads`).
    #[arg(long, num_args = 1.., conflicts_with = "alignments")]
    pub(crate) strlen: Vec<PathBuf>,

    /// Allele read count tables from `winnowr pileup` (e.g. one per shard); needed by models
    /// with pileup features, unless --alignments is given. They must be measured with the
    /// settings the models' training tables were (`pileup --settings`).
    #[arg(long, num_args = 1.., conflicts_with = "alignments")]
    pub(crate) pileup: Vec<PathBuf>,

    /// The indexed BAM or CRAM of the sample's reads, to measure the evidence models need while
    /// scoring, instead of reading tables: the repeat-length evidence of models with
    /// repeat-length features (as `winnowr strlen`), and the allele read counts of models with
    /// pileup features (as `winnowr pileup`), over the models' contigs, with the settings the
    /// models were trained with. Needs --reference. The reads are measured while the models that
    /// do not need them score, each with its own -t threads, so up to twice -t threads run.
    #[arg(long, requires = "reference")]
    pub(crate) alignments: Option<PathBuf>,

    /// The indexed, uncompressed reference FASTA, for --alignments (also used to decode CRAM).
    #[arg(long, requires = "alignments")]
    pub(crate) reference: Option<PathBuf>,
}

/// The options of `winnowr strlen`.
///
/// For every indel record (any FILTER) whose POS is in the regions, finds the repeat tract of its
/// first pure insertion or deletion allele in the reference, and histograms the observed minus
/// reference tract length over the reads that span the tract with aligned flank bases on each
/// side (by default reads with MAPQ at least 20 and five flank bases: see --reads), up to 600
/// reads per site. Unmapped, secondary, supplementary, duplicate and QC-failed reads are not
/// used. Writes the read settings and then one row per indel record with a pure insertion or
/// deletion allele (other records are skipped, and counted in the log); give the output(s) to
/// `extract --strlen` or `score --strlen` (or let `score --alignments` measure them itself).
#[derive(Debug, Args)]
pub(crate) struct StrlenArgs {
    /// The indexed VCF of the calls (the VCF given to `extract` or `score`).
    #[arg(long)]
    pub(crate) vcf: PathBuf,

    /// The indexed BAM or CRAM of the sample's reads.
    #[arg(long)]
    pub(crate) alignments: PathBuf,

    /// The indexed, uncompressed reference FASTA (also used to decode CRAM).
    #[arg(long)]
    pub(crate) reference: PathBuf,

    /// The table to write (`.gz` or `.bgz`, in any case, is BGZF-compressed).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// The regions to measure, as chrom or chrom:start-end (1-based, inclusive).
    #[arg(long, num_args = 1.., default_values = AUTOSOMES)]
    pub(crate) regions: Vec<String>,

    /// Which reads measure the tracts, as comma-separated settings over the defaults: mapqN
    /// (only reads with MAPQ >= N; the MAPQ summaries still cover every spanning read) and flankN
    /// (N aligned flank bases on each side), plus the experimental mates (count a read pair once)
    /// and unit (at least one repeat unit of flank). `legacy` (mapq0,flank2) measures as winnowr
    /// did before the settings were recorded, for models trained then. The settings are recorded
    /// in the table; a model scores only tables measured as its training tables were.
    #[arg(long, default_value_t = StrlenOptions::default())]
    pub(crate) reads: StrlenOptions,

    /// Experimental: more tables to write from the same pass over the reads, as SETTINGS=PATH
    /// (SETTINGS as for --reads).
    #[arg(long, num_args = 1.., value_name = "SETTINGS=PATH", hide = true)]
    pub(crate) also: Vec<AlsoOutput>,
}

/// One more `winnowr strlen` table, given as `SETTINGS=PATH` (see `--also`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AlsoOutput {
    /// The read settings to measure the table with.
    reads: StrlenOptions,
    /// The table to write.
    path: PathBuf,
}

impl std::str::FromStr for AlsoOutput {
    type Err = anyhow::Error;

    /// Parses `SETTINGS=PATH`, splitting at the first `=`.
    fn from_str(spec: &str) -> Result<Self> {
        let Ok(NamedPath { name, path }) = spec.parse::<NamedPath>() else {
            bail!("expected SETTINGS=PATH, got {spec:?}");
        };
        Ok(AlsoOutput { reads: name.parse()?, path })
    }
}

/// The options of `winnowr pileup`.
///
/// For every SNP and indel record (any FILTER) whose POS is in the regions, counts the reads that
/// support its reference allele, an alternate allele, or neither, in three MAPQ bins: 0, low
/// (from 1) and high (from 20, by default). Every MAPQ is counted, unlike HaplotypeCaller, which
/// ignores reads below its minimum MAPQ; reads that are unmapped, secondary, supplementary,
/// duplicates or QC-failed are not. A SNP's allele is the read's base at it (bases below the
/// minimum base quality are not counted); an indel's is the read's length change across the
/// indel's repeat tract (as `winnowr strlen` measures it, from reads aligned on flank bases
/// outside it), compared with each allele's. Records of other types (e.g. MNPs) and indels
/// without a pure insertion or deletion allele are skipped. Writes the settings and then one row
/// per record; give the output(s) to `extract --pileup` or `score --pileup` (or let `score
/// --alignments` count them itself).
#[derive(Debug, Args)]
pub(crate) struct PileupArgs {
    /// The indexed VCF of the calls (the VCF given to `extract` or `score`).
    #[arg(long)]
    pub(crate) vcf: PathBuf,

    /// The indexed BAM or CRAM of the sample's reads.
    #[arg(long)]
    pub(crate) alignments: PathBuf,

    /// The indexed, uncompressed reference FASTA (also used to decode CRAM).
    #[arg(long)]
    pub(crate) reference: PathBuf,

    /// The table to write (`.gz` or `.bgz`, in any case, is BGZF-compressed).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// The regions to measure, as chrom or chrom:start-end (1-based, inclusive).
    #[arg(long, num_args = 1.., default_values = AUTOSOMES)]
    pub(crate) regions: Vec<String>,

    /// How reads are counted, as comma-separated settings over the defaults: mapqN (the lowest
    /// MAPQ of the high bin), bqN (the minimum base quality of a SNP base) and flankN (the
    /// aligned flank bases an indel read needs on each side of the repeat tract). The settings
    /// are recorded in the table; a model scores only tables measured as its training tables
    /// were.
    #[arg(long, default_value_t = PileupOptions::default())]
    pub(crate) settings: PileupOptions,
}

/// Checks that the directory an output file of `option` goes in exists, so that a mistyped path
/// fails before a long pass over the inputs rather than after it. Advisory: the writers still
/// report a directory that goes missing.
///
/// # Errors
///
/// Returns an error naming the option and the directory if it does not exist.
fn check_output_dir(option: &str, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty())
        && !parent.is_dir()
    {
        bail!("the output directory {} (of {option}) does not exist", parent.display());
    }
    Ok(())
}

/// Runs `winnowr pileup`.
fn pileup(args: &PileupArgs) -> Result<()> {
    check_output_dir("--output", &args.output)?;
    let started = std::time::Instant::now();
    let inputs =
        ReadInputs { vcf: &args.vcf, alignments: &args.alignments, reference: &args.reference };
    let rows = measure_pileup(&inputs, &args.regions, &args.settings)?;
    let n = write_pileup_table(&args.output, &rows, args.settings)?;
    let seconds = started.elapsed().as_secs_f64();
    info!(
        "Wrote {n} records counted with {} to {} in {seconds:.1} s",
        args.settings,
        args.output.display()
    );
    Ok(())
}

/// Runs `winnowr strlen`.
fn strlen(args: &StrlenArgs) -> Result<()> {
    let started = std::time::Instant::now();
    let inputs =
        ReadInputs { vcf: &args.vcf, alignments: &args.alignments, reference: &args.reference };
    let mut outputs: Vec<(StrlenOptions, PathBuf)> = vec![(args.reads, args.output.clone())];
    outputs.extend(args.also.iter().map(|also| (also.reads, also.path.clone())));
    check_output_dir("--output", &args.output)?;
    for (_, path) in &outputs[1..] {
        check_output_dir("--also", path)?;
    }
    // Each table is written in turn, so a repeated path would keep only the last.
    if let Some(path) = repeated(outputs.iter().map(|(_, path)| path)).first() {
        bail!("{} is given as more than one strlen output", path.display());
    }
    let options: Vec<StrlenOptions> = outputs.iter().map(|(options, _)| *options).collect();
    let per_option = measure_regions_with(&inputs, &args.regions, &options)?;
    let n_sites = per_option.first().map_or(0, Vec::len);
    for ((options, path), sites) in outputs.iter().zip(&per_option) {
        let n = write_strlen_table(path, sites, *options)?;
        info!("Wrote {n} sites measured with {options} to {}", path.display());
    }
    let seconds = started.elapsed().as_secs_f64();
    info!(
        "Measured {n_sites} sites in {seconds:.1} s ({:.2} ms/site)",
        1000.0 * seconds / n_sites.max(1) as f64
    );
    Ok(())
}

/// Runs `winnowr extract`.
fn extract(args: &ExtractArgs) -> Result<()> {
    check_output_dir("--output", &args.output)?;
    let truth = TruthInputs {
        happy: args.happy.as_deref(),
        truth_vcf: args.truth_vcf.as_deref(),
        truth_bed: args.truth_bed.as_deref(),
        reference: args.reference.as_deref(),
    };
    truth.check()?;
    let regions = read_region_sets(&args.regions)?;
    let summary = VcfSummary::read(&args.vcf)?;
    let sample = args.sample.clone().unwrap_or_else(|| summary.sample_name(&args.vcf));
    let strlen = if args.strlen.is_empty() {
        None
    } else {
        Some(read_tables::<StrlenOptions>(&args.strlen)?)
    };
    if let Some(strlen) = &strlen {
        info!("The strlen tables were measured with read settings {}", strlen.options);
    }
    let pileup = if args.pileup.is_empty() {
        None
    } else {
        Some(read_tables::<PileupOptions>(&args.pileup)?)
    };
    if let Some(pileup) = &pileup {
        info!("The pileup tables were measured with settings {}", pileup.options);
    }
    let inputs = TableInputs {
        gvcf: args.gvcf.as_deref(),
        regions: &regions,
        strlen: strlen.as_ref(),
        pileup: pileup.as_ref(),
        truth,
    };
    let table = build_feature_table(&sample, &args.vcf, &args.contigs, &summary, &inputs)?;
    let mut timer = PhaseTimer::start();
    table.write_parquet(&args.output)?;
    timer.lap("writing the table");
    info!("Wrote {} calls for {sample} to {}", table.num_rows(), args.output.display());
    Ok(())
}

/// Reads the training feature tables' metadata and checks that the tables can be trained on
/// together, returning their shared region names and contigs.
///
/// # Errors
///
/// Returns an error if there are fewer than two tables, a file is listed twice, or the tables
/// were extracted with different regions or contigs.
fn read_training_infos(
    paths: &[PathBuf],
) -> Result<(Vec<FeatureTableInfo>, Vec<String>, Vec<String>)> {
    if paths.len() < 2 {
        bail!("--tables needs at least two feature tables (one per training sample)");
    }
    let resolved: Vec<PathBuf> = paths
        .iter()
        .map(|path| path.canonicalize().with_context(|| format!("cannot read {}", path.display())))
        .collect::<Result<_>>()?;
    let repeated = repeated(resolved.iter().map(|path| path.display().to_string()));
    if !repeated.is_empty() {
        bail!("--tables lists the same file more than once: {}", repeated.join(", "));
    }
    let infos: Vec<FeatureTableInfo> =
        paths.iter().map(FeatureTable::read_info).collect::<Result<_>>()?;
    let first = &infos[0];
    for info in &infos[1..] {
        if info.region_names != first.region_names {
            bail!(
                "--tables were extracted with different regions: {} has [{}], {} has [{}]",
                first.sample,
                first.region_names.join(", "),
                info.sample,
                info.region_names.join(", ")
            );
        }
        if info.contigs != first.contigs {
            bail!(
                "--tables were extracted from different contigs: {} has [{}], {} has [{}]",
                first.sample,
                first.contigs.join(", "),
                info.sample,
                info.contigs.join(", ")
            );
        }
    }
    let (regions, contigs) = (first.region_names.clone(), first.contigs.clone());
    Ok((infos, regions, contigs))
}

/// The reference contigs the training tables share, to record in the model (`None`, with a
/// warning, if a table does not record them).
///
/// # Errors
///
/// Returns an error if two tables record different contig lengths (different references).
fn shared_reference_contigs(
    infos: &[&FeatureTableInfo],
) -> Result<Option<indexmap::IndexMap<String, Option<u64>>>> {
    if infos.iter().any(|info| info.reference_contigs.is_none()) {
        log::warn!(
            "some --tables do not record their reference contigs (extracted by an older \
             winnowr); the model will not record them either, so score cannot check the reference"
        );
        return Ok(None);
    }
    let first = infos[0];
    for info in &infos[1..] {
        if info.reference_contigs != first.reference_contigs {
            bail!(
                "--tables come from different references: {} and {} record different contig \
                 lengths",
                first.sample,
                info.sample
            );
        }
    }
    Ok(first.reference_contigs.clone())
}

/// The training parameters of a `train` command line.
///
/// # Errors
///
/// Returns an error if --unfixable-flips is given for a promote model, or --validation-fraction is
/// not in `[0, 1)`.
fn training_params(args: &TrainArgs, threads: NonZeroUsize) -> Result<TrainingParams> {
    let params = TrainingParams {
        learning_rate: args.learning_rate,
        max_leaves: args.max_leaves,
        max_depth: args.max_depth,
        min_child_weight: args.min_child_weight,
        reg_lambda: args.reg_lambda,
        max_rounds: args.max_rounds,
        early_stopping_rounds: args.early_stopping_rounds,
        validation_fraction: args.validation_fraction,
        validation_block_size: args.validation_block_size,
        seed: args.seed,
        threads: u32::try_from(threads.get())?,
        unfixable_flips: args.unfixable_flips,
        // Without early stopping the final model already fits every call.
        refit_final: args.refit_final && args.validation_fraction > 0.0,
        cv_folds: if args.cv_folds.eq_ignore_ascii_case("chromosomes") {
            CvFolds::Chromosomes { folds: args.chromosome_folds }
        } else {
            CvFolds::Samples
        },
    };
    if !params.cv_folds.is_default() && !args.sample_groups.is_empty() {
        log::warn!("--sample-groups is unused with --cv-folds chromosomes");
    }
    if args.task == Task::Promote && !args.unfixable_flips.is_default() {
        bail!("--unfixable-flips applies to genotype models, not --task promote");
    }
    if !(0.0..1.0).contains(&params.validation_fraction) {
        bail!(
            "--validation-fraction must be at least 0 and below 1, got {}",
            params.validation_fraction
        );
    }
    Ok(params)
}

/// Runs `winnowr train`.
fn train(args: &TrainArgs, threads: NonZeroUsize) -> Result<()> {
    let wait = winnowr_xgboost::wait_between_parallel_regions(threads);
    log::debug!("OpenMP threads wait between parallel regions: {wait}");
    let params = training_params(args, threads)?;
    let use_strlen = args.repeat_length || args.regenotype;
    if use_strlen && (args.variant_type != VariantType::Indel || args.task != Task::Genotype) {
        bail!(
            "repeat-length features (--repeat-length, or --regenotype) need --variant-type indel \
             and --task genotype, got --variant-type {} --task {}",
            args.variant_type,
            args.task
        );
    }
    let output = if args.output.to_string_lossy().ends_with(MODEL_SUFFIX) {
        args.output.clone()
    } else {
        PathBuf::from(format!("{}{MODEL_SUFFIX}", args.output.display()))
    };
    // Checked now rather than after hours of training.
    check_output_dir("--output", &output)?;
    let (infos, regions, contigs) = read_training_infos(&args.tables)?;
    let info_refs: Vec<&FeatureTableInfo> = infos.iter().collect();
    let reference_contigs = shared_reference_contigs(&info_refs)?;
    let groups: Vec<Vec<String>> =
        args.sample_groups.iter().map(|g| g.split(',').map(str::to_string).collect()).collect();
    let names: Vec<&str> = infos.iter().map(|info| info.sample.as_str()).collect();
    group_samples(&names, &groups)?; // fail before any fitting
    let strlen_options =
        if use_strlen { Some(shared_options::<StrlenOptions>(&info_refs)?) } else { None };
    let pileup_options = if args.pileup_features {
        Some(shared_options::<PileupOptions>(&info_refs)?)
    } else {
        None
    };
    let mut extra_features: Vec<&str> = if use_strlen {
        STRLEN_FEATURES.iter().chain(&LENGTH_CALL_FEATURES).copied().collect()
    } else {
        Vec::new()
    };
    if args.pileup_features {
        extra_features.extend(PILEUP_FEATURES);
    }
    let features = feature_names(
        args.variant_type,
        &regions,
        !args.no_neighbourhood,
        args.task,
        &extra_features,
    );
    // Only the columns training reads, and only the calls of the variant type.
    let selection = TableSelection {
        columns: Some(table_columns(&features)),
        variant_type: Some(args.variant_type),
    };
    let read = |path: &PathBuf| FeatureTable::read_parquet_selected(path, &selection);
    let spec = TrainingSpec {
        variant_type: args.variant_type,
        features: &features,
        region_names: &regions,
        contigs: &contigs,
        reference_contigs,
        params,
        groups: &groups,
        threads,
    };
    let mut timer = PhaseTimer::start();
    let model = if args.task == Task::Promote {
        // Tables are read one at a time (each in parallel) and reduced to their candidates, so
        // only one is ever held whole.
        let samples = args
            .tables
            .iter()
            .map(|path| PromoteSampleCalls::from_table(&read(path)?, args.variant_type))
            .collect::<Result<Vec<_>>>()?;
        timer.lap("reading the tables");
        for sample in &samples {
            let mut counts = [0usize; 3];
            for label in &sample.labels {
                counts[label.index()] += 1;
            }
            info!(
                "{} {}: {} candidates, labels stay={} promote={} promote_flip={}",
                sample.sample,
                args.variant_type,
                sample.labels.len(),
                counts[0],
                counts[1],
                counts[2]
            );
        }
        let (promote_grid, flip_grid): (&[f64], &[f64]) = if args.grid.is_empty() {
            (&DEFAULT_PROMOTE_GRID, &DEFAULT_PROMOTE_FLIP_GRID)
        } else {
            (&args.grid, &args.grid)
        };
        train_promote_model(&samples, &spec, promote_grid, flip_grid)?
    } else {
        let grid: &[f64] = if args.grid.is_empty() { &DEFAULT_GRID } else { &args.grid };
        if use_strlen {
            // Each fold refits the stutter model from its training tables, so they are kept.
            let tables: Vec<FeatureTable> = args.tables.iter().map(read).collect::<Result<_>>()?;
            timer.lap("reading the tables");
            let pairs: Vec<(&FeatureTable, &FeatureTableInfo)> =
                tables.iter().map(|table| (table, table.info())).collect();
            let stutter = fit_stutter_from_feature_tables(&pairs, args.stutter_fit)?;
            let build = |stutter: &StutterModel| -> Result<Vec<SampleCalls>> {
                tables
                    .iter()
                    .map(|table| {
                        let mut sample = SampleCalls::from_table(
                            table,
                            args.variant_type,
                            Some(stutter),
                            args.regenotype,
                        )?;
                        sample.apply_unfixable_flips(args.unfixable_flips)?;
                        Ok(sample)
                    })
                    .collect()
            };
            let refit = |training: &FoldTraining| -> Result<Vec<SampleCalls>> {
                build(&fit_fold_stutter(&pairs, training, args.stutter_fit)?)
            };
            let samples = build(&stutter)?;
            timer.lap("fitting the stutter model and building the samples");
            log_samples(&samples, args);
            train_model(&samples, &spec, grid, Some(stutter), args.regenotype, Some(&refit))?
        } else {
            // Tables are read one at a time (each in parallel) and reduced to their labelled calls.
            let samples = args
                .tables
                .iter()
                .map(|path| {
                    let table = read(path)?;
                    let mut sample =
                        SampleCalls::from_table(&table, args.variant_type, None, false)?;
                    sample.apply_unfixable_flips(args.unfixable_flips)?;
                    Ok(sample)
                })
                .collect::<Result<Vec<_>>>()?;
            timer.lap("reading the tables");
            log_samples(&samples, args);
            train_model(&samples, &spec, grid, None, false, None)?
        }
    };
    let mut model = model;
    model.info.strlen_settings = strlen_options;
    model.info.pileup_settings = pileup_options;
    let mut timer = PhaseTimer::start();
    model.save(&output)?;
    timer.lap("saving the model");
    info!("Wrote model {}", output.display());
    Ok(())
}

/// Logs each training sample's calls, baseline counts and (re-genotype models) labels.
fn log_samples(samples: &[SampleCalls], args: &TrainArgs) {
    for sample in samples {
        let b = sample.baseline;
        info!(
            "{} {}: {} calls TP={} FP={} FN={} F1={:.5}",
            sample.sample,
            args.variant_type,
            sample.labels.len(),
            b.tp,
            b.fp,
            b.fn_,
            b.f1()
        );
        let mut counts = [0usize; 4];
        for label in &sample.labels {
            counts[label.index()] += 1;
        }
        if args.regenotype {
            info!(
                "{} {}: labels drop={} keep={} flip={} regenotype={}",
                sample.sample, args.variant_type, counts[0], counts[1], counts[2], counts[3]
            );
        }
        if !args.unfixable_flips.is_default() {
            let excluded = sample.excluded.iter().filter(|&&e| e).count();
            info!(
                "{} {}: --unfixable-flips {}: training labels drop={} keep={} flip={}, {excluded} \
                 calls excluded from the fits",
                sample.sample,
                args.variant_type,
                args.unfixable_flips,
                counts[0],
                counts[1],
                counts[2]
            );
        }
    }
}

/// Runs `winnowr evaluate`.
fn evaluate(args: &EvaluateArgs, threads: NonZeroUsize) -> Result<()> {
    use winnowr::evaluation::{
        EvaluationPolicy, baseline_metric, evaluate_model, evaluate_promote_model,
        write_policy_metrics,
    };
    check_output_dir("--output", &args.output)?;
    let models = args
        .models
        .iter()
        .map(|path| WinnowrModel::load(path, threads))
        .collect::<Result<Vec<_>>>()?;
    let genotype: Vec<&WinnowrModel> =
        models.iter().filter(|m| m.info.task == Task::Genotype).collect();
    let promote: Vec<&WinnowrModel> =
        models.iter().filter(|m| m.info.task == Task::Promote).collect();
    let variant_type = models[0].info.variant_type;
    if genotype.len() > 1
        || promote.len() > 1
        || models.iter().any(|m| m.info.variant_type != variant_type)
    {
        let described: Vec<String> = args
            .models
            .iter()
            .zip(&models)
            .map(|(path, m)| {
                format!("{} ({} {})", path.display(), m.info.task, m.info.variant_type)
            })
            .collect();
        bail!(
            "--models takes at most one genotype and one promote model of the same variant type, \
             got: {}",
            described.join(", ")
        );
    }
    let features: Vec<&String> = models.iter().flat_map(|m| &m.info.features).collect();
    let selection = TableSelection {
        columns: Some(table_columns(&features)),
        variant_type: Some(variant_type),
    };
    let table = FeatureTable::read_parquet_selected(&args.table, &selection)?;
    let info = table.info();
    check_table_reference(&args.table, info, &models, &args.models)?;
    check_table::<StrlenOptions>(&args.table, info, &models, &args.models)?;
    check_table::<PileupOptions>(&args.table, info, &models, &args.models)?;
    let trained_on: Vec<String> = args
        .models
        .iter()
        .zip(&models)
        .filter(|(_, m)| m.info.training_samples.contains(&info.sample))
        .map(|(path, _)| path.display().to_string())
        .collect();
    if !trained_on.is_empty() {
        log::warn!(
            "{} is a training sample of {}: its metrics are in-sample and optimistic, not held-out",
            info.sample,
            trained_on.join(", ")
        );
    }
    warn_policy_only_regions(&args.regions);
    // The mode is a placeholder: evaluate reports both.
    let policy = args.policy.policy(
        Mode::Genotyper,
        &info.contigs,
        &args.regions,
        !genotype.is_empty() && variant_type == VariantType::Indel,
    )?;
    let stutter = genotype.first().and_then(|m| m.info.stutter.as_ref());
    let regenotype = genotype.first().is_some_and(|m| m.info.regenotype);
    let calls = SampleCalls::from_table(&table, variant_type, stutter, regenotype)?;
    let vt = variant_type.as_str();
    let mut metrics = match genotype.first() {
        Some(model) => {
            let probabilities = model.predict(&calls.calls)?;
            let thresholds = |mode: Mode| model.info.thresholds[&mode].decision();
            let limits = policy.call_limits(&calls.calls, variant_type)?;
            let acting =
                EvaluationPolicy { rule: policy.rule(variant_type), limits: limits.as_deref() };
            evaluate_model(&calls, &probabilities, vt, thresholds, args.tuned, acting)?
        }
        None => vec![baseline_metric(&calls, vt)?],
    };
    if let Some(model) = promote.first() {
        let chosen = model.info.promote_thresholds.context("a promote model has thresholds")?;
        let candidates = PromoteSampleCalls::from_table(&table, variant_type)?;
        let probabilities = model.predict(&candidates.calls)?;
        let withheld = policy.withheld_promotions(&candidates.calls)?;
        let promoted = evaluate_promote_model(
            &candidates,
            &probabilities,
            chosen.t_promote,
            chosen.t_flip,
            &metrics,
            args.tuned,
            withheld.as_deref(),
        )?;
        metrics.extend(promoted);
    }
    for m in &metrics {
        let promoted = if m.t_promote < winnowr::common::NEVER {
            format!(
                " t_promote={} t_promote_flip={} promoted={} (correct {}, flipped {})",
                m.t_promote, m.t_promote_flip, m.promoted, m.promoted_correct, m.promoted_flipped
            )
        } else {
            String::new()
        };
        let regenotyped = if regenotype {
            format!(
                " t_regenotype={} regenotyped={} (fixed {}, broke {})",
                m.t_regenotype, m.regenotyped, m.regenotypes_fixed, m.regenotypes_broke_tp
            )
        } else {
            String::new()
        };
        info!(
            "{} {} {:17} t_drop={} t_flip={} drop={} (TP {}) flip={} (fixed {}, broke {}){}{} \
             TP={} FP={} FN={} F1={:.5} ({:+.5})",
            m.sample,
            m.variant_type,
            m.policy,
            m.t_drop,
            m.t_flip,
            m.dropped,
            m.dropped_tp,
            m.flipped,
            m.flips_fixed,
            m.flips_broke_tp,
            regenotyped,
            promoted,
            m.counts.tp,
            m.counts.fp,
            m.counts.fn_,
            m.counts.f1(),
            m.f1_gain
        );
    }
    write_policy_metrics(&args.output, &metrics)
}

/// Checks a feature table's reference contigs against each model's, warning if the table does
/// not record them.
///
/// # Errors
///
/// Returns an error if the table is not on a model's reference.
fn check_table_reference(
    table: &Path,
    info: &FeatureTableInfo,
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Result<()> {
    let Some(contigs) = &info.reference_contigs else {
        log::warn!(
            "{} does not record its reference contigs (extracted by an older winnowr), so it \
             cannot be checked against the models' reference",
            table.display()
        );
        return Ok(());
    };
    let source = format!("feature table {}", table.display());
    for (path, model) in paths.iter().zip(models) {
        check_reference(path, &model.info, contigs, &source, false)?;
    }
    Ok(())
}

/// Warns about --regions that only `score`'s models use: `evaluate` and `stratify` read the
/// models' region features from the table, and only the confident regions from --regions.
fn warn_policy_only_regions(regions: &[NamedBed]) {
    let ignored: Vec<&str> =
        regions.iter().map(|r| r.name.as_str()).filter(|n| *n != CONFIDENT_REGIONS).collect();
    if !ignored.is_empty() {
        log::warn!(
            "--regions {} are ignored: region features come from the feature table",
            ignored.join(", ")
        );
    }
}

/// Runs `winnowr stratify`.
fn stratify(args: &StratifyArgs, threads: NonZeroUsize) -> Result<()> {
    use winnowr::regions::RegionSet;
    use winnowr::regions::read_stratification_manifest;
    use winnowr::strata::{
        CALL_KEY_COLUMNS, CallOutputs, TRUTH_LABEL_COLUMNS, TruthSets, check_region_names, stratify,
    };
    check_output_dir("--output", &args.output)?;
    if let Some(calls) = &args.calls {
        check_output_dir("--calls", calls)?;
    }
    if let Some(dir) = args.labelled_calls.as_deref().filter(|dir| !dir.is_dir()) {
        bail!("the --labelled-calls directory {} does not exist", dir.display());
    }
    let models = args
        .models
        .iter()
        .map(|path| WinnowrModel::load(path, threads))
        .collect::<Result<Vec<_>>>()?;
    let names: [String; 2] = [args.truth_names[0].clone(), args.truth_names[1].clone()];
    if names[0] == names[1] {
        bail!("--truth-names must differ, got {} twice", names[0]);
    }
    let features: Vec<&String> = models.iter().flat_map(|m| &m.info.features).collect();
    let mut columns = table_columns(&features);
    columns.extend(TRUTH_LABEL_COLUMNS.iter().map(ToString::to_string));
    let table = FeatureTable::read_parquet_selected(
        &args.tables[0],
        &TableSelection { columns: Some(columns), variant_type: None },
    )?;
    let label_columns: Vec<String> =
        CALL_KEY_COLUMNS.iter().chain(&TRUTH_LABEL_COLUMNS).map(ToString::to_string).collect();
    let second = FeatureTable::read_parquet_selected(
        &args.tables[1],
        &TableSelection { columns: Some(label_columns), variant_type: None },
    )?;
    let (table, second) = if args.contigs.is_empty() {
        (table, second)
    } else {
        // Each table with its calls and FN counts on the contigs, listing only those contigs, so
        // that the truth FNs and per-chromosome rates are of those contigs too.
        let restrict = |table: &FeatureTable, happy: &PathBuf| -> Result<FeatureTable> {
            let mut restricted = restrict_table(table, happy, &args.contigs, None)?;
            let info = restricted.info_mut();
            info.contigs.retain(|contig| args.contigs.contains(contig));
            Ok(restricted)
        };
        (restrict(&table, &args.happy[0])?, restrict(&second, &args.happy[1])?)
    };
    let info = table.info();
    if second.info().sample != info.sample {
        bail!("--tables are of different samples: {} and {}", info.sample, second.info().sample);
    }
    check_table_reference(&args.tables[0], info, &models, &args.models)?;
    check_table::<StrlenOptions>(&args.tables[0], info, &models, &args.models)?;
    check_table::<PileupOptions>(&args.tables[0], info, &models, &args.models)?;
    let trained_on: Vec<String> = args
        .models
        .iter()
        .zip(&models)
        .filter(|(_, m)| m.info.training_samples.contains(&info.sample))
        .map(|(path, _)| path.display().to_string())
        .collect();
    if !trained_on.is_empty() {
        log::warn!(
            "{} is a training sample of {}: its metrics are in-sample and optimistic, not held-out",
            info.sample,
            trained_on.join(", ")
        );
    }
    let beds = [
        RegionSet::from_bed(&names[0], &args.truth_beds[0])?,
        RegionSet::from_bed(&names[1], &args.truth_beds[1])?,
    ];
    for bed in &beds {
        bed.check_contigs(&info.contigs)?;
    }
    let mut named = match &args.stratification {
        Some(path) => read_stratification_manifest(path)?,
        None => Vec::new(),
    };
    named.extend(args.report_region.iter().cloned());
    check_region_names(&named.iter().map(|named| named.name.as_str()).collect::<Vec<_>>())?;
    let regions = named
        .iter()
        .map(|named| RegionSet::from_bed(&named.name, &named.path))
        .collect::<Result<Vec<_>>>()?;
    for region in &regions {
        if let Err(e) = region.check_contigs(&info.contigs) {
            log::warn!("region {}: {e:#}", region.name());
        }
    }
    let truths = TruthSets {
        names: &names,
        beds: [&beds[0], &beds[1]],
        happy: [&args.happy[0], &args.happy[1]],
    };
    warn_policy_only_regions(&args.regions);
    if !args.regions.iter().any(|r| r.name == CONFIDENT_REGIONS)
        && named.iter().any(|named| named.name == CONFIDENT_REGIONS)
    {
        log::warn!(
            "{CONFIDENT_REGIONS} is reported within but not given to the policy: pass --regions \
             {CONFIDENT_REGIONS}=PATH to decide calls as score does with it"
        );
    }
    let policy =
        args.policy.policy(args.mode, &info.contigs, &args.regions, has_indel_genotype(&models))?;
    let outputs =
        CallOutputs { calls: args.calls.as_deref(), labelled_dir: args.labelled_calls.as_deref() };
    let (report, by_region) =
        stratify(&table, &second, &models, &policy, truths, &regions, outputs)?;
    for line in report.summary_lines() {
        info!("{line}");
    }
    for line in by_region.summary_lines() {
        info!("{line}");
    }
    let path = |suffix: &str| PathBuf::from(format!("{}.{suffix}", args.output.display()));
    report.write_outcomes(&path("strata.tsv"))?;
    report.write_rates(&path("rates.tsv"))?;
    report.write_calibration(&path("calibration.tsv"))?;
    info!("Wrote {}.{{strata,rates,calibration}}.tsv", args.output.display());
    if !by_region.is_empty() {
        by_region.write_outcomes(&path("regions.strata.tsv"))?;
        by_region.write_rates(&path("regions.rates.tsv"))?;
        by_region.write_calibration(&path("regions.calibration.tsv"))?;
        info!(
            "Wrote {}.regions.{{strata,rates,calibration}}.tsv for {} regions",
            args.output.display(),
            regions.len()
        );
    }
    Ok(())
}

/// Runs `winnowr subset`.
fn subset(args: &SubsetArgs) -> Result<()> {
    use winnowr::regions::RegionSet;
    check_output_dir("--output", &args.output)?;
    let table = FeatureTable::read_parquet(&args.table)?;
    let contigs =
        if args.contigs.is_empty() { table.info().contigs.clone() } else { args.contigs.clone() };
    let exclude =
        args.exclude_bed.as_ref().map(|path| RegionSet::from_bed("exclude", path)).transpose()?;
    let restricted = restrict_table(&table, &args.happy, &contigs, exclude.as_ref())?;
    restricted.write_parquet(&args.output)?;
    let info = restricted.info();
    info!(
        "Wrote {} of {} calls for {} to {} (truth FNs: {} snp, {} indel)",
        restricted.num_rows(),
        table.num_rows(),
        info.sample,
        args.output.display(),
        info.fn_count(VariantType::Snp)?,
        info.fn_count(VariantType::Indel)?
    );
    Ok(())
}

/// Runs `winnowr score`.
fn score(args: &ScoreArgs, threads: NonZeroUsize) -> Result<()> {
    check_output_dir("--output", &args.output)?;
    let models = args
        .models
        .iter()
        .map(|path| WinnowrModel::load(path, threads))
        .collect::<Result<Vec<_>>>()?;
    let supplied: Vec<&str> = args.regions.iter().map(|r| r.name.as_str()).collect();
    for (path, model) in args.models.iter().zip(&models) {
        let mut missing: Vec<&str> = model
            .info
            .region_names
            .iter()
            .map(String::as_str)
            .filter(|name| !supplied.contains(name))
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            bail!(
                "model {} ({} {}) needs region BEDs (--regions NAME=PATH) for: {}",
                path.display(),
                model.info.task,
                model.info.variant_type,
                missing.join(", ")
            );
        }
    }
    let mut unused: Vec<&str> = supplied
        .iter()
        .copied()
        .filter(|name| {
            *name != CONFIDENT_REGIONS
                && !models.iter().any(|m| m.info.region_names.iter().any(|n| n == name))
        })
        .collect();
    if !unused.is_empty() {
        unused.sort_unstable();
        log::warn!("no model uses the --regions {}; they are ignored", unused.join(", "));
    }
    let needs_strlen = models_using::<StrlenOptions>(&models, &args.models);
    let needs_pileup = models_using::<PileupOptions>(&models, &args.models);
    check_supplied::<StrlenOptions>(&needs_strlen, &args.strlen, args.alignments.is_some())?;
    check_supplied::<PileupOptions>(&needs_pileup, &args.pileup, args.alignments.is_some())?;
    if needs_strlen.is_empty() && needs_pileup.is_empty() && args.alignments.is_some() {
        log::warn!("no model uses repeat-length or pileup features; --alignments is ignored");
    }
    // The settings the models' evidence must be measured with.
    let strlen_options = models_options::<StrlenOptions>(&models, &args.models)?;
    let strlen = read_for_models::<StrlenOptions>(&args.strlen, &models, &args.models)?;
    let pileup_options = models_options::<PileupOptions>(&models, &args.models)?;
    let pileup = read_for_models::<PileupOptions>(&args.pileup, &models, &args.models)?;
    // Evidence measured from the reads while scoring, as the models need it (--alignments
    // conflicts with --strlen and --pileup, so nothing is both read and measured).
    let measure = match (&args.alignments, &args.reference) {
        (Some(alignments), Some(reference)) => {
            // Some exactly when a model uses the evidence.
            let (strlen, pileup) = (strlen_options, pileup_options);
            if let Some(options) = strlen {
                info!(
                    "Measuring the repeat-length evidence with the models' read settings {options}"
                );
            }
            if let Some(options) = pileup {
                info!("Counting the pileup evidence with the models' settings {options}");
            }
            (strlen.is_some() || pileup.is_some()).then_some((
                ReadInputs { vcf: &args.vcf, alignments, reference },
                strlen,
                pileup,
            ))
        }
        _ => None,
    };
    let contigs = &models[0].info.contigs;
    if models.iter().any(|model| &model.info.contigs != contigs) {
        let described: Vec<String> = args
            .models
            .iter()
            .zip(&models)
            .map(|(path, m)| format!("{}: {}", path.display(), m.info.contigs.join(",")))
            .collect();
        bail!("--models were trained on different contigs: {}", described.join("; "));
    }
    let summary = VcfSummary::read(&args.vcf)?;
    let source = format!("VCF {}", args.vcf.display());
    for (path, model) in args.models.iter().zip(&models) {
        check_reference(path, &model.info, &summary.contig_lengths, &source, true)?;
    }
    // Only the region BEDs a model uses are read, so an unused one cannot fail the run.
    let used: Vec<NamedBed> = args
        .regions
        .iter()
        .filter(|r| models.iter().any(|m| m.info.region_names.contains(&r.name)))
        .cloned()
        .collect();
    let regions = read_region_sets(&used)?;
    let sample = summary.sample_name(&args.vcf);
    let inputs = TableInputs {
        gvcf: args.gvcf.as_deref(),
        regions: &regions,
        strlen: strlen.as_ref(),
        pileup: pileup.as_ref(),
        ..TableInputs::default()
    };
    let table = build_feature_table(&sample, &args.vcf, contigs, &summary, &inputs)?;
    let mut timer = PhaseTimer::start();
    let policy =
        args.policy.policy(args.mode, contigs, &args.regions, has_indel_genotype(&models))?;
    let decisions = match measure {
        None => decide_calls(&table, &models, &policy)?,
        Some((measure, strlen, pileup)) => {
            let measuring = Measuring { inputs: &measure, strlen, pileup, contigs };
            measure_and_decide(table, &models, &policy, &measuring, threads, &mut timer)?
        }
    };
    timer.lap("scoring");
    write_scored_vcf(&args.vcf, &args.output, decisions, threads)?;
    timer.lap("writing the VCF");
    Ok(())
}

/// Run a parsed command line.
///
/// # Errors
///
/// Returns an error if the thread pool cannot be set up or the subcommand fails.
pub fn run(cli: &Cli) -> Result<()> {
    info!("Running winnowr {} {}", VERSION.as_str(), cli.command.name());
    info!("Command line: {}", command_line(std::env::args_os()));
    let (threads, from) = cli.global.threads(|name| std::env::var(name).ok());
    match from {
        ThreadsFrom::Option => info!("Using {threads} threads (-t)"),
        ThreadsFrom::Environment => info!("Using {threads} threads ({THREADS_VARIABLE})"),
        ThreadsFrom::Default => info!("Using {threads} threads (the default; set -t to choose)"),
    }
    configure_threads(threads)?;
    // Only training needs the full OpenMP team (see check_openmp_environment).
    if matches!(cli.command, Command::Train(_)) {
        winnowr_xgboost::check_openmp_environment(threads, |name| std::env::var(name).ok())
            .map_err(anyhow::Error::msg)?;
    }
    dispatch(&cli.command, threads)
}

/// A command line as one string, for the log: the arguments separated by spaces, each quoted
/// when it is empty or holds whitespace or quotes.
fn command_line<I: IntoIterator<Item = std::ffi::OsString>>(args: I) -> String {
    let quote = |arg: std::ffi::OsString| {
        let arg = arg.to_string_lossy().into_owned();
        let plain = !arg.is_empty()
            && !arg.chars().any(|c| c.is_whitespace() || matches!(c, '\'' | '"' | '\\'));
        if plain { arg } else { format!("'{}'", arg.replace('\'', r"'\''")) }
    };
    args.into_iter().map(quote).collect::<Vec<_>>().join(" ")
}

/// Run a subcommand.
fn dispatch(command: &Command, threads: NonZeroUsize) -> Result<()> {
    match command {
        Command::Extract(args) => extract(args),
        Command::Score(args) => score(args, threads),
        Command::Strlen(args) => strlen(args),
        Command::Pileup(args) => pileup(args),
        Command::Train(args) => train(args, threads),
        Command::Evaluate(args) => evaluate(args, threads),
        Command::Stratify(args) => stratify(args, threads),
        Command::Subset(args) => subset(args),
    }
}

/// Size the global rayon pool to `threads`.
///
/// # Errors
///
/// Returns an error if the global pool was already initialized.
fn configure_threads(threads: NonZeroUsize) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.get())
        .build_global()
        .context("could not set up the thread pool")
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;
    use clap::error::ErrorKind;
    use rstest::rstest;

    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("winnowr").chain(args.iter().copied()))
    }

    #[rstest]
    #[case::plain(&["winnowr", "score", "--vcf", "a.vcf.gz"], "winnowr score --vcf a.vcf.gz")]
    #[case::spaces(&["winnowr", "--vcf", "my calls.vcf"], "winnowr --vcf 'my calls.vcf'")]
    #[case::empty(&["winnowr", ""], "winnowr ''")]
    #[case::quote(&["winnowr", "it's"], r"winnowr 'it'\''s'")]
    fn command_lines_are_logged_quoted(#[case] args: &[&str], #[case] expected: &str) {
        assert_eq!(command_line(args.iter().map(std::ffi::OsString::from)), expected);
    }

    /// The command lines the Python version took, as pipelines run them: `--output`, `--mode` in
    /// upper case, sidecar paths, NAME=PATH regions and one region per `strlen`.
    #[test]
    fn python_command_lines_parse() {
        let score = [
            "score",
            "--vcf",
            "S.vcf.gz",
            "--models",
            "models/snp.winnowr.json",
            "models/indel.winnowr.json",
            "--regions",
            "segdups=regions/segdups.bed.gz",
            "lowmappabilityall=regions/lowmappabilityall.bed.gz",
            "--mode",
            "VETO",
            "--strlen",
            "strlen/S.0001.strlen.tsv.gz",
            "strlen/S.0002.strlen.tsv.gz",
            "--output",
            "S.winnowr.vcf.gz",
        ];
        let cli = parse(&score).unwrap();
        assert_eq!(cli.global.threads, None);
        let Command::Score(args) = cli.command else { panic!("not score") };
        assert_eq!(args.models.len(), 2);
        assert_eq!(
            args.regions.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            ["segdups", "lowmappabilityall"]
        );
        assert_eq!(args.mode, Mode::Veto);
        assert_eq!(args.strlen.len(), 2);
        assert_eq!(args.output, PathBuf::from("S.winnowr.vcf.gz"));
        let genotyper = [&score[..10], &["GENOTYPER", "--output", "o.vcf.gz"]].concat();
        let Command::Score(args) = parse(&genotyper).unwrap().command else { panic!("not score") };
        assert_eq!(args.mode, Mode::Genotyper);

        let strlen = [
            "strlen",
            "--vcf",
            "S.vcf.gz",
            "--alignments",
            "S.bam",
            "--reference",
            "ref.fa",
            "--regions",
            "chr1:1-20000000",
            "--output",
            "S.0001.strlen.tsv.gz",
        ];
        let Command::Strlen(args) = parse(&strlen).unwrap().command else { panic!("not strlen") };
        assert_eq!(args.regions, ["chr1:1-20000000"]);
        assert_eq!(args.output, PathBuf::from("S.0001.strlen.tsv.gz"));
    }

    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    const EVALUATE: [&str; 7] = ["evaluate", "--models", "m", "--table", "t", "-o", "o"];

    #[test]
    fn evaluate_options_parse() {
        let cli = parse(&EVALUATE).unwrap();
        assert_eq!(cli.command.name(), "evaluate");
        assert_eq!(cli.global.threads(|_| None), (DEFAULT_THREADS, ThreadsFrom::Default));
        let Command::Evaluate(args) = cli.command else { panic!("not evaluate") };
        assert!(!args.tuned);
    }

    #[test]
    fn threads_can_precede_or_follow_the_subcommand() {
        let cli = parse(&[&EVALUATE[..], &["--threads", "16"]].concat()).unwrap();
        assert_eq!(cli.global.threads.map(NonZeroUsize::get), Some(16));
        let cli = parse(&[&["-t", "2"][..], &EVALUATE].concat()).unwrap();
        assert_eq!(cli.global.threads.map(NonZeroUsize::get), Some(2));
    }

    #[rstest]
    #[case::unset(None, &[], 4, ThreadsFrom::Default)]
    #[case::from_the_environment(Some("16"), &[], 16, ThreadsFrom::Environment)]
    #[case::first_nesting_level(Some(" 8,2 "), &[], 8, ThreadsFrom::Environment)]
    #[case::option_wins(Some("16"), &["-t", "2"], 2, ThreadsFrom::Option)]
    #[case::empty(Some(""), &[], 4, ThreadsFrom::Default)]
    #[case::zero(Some("0"), &[], 4, ThreadsFrom::Default)]
    #[case::not_a_number(Some("many"), &[], 4, ThreadsFrom::Default)]
    fn threads_come_from_the_option_then_omp_num_threads(
        #[case] env: Option<&str>,
        #[case] option: &[&str],
        #[case] expected: usize,
        #[case] from: ThreadsFrom,
    ) {
        let cli = parse(&[option, &EVALUATE[..]].concat()).unwrap();
        let var =
            |name: &str| (name == THREADS_VARIABLE).then(|| env.map(str::to_string)).flatten();
        let (threads, source) = cli.global.threads(var);
        assert_eq!((threads.get(), source), (expected, from));
    }

    #[rstest]
    #[case::zero(&["-t", "0", "evaluate", "--models", "m", "--table", "t", "-o", "o"], ErrorKind::ValueValidation)]
    #[case::negative(&["--threads=-1", "evaluate", "--models", "m", "--table", "t", "-o", "o"], ErrorKind::ValueValidation)]
    #[case::not_a_number(&["--threads", "many", "evaluate", "--models", "m", "--table", "t", "-o", "o"], ErrorKind::ValueValidation)]
    #[case::missing_value(&["--threads"], ErrorKind::InvalidValue)]
    #[case::evaluate_without_models(&["evaluate", "--table", "t", "-o", "o"], ErrorKind::MissingRequiredArgument)]
    #[case::train_without_tables(&["train", "--variant-type", "snp", "-o", "m"], ErrorKind::MissingRequiredArgument)]
    #[case::train_zero_rounds(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--max-rounds", "0"], ErrorKind::ValueValidation)]
    #[case::train_zero_patience(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--early-stopping-rounds", "0"], ErrorKind::ValueValidation)]
    #[case::train_zero_block_size(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--validation-block-size", "0"], ErrorKind::ValueValidation)]
    #[case::train_huge_seed(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--seed", "9223372036854775808"], ErrorKind::ValueValidation)]
    #[case::train_bad_unfixable_flips(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--unfixable-flips", "keep"], ErrorKind::ValueValidation)]
    #[case::train_bad_variant_type(&["train", "--tables", "a", "--variant-type", "mnp", "-o", "m"], ErrorKind::ValueValidation)]
    #[case::score_without_models(&["score", "--vcf", "v", "-o", "o"], ErrorKind::MissingRequiredArgument)]
    #[case::score_bad_mode(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--mode", "x"], ErrorKind::ValueValidation)]
    #[case::score_alignments_without_reference(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--alignments", "a.cram"], ErrorKind::MissingRequiredArgument)]
    #[case::score_reference_without_alignments(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--reference", "r.fa"], ErrorKind::MissingRequiredArgument)]
    #[case::score_alignments_and_strlen(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--alignments", "a.cram", "--reference", "r.fa", "--strlen", "s.tsv"], ErrorKind::ArgumentConflict)]
    #[case::missing_subcommand(&[], ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand)]
    #[case::unknown_subcommand(&["rescore"], ErrorKind::InvalidSubcommand)]
    #[case::extract_without_vcf(&["extract", "-o", "t.parquet"], ErrorKind::MissingRequiredArgument)]
    #[case::extract_bad_region(&["extract", "--vcf", "v", "-o", "t", "--regions", "x"], ErrorKind::ValueValidation)]
    #[case::train_nan_grid(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--grid", "nan"], ErrorKind::ValueValidation)]
    #[case::train_infinite_grid(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--grid", "0.5", "inf"], ErrorKind::ValueValidation)]
    #[case::train_negative_grid(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--grid=-0.1"], ErrorKind::ValueValidation)]
    #[case::train_grid_above_one(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--grid", "1.5"], ErrorKind::ValueValidation)]
    #[case::train_one_chromosome_fold(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--chromosome-folds", "1"], ErrorKind::ValueValidation)]
    #[case::score_abstain_outside(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--abstain-outside", "c.bed"], ErrorKind::UnknownArgument)]
    #[case::stratify_bad_report_region(&["stratify", "--models", "m", "--truth-names", "a", "b", "--tables", "a", "b", "--truth-beds", "a", "b", "--happy", "a", "b", "-o", "o", "--report-region", "MHC"], ErrorKind::ValueValidation)]
    fn invalid_command_lines_are_rejected(#[case] args: &[&str], #[case] kind: ErrorKind) {
        assert_eq!(parse(args).unwrap_err().kind(), kind);
    }

    #[test]
    fn train_options_parse() {
        let cli =
            parse(&["train", "--tables", "a", "b", "--variant-type", "SNP", "-o", "m"]).unwrap();
        let Command::Train(args) = cli.command else { panic!("not train") };
        assert_eq!((args.variant_type, args.task), (VariantType::Snp, Task::Genotype));
        assert_eq!(args.tables.len(), 2);
        assert!(args.grid.is_empty() && !args.no_neighbourhood && !args.regenotype);
        assert_eq!(args.max_rounds, TrainingParams::default().max_rounds);
        let cli = parse(&[
            "train",
            "--tables",
            "a",
            "--variant-type",
            "indel",
            "--task",
            "PROMOTE",
            "-o",
            "m",
            "--grid",
            "0.5",
            "0.9",
            "--sample-groups",
            "A,B",
            "C,D",
        ])
        .unwrap();
        let Command::Train(args) = cli.command else { panic!("not train") };
        assert_eq!((args.variant_type, args.task), (VariantType::Indel, Task::Promote));
        assert_eq!(args.grid, [0.5, 0.9]);
        let train = ["train", "--tables", "a", "--variant-type", "snp", "-o", "m"];
        let bounds = parse(&[&train[..], &["--grid", "0", "1"]].concat()).unwrap();
        let Command::Train(bounds) = bounds.command else { panic!("not train") };
        assert_eq!(bounds.grid, [0.0, 1.0]);
        let error = parse(&[&train[..], &["--grid", "nan"]].concat()).unwrap_err().to_string();
        assert!(error.contains("must be a probability between 0 and 1, got nan"), "{error}");
        assert_eq!(args.sample_groups, ["A,B", "C,D"]);
        assert_eq!(args.unfixable_flips, UnfixableFlips::Flip);
        let cli = parse(&[
            "train",
            "--tables",
            "a",
            "--variant-type",
            "snp",
            "-o",
            "m",
            "--unfixable-flips",
            "Exclude",
        ])
        .unwrap();
        let Command::Train(args) = cli.command else { panic!("not train") };
        assert_eq!(args.unfixable_flips, UnfixableFlips::Exclude);
        assert_eq!((args.cv_folds.as_str(), args.chromosome_folds), ("samples", 5));
        assert!(args.refit_final);
        let cli = parse(&[
            "train",
            "--tables",
            "a",
            "--variant-type",
            "snp",
            "-o",
            "m",
            "--cv-folds",
            "Chromosomes",
            "--chromosome-folds",
            "4",
            "--refit-final",
        ])
        .unwrap();
        let Command::Train(args) = cli.command else { panic!("not train") };
        assert!(args.cv_folds.eq_ignore_ascii_case("chromosomes"));
        assert_eq!(args.chromosome_folds, 4);
        assert!(args.refit_final);
        let cli = parse(&[
            "train",
            "--tables",
            "a",
            "--variant-type",
            "snp",
            "-o",
            "m",
            "--refit-final",
            "false",
        ])
        .unwrap();
        let Command::Train(args) = cli.command else { panic!("not train") };
        assert!(!args.refit_final);
    }

    /// extract, train, evaluate and score on a small generated sample: the command bodies map
    /// their options as documented, and `score` decides as `evaluate` reports.
    mod end_to_end {
        use std::io::{Read as _, Write as _};

        use super::*;

        const HEADER: &str = "##fileformat=VCFv4.2
##contig=<ID=chr1,length=100000>
##FILTER=<ID=PASS,Description=\"All filters passed\">
##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">
##INFO=<ID=RAW_MQandDP,Number=2,Type=Integer,Description=\"Raw MQ and DP\">
##INFO=<ID=QUALapprox,Number=1,Type=Integer,Description=\"Approximate QUAL\">
##INFO=<ID=VarDP,Number=1,Type=Integer,Description=\"Variant depth\">
##INFO=<ID=AS_RAW_MQ,Number=1,Type=String,Description=\"Allele-specific raw MQ\">
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality\">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allele depths\">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
";

        const HAPPY_HEADER: &str = "##fileformat=VCFv4.2
##contig=<ID=chr1,length=100000>
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=BD,Number=1,Type=String,Description=\"Decision\">
##FORMAT=<ID=BK,Number=1,Type=String,Description=\"Match kind\">
##FORMAT=<ID=BVT,Number=1,Type=String,Description=\"Variant type\">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTRUTH\tQUERY
";

        /// The calls: 60 PASS SNPs, alternately true (high quality, balanced reads) and false.
        const CALLS: usize = 60;

        fn is_true(i: usize) -> bool {
            i.is_multiple_of(2)
        }

        fn write_bgzf_indexed(path: &Path, text: &str) {
            let mut writer = rust_htslib::bgzf::Writer::from_path(path).unwrap();
            writer.write_all(text.as_bytes()).unwrap();
            drop(writer);
            let index = rust_htslib::bcf::index::Type::Tbx;
            rust_htslib::bcf::index::build(path, None, 1, index).unwrap();
        }

        fn read_gzip(path: &Path) -> String {
            let mut text = String::new();
            flate2::read::MultiGzDecoder::new(std::fs::File::open(path).unwrap())
                .read_to_string(&mut text)
                .unwrap();
            text
        }

        fn run(args: &[&str]) {
            let cli = parse(args).unwrap();
            dispatch(&cli.command, NonZeroUsize::new(1).unwrap()).unwrap();
        }

        #[test]
        fn extract_train_evaluate_and_score() {
            let dir = tempfile::TempDir::new().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            let (mut vcf, mut happy) = (HEADER.to_string(), HAPPY_HEADER.to_string());
            for i in 0..CALLS {
                let pos = 1000 + 200 * i;
                let (qual, gq, ad, mq) =
                    if is_true(i) { (60, 99, "10,10", 3600) } else { (5, 5, "18,2", 400) };
                let (mq, dp) = (mq * 20, 20);
                vcf.push_str(&format!(
                    "chr1\t{pos}\t.\tA\tC\t{qual}\tPASS\tDP={dp};RAW_MQandDP={mq},{dp};QUALapprox={qual};VarDP={dp};AS_RAW_MQ={mq}|{mq}\tGT:GQ:AD\t0/1:{gq}:{ad}\n"
                ));
                let decision = if is_true(i) {
                    "0/1:TP:gm:SNP\t0/1:TP:gm:SNP"
                } else {
                    ".:.:.:NOCALL\t0/1:FP:.:SNP"
                };
                happy.push_str(&format!(
                    "chr1\t{pos}\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t{decision}\n"
                ));
            }
            for pos in [50_000, 50_100, 50_200] {
                happy.push_str(&format!(
                    "chr1\t{pos}\t.\tA\tG\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t.:.:.:NOCALL\n"
                ));
            }
            write_bgzf_indexed(&dir.path().join("calls.vcf.gz"), &vcf);
            std::fs::write(dir.path().join("happy.vcf"), happy).unwrap();
            for sample in ["S1", "S2"] {
                run(&[
                    "extract",
                    "--vcf",
                    &path("calls.vcf.gz"),
                    "--happy",
                    &path("happy.vcf"),
                    "--contigs",
                    "chr1",
                    "--sample",
                    sample,
                    "-o",
                    &path(&format!("{sample}.parquet")),
                ]);
            }
            run(&[
                "train",
                "--tables",
                &path("S1.parquet"),
                &path("S2.parquet"),
                "--variant-type",
                "SNP",
                "--validation-fraction",
                "0",
                "--max-rounds",
                "10",
                "--grid",
                "0.3",
                "0.5",
                "0.7",
                "-o",
                &path("snp"),
            ]);
            let model_path = path("snp.winnowr.json");
            let model = WinnowrModel::load(&model_path, NonZeroUsize::new(1).unwrap()).unwrap();
            let info = &model.info;
            assert_eq!((info.task, info.variant_type), (Task::Genotype, VariantType::Snp));
            assert_eq!(info.training_samples, ["S1", "S2"]);
            assert_eq!(info.contigs, ["chr1"]);
            // No early stopping, so nothing to refit.
            assert!(!info.params.refit_final);
            assert_eq!((info.params.max_rounds, info.params.validation_fraction), (10, 0.0));
            assert_eq!(info.params.cv_folds, CvFolds::Samples);
            for thresholds in info.thresholds.values() {
                for t in [thresholds.t_drop, thresholds.t_flip] {
                    assert!([0.3, 0.5, 0.7, winnowr::common::NEVER].contains(&t), "{t}");
                }
            }

            run(&[
                "evaluate",
                "--models",
                &model_path,
                "--table",
                &path("S1.parquet"),
                "-o",
                &path("metrics.tsv"),
            ]);
            let metrics = std::fs::read_to_string(path("metrics.tsv")).unwrap();
            let rows: Vec<Vec<&str>> = metrics.lines().map(|l| l.split('\t').collect()).collect();
            let column = |name: &str| rows[0].iter().position(|c| *c == name).unwrap();
            let row = |policy: &str| rows.iter().find(|r| r[column("policy")] == policy).unwrap();
            let baseline = row("baseline");
            let counts = [column("tp"), column("fp"), column("fn")].map(|c| baseline[c]);
            assert_eq!(counts, ["30", "30", "3"]);
            let genotyper = row("genotyper");
            let dropped: usize = genotyper[column("dropped")].parse().unwrap();
            let flipped: usize = genotyper[column("flipped")].parse().unwrap();
            assert!(dropped > 0, "{metrics}");

            run(&[
                "score",
                "--vcf",
                &path("calls.vcf.gz"),
                "--models",
                &model_path,
                "-o",
                &path("scored.vcf.gz"),
            ]);
            assert!(dir.path().join("scored.vcf.gz.tbi").is_file());
            let scored = read_gzip(&dir.path().join("scored.vcf.gz"));
            let records: Vec<Vec<&str>> = scored
                .lines()
                .filter(|line| !line.starts_with('#'))
                .map(|line| line.split('\t').collect())
                .collect();
            assert_eq!(records.len(), CALLS);
            assert!(records.iter().all(|r| r[7].contains("WINNOWR_P=")));
            // score drops and flips the calls evaluate counts for the same model and sample.
            let n_dropped = records.iter().filter(|r| r[6] == "WinnowrDrop").count();
            let n_flipped = records.iter().filter(|r| r[7].contains("WINNOWR_FLIP")).count();
            assert_eq!((n_dropped, n_flipped), (dropped, flipped));
        }
    }

    #[test]
    fn subset_options_parse() {
        let cli = parse(&["subset", "--table", "t", "--happy", "h", "-o", "o"]).unwrap();
        let Command::Subset(args) = cli.command else { panic!("not subset") };
        assert!(args.contigs.is_empty() && args.exclude_bed.is_none());
        let cli = parse(&[
            "subset",
            "--table",
            "t",
            "--happy",
            "h",
            "--contigs",
            "chr2",
            "chr4",
            "--exclude-bed",
            "x.bed",
            "-o",
            "o",
        ])
        .unwrap();
        let Command::Subset(args) = cli.command else { panic!("not subset") };
        assert_eq!(args.contigs, ["chr2", "chr4"]);
        assert_eq!(args.exclude_bed, Some(PathBuf::from("x.bed")));
    }

    #[test]
    fn extract_options_parse() {
        let cli = parse(&["extract", "--vcf", "in.vcf.gz", "-o", "t.parquet"]).unwrap();
        let Command::Extract(args) = cli.command else { panic!("not extract") };
        assert_eq!(args.contigs, AUTOSOMES.to_vec());
        assert!(args.regions.is_empty() && args.gvcf.is_none() && args.sample.is_none());

        let cli = parse(&[
            "extract",
            "--vcf",
            "in.vcf.gz",
            "--gvcf",
            "in.g.vcf.gz",
            "-o",
            "t.parquet",
            "--regions",
            "a=a.bed",
            "b=b.bed.gz",
            "--contigs",
            "chr20",
            "chr21",
            "--sample",
            "S",
        ])
        .unwrap();
        let Command::Extract(args) = cli.command else { panic!("not extract") };
        assert_eq!(args.regions.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(args.contigs, ["chr20", "chr21"]);
        assert_eq!(args.sample.as_deref(), Some("S"));
        assert_eq!(args.gvcf, Some(PathBuf::from("in.g.vcf.gz")));
    }

    #[test]
    fn score_options_parse() {
        let cli = parse(&[
            "score",
            "--vcf",
            "in.vcf.gz",
            "--models",
            "a.winnowr.json",
            "b.winnowr.json",
            "-o",
            "out.vcf.gz",
            "--mode",
            "veto",
            "--strlen",
            "s1.tsv",
            "s2.tsv",
        ])
        .unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        assert_eq!(args.models.len(), 2);
        assert_eq!(args.mode, Mode::Veto);
        assert_eq!(args.strlen.len(), 2);
        let cli = parse(&["score", "--vcf", "v", "--models", "m", "-o", "o"]).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        assert_eq!(args.mode, Mode::Genotyper);
    }

    /// The policy of a `score` command line with extra options, and `regions`.
    fn score_policy(extra: &[&str], regions: &[NamedBed]) -> Result<Policy> {
        let base = ["score", "--vcf", "v", "--models", "m", "-o", "o"];
        let cli = parse(&[&base[..], extra].concat()).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        args.policy.policy(Mode::Genotyper, &["chr1"], regions, true)
    }

    /// A one-interval BED of chr1 and a `--regions` entry naming it the confident regions.
    fn confident_regions(dir: &tempfile::TempDir) -> Vec<NamedBed> {
        let path = dir.path().join("confident.bed");
        std::fs::write(&path, "chr1\t100\t200\n").unwrap();
        vec![NamedBed { name: CONFIDENT_REGIONS.to_string(), path }]
    }

    #[test]
    fn the_default_policy_keeps_the_fallback_for_indels_and_an_outside_drop_threshold() {
        use winnowr::policy::{DEFAULT_KEEP_FALLBACK, DEFAULT_OUTSIDE_INDEL_T_DROP};
        // Without the confident regions: the indel keep fallback only.
        let policy = score_policy(&[], &[]).unwrap();
        let expected: Vec<(VariantType, Rule)> =
            DEFAULT_KEEP_FALLBACK.iter().map(|&vt| (vt, Rule::KeepFallback)).collect();
        assert_eq!(policy.rules.into_iter().collect::<Vec<_>>(), expected);
        assert!(policy.abstain.is_none());
        // With them: indels outside are dropped at the outside threshold, in place of the
        // model's; no action is withheld.
        let dir = tempfile::TempDir::new().unwrap();
        let policy = score_policy(&[], &confident_regions(&dir)).unwrap();
        let abstain = policy.abstain.unwrap();
        assert_eq!(abstain.indel_t_drop, Some(DEFAULT_OUTSIDE_INDEL_T_DROP));
        assert!(!abstain.flip && !abstain.regenotype && !abstain.promote);
        // Another name is not the confident regions.
        let other = vec![NamedBed { name: "segdups".into(), ..confident_regions(&dir).remove(0) }];
        assert!(score_policy(&[], &other).unwrap().abstain.is_none());
    }

    #[test]
    fn the_default_policy_can_be_turned_off() {
        let dir = tempfile::TempDir::new().unwrap();
        let regions = confident_regions(&dir);
        let policy =
            score_policy(&["--keep-fallback", "none", "--outside-indel-t-drop", "none"], &regions)
                .unwrap();
        assert!(policy.rules.is_empty() && policy.abstain.is_none());
        // --expected-f1 replaces the default keep fallback of its variant types.
        let policy = score_policy(&["--expected-f1", "indel"], &[]).unwrap();
        assert_eq!(policy.rule(VariantType::Indel), Rule::ExpectedF1);
        // Explicit options: SNPs too, another threshold, and withheld flips.
        let policy = score_policy(
            &[
                "--keep-fallback",
                "SNP",
                "indel",
                "--outside-indel-t-drop",
                "0.7",
                "--abstain-actions",
                "flip",
            ],
            &regions,
        )
        .unwrap();
        assert_eq!(policy.rule(VariantType::Snp), Rule::KeepFallback);
        let abstain = policy.abstain.unwrap();
        assert_eq!((abstain.indel_t_drop, abstain.flip), (Some(0.7), true));
        // Withholding actions needs the confident regions.
        let error = score_policy(&["--abstain-actions", "flip"], &[]).unwrap_err();
        assert!(error.to_string().contains(CONFIDENT_REGIONS), "{error}");
    }

    #[rstest]
    #[case::keep_fallback(&["--keep-fallback", "none", "snp"], "--keep-fallback none cannot be given with other values, got: none snp")]
    #[case::keep_fallback_any_case(&["--keep-fallback", "INDEL", "None"], "--keep-fallback none cannot be given with other values, got: indel none")]
    #[case::abstain_actions(&["--abstain-actions", "flip", "none"], "--abstain-actions none cannot be given with other values, got: flip none")]
    fn none_cannot_be_given_with_other_values(#[case] extra: &[&str], #[case] expected: &str) {
        let dir = tempfile::TempDir::new().unwrap();
        let error = score_policy(extra, &confident_regions(&dir)).unwrap_err().to_string();
        assert_eq!(error, expected);
    }

    #[test]
    fn abstain_actions_ignore_case() {
        let dir = tempfile::TempDir::new().unwrap();
        let policy =
            score_policy(&["--abstain-actions", "Flip", "PROMOTE"], &confident_regions(&dir))
                .unwrap();
        let abstain = policy.abstain.unwrap();
        assert!(abstain.flip && abstain.promote && !abstain.regenotype);
        // `none` alone, in any case, withholds nothing.
        let policy =
            score_policy(&["--abstain-actions", "NONE"], &confident_regions(&dir)).unwrap();
        let abstain = policy.abstain.unwrap();
        assert!(!abstain.flip && !abstain.promote && !abstain.regenotype);
    }

    #[test]
    fn the_default_policy_options_come_from_the_constants() {
        let cli = parse(&["score", "--vcf", "v", "--models", "m", "-o", "o"]).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        let expected: Vec<String> = DEFAULT_KEEP_FALLBACK.iter().map(ToString::to_string).collect();
        assert_eq!(args.policy.keep_fallback, expected);
        assert_eq!(
            args.policy.outside_indel_t_drop,
            OutsideTDrop(Some(DEFAULT_OUTSIDE_INDEL_T_DROP))
        );
        // The help shows them.
        let help =
            Cli::command().find_subcommand_mut("score").unwrap().render_long_help().to_string();
        assert!(help.contains("[default: indel]"), "{help}");
        assert!(help.contains("[default: 0.6]"), "{help}");
        assert_eq!(OutsideTDrop(None).to_string(), "none");
    }

    #[test]
    fn stratify_reports_within_named_regions() {
        let base = [
            "stratify",
            "--models",
            "m",
            "--truth-names",
            "v421",
            "v5q",
            "--tables",
            "a",
            "b",
            "--truth-beds",
            "a.bed",
            "b.bed",
            "--happy",
            "a.vcf",
            "b.vcf",
            "-o",
            "o",
        ];
        let extra = ["--report-region", "GRCh38_MHC-v2=mhc.bed", "--regions", "segdups=s.bed"];
        let cli = parse(&[&base[..], &extra].concat()).unwrap();
        let Command::Stratify(args) = cli.command else { panic!("not stratify") };
        let expected = NamedPath { name: "GRCh38_MHC-v2".into(), path: PathBuf::from("mhc.bed") };
        assert_eq!(args.report_region, [expected]);
        assert_eq!(args.regions[0].name, "segdups");
        // --region is not an abbreviation of either.
        let error = parse(&[&base[..], &["--region", "MHC=mhc.bed"]].concat()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn output_directories_are_checked_before_any_work() {
        let dir = tempfile::TempDir::new().unwrap();
        check_output_dir("--output", &dir.path().join("out.vcf.gz")).unwrap();
        check_output_dir("--output", Path::new("out.vcf.gz")).unwrap();
        let missing = dir.path().join("nodir");
        let error = check_output_dir("--calls", &missing.join("calls.tsv.gz")).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("the output directory {} (of --calls) does not exist", missing.display())
        );
        // The subcommands fail on it before reading their inputs (which do not exist here).
        let output = missing.join("out").display().to_string();
        let threads = NonZeroUsize::new(1).unwrap();
        for args in [
            vec!["pileup", "--vcf", "v", "--alignments", "a", "--reference", "r", "-o", &output],
            vec!["strlen", "--vcf", "v", "--alignments", "a", "--reference", "r", "-o", &output],
            vec!["extract", "--vcf", "v", "-o", &output],
            vec!["train", "--tables", "a", "b", "--variant-type", "snp", "-o", &output],
            vec!["evaluate", "--models", "m", "--table", "t", "-o", &output],
            vec!["subset", "--table", "t", "--happy", "h", "-o", &output],
            vec!["score", "--vcf", "v", "--models", "m", "-o", &output],
        ] {
            let cli = parse(&args).unwrap();
            let error = dispatch(&cli.command, threads).unwrap_err().to_string();
            assert!(error.starts_with("the output directory"), "{}: {error}", args[0]);
        }
    }

    /// The training parameters of a `train` command line with extra options.
    fn train_params(extra: &[&str]) -> Result<TrainingParams> {
        let base = ["train", "--tables", "a", "b", "--variant-type", "snp", "-o", "m"];
        let Command::Train(args) = parse(&[&base[..], extra].concat()).unwrap().command else {
            panic!("not train")
        };
        training_params(&args, NonZeroUsize::new(3).unwrap())
    }

    #[test]
    fn train_options_become_training_params() {
        let params = train_params(&[]).unwrap();
        // The command line refits by default (models trained before it could not).
        let expected =
            TrainingParams { threads: 3, refit_final: true, ..TrainingParams::default() };
        assert_eq!(params, expected);
        assert!(params.refit_final && params.cv_folds == CvFolds::Samples);
        let params = train_params(&[
            "--learning-rate",
            "0.05",
            "--max-leaves",
            "7",
            "--max-depth",
            "3",
            "--min-child-weight",
            "2",
            "--reg-lambda",
            "0.5",
            "--max-rounds",
            "9",
            "--early-stopping-rounds",
            "4",
            "--validation-fraction",
            "0.2",
            "--validation-block-size",
            "5000",
            "--seed",
            "11",
            "--unfixable-flips",
            "drop",
            "--cv-folds",
            "CHROMOSOMES",
            "--chromosome-folds",
            "3",
        ])
        .unwrap();
        let expected = TrainingParams {
            learning_rate: 0.05,
            max_leaves: 7,
            max_depth: 3,
            min_child_weight: 2.0,
            reg_lambda: 0.5,
            max_rounds: 9,
            early_stopping_rounds: 4,
            validation_fraction: 0.2,
            validation_block_size: 5000,
            seed: 11,
            threads: 3,
            unfixable_flips: UnfixableFlips::Drop,
            refit_final: true,
            cv_folds: CvFolds::Chromosomes { folds: 3 },
        };
        assert_eq!(params, expected);
        // Without early stopping there is nothing to refit.
        assert!(!train_params(&["--validation-fraction", "0"]).unwrap().refit_final);
        assert!(!train_params(&["--refit-final", "false"]).unwrap().refit_final);
        let error = |extra: &[&str]| train_params(extra).unwrap_err().to_string();
        assert_eq!(
            error(&["--validation-fraction", "1"]),
            "--validation-fraction must be at least 0 and below 1, got 1"
        );
        assert_eq!(
            error(&["--validation-fraction", "NaN"]),
            "--validation-fraction must be at least 0 and below 1, got NaN"
        );
        assert_eq!(
            error(&["--task", "promote", "--unfixable-flips", "exclude"]),
            "--unfixable-flips applies to genotype models, not --task promote"
        );
    }

    #[rstest]
    #[case::above_one(&["--outside-indel-t-drop", "1.5"], ErrorKind::ValueValidation)]
    #[case::not_a_number(&["--outside-indel-t-drop", "high"], ErrorKind::ValueValidation)]
    #[case::unknown_variant_type(&["--keep-fallback", "mnp"], ErrorKind::InvalidValue)]
    fn invalid_policy_options_are_rejected(#[case] extra: &[&str], #[case] kind: ErrorKind) {
        let base = ["score", "--vcf", "v", "--models", "m", "-o", "o"];
        assert_eq!(parse(&[&base[..], extra].concat()).unwrap_err().kind(), kind);
    }

    #[test]
    fn pileup_options_parse() {
        let base = ["pileup", "--vcf", "v", "--alignments", "a.cram", "--reference", "r.fa"];
        let cli = parse(&[&base[..], &["-o", "o.tsv.gz"]].concat()).unwrap();
        let Command::Pileup(args) = cli.command else { panic!("not pileup") };
        assert_eq!(args.regions, AUTOSOMES.to_vec());
        assert_eq!(args.settings, PileupOptions::default());
        let cli = parse(
            &[&base[..], &["-o", "o", "--settings", "mapq30,bq0", "--regions", "chr2"]].concat(),
        )
        .unwrap();
        let Command::Pileup(args) = cli.command else { panic!("not pileup") };
        assert_eq!((args.settings.high_mapq, args.settings.min_base_quality), (30, 0));
        assert_eq!(args.regions, ["chr2"]);
        let error = parse(&[&base[..], &["-o", "o", "--settings", "mapq1"]].concat()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
        assert!(error.to_string().contains("mapq1"), "{error}");
        // score takes pileup tables, or measures them from --alignments, not both.
        let score = ["score", "--vcf", "v", "-o", "o", "--models", "m"];
        let cli = parse(&[&score[..], &["--pileup", "p1.tsv", "p2.tsv"]].concat()).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        assert_eq!(args.pileup.len(), 2);
        let both = [&score[..], &["--pileup", "p.tsv", "--alignments", "a", "--reference", "r"]];
        assert_eq!(parse(&both.concat()).unwrap_err().kind(), ErrorKind::ArgumentConflict);
        let cli = parse(&["extract", "--vcf", "v", "-o", "o", "--pileup", "p.tsv"]).unwrap();
        let Command::Extract(args) = cli.command else { panic!("not extract") };
        assert_eq!(args.pileup, [PathBuf::from("p.tsv")]);
        let train = ["train", "--tables", "a", "b", "--variant-type", "snp", "-o", "m"];
        let cli = parse(&[&train[..], &["--pileup-features"]].concat()).unwrap();
        let Command::Train(args) = cli.command else { panic!("not train") };
        assert!(args.pileup_features);
    }

    #[test]
    fn strlen_rejects_a_repeated_output_before_measuring() {
        let args = ["strlen", "--vcf", "v", "--alignments", "a", "--reference", "r", "-o", "a.tsv"];
        for also in [&["--also", "legacy=a.tsv"][..], &["--also", "legacy=b.tsv", "mapq0=b.tsv"]] {
            let cli = parse(&[&args[..], also].concat()).unwrap();
            let Command::Strlen(args) = cli.command else { panic!("not strlen") };
            let message = strlen(&args).unwrap_err().to_string();
            let path = also.last().unwrap().split_once('=').unwrap().1;
            assert_eq!(message, format!("{path} is given as more than one strlen output"));
        }
        // A malformed --also fails as the command line is parsed.
        for bad in ["mapq0", "=b.tsv", "nonsense=b.tsv"] {
            let error = parse(&[&args[..], &["--also", bad]].concat()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ValueValidation, "{bad}");
        }
    }

    #[test]
    fn strlen_options_parse() {
        let cli = parse(&[
            "strlen",
            "--vcf",
            "v",
            "--alignments",
            "a.cram",
            "--reference",
            "r.fa",
            "-o",
            "o.tsv.gz",
        ])
        .unwrap();
        let Command::Strlen(args) = cli.command else { panic!("not strlen") };
        assert_eq!(args.regions, AUTOSOMES.to_vec());
        assert_eq!(args.reads, StrlenOptions { min_mapq: 20, flank: 5, ..StrlenOptions::LEGACY });
        let cli = parse(&[
            "strlen",
            "--vcf",
            "v",
            "--alignments",
            "a",
            "--reference",
            "r",
            "-o",
            "o",
            "--regions",
            "chr1:1-100",
            "chr2",
        ])
        .unwrap();
        let Command::Strlen(args) = cli.command else { panic!("not strlen") };
        assert_eq!(args.regions, ["chr1:1-100", "chr2"]);
    }
}
