//! The command-line interface: global options, the subcommands, and dispatch.

// The doc comments of the options are their `--help` text, where `<output>`-style placeholders
// are literal, not HTML.
#![allow(rustdoc::invalid_html_tags)]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use log::info;

use crate::common::{AUTOSOMES, PhaseTimer, Task, VariantType, repeated};
use crate::evidence::read_tables;
use crate::extraction::{TableInputs, TruthInputs, build_feature_table};
use crate::features::feature_names;
use crate::labels::{HappyPassOnly, UnfixableFlips};
use crate::model::{CvFolds, MODEL_SUFFIX, TrainingParams, WinnowrModel, check_reference};
use crate::model_evidence::{
    Measuring, check_supplied, check_table, measure_and_decide, models_options, models_using,
    read_for_models, shared_options,
};
use crate::pileup::{PileupOptions, measure_pileup, write_pileup_table};
use crate::pileup_features::PILEUP_FEATURES;
use crate::policy::{
    Abstention, DEFAULT_KEEP_FALLBACK, DEFAULT_OUTSIDE_INDEL_T_DROP, NOT_DIFFICULT_REGIONS, Policy,
    decide_calls,
};
use crate::read_pass::ReadInputs;
use crate::regions::{NamedBed, NamedPath, RegionSet, read_region_sets};
use crate::repeat_features::STRLEN_FEATURES;
use crate::rescore::{Mode, Rule};
use crate::samples::{PromoteSampleCalls, SampleCalls};
use crate::scoring::write_scored_vcf;
use crate::strlen::{StrlenOptions, measure_regions, write_strlen_table};
use crate::stutter::{
    LENGTH_CALL_FEATURES, StutterFit, StutterModel, fit_stutter_from_feature_tables,
};
use crate::subset::restrict_table;
use crate::table::{FeatureTable, FeatureTableInfo, TableSelection};
use crate::thresholds::{DEFAULT_GRID, DEFAULT_PROMOTE_FLIP_GRID, DEFAULT_PROMOTE_GRID};
use crate::training::{
    FoldTraining, TrainingSpec, fit_fold_stutter, group_samples, table_columns, train_model,
    train_promote_model,
};
use crate::vcf::VcfSummary;
use crate::version::{LONG_VERSION, VERSION};

/// The widest the help is wrapped to (narrower terminals wrap it narrower); the user guide's
/// command reference shows the help at this width.
pub(crate) const HELP_WIDTH: usize = 100;

/// The default number of threads, when neither `-t` nor `OMP_NUM_THREADS` gives one: enough to
/// overlap I/O with compute, while staying predictable on shared hosts and under schedulers.
pub(crate) const DEFAULT_THREADS: NonZeroUsize = NonZeroUsize::new(4).expect("4 is non-zero");

/// The environment variable that gives the number of threads when `-t` is not given, as it gives
/// XGBoost's own default.
pub(crate) const THREADS_VARIABLE: &str = "OMP_NUM_THREADS";

/// Machine-learned genotyping and filtering of germline small-variant calls.
#[derive(Debug, Parser)]
#[command(
    name = "winnowr",
    version = VERSION.as_str(),
    long_version = LONG_VERSION.as_str(),
    max_term_width = HELP_WIDTH
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
    /// per truth set and stratum; the `all` rows match `evaluate`'s, except with a genotype and a
    /// promote model in veto mode, which `evaluate` does not report), <output>.rates.tsv (action
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
    /// not-difficult regions (--regions notinalldifficultregions=PATH, GIAB's GRCh38 genome
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

    /// The hap.py output VCF for this sample (run with --engine vcfeval, preferably with
    /// --pass-only; see --happy-pass-only), which labels PASS calls; needed for training and
    /// evaluation tables.
    #[arg(long)]
    pub(crate) happy: Option<PathBuf>,

    /// Whether hap.py was run with --pass-only: yes, no, or auto to infer it from the labels. It
    /// decides whether a --happy label at a non-PASS call's position is that call's own or a PASS
    /// call's that hap.py represented there.
    #[arg(
        long,
        value_enum,
        ignore_case = true,
        value_name = "MODE",
        default_value_t = HappyPassOnly::Auto,
        requires = "happy"
    )]
    pub(crate) happy_pass_only: HappyPassOnly,

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

/// The options of `winnowr train`.
#[derive(Debug, Args)]
pub(crate) struct TrainArgs {
    /// Labelled feature tables (from `extract --happy`), one per training sample: at least two,
    /// of distinct samples.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) tables: Vec<PathBuf>,

    /// The variant type to train for (snp or indel).
    #[arg(long, value_enum, ignore_case = true)]
    pub(crate) variant_type: VariantType,

    /// The model file to write (`.winnowr.json` is appended if missing).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// genotype trains on PASS calls (drop/keep/flip; needs hap.py labels); promote trains on
    /// non-PASS calls (stay/promote/promote+flip; needs tables extracted with --truth-vcf).
    #[arg(long, value_enum, ignore_case = true, default_value_t = Task::Genotype)]
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
    #[arg(long, value_enum, ignore_case = true, default_value_t = UnfixableFlips::Flip)]
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
    #[arg(long, value_enum, ignore_case = true, default_value_t = CvFoldKind::Samples)]
    pub(crate) cv_folds: CvFoldKind,

    /// The number of chromosome groups (folds) with --cv-folds chromosomes.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(2..))]
    pub(crate) chromosome_folds: u32,

    /// Refit the final model on all its training calls, those held out for early stopping
    /// included, for the rounds its early stopping chose. Has no effect without early stopping
    /// (a zero --validation-fraction). Recorded in the model.
    #[arg(long, default_value_t = crate::model::DEFAULT_REFIT_FINAL, num_args = 0..=1,
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

    /// The maximum leaves per tree (0 = unlimited, bounded by --max-depth).
    #[arg(long, default_value_t = TrainingParams::default().max_leaves,
          value_parser = clap::value_parser!(u32).range(..=i64::from(i32::MAX)))]
    pub(crate) max_leaves: u32,

    /// The maximum tree depth (0 = unlimited, bounded by --max-leaves).
    #[arg(long, default_value_t = TrainingParams::default().max_depth,
          value_parser = clap::value_parser!(u32).range(..=i64::from(i32::MAX)))]
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
    /// with these. Only the not-difficult regions (notinalldifficultregions) are read; the models'
    /// region features come from the table.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,
}

/// The drop threshold of indel calls outside the not-difficult regions: a P(drop), or `none` for
/// the model's threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OutsideTDrop(Option<f64>);

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
        let value =
            parse_probability(text).map_err(|error| anyhow::anyhow!("{error} (or none)"))?;
        Ok(OutsideTDrop(Some(value)))
    }
}

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

/// How `train`'s cross-validation forms its folds (see [`CvFolds`]), as --cv-folds takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum CvFoldKind {
    /// Hold out each sample group in turn.
    #[value(help = None::<&str>)]
    Samples,
    /// Hold out groups of whole chromosomes from every sample at once.
    #[value(help = None::<&str>)]
    Chromosomes,
}

/// A value of --keep-fallback: a variant type, or none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum KeepFallback {
    /// SNP genotype models.
    #[value(help = None::<&str>)]
    Snp,
    /// Indel genotype models.
    #[value(help = None::<&str>)]
    Indel,
    /// No variant type (alone).
    #[value(help = None::<&str>)]
    None,
}

impl KeepFallback {
    /// The variant type, unless none.
    fn variant_type(self) -> Option<VariantType> {
        match self {
            KeepFallback::Snp => Some(VariantType::Snp),
            KeepFallback::Indel => Some(VariantType::Indel),
            KeepFallback::None => None,
        }
    }
}

impl From<VariantType> for KeepFallback {
    fn from(variant_type: VariantType) -> Self {
        match variant_type {
            VariantType::Snp => KeepFallback::Snp,
            VariantType::Indel => KeepFallback::Indel,
        }
    }
}

/// A value of --abstain-actions: an action withheld outside the not-difficult regions, or none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum AbstainAction {
    /// Flipping het <-> hom.
    #[value(help = None::<&str>)]
    Flip,
    /// Re-genotyping repeat-length calls.
    #[value(help = None::<&str>)]
    Regenotype,
    /// Promoting filtered calls.
    #[value(help = None::<&str>)]
    Promote,
    /// No action (alone).
    #[value(help = None::<&str>)]
    None,
}

/// Options that change how genotype and promote models act, beyond their thresholds.
///
/// By default ([`DEFAULT_KEEP_FALLBACK`] and [`DEFAULT_OUTSIDE_INDEL_T_DROP`]), indel genotype
/// models also drop calls their thresholds leave alone when P(keep) is below 0.5, and, given the
/// not-difficult regions (`--regions notinalldifficultregions=PATH`), drop indel calls outside them
/// at P(drop) >= 0.6 in place of the model's drop threshold (whether that is lower or higher); the
/// keep fallback still drops those calls when P(keep) is below 0.5.
#[derive(Debug, Args)]
pub(crate) struct PolicyArgs {
    /// Variant types whose genotype model also drops the calls its thresholds leave alone when
    /// P(keep) is below 0.5 (none, alone: no variant type).
    #[arg(long, num_args = 1.., value_enum, ignore_case = true,
          default_values_t = DEFAULT_KEEP_FALLBACK.map(KeepFallback::from))]
    pub(crate) keep_fallback: Vec<KeepFallback>,

    /// Experimental: variant types whose genotype model takes the action of greatest expected F1
    /// gain instead of using its thresholds (and in place of --keep-fallback): flip or
    /// re-genotype when its probability is above 0.5, else drop when P(keep) is below 0.5.
    #[arg(long, num_args = 1.., value_enum, ignore_case = true)]
    pub(crate) expected_f1: Vec<VariantType>,

    /// Experimental: actions withheld from calls outside the not-difficult regions (--regions
    /// notinalldifficultregions=PATH; calls whose REF span is not inside it are "outside"); none,
    /// alone: no action.
    #[arg(long, num_args = 1.., value_enum, ignore_case = true,
          default_values_t = [AbstainAction::None])]
    pub(crate) abstain_actions: Vec<AbstainAction>,

    /// The drop threshold of indel calls outside the not-difficult regions: they are dropped when
    /// P(drop) is at least this (in veto mode, P(drop) + P(flip) (+ P(re-genotype)), as the
    /// model's veto threshold is compared with), in place of the model's threshold, even when that
    /// is higher (none: the model's threshold). The keep fallback (--keep-fallback) still drops
    /// them when P(keep) is below 0.5. With --expected-f1 it applies too, before a flip or
    /// re-genotype.
    #[arg(long, default_value_t = OutsideTDrop(Some(DEFAULT_OUTSIDE_INDEL_T_DROP)),
          value_name = "P|none")]
    pub(crate) outside_indel_t_drop: OutsideTDrop,
}

impl PolicyArgs {
    /// The policy in a mode, checking the not-difficult regions name some of `contigs`.
    ///
    /// The not-difficult regions are the `regions` named [`NOT_DIFFICULT_REGIONS`]. Without them,
    /// the outside indel drop threshold is not applied (with a warning when `indel_genotype`, i.e.
    /// an indel genotype model will score). They are read only if an action is withheld outside
    /// them or an indel genotype model will drop calls outside them.
    ///
    /// # Errors
    ///
    /// Returns an error if `none` is given with other values of --keep-fallback or
    /// --abstain-actions, actions are withheld without the not-difficult regions, or they cannot be
    /// read or name none of `contigs`.
    fn policy<S: AsRef<str>>(
        &self,
        mode: Mode,
        contigs: &[S],
        regions: &[NamedBed],
        indel_genotype: bool,
    ) -> Result<Policy> {
        let keep_fallback =
            without_none("--keep-fallback", &self.keep_fallback, KeepFallback::None)?;
        let abstain_actions =
            without_none("--abstain-actions", &self.abstain_actions, AbstainAction::None)?;
        let mut policy = Policy::new(mode);
        for vt in keep_fallback.iter().filter_map(|value| value.variant_type()) {
            policy.rules.insert(vt, Rule::KeepFallback);
        }
        for &vt in &self.expected_f1 {
            policy.rules.insert(vt, Rule::ExpectedF1);
        }
        let has = |action: AbstainAction| abstain_actions.contains(&action);
        let (flip, regenotype, promote) =
            (has(AbstainAction::Flip), has(AbstainAction::Regenotype), has(AbstainAction::Promote));
        let indel_t_drop = self.outside_indel_t_drop.0;
        let path =
            regions.iter().find(|r| r.name == NOT_DIFFICULT_REGIONS).map(|r| r.path.as_path());
        // Read the not-difficult regions only when something uses them, so an unused BED (e.g. in a
        // run without an indel genotype model) cannot fail the run.
        let outside_t_drop = indel_t_drop.filter(|_| indel_genotype);
        match path {
            Some(path) if flip || regenotype || promote || outside_t_drop.is_some() => {
                let regions = RegionSet::from_bed(NOT_DIFFICULT_REGIONS, path)?;
                regions.check_contigs(contigs)?;
                info!("Not-difficult regions (acting fully inside): {}", path.display());
                policy.abstain =
                    Some(Abstention { regions, flip, regenotype, promote, indel_t_drop });
            }
            Some(_) => {}
            None if flip || regenotype || promote => bail!(
                "--abstain-actions needs the not-difficult regions: --regions \
                 {NOT_DIFFICULT_REGIONS}=PATH"
            ),
            None => {
                if let Some(t_drop) = outside_t_drop {
                    log::warn!(
                        "--regions has no {NOT_DIFFICULT_REGIONS}=PATH, so indel calls are \
                         dropped at the model's threshold everywhere; pass it (GIAB's GRCh38 \
                         genome stratification {NOT_DIFFICULT_REGIONS}.bed.gz) to drop indel \
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

/// The values of a multi-valued option without `none`, which means no value.
///
/// # Errors
///
/// Returns an error if `none` is given with other values.
fn without_none<T: clap::ValueEnum + Copy + PartialEq>(
    option: &str,
    values: &[T],
    none: T,
) -> Result<Vec<T>> {
    if values.len() > 1 && values.contains(&none) {
        let names: Vec<String> = values
            .iter()
            .filter_map(|value| value.to_possible_value())
            .map(|value| value.get_name().to_string())
            .collect();
        bail!("{option} none cannot be given with other values, got: {}", names.join(" "));
    }
    Ok(values.iter().copied().filter(|&value| value != none).collect())
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
    /// with these. Only the not-difficult regions (notinalldifficultregions) are read; the models'
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

    /// Region BEDs as NAME=PATH, with the names the models were trained with, plus the
    /// not-difficult regions as notinalldifficultregions=PATH (GIAB's GRCh38 genome stratification;
    /// outside them, indel calls are dropped at --outside-indel-t-drop in place of the model's
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
    /// and unit (at least one repeat unit of flank). `legacy` (mapq0,flank2) measures reads of any
    /// MAPQ with two flank bases, the settings of tables and models that record none. The settings
    /// are recorded in the table; a model scores only tables measured as its training tables were.
    #[arg(long, default_value_t = StrlenOptions::default())]
    pub(crate) reads: StrlenOptions,
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
    check_output_dir("--output", &args.output)?;
    let sites = measure_regions(&inputs, &args.regions, &args.reads)?;
    let n_sites = sites.len();
    let n = write_strlen_table(&args.output, &sites, args.reads)?;
    info!("Wrote {n} sites measured with {} to {}", args.reads, args.output.display());
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
        happy_pass_only: args.happy_pass_only,
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
            "some --tables do not record their reference contigs; the model will not record them \
             either, so score cannot check the reference"
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
/// Returns an error if --unfixable-flips is given for a promote model, --validation-fraction is
/// not in `[0, 1)`, --learning-rate is not positive and finite, --min-child-weight or
/// --reg-lambda is negative or not finite.
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
        cv_folds: match args.cv_folds {
            CvFoldKind::Samples => CvFolds::Samples,
            CvFoldKind::Chromosomes => CvFolds::Chromosomes { folds: args.chromosome_folds },
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
    // XGBoost accepts NaN for these (its range checks are comparisons), which trains a model of
    // NaN probabilities whose parameters cannot be read back.
    if !(params.learning_rate.is_finite() && params.learning_rate > 0.0) {
        bail!("--learning-rate must be positive and finite, got {}", params.learning_rate);
    }
    for (option, value) in
        [("--min-child-weight", params.min_child_weight), ("--reg-lambda", params.reg_lambda)]
    {
        if !(value.is_finite() && value >= 0.0) {
            bail!("{option} must be at least 0 and finite, got {value}");
        }
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
        strlen_settings: strlen_options,
        pileup_settings: pileup_options,
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
    use crate::evaluation::{
        EvaluationPolicy, baseline_metric, evaluate_model, evaluate_promote_model,
        write_policy_metrics,
    };
    check_output_dir("--output", &args.output)?;
    let models = load_models(&args.models, threads)?;
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
    check_labelled_table(&args.table, info, &models, &args.models)?;
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
        let promoted = if m.t_promote < crate::common::NEVER {
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

/// Loads model files.
///
/// # Errors
///
/// Returns an error if a model cannot be loaded.
fn load_models(paths: &[PathBuf], threads: NonZeroUsize) -> Result<Vec<WinnowrModel>> {
    paths.iter().map(|path| WinnowrModel::load(path, threads)).collect()
}

/// Checks that a labelled feature table (of `evaluate` or `stratify`) suits the models: that it is
/// on their reference and has the evidence they need, measured as they need it. Warns if its
/// sample is a training sample of a model, as its metrics are then in-sample.
///
/// # Errors
///
/// Returns an error if the table is not on a model's reference or lacks evidence a model needs.
fn check_labelled_table(
    table: &Path,
    info: &FeatureTableInfo,
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Result<()> {
    check_table_reference(table, info, models, paths)?;
    check_table::<StrlenOptions>(table, info, models, paths)?;
    check_table::<PileupOptions>(table, info, models, paths)?;
    let trained_on: Vec<String> = paths
        .iter()
        .zip(models)
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
    Ok(())
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
            "{} does not record its reference contigs, so it cannot be checked against the \
             models' reference",
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
/// models' region features from the table, and only the not-difficult regions from --regions.
fn warn_policy_only_regions(regions: &[NamedBed]) {
    let ignored: Vec<&str> =
        regions.iter().map(|r| r.name.as_str()).filter(|n| *n != NOT_DIFFICULT_REGIONS).collect();
    if !ignored.is_empty() {
        log::warn!(
            "--regions {} are ignored: region features come from the feature table",
            ignored.join(", ")
        );
    }
}

/// Runs `winnowr stratify`.
fn stratify(args: &StratifyArgs, threads: NonZeroUsize) -> Result<()> {
    use crate::regions::RegionSet;
    use crate::regions::read_stratification_manifest;
    use crate::strata::{
        CALL_KEY_COLUMNS, CallOutputs, TRUTH_LABEL_COLUMNS, TruthSets, check_region_names, stratify,
    };
    check_output_dir("--output", &args.output)?;
    if let Some(calls) = &args.calls {
        check_output_dir("--calls", calls)?;
    }
    if let Some(dir) = args.labelled_calls.as_deref().filter(|dir| !dir.is_dir()) {
        bail!("the --labelled-calls directory {} does not exist", dir.display());
    }
    let models = load_models(&args.models, threads)?;
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
    check_labelled_table(&args.tables[0], info, &models, &args.models)?;
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
    if !args.regions.iter().any(|r| r.name == NOT_DIFFICULT_REGIONS)
        && named.iter().any(|named| named.name == NOT_DIFFICULT_REGIONS)
    {
        log::warn!(
            "{NOT_DIFFICULT_REGIONS} is reported within but not given to the policy: pass \
             --regions {NOT_DIFFICULT_REGIONS}=PATH to decide calls as score does with it"
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
    use crate::regions::RegionSet;
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
    let models = load_models(&args.models, threads)?;
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
            *name != NOT_DIFFICULT_REGIONS
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
            .map_err(openmp_error)?;
    }
    dispatch(&cli.command, threads)
}

/// The error for an OpenMP environment that training cannot run in: the problem, with advice
/// that names `-t` (see [`winnowr_xgboost::OpenMpProblem`]).
fn openmp_error(error: winnowr_xgboost::Error) -> anyhow::Error {
    use winnowr_xgboost::OpenMpProblem;
    let Some(problem) = error.openmp_problem() else { return error.into() };
    let explanation = problem.explanation();
    match problem {
        OpenMpProblem::Dynamic { .. } => anyhow::anyhow!(problem.to_string()),
        OpenMpProblem::HiddenThreadLimit { variable, limit, .. } => {
            anyhow::anyhow!("{explanation}; use -t {limit} or less, or raise or unset {variable}")
        }
        OpenMpProblem::ThreadLimit { limit, threads } => anyhow::anyhow!(
            "{explanation}, and a model trained so is not the one -t {threads} names; use -t \
             {limit} or less, or raise or unset OMP_THREAD_LIMIT"
        ),
    }
}

/// A command line as one string, for the log, that a POSIX shell reads back as the same arguments:
/// the arguments separated by spaces, each single-quoted unless it is non-empty and holds only
/// characters a shell takes literally (ASCII letters, digits and `-_./=:,@%+`).
fn command_line<I: IntoIterator<Item = std::ffi::OsString>>(args: I) -> String {
    let quote = |arg: std::ffi::OsString| {
        let arg = arg.to_string_lossy().into_owned();
        let plain = !arg.is_empty()
            && arg.chars().all(|c| c.is_ascii_alphanumeric() || "-_./=:,@%+".contains(c));
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
    #[case::settings(&["winnowr", "--regions", "a=x/b.bed.gz", "chr1:1-2,000"], "winnowr --regions a=x/b.bed.gz chr1:1-2,000")]
    #[case::metacharacters(
        &["winnowr", "a;b", "$HOME/x", "*.bed", "a&b", "(x)", "a|b", "~/x", "a>b", "`x`", "é"],
        "winnowr 'a;b' '$HOME/x' '*.bed' 'a&b' '(x)' 'a|b' '~/x' 'a>b' '`x`' 'é'"
    )]
    fn command_lines_are_logged_quoted(#[case] args: &[&str], #[case] expected: &str) {
        assert_eq!(command_line(args.iter().map(std::ffi::OsString::from)), expected);
    }

    /// Command lines as pipelines run them: `--output`, `--mode` in upper case, sidecar paths,
    /// NAME=PATH regions and one region per `strlen`.
    #[test]
    fn pipeline_command_lines_parse() {
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
    #[case::train_bad_unfixable_flips(&["train", "--tables", "a", "--variant-type", "snp", "-o", "m", "--unfixable-flips", "keep"], ErrorKind::InvalidValue)]
    #[case::train_bad_variant_type(&["train", "--tables", "a", "--variant-type", "mnp", "-o", "m"], ErrorKind::InvalidValue)]
    #[case::score_without_models(&["score", "--vcf", "v", "-o", "o"], ErrorKind::MissingRequiredArgument)]
    #[case::score_bad_mode(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--mode", "x"], ErrorKind::ValueValidation)]
    #[case::score_alignments_without_reference(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--alignments", "a.cram"], ErrorKind::MissingRequiredArgument)]
    #[case::score_reference_without_alignments(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--reference", "r.fa"], ErrorKind::MissingRequiredArgument)]
    #[case::score_alignments_and_strlen(&["score", "--vcf", "v", "-o", "o", "--models", "m", "--alignments", "a.cram", "--reference", "r.fa", "--strlen", "s.tsv"], ErrorKind::ArgumentConflict)]
    #[case::missing_subcommand(&[], ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand)]
    #[case::unknown_subcommand(&["rescore"], ErrorKind::InvalidSubcommand)]
    #[case::extract_without_vcf(&["extract", "-o", "t.parquet"], ErrorKind::MissingRequiredArgument)]
    #[case::extract_bad_region(&["extract", "--vcf", "v", "-o", "t", "--regions", "x"], ErrorKind::ValueValidation)]
    #[case::extract_bad_happy_pass_only(&["extract", "--vcf", "v", "-o", "t", "--happy", "h", "--happy-pass-only", "maybe"], ErrorKind::InvalidValue)]
    #[case::extract_happy_pass_only_without_happy(&["extract", "--vcf", "v", "-o", "t", "--happy-pass-only", "no"], ErrorKind::MissingRequiredArgument)]
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

    #[rstest]
    #[case::default(&[], HappyPassOnly::Auto)]
    #[case::auto(&["--happy-pass-only", "auto"], HappyPassOnly::Auto)]
    #[case::yes(&["--happy-pass-only", "yes"], HappyPassOnly::Yes)]
    #[case::no(&["--happy-pass-only=no"], HappyPassOnly::No)]
    #[case::ignoring_case(&["--happy-pass-only", "YES"], HappyPassOnly::Yes)]
    fn extract_happy_pass_only_parses(#[case] option: &[&str], #[case] expected: HappyPassOnly) {
        let extract = ["extract", "--vcf", "v", "-o", "t", "--happy", "h"];
        let Command::Extract(args) = parse(&[&extract[..], option].concat()).unwrap().command
        else {
            panic!("not extract")
        };
        assert_eq!(args.happy_pass_only, expected);
        // The default does not need --happy.
        let Command::Extract(args) = parse(&extract[..5]).unwrap().command else {
            panic!("not extract")
        };
        assert_eq!(args.happy_pass_only, HappyPassOnly::Auto);
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
        assert_eq!((args.cv_folds, args.chromosome_folds), (CvFoldKind::Samples, 5));
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
        assert_eq!(args.cv_folds, CvFoldKind::Chromosomes);
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

        /// Writes the calls (`calls.vcf.gz`) and their hap.py output (`happy.vcf`) to `dir`.
        fn write_calls(dir: &Path) {
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
            write_bgzf_indexed(&dir.join("calls.vcf.gz"), &vcf);
            std::fs::write(dir.join("happy.vcf"), happy).unwrap();
        }

        #[test]
        fn extract_train_evaluate_and_score() {
            let dir = tempfile::TempDir::new().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            write_calls(dir.path());
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
                    assert!([0.3, 0.5, 0.7, crate::common::NEVER].contains(&t), "{t}");
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

        /// A model trained on pileup features records the tables' pileup settings.
        #[test]
        fn train_records_the_pileup_settings() {
            let dir = tempfile::TempDir::new().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            write_calls(dir.path());
            let mut pileup = format!(
                "##pileup_settings=mapq20,bq9,flank5\n{}\n",
                crate::pileup::PILEUP_COLUMNS.join("\t")
            );
            for i in 0..CALLS {
                let (reference, alt) = if is_true(i) { (10, 10) } else { (18, 2) };
                pileup.push_str(&format!(
                    "chr1\t{}\tA\tC\t{}\t0\t0\t{reference}\t0\t0\t{alt}\t0\t0\t0\n",
                    1000 + 200 * i,
                    reference + alt
                ));
            }
            std::fs::write(dir.path().join("pileup.tsv"), pileup).unwrap();
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
                    "--pileup",
                    &path("pileup.tsv"),
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
                "snp",
                "--pileup-features",
                "--max-rounds",
                "3",
                "-o",
                &path("snp"),
            ]);
            let model = WinnowrModel::load(path("snp.winnowr.json"), NonZeroUsize::new(1).unwrap())
                .unwrap();
            let expected: PileupOptions = "mapq20,bq9,flank5".parse().unwrap();
            assert_eq!(model.info.pileup_settings, Some(expected));
        }

        /// `winnowr strlen` measures with the read settings of --reads and records them.
        #[test]
        fn strlen_measures_with_its_read_settings() {
            use crate::strlen::{StrlenTables, measure_regions, sites_to_rows};
            use crate::testing::TestRead;
            let dir = tempfile::TempDir::new().unwrap();
            // A 10-base A tract at 80 between non-repeating flanks, and a deletion of one A.
            let left = "CGTC".repeat(20);
            let reference = format!("{left}{}{}", "A".repeat(10), "GCTG".repeat(20));
            let fasta = crate::testing::write_fasta(dir.path(), &[("chr1", &reference)]);
            let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
            let record = format!("chr1\t{}\t.\tCA\tC\t50\tPASS\t.\tGT\t0/1", left.len());
            let vcf =
                crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &[&record]);
            // Two low-MAPQ reads and the two mates of a pair: the settings measure all of them
            // but count the pair once, so the table differs from one of the defaults.
            let read = |name, pos, mapq| TestRead {
                name,
                pos,
                cigar: "70M",
                flag: 0,
                mapq,
                qual: 30,
                bases: None,
            };
            let reads = [
                read("pair", 45, 60),
                read("ref", 50, 60),
                read("low1", 50, 5),
                read("low2", 50, 5),
                read("pair", 55, 60),
            ];
            let bam =
                crate::testing::write_bam(dir.path(), "reads.bam", reference.len() as u64, &reads);
            let table = dir.path().join("strlen.tsv.gz");
            let (vcf, bam, fasta) =
                (vcf.display().to_string(), bam.display().to_string(), fasta.display().to_string());
            run(&[
                "strlen",
                "--vcf",
                &vcf,
                "--alignments",
                &bam,
                "--reference",
                &fasta,
                "--regions",
                "chr1",
                "--reads",
                "mapq0,mates",
                "-o",
                &table.display().to_string(),
            ]);
            let options: StrlenOptions = "mapq0,mates".parse().unwrap();
            let inputs = ReadInputs {
                vcf: Path::new(&vcf),
                alignments: Path::new(&bam),
                reference: Path::new(&fasta),
            };
            let measured =
                |options| sites_to_rows(&measure_regions(&inputs, &["chr1"], options).unwrap());
            let rows = measured(&options);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].n_spanning, 4);
            assert_ne!(rows, measured(&StrlenOptions::default()));
            assert_eq!(
                read_tables::<StrlenOptions>(&[&table]).unwrap(),
                StrlenTables { options, rows }
            );
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

    /// A one-interval BED of chr1 and a `--regions` entry naming it the not-difficult regions.
    fn not_difficult_regions(dir: &tempfile::TempDir) -> Vec<NamedBed> {
        let path = dir.path().join("not-difficult.bed");
        std::fs::write(&path, "chr1\t100\t200\n").unwrap();
        vec![NamedBed { name: NOT_DIFFICULT_REGIONS.to_string(), path }]
    }

    #[test]
    fn the_default_policy_keeps_the_fallback_for_indels_and_an_outside_drop_threshold() {
        use crate::policy::{DEFAULT_KEEP_FALLBACK, DEFAULT_OUTSIDE_INDEL_T_DROP};
        // Without the not-difficult regions: the indel keep fallback only.
        let policy = score_policy(&[], &[]).unwrap();
        let expected: Vec<(VariantType, Rule)> =
            DEFAULT_KEEP_FALLBACK.iter().map(|&vt| (vt, Rule::KeepFallback)).collect();
        assert_eq!(policy.rules.into_iter().collect::<Vec<_>>(), expected);
        assert!(policy.abstain.is_none());
        // With them: indels outside are dropped at the outside threshold, in place of the
        // model's; no action is withheld.
        let dir = tempfile::TempDir::new().unwrap();
        let policy = score_policy(&[], &not_difficult_regions(&dir)).unwrap();
        let abstain = policy.abstain.unwrap();
        assert_eq!(abstain.indel_t_drop, Some(DEFAULT_OUTSIDE_INDEL_T_DROP));
        assert!(!abstain.flip && !abstain.regenotype && !abstain.promote);
        // A BED under another name is not taken as the not-difficult regions.
        let other =
            vec![NamedBed { name: "segdups".into(), ..not_difficult_regions(&dir).remove(0) }];
        assert!(score_policy(&[], &other).unwrap().abstain.is_none());
    }

    #[test]
    fn withheld_actions_read_the_not_difficult_regions_without_an_indel_genotype_model() {
        let dir = tempfile::TempDir::new().unwrap();
        let cli = parse(&[
            "score",
            "--vcf",
            "v",
            "--models",
            "m",
            "-o",
            "o",
            "--abstain-actions",
            "flip",
        ])
        .unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        let regions = not_difficult_regions(&dir);
        let policy = args.policy.policy(Mode::Genotyper, &["chr1"], &regions, false).unwrap();
        let abstain = policy.abstain.unwrap();
        assert!(abstain.flip && !abstain.regenotype && !abstain.promote);
        assert_eq!(abstain.indel_t_drop, args.policy.outside_indel_t_drop.0);
        // A missing BED is still an error.
        let missing = vec![NamedBed {
            name: NOT_DIFFICULT_REGIONS.to_string(),
            path: dir.path().join("missing.bed"),
        }];
        let error = args.policy.policy(Mode::Genotyper, &["chr1"], &missing, false).unwrap_err();
        assert!(format!("{error:#}").contains("missing.bed"), "{error:#}");
    }

    #[test]
    fn the_not_difficult_regions_are_not_read_without_an_indel_genotype_model() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("missing.bed");
        let regions = vec![NamedBed { name: NOT_DIFFICULT_REGIONS.to_string(), path }];
        let cli = parse(&["score", "--vcf", "v", "--models", "m", "-o", "o"]).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        let policy = args.policy.policy(Mode::Genotyper, &["chr1"], &regions, false).unwrap();
        assert!(policy.abstain.is_none());
        // With an indel genotype model they are read.
        let error = args.policy.policy(Mode::Genotyper, &["chr1"], &regions, true).unwrap_err();
        assert!(format!("{error:#}").contains("missing.bed"), "{error:#}");
    }

    #[test]
    fn the_default_policy_can_be_turned_off() {
        let dir = tempfile::TempDir::new().unwrap();
        let regions = not_difficult_regions(&dir);
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
        // Withholding actions needs the not-difficult regions.
        let error = score_policy(&["--abstain-actions", "flip"], &[]).unwrap_err();
        assert!(error.to_string().contains(NOT_DIFFICULT_REGIONS), "{error}");
    }

    #[rstest]
    #[case::keep_fallback(&["--keep-fallback", "none", "snp"], "--keep-fallback none cannot be given with other values, got: none snp")]
    #[case::keep_fallback_any_case(&["--keep-fallback", "INDEL", "None"], "--keep-fallback none cannot be given with other values, got: indel none")]
    #[case::abstain_actions(&["--abstain-actions", "flip", "none"], "--abstain-actions none cannot be given with other values, got: flip none")]
    fn none_cannot_be_given_with_other_values(#[case] extra: &[&str], #[case] expected: &str) {
        let dir = tempfile::TempDir::new().unwrap();
        let error = score_policy(extra, &not_difficult_regions(&dir)).unwrap_err().to_string();
        assert_eq!(error, expected);
    }

    #[test]
    fn abstain_actions_ignore_case() {
        let dir = tempfile::TempDir::new().unwrap();
        let policy =
            score_policy(&["--abstain-actions", "Flip", "PROMOTE"], &not_difficult_regions(&dir))
                .unwrap();
        let abstain = policy.abstain.unwrap();
        assert!(abstain.flip && abstain.promote && !abstain.regenotype);
        // `none` alone, in any case, withholds nothing.
        let policy =
            score_policy(&["--abstain-actions", "NONE"], &not_difficult_regions(&dir)).unwrap();
        let abstain = policy.abstain.unwrap();
        assert!(!abstain.flip && !abstain.promote && !abstain.regenotype);
    }

    #[test]
    fn the_default_policy_options_come_from_the_constants() {
        let cli = parse(&["score", "--vcf", "v", "--models", "m", "-o", "o"]).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        let expected: Vec<KeepFallback> =
            DEFAULT_KEEP_FALLBACK.iter().map(|&vt| KeepFallback::from(vt)).collect();
        assert_eq!(args.policy.keep_fallback, expected);
        assert_eq!(
            args.policy.outside_indel_t_drop,
            OutsideTDrop(Some(DEFAULT_OUTSIDE_INDEL_T_DROP))
        );
        // The help shows them, as each argument's default.
        let command = Cli::command();
        let score = command.find_subcommand("score").unwrap();
        let shown = |id: &str| -> Vec<String> {
            let arg = score.get_arguments().find(|arg| arg.get_id() == id).unwrap();
            assert!(!arg.is_hide_default_value_set(), "{id}");
            arg.get_default_values().iter().map(|v| v.to_string_lossy().into_owned()).collect()
        };
        let names: Vec<String> = DEFAULT_KEEP_FALLBACK.iter().map(ToString::to_string).collect();
        assert_eq!(shown("keep_fallback"), names);
        assert_eq!(shown("outside_indel_t_drop"), [DEFAULT_OUTSIDE_INDEL_T_DROP.to_string()]);
        assert_eq!(OutsideTDrop(None).to_string(), "none");
    }

    // The messages winnowr gave before the OpenMP check moved into winnowr-xgboost, unchanged.
    #[rstest]
    #[case::dynamic(
        winnowr_xgboost::OpenMpProblem::Dynamic { value: "1".to_string() },
        "OMP_DYNAMIC=1 lets OpenMP give XGBoost fewer threads than it asks for, which XGBoost's \
         training does not handle (it can lose rows or crash); unset OMP_DYNAMIC or set it to false"
    )]
    #[case::hidden(
        winnowr_xgboost::OpenMpProblem::HiddenThreadLimit {
            variable: "KMP_DEVICE_THREAD_LIMIT",
            limit: 1,
            threads: NonZeroUsize::new(2).unwrap(),
        },
        "KMP_DEVICE_THREAD_LIMIT=1 is below the 2 threads requested, so OpenMP would give XGBoost \
         fewer threads than it asks for, which XGBoost's training does not handle (it can lose rows \
         or crash); use -t 1 or less, or raise or unset KMP_DEVICE_THREAD_LIMIT"
    )]
    #[case::thread_limit(
        winnowr_xgboost::OpenMpProblem::ThreadLimit { limit: 1, threads: NonZeroUsize::new(2).unwrap() },
        "OMP_THREAD_LIMIT=1 is below the 2 threads requested, so XGBoost would run with 1 thread, \
         and a model trained so is not the one -t 2 names; use -t 1 or less, or raise or unset \
         OMP_THREAD_LIMIT"
    )]
    fn openmp_errors_name_the_thread_option(
        #[case] problem: winnowr_xgboost::OpenMpProblem,
        #[case] expected: &str,
    ) {
        assert_eq!(openmp_error(problem.into()).to_string(), expected);
    }

    #[test]
    fn other_xgboost_errors_pass_through_openmp_error() {
        let error = winnowr_xgboost::Booster::from_json(b"[]", NonZeroUsize::new(1).unwrap())
            .expect_err("an invalid model");
        let message = error.to_string();
        assert_eq!(openmp_error(error).to_string(), message);
    }

    #[rstest]
    #[case::none("None", Ok(OutsideTDrop(None)))]
    #[case::probability("0.25", Ok(OutsideTDrop(Some(0.25))))]
    #[case::not_a_number("x", Err("not a number: x (or none)"))]
    #[case::above_one("1.5", Err("must be a probability between 0 and 1, got 1.5 (or none)"))]
    #[case::nan("NaN", Err("must be a probability between 0 and 1, got NaN (or none)"))]
    fn parses_the_outside_indel_t_drop(
        #[case] text: &str,
        #[case] expected: std::result::Result<OutsideTDrop, &str>,
    ) {
        let parsed = text.parse::<OutsideTDrop>().map_err(|error| error.to_string());
        assert_eq!(parsed, expected.map_err(str::to_string));
    }

    /// The option enums name their values as the types they stand for do.
    #[test]
    fn option_values_are_named_as_their_types() {
        use clap::ValueEnum;
        let names = |values: &[_]| -> Vec<String> {
            values
                .iter()
                .map(|value: &KeepFallback| {
                    value.to_possible_value().unwrap().get_name().to_string()
                })
                .collect()
        };
        let mut expected: Vec<String> = VariantType::ALL.iter().map(ToString::to_string).collect();
        expected.push("none".to_string());
        assert_eq!(names(KeepFallback::value_variants()), expected);
        for vt in VariantType::ALL {
            assert_eq!(KeepFallback::from(*vt).variant_type(), Some(*vt));
        }
        let samples = CvFoldKind::Samples.to_possible_value().unwrap();
        assert_eq!(
            serde_json::to_string(&CvFolds::Samples).unwrap(),
            format!("\"{}\"", samples.get_name())
        );
        let chromosomes = CvFoldKind::Chromosomes.to_possible_value().unwrap();
        assert_eq!(
            serde_json::to_string(&CvFolds::Chromosomes { folds: 5 }).unwrap(),
            format!("{{\"{}\":{{\"folds\":5}}}}", chromosomes.get_name())
        );
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
    #[case::nan_learning_rate(&["--learning-rate", "nan"], "--learning-rate must be positive and finite, got NaN")]
    #[case::infinite_learning_rate(&["--learning-rate", "inf"], "--learning-rate must be positive and finite, got inf")]
    #[case::zero_learning_rate(&["--learning-rate", "0"], "--learning-rate must be positive and finite, got 0")]
    #[case::infinite_min_child_weight(&["--min-child-weight", "inf"], "--min-child-weight must be at least 0 and finite, got inf")]
    #[case::negative_reg_lambda(&["--reg-lambda=-5"], "--reg-lambda must be at least 0 and finite, got -5")]
    #[case::nan_reg_lambda(&["--reg-lambda", "NaN"], "--reg-lambda must be at least 0 and finite, got NaN")]
    fn invalid_xgboost_options_are_rejected(#[case] extra: &[&str], #[case] expected: &str) {
        assert_eq!(train_params(extra).unwrap_err().to_string(), expected);
    }

    #[test]
    fn boundary_xgboost_options_are_accepted() {
        let params = train_params(&["--min-child-weight", "0", "--reg-lambda", "0"]).unwrap();
        assert_eq!((params.min_child_weight, params.reg_lambda), (0.0, 0.0));
        // XGBoost's CPU trainer takes 0 for no limit on leaves and depth, or both.
        for (leaves, depth) in [("0", "30"), ("0", "31"), ("7", "0"), ("0", "0")] {
            let params = train_params(&["--max-leaves", leaves, "--max-depth", depth]).unwrap();
            assert_eq!(params.max_leaves.to_string(), leaves);
            assert_eq!(params.max_depth.to_string(), depth);
        }
        // Beyond XGBoost's 32-bit parameters, the command line fails.
        let base = ["train", "--tables", "a", "b", "--variant-type", "snp", "-o", "m"];
        for option in ["--max-leaves", "--max-depth"] {
            let error = parse(&[&base[..], &[option, "2147483648"]].concat()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ValueValidation, "{option}");
            assert!(parse(&[&base[..], &[option, "2147483647"]].concat()).is_ok(), "{option}");
        }
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
    fn strlen_rejects_the_removed_also_option() {
        let args = ["strlen", "--vcf", "v", "--alignments", "a", "--reference", "r", "-o", "a.tsv"];
        let error = parse(&[&args[..], &["--also", "legacy=b.tsv"]].concat()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnknownArgument);
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

    /// The environment variable that makes [`documented_cli_help_matches_the_binary`] rewrite the
    /// documented help files instead of comparing them.
    const UPDATE_DOCS: &str = "WINNOWR_UPDATE_DOCS";

    /// The repository's `docs/` directory, or `None` when it is absent (the published crate leaves
    /// it out), in which case the documentation tests have nothing to check.
    fn docs_dir() -> Option<PathBuf> {
        let docs = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs");
        docs.is_dir().then_some(docs)
    }

    /// Text with the trailing white space of each line removed, ending in one newline.
    fn trim_lines(text: &str) -> String {
        let lines: Vec<&str> = text.trim_end().lines().map(str::trim_end).collect();
        format!("{}\n", lines.join("\n"))
    }

    /// The help that `winnowr --help` and `winnowr <command> --help` print in a terminal
    /// [`HELP_WIDTH`] columns wide, with the file name each is documented in (`winnowr` for the
    /// top level), with trailing white space trimmed.
    ///
    /// The help holds no version string (only `-V, --version`'s description), so it changes only
    /// when the options do.
    fn documented_help() -> Vec<(String, String)> {
        let command = Cli::command().term_width(HELP_WIDTH);
        let help = |args: &[&str]| -> String {
            let argv = std::iter::once("winnowr").chain(args.iter().copied()).chain(["--help"]);
            let error = command.clone().try_get_matches_from(argv).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::DisplayHelp, "{error}");
            trim_lines(&error.render().to_string())
        };
        let mut helps = vec![("winnowr".to_string(), help(&[]))];
        for subcommand in command.get_subcommands() {
            let name = subcommand.get_name().to_string();
            helps.push((name.clone(), help(&[name.as_str()])));
        }
        helps
    }

    /// The tool-reference page that shows a documented help file: `tools/overview.md` for the top
    /// level, else `tools/<command>.md`.
    fn tool_page(name: &str) -> String {
        if name == "winnowr" { "tools/overview.md".to_string() } else { format!("tools/{name}.md") }
    }

    /// The help in `docs/_generated/cli/` (included verbatim in the user guide's tool reference)
    /// is the binary's, and each file is included by its tool page, which the book's summary
    /// links. With `WINNOWR_UPDATE_DOCS=1`, rewrites the help files instead of comparing them.
    #[test]
    fn documented_cli_help_matches_the_binary() {
        let Some(docs) = docs_dir() else { return };
        let dir = docs.join("_generated/cli");
        let update = std::env::var_os(UPDATE_DOCS).is_some_and(|value| value == "1");
        let helps = documented_help();
        let mut problems = Vec::new();
        // A file of a command that no longer exists.
        for entry in std::fs::read_dir(&dir).into_iter().flatten() {
            let path = entry.unwrap().path();
            let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let is_help = path.extension().is_some_and(|extension| extension == "txt");
            if is_help && !helps.iter().any(|(command, _)| *command == name) {
                if update {
                    std::fs::remove_file(&path).unwrap();
                } else {
                    problems.push(format!("{} has no command", path.display()));
                }
            }
        }
        let mut stale = Vec::new();
        for (name, help) in &helps {
            let path = dir.join(format!("{name}.txt"));
            if update {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(&path, help).unwrap();
            } else {
                let documented = std::fs::read_to_string(&path).map(|text| trim_lines(&text));
                if documented.ok().as_ref() != Some(help) {
                    stale.push(path.display().to_string());
                }
            }
        }
        if !stale.is_empty() {
            problems.push(format!(
                "the documented CLI help is out of date: {}. Run `{UPDATE_DOCS}=1 cargo test \
                 documented_cli_help_matches_the_binary` and commit the changes",
                stale.join(", ")
            ));
        }
        let summary = std::fs::read_to_string(docs.join("src/SUMMARY.md")).unwrap_or_default();
        for (name, _) in &helps {
            let page = tool_page(name);
            let include = format!("{{{{#include ../../_generated/cli/{name}.txt}}}}");
            let text = std::fs::read_to_string(docs.join("src").join(&page)).unwrap_or_default();
            if !text.contains(&include) {
                problems.push(format!("docs/src/{page} does not contain {include}"));
            }
            if !summary.contains(&format!("({page})")) {
                problems.push(format!("docs/src/SUMMARY.md does not link {page}"));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// The Markdown files whose examples are checked: the README and the user guide's pages.
    fn documented_markdown(docs: &Path) -> Vec<PathBuf> {
        fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, files);
                } else if path.extension().is_some_and(|extension| extension == "md") {
                    files.push(path);
                }
            }
        }
        let mut files = vec![docs.parent().unwrap().join("README.md")];
        walk(&docs.join("src"), &mut files);
        files.sort();
        files
    }

    /// A command line without its `$ ` prompt and any leading `NAME=value` environment
    /// assignments or `time`, so `OMP_NUM_THREADS=8 winnowr ...` is checked too.
    fn strip_command_prefix(line: &str) -> &str {
        let mut line = line.strip_prefix("$ ").unwrap_or(line);
        loop {
            let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
            let assignment = word.split_once('=').is_some_and(|(name, _)| {
                !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            });
            if assignment || word == "time" {
                line = rest.trim_start();
            } else {
                return line;
            }
        }
    }

    /// One line of a shell command split into words as a POSIX shell splits them, for the quoting
    /// the examples use: white space separates words; `'...'` quotes literally; `"..."` quotes
    /// with `\` escaping only `"`, `\`, `$` and `` ` ``; `\` outside quotes escapes the next
    /// character; and `#` at the start of a word, outside quotes, starts a comment. Returns the
    /// words, and whether the line ends in an unquoted `\` (continues on the next line).
    ///
    /// # Errors
    ///
    /// Returns an error naming the quote if a quote is not closed on the line.
    fn shell_words(line: &str) -> Result<(Vec<String>, bool), String> {
        let mut words = Vec::new();
        let mut word: Option<String> = None;
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                c if c.is_whitespace() => words.extend(word.take()),
                '#' if word.is_none() => break,
                '\\' => match chars.next() {
                    Some(next) => word.get_or_insert_default().push(next),
                    None => {
                        words.extend(word.take());
                        return Ok((words, true));
                    }
                },
                '\'' => {
                    let word = word.get_or_insert_default();
                    loop {
                        match chars.next() {
                            Some('\'') => break,
                            Some(c) => word.push(c),
                            None => return Err("unmatched single quote".to_string()),
                        }
                    }
                }
                '"' => {
                    let word = word.get_or_insert_default();
                    loop {
                        match chars.next() {
                            Some('"') => break,
                            Some('\\') => match chars.peek() {
                                Some(&next @ ('"' | '\\' | '$' | '`')) => {
                                    word.push(next);
                                    chars.next();
                                }
                                _ => word.push('\\'),
                            },
                            Some(c) => word.push(c),
                            None => return Err("unmatched double quote".to_string()),
                        }
                    }
                }
                c => word.get_or_insert_default().push(c),
            }
        }
        words.extend(word);
        Ok((words, false))
    }

    /// The `winnowr <command> ...` command lines in a Markdown text's fenced code blocks (``` or
    /// ~~~), with their line numbers, each split into words by [`shell_words`]: `\` continuations
    /// are joined, and `$ ` prompts, leading environment assignments or `time`, and `#` comments
    /// (whole lines, or after a command) are dropped. Lines that do not start with `winnowr `
    /// (e.g. another tool's command) are skipped. A command whose quoting is broken is an `Err`
    /// saying why.
    ///
    /// # Panics
    ///
    /// If a `\` continuation is still open when its code block ends.
    fn example_commands(text: &str) -> Vec<(usize, Result<Vec<String>, String>)> {
        let mut commands = Vec::new();
        let mut in_block = false;
        let mut pending: Option<(usize, String)> = None;
        for (index, line) in text.lines().enumerate() {
            let fence = line.trim_start();
            if fence.starts_with("```") || fence.starts_with("~~~") {
                if let Some((number, _)) = pending {
                    panic!("the command at line {number} continues past the end of its code block");
                }
                in_block = !in_block;
                continue;
            }
            if !in_block {
                continue;
            }
            // Only a command's first line has a prompt or prefix to strip: a continuation line
            // is all arguments, which may look like assignments (e.g. `--regions a=a.bed`). The
            // physical lines are joined as a shell joins them, before splitting into words: a
            // `\` at the end of a line is removed with the newline, and the next line follows
            // as written, so its indentation still separates words and a continuation inside a
            // word joins it.
            let (number, joined) = match pending.take() {
                Some((number, joined)) => (number, joined + line.trim_end()),
                None => {
                    let line = strip_command_prefix(line.trim());
                    if !line.starts_with("winnowr ") {
                        continue;
                    }
                    (index + 1, line.to_string())
                }
            };
            match shell_words(&joined) {
                Err(error) => commands.push((number, Err(error))),
                Ok((_, true)) => {
                    let without = joined.strip_suffix('\\').unwrap_or(&joined).to_string();
                    pending = Some((number, without));
                }
                Ok((words, false)) => commands.push((number, Ok(words))),
            }
        }
        commands
    }

    #[rstest]
    #[case::plain("a b  c", &["a", "b", "c"], false)]
    #[case::single_quotes("a 'b c' d", &["a", "b c", "d"], false)]
    #[case::double_quotes(r#"a "b \"c\" $x \n" d"#, &["a", r#"b "c" $x \n"#, "d"], false)]
    #[case::escapes(r"a b\ c it\'s", &["a", "b c", "it's"], false)]
    #[case::joined(r#"--x="a b"c'd'"#, &["--x=a bcd"], false)]
    #[case::empty_word(r#"a "" b"#, &["a", "", "b"], false)]
    #[case::comment("a b # c 'd", &["a", "b"], false)]
    #[case::hash_in_a_word("a b#c", &["a", "b#c"], false)]
    #[case::hash_in_quotes(r##"a "#1" '#2'"##, &["a", "#1", "#2"], false)]
    #[case::continued(r"a b \", &["a", "b"], true)]
    fn shell_words_split_as_a_shell_does(
        #[case] line: &str,
        #[case] expected: &[&str],
        #[case] continues: bool,
    ) {
        let expected: Vec<String> = expected.iter().map(ToString::to_string).collect();
        assert_eq!(shell_words(line).unwrap(), (expected, continues));
    }

    #[rstest]
    #[case::single("a 'b c", "unmatched single quote")]
    #[case::double(r#"a "b c"#, "unmatched double quote")]
    #[case::escaped_close(r#"a "b\""#, "unmatched double quote")]
    fn shell_words_reject_an_unmatched_quote(#[case] line: &str, #[case] expected: &str) {
        assert_eq!(shell_words(line).unwrap_err(), expected);
    }

    #[test]
    fn example_commands_are_parsed_from_fenced_blocks() {
        let text = "winnowr score --vcf outside.vcf\n\
                    ```console\n\
                    # A comment\n\
                    $ winnowr strlen --vcf a.vcf \\\n  --reference r.fa  # measure\n\
                    samtools index a.bam\n\
                    winnowr --version\n\
                    ```\n\
                    ~~~\n\
                    OMP_NUM_THREADS=8 time winnowr train --help\n\
                    winnowr extract --vcf v.vcf --regions \\\n  a=a.bed\n\
                    ~~~\n";
        let commands: Vec<(usize, String)> = example_commands(text)
            .into_iter()
            .map(|(line, words)| (line, words.unwrap().join(" ")))
            .collect();
        assert_eq!(
            commands,
            [
                (4, "winnowr strlen --vcf a.vcf --reference r.fa".to_string()),
                (7, "winnowr --version".to_string()),
                (10, "winnowr train --help".to_string()),
                (11, "winnowr extract --vcf v.vcf --regions a=a.bed".to_string())
            ]
        );
    }

    #[test]
    #[should_panic(expected = "the command at line 2 continues past the end of its code block")]
    fn example_commands_reject_a_continuation_left_open() {
        example_commands("```\nwinnowr strlen \\\n```\n");
    }

    /// Every `winnowr` command line in the README's and the user guide's code blocks parses, so
    /// a renamed or removed option cannot leave an example behind. The files they name need not
    /// exist: only the command line is parsed.
    #[test]
    fn documented_example_commands_parse() {
        let Some(docs) = docs_dir() else { return };
        let mut failures = Vec::new();
        let mut checked = 0;
        for path in documented_markdown(&docs) {
            let text = std::fs::read_to_string(&path).unwrap();
            let (n, file_failures) = example_failures(&path.display().to_string(), &text);
            checked += n;
            failures.extend(file_failures);
        }
        assert!(checked > 0, "no example commands found under {}", docs.display());
        assert!(failures.is_empty(), "examples that do not parse:\n{}", failures.join("\n"));
    }

    /// The number of `winnowr` example commands in a Markdown text, and a message (naming `file`
    /// and the line) for each that does not parse.
    fn example_failures(file: &str, text: &str) -> (usize, Vec<String>) {
        let mut failures = Vec::new();
        let commands = example_commands(text);
        for (line, command) in &commands {
            let words = match command {
                Ok(words) => words,
                Err(error) => {
                    failures.push(format!("{file}:{line}: {error}"));
                    continue;
                }
            };
            let command = words.join(" ");
            // A command named alone lists it (e.g. the README's command list): it must exist.
            if let [_, name] = words.as_slice()
                && !name.starts_with('-')
            {
                if Cli::command().find_subcommand(name).is_none() {
                    failures.push(format!("{file}:{line}: no command {name}"));
                }
                continue;
            }
            if let Err(error) = Cli::command().try_get_matches_from(words) {
                // Help and version requests are valid command lines.
                if !matches!(error.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
                    let message = error.render().to_string();
                    let first = message.lines().next().unwrap_or_default();
                    failures.push(format!("{file}:{line}: {command}\n  {first}"));
                }
            }
        }
        (commands.len(), failures)
    }

    /// The failures of the example commands in one fenced block.
    fn block_failures(commands: &str) -> Vec<String> {
        example_failures("doc.md", &format!("```console\n{commands}\n```\n")).1
    }

    #[test]
    fn a_quoted_example_argument_is_one_word() {
        let failures = block_failures(r#"winnowr extract --vcf v -o t --sample "sample one""#);
        assert!(failures.is_empty(), "{failures:?}");
        let failures = block_failures(
            "winnowr extract --vcf v -o t --reference 'my ref.fa' \\\n  --sample=it\\'s",
        );
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_hash_inside_quotes_does_not_start_a_comment() {
        let failures =
            block_failures(r#"winnowr extract --vcf v -o t --sample "sample #1" --removed-option"#);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("--removed-option"), "{failures:?}");
    }

    #[test]
    fn an_unmatched_quote_in_an_example_is_reported() {
        let failures =
            block_failures("# setup\nwinnowr extract --vcf v -o t --sample \"sample one");
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].starts_with("doc.md:3:"), "{failures:?}");
        assert!(failures[0].contains("unmatched"), "{failures:?}");
    }

    #[test]
    fn a_continued_example_with_assignments_parses() {
        let failures = block_failures(
            "winnowr extract --vcf v.vcf -o t --regions \\\n  a=a.bed b=b.bed  # two sets",
        );
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_continuation_inside_a_word_joins_without_a_space() {
        // As a shell reads it: `winnowr extract--vcf v -o t`, which is no command.
        let failures = block_failures("winnowr extract\\\n--vcf v -o t");
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("extract--vcf"), "{failures:?}");
        // The same split after a space, or with the next line indented, is two words.
        for text in ["winnowr extract \\\n--vcf v -o t", "winnowr extract\\\n  --vcf v -o t"] {
            let failures = block_failures(text);
            assert!(failures.is_empty(), "{text:?}: {failures:?}");
        }
    }
}
