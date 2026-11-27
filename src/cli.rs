//! The command-line interface: global options, the subcommands, and dispatch.

// The doc comments of the options are their `--help` text, where `<output>`-style placeholders
// are literal, not HTML.
#![allow(rustdoc::invalid_html_tags)]

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use log::info;

use crate::alignment_provenance::{
    AlignmentProvenance, AlignmentTags, LazySource, ReadsAlignment, comparable_tags,
    compare_with_models, describe_differences, incomplete_options_reason, indeterminate_tags,
    merge, resolve_reads, training_provenance,
};
use crate::common::{AUTOSOMES, PhaseTimer, Task, VariantType, repeated};
#[cfg(test)]
use crate::evidence::read_tables;
use crate::evidence::read_tables_with_tags;
use crate::extraction::{TableInputs, TruthInputs, build_feature_table, check_table_inputs};
use crate::feature_source::FeatureSource;
use crate::features::feature_names;
use crate::labels::{HappyPassOnly, UnfixableFlips};
use crate::model::{
    CvFolds, MODEL_SUFFIX, ModelFile, ModelInfo, ReferenceContigs, TrainingParams, WinnowrModel,
    check_reference,
};
use crate::model_evidence::{
    Measuring, check_supplied, check_table, measure_and_decide, models_options, models_using,
    read_for_models, shared_options,
};
use crate::model_set::{
    Check, ModelSet, Problem, ReadsCheck, RegionPlan, ResolvedRegion, RunRegions, SetInputs,
    Severity, alignment_provenance_problems, bundled_bed_files, check_model_set, error_message,
    has_indel_genotype, mismatch_severity, models_feature_source, region_digests_differ,
};
use crate::pileup::{
    MeasuredSite, PileupOptions, PileupTables, measure_annotations_and_pileup, measure_pileup,
    write_pileup_table,
};
use crate::pileup_features::PILEUP_FEATURES;
use crate::policy::{
    Abstention, DEFAULT_KEEP_FALLBACK, DEFAULT_OUTSIDE_INDEL_T_DROP, NOT_DIFFICULT_REGIONS, Policy,
    decide_calls,
};
use crate::raw_records::AnnotationInputs;
use crate::read_annotations::{ReadAnnotationOptions, summarise};
use crate::read_pass::{AlignmentExpectation, ReadInputs};
use crate::regions::{
    NamedBed, NamedPath, RegionDigest, RegionSet, read_region_sets, training_region_digests,
};
use crate::repeat_features::STRLEN_FEATURES;
use crate::rescore::{Mode, Rule};
use crate::samples::{PromoteSampleCalls, SampleCalls};
use crate::scoring::write_scored_vcf;
use crate::strlen::{StrlenOptions, StrlenTables, measure_regions, write_strlen_table};
use crate::stutter::{
    LENGTH_CALL_FEATURES, StutterFit, StutterModel, fit_stutter_from_feature_tables,
};
use crate::subset::{restrict_table, restrict_to_contigs};
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
    /// The number of threads to use in all, including htslib's compression and decompression
    /// (but not htslib's I/O thread per file, which mostly waits on the disk) [default:
    /// OMP_NUM_THREADS when set, else 4].
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
    /// Check a set of models before a run, and report what they need.
    ///
    /// Reads each model's information, but not its booster (which `score` loads, and this does
    /// not check), and runs the model-set checks `score` runs at startup, and more, reporting
    /// every problem: that the models can be scored together (one model per task and variant
    /// type, the same contigs and reference, one feature source, one setting of each evidence),
    /// that every region a model uses has a BED given with --regions or found beside the models
    /// as regions/NAME.bed.gz, .bed.bgz or .bed (each BED is read, and must have intervals on the
    /// models' contigs; for a BED `score` does not read, a problem is only a warning), and that
    /// the models' reference is that of --reference or --vcf. Warnings (e.g.
    /// a model trained by another winnowr version) do not fail it; errors do. With --json, prints
    /// a JSON summary (schema version 1) on stdout, also when it fails.
    CheckModels(CheckModelsArgs),
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
            Self::CheckModels(_) => "check-models",
            Self::Score(_) => "score",
            Self::Strlen(_) => "strlen",
            Self::Pileup(_) => "pileup",
            Self::Subset(_) => "subset",
        }
    }
}

/// The `--alignment-tags` option of the commands that record or compare how reads were aligned.
#[derive(Debug, Args)]
pub(crate) struct AlignmentTagsArgs {
    /// How the reads were aligned, as KEY=VALUE tags (e.g. alt-project=primary; KEY= for a flag
    /// without a value) and KEY! to remove a detected tag, or none to record nothing (and read no
    /// header or table for it). They replace the tags of the same name read from the reads'
    /// header (its @PG records) and evidence tables, each before they must agree, and add the
    /// rest; without either, the tags set are the reads' whole alignment provenance, compared on
    /// those tags only (see the user guide: Alignment provenance).
    #[arg(long, value_name = "KEY=VALUE|KEY!|none", num_args = 1..)]
    pub(crate) alignment_tags: Vec<String>,
}

impl AlignmentTagsArgs {
    /// The tags given (`None` without the option).
    ///
    /// # Errors
    ///
    /// Returns an error if the tags are malformed (see [`AlignmentTags::parse`]).
    fn tags(&self) -> Result<Option<AlignmentTags>> {
        AlignmentTags::parse(&self.alignment_tags)
    }
}

/// The options of `winnowr extract`.
#[derive(Debug, Args)]
pub(crate) struct ExtractArgs {
    /// The indexed, single-sample VCF from HaplotypeCaller (gVCF mode, e.g. --dragen-mode, or a
    /// standard VCF with --alignments or --feature-source vcf).
    #[arg(long)]
    pub(crate) vcf: PathBuf,

    /// The feature table to write (Parquet).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// The indexed gVCF the VCF was genotyped from. It supplies the raw annotations
    /// (RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ) when the VCF lacks them, and is not read
    /// otherwise. Not used with --alignments or --feature-source vcf.
    #[arg(long, conflicts_with = "alignments")]
    pub(crate) gvcf: Option<PathBuf>,

    /// Where the features of HaplotypeCaller's gVCF-mode annotations (mq, dp_norm, qd,
    /// as_alt_frac, sor, n_alt) come from: gvcf (the raw annotations of the VCF or of --gvcf),
    /// alignments (qd, sor and n_alt derived from the VCF's FORMAT/AD, FORMAT/PL and FORMAT/SB or
    /// INFO/SOR, and mq, dp_norm and as_alt_frac measured from the reads of --alignments; for a
    /// VCF without a gVCF), or vcf (as alignments, but with mq and dp_norm from the VCF's INFO/MQ
    /// and INFO/DP and as_alt_frac missing; for a VCF without a gVCF or reads). The table records
    /// it: models train only on tables of one source, and score calls with the features of
    /// theirs [default: alignments with --alignments, else gvcf].
    #[arg(long, value_name = "SOURCE", ignore_case = true, value_enum)]
    pub(crate) feature_source: Option<ExtractSource>,

    /// The indexed BAM or CRAM of the sample's reads, which the alignments feature source
    /// measures each call's mapping qualities and depth from (see --feature-source and
    /// --alignment-settings), on --contigs. Needs --reference.
    #[arg(long, requires = "reference")]
    pub(crate) alignments: Option<PathBuf>,

    /// Which reads of --alignments are counted at a call, as comma-separated settings over the
    /// defaults, each at most once: mapqN (only reads with MAPQ >= N: HaplotypeCaller's minimum,
    /// 20, or 1 in DRAGEN mode; not the high bin, as in `pileup --settings`), flankN (reads within N bases of the call's REF, and the aligned flank bases an indel
    /// read needs on each side of the repeat tract to be counted at an allele) and bqN (the
    /// minimum base quality of a SNP base counted at an allele). The table records them as part of
    /// its feature source: a model scores only calls measured as its training tables were
    /// [default: mapq20,bq10,flank5].
    #[arg(long, value_name = "SETTINGS", requires = "alignments")]
    pub(crate) alignment_settings: Option<ReadAnnotationOptions>,

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

    /// How the reads were aligned, recorded in the table (and so in models trained on it). Read
    /// from the --strlen and --pileup tables and the header of --alignments, which must agree.
    #[command(flatten)]
    pub(crate) alignment_tags: AlignmentTagsArgs,

    /// The depth that the depth feature (dp_norm) divides each call's RAW_MQandDP depth (INFO/DP
    /// with --feature-source vcf, the reads counted with --alignments) by, in place of the median
    /// depth of the calls on --contigs
    /// [default: that median]; at least 1. Use it for calls of only part of the genome (e.g. a
    /// target footprint), whose median depth is skewed, with the median depth of the sample's
    /// genome-wide calls (which a whole-genome extract logs).
    #[arg(long, value_parser = parse_dp_median)]
    pub(crate) dp_median: Option<f64>,

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

    /// The indexed, uncompressed reference FASTA: needed by --alignments (also used to decode
    /// CRAM), and used to left-align indels when matching non-PASS calls to the truth (without
    /// it, alleles are only trimmed).
    #[arg(long)]
    pub(crate) reference: Option<PathBuf>,
}

/// A feature source as `extract --feature-source` names it. It exists so that the value carries no
/// read settings: the alignments source takes them from --alignment-settings, where
/// [`FeatureSource::VcfAlignments`] holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ExtractSource {
    /// The raw annotations of the VCF, or of --gvcf ([`FeatureSource::Gvcf`]).
    #[value(help = None::<&str>)]
    Gvcf,
    /// The VCF's standard annotations ([`FeatureSource::Vcf`]).
    #[value(help = None::<&str>)]
    Vcf,
    /// The VCF's standard annotations and the reads of --alignments
    /// ([`FeatureSource::VcfAlignments`]).
    #[value(help = None::<&str>)]
    Alignments,
}

impl std::fmt::Display for ExtractSource {
    /// The source's name on the command line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use clap::ValueEnum as _;
        let value = self.to_possible_value().expect("every source has a name");
        f.write_str(value.get_name())
    }
}

/// The options of `winnowr train`.
#[derive(Debug, Args)]
pub(crate) struct TrainArgs {
    /// Labelled feature tables (from `extract --happy`), one per training sample: at least two,
    /// of distinct samples, on the same reference. They may cover different contigs: the model
    /// records the contigs of all the tables, and `score` scores the calls on all of them.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) tables: Vec<PathBuf>,

    /// The variant type to train for (snp or indel).
    #[arg(long, value_enum, ignore_case = true)]
    pub(crate) variant_type: VariantType,

    /// The model file to write (`.winnowr.json` is appended if missing).
    #[arg(long, short = 'o')]
    pub(crate) output: PathBuf,

    /// The region BEDs of the decision policy the model will be scored with, as NAME=PATH (today
    /// only notinalldifficultregions=PATH, GIAB's GRCh38 genome stratification): each is read
    /// only to record its digest in the model, so that score can check the BED it is given or
    /// finds beside the model.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) policy_regions: Vec<NamedBed>,

    /// Train on tables of reads aligned differently (e.g. to make a model of both): the model
    /// records only the alignment tags the tables share, and score compares it with the reads on
    /// those tags only. Without it, tables whose recorded alignment provenance differs are an
    /// error.
    #[arg(long)]
    pub(crate) mixed_alignments: bool,

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
    /// region features come from the table. Without them, regions/notinalldifficultregions.bed.gz,
    /// .bed.bgz or .bed beside the models is used.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,

    /// Use a feature table, or a not-difficult regions BED (given or found beside the models),
    /// whose region BED digests are not the ones the models record, with a warning, rather than
    /// failing. Models that record different digests of one region are still an error.
    #[arg(long)]
    pub(crate) allow_region_mismatch: bool,

    /// The contigs to evaluate: only their calls and truth FNs are counted (e.g. the chromosomes
    /// held out from a model's training) [default: all the table's contigs]. Needs --happy, and a
    /// table as extracted, not one from `subset`.
    #[arg(long, num_args = 1.., requires = "happy")]
    pub(crate) contigs: Vec<String>,

    /// The hap.py output VCF the table was labelled from, to recount its truth FNs on --contigs.
    #[arg(long, requires = "contigs")]
    pub(crate) happy: Option<PathBuf>,
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

/// Parses a depth for `dp_norm` to divide by (see [`crate::features::check_dp_median`]).
///
/// # Errors
///
/// Returns an error naming the value if it is not a number, not finite, or below one read.
fn parse_dp_median(text: &str) -> Result<f64> {
    let value: f64 = text.parse().with_context(|| format!("not a number: {text}"))?;
    crate::features::check_dp_median(value)
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
    /// Validates both policy option lists before any potentially blocking BED read.
    fn validate(&self) -> Result<()> {
        without_none("--keep-fallback", &self.keep_fallback, KeepFallback::None)?;
        without_none("--abstain-actions", &self.abstain_actions, AbstainAction::None)?;
        Ok(())
    }

    /// True if the policy reads the not-difficult regions (when it has them): to withhold
    /// actions outside them, or, for an indel genotype model (`indel_genotype`), to drop indel
    /// calls outside them at the outside threshold. The one rule for whether a run reads them.
    fn reads_not_difficult(&self, indel_genotype: bool) -> bool {
        let abstains = self.abstain_actions.iter().any(|action| *action != AbstainAction::None);
        abstains || (indel_genotype && self.outside_indel_t_drop.0.is_some())
    }

    /// Checks that the policy has the not-difficult regions (`found`: given, or found beside the
    /// models) if it withholds actions outside them, which it cannot do without them. `score`
    /// calls it right after it looks for the regions, before any long pass; [`PolicyArgs::policy`]
    /// (which `evaluate` and `stratify` use) calls it too.
    ///
    /// # Errors
    ///
    /// Returns an error if `none` is given with other values of --abstain-actions, or actions are
    /// withheld and the regions were not `found`.
    fn require_not_difficult(&self, found: bool) -> Result<()> {
        let abstain_actions =
            without_none("--abstain-actions", &self.abstain_actions, AbstainAction::None)?;
        if !found && !abstain_actions.is_empty() {
            bail!(
                "--abstain-actions needs the not-difficult regions: pass --regions \
                 {NOT_DIFFICULT_REGIONS}=PATH, or put them beside the models as {}",
                bundled_bed_files(NOT_DIFFICULT_REGIONS)
            );
        }
        Ok(())
    }

    /// The policy in a mode, with the not-difficult regions (`not_difficult`: given, or found
    /// beside the models, and read by the caller only if the policy reads them, see
    /// [`PolicyArgs::reads_not_difficult`], so an unused BED cannot fail the run), checking they
    /// name some of `contigs`.
    ///
    /// Without them, the outside indel drop threshold is not applied (with a warning when
    /// `indel_genotype`, i.e. an indel genotype model will score).
    ///
    /// # Errors
    ///
    /// Returns an error if `none` is given with other values of --keep-fallback or
    /// --abstain-actions, actions are withheld without the not-difficult regions, or they name
    /// none of `contigs`.
    fn policy<S: AsRef<str>>(
        &self,
        mode: Mode,
        contigs: &[S],
        not_difficult: Option<RegionSet>,
        indel_genotype: bool,
    ) -> Result<Policy> {
        let keep_fallback =
            without_none("--keep-fallback", &self.keep_fallback, KeepFallback::None)?;
        let abstain_actions =
            without_none("--abstain-actions", &self.abstain_actions, AbstainAction::None)?;
        self.require_not_difficult(not_difficult.is_some())?;
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
        match not_difficult {
            Some(regions) => {
                regions.check_contigs(contigs)?;
                info!("Not-difficult regions (acting fully inside): {}", regions.source());
                policy.abstain =
                    Some(Abstention { regions, flip, regenotype, promote, indel_t_drop });
            }
            // Without them no action is withheld (see `require_not_difficult`).
            None => {
                if let Some(t_drop) = indel_t_drop.filter(|_| indel_genotype) {
                    log::warn!(
                        "the not-difficult regions are neither given (--regions \
                         {NOT_DIFFICULT_REGIONS}=PATH) nor beside the models ({}), so indel calls \
                         are dropped at the model's threshold everywhere; give them (GIAB's GRCh38 \
                         genome stratification {NOT_DIFFICULT_REGIONS}.bed.gz) to drop indel calls \
                         outside them at P(drop) >= {t_drop} instead",
                        bundled_bed_files(NOT_DIFFICULT_REGIONS)
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
    /// region features come from the table. Not reported within: see --report-region. Without
    /// them, regions/notinalldifficultregions.bed.gz, .bed.bgz or .bed beside the models is used.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,

    /// Use a feature table, or a not-difficult regions BED (given or found beside the models),
    /// whose region BED digests are not the ones the models record, with a warning, rather than
    /// failing. Models that record different digests of one region are still an error.
    #[arg(long)]
    pub(crate) allow_region_mismatch: bool,

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
    /// The indexed, single-sample VCF from HaplotypeCaller (gVCF mode, e.g. --dragen-mode, or a
    /// standard VCF for models trained on features from the VCF, or from the VCF and the reads:
    /// see `extract --feature-source`).
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
    /// annotations (RAW_MQandDP, QUALapprox, VarDP, AS_RAW_MQ) and the models were trained on
    /// features from the gVCF (feature source gvcf). Models trained on features from the VCF
    /// (`extract --feature-source vcf`), or from the VCF and the reads (`extract --alignments`),
    /// ignore it.
    #[arg(long)]
    pub(crate) gvcf: Option<PathBuf>,

    /// The depth that the depth feature (dp_norm) divides each call's RAW_MQandDP depth (INFO/DP
    /// for models trained on features from the VCF, the reads counted for models trained on
    /// features measured from the reads) by, in place of the median depth of the calls
    /// on the models' contigs [default: that median]; at least 1. Use it for calls of only part of
    /// the genome (e.g. a target footprint), whose median depth is skewed, with the median depth
    /// of the sample's genome-wide calls (which a whole-genome extract or score logs), as the
    /// models' training tables were normalized.
    #[arg(long, value_parser = parse_dp_median)]
    pub(crate) dp_median: Option<f64>,

    /// Region BEDs as NAME=PATH, with the names the models were trained with, plus the
    /// not-difficult regions as notinalldifficultregions=PATH (GIAB's GRCh38 genome stratification;
    /// outside them, indel calls are dropped at --outside-indel-t-drop in place of the model's
    /// threshold). A region not given is found beside the models, as regions/NAME.bed.gz,
    /// regions/NAME.bed.bgz or regions/NAME.bed. Each BED used is checked against the digest the
    /// models record (see --allow-region-mismatch).
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
    /// models were trained with. Models trained on features measured from the reads (`extract
    /// --alignments`) need it: each call's mapping qualities and depth are measured from it
    /// before any model scores. Needs --reference. Otherwise the reads are measured while the
    /// models that do not need them score, the two sharing -t threads (score runs within -t
    /// threads in all, apart from htslib's I/O thread per file, so set -t to the number of cores
    /// it may use). With -t 1, or when every
    /// model needs the evidence, the reads are measured first, then the models score.
    #[arg(long, requires = "reference")]
    pub(crate) alignments: Option<PathBuf>,

    /// The indexed, uncompressed reference FASTA, for --alignments (also used to decode CRAM).
    #[arg(long, requires = "alignments")]
    pub(crate) reference: Option<PathBuf>,

    /// How the reads were aligned (read from the header of --alignments and the --strlen and
    /// --pileup tables, which must agree), compared with how the models' training reads were.
    #[command(flatten)]
    pub(crate) provenance: ProvenanceCheckArgs,

    /// Use a region BED (given or found beside the models) whose digest is not the one the
    /// models record, with a warning, rather than failing. Models that record different digests
    /// of one region are still an error: no BED is the one all of them were trained with.
    #[arg(long)]
    pub(crate) allow_region_mismatch: bool,
}

/// The options of `score` and `check-models` that check how the reads were aligned against how
/// the models' training reads were.
#[derive(Debug, Args)]
pub(crate) struct ProvenanceCheckArgs {
    /// How the reads were aligned.
    #[command(flatten)]
    pub(crate) alignment_tags: AlignmentTagsArgs,

    /// Fail when the reads were not aligned as the models' training reads were, when the models
    /// disagree on it, or when the models record it and the reads' is not known; without it,
    /// these are warnings. Models that record no alignment provenance are not checked.
    #[arg(long)]
    pub(crate) strict_provenance: bool,
}

/// The options of `winnowr check-models`.
#[derive(Debug, Args)]
pub(crate) struct CheckModelsArgs {
    /// The model files (`<name>.winnowr.json`), as given to `score --models`.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) models: Vec<PathBuf>,

    /// Region BEDs as NAME=PATH, as given to `score --regions`; each is read and checked (a
    /// problem of one `score` does not read is a warning). A region not given is found beside the
    /// models, as `score` finds it.
    #[arg(long, value_name = "NAME=PATH", num_args = 1..)]
    pub(crate) regions: Vec<NamedBed>,

    /// The reference FASTA with its `.fai` beside it, or the `.fai` itself: only the index is
    /// read, and the models' reference contigs are checked against its lengths.
    #[arg(long)]
    pub(crate) reference: Option<PathBuf>,

    /// The VCF to be scored: only its header is read, and the models' reference contigs are
    /// checked against its contigs.
    #[arg(long)]
    pub(crate) vcf: Option<PathBuf>,

    /// The BAM or CRAM of the reads to be scored: only its header is read (no reference is
    /// needed), and how its reads were aligned (its @PG records) is checked against how the
    /// models' training reads were.
    #[arg(long)]
    pub(crate) alignments: Option<PathBuf>,

    /// How the reads to be scored were aligned, compared with how the models' training reads
    /// were (as in `score`).
    #[command(flatten)]
    pub(crate) provenance: ProvenanceCheckArgs,

    /// A region BED (given or found beside the models) whose digest is not the one the models
    /// record is a warning, not an error (as `score --allow-region-mismatch`). Models that record
    /// different digests of one region are still an error.
    #[arg(long)]
    pub(crate) allow_region_mismatch: bool,

    /// Print the summary as JSON on stdout (schema version 1), with every problem.
    #[arg(long)]
    pub(crate) json: bool,
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

    /// How the reads were aligned, recorded in the table: read from the header of --alignments.
    #[command(flatten)]
    pub(crate) alignment_tags: AlignmentTagsArgs,
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

    /// How the reads were aligned, recorded in the table: read from the header of --alignments.
    #[command(flatten)]
    pub(crate) alignment_tags: AlignmentTagsArgs,
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
    let tags = args.alignment_tags.tags()?;
    let alignment = recorded_alignment(tags.as_ref(), Some(&args.alignments), Vec::new())?;
    let inputs = ReadInputs {
        expected_alignment: alignment_expectation(&alignment, tags.as_ref(), true),
        ..ReadInputs::new(&args.vcf, &args.alignments, &args.reference)
    };
    let rows = measure_pileup(&inputs, &args.regions, &args.settings)?;
    let n = write_pileup_table(&args.output, &rows, args.settings, alignment.provenance.as_ref())?;
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
    check_output_dir("--output", &args.output)?;
    let tags = args.alignment_tags.tags()?;
    let alignment = recorded_alignment(tags.as_ref(), Some(&args.alignments), Vec::new())?;
    let inputs = ReadInputs {
        expected_alignment: alignment_expectation(&alignment, tags.as_ref(), true),
        ..ReadInputs::new(&args.vcf, &args.alignments, &args.reference)
    };
    let sites = measure_regions(&inputs, &args.regions, &args.reads)?;
    let n_sites = sites.len();
    let n = write_strlen_table(&args.output, &sites, args.reads, alignment.provenance.as_ref())?;
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
    let inputs = [
        ("VCF", Some(args.vcf.as_path()), InputIndexes::Vcf),
        ("alignments", args.alignments.as_deref(), InputIndexes::Alignments),
        ("reference", args.reference.as_deref(), InputIndexes::Reference),
        ("hap.py VCF", args.happy.as_deref(), InputIndexes::Vcf),
        ("truth VCF", args.truth_vcf.as_deref(), InputIndexes::Vcf),
        ("truth BED", args.truth_bed.as_deref(), InputIndexes::None),
    ];
    check_output_not_inputs(&args.output, &inputs, &args.regions, &args.strlen, &args.pileup)?;
    // Checked before any input is read.
    let feature_source = extract_feature_source(args)?;
    check_unread_gvcf(feature_source, args.gvcf.as_deref(), UnreadGvcf::Error)?;
    let truth = TruthInputs {
        happy: args.happy.as_deref(),
        happy_pass_only: args.happy_pass_only,
        truth_vcf: args.truth_vcf.as_deref(),
        truth_bed: args.truth_bed.as_deref(),
        reference: truth_reference(args),
    };
    truth.check()?;
    let regions = read_region_sets(&args.regions)?;
    let summary = VcfSummary::read(&args.vcf)?;
    let sample = args.sample.clone().unwrap_or_else(|| summary.sample_name(&args.vcf));
    let tags = args.alignment_tags.tags()?;
    let strlen = if args.strlen.is_empty() {
        None
    } else {
        Some(read_tables_with_tags::<StrlenOptions>(&args.strlen, tags.as_ref())?)
    };
    if let Some(strlen) = &strlen {
        info!("The strlen tables were measured with read settings {}", strlen.options);
    }
    let pileup = if args.pileup.is_empty() {
        None
    } else {
        Some(read_tables_with_tags::<PileupOptions>(&args.pileup, tags.as_ref())?)
    };
    if let Some(pileup) = &pileup {
        info!("The pileup tables were measured with settings {}", pileup.options);
    }
    let alignment = extract_alignment(args, strlen.as_ref(), pileup.as_ref())?;
    let reads = read_inputs(&args.vcf, args.alignments.as_deref(), args.reference.as_deref()).map(
        |mut inputs| {
            inputs.expected_alignment = alignment_expectation(&alignment, tags.as_ref(), true);
            inputs
        },
    );
    let checks = TableChecks {
        vcf: &args.vcf,
        contigs: &args.contigs,
        regions: &regions,
        truth: &truth,
        dp_median: args.dp_median,
    };
    let source = prepare_source(feature_source, args.gvcf.as_deref(), reads, None, &checks)?;
    let inputs = TableInputs {
        annotations: source.annotations(),
        outputs: std::slice::from_ref(&args.output),
        regions: &regions,
        strlen: strlen.as_ref(),
        pileup: pileup.as_ref(),
        truth,
        dp_median: args.dp_median,
        alignment: alignment.provenance.as_ref(),
    };
    let table = build_feature_table(&sample, &args.vcf, &args.contigs, &summary, &inputs)?;
    let mut timer = PhaseTimer::start();
    table.write_parquet(&args.output)?;
    timer.lap("writing the table");
    info!("Wrote {} calls for {sample} to {}", table.num_rows(), args.output.display());
    Ok(())
}

/// How a sample's reads were aligned (see [`resolve_reads`]): from `tags` (`--alignment-tags`),
/// the header of `alignments` (when given) and its evidence `tables` (each named, read only when
/// needed), with the header's warnings logged.
///
/// # Errors
///
/// Returns an error if the header or a table cannot be read, or the sources record different
/// provenance.
fn reads_alignment(
    tags: Option<&AlignmentTags>,
    alignments: Option<&Path>,
    tables: Vec<(String, LazySource<'_>)>,
) -> Result<ReadsAlignment> {
    let reads = resolve_reads(tags, alignments, tables)?;
    if let (Some(path), Some(header)) = (alignments, &reads.header) {
        let header = header.as_ref().map_err(|error| anyhow::anyhow!("{error}"))?;
        let warnings = [header.mixed_warning(path), header.unprofiled_warning(path)];
        for warning in warnings.into_iter().flatten() {
            log::warn!("{warning}");
        }
    }
    Ok(reads)
}

/// The alignment provenance a command records (see [`reads_alignment`]), logged.
///
/// # Errors
///
/// Returns an error if the header or a table cannot be read, or the sources record different
/// provenance.
fn recorded_alignment(
    tags: Option<&AlignmentTags>,
    alignments: Option<&Path>,
    tables: Vec<(String, LazySource<'_>)>,
) -> Result<ReadsAlignment> {
    let reads = reads_alignment(tags, alignments, tables)?;
    match &reads.provenance {
        Some(provenance) => info!("The output records the reads' alignment: {provenance}"),
        None => info!("The reads' alignment is not known, so the output records none"),
    }
    Ok(reads)
}

/// Binds measurement readers to the raw header checked before feature extraction or scoring.
fn alignment_expectation<'a>(
    reads: &'a ReadsAlignment,
    tags: Option<&'a AlignmentTags>,
    strict: bool,
) -> Option<AlignmentExpectation<'a>> {
    let header = reads.header.as_ref()?.as_ref().ok()?;
    Some(AlignmentExpectation {
        provenance: header.provenance.as_ref(),
        header_records: &header.header_records,
        tags,
        strict,
    })
}

/// The alignment provenance `extract` records (see [`recorded_alignment`]): from the --strlen and
/// --pileup tables it read (`strlen`, `pileup`) and the header of --alignments.
///
/// # Errors
///
/// Returns an error if the tags are malformed, the header cannot be read, or the sources record
/// different provenance.
fn extract_alignment(
    args: &ExtractArgs,
    strlen: Option<&StrlenTables>,
    pileup: Option<&PileupTables>,
) -> Result<ReadsAlignment> {
    let tags = args.alignment_tags.tags()?;
    let mut tables: Vec<(String, LazySource<'_>)> = Vec::new();
    if let Some(strlen) = strlen {
        tables.push(("the --strlen tables".to_string(), Box::new(|| Ok(strlen.alignment.clone()))));
    }
    if let Some(pileup) = pileup {
        tables.push(("the --pileup tables".to_string(), Box::new(|| Ok(pileup.alignment.clone()))));
    }
    recorded_alignment(tags.as_ref(), args.alignments.as_deref(), tables)
}

/// The reference `extract` matches non-PASS calls to the truth with: --reference, unless it is
/// there only for --alignments (the truth is not given).
fn truth_reference(args: &ExtractArgs) -> Option<&Path> {
    args.reference.as_deref().filter(|_| args.alignments.is_none() || args.truth_vcf.is_some())
}

/// The reads of `--alignments` and `--reference`, when both are given.
fn read_inputs<'a>(
    vcf: &'a Path,
    alignments: Option<&'a Path>,
    reference: Option<&'a Path>,
) -> Option<ReadInputs<'a>> {
    Some(ReadInputs::new(vcf, alignments?, reference?))
}

/// The inputs of a feature source, with the reads of the alignments source measured.
#[derive(Debug)]
enum SourceInputs<'a> {
    /// The gVCF source, with its gVCF (--gvcf).
    Gvcf(Option<&'a Path>),
    /// The VCF source.
    Vcf,
    /// The alignments source, its reads measured with `options`.
    Reads {
        /// The settings the reads were measured with.
        options: ReadAnnotationOptions,
        /// The sites of every SNP and indel record (see [`measure_annotations_and_pileup`]).
        sites: Vec<MeasuredSite>,
        /// The pileup evidence measured with them, if asked for.
        pileup: Option<PileupTables>,
    },
}

impl SourceInputs<'_> {
    /// The inputs of the feature table's raw annotations.
    fn annotations(&self) -> AnnotationInputs<'_> {
        match self {
            SourceInputs::Gvcf(gvcf) => AnnotationInputs::Gvcf(*gvcf),
            SourceInputs::Vcf => AnnotationInputs::Vcf,
            SourceInputs::Reads { options, sites, .. } => {
                AnnotationInputs::VcfAlignments { sites, options: *options }
            }
        }
    }

    /// The pileup evidence measured with the reads.
    fn pileup(&self) -> Option<&PileupTables> {
        match self {
            SourceInputs::Reads { pileup, .. } => pileup.as_ref(),
            SourceInputs::Gvcf(_) | SourceInputs::Vcf => None,
        }
    }
}

/// What [`prepare_source`] checks of the feature table before a pass over the reads.
struct TableChecks<'a> {
    /// The VCF.
    vcf: &'a Path,
    /// The contigs the table is of.
    contigs: &'a [String],
    /// The region sets the table flags.
    regions: &'a [RegionSet],
    /// The truth inputs that label it.
    truth: &'a TruthInputs<'a>,
    /// The depth `dp_norm` divides by.
    dp_median: Option<f64>,
}

/// The inputs of a feature source for `extract` and `score`: for the alignments source, the reads
/// (and, with `pileup`, the pileup evidence of those settings) measured at every call (see
/// [`measure_annotations_and_pileup`]) and summarised (see [`summarise`]). Before that pass, the
/// table's inputs and the VCF are checked with the checks that building the table runs
/// ([`check_table_inputs`], [`crate::raw_records::check_vcf`]), so that a mistake fails before
/// the pass rather than after it.
///
/// # Errors
///
/// Returns an error if the alignments source is not given the reads, a check fails, the reads
/// cannot be measured, or no read is counted at any call.
fn prepare_source<'a>(
    source: FeatureSource,
    gvcf: Option<&'a Path>,
    reads: Option<ReadInputs<'_>>,
    pileup: Option<PileupOptions>,
    checks: &TableChecks<'_>,
) -> Result<SourceInputs<'a>> {
    let options = match source {
        FeatureSource::Gvcf => return Ok(SourceInputs::Gvcf(gvcf)),
        FeatureSource::Vcf => return Ok(SourceInputs::Vcf),
        FeatureSource::VcfAlignments(options) => options,
    };
    let Some(reads) = reads else {
        bail!(
            "the {source} feature source measures the calls' mapping qualities and depth from the \
             reads: pass --alignments and --reference"
        );
    };
    let TableChecks { vcf, contigs, regions, truth, dp_median } = *checks;
    check_table_inputs(contigs, regions, truth, dp_median)?;
    crate::raw_records::check_vcf(vcf, source, None, contigs)?;
    info!(
        "Measuring the calls' mapping qualities and depth from {} with alignment settings \
         {options}",
        reads.alignments.display()
    );
    if let Some(pileup) = pileup {
        info!("Counting the pileup evidence with settings {pileup}");
    }
    let mut timer = PhaseTimer::start();
    let (sites, pileup) = measure_annotations_and_pileup(&reads, contigs, &options, pileup)?;
    timer.lap_total("measuring the reads");
    let calls =
        sites.iter().filter_map(|site| Some((site.row.chrom.as_str(), site.mapq.as_ref()?)));
    summarise(calls, contigs, &options)?;
    Ok(SourceInputs::Reads { options, sites, pileup })
}

/// The feature source of the table `extract` writes: its --feature-source (by default the
/// alignments source with --alignments, and the gVCF source without), with the read settings of
/// --alignment-settings for the alignments source.
///
/// # Errors
///
/// Returns an error if the alignments source is asked for without --alignments, or --alignments
/// is given with another source.
fn extract_feature_source(args: &ExtractArgs) -> Result<FeatureSource> {
    let default =
        if args.alignments.is_some() { ExtractSource::Alignments } else { ExtractSource::Gvcf };
    match (args.feature_source.unwrap_or(default), &args.alignments) {
        (ExtractSource::Alignments, Some(_)) => {
            Ok(FeatureSource::VcfAlignments(args.alignment_settings.unwrap_or_default()))
        }
        (ExtractSource::Alignments, None) => bail!(
            "--feature-source alignments measures the mapping qualities and depth from the reads: \
             pass --alignments and --reference"
        ),
        (source, Some(_)) => bail!(
            "--alignments is read only for the alignments feature source (the default with it), \
             not --feature-source {source}: leave out one of them"
        ),
        (ExtractSource::Vcf, None) => Ok(FeatureSource::Vcf),
        (ExtractSource::Gvcf, None) => Ok(FeatureSource::Gvcf),
    }
}

/// What [`check_unread_gvcf`] does with a --gvcf that the feature source does not read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnreadGvcf {
    /// An error (`extract`, where --feature-source vcf and --gvcf contradict each other).
    Error,
    /// A warning, ignoring it (`score`, whose source comes from the models).
    Warn,
}

/// Checks a given `gvcf` (--gvcf) against the feature source: only the gVCF source reads it, and
/// for the others `unread` says what to do with it.
///
/// # Errors
///
/// Returns an error if `gvcf` is given with a source that does not read it and `unread` is
/// [`UnreadGvcf::Error`].
fn check_unread_gvcf(source: FeatureSource, gvcf: Option<&Path>, unread: UnreadGvcf) -> Result<()> {
    match (source, gvcf, unread) {
        (FeatureSource::Gvcf, _, _) | (_, None, _) => Ok(()),
        (_, Some(_), UnreadGvcf::Error) => bail!(
            "the {source} feature source derives the features from the VCF, and does not read \
             --gvcf: leave out --gvcf, or use --feature-source gvcf"
        ),
        (_, Some(gvcf), UnreadGvcf::Warn) => {
            log::warn!(
                "the models were trained on features from feature source {source}, not from a \
                 gVCF, so --gvcf {} is ignored",
                gvcf.display()
            );
            Ok(())
        }
    }
}

/// The feature source every one of `items` has (by `source`), or `None` if there are none;
/// logged when it is not the default, as that of `whose` features.
///
/// # Errors
///
/// Returns the error `differ` makes of the first item and the first with another source.
fn shared_feature_source<T, I, F, E>(
    items: I,
    source: F,
    differ: E,
    whose: &str,
) -> Result<Option<FeatureSource>>
where
    I: IntoIterator<Item = T>,
    F: Fn(&T) -> FeatureSource,
    E: FnOnce(&T, &T) -> anyhow::Error,
{
    let shared = crate::feature_source::shared_source(items, source)
        .map_err(|(first, other)| differ(&first, &other))?;
    if let Some(shared) = shared.filter(|shared| !shared.is_default()) {
        info!("{whose} features come from feature source {shared}");
    }
    Ok(shared)
}

/// The training tables' metadata, as [`read_training_infos`] checks and merges it.
struct TrainingInfos {
    /// Each table's metadata, in the order of `--tables`.
    infos: Vec<FeatureTableInfo>,
    /// The region names the tables share.
    regions: Vec<String>,
    /// The contigs of any table (see [`merged_contigs`]).
    contigs: Vec<String>,
    /// The lengths of `contigs`, in their order (see [`merged_reference_contigs`]).
    reference_contigs: Option<ReferenceContigs>,
    /// The feature source the tables share.
    feature_source: FeatureSource,
}

/// Reads the training feature tables' metadata and checks that the tables can be trained on
/// together, merging their contigs and reference contigs.
///
/// The tables may cover different contigs (e.g. half of one sample's chromosomes, with other
/// samples' whole genomes); the model records the contigs of every table.
///
/// # Errors
///
/// Returns an error if there are fewer than two tables, a file is listed twice, the tables were
/// extracted with different regions or feature sources, or two tables record different lengths of
/// a contig.
fn read_training_infos(paths: &[PathBuf]) -> Result<TrainingInfos> {
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
    }
    let feature_source = shared_feature_source(
        &infos,
        |info| info.feature_source,
        |first, other| {
            anyhow::anyhow!(
                "--tables hold features from different sources: {}'s from {}, {}'s from {}; \
                 extract every training table with the same --feature-source (and, with \
                 --alignments, the same --alignment-settings)",
                first.sample,
                first.feature_source,
                other.sample,
                other.feature_source
            )
        },
        "The training tables'",
    )?
    .unwrap_or_default();
    let refs: Vec<&FeatureTableInfo> = infos.iter().collect();
    let contigs = merged_contigs(&refs);
    let partial: Vec<String> = infos
        .iter()
        .filter(|info| info.contigs.len() < contigs.len())
        .map(|info| format!("{} ({} of them)", info.sample, info.contigs.len()))
        .collect();
    if !partial.is_empty() {
        info!(
            "--tables cover different contigs: the model records all {} of them, which these \
             tables cover only some of: {}",
            contigs.len(),
            partial.join(", ")
        );
    }
    for sample in tables_sharing_no_contig(&refs) {
        log::warn!(
            "the --tables of {sample} share no contig with the others, so train cannot check that \
             they are of the same reference (their contig lengths are compared only where they \
             share a contig); differently named contigs (e.g. 1 and chr1) mean different references"
        );
    }
    let reference_contigs = merged_reference_contigs(&refs, &contigs)?;
    let regions = first.region_names.clone();
    Ok(TrainingInfos { infos, regions, contigs, reference_contigs, feature_source })
}

/// The digest of each `--policy-regions` BED, to record in the model.
///
/// # Errors
///
/// Returns an error if a name is not [`NOT_DIFFICULT_REGIONS`] or is given twice, or a BED cannot
/// be read.
fn policy_region_digests(regions: &[NamedBed]) -> Result<BTreeMap<String, RegionDigest>> {
    let mut names = std::collections::BTreeSet::new();
    for bed in regions {
        if bed.name != NOT_DIFFICULT_REGIONS {
            bail!(
                "--policy-regions takes only {NOT_DIFFICULT_REGIONS}=PATH (the policy's only region \
                 set), got {}",
                bed.name
            );
        }
        if !names.insert(&bed.name) {
            bail!("--policy-regions gives {} more than once", bed.name);
        }
    }
    let mut digests = BTreeMap::new();
    for bed in regions {
        let digest = RegionSet::from_bed(&bed.name, &bed.path)?.digest()?;
        info!("The policy's {} regions ({}): {digest}", bed.name, bed.path.display());
        digests.insert(bed.name.clone(), digest);
    }
    Ok(digests)
}

/// The contigs of any of the tables: those of the tables covering the most contigs first, ties
/// broken by sample name, so that a whole-genome table sets the genome's order and the order does
/// not depend on the order of `--tables`. Tables of the same contigs give those contigs, in their
/// order.
fn merged_contigs(infos: &[&FeatureTableInfo]) -> Vec<String> {
    let mut sorted = infos.to_vec();
    sorted.sort_by(|a, b| b.contigs.len().cmp(&a.contigs.len()).then(a.sample.cmp(&b.sample)));
    let merged: indexmap::IndexSet<&String> =
        sorted.into_iter().flat_map(|info| &info.contigs).collect();
    merged.into_iter().cloned().collect()
}

/// The samples of the tables that share no contig with any other table, whose reference
/// [`merged_reference_contigs`] therefore cannot check.
fn tables_sharing_no_contig<'a>(infos: &[&'a FeatureTableInfo]) -> Vec<&'a str> {
    infos
        .iter()
        .enumerate()
        .filter(|&(i, info)| {
            !infos.iter().enumerate().any(|(j, other)| {
                i != j && info.contigs.iter().any(|contig| other.contigs.contains(contig))
            })
        })
        .map(|(_, info)| info.sample.as_str())
        .collect()
}

/// The reference contigs of the training tables, to record in the model: the length of each of
/// `contigs` (the merged contigs, in their order) that the tables record, `None` where no table
/// knows it. `None` in all, with a warning, if a table does not record its reference contigs; the
/// lengths the other tables record are still checked against each other.
///
/// A length one table records and another does not know is not a conflict, as when `score` checks
/// a model's reference.
///
/// # Errors
///
/// Returns an error if two tables record different lengths of a contig (different references).
fn merged_reference_contigs(
    infos: &[&FeatureTableInfo],
    contigs: &[String],
) -> Result<Option<ReferenceContigs>> {
    let mut merged = ReferenceContigs::new();
    for contig in contigs {
        // The first known length, and the sample whose table recorded it.
        let mut known: Option<(u64, &str)> = None;
        for info in infos {
            let length = info.reference_contigs.as_ref().and_then(|lengths| lengths.get(contig));
            let Some(&Some(length)) = length else { continue };
            match known {
                None => known = Some((length, &info.sample)),
                Some((first, sample)) if first != length => bail!(
                    "--tables come from different references: {sample} and {} record different \
                     lengths of {contig} ({first} and {length})",
                    info.sample
                ),
                Some(_) => {}
            }
        }
        merged.insert(contig.clone(), known.map(|(length, _)| length));
    }
    if infos.iter().any(|info| info.reference_contigs.is_none()) {
        log::warn!(
            "some --tables do not record their reference contigs; the model will not record them \
             either, so score cannot check the reference"
        );
        return Ok(None);
    }
    Ok(Some(merged))
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

/// Loads one training input using the same column selection as fitting, checking that the
/// metadata preflight read (`expected`), which the model records, has not changed since: e.g. its
/// alignment provenance and its region BED digests.
///
/// # Errors
///
/// Returns an error naming the table and what changed if it cannot be read, or its metadata has
/// changed.
fn read_training_table(
    path: &Path,
    selection: &TableSelection,
    expected: &FeatureTableInfo,
) -> Result<FeatureTable> {
    let table = FeatureTable::read_parquet_selected(path, selection)?;
    let info = table.info();
    let changed: Vec<&str> = [
        ("alignment provenance", info.alignment != expected.alignment),
        ("region BED digests", info.region_digests != expected.region_digests),
        ("sample", info.sample != expected.sample),
        ("contigs", info.contigs != expected.contigs),
        ("reference contigs", info.reference_contigs != expected.reference_contigs),
        ("region names", info.region_names != expected.region_names),
        ("feature source", info.feature_source != expected.feature_source),
        ("repeat-length settings", info.strlen_settings != expected.strlen_settings),
        ("pileup settings", info.pileup_settings != expected.pileup_settings),
    ]
    .into_iter()
    .filter_map(|(what, changed)| changed.then_some(what))
    .collect();
    if !changed.is_empty() {
        bail!(
            "{} has changed {} since training preflight; read and check the training inputs again",
            path.display(),
            changed.join(" and ")
        );
    }
    Ok(table)
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
    let inputs = || {
        args.tables
            .iter()
            .map(|path| ("feature table", path.as_path()))
            .chain(args.policy_regions.iter().map(|bed| ("policy region BED", bed.path.as_path())))
    };
    crate::scoring::check_output_not_input(&output, inputs())?;
    let TrainingInfos { infos, regions, contigs, reference_contigs, feature_source } =
        read_training_infos(&args.tables)?;
    let provenances: Vec<(&str, Option<&AlignmentProvenance>)> =
        infos.iter().map(|info| (info.sample.as_str(), info.alignment.as_ref())).collect();
    let alignment = training_provenance(&provenances, args.mixed_alignments)?;
    let table_digests: Vec<(&str, &BTreeMap<String, RegionDigest>)> =
        infos.iter().map(|info| (info.sample.as_str(), &info.region_digests)).collect();
    let region_digests = training_region_digests(&table_digests, &regions)?;
    let policy_region_digests = policy_region_digests(&args.policy_regions)?;
    for (name, policy_digest) in &policy_region_digests {
        if let Some(feature_digest) =
            region_digests.get(name).filter(|digest| *digest != policy_digest)
        {
            log::warn!(
                "the {name} feature and policy region digests differ: feature {feature_digest}, \
                 policy {policy_digest}; train records both, but score cannot use one live BED for \
                 both roles (even with --allow-region-mismatch); use the same BED for both roles \
                 to score with that policy, or disable the policy's use of this BED. Table-based \
                 evaluate and stratify can use different BEDs for the two roles"
            );
        }
    }
    let info_refs: Vec<&FeatureTableInfo> = infos.iter().collect();
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
    let read = |path: &PathBuf| {
        let index =
            args.tables.iter().position(|candidate| candidate == path).expect("training input");
        read_training_table(path, &selection, &infos[index])
    };
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
        feature_source,
        alignment,
        region_digests,
        policy_region_digests,
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
    crate::scoring::check_output_not_input(&output, inputs())?;
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
    evaluate_with(args, threads, || {})
}

/// Evaluates with a callback after consuming the policy BED and before any report is written.
fn evaluate_with<F: FnOnce()>(
    args: &EvaluateArgs,
    threads: NonZeroUsize,
    before_reports: F,
) -> Result<()> {
    use crate::evaluation::{
        EvaluationPolicy, baseline_metric, evaluate_model, evaluate_promote_model,
        write_policy_metrics_protecting,
    };
    args.policy.validate()?;
    check_policy_region_names(&args.regions)?;
    check_output_dir("--output", &args.output)?;
    // The output is none of the inputs: checked now, and against the not-difficult regions BED
    // once it is found, before it is read.
    let outputs = [args.output.clone()];
    let inputs = args
        .models
        .iter()
        .map(|model| ("model", model.as_path()))
        .chain([("feature table", args.table.as_path())])
        .chain(args.happy.iter().map(|happy| ("hap.py VCF", happy.as_path())));
    check_no_output_is_input(&outputs, inputs)?;
    let files = ModelFile::resolve_all(&args.models);
    let models = load_models(&files, threads)?;
    check_models_region_digests(&models, &files)?;
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
    // The contigs the calls were extracted from, before any --contigs, which the not-difficult
    // regions are checked against (see `stratify`).
    let extracted = table.info().contigs.clone();
    // clap gives --contigs and --happy together or not at all.
    let table = match &args.happy {
        None => table,
        Some(happy) => restrict_to_contigs(&table, happy, &args.contigs)?,
    };
    let info = table.info();
    check_labelled_table(&args.table, info, &models, &args.models, args.allow_region_mismatch)?;
    warn_policy_only_regions(&args.regions);
    let indel_genotype = !genotype.is_empty() && variant_type == VariantType::Indel;
    let infos: Vec<&ModelInfo> = models.iter().map(|model| &model.info).collect();
    let (not_difficult, protected_beds) = policy_not_difficult(
        &infos,
        &files,
        &args.regions,
        args.policy.reads_not_difficult(indel_genotype),
        args.allow_region_mismatch,
        &outputs,
    )?;
    // The mode is a placeholder: evaluate reports both.
    let policy = args.policy.policy(Mode::Genotyper, &extracted, not_difficult, indel_genotype)?;
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
    before_reports();
    write_policy_metrics_protecting(&args.output, &metrics, &protected_beds)
}

/// Loads model files, each from the file it resolved to, logging their warnings (see
/// [`crate::model::ModelInfo::warnings`]).
///
/// # Errors
///
/// Returns an error if a model cannot be loaded.
fn load_models(files: &[ModelFile], threads: NonZeroUsize) -> Result<Vec<WinnowrModel>> {
    files
        .iter()
        .map(|file| {
            let model = WinnowrModel::load_file(file, threads)?;
            model.info.warnings(&file.path).log();
            Ok(model)
        })
        .collect()
}

/// Checks that a labelled feature table (of `evaluate` or `stratify`) suits the models: that it is
/// on their reference and has the evidence they need, measured as they need it. Warns if its
/// sample is a training sample of a model, as its metrics are then in-sample.
///
/// # Errors
///
/// Returns an error if the table is not on a model's reference, holds features from another
/// source than a model was trained on, lacks evidence a model needs, or flags a region with
/// another BED than a model was trained with (unless `allow_region_mismatch`).
fn check_labelled_table(
    table: &Path,
    info: &FeatureTableInfo,
    models: &[WinnowrModel],
    paths: &[PathBuf],
    allow_region_mismatch: bool,
) -> Result<()> {
    check_table_reference(table, info, models, paths)?;
    check_table_region_digests(table, info, models, paths, allow_region_mismatch)?;
    check_table_feature_source(table, info, models, paths)?;
    check_table::<StrlenOptions>(table, info, models, paths)?;
    check_table::<PileupOptions>(table, info, models, paths)?;
    for warning in table_alignment_warnings(table, info, models, paths) {
        log::warn!("{warning}");
    }
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

/// Checks that the models of `evaluate` or `stratify` record the same digest of each region their
/// features use: no labelled table can then be of the BED all were trained with. It depends on the
/// models alone, so it is checked as soon as they are loaded, before any table is read.
///
/// # Errors
///
/// Returns an error naming the region and two models if the models record different digests of a
/// region, even with `--allow-region-mismatch`.
fn check_models_region_digests(models: &[WinnowrModel], files: &[ModelFile]) -> Result<()> {
    let infos: Vec<&ModelInfo> = models.iter().map(|model| &model.info).collect();
    let paths: Vec<PathBuf> = files.iter().map(|file| file.path.clone()).collect();
    let resolved: Vec<PathBuf> = files.iter().map(|file| file.resolved.clone()).collect();
    let given_dirs: Vec<PathBuf> = files.iter().map(|file| file.given_dir.clone()).collect();
    let set =
        ModelSet { paths: &paths, resolved: &resolved, given_dirs: &given_dirs, infos: &infos };
    if let Some(message) = error_message(&region_digests_differ(&set, false)) {
        bail!("{message}");
    }
    Ok(())
}

/// Checks that a labelled feature table flags each region a model uses with the BED the model was
/// trained with, by their recorded digests. Where a model records one and the table none, it
/// cannot be checked: warned, once per region. Nothing is checked where a model records none, and
/// models that record different digests of a region are [`check_models_region_digests`]'s.
///
/// # Errors
///
/// Returns an error naming the region, the table and the model if their digests differ, unless
/// `allow_mismatch`, which makes that a warning.
fn check_table_region_digests(
    table: &Path,
    info: &FeatureTableInfo,
    models: &[WinnowrModel],
    paths: &[PathBuf],
    allow_mismatch: bool,
) -> Result<()> {
    let mut unchecked: Vec<&str> = Vec::new();
    for (path, model) in paths.iter().zip(models) {
        for (name, trained) in &model.info.region_digests {
            if !model.info.uses_region(name) {
                continue;
            }
            let Some(extracted) = info.region_digests.get(name) else {
                if !unchecked.contains(&name.as_str()) {
                    unchecked.push(name);
                    log::warn!(
                        "feature table {} records no digest of its {name} region BED, so it cannot \
                         be checked against the one the models were trained with",
                        table.display()
                    );
                }
                continue;
            };
            if extracted == trained {
                continue;
            }
            let (severity, advice) =
                mismatch_severity(allow_mismatch, "extract it with the model's BED");
            let message = format!(
                "feature table {} flags the {name} regions with another BED than model {} was \
                 trained with: the table's has {extracted}, the model's {trained}; {advice}",
                table.display(),
                path.display()
            );
            if severity == Severity::Error {
                bail!("{message}");
            }
            log::warn!("{message}");
        }
    }
    Ok(())
}

/// The warnings about a labelled feature table whose reads were not aligned as a model's training
/// reads were (by their recorded provenance), whose alignment is unknown while a model's is
/// known, or whose settings or a model's are incompletely inferred. Models that record none are
/// not checked.
fn table_alignment_warnings(
    table: &Path,
    info: &FeatureTableInfo,
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Vec<String> {
    let (paths, known): (Vec<&PathBuf>, Vec<&AlignmentProvenance>) = paths
        .iter()
        .zip(models)
        .filter_map(|(path, model)| Some((path, model.info.alignment.as_ref()?)))
        .unzip();
    let mut warnings: Vec<String> = paths
        .iter()
        .zip(&known)
        .filter_map(|(path, model)| {
            incomplete_options_reason(model).map(|reason| {
                format!(
                    "model {} records incomplete alignment settings: {reason}; regenerate its \
                     training feature tables with extract --alignment-tags options-inference=explicit \
                     and the intended settings, then retrain from those tables",
                    path.display()
                )
            })
        })
        .collect();
    let Some(extracted) = &info.alignment else {
        warnings.extend(paths.iter().map(|path| {
            format!(
                "model {} records how its training reads were aligned, but feature table {} \
                 records no alignment provenance, so it is not checked",
                path.display(),
                table.display()
            )
        }));
        return warnings;
    };
    if !known.is_empty()
        && let Some(reason) = incomplete_options_reason(extracted)
    {
        warnings.push(format!(
            "feature table {} records incomplete alignment settings: {reason}; extract it again \
             with intended settings declared with --alignment-tags and options-inference=explicit",
            table.display()
        ));
    }
    let compared = merge(&known).ok().flatten();
    let indeterminate: std::collections::BTreeSet<String> = if let Some(merged) = &compared {
        indeterminate_tags(merged, extracted)
    } else {
        known.iter().flat_map(|model| indeterminate_tags(model, extracted)).collect()
    };
    if !indeterminate.is_empty() {
        warnings.push(format!(
            "feature table {} has indeterminate alignment provenance on {}: both the models and table record mixed, which does not establish agreement",
            table.display(),
            indeterminate.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    warnings.extend(compare_with_models(&known, extracted).into_iter().map(
        |(index, differences)| {
            let whose = match (index, paths.as_slice()) {
                (Some(index), _) => format!("model {}'s", paths[index].display()),
                (None, [path]) => format!("model {}'s", path.display()),
                (None, _) => "the models'".to_string(),
            };
            format!(
                "feature table {} is of reads not aligned as {whose} training reads were: {}",
                table.display(),
                describe_differences(&differences, "model", "table")
            )
        },
    ));
    warnings
}

/// Checks that a feature table's features come from the source each model was trained on.
///
/// # Errors
///
/// Returns an error naming the table, the model and both sources if they differ.
fn check_table_feature_source(
    table: &Path,
    info: &FeatureTableInfo,
    models: &[WinnowrModel],
    paths: &[PathBuf],
) -> Result<()> {
    // The table first, then each model by its path.
    let sources = std::iter::once((None, info.feature_source))
        .chain(paths.iter().zip(models).map(|(path, m)| (Some(path), m.info.feature_source)));
    shared_feature_source(
        sources,
        |&(_, source)| source,
        |&(_, measured), &(path, trained)| {
            // The `extract` options that give a table of the model's source.
            let options = match trained {
                FeatureSource::Gvcf => "--feature-source gvcf (with --gvcf if the VCF lacks the \
                                        raw annotations)"
                    .to_string(),
                FeatureSource::Vcf => "--feature-source vcf".to_string(),
                FeatureSource::VcfAlignments(settings) => {
                    format!("--alignments, --reference and --alignment-settings {settings}")
                }
            };
            anyhow::anyhow!(
                "feature table {} holds features from `{measured}`, but model {} was trained on \
                 features from `{trained}`: extract the table again with {options}, or use a \
                 model trained on `{measured}` features",
                table.display(),
                path.expect("only the table has no path").display()
            )
        },
        "The feature table's",
    )?;
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

/// The read settings of the alignments feature source, if the models were trained on it.
///
/// # Errors
///
/// Returns an error, naming a model, if they were and `score` is not given the reads to measure.
fn models_read_annotations(
    feature_source: FeatureSource,
    args: &ScoreArgs,
) -> Result<Option<ReadAnnotationOptions>> {
    let FeatureSource::VcfAlignments(options) = feature_source else { return Ok(None) };
    if args.alignments.is_none() {
        bail!(
            "model {} was trained on features measured from the reads (feature source \
             {feature_source}): pass --alignments and --reference",
            args.models[0].display()
        );
    }
    Ok(Some(options))
}

/// The inputs of the feature table `score` builds: the raw annotations' inputs of the models'
/// feature source (see [`prepare_source`]), `regions`, the repeat-length and pileup evidence,
/// and the depth median of the command line, with no truth labels.
fn score_table_inputs<'a>(
    args: &'a ScoreArgs,
    annotations: AnnotationInputs<'a>,
    regions: &'a [RegionSet],
    strlen: Option<&'a crate::strlen::StrlenTables>,
    pileup: Option<&'a PileupTables>,
) -> TableInputs<'a> {
    TableInputs {
        annotations,
        regions,
        strlen,
        pileup,
        dp_median: args.dp_median,
        ..TableInputs::default()
    }
}

/// The suffixes of the reports `stratify` writes at its `--output` prefix (`<output>.<suffix>`):
/// overall, and by report region.
const STRATIFY_REPORTS: [&str; 6] = [
    "strata.tsv",
    "rates.tsv",
    "calibration.tsv",
    "regions.strata.tsv",
    "regions.rates.tsv",
    "regions.calibration.tsv",
];

/// Runs `winnowr stratify`.
fn stratify(args: &StratifyArgs, threads: NonZeroUsize) -> Result<()> {
    stratify_with(args, threads, || {})
}

/// Stratifies with a callback after consuming the policy BED and before any report is written.
fn stratify_with<F: FnOnce()>(
    args: &StratifyArgs,
    threads: NonZeroUsize,
    before_reports: F,
) -> Result<()> {
    use crate::regions::RegionSet;
    use crate::regions::read_stratification_manifest;
    use crate::strata::{
        CALL_KEY_COLUMNS, CallOutputs, TRUTH_LABEL_COLUMNS, TruthSets, check_region_names,
        labelled_calls_path, stratify,
    };
    args.policy.validate()?;
    check_policy_region_names(&args.regions)?;
    check_output_dir("--output", &args.output)?;
    if let Some(calls) = &args.calls {
        check_output_dir("--calls", calls)?;
    }
    if let Some(dir) = args.labelled_calls.as_deref().filter(|dir| !dir.is_dir()) {
        bail!("the --labelled-calls directory {} does not exist", dir.display());
    }
    let mut named = match &args.stratification {
        Some(path) => read_stratification_manifest(path)?,
        None => Vec::new(),
    };
    named.extend(args.report_region.iter().cloned());
    // No output may be an input: checked before any input is read but the manifest, and for the
    // labelled calls (named by the sample) and the not-difficult regions BED once they are known.
    let report_path = |suffix: &str| PathBuf::from(format!("{}.{suffix}", args.output.display()));
    let report_count = if named.is_empty() { 3 } else { STRATIFY_REPORTS.len() };
    let mut outputs: Vec<PathBuf> =
        STRATIFY_REPORTS[..report_count].iter().map(|s| report_path(s)).collect();
    outputs.extend(args.calls.iter().cloned());
    let inputs: Vec<(&str, &Path)> = args
        .models
        .iter()
        .map(|model| ("model", model.as_path()))
        .chain(args.tables.iter().map(|table| ("feature table", table.as_path())))
        .chain(args.truth_beds.iter().map(|bed| ("truth BED", bed.as_path())))
        .chain(args.happy.iter().map(|happy| ("hap.py VCF", happy.as_path())))
        .chain(args.stratification.iter().map(|path| ("stratification manifest", path.as_path())))
        .chain(named.iter().map(|named| ("report region BED", named.path.as_path())))
        .collect();
    check_no_output_is_input(&outputs, inputs.iter().copied())?;
    let files = ModelFile::resolve_all(&args.models);
    let models = load_models(&files, threads)?;
    check_models_region_digests(&models, &files)?;
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
    // The contigs the calls were extracted from, before any --contigs: the region BEDs are checked
    // against them, to catch BEDs whose contig names differ from the calls', which a BED that
    // merely lacks a selected contig is not.
    let extracted = table.info().contigs.clone();
    let (table, second) = if args.contigs.is_empty() {
        (table, second)
    } else {
        (
            restrict_to_contigs(&table, &args.happy[0], &args.contigs)?,
            restrict_to_contigs(&second, &args.happy[1], &args.contigs)?,
        )
    };
    let info = table.info();
    if second.info().sample != info.sample {
        bail!("--tables are of different samples: {} and {}", info.sample, second.info().sample);
    }
    if let Some(dir) = &args.labelled_calls {
        let labelled: Vec<PathBuf> = names
            .iter()
            .flat_map(|truth| {
                VariantType::ALL
                    .iter()
                    .filter(|vt| models.iter().any(|model| model.info.variant_type == **vt))
                    .map(|vt| labelled_calls_path(dir, &info.sample, truth, *vt))
            })
            .collect();
        check_no_output_is_input(&labelled, inputs.iter().copied())?;
        outputs.extend(labelled);
    }
    check_labelled_table(&args.tables[0], info, &models, &args.models, args.allow_region_mismatch)?;
    // The identity of every regular BED read (truth and report BEDs here, the policy's below),
    // retained so no report writer truncates one, even if its path is redirected meanwhile.
    let mut consumed_beds: Vec<crate::io::FileIdentity> = Vec::new();
    let mut read_bed = |name: &str, path: &Path| -> Result<RegionSet> {
        let (regions, identity) = RegionSet::from_bed_with_identity(name, path)?;
        consumed_beds.extend(identity);
        Ok(regions)
    };
    let beds =
        [read_bed(&names[0], &args.truth_beds[0])?, read_bed(&names[1], &args.truth_beds[1])?];
    for bed in &beds {
        bed.check_contigs(&extracted)?;
    }
    check_region_names(&named.iter().map(|named| named.name.as_str()).collect::<Vec<_>>())?;
    let regions =
        named.iter().map(|named| read_bed(&named.name, &named.path)).collect::<Result<Vec<_>>>()?;
    for region in &regions {
        if let Err(e) = region.check_contigs(&extracted) {
            log::warn!("region {}: {e:#}", region.name());
        }
    }
    let truths = TruthSets {
        names: &names,
        beds: [&beds[0], &beds[1]],
        happy: [&args.happy[0], &args.happy[1]],
    };
    warn_policy_only_regions(&args.regions);
    let infos: Vec<&ModelInfo> = models.iter().map(|model| &model.info).collect();
    let indel_genotype = has_indel_genotype(&infos);
    let reads_not_difficult = args.policy.reads_not_difficult(indel_genotype);
    let (not_difficult, policy_beds) = policy_not_difficult(
        &infos,
        &files,
        &args.regions,
        reads_not_difficult,
        args.allow_region_mismatch,
        &outputs,
    )?;
    let mut protected_beds = consumed_beds;
    protected_beds.extend(policy_beds);
    if reads_not_difficult
        && not_difficult.is_none()
        && named.iter().any(|named| named.name == NOT_DIFFICULT_REGIONS)
    {
        log::warn!(
            "{NOT_DIFFICULT_REGIONS} is reported within but not given to the policy: pass \
             --regions {NOT_DIFFICULT_REGIONS}=PATH, or put it beside the models as {}, to decide \
             calls as score does with it",
            bundled_bed_files(NOT_DIFFICULT_REGIONS)
        );
    }
    let policy = args.policy.policy(args.mode, &extracted, not_difficult, indel_genotype)?;
    before_reports();
    let outputs = CallOutputs {
        calls: args.calls.as_deref(),
        labelled_dir: args.labelled_calls.as_deref(),
        protected_beds: &protected_beds,
    };
    let (report, by_region) =
        stratify(&table, &second, &models, &policy, truths, &regions, outputs)?;
    for line in report.summary_lines() {
        info!("{line}");
    }
    for line in by_region.summary_lines() {
        info!("{line}");
    }
    let path = |suffix: &str| PathBuf::from(format!("{}.{suffix}", args.output.display()));
    // The files of STRATIFY_REPORTS.
    report.write_outcomes_protecting(&path("strata.tsv"), &protected_beds)?;
    report.write_rates_protecting(&path("rates.tsv"), &protected_beds)?;
    report.write_calibration_protecting(&path("calibration.tsv"), &protected_beds)?;
    info!("Wrote {}.{{strata,rates,calibration}}.tsv", args.output.display());
    if !by_region.is_empty() {
        by_region.write_outcomes_protecting(&path("regions.strata.tsv"), &protected_beds)?;
        by_region.write_rates_protecting(&path("regions.rates.tsv"), &protected_beds)?;
        by_region
            .write_calibration_protecting(&path("regions.calibration.tsv"), &protected_beds)?;
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

/// The index candidates for an input; variant files use the reader's format detection.
#[derive(Clone, Copy)]
enum InputIndexes {
    None,
    Vcf,
    Alignments,
    Reference,
}

impl InputIndexes {
    fn paths(self, path: &Path) -> Vec<PathBuf> {
        match self {
            Self::None => Vec::new(),
            Self::Vcf => crate::vcf::index_candidates(path),
            Self::Alignments => crate::read_pass::alignment_index_paths(path),
            Self::Reference => crate::vcf::appended_index_paths(path, &["fai", "gzi"]),
        }
    }
}

/// Checks that a command's `output` is none of its input files: `inputs` (each with what it is,
/// if given, and the kind of indexes read with it, which are checked too), the region
/// BEDs, and the strlen and pileup tables. Checked before any input is read, so that a run that
/// would overwrite one fails at once.
///
/// # Errors
///
/// Returns an error naming the input the output is.
fn check_output_not_inputs(
    output: &Path,
    inputs: &[(&str, Option<&Path>, InputIndexes)],
    regions: &[NamedBed],
    strlen: &[PathBuf],
    pileup: &[PathBuf],
) -> Result<()> {
    crate::scoring::check_output_not_input(
        output,
        inputs.iter().filter_map(|&(what, path, _)| Some((what, path?))),
    )?;
    let resolved = resolve_output_inputs(inputs);
    check_output_not_resolved_inputs(output, &resolved, regions, strlen, pileup)
}

/// An input and its index candidates, retained without reopening it during a later alias audit.
struct ResolvedInput<'a> {
    what: &'a str,
    path: &'a Path,
    kind: InputIndexes,
    indexes: Vec<PathBuf>,
    /// When scoring measures reads, the searches validated by those readers supersede preflight.
    observed_indexes: Option<crate::read_pass::ReadIndexObservation>,
}

/// Retains the preflight index search; a later measurement supplies its own observed searches.
fn resolve_output_inputs<'a>(
    inputs: &[(&'a str, Option<&'a Path>, InputIndexes)],
) -> Vec<ResolvedInput<'a>> {
    inputs
        .iter()
        .filter_map(|&(what, path, kind)| {
            let path = path?;
            Some(ResolvedInput {
                what,
                path,
                kind,
                indexes: kind.paths(path),
                observed_indexes: None,
            })
        })
        .collect()
}

/// Audits current aliases using the retained index candidates, without reading input contents.
fn check_output_not_resolved_inputs(
    output: &Path,
    inputs: &[ResolvedInput<'_>],
    regions: &[NamedBed],
    strlen: &[PathBuf],
    pileup: &[PathBuf],
) -> Result<()> {
    crate::scoring::check_output_not_input(
        output,
        inputs.iter().map(|input| (input.what, input.path)),
    )?;
    for input in inputs {
        let what = format!("{} index", input.what);
        if let Some(observed) = &input.observed_indexes {
            for search in observed.searches() {
                crate::scoring::check_output_not_indexes(output, &what, &search)?;
            }
        } else if matches!(input.kind, InputIndexes::Vcf | InputIndexes::Alignments) {
            crate::scoring::check_output_not_indexes(output, &what, &input.indexes)?;
        } else {
            // FASTA sidecars are independent inputs, not alternatives in an index search.
            crate::scoring::check_output_not_input(
                output,
                input.indexes.iter().map(|path| (what.as_str(), path.as_path())),
            )?;
        }
    }
    let beds = regions.iter().map(|bed| ("region BED", bed.path.as_path()));
    let tables = strlen
        .iter()
        .map(|table| ("strlen table", table.as_path()))
        .chain(pileup.iter().map(|table| ("pileup table", table.as_path())));
    crate::scoring::check_output_not_input(output, beds.chain(tables))
}

/// Effective provenance observed per regular evidence file during scoring preflight, by the
/// file's path as given (not its display text, which can be lossy).
type AlignmentSnapshots = Vec<(PathBuf, Option<AlignmentProvenance>)>;

/// The preflight snapshot of `path`, if it was observed as a regular file.
fn snapshot_of<'a>(
    checked: &'a [(PathBuf, Option<AlignmentProvenance>)],
    path: &Path,
) -> Option<&'a Option<AlignmentProvenance>> {
    checked.iter().find(|(checked, _)| checked == path).map(|(_, alignment)| alignment)
}

/// Enforces observed snapshots exactly. A source without one (a stream) is not compared here:
/// the final resolution of the evidence actually loaded (with the retained provenance-only
/// inputs) finds any disagreement, rather than a comparison with stale preflight provenance.
fn check_loaded_alignment(
    path: &Path,
    alignment: Option<&AlignmentProvenance>,
    checked: &[(PathBuf, Option<AlignmentProvenance>)],
    strict: bool,
) -> Result<()> {
    let Some(expected) = snapshot_of(checked, path) else { return Ok(()) };
    if alignment != expected.as_ref() {
        let message = format!(
            "{} has changed alignment provenance since scoring preflight; the loaded evidence is not the evidence checked against the models",
            path.display()
        );
        if strict {
            bail!("{message}");
        }
        log::warn!("{message}");
    }
    Ok(())
}

/// Whether a required evidence file was skipped by scoring preflight (it was not a regular file
/// then), so its one row read may still establish the reads' provenance. Decided from the
/// recorded observation, not the path's current type, which a replacement can change.
fn required_alignment_pending(
    required: &[(&[PathBuf], bool)],
    checked: &[(PathBuf, Option<AlignmentProvenance>)],
) -> bool {
    required.iter().any(|&(paths, required)| {
        required && paths.iter().any(|path| snapshot_of(checked, path).is_none())
    })
}

/// Whether `problem` compares the reads' alignment provenance (rather than the models' alone):
/// scoring reports these, when warnings, from the comparison of the evidence it loaded.
fn reads_alignment_problem(problem: &Problem) -> bool {
    match problem.check {
        Check::AlignmentProvenanceMismatch
        | Check::AlignmentProvenanceNotChecked
        | Check::AlignmentProvenanceIndeterminate => true,
        Check::AlignmentProvenanceIncomplete => problem.model.is_none(),
        _ => false,
    }
}

/// Runs `winnowr score`.
fn score(args: &ScoreArgs, threads: NonZeroUsize) -> Result<()> {
    score_with(args, threads, || {})
}

/// Scores with a callback between consuming the inputs and auditing output aliases.
fn score_with<F: FnOnce()>(
    args: &ScoreArgs,
    threads: NonZeroUsize,
    before_output_check: F,
) -> Result<()> {
    let mut timer = PhaseTimer::start();
    args.policy.validate()?;
    check_policy_region_names(&args.regions)?;
    check_output_dir("--output", &args.output)?;
    // Checked again (for the VCF) as it is written, but now too, before any pass over the reads.
    // Optional inputs are checked once the models say whether they are read.
    let inputs = [("VCF", Some(args.vcf.as_path()), InputIndexes::Vcf)];
    // The index written beside the output must be none of them either.
    let outputs: Vec<PathBuf> = std::iter::once(args.output.clone())
        .chain(crate::scoring::scored_vcf_index(&args.output))
        .collect();
    for output in &outputs {
        check_output_not_inputs(output, &inputs, &[], &[], &[])?;
        let models = args.models.iter().map(|model| ("model", model.as_path()));
        crate::scoring::check_output_not_input(output, models)?;
    }
    if let Some(output_index) = crate::scoring::scored_vcf_index(&args.output) {
        crate::scoring::check_output_not_input(
            &output_index,
            [("scored VCF", args.output.as_path())],
        )?;
    }
    // Each model file is resolved and read once: its information is checked before its booster is
    // parsed, so a set that cannot be scored never pays for parsing the boosters, and its region
    // BEDs are looked for beside the file read, even if a staged symlink is retargeted meanwhile.
    let files = ModelFile::resolve_all(&args.models);
    let resolved: Vec<PathBuf> = files.iter().map(|file| file.resolved.clone()).collect();
    let given_dirs: Vec<PathBuf> = files.iter().map(|file| file.given_dir.clone()).collect();
    let (texts, model_metadata): (Vec<String>, Vec<std::fs::Metadata>) = files
        .iter()
        .map(ModelFile::read_text_with_metadata)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    let load_error = |path: &Path| format!("failed to load model {}", path.display());
    let parts: Vec<(ModelInfo, &str)> = args
        .models
        .iter()
        .zip(&texts)
        .map(|(path, text)| ModelInfo::from_model_json(text).with_context(|| load_error(path)))
        .collect::<Result<_>>()?;
    timer.lap("reading the models");
    let summary = VcfSummary::read(&args.vcf)?;
    let infos: Vec<&ModelInfo> = parts.iter().map(|(info, _)| info).collect();
    // Protect feature inputs before provenance inspection can open the same paths.
    let needs_strlen = models_using::<StrlenOptions>(&infos, &args.models);
    let needs_pileup = models_using::<PileupOptions>(&infos, &args.models);
    let uses_reads =
        infos.iter().any(|info| matches!(info.feature_source, FeatureSource::VcfAlignments(_)))
            || !needs_strlen.is_empty()
            || !needs_pileup.is_empty();
    let inputs = [
        ("alignments", args.alignments.as_deref().filter(|_| uses_reads), InputIndexes::Alignments),
        ("reference", args.reference.as_deref().filter(|_| uses_reads), InputIndexes::Reference),
    ];
    for output in &outputs {
        crate::scoring::check_output_not_input(
            output,
            inputs.iter().filter_map(|&(what, path, _)| Some((what, path?))),
        )?;
    }
    let mut protected_reads = resolve_output_inputs(&inputs);
    for output in &outputs {
        check_output_not_resolved_inputs(
            output,
            &protected_reads,
            &[],
            if needs_strlen.is_empty() { &[] } else { &args.strlen },
            if needs_pileup.is_empty() { &[] } else { &args.pileup },
        )?;
    }
    let observed_indexes = crate::read_pass::ReadIndexObservation::default();
    for input in &mut protected_reads {
        if matches!(input.kind, InputIndexes::Alignments) {
            input.observed_indexes = Some(observed_indexes.clone());
        }
    }
    let checks_provenance = infos.iter().any(|info| info.alignment.is_some());
    let tags = args.provenance.alignment_tags.tags()?;
    let evidence_tags = if checks_provenance { tags.as_ref() } else { Some(&AlignmentTags::None) };
    let required =
        RequiredTables { strlen: !needs_strlen.is_empty(), pileup: !needs_pileup.is_empty() };
    let (checked_reads, checked_tables) = if checks_provenance {
        score_reads_alignment(args, required)?
    } else {
        // Required rows still load, but legacy models need no provenance parsing or agreement.
        (ReadsAlignment::default(), Vec::new())
    };
    let pending_alignment = checks_provenance
        && !matches!(evidence_tags, Some(AlignmentTags::None))
        && required_alignment_pending(
            &[(args.strlen.as_slice(), required.strlen), (args.pileup.as_slice(), required.pileup)],
            &checked_tables,
        );
    let reads = &checked_reads;
    timer.lap("reading the VCF summary and the reads' alignment provenance");
    // Every problem of the model set is reported at once (see `winnowr check-models`). The
    // region BEDs are read later, those a model uses only.
    let set_inputs = SetInputs {
        regions: &args.regions,
        contigs: vec![(format!("VCF {}", args.vcf.display()), &summary.contig_lengths)],
        reads: ReadsCheck {
            alignment: reads.provenance.as_ref(),
            unknown: reads.unknown,
            strict: args.provenance.strict_provenance,
        },
        ..SetInputs::default()
    };
    // The region plan outlives the model files' text (and so `parts`): its BEDs are read after
    // the evidence. The information alone is small.
    let set_infos: Vec<ModelInfo> = parts.iter().map(|(info, _)| info.clone()).collect();
    let infos: Vec<&ModelInfo> = set_infos.iter().collect();
    let set = ModelSet {
        paths: &args.models,
        resolved: &resolved,
        given_dirs: &given_dirs,
        infos: &infos,
    };
    let mut problems = check_model_set(&set, &set_inputs);
    if pending_alignment {
        // A required stream can establish provenance in its one row read. Other checks remain early.
        problems.retain(|problem| problem.check != Check::AlignmentProvenanceNotChecked);
    }
    // The region BEDs the models' features and the policy use, given or beside the models: only
    // those are read, so an unused one cannot fail the run. One a model uses that is neither is a
    // problem of the set.
    let indel_genotype = has_indel_genotype(&infos);
    let reads_not_difficult = args.policy.reads_not_difficult(indel_genotype);
    let run = RunRegions::Features { policy: reads_not_difficult };
    problems.extend(region_digests_differ(&set, reads_not_difficult));
    let (plan, missing) = RegionPlan::new(&set, &args.regions, run);
    problems.extend(missing);
    problems.sort_by_key(|problem| problem.check);
    // Keep BED planning and all errors early, but report reads-side warnings from loaded evidence.
    for problem in problems.iter().filter(|problem| {
        problem.severity == Severity::Warning && !reads_alignment_problem(problem)
    }) {
        log::warn!("{}", problem.message);
    }
    if let Some(message) = error_message(&problems) {
        bail!("{message}");
    }
    let protected_beds: Vec<PathBuf> = plan.paths().map(Path::to_path_buf).collect();
    let prospective_beds: Vec<PathBuf> = plan.prospective_paths().map(Path::to_path_buf).collect();
    timer.lap("checking the model set");
    for output in &outputs {
        crate::scoring::check_output_not_input(
            output,
            plan.paths().map(|path| ("region BED", path)),
        )?;
        crate::scoring::check_output_not_prospective_inputs(
            output,
            prospective_beds.iter().map(|path| ("region BED", path.as_path())),
        )?;
    }
    let mut models: Vec<WinnowrModel> = args
        .models
        .iter()
        .zip(parts)
        .map(|(path, (info, booster))| {
            WinnowrModel::from_parts(info, booster, threads).with_context(|| load_error(path))
        })
        .collect::<Result<_>>()?;
    // The model files' text (each with its serialized booster) is not needed past here.
    drop(texts);
    timer.lap("loading the boosters");
    let infos: Vec<&ModelInfo> = models.iter().map(|model| &model.info).collect();
    let feature_source = models_feature_source(&infos).unwrap_or_default();
    if !feature_source.is_default() {
        info!("The models' features come from feature source {feature_source}");
    }
    check_unread_gvcf(feature_source, args.gvcf.as_deref(), UnreadGvcf::Warn)?;
    let read_annotations = models_read_annotations(feature_source, args)?;
    check_supplied::<StrlenOptions>(&needs_strlen, &args.strlen, args.alignments.is_some())?;
    check_supplied::<PileupOptions>(&needs_pileup, &args.pileup, args.alignments.is_some())?;
    if !uses_reads && args.alignments.is_some() {
        log::warn!(
            "no model uses repeat-length or pileup features; --alignments is ignored for features (it may still be read for alignment provenance)"
        );
    }
    // The settings the models' evidence must be measured with.
    let strlen_options = models_options::<StrlenOptions>(&infos, &args.models)?;
    let expected_alignment =
        alignment_expectation(&checked_reads, tags.as_ref(), args.provenance.strict_provenance);
    let check = |path: &Path, actual: Option<&AlignmentProvenance>| {
        check_loaded_alignment(path, actual, &checked_tables, args.provenance.strict_provenance)
    };
    let strlen = read_for_models::<StrlenOptions>(
        &args.strlen,
        &models,
        &args.models,
        evidence_tags,
        check,
    )?;
    let pileup_options = models_options::<PileupOptions>(&infos, &args.models)?;
    let pileup = read_for_models::<PileupOptions>(
        &args.pileup,
        &models,
        &args.models,
        evidence_tags,
        check,
    )?;
    timer.lap("reading the repeat-length and pileup tables");
    // The region BEDs are read after the evidence, as one producer may feed an evidence stream
    // before a BED stream: reading the BEDs first would wait on it forever.
    // The features' regions, and the not-difficult regions when the policy reads them: by
    // themselves, or (of a model trained with them as a feature) as the features' entry, taken
    // once the features are built.
    let mut regions: Vec<RegionSet> = Vec::new();
    let mut not_difficult: Option<RegionSet> = None;
    let mut not_difficult_feature: Option<usize> = None;
    let mut bed_identities = Vec::new();
    for (region, intervals) in checked_regions(plan.read(args.allow_region_mismatch))? {
        bed_identities.extend(region.input_identities);
        if !infos.iter().any(|info| info.uses_region(&region.name)) {
            not_difficult = Some(intervals);
            continue;
        }
        if reads_not_difficult && region.name == NOT_DIFFICULT_REGIONS {
            not_difficult_feature = Some(regions.len());
        }
        regions.push(intervals);
    }
    for output in &outputs {
        check_output_not_bed_identities(output, &bed_identities)?;
    }
    // Withholding actions needs the not-difficult regions: fail now, not after a pass over the
    // reads. Without them the outside indel threshold alone is not applied, with a warning.
    args.policy
        .require_not_difficult(not_difficult.is_some() || not_difficult_feature.is_some())?;
    timer.lap("reading the region BEDs");
    let scored_reads = if checks_provenance {
        loaded_score_alignment(
            args,
            &checked_reads,
            &checked_tables,
            strlen.as_ref(),
            pileup.as_ref(),
        )?
    } else {
        ReadsAlignment::default()
    };
    let alignment_problems = alignment_provenance_problems(
        &ModelSet {
            paths: &args.models,
            resolved: &resolved,
            given_dirs: &given_dirs,
            infos: &infos,
        },
        ReadsCheck {
            alignment: scored_reads.provenance.as_ref(),
            unknown: scored_reads.unknown,
            strict: args.provenance.strict_provenance,
        },
    );
    for problem in alignment_problems.iter().filter(|problem| {
        problem.severity == Severity::Warning
            && (reads_alignment_problem(problem) || !problems.contains(problem))
    }) {
        log::warn!("{}", problem.message);
    }
    if let Some(message) = error_message(&alignment_problems) {
        bail!("{message}");
    }
    log_alignment_check(
        &infos,
        scored_reads.provenance.as_ref(),
        &alignment_problems,
        uses_reads && args.alignments.is_some(),
    );
    timer.lap("checking the reads' alignment");
    // Evidence measured from the reads while scoring, as the models need it (--alignments
    // conflicts with --strlen and --pileup, so nothing is both read and measured).
    let measure = match (&args.alignments, &args.reference) {
        (Some(alignments), Some(reference)) => {
            // Some exactly when a model uses the evidence. With the alignments feature source,
            // the pileup is counted with the read annotations, before the table is built.
            let strlen = strlen_options;
            let pileup = pileup_options.filter(|_| read_annotations.is_none());
            if let Some(options) = strlen {
                info!(
                    "Measuring the repeat-length evidence with the models' read settings {options}"
                );
            }
            if let Some(options) = pileup {
                info!("Counting the pileup evidence with the models' settings {options}");
            }
            (strlen.is_some() || pileup.is_some()).then_some((
                ReadInputs {
                    observed_indexes: Some(observed_indexes.clone()),
                    expected_alignment,
                    ..ReadInputs::new(&args.vcf, alignments, reference)
                },
                strlen,
                pileup,
            ))
        }
        _ => None,
    };
    // The models' contigs are the same, perhaps in different orders; the first model's order is
    // the order the calls are extracted in.
    let contigs = models[0].info.contigs.clone();
    let sample = summary.sample_name(&args.vcf);
    // The alignments feature source's annotations are measured before the table is built, as
    // dp_norm needs every call's depth; so is the pileup evidence, in the same pass when it
    // reads alleles alike.
    let reads = read_inputs(&args.vcf, args.alignments.as_deref(), args.reference.as_deref()).map(
        |mut inputs| {
            inputs.expected_alignment = expected_alignment;
            inputs.observed_indexes = Some(observed_indexes.clone());
            inputs
        },
    );
    let truth = TruthInputs::default();
    let checks = TableChecks {
        vcf: &args.vcf,
        contigs: &contigs,
        regions: &regions,
        truth: &truth,
        dp_median: args.dp_median,
    };
    let pileup_with_reads = pileup_options.filter(|_| read_annotations.is_some());
    let source =
        prepare_source(feature_source, args.gvcf.as_deref(), reads, pileup_with_reads, &checks)?;
    timer.lap_total("preparing the source");
    // --pileup conflicts with --alignments, so the pileup is either read or measured.
    let pileup = pileup.as_ref().or(source.pileup());
    let mut inputs =
        score_table_inputs(args, source.annotations(), &regions, strlen.as_ref(), pileup);
    inputs.outputs = &outputs;
    let table = build_feature_table(&sample, &args.vcf, &contigs, &summary, &inputs)?;
    timer.lap_total("building the feature table");
    let not_difficult =
        not_difficult.or_else(|| not_difficult_feature.map(|index| regions.swap_remove(index)));
    let policy = args.policy.policy(args.mode, &contigs, not_difficult, indel_genotype)?;
    timer.lap("building the policy");
    let decisions = match measure {
        None => decide_calls(&table, &models, &policy)?,
        Some((measure, strlen, pileup)) => {
            let measuring = Measuring { inputs: &measure, strlen, pileup, contigs: &contigs };
            measure_and_decide(table, &mut models, &policy, &measuring)?
        }
    };
    timer.lap_total("scoring");
    before_output_check();
    // Recheck the retained provenance inputs just before writing: output symlinks may have changed
    // since preflight. Skipped streams were never inputs and remain excluded.
    let output_index = crate::scoring::scored_vcf_index(&args.output);
    for output in std::iter::once(args.output.as_path()).chain(output_index.as_deref()) {
        check_output_not_resolved_inputs(
            output,
            &protected_reads,
            &[],
            if needs_strlen.is_empty() { &[] } else { &args.strlen },
            if needs_pileup.is_empty() { &[] } else { &args.pileup },
        )?;
        let header = args.alignments.as_deref().filter(|_| checked_reads.header.is_some());
        // The files the models resolved to, as well as the paths given: a staged symlink may be
        // retargeted after its model was read.
        let consumed = args
            .models
            .iter()
            .chain(&resolved)
            .map(|path| ("model", path.as_path()))
            .chain(files.iter().map(|file| ("model", file.resolved.as_path())))
            .chain(header.map(|path| ("alignment header", path)))
            .chain(protected_beds.iter().map(|path| ("region BED", path.as_path())))
            .chain(checked_tables.iter().map(|(path, _)| ("provenance table", path.as_path())));
        crate::scoring::check_output_not_input(output, consumed)?;
        for metadata in &model_metadata {
            crate::scoring::check_output_not_loaded_input(output, "model", metadata)?;
        }
        crate::scoring::check_output_not_prospective_inputs(
            output,
            prospective_beds.iter().map(|path| ("region BED", path.as_path())),
        )?;
        check_output_not_bed_identities(output, &bed_identities)?;
    }
    write_scored_vcf(&args.vcf, &args.output, decisions, threads, &bed_identities)?;
    timer.lap_total("writing the VCF");
    Ok(())
}

/// Logs the problems that are warnings, and fails naming every one that is an error.
///
/// # Errors
///
/// Returns an error naming every problem that is one (see [`error_message`]).
fn log_and_fail(problems: &[Problem]) -> Result<()> {
    for problem in problems.iter().filter(|problem| problem.severity == Severity::Warning) {
        log::warn!("{}", problem.message);
    }
    match error_message(problems) {
        Some(message) => bail!("{message}"),
        None => Ok(()),
    }
}

/// The region BEDs a run reads, read and checked (see [`RegionPlan::read`]), with their problems:
/// warnings are logged, and so is each BED used.
///
/// # Errors
///
/// Returns an error naming every problem that is one.
fn checked_regions(
    (resolved, problems): (Vec<(ResolvedRegion, RegionSet)>, Vec<Problem>),
) -> Result<Vec<(ResolvedRegion, RegionSet)>> {
    log_and_fail(&problems)?;
    for (region, _) in &resolved {
        let verified = match region.verified() {
            Some(true) => "the models' own",
            Some(false) => "not the models'",
            None => "unchecked: no model records its digest",
        };
        info!(
            "Region BED {} ({}, {}): {}; {verified}",
            region.name,
            region.origin().as_str(),
            region.path.display(),
            region.digest
        );
    }
    Ok(resolved)
}

/// Rejects repeated policy BED names before a one-shot region input can be consumed.
fn check_policy_region_names(given: &[NamedBed]) -> Result<()> {
    if given.iter().filter(|bed| bed.name == NOT_DIFFICULT_REGIONS).count() > 1 {
        bail!("--regions gives {NOT_DIFFICULT_REGIONS} more than once");
    }
    Ok(())
}

/// The not-difficult regions `evaluate` and `stratify` give the policy when it reads them
/// (`policy`): the `--regions` BED (`given`), else the one beside the models (`infos`, read from
/// `files`), checked against the digests the models record (see
/// [`crate::model_set::RegionPlan::read`]). Each of `outputs` is checked not to be the BED
/// before it is read. Returns the intervals together with every consumed regular BED's identity,
/// retained independently of pathname changes for the report writers' same-handle checks.
///
/// # Errors
///
/// Returns an error if a policy BED name is repeated, an output is the BED, or the BED cannot be
/// read or is not the models' (unless `allow_mismatch`).
fn policy_not_difficult(
    infos: &[&ModelInfo],
    files: &[ModelFile],
    given: &[NamedBed],
    policy: bool,
    allow_mismatch: bool,
    outputs: &[PathBuf],
) -> Result<(Option<RegionSet>, Vec<crate::io::FileIdentity>)> {
    check_policy_region_names(given)?;
    let paths: Vec<PathBuf> = files.iter().map(|file| file.path.clone()).collect();
    let resolved: Vec<PathBuf> = files.iter().map(|file| file.resolved.clone()).collect();
    let given_dirs: Vec<PathBuf> = files.iter().map(|file| file.given_dir.clone()).collect();
    let set = ModelSet { paths: &paths, resolved: &resolved, given_dirs: &given_dirs, infos };
    let (plan, mut problems) = RegionPlan::new(&set, given, RunRegions::Policy(policy));
    // Models recording different policy digests are rejected before the BED is opened: no BED
    // can be both, and an explicit stream would block until its producer writes.
    log_and_fail(&plan.digest_conflicts())?;
    check_no_output_is_input(outputs, plan.paths().map(|path| ("region BED", path)))?;
    for output in outputs {
        crate::scoring::check_output_not_prospective_inputs(
            output,
            plan.prospective_paths().map(|path| ("region BED", path)),
        )?;
    }
    let (regions, read_problems) = plan.read(allow_mismatch);
    problems.extend(read_problems);
    problems.sort_by_key(|problem| problem.check);
    let regions = checked_regions((regions, problems))?;
    for (region, _) in &regions {
        for output in outputs {
            check_output_not_bed_identities(output, &region.input_identities)?;
        }
    }
    let protected_beds =
        regions.iter().flat_map(|(region, _)| region.input_identities.iter().copied()).collect();
    Ok((regions.into_iter().next().map(|(_, regions)| regions), protected_beds))
}

/// Protects regular BEDs by the identity of the opened file whose intervals were read, even after
/// a supplied symlink is retargeted or the file is renamed. Current input paths are checked too.
fn check_output_not_bed_identities(
    output: &Path,
    identities: &[crate::io::FileIdentity],
) -> Result<()> {
    if identities.iter().any(|identity| identity.matches_path(output)) {
        bail!("the output {} is the input region BED; write to another path", output.display());
    }
    Ok(())
}

/// Checks that none of `outputs` is one of `inputs`, each with what it is (see
/// [`crate::scoring::check_output_not_input`]): checked before anything is written, so that a run
/// that would overwrite an input fails without destroying it.
///
/// # Errors
///
/// Returns an error naming the first output that is an input, and the input it is.
fn check_no_output_is_input<'a, I: IntoIterator<Item = (&'a str, &'a Path)>>(
    outputs: &[PathBuf],
    inputs: I,
) -> Result<()> {
    let inputs: Vec<(&str, &Path)> = inputs.into_iter().collect();
    for output in outputs {
        crate::scoring::check_output_not_input(output, inputs.iter().copied())?;
    }
    Ok(())
}

/// How the reads `score` scores were aligned (see [`reads_alignment`]): from the header of
/// --alignments and what the --strlen and --pileup tables record (read before the tables are, see
/// [`crate::evidence::read_alignment`]). Missing-source warnings wait for the final resolution of
/// loaded rows and retained provenance-only inputs.
///
/// The snapshots of `required` tables, whose rows are read later with their provenance, may
/// disagree without `--strict-provenance`: their conflict is left to the merge of the tables as
/// loaded (see [`loaded_score_alignment`]).
///
/// # Errors
///
/// Returns an error if the tags are malformed, the header or a table cannot be read, or the
/// sources record different provenance (other than required tables' snapshots, unless strict).
fn score_reads_alignment(
    args: &ScoreArgs,
    required: RequiredTables,
) -> Result<(ReadsAlignment, AlignmentSnapshots)> {
    score_reads_alignment_with(args, required, crate::evidence::observe_alignment_with_tags)
}

/// Whether the `--strlen` and the `--pileup` tables are required (read for their rows): a
/// required table's provenance is read again with its rows.
#[derive(Debug, Clone, Copy, Default)]
struct RequiredTables {
    /// Whether the `--strlen` tables are.
    strlen: bool,
    /// Whether the `--pileup` tables are.
    pileup: bool,
}

/// Scoring preflight with a supplied observer, allowing tests to replace a path after the initial
/// classification but before the same-handle observation.
fn score_reads_alignment_with<F>(
    args: &ScoreArgs,
    required: RequiredTables,
    mut observe: F,
) -> Result<(ReadsAlignment, AlignmentSnapshots)>
where
    F: FnMut(&Path, Option<&AlignmentTags>) -> Result<crate::evidence::AlignmentObservation>,
{
    let tags = args.provenance.alignment_tags.tags()?;
    if matches!(tags, Some(AlignmentTags::None)) {
        return Ok((reads_alignment(tags.as_ref(), None, Vec::new())?, Vec::new()));
    }
    // Provenance can consume inputs no model uses for features. Header-only inspection reads
    // neither alignment indexes nor the reference, and skips non-regular evidence tables.
    let tables = args
        .strlen
        .iter()
        .map(|path| ("strlen table", path.as_path()))
        .chain(args.pileup.iter().map(|path| ("pileup table", path.as_path())))
        .filter(|(_, path)| crate::io::is_regular_file(path));
    let inputs =
        args.alignments.as_deref().map(|path| ("alignments", path)).into_iter().chain(tables);
    let index = crate::scoring::scored_vcf_index(&args.output);
    for output in std::iter::once(args.output.as_path()).chain(index.as_deref()) {
        crate::scoring::check_output_not_input(output, inputs.clone())?;
    }
    let mut checked = Vec::new();
    // Each group's provenance (individual shards if they conflict), and whether it is required.
    let mut groups: Vec<(String, Option<AlignmentProvenance>, bool)> = Vec::new();
    for (option, paths, required) in
        [("--strlen", &args.strlen, required.strlen), ("--pileup", &args.pileup, required.pileup)]
    {
        let snapshot_start = checked.len();
        let mut sources: Vec<(String, LazySource<'_>)> = Vec::new();
        for path in paths {
            let observation = observe(path, tags.as_ref())?;
            let name = path.display().to_string();
            let alignment = match observation {
                crate::evidence::AlignmentObservation::Regular(alignment) => {
                    // Classification can race publication of a regular file. Protect every input
                    // actually observed, including provenance-only evidence, before retaining it.
                    for output in std::iter::once(args.output.as_path()).chain(index.as_deref()) {
                        crate::scoring::check_output_not_input(output, [(option, path.as_path())])?;
                    }
                    checked.push((path.clone(), alignment.clone()));
                    alignment
                }
                crate::evidence::AlignmentObservation::Skipped => continue,
            };
            sources.push((name, Box::new(move || Ok(alignment))));
        }
        if !sources.is_empty() {
            // Rows can acquire provenance before loading; missing-source warnings belong to the
            // final merge, but conflicts between known sources must still fail preflight, except
            // between required tables without --strict-provenance: their rows' declarations
            // replace these snapshots (e.g. of a shard replaced between two observations), and
            // the merge of the tables loaded rejects shards that still disagree.
            let alignment =
                match crate::alignment_provenance::resolve_reads_quiet(None, None, sources) {
                    Ok(resolved) => resolved.provenance,
                    Err(error) if required && !args.provenance.strict_provenance => {
                        info!(
                            "The {option} tables record different alignment provenance at \
                             startup ({error:#}); they are compared again as loaded"
                        );
                        // Keep every shard for startup comparisons with provenance-only groups.
                        groups.extend(checked[snapshot_start..].iter().map(|(path, alignment)| {
                            (path.display().to_string(), alignment.clone(), required)
                        }));
                        continue;
                    }
                    Err(error) => return Err(error),
                };
            groups.push((format!("the {option} tables"), alignment, required));
        }
    }
    let sources = |required: Option<bool>| -> Vec<(String, LazySource<'_>)> {
        groups
            .iter()
            .filter(|(_, _, group_required)| required.is_none_or(|r| r == *group_required))
            .map(|(name, alignment, _)| {
                let alignment = alignment.clone();
                (name.clone(), Box::new(move || Ok(alignment)) as LazySource<'_>)
            })
            .collect()
    };
    let reads = if args.alignments.is_some() {
        // --alignments conflicts with evidence tables: retain the header's warnings and errors.
        reads_alignment(tags.as_ref(), args.alignments.as_deref(), sources(None))?
    } else {
        let resolve = |sources| {
            crate::alignment_provenance::resolve_reads_quiet(tags.as_ref(), None, sources)
        };
        match resolve(sources(None)) {
            Ok(reads) => reads,
            // As within a group: required groups that disagree only with each other are left to
            // the tables loaded, which replace their snapshots. A conflict with a provenance-only
            // group still fails preflight.
            Err(error)
                if !args.provenance.strict_provenance
                    && groups.iter().filter(|(_, _, required)| *required).count() > 1 =>
            {
                for (name, alignment, _) in groups.iter().filter(|(_, _, required)| *required) {
                    let alignment = alignment.clone();
                    let mut with_others = sources(Some(false));
                    with_others.push((name.clone(), Box::new(move || Ok(alignment))));
                    resolve(with_others)?;
                }
                info!(
                    "The required tables record different alignment provenance at \
                     startup ({error:#}); they are compared again as loaded"
                );
                resolve(sources(Some(false)))?
            }
            Err(error) => return Err(error),
        }
    };
    Ok((reads, checked))
}

/// Resolves the provenance of the evidence actually loaded, retaining preflight-only sources.
/// Required tables replace their snapshots, even when a non-strict snapshot check warned. The
/// alignment header is not reopened: its raw snapshot remains the measurement expectation.
fn loaded_score_alignment(
    args: &ScoreArgs,
    checked_reads: &ReadsAlignment,
    checked_tables: &[(PathBuf, Option<AlignmentProvenance>)],
    strlen: Option<&StrlenTables>,
    pileup: Option<&PileupTables>,
) -> Result<ReadsAlignment> {
    // --alignments conflicts with evidence tables; nothing can add a streamed declaration here.
    if args.alignments.is_some() || (args.strlen.is_empty() && args.pileup.is_empty()) {
        return Ok(checked_reads.clone());
    }
    let tags = args.provenance.alignment_tags.tags()?;
    let mut tables: Vec<(String, LazySource<'_>)> = Vec::new();
    for (option, paths, loaded) in [
        ("--strlen", &args.strlen, strlen.map(|table| &table.alignment)),
        ("--pileup", &args.pileup, pileup.map(|table| &table.alignment)),
    ] {
        if paths.is_empty() {
            continue;
        }
        let alignment = match loaded {
            Some(alignment) => alignment.clone(),
            None => {
                let sources: Vec<(String, LazySource<'_>)> = paths
                    .iter()
                    .filter_map(|path| {
                        let alignment = snapshot_of(checked_tables, path)?.clone();
                        let name = path.display().to_string();
                        Some((name, Box::new(move || Ok(alignment)) as LazySource<'_>))
                    })
                    .collect();
                if sources.is_empty() {
                    continue;
                }
                resolve_reads(None, None, sources)?.provenance
            }
        };
        tables.push((format!("the {option} tables"), Box::new(move || Ok(alignment))));
    }
    resolve_reads(tags.as_ref(), None, tables)
}

/// Logs the final alignment check: models without provenance, no comparable tags, or agreement.
/// An affirmative result requires at least one compared tag and no alignment problem.
fn log_alignment_check(
    infos: &[&ModelInfo],
    reads: Option<&AlignmentProvenance>,
    problems: &[Problem],
    preflight_header_only: bool,
) {
    let alignment_problems = [
        Check::AlignmentProvenanceDiffers,
        Check::AlignmentProvenanceIndeterminate,
        Check::AlignmentProvenanceMismatch,
        Check::AlignmentProvenanceNotChecked,
        Check::AlignmentProvenanceIncomplete,
    ];
    if infos.iter().all(|info| info.alignment.is_none()) {
        info!("The models record no alignment provenance, so the reads' alignment is not checked");
    } else if let Some(reads) = reads
        && !problems.iter().any(|problem| alignment_problems.contains(&problem.check))
    {
        let known: Vec<&AlignmentProvenance> =
            infos.iter().filter_map(|info| info.alignment.as_ref()).collect();
        if let Ok(Some(models)) = merge(&known) {
            if comparable_tags(&models, reads).is_empty() {
                info!(
                    "The models and reads record alignment provenance, but no tags were comparable"
                );
            } else {
                if preflight_header_only {
                    info!(
                        "The preflight alignment header's comparable alignment tags match the models' training provenance: {reads}; measurement workers check their opened headers separately (non-strict mismatches warn and scoring continues)"
                    );
                } else {
                    info!(
                        "The comparable alignment tags match the models' training provenance: {reads}"
                    );
                }
            }
        }
    }
}

/// Runs `winnowr check-models`.
///
/// # Errors
///
/// Returns an error if a problem of the models is an error, or the summary cannot be written.
fn check_models(args: &CheckModelsArgs) -> Result<()> {
    check_models_with_writers(args, std::io::stdout().lock(), std::io::stderr().lock())
}

/// Writes the complete check-models report before returning its failure status.
fn check_models_with_writers<W: std::io::Write, E: std::io::Write>(
    args: &CheckModelsArgs,
    out: W,
    err: E,
) -> Result<()> {
    use crate::check_models::{CheckInputs, check_models, write_json, write_text};
    let tags = args.provenance.alignment_tags.tags();
    let tags_error = tags.as_ref().err().map(|error| format!("{error:#}"));
    let report = check_models(&CheckInputs {
        models: &args.models,
        regions: &args.regions,
        reference: args.reference.as_deref(),
        vcf: args.vcf.as_deref(),
        alignments: args.alignments.as_deref(),
        alignment_tags: tags.as_ref().ok().and_then(Option::as_ref),
        alignment_tags_error: tags_error.as_deref(),
        strict_provenance: args.provenance.strict_provenance,
        allow_region_mismatch: args.allow_region_mismatch,
    });
    if args.json {
        write_json(&report, out)?;
    } else {
        write_text(&report, out, err)?;
    }
    if !report.ok {
        let errors = report.problems.iter().filter(|p| p.severity == Severity::Error).count();
        bail!("the models have {errors} error(s); see the problems reported");
    }
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
        Command::CheckModels(args) => check_models(args),
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

    mod captured_logs {
        use std::cell::RefCell;
        use std::sync::Once;

        thread_local! {
            static RECORDS: RefCell<Option<Vec<(log::Level, String)>>> = const { RefCell::new(None) };
        }

        struct Logger;
        impl log::Log for Logger {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }
            fn log(&self, record: &log::Record<'_>) {
                RECORDS.with(|records| {
                    if let Some(records) = records.borrow_mut().as_mut() {
                        records.push((record.level(), record.args().to_string()));
                    }
                });
            }
            fn flush(&self) {}
        }

        pub fn capture<T>(operation: impl FnOnce() -> T) -> (T, Vec<(log::Level, String)>) {
            static INIT: Once = Once::new();
            INIT.call_once(|| {
                log::set_logger(&Logger).unwrap();
                log::set_max_level(log::LevelFilter::Info);
            });
            RECORDS.with(|records| *records.borrow_mut() = Some(Vec::new()));
            let result = operation();
            let records = RECORDS.with(|records| records.borrow_mut().take().unwrap());
            (result, records)
        }
    }

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
        assert!(args.contigs.is_empty() && args.happy.is_none());
        let restricted =
            [&EVALUATE[..], &["--contigs", "chr2", "chr4", "--happy", "h.vcf"]].concat();
        let Command::Evaluate(args) = parse(&restricted).unwrap().command else {
            panic!("not evaluate")
        };
        assert_eq!(args.contigs, ["chr2", "chr4"]);
        assert_eq!(args.happy, Some(PathBuf::from("h.vcf")));
        // --contigs needs --happy to recount the truth FNs, and --happy is only for --contigs.
        for (extra, missing) in
            [(&["--contigs", "chr2"][..], "--happy <HAPPY>"), (&["--happy", "h.vcf"], "--contigs")]
        {
            let error = parse(&[&EVALUATE[..], extra].concat()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument, "{extra:?}");
            let Some(clap::error::ContextValue::Strings(args)) =
                error.get(clap::error::ContextKind::InvalidArg)
            else {
                panic!("no missing argument named: {error}")
            };
            assert!(args[0].starts_with(missing), "{args:?}");
        }
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
        use std::io::Read as _;

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
            crate::testing::write_bgzf(path, text.as_bytes());
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
            try_run(args).unwrap();
        }

        /// Parses and runs a command line, returning its error.
        fn try_run(args: &[&str]) -> Result<()> {
            let cli = parse(args).unwrap();
            dispatch(&cli.command, NonZeroUsize::new(1).unwrap())
        }

        /// Writes the calls (`calls.vcf.gz`) and their hap.py output (`happy.vcf`) to `dir`.
        fn write_calls(dir: &Path) {
            write_calls_on(dir, &["chr1"]);
        }

        /// Writes the calls and their hap.py output, as [`write_calls`] does, with the same calls
        /// on each of `contigs` and 3 truth FNs on the first contig, 6 on the second, and so on, so
        /// that counts tell the contigs apart.
        fn write_calls_on(dir: &Path, contigs: &[&str]) {
            let declare = |header: &str| {
                let lines: String = contigs
                    .iter()
                    .map(|contig| format!("##contig=<ID={contig},length=100000>\n"))
                    .collect();
                header.replace("##contig=<ID=chr1,length=100000>\n", &lines)
            };
            let (mut vcf, mut happy) = (declare(HEADER), declare(HAPPY_HEADER));
            for (k, contig) in contigs.iter().enumerate() {
                for i in 0..CALLS {
                    let pos = 1000 + 200 * i;
                    let (qual, gq, ad, mq) =
                        if is_true(i) { (60, 99, "10,10", 3600) } else { (5, 5, "18,2", 400) };
                    let (mq, dp) = (mq * 20, 20);
                    vcf.push_str(&format!(
                        "{contig}\t{pos}\t.\tA\tC\t{qual}\tPASS\tDP={dp};RAW_MQandDP={mq},{dp};QUALapprox={qual};VarDP={dp};AS_RAW_MQ={mq}|{mq}\tGT:GQ:AD\t0/1:{gq}:{ad}\n"
                    ));
                    let decision = if is_true(i) {
                        "0/1:TP:gm:SNP\t0/1:TP:gm:SNP"
                    } else {
                        ".:.:.:NOCALL\t0/1:FP:.:SNP"
                    };
                    happy.push_str(&format!(
                        "{contig}\t{pos}\t.\tA\tC\t.\t.\t.\tGT:BD:BK:BVT\t{decision}\n"
                    ));
                }
                for pos in (0..3 * (k + 1)).map(|j| 50_000 + 100 * j) {
                    happy.push_str(&format!(
                        "{contig}\t{pos}\t.\tA\tG\t.\t.\t.\tGT:BD:BK:BVT\t0/1:FN:.:SNP\t.:.:.:NOCALL\n"
                    ));
                }
            }
            write_bgzf_indexed(&dir.join("calls.vcf.gz"), &vcf);
            std::fs::write(dir.join("happy.vcf"), happy).unwrap();
        }

        /// Extracts tables of `write_calls_on` calls on chr1 and chr2 in a temporary directory and
        /// trains snp models on them, with extra `train` options (`folds`).
        struct MixedTables<'a> {
            dir: tempfile::TempDir,
            folds: &'a [&'a str],
        }

        impl<'a> MixedTables<'a> {
            fn new(folds: &'a [&'a str]) -> Self {
                let dir = tempfile::TempDir::new().unwrap();
                write_calls_on(dir.path(), &["chr1", "chr2"]);
                Self { dir, folds }
            }

            fn path(&self, name: &str) -> String {
                self.dir.path().join(name).display().to_string()
            }

            /// Extracts the calls of `sample` on `contigs`, returning the table's path.
            fn extract(&self, sample: &str, contigs: &[&str]) -> String {
                let table = self.path(&format!("{sample}.parquet"));
                let (vcf, happy) = (self.path("calls.vcf.gz"), self.path("happy.vcf"));
                let base = ["extract", "--vcf", &vcf, "--happy", &happy, "--sample", sample];
                run(&[&base[..], &["-o", &table, "--contigs"], contigs].concat());
                table
            }

            /// Trains a snp model (`output`) on `tables` and loads it.
            fn train(&self, tables: &[String], output: &str) -> WinnowrModel {
                let tables: Vec<&str> = tables.iter().map(String::as_str).collect();
                let model = self.path(output);
                let base = ["train", "--variant-type", "snp", "--max-rounds", "10", "-o", &model];
                run(&[&base[..], self.folds, &["--tables"], &tables].concat());
                WinnowrModel::load(format!("{model}{MODEL_SUFFIX}"), NonZeroUsize::new(1).unwrap())
                    .unwrap()
            }
        }

        /// The final suffixed model path must not overwrite any training input, even when an
        /// earlier table cannot be read. Symlinks and hard links count as aliases too.
        #[cfg(unix)]
        #[rstest]
        #[case::table_stem(false, "stem")]
        #[case::policy_stem(true, "stem")]
        #[case::table_full(false, "full")]
        #[case::policy_full(true, "full")]
        #[case::table_symlink(false, "symlink")]
        #[case::policy_symlink(true, "symlink")]
        #[case::table_hardlink(false, "hardlink")]
        #[case::policy_hardlink(true, "hardlink")]
        fn train_protects_all_inputs_from_its_final_output(
            #[case] policy: bool,
            #[case] alias: &str,
        ) {
            let dir = tempfile::TempDir::new().unwrap();
            let input = dir.path().join("input.winnowr.json");
            let original = b"input bytes must survive\n";
            std::fs::write(&input, original).unwrap();
            let output = match alias {
                "stem" => dir.path().join("input"),
                "full" => input.clone(),
                "symlink" | "hardlink" => {
                    let output = dir.path().join("output.winnowr.json");
                    if alias == "symlink" {
                        std::os::unix::fs::symlink(&input, &output).unwrap();
                    } else {
                        std::fs::hard_link(&input, &output).unwrap();
                    }
                    output
                }
                _ => unreachable!(),
            };
            let output = output.display().to_string();
            let input = input.display().to_string();
            let missing = dir.path().join("missing.parquet").display().to_string();
            let bed = format!("{NOT_DIFFICULT_REGIONS}={input}");
            let mut args = vec!["train", "--variant-type", "snp", "-o", &output];
            if policy {
                args.extend(["--policy-regions", &bed, "--tables", &missing, &missing]);
            } else {
                args.extend(["--tables", &missing, &input]);
            }
            let error = try_run(&args).unwrap_err().to_string();
            let what = if policy { "policy region BED" } else { "feature table" };
            assert!(error.contains(&format!("is the input {what}")), "{error}");
            assert_eq!(std::fs::read(&input).unwrap(), original);
        }

        /// Training warns when one name has distinct feature and policy BEDs, but keeps both
        /// digests: evaluate/stratify can intentionally use those roles separately.
        #[rstest]
        #[case::conflict(true, true, true)]
        #[case::agreement(true, true, false)]
        #[case::feature_only(true, false, false)]
        #[case::policy_only(false, true, false)]
        fn train_reports_feature_policy_digest_conflicts(
            #[case] feature: bool,
            #[case] policy: bool,
            #[case] conflict: bool,
        ) {
            let setup = MixedTables::new(&[]);
            let feature_bed = setup.path("feature.bed");
            let policy_bed = setup.path("policy.bed");
            std::fs::write(&feature_bed, "chr1\t0\t100000\n").unwrap();
            let policy_lines = if conflict { "chr1\t100\t1000\n" } else { "chr1\t0\t100000\n" };
            std::fs::write(&policy_bed, policy_lines).unwrap();
            let feature_region = format!("{NOT_DIFFICULT_REGIONS}={feature_bed}");
            let policy_region = format!("{NOT_DIFFICULT_REGIONS}={policy_bed}");
            let tables: Vec<String> = ["S1", "S2"]
                .iter()
                .map(|sample| {
                    let table = setup.path(&format!("{sample}.parquet"));
                    let (vcf, happy) = (setup.path("calls.vcf.gz"), setup.path("happy.vcf"));
                    let mut args = vec![
                        "extract",
                        "--vcf",
                        &vcf,
                        "--happy",
                        &happy,
                        "--sample",
                        sample,
                        "--contigs",
                        "chr1",
                        "-o",
                        &table,
                    ];
                    if feature {
                        args.extend(["--regions", &feature_region]);
                    }
                    run(&args);
                    table
                })
                .collect();
            let output = setup.path("trained");
            let mut args =
                vec!["train", "--variant-type", "snp", "--max-rounds", "2", "-o", &output];
            if policy {
                args.extend(["--policy-regions", &policy_region]);
            }
            args.extend(["--tables", &tables[0], &tables[1]]);
            let (result, logs) = captured_logs::capture(|| try_run(&args));
            result.unwrap();
            let warnings: Vec<&str> = logs
                .iter()
                .filter_map(|(level, message)| {
                    (*level == log::Level::Warn && message.contains("feature and policy"))
                        .then_some(message.as_str())
                })
                .collect();
            assert_eq!(warnings.len(), usize::from(conflict), "{logs:?}");
            let info = ModelInfo::read(format!("{output}{MODEL_SUFFIX}")).unwrap();
            assert_eq!(info.region_digests.contains_key(NOT_DIFFICULT_REGIONS), feature);
            assert_eq!(info.policy_region_digests.contains_key(NOT_DIFFICULT_REGIONS), policy);
            if conflict {
                let feature = &info.region_digests[NOT_DIFFICULT_REGIONS];
                let policy = &info.policy_region_digests[NOT_DIFFICULT_REGIONS];
                assert_ne!(feature, policy);
                for text in
                    [NOT_DIFFICULT_REGIONS.to_string(), feature.to_string(), policy.to_string()]
                {
                    assert!(warnings[0].contains(&text), "{}", warnings[0]);
                }
                assert!(warnings[0].contains("score"), "{}", warnings[0]);
            }
        }

        /// A model's reference contigs, in order.
        fn reference_lengths(model: &WinnowrModel) -> Vec<(String, Option<u64>)> {
            let recorded = model.info.reference_contigs.as_ref().unwrap();
            recorded.iter().map(|(contig, length)| (contig.clone(), *length)).collect()
        }

        /// Tables of the same contigs give the model those contigs, in the tables' order.
        #[test]
        fn train_on_tables_of_the_same_contigs() {
            let tables = MixedTables::new(&[]);
            let same =
                [tables.extract("S1", &["chr2", "chr1"]), tables.extract("S2", &["chr2", "chr1"])];
            let model = tables.train(&same, "same");
            assert_eq!(model.info.contigs, ["chr2", "chr1"]);
            let expected =
                vec![("chr2".to_string(), Some(100_000)), ("chr1".to_string(), Some(100_000))];
            assert_eq!(reference_lengths(&model), expected);
        }

        /// `train` takes tables of different contigs: the model records all of them, in the order
        /// of the table covering the most, and `score` scores the calls on all of them.
        #[rstest]
        #[case::samples(&[])]
        #[case::chromosomes(&["--cv-folds", "chromosomes", "--chromosome-folds", "2"])]
        fn train_on_tables_of_different_contigs(#[case] folds: &[&str]) {
            let tables = MixedTables::new(folds);
            // Half of one sample's contigs, listed first, with another sample's whole table.
            let mixed = [tables.extract("S1", &["chr2"]), tables.extract("S2", &["chr1", "chr2"])];
            let model = tables.train(&mixed, "mixed");
            assert_eq!(model.info.training_samples, ["S1", "S2"]);
            assert_eq!(model.info.contigs, ["chr1", "chr2"]);
            let expected =
                vec![("chr1".to_string(), Some(100_000)), ("chr2".to_string(), Some(100_000))];
            assert_eq!(reference_lengths(&model), expected);

            run(&[
                "score",
                "--vcf",
                &tables.path("calls.vcf.gz"),
                "--models",
                &tables.path("mixed.winnowr.json"),
                "-o",
                &tables.path("scored.vcf.gz"),
            ]);
            let scored = read_gzip(Path::new(&tables.path("scored.vcf.gz")));
            let scored: Vec<&str> = scored
                .lines()
                .filter(|line| !line.starts_with('#') && line.contains("WINNOWR_P="))
                .map(|line| line.split('\t').next().unwrap())
                .collect();
            assert_eq!(scored.iter().filter(|chrom| **chrom == "chr1").count(), CALLS);
            assert_eq!(scored.iter().filter(|chrom| **chrom == "chr2").count(), CALLS);
        }

        /// The baseline and genotyper rows of an `evaluate` metrics file: TP, FP, FN, dropped
        /// and flipped.
        fn evaluated_counts(metrics: &Path) -> [[String; 5]; 2] {
            let metrics = std::fs::read_to_string(metrics).unwrap();
            let rows: Vec<Vec<&str>> = metrics.lines().map(|l| l.split('\t').collect()).collect();
            let column = |name: &str| rows[0].iter().position(|c| *c == name).unwrap();
            let row = |policy: &str| {
                let row = rows.iter().find(|r| r[column("policy")] == policy).unwrap();
                ["tp", "fp", "fn", "dropped", "flipped"].map(|name| row[column(name)].to_string())
            };
            [row("baseline"), row("genotyper")]
        }

        /// `evaluate --contigs` counts only the calls and truth FNs on the contigs.
        #[test]
        fn evaluate_counts_only_the_contigs() {
            let dir = tempfile::TempDir::new().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            write_calls_on(dir.path(), &["chr1", "chr2"]);
            for sample in ["S1", "S2"] {
                run(&[
                    "extract",
                    "--vcf",
                    &path("calls.vcf.gz"),
                    "--happy",
                    &path("happy.vcf"),
                    "--contigs",
                    "chr1",
                    "chr2",
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
                "snp",
                "--validation-fraction",
                "0",
                "--max-rounds",
                "10",
                "-o",
                &path("snp"),
            ]);
            let (model, table, happy) =
                (path("snp.winnowr.json"), path("S1.parquet"), path("happy.vcf"));
            let evaluate = |extra: &[&str], output: &str| {
                let base = ["evaluate", "--models", &model, "--table", &table, "-o", output];
                run(&[&base[..], extra].concat());
                evaluated_counts(Path::new(output))
            };
            let all = evaluate(&[], &path("all.tsv"));
            let chr1 = evaluate(&["--contigs", "chr1", "--happy", &happy], &path("chr1.tsv"));
            let chr2 = evaluate(&["--contigs", "chr2", "--happy", &happy], &path("chr2.tsv"));
            // The same calls on each contig, with 3 truth FNs on chr1 and 6 on chr2.
            assert_eq!(all[0][..3], ["60", "60", "9"]);
            assert_eq!(chr1[0][..3], ["30", "30", "3"]);
            assert_eq!(chr2[0][..3], ["30", "30", "6"]);
            assert!(chr2[1][3] != "0", "the genotyper drops calls: {chr2:?}");
            // The model decides the calls on each contig alike, so the contigs' counts add up.
            for (policy, (one, two)) in all.iter().zip(chr1.iter().zip(&chr2)) {
                for (total, (a, b)) in policy.iter().zip(one.iter().zip(two)) {
                    let sum = a.parse::<usize>().unwrap() + b.parse::<usize>().unwrap();
                    assert_eq!(sum.to_string(), *total, "{all:?} vs {chr1:?} + {chr2:?}");
                }
            }
            // Both contigs, named, are the whole table.
            let both = ["--contigs", "chr1", "chr2", "--happy", &happy];
            assert_eq!(evaluate(&both, &path("both.tsv")), all);

            let error = try_run(&[
                "evaluate",
                "--models",
                &model,
                "--table",
                &table,
                "-o",
                &path("chr3.tsv"),
                "--contigs",
                "chr3",
                "--happy",
                &happy,
            ])
            .unwrap_err();
            assert!(error.to_string().contains("not extracted from contigs chr3"), "{error}");
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
            let [baseline, genotyper] = evaluated_counts(Path::new(&path("metrics.tsv")));
            assert_eq!(baseline[..3], ["30", "30", "3"]);
            let dropped: usize = genotyper[3].parse().unwrap();
            let flipped: usize = genotyper[4].parse().unwrap();
            assert!(dropped > 0, "{genotyper:?}");

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

        /// Writes the calls of [`write_calls`] (`calls.vcf.gz`, with the raw annotations) and
        /// their hap.py output (`happy.vcf`), and the same calls as a VCF of HaplotypeCaller's
        /// default mode (`default.vcf.gz`): INFO/MQ, INFO/SOR and FORMAT/PL in place of the raw
        /// annotations, so that only the vcf feature source can read it.
        fn write_default_mode_calls(dir: &Path) {
            write_calls(dir);
            let is_raw = |key: &str| crate::gvcf::RAW_ANNOTATION_INFO.contains(&key);
            let fields = "##INFO=<ID=MQ,Number=1,Type=Float,Description=\"RMS MQ\">\n\
                 ##INFO=<ID=SOR,Number=1,Type=Float,Description=\"SOR\">\n\
                 ##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"PL\">\n";
            let header: String = HEADER
                .lines()
                .filter(|line| {
                    let id = line.strip_prefix("##INFO=<ID=").and_then(|r| r.split(',').next());
                    !id.is_some_and(is_raw)
                })
                .map(|line| format!("{line}\n"))
                .collect();
            let mut vcf = header.replace("#CHROM", &format!("{fields}#CHROM"));
            for line in read_gzip(&dir.join("calls.vcf.gz")).lines().filter(|l| !l.starts_with('#'))
            {
                let mut columns: Vec<String> = line.split('\t').map(str::to_string).collect();
                // True calls have MQ 60 and false ones MQ 20 (see write_calls_on).
                let mq = if columns[7].contains("RAW_MQandDP=72000") { 60 } else { 20 };
                let qual: u32 = columns[5].parse().unwrap();
                let info: Vec<&str> = columns[7]
                    .split(';')
                    .filter(|entry| !is_raw(entry.split('=').next().unwrap()))
                    .collect();
                columns[7] = format!("{};MQ={mq};SOR=0.7", info.join(";"));
                columns[8] = format!("{}:PL", columns[8]);
                columns[9] = format!("{}:{qual},0,{}", columns[9], 3 * qual);
                vcf.push_str(&columns.join("\t"));
                vcf.push('\n');
            }
            write_bgzf_indexed(&dir.join("default.vcf.gz"), &vcf);
        }

        /// The calls of [`write_default_mode_calls`] in a temporary directory, with a snp model
        /// (`vcf.winnowr.json`) trained on tables of S1 and S2 extracted with the vcf feature
        /// source.
        struct VcfSourceModel {
            dir: tempfile::TempDir,
        }

        impl VcfSourceModel {
            fn new() -> Self {
                let model = Self { dir: tempfile::TempDir::new().unwrap() };
                write_default_mode_calls(model.dir.path());
                let tables = [
                    model.extract("S1", "default.vcf.gz", "vcf"),
                    model.extract("S2", "default.vcf.gz", "vcf"),
                ];
                model.train(&tables, "vcf").unwrap();
                model
            }

            fn path(&self, name: &str) -> String {
                self.dir.path().join(name).display().to_string()
            }

            /// The model's path.
            fn model(&self) -> String {
                self.path("vcf.winnowr.json")
            }

            /// The table of `sample`'s calls in `vcf` extracted with the feature `source`.
            fn extract(&self, sample: &str, vcf: &str, source: &str) -> String {
                let table = self.path(&format!("{sample}.{source}.parquet"));
                run(&[
                    "extract",
                    "--vcf",
                    &self.path(vcf),
                    "--happy",
                    &self.path("happy.vcf"),
                    "--contigs",
                    "chr1",
                    "--sample",
                    sample,
                    "--feature-source",
                    source,
                    "-o",
                    &table,
                ]);
                table
            }

            /// Trains a snp model (`output`) on `tables`.
            fn train<S: AsRef<str>>(&self, tables: &[S], output: &str) -> Result<()> {
                let model = self.path(output);
                let tables: Vec<&str> = tables.iter().map(AsRef::as_ref).collect();
                let base = ["train", "--variant-type", "snp", "--max-rounds", "10", "-o", &model];
                try_run(&[&base[..], &["--tables"], &tables].concat())
            }
        }

        /// Features derived from a VCF (`--feature-source vcf`): the table and the model record
        /// the source; the gVCF source cannot read the VCF, which lacks the raw annotations.
        #[test]
        fn the_vcf_feature_source_is_recorded_in_tables_and_models() {
            let setup = VcfSourceModel::new();
            let table = FeatureTable::read_parquet(setup.path("S1.vcf.parquet")).unwrap();
            assert_eq!(table.info().feature_source, FeatureSource::Vcf);
            assert!(table.f64_column("as_alt_frac").unwrap().iter().all(|v| v.is_nan()));
            let mq = table.f64_column("mq").unwrap();
            assert!(mq.iter().all(|&v| v == 60.0 || v == 20.0), "{mq:?}");
            let model = WinnowrModel::load(setup.model(), NonZeroUsize::new(1).unwrap()).unwrap();
            assert_eq!(model.info.feature_source, FeatureSource::Vcf);
            let text = std::fs::read_to_string(setup.model()).unwrap();
            assert!(text.contains(r#""feature_source":"vcf""#));

            let gvcf = setup.path("gvcf.parquet");
            let extract = ["extract", "--vcf", &setup.path("default.vcf.gz"), "--contigs", "chr1"];
            let error = try_run(&[&extract[..], &["-o", &gvcf]].concat()).unwrap_err();
            assert!(format!("{error:#}").contains("lacks the INFO annotations"), "{error:#}");
        }

        /// Tables of different feature sources do not train together.
        #[test]
        fn train_rejects_tables_of_different_feature_sources() {
            let setup = VcfSourceModel::new();
            let gvcf_table = setup.extract("S2", "calls.vcf.gz", "gvcf");
            let tables = [setup.path("S1.vcf.parquet"), gvcf_table];
            let error = setup.train(&tables, "mixed").unwrap_err().to_string();
            assert!(
                error.contains(
                    "--tables hold features from different sources: S1's from vcf, S2's from gvcf"
                ),
                "{error}"
            );
        }

        #[rstest]
        #[case::gzip("scored.vcf.gz")]
        #[case::bgzf("scored.vcf.bgz")]
        #[case::bcf("scored.bcf")]
        fn score_rejects_output_index_alias_before_loading_models(
            #[case] name: &str,
            #[values(false, true)] hard_link: bool,
        ) {
            let dir = tempfile::TempDir::new().unwrap();
            write_calls(dir.path());
            let vcf = dir.path().join("calls.vcf.gz");
            let model = dir.path().join("missing.winnowr.json");
            let output = dir.path().join(name);
            let sidecar = crate::scoring::scored_vcf_index(&output).unwrap();
            let before = b"preserve the existing scored output";
            std::fs::write(&output, before).unwrap();
            if hard_link {
                std::fs::hard_link(&output, &sidecar).unwrap();
            } else {
                std::os::unix::fs::symlink(name, &sidecar).unwrap();
            }
            let error = format!(
                "{:#}",
                try_run(&[
                    "score",
                    "--vcf",
                    vcf.to_str().unwrap(),
                    "--models",
                    model.to_str().unwrap(),
                    "-o",
                    output.to_str().unwrap(),
                ])
                .unwrap_err()
            );
            assert!(error.ends_with("is the input scored VCF; write to another path"), "{error}");
            assert_eq!(std::fs::read(&output).unwrap(), before);
            assert_eq!(std::fs::read(&sidecar).unwrap(), before);
        }

        /// score derives a vcf-source model's features from the VCF, which only that source can
        /// read; a --gvcf is ignored (so it need not exist, and it may be the output, as on a
        /// rerun).
        #[test]
        fn score_derives_the_features_of_vcf_source_models_from_the_vcf() {
            let setup = VcfSourceModel::new();
            let rerun = setup.path("scored.vcf.gz");
            for (output, extra) in [
                ("scored.vcf.gz", &[][..]),
                ("ignored.vcf.gz", &["--gvcf", "absent.g.vcf.gz"][..]),
                ("scored.vcf.gz", &["--gvcf", rerun.as_str()][..]),
            ] {
                let (vcf, model) = (setup.path("default.vcf.gz"), setup.model());
                let base = ["score", "--vcf", &vcf, "--models", &model];
                run(&[&base[..], &["-o", &setup.path(output)], extra].concat());
                let scored = read_gzip(&setup.dir.path().join(output));
                let records = scored.lines().filter(|line| !line.starts_with('#'));
                assert_eq!(records.filter(|r| r.contains("WINNOWR_P=")).count(), CALLS);
            }
        }

        #[rstest]
        #[case::raw_present(true, true)]
        #[case::header_missing(false, false)]
        #[case::records_missing(true, false)]
        fn gvcf_output_protection_follows_actual_consumption(
            #[case] declared: bool,
            #[case] present: bool,
        ) {
            let setup = VcfSourceModel::new();
            let model_path = setup.model();
            let mut model = WinnowrModel::load(&model_path, NonZeroUsize::new(1).unwrap()).unwrap();
            model.info.feature_source = FeatureSource::Gvcf;
            model.save(&model_path).unwrap();
            let vcf = if !declared {
                setup.path("default.vcf.gz")
            } else if present {
                setup.path("calls.vcf.gz")
            } else {
                let text = read_gzip(&setup.dir.path().join("calls.vcf.gz"));
                let mut stripped = String::new();
                for line in text.lines() {
                    if line.starts_with('#') {
                        stripped.push_str(line);
                    } else {
                        let mut columns: Vec<String> =
                            line.split('\t').map(str::to_string).collect();
                        columns[7] = columns[7]
                            .split(';')
                            .filter(|field| {
                                !crate::gvcf::RAW_ANNOTATION_INFO
                                    .contains(&field.split('=').next().unwrap())
                            })
                            .collect::<Vec<_>>()
                            .join(";");
                        stripped.push_str(&columns.join("\t"));
                    }
                    stripped.push('\n');
                }
                let path = setup.path("declared.vcf.gz");
                write_bgzf_indexed(Path::new(&path), &stripped);
                path
            };
            for command in ["score", "extract"] {
                for sidecar in [false, true] {
                    if sidecar && command == "extract" {
                        continue;
                    }
                    for index in [false, true] {
                        let ext = if command == "score" { "vcf.gz" } else { "parquet" };
                        let output = setup.path(&format!("{command}-{sidecar}-{index}.{ext}"));
                        let target = format!("{output}{}", if sidecar { ".tbi" } else { "" });
                        let gvcf = if index {
                            let path = format!("{output}.g.vcf.gz");
                            std::os::unix::fs::symlink(&target, format!("{path}.csi")).unwrap();
                            path
                        } else {
                            target.clone()
                        };
                        let mut args = vec![command, "--vcf", &vcf, "--gvcf", &gvcf, "-o", &output];
                        if command == "score" {
                            args.extend(["--models", &model_path]);
                        } else {
                            args.extend(["--contigs", "chr1"]);
                        }
                        if present {
                            for _ in 0..2 {
                                try_run(&args).unwrap();
                            }
                            if command == "score" {
                                assert!(read_gzip(Path::new(&output)).contains("WINNOWR_P="));
                            } else {
                                FeatureTable::read_parquet(&output).unwrap();
                            }
                        } else {
                            std::fs::write(&target, b"preserve me").unwrap();
                            let error = format!("{:#}", try_run(&args).unwrap_err());
                            let what = if index { "gVCF index" } else { "gVCF" };
                            assert!(
                                error.ends_with(&format!(
                                    "is the input {what}; write to another path"
                                )),
                                "{error}"
                            );
                            assert_eq!(std::fs::read(&target).unwrap(), b"preserve me");
                        }
                    }
                }
            }
        }

        #[rstest]
        #[case::score_vcf("score", false)]
        #[case::extract_vcf("extract", false)]
        #[case::score_gvcf("score", true)]
        #[case::extract_gvcf("extract", true)]
        fn new_outputs_cannot_mask_consumed_variant_indexes(
            #[case] command: &str,
            #[case] gvcf: bool,
        ) {
            let setup = VcfSourceModel::new();
            let model_path = setup.model();
            if gvcf {
                let mut model =
                    WinnowrModel::load(&model_path, NonZeroUsize::new(1).unwrap()).unwrap();
                model.info.feature_source = FeatureSource::Gvcf;
                model.save(&model_path).unwrap();
            }
            let vcf = setup.path("default.vcf.gz");
            let source = if gvcf { setup.path("calls.vcf.gz") } else { vcf.clone() };
            let candidate = format!("{source}.csi");
            let tbi = format!("{source}.tbi");
            let before = std::fs::read(&tbi).unwrap();
            for sidecar in [false, true] {
                if sidecar && command == "extract" {
                    continue;
                }
                let output = if sidecar {
                    let output = setup.path("output.vcf.gz");
                    std::os::unix::fs::symlink(format!("{output}.tbi"), &candidate).unwrap();
                    output
                } else {
                    candidate.clone()
                };
                let mut args = vec![command, "--vcf", &vcf, "-o", &output];
                if command == "score" {
                    args.extend(["--models", &model_path]);
                } else {
                    args.extend(["--contigs", "chr1"]);
                    if !gvcf {
                        args.extend(["--feature-source", "vcf"]);
                    }
                }
                if gvcf {
                    args.extend(["--gvcf", &source]);
                }
                let error = format!("{:#}", try_run(&args).unwrap_err());
                let what = if gvcf { "gVCF" } else { "VCF" };
                assert!(error.contains(&format!("would mask the input {what} index")), "{error}");
                assert_eq!(std::fs::read(&tbi).unwrap(), before);
                assert!(!Path::new(&candidate).exists());
                assert!(!Path::new(&output).exists());
            }
        }

        #[test]
        fn final_alignment_audit_does_not_invent_unmeasured_indexes() {
            let dir = tempfile::TempDir::new().unwrap();
            let reads = dir.path().join("reads.bam");
            let index = dir.path().join("reads.bam.bai");
            std::fs::write(&reads, b"BAM!").unwrap();
            std::fs::write(&index, b"unused index").unwrap();
            let mut inputs =
                resolve_output_inputs(&[("alignments", Some(&reads), InputIndexes::Alignments)]);
            assert!(check_output_not_resolved_inputs(&index, &inputs, &[], &[], &[]).is_err());
            inputs[0].observed_indexes = Some(crate::read_pass::ReadIndexObservation::default());
            check_output_not_resolved_inputs(&index, &inputs, &[], &[], &[]).unwrap();
        }

        /// Index searches for BAM and CRAM remain separate, even if both indexes exist.
        #[rstest]
        #[case::bam_to_cram(false, "reads.data.crai")]
        #[case::cram_to_bam_csi(true, "reads.data.csi")]
        #[case::cram_to_bam_bai(true, "reads.data.bai")]
        fn review_final_alignment_audit_covers_changed_formats(
            #[case] initially_cram: bool,
            #[case] index_name: &str,
        ) {
            use rust_htslib::bam::{self, Read as _};
            let dir = tempfile::TempDir::new().unwrap();
            let reference = crate::testing::write_fasta(dir.path(), &[("chr1", &"A".repeat(100))]);
            let bam = crate::testing::write_bam(dir.path(), "source.bam", 100, &[]);
            let source = bam::Reader::from_path(&bam).unwrap();
            let header = bam::Header::from_template(source.header());
            let cram = dir.path().join("source.cram");
            let mut writer = bam::Writer::from_path(&cram, &header, bam::Format::Cram).unwrap();
            writer.set_reference(&reference).unwrap();
            drop(writer);
            bam::index::build(&cram, None, bam::index::Type::Bai, 1).unwrap();
            let reads = dir.path().join("reads.data");
            std::fs::copy(if initially_cram { &cram } else { &bam }, &reads).unwrap();
            let mut inputs =
                resolve_output_inputs(&[("alignments", Some(&reads), InputIndexes::Alignments)]);
            let observed = crate::read_pass::ReadIndexObservation::default();
            inputs[0].observed_indexes = Some(observed.clone());
            std::fs::copy(if initially_cram { &bam } else { &cram }, &reads).unwrap();
            // Both formats' indexes coexist, but only the measurement reader's search is active.
            std::fs::copy(bam.with_extension("bam.bai"), dir.path().join("reads.data.bai"))
                .unwrap();
            std::fs::copy(cram.with_extension("cram.crai"), dir.path().join("reads.data.crai"))
                .unwrap();
            if index_name.ends_with(".csi") {
                bam::index::build(&reads, None, bam::index::Type::Csi(14), 1).unwrap();
            }
            let measuring = ReadInputs {
                vcf: Path::new("unused.vcf.gz"),
                alignments: &reads,
                reference: &reference,
                expected_alignment: None,
                observed_indexes: Some(observed),
                stop: None,
                scoring_stop: None,
            };
            drop(measuring.open_alignments().unwrap());
            let index = dir.path().join(index_name);
            let before = std::fs::read(&index).unwrap();
            // No data-path reopen is needed after measurement, even if it disappears.
            std::fs::remove_file(&reads).unwrap();
            let error = check_output_not_resolved_inputs(&index, &inputs, &[], &[], &[])
                .unwrap_err()
                .to_string();
            assert!(error.contains("is the input alignments index"), "{error}");
            assert_eq!(std::fs::read(&index).unwrap(), before);
        }

        #[rstest]
        fn final_alignment_index_audit_covers_both_measurement_paths(
            #[values(false, true)] background_evidence: bool,
            #[values(false, true)] sidecar: bool,
        ) {
            let setup = AlignmentsModel::new();
            let model_path = setup.path("reads.winnowr.json");
            if background_evidence {
                let mut model = WinnowrModel::load(&model_path, NonZeroUsize::MIN).unwrap();
                model.info.feature_source = FeatureSource::Vcf;
                model.save(&model_path).unwrap();
            }
            let output = setup.path("scored.vcf.gz");
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                &setup.path("default.vcf.gz"),
                "--models",
                &model_path,
                "--alignments",
                &setup.path("reads.bam"),
                "--reference",
                &setup.path("ref.fa"),
                "-o",
                &output,
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let index = setup.path("reads.bam.bai");
            let before = std::fs::read(&index).unwrap();
            let error = score_with(&args, NonZeroUsize::MIN, || {
                let destination = if sidecar { format!("{output}.tbi") } else { output.clone() };
                std::os::unix::fs::symlink(&index, destination).unwrap();
            })
            .unwrap_err()
            .to_string();
            assert!(error.contains("is the input alignments index"), "{error}");
            assert_eq!(std::fs::read(&index).unwrap(), before);
        }

        #[test]
        fn new_outputs_cannot_mask_consumed_alignment_indexes() {
            let setup = AlignmentsModel::new();
            let candidate = setup.path("reads.bam.csi");
            let bai = setup.path("reads.bam.bai");
            let before = std::fs::read(&bai).unwrap();
            let reads = setup.reads();
            let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
            for sidecar in [false, true] {
                let output = if sidecar { "output.vcf.gz" } else { "reads.bam.csi" };
                if sidecar {
                    std::os::unix::fs::symlink(format!("{}.tbi", setup.path(output)), &candidate)
                        .unwrap();
                }
                let error = format!("{:#}", setup.score(output, &reads).unwrap_err());
                assert!(error.contains("would mask the input alignments index"), "{error}");
                assert_eq!(std::fs::read(&bai).unwrap(), before);
                assert!(!Path::new(&candidate).exists());
            }
        }

        #[rstest]
        #[case::ordinary("calls.vcf.gz", "calls.vcf")]
        #[case::dotted_parent("batch.v1/calls", "batch")]
        fn local_variant_index_order_matches_htslib(#[case] data: &str, #[case] replacement: &str) {
            let setup = VcfSourceModel::new();
            let vcf = setup.dir.path().join(data);
            std::fs::create_dir_all(vcf.parent().unwrap()).unwrap();
            std::fs::copy(setup.path("default.vcf.gz"), &vcf).unwrap();
            let appended_csi = format!("{}.csi", vcf.display());
            rust_htslib::bcf::index::build(&vcf, None, 1, rust_htslib::bcf::index::Type::Csi(14))
                .unwrap();
            let csi = setup.path(&format!("{replacement}.csi"));
            std::fs::rename(&appended_csi, &csi).unwrap();
            std::fs::copy(setup.path("default.vcf.gz.tbi"), format!("{}.tbi", vcf.display()))
                .unwrap();
            assert_eq!(
                crate::vcf::index_candidates(&vcf),
                vec![
                    PathBuf::from(&appended_csi),
                    PathBuf::from(&csi),
                    PathBuf::from(format!("{}.tbi", vcf.display())),
                    PathBuf::from(setup.path(&format!("{replacement}.tbi")))
                ]
            );
            crate::vcf::open_indexed(&vcf).unwrap();
            let before = std::fs::read(&csi).unwrap();
            for sidecar in [false, true] {
                let output = if sidecar {
                    let output = setup.path("sidecar.vcf.gz");
                    std::os::unix::fs::symlink(&csi, format!("{output}.tbi")).unwrap();
                    output
                } else {
                    csi.clone()
                };
                let error = format!(
                    "{:#}",
                    try_run(&[
                        "score",
                        "--vcf",
                        vcf.to_str().unwrap(),
                        "--models",
                        &setup.model(),
                        "-o",
                        &output
                    ])
                    .unwrap_err()
                );
                assert!(
                    error.ends_with("is the input VCF index; write to another path"),
                    "{error}"
                );
                assert_eq!(std::fs::read(&csi).unwrap(), before);
            }
            std::fs::write(&csi, b"not an index").unwrap();
            let error = crate::vcf::open_indexed(&vcf).map(|_| ()).unwrap_err().to_string();
            assert!(error.ends_with("is not a tabix or CSI index"), "{error}");
            assert!(
                !crate::vcf::has_index(&vcf),
                "a later TBI cannot override the CSI htslib selects"
            );
            std::fs::remove_file(&csi).unwrap();
            std::fs::create_dir(&csi).unwrap();
            assert!(!crate::vcf::has_index(&vcf), "htslib also selects a CSI directory");
            let error = crate::vcf::open_indexed(&vcf).map(|_| ()).unwrap_err().to_string();
            assert!(error.ends_with("is not a tabix or CSI index"), "{error}");
        }

        #[test]
        fn index_replacement_preserves_native_paths_and_loader_rules() {
            use std::os::unix::ffi::{OsStrExt, OsStringExt};
            let data =
                PathBuf::from(std::ffi::OsString::from_vec(b"native\xff/batch.v1/calls".to_vec()));
            let paths = crate::vcf::local_index_paths(&data, &["csi"]);
            assert_eq!(paths[0].as_os_str().as_bytes(), b"native\xff/batch.v1/calls.csi");
            assert_eq!(paths[1].as_os_str().as_bytes(), b"native\xff/batch.csi");
            for (input, expected) in [
                ("calls", vec!["calls.csi"]),
                (".calls", vec![".calls.csi"]),
                ("dir/.calls", vec!["dir/.calls.csi", "dir/.csi"]),
                ("../calls", vec!["../calls.csi", "..csi"]),
            ] {
                assert_eq!(
                    crate::vcf::local_index_paths(Path::new(input), &["csi"]),
                    expected.iter().map(PathBuf::from).collect::<Vec<_>>()
                );
            }
            let dir = tempfile::TempDir::new().unwrap();
            let cram = dir.path().join("batch.v1/reads");
            std::fs::create_dir_all(cram.parent().unwrap()).unwrap();
            std::fs::write(&cram, b"CRAM").unwrap();
            let inputs =
                resolve_output_inputs(&[("alignments", Some(&cram), InputIndexes::Alignments)]);
            assert_eq!(inputs[0].indexes, crate::vcf::local_index_paths(&cram, &["csi", "crai"]));
            assert!(!InputIndexes::Reference.paths(&cram).contains(&dir.path().join("batch.fai")));
            assert!(!InputIndexes::Reference.paths(&cram).contains(&dir.path().join("batch.gzi")));
        }

        #[rstest]
        #[case::csi("csi")]
        #[case::bai("bai")]
        fn dotted_parent_alignment_indexes_are_protected(#[case] extension: &str) {
            let setup = AlignmentsModel::new();
            let bam = setup.path("batch.v1/reads");
            std::fs::create_dir_all(Path::new(&bam).parent().unwrap()).unwrap();
            std::fs::copy(setup.path("reads.bam"), &bam).unwrap();
            let index = setup.path(&format!("batch.{extension}"));
            let kind = if extension == "csi" {
                rust_htslib::bam::index::Type::Csi(14)
            } else {
                rust_htslib::bam::index::Type::Bai
            };
            rust_htslib::bam::index::build(&bam, Some(&index), kind, 1).unwrap();
            let before = std::fs::read(&index).unwrap();
            for sidecar in [false, true] {
                let output = if sidecar {
                    "sidecar.vcf.gz".to_string()
                } else {
                    format!("batch.{extension}")
                };
                if sidecar {
                    std::os::unix::fs::symlink(&index, format!("{}.tbi", setup.path(&output)))
                        .unwrap();
                }
                let error = format!(
                    "{:#}",
                    setup
                        .score(
                            &output,
                            &["--alignments", &bam, "--reference", &setup.path("ref.fa")]
                        )
                        .unwrap_err()
                );
                assert!(
                    error.ends_with("is the input alignments index; write to another path"),
                    "{error}"
                );
                assert_eq!(std::fs::read(&index).unwrap(), before);
            }
        }

        #[test]
        fn hidden_cram_index_is_protected_before_opening_reads() {
            let setup = AlignmentsModel::new();
            let cram = setup.path(".reads");
            let index = setup.path(".crai");
            std::fs::write(&cram, b"CRAM").unwrap();
            std::fs::write(&index, b"preserve the hidden CRAI").unwrap();
            for output in [".crai", "hidden.vcf.gz"] {
                if output.ends_with(".gz") {
                    std::os::unix::fs::symlink(&index, format!("{}.tbi", setup.path(output)))
                        .unwrap();
                }
                let error = format!(
                    "{:#}",
                    setup
                        .score(
                            output,
                            &["--alignments", &cram, "--reference", &setup.path("ref.fa")]
                        )
                        .unwrap_err()
                );
                assert!(
                    error.ends_with("is the input alignments index; write to another path"),
                    "{error}"
                );
                assert_eq!(std::fs::read(&index).unwrap(), b"preserve the hidden CRAI");
            }
        }

        #[test]
        fn unused_cram_indexes_do_not_block_bam_reruns() {
            let setup = AlignmentsModel::new();
            let reads = setup.reads();
            let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
            let bai = setup.path("reads.bam.bai");
            let before = std::fs::read(&bai).unwrap();
            for output in ["reads.bam.crai", "reads.crai"] {
                for _ in 0..2 {
                    setup.score(output, &reads).unwrap();
                }
                let scored = std::fs::read_to_string(setup.path(output)).unwrap();
                assert_eq!(
                    scored
                        .lines()
                        .filter(|line| !line.starts_with('#') && line.contains("WINNOWR_P="))
                        .count(),
                    CALLS
                );
            }
            assert_eq!(std::fs::read(&bai).unwrap(), before);
        }

        #[rstest]
        #[case::appended("reads.cram", "reads.cram.csi")]
        #[case::dotted_parent("batch.v1/reads", "batch.csi")]
        fn cram_csi_candidates_are_protected(
            #[case] data: &str,
            #[case] index: &str,
            #[values("score", "extract")] command: &str,
            #[values(false, true)] existing: bool,
        ) {
            use rust_htslib::bam::{self, Read as _};
            let setup = AlignmentsModel::new();
            let reference = setup.path("ref.fa");
            let mut source = bam::Reader::from_path(setup.path("reads.bam")).unwrap();
            let header = bam::Header::from_template(source.header());
            let cram = setup.dir.path().join(data);
            std::fs::create_dir_all(cram.parent().unwrap()).unwrap();
            let mut writer = bam::Writer::from_path(&cram, &header, bam::Format::Cram).unwrap();
            writer.set_reference(&reference).unwrap();
            for record in source.records() {
                writer.write(&record.unwrap()).unwrap();
            }
            drop(writer);
            bam::index::build(&cram, None, bam::index::Type::Bai, 1).unwrap();
            let index = setup.dir.path().join(index);
            let crai = PathBuf::from(format!("{}.crai", cram.display()));
            if existing {
                std::fs::copy(&crai, &index).unwrap();
            }
            // The default loader accepts a CRAI stored under a CSI candidate name.
            let mut reader = bam::IndexedReader::from_path(&cram).unwrap();
            reader.set_reference(&reference).unwrap();
            reader.fetch(("chr1", 0, 10000)).unwrap();
            assert!(reader.records().next().unwrap().is_ok());
            drop(reader);
            let original = std::fs::read(&crai).unwrap();
            let vcf = setup.path("default.vcf.gz");
            let model = setup.path("reads.winnowr.json");
            let mut args = vec![
                command,
                "--vcf",
                &vcf,
                "--alignments",
                cram.to_str().unwrap(),
                "--reference",
                &reference,
                "-o",
                index.to_str().unwrap(),
            ];
            if command == "score" {
                args.extend(["--models", &model]);
            } else {
                args.extend(["--contigs", "chr1"]);
            }
            let error = format!("{:#}", try_run(&args).unwrap_err());
            let expected = if existing {
                "is the input alignments index; write to another path"
            } else {
                "would mask the input alignments index; write to another path"
            };
            assert!(error.contains(expected), "{error}");
            assert_eq!(std::fs::read(&crai).unwrap(), original);
            if existing {
                assert_eq!(std::fs::read(&index).unwrap(), original);
                if command == "score" {
                    let reads = ["--alignments", cram.to_str().unwrap(), "--reference", &reference];
                    setup.score("csi-indexed.vcf", &reads).unwrap();
                    // A malformed higher-priority candidate must not fall back to the valid CRAI.
                    std::fs::write(&index, b"not an index").unwrap();
                    assert!(bam::IndexedReader::from_path(&cram).is_err());
                    let error = format!("{:#}", setup.score("bad-index.vcf", &reads).unwrap_err());
                    assert!(error.contains("it must be indexed"), "{error}");
                }
            } else {
                assert!(!index.exists());
            }
        }

        #[test]
        fn unused_bai_indexes_do_not_block_cram_reruns() {
            use rust_htslib::bam::{self, Read as _};
            let setup = AlignmentsModel::new();
            let reference = setup.path("ref.fa");
            let mut source = bam::Reader::from_path(setup.path("reads.bam")).unwrap();
            let header = bam::Header::from_template(source.header());
            let cram = setup.path("reads.cram");
            let mut writer = bam::Writer::from_path(&cram, &header, bam::Format::Cram).unwrap();
            writer.set_reference(&reference).unwrap();
            for record in source.records() {
                writer.write(&record.unwrap()).unwrap();
            }
            drop(writer);
            bam::index::build(&cram, None, bam::index::Type::Bai, 1).unwrap();
            let crai = format!("{cram}.crai");
            let before = std::fs::read(&crai).unwrap();
            for output in ["reads.cram.bai", "reads.bai"] {
                for _ in 0..2 {
                    setup
                        .score(output, &["--alignments", &cram, "--reference", &reference])
                        .unwrap();
                }
                let scored = std::fs::read_to_string(setup.path(output)).unwrap();
                assert_eq!(
                    scored
                        .lines()
                        .filter(|line| !line.starts_with('#') && line.contains("WINNOWR_P="))
                        .count(),
                    CALLS
                );
            }
            assert_eq!(std::fs::read(&crai).unwrap(), before);
        }

        #[test]
        fn unused_replaced_fasta_indexes_do_not_block_reruns() {
            let setup = AlignmentsModel::new();
            let fasta = setup.dir.path().join("ref.fa");
            assert_eq!(
                InputIndexes::Reference.paths(&fasta),
                vec![setup.dir.path().join("ref.fa.fai"), setup.dir.path().join("ref.fa.gzi")]
            );
            let index = setup.path("ref.fa.fai");
            let before = std::fs::read(&index).unwrap();
            let reads = setup.reads();
            let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
            for output in ["ref.fai", "ref.gzi"] {
                for _ in 0..2 {
                    setup.score(output, &reads).unwrap();
                }
                let scored = std::fs::read_to_string(setup.path(output)).unwrap();
                assert_eq!(
                    scored
                        .lines()
                        .filter(|line| !line.starts_with('#') && line.contains("WINNOWR_P="))
                        .count(),
                    CALLS
                );
            }
            assert_eq!(std::fs::read(&index).unwrap(), before);
        }

        #[test]
        fn bcf_tabix_names_are_not_protected_inputs() {
            let setup = VcfSourceModel::new();
            let (vcf, bcf, model) =
                (setup.path("default.vcf.gz"), setup.path("calls.bcf"), setup.model());
            crate::scoring::write_scored_vcf(
                Path::new(&vcf),
                Path::new(&bcf),
                std::collections::HashMap::new(),
                NonZeroUsize::new(1).unwrap(),
                &[],
            )
            .unwrap();
            let csi = format!("{bcf}.csi");
            let before = std::fs::read(&csi).unwrap();
            for name in ["calls.bcf.tbi", "calls.tbi"] {
                let output = setup.path(name);
                for _ in 0..2 {
                    run(&["score", "--vcf", &bcf, "--models", &model, "-o", &output]);
                    let scored = std::fs::read_to_string(&output).unwrap();
                    assert_eq!(
                        scored
                            .lines()
                            .filter(|r| !r.starts_with('#') && r.contains("WINNOWR_P="))
                            .count(),
                        CALLS
                    );
                }
                run(&[
                    "extract",
                    "--vcf",
                    &bcf,
                    "--feature-source",
                    "vcf",
                    "--contigs",
                    "chr1",
                    "-o",
                    &output,
                ]);
                FeatureTable::read_parquet(&output).unwrap();
            }
            let error = format!(
                "{:#}",
                try_run(&["score", "--vcf", &bcf, "--models", &model, "-o", &csi]).unwrap_err()
            );
            assert!(error.ends_with("is the input VCF index; write to another path"), "{error}");
            assert_eq!(std::fs::read(&csi).unwrap(), before);
        }

        #[rstest]
        #[case::bam("--alignments", "")]
        #[case::bai("--alignments", ".bai")]
        #[case::csi("--alignments", ".csi")]
        #[case::crai("--alignments", ".crai")]
        #[case::fasta("--reference", "")]
        #[case::fai("--reference", ".fai")]
        #[case::gzi("--reference", ".gzi")]
        #[case::pileup("--pileup", "")]
        #[case::strlen("--strlen", "")]
        fn unused_evidence_does_not_prevent_scoring_again(
            #[case] option: &str,
            #[case] index: &str,
        ) {
            let setup = VcfSourceModel::new();
            let (vcf, model) = (setup.path("default.vcf.gz"), setup.model());
            for suffix in ["", ".tbi"] {
                let output = setup.path(&format!("rerun{suffix}.vcf.gz"));
                let alias = format!("{output}{suffix}");
                let input = if index.is_empty() {
                    alias
                } else {
                    let input = format!("{output}.ignored");
                    std::os::unix::fs::symlink(alias, format!("{input}{index}")).unwrap();
                    input
                };
                let mut args =
                    vec!["score", "--vcf", &vcf, "--models", &model, "-o", &output, option, &input];
                let missing = setup.path("missing");
                match option {
                    "--alignments" => args.extend(["--reference", &missing]),
                    "--reference" => args.extend(["--alignments", &missing]),
                    _ => {}
                }
                for _ in 0..2 {
                    try_run(&args).unwrap();
                    let scored = read_gzip(Path::new(&output));
                    assert_eq!(
                        scored
                            .lines()
                            .filter(|r| !r.starts_with('#') && r.contains("WINNOWR_P="))
                            .count(),
                        CALLS
                    );
                }
            }
        }

        #[rstest]
        #[case::pileup(false)]
        #[case::strlen(true)]
        fn consumed_evidence_is_protected_before_loading(#[case] strlen: bool) {
            let (dir, model_path) = if strlen {
                let setup = VcfSourceModel::new();
                let model = setup.model();
                (setup.dir, model)
            } else {
                let setup = AlignmentsModel::new();
                let model = setup.path("reads.winnowr.json");
                (setup.dir, model)
            };
            let path = |name: &str| dir.path().join(name).display().to_string();
            let mut model = WinnowrModel::load(&model_path, NonZeroUsize::new(1).unwrap()).unwrap();
            model.info.feature_source = FeatureSource::Vcf;
            let table_option = if strlen {
                model.info.stutter = Some(crate::testing::fit_stutter_model(&[]));
                "--strlen"
            } else {
                "--pileup"
            };
            model.save(&model_path).unwrap();
            let (vcf, protected, bam, fasta) = (
                path("default.vcf.gz"),
                path("protected"),
                path("missing.bam"),
                path("missing.fa"),
            );
            std::fs::write(&protected, b"not a readable evidence input").unwrap();
            std::os::unix::fs::symlink(&protected, format!("{bam}.bai")).unwrap();
            std::os::unix::fs::symlink(&protected, format!("{fasta}.fai")).unwrap();
            for what in ["table", "alignments", "reference", "alignments index", "reference index"]
            {
                for suffix in ["", ".tbi"] {
                    let output = path(&format!("{}{}.vcf.gz", what.replace(' ', "_"), suffix));
                    std::os::unix::fs::symlink(&protected, format!("{output}{suffix}")).unwrap();
                    let mut args =
                        vec!["score", "--vcf", &vcf, "--models", &model_path, "-o", &output];
                    let expected = if what == "table" {
                        args.extend([table_option, &protected]);
                        format!("{} table", if strlen { "strlen" } else { "pileup" })
                    } else {
                        let reads = if what == "alignments" { &protected } else { &bam };
                        let reference = if what == "reference" { &protected } else { &fasta };
                        // Only the selected index aliases the protected file.
                        if what == "reference index" {
                            std::fs::remove_file(format!("{bam}.bai")).unwrap();
                        }
                        args.extend(["--alignments", reads, "--reference", reference]);
                        what.to_string()
                    };
                    let error = format!("{:#}", try_run(&args).unwrap_err());
                    assert!(
                        error.ends_with(&format!("is the input {expected}; write to another path")),
                        "{error}"
                    );
                    assert_eq!(
                        std::fs::read(&protected).unwrap(),
                        b"not a readable evidence input"
                    );
                    if what == "reference index" {
                        std::os::unix::fs::symlink(&protected, format!("{bam}.bai")).unwrap();
                    }
                }
            }
        }

        /// A --gvcf the gVCF source reads is checked against the output before it is read, and
        /// is kept.
        #[test]
        fn a_gvcf_that_is_read_is_not_the_output() {
            let setup = VcfSourceModel::new();
            let (vcf, gvcf) = (setup.path("default.vcf.gz"), setup.path("S1.vcf.parquet"));
            let before = std::fs::read(&gvcf).unwrap();
            let extract = ["extract", "--vcf", &vcf, "--gvcf", &gvcf, "--contigs", "chr1"];
            let error = try_run(&[&extract[..], &["-o", &gvcf]].concat()).unwrap_err();
            let message = format!("{error:#}");
            assert!(message.ends_with("is the input gVCF; write to another path"), "{message}");
            assert!(std::fs::read(&gvcf).unwrap() == before);
        }

        /// evaluate rejects a table of another feature source than the model's.
        #[test]
        fn evaluate_rejects_a_table_of_another_feature_source() {
            let setup = VcfSourceModel::new();
            let (table, model) = (setup.extract("S2", "calls.vcf.gz", "gvcf"), setup.model());
            let metrics = setup.path("metrics.tsv");
            let error =
                try_run(&["evaluate", "--models", &model, "--table", &table, "-o", &metrics])
                    .unwrap_err();
            let expected = format!(
                "feature table {table} holds features from `gvcf`, but model {model} was \
                 trained on features from `vcf`: extract the table again with --feature-source \
                 vcf, or use a model trained on `gvcf` features"
            );
            assert_eq!(error.to_string(), expected);
        }

        /// The reads [`write_reads`] writes at call `i`: their number, how many of them have the
        /// alternate allele, and their MAPQ. At a true call 10 of 20 reads have the alternate
        /// allele, at MAPQ 60; at a false one 2 of 30, at MAPQ 30.
        fn default_reads(i: usize) -> (usize, usize, u8) {
            if is_true(i) { (20, 10, 60) } else { (30, 2, 30) }
        }

        /// Writes a reference of chr1 (`ref.fa`, all `A`) and reads over the calls of
        /// [`write_calls`] (`reads.bam`), as `reads` says for each call (see [`default_reads`]).
        fn write_reads(dir: &Path, reads: &dyn Fn(usize) -> (usize, usize, u8)) {
            use crate::testing::TestRead;
            let reference = "A".repeat(100_000);
            crate::testing::write_fasta(dir, &[("chr1", &reference)]);
            // A 50-base read with the alternate base at its 26th base.
            let alt = format!("{}C{}", "A".repeat(25), "A".repeat(24));
            let specs: Vec<(usize, String, bool, u8)> = (0..CALLS)
                .flat_map(|i| {
                    let (n_reads, n_alt, mapq) = reads(i);
                    (0..n_reads).map(move |k| (i, format!("read{i}_{k}"), k < n_alt, mapq))
                })
                .collect();
            let reads: Vec<TestRead<'_>> = specs
                .iter()
                .map(|(i, name, is_alt, mapq)| TestRead {
                    name,
                    // The call's POS is 1-based.
                    pos: (1000 + 200 * i) as i64 - 26,
                    cigar: "50M",
                    flag: 0,
                    mapq: *mapq,
                    qual: 30,
                    quals: None,
                    bases: is_alt.then_some(alt.as_str()),
                })
                .collect();
            crate::testing::write_bam_with_header(
                dir,
                "reads.bam",
                reference.len() as u64,
                &reads,
                &[ALIGNER_RECORD],
            );
        }

        /// The `@PG` record of the aligner of [`write_reads`]' reads.
        const ALIGNER_RECORD: &str = "@PG\tID:bwa-mem3\tPN:bwa-mem3\tVN:0.16.0\tCL:bwa-mem3 mem \
                                      --alt-project primary -t 8 ref.fa reads.fq";

        /// `extract --alignments` stops when no read is counted at any call: the reads are not
        /// of the calls, or none passes the alignment settings.
        #[rstest]
        #[case::no_reads(&|_| (0, 0, 60))]
        #[case::no_mapq(&|_| (20, 10, 255))]
        #[case::below_the_minimum_mapq(&|_| (20, 10, 10))]
        fn extract_needs_a_read_counted(#[case] reads: &dyn Fn(usize) -> (usize, usize, u8)) {
            let dir = tempfile::TempDir::new().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            write_default_mode_calls(dir.path());
            write_reads(dir.path(), reads);
            let error = try_run(&[
                "extract",
                "--vcf",
                &path("default.vcf.gz"),
                "--alignments",
                &path("reads.bam"),
                "--reference",
                &path("ref.fa"),
                "--contigs",
                "chr1",
                "-o",
                &path("t.parquet"),
            ])
            .unwrap_err();
            assert!(
                format!("{error:#}").contains(&format!(
                    "no read was counted at any of the {CALLS} calls with alignment settings"
                )),
                "{error:#}"
            );
        }

        /// The calls of [`write_default_mode_calls`] with the reads of [`write_reads`] and their
        /// pileup table in a temporary directory, with tables of S1 and S2 measured from the reads
        /// (`S1.parquet`, `S2.parquet`, with pileup features) and a snp model trained on them
        /// (`reads.winnowr.json`).
        struct AlignmentsModel {
            dir: tempfile::TempDir,
        }

        impl AlignmentsModel {
            fn new() -> Self {
                Self::new_with_tags(&[])
            }

            /// Supplies verified settings when a test needs known rather than lossy BWA options.
            fn new_with_tags(tags: &[&str]) -> Self {
                let setup = Self { dir: tempfile::TempDir::new().unwrap() };
                write_default_mode_calls(setup.dir.path());
                write_reads(setup.dir.path(), &default_reads);
                let (vcf, pileup) = (setup.path("default.vcf.gz"), setup.path("pileup.tsv.gz"));
                let base = ["pileup", "--vcf", &vcf, "--regions", "chr1", "-o", &pileup];
                let reads = setup.reads();
                let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
                run(&[&base[..], &reads, tags].concat());
                let with_pileup = [&["--pileup", pileup.as_str()][..], tags].concat();
                let tables = [
                    setup.extract("S1", "S1", &with_pileup),
                    setup.extract("S2", "S2", &with_pileup),
                ];
                setup.train(&tables, "reads").unwrap();
                setup
            }

            fn path(&self, name: &str) -> String {
                self.dir.path().join(name).display().to_string()
            }

            /// The options giving the reads.
            fn reads(&self) -> [String; 4] {
                let (bam, fasta) = (self.path("reads.bam"), self.path("ref.fa"));
                ["--alignments".to_string(), bam, "--reference".to_string(), fasta]
            }

            /// Extracts the calls of `sample` from the reads, with `extra` options, as `name`.
            fn extract(&self, sample: &str, name: &str, extra: &[&str]) -> String {
                let table = self.path(&format!("{name}.parquet"));
                let (vcf, happy, reads) =
                    (self.path("default.vcf.gz"), self.path("happy.vcf"), self.reads());
                let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
                let base = ["extract", "--vcf", &vcf, "--happy", &happy];
                let rest = ["--contigs", "chr1", "--sample", sample, "-o", &table];
                run(&[&base[..], &rest, &reads, extra].concat());
                table
            }

            /// Trains a snp model with pileup features on `tables`, as `output`.
            fn train<S: AsRef<str>>(&self, tables: &[S], output: &str) -> Result<()> {
                let output = self.path(output);
                let tables: Vec<&str> = tables.iter().map(AsRef::as_ref).collect();
                let base = ["train", "--variant-type", "snp", "--max-rounds", "10", "-o", &output];
                try_run(&[&base[..], &["--pileup-features", "--tables"], &tables].concat())
            }

            /// Scores the calls with the model, with `extra` options, as `output`.
            fn score(&self, output: &str, extra: &[&str]) -> Result<()> {
                let (vcf, model, output) = (
                    self.path("default.vcf.gz"),
                    self.path("reads.winnowr.json"),
                    self.path(output),
                );
                try_run(
                    &[&["score", "--vcf", &vcf, "--models", &model, "-o", &output][..], extra]
                        .concat(),
                )
            }
        }

        /// Features measured from the reads (`extract --alignments`) train a model that records
        /// the alignments source and its read settings, which `score` measures from the reads
        /// with no --gvcf, in one pass with the model's pileup evidence.
        #[test]
        fn the_alignments_feature_source_trains_and_scores_from_the_reads() {
            let setup = AlignmentsModel::new();
            let default = FeatureSource::VcfAlignments(ReadAnnotationOptions::default());
            let table = FeatureTable::read_parquet(setup.path("S1.parquet")).unwrap();
            assert_eq!(table.info().feature_source, default);
            // The reads' mapping qualities, allele shares and depth, not the VCF's.
            let column = |name: &str| table.f64_column(name).unwrap();
            let expected = |of_true: f64, of_false: f64| -> Vec<f64> {
                (0..CALLS).map(|i| if is_true(i) { of_true } else { of_false }).collect()
            };
            // The VCF's INFO/MQ is 60 and 20, and its INFO/DP 20 at every call.
            assert_eq!(column("mq"), expected(60.0, 30.0));
            assert_eq!(column("as_alt_frac"), expected(0.5, 2.0 / 30.0));
            // The median of 20 and 30 reads is 25.
            assert_eq!(column("dp_norm"), expected(0.8, 1.2));

            let model_path = setup.path("reads.winnowr.json");
            let model = WinnowrModel::load(&model_path, NonZeroUsize::new(1).unwrap()).unwrap();
            assert_eq!(model.info.feature_source, default);
            assert_eq!(model.info.pileup_settings, Some(PileupOptions::default()));
            let json = std::fs::read_to_string(&model_path).unwrap();
            let recorded = r#""feature_source":"alignments:mapq20,bq10,flank5""#;
            assert!(json.contains(recorded), "{json}");

            // score measures the reads, with no gVCF.
            let reads = setup.reads();
            let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
            setup.score("scored.vcf.gz", &reads).unwrap();
            let scored = read_gzip(&setup.dir.path().join("scored.vcf.gz"));
            let records: Vec<&str> = scored.lines().filter(|l| !l.starts_with('#')).collect();
            assert_eq!(records.iter().filter(|r| r.contains("WINNOWR_P=")).count(), CALLS);
            // Calls are dropped, and only false ones: the reads tell them apart.
            let dropped: Vec<usize> = records
                .iter()
                .enumerate()
                .filter(|(_, record)| record.split('\t').nth(6) == Some("WinnowrDrop"))
                .map(|(i, _)| i)
                .collect();
            assert!(!dropped.is_empty(), "{scored}");
            assert!(dropped.iter().all(|&i| !is_true(i)), "{dropped:?}");
        }

        /// Set, to the test's name, in the process a test runs in on its own (see
        /// [`in_own_process`]): a value set outside for another purpose does not match.
        const OWN_PROCESS: &str = "WINNOWR_TEST_IN_OWN_PROCESS";

        /// Whether the calling test, `name` (in this module), runs in a process of its own: if
        /// not, runs it in one (the test binary run again for it alone), and fails unless it
        /// passes there. For a test that sets up the global rayon pool, which a process can set
        /// up only once, before anything uses it.
        fn in_own_process(name: &str) -> bool {
            if std::env::var_os(OWN_PROCESS).is_some_and(|value| value == name) {
                return true;
            }
            // The test's path from the crate root, as the test binary names it.
            let module = module_path!().split_once("::").unwrap().1;
            let dir = tempfile::TempDir::new().unwrap();
            let mut processes = crate::testing::TestProcesses::default();
            processes.spawn_test(
                &format!("{module}::{name}"),
                [(OWN_PROCESS, name)],
                &dir.path().join("child.log"),
            );
            processes.wait_for_success(std::time::Duration::from_secs(30));
            false
        }

        /// `score --alignments` with more than one thread, called as `run` calls it (from outside
        /// the global pool, of -t threads), measures the reads while a model that does not need
        /// them scores, and writes the calls as with one thread.
        #[test]
        fn scoring_from_the_reads_on_threads_shared_writes_as_on_one() {
            if !in_own_process("scoring_from_the_reads_on_threads_shared_writes_as_on_one") {
                return;
            }
            // The global pool, of -t threads as `run` sizes it.
            configure_threads(NonZeroUsize::new(4).unwrap()).unwrap();
            let threads = rayon::current_num_threads();
            assert_eq!(threads, 4);
            let setup = AlignmentsModel::new();
            let vcf = setup.path("default.vcf.gz");
            // Tables of the VCF feature source with pileup evidence, whose models' pileup
            // evidence `score` counts from the reads while scoring (not before the table is
            // built, as for the alignments feature source).
            let tables: Vec<String> = ["S1", "S2"]
                .iter()
                .map(|sample| {
                    let table = setup.path(&format!("{sample}.vcf.parquet"));
                    let (happy, pileup) = (setup.path("happy.vcf"), setup.path("pileup.tsv.gz"));
                    let base = ["extract", "--vcf", &vcf, "--happy", &happy, "--contigs", "chr1"];
                    let rest = ["--sample", sample, "--feature-source", "vcf", "--pileup", &pileup];
                    run(&[&base[..], &rest, &["-o", &table]].concat());
                    table
                })
                .collect();
            let train = |output: &str, extra: &[&str]| {
                let base = ["train", "--variant-type", "snp", "--max-rounds", "10", "-o", output];
                let tables = ["--tables", &tables[0], &tables[1]];
                run(&[&base[..], &tables, extra].concat());
                format!("{output}.winnowr.json")
            };
            // A model that does not need the reads' evidence (no pileup features)...
            let plain = train(&setup.path("plain"), &[]);
            // ... and one that does, of another variant type (scoring none of the calls).
            let snp = train(&setup.path("pileup"), &["--pileup-features"]);
            let mut needing = WinnowrModel::load(&snp, NonZeroUsize::MIN).unwrap();
            assert!(needing.info.pileup_settings.is_some());
            needing.info.variant_type = VariantType::Indel;
            let needing_path = setup.path("indel.winnowr.json");
            needing.save(&needing_path).unwrap();
            let reads = setup.reads();
            let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
            let score = |output: &str| {
                let output = setup.path(output);
                let base =
                    ["score", "--vcf", &vcf, "--models", &plain, &needing_path, "-o", &output];
                parse(&[&base[..], &reads].concat()).unwrap().command
            };
            let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
            let command = score("one.vcf.gz");
            one.install(|| dispatch(&command, NonZeroUsize::MIN)).unwrap();
            dispatch(&score("shared.vcf.gz"), NonZeroUsize::new(threads).unwrap()).unwrap();
            let records = |name: &str| -> Vec<String> {
                let text = read_gzip(&setup.dir.path().join(name));
                text.lines().filter(|line| !line.starts_with('#')).map(str::to_string).collect()
            };
            let shared = records("shared.vcf.gz");
            assert_eq!(shared, records("one.vcf.gz"));
            assert_eq!(shared.iter().filter(|r| r.contains("WINNOWR_P=")).count(), CALLS);
            crate::testing::child_body_ran();
        }

        /// An output that is an input of `score` or `extract` (the VCF, the reads, the
        /// reference, ..., or the index of one), or whose index `score` would write over one,
        /// fails the run before the reads are read, and the input is kept.
        #[test]
        fn the_output_is_checked_before_the_reads() {
            let setup = AlignmentsModel::new();
            let (vcf, model, happy) = (
                setup.path("default.vcf.gz"),
                setup.path("reads.winnowr.json"),
                setup.path("happy.vcf"),
            );
            let (bam, missing, fasta) =
                (setup.path("reads.bam"), setup.path("missing.bam"), setup.path("ref.fa"));
            let score = |output: &str, bam: &str| {
                let base = ["score", "--vcf", &vcf, "--models", &model, "-o", output];
                try_run(&[&base[..], &["--alignments", bam, "--reference", &fasta]].concat())
            };
            let extract = |output: &str, bam: &str| {
                let base = ["extract", "--vcf", &vcf, "--happy", &happy, "--contigs", "chr1"];
                let rest = ["--sample", "S1", "-o", output, "--alignments", bam];
                try_run(&[&base[..], &rest, &["--reference", &fasta]].concat())
            };
            let (tbi, bai, fai) =
                (format!("{vcf}.tbi"), format!("{bam}.bai"), format!("{fasta}.fai"));
            let read = |path: &str| std::fs::read(path).unwrap();
            let all = [&vcf, &bam, &fasta, &tbi, &bai, &fai];
            let before: Vec<Vec<u8>> = all.iter().map(|path| read(path)).collect();
            // The index `score` writes beside its output is checked too.
            let scored = setup.path("scored.vcf.gz");
            std::os::unix::fs::symlink(&bai, format!("{scored}.tbi")).unwrap();
            // The reads are missing, which the pass over them would fail on.
            let cases = [
                (score(&vcf, &missing), "VCF"),
                (extract(&vcf, &missing), "VCF"),
                (score(&bam, &bam), "alignments"),
                (extract(&fasta, &bam), "reference"),
                (extract(&tbi, &missing), "VCF index"),
                (score(&tbi, &missing), "VCF index"),
                (score(&bai, &bam), "alignments index"),
                (extract(&fai, &bam), "reference index"),
                (score(&scored, &bam), "alignments index"),
            ];
            for (result, what) in cases {
                let message = format!("{:#}", result.unwrap_err());
                let expected = format!("is the input {what}; write to another path");
                assert!(message.ends_with(&expected), "{message}");
            }
            let after: Vec<Vec<u8>> = all.iter().map(|path| read(path)).collect();
            assert!(after == before);
        }

        #[test]
        fn unused_beds_do_not_prevent_scoring_again() {
            let setup = AlignmentsModel::new();
            let reads = setup.reads();
            let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
            for name in ["unused", NOT_DIFFICULT_REGIONS] {
                for suffix in ["", ".tbi"] {
                    let region = format!("{name}={}{suffix}", setup.path("rerun.vcf.gz"));
                    let options = [&reads[..], &["--regions", &region]].concat();
                    setup.score("rerun.vcf.gz", &options).unwrap();
                    setup.score("rerun.vcf.gz", &options).unwrap();
                }
            }
        }

        /// A BED only the policy reads is read before the reads are: a missing one fails the
        /// run first, though the reads would fail the pass over them.
        #[test]
        fn a_policy_bed_is_read_before_the_reads() {
            let setup = AlignmentsModel::new();
            // The reads' header is read (for their alignment provenance) before the BEDs, but a
            // pass over the reads needs their index, which this copy does not have.
            let (bam, fasta, bed) =
                (setup.path("unindexed.bam"), setup.path("ref.fa"), setup.path("missing.bed"));
            std::fs::copy(setup.path("reads.bam"), &bam).unwrap();
            let regions = format!("{NOT_DIFFICULT_REGIONS}={bed}");
            let options = [
                "--alignments",
                &bam,
                "--reference",
                &fasta,
                "--abstain-actions",
                "flip",
                "--regions",
                &regions,
            ];
            let error = setup.score("policy.vcf.gz", &options).unwrap_err();
            let message = format!("{error:#}");
            assert!(
                message.contains("missing.bed") && !message.contains("unindexed.bam"),
                "{message}"
            );
        }

        #[rstest]
        #[case::metadata(true)]
        #[case::rows(false)]
        fn pr38_overrides_apply_before_evidence_shard_agreement(#[case] metadata: bool) {
            let setup = AlignmentsModel::new();
            let evidence = read_tables::<PileupOptions>(&[setup.path("pileup.tsv.gz")]).unwrap();
            // The shards conflict: one records primary, the other keep-alt.
            let shard = |name: &str, alt_project: &str| {
                let path = setup.path(name);
                let mut alignment = evidence.alignment.clone().unwrap();
                alignment.0.insert("alt-project".to_string(), alt_project.to_string());
                crate::evidence::write_table::<PileupOptions>(
                    Path::new(&path),
                    evidence.options,
                    Some(&alignment),
                    evidence.rows.iter().map(crate::pileup::PileupRow::to_line),
                )
                .unwrap();
                path
            };
            let (first, other) = (shard("first.tsv", "primary"), shard("other.tsv", "keep-alt"));
            assert!(read_tables::<PileupOptions>(&[&first, &other]).is_err());
            // The effective provenance of the shards with `tags` (none: no overrides).
            let effective = |tags: Option<&str>| -> Result<Option<AlignmentProvenance>> {
                let tags: Vec<&str> =
                    tags.map(|tags| vec!["--alignment-tags", tags]).unwrap_or_default();
                if metadata {
                    let base =
                        ["score", "--vcf", "v", "--models", "m", "-o", "o", "--pileup", &first];
                    let Command::Score(args) =
                        parse(&[&base[..], &[other.as_str()], &tags].concat()).unwrap().command
                    else {
                        panic!("not score")
                    };
                    Ok(score_reads_alignment(&args, RequiredTables::default())?.0.provenance)
                } else {
                    let output = setup.path("overridden.parquet");
                    let (vcf, happy, reads) =
                        (setup.path("default.vcf.gz"), setup.path("happy.vcf"), setup.reads());
                    let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
                    let base = ["extract", "--vcf", &vcf, "--happy", &happy, "--contigs", "chr1"];
                    let rest = ["--sample", "S3", "-o", &output, "--pileup", &first, &other];
                    try_run(&[&base[..], &rest, &reads, &tags].concat())?;
                    Ok(FeatureTable::read_info(&output)?.alignment)
                }
            };
            let error = format!("{:#}", effective(None).unwrap_err());
            assert!(error.contains("aligned differently"), "{error}");
            let set = effective(Some("alt-project=primary")).unwrap().unwrap();
            assert_eq!(set.get("alt-project"), Some("primary"));
            let removed = effective(Some("alt-project!")).unwrap().unwrap();
            assert_eq!(removed.get("alt-project"), None);
            assert_eq!(removed.get("aligner"), Some("bwa-mem3"));
            assert_eq!(effective(Some("none")).unwrap(), None);
        }

        #[rstest]
        #[case::known_changes(Some("primary"), Some("keep-alt"), false)]
        #[case::known_disappears(Some("primary"), None, false)]
        #[case::unknown_becomes_known(None, Some("primary"), false)]
        #[case::mixed_training(Some("primary"), Some("keep-alt"), true)]
        fn pr38_training_rejects_provenance_changed_since_preflight(
            #[case] before: Option<&str>,
            #[case] after: Option<&str>,
            #[case] mixed: bool,
        ) {
            let tables = MixedTables::new(&[]);
            let path = tables.extract("S4", &["chr1", "chr2"]);
            let mut table = FeatureTable::read_parquet(&path).unwrap();
            table.info_mut().alignment = before.map(crate::testing::bwa_provenance);
            table.write_parquet(&path).unwrap();
            let expected = FeatureTable::read_info(&path).unwrap();
            let other = if mixed {
                Some(crate::testing::bwa_provenance("keep-alt"))
            } else {
                expected.alignment.clone()
            };
            training_provenance(
                &[("S4", expected.alignment.as_ref()), ("S5", other.as_ref())],
                mixed,
            )
            .unwrap();
            table.info_mut().alignment = after.map(crate::testing::bwa_provenance);
            table.write_parquet(&path).unwrap();
            let error =
                read_training_table(Path::new(&path), &TableSelection::default(), &expected)
                    .unwrap_err();
            assert!(error.to_string().contains("alignment provenance"), "{error:#}");
        }

        /// A training table whose region digests changed between preflight (which chose the
        /// digests the model records) and its reload is rejected; an unchanged table, with or
        /// without digests, is read.
        #[rstest]
        #[case::digest_changes(Some(&[("chr1", 0, 10)][..]), Some(&[("chr1", 0, 20)][..]), true)]
        #[case::digest_appears(None, Some(&[("chr1", 0, 10)][..]), true)]
        #[case::digest_disappears(Some(&[("chr1", 0, 10)][..]), None, true)]
        #[case::unchanged(Some(&[("chr1", 0, 10)][..]), Some(&[("chr1", 0, 10)][..]), false)]
        #[case::legacy_unchanged(None, None, false)]
        fn training_rejects_region_digests_changed_since_preflight(
            #[case] before: Option<&[(&str, u64, u64)]>,
            #[case] after: Option<&[(&str, u64, u64)]>,
            #[case] rejected: bool,
        ) {
            let digests = |spans: Option<&[(&str, u64, u64)]>| -> BTreeMap<String, RegionDigest> {
                spans
                    .map(|spans| {
                        let digest = RegionSet::from_spans("segdups", spans.iter().copied());
                        BTreeMap::from([("segdups".to_string(), digest.digest().unwrap())])
                    })
                    .unwrap_or_default()
            };
            let tables = MixedTables::new(&[]);
            let path = tables.extract("S4", &["chr1", "chr2"]);
            let mut table = FeatureTable::read_parquet(&path).unwrap();
            table.info_mut().region_digests = digests(before);
            table.write_parquet(&path).unwrap();
            let expected = FeatureTable::read_info(&path).unwrap();
            table.info_mut().region_digests = digests(after);
            table.write_parquet(&path).unwrap();
            let read = read_training_table(Path::new(&path), &TableSelection::default(), &expected);
            if rejected {
                let error = read.unwrap_err().to_string();
                assert!(error.contains("region BED digests"), "{error}");
                assert!(error.contains("since training preflight"), "{error}");
            } else {
                read.unwrap();
            }
        }

        #[rstest]
        #[case::known_changes(Some("primary"), Some("keep-alt"), None, true, false)]
        #[case::known_disappears(Some("primary"), None, None, true, false)]
        #[case::unknown_becomes_known(None, Some("primary"), None, true, false)]
        #[case::warns(Some("primary"), Some("keep-alt"), None, false, true)]
        #[case::overridden(
            Some("primary"),
            Some("keep-alt"),
            Some("alt-project=primary"),
            true,
            true
        )]
        #[case::removed(Some("primary"), Some("keep-alt"), Some("alt-project!"), true, true)]
        #[case::disabled(Some("primary"), Some("keep-alt"), Some("none"), true, true)]
        fn pr38_scoring_checks_each_loaded_evidence_file(
            #[case] before: Option<&str>,
            #[case] after: Option<&str>,
            #[case] tags: Option<&str>,
            #[case] strict: bool,
            #[case] accepted: bool,
        ) {
            use crate::evidence::{AlignmentObservation, observe_alignment_with_tags, write_table};
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("evidence.tsv");
            let tags = AlignmentTags::parse(&tags.into_iter().collect::<Vec<_>>()).unwrap();
            let write = |value: Option<&str>| {
                write_table::<PileupOptions>(
                    &path,
                    PileupOptions::default(),
                    value.map(crate::testing::bwa_provenance).as_ref(),
                    std::iter::empty(),
                )
                .unwrap();
            };
            write(before);
            // The snapshot as scoring takes it: only of a table observed as a regular file.
            let checked = match observe_alignment_with_tags(&path, tags.as_ref()).unwrap() {
                AlignmentObservation::Regular(alignment) => vec![(path.clone(), alignment)],
                AlignmentObservation::Skipped => Vec::new(),
            };
            write(after);
            let model = crate::testing::separable_model_of(
                Task::Genotype,
                VariantType::Snp,
                "pu_alt_mqhigh",
            );
            let result = read_for_models::<PileupOptions>(
                std::slice::from_ref(&path),
                &[model],
                &[PathBuf::from("model.json")],
                tags.as_ref(),
                |path, actual| check_loaded_alignment(path, actual, &checked, strict),
            );
            assert_eq!(result.is_ok(), accepted, "{result:?}");
        }

        #[test]
        fn pr38_a_stream_is_not_compared_with_preflight_provenance() {
            let checked = vec![(
                PathBuf::from("regular.tsv"),
                Some(crate::testing::bwa_provenance("primary")),
            )];
            let streamed = crate::testing::bwa_provenance("keep-alt");
            // The final resolution of the loaded evidence compares the stream instead.
            check_loaded_alignment(Path::new("evidence.fifo"), Some(&streamed), &checked, true)
                .unwrap();
            let error =
                check_loaded_alignment(Path::new("regular.tsv"), Some(&streamed), &checked, true)
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("changed alignment provenance since scoring preflight"));
        }

        /// Paths that display alike (a non-UTF-8 byte and a literal replacement character) keep
        /// their own snapshots.
        #[cfg(unix)]
        #[test]
        fn pr38_snapshots_are_keyed_by_native_path() {
            use std::os::unix::ffi::OsStrExt as _;
            let lossy = PathBuf::from(std::ffi::OsStr::from_bytes(b"evidence-\xff.tsv"));
            let literal = PathBuf::from("evidence-\u{fffd}.tsv");
            assert_eq!(lossy.display().to_string(), literal.display().to_string());
            let (primary, keep_alt) = (
                crate::testing::bwa_provenance("primary"),
                crate::testing::bwa_provenance("keep-alt"),
            );
            let checked = vec![(literal.clone(), Some(primary)), (lossy.clone(), Some(keep_alt))];
            let keep_alt = checked[1].1.as_ref();
            check_loaded_alignment(&lossy, keep_alt, &checked, true).unwrap();
            assert!(check_loaded_alignment(&literal, keep_alt, &checked, true).is_err());
            // Only the other path was observed, so this one's read may still establish provenance.
            let lossy = std::slice::from_ref(&lossy);
            assert!(required_alignment_pending(&[(lossy, true)], &checked[..1]));
        }

        /// Strict deferral follows what preflight observed: a skipped stream stays pending even if
        /// its path is now a regular file, and an observed file never becomes pending.
        #[test]
        fn pr38_strict_deferral_uses_the_preflight_observation() {
            let dir = tempfile::tempdir().unwrap();
            let replaced = dir.path().join("replaced.tsv");
            std::fs::write(&replaced, "").unwrap();
            let observed = dir.path().join("observed.tsv");
            let checked = vec![(observed.clone(), None)];
            let (replaced, observed) =
                (std::slice::from_ref(&replaced), std::slice::from_ref(&observed));
            assert!(required_alignment_pending(&[(replaced, true)], &checked));
            assert!(!required_alignment_pending(&[(replaced, false)], &checked));
            assert!(!required_alignment_pending(&[(observed, true)], &checked));
            assert!(required_alignment_pending(&[(observed, true), (replaced, true)], &checked));
        }

        /// A real pileup-feature model and VCF for exercising the complete scoring flow.
        fn scoring_evidence_inputs(dir: &Path, alignment: Option<AlignmentProvenance>) {
            write_default_mode_calls(dir);
            let mut model = crate::testing::separable_model_of(
                Task::Genotype,
                VariantType::Snp,
                "pu_alt_mqhigh",
            );
            model.info.feature_source = FeatureSource::Vcf;
            model.info.alignment = alignment;
            model.save(dir.join("model.winnowr.json")).unwrap();
        }

        fn scoring_pileup_text(alignment: Option<&AlignmentProvenance>) -> String {
            let alignment = alignment
                .map(|alignment| format!("##alignment={}\n", alignment.to_json()))
                .unwrap_or_default();
            format!(
                "{}default\n{alignment}{}\nchr1\t1000\tA\tC\t3\t0\t0\t2\t0\t0\t1\t0\t0\t0\n",
                crate::pileup::SETTINGS_PREFIX,
                crate::pileup::PILEUP_COLUMNS.join("\t"),
            )
        }

        /// What [`fifo_scoring_child`] runs, in a child process, against a FIFO.
        #[derive(serde::Serialize, serde::Deserialize)]
        enum FifoOperation {
            /// `winnowr` with these arguments.
            Score(Vec<String>),
            /// Replace the consumed BAM with an unread FIFO just before the output audit.
            ReplaceScoredAlignments(Vec<String>),
        }

        #[rstest]
        fn review_invalid_policy_options_fail_before_opening_inputs(
            #[values("score", "evaluate", "stratify")] command: &str,
            #[values("--abstain-actions", "--keep-fallback")] option: &str,
        ) {
            let tables = MixedTables::new(&[]);
            let (model, _) = bundled_model(&tables);
            let output = tables.path("out");
            let input = tables.path("unfed.bed");
            assert!(std::process::Command::new("mkfifo").arg(&input).status().unwrap().success());
            let region = format!("{NOT_DIFFICULT_REGIONS}={input}");
            let table = tables.path("S1.parquet");
            let vcf = tables.path("calls.vcf.gz");
            let happy = tables.path("happy.vcf");
            let mut args = vec![command, "--models", &model, "-o", &output];
            match command {
                "score" => args.extend(["--vcf", &vcf]),
                "evaluate" => args.extend(["--table", &table]),
                _ => args.extend([
                    "--tables",
                    &table,
                    &table,
                    "--truth-names",
                    "v1",
                    "v2",
                    "--truth-beds",
                    &input,
                    &input,
                    "--happy",
                    &happy,
                    &happy,
                ]),
            }
            let action = if option == "--abstain-actions" { "flip" } else { "indel" };
            args.extend(["--regions", &region, option, action, "none"]);
            let mut processes = crate::testing::TestProcesses::default();
            let (result, _) = run_fifo_score(&args, &mut processes);
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.starts_with(&format!("{option} none cannot")), "{error}");
        }

        #[rstest]
        fn review_valid_flip_policy_reads_its_fifo(
            #[values("score", "evaluate", "stratify")] command: &str,
        ) {
            let tables = MixedTables::new(&[]);
            let (model, _) = bundled_model(&tables);
            let (output, input) = (tables.path("out"), tables.path("policy.fifo"));
            let mut processes = crate::testing::TestProcesses::default();
            scoring_fifo_writer(
                &mut processes,
                Path::new(&input),
                "chr1\t0\t100000\nchr2\t0\t100000\n",
            );
            let region = format!("{NOT_DIFFICULT_REGIONS}={input}");
            let (table, vcf, happy, truth) = (
                tables.path("S1.parquet"),
                tables.path("calls.vcf.gz"),
                tables.path("happy.vcf"),
                tables.path("truth.bed"),
            );
            std::fs::write(&truth, b"chr1\t0\t100000\nchr2\t0\t100000\n").unwrap();
            let mut args = vec![command, "--models", &model, "-o", &output];
            match command {
                "score" => args.extend(["--vcf", &vcf]),
                "evaluate" => args.extend(["--table", &table]),
                _ => args.extend([
                    "--tables",
                    &table,
                    &table,
                    "--truth-names",
                    "v1",
                    "v2",
                    "--truth-beds",
                    &truth,
                    &truth,
                    "--happy",
                    &happy,
                    &happy,
                ]),
            }
            args.extend(["--regions", &region, "--abstain-actions", "flip"]);
            let (result, _) = run_fifo_score(&args, &mut processes);
            result.unwrap();
            let written =
                if command == "stratify" { format!("{output}.strata.tsv") } else { output };
            assert!(std::fs::metadata(written).unwrap().len() > 0);
        }

        /// Child of [`run_fifo_operation`]: runs the requested operation, writing its error and
        /// logs as JSON to the response path.
        #[test]
        fn fifo_scoring_child() {
            let Some(request) = std::env::var_os("WINNOWR_FIFO_REQUEST") else { return };
            let operation: FifoOperation =
                serde_json::from_str(&request.to_string_lossy()).unwrap();
            let (result, logs) = captured_logs::capture(|| match operation {
                FifoOperation::Score(args) => {
                    try_run(&args.iter().map(String::as_str).collect::<Vec<_>>())
                }
                FifoOperation::ReplaceScoredAlignments(args) => {
                    let Command::Score(args) =
                        parse(&args.iter().map(String::as_str).collect::<Vec<_>>())
                            .unwrap()
                            .command
                    else {
                        panic!("not score")
                    };
                    score_with(&args, NonZeroUsize::MIN, || {
                        let path = args.alignments.as_ref().unwrap();
                        std::fs::remove_file(path).unwrap();
                        assert!(
                            std::process::Command::new("mkfifo")
                                .arg(path)
                                .status()
                                .unwrap()
                                .success()
                        );
                    })
                }
            });
            let error = result.err().map(|error| format!("{error:#}"));
            let logs: Vec<String> = logs.into_iter().map(|(_, message)| message).collect();
            let response = std::env::var_os("WINNOWR_FIFO_RESPONSE").unwrap();
            std::fs::write(response, serde_json::to_vec(&(error, logs)).unwrap()).unwrap();
            crate::testing::child_body_ran();
        }

        /// Runs `operation` in a child, waiting for it and for the FIFO producers already in
        /// `processes` under one deadline, so that a FIFO opened but never read, or read but
        /// never opened, fails the test instead of hanging it. Returns the operation's result
        /// and log messages.
        fn run_fifo_operation(
            operation: FifoOperation,
            processes: &mut crate::testing::TestProcesses,
        ) -> (Result<()>, Vec<String>) {
            let dir = tempfile::tempdir().unwrap();
            let response = dir.path().join("response.json");
            let log = dir.path().join("child.log");
            let envs = [
                ("WINNOWR_FIFO_REQUEST", serde_json::to_string(&operation).unwrap().into()),
                ("WINNOWR_FIFO_RESPONSE", response.clone().into_os_string()),
            ];
            processes.spawn_test("cli::tests::end_to_end::fifo_scoring_child", envs, &log);
            processes.wait_for_success(std::time::Duration::from_secs(30));
            let (error, logs): (Option<String>, Vec<String>) =
                serde_json::from_slice(&std::fs::read(response).unwrap()).unwrap();
            (error.map_or(Ok(()), |error| Err(anyhow::anyhow!(error))), logs)
        }

        /// [`run_fifo_operation`] for `winnowr` with `args`.
        fn run_fifo_score(
            args: &[&str],
            processes: &mut crate::testing::TestProcesses,
        ) -> (Result<()>, Vec<String>) {
            let args = args.iter().map(|arg| (*arg).to_string()).collect();
            run_fifo_operation(FifoOperation::Score(args), processes)
        }

        /// Makes the FIFO `path` and starts a process (owned by `processes`) that writes `text`
        /// to it once, exiting successfully only if the whole text is read.
        fn scoring_fifo_writer(
            processes: &mut crate::testing::TestProcesses,
            path: &Path,
            text: &str,
        ) {
            assert!(std::process::Command::new("mkfifo").arg(path).status().unwrap().success());
            processes.spawn(
                std::process::Command::new("sh")
                    .args(["-c", "printf '%s' \"$1\" > \"$2\"", "fifo-writer", text])
                    .arg(path),
            );
        }

        #[rstest]
        #[case::fifo_only_match(false, Some("primary"), true, None, true)]
        #[case::fifo_only_mismatch(false, Some("keep-alt"), true, None, false)]
        #[case::fifo_only_unknown(false, None, true, None, false)]
        #[case::fifo_only_warns(false, Some("keep-alt"), false, None, true)]
        #[case::partial_match(true, Some("primary"), true, None, true)]
        #[case::partial_mismatch(true, Some("keep-alt"), true, None, false)]
        #[case::partial_unknown(true, None, true, None, true)]
        #[case::overridden(true, Some("keep-alt"), true, Some("alt-project=primary"), true)]
        #[case::disabled(false, Some("keep-alt"), false, Some("none"), true)]
        #[case::mixed(false, Some("mixed"), true, None, false)]
        #[case::incomplete_strict(false, Some("incomplete"), true, None, false)]
        #[case::incomplete_warns(false, Some("incomplete"), false, None, true)]
        fn scoring_checks_required_fifo_against_models(
            #[case] regular: bool,
            #[case] actual: Option<&str>,
            #[case] strict: bool,
            #[case] tags: Option<&str>,
            #[case] accepted: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            let expected = crate::testing::bwa_provenance(if actual == Some("mixed") {
                "mixed"
            } else {
                "primary"
            });
            scoring_evidence_inputs(dir.path(), Some(expected));
            let partial = crate::testing::provenance(&[("aligner", "bwa-mem3")], true);
            let regular_path = path("regular.tsv");
            std::fs::write(&regular_path, scoring_pileup_text(Some(&partial))).unwrap();
            let fifo = dir.path().join("evidence.fifo");
            let actual = actual.map(|value| {
                if value == "incomplete" {
                    let mut provenance = crate::testing::bwa_provenance("primary");
                    provenance.0.insert("options-inference".to_string(), "incomplete".to_string());
                    provenance
                } else {
                    crate::testing::bwa_provenance(value)
                }
            });
            let mut processes = crate::testing::TestProcesses::default();
            scoring_fifo_writer(&mut processes, &fifo, &scoring_pileup_text(actual.as_ref()));
            let (vcf, model, output) =
                (path("default.vcf.gz"), path("model.winnowr.json"), path("scored.vcf.gz"));
            let fifo = fifo.display().to_string();
            let mut args =
                vec!["score", "--vcf", &vcf, "--models", &model, "-o", &output, "--pileup", &fifo];
            if regular {
                args.push(&regular_path);
            }
            if strict {
                args.push("--strict-provenance");
            }
            if let Some(tags) = tags {
                args.extend(["--alignment-tags", tags]);
            }
            let (result, _) = run_fifo_score(&args, &mut processes);
            assert_eq!(result.is_ok(), accepted, "{result:?}");
            if let Err(error) = result {
                let error = format!("{error:#}");
                assert!(error.contains("alignment") || error.contains("aligned"), "{error}");
            } else {
                assert!(read_gzip(Path::new(&output)).contains("WINNOWR_P="));
            }
        }

        /// `score` reads the required evidence before the region BEDs: one producer that feeds a
        /// `--pileup` stream and then a `--regions` stream finishes, where reading the BED first
        /// would wait forever.
        #[test]
        fn scoring_reads_evidence_streams_before_bed_streams() {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            write_default_mode_calls(dir.path());
            let mut model = crate::testing::separable_model_of(
                Task::Genotype,
                VariantType::Snp,
                "pu_alt_mqhigh",
            );
            model.info.feature_source = FeatureSource::Vcf;
            model.info.region_names = vec!["segdups".to_string()];
            let model_path = path("model.winnowr.json");
            model.save(&model_path).unwrap();
            let (pileup, bed) = (path("pileup.fifo"), path("segdups.fifo"));
            for fifo in [&pileup, &bed] {
                assert!(std::process::Command::new("mkfifo").arg(fifo).status().unwrap().success());
            }
            let mut processes = crate::testing::TestProcesses::default();
            processes.spawn(std::process::Command::new("sh").args([
                "-c",
                "printf '%s' \"$1\" > \"$2\" && printf 'chr1\\t0\\t10000\\n' > \"$3\"",
                "evidence-then-bed",
                &scoring_pileup_text(None),
                &pileup,
                &bed,
            ]));
            let (output, region) = (path("scored.vcf.gz"), format!("segdups={bed}"));
            let args = [
                "score",
                "--vcf",
                &path("default.vcf.gz"),
                "--models",
                &model_path,
                "-o",
                &output,
                "--pileup",
                &pileup,
                "--regions",
                &region,
            ];
            let (result, _) = run_fifo_score(&args, &mut processes);
            result.unwrap();
            assert!(read_gzip(Path::new(&output)).contains("WINNOWR_P="));
        }

        #[test]
        fn scoring_loaded_regular_provenance_replaces_its_snapshot() {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            scoring_evidence_inputs(dir.path(), Some(crate::testing::bwa_provenance("keep-alt")));
            let regular = path("regular.tsv");
            std::fs::write(
                &regular,
                scoring_pileup_text(Some(&crate::testing::bwa_provenance("primary"))),
            )
            .unwrap();
            let replacement =
                scoring_pileup_text(Some(&crate::testing::bwa_provenance("keep-alt")));
            let fifo = path("evidence.fifo");
            assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
            // The group redirection opens the FIFO first: replacing the regular file therefore
            // cannot happen until scoring has finished preflight and starts its required row read.
            let mut processes = crate::testing::TestProcesses::default();
            processes.spawn(std::process::Command::new("sh").args([
                "-c",
                "{ printf '%s' \"$1\" > \"$3\"; printf '%s' \"$1\"; } > \"$2\"",
                "fifo-replacement",
                &replacement,
                &fifo,
                &regular,
            ]));
            let (result, logs) = run_fifo_score(
                &[
                    "score",
                    "--vcf",
                    &path("default.vcf.gz"),
                    "--models",
                    &path("model.winnowr.json"),
                    "-o",
                    &path("scored.vcf.gz"),
                    "--pileup",
                    &fifo,
                    &regular,
                ],
                &mut processes,
            );
            result.unwrap();
            let changed = |name: &str| {
                logs.iter().any(|message| {
                    message.contains(name)
                        && message.contains("changed alignment provenance since scoring preflight")
                })
            };
            // Only the regular file had a preflight snapshot to have changed from.
            assert!(changed("regular.tsv"), "{logs:?}");
            assert!(!changed("evidence.fifo"), "{logs:?}");
            // The stream agrees with the evidence actually loaded: it is not compared with the
            // replaced table's stale preflight provenance.
            assert!(
                !logs.iter().any(|message| message.contains("disagrees with the other reads")),
                "{logs:?}"
            );
            // Without --strict-provenance, the reads are compared with the models once, on the
            // evidence loaded: no mismatch from the replaced table's preflight snapshot.
            assert!(
                !logs.iter().any(|message| message.contains("were not aligned as")),
                "{logs:?}"
            );
            assert!(
                logs.iter().any(|message| message.contains("comparable alignment tags match")
                    && message.contains("alt-project=keep-alt")),
                "{logs:?}"
            );
        }

        #[test]
        fn scoring_required_fifo_retains_unused_regular_provenance() {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            scoring_evidence_inputs(dir.path(), Some(crate::testing::bwa_provenance("primary")));
            std::fs::write(
                path("unused.strlen"),
                format!(
                    "##alignment={}\nunused\n",
                    crate::testing::bwa_provenance("primary").to_json()
                ),
            )
            .unwrap();
            let fifo = dir.path().join("evidence.fifo");
            let mut processes = crate::testing::TestProcesses::default();
            scoring_fifo_writer(
                &mut processes,
                &fifo,
                &scoring_pileup_text(Some(&crate::testing::bwa_provenance("keep-alt"))),
            );
            let (result, _) = run_fifo_score(
                &[
                    "score",
                    "--vcf",
                    &path("default.vcf.gz"),
                    "--models",
                    &path("model.winnowr.json"),
                    "-o",
                    &path("scored.vcf.gz"),
                    "--pileup",
                    fifo.to_str().unwrap(),
                    "--strlen",
                    &path("unused.strlen"),
                ],
                &mut processes,
            );
            assert!(
                result.unwrap_err().to_string().contains(
                    "the --strlen tables and the --pileup tables were aligned differently"
                )
            );
        }

        /// Required regular shards that disagree as loaded fail scoring, strict or not: without
        /// --strict-provenance their preflight conflict is deferred to the loaded tables, which
        /// still reject it.
        #[rstest]
        #[case::strict(true)]
        #[case::deferred(false)]
        fn review_disagreeing_required_shards_fail_once_loaded(#[case] strict: bool) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            scoring_evidence_inputs(dir.path(), Some(crate::testing::bwa_provenance("primary")));
            let (a, b) = (path("a.tsv"), path("b.tsv"));
            let primary = crate::testing::bwa_provenance("primary");
            let keep_alt = crate::testing::bwa_provenance("keep-alt");
            std::fs::write(&a, scoring_pileup_text(Some(&primary))).unwrap();
            std::fs::write(&b, scoring_pileup_text(Some(&keep_alt))).unwrap();
            let (vcf, model, output) =
                (path("default.vcf.gz"), path("model.winnowr.json"), path("scored.vcf.gz"));
            let mut args =
                vec!["score", "--vcf", &vcf, "--models", &model, "-o", &output, "--pileup", &a, &b];
            if strict {
                args.push("--strict-provenance");
            }
            let (result, logs) = captured_logs::capture(|| run_command(&args));
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("were aligned differently"), "{error}");
            assert!(error.contains("alt-project"), "{error}");
            assert!(!Path::new(&output).exists());
            // Without --strict-provenance, scoring defers the conflict of its required shards.
            let deferred = logs.iter().any(|(_, text)| text.contains("compared again as loaded"));
            assert_eq!(deferred, !strict, "{logs:?}");
        }

        /// Without a comparison against preflight provenance, a stream that disagrees with an
        /// unchanged table still fails the resolution of the evidence loaded, strict or not.
        #[rstest]
        #[case::strict(true)]
        #[case::warn(false)]
        fn pr38_a_disagreeing_stream_fails_the_loaded_evidence_merge(#[case] strict: bool) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            scoring_evidence_inputs(dir.path(), Some(crate::testing::bwa_provenance("primary")));
            let regular = path("regular.tsv");
            let primary = crate::testing::bwa_provenance("primary");
            std::fs::write(&regular, scoring_pileup_text(Some(&primary))).unwrap();
            let fifo = dir.path().join("evidence.fifo");
            let mut processes = crate::testing::TestProcesses::default();
            scoring_fifo_writer(
                &mut processes,
                &fifo,
                &scoring_pileup_text(Some(&crate::testing::bwa_provenance("keep-alt"))),
            );
            let (vcf, model, output) =
                (path("default.vcf.gz"), path("model.winnowr.json"), path("scored.vcf.gz"));
            let fifo = fifo.display().to_string();
            let mut args = vec![
                "score", "--vcf", &vcf, "--models", &model, "-o", &output, "--pileup", &fifo,
                &regular,
            ];
            if strict {
                args.push("--strict-provenance");
            }
            let (result, _) = run_fifo_score(&args, &mut processes);
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("were aligned differently"), "{error}");
            assert!(error.contains("alt-project"), "{error}");
        }

        #[test]
        fn scoring_unused_fifo_is_not_opened() {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            scoring_evidence_inputs(dir.path(), Some(crate::testing::bwa_provenance("primary")));
            std::fs::write(
                path("required.tsv"),
                scoring_pileup_text(Some(&crate::testing::bwa_provenance("primary"))),
            )
            .unwrap();
            assert!(
                std::process::Command::new("mkfifo")
                    .arg(path("unused.fifo"))
                    .status()
                    .unwrap()
                    .success()
            );
            let mut processes = crate::testing::TestProcesses::default();
            let (result, logs) = run_fifo_score(
                &[
                    "score",
                    "--vcf",
                    &path("default.vcf.gz"),
                    "--models",
                    &path("model.winnowr.json"),
                    "-o",
                    &path("scored.vcf.gz"),
                    "--pileup",
                    &path("required.tsv"),
                    "--strlen",
                    &path("unused.fifo"),
                    "--strict-provenance",
                ],
                &mut processes,
            );
            result.unwrap();
            assert!(
                !logs.iter().any(|message| message.contains("record no alignment provenance")),
                "{logs:?}"
            );
            assert!(read_gzip(Path::new(&path("scored.vcf.gz"))).contains("WINNOWR_P="));
        }

        #[rstest]
        #[case::conflicting_tags(false, "default", false, false, true)]
        #[case::malformed_tags(true, "default", false, false, true)]
        #[case::different_settings(false, "mapq30", false, false, false)]
        #[case::malformed_settings(true, "invalid", false, false, false)]
        #[case::malformed_header(true, "default", true, false, false)]
        #[case::malformed_row(true, "default", false, true, false)]
        fn scoring_legacy_required_evidence_skips_only_provenance(
            #[case] malformed_tags: bool,
            #[case] settings: &str,
            #[case] bad_header: bool,
            #[case] bad_row: bool,
            #[case] accepted: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            scoring_evidence_inputs(dir.path(), None);
            let first = scoring_pileup_text(Some(&crate::testing::bwa_provenance("primary")));
            std::fs::write(path("first.tsv"), first).unwrap();
            let second = scoring_pileup_text(Some(&crate::testing::bwa_provenance("keep-alt")));
            let mut second = second.replace("settings=default", &format!("settings={settings}"));
            if malformed_tags {
                second = second
                    .lines()
                    .map(|line| {
                        if line.starts_with("##alignment=") { "##alignment=invalid" } else { line }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            if bad_header {
                second = second.replace("chrom\tpos", "wrong\tpos");
            }
            if bad_row {
                second.push_str("\nmalformed-row\n");
            }
            std::fs::write(path("second.tsv"), second).unwrap();
            let result = try_run(&[
                "score",
                "--vcf",
                &path("default.vcf.gz"),
                "--models",
                &path("model.winnowr.json"),
                "-o",
                &path("scored.vcf.gz"),
                "--pileup",
                &path("first.tsv"),
                &path("second.tsv"),
                "--strict-provenance",
            ]);
            assert_eq!(result.is_ok(), accepted, "{result:?}");
            if let Err(error) = result {
                assert!(!format!("{error:#}").contains("alignment"), "{error:#}");
            }
        }

        #[rstest]
        #[case::reads_strict(false, true, true, false, false)]
        #[case::reads_warn(false, true, false, false, true)]
        #[case::model_strict(true, false, true, false, false)]
        #[case::both_strict(true, true, true, false, false)]
        #[case::reads_declared(false, true, true, true, true)]
        #[case::model_stays_incomplete(true, true, true, true, false)]
        fn scoring_incomplete_persisted_provenance_is_not_agreement(
            #[case] model_incomplete: bool,
            #[case] reads_incomplete: bool,
            #[case] strict: bool,
            #[case] declared: bool,
            #[case] accepted: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            let provenance = |incomplete| {
                let mut provenance = crate::testing::bwa_provenance("primary");
                if incomplete {
                    provenance.0.insert("options-inference".to_string(), "incomplete".to_string());
                }
                provenance
            };
            scoring_evidence_inputs(dir.path(), Some(provenance(model_incomplete)));
            std::fs::write(
                path("required.tsv"),
                scoring_pileup_text(Some(&provenance(reads_incomplete))),
            )
            .unwrap();
            let (vcf, model, output, evidence) = (
                path("default.vcf.gz"),
                path("model.winnowr.json"),
                path("scored.vcf.gz"),
                path("required.tsv"),
            );
            let mut args = vec![
                "score",
                "--vcf",
                &vcf,
                "--models",
                &model,
                "-o",
                &output,
                "--pileup",
                &evidence,
                "--alignment-tags",
                "version=9",
            ];
            if declared {
                args.extend(["options-inference=explicit", "alt-project=primary"]);
            }
            if strict {
                args.push("--strict-provenance");
            }
            let (result, logs) = captured_logs::capture(|| try_run(&args));
            assert_eq!(result.is_ok(), accepted, "{result:?}");
            if let Err(error) = result {
                assert!(error.to_string().contains("incomplete alignment settings"), "{error:#}");
            }
            let affirmative =
                logs.iter().any(|(_, message)| message.contains("comparable alignment tags match"));
            assert_eq!(affirmative, declared && !model_incomplete, "{logs:?}");
            if accepted && !declared {
                assert!(
                    logs.iter().any(|(level, message)| *level == log::Level::Warn
                        && message.contains("incomplete alignment settings")),
                    "{logs:?}"
                );
            }
        }

        #[rstest]
        #[case::disjoint(&[("alt-project", "primary")], "aligner=bwa-mem3", false)]
        #[case::comparable(&[("aligner", "bwa-mem3")], "aligner=bwa-mem3", true)]
        #[case::metadata_only(&[("version", "1")], "aligner=bwa-mem3", false)]
        fn scoring_logs_only_actual_comparisons(
            #[case] model_tags: &[(&str, &str)],
            #[case] reads_tags: &str,
            #[case] affirmative: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            write_default_mode_calls(dir.path());
            let mut model =
                crate::testing::separable_model_of(Task::Genotype, VariantType::Snp, "mq");
            model.info.feature_source = FeatureSource::Vcf;
            model.info.alignment = Some(crate::testing::provenance(model_tags, true));
            model.save(path("model.winnowr.json")).unwrap();
            let (result, logs) = captured_logs::capture(|| {
                try_run(&[
                    "score",
                    "--vcf",
                    &path("default.vcf.gz"),
                    "--models",
                    &path("model.winnowr.json"),
                    "-o",
                    &path("scored.vcf.gz"),
                    "--alignment-tags",
                    reads_tags,
                    "--strict-provenance",
                ])
            });
            result.unwrap();
            assert_eq!(
                logs.iter().any(|(_, message)| message.contains("comparable alignment tags match")),
                affirmative,
                "{logs:?}"
            );
            assert_eq!(
                logs.iter().any(|(_, message)| message.contains("no tags were comparable")),
                !affirmative,
                "{logs:?}"
            );
        }

        #[rstest]
        #[case::alignments("--alignments", "alignments")]
        #[case::pileup("--pileup", "pileup table")]
        #[case::strlen("--strlen", "strlen table")]
        fn provenance_only_inputs_are_protected_from_score_outputs(
            #[case] option: &str,
            #[case] what: &str,
            #[values(false, true)] sidecar: bool,
            #[values(false, true)] tags_none: bool,
        ) {
            let setup = VcfSourceModel::new();
            let model_path = setup.model();
            let mut model = WinnowrModel::load(&model_path, NonZeroUsize::new(1).unwrap()).unwrap();
            model.info.alignment = Some(crate::testing::bwa_provenance("primary"));
            model.save(&model_path).unwrap();
            let vcf = setup.path("default.vcf.gz");
            let input = setup.path("provenance-only");
            let output = setup.path("scored.vcf.gz");
            let target = format!("{output}{}", if sidecar { ".tbi" } else { "" });
            let before: &[u8] = if option == "--alignments" {
                b"@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:10000\n"
            } else {
                b"unused\n"
            };
            std::fs::write(&input, before).unwrap();
            std::os::unix::fs::symlink(&input, &target).unwrap();
            let mut args = vec![
                "score",
                "--vcf",
                &vcf,
                "--models",
                &model_path,
                "-o",
                &output,
                option,
                &input,
            ];
            let reference = setup.path("unused.fa");
            if option == "--alignments" {
                args.extend(["--reference", &reference]);
            }
            if tags_none {
                args.extend(["--alignment-tags", "none"]);
                try_run(&args).unwrap();
                assert!(read_gzip(Path::new(&output)).contains("WINNOWR_P="));
            } else {
                let error = format!("{:#}", try_run(&args).unwrap_err());
                assert!(
                    error.ends_with(&format!("is the input {what}; write to another path")),
                    "{error}"
                );
                assert_eq!(std::fs::read(&input).unwrap(), before);
                assert_eq!(std::fs::read(&target).unwrap(), before);
            }
        }

        #[test]
        fn review_measured_reads_log_only_the_preflight_header_result() {
            let mut model =
                crate::testing::separable_model_of(Task::Genotype, VariantType::Snp, "mq");
            model.info.alignment = Some(crate::testing::bwa_provenance("primary"));
            let (_, logs) = captured_logs::capture(|| {
                log_alignment_check(&[&model.info], model.info.alignment.as_ref(), &[], true);
            });
            assert!(
                logs.iter().any(|(_, text)| text.contains("preflight alignment header")),
                "{logs:?}"
            );
            assert!(
                !logs.iter().any(|(_, text)| text.contains("The reads were aligned as")),
                "{logs:?}"
            );
        }

        #[cfg(unix)]
        #[test]
        fn review_final_audit_does_not_reopen_consumed_alignments() {
            let setup = AlignmentsModel::new();
            let args = vec![
                "score".to_string(),
                "--vcf".to_string(),
                setup.path("default.vcf.gz"),
                "--models".to_string(),
                setup.path("reads.winnowr.json"),
                "-o".to_string(),
                setup.path("scored.vcf.gz"),
                "--alignments".to_string(),
                setup.path("reads.bam"),
                "--reference".to_string(),
                setup.path("ref.fa"),
            ];
            let mut processes = crate::testing::TestProcesses::default();
            let (result, _) =
                run_fifo_operation(FifoOperation::ReplaceScoredAlignments(args), &mut processes);
            result.unwrap();
            assert!(read_gzip(Path::new(&setup.path("scored.vcf.gz"))).contains("WINNOWR_P="));
        }

        #[test]
        fn review_partial_agreement_logs_only_comparable_tags() {
            let mut model =
                crate::testing::separable_model_of(Task::Genotype, VariantType::Snp, "mq");
            model.info.alignment = Some(crate::testing::provenance(
                &[("aligner", "bwa-mem3"), ("alt-project", "primary")],
                true,
            ));
            let reads = crate::testing::provenance(&[("aligner", "bwa-mem3")], true);
            for preflight in [false, true] {
                let (_, logs) = captured_logs::capture(|| {
                    log_alignment_check(&[&model.info], Some(&reads), &[], preflight);
                });
                assert!(
                    logs.iter().any(|(_, text)| text.contains("comparable alignment tags match")),
                    "{logs:?}"
                );
                assert!(
                    !logs
                        .iter()
                        .any(|(_, text)| text.contains("aligned as the models' training reads")),
                    "{logs:?}"
                );
            }
        }

        #[rstest]
        #[case::required("--pileup", false)]
        #[case::provenance_only_shards("--strlen", true)]
        #[case::provenance_only_group("--strlen", false)]
        fn review_scoring_missing_provenance_warns_once(
            #[case] option: &str,
            #[case] known_unused_shard: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            let alignment = crate::testing::bwa_provenance("primary");
            scoring_evidence_inputs(dir.path(), Some(alignment.clone()));
            let known = path("known.pileup");
            let missing = path("missing.tsv");
            let unused = path("known.strlen");
            std::fs::write(&known, scoring_pileup_text(Some(&alignment))).unwrap();
            std::fs::write(&unused, format!("##alignment={}\nunused\n", alignment.to_json()))
                .unwrap();
            std::fs::write(
                &missing,
                if option == "--pileup" { scoring_pileup_text(None) } else { "unused\n".into() },
            )
            .unwrap();
            let (vcf, model, output) =
                (path("default.vcf.gz"), path("model.winnowr.json"), path("scored.vcf.gz"));
            let mut arguments =
                vec!["score", "--vcf", &vcf, "--models", &model, "-o", &output, "--pileup", &known];
            if option != "--pileup" {
                arguments.push(option);
            }
            arguments.push(&missing);
            if known_unused_shard {
                arguments.push(&unused);
            }
            let (result, logs) = captured_logs::capture(|| try_run(&arguments));
            result.unwrap();
            let warnings: Vec<_> = logs
                .iter()
                .filter(|(_, text)| text.contains("record no alignment provenance"))
                .collect();
            assert_eq!(warnings.len(), 1, "{logs:?}");
            assert!(read_gzip(Path::new(&output)).contains("WINNOWR_P="));
        }

        #[rstest]
        #[case::still_missing(false)]
        #[case::declaration_added(true)]
        fn review_scoring_required_shard_warns_only_on_loaded_provenance(
            #[case] gains_declaration: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let known = dir.path().join("known.tsv");
            let missing = dir.path().join("missing.tsv");
            let alignment = crate::testing::bwa_provenance("primary");
            let write = |path: &Path, provenance: Option<&AlignmentProvenance>| {
                crate::evidence::write_table::<PileupOptions>(
                    path,
                    PileupOptions::default(),
                    provenance,
                    std::iter::empty(),
                )
                .unwrap();
            };
            write(&known, Some(&alignment));
            write(&missing, None);
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                "v",
                "--models",
                "m",
                "-o",
                "o",
                "--pileup",
                known.to_str().unwrap(),
                missing.to_str().unwrap(),
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let ((reads, checked), preflight) = captured_logs::capture(|| {
                score_reads_alignment(&args, RequiredTables::default()).unwrap()
            });
            if gains_declaration {
                write(&missing, Some(&alignment));
            }
            let (loaded, final_logs) = captured_logs::capture(|| {
                let pileup = crate::evidence::read_tables_checked::<PileupOptions>(
                    &args.pileup,
                    None,
                    |path, actual| check_loaded_alignment(path, actual, &checked, false),
                )
                .unwrap();
                loaded_score_alignment(&args, &reads, &checked, None, Some(&pileup)).unwrap()
            });
            assert_eq!(loaded.provenance, Some(alignment));
            let missing_warnings = |logs: &[(log::Level, String)]| {
                logs.iter()
                    .filter(|(level, text)| {
                        *level == log::Level::Warn
                            && text.contains("record no alignment provenance")
                    })
                    .count()
            };
            assert_eq!(missing_warnings(&preflight), 0, "{preflight:?}");
            assert_eq!(
                missing_warnings(&final_logs),
                usize::from(!gains_declaration),
                "{final_logs:?}"
            );
            if !gains_declaration {
                assert!(
                    final_logs.iter().any(|(_, text)| text.contains("missing.tsv")),
                    "{final_logs:?}"
                );
            }
        }

        #[rstest]
        #[case::required_gains_declaration(true)]
        #[case::provenance_only_stays_unknown(false)]
        fn review_scoring_cross_group_missing_warning_is_final(#[case] required_unknown: bool) {
            use crate::evidence::AlignmentObservation;
            let alignment = crate::testing::bwa_provenance("primary");
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                "v",
                "--models",
                "m",
                "-o",
                "o",
                "--strlen",
                "unused.strlen",
                "--pileup",
                "required.pileup",
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let ((reads, checked), preflight) = captured_logs::capture(|| {
                score_reads_alignment_with(&args, RequiredTables::default(), |path, _| {
                    let unknown = (path == Path::new("required.pileup")) == required_unknown;
                    Ok(AlignmentObservation::Regular((!unknown).then(|| alignment.clone())))
                })
                .unwrap()
            });
            let pileup = PileupTables {
                options: PileupOptions::default(),
                rows: Vec::new(),
                alignment: Some(alignment.clone()),
            };
            let (loaded, final_logs) = captured_logs::capture(|| {
                loaded_score_alignment(&args, &reads, &checked, None, Some(&pileup)).unwrap()
            });
            assert_eq!(loaded.provenance, Some(alignment));
            let warnings = |logs: &[(log::Level, String)]| {
                logs.iter()
                    .filter(|(_, text)| text.contains("record no alignment provenance"))
                    .count()
            };
            assert_eq!(warnings(&preflight), 0, "{preflight:?}");
            assert_eq!(warnings(&final_logs), usize::from(!required_unknown), "{final_logs:?}");
        }

        #[rstest]
        #[case::within_group(false)]
        #[case::across_groups(true)]
        fn review_scoring_preflight_keeps_known_conflicts_early(#[case] across_groups: bool) {
            use crate::evidence::AlignmentObservation;
            let option = if across_groups { "--strlen" } else { "--pileup" };
            let mut arguments = vec![
                "score",
                "--vcf",
                "v",
                "--models",
                "m",
                "-o",
                "o",
                option,
                "primary.tsv",
                "missing.tsv",
            ];
            if across_groups {
                arguments.push("--pileup");
            }
            arguments.push("other.tsv");
            let Command::Score(args) = parse(&arguments).unwrap().command else {
                panic!("not score")
            };
            let (result, logs) = captured_logs::capture(|| {
                score_reads_alignment_with(&args, RequiredTables::default(), |path, _| {
                    Ok(AlignmentObservation::Regular(match path.to_str().unwrap() {
                        "missing.tsv" => None,
                        "primary.tsv" => Some(crate::testing::bwa_provenance("primary")),
                        _ => Some(crate::testing::bwa_provenance("keep-alt")),
                    }))
                })
            });
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains("aligned differently") && error.contains("alt-project"),
                "{error}"
            );
            assert!(
                !logs.iter().any(|(_, text)| text.contains("record no alignment provenance")),
                "{logs:?}"
            );
        }

        /// Without --strict-provenance, snapshots of required shards that disagree (one replaced
        /// between observations) are left to the loaded tables, which replace them; a conflict
        /// with a provenance-only table, or under --strict-provenance, still fails preflight.
        #[rstest]
        #[case::required_lenient(true, false, true)]
        #[case::required_strict(true, true, false)]
        #[case::provenance_only(false, false, false)]
        fn review_scoring_defers_stale_required_shard_conflicts(
            #[case] required: bool,
            #[case] strict: bool,
            #[case] deferred: bool,
        ) {
            use crate::evidence::AlignmentObservation;
            let mut arguments = vec![
                "score", "--vcf", "v", "--models", "m", "-o", "o", "--pileup", "a.tsv", "b.tsv",
            ];
            if strict {
                arguments.push("--strict-provenance");
            }
            let Command::Score(args) = parse(&arguments).unwrap().command else {
                panic!("not score")
            };
            let keep_alt = crate::testing::bwa_provenance("keep-alt");
            let result = score_reads_alignment_with(
                &args,
                RequiredTables { strlen: false, pileup: required },
                |path, _| {
                    // a.tsv was observed before its replacement by a keep-alt shard.
                    let stale = path == Path::new("a.tsv");
                    Ok(AlignmentObservation::Regular(Some(if stale {
                        crate::testing::bwa_provenance("primary")
                    } else {
                        keep_alt.clone()
                    })))
                },
            );
            if !deferred {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("aligned differently"), "{error}");
                return;
            }
            let (reads, checked) = result.unwrap();
            assert_eq!(reads.provenance, None, "the stale snapshots stand for nothing");
            assert_eq!(checked.len(), 2);
            // The shards loaded agree: their provenance is the reads'.
            let pileup = PileupTables {
                options: PileupOptions::default(),
                rows: Vec::new(),
                alignment: Some(keep_alt.clone()),
            };
            let loaded =
                loaded_score_alignment(&args, &reads, &checked, None, Some(&pileup)).unwrap();
            assert_eq!(loaded.provenance, Some(keep_alt));
        }

        /// Even when conflicting required shards are deferred, each must agree with the
        /// provenance-only group during preflight, whichever group is processed first.
        #[rstest]
        #[case::strlen_shards(true)]
        #[case::pileup_shards(false)]
        fn review_scoring_required_shards_check_provenance_only_at_startup(
            #[case] strlen_shards: bool,
            #[values("primary", "keep-alt")] provenance_only: &str,
        ) {
            use crate::evidence::AlignmentObservation;
            let (required_option, unused_option) =
                if strlen_shards { ("--strlen", "--pileup") } else { ("--pileup", "--strlen") };
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                "v",
                "--models",
                "m",
                "-o",
                "o",
                required_option,
                "a.tsv",
                "b.tsv",
                unused_option,
                "unused.tsv",
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let error = score_reads_alignment_with(
                &args,
                RequiredTables { strlen: strlen_shards, pileup: !strlen_shards },
                |path, _| {
                    let setting = match path.to_str().unwrap() {
                        "a.tsv" => "primary",
                        "b.tsv" => "keep-alt",
                        _ => provenance_only,
                    };
                    Ok(AlignmentObservation::Regular(Some(crate::testing::bwa_provenance(setting))))
                },
            )
            .unwrap_err()
            .to_string();
            assert!(
                error.contains("aligned differently") && error.contains("alt-project"),
                "{error}"
            );
        }

        /// Without --strict-provenance, required `--strlen` and `--pileup` tables whose snapshots
        /// disagree are left to the loaded tables too; a conflict with a provenance-only group, or
        /// under --strict-provenance, still fails preflight.
        #[rstest]
        #[case::both_required_lenient(true, true, false, true)]
        #[case::both_required_strict(true, true, true, false)]
        #[case::pileup_provenance_only(true, false, false, false)]
        #[case::strlen_provenance_only(false, true, false, false)]
        fn review_scoring_defers_required_conflicts_across_groups(
            #[case] strlen: bool,
            #[case] pileup: bool,
            #[case] strict: bool,
            #[case] deferred: bool,
        ) {
            use crate::evidence::AlignmentObservation;
            let mut arguments = vec![
                "score", "--vcf", "v", "--models", "m", "-o", "o", "--strlen", "s.tsv", "--pileup",
                "p.tsv",
            ];
            if strict {
                arguments.push("--strict-provenance");
            }
            let Command::Score(args) = parse(&arguments).unwrap().command else {
                panic!("not score")
            };
            let keep_alt = crate::testing::bwa_provenance("keep-alt");
            let result =
                score_reads_alignment_with(&args, RequiredTables { strlen, pileup }, |path, _| {
                    // s.tsv was observed before its replacement by a keep-alt table.
                    Ok(AlignmentObservation::Regular(Some(if path == Path::new("s.tsv") {
                        crate::testing::bwa_provenance("primary")
                    } else {
                        keep_alt.clone()
                    })))
                });
            if !deferred {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("aligned differently"), "{error}");
                return;
            }
            let (reads, checked) = result.unwrap();
            assert_eq!(reads.provenance, None, "the stale snapshots stand for nothing");
            assert_eq!(checked.len(), 2);
            // The tables loaded agree: their provenance is the reads'.
            let strlen = StrlenTables {
                options: crate::strlen::StrlenOptions::default(),
                rows: Vec::new(),
                alignment: Some(keep_alt.clone()),
            };
            let pileup = PileupTables {
                options: PileupOptions::default(),
                rows: Vec::new(),
                alignment: Some(keep_alt.clone()),
            };
            let loaded =
                loaded_score_alignment(&args, &reads, &checked, Some(&strlen), Some(&pileup))
                    .unwrap();
            assert_eq!(loaded.provenance, Some(keep_alt));
        }

        #[rstest]
        #[case::unread(false)]
        #[case::read_without_provenance(true)]
        fn review_preflight_distinguishes_unread_from_missing_provenance(#[case] read: bool) {
            use crate::evidence::AlignmentObservation;
            let dir = tempfile::tempdir().unwrap();
            let fifo = dir.path().join("unread.fifo");
            let regular = dir.path().join("pileup.tsv");
            let tags = crate::testing::provenance(&[("alt-project", "primary")], true);
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                "v",
                "--models",
                "m",
                "-o",
                "o",
                "--pileup",
                fifo.to_str().unwrap(),
                regular.to_str().unwrap(),
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let ((resolved, checked), logs) = captured_logs::capture(|| {
                score_reads_alignment_with(&args, RequiredTables::default(), |path, _| {
                    Ok(if path == fifo {
                        if read {
                            AlignmentObservation::Regular(None)
                        } else {
                            AlignmentObservation::Skipped
                        }
                    } else {
                        AlignmentObservation::Regular(Some(tags.clone()))
                    })
                })
                .unwrap()
            });
            assert_eq!(checked.len(), if read { 2 } else { 1 });
            assert_eq!(resolved.provenance, Some(tags));
            assert!(
                !logs.iter().any(|(_, text)| text.contains("record no alignment provenance")),
                "{logs:?}"
            );
            // These inputs are retained for provenance only, so their one warning comes from
            // the final snapshot merge rather than a required row read.
            let (_, final_logs) = captured_logs::capture(|| {
                loaded_score_alignment(&args, &resolved, &checked, None, None).unwrap()
            });
            let missing = final_logs
                .iter()
                .filter(|(_, text)| text.contains("record no alignment provenance"))
                .count();
            assert_eq!(missing, usize::from(read), "{final_logs:?}");
        }

        #[cfg(unix)]
        #[test]
        fn review_provenance_alias_is_rechecked_before_writing() {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            scoring_evidence_inputs(dir.path(), Some(crate::testing::bwa_provenance("primary")));
            let retained = path("unused.strlen");
            let before = format!(
                "##alignment={}\nunused\n",
                crate::testing::bwa_provenance("primary").to_json()
            );
            std::fs::write(&retained, &before).unwrap();
            let fifo = path("evidence.fifo");
            assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
            let output = path("scored.vcf.gz");
            let text = scoring_pileup_text(Some(&crate::testing::bwa_provenance("primary")));
            let mut processes = crate::testing::TestProcesses::default();
            // Opening the FIFO releases this producer only after scoring preflight has finished.
            processes.spawn(std::process::Command::new("sh").args([
                "-c",
                "{ ln -s \"$1\" \"$2\"; printf '%s' \"$3\"; } > \"$4\"",
                "alias-after-preflight",
                &retained,
                &output,
                &text,
                &fifo,
            ]));
            let (result, _) = run_fifo_score(
                &[
                    "score",
                    "--vcf",
                    &path("default.vcf.gz"),
                    "--models",
                    &path("model.winnowr.json"),
                    "-o",
                    &output,
                    "--strlen",
                    &retained,
                    "--pileup",
                    &fifo,
                ],
                &mut processes,
            );
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("write to another path"), "{error}");
            assert_eq!(std::fs::read_to_string(retained).unwrap(), before);
        }

        /// A staged model symlink retargeted after scoring read it: the output may not alias the
        /// model that was read, though the given path now names another.
        #[cfg(unix)]
        #[test]
        fn review_final_alias_check_protects_a_retargeted_model_symlink() {
            let dir = tempfile::tempdir().unwrap();
            write_default_mode_calls(dir.path());
            let mut model =
                crate::testing::separable_model_of(Task::Genotype, VariantType::Snp, "mq");
            model.info.feature_source = FeatureSource::Vcf;
            let read = dir.path().join("a.winnowr.json");
            model.save(&read).unwrap();
            let original = std::fs::read(&read).unwrap();
            let other = dir.path().join("b.winnowr.json");
            std::fs::write(&other, b"another model").unwrap();
            let staged = dir.path().join("staged.winnowr.json");
            std::os::unix::fs::symlink(&read, &staged).unwrap();
            let output = dir.path().join("scored.vcf");
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                dir.path().join("default.vcf.gz").to_str().unwrap(),
                "--models",
                staged.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let result = score_with(&args, NonZeroUsize::MIN, || {
                std::fs::remove_file(&staged).unwrap();
                std::os::unix::fs::symlink(&other, &staged).unwrap();
                std::os::unix::fs::symlink(&read, &output).unwrap();
            });
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("is the input model; write to another path"), "{error}");
            assert_eq!(std::fs::read(&read).unwrap(), original);
        }

        #[cfg(unix)]
        #[rstest]
        #[case::loaded_model(true)]
        #[case::unrelated(false)]
        fn review_final_alias_check_retains_loaded_model_identity(
            #[case] alias_model: bool,
            #[values(false, true)] sidecar: bool,
            #[values(false, true)] hard_link: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            write_default_mode_calls(dir.path());
            let mut model =
                crate::testing::separable_model_of(Task::Genotype, VariantType::Snp, "mq");
            model.info.feature_source = FeatureSource::Vcf;
            let model_path = dir.path().join("model.winnowr.json");
            model.save(&model_path).unwrap();
            let original = std::fs::read(&model_path).unwrap();
            let retained = dir.path().join("loaded-model.json");
            let unrelated = dir.path().join("unrelated");
            std::fs::write(&unrelated, b"old output").unwrap();
            let output = dir.path().join(if sidecar { "scored.vcf.gz" } else { "scored.vcf" });
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                dir.path().join("default.vcf.gz").to_str().unwrap(),
                "--models",
                model_path.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let result = score_with(&args, NonZeroUsize::MIN, || {
                std::fs::rename(&model_path, &retained).unwrap();
                std::fs::write(&model_path, b"replacement model").unwrap();
                let destination = if sidecar {
                    crate::scoring::scored_vcf_index(&output).unwrap()
                } else {
                    output.clone()
                };
                let target = if alias_model { &retained } else { &unrelated };
                if hard_link {
                    std::fs::hard_link(target, destination).unwrap();
                } else {
                    std::os::unix::fs::symlink(target, destination).unwrap();
                }
            });
            if alias_model {
                let error = format!("{:#}", result.unwrap_err());
                assert!(error.contains("is the input model; write to another path"), "{error}");
            } else {
                result.unwrap();
            }
            assert_eq!(std::fs::read(&retained).unwrap(), original);
            assert_eq!(std::fs::read(&model_path).unwrap(), b"replacement model");
        }

        #[cfg(unix)]
        #[rstest]
        #[case::model(true)]
        #[case::header(false)]
        fn review_final_alias_check_protects_models_and_headers(#[case] overwrite_model: bool) {
            let dir = tempfile::tempdir().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            write_default_mode_calls(dir.path());
            let mut model =
                crate::testing::separable_model_of(Task::Genotype, VariantType::Snp, "mq");
            model.info.feature_source = FeatureSource::Vcf;
            model.info.region_names = vec!["gate".to_string()];
            model.info.alignment = Some(crate::testing::provenance(&[("aligner", "mapper")], true));
            let model_path = path("model.winnowr.json");
            model.save(&model_path).unwrap();
            let sam = path("reads.sam");
            std::fs::write(
                &sam,
                "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:10000\n\
                @PG\tID:m\tPN:mapper\tCL:mapper ref.fa reads.fq\n",
            )
            .unwrap();
            let target = if overwrite_model { &model_path } else { &sam };
            let before = std::fs::read(target).unwrap();
            let fifo = path("gate.bed");
            assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
            let output = path("scored.vcf");
            let mut processes = crate::testing::TestProcesses::default();
            processes.spawn(std::process::Command::new("sh").args([
                "-c",
                "{ ln -s \"$1\" \"$2\"; printf 'chr1\\t0\\t1000\\n'; } > \"$3\"",
                "alias-during-regions",
                target,
                &output,
                &fifo,
            ]));
            let (result, _) = run_fifo_score(
                &[
                    "score",
                    "--vcf",
                    &path("default.vcf.gz"),
                    "--models",
                    &model_path,
                    "-o",
                    &output,
                    "--alignments",
                    &sam,
                    "--reference",
                    &path("unused.fa"),
                    "--regions",
                    &format!("gate={fifo}"),
                    "--alignment-tags",
                    "options-inference=explicit",
                ],
                &mut processes,
            );
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("write to another path"), "{error}");
            assert_eq!(std::fs::read(target).unwrap(), before);
        }

        #[test]
        fn review_observed_provenance_inputs_cannot_alias_output() {
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join("scored.vcf.gz");
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                "v",
                "--models",
                "m",
                "-o",
                output.to_str().unwrap(),
                "--strlen",
                output.to_str().unwrap(),
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            // Initially absent, as with a skipped FIFO; published before the observation opens it.
            let before = b"##alignment={}\nunused\n";
            let result =
                score_reads_alignment_with(&args, RequiredTables::default(), |path, tags| {
                    std::fs::write(path, before).unwrap();
                    crate::evidence::observe_alignment_with_tags(path, tags)
                });
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("write to another path"), "{error}");
            assert_eq!(std::fs::read(output).unwrap(), before);
        }

        #[test]
        fn pr38_score_skips_provenance_only_inputs_for_legacy_models() {
            let tables = MixedTables::new(&[]);
            let model = tables.train(
                &[tables.extract("S4", &["chr1", "chr2"]), tables.extract("S5", &["chr1", "chr2"])],
                "legacy",
            );
            assert!(model.info.alignment.is_none());
            let (vcf, model, output) = (
                tables.path("calls.vcf.gz"),
                tables.path("legacy.winnowr.json"),
                tables.path("legacy.vcf.gz"),
            );
            let unused = tables.path("unused.tsv");
            std::fs::write(&unused, "##alignment=not-json\nunused\n").unwrap();
            try_run(&[
                "score",
                "--vcf",
                &vcf,
                "--models",
                &model,
                "-o",
                &output,
                "--pileup",
                &unused,
                "--strict-provenance",
            ])
            .unwrap();
            try_run(&[
                "score",
                "--vcf",
                &vcf,
                "--models",
                &model,
                "-o",
                &output,
                "--alignments",
                "absent-unused.bam",
                "--reference",
                "absent-unused.fa",
                "--strict-provenance",
            ])
            .unwrap();
            // The same unused table is still read when any model records provenance.
            let mut bearing = WinnowrModel::load(&model, NonZeroUsize::new(1).unwrap()).unwrap();
            bearing.info.alignment = Some(crate::testing::bwa_provenance("primary"));
            bearing.save(&model).unwrap();
            let error = try_run(&[
                "score", "--vcf", &vcf, "--models", &model, "-o", &output, "--pileup", &unused,
            ])
            .unwrap_err();
            assert!(format!("{error:#}").contains("malformed ##alignment="), "{error:#}");
        }

        #[rstest]
        #[case::mixed_warns("mixed", false, true, false)]
        #[case::mixed_strict_fails("mixed", true, false, false)]
        #[case::agreement_is_logged("primary", false, false, true)]
        fn pr38_indeterminate_score_logs(
            #[case] alignment: &str,
            #[case] strict: bool,
            #[case] warns: bool,
            #[case] affirmative: bool,
        ) {
            let tables = MixedTables::new(&[]);
            let mut model = tables.train(
                &[tables.extract("S4", &["chr1", "chr2"]), tables.extract("S5", &["chr1", "chr2"])],
                "logging",
            );
            model.info.alignment = Some(crate::testing::bwa_provenance(alignment));
            let (vcf, model_path, output) = (
                tables.path("calls.vcf.gz"),
                tables.path("logging.winnowr.json"),
                tables.path("logged.vcf.gz"),
            );
            model.save(&model_path).unwrap();
            let tag = format!("alt-project={alignment}");
            let mut args = vec![
                "score",
                "--vcf",
                &vcf,
                "--models",
                &model_path,
                "-o",
                &output,
                "--alignment-tags",
                "aligner=bwa-mem3",
                &tag,
            ];
            if strict {
                args.push("--strict-provenance");
            }
            let (result, logs) = captured_logs::capture(|| try_run(&args));
            assert_eq!(result.is_ok(), !strict, "{result:?}");
            assert_eq!(
                logs.iter().any(|(level, message)| *level == log::Level::Warn
                    && message.contains("indeterminate")),
                warns,
                "{logs:?}"
            );
            assert_eq!(
                logs.iter().any(|(level, message)| *level == log::Level::Info
                    && message.contains("comparable alignment tags match")),
                affirmative,
                "{logs:?}"
            );
        }

        /// The reads' alignment provenance goes from their header through `pileup` and `extract`
        /// into the model, which `score` compares with the reads it scores: a mismatch warns, and
        /// fails with --strict-provenance, as do reads whose alignment is not known.
        #[test]
        fn the_alignment_provenance_goes_from_the_reads_to_the_model() {
            let unresolved = AlignmentsModel::new();
            let model = ModelInfo::read(unresolved.path("reads.winnowr.json")).unwrap();
            let incomplete = model.alignment.as_ref().unwrap();
            assert!(incomplete_options_reason(incomplete).is_some());
            assert_eq!(
                FeatureTable::read_info(unresolved.path("S1.parquet")).unwrap().alignment.as_ref(),
                Some(incomplete)
            );
            assert_eq!(
                crate::evidence::read_alignment(&[unresolved.path("pileup.tsv.gz")])
                    .unwrap()
                    .as_ref(),
                Some(incomplete)
            );
            let raw_reads = unresolved.reads();
            let raw_reads: Vec<&str> = raw_reads.iter().map(String::as_str).collect();
            let error = unresolved
                .score("unresolved.vcf.gz", &[&raw_reads[..], &["--strict-provenance"]].concat())
                .unwrap_err();
            assert!(error.to_string().contains("incomplete alignment settings"), "{error:#}");

            let declared = [
                "--alignment-tags",
                "options-inference=explicit",
                "command=mem",
                "t=8",
                "alt-project=primary",
            ];
            let setup = AlignmentsModel::new_with_tags(&declared);
            let expected = [
                ("aligner", "bwa-mem3"),
                ("alt-project", "primary"),
                ("command", "mem"),
                ("options-inference", "explicit"),
                ("source", "header"),
                ("sq-alt", "no"),
                ("t", "8"),
                ("version", "0.16.0"),
            ];
            let expected = crate::testing::provenance(&expected, false);
            let pileup = [setup.path("pileup.tsv.gz")];
            assert_eq!(crate::evidence::read_alignment(&pileup).unwrap().as_ref(), Some(&expected));
            let table = FeatureTable::read_info(setup.path("S1.parquet")).unwrap();
            assert_eq!(table.alignment.as_ref(), Some(&expected));
            let model_path = setup.path("reads.winnowr.json");
            let model = WinnowrModel::load(&model_path, NonZeroUsize::new(1).unwrap()).unwrap();
            assert_eq!(model.info.alignment.as_ref(), Some(&expected));

            let reads = setup.reads();
            let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
            let strict = [&reads[..], &["--strict-provenance"]].concat();
            setup.score("matching.vcf.gz", &[&strict[..], &declared].concat()).unwrap();
            let keep_alt = [
                "--alignment-tags",
                "options-inference=explicit",
                "command=mem",
                "t=8",
                "alt-project=keep-alt",
            ];
            setup.score("warned.vcf.gz", &[&reads[..], &keep_alt].concat()).unwrap();
            let error =
                setup.score("failed.vcf.gz", &[&strict[..], &keep_alt].concat()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "the reads were not aligned as the models' training reads were: alt-project: \
                 models primary, reads keep-alt. Calls may be scored worse (see the user guide: \
                 Alignment provenance). Pass --alignment-tags if the reads' recorded alignment is \
                 wrong."
            );
            let unknown = ["--alignment-tags", "none"];
            let error =
                setup.score("unknown.vcf.gz", &[&strict[..], &unknown].concat()).unwrap_err();
            assert!(error.to_string().contains("but the reads' alignment is not known"), "{error}");

            // The reads' header and the evidence tables of the same reads must agree.
            let (vcf, other) = (setup.path("default.vcf.gz"), setup.path("other.tsv.gz"));
            let base = ["pileup", "--vcf", &vcf, "--regions", "chr1", "-o", &other];
            run(&[&base[..], &reads, &["--alignment-tags", "aligner=other"]].concat());
            let table = setup.path("disagree.parquet");
            let (happy, sample) =
                (setup.path("happy.vcf"), ["--sample", "S1", "--contigs", "chr1"]);
            let extract =
                ["extract", "--vcf", &vcf, "--happy", &happy, "-o", &table, "--pileup", &other];
            let error = try_run(&[&extract[..], &sample, &reads].concat()).unwrap_err().to_string();
            let bam = setup.path("reads.bam");
            assert_eq!(
                error,
                format!(
                    "the --pileup tables and --alignments {bam} were aligned differently: \
                     aligner: the --pileup tables other, --alignments {bam} bwa-mem3. They \
                     should be of the same reads: measure the tables again from them, pass \
                     --alignment-tags to set (KEY=VALUE) or remove (KEY!) the tags they disagree \
                     on, or pass --alignment-tags none to record nothing"
                )
            );
            // --alignment-tags applies over each source before they must agree.
            let tagged = [&extract[..], &sample, &reads, &["--alignment-tags", "aligner=bwa-mem3"]];
            run(&tagged.concat());

            // A model trained with --mixed-alignments on reads aligned two ways records the tags
            // they share, and scores reads aligned either way, even with --strict-provenance.
            let s3 = setup.extract(
                "S3",
                "S3",
                &[&["--pileup", pileup[0].as_str()][..], &keep_alt].concat(),
            );
            let train =
                ["train", "--variant-type", "snp", "--max-rounds", "10", "--pileup-features"];
            let mixed = setup.path("mixed");
            let tables = [setup.path("S1.parquet"), s3];
            let error = try_run(
                &[&train[..], &["-o", &mixed, "--tables", &tables[0], &tables[1]]].concat(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("are of reads aligned differently"), "{error}");
            run(&[
                &train[..],
                &["-o", &mixed, "--mixed-alignments", "--tables", &tables[0], &tables[1]],
            ]
            .concat());
            let model = setup.path("mixed.winnowr.json");
            let recorded = ModelInfo::read(&model).unwrap().alignment.unwrap();
            assert!(!recorded.to_string().contains("alt-project"), "{recorded}");
            let score = |extra: &[&str]| {
                let out = setup.path("mixed.vcf.gz");
                let tags = if extra.is_empty() { &declared[..] } else { extra };
                try_run(
                    &[
                        &["score", "--vcf", &vcf, "--models", &model, "-o", &out][..],
                        &reads,
                        &["--strict-provenance"],
                        tags,
                    ]
                    .concat(),
                )
            };
            score(&[]).unwrap();
            score(&keep_alt).unwrap();
        }

        #[rstest]
        #[case::given(0)]
        #[case::bundled_compressed(1)]
        #[case::bundled_plain(2)]
        #[case::explicit_bed_ignores_bundled_candidate(3)]
        fn resolved_region_beds_are_protected_from_score_outputs(
            #[case] source: usize,
            #[values(false, true)] sidecar: bool,
        ) {
            let setup = VcfSourceModel::new();
            let model = setup.model();
            let vcf = setup.path("default.vcf.gz");
            std::fs::create_dir(setup.path("regions")).unwrap();
            let given_bed = setup.path("given.bed");
            let compressed_bed = setup.path(&format!("regions/{NOT_DIFFICULT_REGIONS}.bed.gz"));
            let plain_bed = setup.path(&format!("regions/{NOT_DIFFICULT_REGIONS}.bed"));
            let contents = b"chr1\t0\t100000\n";
            for bed in [&given_bed, &plain_bed] {
                std::fs::write(bed, contents).unwrap();
            }
            crate::testing::write_gzip(Path::new(&compressed_bed), contents);
            let input = [&given_bed, &compressed_bed, &plain_bed][source.min(2)];
            let before = std::fs::read(input).unwrap();
            let output = setup.path("scored.vcf.gz");
            let target = format!("{output}{}", if sidecar { ".tbi" } else { "" });
            std::os::unix::fs::symlink(input, &target).unwrap();
            let region = format!("{NOT_DIFFICULT_REGIONS}={given_bed}");
            let mut args = vec![
                "score",
                "--vcf",
                &vcf,
                "--models",
                &model,
                "-o",
                &output,
                "--abstain-actions",
                "flip",
            ];
            if source == 0 || source == 3 {
                args.extend(["--regions", &region]);
            }
            if source == 3 {
                try_run(&args).unwrap();
                assert!(read_gzip(Path::new(&output)).contains("WINNOWR_P="));
                assert_eq!(std::fs::read(&given_bed).unwrap(), contents);
            } else {
                let error = format!("{:#}", try_run(&args).unwrap_err());
                assert!(
                    error.ends_with("is the input region BED; write to another path"),
                    "{error}"
                );
                assert_eq!(std::fs::read(input).unwrap(), before);
                assert_eq!(std::fs::read(&target).unwrap(), before);
            }
        }

        #[rstest]
        fn review_outputs_do_not_create_bundled_bed_candidates(
            #[values("score", "evaluate")] command: &str,
            #[values(false, true)] explicit: bool,
        ) {
            let tables = MixedTables::new(&[]);
            let (model, regions) = bundled_model(&tables);
            let compressed = PathBuf::from(format!("{regions}/{NOT_DIFFICULT_REGIONS}.bed.gz"));
            let contents = b"chr1\t0\t100000\nchr2\t0\t100000\n";
            crate::testing::write_gzip(&compressed, contents);
            let output = format!("{regions}/{NOT_DIFFICULT_REGIONS}.bed");
            let given = tables.path("given.bed");
            std::fs::write(&given, contents).unwrap();
            let region = format!("{NOT_DIFFICULT_REGIONS}={given}");
            let vcf = tables.path("calls.vcf.gz");
            let table = tables.path("S1.parquet");
            let mut args =
                vec![command, "--models", &model, "-o", &output, "--abstain-actions", "flip"];
            if command == "score" {
                args.extend(["--vcf", &vcf]);
            } else {
                args.extend(["--table", &table]);
            }
            if explicit {
                args.extend(["--regions", &region]);
            }
            let result = try_run(&args);
            if explicit {
                result.unwrap();
                assert!(Path::new(&output).is_file());
            } else {
                let error = format!("{:#}", result.unwrap_err());
                assert!(
                    error.contains("region BED") && error.contains("write to another path"),
                    "{error}"
                );
                assert!(!Path::new(&output).exists());
            }
            assert_eq!(read_gzip(&compressed).as_bytes(), contents);
        }

        #[cfg(unix)]
        #[rstest]
        fn review_final_model_guard_keeps_the_resolved_path(
            #[values(false, true)] sidecar: bool,
            #[values(false, true)] collision: bool,
        ) {
            let setup = VcfSourceModel::new();
            let model = PathBuf::from(setup.model());
            let original = std::fs::read(&model).unwrap();
            let other = setup.path("other.winnowr.json");
            std::fs::write(&other, &original).unwrap();
            let alias = setup.path("staged.winnowr.json");
            std::os::unix::fs::symlink(&model, &alias).unwrap();
            let output = setup.path("retargeted-output.vcf.gz");
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                &setup.path("default.vcf.gz"),
                "--models",
                &alias,
                "-o",
                &output,
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let called = std::cell::Cell::new(false);
            let result = score_with(&args, NonZeroUsize::MIN, || {
                called.set(true);
                std::fs::remove_file(&alias).unwrap();
                std::os::unix::fs::symlink(&other, &alias).unwrap();
                if collision {
                    let destination =
                        if sidecar { format!("{output}.tbi") } else { output.clone() };
                    std::os::unix::fs::symlink(&model, destination).unwrap();
                }
            });
            assert!(called.get());
            if collision {
                let error = format!("{:#}", result.unwrap_err());
                assert!(error.contains("is the input model; write to another path"), "{error}");
            } else {
                result.unwrap();
            }
            assert_eq!(std::fs::read(&model).unwrap(), original);
        }

        #[cfg(unix)]
        #[test]
        fn review_final_bed_guard_uses_the_identity_actually_read() {
            let setup = VcfSourceModel::new();
            let path = |name: &str| PathBuf::from(setup.path(name));
            let (bed, other, alias, moved, output) = (
                path("read.bed"),
                path("other.bed"),
                path("given.bed"),
                path("moved.bed"),
                path("scored.vcf.gz"),
            );
            let original = b"chr1\t0\t100000\n";
            std::fs::write(&bed, original).unwrap();
            std::fs::write(&other, b"chr1\t0\t100\n").unwrap();
            std::os::unix::fs::symlink(&bed, &alias).unwrap();
            let region = format!("{NOT_DIFFICULT_REGIONS}={}", alias.display());
            let Command::Score(args) = parse(&[
                "score",
                "--vcf",
                &setup.path("default.vcf.gz"),
                "--models",
                &setup.model(),
                "-o",
                output.to_str().unwrap(),
                "--regions",
                &region,
                "--abstain-actions",
                "flip",
            ])
            .unwrap()
            .command
            else {
                panic!("not score")
            };
            let called = std::cell::Cell::new(false);
            let result = score_with(&args, NonZeroUsize::MIN, || {
                called.set(true);
                std::fs::rename(&bed, &moved).unwrap();
                std::fs::write(&bed, b"chr1\t0\t200\n").unwrap();
                std::fs::remove_file(&alias).unwrap();
                std::os::unix::fs::symlink(&other, &alias).unwrap();
                std::os::unix::fs::symlink(&moved, &output).unwrap();
            });
            assert!(called.get());
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.ends_with("is the input region BED; write to another path"), "{error}");
            assert_eq!(std::fs::read(&moved).unwrap(), original);
        }

        /// `evaluate` and `stratify` reject models recording different policy digests of the
        /// not-difficult regions before the policy BED is opened: the BED here does not exist,
        /// and is not named in the error.
        #[rstest]
        fn conflicting_policy_digests_fail_before_the_policy_bed_is_opened(
            #[values(false, true)] allow_mismatch: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let mut paths = Vec::new();
            let mut models = Vec::new();
            for (task, sha) in [(Task::Genotype, "a"), (Task::Promote, "b")] {
                let mut model = crate::testing::separable_model(task, VariantType::Indel);
                let digest = RegionDigest { sha256: sha.repeat(64), intervals: 1, bases: 10 };
                model.info.policy_region_digests =
                    BTreeMap::from([(NOT_DIFFICULT_REGIONS.to_string(), digest)]);
                let path = dir.path().join(format!("{task}{MODEL_SUFFIX}"));
                model.save(&path).unwrap();
                paths.push(path);
                models.push(model);
            }
            let infos: Vec<&ModelInfo> = models.iter().map(|model| &model.info).collect();
            let files = ModelFile::resolve_all(&paths);
            let missing = dir.path().join("missing.bed");
            let given =
                [NamedBed { name: NOT_DIFFICULT_REGIONS.to_string(), path: missing.clone() }];
            let outputs = [dir.path().join("out")];
            let result =
                policy_not_difficult(&infos, &files, &given, true, allow_mismatch, &outputs);
            let error = format!("{:#}", result.unwrap_err());
            assert!(
                error.starts_with(&format!(
                    "--models were trained with different {NOT_DIFFICULT_REGIONS} region BEDs"
                )),
                "{error}"
            );
            assert!(!error.contains(&missing.display().to_string()), "{error}");
        }

        #[rstest]
        fn review_score_rejects_conflicting_digests_before_bed_reads(
            #[values(false, true)] policy: bool,
            #[values(false, true)] allow: bool,
        ) {
            let dir = tempfile::tempdir().unwrap();
            write_default_mode_calls(dir.path());
            let name = if policy { NOT_DIFFICULT_REGIONS } else { "segdups" };
            let mut paths = Vec::new();
            for (task, sha) in [(Task::Genotype, "a"), (Task::Promote, "b")] {
                let mut model = crate::testing::separable_model_of(task, VariantType::Snp, "mq");
                model.info.feature_source = FeatureSource::Vcf;
                let digest = RegionDigest { sha256: sha.repeat(64), intervals: 1, bases: 10 };
                if policy {
                    model.info.policy_region_digests.insert(name.to_string(), digest);
                } else {
                    model.info.region_names = vec![name.to_string()];
                    model.info.region_digests.insert(name.to_string(), digest);
                }
                let path = dir.path().join(format!("{task}{MODEL_SUFFIX}"));
                model.save(&path).unwrap();
                paths.push(path.display().to_string());
            }
            let vcf = dir.path().join("default.vcf.gz").display().to_string();
            let output = dir.path().join("out.vcf.gz").display().to_string();
            let region = format!("{name}={}", dir.path().join("missing.bed").display());
            let mut args = vec![
                "score",
                "--vcf",
                &vcf,
                "--models",
                &paths[0],
                &paths[1],
                "-o",
                &output,
                "--regions",
                &region,
            ];
            if policy {
                args.extend(["--abstain-actions", "flip"]);
            }
            if allow {
                args.push("--allow-region-mismatch");
            }
            let error = format!("{:#}", try_run(&args).unwrap_err());
            assert!(
                error.starts_with(&format!(
                    "--models were trained with different {name} region BEDs"
                )),
                "{error}"
            );
            assert!(!error.contains("missing.bed"), "{error}");
        }

        /// A snp model trained on `tables`' S1 and S2 tables (`S1.parquet`, `S2.parquet`, of chr1
        /// and chr2), written to `models/snp.winnowr.json` beside an empty `models/regions`.
        /// Returns the model's path and the `regions` directory.
        fn bundled_model(tables: &MixedTables<'_>) -> (String, String) {
            let (a, b) =
                (tables.extract("S1", &["chr1", "chr2"]), tables.extract("S2", &["chr1", "chr2"]));
            let regions = tables.path("models/regions");
            std::fs::create_dir_all(&regions).unwrap();
            let model = tables.path("models/snp");
            let train = ["train", "--variant-type", "snp", "--max-rounds", "10", "-o", &model];
            run(&[&train[..], &["--tables", &a, &b]].concat());
            (format!("{model}{MODEL_SUFFIX}"), regions)
        }

        /// A destination can become the consumed BED after preflight. Every actual writer must
        /// reject its opened handle, not just re-check the BED's now-retargeted supplied path.
        #[rstest]
        #[case::unrelated_output(true, false)]
        #[case::bed_alias(true, true)]
        #[case::no_policy_bed(false, false)]
        fn review_report_writers_protect_the_bed_actually_read(
            #[values(
                "metrics",
                "strata.tsv",
                "rates.tsv",
                "calibration.tsv",
                "regions.strata.tsv",
                "regions.rates.tsv",
                "regions.calibration.tsv",
                "calls",
                "labelled-v1",
                "labelled-v2"
            )]
            destination: &str,
            #[case] policy: bool,
            #[case] overwrite_bed: bool,
        ) {
            let tables = MixedTables::new(&[]);
            let (model, _) = bundled_model(&tables);
            let path = |name: &str| PathBuf::from(tables.path(name));
            let (bed, supplied, moved, other) =
                (path("read.bed"), path("given.bed"), path("moved.bed"), path("other.bed"));
            let original = b"chr1\t0\t100000\nchr2\t0\t100000\n";
            std::fs::write(&bed, original).unwrap();
            std::fs::write(&other, original).unwrap();
            std::os::unix::fs::symlink(&bed, &supplied).unwrap();
            let region = format!("{NOT_DIFFICULT_REGIONS}={}", supplied.display());
            let first = tables.path("S1.parquet");
            let happy = tables.path("happy.vcf");
            let prefix = tables.path("report");
            let calls = tables.path("calls.tsv.gz");
            let labelled = tables.path("labelled");
            std::fs::create_dir(&labelled).unwrap();
            let reported = format!("near={}", other.display());
            let output = match destination {
                "metrics" => path("metrics.tsv"),
                "calls" => PathBuf::from(&calls),
                "labelled-v1" => PathBuf::from(&labelled).join("S1.v1.snp.parquet"),
                "labelled-v2" => PathBuf::from(&labelled).join("S1.v2.snp.parquet"),
                suffix => PathBuf::from(format!("{prefix}.{suffix}")),
            };
            let mutate = || {
                std::fs::rename(&bed, &moved).unwrap();
                std::fs::write(&bed, b"chr1\t0\t5\n").unwrap();
                std::fs::remove_file(&supplied).unwrap();
                std::os::unix::fs::symlink(&other, &supplied).unwrap();
                if overwrite_bed {
                    std::os::unix::fs::symlink(&moved, &output).unwrap();
                } else {
                    std::fs::write(&output, b"stale output that must be truncated").unwrap();
                }
            };
            let policy_args = if policy {
                vec!["--regions", region.as_str(), "--abstain-actions", "flip"]
            } else {
                Vec::new()
            };
            let result = if destination == "metrics" {
                let Command::Evaluate(args) = parse(
                    &[
                        &[
                            "evaluate",
                            "--models",
                            &model,
                            "--table",
                            &first,
                            "-o",
                            output.to_str().unwrap(),
                        ][..],
                        &policy_args,
                    ]
                    .concat(),
                )
                .unwrap()
                .command
                else {
                    panic!("not evaluate")
                };
                evaluate_with(&args, NonZeroUsize::MIN, mutate)
            } else {
                let Command::Stratify(args) = parse(
                    &[
                        &[
                            "stratify",
                            "--models",
                            &model,
                            "--tables",
                            &first,
                            &first,
                            "--truth-names",
                            "v1",
                            "v2",
                            "--truth-beds",
                            other.to_str().unwrap(),
                            other.to_str().unwrap(),
                            "--happy",
                            &happy,
                            &happy,
                            "-o",
                            &prefix,
                            "--calls",
                            &calls,
                            "--labelled-calls",
                            &labelled,
                            "--report-region",
                            &reported,
                        ][..],
                        &policy_args,
                    ]
                    .concat(),
                )
                .unwrap()
                .command
                else {
                    panic!("not stratify")
                };
                stratify_with(&args, NonZeroUsize::MIN, mutate)
            };
            assert_eq!(std::fs::read(&moved).unwrap(), original, "{destination} overwrote its BED");
            if overwrite_bed {
                let error = format!("{:#}", result.unwrap_err());
                assert!(error.contains("is the input region BED"), "{error}");
            } else {
                result.unwrap();
                let bytes = std::fs::read(&output).unwrap();
                if destination.starts_with("labelled-") {
                    assert!(bytes.starts_with(b"PAR1") && bytes.ends_with(b"PAR1"));
                    FeatureTable::read_parquet(&output).unwrap();
                } else if destination == "calls" {
                    assert!(read_gzip(&output).starts_with("row\tchrom\tpos\t"));
                } else {
                    assert!(String::from_utf8(bytes).unwrap().starts_with("sample\t"));
                }
            }
        }

        /// `stratify` also reads its truth and report BEDs: a destination redirected to one after
        /// preflight fails before truncating it, even with no policy BED.
        #[rstest]
        fn review_stratify_writers_protect_truth_and_report_beds(
            #[values("strata.tsv", "regions.rates.tsv", "calls", "labelled-v1")] destination: &str,
            #[values("truth", "report")] bed_kind: &str,
        ) {
            let tables = MixedTables::new(&[]);
            let (model, _) = bundled_model(&tables);
            let path = |name: &str| PathBuf::from(tables.path(name));
            let (truth, reported) = (path("truth.bed"), path("reported.bed"));
            let original = b"chr1\t0\t100000\nchr2\t0\t100000\n";
            std::fs::write(&truth, original).unwrap();
            std::fs::write(&reported, original).unwrap();
            let first = tables.path("S1.parquet");
            let happy = tables.path("happy.vcf");
            let prefix = tables.path("report");
            let calls = tables.path("calls.tsv.gz");
            let labelled = tables.path("labelled");
            std::fs::create_dir(&labelled).unwrap();
            let output = match destination {
                "calls" => PathBuf::from(&calls),
                "labelled-v1" => PathBuf::from(&labelled).join("S1.v1.snp.parquet"),
                suffix => PathBuf::from(format!("{prefix}.{suffix}")),
            };
            let bed = if bed_kind == "truth" { &truth } else { &reported };
            let near = format!("near={}", reported.display());
            let Command::Stratify(args) = parse(&[
                "stratify",
                "--models",
                &model,
                "--tables",
                &first,
                &first,
                "--truth-names",
                "v1",
                "v2",
                "--truth-beds",
                truth.to_str().unwrap(),
                truth.to_str().unwrap(),
                "--happy",
                &happy,
                &happy,
                "-o",
                &prefix,
                "--calls",
                &calls,
                "--labelled-calls",
                &labelled,
                "--report-region",
                &near,
            ])
            .unwrap()
            .command
            else {
                panic!("not stratify")
            };
            let result = stratify_with(&args, NonZeroUsize::MIN, || {
                std::os::unix::fs::symlink(bed, &output).unwrap();
            });
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("is the input region BED"), "{error}");
            assert_eq!(std::fs::read(bed).unwrap(), original, "{destination} overwrote its BED");
        }

        /// The files `evaluate` reads (its model, its table, and the not-difficult regions BED,
        /// given or found beside the models) are not its output: writing would destroy them.
        #[test]
        fn evaluate_does_not_write_over_its_inputs() {
            let tables = MixedTables::new(&[]);
            let (model, regions) = bundled_model(&tables);
            let table = tables.path("S1.parquet");
            let bundled = format!("{regions}/{NOT_DIFFICULT_REGIONS}.bed");
            std::fs::write(&bundled, "chr1\t0\t100000\n").unwrap();
            let given = tables.path("given.bed");
            std::fs::write(&given, "chr1\t0\t100000\n").unwrap();
            let region = format!("{NOT_DIFFICULT_REGIONS}={given}");
            let evaluate = |output: &str, extra: &[&str]| {
                let base = ["evaluate", "--models", &model, "--table", &table, "-o", output];
                try_run(&[&base[..], &["--abstain-actions", "flip"], extra].concat())
            };
            evaluate(&tables.path("metrics.tsv"), &[]).unwrap();
            for (output, what, extra) in [
                (&bundled, "region BED", &[][..]),
                (&given, "region BED", &["--regions", &region][..]),
                (&table, "feature table", &[][..]),
                (&model, "model", &[][..]),
            ] {
                let before = std::fs::read(output).unwrap();
                let result = evaluate(output, extra);
                assert_eq!(std::fs::read(output).unwrap(), before, "evaluate wrote over {output}");
                let error = result.unwrap_err().to_string();
                assert_eq!(
                    error,
                    format!("the output {output} is the input {what}; write to another path")
                );
            }
        }

        /// `evaluate` and `stratify` given models that record different digests of a region fail
        /// on that before they read a feature table (here, ones that do not exist), even with
        /// --allow-region-mismatch.
        #[rstest]
        fn models_of_different_region_beds_fail_before_the_tables_are_read(
            #[values("evaluate", "stratify")] command: &str,
            #[values(false, true)] allow_mismatch: bool,
        ) {
            let dir = tempfile::TempDir::new().unwrap();
            let path = |name: &str| dir.path().join(name).display().to_string();
            let models: Vec<String> = [(Task::Genotype, "a"), (Task::Promote, "b")]
                .into_iter()
                .map(|(task, sha)| {
                    let mut model = crate::testing::separable_model(task, VariantType::Snp);
                    let digest = RegionDigest { sha256: sha.repeat(64), intervals: 1, bases: 10 };
                    model.info.region_names = vec!["segdups".to_string()];
                    model.info.region_digests = BTreeMap::from([("segdups".to_string(), digest)]);
                    let model_path = path(&format!("{task}{MODEL_SUFFIX}"));
                    model.save(&model_path).unwrap();
                    model_path
                })
                .collect();
            let (table, output) = (path("missing.parquet"), path("out"));
            let (bed, happy) = (path("missing.bed"), path("missing.vcf"));
            let mut args = vec![command, "--models", &models[0], &models[1], "-o", &output];
            if command == "evaluate" {
                args.extend(["--table", &table]);
            } else {
                args.extend(["--truth-names", "v1", "v2", "--tables", &table, &table]);
                args.extend(["--truth-beds", &bed, &bed, "--happy", &happy, &happy]);
            }
            if allow_mismatch {
                args.push("--allow-region-mismatch");
            }
            let error = format!("{:#}", try_run(&args).unwrap_err());
            assert!(
                error.starts_with("--models were trained with different segdups region BEDs"),
                "{error}"
            );
        }

        #[test]
        fn review_stratify_ignores_regional_outputs_when_no_regions_are_reported() {
            let tables = MixedTables::new(&[]);
            let (model, _) = bundled_model(&tables);
            let first = tables.path("S1.parquet");
            let output = tables.path("run");
            let truth = tables.path("run.regions.rates.tsv");
            let contents = b"chr1\t0\t100000\nchr2\t0\t100000\n";
            std::fs::write(&truth, contents).unwrap();
            let happy = tables.path("happy.vcf");
            try_run(&[
                "stratify",
                "--models",
                &model,
                "--tables",
                &first,
                &first,
                "--truth-names",
                "v1",
                "v2",
                "--truth-beds",
                &truth,
                &truth,
                "--happy",
                &happy,
                &happy,
                "-o",
                &output,
            ])
            .unwrap();
            assert_eq!(std::fs::read(&truth).unwrap(), contents);
            for suffix in ["strata.tsv", "rates.tsv", "calibration.tsv"] {
                assert!(Path::new(&format!("{output}.{suffix}")).is_file());
            }
            assert!(!Path::new(&format!("{output}.regions.strata.tsv")).exists());
        }

        /// `stratify` writes none of its outputs (--calls, and the reports at the -o prefix) over
        /// a file it reads: its models, tables, truth BEDs, hap.py VCFs, the regions it reports
        /// within, or the not-difficult regions BED it finds beside the models.
        #[test]
        fn stratify_does_not_write_over_its_inputs() {
            let tables = MixedTables::new(&[]);
            let (model, regions) = bundled_model(&tables);
            let first = tables.path("S1.parquet");
            let second = tables.path("S1.second.parquet");
            let (vcf, happy) = (tables.path("calls.vcf.gz"), tables.path("happy.vcf"));
            let extract = ["extract", "--vcf", &vcf, "--happy", &happy, "--sample", "S1"];
            run(&[&extract[..], &["-o", &second, "--contigs", "chr1", "chr2"]].concat());
            let truth = [tables.path("v1.bed"), tables.path("v2.bed")];
            for bed in &truth {
                std::fs::write(bed, "chr1\t0\t100000\nchr2\t0\t100000\n").unwrap();
            }
            let reported = tables.path("reported.strata.tsv");
            std::fs::write(&reported, "chr1\t0\t5000\n").unwrap();
            let bundled = format!("{regions}/{NOT_DIFFICULT_REGIONS}.bed");
            std::fs::write(&bundled, "chr1\t0\t100000\nchr2\t0\t100000\n").unwrap();
            let report = format!("near={reported}");
            let stratify_with = |output: &str, calls: &str, second: &str, extra: &[&str]| {
                let args = [
                    "stratify",
                    "--models",
                    &model,
                    "--truth-names",
                    "v1",
                    "v2",
                    "--tables",
                    &first,
                    second,
                    "--truth-beds",
                    &truth[0],
                    &truth[1],
                    "--happy",
                    &happy,
                    &happy,
                    "-o",
                    output,
                    "--calls",
                    calls,
                    "--report-region",
                    &report,
                    "--abstain-actions",
                    "flip",
                ];
                try_run(&[&args[..], extra].concat())
            };
            let stratify = |output: &str, calls: &str| stratify_with(output, calls, &second, &[]);
            let output = tables.path("strata");
            stratify(&output, &tables.path("calls.tsv.gz")).unwrap();
            for (calls, what) in [
                (&bundled, "region BED"),
                (&truth[0], "truth BED"),
                (&reported, "report region BED"),
                (&second, "feature table"),
                (&model, "model"),
                (&happy, "hap.py VCF"),
            ] {
                let before = std::fs::read(calls).unwrap();
                let result = stratify(&output, calls);
                assert_eq!(std::fs::read(calls).unwrap(), before, "stratify wrote over {calls}");
                let error = result.unwrap_err().to_string();
                assert_eq!(
                    error,
                    format!("the output {calls} is the input {what}; write to another path")
                );
            }
            // A report at the -o prefix: reported.strata.tsv.
            let prefix = tables.path("reported");
            let before = std::fs::read(&reported).unwrap();
            let result = stratify(&prefix, &tables.path("calls.tsv.gz"));
            assert_eq!(std::fs::read(&reported).unwrap(), before, "stratify wrote over it");
            let error = result.unwrap_err().to_string();
            assert_eq!(
                error,
                format!(
                    "the output {reported} is the input report region BED; write to another path"
                )
            );
            // A labelled calls output, <dir>/<sample>.<truth>.<variant type>.parquet: here the
            // second table.
            let labelled = tables.path("labelled");
            std::fs::create_dir(&labelled).unwrap();
            let staged = format!("{labelled}/S1.v1.snp.parquet");
            std::fs::copy(&second, &staged).unwrap();
            let before = std::fs::read(&staged).unwrap();
            let calls = tables.path("calls.tsv.gz");
            let result = stratify_with(&output, &calls, &staged, &["--labelled-calls", &labelled]);
            assert_eq!(std::fs::read(&staged).unwrap(), before, "stratify wrote over it");
            let error = result.unwrap_err().to_string();
            assert_eq!(
                error,
                format!("the output {staged} is the input feature table; write to another path")
            );
        }

        #[test]
        fn review_stratify_labelled_calls_protects_only_scored_variant_types() {
            let tables = MixedTables::new(&[]);
            let (model, _) = bundled_model(&tables);
            let first = tables.path("S1.parquet");
            let labelled = tables.path("labelled");
            std::fs::create_dir(&labelled).unwrap();
            let truth = tables.path("truth.bed");
            std::fs::write(&truth, "chr1\t0\t100000\nchr2\t0\t100000\n").unwrap();
            let happy = tables.path("happy.vcf");
            for variant in ["indel", "snp"] {
                let second = format!("{labelled}/S1.v1.{variant}.parquet");
                std::fs::copy(&first, &second).unwrap();
                let before = std::fs::read(&second).unwrap();
                let result = try_run(&[
                    "stratify",
                    "--models",
                    &model,
                    "--tables",
                    &first,
                    &second,
                    "--truth-names",
                    "v1",
                    "v2",
                    "--truth-beds",
                    &truth,
                    &truth,
                    "--happy",
                    &happy,
                    &happy,
                    "-o",
                    &tables.path("run"),
                    "--labelled-calls",
                    &labelled,
                ]);
                if variant == "indel" {
                    result.unwrap();
                } else {
                    let error = format!("{:#}", result.unwrap_err());
                    assert!(
                        error.ends_with("is the input feature table; write to another path"),
                        "{error}"
                    );
                }
                assert_eq!(std::fs::read(&second).unwrap(), before);
            }
        }

        /// `score` with actions to withhold and no not-difficult regions fails once it has looked
        /// for the region BEDs, before it builds any feature (here, before it reads a missing
        /// gVCF).
        #[test]
        fn score_without_the_not_difficult_regions_to_withhold_actions_fails_early() {
            let tables = MixedTables::new(&[]);
            let (model, _) = bundled_model(&tables);
            // The calls without their raw annotations, which score then reads from the gVCF.
            let calls = read_gzip(Path::new(&tables.path("calls.vcf.gz")));
            let without_raw: String = calls
                .lines()
                .map(|line| {
                    if line.starts_with('#') {
                        return format!("{line}\n");
                    }
                    let mut fields: Vec<&str> = line.split('\t').collect();
                    let depth = fields[7].split(';').find(|kv| kv.starts_with("DP=")).unwrap();
                    fields[7] = depth;
                    format!("{}\n", fields.join("\t"))
                })
                .collect();
            let (vcf, out) = (tables.path("without_raw.vcf.gz"), tables.path("scored.vcf.gz"));
            write_bgzf_indexed(Path::new(&vcf), &without_raw);
            let gvcf = tables.path("missing.g.vcf.gz");
            let base = ["score", "--vcf", &vcf, "--models", &model, "-o", &out, "--gvcf", &gvcf];
            let error = try_run(&[&base[..], &["--abstain-actions", "flip"]].concat())
                .unwrap_err()
                .to_string();
            assert!(
                error.starts_with("--abstain-actions needs the not-difficult regions"),
                "{error}"
            );
            // With them, what fails is the missing gVCF, read later.
            let bed = tables.path("nd.bed");
            std::fs::write(&bed, "chr1\t0\t100000\n").unwrap();
            let region = format!("{NOT_DIFFICULT_REGIONS}={bed}");
            let rest = ["--abstain-actions", "flip", "--regions", &region];
            let error = format!("{:#}", try_run(&[&base[..], &rest].concat()).unwrap_err());
            assert!(error.contains("missing.g.vcf.gz"), "{error}");
            // Without actions to withhold, the regions are optional: the gVCF fails.
            let error = format!("{:#}", try_run(&base).unwrap_err());
            assert!(error.contains("missing.g.vcf.gz"), "{error}");
        }

        /// The region BEDs the tables are extracted with are recorded in the model, which `score`
        /// finds beside the models, with no --regions, scoring as with them; another BED there is
        /// an error naming it, unless --allow-region-mismatch.
        #[test]
        fn score_finds_and_checks_the_region_beds_beside_the_models() {
            let tables = MixedTables::new(&[]);
            let segdups = tables.path("segdups.bed");
            std::fs::write(&segdups, "chr1\t1000\t3000\nchr2\t0\t500\n").unwrap();
            let not_difficult = tables.path("not_difficult.bed");
            std::fs::write(&not_difficult, "chr1\t0\t100000\n").unwrap();
            let regions = format!("segdups={segdups}");
            // Extracts the sample's table, flagging segdups with `bed`, to `out`.
            let extract = |sample: &str, bed: &str, out: &str| {
                let table = tables.path(out);
                let (vcf, happy) = (tables.path("calls.vcf.gz"), tables.path("happy.vcf"));
                let regions = format!("segdups={bed}");
                run(&[
                    "extract",
                    "--vcf",
                    &vcf,
                    "--happy",
                    &happy,
                    "--sample",
                    sample,
                    "-o",
                    &table,
                    "--contigs",
                    "chr1",
                    "chr2",
                    "--regions",
                    &regions,
                ]);
                table
            };
            let (a, b) =
                (extract("S1", &segdups, "S1.parquet"), extract("S2", &segdups, "S2.parquet"));
            let models = tables.path("models");
            std::fs::create_dir_all(format!("{models}/regions")).unwrap();
            let model = format!("{models}/snp");
            let policy = format!("{NOT_DIFFICULT_REGIONS}={not_difficult}");
            run(&[
                "train",
                "--variant-type",
                "snp",
                "--max-rounds",
                "10",
                "-o",
                &model,
                "--policy-regions",
                &policy,
                "--tables",
                &a,
                &b,
            ]);
            let model = format!("{model}{MODEL_SUFFIX}");
            let info = ModelInfo::read(&model).unwrap();
            let digest = RegionSet::from_bed("segdups", &segdups).unwrap().digest().unwrap();
            assert_eq!(info.region_digests.get("segdups"), Some(&digest));
            let policy_digest = RegionSet::from_bed(NOT_DIFFICULT_REGIONS, &not_difficult)
                .unwrap()
                .digest()
                .unwrap();
            assert_eq!(info.policy_region_digests.get(NOT_DIFFICULT_REGIONS), Some(&policy_digest));

            let score = |output: &str, extra: &[&str]| {
                let vcf = tables.path("calls.vcf.gz");
                let out = tables.path(output);
                try_run(
                    &[&["score", "--vcf", &vcf, "--models", &model, "-o", &out][..], extra]
                        .concat(),
                )
                .map(|()| read_gzip(Path::new(&out)))
            };
            let given = score("given.vcf.gz", &["--regions", &regions]).unwrap();
            // Not found beside the models: missing.
            let error = score("missing.vcf.gz", &[]).unwrap_err().to_string();
            assert!(error.contains("needs region BEDs for: segdups"), "{error}");
            std::fs::copy(&segdups, format!("{models}/regions/segdups.bed")).unwrap();
            let bundled = score("bundled.vcf.gz", &[]).unwrap();
            let records = |vcf: &str| -> Vec<String> {
                vcf.lines().filter(|line| !line.starts_with('#')).map(str::to_string).collect()
            };
            assert_eq!(records(&bundled), records(&given));

            // Another BED beside the models.
            let swapped = format!("{models}/regions/segdups.bed");
            std::fs::write(&swapped, "chr1\t0\t10\n").unwrap();
            let error = score("swapped.vcf.gz", &[]).unwrap_err().to_string();
            // Named in the models' directory as resolved (e.g. macOS's /var is /private/var).
            let named = Path::new(&swapped).canonicalize().unwrap();
            assert!(
                error.starts_with(&format!(
                    "the segdups region BED {} (bundled) is not the one",
                    named.display()
                )),
                "{error}"
            );
            score("allowed.vcf.gz", &["--allow-region-mismatch"]).unwrap();
            // The swapped BED flags other calls, so a run with it is not the run with the
            // models' BED: the equality above is of the BED used.
            let flags = |bed: &str| {
                let table = extract("S1", bed, "flags.parquet");
                FeatureTable::read_parquet(&table)
                    .unwrap()
                    .f64_column("region_segdups")
                    .unwrap()
                    .to_vec()
            };
            assert_ne!(flags(&swapped), flags(&segdups));

            // evaluate finds the not-difficult regions beside the models when its policy reads
            // them (here to withhold flips), checked against the digest train recorded.
            std::fs::copy(&not_difficult, format!("{models}/regions/{NOT_DIFFICULT_REGIONS}.bed"))
                .unwrap();
            let evaluate = |extra: &[&str]| {
                let out = tables.path("evaluate.tsv");
                let base = ["evaluate", "--models", &model, "--table", &a, "-o", &out];
                try_run(&[&base[..], &["--abstain-actions", "flip"], extra].concat())
            };
            evaluate(&[]).unwrap();
            std::fs::write(format!("{models}/regions/{NOT_DIFFICULT_REGIONS}.bed"), "chr1\t0\t9\n")
                .unwrap();
            let error = evaluate(&[]).unwrap_err().to_string();
            assert!(error.contains("region BED"), "{error}");
            assert!(error.contains("is not the one the models were trained with"), "{error}");
            evaluate(&["--allow-region-mismatch"]).unwrap();
        }

        /// `--alignment-tags` records provenance in tables extracted without reads; `train` fails
        /// on tables aligned differently unless --mixed-alignments, which records the tags they
        /// share; `score` compares the tags given with the models'; and a model that records none
        /// is not checked, even with --strict-provenance.
        #[test]
        fn alignment_tags_are_recorded_and_compared() {
            let tables = MixedTables::new(&[]);
            let extract = |sample: &str, tags: &[&str]| {
                let table = tables.path(&format!("{sample}.parquet"));
                let (vcf, happy) = (tables.path("calls.vcf.gz"), tables.path("happy.vcf"));
                let base = ["extract", "--vcf", &vcf, "--happy", &happy, "--sample", sample];
                let rest = ["-o", &table, "--contigs", "chr1", "chr2", "--alignment-tags"];
                run(&[&base[..], &rest, tags].concat());
                table
            };
            let primary = ["aligner=bwa-mem3", "alt-project=primary"];
            let keep_alt = ["aligner=bwa-mem3", "alt-project=keep-alt"];
            let a = extract("S1", &primary);
            let b = extract("S2", &primary);
            let c = extract("S3", &keep_alt);
            let model = tables.train(&[a.clone(), b], "same");
            let recorded = model.info.alignment.unwrap();
            assert_eq!(recorded.to_string(), "aligner=bwa-mem3 alt-project=primary source=tags");

            let model_path = tables.path("conflict");
            let train = ["train", "--variant-type", "snp", "--max-rounds", "10", "-o", &model_path];
            let error =
                try_run(&[&train[..], &["--tables", &a, &c]].concat()).unwrap_err().to_string();
            assert!(
                error.starts_with(
                    "the tables of S1 and S3 are of reads aligned differently: alt-project: S1 \
                     primary, S3 keep-alt"
                ),
                "{error}"
            );
            run(&[&train[..], &["--mixed-alignments", "--tables", &a, &c]].concat());
            let mixed = WinnowrModel::load(
                format!("{model_path}{MODEL_SUFFIX}"),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap();
            assert_eq!(mixed.info.alignment.unwrap().to_string(), "aligner=bwa-mem3 source=tags");

            let score = |model: &str, extra: &[&str]| {
                let (vcf, model) = (tables.path("calls.vcf.gz"), tables.path(model));
                let out = tables.path("scored.vcf.gz");
                try_run(
                    &[&["score", "--vcf", &vcf, "--models", &model, "-o", &out][..], extra]
                        .concat(),
                )
            };
            let strict = ["--strict-provenance", "--alignment-tags"];
            score("same.winnowr.json", &[&strict[..], &primary].concat()).unwrap();
            let error = score("same.winnowr.json", &[&strict[..], &keep_alt].concat()).unwrap_err();
            assert!(error.to_string().contains("alt-project: models primary, reads keep-alt"));
            let error = score("same.winnowr.json", &["--strict-provenance"]).unwrap_err();
            assert!(error.to_string().contains("but the reads' alignment is not known"));
            score("same.winnowr.json", &[]).unwrap();
            // A model that records none is not checked.
            tables.train(
                &[tables.extract("S4", &["chr1", "chr2"]), tables.extract("S5", &["chr1", "chr2"])],
                "unknown",
            );
            score("unknown.winnowr.json", &["--strict-provenance"]).unwrap();
        }

        /// How a model of the alignments source is misused: `score` without the reads, or a table
        /// measured with other read settings, given to `train` or `evaluate`.
        #[derive(Debug, Clone, Copy)]
        enum Misuse {
            ScoreWithoutReads,
            TrainOnOtherSettings,
            EvaluateOtherSettings,
        }

        /// Each misuse of a model of the alignments source is rejected, saying what to do.
        #[rstest]
        #[case::score_without_reads(Misuse::ScoreWithoutReads)]
        #[case::train_on_other_settings(Misuse::TrainOnOtherSettings)]
        #[case::evaluate_other_settings(Misuse::EvaluateOtherSettings)]
        fn misuses_of_the_alignments_source_are_rejected(#[case] misuse: Misuse) {
            let setup = AlignmentsModel::new();
            let model = setup.path("reads.winnowr.json");
            let (error, expected) = match misuse {
                Misuse::ScoreWithoutReads => (
                    setup.score("no_reads.vcf.gz", &[]).unwrap_err(),
                    format!(
                        "model {model} was trained on features measured from the reads (feature \
                         source alignments:mapq20,bq10,flank5): pass --alignments and --reference"
                    ),
                ),
                Misuse::TrainOnOtherSettings => {
                    let dragen = setup.extract("S2", "dragen", &["--alignment-settings", "mapq1"]);
                    let tables = [setup.path("S1.parquet"), dragen];
                    let error = setup.train(&tables, "mixed").unwrap_err();
                    let expected = "--tables hold features from different sources: S1's from \
                                    alignments:mapq20,bq10,flank5, S2's from \
                                    alignments:mapq1,bq10,flank5; extract every training table \
                                    with the same --feature-source (and, with --alignments, the \
                                    same --alignment-settings)";
                    (error, expected.to_string())
                }
                Misuse::EvaluateOtherSettings => {
                    let dragen = setup.extract("S2", "dragen", &["--alignment-settings", "mapq1"]);
                    let metrics = setup.path("metrics.tsv");
                    let args = ["evaluate", "--models", &model, "--table", &dragen, "-o", &metrics];
                    let expected = format!(
                        "feature table {dragen} holds features from `alignments:mapq1,bq10,flank5`, \
                         but model {model} was trained on features from \
                         `alignments:mapq20,bq10,flank5`: extract the table again with \
                         --alignments, --reference and --alignment-settings mapq20,bq10,flank5, or use a model \
                         trained on `alignments:mapq1,bq10,flank5` features"
                    );
                    (try_run(&args).unwrap_err(), expected)
                }
            };
            assert_eq!(error.to_string(), expected);
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
                quals: None,
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
            let inputs = ReadInputs::new(Path::new(&vcf), Path::new(&bam), Path::new(&fasta));
            let measured =
                |options| sites_to_rows(&measure_regions(&inputs, &["chr1"], options).unwrap());
            let rows = measured(&options);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].n_spanning, 4);
            assert_ne!(rows, measured(&StrlenOptions::default()));
            assert_eq!(
                read_tables::<StrlenOptions>(&[&table]).unwrap(),
                StrlenTables::new(options, rows)
            );
        }
    }

    /// A training table's metadata: its sample, and its contigs with their lengths.
    fn training_info(sample: &str, lengths: &[(&str, Option<u64>)]) -> FeatureTableInfo {
        FeatureTableInfo {
            sample: sample.to_string(),
            contigs: lengths.iter().map(|(contig, _)| contig.to_string()).collect(),
            reference_contigs: Some(
                lengths.iter().map(|(contig, length)| (contig.to_string(), *length)).collect(),
            ),
            ..crate::testing::call_table(&[]).info().clone()
        }
    }

    /// A labelled table flagging a region with another BED than a model's is an error (a warning
    /// with --allow-region-mismatch); one recording no digest, or a model recording none, is not
    /// checked.
    #[test]
    fn tables_are_checked_against_the_models_region_beds() {
        let digest = |sha: &str| RegionDigest { sha256: sha.repeat(64), intervals: 1, bases: 10 };
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        model.info.region_digests = [("segdups".to_string(), digest("a"))].into_iter().collect();
        model.info.region_names = vec!["segdups".to_string()];
        let paths = [PathBuf::from("m.winnowr.json")];
        let models = std::slice::from_ref(&model);
        let table = Path::new("t.parquet");
        let mut info = crate::testing::call_table(&[]).info().clone();
        // The table records none: not checked (a warning).
        assert!(check_table_region_digests(table, &info, models, &paths, false).is_ok());
        info.region_digests = [("segdups".to_string(), digest("a"))].into_iter().collect();
        assert!(check_table_region_digests(table, &info, models, &paths, false).is_ok());
        info.region_digests = [("segdups".to_string(), digest("b"))].into_iter().collect();
        let error = check_table_region_digests(table, &info, models, &paths, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with("feature table t.parquet flags the segdups regions with another BED"),
            "{error}"
        );
        assert!(check_table_region_digests(table, &info, models, &paths, true).is_ok());
        // The model records none: not checked.
        model.info.region_digests.clear();
        let models = std::slice::from_ref(&model);
        assert!(check_table_region_digests(table, &info, models, &paths, false).is_ok());
    }

    #[rstest]
    fn review_region_digest_checks_only_active_features(
        #[values(false, true)] active: bool,
        #[values(false, true)] missing: bool,
    ) {
        let digest = |value: char| RegionDigest {
            sha256: value.to_string().repeat(64),
            intervals: 1,
            bases: 10,
        };
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        model.info.region_names = if active { vec!["segdups".to_string()] } else { Vec::new() };
        model.info.region_digests = BTreeMap::from([("segdups".to_string(), digest('a'))]);
        let mut info = crate::testing::call_table(&[]).info().clone();
        if !missing {
            info.region_digests.insert("segdups".to_string(), digest('b'));
        }
        let (result, logs) = captured_logs::capture(|| {
            check_table_region_digests(
                Path::new("table.parquet"),
                &info,
                &[model],
                &[PathBuf::from("model.winnowr.json")],
                false,
            )
        });
        assert_eq!(result.is_err(), active && !missing, "{result:?}");
        let warnings: Vec<_> =
            logs.iter().filter(|(_, message)| message.contains("records no digest")).collect();
        assert_eq!(warnings.len(), usize::from(active && missing), "{logs:?}");
    }

    #[rstest]
    fn review_policy_bed_names_are_unique_for_table_commands(
        #[values(false, true)] policy: bool,
        #[values(false, true)] reverse: bool,
    ) {
        let dir = tempfile::TempDir::new().unwrap();
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Indel);
        model.info.contigs = vec!["chr1".to_string()];
        let files = [ModelFile::resolve(&dir.path().join("model.winnowr.json"))];
        let mut given: Vec<NamedBed> = [100, 200]
            .iter()
            .map(|end| {
                let path = dir.path().join(format!("{end}.bed"));
                std::fs::write(&path, format!("chr1\t0\t{end}\n")).unwrap();
                NamedBed { name: NOT_DIFFICULT_REGIONS.to_string(), path }
            })
            .collect();
        if reverse {
            given.reverse();
        }
        let error = policy_not_difficult(&[&model.info], &files, &given, policy, false, &[])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("--regions gives notinalldifficultregions more than once"),
            "{error}"
        );
        let (one, _) =
            policy_not_difficult(&[&model.info], &files, &given[..1], policy, false, &[]).unwrap();
        assert_eq!(one.is_some(), policy);
    }

    /// Models that record different digests of one region are an error, as no labelled table can
    /// be of the BED all were trained with; models that record the same one are not.
    #[rstest]
    #[case::differ("b", true)]
    #[case::agree("a", false)]
    fn models_of_different_region_beds_are_rejected(#[case] other: &str, #[case] rejected: bool) {
        let digest = |sha: &str| RegionDigest { sha256: sha.repeat(64), intervals: 1, bases: 10 };
        let models: Vec<WinnowrModel> = [("a", VariantType::Snp), (other, VariantType::Indel)]
            .into_iter()
            .map(|(sha, variant_type)| {
                let mut model = crate::testing::separable_model(Task::Genotype, variant_type);
                model.info.region_names = vec!["segdups".to_string()];
                model.info.region_digests = BTreeMap::from([("segdups".to_string(), digest(sha))]);
                model
            })
            .collect();
        let paths = [PathBuf::from("snp.winnowr.json"), PathBuf::from("indel.winnowr.json")];
        let result = check_models_region_digests(&models, &ModelFile::resolve_all(&paths));
        if rejected {
            let error = result.unwrap_err().to_string();
            assert!(
                error.starts_with("--models were trained with different segdups region BEDs"),
                "{error}"
            );
        } else {
            result.unwrap();
        }
    }

    #[test]
    fn review_policy_region_names_are_validated_before_any_bed_is_read() {
        let missing = NamedBed {
            name: NOT_DIFFICULT_REGIONS.to_string(),
            path: PathBuf::from("missing-policy-region.bed"),
        };
        let duplicate =
            policy_region_digests(&[missing.clone(), missing.clone()]).unwrap_err().to_string();
        assert!(duplicate.ends_with("more than once"), "{duplicate}");
        let invalid = NamedBed { name: "segdups".to_string(), path: missing.path.clone() };
        let invalid = policy_region_digests(&[missing, invalid]).unwrap_err().to_string();
        assert!(invalid.starts_with("--policy-regions takes only"), "{invalid}");
    }

    /// `--policy-regions` takes the not-difficult regions only, once.
    #[test]
    fn policy_regions_are_the_not_difficult_regions_once() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("nd.bed");
        std::fs::write(&path, "chr1\t0\t10\n").unwrap();
        let not_difficult = NamedBed { name: NOT_DIFFICULT_REGIONS.to_string(), path };
        let recorded = policy_region_digests(std::slice::from_ref(&not_difficult)).unwrap();
        assert_eq!(recorded[NOT_DIFFICULT_REGIONS].bases, 10);
        let error =
            policy_region_digests(&[not_difficult.clone(), not_difficult]).unwrap_err().to_string();
        assert!(error.ends_with("more than once"), "{error}");
        let bed = NamedBed { name: "segdups".to_string(), path: PathBuf::from("x.bed") };
        let error = policy_region_digests(&[bed]).unwrap_err().to_string();
        assert!(
            error.starts_with("--policy-regions takes only notinalldifficultregions"),
            "{error}"
        );
    }

    #[test]
    fn merged_contigs_start_from_the_table_covering_the_most() {
        let odd = training_info("S1", &[("chr1", Some(100)), ("chr3", Some(300))]);
        let all =
            training_info("S2", &[("chr1", Some(100)), ("chr2", Some(200)), ("chr3", Some(300))]);
        for tables in [[&odd, &all], [&all, &odd]] {
            assert_eq!(merged_contigs(&tables), ["chr1", "chr2", "chr3"]);
        }
    }

    #[test]
    fn merged_contigs_do_not_depend_on_the_order_of_the_tables() {
        // Ties on the number of contigs are broken by sample name.
        let first = training_info("S1", &[("chr2", None)]);
        let second = training_info("S2", &[("chr1", None)]);
        for tables in [[&first, &second], [&second, &first]] {
            assert_eq!(merged_contigs(&tables), ["chr2", "chr1"]);
        }
        // Tables of the same contigs in different orders.
        let a = training_info("A", &[("chr2", None), ("chr1", None)]);
        let b = training_info("B", &[("chr1", None), ("chr2", None)]);
        for tables in [[&a, &b], [&b, &a]] {
            assert_eq!(merged_contigs(&tables), ["chr2", "chr1"]);
        }
    }

    #[test]
    fn finds_the_tables_sharing_no_contig_with_the_others() {
        let odd = training_info("S1", &[("chr1", None), ("chr3", None)]);
        let even = training_info("S2", &[("chr2", None)]);
        let all = training_info("S3", &[("chr1", None), ("chr2", None), ("chr3", None)]);
        let b37 = training_info("S4", &[("1", None), ("2", None)]);
        assert!(tables_sharing_no_contig(&[&odd, &even, &all]).is_empty());
        assert_eq!(tables_sharing_no_contig(&[&odd, &all, &b37]), ["S4"]);
        assert_eq!(tables_sharing_no_contig(&[&odd, &even]), ["S1", "S2"]);
    }

    #[test]
    fn merged_reference_contigs_follow_the_merged_contigs() {
        let odd = training_info("S1", &[("chr1", Some(100)), ("chr3", Some(300))]);
        let all =
            training_info("S2", &[("chr1", Some(100)), ("chr2", Some(200)), ("chr3", Some(300))]);
        let contigs = merged_contigs(&[&odd, &all]);
        let merged = merged_reference_contigs(&[&odd, &all], &contigs).unwrap().unwrap();
        let merged: Vec<(&str, Option<u64>)> =
            merged.iter().map(|(contig, length)| (contig.as_str(), *length)).collect();
        assert_eq!(merged, [("chr1", Some(100)), ("chr2", Some(200)), ("chr3", Some(300))]);
    }

    #[test]
    fn an_unknown_length_does_not_conflict_with_a_known_one() {
        let unknown = training_info("S1", &[("chr1", None), ("chr2", None)]);
        let known = training_info("S2", &[("chr1", Some(100))]);
        let contigs = merged_contigs(&[&unknown, &known]);
        let merged = merged_reference_contigs(&[&unknown, &known], &contigs).unwrap().unwrap();
        assert_eq!(merged.get("chr1"), Some(&Some(100)));
        assert_eq!(merged.get("chr2"), Some(&None));
    }

    #[test]
    fn different_lengths_of_a_contig_are_different_references() {
        let all = training_info("S2", &[("chr1", Some(100)), ("chr2", Some(200))]);
        let other = training_info("S3", &[("chr2", Some(250))]);
        let contigs = merged_contigs(&[&all, &other]);
        let message = merged_reference_contigs(&[&all, &other], &contigs).unwrap_err().to_string();
        assert!(
            message.contains("S2 and S3 record different lengths of chr2 (200 and 250)"),
            "{message}"
        );
    }

    #[test]
    fn no_reference_contigs_are_merged_when_a_table_records_none() {
        let all = training_info("S2", &[("chr1", Some(100))]);
        let mut unrecorded = training_info("S1", &[("chr1", Some(100))]);
        unrecorded.reference_contigs = None;
        let tables = [&unrecorded, &all];
        assert_eq!(merged_reference_contigs(&tables, &merged_contigs(&tables)).unwrap(), None);
    }

    #[test]
    fn different_lengths_are_different_references_when_a_table_records_none() {
        let first = training_info("A", &[("chr1", Some(100))]);
        let second = training_info("B", &[("chr1", Some(200)), ("chr2", Some(300))]);
        let mut unrecorded = training_info("C", &[("chr2", None)]);
        unrecorded.reference_contigs = None;
        let tables = [&first, &second, &unrecorded];
        let message =
            merged_reference_contigs(&tables, &merged_contigs(&tables)).unwrap_err().to_string();
        assert!(
            message.contains("A and B record different lengths of chr1 (100 and 200)"),
            "{message}"
        );
    }

    /// `evaluate` and `stratify` warn about a table whose reads were not aligned as a model's
    /// training reads were, or whose alignment is not known; a model that records none is not
    /// checked.
    #[test]
    fn tables_aligned_otherwise_than_the_models_warn() {
        let provenance = crate::testing::bwa_provenance;
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        let paths = [PathBuf::from("m.winnowr.json")];
        let mut info = crate::testing::call_table(&[]).info().clone();
        let table = Path::new("t.parquet");
        let warnings = |model: &WinnowrModel, info: &FeatureTableInfo| {
            table_alignment_warnings(table, info, std::slice::from_ref(model), &paths)
        };
        assert!(warnings(&model, &info).is_empty());
        model.info.alignment = Some(provenance("primary"));
        assert_eq!(
            warnings(&model, &info),
            [
                "model m.winnowr.json records how its training reads were aligned, but feature table \
              t.parquet records no alignment provenance, so it is not checked"
            ]
        );
        info.alignment = Some(provenance("primary"));
        assert!(warnings(&model, &info).is_empty());
        info.alignment = Some(provenance("keep-alt"));
        assert_eq!(
            warnings(&model, &info),
            ["feature table t.parquet is of reads not aligned as model m.winnowr.json's training \
              reads were: alt-project: model primary, table keep-alt"]
        );
    }

    #[rstest]
    #[case::single(&[&[("aligner", "bwa"), ("alt-project", "mixed")]], &[("aligner", "bwa"), ("alt-project", "mixed")], true)]
    #[case::merged(&[&[("alt-project", "mixed"), ("source", "tags")], &[("aligner", "bwa"), ("source", "tags")]], &[("aligner", "bwa"), ("alt-project", "mixed")], true)]
    #[case::conflicting(&[&[("aligner", "bwa"), ("alt-project", "mixed")], &[("aligner", "other"), ("alt-project", "mixed")]], &[("aligner", "bwa"), ("alt-project", "mixed")], true)]
    #[case::version_ignored(&[&[("aligner", "bwa"), ("version", "mixed")]], &[("aligner", "bwa"), ("version", "mixed")], false)]
    #[case::source_ignored(&[&[("aligner", "bwa"), ("source", "mixed")]], &[("aligner", "bwa"), ("source", "mixed")], false)]
    #[case::subset(&[&[("aligner", "bwa"), ("source", "tags")]], &[("aligner", "bwa"), ("alt-project", "mixed")], false)]
    fn pr38_indeterminate_table_warnings(
        #[case] model_tags: &[&[(&str, &str); 2]],
        #[case] table_tags: &[(&str, &str)],
        #[case] indeterminate: bool,
    ) {
        let models: Vec<_> = model_tags
            .iter()
            .map(|tags| {
                let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
                model.info.alignment = Some(tags.iter().copied().collect());
                model
            })
            .collect();
        let paths: Vec<_> =
            (0..models.len()).map(|index| PathBuf::from(format!("m{index}.json"))).collect();
        let info = FeatureTableInfo {
            alignment: Some(table_tags.iter().copied().collect()),
            ..FeatureTableInfo::default()
        };
        let warnings = table_alignment_warnings(Path::new("t.parquet"), &info, &models, &paths);
        assert_eq!(
            warnings.iter().filter(|warning| warning.contains("indeterminate")).count(),
            usize::from(indeterminate),
            "{warnings:?}"
        );
    }

    #[rstest]
    #[case::model(true, false, true)]
    #[case::table(false, true, true)]
    #[case::both(true, true, true)]
    #[case::legacy(false, true, false)]
    fn scoring_table_warnings_include_incomplete_options(
        #[case] model_incomplete: bool,
        #[case] table_incomplete: bool,
        #[case] model_known: bool,
    ) {
        let provenance = |incomplete| {
            let mut provenance = crate::testing::provenance(&[("aligner", "bwa")], false);
            if incomplete {
                provenance.0.insert("options-inference".to_string(), "incomplete".to_string());
            }
            provenance
        };
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        model.info.alignment = model_known.then(|| provenance(model_incomplete));
        let info = FeatureTableInfo {
            alignment: Some(provenance(table_incomplete)),
            ..FeatureTableInfo::default()
        };
        let warnings = table_alignment_warnings(
            Path::new("t.parquet"),
            &info,
            &[model],
            &[PathBuf::from("model.json")],
        );
        assert_eq!(
            warnings
                .iter()
                .filter(|warning| warning.contains("incomplete alignment settings"))
                .count(),
            if model_known {
                usize::from(model_incomplete) + usize::from(table_incomplete)
            } else {
                0
            },
            "{warnings:?}"
        );
        if model_known && model_incomplete {
            assert!(
                warnings.iter().any(|warning| {
                    warning.starts_with("model ")
                        && warning.contains("extract --alignment-tags options-inference=explicit")
                        && warning.contains("then retrain")
                }),
                "{warnings:?}"
            );
        }
    }

    /// `score` takes the models' feature source, and rejects models of different sources.
    #[test]
    fn score_takes_the_feature_source_of_the_models() {
        let info = |task: Task, source: FeatureSource| {
            let mut info = crate::testing::separable_model(task, VariantType::Snp).info;
            info.feature_source = source;
            info
        };
        let paths = [PathBuf::from("g.winnowr.json"), PathBuf::from("v.winnowr.json")];
        let (gvcf, vcf) = (FeatureSource::Gvcf, FeatureSource::Vcf);
        let (a, b) = (info(Task::Genotype, vcf), info(Task::Promote, vcf));
        assert_eq!(models_feature_source(&[&a, &b]), Some(vcf));
        let (a, b) = (info(Task::Genotype, gvcf), info(Task::Promote, vcf));
        assert_eq!(models_feature_source(&[&a, &b]), None);
        let problems = check_model_set(
            &ModelSet {
                paths: &paths,
                resolved: &paths,
                given_dirs: &crate::testing::given_dirs(&paths),
                infos: &[&a, &b],
            },
            &SetInputs::default(),
        );
        assert_eq!(
            error_message(&problems).unwrap(),
            "--models were trained on features from different sources (g.winnowr.json: gvcf; \
             v.winnowr.json: vcf): score with models trained on tables of one --feature-source \
             (and, with --alignments, the same --alignment-settings)"
        );
    }

    /// Saves a genotype model of `variant_type` using the regions `region_names` as `name`,
    /// trained on chr1 of `chr1_length` bp.
    fn save_model(
        dir: &Path,
        name: &str,
        variant_type: VariantType,
        region_names: &[&str],
        chr1_length: u64,
    ) -> String {
        let mut model = crate::testing::separable_model(Task::Genotype, variant_type);
        model.info.region_names = region_names.iter().map(ToString::to_string).collect();
        model.info.reference_contigs =
            Some([("chr1".to_string(), Some(chr1_length))].into_iter().collect());
        let path = dir.join(name);
        model.save(&path).unwrap();
        path.display().to_string()
    }

    /// Runs a command line.
    fn run_command(args: &[&str]) -> Result<()> {
        dispatch(&parse(args).unwrap().command, NonZeroUsize::new(1).unwrap())
    }

    #[rstest]
    #[case::invalid("badtag", false)]
    #[case::reserved("winnowr-merged-complete=yes", false)]
    #[case::valid("alt-project=primary", true)]
    fn review_check_models_reports_alignment_tag_validation(
        #[case] tags: &str,
        #[case] valid: bool,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let model = save_model(dir.path(), "model.winnowr.json", VariantType::Snp, &[], 100_000);
        let Command::CheckModels(args) =
            parse(&["check-models", "--models", &model, "--alignment-tags", tags, "--json"])
                .unwrap()
                .command
        else {
            panic!("not check-models")
        };
        let mut out = Vec::new();
        let mut err = Vec::new();
        let result = check_models_with_writers(&args, &mut out, &mut err);
        let report: serde_json::Value =
            serde_json::from_slice(&out).expect("a JSON report even for invalid tags");
        assert_eq!(result.is_ok(), valid, "{result:?}");
        assert_eq!(report["ok"], valid);
        assert_eq!(report["models"][0]["readable"], true);
        let problems = report["problems"].as_array().unwrap();
        assert_eq!(
            problems.iter().any(|p| p["check"] == "input_unreadable"
                && p["message"].as_str().unwrap().contains("--alignment-tags")),
            !valid,
            "{report}"
        );
        if valid {
            assert_eq!(report["inputs"]["alignment"]["alt-project"], "primary");
        }
    }

    /// With tags that cannot be read, the alignment header is not read: only the tags' error is
    /// reported, not one of an alignment file that is then never opened, nor that neither option
    /// was given. The `--alignments` given is still reported as given.
    #[rstest]
    fn review_check_models_skips_the_header_with_invalid_alignment_tags(
        #[values("badtag", "alt-project=primary")] tags: &str,
        #[values(false, true)] strict: bool,
    ) {
        let valid = !tags.contains("badtag");
        let dir = tempfile::tempdir().unwrap();
        let mut model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        model.info.alignment = Some(crate::testing::bwa_provenance("primary"));
        let model_path = dir.path().join("model.winnowr.json");
        model.save(&model_path).unwrap();
        let alignments = dir.path().join("absent.bam");
        let mut arguments = vec![
            "check-models",
            "--models",
            model_path.to_str().unwrap(),
            "--alignments",
            alignments.to_str().unwrap(),
            "--alignment-tags",
            tags,
            "--json",
        ];
        if strict {
            arguments.push("--strict-provenance");
        }
        let Command::CheckModels(args) = parse(&arguments).unwrap().command else {
            panic!("not check-models")
        };
        let mut out = Vec::new();
        let error = check_models_with_writers(&args, &mut out, Vec::new()).unwrap_err();
        let report: serde_json::Value = serde_json::from_slice(&out).unwrap();
        let problems = report["problems"].as_array().unwrap();
        let unreadable: Vec<&str> = problems
            .iter()
            .filter(|p| p["check"] == "input_unreadable")
            .map(|p| p["message"].as_str().unwrap())
            .collect();
        assert_eq!(unreadable.iter().any(|m| m.contains("absent.bam")), valid, "{unreadable:?}");
        assert_eq!(unreadable.iter().any(|m| m.contains("--alignment-tags")), !valid);
        assert!(
            !problems.iter().any(|p| p["check"] == "alignment_provenance_not_checked"),
            "{report}"
        );
        assert_eq!(report["inputs"]["alignments"], alignments.display().to_string());
        assert!(error.to_string().contains("1 error(s)"), "{error}");
    }

    /// `check-models` succeeds on a good set, and fails on a bad one; `score` reports every
    /// problem of its models at once.
    #[test]
    fn check_models_and_score_report_every_problem() {
        let dir = tempfile::TempDir::new().unwrap();
        let run = run_command;
        // The test VCF's chr1 is 100000 bp.
        let save = |name: &str, variant_type: VariantType, regions: &[&str]| {
            save_model(dir.path(), name, variant_type, regions, 100_000)
        };
        let snp = save("snp.winnowr.json", VariantType::Snp, &["segdups"]);
        let indel = save("indel.winnowr.json", VariantType::Indel, &[]);
        let other_snp = save("other.winnowr.json", VariantType::Snp, &[]);
        let bed = dir.path().join("segdups.bed");
        std::fs::write(&bed, "chr1\t0\t100\n").unwrap();
        let segdups = format!("segdups={}", bed.display());
        let fai = dir.path().join("ref.fa.fai");
        std::fs::write(&fai, "chr1\t100000\t6\t60\t61\n").unwrap();
        let fai = fai.display().to_string();
        let good =
            ["check-models", "--models", &snp, &indel, "--regions", &segdups, "--reference", &fai];
        run(&good).unwrap();
        run(&[&good[..], &["--json"]].concat()).unwrap();
        // The reference is compared: another length of chr1 fails.
        std::fs::write(&fai, "chr1\t99999\t6\t60\t61\n").unwrap();
        let error = run(&good).unwrap_err().to_string();
        assert_eq!(error, "the models have 2 error(s); see the problems reported");
        let bad = ["check-models", "--models", &snp, &other_snp, "--json"];
        let error = run(&bad).unwrap_err().to_string();
        assert_eq!(error, "the models have 2 error(s); see the problems reported");

        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &[]);
        let vcf = vcf.display().to_string();
        let out = dir.path().join("out.vcf.gz").display().to_string();
        let error = run(&["score", "--vcf", &vcf, "--models", &snp, &other_snp, "-o", &out])
            .unwrap_err()
            .to_string();
        let missing = format!(
            "model {snp} (genotype snp) needs region BEDs for: segdups; pass --regions NAME=PATH, \
             or put each beside the models as regions/NAME.bed.gz, regions/NAME.bed.bgz, \
             regions/NAME.bed (looked in: {}",
            // The models' directory as resolved (e.g. macOS's /var is /private/var).
            dir.path().canonicalize().unwrap().join("regions").display()
        );
        let expected = format!(
            "2 problems with the models and their inputs:\n  model_kinds: expected at most one \
             model per task and variant type, got two genotype snp models\n  region_missing: \
             {missing}"
        );
        assert!(error.starts_with(&expected), "{error}");
        // One problem: its message alone, as before.
        let error =
            run(&["score", "--vcf", &vcf, "--models", &snp, "-o", &out]).unwrap_err().to_string();
        assert!(error.starts_with(&missing), "{error}");
    }

    /// `score` checks the model set before it parses the boosters: a set that cannot be scored
    /// fails on its problems, even with a booster XGBoost cannot load.
    #[test]
    fn score_checks_the_set_before_parsing_the_boosters() {
        let dir = tempfile::TempDir::new().unwrap();
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &[]);
        let vcf = vcf.display().to_string();
        let out = dir.path().join("out.vcf.gz").display().to_string();
        let a = save_model(dir.path(), "a.winnowr.json", VariantType::Snp, &[], 100_000);
        let b = save_model(dir.path(), "b.winnowr.json", VariantType::Snp, &[], 100_000);
        let text = std::fs::read_to_string(&b).unwrap();
        let start = text.find("\"booster\"").unwrap();
        std::fs::write(&b, format!("{}\"booster\":{{\"x\":[NaN]}}}}", &text[..start])).unwrap();
        let score = |models: &[&str]| {
            run_command(&[&["score", "--vcf", &vcf, "-o", &out, "--models"], models].concat())
                .unwrap_err()
        };
        let error = score(&[&a, &b]).to_string();
        assert!(error.ends_with("got two genotype snp models"), "{error}");
        let error = format!("{:#}", score(&[&b]));
        assert!(error.starts_with(&format!("failed to load model {b}: ")), "{error}");
    }

    /// `score` rejects a repeated --regions name, even one no model uses, and a VCF that is not on
    /// the models' reference.
    #[test]
    fn score_checks_the_region_names_and_the_vcf_reference() {
        let dir = tempfile::TempDir::new().unwrap();
        let header = crate::testing::vcf_header(&[crate::testing::CALL_FIELDS], &["S1"]);
        let vcf = crate::testing::write_indexed_vcf(dir.path(), "in.vcf.gz", &header, &[]);
        let vcf = vcf.display().to_string();
        let out = dir.path().join("out.vcf.gz").display().to_string();
        let snp = save_model(dir.path(), "snp.winnowr.json", VariantType::Snp, &[], 100_000);
        let bed = dir.path().join("unused.bed");
        std::fs::write(&bed, "chr1\t0\t100\n").unwrap();
        let unused = format!("unused={}", bed.display());
        let score = ["score", "--vcf", &vcf, "--models", &snp, "-o", &out];
        let error = run_command(&[&score[..], &["--regions", &unused, &unused]].concat())
            .unwrap_err()
            .to_string();
        assert_eq!(error, "region names must be unique; repeated: unused");

        let short = save_model(dir.path(), "short.winnowr.json", VariantType::Snp, &[], 1000);
        let error = run_command(&["score", "--vcf", &vcf, "--models", &short, "-o", &out])
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            format!(
                "VCF {vcf} is not on the reference model {short} was trained on: other lengths: \
                 chr1 (100000 bp, model 1000 bp)"
            )
        );
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
        assert_eq!(args.dp_median, None);

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
            "--dp-median",
            "31.5",
        ])
        .unwrap();
        let Command::Extract(args) = cli.command else { panic!("not extract") };
        assert_eq!(args.dp_median, Some(31.5));
        assert_eq!(args.regions.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(args.contigs, ["chr20", "chr21"]);
        assert_eq!(args.sample.as_deref(), Some("S"));
        assert_eq!(args.gvcf, Some(PathBuf::from("in.g.vcf.gz")));
        assert_eq!(args.feature_source, None);
        let source = extract_feature_source(&args).unwrap();
        assert_eq!(source, FeatureSource::Gvcf);
        check_unread_gvcf(source, args.gvcf.as_deref(), UnreadGvcf::Error).unwrap();
        let prepared = prepare(source, args.gvcf.as_deref(), None).unwrap();
        assert_eq!(prepared.annotations(), AnnotationInputs::Gvcf(args.gvcf.as_deref()));
    }

    /// [`prepare_source`] of a source, its gVCF and its reads, with no pileup, for a table of
    /// chr1 with no regions, truth or depth median.
    fn prepare<'a>(
        source: FeatureSource,
        gvcf: Option<&'a Path>,
        reads: Option<ReadInputs<'_>>,
    ) -> Result<SourceInputs<'a>> {
        let (contigs, truth) = (["chr1".to_string()], TruthInputs::default());
        let vcf = Path::new("in.vcf.gz");
        let checks =
            TableChecks { vcf, contigs: &contigs, regions: &[], truth: &truth, dp_median: None };
        prepare_source(source, gvcf, reads, None, &checks)
    }

    /// The options of an `extract` command line with an input VCF and an output table, and
    /// `extra` options.
    fn extract_args(extra: &[&str]) -> ExtractArgs {
        let base = ["extract", "--vcf", "in.vcf.gz", "-o", "t.parquet"];
        let Command::Extract(args) = parse(&[&base[..], extra].concat()).unwrap().command else {
            panic!("not extract")
        };
        args
    }

    /// `extract --feature-source` takes gvcf (the default), vcf or alignments, in any case; vcf
    /// does not read --gvcf, so it cannot be given with it.
    #[test]
    fn extract_takes_a_feature_source() {
        assert_eq!(extract_feature_source(&extract_args(&[])).unwrap(), FeatureSource::Gvcf);
        let vcf = extract_args(&["--feature-source", "VCF"]);
        assert_eq!(vcf.feature_source, Some(ExtractSource::Vcf));
        assert_eq!(extract_feature_source(&vcf).unwrap(), FeatureSource::Vcf);
        check_unread_gvcf(FeatureSource::Vcf, None, UnreadGvcf::Error).unwrap();
        assert_eq!(
            prepare(FeatureSource::Vcf, None, None).unwrap().annotations(),
            AnnotationInputs::Vcf
        );
        let with_gvcf = extract_args(&["--feature-source", "vcf", "--gvcf", "in.g.vcf.gz"]);
        let source = extract_feature_source(&with_gvcf).unwrap();
        let error = check_unread_gvcf(source, with_gvcf.gvcf.as_deref(), UnreadGvcf::Error)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not read --gvcf"), "{error}");
        // score, whose source comes from the models, ignores it.
        check_unread_gvcf(FeatureSource::Vcf, Some(Path::new("g")), UnreadGvcf::Warn).unwrap();
        let base = ["extract", "--vcf", "in.vcf.gz", "-o", "t.parquet"];
        let error = parse(&[&base[..], &["--feature-source", "bam"]].concat()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidValue);
        let command = Cli::command();
        let extract = command.find_subcommand("extract").unwrap();
        let option = extract.get_arguments().find(|a| a.get_id() == "feature_source").unwrap();
        let values: Vec<String> =
            option.get_possible_values().iter().map(|v| v.get_name().to_string()).collect();
        assert_eq!(values, ["gvcf", "vcf", "alignments"]);
        assert_eq!(ExtractSource::Alignments.to_string(), "alignments");
    }

    /// `extract --alignments` selects the alignments source, with the read settings of
    /// --alignment-settings (the defaults without it); the source needs the reads, and no other
    /// source takes them.
    #[test]
    fn extract_alignments_select_the_alignments_source() {
        let reads = ["--alignments", "in.cram", "--reference", "ref.fa"];
        let source =
            |extra: &[&str]| extract_feature_source(&extract_args(&[&reads[..], extra].concat()));
        let default = FeatureSource::VcfAlignments(ReadAnnotationOptions::default());
        assert_eq!(source(&[]).unwrap(), default);
        assert_eq!(source(&["--feature-source", "alignments"]).unwrap(), default);
        let dragen = source(&["--alignment-settings", "mapq1"]).unwrap();
        assert_eq!(dragen.to_string(), "alignments:mapq1,bq10,flank5");
        let measured = SourceInputs::Reads {
            options: ReadAnnotationOptions::default(),
            sites: Vec::new(),
            pileup: None,
        };
        assert_eq!(measured.annotations().feature_source(), default);
        // The source does not read a --gvcf: extract rejects it, and score ignores it.
        assert!(check_unread_gvcf(default, Some(Path::new("g")), UnreadGvcf::Error).is_err());
        check_unread_gvcf(default, Some(Path::new("g")), UnreadGvcf::Warn).unwrap();
        // It needs the reads.
        let error = prepare(default, None, None).unwrap_err().to_string();
        assert!(error.ends_with("pass --alignments and --reference"), "{error}");
        for other in ["gvcf", "vcf"] {
            let error = source(&["--feature-source", other]).unwrap_err().to_string();
            assert!(
                error.starts_with("--alignments is read only for the alignments feature source"),
                "{error}"
            );
        }
        let args = extract_args(&["--feature-source", "alignments"]);
        let error = extract_feature_source(&args).unwrap_err().to_string();
        assert!(error.ends_with("pass --alignments and --reference"), "{error}");
    }

    #[rstest]
    #[case::alignments_without_reference(
        &["--alignments", "in.cram"],
        ErrorKind::MissingRequiredArgument
    )]
    #[case::alignments_with_gvcf(
        &["--alignments", "in.cram", "--reference", "ref.fa", "--gvcf", "in.g.vcf.gz"],
        ErrorKind::ArgumentConflict
    )]
    #[case::settings_without_alignments(
        &["--alignment-settings", "mapq1"],
        ErrorKind::MissingRequiredArgument
    )]
    #[case::bad_settings(
        &["--alignments", "in.cram", "--reference", "ref.fa", "--alignment-settings", "depth3"],
        ErrorKind::ValueValidation
    )]
    fn extract_rejects_misused_alignment_options(#[case] extra: &[&str], #[case] kind: ErrorKind) {
        let base = ["extract", "--vcf", "in.vcf.gz", "-o", "t.parquet"];
        assert_eq!(parse(&[&base[..], extra].concat()).unwrap_err().kind(), kind);
    }

    /// A bad --alignment-settings value is reported through clap with what is expected (the
    /// settings' errors are tested in [`crate::read_settings`]).
    #[test]
    fn bad_alignment_settings_say_what_is_expected() {
        let reads = ["--alignments", "in.cram", "--reference", "ref.fa"];
        let base = ["extract", "--vcf", "in.vcf.gz", "-o", "t.parquet"];
        let args = [&base[..], &reads[..], &["--alignment-settings", "mapq256"]].concat();
        let error = parse(&args).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
        let expected = "bad MAPQ in mapq256: expected an integer from 0 to 255";
        assert!(error.to_string().contains(expected), "{error}");
    }

    /// The reference reaches the truth matching unless it is there only for --alignments.
    #[rstest]
    #[case::without_alignments(&[], true)]
    #[case::for_the_alignments_only(&["--alignments", "in.cram"], false)]
    #[case::with_the_truth(
        &["--alignments", "in.cram", "--truth-vcf", "t.vcf.gz", "--truth-bed", "t.bed"],
        true
    )]
    fn the_reference_matches_the_truth_unless_only_the_reads_need_it(
        #[case] extra: &[&str],
        #[case] kept: bool,
    ) {
        let args = extract_args(&[&["--reference", "ref.fa"][..], extra].concat());
        assert_eq!(truth_reference(&args).is_some(), kept);
    }

    /// A model trained on features measured from the reads needs them to score.
    #[test]
    fn score_needs_alignments_for_models_of_the_alignments_source() {
        let score_args = |extra: &[&str]| {
            let base = ["score", "--vcf", "v", "--models", "m.winnowr.json", "-o", "o"];
            let Command::Score(args) = parse(&[&base[..], extra].concat()).unwrap().command else {
                panic!("not score")
            };
            args
        };
        let options: ReadAnnotationOptions = "mapq1".parse().unwrap();
        let source = FeatureSource::VcfAlignments(options);
        let error = models_read_annotations(source, &score_args(&[])).unwrap_err().to_string();
        assert_eq!(
            error,
            "model m.winnowr.json was trained on features measured from the reads (feature \
             source alignments:mapq1,bq10,flank5): pass --alignments and --reference"
        );
        let with_reads = score_args(&["--alignments", "in.cram", "--reference", "ref.fa"]);
        assert_eq!(models_read_annotations(source, &with_reads).unwrap(), Some(options));
        for other in [FeatureSource::Gvcf, FeatureSource::Vcf] {
            assert_eq!(models_read_annotations(other, &score_args(&[])).unwrap(), None);
        }
    }

    /// `evaluate` and `stratify` reject a table of another feature source than a model's, either
    /// way round.
    #[test]
    fn a_table_of_another_feature_source_than_the_model_is_rejected() {
        let model = crate::testing::separable_model(Task::Genotype, VariantType::Snp);
        let paths = [PathBuf::from("m.winnowr.json")];
        let table = Path::new("t.parquet");
        let mut info = crate::testing::call_table(&[]).info().clone();
        check_table_feature_source(table, &info, std::slice::from_ref(&model), &paths).unwrap();
        info.feature_source = FeatureSource::Vcf;
        let error = check_table_feature_source(table, &info, &[model], &paths).unwrap_err();
        assert_eq!(
            error.to_string(),
            "feature table t.parquet holds features from `vcf`, but model m.winnowr.json was \
             trained on features from `gvcf`: extract the table again with --feature-source gvcf \
             (with --gvcf if the VCF lacks the raw annotations), or use a model trained on `vcf` \
             features"
        );
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
        assert_eq!(args.dp_median, None);
        let cli = parse(&["score", "--vcf", "v", "--models", "m", "-o", "o", "--dp-median", "40"])
            .unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        assert_eq!(args.dp_median, Some(40.0));
    }

    #[rstest]
    #[case::zero("0", "got 0.0")]
    #[case::negative("-3", "got -3.0")]
    #[case::below_one_read("0.5", "got 0.5")]
    #[case::tiny("1e-300", "got 1e-300")]
    #[case::nan("NaN", "got NaN")]
    #[case::infinite("inf", "got inf")]
    #[case::text("deep", "not a number: deep")]
    fn dp_median_must_be_a_depth_of_at_least_one_read(#[case] value: &str, #[case] got: &str) {
        let message = if got.starts_with("got") {
            format!("the depth median must be a finite depth of at least 1, {got}")
        } else {
            got.to_string()
        };
        // `=` so that a negative value is not read as an option.
        let option = format!("--dp-median={value}");
        let extract = ["extract", "--vcf", "v", "-o", "o", &option];
        let score = ["score", "--vcf", "v", "--models", "m", "-o", "o", &option];
        for args in [&extract[..], &score[..]] {
            let error = parse(args).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ValueValidation, "{args:?}");
            assert!(error.to_string().contains(&message), "{error}");
        }
        let one = parse(&["score", "--vcf", "v", "--models", "m", "-o", "o", "--dp-median", "1"]);
        let Command::Score(args) = one.unwrap().command else { panic!("not score") };
        assert_eq!(args.dp_median, Some(1.0), "one read is the smallest depth");
    }

    /// `score` builds its feature table with the depth median of its command line.
    #[test]
    fn score_builds_its_table_with_the_given_depth_median() {
        for (extra, expected) in [(&[][..], None), (&["--dp-median", "40"][..], Some(40.0))] {
            let base = ["score", "--vcf", "v", "--models", "m", "-o", "o"];
            let cli = parse(&[&base[..], extra].concat()).unwrap();
            let Command::Score(args) = cli.command else { panic!("not score") };
            assert_eq!(
                score_table_inputs(&args, AnnotationInputs::default(), &[], None, None).dp_median,
                expected
            );
        }
    }

    /// The policy of a `score` command line with extra options, and the not-difficult regions.
    fn score_policy(extra: &[&str], not_difficult: Option<RegionSet>) -> Result<Policy> {
        let base = ["score", "--vcf", "v", "--models", "m", "-o", "o"];
        let cli = parse(&[&base[..], extra].concat()).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        args.policy.policy(Mode::Genotyper, &["chr1"], not_difficult, true)
    }

    /// Not-difficult regions of one chr1 interval.
    fn not_difficult_regions() -> Option<RegionSet> {
        Some(RegionSet::from_spans(NOT_DIFFICULT_REGIONS, [("chr1", 100, 200)]))
    }

    #[test]
    fn the_default_policy_keeps_the_fallback_for_indels_and_an_outside_drop_threshold() {
        use crate::policy::{DEFAULT_KEEP_FALLBACK, DEFAULT_OUTSIDE_INDEL_T_DROP};
        // Without the not-difficult regions: the indel keep fallback only.
        let policy = score_policy(&[], None).unwrap();
        let expected: Vec<(VariantType, Rule)> =
            DEFAULT_KEEP_FALLBACK.iter().map(|&vt| (vt, Rule::KeepFallback)).collect();
        assert_eq!(policy.rules.into_iter().collect::<Vec<_>>(), expected);
        assert!(policy.abstain.is_none());
        // With them: indels outside are dropped at the outside threshold, in place of the
        // model's; no action is withheld.
        let policy = score_policy(&[], not_difficult_regions()).unwrap();
        let abstain = policy.abstain.unwrap();
        assert_eq!(abstain.indel_t_drop, Some(DEFAULT_OUTSIDE_INDEL_T_DROP));
        assert!(!abstain.flip && !abstain.regenotype && !abstain.promote);
        // Regions on none of the contigs are an error.
        let elsewhere = RegionSet::from_spans(NOT_DIFFICULT_REGIONS, [("chr9", 0, 10)]);
        assert!(score_policy(&[], Some(elsewhere)).is_err());
    }

    #[test]
    fn withheld_actions_read_the_not_difficult_regions_without_an_indel_genotype_model() {
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
        assert!(args.policy.reads_not_difficult(false));
        let policy =
            args.policy.policy(Mode::Genotyper, &["chr1"], not_difficult_regions(), false).unwrap();
        let abstain = policy.abstain.unwrap();
        assert!(abstain.flip && !abstain.regenotype && !abstain.promote);
        assert_eq!(abstain.indel_t_drop, args.policy.outside_indel_t_drop.0);
    }

    #[test]
    fn the_not_difficult_regions_are_not_read_without_an_indel_genotype_model() {
        let cli = parse(&["score", "--vcf", "v", "--models", "m", "-o", "o"]).unwrap();
        let Command::Score(args) = cli.command else { panic!("not score") };
        assert!(!args.policy.reads_not_difficult(false));
        assert!(args.policy.reads_not_difficult(true));
        let policy =
            args.policy.policy(Mode::Genotyper, &["chr1"], not_difficult_regions(), true).unwrap();
        assert!(policy.abstain.is_some());
    }

    /// The policy reads the not-difficult regions to withhold actions outside them, or, for an
    /// indel genotype model, to drop indel calls outside them at the outside threshold.
    #[rstest]
    #[case::default_snp(&[], false, false)]
    #[case::default_indel(&[], true, true)]
    #[case::no_outside_threshold(&["--outside-indel-t-drop", "none"], true, false)]
    #[case::abstain_snp(&["--abstain-actions", "flip"], false, true)]
    #[case::abstain_without_threshold(&["--abstain-actions", "promote", "--outside-indel-t-drop", "none"], false, true)]
    #[case::abstain_none(&["--abstain-actions", "none"], true, true)]
    fn when_the_policy_reads_the_not_difficult_regions(
        #[case] extra: &[&str],
        #[case] indel_genotype: bool,
        #[case] expected: bool,
    ) {
        let base = ["score", "--vcf", "v", "--models", "m", "-o", "o"];
        let Command::Score(args) = parse(&[&base[..], extra].concat()).unwrap().command else {
            panic!("not score")
        };
        assert_eq!(args.policy.reads_not_difficult(indel_genotype), expected);
    }

    #[test]
    fn the_default_policy_can_be_turned_off() {
        // It reads no not-difficult regions (see
        // `when_the_policy_reads_the_not_difficult_regions`), so it is given none.
        let policy =
            score_policy(&["--keep-fallback", "none", "--outside-indel-t-drop", "none"], None)
                .unwrap();
        assert!(policy.rules.is_empty() && policy.abstain.is_none());
        // --expected-f1 replaces the default keep fallback of its variant types.
        let policy = score_policy(&["--expected-f1", "indel"], None).unwrap();
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
            not_difficult_regions(),
        )
        .unwrap();
        assert_eq!(policy.rule(VariantType::Snp), Rule::KeepFallback);
        let abstain = policy.abstain.unwrap();
        assert_eq!((abstain.indel_t_drop, abstain.flip), (Some(0.7), true));
        // Withholding actions needs the not-difficult regions.
        let error = score_policy(&["--abstain-actions", "flip"], None).unwrap_err();
        assert!(error.to_string().contains(NOT_DIFFICULT_REGIONS), "{error}");
    }

    #[rstest]
    #[case::keep_fallback(&["--keep-fallback", "none", "snp"], "--keep-fallback none cannot be given with other values, got: none snp")]
    #[case::keep_fallback_any_case(&["--keep-fallback", "INDEL", "None"], "--keep-fallback none cannot be given with other values, got: indel none")]
    #[case::abstain_actions(&["--abstain-actions", "flip", "none"], "--abstain-actions none cannot be given with other values, got: flip none")]
    fn none_cannot_be_given_with_other_values(#[case] extra: &[&str], #[case] expected: &str) {
        let error = score_policy(extra, not_difficult_regions()).unwrap_err().to_string();
        assert_eq!(error, expected);
    }

    #[test]
    fn abstain_actions_ignore_case() {
        let policy =
            score_policy(&["--abstain-actions", "Flip", "PROMOTE"], not_difficult_regions())
                .unwrap();
        let abstain = policy.abstain.unwrap();
        assert!(abstain.flip && abstain.promote && !abstain.regenotype);
        // `none` alone, in any case, withholds nothing.
        let policy = score_policy(&["--abstain-actions", "NONE"], not_difficult_regions()).unwrap();
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
