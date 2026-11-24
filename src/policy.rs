//! The decision policy: how models' class probabilities become drops, promotions, flips and
//! re-genotypes of a sample's calls, as `score` applies them (see [`crate::scoring`]) and
//! `evaluate` and `stratify` report them.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result, bail};
use log::info;
use rayon::prelude::*;

use crate::common::{PASS, PhaseTimer, Task, VariantType};
use crate::model::{ModelInfo, Predicting, WinnowrModel};
use crate::promote_labels::promote_eligible;
use crate::regenotype::{
    RegenotypeTarget, regenotype_targets, with_regenotype_columns, without_regenotype,
};
use crate::regions::{RegionSet, ref_span};
use crate::rescore::{
    CallFlags, CallLimits, Mode, Promotions, Rule, decide_promote, decide_with_rule,
};
use crate::stutter::add_length_call_features;
use crate::table::FeatureTable;

/// The name of the region set (`--regions NAME=PATH`) that the default policy acts fully inside:
/// GIAB's GRCh38 genome stratification "notinalldifficultregions" (sample-independent).
pub const NOT_DIFFICULT_REGIONS: &str = "notinalldifficultregions";

/// True if a model is an indel genotype model, which the default policy decides with the
/// [`NOT_DIFFICULT_REGIONS`] (see [`DEFAULT_OUTSIDE_INDEL_T_DROP`]).
#[must_use]
pub fn is_indel_genotype(info: &ModelInfo) -> bool {
    info.task == Task::Genotype && info.variant_type == VariantType::Indel
}

/// The variant types whose genotype models use [`Rule::KeepFallback`] by default.
pub const DEFAULT_KEEP_FALLBACK: [VariantType; 1] = [VariantType::Indel];

/// The P(drop) at or above which the default policy drops indel calls outside
/// [`NOT_DIFFICULT_REGIONS`], in place of the model's drop threshold.
pub const DEFAULT_OUTSIDE_INDEL_T_DROP: f64 = 0.6;

/// How models' probabilities become actions: the genotype models' mode and rules, and the
/// actions withheld outside a region set.
#[derive(Debug, Clone)]
pub struct Policy {
    /// The decision mode of genotype models (their thresholds for it are used).
    pub mode: Mode,
    /// The rule of genotype models per variant type ([`Rule::Thresholds`] if absent).
    pub rules: BTreeMap<VariantType, Rule>,
    /// Actions withheld from calls outside a region set.
    pub abstain: Option<Abstention>,
}

impl Policy {
    /// The policy in a mode with the thresholds alone, everywhere (the command line's defaults add
    /// [`DEFAULT_KEEP_FALLBACK`] and [`DEFAULT_OUTSIDE_INDEL_T_DROP`]).
    #[must_use]
    pub fn new(mode: Mode) -> Self {
        Policy { mode, rules: BTreeMap::new(), abstain: None }
    }

    /// The rule of genotype models of a variant type.
    #[must_use]
    pub fn rule(&self, variant_type: VariantType) -> Rule {
        self.rules.get(&variant_type).copied().unwrap_or_default()
    }

    /// The limits on the actions of a genotype model on its calls of a variant type, per call
    /// (`None` when no actions are withheld anywhere).
    ///
    /// # Errors
    ///
    /// Returns an error if the calls lack `chrom`, `pos` or `ref`.
    pub fn call_limits(
        &self,
        calls: &FeatureTable,
        variant_type: VariantType,
    ) -> Result<Option<Vec<CallLimits>>> {
        let Some(abstain) = &self.abstain else { return Ok(None) };
        let outside = abstain.outside(calls)?;
        Ok(Some(outside.into_iter().map(|out| abstain.limits(variant_type, out)).collect()))
    }

    /// Per promotion candidate, whether promoting it is withheld (`None` when no promotions are).
    ///
    /// # Errors
    ///
    /// Returns an error if the candidates lack `chrom`, `pos` or `ref`.
    pub fn withheld_promotions(&self, calls: &FeatureTable) -> Result<Option<Vec<bool>>> {
        match &self.abstain {
            Some(abstain) if abstain.promote => abstain.outside(calls).map(Some),
            _ => Ok(None),
        }
    }
}

/// Actions withheld from calls whose REF span is not inside a region set (e.g. GIAB's
/// "notinalldifficultregions"), where the models' training labels are scarce.
#[derive(Debug, Clone)]
pub struct Abstention {
    /// The calls inside these regions are acted on as usual.
    pub regions: RegionSet,
    /// Do not flip calls outside.
    pub flip: bool,
    /// Do not re-genotype calls outside.
    pub regenotype: bool,
    /// Do not promote (nor flip promoted) calls outside.
    pub promote: bool,
    /// The drop threshold of indel calls outside, in place of the model's (see
    /// [`crate::rescore::CallLimits::t_drop`]): drops for a P(keep) below one half are not held
    /// to it.
    pub indel_t_drop: Option<f64>,
}

impl Abstention {
    /// Per call of a table, whether its REF span is outside the regions.
    ///
    /// # Errors
    ///
    /// Returns an error if the table lacks `chrom`, `pos` or `ref`.
    pub fn outside(&self, table: &FeatureTable) -> Result<Vec<bool>> {
        let (chroms, positions, refs) =
            (table.str_column("chrom")?, table.i64_column("pos")?, table.str_column("ref")?);
        Ok((0..table.num_rows())
            .into_par_iter()
            .map(|i| {
                let (start, end) = ref_span(positions[i], refs[i].len());
                !self.regions.contains(chroms[i], start, end)
            })
            .collect())
    }

    /// The limits on a genotype model's call of a variant type, by whether it is outside.
    #[must_use]
    pub fn limits(&self, variant_type: VariantType, outside: bool) -> CallLimits {
        if !outside {
            return CallLimits::default();
        }
        CallLimits {
            no_flip: self.flip,
            no_regenotype: self.regenotype,
            t_drop: if variant_type == VariantType::Indel { self.indel_t_drop } else { None },
        }
    }
}

/// A call's `(chrom, pos, ref, alt, filter)`, with `alt` comma-joined and `filter` `;`-joined.
pub type CallKey = (String, u64, String, String, String);

/// A model's decision for one call.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// The task of the model that scored the call.
    pub task: Task,
    /// The model's class probabilities ([`crate::labels::Label`] or promote order).
    pub probabilities: Vec<f64>,
    /// Drop the call.
    pub drop: bool,
    /// Promote the call to PASS.
    pub promote: bool,
    /// Switch the call's genotype between `0/1` and `1/1`.
    pub flip: bool,
    /// The re-genotyped call's alleles and genotype, if it is re-genotyped.
    pub regenotype: Option<RegenotypeTarget>,
}

/// The calls a model scores: PASS calls of its variant type (genotype models), or non-PASS calls
/// of its variant type that do not overlap a PASS call (promote models, see
/// [`promote_eligible`]).
///
/// # Errors
///
/// Returns an error if the table lacks `variant_type`, `filter` or (promote models)
/// `overlaps_pass`.
pub fn scored_mask(table: &FeatureTable, model: &WinnowrModel) -> Result<Vec<bool>> {
    let types = table.str_column("variant_type")?;
    let variant_type = model.info.variant_type.as_str();
    let eligible = match model.info.task {
        Task::Genotype => table.str_column("filter")?.iter().map(|f| *f == PASS).collect(),
        Task::Promote => promote_eligible(table)?,
    };
    Ok(types.iter().zip(eligible).map(|(vt, eligible)| *vt == variant_type && eligible).collect())
}

/// Scores each model's calls and decides drops, promotions, flips and re-genotypes.
///
/// # Arguments
///
/// * `table` - the feature table (with strlen evidence for models with a stutter model).
/// * `models` - at most one genotype and one promote model per variant type.
/// * `policy` - how the probabilities become actions.
///
/// Returns the decision for each scored call (the first of calls sharing a key).
///
/// # Errors
///
/// Returns an error if two models share a task and variant type, or a model's features cannot be
/// built.
pub fn decide_calls(
    table: &FeatureTable,
    models: &[WinnowrModel],
    policy: &Policy,
) -> Result<HashMap<CallKey, Decision>> {
    check_model_kinds(models.iter().map(|model| &model.info))?;
    let per_model = models
        .iter()
        .map(|model| decide_model_calls(table, model, policy, Predicting::Whole))
        .collect::<Result<Vec<_>>>()?;
    Ok(merge_decisions(per_model))
}

/// Checks that there is at most one model per task and variant type.
///
/// # Errors
///
/// Returns an error naming the first task and variant type with more than one model (see
/// [`duplicate_kinds`]).
pub fn check_model_kinds<'a, I: IntoIterator<Item = &'a ModelInfo>>(models: I) -> Result<()> {
    match duplicate_kinds(models).first() {
        Some(&(task, variant_type)) => bail!("{}", model_kinds_message(task, variant_type)),
        None => Ok(()),
    }
}

/// Each task and variant type with more than one model, once, in the order they first appear.
#[must_use]
pub fn duplicate_kinds<'a, I: IntoIterator<Item = &'a ModelInfo>>(
    models: I,
) -> Vec<(Task, VariantType)> {
    let mut counts: indexmap::IndexMap<(Task, VariantType), usize> = indexmap::IndexMap::new();
    for info in models {
        *counts.entry((info.task, info.variant_type)).or_default() += 1;
    }
    counts.into_iter().filter(|(_, count)| *count > 1).map(|(kind, _)| kind).collect()
}

/// The message of a task and variant type with more than one model.
#[must_use]
pub fn model_kinds_message(task: Task, variant_type: VariantType) -> String {
    format!(
        "expected at most one model per task and variant type, got two {task} {variant_type} \
         models"
    )
}

/// Merges the decisions of each model, in model order: a call decided by more than one model
/// keeps the first model's decision.
#[must_use]
pub fn merge_decisions(per_model: Vec<Vec<(CallKey, Decision)>>) -> HashMap<CallKey, Decision> {
    let mut decisions = HashMap::with_capacity(per_model.iter().map(Vec::len).sum());
    for model_decisions in per_model {
        for (key, decision) in model_decisions {
            decisions.entry(key).or_insert(decision);
        }
    }
    decisions
}

/// One model's decisions for the rows of a table it scores (see [`scored_mask`]).
#[derive(Debug, Clone)]
pub struct ModelRowDecisions {
    /// The table rows the model scored, in table order.
    pub rows: Vec<usize>,
    /// The scored calls (with the re-genotype columns), one per entry of `rows`.
    pub calls: FeatureTable,
    /// The model's class probabilities for each scored call.
    pub probabilities: crate::rescore::Probabilities,
    /// Drop the call (genotype models).
    pub drop: Vec<bool>,
    /// Promote the call to PASS (promote models).
    pub promote: Vec<bool>,
    /// Flip the call het <-> hom (or flip the promoted call).
    pub flip: Vec<bool>,
    /// Re-genotype the call (re-genotype models; the target may still be unbuildable).
    pub regenotype: Vec<bool>,
}

/// Scores the rows of a table that a model scores and decides their actions, as `score` does,
/// `predicting` the probabilities as asked (see [`Predicting`]).
///
/// # Errors
///
/// Returns an error if the table lacks a column the model needs.
pub fn decide_model_rows(
    table: &FeatureTable,
    model: &WinnowrModel,
    policy: &Policy,
    predicting: Predicting,
    timer: &mut PhaseTimer,
) -> Result<ModelRowDecisions> {
    let mask = scored_mask(table, model)?;
    let rows: Vec<usize> = (0..mask.len()).filter(|&row| mask[row]).collect();
    let calls = table.filter(&mask)?;
    let calls = match &model.info.stutter {
        None => without_regenotype(calls)?,
        Some(stutter) => with_regenotype_columns(add_length_call_features(calls, stutter)?)?,
    };
    let what = format!("{} {}", model.info.task, model.info.variant_type);
    timer.lap(&format!("{what}: selecting the calls"));
    let probabilities = model.predict_with(&calls, predicting)?;
    timer.lap(&format!("{what}: predicting"));
    let n = calls.num_rows();
    let flippable = calls.bool_column("flippable")?;
    let vt = model.info.variant_type;
    let (drop, promote, flip, regenotype) = match model.info.task {
        Task::Promote => {
            let chosen = model.info.promote_thresholds.context("no promote thresholds")?;
            let mut promotions =
                decide_promote(&probabilities, &flippable, chosen.t_promote, chosen.t_flip);
            if let Some(withheld) = policy.withheld_promotions(&calls)? {
                withhold_promotions(&mut promotions, &withheld);
            }
            (vec![false; n], promotions.promote, promotions.flip, vec![false; n])
        }
        Task::Genotype => {
            let mode = policy.mode;
            let thresholds = model.info.thresholds.get(&mode).context("no thresholds for mode")?;
            let limits = policy.call_limits(&calls, vt)?;
            let (regenotypable, regenotypes_to_ref) =
                (calls.bool_column("regenotypable")?, calls.bool_column("regenotypes_to_ref")?);
            let flags = CallFlags {
                flippable: &flippable,
                regenotypable: &regenotypable,
                regenotypes_to_ref: &regenotypes_to_ref,
                limits: limits.as_deref(),
            };
            let actions = decide_with_rule(
                &probabilities,
                &flags,
                thresholds.decision(),
                mode,
                policy.rule(vt),
            );
            (actions.drop, vec![false; n], actions.flip, actions.regenotype)
        }
    };
    Ok(ModelRowDecisions { rows, calls, probabilities, drop, promote, flip, regenotype })
}

/// Withholds the promotions (and flips) of the candidates marked in `withheld`.
pub fn withhold_promotions(promotions: &mut Promotions, withheld: &[bool]) {
    for (row, _) in withheld.iter().enumerate().filter(|(_, out)| **out) {
        promotions.promote[row] = false;
        promotions.flip[row] = false;
    }
}

/// The [`decide_calls`] decisions of one model, in table order, `predicting` its probabilities
/// as asked (see [`Predicting`]).
///
/// # Errors
///
/// Returns an error if the table lacks a column the model needs.
pub fn decide_model_calls(
    table: &FeatureTable,
    model: &WinnowrModel,
    policy: &Policy,
    predicting: Predicting,
) -> Result<Vec<(CallKey, Decision)>> {
    let mut timer = PhaseTimer::start();
    let ModelRowDecisions { calls, probabilities, drop, promote, flip, regenotype, .. } =
        decide_model_rows(table, model, policy, predicting, &mut timer)?;
    let what = format!("{} {}", model.info.task, model.info.variant_type);
    let n = calls.num_rows();
    let targets: Vec<Option<RegenotypeTarget>> =
        if regenotype.contains(&true) { regenotype_targets(&calls)? } else { vec![None; n] };
    let keys = call_keys(&calls)?;
    // The first call with a key gets the decision; the rest are dropped. Keys are built and
    // decisions made in parallel; only the first-wins pass is serial.
    let first: Vec<bool> = {
        let mut seen: HashSet<&CallKey> = HashSet::with_capacity(n);
        keys.iter().map(|key| seen.insert(key)).collect()
    };
    let targets = &targets;
    let probabilities = &probabilities;
    let task = model.info.task;
    let decisions: Vec<(CallKey, Decision)> = keys
        .into_par_iter()
        .enumerate()
        .filter(|&(row, _)| first[row])
        .map(|(row, key)| {
            let decision = Decision {
                task,
                probabilities: probabilities.row(row).to_vec(),
                drop: drop[row],
                promote: promote[row],
                flip: flip[row],
                regenotype: if regenotype[row] { targets[row].clone() } else { None },
            };
            (key, decision)
        })
        .collect();
    // Counted from the decisions, so a call repeating an earlier call's key is not counted.
    let count = |action: fn(&Decision) -> bool| decisions.iter().filter(|(_, d)| action(d)).count();
    let (n_drop, n_promote) = (count(|d| d.drop), count(|d| d.promote));
    let (n_flip, n_regenotype) = (count(|d| d.flip), count(|d| d.regenotype.is_some()));
    let n_decided = decisions.len();
    timer.lap(&format!("{what}: deciding"));
    info!(
        "{} {}: scored {n_decided} calls, drop {n_drop}, promote {n_promote}, flip {n_flip}, \
         re-genotype {n_regenotype}",
        model.info.task, model.info.variant_type
    );
    Ok(decisions)
}

/// The key of every call in a table.
fn call_keys(table: &FeatureTable) -> Result<Vec<CallKey>> {
    let (chroms, positions) = (table.str_column("chrom")?, table.i64_column("pos")?);
    let (refs, alts, filters) =
        (table.str_column("ref")?, table.str_column("alt")?, table.str_column("filter")?);
    Ok((0..table.num_rows())
        .into_par_iter()
        .map(|i| {
            (
                chroms[i].to_string(),
                positions[i] as u64,
                refs[i].to_string(),
                alts[i].to_string(),
                filters[i].to_string(),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::Label;
    use crate::testing::{CallRow, SEPARABLE_P, call_table, separable_model};

    /// Calls with feature `x` (the class a [`separable_model`] gives them) and `flippable`.
    fn calls(rows: &[(CallRow<'_>, f64)]) -> FeatureTable {
        let keys: Vec<CallRow<'_>> = rows.iter().map(|(row, _)| *row).collect();
        let flippable = rows.iter().map(|(row, _)| row.6 == "0/1" || row.6 == "1/1").collect();
        call_table(&keys)
            .with_f64_column("x", rows.iter().map(|(_, x)| *x).collect())
            .unwrap()
            .with_bool_column("flippable", flippable)
            .unwrap()
    }

    fn key(pos: u64, reference: &str, alt: &str, filter: &str) -> CallKey {
        ("chr1".to_string(), pos, reference.into(), alt.into(), filter.into())
    }

    /// A deletion, SNPs and a duplicate key, scored by a SNP genotype and promote model.
    fn table() -> FeatureTable {
        calls(&[
            (("chr1", 100, "A", "C", "PASS", "snp", "0/1", false), 0.0), // drop
            (("chr1", 200, "A", "C", "PASS", "snp", "0/1", false), 1.0), // keep
            (("chr1", 300, "A", "C", "PASS", "snp", "0/1", false), 2.0), // flip
            // The same key again: the first call's decision stands.
            (("chr1", 300, "A", "C", "PASS", "snp", "0/1", false), 0.0),
            (("chr1", 400, "A", "C", "LowQual", "snp", "0/1", false), 1.0), // promote
            (("chr1", 450, "A", "C", "LowQual", "snp", "0/1", false), 2.0), // promote, flip
            // Overlapping a PASS call: promote models leave it alone.
            (("chr1", 500, "A", "C", "LowQual", "snp", "0/1", true), 1.0),
            // No indel model.
            (("chr1", 600, "AT", "A", "PASS", "indel", "0/1", false), 0.0),
        ])
    }

    fn models() -> Vec<WinnowrModel> {
        vec![
            separable_model(Task::Genotype, VariantType::Snp),
            separable_model(Task::Promote, VariantType::Snp),
        ]
    }

    /// The actions of a decision: drop, promote, flip.
    fn actions(decision: &Decision) -> (bool, bool, bool) {
        (decision.drop, decision.promote, decision.flip)
    }

    #[test]
    fn models_score_their_calls() {
        let (table, models) = (table(), models());
        let genotype = scored_mask(&table, &models[0]).unwrap();
        assert_eq!(genotype, [true, true, true, true, false, false, false, false]);
        let promote = scored_mask(&table, &models[1]).unwrap();
        assert_eq!(promote, [false, false, false, false, true, true, false, false]);
    }

    #[test]
    fn calls_are_decided_once_per_key() {
        let decisions = decide_calls(&table(), &models(), &Policy::new(Mode::Genotyper)).unwrap();
        let mut got: Vec<(u64, (bool, bool, bool))> =
            decisions.iter().map(|(key, d)| (key.1, actions(d))).collect();
        got.sort_unstable();
        let expected = [
            (100, (true, false, false)),
            (200, (false, false, false)),
            (300, (false, false, true)),
            (400, (false, true, false)),
            (450, (false, true, true)),
        ];
        assert_eq!(got, expected);
        let flipped = &decisions[&key(300, "A", "C", "PASS")];
        assert_eq!(flipped.task, Task::Genotype);
        assert!(flipped.probabilities[Label::Flip.index()] >= SEPARABLE_P);
        let promoted = &decisions[&key(400, "A", "C", "LowQual")];
        assert_eq!(promoted.task, Task::Promote);
        // One decision per key from each model, the first call's.
        let genotype = decide_model_calls(
            &table(),
            &models()[0],
            &Policy::new(Mode::Genotyper),
            Predicting::Whole,
        )
        .unwrap();
        let got: Vec<(u64, (bool, bool, bool))> =
            genotype.iter().map(|(key, d)| (key.1, actions(d))).collect();
        assert_eq!(got, expected[..3]);
        // Veto mode only drops.
        let veto = decide_calls(&table(), &models(), &Policy::new(Mode::Veto)).unwrap();
        assert_eq!(actions(&veto[&key(300, "A", "C", "PASS")]), (true, false, false));
    }

    #[test]
    fn two_models_of_a_kind_are_an_error() {
        let models = [
            separable_model(Task::Genotype, VariantType::Snp),
            separable_model(Task::Genotype, VariantType::Snp),
        ];
        let message = check_model_kinds(models.iter().map(|m| &m.info)).unwrap_err().to_string();
        assert_eq!(
            message,
            "expected at most one model per task and variant type, got two genotype snp models"
        );
        let message =
            decide_calls(&table(), &models, &Policy::new(Mode::Genotyper)).unwrap_err().to_string();
        assert!(message.starts_with("expected at most one model"), "{message}");
    }

    #[test]
    fn the_first_models_decision_wins() {
        let decision = |drop: bool| Decision {
            task: Task::Genotype,
            probabilities: vec![0.0; 3],
            drop,
            promote: false,
            flip: false,
            regenotype: None,
        };
        let (a, b) = (key(1, "A", "C", "PASS"), key(2, "A", "C", "PASS"));
        let merged = merge_decisions(vec![
            vec![(a.clone(), decision(true))],
            vec![(a.clone(), decision(false)), (b.clone(), decision(false))],
        ]);
        assert_eq!(merged.len(), 2);
        assert!(merged[&a].drop);
        assert!(!merged[&b].drop);
    }

    /// The not-difficult regions chr1:[100, 200) (0-based), withholding `actions`.
    fn abstention(flip: bool, promote: bool, indel_t_drop: Option<f64>) -> Abstention {
        let regions = RegionSet::from_spans(NOT_DIFFICULT_REGIONS, [("chr1", 100, 200)]);
        Abstention { regions, flip, regenotype: false, promote, indel_t_drop }
    }

    #[test]
    fn a_call_is_outside_unless_its_whole_ref_span_is_inside() {
        let table = call_table(&[
            ("chr1", 101, "A", "C", "PASS", "snp", "0/1", false), // [100, 101)
            ("chr1", 200, "A", "C", "PASS", "snp", "0/1", false), // [199, 200)
            ("chr1", 199, "ATT", "A", "PASS", "indel", "0/1", false), // [198, 201)
            ("chr1", 100, "A", "C", "PASS", "snp", "0/1", false), // [99, 100)
            ("chr2", 150, "A", "C", "PASS", "snp", "0/1", false),
        ]);
        let abstain = abstention(true, false, Some(0.7));
        assert_eq!(abstain.outside(&table).unwrap(), [false, false, true, true, true]);
        assert_eq!(abstain.limits(VariantType::Snp, false), CallLimits::default());
        let snp = CallLimits { no_flip: true, no_regenotype: false, t_drop: None };
        assert_eq!(abstain.limits(VariantType::Snp, true), snp);
        let indel = CallLimits { t_drop: Some(0.7), ..snp };
        assert_eq!(abstain.limits(VariantType::Indel, true), indel);
        let policy = Policy { abstain: Some(abstain), ..Policy::new(Mode::Genotyper) };
        let limits = policy.call_limits(&table, VariantType::Indel).unwrap().unwrap();
        assert_eq!(limits[0], CallLimits::default());
        assert_eq!(limits[2], indel);
        // Promotions are withheld only when asked.
        assert_eq!(policy.withheld_promotions(&table).unwrap(), None);
        assert_eq!(
            Policy::new(Mode::Genotyper).call_limits(&table, VariantType::Snp).unwrap(),
            None
        );
    }

    #[test]
    fn the_not_difficult_regions_key_is_the_giab_stratification_name() {
        // Users and trained models pass `--regions notinalldifficultregions=PATH`.
        assert_eq!(NOT_DIFFICULT_REGIONS, "notinalldifficultregions");
    }

    #[test]
    fn actions_outside_the_not_difficult_regions_are_withheld() {
        // Of the scored calls, only chr1:300 and chr1:450 are outside these.
        let regions =
            RegionSet::from_spans(NOT_DIFFICULT_REGIONS, [("chr1", 0, 299), ("chr1", 300, 400)]);
        let abstain = Abstention {
            regions,
            flip: true,
            regenotype: false,
            promote: true,
            indel_t_drop: None,
        };
        let policy = Policy { abstain: Some(abstain), ..Policy::new(Mode::Genotyper) };
        let decisions = decide_calls(&table(), &models(), &policy).unwrap();
        // The flip outside is withheld (and its P(drop) is too low to drop it); the promotion
        // and promote-flip outside are both withheld; the promotion inside stands.
        assert_eq!(actions(&decisions[&key(300, "A", "C", "PASS")]), (false, false, false));
        assert_eq!(actions(&decisions[&key(450, "A", "C", "LowQual")]), (false, false, false));
        assert_eq!(actions(&decisions[&key(400, "A", "C", "LowQual")]), (false, true, false));
        assert_eq!(actions(&decisions[&key(100, "A", "C", "PASS")]), (true, false, false));
        let mut promotions = Promotions { promote: vec![true, true], flip: vec![true, true] };
        withhold_promotions(&mut promotions, &[false, true]);
        assert_eq!((promotions.promote, promotions.flip), (vec![true, false], vec![true, false]));
    }
}
