//! The re-genotype action: which calls it can change, whether the change is right, and the new
//! record.
//!
//! A re-genotype replaces a call's genotype with the repeat-length caller's (see
//! [`crate::stutter`]). [`with_regenotype_columns`] flags the calls it applies to and, for
//! labelled calls, whether it gives the truth genotype; [`regenotyped_alleles`] builds the alleles
//! and genotype that `score` writes.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;

use crate::alleles::Genotype;
use crate::labels::{Label, TRUTH_DELTA_COLUMNS};
use crate::strlen::{CONTEXT, RepeatTract};
use crate::table::FeatureTable;

/// The greatest length change (bp) of a re-genotyped allele: a length call beyond it is taken to
/// come from a corrupt strlen table, and the call gets no new record.
const MAX_DELTA: f64 = 1_000_000.0;

/// The per-call re-genotype columns (not features).
pub const REGENOTYPE_COLUMNS: [&str; 3] = ["regenotypable", "regenotypes_to_ref", "regenotype_ok"];

/// The new alleles of a re-genotyped record, and its genotype's sorted allele indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegenotypeTarget {
    /// The record's REF and ALT alleles, extended when a new allele was added.
    pub alleles: Vec<String>,
    /// The new genotype's sorted allele indices.
    pub genotype: (usize, usize),
}

/// True for a spanning deletion (`*`) or a symbolic allele (e.g. `<NON_REF>`).
fn is_symbolic(allele: &str) -> bool {
    allele == "*" || allele.starts_with('<')
}

/// The alleles and genotype of a record re-genotyped to the given allele length changes.
///
/// A length change of 0 is the REF (a same-length ALT, e.g. `T` in `A>T,AT`, cannot be told
/// from it by length). Any other is the repeat tract resized by that change in its
/// phase: an ALT with exactly that sequence (ignoring case) is reused, otherwise it is added as a
/// new ALT. That needs a tract starting after the record's first base and no later than the end
/// of its REF; when a new ALT is added and the tract extends past the REF, the REF (and every
/// other non-symbolic allele) is extended to the end of the tract. Without such a tract, a length
/// change can only use the one ALT of that length, if there is exactly one.
///
/// # Arguments
///
/// * `pos` - the record's 1-based position.
/// * `alleles` - the record's REF and ALT alleles.
/// * `deltas` - the new genotype's sorted allele length changes.
/// * `tract` - the record's repeat tract (from the strlen evidence), if known.
///
/// Returns `None` if an allele cannot be found or built, or `pos` is 0.
#[must_use]
pub fn regenotyped_alleles<S: AsRef<str>>(
    pos: u64,
    alleles: &[S],
    deltas: (i64, i64),
    tract: Option<&RepeatTract>,
) -> Option<RegenotypeTarget> {
    let alleles: Vec<String> = alleles.iter().map(|a| a.as_ref().to_string()).collect();
    let reference = alleles.first()?;
    // POS 0 (a telomere record) has no 0-based start.
    let start = pos.checked_sub(1)?;
    let end = start + reference.len() as u64;
    let changes: BTreeSet<i64> = [deltas.0, deltas.1].into_iter().filter(|&d| d != 0).collect();
    let mut index_of: BTreeMap<i64, usize> = BTreeMap::from([(0, 0)]);
    let genotype = |index_of: &BTreeMap<i64, usize>| {
        let (a, b) = (index_of[&deltas.0], index_of[&deltas.1]);
        (a.min(b), a.max(b))
    };

    let Some(tract) = tract.filter(|tract| start < tract.start && tract.start <= end) else {
        for &delta in &changes {
            let same_length: Vec<usize> = alleles
                .iter()
                .enumerate()
                .skip(1)
                .filter(|(_, allele)| {
                    !is_symbolic(allele) && allele.len() as i64 - reference.len() as i64 == delta
                })
                .map(|(index, _)| index)
                .collect();
            let [index] = same_length[..] else {
                return None;
            };
            index_of.insert(delta, index);
        }
        let genotype = genotype(&index_of);
        return Some(RegenotypeTarget { alleles, genotype });
    };

    let tract_len = tract.length() as i64;
    if [deltas.0, deltas.1].iter().any(|&delta| tract_len + delta < 0) {
        return None;
    }
    let suffix = if tract.end > end {
        tract.sequence(tract.length())[(end - tract.start) as usize..].to_string()
    } else {
        String::new()
    };
    let extended: Vec<String> = alleles
        .iter()
        .map(|a| if is_symbolic(a) { a.clone() } else { format!("{a}{suffix}") })
        .collect();
    let head = &extended[0][..(tract.start - start) as usize];
    let tail = &extended[0][(tract.end - start) as usize..];
    let mut new_alleles = extended.clone();
    for &delta in &changes {
        let resized = format!("{head}{}{tail}", tract.sequence((tract_len + delta) as u64));
        let existing = new_alleles
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, allele)| allele.eq_ignore_ascii_case(&resized))
            .map(|(index, _)| index);
        let index = existing.unwrap_or_else(|| {
            new_alleles.push(resized);
            new_alleles.len() - 1
        });
        index_of.insert(delta, index);
    }
    let genotype = genotype(&index_of);
    let added = new_alleles.len() > alleles.len();
    Some(RegenotypeTarget { alleles: if added { new_alleles } else { alleles }, genotype })
}

/// The new alleles and genotype of a re-genotyped call, from its feature-table values, or `None`
/// if a value is missing or the new allele cannot be built.
///
/// # Arguments
///
/// * `pos` - the call's 1-based position.
/// * `reference` - the reference allele.
/// * `alt` - the comma-joined alternate alleles.
/// * `deltas` - the length caller's genotype (`lc_delta_lo`, `lc_delta_hi`).
/// * `tract_start` - the repeat tract's 1-based start (`sl_tract_start`).
/// * `tract_len` - the tract length (`sl_tract_len`).
/// * `unit` - the tract's repeat unit (`sl_unit`; empty if missing).
#[must_use]
pub fn regenotype_target(
    pos: u64,
    reference: &str,
    alt: &str,
    deltas: (f64, f64),
    tract_start: f64,
    tract_len: f64,
    unit: &str,
) -> Option<RegenotypeTarget> {
    // `strlen` finds tracts in a context of CONTEXT bases on each side of REF, so a longer tract
    // (or a length change of an allele beyond MAX_DELTA) can only come from a corrupt table, and
    // building its sequence could exhaust memory.
    let max_tract_len = (2 * CONTEXT + reference.len() as u64) as f64;
    if [deltas.0, deltas.1, tract_start, tract_len].iter().any(|v| v.is_nan())
        || unit.is_empty()
        || tract_start < 1.0
        || !(0.0..=max_tract_len).contains(&tract_len)
        || [deltas.0, deltas.1].iter().any(|d| d.abs() > MAX_DELTA)
    {
        return None;
    }
    let start = tract_start as u64 - 1;
    let tract = RepeatTract {
        start,
        end: start.checked_add(tract_len as u64)?,
        period: unit.len() as u64,
        unit: unit.to_string(),
    };
    let alleles: Vec<&str> = std::iter::once(reference).chain(alt.split(',')).collect();
    regenotyped_alleles(pos, &alleles, (deltas.0 as i64, deltas.1 as i64), Some(&tract))
}

/// The new alleles and genotype of every call with a length call whose new record can be built.
///
/// # Errors
///
/// Returns an error if the table lacks the length-call or strlen tract columns.
pub fn regenotype_targets(table: &FeatureTable) -> Result<Vec<Option<RegenotypeTarget>>> {
    let positions = table.i64_column("pos")?;
    let (refs, alts) = (table.str_column("ref")?, table.str_column("alt")?);
    let (lo, hi) = (table.f64_column("lc_delta_lo")?, table.f64_column("lc_delta_hi")?);
    let (starts, lengths) =
        (table.f64_column("sl_tract_start")?, table.f64_column("sl_tract_len")?);
    let units = table.str_column("sl_unit")?;
    Ok((0..table.num_rows())
        .map(|i| {
            if lo[i].is_nan() {
                return None;
            }
            let pos = u64::try_from(positions[i]).ok()?;
            regenotype_target(
                pos,
                refs[i],
                alts[i],
                (lo[i], hi[i]),
                starts[i],
                lengths[i],
                units[i],
            )
        })
        .collect())
}

/// Adds the re-genotype columns, all false, to calls of a model without the re-genotype action.
///
/// # Errors
///
/// Returns an error if a column cannot be added.
pub fn without_regenotype(table: FeatureTable) -> Result<FeatureTable> {
    let n = table.num_rows();
    let mut table = table;
    for name in REGENOTYPE_COLUMNS {
        table = table.with_bool_column(name, vec![false; n])?;
    }
    Ok(table)
}

/// Adds the re-genotype columns to calls with length-call features.
///
/// `regenotypable` marks the calls the length caller disagrees with (it made a call that differs
/// from the called genotype) whose new genotype `score` can write (see [`regenotype_targets`]), so
/// training, evaluation and scoring act on the same calls. A call without a diploid genotype
/// (`./.`, a haploid or other non-diploid GT) is never regenotypable: it has no hap.py label to
/// train the action on, and re-genotyping would write a diploid genotype over it. A genotype with
/// a `*` or symbolic allele is diploid and stays regenotypable. `regenotypes_to_ref` marks the calls
/// whose length-caller genotype is `0/0`. `regenotype_ok` (labelled calls only; otherwise false)
/// marks FPs ([`Label::Flip`], or [`Label::Drop`], e.g. a wrong repeat length that hap.py reports
/// as a local match) whose new genotype is not `0/0` and has the truth alleles' length changes
/// (in a pure repeat tract, alleles of the same length have the same sequence).
///
/// # Errors
///
/// Returns an error if the table lacks the length-call or strlen columns.
pub fn with_regenotype_columns(table: FeatureTable) -> Result<FeatureTable> {
    let (lo, hi) = (table.f64_column("lc_delta_lo")?, table.f64_column("lc_delta_hi")?);
    let is_called = table.f64_column("lc_is_called")?;
    let diploid: Vec<bool> = table
        .str_column("gt")?
        .iter()
        .map(|gt| gt.parse::<Genotype>().is_ok_and(|gt| !gt.is_missing()))
        .collect();
    let buildable: Vec<bool> = regenotype_targets(&table)?.iter().map(Option::is_some).collect();
    let n = table.num_rows();
    let changes: Vec<bool> = (0..n)
        .map(|i| !lo[i].is_nan() && is_called[i] == 0.0 && diploid[i] && buildable[i])
        .collect();
    let to_ref: Vec<bool> = (0..n).map(|i| lo[i] == 0.0 && hi[i] == 0.0).collect();
    let labelled =
        table.has_column("label") && TRUTH_DELTA_COLUMNS.iter().all(|c| table.has_column(c));
    let ok: Vec<bool> = if labelled {
        let labels = table.f64_column("label")?;
        let (truth_lo, truth_hi) =
            (table.f64_column(TRUTH_DELTA_COLUMNS[0])?, table.f64_column(TRUTH_DELTA_COLUMNS[1])?);
        let fp =
            |label: f64| label == Label::Flip.index() as f64 || label == Label::Drop.index() as f64;
        (0..n)
            .map(|i| {
                lo[i] == truth_lo[i]
                    && hi[i] == truth_hi[i]
                    && fp(labels[i])
                    && changes[i]
                    && !to_ref[i]
            })
            .collect()
    } else {
        vec![false; n]
    };
    table
        .with_bool_column(REGENOTYPE_COLUMNS[0], changes)?
        .with_bool_column(REGENOTYPE_COLUMNS[1], to_ref)?
        .with_bool_column(REGENOTYPE_COLUMNS[2], ok)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn tract(start: u64, end: u64, unit: &str) -> RepeatTract {
        RepeatTract { start, end, period: unit.len() as u64, unit: unit.to_string() }
    }

    fn target(alleles: &[&str], genotype: (usize, usize)) -> Option<RegenotypeTarget> {
        Some(RegenotypeTarget {
            alleles: alleles.iter().map(ToString::to_string).collect(),
            genotype,
        })
    }

    // Reference around the record: G at 100 (0-based 99), then the tract AAAA at 100..104, then C.
    #[rstest]
    #[case::reuses_an_alt(&["GA", "G"], (-1, 0), target(&["GA", "G"], (0, 1)))]
    #[case::hom_ref(&["GA", "G"], (0, 0), target(&["GA", "G"], (0, 0)))]
    #[case::adds_a_shorter_allele_extending_the_ref(&["GA", "G"], (-2, -1), target(&["GAAAA", "GAAA", "GAA"], (1, 2)))]
    #[case::adds_a_longer_allele(&["GAAAA", "G"], (0, 1), target(&["GAAAA", "G", "GAAAAA"], (0, 2)))]
    #[case::too_short_for_the_tract(&["GA", "G"], (-5, 0), None)]
    fn alleles_from_the_tract(
        #[case] alleles: &[&str],
        #[case] deltas: (i64, i64),
        #[case] expected: Option<RegenotypeTarget>,
    ) {
        let tract = tract(100, 104, "A");
        assert_eq!(regenotyped_alleles(100, alleles, deltas, Some(&tract)), expected);
    }

    #[test]
    fn existing_alleles_match_ignoring_case() {
        let tract = tract(100, 104, "A");
        let got = regenotyped_alleles(100, &["GAAAA", "gaaa"], (-1, -1), Some(&tract));
        assert_eq!(got, target(&["GAAAA", "gaaa"], (1, 1)));
    }

    #[rstest]
    #[case::one_alt_of_the_length(&["GA", "G", "GAT"], (-1, 0), target(&["GA", "G", "GAT"], (0, 1)))]
    #[case::two_alts_of_the_length(&["GA", "G", "C"], (-1, 0), None)]
    #[case::no_alt_of_the_length(&["GA", "G"], (1, 1), None)]
    #[case::symbolic_ignored(&["G", "<DEL>", "GT"], (1, 1), target(&["G", "<DEL>", "GT"], (2, 2)))]
    fn alleles_without_a_usable_tract(
        #[case] alleles: &[&str],
        #[case] deltas: (i64, i64),
        #[case] expected: Option<RegenotypeTarget>,
    ) {
        assert_eq!(regenotyped_alleles(100, alleles, deltas, None), expected);
        // A tract starting at the record's first base is not usable either.
        let at_first_base = tract(99, 104, "A");
        assert_eq!(regenotyped_alleles(100, alleles, deltas, Some(&at_first_base)), expected);
    }

    #[test]
    fn a_telomere_record_is_not_regenotyped() {
        // At POS 1 this has a target; POS 0 has no 0-based start.
        let alleles = ["GA", "G", "GAT"];
        assert!(regenotyped_alleles(1, &alleles, (-1, 0), None).is_some());
        assert_eq!(regenotyped_alleles(0, &alleles, (-1, 0), None), None);
        assert_eq!(regenotyped_alleles(0, &alleles, (-1, 0), Some(&tract(0, 4, "A"))), None);
    }

    #[test]
    fn only_calls_with_a_diploid_genotype_are_regenotypable() {
        // The length caller calls a 1 bp deletion (-1/0) of the tract at 101..105, which differs
        // from every called genotype below; `G A,*` has a `*` allele but a diploid GT.
        let rows = [
            ("chr1", 100, "GA", "G", "PASS", "INDEL", "1/1", false),
            ("chr1", 100, "GA", "G", "PASS", "INDEL", "./.", false),
            ("chr1", 100, "GA", "G,*", "PASS", "INDEL", "1/2", false),
        ];
        let n = rows.len();
        let table = crate::testing::call_table(&rows)
            .with_f64_column("lc_delta_lo", vec![-1.0; n])
            .and_then(|t| t.with_f64_column("lc_delta_hi", vec![0.0; n]))
            .and_then(|t| t.with_f64_column("lc_is_called", vec![0.0; n]))
            .and_then(|t| t.with_f64_column("sl_tract_start", vec![101.0; n]))
            .and_then(|t| t.with_f64_column("sl_tract_len", vec![4.0; n]))
            .and_then(|t| t.with_str_column("sl_unit", &["A"; 3]))
            .unwrap();
        let table = with_regenotype_columns(table).unwrap();
        assert_eq!(table.bool_column("regenotypable").unwrap(), [true, false, true]);
    }

    #[test]
    fn targets_from_table_values() {
        let got = regenotype_target(100, "GA", "G", (-2.0, -1.0), 101.0, 4.0, "A");
        assert_eq!(got, target(&["GAAAA", "GAAA", "GAA"], (1, 2)));
        assert!(regenotype_target(100, "GA", "G", (f64::NAN, 0.0), 101.0, 4.0, "A").is_none());
        assert!(regenotype_target(100, "GA", "G", (0.0, -1.0), 101.0, 4.0, "").is_none());
        // A tract longer than strlen can find, or an absurd length change, is a corrupt table:
        // no target, rather than an allocation of its sequence.
        let longest = (2 * CONTEXT + 2) as f64;
        assert!(regenotype_target(100, "GA", "G", (-1.0, 0.0), 101.0, longest, "A").is_some());
        assert!(
            regenotype_target(100, "GA", "G", (-1.0, 0.0), 101.0, longest + 1.0, "A").is_none()
        );
        assert!(regenotype_target(100, "GA", "G", (-1.0, 0.0), 101.0, 1e12, "A").is_none());
        assert!(regenotype_target(100, "GA", "G", (0.0, 1e12), 101.0, 4.0, "A").is_none());
        assert!(regenotype_target(100, "GA", "G", (-1.0, 0.0), 1e30, 4.0, "A").is_none());
    }
}
