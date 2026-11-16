//! Basic types and constants shared across winnowr modules.
//!
//! Enums whose command-line values are exactly their variants ([`Task`], [`VariantType`],
//! [`crate::labels::UnfixableFlips`]) derive clap's `ValueEnum`, with each variant's help left out
//! of `--help` (the options describe them); `cli.rs` has its own enums for options whose values
//! differ (e.g. add `none`).

use std::fmt;
use std::str::FromStr;

use anyhow::bail;
use serde::{Deserialize, Serialize};

/// The contigs extracted by default: the GRCh38 autosomes.
pub const AUTOSOMES: [&str; 22] = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
    "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21",
    "chr22",
];

/// A threshold no probability reaches, used to disable an action.
pub const NEVER: f64 = 1.01;

/// The FILTER value of a passing call. A missing FILTER is read as PASS.
pub const PASS: &str = "PASS";

/// True if a path's file name ends with `suffix`, ignoring ASCII case (e.g. `.gz` matches
/// `x.VCF.GZ`), as output writers choose their format.
#[must_use]
pub fn has_suffix(path: &std::path::Path, suffix: &str) -> bool {
    let name = path.to_string_lossy();
    name.len() >= suffix.len()
        && name.is_char_boundary(name.len() - suffix.len())
        && name[name.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
}

/// True if an output path names a BGZF-compressed file: `.gz` or `.bgz`, in any case.
#[must_use]
pub fn is_bgzf_path(path: &std::path::Path) -> bool {
    has_suffix(path, ".gz") || has_suffix(path, ".bgz")
}

/// The items that occur more than once, in sorted order, each once (e.g. to report repeated
/// names).
#[must_use]
pub fn repeated<T: Ord>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut items: Vec<T> = items.into_iter().collect();
    items.sort_unstable();
    let mut repeated: Vec<T> = Vec::new();
    let mut previous: Option<T> = None;
    for item in items {
        if previous.as_ref() == Some(&item) && repeated.last() != Some(&item) {
            repeated.push(item);
        } else {
            previous = Some(item);
        }
    }
    repeated
}

/// What a model decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// For PASS calls: drop (make `0/0`), keep, or flip het <-> hom.
    #[value(help = None::<&str>)]
    Genotype,
    /// For non-PASS calls: leave filtered, promote to PASS, or promote and flip.
    #[value(help = None::<&str>)]
    Promote,
}

/// The type of variant a model is trained for.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum VariantType {
    /// A single-nucleotide polymorphism.
    #[value(help = None::<&str>)]
    Snp,
    /// An insertion or deletion.
    #[value(help = None::<&str>)]
    Indel,
}

/// Implements `as_str`, `Display` and `FromStr` for a field-less enum from its lowercase names.
macro_rules! named_enum {
    ($type:ident, $what:literal, { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl $type {
            /// Every variant, in declaration order.
            pub const ALL: &'static [$type] = &[$($type::$variant),+];

            /// The lowercase name, as written in model files and on the command line.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($type::$variant => $name),+
                }
            }
        }

        impl fmt::Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $type {
            type Err = anyhow::Error;

            fn from_str(s: &str) -> anyhow::Result<Self> {
                match s {
                    $($name => Ok($type::$variant),)+
                    _ => {
                        let names: Vec<&str> = Self::ALL.iter().map(|v| v.as_str()).collect();
                        bail!("unknown {} '{s}' (expected one of: {})", $what, names.join(", "))
                    }
                }
            }
        }
    };
}

named_enum!(Task, "task", { Genotype => "genotype", Promote => "promote" });
named_enum!(VariantType, "variant type", { Snp => "snp", Indel => "indel" });

/// Times the phases of a command, logging how long each took at debug level (enable with
/// `RUST_LOG=winnowr=debug`), to find where time goes without a profiler.
#[derive(Debug)]
pub struct PhaseTimer(std::time::Instant);

impl PhaseTimer {
    /// Starts timing the first phase.
    #[must_use]
    pub fn start() -> PhaseTimer {
        PhaseTimer(std::time::Instant::now())
    }

    /// Logs the time since the timer started or the previous phase ended, and starts the next.
    pub fn lap(&mut self, phase: &str) {
        log::debug!("{phase} took {:.2?}", self.0.elapsed());
        self.0 = std::time::Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn repeated_items_are_sorted_and_listed_once() {
        assert_eq!(repeated(["b", "a", "b", "c", "a", "b"]), ["a", "b"]);
        assert!(repeated(["a", "b"]).is_empty());
        assert!(repeated(Vec::<u8>::new()).is_empty());
    }

    #[test]
    fn autosomes_are_chr1_to_chr22_in_order() {
        let expected: Vec<String> = (1..=22).map(|i| format!("chr{i}")).collect();
        assert_eq!(AUTOSOMES.to_vec(), expected);
    }

    #[rstest]
    #[case(Task::Genotype, "genotype")]
    #[case(Task::Promote, "promote")]
    fn task_names_round_trip(#[case] task: Task, #[case] name: &str) {
        assert_eq!(task.to_string(), name);
        assert_eq!(name.parse::<Task>().unwrap(), task);
    }

    #[rstest]
    #[case(VariantType::Snp, "snp")]
    #[case(VariantType::Indel, "indel")]
    fn variant_type_names_round_trip(#[case] variant_type: VariantType, #[case] name: &str) {
        assert_eq!(variant_type.to_string(), name);
        assert_eq!(name.parse::<VariantType>().unwrap(), variant_type);
    }

    #[rstest]
    #[case::wrong_case("SNP")]
    #[case::empty("")]
    #[case::unknown("mnp")]
    fn unknown_variant_types_are_rejected(#[case] name: &str) {
        let message = name.parse::<VariantType>().unwrap_err().to_string();
        assert!(message.contains("expected one of: snp, indel"), "{message}");
    }

    /// The command line, model files and messages name the variants the same way: clap's
    /// [`clap::ValueEnum`] names, serde's and [`Task::as_str`]'s.
    #[test]
    fn command_line_names_are_the_model_file_names() {
        use clap::ValueEnum;
        assert_eq!(Task::value_variants(), Task::ALL);
        for task in Task::ALL {
            assert_eq!(task.to_possible_value().unwrap().get_name(), task.as_str());
            assert_eq!(serde_json::to_string(task).unwrap(), format!("\"{task}\""));
        }
        assert_eq!(VariantType::value_variants(), VariantType::ALL);
        for variant_type in VariantType::ALL {
            assert_eq!(variant_type.to_possible_value().unwrap().get_name(), variant_type.as_str());
            assert_eq!(serde_json::to_string(variant_type).unwrap(), format!("\"{variant_type}\""));
        }
    }

    #[test]
    fn unknown_tasks_are_rejected() {
        let message = "drop".parse::<Task>().unwrap_err().to_string();
        assert_eq!(message, "unknown task 'drop' (expected one of: genotype, promote)");
    }
}
