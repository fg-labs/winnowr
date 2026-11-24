//! winnowr: machine-learned genotyping and filtering of germline small-variant calls.
//!
//! winnowr extracts features for a sample's calls, trains XGBoost models on samples with truth
//! labels, and scores new samples: dropping calls that are likely `0/0`, promoting filtered calls,
//! flipping het/hom genotypes, and re-genotyping repeat-length calls. The `winnowr` binary
//! provides the command-line interface.
//!
//! The library exists to serve the binary: only its command-line interface ([`cli`]) is public,
//! and it is not a stable API either.

#![forbid(unsafe_code)]
// Items of the private modules are bare `pub` (visible in the crate only, as the modules are
// private); `pub(crate)` is kept for items of the public `cli` module that are not public API.
#![warn(clippy::redundant_pub_crate)]

// Depended on only to enable their `static` features (see Cargo.toml).
use bzip2_sys as _;
use lzma_sys as _;

mod alleles;
mod check_models;
pub mod cli;
mod common;
mod evaluation;
mod evidence;
mod extraction;
mod fasta;
mod feature_source;
mod features;
mod fitting;
mod gvcf;
mod io;
mod labels;
mod model;
mod model_evidence;
mod model_set;
mod neighbourhood;
mod numeric;
mod pileup;
mod pileup_features;
mod policy;
mod promote_labels;
mod raw_records;
mod read_annotations;
mod read_pass;
mod read_settings;
mod regenotype;
mod regions;
mod repeat_features;
mod rescore;
mod samples;
mod scoring;
mod shards;
mod strata;
mod strlen;
mod stutter;
mod subset;
mod table;
#[cfg(test)]
mod testing;
mod thresholds;
mod training;
mod vcf;
mod vcf_source;
mod version;
