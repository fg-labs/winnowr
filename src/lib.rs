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

// The command-line interface names the other modules as `winnowr::...`, as the binary did.
extern crate self as winnowr;

// Depended on only to enable their `static` features (see Cargo.toml).
use bzip2_sys as _;
use lzma_sys as _;

mod alleles;
pub mod cli;
mod common;
mod evaluation;
mod evidence;
mod extraction;
mod fasta;
mod features;
mod gvcf;
mod io;
mod labels;
mod model;
mod model_evidence;
mod neighbourhood;
mod numeric;
mod pileup;
mod pileup_features;
mod policy;
mod promote_labels;
mod raw_records;
mod read_pass;
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
mod version;
