# Changelog

All notable changes to this project are documented in this file.

## [0.1.1] - 2026-11-16

The first release published to crates.io (`winnowr` and `winnowr-xgboost`).

### Features

- `extract --happy-pass-only <auto|yes|no>` (with `--happy`) says whether hap.py was run with
  `--pass-only`. With `auto`, the default, it is inferred as run without `--pass-only` only when
  at least half of the non-PASS calls hap.py could have scored have a record no PASS call
  explains, rather than from a single label. The mode and its evidence are logged, with a warning
  when the labels contradict it.
- A workflow attaches the full source tarball, with the XGBoost sources, and its SHA-256 to each
  GitHub release.
- A user guide, at <https://winnowr.readthedocs.io>.

### Bug Fixes

- A FILTER the VCF header does not declare, or a record htslib flags (e.g. on an undeclared
  contig), is an error naming the record in `extract` and `score`, rather than a crash. In
  `score`, so is an undeclared INFO or FORMAT tag, or one declared only as another kind, rather
  than an exit from htslib or a field written under the wrong name.
- `score` keeps records at POS 0, which its sharded writer left out. `extract` warns that it does
  not read them.
- `score` rejects an `--output` that is a hard link to the input VCF.
- `score` re-genotypes only diploid calls. A corrupt tract length or length change gives a call
  no re-genotype target rather than an unbounded allocation, and a read past the end of the
  reference FASTA is an error.
- A `--regions` span that starts past its contig's end is an error rather than an empty table,
  and `strlen` and `pileup` reject flanks over 10000 bp.
- In genotyper mode with `--expected-f1`, a call's own drop threshold (e.g.
  `--outside-indel-t-drop`) applies before a flip or re-genotype.
- `train` rejects non-finite or out-of-range `--learning-rate`, `--min-child-weight`,
  `--reg-lambda`, `--max-leaves` and `--max-depth` before reading the tables, and warns and skips
  the refit when the validation fraction holds out no block.
- Lowercase inserted bases are matched to the upper-case reference context.
- hap.py labels split from PASS multi-allelic calls, and those at padded SNPs and indels, are
  attached to the right call.
- On Linux, XGBoost's OpenMP wait policy measures other work only on the CPUs winnowr may use (its
  CPU affinity and cgroup quota).
- XGBoost no longer prints an "Empty dataset" warning when winnowr counts a model's classes.

### Performance

- With LLVM's libomp and a positive blocktime, XGBoost's OpenMP teams keep a fixed size (a second
  local XGBoost patch), which avoids resizing them between parallel regions. Models are unchanged.
- `strlen` and `pileup` share contig sequences across workers, and feature extraction allocates
  less per row.

### Refactor

- The hidden `strlen --also` option is removed.
- More enum-like options list their possible values in `--help`.
- `winnowr-xgboost`'s public API is trimmed before its first publish.

## [0.1.0] - 2026-11-15

The first release of winnowr: machine-learned filtering and genotyping of germline small-variant
calls, with XGBoost models trained on hap.py-labelled feature tables.

### Features

- `extract`, `strlen`, `pileup`, `train`, `score`, `evaluate`, `stratify` and `subset`.
- Models are one `<name>.winnowr.json` file (model format 3) with the XGBoost booster embedded.
- Training holds out whole 10 kb genomic blocks for early stopping, then refits the final model on
  every call (`--refit-final`, on by default).
- In `score`, `evaluate` and `stratify`, indel genotype models also drop the calls their thresholds
  leave alone when P(keep) is below 0.5 (`--keep-fallback indel`). Given GIAB's
  `notinalldifficultregions` stratification (`--regions notinalldifficultregions=PATH`), they drop
  indel calls outside it when P(drop) is at least 0.6, in place of the model's drop threshold
  (`--outside-indel-t-drop`).
- Repeat-length features (`winnowr strlen`) measure reads with `--reads mapq20,flank5` by default;
  `--reads legacy` measures reads of any MAPQ with two flank bases.
- Pileup features (`winnowr pileup`, `train --pileup-features`) count reference, alternate and
  other reads at MAPQ 0, 1-19 and 20 or more, with the settings `mapq20,bq10,flank5`. The settings
  are recorded in the pileup table and in each model trained with them, and `score` refuses
  evidence measured with other settings.
- XGBoost 3.5.0-dev (upstream `293cad1a8`) is vendored and linked statically, with one local
  patch that keeps OpenMP teams a fixed size under libgomp.
