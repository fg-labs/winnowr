# Changelog

All notable changes to winnowr are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and winnowr uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). The supported interface is the command
line and the files it reads and writes (see the README); the Python modules are internal.

## [Unreleased]

### Fixed

- winnowr failed at startup on Python 3.13 and 3.14 when installed without its development
  dependencies (`No module named 'typing_extensions'`): fgpyo imports `typing_extensions` but
  declares it only for Python 3.12 and earlier. winnowr now depends on it directly.

### Changed

- `winnowr extract` and `winnowr score` read the raw annotations (`RAW_MQandDP`, `QUALapprox`,
  `VarDP`, `AS_RAW_MQ`) from the VCF when its header declares all four, so `--gvcf` is now
  optional. A VCF without them still needs `--gvcf`, which works as before; when the VCF has them,
  `--gvcf` is not read (with a warning). Without either source, both commands fail with an error
  naming the missing annotations.

## [0.1.0] - Unreleased

First public release.

### Added

- `winnowr extract`: per-call features from a single-sample GATK HaplotypeCaller VCF and the gVCF
  it was genotyped from (VCF and gVCF annotations, user-named region BEDs, neighbourhood
  features), with optional hap.py labels (`--happy`), promote labels (`--truth-vcf`,
  `--truth-bed`, `--reference`) and repeat-length evidence (`--strlen`), written as a Parquet
  feature table.
- `winnowr strlen`: per-read repeat-tract length histograms at indel calls, for the
  repeat-length features; runs per region shard.
- `winnowr train`: per-variant-type XGBoost genotype models (drop, keep, flip het<->hom) and
  promote models for non-PASS calls (stay, promote, promote and flip). Thresholds are chosen by
  leave-one-sample-out cross-validation, or leave-one-group-out with `--sample-groups`, to
  maximise the F1 gain under vcfeval arithmetic. Indel models can add repeat-length features
  (`--repeat-length`) from a stutter model fitted at training time, and a re-genotype class
  (`--regenotype`) that replaces a call's genotype with a stutter-aware length caller's.
- `winnowr evaluate`: F1 of each decision policy on a held-out labelled sample, optionally with
  thresholds tuned on it (`--tuned`).
- `winnowr score`: applies genotype and promote models to a sample and writes the re-genotyped,
  filtered VCF (`--mode GENOTYPER` or `VETO`).
- The model sidecar (`<prefix>.winnowr.json`, format version 2) records the winnowr and XGBoost
  versions and the training reference's contig lengths; `score` and `evaluate` check calls
  against that reference, and older sidecars load with a warning.

### Changed

These change the behaviour of pre-release builds of 0.1.0. Models they trained keep loading and
scoring; retraining with this release can change labels and CV-chosen thresholds.

- `train --strlen` is renamed `train --repeat-length` (`--strlen` takes table paths in `extract`
  and `score`).
- `train --no-neighbourhood` is now the negative form of `--neighbourhood` (on by default);
  `--no-no-neighbourhood` is gone.
- `train` refits the stutter model inside each cross-validation fold on that fold's training
  samples, so held-out samples' length-call features and re-genotype labels no longer depend on
  their own truth.
- `train` fails before any fitting when a sample name is repeated, the tables come from different
  references, `--validation-fraction` is not in [0, 1), or the stutter model would be fitted on no
  reads; the repeat-length error names the flags involved.
- Re-genotype labels also credit calls with the wrong repeat length (hap.py local matches), which
  re-genotyping fixes; re-genotypes that `score` cannot write are no longer credited.
- Truth allele length changes are matched to calls of the same variant type, and a truth FN is
  promote-labelled for at most one non-PASS call.
- `extract` reports how the hap.py labels joined to the calls, warns when hap.py scored non-PASS
  calls (it must be run with `--pass-only`), fails when most hap.py records match no call, tries
  a padded SNP's differing base when its POS has no label, and reports strlen coverage, failing
  when no indel call has strlen evidence. It also records the reference contig lengths in the
  feature table.
- `evaluate` warns when the table's sample is a training sample of the models, and fails when
  the table's reference differs from the models'.
- `score` fails when the VCF's reference differs from the models', and warns about `--regions` or
  `--strlen` inputs no model uses. In the output VCF, a decision applies only to the record that
  was scored (not to other records with the same alleles), flips and re-genotypes set PGT, PID
  and PS to missing, and a re-genotype reuses an ALT only when it is the repeat tract resized in
  its phase.
- `strlen` takes contig names containing colons as whole-contig regions and rejects a region
  start of 0.
- Empty (zero-length) BED intervals are ignored.
