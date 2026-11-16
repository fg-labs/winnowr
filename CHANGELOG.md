# Changelog

All notable changes to this project are documented in this file.

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
