# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-11-15

The first release of the Rust rewrite of winnowr. Its outputs and models differ from the Python
version's by design; see
[Coming from the Python version](https://winnowr.readthedocs.io/en/latest/getting-started/python-version.html)
for the changes in the defaults.

### Features

- `extract`, `strlen`, `pileup`, `train`, `score`, `evaluate`, `stratify` and `subset`.
- Models are one `<name>.winnowr.json` file (model format 3) with the XGBoost booster embedded.
  Models from the Python version (formats 1 and 2) are not read.
- Training holds out whole 10 kb genomic blocks for early stopping, then refits the final model on
  every call (`--refit-final`, on by default).
- Repeat-length features (`winnowr strlen`) measure reads with `--reads mapq20,flank5` by default;
  `--reads legacy` measures as the Python version did.
- Pileup features (`winnowr pileup`, `train --pileup-features`) count reference, alternate and
  other reads at MAPQ 0, 1-19 and 20 or more, with the settings `mapq20,bq10,flank5`. The settings
  are recorded in the pileup table and in each model trained with them, and `score` refuses
  evidence measured with other settings.
- XGBoost 3.5.0-dev (upstream `293cad1a8`) is vendored and linked statically, with one local
  patch that keeps OpenMP teams a fixed size under libgomp.
