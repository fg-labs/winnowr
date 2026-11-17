# Model files

`winnowr train` writes one model per file, named `<name>.winnowr.json` (`train` appends
`.winnowr.json` to `-o` when it is missing). The file is a JSON object holding winnowr's model
information and, under `booster`, the XGBoost booster in XGBoost's JSON format. `score`,
`evaluate` and `stratify` refuse a model file whose name does not end in `.winnowr.json`.

## What a model records

| key                  | what it holds                                                                  |
|----------------------|--------------------------------------------------------------------------------|
| `format_version`     | the model format version, 3                                                    |
| `variant_type`       | `snp` or `indel`                                                               |
| `task`               | `genotype` (scores PASS calls) or `promote` (scores non-PASS calls)            |
| `features`           | the ordered feature names                                                      |
| `region_names`       | the region BEDs the features use, which `score` must get under the same names  |
| `thresholds`         | a genotype model's cross-validated thresholds per mode (`genotyper`, `veto`)   |
| `promote_thresholds` | a promote model's cross-validated thresholds                                   |
| `training_samples`   | the training sample names                                                      |
| `contigs`            | the contigs the training calls came from (of every training table)             |
| `reference_contigs`  | the lengths of those contigs in the reference the calls were made on           |
| `params`             | the training parameters, including the thread count                            |
| `stutter`            | the stutter model of a model with repeat-length features                       |
| `regenotype`         | whether the model has the re-genotype class                                    |
| `strlen_settings`    | the read settings of the repeat-length evidence it was trained on              |
| `pileup_settings`    | the settings of the pileup evidence it was trained on                          |
| `winnowr_version`    | the winnowr version that trained it                                            |
| `xgboost_version`    | the XGBoost version that trained it                                            |

A threshold of 1.01 means the model never takes that action. `params` records `unfixable_flips`
and `cv_folds` only when they are not the defaults, and `refit_final` only when the model was
refitted.

## Compatibility

When it loads a model, winnowr checks:

- **Format version.** This winnowr reads format version 3 only, and refuses other versions:
  train new models with `winnowr train`.
- **winnowr version.** A model trained by another winnowr version (comparing package versions,
  not git commits) loads with a warning.
- **Reference.** `score` stops with an error when a contig the model was trained on is missing from
  the VCF's header or has another length there; `evaluate` and `stratify` stop with an error when a
  contig the feature table records has another length than in the model. A model that records no
  reference contigs is not checked, with a warning.
- **Region names.** `score` stops with an error when a region name the model uses is not given to
  `--regions`.
- **Evidence settings.** A model with repeat-length features scores only strlen evidence measured
  with its `strlen_settings`, and a model with pileup features only pileup evidence measured with
  its `pileup_settings`. Models that record no strlen settings are read as measured with
  any MAPQ and two flank bases (`strlen --reads legacy`).
- **Consistency.** The booster must have the model's features, in order, and its number of classes.

The models given to one `score` run must be trained on the same contigs, with at most one genotype
and one promote model per variant type.
