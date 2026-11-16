# Training

`winnowr train` trains one model: a genotype model (`--task genotype`, the default) or a promote
model (`--task promote`) for one variant type (`--variant-type snp` or `indel`). It reads labelled
feature tables from `winnowr extract --happy`, one per sample, and writes one model file
(`<output>.winnowr.json`).

```console
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type snp -o snp
```

The tables must be of at least two distinct samples, and must have been extracted from the same
contigs, with the same region names, from calls on the same reference (`train` checks the contig
lengths each table records).

## Choosing the thresholds by cross-validation

A model acts on a call when its probability for the class reaches a threshold: `t_drop` and
`t_flip` for a genotype model (and `t_regenotype` for a re-genotype model), `t_promote` and
`t_flip` for a promote model. `train` chooses them by cross-validation on the training samples:
each fold fits a model on the other folds' calls and predicts its own, and the thresholds are
those that give the greatest F1 gain summed over the folds. Genotype models get thresholds for each
mode, `genotyper` and `veto` (see [Scoring](scoring.md)).

The candidates come from a built-in grid from 0.2 up to near certainty, or from `--grid`. "Never
act" is always a candidate, with a gain of zero, so a chosen threshold never lowers the summed F1
of the folds. Among candidates of equal gain the most conservative (the highest) wins, so a model
whose folds give no evidence for acting never acts.

F1 is counted as vcfeval counts it: a genotype mismatch is both an FP and an FN. Dropping an FP
removes the FP; dropping a TP turns it into an FN; a correct flip of an FP turns an FP and an FN
into a TP, and a flip of a TP does the reverse.

### Sample groups and chromosome folds

`--cv-folds` sets how the folds are made:

- `samples` (the default) holds out each sample group in turn. `--sample-groups` lists the samples
  held out together, one comma-separated group per value, e.g. a trio: `HG005,HG006,HG007`. Related
  samples share most of their sites, and would otherwise leak into each other's training folds. A
  sample not in a group is held out alone, so there must be at least two groups.
- `chromosomes` holds out groups of whole chromosomes from every sample at once, so every fold fits
  every sample, and the calls at a site in related samples are held out together.
  `--chromosome-folds` (default 5) sets the number of groups, made of about equal numbers of calls.
  `--sample-groups` is not used.

The choice is recorded in the model.

## Early stopping and refitting

Every fit holds out about a tenth of its calls (`--validation-fraction`, 0.1) to stop boosting
when the validation log-loss has not improved for `--early-stopping-rounds` (20) rounds, up to
`--max-rounds` (3000). The held-out calls are whole genomic blocks of `--validation-block-size`
(10,000 bp), from every sample, so that the calls at the same site in other samples, and
neighbouring calls, do not leak into training. `--validation-fraction 0` turns early stopping
off.

The final model, fitted on every training sample, is then refitted on all of its calls, the held-out
blocks included, for the number of rounds its early stopping chose. `--refit-final false` keeps the
early-stopped fit. The refit has no effect without early stopping, and is skipped, with a warning,
when the validation fraction holds out no block.

The other XGBoost parameters (`--learning-rate`, `--max-leaves`, `--max-depth`,
`--min-child-weight`, `--reg-lambda`, `--seed`) have regularized defaults; see
[train](../tools/train.md). `train` checks them before reading the tables: `--learning-rate` must be
positive, `--min-child-weight` and `--reg-lambda` at least 0, all finite, and `--max-leaves` and
`--max-depth` at most 2147483647, with 0 for no limit.

## Features

By default a model uses the call's features from the feature table, the region flags (one per
`extract --regions` name) and the multi-site neighbourhood features (`--no-neighbourhood` leaves
those out). Two more sets are opt-in:

- `--repeat-length` (indel genotype models): read-length features of the call's repeat tract, from
  tables extracted with `--strlen`. `--regenotype` adds them too, with the re-genotype class. See
  [Repeat-length features](repeat-length.md).
- `--pileup-features`: allele read counts by MAPQ, from tables extracted with `--pileup`. See
  [Pileup features](pileup.md).

## Reproducibility

A model depends on the number of threads; see [Performance tuning](performance.md#threads).

## Evaluating on a sample that also trains

A sample can both train a model and evaluate it, on different chromosomes; see
[Holding out half of a training sample](evaluating.md#holding-out-half-of-a-training-sample).
