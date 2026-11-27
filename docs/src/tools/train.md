# train

Trains a genotype or promote model for one variant type from labelled feature tables, choosing its
thresholds by cross-validation. See [Training](../user-guide/training.md).

The model records a region BED's digest only when every training table records the same digest
under that name. If any table omits it (for example, a legacy table), the model omits that feature
digest too. `score` checks a BED against the digest any of the models it is given records for that
name, so it uses the feature BED unchecked only when no model records a digest for it, as a feature
or by an active policy.
Tables that record different digests under one name are an error. `--policy-regions
notinalldifficultregions=PATH` separately records the not-difficult regions' policy digest, so
that `score` can check them. If its feature and policy digests differ, `train` warns but records
both: table-based `evaluate` and `stratify` can use different BEDs for the two roles, but `score`
cannot use one live BED for both. Use the same BED for both roles to score with that policy, or
disable the policy's use of the BED. See [Scoring](../user-guide/scoring.md).

The model records how the training tables' reads were aligned. Tables whose recorded alignment
provenance differs are an error, unless `--mixed-alignments`, which records only the tags they
share; `score` then compares the model with the reads on those tags only. See
[Alignment provenance](../user-guide/alignment-provenance.md).

## Usage

```console
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type snp -o snp
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type snp --task promote -o snp-promote
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --cv-folds chromosomes --variant-type indel --regenotype -o indel
```

## Options

```text
{{#include ../../_generated/cli/train.txt}}
```
