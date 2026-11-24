# train

Trains a genotype or promote model for one variant type from labelled feature tables, choosing its
thresholds by cross-validation. See [Training](../user-guide/training.md).

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
