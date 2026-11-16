# train

Trains a genotype or promote model for one variant type from labelled feature tables, choosing its
thresholds by cross-validation. See [Training](../user-guide/training.md).

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
