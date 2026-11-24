# evaluate

Reports the F1 of a model's decisions, and of its promote model's, on a labelled sample held out
from training. See [Evaluating and stratifying](../user-guide/evaluating.md).

Its policy reads the not-difficult regions as `score`'s does: the `--regions
notinalldifficultregions=PATH` BED, else `regions/notinalldifficultregions.bed.gz` (`.bed.bgz`,
`.bed`) beside the models, checked against the digest the models record (`train
--policy-regions`). The models' region features come from the table, whose region BED digests are
compared with the models': a difference is an error, and a table that records none is not
checked (a warning). `--allow-region-mismatch` makes a different BED a warning; models that
record different digests of one region are still an error.

## Usage

```console
winnowr evaluate --models snp.winnowr.json snp-promote.winnowr.json --table HG002.parquet \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz -o HG002.snp.tsv
```

## Options

```text
{{#include ../../_generated/cli/evaluate.txt}}
```
