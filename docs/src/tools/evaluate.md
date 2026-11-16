# evaluate

Reports the F1 of a model's decisions, and of its promote model's, on a labelled sample held out
from training. See [Evaluating and stratifying](../user-guide/evaluating.md).

## Usage

```console
winnowr evaluate --models snp.winnowr.json snp-promote.winnowr.json --table HG002.parquet \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz -o HG002.snp.tsv
```

## Options

```text
{{#include ../../_generated/cli/evaluate.txt}}
```
