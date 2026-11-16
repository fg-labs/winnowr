# stratify

Reports models' decisions, outcomes, F1 and P(keep) calibration by benchmark stratum for a sample
with two truth sets, and optionally within named regions such as the GIAB stratifications. See
[Evaluating and stratifying](../user-guide/evaluating.md).

## Usage

```console
winnowr stratify --models snp.winnowr.json snp-promote.winnowr.json \
  --truth-names v421 v5q --tables HG002.v421.parquet HG002.v5q.parquet \
  --truth-beds HG002.v421.bed HG002.v5q.bed --happy HG002.v421.happy.vcf.gz HG002.v5q.happy.vcf.gz \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --stratification stratifications.tsv -o HG002.snp
```

## Options

```text
{{#include ../../_generated/cli/stratify.txt}}
```
