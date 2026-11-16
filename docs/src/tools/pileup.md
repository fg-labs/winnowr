# pileup

Counts the reads supporting each SNP and indel call's alleles in three MAPQ bins, for models with
pileup features. See [Pileup features](../user-guide/pileup.md).

## Usage

```console
winnowr pileup --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa -o sample.pileup.tsv.gz
```

## Options

```text
{{#include ../../_generated/cli/pileup.txt}}
```
