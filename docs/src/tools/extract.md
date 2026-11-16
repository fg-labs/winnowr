# extract

Extracts the features of a sample's SNP and indel calls into a Parquet feature table. With hap.py
output (`--happy`) it also labels the PASS calls, for training and evaluation, and with
`--truth-vcf` and `--truth-bed` the non-PASS calls, for promote models. See [Classes and
labels](../user-guide/classes-and-labels.md).

When the VCF lacks the raw annotations (its header does not declare them, or none of its records
has RAW_MQandDP), they come from the gVCF (`--gvcf`): each call takes them from the gVCF variant
site at its position that has its alleles, or else from the first variant site at its position.
Reference blocks carry no annotations and are skipped.

## Usage

```console
winnowr extract --vcf HG001.vcf.gz --gvcf HG001.g.vcf.gz --happy HG001.happy.vcf.gz \
  --truth-vcf HG001.truth.vcf.gz --truth-bed HG001.truth.bed --reference ref.fa \
  --strlen HG001.strlen.tsv.gz --pileup HG001.pileup.tsv.gz \
  --regions segdups=segdups.bed.gz lowmappabilityall=lowmappabilityall.bed.gz -o HG001.parquet
```

## Options

```text
{{#include ../../_generated/cli/extract.txt}}
```
