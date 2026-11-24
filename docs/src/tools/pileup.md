# pileup

Counts the reads supporting each SNP and indel call's alleles in three MAPQ bins, for models with
pileup features. See [Pileup features](../user-guide/pileup.md).

The table records how the reads were aligned, read from the header of `--alignments` (with
`--alignment-tags` over it), in an `##alignment=` line. See
[Alignment provenance](../user-guide/alignment-provenance.md).

## Usage

```console
winnowr pileup --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa -o sample.pileup.tsv.gz
```

## Options

```text
{{#include ../../_generated/cli/pileup.txt}}
```
