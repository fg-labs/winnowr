# subset

Restricts a labelled feature table to some of its contigs, or away from some regions, and recounts
the truth FNs from the hap.py VCF on the same terms. See [Holding out half of a training
sample](../user-guide/evaluating.md#holding-out-half-of-a-training-sample).

## Usage

```console
# Keep chromosome 22
winnowr subset --table HG002.parquet --happy HG002.happy.vcf.gz --contigs chr22 -o HG002.chr22.parquet
# Leave out the calls in segmental duplications
winnowr subset --table HG002.parquet --happy HG002.happy.vcf.gz --exclude-bed segdups.bed.gz -o HG002.noseg.parquet
```

## Options

```text
{{#include ../../_generated/cli/subset.txt}}
```
