# score

Applies models to a sample's calls and writes the VCF with drops, promotions, flips and
re-genotypes. See [Scoring](../user-guide/scoring.md).

## Usage

```console
# With strlen and pileup tables
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --strlen sample.strlen.tsv.gz --pileup sample.pileup.tsv.gz -o sample.winnowr.vcf.gz
# Measuring the reads while scoring, dropping only
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --alignments sample.cram --reference ref.fa --mode veto -o sample.winnowr.vcf.gz
```

## Options

```text
{{#include ../../_generated/cli/score.txt}}
```
