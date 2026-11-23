# score

Applies models to a sample's calls and writes the VCF with drops, promotions, flips and
re-genotypes. See [Scoring](../user-guide/scoring.md). The models' feature source decides
whether `--gvcf` is read (see [Feature sources](../user-guide/feature-sources.md)).

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
# Models trained with extract --feature-source vcf: no gVCF
winnowr score --vcf sample.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  -o sample.winnowr.vcf.gz
```

## Options

```text
{{#include ../../_generated/cli/score.txt}}
```
