# Quick start: training

A model is trained from samples with a truth set, such as the GIAB benchmarks (HG001 to HG007),
whose calls hap.py has labelled. You need, for each sample:

- the HaplotypeCaller VCF and gVCF, as for [scoring](quick-scoring.md),
- the hap.py output VCF, from hap.py run with `--engine vcfeval --pass-only` against the truth
  set,
- the truth VCF and its confident-regions BED, to label non-PASS calls for promote models,
- the region BEDs to use as features, the same for every sample.

At least two samples are needed to train, and one more held out to evaluate on.

```console
# Extract a labelled feature table per sample (--truth-vcf and --truth-bed label non-PASS calls too)
winnowr extract --vcf HG001.vcf.gz --gvcf HG001.g.vcf.gz --happy HG001.happy.vcf.gz \
  --truth-vcf HG001.truth.vcf.gz --truth-bed HG001.truth.bed --reference ref.fa \
  --regions segdups=segdups.bed.gz -o HG001.parquet
# Train a model per variant type; a trio is held out together in cross-validation
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type snp -o snp
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type indel -o indel
# Report F1 on a held-out sample, one variant type at a time
winnowr evaluate --models snp.winnowr.json --table HG002.parquet \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz -o HG002.snp.tsv
winnowr evaluate --models indel.winnowr.json --table HG002.parquet \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz -o HG002.indel.tsv
```

`evaluate` needs the not-difficult regions to decide indel calls as `score` does; without them its
indel F1 is not that of the calls `score` writes.

Run `extract` the same way for every sample, the held-out one included. `train` writes
`snp.winnowr.json` and `indel.winnowr.json`; score with them as in the
[scoring quick start](quick-scoring.md).

Without gVCFs, extract every table with `--feature-source vcf` (and no `--gvcf`), which derives
the features of gVCF-mode annotations from the VCF's standard annotations; see
[Feature sources](../user-guide/feature-sources.md). `train` takes tables of one feature source
only.

`extract` reads the autosomes `chr1` to `chr22` by default (`--contigs`). The training tables must
have been extracted with the same region names, from calls on the same reference; they may cover
different contigs (see [Training](../user-guide/training.md)).

To train on only some contigs of a sample, e.g. to evaluate it on the others, `subset` keeps them
and recounts the table's truth FNs on them (see
[Holding out half of a training sample](../user-guide/evaluating.md#holding-out-half-of-a-training-sample)):

```console
winnowr subset --table HG002.parquet --happy HG002.happy.vcf.gz --contigs chr1 chr2 -o HG002.chr1-2.parquet
```

## More models

- A promote model per variant type scores non-PASS calls: add `--task promote` to `train`.
- Repeat-length features for indel models: measure each sample with `winnowr strlen`, give the
  tables to `extract --strlen`, and train with `--repeat-length`, or `--regenotype` to also learn
  when to re-genotype a call. See [Repeat-length features](../user-guide/repeat-length.md).
- Pileup features: count each sample with `winnowr pileup`, give the tables to `extract --pileup`,
  and train with `--pileup-features`. See [Pileup features](../user-guide/pileup.md).

```console
# A SNP promote model (tables extracted with --truth-vcf and --truth-bed)
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type snp --task promote -o snp-promote
# An indel model with repeat-length features and re-genotyping (tables extracted with --strlen)
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type indel --regenotype -o indel-regenotype
```

`evaluate` takes the genotype and promote models of one variant type together:

```console
winnowr evaluate --models snp.winnowr.json snp-promote.winnowr.json --table HG002.parquet \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz -o HG002.snp.tsv
```
