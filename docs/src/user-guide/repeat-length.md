# Repeat-length features

An indel call in a homopolymer or short tandem repeat can have the wrong repeat length or the wrong
zygosity. Indel genotype models can use *repeat-length* features: the lengths of the call's
repeat tract seen in the reads.

## Measuring the reads: `winnowr strlen`

```console
winnowr strlen --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa -o sample.strlen.tsv.gz
```

For every indel record (of any FILTER) whose POS is in `--regions` (by default the autosomes `chr1`
to `chr22`), `strlen` finds the repeat tract of its first pure insertion or deletion allele in the
reference, and histograms the observed minus reference tract length over the reads that span the
tract with aligned flank bases on each side. Unmapped, secondary, supplementary, duplicate and
QC-failed reads are not used, and at most 600 reads are measured per site: the cap takes the first
600 such reads overlapping the tract and its flanks, before it checks that they span them and have
the MAPQ `--reads` asks for, so at a very deep site the histogram can hold fewer reads than the
depth. It writes the read settings, then one row per indel record with a pure insertion or deletion
allele; other records are skipped, and counted in the log. Give the table (or one per shard, e.g.
per `--regions` range) to `extract --strlen` or `score --strlen`.

## Read settings

`--reads` sets which reads measure the tracts, as comma-separated settings over the defaults:

- `mapqN`: only reads with MAPQ of at least N (default 20),
- `flankN`: N aligned flank bases on each side of the tract (default 5, at most 10000),
- `legacy`: `mapq0,flank2`, as winnowr measured before the settings were recorded.

The settings are written at the top of the strlen table, kept in the feature table and recorded in
the trained model. A model only scores repeat-length evidence measured with its training tables'
settings: `score --strlen`, `evaluate` and `stratify` stop with an error otherwise, and `score
--alignments` measures with the model's settings. Tables and models made before winnowr recorded
the settings were measured with any MAPQ and two flank bases, and are read that way; to score such
a model from `strlen` tables, measure them with `winnowr strlen --reads legacy`.

## Training with repeat-length features

Give each sample's strlen table to `extract`, then train the indel model with `--repeat-length`:

```console
winnowr extract --vcf HG001.vcf.gz --gvcf HG001.g.vcf.gz --happy HG001.happy.vcf.gz \
  --strlen HG001.strlen.tsv.gz --regions segdups=segdups.bed.gz -o HG001.parquet
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type indel --repeat-length -o indel
```

`train` fits a *stutter model* on the training samples' truth-homozygous calls: the chance that a
read spanning a repeat tract shows a given length difference, given the true allele's length change,
by repeat period and tract length. It is saved with the model. A stutter-aware diploid length caller
then picks the pair of allele lengths that best explains the reads. Its candidates are no change and
the called alleles, always, plus those of the five most frequent observed differences (ties to the
smaller difference) that were seen in at least two reads. The model's features are the read-length
histogram and the length caller's call and log-likelihood margins. Each cross-validation fold refits
the stutter model on its own training samples.

## Re-genotyping

`train --regenotype` (which implies `--repeat-length`) adds a fourth class, *re-genotype*: calls
whose genotype a het <-> hom flip does not fix but the length caller's genotype does. Its threshold,
`t_regenotype`, is chosen by cross-validation with the genotyper's thresholds. In
`--mode genotyper`, `score` re-genotypes a call with a diploid genotype (not `./.` or haploid) that
the length caller disagrees with when P(re-genotype) reaches it and the call is neither dropped nor
flipped (P(drop) reaching `t_drop` comes first, then P(flip) reaching `t_flip`): the call gets the
length caller's genotype, with an ALT allele added when needed, the INFO flag `WINNOWR_REGENOTYPE`
and its old genotype in FORMAT `WINNOWR_OGT`. Re-genotyping to `0/0` is a drop. That order is the
threshold rule: with the experimental `--expected-f1 indel`, `score` instead drops a call outside
the confident regions when P(drop) reaches `--outside-indel-t-drop`, else flips it when P(flip) is
above 0.5, else re-genotypes it when P(re-genotype) is above 0.5, else drops it when P(keep) is
below 0.5 (see [Scoring](scoring.md#how-score-acts-on-indel-calls)).

## Scoring

A model with repeat-length features needs the sample's evidence: `score --strlen` with tables
measured with the model's read settings, or `score --alignments sample.cram --reference ref.fa`,
which measures the reads itself, over the models' contigs, with the models' settings.
