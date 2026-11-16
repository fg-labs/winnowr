# Evaluating and stratifying

## `winnowr evaluate`

`evaluate` reports the F1 of a model's decisions on a labelled sample, with vcfeval's counting (a
genotype mismatch is both an FP and an FN). Give it the genotype model, the promote model, or one
of each, of one variant type, and the labelled feature table of a sample held out from their
training:

```console
winnowr evaluate --models snp.winnowr.json snp-promote.winnowr.json --table HG002.parquet \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz -o HG002.snp.tsv
```

A training sample of the models gives in-sample, optimistic metrics, and `evaluate` warns.
`--regions` takes the region BEDs `score` would get, so that calls are decided as `score` decides
them; only the confident regions (`notinalldifficultregions`) are read, since the models' region
features come from the table. `evaluate` also takes the options that change how `score` acts
(`--keep-fallback`, `--outside-indel-t-drop` and the experimental ones; see
[Scoring](scoring.md)).

The output is a TSV with one row per policy:

| policy                    | what it is                                                          |
|---------------------------|---------------------------------------------------------------------|
| `baseline`                | no action: the calls as made                                        |
| `veto`                    | the model in veto mode, at its veto thresholds                      |
| `genotyper`               | the model in genotyper mode, at its genotyper thresholds            |
| `genotyper_no_regenotype` | (re-genotype models) the genotyper with re-genotyping off           |
| `flip_only`               | flips alone, at the genotyper's `t_flip`                            |
| `promote`, `promote_no_flip` | (promote models) promotions, with and without promote+flip       |
| `genotyper+promote`       | (both models) the genotyper's decisions plus the promotions         |
| `*_tuned`                 | (`--tuned`) thresholds tuned on this sample: optimistic             |

Each row has the thresholds, the counts of each action and how many were right (e.g. `dropped`,
`dropped_tp`, `flipped`, `flips_fixed`, `flips_broke_tp`), the resulting TP, FP and FN, F1, and the
F1 gain over the baseline.

## `winnowr stratify`

`stratify` reports a model set's decisions, outcomes, F1 and P(keep) calibration by *benchmark
stratum*, for one sample with two truth sets (e.g. GIAB v4.2.1 and the T2T Q100 benchmark). It
takes the models (at most one genotype and one promote model per variant type), and per truth set,
in the same order: a name, the labelled feature table (extracted from the same VCF), the
confident-regions BED and the hap.py VCF the table was labelled from.

```console
winnowr stratify --models snp.winnowr.json snp-promote.winnowr.json \
  --truth-names v421 v5q --tables HG002.v421.parquet HG002.v5q.parquet \
  --truth-beds HG002.v421.bed HG002.v5q.bed --happy HG002.v421.happy.vcf.gz HG002.v5q.happy.vcf.gz \
  --regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz -o HG002.snp
```

Each call is in one stratum by whether its REF span is inside each truth set's confident regions:
`both`, `<second>-only`, `<first>-only` or `neither`. Calls are scored as `score` scores them
(`--mode`, default `genotyper`). It writes:

- `<output>.strata.tsv`: the actions, outcomes and F1 of the labelled calls per truth set and
  stratum (the `all` rows match `evaluate`),
- `<output>.rates.tsv`: the action rates of every scored call per stratum, variant type and
  chromosome, which need no truth,
- `<output>.calibration.tsv`: binned P(keep) against the observed keep rate.

`--calls` also writes each scored call's stratum, labels, probabilities and action (gzipped TSV).
`--contigs` reports only the calls and truth FNs on some contigs, e.g. those held out from the
models' training.

### GIAB stratifications

`stratify` also reports each metric within named regions, as hap.py's `--stratification` does,
given a hap.py stratification manifest (`--stratification`, one `NAME<TAB>PATH` BED per line, paths
relative to the manifest's directory) or regions as `--report-region NAME=PATH`. The
[GIAB genome stratifications](https://github.com/usnistgov/giab-stratifications) are one such set.
A call or truth FN is in a region if its REF span overlaps it, regions may overlap, and each truth
set is scored within its own confident regions. The results go to `<output>.regions.strata.tsv`,
`<output>.regions.calibration.tsv` and `<output>.regions.rates.tsv` (action rates per region and
benchmark stratum, including the parts outside the truth sets).

## Holding out half of a training sample

A sample can both train the models and evaluate them, on different chromosomes:

```console
# Keep the even chromosomes of HG002's table for training (truth FNs recounted on them)
winnowr subset --table HG002.parquet --happy HG002.happy.vcf.gz \
  --contigs chr2 chr4 chr6 chr8 chr10 chr12 chr14 chr16 chr18 chr20 chr22 -o HG002.even.parquet
# Train with it
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet HG002.even.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type snp -o snp
# Report on the odd chromosomes, with the full tables
winnowr stratify --models snp.winnowr.json --contigs chr1 chr3 chr5 chr7 chr9 chr11 chr13 chr15 chr17 chr19 chr21 \
  --truth-names v421 v5q --tables HG002.v421.parquet HG002.v5q.parquet \
  --truth-beds HG002.v421.bed HG002.v5q.bed --happy HG002.v421.happy.vcf.gz HG002.v5q.happy.vcf.gz -o HG002.odd.snp
```

`subset` keeps a table's calls on `--contigs` whose REF span does not overlap `--exclude-bed`, and
recounts the truth FNs from the hap.py VCF on the same terms, so that no label from elsewhere enters
training or the threshold search. Since HG002 is then a training sample of the model, `stratify`
warns that its metrics are in-sample and optimistic; that is expected here, as the reported
chromosomes were held out. The features are kept as extracted: per-sample features (e.g.
the depth normalization) still come from all of the sample's calls.
