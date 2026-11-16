# Pileup features

HaplotypeCaller ignores reads below its minimum MAPQ (1 in DRAGEN mode). When an aligner places one
allele's reads at a site with MAPQ 0 (e.g. reads ALT-projected onto the primary assembly, or from a
paralogous copy), a call made from the other reads can have the wrong zygosity, and nothing in the
VCF shows it. *Pileup* features count each allele's reads at every MAPQ, so a model can see them.

## Counting the reads: `winnowr pileup`

```console
winnowr pileup --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa -o sample.pileup.tsv.gz
```

For every SNP and indel record (of any FILTER) whose POS is in `--regions` (by default the
autosomes `chr1` to `chr22`), `pileup` counts the reads supporting its reference allele, an
alternate allele, or neither, in three MAPQ bins: 0, 1-19 and 20 or more. Reads of every MAPQ are
counted; unmapped, secondary, supplementary, duplicate and QC-failed reads are not. At most 600
reads are counted per site: the cap takes the first 600 such reads overlapping the site, before
their allele and base quality are checked, so at a very deep site the reads that support no allele
use up some of it, and the counts can be lower than the depth.

- A SNP's allele is the read's base at it, counted at base quality 10 or more, or when the read
  has no base qualities (`*` in QUAL).
- An indel's allele is the read's length change across the indel's repeat tract (as `strlen`
  measures it, from reads aligned on five flank bases outside it), so an indel placed anywhere in
  the tract counts. When an indel record also has an alternate allele as long as the reference
  (e.g. `A>T,AT`), a read with no length change is told apart by its bases where that allele
  differs.
- Both reads of a fragment whose mates overlap the call are counted.
- Records of other types (e.g. MNPs), and indels without a pure insertion or deletion allele, get
  no counts.

`--settings` changes the counting, as comma-separated settings over the defaults
`mapq20,bq10,flank5`: `mapqN` (the lowest MAPQ of the high bin), `bqN` (the minimum base quality of
a SNP base) and `flankN` (the aligned flank bases an indel read needs on each side of the repeat
tract). The settings are written at the top of the table and kept in the feature table.

## Training and scoring

Give each sample's pileup table to `extract --pileup`, then train with `--pileup-features`:

```console
winnowr extract --vcf HG001.vcf.gz --gvcf HG001.g.vcf.gz --happy HG001.happy.vcf.gz \
  --pileup HG001.pileup.tsv.gz --regions segdups=segdups.bed.gz -o HG001.parquet
winnowr train --tables HG001.parquet HG005.parquet HG006.parquet HG007.parquet \
  --sample-groups HG005,HG006,HG007 --variant-type snp --pileup-features -o snp
```

The model gets the counts and fractions of them (`pu_*`, e.g. `pu_ref_frac`, the reference reads'
share of the reference and alternate reads at any MAPQ), and records the settings. Such a model
needs pileup evidence measured with those settings: `score` stops with an error without `--pileup`
tables or `--alignments`, and `score`, `evaluate` and `stratify` stop with an error when the
settings differ. Models trained without `--pileup-features` neither need nor read it.
