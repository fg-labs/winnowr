# Quick start: scoring

Scoring needs trained models: winnowr distributes none, so [train your own](quick-training.md)
first. It also needs, for the sample:

- the single-sample VCF from HaplotypeCaller (gVCF mode), indexed, with every FILTER, INFO and
  FORMAT tag of its records declared in its header (`extract` and `score` stop with an error naming
  the record otherwise),
- the gVCF it was genotyped from, indexed, when the VCF lacks the raw annotations (RAW_MQandDP,
  QUALapprox, VarDP, AS_RAW_MQ); not for models trained on features from the VCF and the reads
  (`extract --alignments`, which need the sample's BAM or CRAM instead) or from the VCF alone
  (`extract --feature-source vcf`): see [Feature sources](../user-guide/feature-sources.md),
- the region BEDs the models were trained with, under the same names, and, for indel genotype
  models or `--abstain-actions`, GIAB's GRCh38 `notinalldifficultregions` BED from the
  [genome stratifications](https://github.com/usnistgov/giab-stratifications) (without it, `score`
  warns and drops indel calls at the model's threshold everywhere, and fails with
  `--abstain-actions`): each given with `--regions NAME=PATH`, or kept beside the models as
  `regions/NAME.bed.gz` (or `.bed.bgz`, or `.bed`), where `score` finds them (see
  [Scoring](../user-guide/scoring.md)),
- for models with repeat-length or pileup features, the sample's indexed BAM or CRAM and the
  indexed, uncompressed reference FASTA.

```console
# Optional: measure repeat-tract read lengths, for models with repeat-length features
winnowr strlen --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa -o sample.strlen.tsv.gz
# Optional: count each allele's reads by MAPQ, for models with pileup features
winnowr pileup --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa -o sample.pileup.tsv.gz
# Score the calls and write the filtered VCF
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --strlen sample.strlen.tsv.gz --pileup sample.pileup.tsv.gz --mode genotyper -o sample.winnowr.vcf.gz
```

`--mode genotyper` (the default) lets the genotype models drop, flip and re-genotype calls;
`--mode veto` lets them only drop calls. Promote models promote calls in either mode.

Measure the tables with the settings the models were trained with (`strlen --reads`, `pileup
--settings`, which are the defaults unless the training tables used others); `score` stops with an
error when they differ.

Give `score` `--strlen` and `--pileup` only if the models use those features. `score` stops with an
error when a model needs a region BED that is neither given with `--regions` nor found beside the
models, or needs `--strlen` or `--pileup` that is not given (nor `--alignments` and `--reference` to
measure it), and warns about ones no model uses. In place of `--strlen` and `--pileup`, `score
--alignments sample.cram --reference ref.fa` measures the evidence itself, with the settings the
models were trained with:

```console
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --alignments sample.cram --reference ref.fa -o sample.winnowr.vcf.gz
```

Models trained on features from the VCF and the reads (`extract --alignments`) take
`--alignments` and `--reference` and no `--gvcf`: `score` measures the features from the reads,
with the settings the models record (see [Feature sources](../user-guide/feature-sources.md)).

Models trained on features from the VCF alone (`extract --feature-source vcf`) take no `--gvcf`
either: `score` derives the features from the VCF, as the models record.

```console
winnowr score --vcf sample.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  -o sample.winnowr.vcf.gz
```

`strlen` and `pileup` measure the autosomes `chr1` to `chr22` by default; `--regions` takes other
contigs or ranges (e.g. `chr22:20000000-30000000`). A range that starts past its contig's end is an
error.

With the region BEDs beside the models (`models/regions/segdups.bed.gz`, and so on), `score`
needs no `--regions`; it checks each BED against its applicable recorded digests. Feature-region
BEDs without one are used unchecked with a warning. Policy-only BEDs without a policy digest remain
unchecked without that warning:

```console
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz \
  --models models/snp.winnowr.json models/indel.winnowr.json \
  --alignments sample.cram --reference ref.fa -o sample.winnowr.vcf.gz
```

`sample.winnowr.vcf.gz` holds every record of `sample.vcf.gz`: dropped calls get FILTER
`WinnowrDrop`, and scored calls get the models' probabilities in INFO.
[Scoring](../user-guide/scoring.md) describes
the output in full.
