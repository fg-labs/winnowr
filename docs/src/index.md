# winnowr

winnowr is machine-learned genotyping and filtering of germline small-variant calls. It
post-processes single-sample GATK HaplotypeCaller calls (made in gVCF mode, e.g. with
`--dragen-mode`) and writes the same VCF with some calls filtered, some promoted to PASS and some
re-genotyped.

For each PASS call, an XGBoost model for its variant type (SNP or indel) predicts one of three
classes: *drop* (the site is really `0/0`), *keep* (the call is right) or *flip* (the alleles are
right but the zygosity is wrong). An optional *promote* model per variant type scores the non-PASS
calls. Indel models can also use the lengths of repeat tracts measured in the reads, and re-genotype
a call to the genotype those lengths support. Models of either variant type can also use counts of
each allele's reads by mapping quality (MAPQ). See [Classes and
labels](user-guide/classes-and-labels.md).

Each model's thresholds are chosen by cross-validation on the training samples, to raise F1, and
are stored in the model file with it. By default `score` also drops indel calls whose P(keep) is
below 0.5, and uses a drop threshold of 0.6 for indel calls outside GIAB's confident regions. See
[Training](user-guide/training.md) and [Scoring](user-guide/scoring.md).

No trained models are distributed. You train your own from samples with a truth set, such as the
[GIAB](https://www.nist.gov/programs-projects/genome-bottle) benchmarks, labelled by
[hap.py](https://github.com/Illumina/hap.py). The
[training quick start](getting-started/quick-training.md) lists the steps.

> **Demonstration code only.** winnowr is a research prototype, shared to illustrate an approach
> to machine-learned filtering and genotyping of germline variant calls. It is not supported, has
> not been validated, and is not intended for clinical, diagnostic or production use. Use it at
> your own risk.

## The commands

winnowr has eight commands: `strlen`, `pileup`, `extract`, `subset`, `train`, `evaluate`,
`stratify` and `score`. The [tool reference](tools/overview.md) describes each, with its full
`--help`.
