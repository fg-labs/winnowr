[![Build](https://github.com/fg-labs/winnowr/actions/workflows/check.yml/badge.svg)](https://github.com/fg-labs/winnowr/actions/workflows/check.yml)
[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/fg-labs/winnowr/blob/main/LICENSE)

# winnowr

<!-- The opening below is repeated in docs/src/index.md: change both together. -->

**winnowr post-processes single-sample GATK HaplotypeCaller VCFs with models you train on
truth-labelled samples.**

**What sets winnowr apart**

- **Works on a single sample:** no cohort or joint calling.
- **Fixes calls, not just filters them:** it can drop false calls, flip wrong zygosity,
  re-genotype indels in repeats, and rescue true calls that were filtered.
- **Uses evidence the VCF lacks:** repeat-tract lengths and reads below HaplotypeCaller's MAPQ
  cut-off.
- **Tuned for F1:** cross-validation against your truth data picks each model's thresholds.
- **Drops in after HaplotypeCaller:** no re-calling.

**The user guide is at [winnowr.readthedocs.io](https://winnowr.readthedocs.io).**

> [!WARNING]
> **Demonstration code only.** winnowr is a research prototype, shared to illustrate an approach to
> machine-learned filtering and genotyping of germline variant calls. It is not supported, has not
> been validated, and is not intended for clinical, diagnostic or production use. No trained models
> are distributed. Use it at your own risk.

<p>
<a href="https://fulcrumgenomics.com">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fg-labs/winnowr/main/.github/logos/fulcrumgenomics-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/fg-labs/winnowr/main/.github/logos/fulcrumgenomics-light.svg">
  <img alt="Fulcrum Genomics" src="https://raw.githubusercontent.com/fg-labs/winnowr/main/.github/logos/fulcrumgenomics-light.svg" height="100">
</picture>
</a>
</p>

[Visit us at Fulcrum Genomics](https://www.fulcrumgenomics.com) to learn more about how we can
power your bioinformatics with winnowr and beyond.

<a href="mailto:contact@fulcrumgenomics.com?subject=[GitHub inquiry]"><img src="https://img.shields.io/badge/Email_us-%2338b44a.svg?&style=for-the-badge&logo=gmail&logoColor=white"/></a>
<a href="https://www.fulcrumgenomics.com"><img src="https://img.shields.io/badge/Visit_Us-%2326a8e0.svg?&style=for-the-badge&logo=wordpress&logoColor=white"/></a>

## Overview

A genotype model scores the PASS calls of one variant type (SNP or indel), and predicts one of
three classes:

| class | meaning                      | action                            |
|-------|------------------------------|-----------------------------------|
| drop  | the site is really 0/0       | FILTER `WinnowrDrop`              |
| keep  | the call is right            | none                              |
| flip  | right allele, wrong zygosity | switch GT between `0/1` and `1/1` |

An indel genotype model trained with `--regenotype` has a fourth class, re-genotype. A promote model
scores non-PASS calls that do not overlap a PASS call instead, and predicts stay, promote, or
promote and flip.

By default `score` also applies two indel rules that are fixed, not chosen by cross-validation. An
indel call that the thresholds leave alone is dropped when its P(keep) is below 0.5. Given GIAB's
not-difficult regions (`--regions notinalldifficultregions=...`), an indel call outside them is
dropped at a threshold of 0.6, in place of the model's drop threshold.

The [user guide](https://winnowr.readthedocs.io) explains the classes, labels, training and scoring
in full.

## Install

From source, with Rust 1.88 or newer, CMake, a C++ compiler (Apple clang with
`brew install libomp` on macOS, or GCC on Linux) and libclang, which builds htslib's bindings
(`apt-get install libclang-dev` on Debian or Ubuntu; Apple clang provides it on macOS). See
[DEVELOPING](https://github.com/fg-labs/winnowr/blob/main/docs/DEVELOPING.md) for details.
XGBoost and htslib are built from source and linked statically, with the OpenMP runtime's static
library (`libomp.a` from Homebrew on macOS, GCC's `libgomp.a` on Linux). A source build on a Linux
distribution whose `libgomp.a` is not position-independent (e.g. RHEL, Amazon Linux) links the
shared libgomp instead, with a warning, unless built with `RUSTFLAGS='-C relocation-model=static'`
and an explicit `--target`; so does any build whose static-link test fails (see `xgboost/build.rs`).
On macOS, a `WINNOWR_LIBOMP_PREFIX` with only `libomp.dylib` (e.g. conda-forge's `llvm-openmp`)
links the shared libomp, with a warning (see DEVELOPING).

```console
git clone --recursive https://github.com/fg-labs/winnowr.git
cd winnowr
cargo install --locked --path .
```

`--recursive` fetches the vendored XGBoost sources (a git submodule). From version 0.1.1, winnowr
can also be installed from [crates.io](https://crates.io/crates/winnowr), with the same build
requirements; the `winnowr-xgboost` crate bundles the XGBoost sources, so no submodule is needed:

```console
cargo install --locked winnowr
```

Prebuilt binaries are on the [releases page](https://github.com/fg-labs/winnowr/releases), and
releases will also be published to [bioconda](https://bioconda.github.io).

## Quick start

No trained models are distributed, so first [train your own](#train-your-own-models).

### Score a sample

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

Give `score` `--strlen` and `--pileup` only if the models use those features (it warns about tables
no model uses), measured with the settings the models were trained with (`strlen --reads`, `pileup
--settings`); `score` stops with an error when they differ. In place of them, `score --alignments
sample.cram --reference ref.fa` measures the evidence itself. `--mode genotyper` (the default) lets
the genotype models drop, flip and re-genotype calls; `--mode veto` lets them only drop calls. Give
`--regions` the region names the models were trained with, or keep each BED beside the models as
`regions/NAME.bed.gz` (or `.bed.bgz`, or `.bed`) and leave `--regions` out: `score` finds them there, and checks
each BED against its applicable recorded digests. Without one, a feature-region BED is used unchecked
with a warning; a policy-only BED without a policy digest remains unchecked without that warning.

### Train your own models

Run hap.py (`--engine vcfeval --pass-only`) on each sample to train or evaluate on, against its
GIAB truth set.

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
indel F1 is not that of the calls `score` writes. Then score with the models as above. For
repeat-length or pileup features, give `extract` the `--strlen` or `--pileup` tables and `train`
`--repeat-length` (or `--regenotype`) or `--pileup-features`.

Without a gVCF, train on features derived from the VCF and measured from the reads: give every
`extract` of the training and evaluation samples `--alignments` and `--reference` (and no
`--gvcf`), and `score` the same reads options; see
[Feature sources](https://winnowr.readthedocs.io/en/latest/user-guide/feature-sources.html).

Without the reads either, `--feature-source vcf` derives what it can from the VCF alone:

```console
winnowr extract --vcf HG001.vcf.gz --feature-source vcf --happy HG001.happy.vcf.gz -o HG001.parquet
winnowr score --vcf sample.vcf.gz --models snp.winnowr.json indel.winnowr.json -o sample.winnowr.vcf.gz
```

To train on only some contigs of a sample, e.g. to evaluate it on the others, `subset` keeps them
and recounts the table's truth FNs on them:

```console
winnowr subset --table HG002.parquet --happy HG002.happy.vcf.gz --contigs chr1 chr2 -o HG002.chr1-2.parquet
```

## VCF requirements

winnowr reads single-sample VCFs from GATK HaplotypeCaller. Six of its features come from
annotations that only HaplotypeCaller's gVCF mode writes: by default from the VCF's raw annotations
or the gVCF it was genotyped from (`--gvcf`), or, for a standard VCF without its gVCF, derived
from the VCF and the reads (`extract --alignments`) or from the VCF alone
(`extract --feature-source vcf`). [Feature sources](https://winnowr.readthedocs.io/en/latest/user-guide/feature-sources.html)
lists the fields each needs.

## Commands

```console
winnowr strlen    # per-read repeat-tract lengths at indel calls (for repeat-length features)
winnowr pileup    # allele read counts by MAPQ at SNP and indel calls (for pileup features)
winnowr extract   # per-call features (and labels, for training) as a Parquet feature table
winnowr subset    # restrict a labelled feature table to some contigs, or away from some regions
winnowr train     # train a model for one variant type; thresholds chosen by cross-validation
winnowr evaluate  # F1 of a model's decisions on a held-out labelled sample
winnowr stratify  # decisions, F1 and P(keep) calibration by benchmark stratum and GIAB stratification
winnowr check-models  # check a set of models before a run, and report what they need
winnowr score     # apply models to a sample and write the re-genotyped, filtered VCF
```

Each command's options are listed by `winnowr <command> --help` and in the
[tool reference](https://winnowr.readthedocs.io/en/latest/tools/overview.html).

## Documentation

The [user guide](https://winnowr.readthedocs.io) covers installation, the quick starts, classes and
labels, training, scoring, repeat-length and pileup features, evaluation, model files and
performance tuning, with a reference page for every command. Its source is in
[`docs/src`](https://github.com/fg-labs/winnowr/tree/main/docs/src).

## Contributing

See [CONTRIBUTING](https://github.com/fg-labs/winnowr/blob/main/docs/CONTRIBUTING.md) and
[DEVELOPING](https://github.com/fg-labs/winnowr/blob/main/docs/DEVELOPING.md).

## License

winnowr is released under the [MIT License](LICENSE). The vendored XGBoost sources
(`xgboost/vendor/xgboost`) are under the Apache License 2.0.
