[![Build](https://github.com/fg-labs/winnowr/actions/workflows/check.yml/badge.svg)](https://github.com/fg-labs/winnowr/actions/workflows/check.yml)
[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/fg-labs/winnowr/blob/main/LICENSE)

# winnowr

Machine-learned genotyping and filtering of germline small-variant calls.

> [!WARNING]
> **Demonstration code only.** winnowr is a research prototype, shared to illustrate an approach to
> machine-learned filtering and genotyping of germline variant calls. It is not supported, has not
> been validated, and is not intended for clinical, diagnostic or production use. No trained models
> are distributed. Use it at your own risk.

> [!NOTE]
> winnowr has been rewritten in Rust, and its outputs and models differ from the Python version's by design: matching them is not a goal. The main changes in the defaults:
>
> - `train` holds out whole 10 kb genomic blocks, not random calls, for early stopping, then refits the final model on every call (`--refit-final`).
> - `score`, `evaluate` and `stratify` also drop indel calls whose P(keep) is below 0.5 (`--keep-fallback indel`), and use a drop threshold of 0.6 for indel calls outside GIAB's confident regions (`--outside-indel-t-drop`); see [How `score` acts](#how-score-acts).
> - `strlen` measures reads with MAPQ 20 or more and five flank bases (`--reads mapq20,flank5`; the Python version measured reads of any MAPQ with two flank bases, which is `--reads legacy`).
> - `extract` labels a call *keep* only when every hap.py record of it is a TP, labels a multi-allelic call from all of its records, and takes a call's gVCF annotations from the site with its alleles rather than the first site at its position.
> - Pileup features (`winnowr pileup`, `train --pileup-features`) are new and off unless asked for.
>
> No option restores all of the Python version's behaviour. Models from the Python version (two files, model format 1 or 2) are not usable: winnowr refuses them, and ships no converter, so train new models with `winnowr train`. Feature tables written by the Python version can still be read.

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

winnowr post-processes single-sample GATK HaplotypeCaller calls. For each PASS call, a
per-variant-type (SNP, indel) XGBoost model predicts one of three classes:

| class | meaning                      | action                            |
|-------|------------------------------|-----------------------------------|
| drop  | the site is really 0/0       | FILTER `WinnowrDrop`              |
| keep  | the call is right            | none                              |
| flip  | right allele, wrong zygosity | switch GT between `0/1` and `1/1` |

Training labels come from hap.py: a TP is *keep*, an allele-match FP is *flip*, and any other FP
is *drop*. Some allele-match FPs cannot be fixed by a flip (e.g. a `0/1` call whose truth is `1/2`
needs another allele). `winnowr train --unfixable-flips drop` trains those as *drop*, and
`--unfixable-flips exclude` leaves them out of the fits. The default, `flip`, keeps hap.py's label.

`winnowr train` chooses each model's thresholds by cross-validation, by default holding out each
sample group (`--sample-groups`) in turn. `--cv-folds chromosomes` instead holds out groups of
whole chromosomes from every sample at once, so every fold fits every sample. The final model
early-stops on calls held out in 10 kb blocks, then is refitted on every call for the rounds early
stopping chose (`--refit-final false` keeps the early-stopped fit).

An optional *promote* model per variant type scores non-PASS calls (that do not overlap a PASS
call): stay filtered, promote to PASS, or promote and flip het<->hom.

Indel models can also use *repeat-length* features: `winnowr strlen` measures the length of the
repeat tract (homopolymer/STR) around each indel call in the reads, and a stutter model turns those
read lengths into a diploid length genotype. Such models can *re-genotype* a call to the length
caller's genotype, adding an ALT allele if needed. `winnowr score --alignments` measures the
repeat lengths itself while scoring, instead of reading `winnowr strlen` tables.

Models can also use *pileup* features: `winnowr pileup` counts the reads supporting each call's
reference and alternate alleles at MAPQ 0, at MAPQ 1-19 and at MAPQ 20 or more. HaplotypeCaller
ignores reads below its minimum MAPQ (1 in DRAGEN mode), so when an aligner places one allele's
reads at a site with MAPQ 0 (e.g. reads ALT-projected onto the primary assembly, or from a
paralogous copy), a call made from the other reads can have the wrong zygosity, and nothing in
the VCF shows it. `winnowr score --alignments` counts them too.

## Highlights

- **One self-contained binary.** XGBoost and its OpenMP runtime are built from source and linked
  statically: nothing to install alongside the release binaries. A source build on a Linux
  distribution whose `libgomp.a` is not position-independent (e.g. RHEL, Amazon Linux) links the
  shared libgomp instead, with a warning, unless built with
  `RUSTFLAGS='-C relocation-model=static'` (see `xgboost/build.rs`).

## Install

From source, with Rust 1.88 or newer, CMake, a C++ compiler (Apple clang with
`brew install libomp` on macOS, or GCC on Linux) and libclang, which builds htslib's bindings
(`apt-get install libclang-dev` on Debian or Ubuntu; Apple clang provides it on macOS). See
[DEVELOPING](docs/DEVELOPING.md) for details.

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

## Commands

```console
winnowr strlen    # per-read repeat-tract lengths at indel calls (for repeat-length features)
winnowr pileup    # allele read counts by MAPQ at SNP and indel calls (for pileup features)
winnowr extract   # per-call features (and labels, for training) as a Parquet feature table
winnowr subset    # restrict a labelled feature table to some contigs, or away from some regions
winnowr train     # train a model for one variant type; thresholds chosen by cross-validation
winnowr evaluate  # F1 of a model's decisions on a held-out labelled sample
winnowr stratify  # decisions, F1 and P(keep) calibration by benchmark stratum and GIAB stratification
winnowr score     # apply models to a sample and write the re-genotyped, filtered VCF
```

`winnowr stratify --stratification <manifest>` also reports each metric within named regions, such as
the [GIAB genome stratifications](https://github.com/usnistgov/giab-stratifications), as hap.py's
`--stratification` does: the manifest has one `NAME<TAB>PATH` BED per line, regions may overlap, and
each truth set is scored inside its own confident regions.

### How `score` acts

`score` takes the action each model's cross-validated thresholds choose, with two more rules for
indel calls:

- **Keep fallback.** An indel call that the thresholds leave alone is dropped when the model's
  P(keep) is below 0.5. Turn it off with `--keep-fallback none`.
- **Outside the confident regions.** An indel call whose REF span is not inside GIAB's GRCh38
  `notinalldifficultregions` is dropped when P(drop) is at least 0.6, in place of the model's
  drop threshold, whether that is lower or higher (so a model that would never drop does drop
  such calls). The keep fallback still applies: a call there that is not dropped, flipped or
  re-genotyped is dropped when P(keep) is below 0.5. Since P(drop) of 0.6 or more means P(keep)
  is at most 0.4, with both defaults an outside indel call that is neither flipped nor
  re-genotyped is dropped exactly when P(keep) is below 0.5. Give that BED with the models' other
  region BEDs, as
  `--regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz`. The BED comes from
  the [GIAB genome stratifications](https://github.com/usnistgov/giab-stratifications) and is the
  same for every sample. Without it, indels are dropped at the model's threshold everywhere, and
  `score` warns. Change the threshold with `--outside-indel-t-drop`, or turn the rule off with
  `--outside-indel-t-drop none`.

`evaluate` and `stratify` take the same options and `--regions`, so their metrics are those of the
calls `score` writes.

### Repeat-length read settings

`winnowr strlen` measures a repeat tract in reads with MAPQ at least 20 that align at least five
bases outside each end of it (`--reads mapq20,flank5`). The settings are written at the top of
the strlen table, and they are kept in the feature table and in the trained model. A model only
scores repeat-length evidence measured with its training tables' settings: `score --strlen`,
`evaluate` and `stratify` stop with an error otherwise, and `score --alignments` measures with the
model's settings. Tables and models made before winnowr recorded the settings were measured with
any MAPQ and two flank bases, and are read that way. To score such a model from `strlen` tables,
measure them with `winnowr strlen --reads legacy`.

### Pileup features

`winnowr pileup` counts, for every SNP and indel record, the reads supporting its reference allele,
an alternate allele, or neither, in three MAPQ bins: 0, 1-19 and 20 or more. Reads of every MAPQ
are counted; unmapped, secondary, supplementary, duplicate and QC-failed reads are not. A SNP's
allele is the read's base at it, counted at base quality 10 or more. An indel's allele is the
read's length change across the indel's repeat tract (as `winnowr strlen` measures it, from reads
aligned on five flank bases outside it), so an indel placed anywhere in the tract counts. When an
indel record also has an alternate allele as long as the reference (e.g. `A>T,AT`), a read with
no length change is told apart by its bases where that allele differs. Both reads of a fragment
whose mates overlap the call are counted. Records of other types (e.g. MNPs) get no counts. The settings (`--settings mapq20,bq10,flank5`) are
written at the top of the table and kept in the feature table (`extract --pileup`).

`winnowr train --pileup-features` adds the counts and fractions of them (`pu_*`, e.g.
`pu_ref_frac`, the reference reads' share of the reference and alternate reads at any MAPQ) to
the model, and records the settings. Such a model needs pileup evidence measured with those
settings: `score` stops with an error without `--pileup` tables or `--alignments`, and `score`,
`evaluate` and `stratify` stop with an error when the settings differ. Models trained without
`--pileup-features` (every model trained before it existed) neither need nor read it.

To evaluate on a sample that also contributes training labels, hold out chromosomes: train with
`winnowr subset --contigs <training half>` of its table (which recounts the truth FNs from the
hap.py VCF on those contigs, so no label from the other half enters training or the threshold
search), then run `winnowr stratify --contigs <held-out half>` with the full tables.

Each command's options are listed by `winnowr <command> --help`. Every command runs on the
number of threads given by `-t`; without `-t`, on `OMP_NUM_THREADS` when it is set (as XGBoost's
own default does), else on 4. A trained model depends on the thread count too, as XGBoost's
histogram sums do: training again with the same tables, options and threads gives the same model.

## Performance tuning

- **Threads.** Give `-t` at most the cores that are free. `score --alignments` measures the
  reads with `-t` threads while the models that do not need them score with another `-t`, so it
  runs up to twice `-t` threads. Training runs thousands of short
  parallel steps per tree, and loses far more from threads that wait for a core than it gains
  from extra threads. Past about 8-16 threads, more threads gain little.
- **How idle threads wait.** At the start of `train`, winnowr measures the other work on the
  machine (the cores busy over 250 ms, or the load average when every core is busy) and sets how
  XGBoost's OpenMP threads wait between parallel steps. If the threads fit on the free cores,
  they keep polling for the next step for a while (fastest); if the machine is busy, they poll
  only briefly; with more than twice as much work as cores, they sleep at once. Polling threads
  on a busy machine slowed training down 10-100 times. The choice is logged at debug level
  (`RUST_LOG=winnowr=debug`) and does not change the model. Setting `OMP_WAIT_POLICY`, or
  `KMP_BLOCKTIME` (macOS) or `GOMP_SPINCOUNT` (Linux), overrides it. On Linux, winnowr can set
  the wait only when GCC's OpenMP runtime is linked statically (see `xgboost/build.rs`).
- **Memory.** Training is bound by memory bandwidth and latency rather than arithmetic, which
  is why it scales poorly with threads. On Linux, winnowr's allocator (mimalloc, which also
  serves XGBoost's C++ allocations) asks for transparent huge pages (`madvise`) for its memory,
  so it gets them under the default THP setting (`madvise`); setting THP to `always`
  (`/sys/kernel/mm/transparent_hugepage/enabled`) gains nothing more. On macOS, only winnowr's
  own (Rust) allocations use mimalloc.

## Contributing

See [CONTRIBUTING](docs/CONTRIBUTING.md) and [DEVELOPING](docs/DEVELOPING.md).

## License

winnowr is released under the [MIT License](LICENSE). The vendored XGBoost sources
(`xgboost/vendor/xgboost`) are under the Apache License 2.0.
