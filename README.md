# winnowr

[![CI](https://github.com/fg-labs/winnowr/actions/workflows/python_package.yml/badge.svg?branch=main)](https://github.com/fg-labs/winnowr/actions/workflows/python_package.yml?query=branch%3Amain)
[![Python Versions](https://img.shields.io/badge/python-3.12_|_3.13_|_3.14-blue)](https://github.com/fg-labs/winnowr)
[![MyPy Checked](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

Machine-learned genotyping and filtering of germline small-variant calls.

> [!WARNING]
> **Demonstration code only.** winnowr is a research prototype, shared to illustrate an approach to
> machine-learned filtering and genotyping of germline variant calls. It is not supported, has not
> been validated, and is not intended for clinical, diagnostic or production use. No trained models
> are distributed. Use it at your own risk.

`winnowr` post-processes single-sample GATK HaplotypeCaller calls. For each PASS call a
per-variant-type (SNP, indel) XGBoost model predicts one of three classes:

| class | meaning                        | action                          |
|-------|--------------------------------|---------------------------------|
| drop  | the site is really 0/0         | FILTER `WinnowrDrop`            |
| keep  | the call is right              | none                            |
| flip  | right allele, wrong zygosity   | switch GT between `0/1` and `1/1` |

An optional *promote* model per variant type scores non-PASS calls (that do not overlap a PASS
call) with three classes: stay filtered, promote to PASS, or promote and flip het<->hom. A call is
promoted when `P(promote) + P(promote and flip) >= t_promote`, and flipped when also
`P(promote and flip) >= t_flip` and it exceeds `P(promote)`.

Genotype models are trained on hap.py labels from GIAB samples; hap.py must be run with
`--engine vcfeval --pass-only`. With `--pass-only` hap.py does not evaluate non-PASS calls, so
promote models are labelled by exact matching (after left-alignment)
against the truth VCF inside the confident regions, counting only truth records hap.py reports as
FNs. Decision thresholds are chosen by leave-one-sample-out cross-validation over the training
samples (or leave-one-group-out: related samples such as a trio, grouped with
`train --sample-groups`, are held out together), maximising the summed F1 gain computed with vcfeval
rescoring arithmetic (a genotype mismatch counts as one FP plus one FN). Taking no action is always
a candidate, and ties go to the more conservative thresholds. A `veto` mode only drops calls
(`P(drop) + P(flip) >= t_drop`).

Indel genotype models can also use *repeat-length* features: `winnowr strlen` measures, in the
reads, the length of the repeat tract (homopolymer/STR) around each indel call, and a stutter model
fitted on the training samples turns those read lengths into a diploid length genotype. With
`train --regenotype` the indel model gets a fourth class, *re-genotype*: replace the call's genotype
with the length caller's (e.g. a `0/1` call whose truth is `1/2`, or a call with the wrong repeat
length), adding an ALT allele if needed.

## Installation

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then:

```console
uv sync --locked
uv run winnowr --help
```

## Usage

```console
winnowr strlen    # per-read repeat-tract lengths at indel calls (for repeat-length features)
winnowr extract   # per-call features (and labels, for training) as a Parquet feature table
winnowr train     # train a model for one variant type; thresholds chosen by cross-validation
winnowr evaluate  # F1 of a model's decisions on a held-out labelled sample
winnowr score     # apply models to a sample and write the re-genotyped, filtered VCF
```

Each command's options are listed by `winnowr <command> --help`. VCFs and gVCFs must be bgzipped
(or BCF) and indexed. `extract` reads chr1-chr22 by default (`--contigs`); `score` scores the
contigs the models were trained on and copies every other record unchanged. Contig names must
agree across the VCF, gVCF, BEDs, hap.py output and truth VCF (e.g. `chr1` vs `1` is an error).

`extract` and `score` read four raw annotations of HaplotypeCaller's gVCF mode (`RAW_MQandDP`,
`QUALapprox`, `VarDP`, `AS_RAW_MQ`; see [Model inputs](#model-inputs)). A VCF genotyped from the
gVCF (e.g. by `GenotypeGVCFs`) lacks them, so pass the gVCF too (`--gvcf`): its variant sites are
joined to the calls by position. A VCF that keeps them needs no gVCF: when its header declares all
four, they are read from each call, and `--gvcf` is not read. Such a VCF can be extracted from
HaplotypeCaller's own (un-reblocked) gVCF, keeping `RAW_MQandDP`, adding `QUALapprox` and `VarDP`
as `ReblockGVCF --do-qual-approx` computes them for the called alleles (PL[0] after subsetting the
PLs to them and `<NON_REF>`, and their summed AD), and cutting `AS_RAW_MQ` to one value per allele.
Without either source, `extract` and `score` fail.

### Scoring a sample

Given trained models (see below), score a sample's VCF and the gVCF it was genotyped from (or a
VCF that keeps the raw annotations, without `--gvcf`), with the region BEDs under the names the
models were trained with:

```console
winnowr score --vcf S.vcf.gz --gvcf S.g.vcf.gz \
  --models snp.winnowr.json indel.winnowr.json snp-promote.winnowr.json indel-promote.winnowr.json \
  --regions lowmappabilityall=lowmappabilityall.bed.gz segdups=segdups.bed.gz \
            collapsed_duplication_FP_regions=collapsed_duplication_FP_regions.bed.gz \
            AllTandemRepeatsandHomopolymers_slop5=AllTandemRepeatsandHomopolymers_slop5.bed.gz \
  --strlen S.*.strlen.tsv.gz \
  --output S.winnowr.vcf.gz
```

`--strlen` takes the `winnowr strlen` tables of the sample (step 2 below) and is needed only by
models with repeat-length features. `score --mode VETO` only drops calls, using the models' veto
thresholds; the default `--mode GENOTYPER` drops, flips and re-genotypes.

## Train your own models

winnowr ships no trained models. This walkthrough trains them on the public Genome in a Bottle
(GIAB) samples, holding out HG002 for evaluation. GIAB benchmark VCFs and BEDs, and the genome
stratification BEDs used as region features, are on the GIAB FTP site
(<https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/>); sequencing data for the
GIAB samples is available from several public sources. Use the same sequencing chemistry,
coverage, aligner, reference and caller for training as for the samples you will score: models do
not transfer across them.

The training samples are HG001 and the HG005-HG007 trio; HG002 is held out, and so are its parents
HG003 and HG004, which would leak into its evaluation if trained on.

### 1. Calls and hap.py labels, per sample

For each sample `$s`, align the reads to the reference (`ref.fa`), call variants with GATK
HaplotypeCaller in gVCF mode with `--dragen-mode`, and genotype the gVCF (`GenotypeGVCFs`). Keep
both `$s.vcf.gz` and `$s.g.vcf.gz`: winnowr reads annotations (`RAW_MQandDP`, `QUALapprox`, `VarDP`,
`AS_RAW_MQ`) from the gVCF that the final VCF lacks (or extract a VCF that keeps them, see
above, and drop `--gvcf` below). Then compare the calls to the GIAB benchmark
with hap.py, using the vcfeval engine and `--pass-only`:

```console
hap.py $s.truth.vcf.gz $s.vcf.gz -f $s.truth.bed -r ref.fa -o $s.happy \
  --engine vcfeval --pass-only
```

### 2. Repeat-tract lengths (for repeat-length features)

`winnowr strlen` is single-threaded, so run it in shards, e.g. one per contig, and keep every
shard's table:

```console
for c in chr{1..22}; do
  winnowr strlen --vcf $s.vcf.gz --alignments $s.cram --reference ref.fa \
    --regions $c --output $s.$c.strlen.tsv.gz
done
```

### 3. Feature tables

```console
winnowr extract --vcf $s.vcf.gz --gvcf $s.g.vcf.gz --happy $s.happy.vcf.gz \
  --truth-vcf $s.truth.vcf.gz --truth-bed $s.truth.bed --reference ref.fa \
  --strlen $s.chr*.strlen.tsv.gz \
  --regions lowmappabilityall=lowmappabilityall.bed.gz segdups=segdups.bed.gz \
            collapsed_duplication_FP_regions=collapsed_duplication_FP_regions.bed.gz \
            AllTandemRepeatsandHomopolymers_slop5=AllTandemRepeatsandHomopolymers_slop5.bed.gz \
  --output $s.parquet
```

`--truth-vcf`, `--truth-bed` and `--reference` add the labels of promote models; `--strlen` adds
the repeat-length evidence. Region names are free-form, but `score` must be given the same names.

### 4. Train

Train one genotype model per variant type, and optionally a promote model per variant type. The
HG005-HG007 trio is held out together in cross-validation (`--sample-groups`), so related samples
never leak into each other's folds; every other sample is held out alone.

```console
tables="HG001.parquet HG005.parquet HG006.parquet HG007.parquet"
trio="HG005,HG006,HG007"
winnowr train --tables $tables --sample-groups $trio --variant-type SNP --output snp
winnowr train --tables $tables --sample-groups $trio --variant-type INDEL --regenotype \
  --output indel
winnowr train --tables $tables --sample-groups $trio --variant-type SNP --task PROMOTE \
  --output snp-promote
winnowr train --tables $tables --sample-groups $trio --variant-type INDEL --task PROMOTE \
  --output indel-promote
```

Each writes `<output>.xgb.json` and `<output>.winnowr.json` (see [Model files](#model-files)).
`--regenotype` adds the re-genotype class and the repeat-length features (`--repeat-length` adds
only the features); `--no-neighbourhood` leaves out the neighbourhood features.

### 5. Evaluate on the held-out sample

```console
winnowr evaluate --models snp.winnowr.json snp-promote.winnowr.json \
  --table HG002.parquet --output HG002.snp.tsv
winnowr evaluate --models indel.winnowr.json indel-promote.winnowr.json \
  --table HG002.parquet --output HG002.indel.tsv
```

`evaluate` writes one row per policy (a TSV with a header). The policies are `baseline` (no
action), `veto` and `genotyper` at the model's CV-chosen thresholds, `flip_only` (the genotyper's
flips alone), `genotyper_no_regenotype` for re-genotype models, and, with a promote model,
`promote`, `promote_no_flip` and `genotyper+promote`. `--tuned` adds `veto_tuned`,
`genotyper_tuned` and `promote_tuned`, with thresholds tuned on the evaluated sample itself; they
peek at its labels, so they are optimistic and only bound what better thresholds could do. The
columns are the PASS-call thresholds and actions (`t_drop`, `t_flip`, `t_regenotype`, `dropped`,
`dropped_tp`, `flipped`, `flips_fixed`, `flips_broke_tp`, `regenotyped`, `regenotypes_fixed`,
`regenotypes_broke_tp`), the promote thresholds and actions (`t_promote`, `t_promote_flip`,
`promoted`, `promoted_correct`, `promoted_flipped`), and the outcome (`tp`, `fp`, `fn`, `f1`, and
`f1_gain` versus the baseline). Evaluating a sample the models were trained on logs a warning: its
metrics are in-sample.

## Model inputs

Per call, from the VCF: QUAL, INFO/DP, MQRankSum, ReadPosRankSum, BaseQRankSum, GQ, allele balance
(AD), GATK's StrandOddsRatio (SOR) computed from FORMAT/SB, het flag, and for indels the length
change and DRAGstr period/repeat count. From the raw annotations (in the VCF, or else the gVCF): MQ =
sqrt(RAW_MQ/DP) and QD = QUALapprox/VarDP, the alternate share of AS_RAW_MQ, the number of alternate
alleles, and depth normalised to the sample median. Region flags from user-named BEDs (set when the
REF span overlaps the region, so a deletion counts through its deleted bases). Neighbourhood
features over all calls (any FILTER): counts of other calls, hets, low-AB hets and non-PASS calls
within ±500 bp and ±2 kb, mean AB of neighbouring hets, and distance to the nearest call
(`train --no-neighbourhood` drops them). Promote models add ExcessHet, PL, AD (ref and total alt)
and the number of alternate alleles in the record.

Repeat-length features (indel genotype models trained with `--repeat-length` or `--regenotype`;
`score` then needs `--strlen` tables too): `winnowr strlen` finds the repeat tract of each indel
record (the maximal run, in the reference, of the minimal period of the inserted/deleted sequence)
and histograms the observed minus reference tract length over primary, non-duplicate reads of any MAPQ
that span the tract with two aligned bases each side. From the histogram: the top three lengths and
their read fractions, the number of distinct lengths, entropy and spread, the fractions of reads at
the reference and called lengths, the top lengths' distance from the called allele in repeat units,
tract length, period and units, spanning-read fraction, MAPQ and tract base quality. A stutter model
(per-read length-error rates per period class and tract length bin, fitted on the training samples'
truth-homozygous calls and stored in the sidecar) drives a diploid length caller, whose genotype and
log-likelihood margins versus the call, its het<->hom flip and the runner-up are features too.
`extract` and `score` log how many indel calls have strlen evidence, warn about contigs whose
indel calls have none (a missing shard), and fail when no indel call has any (tables from another
VCF).

Missing values (e.g. no gVCF record at a site, or a call without RAW_MQandDP) are left to
XGBoost's missing-value handling.

## Outputs of `score`

- FILTER `WinnowrDrop` on dropped calls (replaces PASS).
- INFO `WINNOWR_P` = `p_drop,p_keep,p_flip` on every PASS call scored by a genotype model.
- INFO `WINNOWR_PROMOTE_P` = `p_stay,p_promote,p_promote_flip` on every non-PASS call scored by a
  promote model; promoted calls get FILTER PASS with the original filter in INFO
  `WINNOWR_OFILTER`.
- On flipped calls: GT switched between `0/1` and `1/1` (unphased), INFO flag `WINNOWR_FLIP`,
  FORMAT `WINNOWR_OGT` with the original genotype, and GQ set to `-10 log10(1 - p_flip)` (capped
  at 99). INFO AC/AF and MLEAC/MLEAF, when present, are recomputed from the new genotype, and the
  physical-phasing FORMAT fields PGT, PID and PS are set to missing. PL is not changed: it still
  reflects the caller's likelihoods.
- Re-genotype models add INFO `WINNOWR_REGENOTYPE_P` (the re-genotype probability). A
  re-genotyped call gets the length caller's genotype, INFO flag `WINNOWR_REGENOTYPE`, FORMAT
  `WINNOWR_OGT` and GQ `-10 log10(1 - p_regenotype)` (AC/AF and PGT/PID/PS as for a flip). Each
  new allele is the repeat tract resized in its phase: an ALT with exactly that sequence is reused,
  otherwise it is added as a new ALT (with the REF extended over the tract if needed) and
  Number=A/R/G fields get missing values for it. A re-genotype to `0/0` is a drop. Calls whose new
  allele cannot be built (e.g. the record starts inside the tract) are never re-genotyped, in
  training and evaluation as in `score`.
- An output ending in `.vcf.gz` is bgzipped and tabix-indexed, `.bcf` is written as indexed BCF,
  anything else as plain VCF.

## Caveats

- The features and defaults were developed on NovaSeq 30x PCR-free data aligned to GRCh38 and
  called with GATK HaplotypeCaller `--dragen-mode` in gVCF mode. Models trained on one chemistry,
  coverage, reference or caller are out of distribution on another.
- Gains depend on the sequencing data, the caller and the training samples. Evaluate models with
  `winnowr evaluate` on a held-out sample of your own data before relying on them.
- Only biallelic `0/1` <-> `1/1` flips are made; multi-allelic genotypes are never flipped (a
  re-genotype model can still re-genotype them).
- Re-genotype labels count the length caller as right when its alleles' length changes equal the
  truth's at the same position. `score` writes the tract resized in its phase, so this holds when
  the truth allele is a pure resize of the same tract.
- Each cross-validation fold refits the stutter model on its training samples, so a held-out
  sample's length-call features and re-genotype labels never depend on its own truth; the final
  model's stutter model is fitted on all training samples.
- `winnowr strlen` is pure Python (about 2 ms per indel record on a CRAM); shard it by region.
- Genotype labels come from hap.py (`--engine vcfeval --pass-only`) and are joined to calls by
  position and variant type. `extract` logs how many hap.py TP/FP records labelled a PASS call,
  warns when hap.py scored non-PASS calls (no `--pass-only`) or some records match no call, and
  fails when most match no call (hap.py was run on another VCF).
- Promote labels use exact allele matching, which misses some indel representations that hap.py
  would match, so indel promotion gains are slightly understated. Promote models see few positive
  examples per sample (truth variants filtered by the caller), so their thresholds are noisier than
  the genotype models'.

## Model files

`train --output PREFIX` writes `PREFIX.xgb.json`, the XGBoost booster, and `PREFIX.winnowr.json`,
the sidecar that `evaluate` and `score` take (the booster is read from beside it). The sidecar
records the variant type and task, the ordered feature names, the region names, the CV-chosen
thresholds, the training samples and contigs, the training parameters, the stutter model of a
repeat-length model, and for compatibility checks:

- `format_version`: the sidecar format (schema) version. This winnowr writes version 2 and reads
  versions 1 and 2; a newer version is an error.
- `winnowr_version` and `xgboost_version`: the versions that trained the model. `score` and
  `evaluate` warn when the model was trained by another winnowr version.
- `reference_contigs`: the length of each training contig in the reference the training calls
  were made on (from the VCF header, recorded in each feature table by `extract`). `score` fails
  when the VCF header lacks one of these contigs or gives it another length, and `evaluate` when the
  feature table records another length. Format 1 sidecars do not record them: they load with a
  warning, and the reference is not checked.

## Interface and stability

The supported interface of winnowr is its command line (`winnowr extract`, `strlen`, `train`,
`evaluate` and `score`, with the flags shown by `--help`) and the files it reads and writes: the
feature tables, the strlen tables, the model files (`<prefix>.xgb.json` and the
`<prefix>.winnowr.json` sidecar) and the scored VCF. The Python modules are internal: they may
change between releases without notice, so do not import them.

## Development

See the [contributing guide](./CONTRIBUTING.md).
