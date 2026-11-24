# check-models

Checks a set of models before a run, without scoring anything, and reports what they need. It
reads each model's information but never its booster (the XGBoost trees), so it is fast even for
large models, and it runs the model-set checks `score` runs at startup, and more, reporting every
problem at once:

- the models can be scored together: at most one model per task and variant type, trained on
  the same contigs and reference, on features from one feature source (`gvcf`, `vcf`, or
  `alignments` with its read settings), and needing the repeat-length and pileup evidence
  measured one way;
- there is a BED for every region a model uses, and no `--regions` name twice: the `--regions`
  BED of the name, else one found beside the models as `regions/NAME.bed.gz`, `.bed.bgz` or
  `.bed` (as `score` finds them; see [Scoring](../user-guide/scoring.md)). Each BED a run uses
  (the features', and the not-difficult regions with an indel genotype model) is read, must have
  intervals on the models' contigs, and is checked against the digest the models record of it: a
  different BED is an error (a warning with `--allow-region-mismatch`, as in `score`, which does
  not cover models that record different digests of one region). Each
  other BED given is read too. A BED that `score` does not read (one no
  model uses, or the not-difficult regions without an indel genotype model) only warns: `score`
  still reads the not-difficult regions with `--abstain-actions`, which `check-models` is not
  given. While a model cannot be read, every BED counts as used;
- the models' reference is that of `--reference` (a FASTA with its `.fai`, or the `.fai`
  itself; only the index is read) and of `--vcf` (only its header is read, so it need not be
  indexed or compressed, though `score` needs it bgzipped and indexed);
- the reads were aligned as the models' training reads were: the header of `--alignments` (a
  BAM or CRAM; only its header is read, so no reference is needed), with `--alignment-tags` over
  it, or `--alignment-tags` alone, which is compared on the tags given only (see
  [Alignment provenance](../user-guide/alignment-provenance.md)). With `--alignment-tags none`,
  the header is not read, `inputs.alignment` is `null` and `inputs.alignment_from` is `tags`. An
  aligner read without a profile is only logged, not a problem.
  These problems are warnings, and errors with `--strict-provenance`, as in `score`.

It does not check that a booster loads, has the model's features or its classes (`score` checks
that when it loads the boosters, after the model-set checks), nor the sample's evidence (the VCF's fields, the reads, or the
`--strlen` and `--pileup` tables): the summary says what the models need instead.

Errors make it fail (a non-zero exit status); warnings, such as a model trained by another
winnowr version, do not.

`score` runs the model-set checks at startup, before it reads any calls, and reports their
problems together: one model per task and variant type, the same contigs and reference, one
feature source and evidence settings, the region names (a BED for every region a model uses, and
no name twice), the VCF's contigs against the models' reference, and the reads' alignment
provenance against the models'. It differs in what comes
before and after them: it stops at the first model whose information it cannot read and at an
unreadable VCF before the checks. After the checks, it loads the boosters (stopping at the first
that does not load) and reads the required `--strlen` and `--pileup` tables, and then reads all
planned region BEDs and reports their problems together, including every unreadable required BED
and every required BED with no interval on the models' contigs. The BEDs come after the evidence
tables so that one producer can feed an evidence stream and then a BED stream. `model_unreadable`,
`input_unreadable`, `reference_not_checked`, `not_difficult_missing`,
`alignment_provenance_unknown` and `alignment_header_mixed` are `check-models`' own (`score` logs
the last as a warning when it reads the header),
and so is reading every BED given.

## Usage

```console
winnowr check-models --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --reference GRCh38.fa
# A JSON summary, for a pipeline to read
winnowr check-models --models snp.winnowr.json indel.winnowr.json --vcf sample.vcf.gz --json
# How the reads to be scored were aligned, from their header
winnowr check-models --models snp.winnowr.json indel.winnowr.json --alignments sample.cram
```

Without `--json`, it prints a line per model (its task and variant type, feature source, what it
needs and the regions it uses) on standard output, and each problem on standard error as
`error: [check] message` or `warning: [check] message`.

## Checks

Each problem has a stable check id. New ids may be added; a caller should ignore ids it does not
know.

| check | severity | what |
|-------|----------|------|
| `model_unreadable` | error | a model file cannot be read: its name, its JSON, its fields, or its information's own checks |
| `format_version` | error | a model file is of a format version this winnowr does not read |
| `duplicate_model` | error | a model file is given twice |
| `model_kinds` | error | more than one model of a task and variant type |
| `contigs_differ` | error | the models were trained on different contigs |
| `reference_differs` | error | the models record different lengths of a contig |
| `reference_mismatch` | error | a model's contigs are missing from, or of other lengths in, `--reference` or `--vcf` |
| `feature_source_differs` | error | the models' features come from different feature sources (read settings included) |
| `strlen_settings_differ` | error | the models need repeat-length evidence measured with different settings |
| `pileup_settings_differ` | error | the models need pileup evidence measured with different settings |
| `region_missing` | error | a region a model uses has no `--regions` BED and none beside the models; the message names the directories looked in |
| `region_repeated` | error | a `--regions` name is given twice |
| `region_unreadable` | error, or warning | a region BED cannot be read; a warning for a BED `score` does not read |
| `region_contigs` | error, or warning | a region BED has no interval on the models' contigs; a warning for a BED `score` does not read |
| `input_unreadable` | error | `--reference`, `--vcf` or `--alignments` cannot be read, or `--alignment-tags` cannot be validated |
| `pileup_alignment_passes` | warning | a model of the `alignments` source counts its pileup with another base quality or flank than its alignment settings, so `score` reads the reads twice |
| `region_unused` | warning | a `--regions` BED no model uses (other than the not-difficult regions) |
| `region_ambiguous` | error | several BEDs of a region beside the models (e.g. `.bed.gz` and `.bed`) with different intervals |
| `region_checksum` | error, or warning with `--allow-region-mismatch` | a region BED (given or found) whose digest is not the one the models record |
| `region_digests_differ` | error | the models record different digests of a region BED |
| `region_unverified` | warning | a region BED a model uses whose digest no model records, so it is used unchecked |
| `policy_region_unrecorded` | warning | an indel genotype model, and no model records the policy digest of the not-difficult regions (`train --policy-regions`), even if their feature digest is checked |
| `not_difficult_missing` | warning | an indel genotype model, and no policy-only `notinalldifficultregions` BED (neither given nor bundled); if a model requires it as a feature, `region_missing` is an error instead |
| `other_winnowr_version` | warning | a model was trained by another winnowr version |
| `no_reference_contigs` | warning | a model records no reference contigs, so its reference cannot be checked |
| `reference_not_checked` | warning | neither `--reference` nor `--vcf` was given |
| `alignment_provenance_differs` | warning, or error with `--strict-provenance` | the models record different alignment provenance |
| `alignment_provenance_mismatch` | warning, or error with `--strict-provenance` | the models' alignment provenance differs from the reads' (`--alignments` or `--alignment-tags`) on a compared tag (not metadata such as `version`, `source`, `options-inference` or the internal merge bookkeeping `winnowr-merged-complete`; only the tags given, for `--alignment-tags` alone); the problem lists the `differences` |
| `alignment_provenance_indeterminate` | warning, or error with `--strict-provenance` | both the models and reads record `mixed` on a compared tag; equal markers do not establish agreement because they do not retain the underlying values |
| `alignment_provenance_incomplete` | warning, or error with `--strict-provenance` | a model or the reads carry unresolved alignment-option inference, including when both carry the same marker; reads require intended settings declared with `--alignment-tags` and `options-inference=explicit`, while incomplete models must be regenerated from declared training provenance |
| `alignment_provenance_unknown` | warning | a model records no alignment provenance (one problem per model) |
| `alignment_provenance_not_checked` | warning, or error with `--strict-provenance` | models record alignment provenance, and the reads' is not known (the message says why: neither `--alignments` nor `--alignment-tags`, a header without a program, or `--alignment-tags none`); not raised when the `--alignments` header cannot be read (`input_unreadable`) |
| `alignment_header_mixed` | warning | the header of `--alignments` has aligner records that normalize differently |

## JSON summary

With `--json`, the summary is printed on standard output as one JSON object, also when the
command fails (read the exit status, or `ok`). Its schema is versioned by `schema_version`, now
1. The version changes only when a key is removed or renamed, or changes meaning: new keys, check
ids and values do not change it, and a caller should ignore those it does not know.

| key | what it holds |
|-----|---------------|
| `schema_version` | 1 |
| `winnowr_version` | the winnowr that checked the models |
| `ok` | `true` if no problem is an error |
| `booster_checked` | always `false`: the boosters are not read |
| `models` | each model, in the order given (below) |
| `set` | what the models share (below); a value is `null` where they disagree |
| `inputs` | `reference`, `vcf` and `alignments` as given (or `null`); `regions`: each BED given (`--regions`), in order, with its `name`, `path`, the models that use it (`used_by`), and, for a BED a run uses, its `digest` (an object of `sha256`, `intervals` and `bases`, as in `models[].region_digests`) and whether it is the models' (`verified`: `true`, `false`, or `null` when no model records one; both `null` for a BED a run does not use); `bundled_regions`: each BED found beside the models and read (as no `--regions` BED of its name was given), with the same keys; `alignment`: the reads' alignment provenance (or `null`), and `alignment_from`: `header`, `tags`, `header+tags` (or `null`) |
| `problems` | every problem: `check`, `severity` (`error` or `warning`), `model` (its path, or `null` for a problem of the set) and `message`; only an `alignment_provenance_mismatch` also has `differences`: each `tag`, and its value in the `models`' and the `reads`' provenance (`null` where it is not set) |

Each entry of `models` has its `path` and `readable`. A model that cannot be read has only
those two keys; a readable one also has:

| key | what it holds |
|-----|---------------|
| `format_version` | the model format version |
| `task`, `variant_type` | `genotype` or `promote`; `snp` or `indel` |
| `regenotype` | whether it has the re-genotype class |
| `n_features` | the number of features |
| `contigs` | the contigs its training calls came from |
| `reference_contigs` | those contigs' lengths in its reference (`null` if it records none) |
| `region_names` | the region sets its features use |
| `region_digests` | the digest of each of those region BEDs (`sha256`, `intervals`, `bases`), by name: `{}` if it records none |
| `policy_region_digests` | the digest of each region BED of the policy it records (`train --policy-regions`): `{}` if none |
| `training_samples` | the training samples |
| `feature_source` | `gvcf`, `vcf`, or `alignments:` and the read settings (e.g. `alignments:mapq20,bq10,flank5`) |
| `alignment_settings` | the read settings of the `alignments` source, else `null` |
| `strlen_settings` | the read settings its repeat-length evidence must be measured with (the legacy `mapq0,flank2` for a model that records none), or `null` |
| `pileup_settings` | the settings its pileup evidence must be measured with, or `null` |
| `needs` | what scoring with it needs: `strlen` and `pileup` (tables, or `--alignments`), `alignments` (the reads, for the `alignments` source) and `gvcf` (the raw annotations, in the VCF or a gVCF, for the `gvcf` source) |
| `trained_with` | the `winnowr` and `xgboost` versions that trained it |
| `alignment` | how its training reads were aligned, as an object of tags, or `null` if it records none |

`set` has `contigs` (the first model's), `reference_contigs` (the models', merged: `null` if any
model records none, or two models record different lengths of a contig), `region_names`
(any model's), `feature_source`, `alignment_settings`, `strlen_settings`, `pileup_settings`,
`needs` (what any model needs) and `not_difficult_regions`: their `name`, whether an indel
genotype model `used` them, whether `--regions` `given` them, the digest the models record of them
as a feature and, when `used`, for the policy (`recorded`, `null` if none does or they differ),
and the BED used (`path`, given or found beside
the models, or `null`); and `alignment`, the models'
alignment provenance (`null` if a model records none, or two record different provenance).

## Options

```text
{{#include ../../_generated/cli/check-models.txt}}
```
