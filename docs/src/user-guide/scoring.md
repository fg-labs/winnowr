# Scoring

`winnowr score` applies models to a sample's VCF and writes it back with the models' decisions.

```console
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  -o sample.winnowr.vcf.gz
```

`--models` takes at most one genotype and one promote model per variant type, all trained on the
same contigs. `score` scores the calls on those contigs. When the models record their reference's
contigs (models trained from tables that record them do), `score` stops with an error if one is
missing from the VCF's header or has another length there.

`--regions` must name every region BED the models were trained with, under the same names. It
also takes the not-difficult regions, `notinalldifficultregions=PATH`, described below. Models with
repeat-length or pileup features need `--strlen` or `--pileup` tables, or `--alignments` and
`--reference` to measure the reads while scoring (see
[Repeat-length features](repeat-length.md) and [Pileup features](pileup.md)).

The models given to one run must share a feature source, which decides where the features of
gVCF-mode annotations come from: models trained on `extract --feature-source vcf` tables need no
`--gvcf`. See [Feature sources](feature-sources.md).

The depth feature (`dp_norm`) is each call's RAW_MQandDP depth (for models of the `vcf` feature
source, see [Feature sources](feature-sources.md)) over the median of that depth across the calls
on the models' contigs, which `score` logs. For a VCF of only part of the genome, e.g. a target
footprint, that median is skewed; `--dp-median` gives the depth to divide by instead (at least 1):
the median depth of the sample's whole-genome calls, as `extract` or `score` logs it, not coverage
from another tool. Normalize as the models' training tables were: nothing records or checks the
depth used, so a footprint VCF scored without `--dp-median` with models trained on whole-genome
tables (or the reverse) is silently on a different depth scale.

## Modes

`--mode` sets which actions genotype models take:

- `genotyper` (the default): drop calls with P(drop) at or above `t_drop`, flip the remaining calls
  with P(flip) at or above `t_flip`, and, for re-genotype models, re-genotype the calls still
  remaining that the repeat-length caller disagrees with when P(re-genotype) is at or above
  `t_regenotype`. Only a `0/1` or `1/1` call can be flipped, and only a call with a diploid genotype
  (not `./.` or haploid) whose length-caller genotype differs from its own, and can be written, can
  be re-genotyped. A call that cannot take an action goes on to the next: e.g. a `1/2` call with
  P(flip) at or above `t_flip` is not flipped, but may be re-genotyped, else is left alone or, for
  an indel, dropped by the keep fallback when P(keep) is below 0.5. That order is the threshold
  rule; the experimental `--expected-f1` (below) acts on probabilities above 0.5 instead.
- `veto`: drop calls whose P(drop) + P(flip) (+ P(re-genotype)) is at or above `t_drop`: the call
  as made is wrong. Never flip or re-genotype.

Each mode uses the thresholds cross-validation chose for it. Promote models act the same way in
both modes.

## How `score` acts on indel calls

`score` takes the action each model's thresholds choose, with two more rules for indel calls:

- **Keep fallback.** An indel call that the thresholds leave alone is dropped when the model's
  P(keep) is below 0.5. `--keep-fallback` lists the variant types this applies to (default
  `indel`); `--keep-fallback none` turns it off.
- **Outside the not-difficult regions.** An indel call whose REF span is not inside GIAB's GRCh38
  `notinalldifficultregions` is dropped when P(drop) is at least 0.6 (in veto mode, P(drop) +
  P(flip) (+ P(re-genotype)), the quantity the veto threshold is compared with), in place of the
  model's drop threshold, whether that is lower or higher (so a model that would never drop does
  drop such calls). The keep fallback still applies there. In either mode, reaching 0.6 means
  P(keep) is at most 0.4, so with both defaults an outside indel call that is neither flipped nor
  re-genotyped is dropped exactly when P(keep) is below 0.5. A call re-genotyped to `0/0` is dropped
  too, whatever its P(keep): re-genotyping to `0/0` is a drop. Give the BED as
  `--regions notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz`. It comes from the
  [GIAB genome stratifications](https://github.com/usnistgov/giab-stratifications) and is the same
  for every sample. Without it, indels are dropped at the model's threshold everywhere, and `score`
  warns. `--outside-indel-t-drop` changes the threshold; `--outside-indel-t-drop none` turns the
  rule off.

Two more options are experimental:

- `--expected-f1 <snp|indel>`: the genotype models of these variant types take the action of
  greatest expected F1 gain in place of the keep fallback. In genotyper mode this replaces their
  thresholds: drop an indel call outside the not-difficult regions when P(drop) reaches
  `--outside-indel-t-drop`, else flip when P(flip) is above 0.5, else re-genotype when
  P(re-genotype) is above 0.5, else drop when P(keep) is below 0.5. In veto mode `t_drop` still
  applies: a call is dropped when P(drop) + P(flip) (+ P(re-genotype)) reaches `t_drop`, or when
  P(keep) is below 0.5.
- `--abstain-actions <flip|regenotype|promote>`: withhold these actions from calls outside the
  not-difficult regions.

`evaluate` and `stratify` take the same options, and the not-difficult regions as
`--regions notinalldifficultregions=PATH`, so their metrics are those of the calls `score` writes.
They read only that BED from `--regions` (the models' region features come from the feature
table), and warn that other names are ignored.

## The output VCF

`score` writes every record of the input VCF, including any at POS 0, which are not read or scored
(`extract` and `score` warn about them). The output format follows the `-o` suffix, in any case:
`.vcf.gz` (or `.vcf.bgz`) is bgzipped and tabix-indexed, `.bcf` is CSI-indexed BCF, and anything
else is plain-text VCF.

Genotype models score PASS calls of their variant type; promote models score the non-PASS calls of
their variant type that do not overlap a PASS call. Other records are written unchanged. A scored
record gets:

| field                       | written to                       | holds                                    |
|-----------------------------|----------------------------------|------------------------------------------|
| INFO `WINNOWR_P`            | calls a genotype model scored    | P(drop), P(keep), P(flip)                |
| INFO `WINNOWR_REGENOTYPE_P` | calls a re-genotype model scored | P(re-genotype)                           |
| INFO `WINNOWR_PROMOTE_P`    | calls a promote model scored     | P(stay), P(promote), P(promote and flip) |
| FILTER `WinnowrDrop`        | dropped calls                    | replaces PASS                            |
| FILTER `PASS`               | promoted calls                   | replaces the FILTER                      |
| INFO `WINNOWR_OFILTER`      | promoted calls                   | the FILTER before promotion              |
| INFO `WINNOWR_FLIP`         | flipped calls                    | a flag                                   |
| INFO `WINNOWR_REGENOTYPE`   | re-genotyped calls               | a flag                                   |
| FORMAT `WINNOWR_OGT`        | flipped or re-genotyped calls    | the genotype before the change           |

A flipped call has its GT switched between `0/1` and `1/1` (unphased), and its GQ set to the
model's confidence in the new genotype, phred-scaled (`-10 log10(1 - p)`) and capped at 99. INFO
AC/AF and MLEAC/MLEAF, when present, are recomputed from the new genotype, and the
physical-phasing FORMAT fields PGT, PID and PS, when present, are set to missing. PL is left as the
caller wrote it.

A re-genotyped call gets the repeat-length caller's genotype the same way, adding an ALT allele
when needed; fields with one value per allele or genotype (Number=A, R or G) get missing values for
the new allele.

## Threads

`score` runs on `-t` threads, or up to twice that with `--alignments`; see
[Performance tuning](performance.md#threads).
