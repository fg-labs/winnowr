# score

Applies models to a sample's calls and writes the VCF with drops, promotions, flips and
re-genotypes. See [Scoring](../user-guide/scoring.md). The models' feature source decides
whether `--gvcf` is read, and whether `--alignments` is needed (see
[Feature sources](../user-guide/feature-sources.md)).

Without `--regions`, the region BEDs are found beside the models (`regions/NAME.bed.gz`,
`.bed.bgz` or `.bed`), and each BED used is checked against its recorded digest when available.
Feature-region BEDs without an applicable recorded digest are used unchecked with a warning.
Policy-only BEDs without a policy digest remain unchecked without that warning: see
[Scoring](../user-guide/scoring.md).

```console
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz \
  --models models/snp.winnowr.json models/indel.winnowr.json -o sample.winnowr.vcf.gz
```

Models that record how their training reads were aligned are compared with how the reads were
aligned (the header of `--alignments` and the `--strlen` and `--pileup` tables, with
`--alignment-tags` applied over each, which must then agree): a difference is a warning, and with
`--strict-provenance` an error.
See [Alignment provenance](../user-guide/alignment-provenance.md).

Regular evidence files are inspected before their rows are loaded, and each loaded file is checked against that effective provenance snapshot. Required FIFO or process-substitution inputs are opened only for their row read; their declarations contribute to the final model check before any calls are scored. An unknown-provenance error waits for a required stream that can supply it, but unrelated model checks still run early. Explicitly supplied regular sources can supply provenance even when no model needs their rows. If every model is legacy and records no provenance, otherwise unused sources are not inspected and required evidence skips provenance parsing and agreement while still validating its settings, column headers and rows.

Incomplete alignment-option inference produces a warning, or an error with `--strict-provenance`, on either the models or the scoring reads, even when both record the same incomplete marker. Supply the intended read settings with `--alignment-tags` and `options-inference=explicit` to assert them; this is a user declaration, not recovery of the original arguments. Models carrying incomplete provenance must be regenerated from explicitly declared training provenance. Unrelated tag overrides do not resolve uncertainty. A comparison with no comparable tags is reported as such, not as agreement, and does not itself introduce a strict-mode error.

`-t` is the total thread count, `--alignments` included: set it to the number of cores `score` may
use (e.g. the CPU count). With `--alignments`, the reads are measured while the models that do not
need them score, within `-t` threads; `-t 1` measures the reads, then scores. winnowr 0.1.2 and
earlier ran `score --alignments` on up to twice `-t` threads: if you passed `-t N/2` to stay
within N cores, pass `-t N` now. See [Performance tuning](../user-guide/performance.md#threads).

## Usage

```console
# With strlen and pileup tables
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --strlen sample.strlen.tsv.gz --pileup sample.pileup.tsv.gz -o sample.winnowr.vcf.gz
# Measuring the reads while scoring, dropping only
winnowr score --vcf sample.vcf.gz --gvcf sample.g.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  --alignments sample.cram --reference ref.fa --mode veto -o sample.winnowr.vcf.gz
# Models trained with extract --feature-source vcf: no gVCF
winnowr score --vcf sample.vcf.gz --models snp.winnowr.json indel.winnowr.json \
  --regions segdups=segdups.bed.gz notinalldifficultregions=GRCh38_notinalldifficultregions.bed.gz \
  -o sample.winnowr.vcf.gz
```

## Options

```text
{{#include ../../_generated/cli/score.txt}}
```
