# extract

Extracts the features of a sample's SNP and indel calls into a Parquet feature table. With hap.py
output (`--happy`) it also labels the PASS calls, for training and evaluation, and with
`--truth-vcf` and `--truth-bed` the non-PASS calls, for promote models. See [Classes and
labels](../user-guide/classes-and-labels.md).

When the VCF lacks the raw annotations (its header does not declare them, or none of its records
has RAW_MQandDP), they come from the gVCF (`--gvcf`): each call takes them from the gVCF variant
site at its position that has its alleles, or else from the first variant site at its position.
Reference blocks carry no annotations and are skipped.

With `--alignments` (and `--reference`), these features are derived from the VCF's standard
annotations and measured from the reads instead, and with `--feature-source vcf` from the VCF
alone; neither takes a gVCF, and the table records the source. See
[Feature sources](../user-guide/feature-sources.md).

The depth feature (`dp_norm`) is each call's RAW_MQandDP depth (for `--alignments` and
`--feature-source vcf`, see [Feature sources](../user-guide/feature-sources.md)) over the median
of that depth across the calls on `--contigs`, which `extract` logs. For a VCF of only part of the genome, e.g. a target
footprint, that median is skewed; `--dp-median` gives the depth to divide by instead (at least 1).
It should be the same kind of median, of the same depths rather than coverage from another tool:
the median `extract` logs for the sample's whole-genome calls. Score the calls with the same
normalization (`score --dp-median`).

The table records the depth it divided by, and whether it was given, in its metadata
(`dp_norm_divisor`), for provenance: nothing checks it. Tables written by older winnowr versions
record none.

The table records the digest of each `--regions` BED (`region_digests`): of its merged
intervals, so the same however the file is compressed or sorted. `train` records a region's digest
in the model only when every training table records it, and fails when two tables record different
ones. If any table records none (e.g. one written by an older winnowr version), the model records
none for that region, with a warning, and `score` checks the BED it uses against the digest the
models record, or, when no model records one, uses it unchecked with a warning.

The table records how the sample's reads were aligned (`alignment`), from the `--strlen` and
`--pileup` tables and the header of `--alignments`, with `--alignment-tags` applied over each,
which must then agree; without any of them, from `--alignment-tags` alone. See
[Alignment provenance](../user-guide/alignment-provenance.md).

## Usage

```console
winnowr extract --vcf HG001.vcf.gz --gvcf HG001.g.vcf.gz --happy HG001.happy.vcf.gz \
  --truth-vcf HG001.truth.vcf.gz --truth-bed HG001.truth.bed --reference ref.fa \
  --strlen HG001.strlen.tsv.gz --pileup HG001.pileup.tsv.gz \
  --regions segdups=segdups.bed.gz lowmappabilityall=lowmappabilityall.bed.gz -o HG001.parquet
# Features derived from a VCF without its gVCF or reads
winnowr extract --vcf HG001.vcf.gz --feature-source vcf --happy HG001.happy.vcf.gz -o HG001.parquet
```

## Options

```text
{{#include ../../_generated/cli/extract.txt}}
```
