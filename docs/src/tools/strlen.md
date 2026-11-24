# strlen

Measures the lengths of the repeat tract around each indel call in the reads, for models with
repeat-length features. It writes one row per indel record with a pure insertion or deletion allele;
other records are skipped, and counted in the log. See [Repeat-length
features](../user-guide/repeat-length.md).

The table records how the reads were aligned, read from the header of `--alignments` (with
`--alignment-tags` over it), in an `##alignment=` line. See
[Alignment provenance](../user-guide/alignment-provenance.md).

## Usage

```console
# Measure the autosomes with the default read settings
winnowr strlen --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa -o sample.strlen.tsv.gz
# One shard of a larger run
winnowr strlen --vcf sample.vcf.gz --alignments sample.cram --reference ref.fa \
  --regions chr22:20000000-30000000 -o sample.chr22-20M.strlen.tsv.gz
```

## Options

```text
{{#include ../../_generated/cli/strlen.txt}}
```
