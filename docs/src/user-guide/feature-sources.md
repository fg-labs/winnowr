# Feature sources

Six features come from annotations that only HaplotypeCaller's gVCF mode writes: `mq`, `dp_norm`,
`qd`, `as_alt_frac`, `sor` and `n_alt`. Where a feature table takes them from is its *feature
source*. `extract --feature-source` (or `extract --alignments`) chooses it, the feature table and
the models trained on it record it, and `score` takes it from the models.

| source | the VCF needs | notes |
|--------|---------------|-------|
| `gvcf` (default) | FORMAT/GT; INFO/RAW_MQandDP, QUALapprox, VarDP and AS_RAW_MQ (e.g. kept from a reblocked `--dragen-mode` gVCF), or else the gVCF it was genotyped from (`--gvcf`) | `sor` from FORMAT/SB |
| `alignments` (the default with `--alignments`) | FORMAT/GT, FORMAT/AD, FORMAT/PL, and FORMAT/SB or INFO/SOR | no gVCF; needs the reads (`--alignments`) and `--reference`, from which `mq`, `dp_norm` and `as_alt_frac` are measured; INFO/MQ is not read, and INFO/DP feeds only the `dp` feature |
| `vcf` | FORMAT/GT, FORMAT/AD, FORMAT/PL, INFO/MQ, INFO/DP, and FORMAT/SB or INFO/SOR | no gVCF or reads; `as_alt_frac` is missing |

The `alignments` and `vcf` sources are for a VCF without its gVCF, e.g. from GenotypeGVCFs or from
HaplotypeCaller run without `-ERC`. Neither is chosen in place of a missing gVCF: `alignments`
only with `--alignments`, and `vcf` only by `--feature-source vcf`. `extract` stops with an error
naming the fields the VCF header does not declare, and neither source takes `--gvcf`. Their
features are:

| feature | `gvcf` source | `alignments` source | `vcf` source |
|---------|---------------|---------------------|--------------|
| `mq` | √(RAW_MQandDP\[0\] / RAW_MQandDP\[1\]) | the root mean square MAPQ of the reads counted at the call | INFO/MQ |
| `dp_norm` | RAW_MQandDP\[1\] over the calls' median | the reads counted at the call, over the calls' median | INFO/DP over the calls' median |
| `qd` | QUALapprox / VarDP | FORMAT/PL\[0\] / VarDP as the `vcf` source computes it, but a zero VarDP is replaced with the reads counted at the call (below) | FORMAT/PL\[0\] / VarDP as ReblockGVCF computes it (below) |
| `as_alt_frac` | the ALT alleles' share of AS_RAW_MQ | the share of the summed squared MAPQs of the reads with a called ALT allele, among those with REF or a called ALT | missing |
| `sor` | StrandOddsRatio of FORMAT/SB | as the `vcf` source | of FORMAT/SB, or INFO/SOR where SB is missing |
| `n_alt` | the alleles of AS_RAW_MQ, less REF and `<NON_REF>` | as the `vcf` source | the ALT alleles a variant genotype calls (every ALT allele for other genotypes), less `<NON_REF>` |

For the `vcf` source, `dp_norm` divides INFO/DP by the calls' median INFO/DP, so a `--dp-median`
for models of the `vcf` source is the median INFO/DP that `extract` or `score` logs for the
sample's whole-genome calls (for the `alignments` source, the median number of reads counted,
logged the same way). `mq` is INFO/MQ where INFO/DP is positive, and missing where INFO/DP is zero
or missing, as the `gvcf` source's is where the RAW_MQandDP depth is zero; `extract` and `score`
count such calls in a warning.

VarDP is the depth ReblockGVCF divides by. ReblockGVCF first drops the ALT alleles the genotype does
not call, with their FORMAT/AD, so for a het or hom-alt genotype VarDP is the FORMAT/AD of the
reference and the called ALT alleles, summed, or FORMAT/DP when that sum is zero; for any other
genotype it is zero. A zero depth is replaced with INFO/DP, or 1 without it (for the `alignments`
source, with the number of reads counted at the call, or 1 without any, so `qd` does not depend on
INFO/DP). winnowr reads a
haploid genotype (e.g. `1` on chrX) as missing, so a haploid call takes INFO/DP, where GATK would
take the AD sum. INFO/QD is not read: GATK divides QUAL rather than PL\[0\], and replaces values of
35 or more with a random draw.

Records that lack a field the header declares lose only the features read from it: without
FORMAT/PL `qd` is missing, and without both FORMAT/SB and INFO/SOR `sor` is; `extract` and `score`
count them in warnings.

## Features measured from the reads

With `--alignments`, `extract` reads the sample's BAM or CRAM once, on `--contigs`, and counts at
each SNP and indel call the reads that:

- are mapped, primary, not duplicates, not supplementary and pass QC;
- have a MAPQ of at least 20 (and not 255, which means it is not available);
- overlap the call's REF bases, or the 5 bases on either side.

Their number is the call's depth, and `dp_norm` is that depth over the calls' median depth (or
`--dp-median`); `mq` is the root mean square of their MAPQs. At most 600 reads are counted at a
call: `extract` and `score` warn with the number of calls on each contig that reach that cap, whose
depth saturates at 600, and so their `dp_norm` at 600 over the median. For `as_alt_frac`, each
SNP allele comes from the read's base (with a base quality of at least 10). For an indel,
annotations choose among the existing ALT repeat-tract windows that cover every length-changing
ALT, called or uncalled, including equivalent placements. They use the shortest qualifying
window, then the leftmost on a tie, independently of ALT order. The read must align 5 flank bases
on each side. This annotation window can differ from the original window used by
[`winnowr pileup`](../tools/pileup.md), whose counts are unchanged. A read must have both the
allele's length change and its bases across the comparison span (a base below the base quality,
or `N`, matches any): a
read of a called indel allele's length must have that allele's bases and not also those of an
uncalled ALT allele of that length, and a read of the reference's length the reference's bases,
with no indel among them. In `A>T,AT` called `0/2`, a read inserting an `A` is not `AT`; in
`A>AT,AG` called `0/1`, a read inserting an `N` is neither. As ReblockGVCF keeps
`AS_RAW_MQ` only for the ALT alleles the genotype calls (every ALT allele for a hom-ref or missing
genotype), `as_alt_frac` is of the reads with the reference or a called ALT allele: reads with an
uncalled ALT allele or neither allele, or whose allele cannot be read, count towards `mq` and
`dp_norm` only. An indel record without a pure insertion or deletion allele (e.g. `CA>GTT,GA`),
whose called ALT alleles have no SNP or pure insertion or deletion allele, or for which no
candidate window covers all length-changing ALTs, has no `as_alt_frac`. For example,
`CAAAAGT>CAAAGT,CAAAA` called `0/1` can use the `GT` tract's window to cover both changes even
though the `A` deletion is listed first; the uncalled `GT` deletion is still excluded from both
allele sums.

A call at which no read is counted has no `mq` or `as_alt_frac`, and a `dp_norm` of 0. When more
than half of the calls have no read counted, their median depth is 0, and `dp_norm` is missing for
every call, those with reads too (models read it as missing); `extract` and `score` go on, and
`--dp-median` gives a depth to divide by instead. `extract` and `score` warn with the number of
calls without a read on each contig that the reads' header lists (so that, e.g., a CRAM with no
reads on one of the contigs shows). They stop with an error when a call's contig is not in the
reads' header, and when no read is counted at any call: the reads are of another sample, reference
or contigs, or none passes the settings.

`--alignment-settings` changes the three thresholds, as comma-separated settings over the defaults
(`mapq20,bq10,flank5`), each given at most once: for a VCF called in DRAGEN mode, whose
HaplotypeCaller counts reads from MAPQ 1, use `--alignment-settings mapq1`. The text is that of
`pileup --settings`, but `mapqN` means another thing there (the lowest MAPQ of the pileup's high
bin, where reads of every MAPQ are counted). The settings are part of the feature source, which is
recorded as e.g. `alignments:mapq20,bq10,flank5`: tables measured with different settings are of
different sources.

The flank sets both the bases beside a call within which a read counts, and the bases an indel
read must align beyond the repeat tract. HaplotypeCaller takes the reads within 2 bases of a
variant; the default flank is 5, the pileup's, so that one pass over the reads measures both (see
below). In the comparison below, `flank2` gave slightly closer `mq` (71.6% of calls equal, against
66.4% with `flank5`, at `mapq1`).

These are the reads as they are aligned, so the values approximate HaplotypeCaller's, which
reassembles and downsamples the reads and assigns reads to alleles by likelihood; both count the
two reads of an overlapping mate pair. See [How close the sources are](#how-close-the-sources-are).

`extract --alignments` measures only the six features above. Repeat-length and pileup features
still come from `winnowr strlen` and `winnowr pileup` tables (`extract --strlen`, `--pileup`).
With `--alignments`, `--reference` is the reads' reference; it is also used to left-align indels
when promote labels are extracted (`--truth-vcf`).

## One source per model

The values of the sources differ, so a model scores only calls whose features come from its own
source:

- `train` stops with an error when its tables are of different sources, and the model records
  theirs (`feature_source` in the model file, not written for `gvcf`).
- `score` derives the calls' features from the models' source, and stops with an error when the
  models' sources differ. For `vcf` and `alignments` models it needs no gVCF, and ignores `--gvcf`
  with a warning. For `alignments` models it needs `--alignments` and `--reference`, and measures
  the reads with the models' settings before any model scores. Models with pileup features get
  their allele read counts from the same pass over the reads, unless the pileup's base quality or
  flank is not the alignment settings' (the reads are then read once for each). `--strlen` and
  `--pileup` tables cannot be given with `--alignments`, so such models' repeat-length and pileup
  evidence is measured from the reads too.
- `evaluate` and `stratify` stop with an error when the table is of another source than a model.

Feature tables have no format version, and winnowr versions before feature sources ignore the
`feature_source` a table records: they read a `vcf` table as a `gvcf` one, without an error, and a
model they train on it records no source, so it would score calls with `gvcf` features. Extract,
train and score a model's tables with one winnowr version that has feature sources. (Those versions
refuse a `vcf` model, whose `feature_source` key they do not know.) The same holds for the
`alignments` source.

Extract every training and evaluation table the same way:

```console
# From the VCF and the reads
winnowr extract --vcf HG001.vcf.gz --alignments HG001.cram --reference GRCh38.fa \
  --happy HG001.happy.vcf.gz -o HG001.parquet
winnowr score --vcf sample.vcf.gz --alignments sample.cram --reference GRCh38.fa \
  --models snp.winnowr.json indel.winnowr.json -o sample.winnowr.vcf.gz

# From the VCF alone
winnowr extract --vcf HG001.vcf.gz --feature-source vcf --happy HG001.happy.vcf.gz -o HG001.parquet
winnowr score --vcf sample.vcf.gz --models snp.winnowr.json indel.winnowr.json -o sample.winnowr.vcf.gz
```

## How close the sources are

### The `vcf` source

#### Native INFO/MQ and INFO/SOR

A separate comparison genotyped a real 30x HG002 reblocked gVCF on chr20 with GenotypeGVCFs.
The default output contained native INFO/MQ and INFO/SOR, with no FORMAT/SB values. A second
GenotypeGVCFs run with `-A StrandBiasBySample` retained four numeric SB counts at every call, so
every baseline `sor` came from FORMAT/SB rather than INFO/SOR. The two
outputs had identical CHROM, POS, REF, ALT and GT at all 122,440 calls. No MQ or SOR values
were synthesized. The baseline joined the original gVCF for its raw annotations; 18 calls used
the documented fallback to the first gVCF site at their position with other alleles.

Extracted with winnowr `40ac9b3` (whose extraction code is unchanged at `f73d4c7`), the tables
had no missing values for these five features:

| feature | equal | median absolute difference | 99th percentile | maximum |
|---------|-------|----------------------------|-----------------|---------|
| `mq` | 69.8% | 0 | 0.0048201 | 0.0050004 |
| `dp_norm` | 100% | 0 | 0 | 0 |
| `qd` | 99.87% | 0 | 0 | 65.8333 |
| `sor` | 0% | 0.00024185 | 0.00049378 | 0.00049994 |
| `n_alt` | 99.87% | 0 | 0 | 1 |

Every MQ difference fits GenotypeGVCFs' two-decimal rounding plus VCF Float32 decoding;
every SOR difference fits its three-decimal rounding. This includes 554 calls with an SB total
of at most two, 550 of them nonzero: `StrandOddsRatio` uses a minimum count of zero, unlike
`FisherStrand`'s two. QD and n_alt each differ at 163 calls; these features remain approximations,
not interchangeable training inputs across sources.

The genotyper was the local GATK development build
`4.7.0.0-hpgc-v5-3-g5302144-SNAPSHOT`, source commit
`5302144f35f3d453f8b612808e95f3ad43e4b977`, with HTSJDK 5.0.2 and Java 17.0.18. It was not an
official GATK release. The source checkout was clean. RMSMappingQuality, StrandOddsRatio and
StrandBiasTest were unchanged from GATK 4.6.2.0; GenotypeGVCFs' relevant source differences
concerned somatic calling and GenomicsDB setup, neither used here. The decoder dependency does
differ from that release's HTSJDK 4.2.0, so this validates this build and dataset, not every
upstream release or caller.

To reproduce, install `bcftools` and a Java runtime compatible with your GATK JAR on `PATH`.
Use a real indexed gVCF and the matching reference FASTA, with its `.fai` index and GATK sequence
dictionary (`.dict`). Build the opt-in library tests and pass the test executable printed by
Cargo to the script:

```console
cargo test --release --lib --locked --no-run
bash scripts/validate-native-info.sh /path/to/gatk.jar GRCh38.fa sample.g.vcf.gz \
  /path/to/winnowr /path/to/winnowr-libtest /scratch/new-native-comparison chr20
```

The script preserves both generated VCFs, checks call keys and absence of SB in the default
output, extracts both feature tables, and reports finite-pair counts and missingness before
summarizing differences. Empty comparisons or no finite paired MQ/SOR observations fail.
Generated data and logs belong outside the repository.

#### Historical synthesized-MQ control

The historical chr20 comparison of a 30x whole-genome HG002 sample (123,031 calls) used a VCF
retaining FORMAT/SB and the paired reblocked gVCF. Its INFO/MQ was synthesized from that gVCF's
RAW_MQandDP and rounded to two decimals. The results below are a control for shared SB handling
and MQ rounding, **not independent validation of native INFO/MQ or the INFO/SOR fallback**:
`sor` is read from FORMAT/SB on both sides, and `mq` is derived from the same gVCF on both sides.

| feature | equal | median | 99th percentile | maximum |
|---------|-------|--------|-----------------|---------|
| `mq` | 69.7% | 0 | 0.0048 | 0.005 |
| `dp_norm` | 100% | 0 | 0 | 0 |
| `qd` | 99.93% | 0 | 0 | 18.5 |
| `sor` | 100% | 0 | 0 | 0 |
| `n_alt` | 99.96% | 0 | 0 | 1 |
| `as_alt_frac` | missing for the `vcf` source | | | |

INFO/DP equalled the gVCF's RAW_MQandDP depth at every call. The 91 calls (0.07%) whose `qd` or
`n_alt` differ are hom-alt calls whose PL\[0\] GenotypeGVCFs recomputed, calls with a `*` allele,
which their gVCF site lacks, and calls that took a gVCF site with other alleles.

To repeat the comparison on other data, extract the same calls with each source and compare the
two tables' feature columns call by call (the tables have the same rows, in the same order):

```console
# A copy of the VCF without raw annotations, so that both sources' tables take them from the gVCF
# (extract reads a VCF's own raw annotations in place of --gvcf when it has them)
bcftools annotate -x INFO/RAW_MQandDP,INFO/QUALapprox,INFO/VarDP,INFO/AS_RAW_MQ \
  -Oz -o sample.noraw.vcf.gz sample.vcf.gz
tabix sample.noraw.vcf.gz
# Only if the VCF has no INFO/MQ: add it from the gVCF's RAW_MQandDP, as GenotypeGVCFs computes it.
# The gVCF site is chosen as extract chooses it (the first site at the call's position with the
# call's REF and every ALT but `*`, or else the first site there), and MQ is left missing where
# that site's depth is 0.
bcftools query -r chr20 -e 'INFO/RAW_MQandDP="."' \
  -f 'G\t%CHROM\t%POS\t%REF\t%ALT\t%INFO/RAW_MQandDP\n' sample.g.vcf.gz > sites.tsv
bcftools query -r chr20 -f 'V\t%CHROM\t%POS\t%REF\t%ALT\n' sample.noraw.vcf.gz \
  | cat sites.tsv - \
  | mawk -F'\t' '
      $1 == "G" { k = $2 FS $3; i = ++n[k]; ref[k, i] = toupper($4); alt[k, i] = "," toupper($5) ","
                  raw[k, i] = $6; next }
      { k = $2 FS $3; pick = n[k] ? 1 : 0; m = split(toupper($5), a, ",")
        for (i = 1; i <= n[k]; i++) {
          if (ref[k, i] != toupper($4)) continue
          ok = 1; for (j = 1; j <= m; j++) if (a[j] != "*" && !index(alt[k, i], "," a[j] ",")) ok = 0
          if (ok) { pick = i; break } }
        if (!pick) next
        split(raw[k, pick], d, ",")
        if (d[2] > 0) printf "%s\t%s\t%s\t%s\t%.2f\n", $2, $3, $4, $5, sqrt(d[1] / d[2]) }' \
  | bgzip > mq.tab.gz
tabix -s1 -b2 -e2 mq.tab.gz
echo '##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS Mapping Quality">' > mq.hdr
bcftools annotate -a mq.tab.gz -c CHROM,POS,REF,ALT,INFO/MQ --pair-logic exact -h mq.hdr \
  -Oz -o sample.mq.vcf.gz sample.noraw.vcf.gz
tabix sample.mq.vcf.gz
# The two feature tables (use sample.noraw.vcf.gz for the vcf source if the VCF already has INFO/MQ)
winnowr extract --vcf sample.noraw.vcf.gz --gvcf sample.g.vcf.gz --contigs chr20 -o gvcf.parquet
winnowr extract --vcf sample.mq.vcf.gz --feature-source vcf --contigs chr20 -o vcf.parquet
```

The numbers above are, for each feature, the share of calls whose values are equal and the median,
99th percentile and maximum of the absolute differences over the calls where both are finite.

### The `alignments` source

On the same calls, with the sample's CRAM, the `gvcf` and `alignments` sources agree as follows.
The VCF was called in DRAGEN mode, so the reads were measured with
`--alignment-settings mapq1`; the second table is of the default settings (`mapq20`), which do
not match this VCF's caller. The "missing" column counts calls with a `gvcf` value and no
`alignments` one.

| feature (`mapq1`) | equal | median | 90th percentile | 99th percentile | maximum | missing |
|-------------------|-------|--------|-----------------|-----------------|---------|---------|
| `mq` | 66.4% | 0 | 0.79 | 3.7 | 48.4 | 23 |
| `dp_norm` | 1.4% | 0.045 | 0.12 | 2.1 | 15.8 | 0 |
| `as_alt_frac` | 56.9% | 0 | 0.048 | 0.52 | 1 | 359 |
| `qd` | 99.93% | 0 | 0 | 0 | 18.5 | 0 |
| `sor` | 100% | 0 | 0 | 0 | 0 | 0 |
| `n_alt` | 99.96% | 0 | 0 | 0 | 1 | 0 |

| feature (`mapq20`) | equal | median | 90th percentile | 99th percentile | maximum | missing |
|--------------------|-------|--------|-----------------|-----------------|---------|---------|
| `mq` | 65.5% | 0 | 3.0 | 10.7 | 46.4 | 1,174 |
| `dp_norm` | 1.1% | 0.040 | 0.22 | 1.6 | 16.1 | 0 |
| `as_alt_frac` | 54.3% | 0 | 0.051 | 0.53 | 1 | 1,620 |

`sor` and `n_alt` are derived as the `vcf` source derives them, and so is `qd` but for its
zero-VarDP fallback; they differ from the `gvcf` source as the `vcf` source's do (above). For the
read-measured features, with `mapq1`:

- `mq` is within 1 of the gVCF's at 92.4% of calls, and within 5 at 99.4%.
- `dp_norm` is within 0.1 at 86.5% of calls, and within 0.25 at 95.6%. It is rarely equal because
  the two depths have different medians (33 reads in the gVCF, 36 counted from the reads: a read is
  counted when it overlaps the 5 bases beside a call, where HaplotypeCaller takes 2, and
  HaplotypeCaller drops reads that winnowr counts, e.g. in reassembly). The large differences are
  where HaplotypeCaller downsamples: in the satellite repeats beside the chr20 centromere the gVCF
  records 30 to 50 reads at calls where 600 (the most that are counted) overlap; 353 calls reach
  that cap.
- `as_alt_frac` is within 0.05 at 90.3% of calls, and within 0.1 at 94.4%. SNPs agree more closely
  (99th percentile 0.19) than indels (median 0.023, 99th percentile 1), whose reads are assigned by
  their length and bases across the repeat tract, not by HaplotypeCaller's likelihoods. Of the 359
  calls without a value, 328 are indels: records without a pure insertion or deletion allele, or
  with no read spanning the repeat tract. The figures were measured before several refinements:
  current annotations choose an ALT-order-independent window covering every length-changing
  ALT, rather than relying on the pileup or called-ALT window. Indel records without a pure
  insertion or deletion allele, without a readable called allele, or without a qualifying window
  have no allele sums. A read of a called indel allele's length also needs that allele's bases;
  reads without bases (SEQ `*`), or with a reference skip (`N`) between the anchors, are in neither
  allele's sum. The historical counts above have not been remeasured with these rules.
- With `mapq20` on this DRAGEN-mode VCF, the 1,174 calls without `mq` are those where every read
  has a MAPQ below 20; their `dp_norm` is 0.

Both measurements logged version `0.1.2-9f83008f6215-dirty`: revision `9f83008f6215`
with uncommitted changes. That version does not identify an exact reproducible source snapshot;
the tables describe that development build, not the current allele matching.

To compare the current implementation with the `gvcf` table, extract a separate table for each
MAPQ threshold and compare them as above:

```console
winnowr extract --vcf sample.vcf.gz --alignments sample.cram --reference GRCh38.fa \
  --alignment-settings mapq1 --contigs chr20 -o alignments-mapq1.parquet
winnowr extract --vcf sample.vcf.gz --alignments sample.cram --reference GRCh38.fa \
  --alignment-settings mapq20 --contigs chr20 -o alignments-mapq20.parquet
```

Train and score with the same source and settings, as `train` and `score` require: the
differences above are between sources, not between runs of one.
