# Feature sources

Six features come from annotations that only HaplotypeCaller's gVCF mode writes: `mq`, `dp_norm`,
`qd`, `as_alt_frac`, `sor` and `n_alt`. Where a feature table takes them from is its *feature
source*. `extract --feature-source` chooses it, the feature table and the models trained on it
record it, and `score` takes it from the models.

| source | the VCF needs | notes |
|--------|---------------|-------|
| `gvcf` (default) | FORMAT/GT; INFO/RAW_MQandDP, QUALapprox, VarDP and AS_RAW_MQ (e.g. kept from a reblocked `--dragen-mode` gVCF), or else the gVCF it was genotyped from (`--gvcf`) | `sor` from FORMAT/SB |
| `vcf` | FORMAT/GT, FORMAT/AD, FORMAT/PL, INFO/MQ, INFO/DP, and FORMAT/SB or INFO/SOR | no gVCF; `as_alt_frac` is missing |

The `vcf` source is for a VCF without its gVCF, e.g. from GenotypeGVCFs or from HaplotypeCaller
run without `-ERC`. It is chosen only by `--feature-source vcf`, never in place of a missing gVCF.
`extract` stops with an error naming the fields the VCF header does not declare, and with
`--feature-source vcf` it does not take `--gvcf`. The features of the `vcf` source are:

| feature | `gvcf` source | `vcf` source |
|---------|---------------|--------------|
| `mq` | √(RAW_MQandDP\[0\] / RAW_MQandDP\[1\]) | INFO/MQ |
| `dp_norm` | RAW_MQandDP\[1\] over the calls' median | INFO/DP over the calls' median |
| `qd` | QUALapprox / VarDP | FORMAT/PL\[0\] / VarDP as ReblockGVCF computes it (below) |
| `as_alt_frac` | the ALT alleles' share of AS_RAW_MQ | missing |
| `sor` | StrandOddsRatio of FORMAT/SB | of FORMAT/SB, or INFO/SOR where SB is missing |
| `n_alt` | the alleles of AS_RAW_MQ, less REF and `<NON_REF>` | the ALT alleles a variant genotype calls (every ALT allele for other genotypes), less `<NON_REF>` |

`dp_norm` divides INFO/DP by the calls' median INFO/DP, so a `--dp-median` for models of the
`vcf` source is the median INFO/DP that `extract` or `score` logs for the sample's whole-genome
calls. `mq` is INFO/MQ where INFO/DP is positive, and missing where INFO/DP is zero or missing, as
the `gvcf` source's is where the RAW_MQandDP depth is zero; `extract` and `score` count such calls
in a warning.

VarDP is the depth ReblockGVCF divides by. ReblockGVCF first drops the ALT alleles the genotype does
not call, with their FORMAT/AD, so for a het or hom-alt genotype VarDP is the FORMAT/AD of the
reference and the called ALT alleles, summed, or FORMAT/DP when that sum is zero; for any other
genotype it is zero. A zero depth is replaced with INFO/DP, or 1 without it. winnowr reads a
haploid genotype (e.g. `1` on chrX) as missing, so a haploid call takes INFO/DP, where GATK would
take the AD sum. INFO/QD is not read: GATK divides QUAL rather than PL\[0\], and replaces values of
35 or more with a random draw.

Records that lack a field the header declares lose only the features read from it: without
FORMAT/PL `qd` is missing, and without both FORMAT/SB and INFO/SOR `sor` is; `extract` and `score`
count them in warnings.

## One source per model

The values of the two sources differ slightly, so a model scores only calls whose features come
from its own source:

- `train` stops with an error when its tables are of different sources, and the model records
  theirs (`feature_source` in the model file, written only for `vcf`).
- `score` derives the calls' features from the models' source, and stops with an error when the
  models' sources differ. For `vcf` models it needs no gVCF, and ignores `--gvcf` with a warning.
- `evaluate` and `stratify` stop with an error when the table is of another source than a model.

Feature tables have no format version, and winnowr versions before feature sources ignore the
`feature_source` a table records: they read a `vcf` table as a `gvcf` one, without an error, and a
model they train on it records no source, so it would score calls with `gvcf` features. Extract,
train and score a model's tables with one winnowr version that has feature sources. (Those versions
refuse a `vcf` model, whose `feature_source` key they do not know.)

Extract every training and evaluation table with the same `--feature-source`:

```console
winnowr extract --vcf HG001.vcf.gz --feature-source vcf --happy HG001.happy.vcf.gz -o HG001.parquet
winnowr score --vcf sample.vcf.gz --models snp.winnowr.json indel.winnowr.json -o sample.winnowr.vcf.gz
```

## How close the sources are

On the chr20 calls of a 30x whole-genome HG002 sample (123,031 calls), from a GenotypeGVCFs VCF
and the reblocked gVCF it was genotyped from, the two sources agree as follows (absolute
differences per call). The VCF has no INFO/MQ, so it was added from the gVCF's RAW_MQandDP and
rounded to two decimals, as GenotypeGVCFs writes it; `mq` therefore differs only by that rounding.

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
