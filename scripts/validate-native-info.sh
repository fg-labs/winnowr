#!/usr/bin/env bash
# Genotype real input twice: default output for INFO fallbacks, and retained SB for the baseline.
# Neither MQ nor SOR is synthesized. Keep all generated VCFs and logs outside the repository.
set -Eeuo pipefail
if [[ $# != 6 && $# != 7 ]]; then
    printf 'Usage: %s GATK_JAR REFERENCE GVCF WINNOWR LIBTEST NEW_OUTPUT_DIRECTORY [CONTIG]\n' "$0" >&2
    exit 2
fi
jar=$1
reference=$2
gvcf=$3
winnowr=$4
test=$5
out=$6
contig=${7:-chr20}
step='creating output directory'
error_log=$out
# Preserve the failing command's status while naming its diagnostic log.
failure() {
    local code=$?
    printf 'Failed during %s; inspect %s (exit %s)\n' "$step" "$error_log" "$code" >&2
    exit "$code"
}
trap failure ERR
mkdir "$out"
step='converting and indexing the real gVCF'
error_log="$out/convert.log"
# GATK/HTSJDK cannot read every BCF version; convert the real gVCF without altering annotations.
bcftools view -r "$contig" -Oz -o "$out/input.g.vcf.gz" "$gvcf" > "$out/convert.log" 2>&1
bcftools index -t "$out/input.g.vcf.gz" >> "$out/convert.log" 2>&1
for mode in native retained-sb; do
    command=(java -Xmx4g -jar "$jar" GenotypeGVCFs -R "$reference" -V "$out/input.g.vcf.gz"
        -L "$contig" --use-jdk-inflater true --use-jdk-deflater true)
    if [[ "$mode" == retained-sb ]]; then command+=(-A StrandBiasBySample); fi
    step="GenotypeGVCFs $mode"
    error_log="$out/$mode.gatk.log"
    "${command[@]}" -O "$out/$mode.vcf.gz" > "$error_log" 2>&1
    step="inspecting $mode header and call keys"
    error_log="$out/$mode.inspect.log"
    bcftools view -h "$out/$mode.vcf.gz" > "$out/$mode.header" 2> "$error_log"
    bcftools query -f '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n' "$out/$mode.vcf.gz" > "$out/$mode.keys" 2>> "$error_log"
done
step='checking identical nonempty call keys'
error_log="$out/native.keys and $out/retained-sb.keys"
cmp "$out/native.keys" "$out/retained-sb.keys"
[[ -s "$out/native.keys" ]]
step='checking native MQ/SOR declarations'
error_log="$out/native.header"
grep -q '^##INFO=<ID=MQ,' "$out/native.header"
grep -q '^##INFO=<ID=SOR,' "$out/native.header"
step='querying native SB values'
error_log="$out/native-fields.log"
bcftools query -u -f '[%SB]\n' "$out/native.vcf.gz" > "$out/native.sb" 2> "$error_log"
if grep -Eq '[0-9]' "$out/native.sb"; then
    printf 'Default GenotypeGVCFs output unexpectedly contains FORMAT/SB\n' >&2
    exit 1
fi
step='querying native MQ/SOR values'
bcftools query -f '%INFO/MQ\t%INFO/SOR\n' "$out/native.vcf.gz" > "$out/native.info" 2>> "$error_log"
step='checking numeric native MQ observations'
error_log="$out/native.info"
grep -Eq '^[0-9]' "$out/native.info"
step='querying retained FORMAT/SB values'
error_log="$out/retained-sb-fields.log"
bcftools query -u -f '[%SB]\n' "$out/retained-sb.vcf.gz" > "$out/retained-sb.sb" 2> "$error_log"
step='checking four numeric FORMAT/SB counts at every retained call'
error_log="$out/retained-sb.sb"
# Every baseline sor must come from FORMAT/SB, never from an INFO/SOR fallback.
(( $(wc -l < "$out/retained-sb.sb") == $(wc -l < "$out/retained-sb.keys") ))
if grep -Evq '^[0-9]+,[0-9]+,[0-9]+,[0-9]+$' "$out/retained-sb.sb"; then
    printf 'Retained-SB output has a call without four numeric FORMAT/SB counts\n' >&2
    exit 1
fi
# Force the baseline to join the original gVCF, even if a GATK version retains raw fields.
remove=''
for field in RAW_MQandDP QUALapprox VarDP AS_RAW_MQ; do
    if grep -q "^##INFO=<ID=$field," "$out/retained-sb.header"; then
        remove="${remove:+$remove,}INFO/$field"
    fi
done
baseline="$out/retained-sb.vcf.gz"
if [[ -n "$remove" ]]; then
    baseline="$out/baseline.vcf.gz"
    step='removing retained raw annotations for the gVCF join'
    error_log="$out/remove.log"
    bcftools annotate -x "$remove" -Oz -o "$baseline" "$out/retained-sb.vcf.gz" > "$out/remove.log" 2>&1
    bcftools index -t "$baseline" >> "$out/remove.log" 2>&1
fi
step='extracting the paired gVCF baseline'
error_log="$out/extract-gvcf.log"
"$winnowr" -t "${THREADS:-8}" extract --vcf "$baseline" --gvcf "$out/input.g.vcf.gz" \
    --contigs "$contig" -o "$out/gvcf.parquet" > "$out/extract-gvcf.log" 2>&1
step='extracting native INFO features'
error_log="$out/extract-vcf.log"
"$winnowr" -t "${THREADS:-8}" extract --vcf "$out/native.vcf.gz" --feature-source vcf \
    --contigs "$contig" -o "$out/vcf.parquet" > "$out/extract-vcf.log" 2>&1
step='comparing native features (test and expected result marker)'
error_log="$out/compare.log"
env WINNOWR_NATIVE_BASELINE="$out/gvcf.parquet" WINNOWR_NATIVE_VCF="$out/vcf.parquet" \
    "$test" --exact validation_tests::compare_native_features --ignored --nocapture \
    --test-threads=1 > "$out/compare.log" 2>&1
grep -q 'test result: ok. 1 passed' "$out/compare.log"
printf 'Native INFO validation completed; see compare.log and native.info.\n'
