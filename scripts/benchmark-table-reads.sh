#!/usr/bin/env bash
# Pass two release-mode libtest executables containing the same validation_tests module.
# Build setup, generation, output writing, equality checks and warmup are not timed.
set -Eeuo pipefail
if [[ $# != 3 ]]; then
    printf 'Usage: %s BEFORE_TEST AFTER_TEST NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
before=$1
after=$2
out=$3
rows=${ROWS:-4835999}
reps=${REPS:-2}
export RAYON_NUM_THREADS=${THREADS:-8}
# Bound digit counts before Bash arithmetic, which otherwise silently wraps large integers.
if [[ ! "$rows" =~ ^[1-9][0-9]{0,9}$ ]] || ((rows < 2 || rows > 1000000000)); then
    printf 'ROWS must be an integer from 2 to 1000000000\n' >&2
    exit 2
fi
if [[ ! "$reps" =~ ^[1-9][0-9]{0,3}$ ]] || ((reps > 1000)); then
    printf 'REPS must be an integer from 1 to 1000\n' >&2
    exit 2
fi
if [[ ! "$RAYON_NUM_THREADS" =~ ^[1-9][0-9]{0,3}$ ]] || ((RAYON_NUM_THREADS > 1024)); then
    printf 'THREADS must be an integer from 1 to 1024\n' >&2
    exit 2
fi
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
input="$out/labelled.parquet"
run() {
    local executable=$1
    error_log=$2
    step='libtest validation (command and expected result marker)'
    shift 2
    env "$@" "$executable" --exact validation_tests::validate_feature_tables \
        --ignored --nocapture --test-threads=1 > "$error_log" 2>&1
    grep -q 'test result: ok. 1 passed' "$error_log"
}
run "$after" "$out/generate.log" WINNOWR_TABLE_ACTION=generate \
    WINNOWR_TABLE_PATH="$input" WINNOWR_TABLE_ROWS="$rows"
completed=0
for selection in full snp; do
    expected=$rows
    columns=58
    if [[ "$selection" == snp ]]; then expected=$((rows - (rows + 4) / 5)); columns=8; fi
    for build in before after; do
        executable=$before
        if [[ "$build" == after ]]; then executable=$after; fi
        run "$executable" "$out/$selection-$build-parity.log" WINNOWR_TABLE_ACTION=read \
            WINNOWR_TABLE_PATH="$input" WINNOWR_TABLE_ROWS="$rows" WINNOWR_TABLE_SELECTION="$selection" \
            WINNOWR_TABLE_OUTPUT="$out/$selection-$build.parquet"
        step="checking $selection parity row/column counts"
        grep -Eq "READ seconds=[0-9.]+ rows=$expected columns=$columns$" "$error_log"
    done
    run "$after" "$out/$selection-compare.log" WINNOWR_TABLE_ACTION=compare \
        WINNOWR_TABLE_PATH="$out/$selection-before.parquet" \
        WINNOWR_TABLE_OTHER="$out/$selection-after.parquet"
    # Parity reads warm filesystem pages; each timed child initializes its own Rayon pool.
    for ((rep = 1; rep <= reps; rep++)); do
        builds='before after'
        if ((rep % 2 == 0)); then builds='after before'; fi
        for build in $builds; do
            executable=$before
            if [[ "$build" == after ]]; then executable=$after; fi
            run "$executable" "$out/$selection-$build-$rep.log" WINNOWR_TABLE_ACTION=read \
                WINNOWR_TABLE_PATH="$input" WINNOWR_TABLE_ROWS="$rows" WINNOWR_TABLE_SELECTION="$selection"
            step="checking $selection timed row/column counts"
            grep -Eq "READ seconds=[0-9.]+ rows=$expected columns=$columns$" "$error_log"
            ((completed += 1))
            printf '%s %s %s ' "$selection" "$build" "$rep"
            grep 'READ seconds=' "$error_log"
        done
    done
done
step='checking completed measurement count'
[[ "$completed" -eq $((4 * reps)) ]]
printf 'All %s measured reads succeeded; full and projected decoded outputs match.\n' "$completed"
