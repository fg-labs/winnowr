#!/usr/bin/env bash
# Fails if a winnowr binary loads XGBoost, OpenMP, or htslib and its codecs as shared libraries
# (they are meant to be linked statically). Usage: check-static-link.sh <binary>
set -euo pipefail

# A shared library whose name starts with one of these, in `ldd` (Linux) or `otool -L` (macOS)
# output. Matching the start of the name, not any substring, keeps out e.g. macOS's
# libcompression.dylib and otool's "compatibility version".
pattern='(^|[/[:space:]])lib(xgboost|g?omp|hts|bz2|lzma|deflate)[^/[:space:]]*\.(so|dylib)'

# The pattern is checked first, so that a pattern that matches nothing cannot pass every binary.
dynamic=(
    $'\tlibgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007f0000000000)'
    $'\t/opt/homebrew/opt/libomp/lib/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)'
    $'\tlibhts.so.3 => /usr/lib/x86_64-linux-gnu/libhts.so.3 (0x00007f0000000000)'
    $'\tlibbz2.so.1.0 => /lib/x86_64-linux-gnu/libbz2.so.1.0 (0x00007f0000000000)'
    $'\tliblzma.so.5 => /lib/x86_64-linux-gnu/liblzma.so.5 (0x00007f0000000000)'
    $'\tlibdeflate.so.0 => /lib/x86_64-linux-gnu/libdeflate.so.0 (0x00007f0000000000)'
    $'\t/usr/local/lib/libxgboost.dylib (compatibility version 0.0.0, current version 0.0.0)'
)
system=(
    $'\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1900.180.0)'
    $'\t/usr/lib/libcompression.dylib (compatibility version 1.0.0, current version 1.0.0)'
    $'\t/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 3502.1.255)'
    $'\tlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f0000000000)'
    $'\tlibstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f0000000000)'
)
for line in "${dynamic[@]}"; do
    if ! grep -qE "$pattern" <<< "$line"; then
        echo "::error::the static-link check misses a shared library: $line"
        exit 1
    fi
done
for line in "${system[@]}"; do
    if grep -qE "$pattern" <<< "$line"; then
        echo "::error::the static-link check flags a system library: $line"
        exit 1
    fi
done

binary=$1
if [ "$(uname -s)" = Darwin ]; then libs=$(otool -L "$binary"); else libs=$(ldd "$binary"); fi
echo "$libs"
if matches=$(grep -E "$pattern" <<< "$libs"); then
    echo "$matches"
    echo "::error::$binary loads XGBoost, OpenMP or htslib (or its codecs) as a shared library"
    exit 1
fi
