# winnowr-xgboost

A statically linked [XGBoost](https://github.com/dmlc/xgboost) with a small safe Rust API, built for
[winnowr](https://github.com/fg-labs/winnowr).

The crate vendors XGBoost (a git submodule pinned to a release tag or an upstream commit), applies
the small fixes in `patches/` to a copy of it (the submodule itself is never modified), builds it
with CMake as a static library, and links it and the OpenMP runtime statically (LLVM's `libomp.a`
on macOS, GCC's `libgomp.a` on Linux), so binaries that use it need no XGBoost or OpenMP shared
libraries at run time. The one exception: on a Linux distribution whose `libgomp.a` was built
without `-fPIC` (e.g. RHEL, Amazon Linux), a position-independent binary links the shared libgomp,
with a build warning, unless built with `RUSTFLAGS='-C relocation-model=static'`. Only the parts of XGBoost's C
API that winnowr uses are bound: dense matrices with labels, weights and feature names, and
boosters that train one round at a time, evaluate, predict, truncate, and save or load JSON.
Every booster takes an explicit thread count, and prediction checks that a matrix's feature
names and count match the model's.

Building needs CMake and, on macOS, Apple clang with `brew install libomp` (or
`WINNOWR_LIBOMP_PREFIX` set to the prefix holding `lib/libomp.a`); on Linux, GCC.

The published `xgb` / `xgboost_lib-sys` crates link XGBoost and OpenMP dynamically (or download a
prebuilt shared library at build time), which is why winnowr carries this small binding instead.

The crate's own code is MIT-licensed; the vendored XGBoost and dmlc-core sources are under the
Apache License 2.0.
