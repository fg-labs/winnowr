# Installation

## From source

winnowr builds from source with:

- Rust 1.88 or newer,
- [CMake](https://cmake.org/),
- a C++ compiler: Apple clang with `brew install libomp` on macOS, or GCC on Linux,
- libclang, which builds htslib's bindings: `apt-get install libclang-dev` on Debian or Ubuntu
  (Apple clang provides it on macOS).

```console
git clone --recursive https://github.com/fg-labs/winnowr.git
cd winnowr
cargo install --locked --path .
```

`--recursive` fetches the vendored XGBoost sources (a git submodule). In a clone made without it,
run `git submodule update --init --recursive`.

XGBoost and htslib are built from source and linked statically, with the OpenMP runtime's static
library (`libomp.a` from Homebrew on macOS, GCC's `libgomp.a` on Linux) except as below, so none of
them needs installing alongside the binary; it still uses the system's C and C++ runtime libraries
(e.g. `libstdc++` on Linux). A source build on a Linux distribution whose `libgomp.a` is not
position-independent (e.g. RHEL, Amazon Linux) links the shared libgomp instead, with a warning,
unless built with `RUSTFLAGS='-C relocation-model=static'` and an explicit `--target`; so does any
build whose static-link test fails (see `xgboost/build.rs`). On macOS, a `WINNOWR_LIBOMP_PREFIX`
with only `libomp.dylib` (e.g. conda-forge's `llvm-openmp`) links the shared libomp, with a warning.

[DEVELOPING.md](https://github.com/fg-labs/winnowr/blob/main/docs/DEVELOPING.md) covers the build
in more detail, including `WINNOWR_LIBOMP_PREFIX` for a libomp installed elsewhere
on macOS.

## From crates.io

From version 0.1.1, winnowr can also be installed from
[crates.io](https://crates.io/crates/winnowr), with the same build requirements:

```console
cargo install --locked winnowr
```

## Releases

Prebuilt binaries for macOS (Apple silicon) and Linux (x86_64 and aarch64) are on the
[releases page](https://github.com/fg-labs/winnowr/releases). winnowr will also be published to
[bioconda](https://bioconda.github.io).

## Check the install

```console
winnowr --version
```

prints the winnowr version (with the git commit it was built from, for a build from a git
checkout) and, on a second line, the XGBoost version.
