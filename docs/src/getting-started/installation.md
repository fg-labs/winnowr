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
library (`libomp.a` from Homebrew on macOS, GCC's `libgomp.a` on Linux), so none of them needs
installing alongside the binary; it still uses the system's C and C++ runtime libraries (e.g.
`libstdc++` on Linux). A source build on a Linux distribution whose `libgomp.a` is not
position-independent (e.g. RHEL, Amazon Linux) links the shared libgomp instead, with a warning,
unless built with `RUSTFLAGS='-C relocation-model=static'` and an explicit `--target`; so does any
build whose static-link test fails (see `xgboost/build.rs`).

[DEVELOPING.md](https://github.com/fg-labs/winnowr/blob/main/docs/DEVELOPING.md) covers the build
in more detail, including `WINNOWR_LIBOMP_PREFIX` for a libomp installed elsewhere
on macOS.

## Releases

Releases will also be published to [crates.io](https://crates.io), to
[bioconda](https://bioconda.github.io) and as prebuilt binaries for macOS (Apple silicon) and Linux
(x86_64 and aarch64) on the [releases page](https://github.com/fg-labs/winnowr/releases).

## Check the install

```console
winnowr --version
```

prints the winnowr version (with the git commit it was built from, for a build from a git
checkout) and, on a second line, the XGBoost version.
