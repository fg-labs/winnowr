//! Builds the vendored XGBoost as a static library and links it, with OpenMP, statically.
//!
//! Supported targets are macOS and Linux:
//!
//! - **macOS**: XGBoost is compiled with Apple clang and LLVM's OpenMP runtime, `libomp.a`, found
//!   under `$WINNOWR_LIBOMP_PREFIX` when set, else under Homebrew's `libomp` prefix.
//! - **Linux**: XGBoost is compiled with GCC and GCC's OpenMP runtime, `libgomp.a`, found by
//!   asking the compiler. (clang on Linux would need LLVM's `libomp.a`, which distributions do not
//!   ship as a static library.) Some distributions (e.g. RHEL and Amazon Linux) build `libgomp.a`
//!   without `-fPIC`, so it cannot be linked into a position-independent executable, Rust's
//!   default; there the shared `libgomp` is linked instead, with a warning, unless the binary is
//!   built non-position-independent (`-C relocation-model=static`).
//!
//! XGBoost is built from a copy of the vendored sources under `OUT_DIR`, with the patches in
//! `patches/` applied (see [`patched_source`]), so the submodule checkout stays pristine.
//!
//! The C++ compiler is the one the `cc` crate selects (honouring `CXX`, `CXX_<target>`,
//! `TARGET_CXX` and compiler wrappers), which is also the one the `cmake` crate builds with. The
//! C++ standard library stays dynamic: it is part of every supported platform.

#[path = "build/patch.rs"]
mod patch;

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("vendor").join("xgboost");
    assert!(
        source.join("CMakeLists.txt").is_file()
            && source.join("dmlc-core/CMakeLists.txt").is_file(),
        "the XGBoost sources are missing from {}: run `git submodule update --init --recursive`",
        source.display()
    );
    declare_rerun_triggers(&manifest_dir, &source);
    println!("cargo:rustc-env=WINNOWR_XGBOOST_VENDORED_VERSION={}", vendored_version(&source));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let source = patched_source(&source, &manifest_dir.join("patches"), &out_dir);

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
    let compiler = cc::Build::new().cpp(true).get_compiler();
    let openmp = match target_os.as_str() {
        "macos" => OpenMp::from_libomp(&compiler),
        "linux" => OpenMp::from_libgomp(&compiler),
        other => panic!("winnowr-xgboost supports macOS and Linux targets, not `{other}`"),
    };

    let mut config = cmake::Config::new(&source);
    config
        .profile("Release")
        .define("BUILD_STATIC_LIB", "ON")
        .define("USE_OPENMP", "ON")
        .define("BUILD_DEPRECATED_CLI", "OFF")
        .define("GOOGLE_TEST", "OFF")
        .define("BUILD_WITH_GIT_HASH", "OFF")
        // Keep build outputs under OUT_DIR rather than the source tree's lib/.
        .define("KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR", "ON");
    if let Some(prefix) = &openmp.cmake_root {
        config.define("OpenMP_ROOT", prefix);
    }
    let installed = config.build();

    // libdmlc is installed beside libxgboost by some CMake versions and left in the build tree by
    // others; lib64 is used on some Linux distributions.
    let library_dirs = ["lib", "lib64", "build/dmlc-core"].map(|dir| installed.join(dir));
    for dir in library_dirs.iter().filter(|dir| dir.is_dir()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=xgboost");
    println!("cargo:rustc-link-lib=static=dmlc");

    // The crate reaches libgomp's internal wait settings only when it links libgomp statically.
    println!("cargo::rustc-check-cfg=cfg(winnowr_static_libgomp)");
    match openmp.static_link(&compiler, &out_dir) {
        StaticLink::Links => {
            link_static_copy(&openmp.archive, &out_dir, openmp.name);
            if openmp.name == "gomp" {
                println!("cargo:rustc-cfg=winnowr_static_libgomp");
            }
        }
        StaticLink::NotPositionIndependent => {
            println!(
                "cargo:warning={} cannot be linked into a position-independent executable (it \
                 was built without -fPIC); linking the shared lib{}, which the binary then needs \
                 at run time. Build with RUSTFLAGS='-C relocation-model=static' (and --target) to \
                 link it statically.",
                openmp.archive.display(),
                openmp.name
            );
            println!("cargo:rustc-link-lib=dylib={}", openmp.name);
        }
        StaticLink::ProbeFailed(error) => {
            println!(
                "cargo:warning=a test program linking {} statically failed to build, for a reason \
                 other than position independence; linking the shared lib{}, which the binary \
                 then needs at run time. The compiler said: {error}",
                openmp.archive.display(),
                openmp.name
            );
            println!("cargo:rustc-link-lib=dylib={}", openmp.name);
        }
    }
    for library in openmp.system_libraries {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}

/// Rerun the build when the vendored sources or the inputs that select OpenMP change.
///
/// Printing any `rerun-if-changed` turns off Cargo's default of rerunning on every package file
/// change, so every directory the CMake build reads is listed (Cargo scans directories
/// recursively). The `cc` crate declares its own compiler variables (`CC`, `CXX`, ...).
fn declare_rerun_triggers(manifest_dir: &Path, source: &Path) {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build");
    println!("cargo:rerun-if-changed={}", manifest_dir.join("patches").display());
    for path in ["CMakeLists.txt", "cmake", "include", "src", "plugin"] {
        println!("cargo:rerun-if-changed={}", source.join(path).display());
    }
    for path in ["CMakeLists.txt", "cmake", "include", "src"] {
        println!("cargo:rerun-if-changed={}", source.join("dmlc-core").join(path).display());
    }
    println!("cargo:rerun-if-env-changed=WINNOWR_LIBOMP_PREFIX");
    println!("cargo:rerun-if-env-changed=HOMEBREW_PREFIX");
}

/// The vendored files and directories that the CMake build reads, relative to the XGBoost source
/// root: the same set that the crate's `Cargo.toml` packages.
const SOURCE_PATHS: [&str; 12] = [
    "CMakeLists.txt",
    "LICENSE",
    "NEWS.md",
    "cmake",
    "include",
    "src",
    "plugin",
    "dmlc-core/CMakeLists.txt",
    "dmlc-core/LICENSE",
    "dmlc-core/cmake",
    "dmlc-core/include",
    "dmlc-core/src",
];

/// Copies the vendored sources that the build reads to `OUT_DIR/xgboost-src`, applies the
/// patches in `patches_dir` (`*.patch`, in file name order) to the copy, and returns its path.
///
/// The patches are unified diffs with paths relative to the XGBoost source root (`a/src/...`,
/// as `git diff` writes them in the submodule), and must apply exactly: a hunk whose lines do not
/// match the source at its stated position fails the build, so updating the submodule cannot
/// silently drop or misplace a patch. Only files whose contents change are rewritten, and files
/// not in the source are removed, so CMake rebuilds only what changed.
fn patched_source(source: &Path, patches_dir: &Path, out_dir: &Path) -> PathBuf {
    let mut files = std::collections::BTreeMap::new();
    for path in SOURCE_PATHS {
        collect_files(source, Path::new(path), &mut files);
    }
    let mut patches: Vec<PathBuf> = std::fs::read_dir(patches_dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", patches_dir.display()))
        .map(|entry| entry.expect("reading a patches directory entry").path())
        .filter(|path| {
            // Skip hidden files, e.g. editor backups or macOS AppleDouble (`._*`) files.
            let hidden =
                path.file_name().is_some_and(|name| name.to_string_lossy().starts_with('.'));
            !hidden && path.extension().is_some_and(|ext| ext == "patch")
        })
        .collect();
    patches.sort();
    for patch in &patches {
        let text = std::fs::read_to_string(patch)
            .unwrap_or_else(|e| panic!("reading {}: {e}", patch.display()));
        for file in patch::parse_patch(&text).unwrap_or_else(|e| panic!("{}: {e}", patch.display()))
        {
            let original = files.get(&file.path).unwrap_or_else(|| {
                panic!("{}: patches {}, which is not in the source", patch.display(), file.path)
            });
            let original = std::str::from_utf8(original)
                .unwrap_or_else(|e| panic!("{}: {} is not UTF-8: {e}", patch.display(), file.path));
            let patched = file
                .apply(original)
                .unwrap_or_else(|e| panic!("{}: {}: {e}", patch.display(), file.path));
            files.insert(file.path, patched.into_bytes());
        }
    }

    let dest = out_dir.join("xgboost-src");
    for (path, contents) in &files {
        let target = dest.join(path);
        if std::fs::read(&target).is_ok_and(|existing| existing == *contents) {
            continue;
        }
        let parent = target.parent().expect("a source file has a parent directory");
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("creating {}: {e}", parent.display()));
        std::fs::write(&target, contents)
            .unwrap_or_else(|e| panic!("writing {}: {e}", target.display()));
    }
    let mut stale = std::collections::BTreeMap::new();
    if dest.is_dir() {
        collect_files(&dest, Path::new(""), &mut stale);
    }
    for path in stale.keys().filter(|path| !files.contains_key(*path)) {
        let target = dest.join(path);
        std::fs::remove_file(&target)
            .unwrap_or_else(|e| panic!("removing the stale {}: {e}", target.display()));
    }
    dest
}

/// Adds every file under `root.join(path)` (a file or a directory) to `files`, keyed by its
/// `/`-separated path relative to `root`; missing paths are skipped.
fn collect_files(
    root: &Path,
    path: &Path,
    files: &mut std::collections::BTreeMap<String, Vec<u8>>,
) {
    let full = root.join(path);
    if full.is_dir() {
        let entries =
            std::fs::read_dir(&full).unwrap_or_else(|e| panic!("reading {}: {e}", full.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|e| panic!("reading {}: {e}", full.display()));
            collect_files(root, &path.join(entry.file_name()), files);
        }
    } else if full.is_file() {
        let key = path
            .components()
            .map(|c| c.as_os_str().to_str().expect("source paths are UTF-8"))
            .collect::<Vec<_>>()
            .join("/");
        let contents =
            std::fs::read(&full).unwrap_or_else(|e| panic!("reading {}: {e}", full.display()));
        files.insert(key, contents);
    }
}

/// The vendored XGBoost's `major.minor.patch` version, from `include/xgboost/version_config.h`.
fn vendored_version(source: &Path) -> String {
    let header = source.join("include/xgboost/version_config.h");
    let text = std::fs::read_to_string(&header)
        .unwrap_or_else(|e| panic!("reading {}: {e}", header.display()));
    let component = |name: &str| {
        let prefix = format!("#define XGBOOST_VER_{name} ");
        text.lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .and_then(|rest| rest.split_whitespace().next())
            .unwrap_or_else(|| panic!("XGBOOST_VER_{name} is missing from {}", header.display()))
            .to_string()
    };
    format!("{}.{}.{}", component("MAJOR"), component("MINOR"), component("PATCH"))
}

/// The OpenMP runtime to link statically, and how CMake should find it.
struct OpenMp {
    /// The static archive, `lib<name>.a`.
    archive: PathBuf,
    /// The library name (`omp` or `gomp`).
    name: &'static str,
    /// The prefix to pass to CMake as `OpenMP_ROOT`, if CMake cannot find it alone.
    cmake_root: Option<PathBuf>,
    /// The shared system libraries the runtime and XGBoost need.
    system_libraries: &'static [&'static str],
}

impl OpenMp {
    /// LLVM's `libomp.a`, for Apple clang on macOS.
    fn from_libomp(compiler: &cc::Tool) -> Self {
        assert!(
            compiler.is_like_clang(),
            "on macOS winnowr-xgboost links LLVM's libomp, which needs clang, but the C++ \
             compiler is {}",
            compiler.path().display()
        );
        let prefix = libomp_prefix();
        Self {
            archive: prefix.join("lib/libomp.a"),
            name: "omp",
            cmake_root: Some(prefix),
            system_libraries: &["c++"],
        }
    }

    /// Whether the static runtime can be linked into the binary: always on macOS; on Linux, if
    /// the binary is not position-independent, or a position-independent test program links
    /// against the archive (it does not when the archive was built without `-fPIC`).
    fn static_link(&self, compiler: &cc::Tool, out_dir: &Path) -> StaticLink {
        if self.name != "gomp" {
            return StaticLink::Links;
        }
        let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
        if rustflags.split('\x1f').any(|flag| flag.ends_with("relocation-model=static")) {
            return StaticLink::Links;
        }
        let dir = out_dir.join("gomp-pie-check");
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("creating {}: {e}", dir.display()));
        let source = dir.join("check.cpp");
        std::fs::write(
            &source,
            "#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n",
        )
        .unwrap_or_else(|e| panic!("writing {}: {e}", source.display()));
        let output = compiler
            .to_command()
            .args(["-fPIE", "-pie"])
            .arg(&source)
            .arg(&self.archive)
            .args(["-lpthread", "-ldl", "-o"])
            .arg(dir.join("check"))
            .output();
        match output {
            Ok(output) if output.status.success() => StaticLink::Links,
            Ok(output) => StaticLink::from_probe_error(&String::from_utf8_lossy(&output.stderr)),
            Err(e) => {
                StaticLink::ProbeFailed(format!("could not run {}: {e}", compiler.path().display()))
            }
        }
    }

    /// GCC's `libgomp.a`, for GCC on Linux.
    fn from_libgomp(compiler: &cc::Tool) -> Self {
        assert!(
            compiler.is_like_gnu() && !compiler.is_like_clang(),
            "on Linux winnowr-xgboost links GCC's static OpenMP runtime (libgomp.a), which needs \
             GCC, but the C++ compiler is {}: set CXX (and CC) to GCC",
            compiler.path().display()
        );
        Self {
            archive: compiler_library(compiler, "libgomp.a"),
            name: "gomp",
            cmake_root: None,
            system_libraries: &["stdc++", "pthread", "dl", "m"],
        }
    }
}

/// Whether the OpenMP runtime's static archive can be linked into the binary.
#[derive(Debug, PartialEq)]
enum StaticLink {
    /// It can.
    Links,
    /// It was built without `-fPIC`, so it cannot be linked into a position-independent
    /// executable.
    NotPositionIndependent,
    /// The test program failed for another reason, given by the end of the compiler's output.
    ProbeFailed(String),
}

impl StaticLink {
    /// The outcome of a test program that failed to build with `stderr`: a linker error about a
    /// relocation (or one telling to recompile with `-fPIC` / `-fPIE`) means the archive is not
    /// position-independent; any other error (e.g. a missing `omp.h`) is reported as it is.
    fn from_probe_error(stderr: &str) -> Self {
        let lower = stderr.to_ascii_lowercase();
        if ["relocation", "-fpic", "-fpie"].iter().any(|word| lower.contains(word)) {
            return Self::NotPositionIndependent;
        }
        let lines: Vec<&str> = stderr.lines().filter(|line| !line.trim().is_empty()).collect();
        let tail = lines[lines.len().saturating_sub(5)..].join(" | ");
        Self::ProbeFailed(if tail.is_empty() { "nothing".to_string() } else { tail })
    }
}

/// The installation prefix of LLVM's OpenMP runtime (holding `lib/libomp.a`) on macOS.
///
/// `WINNOWR_LIBOMP_PREFIX`, when set, must hold it: an explicit override is never silently
/// replaced by another installation.
fn libomp_prefix() -> PathBuf {
    let has_archive = |prefix: &Path| prefix.join("lib/libomp.a").is_file();
    if let Some(prefix) = env::var_os("WINNOWR_LIBOMP_PREFIX").map(PathBuf::from) {
        assert!(
            has_archive(&prefix),
            "WINNOWR_LIBOMP_PREFIX is set to {}, which has no lib/libomp.a",
            prefix.display()
        );
        return prefix;
    }
    let candidates: Vec<PathBuf> = env::var_os("HOMEBREW_PREFIX")
        .map(|p| Path::new(&p).join("opt/libomp"))
        .into_iter()
        .chain(["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"].map(PathBuf::from))
        .collect();
    candidates.iter().find(|prefix| has_archive(prefix)).cloned().unwrap_or_else(|| {
        panic!(
            "libomp.a not found (tried {}): install it (e.g. `brew install libomp`) or set \
             WINNOWR_LIBOMP_PREFIX to the prefix holding lib/libomp.a",
            candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        )
    })
}

/// Copy the static library `archive` into a directory of its own under `out_dir`, as
/// `lib<name>.a`, and link it statically from there.
///
/// Linkers prefer a shared library over a static one of the same name, and the system's
/// `libomp.dylib` / `libgomp.so` sits beside the archive. (Naming the archive verbatim,
/// `-l:libgomp.a`, is not supported by Apple's linker.)
fn link_static_copy(archive: &Path, out_dir: &Path, name: &str) {
    let dir = out_dir.join(format!("static-{name}"));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("creating {}: {e}", dir.display()));
    let copy = dir.join(format!("lib{name}.a"));
    // The copy keeps the archive's (often read-only) permissions, so a rerun of this build script
    // must remove it rather than overwrite it.
    if copy.exists() {
        std::fs::remove_file(&copy)
            .unwrap_or_else(|e| panic!("removing the stale {}: {e}", copy.display()));
    }
    std::fs::copy(archive, &copy)
        .unwrap_or_else(|e| panic!("copying {} to {}: {e}", archive.display(), copy.display()));
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static={name}");
}

/// The path of a library that ships with the compiler (e.g. GCC's `libgomp.a`).
fn compiler_library(compiler: &cc::Tool, name: &str) -> PathBuf {
    // `to_command` keeps any wrapper (e.g. ccache) and the compiler's own arguments.
    let output =
        compiler.to_command().arg(format!("-print-file-name={name}")).output().unwrap_or_else(
            |e| panic!("could not run `{}` to locate {name}: {e}", compiler.path().display()),
        );
    let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    assert!(
        output.status.success() && path.is_absolute() && path.is_file(),
        "`{} -print-file-name={name}` did not find {name}: install GCC's static OpenMP runtime \
         (e.g. the libgcc-<version>-dev package)",
        compiler.path().display()
    );
    path
}
