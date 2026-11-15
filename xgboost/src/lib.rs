//! A statically linked XGBoost with the small safe API that winnowr needs.
//!
//! XGBoost is built from the vendored sources (see `build.rs`) and called through its C API.
//! Only the calls winnowr uses are bound: dense matrices with labels, weights and feature names;
//! boosters that train one round at a time, evaluate, predict, truncate, and save or load JSON.
//!
//! Every booster is given an explicit thread count, which bounds the OpenMP threads XGBoost uses
//! for training and prediction (XGBoost's own default is every core). The bound is per call: a
//! caller that predicts from several threads at once (e.g. rayon workers, each with its own
//! booster) uses up to their sum, so it should divide its thread budget between them.
//!
//! ```
//! use std::num::NonZeroUsize;
//! use winnowr_xgboost::{Booster, DMatrix};
//!
//! let threads = NonZeroUsize::new(2).unwrap();
//! let features = [0.0f32, 1.0, 2.0, 3.0];
//! let mut train = DMatrix::from_dense(&features, 4, 1, f32::NAN).unwrap();
//! train.set_labels(&[0.0, 0.0, 1.0, 1.0]).unwrap();
//! let mut booster = Booster::new(&[&train], threads).unwrap();
//! booster.set_param("objective", "binary:logistic").unwrap();
//! booster.update(0, &train).unwrap();
//! let prediction = booster.predict(&train).unwrap();
//! assert_eq!(prediction.shape, vec![4, 1]);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::num::NonZeroUsize;
use std::ptr;

/// The patch applier of `build.rs`, included here to test it.
#[cfg(test)]
#[path = "../build/patch.rs"]
mod build_patch;

/// LLVM's OpenMP runtime (`libomp`), which XGBoost is linked with on macOS.
#[cfg(target_os = "macos")]
mod libomp {
    use std::ffi::c_char;

    unsafe extern "C" {
        /// Sets defaults for the runtime's settings, as `NAME=VALUE`; only takes effect before the
        /// runtime is first used.
        pub fn kmp_set_defaults(settings: *const c_char);
    }
}

/// GCC's OpenMP runtime (`libgomp`), which XGBoost is linked with on Linux.
///
/// Only a statically linked `libgomp` is reachable: these are its internal (hidden) settings,
/// which the shared library does not export.
#[cfg(winnowr_static_libgomp)]
mod libgomp {
    unsafe extern "C" {
        /// How many times an idle worker thread polls for work before it sleeps; set from
        /// `OMP_WAIT_POLICY` / `GOMP_SPINCOUNT` when the runtime starts, and read at every wait.
        pub static mut gomp_spin_count_var: u64;
        /// The (much smaller) count used instead when a process runs more OpenMP threads than it
        /// has CPUs.
        pub static mut gomp_throttled_spin_count_var: u64;
    }
}

/// Measures how much work other processes give the machine's cores.
///
/// The CPU time counters, sampled over a short window, say how many cores are busy now, but
/// cannot see past every core being busy. The load average (runnable threads) can, but it trails
/// the machine's load by about a minute: right after one training run ends it still counts that
/// run's threads, and the next run would take an idle machine for a busy one. So the load
/// average is used only when the counters show every core busy.
mod cpu_usage {
    use std::ffi::c_int;
    use std::time::Duration;

    unsafe extern "C" {
        /// The system load averages (POSIX; in the C library on macOS and Linux).
        fn getloadavg(loadavg: *mut f64, nelem: c_int) -> c_int;
    }

    /// How long the CPU time counters are sampled over.
    const WINDOW: Duration = Duration::from_millis(250);

    /// How many cores' worth of work other processes have, while this process is idle (it calls
    /// this before starting any work), or `None` if the counters cannot be read.
    pub fn busy_cores() -> Option<f64> {
        let (busy0, total0, cpus) = cpu_ticks()?;
        std::thread::sleep(WINDOW);
        let (busy1, total1, _) = cpu_ticks()?;
        let total = total1.checked_sub(total0).filter(|&t| t > 0)?;
        let busy = busy1.checked_sub(busy0)?.min(total);
        let busy = cpus as f64 * busy as f64 / total as f64;
        if busy < cpus as f64 - 0.5 {
            return Some(busy);
        }
        let mut load = [0.0f64; 1];
        // SAFETY: the pointer is valid for writing the one requested value.
        let read = unsafe { getloadavg(load.as_mut_ptr(), 1) } == 1;
        Some(if read && load[0].is_finite() { busy.max(load[0]) } else { busy })
    }

    /// The machine's busy and total CPU time (in ticks, summed over CPUs), and its CPU count,
    /// from `/proc/stat`.
    #[cfg(target_os = "linux")]
    fn cpu_ticks() -> Option<(u64, u64, usize)> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let first = stat.lines().next()?.strip_prefix("cpu ")?;
        // user nice system idle iowait irq softirq steal (guest time is counted in user).
        let ticks: Vec<u64> =
            first.split_whitespace().take(8).map(str::parse).collect::<Result<_, _>>().ok()?;
        let (total, idle) = (ticks.iter().sum::<u64>(), ticks.get(3)? + ticks.get(4)?);
        let is_cpu = |line: &str| {
            line.strip_prefix("cpu")
                .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
        };
        let cpus = stat.lines().filter(|line| is_cpu(line)).count();
        Some((total - idle, total, cpus.max(1)))
    }

    /// The machine's busy and total CPU time (in ticks, summed over CPUs), and its CPU count,
    /// from the Mach host statistics.
    #[cfg(target_os = "macos")]
    fn cpu_ticks() -> Option<(u64, u64, usize)> {
        unsafe extern "C" {
            fn mach_host_self() -> u32;
            fn host_statistics(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
        }
        const HOST_CPU_LOAD_INFO: i32 = 3;
        // user, system, idle, nice
        let mut ticks = [0u32; 4];
        let mut count = 4u32;
        // SAFETY: `ticks` holds the `count` integers that HOST_CPU_LOAD_INFO writes.
        let status = unsafe {
            host_statistics(
                mach_host_self(),
                HOST_CPU_LOAD_INFO,
                ticks.as_mut_ptr().cast(),
                &raw mut count,
            )
        };
        if status != 0 || count != 4 {
            return None;
        }
        let [user, system, idle, nice] = ticks.map(u64::from);
        let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        Some((user + system + nice, user + system + idle + nice, cpus))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn cpu_ticks() -> Option<(u64, u64, usize)> {
        None
    }
}

/// How OpenMP worker threads wait for the next parallel region, as chosen by
/// [`wait_between_parallel_regions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMpWait {
    /// Idle threads keep polling (spinning) for work for a while before they sleep: the machine
    /// has a core for every thread.
    Spin,
    /// Idle threads spin only briefly before they sleep: the machine is busy.
    SpinBriefly,
    /// Idle threads sleep at once: the machine has more than twice as much work as cores.
    Sleep,
    /// The environment sets how threads wait (the named variable), which is left alone.
    FromEnvironment(&'static str),
    /// The runtime's waiting cannot be changed (a shared `libgomp`), so its default is kept.
    RuntimeDefault,
}

impl OpenMpWait {
    /// The policy for `threads` threads on `cores` cores, while other processes have `busy`
    /// cores' worth of work (`None` if unknown, which counts as busy).
    #[must_use]
    pub fn for_load(threads: usize, cores: usize, busy: Option<f64>) -> Self {
        // A negative or NaN measurement counts as unknown.
        let Some(busy) = busy.filter(|busy| *busy >= 0.0) else {
            return Self::SpinBriefly;
        };
        let demand = threads as f64 + busy.round();
        if demand <= cores as f64 {
            Self::Spin
        } else if demand <= 2.0 * cores as f64 {
            Self::SpinBriefly
        } else {
            Self::Sleep
        }
    }
}

/// The waiting policy chosen for OpenMP threads, and what it was chosen from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaitChoice {
    /// The policy.
    pub wait: OpenMpWait,
    /// The threads XGBoost was asked to use.
    pub threads: usize,
    /// The cores available to the process.
    pub cores: usize,
    /// How many cores' worth of work other processes had when the policy was chosen, if known.
    pub busy: Option<f64>,
}

impl fmt::Display for WaitChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let busy = self.busy.map_or_else(|| "unknown".to_string(), |busy| format!("{busy:.1}"));
        let (threads, cores) = (self.threads, self.cores);
        let context = format!("{threads} threads, {cores} cores, other work {busy} cores");
        match self.wait {
            OpenMpWait::Spin => write!(f, "spin, then sleep ({context}; {})", SETTINGS[0]),
            OpenMpWait::SpinBriefly => {
                write!(f, "spin briefly, then sleep ({context}; {})", SETTINGS[1])
            }
            OpenMpWait::Sleep => write!(f, "sleep at once ({context}; {})", SETTINGS[2]),
            OpenMpWait::FromEnvironment(var) => write!(f, "as set by {var} ({context})"),
            OpenMpWait::RuntimeDefault => {
                write!(f, "the runtime's default, which cannot be changed here ({context})")
            }
        }
    }
}

/// How the spin, brief-spin and sleep policies are set, for the log.
#[cfg(target_os = "macos")]
const SETTINGS: [&str; 3] = ["KMP_BLOCKTIME=200", "KMP_BLOCKTIME=1", "KMP_BLOCKTIME=0"];
#[cfg(not(target_os = "macos"))]
const SETTINGS: [&str; 3] =
    ["libgomp's default spin count", "libgomp's throttled spin count", "no spinning"];

/// The environment variables that set how the OpenMP runtime's threads wait.
#[cfg(target_os = "macos")]
const WAIT_VARIABLES: [&str; 2] = ["OMP_WAIT_POLICY", "KMP_BLOCKTIME"];
#[cfg(not(target_os = "macos"))]
const WAIT_VARIABLES: [&str; 2] = ["OMP_WAIT_POLICY", "GOMP_SPINCOUNT"];

/// Chooses how OpenMP worker threads wait between parallel regions, for XGBoost running on
/// `threads` threads, and returns the choice (to log).
///
/// Training runs thousands of short parallel regions per tree. Threads that sleep as soon as a
/// region ends spend much of the run being woken (on an idle M5, XGBoost's fits ran 5-14 times
/// slower at 8-16 threads). Threads that keep polling (spinning) for the next region avoid that,
/// but when other work wants their cores, spinning threads take time from the threads doing the
/// work, and training slows down many times over (about 100 times on an M5 with four times as
/// much work as cores; over 10 times on Linux with one thread more than cores). So the policy
/// follows the other work on the machine: the cores busy over 250 ms, from the CPU time
/// counters, or the load average when every core is busy. With `threads` plus that work
/// (rounded) at most the available cores, threads spin for long; at most twice the cores,
/// briefly; above that, they sleep at once.
///
/// - **macOS** (LLVM's `libomp`, whose default on Apple silicon is to sleep at once): a
///   `KMP_BLOCKTIME` of 200, 1 or 0 ms.
/// - **Linux** (GCC's `libgomp`): libgomp's default spin count (300,000 polls, a few ms), the
///   count it uses itself for a process with more threads than CPUs (100 polls), or none. A
///   shared `libgomp` cannot be changed, so its default is kept.
///
/// Setting `OMP_WAIT_POLICY`, or `KMP_BLOCKTIME` (macOS) or `GOMP_SPINCOUNT` (Linux), leaves the
/// runtime as the environment sets it. Models are unchanged by how threads wait.
///
/// On macOS the blocktime also sets XGBoost's team sizes (patch 0002 in `patches/`): with a
/// positive blocktime every parallel region uses all `threads`, and with 0 each region's team is
/// capped at its work, which saves waking threads that would sleep at once. Models are unchanged
/// by that too (`tests/blocktime.rs`).
///
/// Call it before XGBoost first runs: on macOS, later calls have no effect.
pub fn wait_between_parallel_regions(threads: NonZeroUsize) -> WaitChoice {
    let cores = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let threads = threads.get();
    if let Some(var) = WAIT_VARIABLES.into_iter().find(|v| std::env::var_os(v).is_some()) {
        return WaitChoice { wait: OpenMpWait::FromEnvironment(var), threads, cores, busy: None };
    }
    let busy = cpu_usage::busy_cores();
    let wait = set_wait(OpenMpWait::for_load(threads, cores, busy));
    WaitChoice { wait, threads, cores, busy }
}

/// Checks that the OpenMP environment gives XGBoost's training the threads that `-t` asks for.
///
/// - A dynamic team (`OMP_DYNAMIC`) lets the runtime grant a parallel region fewer threads than
///   requested. XGBoost's `hist` training assumes the full team: blocks of work are then silently
///   skipped, and training fails a consistency check, crashes, or loses rows. It cannot be undone
///   reliably once the program runs (each host thread has its own copy of the setting). GCC's
///   libgomp (Linux) turns it on only for `true`; LLVM's libomp (macOS) also for `1`, `yes`, `on`,
///   `t` and the other spellings it reads as true.
/// - On macOS, libomp's `KMP_DEVICE_THREAD_LIMIT` (and its older name `KMP_ALL_THREADS`) below
///   the thread count shrinks every team the same way, while XGBoost, which reads only
///   `OMP_THREAD_LIMIT`, still asks for the full team.
/// - An `OMP_THREAD_LIMIT` below the thread count makes XGBoost cap its threads at the limit. That
///   is safe, but the model depends on the thread count, so it would not be the model `-t`
///   names; the limit is fixed at start.
///
/// All are refused up front. Prediction splits its work as the team's threads arrive, so it
/// needs none of these checks.
///
/// `var` looks up an environment variable (`std::env::var` outside tests).
///
/// # Errors
///
/// Returns a message naming the variable to unset or change.
pub fn check_openmp_environment(
    threads: NonZeroUsize,
    var: impl Fn(&str) -> Option<String>,
) -> std::result::Result<(), String> {
    OpenMpRuntime::LINKED.check_environment(threads, var)
}

/// The OpenMP runtime XGBoost is linked with, which sets how the environment is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenMpRuntime {
    /// LLVM's libomp (macOS).
    LibOmp,
    /// GCC's libgomp (Linux).
    LibGomp,
}

impl OpenMpRuntime {
    /// The runtime this crate links.
    const LINKED: Self = if cfg!(target_os = "macos") { Self::LibOmp } else { Self::LibGomp };

    /// The variables that cap a team below the threads XGBoost requests without XGBoost seeing
    /// the cap: libomp's thread limits other than `OMP_THREAD_LIMIT` (which XGBoost reads).
    fn hidden_thread_limits(self) -> &'static [&'static str] {
        match self {
            Self::LibOmp => &["KMP_DEVICE_THREAD_LIMIT", "KMP_ALL_THREADS"],
            Self::LibGomp => &[],
        }
    }

    /// Whether the runtime reads an `OMP_DYNAMIC` value as true.
    ///
    /// libgomp accepts only `true` and `false` (in any case, with surrounding white space).
    /// libomp (`__kmp_str_match_true`) reads a value as true when it starts with `1`, `t`, `y`,
    /// `on` or `.t`, or is `enabled`, in any case, or when it is a prefix of `true`, `yes` or
    /// `.true.` of the length those need.
    fn reads_as_true(self, value: &str) -> bool {
        match self {
            Self::LibGomp => value.trim().eq_ignore_ascii_case("true"),
            Self::LibOmp => {
                // libomp's `__kmp_str_match(target, len, data)`: the characters the two share
                // match (ignoring case), and at least `len` of them are compared.
                let matches = |target: &str, len: usize| {
                    let shared = target.len().min(value.len());
                    target.as_bytes()[..shared].eq_ignore_ascii_case(&value.as_bytes()[..shared])
                        && shared >= len
                };
                matches("true", 1)
                    || matches("on", 2)
                    || matches("1", 1)
                    || matches(".true.", 2)
                    || matches(".t.", 2)
                    || matches("yes", 1)
                    || value.eq_ignore_ascii_case("enabled")
            }
        }
    }

    /// See [`check_openmp_environment`].
    fn check_environment(
        self,
        threads: NonZeroUsize,
        var: impl Fn(&str) -> Option<String>,
    ) -> std::result::Result<(), String> {
        if let Some(value) = var("OMP_DYNAMIC").filter(|value| self.reads_as_true(value)) {
            return Err(format!(
                "OMP_DYNAMIC={value} lets OpenMP give XGBoost fewer threads than it asks for, \
                 which XGBoost's training does not handle (it can lose rows or crash); unset \
                 OMP_DYNAMIC or set it to false"
            ));
        }
        let limit = |name: &str| var(name).and_then(|value| value.trim().parse::<usize>().ok());
        for &name in self.hidden_thread_limits() {
            if let Some(limit) = limit(name).filter(|&limit| limit < threads.get()) {
                return Err(format!(
                    "{name}={limit} is below the {threads} threads requested, so OpenMP would \
                     give XGBoost fewer threads than it asks for, which XGBoost's training does \
                     not handle (it can lose rows or crash); use -t {limit} or less, or raise or \
                     unset {name}"
                ));
            }
        }
        match limit("OMP_THREAD_LIMIT") {
            Some(limit) if limit < threads.get() => Err(format!(
                "OMP_THREAD_LIMIT={limit} is below the {threads} threads requested, so XGBoost \
                 would run with {limit} threads, and a model trained so is not the one -t \
                 {threads} names; use -t {limit} or less, or raise or unset OMP_THREAD_LIMIT"
            )),
            _ => Ok(()),
        }
    }
}

/// Sets the runtime's waiting to `wait` (a spin, brief-spin or sleep policy), and returns the
/// policy in effect.
fn set_wait(wait: OpenMpWait) -> OpenMpWait {
    #[cfg(target_os = "macos")]
    {
        let setting: &std::ffi::CStr = match wait {
            OpenMpWait::Spin => c"KMP_BLOCKTIME=200",
            OpenMpWait::SpinBriefly => c"KMP_BLOCKTIME=1",
            _ => c"KMP_BLOCKTIME=0",
        };
        // SAFETY: the argument is a valid NUL-terminated string, which the runtime only reads.
        unsafe { libomp::kmp_set_defaults(setting.as_ptr()) };
        wait
    }
    #[cfg(winnowr_static_libgomp)]
    {
        // SAFETY: the runtime's settings were initialized when it was loaded, and are written
        // here before XGBoost starts any threads that read them.
        unsafe {
            match wait {
                OpenMpWait::Spin => {}
                OpenMpWait::SpinBriefly => {
                    libgomp::gomp_spin_count_var = libgomp::gomp_throttled_spin_count_var;
                }
                _ => {
                    libgomp::gomp_spin_count_var = 0;
                    libgomp::gomp_throttled_spin_count_var = 0;
                }
            }
        }
        wait
    }
    #[cfg(not(any(target_os = "macos", winnowr_static_libgomp)))]
    {
        let _ = wait;
        OpenMpWait::RuntimeDefault
    }
}

/// Raw declarations of the XGBoost C API functions used here (see `include/xgboost/c_api.h`).
mod ffi {
    use std::ffi::{c_char, c_float, c_int, c_void};

    pub type BstUlong = u64;
    pub type DMatrixHandle = *mut c_void;
    pub type BoosterHandle = *mut c_void;

    unsafe extern "C" {
        pub fn XGBGetLastError() -> *const c_char;
        pub fn XGBoostVersion(major: *mut c_int, minor: *mut c_int, patch: *mut c_int);
        pub fn XGDMatrixCreateFromMat_omp(
            data: *const c_float,
            nrow: BstUlong,
            ncol: BstUlong,
            missing: c_float,
            out: *mut DMatrixHandle,
            nthread: c_int,
        ) -> c_int;
        pub fn XGDMatrixSetFloatInfo(
            handle: DMatrixHandle,
            field: *const c_char,
            array: *const c_float,
            len: BstUlong,
        ) -> c_int;
        pub fn XGDMatrixSetStrFeatureInfo(
            handle: DMatrixHandle,
            field: *const c_char,
            features: *const *const c_char,
            size: BstUlong,
        ) -> c_int;
        pub fn XGDMatrixGetStrFeatureInfo(
            handle: DMatrixHandle,
            field: *const c_char,
            size: *mut BstUlong,
            out_features: *mut *const *const c_char,
        ) -> c_int;
        pub fn XGDMatrixFree(handle: DMatrixHandle) -> c_int;
        pub fn XGDMatrixNumRow(handle: DMatrixHandle, out: *mut BstUlong) -> c_int;
        pub fn XGDMatrixNumCol(handle: DMatrixHandle, out: *mut BstUlong) -> c_int;
        pub fn XGBoosterCreate(
            dmats: *const DMatrixHandle,
            len: BstUlong,
            out: *mut BoosterHandle,
        ) -> c_int;
        pub fn XGBoosterFree(handle: BoosterHandle) -> c_int;
        pub fn XGBoosterSetParams(handle: BoosterHandle, config: *const c_char) -> c_int;
        pub fn XGBoosterUpdateOneIter(
            handle: BoosterHandle,
            iter: c_int,
            dtrain: DMatrixHandle,
        ) -> c_int;
        pub fn XGBoosterEvalOneIter(
            handle: BoosterHandle,
            iter: c_int,
            dmats: *mut DMatrixHandle,
            evnames: *mut *const c_char,
            len: BstUlong,
            out_result: *mut *const c_char,
        ) -> c_int;
        pub fn XGBoosterPredictFromDMatrix(
            handle: BoosterHandle,
            dmat: DMatrixHandle,
            config: *const c_char,
            out_shape: *mut *const BstUlong,
            out_dim: *mut BstUlong,
            out_result: *mut *const c_float,
        ) -> c_int;
        pub fn XGBoosterSaveJsonConfig(
            handle: BoosterHandle,
            out_len: *mut BstUlong,
            out_str: *mut *const c_char,
        ) -> c_int;
        pub fn XGBoosterGetNumFeature(handle: BoosterHandle, out: *mut BstUlong) -> c_int;
        pub fn XGBoosterSetStrFeatureInfo(
            handle: BoosterHandle,
            field: *const c_char,
            features: *const *const c_char,
            size: BstUlong,
        ) -> c_int;
        pub fn XGBoosterGetStrFeatureInfo(
            handle: BoosterHandle,
            field: *const c_char,
            len: *mut BstUlong,
            out_features: *mut *const *const c_char,
        ) -> c_int;
        pub fn XGBoosterSaveModelToBuffer(
            handle: BoosterHandle,
            config: *const c_char,
            out_len: *mut BstUlong,
            out_dptr: *mut *const c_char,
        ) -> c_int;
        pub fn XGBoosterLoadModelFromBuffer(
            handle: BoosterHandle,
            buf: *const c_void,
            len: BstUlong,
        ) -> c_int;
        pub fn XGBoosterSlice(
            handle: BoosterHandle,
            begin_layer: c_int,
            end_layer: c_int,
            step: c_int,
            out: *mut BoosterHandle,
        ) -> c_int;
        pub fn XGBoosterBoostedRounds(handle: BoosterHandle, out: *mut c_int) -> c_int;
    }
}

/// An error from XGBoost (its last-error message) or from invalid arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    /// The error message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "XGBoost: {}", self.0)
    }
}

impl std::error::Error for Error {}

/// A `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Turn an XGBoost C API return code into a `Result`, fetching the error message on failure.
fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: XGBGetLastError returns a NUL-terminated, thread-local string that stays valid
    // until the next XGBoost call on this thread; it is copied immediately.
    let message = unsafe { CStr::from_ptr(ffi::XGBGetLastError()) };
    Err(Error(clean_message(&message.to_string_lossy())))
}

/// The user-facing part of an XGBoost error message.
///
/// XGBoost messages look like `[12:00:00] /build/path/src/learner.cc:123: <message>`, followed by
/// a native `Stack trace:` naming build-machine paths. Keep only `<message>`.
fn clean_message(raw: &str) -> String {
    let message = raw.split("\nStack trace:").next().unwrap_or(raw).trim();
    // The timestamp: `[hh:mm:ss] `.
    let message = match message.strip_prefix('[').and_then(|rest| rest.split_once("] ")) {
        Some((_, rest)) => rest,
        None => message,
    };
    // The source location: `<path>.<ext>:<line>: `.
    let message = match message.split_once(": ") {
        Some((location, rest)) if is_source_location(location) => rest,
        _ => message,
    };
    message.trim().to_string()
}

/// Whether `text` is a `<file>.<ext>:<line>` source location (with no spaces).
fn is_source_location(text: &str) -> bool {
    match text.rsplit_once(':') {
        Some((file, line)) => {
            !file.is_empty()
                && file.contains('.')
                && !file.contains(char::is_whitespace)
                && !line.is_empty()
                && line.bytes().all(|b| b.is_ascii_digit())
        }
        None => false,
    }
}

/// A C string from Rust text, rejecting interior NUL bytes.
fn c_string(text: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Error(format!("text contains a NUL byte: {text:?}")))
}

/// Convert a length to XGBoost's unsigned 64-bit length type.
fn bst_len(len: usize) -> ffi::BstUlong {
    // usize is at most 64 bits on every supported target.
    len as ffi::BstUlong
}

/// Convert a thread count to a C `int`, saturating (OpenMP caps it at its own limit anyway).
fn c_threads(threads: NonZeroUsize) -> c_int {
    c_int::try_from(threads.get()).unwrap_or(c_int::MAX)
}

/// Convert a length reported by XGBoost to `usize`.
fn from_bst_len(len: ffi::BstUlong, what: &str) -> Result<usize> {
    usize::try_from(len).map_err(|_| Error(format!("{what} {len} overflows usize")))
}

/// A slice from a pointer and length returned by XGBoost: empty when the length is zero (when
/// the pointer may be null), and an error for a null pointer with a non-zero length.
///
/// # Safety
///
/// When `len > 0` and `data` is non-null, `data` must point to `len` initialized values that
/// stay valid and unmodified for the returned lifetime.
unsafe fn borrowed_slice<'a, T>(data: *const T, len: usize) -> Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(Error(format!("XGBoost returned a null pointer for {len} values")));
    }
    // SAFETY: the caller guarantees `data` points to `len` valid values; it is non-null.
    Ok(unsafe { std::slice::from_raw_parts(data, len) })
}

/// Copy `len` NUL-terminated strings returned by XGBoost.
///
/// # Safety
///
/// When `len > 0` and `strings` is non-null, `strings` must point to `len` pointers, each null
/// or a valid NUL-terminated string, all valid until this returns.
unsafe fn copy_strings(strings: *const *const c_char, len: ffi::BstUlong) -> Result<Vec<String>> {
    let len = from_bst_len(len, "string count")?;
    // SAFETY: forwarded from the caller.
    let pointers = unsafe { borrowed_slice(strings, len) }?;
    pointers
        .iter()
        .map(|&pointer| {
            if pointer.is_null() {
                return Err(Error("XGBoost returned a null string".into()));
            }
            // SAFETY: forwarded from the caller; checked non-null above.
            Ok(unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned())
        })
        .collect()
}

/// The `(major, minor, patch)` version of the linked XGBoost.
#[must_use]
pub fn version() -> (i32, i32, i32) {
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    // SAFETY: the out-pointers reference live stack variables.
    unsafe { ffi::XGBoostVersion(&mut major, &mut minor, &mut patch) };
    (major, minor, patch)
}

/// A dense, in-memory data matrix (rows of `f32` features) with optional labels and weights.
#[derive(Debug)]
pub struct DMatrix {
    handle: ffi::DMatrixHandle,
}

// SAFETY: a DMatrix handle is owned exclusively by this value and XGBoost does not tie it to the
// creating thread, so it may move between threads. It is not `Sync`: setters mutate it.
unsafe impl Send for DMatrix {}

impl DMatrix {
    /// A matrix from row-major `values` with `num_rows` rows of `num_cols` features.
    ///
    /// Values equal to `missing` (or NaN, always) are treated as missing.
    ///
    /// # Errors
    ///
    /// Returns an error if `values.len() != num_rows * num_cols`, or XGBoost fails.
    pub fn from_dense(
        values: &[f32],
        num_rows: usize,
        num_cols: usize,
        missing: f32,
    ) -> Result<Self> {
        Self::from_dense_with_threads(values, num_rows, num_cols, missing, NonZeroUsize::MIN)
    }

    /// [`DMatrix::from_dense`], copying the values with `threads` OpenMP threads. The matrix is
    /// the same with any thread count.
    ///
    /// # Errors
    ///
    /// Returns an error if `values.len() != num_rows * num_cols`, or XGBoost fails.
    pub fn from_dense_with_threads(
        values: &[f32],
        num_rows: usize,
        num_cols: usize,
        missing: f32,
        threads: NonZeroUsize,
    ) -> Result<Self> {
        let expected = num_rows.checked_mul(num_cols);
        if expected != Some(values.len()) {
            return Err(Error(format!(
                "{} values do not fill {num_rows} rows of {num_cols} columns",
                values.len()
            )));
        }
        let mut handle = ptr::null_mut();
        // SAFETY: `values` holds exactly num_rows * num_cols floats (checked above), which
        // XGBoost copies; `handle` is a valid out-pointer.
        check(unsafe {
            ffi::XGDMatrixCreateFromMat_omp(
                values.as_ptr(),
                bst_len(num_rows),
                bst_len(num_cols),
                missing,
                &mut handle,
                c_threads(threads),
            )
        })?;
        Ok(Self { handle })
    }

    /// The number of rows.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn num_rows(&self) -> Result<usize> {
        let mut rows = 0;
        // SAFETY: the handle is live and `rows` is a valid out-pointer.
        check(unsafe { ffi::XGDMatrixNumRow(self.handle, &mut rows) })?;
        from_bst_len(rows, "row count")
    }

    /// The number of columns (features).
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn num_cols(&self) -> Result<usize> {
        let mut cols = 0;
        // SAFETY: the handle is live and `cols` is a valid out-pointer.
        check(unsafe { ffi::XGDMatrixNumCol(self.handle, &mut cols) })?;
        from_bst_len(cols, "column count")
    }

    /// Set the per-row training labels.
    ///
    /// # Errors
    ///
    /// Returns an error if there is not one label per row, or XGBoost fails.
    pub fn set_labels(&mut self, labels: &[f32]) -> Result<()> {
        self.set_float_info("label", labels)
    }

    /// Set the per-row training weights.
    ///
    /// # Errors
    ///
    /// Returns an error if there is not one weight per row, or XGBoost fails.
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<()> {
        self.set_float_info("weight", weights)
    }

    /// Set a per-row float field after checking its length.
    fn set_float_info(&mut self, field: &str, values: &[f32]) -> Result<()> {
        let rows = self.num_rows()?;
        if values.len() != rows {
            return Err(Error(format!("{} {field}s for {rows} rows", values.len())));
        }
        let name = c_string(field)?;
        // SAFETY: the handle is live, `name` is NUL-terminated, and `values` holds `rows`
        // floats, which XGBoost copies.
        check(unsafe {
            ffi::XGDMatrixSetFloatInfo(self.handle, name.as_ptr(), values.as_ptr(), bst_len(rows))
        })
    }

    /// Set the feature names, one per column.
    ///
    /// A booster trained on the matrix records them, and [`Booster::predict`] requires a matrix
    /// with the same names, in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error if there is not one name per column, a name contains a NUL byte, or
    /// XGBoost fails.
    pub fn set_feature_names<S: AsRef<str>>(&mut self, names: &[S]) -> Result<()> {
        let cols = self.num_cols()?;
        if names.len() != cols {
            return Err(Error(format!("{} feature names for {cols} columns", names.len())));
        }
        let owned = names.iter().map(|n| c_string(n.as_ref())).collect::<Result<Vec<_>>>()?;
        let pointers: Vec<*const c_char> = owned.iter().map(|n| n.as_ptr()).collect();
        let field = c_string("feature_name")?;
        // SAFETY: the handle is live; `pointers` holds `cols` NUL-terminated strings owned by
        // `owned`, which outlives the call; XGBoost copies them.
        check(unsafe {
            ffi::XGDMatrixSetStrFeatureInfo(
                self.handle,
                field.as_ptr(),
                pointers.as_ptr(),
                bst_len(cols),
            )
        })
    }

    /// The feature names (empty if none were set).
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn feature_names(&self) -> Result<Vec<String>> {
        let field = c_string("feature_name")?;
        let (mut len, mut names) = (0, ptr::null());
        // SAFETY: the handle is live, `field` is NUL-terminated, and the out-pointers are valid;
        // the returned strings live in a thread-local buffer that is copied before any other
        // XGBoost call.
        check(unsafe {
            ffi::XGDMatrixGetStrFeatureInfo(self.handle, field.as_ptr(), &mut len, &mut names)
        })?;
        // SAFETY: on success `names` points to `len` NUL-terminated strings.
        unsafe { copy_strings(names, len) }
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        // SAFETY: the handle was created by XGDMatrixCreateFromMat and is freed exactly once.
        // A failure to free cannot be reported from drop, so the code is ignored.
        let _ = unsafe { ffi::XGDMatrixFree(self.handle) };
    }
}

/// Predictions: `values` in row-major order with dimensions `shape`, which is
/// `[rows, outputs per row]` (e.g. the class count for `multi:softprob`, 1 for
/// `binary:logistic`).
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// The dimensions of `values`.
    pub shape: Vec<usize>,
    /// The predicted values, row-major.
    pub values: Vec<f32>,
}

/// A gradient-boosted tree model.
#[derive(Debug)]
pub struct Booster {
    handle: ffi::BoosterHandle,
    /// The OpenMP threads XGBoost may use for this booster.
    threads: NonZeroUsize,
}

// SAFETY: a Booster handle is owned exclusively by this value and is not tied to the creating
// thread. It is not `Sync`: prediction writes into a buffer owned by the booster.
unsafe impl Send for Booster {}

impl Booster {
    /// A new, untrained booster using at most `threads` threads.
    ///
    /// `cache` lists the matrices the booster will train on: [`Booster::update`] needs its
    /// training matrix to have been cached here. The model records the cached matrices' feature
    /// names (see [`DMatrix::set_feature_names`]), which must therefore all be the same.
    ///
    /// # Errors
    ///
    /// Returns an error if the cached matrices have different feature names, or XGBoost fails.
    pub fn new(cache: &[&DMatrix], threads: NonZeroUsize) -> Result<Self> {
        let names = match cache.split_first() {
            Some((first, rest)) => {
                let names = first.feature_names()?;
                for other in rest {
                    let other_names = other.feature_names()?;
                    if other_names != names {
                        return Err(Error(format!(
                            "the cached matrices have different feature names: {names:?} and \
                             {other_names:?}"
                        )));
                    }
                }
                names
            }
            None => Vec::new(),
        };
        let handles: Vec<ffi::DMatrixHandle> = cache.iter().map(|m| m.handle).collect();
        let mut handle = ptr::null_mut();
        // SAFETY: `handles` holds live DMatrix handles borrowed for the call; `handle` is a valid
        // out-pointer.
        check(unsafe {
            ffi::XGBoosterCreate(handles.as_ptr(), bst_len(handles.len()), &mut handle)
        })?;
        let booster = Self::with_threads(handle, threads)?;
        if !names.is_empty() {
            booster.set_feature_names(&names)?;
        }
        Ok(booster)
    }

    /// Record the model's feature names. (XGBoost's C API, unlike its Python package, does not
    /// copy them from the training matrix.)
    fn set_feature_names(&self, names: &[String]) -> Result<()> {
        let owned = names.iter().map(|n| c_string(n)).collect::<Result<Vec<_>>>()?;
        let pointers: Vec<*const c_char> = owned.iter().map(|n| n.as_ptr()).collect();
        let field = c_string("feature_name")?;
        // SAFETY: the handle is live; `pointers` holds NUL-terminated strings owned by `owned`,
        // which outlives the call; XGBoost copies them.
        check(unsafe {
            ffi::XGBoosterSetStrFeatureInfo(
                self.handle,
                field.as_ptr(),
                pointers.as_ptr(),
                bst_len(pointers.len()),
            )
        })
    }

    /// Take ownership of a booster handle and limit it to `threads` threads.
    fn with_threads(handle: ffi::BoosterHandle, threads: NonZeroUsize) -> Result<Self> {
        let mut booster = Self { handle, threads };
        booster.set_param("nthread", &threads.to_string())?;
        Ok(booster)
    }

    /// A booster loaded from an XGBoost JSON model (e.g. one written by [`Booster::to_json`]),
    /// using at most `threads` threads.
    ///
    /// # Errors
    ///
    /// Returns an error if `json` is not a JSON object, or not a valid XGBoost model.
    pub fn from_json(json: &[u8], threads: NonZeroUsize) -> Result<Self> {
        // XGBoost takes the first byte as an object's `{` and picks its parser from the first
        // byte after it that is not white space (C's `isspace`): `"` selects JSON, a letter
        // selects the binary (UBJSON) parser, which does not bounds-check its input, and anything
        // else (or an empty buffer) aborts the process. Only hand it what it will parse as JSON.
        let is_space = |byte: &u8| matches!(byte, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r');
        let key = json.get(1..).and_then(|rest| rest.iter().find(|byte| !is_space(byte)));
        if json.first() != Some(&b'{') || key != Some(&b'"') {
            return Err(Error("the model is not a JSON object".into()));
        }
        let booster = Self::new(&[], threads)?;
        // SAFETY: the handle is live and `json` is a readable buffer of `json.len()` bytes,
        // which XGBoost parses and does not retain.
        check(unsafe {
            ffi::XGBoosterLoadModelFromBuffer(
                booster.handle,
                json.as_ptr().cast::<c_void>(),
                bst_len(json.len()),
            )
        })?;
        Ok(booster)
    }

    /// The threads this booster may use.
    #[must_use]
    pub fn threads(&self) -> NonZeroUsize {
        self.threads
    }

    /// Set a training or runtime parameter (e.g. `objective`, `max_depth`, `eta`): a batch of
    /// one (see [`Booster::set_params`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a string contains a NUL byte, or XGBoost rejects the parameter.
    pub fn set_param(&mut self, name: &str, value: &str) -> Result<()> {
        self.set_params(&[(name, value)])
    }

    /// Set training or runtime parameters (e.g. `objective`, `max_depth`, `eta`) together, in
    /// order (a repeated name, e.g. `eval_metric`, adds to the earlier ones).
    ///
    /// XGBoost configures the booster with each batch as it is set, so an invalid value, or a
    /// batch that is invalid on its own (e.g. a multiclass `objective` without its `num_class`),
    /// is an error here; set parameters that depend on each other in one batch. Unknown names
    /// are ignored with a warning.
    ///
    /// # Errors
    ///
    /// Returns an error if a string contains a NUL byte, or XGBoost rejects the parameters.
    pub fn set_params(&mut self, params: &[(&str, &str)]) -> Result<()> {
        let pairs: Vec<[&str; 2]> = params.iter().map(|&(name, value)| [name, value]).collect();
        let config = serde_json::json!({ "params": pairs }).to_string();
        let config = c_string(&config)?;
        // SAFETY: the handle is live and the configuration is NUL-terminated for the call.
        check(unsafe { ffi::XGBoosterSetParams(self.handle, config.as_ptr()) })
    }

    /// Train one boosting round (`iteration`, counting from 0) on `train`, which should have been
    /// in the cache given to [`Booster::new`]: the booster records the cached matrices' feature
    /// names, and XGBoost reuses its predictions for cached matrices between rounds.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails (e.g. `train` has no labels, or a parameter is invalid).
    pub fn update(&mut self, iteration: i32, train: &DMatrix) -> Result<()> {
        // SAFETY: both handles are live for the call.
        check(unsafe { ffi::XGBoosterUpdateOneIter(self.handle, iteration, train.handle) })
    }

    /// Evaluate the configured metrics on named matrices after round `iteration`.
    ///
    /// Returns XGBoost's evaluation line, e.g. `[3]\tvalidation-mlogloss:0.21`.
    ///
    /// # Errors
    ///
    /// Returns an error if a name contains a NUL byte, or XGBoost fails.
    pub fn evaluate(&mut self, iteration: i32, sets: &[(&DMatrix, &str)]) -> Result<String> {
        let names = sets.iter().map(|(_, name)| c_string(name)).collect::<Result<Vec<_>>>()?;
        let mut handles: Vec<ffi::DMatrixHandle> = sets.iter().map(|(m, _)| m.handle).collect();
        let mut name_pointers: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
        let mut result = ptr::null();
        // SAFETY: the handles are live, the names are NUL-terminated and outlive the call, and
        // `result` receives a booster-owned string that is copied before the next call.
        check(unsafe {
            ffi::XGBoosterEvalOneIter(
                self.handle,
                iteration,
                handles.as_mut_ptr(),
                name_pointers.as_mut_ptr(),
                bst_len(handles.len()),
                &mut result,
            )
        })?;
        if result.is_null() {
            return Err(Error("XGBoost returned no evaluation result".into()));
        }
        // SAFETY: on success `result` is a valid NUL-terminated string owned by the booster.
        Ok(unsafe { CStr::from_ptr(result) }.to_string_lossy().into_owned())
    }

    /// The number of features the model was trained on.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn num_features(&self) -> Result<usize> {
        let mut features = 0;
        // SAFETY: the handle is live and `features` is a valid out-pointer.
        check(unsafe { ffi::XGBoosterGetNumFeature(self.handle, &mut features) })?;
        from_bst_len(features, "feature count")
    }

    /// The feature names the model was trained with (empty if the training matrix had none).
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn feature_names(&self) -> Result<Vec<String>> {
        let field = c_string("feature_name")?;
        let (mut len, mut names) = (0, ptr::null());
        // SAFETY: the handle is live, `field` is NUL-terminated, and the out-pointers are valid;
        // the returned strings live in a booster-owned buffer copied before any other call.
        check(unsafe {
            ffi::XGBoosterGetStrFeatureInfo(self.handle, field.as_ptr(), &mut len, &mut names)
        })?;
        // SAFETY: on success `names` points to `len` NUL-terminated strings.
        unsafe { copy_strings(names, len) }
    }

    /// Check that `data` has the model's features: the same number of columns, and the same
    /// names in the same order. (XGBoost itself only checks that there are not more columns than
    /// features, so a column-order mismatch would otherwise silently give wrong predictions.)
    fn check_features(&self, data: &DMatrix) -> Result<()> {
        let (expected, actual) = (self.num_features()?, data.num_cols()?);
        if expected != actual {
            return Err(Error(format!(
                "the model has {expected} features but the matrix has {actual} columns"
            )));
        }
        let (expected, actual) = (self.feature_names()?, data.feature_names()?);
        if expected != actual {
            return Err(Error(format!(
                "the matrix's feature names {actual:?} do not match the model's {expected:?}"
            )));
        }
        Ok(())
    }

    /// Predict on `data` with every tree.
    ///
    /// Returns `[rows, outputs per row]` values: class probabilities for `multi:softprob`, one
    /// probability for `binary:logistic`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` does not have the model's features (the same count and names,
    /// in order; see [`DMatrix::set_feature_names`]), or XGBoost fails.
    pub fn predict(&self, data: &DMatrix) -> Result<Prediction> {
        self.check_features(data)?;
        let config = c_string(
            r#"{"type": 0, "training": false, "iteration_begin": 0, "iteration_end": 0, "strict_shape": true}"#,
        )?;
        let (mut shape, mut dims, mut result) = (ptr::null(), 0, ptr::null());
        // SAFETY: both handles are live, `config` is NUL-terminated, and the out-pointers are
        // valid; the outputs point into a booster-owned buffer that is copied below before any
        // other call on this booster.
        check(unsafe {
            ffi::XGBoosterPredictFromDMatrix(
                self.handle,
                data.handle,
                config.as_ptr(),
                &mut shape,
                &mut dims,
                &mut result,
            )
        })?;
        let dims = from_bst_len(dims, "dimension count")?;
        // SAFETY: on success `shape` points to `dims` lengths.
        let shape: Vec<usize> = unsafe { borrowed_slice(shape, dims) }?
            .iter()
            .map(|&n| from_bst_len(n, "dimension"))
            .collect::<Result<_>>()?;
        let len = shape.iter().try_fold(1usize, |total, &n| total.checked_mul(n));
        let len = len.ok_or_else(|| Error("prediction size overflows usize".into()))?;
        // SAFETY: on success `result` points to the product of `shape` floats.
        let values = unsafe { borrowed_slice(result, len) }?.to_vec();
        // XGBoost reports `[0, 0]` for a matrix with no rows, whatever the model's outputs.
        let shape = if values.is_empty() { vec![0, self.outputs_per_row()?] } else { shape };
        Ok(Prediction { shape, values })
    }

    /// The number of outputs the model predicts per row: the class count of a multi-class model,
    /// else the target count (1 for a single-target model).
    fn outputs_per_row(&self) -> Result<usize> {
        let (mut len, mut data) = (0, ptr::null());
        // SAFETY: the handle is live and the outputs point to a booster-owned buffer that is
        // copied immediately.
        check(unsafe { ffi::XGBoosterSaveJsonConfig(self.handle, &mut len, &mut data) })?;
        let len = from_bst_len(len, "config size")?;
        // SAFETY: on success `data` points to `len` bytes.
        let config: serde_json::Value =
            serde_json::from_slice(unsafe { borrowed_slice(data.cast::<u8>(), len) }?)
                .map_err(|e| Error(format!("XGBoost returned an invalid config: {e}")))?;
        let parameter = |name: &str| -> Result<usize> {
            let value = &config["learner"]["learner_model_param"][name];
            value
                .as_str()
                .and_then(|text| text.parse().ok())
                .ok_or_else(|| Error(format!("the model config has no {name}: {value}")))
        };
        let classes = parameter("num_class")?;
        Ok(if classes > 0 { classes } else { parameter("num_target")?.max(1) })
    }

    /// The model serialized as XGBoost JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        let config = c_string(r#"{"format": "json"}"#)?;
        let (mut len, mut data) = (0, ptr::null());
        // SAFETY: the handle is live, `config` is NUL-terminated, and the outputs point to a
        // booster-owned buffer that is copied immediately.
        check(unsafe {
            ffi::XGBoosterSaveModelToBuffer(self.handle, config.as_ptr(), &mut len, &mut data)
        })?;
        let len = from_bst_len(len, "model size")?;
        // SAFETY: on success `data` points to `len` bytes.
        Ok(unsafe { borrowed_slice(data.cast::<u8>(), len) }?.to_vec())
    }

    /// The number of boosting rounds trained.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn boosted_rounds(&self) -> Result<usize> {
        let mut rounds: c_int = 0;
        // SAFETY: the handle is live and `rounds` is a valid out-pointer.
        check(unsafe { ffi::XGBoosterBoostedRounds(self.handle, &mut rounds) })?;
        usize::try_from(rounds).map_err(|_| Error(format!("negative round count {rounds}")))
    }

    /// A new booster with only rounds `[0, rounds)` (e.g. to keep the best early-stopping
    /// round), using the same thread count.
    ///
    /// # Errors
    ///
    /// Returns an error if `rounds` is 0 or exceeds the rounds trained, or XGBoost fails.
    pub fn truncated(&self, rounds: usize) -> Result<Self> {
        let trained = self.boosted_rounds()?;
        if rounds == 0 || rounds > trained {
            return Err(Error(format!("cannot keep {rounds} of {trained} rounds")));
        }
        let end = c_int::try_from(rounds).map_err(|_| Error("round count overflow".into()))?;
        let mut handle = ptr::null_mut();
        // SAFETY: the handle is live and `handle` is a valid out-pointer for the new booster.
        check(unsafe { ffi::XGBoosterSlice(self.handle, 0, end, 1, &mut handle) })?;
        Self::with_threads(handle, self.threads)
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        // SAFETY: the handle was created by XGBoosterCreate or XGBoosterSlice and is freed
        // exactly once. A failure to free cannot be reported from drop, so the code is ignored.
        let _ = unsafe { ffi::XGBoosterFree(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const TWO: NonZeroUsize = NonZeroUsize::new(2).expect("2 is non-zero");

    /// Runs the check for `runtime` with `-t 2` and the given variables set.
    fn check_with(runtime: OpenMpRuntime, set: &[(&str, &str)]) -> std::result::Result<(), String> {
        let var = |name: &str| {
            set.iter().find(|(set_name, _)| *set_name == name).map(|(_, value)| value.to_string())
        };
        runtime.check_environment(TWO, var)
    }

    #[rstest]
    #[case::unset(&[], None)]
    #[case::dynamic_false(&[("OMP_DYNAMIC", "false")], None)]
    #[case::dynamic_true(&[("OMP_DYNAMIC", "TRUE ")], Some("OMP_DYNAMIC=TRUE "))]
    #[case::dynamic_one(&[("OMP_DYNAMIC", "1")], None)]
    #[case::dynamic_yes(&[("OMP_DYNAMIC", "yes")], None)]
    #[case::limit_enough(&[("OMP_THREAD_LIMIT", "2")], None)]
    #[case::limit_below(&[("OMP_THREAD_LIMIT", "1")], Some("OMP_THREAD_LIMIT=1 is below the 2"))]
    #[case::limit_unparsable(&[("OMP_THREAD_LIMIT", "lots")], None)]
    #[case::kmp_limit_ignored(&[("KMP_DEVICE_THREAD_LIMIT", "1")], None)]
    fn libgomp_environment_must_give_the_full_team(
        #[case] set: &[(&str, &str)],
        #[case] refused: Option<&str>,
    ) {
        let result = check_with(OpenMpRuntime::LibGomp, set);
        match refused {
            None => assert_eq!(result, Ok(())),
            Some(start) => {
                let message = result.unwrap_err();
                assert!(message.starts_with(start), "{message}");
            }
        }
    }

    #[rstest]
    #[case::unset(&[], None)]
    #[case::dynamic_false(&[("OMP_DYNAMIC", "false")], None)]
    #[case::dynamic_zero(&[("OMP_DYNAMIC", "0")], None)]
    #[case::dynamic_no(&[("OMP_DYNAMIC", "no")], None)]
    #[case::dynamic_off(&[("OMP_DYNAMIC", "off")], None)]
    #[case::dynamic_f(&[("OMP_DYNAMIC", "f")], None)]
    #[case::dynamic_o(&[("OMP_DYNAMIC", "o")], None)]
    #[case::dynamic_empty(&[("OMP_DYNAMIC", "")], None)]
    #[case::dynamic_leading_space(&[("OMP_DYNAMIC", " true")], None)]
    #[case::dynamic_true(&[("OMP_DYNAMIC", "TRUE")], Some("OMP_DYNAMIC=TRUE lets"))]
    #[case::dynamic_t(&[("OMP_DYNAMIC", "t")], Some("OMP_DYNAMIC=t lets"))]
    #[case::dynamic_one(&[("OMP_DYNAMIC", "1")], Some("OMP_DYNAMIC=1 lets"))]
    #[case::dynamic_yes(&[("OMP_DYNAMIC", "Yes")], Some("OMP_DYNAMIC=Yes lets"))]
    #[case::dynamic_y(&[("OMP_DYNAMIC", "y")], Some("OMP_DYNAMIC=y lets"))]
    #[case::dynamic_on(&[("OMP_DYNAMIC", "on")], Some("OMP_DYNAMIC=on lets"))]
    #[case::dynamic_fortran(&[("OMP_DYNAMIC", ".T.")], Some("OMP_DYNAMIC=.T. lets"))]
    #[case::dynamic_enabled(&[("OMP_DYNAMIC", "enabled")], Some("OMP_DYNAMIC=enabled lets"))]
    #[case::dynamic_trailing(&[("OMP_DYNAMIC", "true ")], Some("OMP_DYNAMIC=true  lets"))]
    #[case::device_limit_enough(&[("KMP_DEVICE_THREAD_LIMIT", "2")], None)]
    #[case::device_limit_below(
        &[("KMP_DEVICE_THREAD_LIMIT", "1")],
        Some("KMP_DEVICE_THREAD_LIMIT=1 is below the 2")
    )]
    #[case::all_threads_below(&[("KMP_ALL_THREADS", "1")], Some("KMP_ALL_THREADS=1 is below the 2"))]
    #[case::max_threads_ignored(&[("KMP_MAX_THREADS", "1")], None)]
    #[case::limit_below(&[("OMP_THREAD_LIMIT", "1")], Some("OMP_THREAD_LIMIT=1 is below the 2"))]
    fn libomp_environment_must_give_the_full_team(
        #[case] set: &[(&str, &str)],
        #[case] refused: Option<&str>,
    ) {
        let result = check_with(OpenMpRuntime::LibOmp, set);
        match refused {
            None => assert_eq!(result, Ok(())),
            Some(start) => {
                let message = result.unwrap_err();
                assert!(message.starts_with(start), "{message}");
            }
        }
    }

    #[test]
    fn checks_the_environment_of_the_linked_runtime() {
        let expected =
            if cfg!(target_os = "macos") { OpenMpRuntime::LibOmp } else { OpenMpRuntime::LibGomp };
        assert_eq!(OpenMpRuntime::LINKED, expected);
        let var = |name: &str| (name == "OMP_THREAD_LIMIT").then(|| "1".to_string());
        assert!(check_openmp_environment(TWO, var).unwrap_err().starts_with("OMP_THREAD_LIMIT=1"));
    }

    /// A separable two-feature, three-class dataset: the class is set by the first feature.
    fn three_class_data(rows: usize) -> (Vec<f32>, Vec<f32>) {
        let mut features = Vec::with_capacity(rows * 2);
        let mut labels = Vec::with_capacity(rows);
        for i in 0..rows {
            let class = (i % 3) as f32;
            features.push(class * 10.0 + (i % 7) as f32 * 0.1);
            features.push((i % 5) as f32);
            labels.push(class);
        }
        (features, labels)
    }

    /// A labelled, named training matrix from `three_class_data`.
    fn train_matrix(rows: usize) -> DMatrix {
        let (features, labels) = three_class_data(rows);
        let mut train = DMatrix::from_dense(&features, rows, 2, f32::NAN).unwrap();
        train.set_labels(&labels).unwrap();
        train.set_feature_names(&["signal", "noise"]).unwrap();
        train
    }

    fn trained_booster(rounds: i32) -> (Booster, DMatrix) {
        let train = train_matrix(300);
        let mut booster = Booster::new(&[&train], TWO).unwrap();
        booster
            .set_params(&[
                ("objective", "multi:softprob"),
                ("num_class", "3"),
                ("max_depth", "3"),
                ("seed", "1"),
            ])
            .unwrap();
        for iteration in 0..rounds {
            booster.update(iteration, &train).unwrap();
        }
        (booster, train)
    }

    #[rstest]
    #[case::idle(8, 30, Some(1.8), OpenMpWait::Spin)]
    #[case::fills_the_cores(8, 8, Some(0.3), OpenMpWait::Spin)]
    #[case::one_core_short(8, 8, Some(0.6), OpenMpWait::SpinBriefly)]
    #[case::busy(8, 30, Some(25.0), OpenMpWait::SpinBriefly)]
    #[case::twice_the_cores(8, 30, Some(52.0), OpenMpWait::SpinBriefly)]
    #[case::overloaded(8, 30, Some(53.0), OpenMpWait::Sleep)]
    #[case::more_threads_than_cores(16, 8, Some(0.0), OpenMpWait::SpinBriefly)]
    #[case::unknown(8, 30, None, OpenMpWait::SpinBriefly)]
    #[case::invalid(8, 30, Some(f64::NAN), OpenMpWait::SpinBriefly)]
    fn waits_according_to_the_load(
        #[case] threads: usize,
        #[case] cores: usize,
        #[case] busy: Option<f64>,
        #[case] expected: OpenMpWait,
    ) {
        assert_eq!(OpenMpWait::for_load(threads, cores, busy), expected);
    }

    #[test]
    fn describes_the_wait_choice() {
        let choice =
            WaitChoice { wait: OpenMpWait::SpinBriefly, threads: 8, cores: 30, busy: Some(25.04) };
        let text = choice.to_string();
        assert!(text.starts_with("spin briefly, then sleep"), "{text}");
        assert!(text.contains("8 threads, 30 cores, other work 25.0 cores"), "{text}");
        let choice = WaitChoice { wait: OpenMpWait::FromEnvironment("OMP_WAIT_POLICY"), ..choice };
        assert!(choice.to_string().starts_with("as set by OMP_WAIT_POLICY"));
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn measures_the_other_work() {
        let busy = cpu_usage::busy_cores().expect("the CPU time counters are readable");
        assert!(busy.is_finite() && busy >= 0.0, "{busy}");
    }

    #[test]
    fn links_the_vendored_version() {
        let (major, minor, patch) = version();
        assert_eq!(format!("{major}.{minor}.{patch}"), env!("WINNOWR_XGBOOST_VENDORED_VERSION"));
    }

    #[test]
    fn trains_and_predicts_class_probabilities() {
        let (booster, train) = trained_booster(10);
        let prediction = booster.predict(&train).unwrap();
        assert_eq!(prediction.shape, vec![300, 3]);
        for (row, probabilities) in prediction.values.chunks(3).enumerate() {
            let total: f32 = probabilities.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "row {row} sums to {total}");
            let best = (0..3).max_by(|&a, &b| probabilities[a].total_cmp(&probabilities[b]));
            assert_eq!(best, Some(row % 3), "row {row}: {probabilities:?}");
        }
    }

    #[test]
    fn binary_predictions_have_one_output_per_row() {
        let (features, labels) = three_class_data(60);
        let labels: Vec<f32> = labels.iter().map(|&class| f32::from(class > 0.0)).collect();
        let mut train = DMatrix::from_dense(&features, 60, 2, f32::NAN).unwrap();
        train.set_labels(&labels).unwrap();
        let mut booster = Booster::new(&[&train], TWO).unwrap();
        booster.set_param("objective", "binary:logistic").unwrap();
        booster.update(0, &train).unwrap();
        assert_eq!(booster.predict(&train).unwrap().shape, vec![60, 1]);
    }

    #[test]
    fn empty_matrices_predict_no_rows_with_the_class_count() {
        let (booster, _) = trained_booster(2);
        let mut empty = DMatrix::from_dense(&[], 0, 2, f32::NAN).unwrap();
        empty.set_feature_names(&["signal", "noise"]).unwrap();
        let prediction = booster.predict(&empty).unwrap();
        assert!(prediction.values.is_empty());
        assert_eq!(prediction.shape, vec![0, 3]);
    }

    #[test]
    fn empty_matrices_predict_no_rows_with_one_output_for_binary_models() {
        let (features, labels) = three_class_data(30);
        let labels: Vec<f32> = labels.iter().map(|&class| f32::from(class > 0.0)).collect();
        let mut train = DMatrix::from_dense(&features, 30, 2, f32::NAN).unwrap();
        train.set_labels(&labels).unwrap();
        let mut booster = Booster::new(&[&train], TWO).unwrap();
        booster.set_param("objective", "binary:logistic").unwrap();
        booster.update(0, &train).unwrap();
        let empty = DMatrix::from_dense(&[], 0, 2, f32::NAN).unwrap();
        assert_eq!(booster.predict(&empty).unwrap().shape, vec![0, 1]);
    }

    #[test]
    fn json_round_trip_preserves_predictions_features_and_threads() {
        let (booster, train) = trained_booster(5);
        let json = booster.to_json().unwrap();
        let loaded = Booster::from_json(&json, TWO).unwrap();
        assert_eq!(loaded.boosted_rounds().unwrap(), 5);
        assert_eq!(loaded.threads(), TWO);
        assert_eq!(loaded.feature_names().unwrap(), ["signal", "noise"]);
        assert_eq!(loaded.num_features().unwrap(), 2);
        assert_eq!(loaded.predict(&train).unwrap(), booster.predict(&train).unwrap());
    }

    #[rstest]
    #[case::empty(b"")]
    #[case::whitespace(b"  \n")]
    #[case::ubjson(b"[$U#U\x03abc")]
    #[case::truncated(b"{\"learner\": ")]
    #[case::brace_then_garbage(b"{\x00\x01")]
    #[case::not_a_model(b"{}")]
    fn invalid_models_are_rejected_without_aborting(#[case] json: &[u8]) {
        assert!(Booster::from_json(json, TWO).is_err());
    }

    #[rstest]
    #[case::empty(b"")]
    #[case::whitespace(b"  \n")]
    #[case::ubjson_array(b"[$U#U\x03abc")]
    #[case::ubjson_object(b"{L\x00\x00\x00\x00\x00\x00\x00\x07learner")]
    #[case::ubjson_object_after_space(b"{ \x0bL\x00\x00\x00\x00\x00\x00\x00\x07learner")]
    #[case::letters(b"{abc}")]
    #[case::leading_space(b" {\"learner\": {}}")]
    #[case::brace_then_garbage(b"{\x00\x01")]
    #[case::brace_only(b"{")]
    fn non_json_models_are_rejected_before_reaching_xgboost(#[case] json: &[u8]) {
        let error = Booster::from_json(json, TWO).unwrap_err();
        assert_eq!(error.message(), "the model is not a JSON object");
    }

    #[test]
    fn truncation_keeps_leading_rounds_and_threads() {
        let (booster, train) = trained_booster(6);
        let (two_rounds, _) = trained_booster(2);
        let truncated = booster.truncated(2).unwrap();
        assert_eq!(truncated.boosted_rounds().unwrap(), 2);
        assert_eq!(truncated.threads(), TWO);
        assert_eq!(truncated.predict(&train).unwrap(), two_rounds.predict(&train).unwrap());
        assert_eq!(booster.truncated(0).unwrap_err().message(), "cannot keep 0 of 6 rounds");
        assert_eq!(booster.truncated(7).unwrap_err().message(), "cannot keep 7 of 6 rounds");
    }

    #[test]
    fn evaluates_every_named_set() {
        let (mut booster, train) = trained_booster(3);
        let held_out = train_matrix(30);
        booster.set_param("eval_metric", "mlogloss").unwrap();
        let line = booster.evaluate(2, &[(&train, "train"), (&held_out, "held-out")]).unwrap();
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "{line}");
        assert_eq!(fields[0], "[2]");
        assert!(fields[1].starts_with("train-mlogloss:"), "{line}");
        assert!(fields[2].starts_with("held-out-mlogloss:"), "{line}");
    }

    #[test]
    fn weights_change_the_trained_model() {
        let (features, labels) = three_class_data(90);
        let first_round = |weights: Option<&[f32]>| {
            let mut matrix = DMatrix::from_dense(&features, 90, 2, f32::NAN).unwrap();
            matrix.set_labels(&labels).unwrap();
            if let Some(weights) = weights {
                matrix.set_weights(weights).unwrap();
            }
            let mut booster = Booster::new(&[&matrix], TWO).unwrap();
            booster.set_params(&[("objective", "multi:softprob"), ("num_class", "3")]).unwrap();
            booster.update(0, &matrix).unwrap();
            booster.predict(&matrix).unwrap()
        };
        // Up-weight class 0, so the first round raises row 0's class-0 probability further.
        let weights: Vec<f32> = (0..90).map(|i| if i % 3 == 0 { 20.0 } else { 1.0 }).collect();
        let (unweighted, weighted) = (first_round(None), first_round(Some(&weights)));
        assert!(weighted.values[0] > unweighted.values[0], "{weighted:?} vs {unweighted:?}");
    }

    #[test]
    fn missing_sentinel_values_predict_like_nan() {
        let (booster, _) = trained_booster(5);
        let sentinel = -999.0;
        let mut with_nan = DMatrix::from_dense(&[f32::NAN, 1.0], 1, 2, f32::NAN).unwrap();
        let mut with_sentinel = DMatrix::from_dense(&[sentinel, 1.0], 1, 2, sentinel).unwrap();
        let mut with_value = DMatrix::from_dense(&[sentinel, 1.0], 1, 2, f32::NAN).unwrap();
        for matrix in [&mut with_nan, &mut with_sentinel, &mut with_value] {
            matrix.set_feature_names(&["signal", "noise"]).unwrap();
        }
        let missing = booster.predict(&with_nan).unwrap();
        assert_eq!(booster.predict(&with_sentinel).unwrap(), missing);
        assert_ne!(booster.predict(&with_value).unwrap(), missing);
    }

    #[test]
    fn matrices_do_not_depend_on_their_threads() {
        let (booster, _) = trained_booster(3);
        let mut features = three_class_data(1000).0;
        features[7] = f32::NAN;
        let predict = |threads: NonZeroUsize| {
            let mut matrix =
                DMatrix::from_dense_with_threads(&features, 1000, 2, f32::NAN, threads).unwrap();
            assert_eq!(matrix.num_rows().unwrap(), 1000);
            matrix.set_feature_names(&["signal", "noise"]).unwrap();
            booster.predict(&matrix).unwrap().values
        };
        let one = predict(NonZeroUsize::MIN);
        assert_eq!(predict(NonZeroUsize::new(4).unwrap()), one);
        let mut plain = DMatrix::from_dense(&features, 1000, 2, f32::NAN).unwrap();
        plain.set_feature_names(&["signal", "noise"]).unwrap();
        assert_eq!(booster.predict(&plain).unwrap().values, one);
    }

    #[test]
    fn predict_rejects_mismatched_feature_names() {
        let (booster, _) = trained_booster(3);
        let mut swapped = DMatrix::from_dense(&[1.0, 10.0], 1, 2, f32::NAN).unwrap();
        swapped.set_feature_names(&["noise", "signal"]).unwrap();
        let error = booster.predict(&swapped).unwrap_err();
        assert!(error.message().starts_with("the matrix's feature names"), "{error}");

        let unnamed = DMatrix::from_dense(&[10.0, 1.0], 1, 2, f32::NAN).unwrap();
        let error = booster.predict(&unnamed).unwrap_err();
        assert!(error.message().starts_with("the matrix's feature names []"), "{error}");
    }

    #[test]
    fn predict_rejects_mismatched_column_counts() {
        let (booster, _) = trained_booster(3);
        let mut narrow = DMatrix::from_dense(&[10.0], 1, 1, f32::NAN).unwrap();
        narrow.set_feature_names(&["signal"]).unwrap();
        let error = booster.predict(&narrow).unwrap_err();
        assert_eq!(error.message(), "the model has 2 features but the matrix has 1 columns");
    }

    #[test]
    fn rejects_mismatched_shapes_and_lengths() {
        let error = DMatrix::from_dense(&[1.0, 2.0, 3.0], 2, 2, f32::NAN).unwrap_err();
        assert_eq!(error.message(), "3 values do not fill 2 rows of 2 columns");
        let mut matrix = DMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2, f32::NAN).unwrap();
        assert_eq!(matrix.set_labels(&[1.0]).unwrap_err().message(), "1 labels for 2 rows");
        let error = matrix.set_weights(&[1.0, 1.0, 1.0]).unwrap_err();
        assert_eq!(error.message(), "3 weights for 2 rows");
        let error = matrix.set_feature_names(&["only_one"]).unwrap_err();
        assert_eq!(error.message(), "1 feature names for 2 columns");
        assert_eq!((matrix.num_rows().unwrap(), matrix.num_cols().unwrap()), (2, 2));
    }

    #[test]
    fn invalid_parameters_are_rejected_when_set() {
        let train = train_matrix(30);
        let new_booster = || Booster::new(&[&train], TWO).unwrap();
        let error = new_booster().set_param("max_depth", "not-a-number").unwrap_err();
        assert!(error.message().contains("max_depth"), "{error}");
        assert!(!error.message().contains("Stack trace"), "{error}");
        // A multiclass objective needs its class count in the same batch.
        let error = new_booster().set_param("objective", "multi:softprob").unwrap_err();
        assert!(error.message().contains("num_class"), "{error}");
        let mut booster = new_booster();
        booster.set_params(&[("objective", "multi:softprob"), ("num_class", "3")]).unwrap();
        booster.update(0, &train).unwrap();
    }

    #[test]
    fn cached_matrices_must_share_feature_names() {
        let train = train_matrix(30);
        let mut other = train_matrix(30);
        other.set_feature_names(&["a", "b"]).unwrap();
        let error = Booster::new(&[&train, &other], TWO).unwrap_err();
        assert!(error.message().starts_with("the cached matrices have different"), "{error}");
    }

    #[test]
    fn training_on_an_uncached_matrix_records_no_feature_names() {
        let train = train_matrix(30);
        let mut booster = Booster::new(&[], TWO).unwrap();
        booster.set_params(&[("objective", "multi:softprob"), ("num_class", "3")]).unwrap();
        booster.update(0, &train).unwrap();
        assert_eq!(booster.num_features().unwrap(), 2);
        assert!(booster.feature_names().unwrap().is_empty());
    }

    #[rstest]
    #[case::location_and_stack(
        "[12:00:00] /build/src/learner.cc:123: Invalid max_depth\nStack trace:\n  [bt] (0) x",
        "Invalid max_depth"
    )]
    #[case::location_only("/build/src/data.cc:9: Check failed: rows", "Check failed: rows")]
    #[case::plain("Invalid input", "Invalid input")]
    #[case::colon_in_message("Value: abc is not a number", "Value: abc is not a number")]
    fn error_messages_lose_build_paths_and_stack_traces(#[case] raw: &str, #[case] clean: &str) {
        assert_eq!(clean_message(raw), clean);
    }
}
