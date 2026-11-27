//! A statically linked XGBoost with the small safe API that winnowr needs.
//!
//! XGBoost is built from the vendored sources (see `build.rs`) and called through its C API.
//! Only the calls winnowr uses are bound: dense matrices with labels and feature names;
//! boosters that train one round at a time, evaluate, predict, truncate, and save or load JSON.
//!
//! Every booster is given an explicit thread count, which bounds the OpenMP threads XGBoost uses
//! for training and prediction (XGBoost's own default is every core). The bound is per call: a
//! caller that predicts from several threads at once (e.g. rayon workers, each with its own
//! booster, or sharing one through [`Booster::shared_predictor`]) uses up to their sum, so it
//! should divide its thread budget between them.
//!
//! A [`Booster`] is used from one thread at a time, except that a tree booster can predict from
//! several at once through a [`SharedPredictor`]. XGBoost returns its outputs (predictions,
//! strings, error messages) in buffers of the calling thread, valid until that thread's next
//! XGBoost call; each is copied before then.
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
//! booster.set_params(&[("objective", "binary:logistic")]).unwrap();
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

/// Measures how much work other processes give the cores this process may use.
///
/// The CPU time counters, sampled over a short window, say how many cores are busy now, but
/// cannot see past every core being busy. The load average (runnable threads) can, but it trails
/// the machine's load by about a minute: right after one training run ends it still counts that
/// run's threads, and the next run would take an idle machine for a busy one. So the load
/// average is used only when the counters show every core busy.
///
/// On Linux the counters are read for the CPUs the process may run on (its affinity, e.g. a
/// cpuset or SLURM binding), and the result is scaled to the cores it is allowed
/// (`available_parallelism`, which also applies a cgroup CPU quota, e.g. a container's CPU
/// limit), assuming the other work is spread evenly over the CPUs. The load average counts the
/// whole machine, so it is scaled from the machine's CPUs to the allowed cores the same way.
mod cpu_usage {
    use std::ffi::c_int;
    use std::time::Duration;

    unsafe extern "C" {
        /// The system load averages (POSIX; in the C library on macOS and Linux).
        fn getloadavg(loadavg: *mut f64, nelem: c_int) -> c_int;
    }

    /// How long the CPU time counters are sampled over.
    const WINDOW: Duration = Duration::from_millis(250);

    /// Busy and total CPU time (in ticks, summed over the CPUs measured), the ids of the CPUs
    /// measured, and the machine's CPU count.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Ticks {
        pub busy: u64,
        pub total: u64,
        pub cpu_ids: Vec<usize>,
        pub host_cpus: usize,
    }

    /// How many of its `cores` cores' worth of work other processes have, while this process is
    /// idle (it calls this before starting any work), or `None` if the counters cannot be read.
    pub fn busy_cores(cores: usize) -> Option<f64> {
        let before = cpu_ticks()?;
        std::thread::sleep(WINDOW);
        let after = cpu_ticks()?;
        let mut load = [0.0f64; 1];
        // SAFETY: the pointer is valid for writing the one requested value.
        let read = unsafe { getloadavg(load.as_mut_ptr(), 1) } == 1;
        busy_between(before, after, cores, read.then_some(load[0]))
    }

    /// The other work on `cores` cores between two readings of the counters, falling back to the
    /// (machine-wide) load average when every CPU measured was busy.
    pub(crate) fn busy_between(
        before: Ticks,
        after: Ticks,
        cores: usize,
        load: Option<f64>,
    ) -> Option<f64> {
        // The CPUs measured changed (e.g. the affinity), even if not their number.
        if before.cpu_ids != after.cpu_ids {
            return None;
        }
        let total = after.total.checked_sub(before.total).filter(|&t| t > 0)?;
        let busy = after.busy.checked_sub(before.busy)?.min(total);
        let cpus = after.cpu_ids.len() as f64;
        let busy = cpus * busy as f64 / total as f64;
        // From the CPUs measured to the cores allowed (fewer under a CPU quota).
        let scale = (cores as f64 / cpus).min(1.0);
        // Every CPU is busy when the busy count rounds up to them all and is at least 90% of
        // them (so that half of a single CPU is not).
        if busy < (cpus - 0.5).max(0.9 * cpus) {
            return Some(busy * scale);
        }
        let load = load
            .filter(|load| load.is_finite())
            .map(|load| load * (cores as f64 / after.host_cpus.max(1) as f64).min(1.0));
        Some(load.map_or(busy * scale, |load| (busy * scale).max(load)))
    }

    /// The busy and total CPU time of the CPUs the process may run on (all of them if its
    /// affinity cannot be read), from `/proc/stat` and `/proc/self/status`.
    #[cfg(target_os = "linux")]
    fn cpu_ticks() -> Option<Ticks> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let allowed = std::fs::read_to_string("/proc/self/status").ok().and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
                .and_then(|list| parse_cpu_list(list.trim()))
        });
        stat_ticks(&stat, allowed.as_deref())
    }

    /// The busy and total CPU time in a `/proc/stat`, of the CPUs `allowed` (all if `None`).
    #[cfg(any(target_os = "linux", test))]
    pub(crate) fn stat_ticks(stat: &str, allowed: Option<&[usize]>) -> Option<Ticks> {
        // user nice system idle iowait irq softirq steal (guest time is counted in user).
        let busy_total = |fields: &str| -> Option<(u64, u64)> {
            let ticks: Vec<u64> =
                fields.split_whitespace().take(8).map(str::parse).collect::<Result<_, _>>().ok()?;
            let (total, idle) = (ticks.iter().sum::<u64>(), ticks.get(3)? + ticks.get(4)?);
            Some((total.checked_sub(idle)?, total))
        };
        let per_cpu: Vec<(usize, &str)> = stat
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("cpu")?;
                let digits = rest.find(|c: char| !c.is_ascii_digit())?;
                Some((rest[..digits].parse().ok()?, &rest[digits..]))
            })
            .collect();
        let host_cpus = per_cpu.len().max(1);
        // The whole machine: with no affinity, or one naming none of the CPUs listed (e.g. a
        // container whose /proc/stat numbers its CPUs from 0).
        let whole = || {
            let first = stat.lines().next()?.strip_prefix("cpu ")?;
            let (busy, total) = busy_total(first)?;
            let cpu_ids = per_cpu.iter().map(|(cpu, _)| *cpu).collect();
            Some(Ticks { busy, total, cpu_ids, host_cpus })
        };
        let Some(allowed) = allowed.filter(|allowed| !allowed.is_empty()) else { return whole() };
        let (mut busy, mut total, mut cpu_ids) = (0u64, 0u64, Vec::new());
        for (cpu, fields) in per_cpu.iter().filter(|(cpu, _)| allowed.contains(cpu)) {
            let (cpu_busy, cpu_total) = busy_total(fields)?;
            busy += cpu_busy;
            total += cpu_total;
            cpu_ids.push(*cpu);
        }
        if cpu_ids.is_empty() {
            return whole();
        }
        Some(Ticks { busy, total, cpu_ids, host_cpus })
    }

    /// Parses a Linux CPU list, e.g. `0-3,8,10-11`.
    #[cfg(any(target_os = "linux", test))]
    pub(crate) fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
        let mut cpus = Vec::new();
        for part in list.split(',').filter(|part| !part.is_empty()) {
            match part.split_once('-') {
                Some((first, last)) => {
                    let (first, last): (usize, usize) = (first.parse().ok()?, last.parse().ok()?);
                    if last < first || last - first > 1 << 16 {
                        return None;
                    }
                    cpus.extend(first..=last);
                }
                None => cpus.push(part.parse().ok()?),
            }
        }
        Some(cpus)
    }

    /// The machine's busy and total CPU time (in ticks, summed over CPUs), and its CPU count,
    /// from the Mach host statistics.
    #[cfg(target_os = "macos")]
    fn cpu_ticks() -> Option<Ticks> {
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
        Some(Ticks {
            busy: user + system + nice,
            total: user + system + idle + nice,
            cpu_ids: (0..cpus).collect(),
            host_cpus: cpus,
        })
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn cpu_ticks() -> Option<Ticks> {
        None
    }
}

/// The CPU time this process has used so far, user and system, summed over all its threads
/// (XGBoost's and htslib's included), or `None` where it cannot be read (`getrusage`, on macOS
/// and Linux only). The difference of two readings is the CPU time the whole process used
/// between them, whichever threads used it; winnowr logs it as it times its phases.
///
/// This has nothing to do with XGBoost: it is here because reading it is a C call, and this is
/// winnowr's only crate that allows `unsafe` code, where the other operating-system probes (of
/// the CPUs' load, for how OpenMP threads wait) already are.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[must_use]
pub fn process_cpu_time() -> Option<std::time::Duration> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: the pointer is valid for writing the one `struct rusage` that getrusage writes.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: getrusage succeeded, so it wrote the whole struct.
    let usage = unsafe { usage.assume_init() };
    let duration = |time: libc::timeval| {
        let seconds = u64::try_from(time.tv_sec).ok()?;
        let micros = u64::try_from(time.tv_usec).ok()?;
        Some(std::time::Duration::from_secs(seconds) + std::time::Duration::from_micros(micros))
    };
    Some(duration(usage.ru_utime)? + duration(usage.ru_stime)?)
}

/// The CPU time this process has used so far: `None`, as it is read on macOS and Linux only.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[must_use]
pub fn process_cpu_time() -> Option<std::time::Duration> {
    None
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
    pub(crate) fn for_load(threads: usize, cores: usize, busy: Option<f64>) -> Self {
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
/// follows the other work on the CPUs this process may use: the cores busy over 250 ms, from the
/// CPU time counters, or the load average when every core is busy (on Linux, measured on the
/// process's affinity and scaled to its allowed cores). With `threads` plus that work
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
    let busy = cpu_usage::busy_cores(cores);
    let wait = set_wait(OpenMpWait::for_load(threads, cores, busy));
    WaitChoice { wait, threads, cores, busy }
}

/// Checks that the OpenMP environment gives XGBoost's training the `threads` threads requested.
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
///   is safe, but the model depends on the thread count, so it would not be the model the
///   requested threads train; the limit is fixed at start.
///
/// All are refused up front. Prediction splits its work as the team's threads arrive, so it
/// needs none of these checks.
///
/// `var` looks up an environment variable (`std::env::var` outside tests).
///
/// # Errors
///
/// Returns an error whose [`Error::openmp_problem`] names the variable to unset or change.
pub fn check_openmp_environment(
    threads: NonZeroUsize,
    var: impl Fn(&str) -> Option<String>,
) -> Result<()> {
    OpenMpRuntime::LINKED.check_environment(threads, var).map_err(Error::from)
}

/// The variable, if any, that sets a thread limit of the linked OpenMP runtime that XGBoost does
/// not read: libomp's `KMP_DEVICE_THREAD_LIMIT` or `KMP_ALL_THREADS` (macOS); libgomp has none.
///
/// The limit is process-wide. [`check_openmp_environment`] refuses one below the threads a fit
/// requests, but building a multi-class model's trees at the same time (the gbtree parameter
/// `parallel_target_trees`) runs several OpenMP teams at once, whose threads together can exceed a
/// limit at or a little above the threads requested; the runtime then shrinks teams, and XGBoost's
/// training loses rows. Do not set `parallel_target_trees` when this returns a variable.
///
/// `var` looks up an environment variable (`std::env::var` outside tests).
#[must_use]
pub fn hidden_thread_limit(var: impl Fn(&str) -> Option<String>) -> Option<&'static str> {
    OpenMpRuntime::LINKED.hidden_thread_limits().iter().copied().find(|&name| var(name).is_some())
}

/// Why the OpenMP environment cannot give XGBoost's training the threads requested (see
/// [`check_openmp_environment`]).
///
/// Its [`Display`](fmt::Display) is [`OpenMpProblem::explanation`] followed by what to do, in
/// terms of the threads requested; a caller can instead follow the explanation with advice in its
/// own terms (e.g. naming its thread option). Adding a variant is a breaking change, so callers
/// can match every variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenMpProblem {
    /// `OMP_DYNAMIC` has a value the runtime reads as true.
    Dynamic {
        /// The value.
        value: String,
    },
    /// A thread limit of the runtime that XGBoost does not read (e.g. libomp's
    /// `KMP_DEVICE_THREAD_LIMIT`) is below the threads requested.
    HiddenThreadLimit {
        /// The variable.
        variable: &'static str,
        /// Its limit.
        limit: usize,
        /// The threads requested.
        threads: NonZeroUsize,
    },
    /// `OMP_THREAD_LIMIT` is below the threads requested, so XGBoost would use fewer.
    ThreadLimit {
        /// The limit.
        limit: usize,
        /// The threads requested.
        threads: NonZeroUsize,
    },
}

/// `n` threads, as "1 thread" or "`n` threads".
fn n_threads(n: usize) -> String {
    if n == 1 { "1 thread".to_string() } else { format!("{n} threads") }
}

impl OpenMpProblem {
    /// What is wrong and what it does to training, without what to do about it.
    #[must_use]
    pub fn explanation(&self) -> String {
        /// Why OpenMP giving XGBoost fewer threads than it asks for is a problem.
        const UNHANDLED: &str =
            "which XGBoost's training does not handle (it can lose rows or crash)";
        match self {
            Self::Dynamic { value } => format!(
                "OMP_DYNAMIC={value} lets OpenMP give XGBoost fewer threads than it asks for, \
                 {UNHANDLED}"
            ),
            Self::HiddenThreadLimit { variable, limit, threads } => format!(
                "{variable}={limit} is below the {} requested, so OpenMP would give XGBoost \
                 fewer threads than it asks for, {UNHANDLED}",
                n_threads(threads.get())
            ),
            Self::ThreadLimit { limit, threads } => format!(
                "OMP_THREAD_LIMIT={limit} is below the {} requested, so XGBoost would run with \
                 {}",
                n_threads(threads.get()),
                n_threads(*limit)
            ),
        }
    }
}

impl fmt::Display for OpenMpProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let explanation = self.explanation();
        match self {
            Self::Dynamic { .. } => {
                write!(f, "{explanation}; unset OMP_DYNAMIC or set it to false")
            }
            Self::HiddenThreadLimit { variable, limit, .. } => write!(
                f,
                "{explanation}; request {} or fewer, or raise or unset {variable}",
                n_threads(*limit)
            ),
            Self::ThreadLimit { limit, threads } => write!(
                f,
                "{explanation}, and a model trained so is not the one {} would train; request {} \
                 or fewer, or raise or unset OMP_THREAD_LIMIT",
                n_threads(threads.get()),
                n_threads(*limit)
            ),
        }
    }
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
    ) -> std::result::Result<(), OpenMpProblem> {
        if let Some(value) = var("OMP_DYNAMIC").filter(|value| self.reads_as_true(value)) {
            return Err(OpenMpProblem::Dynamic { value });
        }
        let limit = |name: &str| var(name).and_then(|value| value.trim().parse::<usize>().ok());
        for &variable in self.hidden_thread_limits() {
            if let Some(limit) = limit(variable).filter(|&limit| limit < threads.get()) {
                return Err(OpenMpProblem::HiddenThreadLimit { variable, limit, threads });
            }
        }
        match limit("OMP_THREAD_LIMIT") {
            Some(limit) if limit < threads.get() => {
                Err(OpenMpProblem::ThreadLimit { limit, threads })
            }
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

/// An error from XGBoost (its last-error message), from invalid arguments, or from an OpenMP
/// environment that training cannot run in (see [`Error::openmp_problem`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// The message.
    message: String,
    /// What is wrong with the OpenMP environment, for an error of
    /// [`check_openmp_environment`].
    openmp_problem: Option<OpenMpProblem>,
}

impl Error {
    /// An error with a message.
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), openmp_problem: None }
    }

    /// The error message.
    #[cfg(test)]
    fn message(&self) -> &str {
        &self.message
    }

    /// What is wrong with the OpenMP environment, if this is an error of
    /// [`check_openmp_environment`], so that a caller can explain it in its own terms (e.g. name
    /// its own thread option).
    #[must_use]
    pub fn openmp_problem(&self) -> Option<&OpenMpProblem> {
        self.openmp_problem.as_ref()
    }
}

impl From<OpenMpProblem> for Error {
    fn from(problem: OpenMpProblem) -> Self {
        Self { message: problem.to_string(), openmp_problem: Some(problem) }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "XGBoost: {}", self.message)
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
    Err(Error::new(clean_message(&message.to_string_lossy())))
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
    CString::new(text).map_err(|_| Error::new(format!("text contains a NUL byte: {text:?}")))
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
    usize::try_from(len).map_err(|_| Error::new(format!("{what} {len} overflows usize")))
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
        return Err(Error::new(format!("XGBoost returned a null pointer for {len} values")));
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
                return Err(Error::new("XGBoost returned a null string"));
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

/// A dense, in-memory data matrix (rows of `f32` features) with optional labels and feature names.
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
            return Err(Error::new(format!(
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
    pub(crate) fn num_rows(&self) -> Result<usize> {
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
    pub(crate) fn num_cols(&self) -> Result<usize> {
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

    /// Set a per-row float field after checking its length.
    fn set_float_info(&mut self, field: &str, values: &[f32]) -> Result<()> {
        let rows = self.num_rows()?;
        if values.len() != rows {
            return Err(Error::new(format!("{} {field}s for {rows} rows", values.len())));
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
            return Err(Error::new(format!("{} feature names for {cols} columns", names.len())));
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
    pub(crate) fn feature_names(&self) -> Result<Vec<String>> {
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
// thread. It is not `Sync`, as some of XGBoost's calls on it (e.g. setting a parameter) must not
// run while another is; prediction alone may (see [`SharedPredictor`]).
unsafe impl Send for Booster {}

/// A booster shared between threads to predict with: XGBoost's prediction is thread safe for
/// tree boosters (`gbtree` and `dart`), each thread's output going to a buffer of its own. While
/// it is borrowed the booster cannot be trained or have its parameters set. Each prediction uses
/// the booster's threads (see [`Booster::set_threads`]), so threads predicting at once should
/// each give it one.
#[derive(Debug, Clone, Copy)]
pub struct SharedPredictor<'a>(&'a Booster);

// SAFETY: only `Booster::predict` is reachable through a `SharedPredictor`, and XGBoost documents
// prediction as thread safe for tree boosters (`doc/prediction.rst`, "Thread Safety"), which
// `Booster::shared_predictor` checks. `predict` makes these C API calls, each of which may run on
// several threads at once on one booster:
// * `XGBoosterGetNumFeature` and `XGBoosterGetStrFeatureInfo` (in `check_features`): the former
//   reads the feature count; the latter copies the feature names into the calling thread's
//   buffer (`Learner::GetThreadLocal`, a map from booster to buffers in thread-local storage).
// * `XGBoosterPredictFromDMatrix`: predicts into the calling thread's buffer, with the prediction
//   cache locked (`DMatrixCache`).
// * `XGBoosterSaveJsonConfig` (in `outputs_per_row`, for a matrix of no rows): writes the config
//   to the calling thread's buffer.
// Each first calls `Learner::Configure`, which does nothing on a configured booster and otherwise
// configures it under a lock (`config_lock_`); `shared_predictor` configures it beforehand.
// Errors go to the calling thread's last-error message (`XGBGetLastError`). Nothing can set a
// parameter of the booster, or train it, while it is borrowed.
unsafe impl Sync for SharedPredictor<'_> {}

impl SharedPredictor<'_> {
    /// The threads each prediction uses (see [`Booster::threads`]).
    #[must_use]
    pub fn threads(&self) -> NonZeroUsize {
        self.0.threads
    }

    /// Predicts on `data`, as [`Booster::predict`] does; any number of threads may at once.
    ///
    /// # Errors
    ///
    /// Returns an error as [`Booster::predict`] does.
    pub fn predict(&self, data: &DMatrix) -> Result<Prediction> {
        self.0.predict(data)
    }
}

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
                        return Err(Error::new(format!(
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
            return Err(Error::new("the model is not a JSON object"));
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

    /// Lets this booster use at most `threads` threads from now on (e.g. to share the threads with
    /// other work for a while). Its predictions do not change.
    ///
    /// # Errors
    ///
    /// Returns an error if the count is above what XGBoost takes (`c_int::MAX`), the booster then
    /// unchanged, or XGBoost rejects it (see [`Booster::set_params`] for the threads then).
    pub fn set_threads(&mut self, threads: NonZeroUsize) -> Result<()> {
        self.set_param("nthread", &threads.to_string())
    }

    /// A handle to predict with from several threads at once (see [`SharedPredictor`]), if this
    /// is a tree booster (`gbtree` or `dart`); `None` for another (e.g. `gblinear`), whose
    /// prediction XGBoost does not document as thread safe.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn shared_predictor(&self) -> Result<Option<SharedPredictor<'_>>> {
        if !matches!(self.gradient_booster_name()?.as_str(), "gbtree" | "dart") {
            return Ok(None);
        }
        // Configured now, before any thread predicts (XGBoost also locks it).
        self.num_features()?;
        Ok(Some(SharedPredictor(self)))
    }

    /// The name of the booster's kind (e.g. `gbtree`, `gblinear`), as its config records it.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails or its config has no booster name.
    fn gradient_booster_name(&self) -> Result<String> {
        let config = self.config()?;
        let name = &config["learner"]["gradient_booster"]["name"];
        name.as_str()
            .map(str::to_string)
            .ok_or_else(|| Error::new(format!("the model config has no booster name: {name}")))
    }

    /// Set a training or runtime parameter (e.g. `objective`, `max_depth`, `eta`): a batch of
    /// one (see [`Booster::set_params`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a string contains a NUL byte, or XGBoost rejects the parameter.
    pub(crate) fn set_param(&mut self, name: &str, value: &str) -> Result<()> {
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
    /// An `nthread` changes the booster's threads, as [`Booster::set_threads`] does, so that
    /// [`Booster::threads`] stays the count XGBoost uses. It must be a positive count: XGBoost
    /// takes 0 or less for every thread it can, which [`Booster::threads`] cannot record. Its
    /// alias `n_jobs` is refused (set `nthread`), leaving one name to track.
    ///
    /// # Errors
    ///
    /// Returns an error if a string contains a NUL byte, an `nthread` is not a positive count of at
    /// most `c_int::MAX` or `n_jobs` is set (the booster unchanged), or XGBoost rejects the parameters (the booster's
    /// threads are then the larger of its count and the batch's `nthread`, as XGBoost may use
    /// either).
    pub fn set_params(&mut self, params: &[(&str, &str)]) -> Result<()> {
        // Checked before XGBoost sees the batch, so a refused count changes nothing.
        let threads = threads_set_by(params)?;
        let pairs: Vec<[&str; 2]> = params.iter().map(|&(name, value)| [name, value]).collect();
        let config = serde_json::json!({ "params": pairs }).to_string();
        let config = c_string(&config)?;
        // SAFETY: the handle is live and the configuration is NUL-terminated for the call.
        let set = check(unsafe { ffi::XGBoosterSetParams(self.handle, config.as_ptr()) });
        if let Some(threads) = threads {
            // XGBoost takes up the count before it configures the objective, booster and metrics,
            // which can then fail: on a failure the booster may use either count, so it counts
            // the larger.
            self.threads = if set.is_ok() { threads } else { threads.max(self.threads) };
        }
        set
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
        // `result` receives this thread's string for the booster, copied before the next XGBoost
        // call on this thread.
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
            return Err(Error::new("XGBoost returned no evaluation result"));
        }
        // SAFETY: on success `result` is a valid NUL-terminated string owned by the booster.
        Ok(unsafe { CStr::from_ptr(result) }.to_string_lossy().into_owned())
    }

    /// The number of features the model was trained on.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub(crate) fn num_features(&self) -> Result<usize> {
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
        // the returned strings live in this thread's buffer for the booster, copied below before
        // the next XGBoost call on this thread.
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
            return Err(Error::new(format!(
                "the model has {expected} features but the matrix has {actual} columns"
            )));
        }
        let (expected, actual) = (self.feature_names()?, data.feature_names()?);
        if expected != actual {
            return Err(Error::new(format!(
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
        // valid; the outputs point into this thread's buffer for the booster, copied below before
        // the next XGBoost call on this thread (other threads predicting at once have their own;
        // see `SharedPredictor`).
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
        let len = len.ok_or_else(|| Error::new("prediction size overflows usize"))?;
        // SAFETY: on success `result` points to the product of `shape` floats.
        let values = unsafe { borrowed_slice(result, len) }?.to_vec();
        // XGBoost reports `[0, 0]` for a matrix with no rows, whatever the model's outputs.
        let shape = if values.is_empty() { vec![0, self.outputs_per_row()?] } else { shape };
        Ok(Prediction { shape, values })
    }

    /// The number of outputs the model predicts per row: the class count of a multi-class model,
    /// else the target count (1 for a single-target model), as the model's config records it.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails or its config has no class or target count.
    pub fn outputs_per_row(&self) -> Result<usize> {
        let config = self.config()?;
        let parameter = |name: &str| -> Result<usize> {
            let value = &config["learner"]["learner_model_param"][name];
            value
                .as_str()
                .and_then(|text| text.parse().ok())
                .ok_or_else(|| Error::new(format!("the model config has no {name}: {value}")))
        };
        let classes = parameter("num_class")?;
        Ok(if classes > 0 { classes } else { parameter("num_target")?.max(1) })
    }

    /// The booster's configuration (its learner and training parameters), as XGBoost's JSON
    /// config.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails or returns invalid JSON.
    fn config(&self) -> Result<serde_json::Value> {
        let (mut len, mut data) = (0, ptr::null());
        // SAFETY: the handle is live and the outputs point to this thread's buffer for the
        // booster, copied immediately (before the next XGBoost call on this thread).
        check(unsafe { ffi::XGBoosterSaveJsonConfig(self.handle, &mut len, &mut data) })?;
        let len = from_bst_len(len, "config size")?;
        // SAFETY: on success `data` points to `len` bytes.
        serde_json::from_slice(unsafe { borrowed_slice(data.cast::<u8>(), len) }?)
            .map_err(|e| Error::new(format!("XGBoost returned an invalid config: {e}")))
    }

    /// The model serialized as XGBoost JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if XGBoost fails.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        let config = c_string(r#"{"format": "json"}"#)?;
        let (mut len, mut data) = (0, ptr::null());
        // SAFETY: the handle is live, `config` is NUL-terminated, and the outputs point to this
        // thread's buffer for the booster, copied immediately.
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
        usize::try_from(rounds).map_err(|_| Error::new(format!("negative round count {rounds}")))
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
            return Err(Error::new(format!("cannot keep {rounds} of {trained} rounds")));
        }
        let end = c_int::try_from(rounds).map_err(|_| Error::new("round count overflow"))?;
        let mut handle = ptr::null_mut();
        // SAFETY: the handle is live and `handle` is a valid out-pointer for the new booster.
        check(unsafe { ffi::XGBoosterSlice(self.handle, 0, end, 1, &mut handle) })?;
        Self::with_threads(handle, self.threads)
    }
}

/// The threads a batch of booster parameters sets: its last `nthread` (the one XGBoost keeps),
/// if any.
///
/// # Errors
///
/// Returns an error if an `nthread` is not a positive count XGBoost can take (it stores the count
/// as a C `int`), or `n_jobs` is set.
fn threads_set_by(params: &[(&str, &str)]) -> Result<Option<NonZeroUsize>> {
    let mut threads = None;
    for &(name, value) in params {
        match name {
            "nthread" => {
                let count = value
                    .parse::<NonZeroUsize>()
                    .ok()
                    .filter(|count| c_int::try_from(count.get()).is_ok())
                    .ok_or_else(|| {
                        Error::new(format!(
                            "nthread must be a positive count of threads, at most {}, not {value:?}",
                            c_int::MAX
                        ))
                    })?;
                threads = Some(count);
            }
            "n_jobs" => return Err(Error::new("set the threads with nthread, not n_jobs")),
            _ => {}
        }
    }
    Ok(threads)
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

    /// The vendored C API header, where XGBoost declares the functions `mod ffi` binds.
    const C_API_HEADER: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/vendor/xgboost/include/xgboost/c_api.h");

    /// A C type from `c_api.h` (e.g. `char const **`, or `const char *` of an array parameter
    /// `name[]` with `array`), as `mod ffi` writes it in Rust.
    fn rust_type(c_type: &str, array: bool) -> String {
        let spaced = c_type.replace('*', " * ");
        let tokens: Vec<&str> = spaced.split_whitespace().collect();
        let stars = tokens.iter().position(|&t| t == "*").unwrap_or(tokens.len());
        let base = tokens[..stars].iter().find(|&&t| t != "const").copied().unwrap_or("?");
        let mut pointee_const = tokens[..stars].contains(&"const");
        let mut rust = match base {
            "char" => "c_char",
            "float" => "c_float",
            "int" => "c_int",
            "void" => "c_void",
            "bst_ulong" => "BstUlong",
            other => other,
        }
        .to_string();
        let mut rest = tokens[stars..].iter().peekable();
        let mut levels = Vec::new();
        while let Some(&token) = rest.next() {
            assert_eq!(token, "*", "{c_type}");
            levels.push(pointee_const);
            pointee_const = rest.next_if(|&&t| t == "const").is_some();
        }
        if array {
            levels.push(pointee_const);
        }
        for is_const in levels {
            rust = format!("{} {rust}", if is_const { "*const" } else { "*mut" });
        }
        rust
    }

    /// The parameter types and return type of `name` as `c_api.h` declares it, in Rust.
    fn declared_in_header(header: &str, name: &str) -> (Vec<String>, Option<String>) {
        let at = header
            .match_indices(&format!("{name}("))
            .map(|(at, _)| at)
            .find(|&at| {
                let line = header[..at].rsplit('\n').next().unwrap_or_default();
                line.starts_with("XGB_DLL") && (line.ends_with(' ') || line.ends_with('*'))
            })
            .unwrap_or_else(|| panic!("{name} is not declared in {C_API_HEADER}"));
        let line_start = header[..at].rfind('\n').map_or(0, |i| i + 1);
        let returns = header[line_start..at].trim_start_matches("XGB_DLL").trim();
        let close = at + header[at..].find(')').expect("a closing parenthesis");
        // Without comments.
        let params: String = header[at + name.len() + 1..close]
            .lines()
            .map(|line| line.split("//").next().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(" ");
        let params = params.trim();
        let params = if params.is_empty() || params == "void" {
            Vec::new()
        } else {
            params
                .split(',')
                .map(|param| {
                    let param = param.trim();
                    let array = param.ends_with("[]");
                    let param = param.trim_end_matches("[]");
                    // The parameter name is the last identifier.
                    let split = param
                        .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                        .map_or(0, |i| i + 1);
                    rust_type(&param[..split], array)
                })
                .collect()
        };
        let returns = (returns != "void").then(|| rust_type(returns, false));
        (params, returns)
    }

    /// Each function in `mod ffi`, with its parameter types and return type.
    fn bound_in_ffi() -> Vec<(String, Vec<String>, Option<String>)> {
        let source = include_str!("lib.rs");
        let module = source.find("\nmod ffi {").expect("mod ffi");
        let start =
            module + source[module..].find("unsafe extern \"C\" {").expect("its extern block");
        let end = start + source[start..].find("\n    }\n").expect("the end of the extern block");
        source[start..end]
            .split("pub fn ")
            .skip(1)
            .map(|declaration| {
                let declaration: String =
                    declaration.split_whitespace().collect::<Vec<_>>().join(" ");
                let open = declaration.find('(').expect("parameters");
                let close = declaration.rfind(')').expect("parameters");
                let name = declaration[..open].to_string();
                let params = declaration[open + 1..close]
                    .split(',')
                    .map(str::trim)
                    .filter(|param| !param.is_empty())
                    .map(|param| param.split_once(':').expect("name: type").1.trim().to_string())
                    .collect();
                let returns = declaration[close + 1..]
                    .trim()
                    .strip_prefix("->")
                    .map(|ty| ty.trim_end_matches(';').trim().to_string());
                (name, params, returns)
            })
            .collect()
    }

    #[test]
    fn ffi_declarations_match_the_c_api_header() {
        let header = std::fs::read_to_string(C_API_HEADER).unwrap();
        // The handle and length types are the header's.
        for typedef in [
            "typedef uint64_t bst_ulong;",
            "typedef void *DMatrixHandle;",
            "typedef void *BoosterHandle;",
        ] {
            assert!(header.contains(typedef), "{typedef}");
        }
        // ... and so are `mod ffi`'s aliases of them.
        let _: fn(ffi::BstUlong) -> u64 = |length| length;
        let _: fn(ffi::DMatrixHandle) -> *mut std::ffi::c_void = |handle| handle;
        let _: fn(ffi::BoosterHandle) -> *mut std::ffi::c_void = |handle| handle;
        let bound = bound_in_ffi();
        // Every declaration is parsed (a binding added to `mod ffi` is added here too).
        let mut names: Vec<&str> = bound.iter().map(|(name, ..)| name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "XGBGetLastError",
                "XGBoostVersion",
                "XGBoosterBoostedRounds",
                "XGBoosterCreate",
                "XGBoosterEvalOneIter",
                "XGBoosterFree",
                "XGBoosterGetNumFeature",
                "XGBoosterGetStrFeatureInfo",
                "XGBoosterLoadModelFromBuffer",
                "XGBoosterPredictFromDMatrix",
                "XGBoosterSaveJsonConfig",
                "XGBoosterSaveModelToBuffer",
                "XGBoosterSetParams",
                "XGBoosterSetStrFeatureInfo",
                "XGBoosterSlice",
                "XGBoosterUpdateOneIter",
                "XGDMatrixCreateFromMat_omp",
                "XGDMatrixFree",
                "XGDMatrixGetStrFeatureInfo",
                "XGDMatrixNumCol",
                "XGDMatrixNumRow",
                "XGDMatrixSetFloatInfo",
                "XGDMatrixSetStrFeatureInfo",
            ]
        );
        // Pointer mutability is not compared: `mod ffi` declares some out-pointers' targets
        // `*const` (read-only views of what XGBoost hands back) where C has them mutable, which
        // has the same ABI. The arity, the base types and the pointer depths must match.
        let erased = |types: &[String]| -> Vec<String> {
            types.iter().map(|t| t.replace("*const ", "*").replace("*mut ", "*")).collect()
        };
        for (name, params, returns) in bound {
            let (header_params, header_returns) = declared_in_header(&header, &name);
            assert_eq!(erased(&params), erased(&header_params), "{name}");
            assert_eq!(
                erased(returns.as_slice()),
                erased(header_returns.as_slice()),
                "{name} returns"
            );
        }
    }

    #[rstest]
    #[case("int", false, "c_int")]
    #[case("const char *", false, "*const c_char")]
    #[case("char const **", false, "*mut *const c_char")]
    #[case("const char ***", false, "*mut *mut *const c_char")]
    #[case("const char *", true, "*mut *const c_char")]
    #[case("const DMatrixHandle", true, "*const DMatrixHandle")]
    #[case("bst_ulong const **", false, "*mut *const BstUlong")]
    #[case("char * const *", false, "*const *mut c_char")]
    fn maps_c_types_to_rust(#[case] c_type: &str, #[case] array: bool, #[case] expected: &str) {
        assert_eq!(rust_type(c_type, array), expected);
    }

    /// Runs the check for `runtime` with two threads and the given variables set.
    fn check_with(runtime: OpenMpRuntime, set: &[(&str, &str)]) -> std::result::Result<(), String> {
        let var = |name: &str| {
            set.iter().find(|(set_name, _)| *set_name == name).map(|(_, value)| value.to_string())
        };
        runtime.check_environment(TWO, var).map_err(|problem| problem.to_string())
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
        let error = check_openmp_environment(TWO, var).unwrap_err();
        assert_eq!(
            error.openmp_problem(),
            Some(&OpenMpProblem::ThreadLimit { limit: 1, threads: TWO })
        );
        assert_eq!(
            error.openmp_problem().unwrap().explanation(),
            "OMP_THREAD_LIMIT=1 is below the 2 threads requested, so XGBoost would run with 1 \
             thread"
        );
        assert_eq!(
            error.to_string(),
            "XGBoost: OMP_THREAD_LIMIT=1 is below the 2 threads requested, so XGBoost would run \
             with 1 thread, and a model trained so is not the one 2 threads would train; request \
             1 thread or fewer, or raise or unset OMP_THREAD_LIMIT"
        );
        assert_eq!(Error::new("other").openmp_problem(), None);
    }

    #[rstest]
    #[case::dynamic(
        OpenMpProblem::Dynamic { value: "true".to_string() },
        "OMP_DYNAMIC=true lets OpenMP give XGBoost fewer threads than it asks for, which XGBoost's \
         training does not handle (it can lose rows or crash); unset OMP_DYNAMIC or set it to false"
    )]
    #[case::hidden(
        OpenMpProblem::HiddenThreadLimit { variable: "KMP_ALL_THREADS", limit: 1, threads: TWO },
        "KMP_ALL_THREADS=1 is below the 2 threads requested, so OpenMP would give XGBoost fewer \
         threads than it asks for, which XGBoost's training does not handle (it can lose rows or \
         crash); request 1 thread or fewer, or raise or unset KMP_ALL_THREADS"
    )]
    #[case::thread_limit(
        OpenMpProblem::ThreadLimit { limit: 1, threads: TWO },
        "OMP_THREAD_LIMIT=1 is below the 2 threads requested, so XGBoost would run with 1 thread, \
         and a model trained so is not the one 2 threads would train; request 1 thread or fewer, \
         or raise or unset OMP_THREAD_LIMIT"
    )]
    fn openmp_problems_explain_what_to_do(#[case] problem: OpenMpProblem, #[case] expected: &str) {
        assert_eq!(problem.to_string(), expected);
        assert!(expected.starts_with(&problem.explanation()), "{expected}");
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

    /// A noisy three-class training matrix of `rows` rows of 6 features: a linear congruential
    /// sequence, labelled by a weighted sum of three features plus noise, so that XGBoost's
    /// histogram sums are inexact and, at 200,000 rows, depend on the thread count.
    fn noisy_matrix(rows: usize) -> DMatrix {
        let cols = 6;
        let mut state: u64 = 12_345;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as f32 / (1u64 << 31) as f32
        };
        let mut features = Vec::with_capacity(rows * cols);
        let mut labels = Vec::with_capacity(rows);
        for _ in 0..rows {
            let row: Vec<f32> = (0..cols).map(|_| next()).collect();
            let score = row[0] * 3.0 + row[1] * 2.0 - row[2] + next() * 1.5;
            labels.push(if score < 1.5 {
                0.0
            } else if score < 3.0 {
                1.0
            } else {
                2.0
            });
            features.extend(row);
        }
        let mut matrix = DMatrix::from_dense(&features, rows, cols, f32::NAN).unwrap();
        matrix.set_labels(&labels).unwrap();
        matrix
    }

    /// A three-class lossguide model of `train` with `threads` threads, building the classes'
    /// trees one after another or at the same time, as XGBoost JSON, and the booster's config.
    fn model_json(
        train: &DMatrix,
        threads: usize,
        parallel_target_trees: bool,
    ) -> (Vec<u8>, serde_json::Value) {
        let mut booster = Booster::new(&[train], NonZeroUsize::new(threads).unwrap()).unwrap();
        booster
            .set_params(&[
                ("objective", "multi:softprob"),
                ("num_class", "3"),
                ("tree_method", "hist"),
                ("grow_policy", "lossguide"),
                ("max_leaves", "64"),
                ("eta", "0.5"),
                ("seed", "1"),
                ("parallel_target_trees", if parallel_target_trees { "true" } else { "false" }),
            ])
            .unwrap();
        for iteration in 0..60 {
            booster.update(iteration, train).unwrap();
        }
        (booster.to_json().unwrap(), booster.config().unwrap())
    }

    /// Patch 0003: building the 3 classes' trees at the same time with 9 threads builds each with
    /// 3, so the model is the one 3 threads build in turn, not the one 9 threads do (whose
    /// histogram sums differ), which shows the trees were built at once.
    ///
    /// XGBoost uses no more threads than the machine has, so with fewer than 9 CPUs the thread
    /// counts cannot differ and only the parameter is checked.
    #[test]
    fn parallel_target_trees_builds_each_class_tree_with_its_share_of_the_threads() {
        let cpus = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let rows = if cpus >= 9 { 200_000 } else { 3_000 };
        let train = noisy_matrix(rows);
        let (at_once, at_once_config) = model_json(&train, 9, true);
        let (in_turn_3, in_turn_config) = model_json(&train, 3, false);
        // The parameter is XGBoost's (patched in), not an unknown one it ignores.
        let parameter = |config: &serde_json::Value| {
            config["learner"]["gradient_booster"]["gbtree_train_param"]["parallel_target_trees"]
                .clone()
        };
        assert_eq!(parameter(&at_once_config), "1", "{at_once_config}");
        assert_eq!(parameter(&in_turn_config), "0", "{in_turn_config}");
        if cpus < 9 {
            eprintln!("only {cpus} CPUs: not checking which threads built the trees");
            return;
        }
        let (in_turn_9, _) = model_json(&train, 9, false);
        assert_ne!(in_turn_9, in_turn_3, "the data must make the model depend on the threads");
        assert_eq!(at_once, in_turn_3);
    }

    #[test]
    fn finds_a_hidden_thread_limit_of_the_linked_runtime() {
        let unset = |_: &str| None;
        assert_eq!(hidden_thread_limit(unset), None);
        for name in ["KMP_DEVICE_THREAD_LIMIT", "KMP_ALL_THREADS"] {
            let set = |variable: &str| (variable == name).then(|| "64".to_string());
            let expected = cfg!(target_os = "macos").then_some(name);
            assert_eq!(hidden_thread_limit(set), expected);
        }
        // OMP_THREAD_LIMIT is XGBoost's own, which it reads.
        assert_eq!(
            hidden_thread_limit(|v: &str| (v == "OMP_THREAD_LIMIT").then(String::new)),
            None
        );
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
        let cores = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let busy = cpu_usage::busy_cores(cores).expect("the CPU time counters are readable");
        assert!(busy.is_finite() && busy >= 0.0, "{busy}");
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn reads_the_process_cpu_time() {
        use std::time::{Duration, Instant};
        let before = process_cpu_time().expect("the process CPU time is readable");
        let started = Instant::now();
        // Real work on this thread for about 50 ms, which the CPU time must count.
        let mut sum = 0u64;
        while started.elapsed() < Duration::from_millis(50) {
            for i in 0..10_000u64 {
                sum = std::hint::black_box(sum.wrapping_add(i));
            }
        }
        std::hint::black_box(sum);
        let after = process_cpu_time().expect("the process CPU time is readable");
        assert!(after > before, "{after:?} <= {before:?}");
    }

    /// A `/proc/stat` of four CPUs, each with `busy` busy and 100 total ticks.
    fn proc_stat(busy: [u64; 4]) -> String {
        let line = |name: &str, busy: u64, total: u64| {
            format!("{name} {busy} 0 0 {} 0 0 0 0 0 0\n", total - busy)
        };
        let mut stat = line("cpu ", busy.iter().sum(), 400);
        for (cpu, busy) in busy.iter().enumerate() {
            stat.push_str(&line(&format!("cpu{cpu}"), *busy, 100));
        }
        stat.push_str("intr 0\nctxt 0\n");
        stat
    }

    #[test]
    fn parses_cpu_lists() {
        assert_eq!(cpu_usage::parse_cpu_list("0-3,8,10-11"), Some(vec![0, 1, 2, 3, 8, 10, 11]));
        assert_eq!(cpu_usage::parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(cpu_usage::parse_cpu_list("3-1"), None);
        assert_eq!(cpu_usage::parse_cpu_list("a"), None);
    }

    #[test]
    fn measures_the_allowed_cpus() {
        use cpu_usage::{Ticks, stat_ticks};
        let stat = proc_stat([100, 100, 0, 50]);
        let all = Ticks { busy: 250, total: 400, cpu_ids: vec![0, 1, 2, 3], host_cpus: 4 };
        assert_eq!(stat_ticks(&stat, None), Some(all.clone()));
        let two = Ticks { busy: 50, total: 200, cpu_ids: vec![2, 3], host_cpus: 4 };
        assert_eq!(stat_ticks(&stat, Some(&[2, 3])), Some(two.clone()));
        // CPUs the file does not list are ignored; with none listed, or no affinity, the whole
        // machine is measured.
        assert_eq!(stat_ticks(&stat, Some(&[2, 3, 9])), Some(two));
        assert_eq!(stat_ticks(&stat, Some(&[9])), Some(all.clone()));
        assert_eq!(stat_ticks(&stat, Some(&[])), Some(all));
    }

    #[rstest]
    // Other work on half of 96 CPUs: half of an 8-core quota's share.
    #[case::quota(96, 96, 8, 0.5, None, 4.0)]
    // A cpuset of 8 CPUs with 2 of them busy.
    #[case::cpuset(8, 96, 8, 0.25, None, 2.0)]
    // Every allowed CPU busy: the load average, scaled from the machine's 96 CPUs to 8 cores.
    #[case::load_average(8, 96, 8, 1.0, Some(192.0), 16.0)]
    #[case::load_below_the_counters(8, 96, 8, 1.0, Some(24.0), 8.0)]
    #[case::load_unreadable(8, 96, 8, 1.0, Some(f64::NAN), 8.0)]
    // Half of one allowed CPU busy is not every CPU busy: the load average is not used.
    #[case::half_of_one_cpu(1, 8, 1, 0.5, Some(24.0), 0.5)]
    fn scales_the_other_work_to_the_allowed_cores(
        #[case] cpus: usize,
        #[case] host_cpus: usize,
        #[case] cores: usize,
        #[case] busy_fraction: f64,
        #[case] load: Option<f64>,
        #[case] expected: f64,
    ) {
        use cpu_usage::{Ticks, busy_between};
        let before = Ticks { busy: 0, total: 0, cpu_ids: (0..cpus).collect(), host_cpus };
        let total = 100 * cpus as u64;
        let busy = (busy_fraction * total as f64) as u64;
        let after = Ticks { busy, total, ..before.clone() };
        let got = busy_between(before, after, cores, load).unwrap();
        assert!((got - expected).abs() < 1e-9, "{got} != {expected}");
    }

    /// One thread on one allowed CPU of eight, half busy, under a load average of 24: the
    /// counters, not the stale machine-wide load, choose the policy.
    #[test]
    fn a_half_busy_single_cpu_spins_briefly() {
        use cpu_usage::{Ticks, busy_between};
        let before = Ticks { busy: 0, total: 0, cpu_ids: vec![0], host_cpus: 8 };
        let after = Ticks { busy: 50, total: 100, ..before.clone() };
        let busy = busy_between(before, after, 1, Some(24.0));
        assert_eq!(OpenMpWait::for_load(1, 1, busy), OpenMpWait::SpinBriefly);
    }

    #[test]
    fn the_other_work_is_unknown_without_two_comparable_readings() {
        use cpu_usage::{Ticks, busy_between};
        let before = Ticks { busy: 10, total: 100, cpu_ids: (0..8).collect(), host_cpus: 8 };
        // The CPUs measured changed between the readings.
        let after = Ticks { busy: 20, total: 200, cpu_ids: (0..4).collect(), ..before.clone() };
        assert_eq!(busy_between(before.clone(), after, 8, None), None);
        // No time passed, or the counters went backwards.
        assert_eq!(busy_between(before.clone(), before.clone(), 8, None), None);
        let backwards = Ticks { total: 50, ..before.clone() };
        assert_eq!(busy_between(before, backwards, 8, None), None);
    }

    /// The affinity moved from CPU 0 to CPU 1 between the readings: the counters of two CPUs
    /// are not comparable, though their number is the same.
    #[test]
    fn the_other_work_is_unknown_when_the_cpus_measured_change() {
        use cpu_usage::{Ticks, busy_between};
        let before = Ticks { busy: 100, total: 1000, cpu_ids: vec![0], host_cpus: 2 };
        let after = Ticks { busy: 900, total: 1025, cpu_ids: vec![1], host_cpus: 2 };
        assert_eq!(busy_between(before.clone(), after.clone(), 1, Some(6.0)), None);
        // On the same CPU, the same counters show it busy.
        let same = Ticks { cpu_ids: vec![0], ..after };
        assert!(busy_between(before, same, 1, Some(6.0)).is_some());
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
    fn outputs_per_row_are_the_class_or_target_count() {
        let (booster, _) = trained_booster(1);
        assert_eq!(booster.outputs_per_row().unwrap(), 3);
        let (features, labels) = three_class_data(30);
        let labels: Vec<f32> = labels.iter().map(|&class| f32::from(class > 0.0)).collect();
        let mut train = DMatrix::from_dense(&features, 30, 2, f32::NAN).unwrap();
        train.set_labels(&labels).unwrap();
        let mut booster = Booster::new(&[&train], TWO).unwrap();
        booster.set_param("objective", "binary:logistic").unwrap();
        booster.update(0, &train).unwrap();
        assert_eq!(booster.outputs_per_row().unwrap(), 1);
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

    /// A shared predictor predicts from many threads at once on one booster, each prediction
    /// that of the booster alone.
    #[test]
    fn shared_predictor_predicts_from_many_threads_at_once() {
        let (mut booster, _) = trained_booster(8);
        booster.set_threads(NonZeroUsize::MIN).unwrap();
        let (features, _) = three_class_data(500);
        let matrix = |values: &[f32]| {
            let mut matrix = DMatrix::from_dense(values, 500, 2, f32::NAN).unwrap();
            matrix.set_feature_names(&["signal", "noise"]).unwrap();
            matrix
        };
        let cases: Vec<_> = (1..=8)
            .map(|run_length| {
                let mut values = features.clone();
                for (row, values) in values.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    values[0] = ((row / run_length) % 3) as f32 * 10.0;
                }
                let expected = booster.predict(&matrix(&values)).unwrap();
                (values, expected)
            })
            .collect();
        for (i, (_, expected)) in cases.iter().enumerate() {
            for (j, (_, other)) in cases[..i].iter().enumerate() {
                assert_ne!(expected.values, other.values, "workers {i} and {j}");
            }
        }
        let shared = booster.shared_predictor().unwrap().unwrap();
        std::thread::scope(|scope| {
            for (values, expected) in &cases {
                let (shared, matrix) = (&shared, &matrix);
                scope.spawn(move || {
                    for _ in 0..25 {
                        assert_eq!(&shared.predict(&matrix(values)).unwrap(), expected);
                    }
                });
            }
        });
        // A matrix of other features fails on any thread.
        let mut wrong = DMatrix::from_dense(&features, 250, 4, f32::NAN).unwrap();
        wrong.set_feature_names(&["a", "b", "c", "d"]).unwrap();
        let message = shared.predict(&wrong).unwrap_err().to_string();
        assert!(message.contains("2 features but the matrix has 4 columns"), "{message}");
    }

    /// XGBoost's error messages are per thread, as `SharedPredictor` relies on: errors raised by
    /// XGBoost on several threads at once, on one booster, while others predict with it, each
    /// come back to the thread that raised it.
    #[test]
    fn xgboost_errors_are_per_thread() {
        let (mut booster, _) = trained_booster(8);
        booster.set_threads(NonZeroUsize::MIN).unwrap();
        let (features, _) = three_class_data(100);
        let matrix = || {
            let mut matrix = DMatrix::from_dense(&features, 100, 2, f32::NAN).unwrap();
            matrix.set_feature_names(&["signal", "noise"]).unwrap();
            matrix
        };
        let expected = booster.predict(&matrix()).unwrap();
        let shared = booster.shared_predictor().unwrap().unwrap();
        // XGBoost fails on an unknown feature-info field, naming it.
        fn fail(shared: &SharedPredictor<'_>, field: &str) -> String {
            let field = c_string(field).unwrap();
            let (mut len, mut names) = (0, ptr::null());
            // SAFETY: the handle is live (borrowed by `shared`), `field` is NUL-terminated, and
            // the out-pointers are valid; XGBoost writes nothing to them when it fails.
            let code = unsafe {
                ffi::XGBoosterGetStrFeatureInfo(
                    shared.0.handle,
                    field.as_ptr(),
                    &mut len,
                    &mut names,
                )
            };
            check(code).unwrap_err().to_string()
        }
        std::thread::scope(|scope| {
            for thread in 0..8 {
                let (shared, matrix, expected) = (&shared, &matrix, &expected);
                scope.spawn(move || {
                    for round in 0..200 {
                        if thread % 2 == 0 {
                            let field = format!("field-of-thread-{thread}-round-{round}");
                            let message = fail(shared, &field);
                            assert!(message.ends_with(&field), "{field}: {message}");
                        } else {
                            assert_eq!(&shared.predict(&matrix()).unwrap(), expected);
                        }
                    }
                });
            }
        });
    }

    /// Only a tree booster is shared to predict from several threads.
    #[test]
    fn only_a_tree_booster_is_shared() {
        let train = train_matrix(60);
        let mut booster = Booster::new(&[&train], TWO).unwrap();
        booster
            .set_params(&[
                ("booster", "gblinear"),
                ("objective", "multi:softprob"),
                ("num_class", "3"),
            ])
            .unwrap();
        booster.update(0, &train).unwrap();
        assert!(booster.shared_predictor().unwrap().is_none());
        let (tree, _) = trained_booster(5);
        assert!(tree.shared_predictor().unwrap().is_some());
    }

    /// Setting a booster's threads sets XGBoost's `nthread`, and changes no prediction.
    #[test]
    fn set_threads_changes_the_threads_not_the_predictions() {
        let (mut booster, train) = trained_booster(5);
        let before = booster.predict(&train).unwrap();
        let nthread = |booster: &Booster| {
            booster.config().unwrap()["learner"]["generic_param"]["nthread"].clone()
        };
        let one = NonZeroUsize::new(1).unwrap();
        booster.set_threads(one).unwrap();
        assert_eq!(booster.threads(), one);
        assert_eq!(nthread(&booster), "1");
        assert_eq!(booster.predict(&train).unwrap(), before);
        let three = NonZeroUsize::new(3).unwrap();
        booster.set_threads(three).unwrap();
        assert_eq!(booster.threads(), three);
        assert_eq!(nthread(&booster), "3");
        assert_eq!(booster.predict(&train).unwrap(), before);
    }

    /// Setting `nthread` among other parameters changes the booster's threads as `set_threads`
    /// does, so [`Booster::threads`] (which a [`SharedPredictor`]'s callers check) is always the
    /// count XGBoost uses; a value that is not a positive count, or the `n_jobs` alias, is
    /// refused before XGBoost sees the batch.
    #[test]
    fn setting_nthread_changes_the_threads() {
        let (mut booster, _) = trained_booster(5);
        let nthread = |booster: &Booster| {
            booster.config().unwrap()["learner"]["generic_param"]["nthread"].clone()
        };
        booster.set_threads(NonZeroUsize::MIN).unwrap();
        booster.set_params(&[("nthread", "4")]).unwrap();
        assert_eq!(booster.threads().get(), 4);
        assert_eq!(nthread(&booster), "4");
        // The last of a batch is the one XGBoost keeps.
        booster.set_params(&[("nthread", "2"), ("eta", "0.2"), ("nthread", "3")]).unwrap();
        assert_eq!(booster.threads().get(), 3);
        assert_eq!(nthread(&booster), "3");
        // XGBoost takes 0 or less for every thread, which `threads` cannot record.
        for value in ["0", "-1", "all", "", "3000000000"] {
            let error = booster.set_params(&[("eta", "0.1"), ("nthread", value)]).unwrap_err();
            assert!(error.message().contains("nthread"), "{error}");
            assert_eq!(booster.threads().get(), 3, "{value}");
            assert_eq!(nthread(&booster), "3", "{value}");
        }
        // A count above what XGBoost stores (a C `int`) is refused before XGBoost sees it.
        let huge = NonZeroUsize::new(usize::try_from(c_int::MAX).unwrap() + 1).unwrap();
        assert!(booster.set_threads(huge).unwrap_err().message().contains("nthread"));
        assert_eq!(booster.threads().get(), 3);
        assert_eq!(nthread(&booster), "3");
        let error = booster.set_params(&[("n_jobs", "4")]).unwrap_err();
        assert!(error.message().contains("n_jobs"), "{error}");
        assert_eq!(booster.threads().get(), 3);
        assert_eq!(nthread(&booster), "3");
        // XGBoost may take up a count before it rejects the rest of the batch: the booster then
        // counts the larger of the two.
        let mut failed = Booster::new(&[], NonZeroUsize::MIN).unwrap();
        let multiclass_without_classes = [("nthread", "4"), ("objective", "multi:softprob")];
        assert!(failed.set_params(&multiclass_without_classes).is_err());
        assert_eq!(failed.threads().get(), 4);
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
