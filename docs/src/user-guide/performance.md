# Performance tuning

## Threads

Every command runs on the number of threads given by `-t`; without `-t`, on `OMP_NUM_THREADS` when
it is set (its first value, when it lists several), else on 4. An `OMP_NUM_THREADS`
that is not a positive number is ignored, with a warning.

- Give `-t` at most the cores that are free: every command, `score --alignments` included, runs
  within `-t` threads in all. `score --alignments` measures the reads while the models that do not
  need them score, the two sharing the `-t` threads: a thread free of one takes up work of the
  other, a stretch of the reads or a chunk of a model's calls at a time. With `-t 1`, or when
  every model needs the reads, it measures the reads first, then scores. How long measuring the
  reads took, and, when the two overlapped, how long scoring the models that do not need them
  took, is logged at info level. `score` writes and indexes its VCF within `-t` threads too:
  htslib's compression and decompression threads are counted in them (apart from an I/O thread
  per file, which mostly waits on the disk).
- At debug level (`RUST_LOG=winnowr=debug`), every command logs how long each of its phases
  took and the CPU time the process used meanwhile, e.g. `writing the VCF: writing the shards took
  1.08s (cpu 12.31s)`. The CPU time is the whole process's, of every thread: phases that run at
  the same time (measuring the reads and scoring the models that do not need them, in `score
  --alignments`) each count the other's, and OpenMP threads polling between parallel steps count
  as busy. So the CPU time over the time taken is how many cores a phase kept busy only for a
  phase that runs alone. Phases nest: a line whose phase ends in "in all" is logged after the
  phases it is made of, and its times include theirs.
- Measured on a 16-core machine (an AWS c7a.4xlarge), `score --alignments` on a 30x whole-genome
  CRAM with four models (two of them needing the reads) took 554 s with `-t 1`, 287 s with `-t 2`,
  149 s with `-t 4`, 83 s with `-t 8` and 51 s with `-t 16`. The total CPU time stays about the
  same at every `-t`; what keeps `-t 16` from being 16 times faster is about 13 s of setup,
  deciding and writing that runs on few threads.
- Training runs thousands of short parallel steps per tree, and loses far more from threads that
  wait for a core than it gains from extra threads. Past about 8-16 threads, more threads gain
  little.
- A trained model depends on the thread count, as XGBoost's histogram sums do: training again with
  the same tables, options and threads gives the same model. The thread count is recorded in the
  model.

## How idle threads wait

At the start of `train`, winnowr measures the other work on the CPUs it may use (the cores busy over
250 ms, or the load average when every core is busy; on Linux, within its CPU affinity and container
CPU limit) and sets how XGBoost's OpenMP threads wait between parallel steps. If the threads fit on
the free cores, they keep polling for the next step for a while (fastest); if the machine is busy,
they poll only briefly; with more than twice as much work as cores, they sleep at once. Polling
threads on a busy machine slowed training down 10-100 times. The choice is logged at debug level
(`RUST_LOG=winnowr=debug`) and does not change the model. Setting `OMP_WAIT_POLICY`, or
`KMP_BLOCKTIME` (macOS) or `GOMP_SPINCOUNT` (Linux), overrides it. On Linux, winnowr can set the
wait only when GCC's OpenMP runtime is linked statically (see `xgboost/build.rs`).

## OpenMP environment variables that `train` refuses

XGBoost's training assumes every parallel step gets the full team of threads that `-t` asks for.
`train` stops with an error, naming the variable to unset or change, when the environment would
give it fewer:

- `OMP_DYNAMIC` set to true (on macOS, also `1`, `yes`, `on`, `t`, `enabled` and the other
  spellings libomp reads as true), which lets the OpenMP runtime give a parallel step fewer
  threads,
- `OMP_THREAD_LIMIT` below the thread count, which caps XGBoost's threads, so the model would not
  be the one `-t` names,
- on macOS, `KMP_DEVICE_THREAD_LIMIT` (or its older name `KMP_ALL_THREADS`) below the thread
  count.

The other commands do not check these: prediction does not need the full team.

## Memory

Training is bound by memory bandwidth and latency rather than arithmetic, which is why it scales
poorly with threads. On Linux, winnowr's allocator (mimalloc, which also serves XGBoost's C++
allocations) asks for transparent huge pages (`madvise`) for its memory, so it gets them under the
default THP setting (`madvise`); setting THP to `always`
(`/sys/kernel/mm/transparent_hugepage/enabled`) gains nothing more. On macOS, only winnowr's own
(Rust) allocations use mimalloc.

## Feature-table reader validation

A controlled comparison read the same programmatically generated labelled Parquet table with
the reader before alignment provenance (`d162e784`, before #38) and the current hardened reader
(`f73d4c7`, including #38 and #47). This measures the cumulative reader changes, not #38 alone.
The table contains 4,835,999 calls, 58 columns and synthetic three-class labels; its Zstandard
file is 47,386,128 bytes. Depths, qualities and strand counts vary deterministically, while many
optional annotations are NaN. Its row count and schema resemble a whole-genome table, but its
synthetic distributions and compressibility do not represent all real samples.

Both builds used Rust 1.100.0's release profile, mimalloc and eight Rayon threads on an Apple
M2 Max (12 cores, 64 GiB RAM). The same portable test harness called
`FeatureTable::read_parquet_selected` in each build. Full reads decoded all columns;
projected reads retained eight columns and 3,868,799 SNP rows. Before timing, a stock
Parquet/Arrow decoder independent of winnowr's reader checked every
decoded column, float bit pattern (including missing values), schema and metadata between
builds for both operations. Post-read checks also verify exact projected columns and generated
call identities. Each timed process initializes its Rayon pool before starting the timer.

The warmed-initialization two-pair probe had about 12% full-read spread (range width over the
mean), so the comparison used seven further pairs, alternating build order with warm input
pages. Times below include decoding, filtering and concatenation, but exclude process startup,
pool initialization, generation, identity checks, output writes, comparison and the returned
table's final drop:

| operation | before: median (range), seconds | current: median (range), seconds |
|-----------|---------------------------------|----------------------------------|
| full table | 0.1932 (0.1905-0.1981) | 0.1857 (0.1804-0.1898) |
| SNP projection | 0.0637 (0.0608-0.0660) | 0.0649 (0.0619-0.0663) |

All reads succeeded and returned the expected row counts. Full reads were faster in this run;
the projected-read ranges overlap and do not resolve small changes. These short, synthetic
reader-only measurements do not establish a general speedup or predict end-to-end training or
evaluation time. The current libtest reader also executes
inactive race-test hooks, including one mutex acquisition per row group, which the historical
libtest and shipping binary lack. That overhead is unmeasured and further limits interpretation
of small differences.

To reproduce, build the library tests in release mode at each revision, copying the unchanged
`src/validation_tests.rs` into the historical tree and adding `#[cfg(test)] mod validation_tests;`
to its `src/lib.rs`. Pass the two test executables printed by Cargo to:

```console
ROWS=4835999 REPS=7 THREADS=8 bash scripts/benchmark-table-reads.sh \
  /path/to/before-libtest /path/to/current-libtest /scratch/new-table-comparison
```

The script generates data outside the repository, verifies decoded parity, then times reads.
Run it without competing CPU-heavy work, and choose repetitions from the observed spread.
