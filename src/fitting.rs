//! Fitting a model's XGBoost booster in training: its parameters, and early stopping on calls held
//! out in whole genomic blocks.

use std::num::NonZeroUsize;

use anyhow::{Context, Result, bail};
use winnowr_xgboost::{Booster, DMatrix};

use crate::model::{TrainingParams, named_matrix};

/// The fewest threads (requested) for which a fit asks XGBoost to build its classes' trees at the
/// same time (`parallel_target_trees`, patch 0003 in `xgboost/patches/`), each with a share of the
/// threads. With fewer, each class's tree using every thread is as fast; with more, a tree of small
/// data uses the threads poorly, and building the classes' trees at once is 25-40% faster per round
/// at 15 and 30 threads (measured on an M5 with real training tables).
///
/// Each class's tree is then the tree a fit with its share of the threads builds, so the model is
/// that of fewer threads: XGBoost's histogram sums depend on the thread count, and a model can
/// differ in its last bits from one built with the classes in turn. A given thread count still
/// always trains the same model.
const MIN_THREADS_FOR_PARALLEL_TARGET_TREES: usize = 9;

/// Whether a fit with `threads` threads asks XGBoost to build its classes' trees at the same time:
/// with at least [`MIN_THREADS_FOR_PARALLEL_TARGET_TREES`], unless the OpenMP runtime has a thread
/// limit XGBoost does not read (see [`winnowr_xgboost::hidden_thread_limit`]), which teams built at
/// once could exceed. XGBoost may still build them in turn (e.g. if its CPU quota leaves too few
/// threads), with a warning.
///
/// `var` looks up an environment variable (`std::env::var` outside tests).
fn builds_class_trees_at_once(threads: NonZeroUsize, var: impl Fn(&str) -> Option<String>) -> bool {
    threads.get() >= MIN_THREADS_FOR_PARALLEL_TARGET_TREES
        && winnowr_xgboost::hidden_thread_limit(var).is_none()
}

/// The XGBoost booster parameters for `num_class` classes, as `(name, value)` pairs, asking
/// XGBoost to build the classes' trees at the same time if `class_trees_at_once` (see
/// [`builds_class_trees_at_once`]).
#[must_use]
pub fn xgboost_params(
    params: &TrainingParams,
    num_class: usize,
    class_trees_at_once: bool,
) -> Vec<(&'static str, String)> {
    let parallel_target_trees = class_trees_at_once;
    vec![
        ("objective", "multi:softprob".to_string()),
        ("num_class", num_class.to_string()),
        ("eval_metric", "mlogloss".to_string()),
        ("tree_method", "hist".to_string()),
        ("grow_policy", "lossguide".to_string()),
        ("eta", params.learning_rate.to_string()),
        ("max_leaves", params.max_leaves.to_string()),
        ("max_depth", params.max_depth.to_string()),
        ("min_child_weight", params.min_child_weight.to_string()),
        ("lambda", params.reg_lambda.to_string()),
        ("seed", params.seed.to_string()),
        ("parallel_target_trees", parallel_target_trees.to_string()),
    ]
}

/// The validation mlogloss in an XGBoost evaluation line (e.g. `[3]\tvalidation-mlogloss:0.21`):
/// the value after `validation-mlogloss:` in the first tab-separated field that starts with it,
/// trimmed and parsed as an `f64`.
fn parse_validation_loss(line: &str) -> Result<f64> {
    line.split('\t')
        .find_map(|field| field.strip_prefix("validation-mlogloss:"))
        .and_then(|value| value.trim().parse().ok())
        .with_context(|| format!("no validation mlogloss in XGBoost's evaluation: {line}"))
}

/// SplitMix64's output function: a bijective mix of 64 bits.
fn mix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform draw in `[0, 1)` for a genomic block, determined by the seed, the contig name and
/// the block's index on the contig (the same on every platform and in every run).
fn block_draw(seed: u64, chrom: &str, block: i64) -> f64 {
    let mut hash = mix64(seed);
    for byte in chrom.bytes() {
        hash = mix64(hash ^ u64::from(byte));
    }
    hash = mix64(hash ^ chrom.len() as u64);
    // Two's complement keeps negative indices (never seen in practice) distinct.
    hash = mix64(hash ^ block as u64);
    (hash >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Which calls to hold out for early stopping: whole genomic blocks.
///
/// Each contig is cut into blocks of `validation_block_size` bp, and a block is held out when a
/// hash of the seed, the contig and the block index is below `validation_fraction`. Every call in
/// a held-out block is held out, from every sample, so about that fraction of the calls is held
/// out. Training calls pool several samples, which share most sites (often the same variant with
/// near-identical features) and whose neighbourhood features span kilobases; holding out whole
/// blocks keeps those copies, and most neighbours, out of training, so the validation loss is not
/// optimistic and early stopping does not stop late. Larger blocks also keep out the neighbours at
/// block edges, but hold out fewer, larger regions, so the validation loss varies more with the
/// seed; the default 10 kb blocks stop at about the round that minimises the loss of a held-out
/// sample. The blocks depend only on the seed, not on the calls, so every cross-validation fold
/// and the final fit hold out the same regions.
///
/// # Arguments
///
/// * `chroms` - the contig of each call.
/// * `positions` - the position of each call.
/// * `params` - the training parameters (the validation fraction, block size and seed).
///
/// # Errors
///
/// Returns an error if `chroms` and `positions` differ in length, or the block size is zero.
pub fn validation_mask<S: AsRef<str>>(
    chroms: &[S],
    positions: &[i64],
    params: &TrainingParams,
) -> Result<Vec<bool>> {
    if chroms.len() != positions.len() {
        bail!("{} contigs do not match {} positions", chroms.len(), positions.len());
    }
    if params.validation_block_size == 0 {
        bail!("the validation block size must be positive");
    }
    let block_size = i64::try_from(params.validation_block_size).unwrap_or(i64::MAX);
    if params.validation_fraction <= 0.0 {
        return Ok(vec![false; positions.len()]);
    }
    // Calls come sorted by position, so consecutive calls usually share a block and its draw.
    let mut last: Option<(&str, i64, bool)> = None;
    let mask = chroms
        .iter()
        .zip(positions)
        .map(|(chrom, &pos)| {
            let chrom = chrom.as_ref();
            let block = pos.div_euclid(block_size);
            match last {
                Some((c, b, held_out)) if c == chrom && b == block => held_out,
                _ => {
                    let held_out =
                        block_draw(params.seed, chrom, block) < params.validation_fraction;
                    last = Some((chrom, block, held_out));
                    held_out
                }
            }
        })
        .collect();
    Ok(mask)
}

/// Fits a multi-class booster, early-stopping on the held-out calls.
///
/// A round is the best when its validation log-loss is strictly lower than every earlier
/// round's; training stops after `early_stopping_rounds` rounds without a new best and the
/// booster is truncated to the best round, as by XGBoost's Python `train`. With no held-out
/// calls every round is kept.
///
/// # Arguments
///
/// * `values` - the calls' features, row-major (see [`to_matrix`](crate::model::to_matrix)).
/// * `labels` - the class of each call.
/// * `is_validation` - whether each call is held out for early stopping rather than trained on
///   (see [`validation_mask`]).
/// * `features` - the feature names, in model order.
/// * `params` - the training parameters.
/// * `num_class` - the number of classes (see [`num_classes`](crate::model::num_classes)).
/// * `threads` - the XGBoost threads.
///
/// # Errors
///
/// Returns an error if the shapes disagree, every call is held out, or XGBoost fails.
pub fn fit_booster<S: AsRef<str>>(
    values: &[f32],
    labels: &[usize],
    is_validation: &[bool],
    features: &[S],
    params: &TrainingParams,
    num_class: usize,
    threads: NonZeroUsize,
) -> Result<Booster> {
    let n_features = features.len();
    if values.len() != labels.len() * n_features {
        bail!(
            "{} feature values do not fill {} calls of {n_features} features",
            values.len(),
            labels.len()
        );
    }
    if is_validation.len() != labels.len() {
        bail!("{} validation flags do not match {} calls", is_validation.len(), labels.len());
    }
    if let Some(label) = labels.iter().find(|&&label| label >= num_class) {
        bail!("label {label} is not one of the {num_class} classes");
    }
    let setup_start = std::time::Instant::now();
    if is_validation.iter().all(|&v| v) {
        bail!("no training calls left after holding out all {} for early stopping", labels.len());
    }
    let held_out = is_validation.iter().filter(|&&v| v).count();
    log::info!(
        "Holding out {held_out} of {} calls ({:.1}%) for early stopping",
        labels.len(),
        100.0 * held_out as f64 / labels.len() as f64
    );
    if held_out == 0 && params.validation_fraction > 0.0 {
        log::warn!(
            "no {} bp block of the {} calls was held out for early stopping (too few blocks for \
             a validation fraction of {}), so all {} rounds are kept",
            params.validation_block_size,
            labels.len(),
            params.validation_fraction,
            params.max_rounds
        );
    }
    let subset = |validation: bool| -> Result<DMatrix> {
        let rows: Vec<usize> =
            (0..labels.len()).filter(|&i| is_validation[i] == validation).collect();
        let mut selected = Vec::with_capacity(rows.len() * n_features);
        for &row in &rows {
            selected.extend_from_slice(&values[row * n_features..(row + 1) * n_features]);
        }
        let mut matrix = named_matrix(&selected, rows.len(), features, threads)?;
        let row_labels: Vec<f32> = rows.iter().map(|&row| labels[row] as f32).collect();
        matrix.set_labels(&row_labels)?;
        Ok(matrix)
    };
    let train = if held_out > 0 {
        subset(false)?
    } else {
        // Nothing is held out: every call trains, without a copy.
        let mut matrix = named_matrix(values, labels.len(), features, threads)?;
        let all_labels: Vec<f32> = labels.iter().map(|&label| label as f32).collect();
        matrix.set_labels(&all_labels)?;
        matrix
    };
    let validation = if held_out > 0 { Some(subset(true)?) } else { None };
    let cache: Vec<&DMatrix> = std::iter::once(&train).chain(validation.as_ref()).collect();
    let mut booster = Booster::new(&cache, threads)?;
    // XGBoost configures the booster with each batch of parameters, and the objective needs
    // `num_class`, so they are set together.
    let class_trees_at_once = builds_class_trees_at_once(threads, |name| std::env::var(name).ok());
    let xgb_params = xgboost_params(params, num_class, class_trees_at_once);
    let pairs: Vec<(&str, &str)> =
        xgb_params.iter().map(|(name, value)| (*name, value.as_str())).collect();
    booster.set_params(&pairs)?;
    let setup = setup_start.elapsed();
    let (mut updating, mut evaluating) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    let mut rounds_run = 0usize;
    let mut best: Option<(usize, f64)> = None;
    let mut since_best = 0;
    for round in 0..params.max_rounds as usize {
        let iteration = i32::try_from(round)?;
        let update_start = std::time::Instant::now();
        booster.update(iteration, &train)?;
        updating += update_start.elapsed();
        rounds_run += 1;
        let Some(validation) = &validation else { continue };
        let evaluate_start = std::time::Instant::now();
        let evaluation = booster.evaluate(iteration, &[(validation, "validation")])?;
        evaluating += evaluate_start.elapsed();
        let loss = parse_validation_loss(&evaluation)?;
        if improves(best, loss) {
            best = Some((round, loss));
            since_best = 0;
        } else {
            since_best += 1;
        }
        if since_best >= params.early_stopping_rounds as usize {
            break;
        }
    }
    let truncate_start = std::time::Instant::now();
    if let Some((round, _)) = best {
        booster = booster.truncated(round + 1)?;
    }
    log::debug!(
        "fit timing ({threads} threads, class trees requested {}): setup {setup:.2?}, {rounds_run} rounds \
         updating {updating:.2?} ({:.2?}/round), evaluating {evaluating:.2?}, truncating {:.2?}",
        if class_trees_at_once { "at once" } else { "in turn" },
        updating / u32::try_from(rounds_run.max(1))?,
        truncate_start.elapsed()
    );
    log::info!("Fitted {} rounds on {} calls", booster.boosted_rounds()?, labels.len() - held_out);
    Ok(booster)
}

/// Whether a round's validation loss improves on the best so far: it is strictly lower, and a NaN
/// loss never improves, not even in the first round (which would then be kept as the best).
fn improves(best: Option<(usize, f64)>, loss: f64) -> bool {
    !loss.is_nan() && best.is_none_or(|(_, best_loss)| loss < best_loss)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::model::predict_probabilities;

    const THREADS: NonZeroUsize = NonZeroUsize::new(1).expect("1 is non-zero");

    #[rstest]
    #[case::one_thread(1, None, false)]
    #[case::below(MIN_THREADS_FOR_PARALLEL_TARGET_TREES - 1, None, false)]
    #[case::at(MIN_THREADS_FOR_PARALLEL_TARGET_TREES, None, true)]
    #[case::above(64, None, true)]
    #[case::hidden_limit(64, Some("KMP_DEVICE_THREAD_LIMIT"), !cfg!(target_os = "macos"))]
    #[case::old_hidden_limit(64, Some("KMP_ALL_THREADS"), !cfg!(target_os = "macos"))]
    fn builds_the_class_trees_at_once_only_with_enough_threads_and_no_hidden_limit(
        #[case] threads: usize,
        #[case] limit: Option<&str>,
        #[case] expected: bool,
    ) {
        let threads = NonZeroUsize::new(threads).expect("the cases are non-zero");
        let var = |name: &str| (Some(name) == limit).then(|| "128".to_string());
        assert_eq!(builds_class_trees_at_once(threads, var), expected);
    }

    #[rstest]
    #[case::at_once(true, "true")]
    #[case::in_turn(false, "false")]
    fn asks_xgboost_for_the_class_trees_at_once(#[case] at_once: bool, #[case] expected: &str) {
        let params = xgboost_params(&TrainingParams::default(), 3, at_once);
        let value = params.iter().find(|(name, _)| *name == "parallel_target_trees");
        assert_eq!(value.map(|(_, value)| value.as_str()), Some(expected));
    }

    #[rstest]
    #[case::first(None, 0.5, true)]
    #[case::lower(Some((0, 0.5)), 0.4, true)]
    #[case::equal(Some((0, 0.5)), 0.5, false)]
    #[case::higher(Some((0, 0.5)), 0.6, false)]
    #[case::nan_first(None, f64::NAN, false)]
    #[case::nan_later(Some((0, 0.5)), f64::NAN, false)]
    #[case::after_nan_rounds(None, 0.7, true)]
    fn validation_losses_improve_only_when_lower_and_not_nan(
        #[case] best: Option<(usize, f64)>,
        #[case] loss: f64,
        #[case] expected: bool,
    ) {
        assert_eq!(improves(best, loss), expected);
    }

    /// Deterministic uniform draws in `[0, 1)` for test data (SplitMix64).
    struct TestRandom(u64);

    impl TestRandom {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(1);
            (mix64(self.0) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
        }
    }

    /// Noisy three-class calls: the class is the tercile of `a` (`b` is noise), except that 30%
    /// of labels are random.
    fn noisy_calls(rows: usize) -> (Vec<f32>, Vec<usize>) {
        let mut rng = TestRandom(7);
        let mut values = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..rows {
            let (a, b) = (rng.next_f64(), rng.next_f64());
            values.extend([a as f32, b as f32]);
            let label = if rng.next_f64() < 0.3 { rng.next_f64() * 3.0 } else { a * 3.0 };
            labels.push(label as usize);
        }
        (values, labels)
    }

    /// Loci of `per_contig` calls 1 kb apart on each of `contigs`.
    fn loci(contigs: &[&str], per_contig: i64) -> (Vec<String>, Vec<i64>) {
        let chroms = contigs
            .iter()
            .flat_map(|c| std::iter::repeat_n(c.to_string(), per_contig as usize))
            .collect();
        let positions =
            contigs.iter().flat_map(|_| (0..per_contig).map(|i| 1 + i * 1000)).collect();
        (chroms, positions)
    }

    #[test]
    fn holds_out_whole_blocks_in_every_sample() {
        let params = TrainingParams { validation_block_size: 100_000, ..TrainingParams::default() };
        let (chroms, positions) = loci(&["chr1", "chr2"], 4000);
        let mask = validation_mask(&chroms, &positions, &params).unwrap();
        // Every call of a block gets its block's decision, whatever its position in the block.
        for (i, &held_out) in mask.iter().enumerate() {
            let block_start = i - (positions[i] as usize - 1) % 100_000 / 1000;
            assert_eq!(held_out, mask[block_start], "call {i}");
        }
        // Another sample's calls at the same and nearby sites, in another order, get the same.
        let shuffled: Vec<usize> = (0..positions.len()).rev().step_by(3).collect();
        let other_chroms: Vec<&str> = shuffled.iter().map(|&i| chroms[i].as_str()).collect();
        let other_positions: Vec<i64> = shuffled.iter().map(|&i| positions[i] + 7).collect();
        let other = validation_mask(&other_chroms, &other_positions, &params).unwrap();
        for (j, &i) in shuffled.iter().enumerate() {
            assert_eq!(other[j], mask[i], "call {i}");
        }
        // The contigs are split independently.
        assert_ne!(mask[..4000], mask[4000..]);
    }

    #[test]
    fn holds_out_about_the_validation_fraction() {
        let (chroms, positions) = loci(&["chr1", "chr2", "chr3", "chrX"], 4000);
        for fraction in [0.1, 0.5] {
            let params = TrainingParams {
                validation_fraction: fraction,
                validation_block_size: 1000,
                ..TrainingParams::default()
            };
            let mask = validation_mask(&chroms, &positions, &params).unwrap();
            let held_out = mask.iter().filter(|&&v| v).count() as f64 / mask.len() as f64;
            assert!((held_out - fraction).abs() < 0.02, "{held_out} for {fraction}");
        }
        let none = TrainingParams { validation_fraction: 0.0, ..TrainingParams::default() };
        assert!(validation_mask(&chroms, &positions, &none).unwrap().iter().all(|&v| !v));
    }

    #[test]
    fn the_seed_chooses_the_blocks() {
        let (chroms, positions) = loci(&["chr1"], 4000);
        let params = TrainingParams { validation_block_size: 10_000, ..TrainingParams::default() };
        let mask = |seed| {
            validation_mask(&chroms, &positions, &TrainingParams { seed, ..params }).unwrap()
        };
        assert_eq!(mask(1), mask(1));
        assert_ne!(mask(1), mask(2));
        // The split is part of how a model was trained, so it must not change between releases.
        let seed_one = mask(1);
        let held_out: Vec<usize> = (0..400).filter(|&b| seed_one[b * 10]).collect();
        assert_eq!(held_out, EXPECTED_HELD_OUT_BLOCKS);
    }

    /// The 10 kb blocks of chr1 that seed 1 holds out with the default fraction, among the first
    /// 400.
    const EXPECTED_HELD_OUT_BLOCKS: [usize; 35] = [
        13, 28, 46, 48, 52, 66, 71, 72, 76, 77, 104, 113, 132, 141, 147, 158, 182, 192, 199, 208,
        229, 254, 257, 260, 283, 285, 295, 309, 341, 362, 365, 377, 383, 393, 394,
    ];

    #[test]
    fn the_validation_mask_rejects_bad_inputs() {
        let params = TrainingParams::default();
        let message = validation_mask(&["chr1"], &[1, 2], &params).unwrap_err().to_string();
        assert_eq!(message, "1 contigs do not match 2 positions");
        let zero = TrainingParams { validation_block_size: 0, ..params };
        let message = validation_mask(&["chr1"], &[1], &zero).unwrap_err().to_string();
        assert_eq!(message, "the validation block size must be positive");
    }

    #[test]
    fn fits_with_early_stopping_and_predicts() {
        let (values, labels) = noisy_calls(600);
        let params = TrainingParams {
            learning_rate: 0.5,
            max_rounds: 400,
            early_stopping_rounds: 5,
            min_child_weight: 1.0,
            ..TrainingParams::default()
        };
        let is_validation: Vec<bool> = (0..labels.len()).map(|i| i % 5 == 0).collect();
        let booster =
            fit_booster(&values, &labels, &is_validation, &["a", "b"], &params, 3, THREADS)
                .unwrap();
        let rounds = booster.boosted_rounds().unwrap();
        assert!(rounds > 0 && rounds < 400, "{rounds} rounds");
        let probabilities =
            predict_probabilities(&booster, &[0.1, 0.5, 0.5, 0.5, 0.9, 0.5], 3, &["a", "b"], 3)
                .unwrap();
        for (row, expected) in [0, 1, 2].into_iter().enumerate() {
            let p = probabilities.row(row);
            let argmax = (0..3).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
            assert_eq!(argmax, expected, "{p:?}");
        }
        let empty = predict_probabilities(&booster, &[], 0, &["a", "b"], 3).unwrap();
        assert!(empty.is_empty());

        // Without held-out calls every round is kept.
        let params = TrainingParams { max_rounds: 7, ..params };
        let none = vec![false; labels.len()];
        let booster =
            fit_booster(&values, &labels, &none, &["a", "b"], &params, 3, THREADS).unwrap();
        assert_eq!(booster.boosted_rounds().unwrap(), 7);
    }

    #[test]
    fn fitting_rejects_bad_inputs() {
        let (values, labels) = noisy_calls(3);
        let params = TrainingParams::default();
        let fit = |values: &[f32], labels: &[usize], is_validation: &[bool]| {
            fit_booster(values, labels, is_validation, &["a", "b"], &params, 3, THREADS)
                .unwrap_err()
                .to_string()
        };
        assert_eq!(
            fit(&values[..4], &labels[..2], &[true, true]),
            "no training calls left after holding out all 2 for early stopping"
        );
        assert_eq!(
            fit(&values, &labels[..2], &[false, false]),
            "6 feature values do not fill 2 calls of 2 features"
        );
        assert_eq!(fit(&values, &labels, &[false]), "1 validation flags do not match 3 calls");
        assert_eq!(fit(&values, &[0, 1, 3], &[false; 3]), "label 3 is not one of the 3 classes");
    }

    #[test]
    fn parses_the_validation_loss() {
        assert_eq!(parse_validation_loss("[3]\tvalidation-mlogloss:0.21").unwrap(), 0.21);
        assert!(parse_validation_loss("[3]\ttrain-mlogloss:0.21").is_err());
    }
}
