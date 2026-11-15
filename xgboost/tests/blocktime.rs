//! How long idle OpenMP threads wait (libomp's `KMP_BLOCKTIME`) must not change a trained model.
//!
//! With patch 0002, XGBoost keeps its OpenMP teams at the requested size when the blocktime is
//! positive and caps them at the work size when it is 0. libomp reads `KMP_BLOCKTIME` once, when
//! it starts, so each setting trains in a fresh process: this test runs its own binary again, once
//! per setting, and compares the models. With libgomp (Linux) the setting is ignored, and the
//! models are equal too.

use std::num::NonZeroUsize;
use std::process::Command;

use winnowr_xgboost::{Booster, DMatrix};

/// Set in the child processes, to the blocktime they train with.
const CHILD: &str = "WINNOWR_XGBOOST_BLOCKTIME_CHILD";

/// Trains a small multi-class lossguide model with 4 threads and prints it as XGBoost JSON.
#[test]
fn train_and_print_a_model() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let (rows, cols) = (2_000, 6);
    // A fixed, varied matrix: a linear congruential sequence, with some values missing.
    let mut state: u64 = 12_345;
    let mut next = || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f32 / (1u64 << 31) as f32
    };
    let features: Vec<f32> =
        (0..rows * cols).map(|_| if next() < 0.05 { f32::NAN } else { next() }).collect();
    let labels: Vec<f32> = features
        .chunks(cols)
        .map(|row| f32::from(u8::from(row[0] > 0.5) + u8::from(row[1] > 0.7)))
        .collect();
    let mut train = DMatrix::from_dense(&features, rows, cols, f32::NAN).unwrap();
    train.set_labels(&labels).unwrap();
    let threads = NonZeroUsize::new(4).unwrap();
    let mut booster = Booster::new(&[&train], threads).unwrap();
    booster
        .set_params(&[
            ("objective", "multi:softprob"),
            ("num_class", "3"),
            ("tree_method", "hist"),
            ("grow_policy", "lossguide"),
            ("max_leaves", "31"),
            ("max_depth", "0"),
            ("seed", "1"),
        ])
        .unwrap();
    for iteration in 0..10 {
        booster.update(iteration, &train).unwrap();
    }
    let json = String::from_utf8(booster.to_json().unwrap()).unwrap();
    println!("MODEL:{json}");
}

#[test]
fn the_blocktime_does_not_change_the_model() {
    if std::env::var_os(CHILD).is_some() {
        return;
    }
    let train_with = |blocktime: &str| -> String {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "train_and_print_a_model", "--nocapture", "--test-threads=1"])
            .env(CHILD, blocktime)
            .env("KMP_BLOCKTIME", blocktime)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "KMP_BLOCKTIME={blocktime}: {stdout}");
        // libtest prints "test train_and_print_a_model ... " before the test's own output.
        let model = stdout.lines().find_map(|line| line.split_once("MODEL:").map(|(_, json)| json));
        model
            .unwrap_or_else(|| panic!("KMP_BLOCKTIME={blocktime} printed no model: {stdout}"))
            .to_string()
    };
    let (sleeping, waiting) = (train_with("0"), train_with("200"));
    assert!(sleeping.len() > 1000, "{sleeping}");
    assert!(sleeping == waiting, "the models differ between KMP_BLOCKTIME=0 and 200");
}
