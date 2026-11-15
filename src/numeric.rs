//! Floating-point sums in the same order as the Python version's, so results match it bit for
//! bit: NumPy's pairwise `ndarray.sum()` and Python's compensated built-in `sum()`.

/// Sums values in NumPy's order (pairwise, in blocks of eight), so that sums match the Python
/// version's bit for bit.
#[must_use]
pub fn numpy_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |sum, &value| sum + value);
    }
    if n <= BLOCK {
        let mut r: [f64; 8] = values[..8].try_into().expect("eight values");
        let whole = n - n % 8;
        let mut i = 8;
        while i < whole {
            for (j, partial) in r.iter_mut().enumerate() {
                *partial += values[i + j];
            }
            i += 8;
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &value in &values[whole..] {
            sum += value;
        }
        return sum;
    }
    let mut half = n / 2;
    half -= half % 8;
    numpy_sum(&values[..half]) + numpy_sum(&values[half..])
}

/// Sums values as Python 3.12+'s built-in `sum()` does for floats: Neumaier's improved
/// Kahan–Babuška compensated summation, starting from 0.
#[must_use]
pub fn python_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for value in values {
        let total = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - total) + value;
        } else {
            compensation += (value - total) + sum;
        }
        sum = total;
    }
    // As CPython: keep the sign of a negative zero, and do not turn an infinite sum into NaN.
    if compensation != 0.0 && compensation.is_finite() {
        sum += compensation;
    }
    sum
}

/// Formats a float as Python's `str()` does: the shortest representation that round-trips, in
/// scientific notation (with a signed, at least two-digit exponent) below 1e-4 or from 1e16.
#[must_use]
pub fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // Rust's `{:e}` is the shortest round-tripping representation, e.g. `-1.25e-5`.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("`{:e}` has an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` has an integer exponent");
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let text = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{text}")
    } else {
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.unsigned_abs())
    }
}

/// Formats a metric value as the Python version's metrics files do: Python's `str(round(value,
/// 5))`. Both round the double's exact value to five decimals, ties to even.
#[must_use]
pub fn metric_float(value: f64) -> String {
    let rounded: f64 = format!("{value:.5}").parse().unwrap_or(value);
    python_float(rounded)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn numpy_sum_matches_sequential_sums_for_short_arrays() {
        assert_eq!(numpy_sum(&[]), 0.0);
        assert_eq!(numpy_sum(&[0.1, 0.2, 0.3]), 0.1 + 0.2 + 0.3);
        // Eight or more values are summed in eight interleaved partial sums.
        let values: Vec<f64> = (1..=10).map(|i| f64::from(i) * 0.1).collect();
        let r: Vec<f64> = (0..8).map(|j| values[j]).collect();
        let expected = ((r[0] + r[1]) + (r[2] + r[3]))
            + ((r[4] + r[5]) + (r[6] + r[7]))
            + values[8]
            + values[9];
        assert_eq!(numpy_sum(&values), expected);
        let long: Vec<f64> = (0..300).map(|i| f64::from(i).sqrt()).collect();
        assert!((numpy_sum(&long) - long.iter().sum::<f64>()).abs() < 1e-9);
    }

    #[test]
    fn python_sum_compensates_rounding() {
        assert_eq!(python_sum([]), 0.0);
        assert_eq!(python_sum([0.1; 10]), 1.0);
        assert_ne!([0.1; 10].iter().sum::<f64>(), 1.0);
        assert_eq!(python_sum([1e100, 1.0, -1e100]), 1.0);
        assert_eq!(python_sum([f64::INFINITY, 1.0]), f64::INFINITY);
        // The two orders of these terms sum to the same value, as in Python.
        let terms = [
            -6.397755605663211,
            -6.057451687708123,
            -1.7304662901906387,
            -12.114903375416246,
            -6.057451687708123,
        ];
        let swapped = [terms[0], terms[1], terms[3], terms[2], terms[4]];
        assert_eq!(python_sum(terms), -32.35802864668634);
        assert_eq!(python_sum(swapped), -32.35802864668634);
    }

    #[rstest]
    #[case(0.0, "0.0")]
    #[case(-0.0, "-0.0")]
    #[case(1.0, "1.0")]
    #[case(1.01, "1.01")]
    #[case(0.5, "0.5")]
    #[case(1e-4, "0.0001")]
    #[case(1.0 - 1e-10, "0.9999999999")]
    #[case(1e-5, "1e-05")]
    #[case(-2.5e-7, "-2.5e-07")]
    #[case(0.00867123, "0.00867123")]
    #[case(123456.0, "123456.0")]
    #[case(1e15, "1000000000000000.0")]
    #[case(1e16, "1e+16")]
    #[case(1.5e300, "1.5e+300")]
    #[case(f64::NAN, "nan")]
    #[case(f64::INFINITY, "inf")]
    fn formats_floats_as_python(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(python_float(value), expected);
    }

    // Expected values from Python's `str(round(value, 5))`.
    #[rstest]
    #[case(0.9941015342732787, "0.9941")]
    #[case(0.00198187845033726, "0.00198")]
    #[case(1.0 - 1e-10, "1.0")]
    #[case(1.01, "1.01")]
    #[case(0.0, "0.0")]
    #[case(0.015625, "0.01562")]
    #[case(0.046875, "0.04688")]
    #[case(-0.00026699862934476126, "-0.00027")]
    #[case(0.000001, "0.0")]
    #[case(f64::NAN, "nan")]
    fn formats_metrics_as_python(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(metric_float(value), expected);
    }
}
