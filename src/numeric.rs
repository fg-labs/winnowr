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

#[cfg(test)]
mod tests {
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
}
