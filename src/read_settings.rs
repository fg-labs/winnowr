//! Read settings written as comma-separated text, e.g. `mapq20,bq10,flank5`: a MAPQ, a minimum
//! base quality and a flank.
//!
//! Two kinds of settings share the text form, the base quality and the flank, but not the meaning
//! of `mapqN`: for the pileup ([`crate::pileup::PileupOptions`]) it is the lowest MAPQ of the
//! high bin (reads of every MAPQ are counted), and for the read annotations
//! ([`crate::read_annotations::ReadAnnotationOptions`]) the lowest MAPQ a read is counted at.

use std::fmt;

use anyhow::{Result, bail};

/// The base quality a SNP base needs for its read to be counted at an allele (by default), as
/// HaplotypeCaller's default minimum base quality.
pub const MIN_BASE_QUALITY: u8 = 10;

/// Reference bases an indel read must align on outside each end of the repeat tract (by
/// default): `winnowr strlen`'s.
pub const FLANK: u64 = crate::strlen::FLANK;

/// The values of a settings text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadSettings {
    /// `mapqN`: what it means is the caller's.
    pub mapq: u8,
    /// `bqN`: the base quality a SNP base needs for its read to be counted at an allele.
    pub min_base_quality: u8,
    /// `flankN`: the aligned bases an indel read needs outside each end of the repeat tract.
    pub flank: u64,
}

impl ReadSettings {
    /// What decides how a read's allele is read: the base quality and the flank (the MAPQ only
    /// decides whether a read is counted, or its bin). Two settings with the same key read every
    /// read's allele alike.
    #[must_use]
    pub fn alleles_key(&self) -> (u8, u64) {
        (self.min_base_quality, self.flank)
    }

    /// Parses comma-separated settings over these (the defaults): `mapqN`, `bqN` and `flankN`,
    /// each at most once, in any order (`default` alone is the defaults). `kind` names the
    /// settings in errors (e.g. `pileup`).
    ///
    /// # Errors
    ///
    /// Returns an error, which says what is expected, if an item is not a setting, is given
    /// twice, or its value is not a number in range (the flank's is
    /// [`crate::strlen::check_flank`]'s).
    pub fn parse_over(self, text: &str, kind: &str) -> Result<Self> {
        self.parse_over_with(text, kind, None, |_, _| Ok(()))
    }

    /// As [`ReadSettings::parse_over`], also accepting, at most once, a setting of the caller's
    /// own: `extra` is its prefix (e.g. `maxreads`, for `maxreadsN`), which must neither start
    /// with nor be a prefix of `mapq`, `bq` or `flank`, and `parse_extra` is given its value (the
    /// text after the prefix) and the whole item.
    ///
    /// # Errors
    ///
    /// Returns an error as [`ReadSettings::parse_over`] does (naming `extra` among the expected
    /// settings), or `parse_extra`'s.
    pub fn parse_over_with<F>(
        mut self,
        text: &str,
        kind: &str,
        extra: Option<&str>,
        mut parse_extra: F,
    ) -> Result<Self>
    where
        F: FnMut(&str, &str) -> Result<()>,
    {
        debug_assert!(extra.is_none_or(|extra| {
            ["mapq", "bq", "flank"]
                .iter()
                .all(|own| !own.starts_with(extra) && !extra.starts_with(own))
        }));
        let extra_name = extra.map(|prefix| format!("{prefix}N"));
        // The item each setting was given as, in the order mapq, bq, flank, extra.
        let mut given: [Option<&str>; 4] = [None; 4];
        for item in text.split(',').map(str::trim).filter(|item| !matches!(*item, "" | "default")) {
            let extra_value = extra.and_then(|prefix| item.strip_prefix(prefix));
            let (index, name, value) = if let Some(value) = item.strip_prefix("mapq") {
                (0, "mapqN", value)
            } else if let Some(value) = item.strip_prefix("bq") {
                (1, "bqN", value)
            } else if let Some(value) = item.strip_prefix("flank") {
                (2, "flankN", value)
            } else if let (Some(value), Some(name)) = (extra_value, extra_name.as_deref()) {
                (3, name, value)
            } else {
                let expected =
                    extra_name.as_ref().map(|name| format!(", {name}")).unwrap_or_default();
                bail!(
                    "unknown {kind} setting '{item}' (expected mapqN, bqN, flankN{expected} or \
                     default)"
                );
            };
            if let Some(first) = given[index].replace(item) {
                bail!("the {kind} setting {name} is given twice ('{first}' and '{item}')");
            }
            // The whole reason is in one message: clap shows only the outermost.
            match index {
                0 => self.mapq = parse_number(value, "MAPQ", item, "0 to 255")?,
                1 => self.min_base_quality = parse_number(value, "base quality", item, "0 to 255")?,
                2 => {
                    let expected = format!("1 to {}", crate::strlen::MAX_FLANK);
                    self.flank = parse_number(value, "flank", item, &expected)?;
                }
                _ => parse_extra(value, item)?,
            }
        }
        crate::strlen::check_flank(self.flank)?;
        Ok(self)
    }
}

/// Parses a setting's number, or says in one message what `item` should have been.
///
/// # Errors
///
/// Returns an error if `value` is not a `T`, saying it is not an integer from `expected`.
pub fn parse_number<T: std::str::FromStr>(
    value: &str,
    what: &str,
    item: &str,
    expected: &str,
) -> Result<T> {
    match value.parse() {
        Ok(number) => Ok(number),
        Err(_) => bail!("bad {what} in {item}: expected an integer from {expected}"),
    }
}

impl fmt::Display for ReadSettings {
    /// Every setting, as [`ReadSettings::parse_over`] parses it back: e.g. `mapq20,bq10,flank5`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mapq{},bq{},flank{}", self.mapq, self.min_base_quality, self.flank)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const DEFAULTS: ReadSettings =
        ReadSettings { mapq: 20, min_base_quality: MIN_BASE_QUALITY, flank: FLANK };

    #[test]
    fn settings_parse_over_the_defaults_and_print() {
        assert_eq!(DEFAULTS.to_string(), "mapq20,bq10,flank5");
        for defaults in ["", "default", " default ,", "mapq20,bq10,flank5"] {
            assert_eq!(DEFAULTS.parse_over(defaults, "test").unwrap(), DEFAULTS, "{defaults:?}");
        }
        let other = DEFAULTS.parse_over(" flank2, bq0 ,mapq255", "test").unwrap();
        assert_eq!(other, ReadSettings { mapq: 255, min_base_quality: 0, flank: 2 });
        assert_eq!(other.to_string(), "mapq255,bq0,flank2");
        assert_eq!(DEFAULTS.parse_over(&other.to_string(), "test").unwrap(), other);
    }

    #[rstest]
    #[case::no_value("bq", "bad base quality in bq: expected an integer from 0 to 255")]
    #[case::not_a_number("mapqx", "bad MAPQ in mapqx: expected an integer from 0 to 255")]
    #[case::too_large("mapq256", "bad MAPQ in mapq256: expected an integer from 0 to 255")]
    #[case::negative("flank-1", "bad flank in flank-1: expected an integer from 1 to 10000")]
    #[case::no_flank("flank0", "the flank must be between 1 and 10000, got 0")]
    #[case::long_flank("flank10001", "the flank must be between 1 and 10000, got 10001")]
    #[case::unknown(
        "depth3",
        "unknown test setting 'depth3' (expected mapqN, bqN, flankN or default)"
    )]
    #[case::wrong_separator(
        "mapq20;bq10",
        "bad MAPQ in mapq20;bq10: expected an integer from 0 to 255"
    )]
    #[case::repeated(
        "mapq1,bq5,mapq30",
        "the test setting mapqN is given twice ('mapq1' and 'mapq30')"
    )]
    #[case::repeated_alike("flank5,flank5", "the test setting flankN is given twice")]
    fn bad_settings_say_what_is_expected(#[case] text: &str, #[case] expected: &str) {
        let message = DEFAULTS.parse_over(text, "test").unwrap_err().to_string();
        assert!(message.starts_with(expected), "{message}");
    }
}
