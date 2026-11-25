//! Where the features that HaplotypeCaller's gVCF mode alone annotates come from.
//!
//! Six features (`mq`, `dp_norm`, `qd`, `as_alt_frac`, `sor` and `n_alt`) are read from the raw
//! annotations of a gVCF (`RAW_MQandDP`, `QUALapprox`, `VarDP`, `AS_RAW_MQ`) and FORMAT/SB. A
//! standard VCF (e.g. from GenotypeGVCFs, or HaplotypeCaller without `-ERC`) lacks the raw
//! annotations; the [`FeatureSource::Vcf`] source derives what it can from the fields such a VCF
//! has instead (see [`crate::raw_records::AnnotationInputs`]), and the
//! [`FeatureSource::VcfAlignments`] source measures the mapping qualities and depth from the reads
//! too. The values differ slightly, so a model scores only calls whose features come from the
//! source it was trained on: feature tables and models record their source.

use std::fmt;

use anyhow::{Context, Result, bail};

use crate::read_annotations::ReadAnnotationOptions;

/// The name of the [`FeatureSource::VcfAlignments`] source, which its read settings follow after
/// a `:` (e.g. `alignments:mapq20,bq10,flank5`).
const ALIGNMENTS: &str = "alignments";

/// Where the gVCF-only features of a feature table, or of the tables a model was trained on, come
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FeatureSource {
    /// The raw annotations of the VCF's records, or of the gVCF it was genotyped from.
    #[default]
    Gvcf,
    /// The standard annotations of the VCF itself (INFO/MQ, INFO/DP, FORMAT/AD, FORMAT/PL, and
    /// FORMAT/SB or INFO/SOR), with no gVCF; `as_alt_frac` is missing.
    Vcf,
    /// As [`FeatureSource::Vcf`], but with the mapping qualities and depth (`mq`, `dp_norm`,
    /// `as_alt_frac`, and VarDP's fallback behind `qd`) measured from the reads with these
    /// settings: INFO/MQ is not read, and INFO/DP feeds only the `dp` feature. Features measured
    /// with other settings are of another source. `extract --alignments` chooses it, with the
    /// settings of `--alignment-settings`.
    VcfAlignments(ReadAnnotationOptions),
}

impl FeatureSource {
    /// True for the default source, [`FeatureSource::Gvcf`]. The default source is not recorded
    /// in feature tables and models, so that theirs are as before sources existed.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == FeatureSource::Gvcf
    }
}

impl fmt::Display for FeatureSource {
    /// The source as [`std::str::FromStr`] parses it back: `gvcf`, `vcf`, or `alignments:`
    /// and the read settings (e.g. `alignments:mapq20,bq10,flank5`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureSource::Gvcf => f.write_str("gvcf"),
            FeatureSource::Vcf => f.write_str("vcf"),
            FeatureSource::VcfAlignments(options) => write!(f, "{ALIGNMENTS}:{options}"),
        }
    }
}

impl std::str::FromStr for FeatureSource {
    type Err = anyhow::Error;

    /// Parses `gvcf`, `vcf`, or `alignments` with its read settings after a `:` (the
    /// defaults without them), ignoring case.
    fn from_str(text: &str) -> Result<Self> {
        let lower = text.trim().to_ascii_lowercase();
        let (name, settings) = match lower.split_once(':') {
            Some((name, settings)) => (name, Some(settings)),
            None => (lower.as_str(), None),
        };
        match (name, settings) {
            ("gvcf", None) => Ok(FeatureSource::Gvcf),
            ("vcf", None) => Ok(FeatureSource::Vcf),
            (ALIGNMENTS, settings) => {
                let options = settings
                    .unwrap_or("default")
                    .parse()
                    .with_context(|| format!("bad read settings in feature source '{text}'"))?;
                Ok(FeatureSource::VcfAlignments(options))
            }
            _ => bail!(
                "unknown feature source '{text}' (expected gvcf, vcf, or {ALIGNMENTS} with \
                 optional read settings, e.g. {ALIGNMENTS}:mapq20,bq10,flank5)"
            ),
        }
    }
}

impl serde::Serialize for FeatureSource {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for FeatureSource {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::gvcf(FeatureSource::Gvcf, "gvcf")]
    #[case::vcf(FeatureSource::Vcf, "vcf")]
    #[case::alignments(
        FeatureSource::VcfAlignments(ReadAnnotationOptions::default()),
        "alignments:mapq20,bq10,flank5"
    )]
    #[case::alignments_with_settings(
        FeatureSource::VcfAlignments(ReadAnnotationOptions {
            settings: crate::read_settings::ReadSettings { mapq: 1, min_base_quality: 0, flank: 2 }
        }),
        "alignments:mapq1,bq0,flank2"
    )]
    fn round_trips_through_text_and_json(#[case] source: FeatureSource, #[case] text: &str) {
        assert_eq!(source.to_string(), text);
        assert_eq!(text.parse::<FeatureSource>().unwrap(), source);
        assert_eq!(text.to_uppercase().parse::<FeatureSource>().unwrap(), source);
        let json = serde_json::to_string(&source).unwrap();
        assert_eq!(json, format!("\"{text}\""));
        assert_eq!(serde_json::from_str::<FeatureSource>(&json).unwrap(), source);
    }

    #[rstest]
    #[case::empty("")]
    #[case::unknown("bam")]
    #[case::settings_of_another_source("vcf:mapq20")]
    fn rejects_unknown_sources(#[case] text: &str) {
        let error = text.parse::<FeatureSource>().unwrap_err().to_string();
        assert!(error.starts_with("unknown feature source"), "{error}");
        let json = format!("\"{text}\"");
        let error = serde_json::from_str::<FeatureSource>(&json).unwrap_err().to_string();
        assert!(error.starts_with("unknown feature source"), "{error}");
    }

    /// Only the gVCF source is the default, which tables and models do not record.
    #[test]
    fn only_the_gvcf_source_is_the_default() {
        assert!(FeatureSource::Gvcf.is_default());
        assert!(!FeatureSource::Vcf.is_default());
        assert!(!FeatureSource::VcfAlignments(ReadAnnotationOptions::default()).is_default());
    }

    /// The read settings are part of the source: without them they are the defaults, and other
    /// settings are another source.
    #[test]
    fn alignment_sources_differ_by_their_read_settings() {
        let default = FeatureSource::VcfAlignments(ReadAnnotationOptions::default());
        assert_eq!("alignments".parse::<FeatureSource>().unwrap(), default);
        assert_eq!("alignments:flank5".parse::<FeatureSource>().unwrap(), default);
        let dragen: FeatureSource = "alignments:mapq1".parse().unwrap();
        assert_ne!(dragen, default);
        assert_eq!(dragen.to_string(), "alignments:mapq1,bq10,flank5");
        let error = format!("{:#}", "alignments:depth3".parse::<FeatureSource>().unwrap_err());
        assert_eq!(
            error,
            "bad read settings in feature source 'alignments:depth3': unknown alignment \
             setting 'depth3' (expected mapqN, bqN, flankN or default)"
        );
    }
}
