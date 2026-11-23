//! Where the features that HaplotypeCaller's gVCF mode alone annotates come from.
//!
//! Six features (`mq`, `dp_norm`, `qd`, `as_alt_frac`, `sor` and `n_alt`) are read from the raw
//! annotations of a gVCF (`RAW_MQandDP`, `QUALapprox`, `VarDP`, `AS_RAW_MQ`) and FORMAT/SB. A
//! standard VCF (e.g. from GenotypeGVCFs, or HaplotypeCaller without `-ERC`) lacks the raw
//! annotations; the [`FeatureSource::Vcf`] source derives what it can from the fields such a VCF
//! has instead (see [`crate::raw_records::AnnotationInputs`]). The values differ slightly, so a
//! model scores only calls whose features come from the source it was trained on: feature tables
//! and models record their source.

use std::fmt;

use anyhow::{Result, bail};

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
}

impl FeatureSource {
    /// Every source, in the order `--help` lists them.
    pub const ALL: [FeatureSource; 2] = [FeatureSource::Gvcf, FeatureSource::Vcf];

    /// True for the default source, [`FeatureSource::Gvcf`]. The default source is not recorded
    /// in feature tables and models, so that theirs are as before sources existed.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == FeatureSource::Gvcf
    }

    /// The source's name, as [`std::str::FromStr`] parses it.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            FeatureSource::Gvcf => "gvcf",
            FeatureSource::Vcf => "vcf",
        }
    }
}

impl fmt::Display for FeatureSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for FeatureSource {
    type Err = anyhow::Error;

    /// Parses `gvcf` or `vcf`, ignoring case.
    fn from_str(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "gvcf" => Ok(FeatureSource::Gvcf),
            "vcf" => Ok(FeatureSource::Vcf),
            _ => bail!("unknown feature source '{text}' (expected gvcf or vcf)"),
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
    #[case::alignments("vcf+alignments")]
    fn rejects_unknown_sources(#[case] text: &str) {
        let error = text.parse::<FeatureSource>().unwrap_err().to_string();
        assert!(error.starts_with("unknown feature source"), "{error}");
        let json = format!("\"{text}\"");
        let error = serde_json::from_str::<FeatureSource>(&json).unwrap_err().to_string();
        assert!(error.starts_with("unknown feature source"), "{error}");
    }
}
