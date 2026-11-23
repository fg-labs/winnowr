//! Read-measured annotations: the mapping-quality sums and depth of a call measured from the
//! alignments, for calls whose VCF has no raw annotations (`RAW_MQandDP`, `AS_RAW_MQ`) and no
//! gVCF to take them from. The reads are measured in the pileup pass (see
//! [`crate::pileup::measure_annotations`]) and joined to the calls in [`crate::raw_records`].
//!
//! The sums approximate HaplotypeCaller's. HaplotypeCaller sums over the reads in its likelihoods
//! after local reassembly and downsampling, counts supplementary alignments, takes the reads
//! within 2 bases of the variant, and assigns reads to alleles by likelihood. Here the reads are
//! those aligned over the call as they are in the alignments. Both count the two reads of an
//! overlapping mate pair.

use std::fmt;

use anyhow::{Result, bail};

use crate::read_pass::MAX_READS;
use crate::read_settings::{FLANK, MIN_BASE_QUALITY, ReadSettings};

/// The MAPQ a read needs to be counted (by default): HaplotypeCaller's default minimum (it is 1
/// in DRAGEN mode).
pub const MIN_MAPQ: u8 = 20;

/// The MAPQ of a read whose mapping quality is not available, which is never counted (as
/// HaplotypeCaller never counts it).
pub const UNAVAILABLE_MAPQ: u8 = 255;

/// Which reads are counted in the read-measured annotations, and how their alleles are read: the
/// shared [`ReadSettings`], whose `mapq` is here the lowest MAPQ a read is counted at.
///
/// Written as comma-separated settings (see [`fmt::Display`] and [`std::str::FromStr`]), e.g.
/// `mapq20,bq10,flank5`. The text is that of [`crate::pileup::PileupOptions`], but `mapqN` is
/// here the lowest MAPQ a read is counted at, where the pileup's is the lowest MAPQ of its high
/// bin (see [`crate::read_settings`]). The flank is both the bases on each side of a call's REF
/// within which a read counts towards its depth, and the aligned bases an indel read needs outside
/// each end of the repeat tract to be counted at an allele.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadAnnotationOptions {
    /// The settings; `mapq` is the MAPQ a read needs to be counted.
    pub settings: ReadSettings,
}

impl Default for ReadAnnotationOptions {
    /// MAPQ [`MIN_MAPQ`], and the pileup's default base quality and flank.
    fn default() -> Self {
        let settings =
            ReadSettings { mapq: MIN_MAPQ, min_base_quality: MIN_BASE_QUALITY, flank: FLANK };
        ReadAnnotationOptions { settings }
    }
}

impl ReadAnnotationOptions {
    /// The MAPQ a read needs to be counted.
    #[must_use]
    pub fn min_mapq(&self) -> u8 {
        self.settings.mapq
    }
}

impl fmt::Display for ReadAnnotationOptions {
    /// Every setting, as [`std::str::FromStr`] parses it back: e.g. `mapq20,bq10,flank5`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.settings.fmt(f)
    }
}

impl std::str::FromStr for ReadAnnotationOptions {
    type Err = anyhow::Error;

    /// Parses comma-separated settings over the defaults: `mapqN` (the minimum MAPQ), `bqN` and
    /// `flankN` (`default` alone is the defaults).
    fn from_str(text: &str) -> Result<Self> {
        let settings = ReadAnnotationOptions::default().settings.parse_over(text, "alignment")?;
        if settings.mapq == UNAVAILABLE_MAPQ {
            bail!(
                "the minimum MAPQ must be from 0 to {}: no read is counted at {UNAVAILABLE_MAPQ}, \
                 which means the MAPQ is not available",
                UNAVAILABLE_MAPQ - 1
            );
        }
        Ok(ReadAnnotationOptions { settings })
    }
}

/// The mapping qualities of the reads counted at one call: the sums the raw annotations of a
/// gVCF hold (`RAW_MQandDP` and `AS_RAW_MQ`), measured from the alignments.
///
/// The reads counted are those overlapping the call's REF with the settings' flank on each side
/// that have at least the settings' MAPQ (see [`ReadAnnotationOptions`]), up to [`MAX_READS`] of
/// them. Overlapping mates are both counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MapqSums {
    /// The reads counted.
    pub n_reads: u64,
    /// The sum of their squared MAPQs.
    pub sum_sq: u64,
    /// The sum of the squared MAPQs of those with the reference allele.
    pub ref_sum_sq: u64,
    /// The sum of the squared MAPQs of those with an alternate allele the genotype calls (any
    /// alternate allele, for a genotype that is not a variant), as ReblockGVCF's `AS_RAW_MQ`
    /// keeps only the called alleles. Reads with an uncalled or other allele, and reads whose
    /// allele cannot be read, are in neither allele's sum.
    pub alt_sum_sq: u64,
    /// False if the alleles the sums are of cannot be read from a read: the record, or the
    /// alternate alleles its genotype calls, has no SNP or pure insertion or deletion allele, or
    /// the anchors its alleles are read between do not take in every alternate allele's change
    /// (e.g. `CAAAAGT>CAAAGT,CAAAA` called `0/1`, whose GT deletion lies past the anchors of the
    /// A deletion). The allele sums are then 0 and mean nothing.
    pub alleles_read: bool,
}

impl MapqSums {
    /// Counts a read with mapping quality `mapq`.
    pub fn add(&mut self, mapq: u8) {
        self.n_reads += 1;
        self.sum_sq += squared(mapq);
    }

    /// Counts a read already [added](Self::add) as having the reference allele (`alt` false) or
    /// a called alternate allele.
    pub fn add_allele(&mut self, mapq: u8, alt: bool) {
        if alt {
            self.alt_sum_sq += squared(mapq);
        } else {
            self.ref_sum_sq += squared(mapq);
        }
    }

    /// True once no more reads are counted: [`MAX_READS`] were, so the call's depth is at least
    /// that, and its sums are those of the first [`MAX_READS`] reads.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.n_reads >= MAX_READS
    }
}

/// The calls of one contig whose reads [`summarise`] summarised.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContigReads {
    /// The calls.
    pub calls: usize,
    /// Those at which no read was counted.
    pub without_reads: usize,
    /// Those whose reads reached the cap ([`MapqSums::is_full`]).
    pub at_cap: usize,
}

/// Summarises the read annotations of the calls measured with `options` on `contigs`, given as
/// each call's contig and sums, in contig order. Warns with the number of calls, per contig, at
/// which no read was counted (their `mq` and `as_alt_frac` are missing, and their `dp_norm` is 0,
/// or missing for every call when the calls' median depth is 0 and no `--dp-median` is given),
/// and with the number whose reads reached the cap of [`MAX_READS`] (their depth saturates).
/// Returns those counts per contig of `contigs`.
///
/// # Errors
///
/// Returns an error if there are calls and no read was counted at any of them (e.g. the reads are
/// of other contigs, or none passes the settings).
pub fn summarise<'a, S, I>(
    calls: I,
    contigs: &[S],
    options: &ReadAnnotationOptions,
) -> Result<Vec<ContigReads>>
where
    S: AsRef<str>,
    I: IntoIterator<Item = (&'a str, &'a MapqSums)>,
{
    let mut counts = vec![ContigReads::default(); contigs.len()];
    // The calls come in contig order, so a contig is looked up only where they move to another.
    let mut current: Option<(&str, Option<usize>)> = None;
    for (chrom, sums) in calls {
        let index = match current {
            Some((name, index)) if name == chrom => index,
            _ => {
                let index = contigs.iter().position(|contig| contig.as_ref() == chrom);
                current = Some((chrom, index));
                index
            }
        };
        if let Some(count) = index.map(|index| &mut counts[index]) {
            count.calls += 1;
            count.without_reads += usize::from(sums.n_reads == 0);
            count.at_cap += usize::from(sums.is_full());
        }
    }
    let total = |count: fn(&ContigReads) -> usize| counts.iter().map(count).sum::<usize>();
    let (n_calls, n_without) = (total(|c| c.calls), total(|c| c.without_reads));
    if n_calls > 0 && n_without == n_calls {
        bail!(
            "no read was counted at any of the {n_calls} calls with alignment settings {options}: \
             are the reads of this sample, aligned to the VCF's reference and contigs, and do any \
             pass the settings?"
        );
    }
    for warning in warnings(&counts, contigs, options) {
        log::warn!("{warning}");
    }
    Ok(counts)
}

/// The warnings of [`summarise`] about `counts`, the calls per contig of `contigs`: the calls
/// without a read counted, and those at the cap, each with its counts by contig (none when there
/// are no such calls).
fn warnings<S: AsRef<str>>(
    counts: &[ContigReads],
    contigs: &[S],
    options: &ReadAnnotationOptions,
) -> Vec<String> {
    // The calls of each contig with `count` of them, as `contig count of calls`, and their total.
    let by_contig = |count: fn(&ContigReads) -> usize| -> (usize, String) {
        let per_contig: Vec<String> = contigs
            .iter()
            .zip(counts)
            .filter(|(_, reads)| count(reads) > 0)
            .map(|(contig, reads)| {
                format!("{} {} of {}", contig.as_ref(), count(reads), reads.calls)
            })
            .collect();
        (counts.iter().map(count).sum(), per_contig.join(", "))
    };
    let mut warnings = Vec::new();
    let (n_without, per_contig) = by_contig(|c| c.without_reads);
    if n_without > 0 {
        warnings.push(format!(
            "{n_without} records had no read counted with alignment settings {options}, so their \
             mq and as_alt_frac are missing and their dp_norm is 0, or missing for every call if \
             the calls' median depth is 0 (give --dp-median for a depth to divide by) (by \
             contig: {per_contig})"
        ));
    }
    let (n_full, per_contig) = by_contig(|c| c.at_cap);
    if n_full > 0 {
        warnings.push(format!(
            "{n_full} records reached the cap of {MAX_READS} reads counted, so their depth is at \
             most that (and their dp_norm at most that over the median depth) (by contig: \
             {per_contig})"
        ));
    }
    warnings
}

/// A MAPQ squared.
fn squared(mapq: u8) -> u64 {
    u64::from(mapq) * u64::from(mapq)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The settings over the annotations' defaults, whose MAPQ is a minimum: any MAPQ but the
    /// unavailable one (the shared parsing is tested in [`crate::read_settings`]).
    #[test]
    fn settings_parse_over_the_defaults_and_reject_the_unavailable_mapq() {
        let default = ReadAnnotationOptions::default();
        assert_eq!(default.settings, ReadSettings { mapq: 20, min_base_quality: 10, flank: 5 });
        assert_eq!(default.to_string(), "mapq20,bq10,flank5");
        // DRAGEN mode's minimum MAPQ, over the other defaults.
        let dragen: ReadAnnotationOptions = "mapq1".parse().unwrap();
        assert_eq!(dragen.min_mapq(), 1);
        assert_eq!(dragen.to_string(), "mapq1,bq10,flank5");
        assert_eq!("mapq254".parse::<ReadAnnotationOptions>().unwrap().min_mapq(), 254);
        let message = "mapq255".parse::<ReadAnnotationOptions>().unwrap_err().to_string();
        assert!(message.starts_with("the minimum MAPQ must be from 0 to 254"), "{message}");
        let message = "depth3".parse::<ReadAnnotationOptions>().unwrap_err().to_string();
        assert!(message.starts_with("unknown alignment setting 'depth3'"), "{message}");
    }

    #[test]
    fn sums_add_reads_and_their_alleles() {
        let mut sums = MapqSums { alleles_read: true, ..MapqSums::default() };
        for (mapq, allele) in [(60, Some(false)), (20, Some(true)), (30, None)] {
            sums.add(mapq);
            if let Some(alt) = allele {
                sums.add_allele(mapq, alt);
            }
        }
        let expected = MapqSums {
            n_reads: 3,
            sum_sq: 3600 + 400 + 900,
            ref_sum_sq: 3600,
            alt_sum_sq: 400,
            alleles_read: true,
        };
        assert_eq!(sums, expected);
        assert!(!sums.is_full());
        assert!(MapqSums { n_reads: MAX_READS, ..sums }.is_full());
    }

    /// The sums of `n_reads` reads of MAPQ 60.
    fn of(n_reads: u64) -> MapqSums {
        MapqSums { n_reads, sum_sq: n_reads * 3600, alleles_read: true, ..MapqSums::default() }
    }

    /// The calls without reads, and at the cap, are counted per contig: e.g. a contig the reads
    /// lack.
    #[test]
    fn calls_without_reads_are_counted_per_contig() {
        let (full, none, some) = (of(MAX_READS), of(0), of(5));
        let calls = [("chr1", &full), ("chr1", &none), ("chr1", &some), ("chr2", &none)];
        let options = ReadAnnotationOptions::default();
        let counts = summarise(calls, &["chr1", "chr2"], &options).unwrap();
        assert_eq!(
            counts,
            [
                ContigReads { calls: 3, without_reads: 1, at_cap: 1 },
                ContigReads { calls: 1, without_reads: 1, at_cap: 0 },
            ]
        );
        assert!(summarise([], &["chr1"], &options).unwrap()[0] == ContigReads::default());
        // The warnings give the counts by contig.
        let [without, at_cap] = warnings(&counts, &["chr1", "chr2"], &options)
            .try_into()
            .unwrap_or_else(|warnings| panic!("{warnings:?}"));
        assert!(without.starts_with("2 records had no read counted"), "{without}");
        assert!(without.ends_with("(by contig: chr1 1 of 3, chr2 1 of 1)"), "{without}");
        assert!(
            at_cap.starts_with(&format!("1 records reached the cap of {MAX_READS}")),
            "{at_cap}"
        );
        assert!(at_cap.ends_with("(by contig: chr1 1 of 3)"), "{at_cap}");
        // Without such calls, nothing is said.
        let quiet = [ContigReads { calls: 2, without_reads: 0, at_cap: 0 }];
        assert!(warnings(&quiet, &["chr1"], &options).is_empty());
        let capped = [ContigReads { calls: 2, without_reads: 0, at_cap: 2 }];
        let [warning] = warnings(&capped, &["chr1"], &options).try_into().unwrap();
        assert!(warning.contains("reached the cap"), "{warning}");
    }

    /// Without a read counted at any call (e.g. the reads are of other contigs, or none passes the
    /// settings), the features would all be missing.
    #[test]
    fn a_read_must_be_counted_at_some_call() {
        let none = of(0);
        let options = ReadAnnotationOptions::default();
        let error = summarise([("chr1", &none), ("chr1", &none)], &["chr1"], &options).unwrap_err();
        assert!(
            error.to_string().starts_with(
                "no read was counted at any of the 2 calls with alignment settings \
                 mapq20,bq10,flank5"
            ),
            "{error}"
        );
    }
}
