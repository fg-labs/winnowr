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

use crate::read_settings::{FLANK, MIN_BASE_QUALITY, ReadSettings, parse_number};

/// The MAPQ a read needs to be counted (by default): HaplotypeCaller's default minimum (it is 1
/// in DRAGEN mode).
pub const MIN_MAPQ: u8 = 20;

/// The MAPQ of a read whose mapping quality is not available, which is never counted (as
/// HaplotypeCaller never counts it).
pub const UNAVAILABLE_MAPQ: u8 = 255;

/// The cap implied by alignment-source text without `maxreadsN`, including existing tables and
/// models. This meaning is permanent, independent of pileup/strlen limits or future CLI defaults.
pub const LEGACY_MAX_READS: u64 = 600;

/// MAPQ 255 is excluded before summing, so every squared MAPQ is at most 254 squared. This cap
/// keeps the total and each allele's subset sum within u64 without checks in the read loop.
const MAX_READS_LIMIT: u64 = u64::MAX / (254 * 254);

/// Which reads are counted in the read-measured annotations, and how their alleles are read: the
/// shared [`ReadSettings`], whose `mapq` is here the lowest MAPQ a read is counted at.
///
/// Written as comma-separated settings (see [`fmt::Display`] and [`std::str::FromStr`]), e.g.
/// `mapq20,bq10,flank5,maxreads1200`. The MAPQ, base quality and flank items are those of
/// [`crate::pileup::PileupOptions`], but `mapqN` is here the lowest MAPQ a read is counted at,
/// where the pileup's is the lowest MAPQ of its high bin (see [`crate::read_settings`]). The
/// flank is both the bases on each side of a call's REF within which a read counts towards its
/// depth, and the aligned bases an indel read needs outside each end of the repeat tract to be
/// counted at an allele. `maxreadsN` caps only these annotations, not pileup or strlen evidence;
/// omitting it always means [`LEGACY_MAX_READS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadAnnotationOptions {
    /// The settings; `mapq` is the MAPQ a read needs to be counted.
    pub settings: ReadSettings,
    /// A validated positive cap whose MAPQ-squared sums cannot overflow.
    max_reads: u64,
}

impl Default for ReadAnnotationOptions {
    /// MAPQ [`MIN_MAPQ`], the pileup's default base quality and flank, and the legacy 600-read cap.
    fn default() -> Self {
        let settings =
            ReadSettings { mapq: MIN_MAPQ, min_base_quality: MIN_BASE_QUALITY, flank: FLANK };
        ReadAnnotationOptions { settings, max_reads: LEGACY_MAX_READS }
    }
}

impl ReadAnnotationOptions {
    /// The MAPQ a read needs to be counted.
    #[must_use]
    pub fn min_mapq(&self) -> u8 {
        self.settings.mapq
    }

    /// The maximum qualifying reads counted at a call, independently of pileup evidence.
    #[must_use]
    pub fn max_reads(&self) -> u64 {
        self.max_reads
    }
}

impl fmt::Display for ReadAnnotationOptions {
    /// Canonical settings text. The legacy cap is omitted so existing source metadata retains its
    /// bytes; every other cap is explicit, even if a future CLI default changes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.settings.fmt(f)?;
        if self.max_reads != LEGACY_MAX_READS {
            write!(f, ",maxreads{}", self.max_reads)?;
        }
        Ok(())
    }
}

impl std::str::FromStr for ReadAnnotationOptions {
    type Err = anyhow::Error;

    /// Parses comma-separated `mapqN` (the minimum MAPQ), `bqN`, `flankN` and `maxreadsN`, each
    /// at most once. Omitted caps, including legacy model strings, always mean 600 reads.
    fn from_str(text: &str) -> Result<Self> {
        let mut max_reads = LEGACY_MAX_READS;
        let expected = format!("1 to {MAX_READS_LIMIT}");
        let parse_max_reads = |value: &str, item: &str| {
            let count: u64 = parse_number(value, "maximum read count", item, &expected)?;
            if !(1..=MAX_READS_LIMIT).contains(&count) {
                bail!("bad maximum read count in {item}: expected an integer from {expected}");
            }
            max_reads = count;
            Ok(())
        };
        let settings = ReadAnnotationOptions::default().settings.parse_over_with(
            text,
            "alignment",
            Some("maxreads"),
            parse_max_reads,
        )?;
        if settings.mapq == UNAVAILABLE_MAPQ {
            bail!(
                "the minimum MAPQ must be from 0 to {}: no read is counted at {UNAVAILABLE_MAPQ}, \
                 which means the MAPQ is not available",
                UNAVAILABLE_MAPQ - 1
            );
        }
        Ok(ReadAnnotationOptions { settings, max_reads })
    }
}

/// The mapping qualities of the reads counted at one call: the sums the raw annotations of a
/// gVCF hold (`RAW_MQandDP` and `AS_RAW_MQ`), measured from the alignments.
///
/// The reads counted are those overlapping the call's REF with the settings' flank on each side
/// that have at least the settings' MAPQ, up to their [`ReadAnnotationOptions::max_reads`] cap.
/// Overlapping mates are both counted.
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

    /// True once `max_reads` qualifying reads were counted. The depth is at least that, and the
    /// sums are those of the first `max_reads` reads.
    #[must_use]
    pub fn is_full(&self, max_reads: u64) -> bool {
        self.n_reads >= max_reads
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
/// and with the number whose reads reached the annotations' configured cap (their depth saturates).
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
            count.at_cap += usize::from(sums.is_full(options.max_reads()));
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
        let max_reads = options.max_reads();
        warnings.push(format!(
            "{n_full} records reached the cap of {max_reads} reads counted, so their depth is at \
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
    fn maxreads_settings_preserve_legacy_text_and_record_nondefault_caps() {
        let legacy = ReadAnnotationOptions::default();
        for cap in [1, 2, 599, 600, 601, 1200, MAX_READS_LIMIT] {
            let options: ReadAnnotationOptions = format!("maxreads{cap}").parse().unwrap();
            let expected = if cap == 600 {
                "mapq20,bq10,flank5".to_string()
            } else {
                format!("mapq20,bq10,flank5,maxreads{cap}")
            };
            assert_eq!(options.to_string(), expected);
            assert_eq!(options.max_reads(), cap);
            assert_eq!(expected.parse::<ReadAnnotationOptions>().unwrap(), options);
            assert_eq!(options == legacy, cap == 600);
        }
        assert_eq!("default,maxreads600".parse::<ReadAnnotationOptions>().unwrap(), legacy);
        let options: ReadAnnotationOptions = " maxreads1200 , bq0 ,mapq1,flank2".parse().unwrap();
        assert_eq!(options.to_string(), "mapq1,bq0,flank2,maxreads1200");
    }

    #[test]
    fn maxreads_rejects_nonpositive_overflowing_and_duplicate_caps() {
        assert_eq!(MAX_READS_LIMIT, 285_925_104_992_708);
        let too_large = (MAX_READS_LIMIT + 1).to_string();
        for value in ["0", "-1", "", "x", "18446744073709551616", too_large.as_str()] {
            let error = format!("maxreads{value}")
                .parse::<ReadAnnotationOptions>()
                .unwrap_err()
                .to_string();
            assert_eq!(
                error,
                format!(
                    "bad maximum read count in maxreads{value}: expected an integer from 1 to \
                     {MAX_READS_LIMIT}"
                )
            );
        }
        let error =
            "maxreads1,default,maxreads2".parse::<ReadAnnotationOptions>().unwrap_err().to_string();
        assert_eq!(
            error,
            "the alignment setting maxreadsN is given twice ('maxreads1' and 'maxreads2')"
        );
    }

    /// Numbers are read as the other settings' are, and the text records the number itself, so
    /// equal caps are one source however they were written.
    #[test]
    fn maxreads_text_is_canonical_for_equal_caps() {
        for (text, expected) in [
            ("maxreads0600", "mapq20,bq10,flank5"),
            ("maxreads+600", "mapq20,bq10,flank5"),
            ("maxreads01200", "mapq20,bq10,flank5,maxreads1200"),
        ] {
            let options: ReadAnnotationOptions = text.parse().unwrap();
            assert_eq!(options.to_string(), expected, "{text}");
            assert_eq!(expected.parse::<ReadAnnotationOptions>().unwrap(), options, "{text}");
        }
    }

    #[test]
    fn maxreads_summary_uses_the_annotation_cap() {
        let options: ReadAnnotationOptions = "maxreads601".parse().unwrap();
        let (below, full) = (of(600), of(601));
        let counts = summarise([("chr1", &below), ("chr1", &full)], &["chr1"], &options).unwrap();
        assert_eq!(counts, [ContigReads { calls: 2, without_reads: 0, at_cap: 1 }]);
        let messages = warnings(&counts, &["chr1"], &options);
        assert_eq!(messages.len(), 1);
        assert!(messages[0].starts_with("1 records reached the cap of 601 reads counted"));
    }

    #[test]
    fn maxreads_safe_limit_keeps_total_and_allele_sums_in_range() {
        let safe_max = MAX_READS_LIMIT;
        let options: ReadAnnotationOptions = format!("maxreads{safe_max}").parse().unwrap();
        let previous = (safe_max - 1) * 64516;
        let mut sums = MapqSums {
            n_reads: safe_max - 1,
            sum_sq: previous,
            alt_sum_sq: previous,
            alleles_read: true,
            ..MapqSums::default()
        };
        assert!(!sums.is_full(options.max_reads()));
        sums.add(254);
        sums.add_allele(254, true);
        assert!(sums.is_full(options.max_reads()));
        assert_eq!(sums.n_reads, safe_max);
        assert_eq!(sums.sum_sq, safe_max * 64516);
        assert_eq!(sums.alt_sum_sq, sums.sum_sq);
        assert_eq!(sums.ref_sum_sq, 0);
        assert!(u64::MAX - sums.sum_sq < 64516);
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
        assert!(!sums.is_full(LEGACY_MAX_READS));
        assert!(MapqSums { n_reads: LEGACY_MAX_READS, ..sums }.is_full(LEGACY_MAX_READS));
    }

    /// The sums of `n_reads` reads of MAPQ 60.
    fn of(n_reads: u64) -> MapqSums {
        MapqSums { n_reads, sum_sq: n_reads * 3600, alleles_read: true, ..MapqSums::default() }
    }

    /// The calls without reads, and at the cap, are counted per contig: e.g. a contig the reads
    /// lack.
    #[test]
    fn calls_without_reads_are_counted_per_contig() {
        let (full, none, some) = (of(LEGACY_MAX_READS), of(0), of(5));
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
            at_cap.starts_with(&format!("1 records reached the cap of {LEGACY_MAX_READS}")),
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
