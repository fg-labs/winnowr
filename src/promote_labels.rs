//! Labels for non-PASS calls, from the truth VCF and its confident regions.
//!
//! hap.py, run with `--pass-only`, only evaluates PASS calls, so non-PASS calls are labelled by
//! exact matching against the truth VCF. A non-PASS call is a promotion *candidate* when it is of
//! a known variant type, its REF span lies inside the truth confident regions and does not
//! overlap any PASS call. Calls and truth records are left-aligned and trimmed (see
//! [`normalize_alleles`]) and matched on position, REF and the ALT list. A candidate is labelled:
//!
//! - [`PromoteLabel::Promote`]: matches a truth record with the same genotype,
//! - [`PromoteLabel::PromoteFlip`]: matches a truth record, and switching its genotype between
//!   `0/1` and `1/1` gives the truth genotype,
//! - [`PromoteLabel::Stay`]: anything else.
//!
//! A match only counts if the truth record is a hap.py FN of the same variant type (not already
//! matched by a PASS call), and only for the first candidate that matches it with a promote
//! label, so a promotion that matches is exactly one FN turned into a TP.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use rust_htslib::bcf::Read as _;

use crate::alleles::{Genotype, NormalizedAlleles, normalize_alleles, trim_shared_bases};
use crate::common::{PASS, VariantType};
use crate::fasta::IndexedFasta;
use crate::labels::{LabelKey, PromoteLabel};
use crate::regions::{RegionSet, ref_span};
use crate::table::FeatureTable;
use crate::vcf::{FieldReader, open_indexed};

/// A normalized `(chrom, pos, ref, comma-joined alts)`.
pub type TruthKey = (String, u64, String, String);

/// A truth record that a candidate can match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruthVariant {
    /// The record's original (un-normalized) position, as hap.py reports it.
    pub pos: u64,
    /// The truth genotype.
    pub genotype: Genotype,
}

/// The normalized match key of a record, left-aligned with `fasta` when given.
fn match_key<S: AsRef<[u8]>>(
    chrom: &str,
    pos: u64,
    alleles: &[S],
    fasta: Option<&mut IndexedFasta>,
) -> Result<TruthKey> {
    let NormalizedAlleles { pos, alleles } = match fasta {
        Some(fasta) => normalize_alleles(pos, alleles, |base_pos: u64| -> Result<u8> {
            fasta
                .fetch(chrom, base_pos - 1, base_pos)?
                .first()
                .copied()
                .with_context(|| format!("no reference base at {chrom}:{base_pos}"))
        })?,
        None => trim_shared_bases(pos, alleles),
    };
    let text = |allele: &[u8]| String::from_utf8_lossy(allele).into_owned();
    let alts: Vec<String> = alleles.iter().skip(1).map(|a| text(a)).collect();
    Ok((
        chrom.to_string(),
        pos,
        alleles.first().map(|r| text(r)).unwrap_or_default(),
        alts.join(","),
    ))
}

/// Reads the truth records at hap.py FN positions, keyed by their normalized alleles (the first
/// record wins on collisions).
///
/// # Arguments
///
/// * `truth_vcf` - the indexed truth VCF (single sample).
/// * `fn_positions` - the hap.py truth FNs; only records at these positions are read.
/// * `reference` - the reference FASTA for left-alignment (without it alleles are only trimmed).
///
/// # Errors
///
/// Returns an error if an FN contig is not in the truth VCF header, or a file cannot be read.
pub fn read_truth_variants(
    truth_vcf: &Path,
    fn_positions: &HashSet<LabelKey>,
    reference: Option<&Path>,
) -> Result<HashMap<TruthKey, TruthVariant>> {
    let mut by_contig: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for (chrom, pos, _) in fn_positions {
        by_contig.entry(chrom.as_str()).or_default().push(*pos);
    }
    let header_check = open_indexed(truth_vcf)?;
    let mut chunks: Vec<(&str, u32, &[u64])> = Vec::new();
    for (contig, positions) in &mut by_contig {
        let Ok(rid) = header_check.header().name2rid(contig.as_bytes()) else {
            bail!(
                "contig {contig} has hap.py FNs but is not in the header of truth VCF {}",
                truth_vcf.display()
            );
        };
        positions.sort_unstable();
        positions.dedup();
        chunks.extend(positions.chunks(TRUTH_FETCH_CHUNK).map(|chunk| (*contig, rid, chunk)));
    }
    drop(header_check);
    // Chunks are read in parallel, each with its own readers, then merged in order, so the first
    // record of a key still wins.
    let read: Vec<Vec<(TruthKey, TruthVariant)>> = chunks
        .par_iter()
        .map(|&(contig, rid, positions)| {
            read_truth_chunk(truth_vcf, contig, rid, positions, reference)
        })
        .collect::<Result<_>>()?;
    let mut truth = HashMap::new();
    for (key, variant) in read.into_iter().flatten() {
        truth.entry(key).or_insert(variant);
    }
    Ok(truth)
}

/// The number of FN positions each [`read_truth_variants`] task fetches.
const TRUTH_FETCH_CHUNK: usize = 1024;

/// The truth records at sorted FN `positions` on one contig, keyed by their normalized alleles, in
/// fetch order.
fn read_truth_chunk(
    truth_vcf: &Path,
    contig: &str,
    rid: u32,
    positions: &[u64],
    reference: Option<&Path>,
) -> Result<Vec<(TruthKey, TruthVariant)>> {
    let mut fasta = reference.map(IndexedFasta::from_path).transpose()?;
    let mut vcf = open_indexed(truth_vcf)?;
    let mut fields = FieldReader::new();
    let mut record = vcf.empty_record();
    let mut truth = Vec::new();
    // One indexed fetch per FN position: FNs are sparse (tens of thousands per sample).
    for &pos in positions {
        // POS 0 (a telomere) has no base to fetch at; its records are not read.
        if pos == 0 || vcf.fetch(rid, pos - 1, Some(pos - 1)).is_err() {
            continue;
        }
        while let Some(read) = vcf.read(&mut record) {
            read.with_context(|| format!("failed to read {}", truth_vcf.display()))?;
            if (record.pos() + 1) as u64 != pos || record.allele_count() < 2 {
                continue;
            }
            let alleles = record.alleles();
            let key = match_key(contig, pos, &alleles, fasta.as_mut())?;
            let genotype = fields.genotype(&record);
            truth.push((key, TruthVariant { pos, genotype }));
        }
    }
    Ok(truth)
}

/// The non-PASS calls that are promotion candidates: [`promote_eligible`] calls whose REF span
/// lies inside the truth confident regions.
///
/// # Errors
///
/// Returns an error if the table lacks the key columns or `overlaps_pass`.
pub fn promote_candidates(table: &FeatureTable, confident: &RegionSet) -> Result<Vec<bool>> {
    let eligible = promote_eligible(table)?;
    let (chroms, positions, refs) =
        (table.str_column("chrom")?, table.i64_column("pos")?, table.str_column("ref")?);
    Ok((0..table.num_rows())
        .map(|i| {
            let (start, end) = ref_span(positions[i], refs[i].len());
            eligible[i] && confident.contains(chroms[i], start, end)
        })
        .collect())
}

/// The calls a promote model scores: non-PASS calls that do not overlap a PASS call. Shared by
/// training (see [`promote_candidates`]) and scoring.
///
/// # Errors
///
/// Returns an error if the table lacks `filter` or `overlaps_pass`.
pub fn promote_eligible(table: &FeatureTable) -> Result<Vec<bool>> {
    let filters = table.str_column("filter")?;
    let overlaps_pass = table.bool_column("overlaps_pass")?;
    Ok(filters
        .iter()
        .zip(overlaps_pass)
        .map(|(filter, overlaps)| *filter != PASS && !overlaps)
        .collect())
}

/// Adds a `promote_label` column (NaN for non-candidates).
///
/// # Arguments
///
/// * `table` - the feature table.
/// * `truth` - the truth records (see [`read_truth_variants`]).
/// * `fn_positions` - the hap.py truth FNs.
/// * `confident` - the truth confident regions.
/// * `reference` - the reference FASTA used for `truth` (it must be the same).
///
/// # Errors
///
/// Returns an error if the table lacks columns or the reference cannot be read.
pub fn attach_promote_labels(
    table: FeatureTable,
    truth: &HashMap<TruthKey, TruthVariant>,
    fn_positions: &HashSet<LabelKey>,
    confident: &RegionSet,
    reference: Option<&Path>,
) -> Result<FeatureTable> {
    let candidates = promote_candidates(&table, confident)?;
    let mut fasta = reference.map(IndexedFasta::from_path).transpose()?;
    let (chroms, positions) = (table.str_column("chrom")?, table.i64_column("pos")?);
    let (refs, alts) = (table.str_column("ref")?, table.str_column("alt")?);
    let (gts, types) = (table.str_column("gt")?, table.str_column("variant_type")?);
    let mut used: HashSet<TruthKey> = HashSet::new();
    let mut labels = vec![f64::NAN; table.num_rows()];
    for i in (0..table.num_rows()).filter(|&i| candidates[i]) {
        let alleles: Vec<&str> = std::iter::once(refs[i]).chain(alts[i].split(',')).collect();
        let key = match_key(chroms[i], positions[i] as u64, &alleles, fasta.as_mut())?;
        let mut label = PromoteLabel::Stay;
        if let (Some(hit), Ok(variant_type)) = (truth.get(&key), types[i].parse::<VariantType>())
            && !used.contains(&key)
            && fn_positions.contains(&(chroms[i].to_string(), hit.pos, variant_type))
        {
            let genotype: Genotype = gts[i].parse().unwrap_or(Genotype::MISSING);
            // A missing genotype (`./.`, or one not diploid) matches nothing, not even another
            // missing one.
            if genotype.is_missing() || hit.genotype.is_missing() {
                // Stay.
            } else if hit.genotype == genotype {
                label = PromoteLabel::Promote;
            } else if genotype.flipped() == Some(hit.genotype) {
                label = PromoteLabel::PromoteFlip;
            }
        }
        if label != PromoteLabel::Stay {
            used.insert(key);
        }
        labels[i] = label.index() as f64;
    }
    table.with_f64_column("promote_label", labels)
}

/// The labelled promotion candidates of one variant type.
///
/// # Errors
///
/// Returns an error if the table lacks `variant_type` or `promote_label`.
pub fn promote_mask(table: &FeatureTable, variant_type: VariantType) -> Result<Vec<bool>> {
    let types = table.str_column("variant_type")?;
    let labels = table.f64_column("promote_label")?;
    Ok((0..table.num_rows())
        .map(|i| types[i] == variant_type.as_str() && !labels[i].is_nan())
        .collect())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::testing::{CALL_FIELDS, call_table, vcf_header, write_fasta, write_indexed_vcf};

    /// A chr1 sequence with `GCACACAT` at 1-based positions 80-87 (a `CA` repeat).
    fn sequence() -> String {
        let mut bases: Vec<u8> = b"ACGT".iter().copied().cycle().take(200).collect();
        bases[79..87].copy_from_slice(b"GCACACAT");
        String::from_utf8(bases).unwrap()
    }

    fn base(at: u64) -> String {
        (sequence().as_bytes()[at as usize - 1] as char).to_string()
    }

    /// The truth VCF: SNPs at 50 (het), 60 (hom) and 70, and the left-aligned `CA` deletion.
    fn write_truth(dir: &TempDir) -> PathBuf {
        let records = [
            format!("chr1\t50\t.\t{}\tG\t.\tPASS\t.\tGT\t0|1", base(50)),
            format!("chr1\t60\t.\t{}\tT\t.\tPASS\t.\tGT\t1/1", base(60)),
            format!("chr1\t70\t.\t{}\tA\t.\tPASS\t.\tGT\t0/1", base(70)),
            "chr1\t80\t.\tGCA\tG\t.\tPASS\t.\tGT\t0/1".to_string(),
            "chr1\t90\t.\tA\t.\t.\tPASS\t.\tGT\t0/0".to_string(),
        ];
        let lines: Vec<&str> = records.iter().map(String::as_str).collect();
        let header = vcf_header(&[CALL_FIELDS], &["HG002"]);
        write_indexed_vcf(dir.path(), "truth.vcf.gz", &header, &lines)
    }

    fn fns() -> HashSet<LabelKey> {
        [(50, VariantType::Snp), (60, VariantType::Snp), (80, VariantType::Indel)]
            .into_iter()
            .map(|(pos, variant_type)| ("chr1".to_string(), pos, variant_type))
            .collect()
    }

    fn confident(dir: &TempDir) -> RegionSet {
        let path = dir.path().join("confident.bed");
        std::fs::write(&path, "chr1\t0\t120\n").unwrap();
        RegionSet::from_bed("confident", &path).unwrap()
    }

    #[test]
    fn reads_truth_records_at_fn_positions() {
        let dir = TempDir::new().unwrap();
        let truth_vcf = write_truth(&dir);
        let reference = write_fasta(dir.path(), &[("chr1", &sequence())]);
        // A POS 0 FN (a telomere) is not read.
        let mut with_telomere = fns();
        with_telomere.insert(("chr1".to_string(), 0, VariantType::Snp));
        let truth = read_truth_variants(&truth_vcf, &with_telomere, Some(&reference)).unwrap();
        let mut keys: Vec<(u64, &str, &str, u64)> =
            truth.iter().map(|(k, v)| (k.1, k.2.as_str(), k.3.as_str(), v.pos)).collect();
        keys.sort_unstable();
        let (b50, b60) = (base(50), base(60));
        assert_eq!(
            keys,
            [(50, b50.as_str(), "G", 50), (60, b60.as_str(), "T", 60), (80, "GCA", "G", 80)]
        );
        let het = truth.values().find(|v| v.pos == 50).unwrap();
        assert_eq!(het.genotype, Genotype::HET);
    }

    #[test]
    fn fn_contigs_missing_from_the_truth_vcf_are_an_error() {
        let dir = TempDir::new().unwrap();
        let truth_vcf = write_truth(&dir);
        let fns: HashSet<LabelKey> = [("chr9".to_string(), 5, VariantType::Snp)].into();
        let message = read_truth_variants(&truth_vcf, &fns, None).unwrap_err().to_string();
        assert!(
            message.starts_with("contig chr9 has hap.py FNs but is not in the header"),
            "{message}"
        );
    }

    #[test]
    fn missing_genotypes_match_nothing() {
        let dir = TempDir::new().unwrap();
        let key = |pos: u64| ("chr1".to_string(), pos, "A".to_string(), "C".to_string());
        let variant = |pos: u64, genotype: Genotype| TruthVariant { pos, genotype };
        let truth: HashMap<TruthKey, TruthVariant> = [
            (key(10), variant(10, Genotype::MISSING)),
            (key(20), variant(20, Genotype::MISSING)),
            (key(30), variant(30, Genotype::HET)),
            (key(40), variant(40, Genotype::HET)),
        ]
        .into_iter()
        .collect();
        let fns: HashSet<LabelKey> =
            [10, 20, 30, 40].map(|pos| ("chr1".to_string(), pos, VariantType::Snp)).into();
        let table = call_table(&[
            // A missing call and a missing (e.g. haploid) truth genotype.
            ("chr1", 10, "A", "C", "LowQual", "snp", "./.", false),
            // A called genotype and a missing truth genotype.
            ("chr1", 20, "A", "C", "LowQual", "snp", "0/1", false),
            // A missing call and a called truth genotype.
            ("chr1", 30, "A", "C", "LowQual", "snp", "./.", false),
            ("chr1", 40, "A", "C", "LowQual", "snp", "0/1", false),
        ]);
        let table = attach_promote_labels(table, &truth, &fns, &confident(&dir), None).unwrap();
        assert_eq!(*table.f64_column("promote_label").unwrap(), [0.0, 0.0, 0.0, 1.0]);
    }

    #[rstest::rstest]
    #[case::left_aligned(true, 1.0)]
    #[case::trimmed_only(false, 0.0)]
    fn labels_promotion_candidates(#[case] left_align: bool, #[case] deletion_label: f64) {
        let dir = TempDir::new().unwrap();
        let truth_vcf = write_truth(&dir);
        let fasta = write_fasta(dir.path(), &[("chr1", &sequence())]);
        let reference = left_align.then_some(fasta.as_path());
        let (b50, b60, b70) = (base(50), base(60), base(70));
        let table = call_table(&[
            ("chr1", 50, &b50, "G", "LowQual", "snp", "0/1", false),
            // A second match of the same truth record stays.
            ("chr1", 50, &b50, "G", "LowQual", "snp", "0/1", false),
            ("chr1", 60, &b60, "T", "LowQual", "snp", "0/1", false),
            // A truth record that is not an FN.
            ("chr1", 70, &b70, "A", "LowQual", "snp", "0/1", false),
            // The deletion, right-shifted in the repeat.
            ("chr1", 84, "ACA", "A", "LowQual", "indel", "0/1", false),
            // Not candidates: PASS, overlapping a PASS call, outside the confident regions.
            ("chr1", 100, "A", "C", "PASS", "snp", "0/1", false),
            ("chr1", 101, "A", "C", "LowQual", "snp", "0/1", true),
            ("chr1", 150, "A", "C", "LowQual", "snp", "0/1", false),
        ]);
        let confident = confident(&dir);
        assert_eq!(
            promote_candidates(&table, &confident).unwrap(),
            [true, true, true, true, true, false, false, false]
        );
        let truth = read_truth_variants(&truth_vcf, &fns(), reference).unwrap();
        let table = attach_promote_labels(table, &truth, &fns(), &confident, reference).unwrap();
        let labels = table.f64_column("promote_label").unwrap();
        let shown: Vec<String> = labels.iter().map(|l| format!("{l}")).collect();
        let deletion = format!("{deletion_label}");
        assert_eq!(shown, ["1", "0", "2", "0", deletion.as_str(), "NaN", "NaN", "NaN"]);
        assert_eq!(
            promote_mask(&table, VariantType::Indel).unwrap(),
            [false, false, false, false, true, false, false, false]
        );
    }
}
