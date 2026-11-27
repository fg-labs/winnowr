//! Helpers for building test inputs (compiled only for tests).

use std::io::Write;
use std::path::{Path, PathBuf};

use rust_htslib::bcf::index::{self, Type};
use rust_htslib::bgzf;

/// Owns the subprocesses of a test that reads or writes a FIFO, which can block forever (e.g. a
/// writer whose reader never opens it). All are waited for under one deadline, and killed and
/// reaped on a timeout or when this is dropped (e.g. as a failed assertion unwinds), so a missed
/// read fails the test instead of hanging it.
#[derive(Default)]
pub struct TestProcesses {
    children: Vec<std::process::Child>,
    /// How many children run a test of this binary, by the log they write (see
    /// [`TestProcesses::spawn_test`]): counted per log, as each log holds only its own children's
    /// output, so this can be reused with the same or another log.
    tests: std::collections::HashMap<PathBuf, usize>,
}

impl TestProcesses {
    /// Starts a child, retaining ownership even if a later spawn or assertion fails.
    pub fn spawn(&mut self, command: &mut std::process::Command) {
        self.children.push(command.spawn().unwrap());
    }

    /// Runs the test `name` (its full path, e.g. `io::tests::child`) of this test binary in a
    /// child, with `envs` set and its output appended to `log`. The test must return at once
    /// unless its environment asks it to do something, as it also runs in the ordinary suite.
    pub fn spawn_test<K, V, I>(&mut self, name: &str, envs: I, log: &Path)
    where
        K: AsRef<std::ffi::OsStr>,
        V: AsRef<std::ffi::OsStr>,
        I: IntoIterator<Item = (K, V)>,
    {
        *self.tests.entry(log.to_path_buf()).or_default() += 1;
        let log = std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap();
        self.spawn(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture", "--test-threads=1"])
                .envs(envs)
                .stdout(log.try_clone().unwrap())
                .stderr(log),
        );
    }

    /// Waits for every child under one deadline. A timeout kills and reaps all children.
    ///
    /// # Errors
    ///
    /// Returns an error if a child cannot be waited for or the deadline passes.
    pub fn wait(
        &mut self,
        timeout: std::time::Duration,
    ) -> anyhow::Result<Vec<std::process::ExitStatus>> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let statuses = self
                .children
                .iter_mut()
                .map(std::process::Child::try_wait)
                .collect::<std::io::Result<Option<Vec<_>>>>()?;
            if let Some(statuses) = statuses {
                return Ok(statuses);
            }
            if std::time::Instant::now() >= deadline {
                self.stop();
                anyhow::bail!(
                    "test subprocesses exceeded {timeout:?}; all children killed and reaped"
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// [`TestProcesses::wait`]s, then asserts every child succeeded and every test child actually
    /// ran its test and its body, by the logs each was given: a misspelled name runs none, and
    /// exits 0. Children that run at once need logs of their own, as their output interleaves.
    pub fn wait_for_success(&mut self, timeout: std::time::Duration) {
        let statuses = self.wait(timeout);
        let output: String =
            self.tests.keys().map(|log| std::fs::read_to_string(log).unwrap_or_default()).collect();
        let statuses = statuses.unwrap_or_else(|error| panic!("{error:#}\n{output}"));
        assert!(statuses.iter().all(std::process::ExitStatus::success), "{statuses:?}\n{output}");
        for (log, &spawned) in &self.tests {
            let output = std::fs::read_to_string(log).unwrap_or_default();
            let ran = output.matches("test result: ok. 1 passed").count();
            assert_eq!(ran, spawned, "not every test child ran its test:\n{output}");
            // A child whose guard variable is missing returns early and still passes. (libtest
            // may print the marker on its `test NAME ...` line.)
            let bodies = output.matches(CHILD_BODY_RAN).count();
            assert_eq!(bodies, spawned, "not every test child ran its body:\n{output}");
        }
    }

    fn stop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        for child in &mut self.children {
            let _ = child.wait();
        }
    }
}

impl Drop for TestProcesses {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(unix)]
#[test]
fn fifo_deadline_kills_and_reaps_both_blocked_processes() {
    let dir = tempfile::tempdir().unwrap();
    let reader = dir.path().join("unfed.fifo");
    let writer = dir.path().join("unread.fifo");
    for fifo in [&reader, &writer] {
        assert!(std::process::Command::new("mkfifo").arg(fifo).status().unwrap().success());
    }
    let mut processes = TestProcesses::default();
    processes.spawn(std::process::Command::new("sh").args(["-c", ": < \"$1\"", "r"]).arg(reader));
    processes
        .spawn(std::process::Command::new("sh").args(["-c", "printf x > \"$1\"", "w"]).arg(writer));
    let error = processes.wait(std::time::Duration::from_millis(100)).unwrap_err();
    assert!(error.to_string().contains("killed and reaped"), "{error}");
    for child in &mut processes.children {
        assert!(!child.try_wait().unwrap().expect("child must have been reaped").success());
    }
}

#[cfg(unix)]
#[test]
fn fifo_deadline_reaps_a_writer_when_the_reader_returns_early() {
    let dir = tempfile::tempdir().unwrap();
    let fifo = dir.path().join("unread.fifo");
    assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
    let mut processes = TestProcesses::default();
    processes.spawn(std::process::Command::new("sh").args(["-c", "exit 0"]));
    // The early reader gets a generous deadline of its own, so that only the blocked writer is
    // raced against the short one below, however slow the runner is to start a shell.
    processes.wait(std::time::Duration::from_secs(10)).unwrap();
    processes
        .spawn(std::process::Command::new("sh").args(["-c", "printf x > \"$1\"", "w"]).arg(fifo));
    assert!(processes.wait(std::time::Duration::from_millis(100)).is_err());
    assert!(processes.children[0].try_wait().unwrap().unwrap().success());
    assert!(!processes.children[1].try_wait().unwrap().unwrap().success());
}

/// Printed by a test child once its body has run: passing alone does not show it, as a child run
/// without its guard variable returns early and passes (see [`TestProcesses::wait_for_success`]).
pub const CHILD_BODY_RAN: &str = "winnowr test child body ran";

/// Marks, at its end, that a test child's body ran (see [`CHILD_BODY_RAN`]).
pub fn child_body_ran() {
    println!("{CHILD_BODY_RAN}");
}

/// Child of [`a_test_child_that_runs_no_test_fails`] and
/// [`a_test_child_that_runs_its_test_succeeds`]: does nothing but mark that its body ran.
#[test]
fn test_child() {
    child_body_ran();
}

/// Child of [`a_test_child_whose_body_did_not_run_fails`]: passes without marking its body, as a
/// child run without its guard variable does.
#[test]
fn test_child_without_body() {}

#[test]
#[should_panic(expected = "not every test child ran its body")]
fn a_test_child_whose_body_did_not_run_fails() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("child.log");
    let mut processes = TestProcesses::default();
    processes.spawn_test(
        "testing::test_child_without_body",
        std::iter::empty::<(&str, &str)>(),
        &log,
    );
    processes.wait_for_success(std::time::Duration::from_secs(10));
}

#[test]
fn a_test_child_that_runs_its_test_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("child.log");
    let mut processes = TestProcesses::default();
    processes.spawn_test("testing::test_child", std::iter::empty::<(&str, &str)>(), &log);
    processes.wait_for_success(std::time::Duration::from_secs(10));
}

#[test]
fn test_processes_can_be_reused_with_the_same_or_another_log() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.log");
    let second = dir.path().join("second.log");
    let mut processes = TestProcesses::default();
    for log in [&first, &second, &first] {
        processes.spawn_test("testing::test_child", std::iter::empty::<(&str, &str)>(), log);
        processes.wait_for_success(std::time::Duration::from_secs(10));
    }
}

#[test]
#[should_panic(expected = "not every test child ran its test")]
fn a_test_child_that_runs_no_test_fails() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("child.log");
    let mut processes = TestProcesses::default();
    processes.spawn_test("testing::no_such_test", std::iter::empty::<(&str, &str)>(), &log);
    processes.wait_for_success(std::time::Duration::from_secs(10));
}

/// Writes `text` gzipped (one gzip member, not BGZF) to `path`.
pub fn write_gzip(path: &Path, text: &[u8]) {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(text).unwrap();
    std::fs::write(path, encoder.finish().unwrap()).unwrap();
}

/// Writes `text` as BGZF (blocked gzip, as `bgzip` writes it) to `path`.
pub fn write_bgzf(path: &Path, text: &[u8]) {
    let mut writer = bgzf::Writer::from_path(path).unwrap();
    writer.write_all(text).unwrap();
}

/// True if the file at `path` starts with a BGZF block: a gzip member whose complete extra field
/// has a complete `BC` subfield (wherever among its subfields), which plain gzip lacks.
pub fn is_bgzf(path: &Path) -> bool {
    is_bgzf_bytes(&std::fs::read(path).unwrap())
}

/// [`is_bgzf`] of the bytes of a file.
fn is_bgzf_bytes(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || bytes[..3] != [0x1f, 0x8b, 8] || bytes[3] & 4 == 0 {
        return false;
    }
    let length = usize::from(u16::from_le_bytes([bytes[10], bytes[11]]));
    bytes
        .get(12..12 + length)
        .is_some_and(|extra| crate::io::has_bgzf_subfield(extra).unwrap_or(false))
}

#[test]
fn is_bgzf_walks_the_extra_subfields_and_rejects_incomplete_ones() {
    let bc = [b'B', b'C', 2, 0, 0x1b, 0];
    let header = |extra: &[u8], xlen: u16| {
        let mut bytes = vec![0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0xff];
        bytes.extend_from_slice(&xlen.to_le_bytes());
        bytes.extend_from_slice(extra);
        bytes
    };
    assert!(is_bgzf_bytes(&header(&bc, 6)));
    let xy_then_bc = [&[b'X', b'Y', 0, 0][..], &bc].concat();
    assert!(is_bgzf_bytes(&header(&xy_then_bc, 10)));
    // BC with no subfield length or payload, or a field shorter than its XLEN.
    assert!(!is_bgzf_bytes(&header(b"BC", 6)));
    assert!(!is_bgzf_bytes(&header(&bc[..4], 6)));
    assert!(!is_bgzf_bytes(&header(&[b'B', b'C', 4, 0, 0, 0], 6)));
    // Plain gzip (no FEXTRA).
    let mut plain = header(&[], 0);
    plain[3] = 0;
    assert!(!is_bgzf_bytes(&plain));
}

/// The INFO and FORMAT declarations of a HaplotypeCaller VCF, without the raw annotations.
pub const CALL_FIELDS: &str = "\
##FILTER=<ID=PASS,Description=\"All filters passed\">
##FILTER=<ID=LowQual,Description=\"Low quality\">
##FILTER=<ID=RefCall,Description=\"Reference call\">
##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">
##INFO=<ID=MQRankSum,Number=1,Type=Float,Description=\"MQ rank sum\">
##INFO=<ID=ReadPosRankSum,Number=1,Type=Float,Description=\"Read position rank sum\">
##INFO=<ID=BaseQRankSum,Number=1,Type=Float,Description=\"Base quality rank sum\">
##INFO=<ID=DRAGstrInfo,Number=2,Type=Integer,Description=\"Period and repeats\">
##INFO=<ID=ExcessHet,Number=1,Type=Float,Description=\"Excess het\">
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality\">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allele depths\">
##FORMAT=<ID=SB,Number=4,Type=Integer,Description=\"Strand bias\">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred likelihoods\">
";

/// The INFO declarations of the raw annotations, as in a reblocked gVCF.
pub const RAW_FIELDS: &str = "\
##INFO=<ID=RAW_MQandDP,Number=2,Type=Integer,Description=\"Raw MQ and DP\">
##INFO=<ID=QUALapprox,Number=1,Type=Integer,Description=\"Approximate QUAL\">
##INFO=<ID=VarDP,Number=1,Type=Integer,Description=\"Variant depth\">
##INFO=<ID=AS_RAW_MQ,Number=1,Type=String,Description=\"Allele-specific raw MQ\">
";

/// The INFO and FORMAT declarations that a VCF from HaplotypeCaller's default (non-gVCF) mode, or
/// from GenotypeGVCFs, adds to [`CALL_FIELDS`], and that the VCF feature source reads.
pub const DEFAULT_MODE_FIELDS: &str = "\
##INFO=<ID=MQ,Number=1,Type=Float,Description=\"RMS mapping quality\">
##INFO=<ID=SOR,Number=1,Type=Float,Description=\"Strand odds ratio\">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read depth\">
";

/// A complete VCF header with contigs `chr1` and `chr2`, the given field declarations, and the
/// given samples.
pub fn vcf_header(fields: &[&str], samples: &[&str]) -> String {
    let mut header = String::from(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=100000>\n##contig=<ID=chr2,length=100000>\n",
    );
    for declarations in fields {
        header.push_str(declarations);
    }
    header.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO");
    if !samples.is_empty() {
        header.push_str("\tFORMAT");
        for sample in samples {
            header.push('\t');
            header.push_str(sample);
        }
    }
    header.push('\n');
    header
}

/// Writes a bgzipped, tabix-indexed VCF of `header` and `records` (one tab-separated line each,
/// sorted) to `dir/name`.
pub fn write_indexed_vcf(dir: &Path, name: &str, header: &str, records: &[&str]) -> PathBuf {
    let path = dir.join(name);
    let mut writer = bgzf::Writer::from_path(&path).unwrap();
    writer.write_all(header.as_bytes()).unwrap();
    for record in records {
        writer.write_all(record.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
    }
    drop(writer);
    index::build(&path, None, 1, Type::Tbx).unwrap();
    path
}

/// Writes `dir/ref.fa`, with 4 bases per line, and its `.fai` index.
pub fn write_fasta(dir: &Path, sequences: &[(&str, &str)]) -> PathBuf {
    let path = dir.join("ref.fa");
    let (mut text, mut fai) = (String::new(), String::new());
    for (name, sequence) in sequences {
        text.push_str(&format!(">{name} description\n"));
        let offset = text.len();
        for chunk in sequence.as_bytes().chunks(4) {
            text.push_str(std::str::from_utf8(chunk).unwrap());
            text.push('\n');
        }
        fai.push_str(&format!("{name}\t{}\t{offset}\t4\t5\n", sequence.len()));
    }
    std::fs::write(&path, text).unwrap();
    std::fs::write(dir.join("ref.fa.fai"), fai).unwrap();
    path
}

/// A read to write to a test BAM.
pub struct TestRead<'a> {
    /// The read name.
    pub name: &'a str,
    /// The 0-based alignment start.
    pub pos: i64,
    /// The CIGAR, e.g. `10M2D10M`.
    pub cigar: &'a str,
    /// The SAM flag.
    pub flag: u16,
    /// The mapping quality.
    pub mapq: u8,
    /// The base quality of every base, unless `quals` gives them.
    pub qual: u8,
    /// The base quality of each base (`None`: `qual` for every base).
    pub quals: Option<&'a [u8]>,
    /// The bases (`None`: all `A`; `*`: no SEQ, and no qualities).
    pub bases: Option<&'a str>,
}

/// The inputs of a read pass written to a directory: the calls (`in.vcf.gz`, of sample `S1`), the
/// reads (`reads.bam`) and the reference (`ref.fa`).
pub struct ReadFiles {
    /// The indexed VCF of the calls.
    pub vcf: PathBuf,
    /// The indexed BAM of the reads.
    pub bam: PathBuf,
    /// The indexed reference FASTA.
    pub fasta: PathBuf,
}

impl ReadFiles {
    /// Writes the `records` (VCF lines), the `reads` (on the first contig) and the reference
    /// `sequences` to `dir`. The VCF's header declares the reference's contigs.
    pub fn write(
        dir: &Path,
        sequences: &[(&str, &str)],
        records: &[&str],
        reads: &[TestRead<'_>],
    ) -> Self {
        let contigs: String = sequences
            .iter()
            .map(|(name, sequence)| format!("##contig=<ID={name},length={}>\n", sequence.len()))
            .collect();
        let header = vcf_header(&[CALL_FIELDS], &["S1"]).replacen(
            "##contig=<ID=chr1,length=100000>\n##contig=<ID=chr2,length=100000>\n",
            &contigs,
            1,
        );
        Self {
            vcf: write_indexed_vcf(dir, "in.vcf.gz", &header, records),
            bam: write_bam_on_contig(
                dir,
                "reads.bam",
                sequences[0].0,
                sequences[0].1.len() as u64,
                reads,
                &[],
            ),
            fasta: write_fasta(dir, sequences),
        }
    }

    /// The inputs of a pass over them.
    pub fn inputs(&self) -> crate::read_pass::ReadInputs<'_> {
        crate::read_pass::ReadInputs::new(&self.vcf, &self.bam, &self.fasta)
    }
}

#[test]
fn review_read_files_reads_follow_the_first_reference_contig() {
    use rust_htslib::bam::Read as _;
    let dir = tempfile::TempDir::new().unwrap();
    let reads = [TestRead {
        name: "first-contig",
        pos: 10,
        cigar: "10M",
        flag: 0,
        mapq: 60,
        qual: 30,
        quals: None,
        bases: None,
    }];
    let files = ReadFiles::write(
        dir.path(),
        &[("chr2", &"A".repeat(40))],
        &["chr2\t11\t.\tA\tC\t50\tPASS\t.\tGT\t0/1"],
        &reads,
    );
    let mut reader = rust_htslib::bam::IndexedReader::from_path(&files.bam).unwrap();
    let tid = reader.header().tid(b"chr2").expect("reads must target the first supplied contig");
    reader.fetch((tid, 0, 40)).unwrap();
    let records: Vec<_> = reader.records().map(|record| record.unwrap()).collect();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].qname(), b"first-contig");
    assert_eq!(reader.header().target_len(tid), Some(40));
}

/// A read pass over a contig the default VCF header does not declare plans and measures it: the
/// fixture's VCF declares the reference's contigs.
#[test]
fn review_read_files_declare_their_contigs_in_the_vcf() {
    let dir = tempfile::TempDir::new().unwrap();
    let files = ReadFiles::write(dir.path(), &[("chrX", &"A".repeat(40))], &[], &[]);
    let shards = crate::read_pass::run_shards(&files.inputs(), &["chrX"], 0, |shard| {
        Ok(shard.chrom.to_string())
    })
    .unwrap();
    assert!(!shards.is_empty());
    assert!(shards.iter().all(|chrom| chrom == "chrX"));
}

/// Writes a coordinate-sorted, indexed BAM of reads on `chr1` (of `length` bases) to `dir/name`.
pub fn write_bam(dir: &Path, name: &str, length: u64, reads: &[TestRead<'_>]) -> PathBuf {
    write_bam_with_header(dir, name, length, reads, &[])
}

/// As [`write_bam`], with `header_lines` (e.g. `@PG` lines, tab-separated) after the `@SQ`.
pub fn write_bam_with_header(
    dir: &Path,
    name: &str,
    length: u64,
    reads: &[TestRead<'_>],
    header_lines: &[&str],
) -> PathBuf {
    write_bam_on_contig(dir, name, "chr1", length, reads, header_lines)
}

/// Writes reads on the named contig, preserving the chr1 defaults of the public fixture helpers.
fn write_bam_on_contig(
    dir: &Path,
    name: &str,
    contig: &str,
    length: u64,
    reads: &[TestRead<'_>],
    header_lines: &[&str],
) -> PathBuf {
    use rust_htslib::bam::{self, header::HeaderRecord, record::CigarString};
    let path = dir.join(name);
    let mut header = bam::Header::new();
    let mut sq = HeaderRecord::new(b"SQ");
    sq.push_tag(b"SN", contig).push_tag(b"LN", length);
    header.push_record(&sq);
    for line in header_lines {
        let mut fields = line.split('\t');
        let kind = fields.next().unwrap().trim_start_matches('@');
        let mut record = HeaderRecord::new(kind.as_bytes());
        for field in fields {
            let (tag, value) = field.split_once(':').unwrap();
            record.push_tag(tag.as_bytes(), value);
        }
        header.push_record(&record);
    }
    let mut writer = bam::Writer::from_path(&path, &header, bam::Format::Bam).unwrap();
    let mut sorted: Vec<&TestRead<'_>> = reads.iter().collect();
    sorted.sort_by_key(|read| read.pos);
    for read in sorted {
        let cigar = CigarString::try_from(read.cigar).unwrap();
        let query_len = cigar
            .iter()
            .filter(|op| matches!(op.char(), 'M' | 'I' | 'S' | '=' | 'X'))
            .map(|op| op.len() as usize)
            .sum::<usize>();
        let (bases, quals) = if read.bases == Some("*") {
            (Vec::new(), Vec::new())
        } else {
            let bases = read.bases.map_or_else(|| vec![b'A'; query_len], |b| b.as_bytes().to_vec());
            assert_eq!(bases.len(), query_len, "the bases of {} do not match its CIGAR", read.name);
            let quals = read.quals.map_or_else(|| vec![read.qual; query_len], <[u8]>::to_vec);
            let name = read.name;
            assert_eq!(quals.len(), query_len, "the qualities of {name} do not match its CIGAR");
            (bases, quals)
        };
        let mut record = bam::Record::new();
        record.set(read.name.as_bytes(), Some(&cigar), &bases, &quals);
        record.set_tid(0);
        record.set_pos(read.pos);
        record.set_mapq(read.mapq);
        record.set_flags(read.flag);
        record.set_mtid(-1);
        record.set_mpos(-1);
        writer.write(&record).unwrap();
    }
    drop(writer);
    bam::index::build(&path, None, bam::index::Type::Bai, 1).unwrap();
    path
}

/// One call of a [`call_table`]: chrom, pos, ref, alt, filter, variant type, genotype and
/// whether it overlaps a PASS call.
pub type CallRow<'a> = (&'a str, i64, &'a str, &'a str, &'a str, &'a str, &'a str, bool);

/// A feature table with only the key columns of calls (enough for labelling).
pub fn call_table(rows: &[CallRow<'_>]) -> crate::table::FeatureTable {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    let strings = |f: for<'r> fn(&'r CallRow<'_>) -> &'r str| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(rows.iter().map(f)))
    };
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("chrom", strings(|r| r.0)),
        ("pos", Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1)))),
        ("ref", strings(|r| r.2)),
        ("alt", strings(|r| r.3)),
        ("filter", strings(|r| r.4)),
        ("variant_type", strings(|r| r.5)),
        ("gt", strings(|r| r.6)),
        ("overlaps_pass", Arc::new(BooleanArray::from_iter(rows.iter().map(|r| Some(r.7))))),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| {
            let data_type: DataType = array.data_type().clone();
            Field::new(*name, data_type, true)
        })
        .collect();
    let arrays = columns.into_iter().map(|(_, array)| array).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let info = crate::table::FeatureTableInfo {
        sample: "S1".to_string(),
        contigs: vec!["chr1".to_string()],
        ..crate::table::FeatureTableInfo::default()
    };
    crate::table::FeatureTable::new(batch, info)
}

/// A het SNP-typed PASS record of the given keys with every numeric value missing (NaN), and no
/// raw annotations.
pub fn raw_record_with_missing_values(
    chrom: &str,
    pos: u64,
    reference: &str,
    alts: &[&str],
) -> crate::raw_records::RawRecord {
    let nan = f64::NAN;
    crate::raw_records::RawRecord {
        chrom: std::sync::Arc::from(chrom),
        pos,
        reference: reference.as_bytes().to_vec(),
        alts: alts.iter().map(|alt| alt.as_bytes().to_vec()).collect(),
        filter: crate::common::PASS.to_string(),
        variant_type: crate::common::VariantType::Snp,
        genotype: crate::alleles::Genotype::HET,
        qual: nan,
        dp: nan,
        mq_rank_sum: nan,
        read_pos_rank_sum: nan,
        base_q_rank_sum: nan,
        str_period: nan,
        str_repeats: nan,
        gq: nan,
        ad_ref: nan,
        ad_alt: nan,
        sb_ref_fwd: nan,
        sb_ref_rev: nan,
        sb_alt_fwd: nan,
        sb_alt_rev: nan,
        excess_het: nan,
        pl_ref: nan,
        pl_het: nan,
        pl_hom: nan,
        ad_alt_total: nan,
        raw: crate::gvcf::RawAnnotation::MISSING,
    }
}

/// The probability a [`separable_model`] gives the class its feature `x` names, at least.
pub const SEPARABLE_P: f64 = 0.9;

/// A three-class model of one feature, `x`, trained so that a call with `x` = k (0, 1 or 2) gets
/// class k with a probability of at least [`SEPARABLE_P`], with thresholds of 0.5 (t_drop and
/// t_flip of a genotype model in every mode; t_promote and t_flip of a promote model).
pub fn separable_model(
    task: crate::common::Task,
    variant_type: crate::common::VariantType,
) -> crate::model::WinnowrModel {
    separable_model_of(task, variant_type, "x")
}

/// A [`separable_model`] of the one feature `feature` rather than `x`; a pileup feature's model
/// has the default pileup settings.
pub fn separable_model_of(
    task: crate::common::Task,
    variant_type: crate::common::VariantType,
    feature: &str,
) -> crate::model::WinnowrModel {
    use std::num::NonZeroUsize;

    use winnowr_xgboost::{Booster, DMatrix};

    use crate::common::{NEVER, Task};
    use crate::model::{MODEL_FORMAT_VERSION, ModelInfo, TrainingParams, WinnowrModel};
    use crate::rescore::Mode;
    use crate::thresholds::{PromoteThresholds, Thresholds};

    let rows = 300;
    let values: Vec<f32> = (0..rows).map(|i| (i % 3) as f32).collect();
    let mut train = DMatrix::from_dense(&values, rows, 1, f32::NAN).unwrap();
    train.set_labels(&values).unwrap();
    train.set_feature_names(&[feature]).unwrap();
    let threads = NonZeroUsize::new(1).unwrap();
    let mut booster = Booster::new(&[&train], threads).unwrap();
    booster
        .set_params(&[("objective", "multi:softprob"), ("num_class", "3"), ("eta", "0.5")])
        .unwrap();
    for iteration in 0..20 {
        booster.update(iteration, &train).unwrap();
    }
    let (thresholds, promote_thresholds) = match task {
        Task::Genotype => {
            let t = Thresholds { t_drop: 0.5, t_flip: 0.5, gain: 0.0, t_regenotype: NEVER };
            (Mode::ALL.iter().map(|&mode| (mode, t)).collect(), None)
        }
        Task::Promote => (
            std::collections::BTreeMap::new(),
            Some(PromoteThresholds { t_promote: 0.5, t_flip: 0.5, gain: 0.0 }),
        ),
    };
    let info = ModelInfo {
        format_version: MODEL_FORMAT_VERSION,
        variant_type,
        task,
        features: vec![feature.to_string()],
        region_names: vec![],
        thresholds,
        promote_thresholds,
        training_samples: vec!["TRAIN".to_string()],
        contigs: vec!["chr1".to_string()],
        reference_contigs: None,
        params: TrainingParams::default(),
        stutter: None,
        regenotype: false,
        strlen_settings: None,
        pileup_settings: crate::pileup_features::PILEUP_FEATURES
            .contains(&feature)
            .then(crate::pileup::PileupOptions::default),
        feature_source: crate::feature_source::FeatureSource::Gvcf,
        alignment: None,
        region_digests: std::collections::BTreeMap::new(),
        policy_region_digests: std::collections::BTreeMap::new(),
        winnowr_version: env!("CARGO_PKG_VERSION").to_string(),
        xgboost_version: "test".to_string(),
    };
    info.validate().unwrap();
    WinnowrModel { booster, info }
}

/// Fits a stutter model with the default [`crate::stutter::StutterFit`] settings (see
/// [`crate::stutter::fit_stutter_model_with`]).
#[must_use]
pub fn fit_stutter_model(sites: &[crate::stutter::StutterSite]) -> crate::stutter::StutterModel {
    crate::stutter::fit_stutter_model_with(sites, crate::stutter::StutterFit::default())
}

/// Manually declared fixture settings, complete unless `tags_only` (source `tags`).
/// Complete BWA fixtures without an inference marker are explicitly certified as declarations;
/// an existing marker is preserved. Actual header/legacy-inference tests must parse raw headers
/// or JSON instead, or deliberately remove the explicit marker after constructing the fixture.
pub fn provenance(
    tags: &[(&str, &str)],
    tags_only: bool,
) -> crate::alignment_provenance::AlignmentProvenance {
    let source = tags_only.then_some(("source", "tags"));
    let mut provenance: crate::alignment_provenance::AlignmentProvenance =
        tags.iter().copied().chain(source).collect();
    if !provenance.0.contains_key("options-inference")
        && crate::alignment_provenance::incomplete_options_reason(&provenance).is_some()
    {
        provenance.0.insert("options-inference".to_string(), "explicit".to_string());
    }
    provenance
}

#[test]
fn declared_bwa_fixtures_are_explicit_without_certifying_incomplete_or_tags_only_inputs() {
    assert_eq!(
        provenance(&[("aligner", "bwa"), ("Y", "")], false).get("options-inference"),
        Some("explicit")
    );
    assert_eq!(
        provenance(&[("aligner", "bwa"), ("options-inference", "incomplete")], false)
            .get("options-inference"),
        Some("incomplete")
    );
    assert_eq!(provenance(&[("aligner", "bwa")], true).get("options-inference"), None);
}

/// Explicitly declared complete fixture settings for bwa-mem3 with `--alt-project alt_project`.
pub fn bwa_provenance(alt_project: &str) -> crate::alignment_provenance::AlignmentProvenance {
    provenance(&[("aligner", "bwa-mem3"), ("alt-project", alt_project)], false)
}

/// The directories of model files `paths` as given (their parents), for a
/// [`crate::model_set::ModelSet::given_dirs`] of models whose directories need no resolving.
pub fn given_dirs(paths: &[std::path::PathBuf]) -> Vec<std::path::PathBuf> {
    paths
        .iter()
        .map(|path| path.parent().map(std::path::Path::to_path_buf).unwrap_or_default())
        .collect()
}
