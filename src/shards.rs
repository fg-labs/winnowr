//! Splitting a VCF's contigs into shards processed in parallel, and the pool of readers the
//! shards reuse.

use std::sync::{Mutex, PoisonError};

use crate::vcf::VcfSummary;

/// A shard of a contig's records: those whose 0-based POS is in `[start, end)`, or from `start`
/// to the contig's end if `end` is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContigShard<'a> {
    /// The contig's index in the contigs the shards were planned for.
    pub index: usize,
    /// The contig.
    pub contig: &'a str,
    /// The 0-based start.
    pub start: u64,
    /// The 0-based, exclusive end (`None`: the contig's end).
    pub end: Option<u64>,
}

/// Splits each of `contigs` into shards of `span` bp by its length in the VCF header (a contig
/// without one is a single shard), in contig order. A contig's last shard is open-ended, so a
/// record past the declared length is still in a shard.
pub fn contig_shards<'a, S: AsRef<str>>(
    summary: &VcfSummary,
    contigs: &'a [S],
    span: u64,
) -> Vec<ContigShard<'a>> {
    let mut shards = Vec::new();
    for (index, contig) in contigs.iter().enumerate() {
        let contig = contig.as_ref();
        let length = summary.contig_lengths.get(contig).copied().flatten().unwrap_or(0);
        let mut start = 0;
        while start + span < length {
            shards.push(ContigShard { index, contig, start, end: Some(start + span) });
            start += span;
        }
        shards.push(ContigShard { index, contig, start, end: None });
    }
    shards
}

/// Reader sets shared by the shards processed in parallel: a shard takes a set and puts it back
/// when done, so at most one set per worker thread is ever opened. A panicking shard does not
/// poison the pool.
pub struct ReaderPool<T>(Mutex<Vec<T>>);

impl<T> ReaderPool<T> {
    /// An empty pool.
    pub fn new() -> Self {
        ReaderPool(Mutex::new(Vec::new()))
    }

    /// Takes the set last put back, if any.
    pub fn take(&self) -> Option<T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).pop()
    }

    /// Takes a set that `prefer` accepts (e.g. one already holding a contig's sequence), else the
    /// set last put back, if any.
    pub fn take_preferring(&self, prefer: impl Fn(&T) -> bool) -> Option<T> {
        let mut pool = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        match pool.iter().position(prefer) {
            Some(index) => Some(pool.swap_remove(index)),
            None => pool.pop(),
        }
    }

    /// Puts a set back for other shards to take.
    pub fn put(&self, readers: T) {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).push(readers);
    }
}

#[cfg(test)]
mod tests {
    use indexmap::IndexMap;

    use super::*;

    #[test]
    fn contigs_are_split_with_an_open_ended_last_shard() {
        let summary = VcfSummary {
            samples: vec![],
            contig_lengths: IndexMap::from([
                ("chr1".to_string(), Some(25)),
                ("chr2".to_string(), Some(20)),
                ("chr3".to_string(), None),
            ]),
        };
        let shards = contig_shards(&summary, &["chr1", "chr2", "chr3", "chrUn"], 10);
        let got: Vec<(usize, &str, u64, Option<u64>)> =
            shards.iter().map(|s| (s.index, s.contig, s.start, s.end)).collect();
        assert_eq!(
            got,
            [
                (0, "chr1", 0, Some(10)),
                (0, "chr1", 10, Some(20)),
                (0, "chr1", 20, None),
                (1, "chr2", 0, Some(10)),
                (1, "chr2", 10, None),
                (2, "chr3", 0, None),
                (3, "chrUn", 0, None),
            ]
        );
    }

    #[test]
    fn the_pool_prefers_a_matching_set_then_the_last_put_back() {
        let pool = ReaderPool::new();
        assert_eq!(pool.take(), None::<&str>);
        for readers in ["chr1", "chr2", "chr3"] {
            pool.put(readers);
        }
        assert_eq!(pool.take_preferring(|r| *r == "chr1"), Some("chr1"));
        assert_eq!(pool.take_preferring(|r| *r == "chrX"), Some("chr2"));
        assert_eq!(pool.take(), Some("chr3"));
        assert_eq!(pool.take(), None);
    }
}
