//! Alignment provenance: how the reads behind an evidence table, a feature table or a model were
//! aligned, as comparable tags (e.g. `aligner=bwa-mem3`, `command=mem`, `alt-project=primary`).
//!
//! The tags are read from the `@PG` records of the reads' header (see [`HeaderProvenance`]), or
//! given with `--alignment-tags` (see [`AlignmentTags`]). `strlen` and `pileup` write them in their
//! tables, `extract` records them in the feature table, `train` in the model, and `score` compares
//! the models' with the reads' (see [`compare_provenance`]).
//!
//! Known BWA-family and minimap2 command lines (`CL`) join argv without reversible quoting. Their
//! option settings are not inferred: reliable header facts are retained with persistent
//! incomplete-option metadata until the user supplies an explicit declaration and intended
//! settings. Unknown tools retain only `--name=value` hints, including explicit empty values;
//! bare and short options and positional arguments are omitted. Paths with either separator are
//! reduced to file names. Raw command lines are never recorded.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, bail};
use log::{info, warn};
use serde::{Deserialize, Serialize};

/// The tag of the aligner's name (`@PG` `PN` or command/ID fallback, lower-cased: not the profile's).
const ALIGNER: &str = "aligner";
/// The tag of the aligner's version (`VN`): recorded and reported, never compared.
const VERSION: &str = "version";
/// The tag recording where the provenance came from, never compared: [`FROM_HEADER`] (the reads'
/// header, perhaps with `--alignment-tags` over it) or [`FROM_TAGS`] (`--alignment-tags` alone).
/// A provenance that records none came from a header.
const SOURCE: &str = "source";
/// The [`SOURCE`] of a provenance read from the reads' header.
const FROM_HEADER: &str = "header";
/// The [`SOURCE`] of a provenance given by `--alignment-tags` alone, or of a conflicting-table
/// `--mixed-alignments` intersection, which records only some tags: it is compared on those only.
const FROM_TAGS: &str = "tags";
/// Whether option settings were asserted explicitly or cannot be inferred from a lossy or missing
/// CL.
const OPTIONS_INFERENCE: &str = "options-inference";
/// Unrecoverable argument boundaries, carried through tables and models.
const INCOMPLETE: &str = "incomplete";
/// The user's declaration that the supplied option tags describe the intended settings.
const EXPLICIT: &str = "explicit";
/// Internal bookkeeping: a successful merge included a complete source, even though another
/// source retained incomplete inference. The union then has exactly the complete source's settings:
/// any extra or different partial setting would have conflicted. Missing settings remain absent.
/// This is not a user declaration and never clears incomplete inference.
const MERGED_COMPLETE: &str = "winnowr-merged-complete";
/// The value recording a complete contributor in a merge.
const YES: &str = "yes";
/// Internal JSON set of tags whose `mixed` value is literal, not a header disagreement marker.
/// Unmarked legacy `mixed` values remain conservative markers.
const LITERAL_MIXED: &str = "winnowr-literal-mixed-tags";
/// Metadata checked separately from alignment-setting equality.
const NOT_COMPARED: [&str; 5] =
    [VERSION, SOURCE, OPTIONS_INFERENCE, MERGED_COMPLETE, LITERAL_MIXED];
/// The tag recording whether the header has ALT contigs (an `@SQ` with `AH`): `yes` or `no`.
const SQ_ALT: &str = "sq-alt";
/// The value of a tag on which a header's aligner records disagree.
pub const MIXED: &str = "mixed";
/// The prefix of the line recording a table's provenance, before its column header.
pub const LINE_PREFIX: &str = "##alignment=";

/// How the reads behind a table or model were aligned: comparable tags, by name. A tag with an
/// empty value is a bare option (e.g. `Y` for bwa's `-Y`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
#[serde(transparent)]
pub struct AlignmentProvenance(pub BTreeMap<String, String>);

impl<'de> Deserialize<'de> for AlignmentProvenance {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ProvenanceVisitor;

        impl<'de> serde::de::Visitor<'de> for ProvenanceVisitor {
            type Value = AlignmentProvenance;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object of strings with unique keys")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut tags = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, String>()? {
                    if tags.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate alignment provenance key {key:?}"
                        )));
                    }
                    tags.insert(key, value);
                }
                Ok(AlignmentProvenance(tags))
            }
        }

        deserializer.deserialize_map(ProvenanceVisitor)
    }
}

/// Why option settings cannot be established, even if the recorded header facts agree.
/// Unmarked legacy header-derived provenance is incomplete too (an absent source means header).
/// Tags-only declarations remain valid; explicit declarations resolve either form of uncertainty.
#[must_use]
pub fn incomplete_options_reason(provenance: &AlignmentProvenance) -> Option<&'static str> {
    let incomplete = match provenance.0.get(OPTIONS_INFERENCE) {
        Some(value) => value != EXPLICIT,
        None => {
            !provenance.is_tags_only()
                && (provenance.0.contains_key(ALIGNER)
                    || provenance.0.get(SOURCE).is_some_and(|source| source == FROM_HEADER))
        }
    };
    incomplete.then_some(
        "the aligner's command line is missing or does not preserve argument boundaries, so its \
         options cannot be inferred; pass --alignment-tags options-inference=explicit with the \
         intended option tags to declare them explicitly",
    )
}

/// The setting tags actually compared, excluding metadata. Partial sources constrain only the tags
/// they record; a merged complete contributor still establishes absence despite incomplete peers.
#[must_use]
pub fn comparable_tags<'a>(
    models: &'a AlignmentProvenance,
    reads: &'a AlignmentProvenance,
) -> BTreeSet<&'a str> {
    let recorded = |side: &AlignmentProvenance, tag: &str| {
        side.has_complete_settings() || side.0.contains_key(tag)
    };
    models
        .0
        .keys()
        .chain(reads.0.keys())
        .map(String::as_str)
        .filter(|tag| !NOT_COMPARED.contains(tag))
        .filter(|tag| recorded(models, tag) && recorded(reads, tag))
        .collect()
}

/// Never let a complete source or a mixed-training intersection erase an incomplete one.
fn retain_incomplete(tags: &mut BTreeMap<String, String>, sources: &[&AlignmentProvenance]) {
    if sources.iter().any(|source| incomplete_options_reason(source).is_some()) {
        tags.insert(OPTIONS_INFERENCE.to_string(), INCOMPLETE.to_string());
    }
}

impl AlignmentProvenance {
    /// The value of a tag.
    #[cfg(test)]
    #[must_use]
    pub fn get(&self, tag: &str) -> Option<&str> {
        self.0.get(tag).map(String::as_str)
    }

    /// True if the provenance records only some tags ([`SOURCE`] is [`FROM_TAGS`]): it was given
    /// by `--alignment-tags` alone, or is a conflicting-table `--mixed-alignments` intersection.
    fn is_tags_only(&self) -> bool {
        self.0.get(SOURCE).is_some_and(|source| source == FROM_TAGS)
    }

    /// Whether missing settings are known absent, independently of inference uncertainty. Tags-only
    /// declarations and mixed-training intersections never assert absence.
    fn has_complete_settings(&self) -> bool {
        !self.is_tags_only()
            && (incomplete_options_reason(self).is_none()
                || self.0.get(MERGED_COMPLETE).is_some_and(|value| value == YES))
    }

    /// The provenance as a JSON object of strings, on one line.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.0).expect("a map of strings is JSON")
    }

    /// The provenance of a JSON object of strings.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is not a JSON object of strings with unique keys.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).context("alignment provenance must be a JSON object of strings")
    }
}

impl<K: Into<String>, V: Into<String>> FromIterator<(K, V)> for AlignmentProvenance {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(tags: I) -> Self {
        AlignmentProvenance(tags.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
    }
}

impl fmt::Display for AlignmentProvenance {
    /// The tags as `KEY=VALUE`, separated by spaces (as `--alignment-tags` takes them).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tags: Vec<String> = self
            .0
            .iter()
            .filter(|(key, _)| ![MERGED_COMPLETE, LITERAL_MIXED].contains(&key.as_str()))
            .map(|(key, value)| format!("{key}={value}"))
            .collect();
        f.write_str(&tags.join(" "))
    }
}

/// A tag on which two provenances differ: its value in the models' (or the first's) and in the
/// reads' (or the other's), `None` where it is not set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceDifference {
    /// The tag.
    pub tag: String,
    /// Its value in the models' provenance.
    pub models: Option<String>,
    /// Its value in the reads' provenance.
    pub reads: Option<String>,
}

/// The compared setting tags (excluding metadata) on which two provenances differ, by
/// tag. A tag set in one only is a difference when the other has complete settings, including a
/// complete contributor retained through a merge. Otherwise only recorded tags constrain it.
/// Incompleteness must be checked separately with [`incomplete_options_reason`], even when the
/// settings agree: preserving complete constraints does not certify incomplete peers.
#[must_use]
pub fn compare_provenance(
    models: &AlignmentProvenance,
    reads: &AlignmentProvenance,
) -> Vec<ProvenanceDifference> {
    #[cfg(test)]
    tests::record_comparison();
    comparable_tags(models, reads)
        .into_iter()
        .filter_map(|tag| {
            let (ours, theirs) = (models.0.get(tag), reads.0.get(tag));
            (ours != theirs).then(|| ProvenanceDifference {
                tag: tag.to_string(),
                models: ours.cloned(),
                reads: theirs.cloned(),
            })
        })
        .collect()
}

/// Compared tags whose equal `mixed` markers cannot establish agreement. This check is separate
/// from generic merging/training: the marker deliberately does not retain the underlying values.
#[must_use]
pub fn indeterminate_tags(
    models: &AlignmentProvenance,
    reads: &AlignmentProvenance,
) -> BTreeSet<String> {
    models
        .0
        .iter()
        .filter(|(tag, value)| {
            !NOT_COMPARED.contains(&tag.as_str())
                && value.as_str() == MIXED
                && reads.0.get(*tag).is_some_and(|other| other == MIXED)
                && (!literal_mixed(models).contains(tag.as_str())
                    || !literal_mixed(reads).contains(tag.as_str()))
        })
        .map(|(tag, _)| tag.clone())
        .collect()
}

/// The tags explicitly recorded as literal `mixed`, not header disagreement markers.
fn literal_mixed(provenance: &AlignmentProvenance) -> BTreeSet<String> {
    provenance
        .0
        .get(LITERAL_MIXED)
        .and_then(|value| serde_json::from_str(value).ok())
        .unwrap_or_default()
}

/// Persists only literal markers still present in `tags`, in the existing string-valued format.
fn record_literals(tags: &mut BTreeMap<String, String>, literals: BTreeSet<String>) {
    let literals: BTreeSet<String> = literals
        .into_iter()
        .filter(|tag| tags.get(tag).is_some_and(|value| value == MIXED))
        .collect();
    if literals.is_empty() {
        tags.remove(LITERAL_MIXED);
    } else {
        tags.insert(
            LITERAL_MIXED.to_string(),
            serde_json::to_string(&literals).expect("strings serialize"),
        );
    }
}

/// A union/intersection keeps a literal only if no contributor supplies a genuine mixed marker.
fn retain_literals(tags: &mut BTreeMap<String, String>, sources: &[&AlignmentProvenance]) {
    let literals =
        tags.iter()
            .filter(|(tag, value)| {
                value.as_str() == MIXED
                    && sources.iter().any(|source| source.0.get(*tag).is_some_and(|v| v == MIXED))
                    && sources.iter().all(|source| {
                        source.0.get(*tag).is_none_or(|v| {
                            v == MIXED && literal_mixed(source).contains(tag.as_str())
                        })
                    })
            })
            .map(|(tag, _)| tag.clone())
            .collect();
    record_literals(tags, literals);
}

/// The differences as `tag: FIRST a, SECOND b`, separated by `; ` (`not set` where a tag is
/// not), naming the two sides `first` and `second`.
#[must_use]
pub fn describe_differences(
    differences: &[ProvenanceDifference],
    first: &str,
    second: &str,
) -> String {
    let value = |value: &Option<String>| match value.as_deref() {
        None => "not set".to_string(),
        Some("") => "set".to_string(),
        Some(value) => value.to_string(),
    };
    differences
        .iter()
        .map(|d| format!("{}: {first} {}, {second} {}", d.tag, value(&d.models), value(&d.reads)))
        .collect::<Vec<_>>()
        .join("; ")
}

/// The tags on which every one of `provenances` agrees (same value in all), and the other tags. A
/// provenance with incomplete options says nothing about an option it lacks, so it constrains only
/// the options it records; one with complete options records their absence too.
fn agreement(provenances: &[&AlignmentProvenance]) -> (BTreeMap<String, String>, BTreeSet<String>) {
    let tags: BTreeSet<&String> = provenances.iter().flat_map(|p| p.0.keys()).collect();
    let mut agreed = BTreeMap::new();
    let mut disagreed = BTreeSet::new();
    for tag in tags.into_iter().filter(|tag| tag.as_str() != LITERAL_MIXED) {
        let constraining = provenances.iter().filter(|p| {
            METADATA_TAGS.contains(&tag.as_str())
                || p.0.contains_key(tag)
                || incomplete_options_reason(p).is_none()
        });
        let mut values = constraining.map(|p| p.0.get(tag));
        let first = values.next().flatten();
        if first.is_some() && values.all(|value| value == first) {
            agreed.insert(tag.clone(), first.cloned().unwrap_or_default());
        } else {
            disagreed.insert(tag.clone());
        }
    }
    (agreed, disagreed)
}

/// Two provenances that differ (see [`conflicts`]): their indexes, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeConflict {
    /// The index of the first.
    pub first: usize,
    /// The index of the second.
    pub second: usize,
    /// How they differ (the first's values as `models`, the second's as `reads`).
    pub differences: Vec<ProvenanceDifference>,
}

/// The provenance of reads that every one of `provenances` describes: the union of their tags
/// (`None` if there are none). Every pair must agree (see [`compare_provenance`]): a tag two
/// record has the same value in both, and a complete header-read provenance lacks no setting
/// another source records. Partial sources constrain only their recorded tags. A complete
/// contributor's absence constraints survive further merges and serialization via internal metadata,
/// independently of incomplete option inference, which survives if any source retains it. The
/// version is the first's read from a header (else the first's), and the source is a header's if
/// any is.
///
/// # Errors
///
/// Returns the first pair (in order) that differ.
pub fn merge(
    provenances: &[&AlignmentProvenance],
) -> std::result::Result<Option<AlignmentProvenance>, MergeConflict> {
    if let Some(conflict) = conflict_pairs(provenances).next() {
        return Err(conflict);
    }
    if provenances.is_empty() {
        return Ok(None);
    }
    // Those read from a header first, so that the version is theirs.
    let mut ordered: Vec<&&AlignmentProvenance> = provenances.iter().collect();
    ordered.sort_by_key(|p| p.is_tags_only());
    let mut tags = BTreeMap::new();
    for provenance in &ordered {
        for (tag, value) in provenance
            .0
            .iter()
            .filter(|(tag, _)| ![SOURCE, MERGED_COMPLETE, LITERAL_MIXED].contains(&tag.as_str()))
        {
            tags.entry(tag.clone()).or_insert_with(|| value.clone());
        }
    }
    if ordered.iter().all(|p| p.is_tags_only()) {
        tags.insert(SOURCE.to_string(), FROM_TAGS.to_string());
    } else if ordered.iter().any(|p| !p.is_tags_only() && p.0.contains_key(SOURCE)) {
        tags.insert(SOURCE.to_string(), FROM_HEADER.to_string());
    }
    retain_incomplete(&mut tags, provenances);
    if tags.get(OPTIONS_INFERENCE).is_some_and(|value| value == INCOMPLETE)
        && provenances.iter().any(|provenance| provenance.has_complete_settings())
    {
        tags.insert(MERGED_COMPLETE.to_string(), YES.to_string());
    }
    retain_literals(&mut tags, provenances);
    Ok(Some(AlignmentProvenance(tags)))
}

/// Every pair of `provenances` that differ (see [`compare_provenance`]), in order: the pairs that
/// keep them from being [`merge`]d.
#[must_use]
pub fn conflicts(provenances: &[&AlignmentProvenance]) -> Vec<MergeConflict> {
    conflict_pairs(provenances).collect()
}

/// Conflicting pairs in input order, compared only as the iterator advances.
fn conflict_pairs<'a>(
    provenances: &'a [&'a AlignmentProvenance],
) -> impl Iterator<Item = MergeConflict> + 'a {
    provenances.iter().enumerate().flat_map(move |(first, a)| {
        provenances.iter().enumerate().skip(first + 1).filter_map(move |(second, b)| {
            let differences = compare_provenance(a, b);
            (!differences.is_empty()).then_some(MergeConflict { first, second, differences })
        })
    })
}

/// How the reads compare with the models' provenances (`models`), each comparison that differs:
/// `(None, differences)` for models that agree, compared as their [`merge`], else
/// `(Some(index), differences)` for each model that differs from the reads.
#[must_use]
pub fn compare_with_models(
    models: &[&AlignmentProvenance],
    reads: &AlignmentProvenance,
) -> Vec<(Option<usize>, Vec<ProvenanceDifference>)> {
    match merge(models) {
        Ok(None) => Vec::new(),
        Ok(Some(merged)) => {
            let differences = compare_provenance(&merged, reads);
            if differences.is_empty() { Vec::new() } else { vec![(None, differences)] }
        }
        Err(_) => models
            .iter()
            .enumerate()
            .map(|(index, model)| (Some(index), compare_provenance(model, reads)))
            .filter(|(_, differences)| !differences.is_empty())
            .collect(),
    }
}

/// The tags that `provenances` (which differ) share, for a `--mixed-alignments` model: a tag is
/// kept when every provenance that records it agrees on it and every one read from a header
/// records it. Generated `mixed` disagreement markers are not shared settings; a literal `mixed`
/// is kept only when every contributor recording that setting identifies it as literal. The result
/// records only some tags ([`SOURCE`] is [`FROM_TAGS`]), so it is compared on those only.
fn shared_tags(provenances: &[&AlignmentProvenance]) -> AlignmentProvenance {
    let tags: BTreeSet<&String> = provenances.iter().flat_map(|p| p.0.keys()).collect();
    let mut shared = BTreeMap::new();
    for tag in tags
        .into_iter()
        .filter(|tag| ![SOURCE, MERGED_COMPLETE, LITERAL_MIXED].contains(&tag.as_str()))
    {
        let mut values = provenances.iter().filter_map(|p| p.0.get(tag));
        let first = values.next().expect("a provenance records the tag");
        let agreed = values.all(|value| value == first);
        let headers_record =
            provenances.iter().filter(|p| !p.is_tags_only()).all(|p| p.0.contains_key(tag));
        let literal = first != MIXED
            || NOT_COMPARED.contains(&tag.as_str())
            || provenances
                .iter()
                .all(|p| !p.0.contains_key(tag) || literal_mixed(p).contains(tag.as_str()));
        if agreed && headers_record && literal {
            shared.insert(tag.clone(), first.clone());
        }
    }
    shared.insert(SOURCE.to_string(), FROM_TAGS.to_string());
    retain_incomplete(&mut shared, provenances);
    retain_literals(&mut shared, provenances);
    AlignmentProvenance(shared)
}

// ------------------------------------------------------------------------------------------------
// The aligner's command line

/// A known aligner whose CL serialization does not preserve argument boundaries.
#[derive(Debug, Clone, Copy)]
struct Profile {
    /// Program names recognized in PN, ID or the command's file name.
    names: &'static [&'static str],
}

/// BWA, mem2 and mem3 join argv with spaces without reversible quoting.
const BWA: Profile = Profile { names: &["bwa", "bwa-mem2", "bwa-mem3", "bwa-mem3-meth"] };

/// minimap2 joins argv with spaces without reversible quoting too (its `format.c`).
const MINIMAP2: Profile = Profile { names: &["minimap2"] };

/// The aligners whose option inference is known to be incomplete.
const PROFILES: &[Profile] = &[BWA, MINIMAP2];

/// Known converters cannot identify the program that aligned their input reads.
const NON_ALIGNERS: &[&str] = &["samtools", "fastq-to-bam"];

/// The options the generic reading of an aligner without a profile leaves out.
const FALLBACK_IGNORED: &[&str] =
    &["t", "threads", "@", "o", "output", "R", "v", "verbose", "tmp-dir"];

/// winnowr's own tags, not option settings: the header facts and metadata every aligner record has,
/// whether or not its options are known. An unprofiled tool's options of these names are left out,
/// so they are never read as winnowr's metadata (e.g. a wrapper's `--options-inference=explicit`).
const METADATA_TAGS: [&str; 7] =
    [ALIGNER, VERSION, SOURCE, SQ_ALT, OPTIONS_INFERENCE, MERGED_COMPLETE, LITERAL_MIXED];

/// The profile of an aligner, by name.
fn profile(aligner: &str) -> Option<&'static Profile> {
    PROFILES.iter().find(|profile| profile.names.contains(&aligner))
}

/// A value as recorded: a path reduced to its file name, regardless of the host's separators.
fn normalized_value(value: &str) -> String {
    value.rsplit(['/', '\\']).find(|part| !part.is_empty()).unwrap_or_default().to_string()
}

/// Records an explicit fallback value, accumulating repeats without confusing a comma inside a
/// value with a boundary between values.
fn record(
    tags: &mut BTreeMap<String, String>,
    repeated: &mut BTreeMap<String, Vec<String>>,
    name: &str,
    value: &str,
    ignored: &dyn Fn(&str) -> bool,
) {
    if name.is_empty() || name.contains(['/', '\\']) || ignored(name) {
        return;
    }
    let value = normalized_value(value);
    let values = repeated.entry(name.to_string()).or_default();
    values.push(value.clone());
    let encoded = if values.len() == 1 && serde_json::from_str::<Vec<String>>(&value).is_err() {
        value
    } else {
        serde_json::to_string(values).expect("strings serialize")
    };
    tags.insert(name.to_string(), encoded);
}

/// Unprofiled hints come only from explicit --name=value, including explicit empty values. Bare
/// and short options are omitted: their values are unknown, not empty. Separate tokens are never
/// consumed as values, since they may be positional reference/read filenames.
fn fallback_tags(command_line: &str) -> BTreeMap<String, String> {
    let ignored = |name: &str| {
        FALLBACK_IGNORED.contains(&name) || METADATA_TAGS.contains(&name) || name.starts_with("rg")
    };
    let mut tags = BTreeMap::new();
    let mut repeated = BTreeMap::new();
    for token in command_line.split_whitespace().skip(1) {
        if token == "--" {
            break;
        }
        let Some((name, value)) =
            token.strip_prefix("--").and_then(|option| option.split_once('='))
        else {
            continue;
        };
        record(&mut tags, &mut repeated, name, value, &ignored);
    }
    tags
}

// ------------------------------------------------------------------------------------------------
// The header

/// One `@PG` record of a header.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Program {
    /// `ID`.
    id: String,
    /// `PN`.
    name: Option<String>,
    /// `VN`.
    version: Option<String>,
    /// `CL`.
    command_line: Option<String>,
    /// `PP`: the `ID` of the program before it.
    previous: Option<String>,
}

impl Program {
    /// The program, lower-cased: `PN`, else a recognized aligner executable in `CL`, else `ID`
    /// without a merge (`-XXXXXXXX`) or `.N` suffix, else the first `CL` word's file name.
    fn program(&self) -> String {
        if let Some(name) = self.name.as_deref().filter(|name| !name.is_empty()) {
            return name.to_ascii_lowercase();
        }
        if let Some(command) = self.command().filter(|command| profile(command).is_some()) {
            return command;
        }
        let id = strip_id_suffix(&self.id);
        if !id.is_empty() {
            return id.to_ascii_lowercase();
        }
        self.command().unwrap_or_default()
    }

    /// The file name of the first word of `CL`, lower-cased.
    fn command(&self) -> Option<String> {
        let first = self.command_line.as_deref().and_then(|cl| cl.split_whitespace().next())?;
        Some(normalized_value(first).to_ascii_lowercase())
    }

    /// The program's profile: that of [`Program::program`], else that of its command (e.g. a
    /// record whose `PN` is not the aligner's, as some tools write).
    fn profile(&self) -> Option<&'static Profile> {
        profile(&self.program()).or_else(|| self.command().and_then(|command| profile(&command)))
    }

    /// Known conversion records remain in PP chains but cannot stand in as aligners.
    fn is_aligner_candidate(&self) -> bool {
        self.profile().is_some()
            || !(NON_ALIGNERS.contains(&self.program().as_str())
                || self.command().is_some_and(|command| NON_ALIGNERS.contains(&command.as_str())))
    }
}

/// An `@PG` `ID` without the suffix `samtools merge` gives a repeated id (`-` and eight
/// hexadecimal digits) or a `.N` suffix (e.g. `samtools.1`).
fn strip_id_suffix(id: &str) -> &str {
    if let Some((stem, suffix)) = id.rsplit_once('-')
        && suffix.len() == 8
        && suffix.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return stem;
    }
    if let Some((stem, suffix)) = id.rsplit_once('.')
        && !suffix.is_empty()
        && suffix.bytes().all(|b| b.is_ascii_digit())
    {
        return stem;
    }
    id
}

/// The ids of a program's ancestors through `PP` (`by_id`: the header's programs by id), stopping
/// at a cycle.
fn ancestors<'a>(program: &'a Program, by_id: &HashMap<&'a str, &'a Program>) -> BTreeSet<&'a str> {
    let mut seen = BTreeSet::new();
    let mut previous = program.previous.as_deref();
    while let Some(id) = previous {
        if !seen.insert(id) {
            break;
        }
        previous = by_id.get(id).and_then(|p| p.previous.as_deref());
    }
    seen
}

/// Selects the latest profiled aligner independently on every terminal PP chain, deduplicated in
/// header order. A shared ancestor may remain current on one branch but stale on another. A chain
/// without a profile contributes its earliest eligible non-converter, or first eligible record
/// when cyclic; terminal profiled cycles
/// contribute all their aligners unless superseded by a profiled record outside the cycle.
fn alignment_records(programs: &[Program]) -> Vec<&Program> {
    let indexes: HashMap<&str, usize> =
        programs.iter().enumerate().map(|(i, p)| (p.id.as_str(), i)).collect();
    let parents: Vec<Option<usize>> = programs
        .iter()
        .map(|program| program.previous.as_deref().and_then(|id| indexes.get(id)).copied())
        .collect();
    let has_children: BTreeSet<usize> = parents.iter().flatten().copied().collect();
    let leaves = (0..programs.len()).filter(|index| !has_children.contains(index));
    let mut covered = BTreeSet::new();
    let mut selected = BTreeSet::new();
    // Leaves cover rooted chains and cycles with descendants. Uncovered nodes then cover isolated
    // cycles, which have no leaf. Shared ancestors must still be visited for each separate leaf.
    for start in leaves.chain(0..programs.len()) {
        if covered.contains(&start) {
            continue;
        }
        let mut chain = Vec::new();
        let mut positions = BTreeMap::new();
        let mut current = Some(start);
        let mut cycle = None;
        while let Some(index) = current {
            if let Some(position) = positions.insert(index, chain.len()) {
                cycle = Some(position);
                break;
            }
            covered.insert(index);
            chain.push(index);
            current = parents[index];
        }
        if let Some(latest) = chain.iter().position(|&index| programs[index].profile().is_some()) {
            if let Some(cycle) = cycle.filter(|&cycle| latest >= cycle) {
                selected.extend(
                    chain[cycle..].iter().copied().filter(|&i| programs[i].profile().is_some()),
                );
            } else {
                selected.insert(chain[latest]);
            }
        } else {
            let mut candidates =
                chain.iter().filter(|&&index| programs[index].is_aligner_candidate());
            let root = if cycle.is_some() { candidates.min() } else { candidates.next_back() };
            if let Some(root) = root {
                selected.insert(*root);
            }
        }
    }
    selected.into_iter().map(|i| &programs[i]).collect()
}

/// The `@PG` records of a header's text, and whether an `@SQ` has an `AH` tag.
fn parse_header(text: &str) -> (Vec<Program>, bool) {
    let mut programs = Vec::new();
    let mut alt = false;
    for line in text.lines() {
        let mut fields = line.split('\t');
        match fields.next() {
            Some("@PG") => {
                let mut program = Program::default();
                for field in fields {
                    let Some((tag, value)) = field.split_once(':') else { continue };
                    let value = value.to_string();
                    match tag {
                        "ID" => program.id = value,
                        "PN" => program.name = Some(value),
                        "VN" => program.version = Some(value),
                        "CL" => program.command_line = Some(value),
                        "PP" => program.previous = Some(value),
                        _ => {}
                    }
                }
                programs.push(program);
            }
            Some("@SQ") => alt |= fields.any(|field| field.starts_with("AH:")),
            _ => {}
        }
    }
    (programs, alt)
}

/// What a header tells of how its reads were aligned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HeaderProvenance {
    /// The provenance (`None` without an eligible aligner record).
    pub provenance: Option<AlignmentProvenance>,
    /// Original `@PG` and `@SQ` lines, retained only in memory to bind preflight declarations to
    /// the header actually measured. Raw command lines are never serialized as provenance.
    pub header_records: Vec<String>,
    /// The compared tags on which the header's aligner records disagree (given [`MIXED`]).
    mixed: Vec<String>,
    /// The aligner read without a profile, if its options were.
    unprofiled: Option<String>,
    /// Whether a program with a command line follows the record read without a profile (it may
    /// then be a tool run before the aligner).
    unprofiled_followed: bool,
}

impl HeaderProvenance {
    /// The provenance of a SAM header's text.
    ///
    /// The aligner records are the `@PG` records of aligners with a [`Profile`] that are not an
    /// ancestor (through `PP`) of a newer aligner (a stale record carried through re-alignment),
    /// selected independently per terminal PP chain, including branches with shared ancestors.
    /// Chains without a profiled aligner contribute their earliest eligible non-converter, or
    /// first eligible record when cyclic, read without a profile. Shared records are included
    /// once. Records of a profiled (lossy) aligner, and records without a command line, retain
    /// header facts and incomplete-option metadata, not inferred options. Where facts or
    /// unprofiled tags disagree, a tag's value is [`MIXED`]; an option a record with incomplete
    /// options lacks is unknown there, not a disagreement.
    #[must_use]
    pub fn from_header_text(text: &str) -> Self {
        let header_records = text
            .lines()
            .filter(|line| matches!(line.split('\t').next(), Some("@PG" | "@SQ")))
            .map(str::to_string)
            .collect();
        let (programs, alt) = parse_header(text);
        if programs.is_empty() {
            return HeaderProvenance { header_records, ..Default::default() };
        }
        let by_id: HashMap<&str, &Program> =
            programs.iter().map(|program| (program.id.as_str(), program)).collect();
        let records = alignment_records(&programs);
        if records.is_empty() {
            return HeaderProvenance { header_records, ..Default::default() };
        }
        let mut unprofiled = None;
        let mut unprofiled_followed = false;
        let provenances: Vec<AlignmentProvenance> = records
            .iter()
            .map(|program| {
                let aligner = program.program();
                let mut tags = match (program.profile(), program.command_line.as_deref()) {
                    (Some(_), _) => {
                        BTreeMap::from([(OPTIONS_INFERENCE.to_string(), INCOMPLETE.to_string())])
                    }
                    (None, Some(cl)) if !cl.trim().is_empty() => {
                        unprofiled = Some(aligner.clone());
                        unprofiled_followed |= programs.iter().any(|other| {
                            other.command_line.is_some()
                                && ancestors(other, &by_id).contains(program.id.as_str())
                        });
                        let mut tags = fallback_tags(cl);
                        tags.insert(OPTIONS_INFERENCE.to_string(), INCOMPLETE.to_string());
                        tags
                    }
                    // No command line records no options: they are unknown, not absent.
                    (None, _) => {
                        BTreeMap::from([(OPTIONS_INFERENCE.to_string(), INCOMPLETE.to_string())])
                    }
                };
                tags.insert(ALIGNER.to_string(), aligner);
                if let Some(version) = &program.version {
                    tags.insert(VERSION.to_string(), version.clone());
                }
                tags.insert(SQ_ALT.to_string(), if alt { "yes" } else { "no" }.to_string());
                tags.insert(SOURCE.to_string(), FROM_HEADER.to_string());
                let literals = tags
                    .iter()
                    .filter(|(_, value)| value.as_str() == MIXED)
                    .map(|(tag, _)| tag.clone())
                    .collect();
                record_literals(&mut tags, literals);
                AlignmentProvenance(tags)
            })
            .collect();
        let refs: Vec<&AlignmentProvenance> = provenances.iter().collect();
        let (mut tags, disagreed) = agreement(&refs);
        for tag in &disagreed {
            tags.insert(tag.clone(), MIXED.to_string());
        }
        retain_incomplete(&mut tags, &refs);
        retain_literals(&mut tags, &refs);
        let mixed =
            disagreed.into_iter().filter(|tag| !NOT_COMPARED.contains(&tag.as_str())).collect();
        HeaderProvenance {
            provenance: Some(AlignmentProvenance(tags)),
            header_records,
            mixed,
            unprofiled,
            unprofiled_followed,
        }
    }

    /// Reads the header of a BAM, CRAM or SAM file (a CRAM's reference is not needed) and its
    /// provenance (see [`HeaderProvenance::from_header_text`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or its header read.
    pub fn read(path: &Path) -> Result<Self> {
        use rust_htslib::bam::Read as _;
        let reader = rust_htslib::bam::Reader::from_path(path)
            .with_context(|| format!("failed to read the header of {}", path.display()))?;
        let text = String::from_utf8_lossy(reader.header().as_bytes()).into_owned();
        Ok(HeaderProvenance::from_header_text(&text))
    }

    /// The warning about aligner records of `path`'s header that disagree, if they do.
    #[must_use]
    pub fn mixed_warning(&self, path: &Path) -> Option<String> {
        (!self.mixed.is_empty()).then(|| {
            format!(
                "the header of {} has aligner records that disagree on {}; they are recorded as \
                 {MIXED} (pass --alignment-tags to record them otherwise)",
                path.display(),
                self.mixed.join(", ")
            )
        })
    }

    /// The warning about an aligner of `path`'s header read without a profile, if one was.
    #[must_use]
    pub fn unprofiled_warning(&self, path: &Path) -> Option<String> {
        self.unprofiled.as_ref().map(|aligner| {
            let followed = if self.unprofiled_followed {
                format!(
                    "; other programs follow it, so {aligner} may be a tool run before the \
                     aligner"
                )
            } else {
                String::new()
            };
            format!(
                "the options of {aligner} (in the header of {}) are read without a profile{followed}; \
                 check them with --alignment-tags",
                path.display()
            )
        })
    }
}

// ------------------------------------------------------------------------------------------------
// --alignment-tags

/// The tags given with `--alignment-tags`: `none` (record nothing), or tags to set and remove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignmentTags {
    /// `none`: record no provenance, and read no source of it.
    None,
    /// The tags to set and to remove.
    Tags(TagEdits),
}

/// Tags to set (`KEY=VALUE`) and remove (`KEY!`) over a provenance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TagEdits {
    /// `KEY=VALUE`: the tags to set, by name.
    pub set: BTreeMap<String, String>,
    /// `KEY!`: the detected tags to remove.
    pub removed: BTreeSet<String>,
}

/// Where the reads' provenance came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceFrom {
    /// The reads' header (or their evidence tables).
    Header,
    /// `--alignment-tags` alone.
    Tags,
    /// The header, with `--alignment-tags` over it.
    HeaderAndTags,
}

impl ProvenanceFrom {
    /// `header`, `tags` or `header+tags`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ProvenanceFrom::Header => "header",
            ProvenanceFrom::Tags => "tags",
            ProvenanceFrom::HeaderAndTags => "header+tags",
        }
    }
}

impl AlignmentTags {
    /// The tags of `--alignment-tags` values (`None` when none is given).
    ///
    /// # Errors
    ///
    /// Returns an error if a value is not `KEY=VALUE` or `KEY!` (a key of letters, digits, `-`,
    /// `_` and `.`), a key is given twice (set or removed), the key is `source` or internal merge
    /// metadata (which winnowr sets), `options-inference` is anything other than `explicit`, or
    /// `none` is given with other values.
    pub fn parse<S: AsRef<str>>(values: &[S]) -> Result<Option<Self>> {
        if values.is_empty() {
            return Ok(None);
        }
        if values.iter().any(|value| value.as_ref() == "none") {
            if values.len() > 1 {
                bail!("--alignment-tags none cannot be given with other tags");
            }
            return Ok(Some(AlignmentTags::None));
        }
        let mut set = BTreeMap::new();
        let mut removed = BTreeSet::new();
        for value in values {
            let value = value.as_ref();
            let (key, tag_value) = match value.split_once('=') {
                Some((key, tag_value)) => (key, Some(tag_value)),
                None => match value.strip_suffix('!') {
                    Some(key) => (key, None),
                    None => bail!("--alignment-tags must be KEY=VALUE, KEY! or none, got: {value}"),
                },
            };
            let valid = |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.');
            if key.is_empty() || !key.chars().all(valid) {
                bail!("--alignment-tags keys are letters, digits, '-', '_' and '.', got: {value}");
            }
            if key == SOURCE {
                bail!(
                    "--alignment-tags cannot give {SOURCE}: winnowr records where the provenance \
                     came from itself"
                );
            }
            if [MERGED_COMPLETE, LITERAL_MIXED].contains(&key) {
                bail!("--alignment-tags cannot give {key}: internal provenance metadata");
            }
            if key == OPTIONS_INFERENCE && tag_value != Some(EXPLICIT) {
                bail!(
                    "--alignment-tags requires {OPTIONS_INFERENCE}={EXPLICIT} with the intended \
                     option tags to resolve incomplete option inference; it cannot be removed"
                );
            }
            if set.contains_key(key) || removed.contains(key) {
                bail!("--alignment-tags gives {key} more than once");
            }
            if let Some(tag_value) = tag_value {
                set.insert(key.to_string(), tag_value.to_string());
            } else {
                removed.insert(key.to_string());
            }
        }
        Ok(Some(AlignmentTags::Tags(TagEdits { set, removed })))
    }
}

/// `provenance` (of the source `name`) with `edits` applied (each replaced tag logged): the tags
/// set set, and those removed removed. Its source is kept.
fn overlay(
    mut provenance: AlignmentProvenance,
    edits: Option<&TagEdits>,
    name: &str,
) -> AlignmentProvenance {
    let Some(TagEdits { set, removed }) = edits else { return provenance };
    let mut literals = literal_mixed(&provenance);
    for (tag, value) in set {
        if value == MIXED {
            literals.insert(tag.clone());
        } else {
            literals.remove(tag);
        }
    }
    for tag in removed {
        literals.remove(tag);
    }
    // Preserve implicit legacy uncertainty before edits can change/remove its identifying tags.
    if incomplete_options_reason(&provenance).is_some() {
        provenance.0.insert(OPTIONS_INFERENCE.to_string(), INCOMPLETE.to_string());
    }
    for (key, value) in set {
        if let Some(old) = provenance.0.insert(key.clone(), value.clone())
            && old != *value
        {
            info!("--alignment-tags gives {key}={value} in place of {key}={old} ({name})");
        }
    }
    for key in removed {
        if provenance.0.remove(key).is_some() {
            info!("--alignment-tags removes {key} ({name})");
        }
    }
    if set.get(OPTIONS_INFERENCE).is_some_and(|value| value == EXPLICIT) {
        // The new declaration replaces the merge's bookkeeping, not just its uncertainty.
        provenance.0.remove(MERGED_COMPLETE);
    }
    record_literals(&mut provenance.0, literals);
    provenance
}

/// Why the reads' provenance is not known.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnknownReason {
    /// Neither the reads (or their evidence tables) nor `--alignment-tags` were given.
    #[default]
    NotGiven,
    /// The reads' header identifies no aligner, and their evidence tables record no provenance.
    NotRecorded,
    /// Tag edits supplied no comparable settings, and no source recorded provenance.
    TagsWithoutSettings,
    /// `--alignment-tags none` was given.
    TagsNone,
    /// The reads' header could not be read (a problem of its own).
    Unreadable,
}

impl UnknownReason {
    /// Why, and what to do.
    #[must_use]
    pub fn explanation(self) -> &'static str {
        match self {
            UnknownReason::NotGiven => {
                "neither --alignments nor --alignment-tags was given: pass either"
            }
            UnknownReason::NotRecorded => {
                "the reads' header identifies no aligner, and their evidence tables record no \
                 alignment provenance: pass --alignment-tags"
            }
            UnknownReason::TagsWithoutSettings => {
                "--alignment-tags supplies no alignment settings: give at least one compared \
                 setting, not only metadata or removals"
            }
            UnknownReason::TagsNone => {
                "--alignment-tags none was given: give the tags instead, or drop it to read them \
                 from the reads"
            }
            UnknownReason::Unreadable => "the reads' header could not be read",
        }
    }
}

/// What is known of how the reads were aligned (see [`resolve_reads`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadsAlignment {
    /// The provenance, if known.
    pub provenance: Option<AlignmentProvenance>,
    /// Where it came from (`None` if from nothing).
    pub from: Option<ProvenanceFrom>,
    /// Why it is not known, when it is not.
    pub unknown: UnknownReason,
    /// The reads' header, when it was read: what it tells, or why it could not be read.
    pub header: Option<std::result::Result<HeaderProvenance, String>>,
}

impl ReadsAlignment {
    /// The error of a header that could not be read, if one could not.
    #[must_use]
    pub fn header_error(&self) -> Option<&str> {
        match &self.header {
            Some(Err(error)) => Some(error),
            _ => None,
        }
    }
}

/// A source of the reads' provenance other than their header (e.g. evidence tables): what it
/// records, read only when needed.
pub type LazySource<'a> = Box<dyn FnOnce() -> Result<Option<AlignmentProvenance>> + 'a>;

/// How the reads were aligned, from `--alignment-tags` (`tags`), the header of `header` (a BAM,
/// CRAM or SAM file) and other `sources` (each named). With `none`, nothing is read, and nothing
/// recorded. Otherwise the tags are applied over each known source (replacing and removing tags),
/// and the sources must then agree (see [`agreed`]); without a known source, the tags set are the
/// provenance (their source [`FROM_TAGS`]) only if they include a comparable setting. Metadata and
/// removals alone establish nothing. A header that cannot be read is not an error here:
/// [`ReadsAlignment::header`] holds why, and the reads' alignment is then unknown
/// ([`UnknownReason::Unreadable`]) unless the tags give it.
///
/// # Errors
///
/// Returns an error if a source cannot be read, or two sources differ.
pub fn resolve_reads(
    tags: Option<&AlignmentTags>,
    header: Option<&Path>,
    sources: Vec<(String, LazySource<'_>)>,
) -> Result<ReadsAlignment> {
    resolve_reads_with_warnings(tags, header, sources, true)
}

/// Resolves sources, optionally warning about unknown sources among known ones.
fn resolve_reads_with_warnings(
    tags: Option<&AlignmentTags>,
    header: Option<&Path>,
    sources: Vec<(String, LazySource<'_>)>,
    warn_unknown: bool,
) -> Result<ReadsAlignment> {
    let edits = match tags {
        Some(AlignmentTags::None) => {
            return Ok(ReadsAlignment {
                from: Some(ProvenanceFrom::Tags),
                unknown: UnknownReason::TagsNone,
                ..Default::default()
            });
        }
        Some(AlignmentTags::Tags(edits)) => Some(edits),
        None => None,
    };
    let mut read: Vec<(String, Option<AlignmentProvenance>)> = Vec::new();
    for (name, source) in sources {
        read.push((name, source()?));
    }
    let header = header.map(|path| {
        let header = HeaderProvenance::read(path).map_err(|error| format!("{error:#}"));
        if let Ok(header) = &header {
            match &header.provenance {
                Some(provenance) => {
                    info!("The reads of {} were aligned with: {provenance}", path.display())
                }
                None => info!(
                    "The header of {} identifies no aligner, so its alignment is unknown",
                    path.display()
                ),
            }
            read.push((format!("--alignments {}", path.display()), header.provenance.clone()));
        }
        header
    });
    let given = !read.is_empty();
    let overlaid: Vec<(String, Option<AlignmentProvenance>)> = read
        .into_iter()
        .map(|(name, provenance)| {
            let provenance = provenance.map(|provenance| overlay(provenance, edits, &name));
            (name, provenance)
        })
        .collect();
    let mut reads = ReadsAlignment { header, ..Default::default() };
    let provenance =
        if warn_unknown { agreed(&overlaid)? } else { agreed_with_warnings(&overlaid, false)? };
    if let Some(provenance) = provenance {
        let from =
            if edits.is_some() { ProvenanceFrom::HeaderAndTags } else { ProvenanceFrom::Header };
        reads.provenance = Some(provenance);
        reads.from = Some(from);
    } else if let Some(set) = edits
        .map(|edits| &edits.set)
        .filter(|set| set.keys().any(|tag| !NOT_COMPARED.contains(&tag.as_str())))
    {
        let mut tags = set.clone();
        tags.insert(SOURCE.to_string(), FROM_TAGS.to_string());
        let literals = set
            .iter()
            .filter(|(_, value)| value.as_str() == MIXED)
            .map(|(tag, _)| tag.clone())
            .collect();
        record_literals(&mut tags, literals);
        reads.provenance = Some(AlignmentProvenance(tags));
        reads.from = Some(ProvenanceFrom::Tags);
    } else {
        reads.unknown = if reads.header_error().is_some() {
            UnknownReason::Unreadable
        } else if edits.is_some() {
            UnknownReason::TagsWithoutSettings
        } else if given {
            UnknownReason::NotRecorded
        } else {
            UnknownReason::NotGiven
        };
    }
    Ok(reads)
}

/// Resolves preflight sources as [`resolve_reads`], deferring missing-source warnings until the
/// required evidence has been read.
///
/// # Errors
///
/// Returns the same errors as [`resolve_reads`].
pub fn resolve_reads_quiet(
    tags: Option<&AlignmentTags>,
    header: Option<&Path>,
    sources: Vec<(String, LazySource<'_>)>,
) -> Result<ReadsAlignment> {
    resolve_reads_with_warnings(tags, header, sources, false)
}

// ------------------------------------------------------------------------------------------------
// Several sources

/// Two named provenances that differ: their names, and how (described with the names).
struct NamedConflict<'a> {
    /// The first's name.
    first: &'a str,
    /// The second's name.
    second: &'a str,
    /// How they differ.
    described: String,
}

/// The [`merge`] of named provenances (each `(name, provenance)`, `None` where unknown), with a
/// warning (made by `unknown_warning` of the names) about those unknown among known ones.
///
/// # Errors
///
/// Returns the first two that differ.
fn merge_named<'a, N: AsRef<str>>(
    named: &'a [(N, Option<&AlignmentProvenance>)],
    unknown_warning: impl Fn(&str) -> String,
    warn_unknown: bool,
) -> std::result::Result<Option<AlignmentProvenance>, NamedConflict<'a>> {
    let (names, known): (Vec<&str>, Vec<&AlignmentProvenance>) =
        named.iter().filter_map(|(name, p)| Some((name.as_ref(), (*p)?))).unzip();
    let unknown: Vec<&str> =
        named.iter().filter(|(_, p)| p.is_none()).map(|(name, _)| name.as_ref()).collect();
    if warn_unknown && !known.is_empty() && !unknown.is_empty() {
        warn!("{}", unknown_warning(&unknown.join(", ")));
    }
    merge(&known).map_err(|MergeConflict { first, second, differences }| {
        let (first, second) = (names[first], names[second]);
        NamedConflict {
            first,
            second,
            described: describe_differences(&differences, first, second),
        }
    })
}

/// The provenance of several sources of one sample's reads (evidence tables, the reads' header),
/// each `(name, provenance)`: their [`merge`] (`None` if none is known). Sources of unknown
/// provenance among known ones are logged.
///
/// # Errors
///
/// Returns an error naming two sources and the tags on which they differ, if they do.
pub fn agreed(
    sources: &[(String, Option<AlignmentProvenance>)],
) -> Result<Option<AlignmentProvenance>> {
    agreed_with_warnings(sources, true)
}

/// Merges sources with the same conflict errors, optionally warning about unknown sources.
fn agreed_with_warnings(
    sources: &[(String, Option<AlignmentProvenance>)],
    warn_unknown: bool,
) -> Result<Option<AlignmentProvenance>> {
    let named: Vec<(&str, Option<&AlignmentProvenance>)> =
        sources.iter().map(|(name, p)| (name.as_str(), p.as_ref())).collect();
    let warning = |unknown: &str| {
        format!("{unknown} record no alignment provenance; that of the others is used")
    };
    merge_named(&named, warning, warn_unknown).map_err(
        |NamedConflict { first, second, described }| {
            anyhow::anyhow!(
                "{first} and {second} were aligned differently: {described}. They should be of the \
             same reads: measure the tables again from them, pass --alignment-tags to set \
             (KEY=VALUE) or remove (KEY!) the tags they disagree on, or pass --alignment-tags \
             none to record nothing"
            )
        },
    )
}

/// The provenance a model trained on `tables` (each `(sample, provenance)`) records: their
/// [`merge`] if they agree, none if none records one, the known ones' (with a warning) if some do
/// not record one, and, with `mixed_allowed`, only the tags they share (see [`shared_tags`], with a
/// warning) if they disagree: such a model is then compared on those tags only.
///
/// # Errors
///
/// Returns an error naming two samples and the tags on which they differ if two tables were
/// aligned differently and `mixed_allowed` is false.
pub fn training_provenance(
    tables: &[(&str, Option<&AlignmentProvenance>)],
    mixed_allowed: bool,
) -> Result<Option<AlignmentProvenance>> {
    if tables.iter().all(|(_, p)| p.is_none()) {
        info!("The training tables record no alignment provenance, so the model records none");
        return Ok(None);
    }
    let warning = |unknown: &str| {
        format!(
            "the tables of {unknown} record no alignment provenance; the model records that of the \
             others"
        )
    };
    match merge_named(tables, warning, true) {
        Ok(merged) => Ok(merged),
        Err(NamedConflict { first, second, described }) => {
            if !mixed_allowed {
                bail!(
                    "the tables of {first} and {second} are of reads aligned differently: \
                     {described}. Train on tables of reads aligned alike, or pass \
                     --mixed-alignments to train on both and record only the tags they share"
                );
            }
            let known: Vec<&AlignmentProvenance> = tables.iter().filter_map(|(_, p)| *p).collect();
            let shared = shared_tags(&known);
            warn!(
                "--mixed-alignments: the tables are of reads aligned differently (e.g. \
                 {described}); the model records only the tags they share, and is compared on \
                 those only: {shared}"
            );
            Ok(Some(shared))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    thread_local! {
        static COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    pub(super) fn record_comparison() {
        COMPARISONS.set(COMPARISONS.get() + 1);
    }

    /// The `@PG` and `@SQ` lines of a CRAM aligned with bwa-mem3, duplicate-marked, sorted and
    /// converted.
    const PIPELINE: &str = "@HD\tVN:1.6\tSO:coordinate\n\
        @SQ\tSN:chr1\tLN:248956422\n\
        @PG\tID:bwa-mem3\tPN:bwa-mem3\tVN:0.16.0\tCL:bwa-mem3 mem -p -C -Y -K 100000000 -t 32 -H read_groups.txt --bam=0 GRCh38_noalt_hs38d1.fasta /dev/stdin\n\
        @PG\tID:DUPBLASTER\tVN:0.3.0\tCL:dupblaster --single-end-strategy picard-exact\tPP:bwa-mem3\n\
        @PG\tID:fgumi\tPN:fgumi\tVN:0.8.0\tCL:mako -i - -o out.bam\tPP:DUPBLASTER\n\
        @PG\tID:samtools\tPN:samtools\tPP:fgumi\tVN:1.25\tCL:samtools view -@ 8 -C -T ref.fasta out.bam\n";

    fn tags(pairs: &[(&str, &str)]) -> AlignmentProvenance {
        crate::testing::provenance(pairs, false)
    }

    fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        tags(pairs).0
    }

    fn of_header(text: &str) -> HeaderProvenance {
        HeaderProvenance::from_header_text(text)
    }

    /// The reads' alignment from `sources` (already read) and `tags`.
    fn resolved(
        sources: Vec<(String, Option<AlignmentProvenance>)>,
        tags: Option<&AlignmentTags>,
    ) -> Result<ReadsAlignment> {
        let sources = sources
            .into_iter()
            .map(|(name, provenance)| (name, Box::new(move || Ok(provenance)) as LazySource<'_>))
            .collect();
        resolve_reads(tags, None, sources)
    }

    fn parse(values: &[&str]) -> AlignmentTags {
        AlignmentTags::parse(values).unwrap().unwrap()
    }

    fn parse_edits(values: &[&str]) -> TagEdits {
        let AlignmentTags::Tags(edits) = parse(values) else { panic!("expected tag edits") };
        edits
    }

    fn named(
        name: &str,
        provenance: Option<AlignmentProvenance>,
    ) -> (String, Option<AlignmentProvenance>) {
        (name.to_string(), provenance)
    }

    #[test]
    fn smart_pairing_requires_explicit_settings_not_lossy_header_inference() {
        let paired = of_header("@PG\tID:bwa\tCL:bwa mem -p ref.fa reads.fq\n").provenance.unwrap();
        let single = of_header("@PG\tID:bwa\tCL:bwa mem ref.fa reads.fq\n").provenance.unwrap();
        assert_eq!(paired, single);
        assert!(incomplete_options_reason(&paired).is_some());
        let paired = resolved(
            vec![named("h", Some(paired))],
            Some(&parse(&["options-inference=explicit", "command=mem", "p="])),
        )
        .unwrap()
        .provenance
        .unwrap();
        let single = resolved(
            vec![named("h", Some(single))],
            Some(&parse(&["options-inference=explicit", "command=mem"])),
        )
        .unwrap()
        .provenance
        .unwrap();
        assert_eq!(
            compare_provenance(&paired, &single)
                .iter()
                .map(|difference| difference.tag.as_str())
                .collect::<Vec<_>>(),
            ["p"]
        );
    }

    #[rstest]
    #[case("O", "6:9", "6,9")]
    #[case("E", "1:3", "1,3")]
    #[case("L", "4:8", "4,8")]
    #[case("h", "5:200", "5,200")]
    fn paired_syntax_cannot_manufacture_inferred_differences(
        #[case] option: &str,
        #[case] colon: &str,
        #[case] comma: &str,
    ) {
        let header = |value: &str| {
            of_header(&format!("@PG\tID:bwa\tCL:bwa mem -{option}{value} ref.fa reads.fq\n"))
                .provenance
                .unwrap()
        };
        let a = header(colon);
        let b = header(comma);
        assert!(compare_provenance(&a, &b).is_empty());
        assert!(incomplete_options_reason(&a).is_some());
        assert!(incomplete_options_reason(&b).is_some());
        let settings = parse(&["options-inference=explicit", &format!("{option}={comma}")]);
        let a = resolved(vec![named("a", Some(a))], Some(&settings)).unwrap().provenance.unwrap();
        let b = resolved(vec![named("b", Some(b))], Some(&settings)).unwrap().provenance.unwrap();
        assert!(compare_provenance(&a, &b).is_empty());
        assert!(incomplete_options_reason(&a).is_none());
        let changed = resolved(
            vec![named("b", Some(header(comma)))],
            Some(&parse(&["options-inference=explicit", &format!("{option}=7,7")])),
        )
        .unwrap()
        .provenance
        .unwrap();
        assert_eq!(compare_provenance(&a, &changed).len(), 1);
    }

    #[rstest]
    #[case("bwa mem -o result -Y ref.fa reads.fq")]
    #[case("bwa-mem2 mem -p -K100 ref.fa reads.fq")]
    #[case("bwa-mem3 mem --alt-project primary ref.fa reads.fq")]
    #[case("bwa-mem3-meth mem -p -Y ref.fa reads.fq")]
    fn lossy_bwa_headers_retain_facts_but_not_inferred_settings(#[case] command: &str) {
        let aligner = command.split_whitespace().next().unwrap();
        let header = of_header(&format!(
            "@SQ\tSN:chr1\tLN:10\tAH:*\n@PG\tID:{aligner}\tPN:{aligner}\tVN:1\tCL:{command}\n"
        ));
        let provenance = header.provenance.unwrap();
        assert_eq!(
            provenance,
            tags(&[
                ("aligner", aligner),
                ("version", "1"),
                ("sq-alt", "yes"),
                ("source", "header"),
                ("options-inference", "incomplete"),
            ])
        );
        assert_eq!(AlignmentProvenance::from_json(&provenance.to_json()).unwrap(), provenance);
    }

    #[rstest]
    #[case::marked("bwa", false)]
    #[case::legacy("bwa", true)]
    #[case::unprofiled("mapper", false)]
    fn constraints_survive_grouped_merges_with_incomplete_sources(
        #[case] aligner: &str,
        #[case] legacy: bool,
    ) {
        let complete = tags(&[("aligner", aligner), ("options-inference", "explicit")]);
        let incomplete = if legacy {
            AlignmentProvenance::from_json(r#"{"aligner":"bwa"}"#).unwrap()
        } else {
            tags(&[("aligner", aligner), ("options-inference", "incomplete")])
        };
        let enabled = tags(&[("aligner", aligner), ("source", "tags"), ("Y", "")]);
        let sources = [&complete, &incomplete, &enabled];
        for order in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let [a, b, c] = order.map(|index| sources[index]);
            assert_eq!(merge(&[a, b, c]).unwrap_err().differences[0].tag, "Y");
            // Only the complete source and the enabled flag conflict as a pair; every other pair
            // must group, so that no order is skipped.
            let pair = merge(&[a, b]);
            if order[..2].contains(&0) && order[..2].contains(&2) {
                assert_eq!(pair.unwrap_err().differences[0].tag, "Y");
            } else {
                let grouped = pair.unwrap().unwrap();
                assert!(incomplete_options_reason(&grouped).is_some());
                let json = grouped.to_json();
                assert_eq!(serde_json::to_string(&grouped).unwrap(), json);
                let restored = AlignmentProvenance::from_json(&json).unwrap();
                assert_eq!(restored, grouped);
                for order in [[&restored, c], [c, &restored]] {
                    let conflict =
                        merge(&order).expect_err("grouping must not erase known absence");
                    assert_eq!(conflict.differences[0].tag, "Y");
                }
            }
        }
    }

    #[test]
    fn constraints_remain_complete_across_repeated_merges_without_certifying_inference() {
        let complete = tags(&[("aligner", "bwa"), ("Y", "")]);
        let incomplete = tags(&[("aligner", "bwa"), ("options-inference", "incomplete")]);
        let partial = tags(&[("aligner", "bwa"), ("source", "tags")]);
        let mut merged = merge(&[&complete, &incomplete]).unwrap().unwrap();
        for _ in 0..3 {
            merged = merge(&[&partial, &merged, &incomplete]).unwrap().unwrap();
            assert!(incomplete_options_reason(&merged).is_some());
            assert_eq!(merged.get(MERGED_COMPLETE), Some(YES));
            assert_eq!(merged.get("Y"), Some(""));
            assert_eq!(comparable_tags(&merged, &complete), BTreeSet::from(["Y", "aligner"]));
            let different = tags(&[("aligner", "bwa"), ("Y", ""), ("p", "")]);
            assert_eq!(compare_provenance(&merged, &different)[0].tag, "p");
            assert_eq!(compare_provenance(&different, &merged)[0].tag, "p");
        }
        assert!(!merged.to_string().contains(MERGED_COMPLETE));
        assert!(indeterminate_tags(&merged, &merged).is_empty());
    }

    #[test]
    fn constraints_are_not_invented_for_partial_sources_or_mixed_training() {
        let incomplete = tags(&[("aligner", "bwa"), ("options-inference", "incomplete")]);
        let partial =
            tags(&[("aligner", "bwa"), ("source", "tags"), ("options-inference", "explicit")]);
        let enabled = tags(&[("aligner", "bwa"), ("Y", "")]);
        let merged = merge(&[&incomplete, &partial]).unwrap().unwrap();
        assert_eq!(merged.get(MERGED_COMPLETE), None);
        assert!(compare_provenance(&merged, &enabled).is_empty());
        let complete = tags(&[("aligner", "bwa")]);
        let merged = merge(&[&incomplete, &complete]).unwrap().unwrap();
        let mixed = training_provenance(&[("a", Some(&merged)), ("b", Some(&enabled))], true)
            .unwrap()
            .unwrap();
        assert_eq!(mixed.get(MERGED_COMPLETE), None);
        assert!(incomplete_options_reason(&mixed).is_some());
        assert!(compare_provenance(&mixed, &enabled).is_empty());
        // A tags-only intersection must not resurrect even stale complete-contributor metadata.
        let mut partial = partial;
        partial.0.insert(MERGED_COMPLETE.to_string(), YES.to_string());
        assert!(!partial.has_complete_settings());
        assert_eq!(merge(&[&partial, &incomplete]).unwrap().unwrap().get(MERGED_COMPLETE), None);
    }

    #[test]
    fn constraints_metadata_is_not_a_comparable_setting_or_mixed_marker() {
        let metadata = tags(&[(MERGED_COMPLETE, MIXED)]);
        assert!(comparable_tags(&metadata, &metadata).is_empty());
        assert!(indeterminate_tags(&metadata, &metadata).is_empty());
        assert!(compare_provenance(&metadata, &tags(&[])).is_empty());
        assert!(metadata.to_string().is_empty());
    }

    #[test]
    fn constraints_follow_overrides_but_cannot_be_supplied_by_users() {
        let complete = tags(&[("aligner", "bwa")]);
        let incomplete = tags(&[("aligner", "bwa"), ("options-inference", "incomplete")]);
        let merged = merge(&[&complete, &incomplete]).unwrap().unwrap();
        assert!(AlignmentTags::parse(&[format!("{MERGED_COMPLETE}=yes")]).is_err());
        assert!(AlignmentTags::parse(&[format!("{MERGED_COMPLETE}!")]).is_err());
        let enabled = overlay(merged.clone(), Some(&parse_edits(&["Y="])), "merged");
        assert_eq!(enabled.get(MERGED_COMPLETE), Some(YES));
        assert!(incomplete_options_reason(&enabled).is_some());
        assert_eq!(compare_provenance(&enabled, &complete)[0].tag, "Y");
        let removed = overlay(enabled, Some(&parse_edits(&["Y!"])), "merged");
        assert_eq!(removed, merged);
        let declared =
            overlay(merged, Some(&parse_edits(&["options-inference=explicit", "Y="])), "merged");
        assert_eq!(declared.get(MERGED_COMPLETE), None);
        assert!(incomplete_options_reason(&declared).is_none());
        assert_eq!(compare_provenance(&declared, &complete)[0].tag, "Y");
    }

    #[test]
    fn conversion_only_headers_do_not_identify_an_aligner() {
        let header = of_header("@PG\tID:samtools.1\tPN:samtools\tCL:samtools view -b reads.sam\n");
        assert_eq!(header.provenance, None);
        assert_eq!(header.unprofiled_warning(Path::new("reads.bam")), None);
    }

    #[test]
    fn conversion_only_cycles_are_unknown_and_explicit_tags_are_tags_only() {
        let header = of_header(
            "@PG\tID:samtools.1\tPP:samtools.2\tCL:samtools view reads.sam\n\
             @PG\tID:samtools.2\tPP:samtools.1\tCL:samtools sort reads.bam\n",
        );
        assert!(header.provenance.is_none());
        let reads =
            resolved(vec![named("header", header.provenance)], Some(&parse(&["aligner=mapper"])))
                .unwrap();
        assert_eq!(reads.from, Some(ProvenanceFrom::Tags));
        assert_eq!(reads.provenance.unwrap(), tags(&[("aligner", "mapper"), ("source", "tags")]));
    }

    #[test]
    fn missing_bwa_command_lines_do_not_establish_default_settings() {
        let header = of_header("@PG\tID:bwa\tPN:bwa\tVN:0.7.17\n");
        let provenance = header.provenance.unwrap();
        assert!(incomplete_options_reason(&provenance).is_some());
        assert_eq!(provenance.get("version"), Some("0.7.17"));
        assert_eq!(provenance.get("t"), None);
        assert_eq!(provenance.get("command"), None);
    }

    #[test]
    fn converters_do_not_hide_an_unknown_aligner_or_add_mixed_tags() {
        let header = of_header(
            "@PG\tID:convert\tPN:samtools\tCL:samtools import reads.fq\n\
             @PG\tID:mm\tPN:mapper\tPP:convert\tCL:mapper --secondary=no ref.fa reads.fq\n\
             @PG\tID:sort\tPN:samtools\tPP:mm\tCL:samtools sort reads.bam\n\
             @PG\tID:unrelated\tPN:samtools\tCL:samtools view other.sam\n",
        );
        assert_eq!(header.provenance.as_ref().unwrap().get("aligner"), Some("mapper"));
        assert_eq!(header.provenance.as_ref().unwrap().get("secondary"), Some("no"));
        assert!(header.mixed.is_empty());
        assert_eq!(header.unprofiled.as_deref(), Some("mapper"));
    }

    #[rstest]
    #[case(r#"{"aligner":"bwa","Y":""}"#, true)]
    #[case(r#"{"aligner":"bwa-mem2","source":"header","K":"100"}"#, true)]
    #[case(r#"{"aligner":"bwa-mem3","source":"header"}"#, true)]
    #[case(r#"{"aligner":"bwa-mem3-meth"}"#, true)]
    #[case(r#"{"aligner":"bwa","source":"tags","Y":""}"#, false)]
    #[case(r#"{"aligner":"bwa","options-inference":"explicit","Y":""}"#, false)]
    #[case(r#"{"aligner":"mapper","source":"header"}"#, true)]
    #[case(r#"{"source":"header","alt-project":"primary"}"#, true)]
    #[case(r#"{"source":"header"}"#, true)]
    #[case(r#"{"source":"tags","alt-project":"primary"}"#, false)]
    #[case(r#"{"source":"header","options-inference":"explicit"}"#, false)]
    fn legacy_header_inference_requires_an_explicit_declaration(
        #[case] json: &str,
        #[case] incomplete: bool,
    ) {
        let provenance = AlignmentProvenance::from_json(json).unwrap();
        assert_eq!(incomplete_options_reason(&provenance).is_some(), incomplete);
        assert!(!provenance.0.contains_key("source") || json.contains("source"));
        assert!(
            !provenance.0.contains_key("options-inference") || json.contains("options-inference")
        );
    }

    #[test]
    fn legacy_incompleteness_survives_merge_mixed_training_and_unrelated_edits() {
        let legacy = AlignmentProvenance::from_json(r#"{"aligner":"bwa","Y":""}"#).unwrap();
        let declared = AlignmentProvenance::from_json(
            r#"{"aligner":"bwa","Y":"","options-inference":"explicit"}"#,
        )
        .unwrap();
        for sources in [[&legacy, &declared], [&declared, &legacy]] {
            let merged = merge(&sources).unwrap().unwrap();
            assert_eq!(merged.get("options-inference"), Some("incomplete"));
        }
        let other = AlignmentProvenance::from_json(r#"{"aligner":"mapper"}"#).unwrap();
        let mixed = training_provenance(&[("a", Some(&legacy)), ("b", Some(&other))], true)
            .unwrap()
            .unwrap();
        assert_eq!(mixed.get("options-inference"), Some("incomplete"));
        for edit in ["sq-alt=no", "aligner=mapper", "aligner!"] {
            let reads =
                resolved(vec![named("legacy", Some(legacy.clone()))], Some(&parse(&[edit])))
                    .unwrap();
            assert!(incomplete_options_reason(&reads.provenance.unwrap()).is_some(), "{edit}");
        }
        let explicit = resolved(
            vec![named("legacy", Some(legacy))],
            Some(&parse(&["options-inference=explicit", "Y="])),
        )
        .unwrap()
        .provenance
        .unwrap();
        assert!(incomplete_options_reason(&explicit).is_none());
    }

    #[test]
    fn legacy_header_without_an_aligner_keeps_uncertainty_across_merges_and_edits() {
        let legacy =
            AlignmentProvenance::from_json(r#"{"source":"header","alt-project":"primary"}"#)
                .unwrap();
        let enabled = tags(&[("source", "tags"), ("alt-project", "primary"), ("Y", "")]);
        for sources in [[&legacy, &enabled], [&enabled, &legacy]] {
            let merged = merge(&sources).expect("missing legacy settings are unknown").unwrap();
            let restored = AlignmentProvenance::from_json(&merged.to_json()).unwrap();
            assert_eq!(restored.get("Y"), Some(""));
            assert!(incomplete_options_reason(&restored).is_some());
            assert_eq!(restored.get(MERGED_COMPLETE), None);
        }
        for edit in ["alt-project!", "aligner=mapper", "aligner!", "sq-alt=no"] {
            let edited = overlay(legacy.clone(), Some(&parse_edits(&[edit])), "legacy");
            assert!(incomplete_options_reason(&edited).is_some(), "{edit}");
        }
        let different = tags(&[("source", "tags"), ("alt-project", "keep-alt")]);
        let shared =
            training_provenance(&[("legacy", Some(&legacy)), ("other", Some(&different))], true)
                .unwrap()
                .unwrap();
        assert!(incomplete_options_reason(&shared).is_some());
        let explicit =
            overlay(legacy, Some(&parse_edits(&["options-inference=explicit"])), "legacy");
        assert!(incomplete_options_reason(&explicit).is_none());
        assert_eq!(compare_provenance(&explicit, &enabled)[0].tag, "Y");
    }

    #[test]
    fn incomplete_headers_do_not_assert_absent_options() {
        let incomplete =
            tags(&[("aligner", "bwa"), ("source", "header"), ("options-inference", "incomplete")]);
        let explicit = tags(&[
            ("aligner", "bwa"),
            ("source", "header"),
            ("options-inference", "explicit"),
            ("Y", ""),
        ]);
        assert!(compare_provenance(&incomplete, &explicit).is_empty());
        assert_eq!(comparable_tags(&incomplete, &explicit), BTreeSet::from(["aligner"]));
        for sources in [[&incomplete, &explicit], [&explicit, &incomplete]] {
            let merged = merge(&sources).unwrap().unwrap();
            assert_eq!(merged.get("Y"), Some(""));
            assert!(incomplete_options_reason(&merged).is_some());
        }
        let declared_absent = resolved(
            vec![named("header", Some(incomplete))],
            Some(&parse(&["options-inference=explicit"])),
        )
        .unwrap()
        .provenance
        .unwrap();
        assert_eq!(
            compare_provenance(&declared_absent, &explicit)
                .iter()
                .map(|difference| difference.tag.as_str())
                .collect::<Vec<_>>(),
            ["Y"]
        );
    }

    #[test]
    fn incomplete_option_state_is_metadata_not_a_compared_setting() {
        let incomplete = tags(&[("aligner", "bwa"), ("options-inference", "incomplete")]);
        let explicit = tags(&[("aligner", "bwa"), ("options-inference", "explicit")]);
        assert!(compare_provenance(&incomplete, &explicit).is_empty());
        assert_eq!(comparable_tags(&incomplete, &explicit), BTreeSet::from(["aligner"]));
        assert!(
            comparable_tags(
                &tags(&[("options-inference", "incomplete")]),
                &tags(&[("options-inference", "explicit")])
            )
            .is_empty()
        );
        let absent = tags(&[("aligner", "bwa"), ("source", "tags")]);
        for sources in [
            [&explicit, &incomplete],
            [&incomplete, &explicit],
            [&absent, &incomplete],
            [&incomplete, &absent],
        ] {
            let merged = merge(&sources).unwrap().unwrap();
            assert_eq!(merged.get("options-inference"), Some("incomplete"));
            assert!(incomplete_options_reason(&merged).is_some());
            let agreed = agreed(&[
                ("a".to_string(), Some(sources[0].clone())),
                ("b".to_string(), Some(sources[1].clone())),
            ])
            .unwrap()
            .unwrap();
            assert!(incomplete_options_reason(&agreed).is_some());
        }
        let different = tags(&[("aligner", "mapper"), ("options-inference", "explicit")]);
        let shared =
            training_provenance(&[("a", Some(&incomplete)), ("b", Some(&different))], true)
                .unwrap()
                .unwrap();
        assert_eq!(shared.get("options-inference"), Some("incomplete"));
        for marker in ["incomplete", "mixed", "unexpected"] {
            assert!(incomplete_options_reason(&tags(&[("options-inference", marker)])).is_some());
        }
        assert!(incomplete_options_reason(&explicit).is_none());
        assert!(incomplete_options_reason(&absent).is_none());
    }

    #[test]
    fn incomplete_option_state_requires_an_explicit_declaration() {
        let detected =
            tags(&[("aligner", "bwa"), ("options-inference", "incomplete"), ("source", "header")]);
        let unrelated =
            resolved(vec![named("header", Some(detected.clone()))], Some(&parse(&["sq-alt=no"])))
                .unwrap();
        assert_eq!(unrelated.provenance.unwrap().get("options-inference"), Some("incomplete"));
        let removal =
            resolved(vec![named("header", Some(detected.clone()))], Some(&parse(&["Y!"]))).unwrap();
        assert!(incomplete_options_reason(&removal.provenance.unwrap()).is_some());
        let suppressed =
            resolved(vec![named("header", Some(detected.clone()))], Some(&AlignmentTags::None))
                .unwrap();
        assert_eq!(suppressed.provenance, None);
        assert_eq!(suppressed.unknown, UnknownReason::TagsNone);
        assert!(AlignmentTags::parse(&["options-inference!"]).is_err());
        assert!(AlignmentTags::parse(&["options-inference=guess"]).is_err());
        let declared = resolved(
            vec![named("header", Some(detected))],
            Some(&parse(&["options-inference=explicit", "command=mem", "Y="])),
        )
        .unwrap();
        assert_eq!(declared.provenance.unwrap().get("options-inference"), Some("explicit"));
        assert_eq!(declared.from, Some(ProvenanceFrom::HeaderAndTags));
    }

    #[test]
    fn a_pipelines_aligner_is_found_and_normalized() {
        let header = of_header(PIPELINE);
        assert_eq!(
            header.provenance.unwrap(),
            tags(&[
                ("aligner", "bwa-mem3"),
                ("options-inference", "incomplete"),
                ("source", "header"),
                ("sq-alt", "no"),
                ("version", "0.16.0"),
            ])
        );
        assert!(header.mixed.is_empty());
        assert_eq!(header.unprofiled, None);
    }

    /// `samtools merge` keeps both inputs' chains, suffixing the second's ids.
    #[test]
    fn a_merged_header_is_one_provenance() {
        let header = of_header(
            "@PG\tID:bwa-mem3\tPN:bwa-mem3\tVN:0.16.0\tCL:bwa-mem3 mem -Y -t 8 ref.fa a.fq\n\
             @PG\tID:samtools\tPN:samtools\tPP:bwa-mem3\tVN:1.25\tCL:samtools sort a.bam\n\
             @PG\tID:bwa-mem3-3D13BF94\tPN:bwa-mem3\tVN:0.16.0\tCL:bwa-mem3 mem -Y -t 8 ref.fa b.fq\n\
             @PG\tID:samtools-21BEF3EE\tPN:samtools\tPP:bwa-mem3-3D13BF94\tVN:1.25\tCL:samtools sort b.bam\n",
        );
        assert!(header.mixed.is_empty());
        assert!(incomplete_options_reason(&header.provenance.unwrap()).is_some());
    }

    #[test]
    fn aligner_records_that_disagree_are_mixed() {
        let header = of_header(
            "@PG\tID:bwa-mem3\tPN:bwa-mem3\tCL:bwa-mem3 mem --alt-project primary ref.fa a.fq\n\
             @PG\tID:bwa\tPN:bwa\tCL:bwa mem ref.fa b.fq\n",
        );
        assert_eq!(header.mixed, ["aligner"]);
        let provenance = header.provenance.clone().unwrap();
        assert_eq!(provenance.get("aligner"), Some(MIXED));
        assert!(incomplete_options_reason(&provenance).is_some());
        assert_eq!(provenance.get("command"), None);
        assert!(header.mixed_warning(Path::new("x.bam")).is_some());
        assert_eq!(header.unprofiled_warning(Path::new("x.bam")), None);
    }

    /// An aligner record that is an ancestor of a newer one is stale.
    #[test]
    fn a_stale_aligner_record_is_ignored() {
        let header = of_header(
            "@PG\tID:bwa\tPN:bwa\tCL:bwa mem -M ref.fa a.fq\n\
             @PG\tID:samtools\tPN:samtools\tPP:bwa\tCL:samtools fastq a.bam\n\
             @PG\tID:bwa-mem3\tPN:bwa-mem3\tPP:samtools\tCL:bwa-mem3 mem -Y ref.fa -\n",
        );
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("aligner"), Some("bwa-mem3"));
        assert_eq!(provenance.get("M"), None);
        assert_eq!(provenance.get("Y"), None);
        assert!(incomplete_options_reason(&provenance).is_some());
    }

    /// Aligner records whose PP chains loop are each other's ancestors: they all stand in.
    #[test]
    fn aligner_records_in_a_pp_cycle_all_stand_in() {
        let header = of_header(
            "@PG\tID:bwa\tPN:bwa\tPP:bwa-mem3\tCL:bwa mem -M ref.fa a.fq\n\
             @PG\tID:bwa-mem3\tPN:bwa-mem3\tPP:bwa\tCL:bwa-mem3 mem -M ref.fa a.fq\n",
        );
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("aligner"), Some(MIXED));
        assert_eq!(provenance.get("M"), None);
        assert!(incomplete_options_reason(&provenance).is_some());
    }

    #[rstest]
    #[case::by_id("@PG\tID:bwa-mem3.1\tCL:x mem -Y ref.fa a.fq\n", "bwa-mem3")]
    #[case::by_merge_suffix("@PG\tID:bwa-mem3-3D13BF94\tCL:x mem -Y ref.fa a.fq\n", "bwa-mem3")]
    #[case::by_cl("@PG\tID:\tCL:/opt/bin/bwa-mem2 mem -Y ref.fa a.fq\n", "bwa-mem2")]
    #[case::by_cl_before_id("@PG\tID:align-s1\tCL:/opt/bin/bwa mem ref.fa a.fq\n", "bwa")]
    #[case::by_cl_before_suffixed_id("@PG\tID:align-s2.1\tCL:minimap2 ref.fa a.fq\n", "minimap2")]
    // A PN that is not the aligner's (as some tools write): the profile is its command's, and
    // the aligner is recorded as the PN.
    #[case::profile_by_cl("@PG\tID:bwa\tPN:bwamem\tCL:bwa mem -Y ref.fa a.fq\n", "bwamem")]
    fn the_aligner_without_its_pn(#[case] text: &str, #[case] aligner: &str) {
        let header = of_header(text);
        assert_eq!(header.unprofiled, None);
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("aligner"), Some(aligner));
        assert_eq!(provenance.get("command"), None);
        assert_eq!(provenance.get("Y"), None);
        assert!(incomplete_options_reason(&provenance).is_some());
    }

    #[test]
    fn an_aligner_without_a_profile_is_read_generically() {
        let header = of_header(
            "@PG\tID:mapper\tPN:mapper\tVN:2.28\tCL:mapper -ax sr -t8 -@4 --secondary=no ref.fa r.fq\n\
             @PG\tID:samtools\tPN:samtools\tPP:mapper\tCL:samtools sort -o x.bam\n",
        );
        assert_eq!(header.unprofiled.as_deref(), Some("mapper"));
        assert_eq!(
            header.provenance.clone().unwrap(),
            tags(&[
                ("aligner", "mapper"),
                ("options-inference", "incomplete"),
                ("secondary", "no"),
                ("source", "header"),
                ("sq-alt", "no"),
                ("version", "2.28"),
            ])
        );
        // samtools follows it, so it may not be the aligner.
        let warning = header.unprofiled_warning(Path::new("x.bam")).unwrap();
        assert!(warning.contains("may be a tool run before the aligner"), "{warning}");
        let alone = of_header("@PG\tID:mapper\tPN:mapper\tCL:mapper -ax sr ref.fa r.fq\n");
        let warning = alone.unprofiled_warning(Path::new("x.bam")).unwrap();
        assert!(!warning.contains("may be a tool"), "{warning}");
    }

    /// minimap2 joins its arguments with spaces, as BWA does: `-o 'out --secondary=no'` and
    /// `-o out --secondary=no` give the same CL, so its options are not inferred.
    #[test]
    fn pr38_minimap2_command_lines_are_incomplete() {
        let header = of_header(
            "@PG\tID:minimap2\tPN:minimap2\tVN:2.28\tCL:minimap2 -ax sr -o out --secondary=no \
             ref.fa r.fq\n",
        );
        assert_eq!(header.unprofiled, None);
        assert_eq!(
            header.provenance.unwrap(),
            tags(&[
                ("aligner", "minimap2"),
                ("options-inference", "incomplete"),
                ("source", "header"),
                ("sq-alt", "no"),
                ("version", "2.28"),
            ])
        );
        // Legacy provenance recording minimap2 without the marker is incomplete too.
        let legacy = AlignmentProvenance(BTreeMap::from([
            ("aligner".to_string(), "minimap2".to_string()),
            ("secondary".to_string(), "no".to_string()),
        ]));
        assert!(incomplete_options_reason(&legacy).is_some());
    }

    /// getopt reads a value attached to a short option (`-oFILE`): an ignored short option with an
    /// attached value is left out as its separate form is, so no path becomes a tag's name.
    #[rstest]
    #[case::output("-o/private/run-a.sam", "-o /private/run-a.sam")]
    #[case::read_group("-R@RG\\tID:a", "-R @RG\\tID:a")]
    #[case::threads("-t8", "-t 8")]
    #[case::verbosity("-v3", "-v 3")]
    fn pr38_attached_values_of_ignored_short_options_are_left_out(
        #[case] attached: &str,
        #[case] separate: &str,
    ) {
        let read = |option: &str| {
            let text = format!("@PG\tID:m\tPN:mapper\tCL:mapper {option} --secondary=no r.fq\n");
            of_header(&text).provenance.unwrap()
        };
        let attached = read(attached);
        assert_eq!(attached, read(separate));
        assert_eq!(attached.get("secondary"), Some("no"));
        assert!(attached.0.keys().all(|key| !key.contains('/') && key != "o"));
    }

    /// A record that cannot establish its options says nothing about one another records: that is
    /// not a disagreement, though the result stays incomplete. Different recorded values still disagree.
    #[test]
    fn pr38_an_incomplete_record_does_not_make_options_mixed() {
        let both = "@PG\tID:a\tPN:mapper\tVN:1\tCL:mapper --secondary=no r1.fq\n\
                    @PG\tID:b\tPN:mapper\tVN:1\n";
        let header = of_header(both);
        assert!(header.mixed.is_empty(), "{:?}", header.mixed);
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("secondary"), Some("no"));
        assert_eq!(provenance.get(OPTIONS_INFERENCE), Some(INCOMPLETE));
        let disagreeing = "@PG\tID:a\tPN:mapper\tVN:1\tCL:mapper --secondary=no r1.fq\n\
                        @PG\tID:b\tPN:mapper\tVN:1\tCL:mapper --secondary=yes r2.fq\n";
        let header = of_header(disagreeing);
        assert_eq!(header.mixed, ["secondary"]);
        assert_eq!(header.provenance.unwrap().get("secondary"), Some(MIXED));
    }

    /// An explicit declaration makes omitted options known absent only on header-derived
    /// provenance; given alone, it is partial and compares only the tags it records.
    #[test]
    fn pr38_only_header_derived_declarations_make_omissions_absent() {
        let model = tags(&[("aligner", "bwa"), ("p", "")]);
        let declaration = parse(&["options-inference=explicit", "aligner=bwa"]);
        let alone = resolved(Vec::new(), Some(&declaration)).unwrap().provenance.unwrap();
        assert_eq!(alone.get(SOURCE), Some(FROM_TAGS));
        assert!(!comparable_tags(&model, &alone).contains("p"));
        let header = of_header("@PG\tID:bwa\tPN:bwa\tVN:0.7.17\tCL:bwa mem -p ref.fa r.fq\n");
        let declared =
            resolved(vec![named("header", header.provenance)], Some(&declaration)).unwrap();
        assert!(comparable_tags(&model, &declared.provenance.unwrap()).contains("p"));
    }

    #[test]
    fn pr38_unprofiled_options_cannot_give_winnowr_metadata() {
        let header = of_header(
            "@PG\tID:mm\tPN:wrapper\tCL:wrapper --options-inference=explicit \
             --winnowr-merged-complete=yes --source=tags --version=9 --aligner=other \
             --sq-alt=yes --secondary=no ref.fa r.fq\n",
        );
        let provenance = header.provenance.unwrap();
        assert_eq!(
            provenance,
            tags(&[
                ("aligner", "wrapper"),
                ("options-inference", "incomplete"),
                ("secondary", "no"),
                ("source", "header"),
                ("sq-alt", "no"),
            ])
        );
        assert!(incomplete_options_reason(&provenance).is_some());
        let incomplete =
            of_header("@PG\tID:mm\tPN:wrapper\tCL:wrapper --options-inference=incomplete r.fq\n");
        assert!(incomplete_options_reason(&incomplete.provenance.unwrap()).is_some());
    }

    #[rstest]
    #[case::absent("")]
    #[case::empty("\tCL:")]
    #[case::blank("\tCL: ")]
    fn pr38_an_unprofiled_record_without_a_command_line_has_incomplete_options(
        #[case] command_line: &str,
    ) {
        let header = of_header(&format!("@PG\tID:mm\tPN:mapper\tVN:2.28{command_line}\n"));
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get(ALIGNER), Some("mapper"));
        assert_eq!(provenance.get(VERSION), Some("2.28"));
        assert_eq!(provenance.get(OPTIONS_INFERENCE), Some(INCOMPLETE));
        let reason = incomplete_options_reason(&provenance).unwrap();
        assert!(reason.contains("missing"), "{reason}");
        // The header says nothing about the option, so it constrains only what it records.
        let declared = tags(&[("aligner", "mapper"), ("secondary", "no")]);
        assert!(!comparable_tags(&declared, &provenance).contains("secondary"));
        // An explicit declaration resolves the uncertainty.
        let explicit = resolved(
            vec![named("header", Some(provenance))],
            Some(&parse(&["options-inference=explicit", "secondary=no"])),
        )
        .unwrap();
        assert!(incomplete_options_reason(&explicit.provenance.unwrap()).is_none());
    }

    #[test]
    fn no_program_is_unknown_and_alt_contigs_are_seen() {
        assert_eq!(
            of_header("@SQ\tSN:chr1\tLN:10\tAH:*\n"),
            HeaderProvenance {
                header_records: vec!["@SQ\tSN:chr1\tLN:10\tAH:*".to_string()],
                ..Default::default()
            }
        );
        let text = format!("{PIPELINE}@SQ\tSN:chr1_alt\tLN:10\tAH:chr1:1-10\n");
        assert_eq!(of_header(&text).provenance.unwrap().get(SQ_ALT), Some("yes"));
    }

    #[test]
    fn independent_unprofiled_chains_are_not_hidden_by_profiled_aligners() {
        let header = of_header(
            "@PG\tID:bwa\tPN:bwa\tCL:bwa mem -Y ref.fa a.fq\n\
             @PG\tID:sort\tPN:samtools\tPP:bwa\tCL:samtools sort a.bam\n\
             @PG\tID:mm\tPN:mapper\tCL:mapper --secondary=no ref.fa b.fq\n\
             @PG\tID:sort2\tPN:samtools\tPP:mm\tCL:samtools sort b.bam\n",
        );
        assert_eq!(header.provenance.as_ref().unwrap().get("aligner"), Some(MIXED));
        assert!(header.mixed_warning(Path::new("reads.bam")).is_some());
        assert!(header.unprofiled_warning(Path::new("reads.bam")).is_some());
    }

    #[test]
    fn shared_root_alignment_branches_are_each_represented() {
        let header = of_header(
            "@PG\tID:root\tPN:fastq-to-bam\tCL:fastq-to-bam r.fq\n\
             @PG\tID:bwa\tPN:bwa\tPP:root\tCL:bwa mem -Y ref.fa a.fq\n\
             @PG\tID:mm\tPN:mapper\tPP:root\tCL:mapper --secondary=no ref.fa b.fq\n",
        );
        assert_eq!(header.provenance.as_ref().unwrap().get("aligner"), Some(MIXED));
        assert!(header.mixed_warning(Path::new("reads.bam")).is_some());
        assert!(header.unprofiled_warning(Path::new("reads.bam")).is_some());
    }

    #[test]
    fn realignment_on_one_branch_does_not_hide_another_branchs_old_alignments() {
        let header = of_header(
            "@PG\tID:old\tPN:bwa\tCL:bwa mem -M ref.fa a.fq\n\
             @PG\tID:sort\tPN:samtools\tPP:old\tCL:samtools sort a.bam\n\
             @PG\tID:new\tPN:bwa-mem3\tPP:old\tCL:bwa-mem3 mem -Y ref.fa a.fq\n",
        );
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("aligner"), Some(MIXED));
        assert_eq!(provenance.get("M"), None);
        assert_eq!(provenance.get("Y"), None);
        assert!(incomplete_options_reason(&provenance).is_some());
    }

    #[test]
    fn an_independent_cycle_is_not_hidden_by_a_rooted_chain() {
        let header = of_header(
            "@PG\tID:bwa\tPN:bwa\tCL:bwa mem ref.fa a.fq\n\
             @PG\tID:mm\tPN:mapper\tPP:tool\tCL:mapper ref.fa b.fq\n\
             @PG\tID:tool\tPN:samtools\tPP:mm\tCL:samtools sort b.bam\n",
        );
        assert_eq!(header.provenance.unwrap().get("aligner"), Some(MIXED));
    }

    #[rstest]
    #[case::paired("mapper -ax sr --MD /private/sample.fa r1.fq r2.fq")]
    #[case::single("mapper -ax sr --MD ref.fa r.fq")]
    fn unknown_options_never_consume_separate_values(#[case] command: &str) {
        assert!(fallback_tags(command).is_empty());
    }

    #[test]
    fn review_fallback_drops_path_bearing_short_keys() {
        for option in ["-x/private/ref.idx", "-I/private/sample.idx", r"-xC:\private\ref.idx"] {
            assert_eq!(
                fallback_tags(&format!("mapper {option} -x --ref=/private/ref.idx")),
                map(&[("ref", "ref.idx")])
            );
        }
    }

    #[test]
    fn review_fallback_values_survive_bare_occurrences() {
        let before = fallback_tags("mapper --custom=a --custom --custom");
        let after = fallback_tags("mapper --custom --custom=a");
        assert_eq!(before.get("custom").unwrap(), "a");
        assert_eq!(before, after);
        assert_ne!(before, fallback_tags("mapper --custom"));
    }

    #[rstest]
    #[case("/private/sample/ref.idx", "ref.idx")]
    #[case(r"C:\private\sample\ref.idx", "ref.idx")]
    #[case(r"C:\private/sample\ref.idx", "ref.idx")]
    #[case(r"\\server\private\sample\ref.idx", "ref.idx")]
    #[case("/private/sample/", "sample")]
    #[case("/", "")]
    #[case("fast", "fast")]
    #[case("", "")]
    fn fallback_option_values_do_not_disclose_directories(
        #[case] value: &str,
        #[case] expected: &str,
    ) {
        let header = of_header(&format!("@PG\tID:m\tPN:mapper\tCL:mapper --index={value}\n"));
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("index"), Some(expected));
        let restored = AlignmentProvenance::from_json(&provenance.to_json()).unwrap();
        assert_eq!(restored.get("index"), Some(expected));
    }

    #[rstest]
    #[case("--mode fast")]
    #[case("--mode")]
    #[case("-m fast")]
    #[case("-mfast")]
    #[case("-m=fast")]
    fn ambiguous_generic_hints_do_not_conflict_with_declared_settings(#[case] option: &str) {
        let header = of_header(&format!("@PG\tID:m\tPN:mapper\tCL:mapper {option} ref.fa r.fq\n"))
            .provenance
            .unwrap();
        assert_eq!(fallback_tags(&format!("mapper {option}")), BTreeMap::new());
        let declared = overlay(
            header.clone(),
            Some(&parse_edits(&["options-inference=explicit", "mode=fast"])),
            "table",
        );
        let merged = agreed(&[named("header", Some(header)), named("table", Some(declared))])
            .unwrap()
            .unwrap();
        assert_eq!(merged.get("mode"), Some("fast"));
        assert!(incomplete_options_reason(&merged).is_some());
    }

    #[test]
    fn explicit_generic_values_including_empty_still_conflict() {
        let header = |value: &str| {
            of_header(&format!("@PG\tID:m\tPN:mapper\tCL:mapper --mode={value}\n"))
                .provenance
                .unwrap()
        };
        for value in ["accurate", ""] {
            let fast = header("fast");
            let other = header(value);
            assert_eq!(compare_provenance(&fast, &other)[0].tag, "mode");
            assert!(agreed(&[named("a", Some(fast)), named("b", Some(other))]).is_err());
        }
        let fast = resolved(Vec::new(), Some(&parse(&["mode=fast"]))).unwrap().provenance.unwrap();
        let accurate =
            resolved(Vec::new(), Some(&parse(&["mode=accurate"]))).unwrap().provenance.unwrap();
        assert_eq!(merge(&[&fast, &accurate]).unwrap_err().differences[0].tag, "mode");
    }

    #[test]
    fn arbitrary_ids_do_not_split_the_same_aligner_across_pp_chains() {
        let header = of_header(
            "@PG\tID:align-s1\tCL:bwa mem ref.fa a.fq\n\
             @PG\tID:sort-s1\tPP:align-s1\tCL:samtools sort a.bam\n\
             @PG\tID:align-s2\tCL:/opt/bin/bwa mem ref.fa b.fq\n\
             @PG\tID:sort-s2\tPP:align-s2\tCL:samtools sort b.bam\n",
        );
        assert_eq!(header.provenance.unwrap().get("aligner"), Some("bwa"));
        assert!(header.mixed.is_empty());
        let declared = |id: &str| {
            let provenance =
                of_header(&format!("@PG\tID:{id}\tCL:bwa mem ref.fa r.fq\n")).provenance.unwrap();
            overlay(provenance, Some(&parse_edits(&["options-inference=explicit"])), id)
        };
        assert!(merge(&[&declared("align-s1"), &declared("align-s2")]).is_ok());
    }

    #[test]
    fn review_unprofiled_settings_are_incomplete() {
        for mode in ["fast", "accurate"] {
            let header =
                of_header(&format!("@PG\tID:m\tPN:mapper\tCL:mapper --mode {mode} ref.fa r.fq\n"));
            assert!(incomplete_options_reason(&header.provenance.unwrap()).is_some());
        }
    }

    #[test]
    fn review_generated_mixed_does_not_inherit_literal_origin() {
        let header = of_header(
            "@PG\tID:a\tPN:mapper\tCL:mapper --mode=mixed\n\
            @PG\tID:b\tPN:mapper\tCL:mapper --mode=fast\n",
        )
        .provenance
        .unwrap();
        assert!(indeterminate_tags(&header, &header).contains("mode"));
        let literal = of_header("@PG\tID:a\tPN:mixed\tCL:mixed ref.fa\n").provenance.unwrap();
        assert!(indeterminate_tags(&literal, &literal).is_empty());
        let literal = AlignmentProvenance::from_json(&literal.to_json()).unwrap();
        assert!(indeterminate_tags(&literal, &literal).is_empty());
    }

    #[test]
    fn review_legacy_unprofiled_headers_keep_unknown_settings() {
        let legacy =
            AlignmentProvenance::from_json(r#"{"aligner":"mapper","mode":"","source":"header"}"#)
                .unwrap();
        assert!(incomplete_options_reason(&legacy).is_some());
        let edited = overlay(legacy.clone(), Some(&parse_edits(&["mode=fast"])), "legacy");
        assert!(incomplete_options_reason(&edited).is_some());
        let merged = merge(&[&legacy, &legacy]).unwrap().unwrap();
        assert!(incomplete_options_reason(&merged).is_some());
        let shared = shared_tags(&[&legacy, &legacy]);
        assert!(incomplete_options_reason(&shared).is_some());
        let partial = tags(&[("source", "tags"), ("aligner", "mapper")]);
        assert!(incomplete_options_reason(&partial).is_none());
    }

    #[test]
    fn review_literal_mixed_is_not_a_header_marker() {
        let edits = parse(&["aligner=mapper", "mode=mixed"]);
        let literal = resolved(Vec::new(), Some(&edits)).unwrap().provenance.unwrap();
        let literal = AlignmentProvenance::from_json(&literal.to_json()).unwrap();
        assert!(indeterminate_tags(&literal, &literal).is_empty());
        let generated = of_header(
            "@PG\tID:a\tPN:mapper\tCL:mapper --mode=a\n\
            @PG\tID:b\tPN:mapper\tCL:mapper --mode=b\n",
        )
        .provenance
        .unwrap();
        assert!(indeterminate_tags(&generated, &generated).contains("mode"));
        let overlaid = resolved(vec![named("header", Some(generated.clone()))], Some(&edits))
            .unwrap()
            .provenance
            .unwrap();
        assert!(!indeterminate_tags(&overlaid, &overlaid).contains("mode"));
        let merged = merge(&[&literal, &literal]).unwrap().unwrap();
        assert!(indeterminate_tags(&merged, &literal).is_empty());
        let retained = merge(&[&generated, &literal]).unwrap().unwrap();
        assert!(indeterminate_tags(&retained, &literal).contains("mode"));
    }

    #[rstest]
    #[case::generated(false, false)]
    #[case::literal_and_generated(true, false)]
    #[case::generated_and_literal(false, true)]
    #[case::literal(true, true)]
    fn mixed_training_keeps_only_literal_mixed_settings(
        #[case] first_literal: bool,
        #[case] second_literal: bool,
    ) {
        let table = |other_mode: &str, batch: &str, literal: bool| {
            let header = of_header(&format!(
                "@PG\tID:a\tPN:mapper\tVN:1\tCL:mapper --mode=fast\n\
                 @PG\tID:b\tPN:mapper\tVN:2\tCL:mapper --mode={other_mode}\n"
            ))
            .provenance
            .unwrap();
            let mut edits = vec![format!("batch={batch}")];
            if literal {
                edits.push("mode=mixed".to_string());
            }
            let AlignmentTags::Tags(edits) = AlignmentTags::parse(&edits).unwrap().unwrap() else {
                panic!("expected edits");
            };
            let table = overlay(header, Some(&edits), "table");
            AlignmentProvenance::from_json(&table.to_json()).unwrap()
        };
        let first = table("accurate", "1", first_literal);
        let second = table("sensitive", "2", second_literal);
        // A partial contributor that omits mode must not discard a shared literal setting.
        let partial = tags(&[("source", "tags"), ("aligner", "mapper")]);
        for tables in [
            [("a", Some(&first)), ("b", Some(&second)), ("c", Some(&partial))],
            [("b", Some(&second)), ("c", Some(&partial)), ("a", Some(&first))],
        ] {
            assert!(training_provenance(&tables, false).is_err());
            let shared = training_provenance(&tables, true).unwrap().unwrap();
            let restored = AlignmentProvenance::from_json(&shared.to_json()).unwrap();
            assert_eq!(restored, shared);
            assert_eq!(shared.get("mode"), (first_literal && second_literal).then_some(MIXED));
            assert_eq!(shared.get("batch"), None);
            assert_eq!(shared.get(VERSION), Some(MIXED));
            assert_eq!(shared.get(SOURCE), Some(FROM_TAGS));
            assert_eq!(shared.get(MERGED_COMPLETE), None);
            assert!(incomplete_options_reason(&shared).is_some());
            assert!(indeterminate_tags(&shared, &shared).is_empty());
            assert_eq!(literal_mixed(&shared).contains("mode"), first_literal && second_literal);
            let merged = merge(&[&restored, &partial]).unwrap().unwrap();
            assert_eq!(merged, shared);
            let edited = overlay(merged, Some(&parse_edits(&["version=2"])), "shared");
            assert_eq!(edited.get("mode"), shared.get("mode"));
            assert!(incomplete_options_reason(&edited).is_some());
            let reads = tags(&[
                ("source", "header"),
                ("aligner", "mapper"),
                ("sq-alt", "no"),
                ("mode", if first_literal && second_literal { MIXED } else { "fast" }),
            ]);
            assert!(compare_provenance(&shared, &reads).is_empty());
        }
    }

    #[test]
    fn mixed_training_drops_legacy_unmarked_mixed_settings() {
        let first =
            AlignmentProvenance::from_json(r#"{"source":"tags","mode":"mixed","batch":"1"}"#)
                .unwrap();
        let literal = resolved(Vec::new(), Some(&parse(&["mode=mixed", "batch=2"])))
            .unwrap()
            .provenance
            .unwrap();
        let shared = training_provenance(&[("a", Some(&first)), ("b", Some(&literal))], true)
            .unwrap()
            .unwrap();
        assert_eq!(shared.get("mode"), None);
        assert!(literal_mixed(&shared).is_empty());
    }

    #[test]
    fn mixed_training_preserves_normal_merge_when_tables_agree() {
        let header = of_header(
            "@PG\tID:a\tPN:mapper\tVN:1\tCL:mapper --mode=fast\n\
             @PG\tID:b\tPN:mapper\tVN:1\tCL:mapper --mode=accurate\n",
        )
        .provenance
        .unwrap();
        let partial = tags(&[("source", "tags"), ("aligner", "mapper")]);
        let expected = merge(&[&partial, &header]).unwrap().unwrap();
        for tables in [
            [("a", Some(&partial)), ("b", Some(&header))],
            [("b", Some(&header)), ("a", Some(&partial))],
        ] {
            let trained = training_provenance(&tables, true).unwrap().unwrap();
            assert_eq!(trained, expected);
            assert_eq!(trained.get(SOURCE), Some(FROM_HEADER));
            assert_eq!(trained.get("mode"), Some(MIXED));
            assert!(indeterminate_tags(&trained, &trained).contains("mode"));
            assert!(incomplete_options_reason(&trained).is_some());
        }
    }

    #[rstest]
    #[case::marker(&["options-inference=explicit"])]
    #[case::version(&["version=1"])]
    #[case::metadata(&["options-inference=explicit", "version=1", "mode!"])]
    fn metadata_only_declarations_leave_reads_unknown(#[case] values: &[&str]) {
        let edits = parse(values);
        let absent = resolved(Vec::new(), Some(&edits)).unwrap();
        assert_eq!(absent.provenance, None);
        assert_eq!(absent.from, None);
        let expected = "--alignment-tags supplies no alignment settings: give at least one \
                        compared setting, not only metadata or removals";
        assert_eq!(absent.unknown.explanation(), expected);
        let unrecorded = resolved(vec![named("table", None)], Some(&edits)).unwrap();
        assert_eq!(unrecorded.provenance, None);
        assert_eq!(unrecorded.unknown.explanation(), expected);
        let unreadable =
            resolve_reads(Some(&edits), Some(Path::new("/nonexistent/reads.bam")), Vec::new())
                .unwrap();
        assert_eq!(unreadable.provenance, None);
        assert_eq!(unreadable.unknown, UnknownReason::Unreadable);
        assert!(unreadable.header_error().is_some());
    }

    #[rstest]
    #[case("command=mem", "command", "mem")]
    #[case("aligner=mapper", "aligner", "mapper")]
    #[case("sq-alt=no", "sq-alt", "no")]
    #[case("p=", "p", "")]
    fn partial_declarations_need_not_overlap_model_settings(
        #[case] setting: &str,
        #[case] key: &str,
        #[case] value: &str,
    ) {
        let model = tags(&[("source", "tags"), ("alt-project", "primary")]);
        let edits = parse(&["options-inference=explicit", setting]);
        let reads = resolved(Vec::new(), Some(&edits)).unwrap();
        assert_eq!(reads.from, Some(ProvenanceFrom::Tags));
        let provenance = reads.provenance.unwrap();
        assert_eq!(provenance.get(key), Some(value));
        assert!(incomplete_options_reason(&provenance).is_none());
        assert!(comparable_tags(&model, &provenance).is_empty());
        assert!(compare_provenance(&model, &provenance).is_empty());
    }

    #[test]
    fn inference_only_override_still_resolves_known_header_settings() {
        let header = of_header("@PG\tID:bwa\tCL:bwa mem ref.fa reads.fq\n").provenance.unwrap();
        let reads = resolved(
            vec![named("header", Some(header))],
            Some(&parse(&["options-inference=explicit"])),
        )
        .unwrap();
        assert_eq!(reads.from, Some(ProvenanceFrom::HeaderAndTags));
        let provenance = reads.provenance.unwrap();
        assert_eq!(provenance.get(ALIGNER), Some("bwa"));
        assert_eq!(provenance.get(SOURCE), Some(FROM_HEADER));
        assert!(incomplete_options_reason(&provenance).is_none());
    }

    #[rstest]
    #[case::aligned(
        "@PG\tID:bwa\tCL:bwa mem -p ref.fa reads.fq\n\
         @PG\tID:sort\tPP:bwa\tCL:samtools sort reads.bam\n"
    )]
    #[case::conversion_only("@PG\tID:samtools\tCL:samtools sort reads.bam\n")]
    #[case::no_program("")]
    fn header_snapshot_preserves_all_raw_pg_and_sq_records(#[case] program: &str) {
        let sq = "@SQ\tSN:chr1\tLN:100\tAH:chr1_alt:1-100";
        let text = format!("@HD\tVN:1.6\n{sq}\n{program}@CO\tcomment\n");
        let header = of_header(&text);
        let mut expected = vec![sq.to_string()];
        expected.extend(program.lines().map(str::to_string));
        assert_eq!(header.header_records, expected);
        if let Some(provenance) = header.provenance {
            let json = provenance.to_json();
            assert!(!json.contains("reads.fq"));
            assert!(!json.contains("ref.fa"));
            assert!(!json.contains("chr1_alt"));
        }
    }

    #[test]
    fn fallback_honors_the_end_of_options() {
        assert_eq!(
            fallback_tags("aligner --mode=fast -- --secret=sample ref.fa r.fq"),
            map(&[("mode", "fast")])
        );
    }

    #[test]
    fn repeated_unknown_values_keep_their_boundaries() {
        let one = fallback_tags("aligner --custom=a,b ref.fa r.fq");
        let two = fallback_tags("aligner --custom=a --custom=b ref.fa r.fq");
        assert_ne!(one, two);
        assert_eq!(two.get("custom").map(String::as_str), Some("[\"a\",\"b\"]"));
        let literal = fallback_tags(r#"aligner --custom=["a","b"] ref.fa r.fq"#);
        assert_ne!(literal, two);
        let literal_then_c = fallback_tags(r#"aligner --custom=["a","b"] --custom=c ref.fa r.fq"#);
        let three = fallback_tags("aligner --custom=a --custom=b --custom=c ref.fa r.fq");
        assert_ne!(literal_then_c, three);
    }

    #[test]
    fn accumulating_values_preserve_explicit_empty_boundaries() {
        assert_eq!(
            fallback_tags("aligner --custom= --custom=a ref.fa r.fq")
                .get("custom")
                .map(String::as_str),
            Some("[\"\",\"a\"]")
        );
        assert_eq!(
            fallback_tags("aligner --custom=a --custom= ref.fa r.fq")
                .get("custom")
                .map(String::as_str),
            Some("[\"a\",\"\"]")
        );
    }

    #[test]
    fn merge_stops_after_the_first_conflicting_comparison() {
        let a = tags(&[("x", "a")]);
        let b = tags(&[("x", "b")]);
        let c = tags(&[("x", "c")]);
        COMPARISONS.set(0);
        assert!(merge(&[&a, &b, &c]).is_err());
        assert_eq!(COMPARISONS.get(), 1);
        COMPARISONS.set(0);
        assert_eq!(conflicts(&[&a, &b, &c]).len(), 3);
        assert_eq!(COMPARISONS.get(), 3);
        COMPARISONS.set(0);
        assert!(merge(&[&a, &a, &a]).is_ok());
        assert_eq!(COMPARISONS.get(), 3);
    }

    #[test]
    fn conflict_pairs_keep_input_order() {
        let a = tags(&[("x", "a")]);
        let b = tags(&[("x", "b")]);
        let c = tags(&[("x", "c")]);
        let all = conflicts(&[&a, &b, &c]);
        assert_eq!(
            all.iter().map(|c| (c.first, c.second)).collect::<Vec<_>>(),
            [(0, 1), (0, 2), (1, 2)]
        );
        assert_eq!(merge(&[&a, &b, &c]).unwrap_err(), all[0]);
    }

    #[test]
    fn alignment_tags_parse() {
        assert_eq!(AlignmentTags::parse::<&str>(&[]).unwrap(), None);
        assert_eq!(AlignmentTags::parse(&["none"]).unwrap(), Some(AlignmentTags::None));
        assert_eq!(
            parse(&["alt-project=primary", "Y=", "K!", "version=1"]),
            AlignmentTags::Tags(TagEdits {
                set: map(&[("alt-project", "primary"), ("Y", ""), ("version", "1")]),
                removed: ["K".to_string()].into_iter().collect(),
            })
        );
        for (bad, message) in [
            (&["a=1", "a=2"][..], "gives a more than once"),
            (&["a=1", "a!"][..], "gives a more than once"),
            (&["a b=1"][..], "keys are letters"),
            (&["!"][..], "keys are letters"),
            (&["a!=1"][..], "keys are letters"),
            (&["x"][..], "must be KEY=VALUE, KEY! or none"),
            (&["none", "a=1"][..], "none cannot be given"),
            (&["source=tags"][..], "cannot give source"),
            (&["source!"][..], "cannot give source"),
        ] {
            let error = AlignmentTags::parse(bad).unwrap_err().to_string();
            assert!(error.contains(message), "{bad:?}: {error}");
        }
    }

    /// The tags set replace and add, the tags removed are removed, over what was detected; alone,
    /// the tags set are the provenance, compared on those tags only.
    #[test]
    fn alignment_tags_apply_over_the_detected_tags() {
        let given = parse(&["alt-project=primary", "Y=", "M!"]);
        let detected = tags(&[
            ("aligner", "bwa-mem3"),
            ("alt-project", "keep-alt"),
            ("M", ""),
            ("source", "header"),
        ]);
        let reads = resolved(vec![named("h", Some(detected.clone()))], Some(&given)).unwrap();
        assert_eq!(
            reads.provenance.unwrap(),
            tags(&[
                ("aligner", "bwa-mem3"),
                ("alt-project", "primary"),
                ("source", "header"),
                ("Y", "")
            ])
        );
        assert_eq!(reads.from, Some(ProvenanceFrom::HeaderAndTags));

        // Alone (a removal has nothing to remove from).
        let reads = resolved(Vec::new(), Some(&parse(&["a=1", "b!"]))).unwrap();
        let provenance = reads.provenance.unwrap();
        assert_eq!(provenance, tags(&[("a", "1"), ("source", "tags")]));
        assert!(provenance.is_tags_only());
        assert_eq!(reads.from.map(ProvenanceFrom::as_str), Some("tags"));
        let reads = resolved(Vec::new(), Some(&parse(&["b!"]))).unwrap();
        assert_eq!(reads.provenance, None);
        assert_eq!(
            reads.unknown.explanation(),
            "--alignment-tags supplies no alignment settings: give at least one compared setting, \
             not only metadata or removals"
        );

        let none =
            resolved(vec![named("h", Some(detected.clone()))], Some(&AlignmentTags::None)).unwrap();
        assert_eq!((none.provenance, none.unknown), (None, UnknownReason::TagsNone));
        let plain = resolved(vec![named("h", Some(detected.clone()))], None).unwrap();
        assert_eq!((plain.provenance, plain.from), (Some(detected), Some(ProvenanceFrom::Header)));
        let unrecorded = resolved(vec![named("h", None)], None).unwrap();
        assert_eq!((unrecorded.provenance, unrecorded.unknown), (None, UnknownReason::NotRecorded));
        let absent = resolved(Vec::new(), None).unwrap();
        assert_eq!(absent, ReadsAlignment::default());
        assert_eq!(
            absent.unknown.explanation(),
            "neither --alignments nor --alignment-tags was given: pass either"
        );
    }

    /// With `none`, no source is read; a header that cannot be read leaves the reads' alignment
    /// unknown (unless the tags give it), with why.
    #[test]
    fn the_resolver_reads_what_it_needs() {
        let unread: LazySource<'_> = Box::new(|| panic!("a source was read with none"));
        let missing = Path::new("/nonexistent/reads.bam");
        let none = resolve_reads(
            Some(&AlignmentTags::None),
            Some(missing),
            vec![("t".to_string(), unread)],
        )
        .unwrap();
        assert_eq!((none.unknown, none.header), (UnknownReason::TagsNone, None));

        let unreadable = resolve_reads(None, Some(missing), Vec::new()).unwrap();
        assert_eq!(unreadable.unknown, UnknownReason::Unreadable);
        let error = unreadable.header_error().unwrap();
        assert!(
            error.starts_with("failed to read the header of /nonexistent/reads.bam"),
            "{error}"
        );
        let tagged = resolve_reads(Some(&parse(&["a=1"])), Some(missing), Vec::new()).unwrap();
        assert_eq!(tagged.provenance, Some(tags(&[("a", "1"), ("source", "tags")])));
        assert!(tagged.header_error().is_some());
    }

    /// The tags are applied over each source before the sources must agree.
    #[test]
    fn alignment_tags_apply_over_each_source() {
        let header =
            tags(&[("aligner", "bwa-mem3"), ("alt-project", "primary"), ("source", "header")]);
        let table =
            tags(&[("aligner", "bwa-mem3"), ("alt-project", "keep-alt"), ("source", "header")]);
        let sources = || vec![named("x", Some(table.clone())), named("y", Some(header.clone()))];
        let error = resolved(sources(), None).unwrap_err().to_string();
        assert!(
            error.starts_with(
                "x and y were aligned differently: alt-project: x keep-alt, y primary. They \
                 should be of the same reads: measure the tables again from them, pass \
                 --alignment-tags"
            ),
            "{error}"
        );
        let given = parse(&["alt-project=primary"]);
        assert_eq!(resolved(sources(), Some(&given)).unwrap().provenance, Some(header.clone()));
        let removed = parse(&["alt-project!"]);
        let provenance = resolved(sources(), Some(&removed)).unwrap().provenance.unwrap();
        assert_eq!(provenance, tags(&[("aligner", "bwa-mem3"), ("source", "header")]));
    }

    #[test]
    fn provenances_are_compared_on_all_tags_but_the_version_and_source() {
        let models = tags(&[("aligner", "bwa-mem3"), ("alt-project", "primary"), ("version", "1")]);
        assert!(compare_provenance(&models, &models).is_empty());
        let reads = tags(&[
            ("aligner", "bwa-mem3"),
            ("sq-alt", "no"),
            ("version", "2"),
            ("source", "header"),
        ]);
        let differences = compare_provenance(&models, &reads);
        assert_eq!(
            describe_differences(&differences, "models", "reads"),
            "alt-project: models primary, reads not set; sq-alt: models not set, reads no"
        );
    }

    #[test]
    fn the_provenance_round_trips_through_json() {
        let provenance: AlignmentProvenance =
            [("aligner", "bwa-mem3"), ("Y", "")].into_iter().collect();
        let json = provenance.to_json();
        assert_eq!(json, r#"{"Y":"","aligner":"bwa-mem3"}"#);
        assert_eq!(AlignmentProvenance::from_json(&json).unwrap(), provenance);
        assert!(AlignmentProvenance::from_json("[1]").is_err());
        assert_eq!(provenance.to_string(), "Y= aligner=bwa-mem3");
    }

    #[rstest]
    #[case(r#"{"aligner":"bwa","aligner":"other"}"#)]
    #[case(r#"{"aligner":"bwa","aligner":"bwa"}"#)]
    #[case(r#"{"aligner":"bwa","\u0061ligner":"bwa"}"#)]
    #[case(r#"{"custom":"first","custom":"second"}"#)]
    #[case(r#"{"":"","":""}"#)]
    fn duplicate_json_keys_are_rejected(#[case] json: &str) {
        let error = format!("{:#}", AlignmentProvenance::from_json(json).unwrap_err());
        assert!(error.contains("duplicate"), "{error}");
        // Metadata in feature tables and models uses Deserialize directly.
        assert!(serde_json::from_str::<AlignmentProvenance>(json).is_err());
    }

    #[test]
    fn unique_json_keys_keep_their_decoded_values() {
        let expected: AlignmentProvenance =
            [("aligner", "bwa"), ("Y", ""), ("", "custom")].into_iter().collect();
        let json = r#"{"\u0061ligner":"bwa","Y":"","":"custom"}"#;
        assert_eq!(AlignmentProvenance::from_json(json).unwrap(), expected);
        assert_eq!(serde_json::from_str::<AlignmentProvenance>(json).unwrap(), expected);
        assert_eq!(AlignmentProvenance::from_json("{}").unwrap(), AlignmentProvenance::default());
        for invalid in ["[]", "null", r#"{"aligner":1}"#, r#"{"aligner":{}}"#, "{} {}"] {
            assert!(AlignmentProvenance::from_json(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn sources_of_one_samples_reads_must_agree() {
        let a = tags(&[("aligner", "bwa-mem3"), ("version", "1")]);
        let b = tags(&[("aligner", "bwa-mem3"), ("version", "2")]);
        let sources = [named("x", Some(a.clone())), named("y", None), named("z", Some(b))];
        assert_eq!(agreed(&sources).unwrap(), Some(a));
        let other = [
            named("x", Some(tags(&[("aligner", "bwa")]))),
            named("y", Some(tags(&[("aligner", "bwa-mem3")]))),
        ];
        let error = agreed(&other).unwrap_err().to_string();
        assert!(error.starts_with("x and y were aligned differently: aligner: x bwa, y bwa-mem3."));
        assert_eq!(agreed(&[named("x", None)]).unwrap(), None);
    }

    /// Merging does not depend on the order: every pair is compared, so a provenance of tags alone
    /// that agrees with two others does not hide their difference.
    #[test]
    fn merging_compares_every_pair() {
        let given = tags(&[("aligner", "bwa-mem3"), ("source", "tags")]);
        let primary =
            tags(&[("aligner", "bwa-mem3"), ("alt-project", "primary"), ("source", "header")]);
        let keep =
            tags(&[("aligner", "bwa-mem3"), ("alt-project", "keep-alt"), ("source", "header")]);
        for order in
            [[&given, &primary, &keep], [&primary, &given, &keep], [&keep, &primary, &given]]
        {
            let conflict = merge(&order).unwrap_err();
            assert_eq!(conflict.differences.len(), 1, "{conflict:?}");
        }
        // Two of tags alone that record different tags merge into their union.
        let a = tags(&[("alt-project", "primary"), ("source", "tags")]);
        let b = tags(&[("Y", ""), ("source", "tags")]);
        assert_eq!(
            merge(&[&a, &b]).unwrap(),
            Some(tags(&[("alt-project", "primary"), ("source", "tags"), ("Y", "")]))
        );
        // With one from a header, the union is from a header, with its version.
        let header =
            tags(&[("alt-project", "primary"), ("source", "header"), ("version", "2"), ("Y", "")]);
        let versioned = tags(&[("source", "tags"), ("version", "1"), ("Y", "")]);
        assert_eq!(merge(&[&versioned, &a, &header]).unwrap(), Some(header));
        assert_eq!(merge(&[]).unwrap(), None);
    }

    #[test]
    fn training_tables_must_agree_unless_mixed_alignments() {
        let primary =
            tags(&[("aligner", "bwa-mem3"), ("alt-project", "primary"), ("source", "header")]);
        let keep =
            tags(&[("aligner", "bwa-mem3"), ("alt-project", "keep-alt"), ("source", "header")]);
        assert_eq!(training_provenance(&[("A", None), ("B", None)], false).unwrap(), None);
        assert_eq!(
            training_provenance(&[("A", None), ("B", Some(&primary))], false).unwrap(),
            Some(primary.clone())
        );
        let conflict = [("A", Some(&primary)), ("B", Some(&keep))];
        let error = training_provenance(&conflict, false).unwrap_err().to_string();
        assert!(
            error.starts_with(
                "the tables of A and B are of reads aligned differently: alt-project: A primary, \
                 B keep-alt"
            ),
            "{error}"
        );
        // A mixed model records the tags they share, and is compared on those only.
        let mixed = training_provenance(&conflict, true).unwrap().unwrap();
        assert_eq!(
            mixed,
            tags(&[("aligner", "bwa-mem3"), ("source", "tags"), ("options-inference", "explicit")])
        );
        assert!(compare_provenance(&mixed, &primary).is_empty());
        assert!(compare_provenance(&mixed, &keep).is_empty());
        // A table of tags alone agrees with one read from a header on the tags it gives; the
        // model records the header's.
        let given = tags(&[("alt-project", "primary"), ("source", "tags")]);
        assert_eq!(
            training_provenance(&[("A", Some(&given)), ("B", Some(&primary))], false).unwrap(),
            Some(primary.clone())
        );
        // Tables of tags alone that conflict with each other are an error, though each agrees
        // with a third.
        let other = tags(&[("alt-project", "keep-alt"), ("source", "tags")]);
        let aligner = tags(&[("aligner", "bwa-mem3"), ("source", "tags")]);
        let tables = [("A", Some(&aligner)), ("B", Some(&given)), ("C", Some(&other))];
        let error = training_provenance(&tables, false).unwrap_err().to_string();
        assert!(error.starts_with("the tables of B and C"), "{error}");
        // With --mixed-alignments, a table of tags alone that does not record a tag does not drop
        // it; one from a header does.
        let shared = training_provenance(
            &[("A", Some(&aligner)), ("B", Some(&given)), ("C", Some(&other))],
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(shared, tags(&[("aligner", "bwa-mem3"), ("source", "tags")]));
        let with_header = training_provenance(
            &[
                ("A", Some(&given)),
                ("B", Some(&other)),
                (
                    "C",
                    Some(&tags(&[
                        ("Y", ""),
                        ("source", "header"),
                        ("options-inference", "explicit"),
                    ])),
                ),
            ],
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            with_header,
            tags(&[("source", "tags"), ("Y", ""), ("options-inference", "explicit")])
        );
        let keep_y = tags(&[
            ("aligner", "bwa-mem3"),
            ("alt-project", "keep-alt"),
            ("source", "header"),
            ("Y", ""),
        ]);
        let shared =
            training_provenance(&[("A", Some(&primary)), ("B", Some(&keep_y))], true).unwrap();
        assert_eq!(
            shared,
            Some(tags(&[
                ("aligner", "bwa-mem3"),
                ("source", "tags"),
                ("options-inference", "explicit")
            ]))
        );
    }

    /// A provenance given by --alignment-tags alone is compared on the tags it gives only; the
    /// source is not compared.
    #[test]
    fn tags_alone_are_compared_on_the_tags_given() {
        let header = resolved(
            vec![named("h", of_header(PIPELINE).provenance)],
            Some(&parse(&["options-inference=explicit", "Y="])),
        )
        .unwrap()
        .provenance
        .unwrap();
        let given = tags(&[("aligner", "bwa-mem3"), ("source", "tags"), ("Y", "")]);
        assert!(compare_provenance(&given, &header).is_empty());
        assert!(compare_provenance(&header, &given).is_empty());
        let other = tags(&[("aligner", "bwa-mem2"), ("source", "tags")]);
        let differences = compare_provenance(&header, &other);
        assert_eq!(
            describe_differences(&differences, "models", "reads"),
            "aligner: models bwa-mem3, reads bwa-mem2"
        );
        // A complete header-derived declaration establishes that an omitted tag is absent.
        let mut fewer = header.clone();
        fewer.0.remove("Y");
        assert_eq!(compare_provenance(&header, &fewer).len(), 1);
    }

    /// PP chains that loop have no root: the first record stands in.
    #[test]
    fn a_pp_cycle_reads_the_first_record() {
        let header = of_header(
            "@PG\tID:a\tPN:aligner-a\tPP:b\tCL:aligner-a -x 1 ref.fa r.fq\n\
             @PG\tID:b\tPN:tool-b\tPP:a\tCL:tool-b\n",
        );
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("aligner"), Some("aligner-a"));
        assert_eq!(provenance.get("x"), None);
        assert_eq!(header.unprofiled.as_deref(), Some("aligner-a"));
    }

    /// A known converter alongside an unknown aligner is not an independent alignment history.
    #[test]
    fn a_conversion_root_does_not_make_an_unknown_aligner_mixed() {
        let header = of_header(
            "@PG\tID:fastq-to-bam\tPN:fastq-to-bam\tCL:fastq-to-bam -i r.fq -o u.bam\n\
             @PG\tID:mapper\tPN:mapper\tCL:mapper -ax sr ref.fa r.fq\n",
        );
        let provenance = header.provenance.unwrap();
        assert_eq!(provenance.get("aligner"), Some("mapper"));
        assert_eq!(provenance.get("ax"), None);
        assert_eq!(provenance.get("source"), Some("header"));
        assert!(header.mixed.is_empty());
    }
}
