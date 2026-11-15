//! A strict applier of the unified diffs in `patches/`, used by `build.rs` (and tested by the
//! crate's unit tests, which include this file).
//!
//! A patch must apply exactly: anything it does not understand, or a hunk whose lines do not match
//! the source at its stated position, is an error, so a patch is never applied in part or in the
//! wrong place.

/// The changes a unified diff makes to one file.
#[derive(Debug, PartialEq)]
pub(crate) struct FilePatch {
    /// The file's path, relative to the source root.
    pub(crate) path: String,
    hunks: Vec<Hunk>,
}

/// One hunk of a unified diff.
#[derive(Debug, PartialEq)]
struct Hunk {
    /// The 1-based line in the original file at which the hunk starts.
    old_start: usize,
    /// The hunk's lines, each with its marker: `' '` (context), `'-'` (removed) or `'+'` (added).
    lines: Vec<(char, String)>,
}

impl FilePatch {
    /// Applies the hunks to `original`, which must match every hunk's context and removed lines
    /// exactly, at the hunk's stated line.
    pub(crate) fn apply(&self, original: &str) -> Result<String, String> {
        let old: Vec<&str> = original.lines().collect();
        let mut new: Vec<&str> = Vec::with_capacity(old.len());
        let mut next = 0; // the index of the next original line to copy
        for hunk in &self.hunks {
            // A hunk that only adds lines states the line before it (0 at the start of the file).
            let removes = hunk.lines.iter().any(|(marker, _)| *marker != '+');
            let start = if removes { hunk.old_start.checked_sub(1) } else { Some(hunk.old_start) };
            let start = start.ok_or_else(|| {
                format!("hunk at line {} has context or removed lines", hunk.old_start)
            })?;
            if start < next || start > old.len() {
                return Err(format!(
                    "hunk at line {} is out of order or out of range",
                    hunk.old_start
                ));
            }
            new.extend_from_slice(&old[next..start]);
            next = start;
            for (marker, line) in &hunk.lines {
                if *marker != '+' {
                    if old.get(next) != Some(&line.as_str()) {
                        return Err(format!(
                            "hunk at line {} does not match the source at line {}",
                            hunk.old_start,
                            next + 1
                        ));
                    }
                    next += 1;
                }
                if *marker != '-' {
                    new.push(line);
                }
            }
        }
        new.extend_from_slice(&old[next..]);
        let mut patched = new.join("\n");
        if original.ends_with('\n') {
            patched.push('\n');
        }
        Ok(patched)
    }
}

/// Parses a unified diff, as `git diff` writes it. Text before the first file's diff (e.g. a
/// description) and after the last hunk of a file (e.g. `git format-patch`'s signature) is
/// ignored, but a line that looks like part of a hunk there is an error (a hunk header that
/// states fewer lines than the hunk has), as is a `diff --git` with no text diff (a binary,
/// mode-only or rename-only change) or a file diffed twice.
pub(crate) fn parse_patch(text: &str) -> Result<Vec<FilePatch>, String> {
    let mut files: Vec<FilePatch> = Vec::new();
    let mut lines = text.lines().peekable();
    // A `diff --git` line not yet followed by its `---` / `+++` lines.
    let mut pending: Option<&str> = None;
    while let Some(line) = lines.next() {
        if line.starts_with("diff --git ") {
            if let Some(diff) = pending.replace(line) {
                return Err(format!("`{diff}` has no text diff, which is not supported"));
            }
            continue;
        }
        if !line.starts_with("--- ") {
            continue;
        }
        pending = None;
        let path = lines
            .next()
            .and_then(|line| line.strip_prefix("+++ b/"))
            .ok_or_else(|| format!("`{line}` is not followed by a `+++ b/<path>` line"))?;
        let path = path.split('\t').next().unwrap_or(path).to_string();
        if files.iter().any(|file| file.path == path) {
            return Err(format!("{path} is diffed more than once"));
        }
        let mut hunks = Vec::new();
        while let Some(header) = lines.next_if(|line| line.starts_with("@@ ")) {
            let (old_start, old_count, new_count) = parse_hunk_header(header)?;
            let mut hunk = Hunk { old_start, lines: Vec::new() };
            let (mut old_seen, mut new_seen) = (0, 0);
            while old_seen < old_count || new_seen < new_count {
                let line = lines.next().ok_or_else(|| format!("hunk `{header}` is truncated"))?;
                let marker = line.chars().next().unwrap_or(' ');
                if !matches!(marker, ' ' | '-' | '+') {
                    return Err(format!("hunk `{header}` has an unexpected line: `{line}`"));
                }
                old_seen += usize::from(marker != '+');
                new_seen += usize::from(marker != '-');
                hunk.lines.push((marker, line.get(1..).unwrap_or("").to_string()));
            }
            if old_seen != old_count || new_seen != new_count {
                return Err(format!("hunk `{header}` does not have the lines it states"));
            }
            match lines.peek() {
                Some(line) if line.starts_with('\\') => {
                    return Err(format!(
                        "hunk `{header}`: files without a final newline are not supported"
                    ));
                }
                Some(line) if is_uncounted_hunk_line(line) => {
                    return Err(format!(
                        "hunk `{header}` has more lines than it states: `{line}` follows it"
                    ));
                }
                _ => {}
            }
            hunks.push(hunk);
        }
        if hunks.is_empty() {
            return Err(format!("the diff of {path} has no hunks"));
        }
        files.push(FilePatch { path, hunks });
    }
    if let Some(diff) = pending {
        return Err(format!("`{diff}` has no text diff, which is not supported"));
    }
    if files.is_empty() {
        return Err("no file diffs found".to_string());
    }
    Ok(files)
}

/// Whether a line after a hunk's stated lines would be a line of the hunk: one that starts with a
/// hunk marker, other than the next file's `--- ` line and `git format-patch`'s `-- ` signature.
fn is_uncounted_hunk_line(line: &str) -> bool {
    let signature = line == "--" || line == "-- ";
    matches!(line.chars().next(), Some(' ' | '+' | '-')) && !line.starts_with("--- ") && !signature
}

/// The original start line and the original and new line counts of a hunk header,
/// `@@ -<start>[,<count>] +<start>[,<count>] @@`.
fn parse_hunk_header(header: &str) -> Result<(usize, usize, usize), String> {
    let invalid = || format!("invalid hunk header: `{header}`");
    let mut fields = header.split_whitespace().skip(1);
    let range = |field: Option<&str>, sign: char| -> Result<(usize, usize), String> {
        let field = field.and_then(|f| f.strip_prefix(sign)).ok_or_else(invalid)?;
        let (start, count) = field.split_once(',').unwrap_or((field, "1"));
        Ok((start.parse().map_err(|_| invalid())?, count.parse().map_err(|_| invalid())?))
    };
    let (old_start, old_count) = range(fields.next(), '-')?;
    let (_, new_count) = range(fields.next(), '+')?;
    Ok((old_start, old_count, new_count))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const SOURCE: &str = "one\ntwo\nthree\nfour\n";

    /// Parses `patch`, which must hold one file's diff, and applies it to [`SOURCE`].
    fn apply(patch: &str) -> Result<String, String> {
        let files = parse_patch(patch)?;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/a.cc");
        files[0].apply(SOURCE)
    }

    #[test]
    fn applies_a_git_diff() {
        let patch = "Describes the change.\n\ndiff --git a/src/a.cc b/src/a.cc\nindex 1..2 100644\n\
                     --- a/src/a.cc\n+++ b/src/a.cc\n@@ -2,2 +2,2 @@\n two\n-three\n+THREE\n\
                     -- \n2.50.0\n";
        assert_eq!(apply(patch), Ok("one\ntwo\nTHREE\nfour\n".to_string()));
    }

    #[rstest]
    #[case::adds_at_the_start("--- a/src/a.cc\n+++ b/src/a.cc\n@@ -0,0 +1 @@\n+zero\n", "zero\n")]
    #[case::adds_after_a_line(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -1,0 +2 @@\n+one and a half\n",
        "one\none and a half\n"
    )]
    fn applies_hunks_that_only_add(#[case] patch: &str, #[case] start: &str) {
        assert!(apply(patch).unwrap().starts_with(start));
    }

    #[rstest]
    #[case::context_at_line_zero(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -0,1 +0,2 @@\n one\n+two\n",
        "hunk at line 0 has context or removed lines"
    )]
    #[case::mismatch(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -2 +2 @@\n-TWO\n+2\n",
        "hunk at line 2 does not match the source at line 2"
    )]
    #[case::out_of_range(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -9 +9 @@\n-nine\n+9\n",
        "hunk at line 9 is out of order or out of range"
    )]
    fn rejects_hunks_that_do_not_apply(#[case] patch: &str, #[case] message: &str) {
        assert_eq!(apply(patch), Err(message.to_string()));
    }

    #[rstest]
    #[case::extra_line(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -2 +2 @@\n-two\n+2\n+2.5\n",
        "hunk `@@ -2 +2 @@` has more lines than it states: `+2.5` follows it"
    )]
    #[case::extra_removed_line(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -2 +2 @@\n-two\n+2\n-three\n",
        "hunk `@@ -2 +2 @@` has more lines than it states: `-three` follows it"
    )]
    #[case::truncated(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -2,2 +2,2 @@\n two\n",
        "hunk `@@ -2,2 +2,2 @@` is truncated"
    )]
    #[case::no_final_newline(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -4 +4 @@\n-four\n+4\n\\ No newline at end of file\n",
        "hunk `@@ -4 +4 @@`: files without a final newline are not supported"
    )]
    #[case::binary(
        "diff --git a/src/a.o b/src/a.o\nGIT binary patch\nliteral 1\n",
        "`diff --git a/src/a.o b/src/a.o` has no text diff, which is not supported"
    )]
    #[case::rename_then_diff(
        "diff --git a/x b/y\nrename from x\nrename to y\ndiff --git a/src/a.cc b/src/a.cc\n\
         --- a/src/a.cc\n+++ b/src/a.cc\n@@ -2 +2 @@\n-two\n+2\n",
        "`diff --git a/x b/y` has no text diff, which is not supported"
    )]
    #[case::deleted_file(
        "--- a/src/a.cc\n+++ /dev/null\n@@ -1 +0,0 @@\n-one\n",
        "`--- a/src/a.cc` is not followed by a `+++ b/<path>` line"
    )]
    #[case::twice(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -2 +2 @@\n-two\n+2\n\
         --- a/src/a.cc\n+++ b/src/a.cc\n@@ -3 +3 @@\n-three\n+3\n",
        "src/a.cc is diffed more than once"
    )]
    #[case::no_hunks("--- a/src/a.cc\n+++ b/src/a.cc\n", "the diff of src/a.cc has no hunks")]
    #[case::no_diff("just text\n", "no file diffs found")]
    #[case::bad_header(
        "--- a/src/a.cc\n+++ b/src/a.cc\n@@ -x +2 @@\n",
        "invalid hunk header: `@@ -x +2 @@`"
    )]
    fn rejects_patches_it_cannot_apply_exactly(#[case] patch: &str, #[case] message: &str) {
        assert_eq!(parse_patch(patch).err().as_deref(), Some(message));
    }
}
