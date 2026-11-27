# Alignment provenance

Models trained on reads aligned one way can score calls from reads aligned another way worse. With
bwa-mem3, for example, ALT projection (`--alt-project primary`) and no projection want opposite
actions at segmental duplications on calls that look alike: on HG002's projection-sensitive sites,
models trained on projected alignments raised SNP F1 from 0.976 to 0.987 on projected alignments,
and lowered it from 0.933 to 0.924 on unprojected ones.

So winnowr records how the reads were aligned, its *alignment provenance*, from the reads to the
model, and `score` compares the models' with the reads it scores.

## What is recorded

The provenance is a set of tags, each a name and a value:

| tag | from | compared |
|-----|------|----------|
| `aligner` | the aligner's `@PG` `PN` or command/ID fallback (below), lower-cased (e.g. `bwa-mem3`), not the profile's name | yes |
| `version` | its `VN` | no: recorded and reported only |
| `source` | where the provenance came from: `header` (the reads' header, perhaps with `--alignment-tags` over it) or `tags` (`--alignment-tags` alone, or a `--mixed-alignments` model's shared tags) | no |
| `command` | explicitly supplied subcommand (e.g. `mem`) | yes |
| one per option | explicit tags, or conservative parsing of an unknown tool's `CL` (below): e.g. `alt-project=primary`, `Y=` for a flag without a value | yes |
| `options-inference` | `incomplete` for BWA/minimap2, unprofiled fallback parsing, or a missing/empty `CL`; `explicit` when the user declares the intended settings | checked separately: unresolved inference warns, or fails strict checking |
| `winnowr-literal-mixed-tags` | internal JSON set of tags whose `mixed` value is literal rather than a disagreement marker | no |
| `winnowr-merged-complete` | internal merge bookkeeping: `yes` when merged provenance included a complete source and stays incomplete; not accepted by `--alignment-tags` | no |
| `sq-alt` | `yes` if an `@SQ` of the header has an `AH` tag (ALT contigs), else `no` | yes |
| any | `--alignment-tags` | yes |

The aligner's version is not compared: upgrades between training and scoring are routine, and a
version that changes behaviour usually shows as an option worth its own tag.

The raw command line is not recorded (it names files and samples, and models are shared); only the
normalized tags are.

### Finding the aligner in the header

The aligner is found among the header's `@PG` records by its program: `PN`, else a recognized
aligner executable in the first word of `CL`, else `ID` without a suffix `samtools merge` adds to a
repeated id (`-` and eight hexadecimal digits) or a `.N` suffix, else the first word of `CL`.
Executable paths are reduced to file names. This keeps arbitrary record IDs from making the same
aligner appear different, while retaining recognized IDs when `CL` does not identify the aligner.
Its command line normally survives sorting, duplicate marking, merging and CRAM conversion:
CRAM stores the header as text.

- Each terminal `PP` chain is checked separately, including branches sharing an ancestor and
  chains without a profiled aligner alongside chains with one. The latest aligner with a profile
  (below), by program or command, is used; downstream tools are not extra aligners. A shared older
  aligner may be stale on a re-aligned branch but still describe reads on another branch. Shared
  selected records are included only once. Mutually ancestral aligners in a terminal cycle all
  stand in unless followed on that chain by a newer aligner outside the cycle.
- Known conversion/postprocessing programs (`samtools`, `fastq-to-bam`) remain in `PP` chains but cannot stand in as aligners. A chain without a profiled aligner contributes its earliest eligible record, or its first eligible record if cyclic, read without a profile (logged). This preserves fallback for unknown aligners, including when conversion precedes them. An unknown program may still be preprocessing; when programs follow it, the warning says so. Check its settings with `--alignment-tags`.
- After `samtools merge`, there may be several aligner records. Their retained header facts and unprofiled option tags are compared: where they disagree, the tag's value is `mixed`, with a warning naming the tags. A record whose options are incomplete (a lossy aligner, or no `CL`) does not disagree about an option it does not record: the others' value is kept, and the result stays incomplete. Lossy BWA and minimap2 option settings remain incomplete rather than being guessed.
- A header without an eligible aligner record gives no provenance (unknown), even if it records conversion or postprocessing programs.

### BWA and minimap2 command lines cannot establish option settings

BWA, bwa-mem2, bwa-mem3 and bwa-mem3-meth join command-line arguments with spaces without reversible quoting. A single output filename `result -Y` following `-o` produces the same `CL` text as the output filename `result` followed by a real `-Y` flag. Shell-style splitting cannot recover boundaries that were never written. This limitation affects ordinary unquoted BWA command lines too, not only visibly unusual filenames. minimap2 writes its `CL` the same way: `-o 'out --secondary=no'` and `-o out --secondary=no` give the same text, though only the second disables secondary alignments.

For these aligners, winnowr retains the header's aligner identity, version and `sq-alt`, but does **not** infer `command` or option tags from `CL`. It records `options-inference=incomplete`, which survives evidence tables, feature tables, model training and provenance merges. Any unresolved input keeps a merged or `--mixed-alignments` result incomplete, even if another input has explicit settings. Missing tags on an incomplete source mean unknown settings, not proof that those options were absent; its recorded facts and supplied values are still compared. Equal incomplete markers do not establish agreement. Default checking warns; `--strict-provenance` requires an explicit declaration and intended settings for the affected reads or models.

A complete contributor's known settings, including absent options, remain constraints when merged with incomplete sources. Grouping the sources or saving and reloading the result does not erase those constraints, and model/read mismatches are reported alongside incomplete-settings problems. Such a merge stores `winnowr-merged-complete=yes` as internal bookkeeping in the existing string-valued tag object; it is not an alignment setting or a user declaration and cannot be set or removed with `--alignment-tags`. It does not resolve incomplete inference. An explicit declaration replaces this bookkeeping, while a deliberate `--mixed-alignments` intersection discards it and constrains only its shared tags.

This rule also covers legacy tables and models that predate `options-inference` and either record an aligner or declare `source=header`, even without an `aligner` tag. An absent `source` means `header`, as before; such header-derived provenance is incomplete until explicitly declared, even when it already contains inferred option tags. Deserialization does not rewrite the stored object. Merging or applying unrelated tag edits preserves the unresolved state, so changing or removing the aligner tag cannot silently clear it. Provenance with `source=tags` remains a valid partial user declaration unless it already carries an incomplete marker; `options-inference=explicit` resolves either new or legacy incompleteness.

Supply settings deliberately, for example:

```console
--alignment-tags options-inference=explicit command=mem p= Y= K=100000000 alt-project=primary
```

`options-inference=explicit` is your assertion that the supplied tags describe the intended settings, not proof that winnowr recovered the original arguments. Review the alignment command or pipeline configuration before making it. On header-derived provenance (the reads' header, or tables measured from it), option tags absent from the result are deliberately absent, and are compared as absent. A declaration given with no header or table provenance (`source=tags`) is partial: only the tags it records are compared, so an option it omits is not checked even when a model records it (declaring only `aligner=bwa` does not check `p`). Existing tags, including legacy inferred values, remain unless explicitly set or removed: review them alongside the declaration. The declaration does not cause the original `CL` to be parsed. If a model carries incomplete inference, establish its training settings and regenerate its provenance from explicitly tagged tables; overriding only scoring reads cannot certify that model.

Smart pairing (`-p`) changes alignment semantics, not just input layout: include `p=` when used. BWA accepts paired values separated by punctuation, including `-O6:9` and `-O6,9`; a later scalar resets both components (`-O6,9 -O7` gives `7,7`). Supply one consistent effective representation for `O`, `E`, `L` and `h`, such as `O=6,9` or `O=7,7`. Explicit tag values are literal declarations, not another argv parser.

Batch size affects insert-size estimation. When declaring effective BWA `mem` settings, record the final positive fixed `K` and omit `t`. Without a positive fixed `K`, omit `K` and supply the effective thread count as `t` (default one, at least one); the verified implementations scale default batches by threads, with variant-specific caps. Do not guess an implicit `K` from a family name or version. A final nonpositive `-K` restores thread-dependent batching. Threads of non-`mem` commands are not batch dependencies. These are effective-setting conventions for the explicit tags, not settings recovered from lossy headers.

### Unknown tools

The fallback for tools without a profile remains conservative: the command line is split into arguments as the shell splits them (quotes and backslash escapes keep whitespace inside an argument), and only explicit `--name=value` options are retained as hints, including explicit empty values (`--name=`). Bare long options and all short options are omitted entirely: `--mode fast`, `--mode`, `-m fast` and `-mfast` do not record a setting. Their values are unknown, not empty, so they cannot conflict with a later explicit declaration such as `mode=fast`. Separate values are never recovered by consuming the next token, which could be a positional reference/read filename. This loses sensitivity to separate values and short options. An option whose argument has quoting or an escaped space in it is left uncompared, as the same run could show its value differently; option text inside another argument's quoted value is not an option. A quote left open omits the rest of the command line. The result records `options-inference=incomplete`: these hints are not proof of complete settings. Strict checking requires `options-inference=explicit` with the intended settings. Differing retained values or explicit declarations still conflict.

Thread/output/read-group/verbosity options (`--threads`, `--output`, `--rg*`, `--verbose`, `--tmp-dir` and their short-name equivalents) are omitted even with explicit values. All short options, including attached values such as `-t8`, `-@8` and `-o/path/out.sam`, are omitted too. Retained explicit values accumulate as JSON arrays in the tag string so a value containing a comma cannot alias two values; a single value that itself looks like a JSON string array is escaped too. Paths with either `/` or `\` separators are reduced to file names, positional arguments are omitted, and `--` ends option parsing. Options named like winnowr's own tags (`aligner`, `version`, `source`, `sq-alt`, `options-inference`, `winnowr-merged-complete`, `winnowr-literal-mixed-tags`) are omitted too, so a tool's own `--options-inference=explicit` is never read as a declaration. Option names containing path syntax (`/` or `\`) are omitted rather than disclosing a path. The raw command line is never stored.

## `--alignment-tags`

`strlen`, `pileup`, `extract`, `score` and `check-models` take `--alignment-tags KEY=VALUE ...`:

```console
--alignment-tags options-inference=explicit aligner=bwa-mem3 command=mem alt-project=primary t=8
```

- Keys are letters, digits, `-`, `_` and `.`; a key given twice (set or removed) is an error, and so
  is `source`, which winnowr records itself, or either internal key `winnowr-merged-complete` or
  `winnowr-literal-mixed-tags` (set or removed). `KEY=` records
  a flag without a value, and `KEY!` removes a detected tag. `options-inference` is reserved: only `options-inference=explicit` is accepted, and `options-inference!` is rejected. Unrelated overrides or removals do not clear incomplete inference.
- With a header or evidence tables to read, the tags are applied over each of them (replacing,
  adding and removing, logged) before they must agree; without one, the tags set are the whole
  provenance only if they include at least one compared setting. Removals have nothing to remove from; removals or metadata alone (such as `options-inference=explicit` or `version=1`) leave provenance unknown.
- `--alignment-tags none` records nothing; with it, neither the header nor the tables are read for provenance. It leaves provenance unknown rather than resolving incomplete inference, so strict checking still fails when models require known reads provenance.
- A provenance given by `--alignment-tags` alone (its `source` is `tags`) records only the tags
  given, so it is compared on those tags only: `--alignment-tags alt-project=primary` agrees with a
  model whose header-read provenance has `alt-project=primary`, whatever else that records. Read
  from a header, a tag set on one side only is a difference only when the side omitting it
  establishes its absence through complete settings (including a complete contributor retained
  through a merge). Without those complete constraints, an omission on an incomplete side is
  unknown, not a difference; incomplete inference is reported separately.

## From the reads to the model

| step | reads the provenance from | records it in |
|------|---------------------------|---------------|
| `strlen`, `pileup` | the header of `--alignments`, `--alignment-tags` | an `##alignment=` line of the table (a JSON object of the tags) |
| `extract` | the `--strlen` and `--pileup` tables and the header of `--alignments` (which must agree after `--alignment-tags`), `--alignment-tags` | the feature table's metadata (`alignment`) |
| `train` | the tables' metadata | the model (`alignment`) |
| `score` | the models; when any model records provenance, the header of `--alignments` and the `--strlen` and `--pileup` tables (which must agree after `--alignment-tags`), `--alignment-tags`; if no model records provenance (including legacy models), provenance resolution and evidence-source agreement checks are skipped | nothing |

Input files, including models and indexes, are assumed not to change during a command.

When provenance is checked, an evidence table may contain at most one `##alignment=` declaration, and each tag name must occur only once in its JSON object. Duplicate names are rejected, even if their values agree.

Sources that still disagree are an error naming two of them and the tags: they should be of the
same reads, so measure the tables again from them, set or remove the tags with
`--alignment-tags`, or record nothing with `--alignment-tags none`. During provenance-only
preflight, `score` inspects regular tables without loading their rows. Pipes and process
substitutions are skipped at this stage, with a warning, so they are not consumed prematurely.
A required stream is then read once for its rows and provenance declaration, which participates
in the final model check before any calls are scored. Unused streams remain unopened.

Every model records the provenance its tables have, not only models with read-evidence features:
mapping quality, depth and allele depths come from the alignments too. Tables extracted with the
`gvcf` or `vcf` feature source and no evidence tables have none unless `--alignment-tags` is
given. Tables and models of unknown provenance (including those of older winnowr versions) are
written as before, without the key.

### `train`

Compared on the compared tags only:

| tables | the model records |
|--------|-------------------|
| all alike | their provenance |
| none records one | none (logged) |
| some record none, the rest alike | the known one, with a warning naming the samples without one |
| two differ | an error naming two samples and the tags; with `--mixed-alignments`, a warning, and only the tags they share (below) |

Every pair of tables is compared, so the order of `--tables` does not matter. A table whose
provenance was given by `--alignment-tags` alone agrees with the others on the tags it gives; the
model records the union of the tables' tags, from a header if any table's was.

When training tables conflict, `--mixed-alignments` records a tags-only intersection: a tag is kept when every table that records it agrees on it and every table read from a header records it. Generated `mixed` disagreement markers are omitted; a literal `mixed` setting is kept only when every table recording that setting identifies it as literal. The model's `source` is `tags`, so `score` compares it with the reads on those tags only. Any unresolved option-inference marker is retained and still warns or fails strict checking; `--mixed-alignments` does not certify unknown settings.

When the tables agree, `--mixed-alignments` keeps their normal merged provenance, including `source=header` when any table is header-derived. Scoring comparisons are not relaxed in that case.

Training on reads aligned differently on purpose (to make a model of both) is allowed with
`--mixed-alignments`; by accident it is an error.

### `score`

`score` runs the model-set checks that do not depend on required stream contents at startup.
After loading required evidence, it compares the models with the effective provenance of the
loaded rows and any provenance-only inputs. An unknown-provenance decision waits for a required
stream that could supply its declaration; other checks still run early. Without
`--strict-provenance`, warnings about the reads' provenance are reported once, from the comparison
of the loaded evidence, so a table replaced after startup is not reported against its earlier
contents, and required tables that disagree at startup are compared again as loaded, failing only
if they still disagree; with it, an error found at startup stops scoring there. Models that agree are
compared as one, the union of their tags (whatever the order of `--models`); models that differ
are each compared. A "not checked" problem says why the reads' provenance is not known: neither
`--alignments` nor `--alignment-tags` was given, the header identifies no aligner (and the tables
record no provenance), or `--alignment-tags none` was given.

| models | reads | default | `--strict-provenance` |
|--------|-------|---------|-----------------------|
| record none | any | logged: not checked | the same |
| record it | unknown | warning: not checked; pass `--alignments` or `--alignment-tags` | error |
| record it | alike | logged | the same |
| record disjoint partial tags | record no tags comparable with the models | logged: no tags were comparable, not proof of agreement | succeeds; omitted settings were not checked |
| record it | different | warning naming each tag with both values | error |
| record it, with `options-inference=incomplete` | any | warning: settings cannot be established; supply explicit declarations | error, even if both markers match |
| record it | `options-inference=incomplete` | warning: settings cannot be established; supply explicit declarations | error, even if both markers match |
| record equal `mixed` values on a compared tag | `mixed` on that tag | warning: indeterminate, not proof of agreement | error |
| differ from each other | | warning, and each model is compared with the reads | error |

Explicit literal `mixed` values are tracked separately in internal `winnowr-literal-mixed-tags` metadata, which survives serialization and merging and cannot be supplied through `--alignment-tags`. Equal literal values are determinate; legacy unmarked `mixed` values remain conservative markers.

Equal `mixed` markers do not retain the underlying values, so the models and reads may have mixed
in different ways. Their comparison is indeterminate on those tags (`alignment_provenance_indeterminate`),
without changing how training tables or other sources merge. The reads-side mixed-header warning
still appears. Non-compared tags (`version`, `source`, `options-inference`, `winnowr-literal-mixed-tags`, and `winnowr-merged-complete`, internal merge bookkeeping) do not create mixed-setting differences. Incomplete option inference is checked separately, including when both sides retain the same marker.

A mismatch reads, for example:

```text
the reads were not aligned as the models' training reads were: alt-project: models primary, reads
keep-alt. Calls may be scored worse (see the user guide: Alignment provenance). Pass
--strict-provenance to make this an error, or --alignment-tags if the reads' recorded alignment is
wrong.
```

`evaluate` and `stratify` compare a feature table's provenance with the models' the same way, as a
warning. [`check-models`](../tools/check-models.md) runs the check without scoring, with the header
of `--alignments` (no reference needed) or `--alignment-tags`.

## When detection fails

The aligner's command line is lost when a step rebuilds the header (e.g. `samtools reheader` with a
hand-written header), when the reads are re-aligned from a file whose old records are carried
along, and with aligners that write no `@PG` record, or one without `CL`. An aligner record without
a `CL` (or with an empty one) keeps its identity and version but records
`options-inference=incomplete`: its options are unknown, not absent. Some settings are not on
the command line at all: bwa-mem3's ALT handling also depends on whether a `.alt` file sits beside
the index, and on environment variables. `sq-alt` catches ALT contigs kept in the output whatever
the command line says; for the rest, give the tags with `--alignment-tags`. Reliable header facts do not resolve incomplete option inference on their own.
