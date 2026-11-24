# Tool reference

Each page has a short description, typical usage, and the command's full `--help`.

| command                 | what it does                                                                       |
|-------------------------|------------------------------------------------------------------------------------|
| [strlen](strlen.md)     | per-read repeat-tract lengths at indel calls (for repeat-length features)          |
| [pileup](pileup.md)     | allele read counts by MAPQ at SNP and indel calls (for pileup features)            |
| [extract](extract.md)   | per-call features (and labels, for training) as a Parquet feature table            |
| [subset](subset.md)     | restrict a labelled feature table to some contigs, or away from some regions       |
| [train](train.md)       | train a model for one variant type; thresholds chosen by cross-validation          |
| [evaluate](evaluate.md) | F1 of a model's decisions on a held-out labelled sample                            |
| [stratify](stratify.md) | decisions, F1 and P(keep) calibration by benchmark stratum and GIAB stratification |
| [check-models](check-models.md) | check a set of models before a run, and report what they need (text or JSON) |
| [score](score.md)       | apply models to a sample and write the re-genotyped, filtered VCF                  |

## Global options

```text
{{#include ../../_generated/cli/winnowr.txt}}
```

`-t`/`--threads` is accepted by every command; see [Performance
tuning](../user-guide/performance.md). Logging goes to standard error, at the `info` level by
default; set `RUST_LOG` (e.g. `RUST_LOG=winnowr=debug`) to change it.
