# winnowr

<!-- The opening below is repeated in README.md: change both together. -->

**winnowr post-processes single-sample GATK HaplotypeCaller VCFs with models you train on
truth-labelled samples.**

**What sets winnowr apart**

- **Works on a single sample:** no cohort or joint calling.
- **Fixes calls, not just filters them:** it can drop false calls, flip wrong zygosity,
  re-genotype indels in repeats, and rescue true calls that were filtered.
- **Uses evidence the VCF lacks:** repeat-tract lengths and reads below HaplotypeCaller's MAPQ
  cut-off.
- **Tuned for F1:** cross-validation against your truth data picks each model's thresholds.
- **Drops in after HaplotypeCaller:** no re-calling.

No trained models are distributed: you train them on samples with a truth set, such as
[GIAB](https://www.nist.gov/programs-projects/genome-bottle), labelled by
[hap.py](https://github.com/Illumina/hap.py). The
[training quick start](getting-started/quick-training.md) lists the steps, and
[Classes and labels](user-guide/classes-and-labels.md) explains what each action means.

The VCF must be genotyped from a HaplotypeCaller gVCF (e.g. one made with `--dragen-mode`). `score`
and `extract` also take that gVCF (`--gvcf`), and read it only when the VCF lacks the raw
annotations.

By default `score` also applies two indel rules that are fixed, not chosen by cross-validation. An
indel call that the thresholds leave alone is dropped when its P(keep) is below 0.5. Given GIAB's
confident regions (`--regions notinalldifficultregions=...`), an indel call outside them is dropped
at a threshold of 0.6, in place of the model's drop threshold. See
[Scoring](user-guide/scoring.md).

> **Demonstration code only.** winnowr is a research prototype, shared to illustrate an approach
> to machine-learned filtering and genotyping of germline variant calls. It is not supported, has
> not been validated, and is not intended for clinical, diagnostic or production use. Use it at
> your own risk.

## The commands

winnowr has eight commands: `strlen`, `pileup`, `extract`, `subset`, `train`, `evaluate`,
`stratify` and `score`. The [tool reference](tools/overview.md) describes each, with its full
`--help`.
