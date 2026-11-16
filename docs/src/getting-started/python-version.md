# Coming from the Python version

winnowr has been rewritten in Rust, and its outputs and models differ from the Python version's by
design: matching them is not a goal. The main changes in the defaults:

- `train` holds out whole 10 kb genomic blocks, not random calls, for early stopping, then refits
  the final model on every call (`--refit-final`). See [Training](../user-guide/training.md).
- `score`, `evaluate` and `stratify` also drop indel calls whose P(keep) is below 0.5
  (`--keep-fallback indel`), and use a drop threshold of 0.6 for indel calls outside GIAB's
  confident regions (`--outside-indel-t-drop`). See [Scoring](../user-guide/scoring.md).
- `strlen` measures reads with MAPQ 20 or more and five flank bases (`--reads mapq20,flank5`); the
  Python version measured reads of any MAPQ with two flank bases, which is `--reads legacy`. See
  [Repeat-length features](../user-guide/repeat-length.md).
- `extract` labels a call *keep* only when every hap.py record of it is a TP, labels a
  multi-allelic call from all of its records, and takes a call's gVCF annotations from the site
  with its alleles rather than the first site at its position.
- Pileup features (`winnowr pileup`, `train --pileup-features`) are new and off unless asked for.

No option restores all of the Python version's behaviour.

Models from the Python version (two files, model format 1 or 2) are not usable: winnowr refuses
them, and ships no converter, so train new models with `winnowr train`. Feature tables written by
the Python version can still be read.
