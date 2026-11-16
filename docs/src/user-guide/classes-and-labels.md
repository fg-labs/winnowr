# Classes and labels

## Genotype models

A genotype model scores the PASS calls of one variant type (SNP or indel) and predicts one of three
classes:

| class | meaning                      | action in `score`                 |
|-------|------------------------------|-----------------------------------|
| drop  | the site is really `0/0`     | FILTER `WinnowrDrop`              |
| keep  | the call is right            | none                              |
| flip  | right allele, wrong zygosity | switch GT between `0/1` and `1/1` |

An indel model trained with `--regenotype` has a fourth class, *re-genotype*: the call's genotype
or alleles are wrong (e.g. a wrong repeat length), a flip does not fix it, and the genotype of the
repeat-length caller does. See [Repeat-length features](repeat-length.md).

## Labels from hap.py

Training labels come from the hap.py output VCF given to `extract --happy`, from hap.py run with
`--engine vcfeval --pass-only`. hap.py gives each PASS call a decision (`BD`) and a match kind
(`BK`):

- a TP is *keep*,
- an FP with an allele match (`BK=am`: right allele, wrong genotype) is *flip*,
- any other FP (`BK` of `lm` or `.`) is *drop*.

Calls that hap.py does not label TP or FP (e.g. outside the truth set's confident regions) are left
out of training and evaluation. The truth FNs are counted too, since F1 needs them.

hap.py splits a multi-allelic call into one record per ALT allele. `extract` labels such a call from
all of its records: it is *keep* only when every record of it is a TP. hap.py also trims a padded
call, where `extract` looks for its label: a padded SNP (e.g. `TTAGA>TTCGA`) to its differing base,
and a padded indel (e.g. `ACGT>ACT`) to the base before its trimmed alleles (`CG>C`, one base on).

A record split from a PASS call, or trimmed from a PASS padded call, labels that call even when a
non-PASS call is at the record's position, since hap.py run with `--pass-only` labels only PASS
calls. If the hap.py VCF also labels non-PASS calls (hap.py ran without `--pass-only`), a label at a
non-PASS call's position stays that call's, as does a record split from a non-PASS call.

The hap.py VCF does not record whether hap.py ran with `--pass-only`. `extract --happy-pass-only
yes` or `no` says how it ran; the default, `auto`, infers it: it takes hap.py to have run without
`--pass-only` only if some label at a non-PASS call's position cannot be a PASS call's. Otherwise it
joins the labels as if hap.py ran with `--pass-only`, and a non-PASS call's label may go to a PASS
call. `extract` logs which way it joined the labels, and why. Run hap.py with `--pass-only` for
reliable labels.

## Flips that a flip cannot fix

Some allele-match FPs cannot be fixed by a het <-> hom flip: a call that is neither `0/1` nor
`1/1`, or whose flip is not the truth genotype (e.g. a `0/1` call whose truth is `1/2`, which needs
another allele). `train --unfixable-flips` sets what genotype training does with them:

- `flip` (the default) trains them as flips, as hap.py labels them,
- `drop` trains them as drops, since dropping is the only action that removes the FP,
- `exclude` leaves them out of the fits.

In cross-validation they are scored as FPs whatever they are trained as. The choice is recorded in
the model.

## Promote models

A promote model scores the non-PASS calls of one variant type that do not overlap a PASS call, and
predicts one of three classes:

| class        | action in `score`                                     |
|--------------|-------------------------------------------------------|
| stay         | none: the call stays filtered                         |
| promote      | FILTER PASS (the old FILTER kept in `WINNOWR_OFILTER`) |
| promote+flip | promote, and switch GT between `0/1` and `1/1`        |

hap.py with `--pass-only` does not evaluate non-PASS calls, so `extract` labels them by matching
them against the truth VCF (`--truth-vcf`) inside its confident regions (`--truth-bed`). Calls and
truth records are trimmed and, given `--reference`, left-aligned, then matched on position, REF and
ALT alleles. A candidate is *promote* when it matches a truth record with the same genotype,
*promote+flip* when switching its genotype between `0/1` and `1/1` gives the truth genotype, and
*stay* otherwise. A match counts only when the truth record is a hap.py FN of the same variant type,
and only for the first candidate that matches it, so a correct promotion turns exactly one FN into
a TP.

Train a promote model with `train --task promote`, from tables extracted with `--happy`,
`--truth-vcf` and `--truth-bed`.
