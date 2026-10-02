# Verification report

Original scientific workflows were verified from the clean code-and-data tree
on 30 September 2026. The general-code/example layout and optional-checksum
entry point were revalidated on 1 October 2026. The lean tree was revalidated
after pruning on 2 October 2026.

## Scientific regression suite

Command:

```bash
MPLBACKEND=Agg python3 -m unittest discover \
  -s examples/paper_results/tests -p 'test_*.py'
```

Result: `Ran 87 tests ... OK`.

The two tests that previously checked prose in a manuscript now check the
underlying archived numerical anchors and explicit metric definitions. No
scientific acceptance threshold or expected numerical result was changed.

## General solver test suite

Command, with the package import path exposed as an editable installation
would expose it:

```bash
PYTHONPATH=pic_kde_2d pytest -q pic_kde_2d/tests
```

Result: `121 passed, 2 xfailed`.

The two strict expected failures retain their original numerical tolerances:
the legacy low-resolution CIC Landau smoke benchmark misses a 25% continuum
rate tolerance, and the legacy PPC-16 spectral-W benchmark misses a 30%
tolerance. They are recorded as known limitations rather than silently relaxed
or deleted. Unexpectedly passing either test is configured to fail the suite so
that its status must be reviewed.

## Exact-width campaign replay

Command:

```bash
python3 reproduce.py exact-width \
  --output /private/tmp/paper1_results_smoke_exact_20260930
```

Result: `PASS_EXACT_WIDTH_REDUCTION_REPRODUCED`.

- primary mean-individual-rate ratio: `0.9556447297592618`
- paired 95% interval: `[0.9452733092216409, 0.9640059783937591]`
- secondary ensemble-amplitude ratio: `0.9706344447819388`
- secondary paired 95% interval: `[0.9553563586841591, 0.9737072604223512]`
- predicted ratio: `0.9513008502587932`

## Bounded analysis smoke test

Command:

```bash
python3 reproduce.py bounded --bootstrap 1000 \
  --output /private/tmp/paper1_results_smoke_bounded_20260930
```

Result: completed successfully and generated machine-readable tables plus
scientific figures without a manuscript or TeX dependency. The default command
uses 10,000 bootstrap draws; 1,000 was used only for this packaging smoke test.

## Wider research-completion replay

Command:

```bash
python3 reproduce.py completion \
  --output /private/tmp/paper1_results_smoke_completion_20260930
```

Result: `PASS_AFFECTED_RESULTS_REBUILT_FROM_CACHED_RAW_DATA`, 35 products.

This replay covered exact transfer/operator checks, the beam-drift scan,
individual versus ensemble rate estimands, seed-rate/finite-marker diagnostics,
interior-width noise replay, the unresolved `v_b=0.818` extension audit, and
the diagnostic figures.

The same workflow was rerun after separating the general code from the frozen
article examples. It again reported
`PASS_AFFECTED_RESULTS_REBUILT_FROM_CACHED_RAW_DATA` with 35 products.

After pruning duplicate and obsolete files, `python3 reproduce.py all
--bootstrap 1000` again completed the scientific regressions, bounded analysis,
35-product completion replay, and exact-width replay. The reduced bootstrap
count was used only for this dependency smoke test; the default remains 10,000.

## Repository integrity behavior

`python3 reproduce.py verify` always checks the required directory structure
and the frozen exact-width configuration and summary hashes. If
`SHA256SUMS.txt` is present, it additionally verifies the complete repository.
If that optional file is removed, scientific analyses and regression tests
continue to work and the skipped full-file check is reported explicitly.

## Content boundary

Automated inspection found no manuscript directory and no file with extension
`.tex`, `.bib`, `.bbl`, `.cls`, or `.bst`. The folder includes scientific
result figures and the small set of logs consumed by reducers or needed for
campaign accounting; ordinary per-run console logs were removed.
