# Frozen article-result examples

This directory is the article-specific layer on top of the general
`pic_kde_2d` implementation. It contains reducers, frozen settings, regression
tests, archived numerical products, and scientific provenance. It contains no
manuscript source.

Use the repository-root entry point rather than modifying archived outputs:

```bash
python3 reproduce.py tests
python3 reproduce.py bounded
python3 reproduce.py completion
python3 reproduce.py exact-width
```

`python3 reproduce.py all` runs all four maintained workflows. Results are
written to a new directory under repository-root `reproduced/`.

Only the current reference products and inputs needed by maintained reducers
are bundled. Historical reports may name superseded `_rN` working directories
that were present during the research audit; those duplicate directories are
not part of this lean distribution. Their final replacements are retained, and
the top-level replay regenerates current products directly from raw histories.

Historical scheduler paths and absolute paths inside archived JSON records are
provenance and may refer to their original MeluXina or workstation locations.
The top-level wrapper supplies portable repository-relative paths for current
reanalysis.
