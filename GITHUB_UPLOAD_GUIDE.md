# GitHub preparation (no upload performed)

This folder is arranged as a repository payload, but the preparation did not
run `git init`, make a commit, configure a remote, or upload anything.

## Before public release

No confirmed project-wide redistribution license was present in the recovered
research trees. Resolve that decision and audit third-party/data redistribution
rights before making the repository public. `CITATION.cff` intentionally has no
license field until then.

## Size

The lean repository is about 350 MB because it includes raw histories. Its
largest current file is below 10 MB, comfortably under GitHub's 100 MB hard
limit; verify this again before the first commit:

```bash
find . -type f -size +99M -print
```

Git LFS is not required by the current per-file sizes. A future release may be
more convenient if immutable raw evidence is attached as a versioned release
asset, but do not remove it from this prepared folder without also revising the
reproduction instructions and checksums.

## Clean-clone checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
python3 reproduce.py verify
python3 reproduce.py tests
python3 reproduce.py exact-width
```

Generated output under `reproduced/` is ignored by Git.

`SHA256SUMS.txt` and `FILE_INVENTORY.csv` are optional but recommended for a
data-bearing release. Regenerate them with `python3 tools/build_checksums.py`
after an intentional repository change.

## Optional repository creation later

Only after licensing and public-data review:

```bash
git init -b main
git add .
git status --short
git commit -m "Add dynamics-aware PIC code and reproducible examples"
```

Inspect the staged file list before committing. Add a remote and push only when
you deliberately choose to publish; this package does not do either.
