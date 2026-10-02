# Commands and provenance

## Local reconstruction

```bash
MPLCONFIGDIR=/tmp/paper1-mpl-cache \
XDG_CACHE_HOME=/tmp/paper1-xdg-cache \
/opt/homebrew/bin/python3 \
  paper1_source_package/reproducibility/retrieve_missing_validation_data_20260921.py
```

The script refuses to overwrite its output directory. It validates the fixed
candidate count and ordering, the fixed seed split, all 40 three-arm pairing
manifests, configuration hashes, source hashes and all history schemas before
performing the analyses.

## Read-only MeluXina inventory

```bash
ssh u103576@meluxina \
  'find /mnt/tier2/users/u103576/paper1_validation -maxdepth 5 \
   \( -iname "*anisotropic*" -o -iname "*weak*landau*" \
      -o -iname "*finite*marker*" -o -iname "*loading*converg*" \
      -o -iname "*ppc*converg*" -o -iname "*beam*drift*" \) -print'
```

```bash
ssh u103576@meluxina 'python3 - <<"PY"
import os, json, collections
root = "/mnt/tier2/users/u103576/paper1_validation"
counts = collections.Counter()
for directory, _, files in os.walk(root):
    for name in files:
        if name != "result.json":
            continue
        path = os.path.join(directory, name)
        try:
            record = json.load(open(path))
        except Exception:
            continue
        if record.get("benchmark") != "two_stream":
            continue
        campaign = os.path.relpath(path, root).split(os.sep)[0]
        counts[(campaign, record.get("ppc"), record.get("marker_dynamics"))] += 1
for key, count in sorted(counts.items()):
    print(key, count)
PY'
```

Observed dynamic history counts were:

- `e2_beam_drift_scan_5201231`: 120, PPC 512;
- `e2_beam_drift_scan_extension_5202227`: 30, PPC 512;
- `e2_compensated_corner_5195640`: 50, PPC 512;
- `e2_new_mode_m10n1_5197185`: 20, PPC 512.

The exact CIC baseline was then identified in the first campaign with a
metadata-filtered search and copied read-only:

```bash
scp -r \
  u103576@meluxina:/mnt/tier2/users/u103576/paper1_validation/e2_beam_drift_scan_5201231/runs/vb0800/cic \
  Paper1_ExactWidth_Validation_Package_20260921/evidence/existing_cic_vb0800_5201231
```

It contains seeds `200--209` at the exact required mode and numerical
configuration. See `CIC_RETRIEVAL_REPORT.md` and the machine-readable
`reuse_audit.json` for the full search outcome, per-file SHA-256 values and the
deterministic initial-state reconstruction.

## Production-runtime initialization reconstruction

The archived scheduler fixes the interpreter and exports its library path. The
authoritative reconstruction used that same invocation and performed no time
integration:

```bash
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
"${PYROOT}/bin/python3" code/verify_initial_state_compatibility.py \
  --project /home/users/u103576/hexapic-main/pic_kde_2d
```

Observed runtime: Python 3.11.10 and NumPy 2.3.5. The Mac reconstruction used
Python 3.14.6 and NumPy 2.5.1. `x`, `y`, `vx`, `vy` and `w_full` hashes agree
across the two runtimes; `w_delta` differs at binary last-bit level, so Mac
aggregate hashes are not used for production pairing. The final audit was run
on MeluXina and its output is preserved verbatim.

Job 5201231 did not save historical particle arrays or hashes. Its input
manifest explicitly hashes the wrapper and the state-construction files except
for the transitive `conventions.py`; `cic.py` was also omitted but does not
construct the initial arrays. The research Python tree was untracked, so the
recorded git revision is context rather than a substitute source hash. These
limits are machine-readable in `runtime_provenance.json`.

## Cost evidence for the unsubmitted convergence specification

```bash
ssh u103576@meluxina \
  'sacct -j 5201231,5201218,5202227,5195640 \
   --format=JobID,JobName,Partition,AllocCPUS,AllocTRES,Elapsed,TotalCPU,State,ExitCode -P'
```

The five job-5195640 CPU array tasks, each containing one ten-seed PPC-512,
`t=60` arm, completed in 27:03--30:34 wall time on one node. The cost estimate
in the JSON scales their mean wall time linearly with PPC and duration and adds
about 26% cap margin.

## Figure verification

Both PDFs were rendered with Poppler at 180 dpi and inspected as PNGs:

```bash
pdftoppm -png -r 180 fig_bandwidth_85_complete_rule.pdf /tmp/pdfs/bandwidth
pdftoppm -png -r 180 fig_landau_three_arm_common_window.pdf /tmp/pdfs/landau
pdfinfo fig_bandwidth_85_complete_rule.pdf
pdfinfo fig_landau_three_arm_common_window.pdf
```

Both are single-page vector PDFs with readable labels, legends and selection
windows. PNG and SVG companions are retained beside them.
