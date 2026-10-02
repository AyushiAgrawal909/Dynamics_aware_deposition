# Reproducing the verified Paper 1 manuscript

The authoritative manuscript is
`Dynamics_constrained_Deposition/introduction.tex`. The older
`paper1_source_package/paper1.tex` is retained only as an audit artifact.
Commands below are run from `/Users/ayushi.agrawal/Downloads` (or from a clone
with the same three sibling directories).

## Environment

- Python 3 with NumPy and Matplotlib for figures and numerical reducers.
- Tectonic 0.17.0 and its cached bundle for the recorded local build.
- Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`) for PDF verification.
- Historical dependency versions remain unknown where the run outputs did not
  record them; see `MISSING_INFORMATION.md`.

For reproducible local font caches, a writable `MPLCONFIGDIR` may be supplied.
The warnings produced when Matplotlib falls back to a temporary font cache do
not change figure data or assertions.

## Regenerate figures

```bash
python3 paper1_source_package/scripts/make_figures.py
(cd pic_kde_2d/paper1_selector_comparison/figures && python3 make_fig_erasure.py)
(cd pic_kde_2d/paper1_selector_comparison/figures && python3 make_fig_compensator.py)
cp paper1_source_package/figs/*.pdf Dynamics_constrained_Deposition/figs/
```

Regenerate the timeout-censored E2 drift figure and paired table from the 120
complete raw histories with:

```bash
MPLCONFIGDIR=/a/writable/cache python3 \
  paper1_source_package/reproducibility/analyze_E2_beam_drift_scan.py \
  --config paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911/frozen_campaign.json \
  --input pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/e2_beam_drift_scan_5201231/runs \
  --timeout-record paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911/campaign_timeout_5201231.json \
  --output /a/new/nonexistent/output/directory
```

Without `--timeout-record`, the same reducer fails loudly on the first missing
`v_b=0.818` history.  The timeout mode permits only a wholly absent drift block
declared by an exhausted-budget Slurm record; it does not accept partial arms.

Reproduce the post-hoc estimator diagnostic, which leaves that registered
reduction unchanged, with:

```bash
python3 paper1_source_package/reproducibility/diagnose_E2_rate_estimator.py \
  --config paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911/frozen_campaign.json \
  --scan-root pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/e2_beam_drift_scan_5201231/runs \
  --registered paper1_source_package/reproducibility/results/e2_beam_drift_scan_analysis_20260911_timeout_r6/E2_beam_drift_scan.json \
  --transfer-summary paper1_source_package/reproducibility/results/e2_beam_drift_scan_transfer_20260911/transfer_summary.json \
  --corner-root pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/e2_joint_corner_4913408 \
  --corner-launcher pic_kde_2d/scripts/submit_E2_joint_corner_sweep_gpu.sbatch \
  --output /a/new/nonexistent/diagnostic/output
```

The estimator diagnostic is in
`results/e2_rate_estimator_diagnostic_20260911_r3`. The wording-only drift
figure regeneration is in
`results/e2_beam_drift_scan_analysis_20260911_timeout_r8`; its CSV, JSON and
bootstrap draws are byte-identical to the registered `r6` reduction.

The separately frozen `v_b=0.818` extension completed all 30 histories. Rebuild
the combined diagnostic result without altering the registered timeout result:

```bash
MPLBACKEND=Agg python3 \
  paper1_source_package/reproducibility/analyze_E2_beam_drift_scan.py \
  --config paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911/frozen_campaign.json \
  --input pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/e2_beam_drift_scan_5201231/runs \
  --extension-input pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260912/e2_beam_drift_scan_extension_5202227 \
  --timeout-record paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911/campaign_timeout_5201231.json \
  --output /a/new/nonexistent/combined-output
```

The final gate-aware rendering is
`results/e2_beam_drift_scan_analysis_20260912_complete_extension_r9`. It omits
the signal-failing `.818` rows from the measured curves and retains their
descriptive values in CSV/JSON. The caption records: no growing fixed-filter
model branch; measurement unavailable under the frozen gates.

## Rebuild the research-completion evidence

One command regenerates only the affected operator tables, combined drift
analysis, estimator/bootstrap tables, seed/loading diagnostics, exact-interior
noise replay, extension audit, and diagnostic figures from cached raw data. It
submits no simulation and refuses to overwrite its output directory:

```bash
MPLBACKEND=Agg python3 \
  paper1_source_package/reproducibility/rebuild_research_completion.py \
  --output-root /a/new/nonexistent/research-completion-output
```

The registered historical outputs remain immutable; rebuilt products receive
new paths and a `rebuild_manifest.json` with per-file SHA-256 values.

Each generator checks archived numerical anchors before writing. The erasure
generator first prints the JSON keys, reports that per-seed arrays are absent
from the JSON, then enumerates the recovered NPZ schemas. The compensator
generator checks `W^2=0.997137575` against measured attenuation `0.0028624`.
`FIGURE_PROVENANCE.md` maps every plotted measurement to a source artifact.

## Run numerical regressions

```bash
python3 -m unittest discover -s paper1_source_package/reproducibility/tests -p 'test_*.py'
```

Expected result: 84 tests, all passing. Task-10's codebase tests are run from
`pic_kde_2d/` with:

```bash
pytest -q tests/test_phase5.py tests/test_secondary_claims_audit.py \
  tests/test_adaptive_sigma_controller.py
```

Final recorded result: 26 passed. Details are in
`pic_kde_2d/reproducibility/SECONDARY_CLAIMS_AUDIT.md`.

## Compile and audit

```bash
mkdir -p tmp/pdfs/task11_build
(cd Dynamics_constrained_Deposition && \
  TECTONIC_CACHE_DIR=../tmp/tectonic_cache tectonic --only-cached \
  --keep-logs --keep-intermediates --outdir ../tmp/pdfs/task11_build \
  introduction.tex)
python3 paper1_source_package/reproducibility/audit_manuscript.py \
  --pdf tmp/pdfs/task11_build/introduction.pdf
```

The final recorded build uses `elsarticle` with `final,1p,times`, produces 37
pages, resolves all bibliography/reference passes, imports nine real PDF
figures, and keeps all tables upright.  The verified artifact after the E2
estimator diagnostic and figure-wording correction is
`Dynamics_constrained_Deposition/output/Dynamics_constrained_Deposition_verified_20260911_r4.pdf`
(SHA-256 `be7ee1a4d5fbf57f969fed1725a47c1d0f614bc008e948b82e94633830505af9`).
The structural audit also checks that the authoritative and generator-copy
figure PDFs hash-match and that every PNG counterpart records 300 dpi.

## Reproduce the fresh weak-Landau confirmation

The completed 40-seed campaign is reduced in two locked stages.  Stage 1 is
already frozen; Stage 2 verifies its hash before computing comparisons:

```bash
python3 paper1_source_package/reproducibility/analyze_weak_landau_confirmatory_v2.py \
  --config paper1_source_package/reproducibility/weak_landau_confirmatory_v2_global_velocity.json \
  analyze \
  --input-low paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_production_20260911/low \
  --input-reference paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_production_20260911/reference \
  --window-decisions paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_stage1_windows_20260911/window_decisions.json \
  --output /a/new/nonexistent/output/directory
```

`make_weak_landau_window_diagnostic.py` materializes the coherent complex
histories, one-period geometric envelopes, frozen fit lines, floors and censor
markers.  Its output must likewise be a new directory because the script
refuses to overwrite an audit artifact.

The separately labelled common-window sensitivity reduction is reproduced
from the same raw histories with:

```bash
python3 paper1_source_package/reproducibility/analyze_weak_landau_common_window_sensitivity.py \
  --input-low paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_production_20260911/low \
  --window-decisions paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_stage1_windows_20260911/window_decisions.json \
  --primary-result paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_stage2_analysis_20260911/confirmatory_result.json \
  --output /a/new/common-window/output
```

The historical-loading, E2 C4 signal-floor, and compensated-prediction audit
uses existing files only:

```bash
MPLCONFIGDIR=/tmp/paper1-mpl \
python3 paper1_source_package/reproducibility/audit_evidence_closure.py \
  --output /a/new/evidence-closure/output
```

## Rebuild the recovered delta-f campaigns

The job-ID-guided MeluXina recovery is preserved under
`pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/`.  From the
workspace root, the archived tables can be regenerated with:

```bash
python3 pic_kde_2d/scripts/analyze_linearized_delta_f_reference.py \
  --input pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/df_linear_ref_4872599 \
  --output /a/new/reference/output --seeds 10 --nboot 2000
python3 pic_kde_2d/scripts/analyze_linearized_delta_f_m12_sigma.py \
  --input pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/df_m12_sigma_4872745 \
  --output /a/new/m12/output --seeds 10 --nboot 2000
python3 pic_kde_2d/scripts/analyze_linearized_delta_f_band.py \
  --input pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/df_band_1_14_4873221 \
  --output /a/new/band/output --seeds 10 --nboot 2000
```

The recorded local reductions reproduce the three archived numeric CSVs to
maximum absolute differences `3.38e-14`, `3.66e-15`, and zero.  File counts,
remote/local aggregate hashes, scheduler states and the materialized sigma-zero
symlink are recorded in
`results/recovered_delta_f_campaigns_20260911/recovery_manifest.json`.

## Evidence hierarchy

1. raw run arrays and embedded metadata;
2. deterministic audited reducers and machine-readable results;
3. manuscript tables/figures generated from those outputs;
4. analytic predictions only where they are labelled analytic.

Unknown historical revisions or settings are not filled from current source.
New simulations were launched only after implementation and transfer checks,
and only where the existing evidence could not answer the registered test.
