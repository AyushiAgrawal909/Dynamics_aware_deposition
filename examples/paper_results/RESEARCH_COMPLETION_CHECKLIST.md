# Research completion checklist reconciliation

Snapshot: 2026-09-12, after recovery of MeluXina job `5202227` and completion
of the existing-data analyses. The authoritative manuscript remains read-only
for this stage.

Status meanings: **COMPLETE** means the requested work has an evidence-based
disposition, including an explicit failed or unresolved scientific outcome;
**PARTIAL** means a named dependency remains; **MISSING** means the artifact or
measurement does not exist; **INVALIDATED** means a former interpretation is
contradicted by a verified defect. A completed computation is not labelled a
scientific pass unless its registered gates pass.

## Brief-wide reconciliation

| Brief item | Status | Evidence and disposition |
|---|---|---|
| 1. Scientific objective and scope | **COMPLETE** | `RESEARCH_COMPLETION_REPORT.md` and `CLAIM_EVIDENCE_MAP.csv` classify each proposed story element. Protected-mode linear rate preservation and a modest exact-interior grid-noise benefit are supported in their stated scopes; unrestricted nonlinear/full-`f`, universal-optimality and speedup claims are not. |
| 2. Ground work in implementation | **PARTIAL** | Current code, frozen configurations, jobs, source hashes and supplied-versus-later code are inventoried in `REPRODUCIBILITY_MANIFEST.md` and `CAMPAIGN_MANIFEST.json`. The exact source revision for historical job `4913408` and several older Python campaigns was never embedded and remains explicitly unknown in `MISSING_INFORMATION.md`; it cannot now be recovered from the outputs. |
| 3. Preserve known experimental record | **COMPLETE** | The registered four-drift result remains `results/e2_beam_drift_scan_analysis_20260911_timeout_r6`; the original 120/150 timeout is preserved. Extension and all post-hoc reductions use new directories. Weak-Landau primary/common-window outputs remain distinct. |
| 4. Numerical formulation and transfer verification | **COMPLETE** | `NUMERICAL_FORMULATION.md`, `FILTER_TRANSFER_AUDIT.md`, `DISPERSION_AUDIT.md`, `CONSERVATION_AUDIT.md`, and `OPERATOR_AND_TRANSFER_AUDIT.md` distinguish spectral Python and finite-difference C++ chains. Exact binomial/compensated transfers and the spectral operator correction are classified; no affected particle history or predicted root required recomputation. |
| 5. Estimator and seed diagnosis | **COMPLETE** | `ESTIMATOR_AND_SEED_DIAGNOSTIC.md` and `results/e2_estimator_comparison_complete_20260912_with_extension/` cover completed drift points, historical Table-5 corner, direct compensated corner, individual rates, 5,000 paired bootstraps, mixtures, seed shares, effective counts and leave-one-out influence. `results/e2_seed_rate_variation_20260912_with_extension/` covers amplitude/phase, fixed window grid, transients/loading moments and 260 finite-marker roots. |
| 6. Recover final drift point | **COMPLETE** (physical conclusion **UNRESOLVED**) | Job `5202227` completed 30/30 files, exit `0:0`, elapsed `02:21:40`, with the frozen campaign and amendment hashes verified. `results/e2_vb0818_extension_audit_20260912/` shows every arm fails fit-start SNR and three seeds/arm leave the linear-`delta f` regime. No accepted near-marginal rate or stabilisation claim is made. |
| 7. Boundary, interior and measured acceptance | **COMPLETE** | Boundary and `R=.95` interior widths remain separate in the frozen campaign. `results/e2_interior_noise_replay_20260911/` measures the exact `.80` interior width after reproducing saved anchors to `5e-15`; no boundary-width benefit is borrowed. |
| 8A. Conditional particle-number diagnostic | **COMPLETE — not triggered** | Current saved-loading finite-marker rates explain the seed spread with correlations `0.9944--0.9999`; a higher-PPC campaign is unnecessary for the narrowed claim. The only remaining question such a run could answer is continuum convergence at `.818`. |
| 8B. Consistent practical comparison | **COMPLETE with narrowed scope** | The weak-Landau v2 same-campaign comparison is retained and still fails CIC duration at 1.721 e-foldings. E2 evidence is explicitly split between projected linear rates and a compatible short full-`f` grid-error replay. A new E2 full-`f` campaign is needed only if coauthors retain a stronger end-to-end claim. |
| 9. Novelty and fair comparison | **COMPLETE for a focused audit** | `NOVELTY_AND_PRIOR_ART.md` checks primary sources and narrows novelty to the implementation-audited protected-mode rule, explicit interior margin and controlled equilibrium scan. KDE, smoothing-induced dispersion and ensemble averaging are not claimed as new; proxy/global-ranking limits are explicit. |
| 10. Figures and source tables | **COMPLETE for affected assets** | The gate-aware drift figure/table is in `results/e2_beam_drift_scan_analysis_20260912_complete_extension_r9/`; the estimator/finite-marker supplement is in `results/e2_estimator_seed_diagnostic_20260912/`. Both have PDF/SVG or PNG outputs and machine-readable source rows. `FIGURE_CAPTIONS.md` supplies factual captions; the manuscript was not edited. |
| 11. Auditable evidence package | **COMPLETE** | All named deliverables exist: report, claim map, operator audit/tables, estimator audit/tables, campaign manifest, novelty matrix, figures/captions, coauthor responses, correction ledger and updated reproducibility instructions. The final-current-code one-command rebuild passed and wrote 35 hashed products in `results/research_completion_rebuild_check_20260912_r3/`; the first failed packaging attempt and successful pre-caption `_r2` check are retained. |
| 12. Stopping condition and handoff | **COMPLETE** | Every material claim has a disposition, the necessary bounded extension is terminal, no further simulation is required for the narrowed paper, 84/84 reproducibility tests and 26/26 secondary-claim tests pass, and `RESEARCH_COMPLETION_REPORT.md` gives the six-part handoff. |

## Requested six-task execution result

| Task | Status | Result |
|---|---|---|
| 1. Recover `v_b=.818` extension | **COMPLETE** | No duplicate was submitted and no budget was extended. Files complete; scientific rate unresolved under frozen SNR/linearity gates. |
| 2. Complete estimator comparison | **COMPLETE** | Registered results preserved as primary; paired individual-rate bootstrap is labelled sensitivity. At `.80`, fixed/CIC is `0.908911 [0.887432,0.926656]` and constrained/CIC is `0.954460 [0.944068,0.963103]`. |
| 3. Diagnose individual rates | **COMPLETE** | Initial amplitude/phase and fixed-window variation do not explain the spread. Seed-specific finite-marker loading does; no higher-PPC run was launched. Historical loading conclusions remain nominal where marker arrays/revision are absent. |
| 4. Correct filter/operator inconsistencies | **COMPLETE** | `sigma_eq=sqrt(n/2)` is low-`k` only; exact E2 `n=4` proxy errors and spectral/FD symbol differences are tabulated. Corrections affect wording/equation metadata, not existing simulations or E2 roots. |
| 5. Quantify exact `R=.95` noise | **COMPLETE at `v_b=.80`** | At `sigma_y=0.1738592384 Delta y`, density-error-power reduction is `1.081314 [1.079778,1.082929]`; pre-gather grid-field reduction is `1.038554 [1.037245,1.040182]`. Boundary results remain separate. |
| 6. Decide practical claim | **COMPLETE with narrower wording** | Supported: scoped projected linear protection plus separately measured exact-width grid-noise benefit. Not supported: one end-to-end E2 full-`f` performance result, particle-force noise reduction, nonlinear fidelity or speedup. |

## Remaining coauthor-dependent or irrecoverable inputs

- Exact historical source revision for job `4913408` and other unversioned
  Python campaigns: **MISSING and irrecoverable from present artifacts**.
- Author-approved CRediT roles and public archive/DOI: **MISSING**, requiring
  author decisions rather than simulation.
- A new E2 full-`f`, same-campaign noise/rate/particle-force experiment:
  **MISSING but not required** for the narrowed claim. It becomes necessary
  only if the manuscript retains an unrestricted end-to-end practical claim.
- A higher-PPC `.818` convergence study: **MISSING but not required**. It would
  answer only whether the finite-marker near-marginal branch converges toward
  the continuum prediction.

No manuscript source was modified in producing this reconciliation.
