# Commands

All paths below are explicit. No scheduler command in this package has been
executed.

## 1. Local read-only preflight

From `/Users/ayushi.agrawal/Downloads`:

```bash
python3 Paper1_ExactWidth_Validation_Package_20260921/code/preflight_exact_width.py \
  --config Paper1_ExactWidth_Validation_Package_20260921/config/q050_x000_vb080_exact_width.json \
  --config-sha256 3b763155abe28c4b105a1824969ff411a6d0b1b6a5f3d9319b23ff3c39ec4800 \
  --project Paper1_ExactWidth_Validation_Package_20260921/research_source \
  --reuse-audit Paper1_ExactWidth_Validation_Package_20260921/evidence/existing_cic_vb0800_5201231/reuse_audit.json \
  --seed 200
```

Omit `--seed 200` to verify initial-state pairing for all ten frozen seeds. The
preflight builds and hashes initial states but performs no time integration.
On macOS it verifies exact CIC/q050 identity within the local runtime and shows
that the production-runtime aggregate differs only because `w_delta` has a
NumPy-dependent last-bit representation. Mac hashes are not used to admit CIC
reuse.

The authoritative reuse audit must run under the frozen MeluXina production
runtime. It is also run automatically by the batch file before q050 starts:

```bash
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/users/u103576/hexapic-main/pic_kde_2d:/home/users/u103576/hexapic-main/pic_kde_2d/scripts:${PYTHONPATH:-}"
"${PYROOT}/bin/python3" \
  /home/users/u103576/hexapic-main/pic_kde_2d/reproducibility/exact_width_q050_validation_20260921/code/audit_existing_cic_reuse.py \
  --config /home/users/u103576/hexapic-main/pic_kde_2d/reproducibility/exact_width_q050_validation_20260921/config/q050_x000_vb080_exact_width.json \
  --config-sha256 3b763155abe28c4b105a1824969ff411a6d0b1b6a5f3d9319b23ff3c39ec4800 \
  --project /home/users/u103576/hexapic-main/pic_kde_2d \
  --input /mnt/tier2/users/u103576/paper1_validation/e2_beam_drift_scan_5201231/runs/vb0800/cic \
  --legacy-campaign-root /mnt/tier2/users/u103576/paper1_validation/e2_beam_drift_scan_5201231 \
  --output /tmp/paper1_q050_reuse_audit_review.json
```

Use a fresh output path because the audit refuses to overwrite evidence. The
package already contains the verified JSON and CSV from this runtime.

## 2. Reproduce the evidence reconciliation

```bash
python3 Paper1_ExactWidth_Validation_Package_20260921/code/reconcile_existing_evidence.py \
  --retrieval paper1_source_package/reproducibility/results/missing_validation_data_retrieval_20260921 \
  --archived-analysis pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/e2_anisotropic_dense_5197174/analysis \
  --output /tmp/paper1_q050_reconciliation
```

## 3. Stage the reviewed run files on MeluXina

This transfers files but does not submit a job:

```bash
rsync -av \
  Paper1_ExactWidth_Validation_Package_20260921/config \
  Paper1_ExactWidth_Validation_Package_20260921/code \
  Paper1_ExactWidth_Validation_Package_20260921/meluxina \
  u103576@meluxina:/home/users/u103576/hexapic-main/pic_kde_2d/reproducibility/exact_width_q050_validation_20260921/
```

On MeluXina, the non-mutating checks are:

```bash
cd /home/users/u103576/hexapic-main/pic_kde_2d
sha256sum reproducibility/exact_width_q050_validation_20260921/config/q050_x000_vb080_exact_width.json
git rev-parse HEAD
```

## 4. Submission command — intentionally not run

Only after explicit authorization and review:

```bash
sbatch /home/users/u103576/hexapic-main/pic_kde_2d/reproducibility/exact_width_q050_validation_20260921/meluxina/submit_q050_exact_width_cpu.sbatch
```

The batch file refuses source/config hash mismatches and pre-existing output,
copies and audits the ten archived CIC histories, runs exactly ten new q050
histories, and analyzes only after all twenty paired histories are present. It
has a hard `01:00:00` wall limit and contains no resubmission or adaptive
extension.
