# Missing-CIC retrieval report

Date: 2026-09-21. The search was read-only; no scheduler job was submitted.

## Result

The exact CIC baseline set required for the q050 validation was recovered from
MeluXina:

```text
/mnt/tier2/users/u103576/paper1_validation/
  e2_beam_drift_scan_5201231/runs/vb0800/cic/seed200--209/
```

Each seed directory contains `mode_history.npz`, `result.json` and
`campaign_manifest.json`. The local preserved copy is
`evidence/existing_cic_vb0800_5201231/`.

The exact q050 width appears on MeluXina only in the ten static selector
records from job 5197174. No already-computed exact-width dynamics histories
were found. Therefore only ten q050 histories remain to be computed; the CIC
arm need not be rerun.

## Mac search

The local search checked the Paper 1 evidence/reproducibility trees for mode
histories, result metadata, candidate labels and the full-precision width
`0.1716392815470568`. It found:

- the full q050 static-selection evidence;
- CIC mode histories for seeds `105--109` at mode `(10,1)`, which are not
  compatible with the required `(12,1)` validation;
- no exact `(12,1)`, PPC-512, `dt=0.02`, `t_end=71.16` CIC set for the proposed
  paired test.

## MeluXina search

The remote search parsed 1,709 result/manifest metadata files under the Paper 1
validation roots and checked candidate names, widths, mode, PPC, time step,
duration and seed. It found the exact CIC set above and the following frozen
campaign identity:

```text
git revision        ed6dd96d16d0b70a53bf9486fd78a270ab3486d6
frozen config SHA   e664460d392a216b953bc87d3859c3bce9e1cce2f1652ab9417c1a58b3baf446
runner SHA          9935058260947f6af08cb801a9200508d7e040da6c79864214c5e49452a75279
```

All ten CIC results match the required mode `(12,1)`, symmetric drift
`v_b=0.80`, PPC 512, `alpha=1e-8`, `dt=0.02`, `t_end=71.16`, zero Gaussian
width and target-mode projection. Each history has 1,780 samples from 0 through
71.16 at cadence 0.04. The maximum saved delta-f ratio ranges from
`1.2974e-4` to `1.5961e-2`, below the frozen limit 0.5.

Applying the already frozen `[24.32,69.16]` reducer to this CIC arm gives:

```text
mean individual-seed rate       0.1244508093
paired-seed bootstrap 95% CI   [0.1111166540, 0.1383865170]
ensemble-amplitude slope        0.1447458990
ensemble-amplitude R^2          0.9994114070
ensemble-amplitude e-foldings   6.490406112
minimum individual-fit R^2      0.9918686021
fit-start signal/floor ratio   22.24892683
maximum delta-f ratio           0.01596124238
```

Thus the recovered baseline is complete and passes the frozen CIC quality
checks. These numbers do not determine the q050/CIC preservation verdict,
which remains unavailable until the q050 histories exist.

## Pairing evidence and limitation

The archived campaign stored source/config hashes and seeds but did not store
the initialized particle arrays or their hashes. The saved `pairing_key` is a
label, not proof of array identity. The audit reconstructs each state with the
archived wrapper and arguments under the MeluXina production runtime. The
production script, `particles2d.py`, `delta_f.py` and `grid2d.py` are covered
by the historical SHA-256 manifest. The reconstructed time-zero target-mode
coefficient equals the saved value exactly.

The q050 runner must match each production-runtime reconstructed aggregate hash
before integration. The reducer then checks all six actual q050 array hashes
against the audit. Historical arrays remain unavailable, so this is a
deterministic reconstruction argument—not a recovered historical hash.

The old manifest omitted `conventions.py` and `cic.py`. Their observed mtimes
predate the job, but mtimes are not cryptographic provenance. The git revision
does not close the gap because the research Python tree was untracked. This
residual limitation is explicit in the audit. The evidence was judged adequate
for conditional reuse; any future hash mismatch stops the campaign rather than
silently assuming pairing.

The `6.4904` duration above belongs to the ensemble-amplitude fit. Individual
fitted durations range from `4.2077` to `7.3459` e-foldings and are reported by
seed in `CIC_QUALITY_GATES.md` and `reuse_audit.csv`.

## Consequence for the prepared job

The reviewed batch specification now copies and audits the ten archived CIC
histories, runs only q050 for seeds `200--209`, and performs the paired
reduction after all twenty histories are present. The expected new cost is
about 0.60 CPU node-hour and the hard cap is one CPU node-hour. These seeds are
independent of the bandwidth-selection seeds but have already been analysed in
job 5201231. The former fresh `105--114` paired design was changed before any
q050 result existed. No submission has been made.
