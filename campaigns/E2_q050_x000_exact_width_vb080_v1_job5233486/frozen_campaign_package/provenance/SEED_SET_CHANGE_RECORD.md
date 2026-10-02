# Seed-set change record

The first prepared design used fresh paired CIC and q050 runs for seeds
`105--114` (twenty new histories). It was replaced before any exact-width q050
dynamics result existed.

The reason was recovery of the complete, configuration-compatible CIC baseline
from job 5201231 for seeds `200--209`. A production-runtime audit reconstructed
the deterministic initial states from the archived wrapper, configuration,
explicit source hashes and seeds, and froze those reconstructed hashes as
pre-integration requirements for the new q050 histories. Reuse reduced the new
campaign to ten q050 histories and the cap from 1.5 to 1.0 CPU node-hour.

The change was evidence- and cost-driven, not outcome-driven. Seeds `200--209`
are independent of bandwidth-selection seeds `100--104` and held-out static
error seeds `105--109`. They are not fresh dynamics seeds: all were already
analysed in the beam-drift campaign. The final report must state both facts.

The old job did not save historical particle arrays or their hashes. The final
design therefore does not infer pairing from equal seed numbers. The q050
runner must reproduce each MeluXina production-runtime reconstructed aggregate
hash before starting integration; the reducer must subsequently match all six
actual q050 array hashes (`x`, `y`, `vx`, `vy`, `w_full`, `w_delta`) to the
reuse audit. Any mismatch stops the campaign and leaves it unresolved. It does
not trigger an automatic fresh-CIC run or resubmission.
