# MeluXina recovery, 2026-09-09--10

This directory contains byte-for-byte copies of submitted Paper 1 campaign
directories recovered from MeluXina.  The originals remain in place on the
cluster.  The copy commands and subsequent hashes are recorded in the Paper 1
reproducibility audit.

The directory also contains newly executed, pre-registered validation outputs
copied from `/mnt/tier2/users/u103576/paper1_validation` after their dependent
reducers completed.  These are distinguished by their Slurm job identifiers;
they are not historical-result recoveries.  As of 2026-09-10 this includes
`weak_landau_duration_pilot_5195658` and
`weak_landau_lscv_exact_5195682`, plus
`e2_compensated_corner_5195640`. The LSCV directory contains 20
independent-seed histories (ten paired CIC and exact-width arms), run-parameter
records and the unmodified reducer products. The E2 directory contains 50
histories (ten seeds for CIC and four compensated candidates) and the direct
binding-corner reduction. Local and remote per-file SHA-256 values match; the
E2 directory's aggregate manifest hash is
`1376b97196782c33193fdb6c36f400579ffbab5bc4aa604bc98c57908f006b7f`.
