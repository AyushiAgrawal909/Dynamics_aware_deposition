# Paper case: oblique two-stream, v_b = 0.80, protected mode (12,1)

| file | content |
|---|---|
| `two_stream_model.yaml` | reference-free selection (model objective, PPC 32) |
| `two_stream_snapshots.yaml` | the paper's reference-assisted selection (seeds 100–104 select, 105–109 evaluate) |
| `paper_spectral_stats.npz` | per-realisation spectral sums of the ten paired reconstruction runs (334 kB) |
| `make_snapshots.py` | regenerates the full snapshots with the repository's PIC code |

```bash
dabs select two_stream_snapshots.yaml -o out_snapshots    # selects binomial (4,8); held-out 76.44% / 37.60%
dabs select two_stream_model.yaml -o out_model            # same choice from the noise model alone
```

The snapshots are paired runs, with the same random stream for both members
of each pair:

- **Production run:** PPC 32.
- **Reference run:** PPC 1024.
- **Common settings:** unfiltered CIC coupling, Δt = 0.05, 80 steps, CIC
  densities saved every second step on t ∈ [0.5, 4] (36 times),
  displacement amplitude 10⁻³ on mode (12,0), seeds 100–109.

To regenerate them, run the following from this directory (about 2 minutes
per seed):

```bash
python make_snapshots.py --seeds 100-109 --out snapshots.npz
dabs stats snapshots.npz --L 64 16 -o paper_spectral_stats.npz
```

The regenerated meshes reproduce every archived per-seed density and field
error of the paper's 115 candidates to within 2e-15 relative.
