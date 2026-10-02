# Novelty Verification: MISE-vs-Dynamics Selection Conflict

## Gate Result

Clean. I did not find the exact selection conflict in the checked prior work.

The closest papers cover adjacent pieces:

1. Wu & Qin: density-estimation framing, MISE, and leave-one-out CV bandwidth
   selection for KDE-PIC.
2. Werner, Adams & Cary 2025: smoothing for grid-instability suppression and
   particle-noise mitigation, including dispersion/stability and noise/heating
   characterization.
3. Barnes & Chacon 2021: finite-grid/aliasing stability of energy-conserving
   PIC, including cold/warm drifting plasmas and finite-grid stability
   thresholds.

None of these papers appears to ask:

```text
Can a density-optimal MISE/CV smoothing choice be dynamically unsafe because it
suppresses the physical mode whose growth/damping rate the simulation is meant
to reproduce?
```

That is the novelty claim supported by the E2 two-stream result.

## Source Audit

### Wu & Qin KDE-PIC

Wu & Qin define the density error by MISE and choose kernel width by a
leave-one-out CV proxy for MISE. They explicitly present the CV function as the
optimization target for kernel width. This is the density-estimation selector
whose failure mode we test.

Their validation cases are Langmuir-wave and Landau-damping simulations. I found
no two-stream, beam-plasma, or other growing-instability benchmark in the paper.
This matters because our failure mode appears in a growing, near-marginal
instability: the MISE/CV selector can choose smoothing that is excellent for
density reconstruction but suppresses the physical growth rate. Wu & Qin do show
improved linear damping-rate measurement for Landau damping, but that does not
test whether a density-optimal width preserves a growing-mode margin.

What they do not provide is a dynamics-side admissibility constraint such as:

```text
choose the largest smoothing width subject to gamma_sigma / gamma_0 remaining
inside a prescribed tolerance.
```

### Werner, Adams & Cary 2025

Werner et al. study smoothing as a way to suppress explicit-PIC grid instability
and noise when the Debye length is under-resolved. They solve/analyze numerical
dispersion, characterize electric-field noise, velocity diffusion, and
unphysical energy exchange as functions of smoothing, cell size, and particles
per cell.

This is close in spirit because it treats smoothing as dynamically relevant,
not just cosmetic density reconstruction. However, the objective is stability
and noise reduction for under-resolved Debye-length simulations. The paper does
not pose or test a density-MISE/CV bandwidth selector against preservation of a
physical instability growth rate.

### Barnes & Chacon 2021

Barnes & Chacon analyze finite spatial-grid effects and aliasing stability in
energy-conserving PIC. They compare energy-conserving and momentum-conserving
PIC stability for stationary and drifting plasmas and discuss finite-grid
instability thresholds. They also discuss smoothing/filtering in the context of
reducing growth rates or improving stability.

This is relevant background for aliasing/stability, but it does not formulate a
density-estimation selector, MISE/CV optimum, or the conflict between such a
selector and physical mode fidelity.

## Differentiation Paragraph

Prior KDE-PIC work selects particle shape width by minimizing density-estimation
error, typically through a MISE or leave-one-out cross-validation objective,
while finite-grid PIC studies analyze how smoothing, particle shapes, and
energy-conserving discretizations affect aliasing instabilities, noise, and
grid-heating stability. These literatures leave open a distinct question: whether
a smoothing width that is optimal for density reconstruction is admissible for
live dynamics. We show that it need not be. In a near-marginal two-stream test,
both high-ppc-reference MISE and a Wu-Qin-style leave-one-out CV selector choose
large smoothing widths that strongly reduce density error but erase the physical
growing mode. The remedy is to treat growth/damping fidelity as a hard
dynamics constraint and maximize noise reduction only over the feasible set.
This turns smoothing selection from a density-estimation problem into a
constrained dynamics-preservation problem.

## Scope Boundary

Do not claim:

```text
No one has studied smoothing effects on PIC dynamics.
```

That would be false. Werner et al. and Barnes-Chacon both do.

Do claim:

```text
The density-selector versus physical-mode-fidelity conflict, and the resulting
constrained selector, is not posed in these prior works.
```

## References Checked

- Wentao Wu and Hong Qin, "Reducing Noise for PIC Simulations Using Kernel
  Density Estimation Algorithm", Physics of Plasmas 25, 102107 (2018).
  https://arxiv.org/abs/1804.11174
- Gregory R. Werner, Luke C. Adams, and John R. Cary, "Suppressing grid
  instability and noise in particle-in-cell simulation by smoothing" (2025).
  https://arxiv.org/abs/2503.05123
- L. Barnes and L. Chacon, "Finite spatial-grid effects in energy-conserving
  particle-in-cell algorithms", Computer Physics Communications 258, 107560
  (2021). https://arxiv.org/abs/1910.10833

This is the defensible novelty.
