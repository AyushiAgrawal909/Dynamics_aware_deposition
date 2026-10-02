# Adaptive-Width Production Conservation Gate

## Decision

Neither current production chain supports a claim of exact particle momentum
conservation for adaptive widths.

### CIC plus post-filter

The implemented chain is

```text
C -> F -> Poisson -> C^T,
```

not the matched chain `C -> F -> Poisson -> F^T -> C^T`. Even though `F` is a
real symmetric periodic convolution, it appears only on the deposition side.
The resulting force operator is structurally unmatched before spatial
adaptivity is introduced. This branch is not an acceptable foundation for an
adaptive-width conservation claim.

### KDE variational chain

The implemented chain deposits a normalized sampled kernel, solves for nodal
potential, and gathers with the derivative of the same kernel. For fixed
bandwidth this is the derivative of a discrete potential energy with respect to
particle position while holding bandwidth fixed. It is the preferable
foundation for adaptivity, but it has two limitations:

1. A fixed grid has only integer-cell translation symmetry. Exact total
   particle momentum would require continuous sub-cell translation invariance
   of the discrete pair energy.
2. If `H_i = H_i(x_i, state)` varies with position, the complete force contains
   a grad-h term

```text
(partial s_i / partial H_i) (partial H_i / partial x_i),
```

   which the current gather omits. A cellwise controller also makes `H_i`
   discontinuous at tile/cell boundaries.

## Phase-1 Consequence

Phase 1 must not use “matched bandwidth” as a conservation proof. Before an
adaptive selector is promoted, it must pass all of the following:

1. PFMG converges to a measured residual under both warm and cold starts.
2. Instantaneous normalized total electric force is bounded for equal and
   unequal widths.
3. Integrated momentum drift is bounded over a complete benchmark and tracks
   the integrated electric impulse.
4. Energy drift is measured while particles cross bandwidth boundaries.
5. Either the missing grad-h force is implemented, or the method is explicitly
   classified as approximate/non-Hamiltonian with empirical error bounds.

Until these gates pass, the production-safe choices are fixed global width or
diagnostic-only adaptive reconstruction. Spatially adaptive post-filtering is
not the fallback because its gather is already unmatched.

