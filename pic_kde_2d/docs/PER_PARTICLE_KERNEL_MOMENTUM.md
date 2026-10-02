# Momentum Conservation with Per-Particle Kernel Widths

## Abstract Result

Different particle widths do **not** break exact momentum conservation by
themselves. If particle `i` deposits and gathers with the same grid shape

```text
S_i = G_{H_i} * W_CIC,
```

then pairwise forces remain antisymmetric even when `H_i != H_j`.

The actual requirements are:

1. Each particle's gather is the exact transpose of its own deposition.
2. The grid map from charge density to each electric-field component is
   skew-adjoint in the grid inner product.
3. The same shapes are used during the deposit and gather forming one force
   evaluation.

The periodic spectral Poisson/gradient operator in the Python `Grid2D` model
satisfies condition 2 up to floating-point roundoff. This theorem does **not**
by itself prove momentum conservation in production HEXAPIC.

## Production HEXAPIC Audit

The production chains are different from the abstract field-interpolation
model:

```text
KDE/direct Gaussian:
particle shape -> rho -> five-point PFMG potential -> derivative(shape)*phi

CIC plus post-filter:
CIC -> symmetric grid filter on rho -> five-point PFMG potential -> CIC gather
```

For KDE, `gather_field_kde` differentiates the normalized particle weights and
interpolates potential. This is a variational potential gather, not the
`E_i = s_i^T E` gather in the skew-adjoint proof. For the post-filter actions,
the periodic convolution is real and symmetric, but it is applied only to
deposited charge; the gather falls back to ordinary CIC. Consequently that
branch is not `S^T A S`, despite using a symmetric filter.

The HYPRE matrix is a symmetric five-point periodic Laplacian. The current
production PFMG tolerance is `1e-9`. A finite-iteration PFMG application is not
an exact inverse and is not guaranteed to be a symmetric linear map because
iteration count, stopping, and the previous potential used as the initial
guess can depend on the right-hand side.

The opt-in audit in `hexapic_momentum_audit/` measures the actual gathered
quantity `sum_p(q_p E_p)` for equal and unequal KDE widths at tolerances
`1e-5` and `1e-9`. Exact production momentum conservation may be claimed only
if that normalized total force is at roundoff and remains there under unequal
widths.

## Continuum Proof

Let `K_alpha(r) = -partial_alpha G(r)` be one component of the electrostatic
Green-function force kernel. For a periodic, translation-invariant Poisson
operator,

```text
K_alpha(-r) = -K_alpha(r).
```

The force on particle `i` due to particle `j` is

```text
F_{i<-j,alpha}
  = q_i q_j integral dx dy
      S_i(x-x_i) K_alpha(x-y) S_j(y-x_j).
```

Exchange `i <-> j` and the dummy variables `x <-> y`:

```text
F_{j<-i,alpha}
  = q_i q_j integral dx dy
      S_i(x-x_i) K_alpha(y-x) S_j(y-x_j)
  = -F_{i<-j,alpha}.
```

No step assumes `S_i = S_j` or `H_i = H_j`. The self-force also vanishes:

```text
F_{i<-i,alpha} = -F_{i<-i,alpha} = 0.
```

Therefore `sum_i F_i = 0` exactly in exact arithmetic.

## Discrete Grid Proof

Let `s_i` be the normalized grid-weight vector for particle `i`, including its
CIC shape and its own Gaussian convolution. Let `Delta V = dx*dy`. Matched
deposition and gather are

```text
rho = (1/Delta V) sum_j q_j s_j,
E_i = s_i^T E,
E = A rho,
```

where `A` maps grid density to one electric-field component. Define

```text
d = sum_i q_i s_i.
```

The total force is

```text
F_total = sum_i q_i E_i
        = (1/Delta V) d^T A d.
```

If `A^T = -A`, then

```text
d^T A d = (d^T A d)^T = d^T A^T d = -d^T A d,
```

so `F_total = 0`. Equivalently, the pair force is

```text
F_{i<-j} = (q_i q_j/Delta V) s_i^T A s_j,
F_{j<-i} = (q_i q_j/Delta V) s_j^T A s_i = -F_{i<-j}.
```

Again, the columns `s_i` may all be different.

For periodic spectral electrostatics,

```text
A_alpha(k) = -i k_alpha / |k|^2,
```

which is odd and purely imaginary, hence skew-adjoint. A periodic centered
finite-difference gradient combined with a symmetric, commuting Poisson inverse
has the same property.

## Why The Variational Gather Needs A Different Proof

Let `G = L^{-1}` be the symmetric discrete Poisson Green matrix and let
`s_i(x_i, H_i)` be particle `i`'s sampled shape. With

```text
d = sum_i q_i s_i,
U = (1/(2 Delta V)) d^T G d,
```

the production KDE force, with `H_i` held fixed during one evaluation, is

```text
F_i = -(q_i/Delta V) (partial s_i / partial x_i)^T G d.
```

Therefore

```text
sum_i F_i = -dU/da at a=0,
```

where `a` translates every particle position by the same continuous amount.
This vanishes only if the *discrete* pair energy is invariant under continuous
simultaneous translation. A fixed nodal grid guarantees invariance under
integer-cell shifts, but generally not under arbitrary sub-cell shifts.
Hence a symmetric Poisson solve plus variational gather proves energy
consistency for fixed shapes; it does not prove exact particle momentum.

Unequal `H_i` does not automatically create an error in a truly
translation-invariant continuum convolution, but it also does not repair the
lost continuous symmetry of a grid-sampled variational scheme. If `H_i`
depends on position, the complete derivative additionally contains
`(partial s_i/partial H_i)(partial H_i/partial x_i)`; the current cellwise
bandwidth gather omits that term.

## What Actually Breaks Momentum Conservation

Per-particle adaptivity causes momentum error only when it breaks one of the
operator identities above. Examples:

- Deposit uses `s_i`, but gather uses a different vector `g_i`.
- `H_i` is recomputed between deposit and gather.
- Deposit and gather normalize or truncate the particle stencil differently.
- The field operator is not skew-adjoint, for example because of boundaries,
  one-sided gradients, nonuniform-grid interpolation, AMR transfers, or an
  inexact/non-symmetric field solve.
- Parallel accumulation and iterative-solver tolerances introduce roundoff or
  residual errors.

For the field-interpolation chain, if `g_i = s_i + e_i`, define

```text
d = sum_i q_i s_i,
e = sum_i q_i e_i.
```

For a possibly imperfect field operator `A`,

```text
F_total
  = (1/(2 Delta V)) d^T (A + A^T) d
    + (1/Delta V) e^T A d.
```

Therefore

```text
|F_total|
  <= ||A + A^T|| ||d||^2 / (2 Delta V)
     + ||e|| ||A|| ||d|| / Delta V.
```

This separates field-operator asymmetry from deposit/gather mismatch and gives
a directly measurable violation bound. It applies directly to an `S^T A S`
chain. The production variational gather must instead be assessed through the
translation derivative above plus iterative-solver residual error.

## Important Energy Caveat

This result is about total momentum, not total energy. If `H_i` changes with
particle position, density, or time, a Hamiltonian derivation generally
produces additional shape-derivative terms (often called `grad-h` terms).
Omitting those terms can cause energy drift even though the instantaneous total
force still sums to zero. Thus:

```text
matched per-particle shapes + skew field operator => exact momentum
```

does not imply exact energy conservation for adaptive `H_i`.
