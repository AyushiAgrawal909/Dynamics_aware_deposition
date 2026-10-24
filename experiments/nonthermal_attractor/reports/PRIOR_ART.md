# Prior art (bounded check, T1 section 8). Status: INCOMPLETE

This note sets the novelty boundary for the programme. **It is marked incomplete.** From this environment the arXiv abstract pages could be fetched only partially; full texts, published versions and the exact run parameters were not read. Every item below says what was verified and what still needs checking by hand before T2's protocol is frozen.

| Work | What it claims (as used here) | Verified here | Still to check |
|---|---|---|---|
| Ewart et al., arXiv:2409.01742 (published PNAS 2025, per the 2 Oct 2026 review) | Collisionless relaxation to Lynden-Bell-type equilibria. In 1D electrostatic beam instabilities, a universal ε⁻² tail for electron-only two-stream. The pair case "freezes" in its ground state. Names the missing piece: a theory of the time evolution of the waterbag content ρ(η). | The arXiv page exists and concerns Lynden-Bell equilibria of collisionless plasma relaxation. The author list on the fetched excerpt (R. J. Ewart, M. L. Nastac, …, A. A. Schekochihin) **differs from the list in the earlier review**. | Exact author list and version. The PNAS DOI. **The run parameters T2 needs: cells, domain length, ppc, beam widths, run length, and the tail-formation time.** The bin-insensitivity procedure for ρ(η). |
| Banik, Bhattacharjee & Sengupta, arXiv:2408.07127 | Universal power-law distribution functions from the self-consistent evolution of *driven* collisionless electrostatic plasmas (quasilinear). Appendix C.2 gives self-similar solutions, and it also obtains an E⁻² tail. | The abstract concerns self-consistent collisionless relaxation of driven electrostatic plasmas and power-law (solar-wind-like) distributions. | Version, Appendix C.2 exponents, and whether the self-similar variables coincide with (α, β) here. |
| Yoon et al. (kappa-tail, spontaneous-scattering school) | Discrete-particle spontaneous scattering produces suprathermal tails; turbulent equilibrium fixes κ = 9/4. | Not fetched | Exact citations |
| Park, Moore & Baalrud (2026) | PIC numerical thermalisation time vs macroparticle weight; validates Okuda–Birdsall–Langdon theory in 1D, 2D and 3D. | Not fetched | Citation and the 1D scaling that T2's stable control should reproduce |
| Self-similarity and nonthermal fixed points in Vlasov tail formation; phase-space cascades; electron-hole coarsening | | Not searched in this pass | A bounded search before any novelty statement |

## Consequences already built into T1/T2

- **Neither "plasma tails are universal" nor "self-similarity applied to plasma" is available as a novelty claim.**
- **A stationary ε⁻² tail cannot discriminate between mechanisms.** Two different theories predict it, and it is the degenerate case of the scaling fit. Synthetic case S3 verifies that the analysis reports only α − 2β there.
- **What would remain novel, pending the checks above:**
  - regulator-converged moving features (T2);
  - predictive transient scaling with identified exponents;
  - attraction in freely relaxing dynamics (T3);
  - and, on the negative side, a demonstration that PIC tails or onsets track ln N or N.
