# Focused novelty and prior-art check

Search checked: 2026-09-14. This is a focused claim audit of primary papers and
author versions, not a proof that no uncatalogued work exists. Searches covered
PIC KDE/CV bandwidth selection, filter/shape-modified dispersion, smoothing for
grid stability, ensemble averaging, and noisy growth-rate measurements.

## Novelty matrix

| Prior result | Primary source | Distinct contribution supported here | Supporting evidence | Limitation |
|---|---|---|---|---|
| PIC deposition cast as KDE; von Mises widths selected by CV/MISE; density-noise and Landau examples | Wu and Qin, 2018, [doi:10.1063/1.5038039](https://doi.org/10.1063/1.5038039), [author version](https://arxiv.org/abs/1804.11174) | Not new: KDE framing, CV/MISE selection, or the general density bias--variance trade-off. Distinct here: an explicit protected-mode dynamical constraint evaluated against the implemented round-trip coupling and separated from the density objective. | selector audits; exact transfer audit; controlled E2 drift design | The practical end-to-end evidence is scoped; no universal selector superiority is established. |
| Noise/error decomposition, bias--variance optimisation and charge-conserving arbitrary-width particle shapes | Evstatiev, Finn, Shadwick and Hengartner, 2021, [doi:10.1016/j.jcp.2021.110394](https://doi.org/10.1016/j.jcp.2021.110394) | Not new: the statistical bias--variance framework, arbitrary-width shapes, or partition-of-unity charge conservation. Distinct here: imposing a selected-mode dynamical admissibility constraint on the exact implemented transfer and measuring the resulting rate/noise trade-off. | density/error estimator audit; transfer audit; validation matrix | This paper does not establish superiority over the variational or arbitrary-width constructions. The 2021 work must be cited separately from the authors' 2013 variational formulation. |
| Particle shapes, filtering and aliases alter discrete PIC dispersion and finite-grid instability | Huang et al., 2016, [doi:10.1016/j.cpc.2016.05.021](https://doi.org/10.1016/j.cpc.2016.05.021); Barnes and Chacon, 2021, [doi:10.1016/j.cpc.2020.107560](https://doi.org/10.1016/j.cpc.2020.107560) | Not new: smoothing-induced dispersion or the need to include discrete operators and aliases. Distinct here: using the audited modewise response as an upper bandwidth bound for a user-declared protected set, with a separately declared interior margin. | `DISPERSION_AUDIT.md`; frozen E2 design; exact transfer tables | The current E2 scan is a projected linear mechanism test, not unrestricted full-f validation. |
| Smoothing can suppress noise and grid instability; required smoothing increases with Debye under-resolution | Werner, Adams and Cary, 2025, [arXiv:2503.05123](https://arxiv.org/abs/2503.05123), [supporting data](https://doi.org/10.5281/zenodo.14984906) | Complementary rather than competing: their stability-motivated lower smoothing requirement can be paired conceptually with this work's protected-dynamics upper bound. | operator audit; protected-mode root calculations | The crossover between lower stability and upper fidelity bounds has not been measured here and remains future work. |
| Sparse-grid and SIAC approaches reduce PIC data noise while targeting retained accuracy | Muralikrishnan et al., 2021, [doi:10.1016/j.jcpx.2021.100094](https://doi.org/10.1016/j.jcpx.2021.100094); Picklo et al., 2024, [doi:10.1016/j.jcp.2024.113126](https://doi.org/10.1016/j.jcp.2024.113126) | Distinct here: the protected quantity is a self-consistent modal growth/damping rate, and the filter enters both deposition and gather. | matched-chain transfer test; rate scan | Comparisons use different objectives and campaign scopes; no global ranking is permitted. |
| Ensemble averaging reduces independent PIC fluctuations and can extend observable weak signals | Touati et al., 2022, [doi:10.1088/1361-6587/ac9016](https://doi.org/10.1088/1361-6587/ac9016), [author version](https://arxiv.org/abs/2208.06375) | Not new: ensemble averaging or its approximate square-root noise scaling. Distinct diagnostic here: quantifying how arithmetic mean-amplitude slopes reweight heterogeneous seed growth rates and separating that estimand from mean individual slopes. | full seed-share histories; effective-seed-count and mixture reconstruction tables | Exponential-mixture weighting is a mathematical consequence, not claimed as a novel physical theory. |
| PIC noise/loading can make measured instability growth differ substantially from continuum theory near sensitive regimes | Tavassoli et al., [arXiv:2112.12697](https://arxiv.org/abs/2112.12697) | Distinct here: a seed-specific finite-marker linear response is calculated for the saved loading protocol and compared one-for-one with measured E2 rates. | correlations 0.994--0.9999 in `ESTIMATOR_AND_SEED_DIAGNOSTIC.md` | The calculation is principal-mode/diagonal and the historical marker arrays were not saved. |
| Numerical benchmark rates require verified setup and numerical convergence | Finn et al., 2023, [doi:10.2140/camcos.2023.18.135](https://doi.org/10.2140/camcos.2023.18.135), [author version](https://arxiv.org/abs/2303.12620) | Not new: numerical verification of Landau benchmarks. Contribution here is procedural: continuum, principal-CIC and grid/time-discrete references are reported separately and model accuracy is not conflated with rate preservation. | dispersion/root convergence tables; validation matrix | Several compensated predictions remain outside measured intervals even when the 10% preservation criterion passes. |

## Supported novelty statement

The strongest defensible distinction is a **model-derived, implementation-
audited protected-mode bandwidth rule with an explicit interior margin**, and
its test under a controlled equilibrium scan where initial density and fixed
filter transfer stay constant while dynamical sensitivity changes. Exact
filter-family transfers are treated as design coordinates rather than replaced
by Gaussian proxies.

The literature checked already establishes KDE-based PIC bandwidth selection,
smoothing-induced dispersion, discrete alias effects, ensemble averaging and
smoothing for numerical stability. Those ingredients must not be presented as
new individually.

## Fair-comparison consequence

Only comparisons with matching estimands and scopes should be ranked. The
held-out Gaussian selector comparison is a finite candidate-family result; the
Wu--Qin implementation is a stated proxy where its particlewise adaptive
algorithm is not reproduced. The compensated E2 results establish selected
rate trade-offs under the projected linear setup, but compatible noise at the
same compensated settings is not available. Therefore the current evidence
does not establish universal, global or end-to-end optimality of Gaussian or
compensated filtering.
