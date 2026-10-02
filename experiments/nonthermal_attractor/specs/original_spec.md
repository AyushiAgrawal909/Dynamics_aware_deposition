# Codex task: test a nonthermal attractor in nonlinear electrostatic plasma relaxation

## Objective and first-pass scope

Implement and run a falsifiable screening study of this hypothesis:

> Freely relaxing nonlinear electrostatic plasma dynamics may approach an intermediate self-similar state in a specified energy sector. If that state attracts a class of preparations, early data may predict later tail development, and perturbations may reveal relaxation exponents.

This is an unproved hypothesis. Do not assume a tail, a scaling regime, universal exponents, or a nonthermal fixed point exists. The useful outcome is a reproducible decision supported by data, including an inconclusive or negative result.

Start now: inspect the actual repository, implement the required diagnostics and analysis, validate them, and run the affordable pilot. Do not stop at a proposed plan. The first pass is a screening experiment, not a publication claim or an attempt to complete the full research program.

Use HEXAPIC where feasible. If its dependencies prevent local execution, complete the analysis implementation and use a clearly separate, validated small 1D1V Vlasov–Poisson reference for the screening calculation. Report precisely which backend ran; never present reference-code output as HEXAPIC output. Prepare reproducible HEXAPIC configurations and commands for the remaining runs.

## Inputs and actual code baseline

Use the repository selected in this Codex session. If only an archive is available, locate hexapic-main.zip (it may be named 01-hexapic-main.zip), list its contents, and extract safely into a new directory without overwriting an existing checkout.

Consult these attachments when available:
- HEXAPIC inspecytion.docx: historical source mapping.
- claude_code_prompt_kde_hexapic.md and claude_code_prompt_adaptive_sigma.md: earlier proposed extensions.
- codex_prompt_landau_damping-2.md: earlier benchmark proposal.
- Wu & Qin, arXiv:1804.11174: smoothing background.

Their descriptions and requested modifications are not proof that features are implemented.

The inspected archive has source/particles.cpp, hexapic.hpp, field_solver_hypre.cpp, diagnostics.cpp, read_input_file.cpp, and main.cpp. It implements CIC scatter and gathering from nodal potential, has three velocity components, and uses HYPRE and openPMD/ADIOS2. The archive does not include the deposition_action_library files assumed by the earlier KDE prompt. Its main.cpp writes argv[1] + ".bp4"; do not assume the historical "-out" option exists. Recheck all of this against the working checkout.

Do not reuse the uploaded discharge input as this experiment: it contains walls, sources, collisions, and imposed fields. Build a new closed-system input.

Preserve user edits and previous results. Use an isolated branch/worktree when git is available; otherwise isolate the new experiment. Avoid unrelated refactoring.

## 1. Audit the physical and numerical model

Read applicable AGENTS.md instructions and map the complete particle-to-field-to-particle step.

Write reports/CODE_AUDIT.md with evidence from actual filenames and functions:
- Spatial and velocity dimensions, physical units, geometry factors, macro weights, charge, and mass.
- Electron load options, RNG seeding, quiet-start support, and whether multiple loads represent one species.
- The actual Poisson, field-gradient, scatter, and gather operators. Do not call the HYPRE path spectral.
- Periodic boundary implementation, unique mesh nodes versus duplicated endpoints, MPI ownership and halos.
- Periodic Poisson neutrality, constant-potential nullspace treatment, and any background charge correction.
- Collision, source, heating, injection, and wall switches.
- Particle and field output, including masses/weights absent from particle records.
- Time staggering of fields and particle velocities, and the physical time represented by each diagnostic.
- Existing tests, build dependencies, run options, and restart/checkpoint availability.

For the closed baseline, set external magnetic field to zero, use periodic boundaries, disable forcing, injection, physical collisions and walls, and use an explicit immobile neutralizing background for the electron-only model. Verify neutrality rather than silently deleting an unexplained net charge.

Implement only missing features necessary for this experiment. Distinguish any solver/coupling correction from an analysis-only change. Verify relevant existing tests after modifying the solver or particle mover.

## 2. Ground the candidate contribution in literature

Create reports/PRIOR_ART.md using accessible primary papers, with exact links, versions, and sections read. Start with:
- Ewart et al., "Relaxation to universal non-Maxwellian equilibria in a collisionless plasma": https://arxiv.org/abs/2409.01742
- Banik, Bhattacharjee & Sengupta, "Universal non-thermal power-law distribution functions from the self-consistent evolution of collisionless electrostatic plasmas": https://arxiv.org/abs/2408.07127

Check the published versions, newer versions, and closely related work on Vlasov self-similarity, phase-space cascades, electron-hole coarsening, and stability of nonthermal fixed points. Bound this search; do not let it replace implementation.

The first paper studies long-time nonlinear beam relaxation and contrasting electron-only/pair-plasma outcomes. The second already derives self-similar solutions in a driven quasilinear model, including Appendix C.2. Therefore, neither "plasma tails are universal" nor "self-similarity applied to plasma" is an available novelty claim.

Compare prior work against the specific candidate: attraction and predictive transient scaling in freely relaxing nonlinear dynamics, with identified numerical and invariant constraints. If sources are unavailable, mark the comparison incomplete rather than inventing conclusions.

## 3. Choose a discriminating physical experiment

First use an effectively 1D electrostatic model. A thin 2D box is not automatically 1D: either enforce y-independent evolution consistently or measure transverse fields/modes and identify it as a 2D experiment when those become significant. Do not reuse a linear mode-projected delta-f runner for nonlinear tail formation.

Start from symmetric counterpropagating electron beams with finite width and a neutralizing background. Prefer a two-waterbag case connected to the nonlinear literature, and use a smooth two-Gaussian preparation as an additional physical test if affordable.

For a waterbag with two intervals centered at ±u, each of full width Delta_v:
- Use f0 = n0/(2 Delta_v) on those intervals.
- Its second moment is u^2 + Delta_v^2/12.
- Avoid singular zero-width beams.

For a Gaussian mixture with equal weights, means ±u and standard deviation w:
- Its second moment is u^2 + w^2.

Use at least three physical preparations for an attraction test. Match density, total momentum, and the intended energy budget where possible, documenting seed-field energy and all remaining differences. Different random seeds of one preparation only estimate sampling uncertainty.

Matching these moments does not match the Vlasov Casimirs. Record the differing phase-space level content and restrict any universality claim to the measured sector and basin. Add a rearrangement or perturbation test preserving level content when practical.

Determine the unstable wavelengths from the actual initial distribution, choose a box supporting multiple unstable wavelengths and hole mergers, and document the finite-size limitation. Resolve the relevant distribution/field scales and prevent finite-grid instability from determining the result. Choose dt using the actual pusher, boundary-routing constraints, and fastest plasma/particle scales.

Use a small deterministic seed spectrum common to the comparisons, with recorded amplitudes and phases. Label noise-seeded runs separately.

Use one stable Maxwellian control to estimate numerical heating/relaxation under comparable numerical settings. Optional electron–positron comparison belongs to a distinct physical model, with its plasma-frequency normalization stated; it is not another random preparation of the electron-only model.

Run a timing smoke test first. Use a default cumulative local pilot budget of about 30 minutes, adjustable by an existing user budget. Within it run the smallest defensible nonlinear case and useful refinements. Generate longer-run and HPC configurations instead of silently launching a large campaign. A short run that cannot reach tail evolution is INCONCLUSIVE, not a rejection.

## 4. Implement diagnostics with an explicit data contract

Use optional diagnostics that do not change baseline particle dynamics. Prefer MPI reductions of histograms and moments over gathering every particle. Save occasional phase-space snapshots for structure diagnostics.

Every run must record:
- Configuration, code revision/archive hash, backend, units and normalization, dimensionality, seed, MPI size, weights, species masses/charges, solver tolerances, and numerical settings.
- Exact diagnostic times and staggering convention.
- Number, total charge, momentum, kinetic energy, discrete field energy, total energy and its drift.
- Velocity distribution; energy distribution; x–vx phase-space density; selected E-mode amplitudes and spectrum.
- Particle-count and sampling information, histogram bin edges, underflow/overflow counts and energies.
- Wall time and output size.

Define N(epsilon,t) = dN_physical/d epsilon, or its explicitly stated density version. For particles in bin b,
    N_b = sum_p_in_b(weight_p) / Delta_epsilon_b.
Counts per logarithmic bin obey dN/d ln(epsilon) = epsilon N(epsilon). Do not plot them as the same quantity.

Use physical species mass in each particle energy and macro weight only in summation. State the frame: laboratory or the fixed initial center-of-momentum frame. For effectively 1D comparison, use epsilon_x = m vx^2/2 as the active-energy diagnostic and also monitor all three velocity energies. Do not compare a 1D energy distribution directly with total 3V energy without accounting for spectator components.

Choose common bin edges before inspecting the held-out data. Include the low-energy part when checking total moments; use strictly positive energies for log-tail analysis. Do not clip energetic particles or renormalize each snapshot to improve collapse.

Check histogram number/energy moments against direct particle sums, quantify finite-bin moment error, and record weighted effective counts. Tail uncertainties must acknowledge non-Poisson quiet loading and correlations. Independent run seeds are the primary resampling unit when available.

Define a fixed tail threshold from initial scales or training data, freeze it before forecasting, and report:
    Q_N(t) = integral_tail N d epsilon
    Q_E(t) = integral_tail epsilon N d epsilon
    tail_energy_fraction = Q_E / K_total.
Also record identifiable moving features, such as a supported high-energy quantile, crossover, or cutoff. Do not use a single largest particle energy as a robust front.

For fields, avoid double-counting periodic endpoints or MPI halos. Use energy consistent with the implemented discrete operator and document diagnostic synchronization.

Track a few structure quantities—hole count, separation scale, potential depth, spectral scale—only with explicit detection rules and resolution checks. A trapped fraction requires a justified local frame and separatrix estimate; an arbitrary low-velocity cut is insufficient.

If measuring entropy, phase-volume content or C2, state the phase-space reconstruction and smoothing scale. Fine-grained Vlasov Casimirs are conserved; reconstructed/coarse-grained estimates need not be. Numerical drift or binning is not evidence of physical invariant breaking.

## 5. Implement identifiable scaling inference

Use the dimensionally explicit hypothesis
    N(epsilon,t) = A_star s^alpha F(s^beta epsilon/epsilon_star),
    s = (t - t0)/t_star > 0,
with all reference scales recorded.

For an entire distribution with finite moments and scaled support:
    number ∝ s^(alpha-beta),
    kinetic energy ∝ s^(alpha-2 beta).
Constant number and kinetic energy force alpha = beta = 0. Particle kinetic energy can exchange with fields. A nontrivial tail sector can exchange particles/energy with the core. Measure those budgets rather than imposing whole-distribution relations on a partial sector.

Fit only training snapshots in a physically defined post-saturation interval:
- Pick saturation/onset and candidate intervals using training data only.
- Fix t0 to a justified event or profile over a bounded range; show its uncertainty.
- Use a common scaled overlap region, conservative rebinning or justified interpolation, and a quantitative loss with uncertainty weighting.
- Penalize or constrain collapsing a vanishingly small overlap region.
- Preserve normalization; do not allow arbitrary amplitude rescaling separately at every time.
- Report alpha–beta covariance, profile loss contours, boundary-hitting fits, and window/bin/threshold sensitivity.
- Resample independent runs where possible; do not treat every correlated snapshot as an independent realization.

Critical identifiability test: if F(x) ∝ x^(-p), then only alpha - p beta is measured by a featureless power-law tail. A stationary epsilon^(-2) tail can appear to collapse for every alpha = 2 beta. Detect and report this degeneracy; do not return falsely precise exponents. Require curvature or a moving feature for separate alpha and beta.

For scaled bins moving as epsilon_(1,2) = epsilon_star x_(1,2) s^(-beta), compare measured sector moments with the appropriate powers s^(alpha-beta) and s^(alpha-2 beta). Account for flux through moving boundaries when differentiating them. Fixed-threshold tail moments have different boundary dependence.

For a fitted tail near p = 2, explicitly track both cutoffs because the tail energy depends logarithmically on their ratio.

## 6. Forecast without refitting and compare alternatives

Before fitting, save a machine-readable protocol defining training/held-out windows, features, bounds, thresholds, model comparison, and numerical-quality gates.

After fitting on early training data, freeze alpha, beta, t0, scales, the template and analysis choices. For t_ref in training:
    N_pred(epsilon,t) = r^alpha N(r^beta epsilon,t_ref),
    r = (t-t0)/(t_ref-t0).
Mask energies requiring information outside the training support; do not extrapolate the unknown template silently.

Measure held-out distribution error, feature motion, tail number and energy errors, uncertainty and forecast horizon. Evaluate individual seeds as well as ensembles.

Compare against:
1. A stationary distribution or stationary power-law tail.
2. A thermal distribution matching allowed measured/training energy information.
3. A one-scale heating model.
4. A relevant quasilinear diffusion baseline when its assumptions can be justified.

Allow each predictive baseline the same training information. A Maxwellian constructed using future kinetic energy is an oracle diagnostic, not a competing early-data forecast.

Reserve at least one physical preparation for an untouched test if the pilot has enough data. Forecasting more snapshots of one run is not evidence of universality. State when the data are too short, sparse, or noise-dominated for comparison.

## 7. Distinguish self-similarity, attraction, and stability

A successful single-trajectory collapse supports only approximate self-similarity.

For attraction, test whether different physical preparations approach the same scaled function within an explicitly defined class. Record permissible invariant-based normalizations in advance. Do not freely warp each trajectory to force agreement.

Examine
    N_tilde(x,s) = F(x) + sum_j c_j s^(-lambda_j) phi_j(x).
A positive real lambda describes algebraic decay in physical time and exponential decay in ln(s). It is distinct from alpha and beta.

An empirical residual-decay fit is a diagnostic, not a derived eigenvalue spectrum. Separate time-shift, normalization and exponent-fit tangent directions from physical residuals. If data permit, use controlled small perturbations and test whether decay exponents persist across them.

Do not assert that N alone obeys a closed Markovian equation. A stability derivation needs justified dynamics for the distribution, fields and relevant correlations. If a closure cannot be supported, write the unresolved terms and proposed tests instead of fabricating it.

## 8. Numerical verification and independent comparison

Use meaningful tests of the science and analysis:
- Particle weights, histogram normalization, moment accounting and MPI reductions.
- An analytic Maxwellian energy distribution in the actual velocity dimension.
- A synthetic moving-cutoff distribution with known alpha/beta and held-out forecasts.
- A stationary power law that must be flagged as unidentifiable.
- Ordinary heating or non-self-similar data that must not produce a false attractor claim.
- Required existing field/pusher/coupling tests if those implementations change.

Use synthetic data only to validate analysis; keep it separate from plasma results.

For at least the most promising physical case, prepare and run affordable independent refinements of particle number, dx, dt, box size and analysis binning. If present, smoothing is a separate numerical arm, not the main physical control. Fixed CIC is the unsmoothed baseline; audit any changed deterministic interaction.

Prefer at least three independent seeds for preliminary PIC uncertainty, and explicitly label smaller samples exploratory. Long-time conclusions require a numerical relaxation/noise assessment; agreement of linear rates and energy alone is insufficient.

Use an independent 1D1V Vlasov–Poisson reference where justified. Validate charge/Poisson sign and normalization, a linear benchmark and relevant nonlinear behavior, resolution convergence and velocity-domain boundaries. Monitor recurrence, filamentation, numerical diffusion and Casimir drift. Call a finite-grid Vlasov run a reference calculation, not exact collisionless truth.

If weak physical collisions are added later, choose a documented operator and the same target in PIC and Vlasov. Examine the order of long-time and weak-collision limits. Do not identify unspecified numerical diffusion with a physical collision frequency.

## 9. Deliverables and completion criteria

Keep the new experiment under experiments/nonthermal_attractor/ or the closest existing repository convention. Suggested contents:
    README.md
    configs/
    diagnostics/
    analysis/
    reference/           # only if needed
    tests/
    scripts/
    reports/
    results/              # small pilot artifacts; large outputs separate

Provide working configuration-driven commands for:
- Smoke/quality checks.
- One nonlinear run.
- Pilot matrix.
- Scaling fit on training data only.
- Held-out forecast evaluation.
- Plot and report generation.
- Production/HPC runs, with environment-specific placeholders explicitly marked.

Use existing dependency management; avoid a new heavy framework. Include manifests linking each plotted curve to backend, configuration and run.

Required outputs:
- CODE_AUDIT.md and PRIOR_ART.md.
- PILOT_PROTOCOL.json/yaml and machine-readable fit/forecast/quality summaries.
- Plots of raw energy distributions with count support; feature motion; collapse; alpha–beta loss landscape; held-out forecasts; number/energy budgets; and numerical sensitivity when data exist.
- PILOT_REPORT.md explaining what ran, actual parameters, uncertainties, failures, compute cost and limitations.
- NEXT_STEPS.md with the smallest decisive next experiment and its cost.

Assign separate statuses to numerical validity, tail detection, scaling identifiability, forecasting, attraction and physical mechanism:
    SUPPORTED / REJECTED / INCONCLUSIVE / NOT_TESTED.
A broad research hypothesis is not rejected merely because one short preparation fails. An insufficient runtime, sampling, overlap or convergence study is INCONCLUSIVE.

Only recommend expanding the campaign if the evidence supports a physical evolving feature, identifiable scaling, forecasts that improve on relevant baselines, and numerical robustness. If evidence fails, explain the failed prediction and avoid rescuing it by inspecting and retuning held-out data.

Finish with a concise account of what changed, which commands actually ran, what the data support, what remains untested, and the next experiment. Do not claim PRL/Nature readiness or priority from a collapse plot.

Begin with the repository audit and then continue through all feasible first-pass implementation and pilot work.

