"""T2 backends: 1D1V spectral PIC (``pic1d1v``) and semi-Lagrangian Vlasov
reference (``vlasov1d1v``), linear theory (``theory``), the shared diagnostic
writer (``diagnostics``) and the configuration-driven runner (``run``).

Normalisation: omega_pe of the total electron density n0 = 1, epsilon_0 = 1,
electron charge -1 and mass 1, immobile neutralising background +n0.
Velocities in units of the beam speed v_b, lengths in v_b / omega_pe.
"""
