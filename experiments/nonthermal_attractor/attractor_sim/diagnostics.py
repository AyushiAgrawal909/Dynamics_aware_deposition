"""
attractor_sim/diagnostics.py
============================

Snapshot diagnostics for both backends, accumulated into a bundle that
satisfies the T1 data contract (``attractor_analysis.schema``).

Diagnostics never modify the particle or grid state.  For the KDK pusher,
x, v and E are synchronised at integer steps, so every snapshot is at
t = n dt exactly.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attractor_analysis import schema  # noqa: E402
from attractor_analysis.energy import energy_histogram  # noqa: E402
from attractor_analysis.waterbag import (particle_eta_floor, reconstruct_f_particles,  # noqa: E402
                                         waterbag_content, coarse_grain_grid)


def exact_rebin(f, nx, dx, v_max, dv, nx_b, nv_b, v_lim, tol=1e-9):
    """Conservatively block-average a Vlasov grid onto the PIC reconstruction
    cells: x in [0, L) with nx_b cells, v in [-v_lim, v_lim) with nv_b cells.

    The blocks must tile those cells exactly (integer block sizes, v_lim on a
    grid face); otherwise ValueError, so no backend silently observes rho(eta)
    through a different coarse-graining.  Returns (f_coarse, mass outside
    [-v_lim, v_lim))."""
    bx = nx / nx_b
    bv = (2.0 * v_lim / nv_b) / dv
    j0 = (v_max - v_lim) / dv
    for name, val in (("nx / nx_b", bx), ("dv_b / dv", bv), ("(v_max - v_lim) / dv", j0)):
        if abs(val - round(val)) > tol * max(1.0, abs(val)):
            raise ValueError(f"exact rebin needs integer {name}, got {val:.6g}")
    bx, bv, j0 = int(round(bx)), int(round(bv)), int(round(j0))
    sub = f[:, j0:j0 + nv_b * bv]
    outside = float((f.sum() - sub.sum()) * dx * dv)
    return coarse_grain_grid(sub, bx, bv), outside


class SnapshotRecorder:
    def __init__(self, eps_edges, eta_edges, recon: dict, n_phase_snapshots: int = 0,
                 phase_decimate: int = 64):
        self.eps_edges = np.asarray(eps_edges)
        self.eta_edges = np.asarray(eta_edges)
        self.recon = recon  # {"nx_b", "nv_b", "v_lim"}
        self.rows: Dict[str, List] = {k: [] for k in (
            "times", "number", "total_charge", "momentum", "kinetic_energy", "field_energy", "total_energy",
            "N_eps", "eps_bin_weight", "eps_bin_energy", "eps_bin_neff", "eps_overflow",
            "rho_eta", "rho_bin_volume", "rho_bin_mass", "Ek2")}
        self.extra: Dict[str, List] = {"f_min": [], "outflow_mass": [], "clip_mass": [], "L2": [],
                                       "mass_outside_recon": []}
        self.k = None
        self.rho_rec = None
        self.phase = {"phase_space_times": [], "phase_space_x": [], "phase_space_v": [], "phase_space_w": []}
        self.f_coarse = {"f_coarse_times": [], "f_coarse": []}
        self.n_phase = n_phase_snapshots
        self.phase_decimate = phase_decimate

    def _common(self, t, b, Ek2, k):
        self.rows["times"].append(t)
        for key in ("number", "total_charge", "momentum", "kinetic_energy", "field_energy", "total_energy"):
            self.rows[key].append(b[key])
        self.rows["Ek2"].append(Ek2)
        self.k = k
        for key in ("f_min", "outflow_mass", "clip_mass", "L2"):
            self.extra[key].append(b.get(key, np.nan))

    def _eps(self, h):
        self.rows["N_eps"].append(h.N)
        self.rows["eps_bin_weight"].append(h.weight)
        self.rows["eps_bin_energy"].append(h.energy)
        self.rows["eps_bin_neff"].append(h.neff)
        self.rows["eps_overflow"].append([h.overflow_weight, h.overflow_energy])

    def _rho(self, wc):
        self.rows["rho_eta"].append(wc.rho)
        self.rows["rho_bin_volume"].append(wc.volume)
        self.rows["rho_bin_mass"].append(wc.mass)
        self.rho_rec = np.array([wc.dx_b, wc.dv_b, wc.eta_floor])

    def record_pic(self, t, sim, phase: bool = False):
        b = sim.budgets()
        self._common(t, b, sim.field_spectrum(), sim.k)
        eps = 0.5 * sim.v * sim.v
        self._eps(energy_histogram(eps, sim.w, self.eps_edges))
        nxb, nvb, vl = self.recon["nx_b"], self.recon["nv_b"], self.recon["v_lim"]
        fh = reconstruct_f_particles(sim.x, sim.v, sim.w, sim.L, vl, nxb, nvb)
        dxb, dvb = sim.L / nxb, 2 * vl / nvb
        self.extra["mass_outside_recon"].append(float(np.sum(sim.w[np.abs(sim.v) >= vl])))
        self._rho(waterbag_content(fh, dxb, dvb, self.eta_edges, particle_eta_floor(sim.w.min(), dxb, dvb)))
        if phase:
            sl = slice(None, None, self.phase_decimate)
            self.phase["phase_space_times"].append(t)
            self.phase["phase_space_x"].append(sim.x[sl].copy())
            self.phase["phase_space_v"].append(sim.v[sl].copy())
            self.phase["phase_space_w"].append(sim.w[sl] * self.phase_decimate)

    def record_vlasov(self, t, sim, phase: bool = False):
        b = sim.budgets()
        self._common(t, b, sim.field_spectrum(), sim.k)
        # energy histogram from cell masses at cell-centre energies
        mass = sim.f * sim.dx * sim.dv
        eps = np.broadcast_to(0.5 * sim.v**2, sim.f.shape).ravel()
        w = mass.ravel()
        h = energy_histogram(eps, w, self.eps_edges)
        h.neff = np.where(h.weight > 0, np.inf, 0.0)  # grid f: no counting noise
        self._eps(h)
        if self.recon.get("vlasov_rebin") == "exact":
            fc, outside = exact_rebin(sim.f, sim.nx, sim.dx, sim.v_max, sim.dv,
                                      self.recon["nx_b"], self.recon["nv_b"], self.recon["v_lim"])
            dxb, dvb = sim.L / self.recon["nx_b"], 2 * self.recon["v_lim"] / self.recon["nv_b"]
        else:
            # protocol v1 (kept to reproduce its bundles): rounded block size,
            # truncated at a block boundary, nearly the whole [-v_max, v_max]
            bx = max(1, sim.nx // self.recon["nx_b"])
            bv = max(1, int(round((self.recon["v_lim"] * 2 / self.recon["nv_b"]) / sim.dv)))
            nx_c, nv_c = (sim.nx // bx) * bx, (sim.nv // bv) * bv
            fc = coarse_grain_grid(sim.f[:nx_c, :nv_c], bx, bv)
            outside = float((sim.f.sum() - sim.f[:nx_c, :nv_c].sum()) * sim.dx * sim.dv)
            dxb, dvb = sim.dx * bx, sim.dv * bv
        self.extra["mass_outside_recon"].append(outside)
        self._rho(waterbag_content(fc, dxb, dvb, self.eta_edges, 0.0))
        if phase:
            self.f_coarse["f_coarse_times"].append(t)
            self.f_coarse["f_coarse"].append(fc.astype(np.float32))

    def to_bundle(self, path, metadata: dict):
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        arrays["eps_edges"] = self.eps_edges
        arrays["eta_edges"] = self.eta_edges
        arrays["rho_reconstruction"] = self.rho_rec
        arrays["k"] = self.k
        for k, v in self.extra.items():
            arrays[k] = np.asarray(v, dtype=float)
        if self.f_coarse["f_coarse_times"]:
            arrays["f_coarse_times"] = np.asarray(self.f_coarse["f_coarse_times"])
            arrays["f_coarse"] = np.asarray(self.f_coarse["f_coarse"])
        if self.phase["phase_space_times"]:
            arrays["phase_space_times"] = np.asarray(self.phase["phase_space_times"])
            for key in ("phase_space_x", "phase_space_v", "phase_space_w"):
                arrays[key] = np.asarray(self.phase[key])
        return schema.write_bundle(path, metadata, arrays)
