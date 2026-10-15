"""Statistical objectives: mean-square error of the filtered mesh density
(or of the mesh field computed from it) for a candidate kernel.

Two objectives are provided.

``SnapshotObjective`` (reference-assisted, as in the paper)
    Paired particle runs at the production marker count (``rho``) and at a
    higher marker count (``rho_ref``), deposited with the baseline shape and
    unfiltered, saved at common times.  For a mesh filter the error is a
    quadratic form in W(k), so the data enter only through per-wavenumber
    sums A = sum |rho_k|^2, B = sum Re(rho_k conj(rho_ref,k)),
    C = sum |rho_ref,k|^2, and any candidate is evaluated in O(N_cells):

        MSE(W) = sum_k w(k) [W^2 A - 2 W B + C] / (N_samples N_cells^2),

    with w = 1 for the density error and w = |kappa|^2 / K^4 for the field
    error (the field after the solve, before return filtering).

``ModelObjective`` (reference-free, model based)
    Uncorrelated markers with the given markers per cell plus a specified
    set of signal modes.  The expected error against the noise-free mesh
    density of the baseline kernel is

        MSE = mean_k w(k) [ |G_s(k) - 1|^2 P_signal(k) + W(k)^2 P_noise(k) ],

    with P_noise = sum_s (q_s n_s)^2 / PPC_s * sum_aliases S(q)^2.  Marker
    correlations from quiet starts or from the plasma's own shielding are not
    included; this objective has not been validated against PIC data in the
    paper and should be treated as a model estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .filters import Kernel
from .grid import Grid, shape_order
from .plasma import Plasma

METRICS = ("density", "field")


def _metric_weight(grid: Grid, metric: str) -> np.ndarray:
    if metric == "density":
        return np.ones(grid.N)
    if metric == "field":
        K = np.stack(grid.fft_wavenumbers(), axis=-1)
        return grid.field_weight(K)
    raise ValueError(f"metric must be one of {METRICS}")


class Objective:
    """Common interface: ``mse(kernel, metric, subset)``."""

    kind = "base"
    reference_free = False

    def __init__(self, grid: Grid):
        self.grid = grid
        self._W_cache: dict = {}
        self._weights = {m: _metric_weight(grid, m) for m in METRICS}

    def transfer(self, kernel: Kernel, cache: bool = True) -> np.ndarray:
        key = (kernel.filter.cache_key(), kernel.order)
        if key in self._W_cache:
            return self._W_cache[key]
        W = kernel.filter.transfer_grid(self.grid)
        if cache:
            if len(self._W_cache) >= 512:
                self._W_cache.pop(next(iter(self._W_cache)))
            self._W_cache[key] = W
        return W

    def supports_order(self, order: int) -> bool:
        return True

    def splits(self) -> dict:
        return {}


# ======================================================================
class ModelObjective(Objective):
    kind = "model"
    reference_free = True

    def __init__(self, grid: Grid, plasma: Plasma, *, ppc: float | None,
                 signal: Sequence[dict] = (), base_order: int | None = None):
        super().__init__(grid)
        self.plasma = plasma
        self.base_order = grid.order if base_order is None else base_order
        self.ppc = ppc
        mobile = plasma.mobile
        n_tot = sum(s.n for s in mobile)
        var = 0.0
        for s in mobile:
            ppc_s = s.ppc if s.ppc is not None else (None if ppc is None else ppc * s.n / n_tot)
            if ppc_s is None or ppc_s <= 0:
                raise ValueError("the model objective needs objective.ppc (total markers per cell) "
                                 f"or a ppc for species '{s.name}'")
            var += (s.q * s.n) ** 2 / ppc_s
        self.node_variance = var / plasma.charge_scale() ** 2   # in units of rho0^2
        self.K = np.stack(grid.fft_wavenumbers(), axis=-1)
        self._noise: dict[int, np.ndarray] = {}
        self.signal = []
        self._signal_power = np.zeros(grid.N)
        S_base = grid.shape_transform(self.K, self.base_order)
        for item in signal:
            m = tuple(int(x) for x in item["mode"])
            amp = float(item.get("amplitude", 0.0))
            grid.mode_vector(m)
            self.signal.append(dict(mode=list(m), amplitude=amp))
            # Powers are per-node spectral densities (MSE = mean_k P): a cosine
            # of relative amplitude A has |rho_k|^2 = (A N / 2)^2 at +-k, i.e.
            # P = N A^2 / 4 at each; a self-conjugate (Nyquist) mode has P = N A^2.
            ip = tuple(mi % ni for mi, ni in zip(m, grid.N))
            im = tuple((-mi) % ni for mi, ni in zip(m, grid.N))
            n = grid.n_cells
            if ip == im:
                self._signal_power[ip] += n * amp**2 * S_base[ip] ** 2
            else:
                for idx in (ip, im):
                    self._signal_power[idx] += 0.25 * n * amp**2 * S_base[idx] ** 2
        self._S_base = S_base

    def noise_power(self, order: int) -> np.ndarray:
        """Per-mode noise power; zero at k = 0 (the total marker charge is fixed)."""
        if order not in self._noise:
            P = self.node_variance * self.grid.alias_power(self.K, order)
            P[(0,) * self.grid.dim] = 0.0
            self._noise[order] = P
        return self._noise[order]

    def mse(self, kernel: Kernel, metric: str = "density", subset: str = "select",
            cache: bool = True) -> float:
        W = self.transfer(kernel, cache)
        w = self._weights[metric]
        noise = W**2 * self.noise_power(kernel.order)
        if kernel.order != self.base_order:
            G = W * self.grid.shape_transform(self.K, kernel.order) / np.where(
                self._S_base == 0, 1.0, self._S_base)
        else:
            G = W
        bias = (G - 1.0) ** 2 * self._signal_power
        return float(np.mean(w * (noise + bias)))

    def info(self) -> dict:
        return dict(kind=self.kind, reference_free=True, ppc=self.ppc,
                    node_variance_rel=self.node_variance, signal=self.signal,
                    note="model estimate: uncorrelated markers + specified signal modes")


# ======================================================================
@dataclass
class _Stats:
    A: np.ndarray   # (R, *N)  sum_t |rho_k|^2
    B: np.ndarray   # (R, *N)  sum_t Re(rho_k conj(ref_k))
    C: np.ndarray   # (R, *N)  sum_t |ref_k|^2
    n_times: int


class SnapshotObjective(Objective):
    """Reference-assisted objective from paired production/reference meshes."""

    kind = "snapshots"
    reference_free = False

    def __init__(self, grid: Grid, rho: np.ndarray, rho_ref: np.ndarray, *,
                 select: Sequence[int] | None = None, evaluate: Sequence[int] | None = None,
                 extra_shapes: dict[str, np.ndarray] | None = None, base_order: int | None = None):
        super().__init__(grid)
        self.base_order = grid.order if base_order is None else base_order
        rho = np.asarray(rho, dtype=float)
        rho_ref = np.asarray(rho_ref, dtype=float)
        rho, rho_ref = self._shape_check(rho), self._shape_check(rho_ref)
        if rho.shape != rho_ref.shape:
            raise ValueError(f"rho {rho.shape} and rho_ref {rho_ref.shape} must match")
        self.n_real, self.n_times = rho.shape[0], rho.shape[1]
        ref_hat = self._fft(rho_ref)
        self._stats = {self.base_order: self._make_stats(self._fft(rho), ref_hat)}
        for name, arr in (extra_shapes or {}).items():
            arr = self._shape_check(np.asarray(arr, dtype=float))
            if arr.shape != rho.shape:
                raise ValueError(f"extra shape '{name}' has shape {arr.shape}, expected {rho.shape}")
            self._stats[shape_order(name)] = self._make_stats(self._fft(arr), ref_hat)
        self._init_subsets(select, evaluate)
        self._sum_cache: dict = {}

    # -------------------------------------------------- compact statistics
    def save_statistics(self, path) -> None:
        """Write the per-realisation spectral sums (all the objective needs).

        The file is far smaller than the snapshots: 3 real arrays of shape
        (realisations, *N) per particle shape.
        """
        arrays = dict(grid_N=np.array(self.grid.N), grid_L=np.array(self.grid.L),
                      n_times=np.array(self.n_times), base_order=np.array(self.base_order))
        for order, st in self._stats.items():
            arrays[f"A_{order}"], arrays[f"B_{order}"], arrays[f"C_{order}"] = st.A, st.B, st.C
        np.savez_compressed(path, **arrays)

    @classmethod
    def from_statistics(cls, grid: Grid, path, *, select=None, evaluate=None):
        """Build the objective from a file written by :meth:`save_statistics`."""
        data = np.load(path)
        if tuple(data["grid_N"]) != tuple(grid.N) or not np.allclose(data["grid_L"], grid.L):
            raise ValueError(f"statistics were computed for N={tuple(data['grid_N'])}, "
                             f"L={tuple(data['grid_L'])}, not for this grid")
        if int(data["base_order"]) != grid.order:
            from .grid import shape_name
            raise ValueError(f"statistics were deposited with {shape_name(int(data['base_order']))}, "
                             f"but the grid shape is {grid.shape}")
        obj = cls.__new__(cls)
        Objective.__init__(obj, grid)
        obj.base_order = int(data["base_order"])
        obj.n_times = int(data["n_times"])
        obj._stats = {}
        for key in data.files:
            if key.startswith("A_"):
                o = int(key[2:])
                obj._stats[o] = _Stats(A=data[f"A_{o}"], B=data[f"B_{o}"], C=data[f"C_{o}"],
                                       n_times=obj.n_times)
        obj.n_real = obj._stats[obj.base_order].A.shape[0]
        obj._init_subsets(select, evaluate)
        obj._sum_cache = {}
        return obj

    def _init_subsets(self, select, evaluate):
        R = self.n_real
        if select is not None:
            select = [int(r) for r in select]
        if evaluate is not None:
            evaluate = [int(r) for r in evaluate]
        if select is None and evaluate is None:
            if R >= 2:
                half = R // 2
                select, evaluate = list(range(half)), list(range(half, R))
            else:
                select, evaluate = [0], [0]
        elif select is None:
            select = [r for r in range(R) if r not in set(evaluate)] or list(evaluate)
        elif evaluate is None:
            evaluate = [r for r in range(R) if r not in set(select)] or list(select)
        self._subsets = dict(select=select, evaluate=evaluate, all=list(range(R)))
        for key, idx in self._subsets.items():
            if not idx or min(idx) < 0 or max(idx) >= R:
                raise ValueError(f"realisation indices for '{key}' must be non-empty and lie in "
                                 f"0..{R - 1}; got {idx}")
        self.held_out = set(select).isdisjoint(evaluate)

    def _shape_check(self, arr):
        N = tuple(self.grid.N)
        if arr.shape[-len(N):] != N:
            raise ValueError(f"snapshot mesh shape {arr.shape[-len(N):]} does not match the grid {N}")
        if arr.ndim == len(N):
            arr = arr[None, None]
        elif arr.ndim == len(N) + 1:
            arr = arr[None]
        elif arr.ndim != len(N) + 2:
            raise ValueError("snapshots must have shape (realisations, times, *N)")
        return arr

    def _fft(self, arr):
        axes = tuple(range(2, arr.ndim))
        return np.fft.fftn(arr, axes=axes)

    def _make_stats(self, low_hat, ref_hat):
        return _Stats(A=np.sum(np.abs(low_hat) ** 2, axis=1),
                      B=np.sum((low_hat * np.conj(ref_hat)).real, axis=1),
                      C=np.sum(np.abs(ref_hat) ** 2, axis=1), n_times=low_hat.shape[1])

    def supports_order(self, order: int) -> bool:
        return order in self._stats

    def splits(self):
        return dict(self._subsets)

    def _sums(self, order: int, subset):
        key = (order, tuple(subset) if not isinstance(subset, str) else subset)
        if key not in self._sum_cache:
            idx = self._subsets[subset] if isinstance(subset, str) else list(subset)
            st = self._stats[order]
            norm = len(idx) * st.n_times * self.grid.n_cells**2
            self._sum_cache[key] = (st.A[idx].sum(0) / norm, st.B[idx].sum(0) / norm,
                                    st.C[idx].sum(0) / norm)
        return self._sum_cache[key]

    def mse(self, kernel: Kernel, metric: str = "density", subset="select",
            cache: bool = True) -> float:
        if kernel.order not in self._stats:
            raise ValueError(f"no snapshots deposited with shape order {kernel.order}")
        A, B, C = self._sums(kernel.order, subset)
        W = self.transfer(kernel, cache)
        return float(np.sum(self._weights[metric] * (W * W * A - 2.0 * W * B + C)))

    def per_realisation(self, kernel: Kernel, metric: str = "density") -> np.ndarray:
        st = self._stats[kernel.order]
        W = self.transfer(kernel)
        w = self._weights[metric]
        axes = tuple(range(1, st.A.ndim))
        norm = st.n_times * self.grid.n_cells**2
        return np.sum(w * (W * W * st.A - 2.0 * W * st.B + st.C), axis=axes) / norm

    def reduction_interval(self, kernel: Kernel, baseline: Kernel, metric: str = "density",
                           subset: str = "evaluate", draws: int = 5000, seed: int = 120912,
                           level: float = 0.95) -> tuple[float, float, float]:
        """Reduction factor N = MSE(baseline)/MSE(kernel) with a paired bootstrap CI."""
        idx = np.array(self._subsets[subset])
        e_c = self.per_realisation(kernel, metric)[idx]
        e_b = self.per_realisation(baseline, metric)[idx]
        point = float(e_b.sum() / e_c.sum())
        if idx.size < 2:
            return point, float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, idx.size, size=(draws, idx.size))
        ratios = e_b[pick].sum(1) / e_c[pick].sum(1)
        lo, hi = np.quantile(ratios, [(1 - level) / 2, 1 - (1 - level) / 2])
        return point, float(lo), float(hi)

    def info(self) -> dict:
        return dict(kind=self.kind, reference_free=False, realisations=self.n_real,
                    times=self.n_times, select=self._subsets["select"],
                    evaluate=self._subsets["evaluate"], held_out=self.held_out,
                    shapes=sorted(self._stats))
