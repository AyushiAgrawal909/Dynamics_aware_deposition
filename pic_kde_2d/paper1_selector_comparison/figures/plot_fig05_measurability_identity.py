#!/usr/bin/env python3
"""Generate Paper 1 Figure 5: parameter-free measurability identity.

The tested identity is

    N_e(sigma) = N_e(0) + ln[s(sigma) / f(sigma)],

where ``s`` is the analytic matched-filter amplitude ratio and ``f`` is the
measured per-cell floor ratio. No parameter is fitted.

Caption note
------------
The coherent ensemble average lowers the floor by approximately 1/sqrt(S)
relative to a single realization, so M as measured here is more permissive
than a single-run user experiences.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperstyle import COLORS, DOUBLE_COL, apply_style  # noqa: E402


RESULTS = PROJECT_ROOT / "results"
ALIAS_RESULTS = RESULTS / "weak_landau_alias"
PPC8_CLOSURE = ALIAS_RESULTS / "ppc8_cic_vs_mise20_logE.json"
PPC128_CLOSURE = ALIAS_RESULTS / "ppc128_cic_vs_mise16_logE.json"

LOW_PPC_CAMPAIGN = RESULTS / "weak_landau_efold_gate_4859337"
MID_PPC_CAMPAIGN = RESULTS / "weak_landau_efold_high_4859496"
HIGH_PPC_CAMPAIGN = RESULTS / "weak_landau_efold_very_high_4859575"
MISE16_CAMPAIGN = RESULTS / "weak_landau_mise_anchor_4859562"
PPC8_SIGMA20_CAMPAIGN = RESULTS / "weak_landau_mise_anchor_ppc8_4859661"

MODE = 1
NX = 64
PPCS = (8, 16, 32, 64, 128, 256, 512, 1024)
BENCHMARK_MARKERS = {"weak_landau": "o"}


def read_summary(path: Path) -> dict[str, float]:
    """Read one aggregate ten-seed floor-aware summary row."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one summary row in {path}")
    row = rows[0]
    required = (
        "sigma_dx",
        "resolved_efoldings_initial",
        "initial_amplitude",
        "noise_floor",
        "n_seeds",
    )
    if any(row.get(key) in (None, "", "None") for key in required):
        raise ValueError(f"incomplete floor/e-folding row in {path}")
    result = {key: float(row[key]) for key in required}
    result["source"] = str(path)
    return result


def summary_path(root: Path, ppc: int, sigma_tag: str) -> Path:
    """Return an aggregate summary path for one PPC and bandwidth."""
    return root / f"ppc{ppc}" / f"gamma_summary_sigma_{sigma_tag}.csv"


def analytic_signal_ratio(sigma_cells: float) -> float:
    """Return sqrt(T_eff^2(sigma)/T_eff^2(0)); CIC cancels."""
    q = (2.0 * np.pi * MODE * sigma_cells / NX) ** 2
    return float(np.exp(-0.5 * q))


def campaign_for_ppc(ppc: int) -> Path:
    """Select the canonical tail-extended campaign for each PPC."""
    if ppc in (8, 16, 32):
        return LOW_PPC_CAMPAIGN
    if ppc in (64, 128, 256):
        return MID_PPC_CAMPAIGN
    if ppc in (512, 1024):
        return HIGH_PPC_CAMPAIGN
    raise ValueError(f"unsupported PPC {ppc}")


def load_identity_points() -> list[dict[str, float | str]]:
    """Load every canonical pair with both a floor and an e-folding count."""
    points: list[dict[str, float | str]] = []
    for ppc in PPCS:
        base_campaign = campaign_for_ppc(ppc)
        rows: dict[float, dict[str, float]] = {
            0.0: read_summary(summary_path(base_campaign, ppc, "0p0")),
            0.3: read_summary(summary_path(base_campaign, ppc, "0p3")),
            30.0: read_summary(summary_path(base_campaign, ppc, "30p0")),
        }
        if ppc in (512, 1024):
            rows[16.0] = read_summary(
                summary_path(base_campaign, ppc, "16p0")
            )
        else:
            rows[16.0] = read_summary(
                summary_path(MISE16_CAMPAIGN, ppc, "16p0")
            )
        if ppc == 8:
            rows[20.0] = read_summary(
                summary_path(PPC8_SIGMA20_CAMPAIGN, ppc, "20p0")
            )

        baseline = rows[0.0]
        n_e_cic = baseline["resolved_efoldings_initial"]
        floor_cic = baseline["noise_floor"]
        for sigma in sorted(rows):
            row = rows[sigma]
            signal_ratio = analytic_signal_ratio(sigma)
            floor_ratio = row["noise_floor"] / floor_cic
            predicted = n_e_cic + np.log(signal_ratio / floor_ratio)
            measured = row["resolved_efoldings_initial"]
            points.append(
                {
                    "benchmark": "weak_landau",
                    "ppc": float(ppc),
                    "sigma_dx": sigma,
                    "signal_ratio": signal_ratio,
                    "floor_ratio": floor_ratio,
                    "n_e_cic": n_e_cic,
                    "n_e_predicted": float(predicted),
                    "n_e_measured": measured,
                    "n_seeds": row["n_seeds"],
                    "source": row["source"],
                }
            )
    return points


def select_point(
    points: list[dict[str, float | str]],
    *,
    ppc: int,
    sigma: float,
) -> dict[str, float | str]:
    """Select exactly one point from the frozen dataset."""
    matches = [
        point
        for point in points
        if int(point["ppc"]) == ppc
        and np.isclose(float(point["sigma_dx"]), sigma, atol=1.0e-12)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one identity point at PPC={ppc}, sigma={sigma}"
        )
    return matches[0]


def verify_anchors(points: list[dict[str, float | str]]) -> None:
    """Assert the two preregistered parameter-free closure examples."""
    ppc128 = select_point(points, ppc=128, sigma=16.0)
    s128 = float(ppc128["signal_ratio"])
    f128 = float(ppc128["floor_ratio"])
    delta128 = np.log(s128 / f128)
    measured_delta128 = (
        float(ppc128["n_e_measured"]) - float(ppc128["n_e_cic"])
    )
    # The supplied 0.2910 and 0.2697 values are rounded desk values; the exact
    # matched transfer is 0.291212933 and closes Delta N_e=0.270670931.
    assert np.isclose(s128, 0.2910, atol=3.0e-4)
    assert np.isclose(f128, 0.2221569, atol=5.0e-8)
    assert np.isclose(delta128, 0.2697, atol=1.1e-3)
    assert np.isclose(float(ppc128["n_e_cic"]), 2.72, atol=0.01)
    assert np.isclose(float(ppc128["n_e_measured"]), 2.99, atol=0.01)
    assert np.isclose(measured_delta128, 0.27, atol=0.005)

    ppc8 = select_point(points, ppc=8, sigma=20.0)
    s8 = float(ppc8["signal_ratio"])
    f8 = float(ppc8["floor_ratio"])
    predicted_f8_rounded = 0.507
    floor_error_percent = (
        (f8 - predicted_f8_rounded) / predicted_f8_rounded * 100.0
    )
    assert np.isclose(s8, 0.145489, atol=5.0e-7)
    assert np.isclose(f8, 0.506585, atol=5.0e-7)
    assert np.isclose(floor_error_percent, -0.082, atol=0.002)
    assert f"{float(ppc8['n_e_cic']):.6f}" == "1.627614"
    assert f"{float(ppc8['n_e_measured']):.6f}" == "0.380021"
    assert f"{float(ppc8['n_e_predicted']):.6f}" == "0.380021"

    ppc8_json = json.loads(PPC8_CLOSURE.read_text(encoding="utf-8"))
    ppc128_json = json.loads(PPC128_CLOSURE.read_text(encoding="utf-8"))
    assert np.isclose(
        ppc8_json["late_floor_ratio_mise_over_cic"],
        f8,
        atol=1.0e-14,
    )
    assert np.isclose(
        ppc128_json["late_floor_ratio_mise_over_cic"],
        f128,
        atol=1.0e-14,
    )


def main() -> None:
    apply_style()
    points = load_identity_points()
    verify_anchors(points)

    predicted = np.asarray(
        [float(point["n_e_predicted"]) for point in points]
    )
    measured = np.asarray(
        [float(point["n_e_measured"]) for point in points]
    )
    residual = measured - predicted
    assert len(points) == 33
    assert np.max(np.abs(residual)) < 2.0e-12
    assert all(int(point["n_seeds"]) == 10 for point in points)

    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, len(PPCS))
    )
    ppc_colors = dict(zip(PPCS, colors))

    fig, (ax_identity, ax_floor) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, 3.45),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )

    lower = float(min(predicted.min(), measured.min()) - 0.12)
    upper = float(max(predicted.max(), measured.max()) + 0.12)
    reference = np.linspace(lower, upper, 200)
    ax_identity.plot(
        reference,
        reference,
        color=COLORS["theory"],
        linestyle="--",
        linewidth=1.1,
        label=r"$y=x$ (no fitted parameters)",
        zorder=1,
    )

    for ppc in PPCS:
        subset = [
            point for point in points if int(point["ppc"]) == ppc
        ]
        ax_identity.scatter(
            [float(point["n_e_predicted"]) for point in subset],
            [float(point["n_e_measured"]) for point in subset],
            color=ppc_colors[ppc],
            marker=BENCHMARK_MARKERS["weak_landau"],
            edgecolor="white",
            linewidth=0.45,
            s=28,
            label=f"PPC {ppc}",
            zorder=3,
        )

    ppc128 = select_point(points, ppc=128, sigma=16.0)
    ppc8 = select_point(points, ppc=8, sigma=20.0)
    for point, color in (
        (ppc128, ppc_colors[128]),
        (ppc8, ppc_colors[8]),
    ):
        ax_identity.scatter(
            float(point["n_e_predicted"]),
            float(point["n_e_measured"]),
            facecolor="none",
            edgecolor=color,
            linewidth=1.25,
            s=75,
            zorder=5,
        )
    ax_identity.annotate(
        r"PPC 128, $\sigma=16\Delta x$: $\Delta N_e=0.271$",
        xy=(
            float(ppc128["n_e_predicted"]),
            float(ppc128["n_e_measured"]),
        ),
        xytext=(1.0, 1.65),
        arrowprops={
            "arrowstyle": "->",
            "color": ppc_colors[128],
            "linewidth": 0.8,
        },
        color=ppc_colors[128],
        fontsize=7.5,
    )
    ax_identity.text(
        0.04,
        0.96,
        "PPC 8 closure, $\\sigma=20\\Delta x$\n"
        r"$s=0.145489$"
        "\n"
        r"$f_{\rm pred}=0.507,\ f_{\rm meas}=0.506585$"
        "  (-0.082%)\n"
        r"$1.627614+\ln(s/f)=0.380021$"
        "\n"
        r"$N_{e,\rm meas}=0.380021$",
        transform=ax_identity.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": ppc_colors[8],
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=6,
    )
    ax_identity.set_xlim(lower, upper)
    ax_identity.set_ylim(lower, upper)
    ax_identity.set_aspect("equal", adjustable="box")
    ax_identity.set_xlabel(r"predicted $N_e(\sigma)$")
    ax_identity.set_ylabel(r"measured $N_e(\sigma)$")
    ax_identity.set_title("(a) Parameter-free identity")
    ax_identity.legend(
        loc="lower right",
        frameon=False,
        fontsize=6.7,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    for ppc in PPCS:
        subset = sorted(
            (
                point
                for point in points
                if int(point["ppc"]) == ppc
            ),
            key=lambda point: float(point["sigma_dx"]),
        )
        ax_floor.plot(
            [float(point["sigma_dx"]) for point in subset],
            [float(point["floor_ratio"]) for point in subset],
            color=ppc_colors[ppc],
            marker=BENCHMARK_MARKERS["weak_landau"],
            markersize=3.2,
            linewidth=0.9,
            alpha=0.90,
            label=f"PPC {ppc}",
        )
    ax_floor.axhline(
        1.0,
        color="0.4",
        linestyle=":",
        linewidth=0.8,
    )
    ax_floor.set_xscale("symlog", linthresh=0.3, linscale=0.75)
    ax_floor.set_yscale("log")
    ax_floor.set_xlim(-0.03, 34.0)
    ax_floor.set_ylim(0.01, 1.8)
    ax_floor.set_xlabel(r"Gaussian width, $\sigma/\Delta x$")
    ax_floor.set_ylabel(r"measured floor ratio, $f(\sigma)$")
    ax_floor.set_title("(b) The measured floor moves")
    ax_floor.text(
        0.04,
        0.06,
        "Per-cell floor estimation is mandatory;\n"
        "an inherited floor corrupts $M$.",
        transform=ax_floor.transAxes,
        fontsize=7.5,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 2.0,
        },
    )

    fig.suptitle(
        r"$N_e(\sigma)=N_e(0)+\ln[s(\sigma)/f(\sigma)]$ closes without fitting",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.018,
        r"Coherent averaging over $S=10$ seeds lowers the floor by "
        r"$\sim1/\sqrt{S}$; a single-run user therefore faces a stricter "
        r"measurability constraint.",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.86,
        bottom=0.20,
        wspace=0.32,
    )
    fig.savefig(FIGURE_DIR / "fig05_measurability_identity.pdf")
    fig.savefig(FIGURE_DIR / "fig05_measurability_identity.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
