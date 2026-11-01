"""Protocol-v2 decision: the adversarial cases from the 2026-10-24 review.

None of these invalid campaigns may produce a scientific GO, and a stationary
convergent feature must not be reported as regulator dependence.
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

from attractor_analysis import convergence as C, decision as D

HERE = Path(__file__).resolve().parents[1]
PROTO = yaml.safe_load((HERE / "protocol" / "T2_PROTOCOL_v2.yaml").read_text())
PROTO["accepted_revisions"]["vlasov"] = ["v2rev"]
TIMES = np.arange(0.0, 4800.0 + 1e-9, 10.0)
CFG = {"pic": {"name": "pic-config"}, "vlasov": {"name": "vlasov-config"}}


def _arr(kind, energy_drift=0.0, mass_gain=0.0, clip=0.0, outflow=0.0, legacy=False, momentum_drift=0.0):
    n = len(TIMES)
    s = TIMES / TIMES[-1]
    arr = {"times": TIMES.copy(), "number": np.full(n, 1000.0) * (1 + mass_gain * s),
           "kinetic_energy": np.full(n, 500.0), "total_energy": 600.0 * (1 + energy_drift * s),
           "momentum": 1e-3 + momentum_drift * s, "N_eps": np.ones((n, 3))}
    if kind == "vlasov":
        if legacy:
            arr["outflow_mass"] = -1000.0 * mass_gain * s   # v1 nets the clip gain into outflow
        else:
            arr["clip_mass"] = 1000.0 * clip * s
            arr["outflow_mass"] = 1000.0 * outflow * s
            arr["number"] = 1000.0 * (1 + (clip - outflow) * s)
    return arr


def _campaign(drop=(), extra=(), vlasov_kw=None, pic_kw=None, control_kw=None, rev="43ca0454373a"):
    out = []
    exp = D.expected_runs(PROTO)["required"]
    for kind, l, s in sorted(exp):
        if (kind, l, s) in drop:
            continue
        md = {"backend": "vlasov1d1v" if kind == "vlasov" else "pic1d1v", "nested_level": l, "seed": s,
              "config": CFG["vlasov" if kind == "vlasov" else "pic"],
              "git_revision": "v2rev" if kind == "vlasov" else rev}
        if kind == "control":
            md["preparation"] = "stable_control"
        kw = {"vlasov": vlasov_kw, "pic": pic_kw, "control": control_kw}[kind] or {}
        out.append((md, _arr(kind, **kw), f"{kind}_L{l}_s{s}"))
    out.extend(extra)
    return out


def _quality(bundles, f_tail=0.008):
    man = D.build_manifest(bundles, PROTO, CFG)
    q = D.run_quality(man["runs"], PROTO, {k: f_tail for k in man["runs"] if k[0] != "control"})
    return man, q


# ---------------------------------------------------------------- manifest (F1, F6)
def test_complete_campaign_has_no_problems():
    man, q = _quality(_campaign())
    assert man["problems"] == []
    assert all(r["pass"] for r in q.values())


def test_missing_vlasov_and_controls_are_invalid_inputs():
    drop = {("vlasov", l, 0) for l in range(4)} | {("control", l, s) for l in range(8) for s in (0, 1)}
    man, _ = _quality(_campaign(drop=drop))
    assert any("missing run ('vlasov', 0, 0)" in p for p in man["problems"])
    g = {"eps_q": {"verdict": C.PASS}, "eta_min": {"verdict": C.PASS}}
    d = D.decide_campaign(man["problems"], True, 8, None, False, g, {"eps_q": True, "eta_min": True})
    assert d["outcome"] == D.INVALID_INPUTS


def test_duplicate_unexpected_wrong_config_and_revision_are_flagged():
    b = _campaign()
    dup = b[0]
    other = ({"backend": "hexapic", "nested_level": 0, "seed": 0, "config": {}, "git_revision": "x"}, dup[1], "hex")
    wrong_cfg = ({**b[1][0], "config": {"changed": True}, "seed": 9}, b[1][1], "seed9")
    man = D.build_manifest(b + [dup, other, wrong_cfg], PROTO, CFG)
    text = " | ".join(man["problems"])
    assert "duplicate run" in text and "not part of T2" in text and "not in the declared matrix" in text
    man2 = D.build_manifest(_campaign(rev="deadbeef"), PROTO, CFG)
    assert any("not accepted" in p for p in man2["problems"])


def test_truncated_run_is_flagged():
    b = _campaign()
    md, arr, p = b[5]
    short = {k: (v[:-10] if np.ndim(v) and len(v) == len(TIMES) else v) for k, v in arr.items()}
    man = D.build_manifest(b[:5] + [(md, short, p)] + b[6:], PROTO, CFG)
    assert any("ends at" in x for x in man["problems"])


def test_end_index_is_inclusive():
    assert TIMES[D.end_index(TIMES, 4800.0)] == 4800.0


# ---------------------------------------------------------------- quality vetoes (F1, F4)
def test_strong_pic_energy_change_fails_quality():
    _, q = _quality(_campaign(pic_kw={"energy_drift": 0.5}))
    assert not q[("pic", 0, 0)]["pass"]


def test_control_heating_counts_against_the_tail():
    _, q = _quality(_campaign(control_kw={"energy_drift": 0.0}))
    assert q[("pic", 3, 1)]["pass"]
    b = _campaign()
    b = [(md, ({**a, "kinetic_energy": a["kinetic_energy"] * (1 + 0.01 * TIMES / 4800)}
               if md.get("preparation") == "stable_control" else a), p) for md, a, p in b]
    _, q2 = _quality(b)
    assert not q2[("pic", 3, 1)]["pass"]


def test_momentum_drift_fails_quality():
    _, q = _quality(_campaign(pic_kw={"momentum_drift": 1e-3}))
    assert not q[("pic", 2, 0)]["pass"]


def test_v1_vlasov_bundle_with_mass_gain_is_rejected():
    _, q = _quality(_campaign(vlasov_kw={"mass_gain": 0.28, "legacy": True}))
    r = q[("vlasov", 3, 0)]
    assert not r["pass"] and any("clipping budget" in f for f in r["fails"])


def test_clipping_and_outflow_are_separate_vetoes():
    _, q = _quality(_campaign(vlasov_kw={"clip": 1e-3}))
    assert any("clipping added" in f for f in q[("vlasov", 0, 0)]["fails"])
    _, q = _quality(_campaign(vlasov_kw={"outflow": 1e-3}))
    assert any("boundary outflow" in f for f in q[("vlasov", 0, 0)]["fails"])
    # clipping gain and boundary loss that cancel in the total are still caught
    _, q = _quality(_campaign(vlasov_kw={"clip": 1e-3, "outflow": 1e-3}))
    assert not q[("vlasov", 0, 0)]["pass"]


def test_usable_levels_are_contiguous_from_finest():
    q = {("pic", l, s): {"pass": not (l == 3 and s == 1)} for l in range(8) for s in range(4)}
    assert D.usable_pic_levels(q, range(8), range(4)) == [0, 1, 2]
    q[("pic", 0, 2)]["pass"] = False
    assert D.usable_pic_levels(q, range(8), range(4)) == []


# ---------------------------------------------------------------- decide_campaign
OK = {"eps_q": True, "eta_min": True}


def test_failed_finest_level_is_invalid_numerics():
    d = D.decide_campaign([], False, 0, None, False, {"eps_q": {"verdict": C.PASS}}, OK)
    assert d["outcome"] == D.INVALID_NUMERICS


def test_triggered_seed_rule_without_seeds_is_incomplete():
    rule = {"triggered": True, "ratio": 0.4}
    d = D.decide_campaign([], True, 8, rule, False, {"eps_q": {"verdict": C.PASS}}, OK)
    assert d["outcome"] == D.INCOMPLETE
    d = D.decide_campaign([], True, 8, rule, True, {"eps_q": {"verdict": C.PASS}}, OK)
    assert d["outcome"] == D.GO_T3


def test_positive_verdict_needs_admissible_reference():
    g = {"eps_q": {"verdict": C.PASS}, "eta_min": {"verdict": C.INCONCLUSIVE, "unresolved": True}}
    assert D.decide_campaign([], True, 8, None, False, g, {"eps_q": False})["outcome"] == D.INCONCLUSIVE
    assert D.decide_campaign([], True, 8, None, False, g, OK)["outcome"] == D.GO_T3


def test_stationary_converged_is_a_stationary_result_not_regulator():
    g = {"eps_q": {"verdict": C.STATIONARY}, "eta_min": {"verdict": C.INCONCLUSIVE, "unresolved": True}}
    assert D.decide_campaign([], True, 8, None, False, g, OK)["outcome"] == D.STATIONARY_RESULT


def test_backend_disagreement_is_inconclusive():
    g = {"eps_q": {"verdict": C.FAIL, "fail_kind": "reference"},
         "eta_min": {"verdict": C.INCONCLUSIVE, "unresolved": True}}
    d = D.decide_campaign([], True, 8, None, False, g, OK)
    assert d["outcome"] == D.INCONCLUSIVE and "backend disagreement" in " ".join(d["reasons"])


def test_regulator_fail_outcomes():
    g = {"eps_q": {"verdict": C.FAIL, "fail_kind": "regulator"},
         "eta_min": {"verdict": C.INCONCLUSIVE, "unresolved": True}}
    assert D.decide_campaign([], True, 8, None, False, g, OK)["outcome"] == D.REGULATOR_RESULT_EPS_Q_ONLY
    g["eta_min"] = {"verdict": C.FAIL, "fail_kind": "regulator"}
    assert D.decide_campaign([], True, 8, None, False, g, OK)["outcome"] == D.REGULATOR_RESULT


def test_too_few_usable_levels_is_inconclusive():
    d = D.decide_campaign([], True, 2, None, False, {"eps_q": {"verdict": C.PASS}}, OK)
    assert d["outcome"] == D.INCONCLUSIVE


# ---------------------------------------------------------------- convergence gate (F2, F3, F7)
t = np.linspace(1, 6, 11)


def _moving(seed=0):
    rng = np.random.default_rng(seed)
    return {l: 10 + 5 * np.sqrt(t) + .01 * 2**l + .005 * 2**(l / 2) * rng.normal(size=(5, len(t)))
            for l in range(5)}


def test_alignment_does_not_cure_reference_disagreement():
    F = _moving()
    g = C.convergence_gate(t, F, (1, 6), reference=np.full_like(t, 10000),
                           onset_by_level={l: np.zeros(5) for l in F},
                           regulator_by_level={l: 8192 / 2**l for l in F})
    assert g["raw"]["reference_ok"] is False
    assert g["verdict"] != C.PASS_ALIGNED and g["fail_kind"] == "reference"
    # without a reference onset, even a matching shape cannot pass aligned
    g2 = C.convergence_gate(t, F, (1, 6), reference=np.full_like(t, 10000),
                            onset_by_level={l: np.zeros(5) for l in F}, reference_onset=0.0)
    assert g2["verdict"] != C.PASS_ALIGNED


def test_genuinely_shifted_trajectories_with_shifted_reference_pass_aligned():
    tt = np.linspace(0, 8, 81)
    rng = np.random.default_rng(2)
    F, ons = {}, {}
    for l in range(5):
        t0 = 1.0 + 0.3 * l + 0.01 * rng.standard_normal(5)
        ons[l] = t0
        F[l] = np.array([np.where(tt > a, 10 * np.sqrt(np.clip(tt - a, 0, None)), 0.0) for a in t0])
        F[l] += 0.01 * 2 ** (l / 2) * rng.standard_normal(F[l].shape)
    ref_on = 0.8
    ref = np.where(tt > ref_on, 10 * np.sqrt(np.clip(tt - ref_on, 0, None)), 0.0)
    g = C.convergence_gate(tt, F, (2.5, 8.0), tau=0.1, reference=ref, onset_by_level=ons, reference_onset=ref_on)
    assert g["verdict"] == C.PASS_ALIGNED


def test_noise_free_stationary_convergence_is_stationary_converged():
    stationary = {l: np.full((4, len(t)), 10 + 1e-6 * 2**l) for l in range(5)}
    g = C.convergence_gate(t, stationary, (1, 6), abs_tol=0.1, motion_floor=0.1)
    assert g["verdict"] == C.STATIONARY


def test_noisy_stationary_tail_like_the_pilot_is_stationary_converged():
    """Pilot-measured eps_q noise: 0.022 per snapshot at 128 ppc, scaled 1/sqrt(ppc)."""
    tt = np.arange(0, 4800.1, 10.0)
    rng = np.random.default_rng(1)
    F = {l: 3.73 + 0.0217 * 2 ** ((l - 7) / 2) * rng.normal(size=(4, tt.size)) for l in range(8)}
    g = C.convergence_gate(tt, F, (500, 4800), tau=0.1, abs_tol=0.0373, motion_floor=0.0373,
                           reference=np.full(tt.size, 3.73), reference_err=np.full(tt.size, 0.002))
    assert g["verdict"] == C.STATIONARY
    # the v1 gate cannot say so
    assert C.convergence_gate(tt, F, (500, 4800), tau=0.1)["verdict"] != C.PASS


def test_stationary_regulator_offset_still_fails():
    rng = np.random.default_rng(1)
    F = {l: 30.0 - 3.5 * l + 0.05 * rng.standard_normal((5, len(t))) for l in range(5)}
    g = C.convergence_gate(t, F, (1, 6), tau=0.1, abs_tol=0.3, motion_floor=0.3)
    assert g["verdict"] == C.FAIL and g["fail_kind"] == "regulator"


def test_noncontiguous_levels_are_inconclusive_not_a_crash():
    F = _moving()
    g = C.convergence_gate(t, {l: F[l] for l in (0, 2, 3)}, (1, 6))
    assert g["verdict"] == C.INCONCLUSIVE


# ---------------------------------------------------------------- Vlasov ladder (F6)
def test_vlasov_ladder_converged_and_not():
    base = 3.7 + 0 * TIMES
    good = {l: base + 1e-4 * 2**l for l in range(4)}
    lad = D.vlasov_ladder(TIMES, good, (500, 4800), tol=0.037)
    assert lad["converged"] and lad["D"][1] < lad["D"][2]
    bad = {l: base + 0.1 * (3 - l) for l in range(4)}  # differences grow toward the fine end
    assert not D.vlasov_ladder(TIMES, bad, (500, 4800), tol=0.037)["converged"]
    assert not D.vlasov_ladder(TIMES, {0: base}, (500, 4800), tol=0.037)["converged"]


def test_onset_seed_rule():
    fit = {"models": {"lnN": {"coef": [100.0, 10.0]}}}
    assert D.onset_seed_rule(np.array([300, 301, 299, 300.5]), fit, 0.1)["triggered"] is False
    assert D.onset_seed_rule(np.array([300, 310, 290, 305]), fit, 0.1)["triggered"] is True
