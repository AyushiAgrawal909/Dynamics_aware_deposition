import numpy as np

from pic_kde_2d.core.particles2d import Particles2D
from pic_kde_2d.methods.adaptive_sigma import (
    PeriodicLogCholeskyField,
    abramson_c4_bandwidth,
    diagonal_log_cholesky_coefficients,
    diagonal_bandwidth_adapter,
    log_cholesky_jacobian,
    log_cholesky_to_matrix,
    normalized_gaussian_shape_derivatives,
)
from pic_kde_2d.validation.controller_refresh import (
    ControllerRefreshAudit,
    cadence_sweep_table,
)
from pic_kde_2d.methods.controller_features import (
    build_controller_feature_field,
    estimate_causal_mode_features,
)


def test_log_cholesky_is_spd_and_jacobian_matches_finite_difference():
    ell = np.array([0.2, -0.3, -0.1])
    _, H = log_cholesky_to_matrix(ell)
    assert np.all(np.linalg.eigvalsh(H) > 0.0)
    jac = log_cholesky_jacobian(ell)
    eps = 1.0e-7
    for q in range(3):
        delta = np.zeros(3)
        delta[q] = eps
        _, plus = log_cholesky_to_matrix(ell + delta)
        _, minus = log_cholesky_to_matrix(ell - delta)
        np.testing.assert_allclose(jac[..., q], (plus - minus) / (2.0 * eps), rtol=2e-8, atol=2e-9)


def test_abramson_rule_is_local_and_never_exceeds_c4():
    density = np.array([0.01, 0.1, 1.0, 10.0])
    bandwidth = abramson_c4_bandwidth(
        density, sigma_pilot=np.array([1.0, 2.0]),
        sigma_c4_max=np.array([1.5, 3.0]), sigma_min=0.1,
    )
    assert bandwidth.shape == (4, 2)
    assert np.all(bandwidth <= np.array([1.5, 3.0]))
    assert np.all(np.diff(bandwidth[:, 0]) <= 0.0)
    assert np.all(np.diff(bandwidth[:, 1]) <= 0.0)


def test_production_diagonal_heads_insert_exactly_zero_shear():
    coefficients = diagonal_log_cholesky_coefficients(
        np.array([[0.1, -0.2], [0.3, 0.4]])
    )
    assert coefficients.shape == (2, 3)
    np.testing.assert_array_equal(coefficients[:, 1], 0.0)
    _, H = log_cholesky_to_matrix(coefficients)
    np.testing.assert_array_equal(H[:, 0, 1], 0.0)


def test_refresh_audit_separates_signed_work_from_absolute_injection():
    audit = ControllerRefreshAudit(cadence_steps=5)
    H0 = np.broadcast_to(np.eye(2), (4, 2, 2))
    H1 = 1.1 * H0
    audit.record(
        step=5, time=0.25, field_energy_before=2.0, field_energy_after=2.1,
        H_old=H0, H_new=H1, staleness_error=0.03,
    )
    audit.record(
        step=10, time=0.5, field_energy_before=2.1, field_energy_after=2.05,
        H_old=H1, H_new=H0, staleness_error=0.05,
    )
    summary = audit.summary()
    np.testing.assert_allclose(summary["cumulative_signed_controller_work"], 0.05)
    np.testing.assert_allclose(summary["cumulative_absolute_controller_work"], 0.15)
    assert summary["max_staleness_error"] == 0.05
    assert cadence_sweep_table([ControllerRefreshAudit(20), audit])[0] == summary


def test_periodic_field_is_c0_and_derivative_matches_finite_difference():
    rng = np.random.default_rng(7)
    nodes = rng.normal(scale=0.1, size=(8, 6, 3))
    field = PeriodicLogCholeskyField(nodes, Lx=8.0, Ly=6.0)
    left = field.evaluate(1.0e-9, 2.37)
    right = field.evaluate(8.0 - 1.0e-9, 2.37)
    np.testing.assert_allclose(left.H, right.H, rtol=0.0, atol=2.0e-9)

    x, y, eps = 2.31, 4.27, 1.0e-6
    center = field.evaluate(x, y)
    finite_x = (field.evaluate(x + eps, y).H - field.evaluate(x - eps, y).H) / (2.0 * eps)
    finite_y = (field.evaluate(x, y + eps).H - field.evaluate(x, y - eps).H) / (2.0 * eps)
    np.testing.assert_allclose(center.dH_dx, finite_x, rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(center.dH_dy, finite_y, rtol=2e-7, atol=2e-9)


def test_normalized_shape_derivative_includes_grad_h_and_conserves_charge():
    coeff = np.zeros((4, 4, 3))
    for i in range(4):
        coeff[i, :, 0] = -0.2 + 0.08 * i
        coeff[i, :, 2] = 0.1
    field = PeriodicLogCholeskyField(coeff, 4.0, 4.0)
    position = np.array([1.31, 2.17])
    evaluation = field.evaluate(*position)
    gx, gy = np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    shape = normalized_gaussian_shape_derivatives(
        nodes, position, evaluation, box_lengths=np.array([4.0, 4.0])
    )
    assert abs(np.sum(shape.weights) - 1.0) < 1.0e-14
    np.testing.assert_allclose(np.sum(shape.dweights_total, axis=0), 0.0, atol=2.0e-15)

    eps = 1.0e-6
    plus = normalized_gaussian_shape_derivatives(
        nodes, position + np.array([eps, 0.0]), field.evaluate(position[0] + eps, position[1]),
        box_lengths=np.array([4.0, 4.0]),
    ).weights
    minus = normalized_gaussian_shape_derivatives(
        nodes, position - np.array([eps, 0.0]), field.evaluate(position[0] - eps, position[1]),
        box_lengths=np.array([4.0, 4.0]),
    ).weights
    np.testing.assert_allclose(shape.dweights_total[:, 0], (plus - minus) / (2.0 * eps), rtol=2e-6, atol=2e-8)


def test_separable_adapter_rejects_rotated_tensor():
    field = PeriodicLogCholeskyField(np.zeros((2, 2, 3)), 2.0, 2.0)
    diagonal_bandwidth_adapter(field.evaluate(0.2, 0.3))
    rotated = np.zeros((2, 2, 3))
    rotated[..., 1] = 0.2
    try:
        diagonal_bandwidth_adapter(PeriodicLogCholeskyField(rotated, 2.0, 2.0).evaluate(0.2, 0.3))
    except ValueError as exc:
        assert "cannot represent" in str(exc)
    else:
        raise AssertionError("rotated covariance must fail closed")


def test_feature_pipeline_detects_counterstreaming_and_causal_growth():
    n = 256
    x = (np.arange(n) + 0.5) * 8.0 / n
    y = np.mod(3.0 * x, 4.0)
    vx = np.where(np.arange(n) % 2 == 0, 0.8, -0.8)
    p = Particles2D(x=x, y=y, vx=vx, vy=np.zeros(n), w=-np.ones(n), Lx=8.0, Ly=4.0)
    times = np.linspace(0.0, 2.0, 12)
    amplitudes = np.exp(0.3 * times)
    features = build_controller_feature_field(p, (8, 4), (times, amplitudes))
    assert features.values.shape == (8, 4, len(features.names))
    assert np.all(np.isfinite(features.values[..., :-1]))
    gamma_index = features.names.index("empirical_gamma")
    drive_index = features.names.index("counterstream_drive_proxy")
    np.testing.assert_allclose(features.values[..., gamma_index], 0.3, atol=1.0e-12)
    assert np.mean(features.values[..., drive_index]) > 1.0
    history = estimate_causal_mode_features(times, amplitudes)
    assert history.r2 > 0.999999
