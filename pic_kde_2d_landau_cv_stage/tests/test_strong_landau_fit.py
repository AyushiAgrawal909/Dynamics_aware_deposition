import numpy as np

from pic_kde_2d.validation.strong_landau_fit import (
    detect_first_bounce,
    fit_initial_decay,
)


def _synthetic_history(gamma=-0.08, bounce_time=15.0):
    times = np.linspace(0.0, 30.0, 301)
    decay = np.exp(gamma * times)
    recovery = np.where(
        times <= bounce_time,
        1.0,
        np.exp(0.20 * (times - bounce_time)),
    )
    mode = decay * recovery * np.exp(1j * 1.28 * times)
    return times, np.stack([mode, mode])


def test_detects_first_bounce_and_stops_before_recovery():
    times, histories = _synthetic_history()
    result = detect_first_bounce(
        histories,
        times,
        envelope_period=2.0,
        search_start=8.0,
        search_stop=25.0,
        slope_window=1.0,
        persistence_points=3,
    )
    assert result["bounce_detected"]
    assert 14.0 <= result["bounce_time"] <= 16.0
    assert result["safe_fit_stop"] < result["bounce_time"]


def test_prebounce_fit_recovers_decay_rate():
    times, histories = _synthetic_history(gamma=-0.08)
    result = fit_initial_decay(
        histories,
        times,
        fit_start=3.0,
        fit_stop=12.0,
        envelope_period=2.0,
        min_duration=3.0,
        min_r_squared=0.99,
    )
    assert result["quality_pass"]
    assert np.isclose(result["gamma"], -0.08, atol=1.0e-4)
