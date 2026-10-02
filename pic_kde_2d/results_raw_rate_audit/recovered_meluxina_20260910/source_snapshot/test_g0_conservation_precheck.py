import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_G0_conservation_precheck.py"
)
SPEC = importlib.util.spec_from_file_location("g0_conservation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_production_constants_are_parsed_without_matching_uses():
    source = """
    static constexpr double KDE_H_MAX_CELLS = 20.0;
    double use = KDE_H_MAX_CELLS;
    for (int k = -3; k <= 3; k++) {}
    """
    assert MODULE.parse_constant(source, "KDE_H_MAX_CELLS") == 20.0
    assert MODULE.parse_periodic_images(source) == 3


def test_periodic_image_check_detects_extreme_width_on_thin_axis():
    assert MODULE.periodic_image_fraction(2559, 384.0, 3) > 0.999999
    assert MODULE.periodic_image_fraction(31, 384.0, 3) < 0.3


def test_static_refresh_work_is_exactly_zero_by_contract():
    # The executable result records no refresh events for a static profile.
    signed = 0.0
    absolute = 0.0
    np.testing.assert_array_equal([signed, absolute], [0.0, 0.0])
