"""Numerical deposition, gather, and filtering method interfaces."""

from .cic import deposit_cic_2d, gather_cic_2d
from .cic_wrappers import cic_deposit_method, cic_gather_method
from .gaussian import deposit_gaussian_2d_aniso, gather_gaussian_2d_aniso
from .gaussian_wrappers import make_gaussian_methods
from .spectral_W import apply_W_to_grid, make_W_box, make_W_gaussian, make_W_wiener, make_W_wiener_estimated
from .spectral_W_wrappers import (
    make_spectral_W_box_methods,
    make_spectral_W_gaussian_methods,
    make_spectral_W_methods,
)
from .sigma_selection import select_sigma_anisotropic_2d, sigma_max_deposit_only, sigma_max_matched
from .wu_qin_bandwidth import (
    CVMinimum,
    WuQinWidthUpdater,
    identify_cv_minimum,
    silverman_local_width_factors,
)

__all__ = [
    "cic_deposit_method",
    "cic_gather_method",
    "deposit_cic_2d",
    "gather_cic_2d",
    "deposit_gaussian_2d_aniso",
    "gather_gaussian_2d_aniso",
    "make_gaussian_methods",
    "make_W_gaussian",
    "make_W_box",
    "make_W_wiener",
    "make_W_wiener_estimated",
    "apply_W_to_grid",
    "make_spectral_W_methods",
    "make_spectral_W_gaussian_methods",
    "make_spectral_W_box_methods",
    "select_sigma_anisotropic_2d",
    "sigma_max_deposit_only",
    "sigma_max_matched",
    "CVMinimum",
    "WuQinWidthUpdater",
    "identify_cv_minimum",
    "silverman_local_width_factors",
]
