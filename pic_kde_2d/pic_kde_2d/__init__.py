"""2D electrostatic PIC with dynamics-aware deposition and filtering."""

from .conventions import (
    CHARGE_TOL,
    DEFAULT_LANDAU_ALPHA,
    DEFAULT_LANDAU_K_LAMBDA_D,
    Q_OVER_M,
    SELF_FORCE_TOL,
    gaussian_transfer_2d,
    gaussian_transfer_mode_2d,
    landau_box_length,
    thermal_velocity,
)

__all__ = [
    "CHARGE_TOL",
    "DEFAULT_LANDAU_ALPHA",
    "DEFAULT_LANDAU_K_LAMBDA_D",
    "Q_OVER_M",
    "SELF_FORCE_TOL",
    "gaussian_transfer_2d",
    "gaussian_transfer_mode_2d",
    "landau_box_length",
    "thermal_velocity",
]
