"""DABS: dynamics-aware selection of deposition/gather filters for PIC.

Choose the filter family and bandwidth of a matched (deposit + gather)
mesh filter that minimises a statistical error objective while keeping the
linear growth or damping rates of protected modes within a tolerance, as in
Agrawal, Lin, Krishnasamy and Bouvry, "Dynamics-aware bandwidth selection for
smoothed charge deposition in 2D electrostatic PIC".

Quick start::

    from dabs import Grid, Plasma, Species, ProtectedMode, ModelObjective, Selector
    grid = Grid([128, 16], [64.0, 16.0], shape="cic")
    plasma = Plasma([Species(-1, 1, 0.5, [0.8, 0], 0.05),
                     Species(-1, 1, 0.5, [-0.8, 0], 0.05)]).resolve(grid.dim)
    obj = ModelObjective(grid, plasma, ppc=32, signal=[{"mode": [12, 0], "amplitude": 1e-3}])
    result = Selector(grid, plasma, [ProtectedMode((12, 1))], obj, tolerance=0.05).run()
    print(result.summary())
"""

__version__ = "0.1.0"

from .grid import Grid  # noqa: E402
from .plasma import Plasma, Species  # noqa: E402
from .dispersion import ModeDispersion  # noqa: E402
from .filters import Binomial, Custom, Gaussian, Kernel, NoFilter, binomial_stencil  # noqa: E402
from .objectives import ModelObjective, SnapshotObjective  # noqa: E402
from .selection import ProtectedMode, SelectionResult, Selector, unstable_modes  # noqa: E402
from .config import load as load_config, run as run_config  # noqa: E402
from .deposit import deposit  # noqa: E402

__all__ = [
    "Grid", "Plasma", "Species", "ModeDispersion", "Gaussian", "Binomial", "Custom", "NoFilter",
    "Kernel", "binomial_stencil", "ModelObjective", "SnapshotObjective", "ProtectedMode",
    "Selector", "SelectionResult", "unstable_modes", "load_config", "run_config", "deposit",
]
