from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype.vale import Is

# Type aliases for numpy arrays with specific dimensions

# Input current matrix: (input_currents, time_points)
INPUT_CURRENT__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]

# Spike train matrix: (pools, neurons_per_pool, time_points)
SPIKE_TRAIN__MATRIX = Annotated[
    npt.NDArray[np.bool_],
    Is[lambda x: x.ndim == 3],
]

# MUAP shape tensor: (grid_positions, muap_index, electrode_grid_rows, electrode_grid_columns, muap_samples)
MUAP_SHAPE__TENSOR = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 5],
]

# Surface EMG tensor: (grid_positions, mu_pools, electrode_grid_rows, electrode_grid_columns, time)
SURFACE_EMG__TENSOR = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 5],
]
