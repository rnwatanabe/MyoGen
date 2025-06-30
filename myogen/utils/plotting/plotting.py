from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from myogen.utils.types import MUAP_SHAPE__TENSOR, SURFACE_EMG__TENSOR


def plot_muaps(save__path: Path, muap_shapes__tensor: MUAP_SHAPE__TENSOR):
    """
    Plot the MUAPs.

    Parameters
    ----------
    save__path: Path
        Path to save the plot
    muap_shapes__tensor: MUAP_SHAPE__TENSOR
        Tensor of shape (n_muaps, n_rows, n_cols, n_samples) containing MUAPs
    """
    for muap_idx in tqdm(range(muap_shapes__tensor.shape[0]), desc="Plotting MUAPs"):
        _, ax = plt.subplots(
            muap_shapes__tensor.shape[1],
            muap_shapes__tensor.shape[2],
            figsize=(
                muap_shapes__tensor.shape[1] * 2,
                muap_shapes__tensor.shape[2] * 2,
            ),
        )
        for row_idx in range(muap_shapes__tensor.shape[1]):
            for col_idx in range(muap_shapes__tensor.shape[2]):
                ax[row_idx, col_idx].plot(
                    muap_shapes__tensor[muap_idx, row_idx, col_idx]
                )
        plt.savefig(save__path / f"muap_{muap_idx}.png")
        plt.close()


def plot_muscle_distribution(
    save__path: Path,
    pos_MUs: list[tuple[float, float]],
    r_MUs: list[float],
    r: float,
    pos_Fibs: list[list[tuple[float, float]]],
    mean_fib_radius: float,
):
    """
    Plot the distribution of motor units and muscle fibers in the muscle.

    Parameters
    ----------
    save__path: Path
        Path to save the plot
    pos_MUs : list
        List of motor unit positions
    r_MUs : list
        List of motor unit radii
    r : float
        Muscle radius
    pos_Fibs : list
        List of fiber positions for each motor unit
    mean_fib_radius : float
        Mean radius of muscle fibers
    """
    _, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(plt.Circle((0, 0), r, color="r", alpha=0.1))

    for i in range(len(pos_MUs)):
        ax.add_patch(plt.Circle(pos_MUs[i], r_MUs[i], color="r", alpha=0.5))
        for j in range(len(pos_Fibs[i])):
            pos_fib = (
                pos_MUs[i][0] + pos_Fibs[i][j][0],
                pos_MUs[i][1] + pos_Fibs[i][j][1],
            )
            ax.add_patch(plt.Circle(pos_fib, mean_fib_radius, color="b", alpha=1))

    ax.set_aspect("equal", adjustable="datalim")
    ax.plot()  # Causes an autoscale update.
    plt.savefig(save__path / "muscle_distribution.png")
    plt.close()


def plot_surface_emg(save__path: Path, surface_emg__tensor: SURFACE_EMG__TENSOR):
    """
    Plot the EMG signal.

    Parameters
    ----------
    save__path: Path
        Path to save the plot
    surface_emg__tensor: SURFACE_EMG__TENSOR
        Tensor of shape (n_pools, n_rows, n_cols, n_time) containing EMG signals
    """
    for pool_idx in tqdm(
        range(surface_emg__tensor.shape[0]), desc="Plotting surface EMG"
    ):
        _, ax = plt.subplots(
            surface_emg__tensor.shape[1],
            surface_emg__tensor.shape[2],
            figsize=(
                surface_emg__tensor.shape[1] * 2,
                surface_emg__tensor.shape[2] * 2,
            ),
        )

        for row_idx in range(surface_emg__tensor.shape[1]):
            for col_idx in range(surface_emg__tensor.shape[2]):
                ax[row_idx, col_idx].plot(
                    surface_emg__tensor[pool_idx, row_idx, col_idx]
                )
        plt.savefig(save__path / f"emg_pool_{pool_idx}.svg")
        plt.close()
