"""
IntramuscularArray class for simulating intramuscular EMG electrodes.
"""

import numpy as np
from matplotlib import pyplot as plt

from myogen import simulator


class IntramuscularArray:
    """
    Class representing an intramuscular needle electrode array.



    Attributes
    ----------
    n_electrodes : int
        Number of electrode points
    spacing : float
        Spacing between electrodes in mm
    arrangement : str
        Arrangement type ('linear' or 'grid')
    pts : np.ndarray
        Electrode positions (x, y, z) in mm - includes all trajectory positions
    pts_init : np.ndarray
        Initial electrode positions before trajectory
    n_points : int
        Number of electrode points (same as n_electrodes)
    normals : np.ndarray
        Normal vectors for each electrode
    diff_mat : np.ndarray
        Differential recording matrix
    n_channels : int
        Number of differential channels
    trajectory_distance : float
        Distance of trajectory movement
    trajectory_steps : int
        Number of trajectory steps
    n_nodes : int
        Number of trajectory nodes
    """

    def __init__(
        self, n_electrodes: int = 2, spacing: float = 0.5, arrangement: str = "linear"
    ):
        """
        Initialize the intramuscular electrode array.

        Parameters
        ----------
        n_electrodes : int, optional
            Number of electrode points (default: 2)
        spacing : float, optional
            Spacing between electrodes in mm (default: 0.5)
        arrangement : str, optional
            Arrangement type: 'linear' or 'grid' (default: 'linear')
        """
        self.n_electrodes = n_electrodes
        self.n_points = n_electrodes  # Alias for compatibility
        self.spacing = spacing
        self.arrangement = arrangement

        # Movement parameters
        self.trajectory_distance = 0.0
        self.trajectory_steps = 1
        self.n_nodes = 1  # Number of trajectory nodes

        # Generate initial electrode points based on arrangement
        self._generate_electrode_points()

        # Additional properties for MUAP calculations
        self.normals = None  # Electrode normal vectors
        self.diff_mat = None  # Differential recording matrix
        self.n_channels = None  # Number of differential channels

        # Initialize basic differential recording
        self._setup_differential_recording()

    def _generate_electrode_points(self):
        """Generate electrode points based on the specified arrangement."""
        if self.arrangement == "linear":
            # Linear arrangement along z-axis (typical for needle electrodes)
            self.pts_init = np.column_stack(
                [
                    np.zeros(self.n_electrodes),
                    np.zeros(self.n_electrodes),
                    np.linspace(
                        0, (self.n_electrodes - 1) * self.spacing, self.n_electrodes
                    ),
                ]
            )
        elif self.arrangement == "grid":
            # Grid arrangement (for specialized multi-channel arrays)
            n_per_side = int(np.ceil(np.sqrt(self.n_electrodes)))
            x_pos, y_pos = np.meshgrid(
                np.linspace(0, (n_per_side - 1) * self.spacing, n_per_side),
                np.linspace(0, (n_per_side - 1) * self.spacing, n_per_side),
            )
            # Take only the required number of points
            x_flat = x_pos.flatten()[: self.n_electrodes]
            y_flat = y_pos.flatten()[: self.n_electrodes]
            self.pts_init = np.column_stack(
                [
                    x_flat,
                    y_flat,
                    np.zeros(self.n_electrodes),  # z = 0 for grid
                ]
            )
        else:
            raise ValueError(f"Unknown arrangement: {self.arrangement}")

        # Initially, pts is the same as pts_init
        self.pts = self.pts_init.copy()

        # Initialize normal vectors (pointing radially outward by default)
        self.normals = np.zeros((self.n_electrodes, 3))
        self.normals[:, 0] = 1  # Default: pointing in x-direction

    def _calc_observation_points(self):
        """Calculate all observation points along the trajectory (like MATLAB version)."""
        if self.n_nodes <= 1:
            # No trajectory, just use initial points
            self.pts = self.pts_init.copy()
            return

        # Calculate all trajectory positions and concatenate them
        all_pts = []
        for i in range(self.n_nodes):
            # Linear movement along x-axis for simplicity
            # In a more complex implementation, this could be any trajectory
            if self.n_nodes > 1:
                t = i / (self.n_nodes - 1)  # Normalized parameter [0, 1]
                offset = np.array([t * self.trajectory_distance, 0, 0])
            else:
                offset = np.array([0, 0, 0])

            # Translate all electrode points by the offset
            trajectory_pts = self.pts_init + offset
            all_pts.append(trajectory_pts)

        # Concatenate all trajectory positions
        self.pts = np.vstack(all_pts)

        # Extend the differential matrix to cover all trajectory nodes
        if self.diff_mat is not None:
            self.diff_mat = np.tile(self.diff_mat, (1, self.n_nodes))

    def _setup_differential_recording(self):
        """Setup differential recording configuration."""
        if self.n_electrodes >= 2:
            # Simple bipolar configuration for linear arrays
            self.n_channels = self.n_electrodes - 1
            self.diff_mat = np.zeros((self.n_channels, self.n_electrodes))
            for i in range(self.n_channels):
                self.diff_mat[i, i] = 1  # Positive electrode
                self.diff_mat[i, i + 1] = -1  # Negative electrode
        else:
            # Single electrode - monopolar recording
            self.n_channels = 1
            self.diff_mat = np.ones((1, self.n_electrodes))

    def set_position(self, position: np.ndarray, orientation: np.ndarray):
        """
        Set the position and orientation of the electrode array.

        Parameters
        ----------
        position : np.ndarray
            3D position [x, y, z] in mm
        orientation : np.ndarray
            3D orientation [rx, ry, rz] in radians
        """
        position = np.array(position)
        orientation = np.array(orientation)

        # Apply rotation matrices for each axis
        # Rotation around x-axis
        rx = orientation[0]
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)],
            ]
        )

        # Rotation around y-axis
        ry = orientation[1]
        Ry = np.array(
            [
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)],
            ]
        )

        # Rotation around z-axis
        rz = orientation[2]
        Rz = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation and translation to each electrode point
        self.pts_init = (R @ self.pts_init.T).T + position
        self.pts = self.pts_init.copy()

    def set_linear_trajectory(self, distance: float, steps: int):
        """
        Set a linear trajectory for electrode movement.

        Parameters
        ----------
        distance : float
            Total distance to move in mm
        steps : int
            Number of steps in the trajectory
        """
        self.trajectory_distance = distance
        self.trajectory_steps = steps
        self.n_nodes = steps  # Set n_nodes to match trajectory steps

        # Calculate all observation points along trajectory
        self._calc_observation_points()

    def get_electrode_positions(self) -> np.ndarray:
        """
        Get current electrode positions.

        Returns
        -------
        np.ndarray
            Array of electrode positions with shape (n_electrodes, 3)
        """
        return self.pts.copy()

    def get_initial_positions(self) -> np.ndarray:
        """
        Get initial electrode positions.

        Returns
        -------
        np.ndarray
            Array of initial electrode positions with shape (n_electrodes, 3)
        """
        return self.pts_init.copy()

    def get_differential_matrix(self) -> np.ndarray:
        """
        Get the differential recording matrix.

        Returns
        -------
        np.ndarray
            Differential matrix with shape (n_channels, n_electrodes)
        """
        return self.diff_mat.copy()

    def traj_mixing_fun(self, t: float, n_nodes: int, node: int) -> float:
        """
        Trajectory mixing function.

        Parameters
        ----------
        t : float
            Normalized parameter (between 0 and 1)
        n_nodes : int
            Number of trajectory nodes
        node : int
            Current node (1-indexed to match MATLAB)

        Returns
        -------
        float
            Mixing weight for this node
        """
        # Convert to 0-indexed
        node_idx = node - 1

        # Normalized node position
        node_pos = node_idx / (n_nodes - 1 + np.finfo(float).eps)

        # Linear interpolation weight (triangular function)
        return max(0, 1 - (n_nodes - 1) * abs(t - node_pos))

    def traj_mixing_mat(self, t: float, n_nodes: int, n_channels: int) -> np.ndarray:
        """
        Generate trajectory mixing matrix - exactly like MATLAB implementation.

        MATLAB: diag(reshape(repmat(traj_mixing_fun(t, n_nodes, 1:n_nodes), n_points, 1), [], 1))

        Parameters
        ----------
        t : float
            Normalized parameter (between 0 and 1)
        n_nodes : int
            Number of trajectory nodes
        n_channels : int
            Number of channels

        Returns
        -------
        np.ndarray
            Diagonal mixing matrix
        """
        # Calculate mixing weights for all nodes (1:n_nodes in MATLAB)
        weights = []
        for node in range(1, n_nodes + 1):  # 1-indexed to match MATLAB
            weight = self.traj_mixing_fun(t, n_nodes, node)
            weights.append(weight)

        # MATLAB: repmat(weights, n_points, 1) then reshape([], 1)
        # This repeats each weight n_points times, then flattens
        weights_array = np.array(weights)
        repeated_weights = np.tile(
            weights_array, (self.n_points, 1)
        )  # repmat in MATLAB
        full_weights = repeated_weights.flatten(
            "F"
        )  # reshape([], 1) with Fortran order

        # Create diagonal matrix
        return np.diag(full_weights)

    def __str__(self):
        """String representation of the electrode array."""
        return (
            f"IntramuscularArray({self.n_electrodes} electrodes, "
            f"{self.spacing}mm spacing, {self.arrangement} arrangement)"
        )

    def __repr__(self):
        """Detailed representation of the electrode array."""
        return self.__str__()


if __name__ == "__main__":
    recruitment_thresholds, _ = simulator.generate_mu_recruitment_thresholds(
        N=100, recruitment_range=25
    )

    Lmuscle = 30
    Dmf: int = (
        400  # Density of muscle fibers per square millimetre (Hamilton-Wright 2005)
    )
    Nmf: int = 34000  # Expected number of muscle fibers in the muscle (40k for FDI, see Feinstein - Morphologic studies ... 1995)
    Lmuscle: float = 30  # [mm]
    Rmuscle: float = np.sqrt((Nmf / Dmf) / np.pi)
    ima = IntramuscularArray(n_electrodes=2, spacing=0.5, arrangement="linear")
    ima.set_position(
        np.array([0, 0, 2 * Lmuscle / 3]) - np.array([0.5, 0, 0]),
        [-np.pi / 2, 0, -np.pi / 2],
    )
    ima.set_linear_trajectory(0.125, 4)
    print(ima.pts)
    print(ima.diff_mat)

    msucle = simulator.Muscle(
        recruitment_thresholds=recruitment_thresholds,
        radius__mm=Rmuscle,
        fiber_density__fibers_per_mm2=50,
        autorun=True,
    )

    x_circle = np.linspace(-Rmuscle, Rmuscle, 1000)
    y_circle_pos = np.sqrt(Rmuscle**2 - x_circle**2)
    x_circle = np.concatenate([x_circle, x_circle[::-1]])
    y_circle = np.concatenate([y_circle_pos, -y_circle_pos[::-1]])

    fig = plt.figure(figsize=(15, 9))

    plt.plot(x_circle, y_circle, "k-", linewidth=1)

    # Plot motor unit innervation_center_positions with labels
    for i in range(len(msucle.innervation_center_positions)):
        plt.text(
            x=msucle.innervation_center_positions[i, 0],
            y=msucle.innervation_center_positions[i, 1],
            s=str(i + 1),
            fontsize=8,
            ha="center",
            va="center",
        )

    plt.axis("equal")
    saved_xlim = plt.xlim()
    saved_ylim = plt.ylim()

    electrode_positions = ima.get_electrode_positions()
    electrode_initial = ima.get_initial_positions()

    for i in range(electrode_positions.shape[0]):
        plt.plot(
            electrode_positions[i, 0], electrode_positions[i, 1], "k.", markersize=8
        )

    for i in range(electrode_initial.shape[0]):
        plt.plot(electrode_initial[i, 0], electrode_initial[i, 1], "bo", markersize=6)

    plt.title("Cross-section view")
    plt.xlabel("Height, mm")
    plt.ylabel("Width, mm")
    plt.grid(True, alpha=0.3)
    plt.show()
