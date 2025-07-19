"""
Electrode configuration framework for EMG simulation.
"""

from typing import Tuple, Literal, Optional, Dict, Any
import numpy as np


class SurfaceElectrodeArray:
    """
    Surface electrode array for EMG recording.

    Represents a grid of surface electrodes with configurable spacing,
    size, and differentiation modes.

    Parameters
    ----------
    num_rows : int
        Number of rows in the electrode array
    num_cols : int
        Number of columns in the electrode array
    inter_electrode_distances__mm : float
        Inter-electrode distances in mm.
    electrode_radius__mm : float, optional
        Radius of the electrodes in mm
    center_point__mm_deg : Tuple[float, float]
        Position along z in mm and rotation around the muscle theta in degrees.
    bending_radius__mm : float, optional
        Bending radius around which the electrode grid is bent. Usually this is equal to the radius of the muscle.
    rotation_angle__deg : float, optional
        Rotation angle of the electrodes in degrees. This is the angle between the electrode grid and the muscle surface.
    differentiation_mode : {"monopolar", "bipolar_longitudinal", "bipolar_transversal", "laplacian"}
        Differentiation mode. Default is monopolar.
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        inter_electrode_distances__mm: float,
        electrode_radius__mm: float,
        center_point__mm_deg: Tuple[float, float] = (0.0, 0.0),
        bending_radius__mm: float = 0.0,
        rotation_angle__deg: float = 0.0,
        differentiation_mode: Literal[
            "monopolar", "bipolar_longitudinal", "bipolar_transversal", "laplacian"
        ] = "monopolar",
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.center_point__mm_deg = center_point__mm_deg
        self.bending_radius__mm = bending_radius__mm
        self.rotation_angle__deg = rotation_angle__deg
        self.inter_electrode_distances__mm = inter_electrode_distances__mm
        self.electrode_radius__mm = electrode_radius__mm
        self.differentiation_mode = differentiation_mode

        self.num_electrodes = num_rows * num_cols

        if self.bending_radius__mm == 0:
            self.bending_radius__mm = np.finfo(np.float32).eps

        # Set up channel configuration based on differentiation mode
        if differentiation_mode == "monopolar":
            self.num_channels = self.num_electrodes
        elif differentiation_mode in ["bipolar_longitudinal", "bipolar_transversal"]:
            # For bipolar, we lose one channel per dimension
            if differentiation_mode == "bipolar_longitudinal":
                self.num_channels = max(1, num_rows - 1) * num_cols
            else:  # bipolar_transversal
                self.num_channels = num_rows * max(1, num_cols - 1)
        elif differentiation_mode == "laplacian":
            # For Laplacian, we lose border electrodes
            self.num_channels = max(1, num_rows - 2) * max(1, num_cols - 2)
        else:
            self.num_channels = self.num_electrodes

        # Create electrode grid in local coordinate system
        self._create_electrode_grid()

    def _create_electrode_grid(self) -> None:
        """Create electrode positions in local coordinate system."""
        self.pos_z = np.zeros((self.num_rows, self.num_cols))
        if self.num_rows % 2 == 1:
            index_center = int((self.num_rows - 1) / 2)
            self.pos_z[index_center, :] = self.center_point__mm_deg[0]
            for i in range(1, index_center + 1):
                self.pos_z[index_center + i, :] = (
                    self.pos_z[index_center + i - 1, :]
                    + self.inter_electrode_distances__mm
                )
                self.pos_z[index_center - i, :] = (
                    self.pos_z[index_center - i + 1, :]
                    - self.inter_electrode_distances__mm
                )
        else:
            index_center1 = int(self.num_rows / 2)
            index_center2 = index_center1 - 1
            self.pos_z[index_center1, :] = (
                self.center_point__mm_deg[0] + self.inter_electrode_distances__mm / 2
            )
            self.pos_z[index_center2, :] = (
                self.center_point__mm_deg[0] - self.inter_electrode_distances__mm / 2
            )
            for i in range(1, index_center2 + 1):
                self.pos_z[index_center1 + i, :] = (
                    self.pos_z[index_center1 + i - 1, :]
                    + self.inter_electrode_distances__mm
                )
                self.pos_z[index_center2 - i, :] = (
                    self.pos_z[index_center2 - i + 1, :]
                    - self.inter_electrode_distances__mm
                )

        self.pos_theta = np.zeros((self.num_rows, self.num_cols))
        if self.num_cols % 2 == 1:
            index_center = int((self.num_cols - 1) / 2)
            self.pos_theta[:, index_center] = self.center_point__mm_deg[1] * np.pi / 180
            for i in range(1, index_center + 1):
                self.pos_theta[:, index_center + i] = (
                    self.pos_theta[:, index_center + i - 1]
                    + self.inter_electrode_distances__mm / self.bending_radius__mm
                )
                self.pos_theta[:, index_center - i] = (
                    self.pos_theta[:, index_center - i + 1]
                    - self.inter_electrode_distances__mm / self.bending_radius__mm
                )
        else:
            index_center1 = int(self.num_cols / 2)
            index_center2 = index_center1 - 1
            self.pos_theta[:, index_center1] = (
                self.center_point__mm_deg[1] * np.pi / 180
                + self.inter_electrode_distances__mm / 2 / self.bending_radius__mm
            )
            self.pos_theta[:, index_center2] = (
                self.center_point__mm_deg[1] * np.pi / 180
                - self.inter_electrode_distances__mm / 2 / self.bending_radius__mm
            )
            for i in range(1, index_center2 + 1):
                self.pos_theta[:, index_center1 + i] = (
                    self.pos_theta[:, index_center1 + i - 1]
                    + self.inter_electrode_distances__mm / self.bending_radius__mm
                )
                self.pos_theta[:, index_center2 - i] = (
                    self.pos_theta[:, index_center2 - i + 1]
                    - self.inter_electrode_distances__mm / self.bending_radius__mm
                )

        ## Rotated detection system (Farina, 2004), eq (36)
        displacement = self.center_point__mm_deg[0] * np.ones(self.pos_z.shape)
        self.pos_z = (
            -self.bending_radius__mm
            * np.sin(self.rotation_angle__deg * np.pi / 180)
            * self.pos_theta
            + np.cos(self.rotation_angle__deg * np.pi / 180)
            * (self.pos_z - displacement)
            + displacement
        )
        self.pos_theta = (
            np.cos(self.rotation_angle__deg * np.pi / 180) * self.pos_theta
            + np.sin(self.rotation_angle__deg * np.pi / 180)
            * (self.pos_z - displacement)
            / self.bending_radius__mm
        )

    def get_H_sf(
        self, ktheta_mesh_kzktheta: np.ndarray, kz_mesh_kzktheta: np.ndarray
    ) -> np.ndarray | float:
        """
        Get the spatial filter for the electrode array.

        Parameters
        ----------
        ktheta_mesh_kzktheta : np.ndarray
            Angular spatial frequency mesh
        kz_mesh_kzktheta : np.ndarray
            Longitudinal spatial frequency mesh

        Returns
        -------
        H_sf : np.ndarray or float
            Spatial filter for the specified differentiation mode
        """
        if self.differentiation_mode == "monopolar":
            H_sf = 1.0

        elif self.differentiation_mode == "bipolar_longitudinal":
            # Differential along muscle fiber direction (z-axis)
            # Apply coordinate transformation for rotation
            alpha_rad = self.rotation_angle__deg * np.pi / 180
            kz_new = ktheta_mesh_kzktheta / self.bending_radius__mm * np.sin(
                alpha_rad
            ) + kz_mesh_kzktheta * np.cos(alpha_rad)
            # Spatial filter for longitudinal differential
            H_sf = np.exp(1j * kz_new) - np.exp(-1j * kz_new)

        elif self.differentiation_mode == "bipolar_transversal":
            # Differential around muscle circumference (theta-axis)
            # Apply coordinate transformation for rotation
            alpha_rad = self.rotation_angle__deg * np.pi / 180
            ktheta_new = ktheta_mesh_kzktheta * np.cos(
                alpha_rad
            ) - kz_mesh_kzktheta * self.bending_radius__mm * np.sin(alpha_rad)
            # Spatial filter for transversal differential
            H_sf = np.exp(1j * ktheta_new / self.bending_radius__mm) - np.exp(
                -1j * ktheta_new / self.bending_radius__mm
            )

        elif self.differentiation_mode == "laplacian":
            # Laplacian (second-order spatial differential)
            # Combination of longitudinal and transversal second derivatives
            alpha_rad = self.rotation_angle__deg * np.pi / 180
            kz_new = ktheta_mesh_kzktheta / self.bending_radius__mm * np.sin(
                alpha_rad
            ) + kz_mesh_kzktheta * np.cos(alpha_rad)
            ktheta_new = ktheta_mesh_kzktheta * np.cos(
                alpha_rad
            ) - kz_mesh_kzktheta * self.bending_radius__mm * np.sin(alpha_rad)

            # Laplacian approximation: -k^2 in frequency domain
            k_total_sq = kz_new**2 + (ktheta_new / self.bending_radius__mm) ** 2
            H_sf = -k_total_sq

        return H_sf


class IntramuscularElectrodeArray:
    """
    Intramuscular electrode array for EMG recording.

    Represents a linear array of intramuscular electrodes (needle electrodes)
    with configurable spacing and differentiation modes.

    Parameters
    ----------
    num_electrodes : int
        Number of electrodes in the array
    inter_electrode_distance__mm : float, default=0.5
        Inter-electrode distance in mm
    position__mm : Tuple[float, float, float], default=(0.0, 0.0, 0.0)
        Position of the electrode array center in mm (x, y, z coordinates)
    orientation__rad : Tuple[float, float, float], default=(0.0, 0.0, 0.0)
        Orientation of the electrode array in radians (roll, pitch, yaw)
    arrangement : Literal["linear", "grid"], default="linear"
        Arrangement type of electrodes
    differentiation_mode : Literal["monopolar", "consecutive", "reference"], default="consecutive"
        Differentiation mode for recording
    trajectory_distance__mm : float, default=0.0
        Distance for trajectory movement in mm
    trajectory_steps : int, default=1
        Number of steps in the trajectory
    """

    def __init__(
        self,
        num_electrodes: int,
        inter_electrode_distance__mm: float = 0.5,
        position__mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation__rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        arrangement: Literal["linear", "grid"] = "linear",
        differentiation_mode: Literal[
            "monopolar", "consecutive", "reference"
        ] = "consecutive",
        trajectory_distance__mm: float = 0.0,
        trajectory_steps: int = 1,
    ):
        self.num_electrodes = num_electrodes
        self.inter_electrode_distance__mm = inter_electrode_distance__mm
        self.position__mm = position__mm
        self.orientation__rad = orientation__rad
        self.arrangement = arrangement
        self.differentiation_mode = differentiation_mode
        self.trajectory_distance__mm = trajectory_distance__mm
        self.trajectory_steps = trajectory_steps

        self.num_points = num_electrodes  # Alias for compatibility
        self.n_nodes = trajectory_steps

        # Set electrode type description
        if self.num_electrodes == 1:
            self.type = "single point"
            self.type_short = "1p"
        elif self.num_electrodes == 2:
            self.type = "2-point differential"
            self.type_short = "2p_diff"
        else:
            self.type = f"intramuscular_{self.num_electrodes}p_array"
            self.type_short = f"{self.num_electrodes}p_array"

        # Set up channel configuration based on differentiation mode
        if differentiation_mode == "monopolar":
            self.num_channels = self.num_electrodes
        elif differentiation_mode == "consecutive":
            # Consecutive differential: adjacent electrode pairs
            self.num_channels = max(1, self.num_electrodes - 1)
        elif differentiation_mode == "reference":
            # Reference differential: all electrodes vs reference
            self.num_channels = max(1, self.num_electrodes - 1)
        else:
            self.num_channels = self.num_electrodes

        # Create electrode positions and differential matrix
        self._create_electrode_positions()
        self._create_differential_matrix()

    def _create_electrode_positions(self) -> None:
        """Create electrode positions in 3D space."""
        if self.arrangement == "linear":
            # Linear arrangement along z-axis (typical for needle electrodes)
            self.pts_origin = np.column_stack(
                [
                    np.zeros(self.num_electrodes),
                    np.zeros(self.num_electrodes),
                    np.linspace(
                        0,
                        (self.num_electrodes - 1) * self.inter_electrode_distance__mm,
                        self.num_electrodes,
                    ),
                ]
            )
        elif self.arrangement == "grid":
            # Grid arrangement (for specialized multi-channel arrays)
            n_per_side = int(np.ceil(np.sqrt(self.num_electrodes)))
            x_pos, y_pos = np.meshgrid(
                np.linspace(
                    0, (n_per_side - 1) * self.inter_electrode_distance__mm, n_per_side
                ),
                np.linspace(
                    0, (n_per_side - 1) * self.inter_electrode_distance__mm, n_per_side
                ),
            )
            # Take only the required number of points
            x_flat = x_pos.flatten()[: self.num_electrodes]
            y_flat = y_pos.flatten()[: self.num_electrodes]
            self.pts_origin = np.column_stack(
                [
                    x_flat,
                    y_flat,
                    np.zeros(self.num_electrodes),  # z = 0 for grid
                ]
            )
        else:
            raise ValueError(f"Unknown arrangement: {self.arrangement}")

        # Apply rotation and translation
        self._apply_transformation()

        # Calculate trajectory points if needed
        if self.trajectory_steps > 1:
            self._calculate_trajectory_points()
        else:
            self.pts = self.pts_init.copy()

    def _apply_transformation(self) -> None:
        """Apply rotation and translation to electrode positions."""
        # Apply rotation matrices for each axis
        # Rotation around x-axis
        rx = self.orientation__rad[0]
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )

        # Rotation around y-axis
        ry = self.orientation__rad[1]
        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )

        # Rotation around z-axis
        rz = self.orientation__rad[2]
        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation and translation
        self.pts_init = (R @ self.pts_origin.T).T + np.array(self.position__mm)

    def _calculate_trajectory_points(self) -> None:
        """Calculate all observation points along the trajectory."""
        all_pts = []
        for i in range(self.n_nodes):
            # Linear movement along x-axis by default
            if self.n_nodes > 1:
                t = i / (self.n_nodes - 1)  # Normalized parameter [0, 1]
                offset = np.array([t * self.trajectory_distance__mm, 0, 0])
            else:
                offset = np.array([0, 0, 0])

            # Translate all electrode points by the offset
            trajectory_pts = self.pts_init + offset
            all_pts.append(trajectory_pts)

        # Concatenate all trajectory positions
        self.pts = np.vstack(all_pts)

        # Extend the differential matrix to cover all trajectory nodes
        if hasattr(self, "diff_mat") and self.diff_mat is not None:
            self.diff_mat = np.tile(self.diff_mat, (1, self.n_nodes))

    def _create_differential_matrix(self) -> None:
        """Create differential recording matrix based on differentiation mode."""
        if self.differentiation_mode == "monopolar":
            # Monopolar recording: each electrode is a separate channel
            self.diff_mat = np.eye(self.num_electrodes)
        elif self.differentiation_mode == "consecutive":
            # Consecutive differential: adjacent electrode pairs
            if self.num_electrodes >= 2:
                self.diff_mat = np.zeros((self.num_channels, self.num_electrodes))
                for i in range(self.num_channels):
                    self.diff_mat[i, i] = 1  # Positive electrode
                    self.diff_mat[i, i + 1] = -1  # Negative electrode
            else:
                self.diff_mat = np.ones((1, self.num_electrodes))
        elif self.differentiation_mode == "reference":
            # Reference differential: all electrodes vs reference (last electrode)
            if self.num_electrodes >= 2:
                self.diff_mat = np.zeros((self.num_channels, self.num_electrodes))
                for i in range(self.num_channels):
                    self.diff_mat[i, i + 1] = 1  # Signal electrode
                    self.diff_mat[i, 0] = -1  # Reference electrode (first)
            else:
                self.diff_mat = np.ones((1, self.num_electrodes))
        else:
            # Default to monopolar
            self.diff_mat = np.eye(self.num_electrodes)

    def get_electrode_positions_for_simulation(
        self, trajectory_node: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get electrode positions formatted for simulation.

        Parameters
        ----------
        trajectory_node : int, default=0
            Trajectory node index to get positions for

        Returns
        -------
        tuple
            Tuple containing:
            - electrode_positions: (num_electrodes, 3) array of electrode positions
            - differential_matrix: (num_channels, num_electrodes) differential matrix
        """
        if self.trajectory_steps > 1 and trajectory_node < self.n_nodes:
            # Get positions for specific trajectory node
            start_idx = trajectory_node * self.num_electrodes
            end_idx = start_idx + self.num_electrodes
            electrode_positions = self.pts[start_idx:end_idx, :]
        else:
            # Use initial positions
            electrode_positions = self.pts_init

        return electrode_positions, self.diff_mat

    def get_differential_matrix(self) -> np.ndarray:
        """
        Get the differential recording matrix.

        Returns
        -------
        np.ndarray
            Differential matrix with shape (num_channels, num_electrodes)
        """
        return self.diff_mat.copy()

    def set_trajectory(self, distance__mm: float, steps: int) -> None:
        """
        Set linear trajectory for electrode movement.

        Parameters
        ----------
        distance__mm : float
            Total distance to move in mm
        steps : int
            Number of steps in the trajectory
        """
        self.trajectory_distance__mm = distance__mm
        self.trajectory_steps = steps
        self.n_nodes = steps

        # Recalculate trajectory points
        if steps > 1:
            self._calculate_trajectory_points()
        else:
            self.pts = self.pts_init.copy()

    def traj_mixing_fun(self, t: float, n_nodes: int, node: int) -> float:
        """
        Trajectory mixing function for interpolating between trajectory nodes.

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
        Generate trajectory mixing matrix for smooth interpolation between nodes.

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
            Diagonal mixing matrix for trajectory interpolation
        """
        # Calculate mixing weights for all nodes
        weights = []
        for node in range(1, n_nodes + 1):  # 1-indexed to match MATLAB
            weight = self.traj_mixing_fun(t, n_nodes, node)
            weights.append(weight)

        # Repeat each weight for each electrode point
        weights_array = np.array(weights)
        repeated_weights = np.tile(weights_array, (self.num_points, 1))
        full_weights = repeated_weights.flatten("F")  # Fortran order to match MATLAB

        # Create diagonal matrix
        return np.diag(full_weights)

    def __str__(self) -> str:
        """String representation of the electrode array."""
        return (
            f"IntramuscularElectrodeArray({self.num_electrodes} electrodes, "
            f"{self.inter_electrode_distance__mm}mm spacing, {self.arrangement} arrangement, "
            f"{self.differentiation_mode} mode)"
        )

    def __repr__(self) -> str:
        """Detailed representation of the electrode array."""
        return self.__str__()

    @classmethod
    def create_single_differential(
        cls,
        inter_electrode_distance__mm: float = 0.5,
        position__mm: Tuple[float, float, float] = (0.0, 0.0, 20.0),
        orientation__rad: Tuple[float, float, float] = (-np.pi / 2, 0, -np.pi / 2),
        trajectory_distance__mm: float = 0.125,
        trajectory_steps: int = 4,
    ) -> "IntramuscularElectrodeArray":
        """
        Create a single-channel differential electrode (equivalent to MATLAB s07 example).

        Parameters
        ----------
        inter_electrode_distance__mm : float, default=0.5
            Distance between electrode contacts in mm
        position__mm : Tuple[float, float, float], default=(0.0, 0.0, 20.0)
            Initial position of electrode array center
        orientation__rad : Tuple[float, float, float], default=(-π/2, 0, -π/2)
            Electrode orientation (matches MATLAB default)
        trajectory_distance__mm : float, default=0.125
            Distance for scanning trajectory
        trajectory_steps : int, default=4
            Number of trajectory steps

        Returns
        -------
        IntramuscularElectrodeArray
            Configured electrode array
        """
        return cls(
            num_electrodes=2,
            inter_electrode_distance__mm=inter_electrode_distance__mm,
            position__mm=position__mm,
            orientation__rad=orientation__rad,
            arrangement="linear",
            differentiation_mode="consecutive",
            trajectory_distance__mm=trajectory_distance__mm,
            trajectory_steps=trajectory_steps,
        )

    @classmethod
    def create_scanning_array(
        cls,
        num_electrodes: int = 16,
        inter_electrode_distance__mm: float = 1.0,
        position__mm: Tuple[float, float, float] = (-5.0, 0.0, 15.0),
        orientation__rad: Tuple[float, float, float] = (-np.pi / 6, 0, -np.pi / 2),
        trajectory_distance__mm: float = 1.0,
        trajectory_steps: int = 19,
    ) -> "IntramuscularElectrodeArray":
        """
        Create a scanning electrode array (equivalent to MATLAB s07 example).

        Parameters
        ----------
        num_electrodes : int, default=16
            Number of electrode contacts
        inter_electrode_distance__mm : float, default=1.0
            Distance between contacts in mm
        position__mm : Tuple[float, float, float], default=(-5.0, 0.0, 15.0)
            Initial position
        orientation__rad : Tuple[float, float, float], default=(-π/6, 0, -π/2)
            Electrode orientation
        trajectory_distance__mm : float, default=1.0
            Scanning distance
        trajectory_steps : int, default=19
            Number of scanning steps

        Returns
        -------
        IntramuscularElectrodeArray
            Configured scanning array
        """
        return cls(
            num_electrodes=num_electrodes,
            inter_electrode_distance__mm=inter_electrode_distance__mm,
            position__mm=position__mm,
            orientation__rad=orientation__rad,
            arrangement="linear",
            differentiation_mode="consecutive",
            trajectory_distance__mm=trajectory_distance__mm,
            trajectory_steps=trajectory_steps,
        )

    @classmethod
    def create_end_to_end_array(
        cls,
        num_electrodes: int = 11,
        inter_electrode_distance__mm: float = 1.0,
        position__mm: Tuple[float, float, float] = (-5.0, 0.0, 20.0),
        orientation__rad: Tuple[float, float, float] = (-np.pi / 2, 0, -np.pi / 2),
        static: bool = True,
    ) -> "IntramuscularElectrodeArray":
        """
        Create an end-to-end electrode array (equivalent to MATLAB s07 example).

        Parameters
        ----------
        num_electrodes : int, default=11
            Number of electrode contacts
        inter_electrode_distance__mm : float, default=1.0
            Distance between contacts in mm
        position__mm : Tuple[float, float, float], default=(-5.0, 0.0, 20.0)
            Array position
        orientation__rad : Tuple[float, float, float], default=(-π/2, 0, -π/2)
            Array orientation
        static : bool, default=True
            Whether electrode is static (no trajectory)

        Returns
        -------
        IntramuscularElectrodeArray
            Configured end-to-end array
        """
        trajectory_distance = 0.0 if static else 1.0
        trajectory_steps = 1 if static else 2

        return cls(
            num_electrodes=num_electrodes,
            inter_electrode_distance__mm=inter_electrode_distance__mm,
            position__mm=position__mm,
            orientation__rad=orientation__rad,
            arrangement="linear",
            differentiation_mode="consecutive",
            trajectory_distance__mm=trajectory_distance,
            trajectory_steps=trajectory_steps,
        )

    def get_electrode_normals(self) -> Optional[np.ndarray]:
        """
        Get electrode normal vectors (for point electrodes, returns None).

        Returns
        -------
        Optional[np.ndarray]
            Normal vectors or None for point electrodes
        """
        # Point electrodes don't have specific orientations
        return None

    def get_visible_area(self, fiber_positions: np.ndarray) -> np.ndarray:
        """
        Calculate visible area/detectability for each fiber position.

        This is a simplified version of the MATLAB get_visible_area function.

        Parameters
        ----------
        fiber_positions : np.ndarray
            Fiber positions (N × 3) in mm

        Returns
        -------
        np.ndarray
            Visibility weights for each fiber
        """
        # Get current electrode positions
        electrode_positions, _ = self.get_electrode_positions_for_simulation()

        # Calculate distances from each fiber to closest electrode
        min_distances = np.inf * np.ones(len(fiber_positions))

        for electrode_pos in electrode_positions:
            distances = np.sqrt(np.sum((fiber_positions - electrode_pos) ** 2, axis=1))
            min_distances = np.minimum(min_distances, distances)

        # Convert distance to visibility (closer = more visible)
        # Use exponential decay with characteristic length scale
        visibility_length_scale = 2.0  # mm
        visibility = np.exp(-min_distances / visibility_length_scale)

        return visibility

    def set_position(self, position__mm: Tuple[float, float, float]) -> None:
        """
        Update electrode array position.

        Parameters
        ----------
        position__mm : Tuple[float, float, float]
            New position in mm
        """
        self.position__mm = position__mm
        self._apply_transformation()

        if self.trajectory_steps > 1:
            self._calculate_trajectory_points()
        else:
            self.pts = self.pts_init.copy()

    def set_orientation(self, orientation__rad: Tuple[float, float, float]) -> None:
        """
        Update electrode array orientation.

        Parameters
        ----------
        orientation__rad : Tuple[float, float, float]
            New orientation in radians (roll, pitch, yaw)
        """
        self.orientation__rad = orientation__rad
        self._apply_transformation()

        if self.trajectory_steps > 1:
            self._calculate_trajectory_points()
        else:
            self.pts = self.pts_init.copy()

    def get_electrode_type_info(self) -> Dict[str, Any]:
        """
        Get electrode type information for simulation logging.

        Returns
        -------
        Dict[str, Any]
            Dictionary with electrode configuration details
        """
        return {
            "type": self.type,
            "type_short": self.type_short,
            "num_electrodes": self.num_electrodes,
            "num_channels": self.num_channels,
            "inter_electrode_distance_mm": self.inter_electrode_distance__mm,
            "differentiation_mode": self.differentiation_mode,
            "arrangement": self.arrangement,
            "trajectory_steps": self.trajectory_steps,
            "trajectory_distance_mm": self.trajectory_distance__mm,
            "position_mm": self.position__mm,
            "orientation_rad": self.orientation__rad,
        }
