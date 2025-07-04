from pathlib import Path

import inspect
import numpy as np
import pandas as pd
import skfmm
from scipy.integrate import dblquad
from scipy.stats import chi2, multivariate_normal
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from myogen import RANDOM_GENERATOR


def _perform_fast_marching(
    speed_map: np.ndarray, seed_points: np.ndarray
) -> np.ndarray:
    """
    Perform fast marching using scikit-fmm to compute distance maps.

    This function implements the Fast Marching Method to solve the Eikonal equation,
    which is used to distribute innervation centers optimally within the muscle cross-section.
    The method ensures that innervation centers are spaced as far apart as possible from
    each other, mimicking the natural distribution of motor unit territories.

    Parameters
    ----------
    speed_map : np.ndarray
        2D speed map (inverse of a density map) defining the propagation speed
        at each grid point. Higher values indicate faster propagation.
        Should be > 1e-10 for valid regions and ≤ 1e-10 for invalid regions.
    seed_points : np.ndarray
        Seed points as 2×N array where each column is a point [x, y].
        Uses 1-based indexing like MATLAB. These are the starting points
        for the distance computation.

    Returns
    -------
    np.ndarray
        Distance map from seed points. Each element represents the minimum
        distance to any of the seed points. Invalid regions (outside the
        circular muscle boundary) are set to -1e10.

    Notes
    -----
    This function is used internally by the muscle distribution algorithm
    to implement a greedy approach for placing innervation centers such that
    each new electrode_grid_center is placed at the location farthest from all previously
    placed centers.
    """
    # Create a mask for valid regions (inside the circular domain)
    valid_mask = speed_map > 1e-10

    # Create a signed distance function for the domain
    # Initialize with large positive values (far from boundary)
    phi = np.ones_like(speed_map) * 1000.0

    # Set seed points to 0 (starting points for Fast Marching)
    for i in range(seed_points.shape[1]):
        x, y = int(seed_points[0, i] - 1), int(seed_points[1, i] - 1)
        # Ensure indices are within bounds
        if 0 <= x < speed_map.shape[0] and 0 <= y < speed_map.shape[1]:
            phi[x, y] = 0.0  # Starting points

    # Set invalid regions (outside circle) to negative values
    phi[~valid_mask] = -1000.0

    # Use scikit-fmm to solve the Eikonal equation
    distance = skfmm.distance(phi, dx=1.0)

    # Set invalid regions to very small values so they won't be selected
    distance[~valid_mask] = -1e10

    return distance


class Muscle:
    """
    A muscle model for simulating motor unit organization and muscle fiber distribution.

    The muscle model consists of:
        - Motor unit territories with biologically realistic size distributions
        - Spatially distributed innervation centers using optimal packing algorithms
        - Muscle fiber assignment based on proximity and self-avoidance principles

    Parameters
    ----------
    recruitment_thresholds : np.ndarray
        Array of recruitment thresholds for each motor unit. The values
        determine the relative sizes of motor unit territories, with larger
        values corresponding to larger territories. Typically ranges from
        -1 to 1, with the largest motor units having thresholds near 1.
    radius__mm : float, default 4.9
        Radius of the muscle cross-section in millimeters. Default value
        corresponds to the First Dorsal Interosseous (FDI) muscle based
        on anatomical measurements (Jacobson et al., 1992).
    fiber_density__fibers_per_mm2 : float, default 400
        Density of muscle fibers per square millimeter. This parameter
        controls the total number of muscle fibers in the muscle and
        affects the granularity of the simulation. Values typically
        range from 300-600 fibers/mm² for human muscles.
    max_innervation_area_to_total_muscle_area__ratio : float, default 0.25
        Ratio defining the maximum territory size relative to total muscle area.
        A value of 0.25 means the largest motor unit can innervate up to 25%
        of the total muscle cross-sectional area. Must be in range (0, 1].
    grid_resolution : int, default 256
        Resolution of the computational grid used for innervation electrode_grid_center
        distribution. Higher values provide more accurate spatial distribution
        but increase computational cost. Recommended range: 128-512.
    autorun : bool, default False
        If True, automatically executes the complete muscle simulation pipeline:
        innervation electrode_grid_center distribution, muscle fiber generation, and fiber-to-
        motor unit assignment. If False, these steps must be called manually.

    Raises
    ------
    ValueError
        If max_innervation_area_to_total_muscle_area__ratio is not in (0, 1].

    Notes
    -----
    The muscle model uses a circular cross-section approximation, which is
    appropriate for many skeletal muscles. The recruitment thresholds are
    used as a proxy for motor unit sizes, following the size principle
    where larger motor units have higher recruitment thresholds.

    Examples
    --------
    >>> # Create a muscle with 120 motor units
    >>> recruitment_thresholds = generate_mu_recruitment_thresholds(N=120)
    >>> muscle = Muscle(
    ...     recruitment_thresholds=recruitment_thresholds,
    ...     radius__mm=4.9,
    ...     fiber_density__fibers_per_mm2=400,
    ...     autorun=True
    ... )
    >>>
    >>> # Access muscle fiber positions for motor unit 0
    >>> fiber_positions = muscle.resulting_fiber_assignment(0)
    >>> print(f"MU 0 has {len(fiber_positions)} muscle fibers")
    """

    def __init__(
        self,
        recruitment_thresholds: np.ndarray,
        radius__mm: float = 4.9,  # Radius of FDI muscle (Jacobson, 1992)
        fiber_density__fibers_per_mm2: float = 400,
        max_innervation_area_to_total_muscle_area__ratio: float = 1 / 4,
        grid_resolution: int = 256,
        autorun: bool = False,
    ):
        self.recruitment_thresholds = recruitment_thresholds
        self._number_of_neurons = len(recruitment_thresholds)
        self.radius__mm = radius__mm
        self.fiber_density__fibers_per_mm2 = fiber_density__fibers_per_mm2
        self.grid_resolution = grid_resolution

        # Validate the ratio
        if not (0 < max_innervation_area_to_total_muscle_area__ratio <= 1):
            raise ValueError(
                '"max_innervation_area_to_total_muscle_area__ratio" must be in (0, 1].'
            )

        self.max_innervation_area_to_total_muscle_area__ratio = (
            max_innervation_area_to_total_muscle_area__ratio
        )

        self.muscle_area__mm2 = np.pi * np.power(self.radius__mm, 2)

        # Calculate innervation areas (same as MATLAB calc_innervation_areas)
        # Use recruitment_thresholds as motor neuron sizes (equivalent to obj.mn_pool.sz)
        muscle_area2max_ia = 1 / self.max_innervation_area_to_total_muscle_area__ratio
        self.desired_innervation_areas__mm2 = (
            self.recruitment_thresholds
            / np.max(self.recruitment_thresholds)
            * self.muscle_area__mm2
            / muscle_area2max_ia
        )

        # Calculate innervation numbers (same as MATLAB calc_innervation_numbers)
        self.desired_number_of_innervated_fibers = np.round(
            self.desired_innervation_areas__mm2
            / np.sum(self.desired_innervation_areas__mm2)
            * self.muscle_area__mm2
            * self.fiber_density__fibers_per_mm2
        ).astype(int)

        self.innervation_center_positions: np.ndarray | None = None

        if autorun:
            self.distribute_innervation_centers()
            self.generate_muscle_fiber_centers()
            self.assign_mfs2mns()

    def distribute_innervation_centers(self) -> None:
        """
        Distribute innervation electrode_grid_center positions using the fast marching method.

        This method implements an optimal packing algorithm to distribute motor unit
        innervation centers within the circular muscle cross-section. The algorithm
        uses the Fast Marching Method to ensure that each new innervation electrode_grid_center is
        placed at the location that maximizes the minimum distance to all previously
        placed centers.

        Returns
        -------
        None
            Results are stored in self.innervation_center_positions as an array
            of shape (n_motor_units, 2) containing [x, y] coordinates in mm.

        Notes
        -----
        This method must be called before generate_muscle_fiber_centers() and
        assign_mfs2mns(). The resulting distribution approximates the optimal
        packing problem for circles, leading to realistic motor unit territory
        arrangements.
        """
        density_map = np.ones((self.grid_resolution, self.grid_resolution))
        X, Y = np.meshgrid(
            np.arange(self.grid_resolution),
            np.arange(self.grid_resolution),
        )
        density_map[
            np.sqrt(
                (X - self.grid_resolution / 2) ** 2
                + (Y - self.grid_resolution / 2) ** 2
            )
            > self.grid_resolution / 2 - 1
        ] = 1e-10

        vertices = np.zeros((2, self._number_of_neurons + 1))
        vertices[:, 0] = [1, 1]

        # MATLAB: for i = 2:(obj.N+1)
        for i in range(1, self._number_of_neurons + 1):
            # Use scikit-fmm for fast marching
            # Create speed map, avoiding division by zero
            ind = np.argmax(_perform_fast_marching(density_map.copy(), vertices[:, :i]))
            x, y = np.unravel_index(ind, (self.grid_resolution, self.grid_resolution))
            vertices[:, i] = [x, y]

        # MATLAB: obj.innervation_center_positions = vertices(:,end:-1:2)';
        # This takes columns from end down to 2 (1-indexed), then transposes
        # In Python: vertices[:, -1:0:-1] gives us columns from end down to 1 (0-indexed)
        self.innervation_center_positions = vertices[:, -1:0:-1].T

        # Only proceed if we have valid innervation_center_positions
        if (
            self.innervation_center_positions.shape[0] > 0
            and self.innervation_center_positions.shape[1] == 2
        ):
            center_offset = self.innervation_center_positions - self.grid_resolution / 2
            max_dist = np.max(
                np.sqrt(center_offset[:, 0] ** 2 + center_offset[:, 1] ** 2)
            )
            if max_dist > 0:  # Avoid division by zero
                self.innervation_center_positions = (
                    center_offset / max_dist * self.radius__mm
                )
            else:
                self.innervation_center_positions = (
                    center_offset  # Keep original if max_dist is 0
                )

    def generate_muscle_fiber_centers(self) -> None:
        """
        Generate muscle fiber electrode_grid_center positions using a pre-computed Voronoi distribution.

        This method creates the spatial distribution of muscle fiber centers
        within the circular muscle cross-section. The distribution is based on a
        Voronoi tessellation pattern that mimics the natural packing of muscle fibers
        observed in histological studies.

        Returns
        -------
        None
            Results are stored in the following attributes:
                - self.mf_centers: Array of shape (n_fibers, 2) with fiber positions [x, y] in mm
                - self.number_of_muscle_fibers: Total number of muscle fibers
                - self.muscle_border: Array of border points for visualization

        Notes
        -----
        This method should be called after distribute_innervation_centers() and
        before assign_mfs2mns(). The Voronoi-based distribution provides more
        realistic fiber spacing compared to regular grids or purely random distributions.

        The reference dataset ('voronoi_pi1e5.csv') contains 100,000 pre-computed
        Voronoi cell centers optimized for circular domains, ensuring efficient
        and consistent fiber distributions across simulations.
        """

        # Expected number of muscle fibers in the muscle
        self.number_of_muscle_fibers = int(
            np.rint((self.radius__mm**2) * np.pi * self.fiber_density__fibers_per_mm2)
        )

        self.mf_centers = pd.read_csv(
            Path(inspect.getfile(self.__class__)).parent / "voronoi_pi1e5.csv",
            header=None,
        ).values

        # Adjust the loaded innervation_center_positions to the expected number of fibers and muscle radius
        self.mf_centers = (self.mf_centers - 5) / 4  # 4 may be unnecessary here
        dists = np.sqrt(self.mf_centers[:, 0] ** 2 + self.mf_centers[:, 1] ** 2)
        sorted_indices = np.argsort(dists)

        if len(sorted_indices) >= self.number_of_muscle_fibers + 1:
            self.mf_centers = (
                self.mf_centers[sorted_indices[: self.number_of_muscle_fibers], :]
                / dists[sorted_indices[self.number_of_muscle_fibers]]
                * self.radius__mm
            )
        else:
            self.mf_centers = (
                self.mf_centers[sorted_indices, :]
                / dists[sorted_indices[-1]]
                * self.radius__mm
            )
            self.number_of_muscle_fibers = len(self.mf_centers)

        # Create muscle border for plotting
        phi_circle = np.linspace(0, 2 * np.pi, 1000)
        phi_circle = phi_circle[:-1]
        self.muscle_border = np.column_stack(
            [self.radius__mm * np.cos(phi_circle), self.radius__mm * np.sin(phi_circle)]
        )

    def assign_mfs2mns(self, n_neighbours: int = 3, conf: float = 0.999):
        """
        Assign muscle fibers to motor neurons using biologically realistic principles.

        This method implements an assignment algorithm that balances

        multiple biological constraints:
            1. Proximity: Fibers closer to innervation centers are more likely to be assigned
            2. Territory size: Each motor unit has a target number of fibers based on its size
            3. Self-avoidance: Neighboring fibers avoid belonging to the same motor unit
            4. Gaussian territories: Fiber territories follow roughly Gaussian distributions

        The assignment uses a probabilistic approach where each fiber is assigned
        based on the posterior probability computed from prior probabilities (target
        fiber numbers) and likelihoods (spatial clustering with Gaussian territories).

        Parameters
        ----------
        n_neighbours : int, default 3
            Number of neighboring fibers to consider for self-avoiding phenomena.
            Higher values increase intermingling between motor units but may slow
            computation. Typical range: 2-5.
        conf : float, default 0.999
            Confidence interval that defines the relationship between innervation
            area and Gaussian distribution variance. Higher values create tighter,
            more compact territories. Should be between 0.9 and 0.999.

        Returns
        -------
        None
            Results are stored in self.assignment as an array of length n_fibers
            where each element indicates the motor unit index (0 to n_motor_units-1)
            assigned to that fiber.

        Raises
        ------
        ValueError
            If innervation_center_positions is None. Call distribute_innervation_centers()
            first.

        Notes
        -----
        The algorithm compensates for out-of-muscle effects by calculating how much
        of each motor unit's Gaussian distribution falls outside the circular muscle
        boundary and adjusting the in-muscle probabilities accordingly.

        The self-avoidance mechanism promotes realistic intermingling by reducing
        the probability of assigning a fiber to a motor unit if its neighbors are
        already assigned to that unit.

        Examples
        --------
        >>> muscle.assign_mfs2mns(n_neighbours=4, conf=0.995)
        >>> assignments = muscle.assignment
        >>> print(f"Fiber 0 belongs to motor unit {assignments[0]}")
        """
        # Ensure innervation_center_positions is available
        if self.innervation_center_positions is None:
            raise ValueError(
                "innervation_center_positions is None. Call distribute_innervation_centers() first."
            )

        # Out-of-muscle area compensation
        # Calculates how much of the MU's gaussian distribution is outside of the
        # muscle border and inflates the rest of the distribution according to it
        borderfun_pos = lambda x: np.real(np.sqrt(self.radius__mm**2 - x**2))
        borderfun_neg = lambda x: np.real(-np.sqrt(self.radius__mm**2 - x**2))
        out_circle_coeff = np.ones(self._number_of_neurons)

        c = chi2.ppf(conf, 2)
        sigma = lambda ia: np.eye(2) * ia / np.pi / c

        for mu in tqdm(
            range(self._number_of_neurons),
            desc="Calculating out-of-circle coefficients",
        ):
            # Create multivariate normal distribution for this motor unit
            mean = self.innervation_center_positions[mu]
            cov = sigma(self.desired_innervation_areas__mm2[mu])

            def probfun(y, x):
                points = (
                    np.column_stack([x.ravel(), y.ravel()])
                    if hasattr(x, "ravel")
                    else np.array([[x, y]])
                )
                return multivariate_normal.pdf(points, mean=mean, cov=cov).reshape(
                    np.array(x).shape
                )

            # Use dblquad for integration (equivalent to MATLAB's integral2)
            result = dblquad(
                probfun, -self.radius__mm, self.radius__mm, borderfun_neg, borderfun_pos
            )
            in_circle_int = result[0]  # dblquad returns (integral, error) or more
            out_circle_coeff[mu] = 1 / in_circle_int

        # Find nearest neighbors for suppression (equivalent to MATLAB's knnsearch)
        if n_neighbours > 0:
            nbrs = NearestNeighbors(n_neighbors=n_neighbours + 1).fit(self.mf_centers)
            _, neighbours = nbrs.kneighbors(self.mf_centers)
            neighbours = neighbours[
                :, 1:
            ]  # Exclude self (equivalent to neighbours(:,2:end))

        # Assignment procedure
        self.assignment = np.full(self.number_of_muscle_fibers, np.nan)
        randomized_mf = RANDOM_GENERATOR.permutation(self.number_of_muscle_fibers)

        for mf in tqdm(randomized_mf, desc="Assigning muscle fibers to motor neurons"):
            probs = np.zeros(self._number_of_neurons)

            for mu in range(self._number_of_neurons):
                # Suppression assignment if neighbours are from the same MU
                # Promotes intermingling
                if n_neighbours > 0 and np.any(self.assignment[neighbours[mf]] == mu):
                    probs[mu] = 0
                else:
                    # A priori probability of the assignment (same as MATLAB)
                    apriori_prob = (
                        self.desired_number_of_innervated_fibers[mu]
                        / self.number_of_muscle_fibers
                    )

                    # Likelihood coming from clustered nature of mf distribution
                    # Use scipy's multivariate_normal.pdf (equivalent to MATLAB's mvnpdf)
                    mean = self.innervation_center_positions[mu]
                    cov = sigma(self.desired_innervation_areas__mm2[mu])
                    clust_hood = multivariate_normal.pdf(
                        self.mf_centers[mf, :], mean=mean, cov=cov
                    )
                    clust_hood = clust_hood * out_circle_coeff[mu]

                    # Final a posteriori probability
                    probs[mu] = apriori_prob * clust_hood

            # Normalize probabilities
            probs = probs / np.sum(probs)

            # Sample from the probability distribution (equivalent to MATLAB's randsample)
            self.assignment[mf] = RANDOM_GENERATOR.choice(
                self._number_of_neurons, p=probs
            )

        print(
            f"Assignment completed. {self.number_of_muscle_fibers} muscle fibers assigned."
        )

    def resulting_fiber_assignment(self, mu: int) -> np.ndarray:
        """
        Get the muscle fiber positions assigned to a specific motor unit.

        Parameters
        ----------
        mu : int
            Motor unit index (0-based). Must be less than the total number of motor units.

        Returns
        -------
        np.ndarray
            Array of shape (n_assigned_fibers, 2) containing the [x, y] coordinates
            (in mm) of all muscle fibers assigned to the specified motor unit.
            If no fibers are assigned to the motor unit, returns an empty array.

        Raises
        ------
        IndexError
            If mu is outside the valid range [0, n_motor_units-1].
        AttributeError
            If the muscle fiber assignment has not been completed yet.

        Examples
        --------
        >>> fiber_positions = muscle.resulting_fiber_assignment(0)
        >>> print(f"Motor unit 0 has {len(fiber_positions)} fibers")
        >>> print(f"First fiber position: x={fiber_positions[0,0]:.2f}, y={fiber_positions[0,1]:.2f}")

        Notes
        -----
        This method should only be called after assign_mfs2mns() has been executed.
        The returned coordinates are in the muscle's coordinate system with the
        origin at the muscle electrode_grid_center.
        """
        return self.mf_centers[
            np.where(
                self.assignment == np.arange(len(self.recruitment_thresholds))[mu]
            )[0]
        ]

    @property
    def resulting_number_of_innervated_fibers(self) -> np.ndarray:
        """
        Calculate the actual number of muscle fibers assigned to each motor unit.

        This property returns the final fiber counts after the assignment process,
        which may differ slightly from the desired counts due to the stochastic
        assignment algorithm and discrete fiber distribution.

        Returns
        -------
        np.ndarray
            Array of length n_motor_units where each element represents the actual
            number of muscle fibers assigned to the corresponding motor unit.
            The sum of all elements equals the total number of muscle fibers.

        Examples
        --------
        >>> actual_counts = muscle.resulting_number_of_innervated_fibers
        >>> desired_counts = muscle.desired_number_of_innervated_fibers
        >>> print(f"Motor unit 0: desired {desired_counts[0]}, actual {actual_counts[0]}")

        Notes
        -----
        This property can be used to assess how well the assignment algorithm
        achieved the target fiber distribution. Large deviations may indicate
        the need to adjust assignment parameters or increase grid resolution.
        """
        return np.bincount(
            self.assignment.astype(int), minlength=self._number_of_neurons
        )

    @property
    def resulting_innervation_areas__mm2(self) -> np.ndarray:
        """
        Calculate the actual innervation areas for each motor unit based on assigned fibers.

        The innervation area is computed as the area of a circle that encompasses
        all muscle fibers assigned to a motor unit, centered on the motor unit's
        innervation electrode_grid_center. This provides a measure of the spatial extent of each
        motor unit territory.

        Returns
        -------
        np.ndarray
            Array of length n_motor_units containing the innervation area (in mm²)
            for each motor unit. Areas are calculated as π × r², where r is the
            maximum distance from the innervation electrode_grid_center to any assigned fiber.

        Raises
        ------
        AttributeError
            If innervation_center_positions is None or assignment has not been completed.

        Examples
        --------
        >>> actual_areas = muscle.resulting_innervation_areas__mm2
        >>> desired_areas = muscle.desired_innervation_areas__mm2
        >>> for i, (actual, desired) in enumerate(zip(actual_areas, desired_areas)):
        ...     print(f"MU {i}: desired {desired:.2f} mm², actual {actual:.2f} mm²")

        Notes
        -----
        The resulting areas may differ from desired areas due to the discrete nature
        of fiber assignment and the constraint of the circular muscle boundary.
        Motor units near the muscle periphery may have smaller actual areas than
        desired due to boundary effects.
        """
        if self.innervation_center_positions is None:
            raise AttributeError(
                "innervation_center_positions is None. Call distribute_innervation_centers() first."
            )

        return np.array(
            [
                np.pi
                * (
                    np.max(
                        np.linalg.norm(
                            self.mf_centers[self.assignment == mu]
                            - self.innervation_center_positions[mu],
                            axis=-1,
                        )
                    )
                    ** 2
                )
                for mu in range(self._number_of_neurons)
            ]
        )
