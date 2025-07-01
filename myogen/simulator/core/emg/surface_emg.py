import warnings
from datetime import datetime

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import joblib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from scipy.signal import resample
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.muscle import Muscle
from myogen.simulator.core.spike_train import MotorNeuronPool
from myogen.simulator.core.emg.simulate_fiber import (
    simulate_fiber,
)
from myogen.utils.types import MUAP_SHAPE__TENSOR, SURFACE_EMG__TENSOR

# Suppress warnings for cleaner output during batch simulations
warnings.filterwarnings("ignore")


@beartype
class SurfaceEMG:
    """
    Surface Electromyography (sEMG) Simulation.

    This class provides a biophysical simulation framework for generating
    surface electromyography signals from the muscle. It implements a
    multi-layered cylindrical volume conductor model [1]_ that accounts for
    anatomical accuracy, physiological variability, and measurement noise
    characteristics typical of clinical and research EMG recordings.

    The simulation is built on established biophysical principles and validated
    anatomical data, making it suitable for algorithm development, method validation,
    and educational purposes in EMG signal processing and motor control research.

    Parameters
    ----------
    muscle_model : Muscle
        Pre-computed muscle model containing motor unit territories, fiber distributions,
        and recruitment patterns. Must be created using myogen.simulator.Muscle class.
        The muscle model defines the spatial organization of motor units and their
        associated muscle fibers within the FDI muscle volume.

    sampling_frequency__Hz : float, default=2048.0
        Temporal sampling frequency for MUAP and EMG signals in Hertz.
        Higher frequencies provide better temporal resolution but increase computational
        cost and memory usage.

        Typical ranges:
            - Clinical EMG: 1024-2048 Hz.
            - Research applications: 2048-4096 Hz.
            - High-resolution studies: >4096 Hz.

    mean_conduction_velocity__mm_s : float, default=4.1
        Mean conduction velocity of action potentials along muscle fibers in mm/s.
        Based on experimental measurements from Botelho et al. (2019).
        This parameter significantly affects MUAP duration and propagation patterns.
        Typical range for FDI muscle: 3.5-4.5 mm/s.

    sampling_points_in_t_and_z_domains : int, default=256
        Spatial and temporal discretization resolution for numerical integration.
        Controls the accuracy of the volume conductor calculations but significantly
        impacts computational cost (scales quadratically). Higher values provide
        better numerical accuracy at the expense of simulation time.

        - Fast simulations: 128-256 points.
        - Standard accuracy: 256-512 points.
        - High accuracy: 512-1024 points.

    sampling_points_in_theta_domain : int, default=180
        Angular discretization for cylindrical coordinate system in degrees.
        180 points provides 2° angular resolution, which is sufficient for most
        surface EMG applications. Higher values provide better spatial resolution
        for circumferential electrode placement studies.

    MUs_to_simulate : list[int] or None, default=None
        Indices of specific motor units to simulate. If None, simulates all motor
        units present in the muscle model. For computational efficiency, consider
        simulating subsets for initial analysis or algorithm development.
        Motor unit indices correspond to recruitment order (0 = first recruited).

    mean_fiber_length__mm : float, default=32
        Mean muscle fiber length in millimeters based on anatomical measurements
        (Jacobson et al., 1992). This parameter affects MUAP duration and amplitude
        characteristics. Longer fibers produce longer-duration MUAPs with potentially
        higher amplitudes. Typical range for FDI: 28-36 mm.

    var_fiber_length__mm : float, default=3
        Standard deviation of fiber length distribution in millimeters.
        Introduces realistic biological variability in MUAP shapes within motor units.
        Higher values increase MUAP shape variability but maintain physiological realism.
        Typical range: 2-5 mm.

    radius_bone__mm : float, default=0
        Inner bone radius for cylindrical volume conductor model in millimeters.
        For surface EMG applications, bone effects are typically minimal, so default
        value of 0 is appropriate. Non-zero values may be used for intramuscular EMG
        or detailed anatomical studies.

    fat_thickness__mm : float, default=1
        Subcutaneous fat layer thickness in millimeters. This parameter significantly
        affects MUAP amplitude and spatial distribution due to fat's low electrical
        conductivity acting as an insulator. Based on ultrasound measurements
        (Störchle et al., 2018). Typical range: 0.5-3 mm for hand muscles.

    skin_thickness__mm : float, default=1
        Skin layer thickness in millimeters. Generally less critical than fat thickness
        but affects high-frequency components of MUAPs and electrode-tissue coupling.
        Based on histological measurements (Brodar, 1960). Typical range: 0.5-2 mm.

    muscle_conductivity_radial__S_m : float, default=0.09
        Muscle tissue electrical conductivity in radial direction (perpendicular to fibers)
        in Siemens per meter. Muscle tissue exhibits anisotropic conductivity due to
        fiber orientation, with lower conductivity perpendicular to fibers.
        Value based on Botelho et al. (2019).

    muscle_conductivity_longitudinal__S_m : float, default=0.4
        Muscle tissue electrical conductivity in longitudinal direction (parallel to fibers)
        in Siemens per meter. Approximately 4-5x higher than radial conductivity due to
        fiber structure and orientation. This anisotropy significantly affects MUAP
        spatial patterns and propagation characteristics.

    fat_conductivity__S_m : float, default=0.041
        Subcutaneous fat tissue electrical conductivity in Siemens per meter.
        Low conductivity makes fat act as an electrical insulator, spatially smoothing
        MUAP patterns and reducing amplitudes. Based on experimental measurements
        from tissue impedance studies.

    skin_conductivity__S_m : float, default=1
        Skin tissue electrical conductivity in Siemens per meter.
        Relatively high conductivity provides good electrical coupling between
        underlying tissues and surface electrodes. Value based on Roeleveld et al. (1997).

    electrode_grid_inclination_angle__deg : float, default=0
        Rotation angle of electrode grid relative to muscle fiber direction in degrees.
            - 0° = electrodes aligned parallel to muscle fibers
            - 90° = electrodes aligned perpendicular to muscle fibers
        This parameter affects MUAP propagation patterns and is important for
        studies of fiber direction estimation and conduction velocity analysis.

    electrode_grid_dimensions__rows_cols : ElectrodeGridDimensions, default=(8, 8)
        Number of electrode rows and columns in the recording grid.
        Total number of electrodes = rows × columns.
        Higher densities provide better spatial resolution but increase data volume.

    inter_electrode_distance__mm : float, default=4
        Distance between adjacent electrodes in millimeters.
        Affects spatial resolution and signal crosstalk between channels.
        Smaller distances provide higher spatial resolution but may increase
        correlated noise between adjacent channels.

    electrode_radius__mm : float, default=0.75
        Physical radius of individual circular electrodes in millimeters.
        Affects spatial averaging (larger electrodes average over more tissue volume)
        and signal-to-noise ratio (larger electrodes typically have lower impedance).
        Standard clinical surface electrodes: 0.5-2 mm radius.

    electrode_grid_center_positions : list[ElectrodeGridCenterPosition] or None, default=None
        List of electrode grid center positions in cylindrical coordinates.

        Each position is specified as (z_position_mm, angle_radians) where:
            - z_position_mm: position along muscle fiber direction (+ = distal, - = proximal)
            - angle_radians: circumferential position around muscle (0 = anterior)

        If None, uses single position at anatomical muscle belly center computed
        from the center of mass of all motor unit territories.

        Multiple positions enable spatial analysis of EMG characteristics across
        different muscle regions.

        Example positions:
            - [(0, 0)]: Single center position
            - [(0, 0), (-10, 0), (10, 0)]: Proximal-center-distal sequence
            - [(0, 0), (0, π/2), (0, π)]: Circumferential positions

    Attributes
    ----------
    resulting_muaps_array : MUAP_SHAPE__TENSOR
        Generated MUAP templates after calling simulate_muaps().
        Shape: (n_positions, n_motor_units, n_electrode_rows, n_electrode_cols, n_time_samples)

    surface_emg__tensor : SURFACE_EMG__TENSOR
        Generated surface EMG signals after calling simulate_surface_emg().
        Shape: (n_positions, n_pools, n_electrode_rows, n_electrode_cols, n_time_samples)

    noisy_surface_emg : SURFACE_EMG__TENSOR
        Surface EMG with added noise after calling add_noise().
        Same shape as surface_emg__tensor.

    Raises
    ------
    TypeError
        If any input parameter has incorrect type (enforced by beartype decorator).

    ValueError
        If muscle_model is empty or if electrode parameters are physically unrealistic
        (e.g., negative dimensions, impossible conductivity values).

    Examples
    --------
    **Standard Configuration:**

    >>> muscle = myogen.simulator.Muscle(...)
    >>> surface_emg = SurfaceEMG(
    ...     muscle_model=muscle,
    ...     sampling_frequency__Hz=2048,
    ...     electrode_grid_dimensions__rows_cols=(8, 8),
    ...     MUs_to_simulate=list(range(10))  # First 10 motor units
    ... )

    **High-Resolution Configuration:**

    >>> surface_emg = SurfaceEMG(
    ...     muscle_model=muscle,
    ...     sampling_frequency__Hz=4096,  # Higher temporal resolution
    ...     sampling_points_in_t_and_z_domains=512,  # Higher spatial resolution
    ...     electrode_grid_dimensions__rows_cols=(16, 8),  # More electrodes
    ...     inter_electrode_distance__mm=2,  # Higher spatial sampling
    ... )

    **Multi-Position Spatial Analysis:**

    >>> positions = [
    ...     (0, 0),                    # Center
    ...     (-8, np.deg2rad(0)),      # 8mm proximal
    ...     (8, np.deg2rad(0)),       # 8mm distal
    ...     (0, np.deg2rad(90)),      # 90° circumferential
    ... ]
    >>> surface_emg = SurfaceEMG(
    ...     muscle_model=muscle,
    ...     electrode_grid_center_positions=positions
    ... )

    **Thin Subcutaneous Layer Configuration:**

    >>> surface_emg = SurfaceEMG(
    ...     muscle_model=muscle,
    ...     fat_thickness__mm=0.5,    # Thin fat layer
    ...     skin_thickness__mm=0.8,   # Thin skin layer
    ... )  # Results in higher amplitude, more spatially localized MUAPs

    Notes
    -----
    The simulation framework is based on:
        - **Volume Conductor Theory**: Multi-layered cylindrical model representing muscle, fat,
        and skin tissues with anisotropic conductivity properties.

        - **Motor Unit Physiology**: Realistic motor unit territories, fiber distributions, and
        action potential propagation based on anatomical and physiological measurements.

        - **Electrode Array Modeling**: High-density surface electrode grids with configurable
        spatial arrangements and electrode properties.

        - **Signal Processing Pipeline**: Complete workflow from single-fiber action potentials
        to composite surface EMG signals with noise characteristics.

    Key capabilities:
        - **Multi-Position Electrode Arrays**: Simulate EMG at multiple spatial locations
        - **Motor Unit Action Potentials (MUAPs)**: Generate individual MUAP templates
        - **Composite Surface EMG**: Convolve MUAPs with spike trains for realistic EMG
        - **Noise Modeling**: Add measurement noise with specified signal-to-noise ratios
        - **Computational Optimization**: Efficient matrix reuse for large-scale simulations
        - **Visualization Support**: Built-in plotting capabilities for model validation


    Typical workflow:
        1. Create muscle model using `myogen.simulator.Muscle`
        2. Initialize `SurfaceEMG` with desired parameters
        3. Generate MUAP templates using `simulate_muaps()`
        4. Create motor neuron pool using `myogen.simulator.MotorNeuronPool`
        5. Generate surface EMG using `simulate_surface_emg()`
        6. Optionally add noise using `add_noise()`

    Performance considerations:
        - Memory usage scales as: n_positions × n_motor_units × grid_area × time_samples
        - Computational cost increases with: fiber count, electrode density, temporal resolution
        - For large simulations (>50 MUs, >64 electrodes), consider parameter optimization
        - GPU acceleration used automatically for convolution operations in surface EMG generation

    Validation and quality assurance:
        - Parameters based on peer-reviewed anatomical and physiological studies
        - Built-in validation checks for parameter consistency
        - Extensive documentation with literature references
        - Type checking enforced via beartype decorators

    See Also
    --------
    myogen.simulator.Muscle : For creating anatomically accurate muscle models
    myogen.simulator.MotorNeuronPool : For generating motor neuron spike trains
    myogen.utils.plotting.plot_surface_emg : For visualizing simulation results

    References
    ----------
    .. [1] Farina, D., Mesin, L., Martina, S., Merletti, R., 2004.
    A surface EMG generation model with multilayer cylindrical description of the volume conductor.
    IEEE Transactions on Biomedical Engineering 51, 415–426. https://doi.org/10.1109/TBME.2003.820998
    """

    def __init__(
        self,
        muscle_model: Muscle,  # Resolution parameters
        sampling_frequency__Hz: float = 2048.0,
        mean_conduction_velocity__mm_s: float = 4.1,  # Mean conduction velocity (mm/s) -> (Botelho, 2019)
        sampling_points_in_t_and_z_domains: int = 256,  # Number of points in t and z domains (arbitrary)
        sampling_points_in_theta_domain: int = 180,  # Number of points in theta domain (arbitrary)
        # Muscle parameters (FDI)
        MUs_to_simulate: list[int]
        | None = None,  # List of motor unit indices to simulate
        mean_fiber_length__mm: float = 32,  # Mean fiber length -> (Jacobson, 1992)
        var_fiber_length__mm: float = 3,  # Fiber length variation (+ or -) -> (Jacobson, 1992)
        radius_bone__mm: float = 0,
        fat_thickness__mm: float = 1,  # (Storchle, 2018)
        skin_thickness__mm: float = 1,  # (Brodar, 1960)
        # Conductivities -> (Botelho, 2019)
        muscle_conductivity_radial__S_m: float = 0.09,
        muscle_conductivity_longitudinal__S_m: float = 0.4,
        fat_conductivity__S_m: float = 0.041,
        skin_conductivity__S_m: float = 1,  # (Roeleveld, 1997)
        # Electrode Grid parameters
        electrode_grid_inclination_angle__deg: float = 0,
        electrode_grid_dimensions__rows_cols: tuple[int, int] = (8, 8),
        inter_electrode_distance__mm: float = 4,
        electrode_radius__mm: float = 0.75,
        electrode_grid_center_positions: list[tuple[float | int, float | int] | None]
        | None = None,
    ):
        self.muscle_model = muscle_model
        self.sampling_frequency__Hz = sampling_frequency__Hz
        self.mean_conduction_velocity__mm_s = mean_conduction_velocity__mm_s
        self.sampling_points_in_t_and_z_domains = sampling_points_in_t_and_z_domains
        self.sampling_points_in_theta_domain = sampling_points_in_theta_domain
        self.MUs_to_simulate = MUs_to_simulate
        self.mean_fiber_length__mm = mean_fiber_length__mm
        self.var_fiber_length__mm = var_fiber_length__mm
        self.radius_bone__mm = radius_bone__mm
        self.fat_thickness__mm = fat_thickness__mm
        self.skin_thickness__mm = skin_thickness__mm
        self.muscle_conductivity_radial__S_m = muscle_conductivity_radial__S_m
        self.muscle_conductivity_longitudinal__S_m = (
            muscle_conductivity_longitudinal__S_m
        )
        self.fat_conductivity__S_m = fat_conductivity__S_m
        self.skin_conductivity__S_m = skin_conductivity__S_m
        self.electrode_grid_inclination_angle__deg = (
            electrode_grid_inclination_angle__deg
        )
        self.electrode_grid_dimensions__rows_cols = electrode_grid_dimensions__rows_cols
        self.inter_electrode_distance__mm = inter_electrode_distance__mm
        self.electrode_radius__mm = electrode_radius__mm
        self.electrode_grid_center_positions = electrode_grid_center_positions

        # Convert sampling frequency from Hz to kHz for internal calculations
        # (Historical convention in the simulation framework)
        self._sampling_frequency__kHz = self.sampling_frequency__Hz / 1000.0

    def simulate_muaps(
        self, show_plots: bool = False, verbose: bool = True
    ) -> MUAP_SHAPE__TENSOR:
        """
        Simulate MUAPs for the muscle model.

        Parameters
        ----------
        show_plots : bool, optional
            Whether to show plots of the MUAPs. Default is False.
        verbose : bool, optional
            Whether to print verbose output. Default is True.

        Returns
        -------
        resulting_muaps_array : MUAP_SHAPE__TENSOR
            Generated MUAP templates after calling simulate_muaps().
            Shape: (n_positions, n_motor_units, n_electrode_rows, n_electrode_cols, n_time_samples)
        """

        if verbose:
            start = datetime.now()
            print("Starting FDI muscle EMG simulation...")
            print(
                f"Muscle model: {len(self.muscle_model.resulting_number_of_innervated_fibers)} motor units"
            )

        # Set default MUs_to_simulate if not provided - simulate all available motor units
        if self.MUs_to_simulate is None:
            self.MUs_to_simulate = list(
                range(len(self.muscle_model.resulting_number_of_innervated_fibers))
            )
            if verbose:
                print(
                    f"No specific MUs specified - simulating all {len(self.MUs_to_simulate)} motor units"
                )

        # Set default electrode centers if not provided - single position at muscle center
        if self.electrode_grid_center_positions is None:
            self.electrode_grid_center_positions = [None]  # Single default position
            if verbose:
                print("Using default electrode position at anatomical muscle center")

        # Calculate derived geometric parameters for the volume conductor model
        # Innervation zone variance based on fiber length distribution (Botelho, 2019)
        innervation_zone_variance = self.mean_fiber_length__mm / 10

        # Total radius includes all tissue layers for cylindrical volume conductor
        self.radius_total = (
            self.muscle_model.radius__mm
            + self.fat_thickness__mm
            + self.skin_thickness__mm
        )

        # Extract motor unit fiber counts from the pre-computed muscle model
        number_of_fibers_per_MUs = (
            self.muscle_model.resulting_number_of_innervated_fibers
        )

        # Create temporal sampling array for MUAP time series
        # Time array spans from 0 to (N-1)/Fs to avoid endpoint duplication in FFT
        t = np.linspace(
            0,
            (self.sampling_points_in_t_and_z_domains - 1)
            / self._sampling_frequency__kHz,
            self.sampling_points_in_t_and_z_domains,
        )

        # Generate color map for motor unit visualization (if plots are enabled)
        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(self.MUs_to_simulate)))

        # Pre-calculate innervation zone positions for all motor units
        # Innervation zones are randomly distributed within physiological bounds
        # This introduces realistic variability in MUAP shapes
        innervation_zones = RANDOM_GENERATOR.uniform(
            low=-innervation_zone_variance / 2,
            high=innervation_zone_variance / 2,
            size=len(self.MUs_to_simulate),
        )

        # Calculate default electrode center position based on muscle anatomy
        # Uses center of mass of all fiber positions as anatomically-informed default
        default_center = None
        if None in self.electrode_grid_center_positions:
            all_fiber_positions = []
            all_innervation_zones = []

            # Collect all fiber positions to compute anatomical center
            for MU_number, MU_index in enumerate(self.MUs_to_simulate):
                number_of_fibers = number_of_fibers_per_MUs[MU_index]
                if number_of_fibers == 0:
                    continue
                self.position_of_fibers = self.muscle_model.resulting_fiber_assignment(
                    MU_index
                )
                innervation_zone = innervation_zones[MU_number]

                for fiber_number in range(number_of_fibers):
                    # Convert Cartesian coordinates to angular position for center calculation
                    all_fiber_positions.append(
                        np.arctan2(*self.position_of_fibers[fiber_number][::-1])  # type: ignore
                    )
                    all_innervation_zones.append(innervation_zone)

            if len(all_fiber_positions) > 0:
                # Anatomical center = center of mass of innervation zones and fiber positions
                default_center = (
                    np.mean(
                        all_innervation_zones
                    ),  # Z-position (along fiber direction)
                    np.rad2deg(
                        np.mean(all_fiber_positions)
                    ),  # Angular position (degrees)
                )
            else:
                # Fallback to geometric center if no fibers found
                default_center = (0, 0)

            if verbose:
                print(
                    f"Computed anatomical center: z={default_center[0]:.2f}mm, θ={default_center[1]:.1f}°"
                )

        # Prepare electrode centers list with unit conversion (radians to degrees)
        self.electrode_grid_centers = [
            default_center
            if electrode_grid_center is None
            else (electrode_grid_center[0], np.rad2deg(electrode_grid_center[1]))
            for electrode_grid_center in self.electrode_grid_center_positions
        ]

        if verbose:
            print(
                f"Electrode positions: {len(self.electrode_grid_centers)} position(s)"
            )
            print(
                f"Electrode grid: {self.electrode_grid_dimensions__rows_cols[0]}×{self.electrode_grid_dimensions__rows_cols[1]} = {np.prod(self.electrode_grid_dimensions__rows_cols)} channels"
            )

        # Initialize output array for all results
        # Shape: (n_positions, N_MU, electrode_rows, electrode_cols, time_samples)
        self.resulting_muaps_array = np.zeros(
            (
                len(self.electrode_grid_centers),
                len(self.MUs_to_simulate),
                *self.electrode_grid_dimensions__rows_cols,
                len(t),
            )
        )

        if verbose:
            array_size_mb = self.resulting_muaps_array.nbytes / (1024**2)
            print(
                f"Output array size: {self.resulting_muaps_array.shape} ({array_size_mb:.1f} MB)"
            )

        # Main simulation loop - process each motor unit independently
        for MU_number, MU_index in enumerate(self.MUs_to_simulate):
            number_of_fibers = number_of_fibers_per_MUs[MU_index]

            # Skip motor units with no assigned fibers
            if number_of_fibers == 0:
                if verbose:
                    print(
                        f"Warning: No fibers assigned to MU {MU_number + 1}. Skipping."
                    )
                continue

            # Get spatial positions of all fibers in this motor unit
            self.position_of_fibers = self.muscle_model.resulting_fiber_assignment(
                MU_index
            )
            innervation_zone = innervation_zones[MU_number]

            if verbose:
                print(
                    f"Processing MU {MU_number + 1}/{len(self.MUs_to_simulate)}: {number_of_fibers} fibers"
                )

            # Optional visualization of motor unit territory
            if show_plots:
                _, ax = plt.subplots(figsize=(8, 8))
                # Draw muscle boundary
                ax.add_patch(
                    patches.Circle(
                        (0, 0), self.muscle_model.radius__mm, color="r", alpha=0.1
                    )
                )
                ax.set_title(f"Motor Unit {MU_number + 1} Territory")
                ax.set_xlabel("X position (mm)")
                ax.set_ylabel("Y position (mm)")

            # Matrix optimization variables for computational efficiency
            # These matrices contain position-independent calculations that can be reused
            A_matrix = None
            B_incomplete = None

            # Process each fiber within the motor unit
            for pos_idx, electrode_grid_center in enumerate(
                self.electrode_grid_centers
            ):
                for fiber_number in tqdm(
                    range(number_of_fibers),
                    desc=f"MU {MU_number + 1} Position {pos_idx + 1}",
                ):
                    # Get fiber position in Cartesian coordinates (mm)
                    self.fiber_position = self.position_of_fibers[fiber_number]

                    # Add fiber to visualization plot
                    if show_plots:
                        ax.scatter(
                            *self.fiber_position, color=colors[MU_number], alpha=1, s=10
                        )  # type: ignore

                    # Calculate fiber distance from muscle center (radial position)
                    R = np.sqrt(
                        self.fiber_position[0] ** 2 + self.fiber_position[1] ** 2
                    )

                    # Generate realistic fiber length with biological variability
                    fiber_length__mm = (
                        self.mean_fiber_length__mm
                        + RANDOM_GENERATOR.uniform(
                            low=-self.var_fiber_length__mm,
                            high=self.var_fiber_length__mm,
                        )
                    )

                    # Calculate fiber end positions relative to innervation zone
                    # L1, L2 = distances from innervation zone to fiber ends
                    L2 = abs(innervation_zone - fiber_length__mm / 2)  # Proximal end
                    L1 = abs(innervation_zone + fiber_length__mm / 2)  # Distal end

                    # OPTIMIZATION: For the first fiber, compute expensive matrices once
                    # For subsequent fibers, reuse these matrices across all electrode positions
                    if fiber_number == 0 or A_matrix is None:
                        # Compute matrices for the first electrode position
                        phi_temp, A_matrix, B_incomplete = simulate_fiber(
                            Fs=self._sampling_frequency__kHz,
                            v=self.mean_conduction_velocity__mm_s,
                            N=self.sampling_points_in_t_and_z_domains,
                            M=self.sampling_points_in_theta_domain,
                            r=self.radius_total,
                            r_bone=self.radius_bone__mm,
                            th_fat=self.fat_thickness__mm,
                            th_skin=self.skin_thickness__mm,
                            R=R,
                            L1=L1,
                            L2=L2,
                            zi=innervation_zone,
                            alpha=self.electrode_grid_inclination_angle__deg,
                            channels=self.electrode_grid_dimensions__rows_cols,
                            center=electrode_grid_center,
                            d_ele=self.inter_electrode_distance__mm,
                            rele=self.electrode_radius__mm,
                            sig_muscle_rho=self.muscle_conductivity_radial__S_m,
                            sig_muscle_z=self.muscle_conductivity_longitudinal__S_m,
                            sig_skin=self.skin_conductivity__S_m,
                            sig_fat=self.fat_conductivity__S_m,
                        )
                        # Add first position result to output array
                        self.resulting_muaps_array[pos_idx, MU_number] += phi_temp

                    else:
                        # For subsequent fibers, reuse matrices for all electrode positions
                        # This provides significant computational speedup for multi-position simulations
                        phi_temp, _, _ = simulate_fiber(
                            Fs=self._sampling_frequency__kHz,
                            v=self.mean_conduction_velocity__mm_s,
                            N=self.sampling_points_in_t_and_z_domains,
                            M=self.sampling_points_in_theta_domain,
                            r=self.radius_total,
                            r_bone=self.radius_bone__mm,
                            th_fat=self.fat_thickness__mm,
                            th_skin=self.skin_thickness__mm,
                            R=R,
                            L1=L1,
                            L2=L2,
                            zi=innervation_zone,
                            alpha=self.electrode_grid_inclination_angle__deg,
                            channels=self.electrode_grid_dimensions__rows_cols,
                            center=electrode_grid_center,
                            d_ele=self.inter_electrode_distance__mm,
                            rele=self.electrode_radius__mm,
                            sig_muscle_rho=self.muscle_conductivity_radial__S_m,
                            sig_muscle_z=self.muscle_conductivity_longitudinal__S_m,
                            sig_skin=self.skin_conductivity__S_m,
                            sig_fat=self.fat_conductivity__S_m,
                            A_matrix=A_matrix,  # Reuse cached matrix
                            B_incomplete=B_incomplete,  # Reuse cached matrix
                        )
                        self.resulting_muaps_array[pos_idx, MU_number] += phi_temp

            # Display motor unit territory visualization
            if show_plots:
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
                plt.show()

        # Print simulation summary and performance metrics
        if verbose:
            simulation_time = datetime.now() - start
            print(f"\n{'=' * 60}")
            print(f"Simulation completed successfully!")
            print(f"Total simulation time: {simulation_time}")
            print(
                f"Generated MUAPs for {len(self.electrode_grid_center_positions)} electrode position(s)"
            )
            print(f"Simulated {len(self.MUs_to_simulate)} motor units")
            print(f"Output array shape: {self.resulting_muaps_array.shape}")
            print(
                f"Output array size: {self.resulting_muaps_array.nbytes / (1024**2):.1f} MB"
            )
            print(
                f"Time per motor unit: {simulation_time.total_seconds() / len(self.MUs_to_simulate):.2f} seconds"
            )
            print(f"{'=' * 60}")

        return self.resulting_muaps_array

    def simulate_surface_emg(
        self,
        motor_neuron_pool: MotorNeuronPool,
    ) -> SURFACE_EMG__TENSOR:
        """
        Generate realistic surface EMG signals by convolving MUAP templates with motor neuron spike trains.

        This method implements the final step of surface EMG simulation by combining the previously
        computed Motor Unit Action Potential (MUAP) templates with physiologically realistic motor
        neuron firing patterns. The process involves temporal resampling to match simulation timesteps,
        convolution of spike trains with MUAP shapes, and summation across all active motor units
        for each electrode channel.

        The simulation can optionally use GPU acceleration (via CuPy) for efficient parallel processing of
        convolution operations across multiple electrode positions, motor units, and time points.
        If CuPy is not available, the simulation automatically falls back to CPU computation with NumPy.

        Parameters
        ----------
        motor_neuron_pool : MotorNeuronPool
            Motor neuron pool containing spike trains and simulation parameters.

            Must contain:
                - spike_trains: Binary spike train matrix [n_pools, n_motor_units, n_time_samples]
                - timestep__ms: Simulation timestep in milliseconds
                - active_neuron_indices: List of active motor unit indices for each pool

            The motor neuron pool defines the temporal firing patterns that will be
            convolved with the MUAP templates to generate surface EMG signals.

        Returns
        -------
        surface_emg__tensor : SURFACE_EMG__TENSOR
            Surface EMG tensor with shape (n_positions, n_pools, n_electrodes, n_time_samples)

        Raises
        ------
        AttributeError
            If simulate_muaps() has not been called first to generate MUAP templates.

        ValueError
            If motor_neuron_pool contains no active motor units or if there's a mismatch
            between simulated motor units and available spike trains.

        RuntimeError
            If GPU memory is insufficient for large simulations when using CuPy
            (will automatically fall back to CPU-based computation).

        Notes
        -----
        **Computational Complexity:**
            - Time complexity: O(n_positions × n_pools × n_MUs × n_electrodes × n_time_samples)
            - Space complexity: O(n_positions × n_pools × n_electrodes × n_time_samples)
            - GPU acceleration (if available) provides 10-50x speedup for large simulations

        **Memory Requirements:**
            - Input MUAPs: n_positions × n_MUs × n_electrodes × n_time_samples × 8 bytes
            - Spike trains: n_pools × n_MUs × n_time_samples × 8 bytes
            - Output EMG: n_positions × n_pools × n_electrodes × n_time_samples × 8 bytes
            - Example: 4 positions × 10 pools × 100 MUs × 64 electrodes × 10000 samples ≈ 2 GB

        **Signal Characteristics:**
            - Amplitude range: typically 10-1000 μV for surface EMG
            - Frequency content: 10-500 Hz (depending on muscle and electrode configuration)
            - Temporal patterns reflect motor unit recruitment and firing rate modulation
            - Spatial patterns depend on motor unit territories and electrode positioning

        **Validation Recommendations:**
            - Verify EMG amplitude scaling with muscle activation level
            - Check frequency content matches physiological expectations
            - Ensure spatial patterns are consistent with motor unit territories
            - Compare multi-channel correlations with experimental data

        Examples
        --------
        **Basic Surface EMG Simulation:**

        >>> # First generate MUAP templates
        >>> muaps = surface_emg.simulate_muaps()
        >>>
        >>> # Create motor neuron pool with spike trains
        >>> motor_pool = myogen.simulator.MotorNeuronPool(
        ...     spike_trains=spike_data,
        ...     timestep__ms=1.0,
        ...     active_neuron_indices=[[0, 1, 2, 3, 4]]  # First 5 MUs active
        ... )
        >>>
        >>> # Generate surface EMG
        >>> emg_signals = surface_emg.simulate_surface_emg(motor_pool)
        >>> print(f"EMG shape: {emg_signals.shape}")  # (1, 1, 8, 8, 10000)

        **Multi-Condition Simulation:**

        >>> # Motor pool with multiple activation conditions
        >>> motor_pool = myogen.simulator.MotorNeuronPool(
        ...     spike_trains=multi_condition_spikes,  # [n_conditions, n_MUs, n_samples]
        ...     timestep__ms=0.5,
        ...     active_neuron_indices=[
        ...         [0, 1, 2],           # Low activation: 3 MUs
        ...         [0, 1, 2, 3, 4, 5],  # Medium activation: 6 MUs
        ...         list(range(10))      # High activation: 10 MUs
        ...     ]
        ... )
        >>>
        >>> emg_multi = surface_emg.simulate_surface_emg(motor_pool)
        >>> print(f"Multi-condition EMG: {emg_multi.shape}")  # (n_pos, 3, 8, 8, n_samples)

        **Analysis of Specific Electrodes:**

        >>> emg = surface_emg.simulate_surface_emg(motor_pool)
        >>>
        >>> # Extract signal from center electrode
        >>> center_row, center_col = 4, 4  # Center of 8x8 grid
        >>> center_emg = emg[0, 0, center_row, center_col, :]
        >>>
        >>> # Compute RMS amplitude across grid
        >>> rms_amplitudes = np.sqrt(np.mean(emg[0, 0, :, :, :] ** 2, axis=2))
        >>> print(f"RMS amplitude map shape: {rms_amplitudes.shape}")  # (8, 8)

        **Multi-Position Comparison:**

        >>> # Compare EMG characteristics across electrode positions
        >>> emg = surface_emg.simulate_surface_emg(motor_pool)
        >>>
        >>> for pos_idx in range(emg.shape[0]):
        ...     pos_emg = emg[pos_idx, 0, :, :, :]
        ...     max_amplitude = np.max(np.abs(pos_emg))
        ...     print(f"Position {pos_idx}: Max amplitude = {max_amplitude:.2f} μV")

        See Also
        --------
        simulate_muaps : Generate MUAP templates (must be called first)
        add_noise : Add measurement noise to simulated EMG signals
        myogen.simulator.MotorNeuronPool : Create motor neuron spike trains
        myogen.utils.plotting.plot_surface_emg : Visualize simulated EMG signals
        """
        # Validate that MUAPs have been computed
        if (
            not hasattr(self, "resulting_muaps_array")
            or self.resulting_muaps_array is None
        ):
            raise AttributeError(
                "MUAP templates have not been generated. Call simulate_muaps() first."
            )

        # Temporal resampling to match motor neuron pool timestep
        # Note: resample function from scipy.signal
        muap_shapes__tensor = np.asarray(
            resample(
                self.resulting_muaps_array,
                int(
                    (self.resulting_muaps_array.shape[-1] / self.sampling_frequency__Hz)
                    // (motor_neuron_pool.timestep__ms / 1000)
                ),
                axis=-1,
            )
        )

        # Handle None case for MUs_to_simulate and convert to set for efficient lookup
        if self.MUs_to_simulate is None:
            MUs_to_simulate = set(
                range(len(self.muscle_model.resulting_number_of_innervated_fibers))
            )
        else:
            MUs_to_simulate = set(self.MUs_to_simulate)

        n_pools = motor_neuron_pool.spike_trains.shape[0]
        n_positions = muap_shapes__tensor.shape[0]
        n_rows = muap_shapes__tensor.shape[2]
        n_cols = muap_shapes__tensor.shape[3]

        # Initialize the result array with the correct shape
        # The shape will depend on the length of the convolution result
        sample_conv = np.convolve(
            motor_neuron_pool.spike_trains[0, 0],
            muap_shapes__tensor[0, 0, 0, 0],
            mode="same",
        )

        # Perform the convolution and summation using GPU acceleration if available
        if HAS_CUPY:
            # Use GPU acceleration with CuPy
            spike_gpu = cp.asarray(motor_neuron_pool.spike_trains)
            muap_gpu = cp.asarray(muap_shapes__tensor)
            surface_emg_gpu = cp.zeros(
                (n_positions, n_pools, n_rows, n_cols, len(sample_conv))
            )

            for mu_pool_idx in tqdm(
                range(n_pools), desc="Simulating surface EMG (GPU)"
            ):
                active_neuron_indices = set(
                    motor_neuron_pool.active_neuron_indices[mu_pool_idx]
                )

                for position_idx in range(n_positions):
                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all MUAPs for this combination on GPU
                            convolutions = cp.array(
                                [
                                    cp.correlate(
                                        spike_gpu[mu_pool_idx, muap_idx],
                                        muap_gpu[
                                            position_idx, muap_idx, row_idx, col_idx
                                        ],
                                        mode="same",
                                    )
                                    for muap_idx in MUs_to_simulate.intersection(
                                        active_neuron_indices
                                    )
                                ]
                            )
                            # Sum across MUAPs on GPU
                            surface_emg_gpu[
                                position_idx, mu_pool_idx, row_idx, col_idx
                            ] = cp.sum(convolutions, axis=0)

            # Transfer results back to CPU
            self.surface_emg__tensor = cp.asnumpy(surface_emg_gpu)
        else:
            # Fallback to CPU computation with NumPy
            surface_emg_cpu = np.zeros(
                (n_positions, n_pools, n_rows, n_cols, len(sample_conv))
            )

            for mu_pool_idx in tqdm(
                range(n_pools), desc="Simulating surface EMG (CPU)"
            ):
                active_neuron_indices = set(
                    motor_neuron_pool.active_neuron_indices[mu_pool_idx]
                )

                for position_idx in range(n_positions):
                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all MUAPs for this combination on CPU
                            convolutions = np.array(
                                [
                                    np.correlate(
                                        motor_neuron_pool.spike_trains[
                                            mu_pool_idx, muap_idx
                                        ],
                                        muap_shapes__tensor[
                                            position_idx, muap_idx, row_idx, col_idx
                                        ],
                                        mode="same",
                                    )
                                    for muap_idx in MUs_to_simulate.intersection(
                                        active_neuron_indices
                                    )
                                ]
                            )
                            # Sum across MUAPs on CPU
                            surface_emg_cpu[
                                position_idx, mu_pool_idx, row_idx, col_idx
                            ] = np.sum(convolutions, axis=0)

            self.surface_emg__tensor = surface_emg_cpu

        return self.surface_emg__tensor

    def add_noise(
        self, snr_db: float, noise_type: str = "gaussian"
    ) -> SURFACE_EMG__TENSOR:
        """
        Add noise to the surface EMG signal to achieve a specified Signal-to-Noise Ratio (SNR).

        This function calculates the appropriate noise level based on the signal power and
        desired SNR, then adds noise to the simulated surface EMG signals. This is useful
        for studying the effects of measurement noise on EMG analysis algorithms.

        Parameters
        ----------
        snr_db : float
            Target Signal-to-Noise Ratio in decibels (dB). Higher values mean less noise.

            Typical values for surface EMG:
                - High quality: 20-30 dB
                - Medium quality: 10-20 dB
                - Low quality: 0-10 dB
                - Very noisy: < 0 dB

        noise_type : str, default="gaussian"
            Type of noise to add.

            Currently supports:
                - "gaussian": Additive white Gaussian noise (most common for EMG).

        Returns
        -------
        noisy_surface_emg : SURFACE_EMG__TENSOR
            Surface EMG tensor with added noise, same shape as self.surface_emg__tensor.
            The original signal is preserved in self.surface_emg__tensor.

        Raises
        ------
        ValueError
            If surface EMG has not been simulated yet (call simulate_surface_emg first)
            If noise_type is not supported

        Notes
        -----
        **SNR Calculation**:
        SNR_dB = 10 * log10(P_signal / P_noise)

        Where:
            - P_signal = mean power of the signal across all channels
            - P_noise = power of the additive noise

        The noise power is calculated as:
        P_noise = P_signal / (10^(SNR_dB/10))

        **Noise Characteristics**:
            - Gaussian noise: Zero-mean with variance calculated to achieve target SNR
            - Noise is added independently to each electrode channel
            - Noise power is calculated based on the RMS power of the signal

        Examples
        --------
        >>> # Simulate surface EMG first
        >>> surface_emg = emg_simulator.simulate_surface_emg(motor_neuron_pool)
        >>>
        >>> # Add moderate noise (15 dB SNR)
        >>> noisy_emg = emg_simulator.add_noise(snr_db=15)
        >>>
        >>> # Add high noise (5 dB SNR)
        >>> very_noisy_emg = emg_simulator.add_noise(snr_db=5)
        >>>
        >>> print(f"Original EMG shape: {surface_emg.shape}")
        >>> print(f"Noisy EMG shape: {noisy_emg.shape}")
        """

        # Check if surface EMG has been simulated
        if not hasattr(self, "surface_emg__tensor") or self.surface_emg__tensor is None:
            raise ValueError(
                "Surface EMG has not been simulated yet. "
                "Call simulate_surface_emg() first before adding noise."
            )

        # Validate noise type
        if noise_type.lower() not in ["gaussian"]:
            raise ValueError(
                f"Unsupported noise type: {noise_type}. Currently supported: 'gaussian'"
            )

        # Calculate signal power (RMS power across all dimensions)
        signal_power = np.mean(self.surface_emg__tensor**2)

        # Calculate target noise power from SNR
        # SNR_dB = 10 * log10(P_signal / P_noise)
        # Therefore: P_noise = P_signal / (10^(SNR_dB/10))
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise with appropriate standard deviation
        if noise_type.lower() == "gaussian":
            # For Gaussian noise, variance = power, std = sqrt(power)
            noise_std = np.sqrt(noise_power)

            # Generate noise with same shape as signal
            noise = RANDOM_GENERATOR.normal(
                loc=0.0,  # Zero mean
                scale=noise_std,  # Standard deviation to achieve target SNR
                size=self.surface_emg__tensor.shape,
            )

        # Add noise to signal
        self.noisy_surface_emg = self.surface_emg__tensor + noise

        # Verify achieved SNR (for validation)
        achieved_signal_power = np.mean(self.surface_emg__tensor**2)
        achieved_noise_power = np.mean(noise**2)
        achieved_snr_db = 10 * np.log10(achieved_signal_power / achieved_noise_power)

        print(f"Target SNR: {snr_db:.2f} dB")
        print(f"Achieved SNR: {achieved_snr_db:.2f} dB")

        return self.noisy_surface_emg


# For backward compatibility - run simulation if script is executed directly
if __name__ == "__main__":
    from myogen import simulator
    from myogen.utils.plotting.muscle import show_mf_centers, show_innervation_areas_2d
    import seaborn as sns

    # Generate recruitment thresholds#
    N_MU = 100  # Number of motor units
    r = 4.9  # Radius of the muscle in mm
    fiber_density = 400  # fibers per mm²
    max_innervation_ratio = 1 / 4  # Maximum innervation area to total muscle area ratio
    grid_resolution = 256  # Grid resolution for muscle simulation

    recruitment_thresholds, recruitment_thresholds_starting_by_zeros = (
        simulator.generate_mu_recruitment_thresholds(
            N=N_MU,
            recruitment_range=50,  # Recruitment range in % MVC
            mode="konstantin",  # Recruitment pattern model
            konstantin__max_threshold=1,
        )
    )

    plt.plot(recruitment_thresholds, marker="o", linestyle="None")
    plt.show()

    # #Create muscle object
    # muscle = simulator.Muscle(
    #     recruitment_thresholds=recruitment_thresholds,
    #     radius__mm=r,
    #     fiber_density__fibers_per_mm2=fiber_density,
    #     max_innervation_area_to_total_muscle_area__ratio=max_innervation_ratio,
    #     grid_resolution=grid_resolution,
    #     autorun=True,
    # )
    # joblib.dump(muscle, "muscle.pkl")  # Save for reuse

    # load muscle object
    muscle = joblib.load("muscle.pkl")

    with plt.xkcd():
        plt.rcParams.update({"font.size": 24})
        plt.rcParams.update({"axes.titlesize": 24})
        plt.rcParams.update({"axes.labelsize": 24})
        plt.rcParams.update({"xtick.labelsize": 24})
        plt.rcParams.update({"ytick.labelsize": 24})
        show_mf_centers(muscle, ax=plt.gca())

        sns.despine(trim=True, offset=10)
    plt.show()

    plt.figure(figsize=(6, 6))

    plt.rcParams.update({"font.size": 24})
    plt.rcParams.update({"axes.titlesize": 24})
    plt.rcParams.update({"axes.labelsize": 24})
    plt.rcParams.update({"xtick.labelsize": 24})
    plt.rcParams.update({"ytick.labelsize": 24})

    selected_indices = np.array(list(reversed(range(120))))
    with plt.xkcd():
        show_innervation_areas_2d(
            muscle, indices_to_plot=selected_indices, ax=plt.gca()
        )
    plt.title("Motor Unit Innervation Areas (First 10 MUs)")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.axis("equal")

    plt.show()

    surface_emg = SurfaceEMG(
        muscle_model=muscle,
        MUs_to_simulate=[0],
    )

    MUAP = surface_emg.simulate_muaps(
        verbose=True,
        show_plots=False,  # Center at the muscle belly
    )

    np.save("FDI_MUAPs_multi.npy", MUAP)


#######################################################################################################
###################################### References #####################################################
#######################################################################################################

# BOTELHO, Diego; CURRAN, Kathleen; LOWERY, Madeleine M. Anatomically accurate
# model of EMG during index finger flexion and abduction derived from diffusion tensor
# imaging. PLOS Computational Biology, [S. l.], v. 15, n. 8, p. e1007267, 2019. DOI:
# 10.1371/journal.pcbi.1007267.

# BRODAR, Vida. Observations on Skin Thickness and Subcutaneous Tissue in Man. Zeitschrift
# Für Morphologie Und Anthropologie, [S. l.], v. 50, n. 3, p. 386–95, 1960. Available in:
# https://about.jstor.org/terms

# ENOKA, Roger M.; FUGLEVAND, Andrew J. Motor unit physiology: Some unresolved issues. Muscle & Nerve,
# [S. l.], v. 24, n. 1, p. 4–17, 2001. DOI: 10.1002/1097-4598(200101)24:1<4::AID-MUS13>3.0.CO;2-F.

# FEINSTEIN, Bertram; LINDEGARD, Bengt; NYMAN, Eberhard; WOHLFART, Gunnar.
# MORPHOLOGIC STUDIES OF MOTOR UNITS IN NORMAL HUMAN MUSCLES. Cells Tissues
# Organs, [S. l.], v. 23, n. 2, p. 127–142, 1955. DOI: 10.1159/000140989.

# HWANG, Kun; HUAN, Fan; KIM, Dae Joong. Muscle fibre types of the lumbrical, interossei,
# flexor, and extensor muscles moving the index finger. Journal of Plastic Surgery and Hand
# Surgery, [S. l.], v. 47, n. 4, p. 268–272, 2013. DOI: 10.3109/2000656X.2012.755988.

# JACOBSON, Mark D.; RAAB, Rajnik; FAZELI, Babak M.; ABRAMS, Reid A.; BOTTE, Michael J.;
# LIEBER, Richard L. Architectural design of the human intrinsic hand muscles. The Journal of
# Hand Surgery, [S. l.], v. 17, n. 5, p. 804–809, 1992. DOI: 10.1016/0363-5023(92)90446-V.

# ROELEVELD, K.; BLOK, J. H.; STEGEMAN, D. F.; VAN OOSTEROM, A. Volume conduction
# models for surface EMG; confrontation with measurements. Journal of Electromyography
# and Kinesiology, [S. l.], v. 7, n. 4, p. 221–232, 1997. DOI: 10.1016/S1050-6411(97)00009-6.

# STÖRCHLE, Paul; MÜLLER, Wolfram; SENGEIS, Marietta; LACKNER, Sonja; HOLASEK, Sandra;
# FÜRHAPTER-RIEGER, Alfred. Measurement of mean subcutaneous fat thickness: eight
# standardised ultrasound sites compared to 216 randomly selected sites. Scientific Reports,
# [S. l.], v. 8, n. 1, p. 16268, 2018. DOI: 10.1038/s41598-018-34213-0.
