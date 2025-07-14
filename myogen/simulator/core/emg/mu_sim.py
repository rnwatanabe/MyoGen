"""
Motor Unit Simulation class for handling individual motor unit simulations
and MUAP (Motor Unit Action Potential) calculations.

Translated from MATLAB MU_Sim class functionality.
"""

from ast import main
from typing import Optional

import numpy as np

from myogen import RANDOM_GENERATOR


class MU_Sim:
    """
    Motor Unit Simulation class for handling individual motor unit simulations
    and MUAP calculations.

    This class represents a single motor unit with its muscle fibers,
    neuromuscular junctions, and action potential calculations.
    """

    def __init__(
        self,
        mf_centers: np.ndarray,
        mf_left_end: np.ndarray,
        mf_right_end: np.ndarray,
        mf_diameters: np.ndarray,
        mf_cv: np.ndarray,
        branch_v: np.ndarray,
        nominal_center: Optional[np.ndarray] = None,
    ):
        """
        Initialize a Motor Unit Simulation object.

        Parameters
        ----------
        mf_centers : np.ndarray
            Muscle fiber electrode_grid_center coordinates (Nx2 array)
        mf_left_end : np.ndarray
            Muscle fiber left end coordinates (Nx2 array)
        mf_right_end : np.ndarray
            Muscle fiber right end coordinates (Nx2 array)
        mf_diameters : np.ndarray
            Muscle fiber diameters
        mf_cv : np.ndarray
            Muscle fiber conduction velocities
        branch_v : np.ndarray
            NMJ branch conduction velocities [mm/s] - [axon_v, branch_v]
        nominal_center : np.ndarray, optional
            Nominal electrode_grid_center of the motor unit
        """
        self.muap = None
        self.sfaps = None

        # Muscle fiber properties
        self.Nmf = len(mf_centers)
        self.mf_centers = mf_centers
        self.mf_left_end = mf_left_end
        self.mf_right_end = mf_right_end
        self.Lmuscle = Lmuscle
        self.mf_diameters = mf_diameters
        self.mf_cv = mf_cv
        self.branch_v = branch_v
        self.nominal_center = nominal_center

        self.n_fibers = len(mf_centers)

        # NMJ (neuromuscular junction) properties
        self.nmj_x = None
        self.nmj_y = None
        self.nmj_z = None
        self.nerve_paths = None
        self.mnap_delays = None

        # SFAP and MUAP storage
        self.sfaps = None  # Single fiber action potentials
        self.muap = None  # Motor unit action potential
        self.muap_original = None  # Original MUAP before differential recording
        self._dt = None  # Sampling time interval (stored from calc_sfaps)
        self._dz = None  # Spatial sampling interval (stored from calc_sfaps)

    def calc_sfaps(
        self,
        dt: float,
        dz: float,
        electrode_pts: np.ndarray,
        electrode_normals: np.ndarray | None = None,
        min_radial_distance__mm: np.floating | None = None,
    ):
        """
        Calculate single fiber action potentials (SFAPs).

        Parameters
        ----------
        dt : float
            Time sampling interval [s]
        dz : float
            Spatial sampling interval [mm]
        electrode_pts : np.ndarray
            Electrode point coordinates (N x 3 array)
        electrode_normals : np.ndarray, optional
            Electrode normal vectors (N x 3 array)
        min_radial_distance__mm : float, optional
            Minimum radial distance from fiber to electrode [millimeters]
        """
        if min_radial_distance__mm is None:
            min_radial_distance__mm = np.mean(self.mf_diameters) * 1000

        n_electrode_pts = electrode_pts.shape[0]

        # Store the sampling interval for use in calc_muap
        self._dt = dt
        self._dz = dz

        t = np.arange(
            0,
            np.maximum(
                2 * (self.nmj_z - self.mf_left_end) / self.mf_cv,
                (self.mf_right_end - self.nmj_z) / self.mf_cv,
            ),
            dt,
        )
        # Check if this motor unit has any fibers
        if self.n_fibers == 0 or len(self.mf_cv) == 0:
            # Create empty SFAP array for motor units with no fibers
            self.sfaps = np.zeros((100, n_electrode_pts, 0))  # Default 100 time samples
            return

        # Check if NMJ positions have been calculated
        if self.nmj_z is None:
            raise ValueError(
                "NMJ positions must be calculated first using sim_nmj_branches_two_layers()"
            )

        # Simplified SFAP calculation
        # In practice, this would involve complex bioelectric field calculations
        # Here we use a simplified model based on distance and conduction velocity

        # Estimate SFAP duration based on fiber properties
        max_cv = np.max(self.mf_cv)
        fiber_length = self.Lmuscle  # Assume fibers span muscle length
        propagation_time = fiber_length / max_cv  # seconds

        # SFAP duration is typically 2-3 times the propagation time
        sfap_duration = 3 * propagation_time
        n_time_samples = int(np.ceil(sfap_duration / dt))

        # Initialize SFAP storage
        self.sfaps = np.zeros((n_time_samples, n_electrode_pts, self.n_fibers))

        # Time vector
        t = np.arange(n_time_samples) * dt

        for fiber_idx in range(self.n_fibers):
            fiber_cv = self.mf_cv[fiber_idx]
            fiber_center = self.mf_centers[fiber_idx, :]
            nmj_z = self.nmj_z[fiber_idx]

            # Simple SFAP model: biphasic waveform
            # The actual implementation would use more sophisticated models
            for pt_idx in range(n_electrode_pts):
                electrode_pos = electrode_pts[pt_idx, :]

                # Distance from fiber to electrode point
                dx = fiber_center[0] - electrode_pos[0]
                dy = fiber_center[1] - electrode_pos[1]
                dz_electrode = electrode_pos[2]

                # Distance varies along fiber length
                distances = np.sqrt(
                    dx**2 + dy**2 + (dz_electrode - (nmj_z + t * fiber_cv)) ** 2
                )

                # Avoid division by zero
                distances = np.maximum(distances, 0.01)  # 0.01 mm minimum distance

                # Simple current dipole model
                # SFAP amplitude is inversely related to distance
                amplitude_factor = 1.0 / distances**2

                # Biphasic waveform (simplified)
                # First derivative of Gaussian
                sigma = sfap_duration / 6  # Standard deviation
                gaussian = np.exp(-((t - sfap_duration / 2) ** 2) / (2 * sigma**2))
                dgaussian_dt = -(t - sfap_duration / 2) / sigma**2 * gaussian

                # Scale by amplitude factor (using mean for simplicity)
                mean_amplitude = np.mean(amplitude_factor)
                self.sfaps[:, pt_idx, fiber_idx] = (
                    dgaussian_dt * mean_amplitude * 1e-3
                )  # Scale to millivolts for visibility

    def sim_nmj_branches_two_layers(
        self,
        n_branches: int,
        endplate_area_center: float,
        branches_z_std: float,
        arborization_z_std: float,
    ):
        """
        Simulate neuromuscular junction branches using a two-layer model.

        Parameters
        ----------
        n_branches : int
            Number of branches
        endplate_area_center : float
            Center of endplate area along z-axis
        branches_z_std : float
            Standard deviation for branch distribution
        arborization_z_std : float
            Standard deviation for arborization distribution
        """
        # Generate branch points along z-axis
        branch_z = RANDOM_GENERATOR.normal(
            endplate_area_center, branches_z_std, n_branches
        )

        # For each muscle fiber, assign to nearest branch and add arborization
        self.nmj_z = np.zeros(self.n_fibers)
        self.nmj_x = self.mf_centers[:, 0].copy()
        self.nmj_y = self.mf_centers[:, 1].copy()

        for i in range(self.n_fibers):
            # Find nearest branch
            distances = np.abs(branch_z - endplate_area_center)
            nearest_branch_idx = np.argmin(distances)

            # Add arborization variability
            self.nmj_z[i] = branch_z[nearest_branch_idx] + RANDOM_GENERATOR.normal(
                0, arborization_z_std
            )

        # Calculate nerve paths (simplified - distance from electrode_grid_center to NMJ)
        self.nerve_paths = np.zeros((self.n_fibers, 1))
        if self.nominal_center is not None:
            center_z = (
                endplate_area_center  # Assume electrode_grid_center is at endplate area
            )
            for i in range(self.n_fibers):
                # Distance from nominal electrode_grid_center to NMJ
                dx = self.nmj_x[i] - self.nominal_center[0]
                dy = self.nmj_y[i] - self.nominal_center[1]
                dz = self.nmj_z[i] - center_z
                self.nerve_paths[i, 0] = np.sqrt(dx**2 + dy**2 + dz**2)
        else:
            # If no nominal electrode_grid_center, use distance from muscle electrode_grid_center
            for i in range(self.n_fibers):
                dz = self.nmj_z[i] - endplate_area_center
                self.nerve_paths[i, 0] = np.abs(dz)

        # Calculate propagation delays (time for signal to reach NMJ)
        # Using axon velocity (first element of branch_v)
        axon_velocity = self.branch_v[0]  # mm/s
        self.mnap_delays = self.nerve_paths[:, 0] / axon_velocity  # seconds

    def calc_muap(self, jitter_std: float = 0.0) -> np.ndarray:
        """
        Calculate Motor Unit Action Potential (MUAP) by summing SFAPs.

        Parameters
        ----------
        jitter_std : float, optional
            Standard deviation of jitter in seconds (default: 0.0 for no jitter)

        Returns
        -------
        np.ndarray
            MUAP waveform
        """
        if self.sfaps is None:
            raise ValueError("SFAPs must be calculated first using calc_sfaps()")

        if self.mnap_delays is None:
            raise ValueError(
                "MNAP delays must be calculated first using sim_nmj_branches_two_layers()"
            )

        n_time_samples, n_electrode_pts, n_fibers = self.sfaps.shape

        # Initialize MUAP
        muap = np.zeros((n_time_samples, n_electrode_pts))

        # If no fibers, return zero MUAP
        if n_fibers == 0:
            self.muap = muap
            return muap

        if jitter_std > 0:
            # Apply jitter to each fiber
            dt = self.dt if self.dt is not None else 1e-4  # Use stored dt or fallback
            jitter_samples = RANDOM_GENERATOR.normal(
                0, jitter_std / dt, n_fibers
            ).astype(int)

            for fiber_idx in range(n_fibers):
                jitter = jitter_samples[fiber_idx]
                delay = int(self.mnap_delays[fiber_idx] / dt) + jitter

                # Add SFAP with delay and jitter
                if 0 <= delay < n_time_samples:
                    end_idx = min(n_time_samples, n_time_samples - delay)
                    muap[delay : delay + end_idx, :] += self.sfaps[
                        :end_idx, :, fiber_idx
                    ]
        else:
            # No jitter - simple summation with propagation delays
            dt = self.dt if self.dt is not None else 1e-4  # Use stored dt or fallback

            for fiber_idx in range(n_fibers):
                delay = int(self.mnap_delays[fiber_idx] / dt)

                # Add SFAP with delay
                if 0 <= delay < n_time_samples:
                    end_idx = min(n_time_samples, n_time_samples - delay)
                    muap[delay : delay + end_idx, :] += self.sfaps[
                        :end_idx, :, fiber_idx
                    ]

        self.muap = muap
        return muap

    def get_elementary_current_response(
        self, t: float, z: float, r: np.ndarray
    ) -> np.ndarray:
        """
        Calculate elementary current response (simplified model).

        Parameters
        ----------
        t : float
            Time
        z : float
            Z coordinate
        r : np.ndarray
            Radial distances

        Returns
        -------
        np.ndarray
            Current response amplitudes
        """
        # Simplified current dipole model
        # In practice, this would use more sophisticated bioelectric field calculations
        r = np.maximum(r, 0.055)  # Minimum distance 55 µm
        return 1.0 / (4 * np.pi * r**2)  # Basic current dipole response


if __name__ == "__main__":
    import time
    from myogen import simulator
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    # Use interactive backend for displaying figures
    # Remove or comment out the Agg backend which is for saving only
    # matplotlib.use("Agg")

    # Simulation parameters
    dt = 1e-4  # Time sampling interval (100 μs)
    fs = 1.0 / dt  # Sampling frequency (10 kHz)
    dz = 0.1  # Spatial sampling interval (0.1 mm)

    # NMJ jitter standard deviation (default value from Hamilton Stashuk)
    nmj_jitter = 35e-6  # seconds

    # NMJ branch conduction velocity [mm/s]
    # 5000mm/s from Pr. Yann Pereon
    branch_v = np.array([5000, 2000])  # [axon_velocity, branch_velocity]

    print(f"NMJ jitter: {nmj_jitter * 1e6:.1f} μs")
    print(f"Branch velocities: {branch_v} mm/s")

    # Generate recruitment thresholds and muscle model
    N = 20  # Reduced number for faster demonstration
    recruitment_thresholds, _ = simulator.generate_mu_recruitment_thresholds(
        N=N, recruitment_range=25
    )

    muscle = simulator.Muscle(
        recruitment_thresholds=recruitment_thresholds,
        radius__mm=4,
        fiber_density__fibers_per_mm2=25,
        autorun=True,
    )

    # Verify muscle assignment is not None
    if muscle.assignment is None:
        raise ValueError("Muscle assignment failed. Check muscle generation.")

    N = len(muscle.recruitment_thresholds)

    print(f"\nInitializing {N} Motor Unit objects...")
    start_time = time.time()

    # Create a simple electrode array for demonstration
    # In practice, this would come from IntramuscularArray or similar
    ima = simulator.IntramuscularArray(
        n_electrodes=2, spacing=0.5, arrangement="linear"
    )
    ima.set_position(
        np.array([0, 0, 2 * muscle.Lmuscle / 3]) - np.array([0.5, 0, 0]),
        [-np.pi / 2, 0, -np.pi / 2],
    )

    # Initialize MU array
    MUs = []

    for i in range(N):
        print(f"Initializing MU {i + 1}/{N}...", end="\r")

        # Ensure muscle fiber properties exist
        if (
            muscle.mf_centers is None
            or muscle.mf_diameters is None
            or muscle.mf_cv is None
        ):
            raise ValueError(
                "Muscle fiber properties not generated. Ensure autorun=True or call _generate_fiber_properties() manually."
            )

        # Get muscle fibers for this motor unit
        fiber_mask = muscle.assignment == i
        mu_mf_centers = muscle.mf_centers[fiber_mask, :]
        mu_mf_diameters = muscle.mf_diameters[fiber_mask]
        mu_mf_cv = muscle.mf_cv[fiber_mask]

        # Handle motor units with no assigned fibers
        if len(mu_mf_centers) == 0:
            print(f"Warning: MU {i + 1} has no assigned muscle fibers")
            # Create dummy arrays to avoid errors
            mu_mf_centers = np.empty((0, 2))
            mu_mf_diameters = np.empty(0)
            mu_mf_cv = np.empty(0)

        # Get nominal center (check if innervation centers exist)
        nominal_center = None
        if muscle.innervation_center_positions is not None:
            nominal_center = muscle.innervation_center_positions[i, :]

        # Create MU_Sim object
        mu_sim = MU_Sim(
            mf_centers=mu_mf_centers,
            mf_index=i,
            Lmuscle=muscle.Lmuscle,
            mf_diameters=mu_mf_diameters,
            mf_cv=mu_mf_cv,
            branch_v=branch_v,
            nominal_center=nominal_center,
        )

        MUs.append(mu_sim)

    print(f"\n✓ {N} Motor Units initialized in {time.time() - start_time:.2f}s")

    ## Generate neuromuscular junction coordinates distribution
    print("\nGenerating neuromuscular junction distributions...")
    start_time = time.time()

    endplate_area_center = muscle.Lmuscle / 2  # Center of endplate area
    # Number of branches based on motor unit size (logarithmic relationship)
    n_branches = 1 + np.round(np.log(muscle.sz / muscle.sz[0]))

    for i in range(N):
        print(f"Processing MU {i + 1}/{N} NMJ distribution...", end="\r")

        # Calculate standard deviations based on motor unit hierarchy
        # Larger motor units have more spread in their NMJ distribution
        size_ratio = np.sum(muscle.sz[: i + 1]) / np.sum(muscle.sz)

        arborization_z_std = 0.5 + size_ratio * 1.5
        branches_z_std = 1.5 + size_ratio * 4

        # Generate NMJ distribution using two-layer model
        MUs[i].sim_nmj_branches_two_layers(
            int(n_branches[i]), endplate_area_center, branches_z_std, arborization_z_std
        )

    print(f"\n✓ NMJ distributions generated in {time.time() - start_time:.2f}s")

    ## Calculate MU SFAPs (Single Fiber Action Potentials)
    print("\nCalculating SFAPs (this may take several minutes)...")
    print("Progress:")

    total_start_time = time.time()
    for i in range(N):
        sfap_start_time = time.time()

        # Calculate SFAPs for this motor unit
        MUs[i].calc_sfaps(dt, dz, electrode_pts)

        sfap_time = time.time() - sfap_start_time
        elapsed_time = time.time() - total_start_time
        estimated_total = elapsed_time * N / (i + 1)
        remaining_time = estimated_total - elapsed_time

        print(
            f"MU {i + 1:3d}/{N}: {sfap_time:5.1f}s | "
            f"Elapsed: {elapsed_time:6.1f}s | "
            f"Remaining: {remaining_time:6.1f}s"
        )

    total_sfap_time = time.time() - total_start_time
    print(f"\n✓ All SFAPs calculated in {total_sfap_time:.1f}s")

    ## Calculate MUAP templates (no jitter)
    print("\nCalculating MUAP templates...")
    start_time = time.time()

    for i in range(N):
        print(f"Calculating MUAP {i + 1}/{N}...", end="\r")
        MUs[i].calc_muap(0)  # No jitter for templates

    print(f"\n✓ MUAP templates calculated in {time.time() - start_time:.2f}s")

    # Find maximum MUAP length for consistent plotting
    max_muap_len = 0
    for i in range(N):
        if MUs[i].muap.shape[0] > max_muap_len:
            max_muap_len = MUs[i].muap.shape[0]

    print(
        f"Maximum MUAP length: {max_muap_len} samples ({max_muap_len / fs * 1000:.1f} ms)"
    )

    ## Create visualizations
    print("\nCreating MUAP visualizations...")

    # Plot MUAPs in several common axes (better for comparison)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        "Motor Unit Action Potentials (MUAPs)\nOrdered from smallest to largest MU",
        fontsize=14,
    )

    plot_duration = 10 / 1000  # 10 ms
    plot_samples = int(plot_duration * fs)

    # Plot first few MUAPs for demonstration
    n_to_plot = min(4, N)  # Plot up to 4 MUAPs

    for i in range(n_to_plot):
        ax = plt.subplot(n_to_plot, 1, i + 1)

        # Get MUAP data (use simple differential: ch2 - ch1)
        muap_data = MUs[i].muap
        if muap_data.shape[1] >= 2:
            # Differential recording
            muap_diff = muap_data[:, 1] - muap_data[:, 0]
        else:
            # Single channel
            muap_diff = muap_data[:, 0]

        # Limit to plot duration
        n_samples = min(plot_samples, len(muap_diff))
        timeline = np.arange(n_samples) / fs * 1000  # ms

        plt.plot(timeline, muap_diff[:n_samples], "b-", linewidth=1)

        plt.xlabel("Time (ms)" if i == n_to_plot - 1 else "")
        plt.ylabel("Amplitude (μV)")
        plt.title(f"MU {i + 1} (Size: {muscle.sz[i]:.2f})")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("Displayed MUAPs visualization")

    ## Optional: Demonstrate jitter effect
    if N > 0:
        print("\nDemonstrating jitter effect...")

        # Select a representative motor unit (around middle size)
        demo_mu_idx = N // 2

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot original MUAP (no jitter) in black
        timeline = np.arange(plot_samples) / fs * 1000
        muap_original = MUs[demo_mu_idx].muap

        if muap_original.shape[1] >= 2:
            muap_original_diff = (muap_original[:, 1] - muap_original[:, 0])[
                :plot_samples
            ]
        else:
            muap_original_diff = muap_original[:, 0][:plot_samples]

        # Plot only first channel for clarity
        ax.plot(timeline, muap_original_diff, "k-", linewidth=2, label="Original MUAP")

        # Plot multiple MUAPs with jitter
        for i in range(10):
            muap_jitter = MUs[demo_mu_idx].calc_muap(nmj_jitter)

            if muap_jitter.shape[1] >= 2:
                muap_jitter_diff = (muap_jitter[:, 1] - muap_jitter[:, 0])[
                    :plot_samples
                ]
            else:
                muap_jitter_diff = muap_jitter[:, 0][:plot_samples]

            ax.plot(
                timeline,
                muap_jitter_diff,
                "g-",
                linewidth=0.5,
                alpha=0.7,
                label="With jitter" if i == 0 else "",
            )

        # Plot original again on top
        ax.plot(timeline, muap_original_diff, "k-", linewidth=2)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        ax.set_title(
            f"Jitter Effect on MUAP (MU {demo_mu_idx + 1})\n"
            f"Jitter std: {nmj_jitter * 1e6:.0f} μs"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("Displayed jitter demonstration")

    # Statistics
    total_fibers = sum(len(mu.mf_centers) for mu in MUs)
    avg_fibers_per_mu = total_fibers / N if N > 0 else 0
    fiber_counts = [len(mu.mf_centers) for mu in MUs]
    non_empty_mus = [count for count in fiber_counts if count > 0]

    print("\n" + "=" * 60)
    print("MUAP initialization completed successfully!")
    print(f"✓ {N} Motor Units with action potentials ready")
    print(f"✓ Electrode configuration: {n_electrodes} electrodes")
    print(f"✓ Total muscle fibers: {muscle.number_of_muscle_fibers}")
    print(f"✓ SFAP calculation time: {total_sfap_time:.1f}s")
    print(f"✓ Maximum MUAP duration: {max_muap_len / fs * 1000:.1f} ms")
    print("✓ All MUAP templates are ready for EMG simulation")

    print(f"\nStatistics:")
    print(f"• Average fibers per MU: {avg_fibers_per_mu:.1f}")
    print(f"• Smallest MU: {min(fiber_counts)} fibers")
    print(f"• Largest MU: {max(fiber_counts)} fibers")
    if non_empty_mus:
        print(f"• Non-empty MUs: {len(non_empty_mus)}/{N}")
        print(f"• Min fibers (non-empty): {min(non_empty_mus)} fibers")
        print(f"• Max fibers (non-empty): {max(non_empty_mus)} fibers")
    print(f"• NMJ jitter: {nmj_jitter * 1e6:.0f} μs")
    print(f"• Branch velocities: [5000, 2000] mm/s")

    # Clean up variables as in MATLAB version
    del branch_v
