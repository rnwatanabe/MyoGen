from typing import Any, Literal
import numpy as np
import pyNN.neuron as sim
from pyNN.neuron.morphology import uniform, centre
from pyNN.parameters import IonicSpecies
import neo

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.spike_train import functions, classes
from myogen.utils.types import SPIKE_TRAIN__MATRIX, INPUT_CURRENT__MATRIX


class MotorNeuronPool:
    """
    Motor neuron pool with specified parameters

    Parameters
    ----------
    recruitment_thresholds: np.ndarray
        Array of recruitment thresholds for the motor neurons
    diameter_soma_min: float
        Minimum diameter of the soma
    diameter_soma_max: float
    y_min: float
        Minimum y coordinate of the soma
    y_max: float
        Maximum y coordinate of the soma
    diameter_dend_min: float
        Minimum diameter of the dendrite
    diameter_dend_max: float
        Maximum diameter of the dendrite
    x_min: float
        Minimum x coordinate of the dendrite
    x_max: float
        Maximum x coordinate of the dendrite
    vt_min: float
        Minimum voltage threshold of the neuron
    vt_max: float
        Maximum voltage threshold of the neuron
    kf_cond_min: float
        Minimum conductance density of the potassium fast channel
    kf_cond_max: float
        Maximum conductance density of the potassium fast channel
    CV: float
        Coefficient of variation of the noise

    Returns
    -------
    MotorNeuronPool
        Motor neuron pool with specified parameters
    """

    def __init__(
        self,
        recruitment_thresholds: np.ndarray,
        diameter_soma_min: float = 77.5,
        diameter_soma_max: float = 82.5,
        y_min: float = 18.0,
        y_max: float = 36.0,
        diameter_dend_min: float = 41.5,
        diameter_dend_max: float = 62.5,
        x_min: float = -5500,
        x_max: float = -6789,
        vt_min: float = 12.35,
        vt_max: float = 20.9,
        kf_cond_min: float = 4,
        kf_cond_max: float = 0.5,
        CV: float = 0.01,
    ):
        self.recruitment_thresholds = recruitment_thresholds
        self.diameter_soma_min = diameter_soma_min
        self.diameter_soma_max = diameter_soma_max
        self.y_min = y_min
        self.y_max = y_max
        self.diameter_dend_min = diameter_dend_min
        self.diameter_dend_max = diameter_dend_max
        self.x_min = x_min
        self.x_max = x_max
        self.vt_min = vt_min
        self.vt_max = vt_max
        self.kf_cond_min = kf_cond_min
        self.kf_cond_max = kf_cond_max
        self.CV = CV

    def _create_motor_neuron_pool(self) -> classes.cell_class:
        """Create a pool of motor neurons with specified parameters

        Returns
        -------
        cell_type: classes.cell_class
            Cell type of the motor neuron pool
        """
        rng = RANDOM_GENERATOR

        n_neurons = len(self.recruitment_thresholds)

        somas = functions.create_somas(
            recruitment_thresholds=self.recruitment_thresholds,
            diameter_min=self.diameter_soma_min,
            diameter_max=self.diameter_soma_max,
            y_min=self.y_min,
            y_max=self.y_max,
            CV=self.CV,
        )
        dends = functions.create_dends(
            recruitment_thresholds=self.recruitment_thresholds,
            somas=somas,
            diameter_min=self.diameter_dend_min,
            diameter_max=self.diameter_dend_max,
            y_min=self.y_min,
            y_max=self.y_max,
            x_min=self.x_min,
            x_max=self.x_max,
            CV=self.CV,
        )

        # scale recruitment thresholds to v_min and v_max
        vt = (
            -70
            + (self.vt_min + (self.vt_max - self.vt_min) * self.recruitment_thresholds)
            + rng.normal(size=n_neurons) * self.CV
        )

        return classes.cell_class(
            morphology=functions.soma_dend(somas, dends),
            cm=1,  # mF / cm**2
            Ra=0.070,  # ohm.mm
            ionic_species={
                "na": IonicSpecies("na", reversal_potential=50),
                "ks": IonicSpecies("ks", reversal_potential=-80),
                "kf": IonicSpecies("kf", reversal_potential=-80),
            },
            pas_soma={
                "conductance_density": functions.create_cond(
                    n_neurons, 7e-4, 7e-4, "soma", CV=10 * self.CV
                ),
                "e_rev": -70,
            },  #
            pas_dend={
                "conductance_density": functions.create_cond(
                    n_neurons, 7e-4, 7e-4, "dendrite", CV=10 * self.CV
                ),
                "e_rev": -70,
            },  #
            na={"conductance_density": uniform("soma", 30), "vt": list(vt)},
            kf={
                "conductance_density": functions.create_cond(
                    n_neurons, self.kf_cond_min, self.kf_cond_max, "soma", CV=self.CV
                ),
                "vt": list(vt),
            },
            ks={"conductance_density": uniform("soma", 0.1), "vt": list(vt)},
            syn={"locations": centre("dendrite"), "e_syn": 0, "tau_syn": 0.6},
        )

    def generate_spike_trains(
        self,
        input_current__matrix: INPUT_CURRENT__MATRIX,
        timestep__ms: float = 0.05,
        noise_mean__nA: float = 30,  # noqa N803
        noise_stdev__nA: float = 30,  # noqa N803
        what_to_record: list[
            dict[Literal["variables", "to_file", "sampling_interval", "locations"], Any]
        ] = [
            {"variables": ["v"], "locations": ["dendrite", "soma"]},
        ],
    ) -> tuple[SPIKE_TRAIN__MATRIX, list[np.ndarray], list[neo.Segment]]:
        """
        Generate the spike trains for as many neuron pools as input currents there are

        Each motor neuron pools have each "neurons_per_pool" neurons.
        The input currents are injected into each pool, and the spike trains are recorded.

        Parameters
        ----------
        input_current__matrix : INPUT_CURRENT__MATRIX
            Matrix of shape (n_pools, t_points) containing current values
            Each row represents the current for one pool
        timestep__ms : float
            Simulation timestep__ms in ms
        noise_mean__nA : float
            Mean of the noise current in nA
        noise_stdev__nA : float
            Standard deviation of the noise current in nA
        what_to_record: WhatToRecord
            List of dictionaries specifying what to record.

            Each dictionary contains the following keys:
                - variables: list of strings specifying the variables to record
                - to_file: bool specifying whether to save the recorded data to a file
                - sampling_interval: int specifying the sampling interval in ms
                - locations: list of strings specifying the locations to record from

            See pyNN documentation for more details: https://pynn.readthedocs.io/en/stable/recording.html.

            Spike trains are recorded by default.

        Returns
        -------
        spike_trains : SPIKE_TRAIN__MATRIX
            Matrix of shape (n_pools, neurons_per_pool, t_points) containing spike trains
            Each row represents the spike train for one pool
            Each column represents the spike train for one neuron
            Each element represents whether the neuron spiked at that time point
        active_neuron_indices : list[np.ndarray]
            List of arrays of indices of the active neurons in each pool
        data : list[neo.core.segment.Segment]
            List of neo segments containing the recorded data
        """
        self.timestep__ms = timestep__ms
        sim.setup(timestep=self.timestep__ms)

        # Create motor neuron pools
        pools: list[sim.Population] = [
            sim.Population(
                len(self.recruitment_thresholds),
                self._create_motor_neuron_pool(),
                initial_values={"v": -70},
            )
            for _ in range(input_current__matrix.shape[0])
        ]

        # Inject currents into each pool - one current per pool
        self.times = np.arange(input_current__matrix.shape[-1]) * self.timestep__ms
        for input_current, pool in zip(input_current__matrix, pools):
            current_source = sim.StepCurrentSource(
                times=self.times, amplitudes=input_current
            )
            current_source.inject_into(pool, location="soma")

            # Add Gaussian noise current to each neuron in the pool
            for neuron in pool:
                noise_source = sim.NoisyCurrentSource(
                    mean=noise_mean__nA,
                    stdev=noise_stdev__nA,
                    start=0,
                    stop=self.times[-1] + self.timestep__ms,
                    dt=self.timestep__ms,
                )
                noise_source.inject_into([neuron], location="soma")

        # Set up recording
        for pool in pools:
            pool.record("spikes")
            for record in what_to_record:
                pool.record(**record)

        # Run simulation
        sim.run(input_current__matrix.shape[-1] * self.timestep__ms)
        sim.end()

        self.data = [pool.get_data().segments[0] for pool in pools]

        # Convert spike times to binary arrays and save
        self.time_indices = np.arange(input_current__matrix.shape[-1])
        self.spike_trains = np.array(
            [
                [
                    np.isin(
                        self.time_indices, np.array(st / self.timestep__ms).astype(int)
                    )
                    for st in d.spiketrains
                ]
                for d in self.data
            ]
        )

        self.active_neuron_indices = [
            np.argwhere(x)[:, 0] for x in np.sum(self.spike_trains, axis=-1) != 0
        ]

        return self.spike_trains, self.active_neuron_indices, self.data

    def compute_mvc_current_threshold(self) -> float:
        """
        Computes the minimum current threshold for maximum voluntary contraction
        using binary search optimization.

        Returns
        -------
        float
            Minimum current threshold in nA needed to activate all neurons
        """

        def test_current_activates_all_neurons(current_nA: float) -> bool:
            """Test if a given current activates all neurons in the pool.

            Parameters
            ----------
            current_nA : float
                Current amplitude in nA

            Returns
            -------
            bool
                True if all neurons are activated, False otherwise
            """
            # Create short test simulation parameters
            test_duration_ms = 500  # Short 500ms test
            test_timestep_ms = 0.5
            n_timepoints = int(test_duration_ms / test_timestep_ms)

            # Create constant current input for testing
            test_current = np.full((1, n_timepoints), current_nA)

            # Run short simulation
            spike_trains, active_neuron_indices, _ = self.generate_spike_trains(
                input_current__matrix=test_current,
                timestep__ms=test_timestep_ms,
                noise_mean__nA=0,  # Reduce noise for more consistent results
                noise_stdev__nA=0,
                what_to_record=[],  # Minimal recording for speed
            )

            # Check if all neurons are active
            n_total_neurons = len(self.recruitment_thresholds)
            n_active_neurons = len(active_neuron_indices[0])

            return n_active_neurons == n_total_neurons

        # Binary search for minimum current
        low_current = 0.0
        high_current = 1000.0  # Start with reasonable upper bound
        tolerance = 1.0  # 1 nA tolerance

        # First, find an upper bound that works
        while not test_current_activates_all_neurons(high_current):
            high_current *= 2
            if high_current > 10000:  # Safety limit
                raise ValueError(
                    "Could not find current that activates all neurons within reasonable range"
                )

        # Binary search between low and high
        while high_current - low_current > tolerance:
            mid_current = (low_current + high_current) / 2

            if test_current_activates_all_neurons(mid_current):
                high_current = mid_current
            else:
                low_current = mid_current

        return high_current

    @property
    def mvc_current_threshold(self) -> float:
        """
        Property that returns the minimum current threshold for maximum voluntary contraction.

        Returns
        -------
        float
            Minimum current threshold in nA needed to activate all neurons
        """
        return self.compute_mvc_current_threshold()
