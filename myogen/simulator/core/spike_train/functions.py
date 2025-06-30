import numpy as np
import pyNN.neuron as sim
from neuroml import Morphology, Segment, Point3DWithDiam as P
from neuron import hoc
from pyNN.morphology import NeuroMLMorphology
from pyNN.neuron.morphology import uniform
from scipy import signal

from myogen import RANDOM_GENERATOR


def create_cond(
    n: int, gmin: float, gmax: float, compartment: str, CV: float = 0.01
) -> list[object]:
    """
    Creates a list of conductances for a given number of neurons.

    Parameters
    ----------
    n : int
        The number of neurons.
    gmin : float
        The minimum conductance.
    gmax : float
        The maximum conductance.
    compartment : str
        The compartment of the conductance.
    CV : float
        The coefficient of variation of the noise.
    Returns
    -------
    cond : list[sim.Compartment]
        A list of conductances.
    """

    rng = RANDOM_GENERATOR
    noise_g = CV * rng.normal(size=n)

    return [
        uniform(
            compartment,
            gmax * np.exp(i / (n - 1) * np.log(gmin / gmax)) * (1 + noise_g[i]),
        )
        for i in range(n)
    ]


def create_somas(
    recruitment_thresholds: np.ndarray,
    diameter_min: float,
    diameter_max: float,
    y_min: float,
    y_max: float,
    CV: float = 0.01,
) -> list[Segment]:
    """
    Creates a list of somas for a given number of neurons.

    The somas are created with a diameter that increases linearly from 77.5 to 82.5 µm.
    The linear increment of the diameter is (82.5 - 77.5) / n µm.

    The y coordinate of the soma is 18 µm.
    The linear increment of the y coordinate is 18 / n µm.

    Parameters
    ----------
    recruitment_thresholds : np.ndarray
        The recruitment thresholds of the neurons.
    diameter_min : float
        The minimum diameter of the soma.
    diameter_max : float
        The maximum diameter of the soma.
    y_min : float
        The minimum y coordinate of the soma.
    y_max : float
        The maximum y coordinate of the soma.
    CV : float
        The coefficient of variation of the noise.

    Returns
    -------
    somas : list[Segment]
        A list of somas.
    """
    rng = RANDOM_GENERATOR
    noise_y = CV * rng.normal(size=len(recruitment_thresholds))
    noise_diameter = CV * rng.normal(size=len(recruitment_thresholds))

    x_proximal = (
        diameter_min
        + (diameter_max - diameter_min) * recruitment_thresholds
        + noise_diameter
    )
    y_proximal = y_min + (y_max - y_min) * recruitment_thresholds + noise_y
    diameter_proximal = (
        diameter_min
        + (diameter_max - diameter_min) * recruitment_thresholds
        + noise_diameter
    )
    diameter_distal = (
        diameter_min
        + (diameter_max - diameter_min) * recruitment_thresholds
        + noise_diameter
    )

    return [
        Segment(
            proximal=P(
                x=x_proximal[i],
                y=y_proximal[i],
                z=0,
                diameter=diameter_proximal[i],
            ),
            distal=P(
                x=0,
                y=0,
                z=0,
                diameter=diameter_distal[i],
            ),
            name="soma",
            id=0,
        )
        for i in range(len(recruitment_thresholds))
    ]


def create_dends(
    recruitment_thresholds: np.ndarray,
    somas: list[Segment],
    diameter_min: float,
    diameter_max: float,
    y_min: float,
    y_max: float,
    x_min: float,
    x_max: float,
    CV: float = 0.01,
) -> list[Segment]:
    """
    Creates a list of dendrites for a given number of neurons.

    The parent of the dendrite is the soma of the neuron.
    The id of the dendrite is 1.

    Parameters
    ----------
    recruitment_thresholds : np.ndarray
        The recruitment thresholds of the neurons.
    somas : list[Segment]
        The list of somas.
    diameter_min : float
        The minimum diameter of the dendrite.
    diameter_max : float
        The maximum diameter of the dendrite.
    y_min : float
        The minimum y coordinate of the dendrite.
    y_max : float
        The maximum y coordinate of the dendrite.
    x_min : float
        The minimum x coordinate of the dendrite.
    x_max : float
        The maximum x coordinate of the dendrite.
    CV : float
        The coefficient of variation of the noise.
    Returns
    -------
    dends : list[Segment]
        The list of dendrites.
    """
    rng = RANDOM_GENERATOR
    noise_y = CV * rng.normal(size=len(recruitment_thresholds))
    noise_diameter = CV * rng.normal(size=len(recruitment_thresholds))
    noise_x = CV * rng.normal(size=len(recruitment_thresholds))

    x_proximal = x_min + (x_max - x_min) * recruitment_thresholds + noise_x
    y_proximal = y_min + (y_max - y_min) * recruitment_thresholds + noise_y
    diameter_proximal = (
        diameter_min
        + (diameter_max - diameter_min) * recruitment_thresholds
        + noise_diameter
    )
    diameter_distal = (
        diameter_min
        + (diameter_max - diameter_min) * recruitment_thresholds
        + noise_diameter
    )

    return [
        Segment(
            proximal=P(
                x=0,
                y=y_proximal[i],
                z=0,
                diameter=diameter_proximal[i],
            ),
            distal=P(
                x=x_proximal[i],
                y=y_proximal[i],
                z=0,
                diameter=diameter_distal[i],
            ),
            name="dendrite",
            parent=somas[i],
            id=1,
        )
        for i in range(len(recruitment_thresholds))
    ]


def soma_dend(somas: list[Segment], dends: list[Segment]) -> list[NeuroMLMorphology]:
    """
    Combines a list of somas and a list of dendrites into a list of NeuroMLMorphology objects.

    Parameters
    ----------
    somas : list[Segment]
        The list of somas.
    dends : list[Segment]
        The list of dendrites.

    Returns
    -------
    combined : list[NeuroMLMorphology]
        The list of NeuroMLMorphology objects.
    """
    return [
        NeuroMLMorphology(Morphology(segments=(s, d))) for s, d in zip(somas, dends)
    ]


def generate_spike_times(i: int | list[int]) -> sim.Sequence | list[sim.Sequence]:
    """
    Generates spike times for a given neuron.

    The spike times are generated using a gamma point process.
    The input rate is 83 Hz and the time window is 100 ms.

    If a list of indices is provided, the spike times are generated for each neuron in the list.
    If a single index is provided, the spike times are generated for the neuron with the given index.

    Parameters
    ----------
    i : int | list[int]
        The index of the neuron or a list of indices of the neurons.

    Returns
    -------
    spike_times : Sequence | list[Sequence]
        The spike times of the neuron or a list of spike times of the neurons.
    """
    input_rate = 83  # Hz
    Tf = 100  # ms
    number = int(Tf * input_rate / 1000.0)

    # Generate spike times using gamma point process
    if isinstance(i, list):
        # For multiple neurons, create a list of spike time sequences
        return [
            sim.Sequence(
                np.random.exponential(1000.0 / input_rate, size=number).cumsum()
            )
            for _ in i
        ]
    else:
        # For a single neuron, return one sequence
        return sim.Sequence(
            np.random.exponential(1000.0 / input_rate, size=number).cumsum()
        )


def plot_disparos_neuronios(
    spiketrains,
    neuronio,
    delta_t=0.00005,
    filtro_ordem=4,
    freq_corte=0.001,
    tempo_max=1000,
):
    """
    Plots the Dirac impulse for the spike times of a neuron.

    Parameters
    ----------
    spiketrains : list[list[float]]
        The spike times of the neurons.
    neuronio : int
        The index of the neuron to be processed.
    delta_t : float
        The time interval.
    filtro_ordem : int
        The order of the Butterworth filter.
    freq_corte : float
        The cutoff frequency for the Butterworth filter.
    tempo_max : float
        The maximum time for the x-axis (in milliseconds).
    """

    # Array with the spike times of the neuron
    tempos_neuronios = spiketrains

    # Creation of the time vector
    t = np.arange(0, tempo_max, delta_t)
    impulso_dirac = np.zeros_like(t)

    # Adds the Dirac impulse at each spike time of the neuron
    for tempo in tempos_neuronios:
        idx = np.argmin(
            np.abs(t - tempo / 1000)
        )  # finds the index closest to the spike time
        impulso_dirac[idx] = 1 / delta_t

    # Butterworth filter
    b, a = signal.butter(filtro_ordem, freq_corte)

    # Application of the filter
    filtered_impulso = signal.filtfilt(b, a, impulso_dirac)

    # Plot the results
    # plt.plot(t, filtered_impulso, label="Disparo do Neurônio (Filtrado)")
    # plt.title("Tempos de Disparo do Neurônio (Filtrado)")
    # plt.xlabel("Tempo (ms)")
    # plt.ylabel("Amplitude")
    # plt.xlim(0, tempo_max)
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(t, impulso_dirac, label="Disparo do Neurônio (Impulso de Dirac)")
    # plt.title("Tempos de Disparo do Neurônio (Impulso de Dirac)")
    # plt.xlabel("Tempo (ms)")
    # plt.ylabel("Amplitude")
    # plt.xlim(0, tempo_max)
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return filtered_impulso, t


# plot_disparos_neuronios(data.spiketrains, neuronio=1)

# def neuromuscular_system(cells, n, calcium):
#     muscle_units = dict()
#     force_objects = dict()
#     neuromuscular_junctions = dict()
#
#     for i in range(n):
#         muscle_units[i] = h.Section(name=f'mu{i}')
#         if calcium:
#             force_objects[i] = h.muscle_unit_calcium(muscle_units[i](0.5))
#         else:
#             force_objects[i] = h.muscle_unit(muscle_units[i](0.5))
#         neuromuscular_junctions[i] = h.NetCon(cells.all_cells[i]._cell.sections[0](0.5)._ref_v, force_objects[i], sec=cells.all_cells[i]._cell.sections[0])
#
#         force_objects[i].Fmax = 0.03 + (3 - 0.03)*i/n
#         force_objects[i].Tc = 140 + (96 - 140)*i/n

# return muscle_units, force_objects, neuromuscular_junctions


def neuromuscular_system(cells, n, h, Umax=1000):
    """
    Creates a neuromuscular system.

    Parameters
    ----------
    cells : list[Neuron]
        The list of neurons.
    n : int
        The number of neurons.
    h : object
        The object of the simulation.
    Umax : float
        The maximum voltage of the neurons.

    Returns
    -------
    muscle_units : dict
        The dictionary of muscle units.
    force_objects : dict
        The dictionary of force objects.
    neuromuscular_junctions : dict
        The dictionary of neuromuscular junctions.
    """

    muscle_units = dict()
    force_objects = dict()
    neuromuscular_junctions = dict()

    for i in range(n):
        muscle_units[i] = h.Section(name=f"mu{i}")
        force_objects[i] = h.muscle_unit_calcium(muscle_units[i](0.5))
        neuromuscular_junctions[i] = h.NetCon(
            cells.all_cells[i]._cell.sections[0](0.5)._ref_v,
            force_objects[i],
            sec=cells.all_cells[i]._cell.sections[0],
        )

        force_objects[i].Fmax = 0.03 * np.exp(i / 99 * np.log(3 / 0.03))
        force_objects[i].Tc = 140 * np.exp(i / 99 * np.log(96 / 140))
        force_objects[i].Umax = Umax
        neuromuscular_junctions[i].delay = (
            0.86 / (44 * np.exp(i / 99 * np.log(53 / 44))) * 1000
        )

    return muscle_units, force_objects, neuromuscular_junctions


def soma_força(force_objects: dict, h: hoc.HocObject, f) -> object:
    """
    Sums the forces of the muscle units.

    Parameters
    ----------
    force_objects : dict
        The dictionary of force objects.
    h : hoc.HocObject
        The object of the simulation.
    f
        The list of forces.

    Returns
    -------
    force_total : object
        The total force.
    """
    max_len = max(len(f[i]) for i in force_objects.keys())

    # Creates a vector to store the total force over time
    force_total = h.Vector(max_len)
    force_total.fill(0)  # Initializes with zeros

    # Sums the forces of all muscle units
    for i in force_objects.keys():
        individual_force = f[i]
        force_total.add(individual_force)  # Adds each force vector to the total

    return force_total
