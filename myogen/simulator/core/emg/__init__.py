"""
EMG (Electromyography) domain components.

This module contains EMG simulation classes and functions.
"""

from myogen.simulator.core.emg.surface.surface_emg import SurfaceEMG
from myogen.simulator.core.emg.intramuscular.intramuscular_emg import IntramuscularEMG
from myogen.simulator.core.emg.electrodes import (
    SurfaceElectrodeArray,
    IntramuscularElectrodeArray,
)

__all__ = [
    "SurfaceEMG",
    "IntramuscularEMG",
    "SurfaceElectrodeArray",
    #"IntramuscularElectrodeArray",
]
