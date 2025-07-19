"""
MyoGen Simulator Module

This module provides high-level simulation functions for muscle and EMG modeling.
NMODL files are automatically loaded when needed.
"""

import warnings

# Always import all public APIs (they will fail gracefully if NMODL not loaded)
from myogen.simulator.core.physiological_distribution import (
    generate_mu_recruitment_thresholds,
)
from myogen.simulator.core.emg import (
    SurfaceEMG,
    #IntramuscularEMG,
    SurfaceElectrodeArray,
    #IntramuscularElectrodeArray,
)
from myogen.simulator.core.muscle import Muscle
from myogen.simulator.core.spike_train import MotorNeuronPool
from myogen.simulator.core.force import ForceModel

__all__ = [
    "generate_mu_recruitment_thresholds",
    "Muscle",
    "MotorNeuronPool",
    "SurfaceEMG",
    #"IntramuscularEMG",
    "SurfaceElectrodeArray",
    #"IntramuscularElectrodeArray",
    "ForceModel",
]
