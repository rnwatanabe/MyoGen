"""
Core components for the simulator package.

This module contains the core classes and functions that are used across
the simulator package, organized to eliminate circular dependencies.
"""

from .muscle import Muscle
from .physiological_distribution import generate_mu_recruitment_thresholds
from .spike_train import MotorNeuronPool
from .emg import SurfaceEMG

__all__ = [
    "Muscle",
    "generate_mu_recruitment_thresholds",
    "MotorNeuronPool",
    "SurfaceEMG",
]
