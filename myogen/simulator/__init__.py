"""
MyoGen Simulator Module

This module provides high-level simulation functions for muscle and EMG modeling.
NMODL files are automatically loaded when needed.
"""

import warnings
from typing import Optional

def _ensure_nmodl_loaded() -> bool:
    """
    Ensure NMODL files are loaded, with intelligent fallback logic.
    
    Returns:
        bool: True if NMODL loading was successful or not needed, False if failed
    """
    try:
        # First, try importing core components to see if NMODL is already available
        from myogen.simulator.core.physiological_distribution import (
            generate_mu_recruitment_thresholds,
        )
        return True
    except Exception as first_error:
        # If that fails, try loading NMODL files manually
        try:
            from myogen.utils import setup_myogen
            setup_myogen()

            return True
        except Exception as second_error:
            warnings.warn(
                f"NMODL loading failed. Some features may not work properly.\n"
                f"First error: {first_error}\n"
                f"Second error: {second_error}\n"
                f"Please ensure NEURON is properly installed.",
                UserWarning
            )
            return False

# Automatically ensure NMODL is loaded when simulator is imported
_nmodl_loaded = _ensure_nmodl_loaded()

# Import all public APIs
if _nmodl_loaded:
    from myogen.simulator.core.physiological_distribution import (
        generate_mu_recruitment_thresholds,
    )
    from myogen.simulator.core.emg import (
        SurfaceEMG,
    )
    from myogen.simulator.core.muscle import Muscle
    from myogen.simulator.core.spike_train import MotorNeuronPool

    __all__ = [
        "generate_mu_recruitment_thresholds",
        "Muscle",
        "MotorNeuronPool",
        "SurfaceEMG",
    ]
else:
    # Provide fallback or limited functionality
    warnings.warn(
        "Some simulator features may not be available due to NMODL loading issues.",
        UserWarning
    )
    __all__ = []
