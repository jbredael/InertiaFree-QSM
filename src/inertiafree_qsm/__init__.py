# -*- coding: utf-8 -*-
"""Inertia-Free Quasi-Steady Model for Airborne Wind Energy Systems.

This package provides a quasi-steady model for simulating pumping kite power systems,
based on the model presented in "Quasi-Steady Model of a Pumping Kite Power System"
by R. Van der Vlugt et al.

The model uses 2D idealized trajectories for computational efficiency and is suitable
for power curve generation and system optimization.
"""

from .qsm import (
    Environment,
    LogProfile,
    NormalisedWindTable1D,
    SystemProperties,
    KiteKinematics,
    SteadyState,
    Phase,
    RetractionPhase,
    TransitionPhase,
    TractionPhase,
    Cycle,
    OperationalLimitViolation,
    SteadyStateError,
    PhaseError,
)

from .config_loader import (
    load_yaml,
    load_system_config,
    load_wind_resource,
    load_simulation_settings,
    create_wind_profile_from_resource,
    get_reference_wind_speeds,
    calculate_power_curves,
)

__version__ = "0.1.0"

__all__ = [
    # Environment classes
    "Environment",
    "LogProfile",
    "NormalisedWindTable1D",
    # System classes
    "SystemProperties",
    # Kinematics classes
    "KiteKinematics",
    "SteadyState",
    # Phase classes
    "Phase",
    "RetractionPhase",
    "TransitionPhase",
    "TractionPhase",
    "Cycle",
    # Exceptions
    "OperationalLimitViolation",
    "SteadyStateError",
    "PhaseError",
    # Config loaders
    "load_yaml",
    "load_system_config",
    "load_wind_resource",
    "load_simulation_settings",
    "create_wind_profile_from_resource",
    "get_reference_wind_speeds",
    # Main function
    "calculate_power_curves",
]
