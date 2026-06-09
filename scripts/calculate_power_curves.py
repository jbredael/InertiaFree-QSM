#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate power curves for an Airborne Wind Energy system.

This script generates optimized power curves and can also export single-point
simulations.

How it works:
1. Choose the system, wind resource, and simulation settings YAML files below.
2. Create a ``PowerCurveConstructor`` from those inputs.
3. Call either ``generate_power_curves`` for a full optimized power curve, or
   ``simulate_single_wind_speed`` for a single wind speed point using either optimization or direct simulation.
4. The constructor writes the power-curve YAML and, when enabled, matching plots
   and time-history sidecar files.

Usage:
    python calculate_power_curves.py
"""

import sys
from pathlib import Path

import numpy as np

# Add the src directory to the path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inertiafree_qsm import PowerCurveConstructor

# Define file paths
PROJECT_ROOT = Path(__file__).parent.parent

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "systems" / "kitepower V3_25.yml"

WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource" / "clustered_case_1.yml"

SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "settings" / "example_settings.yml"

OUTPUT_PATH_OPTIMIZED = PROJECT_ROOT / "results" / "power_curves.yml"
OUTPUT_PATH_DIRECT_SINGLE_POINT = PROJECT_ROOT / "results" / "power_single_point_direct.yml"
OUTPUT_PATH_OPTIMIZED_SINGLE_POINT = PROJECT_ROOT / "results" / "power_single_point_optimized.yml"

if __name__ == "__main__":

    # Create power curve constructor
    constructor = PowerCurveConstructor(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_file=True, verbose=False
    )
    
    constructor.print_summary()
    
    # Generate optimized power curves
    result = constructor.generate_power_curves(
        profile_ids=None, # None = all profiles; or list of profile IDs to include
        output_path=OUTPUT_PATH_OPTIMIZED,
        verbose=True,
        show_plot=True,
        save_plot=True,
        validate_file=True,
    )

    # Simulate a single optimized wind speed
    # result = constructor.simulate_single_wind_speed(
    #     wind_speed=8, profile_id=1, method="optimization",
    #     output_path=OUTPUT_PATH_OPTIMIZED_SINGLE_POINT,
    #     verbose=True, show_plot=True, save_plot=True,
    #     validate_file=True
    # )

    # # Simulate a single wind speed using direct simulation (no optimization)
    # # Values used are then from Cycle settings and Phase settings
    # result = constructor.simulate_single_wind_speed(
    #     wind_speed=15, method="direct",
    #     output_path=OUTPUT_PATH_DIRECT_SINGLE_POINT,
    #     verbose=True, show_plot=True, save_plot=True,
    #     validate_file=True
    # )
  

