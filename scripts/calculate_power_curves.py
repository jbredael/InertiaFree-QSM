#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate power curves for an Airborne Wind Energy system.

This script generates power curves using both direct simulation and optimization-based
methods, then compares and exports the results.

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

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_20.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource6.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings.yml"
OUTPUT_PATH_DIRECT = PROJECT_ROOT / "results" / "power_curves_direct_simulation.yml"
OUTPUT_PATH_OPTIMIZED = PROJECT_ROOT / "results" / "power_curves_optimized.yml"
OUTPUT_PATH_DIRECT_SINGLE_POINT = PROJECT_ROOT / "results" / "power_curve_single_point.yml"
OUTPUT_PATH_OPTIMIZED_SINGLE_POINT = PROJECT_ROOT / "results" / "power_curve_single_point_optimized.yml"

if __name__ == "__main__":

    # Create power curve constructor
    constructor = PowerCurveConstructor(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_file=True, verbose=False
    )
    
    # constructor.print_summary()
    
    # # Generate power curves using direct simulation
    # result = constructor.generate_power_curves_direct(
    #     cluster_ids=None,
    #     output_path=OUTPUT_PATH_DIRECT,
    #     verbose=True,
    #     show_plot=False,
    #     save_plot=True, 
    #     validate_file=True
    # )

    # # Generate power curves using optimized simulation
    # result = constructor.generate_power_curves_optimized(
    #     cluster_ids=[3],
    #     output_path=OUTPUT_PATH_OPTIMIZED,
    #     verbose=True,
    #     show_plot=False,
    #     save_plot=True,
    #     validate_file=True,
    # )

    # result = constructor.simulate_single_wind_speed(
    #     wind_speed=10, method="direct",
    #     output_path=OUTPUT_PATH_DIRECT_SINGLE_POINT,
    #     verbose=True, show_plot=True, save_plot=True,
    #     validate_file=True
    # )

    result = constructor.simulate_single_wind_speed(
        wind_speed=15, method="optimization",
        output_path=OUTPUT_PATH_OPTIMIZED_SINGLE_POINT,
        verbose=True, show_plot=True, save_plot=True,
        validate_file=True
    )

