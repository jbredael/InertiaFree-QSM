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

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_Mark.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings_Mark.yml"
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
        validate_file=True
    )
    
    constructor.print_summary()
    
    # # Generate power curves using direct simulation
    # result = constructor.generate_power_curves_direct(
    #     output_path=OUTPUT_PATH_DIRECT,
    #     verbose=True,
    #     show_plot=False,
    #     save_plot=True, 
    #     validate_file=True
    # )

    # Generate power curves using optimized simulation
    result = constructor.generate_power_curves_optimized(
        output_path=OUTPUT_PATH_OPTIMIZED,
        verbose=True,
        show_plot=False,
        save_plot=True,
        validate_file=True
    )

    result = constructor.simulate_single_wind_speed(
        wind_speed=10.0, method="direct",
        output_path=OUTPUT_PATH_DIRECT_SINGLE_POINT,
        verbose=True, show_plot=False, save_plot=True,
        validate_file=True
    )

    result = constructor.simulate_single_wind_speed(
        wind_speed=10.0, method="optimization",
        output_path=OUTPUT_PATH_OPTIMIZED_SINGLE_POINT,
        verbose=True, show_plot=False, save_plot=True,
        validate_file=True
    )

    print("=" * 80)
    print("POWER CURVE GENERATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. Direct simulation: {OUTPUT_PATH_DIRECT}")
    print(f"  2. Optimized curves: {OUTPUT_PATH_OPTIMIZED}")
    print(f"  3. Single point simulation: {OUTPUT_PATH_DIRECT_SINGLE_POINT}")
    print(f"  4. Single point optimized: {OUTPUT_PATH_OPTIMIZED_SINGLE_POINT}")
    print("\nAll done!")

