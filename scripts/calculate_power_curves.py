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

# Add the src directory to the path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inertiafree_qsm import PowerCurveConstructor2
from inertiafree_qsm.config_loader import (
    get_direct_simulation_wind_speeds,
    get_optimization_wind_speed_settings,
)


# Define file paths
PROJECT_ROOT = Path(__file__).parent.parent

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_20.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource20.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings_config.yml"
OUTPUT_PATH_DIRECT = PROJECT_ROOT / "results" / "power_curves_direct_simulation.yml"
OUTPUT_PATH_OPTIMIZED = PROJECT_ROOT / "results" / "power_curves_optimized.yml"


def generate_direct_simulation_power_curves():
    """Generate power curves using direct simulation method."""
    print("=" * 80)
    print("GENERATING POWER CURVES - DIRECT SIMULATION METHOD")
    print("=" * 80)
    
    # Load wind speed settings from config
    wind_speeds = get_direct_simulation_wind_speeds(SIMULATION_SETTINGS_PATH)
    print(f"Wind speed range: {wind_speeds[0]:.1f} - {wind_speeds[-1]:.1f} m/s")
    print(f"Number of points: {len(wind_speeds)}")
    
    # Create power curve constructor
    constructor = PowerCurveConstructor2(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_inputs=False,
    )
    
    constructor.print_summary()
    
    # Generate power curves using direct simulation
    result = constructor.generate_power_curves_direct(
        wind_speeds=wind_speeds,
        cluster_ids=None,  # Calculate all clusters
        output_path=OUTPUT_PATH_DIRECT,
        verbose=True,
    )
    
    print(f"\nDirect simulation results saved to: {OUTPUT_PATH_DIRECT}\n")
    return result, constructor


def generate_optimized_power_curves(constructor):
    """Generate power curves using optimization-based method."""    
    print("=" * 80)
    print("GENERATING POWER CURVES - OPTIMIZATION-BASED METHOD")
    print("=" * 80)
    
    # Load wind speed settings from config
    wind_settings = get_optimization_wind_speed_settings(SIMULATION_SETTINGS_PATH)
    vw_cut_in = wind_settings['cut_in']
    vw_cut_out = wind_settings['cut_out']
    n_points = wind_settings['n_points']
    fine_n_points = wind_settings['fine_n_points_near_cutout']
    fine_range = wind_settings['fine_range_m_s']
    
    if vw_cut_in is not None:
        print(f"Cut-in wind speed: {vw_cut_in:.1f} m/s (from config)")
    else:
        print("Cut-in wind speed: auto-estimation")
    
    if vw_cut_out is not None:
        print(f"Cut-out wind speed: {vw_cut_out:.1f} m/s (from config)")
    else:
        print("Cut-out wind speed: auto-estimation")
    
    print(f"Number of points: {n_points}")
    if fine_n_points > 0:
        print(f"Fine resolution: {fine_n_points} points over last {fine_range:.1f} m/s")
    
    # Generate optimized power curve for first cluster
    # Note: generate_power_curve now builds full output and saves automatically
    output = constructor.generate_power_curve(
        cluster_id=0,
        vw_cut_in=vw_cut_in,
        vw_cut_out=vw_cut_out,
        n_points=n_points,
        fine_n_points_near_cutout=fine_n_points,
        fine_range_m_s=fine_range,
        output_path=OUTPUT_PATH_OPTIMIZED,
        verbose=True,
    )
    
    return output


if __name__ == "__main__":
    # Generate power curves using both methods
    direct_result, constructor = generate_direct_simulation_power_curves()
    # optimized_result = generate_optimized_power_curves(constructor)

    print("=" * 80)
    print("POWER CURVE GENERATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. Direct simulation: {OUTPUT_PATH_DIRECT}")
    print(f"  2. Optimized curves: {OUTPUT_PATH_OPTIMIZED}")
    print("\nAll done!")

