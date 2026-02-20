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

import matplotlib.pyplot as plt
import numpy as np

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

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "soft_kite_pumping_ground_gen_system.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings_config.yml"
OUTPUT_PATH_DIRECT = PROJECT_ROOT / "results" / "power_curves_direct_simulation.yml"
OUTPUT_PATH_OPTIMIZED = PROJECT_ROOT / "results" / "power_curves_optimized.yml"
COMPARISON_PLOT_PATH = PROJECT_ROOT / "results" / "power_curves_comparison.png"


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
        plot=False,
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
    power_curve = constructor.generate_power_curve(
        cluster_id=0,
        vw_cut_in=vw_cut_in,
        vw_cut_out=vw_cut_out,
        n_points=n_points,
        fine_n_points_near_cutout=fine_n_points,
        fine_range_m_s=fine_range,
        verbose=True,
    )
    
    # Build full output in awesIO format (same structure as direct simulation)
    output = constructor._build_full_power_curve_output(
        cluster_ids=[0],
        method_name='Optimization-Based'
    )
    
    # Save to file
    import yaml
    with open(OUTPUT_PATH_OPTIMIZED, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nOptimized results saved to: {OUTPUT_PATH_OPTIMIZED}")
    
    # Export detailed optimization results
    opt_results_path = OUTPUT_PATH_OPTIMIZED.with_name('optimization_details.yml')
    constructor.export_results(opt_results_path)
    print(f"Optimization details saved to: {opt_results_path}\n")
    
    return output


def plot_comparison(direct_result, optimized_result, constructor):
    """Create comparison plots of direct simulation vs optimized power curves."""
    print("=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    # Extract data
    direct_ws = np.array(direct_result['reference_wind_speeds_m_s'])
    direct_power = np.array(direct_result['power_curves'][0]['cycle_power_w'])
    
    opt_ws = np.array(optimized_result['reference_wind_speeds_m_s'])
    opt_power = np.array(optimized_result['power_curves'][0]['cycle_power_w'])
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Power curves comparison
    ax = axes[0, 0]
    ax.plot(direct_ws, np.array(direct_power) / 1000, 'o-', label='Direct Simulation', markersize=4)
    ax.plot(opt_ws, np.array(opt_power) / 1000, 's-', label='Optimization-Based', markersize=4)
    ax.set_xlabel('Wind Speed at 100m [m/s]')
    ax.set_ylabel('Cycle Average Power [kW]')
    ax.set_title('Power Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Power difference
    ax = axes[0, 1]
    # Interpolate to common wind speeds for comparison
    common_ws = np.linspace(max(direct_ws[0], opt_ws[0]), min(direct_ws[-1], opt_ws[-1]), 50)
    direct_power_interp = np.interp(common_ws, direct_ws, direct_power)
    opt_power_interp = np.interp(common_ws, opt_ws, opt_power)
    power_diff_percent = (opt_power_interp - direct_power_interp) / direct_power_interp * 100
    
    ax.plot(common_ws, power_diff_percent, 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Wind Speed at 100m [m/s]')
    ax.set_ylabel('Power Difference [%]')
    ax.set_title('Optimization Gain (Optimized - Direct) / Direct')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cycle times comparison
    ax = axes[1, 0]
    direct_cycle_time = np.array(direct_result['power_curves'][0]['cycle_time_s'])
    opt_cycle_time = np.array(optimized_result['power_curves'][0]['cycle_time_s'])
    ax.plot(direct_ws, direct_cycle_time, 'o-', label='Direct Simulation', markersize=4)
    ax.plot(opt_ws, opt_cycle_time, 's-', label='Optimization-Based', markersize=4)
    ax.set_xlabel('Wind Speed at 100m [m/s]')
    ax.set_ylabel('Cycle Time [s]')
    ax.set_title('Cycle Duration Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reel-out/in power comparison
    ax = axes[1, 1]
    direct_ro_power = np.array(direct_result['power_curves'][0]['reel_out_power_w'])
    direct_ri_power = np.array(direct_result['power_curves'][0]['reel_in_power_w'])
    opt_ro_power = np.array(optimized_result['power_curves'][0]['reel_out_power_w'])
    opt_ri_power = np.array(optimized_result['power_curves'][0]['reel_in_power_w'])
    
    ax.plot(direct_ws, np.array(direct_ro_power) / 1000, 'b-', label='Direct: Reel-out', linewidth=1.5)
    ax.plot(direct_ws, np.array(direct_ri_power) / 1000, 'b--', label='Direct: Reel-in', linewidth=1.5)
    ax.plot(opt_ws, np.array(opt_ro_power) / 1000, 'r-', label='Opt: Reel-out', linewidth=1.5)
    ax.plot(opt_ws, np.array(opt_ri_power) / 1000, 'r--', label='Opt: Reel-in', linewidth=1.5)
    ax.set_xlabel('Wind Speed at 100m [m/s]')
    ax.set_ylabel('Phase Power [kW]')
    ax.set_title('Phase Power Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=200, bbox_inches='tight')
    print(f"Comparison plot saved to: {COMPARISON_PLOT_PATH}")
    
    # Create detailed optimization diagnostics plots
    print("\nGenerating detailed optimization diagnostics...")
    
    # Plot optimal trajectories
    fig_traj = plt.figure(figsize=(8, 6))
    constructor.plot_optimal_trajectories()
    traj_path = OUTPUT_PATH_OPTIMIZED.with_name('optimization_trajectories.png')
    plt.savefig(traj_path, dpi=200, bbox_inches='tight')
    print(f"Optimization trajectories saved to: {traj_path}")
    
    # Plot optimization results
    constructor.plot_optimization_results(
        opt_variable_labels=[
            "Reel-out\nforce [N]",
            "Reel-in\nforce [N]",
            "Elevation\nangle [rad]",
            "Reel-in tether\nlength [m]",
            "Minimum tether\nlength [m]"
        ],
        tether_force_limits=[
            constructor.sys_props.tether_force_min_limit,
            constructor.sys_props.tether_force_max_limit
        ],
        reeling_speed_limits=[
            constructor.sys_props.reeling_speed_min_limit,
            constructor.sys_props.reeling_speed_max_limit
        ]
    )
    opt_diag_path = OUTPUT_PATH_OPTIMIZED.with_name('optimization_diagnostics.png')
    plt.savefig(opt_diag_path, dpi=200, bbox_inches='tight')
    print(f"Optimization diagnostics saved to: {opt_diag_path}")
    
    print()


if __name__ == "__main__":
    # Generate power curves using both methods
    direct_result, constructor = generate_direct_simulation_power_curves()
    # optimized_result = generate_optimized_power_curves(constructor)
    
    # Create comparison plots
    # plot_comparison(direct_result, optimized_result, constructor)
    
        
    print("=" * 80)
    print("POWER CURVE GENERATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. Direct simulation: {OUTPUT_PATH_DIRECT}")
    print(f"  2. Optimized curves: {OUTPUT_PATH_OPTIMIZED}")
    print(f"  3. Optimization details: {OUTPUT_PATH_OPTIMIZED.with_name('optimization_details.yml')}")
    print(f"  4. Comparison plot: {COMPARISON_PLOT_PATH}")
    print(f"  5. Optimization trajectories: {OUTPUT_PATH_OPTIMIZED.with_name('optimization_trajectories.png')}")
    print(f"  6. Optimization diagnostics: {OUTPUT_PATH_OPTIMIZED.with_name('optimization_diagnostics.png')}")
    print("\nAll done! ✓")
    
    # Close all figures to prevent display
    plt.close('all')

