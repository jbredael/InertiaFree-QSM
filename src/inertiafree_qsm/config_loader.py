# -*- coding: utf-8 -*-
"""Configuration loader for AWE power model.

Loads system configuration, wind resource data, and simulation settings from YAML files
and provides a simple interface for calculating power curves.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from awesio.validator import validate

from .qsm import (
    Cycle,
    NormalisedWindTable1D,
    SystemProperties,
    TractionPhase,
)


def load_yaml(file_path):
    """Load a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_system_config(file_path):
    """Load system configuration from YAML file.

    Extracts the relevant parameters for the QSM model from the awesIO
    system configuration format.

    Args:
        file_path (str): Path to the system configuration YAML file.

    Returns:
        dict: System properties dictionary compatible with SystemProperties class.
    """
    config = load_yaml(file_path)
    components = config.get('components', {})

    # Extract wing parameters
    wing = components.get('wing', {})
    wing_structure = wing.get('structure', {})
    wing_aero = wing.get('aerodynamics', {}).get('simple_aero_model', {})

    # Extract tether parameters
    tether = components.get('tether', {})
    tether_aero = tether.get('aerodynamics', {})
    tether_structure = tether.get('structure', {})

    # Extract ground station parameters
    ground_station = components.get('ground_station', {})
    drum = ground_station.get('drum', {})

    # Extract KCU parameters
    kcu = components.get('kcu', {})
    kcu_structure = kcu.get('structure', {})

    # Calculate kite mass (wing + bridle + KCU)
    wing_mass = wing_structure.get('mass_kg', 20.0)
    bridle = components.get('bridle', {})
    bridle_mass = bridle.get('structure', {}).get('mass_kg', 0.0)
    kcu_mass = kcu_structure.get('mass_kg', 0.0)
    kite_mass = wing_mass + bridle_mass + kcu_mass

    # Calculate lift and drag coefficients
    # Note: For QSM, 'powered' = high L/D for traction (reel-out)
    #       'depowered' = low L/D for retraction (reel-in)
    # Some YAML configs may have these inverted or use different conventions.
    cl_powered = wing_aero.get('lift_coefficient_reel_out', 0.8)
    cd_powered = wing_aero.get('drag_coefficient_reel_out', 0.2)

    # For depowered state, use values that give LOW L/D ratio (~2-4)
    # This is typical for a depowered kite during retraction
    cl_depowered = wing_aero.get('lift_coefficient_reel_in', 0.2)
    cd_depowered = wing_aero.get('drag_coefficient_reel_in', 0.1)

    # Sanity check: depowered L/D should be lower than powered L/D
    ld_powered = cl_powered / cd_powered if cd_powered > 0 else 4.0
    ld_depowered = cl_depowered / cd_depowered if cd_depowered > 0 else 2.0

    if ld_depowered > ld_powered:
        # If depowered has higher L/D than powered, the values are likely
        # swapped or incorrectly specified. Use typical depowered values.
        cl_depowered = 0.2
        cd_depowered = 0.1

    sys_props = {
        'kite_projected_area': wing_structure.get('projected_surface_area_m2', 20.0),
        'kite_mass': kite_mass,
        'tether_density': tether_structure.get('density_kg_m3', 724.0),
        'tether_diameter': tether_structure.get('diameter_m', 0.004),
        'tether_force_max_limit': tether_structure.get('max_tether_force_n', 10000.0),
        'tether_force_min_limit': tether_structure.get('max_tether_force_n', 10000.0) * 0.03,
        'kite_lift_coefficient_powered': cl_powered,
        'kite_drag_coefficient_powered': cd_powered,
        'kite_lift_coefficient_depowered': cl_depowered,
        'kite_drag_coefficient_depowered': cd_depowered,
        'reeling_speed_min_limit': 0.0,
        'reeling_speed_max_limit': drum.get('max_tether_speed_m_s', 10.0),
        'tether_drag_coefficient': tether_aero.get('drag_coefficient', 1.1),
    }

    return sys_props


def load_wind_resource(file_path):
    """Load wind resource data from YAML file.

    Args:
        file_path (str): Path to the wind resource YAML file.

    Returns:
        dict: Wind resource data containing altitude profiles and metadata.
    """
    config = load_yaml(file_path)

    wind_resource = {
        'metadata': config.get('metadata', {}),
        'altitudes': np.array(config.get('altitudes', [])),
        'wind_speed_bins': config.get('wind_speed_bins', {}),
        'wind_direction_bins': config.get('wind_direction_bins', {}),
        'clusters': config.get('clusters', []),
    }

    return wind_resource


def load_simulation_settings(file_path):
    """Load simulation settings from YAML file.

    Args:
        file_path (str): Path to the simulation settings YAML file.

    Returns:
        dict: Simulation settings compatible with the Cycle class.
    """
    config = load_yaml(file_path)

    # Parse cycle settings
    cycle_config = config.get('cycle', {})
    retraction_config = config.get('retraction', {})
    transition_config = config.get('transition', {})
    traction_config = config.get('traction', {})

    # Parse control tuples - convert lists to tuples
    def parse_control(control_config):
        if isinstance(control_config, list):
            return tuple(control_config)
        return control_config

    # Build settings dictionary
    settings = {
        'cycle': {
            'elevation_angle_traction': float(cycle_config.get(
                'elevation_angle_traction', 35 * np.pi / 180.
            )),
            'tether_length_start_retraction': float(cycle_config.get(
                'tether_length_start_retraction', 500.0
            )),
            'tether_length_end_retraction': float(cycle_config.get(
                'tether_length_end_retraction', 200.0
            )),
        },
        'retraction': {
            'control': parse_control(retraction_config.get('control', ('tether_force_ground', 10000))),
            'time_step': float(retraction_config.get('time_step', 0.25)),
        },
        'transition': {
            'control': parse_control(transition_config.get('control', ('reeling_speed', 0.0))),
            'time_step': float(transition_config.get('time_step', 0.25)),
        },
        'traction': {
            'control': parse_control(traction_config.get('control', ('reeling_factor', 0.37))),
            'time_step': float(traction_config.get('time_step', 0.25)),
            'azimuth_angle': float(traction_config.get(
                'azimuth_angle', 15. * np.pi / 180.
            )),
            'course_angle': float(traction_config.get(
                'course_angle', 110. * np.pi / 180.
            )),
        },
    }

    return settings


def create_wind_profile_from_resource(wind_resource, cluster_id=0):
    """Create a wind profile lookup table from wind resource data.

    Args:
        wind_resource (dict): Wind resource data from load_wind_resource.
        cluster_id (int): Which cluster profile to use (0-indexed, but clusters
            in the file use 1-indexed IDs).

    Returns:
        tuple: (heights, normalised_wind_speeds, reference_height)
    """
    altitudes = wind_resource['altitudes']
    clusters = wind_resource['clusters']

    if not clusters:
        raise ValueError("No clusters found in wind resource data")

    # Find the cluster with matching ID (clusters use 1-indexed IDs)
    cluster = None
    for c in clusters:
        if c.get('id') == cluster_id + 1:  # Convert 0-indexed to 1-indexed
            cluster = c
            break

    if cluster is None:
        # Fallback to index-based access
        if cluster_id >= len(clusters):
            raise ValueError(f"Cluster ID {cluster_id} not found in wind resource data")
        cluster = clusters[cluster_id]

    u_normalized = np.array(cluster.get('u_normalized', []))

    # Reference height from metadata
    metadata = wind_resource.get('metadata', {})
    reference_height = metadata.get('reference_height_m', 100.0)

    return altitudes, u_normalized, reference_height


def get_reference_wind_speeds(wind_resource):
    """Get the reference wind speed bin centers from wind resource.

    Args:
        wind_resource (dict): Wind resource data from load_wind_resource.

    Returns:
        np.ndarray: Array of reference wind speed bin centers.
    """
    wind_speed_bins = wind_resource.get('wind_speed_bins', {})
    return np.array(wind_speed_bins.get('bin_centers_m_s', []))


def calculate_power_curves(
    system_config_path,
    wind_resource_path,
    simulation_settings_path,
    output_path,
    wind_speeds=None,
    verbose=True,
    plot=True,
):
    """Calculate power curves for all wind clusters and save to YAML.

    This is the main entry point for power curve calculation. It loads all
    configuration files, validates them using awesIO, runs the QSM model for
    each wind speed and cluster, saves the results to a YAML file in awesIO
    power_curves_schema format, and optionally plots the results.

    Args:
        system_config_path (str): Path to the system configuration YAML file.
        wind_resource_path (str): Path to the wind resource YAML file.
        simulation_settings_path (str): Path to the simulation settings YAML file.
        output_path (str): Path to the output YAML file for power curves.
        wind_speeds (array-like, optional): Wind speeds to evaluate [m/s].
            Defaults to 4.0 to 25.5 m/s in 0.5 m/s steps.
        verbose (bool): Whether to print progress messages. Defaults to True.
        plot (bool): Whether to plot the power curves. Defaults to True.

    Returns:
        dict: The output dictionary containing metadata and power curves.
    """
    # Default wind speed range
    if wind_speeds is None:
        wind_speeds = np.arange(4.0, 26.0, 0.5)

    # Validate input files using awesIO
    if verbose:
        print("Validating input files...")

    validate(str(system_config_path))
    if verbose:
        print(f"  ✓ System config validated: {system_config_path}")

    validate(str(wind_resource_path))
    if verbose:
        print(f"  ✓ Wind resource validated: {wind_resource_path}")

    if verbose:
        print(f"  ✓ Simulation settings loaded: {simulation_settings_path}\n")

    # Load configurations
    sys_props_dict = load_system_config(system_config_path)
    wind_resource = load_wind_resource(wind_resource_path)
    simulation_settings = load_simulation_settings(simulation_settings_path)

    # Create system properties object
    sys_props = SystemProperties(sys_props_dict)

    # Get number of clusters and cluster info
    n_clusters = wind_resource.get('metadata', {}).get('n_clusters', 1)
    clusters = wind_resource.get('clusters', [])
    altitudes = wind_resource.get('altitudes', [])

    if verbose:
        print("System Configuration:")
        print(f"  Kite projected area: {sys_props.kite_projected_area:.1f} m²")
        print(f"  Kite mass: {sys_props.kite_mass:.1f} kg")
        print(f"  Max tether force: {sys_props.tether_force_max_limit:.0f} N")
        print(f"  Tether diameter: {sys_props.tether_diameter*1000:.1f} mm")
        print(f"  Number of wind clusters: {n_clusters}\n")

    # Calculate operating altitude and tether length
    tether_length_start = simulation_settings['cycle']['tether_length_start_retraction']
    tether_length_end = simulation_settings['cycle']['tether_length_end_retraction']
    elevation_angle = simulation_settings['cycle']['elevation_angle_traction']
    avg_tether_length = (tether_length_start + tether_length_end) / 2
    operating_altitude = avg_tether_length * np.sin(elevation_angle)

    # Calculate power curves for all clusters
    power_curves = []
    all_results = []  # Store for plotting

    for cluster_id in range(n_clusters):
        power_curve, results = _calculate_single_power_curve(
            cluster_id,
            wind_speeds,
            wind_resource,
            simulation_settings,
            sys_props,
            verbose,
        )
        power_curves.append(power_curve)
        all_results.append(results)

    # Determine cut-in and cut-out wind speeds from first cluster
    first_curve = power_curves[0]
    cycle_powers = first_curve['cycle_power_w']
    cut_in_ws = _find_cut_in_wind_speed(wind_speeds, cycle_powers)
    cut_out_ws = float(wind_speeds[-1])

    # Calculate nominal power (max power from all clusters)
    nominal_power = max(
        max(pc['cycle_power_w']) for pc in power_curves
    )

    # Prepare output in awesIO power_curves_schema format
    output = {
        'metadata': {
            'name': 'Ground-Gen Power Curves',
            'description': 'Power curves for pumping ground-gen AWE system',
            'note': 'Power curve data generated from QSM model',
            'awesIO_version': '0.1.0',
            'schema': 'power_curves_schema.yml',
            'time_created': datetime.now().isoformat(),
            'model_config': {
                'wing_area_m2': float(sys_props.kite_projected_area),
                'nominal_power_w': float(nominal_power),
                'nominal_tether_force_n': float(sys_props.tether_force_max_limit),
                'cut_in_wind_speed_m_s': float(cut_in_ws),
                'cut_out_wind_speed_m_s': float(cut_out_ws),
                'operating_altitude_m': float(operating_altitude),
                'tether_length_operational_m': float(avg_tether_length),
            },
            'wind_resource': {
                'n_clusters': n_clusters,
                'reference_height_m': float(wind_resource.get('metadata', {}).get(
                    'reference_height_m', 100.0
                )),
            },
        },
        'altitudes_m': [float(a) for a in altitudes],
        'reference_wind_speeds_m_s': [float(ws) for ws in wind_speeds],
        'power_curves': power_curves,
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    if verbose:
        print(f"\nPower curves saved to {output_path}")

    # Validate output file
    validate(str(output_path))
    if verbose:
        print(f"  ✓ Output file validated against power_curves_schema")

    # Plot if requested
    if plot:
        _plot_power_curves(wind_speeds, power_curves, output_path)

    return output


def _find_cut_in_wind_speed(wind_speeds, cycle_powers):
    """Find the cut-in wind speed where power first becomes positive.

    Args:
        wind_speeds (array-like): Wind speeds [m/s].
        cycle_powers (list): Cycle power values [W].

    Returns:
        float: Cut-in wind speed [m/s].
    """
    for ws, power in zip(wind_speeds, cycle_powers):
        if power > 0:
            return float(ws)
    return float(wind_speeds[0])


def _plot_power_curves(wind_speeds, power_curves, output_path):
    """Plot the power curves for all clusters.

    Args:
        wind_speeds (array-like): Wind speeds [m/s].
        power_curves (list): List of power curve dictionaries.
        output_path (Path): Output path for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for pc in power_curves:
        profile_id = pc['profile_id']
        cycle_power_kw = np.array(pc['cycle_power_w']) / 1000.0
        ax.plot(wind_speeds, cycle_power_kw, label=f'Cluster {profile_id}', linewidth=1.5)

    ax.set_xlabel('Reference Wind Speed [m/s]', fontsize=12)
    ax.set_ylabel('Cycle Average Power [kW]', fontsize=12)
    ax.set_title('Power Curves by Wind Profile Cluster', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([wind_speeds[0], wind_speeds[-1]])
    ax.set_ylim(bottom=0)
    ax.axhline(y=0, color='k', linewidth=0.5)

    # Save plot
    plot_path = output_path.with_suffix('.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Power curve plot saved to {plot_path}")


def _calculate_single_power_curve(
    cluster_id,
    wind_speeds,
    wind_resource,
    simulation_settings,
    sys_props,
    verbose,
):
    """Calculate power curve for a single wind cluster.

    Args:
        cluster_id (int): The cluster ID (0-indexed).
        wind_speeds (array-like): Wind speeds to evaluate [m/s].
        wind_resource (dict): Wind resource data.
        simulation_settings (dict): Simulation settings.
        sys_props (SystemProperties): System properties object.
        verbose (bool): Whether to print progress messages.

    Returns:
        tuple: (power_curve dict in awesIO format, list of detailed results)
    """
    # Create environment state for this cluster
    altitudes, u_normalized, h_ref = create_wind_profile_from_resource(
        wind_resource, cluster_id
    )

    # Get cluster data for probability weight
    clusters = wind_resource.get('clusters', [])
    cluster_data = None
    for c in clusters:
        if c.get('id') == cluster_id + 1:
            cluster_data = c
            break
    if cluster_data is None and cluster_id < len(clusters):
        cluster_data = clusters[cluster_id]

    # Get probability weight (frequency) from cluster data
    probability_weight = cluster_data.get('frequency', 1.0 / len(clusters)) if cluster_data else 1.0 / max(1, len(clusters))

    # Get v_normalized if available
    v_normalized = cluster_data.get('v_normalized', [0.0] * len(u_normalized)) if cluster_data else [0.0] * len(u_normalized)

    env_state = NormalisedWindTable1D()
    env_state.heights = list(altitudes)
    env_state.normalised_wind_speeds = list(u_normalized)
    env_state.h_ref = h_ref
    env_state.set_reference_height(h_ref)

    # Calculate speed ratio at operating altitude
    tether_length_start = simulation_settings['cycle']['tether_length_start_retraction']
    tether_length_end = simulation_settings['cycle']['tether_length_end_retraction']
    elevation_angle = simulation_settings['cycle']['elevation_angle_traction']
    avg_tether_length = (tether_length_start + tether_length_end) / 2
    operating_altitude = avg_tether_length * np.sin(elevation_angle)

    env_state.set_reference_wind_speed(1.0)
    env_state.calculate(operating_altitude)
    speed_ratio = env_state.wind_speed

    # Run simulation for each wind speed
    if verbose:
        print(f"Calculating power curve for cluster {cluster_id + 1}...")

    results = []
    cycle_powers = []
    reel_out_powers = []
    reel_in_powers = []
    reel_out_times = []
    reel_in_times = []
    cycle_times = []

    for i, ws in enumerate(wind_speeds):
        if verbose:
            print(f"  Wind speed {ws:.1f} m/s ({i+1}/{len(wind_speeds)})")

        result = _run_single_simulation(
            ws, env_state, simulation_settings, sys_props
        )
        results.append(result)

        cycle_powers.append(result['cycle_power_w'])
        reel_out_powers.append(result['reel_out_power_w'])
        reel_in_powers.append(result['reel_in_power_w'])
        reel_out_times.append(result['reel_out_time_s'])
        reel_in_times.append(result['reel_in_time_s'])
        cycle_times.append(result['cycle_time_s'])

    # Build power curve in awesIO format (profile_id is 1-indexed)
    power_curve = {
        'profile_id': cluster_id + 1,
        'speed_ratio_at_operating_altitude': float(speed_ratio),
        'u_normalized': [float(u) for u in u_normalized],
        'v_normalized': [float(v) for v in v_normalized],
        'probability_weight': float(probability_weight),
        'cycle_power_w': cycle_powers,
        'reel_out_power_w': reel_out_powers,
        'reel_in_power_w': reel_in_powers,
        'reel_out_time_s': reel_out_times,
        'reel_in_time_s': reel_in_times,
        'cycle_time_s': cycle_times,
    }

    return power_curve, results


def _run_single_simulation(wind_speed, env_state, simulation_settings, sys_props):
    """Run a single cycle simulation for one wind speed.

    Args:
        wind_speed (float): Reference wind speed [m/s].
        env_state (NormalisedWindTable1D): Environment state object.
        simulation_settings (dict): Simulation settings.
        sys_props (SystemProperties): System properties object.

    Returns:
        dict: Simulation result with power and timing data.
    """
    env_state.set_reference_wind_speed(wind_speed)

    settings = simulation_settings.copy()
    settings['cycle'] = simulation_settings['cycle'].copy()
    settings['cycle']['traction_phase'] = TractionPhase

    cycle = Cycle(settings, impose_operational_limits=True)

    try:
        error_in_phase, average_power = cycle.run_simulation(
            sys_props, env_state, print_summary=False
        )

        # Extract phase-specific data
        traction_duration = cycle.traction_phase.duration if hasattr(cycle, 'traction_phase') else 0.0
        retraction_duration = cycle.retraction_phase.duration if hasattr(cycle, 'retraction_phase') else 0.0
        transition_duration = cycle.transition_phase.duration if hasattr(cycle, 'transition_phase') else 0.0

        traction_energy = cycle.traction_phase.energy if hasattr(cycle, 'traction_phase') else 0.0
        retraction_energy = cycle.retraction_phase.energy if hasattr(cycle, 'retraction_phase') else 0.0

        reel_out_time = traction_duration
        reel_in_time = retraction_duration + transition_duration
        cycle_time = cycle.duration

        reel_out_power = traction_energy / traction_duration if traction_duration > 0 else 0.0
        reel_in_power = retraction_energy / retraction_duration if retraction_duration > 0 else 0.0

        return {
            'cycle_power_w': float(average_power),
            'reel_out_power_w': float(reel_out_power),
            'reel_in_power_w': float(reel_in_power),
            'reel_out_time_s': float(reel_out_time),
            'reel_in_time_s': float(reel_in_time),
            'cycle_time_s': float(cycle_time),
            'success': error_in_phase is None,
        }
    except Exception:
        return {
            'cycle_power_w': 0.0,
            'reel_out_power_w': 0.0,
            'reel_in_power_w': 0.0,
            'reel_out_time_s': 0.0,
            'reel_in_time_s': 0.0,
            'cycle_time_s': 0.0,
            'success': False,
        }
