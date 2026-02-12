# -*- coding: utf-8 -*-
"""Configuration loader for AWE power model.

Loads system configuration, wind resource data, and simulation settings from YAML files
and provides a simple interface for calculating power curves.
"""

import numpy as np
import yaml
from awesio.validator import validate

def load_yaml(file_path):
    """Load a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(file_path, 'r') as f:
        return yaml.full_load(f)


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


def get_cluster_data(wind_resource, cluster_id):
    """Get data for a specific cluster.

    Args:
        wind_resource (dict): Wind resource data.
        cluster_id (int): The cluster ID (0-indexed).

    Returns:
        dict: Cluster data including id, u_normalized, v_normalized, frequency, etc.
    """
    clusters = wind_resource.get('clusters', [])
    if cluster_id >= len(clusters):
        raise ValueError(f"Cluster ID {cluster_id} not found. Available clusters: 0-{len(clusters)-1}")
    return clusters[cluster_id]
