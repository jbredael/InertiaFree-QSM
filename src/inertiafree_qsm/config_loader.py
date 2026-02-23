# -*- coding: utf-8 -*-
"""Configuration loader for AWE power model.

Loads system configuration, wind resource data, and simulation settings from YAML files
and provides a simple interface for calculating power curves.
"""

import numpy as np
import yaml
# from awesio.validator import validate  # Unused import, commented out

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
    kcu = components.get('control_system', {})
    kcu_structure = kcu.get('structure', {})

    # Calculate kite mass (wing + bridle + KCU)
    wing_mass = wing_structure.get('mass_kg')
    bridle = components.get('bridle', {})
    bridle_mass = bridle.get('structure', {}).get('mass_kg')
    kcu_mass = kcu_structure.get('mass_kg')
    kite_mass = wing_mass + bridle_mass + kcu_mass

    # Lift and drag coefficients
    cl_powered = wing_aero.get('lift_coefficient_reel_out')
    cd_powered = wing_aero.get('drag_coefficient_reel_out')

    cl_depowered = wing_aero.get('lift_coefficient_reel_in')
    cd_depowered = wing_aero.get('drag_coefficient_reel_in')

    sys_props = {
        'kite_projected_area': wing_structure.get('projected_surface_area_m2'),
        'kite_mass': kite_mass,
        'tether_density': tether_structure.get('density_kg_m3'),
        'tether_diameter': tether_structure.get('diameter_m'),
        'tether_force_max_limit': tether_structure.get('max_tether_force_n'),
        'tether_force_min_limit': tether_structure.get('max_tether_force_n') * 0.03,
        'kite_lift_coefficient_powered': cl_powered,
        'kite_drag_coefficient_powered': cd_powered,
        'kite_lift_coefficient_depowered': cl_depowered,
        'kite_drag_coefficient_depowered': cd_depowered,
        'reeling_speed_min_limit': 0.0,
        'reeling_speed_max_limit': drum.get('max_tether_speed_m_s'),
        'tether_drag_coefficient': tether_aero.get('drag_coefficient'),
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

    All angles in the config file are expected to be in degrees and will be
    converted to radians internally.

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

    # Helper function to convert degrees to radians
    def deg_to_rad(angle_deg):
        return float(angle_deg) * np.pi / 180.0

    # Build settings dictionary
    settings = {
        'cycle': {
            'elevation_angle_traction': deg_to_rad(cycle_config.get(
                'elevation_angle_traction'
            )),
            'tether_length_start_retraction': float(cycle_config.get(
                'tether_length_start_retraction'
            )),
            'tether_length_end_retraction': float(cycle_config.get(
                'tether_length_end_retraction'
            )),
        },
        'retraction': {
            'control': parse_control(retraction_config.get('control', ('tether_force_ground'))),
            'time_step': float(retraction_config.get('time_step')),
        },
        'transition': {
            'control': parse_control(transition_config.get('control', ('reeling_speed'))),
            'time_step': float(transition_config.get('time_step')),
        },
        'traction': {
            'control': parse_control(traction_config.get('control', ('reeling_factor'))),
            'time_step': float(traction_config.get('time_step')),
            'azimuth_angle': deg_to_rad(traction_config.get(
                'azimuth_angle'
            )),
            'course_angle': deg_to_rad(traction_config.get('course_angle'
            )),
        },
    }

    return settings


def get_direct_simulation_wind_speeds(file_path):
    """Get wind speed settings for direct simulation method.

    Args:
        file_path (str): Path to the simulation settings YAML file.

    Returns:
        np.ndarray: Array of wind speeds for direct simulation [m/s].
    """
    config = load_yaml(file_path)
    direct_config = config.get('direct_simulation', {})
    wind_config = direct_config.get('wind_speeds', {})
    
    cut_in = float(wind_config.get('cut_in'))
    cut_out = float(wind_config.get('cut_out'))
    step = float(wind_config.get('step'))
    
    return np.arange(cut_in, cut_out, step)


def get_optimization_wind_speed_settings(file_path):
    """Get wind speed settings for optimization method.

    Args:
        file_path (str): Path to the simulation settings YAML file.

    Returns:
        dict: Dictionary with keys 'cut_in', 'cut_out', 'n_points',
            'fine_n_points_near_cutout', 'fine_range_m_s'.
            Values can be None if auto-estimation is requested.
    """
    config = load_yaml(file_path)
    opt_config = config.get('optimization', {})
    wind_config = opt_config.get('wind_speeds', {})
    fine_config = wind_config.get('fine_resolution', {})
    
    # Read values, allowing None for auto-estimation
    cut_in = wind_config.get('cut_in', None)
    cut_out = wind_config.get('cut_out', None)
    n_points = wind_config.get('n_points', 50)
    
    # Read fine resolution settings
    fine_n_points = fine_config.get('n_points_near_cutout')
    fine_range = fine_config.get('range_m_s')
    
    # Convert to float if not None
    if cut_in is not None:
        cut_in = float(cut_in)
    if cut_out is not None:
        cut_out = float(cut_out)
    n_points = int(n_points)
    fine_n_points = int(fine_n_points)
    fine_range = float(fine_range)
    
    return {
        'cut_in': cut_in,
        'cut_out': cut_out,
        'n_points': n_points,
        'fine_n_points_near_cutout': fine_n_points,
        'fine_range_m_s': fine_range,
    }


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
    reference_height = metadata.get('reference_height_m')

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
