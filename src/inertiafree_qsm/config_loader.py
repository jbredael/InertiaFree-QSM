# -*- coding: utf-8 -*-
"""Configuration loader for AWE power model.

Loads system configuration, wind resource data, and simulation settings from YAML files
and returns them as dictionaries.
"""

import numpy as np
import yaml


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
        dict: Wind resource data containing altitude profiles, metadata,
            wind speed bins, wind direction bins, and clusters.
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
    """Load all simulation settings from YAML file.

    All angles in the config file are expected to be in degrees and will be
    converted to radians internally. Optimizer x0 elevation is also converted.
    Tether length bounds expressed as fractions are converted to metres using
    ``cycle.tether_length_start_retraction`` as the reference length.

    Args:
        file_path (str): Path to the simulation settings YAML file.

    Returns:
        dict: Complete simulation settings dictionary including keys
            ``'cycle'``, ``'retraction'``, ``'transition'``, ``'traction'``,
            ``'direct_simulation'``, and ``'optimization'``.
    """
    config = load_yaml(file_path)

    # --- helper functions ---------------------------------------------------
    def parse_control(control_config):
        if isinstance(control_config, list):
            return tuple(control_config)
        return control_config

    def deg_to_rad(angle_deg):
        return float(angle_deg) * np.pi / 180.0

    # --- cycle settings -----------------------------------------------------
    cycle_config = config.get('cycle', {})
    retraction_config = config.get('retraction', {})
    transition_config = config.get('transition', {})
    traction_config = config.get('traction', {})

    reference_tether_length = float(cycle_config.get('tether_length_start_retraction'))

    cycle = {
        'elevation_angle_traction': deg_to_rad(
            cycle_config.get('elevation_angle_traction')
        ),
        'tether_length_start_retraction': reference_tether_length,
        'tether_length_end_retraction': float(
            cycle_config.get('tether_length_end_retraction')
        ),
    }

    retraction = {
        'control': parse_control(
            retraction_config.get('control', ('tether_force_ground',))
        ),
        'time_step': float(retraction_config.get('time_step')),
    }

    transition = {
        'control': parse_control(
            transition_config.get('control', ('reeling_speed',))
        ),
        'time_step': float(transition_config.get('time_step')),
    }

    traction = {
        'control': parse_control(
            traction_config.get('control', ('reeling_factor',))
        ),
        'time_step': float(traction_config.get('time_step')),
        'azimuth_angle': deg_to_rad(traction_config.get('azimuth_angle')),
        'course_angle': deg_to_rad(traction_config.get('course_angle')),
    }

    # --- direct simulation settings -----------------------------------------
    direct_config = config.get('direct_simulation', {})
    direct_wind = direct_config.get('wind_speeds', {})
    direct_simulation = {
        'wind_speeds': {
            'cut_in': float(direct_wind.get('cut_in')),
            'cut_out': float(direct_wind.get('cut_out')),
            'step': float(direct_wind.get('step')),
        },
    }

    # --- optimisation settings ----------------------------------------------
    opt_config = config.get('optimization', {})
    opt_wind = opt_config.get('wind_speeds', {})
    opt_fine = opt_wind.get('fine_resolution', {})
    opt_optimizer = opt_config.get('optimizer', {})
    opt_bounds_cfg = opt_config.get('bounds', {})

    # Build x0 with elevation angle in radians
    x0_list = list(opt_optimizer.get('x0', []))
    x0_list[2] = deg_to_rad(x0_list[2])

    # Build bounds array: force rows left as nan (filled from system props),
    # elevation in radians, tether lengths as fractions * reference length.
    elev_min = deg_to_rad(opt_bounds_cfg.get('elevation_angle_min', 25.0))
    elev_max = deg_to_rad(opt_bounds_cfg.get('elevation_angle_max', 60.0))
    diff_min = float(opt_bounds_cfg.get('tether_length_diff_fraction_min', 0.1)) * reference_tether_length
    diff_max = float(opt_bounds_cfg.get('tether_length_diff_fraction_max', 0.8)) * reference_tether_length
    min_min = float(opt_bounds_cfg.get('min_tether_length_fraction_min', 0.2)) * reference_tether_length
    min_max = float(opt_bounds_cfg.get('min_tether_length_fraction_max', 0.8)) * reference_tether_length

    optimization = {
        'wind_speeds': {
            'cut_in': opt_wind.get('cut_in'),
            'cut_out': opt_wind.get('cut_out'),
            'n_points': int(opt_wind.get('n_points')),
            'fine_resolution': {
                'n_points_near_cutout': int(opt_fine.get('n_points_near_cutout')),
                'range_m_s': float(opt_fine.get('range_m_s')),
            },
        },
        'optimizer': {
            'max_iterations': int(opt_optimizer.get('max_iterations')),
            'ftol': float(opt_optimizer.get('ftol')),
            'eps': float(opt_optimizer.get('eps')),
            'x0': np.array(x0_list, dtype=float),
            'scaling': np.array(opt_optimizer.get('scaling', []), dtype=float),
        },
        'bounds': np.array([
            [np.nan, np.nan],
            [np.nan, np.nan],
            [elev_min, elev_max],
            [diff_min, diff_max],
            [min_min, min_max],
        ]),
    }

    settings = {
        'cycle': cycle,
        'retraction': retraction,
        'transition': transition,
        'traction': traction,
        'direct_simulation': direct_simulation,
        'optimization': optimization,
    }

    return settings
