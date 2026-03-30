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


def load_system_config(file_path, validate_file=False):
    """Load system configuration from YAML file.

    Extracts the relevant parameters for the QSM model from the awesIO
    system configuration format.

    Args:
        file_path (str): Path to the system configuration YAML file.
        validate_file (bool): If True, validate the YAML against the awesIO schema.

    Returns:
        dict: System properties dictionary compatible with SystemProperties class.
    """
    if validate_file:
        try:
            from awesio.validator import validate as awesio_validate
            awesio_validate(input=file_path)
            print(f"  ✓ {file_path.name} validated against system_schema")
        except ImportError:
            print("  awesIO not installed; skipping validation.")
        except Exception as e:
            print(f"  Validation failed: {e}")

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
        'max_tether_length': tether_structure.get('length_m'),
        'max_generator_power': ground_station.get('generator', {}).get('max_power_w'),
    }

    return sys_props


def load_wind_resource(file_path, validate_file=True):
    """Load wind resource data from YAML file.

    Args:
        file_path (str): Path to the wind resource YAML file.
        validate_file (bool): If True, validate the YAML against the awesIO schema.

    Returns:
        dict: Wind resource data containing altitude profiles, metadata,
            wind speed bins, wind direction bins, and clusters.
    """
    if validate_file:
        try:
            from awesio.validator import validate as awesio_validate
            awesio_validate(input=file_path)
            print(f"  ✓ {file_path.name} validated against system_schema")
        except ImportError:
            print("  awesIO not installed; skipping validation.")
        except Exception as e:
            print(f"  Validation failed: {e}")
    config = load_yaml(file_path)

    wind_resource = {
        'metadata': config.get('metadata', {}),
        'altitudes': np.array(config.get('altitudes', [])),
        'wind_speed_bins': config.get('wind_speed_bins', {}),
        'wind_direction_bins': config.get('wind_direction_bins', {}),
        'clusters': config.get('clusters', []),
    }

    return wind_resource


def load_simulation_settings(file_path, sys_props, verbose=False):
    """Load all simulation settings from YAML file.

    All angles in the config file are expected to be in degrees and will be
    converted to radians internally. Optimizer x0 elevation is also converted.
    Tether length settings in the cycle section are expressed as fractions of
    the maximum tether length from the system configuration. Optimizer bounds
    for start and end tether lengths are also fractions.

    When ``tether_length_start_retraction`` is ``null`` in the YAML file, it
    defaults to 1.0 (the full tether length from the system configuration).

    Args:
        file_path (str): Path to the simulation settings YAML file.
        sys_props (dict): System properties dictionary (from ``load_system_config``).
            Must contain ``'max_tether_length'``.
        verbose (bool): If True, print all loaded settings.

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

    # --- reference tether length and force limits from system config --------
    max_tether_length = float(sys_props['max_tether_length'])
    force_min_limit = float(sys_props['tether_force_min_limit'])
    force_max_limit = float(sys_props['tether_force_max_limit'])

    # --- cycle settings -----------------------------------------------------
    cycle_config = config.get('cycle', {})
    retraction_config = config.get('retraction', {})
    transition_config = config.get('transition', {})
    traction_config = config.get('traction', {})
    ss_config = config.get('steady_state', {})
    phase_solver_config = config.get('phase_solver', {})
    cw_config = config.get('crosswind_pattern', {})
    lissajous_config = cw_config.get('lissajous', {})
    max_time_points = int(phase_solver_config.get('max_time_points'))


    tetherLengthStartTraction = cycle_config.get('tether_length_start_traction') * max_tether_length
    tetherLengthEndTraction = cycle_config.get('tether_length_end_traction') * max_tether_length

    cycle = {
        'elevation_angle_traction': deg_to_rad(
            cycle_config.get('elevation_angle_traction')
        ),
        'tether_length_start_traction': tetherLengthStartTraction,
        'tether_length_end_traction': tetherLengthEndTraction,
        'include_transition_energy': bool(cycle_config.get('include_transition_energy',)),
        'n_traction_points': int(cw_config.get('n_traction_points')),
        'n_pattern_eval_points': int(cw_config.get('n_pattern_eval_points')),
    }

    retraction = {
        'control': parse_control(
            retraction_config.get('control', ('tether_force_ground',))
        ),
        'time_step': float(retraction_config.get('time_step')),
        'max_time_points': max_time_points,
        'azimuth_angle': deg_to_rad(retraction_config.get('azimuth_angle')),
        'course_angle': deg_to_rad(retraction_config.get('course_angle')),

    }

    transition = {
        'control': parse_control(
            transition_config.get('control', ('reeling_speed',))
        ),
        'time_step': float(transition_config.get('time_step')),
        'max_time_points': max_time_points,
        'azimuth_angle': deg_to_rad(transition_config.get('azimuth_angle')),
        'course_angle': deg_to_rad(transition_config.get('course_angle')),
    }

    traction = {
        'control': parse_control(
            traction_config.get('control', ('reeling_factor',))
        ),
        'time_step': float(traction_config.get('time_step')),
        'azimuth_angle': deg_to_rad(traction_config.get('azimuth_angle')),
        'course_angle': deg_to_rad(traction_config.get('course_angle')),
        'max_time_points': max_time_points,
        'lissajous_elevation_amplitude': float(lissajous_config.get('elevation_amplitude')),
        'lissajous_azimuth_amplitude': float(lissajous_config.get('azimuth_amplitude')),
    }

    # --- direct simulation settings -----------------------------------------
    direct_config = config.get('direct_simulation', {})
    direct_wind = direct_config.get('wind_speeds', {})
    direct_fine = direct_wind.get('fine_resolution', {})
    direct_simulation = {
        'wind_speeds': {
            'cut_in': direct_wind.get('cut_in'),
            'cut_out': direct_wind.get('cut_out'),
            'n_points': int(direct_wind.get('n_points', 30)),
            'fine_resolution': {
                'n_points_near_cutout': int(direct_fine.get('n_points_near_cutout', 0)),
                'range_m_s': float(direct_fine.get('range_m_s', 2.0)),
            },
        },
    }

    # --- optimisation settings ----------------------------------------------
    opt_config = config.get('optimization', {})
    opt_wind = opt_config.get('wind_speeds', {})
    opt_fine = opt_wind.get('fine_resolution', {})
    opt_optimizer = opt_config.get('optimizer', {})
    opt_bounds_cfg = opt_config.get('bounds', {})
    opt_constraints_cfg = opt_config.get('constraints', {})

    # Build x0 with elevation angle in radians
    x0_list = list(opt_optimizer.get('x0', []))
    x0_list[2] = deg_to_rad(x0_list[2])

    # Build bounds array: force bounds from system properties,
    # elevation in radians, tether lengths as fractions * max tether length.
    elev_min = deg_to_rad(opt_bounds_cfg.get('elevation_angle_min'))
    elev_max = deg_to_rad(opt_bounds_cfg.get('elevation_angle_max'))
    start_min = float(opt_bounds_cfg.get('tether_length_start_fraction_min')) * max_tether_length
    start_max = float(opt_bounds_cfg.get('tether_length_start_fraction_max')) * max_tether_length
    end_min = float(opt_bounds_cfg.get('tether_length_end_fraction_min')) * max_tether_length
    end_max = float(opt_bounds_cfg.get('tether_length_end_fraction_max')) * max_tether_length

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
            [force_min_limit, force_max_limit],
            [force_min_limit, force_max_limit],
            [elev_min, elev_max],
            [start_min, start_max],
            [end_min, end_max],
        ]),
        'constraints': {
            'min_crosswind_patterns': int(opt_constraints_cfg.get('min_crosswind_patterns', 1)),
        },
    }

    steady_state = {
        'max_iterations': int(ss_config.get('max_iterations', 250)),
        'convergence_tolerance': float(ss_config.get('convergence_tolerance', 1e-6)),
    }

    settings = {
        'cycle': cycle,
        'retraction': retraction,
        'transition': transition,
        'traction': traction,
        'steady_state': steady_state,
        'direct_simulation': direct_simulation,
        'optimization': optimization,
    }

    if verbose:
        _print_simulation_settings(settings)

    return settings


def _print_simulation_settings(settings):
    """Print loaded simulation settings.

    Args:
        settings (dict): Parsed simulation settings.
    """
    cycle = settings['cycle']
    retraction = settings['retraction']
    transition = settings['transition']
    traction = settings['traction']
    direct = settings['direct_simulation']
    opt = settings['optimization']

    tether_length_start_traction = float(cycle.get('tether_length_start_traction', 0.0))
    tether_length_end_traction = float(cycle.get('tether_length_end_traction', 0.0))

    print("\nSimulation Settings")
    print("=" * 60)

    # -- Cycle ---------------------------------------------------------------
    print("\n  Cycle:")
    print(f"    Tether length start traction      : {tether_length_start_traction:.1f} m")
    print(f"    Tether length end traction        : {tether_length_end_traction:.1f} m")
    print(f"    Elevation angle traction             : "
          f"{np.degrees(cycle['elevation_angle_traction']):.1f} deg")

    # -- Phase controls ------------------------------------------------------
    print("\n  Retraction:")
    print(f"    Control  : {retraction['control']}")
    print(f"    Time step: {retraction['time_step']} s")
    print(f"    Azimuth angle : {np.degrees(retraction['azimuth_angle']):.1f} deg")
    print(f"    Course angle  : {np.degrees(retraction['course_angle']):.1f} deg")

    print("\n  Transition:")
    print(f"    Control  : {transition['control']}")
    print(f"    Time step: {transition['time_step']} s")
    print(f"    Azimuth angle : {np.degrees(transition['azimuth_angle']):.1f} deg")
    print(f"    Course angle  : {np.degrees(transition['course_angle']):.1f} deg")

    print("\n  Traction:")
    print(f"    Control      : {traction['control']}")
    print(f"    Time step    : {traction['time_step']} s")
    print(f"    Azimuth angle : {np.degrees(traction['azimuth_angle']):.1f} deg")
    print(f"    Course angle  : {np.degrees(traction['course_angle']):.1f} deg")

    # -- Direct simulation ---------------------------------------------------
    print("\n  Direct simulation:")
    dw = direct['wind_speeds']
    print(f"    Wind speeds  : cut-in={dw['cut_in']}, cut-out={dw['cut_out']}, "
          f"n_points={dw['n_points']}")
    dfr = dw['fine_resolution']
    print(f"    Fine res     : {dfr['n_points_near_cutout']} pts, "
          f"range={dfr['range_m_s']:.1f} m/s")

    # -- Optimisation --------------------------------------------------------
    print("\n  Optimisation:")
    ow = opt['wind_speeds']
    print(f"    Wind speeds  : cut-in={ow['cut_in']}, cut-out={ow['cut_out']}, "
          f"n_points={ow['n_points']}")
    fr = ow['fine_resolution']
    print(f"    Fine res     : {fr['n_points_near_cutout']} pts, "
          f"range={fr['range_m_s']:.1f} m/s")
    op = opt['optimizer']
    print(f"    Max iter     : {op['max_iterations']}")
    print(f"    ftol / eps   : {op['ftol']:.1e} / {op['eps']:.1e}")
    print(f"    x0           : {op['x0']}")
    print(f"    scaling      : {op['scaling']}")

    bounds = opt['bounds']
    print("\n    Optimiser bounds (absolute values used internally):")
    print(f"      Tether force out [N]       : [{bounds[0,0]:.1f}, {bounds[0,1]:.1f}]  "
          f"(from system props)")
    print(f"      Tether force in  [N]       : [{bounds[1,0]:.1f}, {bounds[1,1]:.1f}]  "
          f"(from system props)")
    print(f"      Elevation angle  [deg]     : "
          f"[{np.degrees(bounds[2,0]):.1f}, {np.degrees(bounds[2,1]):.1f}]")
    print(f"      Start tether length  [m]   : [{bounds[3,0]:.1f}, {bounds[3,1]:.1f}]")
    print(f"      End tether length    [m]   : [{bounds[4,0]:.1f}, {bounds[4,1]:.1f}]")

    print("=" * 60)
