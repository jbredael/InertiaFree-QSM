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




def load_system_and_simulation_settings(system_config_path, simulation_settings_path,
                                        validate_file=False, verbose=False):
    """Load and combine system and simulation configuration in one call.

    Args:
        system_config_path (str or Path): Path to system configuration YAML.
        simulation_settings_path (str or Path): Path to simulation settings YAML.
        validate_file (bool): Validate system config against awesIO schema when True.
        verbose (bool): Print parsed simulation settings when True.

    Returns:
        tuple: (settings, sys_props) where:
            - settings (dict): Parsed simulation settings dictionary.
            - sys_props (dict): System properties dictionary for SystemProperties.
    """
    if validate_file:
        try:
            from awesio.validator import validate as awesio_validate
            awesio_validate(input=system_config_path)
            config_name = getattr(system_config_path, 'name', str(system_config_path))
            print(f"  ✓ {config_name} validated against system_schema")
        except ImportError:
            print("  awesIO not installed; skipping validation.")
        except Exception as e:
            print(f"  Validation failed: {e}")

    system_config = load_yaml(system_config_path)
    sim_config = load_yaml(simulation_settings_path)

    # Build base system properties from system configuration.
    components = system_config.get('components', {})
    wing = components.get('wing', {})
    wing_structure = wing.get('structure', {})
    tether = components.get('tether', {})
    tether_structure = tether.get('structure', {})
    ground_station = components.get('ground_station', {})
    drum = ground_station.get('drum', {})
    kcu = components.get('control_system', {})
    kcu_structure = kcu.get('structure', {})

    wing_mass = wing_structure.get('mass')
    bridle = components.get('bridle', {})
    bridle_mass = bridle.get('structure', {}).get('mass')
    kcu_mass = kcu_structure.get('mass')
    kite_mass = wing_mass + bridle_mass + kcu_mass

    tether_force_max_limit = tether_structure.get('max_tether_force')
    tether_force_min_limit = tether_structure.get('min_tether_force')
    if tether_force_min_limit is None and tether_force_max_limit is not None:
        tether_force_min_limit = 0.03 * tether_force_max_limit

    base_sys_props = {
        'kite_projected_area': wing_structure.get('projected_surface_area'),
        'kite_mass': kite_mass,
        'tether_density': tether_structure.get('density'),
        'tether_diameter': tether_structure.get('diameter'),
        'tether_force_min_limit': tether_force_min_limit,
        'tether_force_max_limit': tether_force_max_limit,
        'reeling_speed_min_limit': drum.get('min_tether_speed'),
        'reeling_speed_max_limit': drum.get('max_tether_speed'),
        'max_tether_length': tether_structure.get('length'),
        'max_generator_power': ground_station.get('generator', {}).get('max_power'),
    }

    # Parse simulation settings.
    cycle_config = sim_config.get('cycle', {})
    retraction_config = sim_config.get('retraction', {})
    transition_config = sim_config.get('transition', {})
    traction_config = sim_config.get('traction', {})
    ss_config = sim_config.get('steady_state', {})
    phase_solver_config = sim_config.get('phase_solver', {})
    cw_config = sim_config.get('crosswind_pattern', {})
    lissajous_config = cw_config.get('lissajous', {})

    aero_section = sim_config.get('aerodynamics', {})

    def resolve_aero(sim_key, sys_key):
        val = aero_section.get(sim_key)
        if val is None:
            val = base_sys_props.get(sys_key)
        if val is None:
            raise KeyError(
                f"Missing aerodynamic setting '{sim_key}' in simulation settings "
                f"and no fallback '{sys_key}' in system config."
            )
        return float(val)

    aero_config = {
        'kite_lift_coefficient_powered': resolve_aero(
            'kite_lift_coefficient_reel_out', 'kite_lift_coefficient_powered'
        ),
        'kite_drag_coefficient_powered': resolve_aero(
            'kite_drag_coefficient_reel_out', 'kite_drag_coefficient_powered'
        ),
        'kite_lift_coefficient_depowered': resolve_aero(
            'kite_lift_coefficient_reel_in', 'kite_lift_coefficient_depowered'
        ),
        'kite_drag_coefficient_depowered': resolve_aero(
            'kite_drag_coefficient_reel_in', 'kite_drag_coefficient_depowered'
        ),
        'tether_drag_coefficient': resolve_aero(
            'tether_drag_coefficient', 'tether_drag_coefficient'
        ),
    }

    max_tether_length = float(base_sys_props['max_tether_length'])
    force_max_limit = float(base_sys_props['tether_force_max_limit'])
    minimum_tether_force = cycle_config.get('minimum_tether_force')
    if minimum_tether_force is None:
        minimum_tether_force = base_sys_props.get('tether_force_min_limit')
    if minimum_tether_force is None:
        minimum_tether_force = 0.03 * force_max_limit
    force_min_limit = float(minimum_tether_force)

    max_time_points = int(phase_solver_config.get('max_time_points'))

    direct_config = sim_config.get('direct_simulation', {})

    start_fraction_raw = cycle_config.get('tether_length_start_retraction')
    start_fraction = float(start_fraction_raw) if start_fraction_raw is not None else 1.0

    end_fraction_raw = cycle_config.get('tether_length_end_retraction')
    end_fraction = float(end_fraction_raw) if end_fraction_raw is not None else 0.5

    tetherLengthStartRetraction = start_fraction * max_tether_length
    tetherLengthEndRetraction = end_fraction * max_tether_length

    cycle = {
        'minimum_tether_force': force_min_limit,
        'minimum_height': float(cycle_config.get('minimum_height', 0.0)),
        'elevation_angle_traction': np.deg2rad(
            cycle_config.get('elevation_angle_traction')
        ),
        'tether_length_start_retraction': tetherLengthStartRetraction,
        'tether_length_end_retraction': tetherLengthEndRetraction,
        'include_transition_energy': bool(cycle_config.get('include_transition_energy',)),
        'n_traction_points': int(cw_config.get('n_traction_points')),
        'n_pattern_eval_points': int(cw_config.get('n_pattern_eval_points')),
    }

    retraction = {
        'control': tuple(retraction_config.get('control')),
        # control: (control_type, setpoint)
        'time_step': float(retraction_config.get('time_step')),
        'max_time_points': max_time_points,
        'azimuth_angle': np.deg2rad(retraction_config.get('azimuth_angle')),
        'course_angle': np.deg2rad(retraction_config.get('course_angle')),
    }

    transition = {
        'control': tuple(transition_config.get('control')),
        'time_step': float(transition_config.get('time_step')),
        'max_time_points': max_time_points,
        'azimuth_angle': np.deg2rad(transition_config.get('azimuth_angle')),
        'course_angle': np.deg2rad(transition_config.get('course_angle')),
    }

    traction = {
        'control': tuple(traction_config.get('control')),
        'time_step': float(traction_config.get('time_step')),
        'azimuth_angle': np.deg2rad(traction_config.get('azimuth_angle')),
        'course_angle': np.deg2rad(traction_config.get('course_angle')),
        'max_time_points': max_time_points,
        'lissajous_elevation_amplitude': float(lissajous_config.get('elevation_amplitude')),
        'lissajous_azimuth_amplitude': float(lissajous_config.get('azimuth_amplitude')),
    }

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
        'tether_length_start_retraction': tetherLengthStartRetraction,
        'tether_length_end_retraction': tetherLengthEndRetraction,
    }

    opt_config = sim_config.get('optimization', {})
    opt_wind = opt_config.get('wind_speeds', {})
    opt_fine = opt_wind.get('fine_resolution', {})
    opt_optimizer = opt_config.get('optimizer', {})
    opt_bounds_cfg = opt_config.get('bounds', {})
    opt_constraints_cfg = opt_config.get('constraints') or {}

    x0_list = list(opt_optimizer.get('x0', []))

    start_min = float(opt_bounds_cfg.get('fraction_tether_length_retraction_start_min',
                       opt_bounds_cfg.get('tether_length_start_fraction_min'))) * max_tether_length
    start_max = float(opt_bounds_cfg.get('fraction_tether_length_retraction_start_max',
                       opt_bounds_cfg.get('tether_length_start_fraction_max'))) * max_tether_length
    end_min = float(opt_bounds_cfg.get('fraction_tether_length_retraction_end_min',
                     opt_bounds_cfg.get('tether_length_end_fraction_min'))) * max_tether_length
    end_max = float(opt_bounds_cfg.get('fraction_tether_length_retraction_end_max',
                     opt_bounds_cfg.get('tether_length_end_fraction_max'))) * max_tether_length

    rs_out_min = float(opt_bounds_cfg.get('reeling_speed_traction_min'))
    rs_out_max = float(opt_bounds_cfg.get('reeling_speed_traction_max'))
    rs_in_min = float(opt_bounds_cfg.get('reeling_speed_retraction_min'))
    rs_in_max = float(opt_bounds_cfg.get('reeling_speed_retraction_max'))

    elev_min_deg = float(opt_bounds_cfg.get('elevation_angle_traction_min', 10.0))
    elev_max_deg = float(opt_bounds_cfg.get('elevation_angle_traction_max', 50.0))

    opt_optimize_vars = opt_optimizer.get('optimize_variables', {})

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
            'optimize_variables': {
                'reeling_speed_traction': bool(opt_optimize_vars.get('reeling_speed_traction', True)),
                'reeling_speed_retraction': bool(opt_optimize_vars.get('reeling_speed_retraction', True)),
                'fraction_tether_length_retraction_start': bool(opt_optimize_vars.get('fraction_tether_length_retraction_start', True)),
                'fraction_tether_length_retraction_end': bool(opt_optimize_vars.get('fraction_tether_length_retraction_end', True)),
                'elevation_angle_traction': bool(opt_optimize_vars.get('elevation_angle_traction', False)),
            },
        },
        'bounds': {
            'reeling_speed_out': (rs_out_min, rs_out_max),
            'reeling_speed_in': (rs_in_min, rs_in_max),
            'tether_length_start': (start_min, start_max),
            'tether_length_end': (end_min, end_max),
            'elevation_angle_traction': (np.deg2rad(elev_min_deg), np.deg2rad(elev_max_deg)),
        },
        'constraints': {
            'min_crosswind_patterns': int(opt_constraints_cfg.get('min_crosswind_patterns', 1)),
            'min_tether_length_fraction_difference': float(opt_constraints_cfg.get('min_tether_length_fraction_difference', 0.05)),
            'max_difference_elevation_angle_steps': (
                float(opt_constraints_cfg['max_difference_elevation_angle_steps'])
                if opt_constraints_cfg.get('max_difference_elevation_angle_steps') is not None
                else None
            ),
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

    merged_sys_props = dict(base_sys_props)
    merged_sys_props.update(aero_config)
    merged_sys_props['tether_force_min_limit'] = force_min_limit

    if verbose:
        _print_simulation_settings(settings, max_tether_length,
                                   start_fraction, end_fraction,
                                   opt_bounds_cfg)

    return settings, merged_sys_props


def _print_simulation_settings(settings, maxTetherLength, startFraction,
                               endFraction, optBoundsCfg):
    """Print loaded simulation settings.

    Args:
        settings (dict): Parsed simulation settings.
        maxTetherLength (float): Maximum tether length from system config [m].
        startFraction (float): Start-of-retraction tether length fraction [-].
        endFraction (float): End-of-retraction tether length fraction [-].
        optBoundsCfg (dict): Raw optimiser bounds section from the YAML file.
    """
    cycle = settings['cycle']
    retraction = settings['retraction']
    transition = settings['transition']
    traction = settings['traction']
    direct = settings['direct_simulation']
    opt = settings['optimization']

    print("\nSimulation Settings")
    print("=" * 60)

    # -- Cycle ---------------------------------------------------------------
    print("\n  Cycle:")
    print(f"    Max tether length (system config)   : {maxTetherLength:.1f} m")
    print(f"    Tether length start retraction       : {startFraction:.2f}  "
          f"({cycle['tether_length_start_retraction']:.1f} m)")
    print(f"    Tether length end retraction         : {endFraction:.2f}  "
          f"({cycle['tether_length_end_retraction']:.1f} m)")
    elev_trac = cycle['elevation_angle_traction']
    elev_trac_arr = np.asarray(elev_trac).flatten()
    if elev_trac_arr.size == 1:
        elev_trac_str = f"{np.degrees(elev_trac_arr[0]):.1f} deg"
    else:
        elev_trac_str = "[" + ", ".join(f"{v:.1f}" for v in np.degrees(elev_trac_arr)) + "] deg"
    print(f"    Elevation angle traction             : {elev_trac_str}")

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

    startFracMin = float(optBoundsCfg.get('tether_length_start_fraction_min', 0))
    startFracMax = float(optBoundsCfg.get('tether_length_start_fraction_max', 0))
    endFracMin = float(optBoundsCfg.get('tether_length_end_fraction_min', 0))
    endFracMax = float(optBoundsCfg.get('tether_length_end_fraction_max', 0))
    print(f"      Start tether length  [m]   : [{bounds[0,0]:.1f}, {bounds[0,1]:.1f}]  "
          f"(fractions: [{startFracMin}, {startFracMax}])")
    print(f"      End tether length    [m]   : [{bounds[1,0]:.1f}, {bounds[1,1]:.1f}]  "
          f"(fractions: [{endFracMin}, {endFracMax}])")

    print("=" * 60)
