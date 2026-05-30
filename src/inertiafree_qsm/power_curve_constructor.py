# -*- coding: utf-8 -*-
"""Power curve generation for AWE systems.

This module provides functionality to generate power curves using either direct
simulation or sequential optimization with warm starts. It includes constraint
handling and detailed diagnostics.

All wind speeds are referenced at the height specified in the wind_resource.yml
metadata (reference_height). The model uses the wind profile to derive wind
speeds at operating altitudes.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

from .config_loader import (
    load_system_and_simulation_settings,
    load_wind_resource,
)
from .qsm import (
    Cycle,
    NormalisedWindTable1D,
    OperationalLimitViolation,
    PhaseError,
    SteadyStateError,
    SystemProperties,
    TractionPhase,
)
from .cycle_optimizer import CycleOptimizer
from . import plotting


class PowerCurveConstructor:
    """Generate power curves for an AWE system using direct simulation or optimization.

    This class supports two methods:
    1. Direct simulation: Fast method using pre-defined cycle settings
    2. Sequential optimization: Uses warm starts for optimal cycle performance

    Args:
        system_config_path (str): Path to the system configuration YAML file.
        wind_resource_path (str): Path to the wind resource YAML file.
        simulation_settings_path (str): Path to the simulation settings YAML file.

    Attributes:
        sys_props (SystemProperties): System properties object.
        wind_resource (dict): Wind resource data.
        simulation_settings (dict): Simulation settings.
        reference_height (float): Reference height for wind speed [m].
        wind_speeds (array): Wind speeds at reference height [m/s].
        x_opts (list): Optimal solutions for each wind speed (optimization only).
        x0_list (list): Starting points for each optimization (optimization only).
        optimization_details (list): Optimization algorithm details (optimization only).
        constraints (list): Constraint values at optimal points (optimization only).
        performance_indicators (list): Performance KPIs for each wind speed.
    """

    def __init__(
        self,
        system_config_path,
        wind_resource_path,
        simulation_settings_path,
        validate_file=True, verbose=False):
        """Initialize the power curve optimizer with configuration files."""
        self.system_config_path = Path(system_config_path)
        self.wind_resource_path = Path(wind_resource_path)
        self.simulation_settings_path = Path(simulation_settings_path)

        # Load system config and simulation settings in one call.
        self.simulation_settings, self.sys_props_dict = load_system_and_simulation_settings(
            self.system_config_path,
            self.simulation_settings_path,
            validate_file=validate_file,
            verbose=verbose,
        )
        self.wind_resource = load_wind_resource(self.wind_resource_path, validate_file=validate_file)
        # Create system properties object
        self.sys_props = SystemProperties(self.sys_props_dict)

        # Reference height from wind resource metadata
        self.reference_height = self.wind_resource.get('metadata', {}).get(
            'reference_height'
        )

        # Number of profiles
        self.n_profiles = self.wind_resource.get('metadata', {}).get('n_profiles')

 
        # Initialize result storage
        self.wind_speeds = None
        self.x_opts = []
        self.x0_list = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

    def create_environment(self, profile_id=1):
        """Create an environment state object for a specific wind profile.

        Args:
            profile_id (int): The profile ID (1-indexed).

        Returns:
            NormalisedWindTable1D: Environment state with wind profile.
        """

        profiles = self.wind_resource['profiles']

        if not profiles:
            raise ValueError("No profiles found in wind resource data")

        # Find the profile with matching ID (profiles use 1-indexed IDs)
        profile = None
        for c in profiles:
            if c.get('id') == profile_id:
                profile = c
                break

        if profile is None:
            if profile_id >= len(profiles):
                raise ValueError(
                    f"profile ID {profile_id} not found in wind resource data"
                )
            profile = profiles[profile_id]

        env_state = NormalisedWindTable1D()
        env_state.h_ref = self.wind_resource.get('metadata', {}).get('reference_height')
        env_state.heights = list(self.wind_resource['altitudes'])
        env_state.normalised_wind_speeds = list(np.array(profile.get('u_normalized', [])))
        env_state.set_reference_height(env_state.h_ref)

        return env_state

    def _generate_wind_speed_array(self, settings_key='optimization', profile_id=None, verbose=False):
        """Generate wind speed array from configuration settings.

        Args:
            settings_key (str): Key in simulation_settings to use ('optimization' or 'direct_simulation').
            profile_id (int, optional): Not used (kept for compatibility).
            verbose (bool): Whether to print messages.

        Returns:
            np.ndarray: Array of wind speeds [m/s].

        Raises:
            ValueError: If cut_in or cut_out wind speeds are not specified in configuration.
        """
        settings = self.simulation_settings[settings_key]
        vw_cut_in = settings['wind_speeds']['cut_in']
        vw_cut_out = settings['wind_speeds']['cut_out']
        n_points = settings['wind_speeds']['n_points']
        fine_n_points_near_cutout = settings['wind_speeds']['fine_resolution']['n_points_near_cutout']
        fine_range_m_s = settings['wind_speeds']['fine_resolution']['range_m_s']

        # Check that cut-in and cut-out are specified
        if vw_cut_in is None:
            raise ValueError(f"cut_in wind speed must be specified in {settings_key} settings")
        if vw_cut_out is None:
            raise ValueError(f"cut_out wind speed must be specified in {settings_key} settings")

        # Generate wind speed array with optional fine resolution
        if fine_n_points_near_cutout > 0:
            wind_speeds = np.linspace(vw_cut_in, vw_cut_out - fine_range_m_s, n_points, endpoint=False)
            fine_points = np.linspace(
                vw_cut_out - fine_range_m_s,
                vw_cut_out - 0.05,
                fine_n_points_near_cutout,
            )
            wind_speeds = np.concatenate((wind_speeds, fine_points))
        else:
            wind_speeds = np.linspace(vw_cut_in, vw_cut_out, n_points)

        return wind_speeds


    def simulate_single_wind_speed(self, wind_speed, profile_id=1, method='direct',
                                    output_path=None, verbose=False,
                                    show_plot=False, save_plot=False,
                                    validate_file=False):
        """Simulate a single cycle at one wind speed.

        Produces a full power-curve YAML file (with a single wind speed entry)
        via ``_build_and_save_output`` and optionally plots the cycle detail.

        Args:
            wind_speed (float): Reference wind speed at reference height [m/s].
            profile_id (int): profile ID (1-indexed).
            method (str): Simulation method, either 'direct' or 'optimization'.
            output_path (str or Path, optional): Path to save the YAML output.
            show_plot (bool): If True, display the cycle detail plot. Defaults to False.
            save_plot (bool): If True, save the cycle detail plot as a PNG alongside
                the YAML output. Requires ``output_path``. Defaults to False.
            validate_file (bool): If True, validate the saved YAML against the
                awesIO schema. Defaults to False.

        Returns:
            dict: Full power-curve output dict (single wind speed).

        Raises:
            ValueError: If method is not 'direct' or 'optimization'.
        """
        env_state = self.create_environment(profile_id)

        if method == 'direct':
            entry = self._run_single_simulation_direct(wind_speed, env_state)
        elif method == 'optimization':
            entry = self._run_single_simulation_optimized(wind_speed, env_state, show_plot=show_plot,
                                                           verbose=verbose)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'direct' or 'optimization'."
            )

        output = self._build_and_save_output(
            profile_ids=[profile_id],
            wind_speed_data_per_profile=[[entry]],
            method_name=method.replace('_', ' ').title(),
            output_path=output_path,
            verbose=output_path is not None,
            validate_file=validate_file,
        )

        if (show_plot or save_plot) and output_path is not None:
            fig_path = Path(output_path).with_suffix('.pdf') if save_plot else None
            plotting.plot_cycle_detail(
                output_path, wind_speed, profile_id=profile_id,
                output_path=fig_path, show_plot=show_plot,
            )

        if method == 'optimization' and (show_plot or save_plot):
            history = getattr(self, 'last_optimization_history', None)
            var_names = getattr(self, 'last_optimization_var_names', None)
            if history:
                evo_path = None
                if save_plot and output_path is not None:
                    evo_path = Path(output_path).with_name(
                        Path(output_path).stem + '_opt_evolution.pdf'
                    )
                plotting.plot_optimization_evolution(
                    history, wind_speed, var_names=var_names,
                    output_path=evo_path, show_plot=show_plot,
                )

        return output

    def generate_power_curves_direct(
        self,
        wind_speeds=None,
        profile_ids=None,
        output_path=None,
        verbose=True,
        show_plot=False,
        save_plot=False,
        validate_file=False,):
        """Generate power curves using direct simulation (non-optimized).

        This method runs the QSM model with pre-defined cycle settings for each
        wind speed. It's much faster than optimization but may not find optimal
        performance.

        Args:
            wind_speeds (array-like, optional): Wind speeds to evaluate [m/s].
                Defaults to wind speeds defined in simulation settings.
            profile_ids (list, optional): List of profile IDs (1-indexed) to calculate.
                If None, calculates all profiles/profiles.
            output_path (str, optional): Path to save the output YAML file.
            verbose (bool): Whether to print progress messages.
            show_plot (bool): If True, display the power curve plot after generation.
                Defaults to False.
            save_plot (bool): If True, save the power curve plot as a PNG alongside
                the YAML output. Requires ``output_path``. Defaults to False.
            validate_file (bool): If True, validate the saved YAML against the
                awesIO schema. Defaults to False.

        Returns:
            dict: Power curve data in awesIO format.
        """
        # Determine which profiles to process
        if profile_ids is None:
            profile_ids = list(range(1, self.n_profiles + 1))
        elif not isinstance(profile_ids, (list, tuple)):
            profile_ids = [profile_ids]

        # Determine wind speeds once, estimating from the first profile if needed
        if wind_speeds is not None:
            wind_speeds = np.array(wind_speeds)
            if verbose:
                print(f"Using custom wind speeds: {wind_speeds}")
        else:
            wind_speeds = self._generate_wind_speed_array(
                settings_key='direct_simulation',
                profile_id=profile_ids[0],
                verbose=verbose
            )

        # Calculate power curves for selected profiles
        wind_speed_data_per_profile = []
        for profile_id in profile_ids:
            wind_speed_data = self._calculate_power_curve_direct(
                profile_id, wind_speeds, verbose
            )
            wind_speed_data_per_profile.append(wind_speed_data)

        # Build full output with metadata and save
        output = self._build_and_save_output(
            profile_ids=profile_ids,
            wind_speed_data_per_profile=wind_speed_data_per_profile,
            method_name='Direct Simulation',
            output_path=output_path,
            verbose=verbose,
            validate_file=validate_file,
        )

        if (show_plot or save_plot) and output_path is not None:
            fig_path = Path(output_path).with_suffix('.pdf') if save_plot else None
            plotting.plot_power_curve(output_path, output_path=fig_path, show_plot=show_plot)

        return output

    def generate_power_curves_optimized(
        self,
        wind_speeds=None,
        profile_ids=None,
        output_path=None,
        verbose=True,
        show_plot=False,
        save_plot=False,
        validate_file=False,):
        """Generate power curves using SLSQP optimization.

        This method optimizes cycle parameters (reeling factors and tether
        lengths) for each wind speed to maximize average cycle power.

        Args:
            wind_speeds (array-like, optional): Wind speeds to evaluate [m/s].
                Defaults to wind speeds defined in optimization settings.
            profile_ids (list, optional): List of profile IDs (1-indexed) to calculate.
                If None, calculates all profiles/profiles.
            output_path (str, optional): Path to save the output YAML file.
            verbose (bool): Whether to print progress messages.
            show_plot (bool): If True, display the power curve plot after generation.
                Defaults to False.
            save_plot (bool): If True, save the power curve plot as a PNG alongside
                the YAML output. Requires ``output_path``. Defaults to False.
            validate_file (bool): If True, validate the saved YAML against the
                awesIO schema. Defaults to False.

        Returns:
            dict: Power curve data in awesIO format.
        """
        if profile_ids is None:
            profile_ids = list(range(1, self.n_profiles + 1))
        elif not isinstance(profile_ids, (list, tuple)):
            profile_ids = [profile_ids]

        if wind_speeds is not None:
            wind_speeds = np.array(wind_speeds)
            if verbose:
                print(f"Using custom wind speeds: {wind_speeds}")
        else:
            wind_speeds = self._generate_wind_speed_array(
                settings_key='optimization',
                profile_id=profile_ids[0],
                verbose=verbose,
            )

        wind_speed_data_per_profile = []
        for profile_id in profile_ids:
            wind_speed_data = self._calculate_power_curve_optimized(
                profile_id, wind_speeds, verbose,
            )
            wind_speed_data_per_profile.append(wind_speed_data)

        output = self._build_and_save_output(
            profile_ids=profile_ids,
            wind_speed_data_per_profile=wind_speed_data_per_profile,
            method_name='Optimization',
            output_path=output_path,
            verbose=verbose,
            validate_file=validate_file,
        )

        if (show_plot or save_plot) and output_path is not None:
            fig_path = Path(output_path).with_suffix('.pdf') if save_plot else None
            plotting.plot_power_curve(output_path, output_path=fig_path, show_plot=show_plot)

        return output

    def _prepare_warm_start(self, x_opt, var_names):
        """Build an updated x0 list from the previous optimal solution for a warm start.

        Only tether length fractions are propagated.  They change smoothly
        across wind speeds, making them safe to warm-start.  Reeling speeds and
        elevation angles are left at their YAML defaults: reeling speeds are
        handled by ``_adapt_x0``; elevation angles warm-started from the
        previous (lower) wind speed tend to land near infeasibility boundaries
        at the next wind speed, creating FD gradient cliffs.

        Args:
            x_opt (np.ndarray): Unscaled optimal variable values from the previous
                optimization.
            var_names (list): Variable names matching x_opt, as returned by
                ``CycleOptimizer.last_var_names``.

        Returns:
            list: Updated x0 list in the same index order as
                ``simulation_settings['optimization']['optimizer']['x0']``.
        """
        x0 = list(self.simulation_settings['optimization']['optimizer']['x0'])
        val_by_name = dict(zip(var_names, x_opt))

        # Propagate tether fractions and reel-in speed — they vary smoothly with
        # wind speed.  Reel-in warm-starting prevents SLSQP from landing on a
        # high-speed local minimum (e.g. -10 m/s) when the true optimum is near
        # -6 m/s.  Reel-out is left to _adapt_x0 (Betz-limit cap).
        name_to_idx = {'frac_end': 2, 'frac_start': 3, 'reeling_speed_in': 1}
        for name, idx in name_to_idx.items():
            if name in val_by_name and idx < len(x0):
                x0[idx] = float(val_by_name[name])

        return x0

    def _calculate_power_curve_optimized(self, profile_id, wind_speeds, verbose):
        """Calculate power curve for a single wind profile using optimization.

        Args:
            profile_id (int): The profile ID (1-indexed).
            wind_speeds (array-like): Wind speeds to evaluate [m/s].
            verbose (bool): Whether to print progress messages.

        Returns:
            list: Wind speed data entries for this profile.
        """
        env_state = self.create_environment(profile_id)

        if verbose:
            print(f"Optimizing power curve for profile {profile_id}...")

        # Save original x0 so it can be restored after the loop.
        original_x0 = list(self.simulation_settings['optimization']['optimizer']['x0'])

        wind_speed_data = []
        for i, ws in enumerate(wind_speeds):
            if verbose:
                print(f"  Wind speed {ws:.1f} m/s ({i+1}/{len(wind_speeds)})")
            entry = self._run_single_simulation_optimized(ws, env_state, verbose=verbose)
            wind_speed_data.append(entry)

            sim_successful = entry.get('successful', False)

            # Warm start: only seed x0 from the previous optimal if that run succeeded.
            if sim_successful:
                x_opt = getattr(self, 'last_x_opt', None)
                var_names = getattr(self, 'last_optimization_var_names', None)
                if x_opt is not None and var_names is not None:
                    warm_x0 = self._prepare_warm_start(x_opt, var_names)
                    self.simulation_settings['optimization']['optimizer']['x0'] = warm_x0
            else:
                # Reset to original x0 so the next wind speed starts cold.
                self.simulation_settings['optimization']['optimizer']['x0'] = list(original_x0)

        # Restore original x0 so subsequent calls are unaffected.
        self.simulation_settings['optimization']['optimizer']['x0'] = original_x0

        return wind_speed_data

    def _calculate_power_curve_direct(self, profile_id, wind_speeds, verbose):
        """Calculate power curve for a single wind profile using direct simulation.

        Args:
            profile_id (int): The profile ID (1-indexed).
            wind_speeds (array-like): Wind speeds to evaluate [m/s].
            verbose (bool): Whether to print progress messages.

        Returns:
            list: Wind speed data entries for this profile.
        """
        env_state = self.create_environment(profile_id)

        if verbose:
            print(f"Calculating power curve for profile {profile_id}...")

        wind_speed_data = []
        for i, ws in enumerate(wind_speeds):
            if verbose:
                print(f"  Wind speed {ws:.1f} m/s ({i+1}/{len(wind_speeds)})")
            wind_speed_data.append(self._run_single_simulation_direct(ws, env_state))

        return wind_speed_data

    def _run_single_simulation_direct(self, wind_speed, env_state):
        """Run a single cycle simulation for one wind speed.

        Args:
            wind_speed (float): Reference wind speed [m/s].
            env_state (NormalisedWindTable1D): Environment state object.

        Returns:
            dict: Wind speed entry with performance data and time history.
        """
        env_state.set_reference_wind_speed(wind_speed)

        settings = self.simulation_settings.copy()
        settings['cycle'] = self.simulation_settings['cycle'].copy()
        settings['cycle']['traction_phase'] = TractionPhase

        cycle = Cycle(settings, impose_operational_limits=True)

        steady_state_config = self.simulation_settings.get('steady_state')
        try:
            error_in_phase, _ = cycle.run_simulation(
                self.sys_props, env_state, steady_state_config, print_summary=False, enable_limit_violation_error=True
            )

            traction = getattr(cycle, 'traction_phase')
            retraction = getattr(cycle, 'retraction_phase')
            transition_riro = getattr(cycle, 'transition_riro_phase')
            transition_rori = getattr(cycle, 'transition_rori_phase')

            # Check minimum altitude constraint.
            minimum_height = self.simulation_settings.get('cycle', {}).get('minimum_height', 0.0)
            if minimum_height > 0 and traction.kinematics:
                min_altitude = min(k.z for k in traction.kinematics)
                if min_altitude < minimum_height:
                    raise ValueError(
                        f"Minimum altitude violated during traction phase: "
                        f"{min_altitude:.1f} m < {minimum_height:.1f} m. "
                        f"Adjust elevation_angle_traction or tether length settings."
                    )

            kpi = {
                'sim_successful': error_in_phase is None,
                'average_power': {
                    'cycle': cycle.average_power,
                    'in': retraction.average_power if retraction else 0.0,
                    'trans_riro': transition_riro.average_power if transition_riro else 0.0,
                    'trans_rori': transition_rori.average_power if transition_rori else 0.0,
                    'out': traction.average_power if traction else 0.0,
                },
                'duration': {
                    'cycle': cycle.duration,
                    'in': retraction.duration if retraction else 0.0,
                    'trans_riro': transition_riro.duration if transition_riro else 0.0,
                    'trans_rori': transition_rori.duration if transition_rori else 0.0,
                    'out': traction.duration if traction else 0.0,
                },
                'time': cycle.time,
                'kinematics': cycle.kinematics,
                'steady_states': cycle.steady_states,
            }

            return self._build_wind_speed_entry(wind_speed, kpi)

        except Exception as e:
            print(f"    ERROR at {wind_speed:.1f} m/s: {type(e).__name__}: {e}")
            kpi = {
                'sim_successful': False,
                'average_power': {
                    'cycle': 0.0,
                    'in': 0.0,
                    'trans_riro': 0.0,
                    'trans_rori': 0.0,
                    'out': 0.0,
                },
                'duration': {
                    'cycle': 0.0,
                    'in': 0.0,
                    'trans_riro': 0.0,
                    'trans_rori': 0.0,
                    'out': 0.0,
                },
            }
            return self._build_wind_speed_entry(wind_speed, kpi)

    def _run_single_simulation_optimized(self, wind_speed, env_state, show_plot=False,
                                          verbose=False):
        """Run a single optimized cycle simulation for one wind speed.

        Args:
            wind_speed (float): Reference wind speed [m/s].
            env_state (NormalisedWindTable1D): Environment state object.
            show_plot (bool): If True, show optimization evolution and cycle plots.
            verbose (bool): If True, print per-iteration progress. Defaults to False.

        Returns:
            dict: Wind speed entry with performance data and optional time history.
        """
        optimizer = CycleOptimizer(self.simulation_settings, self.sys_props, env_state)
        kpi = optimizer.optimize(wind_speed, verbose=verbose)

        optResult = kpi.pop('optimization_result', None)
        if optResult is not None:
            print(f"    Optimizer status: {optResult.message}  "
                  f"(nit={optResult.nit}, nfev={optResult.nfev})")
            print(f"    Optimal x: {optimizer.last_x_opt}")
            print(f"    Cycle power: {kpi['average_power']['cycle']:.1f} W")

        self.last_optimization_history = optimizer.history
        self.last_optimization_var_names = getattr(optimizer, 'last_var_names', None)
        self.last_x_opt = getattr(optimizer, 'last_x_opt', None)

        sim_ok = kpi.get('sim_successful', False)
        if not sim_ok:
            print(f"    Wind speed {wind_speed:.1f} m/s: simulation was not successful.")

        return self._build_wind_speed_entry(wind_speed, kpi)


    def _build_and_save_output(
        self,
        profile_ids,
        wind_speed_data_per_profile,
        method_name,
        output_path=None,
        verbose=True,
        validate_file=False,):
        """Build complete power curve output with metadata and optionally save.

        Constructs per-profile power curve dicts with wind profile metadata,
        wraps them in a full output structure, and optionally saves to YAML.

        Args:
            profile_ids (list): List of profile IDs (0-indexed).
            wind_speed_data_per_profile (list): List of wind speed data lists,
                one per profile. Each inner list contains dicts with
                wind_speed_m_s, success, performance, and optionally time_history.
            method_name (str): Name of the method (e.g., 'Optimization-Based').
            output_path (str, optional): Path to save the output YAML file.
            verbose (bool): Whether to print progress.
            validate_file (bool): If True, validate the saved YAML against the
                awesIO schema after saving. Defaults to False.

        Returns:
            dict: Complete power curve data in awesIO format.
        """
        # Extract wind speeds from first profile's data
        wind_speeds = [entry['wind_speed'] for entry in wind_speed_data_per_profile[0]]
        
        # Check if there are any successful wind speeds
        if len(wind_speeds) == 0:
            raise ValueError(
                "No successful simulations for any wind speed. "
                "Cannot generate power curve output. "
                "Try adjusting the cut_in wind speed or optimizer settings in the configuration file."
            )

        # Build per-profile power curve dicts
        power_curves = []
        for profile_id, wind_speed_data in zip(profile_ids, wind_speed_data_per_profile):
            profiles = self.wind_resource.get('profiles', [])
            profile_data = next((c for c in profiles if c.get('id') == profile_id), None)
            if profile_data is None:
                raise ValueError(
                    f"profile ID {profile_id} not found in wind resource data."
                )
            v_normalized = profile_data.get(
                'v_normalized',
                [0.0] * len(profile_data.get('u_normalized', [])),
            )
            power_curves.append({
                'profile_id': profile_id,
                'probability_weight': float(profile_data.get('frequency', 1.0 / self.n_profiles)),
                'wind_profile': {
                    'u_normalized': [float(u) for u in profile_data.get('u_normalized', [])],
                    'v_normalized': [float(v) for v in v_normalized],
                },
                'wind_speed_data': wind_speed_data,
            })

        # Determine cut-in and cut-out from first profile
        cycle_powers = [entry['performance']['power']['average_cycle_power']
                        for entry in wind_speed_data_per_profile[0]]
        cut_in_ws = self._find_cut_in_wind_speed(wind_speeds, cycle_powers)
        if cut_in_ws is None:
            cut_in_ws = wind_speeds[0]  # Fallback to first wind speed
        cut_out_ws = wind_speeds[-1]

        # Calculate nominal power (max across all profiles)
        nominal_power = max(
            max(entry['performance']['power']['average_cycle_power'] for entry in wsd)
            for wsd in wind_speed_data_per_profile
        )

        # Prepare output in awesIO power_curves_schema format
        altitudes = self.wind_resource.get('altitudes', [])
        output = {
            'metadata': {
                'name': f'Ground-Gen Power Curves ({method_name})',
                'description': 'Power curves for pumping ground-gen AWE system',
                'note': f'Power curve data generated from QSM {method_name.lower()}. Wind speeds at {self.reference_height:.0f}m reference height.',
                'awesIO_version': '0.1.0',
                'schema': 'power_curves_schema.yml',
                'time_created': datetime.now().isoformat(),
                'model_config': {
                    'wing_area': float(self.sys_props.kite_projected_area),
                    'wing_mass': float(self.sys_props.kite_mass),
                },
                'wind_resource': {
                    'n_profiles': self.n_profiles,
                    'n_profiles_calculated': len(profile_ids),
                    'profile_ids_calculated': list(profile_ids),
                    'reference_height': float(self.reference_height),
                },
            },
            'altitudes': [float(a) for a in altitudes],
            'reference_wind_speeds': [float(ws) for ws in wind_speeds],
            'power_curves': power_curves,
        }

        # Save to file if path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # --- Extract time histories into a binary .npz sidecar ----------
            # Keys: p{profile_id}_ws{index}_{channel}
            # The inline time_history dicts are stripped from the YAML output
            # to avoid serialising thousands of floats as text.
            TIME_HISTORY_CHANNELS = (
                'time', 'altitude', 'tether_force', 'power',
                'reel_speed', 'tether_length', 'elevation_angle', 'wind_speed',
                'kite_wind_speed', 'kite_tangential_speed', 'kite_apparent_wind_speed',
            )
            npz_arrays = {}
            for pc in output['power_curves']:
                pid = pc['profile_id']
                for ws_idx, entry in enumerate(pc['wind_speed_data']):
                    th = entry.pop('time_history', None)
                    if th:
                        for ch in TIME_HISTORY_CHANNELS:
                            if ch in th:
                                npz_arrays[f'p{pid}_ws{ws_idx}_{ch}'] = np.array(th[ch])

            if npz_arrays:
                npz_path = output_path.with_suffix('.npz')
                np.savez_compressed(npz_path, **npz_arrays)
                output['time_history'] = {
                    'channels': list(TIME_HISTORY_CHANNELS),
                    'filename': npz_path.name,
                }
                if verbose:
                    print(f"Time histories saved to {npz_path}")

            # -----------------------------------------------------------------

            class IndentedDumper(yaml.Dumper):
                def increase_indent(self, flow=False, indentless=False):
                    return super(IndentedDumper, self).increase_indent(flow, indentless=False)

            with open(output_path, 'w') as f:
                yaml.dump(output, f, Dumper=IndentedDumper,
                          default_flow_style=False, sort_keys=False,
                          indent=2, width=1000)

            if verbose:
                print(f"Power curves saved to {output_path}")

            if validate_file:
                try:
                    from awesio.validator import validate as awesio_validate
                    print(f"\nValidating {output_path} against awesIO schema...")
                    awesio_validate(str(output_path))
                    print("  Validation passed.")
                except ImportError:
                    print("  awesIO not installed; skipping validation.")
                except Exception as e:
                    print(f"  Validation failed: {e}")

        return output

    def print_summary(self):
        """Print a summary of the system configuration."""
        print("System Configuration:")
        print(f"  Kite projected area: {self.sys_props.kite_projected_area:.1f} m²")
        print(f"  Kite mass: {self.sys_props.kite_mass:.1f} kg")
        print(f"  Max tether force: {self.sys_props.tether_force_max_limit:.0f} N")
        print(f"  Min tether force: {self.sys_props.tether_force_min_limit:.0f} N")
        print(f"  Tether diameter: {self.sys_props.tether_diameter * 1000:.1f} mm")
        print(f"  Max reeling speed: {self.sys_props.reeling_speed_max_limit:.1f} m/s")
        print("\nOperating Parameters:")
        print(f"  Reference height: {self.reference_height:.0f} m")
        print(f"  Number of wind profiles: {self.n_profiles}")

    def _build_wind_speed_entry(self, wind_speed, kpi):
        """Build a wind speed result entry dict from a KPI dict.

        Args:
            wind_speed (float): Reference wind speed [m/s].
            kpi (dict): Performance KPI dict from the optimizer or cycle.

        Returns:
            dict: Wind speed entry with performance data and optional time history.
        """
        def _safe_float(val):
            if val is None:
                return float('nan')
            return float(val)

        time_history = None
        sim_successful = kpi.get('sim_successful', False)
        if (sim_successful
                and 'kinematics' in kpi and kpi['kinematics']
                and 'steady_states' in kpi and kpi['steady_states']):
            if 'time' in kpi and kpi['time']:
                time_list = list(kpi['time'])
            else:
                n_pts = len(kpi['kinematics'])
                time_list = list(np.linspace(0, kpi['duration']['cycle'], n_pts))
            time_history = self._extract_time_history(
                time_list, kpi['kinematics'], kpi['steady_states'],
                wind_speed=float(wind_speed),
            )

        entry = {
            'wind_speed': float(wind_speed),
            'successful': bool(kpi.get('sim_successful', False)),
            'performance': {
                'power': {
                    'average_cycle_power': _safe_float(kpi['average_power']['cycle']),
                    'average_reel_out_power': _safe_float(kpi['average_power']['out']),
                    'average_reel_in_power': _safe_float(kpi['average_power']['in']),
                    'average_transition_rori_power': _safe_float(
                        kpi['average_power'].get('trans_rori', 0.0)
                    ),
                    'average_transition_riro_power': _safe_float(
                        kpi['average_power'].get('trans_riro',
                            kpi['average_power'].get('trans', 0.0))
                    ),
                },
                'timing': {
                    'reel_out_time': _safe_float(kpi['duration']['out']),
                    'transition_rori_time': _safe_float(
                        kpi['duration'].get('trans_rori', 0.0)
                    ),
                    'reel_in_time': _safe_float(kpi['duration']['in']),
                    'transition_riro_time': _safe_float(
                        kpi['duration'].get('trans_riro',
                            kpi['duration'].get('trans', 0.0))
                    ),
                    'cycle_time': _safe_float(kpi['duration']['cycle']),
                },
            },
        }
        if time_history is not None:
            entry['time_history'] = time_history

        return entry

    def _extract_time_history(self, time_list, kinematics, steady_states,
                               wind_speed=None):
        """Extract time history data from kinematics and steady state objects.

        This unified method is used by both direct simulation and optimization.
        Both paths provide lists of kinematics and steady-state objects with the
        same attributes.

        Args:
            time_list (list): Time values [s].
            kinematics (list): Kinematics objects with z, straight_tether_length,
                elevation_angle attributes.
            steady_states (list): Steady state objects with tether_force_ground,
                power_ground, reeling_speed, wind_speed attributes.
            wind_speed (float, optional): Reference wind speed [m/s]. Used as
                fallback when the steady-state wind speed is unavailable.

        Returns:
            dict: Time history data with altitude, forces, power, speeds, etc.
                Returns None if inputs are empty.
        """
        if not time_list or not kinematics or not steady_states:
            return None

        time_full = [float(t) for t in time_list]
        altitude_full = []
        tether_force_full = []
        power_full = []
        reel_speed_full = []
        tether_length_full = []
        elevation_angle_full = []
        wind_speed_full = []
        kite_wind_speed_full = []
        kite_tangential_speed_full = []
        kite_apparent_wind_speed_full = []

        for kin, ss in zip(kinematics, steady_states):
            altitude_full.append(float(kin.z))
            tether_force_full.append(float(ss.tether_force_ground))
            power_full.append(float(ss.power_ground))
            reel_speed_full.append(float(ss.reeling_speed))
            tether_length_full.append(float(kin.straight_tether_length))
            elevation_angle_full.append(float(kin.elevation_angle))
            ws_val = ss.wind_speed if ss.wind_speed is not None else wind_speed
            wind_speed_full.append(float(ws_val) if ws_val is not None else float('nan'))
            kws = getattr(ss, 'wind_speed', None)
            kite_wind_speed_full.append(float(kws) if kws is not None else float('nan'))
            kts = getattr(ss, 'kite_tangential_speed', None)
            kite_tangential_speed_full.append(float(kts) if kts is not None else float('nan'))
            aws = getattr(ss, 'apparent_wind_speed', None)
            kite_apparent_wind_speed_full.append(float(aws) if aws is not None else float('nan'))

        return {
            'time': time_full,
            'altitude': altitude_full,
            'tether_force': tether_force_full,
            'power': power_full,
            'reel_speed': reel_speed_full,
            'tether_length': tether_length_full,
            'elevation_angle': elevation_angle_full,
            'wind_speed': wind_speed_full,
            'kite_wind_speed': kite_wind_speed_full,
            'kite_tangential_speed': kite_tangential_speed_full,
            'kite_apparent_wind_speed': kite_apparent_wind_speed_full,
        }

    @staticmethod
    def _find_cut_in_wind_speed(wind_speeds, cycle_powers):
        """Find the cut-in wind speed where average cycle power first becomes positive.

        Args:
            wind_speeds (list): Wind speeds [m/s].
            cycle_powers (list): Average cycle power values [W].

        Returns:
            float or None: Cut-in wind speed, or None if no positive power found.
        """
        for ws, p in zip(wind_speeds, cycle_powers):
            if p is not None and p > 0:
                return ws
        return None




