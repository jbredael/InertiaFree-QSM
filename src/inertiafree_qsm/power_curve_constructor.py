# -*- coding: utf-8 -*-
"""Power curve generation for AWE systems.

This module provides functionality to generate power curves using either direct
simulation or sequential optimization with warm starts. It includes constraint
handling and detailed diagnostics.

All wind speeds are referenced at the height specified in the wind_resource.yml
metadata (reference_height_m). The model uses the wind profile to derive wind
speeds at operating altitudes.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

from .config_loader import (
    load_system_config,
    load_wind_resource,
    load_simulation_settings,
)
from .qsm import (
    Cycle,
    KiteKinematics,
    NormalisedWindTable1D,
    OperationalLimitViolation,
    PhaseError,
    SteadyState,
    SteadyStateError,
    SystemProperties,
    TractionConstantElevation,
    TractionPhase,
    TractionPhaseHybrid,
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
        validate_file=True,):
        """Initialize the power curve optimizer with configuration files."""
        self.system_config_path = Path(system_config_path)
        self.wind_resource_path = Path(wind_resource_path)
        self.simulation_settings_path = Path(simulation_settings_path)

        # Load configurations (validation is internal to loader functions)
        self.sys_props_dict = load_system_config(self.system_config_path, validate_file=validate_file)
        self.wind_resource = load_wind_resource(self.wind_resource_path, validate_file=validate_file)
        self.simulation_settings = load_simulation_settings(
            self.simulation_settings_path, self.sys_props_dict, verbose=True
        )

        # Create system properties object
        self.sys_props = SystemProperties(self.sys_props_dict)

        # Reference height from wind resource metadata
        self.reference_height = self.wind_resource.get('metadata', {}).get(
            'reference_height_m'
        )

        # Number of clusters
        self.n_clusters = self.wind_resource.get('metadata', {}).get('n_clusters')

        self.tether_length_start = self.simulation_settings['cycle'][
            'tether_length_start_retraction']
        self.tether_length_end = self.simulation_settings['cycle'][
            'tether_length_end_retraction']
        
        # Get traction settings
        self.phi_traction = self.simulation_settings['traction']['azimuth_angle']
        self.chi_traction = self.simulation_settings['traction']['course_angle']

        # Initialize result storage
        self.wind_speeds = None
        self.x_opts = []
        self.x0_list = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

    def create_environment(self, cluster_id=1):
        """Create an environment state object for a specific wind cluster.

        Args:
            cluster_id (int): The cluster ID (1-indexed).

        Returns:
            NormalisedWindTable1D: Environment state with wind profile.
        """

        clusters = self.wind_resource['clusters']

        if not clusters:
            raise ValueError("No clusters found in wind resource data")

        # Find the cluster with matching ID (clusters use 1-indexed IDs)
        cluster = None
        for c in clusters:
            if c.get('id') == cluster_id:
                cluster = c
                break

        if cluster is None:
            if cluster_id >= len(clusters):
                raise ValueError(
                    f"Cluster ID {cluster_id} not found in wind resource data"
                )
            cluster = clusters[cluster_id]

        env_state = NormalisedWindTable1D()
        env_state.h_ref = self.wind_resource.get('metadata', {}).get('reference_height_m')
        env_state.heights = list(self.wind_resource['altitudes'])
        env_state.normalised_wind_speeds = list(np.array(cluster.get('u_normalized', [])))
        env_state.set_reference_height(env_state.h_ref)

        return env_state

    def _generate_wind_speed_array(self, settings_key='optimization', cluster_id=None, verbose=False):
        """Generate wind speed array from configuration settings.

        Args:
            settings_key (str): Key in simulation_settings to use ('optimization' or 'direct_simulation').
            cluster_id (int, optional): Not used (kept for compatibility).
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


    def run_predefined_sequence(self, optimization_sequence, x0_start, wind_speeds,
                                save_plot_dir=None):
        """Run sequential optimizations with warm starts.

        Args:
            optimization_sequence (dict): Dict mapping wind speed thresholds to
                optimizer configurations.
            x0_start (array): Initial starting point.
            wind_speeds (array): Wind speeds to evaluate [m/s].
            save_plot_dir (Path or str, optional): Directory to save per-wind-speed
                optimization evolution PDFs.
        """
        self.wind_speeds = []

        wind_speed_thresholds = iter(sorted(list(optimization_sequence)))
        vw_switch = next(wind_speed_thresholds)

        x_opt_last, vw_last = None, None
        for i, vw in enumerate(wind_speeds):
            if vw > vw_switch:
                try:
                    vw_switch = next(wind_speed_thresholds)
                except StopIteration:
                    pass

            power_optimizer = optimization_sequence[vw_switch]['power_optimizer']
            dx0 = optimization_sequence[vw_switch].get('dx0', None)

            if x_opt_last is None:
                x0_next = x0_start
            else:
                x0_next = x_opt_last + dx0 * (vw - vw_last)

            print(f"[{i}] Processing v={vw:.2f} m/s")
            optimization_succeeded = False
            try:
                x_opt, sim_successful = self.run_optimization(
                    vw, power_optimizer, x0_next, save_plot_dir=save_plot_dir
                )
                optimization_succeeded = True
                final_vw = vw
            except (OperationalLimitViolation, SteadyStateError, PhaseError):
                try:  # Retry for slightly different wind speed
                    x_opt, sim_successful = self.run_optimization(
                        vw + 1e-2, power_optimizer, x0_next, save_plot_dir=save_plot_dir
                    )
                    optimization_succeeded = True
                    final_vw = vw + 1e-2
                except (OperationalLimitViolation, SteadyStateError, PhaseError):
                    print(f"  Failed at {vw:.2f} m/s, skipping this wind speed...")
                    optimization_succeeded = False

            if optimization_succeeded:
                self.wind_speeds.append(final_vw)
                if sim_successful:
                    x_opt_last = x_opt
                    vw_last = final_vw
        
        # Convert to numpy array
        self.wind_speeds = np.array(self.wind_speeds)
        
        if len(self.wind_speeds) == 0:
            print(
                f"Optimization failed at all wind speeds. "
                f"Try lowering the cut_in wind speed in the configuration file."
            )
   
    def run_optimization(self, wind_speed, power_optimizer, x0, show_plot=False,
                         save_plot_dir=None):
        """Run single optimization for given wind speed.

        Args:
            wind_speed (float): Wind speed at reference height [m/s].
            power_optimizer (CycleOptimizer): Optimizer instance.
            x0 (array): Starting point for optimization.
            show_plot (bool): If True, show the cycle trajectory/time plots at
                the optimal point.
            save_plot_dir (Path or str, optional): Directory in which to save the
                optimization evolution PDF after each wind speed.

        Returns:
            tuple: (x_opt, sim_successful)
        """
        power_optimizer.environment_state.set_reference_wind_speed(wind_speed)

        power_optimizer.x0_real_scale = x0
        x_opt = power_optimizer.optimize()

        self.x0_list.append(x0)
        self.x_opts.append(x_opt)
        self.optimization_details.append(power_optimizer.op_res)

        # Always evaluate with relaxed errors for consistency: the optimizer
        # itself uses relax_errors=True throughout, so the optimal point is
        # only guaranteed to be valid under those same relaxed conditions.
        # sim_successful reflects whether the optimizer converged, not whether
        # the re-evaluation passes strict steady-state checks.
        cons, kpis = power_optimizer.eval_point(relax_errors=True, plot_result=show_plot)
        sim_successful = bool(power_optimizer.op_res.get('success', False))

        if save_plot_dir is not None or show_plot:
            opt_evo_path = None
            if save_plot_dir is not None:
                opt_evo_path = (
                    Path(save_plot_dir) / f'opt_evolution_v{wind_speed:.1f}ms.pdf'
                )
            power_optimizer.plot_opt_evolution(output_path=opt_evo_path, show_plot=show_plot)

        self.constraints.append(cons)
        kpis['sim_successful'] = sim_successful
        self.performance_indicators.append(kpis)

        return x_opt, sim_successful


    def simulate_single_wind_speed(self, wind_speed, cluster_id=1, method='direct',
                                    output_path=None, verbose=False,
                                    show_plot=False, save_plot=False,
                                    validate_file=False):
        """Simulate a single cycle at one wind speed.

        Produces a full power-curve YAML file (with a single wind speed entry)
        via ``_build_and_save_output`` and optionally plots the cycle detail.

        Args:
            wind_speed (float): Reference wind speed at reference height [m/s].
            cluster_id (int): Cluster ID (1-indexed).
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
        env_state = self.create_environment(cluster_id)

        if method == 'direct':
            entry = self._run_single_simulation_direct(wind_speed, env_state)
        elif method == 'optimization':
            entry = self._run_single_simulation_optimized(wind_speed, env_state, show_plot=show_plot)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'direct' or 'optimization'."
            )

        output = self._build_and_save_output(
            cluster_ids=[cluster_id],
            wind_speed_data_per_cluster=[[entry]],
            method_name=method.replace('_', ' ').title(),
            output_path=output_path,
            verbose=output_path is not None,
            validate_file=validate_file,
        )

        if (show_plot or save_plot) and output_path is not None:
            fig_path = Path(output_path).with_suffix('.pdf') if save_plot else None
            plotting.plot_cycle_detail(
                output_path, wind_speed, profile_id=cluster_id,
                output_path=fig_path, show_plot=show_plot,
            )

        return output

    def generate_power_curves_direct(
        self,
        wind_speeds=None,
        cluster_ids=None,
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
            cluster_ids (list, optional): List of cluster IDs (1-indexed) to calculate.
                If None, calculates all clusters/profiles.
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
        # Determine which clusters to process
        if cluster_ids is None:
            cluster_ids = list(range(1, self.n_clusters + 1))
        elif not isinstance(cluster_ids, (list, tuple)):
            cluster_ids = [cluster_ids]

        # Determine wind speeds once, estimating from the first cluster if needed
        if wind_speeds is not None:
            wind_speeds = np.array(wind_speeds)
            if verbose:
                print(f"Using custom wind speeds: {wind_speeds}")
        else:
            wind_speeds = self._generate_wind_speed_array(
                settings_key='direct_simulation',
                cluster_id=cluster_ids[0],
                verbose=verbose
            )

        # Calculate power curves for selected clusters
        wind_speed_data_per_cluster = []
        for cluster_id in cluster_ids:
            wind_speed_data = self._calculate_power_curve_direct(
                cluster_id, wind_speeds, verbose
            )
            wind_speed_data_per_cluster.append(wind_speed_data)

        # Build full output with metadata and save
        output = self._build_and_save_output(
            cluster_ids=cluster_ids,
            wind_speed_data_per_cluster=wind_speed_data_per_cluster,
            method_name='Direct Simulation',
            output_path=output_path,
            verbose=verbose,
            validate_file=validate_file,
        )

        if (show_plot or save_plot) and output_path is not None:
            fig_path = Path(output_path).with_suffix('.pdf') if save_plot else None
            plotting.plot_power_curve(output_path, output_path=fig_path, show_plot=show_plot)

        return output

    def _calculate_power_curve_direct(self, cluster_id, wind_speeds, verbose):
        """Calculate power curve for a single wind cluster using direct simulation.

        Args:
            cluster_id (int): The cluster ID (1-indexed).
            wind_speeds (array-like): Wind speeds to evaluate [m/s].
            verbose (bool): Whether to print progress messages.

        Returns:
            list: Wind speed data entries for this cluster.
        """
        env_state = self.create_environment(cluster_id)

        if verbose:
            print(f"Calculating power curve for cluster {cluster_id}...")

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

        steady_state_config = self.simulation_settings.get('steady_state', {})
        try:
            error_in_phase, average_power = cycle.run_simulation(
                self.sys_props, env_state, steady_state_config, print_summary=False
            )

            traction = getattr(cycle, 'traction_phase', None)
            retraction = getattr(cycle, 'retraction_phase', None)
            transition = getattr(cycle, 'transition_phase', None)

            reel_out_time = traction.duration if traction else 0.0
            reel_in_time = ((retraction.duration if retraction else 0.0)
                           + (transition.duration if transition else 0.0))
            reel_out_power = (traction.energy / traction.duration
                             if traction and traction.duration > 0 else 0.0)
            reel_in_power = (retraction.energy / retraction.duration
                            if retraction and retraction.duration > 0 else 0.0)

            time_history = self._extract_time_history(
                cycle.time, cycle.kinematics, cycle.steady_states
            )

            entry = {
                'wind_speed_m_s': float(wind_speed),
                'success': error_in_phase is None,
                'performance': {
                    'power': {
                        'average_cycle_power_w': float(average_power),
                        'average_reel_out_power_w': float(reel_out_power),
                        'average_reel_in_power_w': float(reel_in_power),
                    },
                    'timing': {
                        'reel_out_time_s': float(reel_out_time),
                        'reel_in_time_s': float(reel_in_time),
                        'cycle_time_s': float(cycle.duration),
                    },
                },
            }
            if time_history is not None:
                entry['time_history'] = time_history
            return entry

        except Exception as e:
            print(f"    ERROR at {wind_speed:.1f} m/s: {type(e).__name__}: {e}")
            return {
                'wind_speed_m_s': float(wind_speed),
                'success': False,
                'performance': {
                    'power': {
                        'average_cycle_power_w': 0.0,
                        'average_reel_out_power_w': 0.0,
                        'average_reel_in_power_w': 0.0,
                    },
                    'timing': {
                        'reel_out_time_s': 0.0,
                        'reel_in_time_s': 0.0,
                        'cycle_time_s': 0.0,
                    },
                },
            }

    def generate_power_curves_optimized(
        self,
        wind_speeds=None,
        cluster_ids=None,
        output_path=None,
        verbose=True,
        show_plot=False,
        save_plot=False,
        validate_file=False,
        opt_plots_dir=None,):
        """Generate optimized power curves for one or more wind profile clusters.

        This method performs sequential optimization to find optimal cycle parameters
        at each wind speed. The elevation angle from simulation settings is NOT used
        as a constraint - it is only used to calculate the operating altitude.

        Args:
            wind_speeds (array-like, optional): Wind speeds to evaluate [m/s]. If None,
                defaults to range defined in simulation settings.
            cluster_ids (list, optional): List of cluster IDs (1-indexed) to calculate.
                If None, calculates all clusters.
            output_path (str, optional): Path to save the output YAML file.
            verbose (bool): Whether to print progress.
            show_plot (bool): If True, display the power curve plot after generation.
                Defaults to False.
            save_plot (bool): If True, save the power curve plot as a PDF alongside
                the YAML output. Requires ``output_path``. Defaults to False.
            validate_file (bool): If True, validate the saved YAML against the
                awesIO schema. Defaults to False.
            opt_plots_dir (str or Path, optional): Directory to save per-wind-speed
                optimization evolution PDFs. If None, no plots are saved.

        Returns:
            dict: Power curve data in awesIO format.
        """
        # Determine which clusters to process
        if cluster_ids is None:
            cluster_ids = list(range(1, self.n_clusters + 1))
        elif not isinstance(cluster_ids, (list, tuple)):
            cluster_ids = [cluster_ids]

        # Determine wind speeds once, estimating from the first cluster if needed
        if wind_speeds is not None:
            wind_speeds = np.array(wind_speeds)
        else:
            wind_speeds = self._generate_wind_speed_array(
                settings_key='optimization',
                cluster_id=cluster_ids[0],
                verbose=verbose
            )

        # Calculate optimized power curves for each cluster
        wind_speed_data_per_cluster = []
        for cluster_id in cluster_ids:
            wind_speed_data = self._calculate_power_curve_optimized(
                cluster_id, wind_speeds, verbose, save_plot_dir=opt_plots_dir
            )
            wind_speed_data_per_cluster.append(wind_speed_data)

        # Build full output with metadata and save
        output = self._build_and_save_output(
            cluster_ids=cluster_ids,
            wind_speed_data_per_cluster=wind_speed_data_per_cluster,
            method_name='Optimization-Based',
            output_path=output_path,
            verbose=verbose,
            validate_file=validate_file,
        )

        if (show_plot or save_plot) and output_path is not None:
            fig_path = Path(output_path).with_suffix('.pdf') if save_plot else None
            plotting.plot_power_curve(output_path, output_path=fig_path, show_plot=show_plot)

        return output

    def _calculate_power_curve_optimized(self, cluster_id, wind_speeds, verbose,
                                          save_plot_dir=None):
        """Calculate optimized power curve for a single wind cluster.

        Args:
            cluster_id (int): The cluster ID (1-indexed).
            wind_speeds (array-like): Wind speeds to evaluate [m/s].
            verbose (bool): Whether to print progress messages.
            save_plot_dir (Path or str, optional): Directory to save per-wind-speed
                optimization evolution PDFs.

        Returns:
            list: Wind speed data entries for this cluster.
        """
        # Reset per-run optimization state so clusters don't bleed into each other
        self.x_opts = []
        self.x0_list = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

        env_state = self.create_environment(cluster_id)

        # Configure optimization sequence
        cycle_sim_settings_phase1 = deepcopy(self.simulation_settings)
        cycle_sim_settings_phase1['cycle']['traction_phase'] = TractionPhase

        cycle_sim_settings_phase2 = deepcopy(cycle_sim_settings_phase1)
        cycle_sim_settings_phase2['cycle']['traction_phase'] = TractionPhaseHybrid

        opt_settings = self.simulation_settings['optimization']
        opt_config = {
            'x0': opt_settings['optimizer']['x0'].copy(),
            'scaling': opt_settings['optimizer']['scaling'].copy(),
            'bounds': opt_settings['bounds'].copy(),
            'ftol': opt_settings['optimizer']['ftol'],
            'eps': opt_settings['optimizer']['eps'],
        }

        op_cycle_phase1 = CycleOptimizer(
            cycle_sim_settings_phase1,
            self.sys_props,
            env_state,
            optimizer_config=opt_config,
        )

        op_cycle_phase2 = CycleOptimizer(
            cycle_sim_settings_phase2,
            self.sys_props,
            env_state,
            optimizer_config=opt_config,
        )

        optimization_sequence = {
            7.0: {
                'power_optimizer': op_cycle_phase1,
                'dx0': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            17.0: {
                'power_optimizer': op_cycle_phase2,
                'dx0': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            np.inf: {
                'power_optimizer': op_cycle_phase2,
                'dx0': np.array([0.0, 0.0, 0.1, 0.0, 0.0]),
            },
        }

        x0_start = np.array(opt_settings['optimizer']['x0'].copy())

        if verbose:
            print(f"Running optimizations for cluster {cluster_id}...")

        self.run_predefined_sequence(
            optimization_sequence, x0_start, wind_speeds,
            save_plot_dir=save_plot_dir,
        )

        # Build wind speed data entries from optimization KPIs
        wind_speed_data = [
            self._build_wind_speed_entry(ws, kpi)
            for ws, kpi in zip(self.wind_speeds, self.performance_indicators)
        ]

        return wind_speed_data

    def _run_single_simulation_optimized(self, wind_speed, env_state, show_plot=False):
        """Run a single optimized cycle simulation for one wind speed.

        Args:
            wind_speed (float): Reference wind speed [m/s].
            env_state (NormalisedWindTable1D): Environment state object.
            show_plot (bool): If True, show optimization evolution and cycle plots.

        Returns:
            dict: Wind speed entry with performance data and optional time history.
        """
        # Reset per-run state to avoid cross-contamination.
        self.x_opts = []
        self.x0_list = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

        cycle_sim_settings = deepcopy(self.simulation_settings)
        cycle_sim_settings['cycle']['traction_phase'] = TractionPhaseHybrid

        opt_settings = self.simulation_settings['optimization']
        opt_config = {
            'x0': opt_settings['optimizer']['x0'].copy(),
            'scaling': opt_settings['optimizer']['scaling'].copy(),
            'bounds': opt_settings['bounds'].copy(),
            'ftol': opt_settings['optimizer']['ftol'],
            'eps': opt_settings['optimizer']['eps'],
        }

        power_optimizer = CycleOptimizer(
            cycle_sim_settings,
            self.sys_props,
            env_state,
            optimizer_config=opt_config,
        )

        x0 = np.array(opt_settings['optimizer']['x0'].copy())
        self.run_optimization(wind_speed, power_optimizer, x0, show_plot=show_plot)

        kpi = self.performance_indicators[-1]
        return self._build_wind_speed_entry(wind_speed, kpi)




    def _build_and_save_output(
        self,
        cluster_ids,
        wind_speed_data_per_cluster,
        method_name,
        output_path=None,
        verbose=True,
        validate_file=False,):
        """Build complete power curve output with metadata and optionally save.

        Constructs per-cluster power curve dicts with wind profile metadata,
        wraps them in a full output structure, and optionally saves to YAML.

        Args:
            cluster_ids (list): List of cluster IDs (0-indexed).
            wind_speed_data_per_cluster (list): List of wind speed data lists,
                one per cluster. Each inner list contains dicts with
                wind_speed_m_s, success, performance, and optionally time_history.
            method_name (str): Name of the method (e.g., 'Optimization-Based').
            output_path (str, optional): Path to save the output YAML file.
            verbose (bool): Whether to print progress.
            validate_file (bool): If True, validate the saved YAML against the
                awesIO schema after saving. Defaults to False.

        Returns:
            dict: Complete power curve data in awesIO format.
        """
        # Extract wind speeds from first cluster's data
        wind_speeds = [entry['wind_speed_m_s'] for entry in wind_speed_data_per_cluster[0]]
        
        # Check if there are any successful wind speeds
        if len(wind_speeds) == 0:
            raise ValueError(
                "No successful simulations for any wind speed. "
                "Cannot generate power curve output. "
                "Try adjusting the cut_in wind speed or optimizer settings in the configuration file."
            )

        # Build per-cluster power curve dicts
        power_curves = []
        for cluster_id, wind_speed_data in zip(cluster_ids, wind_speed_data_per_cluster):
            clusters = self.wind_resource.get('clusters', [])
            cluster_data = next((c for c in clusters if c.get('id') == cluster_id), None)
            if cluster_data is None:
                raise ValueError(
                    f"Cluster ID {cluster_id} not found in wind resource data."
                )
            v_normalized = cluster_data.get(
                'v_normalized',
                [0.0] * len(cluster_data.get('u_normalized', [])),
            )
            power_curves.append({
                'profile_id': cluster_id,
                'probability_weight': float(cluster_data.get('frequency', 1.0 / self.n_clusters)),
                'wind_profile': {
                    'u_normalized': [float(u) for u in cluster_data.get('u_normalized', [])],
                    'v_normalized': [float(v) for v in v_normalized],
                },
                'wind_speed_data': wind_speed_data,
            })

        # Determine cut-in and cut-out from first cluster
        cycle_powers = [entry['performance']['power']['average_cycle_power_w']
                        for entry in wind_speed_data_per_cluster[0]]
        cut_in_ws = self._find_cut_in_wind_speed(wind_speeds, cycle_powers)
        if cut_in_ws is None:
            cut_in_ws = wind_speeds[0]  # Fallback to first wind speed
        cut_out_ws = wind_speeds[-1]

        # Calculate nominal power (max across all clusters)
        nominal_power = max(
            max(entry['performance']['power']['average_cycle_power_w'] for entry in wsd)
            for wsd in wind_speed_data_per_cluster
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
                    'wing_area_m2': float(self.sys_props.kite_projected_area),
                    'nominal_power_w': float(nominal_power),
                    'nominal_tether_force_n': float(self.sys_props.tether_force_max_limit),
                    'cut_in_wind_speed_m_s': float(cut_in_ws),
                    'cut_out_wind_speed_m_s': float(cut_out_ws),
                },
                'wind_resource': {
                    'n_clusters': self.n_clusters,
                    'n_profiles_calculated': len(cluster_ids),
                    'profile_ids_calculated': list(cluster_ids),
                    'reference_height_m': float(self.reference_height),
                },
            },
            'altitudes_m': [float(a) for a in altitudes],
            'reference_wind_speeds_m_s': [float(ws) for ws in wind_speeds],
            'power_curves': power_curves,
        }

        # Save to file if path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            class IndentedDumper(yaml.Dumper):
                def increase_indent(self, flow=False, indentless=False):
                    return super(IndentedDumper, self).increase_indent(flow, indentless=False)

            with open(output_path, 'w') as f:
                yaml.dump(output, f, Dumper=IndentedDumper,
                          default_flow_style=False, sort_keys=False,
                          indent=2, width=1000)

            if verbose:
                print(f"\nPower curves saved to {output_path}")

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
        print(f"  Number of wind clusters: {self.n_clusters}")

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
        if ('kinematics' in kpi and kpi['kinematics']
                and 'steady_states' in kpi and kpi['steady_states']):
            if 'time' in kpi and kpi['time']:
                time_list = list(kpi['time'])
            else:
                n_pts = len(kpi['kinematics'])
                time_list = list(np.linspace(0, kpi['duration']['cycle'], n_pts))
            time_history = self._extract_time_history(
                time_list, kpi['kinematics'], kpi['steady_states']
            )

        entry = {
            'wind_speed_m_s': float(wind_speed),
            'success': bool(kpi.get('sim_successful', True)),
            'performance': {
                'power': {
                    'average_cycle_power_w': _safe_float(kpi['average_power']['cycle']),
                    'average_reel_out_power_w': _safe_float(kpi['average_power']['out']),
                    'average_reel_in_power_w': _safe_float(kpi['average_power']['in']),
                },
                'timing': {
                    'reel_out_time_s': _safe_float(kpi['duration']['out']),
                    'reel_in_time_s': (_safe_float(kpi['duration']['in'])
                                       + _safe_float(kpi['duration'].get('trans', 0.0))),
                    'cycle_time_s': _safe_float(kpi['duration']['cycle']),
                },
            },
        }
        if time_history is not None:
            entry['time_history'] = time_history

        return entry

    def _extract_time_history(self, time_list, kinematics, steady_states):
        """Extract time history data from kinematics and steady state objects.

        This unified method is used by both direct simulation and optimization.
        Both paths provide lists of kinematics and steady-state objects with the
        same attributes.

        Args:
            time_list (list): Time values [s].
            kinematics (list): Kinematics objects with z, straight_tether_length,
                elevation_angle attributes.
            steady_states (list): Steady state objects with tether_force_ground,
                power_ground, reeling_speed attributes.

        Returns:
            dict: Time history data with altitude, forces, power, speeds, etc.
                Downsampled to ~2 second intervals. Returns None if inputs are
                empty.
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

        for kin, ss in zip(kinematics, steady_states):
            altitude_full.append(float(kin.z))
            tether_force_full.append(float(ss.tether_force_ground))
            power_full.append(float(ss.power_ground))
            reel_speed_full.append(float(ss.reeling_speed))
            tether_length_full.append(float(kin.straight_tether_length))
            elevation_angle_full.append(float(np.degrees(kin.elevation_angle)))

        # Downsample to ~2 second intervals
        indices = self._downsample_indices(time_full, interval_s=2.0)

        return {
            'time_s': [time_full[i] for i in indices],
            'altitude_m': [altitude_full[i] for i in indices],
            'tether_force_n': [tether_force_full[i] for i in indices],
            'power_w': [power_full[i] for i in indices],
            'reel_speed_m_s': [reel_speed_full[i] for i in indices],
            'tether_length_m': [tether_length_full[i] for i in indices],
            'elevation_angle_deg': [elevation_angle_full[i] for i in indices],
        }

    @staticmethod
    def _downsample_indices(time_vector, interval_s=2.0):
        """Find indices for downsampling to approximately fixed time intervals.

        Always includes the first and last point.

        Args:
            time_vector (list): Time values [s].
            interval_s (float): Desired time interval [s].

        Returns:
            list: Indices to keep.
        """
        if not time_vector:
            return []
        
        if len(time_vector) == 1:
            return [0]
        
        indices = [0]  # Always include first point
        last_time = time_vector[0]
        
        for i, t in enumerate(time_vector[1:-1], start=1):
            if t - last_time >= interval_s:
                indices.append(i)
                last_time = t
        
        # Always include last point
        if indices[-1] != len(time_vector) - 1:
            indices.append(len(time_vector) - 1)
        
        return indices

    @staticmethod
    def _find_cut_in_wind_speed(wind_speeds, cycle_powers):
        """Find the cut-in wind speed where power first becomes positive.

        Args:
            wind_speeds (array-like): Wind speeds [m/s].
            cycle_powers (list): Cycle power values [W].

        Returns:
            float: Cut-in wind speed [m/s], or None if no data available.
        """
        if len(wind_speeds) == 0:
            return None
        for ws, power in zip(wind_speeds, cycle_powers):
            if power > 0:
                return float(ws)
        return float(wind_speeds[0])


