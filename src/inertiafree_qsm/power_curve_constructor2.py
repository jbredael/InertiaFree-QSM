# -*- coding: utf-8 -*-
"""Power curve generation for AWE systems.

This module provides functionality to generate power curves using either direct
simulation or sequential optimization with warm starts. It includes automatic
cut-in/cut-out estimation, constraint handling, and detailed diagnostics.

All wind speeds are referenced at the height specified in the wind_resource.yml
metadata (reference_height_m). The model uses the wind profile to derive wind
speeds at operating altitudes.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .config_loader import (
    load_system_config,
    load_wind_resource,
    load_simulation_settings,
    create_wind_profile_from_resource,
    get_cluster_data,
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
from .cycle_optimizer import OptimizerCycle


class PowerCurveConstructor2:
    """Generate power curves for an AWE system using direct simulation or optimization.

    This class supports two methods:
    1. Direct simulation: Fast method using pre-defined cycle settings
    2. Sequential optimization: Uses warm starts for optimal cycle performance

    Args:
        system_config_path (str): Path to the system configuration YAML file.
        wind_resource_path (str): Path to the wind resource YAML file.
        simulation_settings_path (str): Path to the simulation settings YAML file.
        validate_inputs (bool): Whether to validate input files using awesIO.

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
        validate_inputs=True,
    ):
        """Initialize the power curve optimizer with configuration files."""
        self.system_config_path = Path(system_config_path)
        self.wind_resource_path = Path(wind_resource_path)
        self.simulation_settings_path = Path(simulation_settings_path)

        # Load configurations (validation is internal to loader functions)
        self.sys_props_dict = load_system_config(self.system_config_path)
        self.wind_resource = load_wind_resource(self.wind_resource_path)
        self.simulation_settings = load_simulation_settings(
            self.simulation_settings_path
        )

        # Create system properties object
        self.sys_props = SystemProperties(self.sys_props_dict)

        # Reference height from wind resource metadata
        self.reference_height = self.wind_resource.get('metadata', {}).get(
            'reference_height_m'
        )

        # Number of clusters
        self.n_clusters = self.wind_resource.get('metadata', {}).get('n_clusters')

        # Calculate operating parameters
        self._calculate_operating_parameters()

        # Initialize result storage
        self.wind_speeds = None
        self.x_opts = []
        self.x0_list = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

    def _calculate_operating_parameters(self):
        """Calculate operating altitude and tether length from settings."""
        tether_length_start = self.simulation_settings['cycle'][
            'tether_length_start_retraction'
        ]
        tether_length_end = self.simulation_settings['cycle'][
            'tether_length_end_retraction'
        ]
        elevation_angle = self.simulation_settings['cycle']['elevation_angle_traction']

        self.tether_length_start = tether_length_start
        self.tether_length_end = tether_length_end
        self.avg_tether_length = (tether_length_start + tether_length_end) / 2
        self.elevation_angle = elevation_angle
        self.operating_altitude = self.avg_tether_length * np.sin(elevation_angle)

        # Get traction settings
        self.phi_traction = self.simulation_settings['traction']['azimuth_angle']
        self.chi_traction = self.simulation_settings['traction']['course_angle']

    def create_environment(self, cluster_id=0):
        """Create an environment state object for a specific wind cluster.

        Args:
            cluster_id (int): The cluster ID (0-indexed).

        Returns:
            NormalisedWindTable1D: Environment state with wind profile.
        """
        altitudes, u_normalized, h_ref = create_wind_profile_from_resource(
            self.wind_resource, cluster_id
        )

        env_state = NormalisedWindTable1D()
        env_state.heights = list(altitudes)
        env_state.normalised_wind_speeds = list(u_normalized)
        env_state.h_ref = h_ref
        env_state.set_reference_height(h_ref)

        return env_state

    def calc_tether_force_traction(self, env_state, straight_tether_length):
        """Calculate tether force for minimum reel-out speed.

        Args:
            env_state: Environment state object.
            straight_tether_length (float): Straight tether length [m].

        Returns:
            float: Tether force at ground [N].
        """
        theta_ro_ci = 25 * np.pi / 180.  # Representative elevation at cut-in
        kinematics = KiteKinematics(
            straight_tether_length,
            self.phi_traction,
            theta_ro_ci,
            self.chi_traction,
        )
        env_state.calculate(kinematics.z)
        self.sys_props.update(kinematics.straight_tether_length, True)
        ss = SteadyState({'enable_steady_state_errors': True})
        ss.control_settings = (
            'reeling_speed',
            self.sys_props.reeling_speed_min_limit,
        )
        ss.find_state(self.sys_props, env_state, kinematics)
        return ss.tether_force_ground

    def estimate_cut_in_wind_speed(self, env_state, v_start=5.6, v_max=10.0, dv=0.01):
        """Estimate cut-in wind speed for feasible steady flight states.

        Iteratively determine lowest wind speed for which, along the entire
        reel-out path, feasible steady flight states with minimum reel-out
        speed are found.

        Args:
            env_state: Environment state object.
            v_start (float): Starting wind speed [m/s].
            v_max (float): Maximum wind speed to check [m/s].
            dv (float): Wind speed step size [m/s].

        Returns:
            tuple: (cut_in_wind_speed, critical_force)
        """
        l0 = self.tether_length_end  # Tether length at start of reel-out
        l1 = self.tether_length_start  # Tether length at end of reel-out

        v = v_start
        while v < v_max:
            env_state.set_reference_wind_speed(v)
            try:
                tether_force_start = self.calc_tether_force_traction(env_state, l0)
                tether_force_end = self.calc_tether_force_traction(env_state, l1)

                critical_force = min(tether_force_start, tether_force_end)

                if (
                    tether_force_start > self.sys_props.tether_force_min_limit
                    and tether_force_end > self.sys_props.tether_force_min_limit
                ):
                    if v == v_start:
                        raise ValueError("Starting speed is too high.")
                    return v, critical_force
            except SteadyStateError:
                pass

            v += dv

        return v_max, self.sys_props.tether_force_min_limit

    def calc_n_crosswind_patterns(self, env_state, theta=60.0 * np.pi / 180.0):
        """Calculate number of crosswind manoeuvres flown.

        Args:
            env_state: Environment state object.
            theta (float): Elevation angle [rad].

        Returns:
            float: Number of crosswind patterns.
        """
        trac = TractionPhaseHybrid(
            {
                'control': (
                    'tether_force_ground',
                    self.sys_props.tether_force_max_limit,
                ),
                'azimuth_angle': self.phi_traction,
                'course_angle': self.chi_traction,
            }
        )
        trac.enable_limit_violation_error = True

        trac.tether_length_start = self.tether_length_end
        trac.tether_length_start_aim = self.tether_length_end
        trac.elevation_angle = TractionConstantElevation(theta)
        trac.tether_length_end = self.tether_length_start
        trac.finalize_start_and_end_kite_obj()
        trac.run_simulation(
            self.sys_props, env_state, {'enable_steady_state_errors': True}
        )

        return trac.n_crosswind_patterns

    def estimate_max_wind_speed_at_elevation(
        self, env_state, theta=60.0 * np.pi / 180.0, v_start=18.0, v_max=30.0, dv=0.1
    ):
        """Determine maximum wind speed allowing at least one crosswind pattern.

        Args:
            env_state: Environment state object.
            theta (float): Elevation angle [rad].
            v_start (float): Starting wind speed [m/s].
            v_max (float): Maximum wind speed to check [m/s].
            dv (float): Wind speed step size [m/s].

        Returns:
            float: Maximum feasible wind speed [m/s].
        """
        # Check if starting wind speed gives feasible solution
        env_state.set_reference_wind_speed(v_start)
        try:
            n_cw_patterns = self.calc_n_crosswind_patterns(env_state, theta)
        except SteadyStateError as e:
            if e.code != 8:
                raise ValueError(
                    "No feasible solution found for first assessed cut-out wind speed."
                )

        # Increase wind speed until crosswind patterns drop below one
        v = v_start + dv
        while v < v_max:
            env_state.set_reference_wind_speed(v)
            try:
                n_cw_patterns = self.calc_n_crosswind_patterns(env_state, theta)
                if n_cw_patterns < 1.0:
                    return v
            except SteadyStateError as e:
                if e.code != 8:  # Speed too low when e.code == 8
                    return v

            v += dv

        raise ValueError("Iteration did not find feasible cut-out speed.")

    def estimate_cut_out_wind_speed(self, env_state):
        """Estimate cut-out wind speed by iterating elevation angle.

        Elevation angle is increased with wind speed as a last means of
        de-powering. This finds the wind speed at which elevation angle
        reaches its upper limit.

        Args:
            env_state: Environment state object.

        Returns:
            tuple: (cut_out_wind_speed, elevation_angle)
        """
        beta = 60 * np.pi / 180.0
        dbeta = 1 * np.pi / 180.0
        vw_last = 0.0

        while beta > 25 * np.pi / 180.0:
            vw = self.estimate_max_wind_speed_at_elevation(env_state, beta)
            if vw <= vw_last:
                return vw_last, beta + dbeta
            vw_last = vw
            beta -= dbeta

        return vw_last, beta

    def run_optimization(self, wind_speed, power_optimizer, x0):
        """Run single optimization for given wind speed.

        Args:
            wind_speed (float): Wind speed at reference height [m/s].
            power_optimizer (OptimizerCycle): Optimizer instance.
            x0 (array): Starting point for optimization.

        Returns:
            tuple: (x_opt, sim_successful)
        """
        power_optimizer.environment_state.set_reference_wind_speed(wind_speed)

        power_optimizer.x0_real_scale = x0
        x_opt = power_optimizer.optimize()

        self.x0_list.append(x0)
        self.x_opts.append(x_opt)
        self.optimization_details.append(power_optimizer.op_res)

        try:
            cons, kpis = power_optimizer.eval_point()
            sim_successful = True
        except (SteadyStateError, OperationalLimitViolation, PhaseError) as e:
            print(f"Error occurred while evaluating optimal point: {e}")
            cons, kpis = power_optimizer.eval_point(relax_errors=True)
            sim_successful = False

        self.constraints.append(cons)
        kpis['sim_successful'] = sim_successful
        self.performance_indicators.append(kpis)

        return x_opt, sim_successful

    def run_predefined_sequence(self, optimization_sequence, x0_start, wind_speeds):
        """Run sequential optimizations with warm starts.

        Args:
            optimization_sequence (dict): Dict mapping wind speed thresholds to
                optimizer configurations.
            x0_start (array): Initial starting point.
            wind_speeds (array): Wind speeds to evaluate [m/s].
        """
        self.wind_speeds = wind_speeds

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
            try:
                x_opt, sim_successful = self.run_optimization(vw, power_optimizer, x0_next)
            except (OperationalLimitViolation, SteadyStateError, PhaseError):
                try:  # Retry for slightly different wind speed
                    x_opt, sim_successful = self.run_optimization(
                        vw + 1e-2, power_optimizer, x0_next
                    )
                    self.wind_speeds[i] = vw + 1e-2
                except (OperationalLimitViolation, SteadyStateError, PhaseError):
                    self.wind_speeds = self.wind_speeds[:i]
                    print(
                        f"Optimization sequence stopped prematurely. "
                        f"{self.wind_speeds[-1]:.1f} m/s is the highest successful wind speed."
                    )
                    break

            if sim_successful:
                x_opt_last = x_opt
                vw_last = vw

    def generate_power_curve(
        self,
        cluster_id=0,
        vw_cut_in=None,
        vw_cut_out=None,
        n_points=50,
        verbose=True,
    ):
        """Generate optimized power curve for a wind profile cluster.

        Args:
            cluster_id (int): The cluster ID (0-indexed).
            vw_cut_in (float, optional): Cut-in wind speed [m/s].
            vw_cut_out (float, optional): Cut-out wind speed [m/s].
            n_points (int): Number of points in power curve.
            verbose (bool): Whether to print progress.

        Returns:
            dict: Power curve data in awesIO format.
        """
        # Create environment
        env_state = self.create_environment(cluster_id)

        # Estimate operational limits if not provided
        if vw_cut_in is None or vw_cut_out is None:
            if verbose:
                print(f"Estimating operational limits for cluster {cluster_id + 1}...")

            if vw_cut_in is None:
                vw_cut_in, critical_force = self.estimate_cut_in_wind_speed(env_state)
                if verbose:
                    print(f"  Cut-in wind speed: {vw_cut_in:.2f} m/s")

            if vw_cut_out is None:
                vw_cut_out, elev_cut_out = self.estimate_cut_out_wind_speed(env_state)
                if verbose:
                    print(f"  Cut-out wind speed: {vw_cut_out:.2f} m/s")

        # Generate wind speed array
        wind_speeds = np.linspace(vw_cut_in, vw_cut_out - 1, n_points)
        wind_speeds = np.concatenate(
            (wind_speeds, np.linspace(vw_cut_out - 1, vw_cut_out - 0.05, 15))
        )

        # Configure optimization sequence
        cycle_sim_settings_phase1 = deepcopy(self.simulation_settings)
        cycle_sim_settings_phase1['cycle']['traction_phase'] = TractionPhase
        cycle_sim_settings_phase1['cycle']['include_transition_energy'] = False

        cycle_sim_settings_phase2 = deepcopy(cycle_sim_settings_phase1)
        cycle_sim_settings_phase2['cycle']['traction_phase'] = TractionPhaseHybrid

        # Create optimizers
        op_cycle_phase1 = OptimizerCycle(
            cycle_sim_settings_phase1,
            self.sys_props,
            env_state,
            reduce_x=np.array([0, 1, 2, 3]),
        )
        op_cycle_phase1.bounds_real_scale[2][1] = 30 * np.pi / 180.0

        op_cycle_phase2 = OptimizerCycle(
            cycle_sim_settings_phase2,
            self.sys_props,
            env_state,
            reduce_x=np.array([0, 1, 2, 3]),
        )

        # Define sequential optimization strategy
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

        # Define starting point
        if vw_cut_in is not None:
            x0_start = np.array(
                [
                    critical_force if 'critical_force' in locals() else 5000.0,
                    300.0,
                    25 * np.pi / 180.0,
                    150.0,
                    200.0,
                ]
            )
        else:
            x0_start = np.array([5000.0, 300.0, 25 * np.pi / 180.0, 150.0, 200.0])

        # Run sequential optimizations
        if verbose:
            print(f"Running optimizations for cluster {cluster_id + 1}...")

        self.run_predefined_sequence(optimization_sequence, x0_start, wind_speeds)

        # Build power curve output
        return self._build_power_curve_output(cluster_id)

    def _build_power_curve_output(self, cluster_id):
        """Build power curve output in awesIO format.

        Args:
            cluster_id (int): The cluster ID (0-indexed).

        Returns:
            dict: Power curve data in awesIO format.
        """
        cluster_data = get_cluster_data(self.wind_resource, cluster_id)
        env_state = self.create_environment(cluster_id)

        # Calculate speed ratio at operating altitude
        env_state.set_reference_wind_speed(1.0)
        env_state.calculate(self.operating_altitude)
        speed_ratio = env_state.wind_speed

        # Extract data from performance indicators
        cycle_powers = []
        reel_out_powers = []
        reel_in_powers = []
        reel_out_times = []
        reel_in_times = []
        cycle_times = []

        for kpi in self.performance_indicators:
            cycle_powers.append(kpi['average_power']['cycle'])
            reel_out_powers.append(kpi['average_power']['out'])
            reel_in_powers.append(kpi['average_power']['in'])
            reel_out_times.append(kpi['duration']['out'])
            reel_in_times.append(
                kpi['duration']['in'] + kpi['duration'].get('trans', 0.0)
            )
            cycle_times.append(kpi['duration']['cycle'])

        power_curve = {
            'profile_id': cluster_id + 1,
            'speed_ratio_at_operating_altitude': float(speed_ratio),
            'u_normalized': [float(u) for u in cluster_data.get('u_normalized', [])],
            'v_normalized': [float(v) for v in cluster_data.get('v_normalized', [])],
            'probability_weight': float(cluster_data.get('frequency', 1.0 / self.n_clusters)),
            'cycle_power_w': cycle_powers,
            'reel_out_power_w': reel_out_powers,
            'reel_in_power_w': reel_in_powers,
            'reel_out_time_s': reel_out_times,
            'reel_in_time_s': reel_in_times,
            'cycle_time_s': cycle_times,
        }

        return power_curve

    def export_results(self, file_path):
        """Export optimization results to YAML file.

        Args:
            file_path (str): Path to output YAML file.
        """
        export_dict = {
            'wind_speeds': [float(v) for v in self.wind_speeds],
            'x_opts': [x.tolist() for x in self.x_opts],
            'x0_list': [x.tolist() for x in self.x0_list],
            'constraints': [c.tolist() if hasattr(c, 'tolist') else c for c in self.constraints],
            'performance_indicators': self.performance_indicators,
            'optimization_details': [
                {k: v for k, v in od.items() if k not in ['x', 'fun']}
                for od in self.optimization_details
            ],
        }

        with open(file_path, 'w') as f:
            yaml.dump(export_dict, f, default_flow_style=False)

    def import_results(self, file_path):
        """Import optimization results from YAML file.

        Args:
            file_path (str): Path to input YAML file.
        """
        with open(file_path, 'r') as f:
            import_dict = yaml.safe_load(f)

        self.wind_speeds = np.array(import_dict['wind_speeds'])
        self.x_opts = [np.array(x) for x in import_dict['x_opts']]
        self.x0_list = [np.array(x) for x in import_dict['x0_list']]
        self.constraints = import_dict['constraints']
        self.performance_indicators = import_dict['performance_indicators']
        self.optimization_details = import_dict['optimization_details']

    def plot_optimal_trajectories(
        self, wind_speed_ids=None, ax=None, circle_radius=200, elevation_line=25 * np.pi / 180
    ):
        """Plot optimal kite trajectories for selected wind speeds.

        Args:
            wind_speed_ids (list, optional): Indices of wind speeds to plot.
            ax: Matplotlib axes object.
            circle_radius (float): Radius for reference circle [m].
            elevation_line (float): Elevation angle for reference line [rad].
        """
        if ax is None:
            plt.figure(figsize=(6, 3.5))
            plt.subplots_adjust(right=0.65)
            ax = plt.gca()

        if wind_speed_ids is None:
            if len(self.wind_speeds) > 8:
                wind_speed_ids = [
                    int(a) for a in np.linspace(0, len(self.wind_speeds) - 1, 6)
                ]
            else:
                wind_speed_ids = range(len(self.wind_speeds))

        for i in wind_speed_ids:
            v = self.wind_speeds[i]
            kpis = self.performance_indicators[i]
            if kpis is None:
                print(f"No trajectory available for {v} m/s wind speed.")
                continue

            x_kite, z_kite = zip(*[(kp.x, kp.z) for kp in kpis['kinematics']])
            ax.plot(x_kite, z_kite, label=f"$v_{{{int(self.reference_height)}m}}$={v:.1f} m/s")

        # Plot reference circle
        phi = np.linspace(0, 2 * np.pi / 3, 40)
        x_circle = np.cos(phi) * circle_radius
        z_circle = np.sin(phi) * circle_radius
        ax.plot(x_circle, z_circle, 'k--', linewidth=1)

        # Plot elevation line
        x_elev = np.linspace(0, 400, 40)
        z_elev = np.tan(elevation_line) * x_elev
        ax.plot(x_elev, z_elev, 'k--', linewidth=1)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.grid()
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    def plot_optimization_results(
        self,
        opt_variable_labels=None,
        opt_variable_bounds=None,
        tether_force_limits=None,
        reeling_speed_limits=None,
    ):
        """Plot comprehensive optimization results and diagnostics.

        Args:
            opt_variable_labels (list, optional): Labels for optimization variables.
            opt_variable_bounds (array, optional): Bounds for optimization variables.
            tether_force_limits (list, optional): [min, max] tether force limits.
            reeling_speed_limits (list, optional): [min, max] reeling speed limits.
        """
        assert self.x_opts, "No optimization results available for plotting."

        xf, x0 = self.x_opts, self.x0_list
        cons = self.constraints
        kpis, opt_details = self.performance_indicators, self.optimization_details

        n_opt_vars = len(xf[0])
        fig, ax = plt.subplots(max([n_opt_vars, 6]), 2, sharex=True, figsize=(12, 10))

        # Plot optimization variables
        for i in range(n_opt_vars):
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in xf], label='x_opt')
            ax[i, 0].plot(
                self.wind_speeds,
                [a[i] for a in x0],
                'o',
                markerfacecolor='None',
                label='x0',
            )

            if opt_variable_labels:
                ax[i, 0].set_ylabel(opt_variable_labels[i])
            else:
                ax[i, 0].set_ylabel(f"x[{i}]")

            if opt_variable_bounds is not None:
                ax[i, 0].axhline(opt_variable_bounds[i, 0], linestyle='--', color='k')
                ax[i, 0].axhline(opt_variable_bounds[i, 1], linestyle='--', color='k')

            ax[i, 0].grid()
        ax[0, 0].legend()

        # Plot optimization iterations
        nits = np.array([od['nit'] for od in opt_details])
        ax[0, 1].plot(self.wind_speeds, nits)
        mask_opt_failed = np.array([~od['success'] for od in opt_details])
        ax[0, 1].plot(
            self.wind_speeds[mask_opt_failed],
            nits[mask_opt_failed],
            'x',
            label='opt failed',
        )
        mask_sim_failed = np.array([~kpi['sim_successful'] for kpi in kpis])
        ax[0, 1].plot(
            self.wind_speeds[mask_sim_failed],
            nits[mask_sim_failed],
            'x',
            label='sim failed',
        )
        ax[0, 1].grid()
        ax[0, 1].legend()
        ax[0, 1].set_ylabel('Optimization iterations [-]')

        # Plot power
        cons_threshold = -0.1
        mask_cons_adhered = np.array(
            [all([c >= cons_threshold for c in con]) for con in cons]
        )
        mask_plot_power = ~mask_sim_failed & mask_cons_adhered
        power = np.array([kpi['average_power']['cycle'] for kpi in kpis])
        power[~mask_plot_power] = np.nan
        ax[1, 1].plot(self.wind_speeds, power)
        ax[1, 1].grid()
        ax[1, 1].set_ylabel('Mean power [W]')

        # Plot tether forces
        max_force_in = np.array([kpi['max_tether_force']['in'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_in, label='max_tether_force.in')
        max_force_out = np.array([kpi['max_tether_force']['out'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_out, label='max_tether_force.out')
        if 'trans' in kpis[0]['max_tether_force']:
            max_force_trans = np.array([kpi['max_tether_force']['trans'] for kpi in kpis])
            ax[2, 1].plot(self.wind_speeds, max_force_trans, label='max_tether_force.trans')
        if tether_force_limits:
            ax[2, 1].axhline(tether_force_limits[0], linestyle='--', color='k')
            ax[2, 1].axhline(tether_force_limits[1], linestyle='--', color='k')
        ax[2, 1].grid()
        ax[2, 1].set_ylabel('Tether force [N]')
        ax[2, 1].legend(loc=2)

        # Plot reeling speeds
        max_speed_in = np.array([kpi['max_reeling_speed']['in'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_in, label='max_reeling_speed.in')
        max_speed_out = np.array([kpi['max_reeling_speed']['out'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_out, label='max_reeling_speed.out')
        min_speed_in = np.array([kpi['min_reeling_speed']['in'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_in, label='min_reeling_speed.in')
        min_speed_out = np.array([kpi['min_reeling_speed']['out'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_out, label='min_reeling_speed.out')
        if reeling_speed_limits:
            ax[3, 1].axhline(reeling_speed_limits[0], linestyle='--', color='k')
            ax[3, 1].axhline(reeling_speed_limits[1], linestyle='--', color='k')
        ax[3, 1].grid()
        ax[3, 1].set_ylabel('Reeling speed [m/s]')
        ax[3, 1].legend(loc=2)

        # Plot constraint matrix (color coded)
        cons_matrix = np.array(cons).transpose()
        n_cons = cons_matrix.shape[0]

        cons_threshold_magenta = -0.1
        cons_threshold_red = -1e-6

        color_code_matrix = np.where(cons_matrix < cons_threshold_magenta, -2, 0)
        color_code_matrix = np.where(
            (cons_matrix >= cons_threshold_magenta) & (cons_matrix < cons_threshold_red),
            -1,
            color_code_matrix,
        )
        color_code_matrix = np.where(
            (cons_matrix >= cons_threshold_red) & (cons_matrix < 1e-3),
            1,
            color_code_matrix,
        )
        color_code_matrix = np.where(cons_matrix == 0.0, 0, color_code_matrix)
        color_code_matrix = np.where(cons_matrix >= 1e-3, 2, color_code_matrix)

        cmap = mpl.colors.ListedColormap(['r', 'm', 'y', 'g', 'b'])
        bounds = [-2, -1, 0, 1, 2]
        mpl.colors.BoundaryNorm(bounds, cmap.N)
        im1 = ax[4, 1].matshow(
            color_code_matrix,
            cmap=cmap,
            vmin=bounds[0],
            vmax=bounds[-1],
            extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0],
        )
        ax[4, 1].set_yticks(np.array(range(n_cons)) + 0.5)
        ax[4, 1].set_yticklabels(range(n_cons))
        ax[4, 1].set_ylabel("Constraint ID's")

        # Add colorbar
        ax_pos = ax[4, 1].get_position()
        h_cbar = ax_pos.y1 - ax_pos.y0
        w_cbar = 0.012
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = np.arange(-2 + 4 / 10.0, 2.0, 4 / 5.0)
        cbar_ticks_labels = ['<-.1', '<0', '0', '~0', '>0']
        cbar = fig.colorbar(im1, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        # Plot constraint matrix (continuous)
        plot_cons_range = [-0.1, 0.1]
        im2 = ax[5, 1].matshow(
            cons_matrix,
            vmin=plot_cons_range[0],
            vmax=plot_cons_range[1],
            cmap=mpl.cm.YlGnBu_r,
            extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0],
        )
        ax[5, 1].set_yticks(np.array(range(n_cons)) + 0.5)
        ax[5, 1].set_yticklabels(range(n_cons))
        ax[5, 1].set_ylabel("Constraint ID's")

        # Add colorbar
        ax_pos = ax[5, 1].get_position()
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = plot_cons_range[:]
        cbar_ticks_labels = [str(v) for v in cbar_ticks]
        if plot_cons_range[0] < np.min(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(1, np.min(cons_matrix))
            cbar_ticks_labels.insert(1, f'min: {np.min(cons_matrix):.2E}')
        if plot_cons_range[0] < np.max(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(-1, np.max(cons_matrix))
            cbar_ticks_labels.insert(-1, f'max: {np.max(cons_matrix):.2E}')
        cbar = fig.colorbar(im2, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        ax[-1, 0].set_xlabel('Wind speeds [m/s]')
        ax[-1, 1].set_xlabel('Wind speeds [m/s]')
        ax[0, 0].set_xlim([self.wind_speeds[0], self.wind_speeds[-1]])

        plt.tight_layout()

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
        print(f"  Operating altitude: {self.operating_altitude:.1f} m")
        print(f"  Average tether length: {self.avg_tether_length:.1f} m")
        print(f"  Elevation angle: {np.degrees(self.elevation_angle):.1f}°")
        print(f"  Number of wind clusters: {self.n_clusters}")

    def generate_power_curves_direct(
        self,
        wind_speeds=None,
        cluster_ids=None,
        output_path=None,
        verbose=True,
        plot=False,
    ):
        """Generate power curves using direct simulation (non-optimized).

        This method runs the QSM model with pre-defined cycle settings for each
        wind speed. It's much faster than optimization but may not find optimal
        performance.

        Args:
            wind_speeds (array-like, optional): Wind speeds to evaluate [m/s].
                Defaults to 4.0 to 25.5 m/s in 0.5 m/s steps.
            cluster_ids (list, optional): List of cluster IDs (0-indexed) to calculate.
                If None, calculates all clusters.
            output_path (str, optional): Path to save the output YAML file.
            verbose (bool): Whether to print progress messages.
            plot (bool): Whether to plot the power curves.

        Returns:
            dict: Power curve data in awesIO format.
        """
        # Default wind speed range
        if wind_speeds is None:
            wind_speeds = np.arange(4.0, 26.0, 0.5)

        # Determine which clusters to process
        if cluster_ids is None:
            cluster_ids = list(range(self.n_clusters))
        elif not isinstance(cluster_ids, (list, tuple)):
            cluster_ids = [cluster_ids]

        # Calculate power curves for selected clusters
        power_curves = []
        all_results = []

        for cluster_id in cluster_ids:
            power_curve, results = self._calculate_power_curve_direct(
                cluster_id,
                wind_speeds,
                verbose,
            )
            power_curves.append(power_curve)
            all_results.append(results)

        # Determine cut-in and cut-out wind speeds from first cluster
        first_curve = power_curves[0]
        cycle_powers = first_curve['cycle_power_w']
        cut_in_ws = self._find_cut_in_wind_speed(wind_speeds, cycle_powers)
        cut_out_ws = float(wind_speeds[-1])

        # Calculate nominal power (max power from all clusters)
        nominal_power = max(
            max(pc['cycle_power_w']) for pc in power_curves
        )

        # Prepare output in awesIO power_curves_schema format
        altitudes = self.wind_resource.get('altitudes', [])
        output = {
            'metadata': {
                'name': 'Ground-Gen Power Curves (Direct Simulation)',
                'description': 'Power curves for pumping ground-gen AWE system',
                'note': f'Power curve data generated from QSM direct simulation. Wind speeds at {self.reference_height:.0f}m reference height.',
                'awesIO_version': '0.1.0',
                'schema': 'power_curves_schema.yml',
                'time_created': datetime.now().isoformat(),
                'model_config': {
                    'wing_area_m2': float(self.sys_props.kite_projected_area),
                    'nominal_power_w': float(nominal_power),
                    'nominal_tether_force_n': float(self.sys_props.tether_force_max_limit),
                    'cut_in_wind_speed_m_s': float(cut_in_ws),
                    'cut_out_wind_speed_m_s': float(cut_out_ws),
                    'operating_altitude_m': float(self.operating_altitude),
                    'tether_length_operational_m': float(self.avg_tether_length),
                },
                'wind_resource': {
                    'n_clusters': self.n_clusters,
                    'n_profiles_calculated': len(cluster_ids),
                    'profile_ids_calculated': [cid + 1 for cid in cluster_ids],
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

            with open(output_path, 'w') as f:
                yaml.dump(output, f, default_flow_style=False, sort_keys=False)

            if verbose:
                print(f"\nPower curves saved to {output_path}")

        # Plot if requested
        if plot:
            self._plot_power_curves_direct(wind_speeds, power_curves, output_path)

        return output

    def _calculate_power_curve_direct(self, cluster_id, wind_speeds, verbose):
        """Calculate power curve for a single wind cluster using direct simulation.

        Args:
            cluster_id (int): The cluster ID (0-indexed).
            wind_speeds (array-like): Wind speeds to evaluate [m/s].
            verbose (bool): Whether to print progress messages.

        Returns:
            tuple: (power_curve dict in awesIO format, list of detailed results)
        """
        # Create environment state for this cluster
        env_state = self.create_environment(cluster_id)

        # Get cluster data
        cluster_data = get_cluster_data(self.wind_resource, cluster_id)
        probability_weight = cluster_data.get('frequency', 1.0 / self.n_clusters)
        v_normalized = cluster_data.get('v_normalized', [0.0] * len(cluster_data.get('u_normalized', [])))

        # Calculate speed ratio at operating altitude
        env_state.set_reference_wind_speed(1.0)
        env_state.calculate(self.operating_altitude)
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

            result = self._run_single_simulation_direct(ws, env_state)
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
            'u_normalized': [float(u) for u in cluster_data.get('u_normalized', [])],
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

    def _run_single_simulation_direct(self, wind_speed, env_state):
        """Run a single cycle simulation for one wind speed using direct simulation.

        Args:
            wind_speed (float): Reference wind speed [m/s].
            env_state (NormalisedWindTable1D): Environment state object.

        Returns:
            dict: Simulation result with power and timing data.
        """
        env_state.set_reference_wind_speed(wind_speed)

        settings = self.simulation_settings.copy()
        settings['cycle'] = self.simulation_settings['cycle'].copy()
        settings['cycle']['traction_phase'] = TractionPhase

        cycle = Cycle(settings, impose_operational_limits=True)

        try:
            error_in_phase, average_power = cycle.run_simulation(
                self.sys_props, env_state, print_summary=False
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

    @staticmethod
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

    @staticmethod
    def _plot_power_curves_direct(wind_speeds, power_curves, output_path):
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
        if output_path is not None:
            plot_path = output_path.with_suffix('.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Power curve plot saved to {plot_path}")

