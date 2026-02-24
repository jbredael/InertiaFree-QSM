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
        altitudes = self.wind_resource['altitudes']
        clusters = self.wind_resource['clusters']

        if not clusters:
            raise ValueError("No clusters found in wind resource data")

        # Find the cluster with matching ID (clusters use 1-indexed IDs)
        cluster = None
        for c in clusters:
            if c.get('id') == cluster_id + 1:
                cluster = c
                break

        if cluster is None:
            if cluster_id >= len(clusters):
                raise ValueError(
                    f"Cluster ID {cluster_id} not found in wind resource data"
                )
            cluster = clusters[cluster_id]

        u_normalized = np.array(cluster.get('u_normalized', []))
        h_ref = self.wind_resource.get('metadata', {}).get('reference_height_m')

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

    def estimate_cut_in_wind_speed(self, env_state, v_start=1.0, v_max=10.0, dv=0.01):
        """Estimate cut-in wind speed for feasible steady flight states.

        Iteratively determine lowest wind speed for which, along the entire
        reel-out path, feasible steady flight states with minimum reel-out
        speed are found.

        Args:
            env_state: Environment state object.
            v_start (float): Starting wind speed [m/s]. Should be below the actual
                cut-in speed to allow upward search.
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
                        # Starting speed already feasible, return it as lower bound
                        print(f"  Note: Cut-in speed may be below {v_start:.1f} m/s")
                        return v_start, critical_force
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
        self, env_state, theta=60.0 * np.pi / 180.0, v_start=18.0, v_max=30.0, dv=0.1):
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
        fine_n_points_near_cutout=15,
        fine_range_m_s=1.0,
        output_path=None,
        verbose=True,
    ):
        """Generate optimized power curve for a wind profile cluster.

        This method performs sequential optimization to find optimal cycle parameters
        at each wind speed. The elevation angle from simulation settings is NOT used
        as a constraint - it is only used to calculate the operating altitude.

        Args:
            cluster_id (int): The cluster ID (0-indexed).
            vw_cut_in (float, optional): Cut-in wind speed [m/s].
            vw_cut_out (float, optional): Cut-out wind speed [m/s].
            n_points (int): Number of points in power curve.
            fine_n_points_near_cutout (int): Number of fine-resolution points near cut-out.
                Set to 0 to disable fine resolution near cut-out.
            fine_range_m_s (float): Wind speed range [m/s] for fine resolution.
                Fine points span from (cut_out - range) to (cut_out - 0.05).
            output_path (str, optional): Path to save the output YAML file.
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
        wind_speeds = np.linspace(vw_cut_in, vw_cut_out - fine_range_m_s, n_points)
        
        # Add fine resolution points near cut-out if requested
        if fine_n_points_near_cutout > 0:
            fine_points = np.linspace(
                vw_cut_out - fine_range_m_s, 
                vw_cut_out - 0.05, 
                fine_n_points_near_cutout
            )
            wind_speeds = np.concatenate((wind_speeds, fine_points))

        # Configure optimization sequence
        cycle_sim_settings_phase1 = deepcopy(self.simulation_settings)
        cycle_sim_settings_phase1['cycle']['traction_phase'] = TractionPhase
        cycle_sim_settings_phase1['cycle']['include_transition_energy'] = False

        cycle_sim_settings_phase2 = deepcopy(cycle_sim_settings_phase1)
        cycle_sim_settings_phase2['cycle']['traction_phase'] = TractionPhaseHybrid

        # Create optimizers — build optimizer_config dicts from settings
        opt_settings = self.simulation_settings['optimization']
        opt_config = {
            'x0': opt_settings['optimizer']['x0'].copy(),
            'scaling': opt_settings['optimizer']['scaling'].copy(),
            'bounds': opt_settings['bounds'].copy(),
        }

        # Phase 1: cap elevation upper bound at 30 deg
        opt_config_phase1 = {
            'x0': opt_config['x0'].copy(),
            'scaling': opt_config['scaling'].copy(),
            'bounds': opt_config['bounds'].copy(),
        }
        opt_config_phase1['bounds'][2, 1] = 30 * np.pi / 180.0

        op_cycle_phase1 = OptimizerCycle(
            cycle_sim_settings_phase1,
            self.sys_props,
            env_state,
            optimizer_config=opt_config_phase1,
            reduce_x=np.array([0, 1, 2, 3]),
        )

        op_cycle_phase2 = OptimizerCycle(
            cycle_sim_settings_phase2,
            self.sys_props,
            env_state,
            optimizer_config=opt_config,
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

        # Build wind speed data entries from optimization KPIs
        wind_speed_data = []
        for ws, kpi in zip(self.wind_speeds, self.performance_indicators):
            # Get time history from KPI kinematics/steady_states
            time_history = None
            if ('kinematics' in kpi and kpi['kinematics']
                    and 'steady_states' in kpi and kpi['steady_states']):
                # Build time vector from KPI
                if 'time' in kpi and kpi['time']:
                    time_list = list(kpi['time'])
                else:
                    n_pts = len(kpi['kinematics'])
                    time_list = list(np.linspace(0, kpi['duration']['cycle'], n_pts))
                time_history = self._extract_time_history(
                    time_list, kpi['kinematics'], kpi['steady_states']
                )

            def _safe_float(val):
                """Convert value to float, returning NaN for None."""
                if val is None:
                    return float('nan')
                return float(val)

            wind_speed_entry = {
                'wind_speed_m_s': float(ws),
                'success': bool(kpi.get('sim_successful', True)),
                'performance': {
                    'power': {
                        'average_cycle_power_w': _safe_float(kpi['average_power']['cycle']),
                        'average_reel_out_power_w': _safe_float(kpi['average_power']['out']),
                        'average_reel_in_power_w': _safe_float(kpi['average_power']['in']),
                    },
                    'timing': {
                        'reel_out_time_s': _safe_float(kpi['duration']['out']),
                        'reel_in_time_s': _safe_float(kpi['duration']['in']) + _safe_float(kpi['duration'].get('trans', 0.0)),
                        'cycle_time_s': _safe_float(kpi['duration']['cycle']),
                    },
                },
            }
            if time_history is not None:
                wind_speed_entry['time_history'] = time_history

            wind_speed_data.append(wind_speed_entry)

        # Build full output with metadata and save
        output = self._build_and_save_output(
            cluster_ids=[cluster_id],
            wind_speed_data_per_cluster=[wind_speed_data],
            method_name='Optimization-Based',
            output_path=output_path,
            verbose=verbose,
        )

        return output

    def _build_and_save_output(
        self,
        cluster_ids,
        wind_speed_data_per_cluster,
        method_name,
        output_path=None,
        verbose=True,
    ):
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

        Returns:
            dict: Complete power curve data in awesIO format.
        """
        # Extract wind speeds from first cluster's data
        wind_speeds = [entry['wind_speed_m_s'] for entry in wind_speed_data_per_cluster[0]]

        # Build per-cluster power curve dicts
        power_curves = []
        for cluster_id, wind_speed_data in zip(cluster_ids, wind_speed_data_per_cluster):
            clusters = self.wind_resource.get('clusters', [])
            if cluster_id >= len(clusters):
                raise ValueError(
                    f"Cluster ID {cluster_id} not found. "
                    f"Available clusters: 0-{len(clusters)-1}"
                )
            cluster_data = clusters[cluster_id]
            v_normalized = cluster_data.get(
                'v_normalized',
                [0.0] * len(cluster_data.get('u_normalized', [])),
            )
            power_curves.append({
                'profile_id': cluster_id + 1,
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

            class IndentedDumper(yaml.Dumper):
                def increase_indent(self, flow=False, indentless=False):
                    return super(IndentedDumper, self).increase_indent(flow, indentless=False)

            with open(output_path, 'w') as f:
                yaml.dump(output, f, Dumper=IndentedDumper,
                          default_flow_style=False, sort_keys=False,
                          indent=2, width=1000)

            if verbose:
                print(f"\nPower curves saved to {output_path}")

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
        )

        return output

    def _calculate_power_curve_direct(self, cluster_id, wind_speeds, verbose):
        """Calculate power curve for a single wind cluster using direct simulation.

        Args:
            cluster_id (int): The cluster ID (0-indexed).
            wind_speeds (array-like): Wind speeds to evaluate [m/s].
            verbose (bool): Whether to print progress messages.

        Returns:
            list: Wind speed data entries for this cluster.
        """
        env_state = self.create_environment(cluster_id)

        if verbose:
            print(f"Calculating power curve for cluster {cluster_id + 1}...")

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

        try:
            error_in_phase, average_power = cycle.run_simulation(
                self.sys_props, env_state, print_summary=False
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
            elevation_angle_full.append(float(kin.elevation_angle))

        # Downsample to ~2 second intervals
        indices = self._downsample_indices(time_full, interval_s=2.0)

        return {
            'time_s': [time_full[i] for i in indices],
            'altitude_m': [altitude_full[i] for i in indices],
            'tether_force_n': [tether_force_full[i] for i in indices],
            'power_w': [power_full[i] for i in indices],
            'reel_speed_m_s': [reel_speed_full[i] for i in indices],
            'tether_length_m': [tether_length_full[i] for i in indices],
            'elevation_angle_rad': [elevation_angle_full[i] for i in indices],
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
            float: Cut-in wind speed [m/s].
        """
        for ws, power in zip(wind_speeds, cycle_powers):
            if power > 0:
                return float(ws)
        return float(wind_speeds[0])


