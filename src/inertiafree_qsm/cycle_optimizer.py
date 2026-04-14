"""Cycle optimizer for AWE pumping cycle power optimization.

This module provides SLSQP-based optimization for pumping cycle parameters
to maximize power output while respecting operational constraints.
"""

from copy import deepcopy

import numpy as np
from scipy import optimize as op

from .qsm import Cycle, TractionPhase


class CycleOptimizer:
    """SLSQP-based optimizer for AWE pumping cycle parameters.

    Optimizes reeling speeds, tether lengths, and optionally the traction
    elevation angle array to maximize average cycle power while respecting
    operational bounds.

    The active decision variable vector depends on ``optimize_variables`` in the
    simulation settings.  The four base variables are:
        [reeling_speed_out, reeling_speed_in,
         fraction_tether_length_retraction_end,
         fraction_tether_length_traction_end]
    When ``elevation_angle_traction`` is enabled (and regime 3 does not
    dominate the full traction phase), N additional elevation angle values
    (one per entry in ``cycle.elevation_angle_traction``) are appended.

    Args:
        simulation_settings (dict): Full simulation settings dict (as returned
            by ``load_system_and_simulation_settings``).
        sys_props (SystemProperties): System properties object.
        env_state (NormalisedWindTable1D): Environment state with wind profile.

    Attributes:
        history (list): List of dicts recording every objective evaluation,
            each with keys ``x`` (variable vector) and ``power`` (cycle power [W]).
    """
    # Large penalty for failed simulations to steer the optimizer away.
    PENALTY = 1e6

    def __init__(self, simulation_settings, sys_props, env_state):
        self.simulation_settings = simulation_settings
        self.sys_props = sys_props
        self.env_state = env_state
        self.history = []

        opt_config = simulation_settings['optimization']
        self.optimizer_config = opt_config['optimizer']
        self.boundsDict = opt_config['bounds']
        self.constraintsDict = opt_config['constraints']
        self.opt_vars = self.optimizer_config.get('optimize_variables', {})
        self.minimum_height = simulation_settings.get('cycle', {}).get('minimum_height', 0.0)

        maxTetherLength = sys_props.max_tether_length
        fracStartMin = self.boundsDict['tether_length_traction_end'][0] / maxTetherLength
        fracStartMax = self.boundsDict['tether_length_traction_end'][1] / maxTetherLength
        fracEndMin = self.boundsDict['tether_length_retraction_end'][0] / maxTetherLength
        fracEndMax = self.boundsDict['tether_length_retraction_end'][1] / maxTetherLength

        # Nominal (fixed) values from settings, used when a variable is not optimised.
        self._nominal_rs_out = float(simulation_settings['traction']['control'][1])
        self._nominal_rs_in = float(simulation_settings['retraction']['control'][1])
        self._nominal_frac_end = (
            simulation_settings['cycle']['tether_length_end_retraction'] / maxTetherLength
        )
        self._nominal_frac_start = (
            simulation_settings['cycle']['tether_length_end_traction'] / maxTetherLength
        )
        self._nominal_elevation = np.asarray(
            simulation_settings['cycle']['elevation_angle_traction']
        ).flatten()  # radians
        self._n_elev = self._nominal_elevation.size

        # Build the initial list of active optimisation variables.
        # Each entry: (name, x0_value, (lo_bound, hi_bound), scaling)
        self._var_specs = []
        x0_base = np.array(self.optimizer_config['x0'], dtype=float)
        scaling_base = np.array(self.optimizer_config['scaling'], dtype=float)

        if self.opt_vars.get('reeling_speed_traction', True):
            x0_rs_out = x0_base[0] if len(x0_base) > 0 else self._nominal_rs_out
            sc = scaling_base[0] if len(scaling_base) > 0 else 1.0
            self._var_specs.append(
                ('reeling_speed_out', x0_rs_out, self.boundsDict['reeling_speed_out'], sc)
            )

        if self.opt_vars.get('reeling_speed_retraction', True):
            x0_rs_in = x0_base[1] if len(x0_base) > 1 else self._nominal_rs_in
            sc = scaling_base[1] if len(scaling_base) > 1 else 1.0
            self._var_specs.append(
                ('reeling_speed_in', x0_rs_in, self.boundsDict['reeling_speed_in'], sc)
            )

        if self.opt_vars.get('fraction_tether_length_retraction_end', True):
            x0_frac_end = x0_base[2] if len(x0_base) > 2 else self._nominal_frac_end
            sc = scaling_base[2] if len(scaling_base) > 2 else 1.0
            self._var_specs.append(
                ('frac_end', x0_frac_end, (fracEndMin, fracEndMax), sc)
            )

        if self.opt_vars.get('fraction_tether_length_traction_end', True):
            x0_frac_start = x0_base[3] if len(x0_base) > 3 else self._nominal_frac_start
            sc = scaling_base[3] if len(scaling_base) > 3 else 1.0
            self._var_specs.append(
                ('frac_start', x0_frac_start, (fracStartMin, fracStartMax), sc)
            )

        # Elevation angle variables are added after checking for full regime 3 in optimize().
        self._optimise_elevation = bool(self.opt_vars.get('elevation_angle_traction', False))
        self._elev_bounds = self.boundsDict['elevation_angle_traction']  # (lo_rad, hi_rad)
        # Elevation variables are stored in DEGREES in the optimizer.
        # x0_base[4..4+N-1] hold the per-angle starting values; x0_base[-1] holds
        # elevation_end_rori when that variable is optimised and embedded in the array.
        # Determine if RORI end elevation is embedded as the last element of x0_base.
        # When elevation_angle_end_trans_rori is optimised AND the x0 array has been
        # extended (len > 5), the last entry encodes that variable's starting value.
        _rori_in_x0 = (
            bool(self.opt_vars.get('elevation_angle_end_trans_rori', False))
            and len(x0_base) > 5
        )
        # Traction elevation entries occupy positions 4..len-2 (or 4..len-1 when RORI is
        # not embedded).  A single value at index 4 is broadcast to all angles.
        _n_trac_in_x0 = len(x0_base) - 4 - (1 if _rori_in_x0 else 0)
        _elev_x0_scalar = float(x0_base[4]) if len(x0_base) > 4 else np.degrees(self._nominal_elevation[0])
        self._elev_x0_deg = [
            float(x0_base[4 + i]) if i < _n_trac_in_x0 else _elev_x0_scalar
            for i in range(self._n_elev)
        ]
        self._elev_scaling = float(scaling_base[4]) if len(scaling_base) > 4 else 1.0
        self._elev_bounds_deg = (
            np.degrees(self._elev_bounds[0]),
            np.degrees(self._elev_bounds[1]),
        )

        # Elevation angle end for RORI transition phase (variable stored in degrees).
        self._nominal_elev_end_rori_deg = np.degrees(
            simulation_settings.get('transition_rori', {}).get('elevation_angle_end', np.deg2rad(50.0))
        )
        if bool(self.opt_vars.get('elevation_angle_end_trans_rori', False)):
            if _rori_in_x0:
                x0_end_rori = float(x0_base[-1])
                sc_end_rori = float(scaling_base[-1]) if len(scaling_base) >= len(x0_base) else 1.0
            else:
                x0_end_rori = float(self.optimizer_config.get(
                    'x0_elevation_angle_end_trans_rori', self._nominal_elev_end_rori_deg
                ))
                sc_end_rori = float(self.optimizer_config.get('scaling_elevation_angle_end_trans_rori', 1.0))
            bounds_end_rori = self.boundsDict.get(
                'elevation_angle_end_trans_rori',
                (np.deg2rad(30.0), np.deg2rad(70.0)),
            )
            bounds_end_rori_deg = (np.degrees(bounds_end_rori[0]), np.degrees(bounds_end_rori[1]))
            self._var_specs.append((
                'elevation_end_rori',
                x0_end_rori,
                bounds_end_rori_deg,
                sc_end_rori,
            ))

        # Coarse time steps used during optimization iterations (from opt_phase_timestep).
        # None values mean: keep the phase setting as-is.
        _opt_ts = self.optimizer_config.get('opt_phase_timestep', {})
        self._opt_time_steps = {
            'retraction': _opt_ts.get('retraction'),
            'transition_riro': _opt_ts.get('transition_riro'),
            'traction': _opt_ts.get('traction'),
            'transition_rori': _opt_ts.get('transition_rori'),
        }

        # Cache for the last evaluated point (avoids double simulation from SLSQP
        # calling the objective and constraint functions at the same x).
        self._cache_x = None
        self._cache_kpi = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self, wind_speed, verbose=False):
        """Run SLSQP optimization for a single wind speed.

        Args:
            wind_speed (float): Reference wind speed [m/s].
            verbose (bool): If True, print a progress line after each iteration.

        Returns:
            dict: KPI dict with the same structure as the direct simulation
                output, plus an ``optimization_result`` key containing the
                ``scipy.optimize.OptimizeResult``.
        """
        self.env_state.set_reference_wind_speed(wind_speed)
        self.history = []
        self._iter_count = 0
        self._cache_x = None
        self._cache_kpi = None

        # Start with base variable specs (no elevation angles yet).
        var_specs = list(self._var_specs)

        # Probe whether elevation angle optimisation is useful.
        if self._optimise_elevation:
            probe_kpi = self._evaluate_from_specs(var_specs)
            traction_n = probe_kpi.get('traction_n_time_points', 1)
            regime3_n = probe_kpi.get('traction_regime3_count', 0)
            full_regime3 = traction_n > 0 and regime3_n >= traction_n
            if full_regime3:
                print(
                    "    Elevation angle optimisation disabled: "
                    "entire traction phase is in regime 3 (power limit active)."
                )
            else:
                for i in range(self._n_elev):
                    var_specs.append((
                        f'elevation_{i}',
                        self._elev_x0_deg[i],  # degrees, per-angle starting point
                        self._elev_bounds_deg,
                        self._elev_scaling,
                    ))

        # If no variables remain active (e.g. all base vars disabled AND regime 3
        # disabled elevation), skip the optimizer and return the nominal result.
        if not var_specs:
            print("    No active optimisation variables — returning nominal simulation.")
            kpi = self._evaluate_from_specs([], use_opt_timesteps=False)
            kpi['optimization_result'] = None
            self.last_var_names = []
            return kpi

        x0 = np.array([spec[1] for spec in var_specs], dtype=float)
        scaling = np.array([spec[3] for spec in var_specs], dtype=float)
        bounds = [spec[2] for spec in var_specs]

        # Scaling convention: x_scaled = x_physical / scaling
        # (divide), so x_physical = x_scaled * scaling.
        # With scaling=1 this is identical to the unscaled case.
        x0_scaled = x0 / scaling
        scaled_bounds = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, scaling)]

        # Build list of variable names for decoding.
        var_names = [spec[0] for spec in var_specs]

        # Indices of frac_end and frac_start for the tether-length constraint.
        idx_frac_end = var_names.index('frac_end') if 'frac_end' in var_names else None
        idx_frac_start = var_names.index('frac_start') if 'frac_start' in var_names else None

        min_frac_diff = self.constraintsDict['min_tether_length_fraction_difference']
        constraints = []
        if idx_frac_end is not None and idx_frac_start is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, ie=idx_frac_end, ist=idx_frac_start, s=scaling: (
                    x[ist] * s[ist] - x[ie] * s[ie] - min_frac_diff
                ),
            })

        # Minimum altitude constraint (traction phase only).
        if self.minimum_height > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, s=scaling, vn=var_names: (
                    self._cached_evaluate(x * s, vn).get('min_altitude_traction', 0.0)
                    - self.minimum_height
                ),
            })

        # Max tether length constraint for RORI transition phase.
        # Enforces that the tether does not exceed the physical maximum during the
        # RORI phase, where the kite is powered and can reel out past frac_start.
        max_tl = self.sys_props.max_tether_length
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, s=scaling, vn=var_names, mtl=max_tl: (
                mtl - self._cached_evaluate(x * s, vn).get('max_tether_length_rori', 0.0)
            ),
        })

        # Maximum step difference between consecutive elevation angles.
        max_diff_elev = self.constraintsDict.get('max_difference_elevation_angle_steps')
        if max_diff_elev is not None and max_diff_elev > 0:
            # Only traction elevation variables have names of the form 'elevation_<int>'.
            elev_indices = [i for i, n in enumerate(var_names)
                            if n.startswith('elevation_') and n.split('_')[1].isdigit()]
            elev_sc = self._elev_scaling
            for k in range(len(elev_indices) - 1):
                j0 = elev_indices[k]
                j1 = elev_indices[k + 1]
                # elevation_{k+1} - elevation_k <= max_diff_elev
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, j0=j0, j1=j1, sc=elev_sc, md=max_diff_elev: (
                        md - (x[j1] * sc - x[j0] * sc)
                    ),
                })
                # elevation_k - elevation_{k+1} <= max_diff_elev
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, j0=j0, j1=j1, sc=elev_sc, md=max_diff_elev: (
                        md + (x[j1] * sc - x[j0] * sc)
                    ),
                })

        # Constraint: elevation_end_rori >= all traction elevation angles.
        # This ensures the RORI transition always goes upward (traction → retraction).
        idx_end_rori = var_names.index('elevation_end_rori') if 'elevation_end_rori' in var_names else None
        if idx_end_rori is not None:
            sc_end_rori = scaling[idx_end_rori]
            trac_elev_indices = [i for i, n in enumerate(var_names)
                                 if n.startswith('elevation_') and n.split('_')[1].isdigit()]
            if trac_elev_indices:
                # Constrain against each optimised traction elevation angle.
                for idx_elev in trac_elev_indices:
                    sc_elev = scaling[idx_elev]
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, ie=idx_end_rori, je=idx_elev, se=sc_end_rori, st=sc_elev: (
                            x[ie] * se - x[je] * st
                        ),
                    })
            else:
                # Traction elevation not optimised: constrain against nominal maximum.
                nominal_max_elev_deg = float(np.degrees(np.max(self._nominal_elevation)))
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, ie=idx_end_rori, se=sc_end_rori, nmax=nominal_max_elev_deg: (
                        x[ie] * se - nmax
                    ),
                })

        def _objective_scaled(x_scaled):
            return self._objective(x_scaled * scaling, var_names)

        def _callback(xk):
            self._iter_count += 1
            x_unscaled = xk * scaling
            current_power = self.history[-1]['power'] if self.history else float('nan')
            if verbose:
                base_str = (
                    f"      iter {self._iter_count:3d} | "
                    f"power={current_power / 1000:+.3f} kW"
                )
                for name, val in zip(var_names, x_unscaled):
                    base_str += f"  {name}={val:+.4f}"
                print(base_str)

        result = op.minimize(
            _objective_scaled,
            x0_scaled,
            method='SLSQP',
            bounds=scaled_bounds,
            constraints=constraints,
            callback=_callback,
            options={
                'maxiter': self.optimizer_config['max_iterations'],
                'ftol': self.optimizer_config['ftol'],
                'eps': self.optimizer_config['eps'],
                'disp': False,
            },
        )

        x_opt = result.x * scaling
        self.last_x_opt = x_opt
        # Final evaluation uses the full-resolution phase time steps so that
        # time histories and KPIs are computed at the same fidelity as a direct
        # simulation, not at the coarser steps used during optimization.
        print("    Re-evaluating optimal solution with full-resolution time steps...")
        kpi = self._evaluate_from_names(x_opt, var_names, use_opt_timesteps=False)
        kpi['optimization_result'] = result
        self.last_var_names = var_names
        return kpi

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _objective(self, x_unscaled, var_names):
        """Negative average cycle power (SLSQP minimises).

        Args:
            x_unscaled (np.ndarray): Unscaled decision variables.
            var_names (list): Variable name list matching x_unscaled.

        Returns:
            float: Negative cycle power or large penalty on failure.
        """
        kpi = self._cached_evaluate(x_unscaled, var_names)
        power = kpi['average_power']['cycle']
        if not kpi['sim_successful']:
            power = -self.PENALTY
        self.history.append({'x': x_unscaled.copy(), 'power': power})
        return -power

    def _cached_evaluate(self, x_unscaled, var_names):
        """Return evaluation result, re-using cache when x is unchanged.

        Args:
            x_unscaled (np.ndarray): Unscaled decision variables.
            var_names (list): Variable name list matching x_unscaled.

        Returns:
            dict: KPI dict.
        """
        if self._cache_x is not None and np.allclose(x_unscaled, self._cache_x, rtol=0, atol=1e-12):
            return self._cache_kpi
        kpi = self._evaluate_from_names(x_unscaled, var_names)
        self._cache_x = x_unscaled.copy()
        self._cache_kpi = kpi
        return kpi

    def _build_settings(self, values_by_name, use_opt_timesteps=True):
        """Build a settings dict with decision variables applied.

        Args:
            values_by_name (dict): Mapping from variable name to value.
            use_opt_timesteps (bool): When True, override phase time steps with
                the coarser ``opt_phase_timestep`` values to speed up optimizer
                iterations. When False, the original phase time steps are kept
                for a full-resolution evaluation. Defaults to True.

        Returns:
            dict: Deep-copied simulation settings with overrides applied.
        """
        settings = deepcopy(self.simulation_settings)
        maxTetherLength = self.sys_props.max_tether_length

        rs_out = values_by_name.get('reeling_speed_out', self._nominal_rs_out)
        rs_in = values_by_name.get('reeling_speed_in', self._nominal_rs_in)
        frac_end = values_by_name.get('frac_end', self._nominal_frac_end)
        frac_start = values_by_name.get('frac_start', self._nominal_frac_start)

        settings['traction']['control'] = ('reeling_speed', float(rs_out))
        settings['retraction']['control'] = ('reeling_speed', float(rs_in))
        settings['cycle']['tether_length_end_retraction'] = frac_end * maxTetherLength
        settings['cycle']['tether_length_end_traction'] = frac_start * maxTetherLength

        # Apply coarser time steps during optimizer iterations.
        if use_opt_timesteps:
            for phase in ('retraction', 'transition_riro', 'traction', 'transition_rori'):
                ts = self._opt_time_steps.get(phase)
                if ts is not None:
                    settings[phase]['time_step'] = ts

        # Override elevation angles if any elevation variable is in values_by_name.
        # Optimizer elevation variables are in degrees; convert to radians for simulation.
        # Only match traction-angle variables of the form 'elevation_<int>'.
        elev_keys = [k for k in values_by_name if k.startswith('elevation_') and k.split('_')[1].isdigit()]
        if elev_keys:
            elev_array = np.array(self._nominal_elevation, dtype=float)  # radians
            for k in elev_keys:
                idx = int(k.split('_')[1])
                elev_array[idx] = np.deg2rad(values_by_name[k])  # degrees → radians
            settings['cycle']['elevation_angle_traction'] = elev_array

        # Apply RORI transition end elevation angle if optimised (stored in degrees).
        elev_end_rori = values_by_name.get('elevation_end_rori')
        if elev_end_rori is not None:
            settings['transition_rori']['elevation_angle_end'] = np.deg2rad(float(elev_end_rori))

        settings['cycle']['traction_phase'] = TractionPhase
        return settings

    def _evaluate_from_specs(self, var_specs, use_opt_timesteps=True):
        """Evaluate at the x0 values of the given var_specs list.

        Args:
            var_specs (list): List of (name, x0, bounds, scaling) tuples.
            use_opt_timesteps (bool): Passed through to ``_evaluate_from_names``.
                Defaults to True.

        Returns:
            dict: KPI dict.
        """
        x0 = np.array([s[1] for s in var_specs], dtype=float)
        names = [s[0] for s in var_specs]
        return self._evaluate_from_names(x0, names, use_opt_timesteps=use_opt_timesteps)

    def _evaluate_from_names(self, x_unscaled, var_names, use_opt_timesteps=True):
        """Run one cycle simulation for the given decision variable vector.

        Args:
            x_unscaled (np.ndarray): Unscaled decision variables.
            var_names (list): Variable names matching x_unscaled.
            use_opt_timesteps (bool): When True, override phase time steps with
                the coarser opt values. Defaults to True.

        Returns:
            dict: KPI dict compatible with ``_build_wind_speed_entry``.
        """
        values_by_name = dict(zip(var_names, x_unscaled))
        settings = self._build_settings(values_by_name, use_opt_timesteps=use_opt_timesteps)
        cycle = Cycle(settings, impose_operational_limits=True)
        steady_state_config = self.simulation_settings.get('steady_state')

        try:
            error_in_phase, _ = cycle.run_simulation(
                self.sys_props, self.env_state, steady_state_config,
                print_summary=False,
            )

            traction = cycle.traction_phase
            retraction = cycle.retraction_phase
            transition_riro = cycle.transition_riro_phase
            transition_rori = cycle.transition_rori_phase

            # Minimum altitude during traction.
            min_altitude = (
                min(k.z for k in traction.kinematics)
                if traction.kinematics else 0.0
            )

            # Maximum tether length reached during RORI transition phase.
            max_tether_rori = (
                max(k.straight_tether_length for k in transition_rori.kinematics)
                if transition_rori is not None and transition_rori.kinematics
                else 0.0
            )

            # Regime 3 statistics for traction phase.
            traction_n = len(traction.steady_states)
            traction_regime3 = getattr(traction, 'regime3_count', 0)

            return {
                'sim_successful': error_in_phase is None,
                'average_power': {
                    'cycle': cycle.average_power if error_in_phase is None else 0.0,
                    'in': retraction.average_power if retraction else 0.0,
                    'trans_riro': transition_riro.average_power if transition_riro else 0.0,
                    'trans_rori': transition_rori.average_power if transition_rori else 0.0,
                    'out': traction.average_power if traction else 0.0,
                },
                'duration': {
                    'cycle': cycle.duration if error_in_phase is None else 0.0,
                    'in': retraction.duration if retraction else 0.0,
                    'trans_riro': transition_riro.duration if transition_riro else 0.0,
                    'trans_rori': transition_rori.duration if transition_rori else 0.0,
                    'out': traction.duration if traction else 0.0,
                },
                'time': cycle.time,
                'kinematics': cycle.kinematics,
                'steady_states': cycle.steady_states,
                'min_altitude_traction': min_altitude,
                'max_tether_length_rori': max_tether_rori,
                'traction_n_time_points': traction_n,
                'traction_regime3_count': traction_regime3,
            }

        except Exception:
            return {
                'sim_successful': False,
                'average_power': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0, 'trans_rori': 0.0, 'out': 0.0},
                'duration': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0, 'trans_rori': 0.0, 'out': 0.0},
                'min_altitude_traction': 0.0,
                'max_tether_length_rori': 0.0,
                'traction_n_time_points': 0,
                'traction_regime3_count': 0,
            }
