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

    Optimizes reeling speeds, tether lengths, traction elevation angles,
    and the RORI transition end elevation angle to maximize average cycle
    power while respecting operational bounds.

    The active decision variable vector depends on ``optimize_variables`` in
    the simulation settings.  The variables are:
        [reeling_speed_out, reeling_speed_in,
         fraction_tether_length_retraction_end,
         fraction_tether_length_traction_end,
         elevation_0, ..., elevation_{N-1},
         elevation_end_rori]

    Traction elevation angles are included when ``elevation_angle_traction``
    is enabled and regime 3 does not dominate the full traction phase.
    ``elevation_end_rori`` is included when ``elevation_angle_end_trans_rori``
    is enabled.  All elevation variables are stored in degrees inside the
    optimizer and converted to radians for simulation.

    Args:
        simulation_settings (dict): Full simulation settings dict (as returned
            by ``load_system_and_simulation_settings``).
        sys_props (SystemProperties): System properties object.
        env_state (NormalisedWindTable1D): Environment state with wind profile.

    Attributes:
        history (list): List of dicts recording every objective evaluation,
            each with keys ``x`` (variable vector), ``power`` (cycle power [W]),
            and ``feasible`` (bool).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, simulation_settings, sys_props, env_state):
        self.simulation_settings = simulation_settings
        self.sys_props = sys_props
        self.env_state = env_state
        self.history = []

        opt_config = simulation_settings['optimization']
        self.optimizer_config = opt_config['optimizer']
        self.bounds_dict = opt_config['bounds']
        self.constraints_dict = opt_config['constraints']
        self.opt_vars = self.optimizer_config.get('optimize_variables', {})
        self.minimum_height = simulation_settings.get('cycle', {}).get('minimum_height', 0.0)

        max_tether_length = sys_props.max_tether_length
        self._max_tether_length = max_tether_length

        # --- Nominal (fixed) values from settings ---
        self._nominal_rs_out = float(simulation_settings['traction']['control'][1])
        self._nominal_rs_in = float(simulation_settings['retraction']['control'][1])
        self._nominal_frac_end = (
            simulation_settings['cycle']['tether_length_end_retraction'] / max_tether_length
        )
        self._nominal_frac_start = (
            simulation_settings['cycle']['tether_length_end_traction'] / max_tether_length
        )
        self._nominal_elevation = np.asarray(
            simulation_settings['cycle']['elevation_angle_traction']
        ).flatten()  # radians
        self._n_elev = self._nominal_elevation.size
        self._nominal_elev_end_rori_deg = np.degrees(
            simulation_settings.get('transition_rori', {}).get(
                'elevation_angle_end', np.deg2rad(50.0)
            )
        )

        # --- Parse optimizer arrays ---
        x0_base = np.array(self.optimizer_config['x0'], dtype=float)
        scaling_base = np.array(self.optimizer_config['scaling'], dtype=float)

        # --- Build base variable specs (no elevation angles yet) ---
        self._base_var_specs = self._build_base_var_specs(
            x0_base, scaling_base, max_tether_length,
        )

        # --- Parse elevation angle config ---
        self._optimise_elevation = bool(self.opt_vars.get('elevation_angle_traction', False))
        self._elev_bounds_rad = self.bounds_dict['elevation_angle_traction']
        self._elev_bounds_deg = (
            np.degrees(self._elev_bounds_rad[0]),
            np.degrees(self._elev_bounds_rad[1]),
        )
        self._elev_scaling = float(scaling_base[4]) if len(scaling_base) > 4 else 1.0
        self._elev_x0_deg = self._parse_elevation_x0(x0_base)

        # --- Parse RORI end elevation config ---
        self._rori_var_spec = self._parse_rori_var_spec(x0_base, scaling_base)

        # --- Parse coarse time steps for optimizer iterations ---
        opt_ts = self.optimizer_config.get('opt_phase_timestep', {})
        self._opt_time_steps = {
            'retraction': opt_ts.get('retraction'),
            'transition_riro': opt_ts.get('transition_riro'),
            'traction': opt_ts.get('traction'),
            'transition_rori': opt_ts.get('transition_rori'),
        }

        # --- Evaluation cache ---
        self._cache_x = None
        self._cache_kpi = None

    # ------------------------------------------------------------------
    # Variable specification helpers
    # ------------------------------------------------------------------

    def _build_base_var_specs(self, x0_base, scaling_base, max_tether_length):
        """Build variable specs for the four base decision variables.

        Args:
            x0_base (np.ndarray): Raw x0 array from config.
            scaling_base (np.ndarray): Raw scaling array from config.
            max_tether_length (float): Maximum tether length [m].

        Returns:
            list: List of (name, x0, (lo, hi), scaling) tuples.
        """
        specs = []
        fracStartMin = self.bounds_dict['tether_length_traction_end'][0] / max_tether_length
        fracStartMax = self.bounds_dict['tether_length_traction_end'][1] / max_tether_length
        fracEndMin = self.bounds_dict['tether_length_retraction_end'][0] / max_tether_length
        fracEndMax = self.bounds_dict['tether_length_retraction_end'][1] / max_tether_length

        if self.opt_vars.get('reeling_speed_traction', True):
            x0_val = x0_base[0] if len(x0_base) > 0 else self._nominal_rs_out
            sc = scaling_base[0] if len(scaling_base) > 0 else 1.0
            specs.append(('reeling_speed_out', float(x0_val),
                          self.bounds_dict['reeling_speed_out'], float(sc)))

        if self.opt_vars.get('reeling_speed_retraction', True):
            x0_val = x0_base[1] if len(x0_base) > 1 else self._nominal_rs_in
            sc = scaling_base[1] if len(scaling_base) > 1 else 1.0
            specs.append(('reeling_speed_in', float(x0_val),
                          self.bounds_dict['reeling_speed_in'], float(sc)))

        if self.opt_vars.get('fraction_tether_length_retraction_end', True):
            x0_val = x0_base[2] if len(x0_base) > 2 else self._nominal_frac_end
            sc = scaling_base[2] if len(scaling_base) > 2 else 1.0
            specs.append(('frac_end', float(x0_val), (fracEndMin, fracEndMax), float(sc)))

        if self.opt_vars.get('fraction_tether_length_traction_end', True):
            x0_val = x0_base[3] if len(x0_base) > 3 else self._nominal_frac_start
            sc = scaling_base[3] if len(scaling_base) > 3 else 1.0
            specs.append(('frac_start', float(x0_val), (fracStartMin, fracStartMax), float(sc)))

        return specs

    def _parse_elevation_x0(self, x0_base):
        """Parse per-angle starting values for traction elevation from x0_base.

        Args:
            x0_base (np.ndarray): Raw x0 array from config.

        Returns:
            list: Per-angle x0 values in degrees.
        """
        roriInX0 = (
            bool(self.opt_vars.get('elevation_angle_end_trans_rori', False))
            and len(x0_base) > 5
        )
        nTracInX0 = len(x0_base) - 4 - (1 if roriInX0 else 0)
        scalarDeg = float(x0_base[4]) if len(x0_base) > 4 else np.degrees(self._nominal_elevation[0])
        return [
            float(x0_base[4 + i]) if i < nTracInX0 else scalarDeg
            for i in range(self._n_elev)
        ]

    def _parse_rori_var_spec(self, x0_base, scaling_base):
        """Parse the RORI end-elevation variable spec if enabled.

        Args:
            x0_base (np.ndarray): Raw x0 array from config.
            scaling_base (np.ndarray): Raw scaling array from config.

        Returns:
            tuple or None: (name, x0, (lo_deg, hi_deg), scaling) or None.
        """
        if not bool(self.opt_vars.get('elevation_angle_end_trans_rori', False)):
            return None

        roriInX0 = len(x0_base) > 5
        if roriInX0:
            x0_val = float(x0_base[-1])
            sc = float(scaling_base[-1]) if len(scaling_base) >= len(x0_base) else 1.0
        else:
            x0_val = float(self.optimizer_config.get(
                'x0_elevation_angle_end_trans_rori', self._nominal_elev_end_rori_deg))
            sc = float(self.optimizer_config.get('scaling_elevation_angle_end_trans_rori', 1.0))

        bounds_rad = self.bounds_dict.get(
            'elevation_angle_end_trans_rori', (np.deg2rad(30.0), np.deg2rad(70.0)))
        bounds_deg = (np.degrees(bounds_rad[0]), np.degrees(bounds_rad[1]))
        return ('elevation_end_rori', x0_val, bounds_deg, sc)

    def _assemble_var_specs(self, probe_kpi):
        """Assemble the full variable-spec list for this optimisation call.

        Starts from the base specs, conditionally adds elevation angles
        and RORI end elevation, and removes reeling_speed_out when the
        full traction phase is in regime 2.

        Args:
            probe_kpi (dict): KPI from a probe evaluation at nominal x0.

        Returns:
            list: Final variable spec list.
        """
        var_specs = list(self._base_var_specs)

        traction_n = probe_kpi.get('traction_n_time_points', 1)
        regime2_n = probe_kpi.get('traction_regime2_count', 0)
        regime3_n = probe_kpi.get('traction_regime3_count', 0)

        # Elevation angles (traction).
        if self._optimise_elevation:
            full_regime3 = traction_n > 0 and regime3_n >= traction_n
            if full_regime3:
                print("    Elevation angle optimisation disabled: "
                      "entire traction phase is in regime 3 (power limit active).")
            else:
                for i in range(self._n_elev):
                    var_specs.append((
                        f'elevation_{i}',
                        self._elev_x0_deg[i],
                        self._elev_bounds_deg,
                        self._elev_scaling,
                    ))

        # RORI end elevation.
        if self._rori_var_spec is not None:
            var_specs.append(self._rori_var_spec)

        # Disable reel-out speed when full regime 2.
        full_regime2 = traction_n > 0 and regime2_n >= traction_n
        if full_regime2:
            var_specs = [s for s in var_specs if s[0] != 'reeling_speed_out']
            print("    Reeling speed (traction) optimisation disabled: "
                  "entire traction phase is in regime 2 (force-controlled).")

        return var_specs

    # ------------------------------------------------------------------
    # x0 adaptation and repair
    # ------------------------------------------------------------------

    def _adapt_x0(self, x0, var_names, bounds, wind_speed):
        """Adapt the starting point to the current wind speed.

        For cold starts the YAML x0 may specify a reel-out speed that exceeds
        one-third of the wind speed, which would place the kite close to its
        stall condition and produce a poor starting point.  The one-third cap
        is the theoretical Betz-limit reeling factor for maximum power, so any
        reasonable optimum lies below it.  Warm-start values that already lie
        below the cap are passed through unchanged.

        Args:
            x0 (np.ndarray): Unscaled starting point (modified in place).
            var_names (list): Variable names matching x0.
            bounds (list): (lo, hi) tuples.
            wind_speed (float): Reference wind speed [m/s].
        """
        for i, name in enumerate(var_names):
            lo, hi = bounds[i]
            if name == 'reeling_speed_out':
                x0[i] = np.clip(min(x0[i], wind_speed / 3.0), lo, hi)

    def _repair_x0(self, x0, var_names, bounds, wind_speed=None):
        """Try to produce a feasible starting point when x0 fails.

        Args:
            x0 (np.ndarray): Original starting point (unscaled).
            var_names (list): Variable names matching x0.
            bounds (list): (lo, hi) tuples for each variable.
            wind_speed (float, optional): Current wind speed for adaptive scaling.

        Returns:
            np.ndarray or None: Repaired x0 if successful, None otherwise.
        """
        candidates = []
        if wind_speed is not None:
            for rfOut in [0.15, 0.10, 0.25]:
                for rfIn in [0.25, 0.15, 0.40]:
                    rsOut = max(wind_speed * rfOut, 0.5)
                    rsIn = min(-wind_speed * rfIn, -0.5)
                    candidates.append((rsOut, rsIn, 1.0))
                    candidates.append((rsOut, rsIn, 1.3))
        for elevFactor in [1.0, 1.3, 1.6]:
            candidates.append((None, None, elevFactor))

        best_x = None
        best_power = -np.inf

        for rsOutVal, rsInVal, elevFactor in candidates:
            xTry = x0.copy()
            for i, name in enumerate(var_names):
                lo, hi = bounds[i]
                if name == 'reeling_speed_out':
                    xTry[i] = np.clip(rsOutVal if rsOutVal is not None else x0[i] * 0.5, lo, hi)
                elif name == 'reeling_speed_in':
                    xTry[i] = np.clip(rsInVal if rsInVal is not None else x0[i] * 0.5, lo, hi)
                elif name.startswith('elevation_') and name.split('_')[1].isdigit():
                    xTry[i] = np.clip(x0[i] * elevFactor, lo, hi)
            kpi = self._run_cycle(xTry, var_names)
            if kpi['sim_successful'] and kpi['average_power']['cycle'] > best_power:
                best_power = kpi['average_power']['cycle']
                best_x = xTry.copy()
                if best_power > 100.0:
                    break
        return best_x

    # ------------------------------------------------------------------
    # Constraint assembly
    # ------------------------------------------------------------------

    def _build_constraints(self, var_names, scaling):
        """Build the list of SLSQP constraint dicts.

        Args:
            var_names (list): Active variable names.
            scaling (np.ndarray): Scaling vector.

        Returns:
            list: Constraint dicts for ``scipy.optimize.minimize``.
        """
        constraints = []

        # Tether fraction ordering.
        idx_frac_end = var_names.index('frac_end') if 'frac_end' in var_names else None
        idx_frac_start = var_names.index('frac_start') if 'frac_start' in var_names else None
        min_frac_diff = self.constraints_dict['min_tether_length_fraction_difference']
        if idx_frac_end is not None and idx_frac_start is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, ie=idx_frac_end, ist=idx_frac_start, s=scaling: (
                    x[ist] * s[ist] - x[ie] * s[ie] - min_frac_diff
                ),
            })

        # Minimum altitude.
        if self.minimum_height > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, s=scaling, vn=var_names: (
                    self._cached_run_cycle(x * s, vn).get('min_altitude_traction', 0.0)
                    - self.minimum_height
                ),
            })

        # Max tether length during RORI.
        max_tl = self._max_tether_length
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, s=scaling, vn=var_names, mtl=max_tl: (
                mtl - self._cached_run_cycle(x * s, vn).get('max_tether_length_rori', 0.0)
            ),
        })

        # Max step difference between consecutive elevation angles.
        max_diff_elev = self.constraints_dict.get('max_difference_elevation_angle_steps')
        if max_diff_elev is not None and max_diff_elev > 0:
            elev_indices = [i for i, n in enumerate(var_names)
                           if n.startswith('elevation_') and n.split('_')[1].isdigit()]
            elev_sc = self._elev_scaling
            for k in range(len(elev_indices) - 1):
                j0 = elev_indices[k]
                j1 = elev_indices[k + 1]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, j0=j0, j1=j1, sc=elev_sc, md=max_diff_elev: (
                        md - (x[j1] * sc - x[j0] * sc)
                    ),
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, j0=j0, j1=j1, sc=elev_sc, md=max_diff_elev: (
                        md + (x[j1] * sc - x[j0] * sc)
                    ),
                })

        # elevation_end_rori >= all traction elevation angles.
        idx_end_rori = var_names.index('elevation_end_rori') if 'elevation_end_rori' in var_names else None
        if idx_end_rori is not None:
            sc_end_rori = scaling[idx_end_rori]
            trac_elev_indices = [i for i, n in enumerate(var_names)
                                 if n.startswith('elevation_') and n.split('_')[1].isdigit()]
            if trac_elev_indices:
                for idx_elev in trac_elev_indices:
                    sc_elev = scaling[idx_elev]
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, ie=idx_end_rori, je=idx_elev, se=sc_end_rori, st=sc_elev: (
                            x[ie] * se - x[je] * st
                        ),
                    })
            else:
                nominal_max_elev_deg = float(np.degrees(np.max(self._nominal_elevation)))
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, ie=idx_end_rori, se=sc_end_rori, nmax=nominal_max_elev_deg: (
                        x[ie] * se - nmax
                    ),
                })

        return constraints

    # ------------------------------------------------------------------
    # Post-optimisation finalisation
    # ------------------------------------------------------------------

    def _finalise_result(self, result, scaling, var_names):
        """Re-evaluate at full resolution and rescue from history if needed.

        Args:
            result (scipy.optimize.OptimizeResult): SLSQP result.
            scaling (np.ndarray): Scaling vector.
            var_names (list): Active variable names.

        Returns:
            tuple: (kpi dict, optimal x vector).
        """
        x_opt = result.x * scaling

        print("    Re-evaluating optimal solution with full-resolution time steps...")
        kpi = self._run_cycle(x_opt, var_names, use_opt_timesteps=False)
        slsqp_power = kpi['average_power']['cycle']

        # Always evaluate the best-of-history point at full resolution.
        # Coarse-timestep history powers are not directly comparable to the
        # full-resolution SLSQP result, so we never screen on the coarse power.
        best_hist = max(
            (e for e in self.history if e.get('feasible', False)),
            key=lambda e: e['power'],
            default=None,
        )
        if best_hist is not None and not np.allclose(
            best_hist['x'], x_opt, rtol=0, atol=1e-6
        ):
            kpi_hist = self._run_cycle(best_hist['x'], var_names, use_opt_timesteps=False)
            if (kpi_hist['sim_successful']
                    and kpi_hist['average_power']['cycle'] > slsqp_power):
                gain = kpi_hist['average_power']['cycle'] - slsqp_power
                print(f"    Rescue: best-of-history yields "
                      f"{kpi_hist['average_power']['cycle']:.1f}W "
                      f"(+{gain:.1f}W over SLSQP result)")
                x_opt = best_hist['x']
                kpi = kpi_hist

        return kpi, x_opt

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self, wind_speed, verbose=False):
        """Run the SLSQP optimisation for a given wind speed.

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

        # Probe at nominal x0.
        probe_kpi = self._run_cycle_from_specs(self._base_var_specs)

        # Assemble final variable list (adds elevation, RORI; drops regime-2 vars).
        var_specs = self._assemble_var_specs(probe_kpi)

        if not var_specs:
            print("    No active optimisation variables -- returning nominal simulation.")
            kpi = self._run_cycle_from_specs([], use_opt_timesteps=False)
            kpi['optimization_result'] = None
            self.last_var_names = []
            return kpi

        x0 = np.array([s[1] for s in var_specs], dtype=float)
        scaling = np.array([s[3] for s in var_specs], dtype=float)
        bounds = [s[2] for s in var_specs]
        var_names = [s[0] for s in var_specs]

        # Adapt x0 to wind speed.
        self._adapt_x0(x0, var_names, bounds, wind_speed)

        # Scale.
        x0_scaled = x0 / scaling
        scaled_bounds = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, scaling)]

        # Pre-check and repair.
        kpi_x0 = self._run_cycle(x0, var_names)
        if not kpi_x0['sim_successful'] or kpi_x0['average_power']['cycle'] < 10.0:
            x0_repaired = self._repair_x0(x0, var_names, bounds, wind_speed)
            if x0_repaired is not None:
                x0 = x0_repaired
                x0_scaled = x0 / scaling
                print("    Repaired infeasible/low-power x0 for this wind speed.")

        constraints = self._build_constraints(var_names, scaling)

        def _objective_scaled(x_scaled):
            return self._objective(x_scaled * scaling, var_names)

        def _callback(xk):
            self._iter_count += 1
            if verbose:
                x_unscaled = xk * scaling
                power = self.history[-1]['power'] if self.history else float('nan')
                parts = [f"      iter {self._iter_count:3d} | power={power / 1000:+.3f} kW"]
                for name, val in zip(var_names, x_unscaled):
                    parts.append(f"  {name}={val:+.4f}")
                print("".join(parts))

        # eps is used as the finite-difference step size in scaled variable space.
        # All variables are O(1) in scaled space (scaling values normalise them),
        # so using base_eps directly gives consistent perturbations for all vars:
        #   reel speeds / fractions: ~eps m/s or fraction
        #   elevation angles: ~eps * 30 degrees
        eps = float(self.optimizer_config['eps'])

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
                'eps': eps,
                'disp': False,
            },
        )

        kpi, x_opt = self._finalise_result(result, scaling, var_names)
        kpi['optimization_result'] = result
        self.last_x_opt = x_opt
        self.last_var_names = var_names
        return kpi

    # ------------------------------------------------------------------
    # Objective and caching
    # ------------------------------------------------------------------

    def _objective(self, x_unscaled, var_names):
        """Negative average cycle power (SLSQP minimises).

        Failed simulations return 0 instead of a large penalty so that
        finite-difference gradients near the feasibility boundary stay
        well-behaved.

        Args:
            x_unscaled (np.ndarray): Unscaled decision variables.
            var_names (list): Variable name list matching x_unscaled.

        Returns:
            float: Negative cycle power (0 for failed simulations).
        """
        kpi = self._cached_run_cycle(x_unscaled, var_names)
        is_feasible = kpi['sim_successful']
        power = kpi['average_power']['cycle'] if is_feasible else 0.0
        self.history.append({
            'x': x_unscaled.copy(), 'power': power, 'feasible': is_feasible,
        })
        return -power

    def _cached_run_cycle(self, x_unscaled, var_names):
        """Return evaluation result, re-using cache when x is unchanged.

        Args:
            x_unscaled (np.ndarray): Unscaled decision variables.
            var_names (list): Variable name list matching x_unscaled.

        Returns:
            dict: KPI dict.
        """
        if (self._cache_x is not None
                and np.allclose(x_unscaled, self._cache_x, rtol=0, atol=1e-12)):
            return self._cache_kpi
        kpi = self._run_cycle(x_unscaled, var_names)
        self._cache_x = x_unscaled.copy()
        self._cache_kpi = kpi
        return kpi

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _build_settings(self, values_by_name, use_opt_timesteps=True):
        """Build a settings dict with decision variables applied.

        Args:
            values_by_name (dict): Mapping from variable name to value.
            use_opt_timesteps (bool): When True, override phase time steps with
                the coarser values. Defaults to True.

        Returns:
            dict: Deep-copied simulation settings with overrides applied.
        """
        settings = deepcopy(self.simulation_settings)
        max_tl = self._max_tether_length

        rs_out = values_by_name.get('reeling_speed_out', self._nominal_rs_out)
        rs_in = values_by_name.get('reeling_speed_in', self._nominal_rs_in)
        frac_end = values_by_name.get('frac_end', self._nominal_frac_end)
        frac_start = values_by_name.get('frac_start', self._nominal_frac_start)

        settings['traction']['control'] = ('reeling_speed', float(rs_out))
        settings['retraction']['control'] = ('reeling_speed', float(rs_in))
        settings['cycle']['tether_length_end_retraction'] = frac_end * max_tl
        settings['cycle']['tether_length_end_traction'] = frac_start * max_tl

        if use_opt_timesteps:
            for phase in ('retraction', 'transition_riro', 'traction', 'transition_rori'):
                ts = self._opt_time_steps.get(phase)
                if ts is not None:
                    settings[phase]['time_step'] = ts

        # Elevation angles (degrees -> radians).
        elev_keys = [k for k in values_by_name
                     if k.startswith('elevation_') and k.split('_')[1].isdigit()]
        if elev_keys:
            elev_array = np.array(self._nominal_elevation, dtype=float)
            for k in elev_keys:
                idx = int(k.split('_')[1])
                elev_array[idx] = np.deg2rad(values_by_name[k])
            settings['cycle']['elevation_angle_traction'] = elev_array

        elev_end_rori = values_by_name.get('elevation_end_rori')
        if elev_end_rori is not None:
            settings['transition_rori']['elevation_angle_end'] = np.deg2rad(float(elev_end_rori))

        settings['cycle']['traction_phase'] = TractionPhase
        return settings

    def _run_cycle_from_specs(self, var_specs, use_opt_timesteps=True):
        """Evaluate at the x0 values of the given var_specs list.

        Args:
            var_specs (list): List of (name, x0, bounds, scaling) tuples.
            use_opt_timesteps (bool): Passed through to ``_run_cycle``.

        Returns:
            dict: KPI dict.
        """
        x0 = np.array([s[1] for s in var_specs], dtype=float)
        names = [s[0] for s in var_specs]
        return self._run_cycle(x0, names, use_opt_timesteps=use_opt_timesteps)

    def _run_cycle(self, x_unscaled, var_names, use_opt_timesteps=True):
        """Run one cycle simulation for the given decision variable vector.

        Args:
            x_unscaled (np.ndarray): Unscaled decision variables.
            var_names (list): Variable names matching x_unscaled.
            use_opt_timesteps (bool): When True, use coarser time steps.

        Returns:
            dict: KPI dict.
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

            min_altitude = (
                min(k.z for k in traction.kinematics)
                if traction.kinematics else 0.0
            )
            altitude_ok = (self.minimum_height <= 0 or min_altitude >= self.minimum_height)

            apparent_wind_ok = not any(
                ss.apparent_wind_speed is not None and ss.apparent_wind_speed < 0
                for ss in traction.steady_states
            ) if traction.steady_states else True

            cycle_ok = error_in_phase is None and altitude_ok and apparent_wind_ok

            max_tether_rori = (
                max(k.straight_tether_length for k in transition_rori.kinematics)
                if transition_rori is not None and transition_rori.kinematics
                else 0.0
            )

            traction_n = len(traction.steady_states)
            traction_regime2 = getattr(traction, 'regime2_count', 0)
            traction_regime3 = getattr(traction, 'regime3_count', 0)

            return {
                'sim_successful': cycle_ok,
                'average_power': {
                    'cycle': cycle.average_power if cycle_ok else 0.0,
                    'in': retraction.average_power if retraction else 0.0,
                    'trans_riro': transition_riro.average_power if transition_riro else 0.0,
                    'trans_rori': transition_rori.average_power if transition_rori else 0.0,
                    'out': traction.average_power if traction else 0.0,
                },
                'duration': {
                    'cycle': cycle.duration if cycle_ok else 0.0,
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
                'traction_regime2_count': traction_regime2,
                'traction_regime3_count': traction_regime3,
            }

        except Exception:
            return {
                'sim_successful': False,
                'average_power': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0,
                                  'trans_rori': 0.0, 'out': 0.0},
                'duration': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0,
                             'trans_rori': 0.0, 'out': 0.0},
                'min_altitude_traction': 0.0,
                'max_tether_length_rori': 0.0,
                'traction_n_time_points': 0,
                'traction_regime2_count': 0,
                'traction_regime3_count': 0,
            }
