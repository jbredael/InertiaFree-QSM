"""Cycle optimizer for AWE pumping cycle power optimization.

This module provides SLSQP-based optimization for pumping cycle parameters
to maximize power output while respecting operational constraints.
"""

from copy import deepcopy

import numpy as np
from scipy import optimize as op

from .qsm import Cycle, TractionPhase, build_cycle_energy_power_kpis


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

    _ALTITUDE_TOLERANCE = 5e-2  # m; absorbs timestep/interpolation roundoff at active constraints.
    _TETHER_LENGTH_TOLERANCE = 5e-2  # m; same for RORI headroom at the drum limit.

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
        self._cache_use_opt_timesteps = None
        self._cache_kpi = None

        # Deterministic multi-start settings.  A small number of extra starts
        # is usually cheaper than accepting an isolated local optimum in the
        # power curve and having to rerun that wind speed manually.
        self._multi_start_enabled = bool(
            self.optimizer_config.get('multi_start_enabled', False)
        )
        self._multi_start_max_starts = int(
            self.optimizer_config.get('multi_start_max_starts', 2)
        )
        self._multi_start_max_candidates = int(
            self.optimizer_config.get('multi_start_max_candidates', 10)
        )

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

        # Disable reel-out speed when full regime 2 or full regime 3.
        # In both regimes the traction reeling speed is determined by the
        # force/power limit control law, not by the speed setpoint.
        full_regime2 = traction_n > 0 and regime2_n >= traction_n
        full_regime3 = traction_n > 0 and regime3_n >= traction_n
        if full_regime2 or full_regime3:
            var_specs = [s for s in var_specs if s[0] != 'reeling_speed_out']
            reason = 'force-controlled' if full_regime2 else 'power-limited'
            print(f"    Reeling speed (traction) optimisation disabled: "
                  f"entire traction phase is in regime {'2' if full_regime2 else '3'} ({reason}).")

        return var_specs

    # ------------------------------------------------------------------
    # x0 adaptation and repair
    # ------------------------------------------------------------------

    def _adapt_x0(self, x0, var_names, bounds, wind_speed):
        """Adapt the starting point to the current wind speed.

        Reeling-out speed is capped at wind_speed/3 (Betz-limit reeling factor).
        Reeling-in speed is soft-adapted at high wind: if the starting value is
        much slower than a wind-speed-proportional estimate (typical optimum is
        ~0.27 × wind_speed), override it.  This prevents cold starts from placing
        SLSQP far from the optimum when the warm-start chain was broken.
        Warm-started values that are already in a reasonable range pass through
        unchanged.

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
            elif name == 'reeling_speed_in':
                # Cold-start YAML default is -2 m/s; at high wind the optimum is
                # roughly -0.27 × wind_speed.  Only override when the current
                # value is less than half the estimated optimum speed (magnitude).
                target = np.clip(-wind_speed * 0.27, lo, hi)
                if abs(x0[i]) < abs(target) * 0.5:
                    x0[i] = target

    def _repair_x0(self, x0, var_names, bounds, wind_speed=None):
        """Try to produce a feasible starting point when x0 fails.

        Two-pass strategy:

        Pass 1 — sweep ``elevation_end_rori`` through the full variable range
        (5° steps) while keeping all other variables at x0.  This handles the
        common cold-start failure where the YAML default (50°) lies outside the
        feasibility window (e.g. ~68-75° at high wind).

        Pass 2 — sweep wind-speed-proportional ``reeling_speed_in`` /
        ``reeling_speed_out`` candidates, starting from the best point found in
        Pass 1 (or x0 if Pass 1 found nothing).  The existing elevation-factor
        sweep also runs here to cover medium-wind cases.

        Args:
            x0 (np.ndarray): Original starting point (unscaled).
            var_names (list): Variable names matching x0.
            bounds (list): (lo, hi) tuples for each variable.
            wind_speed (float, optional): Current wind speed for adaptive scaling.

        Returns:
            np.ndarray or None: Repaired x0 if successful, None otherwise.
        """
        best_x = None
        best_power = -np.inf

        # --- Pass 1: sweep elevation_end_rori across full range in 5° steps ---
        # Locates the feasibility window when the initial rori value is wrong.
        rori_idx = next(
            (i for i, n in enumerate(var_names) if n == 'elevation_end_rori'), None
        )
        if rori_idx is not None:
            lo_r, hi_r = bounds[rori_idx]
            for rori_val in np.arange(lo_r + 2.5, hi_r, 5.0):
                xTry = x0.copy()
                xTry[rori_idx] = float(rori_val)
                kpi = self._run_cycle(xTry, var_names)
                if kpi['sim_successful'] and kpi['average_power']['cycle'] > best_power:
                    best_power = kpi['average_power']['cycle']
                    best_x = xTry.copy()

        # --- Pass 2: sweep rs_in / rs_out / elevation factor candidates ---
        # Use the best rori from Pass 1 as the base so Pass 2 searches within
        # a known-feasible rori neighbourhood.
        base = best_x if best_x is not None else x0

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

        for rsOutVal, rsInVal, elevFactor in candidates:
            xTry = base.copy()
            for i, name in enumerate(var_names):
                lo, hi = bounds[i]
                if name == 'reeling_speed_out':
                    xTry[i] = np.clip(rsOutVal if rsOutVal is not None else base[i] * 0.5, lo, hi)
                elif name == 'reeling_speed_in':
                    xTry[i] = np.clip(rsInVal if rsInVal is not None else base[i] * 0.5, lo, hi)
                elif name.startswith('elevation_') and name.split('_')[1].isdigit():
                    xTry[i] = np.clip(base[i] * elevFactor, lo, hi)
            kpi = self._run_cycle(xTry, var_names)
            if kpi['sim_successful'] and kpi['average_power']['cycle'] > best_power:
                best_power = kpi['average_power']['cycle']
                best_x = xTry.copy()
                if best_power > 100.0:
                    break
        return best_x

    def _reset_cache(self):
        """Clear the one-point simulation cache used by objective/constraints."""
        self._cache_x = None
        self._cache_use_opt_timesteps = None
        self._cache_kpi = None

    @staticmethod
    def _is_elevation_var(name):
        parts = name.split('_')
        return name.startswith('elevation_') and len(parts) > 1 and parts[1].isdigit()

    def _project_candidate(self, x, var_names, bounds):
        """Clip a candidate to bounds and simple algebraic constraints."""
        x = np.asarray(x, dtype=float).copy()
        for i, (lo, hi) in enumerate(bounds):
            x[i] = np.clip(x[i], lo, hi)

        idx_frac_end = var_names.index('frac_end') if 'frac_end' in var_names else None
        idx_frac_start = var_names.index('frac_start') if 'frac_start' in var_names else None
        if idx_frac_end is not None and idx_frac_start is not None:
            min_diff = self.constraints_dict['min_tether_length_fraction_difference']
            if x[idx_frac_start] - x[idx_frac_end] < min_diff:
                lo_end, _ = bounds[idx_frac_end]
                _, hi_start = bounds[idx_frac_start]
                x[idx_frac_end] = min(x[idx_frac_end], x[idx_frac_start] - min_diff)
                x[idx_frac_end] = max(x[idx_frac_end], lo_end)
                if x[idx_frac_start] - x[idx_frac_end] < min_diff:
                    x[idx_frac_start] = min(hi_start, x[idx_frac_end] + min_diff)

        elev_indices = [
            i for i, name in enumerate(var_names)
            if self._is_elevation_var(name)
        ]
        max_diff_elev = self.constraints_dict.get('max_difference_elevation_angle_steps')
        if max_diff_elev is not None and max_diff_elev > 0 and len(elev_indices) > 1:
            for i0, i1 in zip(elev_indices[:-1], elev_indices[1:]):
                x[i1] = np.clip(x[i1], x[i0] - max_diff_elev, x[i0] + max_diff_elev)
                lo, hi = bounds[i1]
                x[i1] = np.clip(x[i1], lo, hi)
            for i0, i1 in zip(reversed(elev_indices[:-1]), reversed(elev_indices[1:])):
                x[i0] = np.clip(x[i0], x[i1] - max_diff_elev, x[i1] + max_diff_elev)
                lo, hi = bounds[i0]
                x[i0] = np.clip(x[i0], lo, hi)

        idx_end_rori = (
            var_names.index('elevation_end_rori')
            if 'elevation_end_rori' in var_names else None
        )
        if idx_end_rori is not None:
            if elev_indices:
                lower_limit = max(x[i] for i in elev_indices)
            else:
                lower_limit = float(np.degrees(np.max(self._nominal_elevation)))
            lo, hi = bounds[idx_end_rori]
            x[idx_end_rori] = np.clip(max(x[idx_end_rori], lower_limit), lo, hi)

        return x

    def _build_multistart_candidates(self, x0, var_names, bounds, wind_speed):
        """Create deterministic candidate starts for SLSQP.

        The candidates are deliberately simple and physics-oriented: they vary
        reel speeds, cycle length fractions, traction elevation level, and the
        minimum-height boundary.  They are screened before SLSQP, so generating
        more candidates does not mean all of them become optimizer starts.
        """
        candidates = []
        seen = set()

        def add(x_candidate):
            x_projected = self._project_candidate(x_candidate, var_names, bounds)
            key = tuple(np.round(x_projected, 8))
            if key not in seen:
                seen.add(key)
                candidates.append(x_projected)

        base = self._project_candidate(x0, var_names, bounds)
        add(base)

        idx_rs_out = (
            var_names.index('reeling_speed_out')
            if 'reeling_speed_out' in var_names else None
        )
        idx_rs_in = (
            var_names.index('reeling_speed_in')
            if 'reeling_speed_in' in var_names else None
        )
        idx_frac_end = var_names.index('frac_end') if 'frac_end' in var_names else None
        idx_frac_start = var_names.index('frac_start') if 'frac_start' in var_names else None
        idx_end_rori = (
            var_names.index('elevation_end_rori')
            if 'elevation_end_rori' in var_names else None
        )
        elev_indices = [
            i for i, name in enumerate(var_names)
            if self._is_elevation_var(name)
        ]

        # Reeling speeds: cover common pumping-cycle ratios around v_w/3.
        for rf_out, rf_in in ((0.20, 0.22), (1.0 / 3.0, 0.27), (0.15, 0.40)):
            x = base.copy()
            if idx_rs_out is not None:
                lo, hi = bounds[idx_rs_out]
                x[idx_rs_out] = np.clip(max(wind_speed * rf_out, 0.5), lo, hi)
            if idx_rs_in is not None:
                lo, hi = bounds[idx_rs_in]
                x[idx_rs_in] = np.clip(min(-wind_speed * rf_in, -0.5), lo, hi)
            add(x)

        # Tether length fractions: small changes can move between local optima
        # when altitude/RORI constraints are active.
        for delta_end, delta_start in ((-0.05, 0.0), (0.03, 0.0),
                                       (0.0, -0.03), (0.0, 0.03),
                                       (-0.07, 0.03)):
            x = base.copy()
            if idx_frac_end is not None:
                x[idx_frac_end] += delta_end
            if idx_frac_start is not None:
                x[idx_frac_start] += delta_start
            add(x)

        # Uniform traction-elevation shifts.
        if elev_indices:
            for delta_deg in (-5.0, -2.0, 2.0, 5.0, 8.0):
                x = base.copy()
                for idx in elev_indices:
                    x[idx] += delta_deg
                add(x)

        # Minimum-height boundary starts.  The optimum often lies very near
        # z_min, where SLSQP finite differences can otherwise step into a
        # failed simulation and get a misleading local gradient.
        if self.minimum_height > 0 and elev_indices:
            frac_end = (
                base[idx_frac_end] if idx_frac_end is not None
                else self._nominal_frac_end
            )
            frac_start = (
                base[idx_frac_start] if idx_frac_start is not None
                else self._nominal_frac_start
            )
            r0 = frac_end * self._max_tether_length
            r1 = frac_start * self._max_tether_length
            if r1 > r0:
                knot_lengths = np.linspace(r0, r1, len(elev_indices))
                for margin_m in (0.5, 3.0, 7.0):
                    x = base.copy()
                    ratios = np.clip(
                        (self.minimum_height + margin_m) / knot_lengths,
                        0.0, 1.0,
                    )
                    floor_angles = np.degrees(np.arcsin(ratios))
                    for idx, elev_floor in zip(elev_indices, floor_angles):
                        x[idx] = elev_floor
                    add(x)

        # RORI end elevation variants; keep it above traction elevation.
        if idx_end_rori is not None:
            for delta_deg in (-5.0, 5.0, 10.0):
                x = base.copy()
                x[idx_end_rori] += delta_deg
                add(x)
            if elev_indices:
                x = base.copy()
                x[idx_end_rori] = max(x[i] for i in elev_indices) + 3.0
                add(x)

        return candidates[:self._multi_start_max_candidates]

    def _select_multistart_points(self, x0, var_names, bounds, wind_speed):
        """Screen candidates and return the starts used for SLSQP."""
        x0 = self._project_candidate(x0, var_names, bounds)
        if not self._multi_start_enabled or self._multi_start_max_starts <= 1:
            return [x0]

        candidates = self._build_multistart_candidates(
            x0, var_names, bounds, wind_speed,
        )
        scored = []
        for idx, x_candidate in enumerate(candidates):
            kpi = self._run_cycle(x_candidate, var_names)
            feasible = kpi.get('sim_successful', False)
            power = kpi['average_power']['cycle'] if feasible else -np.inf
            scored.append((idx, x_candidate, feasible, power))

        selected = [x0]
        selected_keys = {tuple(np.round(x0, 8))}
        feasible_candidates = sorted(
            (item for item in scored[1:] if item[2]),
            key=lambda item: item[3],
            reverse=True,
        )
        fallback_candidates = sorted(
            (item for item in scored[1:] if not item[2]),
            key=lambda item: item[0],
        )

        for _, x_candidate, _, _ in feasible_candidates + fallback_candidates:
            if len(selected) >= self._multi_start_max_starts:
                break
            key = tuple(np.round(x_candidate, 8))
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(x_candidate)

        return selected

    # ------------------------------------------------------------------
    # Constraint assembly
    # ------------------------------------------------------------------

    def _build_constraints(self, var_names, scaling, use_opt_timesteps=True):
        """Build the list of SLSQP constraint dicts.

        Args:
            var_names (list): Active variable names.
            scaling (np.ndarray): Scaling vector.
            use_opt_timesteps (bool): Passed through to cycle evaluations
                used by simulation-based constraints.

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
                'fun': lambda x, s=scaling, vn=var_names, uot=use_opt_timesteps: (
                    self._cached_run_cycle(x * s, vn, use_opt_timesteps=uot)
                    .get('min_altitude_traction', 0.0)
                    - (self.minimum_height - self._ALTITUDE_TOLERANCE)
                ),
            })

        # Max tether length during RORI.
        max_tl = self._max_tether_length
        if max_tl is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, s=scaling, vn=var_names, mtl=max_tl, uot=use_opt_timesteps: (
                    (mtl + self._TETHER_LENGTH_TOLERANCE)
                    - self._cached_run_cycle(x * s, vn, use_opt_timesteps=uot)
                    .get('max_tether_length_rori', 0.0)
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

    def _repair_final_solution(self, x0, var_names, bounds):
        """Try small full-resolution repairs when the returned point is infeasible.

        This is intentionally conservative: it only nudges variables in
        directions that add geometric margin to common active constraints.
        It is used after SLSQP/polish, where the objective may be flat because
        failed cycle evaluations return zero power.

        Args:
            x0 (np.ndarray): Unscaled point to repair.
            var_names (list): Active variable names.
            bounds (list): Unscaled variable bounds.

        Returns:
            tuple: (x_repaired, kpi_repaired), or (x0, failed_kpi).
        """
        base_kpi = self._run_cycle(x0, var_names, use_opt_timesteps=False)
        if base_kpi.get('sim_successful', False):
            return x0, base_kpi

        candidates = []

        def add_candidate(x_candidate):
            candidates.append(np.asarray(x_candidate, dtype=float))

        elev_indices = [
            i for i, n in enumerate(var_names)
            if n.startswith('elevation_') and n.split('_')[1].isdigit()
        ]
        idx_frac_start = (
            var_names.index('frac_start') if 'frac_start' in var_names else None
        )
        idx_frac_end = (
            var_names.index('frac_end') if 'frac_end' in var_names else None
        )
        idx_end_rori = (
            var_names.index('elevation_end_rori')
            if 'elevation_end_rori' in var_names else None
        )

        # Raise the traction elevation schedule slightly. This directly adds
        # altitude margin while preserving the shape of the schedule.
        for delta_deg in (0.25, 0.5, 1.0, 2.0):
            if elev_indices:
                x = x0.copy()
                for idx in elev_indices:
                    lo, hi = bounds[idx]
                    x[idx] = np.clip(x[idx] + delta_deg, lo, hi)
                if idx_end_rori is not None:
                    lo, hi = bounds[idx_end_rori]
                    x[idx_end_rori] = np.clip(
                        max(x[idx_end_rori], max(x[i] for i in elev_indices)),
                        lo, hi,
                    )
                add_candidate(x)

        # Shorten the traction end length slightly. This helps both minimum
        # altitude misses and RORI headroom misses without changing controls.
        if idx_frac_start is not None:
            min_frac_diff = self.constraints_dict['min_tether_length_fraction_difference']
            for delta_frac in (0.0025, 0.005, 0.01, 0.02):
                x = x0.copy()
                lo, hi = bounds[idx_frac_start]
                lower_limit = lo
                if idx_frac_end is not None:
                    lower_limit = max(lower_limit, x[idx_frac_end] + min_frac_diff)
                x[idx_frac_start] = np.clip(x[idx_frac_start] - delta_frac,
                                            lower_limit, hi)
                add_candidate(x)

                if elev_indices:
                    x_raised = x.copy()
                    for idx in elev_indices:
                        lo_e, hi_e = bounds[idx]
                        x_raised[idx] = np.clip(x_raised[idx] + 0.5, lo_e, hi_e)
                    if idx_end_rori is not None:
                        lo_r, hi_r = bounds[idx_end_rori]
                        x_raised[idx_end_rori] = np.clip(
                            max(x_raised[idx_end_rori],
                                max(x_raised[i] for i in elev_indices)),
                            lo_r, hi_r,
                        )
                    add_candidate(x_raised)

        # Lower the RORI end elevation when the transition reels out too far.
        if idx_end_rori is not None:
            rori_min, rori_max = bounds[idx_end_rori]
            lower_limit = rori_min
            if elev_indices:
                lower_limit = max(lower_limit, max(x0[i] for i in elev_indices))
            else:
                lower_limit = max(
                    lower_limit, float(np.degrees(np.max(self._nominal_elevation)))
                )
            for delta_deg in (0.5, 1.0, 2.0, 5.0):
                x = x0.copy()
                x[idx_end_rori] = np.clip(
                    x[idx_end_rori] - delta_deg, lower_limit, rori_max,
                )
                add_candidate(x)

        best_x = x0
        best_kpi = base_kpi
        best_power = -np.inf
        seen = set()
        for x_candidate in candidates:
            key = tuple(np.round(x_candidate, 10))
            if key in seen:
                continue
            seen.add(key)
            kpi = self._run_cycle(
                x_candidate, var_names, use_opt_timesteps=False,
            )
            if (kpi.get('sim_successful', False)
                    and kpi['average_power']['cycle'] > best_power):
                best_x = x_candidate
                best_kpi = kpi
                best_power = kpi['average_power']['cycle']

        if best_kpi.get('sim_successful', False):
            print(f"    Final repair: {best_power:.1f} W "
                  f"(reason was {base_kpi.get('failure_reason')})")

        return best_x, best_kpi

    def _polish(self, x0, var_names, scaling, scaled_bounds, constraints, eps, maxiter):
        """Refine the coarse optimum with a short full-resolution SLSQP pass.

        The coarse-timestep objective used during the main optimisation may have
        a different landscape from the fine-timestep objective used for the final
        evaluation.  Starting a second, short SLSQP run at full resolution from
        the coarse optimum corrects the resulting discrepancy.

        Args:
            x0 (np.ndarray): Unscaled starting point (coarse optimum).
            var_names (list): Active variable names.
            scaling (np.ndarray): Scaling vector.
            scaled_bounds (list): Scaled (lo/s, hi/s) tuples.
            constraints (list): SLSQP constraint dicts.
            eps (float): FD step size in scaled space.
            maxiter (int): Maximum polish iterations.

        Returns:
            tuple: (x_polished unscaled, kpi_polished dict).
        """
        def _obj_fine(x_scaled):
            kpi = self._run_cycle(x_scaled * scaling, var_names, use_opt_timesteps=False)
            return -(kpi['average_power']['cycle'] if kpi['sim_successful'] else 0.0)

        polish_result = op.minimize(
            _obj_fine,
            x0 / scaling,
            method='SLSQP',
            bounds=scaled_bounds,
            constraints=constraints,
            options={
                'maxiter': maxiter,
                'ftol': self.optimizer_config['ftol'],
                'eps': eps,
                'disp': False,
            },
        )
        x_polished = polish_result.x * scaling
        kpi_polished = self._run_cycle(x_polished, var_names, use_opt_timesteps=False)
        return x_polished, kpi_polished

    def _run_slsqp(self, x_start, var_names, scaling, scaled_bounds, constraints, eps,
                   maxiter, callback=None):
        """Run one SLSQP optimization from an unscaled starting point."""
        self._reset_cache()
        return op.minimize(
            lambda x_scaled: self._objective(x_scaled * scaling, var_names),
            x_start / scaling,
            method='SLSQP',
            bounds=scaled_bounds,
            constraints=constraints,
            callback=callback,
            options={
                'maxiter': maxiter,
                'ftol': self.optimizer_config['ftol'],
                'eps': eps,
                'disp': False,
            },
        )

    def _finalise_result(self, result, scaling, var_names):
        kpi, x_opt, _ = self._finalise_best_result([result], scaling, var_names)
        return kpi, x_opt

    def _finalise_best_result(self, results, scaling, var_names):
        """Re-evaluate at full resolution and rescue from history if needed.

        Args:
            results (list): SLSQP results from one or more starts.
            scaling (np.ndarray): Scaling vector.
            var_names (list): Active variable names.

        Returns:
            tuple: (kpi dict, optimal x vector, selected OptimizeResult).
        """
        if not results:
            raise ValueError("At least one SLSQP result is required.")

        msg = "solution" if len(results) == 1 else f"{len(results)} candidate solutions"
        print(f"    Re-evaluating {msg} with full-resolution time steps...")

        best_power = -np.inf
        best_kpi = None
        best_x = None
        best_result = results[0]
        best_source = None
        primary_power = -np.inf

        for i, result in enumerate(results):
            x_candidate = result.x * scaling
            kpi_candidate = self._run_cycle(
                x_candidate, var_names, use_opt_timesteps=False,
            )
            power = (
                kpi_candidate['average_power']['cycle']
                if kpi_candidate.get('sim_successful', False)
                else -np.inf
            )
            if i == 0:
                primary_power = power
            if power > best_power:
                best_power = power
                best_kpi = kpi_candidate
                best_x = x_candidate
                best_result = result
                best_source = f"SLSQP start {i + 1}"

        # Scan the top-N feasible history entries (sorted by coarse power, best
        # first) at full resolution, keeping the best result found.  All N are
        # evaluated rather than stopping at the first feasible one, because the
        # coarse-timestep power ranking can differ from the fine-timestep ranking.
        MAX_RESCUE_ATTEMPTS = int(
            self.optimizer_config.get('final_rescue_attempts', 12)
        )
        feasible_history = sorted(
            (e for e in self.history if e.get('feasible', False)),
            key=lambda e: e['power'],
            reverse=True,
        )
        attempts = 0
        for hist_entry in feasible_history:
            if best_x is not None and np.allclose(hist_entry['x'], best_x, rtol=0, atol=1e-6):
                continue
            attempts += 1
            if attempts > MAX_RESCUE_ATTEMPTS:
                break
            kpi_hist = self._run_cycle(hist_entry['x'], var_names, use_opt_timesteps=False)
            if (kpi_hist['sim_successful']
                    and kpi_hist['average_power']['cycle'] > best_power):
                best_power = kpi_hist['average_power']['cycle']
                best_kpi = kpi_hist
                best_x = hist_entry['x']
                best_source = "history rescue"

        if best_kpi is None:
            best_x = results[0].x * scaling
            best_kpi = self._run_cycle(best_x, var_names, use_opt_timesteps=False)

        if best_power > primary_power + 1e-6:
            gain_base = primary_power if primary_power > -np.inf else 0.0
            print(f"    Multi-start/rescue selected {best_source}: "
                  f"{best_power:.1f} W (+{best_power - gain_base:.1f} W).")

        return best_kpi, best_x, best_result

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
        self._cache_use_opt_timesteps = None
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

        # Scale bounds for SLSQP's internal variable representation.
        scaled_bounds = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, scaling)]

        # Pre-check and repair.
        kpi_x0 = self._run_cycle(x0, var_names)
        if not kpi_x0['sim_successful'] or kpi_x0['average_power']['cycle'] < 10.0:
            x0_repaired = self._repair_x0(x0, var_names, bounds, wind_speed)
            if x0_repaired is not None:
                x0 = x0_repaired
                print("    Repaired infeasible/low-power x0 for this wind speed.")

        constraints = self._build_constraints(var_names, scaling, use_opt_timesteps=True)

        def _callback(xk):
            self._iter_count += 1
            if verbose:
                x_unscaled = xk * scaling
                # Find the history entry for the actual iterate xk rather than
                # using history[-1], which is typically the last FD gradient
                # evaluation point (often at a failing x+eps location) and would
                # show power=0 even when xk itself is feasible.
                power = float('nan')
                for entry in reversed(self.history):
                    if np.allclose(entry['x'], x_unscaled, rtol=0, atol=1e-6):
                        power = entry['power']
                        break
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

        start_points = self._select_multistart_points(
            x0, var_names, bounds, wind_speed,
        )
        if len(start_points) > 1:
            print(f"    Multi-start: {len(start_points)} SLSQP starts selected.")

        results = []
        max_iterations = int(self.optimizer_config['max_iterations'])
        secondary_max_iterations = int(
            self.optimizer_config.get('multi_start_max_iterations', max_iterations)
        )
        for i_start, x_start in enumerate(start_points):
            maxiter = max_iterations if i_start == 0 else secondary_max_iterations
            result_i = self._run_slsqp(
                x_start, var_names, scaling, scaled_bounds, constraints,
                eps, maxiter, callback=_callback,
            )
            results.append(result_i)

        kpi, x_opt, result = self._finalise_best_result(results, scaling, var_names)

        # Polish: short fine-resolution SLSQP pass starting from the coarse
        # optimum.  Corrects for discrepancies between the coarse and fine
        # timestep landscapes so the returned solution is a local optimum of
        # the full-resolution objective.
        polish_maxiter = self.optimizer_config.get('polish_max_iterations', 5)
        if polish_maxiter > 0:
            fine_constraints = self._build_constraints(
                var_names, scaling, use_opt_timesteps=False,
            )
            x_polished, kpi_polished = self._polish(
                x_opt, var_names, scaling, scaled_bounds, fine_constraints,
                eps, polish_maxiter,
            )
            if (kpi_polished['sim_successful']
                    and (not kpi.get('sim_successful', False)
                         or kpi_polished['average_power']['cycle'] > kpi['average_power']['cycle'])):
                gain = (
                    kpi_polished['average_power']['cycle']
                    - (kpi['average_power']['cycle'] if kpi.get('sim_successful', False) else 0.0)
                )
                print(f"    Polish: {kpi_polished['average_power']['cycle']:.1f} W "
                      f"(+{gain:.1f} W)")
                x_opt = x_polished
                kpi = kpi_polished

        if not kpi.get('sim_successful', False):
            x_repaired, kpi_repaired = self._repair_final_solution(
                x_opt, var_names, bounds,
            )
            if kpi_repaired.get('sim_successful', False):
                x_opt = x_repaired
                kpi = kpi_repaired

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

    def _cached_run_cycle(self, x_unscaled, var_names, use_opt_timesteps=True):
        """Return evaluation result, re-using cache when x is unchanged.

        Args:
            x_unscaled (np.ndarray): Unscaled decision variables.
            var_names (list): Variable name list matching x_unscaled.
            use_opt_timesteps (bool): Passed through to ``_run_cycle``.

        Returns:
            dict: KPI dict.
        """
        if (self._cache_x is not None
                and self._cache_use_opt_timesteps == use_opt_timesteps
                and np.allclose(x_unscaled, self._cache_x, rtol=0, atol=1e-12)):
            return self._cache_kpi
        kpi = self._run_cycle(
            x_unscaled, var_names, use_opt_timesteps=use_opt_timesteps,
        )
        self._cache_x = x_unscaled.copy()
        self._cache_use_opt_timesteps = use_opt_timesteps
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
            max_tether_rori = (
                max(k.straight_tether_length for k in transition_rori.kinematics)
                if transition_rori is not None and transition_rori.kinematics
                else 0.0
            )
            altitude_ok = (
                self.minimum_height <= 0
                or min_altitude >= self.minimum_height - self._ALTITUDE_TOLERANCE
            )
            max_tether_ok = (
                self._max_tether_length is None
                or max_tether_rori <= self._max_tether_length + self._TETHER_LENGTH_TOLERANCE
            )

            apparent_wind_ok = not any(
                ss.apparent_wind_speed is not None and ss.apparent_wind_speed < 0
                for ss in traction.steady_states
            ) if traction.steady_states else True

            failure_reasons = []
            if error_in_phase is not None:
                failure_reasons.append(f'phase:{error_in_phase}')
            if not altitude_ok:
                failure_reasons.append('minimum_height')
            if not max_tether_ok:
                failure_reasons.append('max_tether_length_rori')
            if not apparent_wind_ok:
                failure_reasons.append('apparent_wind')

            cycle_ok = not failure_reasons

            traction_n = len(traction.steady_states)
            traction_regime2 = getattr(traction, 'regime2_count', 0)
            traction_regime3 = getattr(traction, 'regime3_count', 0)
            mechanical_energy, mechanical_power, electrical_energy, electrical_power = (
                build_cycle_energy_power_kpis(cycle, self.sys_props)
            )
            if not cycle_ok:
                mechanical_power['cycle'] = 0.0
                mechanical_energy['cycle'] = 0.0
                electrical_power['cycle'] = 0.0
                electrical_energy['cycle'] = 0.0

            return {
                'sim_successful': cycle_ok,
                'average_power': mechanical_power,
                'energy': mechanical_energy,
                'electrical_average_power': electrical_power,
                'electrical_energy': electrical_energy,
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
                'phase_sizes': {
                    'traction': len(traction.kinematics) if traction and traction.kinematics else 0,
                    'transition_rori': len(transition_rori.kinematics) if transition_rori and transition_rori.kinematics else 0,
                    'retraction': len(retraction.kinematics) if retraction and retraction.kinematics else 0,
                    'transition_riro': len(transition_riro.kinematics) if transition_riro and transition_riro.kinematics else 0,
                },
                'min_altitude_traction': min_altitude,
                'max_tether_length_rori': max_tether_rori,
                'traction_n_time_points': traction_n,
                'traction_regime2_count': traction_regime2,
                'traction_regime3_count': traction_regime3,
                'failure_reason': None if cycle_ok else ','.join(failure_reasons),
            }

        except Exception as exc:
            return {
                'sim_successful': False,
                'average_power': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0,
                                  'trans_rori': 0.0, 'out': 0.0},
                'energy': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0,
                           'trans_rori': 0.0, 'out': 0.0},
                'electrical_average_power': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0,
                                             'trans_rori': 0.0, 'out': 0.0},
                'electrical_energy': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0,
                                      'trans_rori': 0.0, 'out': 0.0},
                'duration': {'cycle': 0.0, 'in': 0.0, 'trans_riro': 0.0,
                             'trans_rori': 0.0, 'out': 0.0},
                'min_altitude_traction': 0.0,
                'max_tether_length_rori': 0.0,
                'traction_n_time_points': 0,
                'traction_regime2_count': 0,
                'traction_regime3_count': 0,
                'failure_reason': f'{type(exc).__name__}: {exc}',
            }
