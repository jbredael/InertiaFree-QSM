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

    Optimizes reeling speeds and tether lengths to maximize average cycle
    power while respecting operational bounds.

    The decision variable vector is:
        x = [reeling_speed_out, reeling_speed_in,
             fraction_tether_length_retraction_end,
             fraction_tether_length_retraction_start]

    Args:
        simulation_settings (dict): Full simulation settings dict (as returned
            by ``load_system_and_simulation_settings``).
        sys_props (SystemProperties): System properties object.
        env_state (NormalisedWindTable1D): Environment state with wind profile.

    Attributes:
        history (list): List of dicts recording every objective evaluation,
            each with keys ``x`` (variable vector) and ``power`` (cycle power [W]).
    """
    # Large penalty for failed simulations to steer the optimizer away from
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
        
        # Convert tether length bounds from absolute to fractional values.
        maxTetherLength = sys_props.max_tether_length
        fracStartMin = self.boundsDict['tether_length_start'][0] / maxTetherLength
        fracStartMax = self.boundsDict['tether_length_start'][1] / maxTetherLength
        fracEndMin = self.boundsDict['tether_length_end'][0] / maxTetherLength
        fracEndMax = self.boundsDict['tether_length_end'][1] / maxTetherLength

        # Bounds: [rs_out, rs_in, frac_end, frac_start]
        self.bounds = [
            self.boundsDict['reeling_speed_out'],
            self.boundsDict['reeling_speed_in'],
            (fracEndMin, fracEndMax),
            (fracStartMin, fracStartMax),
        ]

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

        x0 = np.array(self.optimizer_config['x0'], dtype=float)
        scaling = np.array(self.optimizer_config['scaling'], dtype=float)
        x0Scaled = x0 * scaling

        scaledBounds = [
            (lo * s, hi * s)
            for (lo, hi), s in zip(self.bounds, scaling)
        ]

        min_frac_diff = self.constraintsDict['min_tether_length_fraction_difference']
        constraints = [{
            'type': 'ineq',
            'fun': lambda x: x[3] / scaling[3] - x[2] / scaling[2] - min_frac_diff,
        }]

        def _callback(xk):
            self._iter_count += 1
            xUnscaled = xk / scaling
            # Best power seen so far (most recent history entry is the current point)
            currentPower = self.history[-1]['power'] if self.history else float('nan')
            if verbose:
                print(
                    f"      iter {self._iter_count:3d} | "
                    f"rf_out={xUnscaled[0]:+.4f}  rf_in={xUnscaled[1]:+.4f}  "
                    f"frac_end={xUnscaled[2]:.4f}  frac_start={xUnscaled[3]:.4f} | "
                    f"power={currentPower/1000:+.3f} kW"
                )

        result = op.minimize(
            lambda x: self._objective(x / scaling),
            x0Scaled,
            method='SLSQP',
            bounds=scaledBounds,
            constraints=constraints,
            callback=_callback,
            options={
                'maxiter': self.optimizer_config['max_iterations'],
                'ftol': self.optimizer_config['ftol'],
                'eps': self.optimizer_config['eps'],
                'disp': False,
            },
        )

        xOpt = result.x / scaling
        kpi = self._evaluate(xOpt)
        kpi['optimization_result'] = result
        return kpi

    def _objective(self, x):
        """Objective function: negative average cycle power.

        Args:
            x (np.ndarray): Decision variables (unscaled).

        Returns:
            float: Negative average cycle power, or a large penalty on failure.
        """
        kpi = self._evaluate(x)
        power = kpi['average_power']['cycle']

        if not kpi['sim_successful']:
            power = -self.PENALTY

        self.history.append({'x': x.copy(), 'power': power})
        return -power

    def _evaluate(self, x):
        """Run a cycle simulation with the given decision variables.

        Args:
            x (np.ndarray): Decision variables
                [rs_out, rs_in, frac_end, frac_start].

        Returns:
            dict: KPI dict compatible with ``_build_wind_speed_entry``.
        """
        rsOut, rsIn, fracEnd, fracStart = x
        maxTetherLength = self.sys_props.max_tether_length

        settings = deepcopy(self.simulation_settings)
        settings['cycle']['tether_length_start_retraction'] = fracStart * maxTetherLength
        settings['cycle']['tether_length_end_retraction'] = fracEnd * maxTetherLength
        settings['traction']['control'] = ('reeling_speed', float(rsOut))
        settings['retraction']['control'] = ('reeling_speed', float(rsIn))
        settings['cycle']['traction_phase'] = TractionPhase

        cycle = Cycle(settings, impose_operational_limits=True)
        steadyStateConfig = self.simulation_settings.get('steady_state')

        try:
            errorInPhase, _ = cycle.run_simulation(
                self.sys_props, self.env_state, steadyStateConfig,
                print_summary=False,
            )

            traction = getattr(cycle, 'traction_phase')
            retraction = getattr(cycle, 'retraction_phase')
            transition = getattr(cycle, 'transition_phase')

            return {
                'sim_successful': errorInPhase is None,
                'average_power': {
                    'cycle': cycle.average_power if errorInPhase is None else 0.0,
                    'in': retraction.average_power if retraction else 0.0,
                    'trans': transition.average_power if transition else 0.0,
                    'out': traction.average_power if traction else 0.0,
                },
                'duration': {
                    'cycle': cycle.duration if errorInPhase is None else 0.0,
                    'in': retraction.duration if retraction else 0.0,
                    'trans': transition.duration if transition else 0.0,
                    'out': traction.duration if traction else 0.0,
                },
                'time': cycle.time,
                'kinematics': cycle.kinematics,
                'steady_states': cycle.steady_states,
            }

        except Exception:
            return {
                'sim_successful': False,
                'average_power': {
                    'cycle': 0.0, 'in': 0.0, 'trans': 0.0, 'out': 0.0,
                },
                'duration': {
                    'cycle': 0.0, 'in': 0.0, 'trans': 0.0, 'out': 0.0,
                },
            }
