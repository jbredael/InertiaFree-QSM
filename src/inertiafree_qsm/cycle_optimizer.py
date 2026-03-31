"""Cycle optimizer for AWE pumping cycle power optimization.

This module provides SLSQP-based optimization for pumping cycle parameters
to maximize power output while respecting operational constraints.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import optimize as op

from .qsm import Cycle
from .utils import flatten_dict


class OptimizerError(Exception):
    """Raised when the optimization encounters an invalid state."""

    pass


class CycleOptimizer:
    """Tether force controlled cycle optimizer using SLSQP.

    Optimizes pumping cycle parameters to maximize average power output while
    respecting physical constraints. Zero reeling speed is used as setpoint
    for the transition phase.

    The optimization variables are:
        - x[0]: Reel-out (traction) tether force [N]
        - x[1]: Reel-in (retraction) tether force [N]
        - x[2]: Elevation angle during traction [rad]
        - x[3]: Tether length at start of retraction [m]
        - x[4]: Tether length at end of retraction [m]

    Args:
        cycle_settings (dict): Cycle simulation settings.
        system_properties (SystemProperties): System properties object.
        environment_state: Environment / wind profile object.
        optimizer_config (dict): Dictionary with keys ``'x0'`` (np.ndarray),
            ``'scaling'`` (np.ndarray), ``'bounds'`` (5x2 array), ``'ftol'``,
            and ``'eps'``.
        reduce_x (np.ndarray, optional): Indices of optimization variables to
            include. Defaults to None (all variables).
        reduce_ineq_cons (int or np.ndarray, optional): Inequality constraint
            indices to enforce. Defaults to None (all four constraints).
    """

    # Optimization variable labels for plotting.
    OPT_VARIABLE_LABELS = [
        "Reel-out\nforce [N]",
        "Reel-in\nforce [N]",
        "Elevation\nangle [rad]",
        "End Reel-out tether\nlength [m]",
        "Start Reel-out tether\nlength [m]",
    ]

    def __init__(
        self,
        cycle_settings,
        system_properties,
        environment_state,
        optimizer_config,
        reduce_x=None,
        reduce_ineq_cons=None,
    ):
        """Initialize the cycle optimizer."""
        # System and environment references.
        self.system_properties = system_properties
        self.environment_state = environment_state

        # Build optimization vector configuration from optimizer_config.
        self.x0_real_scale = np.array(optimizer_config["x0"], dtype=float)
        self.scaling_x = np.array(optimizer_config["scaling"], dtype=float)
        self.bounds_real_scale = np.array(optimizer_config["bounds"], dtype=float)

        assert isinstance(self.x0_real_scale, np.ndarray)
        assert isinstance(self.scaling_x, np.ndarray)
        assert isinstance(self.bounds_real_scale, np.ndarray)

        # Variable and constraint reduction.
        if reduce_x is None:
            self.reduce_x = np.arange(len(self.x0_real_scale))
        else:
            assert isinstance(reduce_x, np.ndarray)
            self.reduce_x = reduce_x

        if reduce_ineq_cons is None:
            self.reduce_ineq_cons = np.arange(4)
        elif isinstance(reduce_ineq_cons, int):
            self.reduce_ineq_cons = np.arange(reduce_ineq_cons)
        else:
            self.reduce_ineq_cons = reduce_ineq_cons

        # Solver tolerances.
        self.ftol = optimizer_config["ftol"]
        self.eps = optimizer_config["eps"]

        # Scaled starting point (set during optimize()).
        self.x0 = None
        # Optimal solution in real scale.
        self.x_opt_real_scale = None

        # Operational attributes for caching objective/constraint evaluations.
        self.x_last = None
        self.obj = None
        self.ineq_cons = None
        self.x_progress = []

        # Optimization result dictionary.
        self.op_res = None

        # Store cycle settings, printing any keys that will be overruled.
        cycle_settings.setdefault("cycle", {})
        cycle_keys = list(flatten_dict(cycle_settings))
        overruled_keys = [
            k
            for k in [
                "cycle.elevation_angle_traction",
                "cycle.tether_length_start_retraction",
                "cycle.tether_length_end_retraction",
                "retraction.control",
                "transition.control",
                "traction.control",
            ]
            if k in cycle_keys
        ]
        if overruled_keys:
            print("Overruled cycle setting: " + ", ".join(overruled_keys) + ".")
        self.cycle_settings = cycle_settings

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def optimize(self, maxiter=30, iprint=-1):
        """Run the SLSQP optimization.

        Args:
            maxiter (int): Maximum number of optimizer iterations.
            iprint (int): SLSQP print level (-1 = silent, 0 = final, 1 = iter).

        Returns:
            np.ndarray: Optimal solution in real (unscaled) units.
        """
        self._clear_result_attributes()

        # Build scaled starting point and bounds.
        self.x0 = self.x0_real_scale * self.scaling_x
        bounds = self.bounds_real_scale.copy()
        bounds[:, 0] *= self.scaling_x
        bounds[:, 1] *= self.scaling_x

        starting_point = self.x0[self.reduce_x]
        bounds = bounds[self.reduce_x]

        # Build constraint list.
        cons = [
            {"type": "ineq", "fun": self._cons_fun, "args": (i,)}
            for i in self.reduce_ineq_cons
        ]

        options = {
            "disp": True,
            "maxiter": maxiter,
            "ftol": self.ftol,
            "eps": self.eps,
            "iprint": iprint,
        }

        self.op_res = dict(
            op.minimize(
                self._obj_fun,
                starting_point,
                bounds=bounds,
                method="SLSQP",
                options=options,
                callback=self._callback_fun,
                constraints=cons,
            )
        )

        # Reconstruct full solution vector and descale.
        res_x = self.x0.copy()
        res_x[self.reduce_x] = self.op_res["x"]
        self.x_opt_real_scale = res_x / self.scaling_x

        return self.x_opt_real_scale

    def eval_point(self, x_real_scale=None, plot_result=False, relax_errors=False):
        """Evaluate simulation at a given point.

        Uses either the provided vector, the optimal solution, or the starting
        point (in that order of precedence).

        Args:
            x_real_scale (np.ndarray, optional): Optimization vector.
            plot_result (bool): Whether to plot the cycle trajectory.
            relax_errors (bool): Whether to suppress steady-state errors.

        Returns:
            tuple: (constraint_values, performance_indicators).
        """
        if x_real_scale is None:
            x_real_scale = (
                self.x_opt_real_scale
                if self.x_opt_real_scale is not None
                else self.x0_real_scale
            )
        kpis = self.eval_performance_indicators(x_real_scale, plot_result, relax_errors)
        cons = self.eval_fun(x_real_scale, scale_x=False, relax_errors=relax_errors)[1]
        return cons, kpis

    def check_gradient(self, x_real_scale=None, step_size=1e-6):
        """Compute forward finite-difference gradient at a point.

        Args:
            x_real_scale (np.ndarray, optional): Point at which to evaluate.
            step_size (float): Finite difference step.

        Returns:
            tuple: (gradient, logarithmic_sensitivities).
        """
        self.x0 = self.x0_real_scale * self.scaling_x

        if x_real_scale is None:
            x_real_scale = (
                self.x_opt_real_scale
                if self.x_opt_real_scale is not None
                else self.x0_real_scale
            )
        x_ref = x_real_scale * self.scaling_x

        obj_ref = self._obj_fun(x_ref)
        gradient, log_sens = [], []
        for i, xi in enumerate(x_ref):
            x_pert = x_ref.copy()
            x_pert[i] += step_size
            grad = (self._obj_fun(x_pert) - obj_ref) / step_size
            gradient.append(grad)
            log_sens.append(xi / obj_ref * grad)

        return gradient, log_sens

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_opt_evolution(self, output_path=None, show_plot=False):
        """Plot the evolution of optimization variables.

        Each variable is shown in a single panel with dual y-axes (scaled and
        real values). The bottom row shows objective and constraint functions.

        Args:
            output_path (str or Path, optional): Save figure as PDF.
            show_plot (bool): Display figure interactively.
        """
        if not self.x_progress:
            return

        n_vars = len(self.x_progress[0])
        n_rows = n_vars + 1
        fig = plt.figure(figsize=(9, 3 * n_rows))
        gs = GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle(
            f"Optimization Evolution  –  v = {self.environment_state.wind_speed:.1f} m/s",
            fontweight="bold",
            fontsize=13,
        )

        scaling = self.scaling_x[self.reduce_x]
        iters = range(len(self.x_progress))

        for i, var_idx in enumerate(self.reduce_x):
            label = self.OPT_VARIABLE_LABELS[var_idx].replace("\n", " ")

            ax_l = fig.add_subplot(gs[i, :])
            ax_r = ax_l.twinx()

            scaled_vals = [x[i] for x in self.x_progress]
            real_factor = (1.0 / scaling[i]) * (
                180.0 / np.pi if var_idx == 2 else 1.0
            )
            label_real = label.replace("[rad]", "[deg]") if var_idx == 2 else label

            ax_l.plot(iters, scaled_vals, color="C0", linewidth=1.5)

            ax_l.set_ylabel("scaled [-]", fontsize=8, color="C0")
            ax_r.set_ylabel(label_real, fontsize=8, color="C1")
            ax_l.tick_params(axis="y", labelcolor="C0", labelsize=7)
            ax_r.tick_params(axis="y", labelcolor="C1", labelsize=7)
            ax_l.ticklabel_format(axis="y", useOffset=False, style="plain")
            ax_r.ticklabel_format(axis="y", useOffset=False, style="plain")

            lo, hi = ax_l.get_ylim()
            ax_r.set_ylim(lo * real_factor, hi * real_factor)

            ax_l.set_title(label, fontsize=8, loc="left")
            ax_l.set_xlabel("Iteration [-]", fontsize=8)
            ax_l.grid(True, alpha=0.4)

        # Objective panel.
        ax_obj = fig.add_subplot(gs[n_vars, 0])
        obj_res = [self._obj_fun(x) for x in self.x_progress]
        ax_obj.plot(iters, obj_res, linewidth=1.5)
        ax_obj.grid(True, alpha=0.4)
        ax_obj.set_ylabel("Objective [-]", fontsize=8)
        ax_obj.set_xlabel("Iteration [-]")

        # Constraints panel.
        ax_cons = fig.add_subplot(gs[n_vars, 1])
        cons_res = [self._cons_fun(x, -1)[self.reduce_ineq_cons] for x in self.x_progress]
        cons_lines = ax_cons.plot(cons_res)
        active_cons = [any(c < -1e-6 for c in res) for res in cons_res]
        ax_cons.fill_between(
            iters,
            0,
            1,
            where=active_cons,
            alpha=0.4,
            transform=ax_cons.get_xaxis_transform(),
        )
        ax_cons.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        ax_cons.legend(
            cons_lines,
            [f"constraint {j}" for j in range(len(cons_lines))],
            fontsize=7,
            loc="upper right",
        )
        ax_cons.grid(True, alpha=0.4)
        ax_cons.set_ylabel("Constraint [-]", fontsize=8)
        ax_cons.set_xlabel("Iteration [-]")

        plt.tight_layout()

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
            print(f"  Saved optimization evolution plot → {out}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    # def perform_local_sensitivity_analysis(self):
    #     """Sweep each variable individually and plot objective/constraints.

    #     Produces one subplot per optimization variable showing how the
    #     objective and constraint functions vary across the variable's bounds.
    #     """
    #     red_x = self.reduce_x
    #     n_plots = len(red_x)
    #     bounds = self.bounds_real_scale[red_x]

    #     x_ref_real_scale = (
    #         self.x_opt_real_scale
    #         if self.x_opt_real_scale is not None
    #         else self.x0_real_scale
    #     )
    #     f_ref, cons_ref = self.eval_fun(x_ref_real_scale, scale_x=False)

    #     fig, ax = plt.subplots(n_plots)
    #     if n_plots == 1:
    #         ax = [ax]
    #     fig.subplots_adjust(hspace=0.3)

    #     for i, b in enumerate(bounds):
    #         lb, ub = b
    #         xi_sweep = np.linspace(lb, ub, 50)
    #         f, g, active_g = [], [], []
    #         for xi in xi_sweep:
    #             x_full = list(x_ref_real_scale)
    #             x_full[red_x[i]] = xi
    #             try:
    #                 res_eval = self.eval_fun(x_full, scale_x=False)
    #                 f.append(res_eval[0])
    #                 cons = res_eval[1][self.reduce_ineq_cons]
    #                 g.append(res_eval[1])
    #                 active_g.append(any(c < -1e-6 for c in cons))
    #             except Exception:
    #                 f.append(None)
    #                 g.append(None)
    #                 active_g.append(False)

    #         ax[i].plot(xi_sweep, f, "--", label="objective")
    #         x_ref = x_ref_real_scale[red_x[i]]
    #         ax[i].plot(x_ref, f_ref, "x", label="x_ref", markersize=12)

    #         for i_cons in self.reduce_ineq_cons:
    #             cons_line = ax[i].plot(
    #                 xi_sweep,
    #                 [c[i_cons] if c is not None else None for c in g],
    #                 label=f"constraint {i_cons}",
    #             )
    #             clr = cons_line[0].get_color()
    #             ax[i].plot(x_ref, cons_ref[i_cons], "s", markerfacecolor="None", color=clr)

    #         ax[i].fill_between(
    #             xi_sweep, 0, 1, where=active_g, alpha=0.4, transform=ax[i].get_xaxis_transform()
    #         )
    #         ax[i].set_xlabel(self.OPT_VARIABLE_LABELS[red_x[i]])
    #         ax[i].set_ylabel("Response [-]")
    #         ax[i].grid()

    #     ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    #     ax[0].set_title(f"v={self.environment_state.wind_speed:.1f}m/s")
    #     plt.subplots_adjust(right=0.7)
    #     plt.show()

    # -------------------------------------------------------------------------
    # Objective and constraint evaluation
    # -------------------------------------------------------------------------

    def eval_fun(self, x, scale_x=True, **kwargs):
        """Compute objective and constraint functions.

        Args:
            x (np.ndarray): Optimization vector (scaled or real).
            scale_x (bool): Whether x is in scaled units.
            **kwargs: Passed to eval_performance_indicators.

        Returns:
            tuple: (objective, constraint_array).
        """
        x_real = x / self.scaling_x if scale_x else np.asarray(x)
        res = self.eval_performance_indicators(x_real, **kwargs)

        # Reference wind power at 100 m altitude.
        env = self.environment_state
        env.calculate(100.0)
        power_wind_100m = 0.5 * env.air_density * env.wind_speed**3

        # Objective: negative normalized power (minimization).
        obj = (
            -res["average_power"]["cycle"]
            / power_wind_100m
            / self.system_properties.kite_projected_area
        )

        # Constraint 0: reel-out force setpoint tracking.
        min_force_out = res["min_tether_force"]["out"]
        if min_force_out == np.inf:
            min_force_out = 0.0
        c0 = (min_force_out - x_real[0]) * 1e-2 + 1e-6

        # Constraint 1: reel-in force setpoint tracking.
        c1 = (res["max_tether_force"]["in"] - x_real[1]) * 1e-2 + 1e-6

        # Constraint 2: peak tether force within limit.
        force_max = self.system_properties.tether_force_max_limit
        c2 = -(res["max_tether_force"]["out"] - force_max) / force_max + 1e-6

        # Constraint 3: minimum crosswind patterns.
        min_cw = (
            self.cycle_settings.get("optimization", {})
            .get("constraints", {})
            .get("min_crosswind_patterns")
        )
        n_cw = res["n_crosswind_patterns"]
        c3 = (n_cw - min_cw) if n_cw is not None else 0.0

        ineq_cons = np.array([c0, c1, c2, c3])
        return obj, ineq_cons

    def eval_performance_indicators(self, x_real_scale, plot_result=False, relax_errors=False):
        """Run the cycle simulation and extract KPIs.

        Args:
            x_real_scale (np.ndarray): Optimization vector in real units.
            plot_result (bool): Whether to plot the cycle trajectory.
            relax_errors (bool): Whether to suppress steady-state errors.

        Returns:
            dict: Performance indicators including power, forces, durations.
        """
        (
            tether_force_traction,
            tether_force_retraction,
            elevation_angle,
            tether_length_start,
            tether_length_end,
        ) = x_real_scale

        # Update cycle settings with current optimization variables.
        self.cycle_settings["cycle"]["elevation_angle_traction"] = elevation_angle
        self.cycle_settings["cycle"]["tether_length_start_retraction"] = tether_length_start
        self.cycle_settings["cycle"]["tether_length_end_retraction"] = tether_length_end

        self.cycle_settings["retraction"]["control"] = (
            "tether_force_ground",
            tether_force_retraction,
        )
        self.cycle_settings["transition"]["control"] = ("reeling_speed", 0.0)
        self.cycle_settings["traction"]["control"] = (
            "tether_force_ground",
            tether_force_traction,
        )

        # Run cycle simulation.
        cycle = Cycle(self.cycle_settings)
        ss_config = self.cycle_settings.get("steady_state", {})
        iterative_config = {
            "enable_steady_state_errors": not relax_errors,
            "max_iterations": ss_config.get("max_iterations"),
            "convergence_tolerance": ss_config.get("convergence_tolerance"),
        }
        cycle.run_simulation(
            self.system_properties,
            self.environment_state,
            iterative_config,
            not relax_errors,
        )

        if plot_result:
            cycle.trajectory_plot(steady_state_markers=True)
            phase_switch = [
                cycle.transition_phase.time[0],
                cycle.traction_phase.time[0],
            ]
            cycle.time_plot(
                [
                    "straight_tether_length",
                    "reeling_speed",
                    "tether_force_ground",
                    "power_ground",
                ],
                plot_markers=phase_switch,
            )
            plt.tight_layout()
            plt.show()

        return {
            "average_power": {
                "cycle": cycle.average_power,
                "in": cycle.retraction_phase.average_power,
                "trans": cycle.transition_phase.average_power,
                "out": cycle.traction_phase.average_power,
            },
            "min_tether_force": {
                "in": cycle.retraction_phase.min_tether_force,
                "trans": cycle.transition_phase.min_tether_force,
                "out": cycle.traction_phase.min_tether_force,
            },
            "max_tether_force": {
                "in": cycle.retraction_phase.max_tether_force,
                "trans": cycle.transition_phase.max_tether_force,
                "out": cycle.traction_phase.max_tether_force,
            },
            "min_reeling_speed": {
                "in": cycle.retraction_phase.min_reeling_speed,
                "out": cycle.traction_phase.min_reeling_speed,
            },
            "max_reeling_speed": {
                "in": cycle.retraction_phase.max_reeling_speed,
                "out": cycle.traction_phase.max_reeling_speed,
            },
            "n_crosswind_patterns": getattr(
                cycle.traction_phase, "n_crosswind_patterns", None
            ),
            "min_height": min(
                cycle.traction_phase.kinematics[0].z,
                cycle.traction_phase.kinematics[-1].z,
            ),
            "max_elevation_angle": cycle.transition_phase.kinematics[0].elevation_angle,
            "duration": {
                "cycle": cycle.duration,
                "in": cycle.retraction_phase.duration,
                "trans": cycle.transition_phase.duration,
                "out": cycle.traction_phase.duration,
            },
            "duty_cycle": cycle.duty_cycle,
            "pumping_efficiency": cycle.pumping_efficiency,
            "kinematics": cycle.kinematics,
            "steady_states": cycle.steady_states,
            "time": cycle.time,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _clear_result_attributes(self):
        """Reset state before a new optimization run."""
        self.x0 = None
        self.x_last = None
        self.obj = None
        self.ineq_cons = None
        self.x_progress = []
        self.x_opt_real_scale = None
        self.op_res = None

    def _obj_fun(self, x):
        """Objective function wrapper for scipy.optimize.

        Caches simulation results to avoid redundant evaluations when scipy
        calls objective and constraints with the same vector.
        """
        x_full = self.x0.copy()
        x_full[self.reduce_x] = x
        if not np.array_equal(x_full, self.x_last):
            self.obj, self.ineq_cons = self.eval_fun(x_full)
            self.x_last = x_full.copy()
        return self.obj

    def _cons_fun(self, x, return_i=-1):
        """Constraint function wrapper for scipy.optimize.

        Args:
            x (np.ndarray): Optimization vector (reduced, scaled).
            return_i (int): Index of single constraint to return, or -1 for all.

        Returns:
            float or np.ndarray: Constraint value(s).
        """
        x_full = self.x0.copy()
        x_full[self.reduce_x] = x
        if not np.array_equal(x_full, self.x_last):
            self.obj, self.ineq_cons = self.eval_fun(x_full)
            self.x_last = x_full.copy()
        return self.ineq_cons[return_i] if return_i > -1 else self.ineq_cons

    def _callback_fun(self, x):
        """Callback invoked by scipy at each iteration."""
        if np.isnan(x).any():
            raise OptimizerError("Optimization vector contains NaN's.")
        self.x_progress.append(x.copy())

        # Print iteration summary.
        iter_num = len(self.x_progress)
        x_full_scaled = self.x0.copy()
        x_full_scaled[self.reduce_x] = x
        x_full_real = x_full_scaled / self.scaling_x
        parts = []
        for idx, val in enumerate(x_full_real):
            label = self.OPT_VARIABLE_LABELS[idx].replace("\n", " ")
            fixed = idx not in self.reduce_x
            suffix = " (fixed)" if fixed else ""
            if idx == 2:
                parts.append(f"{label}: {np.degrees(val):.2f} deg{suffix}")
            else:
                parts.append(f"{label}: {val:.4g}{suffix}")
        print(f"  iter {iter_num:3d} | " + " | ".join(parts))
