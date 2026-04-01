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

