import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle

from qsm import SteadyStateError, OperationalLimitViolation, PhaseError
from utils import flatten_dict


class PowerCurveConstructor:
    def __init__(self, wind_speeds):
        self.wind_speeds = wind_speeds

        self.x_opts = []
        self.x0 = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

    def run_optimization(self, wind_speed, power_optimizer, x0):
        #TODO: evaluate if robustness can be improved by running multiple optimizations using different starting points
        power_optimizer.environment_state.set_reference_wind_speed(wind_speed)

        print("x0:", x0)
        power_optimizer.x0_real_scale = x0
        x_opt = power_optimizer.optimize()
        self.x0.append(x0)
        self.x_opts.append(x_opt)
        self.optimization_details.append(power_optimizer.op_res)

        try:
            cons, kpis = power_optimizer.eval_point()
            sim_successful = True
        except (SteadyStateError, OperationalLimitViolation, PhaseError) as e:
            print("Error occurred while evaluating the resulting optimal point: {}".format(e))
            cons, kpis = power_optimizer.eval_point(relax_errors=True)
            sim_successful = False

        print("cons:", cons)
        self.constraints.append(cons)
        kpis['sim_successful'] = sim_successful
        self.performance_indicators.append(kpis)
        return x_opt, sim_successful

    def run_predefined_sequence(self, seq, x0_start):
        wind_speed_tresholds = iter(sorted(list(seq)))
        vw_switch = next(wind_speed_tresholds)

        x_opt_last, vw_last = None, None
        for i, vw in enumerate(self.wind_speeds):
            if vw > vw_switch:
                vw_switch = next(wind_speed_tresholds)

            power_optimizer = seq[vw_switch]['power_optimizer']
            dx0 = seq[vw_switch].get('dx0', None)

            if x_opt_last is None:
                x0_next = x0_start
            else:
                x0_next = x_opt_last + dx0*(vw - vw_last)

            print("[{}] Processing v={:.2f}m/s".format(i, vw))
            try:
                x_opt, sim_successful = self.run_optimization(vw, power_optimizer, x0_next)
            except (OperationalLimitViolation, SteadyStateError, PhaseError):
                try:  # Retry for a slightly different wind speed.
                    x_opt, sim_successful = self.run_optimization(vw+1e-2, power_optimizer, x0_next)
                    self.wind_speeds[i] = vw+1e-2
                except (OperationalLimitViolation, SteadyStateError, PhaseError):
                    self.wind_speeds = self.wind_speeds[:i]
                    print("Optimization sequence stopped prematurely due to failed optimization. {:.1f} m/s is the "
                          "highest wind speed for which the optimization was successful.".format(self.wind_speeds[-1]))
                    break

            if sim_successful:
                x_opt_last = x_opt
                vw_last = vw

    def plot_optimal_trajectories(self, wind_speed_ids=None, ax=None, circle_radius=200, elevation_line=25*np.pi/180):
        if ax is None:
            plt.figure(figsize=(6, 3.5))
            plt.subplots_adjust(right=0.65)
            ax = plt.gca()

        if wind_speed_ids is None:
            if len(self.wind_speeds) > 8:
                wind_speed_ids = [int(a) for a in np.linspace(0, len(self.wind_speeds)-1, 6)]
            else:
                wind_speed_ids = range(len(self.wind_speeds))

        for i in wind_speed_ids:
            v = self.wind_speeds[i]
            kpis = self.performance_indicators[i]
            if kpis is None:
                print("No trajectory available for {} m/s wind speed.".format(v))
                continue

            x_kite, z_kite = zip(*[(kp.x, kp.z) for kp in kpis['kinematics']])
            # try:
            #     z_traj = [kp.z for kp in kite_positions['trajectory']]
            # except AttributeError:
            #     z_traj = [np.sin(kp.elevation_angle)*kp.straight_tether_length for kp in kite_positions['trajectory']]
            ax.plot(x_kite, z_kite, label="$v_{100m}$="+"{:.1f} ".format(v) + "m s$^{-1}$")

        # Plot semi-circle at constant tether length bound.
        phi = np.linspace(0, 2*np.pi/3, 40)
        x_circle = np.cos(phi) * circle_radius
        z_circle = np.sin(phi) * circle_radius
        ax.plot(x_circle, z_circle, 'k--', linewidth=1)

        # Plot elevation line.
        x_elev = np.linspace(0, 400, 40)
        z_elev = np.tan(elevation_line)*x_elev
        ax.plot(x_elev, z_elev, 'k--', linewidth=1)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.grid()
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    def plot_optimization_results(self, opt_variable_labels=None, opt_variable_bounds=None, tether_force_limits=None,
                                  reeling_speed_limits=None):
        assert self.x_opts, "No optimization results available for plotting."
        xf, x0 = self.x_opts, self.x0
        cons = self.constraints
        kpis, opt_details = self.performance_indicators, self.optimization_details
        try:
            performance_indicators = next(list(flatten_dict(kpi)) for kpi in kpis if kpi is not None)
        except StopIteration:
            performance_indicators = []

        n_opt_vars = len(xf[0])
        fig, ax = plt.subplots(max([n_opt_vars, 6]), 2, sharex=True)

        # In the left column plot each optimization variable against the wind speed.
        for i in range(n_opt_vars):
            # Plot optimal and starting points.
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in xf], label='x_opt')
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in x0], 'o', markerfacecolor='None', label='x0')

            if opt_variable_labels:
                label = opt_variable_labels[i]
                ax[i, 0].set_ylabel(label)
            else:
                ax[i, 0].set_ylabel("x[{}]".format(i))

            if opt_variable_bounds is not None:
                ax[i, 0].axhline(opt_variable_bounds[i, 0], linestyle='--', color='k')
                ax[i, 0].axhline(opt_variable_bounds[i, 1], linestyle='--', color='k')

            ax[i, 0].grid()
        ax[0, 0].legend()

        # In the right column plot the number of iterations in the upper panel.
        nits = np.array([od['nit'] for od in opt_details])
        ax[0, 1].plot(self.wind_speeds, nits)
        mask_opt_failed = np.array([~od['success'] for od in opt_details])
        ax[0, 1].plot(self.wind_speeds[mask_opt_failed], nits[mask_opt_failed], 'x', label='opt failed')
        mask_sim_failed = np.array([~kpi['sim_successful'] for kpi in kpis])
        ax[0, 1].plot(self.wind_speeds[mask_sim_failed], nits[mask_sim_failed], 'x', label='sim failed')
        ax[0, 1].grid()
        ax[0, 1].legend()
        ax[0, 1].set_ylabel('Optimization iterations [-]')

        # In the second panel, plot the optimal power.
        cons_treshold = -.1
        mask_cons_adhered = np.array([all([c >= cons_treshold for c in con]) for con in cons])
        mask_plot_power = ~mask_sim_failed & mask_cons_adhered
        power = np.array([kpi['average_power']['cycle'] for kpi in kpis])
        power[~mask_plot_power] = np.nan
        ax[1, 1].plot(self.wind_speeds, power)
        ax[1, 1].grid()
        ax[1, 1].set_ylabel('Mean power [W]')

        # In the third panel, plot the tether force related performance indicators.
        max_force_in = np.array([kpi['max_tether_force']['in'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_in, label='max_tether_force.in')
        max_force_out = np.array([kpi['max_tether_force']['out'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_out, label='max_tether_force.out')
        max_force_trans = np.array([kpi['max_tether_force']['trans'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_trans, label='max_tether_force.trans')
        if tether_force_limits:
            ax[3, 1].axhline(tether_force_limits[0], linestyle='--', color='k')
            ax[3, 1].axhline(tether_force_limits[1], linestyle='--', color='k')
        ax[2, 1].grid()
        ax[2, 1].set_ylabel('Tether force [N]')
        ax[2, 1].legend(loc=2)
        ax[2, 1].annotate('Violation occurring before\nswitch to force controlled',
                          xy=(0.05, 0.10), xycoords='axes fraction')

        # Plot reeling speed related performance indicators.
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

        # Plot constraint matrix.
        cons_matrix = np.array(cons).transpose()
        n_cons = cons_matrix.shape[0]

        cons_treshold_magenta = -.1
        cons_treshold_red = -1e-6

        # Assign color codes based on the constraint values.
        color_code_matrix = np.where(cons_matrix < cons_treshold_magenta, -2, 0)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_magenta) & (cons_matrix < cons_treshold_red), -1,
                                     color_code_matrix)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_red) & (cons_matrix < 1e-3), 1, color_code_matrix)
        color_code_matrix = np.where(cons_matrix == 0., 0, color_code_matrix)
        color_code_matrix = np.where(cons_matrix >= 1e-3, 2, color_code_matrix)

        # Plot color code matrix.
        cmap = mpl.colors.ListedColormap(['r', 'm', 'y', 'g', 'b'])
        bounds = [-2, -1, 0, 1, 2]
        mpl.colors.BoundaryNorm(bounds, cmap.N)
        im1 = ax[4, 1].matshow(color_code_matrix, cmap=cmap, vmin=bounds[0], vmax=bounds[-1],
                                    extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
        ax[4, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[4, 1].set_yticklabels(range(n_cons))
        ax[4, 1].set_ylabel('Constraint id\'s')

        # Add colorbar.
        ax_pos = ax[4, 1].get_position()
        h_cbar = ax_pos.y1 - ax_pos.y0
        w_cbar = .012
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = np.arange(-2+4/10., 2., 4/5.)
        cbar_ticks_labels = ['<-.1', '<0', '0', '~0', '>0']
        cbar = fig.colorbar(im1, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        # Plot constraint matrix with linear mapping the colors from data values between plot_cons_range.
        plot_cons_range = [-.1, .1]
        im2 = ax[5, 1].matshow(cons_matrix, vmin=plot_cons_range[0], vmax=plot_cons_range[1], cmap=mpl.cm.YlGnBu_r,
                               extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
        ax[5, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[5, 1].set_yticklabels(range(n_cons))
        ax[5, 1].set_ylabel('Constraint id\'s')

        # Add colorbar.
        ax_pos = ax[5, 1].get_position()
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = plot_cons_range[:]
        cbar_ticks_labels = [str(v) for v in cbar_ticks]
        if plot_cons_range[0] < np.min(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(1, np.min(cons_matrix))
            cbar_ticks_labels.insert(1, 'min: {:.2E}'.format(np.min(cons_matrix)))
        if plot_cons_range[0] < np.max(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(-1, np.max(cons_matrix))
            cbar_ticks_labels.insert(-1, 'max: {:.2E}'.format(np.max(cons_matrix)))
        cbar = fig.colorbar(im2, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        ax[-1, 0].set_xlabel('Wind speeds [m/s]')
        ax[-1, 1].set_xlabel('Wind speeds [m/s]')
        ax[0, 0].set_xlim([self.wind_speeds[0], self.wind_speeds[-1]])

    def export_results(self, file_name):
        export_dict = self.__dict__
        # for k, v in export_dict.items():
        #     if isinstance(v, np.ndarray):
        #         export_dict[k] = v.copy().tolist()
        with open(file_name, 'wb') as f:
            pickle.dump(export_dict, f)

    def import_results(self, file_name):
        with open(file_name, 'rb') as f:
            import_dict = pickle.load(f)
        for k, v in import_dict.items():
            setattr(self, k, v)