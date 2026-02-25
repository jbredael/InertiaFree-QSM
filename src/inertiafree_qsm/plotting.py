"""Plotting functions for AWE power curve analysis and cycle diagnostics.

This module consolidates all plotting functionality used by the power curve
constructor and optimizer. Functions accept data dicts (as produced by the
YAML output) so that plotting is decoupled from the simulation objects.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helper: load YAML power curve data
# ---------------------------------------------------------------------------

def load_power_curve_data(file_path):
    """Load power curve data from a YAML file.

    Args:
        file_path (str or Path): Path to the YAML file.

    Returns:
        dict: Power curve data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def find_wind_speed_data(data, target_wind_speed, profile_id=None):
    """Find the data entry for a specific wind speed and profile.

    Args:
        data (dict): Power curve data loaded from YAML.
        target_wind_speed (float): Target wind speed [m/s].
        profile_id (int, optional): Profile ID (1-based). If None, the first
            profile containing the wind speed is used.

    Returns:
        dict or None: Wind speed data entry, or None if not found.
    """
    powerCurves = data.get('power_curves', [])
    if not powerCurves:
        return None

    for profile in powerCurves:
        if profile_id is not None and profile.get('profile_id') != profile_id:
            continue
        for entry in profile.get('wind_speed_data', []):
            if abs(entry['wind_speed_m_s'] - target_wind_speed) < 0.01:
                return entry

    if profile_id is not None:
        availableIds = [p.get('profile_id') for p in powerCurves]
        print(
            f"Profile ID {profile_id} not found or does not contain "
            f"wind speed {target_wind_speed} m/s. Available: {availableIds}"
        )
    return None


# ---------------------------------------------------------------------------
# Power-curve plot
# ---------------------------------------------------------------------------

def plot_power_curve(file_path, output_path=None, show_plot=True):
    """Plot power curve from a YAML data file.

    Generates a 2x2 figure with cycle power, power components, cycle time,
    and power coefficient.

    Args:
        file_path (str or Path): Path to the YAML file.
        output_path (str or Path, optional): Path to save the figure.
        show_plot (bool): Whether to display the plot. Defaults to True.

    Returns:
        tuple: (fig, axes) matplotlib figure and axes array.
    """
    data = load_power_curve_data(file_path)
    metadata = data.get('metadata', {})
    name = metadata.get('name', 'Power Curve')
    referenceHeight = metadata.get('wind_resource', {}).get('reference_height_m', 100)
    powerCurves = data.get('power_curves', [])

    if not powerCurves:
        raise ValueError("No power curve data found in file")

    nProfiles = len(powerCurves)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(name, fontsize=16, fontweight='bold')

    useMarkers = nProfiles <= 5
    markerSize = 4 if nProfiles <= 10 else 0
    lineWidth = 1.5 if nProfiles <= 10 else 1.0
    showComponentLegend = True

    for idx, profile in enumerate(powerCurves):
        profileId = profile.get('profile_id', 'Unknown')
        windSpeedData = profile.get('wind_speed_data', [])
        if not windSpeedData:
            continue

        windSpeeds = [e['wind_speed_m_s'] for e in windSpeedData]
        cyclePowerKw = [e['performance']['power']['average_cycle_power_w'] / 1000
                        for e in windSpeedData]
        reelOutPowerKw = [e['performance']['power']['average_reel_out_power_w'] / 1000
                          for e in windSpeedData]
        reelInPowerKw = [e['performance']['power']['average_reel_in_power_w'] / 1000
                         for e in windSpeedData]
        cycleTime = [e['performance']['timing']['cycle_time_s'] for e in windSpeedData]

        color = plt.cm.tab20(idx % 20) if nProfiles > 10 else None
        marker = 'o' if useMarkers else ''

        axes[0, 0].plot(windSpeeds, cyclePowerKw, marker=marker, linewidth=lineWidth,
                        markersize=markerSize, label=f'Profile {profileId}', color=color)

        if showComponentLegend:
            axes[0, 1].plot(windSpeeds, reelOutPowerKw, 's-', linewidth=1.5,
                            markersize=4, label='Reel-out', alpha=0.7)
            axes[0, 1].plot(windSpeeds, reelInPowerKw, '^-', linewidth=1.5,
                            markersize=4, label='Reel-in', alpha=0.7)
            axes[0, 1].plot(windSpeeds, cyclePowerKw, 'o-', linewidth=1.5,
                            markersize=4, label='Cycle', alpha=0.7)
            showComponentLegend = False
        else:
            axes[0, 1].plot(windSpeeds, reelOutPowerKw, 's-', linewidth=1.0,
                            markersize=2, alpha=0.4, color='C0')
            axes[0, 1].plot(windSpeeds, reelInPowerKw, '^-', linewidth=1.0,
                            markersize=2, alpha=0.4, color='C1')
            axes[0, 1].plot(windSpeeds, cyclePowerKw, 'o-', linewidth=1.0,
                            markersize=2, alpha=0.4, color='C2')

        axes[1, 0].plot(windSpeeds, cycleTime, marker=marker, linewidth=lineWidth,
                        markersize=markerSize, label=f'Profile {profileId}', color=color)

        if 'model_config' in metadata:
            wingArea = metadata['model_config'].get('wing_area_m2')
            if wingArea:
                AIR_DENSITY = 1.225
                powerCoeff = []
                for e in windSpeedData:
                    ws = e['wind_speed_m_s']
                    cpVal = e['performance']['power']['average_cycle_power_w']
                    if ws > 0 and cpVal >= 0:
                        availPower = 0.5 * AIR_DENSITY * wingArea * ws ** 3
                        powerCoeff.append(cpVal / availPower if availPower > 0 else 0)
                    else:
                        powerCoeff.append(0)
                axes[1, 1].plot(windSpeeds, powerCoeff, marker=marker,
                                linewidth=lineWidth, markersize=markerSize,
                                label=f'Profile {profileId}', color=color)

    # Axis labels and formatting
    for axObj, ylabel, title in [
        (axes[0, 0], 'Cycle Power (kW)', 'Power Curve'),
        (axes[1, 0], 'Cycle Time (s)', 'Cycle Duration'),
    ]:
        axObj.set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
        axObj.set_ylabel(ylabel, fontsize=11)
        axObj.set_title(title, fontsize=12, fontweight='bold')
        axObj.grid(True, alpha=0.3)
        if nProfiles <= 15:
            axObj.legend(fontsize=8, ncol=1 if nProfiles <= 8 else 2)

    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    axes[0, 1].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
    axes[0, 1].set_ylabel('Power (kW)', fontsize=11)
    axes[0, 1].set_title('Power Components (Profile 1)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    if 'model_config' in metadata and metadata['model_config'].get('wing_area_m2'):
        axes[1, 1].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
        axes[1, 1].set_ylabel('Power Coefficient', fontsize=11)
        axes[1, 1].set_title('Power Coefficient', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if nProfiles <= 15:
            axes[1, 1].legend(fontsize=8, ncol=1 if nProfiles <= 8 else 2)
    else:
        axes[1, 1].text(0.5, 0.5, 'Wing area not available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Power Coefficient', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# Single-wind-speed cycle detail plot
# ---------------------------------------------------------------------------

def plot_cycle_detail(file_path, wind_speed, profile_id=None,
                      output_path=None, show_plot=True):
    """Plot detailed cycle time histories for one wind speed.

    Generates a 3x2 figure with altitude, tether force, power, reel speed,
    tether length, and elevation angle.

    Args:
        file_path (str or Path): Path to the YAML file.
        wind_speed (float): Wind speed to plot [m/s].
        profile_id (int, optional): Profile ID (1-based).
        output_path (str or Path, optional): Path to save the figure.
        show_plot (bool): Whether to display the plot. Defaults to True.

    Returns:
        tuple: (fig, axes) or None if data is missing.
    """
    data = load_power_curve_data(file_path)
    metadata = data.get('metadata', {})
    referenceHeight = metadata.get('wind_resource', {}).get('reference_height_m', 100)
    profileLabel = f" (Profile {profile_id})" if profile_id is not None else ""

    wsData = find_wind_speed_data(data, wind_speed, profile_id)
    if wsData is None:
        print(f"Wind speed {wind_speed} m/s not found in data")
        return None

    if not wsData['success']:
        print(f"Simulation failed for wind speed {wind_speed} m/s")
        return None

    performance = wsData['performance']
    power = performance['power']
    timing = performance['timing']
    timeHistory = wsData.get('time_history', {})

    if not timeHistory:
        print(f"No time history data available for wind speed {wind_speed} m/s")
        return None

    time = np.array(timeHistory.get('time_s', []))
    altitude = np.array(timeHistory.get('altitude_m', []))
    tetherForce = np.array(timeHistory.get('tether_force_n', []))
    powerInst = np.array(timeHistory.get('power_w', []))
    reelSpeed = np.array(timeHistory.get('reel_speed_m_s', []))
    tetherLength = np.array(timeHistory.get('tether_length_m', []))
    elevationAngleDeg = np.array(timeHistory.get('elevation_angle_deg', []))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f'Detailed Cycle Analysis - Wind Speed: {wind_speed} m/s '
        f'at {referenceHeight}m{profileLabel}',
        fontsize=16, fontweight='bold',
    )

    reelOutTime = timing['reel_out_time_s']

    # Altitude
    ax = axes[0, 0]
    ax.plot(time, altitude, linewidth=2, color='steelblue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Kite Altitude', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=altitude.mean(), color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {altitude.mean():.1f} m')
    ax.legend(fontsize=9)

    # Tether force
    ax = axes[0, 1]
    ax.plot(time, tetherForce / 1000, linewidth=2, color='orangered')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tether Force (kN)')
    ax.set_title('Tether Force', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=tetherForce.mean() / 1000, color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {tetherForce.mean()/1000:.1f} kN')
    ax.legend(fontsize=9)

    # Instantaneous power
    ax = axes[1, 0]
    ax.plot(time, powerInst / 1000, linewidth=2, color='darkgreen')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Instantaneous Power', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axhline(y=power['average_cycle_power_w'] / 1000, color='r', linestyle='--',
               alpha=0.5, label=f"Avg Cycle: {power['average_cycle_power_w']/1000:.1f} kW")
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5,
               label='Phase transition')
    ax.legend(fontsize=9)

    # Reel speed
    ax = axes[1, 1]
    ax.plot(time, reelSpeed, linewidth=2, color='purple')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reel Speed (m/s)')
    ax.set_title('Tether Reel Speed', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5,
               label='Phase transition')
    ax.legend(fontsize=9)

    # Tether length
    ax = axes[2, 0]
    ax.plot(time, tetherLength, linewidth=2, color='brown')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tether Length (m)')
    ax.set_title('Tether Length', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=tetherLength.mean(), color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {tetherLength.mean():.1f} m')
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)

    # Elevation angle
    ax = axes[2, 1]
    ax.plot(time, elevationAngleDeg, linewidth=2, color='teal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Elevation Angle (deg)')
    ax.set_title('Elevation Angle', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=elevationAngleDeg.mean(), color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {elevationAngleDeg.mean():.1f} deg')
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)

    # Performance summary text box
    summaryText = (
        f"Performance Summary:\n"
        f"Cycle Power: {power['average_cycle_power_w']/1000:.2f} kW\n"
        f"Reel-Out Power: {power['average_reel_out_power_w']/1000:.2f} kW\n"
        f"Reel-In Power: {power['average_reel_in_power_w']/1000:.2f} kW\n"
        f"Cycle Time: {timing['cycle_time_s']:.2f} s\n"
        f"Reel-Out Time: {timing['reel_out_time_s']:.2f} s\n"
        f"Reel-In Time: {timing['reel_in_time_s']:.2f} s"
    )
    fig.text(0.99, 0.01, summaryText, fontsize=9, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# Optimizer sensitivity / efficiency indicators plot
# ---------------------------------------------------------------------------

def plot_sensitivity_efficiency_indicators(optimizer, i_x=0):
    """Sweep the search range of one optimization variable and plot efficiency indicators.

    Args:
        optimizer (Optimizer): Optimizer instance with evaluated results.
        i_x (int): Index of the optimization variable to sweep. Defaults to 0.

    Returns:
        tuple: (fig, ax) matplotlib figure and axes.
    """
    ref_point_label = "x_ref"

    if optimizer.x_opt_real_scale is not None:
        xRealScale = optimizer.x_opt_real_scale
    else:
        xRealScale = optimizer.x0_real_scale

    xRef = xRealScale[i_x]
    powerCycleRef = optimizer.eval_performance_indicators(
        xRealScale, scale_x=False)['average_power']['cycle']
    powerOutRef = optimizer.eval_performance_indicators(
        xRealScale, scale_x=False)['average_power']['out']
    xlabel = optimizer.OPT_VARIABLE_LABELS[i_x]

    lb, ub = optimizer.bounds_real_scale[i_x]
    xiSweep = np.linspace(lb, ub, 100)
    powerCycleNorm, powerOutNorm, g, activeG, dutyCycle, pumpingEff = (
        [], [], [], [], [], [])

    for xi in xiSweep:
        xFull = list(xRealScale)
        xFull[i_x] = xi

        try:
            resEval = optimizer.eval_fun(xFull, scale_x=False)
            kpis = optimizer.eval_performance_indicators(xFull, scale_x=False)
            powerCycleNorm.append(kpis['average_power']['cycle'] / powerCycleRef)
            if kpis['average_power']['out']:
                powerOutNorm.append(kpis['average_power']['out'] / powerOutRef)
            else:
                powerOutNorm.append(None)
            cons = resEval[1][optimizer.reduce_ineq_cons]
            g.append(resEval[1])
            activeG.append(any([c < -1e-6 for c in cons]))
            dutyCycle.append(kpis['duty_cycle'])
            pumpingEff.append(kpis['pumping_efficiency'])
        except Exception:
            powerCycleNorm.append(None)
            powerOutNorm.append(None)
            dutyCycle.append(None)
            pumpingEff.append(None)
            g.append(None)
            activeG.append(False)

    fig, ax = plt.subplots()
    ax.plot(xiSweep, powerCycleNorm, '--', label='normalized cycle power')
    ax.plot(xiSweep, powerOutNorm, '--', label='normalized traction power')
    ax.plot(xiSweep, dutyCycle, label='duty cycle')
    ax.plot(xiSweep, pumpingEff, label='pumping efficiency')

    ax.plot(xRef, 1, 'x', label=ref_point_label, markersize=12)
    ax.fill_between(xiSweep, 0, 1, where=activeG, alpha=0.4,
                    transform=ax.get_xaxis_transform())
    ax.set_xlabel(xlabel.replace('\n', ' '))
    ax.set_ylabel("Response [-]")
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("v={:.1f}m/s".format(optimizer.environment_state.wind_speed))
    plt.subplots_adjust(right=0.7)

    return fig, ax
