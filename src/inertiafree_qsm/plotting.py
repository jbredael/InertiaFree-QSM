"""Plotting functions for AWE power curve analysis and cycle diagnostics.

This module consolidates all plotting functionality used by the power curve
constructor and optimizer. Functions accept data dicts (as produced by the
YAML output) so that plotting is decoupled from the simulation objects.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    'font.family'         : 'serif',
    'font.size'           : 10,
    'axes.labelsize'      : 10,
    'xtick.labelsize'     : 9,
    'ytick.labelsize'     : 9,
    'legend.fontsize'     : 9,
    'axes.prop_cycle'     : mpl.cycler('color', [
        '#0072B2', '#D55E00', '#009E73',
        '#E69F00', '#CC79A7', '#56B4E9',
    ]),
    'lines.linewidth'     : 1.5,
    'axes.linewidth'      : 0.8,
    'xtick.direction'     : 'in',
    'ytick.direction'     : 'in',
    'xtick.minor.visible' : True,
    'ytick.minor.visible' : True,
    'xtick.major.size'    : 4,
    'ytick.major.size'    : 4,
    'xtick.minor.size'    : 2,
    'ytick.minor.size'    : 2,
    'xtick.major.width'   : 0.8,
    'ytick.major.width'   : 0.8,
    'xtick.minor.width'   : 0.6,
    'ytick.minor.width'   : 0.6,
    'lines.markersize'    : 4,
    'errorbar.capsize'    : 3,
    'axes.xmargin'        : 0.02,
    'axes.ymargin'        : 0.02,
    'legend.frameon'      : False,
    'savefig.bbox'        : 'tight',
    'savefig.dpi'         : 300,
    **(
        {'text.usetex'        : True,
         'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
         'pgf.texsystem'     : 'pdflatex',
         'pgf.rcfonts'       : False}
        if __import__('shutil').which('latex') else
        {'text.usetex'        : False,
         'mathtext.fontset'  : 'cm'}
    ),
})


# ---------------------------------------------------------------------------
# Helper: load YAML power curve data
# ---------------------------------------------------------------------------

def load_power_curve_data(file_path):
    """Load power curve data from a YAML file.

    If a companion ``<stem>.npz`` sidecar file is referenced in the YAML
    (``time_history_file`` key), time history arrays are loaded from it and
    injected back into each wind speed entry so that callers see a uniform
    ``time_history`` dict regardless of storage format.

    Args:
        file_path (str or Path): Path to the YAML file.

    Returns:
        dict: Power curve data with time histories populated.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    # Load binary sidecar if present and inject time histories
    npz_name = data.get('time_history_file')
    if npz_name:
        npz_path = file_path.parent / npz_name
        if npz_path.exists():
            TIME_HISTORY_CHANNELS = (
                'time', 'altitude', 'tether_force', 'power',
                'reel_speed', 'tether_length', 'elevation_angle', 'wind_speed',
            )
            npz = np.load(npz_path)
            for pc in data.get('power_curves', []):
                pid = pc['profile_id']
                for ws_idx, entry in enumerate(pc.get('wind_speed_data', [])):
                    prefix = f'p{pid}_ws{ws_idx}_'
                    th = {
                        ch: npz[f'{prefix}{ch}'].tolist()
                        for ch in TIME_HISTORY_CHANNELS
                        if f'{prefix}{ch}' in npz
                    }
                    if th:
                        entry['time_history'] = th

    return data


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
            if abs(entry['wind_speed'] - target_wind_speed) < 0.01:
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
    referenceHeight = metadata.get('wind_resource', {}).get('reference_height', 100)
    powerCurves = data.get('power_curves', [])

    if not powerCurves:
        raise ValueError("No power curve data found in file")

    nProfiles = len(powerCurves)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(name, fontsize=14, fontweight='bold')

    useMarkers = nProfiles <= 5
    markerSize = 4 if nProfiles <= 10 else 0
    lineWidth = 1.5 if nProfiles <= 10 else 1.0
    showComponentLegend = True

    for idx, profile in enumerate(powerCurves):
        profileId = profile.get('profile_id', 'Unknown')
        windSpeedData = profile.get('wind_speed_data', [])
        if not windSpeedData:
            continue

        windSpeeds = [e['wind_speed'] for e in windSpeedData]
        cyclePowerKw = [e['performance']['power']['average_cycle_power'] / 1000
                        for e in windSpeedData]
        reelOutPowerKw = [e['performance']['power']['average_reel_out_power'] / 1000
                          for e in windSpeedData]
        reelInPowerKw = [e['performance']['power']['average_reel_in_power'] / 1000
                         for e in windSpeedData]
        cycleTime = [e['performance']['timing']['cycle_time'] for e in windSpeedData]

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
            wingArea = metadata['model_config'].get('wing_area')
            if wingArea:
                AIR_DENSITY = 1.225
                powerCoeff = []
                for e in windSpeedData:
                    ws = e['wind_speed']
                    cpVal = e['performance']['power']['average_cycle_power']
                    if ws > 0 and cpVal >= 0:
                        availPower = 0.5 * AIR_DENSITY * wingArea * ws ** 3
                        powerCoeff.append(cpVal / availPower if availPower > 0 else 0)
                    else:
                        powerCoeff.append(0)
                axes[1, 1].plot(windSpeeds, powerCoeff, marker=marker,
                                linewidth=lineWidth, markersize=markerSize,
                                label=f'Profile {profileId}', color=color)

    # Axis labels and formatting
    windSpeedLabel = f'Wind Speed at {referenceHeight}m (m/s)'
    for axObj, ylabel, title in [
        (axes[0, 0], 'Cycle Power (kW)', 'Power Curve'),
        (axes[1, 0], 'Cycle Time (s)', 'Cycle Duration'),
    ]:
        axObj.set_xlabel(windSpeedLabel)
        axObj.set_ylabel(ylabel)
        axObj.set_title(title, fontweight='bold')
        axObj.grid(True, alpha=0.3)
        if nProfiles <= 15:
            axObj.legend(ncol=1 if nProfiles <= 8 else 2)

    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    axes[0, 1].set_xlabel(windSpeedLabel)
    axes[0, 1].set_ylabel('Power (kW)')
    axes[0, 1].set_title('Power Components (Profile 1)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    if 'model_config' in metadata and metadata['model_config'].get('wing_area'):
        axes[1, 1].set_xlabel(windSpeedLabel)
        axes[1, 1].set_ylabel('Power Coefficient')
        axes[1, 1].set_title('Power Coefficient', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if nProfiles <= 15:
            axes[1, 1].legend(ncol=1 if nProfiles <= 8 else 2)
    else:
        axes[1, 1].text(0.5, 0.5, 'Wing area not available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlabel(windSpeedLabel)
        axes[1, 1].set_ylabel('Power Coefficient')
        axes[1, 1].set_title('Power Coefficient', fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)

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
    referenceHeight = metadata.get('wind_resource', {}).get('reference_height', 100)
    profileLabel = f" (Profile {profile_id})" if profile_id is not None else ""

    wsData = find_wind_speed_data(data, wind_speed, profile_id)
    if wsData is None:
        print(f"Wind speed {wind_speed} m/s not found in data")
        return None

    sim_warning = None
    if not wsData['success']:
        sim_warning = f"Warning: simulation reported success=False for {wind_speed} m/s"
        print(sim_warning)

    performance = wsData['performance']
    power = performance['power']
    timing = performance['timing']
    timeHistory = wsData.get('time_history', {})

    if not timeHistory:
        print(f"No time history data available for wind speed {wind_speed} m/s")
        return None

    time = np.array(timeHistory.get('time', []))
    altitude = np.array(timeHistory.get('altitude', []))
    tetherForce = np.array(timeHistory.get('tether_force', []))
    powerInst = np.array(timeHistory.get('power', []))
    reelSpeed = np.array(timeHistory.get('reel_speed', []))
    tetherLength = np.array(timeHistory.get('tether_length', []))
    elevationAngleRad = np.array(timeHistory.get('elevation_angle', []))
    elevationAngleDeg = np.degrees(elevationAngleRad)

    windSpeedArr = np.array(timeHistory.get('wind_speed', []))


    fig = plt.figure(figsize=(16, 8))
    gs = mpl.gridspec.GridSpec(3, 3, figure=fig)

    axAlt  = fig.add_subplot(gs[0:1, 0:1])
    axTf   = fig.add_subplot(gs[0:1, 1:2])
    axPow  = fig.add_subplot(gs[0:1, 2:3])
    axReel = fig.add_subplot(gs[1:2, 0:1])
    axTL   = fig.add_subplot(gs[1:2, 1:2])
    axElev = fig.add_subplot(gs[1:2, 2:3])
    axTraj = fig.add_subplot(gs[2:3, 0:1])
    axWind = fig.add_subplot(gs[2:3, 1:2])
    axes   = np.array([[axAlt, axTf], [axPow, axReel],
                       [axTL, axElev], [axTraj, axWind]])

    title = (
        f'Detailed Cycle Analysis - Wind Speed: {wind_speed} m/s '
        f'at {referenceHeight}m{profileLabel}'
    )
    if sim_warning:
        title += '\n(simulation converged with relaxed errors)'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Phase boundary indices in the reordered time series
    # (traction -> retraction -> transition).
    reelOutTime = timing['reel_out_time']
    reelInTime = timing['reel_in_time']
    reelInStartIdx = np.searchsorted(time, reelOutTime)
    transitionStartIdx = np.searchsorted(time, reelOutTime + reelInTime)

    # Altitude
    ax = axAlt
    ax.plot(time, altitude, color='steelblue', label='Altitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Kite Altitude', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=time[reelInStartIdx], color='blue', linestyle=':', alpha=0.5, label='Start reel-in')
    ax.axvline(x=time[transitionStartIdx], color='green', linestyle=':', alpha=0.5, label='Start transition')
    ax.legend()

    # Tether force
    ax = axTf
    ax.plot(time, tetherForce / 1000, color='orangered', label='Tether force')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tether Force (kN)')
    ax.set_title('Tether Force', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=time[reelInStartIdx], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=time[transitionStartIdx], color='green', linestyle=':', alpha=0.5)
    ax.legend()

    # Instantaneous power
    ax = axPow
    ax.plot(time, powerInst / 1000, color='darkgreen', label='Power')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Instantaneous Power', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=time[reelInStartIdx], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=time[transitionStartIdx], color='green', linestyle=':', alpha=0.5)
    ax.legend()

    # Reel speed (left axis) + reeling factor (right axis)
    ax = axReel
    ax.plot(time, reelSpeed, color='purple', label='Reel speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reel Speed (m/s)')
    ax.set_title('Tether Reel Speed', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=time[reelInStartIdx], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=time[transitionStartIdx], color='green', linestyle=':', alpha=0.5)

    if windSpeedArr.size == reelSpeed.size and np.all(windSpeedArr > 0):
        reelingFactor = reelSpeed / windSpeedArr
        axRf = ax.twinx()
        axRf.plot(time, reelingFactor, color='purple', linestyle='--',
                  alpha=0.5, label='Reeling factor')
        axRf.set_ylabel('Reeling Factor (−)')
        axRf.tick_params(axis='y', labelcolor='purple')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = axRf.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)
    else:
        ax.legend()

    # Tether length
    ax = axTL
    ax.plot(time, tetherLength, color='brown', label='Tether length')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tether Length (m)')
    ax.set_title('Tether Length', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=time[reelInStartIdx], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=time[transitionStartIdx], color='green', linestyle=':', alpha=0.5)
    ax.legend()

    # Elevation angle
    ax = axElev
    ax.plot(time, elevationAngleDeg, color='teal', label='Elevation angle')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Elevation Angle (deg)')
    ax.set_title('Elevation Angle', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=time[reelInStartIdx], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=time[transitionStartIdx], color='green', linestyle=':', alpha=0.5)
    ax.legend()

    # 2-D trajectory (side view): horizontal distance vs altitude
    horizontalDist = tetherLength * np.cos(elevationAngleRad)

    ax = axTraj
    ax.plot(
        horizontalDist[:reelInStartIdx + 1],
        altitude[:reelInStartIdx + 1],
        color='steelblue', label='Reel-out', marker='o', markersize=2
    )
    ax.plot(
        horizontalDist[reelInStartIdx:transitionStartIdx + 1],
        altitude[reelInStartIdx:transitionStartIdx + 1],
        color='orangered', label='Reel-in', linestyle='--', marker='s', markersize=2
    )
    ax.plot(
        horizontalDist[transitionStartIdx:],
        altitude[transitionStartIdx:],
        color='green', label='Transition', linestyle='-', marker='^', markersize=2
    )
    ax.set_xlabel('Horizontal Distance (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Kite Trajectory (Side View)', fontweight='bold')
    ax.set_ylim(bottom=0, top=1.1 * max(max(altitude), max(horizontalDist)))
    ax.set_xlim(left=-1.1 * max(max(altitude), max(horizontalDist)), right=1.1 * max(max(altitude), max(horizontalDist)))
    # ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend()
    

    # Wind profile
    ax = axWind

    altitudesArr = np.array(data.get('altitudes', []))
    profileEntry = None
    for pc in data.get('power_curves', []):
        if profile_id is None or pc.get('profile_id') == profile_id:
            profileEntry = pc
            break

    if profileEntry is not None and altitudesArr.size > 0:
        uNorm = np.array(profileEntry.get('wind_profile', {}).get('u_normalized', []))
        if uNorm.size == altitudesArr.size:
            windProfileActual = wind_speed * uNorm
            ax.plot(windProfileActual, altitudesArr, color='steelblue', linewidth=1.5,
                    label='Wind speed')
            ax.plot(wind_speed, referenceHeight, 'o', color='darkred', markersize=7,
                    zorder=5, label=f'Ref. height ({referenceHeight:.0f} m)')
    else:
        ax.text(0.5, 0.5, 'Wind profile\nnot available',
                ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Wind Profile', fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.1 * max(max(altitude), max(horizontalDist)))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Sync wind-profile y-axis to trajectory (both start at 0).
    ax.set_ylim(axTraj.get_ylim())

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)

    if show_plot:
        plt.show()

    plt.close()

    return fig, axes


# ---------------------------------------------------------------------------
# Optimization evolution plot
# ---------------------------------------------------------------------------

_VAR_LABEL_MAP = {
    'reeling_speed_out': r'$v_\mathrm{reel,out}$ (m/s)',
    'reeling_speed_in': r'$v_\mathrm{reel,in}$ (m/s)',
    'frac_end': r'$\ell_\mathrm{end}/\ell_\mathrm{max}$ (−)',
    'frac_start': r'$\ell_\mathrm{start}/\ell_\mathrm{max}$ (−)',
}


def _build_var_label(name):
    """Return a display label for a decision-variable name.

    Args:
        name (str): Internal variable name from ``CycleOptimizer``.

    Returns:
        str: Human-readable (LaTeX-compatible) label.
    """
    if name in _VAR_LABEL_MAP:
        return _VAR_LABEL_MAP[name]
    if name.startswith('elevation_'):
        idx = name.split('_')[1]
        return rf'$\beta_{{{idx}}}$ (deg)'
    return name


def plot_optimization_evolution(history, wind_speed, var_names=None,
                                output_path=None, show_plot=True):
    """Plot how cycle power and decision variables evolve during optimization.

    Args:
        history (list): List of dicts with keys ``x`` (ndarray) and
            ``power`` (float, in W), one per objective evaluation.
        wind_speed (float): Reference wind speed [m/s] (used in title).
        var_names (list of str, optional): Internal variable names matching
            each element of ``h['x']``. When provided the decision-variable
            panel uses descriptive labels and handles any number of variables
            (including elevation angle entries). When ``None``, the four base
            variable labels are assumed for backward compatibility.
        output_path (str or Path, optional): Path to save the figure.
        show_plot (bool): Whether to display the plot. Defaults to True.

    Returns:
        tuple: (fig, axes) matplotlib figure and axes array.
    """
    if not history:
        return None

    powers = [h['power'] / 1000 for h in history]
    evaluations = list(range(1, len(powers) + 1))

    bestPowers = []
    currentBest = -np.inf
    for p in powers:
        currentBest = max(currentBest, p)
        bestPowers.append(currentBest)

    # Build labels for every decision variable.
    # If var_names is not supplied fall back to the legacy four-variable list.
    nVars = len(history[0]['x'])
    if var_names is not None:
        labels = [_build_var_label(n) for n in var_names]
    else:
        _legacy = [r'$f_\mathrm{out}$', r'$f_\mathrm{in}$',
                   r'$\ell_\mathrm{end}$', r'$\ell_\mathrm{start}$']
        labels = _legacy[:nVars]

    # Separate base variables from elevation angle variables for cleaner panels.
    elevIndices = [i for i, n in enumerate(var_names or [])
                   if n.startswith('elevation_')]
    baseIndices = [i for i in range(nVars) if i not in elevIndices]

    nPanels = 2
    if elevIndices:
        nPanels = 3   # power | base vars | elevation vars
    fig, axes = plt.subplots(nPanels, 1, figsize=(10, 4 * nPanels))
    if nPanels == 1:
        axes = [axes]
    fig.suptitle(
        f'Optimization Evolution — {wind_speed:.1f} m/s',
        fontsize=13, fontweight='bold',
    )

    # --- Power evolution ---
    ax = axes[0]
    ax.plot(evaluations, powers, 'o', alpha=0.35, markersize=3, label='Evaluation')
    ax.plot(evaluations, bestPowers, '-', linewidth=2, label='Best so far')
    maxPow = max(powers) if powers else 1.0
    ax.set_ylim(bottom=-0.1 * abs(maxPow), top=1.1 * abs(maxPow))
    ax.set_xlabel('Function evaluation')
    ax.set_ylabel('Cycle power (kW)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Base decision-variable evolution ---
    ax = axes[1]
    for i in baseIndices:
        values = [h['x'][i] for h in history]
        ax.plot(evaluations, values, 'o-', alpha=0.6, markersize=3, label=labels[i])
    ax.set_xlabel('Function evaluation')
    ax.set_ylabel('Variable value')
    ax.set_title('Base decision variables', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Elevation angle evolution (only when present) ---
    if elevIndices:
        ax = axes[2]
        for i in elevIndices:
            values = [h['x'][i] for h in history]
            ax.plot(evaluations, values, 'o-', alpha=0.6, markersize=3,
                    label=labels[i])
        ax.set_xlabel('Function evaluation')
        ax.set_ylabel('Elevation angle (deg)')
        ax.set_title('Traction elevation angles', fontweight='bold')
        ax.legend(ncol=max(1, len(elevIndices) // 8))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)

    if show_plot:
        plt.show()

    plt.close()

    return fig, axes
