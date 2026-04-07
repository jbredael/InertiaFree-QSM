"""Compare optimization results across different time-step configurations.

Edit the DATASETS list below to point to the .npz and companion .yml files for
each configuration to compare, then run:
    python scripts/compare_timestep_results.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── INPUT ─────────────────────────────────────────────────────────────────────
DATASETS = [
    {
        "label": "dt = 0.25 / 0.05 / 0.25 s",
        "npz": Path("results/0.25 0.05 0.25/power_curve_single_point_optimized.npz"),
        "yml": Path("results/0.25 0.05 0.25/power_curve_single_point_optimized.yml"),
    },
    {
        "label": "dt = 1.0 / 0.2 / 1.0 s",
        "npz": Path("results/1.0 0.2 1.0/power_curve_single_point_optimized.npz"),
        "yml": Path("results/1.0 0.2 1.0/power_curve_single_point_optimized.yml"),
    },
    {
        "label": "dt = 2.5 / 0.5 / 2.5 s",
        "npz": Path("results/2.5 0.5 2.5/power_curve_single_point_optimized.npz"),
        "yml": Path("results/2.5 0.5 2.5/power_curve_single_point_optimized.yml"),
    },
    {
        "label": "dt = 2.5 / 0.2 / 2.5 s",
        "npz": Path("results/2.5 0.2 2.5/power_curve_single_point_optimized.npz"),
        "yml": Path("results/2.5 0.2 2.5/power_curve_single_point_optimized.yml"),
    },
    {
        "label": "dt = 5.0 / 1.0 / 5.0 s",
        "npz": Path("results/5.0 1.0 5.0/power_curve_single_point_optimized.npz"),
        "yml": Path("results/5.0 1.0 5.0/power_curve_single_point_optimized.yml"),
    },
]
PROFILE_ID = 1          # Wind profile / cluster ID to compare
OUTPUT_PATH = None      # Path to save the figure (e.g. Path("results/comparison.pdf")); None = show only
SHOW_PLOT = True
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(ds, profileId=1):
    """Load a dataset from its YAML and NPZ files.

    Args:
        ds (dict): Dataset dict with 'label', 'yml', and 'npz' keys.
        profileId (int): Wind profile / cluster ID to load. Defaults to 1.

    Returns:
        dict: Loaded dataset with keys 'label', 'wind_speeds', 'cycle_power',
            'reel_out_power', 'reel_in_power', 'cycle_time', 'reel_out_time',
            'reel_in_time', and 'time_histories'.
    """
    ymlPath = Path(ds["yml"])
    npzPath = Path(ds["npz"])

    with open(ymlPath) as f:
        ymlData = yaml.safe_load(f)

    windSpeeds = np.array(ymlData["reference_wind_speeds"], dtype=float)

    # Extract per-wind-speed performance from YAML for the requested profile.
    cyclePower = []
    reelOutPower = []
    reelInPower = []
    cycleTime = []
    reelOutTime = []
    reelInTime = []

    profileData = None
    for pc in ymlData.get("power_curves", []):
        if pc.get("profile_id") == profileId:
            profileData = pc
            break

    if profileData is None:
        raise ValueError(
            f"Profile ID {profileId} not found in {ymlPath}."
        )

    for entry in profileData["wind_speed_data"]:
        perf = entry["performance"]
        cyclePower.append(perf["power"]["average_cycle_power"] / 1e3)   # kW
        reelOutPower.append(perf["power"]["average_reel_out_power"] / 1e3)
        reelInPower.append(perf["power"]["average_reel_in_power"] / 1e3)
        cycleTime.append(perf["timing"]["cycle_time"])
        reelOutTime.append(perf["timing"]["reel_out_time"])
        reelInTime.append(perf["timing"]["reel_in_time"])

    # Load NPZ time histories.
    npzData = np.load(npzPath)
    timeHistories = {}
    for wsIdx in range(len(windSpeeds)):
        prefix = f"p{profileId}_ws{wsIdx}_"
        keys = [k for k in npzData.files if k.startswith(prefix)]
        if not keys:
            continue
        timeHistories[wsIdx] = {
            k[len(prefix):]: npzData[k] for k in keys
        }

    return {
        "label": ds["label"],
        "wind_speeds": windSpeeds,
        "cycle_power": np.array(cyclePower),
        "reel_out_power": np.array(reelOutPower),
        "reel_in_power": np.array(reelInPower),
        "cycle_time": np.array(cycleTime),
        "reel_out_time": np.array(reelOutTime),
        "reel_in_time": np.array(reelInTime),
        "time_histories": timeHistories,
    }


def extract_optimized_variables(th):
    """Extract scalar optimized variable estimates from a time-history dict.

    The traction phase is identified by positive reeling speed; the retraction
    phase by negative reeling speed.

    Args:
        th (dict): Time-history dict with arrays 'reel_speed', 'tether_length',
            and 'elevation_angle'.

    Returns:
        dict: Dict with keys 'mean_rs_traction', 'mean_rs_retraction',
            'max_tether_length', 'min_tether_length', 'mean_elevation_traction'.
            Returns None if time history is empty.
    """
    if not th:
        return None

    reelSpeed = th["reel_speed"]
    tetherLength = th["tether_length"]
    elevAngle = np.degrees(th["elevation_angle"])

    isTraction = reelSpeed > 0
    isRetraction = reelSpeed < 0

    meanRsTraction = float(np.mean(reelSpeed[isTraction])) if np.any(isTraction) else float("nan")
    meanRsRetraction = float(np.mean(np.abs(reelSpeed[isRetraction]))) if np.any(isRetraction) else float("nan")
    maxTetherLength = float(np.max(tetherLength))
    minTetherLength = float(np.min(tetherLength))
    meanElevTraction = float(np.mean(elevAngle[isTraction])) if np.any(isTraction) else float("nan")

    return {
        "mean_rs_traction": meanRsTraction,
        "mean_rs_retraction": meanRsRetraction,
        "max_tether_length": maxTetherLength,
        "min_tether_length": minTetherLength,
        "mean_elevation_traction": meanElevTraction,
    }


def build_opt_var_arrays(dataset):
    """Build per-wind-speed arrays of optimized variable estimates.

    Args:
        dataset (dict): Loaded dataset dict as returned by load_dataset.

    Returns:
        dict: Dict mapping variable name to numpy array (one value per wind speed).
    """
    nWs = len(dataset["wind_speeds"])
    keys = ["mean_rs_traction", "mean_rs_retraction",
            "max_tether_length", "min_tether_length", "mean_elevation_traction"]
    arrays = {k: np.full(nWs, np.nan) for k in keys}

    for wsIdx, th in dataset["time_histories"].items():
        optVars = extract_optimized_variables(th)
        if optVars is not None:
            for k in keys:
                arrays[k][wsIdx] = optVars[k]

    return arrays


def plot_comparison(datasets, outputPath=None, showPlot=True):
    """Plot cycle power and optimized variable comparison across datasets.

    Creates a figure with six subplots:
    - Average cycle power [kW]
    - Average reel-out power [kW]
    - Mean traction reeling speed [m/s]
    - Mean retraction reeling speed [m/s]
    - Max / min tether length [m]
    - Mean elevation angle during traction [deg]

    Args:
        datasets (list): List of loaded dataset dicts.
        outputPath (Path, optional): File path to save the figure. Defaults to None.
        showPlot (bool): Whether to display the figure. Defaults to True.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Time-step sensitivity: cycle power & optimized variables", fontsize=13)

    markers = ["o", "s", "^", "D", "v", "P", "*"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, ds in enumerate(datasets):
        ws = ds["wind_speeds"]
        label = ds["label"]
        style = dict(marker=markers[i % len(markers)], color=colors[i % len(colors)],
                     linewidth=1.5, markersize=5)

        optVars = build_opt_var_arrays(ds)

        # Row 0, col 0 — cycle power
        axes[0, 0].plot(ws, ds["cycle_power"], label=label, **style)

        # Row 0, col 1 — reel-out power
        axes[0, 1].plot(ws, ds["reel_out_power"], label=label, **style)

        # Row 1, col 0 — mean traction reeling speed
        axes[1, 0].plot(ws, optVars["mean_rs_traction"], label=label, **style)

        # Row 1, col 1 — mean retraction reeling speed
        axes[1, 1].plot(ws, optVars["mean_rs_retraction"], label=label, **style)

        # Row 2, col 0 — tether lengths (max and min, same color, different linestyle)
        axes[2, 0].plot(ws, optVars["max_tether_length"], label=f"{label} (max)",
                        linestyle="-", **style)
        axes[2, 0].plot(ws, optVars["min_tether_length"], label=f"{label} (min)",
                        linestyle="--", color=colors[i % len(colors)],
                        marker=markers[i % len(markers)], markersize=5, linewidth=1.5)

        # Row 2, col 1 — mean elevation angle during traction
        axes[2, 1].plot(ws, optVars["mean_elevation_traction"], label=label, **style)

    xlabels = "Wind speed [m/s]"
    titles_ylabels = [
        ("Average cycle power",           "Power [kW]"),
        ("Average reel-out power",         "Power [kW]"),
        ("Mean traction reeling speed",    "Reeling speed [m/s]"),
        ("Mean retraction reeling speed",  "Reeling speed [m/s]"),
        ("Tether length (max / min)",      "Length [m]"),
        ("Mean elevation angle (traction)","Elevation angle [°]"),
    ]

    for ax, (title, ylabel) in zip(axes.flat, titles_ylabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabels)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    if outputPath is not None:
        outputPath = Path(outputPath)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, dpi=150)
        print(f"Figure saved to {outputPath}")

    if showPlot:
        plt.show()

    return fig


def main():
    """Load all datasets and generate the comparison figure."""
    datasets = []
    for ds in DATASETS:
        print(f"Loading: {ds['label']}")
        datasets.append(load_dataset(ds, profileId=PROFILE_ID))

    plot_comparison(datasets, outputPath=OUTPUT_PATH, showPlot=SHOW_PLOT)


if __name__ == "__main__":
    main()
