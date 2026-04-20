"""Plot detailed cycle statistics for selected wind speeds in a power curve file.

Edit the INPUT section below, then run:
    python scripts/plot_cycle_details.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inertiafree_qsm.plotting import load_power_curve_data, plot_cycle_detail

# ── INPUT ─────────────────────────────────────────────────────────────────────
# POWER_CURVE_FILE = Path("results/power_curves_direct_simulation.yml")
POWER_CURVE_FILE = Path("results/power_curves_optimized.yml")
PROFILE_ID = 1           # int profile_id to filter on; None = use all profiles
WIND_SPEED_MIN = None    # float [m/s], None = no lower limit
WIND_SPEED_MAX = None    # float [m/s], None = no upper limit
OUTPUT_DIR = Path("results/cycle_details_optimized")  # folder to save PDFs; None = show only
SHOW_PLOT = False
# ──────────────────────────────────────────────────────────────────────────────


def plot_all_cycle_details(
    power_curve_file,
    profile_id=None,
    wind_speed_min=None,
    wind_speed_max=None,
    output_dir=None,
    show_plot=True,
):
    """Plot cycle detail figures for all matching wind speeds in a power curve file.

    Args:
        power_curve_file (str or Path): Path to the power curve YAML file.
        profile_id (int, optional): Profile ID to filter on. Defaults to None
            (all profiles).
        wind_speed_min (float, optional): Minimum wind speed to include [m/s].
            Defaults to None (no lower limit).
        wind_speed_max (float, optional): Maximum wind speed to include [m/s].
            Defaults to None (no upper limit).
        output_dir (str or Path, optional): Directory to save output PDFs. If
            None, figures are only shown interactively. Defaults to None.
        show_plot (bool): Whether to display figures interactively.
    """
    data = load_power_curve_data(power_curve_file)
    n_plotted = 0

    for profile in data.get("power_curves", []):
        pid = profile.get("profile_id", 1)
        if profile_id is not None and pid != profile_id:
            continue

        for entry in profile.get("wind_speed_data", []):
            ws = entry["wind_speed"]

            if wind_speed_min is not None and ws < wind_speed_min:
                continue
            if wind_speed_max is not None and ws > wind_speed_max:
                continue

            if not entry.get("time_history"):
                print(f"  Skipping v={ws:.1f} m/s (profile {pid}) — no time history")
                continue

            fig_path = None
            if output_dir is not None:
                dir_path = Path(output_dir)
                dir_path.mkdir(parents=True, exist_ok=True)
                fig_path = dir_path / f"cycle_detail_profile{pid}_v{ws:.1f}ms.pdf"

            print(f"  Plotting cycle detail: v={ws:.1f} m/s, profile {pid}")
            try:
                plot_cycle_detail(
                    power_curve_file,
                    ws,
                    profile_id=pid,
                    output_path=fig_path,
                    show_plot=show_plot,
                )
            except Exception as e:
                print(f"    Error plotting v={ws:.1f} m/s, profile {pid}: {e}")
                continue
            n_plotted += 1

    print(f"Done — plotted {n_plotted} wind speed(s).")


if __name__ == "__main__":
    plot_all_cycle_details(
        POWER_CURVE_FILE,
        PROFILE_ID,
        WIND_SPEED_MIN,
        WIND_SPEED_MAX,
        OUTPUT_DIR,
        SHOW_PLOT,
    )
