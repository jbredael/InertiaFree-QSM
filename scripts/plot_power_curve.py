"""Plot power curves from YAML data files.

This script reads power curve data from a YAML file and generates plots
showing power vs wind speed and other relevant metrics.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Specify the path to your power curve YAML file here
POWER_CURVE_FILE_DIRECT = r'results/power_curves_direct_simulation.yml'
POWER_CURVE_FILE_OPTIMIZED = r'results/power_curves_optimized.yml'
RESULTS_DIR = Path('results')

def load_power_curve_data(filePath):
    """Load power curve data from a YAML file.

    Args:
        filePath (str or Path): Path to the YAML file containing power curve data.

    Returns:
        dict: Power curve data loaded from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    filePath = Path(filePath)
    
    if not filePath.exists():
        raise FileNotFoundError(f"File not found: {filePath}")
    
    with open(filePath, 'r') as file:
        data = yaml.safe_load(file)
    
    return data


def plot_power_curve(filePath, output_path=None, show_plot=True):
    """Plot power curve from YAML data file.

    Args:
        filePath (str or Path): Path to the YAML file containing power curve data.
        output_path (str or Path, optional): Path to save the figure. 
            If None, figure is not saved. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. 
            Defaults to True.

    Returns:
        None
    """
    # Load data
    data = load_power_curve_data(filePath)
    
    # Extract metadata
    metadata = data.get('metadata', {})
    name = metadata.get('name', 'Power Curve')
    referenceHeight = metadata.get('wind_resource', {}).get('reference_height_m', 100)
    
    # Extract power curves
    powerCurves = data.get('power_curves', [])
    
    if not powerCurves:
        raise ValueError("No power curve data found in file")
    
    n_profiles = len(powerCurves)
    print(f"Plotting {n_profiles} wind profile(s)")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(name, fontsize=16, fontweight='bold')
    
    # Determine marker and line style based on number of profiles
    use_markers = n_profiles <= 5
    marker_size = 4 if n_profiles <= 10 else 0
    line_width = 1.5 if n_profiles <= 10 else 1.0
    
    # Track if we need to show the "first profile" legend for plot 2
    show_component_legend = True
    
    # Plot for each profile
    for idx, profile in enumerate(powerCurves):
        profileId = profile.get('profile_id', 'Unknown')
        windSpeedData = profile.get('wind_speed_data', [])
        
        if not windSpeedData:
            print(f"No wind speed data for profile {profileId}")
            continue
        
        # Extract data arrays
        windSpeeds = [entry['wind_speed_m_s'] for entry in windSpeedData]
        cyclePower = [entry['performance']['power']['average_cycle_power_w'] for entry in windSpeedData]
        reelOutPower = [entry['performance']['power']['average_reel_out_power_w'] for entry in windSpeedData]
        reelInPower = [entry['performance']['power']['average_reel_in_power_w'] for entry in windSpeedData]
        cycleTime = [entry['performance']['timing']['cycle_time_s'] for entry in windSpeedData]
        
        # Convert power to kW
        cyclePowerKw = [p / 1000 for p in cyclePower]
        reelOutPowerKw = [p / 1000 for p in reelOutPower]
        reelInPowerKw = [p / 1000 for p in reelInPower]
        
        # Get color for this profile
        color = plt.cm.tab20(idx % 20) if n_profiles > 10 else None
        
        # Plot 1: Cycle power vs wind speed
        marker = 'o' if use_markers else ''
        axes[0, 0].plot(windSpeeds, cyclePowerKw, marker=marker, linewidth=line_width, 
                        markersize=marker_size, label=f'Profile {profileId}', color=color)
        
        # Plot 2: Power components (only for first profile to avoid clutter)
        if show_component_legend:
            axes[0, 1].plot(windSpeeds, reelOutPowerKw, 's-', linewidth=1.5, 
                            markersize=4, label='Reel-out', alpha=0.7)
            axes[0, 1].plot(windSpeeds, reelInPowerKw, '^-', linewidth=1.5, 
                            markersize=4, label='Reel-in', alpha=0.7)
            axes[0, 1].plot(windSpeeds, cyclePowerKw, 'o-', linewidth=1.5, 
                            markersize=4, label='Cycle', alpha=0.7)
            show_component_legend = False
        else:
            axes[0, 1].plot(windSpeeds, reelOutPowerKw, 's-', linewidth=1.0, 
                            markersize=2, alpha=0.4, color='C0')
            axes[0, 1].plot(windSpeeds, reelInPowerKw, '^-', linewidth=1.0, 
                            markersize=2, alpha=0.4, color='C1')
            axes[0, 1].plot(windSpeeds, cyclePowerKw, 'o-', linewidth=1.0, 
                            markersize=2, alpha=0.4, color='C2')
        
        # Plot 3: Cycle time
        axes[1, 0].plot(windSpeeds, cycleTime, marker=marker, linewidth=line_width, 
                        markersize=marker_size, label=f'Profile {profileId}', color=color)
        
        # Plot 4: Power coefficient
        if 'model_config' in metadata:
            wingArea = metadata['model_config'].get('wing_area_m2', None)
            if wingArea:
                airDensity = 1.225  # kg/m³ at sea level
                powerCoeff = []
                for entry in windSpeedData:
                    ws = entry['wind_speed_m_s']
                    cp_val = entry['performance']['power']['average_cycle_power_w']
                    if ws > 0 and cp_val >= 0:
                        availablePower = 0.5 * airDensity * wingArea * ws**3
                        cp = cp_val / availablePower if availablePower > 0 else 0
                        powerCoeff.append(cp)
                    else:
                        powerCoeff.append(0)
                
                axes[1, 1].plot(windSpeeds, powerCoeff, marker=marker, linewidth=line_width, 
                                markersize=marker_size, label=f'Profile {profileId}', color=color)
    
    # Configure Plot 1
    axes[0, 0].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
    axes[0, 0].set_ylabel('Cycle Power (kW)', fontsize=11)
    axes[0, 0].set_title('Power Curve', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    if n_profiles <= 15:
        axes[0, 0].legend(fontsize=8, ncol=1 if n_profiles <= 8 else 2)
    
    # Configure Plot 2
    axes[0, 1].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
    axes[0, 1].set_ylabel('Power (kW)', fontsize=11)
    axes[0, 1].set_title(f'Power Components (Profile 1)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Configure Plot 3
    axes[1, 0].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
    axes[1, 0].set_ylabel('Cycle Time (s)', fontsize=11)
    axes[1, 0].set_title('Cycle Duration', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    if n_profiles <= 15:
        axes[1, 0].legend(fontsize=8, ncol=1 if n_profiles <= 8 else 2)
    
    # Configure Plot 4
    if 'model_config' in metadata and metadata['model_config'].get('wing_area_m2'):
        axes[1, 1].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
        axes[1, 1].set_ylabel('Power Coefficient', fontsize=11)
        axes[1, 1].set_title('Power Coefficient', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if n_profiles <= 15:
            axes[1, 1].legend(fontsize=8, ncol=1 if n_profiles <= 8 else 2)
    else:
        axes[1, 1].text(0.5, 0.5, 'Wing area not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Power Coefficient', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure if output path is specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    # Display plot
    if show_plot:
        plt.show()


def main():
    """Main function to run the power curve plotting script."""
    # Plot the power curve
    plot_power_curve(
        filePath=POWER_CURVE_FILE_DIRECT,
        output_path=RESULTS_DIR / 'power_curve.png',
        show_plot=True
    )

    plot_power_curve(
        filePath=POWER_CURVE_FILE_OPTIMIZED,
        output_path=RESULTS_DIR / 'power_curve_optimized.png',
        show_plot=True
    )


if __name__ == '__main__':
    main()
