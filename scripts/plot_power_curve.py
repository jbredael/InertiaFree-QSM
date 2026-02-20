"""Plot power curves from YAML data files.

This script reads power curve data from a YAML file and generates plots
showing power vs wind speed and other relevant metrics.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Specify the path to your power curve YAML file here
POWER_CURVE_FILE = r'results/power_curves_direct_simulation.yml'


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
    
    # Extract wind speeds and power data
    windSpeeds = data.get('reference_wind_speeds_m_s', [])
    powerCurves = data.get('power_curves', [])
    
    if not powerCurves:
        raise ValueError("No power curve data found in file")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(name, fontsize=16, fontweight='bold')
    
    # Plot for each profile (typically just one)
    for profile in powerCurves:
        profileId = profile.get('profile_id', 'Unknown')
        cyclePower = profile.get('cycle_power_w', [])
        reelOutPower = profile.get('reel_out_power_w', [])
        reelInPower = profile.get('reel_in_power_w', [])
        cycleTime = profile.get('cycle_time_s', [])
        
        # Convert power to kW
        cyclePowerKw = [p / 1000 for p in cyclePower]
        reelOutPowerKw = [p / 1000 for p in reelOutPower]
        reelInPowerKw = [p / 1000 for p in reelInPower]
        
        # Plot 1: Cycle power vs wind speed
        axes[0, 0].plot(windSpeeds, cyclePowerKw, 'o-', linewidth=2, 
                        markersize=6, label=f'Profile {profileId}')
        axes[0, 0].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
        axes[0, 0].set_ylabel('Cycle Power (kW)', fontsize=11)
        axes[0, 0].set_title('Power Curve', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 2: Reel-out and reel-in power
        axes[0, 1].plot(windSpeeds, reelOutPowerKw, 's-', linewidth=2, 
                        markersize=5, label='Reel-out Power')
        axes[0, 1].plot(windSpeeds, reelInPowerKw, '^-', linewidth=2, 
                        markersize=5, label='Reel-in Power')
        axes[0, 1].plot(windSpeeds, cyclePowerKw, 'o-', linewidth=2, 
                        markersize=5, label='Cycle Power')
        axes[0, 1].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
        axes[0, 1].set_ylabel('Power (kW)', fontsize=11)
        axes[0, 1].set_title('Power Components', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 3: Cycle time
        axes[1, 0].plot(windSpeeds, cycleTime, 'o-', linewidth=2, 
                        markersize=6, color='green')
        axes[1, 0].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
        axes[1, 0].set_ylabel('Cycle Time (s)', fontsize=11)
        axes[1, 0].set_title('Cycle Duration', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Power coefficient (if possible to calculate)
        # Calculate approximate power coefficient
        if 'model_config' in metadata:
            wingArea = metadata['model_config'].get('wing_area_m2', None)
            if wingArea:
                airDensity = 1.225  # kg/m³ at sea level
                powerCoeff = []
                for i, ws in enumerate(windSpeeds):
                    if ws > 0 and cyclePower[i] >= 0:
                        availablePower = 0.5 * airDensity * wingArea * ws**3
                        cp = cyclePower[i] / availablePower if availablePower > 0 else 0
                        powerCoeff.append(cp)
                    else:
                        powerCoeff.append(0)
                
                axes[1, 1].plot(windSpeeds, powerCoeff, 'o-', linewidth=2, 
                                markersize=6, color='purple')
                axes[1, 1].set_xlabel(f'Wind Speed at {referenceHeight}m (m/s)', fontsize=11)
                axes[1, 1].set_ylabel('Power Coefficient', fontsize=11)
                axes[1, 1].set_title('Power Coefficient', fontsize=12, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Wing area not available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Power Coefficient', fontsize=12, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'Model config not available', 
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
        filePath=POWER_CURVE_FILE,
        output_path=None,  # Set to a path like 'results/plot.png' to save
        show_plot=True
    )


if __name__ == '__main__':
    main()
