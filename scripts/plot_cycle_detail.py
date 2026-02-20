"""Plot detailed cycle information for a specific wind speed.

This script reads power curve data from a YAML file and generates detailed plots
showing all time history variables for a specific wind speed.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Specify the path to your power curve YAML file here
POWER_CURVE_FILE = r'results/power_curves_direct_simulation.yml'
WIND_SPEED = 10.0  # m/s


def load_power_curve_data(filePath):
    """Load power curve data from a YAML file.

    Args:
        filePath (str or Path): Path to the YAML file containing power curve data.

    Returns:
        dict: Power curve data loaded from the file.
    """
    filePath = Path(filePath)
    
    if not filePath.exists():
        raise FileNotFoundError(f"File not found: {filePath}")
    
    with open(filePath, 'r') as file:
        data = yaml.safe_load(file)
    
    return data


def find_wind_speed_data(data, targetWindSpeed):
    """Find the data for a specific wind speed.

    Args:
        data (dict): Power curve data.
        targetWindSpeed (float): Target wind speed in m/s.

    Returns:
        dict: Wind speed data entry, or None if not found.
    """
    powerCurves = data.get('power_curves', [])
    
    for profile in powerCurves:
        windSpeedData = profile.get('wind_speed_data', [])
        for entry in windSpeedData:
            if abs(entry['wind_speed_m_s'] - targetWindSpeed) < 0.01:
                return entry
    
    return None


def plot_cycle_detail(filePath, windSpeed, outputPath=None, showPlot=True):
    """Plot detailed cycle information for a specific wind speed.

    Args:
        filePath (str or Path): Path to the YAML file containing power curve data.
        windSpeed (float): Wind speed to plot in m/s.
        outputPath (str or Path, optional): Path to save the figure. 
            If None, figure is not saved. Defaults to None.
        showPlot (bool, optional): Whether to display the plot. 
            Defaults to True.
    """
    # Load data
    data = load_power_curve_data(filePath)
    
    # Extract metadata
    metadata = data.get('metadata', {})
    name = metadata.get('name', 'Power Curve')
    referenceHeight = metadata.get('wind_resource', {}).get('reference_height_m', 100)
    
    # Find the specific wind speed data
    wsData = find_wind_speed_data(data, windSpeed)
    
    if wsData is None:
        print(f"Wind speed {windSpeed} m/s not found in data")
        return
    
    if not wsData['success']:
        print(f"Simulation failed for wind speed {windSpeed} m/s")
        return
    
    # Extract performance data
    performance = wsData['performance']
    power = performance['power']
    timing = performance['timing']
    
    # Extract time history
    timeHistory = wsData.get('time_history', {})
    
    if not timeHistory:
        print(f"No time history data available for wind speed {windSpeed} m/s")
        return
    
    time = np.array(timeHistory.get('time_s', []))
    altitude = np.array(timeHistory.get('altitude_m', []))
    tetherForce = np.array(timeHistory.get('tether_force_n', []))
    power_inst = np.array(timeHistory.get('power_w', []))
    reelSpeed = np.array(timeHistory.get('reel_speed_m_s', []))
    tetherLength = np.array(timeHistory.get('tether_length_m', []))
    elevationAngle = np.array(timeHistory.get('elevation_angle_rad', []))
    
    # Convert elevation angle to degrees
    elevationAngleDeg = np.degrees(elevationAngle)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Detailed Cycle Analysis - Wind Speed: {windSpeed} m/s at {referenceHeight}m', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Altitude vs Time
    ax = axes[0, 0]
    ax.plot(time, altitude, linewidth=2, color='steelblue')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Altitude (m)', fontsize=11)
    ax.set_title('Kite Altitude', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=altitude.mean(), color='r', linestyle='--', alpha=0.5, 
               label=f'Mean: {altitude.mean():.1f} m')
    ax.legend(fontsize=9)
    
    # Plot 2: Tether Force vs Time
    ax = axes[0, 1]
    ax.plot(time, tetherForce / 1000, linewidth=2, color='orangered')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Tether Force (kN)', fontsize=11)
    ax.set_title('Tether Force', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=(tetherForce.mean() / 1000), color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {tetherForce.mean()/1000:.1f} kN')
    ax.legend(fontsize=9)
    
    # Plot 3: Power vs Time
    ax = axes[1, 0]
    ax.plot(time, power_inst / 1000, linewidth=2, color='darkgreen')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Power (kW)', fontsize=11)
    ax.set_title('Instantaneous Power', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axhline(y=(power['average_cycle_power_w'] / 1000), color='r', linestyle='--', alpha=0.5,
               label=f"Avg Cycle: {power['average_cycle_power_w']/1000:.1f} kW")
    
    # Mark reel-out and reel-in phases
    reelOutTime = timing['reel_out_time_s']
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5, label='Phase transition')
    ax.text(reelOutTime/2, ax.get_ylim()[1]*0.9, 'Reel-Out', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(reelOutTime + (time[-1] - reelOutTime)/2, ax.get_ylim()[1]*0.9, 'Reel-In', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.legend(fontsize=9)
    
    # Plot 4: Reel Speed vs Time
    ax = axes[1, 1]
    ax.plot(time, reelSpeed, linewidth=2, color='purple')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Reel Speed (m/s)', fontsize=11)
    ax.set_title('Tether Reel Speed', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5, label='Phase transition')
    ax.legend(fontsize=9)
    
    # Plot 5: Tether Length vs Time
    ax = axes[2, 0]
    ax.plot(time, tetherLength, linewidth=2, color='brown')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Tether Length (m)', fontsize=11)
    ax.set_title('Tether Length', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=tetherLength.mean(), color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {tetherLength.mean():.1f} m')
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Plot 6: Elevation Angle vs Time
    ax = axes[2, 1]
    ax.plot(time, elevationAngleDeg, linewidth=2, color='teal')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Elevation Angle (°)', fontsize=11)
    ax.set_title('Elevation Angle', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=elevationAngleDeg.mean(), color='r', linestyle='--', alpha=0.5,
               label=f'Mean: {elevationAngleDeg.mean():.1f}°')
    ax.axvline(x=reelOutTime, color='blue', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Add text box with performance summary
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
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure if output path is specified
    if outputPath:
        outputPath = Path(outputPath)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outputPath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {outputPath}")
    
    # Display plot
    if showPlot:
        plt.show()


def main():
    """Main function to run the detailed cycle plotting script."""
    # Plot the cycle detail
    plot_cycle_detail(
        filePath=POWER_CURVE_FILE,
        windSpeed=WIND_SPEED,
        outputPath='results/cycle_detail_10ms.png',  # Set to None to not save
        showPlot=True
    )


if __name__ == '__main__':
    main()
