"""Plotting utilities for AWE power model visualization.

This module provides plotting functions that can be used by external toolchains
to visualize power curve analysis results.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid QPainter errors

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_comprehensive_analysis(
    data: Dict[str, np.ndarray],
    model_params: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Create comprehensive subplot analysis of power curve.

    This function creates a multi-panel figure showing:
    - Power output (cycle, reel-out, reel-in)
    - Cycle times (total, reel-out, reel-in)
    - Tether forces
    - Reel speeds and elevation angles
    - Reeling factors (gamma)
    - Energy per cycle (reel-out, reel-in, net)

    Args:
        data (Dict[str, np.ndarray]): Dictionary with power curve data arrays.
            Must contain: windSpeed, power, reelOutPower, reelInPower,
            reelOutTime, reelInTime, tetherForceOut, tetherForceIn,
            reelOutSpeed, reelInSpeed, gammaOut, gammaIn
        model_params (Dict, optional): Model parameters for annotations.
            Can include: wingArea, nominalGeneratorPower, nominalTetherForce,
            nominalWindSpeedForce, nominalWindSpeedPower, cutOutWindSpeed,
            elevationAngleOut, elevationAngleIn
        save_path (str, optional): Path to save figure. If None, not saved.
        show (bool): Whether to display the figure. Default True.

    Returns:
        plt.Figure: The created figure object.

    Example:
        >>> from src.core import PowerModel
        >>> from src.core.plotting import plot_comprehensive_analysis
        >>> model = PowerModel(config)
        >>> data = model.generate_power_curve(numPoints=100)
        >>> params = {
        ...     'wingArea': model.wingArea,
        ...     'nominalGeneratorPower': model.nominalGeneratorPower,
        ...     'nominalTetherForce': model.nominalTetherForce,
        ...     'nominalWindSpeedForce': model.nominalWindSpeedForce,
        ...     'nominalWindSpeedPower': model.nominalWindSpeedPower,
        ...     'cutOutWindSpeed': model.cutOutWindSpeed,
        ...     'elevationAngleOut': model.elevationAngleOut,
        ...     'elevationAngleIn': model.elevationAngleIn,
        ... }
        >>> fig = plot_comprehensive_analysis(data, params)
    """
    windSpeed = data['windSpeed']
    
    # Create figure with subplots (4 rows, 2 columns)
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)
    
    # Subplot 1: Power curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(windSpeed, data['power']/1000, 'b-', linewidth=2.5, label='Net Cycle Power')
    ax1.plot(windSpeed, data['reelOutPower']/1000, 'g--', linewidth=1.5, alpha=0.7, label='Reel-Out Power')
    ax1.plot(windSpeed, data['reelInPower']/1000, 'r--', linewidth=1.5, alpha=0.7, label='Reel-In Power')
    ax1.fill_between(windSpeed, 0, data['power']/1000, alpha=0.2, color='blue')
    
    if model_params:
        if 'nominalGeneratorPower' in model_params:
            ax1.axhline(y=model_params['nominalGeneratorPower']/1000, color='purple', 
                       linestyle=':', alpha=0.5, label='Nominal Power')
        if 'nominalWindSpeedForce' in model_params:
            ax1.axvline(x=model_params['nominalWindSpeedForce'], color='orange', 
                       linestyle=':', alpha=0.5)
        if 'nominalWindSpeedPower' in model_params and 'cutOutWindSpeed' in model_params:
            if model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']:
                ax1.axvline(x=model_params['nominalWindSpeedPower'], color='red', 
                           linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax1.set_ylabel('Power (kW)', fontsize=11)
    ax1.set_title('Power Output', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cycle times
    ax2 = fig.add_subplot(gs[0, 1])
    cycleTime = data['reelOutTime'] + data['reelInTime']
    ax2.plot(windSpeed, cycleTime, 'k-', linewidth=2.5, label='Total Cycle Time')
    ax2.plot(windSpeed, data['reelOutTime'], 'g--', linewidth=1.5, alpha=0.7, label='Reel-Out Time')
    ax2.plot(windSpeed, data['reelInTime'], 'r--', linewidth=1.5, alpha=0.7, label='Reel-In Time')
    
    if model_params:
        if 'nominalWindSpeedForce' in model_params:
            ax2.axvline(x=model_params['nominalWindSpeedForce'], color='orange', 
                       linestyle=':', alpha=0.5)
        if 'nominalWindSpeedPower' in model_params and 'cutOutWindSpeed' in model_params:
            if model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']:
                ax2.axvline(x=model_params['nominalWindSpeedPower'], color='red', 
                           linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax2.set_ylabel('Time (s)', fontsize=11)
    ax2.set_title('Cycle Times', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Tether forces
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(windSpeed, data['tetherForceOut']/1000, 'g-', linewidth=2, label='Reel-Out Force')
    ax3.plot(windSpeed, data['tetherForceIn']/1000, 'r-', linewidth=2, label='Reel-In Force')
    
    if model_params:
        if 'nominalTetherForce' in model_params:
            ax3.axhline(y=model_params['nominalTetherForce']/1000, color='purple', 
                       linestyle=':', alpha=0.5, label='Nominal Force')
        if 'nominalWindSpeedForce' in model_params:
            ax3.axvline(x=model_params['nominalWindSpeedForce'], color='orange', 
                       linestyle=':', alpha=0.5)
        if 'nominalWindSpeedPower' in model_params and 'cutOutWindSpeed' in model_params:
            if model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']:
                ax3.axvline(x=model_params['nominalWindSpeedPower'], color='red', 
                           linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax3.set_ylabel('Tether Force (kN)', fontsize=11)
    ax3.set_title('Tether Forces', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Reel speeds and elevation angles
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    
    l1 = ax4.plot(windSpeed, data['reelOutSpeed'], 'g-', linewidth=2, label='Reel-Out Speed')
    l2 = ax4.plot(windSpeed, data['reelInSpeed'], 'r-', linewidth=2, label='Reel-In Speed')
    
    if model_params:
        if 'nominalWindSpeedForce' in model_params:
            ax4.axvline(x=model_params['nominalWindSpeedForce'], color='orange', 
                       linestyle=':', alpha=0.5)
        if 'nominalWindSpeedPower' in model_params and 'cutOutWindSpeed' in model_params:
            if model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']:
                ax4.axvline(x=model_params['nominalWindSpeedPower'], color='red', 
                           linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax4.set_ylabel('Reel Speed (m/s)', fontsize=11, color='black')
    ax4.tick_params(axis='y', labelcolor='black')
    ax4.grid(True, alpha=0.3)
    
    # Plot elevation angles on right y-axis if provided
    if model_params and 'elevationAngleOut' in model_params and 'elevationAngleIn' in model_params:
        elevOut_deg = np.rad2deg(model_params['elevationAngleOut'])
        elevIn_deg = np.rad2deg(model_params['elevationAngleIn'])
        l3 = ax4_twin.plot(windSpeed, np.ones_like(windSpeed)*elevOut_deg, 'g--', 
                          linewidth=1.5, alpha=0.5, label=f'Elev Out ({elevOut_deg:.1f}°)')
        l4 = ax4_twin.plot(windSpeed, np.ones_like(windSpeed)*elevIn_deg, 'r--', 
                          linewidth=1.5, alpha=0.5, label=f'Elev In ({elevIn_deg:.1f}°)')
        ax4_twin.set_ylabel('Elevation Angle (°)', fontsize=11, color='gray')
        ax4_twin.tick_params(axis='y', labelcolor='gray')
        
        # Combine legends
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left', fontsize=9)
    else:
        ax4.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='upper left', fontsize=9)
    
    ax4.set_title('Reel Speeds & Elevation Angles', fontsize=12, fontweight='bold')
    
    # Subplot 5: Energy per cycle
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Calculate energies (power * time)
    energyOut = data['reelOutPower'] * data['reelOutTime'] / 3600000  # Convert to kWh
    energyIn = data['reelInPower'] * data['reelInTime'] / 3600000      # Convert to kWh
    cycleEnergy = data['power'] * (data['reelOutTime'] + data['reelInTime']) / 3600000  # kWh
    
    ax5.plot(windSpeed, energyOut, 'g-', linewidth=2.5, label='Reel-Out Energy')
    ax5.plot(windSpeed, energyIn, 'r-', linewidth=2.5, label='Reel-In Energy')
    ax5.plot(windSpeed, cycleEnergy, 'b-', linewidth=2.5, label='Net Cycle Energy')
    ax5.fill_between(windSpeed, 0, cycleEnergy, alpha=0.2, color='blue')
    
    if model_params:
        if 'nominalWindSpeedForce' in model_params:
            ax5.axvline(x=model_params['nominalWindSpeedForce'], color='orange', 
                       linestyle=':', alpha=0.5)
        if 'nominalWindSpeedPower' in model_params and 'cutOutWindSpeed' in model_params:
            if model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']:
                ax5.axvline(x=model_params['nominalWindSpeedPower'], color='red', 
                           linestyle=':', alpha=0.5)
    
    ax5.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax5.set_ylabel('Energy per Cycle (kWh)', fontsize=11)
    ax5.set_title('Energy per Cycle', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Reeling factors (gamma) - spans bottom
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(windSpeed, data['gammaOut'], 'g-', linewidth=2.5, label='Reel-Out Factor (γ_out)')
    ax6.plot(windSpeed, data['gammaIn'], 'r-', linewidth=2.5, label='Reel-In Factor (γ_in)')
    
    if model_params:
        if 'nominalWindSpeedForce' in model_params:
            ax6.axvline(x=model_params['nominalWindSpeedForce'], color='orange', 
                       linestyle=':', alpha=0.5,
                       label=f"Force Limit ({model_params['nominalWindSpeedForce']:.1f} m/s)")
        if 'nominalWindSpeedPower' in model_params and 'cutOutWindSpeed' in model_params:
            if model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']:
                ax6.axvline(x=model_params['nominalWindSpeedPower'], color='red', 
                           linestyle=':', alpha=0.5,
                           label=f"Power Limit ({model_params['nominalWindSpeedPower']:.1f} m/s)")
    
    ax6.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax6.set_ylabel('Reeling Factor (-)', fontsize=11)
    ax6.set_title('Dimensionless Reeling Factors', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Overall title
    title = 'AWE Power Curve Analysis (Luchsinger Model)'
    if model_params and 'wingArea' in model_params and 'nominalGeneratorPower' in model_params:
        title += f"\n{model_params['wingArea']} m² wing"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def extract_model_params(model) -> Dict:
    """Extract model parameters for plotting.

    Args:
        model: PowerModel instance.

    Returns:
        Dict: Dictionary of model parameters.
    """
    return {
        'wingArea': model.wingArea,
        'nominalGeneratorPower': model.nominalGeneratorPower,
        'nominalTetherForce': model.nominalTetherForce,
        'nominalWindSpeedForce': model.nominalWindSpeedForce,
        'nominalWindSpeedPower': model.nominalWindSpeedPower,
        'cutOutWindSpeed': model.cutOutWindSpeed,
        'elevationAngleOut': model.elevationAngleOut,
        'elevationAngleIn': model.elevationAngleIn,
    }
