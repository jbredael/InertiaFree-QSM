"""Pure calculation functions for Luchsinger power model."""

import numpy as np

def calculate_force_factor_out(liftCoefficient: float,
                               dragCoefficient: float) -> float:
    """Calculate the dimensionless force factor for reel-out phase.
    
    This factor characterizes the kite's power generation capability
    during the reel-out phase.
    
    Args:
        liftCoefficient (float): Lift coefficient during reel-out.
        dragCoefficient (float): Drag coefficient during reel-out.
            
    Returns:
        float: Force factor f_out = C_L³ / C_D².
    """
    if dragCoefficient <= 0:
        raise ValueError("Drag coefficient must be positive")
    
    return (liftCoefficient**3) / (dragCoefficient**2)


def calculate_force_factor_in(dragCoefficient: float) -> float:
    """Calculate the dimensionless force factor for reel-in phase.
    
    During reel-in, the kite is depowered and the force is proportional
    to drag only.
    
    Args:
        dragCoefficient (float): Drag coefficient during reel-in.
            
    Returns:
        float: Force factor f_in = C_D.
    """
    return dragCoefficient


def calculate_tether_force_out(airDensity: float,
                           windSpeed: float,
                           wingArea: float,
                           gammaOut: float,
                           elevationAngle: float,
                           forceFactor: float) -> float:
    """Calculate tether force during reel-out phase.
    
    Uses the Luchsinger model formulation for tether force.
    
    Args:
        airDensity (float): Air density in kg/m³.
        windSpeed (float): Wind speed in m/s.
        wingArea (float): Projected wing area in m².
        gammaOut (float): Dimensionless reel-out velocity (v_out / v_wind).
        elevationAngle (float): Tether elevation angle in radians.
        forceFactor (float): Kite force factor (C_L³/C_D² for reel-out).
        
    Returns:
        float: Tether force in N.
    """
    effectiveWindFactor = (np.cos(elevationAngle) - gammaOut)**2
    tetherForce = 0.5 * airDensity * windSpeed**2 * wingArea * effectiveWindFactor * forceFactor
    
    return max(0.0, tetherForce)


def calculate_tether_force_in(airDensity: float,
                              windSpeed: float,
                              wingArea: float,
                              gammaIn: float,
                              elevationAngle: float,
                              forceFactor: float) -> float:
    """Calculate tether force during reel-in phase.
    
    Uses the Luchsinger model formulation for tether force during retraction.
    
    Args:
        airDensity (float): Air density in kg/m³.
        windSpeed (float): Wind speed in m/s.
        wingArea (float): Projected wing area in m².
        gammaIn (float): Dimensionless reel-in velocity (v_in / v_wind).
        elevationAngle (float): Tether elevation angle in radians.
        forceFactor (float): Kite force factor (C_D for reel-in).
        
    Returns:
        float: Tether force in N.
    """
    effectiveWindFactor = 1 + 2 * np.cos(elevationAngle) * gammaIn + gammaIn**2
    tetherForce = 0.5 * airDensity * windSpeed**2 * wingArea * effectiveWindFactor * forceFactor
    
    return max(0.0, tetherForce)

def calculate_cycle_power(powerOut: float,
                          powerIn: float,
                          timeOut: float,
                          timeIn: float,
                          storageEfficiency: float = 0.95) -> float:
    """Calculate average cycle power for a complete pumping cycle.
    
    The cycle power accounts for energy generated during reel-out,
    energy consumed during reel-in, and losses from braking and storage.
    
    Args:
        powerOut (float): Electrical power during reel-out in W.
        powerIn (float): Electrical power during reel-in in W.
        timeOut (float): Reel-out phase duration in s.
        timeIn (float): Reel-in phase duration in s.
        storageEfficiency (float): Energy storage efficiency.
        
    Returns:
        float: Average cycle power in W.
    """
    cycleTime = timeOut + timeIn
    
    if cycleTime <= 0:
        return 0.0
    
    # Energy balance
    energyOut = powerOut * timeOut
    energyIn = (powerIn * timeIn) / storageEfficiency
    
    cyclePower = (energyOut - energyIn) / cycleTime
    
    return max(0.0, cyclePower)


def calculate_cycle_results(tetherForceOut: float,
                           tetherForceIn: float,
                           reelOutSpeed: float,
                           reelInSpeed: float,
                           reelingLength: float,
                           gammaOut: float,
                           gammaIn: float,
                           generatorEfficiency: float,
                           storageEfficiency: float) -> dict:
    """Calculate complete cycle results including power, time, and forces.
    
    This function encapsulates all power and time calculations for a
    complete pumping cycle.
    
    Args:
        tetherForceOut (float): Tether force during reel-out in N.
        tetherForceIn (float): Tether force during reel-in in N.
        reelOutSpeed (float): Reel-out speed in m/s.
        reelInSpeed (float): Reel-in speed in m/s.
        reelingLength (float): Total reeling length in m.
        gammaOut (float): Dimensionless reel-out velocity factor.
        gammaIn (float): Dimensionless reel-in velocity factor.
        generatorEfficiency (float): Generator efficiency (0-1).
        storageEfficiency (float): Storage efficiency (0-1).
        
    Returns:
        dict: Dictionary with complete cycle results.
    """
    # Calculate mechanical and electrical power
    mechPowerOut = tetherForceOut * reelOutSpeed
    elecPowerOut = mechPowerOut * generatorEfficiency
    
    mechPowerIn = tetherForceIn * reelInSpeed
    elecPowerIn = mechPowerIn / generatorEfficiency
    
    # Calculate phase durations
    timeOut = reelingLength / reelOutSpeed if reelOutSpeed > 0 else float('inf')
    timeIn = reelingLength / reelInSpeed if reelInSpeed > 0 else float('inf')
    
    # Calculate cycle power
    cyclePower = calculate_cycle_power(
        elecPowerOut, elecPowerIn, timeOut, timeIn, storageEfficiency
    )
    
    return {
        'cyclePower': cyclePower,
        'reelOutPower': elecPowerOut,
        'reelInPower': elecPowerIn,
        'reelOutTime': timeOut,
        'reelInTime': timeIn,
        'tetherForceOut': tetherForceOut,
        'tetherForceIn': tetherForceIn,
        'reelOutSpeed': reelOutSpeed,
        'reelInSpeed': reelInSpeed,
        'gammaOut': gammaOut,
        'gammaIn': gammaIn,
    }
