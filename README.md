# InertiaFree-QSM


**# DISCLAIMER: STILL IN DEVELOPMENT**

A simple, configurable power model for pumping kite airborne wind energy (AWE) systems based on the Luchsinger model.

## Overview

This repository provides a standalone power model for AWE pumping kite systems. It calculates power curves from cut-in to cut-out wind speed, accounting for:

- Aerodynamic forces on the kite
- Ground station generator and storage efficiency losses
- Optimal reeling velocity control
- Three operating regions (force-limited, power-limited)
- Discretized reel in and reel out phase
- Kite mass implementation

## Purpose

This repository is designed to be:

1. **Tested/Used locally** using the scripts in the `scripts/` folder with configuration from `data/`
2. **Integrated into a larger toolchain** where configuration values are set externally and passed to the `PowerModel` class

When used as part of another toolchain, the wrapper code can create a configuration dictionary programmatically, bypassing the YAML files entirely.

## Features

- **Simple API**: Calculate power at any wind speed or generate full power curves
- **Configurable**: All parameters via YAML or Python dictionaries
- **Reeling-factor optimization**: Includes optimal reeling factor calculation
- **Standalone**: No external dependencies beyond NumPy, SciPy, and PyYAML

## Installation

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- PyYAML >= 5.4.0
- Matplotlib >= 3.4.0

### Install dependencies

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda install numpy scipy pyyaml matplotlib
```

## Quick Start

### Using the Script

```bash
# Generate and plot power curve with an example 100kW configuration
python scripts/calculate_power_curves.py
```

### Using as a Library

```python
from src.power_luchsinger import PowerModel

# Load from YAML configuration
model = PowerModel.from_yaml('data/100kW_system_example.yml')

# Calculate power at a specific wind speed
power = model.calculate_power(windSpeed=10.0)
print(f"Power at 10 m/s: {power:.0f} W")

# Generate complete power curve
power_curve = model.generate_power_curve(numPoints=100)
print(f"Wind speeds: {power_curve['windSpeed']}")
print(f"Power output: {power_curve['power']}")
```

### Using with a Custom Configuration Dictionary

```python
from src.power_luchsinger import PowerModel

config = {
    'kite': {
        'wingArea': 50.0,
        'liftCoefficientOut': 0.8,
        'dragCoefficientOut': 0.12,
        'dragCoefficientIn': 0.06,
    },
    'tether': {
        'maxLength': 500.0,
        'minLength': 250.0,
    },
    'atmosphere': {
        'airDensity': 1.225,  # kg/m³
    },
    'groundStation': {
        'nominalTetherForce': 4000.0,      # N
        'nominalGeneratorPower': 60000.0,  # W
        'reelOutSpeedLimit': 10.0,         # m/s
        'reelInSpeedLimit': 30.0,          # m/s
    },
    'operational': {
        'cutInWindSpeed': 4.0,    # m/s
        'cutOutWindSpeed': 25.0,  # m/s
        'elevationAngleOut': 25.0,  # degrees
        'elevationAngleIn': 45.0,   # degrees
    },
}

model = PowerModel(config)
power_curve = model.generate_power_curve()
```

## Configuration Reference

### Required Configuration Sections

| Section | Description |
|---------|-------------|
| `kite` | Aerodynamic parameters |
| `tether` | Tether properties |
| `atmosphere` | Atmospheric conditions |
| `groundStation` | Ground station parameters |
| `operational` | Operating envelope |

### Kite Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `wingArea` | Projected wing area | m² |
| `liftCoefficientOut` | Lift coefficient during reel-out | - |
| `dragCoefficientOut` | Drag coefficient during reel-out | - |
| `dragCoefficientIn` | Drag coefficient during reel-in | - |

### Tether Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `maxLength` | Maximum tether length | m |
| `minLength` | Minimum tether length | m |

### Atmosphere Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `airDensity` | Air density | kg/m³ |

### Ground Station Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `nominalTetherForce` | Maximum tether force | N |
| `nominalGeneratorPower` | Maximum generator power | W |
| `reelOutSpeedLimit` | Maximum reel-out speed | m/s |
| `reelInSpeedLimit` | Maximum reel-in speed | m/s |

### Operational Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `cutInWindSpeed` | Cut-in wind speed | m/s |
| `cutOutWindSpeed` | Cut-out wind speed | m/s |
| `elevationAngleOut` | Elevation angle during reel-out | degrees |
| `elevationAngleIn` | Elevation angle during reel-in | degrees |

## Physical Model

### Luchsinger Pumping Cycle Model

The model implements a ground-based pumping kite power system with two phases:

1. **Reel-out phase**: Kite flies crosswind generating high tether force, pulling out tether and driving the generator
2. **Reel-in phase**: Kite is depowered (reduced angle of attack), tether is reeled in using the generator

### Operating Regions

| Region | Description |
|--------|-------------|
| **Region 1** | Below force limit - optimal reeling velocities |
| **Region 2** | Force-limited - tether force at maximum |
| **Region 3** | Power-limited - generator power at maximum |


## Project Structure

```
├── src/
│   └── inertiafree_qsm/
│       ├── __init__.py
│       ├── power_model.py      # Main PowerModel class
│       ├── calculations.py     # Pure calculation functions
│       ├── plotting.py         # Plotting utilities for toolchains
│       └── default_config.yml  # Default configuration parameters
├── scripts/
│   ├── __init__.py
│   └── calculate_power_curves.py  # Main script for generating plots
├── data/
│   └── 100kW_system_example.yml    # Example configuration
├── results/                    # Generated plots and outputs
├── requirements.txt
├── README.md                   # This file
└── LICENSE
```

## References

1. R.H. Luchsinger: "Pumping cycle kite power". In *Airborne Wind Energy*, Springer, 2013. https://doi.org/10.1007/978-3-642-39965-7_3

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
