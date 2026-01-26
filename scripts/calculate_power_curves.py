#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate power curves for an Airborne Wind Energy system.

This script loads configuration from YAML files and outputs power curves.

Usage:
    python calculate_power_curves.py
"""

import sys
from pathlib import Path

# Add the src directory to the path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inertiafree_qsm import calculate_power_curves


# Define file paths
PROJECT_ROOT = Path(__file__).parent.parent

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "soft_kite_pumping_ground_gen_system.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings_config.yml"
OUTPUT_PATH = PROJECT_ROOT / "results" / "soft_kite_pumping_ground_gen_power_curves.yml"


if __name__ == "__main__":
    calculate_power_curves(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        output_path=OUTPUT_PATH,
    )

