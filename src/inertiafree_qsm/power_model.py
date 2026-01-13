"""Core power estimation model for airborne wind energy systems based on the Luchsinger model [1].

This module provides the main PowerModel class for calculating power output
of pumping kite systems. The model is planet-agnostic and configurable via
dictionaries or YAML files.

References:
    [1] R.H. Luchsinger: "Pumping cycle kite power". Springer, 2013.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import copy
import logging

import numpy as np
import yaml
from scipy import optimize as op

from src.inertiafree_qsm.calculations import (
    calculate_force_factor_out,
    calculate_force_factor_in,
    calculate_tether_force_out,
    calculate_tether_force_in,
    calculate_cycle_results
)

# Configure logger
logger = logging.getLogger(__name__)


class PowerModel:
    """Calculate power output for airborne wind energy systems.

    This model is configurable via YAML files or
    dictionaries. All physical parameters must be provided through
    configuration.

    The model implements the Luchsinger pumping cycle model [1].

    Attributes:
        config (Dict): Complete configuration dictionary.
        wingArea (float): Projected wing area in m².
        airDensity (float): Air density in kg/m³.
        nominalTetherForce (float): Maximum tether force in N.
        nominalGeneratorPower (float): Maximum generator power in W.
    """

    # Path to default configuration file
    _DEFAULT_CONFIG_PATH = Path(__file__).parent / 'default_config.yml'

    # Required configuration sections
    REQUIRED_SECTIONS = ['kite', 'tether', 'atmosphere', 'groundStation', 'operational']

    # Required keys within each section
    REQUIRED_KEYS = {
        'kite': ['wingArea', 'liftCoefficientOut', 'dragCoefficientOut',
                 'dragCoefficientIn'],
        'tether': ['maxLength', 'minLength'],
        'atmosphere': ['airDensity'],
        'groundStation': ['nominalTetherForce', 'nominalGeneratorPower',
                          'reelOutSpeedLimit', 'reelInSpeedLimit'],
        'operational': ['cutInWindSpeed', 'cutOutWindSpeed',
                        'elevationAngleOut', 'elevationAngleIn'],
    }

    @classmethod
    def _load_default_config(cls) -> Dict[str, Any]:
        """Load default configuration from YAML file.

        Returns:
            Dict: Default configuration dictionary.

        Raises:
            FileNotFoundError: If default config file doesn't exist.
        """
        if not cls._DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Default configuration file not found: {cls._DEFAULT_CONFIG_PATH}"
            )

        with open(cls._DEFAULT_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)

    @classmethod
    def _merge_configs(cls, user_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with defaults.

        Args:
            user_config (Dict): User-provided configuration.
            default_config (Dict): Default configuration.

        Returns:
            Dict: Merged configuration with user values taking precedence.
        """
        merged = copy.deepcopy(default_config)
        defaults_used = []

        # Track which sections and keys are missing or None/empty from user config
        for section, default_values in default_config.items():
            if section not in user_config or user_config[section] is None:
                # Entire section missing or None
                if isinstance(default_values, dict):
                    for key, value in default_values.items():
                        defaults_used.append(f"{section}.{key} = {value}")
                else:
                    defaults_used.append(f"{section} = {default_values}")
            elif isinstance(default_values, dict) and isinstance(user_config[section], dict):
                # Section exists, check individual keys
                for key, value in default_values.items():
                    if key not in user_config[section] or user_config[section][key] is None:
                        defaults_used.append(f"{section}.{key} = {value}")

        # Merge user config into defaults
        for section, values in user_config.items():
            if section not in merged:
                merged[section] = {}

            if isinstance(values, dict):
                for key, value in values.items():
                    # Only use user value if it's not None
                    if value is not None:
                        merged[section][key] = value
            elif values is not None:
                merged[section] = values

        # Print defaults used
        if defaults_used:
            print("\n=== Default values used for missing configuration parameters ===")
            for default_info in defaults_used:
                print(f"  {default_info}")
            print("================================================================\n")

        return merged

    def __init__(self, config: Dict[str, Any]):
        """Initialize power model with configuration parameters.

        Args:
            config (Dict): Dictionary containing model parameters.
                Missing values will be filled from default_config.yaml.

        Raises:
            ValueError: If required configuration keys are missing.
            ValueError: If parameter values are physically invalid.
        """
        # Load defaults and merge with user config
        default_config = self._load_default_config()
        self.config = self._merge_configs(config, default_config)

        # Extract parameters first
        self._extract_parameters()

        # Then validate physical constraints
        self._validate_physical_constraints()

        # Finally compute derived parameters
        self._compute_derived_parameters()

    def _validate_physical_constraints(self) -> None:
        """Validate that parameter values are physically reasonable.

        Raises:
            ValueError: If values are physically invalid.
        """
        if self.wingArea <= 0:
            raise ValueError("Wing area must be positive")
        if self.liftCoefficientOut <= 0:
            raise ValueError("Lift coefficient must be positive")
        if self.dragCoefficientKiteOut <= 0:
            raise ValueError("Drag coefficient must be positive")

        if self.tetherMaxLength <= self.tetherMinLength:
            raise ValueError("Max tether length must exceed min length")

        if self.airDensity <= 0:
            raise ValueError("Air density must be positive")

        if self.nominalTetherForce <= 0:
            raise ValueError("Nominal tether force must be positive")
        if self.nominalGeneratorPower <= 0:
            raise ValueError("Nominal generator power must be positive")

        if self.cutInWindSpeed >= self.cutOutWindSpeed:
            raise ValueError("Cut-in wind speed must be less than cut-out")

    def _extract_parameters(self) -> None:
        """Extract parameters from config to instance variables."""
        # Kite parameters
        kite = self.config['kite']
        self.wingArea = kite['wingArea']
        self.liftCoefficientOut = kite['liftCoefficientOut']
        self.dragCoefficientKiteOut = kite['dragCoefficientOut']
        self.dragCoefficientKiteIn = kite['dragCoefficientIn']

        # Tether parameters
        tether = self.config['tether']
        self.tetherMaxLength = tether['maxLength']
        self.tetherMinLength = tether['minLength']

        # Atmosphere parameters
        atm = self.config['atmosphere']
        self.airDensity = atm['airDensity']

        # Ground station parameters
        gs = self.config['groundStation']
        self.nominalTetherForce = gs['nominalTetherForce']
        self.nominalGeneratorPower = gs['nominalGeneratorPower']
        self.reelOutSpeedLimit = gs['reelOutSpeedLimit']
        self.reelInSpeedLimit = gs['reelInSpeedLimit']

        # Efficiency parameters (only two: generator and storage)
        self.generatorEfficiency = gs.get('generatorEfficiency')
        self.storageEfficiency = gs.get('storageEfficiency')

        # Operational parameters
        operational = self.config['operational']
        self.cutInWindSpeed = operational['cutInWindSpeed']
        self.cutOutWindSpeed = operational['cutOutWindSpeed']
        self.elevationAngleOut = np.radians(operational['elevationAngleOut'])
        self.elevationAngleIn = np.radians(operational['elevationAngleIn'])

    def _compute_derived_parameters(self) -> None:
        """Compute derived parameters from base configuration."""
        # Operational tether length
        self.operationalLength = (
            (self.tetherMaxLength - self.tetherMinLength) / 2 +
            self.tetherMinLength
        )
        self.reelingLength = self.tetherMaxLength - self.tetherMinLength

        # Total drag coefficients
        self.dragCoefficientOut = self.dragCoefficientKiteOut
        self.dragCoefficientIn = self.dragCoefficientKiteIn

        # Force factors
        self.forceFactorOut = calculate_force_factor_out(
            self.liftCoefficientOut, self.dragCoefficientOut
        )
        self.forceFactorIn = calculate_force_factor_in(self.dragCoefficientIn)

        # Compute nominal wind speeds for force and power limits
        self._compute_nominal_wind_speeds()

    def _compute_nominal_wind_speeds(self) -> None:
        """Compute wind speeds at which force and power limits are reached."""
        windSpeeds = np.arange(self.cutInWindSpeed, self.cutOutWindSpeed, 0.1)

        self.nominalWindSpeedForce = self.cutOutWindSpeed
        self.nominalGammaOutForce = 0.33

        self.nominalWindSpeedPower = self.cutOutWindSpeed
        self.nominalGammaOutPower = 0.33

        # Find force limit
        for vw in windSpeeds:
            gammaOutMax = self.reelOutSpeedLimit / vw
            gammaInMax = self.reelInSpeedLimit / vw

            gammaOut, gammaIn = self._optimize_gamma_out_in_region1(
                self.elevationAngleOut, self.elevationAngleIn,
                self.forceFactorOut, self.forceFactorIn,
                gammaOutMax, gammaInMax
            )

            tetherForce = calculate_tether_force_out(
                self.airDensity, vw, self.wingArea,
                gammaOut, self.elevationAngleOut, self.forceFactorOut
            )

            if tetherForce >= self.nominalTetherForce:
                self.nominalWindSpeedForce = vw
                self.nominalGammaOutForce = gammaOut
                break

        # Find power limit (only for winds above force limit)
        for vw in windSpeeds:
            if vw <= self.nominalWindSpeedForce:
                continue

            mu = vw / self.nominalWindSpeedForce
            gammaOut = (
                np.cos(self.elevationAngleOut) -
                (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu
            )
            vOut = vw * gammaOut

            # Simple power calculation
            mechPower = self.nominalTetherForce * vOut
            elecPower = mechPower * self.generatorEfficiency

            if elecPower >= self.nominalGeneratorPower:
                self.nominalWindSpeedPower = vw
                self.nominalGammaOutPower = gammaOut
                break

        # Compute nominal reel-out speed for power-limited region
        self.nominalReelOutSpeed = (
            self.nominalGeneratorPower / 
            (self.nominalTetherForce * self.generatorEfficiency)
        )


    def calculate_power(self,
                       windSpeed: float,
                       airDensity: float = None) -> Dict[str, float]:
        """Calculate power output for given wind speed.

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³. If None, uses
                atmosphere.airDensity from config.

        Returns:
            Dict[str, float]: Dictionary with keys:
                - 'cyclePower': Average cycle power (W)
                - 'reelOutPower': Reel-out power (W)
                - 'reelInPower': Reel-in power (W)
                - 'reelOutTime': Reel-out time (s)
                - 'reelInTime': Reel-in time (s)
                - 'tetherForceOut': Tether force during reel-out (N)
                - 'tetherForceIn': Tether force during reel-in (N)
                - 'reelOutSpeed': Reel-out speed (m/s)
                - 'reelInSpeed': Reel-in speed (m/s)
                - 'gammaOut': Reel-out factor (-)
                - 'gammaIn': Reel-in factor (-)
        """
        if airDensity is None:
            airDensity = self.airDensity

        if windSpeed < self.cutInWindSpeed or windSpeed > self.cutOutWindSpeed:
            return {
                'cyclePower': 0.0,
                'reelOutPower': 0.0,
                'reelInPower': 0.0,
                'reelOutTime': 0.0,
                'reelInTime': 0.0,
                'tetherForceOut': 0.0,
                'tetherForceIn': 0.0,
                'reelOutSpeed': 0.0,
                'reelInSpeed': 0.0,
                'gammaOut': 0.0,
                'gammaIn': 0.0,
            }

        if windSpeed < self.nominalWindSpeedForce:
            return self._calculate_power_region1(windSpeed, airDensity)
        elif windSpeed < self.nominalWindSpeedPower:
            return self._calculate_power_region2(windSpeed, airDensity)
        else:
            return self._calculate_power_region3(windSpeed, airDensity)

    def _calculate_power_region1(self,
                                  windSpeed: float,
                                  airDensity: float) -> Dict[str, float]:
        """Calculate power in Region 1 (below force limit).

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³.

        Returns:
            Dict with power and time details.
        """
        gammaOutMax = self.reelOutSpeedLimit / windSpeed
        gammaInMax = self.reelInSpeedLimit / windSpeed

        gammaOut, gammaIn = self._optimize_gamma_out_in_region1(
            self.elevationAngleOut, self.elevationAngleIn,
            self.forceFactorOut, self.forceFactorIn,
            gammaOutMax, gammaInMax
        )

        vOut = windSpeed * gammaOut
        vIn = windSpeed * gammaIn

        tetherForceOut = calculate_tether_force_out(
            airDensity, windSpeed, self.wingArea,
            gammaOut, self.elevationAngleOut, self.forceFactorOut
        )
        tetherForceIn = calculate_tether_force_in(
            airDensity, windSpeed, self.wingArea,
            gammaIn, self.elevationAngleIn, self.forceFactorIn
        )

        return calculate_cycle_results(
            tetherForceOut, tetherForceIn, vOut, vIn,
            self.reelingLength, gammaOut, gammaIn,
            self.generatorEfficiency, self.storageEfficiency
        )

    def _optimize_gamma_out_in_region1(self,
                                        elevationAngleOut: float,
                                        elevationAngleIn: float,
                                        forceFactorOut: float,
                                        forceFactorIn: float,
                                        gammaOutMax: float,
                                        gammaInMax: float) -> Tuple[float, float]:
        """Calculate optimal dimensionless reeling velocity factors.
        
        Optimizes the cycle power factor by finding the optimal reeling
        velocities for both reel-out and reel-in phases.
        
        Args:
            elevationAngleOut (float): Elevation angle during reel-out in rad.
            elevationAngleIn (float): Elevation angle during reel-in in rad.
            forceFactorOut (float): Force factor during reel-out.
            forceFactorIn (float): Force factor during reel-in.
            gammaOutMax (float): Maximum gamma_out (v_out_max / v_wind).
            gammaInMax (float): Maximum gamma_in (v_in_max / v_wind).
            
        Returns:
            Tuple[float, float]: (optimal gamma_out, optimal gamma_in).
        """
        from scipy import optimize as op
        
        def objective(x):
            gammaOut, gammaIn = x
            # Cycle power factor from Luchsinger model
            powerFactor = (
                (np.cos(elevationAngleOut) - gammaOut)**2 -
                (forceFactorIn / forceFactorOut) *
                (1 + 2 * np.cos(elevationAngleIn) * gammaIn + gammaIn**2)
            ) * ((gammaOut * gammaIn) / (gammaOut + gammaIn))
            return -powerFactor  # Minimize negative = maximize
        
        bounds = ((0.001, gammaOutMax), (0.001, gammaInMax))
        result = op.minimize(objective, (0.001, 0.001), bounds=bounds, method='SLSQP')
        
        return result['x'][0], result['x'][1]

    def _calculate_power_region2(self,
                                  windSpeed: float,
                                  airDensity: float) -> Dict[str, float]:
        """Calculate power in Region 2 (force-limited, below power limit).

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³.

        Returns:
            Dict with power and time details.
        """
        mu = windSpeed / self.nominalWindSpeedForce
        gammaInMax = self.reelInSpeedLimit / windSpeed

        gammaOut = (
            np.cos(self.elevationAngleOut) -
            (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu
        )
        vOut = windSpeed * gammaOut
        tetherForceOut = self.nominalTetherForce

        gammaIn = self._optimize_gamma_in_region2(mu, gammaInMax)
        vIn = windSpeed * gammaIn

        tetherForceIn = calculate_tether_force_in(
            airDensity, windSpeed, self.wingArea,
            gammaIn, self.elevationAngleIn, self.forceFactorIn
        )

        return calculate_cycle_results(
            tetherForceOut, tetherForceIn, vOut, vIn,
            self.reelingLength, gammaOut, gammaIn,
            self.generatorEfficiency, self.storageEfficiency
        )

    def _optimize_gamma_in_region2(self, mu: float, gammaInMax: float) -> float:
        """Optimize gamma_in for Region 2 operation.

        Args:
            mu (float): Wind speed ratio to nominal force wind speed.
            gammaInMax (float): Maximum gamma_in.

        Returns:
            float: Optimal gamma_in.
        """
        def objective(x):
            gammaIn = x[0]
            gammaOutEff = (
                mu * np.cos(self.elevationAngleOut) -
                np.cos(self.elevationAngleOut) +
                self.nominalGammaOutForce
            )

            powerFactor = (
                (1 / mu**2) * (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce)**2 -
                (self.forceFactorIn / self.forceFactorOut) *
                (1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2)
            ) * (
                (gammaIn * gammaOutEff) /
                (mu * gammaIn + gammaOutEff)
            )
            return -powerFactor

        result = op.minimize(
            objective, [0.001],
            bounds=[(0.001, gammaInMax)],
            method='SLSQP'
        )

        return result['x'][0]

    def _calculate_power_region3(self,
                                  windSpeed: float,
                                  airDensity: float) -> Dict[str, float]:
        """Calculate power in Region 3 (power-limited).

        Args:
            windSpeed (float): Wind speed in m/s.
            airDensity (float): Air density in kg/m³.

        Returns:
            Dict with power and time details.
        """
        mu = windSpeed / self.nominalWindSpeedPower
        gammaInMax = self.reelInSpeedLimit / windSpeed

        vOut = self.nominalReelOutSpeed
        gammaOut = vOut / windSpeed
        elecPowerOut = self.nominalGeneratorPower
        tetherForceOut = self.nominalTetherForce

        gammaIn = self._optimize_gamma_in_region3(mu, gammaInMax)
        vIn = windSpeed * gammaIn

        tetherForceIn = calculate_tether_force_in(
            airDensity, windSpeed, self.wingArea,
            gammaIn, self.elevationAngleIn, self.forceFactorIn
        )

        return calculate_cycle_results(
            tetherForceOut, tetherForceIn, vOut, vIn,
            self.reelingLength, gammaOut, gammaIn,
            self.generatorEfficiency, self.storageEfficiency
        )

    def _optimize_gamma_in_region3(self, mu: float, gammaInMax: float) -> float:
        """Optimize gamma_in for Region 3 operation.

        Args:
            mu (float): Wind speed ratio to nominal power wind speed.
            gammaInMax (float): Maximum gamma_in.

        Returns:
            float: Optimal gamma_in.
        """
        def objective(x):
            gammaIn = x[0]

            powerFactor = (
                (1 / mu**2) *
                (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce)**2 -
                (self.forceFactorIn / self.forceFactorOut) *
                (1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2)
            ) * (
                (self.nominalGammaOutPower * gammaIn) /
                (self.nominalGammaOutPower + mu * gammaIn)
            )
            return -powerFactor

        result = op.minimize(
            objective, [0.001],
            bounds=[(0.001, gammaInMax)],
            method='SLSQP'
        )

        return result['x'][0]

    def generate_power_curve(self,
                            numPoints: int = 100) -> Dict[str, np.ndarray]:
        """Generate power curve from cut-in to cut-out wind speed.

        Args:
            numPoints (int): Number of wind speed points. Defaults to 100.

        Returns:
            Dict[str, np.ndarray]: Dictionary with keys:
                - 'windSpeed': Wind speed array in m/s
                - 'power': Cycle power output array in W
                - 'reelOutPower': Reel-out power array in W
                - 'reelInPower': Reel-in power array in W
                - 'reelOutTime': Reel-out time array in s
                - 'reelInTime': Reel-in time array in s
                - 'tetherForceOut': Tether force during reel-out array in N
                - 'tetherForceIn': Tether force during reel-in array in N
                - 'reelOutSpeed': Reel-out speed array in m/s
                - 'reelInSpeed': Reel-in speed array in m/s
                - 'gammaOut': Reel-out factor array (-)
                - 'gammaIn': Reel-in factor array (-)
        """
        windSpeeds = np.linspace(self.cutInWindSpeed, self.cutOutWindSpeed, numPoints)
        
        results = [self.calculate_power(ws) for ws in windSpeeds]

        return {
            'windSpeed': windSpeeds,
            'power': np.array([r['cyclePower'] for r in results]),
            'reelOutPower': np.array([r['reelOutPower'] for r in results]),
            'reelInPower': np.array([r['reelInPower'] for r in results]),
            'reelOutTime': np.array([r['reelOutTime'] for r in results]),
            'reelInTime': np.array([r['reelInTime'] for r in results]),
            'tetherForceOut': np.array([r['tetherForceOut'] for r in results]),
            'tetherForceIn': np.array([r['tetherForceIn'] for r in results]),
            'reelOutSpeed': np.array([r['reelOutSpeed'] for r in results]),
            'reelInSpeed': np.array([r['reelInSpeed'] for r in results]),
            'gammaOut': np.array([r['gammaOut'] for r in results]),
            'gammaIn': np.array([r['gammaIn'] for r in results]),
        }

    @classmethod
    def from_yaml(cls, yamlPath: Path) -> 'PowerModel':
        """Load configuration from YAML file and create model instance.

        Args:
            yamlPath (Path): Path to YAML configuration file.

        Returns:
            PowerModel: Initialized power model instance.

        Raises:
            FileNotFoundError: If YAML file doesn't exist.
            ValueError: If YAML contains invalid configuration.
        """
        yamlPath = Path(yamlPath)

        if not yamlPath.exists():
            raise FileNotFoundError(f"Configuration file not found: {yamlPath}")

        with open(yamlPath, 'r') as f:
            config = yaml.safe_load(f)

        return cls(config)

