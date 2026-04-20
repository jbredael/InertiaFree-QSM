# InertiaFree-QSM

InertiaFree-QSM is a quasi-steady modeling workflow for evaluating kite power system cycle performance and power curves.
The modeling approach follows work by van der Vlugt et al. [1] and the code is based on the work of Schelbergen [2].

**Disclaimer:** This repository is still in development.

## Project structure

```text
InertiaFree-QSM/
├── data/         # Input configuration and wind resource files (YAML)
├── results/      # Generated results and plots
├── scripts/      # Entry-point scripts for calculations and plotting
└── src/          # Core package code (inertiafree_qsm)
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jbredael/InertiaFree-QSM.git
    cd InertiaFree-QSM
    ```

2. Create and activate a virtual environment:

    Linux / macOS:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    Windows (PowerShell):
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    ```

3. Install the package:

    For users (run the example scripts):
    ```bash
    pip install .
    ```

    For developers (editable install with dev tools):
    ```bash
    pip install -e .[dev]
    ```

4. To deactivate the virtual environment when you are done:
    ```bash
    deactivate
    ```

## Inputs

All input files are YAML files. Three input files are required:

| File | Description |
|------|-------------|
| **System configuration** | Kite and tether system properties (e.g. `kitepower V3_20.yml`). This is an [awesIO](https://github.com/awegroup/awesIO) standard file. |
| **Wind resource** | Wind profile and cluster data (e.g. `wind_resource.yml`). This is an [awesIO](https://github.com/awegroup/awesIO) standard file. |
| **Simulation settings** | Solver, optimizer, and cycle parameters (e.g. `simulation_settings.yml`). |

Example `simulation_settings.yml`:

```yaml
# Simulation settings for the inertia-free quasi-steady model
# All angles are in degrees

general:
  method: 'direct_simulation'

aerodynamics:
  kite_lift_coefficient_reel_out: 0.63
  kite_drag_coefficient_reel_out: 0.14
  kite_lift_coefficient_reel_in: 0.4
  kite_drag_coefficient_reel_in: 0.12
  tether_drag_coefficient: 1.1

direct_simulation:
  wind_speeds:
    cut_in: 6.0
    cut_out: 25.0
    n_points: 20
    fine_resolution:
      n_points_near_cutout: 0
      range_m_s: 2.0

optimization:
  wind_speeds:
    cut_in: 3.0
    cut_out: 25.0
    n_points: 25
    fine_resolution:
      n_points_near_cutout: 0
      range_m_s: 2.0
  optimizer:
    optimize_variables:
      reeling_speed_traction: true
      reeling_speed_retraction: true
      fraction_tether_length_traction_end: true
      fraction_tether_length_retraction_end: true
      elevation_angle_traction: true
      elevation_angle_end_trans_rori: true
    max_iterations: 200
    ftol: 5.0e-2
    eps: 5.0e-2
    x0: [2, -2, 0.65, 0.9, 30.0, 50.0]
    scaling: [1, 1, 1, 1, 30, 30]
  bounds:
    reeling_speed_traction_min: 0.01
    reeling_speed_traction_max: 15.0
    reeling_speed_retraction_min: -15.0
    reeling_speed_retraction_max: -0.01
    fraction_tether_length_traction_end_min: 0.8
    fraction_tether_length_traction_end_max: 0.95
    fraction_tether_length_retraction_end_min: 0.4
    fraction_tether_length_retraction_end_max: 0.8
    elevation_angle_traction_min: 10.0
    elevation_angle_traction_max: 60.0
    elevation_angle_end_trans_rori_min: 30.0
    elevation_angle_end_trans_rori_max: 80.0
  constraints:
    min_tether_length_fraction_difference: 0.1
    max_difference_elevation_angle_steps: 10.0

cycle:
  minimum_tether_force: 750.0
  minimum_height: 100.0
  elevation_angle_traction: [30.0, 30.0, 30.0, 30.0]
  tether_length_end_traction: 0.95
  tether_length_end_retraction: 0.65
  include_transition_energy: true
  elevation_angle_end_trans_rori: 50.0

retraction:
  control: ['reeling_speed', -2.0]
  time_step: 0.25
  azimuth_angle: 0.0
  course_angle: 180.0

transition_riro:
  control: ['reeling_speed', 0]
  time_step: 0.05
  azimuth_angle: 0.0
  course_angle: 0.0

transition_rori:
  control: ['reeling_speed', 0]
  time_step: 0.05
  azimuth_angle: 0.0
  course_angle: 180.0

traction:
  control: ['reeling_speed', 2.0]
  time_step: 0.25
  azimuth_angle: 11.5
  course_angle: 93.0

steady_state:
  max_iterations: 250
  convergence_tolerance: 1.0e-3

phase_solver:
  max_time_points: 5000
```

## Usage

Use the example scripts in the `scripts/` directory to run simulations and generate plots:
```bash
python scripts/calculate_power_curves.py
```

### Creating a constructor

All workflows start by instantiating a `PowerCurveConstructor` with paths to the three input files:

```python
from inertiafree_qsm import PowerCurveConstructor

constructor = PowerCurveConstructor(
    system_config_path="data/kitepower V3_20.yml",
    wind_resource_path="data/wind_resource.yml",
    simulation_settings_path="data/simulation_settings.yml",
)
```

### Generating power curves (direct simulation)

`generate_power_curves_direct` runs the QSM with pre-defined cycle parameters from the simulation settings file. This is the fastest method but does not optimize cycle performance.

```python
result = constructor.generate_power_curves_direct(
    cluster_ids=None,        # None = all clusters
    output_path="results/power_curves_direct.yml",
    verbose=True,
    show_plot=True,
    save_plot=True,
)
```

### Generating power curves (optimization)

`generate_power_curves_optimized` uses SLSQP optimization to find the reeling speeds, tether lengths, and elevation angles that maximize average cycle power at each wind speed. Warm starts are used between consecutive wind speeds for faster convergence.

```python
result = constructor.generate_power_curves_optimized(
    cluster_ids=None,        # None = all clusters
    output_path="results/power_curves_optimized.yml",
    verbose=True,
    show_plot=True,
    save_plot=True,
)
```

### Simulating a single wind speed

`simulate_single_wind_speed` evaluates one wind speed point using either method and returns the same output structure as the full power curve methods.

```python
result = constructor.simulate_single_wind_speed(
    wind_speed=8.0,          # Reference wind speed [m/s]
    cluster_id=1,
    method="optimization",   # 'direct' or 'optimization'
    output_path="results/single_point.yml",
    show_plot=True,
    save_plot=True,
)
```

## Output

The main output is a power curves YAML file in the [awesIO](https://github.com/awegroup/awesIO) standard format. It contains the power curve data for each wind cluster, including cut-in and cut-out wind speeds, nominal power, and per-wind-speed performance indicators.

The time history of the cycle simulation (kite kinematics and tether states at each time step) is saved as a separate `.npz` file alongside the YAML output.



## References

[1] R. van der Vlugt, A. Bley, M. Noom, and R. Schmehl: "Quasi-Steady Model of a Pumping Kite Power System". In Renewable Energy, 131, 2019, pp. 83--99. https://doi.org/10.1016/j.renene.2018.07.023

[2] M. Schelbergen: "Power to the Airborne Wind Energy Performance Model". 2024. https://doi.org/10.4233/uuid:353d390a-9b79-44f1-9847-136a6b880e12