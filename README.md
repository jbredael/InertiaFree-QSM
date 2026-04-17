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


## References

[1] R. van der Vlugt, A. Bley, M. Noom, and R. Schmehl: "Quasi-Steady Model of a Pumping Kite Power System". In Renewable Energy, 131, 2019, pp. 83--99. https://doi.org/10.1016/j.renene.2018.07.023

[2] M. Schelbergen: "Power to the Airborne Wind Energy Performance Model". 2024. https://doi.org/10.4233/uuid:353d390a-9b79-44f1-9847-136a6b880e12