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

### Option 1: pip

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: conda

```bash
conda create -n inertiafree-qsm python=3.11 -y
conda activate inertiafree-qsm
pip install -r requirements.txt
```


## References

[1] R. van der Vlugt, A. Bley, M. Noom, and R. Schmehl: "Quasi-Steady Model of a Pumping Kite Power System". In Renewable Energy, 131, 2019, pp. 83--99. https://doi.org/10.1016/j.renene.2018.07.023

[2] M. Schelbergen: "Power to the Airborne Wind Energy Performance Model". 2024. https://doi.org/10.4233/uuid:353d390a-9b79-44f1-9847-136a6b880e12