#!/usr/bin/env python3
"""Profile the optimization loop to identify performance bottlenecks.

Measures wall-clock time at each level: optimizer, objective, cycle phases,
steady-state solver, and deepcopy overhead.
"""

import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inertiafree_qsm import PowerCurveConstructor
from inertiafree_qsm.cycle_optimizer import CycleOptimizer

PROJECT_ROOT = Path(__file__).parent.parent
SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_test.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings_Oriol.yml"


def profile_single_wind_speed(constructor, wind_speed):
    """Run optimisation for one wind speed and collect profiling data."""
    env_state = constructor.create_environment(cluster_id=1)
    optimizer = CycleOptimizer(
        constructor.simulation_settings, constructor.sys_props, env_state,
    )

    # --- cProfile run ---
    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()
    kpi = optimizer.optimize(wind_speed, verbose=False)
    elapsed = time.perf_counter() - t0
    profiler.disable()

    optResult = kpi.get('optimization_result')
    nit = optResult.nit if optResult else '?'
    nfev = optResult.nfev if optResult else '?'
    cyclePower = kpi['average_power']['cycle']

    print(f"\n{'='*72}")
    print(f"  Wind speed: {wind_speed} m/s")
    print(f"  Total wall time: {elapsed:.3f} s")
    print(f"  Iterations: {nit},  Function evals: {nfev}")
    print(f"  Cycle power: {cyclePower:.1f} W")
    print(f"  Time per function eval: {elapsed / nfev * 1000:.1f} ms")
    print(f"{'='*72}")

    # Print top-30 cumulative time entries
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(40)
    print(stream.getvalue())

    # Also print by tottime (self-time, excluding sub-calls)
    stream2 = StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.strip_dirs()
    stats2.sort_stats('tottime')
    stats2.print_stats(30)
    print("\n--- Sorted by self-time (tottime) ---")
    print(stream2.getvalue())

    return elapsed, nfev


if __name__ == "__main__":
    constructor = PowerCurveConstructor(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_file=False, verbose=False,
    )

    windSpeeds = [8.0, 12.0, 18.0]
    results = []
    for ws in windSpeeds:
        elapsed, nfev = profile_single_wind_speed(constructor, ws)
        results.append((ws, elapsed, nfev))

    print("\n\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'WS (m/s)':>10}  {'Total (s)':>10}  {'nfev':>6}  {'ms/eval':>10}")
    for ws, elapsed, nfev in results:
        print(f"  {ws:10.1f}  {elapsed:10.3f}  {nfev:6d}  {elapsed/nfev*1000:10.1f}")
