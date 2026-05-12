#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined moving-vessel AWE optimizer and comparison framework.

This script combines traction-mode optimization, pumping-cycle optimization,
and direct mode comparison into a single configurable workflow using the
InertiaFree-QSM framework extended for moving-vessel applications.

WORKFLOWS
---------
1. Traction optimization heading sweep
2. Pumping-cycle optimization heading sweep
3. Traction-vs-pumping overlay comparison

PURPOSE
-------
The script compares dual-mode airborne wind energy operation on a moving
vessel using a consistent system-level metric under identical wind and
ship-motion conditions.

The vessel is currently treated as a prescribed moving reference frame:

    - fixed ship speed
    - fixed ship heading
    - moving-vessel apparent wind coupling
    - no vessel PPP/hydrodynamic equilibrium yet

MODE DEFINITIONS
----------------

Traction:
    Steady-state pure traction optimization.

    Objective:
        P_equiv_traction = Fx_ship * V_ship

    where:
        Fx_ship = forward ship-force component [N]
        V_ship  = prescribed ship speed [m/s]

Pumping:
    Cyclic quasi-steady pumping optimization with moving-vessel apparent wind.

    Objective:
        P_equiv_pumping = P_cycle + Fx_avg * V_ship

    where:
        P_cycle = cycle-averaged generated power [W]
        Fx_avg  = cycle-averaged forward ship-force [N]

This accounts for propulsion assistance or propulsion penalty caused by
the pumping cycle aerodynamic loading.

PUMPING FILTERING
-----------------
Optional rejection of traction-dominated pumping solutions:

    pump_ratio = P_cycle / abs(Fx_avg * V_ship)

If enabled, cases below:

    PUMPING_MIN_PUMPING_RATIO_DIAGNOSTIC

are rejected to prevent classifying force-dominated solutions as valid
pumping operation.

HEADING SWEEP CONVENTION
------------------------
- True wind direction fixed at 0°.
- Ship heading ψ swept relative to wind.
- QSM solves in apparent-wind-aligned coordinates.
- Forces projected back to ship coordinates.
- 0–180° computed directly.
- Optional mirroring to 180–360° for plotting.

OPTIMIZATION
------------
Traction:
    - SLSQP optimization
    - multistart candidate search
    - warm-start continuation
    - tether-force and surge constraints

Pumping:
    - CycleOptimizer-based optimization
    - multistart initialization
    - warm-start continuation
    - feasibility filtering
    - apparent wind activation threshold

OUTPUTS
-------
CSV:
    - optimized traction results
    - optimized pumping results
    - candidate optimizer diagnostics
    - common-format comparison data
    - summary comparison data

Plots:
    - traction polar map
    - pumping polar map
    - overlay comparison
    - traction operating angle diagnostics

DEPENDENCIES
------------
Requires:

    src/inertiafree_qsm/
    data/kitepower V3_20.yml
    data/wind_resource.yml
    data/simulation_settings.yml

CURRENT LIMITATION
------------------
This implementation uses prescribed vessel motion only.

Future extension:
    full two-way vessel PPP coupling with speed/leeway equilibrium.
"""

from __future__ import annotations

import sys
import csv
import io
import time
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op


# =============================================================================
# USER SETTINGS
# =============================================================================

# ---- Select what to run ------------------------------------------------------
RUN_TRACTION = True
RUN_PUMPING = True
RUN_OVERLAY = True

# ---- Select plots ------------------------------------------------------------
PLOT_TRACTION_POLAR = True
PLOT_TRACTION_OPERATING_ANGLES = True
PLOT_PUMPING_POLAR = True
PLOT_OVERLAY_POLAR = True

SHOW_FIGURES = True
SAVE_FIGURES = True
SAVE_CSV = True
SAVE_CANDIDATE_LOGS = True

# ---- Sweep settings ----------------------------------------------------------
TRUE_WIND_SPEEDS = np.array([16.0], dtype=float)
SHIP_SPEED = 4.0
CLUSTER_ID = 1
TETHER_LENGTH = 500.0
HEADING_STEP_DEG = 15.0
MIRROR_RESULTS_FOR_PLOT = True

# True: compare only wind-speed/heading points existing in both mode results.
USE_COMMON_HEADINGS_ONLY_FOR_OVERLAY = True

# Positive-benefit plotting options.
PLOT_ONLY_POSITIVE_EQUIVALENT_POWER = True
MIN_PLOT_BENEFIT_KW = 0.01

# ---- Traction optimizer settings -------------------------------------------
TRACTION_MAX_CANDIDATE_STARTS = 25
TRACTION_OPTIMIZER_VERBOSE = False
TRACTION_MAX_OPTIMIZER_ITERATIONS = 800
TRACTION_OPTIMIZER_FTOL = 1.0e-9
TRACTION_OPTIMIZER_EPS = 5.0e-4
TRACTION_FORCE_TOLERANCE = 1.0
TRACTION_FORCE_SAFETY_MARGIN = 5.0
TRACTION_USE_TETHER_FORCE_LIMIT = True
TRACTION_DIAGNOSTIC_TETHER_FORCE_MAX = 1.0e9

# ---- Pumping optimizer settings --------------------------------------------
PUMPING_MAX_MULTISTART_CANDIDATES = 3
PUMPING_MAX_OPTIMIZER_ITERATIONS_DEBUG = 80
PUMPING_OPTIMIZER_VERBOSE = False
PUMPING_QUIET_NATIVE_OPTIMIZER_OUTPUT = True
PUMPING_FORCE_TOLERANCE = 1.0
PUMPING_ZERO_POWER_THRESHOLD = 0.0
PUMPING_MIN_P_CYCLE_FOR_VALID_PUMPING = 0.0
PUMPING_MIN_TETHER_LENGTH_FRACTION_DIFFERENCE = 0.05
PUMPING_MIN_PUMPING_RATIO_DIAGNOSTIC = 0.30
PUMPING_REJECT_TRACTION_DOMINATED = False
PUMPING_USE_TETHER_FORCE_LIMIT = True
PUMPING_DIAGNOSTIC_TETHER_FORCE_MAX = 1.0e9

# ---- Output names ------------------------------------------------------------
OUTPUT_TAG = f"{int(SHIP_SPEED * 10):03d}ship_{int(HEADING_STEP_DEG):02d}deg"


# =============================================================================
# Project paths and imports
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name == "scripts" else THIS_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "results" / "combined_mode_sweep"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_20.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings.yml"

TRACTION_CSV_PATH = RESULTS_DIR / f"optimized_traction_{OUTPUT_TAG}.csv"
TRACTION_AUTOSAVE_CSV_PATH = RESULTS_DIR / f"optimized_traction_{OUTPUT_TAG}_autosave.csv"
TRACTION_CANDIDATE_LOG_PATH = RESULTS_DIR / f"optimized_traction_candidate_log_{OUTPUT_TAG}.csv"

PUMPING_CSV_PATH = RESULTS_DIR / f"optimized_pumping_{OUTPUT_TAG}.csv"
PUMPING_AUTOSAVE_CSV_PATH = RESULTS_DIR / f"optimized_pumping_{OUTPUT_TAG}_autosave.csv"
PUMPING_CANDIDATE_LOG_PATH = RESULTS_DIR / f"optimized_pumping_candidate_log_{OUTPUT_TAG}.csv"

COMMON_FORMAT_CSV_PATH = RESULTS_DIR / f"traction_pumping_common_format_{OUTPUT_TAG}.csv"
COMPARISON_SUMMARY_CSV_PATH = RESULTS_DIR / f"traction_pumping_summary_{OUTPUT_TAG}.csv"

TRACTION_POLAR_FIG_PATH = PLOTS_DIR / f"traction_polar_{OUTPUT_TAG}.png"
TRACTION_ANGLES_FIG_PATH = PLOTS_DIR / f"traction_angles_{OUTPUT_TAG}.png"
PUMPING_POLAR_FIG_PATH = PLOTS_DIR / f"pumping_polar_{OUTPUT_TAG}.png"
OVERLAY_POLAR_FIG_PATH = PLOTS_DIR / f"overlay_traction_pumping_{OUTPUT_TAG}.png"

from inertiafree_qsm import PowerCurveConstructor
from inertiafree_qsm.operating_point import VesselState, WindCondition
from inertiafree_qsm.pure_traction import PureTractionInput, PureTractionSolver
from inertiafree_qsm.coordinate_transforms import (
    qsm_kite_position_to_global_force,
    qsm_kite_position_to_ship_force,
)
from inertiafree_qsm.vessel_coupling import TrueWind, VesselMotion, compute_apparent_wind
from inertiafree_qsm.cycle_optimizer import CycleOptimizer


# =============================================================================
# Shared helpers
# =============================================================================

GRAVITATIONAL_ACCELERATION = 9.80665
EPS_POWER = 1.0e-9


def build_sweep_headings() -> np.ndarray:
    if MIRROR_RESULTS_FOR_PLOT:
        return np.arange(0.0, 181.0, HEADING_STEP_DEG)
    return np.arange(0.0, 360.0, HEADING_STEP_DEG)


SWEEP_HEADINGS = build_sweep_headings()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ["true", "1", "yes", "y"]


def format_s(value_s: float) -> str:
    if not np.isfinite(value_s):
        return "   nan"
    return f"{value_s:6.1f}"


def format_kw_from_w(value_w: float) -> str:
    if not np.isfinite(value_w):
        return "    nan"
    return f"{value_w / 1000.0:7.3f}"


def write_csv(rows: list[dict[str, Any]], path: Path, quiet: bool = False) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not quiet:
        print(f"Saved CSV:\n  {path}")


def close_polar_curve(theta: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if len(theta) == 0:
        return theta, radius
    return (
        np.concatenate([theta, [theta[0] + 2.0 * np.pi]]),
        np.concatenate([radius, [radius[0]]]),
    )


def compute_case_apparent_wind(true_wind_speed: float, ship_speed: float, heading_deg: float):
    true_wind = TrueWind(speed=float(true_wind_speed), direction_to=np.deg2rad(0.0))
    vessel_heading = np.deg2rad(float(heading_deg))
    vessel_motion = VesselMotion(speed=float(ship_speed), heading=vessel_heading)
    apparent_wind = compute_apparent_wind(true_wind, vessel_motion)
    return apparent_wind, vessel_heading


def result_radius_kw(value_w: float, accepted: bool) -> float:
    if not accepted or not np.isfinite(value_w):
        return 0.0
    value_kw = value_w / 1000.0
    if PLOT_ONLY_POSITIVE_EQUIVALENT_POWER:
        if value_kw <= MIN_PLOT_BENEFIT_KW:
            return 0.0
        return max(value_kw, 0.0)
    return value_kw

def build_force_limit_text(
    physical_tether_force_max: float,
    active_tether_force_max: float,
    force_limit_enabled: bool,
    mode_name: str,
) -> str:
    if force_limit_enabled:
        return (
            f"{mode_name} force limit ON: "
            f"T ≤ {physical_tether_force_max / 1000.0:.2f} kN"
        )

    return (
        f"{mode_name} force limit OFF diagnostic: "
        f"physical limit = {physical_tether_force_max / 1000.0:.2f} kN, "
        f"optimizer/check limit = {active_tether_force_max / 1000.0:.0f} kN"
    )

# =============================================================================
# Constructor setup
# =============================================================================


def create_constructor_and_environment():
    constructor = PowerCurveConstructor(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_file=False,
        verbose=False,
    )
    env_state = constructor.create_environment(cluster_id=CLUSTER_ID)
    return constructor, env_state


# =============================================================================
# Traction optimization
# =============================================================================

TRACTION_INITIAL_GUESS_DEG = np.array([11.5, 30.0, 93.0], dtype=float)
TRACTION_BOUNDS_DEG = [(-90.0, 90.0), (10.0, 80.0), (-180.0, 180.0)]
TRACTION_SCALING = np.array([30.0, 30.0, 180.0], dtype=float)


def solve_traction_with_measured_tether_force(solver, wind, traction_input, vessel_state):
    old_limit = getattr(solver.sys_props, "tether_force_max_limit", None)
    if old_limit is None:
        raise ValueError("sys_props.tether_force_max_limit is not defined.")
    solver.sys_props.tether_force_max_limit = 1.0e12
    try:
        result = solver.solve(wind=wind, traction_input=traction_input, vessel_state=vessel_state)
    finally:
        solver.sys_props.tether_force_max_limit = old_limit
    return result


def evaluate_traction_operating_point(solver, true_wind_speed, ship_speed, heading_deg, x_deg):
    x_deg = np.asarray(x_deg, dtype=float)
    azimuth_deg, elevation_deg, course_deg = x_deg

    apparent_wind, vessel_heading = compute_case_apparent_wind(true_wind_speed, ship_speed, heading_deg)

    traction_input = PureTractionInput(
        tether_length=TETHER_LENGTH,
        azimuth_angle=np.deg2rad(azimuth_deg),
        elevation_angle=np.deg2rad(elevation_deg),
        course_angle=np.deg2rad(course_deg),
    )

    wind_qsm = WindCondition(speed=apparent_wind.speed, direction=0.0)
    vessel_state_qsm = VesselState(speed=float(ship_speed), leeway_angle=0.0, heading=0.0)

    result = solve_traction_with_measured_tether_force(
        solver=solver,
        wind=wind_qsm,
        traction_input=traction_input,
        vessel_state=vessel_state_qsm,
    )

    Fx_ship, Fy_ship = qsm_kite_position_to_ship_force(
        tether_force_ground=result.tether_force_ground,
        azimuth_angle_qsm=traction_input.azimuth_angle,
        elevation_angle=traction_input.elevation_angle,
        apparent_wind_direction=apparent_wind.direction_to,
        vessel_heading=vessel_heading,
    )

    P_equiv = Fx_ship * ship_speed
    return {
        "evaluation_success": True,
        "error_message": "",
        "azimuth_angle_deg": float(azimuth_deg),
        "elevation_angle_deg": float(elevation_deg),
        "course_angle_deg": float(course_deg),
        "true_wind_speed": float(true_wind_speed),
        "ship_speed": float(ship_speed),
        "heading_deg": float(heading_deg),
        "apparent_wind_speed": float(apparent_wind.speed),
        "apparent_wind_direction_deg": float(np.rad2deg(apparent_wind.direction_to)),
        "tether_force_ground": float(result.tether_force_ground),
        "Fx": float(Fx_ship),
        "Fy": float(Fy_ship),
        "P_equiv_traction": float(P_equiv),
        "P_equiv_traction_kW": float(P_equiv / 1000.0),
    }


def add_projection_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    Fx = safe_float(result.get("Fx"))
    T = safe_float(result.get("tether_force_ground"))
    if np.isfinite(Fx) and np.isfinite(T) and T > 0.0:
        ratio = Fx / T
        loss = np.rad2deg(np.arccos(np.clip(ratio, -1.0, 1.0)))
    else:
        ratio = np.nan
        loss = np.nan
    result["Fx_over_tether_force"] = float(ratio)
    result["projection_loss_angle_deg"] = float(loss)
    return result


class TractionOptimizer:
    def __init__(self, solver, true_wind_speed, ship_speed, heading_deg, tether_force_max):
        self.solver = solver
        self.true_wind_speed = float(true_wind_speed)
        self.ship_speed = float(ship_speed)
        self.heading_deg = float(heading_deg)
        self.tether_force_max = float(tether_force_max)
        self.history = []
        self._cache_x_scaled = None
        self._cache_result = None

    def _evaluate_scaled(self, x_scaled):
        x_scaled = np.asarray(x_scaled, dtype=float)
        if self._cache_x_scaled is not None and np.allclose(x_scaled, self._cache_x_scaled, rtol=0.0, atol=1.0e-12):
            return self._cache_result
        x_deg = x_scaled * TRACTION_SCALING
        try:
            result = evaluate_traction_operating_point(
                self.solver, self.true_wind_speed, self.ship_speed, self.heading_deg, x_deg
            )
        except Exception as exc:
            result = {
                "evaluation_success": False,
                "error_message": str(exc),
                "azimuth_angle_deg": float(x_deg[0]),
                "elevation_angle_deg": float(x_deg[1]),
                "course_angle_deg": float(x_deg[2]),
                "true_wind_speed": self.true_wind_speed,
                "ship_speed": self.ship_speed,
                "heading_deg": self.heading_deg,
                "Fx": -np.inf,
                "Fy": np.nan,
                "tether_force_ground": np.inf,
                "P_equiv_traction": -np.inf,
                "P_equiv_traction_kW": -np.inf,
                "apparent_wind_speed": np.nan,
                "apparent_wind_direction_deg": np.nan,
            }
        self._cache_x_scaled = x_scaled.copy()
        self._cache_result = result
        return result

    def objective(self, x_scaled):
        result = self._evaluate_scaled(x_scaled)
        if not result.get("evaluation_success", False):
            return 1.0e12
        P = result["P_equiv_traction"]
        if not np.isfinite(P):
            return 1.0e12
        return -P / 1000.0

    def constraint_positive_surge(self, x_scaled):
        result = self._evaluate_scaled(x_scaled)
        if not result.get("evaluation_success", False):
            return -1.0
        return result["Fx"] / self.tether_force_max

    def constraint_tether_force(self, x_scaled):
        result = self._evaluate_scaled(x_scaled)
        if not result.get("evaluation_success", False):
            return -1.0
        return (self.tether_force_max - TRACTION_FORCE_SAFETY_MARGIN - result["tether_force_ground"]) / self.tether_force_max

    def optimize(self, x0_deg, verbose=False):
        x0_scaled = np.asarray(x0_deg, dtype=float) / TRACTION_SCALING
        bounds_scaled = [(lo / s, hi / s) for (lo, hi), s in zip(TRACTION_BOUNDS_DEG, TRACTION_SCALING)]
        constraints = [
            {"type": "ineq", "fun": self.constraint_positive_surge},
            {"type": "ineq", "fun": self.constraint_tether_force},
        ]
        opt_result = op.minimize(
            fun=self.objective,
            x0=x0_scaled,
            method="SLSQP",
            bounds=bounds_scaled,
            constraints=constraints,
            options={
                "maxiter": TRACTION_MAX_OPTIMIZER_ITERATIONS,
                "ftol": TRACTION_OPTIMIZER_FTOL,
                "eps": TRACTION_OPTIMIZER_EPS,
                "disp": verbose,
            },
        )
        final = self._evaluate_scaled(opt_result.x)
        final["optimizer_success"] = bool(opt_result.success)
        final["optimizer_message"] = str(opt_result.message)
        final["positive_surge_ok"] = bool(final["Fx"] >= 0.0)
        final["tether_constraint_violation"] = float(max(0.0, final["tether_force_ground"] - self.tether_force_max))
        final["optimizer_tether_constraint_violation"] = float(
            max(0.0, final["tether_force_ground"] - (self.tether_force_max - TRACTION_FORCE_SAFETY_MARGIN))
        )
        final["physical_tether_constraint_ok"] = bool(final["tether_force_ground"] <= self.tether_force_max + TRACTION_FORCE_TOLERANCE)
        final["tether_constraint_ok"] = bool(
            final["tether_force_ground"] <= self.tether_force_max - TRACTION_FORCE_SAFETY_MARGIN + TRACTION_FORCE_TOLERANCE
        )
        final["accepted_result"] = bool(final["evaluation_success"] and final["positive_surge_ok"] and final["tether_constraint_ok"])
        final["accepted_despite_optimizer_message"] = bool(final["accepted_result"] and not final["optimizer_success"])
        final["constraint_active"] = bool(
            np.isclose(final["tether_force_ground"], self.tether_force_max - TRACTION_FORCE_SAFETY_MARGIN, rtol=0.0, atol=25.0)
        )
        final["x_opt_deg"] = np.array([final["azimuth_angle_deg"], final["elevation_angle_deg"], final["course_angle_deg"]], dtype=float)
        return add_projection_diagnostics(final)


def traction_candidate_starts(warm_start_deg=None):
    starts = []
    if warm_start_deg is not None:
        starts.append(np.asarray(warm_start_deg, dtype=float))
    starts.extend([
        TRACTION_INITIAL_GUESS_DEG,
        np.array([-11.5, 30.0, -93.0]),
        np.array([0.0, 10.0, 90.0]), np.array([0.0, 30.0, 90.0]), np.array([0.0, 45.0, 90.0]),
        np.array([0.0, 10.0, -90.0]), np.array([0.0, 30.0, -90.0]), np.array([0.0, 45.0, -90.0]),
        np.array([30.0, 20.0, 90.0]), np.array([-30.0, 20.0, -90.0]),
        np.array([60.0, 30.0, 90.0]), np.array([-60.0, 30.0, -90.0]),
        np.array([0.0, 10.0, 0.0]), np.array([15.0, 10.0, 0.0]), np.array([-15.0, 10.0, 0.0]),
        np.array([30.0, 10.0, 0.0]), np.array([-30.0, 10.0, 0.0]),
        np.array([60.0, 10.0, -30.0]), np.array([60.0, 15.0, 30.0]),
        np.array([-60.0, 10.0, -30.0]), np.array([-60.0, 15.0, 30.0]),
    ])
    unique = []
    seen = set()
    for x in starts:
        key = tuple(np.round(x, 6))
        if key not in seen:
            seen.add(key)
            unique.append(x.astype(float))
    if TRACTION_MAX_CANDIDATE_STARTS is not None:
        unique = unique[:TRACTION_MAX_CANDIDATE_STARTS]
    return unique


def is_feasible_traction_result(result, tether_force_max):
    if not result.get("evaluation_success", False):
        return False
    P = safe_float(result.get("P_equiv_traction"))
    Fx = safe_float(result.get("Fx"))
    T = safe_float(result.get("tether_force_ground"))
    return np.isfinite(P) and np.isfinite(Fx) and np.isfinite(T) and Fx >= -1e-6 and T <= tether_force_max - TRACTION_FORCE_SAFETY_MARGIN + TRACTION_FORCE_TOLERANCE


def build_traction_candidate_log_row(result, candidate_index, x0_deg, selected_best=False):
    return {
        "true_wind_speed": result.get("true_wind_speed", np.nan),
        "ship_speed": result.get("ship_speed", np.nan),
        "heading_deg": result.get("heading_deg", np.nan),
        "candidate_index": candidate_index,
        "selected_best": selected_best,
        "candidate_runtime_s": result.get("candidate_runtime_s", np.nan),
        "case_runtime_s": result.get("case_runtime_s", np.nan),
        "start_azimuth_angle_deg": float(x0_deg[0]),
        "start_elevation_angle_deg": float(x0_deg[1]),
        "start_course_angle_deg": float(x0_deg[2]),
        "evaluation_success": result.get("evaluation_success", False),
        "optimizer_success": result.get("optimizer_success", False),
        "accepted_result": result.get("accepted_result", False),
        "optimizer_message": result.get("optimizer_message", ""),
        "optimized_azimuth_angle_deg": result.get("azimuth_angle_deg", np.nan),
        "optimized_elevation_angle_deg": result.get("elevation_angle_deg", np.nan),
        "optimized_course_angle_deg": result.get("course_angle_deg", np.nan),
        "optimized_Fx": result.get("Fx", np.nan),
        "optimized_Fy": result.get("Fy", np.nan),
        "optimized_tether_force_ground": result.get("tether_force_ground", np.nan),
        "Fx_over_tether_force": result.get("Fx_over_tether_force", np.nan),
        "projection_loss_angle_deg": result.get("projection_loss_angle_deg", np.nan),
        "optimized_P_equiv_traction": result.get("P_equiv_traction", np.nan),
        "optimized_P_equiv_traction_kW": result.get("P_equiv_traction_kW", np.nan),
    }


def optimize_traction_heading_with_retries(solver, true_wind_speed, ship_speed, heading_deg, tether_force_max, warm_start_deg=None):
    case_start = time.perf_counter()
    results = []
    candidate_rows = []
    runtimes = []
    starts = traction_candidate_starts(warm_start_deg)

    for idx, x0 in enumerate(starts):
        t0 = time.perf_counter()
        opt = TractionOptimizer(solver, true_wind_speed, ship_speed, heading_deg, tether_force_max)
        result = opt.optimize(x0_deg=x0, verbose=TRACTION_OPTIMIZER_VERBOSE)
        dt = time.perf_counter() - t0
        result["candidate_index"] = idx
        result["candidate_runtime_s"] = float(dt)
        results.append(result)
        runtimes.append(float(dt))
        candidate_rows.append(build_traction_candidate_log_row(result, idx, x0, False))

    case_runtime = time.perf_counter() - case_start
    for r in results:
        r["case_runtime_s"] = float(case_runtime)
        r["n_candidates_tried"] = len(results)
        r["max_candidate_runtime_s"] = float(max(runtimes)) if runtimes else 0.0
        r["mean_candidate_runtime_s"] = float(np.mean(runtimes)) if runtimes else 0.0
        r["sum_candidate_runtime_s"] = float(np.sum(runtimes)) if runtimes else 0.0

    feasible = [r for r in results if is_feasible_traction_result(r, tether_force_max)]
    pool = feasible if feasible else [r for r in results if r.get("evaluation_success", False) and np.isfinite(r.get("P_equiv_traction", np.nan))]
    best = max(pool, key=lambda r: r["P_equiv_traction"]) if pool else results[0]
    best["selected_candidate_index"] = int(best.get("candidate_index", -1))
    best["selected_candidate_runtime_s"] = float(best.get("candidate_runtime_s", np.nan))

    for row in candidate_rows:
        row["case_runtime_s"] = float(case_runtime)
        row["selected_best"] = int(row["candidate_index"]) == best["selected_candidate_index"]

    return best, candidate_rows


def build_traction_output_row(result, tether_force_max):
    return {
        "true_wind_speed": result["true_wind_speed"],
        "ship_speed": result["ship_speed"],
        "heading_deg": result["heading_deg"],
        "apparent_wind_speed": result["apparent_wind_speed"],
        "apparent_wind_direction_deg": result["apparent_wind_direction_deg"],
        "optimized_success": result["evaluation_success"],
        "optimizer_success": result["optimizer_success"],
        "accepted_result": result["accepted_result"],
        "accepted_despite_optimizer_message": result.get("accepted_despite_optimizer_message", False),
        "optimizer_message": result["optimizer_message"],
        "case_runtime_s": result.get("case_runtime_s", np.nan),
        "n_candidates_tried": result.get("n_candidates_tried", np.nan),
        "selected_candidate_index": result.get("selected_candidate_index", np.nan),
        "selected_candidate_runtime_s": result.get("selected_candidate_runtime_s", np.nan),
        "max_candidate_runtime_s": result.get("max_candidate_runtime_s", np.nan),
        "mean_candidate_runtime_s": result.get("mean_candidate_runtime_s", np.nan),
        "sum_candidate_runtime_s": result.get("sum_candidate_runtime_s", np.nan),
        "optimized_azimuth_angle_deg": result["azimuth_angle_deg"],
        "optimized_elevation_angle_deg": result["elevation_angle_deg"],
        "optimized_course_angle_deg": result["course_angle_deg"],
        "optimized_Fx": result["Fx"],
        "optimized_Fy": result["Fy"],
        "optimized_tether_force_ground": result["tether_force_ground"],
        "Fx_over_tether_force": result.get("Fx_over_tether_force", np.nan),
        "projection_loss_angle_deg": result.get("projection_loss_angle_deg", np.nan),
        "tether_force_max": tether_force_max,
        "tether_force_effective_limit": tether_force_max - TRACTION_FORCE_SAFETY_MARGIN,
        "tether_constraint_violation": result["tether_constraint_violation"],
        "optimizer_tether_constraint_violation": result["optimizer_tether_constraint_violation"],
        "positive_surge_ok": result["positive_surge_ok"],
        "physical_tether_constraint_ok": result["physical_tether_constraint_ok"],
        "tether_constraint_ok": result["tether_constraint_ok"],
        "constraint_active": result["constraint_active"],
        "optimized_P_equiv_traction": result["P_equiv_traction"],
        "optimized_P_equiv_traction_kW": result["P_equiv_traction"] / 1000.0,
    }


def print_traction_progress(wind_speed, i, total, heading_deg, row, accepted_count, failed_count, best_power):
    if row["accepted_result"]:
        P_kw = row["optimized_P_equiv_traction"] / 1000.0
        Fx = row["optimized_Fx"]
        T = row["optimized_tether_force_ground"]
        az = row["optimized_azimuth_angle_deg"]
        beta = row["optimized_elevation_angle_deg"]
        course = row["optimized_course_angle_deg"]
        Fx_over_T = row.get("Fx_over_tether_force", np.nan)
        proj_loss = row.get("projection_loss_angle_deg", np.nan)
    else:
        P_kw = Fx = T = az = beta = course = Fx_over_T = proj_loss = np.nan

    best_kw = best_power / 1000.0 if np.isfinite(best_power) else np.nan
    msg = (
        f"  Vw={wind_speed:5.1f} m/s | {i:03d}/{total:03d} | ψ={heading_deg:7.2f} deg | "
        f"t={format_s(row.get('case_runtime_s', np.nan))}s | "
        f"cand={int(row.get('n_candidates_tried', 0)) if np.isfinite(row.get('n_candidates_tried', np.nan)) else 0:02d} | "
        f"tcand,max={format_s(row.get('max_candidate_runtime_s', np.nan))}s | "
        f"accepted={accepted_count:03d} | failed={failed_count:03d} | "
        f"Peq={P_kw:9.3f} kW | Fx={Fx:9.1f} N | T={T:9.1f} N | "
        f"az={az:7.2f}° | β={beta:6.2f}° | course={course:8.2f}° | "
        f"Fx/T={Fx_over_T:6.3f} | loss={proj_loss:6.2f}° | best={best_kw:9.3f} kW"
    )
    sys.stdout.write("\r" + msg.ljust(360))
    sys.stdout.flush()


def run_traction_sweep(constructor, env_state):
    solver = PureTractionSolver(
        sys_props=constructor.sys_props,
        env_state=env_state,
        steady_state_config=constructor.simulation_settings.get("steady_state"),
    )
    physical_tether_force_max = float(constructor.sys_props.tether_force_max_limit)
    tether_force_max = physical_tether_force_max if TRACTION_USE_TETHER_FORCE_LIMIT else TRACTION_DIAGNOSTIC_TETHER_FORCE_MAX

    print("\nCONSTRAINED TRACTION OPTIMIZED WIND-HEADING SWEEP")
    print("-------------------------------------------------")
    print(f"True wind speeds      : {TRUE_WIND_SPEEDS}")
    print(f"Ship speed            : {SHIP_SPEED:.3f} m/s")
    print(f"Tether length         : {TETHER_LENGTH:.3f} m")
    print(f"Computed headings     : {SWEEP_HEADINGS[0]:.1f}–{SWEEP_HEADINGS[-1]:.1f} deg")
    print(f"Heading step          : {HEADING_STEP_DEG:.3f} deg")
    print(f"Mirrored for plot     : {MIRROR_RESULTS_FOR_PLOT}")
    print(f"Candidate starts      : {'all' if TRACTION_MAX_CANDIDATE_STARTS is None else TRACTION_MAX_CANDIDATE_STARTS}")
    print(f"Tether force max      : {tether_force_max:.3f} N")
    print("Objective             : maximize P_equiv_traction = Fx_ship * V_ship")

    all_rows = []
    all_candidate_rows = []

    for wind_speed in TRUE_WIND_SPEEDS:
        print(f"\nRunning optimized traction sweep: true wind = {wind_speed:.1f} m/s, ship speed = {SHIP_SPEED:.1f} m/s")
        warm_start_deg = TRACTION_INITIAL_GUESS_DEG.copy()
        accepted_count = 0
        failed_count = 0
        best_power = -np.inf

        for i, heading_deg in enumerate(SWEEP_HEADINGS, start=1):
            best, case_candidates = optimize_traction_heading_with_retries(
                solver, wind_speed, SHIP_SPEED, heading_deg, tether_force_max, warm_start_deg
            )
            if best.get("accepted_result", False):
                warm_start_deg = best["x_opt_deg"]
            row = build_traction_output_row(best, tether_force_max)
            all_rows.append(row)
            all_candidate_rows.extend(case_candidates)
            if row["accepted_result"]:
                accepted_count += 1
                best_power = max(best_power, row["optimized_P_equiv_traction"])
            else:
                failed_count += 1
            if SAVE_CSV:
                write_csv(all_rows, TRACTION_AUTOSAVE_CSV_PATH, quiet=True)
            print_traction_progress(wind_speed, i, len(SWEEP_HEADINGS), heading_deg, row, accepted_count, failed_count, best_power)
        print("")

    if SAVE_CSV:
        write_csv(all_rows, TRACTION_CSV_PATH)
    if SAVE_CANDIDATE_LOGS:
        write_csv(all_candidate_rows, TRACTION_CANDIDATE_LOG_PATH)
    print_traction_summary(all_rows)
    return all_rows, all_candidate_rows


def print_traction_summary(rows):
    print("\nTRACTION SUMMARY")
    print("----------------")
    for wind_speed in sorted(set(float(r["true_wind_speed"]) for r in rows)):
        group = [r for r in rows if np.isclose(float(r["true_wind_speed"]), wind_speed)]
        accepted = [r for r in group if r.get("accepted_result", False)]
        if accepted:
            best = max(accepted, key=lambda r: r["optimized_P_equiv_traction"])
            print(f"Vw={wind_speed:5.1f} m/s | accepted={len(accepted):02d} | failed={len(group)-len(accepted):02d} | best Peq={best['optimized_P_equiv_traction_kW']:9.3f} kW at ψ={best['heading_deg']:7.2f} deg")
        else:
            print(f"Vw={wind_speed:5.1f} m/s | accepted=00 | failed={len(group):02d} | best Peq=nan")


# =============================================================================
# Pumping optimization
# =============================================================================


def calculate_static_takeoff_wind_speed(sys_props: Any, env_state: Any) -> float:
    kite_mass = safe_float(getattr(sys_props, "kite_mass", np.nan))
    kite_area = safe_float(getattr(sys_props, "kite_projected_area", np.nan))
    lift_coefficient = safe_float(getattr(sys_props, "kite_lift_coefficient_powered", np.nan))
    air_density = safe_float(getattr(env_state, "rho_0", np.nan))
    if not np.isfinite(air_density):
        air_density = safe_float(getattr(env_state, "air_density", np.nan))
    invalid = [name for name, val in {
        "kite_mass": kite_mass,
        "kite_projected_area": kite_area,
        "kite_lift_coefficient_powered": lift_coefficient,
        "air_density": air_density,
    }.items() if not np.isfinite(val) or val <= 0.0]
    if invalid:
        raise ValueError("Cannot calculate static take-off wind speed. Invalid values: " + ", ".join(invalid))
    return float(np.sqrt(2.0 * GRAVITATIONAL_ACCELERATION * kite_mass / (air_density * lift_coefficient * kite_area)))


def compute_pumping_quality_diagnostics(P_cycle: float, P_prop_equiv: float):
    P_prop_abs = abs(float(P_prop_equiv))
    P_cycle_abs = abs(float(P_cycle))
    pump_ratio = float(P_cycle / P_prop_abs) if P_prop_abs > EPS_POWER else np.inf
    denom = P_cycle_abs + P_prop_abs
    P_prop_fraction_abs = float(P_prop_abs / denom) if denom > EPS_POWER else np.nan
    traction_dominated_flag = bool(np.isfinite(pump_ratio) and pump_ratio < PUMPING_MIN_PUMPING_RATIO_DIAGNOSTIC and P_prop_abs > EPS_POWER)
    return {
        "P_prop_abs": float(P_prop_abs),
        "pump_ratio": float(pump_ratio),
        "P_prop_fraction_abs": float(P_prop_fraction_abs),
        "traction_dominated_flag": traction_dominated_flag,
    }


def extract_cycle_samples_from_kpi(kpi: dict[str, Any]):
    if "kinematics" not in kpi or "steady_states" not in kpi or "time" not in kpi:
        raise KeyError("KPI does not contain required cycle histories: kinematics, steady_states, time.")
    kinematics = list(kpi["kinematics"])
    steady_states = list(kpi["steady_states"])
    time_values = np.asarray(kpi["time"], dtype=float)
    n = min(len(kinematics), len(steady_states), len(time_values))
    if n <= 0:
        raise ValueError("No overlapping cycle samples found.")
    kinematics = kinematics[:n]
    steady_states = steady_states[:n]
    time_values = time_values[:n]
    return {
        "time": time_values,
        "tether_force_ground": np.array([ss.tether_force_ground for ss in steady_states], dtype=float),
        "azimuth_angle_qsm": np.array([kin.azimuth_angle for kin in kinematics], dtype=float),
        "elevation_angle": np.array([kin.elevation_angle for kin in kinematics], dtype=float),
    }


def time_average(values, time_values):
    values = np.asarray(values, dtype=float)
    time_values = np.asarray(time_values, dtype=float)
    if len(values) == 1:
        return float(values[0])
    duration = float(time_values[-1] - time_values[0])
    if duration <= 0.0:
        return float(np.mean(values))
    integral = np.trapezoid(values, time_values) if hasattr(np, "trapezoid") else np.trapz(values, time_values)
    return float(integral / duration)


def compute_cycle_average_ship_forces_from_kpi(kpi, apparent_wind_direction, vessel_heading):
    samples = extract_cycle_samples_from_kpi(kpi)
    time_values = samples["time"]
    T = samples["tether_force_ground"]
    az = samples["azimuth_angle_qsm"]
    el = samples["elevation_angle"]
    n = len(time_values)
    Fx_ship = np.zeros(n)
    Fy_ship = np.zeros(n)
    Fx_global = np.zeros(n)
    Fy_global = np.zeros(n)
    for i in range(n):
        Fx_global[i], Fy_global[i] = qsm_kite_position_to_global_force(T[i], az[i], el[i], apparent_wind_direction)
        Fx_ship[i], Fy_ship[i] = qsm_kite_position_to_ship_force(T[i], az[i], el[i], apparent_wind_direction, vessel_heading)
    return {
        "Fx_avg": time_average(Fx_ship, time_values),
        "Fy_avg": time_average(Fy_ship, time_values),
        "Fx_global_avg": time_average(Fx_global, time_values),
        "Fy_global_avg": time_average(Fy_global, time_values),
        "mean_tether_force": time_average(T, time_values),
        "max_tether_force": float(np.max(T)),
        "n_force_samples": int(n),
    }


class MovingVesselPumpingOptimizer(CycleOptimizer):
    def __init__(self, simulation_settings, sys_props, env_state, ship_speed, apparent_wind_direction, vessel_heading):
        super().__init__(simulation_settings=simulation_settings, sys_props=sys_props, env_state=env_state)
        self.ship_speed = float(ship_speed)
        self.apparent_wind_direction = float(apparent_wind_direction)
        self.vessel_heading = float(vessel_heading)
        self.objective_name = "P_equiv_pumping"
        self.last_var_names = []
        self.last_x_opt = None

    def _objective_metric(self, kpi):
        if not kpi.get("sim_successful", False):
            return 0.0, {}
        P_cycle = float(kpi["average_power"]["cycle"])
        force_data = compute_cycle_average_ship_forces_from_kpi(kpi, self.apparent_wind_direction, self.vessel_heading)
        Fx_avg = force_data["Fx_avg"]
        P_prop_equiv = Fx_avg * self.ship_speed
        P_equiv_pumping = P_cycle + P_prop_equiv
        data = {
            "P_cycle": P_cycle,
            "Fx_avg": Fx_avg,
            "Fy_avg": force_data["Fy_avg"],
            "Fx_global_avg": force_data["Fx_global_avg"],
            "Fy_global_avg": force_data["Fy_global_avg"],
            "mean_tether_force": force_data["mean_tether_force"],
            "max_tether_force": force_data["max_tether_force"],
            "P_prop_equiv": P_prop_equiv,
            "P_prop_penalty": max(-P_prop_equiv, 0.0),
            "P_equiv_pumping": P_equiv_pumping,
            "n_force_samples": force_data["n_force_samples"],
            **compute_pumping_quality_diagnostics(P_cycle, P_prop_equiv),
        }
        return float(P_equiv_pumping), data

    def _objective(self, x_unscaled, var_names):
        kpi = self._cached_run_cycle(x_unscaled, var_names)
        feasible = bool(kpi["sim_successful"])
        if feasible:
            objective_value, objective_data = self._objective_metric(kpi)
        else:
            objective_value, objective_data = 0.0, {}
        cycle_power = float(kpi["average_power"]["cycle"]) if feasible else 0.0
        self.history.append({
            "x": x_unscaled.copy(),
            "power": cycle_power,
            "objective_value": objective_value,
            "objective_name": self.objective_name,
            "feasible": feasible,
            **objective_data,
        })
        return -objective_value

    def _finalise_result(self, result, scaling, var_names):
        x_opt = result.x * scaling
        kpi = self._run_cycle(x_opt, var_names, use_opt_timesteps=False)
        if kpi["sim_successful"]:
            objective, data = self._objective_metric(kpi)
        else:
            objective, data = 0.0, {}
        kpi["objective_value"] = objective
        kpi["objective_name"] = self.objective_name
        kpi["objective_data"] = data

        best_hist = max(
            (e for e in self.history if e.get("feasible", False) and np.isfinite(e.get("objective_value", np.nan))),
            key=lambda e: e["objective_value"],
            default=None,
        )
        if best_hist is not None and not np.allclose(best_hist["x"], x_opt, rtol=0.0, atol=1.0e-6):
            kpi_hist = self._run_cycle(best_hist["x"], var_names, use_opt_timesteps=False)
            if kpi_hist["sim_successful"]:
                hist_objective, hist_data = self._objective_metric(kpi_hist)
                if hist_objective > objective:
                    x_opt = best_hist["x"]
                    kpi = kpi_hist
                    kpi["objective_value"] = hist_objective
                    kpi["objective_name"] = self.objective_name
                    kpi["objective_data"] = hist_data

        self.last_var_names = list(var_names)
        self.last_x_opt = np.asarray(x_opt, dtype=float)
        return kpi, x_opt


def get_tether_force_max(constructor):
    value = getattr(constructor.sys_props, "tether_force_max_limit", np.inf)
    return np.inf if value is None else float(value)


def get_max_tether_length(constructor):
    value = getattr(constructor.sys_props, "max_tether_length", np.nan)
    return np.nan if value is None else float(value)


def get_base_x0(simulation_settings):
    return np.array(simulation_settings["optimization"]["optimizer"]["x0"], dtype=float)


def set_base_x0(simulation_settings, x0_base):
    settings = deepcopy(simulation_settings)
    settings["optimization"]["optimizer"]["x0"] = [float(v) for v in np.asarray(x0_base, dtype=float)]
    return settings


def clip_reeling_speeds_in_x0(x0_base, simulation_settings, apparent_wind_speed, out_factor, in_factor):
    x = np.asarray(x0_base, dtype=float).copy()
    bounds = simulation_settings["optimization"]["bounds"]
    if len(x) > 0 and "reeling_speed_out" in bounds:
        lo, hi = bounds["reeling_speed_out"]
        x[0] = np.clip(out_factor * apparent_wind_speed, lo, hi)
    if len(x) > 1 and "reeling_speed_in" in bounds:
        lo, hi = bounds["reeling_speed_in"]
        x[1] = np.clip(-abs(in_factor * apparent_wind_speed), lo, hi)
    return x


def active_solution_to_base_x0(x0_base, var_names, x_opt):
    x0 = np.asarray(x0_base, dtype=float).copy()
    for name, value in zip(var_names, x_opt):
        if name == "reeling_speed_out" and len(x0) > 0:
            x0[0] = value
        elif name == "reeling_speed_in" and len(x0) > 1:
            x0[1] = value
        elif name == "frac_end" and len(x0) > 2:
            x0[2] = value
        elif name == "frac_start" and len(x0) > 3:
            x0[3] = value
        elif name.startswith("elevation_") and name.split("_")[1].isdigit():
            idx = int(name.split("_")[1])
            base_idx = 4 + idx
            if base_idx < len(x0):
                x0[base_idx] = value
        elif name == "elevation_end_rori" and len(x0) > 5:
            x0[-1] = value
    return x0


def build_pumping_x0_candidates(simulation_settings, apparent_wind_speed, warm_x0_base=None):
    base = get_base_x0(simulation_settings)
    candidates = []
    if warm_x0_base is not None:
        candidates.append(np.asarray(warm_x0_base, dtype=float))
    candidates.append(base)
    for out_factor, in_factor in [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40)]:
        candidates.append(clip_reeling_speeds_in_x0(base, simulation_settings, apparent_wind_speed, out_factor, in_factor))
    if len(base) > 4:
        for factor in [0.85, 1.15, 1.30]:
            x = base.copy()
            x[4:] *= factor
            candidates.append(x)
    unique = []
    seen = set()
    for x in candidates:
        key = tuple(np.round(x, 8))
        if key not in seen:
            seen.add(key)
            unique.append(x)
    return unique[:PUMPING_MAX_MULTISTART_CANDIDATES]


def is_physically_usable_pumping_kpi(kpi, tether_force_max):
    if kpi.get("case_status") == "inactive_low_apparent_wind":
        return False
    if not kpi.get("sim_successful", False):
        return False
    data = kpi.get("objective_data", {})
    for key in ["P_cycle", "Fx_avg", "Fy_avg", "mean_tether_force", "max_tether_force", "P_prop_equiv", "P_equiv_pumping"]:
        if not np.isfinite(data.get(key, np.nan)):
            return False
    if float(data.get("P_cycle", 0.0)) < PUMPING_MIN_P_CYCLE_FOR_VALID_PUMPING:
        return False
    if PUMPING_REJECT_TRACTION_DOMINATED and bool(data.get("traction_dominated_flag", False)):
        return False
    if np.isfinite(tether_force_max) and float(data["max_tether_force"]) > tether_force_max + PUMPING_FORCE_TOLERANCE:
        return False
    return True


def build_inactive_low_apparent_wind_kpi(true_wind_speed, ship_speed, heading_deg, apparent_wind, min_apparent_wind_speed):
    return {
        "case_status": "inactive_low_apparent_wind",
        "sim_successful": False,
        "case_successful": True,
        "case_error_message": "inactive_low_apparent_wind",
        "average_power": {"cycle": 0.0, "in": 0.0, "trans_riro": 0.0, "trans_rori": 0.0, "out": 0.0},
        "duration": {"cycle": 0.0, "in": 0.0, "trans_riro": 0.0, "trans_rori": 0.0, "out": 0.0},
        "optimization_result": None,
        "objective_value": 0.0,
        "objective_name": "P_equiv_pumping",
        "objective_data": {
            "P_cycle": 0.0, "Fx_avg": 0.0, "Fy_avg": 0.0, "Fx_global_avg": 0.0, "Fy_global_avg": 0.0,
            "mean_tether_force": 0.0, "max_tether_force": 0.0, "P_prop_equiv": 0.0,
            "P_prop_penalty": 0.0, "P_equiv_pumping": 0.0, "n_force_samples": 0,
            **compute_pumping_quality_diagnostics(0.0, 0.0),
        },
        "true_wind_speed": float(true_wind_speed),
        "ship_speed": float(ship_speed),
        "heading_deg": float(heading_deg),
        "apparent_wind_speed": float(apparent_wind.speed),
        "min_apparent_wind_speed": float(min_apparent_wind_speed),
        "apparent_wind_direction_deg": float(np.rad2deg(apparent_wind.direction_to)),
        "last_var_names": [],
        "last_x_opt": None,
        "case_runtime_s": 0.0,
        "n_candidates_tried": 0,
        "selected_candidate_index": -1,
        "selected_candidate_runtime_s": 0.0,
        "max_candidate_runtime_s": 0.0,
        "mean_candidate_runtime_s": 0.0,
        "sum_candidate_runtime_s": 0.0,
    }


def optimize_one_pumping_heading_case(constructor, env_state, true_wind_speed, ship_speed, heading_deg, tether_force_max, min_apparent_wind_speed, warm_x0_base=None):
    case_start = time.perf_counter()
    apparent_wind, vessel_heading = compute_case_apparent_wind(true_wind_speed, ship_speed, heading_deg)

    if apparent_wind.speed < min_apparent_wind_speed:
        kpi = build_inactive_low_apparent_wind_kpi(true_wind_speed, ship_speed, heading_deg, apparent_wind, min_apparent_wind_speed)
        kpi["case_runtime_s"] = time.perf_counter() - case_start
        return kpi, warm_x0_base, []

    x0_candidates = build_pumping_x0_candidates(constructor.simulation_settings, apparent_wind.speed, warm_x0_base)
    results = []
    runtimes = []

    for idx, x0_base in enumerate(x0_candidates):
        t0 = time.perf_counter()
        optimizer = None
        try:
            settings = set_base_x0(constructor.simulation_settings, x0_base)
            optimizer = MovingVesselPumpingOptimizer(
                simulation_settings=settings,
                sys_props=constructor.sys_props,
                env_state=env_state,
                ship_speed=ship_speed,
                apparent_wind_direction=apparent_wind.direction_to,
                vessel_heading=vessel_heading,
            )
            if PUMPING_QUIET_NATIVE_OPTIMIZER_OUTPUT:
                with contextlib.redirect_stdout(io.StringIO()):
                    kpi = optimizer.optimize(wind_speed=apparent_wind.speed, verbose=PUMPING_OPTIMIZER_VERBOSE)
            else:
                kpi = optimizer.optimize(wind_speed=apparent_wind.speed, verbose=PUMPING_OPTIMIZER_VERBOSE)
            kpi["case_status"] = "evaluated"
            kpi["case_successful"] = True
            kpi["case_error_message"] = ""
        except Exception as exc:
            kpi = {
                "case_status": "exception",
                "sim_successful": False,
                "case_successful": False,
                "case_error_message": str(exc),
                "average_power": {"cycle": 0.0, "in": 0.0, "trans_riro": 0.0, "trans_rori": 0.0, "out": 0.0},
                "duration": {"cycle": 0.0, "in": 0.0, "trans_riro": 0.0, "trans_rori": 0.0, "out": 0.0},
                "optimization_result": None,
                "objective_value": np.nan,
                "objective_name": "P_equiv_pumping",
                "objective_data": {},
            }
        dt = time.perf_counter() - t0
        runtimes.append(float(dt))
        kpi.update({
            "candidate_index": int(idx),
            "candidate_runtime_s": float(dt),
            "true_wind_speed": float(true_wind_speed),
            "ship_speed": float(ship_speed),
            "heading_deg": float(heading_deg),
            "apparent_wind_speed": float(apparent_wind.speed),
            "min_apparent_wind_speed": float(min_apparent_wind_speed),
            "apparent_wind_direction_deg": float(np.rad2deg(apparent_wind.direction_to)),
            "last_var_names": getattr(optimizer, "last_var_names", []) if optimizer is not None else [],
            "last_x_opt": getattr(optimizer, "last_x_opt", None) if optimizer is not None else None,
        })
        results.append(kpi)

    case_runtime = time.perf_counter() - case_start
    for r in results:
        r["case_runtime_s"] = float(case_runtime)
        r["n_candidates_tried"] = len(results)
        r["max_candidate_runtime_s"] = float(max(runtimes)) if runtimes else 0.0
        r["mean_candidate_runtime_s"] = float(np.mean(runtimes)) if runtimes else 0.0
        r["sum_candidate_runtime_s"] = float(np.sum(runtimes)) if runtimes else 0.0

    feasible = [r for r in results if is_physically_usable_pumping_kpi(r, tether_force_max)]
    pool = feasible if feasible else [r for r in results if r.get("case_successful", False) and r.get("sim_successful", False) and np.isfinite(r.get("objective_value", np.nan))]
    best = max(pool, key=lambda r: r.get("objective_data", {}).get("P_equiv_pumping", r.get("objective_value", -np.inf))) if pool else results[0]
    best["selected_candidate_index"] = int(best.get("candidate_index", -1))
    best["selected_candidate_runtime_s"] = float(best.get("candidate_runtime_s", np.nan))

    candidate_rows = [build_pumping_candidate_log_row(r, x0_candidates[int(r.get("candidate_index", 0))], int(r.get("candidate_index", -1)) == best["selected_candidate_index"]) for r in results]

    new_warm_x0 = None
    if best.get("last_x_opt") is not None and best.get("last_var_names"):
        new_warm_x0 = active_solution_to_base_x0(get_base_x0(constructor.simulation_settings), best["last_var_names"], np.asarray(best["last_x_opt"], dtype=float))
    return best, new_warm_x0, candidate_rows


def build_pumping_output_row(kpi, tether_force_max, max_tether_length):
    data = kpi.get("objective_data", {})
    opt_result = kpi.get("optimization_result", None)
    case_status = kpi.get("case_status", "unknown")
    inactive = case_status == "inactive_low_apparent_wind"
    if inactive:
        optimizer_success = True
        optimizer_message = "inactive_low_apparent_wind"
    elif opt_result is None:
        optimizer_success = bool(kpi.get("sim_successful", False))
        optimizer_message = "No active optimization variables or optimization not run."
    else:
        optimizer_success = bool(opt_result.success)
        optimizer_message = str(opt_result.message)
    physical_feasible = is_physically_usable_pumping_kpi(kpi, tether_force_max)
    max_tether_force = data.get("max_tether_force", np.nan)
    if np.isfinite(tether_force_max) and np.isfinite(max_tether_force):
        tether_violation = max(0.0, max_tether_force - tether_force_max)
        physical_tether_ok = bool(max_tether_force <= tether_force_max + PUMPING_FORCE_TOLERANCE)
        constraint_active = bool(np.isclose(max_tether_force, tether_force_max, rtol=0.0, atol=25.0))
    else:
        tether_violation = 0.0
        physical_tether_ok = True
        constraint_active = False

    row = {
        "true_wind_speed": kpi.get("true_wind_speed", np.nan),
        "ship_speed": kpi.get("ship_speed", np.nan),
        "heading_deg": kpi.get("heading_deg", np.nan),
        "apparent_wind_speed": kpi.get("apparent_wind_speed", np.nan),
        "min_apparent_wind_speed": kpi.get("min_apparent_wind_speed", np.nan),
        "apparent_wind_direction_deg": kpi.get("apparent_wind_direction_deg", np.nan),
        "case_status": case_status,
        "inactive_low_apparent_wind": inactive,
        "case_successful": kpi.get("case_successful", False),
        "sim_successful": kpi.get("sim_successful", False),
        "optimizer_success": optimizer_success,
        "accepted_result": bool(physical_feasible),
        "physical_feasible": bool(physical_feasible),
        "accepted_despite_optimizer_message": bool(physical_feasible and not optimizer_success),
        "case_error_message": kpi.get("case_error_message", ""),
        "optimizer_message": optimizer_message,
        "case_runtime_s": kpi.get("case_runtime_s", np.nan),
        "n_candidates_tried": kpi.get("n_candidates_tried", np.nan),
        "selected_candidate_index": kpi.get("selected_candidate_index", np.nan),
        "selected_candidate_runtime_s": kpi.get("selected_candidate_runtime_s", np.nan),
        "max_candidate_runtime_s": kpi.get("max_candidate_runtime_s", np.nan),
        "mean_candidate_runtime_s": kpi.get("mean_candidate_runtime_s", np.nan),
        "sum_candidate_runtime_s": kpi.get("sum_candidate_runtime_s", np.nan),
        "P_cycle": data.get("P_cycle", np.nan),
        "P_in": kpi.get("average_power", {}).get("in", np.nan),
        "P_trans_riro": kpi.get("average_power", {}).get("trans_riro", np.nan),
        "P_trans_rori": kpi.get("average_power", {}).get("trans_rori", np.nan),
        "P_out": kpi.get("average_power", {}).get("out", np.nan),
        "cycle_duration": kpi.get("duration", {}).get("cycle", np.nan),
        "Fx_avg": data.get("Fx_avg", np.nan),
        "Fy_avg": data.get("Fy_avg", np.nan),
        "Fx_global_avg": data.get("Fx_global_avg", np.nan),
        "Fy_global_avg": data.get("Fy_global_avg", np.nan),
        "mean_tether_force": data.get("mean_tether_force", np.nan),
        "max_tether_force": data.get("max_tether_force", np.nan),
        "tether_force_max": tether_force_max,
        "tether_constraint_violation": tether_violation,
        "physical_tether_constraint_ok": physical_tether_ok,
        "constraint_active": constraint_active,
        "P_prop_equiv": data.get("P_prop_equiv", np.nan),
        "P_prop_penalty": data.get("P_prop_penalty", np.nan),
        "P_equiv_pumping": data.get("P_equiv_pumping", np.nan),
        "P_prop_abs": data.get("P_prop_abs", np.nan),
        "pump_ratio": data.get("pump_ratio", np.nan),
        "P_prop_fraction_abs": data.get("P_prop_fraction_abs", np.nan),
        "traction_dominated_flag": data.get("traction_dominated_flag", False),
        "n_force_samples": data.get("n_force_samples", np.nan),
        "max_tether_length": max_tether_length,
        "stroke_fraction": np.nan,
        "stroke_length_m": np.nan,
    }
    var_names = kpi.get("last_var_names", [])
    x_opt = kpi.get("last_x_opt", None)
    if x_opt is not None:
        for name, value in zip(var_names, x_opt):
            row[f"opt_{name}"] = float(value)
    if np.isfinite(row.get("opt_frac_start", np.nan)) and np.isfinite(row.get("opt_frac_end", np.nan)):
        row["stroke_fraction"] = float(row["opt_frac_start"] - row["opt_frac_end"])
        if np.isfinite(max_tether_length):
            row["stroke_length_m"] = row["stroke_fraction"] * max_tether_length
    return row


def build_pumping_candidate_log_row(kpi, x0_base, selected_best):
    data = kpi.get("objective_data", {})
    opt_result = kpi.get("optimization_result", None)
    row = {
        "true_wind_speed": kpi.get("true_wind_speed", np.nan),
        "ship_speed": kpi.get("ship_speed", np.nan),
        "heading_deg": kpi.get("heading_deg", np.nan),
        "apparent_wind_speed": kpi.get("apparent_wind_speed", np.nan),
        "apparent_wind_direction_deg": kpi.get("apparent_wind_direction_deg", np.nan),
        "candidate_index": kpi.get("candidate_index", np.nan),
        "selected_best": bool(selected_best),
        "candidate_runtime_s": kpi.get("candidate_runtime_s", np.nan),
        "case_runtime_s": kpi.get("case_runtime_s", np.nan),
        "case_status": kpi.get("case_status", "unknown"),
        "case_successful": kpi.get("case_successful", False),
        "sim_successful": kpi.get("sim_successful", False),
        "optimizer_success": bool(opt_result.success) if opt_result is not None else bool(kpi.get("sim_successful", False)),
        "optimizer_message": str(opt_result.message) if opt_result is not None else "No active optimization variables or optimization not run.",
        "case_error_message": kpi.get("case_error_message", ""),
        "objective_value": kpi.get("objective_value", np.nan),
        "P_cycle": data.get("P_cycle", np.nan),
        "P_prop_equiv": data.get("P_prop_equiv", np.nan),
        "P_equiv_pumping": data.get("P_equiv_pumping", np.nan),
        "P_prop_abs": data.get("P_prop_abs", np.nan),
        "pump_ratio": data.get("pump_ratio", np.nan),
        "P_prop_fraction_abs": data.get("P_prop_fraction_abs", np.nan),
        "traction_dominated_flag": data.get("traction_dominated_flag", False),
        "Fx_avg": data.get("Fx_avg", np.nan),
        "Fy_avg": data.get("Fy_avg", np.nan),
        "mean_tether_force": data.get("mean_tether_force", np.nan),
        "max_tether_force": data.get("max_tether_force", np.nan),
    }
    for i, value in enumerate(np.asarray(x0_base, dtype=float)):
        row[f"start_x0_{i}"] = float(value)
    x_opt = kpi.get("last_x_opt", None)
    if x_opt is not None:
        for name, value in zip(kpi.get("last_var_names", []), x_opt):
            row[f"opt_{name}"] = float(value)
    return row


def print_pumping_progress(wind_speed, i, total, heading_deg, accepted_count, inactive_count, failed_count, row, best_power):
    if row.get("accepted_result", False):
        p_cycle = row.get("P_cycle", np.nan)
        p_prop = row.get("P_prop_equiv", np.nan)
        p_penalty = row.get("P_prop_penalty", np.nan)
        p_equiv = row.get("P_equiv_pumping", np.nan)
        fx_avg = row.get("Fx_avg", np.nan)
        pump_ratio = row.get("pump_ratio", np.nan)
    else:
        p_cycle = p_prop = p_penalty = p_equiv = fx_avg = pump_ratio = np.nan
    msg = (
        f"  Vw={wind_speed:5.1f} m/s | {i:03d}/{total:03d} | ψ={heading_deg:7.2f} deg | "
        f"t={format_s(row.get('case_runtime_s', np.nan))}s | "
        f"cand={int(row.get('n_candidates_tried', 0)) if np.isfinite(row.get('n_candidates_tried', np.nan)) else 0:02d} | "
        f"tcand,max={format_s(row.get('max_candidate_runtime_s', np.nan))}s | "
        f"accepted={accepted_count:03d} | inactive={inactive_count:03d} | failed={failed_count:03d} | "
        f"Pcyc={format_kw_from_w(p_cycle)} kW | Pprop={format_kw_from_w(p_prop)} kW | "
        f"pen={format_kw_from_w(p_penalty)} kW | Peq={format_kw_from_w(p_equiv)} kW | "
        f"Fx={fx_avg:9.1f} N | ratio={pump_ratio:6.3f} | "
        f"tracdom={str(row.get('traction_dominated_flag', False)):5s} | best={format_kw_from_w(best_power)} kW"
    )
    sys.stdout.write("\r" + msg.ljust(360))
    sys.stdout.flush()


def run_pumping_sweep(constructor, env_state):
    optimizer_settings = constructor.simulation_settings["optimization"]["optimizer"]
    optimizer_settings["max_iterations"] = min(int(optimizer_settings.get("max_iterations", 200)), PUMPING_MAX_OPTIMIZER_ITERATIONS_DEBUG)
    constructor.simulation_settings["optimization"]["constraints"]["min_tether_length_fraction_difference"] = PUMPING_MIN_TETHER_LENGTH_FRACTION_DIFFERENCE

    min_apparent_wind_speed = calculate_static_takeoff_wind_speed(constructor.sys_props, env_state)
    physical_tether_force_max = get_tether_force_max(constructor)

    tether_force_max = (
        physical_tether_force_max
        if PUMPING_USE_TETHER_FORCE_LIMIT
        else PUMPING_DIAGNOSTIC_TETHER_FORCE_MAX
    )
    max_tether_length = get_max_tether_length(constructor)

    print("\nOPTIMIZED PUMPING MOVING-VESSEL WIND-HEADING SWEEP")
    print("--------------------------------------------------")
    print(f"True wind speeds      : {TRUE_WIND_SPEEDS}")
    print(f"Ship speed            : {SHIP_SPEED:.3f} m/s")
    print(f"Computed headings     : {SWEEP_HEADINGS[0]:.1f}–{SWEEP_HEADINGS[-1]:.1f} deg")
    print(f"Heading step          : {HEADING_STEP_DEG:.3f} deg")
    print(f"Mirrored for plot     : {MIRROR_RESULTS_FOR_PLOT}")
    print(f"Multistart candidates : {PUMPING_MAX_MULTISTART_CANDIDATES}")
    print(f"Max optimizer iters   : {optimizer_settings['max_iterations']}")
    print(f"Min apparent wind     : {min_apparent_wind_speed:.3f} m/s")
    print(f"Min stroke fraction   : {PUMPING_MIN_TETHER_LENGTH_FRACTION_DIFFERENCE:.3f}")
    print(f"Tether force max      : {tether_force_max:.3f} N")
    print("Objective             : maximize P_equiv_pumping = P_cycle + Fx_avg * V_ship")

    all_rows = []
    all_candidate_rows = []

    for wind_speed in TRUE_WIND_SPEEDS:
        print(f"\nRunning optimized pumping sweep: true wind = {wind_speed:.1f} m/s, ship speed = {SHIP_SPEED:.1f} m/s")
        warm_x0_base = None
        accepted_count = inactive_count = failed_count = 0
        best_power = -np.inf
        for i, heading_deg in enumerate(SWEEP_HEADINGS, start=1):
            kpi, new_warm, case_candidates = optimize_one_pumping_heading_case(
                constructor, env_state, wind_speed, SHIP_SPEED, heading_deg, tether_force_max, min_apparent_wind_speed, warm_x0_base
            )
            row = build_pumping_output_row(kpi, tether_force_max, max_tether_length)
            all_rows.append(row)
            all_candidate_rows.extend(case_candidates)
            if row.get("accepted_result", False):
                accepted_count += 1
                warm_x0_base = new_warm
                best_power = max(best_power, row.get("P_equiv_pumping", -np.inf))
            elif row.get("inactive_low_apparent_wind", False):
                inactive_count += 1
            else:
                failed_count += 1
            if SAVE_CSV:
                write_csv(all_rows, PUMPING_AUTOSAVE_CSV_PATH, quiet=True)
            print_pumping_progress(wind_speed, i, len(SWEEP_HEADINGS), heading_deg, accepted_count, inactive_count, failed_count, row, best_power)
        print("")

    if SAVE_CSV:
        write_csv(all_rows, PUMPING_CSV_PATH)
    if SAVE_CANDIDATE_LOGS:
        write_csv(all_candidate_rows, PUMPING_CANDIDATE_LOG_PATH)
    print_pumping_summary(all_rows)
    return all_rows, all_candidate_rows


def print_pumping_summary(rows):
    print("\nPUMPING SUMMARY")
    print("---------------")
    for wind_speed in sorted(set(float(r["true_wind_speed"]) for r in rows)):
        group = [r for r in rows if np.isclose(float(r["true_wind_speed"]), wind_speed)]
        accepted = [r for r in group if r.get("accepted_result", False)]
        inactive = [r for r in group if r.get("inactive_low_apparent_wind", False)]
        failed = [r for r in group if not r.get("accepted_result", False) and not r.get("inactive_low_apparent_wind", False)]
        if accepted:
            best = max(accepted, key=lambda r: r.get("P_equiv_pumping", -np.inf))
            print(f"Vw={wind_speed:5.1f} m/s | accepted={len(accepted):02d} | inactive={len(inactive):02d} | failed={len(failed):02d} | best Peq={best['P_equiv_pumping']/1000.0:9.3f} kW at ψ={best['heading_deg']:7.2f} deg")
        else:
            print(f"Vw={wind_speed:5.1f} m/s | accepted=00 | inactive={len(inactive):02d} | failed={len(failed):02d} | best Peq=nan")


# =============================================================================
# Common-format extraction and overlay comparison
# =============================================================================


def extract_traction_common_rows(rows):
    out = []
    for r in rows:
        P = safe_float(r.get("optimized_P_equiv_traction"))
        out.append({
            "mode": "traction",
            "true_wind_speed": safe_float(r.get("true_wind_speed")),
            "ship_speed": safe_float(r.get("ship_speed")),
            "heading_deg": safe_float(r.get("heading_deg")),
            "accepted_result": parse_bool(r.get("accepted_result", False)),
            "P_equiv_W": P,
            "P_equiv_kW": P / 1000.0 if np.isfinite(P) else np.nan,
            "Fx_or_Fx_avg": safe_float(r.get("optimized_Fx")),
            "Fy_or_Fy_avg": safe_float(r.get("optimized_Fy")),
        })
    return out


def extract_pumping_common_rows(rows):
    out = []
    for r in rows:
        P = safe_float(r.get("P_equiv_pumping"))
        Pcyc = safe_float(r.get("P_cycle"))
        Pprop = safe_float(r.get("P_prop_equiv"))
        out.append({
            "mode": "pumping",
            "true_wind_speed": safe_float(r.get("true_wind_speed")),
            "ship_speed": safe_float(r.get("ship_speed")),
            "heading_deg": safe_float(r.get("heading_deg")),
            "accepted_result": parse_bool(r.get("accepted_result", False)),
            "case_status": r.get("case_status", ""),
            "P_equiv_W": P,
            "P_equiv_kW": P / 1000.0 if np.isfinite(P) else np.nan,
            "P_cycle_W": Pcyc,
            "P_cycle_kW": Pcyc / 1000.0 if np.isfinite(Pcyc) else np.nan,
            "P_prop_equiv_W": Pprop,
            "P_prop_equiv_kW": Pprop / 1000.0 if np.isfinite(Pprop) else np.nan,
            "Fx_or_Fx_avg": safe_float(r.get("Fx_avg")),
            "Fy_or_Fy_avg": safe_float(r.get("Fy_avg")),
        })
    return out


def filter_to_common_points(traction_rows, pumping_rows):
    t_points = {(round(r["true_wind_speed"], 6), round(r["heading_deg"], 6)) for r in traction_rows if np.isfinite(r["true_wind_speed"]) and np.isfinite(r["heading_deg"])}
    p_points = {(round(r["true_wind_speed"], 6), round(r["heading_deg"], 6)) for r in pumping_rows if np.isfinite(r["true_wind_speed"]) and np.isfinite(r["heading_deg"])}
    common = t_points.intersection(p_points)
    return (
        [r for r in traction_rows if (round(r["true_wind_speed"], 6), round(r["heading_deg"], 6)) in common],
        [r for r in pumping_rows if (round(r["true_wind_speed"], 6), round(r["heading_deg"], 6)) in common],
    )


def build_comparison_summary(traction_rows, pumping_rows):
    summary = []
    for wind_speed in TRUE_WIND_SPEEDS:
        tg = [r for r in traction_rows if np.isclose(r["true_wind_speed"], wind_speed) and r.get("accepted_result", False) and np.isfinite(r.get("P_equiv_kW", np.nan))]
        pg = [r for r in pumping_rows if np.isclose(r["true_wind_speed"], wind_speed) and r.get("accepted_result", False) and np.isfinite(r.get("P_equiv_kW", np.nan))]
        bt = max(tg, key=lambda r: r["P_equiv_kW"]) if tg else None
        bp = max(pg, key=lambda r: r["P_equiv_kW"]) if pg else None
        bt_kw = bt["P_equiv_kW"] if bt else np.nan
        bp_kw = bp["P_equiv_kW"] if bp else np.nan
        if np.isfinite(bt_kw) and np.isfinite(bp_kw):
            best_mode = "pumping" if bp_kw > bt_kw else "traction"
            delta = bp_kw - bt_kw
        else:
            best_mode = "unavailable"
            delta = np.nan
        summary.append({
            "true_wind_speed": float(wind_speed),
            "ship_speed": SHIP_SPEED,
            "n_traction_rows": len(tg),
            "n_pumping_rows": len(pg),
            "best_traction_kW": bt_kw,
            "best_traction_heading_deg": bt["heading_deg"] if bt else np.nan,
            "best_pumping_kW": bp_kw,
            "best_pumping_heading_deg": bp["heading_deg"] if bp else np.nan,
            "delta_best_pumping_minus_traction_kW": delta,
            "best_mode_by_peak_value": best_mode,
        })
    return summary


def print_comparison_summary(summary):
    print("\nCOMPARISON SUMMARY")
    print("------------------")
    for r in summary:
        print(
            f"Vw={r['true_wind_speed']:5.1f} m/s | "
            f"best traction={r['best_traction_kW']:8.3f} kW at ψ={r['best_traction_heading_deg']:6.1f}° | "
            f"best pumping={r['best_pumping_kW']:8.3f} kW at ψ={r['best_pumping_heading_deg']:6.1f}° | "
            f"best mode={r['best_mode_by_peak_value']}"
        )


# =============================================================================
# Plotting
# =============================================================================


def build_polar_points_from_common(rows, mirror=True):
    point_map = {}
    for r in rows:
        heading = safe_float(r.get("heading_deg"))
        if not np.isfinite(heading):
            continue
        heading = heading % 360.0
        P_w = safe_float(r.get("P_equiv_W"))
        radius = result_radius_kw(P_w, r.get("accepted_result", False))
        point_map[heading] = max(point_map.get(heading, 0.0), radius)
        if mirror and 0.0 < heading < 180.0:
            point_map[360.0 - heading] = max(point_map.get(360.0 - heading, 0.0), radius)
    if not point_map:
        return np.array([]), np.array([])
    headings = np.array(sorted(point_map.keys()), dtype=float)
    radii = np.array([point_map[h] for h in headings], dtype=float)
    headings_closed = np.concatenate([headings, [headings[0] + 360.0]])
    radii_closed = np.concatenate([radii, [radii[0]]])
    return np.deg2rad(headings_closed), radii_closed


def finalize_figure(fig, output_path: Path | None):
    if SAVE_FIGURES and output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure:\n  {output_path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_single_mode_polar(common_rows, mode_name, output_path, force_limit_text=""):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    max_radius = 0.0
    for wind_speed in TRUE_WIND_SPEEDS:
        group = [r for r in common_rows if np.isclose(r["true_wind_speed"], wind_speed)]
        theta, radius = build_polar_points_from_common(group, mirror=MIRROR_RESULTS_FOR_PLOT)
        if len(theta) == 0:
            continue
        ax.plot(theta, radius, marker="o", markersize=3, linewidth=1.8, label=f"{wind_speed:.0f} m/s")
        max_radius = max(max_radius, float(np.nanmax(radius)))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.10 * max_radius if max_radius > 0.0 else 1.0)
    ax.set_rlabel_position(135)
    mirror_text = "0–180° computed, 180–360° mirrored for plotting only" if MIRROR_RESULTS_FOR_PLOT else "0–360° computed directly"
    ax.set_title(f"Positive equivalent {mode_name} power\n"f"Ship speed = {SHIP_SPEED:.1f} m/s\n"f"{force_limit_text}\n" f"{mirror_text}",pad=28)
    ax.text(np.deg2rad(135), 1.05 * max_radius if max_radius > 0.0 else 0.90, f"P_equiv_{mode_name} [kW]", ha="center", va="center")
    ax.legend(title="True wind speed", loc="upper right", bbox_to_anchor=(1.35, 1.10))
    ax.grid(True)
    finalize_figure(fig, output_path)


def plot_traction_operating_angles(rows):
    accepted = [r for r in rows if r.get("accepted_result", False)]
    if not accepted:
        return
    diagnostic_wind_speed = float(TRUE_WIND_SPEEDS[-1])
    group = [r for r in accepted if np.isclose(float(r["true_wind_speed"]), diagnostic_wind_speed)]
    if not group:
        return
    group = sorted(group, key=lambda r: r["heading_deg"])
    headings = np.array([r["heading_deg"] for r in group], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(headings, [r["optimized_azimuth_angle_deg"] for r in group], linewidth=1.8, label="azimuth angle")
    ax.plot(headings, [r["optimized_elevation_angle_deg"] for r in group], linewidth=1.8, label="elevation angle")
    ax.plot(headings, [r["optimized_course_angle_deg"] for r in group], linewidth=1.8, label="course angle")
    ax.set_xlabel("Ship heading ψ [deg]")
    ax.set_ylabel("Optimized angle [deg]")
    ax.set_title(f"Optimized traction operating angles\nTrue wind = {diagnostic_wind_speed:.1f} m/s, ship speed = {SHIP_SPEED:.1f} m/s")
    ax.legend()
    ax.grid(True)
    finalize_figure(fig, TRACTION_ANGLES_FIG_PATH)


def plot_overlay_polar(traction_common,pumping_common,traction_force_limit_text="",pumping_force_limit_text="",):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    max_radius = 0.0
    for wind_speed in TRUE_WIND_SPEEDS:
        tg = [r for r in traction_common if np.isclose(r["true_wind_speed"], wind_speed)]
        pg = [r for r in pumping_common if np.isclose(r["true_wind_speed"], wind_speed)]
        theta_t, radius_t = build_polar_points_from_common(tg, mirror=MIRROR_RESULTS_FOR_PLOT)
        theta_p, radius_p = build_polar_points_from_common(pg, mirror=MIRROR_RESULTS_FOR_PLOT)
        line = None
        if len(theta_t) > 0:
            line, = ax.plot(theta_t, radius_t, linestyle="-", marker="o", markersize=3, linewidth=2.0, label=f"Traction, {wind_speed:.0f} m/s")
            max_radius = max(max_radius, float(np.nanmax(radius_t)))
        if len(theta_p) > 0:
            ax.plot(theta_p, radius_p, linestyle="--", marker="o", markersize=3, linewidth=2.0, color=line.get_color() if line else None, label=f"Pumping, {wind_speed:.0f} m/s")
            max_radius = max(max_radius, float(np.nanmax(radius_p)))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.10 * max_radius if max_radius > 0.0 else 1.0)
    ax.set_rlabel_position(135)
    mirror_text = "0–180° computed, 180–360° mirrored for plotting only" if MIRROR_RESULTS_FOR_PLOT else "CSV headings plotted directly"
    heading_text = "common headings only" if USE_COMMON_HEADINGS_ONLY_FOR_OVERLAY else "all available headings"
    ax.set_title( f"Prescribed-motion equivalent power comparison: traction vs pumping\n" f"Ship speed = {SHIP_SPEED:.1f} m/s, {heading_text}\n" f"{traction_force_limit_text}\n" f"{pumping_force_limit_text}\n" f"{mirror_text}",pad=28,)
    ax.text(np.deg2rad(135), 1.05 * max_radius if max_radius > 0.0 else 0.90, "Positive P_equiv [kW]", ha="center", va="center")
    ax.legend(title="Mode and true wind speed", loc="upper right", bbox_to_anchor=(1.42, 1.12))
    ax.grid(True)
    finalize_figure(fig, OVERLAY_POLAR_FIG_PATH)


# =============================================================================
# Main
# =============================================================================


def main():
    script_start = time.perf_counter()

    print("\nCOMBINED MOVING-VESSEL AWE MODE SWEEP")
    print("-------------------------------------")
    print(f"Run traction          : {RUN_TRACTION}")
    print(f"Run pumping           : {RUN_PUMPING}")
    print(f"Run overlay           : {RUN_OVERLAY}")
    print(f"True wind speeds      : {TRUE_WIND_SPEEDS}")
    print(f"Ship speed            : {SHIP_SPEED:.3f} m/s")
    print(f"Headings              : {SWEEP_HEADINGS[0]:.1f}–{SWEEP_HEADINGS[-1]:.1f} deg, step {HEADING_STEP_DEG:.1f} deg")
    print(f"Results directory     : {RESULTS_DIR}")

    constructor, env_state = create_constructor_and_environment()
    physical_tether_force_max = get_tether_force_max(constructor)

    traction_active_tether_force_max = (
        physical_tether_force_max
        if TRACTION_USE_TETHER_FORCE_LIMIT
        else TRACTION_DIAGNOSTIC_TETHER_FORCE_MAX
    )

    pumping_active_tether_force_max = (
        physical_tether_force_max
        if PUMPING_USE_TETHER_FORCE_LIMIT
        else PUMPING_DIAGNOSTIC_TETHER_FORCE_MAX
    )

    traction_force_limit_text = build_force_limit_text(
        physical_tether_force_max=physical_tether_force_max,
        active_tether_force_max=traction_active_tether_force_max,
        force_limit_enabled=TRACTION_USE_TETHER_FORCE_LIMIT,
        mode_name="Traction",
    )      

    pumping_force_limit_text = build_force_limit_text(
        physical_tether_force_max=physical_tether_force_max,
        active_tether_force_max=pumping_active_tether_force_max,
        force_limit_enabled=PUMPING_USE_TETHER_FORCE_LIMIT,
        mode_name="Pumping",
    )
    traction_rows = []
    pumping_rows = []

    if RUN_TRACTION:
        traction_rows, _ = run_traction_sweep(constructor, env_state)

    # Recreate constructor before pumping so traction-side solver limit changes/settings do not leak.
    if RUN_PUMPING:
        constructor, env_state = create_constructor_and_environment()
        pumping_rows, _ = run_pumping_sweep(constructor, env_state)

    traction_common = extract_traction_common_rows(traction_rows) if traction_rows else []
    pumping_common = extract_pumping_common_rows(pumping_rows) if pumping_rows else []

    if RUN_OVERLAY:
        if not traction_common or not pumping_common:
            print("\nOverlay skipped because one of the mode result sets is empty. Enable both RUN_TRACTION and RUN_PUMPING, or load rows from CSV manually.")
        else:
            if USE_COMMON_HEADINGS_ONLY_FOR_OVERLAY:
                traction_common, pumping_common = filter_to_common_points(traction_common, pumping_common)
            common_rows = traction_common + pumping_common
            summary_rows = build_comparison_summary(traction_common, pumping_common)
            print_comparison_summary(summary_rows)
            if SAVE_CSV:
                write_csv(common_rows, COMMON_FORMAT_CSV_PATH)
                write_csv(summary_rows, COMPARISON_SUMMARY_CSV_PATH)

    if PLOT_TRACTION_POLAR and traction_common:
        plot_single_mode_polar(
            traction_common,
            "traction",
            TRACTION_POLAR_FIG_PATH,
            force_limit_text=traction_force_limit_text,
        )
    if PLOT_TRACTION_OPERATING_ANGLES and traction_rows:
        plot_traction_operating_angles(traction_rows)
    if PLOT_PUMPING_POLAR and pumping_common:
        plot_single_mode_polar(
            pumping_common,
            "pumping",
            PUMPING_POLAR_FIG_PATH,
            force_limit_text=pumping_force_limit_text,
        )
    if PLOT_OVERLAY_POLAR and RUN_OVERLAY and traction_common and pumping_common:
        plot_overlay_polar(traction_common,pumping_common,traction_force_limit_text=traction_force_limit_text, pumping_force_limit_text=pumping_force_limit_text,)

    runtime = time.perf_counter() - script_start
    print(f"\nDone. Wall-clock runtime: {runtime:.1f} s ({runtime / 60.0:.2f} min)")


if __name__ == "__main__":
    main()
