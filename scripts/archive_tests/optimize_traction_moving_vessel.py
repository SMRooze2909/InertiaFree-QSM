#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constrained traction optimization heading sweep with moving-vessel apparent wind.

Objective:
    maximize P_equiv_traction = Fx_ship * V_ship

where:
    Fx_ship > 0 helps propulsion
    Fx_ship < 0 adds resistance, but negative-surge solutions are rejected

Decision variables:
    x = [azimuth_angle, elevation_angle, course_angle]

All optimizer variables are handled in degrees.

Physical convention:
- QSM solve is performed in the apparent-wind-aligned frame.
- QSM x-axis is treated as aligned with apparent wind direction.
- Coordinate transformation is then applied:
      QSM frame -> global frame -> ship frame
- Tether force limit is enforced in the final optimized result.
- During optimizer trial evaluations, the internal solver tether limit is
  temporarily disabled so SLSQP can evaluate infeasible points and enforce
  the tether-force constraint itself.

Plotting:
- Only headings from 0 to 180 deg are computed.
- If MIRROR_POLAR_PLOT = True, headings from 180 to 360 deg are mirrored for
  plotting only, assuming port/starboard vessel symmetry.
- The only plot produced is the polar plot of positive P_equiv_traction [kW].
"""

import sys
from pathlib import Path
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op


# =============================================================================
# Path setup
# =============================================================================

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# Project imports
# =============================================================================

from inertiafree_qsm import PowerCurveConstructor
from inertiafree_qsm.operating_point import VesselState, WindCondition
from inertiafree_qsm.pure_traction import PureTractionInput, PureTractionSolver
from inertiafree_qsm.coordinate_transforms import qsm_kite_position_to_ship_force
from inertiafree_qsm.vessel_coupling import (
    TrueWind,
    VesselMotion,
    compute_apparent_wind,
)


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_20.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings.yml"

CSV_OUTPUT_PATH = RESULTS_DIR / "optimized_traction_heading_wind_sweep.csv"
CSV_AUTOSAVE_PATH = RESULTS_DIR / "optimized_traction_heading_wind_sweep_autosave.csv"
CSV_CANDIDATE_LOG_PATH = RESULTS_DIR / "optimized_traction_candidate_log.csv"


# =============================================================================
# Sweep settings
# =============================================================================

TRUE_WIND_SPEEDS = np.array([18.0], dtype=float)
#TRUE_WIND_SPEEDS = np.array([10.0, 14.0, 18.0], dtype=float)
SHIP_SPEED = 8.0
CLUSTER_ID = 1
TETHER_LENGTH = 500.0

# =============================================================================
# Mirroring switch (RUNTIME SAVING PURPOSES)
# =============================================================================

MIRROR_RESULTS_FOR_PLOT = True
# True  = compute only 0–180 deg, mirror 180–360 deg for polar plotting
# False = compute full 0–360 deg, no mirroring

HEADING_STEP_DEG = 15.0

if MIRROR_RESULTS_FOR_PLOT:
    SWEEP_HEADINGS = np.arange(0.0, 181.0, HEADING_STEP_DEG)
else:
    SWEEP_HEADINGS = np.arange(0.0, 360.0, HEADING_STEP_DEG)

# If True, SciPy/SLSQP prints detailed optimizer output for every candidate start.
# Keep False for normal heading sweeps to avoid cluttering the console.
OPTIMIZER_VERBOSE = False

# None = use all starts. For faster testing use e.g. 20.
MAX_CANDIDATE_STARTS = 25

# Save partial results while running.
AUTOSAVE_AFTER_EACH_CASE = True
AUTOSAVE_CANDIDATE_LOG_AFTER_EACH_CASE = False

# Diagnostic plot wind speed.
DIAGNOSTIC_WIND_SPEED = 18.0
PLOT_OPERATING_ANGLES = True

# SLSQP settings.
MAX_OPTIMIZER_ITERATIONS = 800
OPTIMIZER_FTOL = 1.0e-9
OPTIMIZER_EPS = 5.0e-4

# =============================================================================
# Diagnostic force-limit switch
# =============================================================================

USE_TETHER_FORCE_LIMIT = True

# Use a plain float, not scientific notation in YAML.
DIAGNOSTIC_TETHER_FORCE_MAX = 1000000000.0

# =============================================================================
# Optimizer settings
# =============================================================================

INITIAL_GUESS_DEG = np.array(
    [
        11.5,   # azimuth angle [deg]
        30.0,   # elevation angle [deg]
        93.0,   # course angle [deg]
    ],
    dtype=float,
)

BOUNDS_DEG = [
    (-90.0, 90.0),       # azimuth angle [deg]
    (10.0, 80.0),        # elevation angle [deg]
    (-180.0, 180.0),     # course angle [deg]
]

SCALING = np.array(
    [
        30.0,
        30.0,
        180.0,
    ],
    dtype=float,
)

FORCE_TOLERANCE = 1.0
FORCE_SAFETY_MARGIN = 5.0


# =============================================================================
# Evaluation helpers
# =============================================================================

def solve_traction_with_measured_tether_force(
    solver,
    wind,
    traction_input,
    vessel_state,
):
    """
    Run the traction solver while allowing tether-force-limit violations
    to be evaluated.

    The real tether limit is restored immediately afterwards and is enforced
    by the optimizer constraint.
    """

    old_limit = getattr(solver.sys_props, "tether_force_max_limit", None)

    if old_limit is None:
        raise ValueError(
            "sys_props.tether_force_max_limit is not defined. "
            "Cannot run constrained traction optimization."
        )

    solver.sys_props.tether_force_max_limit = 1.0e12

    try:
        result = solver.solve(
            wind=wind,
            traction_input=traction_input,
            vessel_state=vessel_state,
        )
    finally:
        solver.sys_props.tether_force_max_limit = old_limit

    return result


def evaluate_traction_operating_point(
    solver,
    true_wind_speed,
    ship_speed,
    heading_deg,
    x_deg,
):
    """
    Evaluate one traction operating point.

    x_deg:
        [azimuth_angle_deg, elevation_angle_deg, course_angle_deg]
    """

    x_deg = np.asarray(x_deg, dtype=float)

    azimuth_deg = x_deg[0]
    elevation_deg = x_deg[1]
    course_deg = x_deg[2]

    true_wind = TrueWind(
        speed=float(true_wind_speed),
        direction_to=np.deg2rad(0.0),
    )

    vessel_heading = np.deg2rad(float(heading_deg))

    vessel_motion = VesselMotion(
        speed=float(ship_speed),
        heading=vessel_heading,
    )

    apparent_wind = compute_apparent_wind(true_wind, vessel_motion)

    traction_input = PureTractionInput(
        tether_length=TETHER_LENGTH,
        azimuth_angle=np.deg2rad(azimuth_deg),
        elevation_angle=np.deg2rad(elevation_deg),
        course_angle=np.deg2rad(course_deg),
    )

    # QSM is run in apparent-wind-aligned frame.
    wind_qsm = WindCondition(
        speed=apparent_wind.speed,
        direction=0.0,
    )

    vessel_state_qsm = VesselState(
        speed=float(ship_speed),
        leeway_angle=0.0,
        heading=0.0,
    )

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

    P_equiv_traction = Fx_ship * ship_speed

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
        "P_equiv_traction": float(P_equiv_traction),
        "P_equiv_traction_kW": float(P_equiv_traction / 1000.0),
    }


def is_physically_feasible_result(result, tether_force_max):
    """
    Check whether a result is physically usable for this constrained problem.

    This deliberately does not require optimizer_success=True, because SLSQP can
    return "iteration limit reached" even when it has reached a good feasible
    boundary point.
    """

    if not result.get("evaluation_success", False):
        return False

    P_equiv = result.get("P_equiv_traction", np.nan)
    Fx = result.get("Fx", np.nan)
    tether_force = result.get("tether_force_ground", np.nan)

    if not np.isfinite(P_equiv):
        return False

    if not np.isfinite(Fx):
        return False

    if not np.isfinite(tether_force):
        return False

    positive_surge_ok = Fx >= -1.0e-6

    tether_ok = (
        tether_force
        <= tether_force_max - FORCE_SAFETY_MARGIN + FORCE_TOLERANCE
    )

    return positive_surge_ok and tether_ok

def add_projection_diagnostics(result):
    """
    Add Fx/T and projection loss angle diagnostics.
    """

    Fx = result.get("Fx", np.nan)
    T = result.get("tether_force_ground", np.nan)

    if np.isfinite(Fx) and np.isfinite(T) and T > 0.0:
        ratio = Fx / T
        ratio_clipped = np.clip(ratio, -1.0, 1.0)
        projection_loss_angle_deg = np.rad2deg(np.arccos(ratio_clipped))
    else:
        ratio = np.nan
        projection_loss_angle_deg = np.nan

    result["Fx_over_tether_force"] = float(ratio)
    result["projection_loss_angle_deg"] = float(projection_loss_angle_deg)

    return result


def build_force_limit_plot_label(
    physical_tether_force_max: float,
    optimizer_tether_force_max: float,
) -> str:
    """
    Text label for plot titles showing whether the force limit is active.
    """

    if USE_TETHER_FORCE_LIMIT:
        return (
            f"Force limit ON: "
            f"T ≤ {physical_tether_force_max / 1000.0:.2f} kN"
        )

    return (
        f"Force limit OFF diagnostic: "
        f"physical limit = {physical_tether_force_max / 1000.0:.2f} kN, "
        f"optimizer limit = {optimizer_tether_force_max / 1000.0:.0f} kN"
    )

def format_s(value_s):
    if not np.isfinite(value_s):
        return "   nan"
    return f"{value_s:6.1f}"

# =============================================================================
# Optimizer
# =============================================================================

class TractionOptimizer:
    def __init__(
        self,
        solver,
        true_wind_speed,
        ship_speed,
        heading_deg,
        tether_force_max,
    ):
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

        if (
            self._cache_x_scaled is not None
            and np.allclose(
                x_scaled,
                self._cache_x_scaled,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            return self._cache_result

        x_deg = x_scaled * SCALING

        try:
            result = evaluate_traction_operating_point(
                solver=self.solver,
                true_wind_speed=self.true_wind_speed,
                ship_speed=self.ship_speed,
                heading_deg=self.heading_deg,
                x_deg=x_deg,
            )

        except Exception as exc:
            result = {
                "evaluation_success": False,
                "error_message": str(exc),
                "azimuth_angle_deg": float(x_deg[0]),
                "elevation_angle_deg": float(x_deg[1]),
                "course_angle_deg": float(x_deg[2]),
                "true_wind_speed": float(self.true_wind_speed),
                "ship_speed": float(self.ship_speed),
                "heading_deg": float(self.heading_deg),
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
        """
        SLSQP minimizes, so return -P_equiv_traction.

        Objective is scaled from W to kW for better numerical conditioning.
        """

        result = self._evaluate_scaled(x_scaled)

        if not result["evaluation_success"]:
            return 1.0e12

        P_equiv = result["P_equiv_traction"]

        if not np.isfinite(P_equiv):
            return 1.0e12

        Fx = result["Fx"]
        tether_force = result["tether_force_ground"]

        feasible = (
            Fx >= 0.0
            and tether_force <= (
                self.tether_force_max
                - FORCE_SAFETY_MARGIN
                + FORCE_TOLERANCE
            )
        )

        self.history.append(
            {
                "x_deg": x_scaled * SCALING,
                "Fx": Fx,
                "Fy": result["Fy"],
                "tether_force_ground": tether_force,
                "P_equiv_traction": P_equiv,
                "P_equiv_traction_kW": P_equiv / 1000.0,
                "feasible": feasible,
            }
        )

        return -P_equiv / 1000.0

    def constraint_positive_surge(self, x_scaled):
        """
        Dimensionless constraint:
            Fx / tether_force_max >= 0
        """

        result = self._evaluate_scaled(x_scaled)

        if not result["evaluation_success"]:
            return -1.0

        return result["Fx"] / self.tether_force_max

    def constraint_tether_force(self, x_scaled):
        """
        Dimensionless constraint:
            (tether_force_max - safety_margin - tether_force) / tether_force_max >= 0
        """

        result = self._evaluate_scaled(x_scaled)

        if not result["evaluation_success"]:
            return -1.0

        return (
            self.tether_force_max
            - FORCE_SAFETY_MARGIN
            - result["tether_force_ground"]
        ) / self.tether_force_max

    def optimize(self, x0_deg, verbose=False):
        x0_deg = np.asarray(x0_deg, dtype=float)
        x0_scaled = x0_deg / SCALING

        bounds_scaled = [
            (lo / scale, hi / scale)
            for (lo, hi), scale in zip(BOUNDS_DEG, SCALING)
        ]

        constraints = [
            {
                "type": "ineq",
                "fun": self.constraint_positive_surge,
            },
            {
                "type": "ineq",
                "fun": self.constraint_tether_force,
            },
        ]

        opt_result = op.minimize(
            fun=self.objective,
            x0=x0_scaled,
            method="SLSQP",
            bounds=bounds_scaled,
            constraints=constraints,
            options={
                "maxiter": MAX_OPTIMIZER_ITERATIONS,
                "ftol": OPTIMIZER_FTOL,
                "eps": OPTIMIZER_EPS,
                "disp": verbose,
            },
        )

        final_result = self._evaluate_scaled(opt_result.x)

        physical_tether_violation = max(
            0.0,
            final_result["tether_force_ground"] - self.tether_force_max,
        )

        optimizer_tether_violation = max(
            0.0,
            final_result["tether_force_ground"]
            - (self.tether_force_max - FORCE_SAFETY_MARGIN),
        )

        final_result["optimizer_success"] = bool(opt_result.success)
        final_result["optimizer_message"] = str(opt_result.message)

        final_result["positive_surge_ok"] = bool(final_result["Fx"] >= 0.0)

        final_result["tether_constraint_violation"] = float(physical_tether_violation)

        final_result["optimizer_tether_constraint_violation"] = float(
            optimizer_tether_violation
        )

        final_result["physical_tether_constraint_ok"] = bool(
            final_result["tether_force_ground"]
            <= self.tether_force_max + FORCE_TOLERANCE
        )

        final_result["tether_constraint_ok"] = bool(
            final_result["tether_force_ground"]
            <= self.tether_force_max - FORCE_SAFETY_MARGIN + FORCE_TOLERANCE
        )

        final_result["accepted_result"] = bool(
            final_result["evaluation_success"]
            and final_result["positive_surge_ok"]
            and final_result["tether_constraint_ok"]
        )

        final_result["accepted_despite_optimizer_message"] = bool(
            final_result["accepted_result"]
            and not final_result["optimizer_success"]
        )

        final_result["constraint_active"] = bool(
            np.isclose(
                final_result["tether_force_ground"],
                self.tether_force_max - FORCE_SAFETY_MARGIN,
                rtol=0.0,
                atol=25.0,
            )
        )

        final_result["x_opt_deg"] = np.array(
            [
                final_result["azimuth_angle_deg"],
                final_result["elevation_angle_deg"],
                final_result["course_angle_deg"],
            ],
            dtype=float,
        )
        final_result = add_projection_diagnostics(final_result)
        return final_result


# =============================================================================
# Multistart helpers
# =============================================================================

def unique_candidate_starts(candidate_starts, decimals=6):
    """
    Remove duplicate optimizer starts.
    """

    unique = []
    seen = set()

    for x in candidate_starts:
        x = np.asarray(x, dtype=float)
        key = tuple(np.round(x, decimals=decimals))

        if key not in seen:
            seen.add(key)
            unique.append(x)

    return unique


def build_candidate_starts(warm_start_deg=None):
    """
    Build physically diverse starting points.

    Includes:
    - crosswind-like guesses
    - port/starboard guesses
    - low-elevation / near-zero-course traction branch
    - high-side / force-limited branch
    """

    candidate_starts = []

    if warm_start_deg is not None:
        candidate_starts.append(np.asarray(warm_start_deg, dtype=float))

    candidate_starts.extend(
        [
            INITIAL_GUESS_DEG,
            np.array(
                [
                    -INITIAL_GUESS_DEG[0],
                    INITIAL_GUESS_DEG[1],
                    -INITIAL_GUESS_DEG[2],
                ],
                dtype=float,
            ),

            # Crosswind-like guesses
            np.array([0.0, 10.0, 90.0], dtype=float),
            np.array([0.0, 30.0, 90.0], dtype=float),
            np.array([0.0, 45.0, 90.0], dtype=float),
            np.array([0.0, 10.0, -90.0], dtype=float),
            np.array([0.0, 30.0, -90.0], dtype=float),
            np.array([0.0, 45.0, -90.0], dtype=float),

            # Explicit port/starboard guesses
            np.array([30.0, 20.0, 90.0], dtype=float),
            np.array([-30.0, 20.0, -90.0], dtype=float),
            np.array([60.0, 30.0, 90.0], dtype=float),
            np.array([-60.0, 30.0, -90.0], dtype=float),

            # Low-elevation / near-zero-course traction branch
            np.array([0.0, 10.0, 0.0], dtype=float),
            np.array([15.0, 10.0, 0.0], dtype=float),
            np.array([-15.0, 10.0, 0.0], dtype=float),
            np.array([30.0, 10.0, 0.0], dtype=float),
            np.array([-30.0, 10.0, 0.0], dtype=float),
            np.array([0.0, 15.0, 0.0], dtype=float),
            np.array([15.0, 15.0, 0.0], dtype=float),
            np.array([-15.0, 15.0, 0.0], dtype=float),
            np.array([30.0, 15.0, 0.0], dtype=float),
            np.array([-30.0, 15.0, 0.0], dtype=float),

            # Near-zero-course variations
            np.array([0.0, 10.0, 15.0], dtype=float),
            np.array([0.0, 10.0, -15.0], dtype=float),
            np.array([15.0, 10.0, 15.0], dtype=float),
            np.array([15.0, 10.0, -15.0], dtype=float),
            np.array([-15.0, 10.0, 15.0], dtype=float),
            np.array([-15.0, 10.0, -15.0], dtype=float),

            # High-side / force-limited branch
            np.array([60.0, 10.0, -30.0], dtype=float),
            np.array([60.0, 15.0, -30.0], dtype=float),
            np.array([60.0, 10.0, 30.0], dtype=float),
            np.array([60.0, 15.0, 30.0], dtype=float),
            np.array([60.0, 10.0, 0.0], dtype=float),
            np.array([60.0, 15.0, 0.0], dtype=float),
            np.array([60.0, 10.0, -60.0], dtype=float),
            np.array([60.0, 15.0, -60.0], dtype=float),
            np.array([60.0, 10.0, 60.0], dtype=float),
            np.array([60.0, 15.0, 60.0], dtype=float),

            np.array([-60.0, 10.0, -30.0], dtype=float),
            np.array([-60.0, 15.0, -30.0], dtype=float),
            np.array([-60.0, 10.0, 30.0], dtype=float),
            np.array([-60.0, 15.0, 30.0], dtype=float),
            np.array([-60.0, 10.0, 0.0], dtype=float),
            np.array([-60.0, 15.0, 0.0], dtype=float),
            np.array([-60.0, 10.0, -60.0], dtype=float),
            np.array([-60.0, 15.0, -60.0], dtype=float),
            np.array([-60.0, 10.0, 60.0], dtype=float),
            np.array([-60.0, 15.0, 60.0], dtype=float),
        ]
    )

    unique = unique_candidate_starts(candidate_starts)

    if MAX_CANDIDATE_STARTS is not None:
        unique = unique[:MAX_CANDIDATE_STARTS]

    return unique


def optimize_heading_with_retries(
    solver,
    true_wind_speed,
    ship_speed,
    heading_deg,
    tether_force_max,
    warm_start_deg=None,
):
    """
    Optimize one heading using multiple physically distinct initial guesses.

    Returns:
        best_result, candidate_log_rows
    """

    case_start_time = time.perf_counter()
    candidate_runtimes = []

    candidate_starts = build_candidate_starts(warm_start_deg=warm_start_deg)

    results = []
    candidate_log_rows = []

    for candidate_index, x0 in enumerate(candidate_starts):
        candidate_start_time = time.perf_counter()

        optimizer = TractionOptimizer(
            solver=solver,
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=heading_deg,
            tether_force_max=tether_force_max,
        )

        result = optimizer.optimize(
            x0_deg=x0,
            verbose=OPTIMIZER_VERBOSE,
        )

        candidate_runtime_s = time.perf_counter() - candidate_start_time
        candidate_runtimes.append(float(candidate_runtime_s))

        result = add_projection_diagnostics(result)

        result["candidate_index"] = int(candidate_index)
        result["candidate_runtime_s"] = float(candidate_runtime_s)

        results.append(result)

        candidate_log_rows.append(
            build_candidate_log_row(
                result=result,
                candidate_index=candidate_index,
                x0_deg=x0,
                selected_best=False,
            )
        )

    case_runtime_s = time.perf_counter() - case_start_time

    max_candidate_runtime_s = (
        float(max(candidate_runtimes)) if candidate_runtimes else 0.0
    )
    mean_candidate_runtime_s = (
        float(np.mean(candidate_runtimes)) if candidate_runtimes else 0.0
    )
    sum_candidate_runtime_s = (
        float(np.sum(candidate_runtimes)) if candidate_runtimes else 0.0
    )

    for r in results:
        r["case_runtime_s"] = float(case_runtime_s)
        r["n_candidates_tried"] = int(len(results))
        r["max_candidate_runtime_s"] = max_candidate_runtime_s
        r["mean_candidate_runtime_s"] = mean_candidate_runtime_s
        r["sum_candidate_runtime_s"] = sum_candidate_runtime_s

    feasible_results = [
        r for r in results
        if is_physically_feasible_result(r, tether_force_max)
    ]

    if feasible_results:
        best = max(feasible_results, key=lambda r: r["P_equiv_traction"])
    else:
        successful = [
            r for r in results
            if r.get("evaluation_success", False)
            and np.isfinite(r.get("P_equiv_traction", np.nan))
        ]

        if successful:
            best = max(successful, key=lambda r: r["P_equiv_traction"])
        else:
            best = results[0]

    best_power = best.get("P_equiv_traction", np.nan)
    best_x = best.get("x_opt_deg", np.array([np.nan, np.nan, np.nan]))

    for row in candidate_log_rows:
        same_power = np.isclose(
            row.get("optimized_P_equiv_traction", np.nan),
            best_power,
            rtol=0.0,
            atol=1.0e-6,
        )

        same_x = (
            np.isclose(
                row.get("optimized_azimuth_angle_deg", np.nan),
                best_x[0],
                atol=1.0e-6,
            )
            and np.isclose(
                row.get("optimized_elevation_angle_deg", np.nan),
                best_x[1],
                atol=1.0e-6,
            )
            and np.isclose(
                row.get("optimized_course_angle_deg", np.nan),
                best_x[2],
                atol=1.0e-6,
            )
        )

        row["selected_best"] = bool(same_power and same_x)

    selected_candidate_index = int(best.get("candidate_index", -1))
    selected_candidate_runtime_s = float(best.get("candidate_runtime_s", np.nan))

    best["selected_candidate_index"] = selected_candidate_index
    best["selected_candidate_runtime_s"] = selected_candidate_runtime_s

    for row in candidate_log_rows:
        row["case_runtime_s"] = float(case_runtime_s)
        row["n_candidates_tried"] = int(len(results))
        row["max_candidate_runtime_s"] = max_candidate_runtime_s
        row["mean_candidate_runtime_s"] = mean_candidate_runtime_s
        row["sum_candidate_runtime_s"] = sum_candidate_runtime_s

    return best, candidate_log_rows


# =============================================================================
# Output rows
# =============================================================================
def build_candidate_log_row(
    result,
    candidate_index,
    x0_deg,
    selected_best=False,
):
    """
    Save one row per multistart candidate.
    """

    return {
        "true_wind_speed": result.get("true_wind_speed", np.nan),
        "ship_speed": result.get("ship_speed", np.nan),
        "heading_deg": result.get("heading_deg", np.nan),
        "candidate_index": candidate_index,
        "selected_best": selected_best,
        "candidate_runtime_s": result.get("candidate_runtime_s", np.nan),
        "case_runtime_s": result.get("case_runtime_s", np.nan),
        "n_candidates_tried": result.get("n_candidates_tried", np.nan),
        "max_candidate_runtime_s": result.get("max_candidate_runtime_s", np.nan),
        "mean_candidate_runtime_s": result.get("mean_candidate_runtime_s", np.nan),
        "sum_candidate_runtime_s": result.get("sum_candidate_runtime_s", np.nan),
        "start_azimuth_angle_deg": float(x0_deg[0]),
        "start_elevation_angle_deg": float(x0_deg[1]),
        "start_course_angle_deg": float(x0_deg[2]),
        "evaluation_success": result.get("evaluation_success", False),
        "optimizer_success": result.get("optimizer_success", False),
        "accepted_result": result.get("accepted_result", False),
        "accepted_despite_optimizer_message": result.get(
            "accepted_despite_optimizer_message",
            False,
        ),
        "optimizer_message": result.get("optimizer_message", ""),
        "optimized_azimuth_angle_deg": result.get("azimuth_angle_deg", np.nan),
        "optimized_elevation_angle_deg": result.get("elevation_angle_deg", np.nan),
        "optimized_course_angle_deg": result.get("course_angle_deg", np.nan),
        "optimized_Fx": result.get("Fx", np.nan),
        "optimized_Fy": result.get("Fy", np.nan),
        "optimized_tether_force_ground": result.get("tether_force_ground", np.nan),
        "Fx_over_tether_force": result.get("Fx_over_tether_force", np.nan),
        "projection_loss_angle_deg": result.get("projection_loss_angle_deg", np.nan),
        "constraint_active": result.get("constraint_active", False),
        "tether_constraint_violation": result.get("tether_constraint_violation", np.nan),
        "optimizer_tether_constraint_violation": result.get(
            "optimizer_tether_constraint_violation",
            np.nan,
        ),
        "optimized_P_equiv_traction": result.get("P_equiv_traction", np.nan),
        "optimized_P_equiv_traction_kW": result.get("P_equiv_traction_kW", np.nan),
    }

def save_candidate_log_to_csv(rows, output_path, quiet=False):
    if not rows:
        return

    fieldnames = [
        "true_wind_speed",
        "ship_speed",
        "heading_deg",
        "candidate_index",
        "start_azimuth_angle_deg",
        "start_elevation_angle_deg",
        "start_course_angle_deg",
        "evaluation_success",
        "optimizer_success",
        "accepted_result",
        "accepted_despite_optimizer_message",
        "optimizer_message",
        "optimized_azimuth_angle_deg",
        "optimized_elevation_angle_deg",
        "optimized_course_angle_deg",
        "optimized_Fx",
        "optimized_Fy",
        "optimized_tether_force_ground",
        "Fx_over_tether_force",
        "projection_loss_angle_deg",
        "constraint_active",
        "tether_constraint_violation",
        "optimizer_tether_constraint_violation",
        "optimized_P_equiv_traction",
        "optimized_P_equiv_traction_kW",
        "selected_best",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not quiet:
        print(f"\nSaved traction candidate log to:\n  {output_path}")
        

def build_output_row(optimized_result, tether_force_max):
    return {
        "true_wind_speed": optimized_result["true_wind_speed"],
        "ship_speed": optimized_result["ship_speed"],
        "heading_deg": optimized_result["heading_deg"],
        "apparent_wind_speed": optimized_result["apparent_wind_speed"],
        "apparent_wind_direction_deg": optimized_result["apparent_wind_direction_deg"],
        "optimized_success": optimized_result["evaluation_success"],
        "optimizer_success": optimized_result["optimizer_success"],
        "accepted_result": optimized_result["accepted_result"],
        "accepted_despite_optimizer_message": optimized_result.get(
            "accepted_despite_optimizer_message",
            False,
        ),
        "optimizer_message": optimized_result["optimizer_message"],
        "case_runtime_s": optimized_result.get("case_runtime_s", np.nan),
        "n_candidates_tried": optimized_result.get("n_candidates_tried", np.nan),
        "selected_candidate_index": optimized_result.get(
            "selected_candidate_index",
            np.nan,
        ),
        "selected_candidate_runtime_s": optimized_result.get(
            "selected_candidate_runtime_s",
            np.nan,
        ),
        "max_candidate_runtime_s": optimized_result.get(
            "max_candidate_runtime_s",
            np.nan,
        ),
        "mean_candidate_runtime_s": optimized_result.get(
            "mean_candidate_runtime_s",
            np.nan,
        ),
        "sum_candidate_runtime_s": optimized_result.get(
            "sum_candidate_runtime_s",
            np.nan,
        ),
        "optimized_azimuth_angle_deg": optimized_result["azimuth_angle_deg"],
        "optimized_elevation_angle_deg": optimized_result["elevation_angle_deg"],
        "optimized_course_angle_deg": optimized_result["course_angle_deg"],
        "optimized_Fx": optimized_result["Fx"],
        "optimized_Fy": optimized_result["Fy"],
        "optimized_tether_force_ground": optimized_result["tether_force_ground"],
        "Fx_over_tether_force": optimized_result.get("Fx_over_tether_force", np.nan),
        "projection_loss_angle_deg": optimized_result.get(
            "projection_loss_angle_deg",
            np.nan,
        ),
        "tether_force_max": tether_force_max,
        "tether_force_effective_limit": tether_force_max - FORCE_SAFETY_MARGIN,
        "tether_constraint_violation": optimized_result["tether_constraint_violation"],
        "optimizer_tether_constraint_violation": optimized_result[
            "optimizer_tether_constraint_violation"
        ],
        "positive_surge_ok": optimized_result["positive_surge_ok"],
        "physical_tether_constraint_ok": optimized_result[
            "physical_tether_constraint_ok"
        ],
        "tether_constraint_ok": optimized_result["tether_constraint_ok"],
        "constraint_active": optimized_result["constraint_active"],
        "optimized_P_equiv_traction": optimized_result["P_equiv_traction"],
        "optimized_P_equiv_traction_kW": optimized_result["P_equiv_traction"] / 1000.0,
    }


# =============================================================================
# Sweep functions
# =============================================================================

def print_progress(
    wind_speed,
    i,
    total,
    heading_deg,
    row,
    accepted_count,
    failed_count,
    best_power,
):
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
        P_kw = np.nan
        Fx = np.nan
        T = np.nan
        az = np.nan
        beta = np.nan
        course = np.nan
        Fx_over_T = np.nan
        proj_loss = np.nan

    best_kw = best_power / 1000.0 if np.isfinite(best_power) else np.nan

    runtime_s = row.get("case_runtime_s", np.nan)
    n_candidates = row.get("n_candidates_tried", np.nan)
    max_candidate_runtime_s = row.get("max_candidate_runtime_s", np.nan)

    msg = (
        f"  Vw={wind_speed:5.1f} m/s | "
        f"{i:03d}/{total:03d} | "
        f"ψ={heading_deg:7.2f} deg | "
        f"t={format_s(runtime_s)}s | "
        f"cand={int(n_candidates) if np.isfinite(n_candidates) else 0:02d} | "
        f"tcand,max={format_s(max_candidate_runtime_s)}s | "
        f"accepted={accepted_count:03d} | "
        f"failed={failed_count:03d} | "
        f"Peq={P_kw:9.3f} kW | "
        f"Fx={Fx:9.1f} N | "
        f"T={T:9.1f} N | "
        f"az={az:7.2f}° | "
        f"β={beta:6.2f}° | "
        f"course={course:8.2f}° | "
        f"Fx/T={Fx_over_T:6.3f} | "
        f"loss={proj_loss:6.2f}° | "
        f"best={best_kw:9.3f} kW"
    )

    sys.stdout.write("\r" + msg.ljust(360))
    sys.stdout.flush()


def run_heading_sweep(
    solver,
    true_wind_speed,
    ship_speed,
    headings,
    tether_force_max,
    autosave_prefix_rows=None,
    autosave_prefix_candidate_rows=None,
):
    rows = []
    candidate_rows = []

    warm_start_deg = INITIAL_GUESS_DEG.copy()

    accepted_count = 0
    failed_count = 0
    best_power = -np.inf

    total = len(headings)

    for i, heading_deg in enumerate(headings, start=1):
        optimized_result, case_candidate_rows = optimize_heading_with_retries(
            solver=solver,
            true_wind_speed=float(true_wind_speed),
            ship_speed=float(ship_speed),
            heading_deg=float(heading_deg),
            tether_force_max=tether_force_max,
            warm_start_deg=warm_start_deg,
        )

        if optimized_result["accepted_result"]:
            warm_start_deg = optimized_result["x_opt_deg"]

        row = build_output_row(
            optimized_result=optimized_result,
            tether_force_max=tether_force_max,
        )

        rows.append(row)
        candidate_rows.extend(case_candidate_rows)

        if row["accepted_result"]:
            accepted_count += 1
            best_power = max(best_power, row["optimized_P_equiv_traction"])
        else:
            failed_count += 1

        if AUTOSAVE_AFTER_EACH_CASE:
            prefix_rows = autosave_prefix_rows if autosave_prefix_rows is not None else []
            save_results_to_csv(prefix_rows + rows, CSV_AUTOSAVE_PATH, quiet=True)

            prefix_candidate_rows = (
                autosave_prefix_candidate_rows
                if autosave_prefix_candidate_rows is not None
                else []
            )
            if AUTOSAVE_CANDIDATE_LOG_AFTER_EACH_CASE:
                save_candidate_log_to_csv(
                    prefix_candidate_rows + candidate_rows,
                    CSV_CANDIDATE_LOG_PATH,
                    quiet=True,
            )

        print_progress(
            wind_speed=true_wind_speed,
            i=i,
            total=total,
            heading_deg=float(heading_deg),
            row=row,
            accepted_count=accepted_count,
            failed_count=failed_count,
            best_power=best_power,
        )

    print("")
    return rows, candidate_rows


def run_wind_heading_sweep(
    solver,
    true_wind_speeds,
    ship_speed,
    headings,
    tether_force_max,
):
    all_rows = []
    all_candidate_rows = []

    for true_wind_speed in true_wind_speeds:
        print(
            f"\nRunning optimized traction sweep: "
            f"true wind = {true_wind_speed:.1f} m/s, "
            f"ship speed = {ship_speed:.1f} m/s"
        )

        rows, candidate_rows = run_heading_sweep(
            solver=solver,
            true_wind_speed=float(true_wind_speed),
            ship_speed=ship_speed,
            headings=headings,
            tether_force_max=tether_force_max,
            autosave_prefix_rows=all_rows,
            autosave_prefix_candidate_rows=all_candidate_rows,
        )

        all_rows.extend(rows)
        all_candidate_rows.extend(candidate_rows)

    return all_rows, all_candidate_rows


# =============================================================================
# CSV
# =============================================================================

def save_results_to_csv(rows, output_path, quiet=False):
    fieldnames = [
        "true_wind_speed",
        "ship_speed",
        "heading_deg",
        "apparent_wind_speed",
        "apparent_wind_direction_deg",
        "optimized_success",
        "optimizer_success",
        "accepted_result",
        "accepted_despite_optimizer_message",
        "optimizer_message",
        "optimized_azimuth_angle_deg",
        "optimized_elevation_angle_deg",
        "optimized_course_angle_deg",
        "optimized_Fx",
        "optimized_Fy",
        "optimized_tether_force_ground",
        "Fx_over_tether_force",
        "projection_loss_angle_deg",
        "tether_force_max",
        "tether_force_effective_limit",
        "tether_constraint_violation",
        "optimizer_tether_constraint_violation",
        "positive_surge_ok",
        "physical_tether_constraint_ok",
        "tether_constraint_ok",
        "constraint_active",
        "optimized_P_equiv_traction",
        "optimized_P_equiv_traction_kW",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not quiet:
        print(f"\nSaved optimized traction wind-heading sweep to:\n  {output_path}")

# =============================================================================
# Summary
# =============================================================================

def print_compact_summary(rows):
    print("\nCOMPACT SUMMARY")
    print("---------------")

    wind_speeds = sorted(set(float(r["true_wind_speed"]) for r in rows))

    for wind_speed in wind_speeds:
        group = [
            r for r in rows
            if np.isclose(float(r["true_wind_speed"]), wind_speed)
        ]

        accepted = [r for r in group if r["accepted_result"]]

        if accepted:
            best = max(accepted, key=lambda r: r["optimized_P_equiv_traction"])
            best_kw = best["optimized_P_equiv_traction"] / 1000.0
            best_heading = best["heading_deg"]

            print(
                f"Vw={wind_speed:5.1f} m/s | "
                f"accepted={len(accepted):02d} | "
                f"failed={len(group) - len(accepted):02d} | "
                f"best Peq={best_kw:9.3f} kW at ψ={best_heading:7.2f} deg"
            )
        else:
            print(
                f"Vw={wind_speed:5.1f} m/s | "
                f"accepted=00 | "
                f"failed={len(group):02d} | "
                f"best Peq=nan"
            )

def print_selected_solution_table(rows, target_wind_speed=DIAGNOSTIC_WIND_SPEED):
    """
    Print selected optimized traction variables and projection diagnostics.
    """

    selected = [
        r for r in rows
        if np.isclose(float(r["true_wind_speed"]), target_wind_speed)
        and r.get("accepted_result", False)
    ]

    if not selected:
        print(f"\nNo accepted rows found for Vw={target_wind_speed:.1f} m/s.")
        return

    selected = sorted(selected, key=lambda r: r["heading_deg"])

    print("\nSELECTED TRACTION SOLUTIONS")
    print("---------------------------")
    print(
        " heading |   Peq |     Fx |      T |     az |   beta |   course |  Fx/T | loss"
    )
    print(
        "   [deg] |  [kW] |    [N] |    [N] |  [deg] |  [deg] |    [deg] |   [-] | [deg]"
    )
    print("-" * 92)

    for r in selected:
        print(
            f"{r['heading_deg']:8.1f} | "
            f"{r['optimized_P_equiv_traction_kW']:6.2f} | "
            f"{r['optimized_Fx']:6.0f} | "
            f"{r['optimized_tether_force_ground']:6.0f} | "
            f"{r['optimized_azimuth_angle_deg']:6.1f} | "
            f"{r['optimized_elevation_angle_deg']:6.1f} | "
            f"{r['optimized_course_angle_deg']:8.1f} | "
            f"{r.get('Fx_over_tether_force', np.nan):5.3f} | "
            f"{r.get('projection_loss_angle_deg', np.nan):5.1f}"
        )

# =============================================================================
# Plotting
# =============================================================================

def close_polar_curve(theta, radius):
    theta = np.asarray(theta, dtype=float)
    radius = np.asarray(radius, dtype=float)

    return (
        np.concatenate([theta, [theta[0] + 2.0 * np.pi]]),
        np.concatenate([radius, [radius[0]]]),
    )


def make_rows_for_polar_plot(rows):
    """
    Build plotting rows.

    If MIRROR_RESULTS_FOR_PLOT=True:
        computed heading h in (0, 180) is mirrored to 360 - h.

    If MIRROR_RESULTS_FOR_PLOT=False:
        rows are returned unchanged.
    """

    if not MIRROR_RESULTS_FOR_PLOT:
        return [dict(row, mirrored_for_plot=False) for row in rows]

    plot_rows = []

    for row in rows:
        base = dict(row)
        base["mirrored_for_plot"] = False
        plot_rows.append(base)

        heading = float(row["heading_deg"])

        if 0.0 < heading < 180.0:
            mirrored = dict(row)
            mirrored["heading_deg"] = 360.0 - heading
            mirrored["mirrored_for_plot"] = True

            if np.isfinite(mirrored.get("optimized_Fy", np.nan)):
                mirrored["optimized_Fy"] = -float(mirrored["optimized_Fy"])

            plot_rows.append(mirrored)

    return plot_rows

def plot_polar_positive_p_equiv_traction(rows, force_limit_text=""):
    plot_rows = make_rows_for_polar_plot(rows)

    if not plot_rows:
        print("No rows available for polar plot.")
        return

    wind_speeds = sorted(set(float(r["true_wind_speed"]) for r in plot_rows))

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={"projection": "polar"},
    )

    max_radius = 0.0

    for wind_speed in wind_speeds:
        group = [
            r for r in plot_rows
            if np.isclose(float(r["true_wind_speed"]), wind_speed)
        ]

        rows_sorted = sorted(group, key=lambda r: r["heading_deg"])

        headings = np.array(
            [r["heading_deg"] for r in rows_sorted],
            dtype=float,
        )

        P_equiv = np.array(
            [
                r["optimized_P_equiv_traction"]
                if (
                    r.get("accepted_result", False)
                    and np.isfinite(r.get("optimized_P_equiv_traction", np.nan))
                )
                else np.nan
                for r in rows_sorted
            ],
            dtype=float,
        )

        theta = np.deg2rad(headings)
        radius = np.maximum(P_equiv, 0.0) / 1000.0

        theta_closed, radius_closed = close_polar_curve(theta, radius)

        if np.any(np.isfinite(radius_closed)):
            max_radius = max(max_radius, float(np.nanmax(radius_closed)))

        ax.plot(
            theta_closed,
            radius_closed,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"{wind_speed:.0f} m/s",
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.10 * max_radius if max_radius > 0.0 else 1.0)
    ax.set_rlabel_position(135)

    if MIRROR_RESULTS_FOR_PLOT:
        mirror_text = "0–180° computed, 180–360° mirrored for plotting only"
    else:
        mirror_text = "0–360° computed directly"

    ax.set_title(
        f"Positive equivalent traction power\n"
        f"Ship speed = {SHIP_SPEED:.1f} m/s\n"
        f"{force_limit_text}\n"
        f"{mirror_text}",
        pad=25,
    )

    ax.text(
        np.deg2rad(135),
        1.07 * max_radius if max_radius > 0.0 else 0.95,
        "P_equiv_traction [kW]",
        ha="center",
        va="center",
    )

    ax.legend(
        title="True wind speed",
        loc="upper right",
        bbox_to_anchor=(1.35, 1.10),
    )

    ax.grid(True)

    

def plot_optimized_operating_angles(rows, force_limit_text=""):
    """
    Plot optimized azimuth, elevation, and course angle versus ship heading.

    This plot uses only actually computed rows, not mirrored rows.
    That is safer because azimuth/course mirroring depends on the exact
    sign convention used in the QSM frame.
    """

    if not PLOT_OPERATING_ANGLES:
        return

    accepted_rows = [
        r for r in rows
        if r.get("accepted_result", False)
        and np.isfinite(r.get("optimized_azimuth_angle_deg", np.nan))
        and np.isfinite(r.get("optimized_elevation_angle_deg", np.nan))
        and np.isfinite(r.get("optimized_course_angle_deg", np.nan))
    ]

    diagnostic_group = [
        r for r in accepted_rows
        if np.isclose(float(r["true_wind_speed"]), DIAGNOSTIC_WIND_SPEED)
    ]

    if not diagnostic_group:
        print(
            f"No accepted optimized angle results found for "
            f"DIAGNOSTIC_WIND_SPEED = {DIAGNOSTIC_WIND_SPEED:.1f} m/s."
        )
        return

    rows_sorted = sorted(diagnostic_group, key=lambda r: r["heading_deg"])

    headings = np.array(
        [r["heading_deg"] for r in rows_sorted],
        dtype=float,
    )

    azimuth = np.array(
        [r["optimized_azimuth_angle_deg"] for r in rows_sorted],
        dtype=float,
    )

    elevation = np.array(
        [r["optimized_elevation_angle_deg"] for r in rows_sorted],
        dtype=float,
    )

    course = np.array(
        [r["optimized_course_angle_deg"] for r in rows_sorted],
        dtype=float,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        headings,
        azimuth,
        linewidth=1.8,
        label="azimuth angle",
    )

    plt.plot(
        headings,
        elevation,
        linewidth=1.8,
        label="elevation angle",
    )

    plt.plot(
        headings,
        course,
        linewidth=1.8,
        label="course angle",
    )

    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Optimized angle [deg]")
    plt.title(
        f"Optimized traction operating angles\n"
        f"True wind = {DIAGNOSTIC_WIND_SPEED:.1f} m/s, "
        f"ship speed = {SHIP_SPEED:.1f} m/s\n"
        f"{force_limit_text}"
    )

    plt.legend()
    plt.grid(True)


# =============================================================================
# Main
# =============================================================================

def main():
    constructor = PowerCurveConstructor(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_file=False,
        verbose=False,
    )

    env_state = constructor.create_environment(cluster_id=CLUSTER_ID)

    solver = PureTractionSolver(
        sys_props=constructor.sys_props,
        env_state=env_state,
        steady_state_config=constructor.simulation_settings.get("steady_state"),
    )

    physical_tether_force_max = float(constructor.sys_props.tether_force_max_limit)

    if USE_TETHER_FORCE_LIMIT:
        tether_force_max = physical_tether_force_max
    else:
        tether_force_max = DIAGNOSTIC_TETHER_FORCE_MAX

    force_limit_plot_text = build_force_limit_plot_label(
        physical_tether_force_max=physical_tether_force_max,
        optimizer_tether_force_max=tether_force_max,
    )
    print("\nCONSTRAINED TRACTION OPTIMIZED WIND-HEADING SWEEP")
    print("-------------------------------------------------")
    print(f"True wind speeds      : {TRUE_WIND_SPEEDS}")
    print(f"Ship speed            : {SHIP_SPEED:.3f} m/s")
    print(f"Tether length         : {TETHER_LENGTH:.3f} m")
    print(f"Tether force max      : {tether_force_max:.3f} N")
    print(f"Effective T limit     : {tether_force_max - FORCE_SAFETY_MARGIN:.3f} N")
    print(f"Computed headings     : {SWEEP_HEADINGS[0]:.1f}–{SWEEP_HEADINGS[-1]:.1f} deg")
    print(f"Heading step          : {SWEEP_HEADINGS[1] - SWEEP_HEADINGS[0]:.3f} deg")
    print(f"Mirrored for plot     : {MIRROR_RESULTS_FOR_PLOT}")
    print(f"Candidate starts      : {'all' if MAX_CANDIDATE_STARTS is None else MAX_CANDIDATE_STARTS}")
    print(f"Physical tether force max : {physical_tether_force_max:.3f} N")
    print(f"Optimizer tether limit    : {tether_force_max:.3f} N")
    print(f"Force limit active        : {USE_TETHER_FORCE_LIMIT}")
    print("\nObjective")
    print("---------")
    print("Maximize P_equiv_traction = Fx_ship * V_ship")
    print("Positive Fx_ship helps propulsion.")
    print("The polar plot radius is P_equiv_traction [kW], not Fx.")

    rows, candidate_rows = run_wind_heading_sweep(
        solver=solver,
        true_wind_speeds=TRUE_WIND_SPEEDS,
        ship_speed=SHIP_SPEED,
        headings=SWEEP_HEADINGS,
        tether_force_max=tether_force_max,
    )

    save_results_to_csv(rows, CSV_OUTPUT_PATH)
    save_candidate_log_to_csv(candidate_rows, CSV_CANDIDATE_LOG_PATH)

    print_compact_summary(rows)
    print_selected_solution_table(rows, target_wind_speed=DIAGNOSTIC_WIND_SPEED)

    plot_optimized_operating_angles(
        rows,
        force_limit_text=force_limit_plot_text,
    )

    plot_polar_positive_p_equiv_traction(
        rows,
        force_limit_text=force_limit_plot_text,
    )
    plt.show()

if __name__ == "__main__":
    main()