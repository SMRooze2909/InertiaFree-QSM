#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constrained traction optimization heading sweep with moving-vessel apparent wind.

Goal:
    Move from one optimized traction operating point to an optimized heading sweep.

Objective:
    maximize P_equiv_traction = Fx * V_ship

Decision variables:
    x = [azimuth_angle, elevation_angle, course_angle]

All optimizer variables are handled in degrees.

Important:
- QSM solve is performed in the apparent-wind-aligned frame.
- Coordinate transformation is then applied:
      QSM frame -> global frame -> ship frame
- The tether force limit is not ignored in the final result.
- During optimizer trial evaluations, the solver limit is temporarily disabled
  so SLSQP can evaluate infeasible points and enforce the tether constraint itself.
"""

import sys
from pathlib import Path
from dataclasses import replace
import csv

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

CSV_OUTPUT_PATH = RESULTS_DIR / "optimized_traction_heading_sweep.csv"


# =============================================================================
# Heading sweep settings
# =============================================================================

TRUE_WIND_SPEED = 15.0      # m/s
SHIP_SPEED = 5.0            # m/s
CLUSTER_ID = 1
TETHER_LENGTH = 500.0       # m

# Do not include 360 here, because 0 and 360 are the same heading.
# A separate periodicity check is included below.
SWEEP_HEADINGS = np.arange(0.0, 360.0, 5.0)

OPTIMIZER_VERBOSE = False


# =============================================================================
# Initial operating point and optimizer settings
# =============================================================================

X0_DEG = np.array(
    [
        11.5,   # azimuth angle [deg]
        30.0,   # elevation angle [deg]
        93.0,   # course angle [deg]
    ],
    dtype=float,
)

BOUNDS_DEG = [
    (-90.0, 90.0),   # azimuth angle [deg]
    (10.0, 80.0),    # elevation angle [deg]
    (0.0, 180.0),    # course angle [deg]
]

SCALING = np.array(
    [
        30.0,   # azimuth scaling
        30.0,   # elevation scaling
        90.0,   # course scaling
    ],
    dtype=float,
)

FORCE_TOLERANCE = 1.0          # N, numerical tolerance for boundary solutions
FORCE_SAFETY_MARGIN = 5.0      # N, keeps optimized solution slightly below limit
ZERO_POWER_THRESHOLD = 10.0    # W, below this traction is treated as inactive


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
    base_input,
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
        speed=true_wind_speed,
        direction_to=np.deg2rad(0.0),
    )

    vessel_heading = np.deg2rad(heading_deg)

    vessel_motion = VesselMotion(
        speed=ship_speed,
        heading=vessel_heading,
    )

    app = compute_apparent_wind(true_wind, vessel_motion)

    traction_input = replace(
        base_input,
        azimuth_angle=np.deg2rad(azimuth_deg),
        elevation_angle=np.deg2rad(elevation_deg),
        course_angle=np.deg2rad(course_deg),
    )

    # QSM is run in apparent-wind-aligned frame.
    wind_qsm = WindCondition(
        speed=app.speed,
        direction=0.0,
    )

    vessel_state_qsm = VesselState(
        speed=ship_speed,
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
        apparent_wind_direction=app.direction_to,
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
        "apparent_wind_speed": float(app.speed),
        "apparent_wind_direction_deg": float(np.rad2deg(app.direction_to)),
        "tether_force_ground": float(result.tether_force_ground),
        "Fx": float(Fx_ship),
        "Fy": float(Fy_ship),
        "P_equiv_traction": float(P_equiv_traction),
    }


def evaluate_traction_operating_point_safe(
    solver,
    base_input,
    true_wind_speed,
    ship_speed,
    heading_deg,
    x_deg,
):
    try:
        return evaluate_traction_operating_point(
            solver=solver,
            base_input=base_input,
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=heading_deg,
            x_deg=x_deg,
        )

    except Exception as exc:
        return {
            "evaluation_success": False,
            "error_message": str(exc),
            "azimuth_angle_deg": float(x_deg[0]),
            "elevation_angle_deg": float(x_deg[1]),
            "course_angle_deg": float(x_deg[2]),
            "true_wind_speed": float(true_wind_speed),
            "ship_speed": float(ship_speed),
            "heading_deg": float(heading_deg),
            "apparent_wind_speed": np.nan,
            "apparent_wind_direction_deg": np.nan,
            "tether_force_ground": np.nan,
            "Fx": np.nan,
            "Fy": np.nan,
            "P_equiv_traction": np.nan,
        }


# =============================================================================
# Optimizer
# =============================================================================

class TractionOptimizer:
    def __init__(
        self,
        solver,
        base_input,
        true_wind_speed,
        ship_speed,
        heading_deg,
        tether_force_max,
    ):
        self.solver = solver
        self.base_input = base_input
        self.true_wind_speed = true_wind_speed
        self.ship_speed = ship_speed
        self.heading_deg = heading_deg
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
                base_input=self.base_input,
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
                "Fx": -np.inf,
                "Fy": np.nan,
                "tether_force_ground": np.inf,
                "P_equiv_traction": -np.inf,
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

        The small safety margin prevents SLSQP from returning points that are
        numerically just above the force limit.
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
                "maxiter": 200,
                "ftol": 1.0e-8,
                "eps": 1.0e-3,
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
            final_result["optimizer_success"]
            and final_result["evaluation_success"]
            and final_result["positive_surge_ok"]
            and final_result["tether_constraint_ok"]
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

        return final_result


# =============================================================================
# Multistart helper
# =============================================================================

def optimize_heading_with_retries(
    solver,
    base_input,
    true_wind_speed,
    ship_speed,
    heading_deg,
    tether_force_max,
    warm_start_deg=None,
):
    """
    Optimize one heading using a small set of initial guesses.

    This makes the heading sweep more robust than relying on one x0.
    """

    candidate_starts = []

    if warm_start_deg is not None:
        candidate_starts.append(np.asarray(warm_start_deg, dtype=float))

    candidate_starts.extend(
        [
            X0_DEG,
            np.array([0.0, 10.0, 90.0], dtype=float),
            np.array([0.0, 30.0, 90.0], dtype=float),
            np.array([0.0, 45.0, 90.0], dtype=float),
        ]
    )

    results = []

    for x0 in candidate_starts:
        optimizer = TractionOptimizer(
            solver=solver,
            base_input=base_input,
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=heading_deg,
            tether_force_max=tether_force_max,
        )

        result = optimizer.optimize(
            x0_deg=x0,
            verbose=OPTIMIZER_VERBOSE,
        )

        results.append(result)

    accepted = [r for r in results if r["accepted_result"]]

    if accepted:
        return max(accepted, key=lambda r: r["P_equiv_traction"])

    successful = [r for r in results if r["evaluation_success"]]

    if successful:
        return max(successful, key=lambda r: r["P_equiv_traction"])

    return results[0]


# =============================================================================
# Heading sweep
# =============================================================================

def build_output_row(fixed_result, optimized_result, tether_force_max):
    improvement = (
        optimized_result["P_equiv_traction"]
        - fixed_result["P_equiv_traction"]
    )

    return {
        "true_wind_speed": fixed_result["true_wind_speed"],
        "ship_speed": fixed_result["ship_speed"],
        "heading_deg": fixed_result["heading_deg"],
        "apparent_wind_speed": fixed_result["apparent_wind_speed"],
        "apparent_wind_direction_deg": fixed_result["apparent_wind_direction_deg"],

        "fixed_success": fixed_result["evaluation_success"],
        "fixed_error_message": fixed_result["error_message"],
        "fixed_azimuth_angle_deg": fixed_result["azimuth_angle_deg"],
        "fixed_elevation_angle_deg": fixed_result["elevation_angle_deg"],
        "fixed_course_angle_deg": fixed_result["course_angle_deg"],
        "fixed_Fx": fixed_result["Fx"],
        "fixed_Fy": fixed_result["Fy"],
        "fixed_tether_force_ground": fixed_result["tether_force_ground"],
        "fixed_P_equiv_traction": fixed_result["P_equiv_traction"],

        "optimized_success": optimized_result["evaluation_success"],
        "optimizer_success": optimized_result["optimizer_success"],
        "accepted_result": optimized_result["accepted_result"],
        "optimizer_message": optimized_result["optimizer_message"],
        "optimized_azimuth_angle_deg": optimized_result["azimuth_angle_deg"],
        "optimized_elevation_angle_deg": optimized_result["elevation_angle_deg"],
        "optimized_course_angle_deg": optimized_result["course_angle_deg"],
        "optimized_Fx": optimized_result["Fx"],
        "optimized_Fy": optimized_result["Fy"],
        "optimized_tether_force_ground": optimized_result["tether_force_ground"],
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

        "improvement_P_equiv": improvement,
        "preferred_over_fixed": improvement > 0.0,
    }


def run_heading_sweep(
    solver,
    base_input,
    true_wind_speed,
    ship_speed,
    headings,
    tether_force_max,
):
    rows = []
    warm_start_deg = X0_DEG.copy()

    total = len(headings)

    for i, heading_deg in enumerate(headings, start=1):
        print(f"  [{i:03d}/{total:03d}] heading = {heading_deg:7.2f} deg")
        
        

        fixed_result = evaluate_traction_operating_point_safe(
            solver=solver,
            base_input=base_input,
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=float(heading_deg),
            x_deg=X0_DEG,
        )

        optimized_result = optimize_heading_with_retries(
            solver=solver,
            base_input=base_input,
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=float(heading_deg),
            tether_force_max=tether_force_max,
            warm_start_deg=warm_start_deg,
        )

        if optimized_result["accepted_result"]:
            warm_start_deg = optimized_result["x_opt_deg"]

        row = build_output_row(
            fixed_result=fixed_result,
            optimized_result=optimized_result,
            tether_force_max=tether_force_max,
        )

        rows.append(row)

    print("")
    return rows


# =============================================================================
# CSV
# =============================================================================

def save_results_to_csv(rows, output_path):
    fieldnames = [
        "true_wind_speed",
        "ship_speed",
        "heading_deg",
        "apparent_wind_speed",
        "apparent_wind_direction_deg",

        "fixed_success",
        "fixed_error_message",
        "fixed_azimuth_angle_deg",
        "fixed_elevation_angle_deg",
        "fixed_course_angle_deg",
        "fixed_Fx",
        "fixed_Fy",
        "fixed_tether_force_ground",
        "fixed_P_equiv_traction",

        "optimized_success",
        "optimizer_success",
        "accepted_result",
        "optimizer_message",
        "optimized_azimuth_angle_deg",
        "optimized_elevation_angle_deg",
        "optimized_course_angle_deg",
        "optimized_Fx",
        "optimized_Fy",
        "optimized_tether_force_ground",
        "tether_force_max",
        "tether_force_effective_limit",
        "tether_constraint_violation",
        "optimizer_tether_constraint_violation",
        "positive_surge_ok",
        "physical_tether_constraint_ok",
        "tether_constraint_ok",
        "constraint_active",
        "optimized_P_equiv_traction",

        "improvement_P_equiv",
        "preferred_over_fixed",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"\nSaved optimized traction heading sweep to:\n  {output_path}")


# =============================================================================
# Verification helpers
# =============================================================================

def heading_intervals(headings, mask, step_deg=5.0):
    headings = np.asarray(headings, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    selected = np.sort(headings[mask])

    if len(selected) == 0:
        return []

    intervals = []
    start = selected[0]
    previous = selected[0]

    for h in selected[1:]:
        if h - previous <= step_deg * 1.5:
            previous = h
        else:
            intervals.append((start, previous))
            start = h
            previous = h

    intervals.append((start, previous))

    return intervals


def format_heading_intervals(intervals):
    if not intervals:
        return "none"

    return ", ".join(
        f"{start:.1f}–{end:.1f} deg"
        for start, end in intervals
    )


def report_jump_check(name, values):
    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return

    diffs = np.abs(np.diff(values))
    max_jump = np.nanmax(diffs)
    median_jump = np.nanmedian(diffs)

    print(f"Max |Δ{name}| between headings : {max_jump:12.3f}")
    print(f"Med |Δ{name}| between headings : {median_jump:12.3f}")

    if median_jump > 1.0e-9 and max_jump / median_jump > 10.0:
        print(f"CHECK: {name} has a sharp jump. Inspect plot near that heading.")
    else:
        print(f"PASS: {name} varies reasonably smoothly.")

def report_largest_jumps(name, headings, values, n=5):
    """
    Report the largest jumps between neighbouring heading points.

    This helps identify whether sharp changes occur near physically expected
    transition regions, for example between force-limited traction and
    near-zero traction operation.
    """

    headings = np.asarray(headings, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return

    jumps = np.abs(np.diff(values))
    idx_sorted = np.argsort(jumps)[::-1][:n]

    print(f"\nLargest {name} jumps")
    print("-" * (len(name) + 14))

    for idx in idx_sorted:
        print(
            f"{headings[idx]:7.2f} -> {headings[idx + 1]:7.2f} deg | "
            f"Δ{name} = {jumps[idx]:12.3f}"
        )

def verify_heading_sweep(rows):
    accepted = [r for r in rows if r["accepted_result"]]
    not_accepted = [r for r in rows if not r["accepted_result"]]
    optimizer_failed = [r for r in rows if not r["optimizer_success"]]

    if optimizer_failed:
        print("\nOptimizer failure summary")
        print("-------------------------")

        failure_messages = {}

        for r in optimizer_failed:
            msg = r["optimizer_message"]
            failure_messages[msg] = failure_messages.get(msg, 0) + 1

        for msg, count in failure_messages.items():
            print(f"{count:4d} cases | {msg}")

    print("\nOPTIMIZED TRACTION HEADING SWEEP CHECKS")
    print("---------------------------------------")
    print(f"Total headings              : {len(rows)}")
    print(f"Accepted optimized cases    : {len(accepted)}")
    print(f"Rejected / failed cases     : {len(not_accepted)}")

    if not accepted:
        print("No accepted cases. Cannot perform detailed checks.")
        return

    headings = np.array([r["heading_deg"] for r in accepted])
    ship_speed = np.array([r["ship_speed"] for r in accepted])

    fixed_P = np.array([r["fixed_P_equiv_traction"] for r in accepted])
    opt_P = np.array([r["optimized_P_equiv_traction"] for r in accepted])

    fixed_Fx = np.array([r["fixed_Fx"] for r in accepted])
    opt_Fx = np.array([r["optimized_Fx"] for r in accepted])

    opt_T = np.array([r["optimized_tether_force_ground"] for r in accepted])
    T_max = np.array([r["tether_force_max"] for r in accepted])
    T_eff = np.array([r["tether_force_effective_limit"] for r in accepted])

    improvement = np.array([r["improvement_P_equiv"] for r in accepted])

    opt_azimuth = np.array([r["optimized_azimuth_angle_deg"] for r in accepted])
    opt_elevation = np.array([r["optimized_elevation_angle_deg"] for r in accepted])
    opt_course = np.array([r["optimized_course_angle_deg"] for r in accepted])

    step_deg = float(np.median(np.diff(np.sort(headings)))) if len(headings) > 1 else 5.0

    print("\nBasic performance")
    print("-----------------")
    print(f"Max optimized P_equiv       : {np.nanmax(opt_P) / 1000.0:12.3f} kW")
    print(f"Min optimized P_equiv       : {np.nanmin(opt_P) / 1000.0:12.3f} kW")
    print(f"Max optimized Fx            : {np.nanmax(opt_Fx):12.3f} N")
    print(f"Min optimized Fx            : {np.nanmin(opt_Fx):12.3f} N")
    print(f"Max tether force            : {np.nanmax(opt_T):12.3f} N")
    print(f"Min physical tether margin  : {np.nanmin(T_max - opt_T):12.6f} N")
    print(f"Min optimizer tether margin : {np.nanmin(T_eff - opt_T):12.6f} N")
    print(f"Mean improvement            : {np.nanmean(improvement) / 1000.0:12.3f} kW")
    print(f"Min improvement             : {np.nanmin(improvement) / 1000.0:12.3f} kW")

    print("\nFormula consistency")
    print("-------------------")
    fixed_formula_error = np.nanmax(np.abs(fixed_P - fixed_Fx * ship_speed))
    opt_formula_error = np.nanmax(np.abs(opt_P - opt_Fx * ship_speed))

    print(f"Max fixed P-FxV error       : {fixed_formula_error:12.6e} W")
    print(f"Max optimized P-FxV error   : {opt_formula_error:12.6e} W")

    if fixed_formula_error < 1.0e-6 and opt_formula_error < 1.0e-6:
        print("PASS: P_equiv = Fx * V_ship is internally consistent.")
    else:
        print("CHECK: P_equiv formula mismatch detected.")

    print("\nConstraint checks")
    print("-----------------")
    negative_fx_count = np.count_nonzero(opt_Fx < -1.0e-6)
    physical_tether_violation = np.maximum(0.0, opt_T - T_max)
    optimizer_tether_violation = np.maximum(0.0, opt_T - T_eff)

    physical_violations = np.count_nonzero(physical_tether_violation > FORCE_TOLERANCE)
    optimizer_violations = np.count_nonzero(optimizer_tether_violation > FORCE_TOLERANCE)

    print(f"Negative optimized Fx cases : {negative_fx_count}")
    print(f"Physical T violations > tol : {physical_violations}")
    print(f"Optimizer T violations > tol: {optimizer_violations}")
    print(f"Max physical T violation    : {np.nanmax(physical_tether_violation):12.6f} N")
    print(f"Max optimizer T violation   : {np.nanmax(optimizer_tether_violation):12.6f} N")

    if negative_fx_count == 0 and optimizer_violations == 0:
        print("PASS: All accepted optimized points satisfy optimizer constraints.")
    else:
        print("CHECK: Some accepted points violate optimizer constraints.")

    print("\nImprovement checks")
    print("------------------")
    worse_count = np.count_nonzero(improvement < -1.0e-6)
    print(f"Optimized worse than fixed  : {worse_count}")

    if worse_count == 0:
        print("PASS: Optimized traction is never worse than fixed traction.")
    else:
        print("CHECK: Some optimized cases are worse than fixed.")

    print("\nInactive traction regions")
    print("-------------------------")
    inactive_mask = opt_P <= ZERO_POWER_THRESHOLD
    inactive_intervals = heading_intervals(headings, inactive_mask, step_deg)
    active_intervals = heading_intervals(headings, ~inactive_mask, step_deg)

    print(f"Active traction headings    : {format_heading_intervals(active_intervals)}")
    print(f"Inactive/zero headings      : {format_heading_intervals(inactive_intervals)}")

    print("\nBound activity")
    print("--------------")
    angle_tol = 1.0e-2

    az_min, az_max = BOUNDS_DEG[0]
    el_min, el_max = BOUNDS_DEG[1]
    co_min, co_max = BOUNDS_DEG[2]

    print(f"Azimuth at lower bound      : {np.count_nonzero(np.isclose(opt_azimuth, az_min, atol=angle_tol))}")
    print(f"Azimuth at upper bound      : {np.count_nonzero(np.isclose(opt_azimuth, az_max, atol=angle_tol))}")
    print(f"Elevation at lower bound    : {np.count_nonzero(np.isclose(opt_elevation, el_min, atol=angle_tol))}")
    print(f"Elevation at upper bound    : {np.count_nonzero(np.isclose(opt_elevation, el_max, atol=angle_tol))}")
    print(f"Course at lower bound       : {np.count_nonzero(np.isclose(opt_course, co_min, atol=angle_tol))}")
    print(f"Course at upper bound       : {np.count_nonzero(np.isclose(opt_course, co_max, atol=angle_tol))}")

    print("\nSmoothness checks")
    print("-----------------")
    report_jump_check("P_equiv", opt_P)
    report_jump_check("Fx", opt_Fx)
    report_jump_check("tether force", opt_T)

    report_largest_jumps("P_equiv", headings, opt_P)
    report_largest_jumps("Fx", headings, opt_Fx)
    report_largest_jumps("tether force", headings, opt_T)

    if not_accepted:
        print("\nRejected headings")
        print("-----------------")
        for r in not_accepted:
            print(
                f"heading {r['heading_deg']:7.2f} deg | "
                f"optimizer_success={r['optimizer_success']} | "
                f"accepted={r['accepted_result']} | "
                f"message={r['optimizer_message']}"
            )


def periodic_heading_check(
    solver,
    base_input,
    true_wind_speed,
    ship_speed,
    tether_force_max,
):
    """
    Check that heading 0 deg and 360 deg give the same optimized result.
    """

    print("\nPERIODIC HEADING CHECK")
    print("----------------------")

    result_0 = optimize_heading_with_retries(
        solver=solver,
        base_input=base_input,
        true_wind_speed=true_wind_speed,
        ship_speed=ship_speed,
        heading_deg=0.0,
        tether_force_max=tether_force_max,
        warm_start_deg=X0_DEG,
    )

    result_360 = optimize_heading_with_retries(
        solver=solver,
        base_input=base_input,
        true_wind_speed=true_wind_speed,
        ship_speed=ship_speed,
        heading_deg=360.0,
        tether_force_max=tether_force_max,
        warm_start_deg=result_0["x_opt_deg"],
    )

    dP = abs(result_0["P_equiv_traction"] - result_360["P_equiv_traction"])
    dFx = abs(result_0["Fx"] - result_360["Fx"])
    dT = abs(result_0["tether_force_ground"] - result_360["tether_force_ground"])

    print(f"P_equiv at 0 deg      : {result_0['P_equiv_traction'] / 1000.0:12.6f} kW")
    print(f"P_equiv at 360 deg    : {result_360['P_equiv_traction'] / 1000.0:12.6f} kW")
    print(f"|ΔP|                  : {dP:12.6e} W")
    print(f"|ΔFx|                 : {dFx:12.6e} N")
    print(f"|ΔT|                  : {dT:12.6e} N")

    if dP < 1.0 and dFx < 1.0e-2 and dT < 1.0e-2:
        print("PASS: heading periodicity is consistent.")
    else:
        print("CHECK: 0 deg and 360 deg are not identical.")


def local_grid_check_optimizer(
    rows,
    solver,
    base_input,
    true_wind_speed,
    ship_speed,
    tether_force_max,
    headings_to_check=None,
):
    """
    Check whether the optimizer result is locally optimal by sampling nearby
    points around the optimized solution.

    This does not prove global optimality, but it catches obvious optimizer
    failures and bad local minima.
    """

    if headings_to_check is None:
        headings_to_check = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    rows_by_heading = {
        round(float(r["heading_deg"]), 6): r
        for r in rows
    }

    azimuth_deltas = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    elevation_deltas = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    course_deltas = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])

    effective_tether_limit = tether_force_max - FORCE_SAFETY_MARGIN

    print("\nLOCAL GRID OPTIMALITY CHECK")
    print("---------------------------")

    for heading_deg in headings_to_check:
        heading_key = round(float(heading_deg), 6)

        if heading_key not in rows_by_heading:
            print(f"heading {heading_deg:7.2f} deg: skipped, not in sweep")
            continue

        row = rows_by_heading[heading_key]

        if not row["accepted_result"]:
            print(f"heading {heading_deg:7.2f} deg: skipped, optimizer result not accepted")
            continue

        x_opt = np.array(
            [
                row["optimized_azimuth_angle_deg"],
                row["optimized_elevation_angle_deg"],
                row["optimized_course_angle_deg"],
            ],
            dtype=float,
        )

        P_opt = float(row["optimized_P_equiv_traction"])

        best_P = -np.inf
        best_x = None
        feasible_count = 0

        for da in azimuth_deltas:
            for de in elevation_deltas:
                for dc in course_deltas:
                    x_test = x_opt + np.array([da, de, dc], dtype=float)

                    x_test[0] = np.clip(x_test[0], BOUNDS_DEG[0][0], BOUNDS_DEG[0][1])
                    x_test[1] = np.clip(x_test[1], BOUNDS_DEG[1][0], BOUNDS_DEG[1][1])
                    x_test[2] = np.clip(x_test[2], BOUNDS_DEG[2][0], BOUNDS_DEG[2][1])

                    result = evaluate_traction_operating_point_safe(
                        solver=solver,
                        base_input=base_input,
                        true_wind_speed=true_wind_speed,
                        ship_speed=ship_speed,
                        heading_deg=heading_deg,
                        x_deg=x_test,
                    )

                    if not result["evaluation_success"]:
                        continue

                    if result["Fx"] < -1.0e-6:
                        continue

                    if result["tether_force_ground"] > effective_tether_limit + FORCE_TOLERANCE:
                        continue

                    feasible_count += 1

                    if result["P_equiv_traction"] > best_P:
                        best_P = result["P_equiv_traction"]
                        best_x = x_test.copy()

        local_gain = best_P - P_opt

        print(
            f"heading {heading_deg:7.2f} deg | "
            f"P_opt={P_opt / 1000.0:9.3f} kW | "
            f"best local={best_P / 1000.0:9.3f} kW | "
            f"gain={local_gain:9.3f} W | "
            f"feasible samples={feasible_count:3d}"
        )

        if local_gain > 100.0:
            print(
                f"  CHECK: nearby grid found a better point at "
                f"[az, el, course] = {best_x}"
            )
        else:
            print("  PASS: no meaningfully better nearby point found.")


# =============================================================================
# Plotting
# =============================================================================

def plot_heading_sweep(rows):
    accepted_rows = [r for r in rows if r["accepted_result"]]

    if not accepted_rows:
        print("No accepted optimized results to plot.")
        return

    rows_sorted = sorted(rows, key=lambda r: r["heading_deg"])

    headings = np.array([r["heading_deg"] for r in rows_sorted])
    fixed_P = np.array([r["fixed_P_equiv_traction"] for r in rows_sorted])
    opt_P = np.array([r["optimized_P_equiv_traction"] for r in rows_sorted])
    improvement = np.array([r["improvement_P_equiv"] for r in rows_sorted])

    fixed_Fx = np.array([r["fixed_Fx"] for r in rows_sorted])
    opt_Fx = np.array([r["optimized_Fx"] for r in rows_sorted])

    opt_T = np.array([r["optimized_tether_force_ground"] for r in rows_sorted])
    T_max = np.array([r["tether_force_max"] for r in rows_sorted])
    T_eff = np.array([r["tether_force_effective_limit"] for r in rows_sorted])

    opt_azimuth = np.array([r["optimized_azimuth_angle_deg"] for r in rows_sorted])
    opt_elevation = np.array([r["optimized_elevation_angle_deg"] for r in rows_sorted])
    opt_course = np.array([r["optimized_course_angle_deg"] for r in rows_sorted])

    accepted_mask = np.array([r["accepted_result"] for r in rows_sorted], dtype=bool)

    plt.figure()
    plt.plot(headings, fixed_P / 1000.0, label="Fixed traction")
    plt.plot(headings, opt_P / 1000.0, label="Optimized traction")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.scatter(
        headings[~accepted_mask],
        opt_P[~accepted_mask] / 1000.0,
        marker="x",
        label="not accepted",
    )
    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("P_equiv,traction [kW]")
    plt.title(
        f"Fixed vs optimized traction benefit\n"
        f"True wind = {TRUE_WIND_SPEED:.1f} m/s, ship speed = {SHIP_SPEED:.1f} m/s"
    )
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(headings, improvement / 1000.0, label="Optimization improvement")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Improvement [kW]")
    plt.title("Optimized traction improvement over fixed operating point")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(headings, fixed_Fx, label="Fixed Fx")
    plt.plot(headings, opt_Fx, label="Optimized Fx")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Surge force Fx [N]")
    plt.title("Fixed vs optimized surge force")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(headings, opt_T, label="Optimized tether force")
    plt.plot(headings, T_max, linestyle="--", label="Physical tether limit")
    plt.plot(headings, T_eff, linestyle=":", label="Optimizer effective limit")
    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Tether force [N]")
    plt.title("Optimized tether force vs limit")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(headings, opt_azimuth, label="azimuth angle")
    plt.plot(headings, opt_elevation, label="elevation angle")
    plt.plot(headings, opt_course, label="course angle")
    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Optimized angle [deg]")
    plt.title("Optimized traction operating angles")
    plt.legend()
    plt.grid(True)

    plt.show()


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

    tether_force_max = constructor.sys_props.tether_force_max_limit

    base_input = PureTractionInput(
        tether_length=TETHER_LENGTH,
        elevation_angle=np.deg2rad(X0_DEG[1]),
        azimuth_angle=np.deg2rad(X0_DEG[0]),
        course_angle=np.deg2rad(X0_DEG[2]),
    )

    print("\nCONSTRAINED TRACTION OPTIMIZED HEADING SWEEP")
    print("--------------------------------------------")
    print(f"True wind speed   : {TRUE_WIND_SPEED:.3f} m/s")
    print(f"Ship speed        : {SHIP_SPEED:.3f} m/s")
    print(f"Tether length     : {TETHER_LENGTH:.3f} m")
    print(f"Tether force max  : {tether_force_max:.3f} N")
    print(f"Effective T limit : {tether_force_max - FORCE_SAFETY_MARGIN:.3f} N")
    print(f"Headings          : {len(SWEEP_HEADINGS)}")

    rows = run_heading_sweep(
        solver=solver,
        base_input=base_input,
        true_wind_speed=TRUE_WIND_SPEED,
        ship_speed=SHIP_SPEED,
        headings=SWEEP_HEADINGS,
        tether_force_max=tether_force_max,
    )

    save_results_to_csv(rows, CSV_OUTPUT_PATH)

    verify_heading_sweep(rows)

    periodic_heading_check(
        solver=solver,
        base_input=base_input,
        true_wind_speed=TRUE_WIND_SPEED,
        ship_speed=SHIP_SPEED,
        tether_force_max=tether_force_max,
    )

    local_grid_check_optimizer(
        rows=rows,
        solver=solver,
        base_input=base_input,
        true_wind_speed=TRUE_WIND_SPEED,
        ship_speed=SHIP_SPEED,
        tether_force_max=tether_force_max,
    )

    plot_heading_sweep(rows)


if __name__ == "__main__":
    main()