#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized pumping-cycle wind-heading sweep with moving-vessel apparent wind.

This version:
- Computes only headings from 0 to 180 deg.
- Mirrors the 0-180 deg result to 180-360 deg for plotting only.
- Saves only the actually computed 0-180 deg results to CSV.
- Produces only one plot: a mirrored polar plot of positive P_equiv_pumping.
- Adds runtime logging per heading case and per multistart candidate.
- Adds candidate-level logging to diagnose slow starts and local optima.
- Adds pumping-quality diagnostics to detect traction-dominated pumping solutions.

Important:
- This version does NOT hard-timeout inside optimizer.optimize().
- The runtime logging measures how long candidates take after they return.
- A true hard timeout requires multiprocessing or modifying the optimizer internals.

Objective:
    maximize P_equiv_pumping = P_cycle + Fx_avg * V_ship

where:
    P_cycle       = cycle-averaged pumping power [W]
    Fx_avg        = cycle-averaged ship-frame surge force [N]
    V_ship        = ship speed [m/s]

Physical convention:
- QSM pumping cycle is run in its native apparent-wind-aligned frame.
- QSM x-axis is treated as aligned with apparent wind direction.
- QSM azimuth_angle is kite position direction from ground station to kite.
- Tether force on the vessel acts toward the kite.
- Force transformation:
      QSM frame -> global frame -> ship frame
"""

from __future__ import annotations

import sys
import csv
import io
import contextlib
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Path setup
# =============================================================================

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# Project imports
# =============================================================================

from inertiafree_qsm import PowerCurveConstructor
from inertiafree_qsm.coordinate_transforms import (
    qsm_kite_position_to_global_force,
    qsm_kite_position_to_ship_force,
)
from inertiafree_qsm.vessel_coupling import (
    TrueWind,
    VesselMotion,
    compute_apparent_wind,
)

from inertiafree_qsm.cycle_optimizer import CycleOptimizer



# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_20.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings.yml"

CSV_OUTPUT_PATH = RESULTS_DIR / "optimized_pumping_heading_wind_sweep.csv"
CSV_AUTOSAVE_PATH = RESULTS_DIR / "optimized_pumping_heading_wind_sweep_autosave.csv"
CSV_CANDIDATE_LOG_PATH = RESULTS_DIR / "optimized_pumping_candidate_log.csv"


# =============================================================================
# Sweep settings
# =============================================================================

SHIP_SPEED = 4.0 #this is the prescribed ship speed in m/s
CLUSTER_ID = 1 #Cluster 1 is one representative ERA5 wind-profile shape, normalized around the 200 m wind speed.

# Use 2 or 3 wind speeds for debugging/plotting.
TRUE_WIND_SPEEDS = np.array([14.0], dtype=float)
#TRUE_WIND_SPEEDS = np.array([10.0, 14.0, 18.0], dtype=float)

# =============================================================================
# Mirroring switch (RUNTIME SAVING PURPOSES)
# =============================================================================

MIRROR_RESULTS_FOR_PLOT = True
# True  = compute only 0–180 deg, mirror 180–360 deg for polar plotting 
# False = compute full 0–360 deg, no mirroring


# Angular resolution of the heading sweep, smaller step --> more detail and optimizer jitter between nearby headings
# Larger step --> smoother-looking plots but can hide local jumps or switching behavior.
# It affects plotting resolution and runtime
HEADING_STEP_DEG = 15.0

if MIRROR_RESULTS_FOR_PLOT:
    SWEEP_HEADINGS = np.arange(0.0, 181.0, HEADING_STEP_DEG)
else:
    SWEEP_HEADINGS = np.arange(0.0, 360.0, HEADING_STEP_DEG)


# Positive benefit plot: negative or failed/inactive results are plotted as zero.
PLOT_ONLY_POSITIVE_EQUIVALENT_BENEFIT = True


# =============================================================================
# Numerical / reporting settings
# =============================================================================

# Maximum number of different initial guesses tried for each wind/heading case.
# Higher values --> improved robustness and increased runtime.
# Too low can cause neighboring headings to jump between different local solutions.
MAX_MULTISTART_CANDIDATES = 3

# Caps the native optimizer iterations per candidate for faster debug sweeps.
# Higher values usually reduce noisy/jittery results but increase runtime.
# Too low can stop the optimizer before it reaches a stable optimum.
MAX_OPTIMIZER_ITERATIONS_DEBUG = 80

# Print detailed optimizer progress from scipy/QSM when True.
OPTIMIZER_VERBOSE = False

# Suppress native optimizer stdout when True, keeping the sweep progress readable.
QUIET_NATIVE_OPTIMIZER_OUTPUT = True
# Save the CSV after every evaluated case, so partial results survive interruption.
AUTOSAVE_AFTER_EACH_CASE = True
AUTOSAVE_CANDIDATE_LOG_AFTER_EACH_CASE = False

# Numerical tolerance [N] for checking tether-force constraint violations.
FORCE_TOLERANCE = 1.0
# Power threshold [W] below which a case is treated as effectively inactive.
# Increasing this hides very small-power cases in plots/results.
# It does not fix physics; it only filters negligible results.
# Too high may remove valid low-wind pumping cases.
ZERO_POWER_THRESHOLD = 0
GRAVITATIONAL_ACCELERATION = 9.80665
# Keep at 0.0 for now. Later set e.g. 100.0 W if you want to reject “valid force but negligible cycle power” cases.
# Useful for avoiding traction-like pumping solutions.
# Too high may reject realistic low-wind pumping cases.
MIN_P_CYCLE_FOR_VALID_PUMPING = 0

# Too high may reject otherwise feasible low-wind cases.
# Minimum stroke as fraction of max tether length.
# Example: 0.05 and max_tether_length = 600 m gives minimum 30 m stroke.
# Minimum allowed reel-out/reel-in stroke as a fraction of max tether length.
# Increasing this forces a larger pumping stroke, making the solution more physically pump-like.
# Too low can allow short-stroke/traction-like solutions and jitter.
MIN_TETHER_LENGTH_FRACTION_DIFFERENCE = 0.05

# Diagnostic only for now. This does NOT reject cases unless
# REJECT_TRACTION_DOMINATED_PUMPING is set to True.
MIN_PUMPING_RATIO_DIAGNOSTIC = 0.30
REJECT_TRACTION_DOMINATED_PUMPING = False

# Very small number to avoid division by zero in diagnostic ratios.
EPS_POWER = 1.0e-9


# =============================================================================
# Formatting helpers
# =============================================================================

def format_kw(value_w: float) -> str:
    if not np.isfinite(value_w):
        return "    nan"
    return f"{value_w / 1000.0:7.3f}"


def format_s(value_s: float) -> str:
    if not np.isfinite(value_s):
        return "   nan"
    return f"{value_s:6.1f}"


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value if np.isfinite(value) else default


# =============================================================================
# Pumping-quality diagnostics
# =============================================================================

def compute_pumping_quality_diagnostics(
    P_cycle: float,
    P_prop_equiv: float,
) -> dict[str, float | bool]:
    """
    Diagnose whether the accepted pumping solution is genuinely pumping-like
    or mostly traction-like.

    Definitions:
    - P_prop_abs: absolute propulsion-equivalent contribution |Fx_avg * V_ship|
    - pump_ratio: P_cycle / |P_prop_equiv|
    - P_prop_fraction_abs: |P_prop_equiv| / (|P_cycle| + |P_prop_equiv|)
    - traction_dominated_flag: True if propulsion term dominates too strongly

    Interpretation:
    - pump_ratio << 1 means the case is mainly force/traction benefit,
      not real cycle generation.
    - This is diagnostic by default, not a hard rejection.
    """

    P_cycle = float(P_cycle)
    P_prop_equiv = float(P_prop_equiv)

    P_prop_abs = abs(P_prop_equiv)
    P_cycle_abs = abs(P_cycle)

    if P_prop_abs > EPS_POWER:
        pump_ratio = P_cycle / P_prop_abs
    else:
        pump_ratio = np.inf

    denominator = P_cycle_abs + P_prop_abs
    if denominator > EPS_POWER:
        P_prop_fraction_abs = P_prop_abs / denominator
    else:
        P_prop_fraction_abs = np.nan

    traction_dominated_flag = bool(
        np.isfinite(pump_ratio)
        and pump_ratio < MIN_PUMPING_RATIO_DIAGNOSTIC
        and P_prop_abs > EPS_POWER
    )

    return {
        "P_prop_abs": float(P_prop_abs),
        "pump_ratio": float(pump_ratio),
        "P_prop_fraction_abs": float(P_prop_fraction_abs),
        "traction_dominated_flag": traction_dominated_flag,
    }


# =============================================================================
# Static take-off limit
# =============================================================================

def calculate_static_takeoff_wind_speed(sys_props: Any, env_state: Any) -> float:
    """
    Static take-off wind speed from lift balancing kite weight:

        0.5 * rho * C_L * S * v_sto**2 = m * g

    This is a simple low-wind activity filter.
    """

    kite_mass = safe_float(getattr(sys_props, "kite_mass", np.nan))
    kite_area = safe_float(getattr(sys_props, "kite_projected_area", np.nan))
    lift_coefficient = safe_float(
        getattr(sys_props, "kite_lift_coefficient_powered", np.nan)
    )
    air_density = safe_float(getattr(env_state, "rho_0", np.nan))

    if not np.isfinite(air_density):
        air_density = safe_float(getattr(env_state, "air_density", np.nan))

    values = {
        "kite_mass": kite_mass,
        "kite_projected_area": kite_area,
        "kite_lift_coefficient_powered": lift_coefficient,
        "air_density": air_density,
    }

    invalid = [
        name
        for name, value in values.items()
        if not np.isfinite(value) or value <= 0.0
    ]

    if invalid:
        raise ValueError(
            "Cannot calculate static take-off wind speed. "
            f"Invalid values: {', '.join(invalid)}"
        )

    return float(
        np.sqrt(
            2.0
            * GRAVITATIONAL_ACCELERATION
            * kite_mass
            / (air_density * lift_coefficient * kite_area)
        )
    )


# =============================================================================
# Cycle force extraction aligned with qsm.py
# =============================================================================

def extract_cycle_samples_from_kpi(kpi: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Extract cycle force and kite-position histories from the QSM pumping KPI.

    Aligned with qsm.py:
    - kpi['kinematics'] contains KiteKinematics objects
    - kpi['steady_states'] contains SteadyState objects
    - KiteKinematics.azimuth_angle [rad]
    - KiteKinematics.elevation_angle [rad]
    - SteadyState.tether_force_ground [N]
    """

    if "kinematics" not in kpi:
        raise KeyError("KPI does not contain 'kinematics'.")

    if "steady_states" not in kpi:
        raise KeyError("KPI does not contain 'steady_states'.")

    if "time" not in kpi:
        raise KeyError("KPI does not contain 'time'.")

    kinematics = list(kpi["kinematics"])
    steady_states = list(kpi["steady_states"])
    time_values = np.asarray(kpi["time"], dtype=float)

    if len(kinematics) == 0:
        raise ValueError("kpi['kinematics'] is empty.")

    if len(steady_states) == 0:
        raise ValueError("kpi['steady_states'] is empty.")

    n = min(len(kinematics), len(steady_states), len(time_values))

    if n <= 0:
        raise ValueError("No overlapping cycle samples found.")

    kinematics = kinematics[:n]
    steady_states = steady_states[:n]
    time_values = time_values[:n]

    tether_force_ground = np.array(
        [ss.tether_force_ground for ss in steady_states],
        dtype=float,
    )

    azimuth_angle_qsm = np.array(
        [kin.azimuth_angle for kin in kinematics],
        dtype=float,
    )

    elevation_angle = np.array(
        [kin.elevation_angle for kin in kinematics],
        dtype=float,
    )

    if not np.all(np.isfinite(tether_force_ground)):
        raise ValueError("Non-finite tether_force_ground values found.")

    if not np.all(np.isfinite(azimuth_angle_qsm)):
        raise ValueError("Non-finite azimuth_angle values found.")

    if not np.all(np.isfinite(elevation_angle)):
        raise ValueError("Non-finite elevation_angle values found.")

    if not np.all(np.isfinite(time_values)):
        raise ValueError("Non-finite time values found.")

    if len(time_values) > 1 and time_values[-1] <= time_values[0]:
        raise ValueError("Cycle time array is not increasing.")

    return {
        "time": time_values,
        "tether_force_ground": tether_force_ground,
        "azimuth_angle_qsm": azimuth_angle_qsm,
        "elevation_angle": elevation_angle,
    }


def time_average(values: np.ndarray, time_values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    time_values = np.asarray(time_values, dtype=float)

    if len(values) == 1:
        return float(values[0])

    duration = float(time_values[-1] - time_values[0])

    if duration <= 0.0:
        return float(np.mean(values))

    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(values, time_values)
    else:
        integral = np.trapz(values, time_values)

    return float(integral / duration)


def compute_cycle_average_ship_forces_from_kpi(
    kpi: dict[str, Any],
    apparent_wind_direction: float,
    vessel_heading: float,
) -> dict[str, float]:
    """
    Project cycle force history from QSM frame to ship axes and average over time.
    """

    samples = extract_cycle_samples_from_kpi(kpi)

    time_values = samples["time"]
    tether_force_ground = samples["tether_force_ground"]
    azimuth_angle_qsm = samples["azimuth_angle_qsm"]
    elevation_angle = samples["elevation_angle"]

    n = len(time_values)

    Fx_ship = np.zeros(n, dtype=float)
    Fy_ship = np.zeros(n, dtype=float)
    Fx_global = np.zeros(n, dtype=float)
    Fy_global = np.zeros(n, dtype=float)

    for i in range(n):
        Fx_global[i], Fy_global[i] = qsm_kite_position_to_global_force(
            tether_force_ground=tether_force_ground[i],
            azimuth_angle_qsm=azimuth_angle_qsm[i],
            elevation_angle=elevation_angle[i],
            apparent_wind_direction=apparent_wind_direction,
        )

        Fx_ship[i], Fy_ship[i] = qsm_kite_position_to_ship_force(
            tether_force_ground=tether_force_ground[i],
            azimuth_angle_qsm=azimuth_angle_qsm[i],
            elevation_angle=elevation_angle[i],
            apparent_wind_direction=apparent_wind_direction,
            vessel_heading=vessel_heading,
        )

    return {
        "Fx_avg": time_average(Fx_ship, time_values),
        "Fy_avg": time_average(Fy_ship, time_values),
        "Fx_global_avg": time_average(Fx_global, time_values),
        "Fy_global_avg": time_average(Fy_global, time_values),
        "mean_tether_force": time_average(tether_force_ground, time_values),
        "max_tether_force": float(np.max(tether_force_ground)),
        "n_force_samples": int(n),
    }


# =============================================================================
# Moving-vessel pumping optimizer
# =============================================================================

class MovingVesselPumpingOptimizer(CycleOptimizer):
    """
    Subclass of CycleOptimizer that maximizes:

        P_equiv_pumping = P_cycle + Fx_avg * V_ship

    instead of only:

        P_cycle
    """

    def __init__(
        self,
        simulation_settings,
        sys_props,
        env_state,
        ship_speed: float,
        apparent_wind_direction: float,
        vessel_heading: float,
    ):
        super().__init__(
            simulation_settings=simulation_settings,
            sys_props=sys_props,
            env_state=env_state,
        )

        self.ship_speed = float(ship_speed)
        self.apparent_wind_direction = float(apparent_wind_direction)
        self.vessel_heading = float(vessel_heading)
        self.objective_name = "P_equiv_pumping"

        self.last_var_names = []
        self.last_x_opt = None

    def _objective_metric(self, kpi: dict[str, Any]) -> tuple[float, dict[str, float]]:
        if not kpi.get("sim_successful", False):
            return 0.0, {}

        P_cycle = float(kpi["average_power"]["cycle"])

        force_data = compute_cycle_average_ship_forces_from_kpi(
            kpi=kpi,
            apparent_wind_direction=self.apparent_wind_direction,
            vessel_heading=self.vessel_heading,
        )

        Fx_avg = force_data["Fx_avg"]
        P_prop_equiv = Fx_avg * self.ship_speed
        P_prop_penalty = max(-P_prop_equiv, 0.0)
        P_equiv_pumping = P_cycle + P_prop_equiv

        quality = compute_pumping_quality_diagnostics(
            P_cycle=P_cycle,
            P_prop_equiv=P_prop_equiv,
        )

        objective_data = {
            "P_cycle": P_cycle,
            "Fx_avg": Fx_avg,
            "Fy_avg": force_data["Fy_avg"],
            "Fx_global_avg": force_data["Fx_global_avg"],
            "Fy_global_avg": force_data["Fy_global_avg"],
            "mean_tether_force": force_data["mean_tether_force"],
            "max_tether_force": force_data["max_tether_force"],
            "P_prop_equiv": P_prop_equiv,
            "P_prop_penalty": P_prop_penalty,
            "P_equiv_pumping": P_equiv_pumping,
            "n_force_samples": force_data["n_force_samples"],
            **quality,
        }

        return float(P_equiv_pumping), objective_data

    def _objective(self, x_unscaled, var_names):
        """
        Negative selected objective because SLSQP minimizes.
        """

        kpi = self._cached_run_cycle(x_unscaled, var_names)
        is_feasible = bool(kpi["sim_successful"])

        if is_feasible:
            objective_value, objective_data = self._objective_metric(kpi)
        else:
            objective_value = 0.0
            objective_data = {}

        cycle_power = (
            float(kpi["average_power"]["cycle"])
            if is_feasible
            else 0.0
        )

        self.history.append({
            "x": x_unscaled.copy(),
            "power": cycle_power,
            "objective_value": objective_value,
            "objective_name": self.objective_name,
            "feasible": is_feasible,
            **objective_data,
        })

        return -objective_value

    def _finalise_result(self, result, scaling, var_names):
        """
        Re-evaluate at full resolution and rescue based on P_equiv_pumping,
        not P_cycle alone.
        """

        x_opt = result.x * scaling

        kpi = self._run_cycle(
            x_opt,
            var_names,
            use_opt_timesteps=False,
        )

        if kpi["sim_successful"]:
            slsqp_objective, slsqp_data = self._objective_metric(kpi)
        else:
            slsqp_objective = 0.0
            slsqp_data = {}

        kpi["objective_value"] = slsqp_objective
        kpi["objective_name"] = self.objective_name
        kpi["objective_data"] = slsqp_data

        best_hist = max(
            (
                entry for entry in self.history
                if entry.get("feasible", False)
                and np.isfinite(entry.get("objective_value", np.nan))
            ),
            key=lambda entry: entry["objective_value"],
            default=None,
        )

        if best_hist is not None and not np.allclose(
            best_hist["x"],
            x_opt,
            rtol=0.0,
            atol=1.0e-6,
        ):
            kpi_hist = self._run_cycle(
                best_hist["x"],
                var_names,
                use_opt_timesteps=False,
            )

            if kpi_hist["sim_successful"]:
                hist_objective, hist_data = self._objective_metric(kpi_hist)

                if hist_objective > slsqp_objective:
                    x_opt = best_hist["x"]
                    kpi = kpi_hist
                    kpi["objective_value"] = hist_objective
                    kpi["objective_name"] = self.objective_name
                    kpi["objective_data"] = hist_data

        self.last_var_names = list(var_names)
        self.last_x_opt = np.asarray(x_opt, dtype=float)

        return kpi, x_opt


# =============================================================================
# Apparent wind helpers
# =============================================================================

def compute_case_apparent_wind(
    true_wind_speed: float,
    ship_speed: float,
    heading_deg: float,
):
    true_wind = TrueWind(
        speed=float(true_wind_speed),
        direction_to=np.deg2rad(0.0),
    )

    vessel_heading = np.deg2rad(float(heading_deg))

    vessel_motion = VesselMotion(
        speed=float(ship_speed),
        heading=vessel_heading,
    )

    apparent_wind = compute_apparent_wind(
        true_wind,
        vessel_motion,
    )

    return apparent_wind, vessel_heading


# =============================================================================
# General helpers
# =============================================================================

def get_tether_force_max(constructor: PowerCurveConstructor) -> float:
    value = getattr(constructor.sys_props, "tether_force_max_limit", np.inf)

    if value is None:
        return np.inf

    return float(value)


def get_max_tether_length(constructor: PowerCurveConstructor) -> float:
    value = getattr(constructor.sys_props, "max_tether_length", np.nan)

    if value is None:
        return np.nan

    return float(value)


def get_base_x0(simulation_settings: dict[str, Any]) -> np.ndarray:
    return np.array(
        simulation_settings["optimization"]["optimizer"]["x0"],
        dtype=float,
    )


def set_base_x0(
    simulation_settings: dict[str, Any],
    x0_base: np.ndarray,
) -> dict[str, Any]:
    settings = deepcopy(simulation_settings)
    settings["optimization"]["optimizer"]["x0"] = [
        float(v) for v in np.asarray(x0_base, dtype=float)
    ]
    return settings


# =============================================================================
# Multistart helpers
# =============================================================================

def clip_reeling_speeds_in_x0(
    x0_base: np.ndarray,
    simulation_settings: dict[str, Any],
    apparent_wind_speed: float,
    out_factor: float,
    in_factor: float,
) -> np.ndarray:
    """
    Build a physically reasonable x0 variation.

    Index convention follows CycleOptimizer:
        x0[0] = reeling_speed_out
        x0[1] = reeling_speed_in
    """

    x = np.asarray(x0_base, dtype=float).copy()
    bounds = simulation_settings["optimization"]["bounds"]

    if len(x) > 0 and "reeling_speed_out" in bounds:
        lo, hi = bounds["reeling_speed_out"]
        x[0] = np.clip(out_factor * apparent_wind_speed, lo, hi)

    if len(x) > 1 and "reeling_speed_in" in bounds:
        lo, hi = bounds["reeling_speed_in"]
        x[1] = np.clip(-abs(in_factor * apparent_wind_speed), lo, hi)

    return x


def active_solution_to_base_x0(
    x0_base: np.ndarray,
    var_names: list[str],
    x_opt: np.ndarray,
) -> np.ndarray:
    """
    Map active optimizer variables back to the base x0 layout for warm-starting.
    """

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


def unique_x0_candidates(
    candidates: list[np.ndarray],
    decimals: int = 8,
) -> list[np.ndarray]:
    unique = []
    seen = set()

    for x in candidates:
        x = np.asarray(x, dtype=float)
        key = tuple(np.round(x, decimals=decimals))

        if key not in seen:
            seen.add(key)
            unique.append(x)

    return unique


def build_x0_candidates(
    simulation_settings: dict[str, Any],
    apparent_wind_speed: float,
    warm_x0_base: np.ndarray | None = None,
) -> list[np.ndarray]:
    base = get_base_x0(simulation_settings)

    candidates = []

    if warm_x0_base is not None:
        candidates.append(np.asarray(warm_x0_base, dtype=float))

    candidates.append(base)

    candidates.append(
        clip_reeling_speeds_in_x0(
            base,
            simulation_settings,
            apparent_wind_speed,
            out_factor=0.10,
            in_factor=0.20,
        )
    )

    candidates.append(
        clip_reeling_speeds_in_x0(
            base,
            simulation_settings,
            apparent_wind_speed,
            out_factor=0.20,
            in_factor=0.30,
        )
    )

    candidates.append(
        clip_reeling_speeds_in_x0(
            base,
            simulation_settings,
            apparent_wind_speed,
            out_factor=0.30,
            in_factor=0.40,
        )
    )

    if len(base) > 4:
        for factor in [0.85, 1.15, 1.30]:
            x = base.copy()
            x[4:] = x[4:] * factor
            candidates.append(x)

    candidates = unique_x0_candidates(candidates)

    return candidates[:MAX_MULTISTART_CANDIDATES]


# =============================================================================
# One-case optimization
# =============================================================================

def build_inactive_low_apparent_wind_kpi(
    true_wind_speed: float,
    ship_speed: float,
    heading_deg: float,
    apparent_wind,
    min_apparent_wind_speed: float,
    warm_x0_base: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """
    Return a clean inactive result for low apparent wind cases.
    """

    kpi = {
        "case_status": "inactive_low_apparent_wind",
        "sim_successful": False,
        "case_successful": True,
        "case_error_message": "inactive_low_apparent_wind",
        "average_power": {
            "cycle": 0.0,
            "in": 0.0,
            "trans_riro": 0.0,
            "trans_rori": 0.0,
            "out": 0.0,
        },
        "duration": {
            "cycle": 0.0,
            "in": 0.0,
            "trans_riro": 0.0,
            "trans_rori": 0.0,
            "out": 0.0,
        },
        "optimization_result": None,
        "objective_value": 0.0,
        "objective_name": "P_equiv_pumping",
        "objective_data": {
            "P_cycle": 0.0,
            "Fx_avg": 0.0,
            "Fy_avg": 0.0,
            "Fx_global_avg": 0.0,
            "Fy_global_avg": 0.0,
            "mean_tether_force": 0.0,
            "max_tether_force": 0.0,
            "P_prop_equiv": 0.0,
            "P_prop_penalty": 0.0,
            "P_equiv_pumping": 0.0,
            "n_force_samples": 0,
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

    return kpi, warm_x0_base


def is_physically_usable_kpi(
    kpi: dict[str, Any],
    tether_force_max: float,
) -> bool:
    if kpi.get("case_status") == "inactive_low_apparent_wind":
        return False

    if not kpi.get("sim_successful", False):
        return False

    data = kpi.get("objective_data", {})

    required = [
        "P_cycle",
        "Fx_avg",
        "Fy_avg",
        "mean_tether_force",
        "max_tether_force",
        "P_prop_equiv",
        "P_equiv_pumping",
        "pump_ratio",
        "traction_dominated_flag",
    ]

    for key in required:
        if key not in data:
            return False

    numeric_required = [
        "P_cycle",
        "Fx_avg",
        "Fy_avg",
        "mean_tether_force",
        "max_tether_force",
        "P_prop_equiv",
        "P_equiv_pumping",
    ]

    for key in numeric_required:
        if not np.isfinite(data.get(key, np.nan)):
            return False

    if float(data.get("P_cycle", 0.0)) < MIN_P_CYCLE_FOR_VALID_PUMPING:
        return False

    if REJECT_TRACTION_DOMINATED_PUMPING:
        if bool(data.get("traction_dominated_flag", False)):
            return False

    if np.isfinite(tether_force_max):
        max_tether_force = float(data["max_tether_force"])

        if max_tether_force > tether_force_max + FORCE_TOLERANCE:
            return False

    return True


def add_runtime_metadata_to_results(
    results: list[dict[str, Any]],
    candidate_runtimes: list[float],
    case_runtime_s: float,
) -> None:
    """
    Attach case-level and candidate-level runtime metadata to every candidate KPI.
    """

    n_candidates = len(results)
    max_candidate_runtime_s = (
        float(max(candidate_runtimes)) if candidate_runtimes else 0.0
    )
    mean_candidate_runtime_s = (
        float(np.mean(candidate_runtimes)) if candidate_runtimes else 0.0
    )
    sum_candidate_runtime_s = (
        float(np.sum(candidate_runtimes)) if candidate_runtimes else 0.0
    )

    for candidate_index, kpi_result in enumerate(results):
        kpi_result["case_runtime_s"] = float(case_runtime_s)
        kpi_result["n_candidates_tried"] = int(n_candidates)
        kpi_result["candidate_index"] = int(candidate_index)
        kpi_result["candidate_runtime_s"] = (
            float(candidate_runtimes[candidate_index])
            if candidate_index < len(candidate_runtimes)
            else np.nan
        )
        kpi_result["max_candidate_runtime_s"] = max_candidate_runtime_s
        kpi_result["mean_candidate_runtime_s"] = mean_candidate_runtime_s
        kpi_result["sum_candidate_runtime_s"] = sum_candidate_runtime_s


def optimize_one_heading_case(
    constructor: PowerCurveConstructor,
    env_state: Any,
    true_wind_speed: float,
    ship_speed: float,
    heading_deg: float,
    tether_force_max: float,
    min_apparent_wind_speed: float,
    warm_x0_base: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray | None, list[dict[str, Any]]]:
    """
    Optimize one true-wind / ship-heading case.

    Returns:
        best_kpi, new_warm_x0, candidate_log_rows
    """

    case_start_time = time.perf_counter()
    candidate_runtimes: list[float] = []

    apparent_wind, vessel_heading = compute_case_apparent_wind(
        true_wind_speed=true_wind_speed,
        ship_speed=ship_speed,
        heading_deg=heading_deg,
    )

    if apparent_wind.speed < min_apparent_wind_speed:
        kpi, new_warm_x0 = build_inactive_low_apparent_wind_kpi(
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=heading_deg,
            apparent_wind=apparent_wind,
            min_apparent_wind_speed=min_apparent_wind_speed,
            warm_x0_base=warm_x0_base,
        )
        kpi["case_runtime_s"] = float(time.perf_counter() - case_start_time)
        return kpi, new_warm_x0, []

    x0_candidates = build_x0_candidates(
        simulation_settings=constructor.simulation_settings,
        apparent_wind_speed=apparent_wind.speed,
        warm_x0_base=warm_x0_base,
    )

    results = []

    for candidate_index, x0_base in enumerate(x0_candidates):
        candidate_start_time = time.perf_counter()
        optimizer = None

        try:
            settings = set_base_x0(
                simulation_settings=constructor.simulation_settings,
                x0_base=x0_base,
            )

            optimizer = MovingVesselPumpingOptimizer(
                simulation_settings=settings,
                sys_props=constructor.sys_props,
                env_state=env_state,
                ship_speed=ship_speed,
                apparent_wind_direction=apparent_wind.direction_to,
                vessel_heading=vessel_heading,
            )

            if QUIET_NATIVE_OPTIMIZER_OUTPUT:
                with contextlib.redirect_stdout(io.StringIO()):
                    kpi = optimizer.optimize(
                        wind_speed=apparent_wind.speed,
                        verbose=OPTIMIZER_VERBOSE,
                    )
            else:
                kpi = optimizer.optimize(
                    wind_speed=apparent_wind.speed,
                    verbose=OPTIMIZER_VERBOSE,
                )

            kpi["case_status"] = "evaluated"
            kpi["case_successful"] = True
            kpi["case_error_message"] = ""

        except Exception as exc:
            kpi = {
                "case_status": "exception",
                "sim_successful": False,
                "case_successful": False,
                "case_error_message": str(exc),
                "average_power": {
                    "cycle": 0.0,
                    "in": 0.0,
                    "trans_riro": 0.0,
                    "trans_rori": 0.0,
                    "out": 0.0,
                },
                "duration": {
                    "cycle": 0.0,
                    "in": 0.0,
                    "trans_riro": 0.0,
                    "trans_rori": 0.0,
                    "out": 0.0,
                },
                "optimization_result": None,
                "objective_value": np.nan,
                "objective_name": "P_equiv_pumping",
                "objective_data": {},
            }

        candidate_runtime_s = time.perf_counter() - candidate_start_time
        candidate_runtimes.append(float(candidate_runtime_s))

        kpi["candidate_index"] = int(candidate_index)
        kpi["candidate_runtime_s"] = float(candidate_runtime_s)
        kpi["true_wind_speed"] = float(true_wind_speed)
        kpi["ship_speed"] = float(ship_speed)
        kpi["heading_deg"] = float(heading_deg)
        kpi["apparent_wind_speed"] = float(apparent_wind.speed)
        kpi["min_apparent_wind_speed"] = float(min_apparent_wind_speed)
        kpi["apparent_wind_direction_deg"] = float(
            np.rad2deg(apparent_wind.direction_to)
        )

        if optimizer is not None:
            kpi["last_var_names"] = getattr(optimizer, "last_var_names", [])
            kpi["last_x_opt"] = getattr(optimizer, "last_x_opt", None)
        else:
            kpi["last_var_names"] = []
            kpi["last_x_opt"] = None

        results.append(kpi)

    case_runtime_s = time.perf_counter() - case_start_time
    add_runtime_metadata_to_results(
        results=results,
        candidate_runtimes=candidate_runtimes,
        case_runtime_s=case_runtime_s,
    )

    feasible_results = [
        r for r in results
        if is_physically_usable_kpi(r, tether_force_max)
    ]

    if feasible_results:
        best = max(
            feasible_results,
            key=lambda r: r["objective_data"]["P_equiv_pumping"],
        )
    else:
        evaluated_results = [
            r for r in results
            if r.get("case_successful", False)
            and r.get("sim_successful", False)
            and np.isfinite(r.get("objective_value", np.nan))
        ]

        if evaluated_results:
            best = max(
                evaluated_results,
                key=lambda r: r.get("objective_value", -np.inf),
            )
        else:
            best = results[0]

    selected_candidate_index = int(best.get("candidate_index", -1))
    selected_candidate_runtime_s = float(best.get("candidate_runtime_s", np.nan))

    best["selected_candidate_index"] = selected_candidate_index
    best["selected_candidate_runtime_s"] = selected_candidate_runtime_s

    candidate_log_rows = []
    best_objective = best.get("objective_data", {}).get("P_equiv_pumping", np.nan)
    best_x = best.get("last_x_opt", None)

    for r in results:
        selected_best = False
        if r.get("candidate_index", -999) == selected_candidate_index:
            selected_best = True

        candidate_log_rows.append(
            build_candidate_log_row(
                kpi=r,
                x0_base=x0_candidates[int(r.get("candidate_index", 0))],
                selected_best=selected_best,
                best_objective=best_objective,
                best_x=best_x,
            )
        )

    new_warm_x0 = None

    if (
        best.get("last_x_opt") is not None
        and best.get("last_var_names") is not None
        and len(best.get("last_var_names")) > 0
    ):
        base = get_base_x0(constructor.simulation_settings)
        new_warm_x0 = active_solution_to_base_x0(
            x0_base=base,
            var_names=best["last_var_names"],
            x_opt=np.asarray(best["last_x_opt"], dtype=float),
        )

    return best, new_warm_x0, candidate_log_rows


# =============================================================================
# Output rows
# =============================================================================

def build_output_row(
    kpi: dict[str, Any],
    tether_force_max: float,
    max_tether_length: float,
) -> dict[str, Any]:
    data = kpi.get("objective_data", {})

    opt_result = kpi.get("optimization_result", None)
    case_status = kpi.get("case_status", "unknown")
    inactive_low_apparent_wind = case_status == "inactive_low_apparent_wind"

    if inactive_low_apparent_wind:
        optimizer_success = True
        optimizer_message = "inactive_low_apparent_wind"
    elif opt_result is None:
        optimizer_success = bool(kpi.get("sim_successful", False))
        optimizer_message = "No active optimization variables or optimization not run."
    else:
        optimizer_success = bool(opt_result.success)
        optimizer_message = str(opt_result.message)

    physical_feasible = is_physically_usable_kpi(kpi, tether_force_max)
    accepted_result = bool(physical_feasible)

    accepted_despite_optimizer_message = bool(
        accepted_result and not optimizer_success
    )

    max_tether_force = data.get("max_tether_force", np.nan)

    if np.isfinite(tether_force_max) and np.isfinite(max_tether_force):
        physical_tether_violation = max(0.0, max_tether_force - tether_force_max)
        physical_tether_ok = bool(
            max_tether_force <= tether_force_max + FORCE_TOLERANCE
        )
        constraint_active = bool(
            np.isclose(
                max_tether_force,
                tether_force_max,
                rtol=0.0,
                atol=25.0,
            )
        )
    else:
        physical_tether_violation = 0.0
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
        "inactive_low_apparent_wind": inactive_low_apparent_wind,
        "case_successful": kpi.get("case_successful", False),
        "sim_successful": kpi.get("sim_successful", False),
        "optimizer_success": optimizer_success,
        "accepted_result": accepted_result,
        "physical_feasible": physical_feasible,
        "accepted_despite_optimizer_message": accepted_despite_optimizer_message,
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
        "tether_constraint_violation": physical_tether_violation,
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

    frac_start = row.get("opt_frac_start", np.nan)
    frac_end = row.get("opt_frac_end", np.nan)

    if np.isfinite(frac_start) and np.isfinite(frac_end):
        stroke_fraction = float(frac_start - frac_end)
        row["stroke_fraction"] = stroke_fraction

        if np.isfinite(max_tether_length):
            row["stroke_length_m"] = stroke_fraction * max_tether_length

    return row


def build_candidate_log_row(
    kpi: dict[str, Any],
    x0_base: np.ndarray,
    selected_best: bool,
    best_objective: float,
    best_x: np.ndarray | None,
) -> dict[str, Any]:
    data = kpi.get("objective_data", {})
    opt_result = kpi.get("optimization_result", None)

    if opt_result is None:
        optimizer_success = bool(kpi.get("sim_successful", False))
        optimizer_message = "No active optimization variables or optimization not run."
    else:
        optimizer_success = bool(opt_result.success)
        optimizer_message = str(opt_result.message)

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
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "case_error_message": kpi.get("case_error_message", ""),
        "objective_value": kpi.get("objective_value", np.nan),
        "best_objective": best_objective,
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

    x0_base = np.asarray(x0_base, dtype=float)
    for i, value in enumerate(x0_base):
        row[f"start_x0_{i}"] = float(value)

    var_names = kpi.get("last_var_names", [])
    x_opt = kpi.get("last_x_opt", None)

    if x_opt is not None:
        for name, value in zip(var_names, x_opt):
            row[f"opt_{name}"] = float(value)

    return row


# =============================================================================
# CSV helpers
# =============================================================================

def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    base_fields = [
        "true_wind_speed",
        "ship_speed",
        "heading_deg",
        "apparent_wind_speed",
        "min_apparent_wind_speed",
        "apparent_wind_direction_deg",
        "case_status",
        "inactive_low_apparent_wind",
        "case_successful",
        "sim_successful",
        "optimizer_success",
        "accepted_result",
        "physical_feasible",
        "accepted_despite_optimizer_message",
        "case_error_message",
        "optimizer_message",
        "case_runtime_s",
        "n_candidates_tried",
        "selected_candidate_index",
        "selected_candidate_runtime_s",
        "max_candidate_runtime_s",
        "mean_candidate_runtime_s",
        "sum_candidate_runtime_s",
    ]

    opt_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith("opt_")
        }
    )

    result_fields = [
        "P_cycle",
        "P_in",
        "P_trans_riro",
        "P_trans_rori",
        "P_out",
        "cycle_duration",
        "Fx_avg",
        "Fy_avg",
        "Fx_global_avg",
        "Fy_global_avg",
        "mean_tether_force",
        "max_tether_force",
        "tether_force_max",
        "tether_constraint_violation",
        "physical_tether_constraint_ok",
        "constraint_active",
        "P_prop_equiv",
        "P_prop_penalty",
        "P_equiv_pumping",
        "P_prop_abs",
        "pump_ratio",
        "P_prop_fraction_abs",
        "traction_dominated_flag",
        "n_force_samples",
        "max_tether_length",
        "stroke_fraction",
        "stroke_length_m",
    ]

    return base_fields + opt_fields + result_fields


def collect_candidate_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fixed_fields = [
        "true_wind_speed",
        "ship_speed",
        "heading_deg",
        "apparent_wind_speed",
        "apparent_wind_direction_deg",
        "candidate_index",
        "selected_best",
        "candidate_runtime_s",
        "case_runtime_s",
        "case_status",
        "case_successful",
        "sim_successful",
        "optimizer_success",
        "optimizer_message",
        "case_error_message",
        "objective_value",
        "best_objective",
        "P_cycle",
        "P_prop_equiv",
        "P_equiv_pumping",
        "P_prop_abs",
        "pump_ratio",
        "P_prop_fraction_abs",
        "traction_dominated_flag",
        "Fx_avg",
        "Fy_avg",
        "mean_tether_force",
        "max_tether_force",
    ]

    dynamic_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith("start_x0_") or key.startswith("opt_")
        }
    )

    return fixed_fields + dynamic_fields


def save_results_to_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    quiet: bool = False,
) -> None:
    if not rows:
        return

    fieldnames = collect_fieldnames(rows)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not quiet:
        print(f"\nSaved optimized pumping wind-heading sweep to:\n  {output_path}")


def save_candidate_log_to_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    quiet: bool = False,
) -> None:
    if not rows:
        return

    fieldnames = collect_candidate_fieldnames(rows)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not quiet:
        print(f"\nSaved pumping candidate log to:\n  {output_path}")


# =============================================================================
# Sweep functions
# =============================================================================

def print_progress(
    wind_speed: float,
    i: int,
    total: int,
    heading_deg: float,
    accepted_count: int,
    inactive_count: int,
    failed_count: int,
    row: dict[str, Any],
    best_power: float,
) -> None:
    if row.get("accepted_result", False):
        p_cycle = row.get("P_cycle", np.nan)
        p_prop = row.get("P_prop_equiv", np.nan)
        p_penalty = row.get("P_prop_penalty", np.nan)
        p_equiv = row.get("P_equiv_pumping", np.nan)
        fx_avg = row.get("Fx_avg", np.nan)
        pump_ratio = row.get("pump_ratio", np.nan)
    else:
        p_cycle = np.nan
        p_prop = np.nan
        p_penalty = np.nan
        p_equiv = np.nan
        fx_avg = np.nan
        pump_ratio = np.nan

    runtime_s = row.get("case_runtime_s", np.nan)
    n_candidates = row.get("n_candidates_tried", np.nan)
    max_candidate_runtime_s = row.get("max_candidate_runtime_s", np.nan)
    traction_dom = row.get("traction_dominated_flag", False)

    msg = (
        f"  Vw={wind_speed:5.1f} m/s | "
        f"{i:03d}/{total:03d} | "
        f"ψ={heading_deg:7.2f} deg | "
        f"t={format_s(runtime_s)}s | "
        f"cand={int(n_candidates) if np.isfinite(n_candidates) else 0:02d} | "
        f"tcand,max={format_s(max_candidate_runtime_s)}s | "
        f"accepted={accepted_count:03d} | "
        f"inactive={inactive_count:03d} | "
        f"failed={failed_count:03d} | "
        f"Pcyc={format_kw(p_cycle)} kW | "
        f"Pprop={format_kw(p_prop)} kW | "
        f"pen={format_kw(p_penalty)} kW | "
        f"Peq={format_kw(p_equiv)} kW | "
        f"Fx={fx_avg:9.1f} N | "
        f"ratio={pump_ratio:6.3f} | "
        f"tracdom={str(traction_dom):5s} | "
        f"best={format_kw(best_power)} kW"
    )

    sys.stdout.write("\r" + msg.ljust(360))
    sys.stdout.flush()


def run_heading_sweep(
    constructor: PowerCurveConstructor,
    env_state: Any,
    true_wind_speed: float,
    ship_speed: float,
    headings: np.ndarray,
    tether_force_max: float,
    max_tether_length: float,
    min_apparent_wind_speed: float,
    autosave_prefix_rows: list[dict[str, Any]] | None = None,
    autosave_prefix_candidate_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    candidate_rows = []

    warm_x0_base = None
    accepted_count = 0
    inactive_count = 0
    failed_count = 0
    best_power = -np.inf

    total = len(headings)

    for i, heading_deg in enumerate(headings, start=1):
        kpi, new_warm_x0, case_candidate_rows = optimize_one_heading_case(
            constructor=constructor,
            env_state=env_state,
            true_wind_speed=float(true_wind_speed),
            ship_speed=ship_speed,
            heading_deg=float(heading_deg),
            tether_force_max=tether_force_max,
            min_apparent_wind_speed=min_apparent_wind_speed,
            warm_x0_base=warm_x0_base,
        )

        row = build_output_row(
            kpi=kpi,
            tether_force_max=tether_force_max,
            max_tether_length=max_tether_length,
        )

        rows.append(row)
        candidate_rows.extend(case_candidate_rows)

        if row["accepted_result"]:
            accepted_count += 1
            warm_x0_base = new_warm_x0
            best_power = max(best_power, row["P_equiv_pumping"])
        elif row["inactive_low_apparent_wind"]:
            inactive_count += 1
        else:
            failed_count += 1

        if AUTOSAVE_AFTER_EACH_CASE:
            prefix_rows = autosave_prefix_rows if autosave_prefix_rows is not None else []
            save_results_to_csv(
                rows=prefix_rows + rows,
                output_path=CSV_AUTOSAVE_PATH,
                quiet=True,
            )

            prefix_candidate_rows = (
                autosave_prefix_candidate_rows
                if autosave_prefix_candidate_rows is not None
                else []
            )
            if AUTOSAVE_CANDIDATE_LOG_AFTER_EACH_CASE:
                save_candidate_log_to_csv(
                    rows=prefix_candidate_rows + candidate_rows,
                    output_path=CSV_CANDIDATE_LOG_PATH,
                    quiet=True,
                )

        print_progress(
            wind_speed=true_wind_speed,
            i=i,
            total=total,
            heading_deg=float(heading_deg),
            accepted_count=accepted_count,
            inactive_count=inactive_count,
            failed_count=failed_count,
            row=row,
            best_power=best_power,
        )

    print("")
    return rows, candidate_rows


def run_wind_heading_sweep(
    constructor: PowerCurveConstructor,
    env_state: Any,
    true_wind_speeds: np.ndarray,
    ship_speed: float,
    headings: np.ndarray,
    tether_force_max: float,
    max_tether_length: float,
    min_apparent_wind_speed: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    all_candidate_rows = []

    for true_wind_speed in true_wind_speeds:
        print(
            f"\nRunning optimized pumping sweep: "
            f"true wind = {true_wind_speed:.1f} m/s, "
            f"ship speed = {ship_speed:.1f} m/s"
        )

        rows, candidate_rows = run_heading_sweep(
            constructor=constructor,
            env_state=env_state,
            true_wind_speed=float(true_wind_speed),
            ship_speed=ship_speed,
            headings=headings,
            tether_force_max=tether_force_max,
            max_tether_length=max_tether_length,
            min_apparent_wind_speed=min_apparent_wind_speed,
            autosave_prefix_rows=all_rows,
            autosave_prefix_candidate_rows=all_candidate_rows,
        )

        all_rows.extend(rows)
        all_candidate_rows.extend(candidate_rows)

    return all_rows, all_candidate_rows


# =============================================================================
# Summary
# =============================================================================

def print_compact_summary(rows: list[dict[str, Any]]) -> None:
    print("\nCOMPACT SUMMARY")
    print("---------------")

    wind_speeds = sorted(set(float(r["true_wind_speed"]) for r in rows))

    for wind_speed in wind_speeds:
        group = [
            r for r in rows
            if np.isclose(float(r["true_wind_speed"]), wind_speed)
        ]

        accepted = [r for r in group if r.get("accepted_result", False)]
        inactive = [r for r in group if r.get("inactive_low_apparent_wind", False)]
        failed = [
            r for r in group
            if not r.get("accepted_result", False)
            and not r.get("inactive_low_apparent_wind", False)
        ]
        traction_dominated = [
            r for r in group
            if bool(r.get("traction_dominated_flag", False))
        ]

        if accepted:
            best = max(accepted, key=lambda r: r.get("P_equiv_pumping", -np.inf))
            max_p = best["P_equiv_pumping"] / 1000.0
            best_heading = best["heading_deg"]
        else:
            max_p = np.nan
            best_heading = np.nan

        runtimes = [
            float(r.get("case_runtime_s", np.nan))
            for r in group
            if np.isfinite(r.get("case_runtime_s", np.nan))
        ]
        total_runtime = float(np.sum(runtimes)) if runtimes else np.nan
        max_runtime = float(np.max(runtimes)) if runtimes else np.nan

        print(
            f"Vw={wind_speed:5.1f} m/s | "
            f"accepted={len(accepted):02d} | "
            f"inactive={len(inactive):02d} | "
            f"failed={len(failed):02d} | "
            f"tracdom={len(traction_dominated):02d} | "
            f"best Peq={max_p:8.3f} kW at ψ={best_heading:7.2f} deg | "
            f"runtime total={total_runtime:7.1f}s | "
            f"runtime max={max_runtime:6.1f}s"
        )

    all_runtimes = [
        float(r.get("case_runtime_s", np.nan))
        for r in rows
        if np.isfinite(r.get("case_runtime_s", np.nan))
    ]

    if all_runtimes:
        print(
            f"\nTotal sweep runtime from logged cases: "
            f"{np.sum(all_runtimes):.1f} s "
            f"({np.sum(all_runtimes) / 60.0:.2f} min)"
        )


def print_slowest_cases(rows: list[dict[str, Any]], n: int = 10) -> None:
    """
    Print the slowest accepted/failed selected cases.
    """

    valid = [
        r for r in rows
        if np.isfinite(r.get("case_runtime_s", np.nan))
    ]

    if not valid:
        return

    slowest = sorted(
        valid,
        key=lambda r: float(r.get("case_runtime_s", -np.inf)),
        reverse=True,
    )[:n]

    print("\nSLOWEST SELECTED CASES")
    print("----------------------")
    print(
        " Vw | heading | runtime | cand | max cand | Vapp | accepted | status | Peq [kW] | Pcyc [kW] | Pprop [kW] | ratio | tracdom | constraint"
    )

    for r in slowest:
        print(
            f"{float(r.get('true_wind_speed', np.nan)):4.1f} | "
            f"{float(r.get('heading_deg', np.nan)):7.1f} | "
            f"{float(r.get('case_runtime_s', np.nan)):7.1f} | "
            f"{int(r.get('n_candidates_tried', 0)) if np.isfinite(r.get('n_candidates_tried', np.nan)) else 0:4d} | "
            f"{float(r.get('max_candidate_runtime_s', np.nan)):8.1f} | "
            f"{float(r.get('apparent_wind_speed', np.nan)):5.2f} | "
            f"{str(r.get('accepted_result', False)):8s} | "
            f"{str(r.get('case_status', '')):9s} | "
            f"{float(r.get('P_equiv_pumping', np.nan)) / 1000.0:8.3f} | "
            f"{float(r.get('P_cycle', np.nan)) / 1000.0:8.3f} | "
            f"{float(r.get('P_prop_equiv', np.nan)) / 1000.0:9.3f} | "
            f"{float(r.get('pump_ratio', np.nan)):6.3f} | "
            f"{str(r.get('traction_dominated_flag', False)):7s} | "
            f"{str(r.get('constraint_active', False)):10s}"
        )


def print_slowest_candidates(candidate_rows: list[dict[str, Any]], n: int = 15) -> None:
    """
    Print slowest candidate starts, not only selected best cases.
    This is important for diagnosing whether one bad multistart is causing runtime.
    """

    valid = [
        r for r in candidate_rows
        if np.isfinite(r.get("candidate_runtime_s", np.nan))
    ]

    if not valid:
        return

    slowest = sorted(
        valid,
        key=lambda r: float(r.get("candidate_runtime_s", -np.inf)),
        reverse=True,
    )[:n]

    print("\nSLOWEST CANDIDATES")
    print("------------------")
    print(
        " Vw | heading | cand | selected | runtime | success | status | Peq [kW] | Pcyc [kW] | Pprop [kW] | ratio | tracdom | message"
    )

    for r in slowest:
        message = str(r.get("optimizer_message", ""))[:45]
        print(
            f"{float(r.get('true_wind_speed', np.nan)):4.1f} | "
            f"{float(r.get('heading_deg', np.nan)):7.1f} | "
            f"{int(r.get('candidate_index', -1)):4d} | "
            f"{str(r.get('selected_best', False)):8s} | "
            f"{float(r.get('candidate_runtime_s', np.nan)):7.1f} | "
            f"{str(r.get('sim_successful', False)):7s} | "
            f"{str(r.get('case_status', '')):9s} | "
            f"{float(r.get('P_equiv_pumping', np.nan)) / 1000.0:8.3f} | "
            f"{float(r.get('P_cycle', np.nan)) / 1000.0:8.3f} | "
            f"{float(r.get('P_prop_equiv', np.nan)) / 1000.0:9.3f} | "
            f"{float(r.get('pump_ratio', np.nan)):6.3f} | "
            f"{str(r.get('traction_dominated_flag', False)):7s} | "
            f"{message}"
        )


# =============================================================================
# Polar plotting only
# =============================================================================

def close_polar_curve(theta: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    radius = np.asarray(radius, dtype=float)

    return (
        np.concatenate([theta, [theta[0] + 2.0 * np.pi]]),
        np.concatenate([radius, [radius[0]]]),
    )


def result_radius_kw(row: dict[str, Any]) -> float:
    """
    Radius used in polar plot.

    Failed, inactive, non-finite, and negative-equivalent-power cases are shown
    as zero because this plot represents positive equivalent pumping benefit.
    """

    if not row.get("accepted_result", False):
        return 0.0

    value_w = row.get("P_equiv_pumping", np.nan)

    if not np.isfinite(value_w):
        return 0.0

    value_kw = value_w / 1000.0

    if PLOT_ONLY_POSITIVE_EQUIVALENT_BENEFIT:
        return max(value_kw, 0.0)

    return value_kw


def make_rows_for_polar_plot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build plotting rows.

    If MIRROR_RESULTS_FOR_PLOT=True:
        computed heading h in (0, 180) is mirrored to 360 - h.

    If MIRROR_RESULTS_FOR_PLOT=False:
        rows are returned unchanged because the sweep already computed 0–360 deg.
    """

    if not MIRROR_RESULTS_FOR_PLOT:
        return [dict(row, mirrored_for_plot=False) for row in rows]

    plot_rows = []

    for row in rows:
        base = dict(row)
        base["mirrored_for_plot"] = False
        plot_rows.append(base)

        heading = float(row["heading_deg"]) % 360.0

        if 0.0 < heading < 180.0:
            mirrored = dict(row)
            mirrored["heading_deg"] = 360.0 - heading
            mirrored["mirrored_for_plot"] = True

            if np.isfinite(mirrored.get("Fy_avg", np.nan)):
                mirrored["Fy_avg"] = -float(mirrored["Fy_avg"])

            plot_rows.append(mirrored)

    return plot_rows


def plot_polar_positive_p_equiv_pumping(rows: list[dict[str, Any]]) -> None:
    plot_rows = make_rows_for_polar_plot(rows)

    if not plot_rows:
        print("No rows available for polar plot.")
        return

    wind_speeds = sorted(set(float(r["true_wind_speed"]) for r in plot_rows))

    fig, ax = plt.subplots(
        figsize=(9, 9),
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

        radii_kw = np.array(
            [result_radius_kw(r) for r in rows_sorted],
            dtype=float,
        )

        theta = np.deg2rad(headings)

        theta_closed, radii_closed = close_polar_curve(theta, radii_kw)

        if np.any(np.isfinite(radii_closed)):
            max_radius = max(max_radius, float(np.nanmax(radii_closed)))

        ax.plot(
            theta_closed,
            radii_closed,
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
        f"Positive equivalent pumping benefit\n"
        f"Ship speed = {SHIP_SPEED:.1f} m/s\n"
        f"{mirror_text}",
        pad=28,
    )

    ax.text(
        np.deg2rad(135),
        1.05 * max_radius if max_radius > 0.0 else 0.90,
        "P_equiv_pumping [kW]",
        ha="center",
        va="center",
    )

    ax.legend(
        title="True wind speed",
        loc="upper right",
        bbox_to_anchor=(1.35, 1.10),
    )

    ax.grid(True)
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    sweep_start_time = time.perf_counter()

    constructor = PowerCurveConstructor(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_file=False,
        verbose=False,
    )

    optimizer_settings = constructor.simulation_settings["optimization"]["optimizer"]
    optimizer_settings["max_iterations"] = min(
        int(optimizer_settings.get("max_iterations", 200)),
        MAX_OPTIMIZER_ITERATIONS_DEBUG,
    )

    constructor.simulation_settings["optimization"]["constraints"][
        "min_tether_length_fraction_difference"
    ] = MIN_TETHER_LENGTH_FRACTION_DIFFERENCE

    env_state = constructor.create_environment(cluster_id=CLUSTER_ID)

    min_apparent_wind_speed = calculate_static_takeoff_wind_speed(
        sys_props=constructor.sys_props,
        env_state=env_state,
    )

    tether_force_max = get_tether_force_max(constructor)
    max_tether_length = get_max_tether_length(constructor)

    max_generator_power = safe_float(
        getattr(constructor.sys_props, "max_generator_power", np.nan)
    )
    reeling_speed_min = safe_float(
        getattr(constructor.sys_props, "reeling_speed_min_limit", np.nan)
    )
    reeling_speed_max = safe_float(
        getattr(constructor.sys_props, "reeling_speed_max_limit", np.nan)
    )

    print(f"Max generator power  : {max_generator_power:.3f} W")
    print(f"Reeling speed min    : {reeling_speed_min:.3f} m/s")
    print(f"Reeling speed max    : {reeling_speed_max:.3f} m/s")

    print("\nOPTIMIZED PUMPING MOVING-VESSEL WIND-HEADING SWEEP")
    print("--------------------------------------------------")
    print(f"True wind speeds      : {TRUE_WIND_SPEEDS}")
    print(f"Ship speed            : {SHIP_SPEED:.3f} m/s")
    print(f"Computed headings     : {SWEEP_HEADINGS[0]:.1f}–{SWEEP_HEADINGS[-1]:.1f} deg")
    print(f"Heading step          : {HEADING_STEP_DEG:.3f} deg")
    print(f"Mirrored for plot     : {MIRROR_RESULTS_FOR_PLOT}")
    print(f"Multistart candidates : {MAX_MULTISTART_CANDIDATES}")
    print(f"Max optimizer iters   : {optimizer_settings['max_iterations']}")
    print(f"Min apparent wind     : {min_apparent_wind_speed:.3f} m/s")
    print("Min apparent wind source: static take-off formula")
    print(f"Min stroke fraction   : {MIN_TETHER_LENGTH_FRACTION_DIFFERENCE:.3f}")
    print(f"Min pumping ratio diag: {MIN_PUMPING_RATIO_DIAGNOSTIC:.3f}")
    print(f"Reject trac-dominated : {REJECT_TRACTION_DOMINATED_PUMPING}")

    if np.isfinite(tether_force_max):
        print(f"Tether force max      : {tether_force_max:.3f} N")
    else:
        print("Tether force max      : not available")

    if np.isfinite(max_tether_length):
        print(f"Max tether length     : {max_tether_length:.3f} m")
        print(
            f"Min stroke length     : "
            f"{MIN_TETHER_LENGTH_FRACTION_DIFFERENCE * max_tether_length:.3f} m"
        )
    else:
        print("Max tether length     : not available")

    print("\nObjective")
    print("---------")
    print("Maximize P_equiv_pumping = P_cycle + Fx_avg * V_ship")
    print("Positive Fx_avg helps propulsion; negative Fx_avg penalizes propulsion.")
    print("Progress shows runtime, candidate count, pump ratio and traction-dominated flag.")

    rows, candidate_rows = run_wind_heading_sweep(
        constructor=constructor,
        env_state=env_state,
        true_wind_speeds=TRUE_WIND_SPEEDS,
        ship_speed=SHIP_SPEED,
        headings=SWEEP_HEADINGS,
        tether_force_max=tether_force_max,
        max_tether_length=max_tether_length,
        min_apparent_wind_speed=min_apparent_wind_speed,
    )

    save_results_to_csv(
        rows=rows,
        output_path=CSV_OUTPUT_PATH,
    )
    save_candidate_log_to_csv(
        rows=candidate_rows,
        output_path=CSV_CANDIDATE_LOG_PATH,
    )

    print_compact_summary(rows)
    print_slowest_cases(rows, n=10)
    print_slowest_candidates(candidate_rows, n=15)

    sweep_runtime_s = time.perf_counter() - sweep_start_time
    print(
        f"\nWall-clock sweep runtime: {sweep_runtime_s:.1f} s "
        f"({sweep_runtime_s / 60.0:.2f} min)"
    )

    plot_polar_positive_p_equiv_pumping(rows)


if __name__ == "__main__":
    main()
