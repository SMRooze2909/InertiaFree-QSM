#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verification script for QSM pumping model with moving-vessel apparent wind.

Physical convention:
- QSM pumping cycle is run in its native frame.
- QSM x-axis is treated as aligned with apparent wind direction.
- QSM azimuth_angle is kite position direction from ground station to kite.
- Tether force on vessel acts toward kite.
- Force transformation:
      QSM frame -> global frame -> ship frame

Metric:
    P_equiv_pumping = P_cycle + Fx_avg * V_ship

where:
    Fx_avg > 0 helps propulsion
    Fx_avg < 0 adds resistance

This script:
- runs a wind-speed / ship-speed / heading matrix
- saves successful and failed cases to CSV
- performs basic physical sanity checks
"""

import sys
from pathlib import Path
from copy import deepcopy
import csv

import numpy as np
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inertiafree_qsm.coordinate_transforms import (
    qsm_kite_position_to_ship_force,
    qsm_kite_position_to_global_force,
)
from inertiafree_qsm.qsm import Cycle, TractionPhase
from inertiafree_qsm.power_curve_constructor import PowerCurveConstructor
from inertiafree_qsm.vessel_coupling import (
    TrueWind,
    VesselMotion,
    compute_apparent_wind,
)


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_20.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings.yml"

CSV_OUTPUT_PATH = RESULTS_DIR / "pumping_moving_vessel_matrix.csv"

COLOR_FX = "tab:red"
COLOR_FY = "tab:purple"
COLOR_APPARENT_WIND = "tab:blue"
COLOR_CYCLE_POWER = "tab:green"
COLOR_EQUIV_POWER = "tab:orange"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_value(obj, names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"Could not find any of these fields: {names}")


def time_average(time, values):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(time) < 2:
        return float(values[0])

    duration = time[-1] - time[0]
    if duration <= 0:
        return float(np.mean(values))

    return float(np.trapezoid(values, time) / duration)


def project_cycle_forces_to_ship_axes_apparent_wind_frame(
    time,
    kinematics,
    steady_states,
    apparent_wind_direction,
    vessel_heading,
    debug=False,
    debug_label="",
):
    Fx = []
    Fy = []
    tether_forces = []

    debug_indices = set()
    if debug and len(kinematics) > 0:
        debug_indices = {0, len(kinematics) // 2, len(kinematics) - 1}

    for i, (kin, ss) in enumerate(zip(kinematics, steady_states)):

        azimuth = float(get_value(
            kin,
            ["azimuth_angle", "azimuth", "phi", "azimuth_angle_rad"],
        ))

        elevation = float(get_value(
            kin,
            ["elevation_angle", "elevation", "beta", "elevation_angle_rad"],
        ))

        tether_force = float(get_value(
            ss,
            ["tether_force_ground", "tether_force", "force", "T"],
        ))

        if abs(azimuth) > 2.0 * np.pi:
            azimuth = np.deg2rad(azimuth)

        if abs(elevation) > 2.0 * np.pi:
            elevation = np.deg2rad(elevation)

        Fx_ship, Fy_ship = qsm_kite_position_to_ship_force(
            tether_force_ground=tether_force,
            azimuth_angle_qsm=azimuth,
            elevation_angle=elevation,
            apparent_wind_direction=apparent_wind_direction,
            vessel_heading=vessel_heading,
        )

        Fx_global, Fy_global = qsm_kite_position_to_global_force(
            tether_force_ground=tether_force,
            azimuth_angle_qsm=azimuth,
            elevation_angle=elevation,
            apparent_wind_direction=apparent_wind_direction,
        )

        Fx.append(Fx_ship)
        Fy.append(Fy_ship)
        tether_forces.append(tether_force)

        if i in debug_indices:
            force_direction_global = apparent_wind_direction + azimuth
            F_horizontal = tether_force * np.cos(elevation)

            print("\nCOORDINATE DEBUG", debug_label)
            print(f"  index                  : {i}")
            print(f"  time                   : {float(time[i]):8.3f} s")
            print(f"  azimuth_qsm            : {np.rad2deg(azimuth):8.3f} deg")
            print(f"  elevation              : {np.rad2deg(elevation):8.3f} deg")
            print(f"  apparent wind dir      : {np.rad2deg(apparent_wind_direction):8.3f} deg")
            print(f"  vessel heading         : {np.rad2deg(vessel_heading):8.3f} deg")
            print(f"  force dir global       : {np.rad2deg(force_direction_global):8.3f} deg")
            print(f"  tether force ground    : {tether_force:8.3f} N")
            print(f"  horizontal force       : {F_horizontal:8.3f} N")
            print(f"  Fx_global              : {Fx_global:8.3f} N")
            print(f"  Fy_global              : {Fy_global:8.3f} N")
            print(f"  Fx_ship surge          : {Fx_ship:8.3f} N")
            print(f"  Fy_ship sway           : {Fy_ship:8.3f} N")

    Fx = np.asarray(Fx)
    Fy = np.asarray(Fy)
    tether_forces = np.asarray(tether_forces)

    return {
        "Fx": Fx,
        "Fy": Fy,
        "Fx_avg": time_average(time, Fx),
        "Fy_avg": time_average(time, Fy),
        "mean_tether_force": time_average(time, tether_forces),
        "max_tether_force": float(np.max(tether_forces)),
    }


# ---------------------------------------------------------------------
# Pumping-cycle evaluation
# ---------------------------------------------------------------------

def run_pumping_cycle_for_apparent_wind(
    constructor,
    apparent_wind_speed,
    cluster_id=1,
):
    env_state = constructor.create_environment(cluster_id)
    env_state.set_reference_wind_speed(apparent_wind_speed)

    settings = deepcopy(constructor.simulation_settings)
    settings["cycle"]["traction_phase"] = TractionPhase

    steady_state_config = constructor.simulation_settings.get("steady_state")

    cycle = Cycle(
        settings,
        impose_operational_limits=True,
    )

    error_in_phase, _ = cycle.run_simulation(
        constructor.sys_props,
        env_state,
        steady_state_config,
        print_summary=False,
        enable_limit_violation_error=True,
    )

    return cycle, error_in_phase


def evaluate_heading(
    constructor,
    true_wind_speed,
    ship_speed,
    heading_deg,
    cluster_id=1,
    debug=False,
):
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

    cycle, error_in_phase = run_pumping_cycle_for_apparent_wind(
        constructor=constructor,
        apparent_wind_speed=app.speed,
        cluster_id=cluster_id,
    )

    projection = project_cycle_forces_to_ship_axes_apparent_wind_frame(
        time=cycle.time,
        kinematics=cycle.kinematics,
        steady_states=cycle.steady_states,
        apparent_wind_direction=app.direction_to,
        vessel_heading=vessel_heading,
        debug=debug,
        debug_label=(
            f"wind {true_wind_speed:.1f} m/s, "
            f"ship {ship_speed:.1f} m/s, "
            f"heading {heading_deg:.1f} deg"
        ),
    )

    Fx_avg = projection["Fx_avg"]
    Fy_avg = projection["Fy_avg"]
    P_cycle = cycle.average_power
    P_equiv_pumping = P_cycle + Fx_avg * ship_speed

    return {
        "success": True,
        "error_type": "",
        "error_message": "",
        "true_wind_speed": true_wind_speed,
        "ship_speed": ship_speed,
        "heading_deg": heading_deg,
        "apparent_wind_speed": app.speed,
        "apparent_wind_direction_deg": np.rad2deg(app.direction_to),
        "cycle_power": P_cycle,
        "cycle_duration": cycle.duration,
        "Fx_avg": Fx_avg,
        "Fy_avg": Fy_avg,
        "mean_tether_force": projection["mean_tether_force"],
        "max_tether_force": projection["max_tether_force"],
        "P_equiv_pumping": P_equiv_pumping,
        "error_in_phase": error_in_phase,
    }


def evaluate_heading_safe(
    constructor,
    true_wind_speed,
    ship_speed,
    heading_deg,
    cluster_id=1,
    debug=False,
):
    try:
        return evaluate_heading(
            constructor=constructor,
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=heading_deg,
            cluster_id=cluster_id,
            debug=debug,
        )
    except Exception as e:
        true_wind = TrueWind(
            speed=true_wind_speed,
            direction_to=np.deg2rad(0.0),
        )
        vessel_motion = VesselMotion(
            speed=ship_speed,
            heading=np.deg2rad(heading_deg),
        )
        app = compute_apparent_wind(true_wind, vessel_motion)

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "true_wind_speed": true_wind_speed,
            "ship_speed": ship_speed,
            "heading_deg": heading_deg,
            "apparent_wind_speed": app.speed,
            "apparent_wind_direction_deg": np.rad2deg(app.direction_to),
            "cycle_power": np.nan,
            "cycle_duration": np.nan,
            "Fx_avg": np.nan,
            "Fy_avg": np.nan,
            "mean_tether_force": np.nan,
            "max_tether_force": np.nan,
            "P_equiv_pumping": np.nan,
            "error_in_phase": "",
        }


# ---------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------
def heading_intervals(headings, mask, step_deg=5.0):
    """
    Convert heading mask into continuous heading intervals.

    Handles circular heading wrap-around, so e.g.
    0–60 and 300–360 becomes one wrapped interval:
        300–60 deg
    """
    headings = np.asarray(headings, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    selected = headings[mask]

    if len(selected) == 0:
        return []

    selected = np.sort(selected)

    # Remove duplicate 360 if 0 is also present
    if 0.0 in selected and 360.0 in selected:
        selected = selected[selected != 360.0]

    if len(selected) == 0:
        return [(0.0, 360.0)]

    intervals = []
    start = selected[0]
    prev = selected[0]

    for h in selected[1:]:
        if h - prev <= step_deg * 1.5:
            prev = h
        else:
            intervals.append((start, prev))
            start = h
            prev = h

    intervals.append((start, prev))

    # Merge circular wrap-around interval
    if len(intervals) > 1:
        first_start, first_end = intervals[0]
        last_start, last_end = intervals[-1]

        if first_start <= step_deg * 0.5 and last_end >= 360.0 - step_deg * 1.5:
            merged = (last_start, first_end)
            intervals = [merged] + intervals[1:-1]

    return intervals


def format_heading_intervals(intervals):
    if not intervals:
        return "none"

    return ", ".join(
        f"{start:.1f}–{end:.1f} deg"
        if start <= end
        else f"{start:.1f}–360.0 deg and 0.0–{end:.1f} deg"
        for start, end in intervals
    )

def verify_results(results):
    print("\nPHYSICAL VERIFICATION CHECKS")
    print("----------------------------")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Successful cases : {len(successful)}")
    print(f"Failed cases     : {len(failed)}")

    if failed:
        print("\nFailure summary:")
        failure_types = {}
        for r in failed:
            failure_types[r["error_type"]] = failure_types.get(r["error_type"], 0) + 1

        for error_type, count in failure_types.items():
            print(f"  {error_type}: {count}")

    groups = {}
    for r in successful:
        key = (r["true_wind_speed"], r["ship_speed"])
        groups.setdefault(key, []).append(r)

    for key, group in groups.items():
        true_wind_speed, ship_speed = key
        group = sorted(group, key=lambda r: r["heading_deg"])

        headings = np.array([r["heading_deg"] for r in group])
        V_app = np.array([r["apparent_wind_speed"] for r in group])
        P_cycle = np.array([r["cycle_power"] for r in group])
        Fx = np.array([r["Fx_avg"] for r in group])
        P_equiv = np.array([r["P_equiv_pumping"] for r in group])

        print(
            f"\nCase: true wind = {true_wind_speed:.1f} m/s, "
            f"ship speed = {ship_speed:.1f} m/s"
        )

        # Check 1: P_cycle should generally increase with apparent wind.
        if len(V_app) > 3 and np.std(P_cycle) > 1e-9:
            corr = np.corrcoef(V_app, P_cycle)[0, 1]
            print(f"  Corr(V_app, P_cycle)     : {corr: .3f}")

            if corr > 0.5:
                print("  PASS: P_cycle generally follows apparent wind.")
            else:
                print("  CHECK: P_cycle does not strongly follow apparent wind.")

        # Check 2: Fx should not jump unrealistically between neighboring headings.
        if len(Fx) > 3:
            dFx = np.abs(np.diff(Fx))
            max_dFx = np.nanmax(dFx)
            median_dFx = np.nanmedian(dFx)

            print(f"  Max |ΔFx| between headings: {max_dFx: .1f} N")
            print(f"  Med |ΔFx| between headings: {median_dFx: .1f} N")

            if median_dFx > 0 and max_dFx / median_dFx > 8:
                print("  CHECK: possible force discontinuity.")
            else:
                print("  PASS: Fx variation is reasonably smooth.")

        # Check 3: P_equiv relation should match definition.
        P_equiv_reconstructed = P_cycle + Fx * ship_speed
        max_error = np.nanmax(np.abs(P_equiv - P_equiv_reconstructed))

        print(f"  Max P_equiv formula error: {max_error: .6f} W")

        if max_error < 1e-6:
            print("  PASS: P_equiv formula is internally consistent.")
        else:
            print("  CHECK: P_equiv formula mismatch.")

        # Check 4: report beneficial/harmful heading range.
        step_deg = float(np.median(np.diff(np.sort(headings)))) if len(headings) > 1 else 5.0

        beneficial_intervals = heading_intervals(
            headings=headings,
            mask=P_equiv > 0.0,
            step_deg=step_deg,
        )

        harmful_intervals = heading_intervals(
            headings=headings,
            mask=P_equiv < 0.0,
            step_deg=step_deg,
        )

        print(
            "  Beneficial headings      : "
            f"{format_heading_intervals(beneficial_intervals)}"
        )

        print(
            f"  Harmful headings         : "
            f"{format_heading_intervals(harmful_intervals)}"
        )


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def save_results_to_csv(results, output_path):
    fieldnames = [
        "success",
        "error_type",
        "error_message",
        "true_wind_speed",
        "ship_speed",
        "heading_deg",
        "apparent_wind_speed",
        "apparent_wind_direction_deg",
        "cycle_power",
        "cycle_duration",
        "Fx_avg",
        "Fy_avg",
        "mean_tether_force",
        "max_tether_force",
        "P_equiv_pumping",
        "error_in_phase",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({key: r.get(key, "") for key in fieldnames})

    print(f"\nSaved results to:\n  {output_path}")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_heading_sweep(results, ship_speed, true_wind_speed):
    group = [
        r for r in results
        if r["success"]
        and r["ship_speed"] == ship_speed
        and r["true_wind_speed"] == true_wind_speed
    ]

    if not group:
        print("No successful results to plot.")
        return

    group = sorted(group, key=lambda r: r["heading_deg"])

    headings = np.array([r["heading_deg"] for r in group])
    apparent_wind = np.array([r["apparent_wind_speed"] for r in group])
    P_cycle = np.array([r["cycle_power"] for r in group])
    Fx_avg = np.array([r["Fx_avg"] for r in group])
    Fy_avg = np.array([r["Fy_avg"] for r in group])
    P_equiv = np.array([r["P_equiv_pumping"] for r in group])

    plt.figure()
    plt.plot(headings, P_cycle / 1000.0, label="P_cycle", color=COLOR_CYCLE_POWER)
    plt.plot(
        headings,
        P_equiv / 1000.0,
        label="P_equiv_pumping = P_cycle + Fx_avg V_ship",
        color=COLOR_EQUIV_POWER,
    )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Power [kW]")
    plt.title(
        f"Pumping equivalent benefit vs heading\n"
        f"True wind = {true_wind_speed:.1f} m/s, ship speed = {ship_speed:.1f} m/s"
    )
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(headings, Fx_avg, label="Fx_avg surge", color=COLOR_FX)
    plt.plot(headings, Fy_avg, label="Fy_avg sway", color=COLOR_FY)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Cycle-averaged force [N]")
    plt.title("Cycle-averaged pumping forces vs heading")
    plt.legend()
    plt.grid(True)

    fig, ax1 = plt.subplots()
    ax1.plot(
        headings,
        apparent_wind,
        label="Apparent wind speed",
        color=COLOR_APPARENT_WIND,
    )
    ax1.set_xlabel("Ship heading ψ [deg]")
    ax1.set_ylabel("Apparent wind speed [m/s]", color=COLOR_APPARENT_WIND)
    ax1.tick_params(axis="y", labelcolor=COLOR_APPARENT_WIND)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(headings, P_cycle / 1000.0, label="P_cycle", color=COLOR_CYCLE_POWER)
    ax2.set_ylabel("Cycle power [kW]", color=COLOR_CYCLE_POWER)
    ax2.tick_params(axis="y", labelcolor=COLOR_CYCLE_POWER)

    plt.title("Apparent wind and pumping cycle power vs heading")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    constructor = PowerCurveConstructor(
        system_config_path=SYSTEM_CONFIG_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        simulation_settings_path=SIMULATION_SETTINGS_PATH,
        validate_file=False,
        verbose=False,
    )

    cluster_id = 1

    wind_speeds = [8.0, 10.0, 12.0, 15.0, 18.0]
    ship_speeds = [3.0, 5.0]
    sweep_headings = np.linspace(0, 360, 73)

    all_results = []

    print("\nPUMPING MOVING-VESSEL MATRIX RUN")
    print("--------------------------------")
    print(f"Wind speeds : {wind_speeds}")
    print(f"Ship speeds : {ship_speeds}")
    print(f"Headings    : {len(sweep_headings)} cases per wind/ship speed\n")

    total_cases = len(wind_speeds) * len(ship_speeds) * len(sweep_headings)
    case_counter = 0

    for true_wind_speed in wind_speeds:
        for ship_speed in ship_speeds:
            print(
                f"\nRunning case group: "
                f"true wind = {true_wind_speed:.1f} m/s, "
                f"ship speed = {ship_speed:.1f} m/s"
            )

            for heading_deg in sweep_headings:
                case_counter += 1

                print(
                    f"  [{case_counter:03d}/{total_cases:03d}] "
                    f"heading {heading_deg:6.1f} deg",
                    end="\r",
                )

                result = evaluate_heading_safe(
                    constructor=constructor,
                    true_wind_speed=true_wind_speed,
                    ship_speed=ship_speed,
                    heading_deg=float(heading_deg),
                    cluster_id=cluster_id,
                    debug=False,
                )

                all_results.append(result)

            print("")

    save_results_to_csv(all_results, CSV_OUTPUT_PATH)
    verify_results(all_results)

    # Plot one representative case.
    plot_heading_sweep(
        results=all_results,
        ship_speed=5.0,
        true_wind_speed=15.0,
    )


if __name__ == "__main__":
    main()