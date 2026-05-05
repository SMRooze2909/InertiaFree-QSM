#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNCONSTRAINED traction diagnostic with moving-vessel apparent wind.

IMPORTANT:
This script temporarily disables the maximum tether-force constraint when the
traction solver fails because the tether force exceeds the limit.

Therefore, this is NOT a final constrained operational-envelope model.

Purpose:
- Verify apparent-wind coupling
- Verify coordinate transformation
- Inspect smooth force trends over all headings
- Generate diagnostic traction maps

Coordinate convention:
- QSM/traction solve is run in an apparent-wind-aligned frame.
- QSM azimuth_angle is kite position direction from ground station to kite.
- Tether force on vessel acts toward kite.
- Force transformation:
      QSM frame -> global frame -> ship frame

Metric:
    P_equiv_traction = Fx * V_ship

where:
    Fx > 0 helps propulsion
    Fx < 0 adds resistance
"""

import sys
from pathlib import Path
from dataclasses import replace
import csv

import numpy as np
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inertiafree_qsm import PowerCurveConstructor
from inertiafree_qsm.operating_point import VesselState, WindCondition
from inertiafree_qsm.pure_traction import PureTractionInput, PureTractionSolver
from inertiafree_qsm.coordinate_transforms import qsm_kite_position_to_ship_force
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

CSV_OUTPUT_PATH = RESULTS_DIR / "traction_moving_vessel_matrix_UNCONSTRAINED.csv"

COLOR_FX = "tab:red"
COLOR_FY = "tab:purple"
COLOR_TETHER_FORCE = "tab:orange"
COLOR_APPARENT_WIND = "tab:blue"
COLOR_EQUIV_POWER = "tab:green"


# ---------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------

def solve_traction_unconstrained_if_needed(
    solver,
    wind,
    traction_input,
    vessel_state,
):
    """
    Run traction solver.

    If the solver fails because the tether force exceeds the maximum limit,
    temporarily disable the max tether-force limit and rerun.

    This is diagnostic only.
    """

    try:
        result = solver.solve(wind, traction_input, vessel_state)
        return result, False

    except ValueError as e:
        error_text = str(e)

        if "tether force exceeds limit" not in error_text:
            raise e

        old_limit = solver.sys_props.tether_force_max_limit
        solver.sys_props.tether_force_max_limit = 1e12

        try:
            result = solver.solve(wind, traction_input, vessel_state)
        finally:
            solver.sys_props.tether_force_max_limit = old_limit

        return result, True


def solve_traction_apparent_wind_frame_unconstrained(
    solver,
    wind_speed_apparent,
    traction_input,
    ship_speed,
):
    """
    Run traction solver in apparent-wind-aligned QSM frame.

    Wind direction is set to zero because global rotation is done afterward
    using coordinate_transforms.py.
    """

    wind = WindCondition(
        speed=wind_speed_apparent,
        direction=0.0,
    )

    vessel_state = VesselState(
        speed=ship_speed,
        leeway_angle=0.0,
        heading=0.0,
    )

    return solve_traction_unconstrained_if_needed(
        solver=solver,
        wind=wind,
        traction_input=traction_input,
        vessel_state=vessel_state,
    )


def evaluate_heading(
    solver,
    base_input,
    true_wind_speed,
    ship_speed,
    heading_deg,
):
    """
    Evaluate one heading for unconstrained diagnostic traction.
    """

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

    # Keep kite azimuth fixed in the apparent-wind/QSM frame.
    traction_input = replace(
        base_input,
        azimuth_angle=base_input.azimuth_angle,
    )

    result, limit_was_ignored = solve_traction_apparent_wind_frame_unconstrained(
        solver=solver,
        wind_speed_apparent=app.speed,
        traction_input=traction_input,
        ship_speed=ship_speed,
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
        "success": True,
        "error_type": "",
        "error_message": "",
        "limit_was_ignored": limit_was_ignored,
        "true_wind_speed": true_wind_speed,
        "ship_speed": ship_speed,
        "heading_deg": heading_deg,
        "apparent_wind_speed": app.speed,
        "apparent_wind_direction_deg": np.rad2deg(app.direction_to),
        "kite_azimuth_qsm_deg": np.rad2deg(traction_input.azimuth_angle),
        "kite_elevation_deg": np.rad2deg(traction_input.elevation_angle),
        "course_angle_deg": np.rad2deg(traction_input.course_angle),
        "tether_force_ground": result.tether_force_ground,
        "Fx": Fx_ship,
        "Fy": Fy_ship,
        "P_equiv_traction": P_equiv_traction,
    }


def evaluate_heading_safe(
    solver,
    base_input,
    true_wind_speed,
    ship_speed,
    heading_deg,
):
    try:
        return evaluate_heading(
            solver=solver,
            base_input=base_input,
            true_wind_speed=true_wind_speed,
            ship_speed=ship_speed,
            heading_deg=heading_deg,
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
            "limit_was_ignored": False,
            "true_wind_speed": true_wind_speed,
            "ship_speed": ship_speed,
            "heading_deg": heading_deg,
            "apparent_wind_speed": app.speed,
            "apparent_wind_direction_deg": np.rad2deg(app.direction_to),
            "kite_azimuth_qsm_deg": np.nan,
            "kite_elevation_deg": np.nan,
            "course_angle_deg": np.nan,
            "tether_force_ground": np.nan,
            "Fx": np.nan,
            "Fy": np.nan,
            "P_equiv_traction": np.nan,
        }


# ---------------------------------------------------------------------
# Heading interval helpers
# ---------------------------------------------------------------------

def heading_intervals(headings, mask, step_deg=5.0):
    headings = np.asarray(headings, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    selected = np.sort(headings[mask])

    if len(selected) == 0:
        return []

    if 0.0 in selected and 360.0 in selected:
        selected = selected[selected != 360.0]

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

    if len(intervals) > 1:
        first_start, first_end = intervals[0]
        last_start, last_end = intervals[-1]

        if first_start <= step_deg * 0.5 and last_end >= 360.0 - step_deg * 1.5:
            intervals = [(last_start, first_end)] + intervals[1:-1]

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


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

def verify_results(results):
    print("\nUNCONSTRAINED TRACTION VERIFICATION CHECKS")
    print("------------------------------------------")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    limit_ignored = [r for r in successful if r["limit_was_ignored"]]

    print(f"Successful cases             : {len(successful)}")
    print(f"Failed cases                 : {len(failed)}")
    print(f"Cases above tether limit      : {len(limit_ignored)}")
    print("Note: above-limit cases were solved by temporarily disabling the tether limit.")

    groups = {}
    for r in successful:
        key = (r["true_wind_speed"], r["ship_speed"])
        groups.setdefault(key, []).append(r)

    for key, group in groups.items():
        true_wind_speed, ship_speed = key
        group = sorted(group, key=lambda r: r["heading_deg"])

        headings = np.array([r["heading_deg"] for r in group])
        V_app = np.array([r["apparent_wind_speed"] for r in group])
        T = np.array([r["tether_force_ground"] for r in group])
        Fx = np.array([r["Fx"] for r in group])
        P_equiv = np.array([r["P_equiv_traction"] for r in group])
        ignored = np.array([r["limit_was_ignored"] for r in group], dtype=bool)

        print(
            f"\nCase: true wind = {true_wind_speed:.1f} m/s, "
            f"ship speed = {ship_speed:.1f} m/s"
        )

        print(f"  Above-limit headings solved unconstrained: {np.count_nonzero(ignored)} / {len(group)}")

        if len(V_app) > 3 and np.std(T) > 1e-9:
            corr = np.corrcoef(V_app, T)[0, 1]
            print(f"  Corr(V_app, tether force): {corr: .3f}")

            if corr > 0.5:
                print("  PASS: tether force generally follows apparent wind.")
            else:
                print("  CHECK: tether force does not strongly follow apparent wind.")

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

        P_equiv_reconstructed = Fx * ship_speed
        max_error = np.nanmax(np.abs(P_equiv - P_equiv_reconstructed))

        print(f"  Max P_equiv formula error: {max_error: .6f} W")

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
            f"  Beneficial headings      : "
            f"{format_heading_intervals(beneficial_intervals)}"
        )
        print(
            f"  Harmful headings         : "
            f"{format_heading_intervals(harmful_intervals)}"
        )


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def save_results_to_csv(results, output_path):
    fieldnames = [
        "success",
        "error_type",
        "error_message",
        "limit_was_ignored",
        "true_wind_speed",
        "ship_speed",
        "heading_deg",
        "apparent_wind_speed",
        "apparent_wind_direction_deg",
        "kite_azimuth_qsm_deg",
        "kite_elevation_deg",
        "course_angle_deg",
        "tether_force_ground",
        "Fx",
        "Fy",
        "P_equiv_traction",
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

def plot_heading_sweep(results, true_wind_speed, ship_speed):
    group = [
        r for r in results
        if r["success"]
        and r["true_wind_speed"] == true_wind_speed
        and r["ship_speed"] == ship_speed
    ]

    if not group:
        print("No successful results to plot.")
        return

    group = sorted(group, key=lambda r: r["heading_deg"])

    headings = np.array([r["heading_deg"] for r in group])
    V_app = np.array([r["apparent_wind_speed"] for r in group])
    T = np.array([r["tether_force_ground"] for r in group])
    Fx = np.array([r["Fx"] for r in group])
    Fy = np.array([r["Fy"] for r in group])
    P_equiv = np.array([r["P_equiv_traction"] for r in group])
    limit_ignored = np.array([r["limit_was_ignored"] for r in group], dtype=bool)

    plt.figure()
    plt.plot(headings, Fx, label="Fx surge", color=COLOR_FX)
    plt.plot(headings, Fy, label="Fy sway", color=COLOR_FY)
    plt.axhline(0.0, linestyle="--", linewidth=1)

    if np.any(limit_ignored):
        plt.scatter(
            headings[limit_ignored],
            Fx[limit_ignored],
            marker="x",
            label="above tether limit",
            color="black",
        )

    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Force [N]")
    plt.title(
        f"UNCONSTRAINED traction force vs heading\n"
        f"True wind = {true_wind_speed:.1f} m/s, ship speed = {ship_speed:.1f} m/s"
    )
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(
        headings,
        P_equiv / 1000.0,
        label="P_equiv_traction = Fx V_ship",
        color=COLOR_EQUIV_POWER,
    )
    plt.axhline(0.0, linestyle="--", linewidth=1)

    if np.any(limit_ignored):
        plt.scatter(
            headings[limit_ignored],
            P_equiv[limit_ignored] / 1000.0,
            marker="x",
            label="above tether limit",
            color="black",
        )

    plt.xlabel("Ship heading ψ [deg]")
    plt.ylabel("Equivalent propulsion benefit [kW]")
    plt.title(
        f"UNCONSTRAINED traction equivalent benefit vs heading\n"
        f"True wind = {true_wind_speed:.1f} m/s, ship speed = {ship_speed:.1f} m/s"
    )
    plt.legend()
    plt.grid(True)

    fig, ax1 = plt.subplots()
    ax1.plot(headings, T, label="Tether force", color=COLOR_TETHER_FORCE)
    ax1.set_xlabel("Ship heading ψ [deg]")
    ax1.set_ylabel("Tether force [N]", color=COLOR_TETHER_FORCE)
    ax1.tick_params(axis="y", labelcolor=COLOR_TETHER_FORCE)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(headings, V_app, label="Apparent wind speed", color=COLOR_APPARENT_WIND)
    ax2.set_ylabel("Apparent wind speed [m/s]", color=COLOR_APPARENT_WIND)
    ax2.tick_params(axis="y", labelcolor=COLOR_APPARENT_WIND)

    plt.title("UNCONSTRAINED apparent wind and tether force vs heading")

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

    env_state = constructor.create_environment(cluster_id=1)

    solver = PureTractionSolver(
        sys_props=constructor.sys_props,
        env_state=env_state,
        steady_state_config=constructor.simulation_settings.get("steady_state"),
    )

    base_input = PureTractionInput(
        tether_length=500.0,
        elevation_angle=np.deg2rad(30.0),
        azimuth_angle=np.deg2rad(11.5),
        course_angle=np.deg2rad(93.0),
    )

    wind_speeds = [8.0, 10.0, 12.0, 15.0, 18.0]
    ship_speeds = [3.0, 5.0]
    sweep_headings = np.linspace(0, 360, 73)

    all_results = []

    print("\nUNCONSTRAINED TRACTION MOVING-VESSEL MATRIX RUN")
    print("------------------------------------------------")
    print("WARNING: tether-force limit is temporarily ignored when exceeded.")
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
                    solver=solver,
                    base_input=base_input,
                    true_wind_speed=true_wind_speed,
                    ship_speed=ship_speed,
                    heading_deg=float(heading_deg),
                )

                all_results.append(result)

            print("")

    save_results_to_csv(all_results, CSV_OUTPUT_PATH)
    verify_results(all_results)

    plot_heading_sweep(
        results=all_results,
        true_wind_speed=15.0,
        ship_speed=5.0,
    )


if __name__ == "__main__":
    main()