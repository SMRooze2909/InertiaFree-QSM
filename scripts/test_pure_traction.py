"""
test_pure_traction.py

PURPOSE:
--------
Quick test script to validate the pure traction solver.

CURRENT ROLE:
-------------
- Loads system + wind configuration
- Runs fixed-ground traction sanity checks
- Prints tether force, apparent wind, Fx, Fy
- Saves sweep results to CSV

IMPORTANT:
----------
This file is NOT part of the core model.
It is only for testing and debugging.
"""

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from inertiafree_qsm import PowerCurveConstructor
from inertiafree_qsm.operating_point import VesselState, WindCondition
from inertiafree_qsm.pure_traction import PureTractionInput, PureTractionSolver


PROJECT_ROOT = Path(__file__).parent.parent

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "data" / "kitepower V3_20.yml"
WIND_RESOURCE_PATH = PROJECT_ROOT / "data" / "wind_resource.yml"
SIMULATION_SETTINGS_PATH = PROJECT_ROOT / "data" / "simulation_settings.yml"
OUTPUT_CSV_PATH = PROJECT_ROOT / "results" / "pure_traction_sweep_results.csv"


def print_result(label, result):
    print(
        f"{label:<12} | "
        f"V_kite={result.wind_speed_at_kite:7.3f} m/s | "
        f"V_app={result.apparent_wind_speed:7.3f} m/s | "
        f"T={result.tether_force_ground:8.1f} N | "
        f"Fx={result.vessel_forces.surge:8.1f} N | "
        f"Fy={result.vessel_forces.sway:8.1f} N | "
        f"v_tan={result.kite_tangential_speed:7.3f} m/s"
    )


def build_row(label, wind, traction_input, vessel_state, result=None, status="success"):
    row = {
        "label": label,
        "wind_speed_ref_m_s": wind.speed,
        "wind_direction_rad": wind.direction,
        "tether_length_m": traction_input.tether_length,
        "elevation_deg": np.rad2deg(traction_input.elevation_angle),
        "azimuth_deg": np.rad2deg(traction_input.azimuth_angle),
        "course_angle_deg": np.rad2deg(traction_input.course_angle),
        "heading_deg": np.rad2deg(vessel_state.heading),
        "status": status,
    }

    if result is None:
        row.update({
            "wind_speed_at_kite_m_s": "",
            "apparent_wind_speed_m_s": "",
            "tether_force_ground_N": "",
            "aerodynamic_force_N": "",
            "kite_tangential_speed_m_s": "",
            "Fx_surge_N": "",
            "Fy_sway_N": "",
        })
    else:
        row.update({
            "wind_speed_at_kite_m_s": result.wind_speed_at_kite,
            "apparent_wind_speed_m_s": result.apparent_wind_speed,
            "tether_force_ground_N": result.tether_force_ground,
            "aerodynamic_force_N": result.aerodynamic_force,
            "kite_tangential_speed_m_s": result.kite_tangential_speed,
            "Fx_surge_N": result.vessel_forces.surge,
            "Fy_sway_N": result.vessel_forces.sway,
        })

    return row


def run_case(label, solver, wind, traction_input, vessel_state, rows):
    try:
        result = solver.solve(wind, traction_input, vessel_state)
        print_result(label, result)
        rows.append(build_row(label, wind, traction_input, vessel_state, result))

    except ValueError as e:
        status = f"failed: {e}"
        print(f"{label:<12} | FAILED: {e}")
        rows.append(build_row(label, wind, traction_input, vessel_state, status=status))


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

    base_wind = WindCondition(speed=8.0, direction=0.0)

    # Fixed-ground validation: speed/leeway are not used yet.
    # Heading is only used for force-axis rotation.
    base_vessel_state = VesselState(
        speed=0.0,
        leeway_angle=0.0,
        heading=0.0,
    )

    base_input = PureTractionInput(
        tether_length=500.0,
        elevation_angle=np.deg2rad(30.0),
        azimuth_angle=np.deg2rad(11.5),
        course_angle=np.deg2rad(93.0),
    )

    rows = []

    print("\nBASELINE")
    print("--------")
    run_case("baseline", solver, base_wind, base_input, base_vessel_state, rows)

    print("\nTEST 1 — Wind speed sweep")
    print("-------------------------")
    for ws in [4, 6, 8, 10, 12]:
        wind = WindCondition(speed=float(ws), direction=0.0)
        run_case(f"{ws:>4.1f} m/s", solver, wind, base_input, base_vessel_state, rows)

    print("\nTEST 2 — Azimuth sign check")
    print("---------------------------")
    for az_deg in [-20, -10, 0, 10, 20]:
        traction_input = replace(base_input, azimuth_angle=np.deg2rad(az_deg))
        run_case(f"az={az_deg:+}", solver, base_wind, traction_input, base_vessel_state, rows)

    print("\nTEST 3 — Elevation check")
    print("------------------------")
    for elev_deg in [20, 30, 40, 50, 60]:
        traction_input = replace(base_input, elevation_angle=np.deg2rad(elev_deg))
        run_case(f"el={elev_deg}", solver, base_wind, traction_input, base_vessel_state, rows)

    print("\nTEST 4 — Heading rotation check")
    print("-------------------------------")
    for heading_deg in [0, 30, 60, 90]:
        vessel_state = replace(base_vessel_state, heading=np.deg2rad(heading_deg))
        run_case(f"hdg={heading_deg}", solver, base_wind, base_input, vessel_state, rows)

    print("\nTEST 5 — Course angle check")
    print("---------------------------")
    for course_deg in [60, 75, 90, 105, 120]:
        traction_input = replace(base_input, course_angle=np.deg2rad(course_deg))
        run_case(f"chi={course_deg}", solver, base_wind, traction_input, base_vessel_state, rows)

    print("\nTEST 6 — Tether length check")
    print("----------------------------")
    for tether_length in [300, 400, 500, 600]:
        traction_input = replace(base_input, tether_length=float(tether_length))
        run_case(f"L={tether_length}", solver, base_wind, traction_input, base_vessel_state, rows)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print("\nSaved sweep results to:")
        print(OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()