#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test pumping-cycle force projection.

This script does not modify QSM pumping physics.
It runs one existing QSM pumping cycle and post-processes the tether force into
ship-fixed surge/sway loads for different vessel headings.
"""

import sys
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from inertiafree_qsm.qsm import Cycle, TractionPhase
from inertiafree_qsm.power_curve_constructor import PowerCurveConstructor
from inertiafree_qsm.force_projection import project_cycle_forces_to_ship_axes


def run_cycle(reference_wind_speed=8.0, cluster_id=1):
    kite_settings_file = repo_root / "data" / "kitepower V3_20.yml"
    wind_resource_file = repo_root / "data" / "wind_resource.yml"
    simulation_settings_file = repo_root / "data" / "simulation_settings.yml"

    pcc = PowerCurveConstructor(
        kite_settings_file,
        wind_resource_file,
        simulation_settings_file,
    )

    env_state = pcc.create_environment(cluster_id)
    env_state.set_reference_wind_speed(reference_wind_speed)

    settings = pcc.simulation_settings.copy()
    settings["cycle"] = pcc.simulation_settings["cycle"].copy()
    settings["cycle"]["traction_phase"] = TractionPhase

    steady_state_config = pcc.simulation_settings.get("steady_state")

    cycle = Cycle(
        settings,
        impose_operational_limits=True,
    )

    error_in_phase, _ = cycle.run_simulation(
        pcc.sys_props,
        env_state,
        steady_state_config,
        print_summary=False,
        enable_limit_violation_error=True,
    )

    if error_in_phase is not None:
        print(f"Warning: phase error detected: {error_in_phase}")

    return cycle


def export_csv(cycle, projection, heading_deg):
    output_dir = repo_root / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"pumping_force_projection_heading_{heading_deg:+.0f}deg.csv"

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "Fx_surge_N",
            "Fy_sway_N",
            "tether_force_N",
            "power_ground_W",
            "elevation_rad",
            "azimuth_rad",
            "reeling_speed_mps",
        ])

        for t, fx, fy, T, kin, ss in zip(
            projection.time,
            projection.Fx,
            projection.Fy,
            projection.tether_force,
            cycle.kinematics,
            cycle.steady_states,
        ):
            writer.writerow([
                t,
                fx,
                fy,
                T,
                ss.power_ground,
                kin.elevation_angle,
                kin.azimuth_angle,
                ss.reeling_speed,
            ])

    print(f"CSV exported to: {output_file}")


def plot_time_history(cycle, projection, heading_deg):
    time = np.asarray(projection.time)
    Fx = np.asarray(projection.Fx)
    Fy = np.asarray(projection.Fy)
    T = np.asarray(projection.tether_force)
    power = np.asarray([ss.power_ground for ss in cycle.steady_states])
    reel_speed = np.asarray([ss.reeling_speed for ss in cycle.steady_states])

    plt.figure()
    plt.plot(time, T, label="Tether force T")
    plt.plot(time, Fx, label="Fx / surge")
    plt.plot(time, Fy, label="Fy / sway")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title(f"Pumping-cycle force projection, heading = {heading_deg:+.0f}°")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, power, label="Ground power")
    plt.plot(time, reel_speed, label="Reeling speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Power [W] / Reeling speed [m/s]")
    plt.title("Pumping-cycle power and reeling speed")
    plt.grid(True)
    plt.legend()


def main():
    cluster_id = 1
    reference_wind_speed = 8.0

    cycle = run_cycle(
        reference_wind_speed=reference_wind_speed,
        cluster_id=cluster_id,
    )

    headings_deg = [0.0, 45.0, 90.0, 180.0, -90.0]

    print("\nPumping-cycle force projection heading sweep")
    print("--------------------------------------------")
    print(f"Reference wind speed : {reference_wind_speed:.2f} m/s")
    print(f"Cluster ID           : {cluster_id}")
    print(f"Cycle average power  : {cycle.average_power:.3f} W")
    print(f"Cycle duration       : {cycle.duration:.3f} s")
    print("")

    results = {}

    for heading_deg in headings_deg:
        projection = project_cycle_forces_to_ship_axes(
            time=cycle.time,
            kinematics=cycle.kinematics,
            steady_states=cycle.steady_states,
            vessel_heading=np.deg2rad(heading_deg),
        )

        results[heading_deg] = projection

        print(f"Heading {heading_deg:+7.1f} deg")
        print(f"  Average Fx / surge : {projection.average_forces.surge:10.3f} N")
        print(f"  Average Fy / sway  : {projection.average_forces.sway:10.3f} N")
        print(f"  Mean tether force  : {projection.mean_tether_force:10.3f} N")
        print(f"  Max tether force   : {projection.max_tether_force:10.3f} N")

        horizontal_avg = np.sqrt(
            projection.average_forces.surge**2
            + projection.average_forces.sway**2
        )

        print(f"  Avg horizontal mag : {horizontal_avg:10.3f} N")
        print("")

    # Export and plot the baseline heading.
    baseline_heading = 0.0
    baseline_projection = results[baseline_heading]

    export_csv(cycle, baseline_projection, baseline_heading)
    plot_time_history(cycle, baseline_projection, baseline_heading)

    # Simple rotation sanity check.
    p0 = results[0.0]
    p180 = results[180.0]

    print("\nRotation sanity check")
    print("---------------------")
    print("For a 180° vessel-heading change, Fx and Fy should approximately flip sign.")
    print(f"Fx(0 deg)    = {p0.average_forces.surge:.3f} N")
    print(f"Fx(180 deg)  = {p180.average_forces.surge:.3f} N")
    print(f"Fy(0 deg)    = {p0.average_forces.sway:.3f} N")
    print(f"Fy(180 deg)  = {p180.average_forces.sway:.3f} N")

    plt.show()


if __name__ == "__main__":
    main()