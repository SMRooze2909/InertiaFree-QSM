#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test vessel apparent wind utilities.

This script checks whether:
true wind + ship speed + ship heading -> apparent wind
behaves as expected.
"""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from inertiafree_qsm.vessel_coupling import (
    TrueWind,
    VesselMotion,
    compute_apparent_wind,
    relative_wind_angle_to_ship,
)


def test_heading_sweep(true_wind, ship_speed):
    print("\n=== TEST 1: Heading sweep ===")
    print("-----------------------------")

    headings_deg = [0, 45, 90, 135, 180, -90]

    for heading_deg in headings_deg:
        vessel = VesselMotion(
            speed=ship_speed,
            heading=np.deg2rad(heading_deg),
        )

        app = compute_apparent_wind(true_wind, vessel)
        rel_angle = relative_wind_angle_to_ship(app, vessel)

        print(f"Ship heading {heading_deg:+7.1f} deg")
        print(f"  Apparent wind speed      : {app.speed:8.3f} m/s")
        print(f"  Apparent wind direction  : {np.rad2deg(app.direction_to):8.3f} deg")
        print(f"  Relative wind angle      : {np.rad2deg(rel_angle):8.3f} deg")
        print("")


def test_ship_speed_sweep(true_wind):
    print("\n=== TEST 2: Ship speed sweep (heading = 0°) ===")
    print("----------------------------------------------")

    heading = 0.0
    speeds = [0, 2, 4, 6, 8]

    for V in speeds:
        vessel = VesselMotion(
            speed=V,
            heading=np.deg2rad(heading),
        )

        app = compute_apparent_wind(true_wind, vessel)

        expected = true_wind.speed - V

        print(f"Ship speed {V:5.1f} m/s")
        print(f"  Apparent wind speed : {app.speed:8.3f} m/s")
        print(f"  Expected (8 - V)    : {expected:8.3f} m/s")
        print("")


def test_zero_apparent_wind(true_wind):
    print("\n=== TEST 3: Zero apparent wind case ===")
    print("--------------------------------------")

    vessel = VesselMotion(
        speed=true_wind.speed,
        heading=0.0,
    )

    app = compute_apparent_wind(true_wind, vessel)

    print("Ship speed = wind speed, same direction")
    print(f"  Apparent wind speed : {app.speed:.6f} m/s")
    print("  Expected            : ~0 m/s\n")


def main():
    true_wind = TrueWind(
        speed=8.0,
        direction_to=np.deg2rad(0.0),  # wind blowing toward +x
    )

    ship_speed = 5.0

    print("\nVessel apparent wind test")
    print("==========================")
    print(f"True wind speed      : {true_wind.speed:.2f} m/s")
    print("True wind direction  : 0 deg, blowing toward +x")
    print(f"Reference ship speed : {ship_speed:.2f} m/s")

    # Run tests
    test_heading_sweep(true_wind, ship_speed)
    test_ship_speed_sweep(true_wind)
    test_zero_apparent_wind(true_wind)


if __name__ == "__main__":
    main()