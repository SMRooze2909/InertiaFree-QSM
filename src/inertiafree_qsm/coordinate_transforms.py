#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordinate transformations for coupling the InertiaFree-QSM kite model
to a moving vessel.

Why this file is needed
-----------------------
The original QSM pumping model is formulated in a fixed ground reference frame.
For the moving-vessel wrapper, the QSM cycle is currently run using only the
apparent wind speed. This means that the QSM ground-frame x-axis is treated as
an apparent-wind-aligned reference axis.

However, the vessel has its own heading-dependent ship frame:

    x_ship = surge direction, positive forward
    y_ship = sway direction

To evaluate whether pumping helps or hurts propulsion, the cycle-averaged
tether force must be projected into the ship frame.

Important convention from qsm.py
--------------------------------
In qsm.py, KiteKinematics.azimuth_angle is the position angle of the kite
relative to the ground reference frame x-axis:

    x = r cos(elevation) cos(azimuth)
    y = r cos(elevation) sin(azimuth)
    z = r sin(elevation)

Therefore, azimuth_angle is NOT itself a force angle, but the kite position
direction from the ground station to the kite.

The tether force acting on the vessel is assumed to act along the tether
towards the kite. Therefore, the horizontal vessel force uses the same
horizontal direction as the kite position vector.

Transformation chain
--------------------
QSM apparent-wind-aligned frame
    -> global/inertial frame
    -> ship surge/sway frame

The QSM frame is aligned with apparent_wind_direction because the QSM cycle is
run with apparent wind speed and environment_state.downwind_direction = 0.
"""

import numpy as np


def wrap_angle_rad(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def horizontal_tether_force(tether_force_ground, elevation_angle):
    """
    Compute horizontal component of ground tether force.

    Parameters
    ----------
    tether_force_ground : float
        Tether force at ground/vessel connection [N].
    elevation_angle : float
        Elevation angle above horizontal [rad].

    Returns
    -------
    float
        Horizontal-plane force magnitude [N].
    """
    return tether_force_ground * np.cos(elevation_angle)


def qsm_kite_position_to_global_force(
    tether_force_ground,
    azimuth_angle_qsm,
    elevation_angle,
    apparent_wind_direction,
):
    """
    Convert QSM kite position/tether force into global horizontal force.

    Parameters
    ----------
    tether_force_ground : float
        Tether force magnitude at the ground/vessel connection [N].
    azimuth_angle_qsm : float
        QSM kite azimuth angle [rad]. In qsm.py this is the kite position
        direction from the ground station to the kite, measured from QSM x-axis.
    elevation_angle : float
        Kite/tether elevation angle above horizontal [rad].
    apparent_wind_direction : float
        Global direction the apparent wind is blowing toward [rad].

    Returns
    -------
    tuple[float, float]
        Fx_global, Fy_global [N].
    """

    F_horizontal = horizontal_tether_force(
        tether_force_ground=tether_force_ground,
        elevation_angle=elevation_angle,
    )

    # QSM x-axis is treated as aligned with apparent wind direction.
    # Azimuth is ground station -> kite.
    # Tether force on vessel points toward kite.
    force_direction_global = apparent_wind_direction + azimuth_angle_qsm

    Fx_global = F_horizontal * np.cos(force_direction_global)
    Fy_global = F_horizontal * np.sin(force_direction_global)

    return Fx_global, Fy_global


def global_force_to_ship_force(Fx_global, Fy_global, vessel_heading):
    """
    Rotate global horizontal force components into ship surge/sway axes.

    Parameters
    ----------
    Fx_global : float
        Global x-force [N].
    Fy_global : float
        Global y-force [N].
    vessel_heading : float
        Global direction of ship bow [rad].

    Returns
    -------
    tuple[float, float]
        Fx_ship, Fy_ship [N].
        Fx_ship positive = forward surge force.
    """

    c = np.cos(vessel_heading)
    s = np.sin(vessel_heading)

    Fx_ship = Fx_global * c + Fy_global * s
    Fy_ship = -Fx_global * s + Fy_global * c

    return Fx_ship, Fy_ship


def qsm_kite_position_to_ship_force(
    tether_force_ground,
    azimuth_angle_qsm,
    elevation_angle,
    apparent_wind_direction,
    vessel_heading,
):
    """
    Full transformation from QSM kite state to vessel surge/sway force.

    Returns
    -------
    tuple[float, float]
        Fx_ship, Fy_ship [N].
    """

    Fx_global, Fy_global = qsm_kite_position_to_global_force(
        tether_force_ground=tether_force_ground,
        azimuth_angle_qsm=azimuth_angle_qsm,
        elevation_angle=elevation_angle,
        apparent_wind_direction=apparent_wind_direction,
    )

    return global_force_to_ship_force(
        Fx_global=Fx_global,
        Fy_global=Fy_global,
        vessel_heading=vessel_heading,
    )