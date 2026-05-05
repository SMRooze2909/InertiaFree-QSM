# -*- coding: utf-8 -*-
"""
vessel_coupling.py

PURPOSE
-------
Small vessel-coupling utilities for kite-assisted ship modelling.

This file does NOT change the QSM kite physics.

It provides a clean interface for:
    1. Representing vessel motion in the same ground/world frame as the kite model
    2. Computing apparent wind from true wind and vessel velocity
    3. Later connecting kite forces Fx/Fy to a vessel/PPP/OpenWAVES model

WHY THIS FILE EXISTS
--------------------
The current QSM model is fixed-ground:
    true wind -> kite cycle -> tether force and power

For maritime use, the ship moves. Therefore the kite should eventually see:
    apparent wind = true wind - vessel velocity

This file is the first isolated step toward that coupling.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VesselMotion:
    """
    Vessel motion state in the ground/world reference frame.

    Parameters
    ----------
    speed : float
        Ship speed over ground [m/s].

    heading : float
        Ship heading [rad], measured in the ground/world frame.
        heading = 0 means the ship moves in +x direction.

    Notes
    -----
    This is intentionally simple:
    - no leeway yet
    - no current yet
    - no yaw rate yet
    - no wave-induced motion yet
    """

    speed: float
    heading: float


@dataclass
class TrueWind:
    """
    True wind condition in the ground/world reference frame.

    Parameters
    ----------
    speed : float
        True wind speed [m/s].

    direction_to : float
        Direction the wind is blowing TO [rad].
        direction_to = 0 means wind velocity points in +x direction.

    Important
    ---------
    Many nautical conventions describe wind direction as where the wind comes FROM.
    Here we use direction TO, because it directly defines the wind velocity vector.

    If you later use metocean data with wind direction FROM, convert it first:
        direction_to = direction_from + pi
    """

    speed: float
    direction_to: float


@dataclass
class ApparentWind:
    """
    Apparent wind result seen from the moving vessel.

    Parameters
    ----------
    speed : float
        Magnitude of apparent wind [m/s].

    direction_to : float
        Direction the apparent wind vector points TO [rad].

    vector_world : np.ndarray
        Apparent wind vector in ground/world axes [m/s].

    true_wind_vector_world : np.ndarray
        True wind vector in ground/world axes [m/s].

    vessel_velocity_world : np.ndarray
        Vessel velocity vector in ground/world axes [m/s].
    """

    speed: float
    direction_to: float
    vector_world: np.ndarray
    true_wind_vector_world: np.ndarray
    vessel_velocity_world: np.ndarray


def vector_from_speed_and_direction(speed: float, direction_to: float) -> np.ndarray:
    """
    Convert speed and direction into a 2D vector.

    This helper is used for both wind and vessel velocity.

    Convention
    ----------
    direction_to is measured from +x toward +y.

    Example
    -------
    speed = 8, direction_to = 0:
        vector = [8, 0]

    speed = 8, direction_to = 90 deg:
        vector = [0, 8]
    """

    return np.array(
        [
            speed * np.cos(direction_to),
            speed * np.sin(direction_to),
        ],
        dtype=float,
    )


def compute_vessel_velocity_world(vessel: VesselMotion) -> np.ndarray:
    """
    Compute the vessel velocity vector in the ground/world frame.

    Why this matters
    ----------------
    Apparent wind depends on the relative motion between air and ship.

    If the ship moves into the wind:
        apparent wind increases

    If the ship moves with the wind:
        apparent wind decreases
    """

    return vector_from_speed_and_direction(
        speed=vessel.speed,
        direction_to=vessel.heading,
    )


def compute_true_wind_vector_world(wind: TrueWind) -> np.ndarray:
    """
    Compute the true wind velocity vector in the ground/world frame.

    Important
    ---------
    wind.direction_to is the direction the wind velocity points TO.

    This avoids ambiguity in the vector equation:
        apparent wind = true wind - vessel velocity
    """

    return vector_from_speed_and_direction(
        speed=wind.speed,
        direction_to=wind.direction_to,
    )


def compute_apparent_wind(
    wind: TrueWind,
    vessel: VesselMotion,
) -> ApparentWind:
    """
    Compute apparent wind from true wind and vessel motion.

    Core equation
    -------------
        V_app = V_true_wind - V_vessel

    This is the wind velocity relative to the moving vessel.

    Physical interpretation
    -----------------------
    - Ship sailing with the wind:
        vessel velocity points in same direction as wind
        apparent wind speed decreases

    - Ship sailing against the wind:
        vessel velocity points opposite to wind
        apparent wind speed increases

    - Ship sailing across the wind:
        apparent wind direction rotates and speed changes according to vector addition

    Notes
    -----
    This does NOT yet feed apparent wind into the QSM.
    It only computes the relative wind vector cleanly and transparently.
    """

    true_wind_vector = compute_true_wind_vector_world(wind)
    vessel_velocity = compute_vessel_velocity_world(vessel)

    apparent_wind_vector = true_wind_vector - vessel_velocity

    apparent_speed = float(np.linalg.norm(apparent_wind_vector))

    apparent_direction_to = float(
        np.arctan2(apparent_wind_vector[1], apparent_wind_vector[0])
    )

    return ApparentWind(
        speed=apparent_speed,
        direction_to=apparent_direction_to,
        vector_world=apparent_wind_vector,
        true_wind_vector_world=true_wind_vector,
        vessel_velocity_world=vessel_velocity,
    )


def wrap_angle_rad(angle: float) -> float:
    """
    Wrap an angle to [-pi, pi].

    This is useful later when comparing:
        apparent wind direction
        ship heading
        kite azimuth

    Example
    -------
    190 deg becomes -170 deg.
    """

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def relative_wind_angle_to_ship(
    apparent_wind: ApparentWind,
    vessel: VesselMotion,
) -> float:
    """
    Compute apparent wind angle relative to ship heading.

    Returns
    -------
    float
        Relative apparent wind angle [rad].

    Convention
    ----------
    0 rad means apparent wind vector points in the same direction as the ship heading.
    pi rad or -pi rad means apparent wind vector points opposite to ship heading.

    Why this is useful
    ------------------
    This will later help classify conditions such as:
        - following apparent wind
        - beam apparent wind
        - head apparent wind

    For kite-assisted ships, this angle strongly affects whether kite forces help
    propulsion or create a propulsion penalty.
    """

    return wrap_angle_rad(apparent_wind.direction_to - vessel.heading)