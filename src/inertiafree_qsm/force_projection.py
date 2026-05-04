"""
force_projection.py

PURPOSE:
--------
Convert scalar tether force from QSM into ship-fixed forces (Fx, Fy).

The QSM returns:
    tether_force (scalar along tether)

This module converts it to:
    surge force (Fx)
    sway force (Fy)

CURRENT ROLE:
-------------
- Projects tether force into horizontal plane
- Rotates forces into vessel reference frame

NEXT STEPS:
-----------
- Include full 3D force decomposition (lift/drag-based instead of tether-only)
- Include yaw moment contribution (important for steering/leeway)
- Align sign conventions with OpenWAVES exactly

ASSUMPTION (current version):
----------------------------
- Only horizontal component of tether force is used
- No vessel leeway influence yet (pure geometric projection)
"""

import numpy as np
from .operating_point import VesselForces
from dataclasses import dataclass
from typing import Sequence, List


def project_tether_force_to_ship_axes(
    tether_force: float,
    elevation_angle: float,
    azimuth_angle: float,
    vessel_heading: float,
) -> VesselForces:

    horizontal_force = tether_force * np.cos(elevation_angle)

    fx_ground = horizontal_force * np.cos(azimuth_angle)
    fy_ground = horizontal_force * np.sin(azimuth_angle)

    c = np.cos(vessel_heading)
    s = np.sin(vessel_heading)

    surge = fx_ground * c + fy_ground * s
    sway = -fx_ground * s + fy_ground * c

    return VesselForces(
        surge=float(surge),
        sway=float(sway),
    )

    """
    Project scalar tether tension into vessel-fixed surge/sway forces.

    Assumptions
    -----------
    - Angles are in radians.
    - elevation_angle is measured upward from the horizontal plane.
    - azimuth_angle is measured in the ground/world frame from +x toward +y.
    - azimuth_angle points from the vessel attachment point toward the kite.
    - vessel_heading is measured in the same ground/world frame.
    - Returned force is the force applied by the tether on the vessel.

    Notes
    -----
    If azimuth_angle instead describes the force on the kite, the sign must be reversed.
    """



@dataclass
class CycleForceProjection:
    average_forces: VesselForces
    max_tether_force: float
    mean_tether_force: float
    time: List[float]
    Fx: List[float]
    Fy: List[float]
    tether_force: List[float]


def project_steady_state_to_ship_axes(
    kinematics,
    steady_state,
    vessel_heading: float,
) -> VesselForces:
    return project_tether_force_to_ship_axes(
        tether_force=steady_state.tether_force_ground,
        elevation_angle=kinematics.elevation_angle,
        azimuth_angle=kinematics.azimuth_angle,
        vessel_heading=vessel_heading,
    )


def project_cycle_forces_to_ship_axes(
    time: Sequence[float],
    kinematics: Sequence,
    steady_states: Sequence,
    vessel_heading: float,
) -> CycleForceProjection:

    if len(time) == 0:
        raise ValueError("time array is empty.")

    if len(kinematics) != len(steady_states):
        raise ValueError("kinematics and steady_states must have same length.")

    if len(time) != len(kinematics):
        raise ValueError("time, kinematics, and steady_states must have same length.")

    time_array = np.asarray(time, dtype=float)
    duration = time_array[-1] - time_array[0]

    if duration <= 0:
        raise ValueError("cycle duration must be positive.")

    Fx = []
    Fy = []
    tether_force = []

    for kin, ss in zip(kinematics, steady_states):
        forces = project_steady_state_to_ship_axes(
            kinematics=kin,
            steady_state=ss,
            vessel_heading=vessel_heading,
        )

        Fx.append(forces.surge)
        Fy.append(forces.sway)
        tether_force.append(float(ss.tether_force_ground))

    Fx_array = np.asarray(Fx)
    Fy_array = np.asarray(Fy)
    tether_array = np.asarray(tether_force)

    Fx_avg = np.trapezoid(Fx_array, time_array) / duration
    Fy_avg = np.trapezoid(Fy_array, time_array) / duration
    mean_tether = np.trapezoid(tether_array, time_array) / duration
    max_tether = np.max(tether_array)

    return CycleForceProjection(
        average_forces=VesselForces(
            surge=float(Fx_avg),
            sway=float(Fy_avg),
        ),
        max_tether_force=float(max_tether),
        mean_tether_force=float(mean_tether),
        time=time_array.tolist(),
        Fx=Fx_array.tolist(),
        Fy=Fy_array.tolist(),
        tether_force=tether_array.tolist(),
    )