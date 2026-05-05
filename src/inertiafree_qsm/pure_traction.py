"""
pure_traction.py

PURPOSE:
--------
Compute kite forces for a single traction operating point using QSM physics,
but WITHOUT running a pumping cycle.

This is the first step toward a coupled kite–vessel model.

CURRENT ROLE:
-------------
- Uses QSM steady-state solver
- Assumes fixed tether (reeling_factor = 0)
- Computes:
    - tether force
    - apparent wind
    - aerodynamic force
- Converts tether force into ship forces (Fx, Fy)

IMPORTANT LIMITATION:
--------------------
The QSM still assumes a FIXED ground point.
Vessel motion is NOT yet included in apparent wind.

NEXT STEPS:
-----------
1. Replace wind model with:
       apparent wind = true wind - vessel velocity
2. Include leeway in apparent wind computation
3. Couple with OpenWAVES (iterative loop)
4. Replace tether-based projection with aerodynamic force projection

FINAL GOAL:
-----------
Return consistent kite forces (Fx, Fy) that can be used
inside a vessel equilibrium solver (PPP / OpenWAVES).
"""

from dataclasses import dataclass

from .qsm import KiteKinematics, SteadyState
from .operating_point import VesselState, VesselForces, WindCondition
from .force_projection import project_tether_force_to_ship_axes


@dataclass
class PureTractionInput:
    tether_length: float
    elevation_angle: float
    azimuth_angle: float
    course_angle: float


@dataclass
class PureTractionResult:
    wind_speed_at_kite: float
    apparent_wind_speed: float
    tether_force_ground: float
    aerodynamic_force: float
    kite_tangential_speed: float
    vessel_forces: VesselForces


class PureTractionSolver:
    def __init__(self, sys_props, env_state, steady_state_config=None):
        self.sys_props = sys_props
        self.env_state = env_state
        self.steady_state_config = steady_state_config or {}

    def solve(self, wind, traction_input, vessel_state, enforce_tether_limit=True):

        self.env_state.set_reference_wind_speed(wind.speed)
        self.env_state.downwind_direction = wind.direction

        kin = KiteKinematics(
            straight_tether_length=traction_input.tether_length,
            azimuth_angle=traction_input.azimuth_angle,
            elevation_angle=traction_input.elevation_angle,
            course_angle=traction_input.course_angle,
        )

        self.env_state.calculate(kin.z)
        self.sys_props.update(traction_input.tether_length, kite_powered=True)

        ss = SteadyState(self.steady_state_config)

        # Key: fixed tether traction
        ss.control_settings = ("reeling_factor", 0.0)
        ss.find_state(self.sys_props, self.env_state, kin)
        max_force = getattr(self.sys_props, "tether_force_max_limit", None)
        if enforce_tether_limit and max_force is not None and ss.tether_force_ground > max_force:
            raise ValueError(
                f"Pure traction tether force exceeds limit: "
                f"{ss.tether_force_ground:.1f} N > {max_force:.1f} N"
    )
        vessel_forces = project_tether_force_to_ship_axes(
            tether_force=ss.tether_force_ground,
            elevation_angle=traction_input.elevation_angle,
            azimuth_angle=traction_input.azimuth_angle,
            vessel_heading=vessel_state.heading,
        )

        return PureTractionResult(
            wind_speed_at_kite=ss.wind_speed,
            apparent_wind_speed=ss.apparent_wind_speed,
            tether_force_ground=ss.tether_force_ground,
            aerodynamic_force=ss.aerodynamic_force,
            kite_tangential_speed=ss.kite_tangential_speed,
            vessel_forces=vessel_forces,
        )