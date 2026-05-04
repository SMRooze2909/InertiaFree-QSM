"""
operating_point.py

PURPOSE:
--------
Defines the data structures used to exchange information between:
- kite model (QSM)
- vessel model (OpenWAVES / PPP)

This file does NOT contain any physics. It only defines clean interfaces.

CURRENT ROLE:
-------------
- VesselCommand: prescribed inputs (speed + heading)
- VesselState: current/iterated vessel state (speed, leeway)
- VesselForces: forces from kite acting on vessel
- WindCondition: external wind input

NEXT STEPS:
-----------
- Extend VesselForces with yaw moment if needed
- Possibly add "propulsion power" output container

IMPORTANT:
----------
This file must stay lightweight and independent of both QSM and OpenWAVES.
"""

from dataclasses import dataclass


@dataclass
class VesselCommand:
    target_speed: float
    heading: float


@dataclass
class VesselState:
    speed: float
    leeway_angle: float
    heading: float


@dataclass
class VesselForces:
    surge: float
    sway: float
    yaw: float = 0.0


@dataclass
class WindCondition:
    speed: float
    direction: float