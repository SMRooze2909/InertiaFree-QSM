"""
vessel_coupling.py

PURPOSE:
--------
Couple kite model (QSM) with vessel model (OpenWAVES / PPP).

This is the CORE of the thesis:
    two-way interaction between kite and vessel.

CURRENT ROLE:
-------------
(Not implemented yet)

This file will:
- iterate between kite and vessel
- enforce consistent apparent wind and leeway

COUPLING LOOP:
--------------
1. Guess vessel state (speed, leeway)
2. Compute kite forces using QSM
3. Send Fx/Fy to vessel model
4. Solve vessel equilibrium (PPP/OpenWAVES)
5. Update vessel state
6. Repeat until convergence

NEXT STEPS:
-----------
- Implement OpenWAVES adapter
- Add convergence criteria
- Add propulsion power calculation
- Add pumping mode coupling

IMPORTANT:
----------
This file must NOT contain:
- kite physics (QSM)
- detailed vessel physics

It only coordinates the interaction.
"""