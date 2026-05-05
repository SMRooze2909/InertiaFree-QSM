readme_text = """# InertiaFree-QSM (Extended for Moving-Vessel Applications)

This repository is a fork and extension of the original InertiaFree-QSM model:
https://github.com/jbredael/InertiaFree-QSM

The original model implements a quasi-steady pumping-cycle simulation for airborne wind energy systems based on van der Vlugt et al. (2019) and Schelbergen (2024).

This fork extends the framework toward maritime applications, enabling modelling of kite systems on moving vessels with consistent traction and pumping evaluation.


EXTENSION SCOPE
---------------

Main additions:
- Moving-vessel apparent wind coupling
- Consistent traction and pumping evaluation
- Ship-frame force projection
- Equivalent propulsion-power metrics
- Heading-dependent operational maps


ADDED FILES AND SCRIPTS
-----------------------

scripts/

test_pumping_with_apparent_wind.py  
Runs the pumping-cycle model using apparent wind from a moving vessel.  
Projects cycle forces into ship axes and computes equivalent propulsion benefit.

test_traction_with_apparent_wind.py  
Evaluates traction mode under apparent wind conditions.  
Includes both constrained and unconstrained operation to diagnose force behavior.

pumping_moving_vessel_matrix.py  
Runs large parameter sweeps over:
- wind speed
- ship speed
- heading  
Outputs CSV files and performs physical consistency checks.

traction_moving_vessel_matrix.py  
Same as pumping matrix, but for traction mode.  
Includes unconstrained solver option to investigate force envelope and constraint effects.

test_pumping_force_projection.py  
Standalone verification of force projection logic from QSM frame to ship frame.

test_pure_traction.py  
Baseline traction solver test without vessel coupling.

test_vessel_coupling.py  
Validation script for apparent wind computation and vessel motion coupling.


src/inertiafree_qsm/

coordinate_transforms.py  
Core transformation logic:
- QSM frame -> global frame -> ship frame  
Ensures physically consistent mapping of tether forces.

vessel_coupling.py  
Implements:
- true wind definition
- vessel motion
- apparent wind computation

force_projection.py  
Utilities for projecting distributed and point forces into model DOFs or vessel axes.

operating_point.py  
Defines operating-point structures for traction calculations:
- vessel state
- wind condition

pure_traction.py  
Standalone quasi-steady traction solver:
- computes tether force
- computes resulting vessel forces


MOVING-VESSEL COUPLING
----------------------

The vessel motion is included through the apparent wind:

    V_app = V_true - V_ship

The QSM model is evaluated using the apparent wind speed and direction.


COORDINATE SYSTEM CONVENTION
----------------------------

The QSM model operates in an apparent-wind-aligned frame.

Transformation chain:

    QSM frame -> global frame -> ship frame

Key assumptions:
- QSM azimuth = kite position direction
- tether force acts from vessel toward kite
- ship axes:
    Fx = surge (forward)
    Fy = sway


PUMPING MODE
------------

Outputs:
- cycle power: P_cycle
- cycle-averaged forces: Fx_avg, Fy_avg

Equivalent propulsion benefit:

    P_equiv_pumping = P_cycle + Fx_avg * V_ship


TRACTION MODE
-------------

Outputs:
- tether force
- Fx (surge)
- Fy (sway)

Equivalent propulsion benefit:

    P_equiv_traction = Fx * V_ship


UNCONSTRAINED TRACTION DIAGNOSTIC
--------------------------------

If tether force exceeds limit:
- constraint is temporarily disabled
- solution is still computed

Purpose:
- verify coordinate system correctness
- understand force envelope

These results are NOT physically feasible.


CURRENT MODELLING STATUS
------------------------

Implemented:
- apparent wind coupling
- coordinate transformations
- pumping cycle simulation
- traction force evaluation
- equivalent propulsion metrics
- heading sweeps
- matrix simulations with verification checks

Not yet implemented:
- vessel equilibrium (PPP)
- leeway angle
- resistance model
- propulsion system model
- traction optimization
- depowering control
- economic analysis


NEXT STEPS
----------

1) TRACTION OPTIMIZATION  
Optimize:
- azimuth angle
- elevation angle
- course angle  
Objective:
    maximize Fx or P_equiv_traction  
Constraint:
    tether_force <= limit

2) CONSTRAINED TRACTION MODEL  
Replace unconstrained solver with:
- depowering strategy OR
- infeasible masking

3) FULL VESSEL COUPLING  
Couple:
    kite forces + resistance + propulsion  
Solve full equilibrium

4) UNIFIED MODE COMPARISON  
Compare traction and pumping using:
    equivalent propulsion benefit

5) ECONOMIC EVALUATION  
Combine with wind statistics to compute:
- fuel savings
- system value


REFERENCES
----------

[1] R. van der Vlugt et al., 2019  
Quasi-Steady Model of a Pumping Kite Power System  
Renewable Energy 131, pp. 83–99  

[2] M. Schelbergen, 2024  
Power to the Airborne Wind Energy Performance Model  
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_text)