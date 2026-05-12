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
- Traction mode optimization
- Pumping mode optimization with vessel-force correction
- Traction versus pumping comparison scripts
- Phase-based result storage and verification outputs


ADDED FILES AND SCRIPTS
-----------------------

scripts/

test_pumping_with_apparent_wind.py
Runs the pumping-cycle model using apparent wind from a moving vessel.
Projects cycle forces into ship axes and computes equivalent propulsion benefit.

test_traction_with_apparent_wind.py
Evaluates traction mode under apparent wind conditions.
Used as an early verification script before the optimized traction sweep was added.

test_pumping_force_projection.py
Standalone verification of force projection logic from QSM frame to global and ship frame.

test_pure_traction.py
Baseline traction solver test without vessel coupling.

test_vessel_coupling.py
Validation script for apparent wind computation and vessel motion coupling.

optimize_traction_moving_vessel.py
Main optimized traction-mode heading-sweep script for the moving-vessel model.
For each true wind speed and ship heading, it:
- computes the apparent wind from prescribed ship motion
- runs the traction solver in the apparent-wind-aligned QSM frame
- optimizes azimuth angle, elevation angle, and course angle
- enforces positive surge force and tether-force constraints
- computes P_equiv_traction = Fx_ship * V_ship
- saves CSV output and diagnostic plots

The script is used for Phase 2: Traction mode verification.
It includes diagnostics for:
- tether-force constraint activity
- Fx/T projection ratio
- projection loss angle
- apparent wind speed and angle
- optimized operating angles
- constrained and no-force-limit diagnostic runs

optimize_pumping_moving_vessel.py
Main optimized pumping-mode heading-sweep script for the moving-vessel model.
For each true wind speed and ship heading, it:
- computes the apparent wind from prescribed ship motion
- runs the QSM pumping-cycle optimizer in the apparent-wind-aligned frame
- extracts the cycle force history
- projects the cycle forces into the ship frame
- computes cycle-averaged Fx_avg and Fy_avg
- maximizes P_equiv_pumping = P_cycle + Fx_avg * V_ship
- saves CSV output and diagnostic plots

The script is used for Phase 3: Pumping mode verification.
It includes diagnostics for:
- cycle power
- propulsion-equivalent force contribution
- propulsion penalty
- tether-force constraint activity
- stroke length and stroke fraction
- optimized reeling speeds and elevation variables
- apparent wind speed and angle

operationalmap.py
Comparison and operational-map script for traction and pumping results.
It reads the optimized traction and pumping CSV files and compares both modes on the same equivalent-power basis.

The script is used after Phase 2 and Phase 3 to support Phase 4 and later comparison work.
It can be used to:
- overlay traction and pumping polar plots
- compare P_equiv_traction and P_equiv_pumping
- identify which mode gives the highest equivalent benefit
- create mode-selection maps over wind speed and heading
- support later switching-boundary and operational-strategy analysis


src/inertiafree_qsm/

coordinate_transforms.py
Core transformation logic:
- QSM frame -> global frame -> ship frame

Ensures physically consistent mapping of tether forces from the apparent-wind-aligned kite frame to the vessel axes.

vessel_coupling.py
Implements:
- true wind definition
- vessel motion definition
- apparent wind computation

Currently used for prescribed vessel motion. Later this module can be extended or connected to a vessel equilibrium model with leeway.

force_projection.py
Utilities for projecting forces into vessel axes or other model coordinate systems.

operating_point.py
Defines operating-point structures for traction calculations:
- vessel state
- wind condition

pure_traction.py
Standalone quasi-steady traction solver:
- computes tether force
- computes traction operating point
- supports moving-vessel force projection through the surrounding coupling scripts


MOVING-VESSEL COUPLING
----------------------

The vessel is currently introduced as a prescribed moving ground station.
The ship speed and heading are imposed.

The apparent wind is computed from the true wind vector and the vessel velocity vector:

    V_app = V_true - V_ship

The QSM model is evaluated using the apparent wind speed and direction.
The QSM frame is aligned with the apparent wind direction.

The current coupling is one-way:

    prescribed ship motion -> apparent wind -> kite model -> ship-frame forces

The vessel response is not yet solved.


COORDINATE SYSTEM CONVENTION
----------------------------

The QSM model operates in an apparent-wind-aligned frame.

Transformation chain:

    QSM frame -> global frame -> ship frame

Key assumptions:
- QSM azimuth angle defines the kite position direction
- tether force on the vessel acts from the vessel toward the kite
- ship axes:
    Fx = surge force, positive forward
    Fy = sway force, positive lateral


TRACTION MODE
-------------

Traction mode is represented by one optimized quasi-steady operating point.

Outputs:
- tether force
- Fx_ship
- Fy_ship
- optimized azimuth angle
- optimized elevation angle
- optimized course angle

Equivalent propulsion benefit:

    P_equiv_traction = Fx_ship * V_ship

Current traction optimization variables:

    x = [azimuth_angle, elevation_angle, course_angle]

Main constraints:
- Fx_ship >= 0
- tether_force <= tether_force_max

Current depowering strategy:
- traction mode uses one powered aerodynamic state
- CL, CD, and AoA are not directly optimized
- when the tether force limit is active, the optimizer geometrically depowers the kite by changing azimuth, elevation, and course angle
- this can reduce Fx/T and increase projection loss

A future extension is to include explicit aerodynamic depowering:

    x = [azimuth_angle, elevation_angle, course_angle, depower]

with:

    CL = CL(AoA)
    CD = CD(AoA)


PUMPING MODE
------------

Pumping mode is represented by an optimized quasi-steady pumping cycle.

Outputs:
- cycle power: P_cycle
- cycle-averaged forces: Fx_avg, Fy_avg
- mean and maximum tether force
- optimized reeling speeds
- optimized stroke fractions
- optimized elevation variables

Equivalent propulsion benefit:

    P_equiv_pumping = P_cycle + Fx_avg * V_ship

Positive Fx_avg contributes to propulsion.
Negative Fx_avg creates a propulsion penalty.

Current pumping-cycle power is mechanical QSM cycle power.
No drivetrain, generator, electrical, or propulsion-reuse efficiency is included yet.

A future placeholder correction can be introduced as:

    P_equiv_pumping_eff = eta_use * P_cycle + Fx_avg * V_ship

OPTIMIZER ARCHITECTURE
----------------------

Both traction and pumping optimizers follow the same general structure:

    true wind + prescribed vessel motion
        -> apparent wind
        -> QSM model evaluation in apparent-wind-aligned frame
        -> force projection into ship frame
        -> equivalent-power objective
        -> optimized result storage

The main difference is that traction mode optimizes one steady operating point,
whereas pumping mode optimizes a full cycle.

Traction optimization:
- steady quasi-steady operating point
- decision variables:
    x = [azimuth_angle, elevation_angle, course_angle]
- objective:
    maximize P_equiv_traction = Fx_ship * V_ship
- constraints:
    Fx_ship >= 0
    tether_force <= tether_force_max
- multistart optimization is used because different local optima can exist
  for port/starboard, crosswind, low-elevation, and force-limited branches
- warm-starting is used between neighbouring headings

Pumping optimization:
- full quasi-steady pumping cycle
- decision variables are inherited from the QSM CycleOptimizer and can include:
    reeling_speed_out
    reeling_speed_in
    frac_start
    frac_end
    elevation variables
- original QSM objective:
    maximize P_cycle
- moving-vessel objective:
    maximize P_equiv_pumping = P_cycle + Fx_avg * V_ship
- cycle force history is extracted and projected into ship axes
- Fx_avg and Fy_avg are time-averaged over the cycle
- multistart and warm-starting are used to improve robustness

Important runtime difference:
- traction is relatively cheap because each objective evaluation solves one
  steady operating point
- pumping is more expensive because each objective evaluation runs a full
  pumping-cycle simulation and post-processes force histories

Known optimizer sensitivity:
- low apparent wind can lead to very small or flat objective values
- high wind can activate the tether-force constraint, making the optimum lie
  close to a constraint boundary
- pumping can become partly traction-like when Fx_avg * V_ship dominates
  P_cycle
- nearby headings may converge to different local optima if multistart,
  warm-starting, or iteration limits are not chosen carefully

For future PPP coupling, the optimizers should preferably be used to generate
offline performance maps over apparent wind speed and direction. The coupled
vessel solver can then interpolate these maps instead of running the full kite
optimization inside every vessel-equilibrium iteration.

OPERATIONAL MAP / MODE COMPARISON
---------------------------------

The operational comparison reads the optimized traction and pumping results and compares both modes on the same metric.

Traction:

    P_equiv_traction = Fx_ship * V_ship

Pumping:

    P_equiv_pumping = P_cycle + Fx_avg * V_ship

The comparison can be used to identify:
- traction-favourable regions
- pumping-favourable regions
- zero-benefit or infeasible regions
- switching boundaries between modes

At the current stage, this is still based on prescribed ship speed.
It is not yet a full vessel-response or fuel-saving calculation.



CURRENT MODELLING STATUS
------------------------

Implemented:
- moving-vessel apparent wind coupling
- QSM-to-global-to-ship force transformations
- traction force evaluation
- constrained traction optimization
- no-force-limit traction diagnostic
- pumping-cycle simulation with moving-vessel apparent wind
- cycle-averaged ship-frame force projection
- equivalent propulsion metrics for traction and pumping
- heading sweeps over true wind speed and ship heading
- phase-based CSV and plot output
- traction versus pumping comparison on a common equivalent-power basis

Not yet implemented:
- full vessel equilibrium / PPP coupling
- leeway angle
- hydrodynamic resistance and side-force model
- propulsion system model
- rudder or yaw-equilibrium model
- explicit aerodynamic depowering in traction mode
- drivetrain/generator/propulsion-reuse efficiency correction for pumping
- economic analysis based on route or wind statistics
- optimizer timeout protection for long-running marginal cases
- runtime benchmarking for traction and pumping optimizers
- offline kite performance maps for efficient PPP coupling


NEXT STEPS
----------

1) POWER AND EFFICIENCY CHECK
Add a placeholder efficiency factor for pumping power:

    P_equiv_pumping_eff = eta_use * P_cycle + Fx_avg * V_ship

Use this to test sensitivity to drivetrain, generator, electrical, and propulsion-reuse efficiency.

2) EXPLICIT TRACTION DEPOWERING
Extend the traction optimization variables with a depowering or AoA variable:

    x = [azimuth_angle, elevation_angle, course_angle, depower]

Map depower to aerodynamic coefficients:

    CL = CL(AoA)
    CD = CD(AoA)

This separates aerodynamic depowering from geometric depowering.

3) OPTIMIZER ROBUSTNESS AND RUNTIME CONTROL
Before coupling to the PPP, make both traction and pumping optimizers robust and
predictable in runtime.

Required checks:
- add runtime logging per case
- add timeout protection for marginal cases
- identify local-optimum jumps between neighbouring headings
- benchmark runtime for traction and pumping separately
- reduce pumping optimization dimensionality if needed
- define inactive or low-benefit regions to avoid wasting runtime

This is necessary because full vessel coupling may require repeated evaluations
of kite performance inside an outer equilibrium loop.

4) FULL VESSEL COUPLING
Replace prescribed vessel speed with a vessel equilibrium model.

Couple:

    kite forces + hydrodynamic forces + propulsion forces

Solve for:
- ship speed
- leeway angle
- required propulsion power
- possibly rudder force or yaw equilibrium

5) UNIFIED MODE COMPARISON
Compare traction and pumping using propulsion-power reduction from the coupled vessel model, rather than only prescribed-speed equivalent power.

6) OPERATIONAL STRATEGY
Use the coupled results to define mode-selection boundaries between traction and pumping.

7) ECONOMIC EVALUATION
Combine operational maps with wind statistics or route data to estimate:
- propulsion energy reduction
- fuel savings
- value of dual-mode operation


REFERENCES
----------

[1] R. van der Vlugt et al., 2019
Quasi-Steady Model of a Pumping Kite Power System
Renewable Energy 131, pp. 83-99

[2] M. Schelbergen, 2024
Power to the Airborne Wind Energy Performance Model
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_text)