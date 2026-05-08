#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overlay traction and pumping equivalent power in one polar plot.

Inputs:
- results/optimized_traction_heading_wind_sweep.csv
- results/optimized_pumping_heading_wind_sweep.csv

Plot convention:
- Same true wind speed = same color
- Solid line  = traction mode
- Dashed line = pumping mode
- Radius      = positive P_equiv [kW]

Traction:
    P_equiv_traction = Fx_ship * V_ship

Pumping:
    P_equiv_pumping = P_cycle + Fx_avg * V_ship

Mirroring:
- If MIRROR_RESULTS_FOR_PLOT = True:
    computed headings in 0–180 deg are mirrored to 180–360 deg for plotting only.
- If MIRROR_RESULTS_FOR_PLOT = False:
    the script plots only the headings present in the CSV files.

So Optimize_pumping_moving_vessel.py and Optimize_traction_moving_vessel.py 
both need to be run first under the same windsweep / vessel speed and then this script can be run to compare the results. 
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PHASE2_RESULTS_DIR = RESULTS_DIR / "Phase 2 - Traction mode verification"
PHASE3_RESULTS_DIR = RESULTS_DIR / "Phase 3 - Pumping mode verification"
PHASE5_RESULTS_DIR = RESULTS_DIR / "Phase 5 - Traction pumping overlay comparison"
PHASE5_PLOTS_DIR = PHASE5_RESULTS_DIR / "plots"

PHASE5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PHASE5_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TRACTION_CSV_PATH = (
    PHASE2_RESULTS_DIR
    / "optimized_traction_heading_wind_sweep_force_limited_10ms_14ms_18ms.csv"
)

PUMPING_CSV_PATH = (
    PHASE3_RESULTS_DIR
    / "optimized_pumping_heading_wind_sweep_Peq_objective_10ms_14ms_18ms_15deg.csv"
)

COMMON_FORMAT_CSV_PATH = (
    PHASE5_RESULTS_DIR / "traction_pumping_comparison_common_format.csv"
)

COMPARISON_SUMMARY_CSV_PATH = (
    PHASE5_RESULTS_DIR / "traction_pumping_comparison_summary.csv"
)

FIGURE_OUTPUT_PATH = (
    PHASE5_PLOTS_DIR / "overlay_traction_pumping_polar_pequiv.png"
)


# =============================================================================
# Plot / output settings
# =============================================================================

TRUE_WIND_SPEEDS_TO_PLOT = np.array([10.0, 14.0, 18.0], dtype=float)

MIRROR_RESULTS_FOR_PLOT = True
# True  = mirror 0–180 deg results to 180–360 deg for plotting only
# False = plot only headings that are present in the CSV files

USE_COMMON_HEADINGS_ONLY = True
# True  = compare only wind-speed / heading points that exist in both CSV files
# False = plot all available points from both CSV files

PLOT_ONLY_POSITIVE_EQUIVALENT_POWER = True
# True  = negative P_equiv values are plotted as zero
# False = negative P_equiv values are kept

MIN_PLOT_BENEFIT_KW = 0.01
# Values below this are plotted as zero to avoid numerical round-off showing as benefit.
# 0.01 kW = 10 W.

SAVE_FIGURE = True
SAVE_COMMON_FORMAT_CSV = True
SAVE_COMPARISON_SUMMARY_CSV = True

SHOW_FIGURE = False
SHOW_MARKERS = True


# =============================================================================
# CSV helpers
# =============================================================================

def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in ["true", "1", "yes", "y"]


def safe_float(value, default=np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value if np.isfinite(value) else default


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found:\n{path}")

    rows = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"No rows to write for:\n  {path}")
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"Saved CSV:\n  {path}")


# =============================================================================
# Data extraction
# =============================================================================

def extract_traction_rows(csv_rows: list[dict]) -> list[dict]:
    """
    Convert traction CSV rows to common comparison format.
    """

    extracted = []

    for row in csv_rows:
        true_wind_speed = safe_float(row.get("true_wind_speed"))

        if not any(np.isclose(true_wind_speed, target) for target in TRUE_WIND_SPEEDS_TO_PLOT):
            continue

        accepted = parse_bool(row.get("accepted_result", False))

        p_equiv_w = safe_float(row.get("optimized_P_equiv_traction", np.nan))
        fx_ship = safe_float(row.get("optimized_Fx", np.nan))
        fy_ship = safe_float(row.get("optimized_Fy", np.nan))

        extracted.append(
            {
                "mode": "traction",
                "true_wind_speed": true_wind_speed,
                "ship_speed": safe_float(row.get("ship_speed")),
                "heading_deg": safe_float(row.get("heading_deg")),
                "accepted_result": accepted,
                "P_equiv_W": p_equiv_w,
                "P_equiv_kW": p_equiv_w / 1000.0 if np.isfinite(p_equiv_w) else np.nan,
                "Fx_or_Fx_avg": fx_ship,
                "Fy_or_Fy_avg": fy_ship,
                "source_file": TRACTION_CSV_PATH.name,
            }
        )

    return extracted


def extract_pumping_rows(csv_rows: list[dict]) -> list[dict]:
    """
    Convert pumping CSV rows to common comparison format.
    """

    extracted = []

    for row in csv_rows:
        true_wind_speed = safe_float(row.get("true_wind_speed"))

        if not any(np.isclose(true_wind_speed, target) for target in TRUE_WIND_SPEEDS_TO_PLOT):
            continue

        accepted = parse_bool(row.get("accepted_result", False))

        p_equiv_w = safe_float(row.get("P_equiv_pumping", np.nan))
        p_cycle_w = safe_float(row.get("P_cycle", np.nan))
        p_prop_w = safe_float(row.get("P_prop_equiv", np.nan))
        fx_avg = safe_float(row.get("Fx_avg", np.nan))
        fy_avg = safe_float(row.get("Fy_avg", np.nan))

        extracted.append(
            {
                "mode": "pumping",
                "true_wind_speed": true_wind_speed,
                "ship_speed": safe_float(row.get("ship_speed")),
                "heading_deg": safe_float(row.get("heading_deg")),
                "accepted_result": accepted,
                "case_status": row.get("case_status", ""),
                "P_equiv_W": p_equiv_w,
                "P_equiv_kW": p_equiv_w / 1000.0 if np.isfinite(p_equiv_w) else np.nan,
                "P_cycle_W": p_cycle_w,
                "P_cycle_kW": p_cycle_w / 1000.0 if np.isfinite(p_cycle_w) else np.nan,
                "P_prop_equiv_W": p_prop_w,
                "P_prop_equiv_kW": p_prop_w / 1000.0 if np.isfinite(p_prop_w) else np.nan,
                "Fx_or_Fx_avg": fx_avg,
                "Fy_or_Fy_avg": fy_avg,
                "source_file": PUMPING_CSV_PATH.name,
            }
        )

    return extracted


def filter_to_common_wind_heading_points(
    traction_rows: list[dict],
    pumping_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Keep only wind-speed / heading points available in both modes.

    This avoids visually comparing traction at 5 deg resolution with pumping
    at 15 deg resolution.
    """

    traction_points = {
        (
            round(float(r["true_wind_speed"]), 6),
            round(float(r["heading_deg"]), 6),
        )
        for r in traction_rows
        if np.isfinite(safe_float(r.get("true_wind_speed")))
        and np.isfinite(safe_float(r.get("heading_deg")))
    }

    pumping_points = {
        (
            round(float(r["true_wind_speed"]), 6),
            round(float(r["heading_deg"]), 6),
        )
        for r in pumping_rows
        if np.isfinite(safe_float(r.get("true_wind_speed")))
        and np.isfinite(safe_float(r.get("heading_deg")))
    }

    common_points = traction_points.intersection(pumping_points)

    traction_filtered = [
        r for r in traction_rows
        if (
            round(float(r["true_wind_speed"]), 6),
            round(float(r["heading_deg"]), 6),
        )
        in common_points
    ]

    pumping_filtered = [
        r for r in pumping_rows
        if (
            round(float(r["true_wind_speed"]), 6),
            round(float(r["heading_deg"]), 6),
        )
        in common_points
    ]

    return traction_filtered, pumping_filtered


def build_common_format_rows(
    traction_rows: list[dict],
    pumping_rows: list[dict],
) -> list[dict]:
    """
    Combine traction and pumping rows into one common-format table.
    """

    common_rows = []

    for row in traction_rows + pumping_rows:
        common_rows.append(
            {
                "mode": row.get("mode", ""),
                "true_wind_speed": row.get("true_wind_speed", np.nan),
                "ship_speed": row.get("ship_speed", np.nan),
                "heading_deg": row.get("heading_deg", np.nan),
                "accepted_result": row.get("accepted_result", False),
                "case_status": row.get("case_status", ""),
                "P_equiv_W": row.get("P_equiv_W", np.nan),
                "P_equiv_kW": row.get("P_equiv_kW", np.nan),
                "P_cycle_W": row.get("P_cycle_W", ""),
                "P_cycle_kW": row.get("P_cycle_kW", ""),
                "P_prop_equiv_W": row.get("P_prop_equiv_W", ""),
                "P_prop_equiv_kW": row.get("P_prop_equiv_kW", ""),
                "Fx_or_Fx_avg": row.get("Fx_or_Fx_avg", np.nan),
                "Fy_or_Fy_avg": row.get("Fy_or_Fy_avg", np.nan),
                "source_file": row.get("source_file", ""),
            }
        )

    return common_rows


# =============================================================================
# Plotting helpers
# =============================================================================

def result_radius_kw(row: dict) -> float:
    """
    Radius used in the polar plot.

    Failed / inactive / non-finite results are plotted as zero.
    Negative equivalent power can also be clipped to zero.
    Very small positive values are clipped to zero to avoid numerical round-off.
    """

    if not row.get("accepted_result", False):
        return 0.0

    value_kw = safe_float(row.get("P_equiv_kW", np.nan))

    if not np.isfinite(value_kw):
        return 0.0

    if PLOT_ONLY_POSITIVE_EQUIVALENT_POWER:
        if value_kw <= MIN_PLOT_BENEFIT_KW:
            return 0.0
        return value_kw

    return value_kw


def build_polar_points(
    rows_for_one_mode_and_wind: list[dict],
    mirror: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build heading/radius arrays for polar plotting.

    If mirror=True:
        heading h in (0, 180) is mirrored to 360 - h.
    """

    point_map = {}

    for row in rows_for_one_mode_and_wind:
        heading = safe_float(row.get("heading_deg"))

        if not np.isfinite(heading):
            continue

        heading = heading % 360.0
        radius = result_radius_kw(row)

        point_map[heading] = max(point_map.get(heading, 0.0), radius)

        if mirror and 0.0 < heading < 180.0:
            mirrored_heading = 360.0 - heading
            point_map[mirrored_heading] = max(
                point_map.get(mirrored_heading, 0.0),
                radius,
            )

    if not point_map:
        return np.array([]), np.array([])

    headings_deg = np.array(sorted(point_map.keys()), dtype=float)
    radii_kw = np.array([point_map[h] for h in headings_deg], dtype=float)

    headings_closed = np.concatenate([headings_deg, [headings_deg[0] + 360.0]])
    radii_closed = np.concatenate([radii_kw, [radii_kw[0]]])

    return np.deg2rad(headings_closed), radii_closed


def get_common_ship_speed(traction_rows: list[dict], pumping_rows: list[dict]) -> float:
    all_rows = traction_rows + pumping_rows
    speeds = [
        safe_float(r.get("ship_speed"))
        for r in all_rows
        if np.isfinite(safe_float(r.get("ship_speed")))
    ]

    if not speeds:
        return np.nan

    return float(np.nanmedian(speeds))


# =============================================================================
# Summary
# =============================================================================

def build_comparison_summary(
    traction_rows: list[dict],
    pumping_rows: list[dict],
) -> list[dict]:
    summary_rows = []

    for wind_speed in TRUE_WIND_SPEEDS_TO_PLOT:
        traction_group = [
            r for r in traction_rows
            if np.isclose(float(r["true_wind_speed"]), wind_speed)
            and r.get("accepted_result", False)
            and np.isfinite(safe_float(r.get("P_equiv_kW")))
        ]

        pumping_group = [
            r for r in pumping_rows
            if np.isclose(float(r["true_wind_speed"]), wind_speed)
            and r.get("accepted_result", False)
            and np.isfinite(safe_float(r.get("P_equiv_kW")))
        ]

        if traction_group:
            best_traction = max(traction_group, key=lambda r: r["P_equiv_kW"])
            best_traction_kw = best_traction["P_equiv_kW"]
            best_traction_heading = best_traction["heading_deg"]
        else:
            best_traction_kw = np.nan
            best_traction_heading = np.nan

        if pumping_group:
            best_pumping = max(pumping_group, key=lambda r: r["P_equiv_kW"])
            best_pumping_kw = best_pumping["P_equiv_kW"]
            best_pumping_heading = best_pumping["heading_deg"]
        else:
            best_pumping_kw = np.nan
            best_pumping_heading = np.nan

        if np.isfinite(best_traction_kw) and np.isfinite(best_pumping_kw):
            delta_best_pumping_minus_traction = best_pumping_kw - best_traction_kw
            best_mode = "pumping" if best_pumping_kw > best_traction_kw else "traction"
        else:
            delta_best_pumping_minus_traction = np.nan
            best_mode = "unavailable"

        summary_rows.append(
            {
                "true_wind_speed": float(wind_speed),
                "ship_speed": get_common_ship_speed(traction_rows, pumping_rows),
                "n_traction_rows": len(traction_group),
                "n_pumping_rows": len(pumping_group),
                "best_traction_kW": best_traction_kw,
                "best_traction_heading_deg": best_traction_heading,
                "best_pumping_kW": best_pumping_kw,
                "best_pumping_heading_deg": best_pumping_heading,
                "delta_best_pumping_minus_traction_kW": delta_best_pumping_minus_traction,
                "best_mode_by_peak_value": best_mode,
            }
        )

    return summary_rows


def print_comparison_summary(summary_rows: list[dict]) -> None:
    print("\nCOMPARISON SUMMARY")
    print("------------------")

    for row in summary_rows:
        print(
            f"Vw={row['true_wind_speed']:5.1f} m/s | "
            f"best traction={row['best_traction_kW']:8.3f} kW "
            f"at ψ={row['best_traction_heading_deg']:6.1f}° | "
            f"best pumping={row['best_pumping_kW']:8.3f} kW "
            f"at ψ={row['best_pumping_heading_deg']:6.1f}° | "
            f"best mode={row['best_mode_by_peak_value']}"
        )


# =============================================================================
# Plotting
# =============================================================================

def plot_overlay_polar(
    traction_rows: list[dict],
    pumping_rows: list[dict],
) -> None:
    fig, ax = plt.subplots(
        figsize=(9, 9),
        subplot_kw={"projection": "polar"},
    )

    max_radius = 0.0
    marker = "o" if SHOW_MARKERS else None

    for wind_speed in TRUE_WIND_SPEEDS_TO_PLOT:
        traction_group = [
            r for r in traction_rows
            if np.isclose(float(r["true_wind_speed"]), wind_speed)
        ]

        pumping_group = [
            r for r in pumping_rows
            if np.isclose(float(r["true_wind_speed"]), wind_speed)
        ]

        theta_trac, radius_trac = build_polar_points(
            traction_group,
            mirror=MIRROR_RESULTS_FOR_PLOT,
        )

        theta_pump, radius_pump = build_polar_points(
            pumping_group,
            mirror=MIRROR_RESULTS_FOR_PLOT,
        )

        traction_line = None

        if len(theta_trac) > 0:
            traction_line, = ax.plot(
                theta_trac,
                radius_trac,
                linestyle="-",
                marker=marker,
                markersize=3,
                linewidth=2.0,
                label=f"Traction, {wind_speed:.0f} m/s",
            )

            max_radius = max(max_radius, float(np.nanmax(radius_trac)))

        if len(theta_pump) > 0:
            if traction_line is not None:
                color = traction_line.get_color()
            else:
                color = None

            ax.plot(
                theta_pump,
                radius_pump,
                linestyle="--",
                marker=marker,
                markersize=3,
                linewidth=2.0,
                color=color,
                label=f"Pumping, {wind_speed:.0f} m/s",
            )

            max_radius = max(max_radius, float(np.nanmax(radius_pump)))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    if max_radius > 0.0:
        ax.set_ylim(0.0, 1.10 * max_radius)
    else:
        ax.set_ylim(0.0, 1.0)

    ax.set_rlabel_position(135)

    ship_speed = get_common_ship_speed(traction_rows, pumping_rows)

    if MIRROR_RESULTS_FOR_PLOT:
        mirror_text = "0–180° computed, 180–360° mirrored for plotting only"
    else:
        mirror_text = "CSV headings plotted directly"

    if USE_COMMON_HEADINGS_ONLY:
        heading_text = "common headings only"
    else:
        heading_text = "all available headings"

    ax.set_title(
        "Prescribed-motion equivalent power comparison: traction vs pumping\n"
        f"Ship speed = {ship_speed:.1f} m/s, {heading_text}\n"
        f"{mirror_text}",
        pad=28,
    )

    ax.text(
        np.deg2rad(135),
        1.05 * max_radius if max_radius > 0.0 else 0.90,
        "Positive P_equiv [kW]",
        ha="center",
        va="center",
    )

    ax.legend(
        title="Mode and true wind speed",
        loc="upper right",
        bbox_to_anchor=(1.42, 1.12),
    )

    ax.grid(True)

    if SAVE_FIGURE:
        fig.savefig(FIGURE_OUTPUT_PATH, dpi=300, bbox_inches="tight")
        print(f"\nSaved overlay polar plot to:\n  {FIGURE_OUTPUT_PATH}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("\nPHASE 5 — TRACTION AND PUMPING COMPARISON")
    print("-----------------------------------------")
    print(f"Traction CSV          : {TRACTION_CSV_PATH}")
    print(f"Pumping CSV           : {PUMPING_CSV_PATH}")
    print(f"Output directory      : {PHASE5_RESULTS_DIR}")
    print(f"Plot directory        : {PHASE5_PLOTS_DIR}")
    print(f"True wind speeds      : {TRUE_WIND_SPEEDS_TO_PLOT}")
    print(f"Use common headings   : {USE_COMMON_HEADINGS_ONLY}")
    print(f"Mirrored for plot     : {MIRROR_RESULTS_FOR_PLOT}")
    print(f"Positive only plot    : {PLOT_ONLY_POSITIVE_EQUIVALENT_POWER}")
    print(f"Min plotted benefit   : {MIN_PLOT_BENEFIT_KW:.3f} kW")
    print("Solid line            : traction")
    print("Dashed line           : pumping")
    print("Radius                : positive P_equiv [kW]")
    print("Model status          : prescribed ship motion, no PPP coupling yet")

    traction_csv_rows = read_csv_rows(TRACTION_CSV_PATH)
    pumping_csv_rows = read_csv_rows(PUMPING_CSV_PATH)

    traction_rows = extract_traction_rows(traction_csv_rows)
    pumping_rows = extract_pumping_rows(pumping_csv_rows)

    print("\nRaw extracted rows")
    print("------------------")
    print(f"Traction rows         : {len(traction_rows)}")
    print(f"Pumping rows          : {len(pumping_rows)}")

    if USE_COMMON_HEADINGS_ONLY:
        traction_rows, pumping_rows = filter_to_common_wind_heading_points(
            traction_rows=traction_rows,
            pumping_rows=pumping_rows,
        )

        print("\nAfter common-heading filtering")
        print("------------------------------")
        print(f"Traction rows         : {len(traction_rows)}")
        print(f"Pumping rows          : {len(pumping_rows)}")

    common_rows = build_common_format_rows(
        traction_rows=traction_rows,
        pumping_rows=pumping_rows,
    )

    summary_rows = build_comparison_summary(
        traction_rows=traction_rows,
        pumping_rows=pumping_rows,
    )

    print_comparison_summary(summary_rows)

    if SAVE_COMMON_FORMAT_CSV:
        write_csv(common_rows, COMMON_FORMAT_CSV_PATH)

    if SAVE_COMPARISON_SUMMARY_CSV:
        write_csv(summary_rows, COMPARISON_SUMMARY_CSV_PATH)

    plot_overlay_polar(
        traction_rows=traction_rows,
        pumping_rows=pumping_rows,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()