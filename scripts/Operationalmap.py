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

TRACTION_CSV_PATH = RESULTS_DIR / "optimized_traction_heading_wind_sweep.csv"
PUMPING_CSV_PATH = RESULTS_DIR / "optimized_pumping_heading_wind_sweep.csv"


# =============================================================================
# Plot settings
# =============================================================================

TRUE_WIND_SPEEDS_TO_PLOT = np.array([10.0, 14.0, 18.0], dtype=float)

MIRROR_RESULTS_FOR_PLOT = True
# True  = mirror 0–180 deg results to 180–360 deg for plotting only
# False = plot only headings that are present in the CSV files

PLOT_ONLY_POSITIVE_EQUIVALENT_POWER = True
# True  = negative P_equiv values are plotted as zero
# False = negative P_equiv values are kept, but polar plots with negative radius
#         are harder to interpret, so True is recommended.

SAVE_FIGURE = True
FIGURE_OUTPUT_PATH = RESULTS_DIR / "overlay_traction_pumping_polar_pequiv.png"

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
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# =============================================================================
# Data extraction
# =============================================================================

def extract_traction_rows(csv_rows: list[dict]) -> list[dict]:
    """
    Convert traction CSV rows to a common format:

        {
            mode,
            true_wind_speed,
            ship_speed,
            heading_deg,
            accepted_result,
            P_equiv_W,
            P_equiv_kW,
        }
    """

    extracted = []

    for row in csv_rows:
        accepted = parse_bool(row.get("accepted_result", False))

        p_equiv_w = safe_float(
            row.get("optimized_P_equiv_traction", np.nan)
        )

        extracted.append(
            {
                "mode": "traction",
                "true_wind_speed": safe_float(row.get("true_wind_speed")),
                "ship_speed": safe_float(row.get("ship_speed")),
                "heading_deg": safe_float(row.get("heading_deg")),
                "accepted_result": accepted,
                "P_equiv_W": p_equiv_w,
                "P_equiv_kW": p_equiv_w / 1000.0,
            }
        )

    return extracted


def extract_pumping_rows(csv_rows: list[dict]) -> list[dict]:
    """
    Convert pumping CSV rows to a common format:

        {
            mode,
            true_wind_speed,
            ship_speed,
            heading_deg,
            accepted_result,
            P_equiv_W,
            P_equiv_kW,
        }
    """

    extracted = []

    for row in csv_rows:
        accepted = parse_bool(row.get("accepted_result", False))

        p_equiv_w = safe_float(
            row.get("P_equiv_pumping", np.nan)
        )

        extracted.append(
            {
                "mode": "pumping",
                "true_wind_speed": safe_float(row.get("true_wind_speed")),
                "ship_speed": safe_float(row.get("ship_speed")),
                "heading_deg": safe_float(row.get("heading_deg")),
                "accepted_result": accepted,
                "P_equiv_W": p_equiv_w,
                "P_equiv_kW": p_equiv_w / 1000.0,
            }
        )

    return extracted


def result_radius_kw(row: dict) -> float:
    """
    Radius used in the polar plot.

    Failed / inactive / non-finite results are plotted as zero.
    Negative equivalent power can also be clipped to zero.
    """

    if not row.get("accepted_result", False):
        return 0.0

    value_kw = safe_float(row.get("P_equiv_kW", np.nan))

    if not np.isfinite(value_kw):
        return 0.0

    if PLOT_ONLY_POSITIVE_EQUIVALENT_POWER:
        return max(value_kw, 0.0)

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
        heading = safe_float(row.get("heading_deg")) % 360.0
        radius = result_radius_kw(row)

        if not np.isfinite(heading):
            continue

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


# =============================================================================
# Plotting
# =============================================================================

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


def plot_overlay_polar(
    traction_rows: list[dict],
    pumping_rows: list[dict],
) -> None:
    fig, ax = plt.subplots(
        figsize=(9, 9),
        subplot_kw={"projection": "polar"},
    )

    max_radius = 0.0

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

        marker = "o" if SHOW_MARKERS else None

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

    ax.set_title(
        "Equivalent power comparison: traction vs pumping\n"
        f"Ship speed = {ship_speed:.1f} m/s\n"
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
        print(f"Saved overlay polar plot to:\n  {FIGURE_OUTPUT_PATH}")

    plt.show()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    traction_csv_rows = read_csv_rows(TRACTION_CSV_PATH)
    pumping_csv_rows = read_csv_rows(PUMPING_CSV_PATH)

    traction_rows = extract_traction_rows(traction_csv_rows)
    pumping_rows = extract_pumping_rows(pumping_csv_rows)

    print("\nOVERLAY TRACTION AND PUMPING POLAR PLOT")
    print("---------------------------------------")
    print(f"Traction CSV          : {TRACTION_CSV_PATH}")
    print(f"Pumping CSV           : {PUMPING_CSV_PATH}")
    print(f"True wind speeds      : {TRUE_WIND_SPEEDS_TO_PLOT}")
    print(f"Mirrored for plot     : {MIRROR_RESULTS_FOR_PLOT}")
    print("Solid line            : traction")
    print("Dashed line           : pumping")
    print("Radius                : positive P_equiv [kW]")

    plot_overlay_polar(
        traction_rows=traction_rows,
        pumping_rows=pumping_rows,
    )


if __name__ == "__main__":
    main()