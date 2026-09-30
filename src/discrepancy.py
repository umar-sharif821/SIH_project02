"""Rule engine that compares the harmonized sources and flags *possible* discrepancies.

Nothing here declares a legal error — every row is a lead for an officer to verify."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

import config

ACTION = "Officer verification required"
SEVERITY = {
    "Unrecorded building": "High",
    "Boundary encroachment": "High",
    "Recorded building not found": "High",
    "Parcel area mismatch": "Medium",
    "Built-up area mismatch": "Medium",
    "Utility on vacant parcel": "Medium",
    "Structure spans multiple parcels": "Low",
    "No municipal record": "Low",
    "Orphan record": "Low",
}
RULES = list(SEVERITY)


def _pct(a, b) -> str:
    return f"{(a - b) / b * 100:+.0f}%" if b else "n/a"


def find_discrepancies(view: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame,
                       municipal: pd.DataFrame, utility: pd.DataFrame, profile: str = "ai") -> gpd.GeoDataFrame:
    """view: unified per-parcel layer (see harmonize.unified_view);
    buildings: matched detections (see match.match_buildings);
    profile: which building source produced `buildings` (see config.RULE_PROFILES)."""
    prof = config.RULE_PROFILES[profile]
    rows: list[dict] = []

    def add(pid, kind, evidence, sources, geom=None):
        rows.append({"parcel_id": pid, "issue": kind, "severity": SEVERITY[kind],
                     "evidence": evidence, "sources_compared": sources, "action": ACTION, "geometry": geom})

    has_rec = view.municipal_pid.notna()
    rec_bld = view.has_building.fillna(False).astype(bool)
    det = view.detected_buildings

    for r in view[has_rec & ~rec_bld & (det > 0)].itertuples():
        add(r.parcel_id, "Unrecorded building",
            f"AI detected {r.detected_buildings} building(s), {r.detected_builtup_sqm:.0f} m² inside parcel; "
            f"municipal record {r.municipal_pid} says no building",
            "imagery-AI × cadastral × municipal", r.geometry)

    for r in view[has_rec & rec_bld & (det == 0)].itertuples():
        add(r.parcel_id, "Recorded building not found",
            f"Municipal record {r.municipal_pid} lists {r.recorded_builtup_sqm:.0f} m² built-up; "
            f"no building detected in imagery", "municipal × imagery-AI", r.geometry)

    on_parcel = buildings[buildings.parcel_id.notna()]
    crossing = on_parcel[(on_parcel.n_parcels >= 2) & (on_parcel.overlap_ratio < config.ENCROACHMENT_MIN_OVERLAP)]
    single = ((crossing.n_parcels == 2) & (crossing.area_sqm <= config.MAX_SINGLE_STRUCTURE_SQM)
              & (crossing.overlap_ratio <= prof["encroach_max_overlap"]))
    for b in crossing[single].itertuples():
        own = f"; {b.intrudes_on} also has its own structure" if b.intrudes_on else ""
        add(b.parcel_id, "Boundary encroachment",
            f"Structure {b.building_id} ({b.area_sqm:.0f} m²): {b.overlap_ratio:.0%} in {b.parcel_id}, "
            f"remainder crosses into {b.also_touches}{own}", "imagery-AI × cadastral", b.geometry)
    spill = on_parcel[(on_parcel.n_parcels == 1) & (on_parcel.overlap_ratio >= 0.5)
                      & (on_parcel.outside_ratio > config.MAX_OUTSIDE_RATIO)
                      & (on_parcel.area_sqm >= config.MIN_SPILL_STRUCTURE_SQM)]
    for b in spill.itertuples():
        add(b.parcel_id, "Boundary encroachment",
            f"Structure {b.building_id} ({b.area_sqm:.0f} m²): {b.outside_ratio:.0%} lies outside every parcel "
            f"(road / public land)", "imagery-AI × cadastral", b.geometry)
    for b in crossing[~single].itertuples():
        add(b.parcel_id, "Structure spans multiple parcels",
            f"Structure {b.building_id} ({b.area_sqm:.0f} m²) covers {b.n_parcels} parcels "
            f"({b.parcel_id}, {b.also_touches}) — attached construction or merged AI detection",
            "imagery-AI × cadastral", b.geometry)

    ratio = (view.recorded_area_sqm - view.computed_area_sqm).abs() / view.computed_area_sqm
    for r in view[ratio > config.AREA_MISMATCH_TOL].itertuples():
        add(r.parcel_id, "Parcel area mismatch",
            f"Recorded {r.recorded_area_sqm:.0f} m² vs {r.computed_area_sqm:.0f} m² measured from boundary "
            f"({_pct(r.recorded_area_sqm, r.computed_area_sqm)})", "cadastral attributes × cadastral geometry", r.geometry)

    bu = view[has_rec & rec_bld & (det > 0) & (view.recorded_builtup_sqm > 0)]
    bu_ratio = bu.detected_builtup_sqm / bu.recorded_builtup_sqm
    for r in bu[bu_ratio > prof["builtup_ratio_high"]].itertuples():
        add(r.parcel_id, "Built-up area mismatch",
            f"Municipal built-up {r.recorded_builtup_sqm:.0f} m² vs AI-measured {r.detected_builtup_sqm:.0f} m² "
            f"({_pct(r.recorded_builtup_sqm, r.detected_builtup_sqm)})", "municipal × imagery-AI", r.geometry)

    conn = view.electricity_conn.fillna(False).astype(bool) | view.water_conn.fillna(False).astype(bool)
    # utility disagrees with BOTH other sources (municipal says no building, imagery sees none);
    # if municipal claims a building, "Recorded building not found" already covers the parcel
    for r in view[conn & (det == 0) & ~rec_bld].itertuples():
        kinds = " & ".join(k for k, v in [("electricity", r.electricity_conn), ("water", r.water_conn)]
                         if pd.notna(v) and bool(v))
        add(r.parcel_id, "Utility on vacant parcel",
            f"Active {kinds} connection ({r.utility_ref}) but municipal record shows no building "
            f"and none is visible in imagery", "utility × municipal × imagery-AI", r.geometry)

    for r in view[~has_rec].itertuples():
        add(r.parcel_id, "No municipal record",
            f"Parcel {r.survey_no} exists in cadastral map but has no municipal property record",
            "cadastral × municipal", r.geometry)

    known = set(view.parcel_id)
    for df, label, ref in [(municipal, "Municipal", "municipal_pid"), (utility, "Utility", "utility_ref")]:
        for r in df[~df.parcel_id.isin(known)].itertuples():
            add(r.parcel_id or "?", "Orphan record",
                f"{label} record {getattr(r, ref)} refers to a parcel absent from the cadastral map",
                f"{label.lower()} × cadastral")

    out = gpd.GeoDataFrame(rows, columns=["parcel_id", "issue", "severity", "evidence", "sources_compared",
                                          "action", "geometry"], geometry="geometry", crs=view.crs)
    order = {k: i for i, k in enumerate(["High", "Medium", "Low"])}
    return out.sort_values(["severity", "issue", "parcel_id"], key=lambda s: s.map(order) if s.name == "severity" else s
                           ).reset_index(drop=True)
