"""Standardize every source onto one schema: common CRS, one parcel ID, clean types."""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

import config
from src.ingest import SOURCES

TRUE_VALUES = {"yes", "y", "1", "true", "t"}
FALSE_VALUES = {"no", "n", "0", "false", "f", ""}


@dataclass
class Harmonized:
    parcels: gpd.GeoDataFrame       # metric CRS, one row per parcel
    municipal: pd.DataFrame
    utility: pd.DataFrame
    osm: gpd.GeoDataFrame           # metric CRS
    log: list[str] = field(default_factory=list)


def canonical_id(number) -> str:
    return f"P{int(number)}"


def normalize_ids(df: pd.DataFrame, key: str, log: list[str]) -> pd.DataFrame:
    src = SOURCES[key]
    nums = df[src.id_column].astype(str).str.strip().str.extract(src.id_regex, expand=False)
    bad = nums.isna().sum()
    df = df.assign(parcel_id=nums.map(lambda n: canonical_id(n) if isinstance(n, str) else None))
    example = f"{df[src.id_column].iloc[0]} → {df.parcel_id.iloc[0]}"
    log.append(f"{src.label}: parcel ID '{src.id_column}' normalized (e.g. {example})"
               + (f"; {bad} unparseable IDs" if bad else ""))
    return df


def parse_bool(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.strip().str.lower()
    unknown = ~v.isin(TRUE_VALUES | FALSE_VALUES)
    if unknown.any():
        raise ValueError(f"Unrecognised yes/no values: {sorted(v[unknown].unique())}")
    return v.isin(TRUE_VALUES)


def to_metric(gdf: gpd.GeoDataFrame, label: str, log: list[str]) -> gpd.GeoDataFrame:
    before = str(gdf.crs)
    out = gdf.to_crs(config.METRIC_CRS)
    log.append(f"{label}: CRS {before} → {config.METRIC_CRS}" if before != config.METRIC_CRS
               else f"{label}: already in {config.METRIC_CRS}")
    return out


def harmonize(raw: dict) -> Harmonized:
    log: list[str] = []

    cad = normalize_ids(raw["cadastral"], "cadastral", log)
    cad = to_metric(cad, SOURCES["cadastral"].label, log)
    cad = cad.rename(columns=SOURCES["cadastral"].column_map)
    cad["recorded_area_sqm"] = pd.to_numeric(cad.recorded_area_sqm)
    cad["computed_area_sqm"] = cad.area.round(1)
    cad = cad[["parcel_id", "survey_no", "land_use", "recorded_area_sqm", "computed_area_sqm", "geometry"]]

    muni = normalize_ids(raw["municipal"], "municipal", log).rename(columns=SOURCES["municipal"].column_map)
    variants = sorted(raw["municipal"]["Has_Building"].astype(str).unique())
    muni["has_building"] = parse_bool(muni.has_building)
    muni["recorded_builtup_sqm"] = pd.to_numeric(muni.recorded_builtup_sqm)
    log.append(f"{SOURCES['municipal'].label}: 'Has_Building' values {variants} → true/false")

    util = normalize_ids(raw["utility"], "utility", log).rename(columns=SOURCES["utility"].column_map)
    util["water_conn"] = parse_bool(util.water_conn)
    util["electricity_conn"] = parse_bool(util.electricity_conn)

    osm = to_metric(raw["osm"], SOURCES["osm"].label, log)
    log.append(f"All measurements (area, overlap) computed in {config.METRIC_CRS}; maps shown in {config.DISPLAY_CRS}")

    return Harmonized(cad.reset_index(drop=True), muni, util, osm, log)


def estimate_shift(moving: gpd.GeoSeries, fixed: gpd.GeoSeries,
                   max_shift: float = config.MAX_COREG_SHIFT_M, res: float = 0.5) -> tuple[float, float, float]:
    """Translation (dx, dy) in metres that best lays `moving` over `fixed`, found by
    FFT cross-correlation of the two rasterized layers. Returns (dx, dy, overlap gain)."""
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    x0, y0, x1, y1 = np.array([*moving.total_bounds[:2], *moving.total_bounds[2:]])
    fx0, fy0, fx1, fy1 = fixed.total_bounds
    x0, y0 = min(x0, fx0) - max_shift, min(y0, fy0) - max_shift
    x1, y1 = max(x1, fx1) + max_shift, max(y1, fy1) + max_shift
    shape = (int(np.ceil((y1 - y0) / res)), int(np.ceil((x1 - x0) / res)))
    tf = from_origin(x0, y1, res, res)
    A = rasterize(((g, 1) for g in moving), out_shape=shape, transform=tf, dtype="float32")
    B = rasterize(((g, 1) for g in fixed), out_shape=shape, transform=tf, dtype="float32")

    corr = np.real(np.fft.ifft2(np.fft.fft2(B) * np.conj(np.fft.fft2(A))))
    k = int(max_shift / res)
    rows = np.r_[0:k + 1, shape[0] - k:shape[0]]
    cols = np.r_[0:k + 1, shape[1] - k:shape[1]]
    win = corr[np.ix_(rows, cols)]
    i, j = np.unravel_index(np.argmax(win), win.shape)
    dr, dc = rows[i], cols[j]
    dr = dr - shape[0] if dr > shape[0] // 2 else dr
    dc = dc - shape[1] if dc > shape[1] // 2 else dc
    gain = win[i, j] / corr[0, 0] - 1 if corr[0, 0] > 0 else 0.0
    return float(dc * res), float(-dr * res), float(gain)


def coregister(h: Harmonized, detections: gpd.GeoDataFrame, log: list[str]) -> tuple[float, float]:
    """Align the vector layers (cadastral + reference footprints) to the orthoimagery,
    using AI-detected roofs as tie features. Applied only when it clearly helps."""
    aoi = h.parcels.union_all().envelope
    fixed = detections.geometry[detections.intersects(aoi)]
    moving = h.osm.geometry[h.osm.intersects(aoi)]
    if len(fixed) < 10 or len(moving) < 10:
        log.append("Co-registration: too few tie features — skipped")
        return 0.0, 0.0
    dx, dy, gain = estimate_shift(moving, fixed)
    if (dx ** 2 + dy ** 2) ** 0.5 < 0.5 or gain < config.MIN_COREG_GAIN:
        log.append(f"Co-registration: vector layers already aligned with imagery (best shift {dx:+.1f}, {dy:+.1f} m)")
        return 0.0, 0.0
    h.parcels["geometry"] = h.parcels.translate(dx, dy)
    h.osm["geometry"] = h.osm.translate(dx, dy)
    log.append(f"Co-registration: cadastral + reference layers shifted {dx:+.1f} m E, {dy:+.1f} m N to match "
               f"imagery (roof overlap +{gain:.0%})")
    return dx, dy


def unified_view(h: Harmonized, parcel_stats: pd.DataFrame) -> gpd.GeoDataFrame:
    """One row per parcel with every source's attributes side by side, plus provenance."""
    m = h.municipal.drop_duplicates("parcel_id")
    u = h.utility.drop_duplicates("parcel_id")
    view = (h.parcels
            .merge(m[["parcel_id", "municipal_pid", "has_building", "recorded_builtup_sqm", "property_type"]],
                   on="parcel_id", how="left")
            .merge(u[["parcel_id", "utility_ref", "water_conn", "electricity_conn"]], on="parcel_id", how="left")
            .merge(parcel_stats, on="parcel_id", how="left"))
    view["detected_buildings"] = view.detected_buildings.fillna(0).astype(int)
    view["detected_builtup_sqm"] = view.detected_builtup_sqm.fillna(0.0).round(1)
    view["sources"] = view.apply(
        lambda r: ", ".join(s for s, ok in [("cadastral", True),
                                            ("municipal", pd.notna(r.municipal_pid)),
                                            ("utility", pd.notna(r.utility_ref)),
                                            ("imagery-AI", True)] if ok), axis=1)
    return gpd.GeoDataFrame(view, geometry="geometry", crs=h.parcels.crs)
