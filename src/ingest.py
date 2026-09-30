"""Load each raw source as-is. Every source keeps its own CRS, columns and ID format;
the SOURCES registry records how each one maps onto the common schema."""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

import config


@dataclass
class Source:
    key: str
    label: str
    department: str
    file: str
    kind: str                       # "vector" | "table" | "raster"
    id_column: str | None = None    # column holding the parcel reference
    id_regex: str | None = None     # regex whose first group is the parcel number
    column_map: dict[str, str] = field(default_factory=dict)  # source col -> common col


SOURCES: dict[str, Source] = {s.key: s for s in [
    Source("cadastral", "Cadastral parcels", "Revenue Dept. (synthetic)",
           "cadastral_parcels.geojson", "vector",
           id_column="Survey_No", id_regex=r"^\d+/(\d+)$",
           column_map={"Survey_No": "survey_no", "Area_SqM": "recorded_area_sqm",
                       "Land_Use": "land_use"}),
    Source("municipal", "Municipal property records", "Municipal Corporation (synthetic)",
           "municipal_records.csv", "table",
           id_column="PID", id_regex=r"^MC-\d+-0*(\d+)$",
           column_map={"PID": "municipal_pid", "Has_Building": "has_building",
                       "Built_Up_Area_SqM": "recorded_builtup_sqm",
                       "Property_Type": "property_type", "Last_Assessed": "last_assessed"}),
    Source("utility", "Utility connections", "Electricity & Water Boards (synthetic)",
           "utility_connections.csv", "table",
           id_column="parcel_ref", id_regex=r"^S\d+-P(\d+)$",
           column_map={"parcel_ref": "utility_ref", "Water_Conn": "water_conn",
                       "Elec_Conn": "electricity_conn"}),
    Source("osm", "Reference building footprints", "OpenStreetMap contributors",
           "osm_buildings.geojson", "vector"),
    Source("imagery", "Aerial imagery (ORI)", "Esri World Imagery",
           "imagery.tif", "raster"),
]}


def path(key: str):
    return config.RAW_DIR / SOURCES[key].file


def load(key: str):
    src = SOURCES[key]
    p = path(key)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `python scripts/prepare_data.py` first")
    if src.kind == "vector":
        return gpd.read_file(p)
    if src.kind == "table":
        return pd.read_csv(p, dtype=str)
    return p  # rasters are opened lazily by path


def load_all() -> dict:
    return {k: load(k) for k in SOURCES}


def describe(key: str, obj) -> dict:
    """Summary used by the dashboard's 'Sources' tab."""
    src = SOURCES[key]
    info = {"Source": src.label, "Provider": src.department, "File": src.file, "Type": src.kind}
    if src.kind == "vector":
        info.update(Records=len(obj), CRS=str(obj.crs), Columns=", ".join(c for c in obj.columns if c != "geometry"))
    elif src.kind == "table":
        info.update(Records=len(obj), CRS="none (non-spatial)", Columns=", ".join(obj.columns))
    else:
        import rasterio
        with rasterio.open(obj) as ds:
            res = abs(ds.transform.a)
            info.update(Records=f"{ds.width}×{ds.height} px, {ds.count} bands", CRS=str(ds.crs),
                        Columns=f"pixel size {res:.2f} (CRS units)")
    info["ID format"] = src.id_column or "—"
    return info
