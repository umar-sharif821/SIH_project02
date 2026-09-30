"""Build the demo datasets in data/raw/.

Real, open data:
  * imagery.tif          Esri World Imagery tiles for the AOI (EPSG:3857 GeoTIFF)
  * osm_buildings.geojson OpenStreetMap building footprints (EPSG:4326)
  * osm_roads.geojson     OpenStreetMap roads (EPSG:4326)

SYNTHETIC data (clearly labelled — no real land records are used):
  * cadastral_parcels.geojson  parcels generated around OSM buildings (EPSG:32643, "Survey_No")
  * municipal_records.csv      property-tax style records ("PID", messy Yes/No values)
  * utility_connections.csv    water / electricity connections ("parcel_ref")
  * ground_truth_discrepancies.json  the errors deliberately injected, for verification

Each source uses a different CRS, column names and ID format on purpose, so the
pipeline has something real to harmonize.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SECTOR = 22
FIRST_PARCEL_NO = 101
PLOT_MARGIN_M = 12          # max yard width around a building
VACANT_HALF_SIDE_M = 16     # vacant plots are at most 32 × 32 m
ROAD_WIDTH_M = {
    "primary": 12, "secondary": 10, "tertiary": 8,
    "residential": 6, "unclassified": 6, "living_street": 5, "service": 4,
}


# ---------------------------------------------------------------- downloads
def download_imagery(path: Path) -> None:
    import contextily as ctx

    m = 0.0006  # ~60 m margin so parcels on the AOI edge are fully covered
    w, s, e, n = config.AOI_BBOX
    w, s, e, n = w - m, s - m, e + m, n + m
    source = ctx.providers
    for part in config.IMAGERY_SOURCE.split("."):
        source = source[part]
    ctx.bounds2raster(w, s, e, n, str(path), zoom=config.IMAGERY_ZOOM, source=source, ll=True)
    print(f"  imagery  -> {path.name}")


def download_osm() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    import osmnx as ox

    ox.settings.useragent = "sih26013-prototype/0.1"
    ox.settings.requests_timeout = 180
    ox.settings.use_cache = False
    bbox = config.AOI_BBOX  # (left, bottom, right, top)

    def fetch(tags):
        for url in config.OVERPASS_URLS:
            ox.settings.overpass_url = url
            try:
                return ox.features_from_bbox(bbox, tags=tags).reset_index()
            except Exception as e:  # busy server → next mirror
                print(f"  Overpass {url} failed: {type(e).__name__}")
        raise RuntimeError("all Overpass servers failed — retry later")

    b = fetch({"building": True})
    b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    b = b.rename(columns={"id": "osm_id"})[["osm_id", "building", "geometry"]]
    b["building"] = b["building"].astype(str)

    r = fetch({"highway": list(ROAD_WIDTH_M)})
    r = r[r.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    r = r.rename(columns={"id": "osm_id"})[["osm_id", "highway", "geometry"]]
    return b.reset_index(drop=True), r.reset_index(drop=True)


# ---------------------------------------------------------------- parcels
def build_parcels(buildings_m: gpd.GeoDataFrame, roads_m: gpd.GeoDataFrame,
                  aoi_m, rng: random.Random) -> gpd.GeoDataFrame:
    """Generalized Voronoi parcels: every building gets the land closest to it,
    plus a few vacant plots; road corridors are cut out."""
    bgeoms = list(buildings_m.geometry)
    union_b = shapely.union_all(bgeoms)

    road_area = shapely.union_all([
        g.buffer(ROAD_WIDTH_M.get(h, 5) / 2, cap_style="flat")
        for g, h in zip(roads_m.geometry, roads_m.highway)
    ]) if len(roads_m) else shapely.Polygon()
    road_area = road_area.difference(union_b.buffer(1.0))

    # vacant plot seeds: open land far from buildings and roads
    open_land = aoi_m.buffer(-10).difference(union_b.buffer(14)).difference(road_area.buffer(8))
    vacant: list[Point] = []
    minx, miny, maxx, maxy = aoi_m.bounds
    for _ in range(4000):
        if len(vacant) >= 8 or open_land.is_empty:
            break
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if open_land.contains(p) and all(p.distance(q) > 30 for q in vacant):
            vacant.append(p)

    # seed points: building outlines sampled every 2 m, plus vacant points
    seeds, owner = [], []
    for i, g in enumerate(bgeoms):
        pts = shapely.get_coordinates(shapely.segmentize(g.boundary, 2.0))
        seeds.extend(map(tuple, pts))
        owner.extend([i] * len(pts))
    for j, p in enumerate(vacant):
        seeds.append((p.x, p.y))
        owner.append(len(bgeoms) + j)

    seed_pts = gpd.GeoDataFrame({"owner": owner}, geometry=gpd.points_from_xy(*zip(*seeds)),
                                crs=buildings_m.crs).drop_duplicates(subset="geometry")
    cells = shapely.voronoi_polygons(MultiPoint(list(seed_pts.geometry)), extend_to=aoi_m.buffer(50))
    cells = gpd.GeoDataFrame(geometry=list(cells.geoms), crs=buildings_m.crs)
    cells = gpd.sjoin(cells, seed_pts, predicate="contains", how="inner")
    parcels = cells.dissolve(by="owner").reset_index()[["owner", "geometry"]]

    # plot-sized parcels: never more than a yard's width from the building / vacant seed
    reach = [bgeoms[o].buffer(PLOT_MARGIN_M, join_style="mitre") if o < len(bgeoms)
             else vacant[o - len(bgeoms)].buffer(VACANT_HALF_SIDE_M, cap_style="square")
             for o in parcels.owner]
    parcels["geometry"] = (parcels.geometry.intersection(gpd.GeoSeries(reach, index=parcels.index, crs=parcels.crs))
                           .intersection(aoi_m).difference(road_area).buffer(0).simplify(0.5))
    # keep only the part of each cell that holds its seed feature
    keep = []
    for own, g in zip(parcels.owner, parcels.geometry):
        anchor = bgeoms[own].representative_point() if own < len(bgeoms) else vacant[own - len(bgeoms)]
        parts = list(getattr(g, "geoms", [g]))
        best = next((p for p in parts if p.buffer(0.5).contains(anchor)), max(parts, key=lambda p: p.area))
        keep.append(best)
    parcels["geometry"] = keep
    parcels["vacant"] = parcels.owner >= len(bgeoms)
    parcels = parcels[parcels.area > 40].copy()

    # parcel numbers ordered north→south, west→east
    c = parcels.geometry.centroid
    parcels["_row"] = (-c.y // 40).astype(int)
    parcels = parcels.assign(_x=c.x).sort_values(["_row", "_x"]).reset_index(drop=True)
    parcels["parcel_no"] = range(FIRST_PARCEL_NO, FIRST_PARCEL_NO + len(parcels))
    return parcels.drop(columns=["_row", "_x", "owner"])


def inject_encroachment(parcels: gpd.GeoDataFrame, buildings_m: gpd.GeoDataFrame,
                        idx: int) -> bool:
    """Move part of parcel idx's building into a neighbouring parcel, so the building
    crosses the recorded boundary (the classic 'boundary encroachment' case)."""
    P = parcels.geometry.iloc[idx]
    inside = buildings_m[buildings_m.intersects(P)]
    if inside.empty:
        return False
    bld = inside.geometry.iloc[inside.area.argmax()]
    bx0, by0, bx1, by1 = bld.bounds
    px0, py0, px1, py1 = P.bounds
    cut = 0.35
    slabs = {
        "east": box(bx1 - cut * (bx1 - bx0), py0 - 1, px1 + 1, py1 + 1),
        "west": box(px0 - 1, py0 - 1, bx0 + cut * (bx1 - bx0), py1 + 1),
        "north": box(px0 - 1, by1 - cut * (by1 - by0), px1 + 1, py1 + 1),
        "south": box(px0 - 1, py0 - 1, px1 + 1, by0 + cut * (by1 - by0)),
    }
    for slab in slabs.values():
        R = P.intersection(slab)
        if R.is_empty or R.area < 5:
            continue
        others = parcels.drop(index=parcels.index[idx])
        shared = others.geometry.boundary.intersection(R.buffer(0.3)).length
        if shared.max() < 3:
            continue
        n = shared.idxmax()
        parcels.loc[parcels.index[idx], "geometry"] = P.difference(R).buffer(0)
        parcels.loc[n, "geometry"] = shapely.union(parcels.loc[n, "geometry"], R).buffer(0.01).buffer(-0.01)
        return True
    return False


# ---------------------------------------------------------------- main
def main() -> None:
    rng = random.Random(config.SEED)
    np.random.seed(config.SEED)
    raw = config.RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)

    img = raw / "imagery.tif"
    if not img.exists():
        download_imagery(img)
    else:
        print("  imagery  (cached)")

    b_path, r_path = raw / "osm_buildings.geojson", raw / "osm_roads.geojson"
    if not b_path.exists():
        b, r = download_osm()
        b.to_file(b_path, driver="GeoJSON")
        r.to_file(r_path, driver="GeoJSON")
    buildings = gpd.read_file(b_path)
    roads = gpd.read_file(r_path)
    print(f"  OSM      -> {len(buildings)} buildings, {len(roads)} road segments")

    aoi_m = gpd.GeoSeries([box(*config.AOI_BBOX)], crs=config.DISPLAY_CRS).to_crs(config.METRIC_CRS).iloc[0]
    b_m = buildings.to_crs(config.METRIC_CRS)
    b_m = b_m[b_m.intersects(aoi_m) & (b_m.area > 8)].reset_index(drop=True)
    r_m = roads.to_crs(config.METRIC_CRS)
    # parcel fabric covers the AOI plus any building straddling its edge, so no
    # building is artificially cut by the AOI box
    extent = shapely.union(aoi_m, shapely.union_all(list(b_m.buffer(3).envelope)))
    extent = shapely.union(aoi_m, extent.intersection(aoi_m.buffer(60)))

    parcels = build_parcels(b_m, r_m, extent, rng)
    truth: list[dict] = []
    pid = lambda i: f"P{int(parcels.parcel_no.iloc[i])}"  # noqa: E731

    # which parcel each OSM building belongs to (largest overlap)
    def assign(parcels_now):
        ov = gpd.overlay(b_m.reset_index(names="bid"), parcels_now.reset_index(names="pidx"),
                         how="intersection", keep_geom_type=True)
        ov["a"] = ov.area
        best = ov.sort_values("a").groupby("bid").tail(1)
        return best.groupby("pidx")["a"].sum()

    built = parcels.index.isin(assign(parcels).index)
    built_idx = [i for i in range(len(parcels)) if built[i]]
    vacant_idx = [i for i in range(len(parcels)) if not built[i]]
    rng.shuffle(built_idx)
    rng.shuffle(vacant_idx)
    print(f"  parcels  -> {len(parcels)} ({len(built_idx)} built, {len(vacant_idx)} vacant)")
    if len(vacant_idx) < 4 or len(built_idx) < 12:
        sys.exit("AOI too small or too sparse for the demo — enlarge AOI_BBOX in config.py")

    # --- encroachment (geometry change must happen before areas are recorded)
    enc = 0
    for i in built_idx[:]:
        if enc == 2:
            break
        if inject_encroachment(parcels, b_m, i):
            truth.append({"parcel_id": pid(i), "type": "Boundary encroachment"})
            built_idx.remove(i)
            enc += 1
    built_area = assign(parcels)

    # --- cadastral layer (UTM, "Survey_No" = "22/<no>")
    cad = gpd.GeoDataFrame({
        "Survey_No": [f"{SECTOR}/{n}" for n in parcels.parcel_no],
        "Sector": SECTOR,
        "Land_Use": np.where(parcels.index.isin(built_area.index), "Built-up", "Vacant"),
        "Area_SqM": parcels.area.round(1),
    }, geometry=parcels.geometry, crs=config.METRIC_CRS)
    for i in built_idx[:2]:
        cad.loc[i, "Area_SqM"] = round(cad.loc[i, "Area_SqM"] * rng.choice([1.25, 0.78]), 1)
        truth.append({"parcel_id": pid(i), "type": "Parcel area mismatch"})
    area_mis = built_idx[:2]

    # --- municipal records ("PID" = "MC-022-0<no>", messy Yes/No)
    yes, no = ["Yes", "Y", "yes", "1"], ["No", "N", "no", "0"]
    rows = []
    for i in range(len(parcels)):
        a = float(built_area.get(i, 0.0))
        rows.append({
            "PID": f"MC-0{SECTOR}-{int(parcels.parcel_no.iloc[i]):04d}",
            "Property_Type": "Residential" if a else "Plot",
            "Has_Building": rng.choice(yes) if a else rng.choice(no),
            "Built_Up_Area_SqM": round(a * rng.uniform(0.95, 1.05), 1) if a else 0.0,
            "Last_Assessed": rng.choice([2021, 2022, 2023, 2024]),
        })
    muni = pd.DataFrame(rows)

    pool = [i for i in built_idx if i not in area_mis]
    for i in pool[:3]:
        muni.loc[i, ["Has_Building", "Built_Up_Area_SqM", "Property_Type"]] = [rng.choice(no), 0.0, "Plot"]
        truth.append({"parcel_id": pid(i), "type": "Unrecorded building"})
    for i in pool[3:5]:
        muni.loc[i, "Built_Up_Area_SqM"] = round(muni.loc[i, "Built_Up_Area_SqM"] * 0.35, 1)
        truth.append({"parcel_id": pid(i), "type": "Built-up area mismatch"})
    for i in vacant_idx[:2]:
        muni.loc[i, ["Has_Building", "Built_Up_Area_SqM", "Property_Type"]] = [rng.choice(yes), 140.0, "Residential"]
        truth.append({"parcel_id": pid(i), "type": "Recorded building not found"})
    missing = pool[5]
    muni = muni.drop(index=missing)
    truth.append({"parcel_id": pid(missing), "type": "No municipal record"})
    for n in (901, 902):
        muni.loc[len(muni) + n] = [f"MC-0{SECTOR}-{n:04d}", "Residential", "Yes", 180.0, 2020]
        truth.append({"parcel_id": f"P{n}", "type": "Orphan record"})

    # --- utility connections ("parcel_ref" = "S22-P<no>")
    util = pd.DataFrame({
        "parcel_ref": [f"S{SECTOR}-P{n}" for n in parcels.parcel_no],
        "Water_Conn": ["Y" if i in built_area.index else "N" for i in range(len(parcels))],
        "Elec_Conn": ["Y" if i in built_area.index else "N" for i in range(len(parcels))],
    })
    for i in vacant_idx[2:4]:
        util.loc[i, "Elec_Conn"] = "Y"
        truth.append({"parcel_id": pid(i), "type": "Utility on vacant parcel"})

    cad.to_file(raw / "cadastral_parcels.geojson", driver="GeoJSON")
    muni.sample(frac=1, random_state=config.SEED).to_csv(raw / "municipal_records.csv", index=False)
    util.to_csv(raw / "utility_connections.csv", index=False)
    (raw / "ground_truth_discrepancies.json").write_text(json.dumps(truth, indent=2))
    print(f"  synthetic cadastral / municipal / utility written; {len(truth)} discrepancies injected")


if __name__ == "__main__":
    main()
