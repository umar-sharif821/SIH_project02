"""AI feature extraction: building footprints from the orthoimagery.

Uses a pretrained YOLOv8 instance-segmentation model (no training). The GeoTIFF is
cut into overlapping tiles, each mask polygon is mapped from pixel to map
coordinates with the raster's affine transform, and tile-edge duplicates are merged.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

import config

AI_SOURCE = "AI (YOLOv8-seg, pretrained)"
OSM_SOURCE = "OSM reference (pre-extracted)"


def load_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights = hf_hub_download(config.HF_MODEL_REPO, config.HF_MODEL_FILE, local_dir=config.MODEL_DIR)
    return YOLO(weights)


def _tiles(h: int, w: int, size: int, overlap: int):
    step = size - overlap
    ys = list(range(0, max(h - size, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - size, 0) + 1, step)) or [0]
    if ys[-1] + size < h:
        ys.append(h - size)
    if xs[-1] + size < w:
        xs.append(w - size)
    return [(y, x) for y in ys for x in xs]


def detect_buildings(image_path, model=None, conf: float = config.DETECT_CONF,
                     scale: float = config.DETECT_SCALE, progress=None) -> gpd.GeoDataFrame:
    import cv2
    import rasterio

    model = model or load_model()
    with rasterio.open(image_path) as ds:
        rgb = ds.read([1, 2, 3])
        transform, crs = ds.transform, ds.crs
    img = np.ascontiguousarray(rgb.transpose(1, 2, 0)[:, :, ::-1])  # RGB -> BGR for Ultralytics
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    size, overlap = config.DETECT_TILE_PX, config.DETECT_OVERLAP_PX
    tiles = _tiles(H, W, size, overlap)

    records = []
    for k, (y, x) in enumerate(tiles):
        tile = img[y:y + size, x:x + size]
        res = model.predict(tile, conf=conf, imgsz=size, verbose=False, retina_masks=True)[0]
        if res.masks is not None:
            th, tw = tile.shape[:2]
            for poly, c in zip(res.masks.xy, res.boxes.conf.cpu().numpy()):
                if len(poly) < 3:
                    continue
                touches_edge = (
                    (poly[:, 0].min() < 3 and x > 0) or (poly[:, 1].min() < 3 and y > 0)
                    or (poly[:, 0].max() > tw - 3 and x + tw < W) or (poly[:, 1].max() > th - 3 and y + th < H)
                )
                cols = (poly[:, 0] + x) / scale
                rows = (poly[:, 1] + y) / scale
                xs, ys = transform * (cols, rows)
                g = Polygon(zip(xs, ys)).buffer(0)
                if not g.is_empty:
                    records.append({"confidence": float(c), "edge": touches_edge, "geometry": g})
        if progress:
            progress((k + 1) / len(tiles))

    if not records:
        return _empty()
    gdf = gpd.GeoDataFrame(records, crs=crs).to_crs(config.METRIC_CRS)
    gdf = _merge_tiles(gdf)
    gdf["geometry"] = gdf.geometry.simplify(0.4).buffer(0)
    gdf = gdf[gdf.area >= config.DETECT_MIN_AREA_SQM].reset_index(drop=True)
    gdf["building_id"] = [f"D{i + 1:04d}" for i in range(len(gdf))]
    gdf["area_sqm"] = gdf.area.round(1)
    gdf["source"] = AI_SOURCE
    return gdf[["building_id", "confidence", "area_sqm", "source", "geometry"]]


def second_look(image_path, parcels: gpd.GeoDataFrame, model=None,
                conf: float = config.RECHECK_CONF, crop_px: int = config.RECHECK_CROP_PX) -> gpd.GeoDataFrame:
    """Targeted re-examination of specific parcels: a crop centred on each parcel is
    enlarged to the model's input size, which recovers buildings the full-image pass
    missed. Used only where another source says a building should exist."""
    import rasterio

    if parcels.empty:
        return _empty()
    model = model or load_model()
    with rasterio.open(image_path) as ds:
        img = np.ascontiguousarray(ds.read([1, 2, 3]).transpose(1, 2, 0)[:, :, ::-1])
        transform, crs = ds.transform, ds.crs
    H, W = img.shape[:2]
    targets = parcels.to_crs(crs)
    records = []
    for pid, g in zip(targets.parcel_id, targets.geometry):
        col, row = ~transform * (g.centroid.x, g.centroid.y)
        y0 = int(min(max(row - crop_px // 2, 0), H - crop_px))
        x0 = int(min(max(col - crop_px // 2, 0), W - crop_px))
        res = model.predict(img[y0:y0 + crop_px, x0:x0 + crop_px], conf=conf, imgsz=config.DETECT_TILE_PX,
                            verbose=False, retina_masks=True)[0]
        if res.masks is None:
            continue
        for poly, c in zip(res.masks.xy, res.boxes.conf.cpu().numpy()):
            if len(poly) < 3:
                continue
            xs, ys = transform * (poly[:, 0] + x0, poly[:, 1] + y0)
            p = Polygon(zip(xs, ys)).buffer(0)
            if p.intersects(g):
                records.append({"confidence": float(c), "checked_parcel": pid, "geometry": p})
    if not records:
        return _empty()
    gdf = gpd.GeoDataFrame(records, crs=crs).to_crs(config.METRIC_CRS)
    gdf = gdf[gdf.area >= config.DETECT_MIN_AREA_SQM].reset_index(drop=True)
    gdf["building_id"] = [f"S{i + 1:04d}" for i in range(len(gdf))]
    gdf["area_sqm"] = gdf.area.round(1)
    gdf["source"] = AI_SOURCE + " — second look"
    return gdf[["building_id", "confidence", "area_sqm", "source", "geometry"]]


def _merge_tiles(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Non-maximum suppression across overlapping tiles: prefer detections that are
    fully inside a tile; edge-cut fragments are kept only where nothing else covers them."""
    kept: list[dict] = []

    def overlaps(g, frac):
        return any(g.intersection(k["geometry"]).area > frac * min(g.area, k["geometry"].area) for k in kept)

    for _, r in gdf[~gdf.edge].sort_values("confidence", ascending=False).iterrows():
        if not overlaps(r.geometry, 0.5):
            kept.append(r.to_dict())
    frags = gdf[gdf.edge]
    frags = frags[[not overlaps(g, 0.3) for g in frags.geometry]]
    if len(frags):
        merged = shapely.union_all(list(frags.geometry))
        for part in getattr(merged, "geoms", [merged]):
            c = frags[frags.intersects(part)].confidence.max()
            kept.append({"confidence": float(c), "edge": True, "geometry": part})
    return gpd.GeoDataFrame(kept, crs=gdf.crs)


def _empty() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["building_id", "confidence", "area_sqm", "source", "geometry"],
                            geometry="geometry", crs=config.METRIC_CRS)


def reference_buildings(osm_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fallback 'detections' from pre-extracted OSM footprints."""
    g = osm_m[["geometry"]].copy().reset_index(drop=True)
    g["building_id"] = [f"R{i + 1:04d}" for i in range(len(g))]
    g["confidence"] = np.nan
    g["area_sqm"] = g.area.round(1)
    g["source"] = OSM_SOURCE
    return g[["building_id", "confidence", "area_sqm", "source", "geometry"]]


def evaluate(detected: gpd.GeoDataFrame, reference: gpd.GeoDataFrame, area=None,
             iou_threshold: float = 0.3) -> dict:
    """Object-level precision / recall against OSM footprints (one-to-one at IoU >= threshold),
    counting only features whose centre lies in `area` (where the reference is complete)."""
    ref = reference
    if area is not None:
        ref = ref[ref.representative_point().within(area)].reset_index(drop=True)
        detected = detected[detected.representative_point().within(area)].reset_index(drop=True)
    if detected.empty or ref.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "detected": len(detected), "reference": len(ref), "matched": 0}
    pairs = gpd.sjoin(detected[["geometry"]].reset_index(names="d"),
                      ref[["geometry"]].reset_index(names="r"), predicate="intersects")
    rg = ref.geometry
    pairs["iou"] = [
        detected.geometry.iloc[d].intersection(rg.loc[r]).area / detected.geometry.iloc[d].union(rg.loc[r]).area
        for d, r in zip(pairs.d, pairs.r)
    ]
    pairs = pairs[pairs.iou >= iou_threshold].sort_values("iou", ascending=False)
    used_d, used_r = set(), set()
    for d, r in zip(pairs.d, pairs.r):
        if d not in used_d and r not in used_r:
            used_d.add(d)
            used_r.add(r)
    m = len(used_d)
    p, rc = m / len(detected), m / len(ref)
    return {"precision": round(p, 3), "recall": round(rc, 3),
            "f1": round(2 * p * rc / (p + rc), 3) if m else 0.0,
            "detected": len(detected), "reference": len(ref), "matched": m, "iou_threshold": iou_threshold}


def cached(mode: str) -> gpd.GeoDataFrame | None:
    p = config.PROCESSED_DIR / f"detected_buildings_{mode}.geojson"
    return gpd.read_file(p).to_crs(config.METRIC_CRS) if p.exists() else None


def save(gdf: gpd.GeoDataFrame, mode: str) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = gdf.copy()
    out["confidence"] = pd.to_numeric(out.confidence)
    out.to_crs(config.DISPLAY_CRS).to_file(config.PROCESSED_DIR / f"detected_buildings_{mode}.geojson", driver="GeoJSON")
