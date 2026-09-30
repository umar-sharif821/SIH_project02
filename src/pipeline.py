"""End-to-end pipeline: ingest → harmonize → AI extraction → spatial matching → discrepancies.

CLI:  python -m src.pipeline [--mode ai|osm] [--redetect] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

import config
from src import detect, discrepancy, harmonize, ingest, match


@dataclass
class Result:
    raw: dict
    h: harmonize.Harmonized
    buildings: gpd.GeoDataFrame
    view: gpd.GeoDataFrame
    issues: gpd.GeoDataFrame
    metrics: dict
    mode: str
    log: list[str] = field(default_factory=list)


def get_buildings(h: harmonize.Harmonized, mode: str, redetect: bool, log: list[str], progress=None):
    if mode == "osm":
        log.append("Buildings: using OSM reference footprints (AI step skipped)")
        return detect.reference_buildings(h.osm), "osm"
    if not redetect and (b := detect.cached("ai")) is not None:
        log.append(f"Buildings: {len(b)} AI detections loaded from cache")
        return b, "ai"
    try:
        t = time.time()
        b = detect.detect_buildings(ingest.path("imagery"), progress=progress)
        detect.save(b, "ai")
        log.append(f"Buildings: AI extracted {len(b)} footprints in {time.time() - t:.0f}s")
        return b, "ai"
    except Exception as e:  # model download / inference failure → keep the demo running
        log.append(f"Buildings: AI extraction failed ({e!r}); falling back to OSM reference footprints")
        return detect.reference_buildings(h.osm), "osm"


def second_look(h: harmonize.Harmonized, b: gpd.GeoDataFrame, stats: pd.DataFrame,
                redetect: bool, log: list[str]) -> gpd.GeoDataFrame:
    """Re-examine parcels the municipal record calls built but where the first AI pass saw nothing."""
    seen = set(stats.loc[stats.detected_buildings > 0, "parcel_id"])
    claimed = set(h.municipal.loc[h.municipal.has_building, "parcel_id"])
    targets = h.parcels[h.parcels.parcel_id.isin(claimed - seen)]
    cache = config.PROCESSED_DIR / "detected_buildings_ai_second.geojson"
    ids_file = cache.with_suffix(".parcels.json")
    wanted = sorted(targets.parcel_id)
    if not redetect and cache.exists() and ids_file.exists() and json.loads(ids_file.read_text()) == wanted:
        extra = gpd.read_file(cache).to_crs(config.METRIC_CRS)
    else:
        try:
            extra = detect.second_look(ingest.path("imagery"), targets)
        except Exception as e:
            log.append(f"Second look: skipped ({e!r})")
            return b
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        extra.to_crs(config.DISPLAY_CRS).to_file(cache, driver="GeoJSON")
        ids_file.write_text(json.dumps(wanted))
    recovered = sorted({pid for pid, g in zip(targets.parcel_id, targets.geometry)
                        if extra.intersection(g).area.sum() >= config.MIN_PRESENCE_SQM}) if len(extra) else []
    log.append(f"Second look: re-examined {len(targets)} parcels recorded as built but not detected; "
               f"found buildings on {len(recovered)}" + (f" ({', '.join(recovered)})" if recovered else ""))
    return pd.concat([b[extra.columns.intersection(b.columns)], extra], ignore_index=True) if len(extra) else b


def run(mode: str = "ai", redetect: bool = False, progress=None) -> Result:
    raw = ingest.load_all()
    h = harmonize.harmonize(raw)
    log = list(h.log)

    b, mode = get_buildings(h, mode, redetect, log, progress)
    if mode == "ai":
        harmonize.coregister(h, b, log)
    b, stats = match.match_buildings(b, h.parcels)
    if mode == "ai":
        b = second_look(h, b, stats, redetect, log)
        b, stats = match.match_buildings(b, h.parcels)
    log.append(f"Matching: {b.parcel_id.notna().sum()} of {len(b)} buildings assigned to parcels")

    view = harmonize.unified_view(h, stats)
    issues = discrepancy.find_discrepancies(view, b, h.municipal, h.utility, profile=mode)

    metrics = evaluate(h, b) if mode == "ai" else {}

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    view.to_crs(config.DISPLAY_CRS).to_file(config.PROCESSED_DIR / f"harmonized_parcels_{mode}.geojson", driver="GeoJSON")
    issues.drop(columns="geometry").to_csv(config.PROCESSED_DIR / f"discrepancies_{mode}.csv", index=False)
    return Result(raw, h, b, view, issues, metrics, mode, log)


def evaluate(h: harmonize.Harmonized, b: gpd.GeoDataFrame) -> dict:
    """How well the AI layer agrees with OSM, per object and per parcel."""
    import shapely
    from shapely.geometry import box

    aoi = gpd.GeoSeries([box(*config.AOI_BBOX)], crs=config.DISPLAY_CRS).to_crs(config.METRIC_CRS).iloc[0]
    obj = detect.evaluate(b, detect.reference_buildings(h.osm), aoi.buffer(-15))
    osm_cov = h.parcels.geometry.intersection(shapely.union_all(list(h.osm.geometry))).area
    det_cov = h.parcels.geometry.intersection(shapely.union_all(list(b.geometry))).area
    built, found = osm_cov > config.MIN_PRESENCE_SQM, det_cov > config.MIN_PRESENCE_SQM
    return {"object": obj,
            "parcel": {"built_found": int((built & found).sum()), "built_total": int(built.sum()),
                       "vacant_false": int((~built & found).sum()), "vacant_total": int((~built).sum())}}


def check_against_truth(issues: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    truth = json.loads((config.RAW_DIR / "ground_truth_discrepancies.json").read_text())
    found = set(zip(issues.parcel_id, issues.issue))
    hit = [t for t in truth if (t["parcel_id"], t["type"]) in found]
    miss = [t for t in truth if (t["parcel_id"], t["type"]) not in found]
    return hit, miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ai", "osm"], default="ai")
    ap.add_argument("--redetect", action="store_true", help="ignore cached AI detections")
    ap.add_argument("--check", action="store_true", help="fail if any injected discrepancy is missed")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    res = run(a.mode, a.redetect)
    print("\n".join("  · " + line for line in res.log))
    if res.metrics:
        print(f"\nDetection vs OSM: {res.metrics}")
    print(f"\n{len(res.issues)} possible discrepancies across {res.issues.parcel_id.nunique()} parcels:")
    print(res.issues.groupby(["severity", "issue"]).size().to_string())

    hit, miss = check_against_truth(res.issues)
    print(f"\nInjected discrepancies recovered: {len(hit)}/{len(hit) + len(miss)}")
    for m in miss:
        print(f"  MISSED  {m['parcel_id']:>6}  {m['type']}")
    if a.check and miss:
        sys.exit(1)


if __name__ == "__main__":
    main()
