"""Deterministic GIS step: connect extracted buildings to the parcels they sit on.

Matching is area-based. Aerial AI often merges attached (row) houses into one polygon,
so rather than forcing each detection onto a single parcel we measure
  * per parcel:    how much detected roof area lies inside it
  * per detection: how its footprint is shared between parcels
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

import config


def match_buildings(buildings: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame
                    ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Returns (buildings with parcel shares, per-parcel statistics).

    Building columns added:
      parcel_id      parcel holding the largest share (None if it sits on no parcel)
      overlap_ratio  that largest share as a fraction of the footprint
      n_parcels      number of parcels holding a significant share
      also_touches   the other significant parcels
      outside_ratio  fraction of the footprint on no parcel (road / open land)
    """
    b = buildings.to_crs(config.METRIC_CRS).reset_index(drop=True)
    p = parcels.to_crs(config.METRIC_CRS)[["parcel_id", "geometry"]]
    b["_area"] = b.area

    ov = gpd.overlay(b[["building_id", "geometry"]], p, how="intersection", keep_geom_type=True)
    ov["share_sqm"] = ov.area
    ov = ov.merge(b[["building_id", "_area"]], on="building_id")
    significant = ov[(ov.share_sqm >= config.MIN_BUILDING_SHARE_SQM)
                     & (ov.share_sqm >= config.MIN_PARCEL_SHARE * ov._area)]
    significant = significant.sort_values("share_sqm", ascending=False)

    first = significant.drop_duplicates("building_id").set_index("building_id")
    groups = significant.groupby("building_id")
    b["parcel_id"] = b.building_id.map(first.parcel_id)
    b["overlap_ratio"] = (b.building_id.map(first.share_sqm) / b._area).fillna(0).clip(upper=1).round(3)
    b["n_parcels"] = b.building_id.map(groups.size()).fillna(0).astype(int)
    b["also_touches"] = b.building_id.map(groups.parcel_id.agg(lambda s: ", ".join(s.iloc[1:]))).fillna("")
    # A structure that crosses into a parcel which *also has its own separate structure*
    # is a likely intrusion; one covering two otherwise-empty-looking parcels is more
    # likely attached construction or two houses merged by the AI.
    parcel_total = ov.groupby("parcel_id").share_sqm.sum()
    minor = significant.drop(index=significant.drop_duplicates("building_id").index)
    own = parcel_total.reindex(minor.parcel_id).to_numpy() - minor.share_sqm.to_numpy()
    intruded = minor[own >= config.MIN_PRESENCE_SQM]
    b["intrudes_on"] = b.building_id.map(intruded.groupby("building_id").parcel_id.agg(", ".join)).fillna("")
    b["outside_ratio"] =(1 - b.building_id.map(ov.groupby("building_id").share_sqm.sum()).fillna(0) / b._area
                          ).clip(lower=0).round(3)
    b = b.drop(columns="_area")

    stats = (ov.groupby("parcel_id")
             .agg(detected_builtup_sqm=("share_sqm", "sum"),
                  building_ids=("building_id", lambda s: ", ".join(sorted(s))))
             .reset_index())
    counted = significant.groupby("parcel_id").size().rename("detected_buildings")
    stats = stats.merge(counted, on="parcel_id", how="left")
    stats["detected_buildings"] = stats.detected_buildings.fillna(0).astype(int)
    stats.loc[stats.detected_builtup_sqm < config.MIN_PRESENCE_SQM, "detected_buildings"] = 0
    return b, stats
