"""SIH26013 — multi-source land data harmonization dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import io

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import config
from src import detect, ingest, pipeline, report
from src.discrepancy import SEVERITY

st.set_page_config(page_title="SIH26013 · Land Data Harmonization", page_icon="🗺️", layout="wide")

SEV_COLOR = {"High": "#e5484d", "Medium": "#f5a524", "Low": "#f7d154"}
MODES = {"ai": "AI extraction from imagery (YOLOv8-seg)", "osm": "OSM reference footprints (no AI)"}


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner=False)
def run_pipeline(mode: str, nonce: int):
    return pipeline.run(mode)


@st.cache_resource(show_spinner=False)
def imagery_overlay(max_px: int = 2600):
    """Local GeoTIFF → PNG data URL + lat/lon bounds, so the map works offline."""
    import rasterio
    from PIL import Image

    with rasterio.open(ingest.path("imagery")) as ds:
        f = max(ds.width, ds.height) / max_px
        arr = ds.read([1, 2, 3], out_shape=(3, int(ds.height / f), int(ds.width / f)))
        b = gpd.GeoSeries(gpd.points_from_xy([ds.bounds.left, ds.bounds.right], [ds.bounds.bottom, ds.bounds.top]),
                          crs=ds.crs).to_crs(config.DISPLAY_CRS)
    buf = io.BytesIO()
    Image.fromarray(np.transpose(arr, (1, 2, 0))).save(buf, format="JPEG", quality=85)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return url, [[b.y[0], b.x[0]], [b.y[1], b.x[1]]]


def geojson(gdf: gpd.GeoDataFrame, cols: list[str]) -> str:
    g = gdf[cols + ["geometry"]].to_crs(config.DISPLAY_CRS).copy()
    for c in cols:
        g[c] = g[c].map(lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else
                        (round(v, 1) if isinstance(v, float) else v)).astype(str)
    return g.to_json(drop_id=True)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🗺️ SIH26013")
st.sidebar.caption("Multi-source geospatial harmonization with AI feature extraction")
mode = st.sidebar.radio("Building evidence", list(MODES), format_func=MODES.get, index=0,
                        help="AI mode extracts buildings from the orthoimagery. OSM mode uses pre-mapped footprints "
                             "— useful as a reference and as a fallback if the model is unavailable.")
if "nonce" not in st.session_state:
    st.session_state.nonce = 0
if mode == "ai" and st.sidebar.button("Re-run AI extraction", help="Ignore cached detections and run the model again"):
    bar = st.sidebar.progress(0.0, "Running YOLOv8-seg on image tiles…")
    try:
        b = detect.detect_buildings(ingest.path("imagery"),
                                    progress=lambda f: bar.progress(f, f"Tiles processed: {f:.0%}"))
        detect.save(b, "ai")
        for p in config.PROCESSED_DIR.glob("detected_buildings_ai_second*"):
            p.unlink()
        st.session_state.nonce += 1
    except Exception as e:
        st.sidebar.error(f"AI extraction failed: {e}")
    bar.empty()

with st.spinner("Harmonizing sources and checking for discrepancies…"):
    try:
        R = run_pipeline(mode, st.session_state.nonce)
    except FileNotFoundError as e:
        st.error(f"{e}")
        st.stop()
if R.mode != mode:
    st.sidebar.warning("AI extraction unavailable — showing OSM reference footprints instead.")

st.sidebar.divider()
st.sidebar.markdown("**Map layers**")
show = {k: st.sidebar.checkbox(k, v) for k, v in [("Imagery (ORI)", True), ("Cadastral parcels", True),
                                                   ("Detected buildings", True), ("OSM reference footprints", False)]}
sev_filter = st.sidebar.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
st.sidebar.divider()
st.sidebar.caption(f"**Area:** {config.AOI_NAME}\n\n"
                   "Proof of concept. Cadastral, municipal and utility records are **synthetic**; imagery © Esri, "
                   "reference buildings © OpenStreetMap contributors. Flags are leads for officer verification, "
                   "not legal findings.")

view, issues, blds = R.view, R.issues, R.buildings
shown = issues[issues.severity.isin(sev_filter)]

# ------------------------------------------------------------------ header
st.title("One harmonized view of land records")
st.caption("Cadastral map · municipal property records · utility connections · aerial imagery — "
           "aligned on one map, buildings extracted by AI, linked to parcels, inconsistencies flagged.")

k = st.columns(5)
k[0].metric("Sources integrated", len(ingest.SOURCES))
k[1].metric("Parcels harmonized", len(view))
k[2].metric("Buildings extracted" if R.mode == "ai" else "Reference buildings",
            int(blds.parcel_id.notna().sum()), help="Structures linked to a parcel")
k[3].metric("Possible discrepancies", len(issues), help=f"on {issues.parcel_id.nunique()} parcels")
k[4].metric("High severity", int((issues.severity == "High").sum()))

tabs = st.tabs(["① Sources", "② Harmonization", "③ Map", "④ Discrepancies", "⑤ AI quality"])

# ------------------------------------------------------------------ ① sources
with tabs[0]:
    st.subheader("Five sources, five formats")
    st.write("Each department's data arrives in its own coordinate system, column names and ID format. "
             "Nothing here lines up until it is harmonized.")
    st.dataframe(pd.DataFrame([ingest.describe(key, R.raw[key]) for key in ingest.SOURCES]).astype(str),
                 hide_index=True, width="stretch")
    c = st.columns(3)
    for col, key in zip(c, ["cadastral", "municipal", "utility"]):
        with col:
            src = ingest.SOURCES[key]
            st.markdown(f"**{src.label}** · `{src.id_column}`")
            df = R.raw[key]
            st.dataframe(df.drop(columns="geometry", errors="ignore").head(8), hide_index=True, width="stretch")

# ------------------------------------------------------------------ ② harmonization
with tabs[1]:
    st.subheader("What the pipeline did")
    for line in R.log:
        st.markdown(f"- {line}")
    st.subheader("Unified parcel table")
    st.caption("One row per parcel with every source side by side. Measurements are in metres (UTM 43N).")
    tbl = view.drop(columns="geometry").rename(columns={
        "survey_no": "Survey No (cadastral)", "recorded_area_sqm": "Recorded area m²",
        "computed_area_sqm": "Measured area m²", "municipal_pid": "Municipal PID",
        "has_building": "Municipal: building?", "recorded_builtup_sqm": "Municipal built-up m²",
        "utility_ref": "Utility ref", "water_conn": "Water", "electricity_conn": "Electricity",
        "detected_buildings": "AI: structures", "detected_builtup_sqm": "AI: roof area m²"})
    st.dataframe(tbl, hide_index=True, width="stretch", height=420)

# ------------------------------------------------------------------ ③ map
with tabs[2]:
    worst = (shown.assign(rank=shown.severity.map({"High": 0, "Medium": 1, "Low": 2}))
             .sort_values("rank").drop_duplicates("parcel_id").set_index("parcel_id").severity)
    issue_txt = shown.groupby("parcel_id").issue.agg(lambda s: "; ".join(sorted(set(s))))
    pv = view.assign(status=view.parcel_id.map(worst).fillna("OK"), issues=view.parcel_id.map(issue_txt).fillna("—"))

    focus = st.selectbox("Zoom to parcel", ["(whole area)"] + sorted(shown.parcel_id.unique()), key="focus")
    b4326 = view.to_crs(config.DISPLAY_CRS)
    if focus != "(whole area)" and focus in set(view.parcel_id):
        c = b4326[b4326.parcel_id == focus].geometry.iloc[0].centroid
        m = folium.Map(location=[c.y, c.x], zoom_start=20, max_zoom=22, tiles=None)
    else:
        x0, y0, x1, y1 = b4326.total_bounds
        m = folium.Map(location=[(y0 + y1) / 2, (x0 + x1) / 2], zoom_start=17, max_zoom=22, tiles=None)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Imagery © Esri", name="Satellite (online)", max_zoom=22, max_native_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map (online)", max_zoom=22, max_native_zoom=19).add_to(m)
    if show["Imagery (ORI)"]:
        url, bounds = imagery_overlay()
        folium.raster_layers.ImageOverlay(url, bounds, name="Orthoimagery (local GeoTIFF)", opacity=1).add_to(m)

    if show["Cadastral parcels"]:
        folium.GeoJson(
            geojson(pv, ["parcel_id", "survey_no", "municipal_pid", "has_building", "detected_builtup_sqm",
                         "recorded_builtup_sqm", "status", "issues"]),
            name="Cadastral parcels",
            style_function=lambda f: {
                "color": SEV_COLOR.get(f["properties"]["status"], "#ffffff"),
                "weight": 2.5 if f["properties"]["status"] != "OK" else 1,
                "fillColor": SEV_COLOR.get(f["properties"]["status"], "#ffffff"),
                "fillOpacity": 0.35 if f["properties"]["status"] != "OK" else 0.02},
            tooltip=folium.GeoJsonTooltip(
                ["parcel_id", "survey_no", "municipal_pid", "has_building", "recorded_builtup_sqm",
                 "detected_builtup_sqm", "issues"],
                aliases=["Parcel", "Survey No", "Municipal PID", "Recorded building?", "Recorded built-up m²",
                         "Detected roof m²", "Flags"], sticky=True),
        ).add_to(m)
    if show["Detected buildings"]:
        folium.GeoJson(
            geojson(blds, ["building_id", "source", "confidence", "area_sqm", "parcel_id", "overlap_ratio"]),
            name="Buildings (" + ("AI" if R.mode == "ai" else "reference") + ")",
            style_function=lambda f: {"color": "#22d3ee", "weight": 1.5, "fillOpacity": 0.1},
            tooltip=folium.GeoJsonTooltip(["building_id", "source", "confidence", "area_sqm", "parcel_id", "overlap_ratio"],
                                          aliases=["ID", "Source", "Confidence", "Area m²", "Parcel", "Share in parcel"]),
        ).add_to(m)
    if show["OSM reference footprints"]:
        folium.GeoJson(geojson(R.h.osm.assign(n=1), ["n"]), name="OSM reference",
                       style_function=lambda f: {"color": "#d946ef", "weight": 1, "dashArray": "4 3",
                                                 "fillOpacity": 0}).add_to(m)
    enc = shown[shown.issue.isin(["Boundary encroachment", "Structure spans multiple parcels"]) & shown.geometry.notna()]
    if len(enc):
        folium.GeoJson(geojson(enc, ["parcel_id", "issue", "severity", "evidence"]), name="Boundary-crossing structures",
                       style_function=lambda f: {"color": SEV_COLOR[f["properties"]["severity"]], "weight": 3,
                                                 "fillOpacity": 0.25, "dashArray": "6 4"},
                       tooltip=folium.GeoJsonTooltip(["parcel_id", "issue", "evidence"],
                                                     aliases=["Parcel", "Flag", "Evidence"])).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=640, use_container_width=True, returned_objects=[], key=f"map-{mode}-{focus}")
    st.caption("Parcel colour = most severe flag (red High · amber Medium · yellow Low). Cyan = extracted buildings. "
               "Hover a parcel for all sources' values.")

# ------------------------------------------------------------------ ④ discrepancies
with tabs[3]:
    st.subheader("Verification list")
    kinds = st.multiselect("Issue type", list(SEVERITY), default=[k for k in SEVERITY if k in set(shown.issue)])
    lst = shown[shown.issue.isin(kinds)]
    counts = lst.groupby(["severity", "issue"]).size().rename("count").reset_index()
    c1, c2 = st.columns([1, 2])
    c1.dataframe(counts.sort_values("severity", key=lambda s: s.map({"High": 0, "Medium": 1, "Low": 2})),
                 hide_index=True, width="stretch")
    c2.dataframe(lst[["parcel_id", "severity", "issue", "evidence", "sources_compared"]], hide_index=True,
                 width="stretch", height=360,
                 column_config={"parcel_id": st.column_config.TextColumn("Parcel", width="small"),
                                "severity": st.column_config.TextColumn("Severity", width="small"),
                                "issue": st.column_config.TextColumn("Issue", width="medium"),
                                "evidence": st.column_config.TextColumn("Evidence", width="large"),
                                "sources_compared": st.column_config.TextColumn("Sources compared")})

    d = st.columns(3)
    d[0].download_button("⬇ CSV", report.to_csv(lst), "discrepancies.csv", "text/csv")
    d[1].download_button("⬇ GeoJSON", report.to_geojson(lst), "discrepancies.geojson", "application/geo+json")
    d[2].download_button("⬇ PDF report", report.to_pdf(lst, view, MODES[R.mode], R.log),
                         "verification_report.pdf", "application/pdf")

    st.subheader("Parcel dossier")
    pid = st.selectbox("Parcel", sorted(view.parcel_id, key=lambda p: int(p[1:])),
                       index=None, placeholder="Pick a parcel to see every source's record")
    if pid:
        r = view[view.parcel_id == pid].iloc[0]
        cols = st.columns(4)
        cols[0].markdown(f"**Cadastral**  \nSurvey No `{r.survey_no}`  \nRecorded {r.recorded_area_sqm:.0f} m²  \n"
                         f"Measured {r.computed_area_sqm:.0f} m²")
        cols[1].markdown("**Municipal**  \n" + (f"PID `{r.municipal_pid}`  \nBuilding: {'yes' if r.has_building else 'no'}  \n"
                                                 f"Built-up {r.recorded_builtup_sqm:.0f} m²" if pd.notna(r.municipal_pid)
                                                 else "_no record_"))
        cols[2].markdown("**Utility**  \n" + (f"Ref `{r.utility_ref}`  \nWater: {'yes' if r.water_conn else 'no'}  \n"
                                               f"Electricity: {'yes' if r.electricity_conn else 'no'}"
                                               if pd.notna(r.utility_ref) else "_no record_"))
        cols[3].markdown(f"**Imagery ({'AI' if R.mode == 'ai' else 'reference'})**  \n"
                         f"Structures: {r.detected_buildings}  \nRoof area {r.detected_builtup_sqm:.0f} m²  \n"
                         f"{r.building_ids if isinstance(r.building_ids, str) else ''}")
        pi = issues[issues.parcel_id == pid]
        if pi.empty:
            st.success("All sources agree for this parcel.")
        for x in pi.itertuples():
            st.warning(f"**{x.issue}** ({x.severity}) — {x.evidence}  \n_{x.action}; compared: {x.sources_compared}_")

# ------------------------------------------------------------------ ⑤ AI quality
with tabs[4]:
    if R.mode != "ai" or not R.metrics:
        st.info("Switch to AI mode in the sidebar to see how the AI extraction compares with OSM footprints.")
    else:
        o, p = R.metrics["object"], R.metrics["parcel"]
        st.subheader("How far can the AI layer be trusted?")
        q = st.columns(4)
        q[0].metric("Built parcels where AI found a building", f"{p['built_found']}/{p['built_total']}",
                    f"{p['built_found'] / p['built_total']:.0%}", delta_color="off")
        q[1].metric("Vacant parcels wrongly marked built", f"{p['vacant_false']}/{p['vacant_total']}")
        q[2].metric("Object precision (IoU ≥ 0.3)", f"{o['precision']:.0%}")
        q[3].metric("Object recall (IoU ≥ 0.3)", f"{o['recall']:.0%}")
        st.markdown(
            f"""
- **Model:** `{config.HF_MODEL_REPO}` — pretrained YOLOv8 instance segmentation, used as-is (no training).
  Runs locally on CPU; no external AI API.
- **Parcel level is what the rules use**, and it is strong. Object-level scores are lower mainly because the
  model outlines roofs (larger than surveyed footprints) and merges attached houses into one polygon.
- The rules are calibrated for this: roof areas must disagree by more than
  {config.RULE_PROFILES['ai']['builtup_ratio_high']}× before a built-up mismatch is flagged, and large merged
  polygons are reported as *spans multiple parcels* (Low) rather than encroachment.
- A **second look** re-examines, at 2× zoom, parcels where records claim a building but the first pass saw none.
- Remaining “recorded building not found” flags on built parcels are model misses. That is why every flag
  goes to an officer instead of being auto-corrected.
""")
        st.caption("Reference: OpenStreetMap footprints inside the area of interest. OSM itself is incomplete "
                   "and 1–2 m offset from the imagery, so these scores are approximate.")
