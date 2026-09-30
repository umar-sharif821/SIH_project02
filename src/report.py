"""Exports: discrepancy CSV / GeoJSON and a PDF verification report for officers."""
from __future__ import annotations

from datetime import date

import geopandas as gpd

import config
from src.discrepancy import SEVERITY

_ASCII = {"—": "-", "–": "-", "→": "->", "×": "x", "’": "'", "“": '"', "”": '"', "…": "..."}


def _txt(s) -> str:
    s = str(s)
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def to_csv(issues: gpd.GeoDataFrame) -> bytes:
    return issues.drop(columns="geometry").to_csv(index=False).encode("utf-8")


def to_geojson(issues: gpd.GeoDataFrame) -> bytes:
    g = issues[issues.geometry.notna()]
    return g.to_crs(config.DISPLAY_CRS).to_json(drop_id=True).encode("utf-8")


def to_pdf(issues: gpd.GeoDataFrame, view: gpd.GeoDataFrame, mode_label: str, log: list[str]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, _txt("Land data discrepancy report - officer verification list"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _txt(
        f"Area: {config.AOI_NAME}   |   Generated: {date.today():%d %b %Y}   |   Building source: {mode_label}\n"
        "Proof-of-concept (SIH26013). Cadastral, municipal and utility records are SYNTHETIC. Every row below is a "
        "possible discrepancy for field / record verification - not a legal finding."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    counts = issues.groupby("issue").size()
    lines = [f"Parcels harmonized: {len(view)}   |   Possible discrepancies: {len(issues)} "
             f"on {issues.parcel_id.nunique()} parcels"]
    lines += [f"  {SEVERITY[k]:<6}  {k}: {counts.get(k, 0)}" for k in SEVERITY if counts.get(k, 0)]
    pdf.multi_cell(0, 4.5, _txt("\n".join(lines)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Harmonization steps", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 4, _txt("\n".join("- " + s for s in log)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verification list", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(col_widths=(16, 42, 16, 150, 49), text_align="LEFT", line_height=4.2) as t:
        t.row(["Parcel", "Issue", "Severity", "Evidence", "Sources compared"])
        for r in issues.itertuples():
            t.row([_txt(r.parcel_id), _txt(r.issue), r.severity, _txt(r.evidence), _txt(r.sources_compared)])
    return bytes(pdf.output())
