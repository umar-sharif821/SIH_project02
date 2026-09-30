"""Each discrepancy rule fires on a hand-built case and stays quiet on a clean one."""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import discrepancy, harmonize, match  # noqa: E402

CRS = config.METRIC_CRS
X0, Y0 = 500000, 3400000  # somewhere valid in UTM 43N


def sq(x, y, w, h):
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + h)


def scenario(*, buildings, recorded_area=None, muni=None, util=None):
    """Two 20×20 m parcels side by side: P101 (left) and P102 (right)."""
    parcels = gpd.GeoDataFrame({
        "parcel_id": ["P101", "P102"], "survey_no": ["22/101", "22/102"], "land_use": ["Built-up"] * 2,
        "recorded_area_sqm": recorded_area or [400.0, 400.0], "computed_area_sqm": [400.0, 400.0],
    }, geometry=[sq(0, 0, 20, 20), sq(20, 0, 20, 20)], crs=CRS)
    muni = muni if muni is not None else pd.DataFrame({
        "parcel_id": ["P101", "P102"], "municipal_pid": ["MC-022-0101", "MC-022-0102"],
        "has_building": [True, True], "recorded_builtup_sqm": [100.0, 100.0], "property_type": ["Residential"] * 2})
    util = util if util is not None else pd.DataFrame({
        "parcel_id": ["P101", "P102"], "utility_ref": ["S22-P101", "S22-P102"],
        "water_conn": [True, True], "electricity_conn": [True, True]})
    b = gpd.GeoDataFrame({"building_id": [f"D{i}" for i in range(len(buildings))],
                          "confidence": 0.9, "area_sqm": [g.area for g in buildings], "source": "test"},
                         geometry=buildings, crs=CRS)
    h = harmonize.Harmonized(parcels, muni, util, b.iloc[0:0])
    b, stats = match.match_buildings(b, parcels)
    view = harmonize.unified_view(h, stats)
    return discrepancy.find_discrepancies(view, b, muni, util)


CLEAN = [sq(5, 5, 10, 10), sq(25, 5, 10, 10)]  # one 100 m² building inside each parcel


def types(issues, pid=None):
    return set(issues.issue if pid is None else issues[issues.parcel_id == pid].issue)


def test_clean_scenario_has_no_issues():
    assert scenario(buildings=CLEAN).empty


def test_unrecorded_building():
    muni = pd.DataFrame({"parcel_id": ["P101", "P102"], "municipal_pid": ["a", "b"],
                         "has_building": [False, True], "recorded_builtup_sqm": [0.0, 100.0],
                         "property_type": ["Plot", "Residential"]})
    assert types(scenario(buildings=CLEAN, muni=muni)) == {"Unrecorded building"}


def test_recorded_building_not_found():
    # utilities agree with the municipal record here, so only the building rule fires
    assert types(scenario(buildings=[CLEAN[1]]), "P101") == {"Recorded building not found"}


def test_boundary_encroachment():
    straddling = [sq(13, 5, 10, 10), sq(28, 5, 10, 10)]  # 30% of the first building crosses into P102
    issues = scenario(buildings=straddling)
    enc = issues[issues.issue == "Boundary encroachment"]
    assert list(enc.parcel_id) == ["P101"]
    assert "P102" in enc.evidence.iloc[0]


def test_structure_spilling_onto_road():
    spill = [sq(-5, 5, 15, 10), CLEAN[1]]  # 150 m², a third of it west of P101 (no parcel there)
    issues = scenario(buildings=spill)
    enc = issues[issues.issue == "Boundary encroachment"]
    assert list(enc.parcel_id) == ["P101"] and "outside every parcel" in enc.evidence.iloc[0]


def test_merged_structure_is_not_called_encroachment():
    # one large polygon half on each parcel: attached houses or an AI merge, not an intrusion
    merged = [sq(2, 2, 36, 18)]
    assert types(scenario(buildings=merged)) == {"Structure spans multiple parcels"}


def test_parcel_area_mismatch():
    assert types(scenario(buildings=CLEAN, recorded_area=[400.0, 500.0])) == {"Parcel area mismatch"}


def test_builtup_area_mismatch():
    muni = pd.DataFrame({"parcel_id": ["P101", "P102"], "municipal_pid": ["a", "b"],
                         "has_building": [True, True], "recorded_builtup_sqm": [80.0, 100.0],
                         "property_type": ["Residential"] * 2})
    big = [sq(1, 1, 18, 18), CLEAN[1]]  # 324 m² detected vs 80 m² recorded (4x)
    assert types(scenario(buildings=big, muni=muni)) == {"Built-up area mismatch"}


def test_utility_on_vacant_parcel():
    muni = pd.DataFrame({"parcel_id": ["P101", "P102"], "municipal_pid": ["a", "b"],
                         "has_building": [True, False], "recorded_builtup_sqm": [100.0, 0.0],
                         "property_type": ["Residential", "Plot"]})
    assert types(scenario(buildings=[CLEAN[0]], muni=muni)) == {"Utility on vacant parcel"}


def test_missing_and_orphan_records():
    muni = pd.DataFrame({"parcel_id": ["P101", "P999"], "municipal_pid": ["a", "MC-022-0999"],
                         "has_building": [True, True], "recorded_builtup_sqm": [100.0, 100.0],
                         "property_type": ["Residential"] * 2})
    issues = scenario(buildings=CLEAN, muni=muni)
    assert types(issues, "P102") == {"No municipal record"}
    assert types(issues, "P999") == {"Orphan record"}


@pytest.mark.parametrize("raw,expected", [("Yes", True), (" y ", True), ("1", True), ("No", False), ("N", False), ("0", False)])
def test_parse_bool_variants(raw, expected):
    assert harmonize.parse_bool(pd.Series([raw])).iloc[0] == expected


def test_parse_bool_rejects_garbage():
    with pytest.raises(ValueError):
        harmonize.parse_bool(pd.Series(["maybe"]))
