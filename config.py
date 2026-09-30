"""Central configuration for the SIH26013 prototype."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

# Area of interest (west, south, east, north) in EPSG:4326.
# Default: a residential block in north Chandigarh — detached houses, well mapped in OSM.
AOI_NAME = "Chandigarh — residential block (demo AOI)"
AOI_BBOX = (76.7879, 30.7602, 76.7927, 30.7641)

DISPLAY_CRS = "EPSG:4326"   # for maps and GeoJSON exchange
METRIC_CRS = "EPSG:32643"   # UTM zone 43N — used for all area/distance calculations

# OpenStreetMap download — tried in order (public Overpass servers are often busy)
OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.private.coffee/api",
]

# Imagery download
IMAGERY_ZOOM = 19           # ~0.26 m/px at this latitude
IMAGERY_SOURCE = "Esri.WorldImagery"

# AI building detection
HF_MODEL_REPO = "keremberke/yolov8m-building-segmentation"
HF_MODEL_FILE = "best.pt"
DETECT_TILE_PX = 640
DETECT_OVERLAP_PX = 128
DETECT_CONF = 0.10         # low threshold: recall matters more than per-object precision here
DETECT_SCALE = 1.0          # resample imagery before inference (0.5 ≈ one zoom level coarser)
DETECT_MIN_AREA_SQM = 15.0
RECHECK_CROP_PX = 320       # "second look": crop around a parcel, enlarged 2x for the model
RECHECK_CONF = 0.25

# Discrepancy rule thresholds
ENCROACHMENT_MIN_OVERLAP = 0.80     # structures less than 80% inside one parcel are reviewed
MAX_SINGLE_STRUCTURE_SQM = 600.0    # larger AI polygons are usually several merged houses
AREA_MISMATCH_TOL = 0.10            # recorded vs computed parcel area

# Thresholds that depend on where building footprints come from. AI roof masks read
# ~1.3x the recorded footprint (overhangs, merged yards) and bleed across close
# boundaries; surveyed/OSM footprints do not.
RULE_PROFILES = {
    "ai":  {"encroach_max_overlap": 0.75,   # called encroachment when <= this share is inside
            "builtup_ratio_high": 3.5},     # detected / recorded above this → under-declared built-up
    "osm": {"encroach_max_overlap": 0.80,
            "builtup_ratio_high": 2.0},
}
MIN_PRESENCE_SQM = 15.0             # detected roof area needed to call a parcel "built"
MIN_BUILDING_SHARE_SQM = 10.0       # a detection's share of a parcel counts only above this area...
MIN_PARCEL_SHARE = 0.10             # ...and above this fraction of the detection
MAX_OUTSIDE_RATIO = 0.25            # detection mostly on one parcel but spilling onto road/open land
MIN_SPILL_STRUCTURE_SQM = 60.0      # ignore small objects (sheds, vehicles) for the spill rule
MAX_COREG_SHIFT_M = 10.0            # search radius for vector→imagery co-registration
MIN_COREG_GAIN = 0.01               # apply the shift only if roof overlap improves by >= 1%

SEED = 26013
