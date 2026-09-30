# SIH26013 — Multi-source land data harmonization (prototype)

Different departments hold different data about the same land. This prototype puts a cadastral map,
municipal property records, utility connections and aerial imagery onto one map. It extracts buildings
from the imagery with a pretrained AI model, links them to parcels, and flags inconsistencies for an
officer to verify.

> **Proof of concept.** Cadastral, municipal and utility records are **synthetic** (clearly labelled) and
> were generated for a real neighbourhood in Chandigarh. Imagery is Esri World Imagery; reference footprints
> are © OpenStreetMap contributors. Every flag is a lead for verification, not a legal finding.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows  (source .venv/bin/activate on Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/prepare_data.py             # download imagery + OSM, generate synthetic records (once)
python -m src.pipeline --check             # run the whole pipeline in the terminal
streamlit run app.py                       # dashboard at http://localhost:8501
```

The first AI run downloads the model weights (~50 MB) from Hugging Face. Results are cached in
`data/processed/`, so after that the dashboard works offline.

## Pipeline

```
sources ─► ingest ─► harmonize ─► AI extraction ─► co-register ─► match ─► rules ─► dashboard / report
```

| Stage | Module | What happens |
|---|---|---|
| Ingest | `src/ingest.py` | Load 5 sources as-is. Each one uses its own CRS, columns and ID format (`22/101`, `MC-022-0101`, `S22-P101`) |
| Harmonize | `src/harmonize.py` | Reproject to UTM 43N, normalize IDs to `P101`, clean messy yes/no values (`Y`, `yes`, `1`…) |
| AI extraction | `src/detect.py` | Pretrained YOLOv8-seg ([keremberke/yolov8m-building-segmentation](https://huggingface.co/keremberke/yolov8m-building-segmentation)) runs on overlapping image tiles. Masks are converted to map polygons and tile-edge duplicates merged. A **second look** re-examines, at 2× zoom, parcels that records say are built but where the first pass found nothing |
| Co-register | `src/harmonize.py` | FFT cross-correlation of vector footprints against AI roofs estimates the vector↔imagery offset (+1 m E, −2 m N here) and corrects it |
| Match | `src/match.py` | Area-based building↔parcel linkage: roof area per parcel, and each structure's share per parcel |
| Rules | `src/discrepancy.py` | 9 discrepancy types (below), with thresholds in `config.py` |
| Output | `app.py`, `src/report.py` | Streamlit + Folium dashboard; CSV / GeoJSON / PDF verification list |

### Discrepancy rules

| Flag | Severity | Sources compared |
|---|---|---|
| Unrecorded building | High | imagery-AI × municipal |
| Recorded building not found | High | municipal × imagery-AI |
| Boundary encroachment | High | imagery-AI × cadastral |
| Parcel area mismatch | Medium | cadastral attribute × cadastral geometry |
| Built-up area mismatch (under-declared) | Medium | municipal × imagery-AI |
| Utility on vacant parcel | Medium | utility × municipal × imagery-AI |
| Structure spans multiple parcels | Low | imagery-AI × cadastral |
| No municipal record / Orphan record | Low | cadastral × municipal / utility |

## Results on the demo area (120 parcels, 112 buildings)

`prepare_data.py` injects 16 known discrepancies (`data/raw/ground_truth_discrepancies.json`), and
`python -m src.pipeline --check` verifies that they are recovered.

| Building evidence | Injected issues recovered | Total flags |
|---|---|---|
| OSM reference footprints | 16 / 16 | 16 (no false flags) |
| AI extraction from imagery | 16 / 16 | 48 |

AI quality compared with OSM:
- **Parcel level** (what the rules use): the AI finds a building on 103 of 112 built parcels. It wrongly marks 1 of 8 vacant plots as built.
- **Object level**: precision is about 46% and recall about 45% at IoU ≥ 0.3. The model outlines roofs rather than footprints and merges attached houses. The rules account for this by using different thresholds for AI and OSM building sources (`RULE_PROFILES` in `config.py`).

The extra AI-mode flags are mostly model misses ("recorded building not found" on parcels that are
actually built) and merged polygons. That is why every flag goes to an officer instead of being
auto-corrected.

## Demo script (≈4 min)

1. **① Sources**: five datasets, five formats. The CRS, columns and parcel IDs all differ.
2. **② Harmonization**: the pipeline's steps: ID normalization, CRS alignment, co-registration, AI extraction, second look. Then the unified parcel table.
3. **③ Map**: imagery, parcels coloured by flag severity, and AI-extracted buildings in cyan. Zoom to `P155` to see a building crossing its recorded boundary.
4. **④ Discrepancies**: the verification list. Open the parcel dossier for `P113` (under-declared built-up area) and `P118` (records say a building exists; the imagery shows an empty plot). Download the PDF report.
5. **⑤ AI quality**: how far to trust the AI layer.
6. Switch the sidebar to **OSM reference** to show the fallback and the clean 16/16 result.

## Changing the area

Edit `AOI_BBOX` in `config.py`, delete `data/raw/` and `data/processed/`, then run `prepare_data.py`
again. Areas with detached houses and good OSM coverage work best.

## Tests

```bash
pytest tests/
```
