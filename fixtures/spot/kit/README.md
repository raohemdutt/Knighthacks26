# SPOT challenge kit (NextEra Energy, Knight Hacks IX)

This folder holds NextEra's **synthetic, fictional** challenge kit for the Knight Hacks IX
sponsor track "SPOT Management Dashboard", **as handed to the team** ("SPOT_KnightHacks_Challenge_Kit
Part One"), in the kit's own folder structure and byte for byte (the data files keep their Windows
line endings). Nothing here describes a real facility, robot, person or event. This `README.md` is
ours; everything in the numbered folders is the sponsor's.

| Folder | What it is |
| --- | --- |
| `00_START_HERE/` | the sponsor's text: `Challenge_Brief.md`, `START_HERE.md`, `Data_Assumptions.md` |
| `02_DATA/` | the six tables (`missions`, `actions`, `readings`, `anomalies`, `locations`, `assets`), `replay.ndjson`, and `validation_results.json` (the sponsor's own counts and headline metrics, used as a cross-check) |
| `03_DATA_DICTIONARY/` | `data_dictionary.csv`, the field definitions |
| `04_ASSETS/` | `fictional_floor_plan.png` (both floors with WP01 to WP15 pinned), `waypoint_reference_legend.png`, `facility_map_and_sample_assets_overview.png`, and six `sample_inspection_images/*.jpg`. Per the kit's `04_ASSETS/README.md` these images are "synthetic and fictional, created for the Knight Hacks challenge". |

**Left out on purpose:** `01_REFERENCE/Approved_Orbit_Screenshots/` (18 MB of product screenshots that are
not ours to redistribute) and `05_STARTER_CODE/` (three example scripts; the app never runs anything from
the kit, it only reads the files as data).

## Using another copy

Point `ATLAS_SPOT_KIT` at any folder that holds the kit and restart the gateway. Both layouts are read:
the numbered folders above, or all the files flat in one folder (images then in `assets/`). The loader
reports everything odd it finds (`GET /api/spot/validation`).

## Files next to this folder that are ours

- `../waypoint_pixels.json`: where each waypoint's pin sits on `fictional_floor_plan.png`, read off the image
  by eye, so live markers can be drawn over the plan. The kit gives no pixel positions, and
  `locations.csv` coordinates do not line up with the image (the fit error is reported by `GET /api/spot/locations`).
- `../floor_plan.jpg`: the same plan re-encoded as JPEG (`sips -s format jpeg -s formatOptions 72`, 0.3 MB
  instead of 1.3 MB) so the page loads fast. Served only while it is newer than the PNG; delete it and the PNG is served.

Without the images the app still works: the map falls back to a schematic drawn from `locations.csv`.
