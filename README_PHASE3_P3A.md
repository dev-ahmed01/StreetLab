# StreetLab Phase 3 P3A — Observation Platform

Phase 3 separates observation extraction from traffic-behaviour simulation.

StreetLab does not need to own every detector or tracker. It accepts reliable
observations from multiple extraction tools, normalizes them into one schema,
preserves provenance, builds site-calibration targets, and feeds those targets
into the existing Decision Lab.

## Product boundary

P3A answers: What did the source data observe at this site?

Phase 1 still answers: How do StreetLab's calibrated Indian vehicle classes
behave in simulation?

P3A must not silently retrain or overwrite Phase-1 personas.

Currently calibrated behaviour families:
- MOTORCYCLE — CALIBRATED
- CAR — CALIBRATED
- AUTO_RICKSHAW — CALIBRATED

Other normalized classes such as BUS, HEAVY_VEHICLE and LIGHT_COMMERCIAL can
enter the observation package, but their simulation behaviour remains ASSUMED
until separately calibrated.

## Provider architecture

Raw aerial video -> external Geo-trax -> GeoTraxAdapter
FLUID trajectories -----------------> StreetLab Observation Package
SimJam fixed-camera exports --------> StreetLab Observation Package
Custom CSV/JSON + explicit mapping -> StreetLab Observation Package

Optional signal and route files normalize into the same package.

### Geo-trax
Geo-trax is treated as an external provider, not vendored into StreetLab.
StreetLab builds extraction plans for pixel-coordinate extraction and full
georeferenced extraction. Full georeferenced outputs normalize through
GeoTraxAdapter.

### FLUID
FLUID is the benchmark schema because it already contains world-coordinate
trajectories, vehicle classes, speeds, accelerations, turn movements, signals,
routes and travel times. FLUID traffic behaviour is not adopted as Indian
behaviour.

### SimJam
SimJam is treated as an external fixed-camera export source. P3A joins its
vehicle_tracks_xy.csv with the companion vehicle summary CSV. Its average
vehicle speed is preserved as track-average metadata, not misrepresented as an
instantaneous point speed.

### OpenTrafficCam and other tools
GPL or externally licensed tools are not copied into StreetLab core. Their
exports can enter through generic mappings when a stable schema is available.

## Provider-neutral schema

Each trajectory point can include:
- source provider and source track ID
- original source class
- canonical StreetLab class
- CALIBRATED or ASSUMED behaviour support
- evidence provenance
- time and world X/Y
- speed and acceleration
- movement and entry/exit direction
- frame and confidence
- lane and road section

Signal records normalize direction, turn, state, begin/end/duration and cycle.

Route records normalize vehicle, class, entry/exit times, travel time,
movement, turn and signal interaction where available.

## Indian class normalization

moped / scooter / motorcycle / 2W -> MOTORCYCLE -> CALIBRATED
car / taxi -> CAR -> CALIBRATED
auto / auto-rickshaw / 3W -> AUTO_RICKSHAW -> CALIBRATED
LCV / LTV / van -> LIGHT_COMMERCIAL -> ASSUMED
truck -> HEAVY_VEHICLE -> ASSUMED
bus -> BUS -> ASSUMED
unknown -> OTHER -> ASSUMED

The source label is always retained.

## Arbitrary new datasets

Unknown CSV/JSON files can enter through explicit mappings for trajectory,
signal and route columns. StreetLab refuses unknown schemas rather than
guessing column meaning.

## File pipeline

Auto-detected FLUID/Geo-trax example:

python scripts/phase3_calibrate_observation.py --tracks data/tracks.csv --signals data/signals.csv --routes data/routes.csv --output artifacts/phase3_observation_package.json

For custom schemas, use GenericTrajectoryMapping, GenericSignalMapping and
GenericRouteMapping.

## Raw aerial video planning

Without georeferencing assets, StreetLab plans:

geotrax batch junction.mp4 --no-geo

This is not calibration-ready because world-coordinate site calibration is
incomplete.

With orthophotos, segmentations and master frames, StreetLab plans the full
Geo-trax georeferenced pipeline. The resulting CSV can then be ingested by P3A.

## Quality diagnostics

Every package reports total points, missing speeds, negative speeds, extreme
speeds, unknown-class tracks, and whether persona calibration was modified.
The final field must remain false in P3A.

## Scientific guardrail

P3A performs site calibration and observation normalization. It does not claim
a new Indian behaviour calibration, that foreign behaviour represents India,
or that every raw video is calibration-ready without spatial reference data.

Later Phase 3 combines site observations + real network + real demand + signal
plan + Phase-1 personas -> SUMO baseline -> fidelity gate -> Decision Lab.
