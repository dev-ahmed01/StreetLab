"""M3 guided source-pixel -> local metric-plane reconstruction.

User-supplied surveyed correspondences define a LOCAL metric plane, not GPS.
A separate check point is necessary to claim a checked scale; fitted anchors
alone cannot validate the transform. All observed movement counts are tracker-ID
hypotheses, not verified distinct physical vehicle counts.
"""
from __future__ import annotations

from collections import defaultdict, Counter
import csv
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from streetlab_integration.video_jobs import JobError, VideoStore, _uuid
from streetlab_integration.primary_runner import verified_run, sha
from streetlab_integration.worker import job_report


class ReconstructionError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _point(value: Any, width: int, height: int, *, metric: bool = False) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ReconstructionError("Each point must contain two finite coordinates")
    x, y = map(float, value)
    if metric:
        if abs(x) > 100_000 or abs(y) > 100_000:
            raise ReconstructionError("Metric local-plane coordinates exceed limit")
    elif not (0 <= x <= width and 0 <= y <= height):
        raise ReconstructionError("Calibration and zone points must be inside the actual source image")
    return x, y


def _triangle_extent(points: list[tuple[float, float]]) -> float:
    return max((abs((b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1])) / 2
                for a, b, c in combinations(points, 3)), default=0.0)


def _normalize(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords-center, axis=1).mean()
    if radius < 1e-9:
        raise ReconstructionError("Control points are coincident")
    s = math.sqrt(2) / radius
    matrix = np.array([[s, 0, -s*center[0]], [0, s, -s*center[1]], [0, 0, 1.]])
    homogeneous = np.c_[coords, np.ones(len(coords))]
    return (matrix @ homogeneous.T).T[:, :2], matrix


def _homography(anchors: list[dict]) -> np.ndarray:
    image = np.array([p["pixel"] for p in anchors], dtype=float)
    world = np.array([p["world_m"] for p in anchors], dtype=float)
    if (_triangle_extent(list(map(tuple, image))) < 1.
        or _triangle_extent(list(map(tuple, world))) < .01):
        raise ReconstructionError("Non-collinear, well-spaced control points are required")
    i, Ti = _normalize(image)
    w, Tw = _normalize(world)
    a = []
    for (u, v), (x, y) in zip(i, w):
        a += [[-u, -v, -1, 0, 0, 0, u*x, v*x, x],
              [0, 0, 0, -u, -v, -1, u*y, v*y, y]]
    a = np.asarray(a, dtype=float)
    if np.linalg.matrix_rank(a, tol=1e-9) < 8:
        raise ReconstructionError("Degenerate calibration layout")
    _, _, vh = np.linalg.svd(a, full_matrices=True)
    h = np.linalg.inv(Tw) @ vh[-1].reshape((3, 3)) @ Ti
    if not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-10:
        raise ReconstructionError("Unstable camera-to-ground transform")
    return h / h[2, 2]


def _transform(h: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    p = h @ np.array([*point, 1.0])
    if not np.all(np.isfinite(p)) or abs(p[2]) < 1e-9:
        raise ReconstructionError("Point projects beyond the calibrated plane")
    x, y = p[0]/p[2], p[1]/p[2]
    if max(abs(x), abs(y)) > 100_000:
        raise ReconstructionError("Unstable or unbounded world projection")
    return float(x), float(y)


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    return .5*abs(sum(x*y2-x2*y for (x, y), (x2, y2)
                        in zip(vertices, vertices[1:]+vertices[:1])))


def _orientation(a, b, c) -> float:
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _crosses(a, b, c, d) -> bool:
    return (_orientation(a, b, c)*_orientation(a, b, d) < -1e-8
            and _orientation(c, d, a)*_orientation(c, d, b) < -1e-8)


def _polygon(points: Any, width: int, height: int) -> list[list[float]]:
    if not isinstance(points, list) or not 3 <= len(points) <= 24:
        raise ReconstructionError("A zone polygon needs between 3 and 24 vertices")
    verts = [_point(v, width, height) for v in points]
    if len(set(verts)) != len(verts):
        raise ReconstructionError("Zone polygon cannot repeat vertices")
    if _polygon_area(verts) < 4:
        raise ReconstructionError("Zone polygon is too small or degenerate")
    n = len(verts)
    for i in range(n):
        for j in range(i+1, n):
            if (abs(i-j) <= 1 or {i, j} == {0, n-1}):
                continue
            if _crosses(verts[i], verts[(i+1)%n], verts[j], verts[(j+1)%n]):
                raise ReconstructionError("Zone polygon crosses itself")
    return [list(p) for p in verts]


def _inside(p: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = p
    inside = False
    for (a, b), (c, d) in zip(polygon, polygon[1:]+polygon[:1]):
        if (b > y) != (d > y) and x < (c-a)*(y-b)/(d-b) + a:
            inside = not inside
    return inside


def validate_model(raw: dict, *, width: int, height: int) -> tuple[dict, dict, np.ndarray | None]:
    if not isinstance(raw, dict):
        raise ReconstructionError("A structured reconstruction object is required")
    anchors = raw.get("anchors", [])
    checks = raw.get("checks", [])
    zones = raw.get("zones", [])
    movements = raw.get("movements", [])
    if (not isinstance(anchors, list) or not isinstance(checks, list)
        or not isinstance(zones, list) or not isinstance(movements, list)
        or not 0 <= len(anchors) <= 4 or len(checks) > 8 or len(zones) > 24
        or len(movements) > 48):
        raise ReconstructionError("Calibration or topology exceeds supported local limits")
    scale_basis = str(raw.get("scale_basis", "")).strip()
    if len(scale_basis) > 500:
        raise ReconstructionError("Scale-evidence note is too long")
    def controls(entries):
        result = []
        for item in entries:
            if not isinstance(item, dict) or item.get("provenance") != "OBSERVED_MANUAL":
                raise ReconstructionError("Metric correspondences need explicit OBSERVED_MANUAL provenance")
            result.append({"pixel": list(_point(item.get("pixel"), width, height)),
                           "world_m": list(_point(item.get("world_m"), width, height, metric=True)),
                           "provenance": "OBSERVED_MANUAL"})
        return result
    fixed = controls(anchors)
    heldout = controls(checks)
    seen = [tuple(p["pixel"]) for p in fixed+heldout]
    if len(set(seen)) != len(seen):
        raise ReconstructionError("Calibration/check points must not repeat image coordinates")
    if len(fixed) not in (0, 4):
        raise ReconstructionError("Use four independent ground-plane anchors or leave calibration empty")
    normalized_zones = []
    zone_ids = set()
    for zone in zones:
        if not isinstance(zone, dict) or zone.get("provenance") != "OBSERVED_MANUAL":
            raise ReconstructionError("Zones must be explicitly OBSERVED_MANUAL")
        id_ = zone.get("id")
        if (not isinstance(id_, str) or not 1 <= len(id_) <= 30
            or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in id_)
            or id_ in zone_ids or zone.get("role") not in {"APPROACH", "EXIT"}):
            raise ReconstructionError("Zones need unique lowercase IDs and an APPROACH/EXIT role")
        lane_count = zone.get("lane_count")
        if lane_count is not None and (type(lane_count) is not int or not 1 <= lane_count <= 12):
            raise ReconstructionError("Lane count must be an explicitly verified integer in 1–12")
        zone_ids.add(id_)
        normalized_zones.append({
            "id": id_, "role": zone["role"],
            "label": str(zone.get("label", id_))[:80],
            "polygon_pixels": _polygon(zone.get("polygon_pixels"), width, height),
            "lane_count": lane_count,
            "provenance": "OBSERVED_MANUAL",
        })
    by_id = {z["id"]: z for z in normalized_zones}
    explicit_movements = []
    pairs = set()
    for movement in movements:
        if not isinstance(movement, dict):
            raise ReconstructionError("Invalid movement definition")
        entry, exit_ = movement.get("from_zone"), movement.get("to_zone")
        if (entry not in by_id or exit_ not in by_id
            or by_id[entry]["role"] != "APPROACH" or by_id[exit_]["role"] != "EXIT"
            or (entry, exit_) in pairs
            or movement.get("provenance") != "OBSERVED_MANUAL"):
            raise ReconstructionError("Movements must join known approach/exit zones, uniquely and manually")
        pairs.add((entry, exit_))
        explicit_movements.append({"from_zone": entry, "to_zone": exit_,
                                   "provenance": "OBSERVED_MANUAL"})
    model = {"schema_version": 1, "coordinate_system": "LOCAL_GROUND_PLANE_METERS_NOT_GPS",
             "camera_assumption": "STATIC_PLANAR_GROUND_ONLY",
             "width": width, "height": height, "scale_basis": scale_basis,
             "anchors": fixed, "checks": heldout,
             "zones": normalized_zones, "movements": explicit_movements}
    missing = []
    h = None
    residuals = []
    if len(fixed) != 4:
        missing.append("four measured ground-plane anchor correspondences")
    else:
        h = _homography(fixed)
        if not scale_basis:
            missing.append("documented physical scale measurement/provenance")
        if not heldout:
            missing.append("at least one independently measured check point")
        for c in heldout:
            estimated = _transform(h, tuple(c["pixel"]))
            residuals.append(round(math.dist(estimated, c["world_m"]), 5))
        if residuals and max(residuals) > 1.0:
            missing.append("check-point error above 1 meter; recalibrate camera")
    if not any(z["role"] == "APPROACH" for z in normalized_zones):
        missing.append("source-image approach polygon")
    if not any(z["role"] == "EXIT" for z in normalized_zones):
        missing.append("source-image exit polygon")
    if not explicit_movements:
        missing.append("manually reviewed permitted movements")
    if any(z["lane_count"] is None for z in normalized_zones if z["role"] == "APPROACH"):
        missing.append("verified lane counts for each approach")
    metric_ready = bool(h is not None and scale_basis and heldout and max(residuals, default=0) <= 1.0)
    quality = {
        "status": "GUIDED_GEOMETRY_REVIEWED" if not missing else "NEEDS_DATA",
        "metric_transform_checked": metric_ready,
        "independent_checkpoint_error_m": residuals if metric_ready else residuals,
        "missing_evidence": missing,
        "provenance": {"controls": "OBSERVED_MANUAL", "geometry": "OBSERVED_MANUAL",
                       "model": "CALIBRATED" if metric_ready else "INFERRED_UNVERIFIED"},
        "uncertainty": [
            "Ground plane is local and manually supplied; this is not GPS georeferencing.",
            "The 1m check gate is a QA heuristic, not a certified accuracy guarantee.",
            "User-reported measurement correctness and site coverage are not externally validated.",
            "Height variation, camera motion and lens distortion can invalidate planar homography.",
            "Road centerlines, lane connectivity and signal plans require further field review.",
        ],
        "real_site_sumo_allowed": False,
        "simulation_gate": {"status": "NEEDS_DATA",
                            "missing_evidence": ["verified route demand", "confirmed physical counts",
                                                 "signal plan or explicitly reviewed assumptions",
                                                 "SUMO network fidelity and route connectivity"]},
    }
    return model, quality, h if metric_ready else None


def _movement_candidate(zones: list[dict], movements: list[dict],
                        points: list[tuple[int, float, float]]) -> dict:
    """Zone visitation pattern; ambiguous IDs are not converted to physical counts."""
    if len(points) < 2:
        return {"status": "INSUFFICIENT_TRACK", "movement": None}
    sequence = []
    for _, x, y in points:
        hits = [z["id"] for z in zones if _inside((x, y), z["polygon_pixels"])]
        if len(hits) > 1:
            return {"status": "AMBIGUOUS_OVERLAP", "movement": None}
        if hits and (not sequence or hits[0] != sequence[-1]):
            sequence.append(hits[0])
    by_id = {z["id"]: z for z in zones}
    entries = [p for p in sequence if by_id[p]["role"] == "APPROACH"]
    exits = [p for p in sequence if by_id[p]["role"] == "EXIT"]
    if not entries or not exits:
        return {"status": "INSUFFICIENT_ZONE_COVERAGE", "movement": None}
    pair = (entries[0], exits[-1])
    if (sequence.index(exits[-1]) <= sequence.index(entries[0]) or
        len(set(entries)) > 1 or len(set(exits)) > 1):
        return {"status": "AMBIGUOUS_SEQUENCE", "movement": None}
    if not any((m["from_zone"], m["to_zone"]) == pair for m in movements):
        return {"status": "UNREGISTERED_MOVEMENT", "movement": list(pair)}
    return {"status": "TRACK_ID_CANDIDATE_ONLY", "movement": list(pair)}


def reconstruct(store: VideoStore, project_id: str, job_id: str, raw: dict) -> dict:
    _uuid(project_id)
    job = store.job(job_id)
    if job["project_id"] != project_id or job["status"] != "SUCCEEDED":
        raise ReconstructionError("Choose a successful observation job from this project")
    src = store.source(job["source_id"])
    _ = job_report(store, job_id)  # Full independent M1 receipt verification.
    run = store.root / "runs" / job_id
    manifest = verified_run(run, src["sha256"])
    if manifest.get("status") != "STREETLAB_M2_OBSERVED_AUTO_PIXEL_ONLY":
        raise ReconstructionError("Untrusted native observation run")
    dims = src["metadata"]
    model, quality, h = validate_model(raw, width=dims["width"], height=dims["height"])
    model["source_video_sha256"] = src["sha256"]
    model["source_track_sha256"] = job["report_sha256"]
    model["source_job_id"] = job_id
    model["source_project_id"] = project_id
    model["source_model_sha256"] = manifest["model_tree_sha256"]
    revision = hashlib.sha256(_json_bytes(model)).hexdigest()
    folder = store.root / "projects" / project_id / "reconstructions"
    target = folder / revision
    if not target.exists():
        folder.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".spatial-", dir=folder))
        try:
            (stage / "site_model.json").write_bytes(_json_bytes(model))
            tracks = run / "primary_ios030.txt"
            trajectories: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
            count = 0
            if h is not None:
                with tracks.open(newline="", encoding="utf-8-sig") as source, (
                    stage / "source_to_world.csv").open("x", newline="", encoding="utf-8") as destination:
                    output = csv.writer(destination)
                    output.writerow(["source_frame","source_track_id","vehicle_class_id",
                                     "x_source_px","y_source_px","x_local_m","y_local_m",
                                     "provenance"])
                    previous_frame = -1
                    for row in csv.reader(source):
                        if len(row) != 14:
                            raise ReconstructionError("Invalid source-native observation row")
                        values = [float(v) for v in row]
                        if not all(math.isfinite(v) for v in values):
                            raise ReconstructionError("Nonfinite source-native track value")
                        frame, tid, cls = int(values[0]), int(values[1]), int(values[10])
                        if (values[0] != frame or values[1] != tid
                            or frame < previous_frame or cls not in (0,1,2,3)):
                            raise ReconstructionError("Native track sort/schema violation")
                        previous_frame = frame
                        x, y = values[2], values[3]
                        if 0 <= x <= dims["width"] and 0 <= y <= dims["height"]:
                            # A homography does not make observations physically ground-truth.
                            wx, wy = _transform(h, (x, y))
                            output.writerow([frame,tid,cls,round(x,3),round(y,3),
                                             round(wx,4),round(wy,4),"CALIBRATED"])
                            trajectories[tid].append((frame,x,y))
                            count += 1
            votes = Counter()
            samples = []
            for tid, pts in sorted(trajectories.items()):
                outcome = _movement_candidate(model["zones"], model["movements"], pts)
                votes[outcome["status"]] += 1
                if len(samples) < 160:
                    samples.append({"source_track_id":tid, **outcome})
            quality["mapped_observations"] = count
            quality["tracked_id_candidate_movements"] = dict(votes)
            quality["candidate_samples"] = samples
            quality["observed_vehicle_count_physical_ground_truth"] = False
            quality["traffic_demand_confirmed"] = False
            (stage / "quality.json").write_bytes(_json_bytes(quality))
            checks = {p.name:sha(p) for p in stage.iterdir()}
            (stage / "SHA256SUMS.json").write_bytes(_json_bytes(checks))
            os.replace(stage, target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    # Validate both pre-existing and newly-published immutable revisions.
    result = verified_reconstruction(store, project_id, revision)
    with store.connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS site_revisions("
                     "project_id TEXT NOT NULL,revision TEXT NOT NULL,job_id TEXT NOT NULL,"
                     "created REAL NOT NULL,PRIMARY KEY(project_id,revision))")
        conn.execute("CREATE TABLE IF NOT EXISTS site_latest("
                     "project_id TEXT PRIMARY KEY, revision TEXT NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO site_revisions VALUES(?,?,?,?)",
                     (project_id, revision, job_id, time.time()))
        conn.execute("INSERT INTO site_latest VALUES(?,?) ON CONFLICT(project_id)"
                     " DO UPDATE SET revision=excluded.revision", (project_id, revision))
    return result


def verified_reconstruction(store: VideoStore, project_id: str, revision: str) -> dict:
    _uuid(project_id)
    if len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
        raise ReconstructionError("Invalid revision digest")
    folder = store.root / "projects" / project_id / "reconstructions" / revision
    if folder.is_symlink():
        raise ReconstructionError("Untrusted reconstruction directory")
    checks = json.loads((folder / "SHA256SUMS.json").read_text(encoding="utf-8"))
    allowed = {"site_model.json", "quality.json", "source_to_world.csv"}
    if not set(checks) <= allowed or not {"site_model.json", "quality.json"} <= set(checks):
        raise ReconstructionError("Unknown reconstruction receipt contents")
    for name, digest in checks.items():
        member = folder / name
        if member.is_symlink() or sha(member) != digest:
            raise ReconstructionError("Reconstruction evidence integrity failed")
    model = json.loads((folder / "site_model.json").read_text(encoding="utf-8"))
    quality = json.loads((folder / "quality.json").read_text(encoding="utf-8"))
    if hashlib.sha256(_json_bytes(model)).hexdigest() != revision:
        raise ReconstructionError("Reconstruction content-addressed manifest changed")
    if model["source_project_id"] != project_id:
        raise ReconstructionError("Reconstruction project binding differs")
    bound_job = store.job(model["source_job_id"])
    if (bound_job["project_id"] != project_id
        or bound_job["report_sha256"] != model["source_track_sha256"]
        or store.source(bound_job["source_id"])["sha256"] != model["source_video_sha256"]):
        raise ReconstructionError("Reconstruction source provenance mismatch")
    if quality["metric_transform_checked"] != ("source_to_world.csv" in checks):
        raise ReconstructionError("Unverified metric artifact mismatch")
    return {"revision":revision, "model":model, "quality":quality}


def latest_reconstruction(store: VideoStore, project_id: str) -> dict:
    store.project(project_id)
    with store.connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS site_latest("
                     "project_id TEXT PRIMARY KEY,revision TEXT NOT NULL)")
        row = conn.execute("SELECT revision FROM site_latest WHERE project_id=?",
                           (project_id,)).fetchone()
    if row is None:
        return {"status":"NOT_CONFIGURED", "quality":{
            "status":"NEEDS_DATA","missing_evidence":["surveyed source-image calibration and site geometry"],
            "real_site_sumo_allowed":False}}
    return verified_reconstruction(store, project_id, row["revision"])
