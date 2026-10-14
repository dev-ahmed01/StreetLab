"""Phase 3 -> Phase 2 M1 bridge: source-pixel tracks, not fabricated site truth.

Imports genuine 14-column primary tracking exports (including the frozen W04
native IoS .30 baseline). Creates a local, immutable, hashed observation
receipt and a sanitized read-only dashboard view. The only observed evidence
eligible for the Phase-2 gate is uncalibrated TRAJECTORIES. All real-site
simulation inputs remain explicitly NEEDS_DATA.

This is a separate integration module. It does NOT edit or import a holdout
policy lock, consume reviewer labels, compute false vehicle counts, or promote
new tracking classes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from streetlab_phase2.models import DecisionType
from streetlab_phase2.study import (
    EvidenceItem, EvidenceProvenance, ObservationPackage, StudyContract,
    evaluate_study,
)

STATUS = "STREETLAB_INTEGRATION_M1_PIXEL_OBSERVATION"
CLASSES = {0: "CAR", 1: "BUS", 2: "HEAVY_VEHICLE", 3: "MOTORCYCLE"}
NATIVE_COLUMNS = 14
PREVIEW_LIMIT = 160
MAX_SOURCE_BYTES = 1024 * 1024 * 1024  # explicit bounded local import
REPORT_KEYS = (
    "schema_version", "status", "source_provider", "source_coordinate_system",
    "source_tracking_sha256", "observed_points", "observed_frames",
    "frame_span", "tracking_identity_count", "tracks_by_class",
    "class_flips_on_track_ids", "mean_detector_confidence",
    "observed_duration_seconds", "fps", "data_readiness",
    "study_gate", "preview_points", "limitations",
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for buf in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def _valid_int(value: float, what: str) -> int:
    if not math.isfinite(value) or not value.is_integer() or value < 0:
        raise ValueError("Invalid native tracking " + what)
    return int(value)


def load_native_tracks(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Strict, stable original-frame + original-ID parser; no made-up IDs."""
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_SOURCE_BYTES:
        raise ValueError("Missing, empty or oversized native tracking input")
    source_sha = sha(path)
    points = []
    seen = set()
    prev_frame = -1
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or len(row) != NATIVE_COLUMNS:
                raise ValueError(f"Line {line_number}: expected exactly 14 numeric fields")
            try:
                values = [float(x) for x in row]
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: non-numeric native track") from exc
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"Line {line_number}: nonfinite source evidence")
            frame = _valid_int(values[0], "frame")
            track_id = _valid_int(values[1], "tracker ID")
            cls = _valid_int(values[10], "class ID")
            if cls not in CLASSES or frame < prev_frame:
                raise ValueError("Unsorted source frames or unknown frozen class index")
            if (frame, track_id) in seen:
                raise ValueError("Duplicate real tracker ID in one source frame")
            seen.add((frame, track_id))
            prev_frame = frame
            x, y, w, h = values[2:6]
            conf = values[11]
            if (x < 0 or y < 0 or w <= 0 or h <= 0
                or not 0 <= conf <= 1
                or x-w/2 < -2 or y-h/2 < -2):
                raise ValueError(f"Line {line_number}: invalid pixel geometry or confidence")
            points.append({
                "frame": frame, "source_track_id": track_id,
                "vehicle_class": CLASSES[cls],
                "center_x_px": round(x, 4),
                "center_y_px": round(y, 4),
                "width_px": round(w, 4), "height_px": round(h, 4),
                "detector_confidence": round(conf, 6),
            })
    if not points:
        raise ValueError("Cannot publish empty tracking as observed traffic")
    return points, source_sha


def _gate() -> dict[str, Any]:
    """Reuse *actual* Phase 2 M6 sufficiency contract, not duplicated logic."""
    contract = StudyContract(
        decision_type=DecisionType.APPLY_DETOUR,
        area="real_site_not_georeferenced",
        baseline="observed_uncalibrated_pixel_tracks",
        scenario_family="real_site_calibration_pending",
        requested_metrics=(
            "max_approach_queue_vehicles", "mean_network_speed_mps",
            "rerouted_vehicles",
        ),
        decision_question="Can a real observed junction support a detour simulation?",
        unsupported_claims=(
            "No observed road network, route demand, movement turn counts, "
            "speed in m/s, or alternate route comes from these pixels.",
        ),
    )
    return evaluate_study(
        contract,
        ObservationPackage(items=(EvidenceItem(
            "trajectories", EvidenceProvenance.OBSERVED_AUTO,
            "Image pixel centers from native 14-column ByteTrack output; "
            "not physically calibrated world trajectories.",
        ),)),
    )


def build_report(points: list[dict], track_sha: str, *,
                 fps: float | None, label: str) -> dict[str, Any]:
    if not points or len(track_sha) != 64 or any(
        char not in "0123456789abcdef" for char in track_sha
    ):
        raise ValueError("An SHA-verified nonempty source is required")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("Source FPS must be positive and finite")
    if not label or len(label) > 120 or any(c in "\r\n\x00" for c in label):
        raise ValueError("A short non-sensitive source label is required")
    frames = {p["frame"] for p in points}
    track_counts = defaultdict(Counter)
    confidence = 0.
    for p in points:
        track_counts[p["source_track_id"]][p["vehicle_class"]] += 1
        confidence += p["detector_confidence"]
    classes = Counter()
    flips = 0
    for counts in track_counts.values():
        if len(counts) > 1:
            flips += 1
        maximum = max(counts.values())
        winner = min(k for k, value in counts.items() if value == maximum)
        classes[winner] += 1
    earliest, latest = min(frames), max(frames)
    gate = _gate()
    if gate["status"] != "NEEDS_DATA":
        raise RuntimeError("Pixel-only observations must never unlock SUMO simulation")
    selected = [
        points[index]
        for index in sorted({
            round(j * (len(points) - 1) / max(1, min(len(points), PREVIEW_LIMIT) - 1))
            for j in range(min(len(points), PREVIEW_LIMIT))
        })
    ]
    return {
        "schema_version": 1, "status": STATUS,
        "source_provider": label,
        "source_coordinate_system": "SOURCE_IMAGE_PIXELS_UNCALIBRATED",
        "source_tracking_sha256": track_sha,
        "observed_points": len(points),
        "observed_frames": len(frames),
        "frame_span": {"first": earliest, "last": latest,
                       "total_span_frames": latest - earliest + 1},
        "tracking_identity_count": len(track_counts),
        "tracks_by_class": dict(sorted(classes.items())),
        "class_flips_on_track_ids": flips,
        "mean_detector_confidence": round(confidence / len(points), 4),
        "observed_duration_seconds": (
            round((latest - earliest) / fps, 3) if fps is not None else None),
        "fps": fps,
        "data_readiness": {
            "pixel_trajectories": "OBSERVED_AUTO",
            "world_trajectories": "NEEDS_CALIBRATION",
            "real_site_geometry": "MISSING",
            "real_site_demand": "MISSING",
            "turn_movements": "MISSING",
            "speed_mps": "MISSING",
            "alternate_route": "MISSING",
            "simulation_source": "SEPARATE_SYNTHETIC_DEMO_ONLY",
            "real_site_simulation_allowed": False,
            "vehicle_count_is_physical_ground_truth": False,
            "persona_calibration_modified": False,
        },
        "study_gate": gate,
        "preview_points": selected,
        "limitations": [
            "Distinct tracker IDs are not guaranteed distinct physical vehicles.",
            "Pixel centers do not provide meters, m/s, road lane geometry, turns or demand.",
            "Demo SUMO network is synthetic and not reconstructed from this source.",
            "Class support and object counts remain detector-dependent research observations.",
        ],
    }


def ingest_track_file(path: Path, workdir: Path, *,
                      fps: float | None = None,
                      source_label: str = "native_bytetrack") -> tuple[Path, dict]:
    """Write immutable per-SHA receipt + atomically update local latest index."""
    path, workdir = Path(path), Path(workdir)
    points, fingerprint = load_native_tracks(path)
    report = build_report(points, fingerprint, fps=fps, label=source_label)
    destination = workdir / "observations" / fingerprint
    if destination.exists():
        raise FileExistsError("Source already imported. Receipts must remain immutable.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".receipt-", dir=destination.parent))
    try:
        data = stage / "track_points_pixels.csv"
        with data.open("x", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
        (stage / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        with (stage / "SHA256SUMS.txt").open("x", encoding="utf-8") as out:
            for f in sorted(stage.iterdir()):
                if f.name != "SHA256SUMS.txt":
                    out.write(sha(f) + "  " + f.name + "\n")
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    # Fixed shape: no arbitrary user-controlled path in the app reader.
    index_path = workdir / "observations" / "latest.json"
    temporary = index_path.with_name(".latest.new")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump({"schema_version": 1, "track_sha256": fingerprint}, out)
    os.replace(temporary, index_path)
    return destination, report


def latest_report(workdir: Path) -> dict[str, Any] | None:
    """Read only SHA-addressed local receipt; never trust a path from JSON."""
    folder = Path(workdir) / "observations"
    index = folder / "latest.json"
    if not index.is_file():
        return None
    pointer = json.loads(index.read_text(encoding="utf-8"))
    fingerprint = pointer.get("track_sha256")
    if (pointer.get("schema_version") != 1 or not isinstance(fingerprint, str)
        or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("Corrupt local observation index")
    receipt = folder / fingerprint
    file = receipt / "report.json"
    checks = receipt / "SHA256SUMS.txt"
    expected = {}
    for line in checks.read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", 1)
        if name not in {"report.json", "track_points_pixels.csv"}:
            raise ValueError("Untrusted observation receipt member")
        expected[name] = value
    if set(expected) != {"report.json", "track_points_pixels.csv"}:
        raise ValueError("Incomplete observation receipt")
    for name, digest in expected.items():
        if sha(receipt / name) != digest:
            raise ValueError("Original observation receipt integrity failure")
    report = json.loads(file.read_text(encoding="utf-8"))
    if report.get("source_tracking_sha256") != fingerprint or report.get("status") != STATUS:
        raise ValueError("Mismatched or untrusted observation report")
    if report.get("data_readiness", {}).get("real_site_simulation_allowed") is not False:
        raise ValueError("Source-only observations cannot authorize site simulation")
    return {key: report[key] for key in REPORT_KEYS}
