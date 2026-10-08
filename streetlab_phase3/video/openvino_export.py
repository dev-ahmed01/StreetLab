"""Prepare a non-quantized OpenVINO mirror of the frozen YOLO .pt.

All exports are isolated, immutable and linked to exact source checkpoint SHA.
Optional Ultralytics/OpenVINO dependencies load only when export is requested.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from streetlab_phase3.video.engine_shootout import sha256_file


def hash_model_tree(model_dir: Path) -> str:
    """Stable SHA256 over export content and relative paths, excluding symlinks."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"OpenVINO model directory missing: {model_dir}")
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Empty OpenVINO export")
    h = hashlib.sha256()
    for p in files:
        if p.is_symlink():
            raise ValueError(f"Symlink not permitted inside export: {p}")
        name = p.relative_to(model_dir).as_posix().encode("utf-8")
        h.update(len(name).to_bytes(4, "big"))
        h.update(name)
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def validate_export_source(model_dir: Path, source_checkpoint: Path,
                           image_size: int) -> dict[str, Any]:
    """Fail closed if the candidate is not a recorded export of these weights."""
    if not source_checkpoint.is_file() or image_size < 320:
        raise ValueError("Valid existing source checkpoint and image size required")
    provenance = model_dir.parent / "streetlab_openvino_export.json"
    data = json.loads(provenance.read_text(encoding="utf-8"))
    if (data.get("status") != "EXPERIMENTAL_OPENVINO_EXPORT_NOT_PROMOTED"
        or data.get("source_sha256") != sha256_file(source_checkpoint)
        or data.get("image_size") != image_size
        or data.get("format") != "openvino"
        or data.get("int8") is not False
        or data.get("exported_model_dir") != model_dir.name):
        raise ValueError("OpenVINO provenance does not match the frozen checkpoint / input")
    if not any(model_dir.glob("*.xml")) or not any(model_dir.glob("*.bin")):
        raise ValueError("Incomplete OpenVINO XML/BIN export")
    digest = hash_model_tree(model_dir)
    if data.get("export_sha256") != digest:
        raise ValueError("OpenVINO model files were modified after export")
    return data


def _default_exporter(weights: Path, image_size: int) -> Path:
    from ultralytics import YOLO
    result = YOLO(str(weights)).export(
        format="openvino", imgsz=image_size, int8=False,
        batch=1, device="cpu", verbose=False)
    return Path(result)


def export_isolated_openvino(
    *, weights: Path, output_dir: Path, image_size: int = 640,
    exporter: Callable[[Path, int], Path] | None = None,
) -> dict[str, Any]:
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise ValueError("Expected existing .pt source checkpoint")
    if not isinstance(image_size, int) or image_size < 320 or image_size % 32:
        raise ValueError("Image size must be a multiple of 32 and >= 320")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-",
                                  dir=output_dir.parent))
    if exporter is None:
        exporter = _default_exporter
    try:
        staged_weights = stage / "checkpoint.pt"
        shutil.copy2(weights, staged_weights)
        if sha256_file(staged_weights) != sha256_file(weights):
            raise IOError("Checkpoint copy mismatch")
        exported = Path(exporter(staged_weights, image_size)).resolve()
        if not exported.is_dir() or exported.parent != stage.resolve():
            raise ValueError("OpenVINO converter wrote outside isolated export stage")
        if not any(exported.glob("*.xml")) or not any(exported.glob("*.bin")):
            raise ValueError("OpenVINO converter did not produce XML and BIN files")
        model_dir = stage / "checkpoint_openvino_model"
        if exported != model_dir:
            raise ValueError("Unexpected OpenVINO export directory name")
        staged_weights.unlink()
        data = {
            "status": "EXPERIMENTAL_OPENVINO_EXPORT_NOT_PROMOTED",
            "source_weights": str(weights),
            "source_sha256": sha256_file(weights),
            "format": "openvino",
            "image_size": image_size,
            "int8": False,
            "exported_model_dir": model_dir.name,
            "export_sha256": hash_model_tree(model_dir),
            "eligible_for_promotion": False,
            "note": (
                "Conversion changes numerical execution. No speed, class parity, "
                "detection recall, or tracking validation is inferred from export."
            ),
        }
        (stage / "streetlab_openvino_export.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"OpenVINO output path became occupied: {output_dir}")
        os.replace(stage, output_dir)
        return {
            **data, "export_root": str(output_dir),
            "model_path": str(output_dir / model_dir.name),
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
