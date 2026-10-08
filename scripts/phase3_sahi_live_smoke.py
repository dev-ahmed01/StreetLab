"""Optional live dependency smoke: real SAHI + real Ultralytics weights.

This verifies inference APIs on an upstream public sample image, NOT FLUID accuracy.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.engine_shootout import load_class_map
from streetlab_phase3.video.sahi_detection_audit import convert_sahi_predictions


def main() -> None:
    import cv2
    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    sample_path = Path(".sahi-live-smoke-sample.jpeg")
    image_url = "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg"
    urllib.request.urlretrieve(image_url, sample_path)
    bgr = cv2.imread(str(sample_path))
    if bgr is None or bgr.size == 0:
        raise RuntimeError("Official SAHI sample could not be decoded")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Upstream model and public sample only. No private footage/annotations.
    detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path="yolo11n.pt",
        confidence_threshold=0.15, device="cpu", image_size=640)
    ordinary = get_prediction(rgb, detector, verbose=0)
    tiled = get_sliced_prediction(
        rgb, detector, slice_height=640, slice_width=640,
        overlap_height_ratio=0.20, overlap_width_ratio=0.20,
        perform_standard_pred=False, postprocess_type="NMS",
        postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
        verbose=0)
    taxonomy = load_class_map(None)
    full = convert_sahi_predictions(ordinary.object_prediction_list, 0, taxonomy)
    slices = convert_sahi_predictions(tiled.object_prediction_list, 0, taxonomy)
    if not ordinary.object_prediction_list and not tiled.object_prediction_list:
        raise AssertionError("Real model returned no objects in both paths")
    report = {
        "source": image_url,
        "checkpoint": "yolo11n.pt - official Ultralytics COCO model",
        "smoke_only": True,
        "not_a_fluid_benchmark": True,
        "standard_raw_predictions": len(ordinary.object_prediction_list),
        "sliced_raw_predictions": len(tiled.object_prediction_list),
        "streetlab_supported_standard": len(full),
        "streetlab_supported_sliced": len(slices),
        "image_height": int(rgb.shape[0]),
        "image_width": int(rgb.shape[1]),
    }
    Path("sahi_live_smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
