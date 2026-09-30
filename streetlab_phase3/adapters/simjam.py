from __future__ import annotations

from collections.abc import Mapping, Sequence

from streetlab_phase2.study import EvidenceProvenance

from ..class_mapping import map_vehicle_class
from ..models import TrackPoint


class SimJamBundleAdapter:
    """Normalize SimJam fixed-camera exports without importing SimJam code.

    SimJam writes point trajectories in vehicle_tracks_xy.csv and class/average
    speed metadata in a companion vehicle summary CSV. StreetLab joins those
    exports at the file boundary and preserves the fact that the reported speed
    is a track-level average, not an instantaneous point speed.
    """

    name = "simjam"

    TRACK_REQUIRED = {"frame", "time_s", "vehicle_id", "x_m", "y_m"}
    SUMMARY_REQUIRED = {"vehicle_id", "label", "avg_speed_kmh"}

    def normalize_bundle(
        self,
        track_rows: Sequence[Mapping[str, object]],
        summary_rows: Sequence[Mapping[str, object]],
    ) -> list[TrackPoint]:
        if not track_rows:
            raise ValueError("SimJam track input is empty")
        if not summary_rows:
            raise ValueError("SimJam vehicle summary input is empty")
        if not self.TRACK_REQUIRED.issubset(track_rows[0].keys()):
            raise ValueError("Rows do not match the SimJam trajectory schema")
        if not self.SUMMARY_REQUIRED.issubset(summary_rows[0].keys()):
            raise ValueError("Rows do not match the SimJam summary schema")

        summary_by_id = {str(row["vehicle_id"]): row for row in summary_rows}
        result: list[TrackPoint] = []

        for row in track_rows:
            track_id = str(row["vehicle_id"])
            if track_id not in summary_by_id:
                raise ValueError(
                    f"Missing SimJam summary metadata for vehicle_id={track_id}"
                )
            summary = summary_by_id[track_id]
            source_class = str(summary.get("label", "unknown"))
            mapping = map_vehicle_class(source_class)
            avg_speed_kmh = summary.get("avg_speed_kmh")

            result.append(
                TrackPoint(
                    source_provider=self.name,
                    source_track_id=track_id,
                    source_class=source_class,
                    vehicle_class=mapping.canonical,
                    behavior_support=mapping.behavior_support,
                    provenance=EvidenceProvenance.OBSERVED_AUTO,
                    time_s=float(row["time_s"]),
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    speed_mps=(
                        float(avg_speed_kmh) / 3.6
                        if avg_speed_kmh not in (None, "")
                        else None
                    ),
                    frame_number=(
                        int(float(row["frame"]))
                        if row.get("frame") not in (None, "")
                        else None
                    ),
                    metadata={
                        "speed_kind": "track_average",
                        "img_x": (
                            float(row["img_x"])
                            if row.get("img_x") not in (None, "")
                            else None
                        ),
                        "img_y": (
                            float(row["img_y"])
                            if row.get("img_y") not in (None, "")
                            else None
                        ),
                        "summary_start_frame": summary.get("start_frame"),
                        "summary_end_frame": summary.get("end_frame"),
                        "summary_start_time_s": summary.get("start_time_s"),
                        "summary_end_time_s": summary.get("end_time_s"),
                    },
                )
            )
        return result
