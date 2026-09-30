from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionPlan:
    command: tuple[str, ...]
    georeferenced: bool
    calibration_ready: bool
    license_boundary: str
    expected_output: str


class GeoTraxVideoProvider:
    """External Geo-trax provider.

    StreetLab does not vendor Geo-trax. This object builds an explicit external
    extraction plan so the resulting CSV can later be normalized by
    GeoTraxAdapter.
    """

    def __init__(self, executable: str = "geotrax") -> None:
        self.executable = executable

    def plan(
        self,
        *,
        video: str,
        orthophotos: str | None = None,
        segmentations: str | None = None,
        master_frames: str | None = None,
    ) -> ExtractionPlan:
        provided = [orthophotos, segmentations, master_frames]
        if not any(provided):
            return ExtractionPlan(
                command=(self.executable, "batch", video, "--no-geo"),
                georeferenced=False,
                calibration_ready=False,
                license_boundary="EXTERNAL_PROVIDER",
                expected_output="pixel-coordinate trajectory export",
            )
        if not all(provided):
            raise ValueError(
                "Full Geo-trax site calibration requires orthophotos, "
                "segmentations, and master_frames together"
            )
        return ExtractionPlan(
            command=(
                self.executable,
                "batch",
                video,
                "-orf",
                str(orthophotos),
                "-osf",
                str(segmentations),
                "-mf",
                str(master_frames),
            ),
            georeferenced=True,
            calibration_ready=True,
            license_boundary="EXTERNAL_PROVIDER",
            expected_output="Geo-trax georeferenced trajectory CSV",
        )
