"""Experimental W04 rare-vehicle-preserving global box suppression.

Frozen 25-policy results stay immutable. This pure geometry module does NOT
consult FLUID truth or modify detector classes. No production promotion.
"""
from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Sequence

from .integrated_box_lab import RawBox, MergedBox, merge_boxes, overlap

MODES = ("rare_iou05", "rare_center_gate", "rare_cross_tile_center_gate")
RARE = ("MOTORCYCLE", "HEAVY_VEHICLE")


def _distance_guard(a: RawBox, b: RawBox, ratio: float = .35) -> bool:
    min_extent = min(
        a.x2 - a.x1, a.y2 - a.y1, b.x2 - b.x1, b.y2 - b.y1
    )
    return hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) <= ratio * min_extent


def _guarded_hard_nms(raw: Sequence[RawBox], *, different_tile_required: bool) -> list[MergedBox]:
    remaining = sorted(raw, key=lambda p: (
        -p.confidence, p.x1, p.y1, p.x2, p.y2, p.vehicle_class, p.source_index
    ))
    out: list[MergedBox] = []
    while remaining:
        winner = remaining.pop(0)
        if winner.confidence < .15:
            break
        out.append(MergedBox(
            winner.frame, winner.x1, winner.y1, winner.x2, winner.y2,
            winner.confidence, winner.vehicle_class, (winner.source_index,)
        ))
        survivors = []
        for contender in remaining:
            assert winner.frame == contender.frame
            assert winner.vehicle_class == contender.vehicle_class
            suppress = (
                overlap(winner, contender, "ios") >= .30
                and _distance_guard(winner, contender)
                and (not different_tile_required or winner.tile_id != contender.tile_id)
            )
            if not suppress:
                survivors.append(contender)
        remaining = survivors
    return out


def merge_rare_preserving(
    raw: Sequence[RawBox], *, mode: str = "rare_iou05"
) -> list[MergedBox]:
    """Hybrid: frozen CAR/BUS and nonvehicle IoS0.30; selected rare-class rule.

    The nonrare behavior remains the original NMS rule. The rare IoU0.50 mode
    deliberately retains partly overlapping motorcycles/heavy vehicles which
    original IoS0.30 would suppress. Do not infer physical recall from counts.
    """
    if mode not in MODES:
        raise ValueError("Unsupported W04 rare-suppression policy")
    grouped: dict[tuple[int,str], list[RawBox]] = defaultdict(list)
    for item in raw:
        if not isinstance(item, RawBox):
            raise TypeError("Original raw XYXY boxes are required")
        grouped[(item.frame, item.vehicle_class)].append(item)
    results: list[MergedBox] = []
    for (frame, klass), group in sorted(grouped.items()):
        if klass not in RARE:
            output = merge_boxes(group, policy="hard_nms", threshold=.30,
                                 metric="ios", score_floor=.15)
        elif mode == "rare_iou05":
            output = merge_boxes(group, policy="hard_nms", threshold=.50,
                                 metric="iou", score_floor=.15)
        else:
            output = _guarded_hard_nms(
                group, different_tile_required=(mode == "rare_cross_tile_center_gate")
            )
        if any(x.frame != frame or x.vehicle_class != klass for x in output):
            raise ValueError("Global merge changed source frame/class")
        results.extend(output)
    return sorted(results, key=lambda p: (
        p.frame, p.vehicle_class, -p.confidence, p.x1, p.y1, p.x2, p.y2
    ))
