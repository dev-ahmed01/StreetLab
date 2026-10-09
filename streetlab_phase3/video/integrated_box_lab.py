"""Isolated pre-tracker box merger research: NO FLUID labels in decision rules.

The input must contain actual premerge, per-tile xyxy boxes. Cached centers do
NOT qualify. The module is deterministic and contains no inference runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

CANONICAL = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
EXTRAS = ("PEDESTRIAN", "BICYCLE")
ALL_CLASSES = CANONICAL + EXTRAS


@dataclass(frozen=True)
class RawBox:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    vehicle_class: str
    tile_id: int
    source_index: int

    def __post_init__(self) -> None:
        if (type(self.frame) is not int or self.frame < 0
            or type(self.tile_id) is not int or self.tile_id < 0
            or type(self.source_index) is not int or self.source_index < 0
            or self.vehicle_class not in ALL_CLASSES
            or not all(math.isfinite(z) for z in
                       (self.x1, self.y1, self.x2, self.y2, self.confidence))
            or not 0 <= self.x1 < self.x2
            or not 0 <= self.y1 < self.y2
            or not 0 <= self.confidence <= 1):
            raise ValueError(f"Invalid original raw box: {self}")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * .5, (self.y1 + self.y2) * .5)


@dataclass(frozen=True)
class MergedBox:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    vehicle_class: str
    contributors: tuple[int, ...]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * .5, (self.y1 + self.y2) * .5)


def overlap(a: RawBox | MergedBox, b: RawBox | MergedBox, metric: str = "iou") -> float:
    if metric not in ("iou", "ios"):
        raise ValueError("metric must be iou or ios")
    area = max(0., min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0., min(a.y2, b.y2) - max(a.y1, b.y1))
    a_area = (a.x2-a.x1)*(a.y2-a.y1)
    b_area = (b.x2-b.x1)*(b.y2-b.y1)
    den = ((a_area + b_area - area) if metric == "iou" else min(a_area,b_area))
    return area / den if den > 0 else 0.


def _merged(p: RawBox, score: float | None = None) -> MergedBox:
    return MergedBox(p.frame,p.x1,p.y1,p.x2,p.y2,
                     p.confidence if score is None else score,
                     p.vehicle_class,(p.source_index,))


def _sort(boxes: Sequence[RawBox | MergedBox]) -> list[Any]:
    return sorted(boxes, key=lambda p: (-p.confidence, p.x1,p.y1,p.x2,p.y2,
                                       p.vehicle_class, getattr(p,'source_index', -1)))


def merge_boxes(raw: Sequence[RawBox], *, policy: str,
                threshold: float = .5, metric: str = "iou",
                score_floor: float = .15, sigma: float = .5,
                class_agnostic: bool = False) -> list[MergedBox]:
    """Purely geometric/score-driven; never receives truth or matching labels.

    hard_nms: greedy IoU/IoS elimination.
    soft_linear/soft_gaussian: update remaining box scores and rerank per step.
    weighted_fusion: deterministic score-weighted coordinate clusters, one
      output per cluster; scores use max original confidence (not sum), so
      confidences cannot inflate beyond the best contributing prediction.

    All modes operate PER FRAME. Cross-class merging is never allowed for
    fusion because averaging class labels is undefined; class_agnostic=True
    is only an exploratory NMS/Soft-NMS variant, not default.
    """
    if policy not in ("none","hard_nms","soft_linear","soft_gaussian","weighted_fusion"):
        raise ValueError(f"Unknown box merge policy: {policy}")
    if not 0 < threshold <= 1 or not 0 <= score_floor <= 1 or sigma <= 0:
        raise ValueError("Invalid merge parameters")
    if metric not in ("iou", "ios"):
        raise ValueError("Invalid overlap metric")
    if policy == "weighted_fusion" and class_agnostic:
        raise ValueError("Class-agnostic weighted fusion is undefined; reject it")
    if len(set((p.frame,p.source_index) for p in raw)) != len(raw):
        raise ValueError("Repeated raw source index inside a frame")
    groups: dict[tuple[int,str], list[RawBox]] = {}
    for p in raw:
        groups.setdefault((p.frame, "ALL" if class_agnostic else p.vehicle_class), []).append(p)
    output = []
    for key in sorted(groups):
        remaining = _sort(groups[key])
        if policy == "none":
            output.extend(_merged(p) for p in remaining if p.confidence >= score_floor)
        elif policy == "hard_nms":
            while remaining:
                best = remaining.pop(0)
                if best.confidence < score_floor:
                    break
                output.append(_merged(best))
                remaining = [p for p in remaining if overlap(best,p,metric) < threshold]
        elif policy.startswith("soft_"):
            pairs: list[tuple[RawBox,float]] = [(p,p.confidence) for p in remaining]
            while pairs:
                pairs.sort(key=lambda z: (-z[1],z[0].x1,z[0].y1,z[0].source_index))
                best, conf = pairs.pop(0)
                if conf < score_floor:
                    break
                output.append(_merged(best,conf))
                updated = []
                for p,score in pairs:
                    o = overlap(best,p,metric)
                    if policy == "soft_linear":
                        changed = score * (1.-o if o >= threshold else 1.)
                    else:
                        changed = score * math.exp(-(o*o)/sigma)
                    if changed >= score_floor:
                        updated.append((p,changed))
                pairs = updated
        else:
            # Greedy WBF clustering against the up-to-date weighted
            # representative, deterministic with sorted score-first inputs.
            clusters: list[list[RawBox]] = []
            for p in remaining:
                if p.confidence < score_floor:
                    continue
                candidates=[]
                for i, members in enumerate(clusters):
                    rep = _fused(members)
                    o = overlap(p,rep,metric)
                    if o >= threshold:
                        candidates.append((o,i))
                if candidates:
                    idx = max(candidates,key=lambda z:(z[0],-z[1]))[1]
                    clusters[idx].append(p)
                else:
                    clusters.append([p])
            output.extend(_fused(members) for members in clusters)
    return sorted(output,key=lambda p:(p.frame,p.vehicle_class,-p.confidence,p.x1,p.y1))


def _fused(members: Sequence[RawBox]) -> MergedBox:
    if not members or len({p.frame for p in members}) != 1 or len({p.vehicle_class for p in members}) != 1:
        raise ValueError("Cannot fuse across frames or classes")
    weight = sum(max(p.confidence,1e-9) for p in members)
    def avg(name: str) -> float:
        return sum(getattr(p,name)*max(p.confidence,1e-9) for p in members)/weight
    return MergedBox(members[0].frame,avg("x1"),avg("y1"),avg("x2"),avg("y2"),
                     max(p.confidence for p in members),members[0].vehicle_class,
                     tuple(sorted(p.source_index for p in members)))


def candidate_matrix() -> list[dict[str, Any]]:
    """Prespecified, reproducible comparison; W04 tuned, NOT test-set selection."""
    configs=[dict(name="raw_unmerged",policy="none",threshold=.5,metric="iou",
                  score_floor=.15,sigma=.5,class_agnostic=False)]
    for policy in ("hard_nms", "soft_linear", "soft_gaussian", "weighted_fusion"):
        for metric in ("iou", "ios"):
            for threshold in (.30,.50,.70):
                configs.append(dict(name=f"{policy}_{metric}_{threshold:.2f}",
                                    policy=policy,threshold=threshold,metric=metric,
                                    score_floor=.15,sigma=.5,class_agnostic=False))
    # No cross-class NMS by default: real bus detections were mislabeled CAR
    # in May-26 FLUID, so cross-class suppression may delete real traffic.
    return configs


def box_dict(box: RawBox | MergedBox) -> dict[str, Any]:
    return asdict(box)

# Only the original W04 four-class detector scorer may be used for this
# numeric benchmark. OTHER / PEDESTRIAN / BICYCLE labels must remain visible
# in evidence, but cannot silently enter the historical denominator.
ALIASES = {
    "car": "CAR", "taxi": "CAR", "sedan": "CAR",
    "bus": "BUS", "truck": "HEAVY_VEHICLE", "heavy truck": "HEAVY_VEHICLE",
    "hgv": "HEAVY_VEHICLE", "motorcycle": "MOTORCYCLE",
    "motorbike": "MOTORCYCLE", "moped": "MOTORCYCLE", "scooter": "MOTORCYCLE",
    "bike": "MOTORCYCLE", "2w": "MOTORCYCLE",
}


def score_centers(
    predictions: Sequence[RawBox | MergedBox | dict[str, Any]],
    labels: Sequence[dict[str, Any]], frames: Sequence[int],
    *, frame_offset: int = 1, radius: float = 50.,
) -> dict[str, Any]:
    """Frozen class-aware 50px max-cardinality detector-only scorer.

    `labels` are original FLUID row dictionaries (frame/id/cx/cy/type).
    Returns counts and recall, not identity/physical FP judgments.
    """
    if frame_offset != 1 or radius != 50 or not frames or len(set(frames))!=len(frames):
        raise ValueError("Frozen +1 and 50px detector window required")
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    tf = set(frames)
    grouped_truth: dict[tuple[int,str],list[tuple[float,float]]] = {}
    raw_omitted = 0
    for row in labels:
        src = int(float(row['frame'])) - frame_offset
        if src not in tf:
            continue
        klass = ALIASES.get(str(row['type']).strip().lower())
        if klass is None:
            continue
        if row.get('cx') in ('',None) or row.get('cy') in ('',None):
            raw_omitted += 1
            continue
        x,y = float(row['cx']),float(row['cy'])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Invalid FLUID center")
        grouped_truth.setdefault((src,klass),[]).append((x,y))
    grouped_preds: dict[tuple[int,str],list[tuple[float,float]]] = {}
    for p in predictions:
        if isinstance(p,dict):
            frame = int(p['frame']); klass=p['vehicle_class']
            xy=(float(p['x_px']),float(p['y_px']))
        else:
            frame=p.frame;klass=p.vehicle_class;xy=p.center
        if frame not in tf:
            raise ValueError("Out-of-sample prediction")
        if klass not in ALL_CLASSES:
            raise ValueError("Unrecognized detection class")
        if klass in CANONICAL:
            grouped_preds.setdefault((frame,klass),[]).append(xy)
    per_class = {}
    for klass in CANONICAL:
        nt=npred=nmatch=0
        for frame in frames:
            true_xy = grouped_truth.get((frame,klass),[])
            pred_xy = grouped_preds.get((frame,klass),[])
            n,m=len(pred_xy),len(true_xy)
            nt+=m;npred+=n
            if not n or not m:
                continue
            a=np.asarray(pred_xy);b=np.asarray(true_xy)
            dist=np.sqrt(np.sum((a[:,None,:]-b[None,:,:])**2,axis=2))
            penalty=radius+1
            impossible=(n+m+1)*penalty*4
            cost=np.full((n+m,n+m),impossible)
            cost[:n,:m]=np.where(dist<=radius,dist,impossible)
            cost[:n,m:]=penalty;cost[n:,:m]=penalty;cost[n:,m:]=0.
            rows,cols=linear_sum_assignment(cost)
            nmatch+=sum(int(i<n and j<m and dist[i,j]<=radius)
                        for i,j in zip(rows,cols))
        per_class[klass] = {
            "truth":nt,"predicted":npred,"matched":nmatch,
            "recall":nmatch/nt if nt else None,
            "precision":nmatch/npred if npred else None,
        }
    tr=sum(x['truth'] for x in per_class.values())
    pr=sum(x['predicted'] for x in per_class.values())
    ma=sum(x['matched'] for x in per_class.values())
    return {
        "truth_points":tr,"predicted_points":pr,"matched_points":ma,
        "recall":ma/tr if tr else None,
        "precision":ma/pr if pr else None,
        "per_class":per_class,"unprojected_supported_truth_rows":raw_omitted,
        "track_identity_available":False,
    }


def evaluate_candidates(raw: Sequence[RawBox], labels: Sequence[dict[str,Any]],
                        frames: Sequence[int],
                        configs: Sequence[dict[str,Any]] | None=None) -> dict[str,Any]:
    """Predeclare full matrix, compare each candidate, NEVER auto-promote."""
    configs=list(configs if configs is not None else candidate_matrix())
    if not configs or len(set(c['name'] for c in configs))!=len(configs):
        raise ValueError("Nonempty unique candidate matrix required")
    metrics=[]
    ids={(r.frame,r.source_index) for r in raw}
    for settings in configs:
        merged=merge_boxes(raw,**{k:v for k,v in settings.items() if k!='name'})
        score=score_centers(merged,labels,frames)
        surviving={(p.frame,i) for p in merged for i in p.contributors}
        dropped=ids-surviving
        raw_by_id={(p.frame,p.source_index):p for p in raw}
        removed_by_class={cls:sum(raw_by_id[k].vehicle_class==cls for k in dropped)
                          for cls in ALL_CLASSES}
        metrics.append({
            "name":settings['name'],"settings":settings,"frozen_label_score":score,
            "output_box_count_all_classes":len(merged),
            "merged_cluster_count":sum(len(b.contributors)>1 for b in merged),
            "dropped_original_raw_boxes_by_class":removed_by_class,
            "guardrails":{
                "physical_presence_reviewed":False,
                "long_run_tracking_validated":False,
                "unseen_video_validated":False,
                "eligible_for_production":False,
            },
        })
    # Do not rank exclusively by class-aware precision from the mismatched
    # W04 FLUID ontology. Sort only as a reproducible display, not 'best'.
    metrics.sort(key=lambda r:r['name'])
    return {
        "status":"EXPERIMENTAL_PRETRACK_BOX_MATRIX_W04_TUNING_ONLY",
        "eligible_for_promotion":False,
        "frames":list(frames),"raw_box_count":len(raw),
        "candidate_count":len(metrics),"candidates":metrics,
        "notes":(
            "New tiling + per-tile inference is an independent capture, NOT a "
            "byte-identical reproduction of prior SAHI post-NMS CSV output. "
            "W04 FLUID classes contain known mismatches; class-aware precision "
            "cannot prove physical false alarms. A candidate retaining matched "
            "FLUID points may still delete real buses and motorcycles. "
            "No candidate automatically qualifies for tracking or production. "
            "Full box capture is pre-global-merge, not pre-local-model NMS. "
            "No tracking identities or continuous-video throughput measured."
        ),
    }
