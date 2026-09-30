from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from statistics import mean, median
from typing import Any, Iterable

from .models import BranchSpec, ResponsePolicy


DEFAULT_METRICS = (
    "arrivals_after_decision",
    "max_approach_queue_vehicles",
    "mean_network_speed_mps",
    "mean_instant_waiting_time_s",
    "rerouted_vehicles",
    "heterogeneous_guided_responders",
    "heterogeneous_local_responders",
    "heterogeneous_guided_rerouted",
    "heterogeneous_local_rerouted",
)

DEFAULT_DELTAS = (
    "arrivals_delta",
    "max_queue_delta",
    "mean_speed_delta_mps",
    "mean_waiting_delta_s",
)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("values must not be empty")
    low = min(vals)
    high = max(vals)
    return {
        "min": low,
        "mean": mean(vals),
        "median": median(vals),
        "p10": _percentile(vals, 0.10),
        "p90": _percentile(vals, 0.90),
        "max": high,
        "spread": high - low,
    }


def _member_branch(spec: BranchSpec, *, seed: str) -> BranchSpec:
    decision = spec.decision
    if (
        decision is None
        or decision.response_policy != ResponsePolicy.HETEROGENEOUS_RESPONSE
    ):
        return spec

    metadata = deepcopy(decision.metadata)
    assumptions = metadata.setdefault("response_assumptions", {})
    if not isinstance(assumptions, dict):
        raise ValueError("response_assumptions metadata must be an object")
    assumptions["seed"] = seed
    assumptions["provenance"] = "ASSUMED"

    return replace(
        spec,
        decision=replace(decision, metadata=metadata),
    )


def _aggregate_key(
    members: list[dict[str, Any]],
    branch_name: str,
    container: str,
    keys: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    rows = [
        next(b for b in member["branches"] if b["name"] == branch_name)
        for member in members
    ]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            row[container][key]
            for row in rows
            if key in row.get(container, {})
        ]
        if values:
            out[key] = summarize_values(values)
    return out


class EnsembleRunner:
    """Run repeated counterfactual members and summarize outcome distributions.

    M4 deliberately varies only explicit scenario-assumption assignment seeds.
    It does not perturb Phase-1 calibrated microscopic parameters and does not
    convert the resulting range into a probability forecast.
    """

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def run(
        self,
        *,
        snapshot_id: str,
        branches: list[BranchSpec],
        horizon_s: float,
        members: int,
        base_seed: str = "streetlab-m4",
    ) -> dict[str, Any]:
        if members < 2:
            raise ValueError("members must be >= 2 for an ensemble")
        if not branches:
            raise ValueError("branches must not be empty")
        names = [b.name for b in branches]
        if len(names) != len(set(names)):
            raise ValueError("branch names must be unique")
        if horizon_s <= 0:
            raise ValueError("horizon_s must be > 0")
        if not base_seed:
            raise ValueError("base_seed must not be empty")

        member_reports: list[dict[str, Any]] = []
        snapshot_hashes: set[str] = set()

        for index in range(members):
            member_seed = f"{base_seed}-{index:03d}"
            member_branches = [
                _member_branch(spec, seed=member_seed)
                for spec in branches
            ]
            report = self.runtime.run_branches(
                snapshot_id=snapshot_id,
                branches=member_branches,
                horizon_s=horizon_s,
            )
            snapshot = report.get("snapshot", {})
            snapshot_hash = str(snapshot.get("sha256", ""))
            if snapshot_hash:
                snapshot_hashes.add(snapshot_hash)

            member_reports.append(
                {
                    "member_index": index,
                    "member_seed": member_seed,
                    "snapshot": snapshot,
                    "same_snapshot_for_all_branches": report.get(
                        "same_snapshot_for_all_branches", False
                    ),
                    "branches": report["branches"],
                }
            )

        same_snapshot = (
            len(snapshot_hashes) == 1
            and all(
                member["same_snapshot_for_all_branches"]
                for member in member_reports
            )
        )

        branch_summaries: dict[str, Any] = {}
        for name in names:
            branch_summaries[name] = {
                "metrics": _aggregate_key(
                    member_reports,
                    name,
                    "metrics",
                    DEFAULT_METRICS,
                ),
                "vs_baseline": _aggregate_key(
                    member_reports,
                    name,
                    "vs_baseline",
                    DEFAULT_DELTAS,
                ),
            }

        return {
            "ensemble": {
                "members": members,
                "base_seed": base_seed,
                "uncertainty_source": "ASSUMPTION_ASSIGNMENT",
                "provenance": "ASSUMED",
                "same_snapshot_for_all_members": same_snapshot,
                "interpretation": (
                    "Outcome spread reflects repeated plausible assignment of "
                    "explicit scenario assumptions. It is not a calibrated "
                    "probability forecast of the future."
                ),
            },
            "branch_summaries": branch_summaries,
            "member_runs": member_reports,
        }
