from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np

from streetlab_phase1.sumo_stage_a import (
    StageAConfig,
    prepare_corridor,
    run_stage_a_once,
)


CASES = [
    {
        "name": "D0_CURRENT_FEASIBLE_REFERENCE",
        "multipliers": {1: 0.90, 2: 1.00, 6: 0.96},
        "min_gap_lat_m": 0.60,
        "lc_pushy": 0.00,
    },
    {
        "name": "D1_SPEED_TARGET_NEUTRAL_LATERAL",
        "multipliers": {1: 0.90, 2: 0.90, 6: 0.90},
        "min_gap_lat_m": 0.60,
        "lc_pushy": 0.00,
    },
    {
        "name": "D2_SPEED_TARGET_LAT_050_PUSHY_010",
        "multipliers": {1: 0.90, 2: 0.90, 6: 0.90},
        "min_gap_lat_m": 0.50,
        "lc_pushy": 0.10,
    },
    {
        "name": "D3_SPEED_TARGET_LAT_040_PUSHY_015",
        "multipliers": {1: 0.90, 2: 0.90, 6: 0.90},
        "min_gap_lat_m": 0.40,
        "lc_pushy": 0.15,
    },
    {
        "name": "D4_SPEED_TARGET_LAT_035_PUSHY_020",
        "multipliers": {1: 0.90, 2: 0.90, 6: 0.90},
        "min_gap_lat_m": 0.35,
        "lc_pushy": 0.20,
    },
]


def _strict_gate(result: dict, cfg: StageAConfig) -> bool:
    s = result["simulation"]
    p95 = s.get("p95_depart_delay_s")
    return bool(
        s.get("departed_fraction", 0.0) >= cfg.min_departed_fraction_gate
        and p95 is not None
        and p95 <= cfg.max_p95_depart_delay_gate_s
        and s.get("teleport_events", 0) == 0
        and s.get("collision_events", 0) == 0
    )


def _summary_row(result: dict, case: dict, cfg: StageAConfig) -> dict:
    speed_fit = result.get("speed_fit", [])
    med_abs = [
        abs(float(r["median_delta_mps"]))
        for r in speed_fit
        if r.get("status") == "OK" and r.get("median_delta_mps") is not None
    ]
    return {
        "case": case["name"],
        "multipliers": case["multipliers"],
        "min_gap_lat_m": case["min_gap_lat_m"],
        "lc_pushy": case["lc_pushy"],
        "objective": result.get("objective"),
        "mean_abs_median_speed_error_mps": float(np.mean(med_abs)) if med_abs else None,
        "strict_gate_pass": _strict_gate(result, cfg),
        "departed_fraction": result["simulation"].get("departed_fraction"),
        "mean_depart_delay_s": result["simulation"].get("mean_depart_delay_s"),
        "p95_depart_delay_s": result["simulation"].get("p95_depart_delay_s"),
        "teleport_events": result["simulation"].get("teleport_events"),
        "collision_events": result["simulation"].get("collision_events"),
        "speed_fit": speed_fit,
    }


def run_coupling_diagnostic(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root)

    rows = []
    full_results = []

    for case in CASES:
        cfg = StageAConfig()
        # Keep the C3 longitudinal capacity baseline fixed.
        cfg.default_tau_s = 0.72
        cfg.feeder_length_m = 400.0
        # Only lateral permissiveness changes across D1-D4.
        cfg.default_min_gap_lat_m = float(case["min_gap_lat_m"])
        cfg.default_lc_pushy = float(case["lc_pushy"])

        ctx = prepare_corridor(project_root, cfg)
        print(
            f"Running {case['name']} "
            f"(speed={case['multipliers']}, "
            f"minGapLat={cfg.default_min_gap_lat_m}, "
            f"lcPushy={cfg.default_lc_pushy})..."
        )
        result = run_stage_a_once(
            ctx,
            cfg,
            {int(k): float(v) for k, v in case["multipliers"].items()},
            case["name"].lower(),
        )
        strict = _strict_gate(result, cfg)
        print(
            f"  objective={result['objective']:.4f}, "
            f"departed={result['simulation']['departed_fraction']:.3f}, "
            f"p95_delay={result['simulation']['p95_depart_delay_s']}, "
            f"collisions={result['simulation']['collision_events']}, "
            f"strict_gate={strict}"
        )
        rows.append(_summary_row(result, case, cfg))
        full_results.append({
            "case": case,
            "config": asdict(cfg),
            "result": result,
            "strict_gate_pass": strict,
        })

    passing = [r for r in rows if r["strict_gate_pass"]]
    best_passing = None
    if passing:
        best_passing = min(
            passing,
            key=lambda r: (
                float("inf") if r["objective"] is None else r["objective"],
                float("inf")
                if r["mean_abs_median_speed_error_mps"] is None
                else r["mean_abs_median_speed_error_mps"],
            ),
        )

    report = {
        "sprint": "1E_A3_SPEED_CAPACITY_COUPLING_DIAGNOSTIC",
        "purpose": (
            "Test whether the speed values suggested by the real data can be "
            "carried without artificial departure queues once lateral mixed-traffic "
            "capacity is made incrementally more permissive."
        ),
        "fixed_longitudinal_baseline": {
            "tau_s": 0.72,
            "meaning": (
                "Capacity-enabling baseline only. This is still NOT final spacing calibration."
            ),
        },
        "candidate_speed_multipliers": {
            "motorcycle": 0.90,
            "car": 0.90,
            "auto_rickshaw": 0.90,
            "meaning": (
                "Diagnostic target based on Stage-A history; not frozen final calibration."
            ),
        },
        "strict_gate": {
            "departed_fraction_min": 0.95,
            "p95_depart_delay_s_max": 5.0,
            "teleports_required": 0,
            "collisions_required": 0,
        },
        "results": rows,
        "best_strictly_passing_case": best_passing,
        "interpretation": [
            "If D1 passes, speed and capacity can coexist without extra lateral permissiveness.",
            "If D1 fails but D2/D3 passes, Stage A proved speed and lateral capacity are coupled.",
            "If only D4 passes, treat it as a warning: do not freeze parameters until lateral distributions are calibrated.",
            "If none pass, stop and revisit geometry/demand or longitudinal assumptions before further tuning.",
        ],
        "next_step_if_coupling_confirmed": (
            "Replace sequential A->B->C freezing with a constrained joint calibration "
            "of speedFactor + tau + minGapLat/lcPushy against speed, headway/front-gap "
            "and lateral-clearance targets, while keeping validation untouched."
        ),
        "full_results": full_results,
    }

    out = project_root / "data" / "processed" / "sprint1e_a3_coupling_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
