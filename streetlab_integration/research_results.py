"""Fail-closed, dependency-free results/figure exporter for StreetLab research SUMO.

Reads outputs of research_submission.py. It reports *conditional synthetic*
comparisons only; it does not measure real road effects or certify a digital twin.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

BRANCHES = ("BASELINE", "DETOUR_GUIDED", "DETOUR_MIXED")
METRICS = (
    ("max_approach_queue_vehicles", "Maximum approach queue", "vehicles", "figure_01_queue.svg"),
    ("arrivals_after_decision", "Arrivals after decision", "vehicles", "figure_02_arrivals.svg"),
    ("mean_network_speed_mps", "Mean network speed", "m/s", "figure_03_speed.svg"),
    ("mean_instant_waiting_time_s", "Mean instantaneous waiting time", "seconds", "figure_04_waiting.svg"),
    ("rerouted_vehicles", "Rerouted vehicles", "vehicles", "figure_05_reroutes.svg"),
)

class ResultsError(ValueError):
    """The source artifact does not support this research claim."""


def _required(ok: bool, message: str) -> None:
    if not ok:
        raise ResultsError(message)


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResultsError(f"Could not read valid JSON: {path}") from exc
    _required(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _number(value: object, path: str) -> float:
    _required(type(value) in (float, int) and math.isfinite(value), f"Invalid finite number at {path}")
    return float(value)


def _distribution(value: object, path: str) -> dict:
    _required(isinstance(value, dict), f"Missing metric distribution: {path}")
    result = {k: _number(value.get(k), f"{path}.{k}") for k in ("min", "median", "max")}
    _required(result["min"] - 1e-9 <= result["median"] <= result["max"] + 1e-9,
              f"Inconsistent distribution bounds: {path}")
    return result


def analyze(evidence: dict, study: dict) -> dict:
    """Check claim boundaries before returning values for presentation."""
    _required(evidence.get("track") == "RESEARCH_SUBMISSION_NO_NEW_FIELD_SURVEY", "Wrong research track")
    _required(evidence.get("research_workflow_status") ==
              "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION", "Unexecuted research workflow")
    for key in ("field_census_claimed", "external_site_transfer_validated",
                "real_world_intervention_effect_proven", "production_release_approved"):
        _required(evidence.get(key) is False, f"Unsupported evidence claim: {key}")
    _required(evidence.get("datasets_are_from_different_sites") is True, "Dataset separation not confirmed")
    datasets = evidence.get("dataset_summaries", {})
    chennai = datasets.get("chennai", {})
    fluid = datasets.get("fluid_w04", {})
    _required(chennai.get("holdout_generalization_pass") is False and
              chennai.get("capacity_gate_pass") is False and
              chennai.get("calibration_locked") is True and
              chennai.get("retuning_allowed") is False, "Frozen Chennai failure/gates not preserved")
    _required(fluid.get("scores_recomputed_in_this_run") is False and
              "development" in str(fluid.get("data_split", "")).lower(), "FLUID split incorrectly promoted")
    native = evidence.get("native_sumo", {})
    _required(native.get("status") == "SUMO_EXECUTED_SYNTHETIC_ONLY" and
              native.get("site_specific_field_validation") is False and
              native.get("production_release_approved") is False and
              native.get("simulation_not_real_world_effect") is True, "Native evidence cannot support field claims")
    _required(study.get("incomplete_study", {}).get("status") == "NEEDS_DATA" and
              study.get("supported_study", {}).get("status") == "SUPPORTED", "Evidence sufficiency guard failed")
    guard = study.get("guardrails", {})
    _required(guard.get("confidence_fabricated") is False and
              guard.get("live_runtime_mutated_by_evaluated_branch") is False and
              guard.get("authority_remains_final_decision_maker") is True, "Native guardrails failed")
    ensemble = study.get("ensemble", {})
    meta = ensemble.get("metadata", {})
    _required(meta.get("provenance") == "ASSUMED" and
              meta.get("uncertainty_source") == "ASSUMPTION_ASSIGNMENT" and
              meta.get("same_snapshot_for_all_members") is True and
              type(meta.get("members")) is int and meta["members"] >= 2, "Not a paired synthetic ensemble")
    _required(native.get("ensemble_members") == meta["members"], "Ensemble size mismatch")
    summaries = ensemble.get("branch_summaries", {})
    _required(isinstance(summaries, dict) and set(BRANCHES).issubset(summaries), "Missing branch summaries")
    _required(summaries == native.get("scorecard"), "Native result and evidence scorecards disagree")
    table = []
    for metric, label, unit, _ in METRICS:
        row = {"key": metric, "name": label, "unit": unit, "branches": {}}
        for branch in BRANCHES:
            dist = _distribution(summaries[branch].get("metrics", {}).get(metric), f"{branch}.{metric}")
            _required(dist["min"] >= 0, f"Negative physical metric: {branch}.{metric}")
            row["branches"][branch] = dist
        table.append(row)
    # Paired deltas come from the engine's per-member comparisons, NOT a subtraction
    # of two independent quantiles (those operations are generally different).
    delta_names = ("arrivals_delta", "max_queue_delta", "mean_speed_delta_mps", "mean_waiting_delta_s")
    paired = {}
    for branch in BRANCHES[1:]:
        paired[branch] = {}
        for delta in delta_names:
            paired[branch][delta] = _distribution(
                summaries[branch].get("vs_baseline", {}).get(delta), f"{branch}.vs_baseline.{delta}")
    baseline_arrivals = table[1]["branches"]["BASELINE"]["median"]
    weaker_arrivals = [b for b in BRANCHES[1:] if table[1]["branches"][b]["median"] < baseline_arrivals]
    return {"members": meta["members"], "metrics": table, "paired": paired,
            "reduced_arrival_branches": weaker_arrivals,
            "chennai_status": chennai.get("phase1_status"),
            "fluid_precision": _number(fluid.get("reported_point_precision_pct"), "FLUID precision"),
            "fluid_recall": _number(fluid.get("reported_point_recall_pct"), "FLUID recall")}


def _fmt(value: float, metric: str) -> str:
    return f"{value:.3f}" if metric in ("mean_network_speed_mps", "mean_instant_waiting_time_s") else f"{value:.0f}"


def report(data: dict, evidence_hash: str, native_hash: str) -> str:
    lines = [
        "# StreetLab — Native SUMO research results (synthetic junction)", "",
        "**Evidence classification:** Executed native SUMO, assumed junction geometry, demand, movements and available detour. **Not a measured traffic improvement.**",
        f"**Paired assignment ensemble:** {data['members']} members. Its min–max range describes modeled assumption sensitivity, **not a confidence interval**.",
        "**Source artifacts:** `submission_evidence.json` (SHA-256 `" + evidence_hash + "`), `native_sumo_study.json` (SHA-256 `" + native_hash + "`).", "",
        "## Actual branch values (ensemble medians)", "",
        "| Metric | BASELINE | DETOUR_GUIDED | DETOUR_MIXED |", "|---|---:|---:|---:|",
    ]
    for row in data["metrics"]:
        lines.append("| " + row["name"] + " (" + row["unit"] + ") | " + " | ".join(
            _fmt(row["branches"][branch]["median"], row["key"]) for branch in BRANCHES) + " |")
    lines += ["", "## Modeled scenario comparisons (paired median deltas versus baseline)", "",
              "| Paired outcome | DETOUR_GUIDED | DETOUR_MIXED |", "|---|---:|---:|"]
    for name, label, suffix in (
        ("arrivals_delta", "Arrivals", " vehicles"), ("max_queue_delta", "Max queue", " vehicles"),
        ("mean_speed_delta_mps", "Mean speed", " m/s"), ("mean_waiting_delta_s", "Mean instantaneous waiting", " s"),
    ):
        lines.append("| " + label + " | " + " | ".join(
            f"{data['paired'][branch][name]['median']:+.3f}{suffix}" for branch in BRANCHES[1:]) + " |")
    lines += ["", "## Min–max across assignment members", "",
              "| Metric | Scenario | Minimum | Median | Maximum |", "|---|---|---:|---:|---:|"]
    for row in data["metrics"]:
        for branch in BRANCHES:
            s = row["branches"][branch]
            lines.append(f"| {row['name']} ({row['unit']}) | {branch} | " + " | ".join(_fmt(s[k], row["key"]) for k in ("min", "median", "max")) + " |")
    lines += ["", "## Interpretation and limitations", "",
              "- Within the specified synthetic network, queue and waiting metrics are lower for the evaluated detour scenarios. This is a conditional simulation outcome, not a field-measured effect."]
    if data["reduced_arrival_branches"]:
        lines.append("- **Trade-off:** post-decision arrivals decreased under " + ", ".join(data["reduced_arrival_branches"]) + ". Queue reduction alone cannot establish greater throughput or overall user benefit; route/network-wide impacts need matched measurements.")
    lines += [
        "- An identical median and zero spread in some branches do not prove zero real-world uncertainty: only the stated assumption assignments were explored.",
        "- Waiting is a **mean instantaneous waiting-time measure**, not per-trip end-to-end delay. Do not interpret it as saved journey minutes.",
        "- The earlier Chennai held-out generalization **failed** and remains frozen (" + str(data["chennai_status"]) + ").",
        f"- W04 previously reported FLUID-label precision {data['fluid_precision']:.2f}% / recall {data['fluid_recall']:.2f}% are **development-only**, not new blind validation.",
        "- Unmeasured junction geometry, demand, turning proportions, alternate-route capacity and traveler response prevent operational/site-certified claims.",
        "- The three ensemble members are neither independent real-world observations nor statistically calibrated confidence bounds.",
        "", "## Reproduction", "",
        "Run `python -m streetlab_integration.research_results --run-dir <existing-research-output-folder>` after a successful `STREETLAB_RESEARCH.ps1` run. Source hashes and artifacts remain available for independent inspection.", "",
    ]
    return "\n".join(lines)


def _figure(row: dict) -> str:
    width, height = 810, 360
    vals = [row["branches"][branch]["median"] for branch in BRANCHES]
    maximum = max(vals) or 1
    ceiling = maximum * 1.18
    labels = ("Baseline", "Guided detour", "Mixed detour")
    palette = ("#465e74", "#137c72", "#71a9aa")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(row["name"])} synthetic scenario medians">',
        '<rect width="810" height="360" fill="white"/>',
        f'<text x="38" y="36" font-family="Arial,sans-serif" font-size="21" font-weight="bold" fill="#233942">{escape(row["name"])}</text>',
        f'<text x="38" y="60" font-family="Arial,sans-serif" font-size="12" fill="#5c6e78">{escape(row["unit"])} · simulation medians · assumed junction, not real-site effect</text>',
        '<line x1="55" y1="274" x2="760" y2="274" stroke="#89959b"/>',
    ]
    for idx, (branch, val) in enumerate(zip(BRANCHES, vals)):
        x = 130 + 240 * idx
        bar = 179 * val / ceiling
        elements += [
            f'<rect x="{x}" y="{274-bar:.2f}" width="100" height="{bar:.2f}" rx="5" fill="{palette[idx]}"/>',
            f'<text x="{x+50}" y="{max(86,265-bar):.2f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="bold" fill="#233942">{_fmt(val,row["key"])} </text>',
            f'<text x="{x+50}" y="300" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#233942">{labels[idx]}</text>',
        ]
    elements += [
        '<text x="38" y="341" font-family="Arial,sans-serif" font-size="11" fill="#687982">3 assignment members; no statistical confidence interval or field validation</text>',
        '</svg>',
    ]
    return "\n".join(elements) + "\n"


def outputs(evidence: dict, native: dict, evidence_hash: str, native_hash: str) -> dict[str, str]:
    parsed = analyze(evidence, native)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["metric", "unit", "scenario", "min", "median", "max", "evidence_class"])
    for row in parsed["metrics"]:
        for branch in BRANCHES:
            d = row["branches"][branch]
            writer.writerow([row["key"], row["unit"], branch, d["min"], d["median"], d["max"], "SYNTHETIC_ASSUMPTION_ASSIGNMENT"])
    files = {"research_results.md": report(parsed, evidence_hash, native_hash), "research_metrics.csv": buffer.getvalue()}
    for row, (_, _, _, filename) in zip(parsed["metrics"], METRICS):
        files[filename] = _figure(row)
    return files


def write_results(run_dir: Path) -> list[str]:
    run_dir = Path(run_dir)
    evidence_path, native_path = (run_dir / name for name in ("submission_evidence.json", "native_sumo_study.json"))
    raw_evidence, raw_native = evidence_path.read_bytes(), native_path.read_bytes()
    evidence_hash, native_hash = (hashlib.sha256(raw).hexdigest() for raw in (raw_evidence, raw_native))
    files = outputs(_load(evidence_path), _load(native_path), evidence_hash, native_hash)
    names = list(files) + ["research_results_manifest.json"]
    _required(all(not (run_dir / name).exists() for name in names), "Refusing to overwrite existing research results")
    manifest = {"schema_version": 1, "classification": "SYNTHETIC_RESEARCH_ONLY",
                "source_artifacts_sha256": {evidence_path.name: evidence_hash, native_path.name: native_hash},
                "produced_sha256": {name: hashlib.sha256(content.encode("utf-8")).hexdigest() for name, content in files.items()},
                "notes": "These hashes bind figures and tables to this simulation run, not to real-world field evidence."}
    for name, content in files.items():
        with (run_dir / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    with (run_dir / "research_results_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate checked, synthetic-only StreetLab results, tables and SVG figures")
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing successful native SUMO research run directory")
    args = parser.parse_args()
    try:
        names = write_results(args.run_dir)
    except (ResultsError, OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"Research results FAILED (no measured field claims): {exc}\n")
    print("Generated native research results: " + ", ".join(names))


if __name__ == "__main__":
    main()