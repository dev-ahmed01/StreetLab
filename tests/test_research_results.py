"""Cross-check result exporter and its fail-closed scientific claims."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from streetlab_integration.research_results import ResultsError, analyze, write_results


def sample():
    metrics = {
        "max_approach_queue_vehicles": (13, 5, 5),
        "arrivals_after_decision": (85, 76, 76),
        "mean_network_speed_mps": (9.96, 10.83, 10.78),
        "mean_instant_waiting_time_s": (0.245, 0.008, 0.015),
        "rerouted_vehicles": (0, 74, 73),
    }
    names = ("BASELINE", "DETOUR_GUIDED", "DETOUR_MIXED")
    score = {branch: {"metrics": {key: {"min": vals[i], "median": vals[i], "max": vals[i]}
               for key, vals in metrics.items()},
               "vs_baseline": {delta: {"min": 0, "median": 0, "max": 0} for delta in
                  ("arrivals_delta", "max_queue_delta", "mean_speed_delta_mps", "mean_waiting_delta_s")}}
             for i, branch in enumerate(names)}
    evidence = {"track": "RESEARCH_SUBMISSION_NO_NEW_FIELD_SURVEY",
                "research_workflow_status": "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION",
                "field_census_claimed": False, "external_site_transfer_validated": False,
                "real_world_intervention_effect_proven": False, "production_release_approved": False,
                "datasets_are_from_different_sites": True,
                "dataset_summaries": {
                    "chennai": {"holdout_generalization_pass": False, "capacity_gate_pass": False,
                                "calibration_locked": True, "retuning_allowed": False, "phase1_status": "FAIL"},
                    "fluid_w04": {"scores_recomputed_in_this_run": False, "data_split": "development-only",
                                  "reported_point_precision_pct": 86.70, "reported_point_recall_pct": 91.37}},
                "native_sumo": {"status": "SUMO_EXECUTED_SYNTHETIC_ONLY", "site_specific_field_validation": False,
                                "production_release_approved": False, "simulation_not_real_world_effect": True,
                                "ensemble_members": 3, "scorecard": score}}
    native = {"incomplete_study": {"status": "NEEDS_DATA"}, "supported_study": {"status": "SUPPORTED"},
              "guardrails": {"confidence_fabricated": False, "live_runtime_mutated_by_evaluated_branch": False,
                             "authority_remains_final_decision_maker": True},
              "ensemble": {"metadata": {"provenance": "ASSUMED", "uncertainty_source": "ASSUMPTION_ASSIGNMENT",
                                        "same_snapshot_for_all_members": True, "members": 3},
                           "branch_summaries": score}}
    return evidence, native


def test_exporter_creates_bound_result_files(tmp_path):
    evidence, native = sample()
    (tmp_path / "submission_evidence.json").write_text(json.dumps(evidence))
    (tmp_path / "native_sumo_study.json").write_text(json.dumps(native))
    names = write_results(tmp_path)
    assert "research_results.md" in names and "figure_02_arrivals.svg" in names
    report = (tmp_path / "research_results.md").read_text()
    assert "Trade-off" in report and "not a confidence interval" in report
    assert "failed" in report.lower() and "development-only" in report
    assert "not a measured traffic improvement" in report.lower()
    assert (tmp_path / "research_metrics.csv").read_text().count("SYNTHETIC_ASSUMPTION_ASSIGNMENT") == 15
    manifest = json.loads((tmp_path / "research_results_manifest.json").read_text())
    assert manifest["source_artifacts_sha256"]["native_sumo_study.json"] == hashlib.sha256((tmp_path / "native_sumo_study.json").read_bytes()).hexdigest()
    with pytest.raises(ResultsError, match="overwrite"):
        write_results(tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda e, n: e.update(production_release_approved=True),
    lambda e, n: e["dataset_summaries"]["chennai"].update(holdout_generalization_pass=True),
    lambda e, n: e["dataset_summaries"]["fluid_w04"].update(scores_recomputed_in_this_run=True),
    lambda e, n: n["ensemble"]["metadata"].update(provenance="OBSERVED"),
    lambda e, n: n["ensemble"]["metadata"].update(members=1),
    lambda e, n: e["native_sumo"].update(scorecard={}),
    lambda e, n: n["ensemble"]["branch_summaries"]["BASELINE"]["metrics"]["arrivals_after_decision"].update(median=float("nan")),
    lambda e, n: n["ensemble"]["branch_summaries"]["BASELINE"]["metrics"]["arrivals_after_decision"].update(min=100),
])
def test_rejects_unsubstantiated_or_invalid_evidence(mutate):
    evidence, native = sample()
    mutate(evidence, native)
    with pytest.raises(ResultsError):
        analyze(evidence, native)