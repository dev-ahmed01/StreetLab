"""Source integrity and local HTTP tests for the native SUMO research view."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from streetlab_integration.research_bundle_view import (
    BundleDisplayError, read_bundle,
)
from streetlab_integration.research_api import mount_research_routes
from streetlab_integration.research_release import package
from streetlab_integration.research_results import write_results


def _fixture(folder: Path) -> Path:
    folder.mkdir()
    keys = {
        "max_approach_queue_vehicles": (13, 5, 5),
        "arrivals_after_decision": (85, 76, 76),
        "mean_network_speed_mps": (9.96, 10.83, 10.78),
        "mean_instant_waiting_time_s": (0.245, 0.008, 0.015),
        "rerouted_vehicles": (0, 74, 73),
    }
    branches = ("BASELINE", "DETOUR_GUIDED", "DETOUR_MIXED")
    score = {
        branch: {
            "metrics": {key: {"min": vals[i], "median": vals[i], "max": vals[i]}
                        for key, vals in keys.items()},
            "vs_baseline": {name: {"min": 0, "median": 0, "max": 0}
                for name in ("arrivals_delta", "max_queue_delta",
                             "mean_speed_delta_mps", "mean_waiting_delta_s")},
        }
        for i, branch in enumerate(branches)
    }
    evidence = {
        "track": "RESEARCH_SUBMISSION_NO_NEW_FIELD_SURVEY",
        "research_workflow_status": "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION",
        "field_census_claimed": False,
        "external_site_transfer_validated": False,
        "real_world_intervention_effect_proven": False,
        "production_release_approved": False,
        "datasets_are_from_different_sites": True,
        "dataset_summaries": {
            "chennai": {"holdout_generalization_pass": False,
                        "capacity_gate_pass": False, "calibration_locked": True,
                        "retuning_allowed": False, "phase1_status": "FAIL"},
            "fluid_w04": {"scores_recomputed_in_this_run": False,
                          "data_split": "development-only",
                          "reported_point_precision_pct": 86.70,
                          "reported_point_recall_pct": 91.37},
        },
        "native_sumo": {
            "status": "SUMO_EXECUTED_SYNTHETIC_ONLY",
            "site_specific_field_validation": False,
            "production_release_approved": False,
            "simulation_not_real_world_effect": True,
            "ensemble_members": 3,
            "scorecard": score,
        },
    }
    native = {
        "incomplete_study": {"status": "NEEDS_DATA"},
        "supported_study": {"status": "SUPPORTED"},
        "guardrails": {
            "confidence_fabricated": False,
            "live_runtime_mutated_by_evaluated_branch": False,
            "authority_remains_final_decision_maker": True,
        },
        "ensemble": {
            "metadata": {"provenance": "ASSUMED",
                         "uncertainty_source": "ASSUMPTION_ASSIGNMENT",
                         "same_snapshot_for_all_members": True, "members": 3},
            "branch_summaries": score,
        },
    }
    (folder / "submission_evidence.json").write_text(json.dumps(evidence))
    (folder / "native_sumo_study.json").write_text(json.dumps(native))
    (folder / "native_sumo.log").write_text("actual native SUMO was executed in upstream fixture\n")
    (folder / "submission_summary.md").write_text("Only synthetic parameters\n")
    write_results(folder)
    archive = folder.parent / "research.zip"
    package(folder, archive)
    return archive


def test_verified_synthetic_medians_render_without_physical_validation(tmp_path):
    archive = _fixture(tmp_path / "run")
    result = read_bundle(archive)
    assert result["status"] == "VERIFIED_NATIVE_SUMO_SYNTHETIC_RESEARCH"
    assert result["ensemble_members"] == 3
    arrivals = next(x for x in result["metrics"] if x["key"] == "arrivals_after_decision")
    assert arrivals["medians"] == {
        "BASELINE": 85.0, "DETOUR_GUIDED": 76.0, "DETOUR_MIXED": 76.0
    }
    assert result["production_release_approved"] is False
    assert result["independent_holdout_pass"] is False
    assert result["historical_may25_audit_verified_by_this_endpoint"] is False


def test_uncorrupted_sumo_bundle_is_available_over_readonly_local_http(tmp_path, monkeypatch):
    archive = _fixture(tmp_path / "run")
    monkeypatch.setenv("STREETLAB_RESEARCH_BUNDLE_PATH", str(archive))
    app = FastAPI()
    mount_research_routes(app)
    with TestClient(app) as client:
        response = client.get("/api/research/simulation")
        assert response.status_code == 200
        assert len(response.json()["metrics"]) == 5
        page = client.get("/research")
        assert page.status_code == 200
        assert 'id="simulationRows"' in page.text
        assert "/api/research/simulation" in page.text


def test_unconfigured_or_missing_bundle_stays_unverified(monkeypatch, tmp_path):
    monkeypatch.setenv("STREETLAB_RESEARCH_BUNDLE_PATH", str(tmp_path / "absent.zip"))
    app = FastAPI()
    mount_research_routes(app)
    with TestClient(app) as client:
        result = client.get("/api/research/simulation")
        assert result.status_code == 409
        assert "unavailable or inconsistent" in result.json()["detail"]


def test_modified_evidence_zip_refused(tmp_path):
    archive = _fixture(tmp_path / "run")
    with ZipFile(archive, "a") as stream:
        stream.writestr("evidence/unsupported_claim.txt", "production=approved")
    with pytest.raises(BundleDisplayError, match="ZIP member"):
        read_bundle(archive)


def test_changed_source_hash_refused(tmp_path):
    archive = _fixture(tmp_path / "run")
    with ZipFile(archive) as stream:
        records = {name: stream.read(name) for name in stream.namelist()}
    source = json.loads(records["evidence/submission_evidence.json"])
    source["production_release_approved"] = True
    records["evidence/submission_evidence.json"] = json.dumps(source).encode()
    with ZipFile(archive, "w") as stream:
        for name, data in records.items():
            stream.writestr(name, data)
    with pytest.raises(BundleDisplayError, match="SHA mismatch"):
        read_bundle(archive)


def test_missing_or_malicious_zip_path_refused(tmp_path):
    with pytest.raises(BundleDisplayError, match="missing"):
        read_bundle(tmp_path / "not_there.zip")
    source = _fixture(tmp_path / "run")
    alias = tmp_path / "linked.zip"
    alias.symlink_to(source)
    with pytest.raises(BundleDisplayError, match="linked"):
        read_bundle(alias)


def test_malformed_zip_refused(tmp_path):
    corrupt = tmp_path / "research.zip"
    corrupt.write_bytes(b"not a ZIP")
    with pytest.raises(BundleDisplayError, match="valid bounded"):
        read_bundle(corrupt)
