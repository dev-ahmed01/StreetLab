from __future__ import annotations

from fastapi.testclient import TestClient


class FakeService:
    def __init__(self):
        self.study_payload = None
        self.eval_payload = None

    def state(self):
        return {
            "simulation_id": "fake",
            "status": "CREATED",
            "simulation_time_s": 0.0,
            "active_vehicles": 0,
            "mean_speed_mps": 0.0,
            "approach_queue_vehicles": 0,
        }

    def start(self):
        return {**self.state(), "status": "RUNNING"}

    def advance_to(self, target_time_s):
        return {**self.state(), "status": "RUNNING", "simulation_time_s": target_time_s}

    def close(self):
        return {"status": "CLOSED"}

    def check_study(self, **payload):
        self.study_payload = payload
        return {
            "status": "NEEDS_DATA",
            "can_simulate": False,
            "missing_evidence": ["alternate_route"],
            "unsupported_metrics": [],
            "evidence_provenance": {
                "geometry": "ASSUMED",
                "demand": "ASSUMED",
            },
            "message": "StreetLab needs additional evidence before simulation.",
            "interpretation": "StreetLab refuses unsupported simulation.",
        }

    def evaluate_decision(self, **payload):
        self.eval_payload = payload
        return {
            "study": {"status": "SUPPORTED", "can_simulate": True},
            "decision_evaluated": True,
            "decision": {"type": payload["decision_type"]},
            "snapshot": {"sha256": "abc"},
            "ensemble": {
                "ensemble": {
                    "members": payload["members"],
                    "provenance": "ASSUMED",
                    "same_snapshot_for_all_members": True,
                },
                "branch_summaries": {},
                "member_runs": [],
            },
            "guardrails": {
                "authority_remains_final_decision_maker": True,
                "confidence_fabricated": False,
            },
        }


def _client(service):
    from streetlab_phase2.api import create_app
    return TestClient(create_app(service=service))


def test_m6_api_exposes_study_sufficiency_check():
    service = FakeService()
    client = _client(service)

    response = client.post(
        "/api/study/check",
        json={
            "decision_type": "APPLY_DETOUR",
            "area": "demo_junction",
            "baseline": "current_state",
            "scenario_family": "temporary_route_management",
            "requested_metrics": [
                "max_approach_queue_vehicles",
                "rerouted_vehicles",
            ],
            "evidence": [
                {"name": "geometry", "provenance": "ASSUMED"},
                {"name": "demand", "provenance": "ASSUMED"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "NEEDS_DATA"
    assert body["missing_evidence"] == ["alternate_route"]
    assert service.study_payload["decision_type"] == "APPLY_DETOUR"


def test_m6_api_passes_evidence_and_metrics_into_evaluation():
    service = FakeService()
    client = _client(service)

    response = client.post(
        "/api/decision/evaluate",
        json={
            "decision_type": "APPLY_DETOUR",
            "from_edge": "WJ",
            "blocked_edge": "JN",
            "duration_s": 120.0,
            "guided_share": 0.50,
            "members": 3,
            "horizon_s": 120.0,
            "requested_metrics": [
                "max_approach_queue_vehicles",
                "mean_network_speed_mps",
                "rerouted_vehicles",
            ],
            "evidence": [
                {"name": "geometry", "provenance": "ASSUMED"},
                {"name": "demand", "provenance": "ASSUMED"},
                {"name": "turn_movements", "provenance": "ASSUMED"},
                {"name": "alternate_route", "provenance": "ASSUMED"},
                {"name": "speeds", "provenance": "CALIBRATED"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"]["type"] == "APPLY_DETOUR"
    assert service.eval_payload["requested_metrics"] == [
        "max_approach_queue_vehicles",
        "mean_network_speed_mps",
        "rerouted_vehicles",
    ]
    assert service.eval_payload["evidence"][3]["name"] == "alternate_route"


def test_m6_ui_exposes_decision_type_and_evidence_status():
    client = _client(FakeService())

    html = client.get("/").text

    assert "APPLY_DETOUR" in html
    assert 'id="decisionType"' in html
    assert "Evidence sufficiency" in html
    assert 'id="studyStatus"' in html
    assert "NEEDS_DATA" in html
