from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient


@dataclass
class FakeService:
    started: bool = False
    advanced_to: float | None = None
    evaluation_payload: dict | None = None
    closed: bool = False

    def start(self):
        self.started = True
        return {
            "simulation_id": "fake",
            "status": "RUNNING",
            "simulation_time_s": 0.0,
            "active_vehicles": 0,
            "mean_speed_mps": 0.0,
            "approach_queue_vehicles": 0,
        }

    def state(self):
        return {
            "simulation_id": "fake",
            "status": "RUNNING" if self.started else "CREATED",
            "simulation_time_s": self.advanced_to or 0.0,
            "active_vehicles": 12,
            "mean_speed_mps": 8.5,
            "approach_queue_vehicles": 2,
        }

    def advance_to(self, target_time_s: float):
        self.advanced_to = target_time_s
        return self.state()

    def evaluate_decision(self, **payload):
        self.evaluation_payload = payload
        return {
            "decision": {
                "type": "BLOCK_TURN",
                "from_edge": "WJ",
                "blocked_edge": "JN",
                "at_s": self.advanced_to or 0.0,
                "duration_s": payload["duration_s"],
            },
            "snapshot": {
                "snapshot_id": "decision_point",
                "simulation_time_s": self.advanced_to or 0.0,
                "sha256": "abc123",
            },
            "ensemble": {
                "ensemble": {
                    "members": payload["members"],
                    "uncertainty_source": "ASSUMPTION_ASSIGNMENT",
                    "provenance": "ASSUMED",
                    "same_snapshot_for_all_members": True,
                    "interpretation": "Scenario sensitivity, not a probability forecast.",
                },
                "branch_summaries": {
                    "BASELINE": {
                        "metrics": {
                            "max_approach_queue_vehicles": {
                                "min": 10.0,
                                "mean": 10.0,
                                "median": 10.0,
                                "p10": 10.0,
                                "p90": 10.0,
                                "max": 10.0,
                                "spread": 0.0,
                            }
                        },
                        "vs_baseline": {},
                    },
                    "MIXED_RESPONSE": {
                        "metrics": {
                            "max_approach_queue_vehicles": {
                                "min": 5.0,
                                "mean": 6.0,
                                "median": 6.0,
                                "p10": 5.2,
                                "p90": 6.8,
                                "max": 7.0,
                                "spread": 2.0,
                            }
                        },
                        "vs_baseline": {},
                    },
                },
                "member_runs": [],
            },
            "guardrails": {
                "uncertainty_is_probability_forecast": False,
                "authority_remains_final_decision_maker": True,
            },
        }

    def close(self):
        self.closed = True
        return {"status": "CLOSED"}


def _client(service: FakeService):
    from streetlab_phase2.api import create_app

    return TestClient(create_app(service=service))


def test_m5_api_exposes_runtime_lifecycle_and_state():
    service = FakeService()
    client = _client(service)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["product_role"] == "decision_assurance"

    started = client.post("/api/simulation/start")
    assert started.status_code == 200
    assert started.json()["status"] == "RUNNING"

    advanced = client.post(
        "/api/simulation/advance",
        json={"target_time_s": 60.0},
    )
    assert advanced.status_code == 200
    assert advanced.json()["simulation_time_s"] == 60.0

    state = client.get("/api/simulation/state")
    assert state.status_code == 200
    assert state.json()["approach_queue_vehicles"] == 2


def test_m5_api_evaluates_decision_without_recommending_one():
    service = FakeService(started=True, advanced_to=60.0)
    client = _client(service)

    response = client.post(
        "/api/decision/evaluate",
        json={
            "decision_type": "BLOCK_TURN",
            "from_edge": "WJ",
            "blocked_edge": "JN",
            "duration_s": 120.0,
            "guided_share": 0.50,
            "members": 4,
            "horizon_s": 120.0,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert service.evaluation_payload == {
        "decision_type": "BLOCK_TURN",
        "from_edge": "WJ",
        "blocked_edge": "JN",
        "duration_s": 120.0,
        "guided_share": 0.50,
        "members": 4,
        "horizon_s": 120.0,
    }
    assert body["ensemble"]["ensemble"]["provenance"] == "ASSUMED"
    assert body["guardrails"]["uncertainty_is_probability_forecast"] is False
    assert body["guardrails"]["authority_remains_final_decision_maker"] is True
    assert "recommendation" not in body
    assert "best_branch" not in body


def test_m5_ui_is_a_decision_lab_not_a_policy_recommender():
    service = FakeService()
    client = _client(service)

    page = client.get("/")
    assert page.status_code == 200
    html = page.text

    assert "StreetLab Decision Lab" in html
    assert "Current simulation" in html
    assert "Test a decision" in html
    assert "Scenario sensitivity" in html
    assert 'id="networkMap"' in html
    assert 'id="evaluateDecision"' in html
    assert 'id="branchResults"' in html
    assert "authority makes the final decision" in html.lower()
    assert "recommended option" not in html.lower()
    assert "best policy" not in html.lower()


def test_m5_network_endpoint_exposes_demo_graph_for_visualization():
    service = FakeService()
    client = _client(service)

    response = client.get("/api/network")
    assert response.status_code == 200
    body = response.json()

    assert {node["id"] for node in body["nodes"]} == {
        "W", "J", "E", "EO", "NE", "NM", "N"
    }
    assert {edge["id"] for edge in body["edges"]} == {
        "WJ", "JN", "NS", "JE", "EE", "EN", "N2"
    }
    assert body["decision_movement"] == {"from_edge": "WJ", "to_edge": "JN"}
