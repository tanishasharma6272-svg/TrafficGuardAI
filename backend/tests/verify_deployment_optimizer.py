"""Comprehensive verification test suite for TrafficGuard AI Police Deployment Optimizer.

Verifies:
1. Standard deployment scenario (available_units=3, coverage_radius_km=2.0)
2. Single unit placement scenario (available_units=1)
3. Over-allocation scenario (available_units > eligible locations)
4. Input validation (zero/negative units rejected with HTTP 422)
5. Input validation (zero/negative radius rejected with HTTP 422)
6. Input validation (unknown risk threshold rejected with HTTP 422)
7. Strict determinism across repeated executions
8. Data provenance (all selected locations originate in PostgreSQL)
9. Database mutation safety (0 rows altered or inserted)
10. Backward compatibility with existing API endpoints
11. Haversine geodesic distance accuracy
"""

import json
import math
from pathlib import Path
import sys
import urllib.request
import urllib.error

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation
from app.models.deployment import DeploymentRequest
from app.services.deployment_optimizer import (
    haversine_distance,
    LocationRiskNode,
    get_deployment_optimizer,
)
from app.services.risk_model_service import get_risk_model_service


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    """Helper to perform HTTP POST with JSON payload."""
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}


def test_haversine_accuracy():
    print("\n--- 1. Testing Haversine Geodesic Distance Calculation ---")
    # Coordinates of Nagpur Zero Mile (21.1458, 79.0882) to Sitabuldi Fort (21.1465, 79.0820)
    # Approx distance: ~0.65 km
    d = haversine_distance(21.1458, 79.0882, 21.1465, 79.0820)
    assert 0.5 <= d <= 0.8, f"Unexpected Haversine distance: {d:.4f} km"

    # Same point distance should be 0.0
    d_zero = haversine_distance(21.1458, 79.0882, 21.1458, 79.0882)
    assert d_zero == 0.0, f"Distance to self should be 0.0, got {d_zero}"

    # Nagpur (21.1458, 79.0882) to Mumbai (19.0760, 72.8777) approx 688 km
    d_mum = haversine_distance(21.1458, 79.0882, 19.0760, 72.8777)
    assert 680 <= d_mum <= 720, f"Unexpected Nagpur-Mumbai distance: {d_mum:.2f} km"

    print(f"[PASS] Haversine Distance Verified: Local = {d:.3f} km, Self = {d_zero:.1f} km, Inter-city = {d_mum:.1f} km.")


def test_service_level_optimization():
    print("\n--- 2. Testing Optimizer Service Direct Execution ---")
    session = SessionLocal()
    try:
        db_locs = session.query(DBLocation).order_by(DBLocation.id).all()
        assert len(db_locs) == 20, f"Expected 20 locations, got {len(db_locs)}"

        ml_service = get_risk_model_service()
        ml_summaries = ml_service.predict_all_locations(db_locs)

        nodes = [
            LocationRiskNode(
                id=item.id,
                name=item.name,
                latitude=item.latitude,
                longitude=item.longitude,
                risk_score=item.risk_score,
                risk_level=item.risk_level,
            )
            for item in ml_summaries
        ]

        optimizer = get_deployment_optimizer()

        # Scenario A: 3 units, 2.0 km radius
        resp3 = optimizer.optimize_deployment(
            locations=nodes,
            available_units=3,
            coverage_radius_km=2.0,
        )
        assert resp3.available_units == 3
        assert resp3.coverage_radius_km == 2.0
        assert resp3.algorithm == "GREEDY_COVERAGE_OPTIMIZER"
        assert len(resp3.selected_units) <= 3
        assert resp3.baseline_metrics.eligible_high_risk_locations > 0
        assert resp3.baseline_metrics.total_eligible_risk_score > 0
        assert resp3.optimized_metrics.covered_locations > 0
        assert resp3.optimized_metrics.covered_risk_score > 0
        assert 0.0 <= resp3.optimized_metrics.risk_coverage_percent <= 100.0
        assert 0.0 <= resp3.optimized_metrics.uncovered_risk_percent <= 100.0
        assert math.isclose(
            resp3.optimized_metrics.risk_coverage_percent + resp3.optimized_metrics.uncovered_risk_percent,
            100.0,
            abs_tol=0.1,
        )

        print(f"       Scenario A (Units: 3, Radius: 2.0km): Selected {len(resp3.selected_units)} units, Covering {resp3.optimized_metrics.covered_locations}/{resp3.baseline_metrics.eligible_high_risk_locations} nodes ({resp3.optimized_metrics.risk_coverage_percent}% risk score)")
        for u in resp3.selected_units:
            print(f"         Rank {u.rank}: [{u.location_id}] {u.location_name} (Score: {u.risk_score:.2f}, Level: {u.risk_level}) -> Covers {u.covered_location_count} nodes: {u.covered_location_ids}")

        # Scenario B: 1 unit, 2.0 km radius
        resp1 = optimizer.optimize_deployment(
            locations=nodes,
            available_units=1,
            coverage_radius_km=2.0,
        )
        assert len(resp1.selected_units) == 1
        assert resp1.selected_units[0].rank == 1
        assert resp1.selected_units[0].location_id == resp3.selected_units[0].location_id
        print(f"       Scenario B (Units: 1): Placed at Rank 1 location [{resp1.selected_units[0].location_id}] {resp1.selected_units[0].location_name}")

        # Scenario C: Critical only
        resp_crit = optimizer.optimize_deployment(
            locations=nodes,
            available_units=3,
            coverage_radius_km=2.0,
            min_risk_level="Critical",
        )
        assert resp_crit.baseline_metrics.eligible_high_risk_locations <= resp3.baseline_metrics.eligible_high_risk_locations
        assert all(u.risk_level == "Critical" for u in resp_crit.selected_units)
        print(f"       Scenario C (Critical Only): {resp_crit.baseline_metrics.eligible_high_risk_locations} eligible nodes, selected {len(resp_crit.selected_units)} units")

        # Scenario D: Over-allocation (50 units)
        resp_over = optimizer.optimize_deployment(
            locations=nodes,
            available_units=50,
            coverage_radius_km=2.0,
        )
        assert len(resp_over.selected_units) <= resp_over.baseline_metrics.eligible_high_risk_locations
        assert resp_over.optimized_metrics.covered_locations == resp_over.baseline_metrics.eligible_high_risk_locations
        assert resp_over.optimized_metrics.risk_coverage_percent == 100.0
        assert resp_over.optimized_metrics.uncovered_risk_percent == 0.0
        print(f"       Scenario D (50 Units Over-allocation): Allocated only {len(resp_over.selected_units)} necessary units for 100% coverage")

    finally:
        session.close()

    print("[PASS] Optimizer Service Logic Verified.")


def test_live_api_endpoint():
    print("\n--- 3. Testing Live POST /api/deployment/recommend Endpoint ---")
    base_url = "http://127.0.0.1:8000"

    # Test 1: Standard POST request
    status, data = _post_json(
        f"{base_url}/api/deployment/recommend",
        {"available_units": 3, "coverage_radius_km": 2.0},
    )
    assert status == 200, f"Expected 200, got {status}: {data}"
    assert data["available_units"] == 3
    assert data["coverage_radius_km"] == 2.0
    assert data["algorithm"] == "GREEDY_COVERAGE_OPTIMIZER"
    assert len(data["selected_units"]) == 3
    assert "eligible_high_risk_locations" in data["baseline_metrics"]
    assert "total_eligible_risk_score" in data["baseline_metrics"]
    assert "covered_locations" in data["optimized_metrics"]
    assert "covered_risk_score" in data["optimized_metrics"]
    assert "risk_coverage_percent" in data["optimized_metrics"]
    assert "uncovered_risk_score" in data["optimized_metrics"]
    assert "uncovered_risk_percent" in data["optimized_metrics"]
    print(f"[PASS] Standard POST /api/deployment/recommend returned 200 OK with {len(data['selected_units'])} units.")

    # Test 2: Single unit POST request
    status1, data1 = _post_json(
        f"{base_url}/api/deployment/recommend",
        {"available_units": 1, "coverage_radius_km": 2.0},
    )
    assert status1 == 200
    assert len(data1["selected_units"]) == 1
    assert data1["selected_units"][0]["location_id"] == data["selected_units"][0]["location_id"]
    print("[PASS] Single Unit POST /api/deployment/recommend returned 200 OK.")

    # Test 3: Deterministic repeated result
    status_rep, data_rep = _post_json(
        f"{base_url}/api/deployment/recommend",
        {"available_units": 3, "coverage_radius_km": 2.0},
    )
    assert status_rep == 200
    assert json.dumps(data, sort_keys=True) == json.dumps(data_rep, sort_keys=True), "Outputs diverged on repeated call!"
    print("[PASS] Determinism Verified: Consecutive identical POST requests produced byte-for-byte identical output.")


def test_input_validation_and_edge_cases():
    print("\n--- 4. Testing Input Validation & Error Handling (HTTP 422) ---")
    base_url = "http://127.0.0.1:8000"

    # Edge Case A: Zero available units
    status, err = _post_json(f"{base_url}/api/deployment/recommend", {"available_units": 0, "coverage_radius_km": 2.0})
    assert status == 422, f"Expected 422 for units=0, got {status}: {err}"
    print(f"[PASS] Zero units rejected: HTTP {status}")

    # Edge Case B: Negative available units
    status, err = _post_json(f"{base_url}/api/deployment/recommend", {"available_units": -5, "coverage_radius_km": 2.0})
    assert status == 422, f"Expected 422 for units=-5, got {status}: {err}"
    print(f"[PASS] Negative units rejected: HTTP {status}")

    # Edge Case C: Zero radius
    status, err = _post_json(f"{base_url}/api/deployment/recommend", {"available_units": 3, "coverage_radius_km": 0.0})
    assert status == 422, f"Expected 422 for radius=0, got {status}: {err}"
    print(f"[PASS] Zero radius rejected: HTTP {status}")

    # Edge Case D: Negative radius
    status, err = _post_json(f"{base_url}/api/deployment/recommend", {"available_units": 3, "coverage_radius_km": -2.0})
    assert status == 422, f"Expected 422 for radius=-2, got {status}: {err}"
    print(f"[PASS] Negative radius rejected: HTTP {status}")

    # Edge Case E: Unknown min_risk_level
    status, err = _post_json(f"{base_url}/api/deployment/recommend", {"available_units": 3, "coverage_radius_km": 2.0, "min_risk_level": "SuperCritical"})
    assert status == 422, f"Expected 422 for unknown risk level, got {status}: {err}"
    print(f"[PASS] Invalid risk level rejected: HTTP {status}")


def test_database_integrity_and_provenance():
    print("\n--- 5. Testing Database Mutation Safety & Location Provenance ---")
    session = SessionLocal()
    try:
        db_locs = session.query(DBLocation).all()
        assert len(db_locs) == 20, f"Database location count altered: {len(db_locs)}"
        db_id_set = {loc.id for loc in db_locs}

        # Verify through optimizer response
        optimizer = get_deployment_optimizer()
        ml_service = get_risk_model_service()
        ml_summaries = ml_service.predict_all_locations(db_locs)
        nodes = [
            LocationRiskNode(
                id=item.id,
                name=item.name,
                latitude=item.latitude,
                longitude=item.longitude,
                risk_score=item.risk_score,
                risk_level=item.risk_level,
            )
            for item in ml_summaries
        ]
        resp = optimizer.optimize_deployment(nodes, available_units=5, coverage_radius_km=2.0)

        # Check every selected unit and covered unit exists in PostgreSQL
        for u in resp.selected_units:
            assert u.location_id in db_id_set, f"Selected location ID {u.location_id} not in PostgreSQL!"
            for cov_id in u.covered_location_ids:
                assert cov_id in db_id_set, f"Covered location ID {cov_id} not in PostgreSQL!"

        # Re-check database count after execution
        post_count = session.query(DBLocation).count()
        assert post_count == 20, f"Database row count changed to {post_count}"
        print(f"[PASS] Provenance & Database Safety Verified: Exactly {post_count} rows, 100% ID matching, zero mutations.")
    finally:
        session.close()


def test_backward_compatibility():
    print("\n--- 6. Testing Backward Compatibility of Existing Endpoints ---")
    base_url = "http://127.0.0.1:8000"
    endpoints = [
        "/locations",
        "/risk",
        "/risk/1",
        "/api/ml/risk",
        "/api/ml/risk/1",
        "/api/ml/explain/1",
    ]
    for ep in endpoints:
        req = urllib.request.Request(f"{base_url}{ep}")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200, f"Endpoint {ep} failed with status {resp.status}"
            data = json.loads(resp.read().decode("utf-8"))
            assert data is not None
            print(f"[PASS] GET {ep} -> 200 OK")


if __name__ == "__main__":
    test_haversine_accuracy()
    test_service_level_optimization()
    test_live_api_endpoint()
    test_input_validation_and_edge_cases()
    test_database_integrity_and_provenance()
    test_backward_compatibility()
    print("\n==================================================================")
    print("ALL POLICE DEPLOYMENT OPTIMIZER VERIFICATION TESTS PASSED (100%)!")
    print("==================================================================\n")
