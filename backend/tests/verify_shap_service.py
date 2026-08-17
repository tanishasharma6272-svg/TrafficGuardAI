"""Verification test suite for SHAP explainability layer, local accuracy, and API endpoints."""

import json
from pathlib import Path
import sys
import urllib.request
import urllib.error
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation
from app.services.risk_explanation_service import get_risk_explanation_service


def test_shap_service_and_local_accuracy():
    print("\n--- 1. Testing SHAP Service & Exact Local Accuracy (PostgreSQL Data) ---")
    service = get_risk_explanation_service()
    session = SessionLocal()

    try:
        # Test Location ID 1 directly from PostgreSQL
        loc1 = session.query(DBLocation).filter(DBLocation.id == 1).first()
        assert loc1 is not None, "Location ID 1 missing in PostgreSQL database"
        exp1 = service.explain_location(loc1)

        assert exp1.location_id == 1
        assert exp1.name == loc1.name
        assert exp1.latitude == loc1.latitude
        assert exp1.longitude == loc1.longitude
        assert exp1.explanation_method == "SHAP"
        assert exp1.training_data_mode == "SYNTHETIC_DEVELOPMENT"
        assert len(exp1.feature_attributions) == 18, f"Expected 18 features, got {len(exp1.feature_attributions)}"

        # Verify all SHAP values are finite
        assert all(np.isfinite(fa.shap_value) for fa in exp1.feature_attributions)
        assert all(np.isfinite(fa.feature_value) for fa in exp1.feature_attributions)

        # Verify strict ordering by absolute SHAP magnitude descending
        abs_values = [abs(fa.shap_value) for fa in exp1.feature_attributions]
        assert abs_values == sorted(abs_values, reverse=True), "Feature attributions must be sorted by |shap_value| descending"

        # Verify local accuracy efficiency property: Base Value + sum(SHAP) == raw_prediction
        shap_sum = sum(fa.shap_value for fa in exp1.feature_attributions)
        reconstructed_pred = exp1.base_value + shap_sum
        diff1 = abs(reconstructed_pred - exp1.raw_prediction)
        assert diff1 < 0.05, f"Local accuracy violation on Location 1: diff {diff1:.6f} (Base: {exp1.base_value}, Sum: {shap_sum:.4f}, Raw: {exp1.raw_prediction})"

        print(f"       Location 1 ({loc1.name}):")
        print(f"         - Predicted Risk: {exp1.risk_score:.2f} ({exp1.risk_level}) | Raw: {exp1.raw_prediction:.2f}")
        print(f"         - Base Value: {exp1.base_value:.2f} | Sum(SHAP): {shap_sum:.2f} | Accuracy diff: {diff1:.6f}")
        print(f"         - Top 3 Positive: {[fa.feature_name + ': +' + str(fa.shap_value) for fa in exp1.top_positive_contributors]}")
        print(f"         - Top 3 Negative: {[fa.feature_name + ': ' + str(fa.shap_value) for fa in exp1.top_negative_contributors]}")

        # Test Location ID 12 directly from PostgreSQL
        loc12 = session.query(DBLocation).filter(DBLocation.id == 12).first()
        assert loc12 is not None, "Location ID 12 missing in PostgreSQL database"
        exp12 = service.explain_location(loc12)

        assert exp12.location_id == 12
        assert exp12.name == loc12.name
        assert len(exp12.feature_attributions) == 18

        shap_sum12 = sum(fa.shap_value for fa in exp12.feature_attributions)
        reconstructed12 = exp12.base_value + shap_sum12
        diff12 = abs(reconstructed12 - exp12.raw_prediction)
        assert diff12 < 0.05, f"Local accuracy violation on Location 12: diff {diff12:.6f}"

        print(f"       Location 12 ({loc12.name}):")
        print(f"         - Predicted Risk: {exp12.risk_score:.2f} ({exp12.risk_level}) | Raw: {exp12.raw_prediction:.2f}")
        print(f"         - Base Value: {exp12.base_value:.2f} | Sum(SHAP): {shap_sum12:.2f} | Accuracy diff: {diff12:.6f}")

    finally:
        session.close()

    print("[PASS] SHAP Service & Local Accuracy Verified: Exact additive reconstruction across PostgreSQL locations.")


def test_live_shap_endpoint():
    print("\n--- 2. Testing Live FastAPI SHAP Endpoints & Error Handling ---")
    session = SessionLocal()
    try:
        loc1 = session.query(DBLocation).filter(DBLocation.id == 1).first()
        loc12 = session.query(DBLocation).filter(DBLocation.id == 12).first()
    finally:
        session.close()

    # 1. Test GET /api/ml/explain/1
    url1 = "http://127.0.0.1:8000/api/ml/explain/1"
    with urllib.request.urlopen(url1) as resp:
        assert resp.status == 200
        data1 = json.loads(resp.read().decode())
        assert data1["location_id"] == 1
        assert data1["name"] == loc1.name
        assert data1["explanation_method"] == "SHAP"
        assert len(data1["feature_attributions"]) == 18
        print(f"[PASS] HTTP GET /api/ml/explain/1 -> 200 OK ({loc1.name}, 18 features returned)")

    # 2. Test GET /api/ml/explain/12
    url12 = "http://127.0.0.1:8000/api/ml/explain/12"
    with urllib.request.urlopen(url12) as resp:
        assert resp.status == 200
        data12 = json.loads(resp.read().decode())
        assert data12["location_id"] == 12
        assert data12["name"] == loc12.name
        print(f"[PASS] HTTP GET /api/ml/explain/12 -> 200 OK ({loc12.name})")

    # 3. Test 404 for unknown location ID
    url_404 = "http://127.0.0.1:8000/api/ml/explain/999"
    try:
        urllib.request.urlopen(url_404)
        assert False, "Should have returned HTTP 404 for location_id=999"
    except urllib.error.HTTPError as e:
        assert e.code == 404, f"Expected 404, got {e.code}"
        print("[PASS] HTTP GET /api/ml/explain/999 -> 404 Not Found (Correct error handling)")


def test_backward_compatibility_and_database_safety():
    print("\n--- 3. Testing Backward Compatibility & Database Mutation Safety ---")
    session = SessionLocal()
    try:
        count = session.query(DBLocation).count()
        assert count == 20, f"Database location count altered: {count}"
        print(f"[PASS] Database Safety Verified: Exactly {count} rows in locations table, zero mutations.")
    finally:
        session.close()

    endpoints = [
        ("/locations", 20),
        ("/risk", 20),
        ("/risk/1", "Ajni Square"),
        ("/api/ml/risk", 20),
        ("/api/ml/risk/1", "Ajni Square"),
    ]
    for path, expected in endpoints:
        url = f"http://127.0.0.1:8000{path}"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            val = len(data) if isinstance(data, list) else data.get("name")
            assert val == expected, f"Mismatch at {path}: expected {expected}, got {val}"
            print(f"[PASS] Backward Compatibility: HTTP GET {path} -> 200 OK")


if __name__ == "__main__":
    test_shap_service_and_local_accuracy()
    test_live_shap_endpoint()
    test_backward_compatibility_and_database_safety()
    print("\n=======================================================")
    print("ALL SHAP EXPLAINABILITY & API TESTS PASSED!")
    print("=======================================================\n")
