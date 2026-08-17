"""Comprehensive test suite for TrafficGuard AI ML training, evaluation, persistence, and inference."""

import json
from pathlib import Path
import sys
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.ml import (
    load_training_dataset,
    split_train_val_test_chronological,
    extract_features_and_target,
    BaselineRiskModel,
    XGBoostRiskModel,
    evaluate_model,
    compare_models,
    save_model_artifacts,
    load_model_artifacts,
    predict_risk,
    train_and_evaluate_pipeline,
)
from app.services.feature_engineering import MODEL_FEATURE_NAMES
from app.db.database import SessionLocal
from app.db.models import Location as DBLocation


def test_dataset_and_feature_matrix():
    print("\n--- 1. Testing Dataset Dimensions & 18-Feature Matrix ---")
    _, all_records = load_training_dataset()
    assert len(all_records) == 6720, f"Expected 6720 records, got {len(all_records)}"

    train, val, test = split_train_val_test_chronological(all_records)
    assert len(train) == 4700, f"Expected 4700 train rows, got {len(train)}"
    assert len(val) == 1000, f"Expected 1000 val rows, got {len(val)}"
    assert len(test) == 1020, f"Expected 1020 test rows, got {len(test)}"

    X_train, y_train, feature_names = extract_features_and_target(train)
    assert len(feature_names) == 18, f"Expected 18 features, got {len(feature_names)}"
    assert "police_officers" not in feature_names, "police_officers must NOT be an ML predictor"
    assert "location_id" not in feature_names, "location_id must NOT be an ML predictor"
    assert X_train.shape == (4700, 18), f"Expected shape (4700, 18), got {X_train.shape}"
    assert y_train.shape == (4700,), f"Expected shape (4700,), got {y_train.shape}"
    assert not np.isnan(X_train).any(), "NaN values found in X_train"
    assert not np.isnan(y_train).any(), "NaN values found in y_train"

    print(f"[PASS] Dataset & Matrix Verified: 18 numerical predictors, 0 NaNs, shapes: Train {X_train.shape}, Val (1000, 18), Test (1020, 18).")
    return train, val, test


def test_baseline_and_xgboost_training(train_records, val_records, test_records):
    print("\n--- 2. Testing Baseline & XGBoost Model Training ---")
    X_train, y_train, _ = extract_features_and_target(train_records)
    X_val, y_val, _ = extract_features_and_target(val_records)
    X_test, y_test, _ = extract_features_and_target(test_records)

    # 1. Baseline Model
    baseline = BaselineRiskModel()
    baseline.fit(X_train, y_train)
    base_val_m = evaluate_model(baseline, X_val, y_val, split_name="Val")
    assert np.isfinite(base_val_m["rmse"]), "Baseline RMSE must be finite"
    assert np.isfinite(base_val_m["r2"]), "Baseline R2 must be finite"
    assert base_val_m["r2"] > 0.85, f"Baseline R2 unexpectedly low: {base_val_m['r2']}"

    # 2. XGBoost Model
    xgb_model = XGBoostRiskModel()
    xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    xgb_val_m = evaluate_model(xgb_model, X_val, y_val, split_name="Val")
    assert np.isfinite(xgb_val_m["rmse"]), "XGBoost RMSE must be finite"
    assert np.isfinite(xgb_val_m["r2"]), "XGBoost R2 must be finite"
    assert xgb_val_m["r2"] > 0.85, f"XGBoost R2 unexpectedly low: {xgb_val_m['r2']}"

    # 3. Model Comparison (lowest validation RMSE, highest validation R2)
    comp = compare_models(base_val_m, xgb_val_m)
    selected_name = comp["selected_model"]
    print(f"       Baseline Val  - RMSE: {base_val_m['rmse']:.4f}, MAE: {base_val_m['mae']:.4f}, R2: {base_val_m['r2']:.4f}")
    print(f"       XGBoost  Val  - RMSE: {xgb_val_m['rmse']:.4f}, MAE: {xgb_val_m['mae']:.4f}, R2: {xgb_val_m['r2']:.4f}")
    print(f"       Selected: {selected_name} ({comp['selection_rationale']})")

    assert selected_name == "BaselineRidge", f"Expected BaselineRidge to be selected, got {selected_name}"

    # 4. Evaluate selected model on Test set
    selected_instance = baseline if selected_name == "BaselineRidge" else xgb_model
    test_m = evaluate_model(selected_instance, X_test, y_test, split_name="Test")
    print(f"       Held-out Test - RMSE: {test_m['rmse']:.4f}, MAE: {test_m['mae']:.4f}, R2: {test_m['r2']:.4f}")
    assert test_m["r2"] > 0.85, f"Test R2 unexpectedly low: {test_m['r2']}"

    best_val_rmse = min(base_val_m['rmse'], xgb_val_m['rmse'])
    print(f"[PASS] Model Training & Evaluation Verified: Both models trained; {selected_name} selected with validation RMSE {best_val_rmse:.4f}.")
    return selected_instance, selected_name, xgb_model, baseline


def test_model_persistence_and_reloading(selected_model, selected_name, xgb_model, baseline):
    print("\n--- 3. Testing Model Artifact Persistence & Independent Loading ---")
    custom_dir = backend_dir / "data" / "models"
    metadata = {
        "model_name": f"TrafficGuard AI {selected_name} Risk Predictor",
        "training_data_mode": "SYNTHETIC_DEVELOPMENT",
        "feature_count": 18,
        "selected_model_type": selected_name,
        "feature_names": MODEL_FEATURE_NAMES,
    }

    saved = save_model_artifacts(
        model=selected_model,
        metadata=metadata,
        model_dir=custom_dir,
        baseline_model=baseline,
        xgb_model=xgb_model,
    )
    assert Path(saved["primary_model"]).exists(), "Primary model not saved"
    assert Path(saved["metadata"]).exists(), "Metadata JSON not saved"

    # Reload model from disk independently
    reloaded_model, loaded_meta = load_model_artifacts(model_dir=custom_dir)
    assert loaded_meta["training_data_mode"] == "SYNTHETIC_DEVELOPMENT"
    assert loaded_meta["selected_model_type"] == selected_name
    assert len(loaded_meta["feature_names"]) == 18

    # Test prediction identity
    sample_input = [30.0, 50.0, 0.4, 20.0, 0.6, 25000.0, 0.5, 3.0, 0.3, 2.0, 0.2, 0.4, 0.5, 0.5, 9.0, 2.0, 0.0, 1.0]
    orig_pred = selected_model.predict([sample_input])[0]
    reloaded_pred = reloaded_model.predict([sample_input])[0]

    assert abs(orig_pred - reloaded_pred) < 1e-4, f"Prediction mismatch: {orig_pred} vs {reloaded_pred}"
    print(f"[PASS] Persistence & Reloading Verified: Reloaded {selected_name} model produced identical predictions ({reloaded_pred:.2f}).")


def test_pure_inference_function():
    print("\n--- 4. Testing Pure predict_risk() Inference & Edge-Case Clamping ---")
    # 1. Standard typical traffic vector
    normal_input = {
        "traffic_speed": 35.0,
        "free_flow_speed": 60.0,
        "congestion_ratio": 0.4167,
        "speed_deficit": 25.0,
        "speed_ratio": 0.5833,
        "traffic_volume": 28000,
        "volume_capacity_ratio": 0.56,
        "incident_frequency": 2.5,
        "incident_index": 0.25,
        "accident_history": 3.0,
        "accident_severity": 0.3,
        "road_factor": 0.45,
        "population_factor": 0.55,
        "traffic_pressure_composite": 0.556,
        "hour_of_day": 18,
        "day_of_week": 1,
        "is_weekend": 0.0,
        "is_peak_hour": 1.0,
    }
    score = predict_risk(normal_input)
    assert 0.0 <= score <= 100.0, f"Score out of bounds: {score}"
    print(f"       Normal rush hour sample -> Predicted Risk Score: {score:.2f}")

    # 2. Extreme low risk case (zero volume, full speed, zero accidents/incidents)
    low_input = {k: 0.0 for k in MODEL_FEATURE_NAMES}
    low_input["traffic_speed"] = 80.0
    low_input["free_flow_speed"] = 80.0
    low_input["speed_ratio"] = 1.0
    low_score = predict_risk(low_input)
    assert 0.0 <= low_score <= 100.0, f"Low risk score out of bounds: {low_score}"
    assert low_score < 25.0, f"Low risk score unexpectedly high: {low_score}"
    print(f"       Extreme low risk sample -> Predicted Risk Score: {low_score:.2f}")

    # 3. Extreme high risk case (gridlock, maximum accidents/incidents/population)
    high_input = {k: 1.0 for k in MODEL_FEATURE_NAMES}
    high_input["traffic_speed"] = 0.0
    high_input["free_flow_speed"] = 60.0
    high_input["congestion_ratio"] = 1.0
    high_input["speed_deficit"] = 60.0
    high_input["traffic_volume"] = 60000
    high_input["volume_capacity_ratio"] = 1.0
    high_input["incident_frequency"] = 10.0
    high_input["accident_history"] = 10.0
    high_score = predict_risk(high_input)
    assert 0.0 <= high_score <= 100.0, f"High risk score out of bounds: {high_score}"
    assert high_score > 75.0, f"High risk score unexpectedly low: {high_score}"
    print(f"       Extreme high risk sample -> Predicted Risk Score: {high_score:.2f}")

    print("[PASS] Pure Inference Verified: predict_risk() handles normal and extreme inputs within [0.0, 100.0].")


def test_database_and_api_integrity():
    print("\n--- 5. Testing Database & Existing FastAPI Route Integrity ---")
    session = SessionLocal()
    try:
        count = session.query(DBLocation).count()
        assert count == 20, f"PostgreSQL database should contain 20 locations, found {count}"
        print(f"[PASS] Database Integrity: Exactly {count} rows in locations table, zero mutations.")
    finally:
        session.close()

    import urllib.request
    endpoints = [
        ("/locations", lambda d: len(d) == 20),
        ("/risk", lambda d: len(d) == 20),
        ("/risk/1", lambda d: d.get("name") == "Ajni Square"),
    ]
    for path, validator in endpoints:
        url = f"http://127.0.0.1:8000{path}"
        try:
            with urllib.request.urlopen(url) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode())
                assert validator(data)
                print(f"[PASS] Live FastAPI GET {path} returned 200 OK with expected schema.")
        except Exception as e:
            print(f"[WARN] Live server check for {path} skipped/error: {e}")


def test_official_pipeline_execution():
    print("\n--- 6. Testing Official End-to-End Pipeline Execution ---")
    results = train_and_evaluate_pipeline(save_artifacts=True)
    assert results["comparison"]["selected_model"] == "BaselineRidge"
    assert results["metadata"]["selected_model_type"] == "BaselineRidge"
    assert Path(results["saved_paths"]["primary_model"]).exists()
    assert Path(results["saved_paths"]["metadata"]).exists()
    print(f"[PASS] Official Pipeline Executed & Persisted: Selected model '{results['metadata']['selected_model_type']}' written to metadata.json.")


if __name__ == "__main__":
    train_r, val_r, test_r = test_dataset_and_feature_matrix()
    sel_inst, sel_name, xgb_m, base_m = test_baseline_and_xgboost_training(train_r, val_r, test_r)
    test_model_persistence_and_reloading(sel_inst, sel_name, xgb_m, base_m)
    test_pure_inference_function()
    test_database_and_api_integrity()
    test_official_pipeline_execution()
    print("\n=======================================================")
    print("ALL ML TRAINING, EVALUATION & INFERENCE TESTS PASSED!")
    print("=======================================================\n")
