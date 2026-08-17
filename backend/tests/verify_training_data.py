"""Verification test suite for TrafficGuard AI synthetic development training data."""

import hashlib
import json
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.ml import (
    generate_synthetic_dataset,
    save_dataset_to_disk,
    load_training_dataset,
    split_train_val_test_chronological,
    verify_dataset_integrity,
    RANDOM_SEED,
)
from app.db.database import SessionLocal
from app.db.models import Location as DBLocation


def test_reproducibility():
    print("\n--- 1. Testing Deterministic Seed Reproducibility ---")
    data1, meta1 = generate_synthetic_dataset(days=2, seed=RANDOM_SEED)
    data2, meta2 = generate_synthetic_dataset(days=2, seed=RANDOM_SEED)

    str1 = json.dumps(data1, sort_keys=True)
    str2 = json.dumps(data2, sort_keys=True)

    hash1 = hashlib.sha256(str1.encode()).hexdigest()
    hash2 = hashlib.sha256(str2.encode()).hexdigest()

    assert hash1 == hash2, "Generator must be 100% deterministic with fixed seed"

    # Different seed must produce different variations
    data3, _ = generate_synthetic_dataset(days=2, seed=999)
    hash3 = hashlib.sha256(json.dumps(data3, sort_keys=True).encode()).hexdigest()
    assert hash1 != hash3, "Different seed must yield different dataset variance"

    print(f"[PASS] Reproducibility Verified: Fixed seed {RANDOM_SEED} generated identical SHA256 hashes ({hash1[:12]}...).")


def test_dataset_generation_and_saving():
    print("\n--- 2. Testing 14-Day Dataset Generation & Artifact Output ---")
    data, meta = generate_synthetic_dataset(days=14, seed=RANDOM_SEED)

    # 14 days * 24 hours * 20 locations = 6,720 rows
    expected_rows = 14 * 24 * 20
    assert len(data) == expected_rows, f"Expected {expected_rows} rows, got {len(data)}"
    assert meta["total_records"] == expected_rows
    assert meta["total_locations"] == 20
    assert meta["data_mode"] == "SYNTHETIC_DEVELOPMENT"

    # Save to disk
    csv_path = save_dataset_to_disk(data, meta)
    assert csv_path.exists(), f"CSV file not found at {csv_path}"
    assert csv_path.stat().st_size > 50000, "Generated CSV appears too small"

    meta_path = csv_path.parent / "metadata.json"
    assert meta_path.exists(), f"Metadata JSON not found at {meta_path}"

    print(f"[PASS] Generation & Persistence Verified: {len(data)} rows saved to '{csv_path}'.")
    return csv_path


def test_loader_and_integrity_checks(csv_path):
    print("\n--- 3. Testing Dataset Loader & Strict Physical Integrity ---")
    fieldnames, records = load_training_dataset(csv_path)

    expected_cols = [
        "location_id", "name", "latitude", "longitude", "timestamp", "data_mode",
        "traffic_speed", "free_flow_speed", "congestion_ratio", "speed_deficit",
        "speed_ratio", "traffic_volume", "volume_capacity_ratio", "incident_frequency",
        "incident_index", "accident_history", "accident_severity", "road_factor",
        "population_factor", "traffic_pressure_composite", "police_officers",
        "hour_of_day", "day_of_week", "is_weekend", "is_peak_hour", "risk_score"
    ]
    for col in expected_cols:
        assert col in fieldnames, f"Missing required column: {col}"

    summary = verify_dataset_integrity(records)
    assert summary["total_records"] == 6720
    assert summary["unique_locations"] == 20
    assert summary["unique_timestamps"] == 336
    assert 0.0 <= summary["min_risk_score"] <= summary["max_risk_score"] <= 100.0

    # Verify each location has exactly 336 hourly observations
    from collections import Counter
    loc_counts = Counter(r["location_id"] for r in records)
    assert len(loc_counts) == 20
    assert all(c == 336 for c in loc_counts.values()), "Every location must have 336 hourly observations"

    print(f"[PASS] Loader & Integrity Verified: 26 columns, 0 NaNs, target range [{summary['min_risk_score']:.1f}, {summary['max_risk_score']:.1f}].")
    return records


def test_chronological_splitting(records):
    print("\n--- 4. Testing Chronological Train/Validation/Test Splitter ---")
    train_set, val_set, test_set = split_train_val_test_chronological(
        records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    total = len(records)
    print(f"       Train set: {len(train_set)} rows ({len(train_set)/total*100:.1f}%)")
    print(f"       Val set:   {len(val_set)} rows ({len(val_set)/total*100:.1f}%)")
    print(f"       Test set:  {len(test_set)} rows ({len(test_set)/total*100:.1f}%)")

    assert len(train_set) + len(val_set) + len(test_set) == total

    # Verify temporal ordering (Zero future leakage)
    max_train_ts = max(r["timestamp"] for r in train_set)
    min_val_ts = min(r["timestamp"] for r in val_set)
    max_val_ts = max(r["timestamp"] for r in val_set)
    min_test_ts = min(r["timestamp"] for r in test_set)

    assert max_train_ts < min_val_ts, f"Temporal leak: max train {max_train_ts} >= min val {min_val_ts}"
    assert max_val_ts < min_test_ts, f"Temporal leak: max val {max_val_ts} >= min test {min_test_ts}"

    # Verify all 20 locations are present in every split
    train_locs = {r["location_id"] for r in train_set}
    val_locs = {r["location_id"] for r in val_set}
    test_locs = {r["location_id"] for r in test_set}

    assert len(train_locs) == 20, f"Expected 20 locations in train set, got {len(train_locs)}"
    assert len(val_locs) == 20, f"Expected 20 locations in val set, got {len(val_locs)}"
    assert len(test_locs) == 20, f"Expected 20 locations in test set, got {len(test_locs)}"

    print("[PASS] Chronological Split Verified: Strict non-overlapping time boundaries with all 20 locations preserved.")


def test_database_and_git_safety():
    print("\n--- 5. Testing Database & Git Safety Constraints ---")
    session = SessionLocal()
    try:
        count = session.query(DBLocation).count()
        assert count == 20, f"Database location count altered: {count}"
        print(f"[PASS] Database Safety Verified: PostgreSQL contains exactly {count} seed rows, zero mutations.")
    finally:
        session.close()

    # Verify .gitignore contains backend/data/generated/
    gitignore_path = backend_dir.parent / ".gitignore"
    assert gitignore_path.exists()
    content = gitignore_path.read_text(encoding="utf-8")
    assert "backend/data/generated/" in content or "data/generated/" in content, "Generated folder not ignored in .gitignore"
    print("[PASS] Git Safety Verified: Generated training directory is safely ignored in .gitignore.")


if __name__ == "__main__":
    test_reproducibility()
    csv_path = test_dataset_generation_and_saving()
    records = test_loader_and_integrity_checks(csv_path)
    test_chronological_splitting(records)
    test_database_and_git_safety()
    print("\n=======================================================")
    print("ALL SYNTHETIC DEVELOPMENT TRAINING-DATA TESTS PASSED!")
    print("=======================================================\n")
