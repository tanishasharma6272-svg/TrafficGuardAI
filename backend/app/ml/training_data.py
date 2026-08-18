"""Training dataset utilities, validation routines, and chronological splitters for TrafficGuard AI.

NOTE: All datasets loaded and processed by this module are marked as SYNTHETIC_DEVELOPMENT
for ML model prototyping and must not be treated as real-world ground-truth casualty data.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class SyntheticTrainingRecord(BaseModel):
    """Schema representing a single synthetic training observation."""

    location_id: int
    name: str
    latitude: float
    longitude: float
    timestamp: datetime
    data_mode: str = Field(default="SYNTHETIC_DEVELOPMENT")

    # Primary Traffic & Infrastructure Features
    traffic_speed: float = Field(..., ge=0.0)
    free_flow_speed: float = Field(..., gt=0.0)
    congestion_ratio: float = Field(..., ge=0.0, le=1.0)
    speed_deficit: float = Field(..., ge=0.0)
    speed_ratio: float = Field(..., ge=0.0, le=1.0)
    traffic_volume: int = Field(..., ge=0)
    volume_capacity_ratio: float = Field(..., ge=0.0, le=1.0)
    incident_frequency: float = Field(..., ge=0.0)
    incident_index: float = Field(..., ge=0.0, le=1.0)
    accident_history: float = Field(..., ge=0.0)
    accident_severity: float = Field(..., ge=0.0, le=1.0)
    road_factor: float = Field(..., ge=0.0, le=1.0)
    population_factor: float = Field(..., ge=0.0, le=1.0)
    traffic_pressure_composite: float = Field(..., ge=0.0, le=1.0)
    police_officers: int = Field(..., ge=0)

    # Temporal Embeddings
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: float = Field(..., ge=0.0, le=1.0)
    is_peak_hour: float = Field(..., ge=0.0, le=1.0)

    # ML Target Variable (Leakage-free intrinsic road risk)
    risk_score: float = Field(..., ge=0.0, le=100.0)


def load_training_dataset(
    filepath: Optional[Path] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load the synthetic training dataset from CSV into parsed dictionaries.

    Args:
        filepath: Optional path to the CSV file. Defaults to backend/data/generated/synthetic_traffic_training_data.csv.

    Returns:
        Tuple[List[str], List[Dict[str, Any]]]: Header column names and list of row dicts.
    """
    if filepath is None:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        filepath = backend_dir / "data" / "generated" / "synthetic_traffic_training_data.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Training dataset not found at '{filepath}'. Run app.ml.generate_training_data first."
        )

    rows: List[Dict[str, Any]] = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for raw_row in reader:
            row: Dict[str, Any] = {
                "location_id": int(raw_row["location_id"]),
                "name": raw_row["name"],
                "latitude": float(raw_row["latitude"]),
                "longitude": float(raw_row["longitude"]),
                "timestamp": datetime.fromisoformat(raw_row["timestamp"]),
                "data_mode": raw_row["data_mode"],
                "traffic_speed": float(raw_row["traffic_speed"]),
                "free_flow_speed": float(raw_row["free_flow_speed"]),
                "congestion_ratio": float(raw_row["congestion_ratio"]),
                "speed_deficit": float(raw_row["speed_deficit"]),
                "speed_ratio": float(raw_row["speed_ratio"]),
                "traffic_volume": int(raw_row["traffic_volume"]),
                "volume_capacity_ratio": float(raw_row["volume_capacity_ratio"]),
                "incident_frequency": float(raw_row["incident_frequency"]),
                "incident_index": float(raw_row["incident_index"]),
                "accident_history": float(raw_row["accident_history"]),
                "accident_severity": float(raw_row["accident_severity"]),
                "road_factor": float(raw_row["road_factor"]),
                "population_factor": float(raw_row["population_factor"]),
                "traffic_pressure_composite": float(raw_row["traffic_pressure_composite"]),
                "police_officers": int(raw_row["police_officers"]),
                "hour_of_day": int(raw_row["hour_of_day"]),
                "day_of_week": int(raw_row["day_of_week"]),
                "is_weekend": float(raw_row["is_weekend"]),
                "is_peak_hour": float(raw_row["is_peak_hour"]),
                "risk_score": float(raw_row["risk_score"]),
            }
            rows.append(row)

    return list(fieldnames), rows


def split_train_val_test_chronological(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Perform a strict chronological time-based train/validation/test split.

    RATIONALE:
    ----------
    In time-series and spatiotemporal risk estimation, random row splitting causes
    temporal leakage where future observations contaminate historical training folds.
    A strict chronological split ensures the model trains exclusively on past time windows
    and is validated/tested on unseen forward horizons.

    Splits:
        - Earliest 70% timestamps  -> Train set
        - Intermediate 15% timestamps -> Validation set
        - Latest 15% timestamps    -> Test set

    All monitored locations are represented across all splits because observations
    are synchronized across the network grid.

    Args:
        records: List of record dictionaries containing 'timestamp'.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: (train_records, val_records, test_records)
    """
    if not records:
        return [], [], []

    # Extract sorted unique timestamps across the entire dataset
    unique_timestamps = sorted(list({r["timestamp"] for r in records}))
    n_timestamps = len(unique_timestamps)

    if n_timestamps < 3:
        raise ValueError(f"Insufficient unique timestamps ({n_timestamps}) for 3-way chronological split.")

    train_end_idx = int(n_timestamps * train_ratio)
    val_end_idx = int(n_timestamps * (train_ratio + val_ratio))

    train_cutoff = unique_timestamps[train_end_idx]
    val_cutoff = unique_timestamps[val_end_idx]

    train_set: List[Dict[str, Any]] = []
    val_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    for r in records:
        ts = r["timestamp"]
        if ts < train_cutoff:
            train_set.append(r)
        elif ts < val_cutoff:
            val_set.append(r)
        else:
            test_set.append(r)

    return train_set, val_set, test_set


def verify_dataset_integrity(
    records: List[Dict[str, Any]],
    expected_locations: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute comprehensive validation checks on generated training dataset.

    Args:
        records: Parsed dataset rows.
        expected_locations: Optional expected number of unique locations.

    Returns:
        Dict[str, Any]: Summary dictionary of verified metrics.

    Raises:
        AssertionError: If any integrity constraint is violated.
    """
    assert len(records) > 0, "Dataset is empty"

    unique_locations = {r["location_id"] for r in records}
    if expected_locations is not None:
        assert len(unique_locations) == expected_locations, f"Expected {expected_locations} unique location IDs, found {len(unique_locations)}"
    else:
        assert len(unique_locations) > 0, "No unique location IDs found"

    unique_timestamps = {r["timestamp"] for r in records}
    assert len(unique_timestamps) > 0, "No timestamps found"

    min_risk = min(r["risk_score"] for r in records)
    max_risk = max(r["risk_score"] for r in records)
    assert 0.0 <= min_risk, f"risk_score below 0.0: {min_risk}"
    assert max_risk <= 100.0, f"risk_score above 100.0: {max_risk}"

    for i, r in enumerate(records):
        assert r["data_mode"] == "SYNTHETIC_DEVELOPMENT", f"Invalid data_mode at row {i}: {r['data_mode']}"
        assert r["traffic_speed"] >= 0.0, f"Negative speed at row {i}"
        assert r["free_flow_speed"] > 0.0, f"Non-positive free flow speed at row {i}"
        assert r["traffic_volume"] >= 0, f"Negative volume at row {i}"
        assert r["incident_frequency"] >= 0.0, f"Negative incident frequency at row {i}"
        assert r["accident_history"] >= 0.0, f"Negative accident history at row {i}"
        assert 0.0 <= r["road_factor"] <= 1.0, f"road_factor out of range at row {i}"
        assert 0.0 <= r["population_factor"] <= 1.0, f"population_factor out of range at row {i}"
        assert r["police_officers"] >= 0, f"Negative police officers at row {i}"

        # Verify no NaN / None values
        for k, v in r.items():
            assert v is not None, f"Null value in column '{k}' at row {i}"

    return {
        "total_records": len(records),
        "unique_locations": len(unique_locations),
        "unique_timestamps": len(unique_timestamps),
        "min_risk_score": min_risk,
        "max_risk_score": max_risk,
        "data_mode": "SYNTHETIC_DEVELOPMENT",
    }
