"""Deterministic, reproducible synthetic training data generator for TrafficGuard AI.

DISCLAIMER:
-----------
The output of this generator is SYNTHETIC DEVELOPMENT DATA created strictly for
prototyping, training, and benchmarking ML risk prediction models (e.g. XGBoost).
It does NOT represent ground-truth real-world traffic observations or historical casualty data.
"""

import csv
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_backend_root = Path(__file__).resolve().parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.providers.demo_provider import DemoTrafficProvider
from app.services.traffic_normalizer import NormalizedTrafficRecord, normalize_record
from app.services.feature_engineering import extract_features

# Fixed Random Seed for 100% Deterministic Reproducibility
RANDOM_SEED: int = 42

# Time-Series Simulation Parameters (14 Days @ 1-hour resolution = 336 timestamps per location)
SIMULATION_DAYS: int = 14
BASE_START_TIME = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)

# Diurnal Multiplier Profiles
# (Volume multiplier relative to baseline, Speed factor relative to free flow speed)
HOURLY_PROFILES: Dict[int, Dict[str, float]] = {
    0: {"vol": 0.15, "speed_min": 0.90, "speed_max": 1.00},
    1: {"vol": 0.10, "speed_min": 0.92, "speed_max": 1.00},
    2: {"vol": 0.08, "speed_min": 0.94, "speed_max": 1.00},
    3: {"vol": 0.08, "speed_min": 0.94, "speed_max": 1.00},
    4: {"vol": 0.12, "speed_min": 0.92, "speed_max": 1.00},
    5: {"vol": 0.25, "speed_min": 0.88, "speed_max": 0.98},
    6: {"vol": 0.45, "speed_min": 0.80, "speed_max": 0.95},
    7: {"vol": 0.75, "speed_min": 0.65, "speed_max": 0.82},
    8: {"vol": 1.35, "speed_min": 0.38, "speed_max": 0.55},  # Morning Rush Peak
    9: {"vol": 1.45, "speed_min": 0.32, "speed_max": 0.50},  # Morning Rush Peak
    10: {"vol": 1.20, "speed_min": 0.42, "speed_max": 0.60},
    11: {"vol": 1.00, "speed_min": 0.55, "speed_max": 0.72},
    12: {"vol": 1.05, "speed_min": 0.52, "speed_max": 0.70},
    13: {"vol": 0.95, "speed_min": 0.58, "speed_max": 0.75},
    14: {"vol": 0.90, "speed_min": 0.62, "speed_max": 0.78},
    15: {"vol": 1.00, "speed_min": 0.58, "speed_max": 0.74},
    16: {"vol": 1.15, "speed_min": 0.48, "speed_max": 0.65},
    17: {"vol": 1.50, "speed_min": 0.30, "speed_max": 0.48},  # Evening Rush Peak
    18: {"vol": 1.65, "speed_min": 0.25, "speed_max": 0.42},  # Evening Rush Peak
    19: {"vol": 1.55, "speed_min": 0.28, "speed_max": 0.45},  # Evening Rush Peak
    20: {"vol": 1.25, "speed_min": 0.40, "speed_max": 0.58},
    21: {"vol": 0.85, "speed_min": 0.60, "speed_max": 0.76},
    22: {"vol": 0.50, "speed_min": 0.75, "speed_max": 0.90},
    23: {"vol": 0.30, "speed_min": 0.85, "speed_max": 0.96},
}


def compute_synthetic_risk_score(feature_vector: Any, rng: random.Random) -> float:
    """Compute the deterministic, leakage-free target risk score in [0.0, 100.0].

    MATHEMATICAL TARGET FORMULA:
    ----------------------------
    raw_score = (
        30.0 * congestion_ratio
      + 25.0 * incident_index
      + 20.0 * accident_severity
      + 15.0 * road_factor
      + 10.0 * traffic_pressure_composite
      + epsilon
    )
    risk_score = round(clamp(raw_score, 0.0, 100.0), 2)

    NOTE ON POLICE DEPLOYMENT EXCLUSION (NO TARGET LEAKAGE):
    --------------------------------------------------------
    police_officers is intentionally EXCLUDED from this target calculation.
    The ML model must estimate intrinsic traffic/road network hazard independently
    of current police presence. If police presence were included in the target,
    the downstream deployment optimizer would suffer from circular dependency
    and target leakage.

    Args:
        feature_vector: TrafficFeatureVector instance with computed interaction terms.
        rng: Deterministic pseudo-random number generator instance.

    Returns:
        float: Bounded target score in [0.0, 100.0].
    """
    epsilon = rng.uniform(-1.5, 1.5)

    raw_score = (
        30.0 * feature_vector.congestion_ratio
        + 25.0 * feature_vector.incident_index
        + 20.0 * feature_vector.accident_severity
        + 15.0 * feature_vector.road_factor
        + 10.0 * feature_vector.traffic_pressure_composite
        + epsilon
    )

    return round(min(max(raw_score, 0.0), 100.0), 2)


def generate_synthetic_dataset(
    days: int = SIMULATION_DAYS,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate the full historical synthetic dataset across all 20 monitored locations.

    Args:
        days: Number of days to simulate at 1-hour resolution.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]: (List of row dictionaries, Metadata dictionary)
    """
    rng = random.Random(seed)

    # 1. Read existing 20 monitored locations from PostgreSQL via provider
    provider = DemoTrafficProvider()
    base_records = provider.get_traffic_records()

    if len(base_records) == 0:
        raise RuntimeError("No location records found in PostgreSQL database.")

    total_hours = days * 24
    generated_rows: List[Dict[str, Any]] = []

    for hour_step in range(total_hours):
        current_time = BASE_START_TIME + timedelta(hours=hour_step)
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        is_weekend = day_of_week in (5, 6)

        profile = HOURLY_PROFILES[hour_of_day]

        # Weekend modulation (softer morning peak, extended afternoon volume)
        vol_mod = profile["vol"] * (0.80 if is_weekend and hour_of_day in (8, 9) else 1.0)
        vol_mod *= (1.15 if is_weekend and hour_of_day in (13, 14, 15, 16) else 1.0)

        for base in base_records:
            # Add subtle deterministic pseudo-random variance per location per hour
            loc_noise = rng.uniform(-0.06, 0.06)
            effective_vol_factor = max(0.05, vol_mod + loc_noise)

            # Simulated volume
            sim_volume = max(50, round(base.traffic_volume * effective_vol_factor))

            # Simulated speed (inversely coupled with volume loading + road hazard)
            speed_range_min = profile["speed_min"]
            speed_range_max = profile["speed_max"]
            speed_factor = rng.uniform(speed_range_min, speed_range_max)

            # High road hazard and population density reduce speed further
            congestion_drag = 0.08 * base.road_factor + 0.05 * base.population_factor
            effective_speed_factor = max(0.15, min(1.0, speed_factor - congestion_drag))
            sim_speed = round(max(3.0, base.free_flow_speed * effective_speed_factor), 1)

            # Simulated stochastic incidents (rare sporadic bumps)
            incident_noise = rng.uniform(-0.4, 0.4)
            sim_incident_freq = round(max(0.0, base.incident_frequency + incident_noise), 2)

            # Accident history is relatively static with minimal observation noise
            accident_noise = rng.uniform(-0.2, 0.2)
            sim_accident_hist = round(max(0.0, base.accident_history + accident_noise), 2)

            # 2. Normalize observation using existing traffic normalizer
            normalized = normalize_record({
                "location_id": base.location_id,
                "name": base.name,
                "latitude": base.latitude,
                "longitude": base.longitude,
                "coordinate_source": base.coordinate_source,
                "traffic_speed": sim_speed,
                "free_flow_speed": base.free_flow_speed,
                "traffic_volume": sim_volume,
                "incident_frequency": sim_incident_freq,
                "accident_history": sim_accident_hist,
                "road_factor": base.road_factor,
                "population_factor": base.population_factor,
                "police_officers": base.police_officers,
                "data_mode": "SYNTHETIC_DEVELOPMENT",
                "snapshot_timestamp": current_time,
            })

            # 3. Extract pure ML feature vector using existing feature engineering layer
            fv = extract_features(normalized)

            # 4. Compute deterministic leakage-free risk_score target
            target_risk = compute_synthetic_risk_score(fv, rng)

            # 5. Build tabular row dict
            row = {
                "location_id": fv.location_id,
                "name": fv.name,
                "latitude": fv.latitude,
                "longitude": fv.longitude,
                "timestamp": current_time.isoformat(),
                "data_mode": fv.data_mode,
                "traffic_speed": fv.traffic_speed,
                "free_flow_speed": fv.free_flow_speed,
                "congestion_ratio": fv.congestion_ratio,
                "speed_deficit": fv.speed_deficit,
                "speed_ratio": fv.speed_ratio,
                "traffic_volume": fv.traffic_volume,
                "volume_capacity_ratio": fv.volume_capacity_ratio,
                "incident_frequency": fv.incident_frequency,
                "incident_index": fv.incident_index,
                "accident_history": fv.accident_history,
                "accident_severity": fv.accident_severity,
                "road_factor": fv.road_factor,
                "population_factor": fv.population_factor,
                "traffic_pressure_composite": fv.traffic_pressure_composite,
                "police_officers": fv.police_officers,
                "hour_of_day": fv.hour_of_day,
                "day_of_week": fv.day_of_week,
                "is_weekend": fv.is_weekend,
                "is_peak_hour": fv.is_peak_hour,
                "risk_score": target_risk,
            }
            generated_rows.append(row)

    # Sort strictly by timestamp ascending, then location_id
    generated_rows.sort(key=lambda r: (r["timestamp"], r["location_id"]))

    metadata = {
        "dataset_name": "TrafficGuard AI Synthetic Development Training Dataset",
        "data_mode": "SYNTHETIC_DEVELOPMENT",
        "description": "Reproducible synthetic time-series traffic observations for XGBoost risk model prototyping.",
        "disclaimer": "SYNTHETIC BENCHMARK DATA ONLY. Must be replaced with real-world sensor/casualty ingest for production deployment.",
        "random_seed": seed,
        "total_records": len(generated_rows),
        "total_locations": len(base_records),
        "simulation_days": days,
        "time_resolution": "1 hour",
        "start_timestamp": BASE_START_TIME.isoformat(),
        "end_timestamp": (BASE_START_TIME + timedelta(hours=total_hours - 1)).isoformat(),
        "feature_count": 18,
        "target_variable": "risk_score",
        "target_range": [0.0, 100.0],
        "target_formula": "30.0*congestion_ratio + 25.0*incident_index + 20.0*accident_severity + 15.0*road_factor + 10.0*traffic_pressure_composite + U(-1.5, 1.5)",
        "police_leakage_mitigated": True,
        "chronological_split_strategy": "70% Train (earliest), 15% Validation (middle), 15% Test (latest)",
    }

    return generated_rows, metadata


def save_dataset_to_disk(
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """Save generated dataset CSV and metadata JSON to disk under backend/data/generated/.

    Args:
        rows: Generated dataset records.
        metadata: Metadata summary dict.
        output_dir: Output directory path.

    Returns:
        Path: Path to saved CSV file.
    """
    if output_dir is None:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        output_dir = backend_dir / "data" / "generated"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "synthetic_traffic_training_data.csv"
    meta_path = output_dir / "metadata.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return csv_path


if __name__ == "__main__":
    print("Generating reproducible synthetic development training dataset...")
    data, meta = generate_synthetic_dataset(days=14, seed=RANDOM_SEED)
    saved_path = save_dataset_to_disk(data, meta)
    print(f"Successfully generated {len(data)} records across {meta['total_locations']} locations.")
    print(f"Saved dataset artifact to: {saved_path}")
