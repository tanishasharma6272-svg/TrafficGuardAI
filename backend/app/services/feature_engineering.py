"""Pure feature engineering layer preparing structured feature vectors for ML inference.

All transformations are deterministic, fully documented, and strictly mathematical.
No model training, simulation, or fabricated risk scores occur in this module.
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field

from app.services.traffic_normalizer import NormalizedTrafficRecord

# Explicit Physical and Benchmark Normalization Constants
# These represent network design capacity benchmarks matching baseline Nagpur network models.
MAX_DESIGN_CAPACITY_VOLUME: float = 50000.0  # Max 24h design saturation volume for primary arterials
MAX_INCIDENT_FREQUENCY: float = 10.0         # Upper benchmark scale for reported incident frequency
MAX_ACCIDENT_HISTORY: float = 10.0           # Upper benchmark scale for historical accident severity

# Standard Urban Peak Hours (Morning Rush 08:00-10:59, Evening Rush 17:00-20:59)
PEAK_HOURS: Tuple[int, ...] = (8, 9, 10, 17, 18, 19, 20)


class TrafficFeatureVector(BaseModel):
    """Structured feature vector ready for downstream XGBoost / ML risk model inference.

    Contains preserved normalized observations, engineered non-linear kinematic ratios,
    composite stress factors, and temporal embeddings.
    """

    # Identifiers & Metadata
    location_id: int
    name: str
    latitude: float
    longitude: float
    data_mode: str
    snapshot_timestamp: datetime

    # 1. Kinematic Speed Features
    traffic_speed: float = Field(..., description="Observed speed (km/h)")
    free_flow_speed: float = Field(..., description="Design free-flow speed (km/h)")
    congestion_ratio: float = Field(
        ...,
        description="Kinematic congestion loss: 1.0 - (traffic_speed / free_flow_speed), bounded [0.0, 1.0]",
    )
    speed_deficit: float = Field(
        ...,
        description="Absolute speed reduction below free flow limit: max(free_flow_speed - traffic_speed, 0.0) in km/h",
    )
    speed_ratio: float = Field(
        ...,
        description="Velocity attainment ratio: traffic_speed / free_flow_speed, bounded [0.0, 1.0]",
    )

    # 2. Volumetric & Capacity Features
    traffic_volume: int = Field(..., description="Observed vehicle throughput count")
    volume_capacity_ratio: float = Field(
        ...,
        description="Normalized throughput load: min(traffic_volume / 50000.0, 1.0)",
    )

    # 3. Incident & Historical Hazard Features
    incident_frequency: float = Field(..., description="Raw incident frequency index [0-10]")
    incident_index: float = Field(
        ...,
        description="Normalized incident rate: min(incident_frequency / 10.0, 1.0)",
    )
    accident_history: float = Field(..., description="Raw accident history index [0-10]")
    accident_severity: float = Field(
        ...,
        description="Normalized collision severity: min(accident_history / 10.0, 1.0)",
    )

    # 4. Infrastructure & Demographics
    road_factor: float = Field(..., description="Geometric infrastructure hazard factor [0.0, 1.0]")
    population_factor: float = Field(..., description="Surrounding pedestrian density factor [0.0, 1.0]")

    # 5. Composite Interaction Terms
    traffic_pressure_composite: float = Field(
        ...,
        description="Weighted load composite: (0.6 * volume_capacity_ratio) + (0.4 * population_factor)",
    )

    # 6. Operational Resource Deployment
    police_officers: int = Field(..., description="Police officer presence count")

    # 7. Temporal & Diurnal Features
    hour_of_day: int = Field(..., description="Hour of snapshot extraction [0-23]")
    day_of_week: int = Field(..., description="Day of week of snapshot extraction [0=Monday, 6=Sunday]")
    is_weekend: float = Field(..., description="1.0 if Saturday/Sunday else 0.0")
    is_peak_hour: float = Field(..., description="1.0 if hour is in (8,9,10, 17,18,19,20) else 0.0")


def calculate_congestion_ratio(traffic_speed: float, free_flow_speed: float) -> float:
    """Calculate the kinematic congestion ratio.

    Formula:
        congestion_ratio = 1.0 - (traffic_speed / free_flow_speed)
        Clamped to [0.0, 1.0]

    Rationale:
        Measures the proportional loss of velocity below statutory free-flow conditions.
        0.0 indicates free-flowing traffic, while 1.0 indicates complete gridlock/standstill.
    """
    if free_flow_speed <= 0.0:
        return 1.0
    speed_ratio = traffic_speed / free_flow_speed
    return round(min(max(1.0 - speed_ratio, 0.0), 1.0), 6)


def calculate_speed_deficit(traffic_speed: float, free_flow_speed: float) -> float:
    """Calculate absolute speed deficit in km/h.

    Formula:
        speed_deficit = max(free_flow_speed - traffic_speed, 0.0)

    Rationale:
        Direct dimensional indicator (km/h) of velocity lost due to congestion friction.
    """
    return round(max(free_flow_speed - traffic_speed, 0.0), 2)


def calculate_volume_capacity_ratio(traffic_volume: float) -> float:
    """Calculate normalized volumetric saturation ratio against design capacity.

    Formula:
        volume_capacity_ratio = min(traffic_volume / MAX_DESIGN_CAPACITY_VOLUME, 1.0)
        Where MAX_DESIGN_CAPACITY_VOLUME = 50,000 veh/day.

    Rationale:
        Normalizes raw vehicular throughput against the upper benchmark capacity of
        standard multi-lane arterial intersections in the network.
    """
    return round(min(max(traffic_volume / MAX_DESIGN_CAPACITY_VOLUME, 0.0), 1.0), 6)


def calculate_traffic_pressure_composite(
    volume_capacity_ratio: float, population_factor: float
) -> float:
    """Calculate composite physical traffic pressure index.

    Formula:
        traffic_pressure_composite = (0.6 * volume_capacity_ratio) + (0.4 * population_factor)

    Rationale & Mathematical Basis:
        Intersection hazard is a function of both vehicular volume throughput (60% weight)
        and urban conflict point density created by surrounding pedestrian activity (40% weight).
        This linear composite cleanly decouples dynamic vehicle loading from static
        demographic friction without hidden or uncalibrated constants.
    """
    comp = 0.6 * volume_capacity_ratio + 0.4 * population_factor
    return round(min(max(comp, 0.0), 1.0), 6)


def extract_features(record: NormalizedTrafficRecord) -> TrafficFeatureVector:
    """Extract a pure, fully engineered feature vector from a normalized observation.

    Args:
        record: Validated NormalizedTrafficRecord instance.

    Returns:
        TrafficFeatureVector: Complete feature vector ready for tabular ML inference.
    """
    ts = record.snapshot_timestamp

    # 1. Kinematic features
    congestion_ratio = calculate_congestion_ratio(record.traffic_speed, record.free_flow_speed)
    speed_deficit = calculate_speed_deficit(record.traffic_speed, record.free_flow_speed)
    speed_ratio = round(
        min(max(record.traffic_speed / record.free_flow_speed, 0.0), 1.0), 6
    ) if record.free_flow_speed > 0 else 0.0

    # 2. Volume & capacity
    volume_capacity_ratio = calculate_volume_capacity_ratio(float(record.traffic_volume))

    # 3. Incident & history normalization
    incident_index = round(min(max(record.incident_frequency / MAX_INCIDENT_FREQUENCY, 0.0), 1.0), 6)
    accident_severity = round(min(max(record.accident_history / MAX_ACCIDENT_HISTORY, 0.0), 1.0), 6)

    # 4. Composite interaction
    traffic_pressure = calculate_traffic_pressure_composite(
        volume_capacity_ratio=volume_capacity_ratio,
        population_factor=record.population_factor,
    )

    # 5. Temporal feature extraction
    hour = ts.hour
    day = ts.weekday()
    is_weekend = 1.0 if day in (5, 6) else 0.0
    is_peak = 1.0 if hour in PEAK_HOURS else 0.0

    return TrafficFeatureVector(
        location_id=record.location_id,
        name=record.name,
        latitude=record.latitude,
        longitude=record.longitude,
        data_mode=record.data_mode,
        snapshot_timestamp=ts,
        traffic_speed=record.traffic_speed,
        free_flow_speed=record.free_flow_speed,
        congestion_ratio=congestion_ratio,
        speed_deficit=speed_deficit,
        speed_ratio=speed_ratio,
        traffic_volume=record.traffic_volume,
        volume_capacity_ratio=volume_capacity_ratio,
        incident_frequency=record.incident_frequency,
        incident_index=incident_index,
        accident_history=record.accident_history,
        accident_severity=accident_severity,
        road_factor=record.road_factor,
        population_factor=record.population_factor,
        traffic_pressure_composite=traffic_pressure,
        police_officers=record.police_officers,
        hour_of_day=hour,
        day_of_week=day,
        is_weekend=is_weekend,
        is_peak_hour=is_peak,
    )


def extract_feature_batch(records: List[NormalizedTrafficRecord]) -> List[TrafficFeatureVector]:
    """Batch-extract feature vectors for a list of normalized traffic observations.

    Args:
        records: List of NormalizedTrafficRecord instances.

    Returns:
        List[TrafficFeatureVector]: List of engineered feature vectors.
    """
    return [extract_features(rec) for rec in records]


MODEL_FEATURE_NAMES: List[str] = [
    "traffic_speed",
    "free_flow_speed",
    "congestion_ratio",
    "speed_deficit",
    "speed_ratio",
    "traffic_volume",
    "volume_capacity_ratio",
    "incident_frequency",
    "incident_index",
    "accident_history",
    "accident_severity",
    "road_factor",
    "population_factor",
    "traffic_pressure_composite",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
]


def to_numerical_feature_dict(
    vector: TrafficFeatureVector,
    include_police: bool = False,
) -> Dict[str, float]:
    """Convert a TrafficFeatureVector into a numerical dictionary for ML models.

    Args:
        vector: TrafficFeatureVector instance.
        include_police: If True, includes 'police_officers' as an observational field.
                        Defaults to False (18 pure risk predictors to prevent target leakage).

    Returns:
        Dict[str, float]: Numerical feature mapping ready for XGBoost / Scikit-learn.
    """
    feat_dict: Dict[str, float] = {
        "traffic_speed": vector.traffic_speed,
        "free_flow_speed": vector.free_flow_speed,
        "congestion_ratio": vector.congestion_ratio,
        "speed_deficit": vector.speed_deficit,
        "speed_ratio": vector.speed_ratio,
        "traffic_volume": float(vector.traffic_volume),
        "volume_capacity_ratio": vector.volume_capacity_ratio,
        "incident_frequency": vector.incident_frequency,
        "incident_index": vector.incident_index,
        "accident_history": vector.accident_history,
        "accident_severity": vector.accident_severity,
        "road_factor": vector.road_factor,
        "population_factor": vector.population_factor,
        "traffic_pressure_composite": vector.traffic_pressure_composite,
        "hour_of_day": float(vector.hour_of_day),
        "day_of_week": float(vector.day_of_week),
        "is_weekend": vector.is_weekend,
        "is_peak_hour": vector.is_peak_hour,
    }
    if include_police:
        feat_dict["police_officers"] = float(vector.police_officers)
    return feat_dict


def to_feature_matrix(
    vectors: List[TrafficFeatureVector],
    include_police: bool = False,
) -> Tuple[List[str], List[List[float]]]:
    """Convert a list of feature vectors into standard (feature_names, 2D_matrix) tuple.

    Args:
        vectors: List of TrafficFeatureVector instances.
        include_police: Whether to include 'police_officers'. Defaults to False (18 predictors).

    Returns:
        Tuple[List[str], List[List[float]]]: Column names and 2D matrix rows.
    """
    if not vectors:
        return [], []

    sample_dict = to_numerical_feature_dict(vectors[0], include_police=include_police)
    feature_names = list(sample_dict.keys())

    matrix = []
    for vec in vectors:
        row_dict = to_numerical_feature_dict(vec, include_police=include_police)
        matrix.append([row_dict[name] for name in feature_names])

    return feature_names, matrix

