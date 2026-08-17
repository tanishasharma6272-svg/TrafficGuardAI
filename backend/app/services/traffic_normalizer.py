"""Provider-independent traffic data normalization service for TrafficGuard AI."""

from datetime import datetime, timezone
from typing import Any, Iterable, List, Union
from pydantic import BaseModel, Field, field_validator

from app.providers.base import RawTrafficRecord


class NormalizedTrafficRecord(BaseModel):
    """Canonical internal representation of a traffic observation.

    Guarantees strict type validity, physical constraint validation, and non-mutated
    source preservation.

    TIMESTAMP NOTE:
    ---------------
    The snapshot_timestamp represents the provider extraction/snapshot epoch.
    For Demo records, it reflects the retrieval timestamp and does NOT imply
    live real-time sensor generation.
    """

    location_id: int = Field(..., description="Unique integer ID of the monitored location")
    name: str = Field(..., description="Descriptive name of the intersection or road segment")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS-84 latitude coordinate [-90.0, 90.0]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS-84 longitude coordinate [-180.0, 180.0]")
    coordinate_source: str = Field(..., description="Geospatial authority or origin reference")
    traffic_speed: float = Field(..., ge=0.0, description="Current observed speed in km/h (Must be >= 0.0)")
    free_flow_speed: float = Field(..., gt=0.0, description="Free flow speed in km/h (Must be > 0.0)")
    traffic_volume: int = Field(..., ge=0, description="Observed vehicle throughput count (Must be >= 0)")
    incident_frequency: float = Field(..., ge=0.0, description="Reported incident frequency (Must be >= 0.0)")
    accident_history: float = Field(..., ge=0.0, description="Historical collision rate/index (Must be >= 0.0)")
    road_factor: float = Field(..., ge=0.0, le=1.0, description="Road geometry hazard factor in [0.0, 1.0]")
    population_factor: float = Field(..., ge=0.0, le=1.0, description="Pedestrian/population density factor in [0.0, 1.0]")
    police_officers: int = Field(..., ge=0, description="Assigned police officer units (Must be >= 0)")
    data_mode: str = Field(..., description="Data mode (e.g., 'DEMO', 'LIVE')")
    snapshot_timestamp: datetime = Field(
        ...,
        description="UTC datetime of snapshot extraction. Does not denote live event epoch for demo records.",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Location name must not be empty.")
        return v.strip()


def normalize_record(raw: Union[RawTrafficRecord, dict, Any]) -> NormalizedTrafficRecord:
    """Validate and convert a raw provider record into a NormalizedTrafficRecord.

    Strictly validates physical bounds without silently altering or clamping source
    telemetry values. Raises ValueError on contract or physical constraint violations.

    Args:
        raw: RawTrafficRecord instance or compatible dictionary/object.

    Returns:
        NormalizedTrafficRecord: Validated, immutable normalized observation.

    Raises:
        ValueError: If any field fails strict physical constraint validation.
    """
    if isinstance(raw, RawTrafficRecord):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = raw.copy()
    elif hasattr(raw, "__dict__"):
        data = {k: v for k, v in raw.__dict__.items() if not k.startswith("_")}
    else:
        raise ValueError(f"Unsupported record type for normalization: {type(raw).__name__}")

    # Map provider_mode -> data_mode if needed
    if "data_mode" not in data and "provider_mode" in data:
        data["data_mode"] = data.pop("provider_mode")

    # Map id -> location_id if missing
    if "location_id" not in data and "id" in data:
        data["location_id"] = data["id"]

    # Explicit validation checks with clear descriptive errors before instantiation
    loc_id = data.get("location_id")
    name = data.get("name", f"Location-{loc_id}")

    traffic_speed = data.get("traffic_speed")
    if traffic_speed is None or traffic_speed < 0.0:
        raise ValueError(
            f"Invalid traffic_speed {traffic_speed} for '{name}' (ID: {loc_id}). Speed must be >= 0.0 km/h."
        )

    free_flow_speed = data.get("free_flow_speed")
    if free_flow_speed is None or free_flow_speed <= 0.0:
        raise ValueError(
            f"Invalid free_flow_speed {free_flow_speed} for '{name}' (ID: {loc_id}). Free-flow speed must be > 0.0 km/h."
        )

    traffic_volume = data.get("traffic_volume")
    if traffic_volume is None or traffic_volume < 0:
        raise ValueError(
            f"Invalid traffic_volume {traffic_volume} for '{name}' (ID: {loc_id}). Volume must be >= 0."
        )

    incident_frequency = data.get("incident_frequency")
    if incident_frequency is None or incident_frequency < 0.0:
        raise ValueError(
            f"Invalid incident_frequency {incident_frequency} for '{name}' (ID: {loc_id}). Incident frequency must be >= 0.0."
        )

    accident_history = data.get("accident_history")
    if accident_history is None or accident_history < 0.0:
        raise ValueError(
            f"Invalid accident_history {accident_history} for '{name}' (ID: {loc_id}). Accident history must be >= 0.0."
        )

    road_factor = data.get("road_factor")
    if road_factor is None or not (0.0 <= road_factor <= 1.0):
        raise ValueError(
            f"Invalid road_factor {road_factor} for '{name}' (ID: {loc_id}). Expected range [0.0, 1.0]."
        )

    population_factor = data.get("population_factor")
    if population_factor is None or not (0.0 <= population_factor <= 1.0):
        raise ValueError(
            f"Invalid population_factor {population_factor} for '{name}' (ID: {loc_id}). Expected range [0.0, 1.0]."
        )

    police_officers = data.get("police_officers", 0)
    if police_officers < 0:
        raise ValueError(
            f"Invalid police_officers {police_officers} for '{name}' (ID: {loc_id}). Count must be >= 0."
        )

    # Ensure snapshot_timestamp exists and has UTC timezone
    ts = data.get("snapshot_timestamp")
    if ts is None:
        ts = datetime.now(timezone.utc)
    elif isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    data["snapshot_timestamp"] = ts

    # Instantiate NormalizedTrafficRecord with full validation
    return NormalizedTrafficRecord(
        location_id=int(data["location_id"]),
        name=str(data["name"]),
        latitude=float(data["latitude"]),
        longitude=float(data["longitude"]),
        coordinate_source=str(data["coordinate_source"]),
        traffic_speed=float(data["traffic_speed"]),
        free_flow_speed=float(data["free_flow_speed"]),
        traffic_volume=int(data["traffic_volume"]),
        incident_frequency=float(data["incident_frequency"]),
        accident_history=float(data["accident_history"]),
        road_factor=float(data["road_factor"]),
        population_factor=float(data["population_factor"]),
        police_officers=int(data["police_officers"]),
        data_mode=str(data.get("data_mode", "DEMO")),
        snapshot_timestamp=data["snapshot_timestamp"],
    )


def normalize_batch(records: Iterable[Any]) -> List[NormalizedTrafficRecord]:
    """Normalize a batch of raw records from a traffic provider.

    Args:
        records: Iterable of raw provider records or dictionaries.

    Returns:
        List[NormalizedTrafficRecord]: List of validated normalized records.
    """
    return [normalize_record(rec) for rec in records]
