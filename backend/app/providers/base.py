"""Base interface, raw data models, and typed provider status for TrafficGuard AI."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

AggregateState = Literal["LIVE", "PARTIAL", "ERROR", "UNCONFIGURED"]


class ProviderStatus(BaseModel):
    """Typed operational health, configuration, and telemetry status for a traffic data provider."""

    provider: str = Field(..., description="Provider class or implementation name")
    aggregate_state: AggregateState = Field(
        ..., description="Current operational state: 'LIVE', 'PARTIAL', 'ERROR', or 'UNCONFIGURED'"
    )
    successful_count: int = Field(default=0, ge=0, description="Count of successfully queried locations")
    failed_count: int = Field(default=0, ge=0, description="Count of failed location queries")
    total_locations: int = Field(default=0, ge=0, description="Total monitored locations evaluated")
    last_fetch_timestamp: Optional[datetime] = Field(
        default=None, description="UTC datetime when TrafficGuard AI performed the last fetch"
    )
    is_configured: bool = Field(default=True, description="Whether required credentials/settings are present")
    per_location_errors: Dict[int, str] = Field(
        default_factory=dict, description="Map of location_id to error diagnostic messages"
    )


class RawTrafficRecord(BaseModel):
    """Raw traffic observation emitted by a data provider before normalization.

    Attributes:
        location_id: Unique identifier of the monitored location or sensor station.
        name: Common name or arterial intersection description.
        latitude: WGS-84 latitude coordinate.
        longitude: WGS-84 longitude coordinate.
        coordinate_source: Description or URI of the geospatial reference source.
        traffic_speed: Current observed traffic velocity (km/h).
        free_flow_speed: Uncongested design or statutory speed limit (km/h).
        traffic_volume: Observed vehicle throughput count over observation window.
        incident_frequency: Reported traffic incident/stall frequency index.
        accident_history: Historical accident severity and collision history index.
        road_factor: Geometric, physical, or road infrastructure hazard factor.
        population_factor: Surrounding pedestrian and population density factor.
        police_officers: Stationed or assigned traffic police unit count.
        provider_mode: Ingestion mode ("DEMO" for simulated/seed records, "LIVE" for sensors).
        snapshot_timestamp: UTC datetime when TrafficGuard AI retrieved/extracted this snapshot.
            NOTE: For TomTomTrafficProvider, this is the UTC timestamp when the API response was received.
            For DemoTrafficProvider, this represents the extraction timestamp and not live historical events.
        raw_metadata: Provider-specific raw sensor telemetry, diagnostic metadata, and structured provenance.
    """

    location_id: int = Field(..., description="Location ID")
    name: str = Field(..., description="Location name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    coordinate_source: str = Field(..., description="Geospatial reference source")
    traffic_speed: float = Field(..., description="Observed speed (km/h)")
    free_flow_speed: float = Field(..., description="Free flow speed (km/h)")
    traffic_volume: int = Field(..., description="Vehicle volume count")
    incident_frequency: float = Field(..., description="Incident frequency index")
    accident_history: float = Field(..., description="Accident history index")
    road_factor: float = Field(..., description="Road hazard factor")
    population_factor: float = Field(..., description="Population density factor")
    police_officers: int = Field(..., description="Police officer count")
    provider_mode: str = Field(default="DEMO", description="Provider data mode (DEMO / LIVE)")
    snapshot_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC datetime when TrafficGuard AI retrieved/extracted this snapshot",
    )
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider metadata and provenance")


class TrafficProvider(ABC):
    """Abstract Base Class defining the contract for all traffic data ingestion providers."""

    @property
    @abstractmethod
    def provider_mode(self) -> str:
        """Return the operational mode of the provider (e.g. 'DEMO', 'LIVE', 'UNCONFIGURED')."""
        pass

    @abstractmethod
    def get_traffic_records(self) -> List[RawTrafficRecord]:
        """Fetch raw traffic records for all monitored locations in the network.

        Returns:
            List[RawTrafficRecord]: List of raw observations from this provider.
        """
        pass

    @abstractmethod
    def get_location_traffic_record(self, location_id: int) -> Optional[RawTrafficRecord]:
        """Fetch raw traffic record for a single monitored location by ID.

        Args:
            location_id: The integer identifier of the target location.

        Returns:
            Optional[RawTrafficRecord]: The raw record if found, else None.
        """
        pass

    @abstractmethod
    def get_provider_status(self) -> ProviderStatus:
        """Return a typed ProviderStatus operational report.

        Returns:
            ProviderStatus: Health, aggregate state, and diagnostic telemetry.
        """
        pass
