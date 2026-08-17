"""Real-time TomTom Traffic Flow API data provider for TrafficGuard AI."""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation
from app.providers.base import AggregateState, ProviderStatus, RawTrafficRecord, TrafficProvider

TOMTOM_FLOW_BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/12/json"

TOMTOM_PROVENANCE = {
    "traffic_speed": "TOMTOM_REALTIME",
    "free_flow_speed": "TOMTOM_REALTIME",
    "snapshot_timestamp": "TOMTOM_REALTIME",
    "raw_metadata": "TOMTOM_REALTIME",
    "location_id": "POSTGRESQL_CONTEXT",
    "name": "POSTGRESQL_CONTEXT",
    "latitude": "POSTGRESQL_CONTEXT",
    "longitude": "POSTGRESQL_CONTEXT",
    "coordinate_source": "POSTGRESQL_CONTEXT",
    "traffic_volume": "POSTGRESQL_CONTEXT",
    "incident_frequency": "POSTGRESQL_CONTEXT",
    "accident_history": "POSTGRESQL_CONTEXT",
    "road_factor": "POSTGRESQL_CONTEXT",
    "population_factor": "POSTGRESQL_CONTEXT",
    "police_officers": "POSTGRESQL_CONTEXT",
}


class ProviderConfigurationError(Exception):
    """Raised when provider configuration or API credentials are missing or invalid."""
    pass


class ProviderFetchError(Exception):
    """Raised when provider network request fails or returns an unrecoverable error."""

    def __init__(self, message: str, status_code: Optional[int] = None, location_id: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.location_id = location_id


class TomTomTrafficProvider(TrafficProvider):
    """Real-time traffic provider interfacing with the TomTom Traffic Flow Segment API.

    Ingests real-time speed and free-flow velocity telemetry directly from TomTom
    while fusing static and historical contextual attributes (accident history, road factor,
    population density, volume) from the local PostgreSQL store.

    TELEMETRY PRESERVATION POLICY:
    ------------------------------
    Raw TomTom currentSpeed and freeFlowSpeed values are passed through without
    clamping, rounding, or mutation. The canonical normalization layer is responsible
    for validating physical constraints and rejecting invalid observations.

    FAILURE & STATE TRACKING:
    -------------------------
    Per-location status is maintained. If a location request fails, it is not
    silently converted to DEMO data. The provider reports an explicit aggregate
    state: LIVE (100% success), PARTIAL (mixed success/failure), ERROR (0% success),
    or UNCONFIGURED (missing API credentials).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        db: Optional[Session] = None,
    ) -> None:
        """Initialize TomTom traffic provider.

        Args:
            api_key: Optional TomTom API key. If omitted, reads from TOMTOM_API_KEY env.
            timeout: HTTP request timeout in seconds. Defaults to 5.0s.
            db: Optional SQLAlchemy database session.
        """
        self.api_key: Optional[str] = api_key if api_key is not None else os.getenv("TOMTOM_API_KEY")
        self.timeout: float = timeout
        self._db: Optional[Session] = db
        self.last_fetch_results: Dict[int, Dict[str, Any]] = {}
        self.last_fetch_timestamp: Optional[datetime] = None

    @property
    def provider_mode(self) -> str:
        """Return the operational mode. Returns 'LIVE' when configured, else 'UNCONFIGURED'."""
        return "LIVE" if self.is_configured() else "UNCONFIGURED"

    def is_configured(self) -> bool:
        """Check if the TomTom API key is present and non-empty."""
        return bool(self.api_key and self.api_key.strip())

    def _get_session(self) -> Session:
        """Return the injected database session or create a new SessionLocal."""
        if self._db is not None:
            return self._db
        return SessionLocal()

    def get_aggregate_state(self) -> AggregateState:
        """Derive the aggregate operational state.

        Returns:
            AggregateState: 'LIVE', 'PARTIAL', 'ERROR', or 'UNCONFIGURED'.
        """
        if not self.is_configured():
            return "UNCONFIGURED"
        if not self.last_fetch_results:
            return "UNCONFIGURED"

        total = len(self.last_fetch_results)
        successes = sum(1 for r in self.last_fetch_results.values() if r.get("status") == "SUCCESS")

        if successes == total and total > 0:
            return "LIVE"
        elif successes > 0:
            return "PARTIAL"
        else:
            return "ERROR"

    def get_provider_status(self) -> ProviderStatus:
        """Return typed ProviderStatus operational report for TomTomTrafficProvider."""
        state = self.get_aggregate_state()
        errors = {
            loc_id: str(res["error_message"])
            for loc_id, res in self.last_fetch_results.items()
            if res.get("status") == "ERROR" and "error_message" in res
        }
        total = len(self.last_fetch_results)
        successes = sum(1 for r in self.last_fetch_results.values() if r.get("status") == "SUCCESS")
        failures = sum(1 for r in self.last_fetch_results.values() if r.get("status") == "ERROR")

        return ProviderStatus(
            provider="TomTomTrafficProvider",
            aggregate_state=state,
            successful_count=successes,
            failed_count=failures,
            total_locations=total,
            last_fetch_timestamp=self.last_fetch_timestamp,
            is_configured=self.is_configured(),
            per_location_errors=errors,
        )

    def fetch_flow_segment_raw(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Execute HTTP GET to TomTom Flow Segment Data API.

        Args:
            latitude: Latitude coordinate.
            longitude: Longitude coordinate.

        Returns:
            Dict[str, Any]: Parsed JSON flowSegmentData dictionary.

        Raises:
            ProviderConfigurationError: If API key is missing.
            ProviderFetchError: On network, HTTP error, or malformed payload.
        """
        if not self.is_configured():
            raise ProviderConfigurationError(
                "LIVE provider unavailable because TOMTOM_API_KEY is not configured in environment."
            )

        params = {
            "key": self.api_key,
            "point": f"{latitude},{longitude}",
            "unit": "KMPH",
        }
        url = f"{TOMTOM_FLOW_BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url=url,
            headers={
                "User-Agent": "TrafficGuardAI/1.0",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status_code = response.status
                if status_code != 200:
                    raise ProviderFetchError(
                        f"TomTom API returned non-200 status code: {status_code}",
                        status_code=status_code,
                    )
                raw_body = response.read().decode("utf-8")
                payload = json.loads(raw_body)
        except urllib.error.HTTPError as http_err:
            raise ProviderFetchError(
                f"TomTom API HTTP error: {http_err.code} - {http_err.reason}",
                status_code=http_err.code,
            ) from http_err
        except urllib.error.URLError as url_err:
            raise ProviderFetchError(
                f"TomTom API network connectivity error: {url_err.reason}"
            ) from url_err
        except json.JSONDecodeError as json_err:
            raise ProviderFetchError(
                f"Failed to parse TomTom API response as JSON: {json_err}"
            ) from json_err
        except Exception as e:
            raise ProviderFetchError(f"Unexpected error communicating with TomTom API: {e}") from e

        if "flowSegmentData" not in payload:
            raise ProviderFetchError(
                "Malformed TomTom response: 'flowSegmentData' object missing from response."
            )

        return payload["flowSegmentData"]

    def _db_and_flow_to_raw_record(
        self,
        db_loc: DBLocation,
        flow_data: Dict[str, Any],
        snapshot_time: datetime,
    ) -> RawTrafficRecord:
        """Construct a RawTrafficRecord preserving non-mutated TomTom telemetry and PostgreSQL context."""
        # Extract raw velocity values without clamping or rounding
        raw_current_speed = flow_data.get("currentSpeed")
        raw_free_flow_speed = flow_data.get("freeFlowSpeed")

        if raw_current_speed is None or raw_free_flow_speed is None:
            raise ProviderFetchError(
                f"Missing speed telemetry in TomTom response for '{db_loc.name}' (ID: {db_loc.id})"
            )

        return RawTrafficRecord(
            location_id=db_loc.id,
            name=db_loc.name,
            latitude=db_loc.latitude,
            longitude=db_loc.longitude,
            coordinate_source=db_loc.coordinate_source,
            traffic_speed=float(raw_current_speed),
            free_flow_speed=float(raw_free_flow_speed),
            traffic_volume=db_loc.traffic_volume,
            incident_frequency=db_loc.incident_frequency,
            accident_history=db_loc.accident_history,
            road_factor=db_loc.road_factor,
            population_factor=db_loc.population_factor,
            police_officers=db_loc.police_officers,
            provider_mode="LIVE",
            snapshot_timestamp=snapshot_time,
            raw_metadata={
                "provider": "TomTom Traffic Flow API v4",
                "frc": flow_data.get("frc"),
                "confidence": flow_data.get("confidence"),
                "current_travel_time": flow_data.get("currentTravelTime"),
                "free_flow_travel_time": flow_data.get("freeFlowTravelTime"),
                "road_closure": flow_data.get("roadClosure"),
                "source_coordinates": {
                    "lat": db_loc.latitude,
                    "lon": db_loc.longitude,
                },
                "provenance": dict(TOMTOM_PROVENANCE),
            },
        )

    def get_traffic_records(self) -> List[RawTrafficRecord]:
        """Fetch real-time traffic observations across monitored PostgreSQL locations.

        Iterates over all monitored locations in the database. Successful locations
        emit RawTrafficRecord with provider_mode='LIVE'. Failed locations are tracked
        individually without silent fallback to mock data.

        Returns:
            List[RawTrafficRecord]: List of successfully fetched live records.

        Raises:
            ProviderConfigurationError: If TOMTOM_API_KEY is unconfigured.
            ProviderFetchError: If 100% of location requests fail.
        """
        if not self.is_configured():
            raise ProviderConfigurationError(
                "LIVE provider unavailable because TOMTOM_API_KEY is not configured in environment."
            )

        is_internal_session = self._db is None
        session = self._get_session()
        snapshot_time = datetime.now(timezone.utc)
        self.last_fetch_timestamp = snapshot_time
        self.last_fetch_results = {}

        records: List[RawTrafficRecord] = []

        try:
            db_locations = session.query(DBLocation).order_by(DBLocation.id).all()

            for loc in db_locations:
                try:
                    flow_data = self.fetch_flow_segment_raw(loc.latitude, loc.longitude)
                    record = self._db_and_flow_to_raw_record(loc, flow_data, snapshot_time)
                    records.append(record)
                    self.last_fetch_results[loc.id] = {
                        "status": "SUCCESS",
                        "location_name": loc.name,
                        "current_speed": record.traffic_speed,
                        "free_flow_speed": record.free_flow_speed,
                        "timestamp": snapshot_time.isoformat(),
                    }
                except Exception as e:
                    self.last_fetch_results[loc.id] = {
                        "status": "ERROR",
                        "location_name": loc.name,
                        "error_message": str(e),
                        "timestamp": snapshot_time.isoformat(),
                    }

            if not records and db_locations:
                # 100% failure rate
                first_err = next(
                    (r.get("error_message") for r in self.last_fetch_results.values() if "error_message" in r),
                    "Unknown error",
                )
                raise ProviderFetchError(
                    f"All {len(db_locations)} TomTom live location requests failed. First error: {first_err}"
                )

            return records

        finally:
            if is_internal_session:
                session.close()

    def get_location_traffic_record(self, location_id: int) -> Optional[RawTrafficRecord]:
        """Fetch live traffic observation for a single location by ID.

        Args:
            location_id: Monitored location identifier.

        Returns:
            Optional[RawTrafficRecord]: Live raw traffic record if successful, else None.

        Raises:
            ProviderConfigurationError: If TOMTOM_API_KEY is unconfigured.
            ProviderFetchError: On API communication or parsing failure.
        """
        if not self.is_configured():
            raise ProviderConfigurationError(
                "LIVE provider unavailable because TOMTOM_API_KEY is not configured in environment."
            )

        is_internal_session = self._db is None
        session = self._get_session()
        snapshot_time = datetime.now(timezone.utc)
        db_loc: Optional[DBLocation] = None

        try:
            db_loc = session.query(DBLocation).filter(DBLocation.id == location_id).first()
            if not db_loc:
                return None

            flow_data = self.fetch_flow_segment_raw(db_loc.latitude, db_loc.longitude)
            record = self._db_and_flow_to_raw_record(db_loc, flow_data, snapshot_time)
            self.last_fetch_results[location_id] = {
                "status": "SUCCESS",
                "location_name": db_loc.name,
                "current_speed": record.traffic_speed,
                "free_flow_speed": record.free_flow_speed,
                "timestamp": snapshot_time.isoformat(),
            }
            return record
        except Exception as e:
            loc_name = db_loc.name if db_loc is not None else f"ID-{location_id}"
            self.last_fetch_results[location_id] = {
                "status": "ERROR",
                "location_name": loc_name,
                "error_message": str(e),
                "timestamp": snapshot_time.isoformat(),
            }
            raise
        finally:
            if is_internal_session:
                session.close()
