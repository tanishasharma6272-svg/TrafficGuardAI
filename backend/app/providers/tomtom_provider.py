"""Real-time TomTom Traffic Flow API data provider for TrafficGuard AI."""

import concurrent.futures
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation
from app.providers.base import AggregateState, ProviderStatus, RawTrafficRecord, TrafficProvider
from app.providers.tomtom_incident_provider import TomTomIncidentProvider

TOMTOM_FLOW_BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/12/json"

TOMTOM_PROVENANCE = {
    "traffic_speed": "TOMTOM_REALTIME",
    "free_flow_speed": "TOMTOM_REALTIME",
    "snapshot_timestamp": "TOMTOM_REALTIME",
    "raw_metadata": "TOMTOM_REALTIME",
    "current_incident_count": "TOMTOM_REALTIME",
    "current_accident_count": "TOMTOM_REALTIME",
    "current_jam_count": "TOMTOM_REALTIME",
    "current_road_closure_count": "TOMTOM_REALTIME",
    "current_roadworks_count": "TOMTOM_REALTIME",
    "current_broken_down_vehicle_count": "TOMTOM_REALTIME",
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
        max_workers: int = 5,
    ) -> None:
        """Initialize TomTom traffic provider.

        Args:
            api_key: Optional TomTom API key. If omitted, reads from TOMTOM_API_KEY env.
            timeout: HTTP request timeout in seconds. Defaults to 5.0s.
            db: Optional SQLAlchemy database session.
            max_workers: Thread pool concurrency for batch location fetching. Defaults to 5.
        """
        self.api_key: Optional[str] = api_key if api_key is not None else os.getenv("TOMTOM_API_KEY")
        self.timeout: float = timeout
        self._db: Optional[Session] = db
        self.max_workers: int = max_workers
        self.last_fetch_results: Dict[int, Dict[str, Any]] = {}
        self.last_fetch_timestamp: Optional[datetime] = None
        self.incident_provider = TomTomIncidentProvider(
            api_key=self.api_key,
            timeout=self.timeout,
            db=self._db,
        )

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
        incident_telemetry: Optional[Dict[str, Any]] = None,
    ) -> RawTrafficRecord:
        """Construct a RawTrafficRecord preserving non-mutated TomTom telemetry and PostgreSQL context."""
        # Extract raw velocity values without clamping or rounding
        raw_current_speed = flow_data.get("currentSpeed")
        raw_free_flow_speed = flow_data.get("freeFlowSpeed")

        if raw_current_speed is None or raw_free_flow_speed is None:
            raise ProviderFetchError(
                f"Missing speed telemetry in TomTom response for '{db_loc.name}' (ID: {db_loc.id})"
            )

        inc_info = incident_telemetry or {
            "current_incident_count": 0,
            "current_accident_count": 0,
            "current_jam_count": 0,
            "current_road_closure_count": 0,
            "current_roadworks_count": 0,
            "current_broken_down_vehicle_count": 0,
            "incident_provider": "TomTomIncidentDetailsV5",
            "incident_provider_state": "UNCONFIGURED" if not self.is_configured() else "LIVE",
            "incident_snapshot_timestamp": snapshot_time.isoformat(),
            "incidents": [],
        }

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
                "current_incident_count": inc_info.get("current_incident_count", 0),
                "current_accident_count": inc_info.get("current_accident_count", 0),
                "current_jam_count": inc_info.get("current_jam_count", 0),
                "current_road_closure_count": inc_info.get("current_road_closure_count", 0),
                "current_roadworks_count": inc_info.get("current_roadworks_count", 0),
                "current_broken_down_vehicle_count": inc_info.get("current_broken_down_vehicle_count", 0),
                "incident_provider": inc_info.get("incident_provider", "TomTomIncidentDetailsV5"),
                "incident_provider_state": inc_info.get("incident_provider_state", "UNCONFIGURED"),
                "incident_snapshot_timestamp": inc_info.get("incident_snapshot_timestamp"),
                "nearby_incidents": inc_info.get("incidents", []),
                "provenance": dict(TOMTOM_PROVENANCE),
            },
        )

    def get_traffic_records(self) -> List[RawTrafficRecord]:
        """Fetch real-time traffic observations across monitored PostgreSQL locations concurrently.

        Iterates over all monitored locations using bounded thread-pool concurrency for flow telemetry
        and exactly ONE network-wide bounding box request for incident telemetry.
        Successful locations emit RawTrafficRecord with provider_mode='LIVE'. Failed
        locations are tracked individually without silent fallback to mock data.
        Results are returned deterministically sorted by location_id.

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

        try:
            db_locations = session.query(DBLocation).order_by(DBLocation.id).all()
        finally:
            if is_internal_session:
                session.close()

        if not db_locations:
            return []

        # Single network-wide bbox incident request
        incident_telemetry_map: Dict[int, Dict[str, Any]] = {}
        try:
            inc_snapshot = self.incident_provider.get_incident_snapshot(locations=db_locations)
            incident_telemetry_map = self.incident_provider.associate_incidents_to_locations(
                snapshot=inc_snapshot, locations=db_locations
            )
        except Exception as inc_err:
            for loc in db_locations:
                incident_telemetry_map[loc.id] = {
                    "current_incident_count": 0,
                    "current_accident_count": 0,
                    "current_jam_count": 0,
                    "current_road_closure_count": 0,
                    "current_roadworks_count": 0,
                    "current_broken_down_vehicle_count": 0,
                    "incident_provider": "TomTomIncidentDetailsV5",
                    "incident_provider_state": "ERROR",
                    "incident_snapshot_timestamp": snapshot_time.isoformat(),
                    "incidents": [],
                    "error_message": str(inc_err),
                }

        def _fetch_single_location(loc: DBLocation) -> Tuple[int, Optional[RawTrafficRecord], Dict[str, Any]]:
            try:
                flow_data = self.fetch_flow_segment_raw(loc.latitude, loc.longitude)
                loc_incident_info = incident_telemetry_map.get(loc.id)
                record = self._db_and_flow_to_raw_record(loc, flow_data, snapshot_time, loc_incident_info)
                fetch_result = {
                    "status": "SUCCESS",
                    "location_name": loc.name,
                    "current_speed": record.traffic_speed,
                    "free_flow_speed": record.free_flow_speed,
                    "timestamp": snapshot_time.isoformat(),
                }
                return (loc.id, record, fetch_result)
            except Exception as e:
                fetch_result = {
                    "status": "ERROR",
                    "location_name": loc.name,
                    "error_message": str(e),
                    "timestamp": snapshot_time.isoformat(),
                }
                return (loc.id, None, fetch_result)

        records: List[RawTrafficRecord] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_fetch_single_location, loc) for loc in db_locations]
            for future in concurrent.futures.as_completed(futures):
                loc_id, record, fetch_result = future.result()
                self.last_fetch_results[loc_id] = fetch_result
                if record is not None:
                    records.append(record)

        # Guarantee deterministic ordering by location_id
        records.sort(key=lambda r: r.location_id)

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

            # Fetch incident telemetry for all locations in the network
            inc_info = None
            try:
                all_locs = session.query(DBLocation).order_by(DBLocation.id).all()
                inc_snapshot = self.incident_provider.get_incident_snapshot(locations=all_locs)
                matched_map = self.incident_provider.associate_incidents_to_locations(
                    snapshot=inc_snapshot, locations=[db_loc]
                )
                inc_info = matched_map.get(location_id)
            except Exception as inc_err:
                inc_info = {
                    "current_incident_count": 0,
                    "current_accident_count": 0,
                    "current_jam_count": 0,
                    "current_road_closure_count": 0,
                    "current_roadworks_count": 0,
                    "current_broken_down_vehicle_count": 0,
                    "incident_provider": "TomTomIncidentDetailsV5",
                    "incident_provider_state": "ERROR",
                    "incident_snapshot_timestamp": snapshot_time.isoformat(),
                    "incidents": [],
                    "error_message": str(inc_err),
                }

            flow_data = self.fetch_flow_segment_raw(db_loc.latitude, db_loc.longitude)
            record = self._db_and_flow_to_raw_record(db_loc, flow_data, snapshot_time, inc_info)
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
