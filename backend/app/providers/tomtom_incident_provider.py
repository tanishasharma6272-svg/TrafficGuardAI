"""Real-time TomTom Incident Details v5 data provider for TrafficGuard AI."""

from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Dict, List, Literal, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation

TOMTOM_INCIDENT_BASE_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"

# Explicit TomTom iconCategory code definitions per TomTom Incident Details v5 specification
TOMTOM_ICON_CATEGORY_MAP: Dict[int, str] = {
    0: "Unknown",
    1: "Accident",
    2: "Fog",
    3: "Dangerous Conditions",
    4: "Rain",
    5: "Ice",
    6: "Jam",
    7: "Lane Closed",
    8: "Road Closed",
    9: "Road Works",
    10: "Wind",
    11: "Flooding",
    12: "Broken Down Vehicle",
    13: "Cluster",
    14: "Special Event",
}

IncidentProviderState = Literal["LIVE", "ERROR", "UNCONFIGURED"]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two WGS84 coordinate points in kilometers.

    Formula:
        Δφ = radians(lat2 - lat1)
        Δλ = radians(lon2 - lon1)
        a = sin²(Δφ/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin²(Δλ/2)
        c = 2 * atan2(√a, √(1-a))
        d = R * c  (R = 6371.0 km)
    """
    earth_radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    a = math.sin(d_lat / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_km * c


def calculate_min_distance_to_geometry(
    target_lat: float,
    target_lon: float,
    geometry_type: str,
    coordinates: Any,
) -> float:
    """Compute the minimum geodesic distance (in km) from a target point to an incident geometry.

    Args:
        target_lat: Target location latitude.
        target_lon: Target location longitude.
        geometry_type: GeoJSON geometry type ('Point', 'LineString', 'MultiLineString').
        coordinates: Raw coordinates structure matching the geometry type.

    Returns:
        float: Minimum distance in kilometers.
    """
    if not coordinates:
        return float("inf")

    geom_upper = str(geometry_type).upper()

    if geom_upper == "POINT":
        # Point coordinates: [lon, lat]
        if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
            return haversine_distance_km(target_lat, target_lon, coordinates[1], coordinates[0])
        return float("inf")

    elif geom_upper == "LINESTRING":
        # LineString coordinates: [[lon1, lat1], [lon2, lat2], ...]
        min_d = float("inf")
        for vertex in coordinates:
            if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                d = haversine_distance_km(target_lat, target_lon, vertex[1], vertex[0])
                if d < min_d:
                    min_d = d
        return min_d

    elif geom_upper == "MULTILINESTRING":
        # MultiLineString coordinates: [[[lon1, lat1], ...], ...]
        min_d = float("inf")
        for line in coordinates:
            if isinstance(line, (list, tuple)):
                for vertex in line:
                    if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                        d = haversine_distance_km(target_lat, target_lon, vertex[1], vertex[0])
                        if d < min_d:
                            min_d = d
        return min_d

    else:
        # Fallback: check if coordinates is a list of [lon, lat] pairs
        min_d = float("inf")
        if isinstance(coordinates, list):
            for item in coordinates:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], (int, float)):
                    d = haversine_distance_km(target_lat, target_lon, item[1], item[0])
                    if d < min_d:
                        min_d = d
        return min_d


class IncidentRecord(BaseModel):
    """Structured, non-mutated observation of a single TomTom Traffic Incident."""

    incident_id: str = Field(..., description="Unique TomTom incident identifier")
    category: str = Field(default="Unknown", description="Human-readable category description")
    icon_category: int = Field(default=0, ge=0, description="Numeric TomTom icon category code (0-14)")
    magnitude_of_delay: Optional[int] = Field(default=None, description="TomTom magnitude of delay code")
    geometry_type: str = Field(default="Point", description="GeoJSON geometry type")
    coordinates: Any = Field(default_factory=list, description="Raw coordinate structure [lon, lat]")
    representative_latitude: Optional[float] = Field(default=None, description="Representative WGS84 latitude")
    representative_longitude: Optional[float] = Field(default=None, description="Representative WGS84 longitude")
    length_meters: Optional[float] = Field(default=None, description="Incident segment length in meters")
    delay_seconds: Optional[float] = Field(default=None, description="Reported delay in seconds")
    start_time: Optional[str] = Field(default=None, description="Incident start timestamp ISO")
    end_time: Optional[str] = Field(default=None, description="Incident estimated end timestamp ISO")
    from_location: Optional[str] = Field(default=None, description="Origin intersection/road description")
    to_location: Optional[str] = Field(default=None, description="Destination intersection/road description")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Incident events/descriptions")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw provider metadata")


class IncidentSnapshot(BaseModel):
    """Aggregate snapshot of real-time traffic incidents across the monitored network."""

    snapshot_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC datetime when TrafficGuard AI retrieved this incident snapshot",
    )
    status: IncidentProviderState = Field(
        default="UNCONFIGURED",
        description="Operational state: 'LIVE', 'ERROR', or 'UNCONFIGURED'",
    )
    incident_count: int = Field(default=0, ge=0, description="Total active incidents within monitored bounds")
    accident_count: int = Field(default=0, ge=0, description="Active accident incidents (iconCategory=1)")
    jam_count: int = Field(default=0, ge=0, description="Active traffic jam incidents (iconCategory=6)")
    road_closure_count: int = Field(default=0, ge=0, description="Active road closure incidents (iconCategory=8)")
    roadworks_count: int = Field(default=0, ge=0, description="Active roadworks incidents (iconCategory=9)")
    broken_down_vehicle_count: int = Field(
        default=0, ge=0, description="Active broken down vehicle incidents (iconCategory=12)"
    )
    incidents: List[IncidentRecord] = Field(default_factory=list, description="List of individual incident records")
    bbox: Optional[str] = Field(default=None, description="Query bounding box: minLon,minLat,maxLon,maxLat")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error details if status is ERROR")


class TomTomIncidentProvider:
    """Real-time incident provider querying TomTom Incident Details v5 API over network bounding box.

    EXECUTION MODEL:
    ----------------
    Executes exactly ONE network-wide bounding box request per snapshot batch.
    Associates returned incident geometry to monitored intersections using a transparent
    Haversine distance matching rule (default matching radius: 1.0 km).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        db: Optional[Session] = None,
        padding_km: float = 2.0,
        matching_radius_km: float = 1.0,
    ) -> None:
        """Initialize TomTom incident provider.

        Args:
            api_key: Optional TomTom API key. If omitted, reads from TOMTOM_API_KEY env.
            timeout: HTTP request timeout in seconds. Defaults to 5.0s.
            db: Optional SQLAlchemy database session.
            padding_km: Small geographic bounding box margin in kilometers. Defaults to 2.0 km.
            matching_radius_km: Distance threshold to associate incident with monitored location. Defaults to 1.0 km.
        """
        self.api_key: Optional[str] = api_key if api_key is not None else os.getenv("TOMTOM_API_KEY")
        self.timeout: float = timeout
        self._db: Optional[Session] = db
        self.padding_km: float = padding_km
        self.matching_radius_km: float = matching_radius_km
        self.last_snapshot: Optional[IncidentSnapshot] = None

    def is_configured(self) -> bool:
        """Check if the TomTom API key is present and non-empty."""
        return bool(self.api_key and self.api_key.strip())

    def _get_session(self) -> Session:
        """Return the injected database session or create a new SessionLocal."""
        if self._db is not None:
            return self._db
        return SessionLocal()

    def calculate_bounding_box(self, locations: List[Any]) -> str:
        """Dynamically compute padded network bounding box in TomTom order: minLon,minLat,maxLon,maxLat.

        Formula:
            min_lat = min(l.latitude), max_lat = max(l.latitude)
            min_lon = min(l.longitude), max_lon = max(l.longitude)
            mid_lat = (min_lat + max_lat) / 2
            pad_lat = padding_km / 111.0
            pad_lon = padding_km / (111.0 * cos(radians(mid_lat)))
            bbox = f"{min_lon - pad_lon:.6f},{min_lat - pad_lat:.6f},{max_lon + pad_lon:.6f},{max_lat + pad_lat:.6f}"

        Args:
            locations: List of location objects with .latitude and .longitude attributes.

        Returns:
            str: Bounding box string formatted as 'minLon,minLat,maxLon,maxLat'.
        """
        if not locations:
            raise ValueError("Cannot calculate bounding box for an empty location list.")

        lats = [float(loc.latitude) for loc in locations]
        lons = [float(loc.longitude) for loc in locations]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        mid_lat = (min_lat + max_lat) / 2.0

        # Convert padding_km to approximate degrees of latitude and longitude
        pad_lat = self.padding_km / 111.0
        cos_mid = math.cos(math.radians(mid_lat))
        pad_lon = self.padding_km / (111.0 * cos_mid) if cos_mid > 1e-6 else pad_lat

        min_lat_padded = min_lat - pad_lat
        max_lat_padded = max_lat + pad_lat
        min_lon_padded = min_lon - pad_lon
        max_lon_padded = max_lon + pad_lon

        # TomTom expects: minLon,minLat,maxLon,maxLat
        return f"{min_lon_padded:.6f},{min_lat_padded:.6f},{max_lon_padded:.6f},{max_lat_padded:.6f}"

    def fetch_incidents_raw(self, bbox: str) -> Dict[str, Any]:
        """Execute HTTP GET to TomTom Incident Details API v5 using standard urllib.

        Args:
            bbox: Bounding box string formatted as 'minLon,minLat,maxLon,maxLat'.

        Returns:
            Dict[str, Any]: Parsed JSON response payload.

        Raises:
            RuntimeError: On missing credentials, HTTP failure, timeout, or malformed JSON.
        """
        if not self.is_configured():
            raise RuntimeError("TOMTOM_API_KEY is not configured in environment.")

        params = {
            "key": self.api_key,
            "bbox": bbox,
            "language": "en-GB",
            "timeValidityFilter": "present",
            "fields": (
                "{incidents{type,geometry{type,coordinates},"
                "properties{id,iconCategory,magnitudeOfDelay,events{description,code},"
                "startTime,endTime,from,to,length,delay}}}"
            ),
        }
        url = f"{TOMTOM_INCIDENT_BASE_URL}?{urllib.parse.urlencode(params)}"
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
                    raise RuntimeError(f"TomTom Incident API returned non-200 status code: {status_code}")
                raw_body = response.read().decode("utf-8")
                return json.loads(raw_body)
        except urllib.error.HTTPError as http_err:
            raise RuntimeError(f"TomTom Incident API HTTP error: {http_err.code} - {http_err.reason}") from http_err
        except urllib.error.URLError as url_err:
            raise RuntimeError(f"TomTom Incident API network connectivity error: {url_err.reason}") from url_err
        except json.JSONDecodeError as json_err:
            raise RuntimeError(f"Failed to parse TomTom Incident API response as JSON: {json_err}") from json_err
        except Exception as e:
            raise RuntimeError(f"Unexpected error communicating with TomTom Incident API: {e}") from e

    def _parse_incident_feature(self, feature: Dict[str, Any]) -> Optional[IncidentRecord]:
        """Convert a single TomTom GeoJSON Feature into an IncidentRecord."""
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        incident_id = props.get("id")
        if not incident_id:
            return None

        icon_cat = int(props.get("iconCategory", 0))
        cat_name = TOMTOM_ICON_CATEGORY_MAP.get(icon_cat, "Unknown")
        geom_type = geom.get("type", "Point")
        coords = geom.get("coordinates", [])

        # Derive representative lat/lon for point or midpoint/first vertex
        rep_lat: Optional[float] = None
        rep_lon: Optional[float] = None

        if geom_type.upper() == "POINT" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            rep_lon = float(coords[0])
            rep_lat = float(coords[1])
        elif geom_type.upper() in ("LINESTRING", "MULTILINESTRING") and coords:
            # First coordinate pair as representative reference
            first_pt = coords[0] if geom_type.upper() == "LINESTRING" else (coords[0][0] if coords[0] else None)
            if first_pt and len(first_pt) >= 2:
                rep_lon = float(first_pt[0])
                rep_lat = float(first_pt[1])

        events = props.get("events", [])

        return IncidentRecord(
            incident_id=str(incident_id),
            category=cat_name,
            icon_category=icon_cat,
            magnitude_of_delay=props.get("magnitudeOfDelay"),
            geometry_type=geom_type,
            coordinates=coords,
            representative_latitude=rep_lat,
            representative_longitude=rep_lon,
            length_meters=props.get("length"),
            delay_seconds=props.get("delay"),
            start_time=props.get("startTime"),
            end_time=props.get("endTime"),
            from_location=props.get("from"),
            to_location=props.get("to"),
            events=events if isinstance(events, list) else [],
            raw_metadata={
                "properties": props,
                "geometry_type": geom_type,
            },
        )

    def get_incident_snapshot(self, locations: Optional[List[Any]] = None) -> IncidentSnapshot:
        """Fetch network-wide incident snapshot using ONE dynamic bounding box request.

        Args:
            locations: Optional list of location objects. If None, loads from DB.

        Returns:
            IncidentSnapshot: Aggregate snapshot with status 'LIVE', 'ERROR', or 'UNCONFIGURED'.
        """
        snapshot_time = datetime.now(timezone.utc)

        if not self.is_configured():
            snapshot = IncidentSnapshot(
                snapshot_timestamp=snapshot_time,
                status="UNCONFIGURED",
                incident_count=0,
                accident_count=0,
                jam_count=0,
                road_closure_count=0,
                roadworks_count=0,
                broken_down_vehicle_count=0,
                incidents=[],
                error_message="TOMTOM_API_KEY is not configured in environment.",
            )
            self.last_snapshot = snapshot
            return snapshot

        # Resolve location objects
        if locations is None:
            is_internal_session = self._db is None
            session = self._get_session()
            try:
                db_locs = session.query(DBLocation).order_by(DBLocation.id).all()
            finally:
                if is_internal_session:
                    session.close()
            locations = db_locs

        if not locations:
            snapshot = IncidentSnapshot(
                snapshot_timestamp=snapshot_time,
                status="LIVE",
                incident_count=0,
                accident_count=0,
                jam_count=0,
                road_closure_count=0,
                roadworks_count=0,
                broken_down_vehicle_count=0,
                incidents=[],
            )
            self.last_snapshot = snapshot
            return snapshot

        bbox = self.calculate_bounding_box(locations)

        try:
            raw_data = self.fetch_incidents_raw(bbox)
            raw_features = raw_data.get("incidents", [])

            parsed_incidents: List[IncidentRecord] = []
            seen_ids = set()

            for feat in raw_features:
                inc_rec = self._parse_incident_feature(feat)
                if inc_rec and inc_rec.incident_id not in seen_ids:
                    seen_ids.add(inc_rec.incident_id)
                    parsed_incidents.append(inc_rec)

            accidents = sum(1 for inc in parsed_incidents if inc.icon_category == 1)
            jams = sum(1 for inc in parsed_incidents if inc.icon_category == 6)
            closures = sum(1 for inc in parsed_incidents if inc.icon_category == 8)
            roadworks = sum(1 for inc in parsed_incidents if inc.icon_category == 9)
            breakdowns = sum(1 for inc in parsed_incidents if inc.icon_category == 12)

            snapshot = IncidentSnapshot(
                snapshot_timestamp=snapshot_time,
                status="LIVE",
                incident_count=len(parsed_incidents),
                accident_count=accidents,
                jam_count=jams,
                road_closure_count=closures,
                roadworks_count=roadworks,
                broken_down_vehicle_count=breakdowns,
                incidents=parsed_incidents,
                bbox=bbox,
            )
            self.last_snapshot = snapshot
            return snapshot

        except Exception as err:
            snapshot = IncidentSnapshot(
                snapshot_timestamp=snapshot_time,
                status="ERROR",
                incident_count=0,
                accident_count=0,
                jam_count=0,
                road_closure_count=0,
                roadworks_count=0,
                broken_down_vehicle_count=0,
                incidents=[],
                bbox=bbox,
                error_message=str(err),
            )
            self.last_snapshot = snapshot
            return snapshot

    def associate_incidents_to_locations(
        self,
        snapshot: IncidentSnapshot,
        locations: List[Any],
        matching_radius_km: Optional[float] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Spatially associate network incidents with monitored locations within matching radius.

        A single incident may be associated with multiple monitored locations if it falls
        within the radius of each, but is never duplicated within the same location.

        Args:
            snapshot: Ingested network-wide IncidentSnapshot.
            locations: List of location objects with .id, .latitude, and .longitude.
            matching_radius_km: Optional distance override in km (defaults to instance matching_radius_km).

        Returns:
            Dict[int, Dict[str, Any]]: Map of location_id -> location incident telemetry dictionary.
        """
        radius = matching_radius_km if matching_radius_km is not None else self.matching_radius_km
        results: Dict[int, Dict[str, Any]] = {}

        for loc in locations:
            loc_id = int(loc.id)
            loc_lat = float(loc.latitude)
            loc_lon = float(loc.longitude)

            matched_incidents: List[IncidentRecord] = []
            seen_for_loc = set()

            if snapshot.status == "LIVE":
                for inc in snapshot.incidents:
                    if inc.incident_id in seen_for_loc:
                        continue
                    d = calculate_min_distance_to_geometry(
                        loc_lat, loc_lon, inc.geometry_type, inc.coordinates
                    )
                    if d <= radius:
                        seen_for_loc.add(inc.incident_id)
                        matched_incidents.append(inc)

            loc_accidents = sum(1 for inc in matched_incidents if inc.icon_category == 1)
            loc_jams = sum(1 for inc in matched_incidents if inc.icon_category == 6)
            loc_closures = sum(1 for inc in matched_incidents if inc.icon_category == 8)
            loc_roadworks = sum(1 for inc in matched_incidents if inc.icon_category == 9)
            loc_breakdowns = sum(1 for inc in matched_incidents if inc.icon_category == 12)

            results[loc_id] = {
                "current_incident_count": len(matched_incidents),
                "current_accident_count": loc_accidents,
                "current_jam_count": loc_jams,
                "current_road_closure_count": loc_closures,
                "current_roadworks_count": loc_roadworks,
                "current_broken_down_vehicle_count": loc_breakdowns,
                "incident_provider": "TomTomIncidentDetailsV5",
                "incident_provider_state": snapshot.status,
                "incident_snapshot_timestamp": snapshot.snapshot_timestamp.isoformat(),
                "matching_radius_km": radius,
                "incidents": [inc.model_dump() for inc in matched_incidents],
            }

        return results
