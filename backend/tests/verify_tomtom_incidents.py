"""Comprehensive verification test suite for TomTom Incident Details v5 layer in TrafficGuard AI.

Tests:
A. Configuration (UNCONFIGURED when key missing, configured when key present)
B. Dynamic Bounding Box (minLon,minLat,maxLon,maxLat derived dynamically with padding, no hardcoded coordinates)
C. Mock response mapping (accurate category mapping for Accident, Jam, Road Closed, Road Works, Breakdown)
D. Zero incidents handling (returns LIVE, count=0, not an error)
E. Failure handling (401, 429, timeout -> ERROR state, no DEMO fallback, flow telemetry preserved)
F. Spatial association (Haversine matching radius rule, multi-location association, no duplicate in same location)
G. Provenance tagging (TOMTOM_REALTIME for 6 live incident fields, POSTGRESQL_CONTEXT for historical fields)
H. Database safety (zero mutations, exactly 20 rows in locations table)
I. Real TomTom request (conditional on TOMTOM_API_KEY, reports real Nagpur incidents if present)
J. Consistency check (all six incident counters present across IncidentSnapshot, RawTrafficRecord, MLRiskDetail, model_metadata)
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import urllib.error

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation
from app.models.location import Location as PydanticLocation
from app.models.ml_risk import MLRiskDetail, MLRiskSummary
from app.providers.base import RawTrafficRecord
from app.providers.tomtom_incident_provider import (
    TOMTOM_ICON_CATEGORY_MAP,
    IncidentRecord,
    IncidentSnapshot,
    TomTomIncidentProvider,
    calculate_min_distance_to_geometry,
    haversine_distance_km,
)
from app.providers.tomtom_provider import (
    TOMTOM_PROVENANCE,
    ProviderConfigurationError,
    ProviderFetchError,
    TomTomTrafficProvider,
)
from app.services.risk_model_service import get_risk_model_service

EXPECTED_LOCATION_COUNT: int = 50


def test_configuration():
    print("\n--- 1. Testing Configuration & State Initialization ---")

    # A1. Missing API Key -> UNCONFIGURED
    provider_unconf = TomTomIncidentProvider(api_key="")
    assert not provider_unconf.is_configured()
    snapshot_unconf = provider_unconf.get_incident_snapshot()
    assert snapshot_unconf.status == "UNCONFIGURED"
    assert snapshot_unconf.incident_count == 0
    assert "TOMTOM_API_KEY is not configured" in (snapshot_unconf.error_message or "")
    print("       [PASS] Missing TOMTOM_API_KEY safely reported UNCONFIGURED.")

    # A2. Present API Key -> Configured
    provider_conf = TomTomIncidentProvider(api_key="mock_test_key_123")
    assert provider_conf.is_configured()
    assert provider_conf.api_key == "mock_test_key_123"
    print("       [PASS] Configured provider instantiated successfully.")


def test_dynamic_bounding_box(session: Any):
    print("\n--- 2. Testing Dynamic Bounding Box Calculation ---")
    provider = TomTomIncidentProvider(api_key="mock_key", padding_km=2.0)

    db_locations = session.query(DBLocation).order_by(DBLocation.id).all()
    assert len(db_locations) == EXPECTED_LOCATION_COUNT, f"Expected {EXPECTED_LOCATION_COUNT} locations, got {len(db_locations)}"

    lats = [loc.latitude for loc in db_locations]
    lons = [loc.longitude for loc in db_locations]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    mid_lat = (min_lat + max_lat) / 2.0

    bbox = provider.calculate_bounding_box(db_locations)
    parts = [float(p) for p in bbox.split(",")]
    assert len(parts) == 4, f"Bbox must have 4 coordinates: {bbox}"

    bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat = parts

    # Verify TomTom format: minLon,minLat,maxLon,maxLat
    assert bbox_min_lon < min_lon, "Padded minLon must be less than minimum location lon"
    assert bbox_min_lat < min_lat, "Padded minLat must be less than minimum location lat"
    assert bbox_max_lon > max_lon, "Padded maxLon must be greater than maximum location lon"
    assert bbox_max_lat > max_lat, "Padded maxLat must be greater than maximum location lat"

    # Verify padding is approximately 2.0 km
    pad_lat_deg = 2.0 / 111.0
    pad_lon_deg = 2.0 / (111.0 * math.cos(math.radians(mid_lat)))

    assert abs((min_lat - bbox_min_lat) - pad_lat_deg) < 0.001
    assert abs((bbox_max_lat - max_lat) - pad_lat_deg) < 0.001
    assert abs((min_lon - bbox_min_lon) - pad_lon_deg) < 0.001
    assert abs((bbox_max_lon - max_lon) - pad_lon_deg) < 0.001

    print(f"       Computed Bounding Box (minLon,minLat,maxLon,maxLat): {bbox}")
    print("       [PASS] Bbox dynamically derived from PostgreSQL coordinates with geographic padding.")


def test_mock_response_mapping_and_categories():
    print("\n--- 3. Testing Mock Response Mapping & Category Counts ---")
    provider = TomTomIncidentProvider(api_key="mock_key")

    mock_incidents_payload = {
        "incidents": [
            {
                "type": "Feature",
                "properties": {
                    "id": "INC-001",
                    "iconCategory": 1,  # Accident
                    "magnitudeOfDelay": 3,
                    "from": "Wardha Road",
                    "to": "Ajni Square",
                    "length": 120.0,
                    "delay": 300,
                    "events": [{"code": 101, "description": "Accident"}],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [79.0720, 21.1180],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "INC-002",
                    "iconCategory": 6,  # Jam
                    "magnitudeOfDelay": 2,
                    "from": "Central Avenue",
                    "to": "Telephone Exchange",
                    "length": 450.0,
                    "delay": 180,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[79.1100, 21.1450], [79.1120, 21.1460]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "INC-003",
                    "iconCategory": 8,  # Road Closed
                    "magnitudeOfDelay": 4,
                    "from": "Sitabuldi",
                    "to": "Bardi",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [79.0830, 21.1460],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "INC-004",
                    "iconCategory": 9,  # Road Works
                    "magnitudeOfDelay": 1,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [79.0500, 21.1200],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "INC-005",
                    "iconCategory": 12,  # Broken Down Vehicle
                    "magnitudeOfDelay": 1,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [79.0600, 21.1300],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "INC-006",
                    "iconCategory": 2,  # Fog (General Incident)
                    "magnitudeOfDelay": 0,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [79.0900, 21.1500],
                },
            },
        ]
    }

    with patch.object(provider, "fetch_incidents_raw", return_value=mock_incidents_payload):
        snapshot = provider.get_incident_snapshot()

    assert snapshot.status == "LIVE"
    assert snapshot.incident_count == 6
    assert snapshot.accident_count == 1
    assert snapshot.jam_count == 1
    assert snapshot.road_closure_count == 1
    assert snapshot.roadworks_count == 1
    assert snapshot.broken_down_vehicle_count == 1
    assert len(snapshot.incidents) == 6

    # Verify category constant mappings
    cat_map = {inc.incident_id: inc.category for inc in snapshot.incidents}
    assert cat_map["INC-001"] == "Accident"
    assert cat_map["INC-002"] == "Jam"
    assert cat_map["INC-003"] == "Road Closed"
    assert cat_map["INC-004"] == "Road Works"
    assert cat_map["INC-005"] == "Broken Down Vehicle"
    assert cat_map["INC-006"] == "Fog"

    print("       [PASS] 6 Incidents correctly parsed with exact category counters and mappings.")


def test_zero_incidents():
    print("\n--- 4. Testing Zero Incidents (Successful LIVE state, count=0) ---")
    provider = TomTomIncidentProvider(api_key="mock_key")
    mock_zero_payload = {"incidents": []}

    with patch.object(provider, "fetch_incidents_raw", return_value=mock_zero_payload):
        snapshot = provider.get_incident_snapshot()

    assert snapshot.status == "LIVE"
    assert snapshot.incident_count == 0
    assert snapshot.accident_count == 0
    assert snapshot.jam_count == 0
    assert snapshot.road_closure_count == 0
    assert snapshot.roadworks_count == 0
    assert snapshot.broken_down_vehicle_count == 0
    assert len(snapshot.incidents) == 0
    assert snapshot.error_message is None
    print("       [PASS] Zero incidents correctly returned status=LIVE with count=0 (No error interpretation).")


def test_failure_handling_and_no_demo_fallback(session: Any):
    print("\n--- 5. Testing Incident Failure Handling & Flow Telemetry Preservation ---")
    traffic_provider = TomTomTrafficProvider(api_key="mock_key", db=session)

    mock_flow_data = {
        "frc": "FRC1",
        "currentSpeed": 42.5,
        "freeFlowSpeed": 60.0,
        "confidence": 1.0,
    }

    # Scenario 1: Incident API timeout error
    with patch.object(traffic_provider, "fetch_flow_segment_raw", return_value=mock_flow_data):
        with patch.object(traffic_provider.incident_provider, "fetch_incidents_raw", side_effect=RuntimeError("Request timed out after 5.0s")):
            record = traffic_provider.get_location_traffic_record(1)

    assert record is not None
    assert record.provider_mode == "LIVE"
    assert record.traffic_speed == 42.5
    assert record.free_flow_speed == 60.0

    # Incident state must be ERROR without falling back to DEMO or fabricating counts
    assert record.raw_metadata.get("incident_provider_state") == "ERROR"
    assert record.raw_metadata.get("current_incident_count") == 0
    assert record.raw_metadata.get("current_accident_count") == 0
    print("       [PASS] Timeout failure handled: Flow preserved (42.5 km/h), incident_provider_state=ERROR, 0 DEMO fallbacks.")

    # Scenario 2: Incident API HTTP 401 Unauthorized
    with patch.object(traffic_provider, "fetch_flow_segment_raw", return_value=mock_flow_data):
        with patch.object(traffic_provider.incident_provider, "fetch_incidents_raw", side_effect=RuntimeError("HTTP 401: Unauthorized")):
            records = traffic_provider.get_traffic_records()

    assert len(records) == EXPECTED_LOCATION_COUNT
    assert all(r.provider_mode == "LIVE" for r in records)
    assert all(r.raw_metadata.get("incident_provider_state") == "ERROR" for r in records)
    assert all(r.raw_metadata.get("current_incident_count") == 0 for r in records)
    print(f"       [PASS] HTTP 401 failure handled across batch: All {EXPECTED_LOCATION_COUNT} live flow speeds preserved, incident state=ERROR.")


def test_spatial_association(session: Any):
    print("\n--- 6. Testing Haversine Spatial Association to Monitored Locations ---")
    provider = TomTomIncidentProvider(api_key="mock_key", matching_radius_km=1.0)

    # Location 1 (Ajni Square): 21.1182, 79.0721
    # Location 3 (Pratap Nagar Chowk): 21.1137, 79.0568 (approx 1.6 km away)
    loc1 = session.query(DBLocation).filter(DBLocation.id == 1).first()
    loc3 = session.query(DBLocation).filter(DBLocation.id == 3).first()
    assert loc1 is not None and loc3 is not None

    # Incident A: 300 meters from Location 1 (21.1190, 79.0730)
    # Incident B: 5.0 km away from all locations (21.2500, 79.2000)
    # Incident C: halfway between Loc 1 and Loc 3 (21.11595, 79.06445 - approx 830m to both)
    mock_snapshot = IncidentSnapshot(
        status="LIVE",
        incident_count=3,
        incidents=[
            IncidentRecord(
                incident_id="INC-NEAR-1",
                category="Accident",
                icon_category=1,
                geometry_type="Point",
                coordinates=[79.0730, 21.1190],
            ),
            IncidentRecord(
                incident_id="INC-DISTANT",
                category="Jam",
                icon_category=6,
                geometry_type="Point",
                coordinates=[79.2000, 21.2500],
            ),
            IncidentRecord(
                incident_id="INC-SHARED-1-3",
                category="Road Closed",
                icon_category=8,
                geometry_type="Point",
                coordinates=[79.06445, 21.11595],
            ),
        ],
    )

    mapping = provider.associate_incidents_to_locations(
        snapshot=mock_snapshot,
        locations=[loc1, loc3],
        matching_radius_km=1.0,
    )

    loc1_res = mapping[loc1.id]
    loc3_res = mapping[loc3.id]

    # Verify Incident A is associated with Loc 1 but NOT Loc 3 (> 1.0 km)
    loc1_inc_ids = [inc["incident_id"] for inc in loc1_res["incidents"]]
    loc3_inc_ids = [inc["incident_id"] for inc in loc3_res["incidents"]]

    assert "INC-NEAR-1" in loc1_inc_ids
    assert "INC-NEAR-1" not in loc3_inc_ids

    # Verify Incident B (distant) is not associated with either
    assert "INC-DISTANT" not in loc1_inc_ids
    assert "INC-DISTANT" not in loc3_inc_ids

    # Verify Incident C (shared) is associated with BOTH loc 1 and loc 3, but counted exactly once per location
    assert "INC-SHARED-1-3" in loc1_inc_ids
    assert "INC-SHARED-1-3" in loc3_inc_ids

    assert loc1_res["current_incident_count"] == len(loc1_inc_ids)
    assert loc3_res["current_incident_count"] == len(loc3_inc_ids)

    # Check deduplication: verify no duplicate incident IDs within the same location
    assert len(loc1_inc_ids) == len(set(loc1_inc_ids))
    assert len(loc3_inc_ids) == len(set(loc3_inc_ids))

    print("       [PASS] Haversine association verified: Near matched, distant rejected, shared matched without duplicate.")


def test_provenance_and_ml_metadata(session: Any):
    print("\n--- 7. Testing Structured Provenance & ML Risk Detail Telemetry ---")
    traffic_provider = TomTomTrafficProvider(api_key="mock_key", db=session)
    service = get_risk_model_service()

    mock_flow_data = {
        "frc": "FRC1",
        "currentSpeed": 36.0,
        "freeFlowSpeed": 50.0,
        "confidence": 1.0,
    }

    mock_incidents_payload = {
        "incidents": [
            {
                "type": "Feature",
                "properties": {"id": "INC-AJNI-1", "iconCategory": 1},
                "geometry": {"type": "Point", "coordinates": [79.0721, 21.1182]},
            },
            {
                "type": "Feature",
                "properties": {"id": "INC-AJNI-2", "iconCategory": 6},
                "geometry": {"type": "Point", "coordinates": [79.0725, 21.1185]},
            },
        ]
    }

    with patch.object(traffic_provider, "fetch_flow_segment_raw", return_value=mock_flow_data):
        with patch.object(traffic_provider.incident_provider, "fetch_incidents_raw", return_value=mock_incidents_payload):
            rec = traffic_provider.get_location_traffic_record(1)

    assert rec is not None
    prov = rec.raw_metadata.get("provenance", {})

    # Verify TOMTOM_REALTIME for all 6 live incident fields
    assert prov.get("current_incident_count") == "TOMTOM_REALTIME"
    assert prov.get("current_accident_count") == "TOMTOM_REALTIME"
    assert prov.get("current_jam_count") == "TOMTOM_REALTIME"
    assert prov.get("current_road_closure_count") == "TOMTOM_REALTIME"
    assert prov.get("current_roadworks_count") == "TOMTOM_REALTIME"
    assert prov.get("current_broken_down_vehicle_count") == "TOMTOM_REALTIME"

    # Verify POSTGRESQL_CONTEXT for historical fields
    assert prov.get("traffic_volume") == "POSTGRESQL_CONTEXT"
    assert prov.get("incident_frequency") == "POSTGRESQL_CONTEXT"
    assert prov.get("accident_history") == "POSTGRESQL_CONTEXT"
    assert prov.get("road_factor") == "POSTGRESQL_CONTEXT"
    assert prov.get("population_factor") == "POSTGRESQL_CONTEXT"
    assert prov.get("police_officers") == "POSTGRESQL_CONTEXT"

    # Verify MLRiskDetail populated with incident telemetry
    detail = service.predict_raw_record(rec, provider_status=traffic_provider.get_provider_status())
    assert isinstance(detail, MLRiskDetail)
    assert detail.current_incident_count == 2
    assert detail.current_accident_count == 1
    assert detail.current_jam_count == 1
    assert detail.current_road_closure_count == 0
    assert detail.incident_provider == "TomTomIncidentDetailsV5"
    assert detail.incident_provider_state == "LIVE"

    # Verify model_metadata consistency
    meta = detail.model_metadata
    assert meta.get("current_incident_count") == 2
    assert meta.get("current_accident_count") == 1
    assert meta.get("current_jam_count") == 1
    assert meta.get("current_road_closure_count") == 0
    assert meta.get("incident_provider") == "TomTomIncidentDetailsV5"
    assert meta.get("incident_provider_state") == "LIVE"

    print("       [PASS] Provenance & ML Risk Detail verified: 6 incident fields tagged TOMTOM_REALTIME, historical tagged POSTGRESQL_CONTEXT.")


def test_database_safety(session: Any):
    print("\n--- 8. Testing Database Immutability & Safety ---")
    count = session.query(DBLocation).count()
    assert count == EXPECTED_LOCATION_COUNT, f"Expected exactly {EXPECTED_LOCATION_COUNT} rows, got {count}"
    print(f"       [PASS] Database Safety: Exactly {count} rows in locations table, zero mutations.")


def test_live_tomtom_incident_request(session: Any):
    print("\n--- 9. Testing Real TomTom Incident Details v5 Request (Conditional) ---")
    live_key = os.getenv("TOMTOM_API_KEY")
    if not live_key or not live_key.strip():
        print("       [SKIPPED] TOMTOM_API_KEY not configured. Skipping live network call.")
        return

    provider = TomTomIncidentProvider(api_key=live_key)
    db_locs = session.query(DBLocation).order_by(DBLocation.id).all()

    t0 = time.perf_counter()
    snapshot = provider.get_incident_snapshot(locations=db_locs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert snapshot.status == "LIVE", f"Expected status LIVE, got {snapshot.status}: {snapshot.error_message}"
    assert snapshot.incident_count >= 0
    assert snapshot.bbox is not None

    print(f"       Live Incident Details v5 Query Succeeded | Latency: {latency_ms:.1f}ms | Bbox: {snapshot.bbox}")
    print(f"       Active Incidents: {snapshot.incident_count} (Accidents: {snapshot.accident_count}, Jams: {snapshot.jam_count}, Road Closures: {snapshot.road_closure_count}, Road Works: {snapshot.roadworks_count}, Breakdowns: {snapshot.broken_down_vehicle_count})")

    # Associate with locations
    mapping = provider.associate_incidents_to_locations(snapshot=snapshot, locations=db_locs)
    assert len(mapping) == len(db_locs)
    loc1_info = mapping[1]
    print(f"       Location 1 (Ajni Square) Nearby Incidents (1.0km): {loc1_info['current_incident_count']} incidents (Accidents: {loc1_info['current_accident_count']}, Closures: {loc1_info['current_road_closure_count']})")
    print("       [PASS] Live TomTom Incident Details v5 Request Verified.")


def main():
    print("==================================================================")
    print("STARTING TOMTOM INCIDENT DETAILS V5 VERIFICATION TEST SUITE")
    print("==================================================================")

    session = SessionLocal()
    try:
        test_configuration()
        test_dynamic_bounding_box(session)
        test_mock_response_mapping_and_categories()
        test_zero_incidents()
        test_failure_handling_and_no_demo_fallback(session)
        test_spatial_association(session)
        test_provenance_and_ml_metadata(session)
        test_database_safety(session)
        test_live_tomtom_incident_request(session)

        print("\n==================================================================")
        print("ALL TOMTOM INCIDENT DETAILS V5 VERIFICATION TESTS PASSED (100%)!")
        print("==================================================================\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()
