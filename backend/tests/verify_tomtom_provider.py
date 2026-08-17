"""Comprehensive verification test suite for TomTom Traffic Data Provider in TrafficGuard AI."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation
from app.providers.base import (
    AggregateState,
    ProviderStatus,
    RawTrafficRecord,
    TrafficProvider,
)
from app.providers.demo_provider import DEMO_PROVENANCE, DemoTrafficProvider
from app.providers.tomtom_provider import (
    TOMTOM_PROVENANCE,
    ProviderConfigurationError,
    ProviderFetchError,
    TomTomTrafficProvider,
)
from app.providers import get_traffic_provider
from app.services.traffic_normalizer import normalize_record


def test_configuration_and_factory():
    print("\n--- 1. Testing Configuration & Provider Factory (Strict Validation) ---")

    # A1. Missing TRAFFIC_PROVIDER
    with patch.dict(os.environ, {}, clear=True):
        try:
            get_traffic_provider()
            assert False, "Should have raised ProviderConfigurationError when TRAFFIC_PROVIDER is missing"
        except ProviderConfigurationError as e:
            assert "TRAFFIC_PROVIDER environment variable is missing" in str(e)
            print("       [PASS] Missing TRAFFIC_PROVIDER raised ProviderConfigurationError.")

    # A2. Unsupported TRAFFIC_PROVIDER
    try:
        get_traffic_provider("unsupported_custom_provider")
        assert False, "Should have raised ProviderConfigurationError for unsupported provider"
    except ProviderConfigurationError as e:
        assert "Unsupported TRAFFIC_PROVIDER='unsupported_custom_provider'" in str(e)
        print("       [PASS] Unsupported TRAFFIC_PROVIDER raised ProviderConfigurationError.")

    # A3. TRAFFIC_PROVIDER=demo
    demo_p = get_traffic_provider("demo")
    assert isinstance(demo_p, DemoTrafficProvider)
    assert demo_p.provider_mode == "DEMO"
    status_demo = demo_p.get_provider_status()
    assert isinstance(status_demo, ProviderStatus)
    assert status_demo.provider == "DemoTrafficProvider"
    assert status_demo.aggregate_state == "LIVE"
    assert status_demo.successful_count == 20
    print("       [PASS] TRAFFIC_PROVIDER=demo instantiated DemoTrafficProvider with typed ProviderStatus.")

    # A4. TRAFFIC_PROVIDER=tomtom without TOMTOM_API_KEY -> UNCONFIGURED
    tomtom_unconf = get_traffic_provider("tomtom", api_key="")
    assert isinstance(tomtom_unconf, TomTomTrafficProvider)
    assert not tomtom_unconf.is_configured()
    assert tomtom_unconf.provider_mode == "UNCONFIGURED"
    status_unconf = tomtom_unconf.get_provider_status()
    assert isinstance(status_unconf, ProviderStatus)
    assert status_unconf.aggregate_state == "UNCONFIGURED"
    assert status_unconf.is_configured is False
    try:
        tomtom_unconf.get_traffic_records()
        assert False, "Should have raised ProviderConfigurationError on unconfigured provider fetch"
    except ProviderConfigurationError as e:
        assert "LIVE provider unavailable because TOMTOM_API_KEY is not configured" in str(e)
    print("       [PASS] TRAFFIC_PROVIDER=tomtom without API key reported UNCONFIGURED safely.")


def test_raw_telemetry_preservation():
    print("\n--- 2. Testing Raw Telemetry Preservation (No Clamping/Rounding) ---")
    mock_flow_payload = {
        "frc": "FRC1",
        "currentSpeed": 41.8742,
        "freeFlowSpeed": 59.4219,
        "currentTravelTime": 115,
        "freeFlowTravelTime": 78,
        "confidence": 0.94,
        "roadClosure": False,
    }

    provider = TomTomTrafficProvider(api_key="mock_test_key")

    with patch.object(provider, "fetch_flow_segment_raw", return_value=mock_flow_payload):
        record = provider.get_location_traffic_record(1)

    assert record is not None
    assert isinstance(record, RawTrafficRecord)

    # Verify exact raw floating point values are preserved without rounding or clamping
    assert record.traffic_speed == 41.8742, f"traffic_speed was mutated: {record.traffic_speed}"
    assert record.free_flow_speed == 59.4219, f"free_flow_speed was mutated: {record.free_flow_speed}"
    assert record.provider_mode == "LIVE"

    # Verify canonical normalization accepts raw float values
    normalized = normalize_record(record)
    assert normalized.traffic_speed == 41.8742
    assert normalized.free_flow_speed == 59.4219
    print(f"       [PASS] Raw telemetry preserved: speed={record.traffic_speed} km/h, free_flow={record.free_flow_speed} km/h.")


def test_strict_normalizer_rejection_on_invalid_speeds():
    print("\n--- 3. Testing Strict Normalizer Rejection on Invalid Speeds ---")
    provider = TomTomTrafficProvider(api_key="mock_test_key")
    invalid_flow = {
        "frc": "FRC2",
        "currentSpeed": -15.0,  # Invalid negative speed
        "freeFlowSpeed": 50.0,
        "confidence": 0.8,
    }

    with patch.object(provider, "fetch_flow_segment_raw", return_value=invalid_flow):
        raw_rec = provider.get_location_traffic_record(1)

    assert raw_rec is not None, "Expected raw record for location 1"
    # Provider outputs raw non-mutated value (-15.0)
    assert raw_rec.traffic_speed == -15.0

    # Normalization layer MUST reject invalid speed with clear ValueError
    try:
        normalize_record(raw_rec)
        assert False, "Normalizer should have raised ValueError on negative speed"
    except ValueError as e:
        assert "Invalid traffic_speed -15.0" in str(e)
        print(f"       [PASS] Strict normalizer rejection verified: '{e}'")


def test_typed_provider_status_across_states():
    print("\n--- 4. Testing Typed ProviderStatus Across States (LIVE, PARTIAL, ERROR, UNCONFIGURED) ---")
    provider = TomTomTrafficProvider(api_key="mock_test_key")

    mock_flow = {
        "frc": "FRC1",
        "currentSpeed": 40.0,
        "freeFlowSpeed": 60.0,
        "confidence": 0.9,
    }

    # D1. 100% Success -> State: LIVE
    with patch.object(provider, "fetch_flow_segment_raw", return_value=mock_flow):
        records = provider.get_traffic_records()

    assert len(records) == 20
    assert all(r.provider_mode == "LIVE" for r in records)
    status_live = provider.get_provider_status()
    assert isinstance(status_live, ProviderStatus)
    assert status_live.aggregate_state == "LIVE"
    assert status_live.successful_count == 20
    assert status_live.failed_count == 0
    assert status_live.total_locations == 20
    assert len(status_live.per_location_errors) == 0
    print("       Scenario A (All 20 Succeed) -> ProviderStatus(aggregate_state='LIVE', successful_count=20, failed_count=0)")

    # D2. Partial Failure (Calls 3 & 7 fail) -> State: PARTIAL
    call_count = 0
    def mixed_fetch(lat, lon):
        nonlocal call_count
        call_count += 1
        if call_count in (3, 7):
            raise ProviderFetchError("HTTP 429: Rate limit exceeded", status_code=429)
        return mock_flow

    with patch.object(provider, "fetch_flow_segment_raw", side_effect=mixed_fetch):
        records_partial = provider.get_traffic_records()

    assert len(records_partial) == 18
    assert all(r.provider_mode == "LIVE" for r in records_partial)
    status_partial = provider.get_provider_status()
    assert isinstance(status_partial, ProviderStatus)
    assert status_partial.aggregate_state == "PARTIAL"
    assert status_partial.successful_count == 18
    assert status_partial.failed_count == 2
    assert status_partial.total_locations == 20
    assert len(status_partial.per_location_errors) == 2
    print("       Scenario B (Partial Failure) -> ProviderStatus(aggregate_state='PARTIAL', successful_count=18, failed_count=2, 0 DEMO fallbacks)")

    # D3. 100% Failure -> State: ERROR
    with patch.object(provider, "fetch_flow_segment_raw", side_effect=ProviderFetchError("HTTP 500: Server Error", status_code=500)):
        try:
            provider.get_traffic_records()
            assert False, "Should have raised ProviderFetchError on 100% failure"
        except ProviderFetchError as e:
            assert "All 20 TomTom live location requests failed" in str(e)
            status_error = provider.get_provider_status()
            assert isinstance(status_error, ProviderStatus)
            assert status_error.aggregate_state == "ERROR"
            assert status_error.successful_count == 0
            assert status_error.failed_count == 20
            assert len(status_error.per_location_errors) == 20
            print("       Scenario C (100% Failure) -> ProviderStatus(aggregate_state='ERROR', successful_count=0, failed_count=20, 0 DEMO fallbacks)")

    print("[PASS] Typed ProviderStatus Verified across all operational states.")


def test_structured_data_provenance():
    print("\n--- 5. Testing Structured Data Provenance Classification ---")

    # E1. TomTom Live Provenance
    tomtom_p = TomTomTrafficProvider(api_key="mock_key")
    mock_flow = {
        "frc": "FRC1",
        "currentSpeed": 35.0,
        "freeFlowSpeed": 50.0,
        "confidence": 0.95,
    }
    with patch.object(tomtom_p, "fetch_flow_segment_raw", return_value=mock_flow):
        rec_live = tomtom_p.get_location_traffic_record(1)

    assert rec_live is not None, "Expected live record for location 1"
    prov_live = rec_live.raw_metadata.get("provenance")
    assert isinstance(prov_live, dict), "Missing structured provenance dict in live record"
    assert prov_live["traffic_speed"] == "TOMTOM_REALTIME"
    assert prov_live["free_flow_speed"] == "TOMTOM_REALTIME"
    assert prov_live["snapshot_timestamp"] == "TOMTOM_REALTIME"
    assert prov_live["traffic_volume"] == "POSTGRESQL_CONTEXT"
    assert prov_live["accident_history"] == "POSTGRESQL_CONTEXT"
    assert prov_live["incident_frequency"] == "POSTGRESQL_CONTEXT"
    assert prov_live["road_factor"] == "POSTGRESQL_CONTEXT"
    assert prov_live["population_factor"] == "POSTGRESQL_CONTEXT"
    assert prov_live["police_officers"] == "POSTGRESQL_CONTEXT"
    print("       [PASS] TomTom LIVE record provenance correctly distinguishes TOMTOM_REALTIME vs POSTGRESQL_CONTEXT.")

    # E2. Demo Provenance
    demo_p = DemoTrafficProvider()
    rec_demo = demo_p.get_location_traffic_record(1)
    assert rec_demo is not None, "Expected demo record for location 1"
    prov_demo = rec_demo.raw_metadata.get("provenance")
    assert isinstance(prov_demo, dict), "Missing structured provenance dict in demo record"
    assert all(val == "POSTGRESQL_DEMO" for val in prov_demo.values())
    print("       [PASS] Demo record provenance correctly tags all fields as POSTGRESQL_DEMO.")


def test_live_tomtom_api_call_if_configured():
    print("\n--- 6. Testing Real TomTom Traffic Flow API (Conditional on TOMTOM_API_KEY) ---")
    live_key = os.getenv("TOMTOM_API_KEY")
    if not live_key or not live_key.strip():
        print("       [SKIPPED] TOMTOM_API_KEY not configured in environment. Skipping external network call.")
        print("[PASS] Live API Check: Safe bypass verified when credentials are absent.")
        return

    provider = TomTomTrafficProvider(api_key=live_key)

    # Perform single provider request for Location 1 (Ajni Square, Nagpur)
    rec = provider.get_location_traffic_record(1)
    assert rec is not None, "Failed to retrieve location 1 from live TomTom provider"

    # 1. Verify provider_mode == "LIVE"
    assert rec.provider_mode == "LIVE", f"Expected provider_mode 'LIVE', got '{rec.provider_mode}'"

    # 2. Verify traffic_speed and free_flow_speed are numeric and within valid physical bounds
    assert isinstance(rec.traffic_speed, (int, float)), f"traffic_speed must be numeric, got {type(rec.traffic_speed)}"
    assert isinstance(rec.free_flow_speed, (int, float)), f"free_flow_speed must be numeric, got {type(rec.free_flow_speed)}"
    assert rec.traffic_speed >= 0.0, f"traffic_speed cannot be negative: {rec.traffic_speed}"
    assert rec.free_flow_speed > 0.0, f"free_flow_speed must be positive: {rec.free_flow_speed}"

    # 3. Verify structured provenance
    prov = rec.raw_metadata.get("provenance", {})
    assert prov.get("traffic_speed") == "TOMTOM_REALTIME", f"Invalid traffic_speed provenance: {prov.get('traffic_speed')}"
    assert prov.get("free_flow_speed") == "TOMTOM_REALTIME", f"Invalid free_flow_speed provenance: {prov.get('free_flow_speed')}"
    assert prov.get("snapshot_timestamp") == "TOMTOM_REALTIME"
    assert prov.get("raw_metadata") == "TOMTOM_REALTIME"
    assert prov.get("location_id") == "POSTGRESQL_CONTEXT"
    assert prov.get("traffic_volume") == "POSTGRESQL_CONTEXT"
    assert prov.get("accident_history") == "POSTGRESQL_CONTEXT"
    assert prov.get("police_officers") == "POSTGRESQL_CONTEXT"

    # 4. Verify raw_metadata contains returned TomTom telemetry
    assert rec.raw_metadata.get("provider") == "TomTom Traffic Flow API v4"
    assert "confidence" in rec.raw_metadata
    assert "frc" in rec.raw_metadata
    assert "source_coordinates" in rec.raw_metadata

    # 5. Verify record is associated with the requested PostgreSQL location
    assert rec.location_id == 1, f"Expected location_id 1, got {rec.location_id}"
    assert rec.name == "Ajni Square", f"Expected 'Ajni Square', got '{rec.name}'"
    assert rec.latitude == 21.1182, f"Expected latitude 21.1182, got {rec.latitude}"
    assert rec.longitude == 79.0721, f"Expected longitude 79.0721, got {rec.longitude}"

    status = provider.get_provider_status()
    print(f"       Live Nagpur Flow Telemetry: Location = {rec.name} (ID: {rec.location_id})")
    print(f"       Observed Speed = {rec.traffic_speed} km/h, Free-flow = {rec.free_flow_speed} km/h, FRC = {rec.raw_metadata.get('frc')}, Confidence = {rec.raw_metadata.get('confidence')}")
    print(f"       Provider State = {status.aggregate_state}, Success Count = {status.successful_count}, Failed Count = {status.failed_count}")
    print(f"[PASS] Live TomTom API Verified: Successfully ingested real traffic flow for {rec.name}.")


def test_database_safety():
    print("\n--- 7. Testing PostgreSQL Database Immutability ---")
    session = SessionLocal()
    try:
        count = session.query(DBLocation).count()
        assert count == 20, f"Database location count altered: {count}"
        print(f"[PASS] Database Safety: Exactly {count} rows in locations table, zero mutations.")
    finally:
        session.close()


if __name__ == "__main__":
    test_configuration_and_factory()
    test_raw_telemetry_preservation()
    test_strict_normalizer_rejection_on_invalid_speeds()
    test_typed_provider_status_across_states()
    test_structured_data_provenance()
    test_live_tomtom_api_call_if_configured()
    test_database_safety()
    print("\n==================================================================")
    print("ALL TOMTOM TRAFFIC PROVIDER VERIFICATION TESTS PASSED (100%)!")
    print("==================================================================\n")
