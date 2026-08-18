"""Database initialization and seeding script for TrafficGuard AI.

Creates the `locations` table in PostgreSQL and idempotently seeds it
with the 20 monitored locations from backend/app/data/locations.py.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure backend root is on sys.path if run directly as a script
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy import inspect
from app.db.database import Base, engine, SessionLocal
from app.db.models import Location as DBLocation
from app.data.locations import DEMO_LOCATIONS


FLOAT_FIELDS = {
    "latitude",
    "longitude",
    "traffic_speed",
    "free_flow_speed",
    "incident_frequency",
    "accident_history",
    "road_factor",
    "population_factor",
}

ALL_FIELDS = [
    "id",
    "name",
    "latitude",
    "longitude",
    "coordinate_source",
    "traffic_speed",
    "free_flow_speed",
    "traffic_volume",
    "incident_frequency",
    "accident_history",
    "road_factor",
    "population_factor",
    "police_officers",
]


def compare_fields(db_row: DBLocation, demo_dict: Dict[str, Any]) -> List[str]:
    """Compare a stored DB location row with its expected demo values.

    Returns a list of error strings describing any mismatched fields.
    """
    mismatches = []
    for field in ALL_FIELDS:
        db_val = getattr(db_row, field)
        demo_val = demo_dict[field]

        if field in FLOAT_FIELDS:
            if not math.isclose(float(db_val), float(demo_val), rel_tol=1e-5, abs_tol=1e-5):
                mismatches.append(f"{field}: expected {demo_val}, found {db_val}")
        else:
            if db_val != demo_val:
                mismatches.append(f"{field}: expected {demo_val!r}, found {db_val!r}")
    return mismatches


EXPECTED_LOCATION_COUNT: int = 50


def validate_locations_data(locations: List[Any]) -> None:
    """Validate uniqueness, coordinate ranges, and attribute validity of location data."""
    seen_ids = set()
    seen_names = set()
    seen_coords = set()

    for loc in locations:
        loc_id = loc.id
        name = loc.name
        lat = loc.latitude
        lon = loc.longitude
        source = loc.coordinate_source

        if loc_id in seen_ids:
            raise ValueError(f"Duplicate location ID found: {loc_id}")
        seen_ids.add(loc_id)

        if name in seen_names:
            raise ValueError(f"Duplicate location name found: '{name}'")
        seen_names.add(name)

        coord_pair = (round(lat, 5), round(lon, 5))
        if coord_pair in seen_coords:
            raise ValueError(f"Duplicate coordinates found for '{name}': {coord_pair}")
        seen_coords.add(coord_pair)

        # Geographic bounds check for Nagpur metropolitan region
        if not (20.95 <= lat <= 21.35):
            raise ValueError(f"Latitude out of Nagpur bounds for '{name}' (ID: {loc_id}): {lat}")
        if not (78.85 <= lon <= 79.35):
            raise ValueError(f"Longitude out of Nagpur bounds for '{name}' (ID: {loc_id}): {lon}")

        if not source or not source.strip():
            raise ValueError(f"Missing or empty coordinate_source for '{name}' (ID: {loc_id})")


def init_database() -> None:
    """Initialize database tables and seed monitored locations idempotently."""
    print("Connecting to database and ensuring tables exist...")
    Base.metadata.create_all(bind=engine)

    # Validate the full location dataset first
    validate_locations_data(DEMO_LOCATIONS)

    db = SessionLocal()
    try:
        # 1. Take a pre-mutation snapshot of existing database rows (e.g. IDs 1-20)
        existing_rows = {loc.id: loc for loc in db.query(DBLocation).all()}
        pre_snapshot: Dict[int, Dict[str, Any]] = {}
        for loc_id, row in existing_rows.items():
            pre_snapshot[loc_id] = {
                field: getattr(row, field) for field in ALL_FIELDS
            }

        new_rows_to_insert = []
        already_seeded_count = 0

        # 2. Compare existing rows and prepare additive insertion for missing IDs
        for demo_loc in DEMO_LOCATIONS:
            demo_dict = demo_loc.model_dump()
            loc_id = demo_dict["id"]

            if loc_id in existing_rows:
                db_row = existing_rows[loc_id]
                mismatches = compare_fields(db_row, demo_dict)
                if mismatches:
                    error_msg = (
                        f"Integrity Error: Existing location ID {loc_id} ('{db_row.name}') "
                        f"has mismatched fields with DEMO_LOCATIONS:\n  - "
                        + "\n  - ".join(mismatches)
                    )
                    raise ValueError(error_msg)
                already_seeded_count += 1
            else:
                new_rows_to_insert.append(DBLocation(**demo_dict))

        # 3. Additive insertion only (never delete or overwrite existing rows)
        if new_rows_to_insert:
            print(f"Inserting {len(new_rows_to_insert)} new location records (IDs {[r.id for r in new_rows_to_insert]})...")
            db.add_all(new_rows_to_insert)
            db.commit()
            print("Insertion committed successfully.")
        else:
            print(f"All {already_seeded_count} locations are already seeded with matching data.")

        # 4. Post-mutation verification & comparison against pre-snapshot
        verify_database(db, pre_snapshot)

    except Exception as e:
        db.rollback()
        print(f"\n[ERROR] Database initialization aborted: {e}", file=sys.stderr)
        raise
    finally:
        db.close()


def verify_database(db, pre_snapshot: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
    """Verify database integrity and ensure existing locations were completely preserved."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if "locations" not in tables:
        raise RuntimeError("Verification failed: 'locations' table does not exist in database.")

    rows = db.query(DBLocation).order_by(DBLocation.id).all()
    row_count = len(rows)
    if row_count != EXPECTED_LOCATION_COUNT:
        raise RuntimeError(f"Verification failed: Expected {EXPECTED_LOCATION_COUNT} rows, found {row_count}.")

    row_ids = [r.id for r in rows]
    expected_ids = list(range(1, EXPECTED_LOCATION_COUNT + 1))
    if row_ids != expected_ids:
        raise RuntimeError(f"Verification failed: Expected IDs 1-{EXPECTED_LOCATION_COUNT}, found {row_ids}.")

    # Verify pre-snapshot preservation (e.g. IDs 1-20 field by field)
    if pre_snapshot:
        for pre_id, pre_data in pre_snapshot.items():
            db_row = next((r for r in rows if r.id == pre_id), None)
            if db_row is None:
                raise RuntimeError(f"Integrity Error: Previously existing location ID {pre_id} disappeared!")
            mismatches = compare_fields(db_row, pre_data)
            if mismatches:
                raise RuntimeError(
                    f"Integrity Error: Previously existing location ID {pre_id} was mutated:\n  - "
                    + "\n  - ".join(mismatches)
                )

    empty_coord_sources = [r.id for r in rows if not r.coordinate_source or not r.coordinate_source.strip()]
    if empty_coord_sources:
        raise RuntimeError(
            f"Verification failed: Empty coordinate_source found for IDs: {empty_coord_sources}."
        )

    print("\n" + "=" * 50)
    print(" DATABASE INITIALIZATION & VERIFICATION SUMMARY ")
    print("=" * 50)
    print(f"Table 'locations' status : Present")
    print(f"Total location rows      : {row_count} (Expected: {EXPECTED_LOCATION_COUNT})")
    print(f"Location IDs             : {row_ids[0]} to {row_ids[-1]} (All 1-{EXPECTED_LOCATION_COUNT} present)")
    print(f"Preserved pre-snapshot   : {len(pre_snapshot) if pre_snapshot else 0} locations verified 100% identical")
    print(f"Coordinate source valid  : All {row_count} rows have non-empty sources")
    print("=" * 50)
    print("Database is ready and verified successfully.\n")


if __name__ == "__main__":
    init_database()

