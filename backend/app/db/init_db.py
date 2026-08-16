"""Database initialization and seeding script for TrafficGuard AI.

Creates the `locations` table in PostgreSQL and idempotently seeds it
with the 20 monitored locations from backend/app/data/locations.py.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def init_database() -> None:
    """Initialize database tables and seed monitored locations idempotently."""
    print("Connecting to database and ensuring tables exist...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Load all existing locations from DB
        existing_rows = {loc.id: loc for loc in db.query(DBLocation).all()}

        new_rows_to_insert = []
        already_seeded_count = 0

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

        if new_rows_to_insert:
            print(f"Inserting {len(new_rows_to_insert)} new location records...")
            db.add_all(new_rows_to_insert)
            db.commit()
            print("Insertion committed successfully.")
        else:
            print(f"All {already_seeded_count} locations are already seeded with matching data.")

        # Verification step
        verify_database(db)

    except Exception as e:
        db.rollback()
        print(f"\n[ERROR] Database initialization aborted: {e}", file=sys.stderr)
        raise
    finally:
        db.close()


def verify_database(db) -> None:
    """Verify database integrity according to requirements."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if "locations" not in tables:
        raise RuntimeError("Verification failed: 'locations' table does not exist in database.")

    rows = db.query(DBLocation).order_by(DBLocation.id).all()
    row_count = len(rows)
    if row_count != 20:
        raise RuntimeError(f"Verification failed: Expected 20 rows, found {row_count}.")

    row_ids = [r.id for r in rows]
    expected_ids = list(range(1, 21))
    if row_ids != expected_ids:
        raise RuntimeError(f"Verification failed: Expected IDs 1-20, found {row_ids}.")

    total_officers = sum(r.police_officers for r in rows)
    if total_officers != 20:
        raise RuntimeError(
            f"Verification failed: Expected 20 total police officers, found {total_officers}."
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
    print(f"Total location rows      : {row_count} (Expected: 20)")
    print(f"Location IDs             : {row_ids[0]} to {row_ids[-1]} (All 1-20 present)")
    print(f"Total police officers    : {total_officers} (Expected: 20)")
    print(f"Coordinate source valid  : All {row_count} rows have non-empty sources")
    print("=" * 50)
    print("Database is ready and verified successfully.\n")


if __name__ == "__main__":
    init_database()
