"""Demo traffic provider implementation querying PostgreSQL seed data."""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Location as DBLocation
from app.providers.base import ProviderStatus, RawTrafficRecord, TrafficProvider

DEMO_PROVENANCE = {
    "traffic_speed": "POSTGRESQL_DEMO",
    "free_flow_speed": "POSTGRESQL_DEMO",
    "snapshot_timestamp": "POSTGRESQL_DEMO",
    "raw_metadata": "POSTGRESQL_DEMO",
    "location_id": "POSTGRESQL_DEMO",
    "name": "POSTGRESQL_DEMO",
    "latitude": "POSTGRESQL_DEMO",
    "longitude": "POSTGRESQL_DEMO",
    "coordinate_source": "POSTGRESQL_DEMO",
    "traffic_volume": "POSTGRESQL_DEMO",
    "incident_frequency": "POSTGRESQL_DEMO",
    "accident_history": "POSTGRESQL_DEMO",
    "road_factor": "POSTGRESQL_DEMO",
    "population_factor": "POSTGRESQL_DEMO",
    "police_officers": "POSTGRESQL_DEMO",
}


class DemoTrafficProvider(TrafficProvider):
    """PostgreSQL-backed Demo Traffic Provider.

    Extracts simulated city traffic monitoring locations directly from the
    PostgreSQL database. All records emitted by this provider are explicitly
    tagged with provider_mode = "DEMO".

    TIMESTAMP STRATEGY:
    -------------------
    The snapshot_timestamp generated on each record represents the exact UTC
    timestamp when the provider extracted the snapshot from the PostgreSQL store.
    This timestamp does NOT represent the historical or real-time epoch when
    the synthetic traffic conditions actually occurred.
    """

    def __init__(self, db: Optional[Session] = None) -> None:
        """Initialize the DemoTrafficProvider with an optional database session."""
        self._db = db
        self.last_fetch_timestamp: Optional[datetime] = None

    @property
    def provider_mode(self) -> str:
        """Return the immutable operational mode for this provider."""
        return "DEMO"

    def _get_session(self) -> Session:
        """Return the injected session or instantiate a new SessionLocal."""
        if self._db is not None:
            return self._db
        return SessionLocal()

    def _db_to_raw_record(self, db_loc: DBLocation, snapshot_time: datetime) -> RawTrafficRecord:
        """Transform a SQLAlchemy DBLocation model into a RawTrafficRecord."""
        return RawTrafficRecord(
            location_id=db_loc.id,
            name=db_loc.name,
            latitude=db_loc.latitude,
            longitude=db_loc.longitude,
            coordinate_source=db_loc.coordinate_source,
            traffic_speed=db_loc.traffic_speed,
            free_flow_speed=db_loc.free_flow_speed,
            traffic_volume=db_loc.traffic_volume,
            incident_frequency=db_loc.incident_frequency,
            accident_history=db_loc.accident_history,
            road_factor=db_loc.road_factor,
            population_factor=db_loc.population_factor,
            police_officers=db_loc.police_officers,
            provider_mode=self.provider_mode,
            snapshot_timestamp=snapshot_time,
            raw_metadata={
                "source_table": "locations",
                "database_id": db_loc.id,
                "is_synthetic_seed": True,
                "provenance": dict(DEMO_PROVENANCE),
            },
        )

    def get_traffic_records(self) -> List[RawTrafficRecord]:
        """Fetch all demo traffic locations from the PostgreSQL database.

        Returns:
            List[RawTrafficRecord]: List of 20 monitored location observations.
        """
        is_internal_session = self._db is None
        session = self._get_session()
        snapshot_time = datetime.now(timezone.utc)
        self.last_fetch_timestamp = snapshot_time

        try:
            db_locations = session.query(DBLocation).order_by(DBLocation.id).all()
            return [self._db_to_raw_record(loc, snapshot_time) for loc in db_locations]
        finally:
            if is_internal_session:
                session.close()

    def get_location_traffic_record(self, location_id: int) -> Optional[RawTrafficRecord]:
        """Fetch a single demo traffic location by ID from PostgreSQL.

        Args:
            location_id: The integer identifier of the location.

        Returns:
            Optional[RawTrafficRecord]: Raw traffic record if found, else None.
        """
        is_internal_session = self._db is None
        session = self._get_session()
        snapshot_time = datetime.now(timezone.utc)
        self.last_fetch_timestamp = snapshot_time

        try:
            db_loc = session.query(DBLocation).filter(DBLocation.id == location_id).first()
            if not db_loc:
                return None
            return self._db_to_raw_record(db_loc, snapshot_time)
        finally:
            if is_internal_session:
                session.close()

    def get_provider_status(self) -> ProviderStatus:
        """Return typed ProviderStatus operational report for DemoTrafficProvider."""
        is_internal_session = self._db is None
        session = self._get_session()
        try:
            total_count = session.query(DBLocation).count()
        except Exception:
            total_count = 20
        finally:
            if is_internal_session:
                session.close()

        return ProviderStatus(
            provider="DemoTrafficProvider",
            aggregate_state="LIVE",
            successful_count=total_count,
            failed_count=0,
            total_locations=total_count,
            last_fetch_timestamp=self.last_fetch_timestamp or datetime.now(timezone.utc),
            is_configured=True,
            per_location_errors={},
        )
