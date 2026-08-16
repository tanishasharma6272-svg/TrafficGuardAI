"""SQLAlchemy database models for TrafficGuard AI."""

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Location(Base):
    """SQLAlchemy model representing monitored traffic locations and risk parameters."""

    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    coordinate_source: Mapped[str] = mapped_column(String, nullable=False)
    traffic_speed: Mapped[float] = mapped_column(Float, nullable=False)
    free_flow_speed: Mapped[float] = mapped_column(Float, nullable=False)
    traffic_volume: Mapped[int] = mapped_column(Integer, nullable=False)
    incident_frequency: Mapped[float] = mapped_column(Float, nullable=False)
    accident_history: Mapped[float] = mapped_column(Float, nullable=False)
    road_factor: Mapped[float] = mapped_column(Float, nullable=False)
    population_factor: Mapped[float] = mapped_column(Float, nullable=False)
    police_officers: Mapped[int] = mapped_column(Integer, nullable=False)
