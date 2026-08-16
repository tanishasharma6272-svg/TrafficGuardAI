"""SQLAlchemy database models for TrafficGuard AI."""

from sqlalchemy import Column, Float, Integer, String
from app.db.database import Base


class Location(Base):
    """SQLAlchemy model representing monitored traffic locations and risk parameters."""

    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    coordinate_source = Column(String, nullable=False)
    traffic_speed = Column(Float, nullable=False)
    free_flow_speed = Column(Float, nullable=False)
    traffic_volume = Column(Integer, nullable=False)
    incident_frequency = Column(Float, nullable=False)
    accident_history = Column(Float, nullable=False)
    road_factor = Column(Float, nullable=False)
    population_factor = Column(Float, nullable=False)
    police_officers = Column(Integer, nullable=False)
