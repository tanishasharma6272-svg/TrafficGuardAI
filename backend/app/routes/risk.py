"""API routes for traffic locations and risk assessment."""

from typing import List
from fastapi import APIRouter, HTTPException, status
from app.data.locations import get_all_locations, get_location_by_id
from app.models.location import Location, RiskDetail, RiskSummary
from app.services.risk_engine import calculate_risk_score

router = APIRouter(tags=["Traffic & Risk"])


@router.get(
    "/locations",
    response_model=List[Location],
    summary="Get all demo traffic locations",
    description="Returns the full list of simulated city traffic monitoring locations.",
)
def get_locations() -> List[Location]:
    """Retrieve all simulated demo traffic locations."""
    return get_all_locations()


@router.get(
    "/risk",
    response_model=List[RiskSummary],
    summary="Get risk overview for all locations",
    description="Returns calculated risk scores and risk levels for all traffic locations.",
)
def get_risk_overview() -> List[RiskSummary]:
    """Retrieve summarized risk assessments for all locations."""
    locations = get_all_locations()
    summaries: List[RiskSummary] = []

    for loc in locations:
        risk_result = calculate_risk_score(loc)
        summaries.append(
            RiskSummary(
                id=loc.id,
                name=loc.name,
                latitude=loc.latitude,
                longitude=loc.longitude,
                risk_score=risk_result["risk_score"],
                risk_level=risk_result["risk_level"],
                police_officers=loc.police_officers,
            )
        )

    return summaries


@router.get(
    "/risk/{location_id}",
    response_model=RiskDetail,
    summary="Get detailed risk assessment for a specific location",
    description="Returns comprehensive location data, calculated risk score, risk level, congestion, and factor breakdown.",
)
def get_location_risk(location_id: int) -> RiskDetail:
    """Retrieve detailed risk calculation breakdown for a single location by ID."""
    loc = get_location_by_id(location_id)
    if not loc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location with ID {location_id} not found",
        )

    risk_result = calculate_risk_score(loc)

    return RiskDetail(
        id=loc.id,
        name=loc.name,
        latitude=loc.latitude,
        longitude=loc.longitude,
        traffic_speed=loc.traffic_speed,
        free_flow_speed=loc.free_flow_speed,
        traffic_volume=loc.traffic_volume,
        incident_frequency=loc.incident_frequency,
        accident_history=loc.accident_history,
        road_factor=loc.road_factor,
        population_factor=loc.population_factor,
        police_officers=loc.police_officers,
        congestion=risk_result["congestion"],
        risk_score=risk_result["risk_score"],
        risk_level=risk_result["risk_level"],
        contributing_factors=risk_result["contributing_factors"],
    )
