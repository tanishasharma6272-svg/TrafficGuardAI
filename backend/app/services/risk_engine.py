"""Risk calculation engine for traffic monitoring locations.

NOTE: All normalization constants and weights are DEMO DATA assumptions
used for simulation and development, not empirical field calibrations.
"""

from typing import Any, Dict
from app.models.location import Location

# Fixed DEMO DATA normalization constants (Explicit demo assumptions)
MAX_DEMO_TRAFFIC_VOLUME: float = 50000.0
MAX_DEMO_INCIDENT_FREQUENCY: float = 10.0
MAX_DEMO_ACCIDENT_HISTORY: float = 10.0

# Formula Weight Distributions (Total: 1.00 / 100%)
WEIGHT_CONGESTION: float = 0.30
WEIGHT_INCIDENT_FREQUENCY: float = 0.25
WEIGHT_ACCIDENT_HISTORY: float = 0.20
WEIGHT_ROAD_FACTOR: float = 0.15
WEIGHT_TRAFFIC_POPULATION: float = 0.10


def calculate_congestion(location: Location) -> float:
    """Calculate traffic congestion ratio from observed speed and free flow speed.

    Formula: 1 - (traffic_speed / free_flow_speed)
    Protects against division by zero and clamps output to [0.0, 1.0].
    """
    if location.free_flow_speed <= 0.0:
        return 0.0

    speed_ratio = location.traffic_speed / location.free_flow_speed
    congestion = 1.0 - speed_ratio

    # Clamp congestion between 0.0 (no delay / free flow) and 1.0 (standstill)
    return min(max(congestion, 0.0), 1.0)


def get_risk_level(score: float) -> str:
    """Map a numerical risk score (0-100) to a categorical risk level.

    Thresholds:
    - 0-30: Low
    - 31-60: Medium
    - 61-80: High
    - 81-100: Critical
    """
    if score <= 30.0:
        return "Low"
    elif score <= 60.0:
        return "Medium"
    elif score <= 80.0:
        return "High"
    else:
        return "Critical"


def calculate_risk_score(location: Location) -> Dict[str, Any]:
    """Calculate a composite risk score (0-100) and risk breakdown for a location.

    Uses fixed documented DEMO DATA normalization scales:
    - Congestion (30%): derived from speed ratio
    - Incident Frequency (25%): normalized against MAX_DEMO_INCIDENT_FREQUENCY (10.0)
    - Accident History (20%): normalized against MAX_DEMO_ACCIDENT_HISTORY (10.0)
    - Road Factor (15%): normalized in [0.0, 1.0]
    - Traffic/Population Factor (10%): 0.6 * (volume / 50000.0) + 0.4 * population_factor
    """
    # 1. Congestion factor [0.0, 1.0]
    congestion_norm = calculate_congestion(location)

    # 2. Incident frequency factor [0.0, 1.0]
    incident_norm = min(
        max(location.incident_frequency / MAX_DEMO_INCIDENT_FREQUENCY, 0.0),
        1.0,
    )

    # 3. Accident history factor [0.0, 1.0]
    accident_norm = min(
        max(location.accident_history / MAX_DEMO_ACCIDENT_HISTORY, 0.0),
        1.0,
    )

    # 4. Road infrastructure factor [0.0, 1.0]
    road_norm = min(max(location.road_factor, 0.0), 1.0)

    # 5. Traffic & population factor [0.0, 1.0] (Demo assumption: 60% volume + 40% population)
    traffic_volume_norm = min(
        max(location.traffic_volume / MAX_DEMO_TRAFFIC_VOLUME, 0.0),
        1.0,
    )
    pop_norm = min(max(location.population_factor, 0.0), 1.0)
    traffic_pop_norm = min(
        max(0.6 * traffic_volume_norm + 0.4 * pop_norm, 0.0),
        1.0,
    )

    # Calculate weighted sub-components on a 0-100 scale
    congestion_component = round(WEIGHT_CONGESTION * congestion_norm * 100.0, 2)
    incident_frequency_component = round(
        WEIGHT_INCIDENT_FREQUENCY * incident_norm * 100.0, 2
    )
    accident_history_component = round(
        WEIGHT_ACCIDENT_HISTORY * accident_norm * 100.0, 2
    )
    road_factor_component = round(WEIGHT_ROAD_FACTOR * road_norm * 100.0, 2)
    traffic_population_component = round(
        WEIGHT_TRAFFIC_POPULATION * traffic_pop_norm * 100.0, 2
    )

    # Total composite score (bounded strictly to 0.0 - 100.0)
    total_score = round(
        min(
            max(
                congestion_component
                + incident_frequency_component
                + accident_history_component
                + road_factor_component
                + traffic_population_component,
                0.0,
            ),
            100.0,
        ),
        2,
    )

    risk_level = get_risk_level(total_score)

    return {
        "risk_score": total_score,
        "risk_level": risk_level,
        "congestion": round(congestion_norm, 4),
        "contributing_factors": {
            "congestion_component": congestion_component,
            "incident_frequency_component": incident_frequency_component,
            "accident_history_component": accident_history_component,
            "road_factor_component": road_factor_component,
            "traffic_population_component": traffic_population_component,
        },
    }
