"""Algorithmic Police Deployment Optimizer for TrafficGuard AI.

Implements a deterministic greedy maximum-coverage optimization algorithm
operating directly on ML risk scores and geospatial coordinates.

NOTE: This layer performs geometric and risk-density coverage optimization.
It does NOT claim causal risk reduction or retrain any ML model.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Literal, Optional, Sequence, Set

from app.models.deployment import (
    BaselineMetrics,
    DeploymentRecommendationResponse,
    OptimizedMetrics,
    SelectedDeploymentUnit,
)

# Mean Earth radius in kilometers (WGS84 spherical approximation)
EARTH_RADIUS_KM: float = 6371.0088


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle geodesic distance between two points on Earth.

    Uses the Haversine formula:
        phi1 = radians(lat1), phi2 = radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin^2(dphi / 2) + cos(phi1) * cos(phi2) * sin^2(dlambda / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = EARTH_RADIUS_KM * c

    Args:
        lat1: Latitude of origin point in degrees.
        lon1: Longitude of origin point in degrees.
        lat2: Latitude of destination point in degrees.
        lon2: Longitude of destination point in degrees.

    Returns:
        float: Geodesic distance in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    sin_dphi = math.sin(delta_phi / 2.0)
    sin_dlambda = math.sin(delta_lambda / 2.0)

    a = (
        sin_dphi * sin_dphi
        + math.cos(phi1) * math.cos(phi2) * sin_dlambda * sin_dlambda
    )
    # Clamp 'a' to [0.0, 1.0] to guard against floating-point precision edge cases
    a = min(max(a, 0.0), 1.0)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c


@dataclass(frozen=True)
class LocationRiskNode:
    """Decoupled input data structure representing a location and its ML risk assessment."""

    id: int
    name: str
    latitude: float
    longitude: float
    risk_score: float
    risk_level: str


class DeploymentOptimizerService:
    """Deterministic greedy maximum-coverage optimizer for police patrol allocation."""

    def optimize_deployment(
        self,
        locations: Sequence[LocationRiskNode],
        available_units: int,
        coverage_radius_km: float,
        min_risk_level: Optional[Literal["High", "Critical"]] = None,
    ) -> DeploymentRecommendationResponse:
        """Compute optimal police deployment placements using greedy marginal coverage.

        Algorithm:
        1. Filter eligible locations according to the requested min_risk_level threshold:
           - "Critical": locations with risk_level == "Critical" (risk_score > 80.0)
           - "High" or None (default): locations with risk_level in ("Critical", "High") (risk_score > 60.0)
        2. Compute pairwise geographic distances using the Haversine formula to establish
           the coverage neighborhood N(c) for each candidate location.
        3. Iteratively select candidate units that maximize:
           (uncovered_marginal_risk_score, uncovered_location_count, candidate_risk_score, -candidate_id)
        4. Update remaining uncovered eligible locations until available units are exhausted
           or all eligible nodes are covered.
        5. Return structured, deterministic recommendations and coverage metrics.

        Args:
            locations: Monitored locations with ML risk scores and coordinates.
            available_units: Number of deployable patrol units (> 0).
            coverage_radius_km: Patrol coverage radius in kilometers (> 0.0).
            min_risk_level: Optional minimum categorical risk threshold ("High" or "Critical").

        Returns:
            DeploymentRecommendationResponse: Deterministic deployment placements and metrics.
        """
        # Step 1: Filter eligible high-risk locations
        eligible_nodes: List[LocationRiskNode] = []
        for loc in locations:
            if min_risk_level == "Critical":
                if loc.risk_level == "Critical":
                    eligible_nodes.append(loc)
            else:
                # Default / "High": include both "Critical" and "High"
                if loc.risk_level in ("Critical", "High"):
                    eligible_nodes.append(loc)

        total_eligible_locations = len(eligible_nodes)
        total_eligible_risk_score = round(
            sum(node.risk_score for node in eligible_nodes), 2
        )

        # Handle empty eligible set edge case
        if total_eligible_locations == 0:
            return DeploymentRecommendationResponse(
                available_units=available_units,
                coverage_radius_km=coverage_radius_km,
                selected_units=[],
                baseline_metrics=BaselineMetrics(
                    eligible_high_risk_locations=0,
                    total_eligible_risk_score=0.0,
                ),
                optimized_metrics=OptimizedMetrics(
                    covered_locations=0,
                    covered_risk_score=0.0,
                    risk_coverage_percent=0.0,
                    uncovered_risk_score=0.0,
                    uncovered_risk_percent=0.0,
                ),
                algorithm="GREEDY_COVERAGE_OPTIMIZER",
            )

        # Step 2: Precompute coverage neighborhoods N(c) for all candidate locations
        # Candidate locations can cover themselves and any eligible location within coverage_radius_km
        coverage_map: Dict[int, List[LocationRiskNode]] = {}
        for candidate in eligible_nodes:
            covered: List[LocationRiskNode] = []
            for target in eligible_nodes:
                dist = haversine_distance(
                    candidate.latitude,
                    candidate.longitude,
                    target.latitude,
                    target.longitude,
                )
                if dist <= coverage_radius_km:
                    covered.append(target)
            # Sort covered nodes deterministically by ID ascending
            covered.sort(key=lambda node: node.id)
            coverage_map[candidate.id] = covered

        # Step 3: Greedy Iterative Allocation
        uncovered_ids: Set[int] = {node.id for node in eligible_nodes}
        node_lookup: Dict[int, LocationRiskNode] = {
            node.id: node for node in eligible_nodes
        }
        selected_units: List[SelectedDeploymentUnit] = []
        allocated_candidates: Set[int] = set()

        for rank in range(1, available_units + 1):
            if not uncovered_ids:
                # All eligible high-risk locations are fully covered
                break

            best_candidate: Optional[LocationRiskNode] = None
            best_marginal_risk: float = -1.0
            best_marginal_count: int = -1
            best_marginal_uncovered: List[LocationRiskNode] = []

            for candidate in eligible_nodes:
                if candidate.id in allocated_candidates:
                    # Do not place multiple units at the exact same location hub
                    continue

                candidate_coverage = coverage_map[candidate.id]
                marginal_uncovered = [
                    node for node in candidate_coverage if node.id in uncovered_ids
                ]
                marginal_risk = sum(node.risk_score for node in marginal_uncovered)
                marginal_count = len(marginal_uncovered)

                # If candidate offers zero marginal coverage, skip consideration
                if marginal_count == 0 or marginal_risk <= 0.0:
                    continue

                # Ranking tuple:
                # 1. Highest marginal uncovered risk score (primary)
                # 2. Highest count of uncovered locations (secondary)
                # 3. Candidate's own intrinsic risk score (tertiary)
                # 4. Lowest candidate ID for strict determinism (quaternary)
                current_tuple = (
                    marginal_risk,
                    marginal_count,
                    candidate.risk_score,
                    -candidate.id,
                )

                if best_candidate is None:
                    best_candidate = candidate
                    best_marginal_risk = marginal_risk
                    best_marginal_count = marginal_count
                    best_marginal_uncovered = marginal_uncovered
                else:
                    best_tuple = (
                        best_marginal_risk,
                        best_marginal_count,
                        best_candidate.risk_score,
                        -best_candidate.id,
                    )
                    if current_tuple > best_tuple:
                        best_candidate = candidate
                        best_marginal_risk = marginal_risk
                        best_marginal_count = marginal_count
                        best_marginal_uncovered = marginal_uncovered

            # If no candidate provides any marginal coverage, terminate early
            if best_candidate is None or best_marginal_risk <= 0.0:
                break

            # Record selected unit
            all_unit_covered_nodes = coverage_map[best_candidate.id]
            covered_ids = [node.id for node in all_unit_covered_nodes]
            covered_risk_sum = round(
                sum(node.risk_score for node in all_unit_covered_nodes), 2
            )

            selected_units.append(
                SelectedDeploymentUnit(
                    rank=rank,
                    location_id=best_candidate.id,
                    location_name=best_candidate.name,
                    risk_score=best_candidate.risk_score,
                    risk_level=best_candidate.risk_level,
                    latitude=best_candidate.latitude,
                    longitude=best_candidate.longitude,
                    covered_location_ids=covered_ids,
                    covered_location_count=len(covered_ids),
                    covered_risk_score=covered_risk_sum,
                )
            )

            # Mark newly covered locations
            allocated_candidates.add(best_candidate.id)
            for newly_covered in best_marginal_uncovered:
                uncovered_ids.discard(newly_covered.id)

        # Step 4: Compute final coverage metrics
        distinct_covered_ids: Set[int] = set()
        for unit in selected_units:
            distinct_covered_ids.update(unit.covered_location_ids)

        covered_locations_count = len(distinct_covered_ids)
        covered_risk_score = round(
            sum(node_lookup[lid].risk_score for lid in distinct_covered_ids if lid in node_lookup),
            2,
        )

        if total_eligible_risk_score > 0.0:
            risk_coverage_percent = round(
                (covered_risk_score / total_eligible_risk_score) * 100.0, 2
            )
            uncovered_risk_score = round(
                max(total_eligible_risk_score - covered_risk_score, 0.0), 2
            )
            uncovered_risk_percent = round(
                (uncovered_risk_score / total_eligible_risk_score) * 100.0, 2
            )
        else:
            risk_coverage_percent = 0.0
            uncovered_risk_score = 0.0
            uncovered_risk_percent = 0.0

        return DeploymentRecommendationResponse(
            available_units=available_units,
            coverage_radius_km=coverage_radius_km,
            selected_units=selected_units,
            baseline_metrics=BaselineMetrics(
                eligible_high_risk_locations=total_eligible_locations,
                total_eligible_risk_score=total_eligible_risk_score,
            ),
            optimized_metrics=OptimizedMetrics(
                covered_locations=covered_locations_count,
                covered_risk_score=covered_risk_score,
                risk_coverage_percent=risk_coverage_percent,
                uncovered_risk_score=uncovered_risk_score,
                uncovered_risk_percent=uncovered_risk_percent,
            ),
            algorithm="GREEDY_COVERAGE_OPTIMIZER",
        )


# Global singleton instance
_GLOBAL_DEPLOYMENT_OPTIMIZER: Optional[DeploymentOptimizerService] = None


def get_deployment_optimizer() -> DeploymentOptimizerService:
    """Retrieve the singleton DeploymentOptimizerService instance."""
    global _GLOBAL_DEPLOYMENT_OPTIMIZER
    if _GLOBAL_DEPLOYMENT_OPTIMIZER is None:
        _GLOBAL_DEPLOYMENT_OPTIMIZER = DeploymentOptimizerService()
    return _GLOBAL_DEPLOYMENT_OPTIMIZER
