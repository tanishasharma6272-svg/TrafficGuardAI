"""Shared risk-level thresholds and classification policy for TrafficGuard AI.

Single source of truth for risk categorization across all backend services
(rule-based demo engine, ML inference services, and downstream modules).
"""

from typing import Dict, Tuple

# Canonical Risk Level Thresholds (Bounded in [0.0, 100.0])
RISK_THRESHOLD_LOW_MAX: float = 30.0
RISK_THRESHOLD_MEDIUM_MAX: float = 60.0
RISK_THRESHOLD_HIGH_MAX: float = 80.0
RISK_THRESHOLD_CRITICAL_MAX: float = 100.0

# Canonical Level Enums / String Labels
LEVEL_LOW: str = "Low"
LEVEL_MEDIUM: str = "Medium"
LEVEL_HIGH: str = "High"
LEVEL_CRITICAL: str = "Critical"

RISK_LEVEL_DEFINITIONS: Dict[str, Tuple[float, float]] = {
    LEVEL_LOW: (0.0, RISK_THRESHOLD_LOW_MAX),
    LEVEL_MEDIUM: (RISK_THRESHOLD_LOW_MAX, RISK_THRESHOLD_MEDIUM_MAX),
    LEVEL_HIGH: (RISK_THRESHOLD_MEDIUM_MAX, RISK_THRESHOLD_HIGH_MAX),
    LEVEL_CRITICAL: (RISK_THRESHOLD_HIGH_MAX, RISK_THRESHOLD_CRITICAL_MAX),
}


def classify_risk_score(score: float) -> str:
    """Map a numerical risk score in [0.0, 100.0] to a standardized risk level string.

    Policy:
    - score <= 30.0: 'Low'
    - 30.0 < score <= 60.0: 'Medium'
    - 60.0 < score <= 80.0: 'High'
    - score > 80.0: 'Critical'

    Args:
        score: Numerical risk score in range [0.0, 100.0].

    Returns:
        str: Standardized categorical risk level ('Low', 'Medium', 'High', 'Critical').
    """
    if score <= RISK_THRESHOLD_LOW_MAX:
        return LEVEL_LOW
    elif score <= RISK_THRESHOLD_MEDIUM_MAX:
        return LEVEL_MEDIUM
    elif score <= RISK_THRESHOLD_HIGH_MAX:
        return LEVEL_HIGH
    else:
        return LEVEL_CRITICAL
