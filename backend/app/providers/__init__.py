"""Traffic providers package for TrafficGuard AI."""

import os
from typing import Optional
from sqlalchemy.orm import Session

from app.providers.base import (
    AggregateState,
    ProviderStatus,
    RawTrafficRecord,
    TrafficProvider,
)
from app.providers.demo_provider import DemoTrafficProvider
from app.providers.tomtom_incident_provider import (
    IncidentRecord,
    IncidentSnapshot,
    TomTomIncidentProvider,
)
from app.providers.tomtom_provider import (
    ProviderConfigurationError,
    ProviderFetchError,
    TomTomTrafficProvider,
)


def get_traffic_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    db: Optional[Session] = None,
) -> TrafficProvider:
    """Factory helper resolving the configured TrafficProvider instance.

    Inspects `TRAFFIC_PROVIDER` environment variable ('tomtom' vs 'demo').
    Strictly validates configuration and NEVER silently defaults to demo mode.

    Args:
        provider_name: Optional explicit provider name override.
        api_key: Optional explicit API key override.
        db: Optional database session.

    Returns:
        TrafficProvider: Instantiated provider adhering to TrafficProvider ABC.

    Raises:
        ProviderConfigurationError: If TRAFFIC_PROVIDER is missing, empty, or unsupported.
    """
    raw_setting = provider_name if provider_name is not None else os.getenv("TRAFFIC_PROVIDER")

    if raw_setting is None or not raw_setting.strip():
        raise ProviderConfigurationError(
            "TRAFFIC_PROVIDER environment variable is missing. "
            "Must be explicitly configured as 'tomtom' or 'demo'."
        )

    selected = raw_setting.strip().lower()

    if selected == "tomtom":
        return TomTomTrafficProvider(api_key=api_key, db=db)
    elif selected == "demo":
        return DemoTrafficProvider(db=db)
    else:
        raise ProviderConfigurationError(
            f"Unsupported TRAFFIC_PROVIDER='{raw_setting}'. Supported values are 'tomtom' or 'demo'."
        )


__all__ = [
    "TrafficProvider",
    "DemoTrafficProvider",
    "TomTomTrafficProvider",
    "TomTomIncidentProvider",
    "IncidentRecord",
    "IncidentSnapshot",
    "RawTrafficRecord",
    "ProviderStatus",
    "AggregateState",
    "ProviderConfigurationError",
    "ProviderFetchError",
    "get_traffic_provider",
]
