"""Dataset adapter interfaces."""

from nts_med.data.adapters.base import (
    DATA_NOT_AVAILABLE,
    AdapterNotReadyError,
    AdapterStatus,
    DatasetAdapter,
    DatasetSample,
    DiscoveryResult,
    ValidationResult,
    opaque_sample_id,
    require_ready,
    unavailable_discovery,
)
from nts_med.data.adapters.ddr import DDRAdapter
from nts_med.data.adapters.idrid import IDRiDAdapter
from nts_med.data.adapters.maples_dr import MaplesDRAdapter
from nts_med.data.adapters.registry import create_adapter, registered_adapters

__all__ = [
    "DATA_NOT_AVAILABLE",
    "AdapterNotReadyError",
    "AdapterStatus",
    "DDRAdapter",
    "DatasetAdapter",
    "DatasetSample",
    "DiscoveryResult",
    "IDRiDAdapter",
    "MaplesDRAdapter",
    "ValidationResult",
    "create_adapter",
    "opaque_sample_id",
    "registered_adapters",
    "require_ready",
    "unavailable_discovery",
]
