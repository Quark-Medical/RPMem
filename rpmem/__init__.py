"""RPMem: Incremental multi-session memory parametrization via recurrent LoRA fusion."""

__version__ = "0.1.0"

from rpmem.config import GateConfig, HeadConfig, LoRAConfig, RPMemConfig, PerceiverConfig
from rpmem.gate import CMPGate, run_cmp_sessions

__all__ = [
    "CMPGate",
    "run_cmp_sessions",
    "RPMemConfig",
    "GateConfig",
    "HeadConfig",
    "LoRAConfig",
    "PerceiverConfig",
    "RPMemModel",
    "HypernetModel",
]


def __getattr__(name):
    if name == "RPMemModel":
        from rpmem.model import RPMemModel
        return RPMemModel
    if name == "HypernetModel":
        from rpmem.training.hypernet_model import HypernetModel
        return HypernetModel
    raise AttributeError(f"module 'rpmem' has no attribute {name!r}")
