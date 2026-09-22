"""Loading of trusted research checkpoints saved before the package rename.

This is not a sandbox for untrusted pickle files. New compiler checkpoints use
plain configuration dictionaries and can be read with weights_only=True.
"""

import pickle
from types import SimpleNamespace

import torch

from rpmem import config


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "memlora.config":
            classes = {
                "MemLoRAConfig": config.RPMemConfig,
                "LoRAConfig": config.LoRAConfig,
                "PerceiverConfig": config.PerceiverConfig,
                "HeadConfig": config.HeadConfig,
                "GateConfig": config.GateConfig,
            }
            if name in classes:
                return classes[name]
        return super().find_class(module, name)


_legacy_pickle = SimpleNamespace(
    __name__="rpmem_legacy_pickle", Unpickler=_LegacyUnpickler,
    load=pickle.load, loads=pickle.loads,
)


def load_compiler_checkpoint(path, *, map_location="cpu") -> dict:
    """Read a trusted compiler checkpoint, including pre-rename config objects.

    The custom unpickler only remaps the historical config classes. It does not
    install a second package or modify sys.modules process-wide.
    """
    value = torch.load(
        path, map_location=map_location, weights_only=False,
        pickle_module=_legacy_pickle,
    )
    if not isinstance(value, dict):
        raise TypeError("compiler checkpoint must contain a dictionary")
    return value
