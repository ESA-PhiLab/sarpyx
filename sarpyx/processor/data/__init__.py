"""Implemented data readers for SAR processing.

The package exposes only implemented helpers. Writer and formatter placeholders
were intentionally removed from the public surface.
"""

import importlib

_SUBMODULES = ("readers",)
_EXPORT_MODULES = {
    "read_tif": "readers",
    "read_zarr_file": "readers",
}

__all__ = [*_SUBMODULES, *_EXPORT_MODULES]

_module_cache = {}
_value_cache = {}


def __getattr__(name):
    if name in _value_cache:
        return _value_cache[name]

    module_name = name if name in _SUBMODULES else _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = _module_cache.get(module_name)
    if module is None:
        module = importlib.import_module(f".{module_name}", __name__)
        _module_cache[module_name] = module

    value = module if name == module_name else getattr(module, name)
    _value_cache[name] = value
    return value


def __dir__():
    return sorted({*globals(), *__all__})
