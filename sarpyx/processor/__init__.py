"""SAR processing package with lazy submodule imports.

The previous eager imports pulled in heavy optional dependencies at package
import time, which makes lightweight utilities such as ``constants`` slow to
import and can block CLI startup. Keep package import side-effect free.
"""

import importlib
from sarpyx import __version__

__all__ = ['core', 'algorithms', 'data', 'utils']

_module_cache = {}


def __getattr__(name):
    if name in _module_cache:
        return _module_cache[name]
    if name in __all__:
        module = importlib.import_module(f'.{name}', __name__)
        _module_cache[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
