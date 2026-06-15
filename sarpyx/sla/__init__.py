"""
Sub-Look Analysis (SLA) module for sarpyx.

This module provides functionality for analyzing sub-look data in SAR processing,
including handler utilities and analysis tools.
"""

import importlib

from . import metrics
from .metrics import enl, interlook_coherence, dispersion_ratio, phase_variance, stack_metrics
# Import utility functions if they exist in utilis.py
# from .utilis import delete, unzip, delProd, command_line, iterNodes

_EXPORT_MAP = {
    'SubLookAnalysis': 'core',
    'Handler': 'core',
}
_value_cache = {}

__all__ = [
    'SubLookAnalysis',
    'Handler',
    'metrics',
    'enl',
    'interlook_coherence',
    'dispersion_ratio',
    'phase_variance',
    'stack_metrics',
    # Uncomment these when utilis.py functions are properly imported
    # 'delete',
    # 'unzip', 
    # 'delProd',
    # 'command_line',
    # 'iterNodes'
]


def __getattr__(name):
    if name in _value_cache:
        return _value_cache[name]

    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = importlib.import_module(f'.{module_name}', __name__)
    value = getattr(module, name)
    _value_cache[name] = value
    return value
