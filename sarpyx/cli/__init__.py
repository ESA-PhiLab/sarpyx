"""CLI package exposing sarpyx command-line entry points lazily."""

_cache = {}

__all__ = ['main']


def __getattr__(name):
    if name in _cache:
        return _cache[name]
    if name == 'main':
        from .worldsar import main
        _cache[name] = main
        return main
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
