"""CLI package helpers."""

from __future__ import annotations


def main(argv=None):
    from sarpyx.cli.main import main as dispatch

    return dispatch(argv)

__all__ = ["main"]
