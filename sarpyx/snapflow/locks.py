"""Operational locks for WorldSAR processing."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def worldsar_product_lock_path(product_path: str | Path, output_dir: str | Path) -> Path:
    product_name = Path(product_path).name.rstrip("/")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in product_name)[:180]
    return Path(output_dir) / ".worldsar_locks" / f"{safe_name}.lock"


@contextmanager
def worldsar_product_lock(product_path: str | Path, output_dir: str | Path, timeout: float = 0.0):
    lock_path = worldsar_product_lock_path(product_path, output_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = float(timeout or 0.0)
    deadline = time.monotonic() + timeout
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if timeout <= 0 or time.monotonic() >= deadline:
                    holder = _read_lock_holder(handle)
                    raise RuntimeError(
                        f"WorldSAR product lock is held for {Path(product_path).name}: {lock_path}"
                        + (f" by {holder}" if holder else "")
                    )
                time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))
        _write_lock_holder(handle, product_path, output_dir)
        yield lock_path
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _write_lock_holder(handle, product_path: str | Path, output_dir: str | Path) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "product": str(Path(product_path).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()


def _read_lock_holder(handle) -> str:
    try:
        handle.seek(0)
        payload = json.load(handle)
    except Exception:
        return ""
    host = payload.get("host")
    pid = payload.get("pid")
    started = payload.get("started_at")
    parts = [str(value) for value in (host, f"pid={pid}" if pid else None, started) if value]
    return ", ".join(parts)
