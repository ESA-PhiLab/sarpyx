#!/usr/bin/env python3
"""Download and extract the snapflow_v2 Sentinel-1 burst pair with phidown."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a project dependency.
    load_dotenv = None

from phidown.downloader import download_burst_on_demand, get_token

from sarpyx.snapflow.burst_utils import extract_burst_archive


DEFAULT_MASTER_ID = "8ff4f2b3-64d8-4852-8c3b-4b2b8f729b03"
DEFAULT_SLAVE_ID = "2404a519-5e05-4dcc-95e5-b3e4e8a79127"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the same Sentinel-1 bursts selected by notebooks/snapflow_v2.ipynb.",
    )
    parser.add_argument(
        "--pair-json",
        type=Path,
        default=Path("data/output/snapflow_v2/burst_search/selected_burst_pair.json"),
        help="Selected pair JSON from the notebook workflow.",
    )
    parser.add_argument("--master-id", default=None, help="Override master burst UUID.")
    parser.add_argument("--slave-id", default=None, help="Override slave burst UUID.")
    parser.add_argument(
        "--burst-dir",
        type=Path,
        default=Path("data/bursts"),
        help="Burst cache and extraction directory.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/bursts/snapflow_v2_pair.json"),
        help="Where to write downloaded/extracted path summary.",
    )
    parser.add_argument("--retry-count", type=int, default=2, help="Download retry count per burst.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Use phidown resumable product state for burst downloads.",
    )
    return parser.parse_args()


def selected_ids(pair_json: Path, master_id: str | None, slave_id: str | None) -> tuple[str, str]:
    if master_id and slave_id:
        return master_id, slave_id
    if pair_json.exists():
        data = json.loads(pair_json.read_text(encoding="utf-8"))
        return (
            master_id or str(data["master"]["id"]),
            slave_id or str(data["slave"]["id"]),
        )
    return master_id or DEFAULT_MASTER_ID, slave_id or DEFAULT_SLAVE_ID


def credentials() -> tuple[str, str]:
    if load_dotenv is not None:
        load_dotenv(Path(".env"))
    username = os.getenv("CDSE_USERNAME") or os.getenv("CDSE_USR")
    password = os.getenv("CDSE_PASSWORD") or os.getenv("CDSE_PSW")
    if (not username or not password) and Path("secret.env").exists():
        parser = configparser.ConfigParser()
        parser.read("secret.env")
        if parser.has_section("copernicus"):
            username = username or parser.get("copernicus", "username", fallback=None)
            password = password or parser.get("copernicus", "password", fallback=None)
    if not username or not password:
        raise SystemExit(
            "CDSE credentials are required for burst downloads.\n"
            "Set them in secret.env [copernicus] or without printing secrets, for example:\n"
            "  export CDSE_USERNAME='<your CDSE username>'\n"
            "  export CDSE_PASSWORD='<your CDSE password>'\n"
            "Then rerun this command."
        )
    return username, password


def one_zip(role_dir: Path) -> Path | None:
    archives = sorted(role_dir.glob("*.zip"))
    if not archives:
        return None
    if len(archives) > 1:
        names = ", ".join(path.name for path in archives)
        raise RuntimeError(f"Expected at most one cached archive in {role_dir}, found: {names}")
    return archives[0]


def stable_safe_link(actual_safe: Path, link_path: Path) -> Path:
    if link_path.exists() or link_path.is_symlink():
        try:
            if link_path.resolve() == actual_safe.resolve():
                return link_path
        except FileNotFoundError:
            pass
        if link_path.is_symlink():
            link_path.unlink()
        else:
            return actual_safe

    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(actual_safe, link_path.parent)
    link_path.symlink_to(relative_target, target_is_directory=True)
    return link_path


def download_and_extract(
    *,
    role: str,
    burst_id: str,
    token: str,
    burst_dir: Path,
    retry_count: int,
    resume: bool,
) -> dict[str, str]:
    role_dir = burst_dir / role / burst_id
    role_dir.mkdir(parents=True, exist_ok=True)

    archive = one_zip(role_dir)
    if archive is None:
        print(f"Downloading {role} burst {burst_id} with phidown...")
        download_burst_on_demand(
            burst_id=burst_id,
            token=token,
            output_dir=role_dir,
            retry_count=retry_count,
            resume_mode="product" if resume else "off",
            state_file=str(role_dir / "download_state.json") if resume else None,
        )
        archive = one_zip(role_dir)
    else:
        print(f"Using cached {role} archive: {archive}")

    if archive is None:
        raise RuntimeError(f"Download did not produce a ZIP archive in {role_dir}")

    actual_safe = extract_burst_archive(
        archive,
        burst_dir / "extracted" / role / burst_id,
    )
    stable_safe = stable_safe_link(
        actual_safe,
        burst_dir / "extracted" / role / burst_id / f"{role}.SAFE",
    )
    print(f"{role} SAFE: {stable_safe}")
    return {
        "burst_id": burst_id,
        "archive": str(archive),
        "safe": str(stable_safe),
        "actual_safe": str(actual_safe),
    }


def main() -> int:
    args = parse_args()
    master_id, slave_id = selected_ids(args.pair_json, args.master_id, args.slave_id)
    username, password = credentials()
    print("Authenticating with CDSE via phidown...")
    token = get_token(username=username, password=password)

    result = {
        "master": download_and_extract(
            role="master",
            burst_id=master_id,
            token=token,
            burst_dir=args.burst_dir,
            retry_count=args.retry_count,
            resume=args.resume,
        ),
        "slave": download_and_extract(
            role="slave",
            burst_id=slave_id,
            token=token,
            burst_dir=args.burst_dir,
            retry_count=args.retry_count,
            resume=args.resume,
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote summary: {args.summary}")
    print("Run the pipeline with:")
    print(
        "  pipelines/sentinel_insar/run_sentinel_insar.sh "
        f"--master {result['master']['safe']} --slave {result['slave']['safe']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
