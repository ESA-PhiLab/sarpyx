import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "pbs_caller.sh"


def run_with_fake_qsub(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    submitted = tmp_path / "submitted.pbs"
    qsub = fake_bin / "qsub"
    qsub.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "cp \"$1\" \"$PBS_CALLER_TEST_CAPTURE\"\n"
        "printf '12345.server\\n'\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["PBS_CALLER_TEST_CAPTURE"] = str(submitted)

    result = subprocess.run(
        [str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    result.submitted_script = submitted.read_text(encoding="utf-8")
    return result


def test_small_profile_wraps_command_for_qsub(tmp_path: Path):
    result = run_with_fake_qsub(
        tmp_path,
        "--size",
        "small",
        "--",
        "bash",
        "-lc",
        "echo hello world",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "12345.server\n"
    assert "#PBS -N worldsar" in result.submitted_script
    assert "#PBS -S /bin/bash" in result.submitted_script
    assert "#PBS -q cpu_std" in result.submitted_script
    assert "#PBS -l walltime=02:00:00" in result.submitted_script
    assert "#PBS -l select=1:ncpus=32:mem=32g" in result.submitted_script
    assert "bash -lc 'echo hello world'" in result.submitted_script


def test_default_profile_matches_large_worldsar_resources(tmp_path: Path):
    result = run_with_fake_qsub(tmp_path, "sarpyx", "pipeline", "--list")

    assert result.returncode == 0, result.stderr
    assert "#PBS -l walltime=06:00:00" in result.submitted_script
    assert "#PBS -l select=1:ncpus=192:mem=128g" in result.submitted_script
    assert "sarpyx pipeline --list" in result.submitted_script
