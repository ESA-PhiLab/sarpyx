import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def target_names(help_text: str) -> set[str]:
    names = set()
    for line in help_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "Targets:":
            continue
        names.add(stripped.split(maxsplit=1)[0])
    return names


def test_help_exposes_only_user_workflow_targets():
    result = run_make("help")

    assert result.returncode == 0, result.stderr
    assert target_names(result.stdout) == {
        "install",
        "download",
        "pipeline",
        "pipeline-pair",
        "pbs",
    }


def test_pbs_target_wraps_repo_cli_module_without_uv():
    result = run_make(
        "pbs",
        "PBS_ARGS=--dry-run",
        "PBS_SIZE=small",
        "CMD=python -m sarpyx.cli.main pipeline --list",
    )

    assert result.returncode == 0, result.stderr
    assert "#PBS -l walltime=02:00:00" in result.stdout
    assert "exec bash -lc 'python -m sarpyx.cli.main pipeline --list'" in result.stdout
    assert "uv run sarpyx" not in result.stdout


def test_install_uses_local_prefix_environment_and_verifies_snap_sarpyx_phidown(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    env_prefix = tmp_path / "local-env"
    log = tmp_path / "conda.log"
    conda = fake_bin / "conda"
    conda.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$SARPYX_TEST_CONDA_LOG\"\n"
        "if [[ \"$1 $2\" == 'env create' || \"$1 $2\" == 'env update' ]]; then\n"
        "  prefix=''\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    if [[ \"$1\" == '-p' ]]; then prefix=\"$2\"; shift 2; continue; fi\n"
        "    shift\n"
        "  done\n"
        "  mkdir -p \"$prefix/bin\" \"$prefix/opt/esa-snap/bin\"\n"
        "  touch \"$prefix/bin/python\" \"$prefix/bin/sarpyx\" \"$prefix/bin/phidown\" \"$prefix/opt/esa-snap/bin/gpt\"\n"
        "  chmod +x \"$prefix/bin/python\" \"$prefix/bin/sarpyx\" \"$prefix/bin/phidown\" \"$prefix/opt/esa-snap/bin/gpt\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == 'run' ]]; then exit 0; fi\n",
        encoding="utf-8",
    )
    conda.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["SARPYX_TEST_CONDA_LOG"] = str(log)
    env["ENV_PREFIX"] = str(env_prefix)
    result = subprocess.run(
        ["bash", "scripts/install_conda.sh"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert f"env create -p {env_prefix} -f {REPO_ROOT / 'environment.yml'}" in calls
    assert f"python -m pip install -e {REPO_ROOT}[copernicus]" in calls
    assert "run -p " in calls
    assert "command -v gpt" not in calls
    assert str(env_prefix / "opt" / "esa-snap" / "bin" / "gpt") in result.stdout
    assert str(env_prefix / "bin" / "sarpyx") in result.stdout
    assert str(env_prefix / "bin" / "phidown") in result.stdout


def test_pipeline_helper_uses_repo_dispatcher_not_stale_console_script(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "python.log"
    python = fake_bin / "python"
    python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" > \"$SARPYX_TEST_PYTHON_LOG\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["SARPYX_TEST_PYTHON_LOG"] = str(log)
    env["PYTHON"] = str(python)
    env["PIPELINE"] = "s1_tops"
    env["INPUT"] = "/tmp/input.SAFE"
    env["OUTPUT"] = "outputs/make/s1_tops"
    env["CUTS_OUTDIR"] = "outputs/make/s1_tops/tiles"

    result = subprocess.run(
        ["bash", "scripts/run_pipeline.sh", "single"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "-m sarpyx.cli.main pipeline s1_tops" in log.read_text(encoding="utf-8")
    assert "sarpyx pipeline s1_tops" not in result.stdout


def test_pipeline_helper_defaults_python_and_gpt_to_local_conda_prefix(tmp_path: Path):
    env_prefix = tmp_path / "env"
    bin_dir = env_prefix / "bin"
    gpt = env_prefix / "opt" / "esa-snap" / "bin" / "gpt"
    bin_dir.mkdir(parents=True)
    gpt.parent.mkdir(parents=True)
    log = tmp_path / "python.log"
    python = bin_dir / "python"
    python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" > \"$SARPYX_TEST_PYTHON_LOG\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    gpt.write_text("#!/bin/bash\n", encoding="utf-8")
    gpt.chmod(0o755)

    env = os.environ.copy()
    env.pop("PYTHON", None)
    env.pop("GPT_PATH", None)
    env["SARPYX_TEST_PYTHON_LOG"] = str(log)
    env["ENV_PREFIX"] = str(env_prefix)
    env["PIPELINE"] = "s1_tops"
    env["INPUT"] = "/tmp/input.SAFE"
    env["OUTPUT"] = "outputs/make/s1_tops"
    env["CUTS_OUTDIR"] = "outputs/make/s1_tops/tiles"

    result = subprocess.run(
        ["bash", "scripts/run_pipeline.sh", "single"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    call = log.read_text(encoding="utf-8")
    assert call.startswith("-m sarpyx.cli.main pipeline s1_tops")
    assert f"--gpt-path {gpt}" in call
    assert f"Running: {python}" in result.stdout


def test_make_pipeline_ignores_ambient_python_and_gpt_path_env():
    env = os.environ.copy()
    env["PYTHON"] = "/bad/python"
    env["GPT_PATH"] = "/bad/gpt"

    result = subprocess.run(
        ["make", "-n", "pipeline", "PIPELINE=s1_tops", "INPUT=/tmp/input.SAFE"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"PYTHON=\"{REPO_ROOT / '.conda' / 'sarpyx' / 'bin' / 'python'}\"" in result.stdout
    assert f"GPT_PATH=\"{REPO_ROOT / '.conda' / 'sarpyx' / 'opt' / 'esa-snap' / 'bin' / 'gpt'}\"" in result.stdout
    assert "/bad/python" not in result.stdout
    assert "/bad/gpt" not in result.stdout


def test_make_download_ignores_ambient_phidown_env():
    env = os.environ.copy()
    env["PHIDOWN"] = "/bad/phidown"

    result = subprocess.run(
        ["make", "-n", "download", "PRODUCT_NAME=S1_PRODUCT.SAFE"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"PHIDOWN=\"{REPO_ROOT / '.conda' / 'sarpyx' / 'bin' / 'phidown'}\"" in result.stdout
    assert "/bad/phidown" not in result.stdout


def test_download_helper_uses_local_prefix_phidown(tmp_path: Path):
    env_prefix = tmp_path / "env"
    phidown = env_prefix / "bin" / "phidown"
    phidown.parent.mkdir(parents=True)
    log = tmp_path / "phidown.log"
    phidown.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$0 $*\" > \"$SARPYX_TEST_PHIDOWN_LOG\"\n",
        encoding="utf-8",
    )
    phidown.chmod(0o755)

    env = os.environ.copy()
    env["ENV_PREFIX"] = str(env_prefix)
    env["PRODUCT_NAME"] = "S1_PRODUCT.SAFE"
    env["DOWNLOAD_DIR"] = str(tmp_path / "input_data")
    env["SARPYX_TEST_PHIDOWN_LOG"] = str(log)

    result = subprocess.run(
        ["bash", "scripts/download.sh"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").startswith(str(phidown))
