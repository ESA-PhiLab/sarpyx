import shlex
import sys

from sarpyx.snapflow.engine import GPT


def _gpt_for_command(tmp_path):
    product = tmp_path / "product.dim"
    product.write_text("<Dimap_Document/>", encoding="utf-8")
    return GPT(product=product, outdir=tmp_path / "out", gpt_path=sys.executable, timeout=10)


def test_execute_command_streams_and_captures_stdout(tmp_path, capsys):
    gpt = _gpt_for_command(tmp_path)
    code = (
        "import sys,time; "
        "sys.stdout.write('Writing...'); sys.stdout.flush(); "
        "[sys.stdout.write(f'.{i*10}%') or sys.stdout.flush() or time.sleep(0.01) for i in range(1,4)]; "
        "sys.stdout.write(' done.\\n'); sys.stdout.flush()"
    )
    gpt.current_cmd = [sys.executable, "-c", shlex.quote(code)]

    assert gpt._execute_command() is True

    captured = capsys.readouterr()
    assert "Writing....10%.20%.30% done." in captured.out
    assert "GPT Output:" not in captured.out
    assert gpt.last_stdout == "Writing....10%.20%.30% done.\n"
    assert gpt.last_stderr == ""
    assert gpt.last_returncode == 0


def test_execute_command_captures_failure_output(tmp_path, capsys):
    gpt = _gpt_for_command(tmp_path)
    code = (
        "import sys; "
        "sys.stdout.write('partial stdout'); sys.stdout.flush(); "
        "sys.stderr.write('partial stderr'); sys.stderr.flush(); "
        "raise SystemExit(3)"
    )
    gpt.current_cmd = [sys.executable, "-c", shlex.quote(code)]

    assert gpt._execute_command() is False

    captured = capsys.readouterr()
    assert "partial stdout" in captured.out
    assert "partial stderr" in captured.err
    assert gpt.last_stdout == "partial stdout"
    assert gpt.last_stderr == "partial stderr"
    assert gpt.last_returncode == 3
