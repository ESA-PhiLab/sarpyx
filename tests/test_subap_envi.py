from __future__ import annotations

import errno

import numpy as np
import pytest


def _write_envi_float32(path_img, path_hdr, arr2d, band_name, byte_order=1):
    arr = np.asarray(arr2d, dtype=np.float32)
    arr_out = arr.astype(">f4") if byte_order == 1 else arr.astype("<f4")
    arr_out.tofile(path_img)
    lines, samples = arr.shape
    hdr = (
        f"ENVI\nsamples = {samples}\nlines = {lines}\nbands = 1\n"
        f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        f"interleave = bsq\nbyte order = {byte_order}\n"
        f"band names = {{ {band_name} }}\n"
    )
    path_hdr.write_text(hdr, encoding="ascii")


def test_write_envi_bsq_float32_streams_expected_bytes(tmp_path):
    from sarpyx.processor.core.subap_envi import write_envi_bsq_float32

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    img = tmp_path / "band.img"
    hdr = tmp_path / "band.hdr"

    write_envi_bsq_float32(img, hdr, arr, band_name="band", byte_order=0)

    assert img.stat().st_size == arr.size * 4
    np.testing.assert_array_equal(np.fromfile(img, dtype="<f4").reshape(arr.shape), arr)
    assert "band names = { band }" in hdr.read_text(encoding="ascii")


def test_write_envi_bsq_float32_reports_low_space(monkeypatch, tmp_path):
    from sarpyx.processor.core import subap_envi

    img = tmp_path / "band.img"
    hdr = tmp_path / "band.hdr"
    monkeypatch.setattr(subap_envi, "_free_bytes", lambda _path: 1)

    with pytest.raises(OSError, match="Not enough free space") as excinfo:
        subap_envi.write_envi_bsq_float32(img, hdr, np.ones((2, 2), dtype=np.float32), band_name="band")

    assert excinfo.value.errno == errno.ENOSPC
    assert not img.exists()
    assert not hdr.exists()


def test_estimate_subap_output_bytes_accounts_for_existing_files(tmp_path):
    from sarpyx.processor.core.subap_envi import _estimate_subap_output_bytes

    i_path = tmp_path / "i_VV.img"
    q_path = tmp_path / "q_VV.img"
    _write_envi_float32(i_path, tmp_path / "i_VV.hdr", np.ones((5, 7), dtype=np.float32), "i_VV")
    _write_envi_float32(q_path, tmp_path / "q_VV.hdr", np.ones((5, 7), dtype=np.float32), "q_VV")
    band_bytes = 5 * 7 * 4

    assert _estimate_subap_output_bytes(str(tmp_path), {"VV": (str(i_path), str(q_path))}, [2]) == band_bytes * 4

    (tmp_path / "i_VV_SA1.img").write_bytes(b"0" * band_bytes)

    assert _estimate_subap_output_bytes(str(tmp_path), {"VV": (str(i_path), str(q_path))}, [2]) == band_bytes * 3


def test_ensure_subap_output_space_reports_target_dir(monkeypatch, tmp_path):
    from sarpyx.processor.core import subap_envi

    monkeypatch.setattr(subap_envi, "_free_bytes", lambda _path: 10)

    with pytest.raises(OSError, match="target_dir") as excinfo:
        subap_envi._ensure_subap_output_space(str(tmp_path), 20)

    assert excinfo.value.errno == errno.ENOSPC
