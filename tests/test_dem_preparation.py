"""Offline regression tests for catalogue-aware DEM preparation."""
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from sarpyx.utils import dem_utils as dem

WKT = "POLYGON ((139.1 36.1, 139.2 36.1, 139.2 36.2, 139.1 36.2, 139.1 36.1))"
NAME = dem.tile_name(36, 139)
OTHER = dem.tile_name(0, 6)


class DemPreparationTests(unittest.TestCase):
    def catalogue(self, root, names=(NAME,)):
        path = root / "tileList.txt"
        path.write_text("\r\n".join(names) + "\r\n")
        return path

    def test_cached_tiles_need_no_network_or_catalogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / f"{NAME}.tif"
            target.write_bytes(b"valid")
            with patch.object(dem.urllib.request, "urlopen", side_effect=AssertionError("network")):
                results = dem.prepare_tiles_for_wkt(
                    WKT, root, catalogue_path=root / "nonexistent",
                    validate_tile=lambda p: self.assertEqual(p.read_bytes(), b"valid"),
                )
            self.assertEqual(results[0]["status"], "reused")
            self.assertEqual(list(root.iterdir()), [target])

    def test_missing_published_tile_fails_without_raster_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = self.catalogue(root)
            with patch.object(dem, "_download_one", side_effect=AssertionError("download")), \
                 self.assertRaisesRegex(FileNotFoundError, "Required published DEM tiles"):
                dem.prepare_tiles_for_wkt(WKT, root, catalogue_path=catalogue)
            self.assertEqual(list(root.iterdir()), [catalogue])

    def test_unpublished_cell_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = self.catalogue(root, (OTHER,))
            with patch.object(dem.urllib.request, "urlopen", side_effect=AssertionError("network")):
                results = dem.prepare_tiles_for_wkt(WKT, root, catalogue_path=catalogue, download=True)
            self.assertEqual(results[0]["status"], "not_in_catalogue")
            self.assertFalse((root / f"{NAME}.tif").exists())

    def test_opt_in_uses_shared_transfer_and_validates_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = self.catalogue(root)
            seen = []
            def validate(path):
                seen.append(path)
                self.assertEqual(path.read_bytes(), b"valid-raster")
                self.assertFalse((root / f"{NAME}.tif").exists())
            with patch.object(dem.urllib.request, "urlopen", return_value=io.BytesIO(b"valid-raster")) as transfer:
                results = dem.prepare_tiles_for_wkt(
                    WKT, root, catalogue_path=catalogue, download=True, validate_tile=validate,
                )
            self.assertEqual(results[0]["status"], "downloaded")
            self.assertEqual((root / f"{NAME}.tif").read_bytes(), b"valid-raster")
            self.assertEqual(len(seen), 1)
            self.assertTrue(seen[0].name.endswith(".part"))
            transfer.assert_called_once_with(dem.tile_url(36, 139), timeout=300)
            self.assertFalse(list(root.glob("*.part")))

    def test_published_404_is_error_not_ocean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = self.catalogue(root)
            error = urllib.error.HTTPError("test", 404, "Not found", {}, None)
            with patch.object(dem.urllib.request, "urlopen", side_effect=error), \
                 self.assertRaisesRegex(RuntimeError, "published DEM tile"):
                dem.prepare_tiles_for_wkt(WKT, root, catalogue_path=catalogue, download=True)
            self.assertFalse((root / f"{NAME}.tif").exists())
            self.assertEqual(list(root.iterdir()), [catalogue])

    def test_catalogue_outage_is_not_treated_as_ocean(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(dem.urllib.request, "urlopen", side_effect=TimeoutError("catalogue offline")), \
                 self.assertRaisesRegex(TimeoutError, "catalogue offline"):
                dem.prepare_tiles_for_wkt(WKT, tmp)

    def test_invalid_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = self.catalogue(root)
            def reject(path):
                raise ValueError("invalid DEM")
            with patch.object(dem.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")), \
                 self.assertRaisesRegex(RuntimeError, "invalid DEM"):
                dem.prepare_tiles_for_wkt(
                    WKT, root, catalogue_path=catalogue, download=True, validate_tile=reject,
                )
            self.assertEqual(list(root.iterdir()), [catalogue])

    def test_invalid_cached_tile_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / f"{NAME}.tif"
            target.write_bytes(b"bad")
            def reject(path):
                raise ValueError("bad cache")
            with patch.object(dem.urllib.request, "urlopen", side_effect=AssertionError("network")), \
                 self.assertRaisesRegex(ValueError, "bad cache"):
                dem.prepare_tiles_for_wkt(WKT, root, download=True, validate_tile=reject)
            self.assertEqual(target.read_bytes(), b"bad")

    def test_read_only_mode_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing"
            with self.assertRaises(FileNotFoundError):
                dem.prepare_tiles_for_wkt(WKT, root)
            self.assertFalse(root.exists())

    def test_invalid_catalogue_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for text in ("", "<html>service unavailable</html>", dem.tile_name(36, 139, 90)):
                catalogue = root / "tileList.txt"
                catalogue.write_text(text)
                with self.assertRaisesRegex(ValueError, "catalogue"):
                    dem.prepare_tiles_for_wkt(WKT, root, catalogue_path=catalogue)

    def test_live_catalogue_only_not_rasters(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(dem.urllib.request, "urlopen", return_value=io.BytesIO((OTHER + "\r\n").encode())) as get:
                results = dem.prepare_tiles_for_wkt(WKT, tmp)
            get.assert_called_once_with(dem._BASE_URLS[30] + "/tileList.txt", timeout=60)
            self.assertEqual(results[0]["status"], "not_in_catalogue")

    def test_integer_point_occupies_a_tile(self):
        self.assertEqual(dem.tiles_from_wkt("POINT (139 36)")[0][2], NAME)

    def test_legacy_downloader_keeps_list_return_and_shared_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / f"{NAME}.tif"
            with patch.object(dem, "_download_one", return_value=target) as transfer:
                paths = dem.download_tiles_for_wkt(WKT, tmp)
            self.assertEqual(paths, [target])
            transfer.assert_called_once_with(dem.tile_url(36, 139), target, overwrite=False)


if __name__ == "__main__":
    unittest.main()
