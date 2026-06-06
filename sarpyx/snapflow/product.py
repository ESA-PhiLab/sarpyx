"""Product identification and footprint resolution for WorldSAR."""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from sarpyx.utils.wkt_utils import (
    nisar_wkt_extractor,
    sentinel1_wkt_extractor_cdse,
    sentinel1_wkt_extractor_manifest,
    terrasar_wkt_extractor,
)
from sarpyx.snapflow.config import _env

TERRASAR_GEOCODED_VARIANTS = frozenset({"EEC", "GEC"})
TERRASAR_COMPLEX_VARIANTS = frozenset({"SSC", "SLC"})
TERRASAR_DETECTED_VARIANTS = frozenset({"MGD", "GRD"})


def extract_product_id(path: str) -> str | None:
    match = re.search(r"/([^/]+?)_[^/_]+\.dim$", path)
    return match.group(1) if match else None


def _is_terrasar_product(product_path) -> bool:
    as_path = Path(product_path).as_posix().upper()
    return any(token in as_path for token in ("TSX", "TDX", "TERRASAR", "TANDEMX"))


def _xml_has_scene_corners(xml_path: Path) -> bool:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return False
    corners = root.findall(".//sceneCornerCoord")
    return sum(1 for c in corners if c.findtext("lon") is not None and c.findtext("lat") is not None) >= 3


def _safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (target_dir / member.filename).resolve()
            if target_root != member_path and target_root not in member_path.parents:
                raise ValueError(f"Unsafe ZIP member path in {archive_path}: {member.filename}")
        archive.extractall(target_dir)


def _terrasar_archive_extract_dir(archive_path: Path) -> Path:
    stat = archive_path.stat()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", archive_path.stem)
    return archive_path.parent / ".sarpyx_extracted" / f"{safe_stem}_{stat.st_size}_{stat.st_mtime_ns}"


def _extract_terrasar_archive(archive_path: Path) -> Path:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"TerraSAR-X/TanDEM-X archive is not a valid ZIP file: {archive_path}")
    extract_dir = _terrasar_archive_extract_dir(archive_path)
    marker = extract_dir / ".sarpyx_extract_complete"
    if marker.exists():
        return extract_dir
    _safe_extract_zip(archive_path, extract_dir)
    for nested_zip in sorted(extract_dir.rglob("*.zip")) + sorted(extract_dir.rglob("*.ZIP")):
        if nested_zip == archive_path or not nested_zip.is_file():
            continue
        nested_target = nested_zip.with_suffix("")
        nested_marker = nested_target / ".sarpyx_extract_complete"
        if nested_marker.exists():
            continue
        _safe_extract_zip(nested_zip, nested_target)
        nested_marker.write_text("ok\n", encoding="utf-8")
    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def _resolve_terrasar_product_xml(product_path) -> Path:
    product_path = Path(product_path)
    if product_path.is_file():
        suffix = product_path.suffix.lower()
        if suffix == ".xml":
            return product_path
        if suffix == ".zip":
            return _resolve_terrasar_product_xml(_extract_terrasar_archive(product_path))
        raise ValueError("TerraSAR-X/TanDEM-X products must be an XML metadata file, ZIP archive, or directory")
    if not product_path.is_dir():
        raise FileNotFoundError(f"TerraSAR-X/TanDEM-X product path does not exist: {product_path}")
    candidates = sorted(path for path in product_path.rglob("*.xml") if _xml_has_scene_corners(path))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No TerraSAR-X/TanDEM-X metadata XML with sceneCornerCoord found under {product_path}."
        )
    raise ValueError(
        "Multiple TerraSAR-X/TanDEM-X metadata XML files with sceneCornerCoord found under "
        f"{product_path}: {', '.join(str(path) for path in candidates)}."
    )


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_first_text(root: ET.Element, *local_names: str) -> str | None:
    wanted = {name.lower() for name in local_names}
    for element in root.iter():
        if _xml_local_name(element.tag).lower() in wanted and element.text is not None:
            value = element.text.strip()
            if value:
                return value
    return None


def _normalize_metadata_token(value: str | None) -> str | None:
    normalized = value.strip().upper() if value is not None else None
    return normalized or None


def _variant_from_product_type(product_type: str | None) -> str | None:
    product_type = _normalize_metadata_token(product_type)
    return product_type.split("_", 1)[0] if product_type else None


def _read_terrasar_metadata(product_xml: Path) -> dict[str, str | None]:
    root = ET.parse(product_xml).getroot()
    product_type = _xml_first_text(root, "productType")
    product_variant = _xml_first_text(root, "productVariant")
    return {
        "mission": _normalize_metadata_token(_xml_first_text(root, "mission", "platform", "shortName")),
        "variant": _normalize_metadata_token(product_variant) or _variant_from_product_type(product_type),
        "product_type": _normalize_metadata_token(product_type),
        "imaging_mode": _normalize_metadata_token(_xml_first_text(root, "imagingMode", "operationalMode")),
        "image_data_type": _normalize_metadata_token(_xml_first_text(root, "imageDataType")),
        "projection": _normalize_metadata_token(_xml_first_text(root, "projection")),
        "map_projection": _normalize_metadata_token(_xml_first_text(root, "mapProjection")),
        "radiometric_correction": _normalize_metadata_token(_xml_first_text(root, "radiometricCorrection")),
    }


def _terrasar_product_variant(product_xml: Path) -> str | None:
    return _read_terrasar_metadata(product_xml)["variant"]


def _terrasar_is_geocoded(metadata: dict[str, str | None]) -> bool:
    return (
        metadata.get("variant") in TERRASAR_GEOCODED_VARIANTS
        or metadata.get("projection") == "MAP"
        or metadata.get("map_projection") is not None
    )


def _terrasar_is_complex(metadata: dict[str, str | None]) -> bool:
    return metadata.get("variant") in TERRASAR_COMPLEX_VARIANTS or metadata.get("image_data_type") == "COMPLEX"


def _terrasar_is_detected(metadata: dict[str, str | None]) -> bool:
    return metadata.get("variant") in TERRASAR_DETECTED_VARIANTS or metadata.get("image_data_type") == "DETECTED"


def _metadata_indicates_terrasar(metadata: dict[str, str | None]) -> bool:
    mission = metadata.get("mission") or ""
    return (
        any(token in mission for token in ("TSX", "TDX", "TERRASAR", "TANDEMX"))
        or metadata.get("variant") in (TERRASAR_GEOCODED_VARIANTS | TERRASAR_COMPLEX_VARIANTS | TERRASAR_DETECTED_VARIANTS)
        or bool(metadata.get("product_type"))
    )


def _resolve_terrasar_product_xml_if_supported(product_path) -> Path | None:
    if _is_terrasar_product(product_path):
        return _resolve_terrasar_product_xml(product_path)
    try:
        candidate = _resolve_terrasar_product_xml(product_path)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile, ET.ParseError):
        return None
    return candidate if _metadata_indicates_terrasar(_read_terrasar_metadata(candidate)) else None


def infer_product_mode(product_path: Path) -> str:
    name = product_path.name.upper()
    stem = product_path.stem.upper()
    as_path = product_path.as_posix().upper()
    if "NISAR" in as_path or ("GSLC" in as_path and product_path.suffix.lower() == ".h5"):
        return "NISAR"
    if any(t in as_path for t in ("TSX", "TDX", "TERRASAR", "TANDEMX")):
        return "TSX"
    if any(t in as_path for t in ("CSG", "CSK", "COSMO")):
        return "CSG"
    if any(t in as_path for t in ("BIOMASS", "/BIO", "_BIO", "-BIO")):
        return "BM"
    if re.search(r"(?:^|[^A-Z0-9])S1[ABC](?:_|[^A-Z0-9])", as_path):
        mode_match = re.search(r"S1[ABC]_([A-Z0-9]{2})_", stem)
        mode_token = mode_match.group(1) if mode_match else None
        if mode_token in {"IW", "EW"}:
            return "S1TOPS"
        if mode_token in {"SM", "S1", "S2", "S3", "S4", "S5", "S6"}:
            return "S1STRIP"
        if "_IW_" in name or "_EW_" in name or "TOPS" in name:
            return "S1TOPS"
        return "S1TOPS"
    if product_path.exists() and _resolve_terrasar_product_xml_if_supported(product_path) is not None:
        return "TSX"
    raise ValueError(f"Could not infer product mode from input path: {product_path}.")


def resolve_product_wkt(args, product_path, product_mode) -> str:
    product_wkt_value = args.product_wkt if args.product_wkt is not None else _env("PRODUCT_WKT", "product_wkt")
    if product_wkt_value is not None:
        product_wkt = product_wkt_value.strip()
        if not product_wkt:
            raise ValueError("--product-wkt/PRODUCT_WKT cannot be blank.")
        return product_wkt
    if product_mode in {"S1TOPS", "S1STRIP"}:
        product_wkt = sentinel1_wkt_extractor_manifest(product_path, display_results=False)
        product_wkt = product_wkt or sentinel1_wkt_extractor_cdse(product_path.name, display_results=False)
        if product_wkt is None:
            raise ValueError(f"Failed to extract Sentinel-1 WKT for product: {product_path}")
        return product_wkt
    if product_mode == "NISAR":
        return nisar_wkt_extractor(product_path)
    if product_mode == "TSX":
        return terrasar_wkt_extractor(_resolve_terrasar_product_xml(product_path))
    raise ValueError("No --product-wkt/PRODUCT_WKT provided and automatic WKT extraction is unavailable.")
