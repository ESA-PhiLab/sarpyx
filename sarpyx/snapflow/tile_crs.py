"""CRS helpers for WorldSAR grid tiling."""

from __future__ import annotations

import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pyproj

from sarpyx.snapflow.dimap import find_required, indent_xml, materialized_band_names, validate_same_dimensions
from sarpyx.snapflow.gpt import run_gpt_op
from sarpyx.snapflow.raster import _read_crs_wkt, _read_geotransform

SUBAP_IQ_RE = re.compile(
    r"^(?P<part>[iq])_(?:(?P<swath>[A-Z]{2}\d)_)?(?P<pol>HH|HV|VH|VV)_SA(?P<sa>\d+)$",
    re.IGNORECASE,
)


def rectangle_epsg(rectangle: dict) -> int:
    return int(str(rectangle["BL"]["properties"]["epsg"]).split(":")[-1])


def group_rectangles_by_epsg(rectangles: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for rectangle in rectangles:
        grouped.setdefault(rectangle_epsg(rectangle), []).append(rectangle)
    return dict(sorted(grouped.items()))


def dim_epsg(dim_path: Path) -> int | None:
    return pyproj.CRS.from_wkt(_read_crs_wkt(dim_path)).to_epsg()


def prepare_products_by_epsg(
    intermediate_product: Path,
    target_epsgs: list[int],
    gpt_kwargs: dict,
    terrain_correction: dict | None = None,
) -> dict[int, Path]:
    source_epsg = dim_epsg(intermediate_product)
    if source_epsg is None:
        raise RuntimeError(f"Could not resolve EPSG from {intermediate_product}")

    products: dict[int, Path] = {}
    for epsg in sorted(set(target_epsgs)):
        if epsg == source_epsg:
            if terrain_correction:
                source_product = Path(terrain_correction["source_product"])
                params = dict(terrain_correction.get("params") or {})
                params.pop("output_name", None)
                params["map_projection"] = f"EPSG:{epsg}"
                _ensure_subap_bands_terrain_corrected(source_product, intermediate_product, epsg, params, gpt_kwargs)
            products[epsg] = intermediate_product
            continue
        if terrain_correction:
            products[epsg] = _terrain_correct_to_epsg(intermediate_product, epsg, terrain_correction, gpt_kwargs)
            continue
        existing = _find_existing_epsg_product(intermediate_product, epsg)
        if existing:
            print(f"Reusing existing WorldSAR EPSG:{epsg} intermediate: {existing}")
            products[epsg] = existing
            continue
        products[epsg] = _reproject_to_epsg(intermediate_product, epsg, gpt_kwargs)
    return products


def _find_existing_epsg_product(intermediate_product: Path, epsg: int) -> Path | None:
    candidates = []
    for folder_name in ("worldsar_tc_epsg", "worldsar_reprojected"):
        folder = intermediate_product.parent / folder_name
        if folder.is_dir():
            candidates.extend(folder.glob(f"*EPSG{epsg}.dim"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _terrain_correct_to_epsg(intermediate_product: Path, epsg: int, terrain_correction: dict, gpt_kwargs: dict) -> Path:
    source_product = Path(terrain_correction["source_product"])
    params = dict(terrain_correction.get("params") or {})
    params.pop("output_name", None)
    params["map_projection"] = f"EPSG:{epsg}"
    outdir = Path(terrain_correction.get("output_dir") or intermediate_product.parent) / "worldsar_tc_epsg"
    output_name = f"{source_product.stem}_TC_EPSG{epsg}"
    output_path = outdir / f"{output_name}.dim"
    if output_path.exists():
        print(f"Reusing WorldSAR EPSG:{epsg} Terrain-Correction intermediate: {output_path}")
        _ensure_subap_bands_terrain_corrected(source_product, output_path, epsg, params, gpt_kwargs)
        return output_path
    print(f"Creating WorldSAR Terrain-Correction intermediate for EPSG:{epsg}: {source_product}")
    product = run_gpt_op(
        source_product,
        outdir,
        "BEAM-DIMAP",
        "TerrainCorrection",
        output_name=output_name,
        **params,
        **gpt_kwargs,
    )
    _ensure_subap_bands_terrain_corrected(source_product, product, epsg, params, gpt_kwargs)
    return product


def _ensure_subap_bands_terrain_corrected(
    source_product: Path,
    target_product: Path,
    epsg: int,
    terrain_params: dict,
    gpt_kwargs: dict,
) -> None:
    subap_groups = _subap_source_groups(source_product)
    if not subap_groups:
        return
    target_bands = set(materialized_band_names(target_product))
    missing_groups = {
        sa: mapping
        for sa, mapping in subap_groups.items()
        if any(raw_name not in target_bands for raw_name in mapping.values())
    }
    if not missing_groups:
        return
    workdir = target_product.parent / "worldsar_subap_tc"
    workdir.mkdir(parents=True, exist_ok=True)
    for sa, redirect in missing_groups.items():
        redirect_product = _write_subap_redirect_product(source_product, workdir, sa, redirect)
        output_name = f"{source_product.stem}_SA{sa}_TC_EPSG{epsg}"
        subap_tc = workdir / f"{output_name}.dim"
        if not subap_tc.exists():
            params = dict(terrain_params)
            params["map_projection"] = f"EPSG:{epsg}"
            params["source_bands"] = sorted(redirect)
            print(f"Creating WorldSAR subap TC intermediate for SA{sa}, EPSG:{epsg}: {source_product}")
            subap_tc = run_gpt_op(
                redirect_product,
                workdir,
                "BEAM-DIMAP",
                "TerrainCorrection",
                output_name=output_name,
                **params,
                **gpt_kwargs,
            )
        _merge_subap_tc_bands(target_product, subap_tc, redirect)


def _subap_source_groups(source_product: Path) -> dict[int, dict[str, str]]:
    groups: dict[int, dict[str, str]] = {}
    for raw_name in materialized_band_names(source_product):
        match = SUBAP_IQ_RE.match(raw_name)
        if not match:
            continue
        swath = f"{match.group('swath').upper()}_" if match.group("swath") else ""
        base_name = f"{match.group('part').lower()}_{swath}{match.group('pol').upper()}"
        groups.setdefault(int(match.group("sa")), {})[base_name] = raw_name
    return {sa: mapping for sa, mapping in sorted(groups.items()) if _complete_iq_pairs(mapping)}


def _complete_iq_pairs(mapping: dict[str, str]) -> bool:
    prefixes = {}
    for base_name in mapping:
        part, suffix = base_name.split("_", 1)
        prefixes.setdefault(suffix, set()).add(part)
    return bool(prefixes) and all({"i", "q"} <= parts for parts in prefixes.values())


def _write_subap_redirect_product(source_product: Path, workdir: Path, sa: int, redirect: dict[str, str]) -> Path:
    redirect_product = workdir / f"{source_product.stem}_SA{sa}_source.dim"
    source_data_dir = source_product.with_suffix(".data")
    redirect_data_dir = redirect_product.with_suffix(".data")
    if redirect_product.exists():
        redirect_product.unlink()
    if redirect_data_dir.exists():
        shutil.rmtree(redirect_data_dir)
    redirect_data_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source_product)
    root = tree.getroot()
    _rebase_hrefs(root, source_product.parent, redirect_product.parent)
    band_name_by_index = {
        band.findtext("BAND_INDEX"): (band.findtext("BAND_NAME") or "").strip()
        for band in root.findall("./Image_Interpretation/Spectral_Band_Info")
    }
    image_interpretation = find_required(root, "Image_Interpretation")
    data_access = find_required(root, "Data_Access")
    subap_indices = {index for index, band_name in band_name_by_index.items() if "_SA" in band_name}
    for band in list(image_interpretation.findall("Spectral_Band_Info")):
        if "_SA" in (band.findtext("BAND_NAME") or ""):
            image_interpretation.remove(band)
    for data_file in list(data_access.findall("Data_File")):
        band_index = data_file.findtext("BAND_INDEX")
        band_name = band_name_by_index.get(band_index)
        href = data_file.find("DATA_FILE_PATH")
        if band_index in subap_indices:
            data_access.remove(data_file)
            continue
        if band_name in redirect and href is not None:
            _copy_band_files(source_data_dir, redirect_data_dir, redirect[band_name], redirect[band_name])
            href.set("href", _relative_href(redirect_data_dir / f"{redirect[band_name]}.hdr", redirect_product.parent))
    _set_nbands(root)
    indent_xml(root)
    tree.write(redirect_product, encoding="UTF-8", xml_declaration=False)
    return redirect_product


def _rebase_hrefs(root: ET.Element, source_base: Path, target_base: Path) -> None:
    for element in root.findall(".//*[@href]"):
        href = element.get("href")
        if not href:
            continue
        source_path = Path(href)
        if not source_path.is_absolute():
            source_path = source_base / source_path
        element.set("href", _relative_href(source_path, target_base))


def _relative_href(path: Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def _merge_subap_tc_bands(target_product: Path, subap_tc_product: Path, redirect: dict[str, str]) -> None:
    validate_same_dimensions(subap_tc_product, target_product)
    target_tree = ET.parse(target_product)
    target_root = target_tree.getroot()
    subap_root = ET.parse(subap_tc_product).getroot()
    target_image = find_required(target_root, "Image_Interpretation")
    target_data_access = find_required(target_root, "Data_Access")
    existing = {
        (band.findtext("BAND_NAME") or "").strip()
        for band in target_root.findall("./Image_Interpretation/Spectral_Band_Info")
    }
    next_index = _next_band_index(target_root)
    target_data_dir = target_product.with_suffix(".data")
    subap_data_dir = subap_tc_product.with_suffix(".data")
    for base_name, raw_name in sorted(redirect.items()):
        if raw_name in existing:
            continue
        source_band = _spectral_band(subap_root, base_name)
        if source_band is None:
            continue
        _copy_band_files(subap_data_dir, target_data_dir, base_name, raw_name)
        cloned = _clone_band_as(source_band, next_index, raw_name)
        target_image.append(cloned)
        data_file = ET.Element("Data_File")
        ET.SubElement(data_file, "BAND_INDEX").text = str(next_index)
        ET.SubElement(data_file, "DATA_FILE_PATH", href=f"{target_data_dir.name}/{raw_name}.hdr")
        target_data_access.append(data_file)
        existing.add(raw_name)
        next_index += 1
    _set_nbands(target_root)
    indent_xml(target_root)
    target_tree.write(target_product, encoding="UTF-8", xml_declaration=False)


def _copy_band_files(src_data_dir: Path, dst_data_dir: Path, src_name: str, dst_name: str) -> None:
    dst_data_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("hdr", "img"):
        src = src_data_dir / f"{src_name}.{ext}"
        dst = dst_data_dir / f"{dst_name}.{ext}"
        if dst.exists():
            continue
        if ext == "img":
            try:
                dst.hardlink_to(src)
                shutil.copystat(src, dst)
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)


def _spectral_band(root: ET.Element, band_name: str) -> ET.Element | None:
    for band in root.findall("./Image_Interpretation/Spectral_Band_Info"):
        if (band.findtext("BAND_NAME") or "").strip() == band_name:
            return band
    return None


def _clone_band_as(source_band: ET.Element, band_index: int, band_name: str) -> ET.Element:
    cloned = ET.fromstring(ET.tostring(source_band, encoding="unicode"))
    index = cloned.find("BAND_INDEX")
    if index is None:
        index = ET.SubElement(cloned, "BAND_INDEX")
    index.text = str(band_index)
    name = cloned.find("BAND_NAME")
    if name is None:
        name = ET.SubElement(cloned, "BAND_NAME")
    name.text = band_name
    description = cloned.find("BAND_DESCRIPTION")
    if description is None:
        description = ET.SubElement(cloned, "BAND_DESCRIPTION")
    description.text = band_name
    return cloned


def _next_band_index(root: ET.Element) -> int:
    indices = []
    for band in root.findall("./Image_Interpretation/Spectral_Band_Info"):
        value = band.findtext("BAND_INDEX")
        if value is not None and value.strip().isdigit():
            indices.append(int(value.strip()))
    return max(indices, default=-1) + 1


def _set_nbands(root: ET.Element) -> None:
    raster_dimensions = find_required(root, ".//Raster_Dimensions")
    nbands = find_required(raster_dimensions, "NBANDS")
    nbands.text = str(len(root.findall("./Image_Interpretation/Spectral_Band_Info")))


def _reproject_to_epsg(intermediate_product: Path, epsg: int, gpt_kwargs: dict) -> Path:
    pixel_size_x, pixel_size_y = _pixel_sizes(intermediate_product)
    outdir = intermediate_product.parent / "worldsar_reprojected"
    output_name = f"{intermediate_product.stem}_EPSG{epsg}"
    output_path = outdir / f"{output_name}.dim"
    if output_path.exists():
        print(f"Reusing WorldSAR EPSG:{epsg} intermediate: {output_path}")
        return output_path
    print(f"Reprojecting WorldSAR intermediate to EPSG:{epsg}: {intermediate_product}")
    return run_gpt_op(
        intermediate_product,
        outdir,
        "BEAM-DIMAP",
        "reproject",
        crs=f"EPSG:{epsg}",
        resampling="Nearest",
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        output_name=output_name,
        **gpt_kwargs,
    )


def _pixel_sizes(dim_path: Path) -> tuple[float, float]:
    _, px_w, rot_x, _, rot_y, px_h = _read_geotransform(dim_path)
    if rot_x or rot_y:
        raise RuntimeError(f"Rotated geotransforms are not supported for WorldSAR tiling: {dim_path}")
    return abs(px_w), abs(px_h)
