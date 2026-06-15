"""BEAM-DIMAP helpers for WorldSAR metadata merge workflows."""

from __future__ import annotations

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def find_required(parent: ET.Element, tag: str) -> ET.Element:
    element = parent.find(tag)
    if element is None:
        raise RuntimeError(f"Missing <{tag}> inside <{parent.tag}>.")
    return element


def text_required(parent: ET.Element, tag: str) -> str:
    element = find_required(parent, tag)
    if element.text is None:
        raise RuntimeError(f"Tag <{tag}> inside <{parent.tag}> has no text.")
    return element.text.strip()


def get_data_dir_from_dim(dim_path: Path) -> Path:
    return dim_path.with_suffix("").with_name(dim_path.stem + ".data")


def already_has_band(image_interpretation: ET.Element, band_name: str) -> bool:
    for spectral_band in image_interpretation.findall("Spectral_Band_Info"):
        if spectral_band.findtext("BAND_NAME", default="").strip() == band_name:
            return True
    return False


def detect_suffixes_from_src_data(src_data_dir: Path) -> list[str]:
    i_suffixes: set[str] = set()
    q_suffixes: set[str] = set()
    for hdr_path in src_data_dir.glob("*.hdr"):
        stem = hdr_path.stem
        if stem.startswith("i_"):
            i_suffixes.add(stem[2:])
        elif stem.startswith("q_"):
            q_suffixes.add(stem[2:])
    return sorted(i_suffixes & q_suffixes)


def build_band_plan(suffixes: list[str], start_index: int) -> list[dict]:
    bands: list[dict] = []
    index = start_index
    for suffix in suffixes:
        bands.extend(
            [
                {"band_index": index, "band_name": f"i_{suffix}", "file_name": f"i_{suffix}.hdr", "physical_unit": "real", "virtual": False, "expr": None},
                {"band_index": index + 1, "band_name": f"q_{suffix}", "file_name": f"q_{suffix}.hdr", "physical_unit": "imaginary", "virtual": False, "expr": None},
                {
                    "band_index": index + 2,
                    "band_name": f"Intensity_{suffix}",
                    "file_name": None,
                    "physical_unit": "intensity",
                    "virtual": True,
                    "expr": f"i_{suffix} == 0.0 ? 0.0 : i_{suffix} * i_{suffix} + q_{suffix} * q_{suffix}",
                },
            ]
        )
        index += 3
    return bands


def build_data_file(href: str, band_index: int) -> ET.Element:
    data_file = ET.Element("Data_File")
    ET.SubElement(data_file, "BAND_INDEX").text = str(band_index)
    ET.SubElement(data_file, "DATA_FILE_PATH", href=href)
    return data_file


def build_spectral_band_info(
    band_index: int,
    band_name: str,
    width: str,
    height: str,
    physical_unit: str,
    virtual: bool,
    expr: str | None,
) -> ET.Element:
    spectral = ET.Element("Spectral_Band_Info")
    ET.SubElement(spectral, "BAND_INDEX").text = str(band_index)
    ET.SubElement(spectral, "BAND_NAME").text = band_name
    ET.SubElement(spectral, "BAND_DESCRIPTION").text = band_name
    ET.SubElement(spectral, "PHYSICAL_UNIT").text = physical_unit
    ET.SubElement(spectral, "BAND_RASTER_WIDTH").text = str(width)
    ET.SubElement(spectral, "BAND_RASTER_HEIGHT").text = str(height)
    ET.SubElement(spectral, "BAND_SCALING_FACTOR").text = "1.0"
    ET.SubElement(spectral, "BAND_SCALING_OFFSET").text = "0.0"
    ET.SubElement(spectral, "BAND_LOG10_SCALED").text = "false"
    if virtual and expr:
        ET.SubElement(spectral, "VIRTUAL_BAND").text = "true"
        ET.SubElement(spectral, "EXPRESSION").text = expr
    return spectral


def clone_crs_geoposition_pair(root: ET.Element) -> tuple[ET.Element | None, ET.Element | None]:
    raster_data_node = root.find(".//RasterDataNode")
    if raster_data_node is None:
        return None, None
    crs = raster_data_node.find("Coordinate_Reference_System")
    geoposition = raster_data_node.find("Geoposition")
    return crs, geoposition


def copy_src_files(src_data_dir: Path, pdec_data_dir: Path, suffixes: list[str], overwrite: bool) -> None:
    for suffix in suffixes:
        for prefix in ("i", "q"):
            for ext in ("hdr", "img"):
                src = src_data_dir / f"{prefix}_{suffix}.{ext}"
                dst = pdec_data_dir / src.name
                if not src.exists():
                    raise FileNotFoundError(f"Expected source file missing: {src}")
                copy_or_link_raster(src, dst, overwrite=overwrite, hardlink=ext == "img")


def materialized_band_names(dim_path: Path) -> list[str]:
    root = ET.parse(dim_path).getroot()
    names = []
    for data_file in root.findall(".//Data_File"):
        href = data_file.find("DATA_FILE_PATH")
        if href is not None and href.get("href"):
            names.append(Path(href.get("href", "")).stem)
    return sorted(set(names))


def spectral_band_names(dim_path: Path) -> list[str]:
    root = ET.parse(dim_path).getroot()
    names = []
    for spectral_band in root.findall("./Image_Interpretation/Spectral_Band_Info"):
        band_name = spectral_band.findtext("BAND_NAME", default="").strip()
        if band_name:
            names.append(band_name)
    return names


def copy_materialized_src_files(src_data_dir: Path, pdec_data_dir: Path, band_names: list[str], overwrite: bool) -> None:
    for band_name in band_names:
        for ext in ("hdr", "img"):
            src = src_data_dir / f"{band_name}.{ext}"
            if not src.exists():
                if ext == "hdr":
                    raise FileNotFoundError(f"Expected source raster header missing: {src}")
                continue
            dst = pdec_data_dir / src.name
            copy_or_link_raster(src, dst, overwrite=overwrite, hardlink=ext == "img")


def copy_or_link_raster(src: Path, dst: Path, overwrite: bool, hardlink: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    if hardlink:
        try:
            dst.hardlink_to(src)
            shutil.copystat(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def clone_source_raster_metadata(src_dim: Path, pdec_dim: Path, band_names: list[str], backup: bool = False) -> None:
    if backup:
        shutil.copy2(pdec_dim, pdec_dim.with_suffix(pdec_dim.suffix + ".bak"))
    src_root = ET.parse(src_dim).getroot()
    tree = ET.parse(pdec_dim)
    root = tree.getroot()
    image_interpretation = find_required(root, "Image_Interpretation")
    data_access = find_required(root, "Data_Access")
    existing = {band.findtext("BAND_NAME", default="").strip() for band in image_interpretation.findall("Spectral_Band_Info")}
    next_index = _next_band_index(root)
    pdec_data_dir_name = pdec_dim.with_suffix("").name + ".data"
    for source_band in src_root.findall("./Image_Interpretation/Spectral_Band_Info"):
        band_name = source_band.findtext("BAND_NAME", default="").strip()
        if not band_name or band_name in existing or band_name not in band_names:
            continue
        image_interpretation.append(_clone_spectral_band(source_band, next_index))
        data_access.append(build_data_file(f"{pdec_data_dir_name}/{band_name}.hdr", next_index))
        existing.add(band_name)
        next_index += 1
    _set_nbands(root)
    indent_xml(root)
    tree.write(pdec_dim, encoding="UTF-8", xml_declaration=False)


def _clone_spectral_band(source_band: ET.Element, band_index: int) -> ET.Element:
    cloned = copy.deepcopy(source_band)
    index = cloned.find("BAND_INDEX")
    if index is None:
        index = ET.SubElement(cloned, "BAND_INDEX")
    index.text = str(band_index)
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


def validate_same_dimensions(src_dim: Path, pdec_dim: Path) -> None:
    def dims(path: Path) -> tuple[str, str]:
        root = ET.parse(path).getroot()
        raster = find_required(root, ".//Raster_Dimensions")
        return text_required(raster, "NCOLS"), text_required(raster, "NROWS")

    if dims(src_dim) != dims(pdec_dim):
        raise RuntimeError(f"SRC/PDEC raster dimensions differ: {src_dim} vs {pdec_dim}")


def edit_pdec_dim(pdec_dim: Path, suffixes: list[str], is_tops: bool = False, backup: bool = True) -> None:
    if backup:
        shutil.copy2(pdec_dim, pdec_dim.with_suffix(pdec_dim.suffix + ".bak"))
    tree = ET.parse(pdec_dim)
    root = tree.getroot()
    raster_dimensions = find_required(root, ".//Raster_Dimensions")
    ncols = text_required(raster_dimensions, "NCOLS")
    nrows = text_required(raster_dimensions, "NROWS")
    image_interpretation = find_required(root, "Image_Interpretation")
    data_access = find_required(root, "Data_Access")
    start_index = _next_band_index(root)
    bands_to_add = [
        band for band in build_band_plan(suffixes, start_index)
        if not already_has_band(image_interpretation, band["band_name"])
    ]
    for offset, band in enumerate(bands_to_add):
        band["band_index"] = start_index + offset
    pdec_data_dir_name = pdec_dim.with_suffix("").name + ".data"
    for band in bands_to_add:
        if not band["virtual"]:
            href = f"{pdec_data_dir_name}/{band['file_name']}"
            data_access.append(build_data_file(href, band["band_index"]))
    for band in bands_to_add:
        image_interpretation.append(
            build_spectral_band_info(
                band_index=band["band_index"],
                band_name=band["band_name"],
                width=ncols,
                height=nrows,
                physical_unit=band["physical_unit"],
                virtual=band["virtual"],
                expr=band["expr"],
            )
        )
    _set_nbands(root)
    indent_xml(root)
    tree.write(pdec_dim, encoding="UTF-8", xml_declaration=False)
