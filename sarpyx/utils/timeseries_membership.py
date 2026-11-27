"""Use WorldSAR database.parquet/pairs.parquet grouping without copying rasters."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from sarpyx.utils.timeseries import fingerprint
from sarpyx.utils.timeseries_metadata import product_id

SERIES_SCHEMA = pa.schema([
    *[(name, pa.string()) for name in (
        "cell_id", "swath", "kind", "observation_id", "asset_id", "uri",
        "series_id", "modality_id", "modality", "acquisition_mode",
        "source_datetime", "pair_name", "master_datetime", "slave_datetime",
        "source_relative_path", "membership_source",
    )],
    ("height", pa.int64()), ("width", pa.int64()),
])


def _safe(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe source grouping ID: {value!r}")
    return value


class MembershipIndex:
    """Small projected index for the input acquisitions, not a spatial inference."""

    def __init__(self, database, pairs, acquisition_ids):
        self.ids = list(acquisition_ids)
        spellings = [name + suffix for name in self.ids for suffix in ("", ".SAFE", ".zip", ".SAFE.zip")]
        rows = pq.read_table(
            database,
            columns=["major_tom_id", "modality_id", "modality", "acquisition_mode",
                     "product_name", "datetime"],
            filters=[("modality", "=", "S1"), ("product_name", "in", spellings)],
        ).to_pylist()
        if not rows:
            raise ValueError("Input acquisitions have no memberships in database.parquet")
        found_ids = {product_id(row["product_name"]) for row in rows}
        missing = set(self.ids) - found_ids
        if missing:
            raise ValueError(f"Acquisitions missing from database.parquet: {sorted(missing)}")
        modality_ids = sorted({row["modality_id"] for row in rows})
        pair_rows = pq.read_table(
            pairs,
            columns=["series_id", "major_tom_id", "modality_id", "modality",
                     "acquisition_mode", "pair_name", "master_product_name",
                     "slave_product_name", "master_datetime", "slave_datetime",
                     "relative_path"],
            filters=[("modality", "=", "S1"), ("modality_id", "in", modality_ids)],
        ).to_pylist()
        series = {}
        for row in pair_rows:
            group = _safe(row["series_id"])
            previous = series.setdefault(row["modality_id"], group)
            if previous != group:
                raise ValueError(f"Conflicting series IDs for {row['modality_id']}")
        self.acquisitions = defaultdict(dict)
        self.pairs = defaultdict(dict)
        for row in rows:
            modality_id = _safe(row["modality_id"])
            group = series.get(modality_id)
            if group is None:
                # database.parquet has modality_id, not series_id. The source
                # schema encodes it as <modality>_<series_id>.
                prefix = row["modality"] + "_"
                if not modality_id.startswith(prefix):
                    raise ValueError(f"Cannot resolve source series for {modality_id}")
                group = _safe(modality_id[len(prefix):])
            membership = {
                "series_id": group, "modality_id": modality_id,
                "modality": _safe(row["modality"]),
                "acquisition_mode": row["acquisition_mode"],
                "source_datetime": row["datetime"].isoformat(),
            }
            key = (product_id(row["product_name"]), row["major_tom_id"])
            self.acquisitions[key][modality_id] = membership
        for row in pair_rows:
            reference, secondary = product_id(row["master_product_name"]), product_id(row["slave_product_name"])
            if [reference, secondary] != self.ids:
                continue
            key = (reference, secondary, row["major_tom_id"])
            self.pairs[key][row["modality_id"]] = {
                "series_id": _safe(row["series_id"]),
                "modality_id": _safe(row["modality_id"]),
                "modality": _safe(row["modality"]),
                "acquisition_mode": row["acquisition_mode"],
                "pair_name": row["pair_name"],
                "master_datetime": row["master_datetime"].isoformat(),
                "slave_datetime": row["slave_datetime"].isoformat(),
                "source_relative_path": row["relative_path"],
                "membership_source": "pairs.parquet",
            }

    def for_cell(self, cell):
        reference = self.acquisitions.get((self.ids[0], cell), {})
        if len(self.ids) == 1:
            return [{**row, "membership_source": "database.parquet"} for row in reference.values()]
        secondary = self.acquisitions.get((self.ids[1], cell), {})
        memberships = {}
        for modality_id in reference.keys() & secondary.keys():
            left, right = reference[modality_id], secondary[modality_id]
            memberships[modality_id] = {
                **{key: left[key] for key in ("series_id", "modality_id", "modality", "acquisition_mode")},
                "pair_name": "__".join(self.ids),
                "master_datetime": left["source_datetime"],
                "slave_datetime": right["source_datetime"],
                "source_relative_path": None,
                "membership_source": "database.parquet:endpoint_intersection",
            }
        # Explicit source pair membership takes precedence over inferred
        # membership for newly requested edges of the same source timeseries.
        memberships.update(self.pairs.get((*self.ids, cell), {}))
        return list(memberships.values())


def write_series_memberships(dataset, kind, identity, rows):
    """Independent immutable Parquet fragments; no shared-file append or raster copies."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["series_id"], row["modality"])].append(row)
    references = []
    for (series, modality), records in sorted(groups.items()):
        records.sort(key=lambda row: (row["cell_id"], row["swath"], row["uri"]))
        batch_id = fingerprint(records)
        parent = Path(dataset) / "series" / series / modality / (
            "pairs" if kind == "pair" else "acquisitions") / identity
        target = parent / f"{batch_id}.parquet"
        parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=parent, suffix=".parquet", delete=False) as temp:
                temporary = Path(temp.name)
            try:
                pq.write_table(pa.Table.from_pylist(records, schema=SERIES_SCHEMA), temporary, compression="zstd")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        references.append(target.relative_to(dataset).as_posix())
    return references
