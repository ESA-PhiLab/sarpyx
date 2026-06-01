"""Implemented SAR data readers."""

import rasterio
import zarr



def read_tif(tif_path, verbose=False):
    with rasterio.open(tif_path) as dataset:
        # Print dataset properties
        if verbose:
            print("Dataset properties:")
            print(f"Name: {dataset.name}")
            print(f"Mode: {dataset.mode}")
            print(f"Count: {dataset.count}")
            print(f"Width: {dataset.width}")
            print(f"Height: {dataset.height}")
            print(f"CRS: {dataset.crs}")
            print(f"Transform: {dataset.transform}")

        # Read the first band
        band1 = dataset.read(1)
    return band1

 


def read_zarr_file(file_path, array_or_group_key=None):
    """
    Read and extract data from a .zarr file.

    Parameters:
    - file_path: str, the path to the .zarr file.
    - array_or_group_key: str, optional key specifying which array or group to extract from the Zarr store.

    Returns:
    Zarr array or group, depending on what is stored in the file.
    """
    # Open Zarr file
    root = zarr.open(file_path, mode='r')

    if array_or_group_key is None:
        # Return the root group or array if no key is specified
        return root
    else:
        # Otherwise, return the specified array or group
        return root[array_or_group_key]








__all__ = [
    'read_tif',
    'read_zarr_file',
]
