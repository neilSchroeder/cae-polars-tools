"""
High-level scanning interface for Zarr climate data on cloud storage.

This module provides the main public API for scanning Zarr files from S3 and other
cloud storage systems into Polars LazyFrames. It offers a simple, intuitive interface
that abstracts away the complexity of coordinate processing, dimension selection,
and streaming for climate data workflows.

The primary functions in this module serve as convenient entry points for the most
common operations when working with multi-dimensional climate datasets stored in
Zarr format on cloud storage.

Key Functions
-------------
scan_data : Main function for reading Zarr arrays into Polars LazyFrames
get_zarr_data_info : Retrieve comprehensive metadata about Zarr stores

Examples
--------
Basic usage for reading climate data:

>>> from cae_polars.data_access import scan_data
>>>
>>> # Read temperature data from S3
>>> temp_df = scan_data(
...     "s3://climate-bucket/era5.zarr",
...     array_name="temperature",
...     storage_options={"anon": True}  # For public datasets
... )
>>>
>>> # Process the data
>>> result = temp_df.filter(pl.col("time") > "2020-01-01").collect()

Advanced usage with dimension selection:

>>> # Read subset of precipitation data
>>> precip_df = scan_data(
...     "s3://climate-bucket/data.zarr",
...     array_name="precipitation",
...     select_dims={
...         "time": slice("2020-01-01", "2020-12-31"),
...         "lat": slice(30, 60),  # Northern latitudes
...         "lon": slice(-120, -60)  # Eastern US
...     },
...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
... )

Getting dataset information:

>>> from cae_polars.data_access import get_zarr_data_info
>>>
>>> info = get_zarr_data_info("s3://bucket/data.zarr")
>>> print("Available arrays:", list(info["arrays"].keys()))
>>> for array_name, array_info in info["arrays"].items():
...     print(f"{array_name}: {array_info['shape']} {array_info['dtype']}")

Notes
-----
This module is designed for ease of use and provides sensible defaults for most
climate data workflows. For more advanced control, users can directly instantiate
ZarrDataReader objects and call their methods explicitly.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .zarr_reader import ZarrDataReader


def scan_data(
    store_path: str,
    array_name: str | None = None,
    storage_options: dict[str, Any] | None = None,
    group: str | None = None,
    consolidated: bool | None = None,
    select_dims: dict[str, slice | int | list[int]] | None = None,
    chunk_size: int = 10000,
    streaming: bool = True,
) -> pl.LazyFrame | dict[str, pl.LazyFrame]:
    """
    Scan climate data from Zarr files on cloud storage into Polars LazyFrames.

    This is the main entry point for reading climate datasets stored as Zarr arrays
    on S3 and other cloud storage systems. It provides a convenient, high-level
    interface for common operations while allowing full customization of the
    underlying reader components.

    Parameters
    ----------
    store_path : str
        Cloud storage path to the zarr store (e.g., 's3://bucket/path/to/store.zarr')
    array_name : str, optional
        Specific array to read. If None, reads all available arrays in the store
    storage_options : dict, optional
        Cloud storage credentials and configuration options. Common keys include:
        - key: AWS access key ID
        - secret: AWS secret access key
        - token: AWS session token (for temporary credentials)
        - region_name: AWS region name
        - anon: True for anonymous/public access
    group : str, optional
        Specific group within the zarr store to access. Useful for hierarchical
        data organization
    consolidated : bool, optional
        Whether to use consolidated metadata for faster access. Auto-detected if None
    select_dims : dict, optional
        Dimension selection criteria for subsetting data. Keys are dimension names,
        values can be:
        - slice objects: select ranges (e.g., slice(0, 100))
        - integers: select single indices (reduces dimensionality)
        - lists: select multiple specific indices
    chunk_size : int, default 10000
        Size of chunks for streaming processing. Larger values use more memory
        but may be faster for sequential access
    streaming : bool, default True
        Whether to use streaming mode for large arrays. Recommended for datasets
        that don't fit comfortably in memory

    Returns
    -------
    pl.LazyFrame or dict of {str : pl.LazyFrame}
        If array_name is specified, returns a single LazyFrame with coordinate
        columns and data values. If array_name is None, returns a dictionary
        mapping array names to their corresponding LazyFrames.

    Examples
    --------
    Read a specific climate variable:

    >>> temp_df = scan_data(
    ...     "s3://climate-data/era5.zarr",
    ...     array_name="temperature",
    ...     storage_options={"anon": True}
    ... )
    >>> print(temp_df.schema)

    Read with time and spatial subsetting:

    >>> subset_df = scan_data(
    ...     "s3://private-bucket/data.zarr",
    ...     array_name="precipitation",
    ...     select_dims={
    ...         "time": slice(0, 365),      # First year
    ...         "lat": slice(100, 200),     # Latitude band
    ...         "lon": [-120, -110, -100]   # Specific longitudes
    ...     },
    ...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
    ... )

    Read all arrays for exploration:

    >>> all_arrays = scan_data("s3://bucket/data.zarr")
    >>> for name, lazy_df in all_arrays.items():
    ...     print(f"{name}: {lazy_df.schema}")

    Notes
    -----
    This function creates a ZarrDataReader instance internally and handles the
    complexity of coordinate processing and data conversion. For repeated access
    to the same store, consider creating a ZarrDataReader instance directly for
    better performance.
    """
    reader = ZarrDataReader(
        store_path=store_path,
        storage_options=storage_options,
        group=group,
        consolidated=consolidated,
        chunk_size=chunk_size,
    )

    if array_name:
        return reader.read_array(
            array_name=array_name,
            select_dims=select_dims,
            streaming=streaming,
        )
    else:
        # Return info about all arrays
        arrays = reader.list_arrays()
        return reader.read_multiple_arrays(arrays, streaming=streaming)


def get_zarr_data_info(
    store_path: str,
    storage_options: dict[str, Any] | None = None,
    group: str | None = None,
    consolidated: bool | None = None,
) -> dict[str, Any]:
    """
    Get comprehensive information about a Zarr store and its arrays.

    Retrieves detailed metadata about all arrays in a Zarr store, including
    shapes, data types, dimensions, and attributes. This is useful for exploring
    datasets before reading data or for building data catalogs.

    Parameters
    ----------
    store_path : str
        Cloud storage path to the zarr store (e.g., 's3://bucket/data.zarr')
    storage_options : dict, optional
        Cloud storage credentials and configuration options. Same format as
        used in scan_data()
    group : str, optional
        Specific group within the zarr store to inspect
    consolidated : bool, optional
        Whether to use consolidated metadata for faster access

    Returns
    -------
    dict
        Dictionary containing comprehensive store information with structure:
        - 'store_path': Original store path
        - 'group': Group name (if specified)
        - 'arrays': Dictionary mapping array names to their metadata

    Examples
    --------
    Explore a public dataset:

    >>> info = get_zarr_data_info(
    ...     "s3://climate-data/era5.zarr",
    ...     storage_options={"anon": True}
    ... )
    >>> print("Available arrays:", list(info["arrays"].keys()))
    >>>
    >>> # Check specific array details
    >>> temp_info = info["arrays"]["temperature"]
    >>> print(f"Temperature shape: {temp_info['shape']}")
    >>> print(f"Data type: {temp_info['dtype']}")
    >>> print(f"Dimensions: {temp_info['dimensions']}")

    Build a data catalog:

    >>> stores = [
    ...     "s3://bucket/era5.zarr",
    ...     "s3://bucket/gfs.zarr",
    ...     "s3://bucket/cmip6.zarr"
    ... ]
    >>>
    >>> catalog = {}
    >>> for store in stores:
    ...     catalog[store] = get_zarr_data_info(store)
    ...     print(f"{store}: {len(catalog[store]['arrays'])} arrays")

    Notes
    -----
    This function is read-only and does not load actual data, making it fast
    for exploring large datasets. The returned information can be used to
    make informed decisions about dimension selection and memory management
    before reading data with scan_data().
    """
    reader = ZarrDataReader(
        store_path=store_path,
        storage_options=storage_options,
        group=group,
        consolidated=consolidated,
    )

    arrays = reader.list_arrays()
    info = {"store_path": store_path, "group": group, "arrays": {}}

    for array_name in arrays:
        info["arrays"][array_name] = reader.get_array_info(array_name)

    return info


# Legacy aliases for backward compatibility
scan_zarr_s3 = scan_data
zarr_s3_info = get_zarr_data_info
