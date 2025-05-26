"""
High-level scanning interface for Zarr climate data.

This module provides the main public API for scanning Zarr files from S3
into Polars LazyFrames with a simple, intuitive interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import polars as pl

from .zarr_reader import ClimateDataReader


def scan_climate_data(
    store_path: str,
    array_name: Optional[str] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    group: Optional[str] = None,
    consolidated: Optional[bool] = None,
    select_dims: Optional[Dict[str, Union[slice, int, List[int]]]] = None,
    chunk_size: int = 10000,
    streaming: bool = True,
) -> Union[pl.LazyFrame, Dict[str, pl.LazyFrame]]:
    """
    Scan climate data from Zarr files on S3 into Polars LazyFrames.

    This is the main entry point for reading climate datasets stored as Zarr arrays
    on S3. It provides a convenient interface for common operations while allowing
    full customization of the underlying reader.

    Parameters
    ----------
    store_path : str
        S3 path to the zarr store (e.g., 's3://bucket/path/to/store.zarr').
    array_name : str, optional
        Specific array to read. If None, returns info about all arrays.
    storage_options : Dict[str, Any], optional
        S3 credentials and options. Common keys:
        - key: AWS access key ID
        - secret: AWS secret access key
        - token: AWS session token (for temporary credentials)
        - region_name: AWS region name
    group : str, optional
        Group within the zarr store to access.
    consolidated : bool, optional
        Whether to use consolidated metadata for faster access.
    select_dims : Dict[str, Union[slice, int, List[int]]], optional
        Dimension selection criteria. Examples:
        - {"time": slice(0, 100)}: First 100 time steps
        - {"lat": slice(10, 20), "time": 0}: Lat range at single time
    chunk_size : int, default 10000
        Size of chunks for streaming processing.
    streaming : bool, default True
        Whether to use streaming mode for large arrays.

    Returns
    -------
    Union[pl.LazyFrame, Dict[str, pl.LazyFrame]]
        LazyFrame if array_name specified, dict of LazyFrames otherwise.

    Examples
    --------
    Read a specific array::

        lf = scan_climate_data(
            "s3://my-bucket/data.zarr",
            array_name="temperature",
            storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
        )

    Read with dimension selection::

        lf = scan_climate_data(
            "s3://my-bucket/data.zarr",
            array_name="temperature",
            select_dims={"time": slice(0, 100), "lat": slice(10, 20)}
        )

    List all arrays::

        reader = ClimateDataReader("s3://my-bucket/data.zarr")
        arrays = reader.list_arrays()
    """
    reader = ClimateDataReader(
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


def get_climate_data_info(
    store_path: str,
    storage_options: Optional[Dict[str, Any]] = None,
    group: Optional[str] = None,
    consolidated: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Get comprehensive information about a Zarr store and its arrays.

    Parameters
    ----------
    store_path : str
        S3 path to the zarr store.
    storage_options : Dict[str, Any], optional
        S3 credentials and options.
    group : str, optional
        Group within the zarr store.
    consolidated : bool, optional
        Whether to use consolidated metadata.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing store information and array metadata.
    """
    reader = ClimateDataReader(
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
scan_zarr_s3 = scan_climate_data
zarr_s3_info = get_climate_data_info
