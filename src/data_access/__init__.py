"""
Data Access Module for CAE-Polars.

This module provides high-performance, cloud-native tools for accessing and
processing Zarr-based climate data with efficient S3 integration and
Polars-based data manipulation.

The module is organized into several components:

Core Components
---------------
ZarrDataReader : Primary interface for reading Zarr arrays
    High-level reader with streaming support and coordinate processing

S3ZarrStore : S3 storage abstraction
    Handles S3 filesystem connections and Zarr store access

Processing Components
---------------------
CoordinateProcessor : Coordinate array processing
    Optimized handling of dimension coordinates with caching

PolarsConverter : Array to DataFrame conversion
    Efficient conversion of NumPy arrays to Polars DataFrames

Convenience Functions
---------------------
scan_data : Quick array scanning
    Simple interface for reading arrays into LazyFrames

get_zarr_data_info : Store inspection
    Retrieve metadata about Zarr stores and arrays

Examples
--------
Basic usage with the convenience API:

>>> from cae_polars.data_access import scan_data, get_zarr_data_info
>>>
>>> # Get store information
>>> info = get_zarr_data_info("s3://bucket/climate-data.zarr")
>>> print(info["arrays"].keys())
>>>
>>> # Read data with dimension selection
>>> df = scan_data(
...     "s3://bucket/climate-data.zarr",
...     array_name="temperature",
...     select_dims={"time": slice(0, 100)},
...     storage_options={"anon": True}
... )

Advanced usage with the reader class:

>>> from cae_polars.data_access import ZarrDataReader
>>>
>>> reader = ZarrDataReader(
...     "s3://bucket/climate-data.zarr",
...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
... )
>>>
>>> # List available arrays
>>> arrays = reader.list_arrays()
>>>
>>> # Read with streaming for large datasets
>>> lf = reader.read_array("temperature", streaming=True)
>>> df = lf.collect()

Notes
-----
All functions support both local file paths and S3 URLs.
S3 authentication can be provided via storage_options or environment variables.
Large datasets are handled efficiently through streaming and lazy evaluation.
"""

from .coordinate_processor import CoordinateProcessor
from .polars_converter import PolarsConverter
from .zarr_reader import ZarrDataReader
from .zarr_scanner import (  # Legacy aliases
    get_zarr_data_info,
    scan_data,
    scan_zarr_s3,
    zarr_s3_info,
)
from .zarr_storage import S3ZarrStore

__all__ = [
    # Main public API
    "scan_data",
    "get_zarr_data_info",
    "ZarrDataReader",
    # Legacy aliases for backward compatibility
    "scan_zarr_s3",
    "zarr_s3_info",
    # Advanced components for custom workflows
    "S3ZarrStore",
    "CoordinateProcessor",
    "PolarsConverter",
]
