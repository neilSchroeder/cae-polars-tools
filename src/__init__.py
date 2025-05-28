"""
CAE-Polars: High-performance Zarr I/O plugin for Polars.

This package provides efficient reading of Zarr arrays from cloud storage
(especially S3) into Polars DataFrames with streaming support for large datasets.

The package is designed specifically for climate and earth science data workflows,
offering optimized performance for multi-dimensional array processing with
intelligent chunking, coordinate handling, and memory-efficient streaming.

Key Features
------------
- High-performance Zarr array reading from S3 and local storage
- Seamless integration with Polars DataFrames and LazyFrames
- Streaming support for datasets larger than memory
- Optimized coordinate processing for climate data
- Intelligent chunking and dimension selection
- Support for consolidated metadata and hierarchical groups

Basic Usage
-----------
>>> import cae_polars as cp
>>>
>>> # Scan climate data from S3
>>> df = cp.scan_data(
...     "s3://bucket/climate-data.zarr",
...     array_name="temperature",
...     storage_options={"anon": True}
... )
>>>
>>> # Get information about available arrays
>>> info = cp.get_zarr_data_info("s3://bucket/climate-data.zarr")
>>> print(info["arrays"].keys())

Advanced Usage
--------------
>>> # Use the reader directly for more control
>>> reader = cp.ZarrDataReader(
...     "s3://bucket/climate-data.zarr",
...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
... )
>>>
>>> # Read with dimension selection
>>> df = reader.read_array(
...     "temperature",
...     select_dims={"time": slice(0, 100), "lat": slice(10, 20)},
...     streaming=True
... )

See Also
--------
zarr : Python package for working with chunked, compressed, N-dimensional arrays
polars : Fast DataFrame library for Python
s3fs : Filesystem interface to S3
"""

from .data_access import ZarrDataReader, get_zarr_data_info, scan_data

__version__ = "0.1.0"
__all__ = [
    "scan_data",
    "get_zarr_data_info",
    "ZarrDataReader",
]
