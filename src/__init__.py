"""
CAE-Polars: High-performance Zarr I/O plugin for Polars

This package provides efficient reading of Zarr arrays from cloud storage
(especially S3) into Polars DataFrames with streaming support for large datasets.
"""

from .data_access import ZarrDataReader, get_zarr_data_info, scan_data

__version__ = "0.1.0"
__all__ = [
    "scan_data",
    "get_zarr_data_info",
    "ZarrDataReader",
]
