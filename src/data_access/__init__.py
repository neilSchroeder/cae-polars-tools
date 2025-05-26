"""
ClimaKitAE New Core - Modern climate data processing tools.

This package provides high-performance, cloud-native tools for processing
climate data with efficient S3 access and Polars-based data manipulation.
"""

from .coordinate_processor import CoordinateProcessor
from .polars_converter import PolarsConverter
from .zarr_reader import ClimateDataReader
from .zarr_scanner import (  # Legacy aliases
    get_climate_data_info,
    scan_climate_data,
    scan_zarr_s3,
    zarr_s3_info,
)
from .zarr_storage import S3ZarrStore

__all__ = [
    # Main public API
    "scan_climate_data",
    "get_climate_data_info",
    "ClimateDataReader",
    # Legacy aliases for backward compatibility
    "scan_zarr_s3",
    "zarr_s3_info",
    # Advanced components for custom workflows
    "S3ZarrStore",
    "CoordinateProcessor",
    "PolarsConverter",
]
