"""
High-level Zarr data reader for climate datasets.

This module provides the main interface for reading Zarr arrays from S3
with coordinate handling and conversion to Polars DataFrames.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .coordinate_processor import CoordinateProcessor
from .polars_converter import PolarsConverter
from .zarr_storage import S3ZarrStore


class ClimateDataReader:
    """
    High-performance reader for climate data stored as Zarr arrays on S3.

    This class provides the main interface for reading multi-dimensional climate
    datasets from S3 storage with automatic coordinate handling, streaming support
    for large datasets, and conversion to Polars LazyFrames.

    Attributes:
        store: S3 Zarr store manager
        coord_processor: Coordinate array processor
        converter: Polars DataFrame converter
        chunk_size: Number of data points to process per chunk in streaming mode

    Examples:
        Basic usage::

            reader = ClimateDataReader("s3://bucket/data.zarr")
            arrays = reader.list_arrays()
            lf = reader.read_array("temperature")

        With S3 credentials::

            reader = ClimateDataReader(
                "s3://bucket/data.zarr",
                storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
            )
    """

    def __init__(
        self,
        store_path: str,
        storage_options: dict[str, Any] | None = None,
        group: str | None = None,
        consolidated: bool | None = None,
        chunk_size: int = 10000,
    ) -> None:
        """
        Initialize Climate Data Reader.

        Parameters
        ----------
        store_path : str
            S3 path to the zarr store (e.g., 's3://bucket/path/to/store.zarr').
        storage_options : Dict[str, Any], optional
            Options passed to s3fs (credentials, region, etc.).
            Common options include:
            - key: AWS access key ID
            - secret: AWS secret access key
            - token: AWS session token
            - region_name: AWS region
        group : str, optional
            Group within the zarr store to read.
        consolidated : bool, optional
            Whether to use consolidated metadata. If None, will attempt
            to auto-detect and fall back gracefully.
        chunk_size : int, default 10000
            Size of chunks for streaming processing of large arrays.
        """
        self.store = S3ZarrStore(
            store_path=store_path,
            storage_options=storage_options,
            group=group,
            consolidated=consolidated,
        )
        self.coord_processor = CoordinateProcessor()
        self.converter = PolarsConverter(chunk_size=chunk_size)
        self.chunk_size = chunk_size

    def list_arrays(self) -> list[str]:
        """
        List all arrays in the zarr group.

        Returns
        -------
        List[str]
            List of array names available in the zarr store.
        """
        return self.store.list_arrays()

    def get_array_info(self, array_name: str) -> dict[str, Any]:
        """
        Get comprehensive information about a specific array.

        Parameters
        ----------
        array_name : str
            Name of the array to inspect.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing array metadata including shape, dtype,
            dimensions, and attributes.
        """
        return self.store.get_array_info(array_name)

    def read_array(
        self,
        array_name: str,
        select_dims: dict[str, slice | int | list[int]] | None = None,
        streaming: bool = True,
    ) -> pl.LazyFrame:
        """
        Read a zarr array into a Polars LazyFrame with coordinate expansion.

        This method reads multi-dimensional zarr arrays and converts them to a
        flat representation with explicit coordinate columns. For example, a 3D
        array with dimensions (time, lat, lon) becomes a LazyFrame with columns
        ['time', 'lat', 'lon', 'value'].

        Parameters
        ----------
        array_name : str
            Name of the array to read.
        select_dims : Dict[str, Union[slice, int, List[int]]], optional
            Dictionary mapping dimension names to selection criteria.
            Examples:
            - {"time": slice(0, 100)}: Select first 100 time steps
            - {"lat": slice(10, 50), "lon": [0, 5, 10]}: Select lat range and specific lon indices
            - {"time": 0}: Select single time step (reduces dimensions)
        streaming : bool, default True
            Whether to use streaming mode for large arrays. Recommended
            for arrays larger than chunk_size elements.

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame with coordinate columns and 'value' column containing
            the array data. Column types are preserved from the original zarr arrays.

        Raises
        ------
        KeyError
            If array_name is not found in the zarr store.

        Examples
        --------
        Basic usage::

            lf = reader.read_array("temperature")

        With dimension selection::

            lf = reader.read_array(
                "temperature",
                select_dims={"time": slice(0, 12), "lat": slice(100, 200)}
            )
        """
        # Get the zarr array
        array = self.store.get_array(array_name)

        # Get dimension names from array attributes
        dims = getattr(array, "attrs", {}).get(
            "_ARRAY_DIMENSIONS", [f"dim_{i}" for i in range(array.ndim)]
        )

        # Extract coordinate arrays from the zarr group
        group = self.store.open_zarr_group()
        coord_arrays = self.coord_processor.extract_coordinate_arrays(group, dims)

        # Apply dimension selection if provided
        selection, selected_dims, selected_coord_arrays = (
            self.coord_processor.process_dimension_selection(
                dims, coord_arrays, select_dims
            )
        )

        # Read the data with selection applied
        if selection:
            data = array[tuple(selection)]
        else:
            data = array[:]

        # Convert to DataFrame format
        return self.converter.array_to_polars_lazy(
            data, selected_dims, selected_coord_arrays, streaming
        )

    def read_multiple_arrays(
        self, array_names: list[str], streaming: bool = True
    ) -> dict[str, pl.LazyFrame]:
        """
        Read multiple arrays and return a dictionary of LazyFrames.

        Parameters
        ----------
        array_names : List[str]
            Names of arrays to read.
        streaming : bool, default True
            Whether to use streaming mode.

        Returns
        -------
        Dict[str, pl.LazyFrame]
            Dictionary mapping array names to LazyFrames.
        """
        return {
            name: self.read_array(name, streaming=streaming) for name in array_names
        }
