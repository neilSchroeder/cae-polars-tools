"""
High-level Zarr data reader for climate datasets stored on cloud storage.

This module provides the main interface for reading multi-dimensional Zarr arrays
from S3 or other cloud storage systems, with automatic coordinate handling,
intelligent dimension selection, and efficient conversion to Polars DataFrames.

The ZarrDataReader class serves as the primary entry point for accessing climate
datasets, offering both streaming and non-streaming modes for optimal performance
across different dataset sizes.

Key Features
------------
- High-performance reading from S3-based Zarr stores
- Automatic coordinate array extraction and expansion
- Memory-efficient streaming for large datasets
- Flexible dimension selection with slicing and indexing
- Integration with Polars for efficient data manipulation
- Support for consolidated and non-consolidated metadata

Examples
--------
Basic usage with public S3 data:

>>> from cae_polars.data_access import ZarrDataReader
>>>
>>> reader = ZarrDataReader("s3://bucket/climate-data.zarr")
>>> arrays = reader.list_arrays()
>>> print("Available arrays:", arrays)
>>>
>>> # Read temperature data
>>> temp_df = reader.read_array("temperature")
>>> result = temp_df.collect()

With AWS credentials and dimension selection:

>>> reader = ZarrDataReader(
...     "s3://private-bucket/data.zarr",
...     storage_options={
...         "key": "ACCESS_KEY_ID",
...         "secret": "SECRET_ACCESS_KEY",
...         "region_name": "us-west-2"
...     }
... )
>>>
>>> # Read subset of data with dimension selection
>>> subset_df = reader.read_array(
...     "precipitation",
...     select_dims={
...         "time": slice(0, 365),  # First year
...         "lat": slice(100, 200), # Latitude range
...         "lon": [10, 20, 30]     # Specific longitudes
...     }
... )

Reading multiple arrays efficiently:

>>> arrays_dict = reader.read_multiple_arrays(
...     ["temperature", "precipitation", "humidity"]
... )
>>> for name, lazy_df in arrays_dict.items():
...     print(f"{name}: {lazy_df.schema}")

Notes
-----
The reader automatically handles coordinate arrays, dimension metadata, and
memory management. For large datasets, streaming mode is recommended to avoid
memory overflow. The resulting Polars LazyFrames enable efficient downstream
processing with deferred computation.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .coordinate_processor import CoordinateProcessor
from .polars_converter import PolarsConverter
from .zarr_storage import S3ZarrStore


class ZarrDataReader:
    """
    High-performance reader for multi-dimensional climate data stored as Zarr arrays.

    This class provides the main interface for reading climate datasets from S3
    or other cloud storage systems, with automatic coordinate handling, streaming
    support for large datasets, and efficient conversion to Polars LazyFrames.
    It integrates coordinate processing, dimension selection, and data conversion
    into a unified, easy-to-use interface.

    Parameters
    ----------
    store_path : str
        S3 path to the zarr store (e.g., 's3://bucket/path/to/store.zarr')
    storage_options : dict, optional
        Options passed to s3fs for authentication and configuration
    group : str, optional
        Specific group within the zarr store to read from
    consolidated : bool, optional
        Whether to use consolidated metadata for improved performance
    chunk_size : int, default 10000
        Size of chunks for streaming processing of large arrays

    Attributes
    ----------
    store : S3ZarrStore
        S3 Zarr store manager for handling cloud storage operations
    coord_processor : CoordinateProcessor
        Processor for coordinate array extraction and expansion
    converter : PolarsConverter
        Converter for transforming arrays to Polars DataFrames
    chunk_size : int
        Number of data points to process per chunk in streaming mode

    Examples
    --------
    Basic usage with public data:

    >>> reader = ZarrDataReader("s3://bucket/data.zarr")
    >>> arrays = reader.list_arrays()
    >>> temp_data = reader.read_array("temperature")

    With authentication and specific group:

    >>> reader = ZarrDataReader(
    ...     "s3://private-bucket/data.zarr",
    ...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"},
    ...     group="climate_data",
    ...     chunk_size=5000
    ... )

    Reading with dimension selection:

    >>> data = reader.read_array(
    ...     "precipitation",
    ...     select_dims={"time": slice(0, 100), "lat": [10, 20, 30]}
    ... )

    Notes
    -----
    The reader automatically detects and processes coordinate arrays, handles
    dimension metadata, and provides memory-efficient streaming for large datasets.
    It's optimized for climate data workflows but can handle any multi-dimensional
    Zarr arrays with coordinate information.
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
        Initialize Zarr Data Reader.

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
        List all arrays available in the zarr store.

        Returns
        -------
        list of str
            List of array names available in the zarr store. These names can
            be used with read_array() to access the data.

        Examples
        --------
        >>> reader = ZarrDataReader("s3://bucket/data.zarr")
        >>> arrays = reader.list_arrays()
        >>> print("Available arrays:", arrays)
        ['temperature', 'precipitation', 'humidity']
        """
        return self.store.list_arrays()

    def get_array_info(self, array_name: str) -> dict[str, Any]:
        """
        Get comprehensive metadata information about a specific array.

        Retrieves detailed information about an array including its shape,
        data type, dimensions, chunking strategy, and custom attributes.
        This is useful for understanding data structure before reading.

        Parameters
        ----------
        array_name : str
            Name of the array to inspect. Must be a valid array name from
            the zarr store.

        Returns
        -------
        dict
            Dictionary containing comprehensive array metadata:
            - 'shape': Tuple of array dimensions
            - 'dtype': Data type of array elements
            - 'chunks': Chunking configuration
            - 'dimensions': List of dimension names
            - 'attributes': Custom metadata attributes
            - 'size': Total number of elements

        Raises
        ------
        KeyError
            If array_name is not found in the zarr store

        Examples
        --------
        >>> info = reader.get_array_info("temperature")
        >>> print(f"Shape: {info['shape']}")
        >>> print(f"Dimensions: {info['dimensions']}")
        >>> print(f"Data type: {info['dtype']}")
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
        Read multiple arrays efficiently and return a dictionary of LazyFrames.

        Provides a convenient way to read multiple related arrays from the same
        zarr store. Each array is processed independently with the same streaming
        settings and coordinate handling.

        Parameters
        ----------
        array_names : list of str
            Names of arrays to read. All names must exist in the zarr store.
        streaming : bool, default True
            Whether to use streaming mode for all arrays. Large arrays benefit
            from streaming to manage memory usage effectively.

        Returns
        -------
        dict of {str : pl.LazyFrame}
            Dictionary mapping array names to their corresponding LazyFrames.
            Each LazyFrame has the same structure as returned by read_array().

        Raises
        ------
        KeyError
            If any array name in array_names is not found in the zarr store

        Examples
        --------
        Read multiple climate variables:

        >>> arrays_dict = reader.read_multiple_arrays([
        ...     "temperature", "precipitation", "humidity"
        ... ])
        >>>
        >>> # Process each array
        >>> for name, lazy_df in arrays_dict.items():
        ...     print(f"{name}: {lazy_df.schema}")
        ...     result = lazy_df.filter(pl.col("time") > "2020-01-01").collect()

        Combine multiple arrays for analysis:

        >>> data = reader.read_multiple_arrays(["temp", "precip"])
        >>> combined = data["temp"].join(data["precip"], on=["time", "lat", "lon"])

        Notes
        -----
        This method processes arrays sequentially, not in parallel. For very
        large numbers of arrays, consider processing in batches to manage
        memory usage effectively.
        """
        return {
            name: self.read_array(name, streaming=streaming) for name in array_names
        }
