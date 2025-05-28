"""
Polars DataFrame conversion utilities for multi-dimensional climate data.

This module provides efficient conversion of multi-dimensional numpy arrays with
coordinate information into Polars LazyFrames, optimized for climate data workflows.
It handles memory-efficient coordinate expansion and intelligent streaming for large
datasets without creating memory-intensive full meshgrids.

The PolarsConverter class offers multiple conversion strategies based on array
characteristics and memory constraints, enabling efficient processing of both
small in-memory datasets and large streaming datasets.

Primary Capabilities
--------------------
- Memory-efficient coordinate expansion without full meshgrids
- Intelligent streaming for large datasets based on configurable thresholds
- Optimized coordinate processing using efficient numpy operations
- Support for various array dimensionalities (scalar, 1D, multi-dimensional)
- Lazy evaluation with Polars LazyFrames for deferred computation

Examples
--------
Basic array conversion:

>>> import numpy as np
>>> from cae_polars.data_access import PolarsConverter
>>>
>>> converter = PolarsConverter(chunk_size=5000)
>>> data = np.random.rand(100, 50, 25)
>>> dim_names = ["time", "lat", "lon"]
>>> coord_arrays = {
...     "time": np.arange(100),
...     "lat": np.linspace(-90, 90, 50),
...     "lon": np.linspace(-180, 180, 25)
... }
>>>
>>> # Convert to lazy frame
>>> lazy_df = converter.array_to_polars_lazy(data, dim_names, coord_arrays)

Large dataset streaming:

>>> # For large datasets, automatic streaming is used
>>> large_data = np.random.rand(365, 720, 1440)  # Global daily data
>>> lazy_df = converter.array_to_polars_lazy(
...     large_data, ["time", "lat", "lon"], coord_arrays, streaming=True
... )
>>>
>>> # Lazy evaluation - computation happens when needed
>>> result = lazy_df.filter(pl.col("lat") > 0).collect()

Notes
-----
The converter automatically selects between streaming and non-streaming approaches
based on array size and memory constraints. Coordinate arrays are efficiently
expanded using repeat and tile operations rather than meshgrid to minimize memory
usage during the conversion process.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .coordinate_processor import CoordinateProcessor


class PolarsConverter:
    """
    Converts numpy arrays to Polars LazyFrames with coordinate expansion.

    This class provides intelligent conversion strategies based on array size
    and memory constraints, with support for streaming large datasets. It uses
    the CoordinateProcessor for efficient coordinate expansion without creating
    memory-intensive full meshgrids.

    Parameters
    ----------
    chunk_size : int, default 10000
        Size of chunks for streaming processing of large arrays. Arrays larger
        than this threshold will be processed in chunks to manage memory usage.

    Attributes
    ----------
    chunk_size : int
        Configured chunk size for streaming operations
    coord_processor : CoordinateProcessor
        Instance for handling coordinate array processing and expansion

    Examples
    --------
    Basic usage:

    >>> converter = PolarsConverter(chunk_size=5000)
    >>> data = np.random.rand(100, 50)
    >>> dim_names = ["time", "location"]
    >>> coords = {"time": np.arange(100), "location": np.arange(50)}
    >>> lazy_df = converter.array_to_polars_lazy(data, dim_names, coords)

    For large datasets with automatic streaming:

    >>> large_data = np.random.rand(1000, 1000, 100)
    >>> lazy_df = converter.array_to_polars_lazy(
    ...     large_data, ["x", "y", "z"], coords, streaming=True
    ... )

    Notes
    -----
    The converter automatically chooses between streaming and non-streaming
    approaches based on array size relative to the chunk_size threshold.
    Streaming is recommended for large datasets to avoid memory overflow.
    """

    def __init__(self, chunk_size: int = 10000) -> None:
        """
        Initialize Polars converter with specified chunk size.

        Parameters
        ----------
        chunk_size : int, default 10000
            Size of chunks for streaming processing of large arrays. This
            controls the memory vs. performance trade-off for large datasets.
        """
        self.chunk_size = chunk_size
        self.coord_processor = CoordinateProcessor()

    def array_to_polars_lazy(
        self,
        data: np.ndarray,
        dim_names: list[str],
        coord_arrays: dict[str, np.ndarray | None],
        streaming: bool = True,
    ) -> pl.LazyFrame:
        """
        Convert numpy array to Polars LazyFrame with intelligent method selection.

        Automatically chooses the most appropriate conversion strategy based on
        array characteristics and memory constraints. Uses streaming for large
        datasets and direct conversion for smaller ones.

        Parameters
        ----------
        data : np.ndarray
            Input numpy array to convert. Can be scalar, 1D, or multi-dimensional
        dim_names : list of str
            Names of dimensions corresponding to array axes. Length must match
            the number of array dimensions
        coord_arrays : dict of {str : ndarray or None}
            Coordinate arrays for each dimension. Keys should match dim_names.
            If None for a dimension, integer indices will be used as coordinates
        streaming : bool, default True
            Whether to use streaming for large arrays. When True, arrays larger
            than chunk_size will be processed in chunks

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame with coordinate columns and data values. The
            resulting DataFrame has one column per dimension plus a 'value' column

        Notes
        -----
        For scalar data, returns a single-row DataFrame. For 1D arrays, creates
        a simple two-column DataFrame. For multi-dimensional arrays, chooses
        between memory-efficient streaming and direct conversion based on size.
        """
        if data.ndim == 0:
            # Scalar value
            return pl.LazyFrame({"value": [data.item()]})
        elif data.ndim == 1:
            # 1D array - simple case
            coord_array = coord_arrays.get(dim_names[0])
            if coord_array is not None:
                return pl.LazyFrame({dim_names[0]: coord_array, "value": data})
            else:
                return pl.LazyFrame({dim_names[0]: np.arange(len(data)), "value": data})
        else:
            # Multi-dimensional array - choose method based on size and memory efficiency
            memory_threshold = self.chunk_size

            # For very large arrays, always use streaming
            if streaming and data.size > memory_threshold:
                return self._streaming_multidim_to_polars(data, dim_names, coord_arrays)
            else:
                return self._multidim_to_polars(data, dim_names, coord_arrays)

    def _multidim_to_polars(
        self,
        data: np.ndarray,
        dim_names: list[str],
        coord_arrays: dict[str, np.ndarray | None],
    ) -> pl.LazyFrame:
        """
        Convert multi-dimensional array to Polars LazyFrame (non-streaming).

        Uses efficient memory-friendly coordinate expansion without creating
        full meshgrids in memory. This method is optimal for arrays that fit
        comfortably in memory.

        Parameters
        ----------
        data : np.ndarray
            Multi-dimensional numpy array to convert
        dim_names : list of str
            Names corresponding to each dimension in order
        coord_arrays : dict of {str : ndarray or None}
            Coordinate arrays for each dimension. Missing coordinates are
            replaced with integer indices

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame with expanded coordinates and flattened data values

        Notes
        -----
        This method flattens the data array once and uses efficient coordinate
        expansion from the CoordinateProcessor. It's suitable for datasets that
        can fit in memory without causing performance issues.
        """
        # Flatten data once and reuse
        flat_values = data.ravel()

        # Create coordinate columns efficiently using broadcasting
        flat_data = {"value": flat_values}

        # Get coordinate expansions from processor
        coord_expansions = self.coord_processor.create_coordinate_expansions(
            data.shape, dim_names, coord_arrays
        )

        # Add coordinate columns to the data
        flat_data.update(coord_expansions)

        return pl.LazyFrame(flat_data)

    def _streaming_multidim_to_polars(
        self,
        data: np.ndarray,
        dim_names: list[str],
        coord_arrays: dict[str, np.ndarray | None],
    ) -> pl.LazyFrame:
        """
        Convert multi-dimensional array to Polars LazyFrame with memory-efficient streaming.

        Uses a chunked processing approach that avoids creating large intermediate
        arrays in memory. This method is essential for processing very large datasets
        that would otherwise cause memory overflow.

        Parameters
        ----------
        data : np.ndarray
            Multi-dimensional numpy array to convert
        dim_names : list of str
            Names corresponding to each dimension in order
        coord_arrays : dict of {str : ndarray or None}
            Coordinate arrays for each dimension. Missing coordinates are
            replaced with integer indices

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame created with efficient lazy evaluation and chunked
            processing. All chunks are concatenated into a single LazyFrame

        Notes
        -----
        This method processes the data in chunks of size self.chunk_size to manage
        memory usage. Each chunk is processed independently and then concatenated
        using Polars' efficient concatenation. For arrays smaller than chunk_size,
        it falls back to the non-streaming method.
        """
        total_size = data.size

        # If data is small enough, use non-streaming approach
        if total_size <= self.chunk_size:
            return self._multidim_to_polars(data, dim_names, coord_arrays)

        # For large data, create smaller chunks and process iteratively
        chunks = []
        flat_data = data.ravel()

        # Process in chunks to avoid memory overflow
        for start_idx in range(0, total_size, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_size)

            # Create chunk data dictionary
            chunk_data = {"value": flat_data[start_idx:end_idx]}

            # Generate coordinate chunks efficiently
            coord_chunks = self.coord_processor.create_streaming_coordinate_chunks(
                data.shape, dim_names, coord_arrays, start_idx, end_idx
            )

            # Add coordinate chunks to the data
            chunk_data.update(coord_chunks)

            chunks.append(pl.DataFrame(chunk_data).lazy())

        # Concatenate all chunks
        if not chunks:
            return pl.LazyFrame({name: [] for name in dim_names + ["value"]})

        return pl.concat(chunks)
