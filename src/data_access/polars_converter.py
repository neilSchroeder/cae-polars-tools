"""
Polars DataFrame conversion utilities for multi-dimensional climate data.

This module handles the conversion of numpy arrays with coordinate information
into Polars LazyFrames with optimized streaming and memory management.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .coordinate_processor import CoordinateProcessor


class PolarsConverter:
    """
    Converts numpy arrays to Polars LazyFrames with coordinate expansion.

    This class provides intelligent conversion strategies based on array size
    and memory constraints, with support for streaming large datasets.
    """

    def __init__(self, chunk_size: int = 10000) -> None:
        """
        Initialize Polars converter.

        Parameters
        ----------
        chunk_size : int, default 10000
            Size of chunks for streaming processing of large arrays.
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
        Convert numpy array to Polars LazyFrame with appropriate method.

        Uses intelligent chunking based on array characteristics and memory constraints.

        Parameters
        ----------
        data : np.ndarray
            Input numpy array to convert.
        dim_names : List[str]
            Names of dimensions corresponding to array axes.
        coord_arrays : Dict[str, Optional[np.ndarray]]
            Coordinate arrays for each dimension. Keys should match dim_names.
        streaming : bool, default True
            Whether to use streaming for large arrays.

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame with coordinate columns and values.
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
        full meshgrids in memory.

        Parameters
        ----------
        data : np.ndarray
            Multi-dimensional numpy array to convert.
        dim_names : List[str]
            Names corresponding to each dimension.
        coord_arrays : Dict[str, Optional[np.ndarray]]
            Coordinate arrays for each dimension.

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame with expanded coordinates.
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

        Uses a true lazy approach that avoids creating large intermediate arrays.

        Parameters
        ----------
        data : np.ndarray
            Multi-dimensional numpy array to convert.
        dim_names : List[str]
            Names corresponding to each dimension.
        coord_arrays : Dict[str, Optional[np.ndarray]]
            Coordinate arrays for each dimension.

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame created with efficient lazy evaluation.
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
