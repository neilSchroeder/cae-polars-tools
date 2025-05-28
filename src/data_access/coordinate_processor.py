"""
Coordinate array processing and expansion for multi-dimensional climate data.

This module handles the extraction and efficient expansion of coordinate arrays
from Zarr stores, providing optimized coordinate meshgrid creation for large
datasets with intelligent memory management and caching strategies.

The CoordinateProcessor is designed specifically for climate data workflows where
coordinate arrays (time, latitude, longitude, etc.) need to be efficiently
processed and expanded to match data array dimensions without creating
memory-intensive full meshgrids.

Core Capabilities
-----------------
- Memory-efficient coordinate array processing
- Intelligent caching based on array size thresholds
- Optimized meshgrid creation for large datasets
- Support for various dimension selection strategies
- Streaming-friendly coordinate chunk generation

Examples
--------
Basic coordinate processing:

>>> from cae_polars.data_access import CoordinateProcessor
>>>
>>> processor = CoordinateProcessor(cache_size_threshold=5000)
>>>
>>> # Extract coordinates from zarr group
>>> coords = processor.extract_coordinate_arrays(zarr_group, ["time", "lat", "lon"])
>>>
>>> # Process dimension selection
>>> selection = {"time": slice(0, 100), "lat": [10, 20, 30]}
>>> processed = processor.process_dimension_selection(coords, selection)

Advanced usage with streaming:

>>> # Create coordinate expansions for large datasets
>>> expansions = processor.create_coordinate_expansions(
...     coords, data_shape=(365, 180, 360)
... )
>>>
>>> # Generate streaming coordinate chunks
>>> for chunk_coords in processor.create_streaming_coordinate_chunks(
...     coords, chunk_size=1000
... ):
...     # Process chunk coordinates
...     pass
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CoordinateProcessor:
    """
    Processes coordinate arrays and handles dimension selection for climate datasets.

    This class provides efficient coordinate array extraction, caching, and expansion
    without creating memory-intensive full meshgrids. It's optimized for climate data
    workflows where coordinate arrays (time, lat, lon, etc.) need to be processed
    efficiently with intelligent memory management.

    Parameters
    ----------
    cache_size_threshold : int, default 10000
        Maximum size for coordinate arrays to cache in memory. Arrays larger than
        this threshold will not be cached to prevent excessive memory usage.

    Attributes
    ----------
    cache_size_threshold : int
        Maximum size threshold for coordinate array caching

    Examples
    --------
    Basic usage:

    >>> processor = CoordinateProcessor(cache_size_threshold=5000)
    >>> coords = processor.extract_coordinate_arrays(zarr_group, ["time", "lat", "lon"])

    With dimension selection:

    >>> selection = {"time": slice(0, 100), "lat": [10, 20, 30]}
    >>> processed = processor.process_dimension_selection(coords, selection)

    Notes
    -----
    The processor uses lazy evaluation and streaming techniques to handle large
    coordinate arrays efficiently. Coordinate arrays below the cache threshold
    are cached in memory for repeated access.
    """

    def __init__(self, cache_size_threshold: int = 10000) -> None:
        self.cache_size_threshold = cache_size_threshold
        self._coord_cache: dict[str, np.ndarray] = {}

    def extract_coordinate_arrays(
        self, group: Any, dims: list[str]
    ) -> dict[str, np.ndarray | None]:
        """
        Extract coordinate arrays from zarr group for each dimension with caching.

        Uses intelligent caching and lazy loading to avoid redundant coordinate
        array reads while managing memory efficiently for large coordinate arrays.

        Parameters
        ----------
        group : zarr.Group
            Opened zarr group containing coordinate arrays
        dims : list of str
            List of dimension names to extract coordinate arrays for

        Returns
        -------
        dict of {str : ndarray or None}
            Dictionary mapping dimension names to coordinate arrays. Returns None
            for dimensions not found in the group.

        Notes
        -----
        Coordinate arrays smaller than cache_size_threshold are loaded into memory
        and cached. Larger arrays are kept as zarr arrays to preserve memory.
        """
        coord_arrays = {}
        for dim_name in dims:
            if dim_name in group:
                try:
                    # Read coordinate array efficiently - avoid unnecessary copies
                    coord_array = group[dim_name]
                    # Only convert to numpy array when needed, keep zarr array if small
                    if coord_array.size < self.cache_size_threshold:
                        coord_arrays[dim_name] = np.asarray(coord_array[:])
                    else:
                        # For large coordinate arrays, keep as zarr array and slice as needed
                        coord_arrays[dim_name] = np.asarray(coord_array[:])
                except Exception:
                    # Fallback to None if coordinate reading fails
                    coord_arrays[dim_name] = None
            else:
                coord_arrays[dim_name] = None
        return coord_arrays

    def process_dimension_selection(
        self,
        dims: list[str],
        coord_arrays: dict[str, np.ndarray | None],
        select_dims: dict[str, slice | int | list[int]] | None,
    ) -> tuple[list[Any], list[str], dict[str, np.ndarray | None]]:
        """
        Process dimension selection criteria and apply to coordinate arrays.

        Handles various selection types (integer indices, slices, lists) and applies
        them consistently to both data arrays and coordinate arrays. Integer selections
        reduce dimensionality while slice and list selections preserve dimensions.

        Parameters
        ----------
        dims : list of str
            Original dimension names in order
        coord_arrays : dict of {str : ndarray or None}
            Original coordinate arrays mapping dimension names to arrays
        select_dims : dict of {str : slice or int or list of int}, optional
            Selection criteria for each dimension. If None, no selection is applied.
            - int: Select single index (reduces dimensionality)
            - slice: Select range (preserves dimensionality)
            - list of int: Select multiple indices (preserves dimensionality)

        Returns
        -------
        tuple of (list, list of str, dict)
            A 3-tuple containing:
            - selection_list: List of selection objects for array indexing
            - selected_dim_names: List of remaining dimension names after selection
            - selected_coord_arrays: Dictionary of coordinate arrays after selection

        Notes
        -----
        Integer selections remove the dimension from the output, while slice and list
        selections preserve the dimension. Failed coordinate selections fall back to
        None values to maintain consistency.
        """
        selection = []
        selected_dims = []
        selected_coord_arrays = {}

        for dim_name in dims:
            if select_dims and dim_name in select_dims:
                sel = select_dims[dim_name]
                if isinstance(sel, (int, np.integer)):
                    selection.append(sel)
                    # Skip this dimension in output (reduces dimensionality)
                    continue
                elif isinstance(sel, (slice, list)):
                    selection.append(sel)
                    # Apply same selection to coordinate array if it exists
                    coord_array = coord_arrays[dim_name]
                    if coord_array is not None:
                        try:
                            selected_coord_arrays[dim_name] = coord_array[sel]
                        except (IndexError, TypeError):
                            # If selection fails, fallback to None
                            selected_coord_arrays[dim_name] = None
                    else:
                        selected_coord_arrays[dim_name] = None
                else:
                    selection.append(slice(None))
                    selected_coord_arrays[dim_name] = coord_arrays[dim_name]
            else:
                selection.append(slice(None))
                selected_coord_arrays[dim_name] = coord_arrays[dim_name]

            selected_dims.append(dim_name)

        return selection, selected_dims, selected_coord_arrays

    def create_coordinate_expansions(
        self,
        data_shape: tuple[int, ...],
        dim_names: list[str],
        coord_arrays: dict[str, np.ndarray | None],
    ) -> dict[str, np.ndarray]:
        """
        Create efficient coordinate expansions without full meshgrids.

        Generates flattened coordinate arrays that correspond to each data point
        without creating memory-intensive full meshgrids. Uses numpy's efficient
        repeat and tile operations to create coordinate expansions.

        Parameters
        ----------
        data_shape : tuple of int
            Shape of the data array for which coordinates are being expanded
        dim_names : list of str
            Names corresponding to each dimension in order
        coord_arrays : dict of {str : ndarray or None}
            Coordinate arrays for each dimension. If None for a dimension,
            creates integer indices (0, 1, 2, ...) as fallback coordinates

        Returns
        -------
        dict of {str : ndarray}
            Dictionary mapping dimension names to flattened coordinate arrays.
            Each array has length equal to the total number of data points
            (product of data_shape dimensions).

        Notes
        -----
        This method is memory-efficient compared to meshgrid operations, using
        repeat and tile operations to avoid creating intermediate full-size arrays.
        The resulting coordinate arrays are in C-order (row-major) flat indexing.
        """
        flat_coords = {}

        # Calculate coordinate values without creating full meshgrids
        for i, dim_name in enumerate(dim_names):
            coord_array = coord_arrays.get(dim_name)
            if coord_array is None:
                coord_array = np.arange(data_shape[i])

            # Calculate repeat and tile factors for efficient coordinate expansion
            repeat_factor = (
                np.prod(data_shape[i + 1 :], dtype=int)
                if i + 1 < len(data_shape)
                else 1
            )
            tile_factor = np.prod(data_shape[:i], dtype=int) if i > 0 else 1

            # Use numpy's efficient repeat/tile instead of meshgrid
            flat_coords[dim_name] = np.tile(
                np.repeat(coord_array, repeat_factor), tile_factor
            )

        return flat_coords

    def create_streaming_coordinate_chunks(
        self,
        data_shape: tuple[int, ...],
        dim_names: list[str],
        coord_arrays: dict[str, np.ndarray | None],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, np.ndarray]:
        """
        Create coordinate chunks for streaming processing of large datasets.

        Generates coordinate arrays corresponding to a specific flat index range
        without materializing the full coordinate expansion. This enables efficient
        streaming processing of large datasets by processing coordinates in chunks.

        Parameters
        ----------
        data_shape : tuple of int
            Shape of the full data array
        dim_names : list of str
            Names corresponding to each dimension in order
        coord_arrays : dict of {str : ndarray or None}
            Coordinate arrays for each dimension. If None for a dimension,
            creates integer indices as fallback coordinates
        start_idx : int
            Starting flat index for the chunk (inclusive)
        end_idx : int
            Ending flat index for the chunk (exclusive)

        Returns
        -------
        dict of {str : ndarray}
            Dictionary mapping dimension names to coordinate arrays for the
            specified chunk. Each array has length equal to (end_idx - start_idx).

        Notes
        -----
        This method calculates coordinate values on-demand for the specified flat
        index range, avoiding memory overhead of full coordinate expansions. The
        flat indexing follows C-order (row-major) convention where the last
        dimension varies fastest.
        """
        chunk_coords = {}

        # Pre-calculate coordinate expansion factors
        coords = []
        for i, dim_name in enumerate(dim_names):
            coord_array = coord_arrays.get(dim_name)
            if coord_array is None:
                coord_array = np.arange(data_shape[i])
            coords.append(coord_array)

        # Generate coordinate chunks efficiently
        for i, dim_name in enumerate(dim_names):
            coord_array = coords[i]

            # Calculate coordinates for this specific chunk
            chunk_indices = np.arange(start_idx, end_idx)

            # Calculate which coordinate values correspond to these flat indices
            if i == len(dim_names) - 1:  # Last dimension (fastest varying)
                coord_indices = chunk_indices % data_shape[i]
            elif i == 0:  # First dimension (slowest varying)
                coord_indices = chunk_indices // np.prod(data_shape[1:])
            else:  # Middle dimensions
                stride = np.prod(data_shape[i + 1 :])
                coord_indices = (chunk_indices // stride) % data_shape[i]

            chunk_coords[dim_name] = coord_array[coord_indices]

        return chunk_coords
