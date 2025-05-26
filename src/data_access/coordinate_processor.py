"""
Coordinate array processing and expansion for multi-dimensional climate data.

This module handles the extraction and efficient expansion of coordinate arrays
from Zarr stores, providing optimized coordinate meshgrid creation for large datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class CoordinateProcessor:
    """
    Processes coordinate arrays and handles dimension selection for climate datasets.

    This class provides efficient coordinate array extraction, caching, and expansion
    without creating memory-intensive full meshgrids.

    Parameters
    ----------
    cache_size_threshold : int, default 10000
        Maximum size for coordinate arrays to cache in memory

    Attributes
    ----------
    cache_size_threshold : int
        Maximum size threshold for coordinate array caching
    """

    def __init__(self, cache_size_threshold: int = 10000) -> None:
        self.cache_size_threshold = cache_size_threshold
        self._coord_cache: Dict[str, np.ndarray] = {}

    def extract_coordinate_arrays(
        self, group: Any, dims: List[str]
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract coordinate arrays from zarr group for each dimension with caching.

        Uses lazy loading and caching to avoid redundant coordinate array reads.

        Parameters
        ----------
        group : zarr.Group
            Opened zarr group
        dims : list of str
            List of dimension names

        Returns
        -------
        dict of {str : ndarray or None}
            Dictionary mapping dimension names to coordinate arrays or None
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
        dims: List[str],
        coord_arrays: Dict[str, Optional[np.ndarray]],
        select_dims: Optional[Dict[str, Union[slice, int, List[int]]]],
    ) -> Tuple[List[Any], List[str], Dict[str, Optional[np.ndarray]]]:
        """
        Process dimension selection criteria and apply to coordinate arrays.

        Parameters
        ----------
        dims : list of str
            Original dimension names
        coord_arrays : dict of {str : ndarray or None}
            Original coordinate arrays
        select_dims : dict of {str : slice or int or list of int}, optional
            Selection criteria for each dimension

        Returns
        -------
        tuple of (list, list of str, dict)
            - selection_list: List of selection objects for array indexing
            - selected_dim_names: List of remaining dimension names after selection
            - selected_coord_arrays: Dictionary of selected coordinate arrays
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
        data_shape: Tuple[int, ...],
        dim_names: List[str],
        coord_arrays: Dict[str, Optional[np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        Create efficient coordinate expansions without full meshgrids.

        Parameters
        ----------
        data_shape : tuple of int
            Shape of the data array
        dim_names : list of str
            Names corresponding to each dimension
        coord_arrays : dict of {str : ndarray or None}
            Coordinate arrays for each dimension

        Returns
        -------
        dict of {str : ndarray}
            Dictionary of expanded coordinate arrays
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
        data_shape: Tuple[int, ...],
        dim_names: List[str],
        coord_arrays: Dict[str, Optional[np.ndarray]],
        start_idx: int,
        end_idx: int,
    ) -> Dict[str, np.ndarray]:
        """
        Create coordinate chunks for streaming processing.

        Parameters
        ----------
        data_shape : tuple of int
            Shape of the data array
        dim_names : list of str
            Names corresponding to each dimension
        coord_arrays : dict of {str : ndarray or None}
            Coordinate arrays for each dimension
        start_idx : int
            Starting flat index for the chunk
        end_idx : int
            Ending flat index for the chunk

        Returns
        -------
        dict of {str : ndarray}
            Dictionary of coordinate chunks
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
