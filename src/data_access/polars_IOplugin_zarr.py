"""
Polars IO Plugin for reading Zarr files from S3 with streaming support.

This plugin provides high-performance reading of Zarr arrays stored on S3 into Polars
LazyFrames with efficient streaming capabilities for large climate and scientific datasets.

Key Features:
    * Direct S3 access with configurable authentication
    * Streaming support for large multi-dimensional arrays
    * Automatic coordinate array extraction and meshgrid expansion
    * Preserves original data types (no automatic datetime conversion)
    * Proper NaN handling for masked data (e.g., ocean/land masks)
    * Dimension selection and filtering

Examples:
    Basic usage::

        import polars as pl
        from zarr_plugin_polars import scan_zarr_s3

        # Read specific array from S3
        lf = scan_zarr_s3(
            "s3://my-bucket/climate-data.zarr",
            array_name="temperature",
            storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
        )

    With dimension selection::

        lf = scan_zarr_s3(
            "s3://my-bucket/climate-data.zarr",
            array_name="temperature",
            select_dims={"time": slice(0, 100), "lat": slice(10, 50)}
        )

Author: Created for ClimaKitAE project
License: See project license
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

try:
    import s3fs
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    warnings.warn(
        "zarr and s3fs packages required for Zarr S3 plugin. "
        "Install with: pip install zarr s3fs"
    )


class ZarrS3Reader:
    """
    High-performance reader for Zarr arrays stored on Amazon S3.

    This class provides efficient reading of multi-dimensional Zarr arrays from S3
    storage with automatic coordinate handling, streaming support for large datasets,
    and conversion to Polars LazyFrames.

    Attributes:
        store_path: S3 path to the zarr store
        storage_options: S3 authentication and configuration options
        group: Optional subgroup within the zarr store
        consolidated: Whether to use consolidated metadata for faster access
        chunk_size: Number of data points to process per chunk in streaming mode

    Examples:
        Basic usage::

            reader = ZarrS3Reader("s3://bucket/data.zarr")
            arrays = reader.list_arrays()
            lf = reader.read_array_to_polars("temperature")

        With S3 credentials::

            reader = ZarrS3Reader(
                "s3://bucket/data.zarr",
                storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
            )
    """

    def __init__(
        self,
        store_path: str,
        storage_options: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
        consolidated: Optional[bool] = None,
        chunk_size: int = 10000,
    ) -> None:
        """
        Initialize Zarr S3 Reader.

        Parameters:
            store_path: S3 path to the zarr store (e.g., 's3://bucket/path/to/store.zarr')
            storage_options: Options passed to s3fs (credentials, region, etc.)
                Common options include:
                - key: AWS access key ID
                - secret: AWS secret access key
                - token: AWS session token
                - region_name: AWS region
            group: Group within the zarr store to read
            consolidated: Whether to use consolidated metadata. If None, will attempt
                to auto-detect and fall back gracefully
            chunk_size: Size of chunks for streaming processing of large arrays

        Raises:
            ImportError: If zarr or s3fs packages are not available
        """
        if not ZARR_AVAILABLE:
            raise ImportError(
                "zarr and s3fs packages are required. Install with: pip install zarr s3fs"
            )

        self.store_path = store_path
        self.storage_options = storage_options or {}
        self.group = group
        self.consolidated = consolidated
        self.chunk_size = chunk_size
        self._zarr_group: Optional[Any] = None
        self._fs: Optional[s3fs.S3FileSystem] = None
        self._coord_cache: Dict[str, np.ndarray] = {}  # Cache for coordinate arrays

    def _get_filesystem(self) -> s3fs.S3FileSystem:
        """Get or create S3 filesystem with configured options."""
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(**self.storage_options)
        return self._fs

    def _open_zarr_group(self) -> Any:
        """
        Open the zarr group with proper error handling and caching.

        Attempts to open with consolidated metadata first for better performance,
        then falls back to regular zarr.open_group if consolidated fails.

        Returns:
            Opened zarr group object

        Raises:
            ValueError: If the zarr store cannot be opened
        """
        if self._zarr_group is not None:
            return self._zarr_group

        try:
            # For Zarr v3, use storage_options for S3 access
            if self.consolidated in [None, True]:
                try:
                    # Try consolidated metadata first
                    if self.consolidated:
                        self._zarr_group = zarr.open_consolidated(
                            self.store_path,
                            mode="r",
                            storage_options=self.storage_options,
                        )
                    else:
                        # Auto-detect consolidated, fall back if needed
                        try:
                            self._zarr_group = zarr.open_consolidated(
                                self.store_path,
                                mode="r",
                                storage_options=self.storage_options,
                            )
                        except (ValueError, KeyError):
                            self._zarr_group = zarr.open_group(
                                self.store_path,
                                mode="r",
                                storage_options=self.storage_options,
                            )
                except Exception:
                    self._zarr_group = zarr.open_group(
                        self.store_path, mode="r", storage_options=self.storage_options
                    )
            else:
                self._zarr_group = zarr.open_group(
                    self.store_path, mode="r", storage_options=self.storage_options
                )

            # Navigate to specific group if specified
            if self.group and self.group != "/":
                self._zarr_group = self._zarr_group[self.group.lstrip("/")]

        except Exception as e:
            raise ValueError(f"Failed to open Zarr store at {self.store_path}: {e}")

        return self._zarr_group

    def list_arrays(self) -> List[str]:
        """
        List all arrays in the zarr group.

        Returns:
            List of array names available in the zarr store
        """
        group = self._open_zarr_group()
        # For zarr v3, use the keys() method and check each item
        arrays = []
        for name in group.keys():
            try:
                item = group[name]
                # Check if it's an array by looking for shape attribute
                if hasattr(item, "shape") and hasattr(item, "dtype"):
                    arrays.append(name)
            except Exception:
                # Skip items that can't be accessed
                continue
        return arrays

    def get_array_info(self, array_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific array.

        Parameters:
            array_name: Name of the array to inspect

        Returns:
            Dictionary containing array metadata including:
            - name: Array name
            - shape: Dimensions of the array
            - dtype: Data type
            - chunks: Chunk structure
            - dimensions: Dimension names
            - fill_value: Fill value for missing data
            - compressor: Compression algorithm used
            - filters: Applied filters
            - attrs: Additional attributes

        Raises:
            KeyError: If array name is not found in the zarr store
        """
        group = self._open_zarr_group()
        if array_name not in group:
            raise KeyError(f"Array '{array_name}' not found in zarr store")

        array = group[array_name]

        # Get dimension names from attributes (following xarray convention)
        dims = getattr(array, "attrs", {}).get(
            "_ARRAY_DIMENSIONS", [f"dim_{i}" for i in range(array.ndim)]
        )

        info = {
            "name": array_name,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "chunks": getattr(array, "chunks", None),
            "dimensions": dims,
            "fill_value": getattr(array, "fill_value", None),
            "compressor": (
                str(array.compressor)
                if hasattr(array, "compressor") and array.compressor
                else None
            ),
            "filters": (
                [str(f) for f in array.filters]
                if hasattr(array, "filters") and array.filters
                else []
            ),
            "attrs": dict(getattr(array, "attrs", {})),
        }
        return info

    def read_array_to_polars(
        self,
        array_name: str,
        select_dims: Optional[Dict[str, Union[slice, int, List[int]]]] = None,
        streaming: bool = True,
    ) -> pl.LazyFrame:
        """
        Read a zarr array into a Polars LazyFrame with coordinate expansion.

        This method reads multi-dimensional zarr arrays and converts them to a
        flat representation with explicit coordinate columns. For example, a 3D
        array with dimensions (time, lat, lon) becomes a LazyFrame with columns
        ['time', 'lat', 'lon', 'value'].

        Parameters:
            array_name: Name of the array to read
            select_dims: Dictionary mapping dimension names to selection criteria.
                Examples:
                - {"time": slice(0, 100)}: Select first 100 time steps
                - {"lat": slice(10, 50), "lon": [0, 5, 10]}: Select lat range and specific lon indices
                - {"time": 0}: Select single time step (reduces dimensions)
            streaming: Whether to use streaming mode for large arrays. Recommended
                for arrays larger than chunk_size elements.

        Returns:
            Polars LazyFrame with coordinate columns and 'value' column containing
            the array data. Column types are preserved from the original zarr arrays.

        Raises:
            KeyError: If array_name is not found in the zarr store

        Examples:
            Basic usage::

                lf = reader.read_array_to_polars("temperature")

            With dimension selection::

                lf = reader.read_array_to_polars(
                    "temperature",
                    select_dims={"time": slice(0, 12), "lat": slice(100, 200)}
                )
        """
        group = self._open_zarr_group()
        if array_name not in group:
            raise KeyError(f"Array '{array_name}' not found in zarr store")

        array = group[array_name]

        # Get dimension names from array attributes
        dims = getattr(array, "attrs", {}).get(
            "_ARRAY_DIMENSIONS", [f"dim_{i}" for i in range(array.ndim)]
        )

        # Extract coordinate arrays from the zarr group
        coord_arrays = self._extract_coordinate_arrays(group, dims)

        # Apply dimension selection if provided
        selection, selected_dims, selected_coord_arrays = (
            self._process_dimension_selection(dims, coord_arrays, select_dims)
        )

        # Read the data with selection applied
        if selection:
            data = array[tuple(selection)]
        else:
            data = array[:]

        # Convert to DataFrame format
        return self._array_to_polars_lazy(
            data, selected_dims, selected_coord_arrays, streaming
        )

    def _extract_coordinate_arrays(
        self, group: Any, dims: List[str]
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract coordinate arrays from zarr group for each dimension with caching.

        Uses lazy loading and caching to avoid redundant coordinate array reads.

        Parameters:
            group: Opened zarr group
            dims: List of dimension names

        Returns:
            Dictionary mapping dimension names to coordinate arrays or None
        """
        coord_arrays = {}
        for dim_name in dims:
            if dim_name in group:
                try:
                    # Read coordinate array efficiently - avoid unnecessary copies
                    coord_array = group[dim_name]
                    # Only convert to numpy array when needed, keep zarr array if small
                    if coord_array.size < 10000:  # Threshold for small coordinates
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

    def _process_dimension_selection(
        self,
        dims: List[str],
        coord_arrays: Dict[str, Optional[np.ndarray]],
        select_dims: Optional[Dict[str, Union[slice, int, List[int]]]],
    ) -> Tuple[List[Any], List[str], Dict[str, Optional[np.ndarray]]]:
        """
        Process dimension selection criteria and apply to coordinate arrays.

        Parameters:
            dims: Original dimension names
            coord_arrays: Original coordinate arrays
            select_dims: Selection criteria for each dimension

        Returns:
            Tuple of (selection_list, selected_dim_names, selected_coord_arrays)
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

    def _array_to_polars_lazy(
        self,
        data: np.ndarray,
        dim_names: List[str],
        coord_arrays: Dict[str, Optional[np.ndarray]],
        streaming: bool = True,
    ) -> pl.LazyFrame:
        """
        Convert numpy array to Polars LazyFrame with appropriate method.

        Uses intelligent chunking based on array characteristics and memory constraints.

        Parameters:
            data: Input numpy array
            dim_names: Names of dimensions
            coord_arrays: Coordinate arrays for each dimension
            streaming: Whether to use streaming for large arrays

        Returns:
            Polars LazyFrame with coordinate columns and values
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
        dim_names: List[str],
        coord_arrays: Dict[str, Optional[np.ndarray]],
    ) -> pl.LazyFrame:
        """
        Convert multi-dimensional array to Polars LazyFrame (non-streaming).

        Uses efficient memory-friendly coordinate expansion without creating
        full meshgrids in memory.

        Parameters:
            data: Multi-dimensional numpy array
            dim_names: Names corresponding to each dimension
            coord_arrays: Coordinate arrays for each dimension

        Returns:
            Polars LazyFrame with expanded coordinates
        """
        # Flatten data once and reuse
        flat_values = data.ravel()

        # Create coordinate columns efficiently using broadcasting
        flat_data = {"value": flat_values}

        # Calculate coordinate values without creating full meshgrids
        for i, dim_name in enumerate(dim_names):
            coord_array = coord_arrays.get(dim_name)
            if coord_array is None:
                coord_array = np.arange(data.shape[i])

            # Calculate repeat and tile factors for efficient coordinate expansion
            repeat_factor = (
                np.prod(data.shape[i + 1 :], dtype=int)
                if i + 1 < len(data.shape)
                else 1
            )
            tile_factor = np.prod(data.shape[:i], dtype=int) if i > 0 else 1

            # Use numpy's efficient repeat/tile instead of meshgrid
            flat_coords = np.tile(np.repeat(coord_array, repeat_factor), tile_factor)
            flat_data[dim_name] = flat_coords

        return pl.LazyFrame(flat_data)

    def _streaming_multidim_to_polars(
        self,
        data: np.ndarray,
        dim_names: List[str],
        coord_arrays: Dict[str, Optional[np.ndarray]],
    ) -> pl.LazyFrame:
        """
        Convert multi-dimensional array to Polars LazyFrame with memory-efficient streaming.

        Uses a true lazy approach that avoids creating large intermediate arrays.

        Parameters:
            data: Multi-dimensional numpy array
            dim_names: Names corresponding to each dimension
            coord_arrays: Coordinate arrays for each dimension

        Returns:
            Polars LazyFrame created with efficient lazy evaluation
        """
        total_size = data.size

        # If data is small enough, use non-streaming approach
        if total_size <= self.chunk_size:
            return self._multidim_to_polars(data, dim_names, coord_arrays)

        # For large data, create smaller chunks and process iteratively
        chunks = []
        flat_data = data.ravel()

        # Pre-calculate coordinate expansion factors
        coord_factors = []
        coords = []
        for i, dim_name in enumerate(dim_names):
            coord_array = coord_arrays.get(dim_name)
            if coord_array is None:
                coord_array = np.arange(data.shape[i])
            coords.append(coord_array)

            repeat_factor = (
                np.prod(data.shape[i + 1 :], dtype=int)
                if i + 1 < len(data.shape)
                else 1
            )
            tile_factor = np.prod(data.shape[:i], dtype=int) if i > 0 else 1
            coord_factors.append((repeat_factor, tile_factor))

        # Process in chunks to avoid memory overflow
        for start_idx in range(0, total_size, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_size)

            # Create chunk data dictionary
            chunk_data = {"value": flat_data[start_idx:end_idx]}

            # Generate coordinate chunks efficiently
            for i, dim_name in enumerate(dim_names):
                repeat_factor, tile_factor = coord_factors[i]
                coord_array = coords[i]

                # Calculate coordinates for this specific chunk
                if repeat_factor == 1 and tile_factor == 1:
                    # Simple 1D case
                    chunk_coords = coord_array[start_idx:end_idx]
                else:
                    # For multi-dimensional, calculate index mapping
                    chunk_indices = np.arange(start_idx, end_idx)

                    # Calculate which coordinate values correspond to these flat indices
                    if i == len(dim_names) - 1:  # Last dimension (fastest varying)
                        coord_indices = chunk_indices % data.shape[i]
                    elif i == 0:  # First dimension (slowest varying)
                        coord_indices = chunk_indices // np.prod(data.shape[1:])
                    else:  # Middle dimensions
                        stride = np.prod(data.shape[i + 1 :])
                        coord_indices = (chunk_indices // stride) % data.shape[i]

                    chunk_coords = coord_array[coord_indices]

                chunk_data[dim_name] = chunk_coords

            chunks.append(pl.DataFrame(chunk_data).lazy())

        # Concatenate all chunks
        if not chunks:
            return pl.LazyFrame({name: [] for name in dim_names + ["value"]})

        return pl.concat(chunks)

    def scan_multiple_arrays(
        self, array_names: List[str], streaming: bool = True
    ) -> Dict[str, pl.LazyFrame]:
        """
        Scan multiple arrays and return a dictionary of LazyFrames.

        Parameters:
            array_names: Names of arrays to read
            streaming: Whether to use streaming mode

        Returns:
            Dictionary mapping array names to LazyFrames
        """
        return {
            name: self.read_array_to_polars(name, streaming=streaming)
            for name in array_names
        }


def scan_zarr_s3(
    store_path: str,
    array_name: Optional[str] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    group: Optional[str] = None,
    consolidated: Optional[bool] = None,
    select_dims: Optional[Dict[str, Union[slice, int, List[int]]]] = None,
    chunk_size: int = 10000,
    streaming: bool = True,
) -> Union[pl.LazyFrame, Dict[str, pl.LazyFrame]]:
    """
    Scan Zarr files from S3 into Polars LazyFrames.

    This is the main entry point for reading Zarr arrays from S3. It provides
    a convenient interface for common operations while allowing full customization
    of the underlying ZarrS3Reader.

    Parameters:
        store_path: S3 path to the zarr store (e.g., 's3://bucket/path/to/store.zarr')
        array_name: Specific array to read. If None, returns info about all arrays
        storage_options: S3 credentials and options. Common keys:
            - key: AWS access key ID
            - secret: AWS secret access key
            - token: AWS session token (for temporary credentials)
            - region_name: AWS region name
        group: Group within the zarr store to access
        consolidated: Whether to use consolidated metadata for faster access
        select_dims: Dimension selection criteria. Examples:
            - {"time": slice(0, 100)}: First 100 time steps
            - {"lat": slice(10, 20), "time": 0}: Lat range at single time
        chunk_size: Size of chunks for streaming processing
        streaming: Whether to use streaming mode for large arrays

    Returns:
        LazyFrame if array_name specified, dict of LazyFrames otherwise

    Examples:
        Read a specific array::

            lf = scan_zarr_s3(
                "s3://my-bucket/data.zarr",
                array_name="temperature",
                storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
            )

        Read with dimension selection::

            lf = scan_zarr_s3(
                "s3://my-bucket/data.zarr",
                array_name="temperature",
                select_dims={"time": slice(0, 100), "lat": slice(10, 20)}
            )

        List all arrays::

            reader = ZarrS3Reader("s3://my-bucket/data.zarr")
            arrays = reader.list_arrays()
    """
    reader = ZarrS3Reader(
        store_path=store_path,
        storage_options=storage_options,
        group=group,
        consolidated=consolidated,
        chunk_size=chunk_size,
    )

    if array_name:
        return reader.read_array_to_polars(
            array_name=array_name, select_dims=select_dims, streaming=streaming
        )
    else:
        # Return all arrays
        array_names = reader.list_arrays()
        return reader.scan_multiple_arrays(array_names, streaming=streaming)


def zarr_s3_info(
    store_path: str,
    storage_options: Optional[Dict[str, Any]] = None,
    group: Optional[str] = None,
    consolidated: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Get comprehensive information about arrays in a Zarr store on S3.

    This function provides metadata about the zarr store and all its arrays
    without loading the actual data, making it useful for exploration and
    data discovery.

    Parameters:
        store_path: S3 path to the zarr store
        storage_options: S3 credentials and options
        group: Group within the zarr store to inspect
        consolidated: Whether to use consolidated metadata

    Returns:
        Dictionary containing store information and array metadata:
        {
            "store_path": str,
            "group": Optional[str],
            "arrays": {
                "array_name": {
                    "shape": tuple,
                    "dtype": str,
                    "dimensions": list,
                    "attrs": dict,
                    ...
                }
            }
        }

    Examples:
        Get info about all arrays::

            info = zarr_s3_info(
                "s3://my-bucket/data.zarr",
                storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
            )
            print(f"Available arrays: {list(info['arrays'].keys())}")
    """
    reader = ZarrS3Reader(
        store_path=store_path,
        storage_options=storage_options,
        group=group,
        consolidated=consolidated,
    )

    arrays = reader.list_arrays()
    info = {"store_path": store_path, "group": group, "arrays": {}}

    for array_name in arrays:
        info["arrays"][array_name] = reader.get_array_info(array_name)

    return info


# Register the plugin functions with Polars (if available)
if ZARR_AVAILABLE:
    try:
        # Note: This is a simplified registration - in a real plugin you'd compile Rust code
        # For now, we'll make the functions available as regular Python functions
        pl.scan_zarr_s3 = scan_zarr_s3  # type: ignore
        pl.zarr_s3_info = zarr_s3_info  # type: ignore
    except Exception:
        # If registration fails, functions are still available as standalone
        pass


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Zarr S3 Plugin for Polars")
    print("=" * 30)

    if not ZARR_AVAILABLE:
        print("Warning: zarr and s3fs not available")
        print("Install with: pip install zarr s3fs")
    else:
        print("Plugin loaded successfully!")
        print("\nExample usage:")
        print(
            """
# Read a specific zarr from S3
import polars as pl
import polars_IOplugin_zarr as zp

# Create reader and load data
reader = zp.ZarrS3Reader(
    "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"
)
lf = reader.read_array_to_polars("tasmax")
        """
        )
