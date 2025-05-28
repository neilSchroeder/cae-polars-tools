"""
S3 filesystem and Zarr storage management for climate data access.

This module handles the low-level S3 connectivity and Zarr store access,
providing a clean interface for managing cloud-based climate datasets.
It abstracts away the complexity of S3 authentication, zarr store opening,
and metadata handling with intelligent fallback strategies.

The S3ZarrStore class serves as the foundation for all cloud-based zarr
data access, handling connection management, error recovery, and optimized
metadata access for improved performance.

Storage Capabilities
--------------------
- Robust S3 filesystem connection management
- Intelligent zarr store opening with consolidated metadata support
- Automatic fallback from consolidated to regular metadata
- Comprehensive array inspection and metadata extraction
- Error handling for common cloud storage issues
- Support for zarr groups and hierarchical data organization

Examples
--------
Basic S3 zarr store access:

>>> from cae_polars.data_access.zarr_storage import S3ZarrStore
>>>
>>> # Public dataset access
>>> store = S3ZarrStore(
...     "s3://noaa-global-warming-pds/climate.zarr",
...     storage_options={"anon": True}
... )
>>> arrays = store.list_arrays()
>>> print("Available arrays:", arrays)

With authentication:

>>> store = S3ZarrStore(
...     "s3://private-bucket/climate-data.zarr",
...     storage_options={
...         "key": "ACCESS_KEY_ID",
...         "secret": "SECRET_ACCESS_KEY",
...         "region_name": "us-west-2"
...     }
... )

Working with zarr groups:

>>> # Access specific group within store
>>> store = S3ZarrStore(
...     "s3://bucket/data.zarr",
...     group="reanalysis/era5",
...     consolidated=True
... )
>>>
>>> # Get detailed array information
>>> temp_info = store.get_array_info("temperature")
>>> print(f"Shape: {temp_info['shape']}")
>>> print(f"Chunks: {temp_info['chunks']}")

Array access and inspection:

>>> # Get zarr array object
>>> temp_array = store.get_array("temperature")
>>> data_subset = temp_array[0:10, :, :]  # Read subset
>>>
>>> # Comprehensive metadata
>>> all_info = {name: store.get_array_info(name) for name in store.list_arrays()}

Notes
-----
This module requires the 'zarr' and 's3fs' packages for cloud storage access.
The class handles various zarr versions and provides graceful fallbacks for
different metadata configurations. It's designed to work reliably with both
consolidated and non-consolidated zarr stores.
"""

from __future__ import annotations

import warnings
from typing import Any

try:
    import s3fs
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    warnings.warn(
        "zarr and s3fs packages required for Zarr S3 functionality. "
        "Install with: pip install zarr s3fs",
        stacklevel=2,
    )


class S3ZarrStore:
    """
    Manages S3 filesystem connections and Zarr store access for climate data.

    This class handles the connection to S3 storage and provides optimized
    access to Zarr stores with automatic fallback for metadata handling.
    It serves as the foundation for cloud-based zarr data access, managing
    authentication, connection pooling, and intelligent metadata strategies.

    Parameters
    ----------
    store_path : str
        S3 path to the zarr store (e.g., 's3://bucket/path/to/store.zarr')
    storage_options : dict, optional
        Options passed to s3fs for authentication and configuration.
        Common options include:
        - key: AWS access key ID
        - secret: AWS secret access key
        - token: AWS session token
        - region_name: AWS region
        - anon: True for anonymous/public access
    group : str, optional
        Group within the zarr store to read. Supports hierarchical organization
    consolidated : bool, optional
        Whether to use consolidated metadata. If None, will attempt
        to auto-detect and fall back gracefully for maximum compatibility

    Attributes
    ----------
    store_path : str
        S3 path to the zarr store
    storage_options : dict
        S3 authentication and configuration options
    group : str or None
        Optional subgroup within the zarr store
    consolidated : bool or None
        Whether to use consolidated metadata for faster access

    Examples
    --------
    Basic usage with public data:

    >>> store = S3ZarrStore(
    ...     "s3://noaa-climate/data.zarr",
    ...     storage_options={"anon": True}
    ... )
    >>> arrays = store.list_arrays()

    With authentication:

    >>> store = S3ZarrStore(
    ...     "s3://private-bucket/data.zarr",
    ...     storage_options={
    ...         "key": "ACCESS_KEY",
    ...         "secret": "SECRET_KEY"
    ...     }
    ... )

    Using groups and consolidated metadata:

    >>> store = S3ZarrStore(
    ...     "s3://bucket/hierarchical.zarr",
    ...     group="climate/reanalysis",
    ...     consolidated=True
    ... )

    Notes
    -----
    The store automatically handles zarr version differences and provides
    robust error handling for common cloud storage issues. Consolidated
    metadata is preferred when available for better performance, with
    graceful fallback to regular metadata access.
    """

    def __init__(
        self,
        store_path: str,
        storage_options: dict[str, Any] | None = None,
        group: str | None = None,
        consolidated: bool | None = None,
    ) -> None:
        if not ZARR_AVAILABLE:
            raise ImportError(
                "zarr and s3fs packages are required. Install with: pip install zarr s3fs"
            )

        self.store_path = store_path
        self.storage_options = storage_options or {}
        self.group = group
        self.consolidated = consolidated
        self._zarr_group: Any | None = None
        self._fs: s3fs.S3FileSystem | None = None

    def get_filesystem(self) -> s3fs.S3FileSystem:
        """
        Get or create S3 filesystem with configured options.

        Creates a cached S3FileSystem instance using the provided storage
        options. The filesystem is reused for multiple operations to improve
        performance and connection management.

        Returns
        -------
        s3fs.S3FileSystem
            Configured S3 filesystem instance with authentication and
            connection settings applied

        Notes
        -----
        The filesystem is created lazily and cached for reuse. Authentication
        credentials and regional settings are applied from storage_options.
        """
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(**self.storage_options)
        return self._fs

    def open_zarr_group(self) -> Any:
        """
        Open the zarr group with proper error handling and caching.

        Attempts to open with consolidated metadata first for better performance,
        then falls back to regular zarr.open_group if consolidated fails. The
        opened group is cached for subsequent access.

        Returns
        -------
        zarr.Group
            Opened zarr group object with all arrays and subgroups accessible

        Raises
        ------
        ValueError
            If the zarr store cannot be opened with any available method

        Notes
        -----
        This method implements a robust opening strategy:
        1. Try consolidated metadata if requested or auto-detect mode
        2. Fall back to regular zarr group opening if consolidated fails
        3. Navigate to specific subgroup if specified
        4. Cache the result for future access
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

    def list_arrays(self) -> list[str]:
        """
        List all arrays available in the zarr group.

        Scans the zarr group to identify all array objects, distinguishing
        them from subgroups and other metadata. This method is compatible
        with different zarr versions and handles access errors gracefully.

        Returns
        -------
        list of str
            List of array names available in the zarr store. Only arrays
            (not subgroups) are included in the result.

        Notes
        -----
        This method examines each item in the group to determine if it's
        an array by checking for shape and dtype attributes. Items that
        cannot be accessed are silently skipped.
        """
        group = self.open_zarr_group()
        # For zarr v3, use the keys() method and check each item
        arrays = []
        for name in group.keys():
            try:
                item = group[name]
                # Check if it's an array by looking for shape attribute
                if hasattr(item, "shape") and hasattr(item, "dtype"):
                    arrays.append(name)
            except (KeyError, AttributeError, TypeError) as e:
                # Skip items that can't be accessed
                continue
            except Exception as e:
                # Log unexpected errors but continue processing
                warnings.warn(
                    f"Error accessing item '{name}' in zarr group: {e}",
                    stacklevel=2,
                )
                continue

        return arrays

    def get_array(self, array_name: str) -> Any:
        """
        Get a specific array from the zarr store.

        Retrieves a zarr array object that can be used for reading data,
        inspecting metadata, or performing array operations. The array
        is accessed lazily and supports standard numpy-like indexing.

        Parameters
        ----------
        array_name : str
            Name of the array to retrieve. Must exist in the zarr store.

        Returns
        -------
        zarr.Array
            Zarr array object supporting lazy data access and numpy-like
            operations including slicing, indexing, and metadata access.

        Raises
        ------
        KeyError
            If array_name is not found in the zarr store

        Examples
        --------
        >>> array = store.get_array("temperature")
        >>> print(f"Shape: {array.shape}")
        >>> data_subset = array[0:10, :, :]  # Read subset
        """
        group = self.open_zarr_group()
        if array_name not in group:
            raise KeyError(f"Array '{array_name}' not found in zarr store")
        return group[array_name]

    def get_array_info(self, array_name: str) -> dict[str, Any]:
        """
        Get comprehensive information about a specific array.

        Parameters
        ----------
        array_name : str
            Name of the array to inspect

        Returns
        -------
        dict
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

        Raises
        ------
        KeyError
            If array name is not found in the zarr store
        """
        array = self.get_array(array_name)

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
