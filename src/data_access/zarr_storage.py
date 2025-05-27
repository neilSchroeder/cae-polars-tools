"""
S3 filesystem and Zarr storage management for climate data access.

This module handles the low-level S3 connectivity and Zarr store access,
providing a clean interface for managing cloud-based climate datasets.
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
        "Install with: pip install zarr s3fs", stacklevel=2
    )


class S3ZarrStore:
    """
    Manages S3 filesystem connections and Zarr store access.

    This class handles the connection to S3 storage and provides optimized
    access to Zarr stores with automatic fallback for metadata handling.

    Parameters
    ----------
    store_path : str
        S3 path to the zarr store (e.g., 's3://bucket/path/to/store.zarr')
    storage_options : dict, optional
        Options passed to s3fs (credentials, region, etc.)
        Common options include:
        - key: AWS access key ID
        - secret: AWS secret access key
        - token: AWS session token
        - region_name: AWS region
    group : str, optional
        Group within the zarr store to read
    consolidated : bool, optional
        Whether to use consolidated metadata. If None, will attempt
        to auto-detect and fall back gracefully

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

        Returns
        -------
        s3fs.S3FileSystem
            Configured S3 filesystem instance
        """
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(**self.storage_options)
        return self._fs

    def open_zarr_group(self) -> Any:
        """
        Open the zarr group with proper error handling and caching.

        Attempts to open with consolidated metadata first for better performance,
        then falls back to regular zarr.open_group if consolidated fails.

        Returns
        -------
        zarr.Group
            Opened zarr group object

        Raises
        ------
        ValueError
            If the zarr store cannot be opened
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
        List all arrays in the zarr group.

        Returns
        -------
        list of str
            List of array names available in the zarr store
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
            except Exception:
                # Skip items that can't be accessed
                continue
        return arrays

    def get_array(self, array_name: str) -> Any:
        """
        Get a specific array from the zarr store.

        Parameters
        ----------
        array_name : str
            Name of the array to retrieve

        Returns
        -------
        zarr.Array
            Zarr array object

        Raises
        ------
        KeyError
            If array name is not found in the zarr store
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
