#!/usr/bin/env python3
"""
Enhanced Zarr S3 Reader with xarray-like metadata access for Polars DataFrames.

This module provides a ZarrS3Reader that creates Polars DataFrames with
xarray-like metadata access patterns, including:
- dataset.attrs['key'] access pattern
- coordinate metadata preservation
- variable metadata preservation
- Rich metadata structure that mirrors xarray
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import s3fs
import xarray as xr
import zarr

# Import the optimized reader
from climakitae.new_core.polars_IOplugin_zarr import ZarrS3Reader as BaseZarrS3Reader


class MetadataDict(dict):
    """
    A dictionary subclass that provides xarray-like attribute access.

    This allows accessing metadata like: metadata.attrs['key']
    while still being a regular dictionary for all other operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        """Allow bracket notation access."""
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Allow bracket notation assignment."""
        super().__setitem__(key, value)

    def get(self, key, default=None):
        """Get with default value."""
        return super().get(key, default)


class PolarsDataFrameWithMetadata:
    """
    A wrapper around Polars DataFrame/LazyFrame that provides xarray-like metadata access.

    This class maintains metadata separately while providing access patterns that
    mirror xarray's .attrs behavior.
    """

    def __init__(
        self,
        dataframe: Union[pl.DataFrame, pl.LazyFrame],
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize the metadata-aware DataFrame wrapper.

        Parameters
        ----------
        dataframe : Union[pl.DataFrame, pl.LazyFrame]
            The underlying Polars DataFrame or LazyFrame
        metadata : Dict[str, Any], optional
            Metadata dictionary containing dataset, coordinate, and variable metadata
        """
        self._dataframe = dataframe
        self._metadata = metadata or {}

        # Create xarray-like attrs access
        self.attrs = MetadataDict(self._metadata.get("dataset_attrs", {}))

        # Create coordinate metadata access
        self.coords_metadata = {
            name: MetadataDict(meta.get("attrs", {}))
            for name, meta in self._metadata.get("coordinates", {}).items()
        }

        # Create variable metadata access
        self.vars_metadata = {
            name: MetadataDict(meta.get("attrs", {}))
            for name, meta in self._metadata.get("data_vars", {}).items()
        }

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying DataFrame."""
        return getattr(self._dataframe, name)

    def __repr__(self):
        """Enhanced representation showing metadata info."""
        df_repr = repr(self._dataframe)

        # Add metadata summary
        metadata_info = []
        if self.attrs:
            metadata_info.append(f"Dataset attributes: {len(self.attrs)} items")
        if self.coords_metadata:
            metadata_info.append(
                f"Coordinate metadata: {len(self.coords_metadata)} coords"
            )
        if self.vars_metadata:
            metadata_info.append(f"Variable metadata: {len(self.vars_metadata)} vars")

        if metadata_info:
            metadata_summary = "Metadata: " + ", ".join(metadata_info)
            return f"{df_repr}\n{metadata_summary}"
        return df_repr

    @property
    def dataframe(self) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Access the underlying Polars DataFrame/LazyFrame."""
        return self._dataframe

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access the complete metadata dictionary."""
        return self._metadata

    def get_coord_attrs(self, coord_name: str) -> MetadataDict:
        """
        Get attributes for a specific coordinate.

        Parameters
        ----------
        coord_name : str
            Name of the coordinate

        Returns
        -------
        MetadataDict
            Coordinate attributes with xarray-like access
        """
        return self.coords_metadata.get(coord_name, MetadataDict())

    def get_var_attrs(self, var_name: str) -> MetadataDict:
        """
        Get attributes for a specific variable.

        Parameters
        ----------
        var_name : str
            Name of the variable

        Returns
        -------
        MetadataDict
            Variable attributes with xarray-like access
        """
        return self.vars_metadata.get(var_name, MetadataDict())

    def collect(self) -> "PolarsDataFrameWithMetadata":
        """
        If this wraps a LazyFrame, collect it and return a new wrapper with DataFrame.
        """
        if isinstance(self._dataframe, pl.LazyFrame):
            return PolarsDataFrameWithMetadata(
                self._dataframe.collect(), self._metadata
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Export both data and metadata to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'data' and 'metadata' keys
        """
        return {
            "data": (
                self._dataframe.to_pandas()
                if hasattr(self._dataframe, "to_pandas")
                else None
            ),
            "metadata": self._metadata,
        }

    def save_metadata(self, filepath: Union[str, Path]):
        """
        Save metadata to a JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to save the metadata file
        """
        with open(filepath, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    @classmethod
    def load_metadata(
        cls,
        dataframe: Union[pl.DataFrame, pl.LazyFrame],
        metadata_filepath: Union[str, Path],
    ) -> "PolarsDataFrameWithMetadata":
        """
        Create a metadata-aware DataFrame by loading metadata from a file.

        Parameters
        ----------
        dataframe : Union[pl.DataFrame, pl.LazyFrame]
            The Polars DataFrame/LazyFrame
        metadata_filepath : Union[str, Path]
            Path to the metadata JSON file

        Returns
        -------
        PolarsDataFrameWithMetadata
            New instance with loaded metadata
        """
        with open(metadata_filepath, "r") as f:
            metadata = json.load(f)
        return cls(dataframe, metadata)


class EnhancedZarrS3Reader(BaseZarrS3Reader):
    """
    Enhanced Zarr S3 Reader that provides xarray-like metadata access with Polars DataFrames.

    This class extends the base ZarrS3Reader to include comprehensive metadata extraction
    and preservation capabilities, providing access patterns that mirror xarray's API.
    """

    def __init__(self, store_path: str, **storage_options):
        """
        Initialize the enhanced reader.

        Parameters
        ----------
        store_path : str
            S3 path to the zarr store
        **storage_options
            Additional storage options for s3fs
        """
        super().__init__(store_path, **storage_options)
        self._xarray_metadata = None
        self._xarray_dataset = None

    def _extract_xarray_metadata(self) -> Dict[str, Any]:
        """
        Extract comprehensive metadata using xarray as reference.

        This method opens the same zarr store with xarray to extract all
        metadata in the same structure that xarray provides.

        Returns
        -------
        Dict[str, Any]
            Complete metadata structure matching xarray's organization
        """
        if self._xarray_metadata is not None:
            return self._xarray_metadata

        try:
            # Open with xarray to get metadata structure
            ds = xr.open_zarr(self.store_path, consolidated=True)

            metadata = {
                "dataset_attrs": dict(ds.attrs),
                "coordinates": {},
                "data_vars": {},
                "dimensions": dict(ds.dims),
                "encoding": getattr(ds, "encoding", {}),
            }

            # Extract coordinate metadata
            for coord_name, coord in ds.coords.items():
                metadata["coordinates"][coord_name] = {
                    "attrs": dict(coord.attrs),
                    "dtype": str(coord.dtype),
                    "shape": coord.shape,
                    "dims": coord.dims,
                    "encoding": getattr(coord, "encoding", {}),
                }

            # Extract data variable metadata
            for var_name, var in ds.data_vars.items():
                metadata["data_vars"][var_name] = {
                    "attrs": dict(var.attrs),
                    "dtype": str(var.dtype),
                    "shape": var.shape,
                    "dims": var.dims,
                    "encoding": getattr(var, "encoding", {}),
                }

            # Store for reuse
            self._xarray_metadata = metadata
            self._xarray_dataset = ds

            return metadata

        except Exception as e:
            print(f"Warning: Could not extract xarray metadata: {e}")
            return {}

    def read_array_to_polars_with_metadata(
        self, array_name: str, streaming: bool = True, **kwargs
    ) -> PolarsDataFrameWithMetadata:
        """
        Read a zarr array to a Polars DataFrame with comprehensive metadata.

        Parameters
        ----------
        array_name : str
            Name of the array to read
        streaming : bool, default True
            Whether to use streaming approach
        **kwargs
            Additional arguments passed to the base reader

        Returns
        -------
        PolarsDataFrameWithMetadata
            Polars DataFrame wrapper with xarray-like metadata access
        """
        # Get the base Polars LazyFrame
        lf = self.read_array_to_polars(array_name, streaming=streaming, **kwargs)

        # Extract comprehensive metadata
        metadata = self._extract_xarray_metadata()

        # Create enhanced wrapper
        return PolarsDataFrameWithMetadata(lf, metadata)

    def read_dataset_to_polars_with_metadata(
        self, variable_names: Optional[List[str]] = None, streaming: bool = True
    ) -> Dict[str, PolarsDataFrameWithMetadata]:
        """
        Read multiple arrays as metadata-aware Polars DataFrames.

        Parameters
        ----------
        variable_names : Optional[List[str]]
            List of variable names to read. If None, reads all data variables.
        streaming : bool, default True
            Whether to use streaming approach

        Returns
        -------
        Dict[str, PolarsDataFrameWithMetadata]
            Dictionary mapping variable names to enhanced DataFrames
        """
        # Extract metadata once
        metadata = self._extract_xarray_metadata()

        # Determine variables to read
        if variable_names is None:
            variable_names = list(metadata.get("data_vars", {}).keys())

        result = {}
        for var_name in variable_names:
            try:
                lf = self.read_array_to_polars(var_name, streaming=streaming)
                result[var_name] = PolarsDataFrameWithMetadata(lf, metadata)
            except Exception as e:
                print(f"Warning: Could not read variable {var_name}: {e}")

        return result

    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available metadata.

        Returns
        -------
        Dict[str, Any]
            Summary of metadata structure and contents
        """
        metadata = self._extract_xarray_metadata()

        summary = {
            "dataset_attributes_count": len(metadata.get("dataset_attrs", {})),
            "coordinates_count": len(metadata.get("coordinates", {})),
            "data_variables_count": len(metadata.get("data_vars", {})),
            "dimensions": metadata.get("dimensions", {}),
            "coordinate_names": list(metadata.get("coordinates", {}).keys()),
            "data_variable_names": list(metadata.get("data_vars", {}).keys()),
        }

        # Add sample attributes
        if metadata.get("dataset_attrs"):
            sample_attrs = dict(list(metadata["dataset_attrs"].items())[:5])
            summary["sample_dataset_attributes"] = sample_attrs

        return summary

    def compare_with_xarray(self, array_name: str) -> Dict[str, Any]:
        """
        Compare data and metadata with xarray for validation.

        Parameters
        ----------
        array_name : str
            Name of the array to compare

        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        # Get Polars data
        polars_enhanced = self.read_array_to_polars_with_metadata(
            array_name, streaming=False
        )
        polars_df = polars_enhanced.collect().dataframe

        # Get xarray data
        if self._xarray_dataset is None:
            self._extract_xarray_metadata()

        xarray_data = self._xarray_dataset[array_name].values.ravel()
        polars_data = polars_df["value"].to_numpy()

        # Compare data
        data_match = np.array_equal(xarray_data, polars_data, equal_nan=True)

        # Compare metadata access
        xr_attrs = dict(self._xarray_dataset[array_name].attrs)
        polars_attrs = dict(polars_enhanced.get_var_attrs(array_name))

        metadata_match = xr_attrs == polars_attrs

        return {
            "data_values_match": data_match,
            "metadata_access_match": metadata_match,
            "xarray_attrs_count": len(xr_attrs),
            "polars_attrs_count": len(polars_attrs),
            "data_shape_polars": polars_df.shape,
            "data_shape_xarray": self._xarray_dataset[array_name].shape,
            "sample_metadata_access": {
                "xarray_style": f"ds['{array_name}'].attrs = {dict(list(xr_attrs.items())[:3])}",
                "polars_style": f"df.get_var_attrs('{array_name}') = {dict(list(polars_attrs.items())[:3])}",
            },
        }


def main():
    """Demonstrate the enhanced metadata capabilities."""
    print("Enhanced Zarr S3 Reader with xarray-like Metadata Access")
    print("=" * 60)

    s3_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    print(f"Loading dataset from: {s3_path}")
    print()

    # Create enhanced reader
    reader = EnhancedZarrS3Reader(s3_path)

    # Get metadata summary
    print("Metadata Summary:")
    print("-" * 20)
    summary = reader.get_metadata_summary()
    for key, value in summary.items():
        if isinstance(value, dict) and len(str(value)) > 100:
            print(f"{key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"{key}: {value}")
    print()

    # Read data with metadata
    print("Reading tasmax with metadata...")
    tasmax_enhanced = reader.read_array_to_polars_with_metadata(
        "tasmax", streaming=False
    )

    print("Enhanced DataFrame representation:")
    print(tasmax_enhanced)
    print()

    # Demonstrate xarray-like metadata access
    print("Demonstrating xarray-like metadata access:")
    print("-" * 45)

    # Dataset-level attributes (like xarray ds.attrs)
    print(f"Dataset attributes count: {len(tasmax_enhanced.attrs)}")
    if tasmax_enhanced.attrs:
        print("Sample dataset attributes:")
        for i, (key, value) in enumerate(list(tasmax_enhanced.attrs.items())[:3]):
            print(
                f"  tasmax_enhanced.attrs['{key}'] = {str(value)[:60]}{'...' if len(str(value)) > 60 else ''}"
            )
    print()

    # Variable metadata (like xarray da.attrs)
    tasmax_attrs = tasmax_enhanced.get_var_attrs("tasmax")
    print(f"Variable 'tasmax' attributes: {len(tasmax_attrs)}")
    for key, value in tasmax_attrs.items():
        print(f"  tasmax_enhanced.get_var_attrs('tasmax')['{key}'] = {value}")
    print()

    # Coordinate metadata
    print("Coordinate metadata:")
    for coord_name in ["time", "lat", "lon"]:
        coord_attrs = tasmax_enhanced.get_coord_attrs(coord_name)
        if coord_attrs:
            print(f"  {coord_name} attributes: {dict(coord_attrs)}")
    print()

    # Compare with xarray
    print("Comparing with xarray for validation...")
    comparison = reader.compare_with_xarray("tasmax")
    print("Comparison results:")
    for key, value in comparison.items():
        print(f"  {key}: {value}")
    print()

    # Demonstrate metadata persistence
    print("Demonstrating metadata persistence:")
    print("-" * 35)

    # Save metadata to file
    metadata_file = "tasmax_metadata.json"
    tasmax_enhanced.save_metadata(metadata_file)
    print(f"Metadata saved to: {metadata_file}")

    # Show how to reload
    df_collected = tasmax_enhanced.collect().dataframe
    restored_enhanced = PolarsDataFrameWithMetadata.load_metadata(
        df_collected, metadata_file
    )
    print(
        f"Metadata successfully restored: {len(restored_enhanced.attrs)} dataset attributes"
    )

    # Verify access pattern works after restoration
    print("Restored metadata access:")
    if restored_enhanced.attrs:
        sample_key = list(restored_enhanced.attrs.keys())[0]
        print(
            f"  restored_enhanced.attrs['{sample_key}'] = {restored_enhanced.attrs[sample_key]}"
        )


if __name__ == "__main__":
    main()
