#!/usr/bin/env python3
"""
Comparison script between ZarrS3Reader and xarray for data integrity and metadata analysis.

This script:
1. Loads the same zarr dataset using both ZarrS3Reader and xarray
2. Compares primary data arrays (time, lat, lon, tasmax)
3. Analyzes metadata structure in xarray dataset
4. Explores options for storing metadata in Polars DataFrames
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
import xarray as xr

# Import our plugin
from climakitae.new_core.polars_IOplugin_zarr import ZarrS3Reader


def compare_arrays(
    arr1: np.ndarray, arr2: np.ndarray, name: str, tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Compare two numpy arrays and return detailed comparison results."""
    results = {
        "name": name,
        "shapes_match": arr1.shape == arr2.shape,
        "dtypes_match": arr1.dtype == arr2.dtype,
        "shape_arr1": arr1.shape,
        "shape_arr2": arr2.shape,
        "dtype_arr1": str(arr1.dtype),
        "dtype_arr2": str(arr2.dtype),
    }

    if arr1.shape == arr2.shape:
        # Check for NaN handling
        nan_mask1 = (
            np.isnan(arr1)
            if np.issubdtype(arr1.dtype, np.floating)
            else np.zeros_like(arr1, dtype=bool)
        )
        nan_mask2 = (
            np.isnan(arr2)
            if np.issubdtype(arr2.dtype, np.floating)
            else np.zeros_like(arr2, dtype=bool)
        )

        results["nan_locations_match"] = np.array_equal(nan_mask1, nan_mask2)
        results["nan_count_arr1"] = np.sum(nan_mask1)
        results["nan_count_arr2"] = np.sum(nan_mask2)

        # Compare non-NaN values
        mask = ~(nan_mask1 | nan_mask2)
        if np.any(mask):
            # Handle datetime comparison separately
            if np.issubdtype(arr1.dtype, np.datetime64) or np.issubdtype(
                arr2.dtype, np.datetime64
            ):
                # For datetime arrays, check if they're exactly equal
                results["values_close"] = np.array_equal(arr1[mask], arr2[mask])
                results["max_difference"] = (
                    0.0 if results["values_close"] else float("inf")
                )
                results["mean_difference"] = (
                    0.0 if results["values_close"] else float("inf")
                )
            else:
                diff = np.abs(arr1[mask] - arr2[mask])
                results["max_difference"] = np.max(diff)
                results["mean_difference"] = np.mean(diff)
                results["values_close"] = np.allclose(
                    arr1[mask], arr2[mask], rtol=tolerance, atol=tolerance
                )
        else:
            results["max_difference"] = 0.0
            results["mean_difference"] = 0.0
            results["values_close"] = True

            # Value ranges
            if np.any(mask):
                if np.issubdtype(arr1.dtype, np.datetime64) or np.issubdtype(
                    arr2.dtype, np.datetime64
                ):
                    # For datetime, show first and last values
                    results["range_arr1"] = (str(arr1[mask][0]), str(arr1[mask][-1]))
                    results["range_arr2"] = (str(arr2[mask][0]), str(arr2[mask][-1]))
                else:
                    results["range_arr1"] = (np.min(arr1[mask]), np.max(arr1[mask]))
                    results["range_arr2"] = (np.min(arr2[mask]), np.max(arr2[mask]))
            else:
                results["range_arr1"] = (np.nan, np.nan)
                results["range_arr2"] = (np.nan, np.nan)

    return results


def analyze_xarray_metadata(ds: xr.Dataset) -> Dict[str, Any]:
    """Analyze the metadata structure in an xarray Dataset."""
    metadata = {
        "dataset_attrs": dict(ds.attrs),
        "coordinates": {},
        "data_vars": {},
        "dimensions": dict(ds.dims),
        "encoding": {},
    }

    # Coordinate metadata
    for coord_name, coord in ds.coords.items():
        metadata["coordinates"][coord_name] = {
            "attrs": dict(coord.attrs),
            "dtype": str(coord.dtype),
            "shape": coord.shape,
            "encoding": getattr(coord, "encoding", {}),
        }

    # Data variable metadata
    for var_name, var in ds.data_vars.items():
        metadata["data_vars"][var_name] = {
            "attrs": dict(var.attrs),
            "dtype": str(var.dtype),
            "shape": var.shape,
            "dims": var.dims,
            "encoding": getattr(var, "encoding", {}),
        }

    # Dataset-level encoding
    metadata["encoding"] = getattr(ds, "encoding", {})

    return metadata


def create_polars_with_metadata(
    lf: pl.LazyFrame, metadata: Dict[str, Any]
) -> pl.LazyFrame:
    """
    Demonstrate how to attach metadata to a Polars LazyFrame.

    Note: Polars doesn't have native metadata support like xarray,
    but we can use various approaches to preserve metadata.
    """
    # Approach 1: Store metadata as a JSON string in a special column
    metadata_json = json.dumps(metadata, default=str)

    # Add metadata as a constant column (will be optimized away in most operations)
    lf_with_meta = lf.with_columns(pl.lit(metadata_json).alias("__metadata__"))

    return lf_with_meta


def extract_metadata_from_polars(lf: pl.LazyFrame) -> Dict[str, Any]:
    """Extract metadata from a Polars LazyFrame that has metadata attached."""
    try:
        # Get the metadata column
        metadata_df = lf.select("__metadata__").collect()
        if len(metadata_df) > 0:
            metadata_json = metadata_df["__metadata__"][0]
            return json.loads(metadata_json)
    except Exception as e:
        print(f"Could not extract metadata: {e}")

    return {}


def main():
    print("Zarr S3 Reader vs xarray Comparison")
    print("=" * 50)

    # Define the S3 path
    s3_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    print(f"Loading dataset from: {s3_path}")
    print()

    # Load with ZarrS3Reader
    print("Loading with ZarrS3Reader...")
    try:
        reader = ZarrS3Reader(s3_path)

        # Get array info
        arrays = reader.list_arrays()
        print(f"Available arrays: {arrays}")

        # Load primary data
        lf_tasmax = reader.read_array_to_polars("tasmax", streaming=False)
        df_tasmax = lf_tasmax.collect()

        print(f"ZarrS3Reader - tasmax shape: {df_tasmax.shape}")
        print(f"ZarrS3Reader - columns: {df_tasmax.columns}")
        print()

    except Exception as e:
        print(f"Error loading with ZarrS3Reader: {e}")
        return

    # Load with xarray
    print("Loading with xarray...")
    try:
        # Use xarray to open the same dataset
        ds = xr.open_zarr(s3_path, consolidated=True)

        print(f"xarray - Dataset dimensions: {dict(ds.dims)}")
        print(f"xarray - Data variables: {list(ds.data_vars.keys())}")
        print(f"xarray - Coordinates: {list(ds.coords.keys())}")
        print()

    except Exception as e:
        print(f"Error loading with xarray: {e}")
        return

    # Compare coordinate arrays
    print("Comparing coordinate arrays...")
    print("-" * 30)

    coordinate_comparisons = {}

    # Compare time
    if "time" in df_tasmax.columns and "time" in ds.coords:
        time_polars = df_tasmax["time"].unique().sort().to_numpy()
        time_xarray = ds.time.values

        # Convert time formats if needed for comparison
        if hasattr(time_xarray, "astype"):
            time_xarray = time_xarray.astype("datetime64[ns]")
        if hasattr(time_polars, "astype"):
            time_polars = time_polars.astype("datetime64[ns]")

        coordinate_comparisons["time"] = compare_arrays(
            time_polars, time_xarray, "time"
        )

    # Compare lat
    if "lat" in df_tasmax.columns and "lat" in ds.coords:
        lat_polars = df_tasmax["lat"].unique().sort().to_numpy()
        lat_xarray = ds.lat.values
        coordinate_comparisons["lat"] = compare_arrays(lat_polars, lat_xarray, "lat")

    # Compare lon
    if "lon" in df_tasmax.columns and "lon" in ds.coords:
        lon_polars = df_tasmax["lon"].unique().sort().to_numpy()
        lon_xarray = ds.lon.values
        coordinate_comparisons["lon"] = compare_arrays(lon_polars, lon_xarray, "lon")

    # Print coordinate comparison results
    for coord_name, results in coordinate_comparisons.items():
        print(f"{coord_name.upper()} Comparison:")
        print(f"  Shapes match: {results['shapes_match']}")
        print(f"  Dtypes match: {results['dtypes_match']}")
        if "values_close" in results:
            print(f"  Values close: {results['values_close']}")
            print(f"  Max difference: {results.get('max_difference', 'N/A')}")
        print()

    # Compare tasmax data values
    print("Comparing tasmax data values...")
    print("-" * 30)

    # Get tasmax values from both sources
    tasmax_polars = df_tasmax["value"].to_numpy()
    tasmax_xarray = ds.tasmax.values.ravel()  # Flatten to match polars format

    data_comparison = compare_arrays(tasmax_polars, tasmax_xarray, "tasmax")

    print("TASMAX Data Comparison:")
    print(f"  Shapes match: {data_comparison['shapes_match']}")
    print(f"  Dtypes match: {data_comparison['dtypes_match']}")
    print(f"  Values close: {data_comparison.get('values_close', 'N/A')}")
    print(f"  Max difference: {data_comparison.get('max_difference', 'N/A')}")
    print(f"  Mean difference: {data_comparison.get('mean_difference', 'N/A')}")
    print(f"  NaN count (Polars): {data_comparison.get('nan_count_arr1', 'N/A')}")
    print(f"  NaN count (xarray): {data_comparison.get('nan_count_arr2', 'N/A')}")
    print(f"  Value range (Polars): {data_comparison.get('range_arr1', 'N/A')}")
    print(f"  Value range (xarray): {data_comparison.get('range_arr2', 'N/A')}")
    print()

    # Analyze xarray metadata
    print("Analyzing xarray metadata structure...")
    print("-" * 40)

    metadata = analyze_xarray_metadata(ds)

    print("Dataset-level attributes:")
    print(f"  Number of dataset attributes: {len(metadata['dataset_attrs'])}")
    if metadata["dataset_attrs"]:
        print("  Sample dataset attributes:")
        for i, (key, value) in enumerate(list(metadata["dataset_attrs"].items())[:5]):
            print(
                f"    {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}"
            )
        if len(metadata["dataset_attrs"]) > 5:
            print(f"    ... and {len(metadata['dataset_attrs']) - 5} more")
    print()

    print("Coordinate metadata:")
    for coord_name, coord_meta in metadata["coordinates"].items():
        print(f"  {coord_name}:")
        print(f"    Attributes: {len(coord_meta['attrs'])}")
        print(f"    Shape: {coord_meta['shape']}")
        print(f"    Dtype: {coord_meta['dtype']}")
        if coord_meta["attrs"]:
            sample_attrs = list(coord_meta["attrs"].items())[:3]
            for attr_name, attr_value in sample_attrs:
                print(
                    f"      {attr_name}: {str(attr_value)[:50]}{'...' if len(str(attr_value)) > 50 else ''}"
                )
    print()

    print("Data variable metadata:")
    for var_name, var_meta in metadata["data_vars"].items():
        print(f"  {var_name}:")
        print(f"    Attributes: {len(var_meta['attrs'])}")
        print(f"    Shape: {var_meta['shape']}")
        print(f"    Dimensions: {var_meta['dims']}")
        print(f"    Dtype: {var_meta['dtype']}")
        if var_meta["attrs"]:
            sample_attrs = list(var_meta["attrs"].items())[:3]
            for attr_name, attr_value in sample_attrs:
                print(
                    f"      {attr_name}: {str(attr_value)[:50]}{'...' if len(str(attr_value)) > 50 else ''}"
                )
    print()

    # Demonstrate metadata preservation in Polars
    print("Demonstrating metadata preservation in Polars...")
    print("-" * 50)

    # Create a version of the LazyFrame with metadata attached
    lf_with_meta = create_polars_with_metadata(lf_tasmax, metadata)

    print("Created Polars LazyFrame with attached metadata")
    print(f"Columns with metadata: {lf_with_meta.collect_schema().names()}")

    # Extract and verify metadata
    extracted_metadata = extract_metadata_from_polars(lf_with_meta)
    print(f"Successfully extracted metadata: {len(extracted_metadata) > 0}")

    if extracted_metadata:
        print(
            f"Extracted {len(extracted_metadata.get('dataset_attrs', {}))} dataset attributes"
        )
        print(
            f"Extracted {len(extracted_metadata.get('coordinates', {}))} coordinate metadata entries"
        )
        print(
            f"Extracted {len(extracted_metadata.get('data_vars', {}))} data variable metadata entries"
        )

    print()
    print("Metadata Preservation Options for Polars:")
    print("1. JSON column approach (demonstrated above)")
    print("2. Separate metadata file alongside the parquet/data files")
    print("3. Custom Polars extension with metadata support")
    print("4. Using Polars DataFrame.meta for simple key-value metadata")
    print("5. Embedding metadata in column names (for simple cases)")

    # Save detailed comparison results
    comparison_results = {
        "coordinate_comparisons": coordinate_comparisons,
        "data_comparison": data_comparison,
        "xarray_metadata": metadata,
        "summary": {
            "data_integrity": data_comparison.get("values_close", False),
            "coordinates_match": all(
                comp.get("values_close", False)
                for comp in coordinate_comparisons.values()
            ),
            "metadata_preserved": len(extracted_metadata) > 0,
        },
    }

    # Save to file
    output_path = Path("zarr_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"\nDetailed comparison results saved to: {output_path}")


if __name__ == "__main__":
    main()
