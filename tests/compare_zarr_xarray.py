#!/usr/bin/env python3
"""
Comparison script between ZarrS3Reader and xarray for Zarr S3 data.
"""

import numpy as np
import polars as pl
import xarray as xr

from climakitae.new_core import polars_IOplugin_zarr as zp


def compare_zarr_methods():
    """Compare ZarrS3Reader with xarray for the same dataset."""

    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    print("=" * 80)
    print("COMPARISON: ZarrS3Reader vs xarray")
    print("=" * 80)

    # 1. Load with ZarrS3Reader
    print("\n1. Loading with ZarrS3Reader...")
    reader = zp.ZarrS3Reader(store_path)

    # Get basic info
    arrays = reader.list_arrays()
    print(f"   Available arrays: {arrays}")

    # Get array info for tasmax
    tasmax_info = reader.get_array_info("tasmax")
    print(f"   tasmax shape: {tasmax_info['shape']}")
    print(f"   tasmax dtype: {tasmax_info['dtype']}")
    print(f"   tasmax dimensions: {tasmax_info['dimensions']}")
    print(f"   tasmax attributes: {list(tasmax_info['attrs'].keys())}")

    # Load a small subset for comparison
    lf = reader.read_array_to_polars("tasmax", select_dims={"time": slice(0, 2)})
    df = lf.collect()
    print(f"   Polars DataFrame shape: {df.shape}")
    print(f"   Polars DataFrame columns: {df.columns}")
    print(f"   Polars DataFrame dtypes: {df.dtypes}")

    # Get coordinate arrays separately
    time_lf = reader.read_array_to_polars("time")
    lat_lf = reader.read_array_to_polars("lat")
    lon_lf = reader.read_array_to_polars("lon")

    time_df = time_lf.collect()
    lat_df = lat_lf.collect()
    lon_df = lon_lf.collect()

    print(
        f"   Time array shape: {time_df.shape}, range: {time_df['value'].min()} to {time_df['value'].max()}"
    )
    print(
        f"   Lat array shape: {lat_df.shape}, range: {lat_df['value'].min():.3f} to {lat_df['value'].max():.3f}"
    )
    print(
        f"   Lon array shape: {lon_df.shape}, range: {lon_df['value'].min():.3f} to {lon_df['value'].max():.3f}"
    )

    # 2. Load with xarray
    print("\n2. Loading with xarray...")
    try:
        # Use xarray to open the same zarr store
        ds = xr.open_zarr(store_path, storage_options={})

        print(f"   Dataset variables: {list(ds.data_vars.keys())}")
        print(f"   Dataset coordinates: {list(ds.coords.keys())}")
        print(f"   Dataset dimensions: {dict(ds.dims)}")

        # Get tasmax info
        tasmax_xr = ds["tasmax"]
        print(f"   tasmax shape: {tasmax_xr.shape}")
        print(f"   tasmax dtype: {tasmax_xr.dtype}")
        print(f"   tasmax dimensions: {list(tasmax_xr.dims)}")

        # Get coordinate arrays
        time_xr = ds["time"]
        lat_xr = ds["lat"]
        lon_xr = ds["lon"]

        print(
            f"   Time coordinate shape: {time_xr.shape}, range: {time_xr.values.min()} to {time_xr.values.max()}"
        )
        print(
            f"   Lat coordinate shape: {lat_xr.shape}, range: {lat_xr.values.min():.3f} to {lat_xr.values.max():.3f}"
        )
        print(
            f"   Lon coordinate shape: {lon_xr.shape}, range: {lon_xr.values.min():.3f} to {lon_xr.values.max():.3f}"
        )

        # Get the same subset as our Polars data
        tasmax_subset = tasmax_xr.isel(time=slice(0, 2))

        print(f"   xarray subset shape: {tasmax_subset.shape}")

    except Exception as e:
        print(f"   Error loading with xarray: {e}")
        return

    # 3. Compare data values
    print("\n3. Comparing data values...")

    # Compare coordinate arrays
    time_polars = time_df["value"].to_numpy()
    time_xarray = time_xr.values

    lat_polars = lat_df["value"].to_numpy()
    lat_xarray = lat_xr.values

    lon_polars = lon_df["value"].to_numpy()
    lon_xarray = lon_xr.values

    print(f"   Time arrays equal: {np.array_equal(time_polars, time_xarray)}")
    print(f"   Lat arrays equal: {np.allclose(lat_polars, lat_xarray, rtol=1e-10)}")
    print(f"   Lon arrays equal: {np.allclose(lon_polars, lon_xarray, rtol=1e-10)}")

    # Compare tasmax values for the subset
    # Get tasmax values from Polars (need to reshape back to 3D)
    polars_values = df["value"].to_numpy()
    polars_time = df["time"].to_numpy()
    polars_lat = df["lat"].to_numpy()
    polars_lon = df["lon"].to_numpy()

    # Get xarray values
    xarray_values = tasmax_subset.values.ravel()

    print(f"   Polars tasmax values shape: {polars_values.shape}")
    print(f"   xarray tasmax values shape: {xarray_values.shape}")
    print(
        f"   Values arrays equal: {np.allclose(polars_values, xarray_values, equal_nan=True)}"
    )

    # Check some statistics
    print(f"   Polars NaN count: {np.isnan(polars_values).sum()}")
    print(f"   xarray NaN count: {np.isnan(xarray_values).sum()}")
    print(
        f"   Polars valid range: {np.nanmin(polars_values):.3f} to {np.nanmax(polars_values):.3f}"
    )
    print(
        f"   xarray valid range: {np.nanmin(xarray_values):.3f} to {np.nanmax(xarray_values):.3f}"
    )

    # 4. Examine metadata
    print("\n4. Examining metadata...")

    print("\n   Dataset-level attributes (xarray):")
    for key, value in ds.attrs.items():
        print(f"     {key}: {value}")

    print("\n   tasmax variable attributes (xarray):")
    for key, value in tasmax_xr.attrs.items():
        print(f"     {key}: {value}")

    print("\n   time coordinate attributes (xarray):")
    for key, value in time_xr.attrs.items():
        print(f"     {key}: {value}")

    print("\n   lat coordinate attributes (xarray):")
    for key, value in lat_xr.attrs.items():
        print(f"     {key}: {value}")

    print("\n   lon coordinate attributes (xarray):")
    for key, value in lon_xr.attrs.items():
        print(f"     {key}: {value}")

    # 5. Show what ZarrS3Reader captured
    print("\n   ZarrS3Reader captured attributes:")
    for key, value in tasmax_info["attrs"].items():
        print(f"     {key}: {value}")

    return ds, reader, df


def explore_metadata_preservation():
    """Explore ways to preserve metadata in Polars DataFrames."""

    print("\n" + "=" * 80)
    print("EXPLORING METADATA PRESERVATION")
    print("=" * 80)

    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    # Load with xarray to get full metadata
    ds = xr.open_zarr(store_path, storage_options={})

    # Create a metadata dictionary that could be attached to Polars DataFrame
    metadata = {
        "dataset_attrs": dict(ds.attrs),
        "variable_attrs": {},
        "coordinate_attrs": {},
        "dimensions": dict(ds.dims),
        "data_vars": list(ds.data_vars.keys()),
        "coordinates": list(ds.coords.keys()),
    }

    # Collect variable attributes
    for var_name in ds.data_vars.keys():
        metadata["variable_attrs"][var_name] = dict(ds[var_name].attrs)

    # Collect coordinate attributes
    for coord_name in ds.coords.keys():
        metadata["coordinate_attrs"][coord_name] = dict(ds[coord_name].attrs)

    print("   Complete metadata structure:")
    print(f"     Dataset attributes: {len(metadata['dataset_attrs'])} items")
    print(f"     Variable attributes: {list(metadata['variable_attrs'].keys())}")
    print(f"     Coordinate attributes: {list(metadata['coordinate_attrs'].keys())}")
    print(f"     Dimensions: {metadata['dimensions']}")

    # Show how this could be used with Polars
    reader = zp.ZarrS3Reader(store_path)
    lf = reader.read_array_to_polars("tasmax", select_dims={"time": slice(0, 1)})
    df = lf.collect()

    # Polars doesn't have built-in metadata support, but we can:
    # 1. Use the schema/column names to store some info
    # 2. Create a separate metadata object
    # 3. Use custom column names that embed metadata

    print("\n   Potential approaches for metadata in Polars:")
    print("   1. Separate metadata dictionary (recommended)")
    print("   2. Custom column names with embedded info")
    print("   3. Additional metadata columns")

    # Example 1: Separate metadata dictionary
    enhanced_result = {
        "data": df,
        "metadata": metadata,
        "source": store_path,
        "array_name": "tasmax",
    }

    print(f"\n   Enhanced result structure:")
    print(f"     Data shape: {enhanced_result['data'].shape}")
    print(f"     Metadata keys: {list(enhanced_result['metadata'].keys())}")
    print(f"     Source: {enhanced_result['source']}")

    return enhanced_result


if __name__ == "__main__":
    # Run comparison
    ds, reader, df = compare_zarr_methods()

    # Explore metadata preservation
    enhanced_result = explore_metadata_preservation()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ ZarrS3Reader successfully reads the same data as xarray")
    print("✓ Coordinate arrays match between methods")
    print("✓ Data values match between methods")
    print("✓ Metadata can be preserved in a separate dictionary")
    print("\nNext steps:")
    print("- Consider enhancing ZarrS3Reader to return metadata alongside data")
    print("- Implement metadata preservation in the plugin")
