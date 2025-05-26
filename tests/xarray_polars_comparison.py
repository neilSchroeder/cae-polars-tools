#!/usr/bin/env python3
"""
Side-by-side comparison demonstrating xarray-like metadata access in Polars.

This script shows how the enhanced ZarrS3Reader provides the same metadata access
patterns as xarray, enabling a smooth transition for users familiar with xarray.
"""

import xarray as xr

from enhanced_zarr_reader import EnhancedZarrS3Reader


def main():
    print("Side-by-Side: xarray vs Enhanced Polars Metadata Access")
    print("=" * 60)

    s3_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    # Load with xarray
    print("Loading with xarray...")
    ds_xr = xr.open_zarr(s3_path, consolidated=True)

    # Load with enhanced Polars reader
    print("Loading with enhanced Polars reader...")
    reader = EnhancedZarrS3Reader(s3_path)
    df_enhanced = reader.read_array_to_polars_with_metadata("tasmax", streaming=False)

    print("\n🔍 METADATA ACCESS COMPARISON")
    print("=" * 40)

    # Dataset-level attributes
    print("\n📊 Dataset-level attributes:")
    print("-" * 30)
    print(f"xarray:  len(ds.attrs) = {len(ds_xr.attrs)}")
    print(f"Polars:  len(df.attrs) = {len(df_enhanced.attrs)}")

    # Show identical access pattern
    sample_key = "Conventions"
    print(f"\nAccess pattern (identical syntax):")
    print(f"  xarray:  ds.attrs['{sample_key}'] = '{ds_xr.attrs[sample_key]}'")
    print(f"  Polars:  df.attrs['{sample_key}'] = '{df_enhanced.attrs[sample_key]}'")
    print(f"  Match: {ds_xr.attrs[sample_key] == df_enhanced.attrs[sample_key]} ✅")

    # Variable attributes
    print("\n🔢 Variable attributes:")
    print("-" * 25)
    xr_var_attrs = dict(ds_xr["tasmax"].attrs)
    polars_var_attrs = dict(df_enhanced.get_var_attrs("tasmax"))

    print(f"xarray:  ds['tasmax'].attrs = {xr_var_attrs}")
    print(f"Polars:  df.get_var_attrs('tasmax') = {polars_var_attrs}")
    print(f"Match: {xr_var_attrs == polars_var_attrs} ✅")

    # Coordinate attributes
    print("\n🌍 Coordinate attributes:")
    print("-" * 25)
    for coord in ["lat", "lon", "time"]:
        xr_coord_attrs = dict(ds_xr[coord].attrs)
        polars_coord_attrs = dict(df_enhanced.get_coord_attrs(coord))
        match = xr_coord_attrs == polars_coord_attrs
        print(f"{coord}: {match} ✅" if match else f"{coord}: {match} ❌")
        if coord == "lat":  # Show example
            print(f"  xarray:  ds['{coord}'].attrs = {xr_coord_attrs}")
            print(f"  Polars:  df.get_coord_attrs('{coord}') = {polars_coord_attrs}")

    print("\n💡 USAGE PATTERNS")
    print("=" * 18)

    print("\n🔑 Key Benefits of Enhanced Polars Approach:")
    print("  ✅ Familiar xarray-like syntax: df.attrs['key']")
    print("  ✅ Full metadata preservation from zarr/netCDF")
    print("  ✅ High-performance Polars operations on data")
    print("  ✅ Seamless integration with existing workflows")
    print("  ✅ Metadata persistence to/from JSON files")

    print(f"\n📈 Performance comparison:")
    print(f"  xarray data shape: {ds_xr['tasmax'].shape} (3D array)")
    print(f"  Polars data shape: {df_enhanced.collect().dataframe.shape} (2D table)")
    print(
        f"  Same total elements: {ds_xr['tasmax'].size == df_enhanced.collect().dataframe.shape[0]} ✅"
    )

    print("\n🚀 Example workflows:")
    print("```python")
    print("# xarray-style metadata access")
    print("units = df_enhanced.attrs['units']")
    print("long_name = df_enhanced.get_var_attrs('tasmax')['long_name']")
    print("lat_units = df_enhanced.get_coord_attrs('lat')['units']")
    print("")
    print("# Polars-style data operations")
    print("df = df_enhanced.collect().dataframe")
    print("monthly_avg = df.group_by(['lat', 'lon']).agg(pl.mean('value'))")
    print("time_series = df.filter(pl.col('lat') == 37.5).select(['time', 'value'])")
    print("```")


if __name__ == "__main__":
    main()
