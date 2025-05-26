#!/usr/bin/env python3
"""
Comprehensive test suite for the refactored Zarr S3 functionality.
"""
import time

import polars as pl

from climakitae.new_core import (
    ClimateDataReader,
    get_climate_data_info,
    scan_climate_data,
)


def test_basic_functionality():
    """Test basic plugin functionality."""
    print("=== Testing Basic Functionality ===")

    # Test reader initialization
    reader = ClimateDataReader(
        "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"
    )
    print("✓ Reader initialized successfully")

    # Test array listing
    arrays = reader.list_arrays()
    print(f"✓ Found arrays: {arrays}")
    assert "tasmax" in arrays, "tasmax array should be present"

    # Test array info
    info = reader.get_array_info("tasmax")
    print(f"✓ Array info - shape: {info['shape']}, dtype: {info['dtype']}")
    print(f"  Dimensions: {info['dimensions']}")

    # Test reading array to Polars
    lf = reader.read_array("tasmax")
    print("✓ Array converted to LazyFrame")

    # Test schema
    schema = lf.collect_schema()
    print(f"✓ Schema: {schema}")
    expected_cols = {"time", "lat", "lon", "value"}
    assert (
        set(schema.keys()) == expected_cols
    ), f"Expected columns {expected_cols}, got {schema.keys()}"

    # Test data types
    assert schema["time"] == pl.Int64, f"Time should be Int64, got {schema['time']}"
    assert schema["lat"] == pl.Float32, f"Lat should be Float32, got {schema['lat']}"
    assert schema["lon"] == pl.Float32, f"Lon should be Float32, got {schema['lon']}"
    assert (
        schema["value"] == pl.Float32
    ), f"Value should be Float32, got {schema['value']}"

    print("✓ All data types correct")


def test_dimension_selection():
    """Test dimension selection functionality."""
    print("\n=== Testing Dimension Selection ===")

    reader = ClimateDataReader(
        "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"
    )

    # Test time slice
    lf_time = reader.read_array("tasmax", select_dims={"time": slice(0, 2)})
    sample_time = lf_time.collect()
    unique_times = sample_time["time"].unique().sort()
    print(f"✓ Time slice (0:2): {unique_times.to_list()}")
    assert len(unique_times) == 2, f"Expected 2 time steps, got {len(unique_times)}"

    # Test single time step
    lf_single = reader.read_array("tasmax", select_dims={"time": 0})
    schema_single = lf_single.collect_schema()
    print(f"✓ Single time step schema: {list(schema_single.keys())}")
    assert (
        "time" not in schema_single
    ), "Time dimension should be eliminated with single selection"

    # Test lat/lon slice
    lf_spatial = reader.read_array(
        "tasmax", select_dims={"lat": slice(0, 10), "lon": slice(0, 10), "time": 0}
    )
    sample_spatial = lf_spatial.collect()
    print(f"✓ Spatial subset: {sample_spatial.height} points")
    assert (
        sample_spatial.height == 100
    ), f"Expected 100 points (10x10), got {sample_spatial.height}"


def test_nan_handling():
    """Test NaN value handling."""
    print("\n=== Testing NaN Handling ===")

    reader = ClimateDataReader(
        "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"
    )
    lf = reader.read_array("tasmax", select_dims={"time": 0})
    df = lf.collect()

    total_points = df.height
    nan_count = df.filter(pl.col("value").is_nan()).height
    valid_count = df.filter(pl.col("value").is_not_nan()).height

    print(f"✓ Total points: {total_points:,}")
    print(f"✓ NaN points: {nan_count:,} ({nan_count/total_points*100:.1f}%)")
    print(f"✓ Valid points: {valid_count:,} ({valid_count/total_points*100:.1f}%)")

    # Check that we have both NaN and valid values
    assert nan_count > 0, "Should have some NaN values (ocean areas)"
    assert valid_count > 0, "Should have some valid values (land areas)"

    # Check valid data range (temperature in Kelvin)
    valid_stats = df.filter(pl.col("value").is_not_nan()).select(
        [
            pl.col("value").min().alias("min_temp"),
            pl.col("value").max().alias("max_temp"),
            pl.col("value").mean().alias("mean_temp"),
        ]
    )

    min_temp = valid_stats["min_temp"][0]
    max_temp = valid_stats["max_temp"][0]
    mean_temp = valid_stats["mean_temp"][0]

    print(
        f"✓ Temperature range: {min_temp:.1f}K to {max_temp:.1f}K (mean: {mean_temp:.1f}K)"
    )

    # Reasonable temperature checks for climate data
    assert 200 < min_temp < 350, f"Min temperature {min_temp}K seems unreasonable"
    assert 200 < max_temp < 350, f"Max temperature {max_temp}K seems unreasonable"
    assert 250 < mean_temp < 320, f"Mean temperature {mean_temp}K seems unreasonable"


def test_convenience_functions():
    """Test the convenience functions."""
    print("\n=== Testing Convenience Functions ===")

    # Test scan_climate_data
    lf = scan_climate_data(
        "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/",
        array_name="tasmax",
        select_dims={"time": slice(0, 1)},
    )
    assert lf is not None, "scan_climate_data should return a LazyFrame"
    print("✓ scan_climate_data function works")

    # Test get_climate_data_info
    info = get_climate_data_info(
        "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"
    )
    print(
        f"✓ get_climate_data_info function works - found {len(info['arrays'])} arrays"
    )
    print(f"  Arrays: {list(info['arrays'].keys())}")


def test_performance():
    """Test performance with different chunk sizes."""
    print("\n=== Testing Performance ===")

    # Test small chunk size (streaming)
    start_time = time.time()
    reader_small = ClimateDataReader(
        "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/",
        chunk_size=1000,
    )
    lf_small = reader_small.read_array("tasmax", select_dims={"time": slice(0, 2)})
    sample_small = lf_small.limit(1000).collect()
    small_time = time.time() - start_time
    print(
        f"✓ Small chunks (streaming): {small_time:.2f}s for {sample_small.height} rows"
    )

    # Test large chunk size (non-streaming)
    start_time = time.time()
    reader_large = ClimateDataReader(
        "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/",
        chunk_size=1000000,
    )
    lf_large = reader_large.read_array("tasmax", select_dims={"time": slice(0, 2)})
    sample_large = lf_large.limit(1000).collect()
    large_time = time.time() - start_time
    print(
        f"✓ Large chunks (non-streaming): {large_time:.2f}s for {sample_large.height} rows"
    )


def main():
    """Run all tests."""
    print("Zarr S3 Plugin - Clean Version Test Suite")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_dimension_selection()
        test_nan_handling()
        test_convenience_functions()
        test_performance()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The refactored Zarr S3 plugin is working perfectly.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
