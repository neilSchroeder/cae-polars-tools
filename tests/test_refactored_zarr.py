"""
Test script for the refactored Zarr S3 functionality.

This script tests the new modular structure and ensures compatibility
with existing functionality.
"""

import sys

sys.path.append("/home/nschroed/Documents/work/climakitae")

import polars as pl

from climakitae.new_core import (
    ClimateDataReader,
    get_climate_data_info,
    scan_climate_data,
    scan_zarr_s3,  # Legacy alias
)


def test_refactored_functionality():
    """Test the refactored Zarr S3 functionality."""

    # Test data path
    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    print("Testing refactored Zarr S3 functionality")
    print("=" * 50)

    try:
        # Test 1: Create reader and list arrays
        print("\n1. Testing ClimateDataReader...")
        reader = ClimateDataReader(store_path)
        arrays = reader.list_arrays()
        print(f"Found arrays: {arrays}")

        # Test 2: Get array info
        print("\n2. Testing array info...")
        if arrays:
            array_name = arrays[0]
            info = reader.get_array_info(array_name)
            print(f"Array '{array_name}' info:")
            print(f"  Shape: {info['shape']}")
            print(f"  Dtype: {info['dtype']}")
            print(f"  Dimensions: {info['dimensions']}")

        # Test 3: Read small subset of data
        print("\n3. Testing data reading with dimension selection...")
        if arrays:
            lf = reader.read_array(
                array_name,
                select_dims={
                    "time": slice(0, 2),
                    "lat": slice(0, 10),
                    "lon": slice(0, 10),
                },
                streaming=False,
            )
            df = lf.collect()
            print(f"Read data shape: {df.shape}")
            print(f"Columns: {df.columns}")
            print(f"First few rows:")
            print(df.head())

        # Test 4: Test high-level scanning interface
        print("\n4. Testing high-level scanning interface...")
        lf2 = scan_climate_data(
            store_path,
            array_name=array_name,
            select_dims={"time": slice(0, 1), "lat": slice(0, 5), "lon": slice(0, 5)},
            streaming=False,
        )
        df2 = lf2.collect()
        print(f"Scan result shape: {df2.shape}")

        # Test 5: Test legacy compatibility
        print("\n5. Testing legacy compatibility...")
        lf3 = scan_zarr_s3(
            store_path,
            array_name=array_name,
            select_dims={"time": slice(0, 1), "lat": slice(0, 3), "lon": slice(0, 3)},
            streaming=False,
        )
        df3 = lf3.collect()
        print(f"Legacy scan result shape: {df3.shape}")

        # Test 6: Test store info
        print("\n6. Testing store info...")
        store_info = get_climate_data_info(store_path)
        print(f"Store contains {len(store_info['arrays'])} arrays")

        print("\n✅ All tests passed! Refactoring successful.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_refactored_functionality()
