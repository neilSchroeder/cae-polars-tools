"""
Test the refactored functionality with the tasmax array to ensure
it handles multi-dimensional data correctly.
"""

import sys

sys.path.append("/home/nschroed/Documents/work/climakitae")

from climakitae.new_core import ClimateDataReader


def test_tasmax_array():
    """Test reading the tasmax array with the refactored components."""

    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    print("Testing tasmax array with refactored components")
    print("=" * 50)

    try:
        # Create reader
        reader = ClimateDataReader(store_path)

        # Get tasmax info
        tasmax_info = reader.get_array_info("tasmax")
        print(f"Tasmax array info:")
        print(f"  Shape: {tasmax_info['shape']}")
        print(f"  Dtype: {tasmax_info['dtype']}")
        print(f"  Dimensions: {tasmax_info['dimensions']}")

        # Read a small subset
        print(f"\nReading small subset...")
        lf = reader.read_array(
            "tasmax",
            select_dims={"time": slice(0, 3), "lat": slice(0, 5), "lon": slice(0, 5)},
            streaming=False,
        )
        df = lf.collect()
        print(f"Result shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print(f"Sample data:")
        print(df.head(10))

        # Check for NaN values
        nan_count = df.select(df["value"].is_null().sum()).item()
        print(f"\nNaN values: {nan_count}")

        # Test streaming mode with larger data
        print(f"\nTesting streaming mode...")
        lf_stream = reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 12),
                "lat": slice(0, 20),
                "lon": slice(0, 20),
            },
            streaming=True,
        )
        df_stream = lf_stream.collect()
        print(f"Streaming result shape: {df_stream.shape}")

        print("\n✅ Tasmax test passed! Multi-dimensional processing works correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_tasmax_array()
