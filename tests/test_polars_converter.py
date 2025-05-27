"""
Unit tests for PolarsConverter class.

Tests conversion of numpy arrays to Polars LazyFrames with coordinate
expansion and memory management.
"""

import numpy as np
import polars as pl
import pytest

from src.data_access.polars_converter import PolarsConverter


@pytest.mark.unit
class TestPolarsConverter:
    """Test suite for PolarsConverter functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.converter = PolarsConverter(chunk_size=1000)

    def test_init(self):
        """Test PolarsConverter initialization."""
        # Test default initialization
        converter = PolarsConverter()
        assert hasattr(converter, "chunk_size")
        assert converter.chunk_size == 10000

        # Test custom initialization
        converter = PolarsConverter(chunk_size=5000)
        assert converter.chunk_size == 5000

    def test_array_to_polars_lazy_1d(self):
        """Test conversion of 1D arrays to LazyFrame."""
        # Create 1D test data
        data_array = np.array([10.5, 20.3, 30.7, 40.1], dtype=np.float32)
        dim_names = ["time"]
        coordinates = {"time": np.array([0, 1, 2, 3])}

        # Convert to LazyFrame
        lf = self.converter.array_to_polars_lazy(data_array, dim_names, coordinates)

        # Verify it's a LazyFrame
        assert isinstance(lf, pl.LazyFrame)

        # Collect and check structure
        df = lf.collect()
        expected_columns = {"time", "value"}
        assert set(df.columns) == expected_columns
        assert len(df) == len(data_array)

        # Check values
        np.testing.assert_array_equal(df["value"].to_numpy(), data_array)
        np.testing.assert_array_equal(df["time"].to_numpy(), coordinates["time"])

    def test_array_to_polars_lazy_multidimensional(
        self, sample_zarr_data, sample_coordinates
    ):
        """Test conversion of multi-dimensional arrays."""
        temp_data = sample_zarr_data["temperature"]["data"]
        dim_names = ["time", "lat", "lon"]

        # Convert multi-dimensional array
        lf = self.converter.array_to_polars_lazy(
            temp_data, dim_names, sample_coordinates, streaming=False
        )

        df = lf.collect()

        # Should have flattened the array
        expected_length = temp_data.size
        assert len(df) == expected_length

        # Check data types
        assert df["value"].dtype == pl.Float32
        assert df["time"].dtype == pl.Int32
        assert df["lat"].dtype == pl.Float32
        assert df["lon"].dtype == pl.Float32

        # Check that all coordinate combinations are present
        assert set(df.columns) == {"time", "lat", "lon", "value"}

    def test_array_to_polars_lazy_with_none_coords(self):
        """Test conversion with missing coordinate arrays."""
        # 2D data with one missing coordinate
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        dim_names = ["time", "lat"]
        coords = {"time": np.array([10, 20]), "lat": None}

        lf = self.converter.array_to_polars_lazy(
            data, dim_names, coords, streaming=False
        )
        df = lf.collect()

        # Should create default coordinates for missing dimension
        assert len(df) == data.size
        assert set(df.columns) == {"time", "lat", "value"}

        # Check that time coordinates are properly expanded
        assert 10 in df["time"].to_list()
        assert 20 in df["time"].to_list()

    def test_streaming_conversion_large_array(self):
        """Test streaming conversion for large arrays."""
        # Create a larger array that should trigger streaming
        large_data = np.random.random((20, 15, 10)).astype(np.float32)
        dim_names = ["time", "lat", "lon"]
        coords = {
            "time": np.arange(20),
            "lat": np.linspace(0, 50, 15),
            "lon": np.linspace(0, 100, 10),
        }

        # Use small chunk size to force streaming
        converter = PolarsConverter(chunk_size=100)

        lf = converter.array_to_polars_lazy(
            large_data, dim_names, coords, streaming=True
        )

        # Should return LazyFrame for streaming
        assert isinstance(lf, pl.LazyFrame)

        # Verify data integrity by sampling
        df_sample = lf.limit(50).collect()
        assert len(df_sample) == 50
        assert set(df_sample.columns) == {"time", "lat", "lon", "value"}

    def test_coordinate_expansion_2d(self):
        """Test coordinate array expansion for 2D grid generation."""
        # Simple 2D test case
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        dim_names = ["x", "y"]
        coords = {"x": np.array([10, 20]), "y": np.array([100, 200])}

        lf = self.converter.array_to_polars_lazy(
            data, dim_names, coords, streaming=False
        )
        df = lf.collect()

        # Should have 4 rows (2x2 grid)
        assert len(df) == 4

        # Check coordinate expansion
        # Sort by coordinates for consistent testing
        df_sorted = df.sort(["x", "y"])

        # Expected pattern: (10,100)=1, (10,200)=2, (20,100)=3, (20,200)=4
        expected_values = [1.0, 2.0, 3.0, 4.0]
        assert df_sorted["value"].to_list() == expected_values

    def test_handle_nan_values(self):
        """Test handling of NaN and null values in data."""
        # Data with NaN values
        data = np.array([1.0, np.nan, 3.0, np.inf, -np.inf], dtype=np.float32)
        dim_names = ["index"]
        coords = {"index": np.arange(5)}

        lf = self.converter.array_to_polars_lazy(
            data, dim_names, coords, streaming=False
        )
        df = lf.collect()

        # Should preserve NaN values (Polars handles them)
        assert len(df) == 5
        assert df.filter(pl.col("value").is_nan()).height >= 1
        assert df.filter(pl.col("value").is_infinite()).height >= 1

    def test_memory_optimization(self):
        """Test memory optimization for different data sizes."""
        # Small data - should use direct conversion
        small_data = np.random.random((5, 5, 5)).astype(np.float32)
        dim_names = ["time", "lat", "lon"]
        small_coords = {"time": np.arange(5), "lat": np.arange(5), "lon": np.arange(5)}

        lf_small = self.converter.array_to_polars_lazy(
            small_data, dim_names, small_coords, streaming=False
        )

        # Large data - should trigger optimization
        large_data = np.random.random((50, 20, 15)).astype(np.float32)
        large_coords = {
            "time": np.arange(50),
            "lat": np.arange(20),
            "lon": np.arange(15),
        }

        converter_small_chunks = PolarsConverter(chunk_size=1000)
        lf_large = converter_small_chunks.array_to_polars_lazy(
            large_data, dim_names, large_coords, streaming=True
        )

        # Both should return LazyFrames but may use different strategies
        assert isinstance(lf_small, pl.LazyFrame)
        assert isinstance(lf_large, pl.LazyFrame)

    def test_data_type_preservation(self):
        """Test preservation of different numpy data types."""
        test_cases = [
            (np.int32, pl.Int32),
            (np.float32, pl.Float32),
            (np.float64, pl.Float64),
            (np.int64, pl.Int64),
        ]

        dim_names = ["index"]
        coords = {"index": np.arange(10)}

        for np_dtype, expected_pl_dtype in test_cases:
            data = np.arange(10, dtype=np_dtype)
            lf = self.converter.array_to_polars_lazy(data, dim_names, coords)
            df = lf.collect()

            assert df["value"].dtype == expected_pl_dtype

    def test_scalar_conversion(self):
        """Test conversion of scalar values."""
        # Single scalar value
        data = np.array(42.0)
        dim_names = []
        coords = {}

        lf = self.converter.array_to_polars_lazy(data, dim_names, coords)
        df = lf.collect()

        assert len(df) == 1
        assert df["value"][0] == 42.0
        assert set(df.columns) == {"value"}

    def test_empty_array_handling(self):
        """Test handling of empty arrays and coordinates."""
        # Empty 1D data array
        empty_data = np.array([], dtype=np.float32)
        dim_names = ["time"]
        empty_coords = {"time": np.array([])}

        lf = self.converter.array_to_polars_lazy(empty_data, dim_names, empty_coords)
        df = lf.collect()

        assert len(df) == 0
        assert set(df.columns) == {"time", "value"}

    def test_single_point_conversion(self):
        """Test conversion of single data point."""
        # Single element 1D array
        data = np.array([42.0], dtype=np.float32)
        dim_names = ["time"]
        coords = {"time": np.array([0])}

        lf = self.converter.array_to_polars_lazy(data, dim_names, coords)
        df = lf.collect()

        assert len(df) == 1
        assert df["value"][0] == 42.0
        assert df["time"][0] == 0

    def test_coordinate_dimension_mismatch(self):
        """Test handling of coordinate-data dimension mismatches."""
        # 2D data but only 1 coordinate provided
        data = np.random.random((3, 4)).astype(np.float32)
        dim_names = ["time", "lat"]
        coords = {"time": np.arange(3)}  # Missing lat coordinate

        # Should handle gracefully by creating default coordinates
        lf = self.converter.array_to_polars_lazy(
            data, dim_names, coords, streaming=False
        )
        df = lf.collect()

        # Should still work with default coordinates for missing dimension
        assert len(df) == data.size
        assert set(df.columns) == {"time", "lat", "value"}

    def test_chunk_size_effects(self):
        """Test effects of different chunk sizes on processing."""
        data = np.random.random((10, 8)).astype(np.float32)
        dim_names = ["x", "y"]
        coords = {"x": np.arange(10), "y": np.arange(8)}

        # Test different chunk sizes
        for chunk_size in [10, 50, 200]:
            converter = PolarsConverter(chunk_size=chunk_size)
            lf = converter.array_to_polars_lazy(data, dim_names, coords, streaming=True)
            df = lf.collect()

            # Results should be consistent regardless of chunk size
            assert len(df) == data.size
            assert set(df.columns) == {"x", "y", "value"}

    def test_streaming_vs_non_streaming(self):
        """Test that streaming and non-streaming give same results."""
        data = np.random.random((15, 10)).astype(np.float32)
        dim_names = ["time", "lat"]
        coords = {"time": np.arange(15), "lat": np.arange(10)}

        # Non-streaming
        lf_direct = self.converter.array_to_polars_lazy(
            data, dim_names, coords, streaming=False
        )
        df_direct = lf_direct.collect().sort(["time", "lat"])

        # Streaming
        lf_streaming = self.converter.array_to_polars_lazy(
            data, dim_names, coords, streaming=True
        )
        df_streaming = lf_streaming.collect().sort(["time", "lat"])

        # Results should be identical
        assert len(df_direct) == len(df_streaming)
        np.testing.assert_array_equal(
            df_direct["value"].to_numpy(), df_streaming["value"].to_numpy()
        )

    def test_coordinate_processor_integration(self):
        """Test integration with CoordinateProcessor."""
        # Test that the converter properly uses the coordinate processor
        data = np.random.random((3, 4, 2)).astype(np.float32)
        dim_names = ["time", "lat", "lon"]
        coords = {
            "time": np.array([0, 1, 2]),
            "lat": np.array([10, 20, 30, 40]),
            "lon": np.array([100, 200]),
        }

        lf = self.converter.array_to_polars_lazy(
            data, dim_names, coords, streaming=False
        )
        df = lf.collect()

        # Verify coordinate expansion worked correctly
        assert len(df) == data.size

        # Check that all coordinate combinations exist
        unique_times = sorted(df["time"].unique().to_list())
        unique_lats = sorted(df["lat"].unique().to_list())
        unique_lons = sorted(df["lon"].unique().to_list())

        assert unique_times == [0, 1, 2]
        assert unique_lats == [10, 20, 30, 40]
        assert unique_lons == [100, 200]
