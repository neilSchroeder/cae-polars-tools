"""
Unit tests for CoordinateProcessor class.

Tests coordinate array processing, dimension selection, and optimization
for multi-dimensional climate data.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.data_access.coordinate_processor import CoordinateProcessor


@pytest.mark.unit
class TestCoordinateProcessor:
    """Test suite for CoordinateProcessor functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.processor = CoordinateProcessor(cache_size_threshold=1000)

    def test_init(self):
        """Test CoordinateProcessor initialization."""
        # Test default initialization
        processor = CoordinateProcessor()
        assert hasattr(processor, "cache_size_threshold")
        assert processor.cache_size_threshold == 10000

        # Test custom initialization
        processor = CoordinateProcessor(cache_size_threshold=5000)
        assert processor.cache_size_threshold == 5000

    def test_extract_coordinate_arrays_basic(self, sample_coordinates):
        """Test coordinate array extraction for basic arrays."""
        # Mock zarr group
        mock_group = Mock()

        # Test successful extraction
        coord_name = "time"
        coord_data = sample_coordinates[coord_name]

        mock_array = Mock()
        mock_array.size = coord_data.size
        mock_array.__getitem__ = Mock(return_value=coord_data)

        mock_group.__contains__ = Mock(return_value=True)
        mock_group.__getitem__ = Mock(return_value=mock_array)

        coords = self.processor.extract_coordinate_arrays(mock_group, [coord_name])

        assert coord_name in coords
        assert coords[coord_name] is not None
        np.testing.assert_array_equal(coords[coord_name], coord_data)

    def test_extract_coordinate_arrays_missing(self):
        """Test coordinate extraction with missing coordinates."""
        mock_group = Mock()
        mock_group.__contains__ = Mock(return_value=False)

        coords = self.processor.extract_coordinate_arrays(mock_group, ["missing_coord"])

        assert "missing_coord" in coords
        assert coords["missing_coord"] is None

    def test_extract_coordinate_arrays_large(self):
        """Test coordinate extraction with large arrays."""
        mock_group = Mock()

        # Create large coordinate array
        large_coord = np.arange(2000)  # Larger than cache threshold

        mock_array = Mock()
        mock_array.size = large_coord.size
        mock_array.__getitem__ = Mock(return_value=large_coord)

        mock_group.__contains__ = Mock(return_value=True)
        mock_group.__getitem__ = Mock(return_value=mock_array)

        coords = self.processor.extract_coordinate_arrays(mock_group, ["large_coord"])

        assert "large_coord" in coords
        assert coords["large_coord"] is not None
        np.testing.assert_array_equal(coords["large_coord"], large_coord)

    def test_extract_coordinate_arrays_exception(self):
        """Test coordinate extraction with read errors."""
        mock_group = Mock()
        mock_group.__contains__ = Mock(return_value=True)
        mock_group.__getitem__ = Mock(side_effect=Exception("Read error"))

        coords = self.processor.extract_coordinate_arrays(mock_group, ["error_coord"])

        assert "error_coord" in coords
        assert coords["error_coord"] is None

    def test_process_dimension_selection_no_selection(self, sample_coordinates):
        """Test dimension selection with no selection criteria."""
        dims = list(sample_coordinates.keys())
        coord_arrays = sample_coordinates.copy()

        selection, selected_dims, selected_coords = (
            self.processor.process_dimension_selection(dims, coord_arrays, None)
        )

        # Should return slice(None) for all dimensions
        assert len(selection) == len(dims)
        assert all(sel == slice(None) for sel in selection)
        assert selected_dims == dims
        assert selected_coords == coord_arrays

    def test_process_dimension_selection_with_slice(self, sample_coordinates):
        """Test dimension selection with slice objects."""
        dims = ["time", "lat", "lon"]
        coord_arrays = sample_coordinates.copy()
        select_dims = {"time": slice(0, 6), "lat": slice(2, 8)}

        selection, selected_dims, selected_coords = (
            self.processor.process_dimension_selection(dims, coord_arrays, select_dims)
        )

        # Check selection objects
        assert selection[0] == slice(0, 6)  # time
        assert selection[1] == slice(2, 8)  # lat
        assert selection[2] == slice(None)  # lon (unselected)

        # Check selected coordinates
        assert len(selected_coords["time"]) == 6
        assert len(selected_coords["lat"]) == 6
        np.testing.assert_array_equal(
            selected_coords["time"], coord_arrays["time"][0:6]
        )
        np.testing.assert_array_equal(selected_coords["lat"], coord_arrays["lat"][2:8])

    def test_process_dimension_selection_with_int(self, sample_coordinates):
        """Test dimension selection with integer selection."""
        dims = ["time", "lat", "lon"]
        coord_arrays = sample_coordinates.copy()
        select_dims = {"time": 5}

        selection, selected_dims, selected_coords = (
            self.processor.process_dimension_selection(dims, coord_arrays, select_dims)
        )

        # Integer selection should reduce dimensionality
        assert selection[0] == 5
        assert len(selected_dims) == 2  # time dimension removed
        assert "time" not in selected_coords or selected_coords["time"] is None

    def test_process_dimension_selection_with_list(self, sample_coordinates):
        """Test dimension selection with list of integers."""
        dims = ["time", "lat", "lon"]
        coord_arrays = sample_coordinates.copy()
        select_dims = {"lat": [1, 3, 5]}

        selection, selected_dims, selected_coords = (
            self.processor.process_dimension_selection(dims, coord_arrays, select_dims)
        )

        # Check list selection
        assert selection[1] == [1, 3, 5]
        assert len(selected_coords["lat"]) == 3
        np.testing.assert_array_equal(
            selected_coords["lat"], coord_arrays["lat"][[1, 3, 5]]
        )

    def test_process_dimension_selection_with_none_coords(self):
        """Test dimension selection with None coordinate arrays."""
        dims = ["time", "lat"]
        coord_arrays = {"time": None, "lat": None}
        select_dims = {"time": slice(0, 5)}

        selection, selected_dims, selected_coords = (
            self.processor.process_dimension_selection(dims, coord_arrays, select_dims)
        )

        assert selected_coords["time"] is None
        assert selected_coords["lat"] is None

    def test_create_coordinate_expansions_basic(self, sample_coordinates):
        """Test coordinate expansion without full meshgrids."""
        data_shape = (12, 10, 15)  # time, lat, lon (match sample_coordinates fixture)
        dim_names = ["time", "lat", "lon"]
        coord_arrays = sample_coordinates.copy()

        flat_coords = self.processor.create_coordinate_expansions(
            data_shape, dim_names, coord_arrays
        )

        # Check that all dimensions are present
        assert set(flat_coords.keys()) == set(dim_names)

        # Check that flattened coordinates have correct total size
        total_size = np.prod(data_shape)
        for coord_name in dim_names:
            assert len(flat_coords[coord_name]) == total_size

        # Verify coordinate patterns
        # Time coordinates should repeat for each lat/lon combination
        expected_time_pattern = np.tile(
            np.repeat(coord_arrays["time"], data_shape[1] * data_shape[2]), 1
        )
        np.testing.assert_array_equal(flat_coords["time"], expected_time_pattern)

    def test_create_coordinate_expansions_with_none(self):
        """Test coordinate expansion with missing coordinate arrays."""
        data_shape = (5, 3)
        dim_names = ["time", "lat"]
        coord_arrays = {"time": None, "lat": np.array([10, 20, 30])}

        flat_coords = self.processor.create_coordinate_expansions(
            data_shape, dim_names, coord_arrays
        )

        # None coordinate should be replaced with arange
        expected_time = np.tile(np.repeat(np.arange(5), 3), 1)
        np.testing.assert_array_equal(flat_coords["time"], expected_time)

        # Lat should be properly expanded
        expected_lat = np.tile(coord_arrays["lat"], 5)
        np.testing.assert_array_equal(flat_coords["lat"], expected_lat)

    def test_create_coordinate_expansions_1d(self):
        """Test coordinate expansion for 1D data."""
        data_shape = (10,)
        dim_names = ["time"]
        coord_arrays = {"time": np.arange(10)}

        flat_coords = self.processor.create_coordinate_expansions(
            data_shape, dim_names, coord_arrays
        )

        # Should be identical to original for 1D
        np.testing.assert_array_equal(flat_coords["time"], coord_arrays["time"])

    def test_create_streaming_coordinate_chunks(self, sample_coordinates):
        """Test streaming coordinate chunk creation."""
        data_shape = (4, 3, 2)  # Small shape for testing
        dim_names = ["time", "lat", "lon"]
        coord_arrays = {
            "time": np.array([0, 1, 2, 3]),
            "lat": np.array([10, 20, 30]),
            "lon": np.array([100, 200]),
        }

        start_idx = 0
        end_idx = 6  # First 6 elements

        chunk_coords = self.processor.create_streaming_coordinate_chunks(
            data_shape, dim_names, coord_arrays, start_idx, end_idx
        )

        # Check that all dimensions are present
        assert set(chunk_coords.keys()) == set(dim_names)

        # Check chunk sizes
        chunk_size = end_idx - start_idx
        for coord_name in dim_names:
            assert len(chunk_coords[coord_name]) == chunk_size

    def test_create_streaming_coordinate_chunks_middle(self):
        """Test streaming coordinate chunks for middle section."""
        data_shape = (3, 2)
        dim_names = ["time", "lat"]
        coord_arrays = {"time": np.array([0, 1, 2]), "lat": np.array([10, 20])}

        start_idx = 2
        end_idx = 5

        chunk_coords = self.processor.create_streaming_coordinate_chunks(
            data_shape, dim_names, coord_arrays, start_idx, end_idx
        )

        # Verify chunk has correct size
        chunk_size = end_idx - start_idx
        assert len(chunk_coords["time"]) == chunk_size
        assert len(chunk_coords["lat"]) == chunk_size

    def test_cache_size_threshold(self):
        """Test that cache size threshold affects coordinate handling."""
        # Small threshold processor
        small_processor = CoordinateProcessor(cache_size_threshold=50)

        # Large threshold processor
        large_processor = CoordinateProcessor(cache_size_threshold=5000)

        # Test with medium-sized array
        coord_data = np.arange(100)
        mock_array = Mock()
        mock_array.size = coord_data.size
        mock_array.__getitem__ = Mock(return_value=coord_data)

        mock_group = Mock()
        mock_group.__contains__ = Mock(return_value=True)
        mock_group.__getitem__ = Mock(return_value=mock_array)

        # Both should handle the array, but processing might differ
        small_coords = small_processor.extract_coordinate_arrays(mock_group, ["coord"])
        large_coords = large_processor.extract_coordinate_arrays(mock_group, ["coord"])

        # Results should be the same regardless of threshold
        np.testing.assert_array_equal(small_coords["coord"], large_coords["coord"])

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty dimensions list
        coords = self.processor.extract_coordinate_arrays(Mock(), [])
        assert coords == {}

        # Empty coordinate selection
        selection, dims, coord_arrays = self.processor.process_dimension_selection(
            [], {}, None
        )
        assert selection == []
        assert dims == []
        assert coord_arrays == {}

        # Empty data shape
        flat_coords = self.processor.create_coordinate_expansions((), [], {})
        assert flat_coords == {}

    def test_memory_efficiency(self):
        """Test memory-efficient coordinate processing."""
        # Test with larger data shapes to verify no excessive memory usage
        large_shape = (100, 50, 20)
        dim_names = ["time", "lat", "lon"]
        coord_arrays = {
            "time": np.arange(100),
            "lat": np.linspace(0, 90, 50),
            "lon": np.linspace(-180, 180, 20),
        }

        # This should not create excessive memory usage
        flat_coords = self.processor.create_coordinate_expansions(
            large_shape, dim_names, coord_arrays
        )

        # Verify correct total size
        total_size = np.prod(large_shape)
        for coord_name in dim_names:
            assert len(flat_coords[coord_name]) == total_size

        # Test streaming chunks for memory efficiency
        chunk_coords = self.processor.create_streaming_coordinate_chunks(
            large_shape, dim_names, coord_arrays, 0, 1000
        )

        for coord_name in dim_names:
            assert len(chunk_coords[coord_name]) == 1000
