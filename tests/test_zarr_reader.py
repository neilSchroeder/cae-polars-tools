"""
Unit tests for ClimateDataReader class.

Tests high-level zarr reading interface with coordinate processing
and Polars DataFrame conversion.
"""

from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from src.data_access.zarr_reader import ClimateDataReader


@pytest.mark.unit
class TestClimateDataReader:
    """Test suite for ClimateDataReader functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.store_path = "s3://test-bucket/climate-data/test.zarr"
        self.storage_options = {"anon": True}

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor")
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_init(self, mock_converter, mock_coord_proc, mock_store):
        """Test ClimateDataReader initialization."""
        reader = ClimateDataReader(
            self.store_path, storage_options=self.storage_options, chunk_size=5000
        )

        # Check that chunk_size is accessible
        assert reader.chunk_size == 5000

        # Check that components were initialized with correct parameters
        mock_store.assert_called_once_with(
            store_path=self.store_path, 
            storage_options=self.storage_options,
            group=None,
            consolidated=None
        )
        mock_coord_proc.assert_called_once()
        mock_converter.assert_called_once_with(chunk_size=5000)

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    def test_list_arrays(self, mock_store):
        """Test listing available arrays in the store."""
        # Mock store behavior
        mock_store_instance = Mock()
        mock_store_instance.list_arrays.return_value = ["temperature", "precipitation"]
        mock_store.return_value = mock_store_instance

        reader = ClimateDataReader(self.store_path)
        arrays = reader.list_arrays()

        assert arrays == ["temperature", "precipitation"]
        mock_store_instance.list_arrays.assert_called_once()

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    def test_get_array_info(self, mock_store):
        """Test getting information about a specific array."""
        # Mock store behavior
        mock_store_instance = Mock()
        mock_info = {
            "shape": (365, 180, 360),
            "dtype": "float32",
            "chunks": (30, 18, 36),
            "attrs": {"units": "K"},
        }
        mock_store_instance.get_array_info.return_value = mock_info
        mock_store.return_value = mock_store_instance

        reader = ClimateDataReader(self.store_path)
        info = reader.get_array_info("temperature")

        assert info == mock_info
        mock_store_instance.get_array_info.assert_called_once_with("temperature")

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor")
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_read_array_basic(self, mock_converter, mock_coord_proc, mock_store):
        """Test basic array reading without selection."""
        # Mock data and coordinates
        test_data = np.random.random((12, 10, 15)).astype(np.float32)
        test_coords = {
            "time": np.arange(12),
            "lat": np.linspace(30, 50, 10),
            "lon": np.linspace(-120, -100, 15),
        }

        # Mock the zarr array
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.attrs = {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}
        mock_array.__getitem__ = Mock(return_value=test_data)

        # Mock store
        mock_store_instance = Mock()
        mock_store_instance.get_array.return_value = mock_array
        mock_store_instance.open_zarr_group.return_value = Mock()
        mock_store.return_value = mock_store_instance

        # Mock coordinate processor
        mock_coord_proc_instance = Mock()
        mock_coord_proc_instance.extract_coordinate_arrays.return_value = test_coords
        mock_coord_proc_instance.process_dimension_selection.return_value = (
            [],  # selection
            ["time", "lat", "lon"],  # selected_dims
            test_coords  # selected_coord_arrays
        )
        mock_coord_proc.return_value = mock_coord_proc_instance

        # Mock converter
        mock_converter_instance = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_converter_instance.array_to_polars_lazy.return_value = mock_lf
        mock_converter.return_value = mock_converter_instance

        reader = ClimateDataReader(self.store_path)
        result = reader.read_array("temperature")

        assert result == mock_lf
        mock_converter_instance.array_to_polars_lazy.assert_called_once()

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor")
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_read_array_with_selection(
        self, mock_converter, mock_coord_proc, mock_store
    ):
        """Test array reading with dimension selection."""
        select_dims = {"time": slice(0, 6), "lat": slice(2, 8), "lon": [5, 10, 15]}

        # Mock the zarr array
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.attrs = {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}
        mock_array.__getitem__ = Mock(return_value=np.random.random((6, 6, 3)))

        # Mock components
        mock_store_instance = Mock()
        mock_store_instance.get_array.return_value = mock_array
        mock_store_instance.open_zarr_group.return_value = Mock()
        mock_store.return_value = mock_store_instance

        mock_coord_proc_instance = Mock()
        mock_coord_proc_instance.extract_coordinate_arrays.return_value = {}
        mock_coord_proc_instance.process_dimension_selection.return_value = (
            [slice(0, 6), slice(2, 8), [5, 10, 15]],  # selection
            ["time", "lat", "lon"],  # selected_dims
            {}  # selected_coord_arrays
        )
        mock_coord_proc.return_value = mock_coord_proc_instance

        mock_converter_instance = Mock()
        mock_converter_instance.array_to_polars_lazy.return_value = Mock(spec=pl.LazyFrame)
        mock_converter.return_value = mock_converter_instance

        reader = ClimateDataReader(self.store_path)
        reader.read_array("temperature", select_dims=select_dims)

        # Check that selection was passed to coordinate processor
        mock_coord_proc_instance.process_dimension_selection.assert_called_once()
        call_args = mock_coord_proc_instance.process_dimension_selection.call_args[0]
        assert call_args[2] == select_dims

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor") 
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_read_array_streaming(self, mock_converter, mock_coord_proc, mock_store):
        """Test array reading with streaming enabled."""
        # Mock the zarr array
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.attrs = {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}
        mock_array.__getitem__ = Mock(return_value=np.random.random((12, 10, 15)))

        # Mock components
        mock_store_instance = Mock()
        mock_store_instance.get_array.return_value = mock_array
        mock_store_instance.open_zarr_group.return_value = Mock()
        mock_store.return_value = mock_store_instance

        mock_coord_proc_instance = Mock()
        mock_coord_proc_instance.extract_coordinate_arrays.return_value = {}
        mock_coord_proc_instance.process_dimension_selection.return_value = (
            [],  # selection
            ["time", "lat", "lon"],  # selected_dims
            {}  # selected_coord_arrays
        )
        mock_coord_proc.return_value = mock_coord_proc_instance

        mock_converter_instance = Mock()
        mock_converter_instance.array_to_polars_lazy.return_value = Mock(spec=pl.LazyFrame)
        mock_converter.return_value = mock_converter_instance

        reader = ClimateDataReader(self.store_path)
        reader.read_array("temperature", streaming=True)

        # Check that streaming was passed to converter
        call_args = mock_converter_instance.array_to_polars_lazy.call_args
        assert call_args[0][3] is True  # streaming parameter is 4th positional arg

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    def test_read_nonexistent_array(self, mock_store):
        """Test reading non-existent array."""
        mock_store_instance = Mock()
        mock_store_instance.get_array.side_effect = KeyError("Array not found")
        mock_store.return_value = mock_store_instance

        reader = ClimateDataReader(self.store_path)

        with pytest.raises(KeyError):
            reader.read_array("nonexistent_array")

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor")
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_coordinate_extraction_failure(
        self, mock_converter, mock_coord_proc, mock_store
    ):
        """Test handling of coordinate extraction failures."""
        # Mock the zarr array
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.attrs = {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}

        # Mock store
        mock_store_instance = Mock()
        mock_store_instance.get_array.return_value = mock_array
        mock_store_instance.open_zarr_group.return_value = Mock()
        mock_store.return_value = mock_store_instance

        # Mock coordinate processor failure
        mock_coord_proc_instance = Mock()
        mock_coord_proc_instance.extract_coordinate_arrays.side_effect = Exception(
            "Coord error"
        )
        mock_coord_proc.return_value = mock_coord_proc_instance

        reader = ClimateDataReader(self.store_path)

        with pytest.raises(Exception, match="Coord error"):
            reader.read_array("temperature")

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor")
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_large_array_handling(self, mock_converter, mock_coord_proc, mock_store):
        """Test handling of large arrays with memory optimization."""
        # Mock the zarr array
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.attrs = {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}
        mock_array.__getitem__ = Mock(return_value=np.random.random((1000, 500, 500)))

        mock_store_instance = Mock()
        mock_store_instance.get_array.return_value = mock_array
        mock_store_instance.open_zarr_group.return_value = Mock()
        mock_store.return_value = mock_store_instance

        mock_coord_proc_instance = Mock()
        mock_coord_proc_instance.extract_coordinate_arrays.return_value = {}
        mock_coord_proc_instance.process_dimension_selection.return_value = (
            [],  # selection
            ["time", "lat", "lon"],  # selected_dims
            {}  # selected_coord_arrays
        )
        mock_coord_proc.return_value = mock_coord_proc_instance

        mock_converter_instance = Mock()
        mock_converter_instance.array_to_polars_lazy.return_value = Mock(spec=pl.LazyFrame)
        mock_converter.return_value = mock_converter_instance

        # Use small chunk size for large array
        reader = ClimateDataReader(self.store_path, chunk_size=1000)
        reader.read_array("large_temperature", streaming=True)

        # Should enable streaming for large arrays
        call_args = mock_converter_instance.array_to_polars_lazy.call_args
        assert call_args[0][3] is True  # streaming parameter

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor")
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_read_multiple_arrays(self, mock_converter, mock_coord_proc, mock_store):
        """Test reading multiple arrays sequentially."""
        arrays_to_read = ["temperature", "precipitation", "wind_speed"]

        # Mock the zarr array
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.attrs = {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}
        mock_array.__getitem__ = Mock(return_value=np.random.random((12, 10, 15)))

        # Mock components
        mock_store_instance = Mock()
        mock_store_instance.get_array.return_value = mock_array
        mock_store_instance.open_zarr_group.return_value = Mock()
        mock_store.return_value = mock_store_instance

        mock_coord_proc_instance = Mock()
        mock_coord_proc_instance.extract_coordinate_arrays.return_value = {}
        mock_coord_proc_instance.process_dimension_selection.return_value = (
            [],  # selection
            ["time", "lat", "lon"],  # selected_dims
            {}  # selected_coord_arrays
        )
        mock_coord_proc.return_value = mock_coord_proc_instance

        mock_converter_instance = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_converter_instance.array_to_polars_lazy.return_value = mock_lf
        mock_converter.return_value = mock_converter_instance

        reader = ClimateDataReader(self.store_path)

        results = reader.read_multiple_arrays(arrays_to_read)

        # Should have read all arrays
        assert len(results) == 3
        for result in results.values():
            assert result == mock_lf

        # Converter should have been called for each array
        assert mock_converter_instance.array_to_polars_lazy.call_count == 3

    def test_invalid_store_path(self):
        """Test initialization with invalid store paths."""
        invalid_paths = ["", "not-a-path", "http://wrong-protocol"]

        for invalid_path in invalid_paths:
            # Should not fail during initialization, failures come during usage
            reader = ClimateDataReader(invalid_path)
            # Just check that object was created
            assert hasattr(reader, 'store')
            assert hasattr(reader, 'coord_processor')
            assert hasattr(reader, 'converter')

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    def test_different_initialization_parameters(self, mock_store):
        """Test initialization with different parameter combinations."""
        test_cases = [
            # (storage_options, group, consolidated, chunk_size)
            ({"anon": True}, None, None, 10000),
            ({"key": "access", "secret": "secret"}, "climate_vars", True, 5000),
            (None, "data", False, 15000),
            ({}, None, None, 1000),
        ]

        for storage_options, group, consolidated, chunk_size in test_cases:
            reader = ClimateDataReader(
                self.store_path,
                storage_options=storage_options,
                group=group,
                consolidated=consolidated,
                chunk_size=chunk_size
            )
            
            assert reader.chunk_size == chunk_size
            mock_store.assert_called_with(
                store_path=self.store_path,
                storage_options=storage_options,
                group=group,
                consolidated=consolidated
            )

    @patch("src.data_access.zarr_reader.S3ZarrStore")
    @patch("src.data_access.zarr_reader.CoordinateProcessor")
    @patch("src.data_access.zarr_reader.PolarsConverter")
    def test_array_without_dimension_attrs(self, mock_converter, mock_coord_proc, mock_store):
        """Test reading array that doesn't have _ARRAY_DIMENSIONS attribute."""
        # Mock the zarr array without _ARRAY_DIMENSIONS
        mock_array = Mock()
        mock_array.ndim = 3
        mock_array.attrs = {}  # No _ARRAY_DIMENSIONS attribute
        mock_array.__getitem__ = Mock(return_value=np.random.random((12, 10, 15)))

        mock_store_instance = Mock()
        mock_store_instance.get_array.return_value = mock_array
        mock_store_instance.open_zarr_group.return_value = Mock()
        mock_store.return_value = mock_store_instance

        mock_coord_proc_instance = Mock()
        mock_coord_proc_instance.extract_coordinate_arrays.return_value = {}
        mock_coord_proc_instance.process_dimension_selection.return_value = (
            [],  # selection
            ["dim_0", "dim_1", "dim_2"],  # default dim names
            {}  # selected_coord_arrays
        )
        mock_coord_proc.return_value = mock_coord_proc_instance

        mock_converter_instance = Mock()
        mock_converter_instance.array_to_polars_lazy.return_value = Mock(spec=pl.LazyFrame)
        mock_converter.return_value = mock_converter_instance

        reader = ClimateDataReader(self.store_path)
        result = reader.read_array("temperature")

        # Should use default dimension names and still work
        assert isinstance(result, Mock)
        mock_coord_proc_instance.extract_coordinate_arrays.assert_called_once()
