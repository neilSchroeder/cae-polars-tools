"""
Unit tests for zarr_scanner module.

Tests high-level scanning interface for Zarr climate data.
"""

from unittest.mock import Mock, patch

import polars as pl
import pytest

from src.data_access.zarr_scanner import (
    get_climate_data_info,
    scan_climate_data,
    zarr_s3_info,
)


@pytest.mark.unit
class TestZarrScanner:
    """Test suite for zarr_scanner functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.store_path = "s3://test-bucket/climate-data/test.zarr"
        self.storage_options = {"anon": True}

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_scan_climate_data_basic(self, mock_reader_class):
        """Test basic climate data scanning."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        result = scan_climate_data(
            self.store_path,
            array_name="temperature",
            storage_options=self.storage_options,
        )

        assert result == mock_lf
        mock_reader_class.assert_called_once_with(
            store_path=self.store_path,
            storage_options=self.storage_options,
            group=None,
            consolidated=None,
            chunk_size=10000,
        )
        mock_reader.read_array.assert_called_once_with(
            array_name="temperature", select_dims=None, streaming=True
        )

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_scan_climate_data_with_selection(self, mock_reader_class):
        """Test climate data scanning with dimension selection."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        select_dims = {
            "time": slice(0, 12),
            "lat": slice(100, 200),
            "lon": slice(300, 400),
        }

        result = scan_climate_data(
            self.store_path,
            array_name="temperature",
            storage_options=self.storage_options,
            select_dims=select_dims,
            chunk_size=5000,
            streaming=False,
        )

        assert result == mock_lf
        mock_reader.read_array.assert_called_once_with(
            array_name="temperature", select_dims=select_dims, streaming=False
        )

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_scan_climate_data_all_arrays(self, mock_reader_class):
        """Test scanning all arrays when array_name is None."""
        mock_reader = Mock()
        mock_reader.list_arrays.return_value = ["temperature", "precipitation"]

        mock_result_dict = {
            "temperature": Mock(spec=pl.LazyFrame),
            "precipitation": Mock(spec=pl.LazyFrame),
        }
        mock_reader.read_multiple_arrays.return_value = mock_result_dict
        mock_reader_class.return_value = mock_reader

        result = scan_climate_data(
            self.store_path, array_name=None, storage_options=self.storage_options
        )

        # Should return dictionary of LazyFrames
        assert result == mock_result_dict
        mock_reader.read_multiple_arrays.assert_called_once_with(
            ["temperature", "precipitation"], streaming=True
        )

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_scan_climate_data_with_group(self, mock_reader_class):
        """Test scanning with zarr group specification."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        scan_climate_data(
            self.store_path,
            array_name="temperature",
            group="climate_vars",
            consolidated=True,
        )

        mock_reader_class.assert_called_once_with(
            store_path=self.store_path,
            storage_options=None,
            group="climate_vars",
            consolidated=True,
            chunk_size=10000,
        )

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_get_climate_data_info(self, mock_reader_class):
        """Test getting climate data information."""
        mock_reader = Mock()
        mock_reader.list_arrays.return_value = ["temperature", "precipitation"]
        mock_reader.get_array_info.side_effect = [
            {
                "shape": (365, 180, 360),
                "dtype": "float32",
                "chunks": (30, 18, 36),
                "attrs": {"units": "K"},
                "dimensions": ["time", "lat", "lon"],
            },
            {
                "shape": (365, 180, 360),
                "dtype": "float32",
                "chunks": (30, 18, 36),
                "attrs": {"units": "mm/day"},
                "dimensions": ["time", "lat", "lon"],
            },
        ]
        mock_reader_class.return_value = mock_reader

        result = get_climate_data_info(
            self.store_path, storage_options=self.storage_options
        )

        assert "store_path" in result
        assert "arrays" in result
        assert result["store_path"] == self.store_path
        assert "temperature" in result["arrays"]
        assert "precipitation" in result["arrays"]

        # Check array info structure
        temp_info = result["arrays"]["temperature"]
        assert temp_info["shape"] == (365, 180, 360)
        assert temp_info["dtype"] == "float32"
        assert temp_info["attrs"]["units"] == "K"

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_zarr_s3_info_legacy_alias(self, mock_reader_class):
        """Test legacy alias zarr_s3_info."""
        mock_reader = Mock()
        mock_reader.list_arrays.return_value = ["temperature"]
        mock_reader.get_array_info.return_value = {
            "shape": (365, 180, 360),
            "dtype": "float32",
            "chunks": (30, 18, 36),
            "attrs": {"units": "K"},
            "dimensions": ["time", "lat", "lon"],
        }
        mock_reader_class.return_value = mock_reader

        # Should work the same as get_climate_data_info
        result = zarr_s3_info(self.store_path)

        assert "store_path" in result
        assert "arrays" in result

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_scan_climate_data_error_handling(self, mock_reader_class):
        """Test error handling in scan_climate_data."""
        # Test reader initialization error
        mock_reader_class.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            scan_climate_data(self.store_path, array_name="temperature")

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_scan_climate_data_nonexistent_array(self, mock_reader_class):
        """Test scanning non-existent array."""
        mock_reader = Mock()
        mock_reader.read_array.side_effect = KeyError("Array not found")
        mock_reader_class.return_value = mock_reader

        with pytest.raises(KeyError):
            scan_climate_data(self.store_path, array_name="nonexistent")

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_different_storage_options(self, mock_reader_class):
        """Test scanning with different storage options."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        storage_options_tests = [
            {"anon": True},
            {"key": "access", "secret": "secret"},
            {"region_name": "us-east-1"},
            None,
        ]

        for options in storage_options_tests:
            result = scan_climate_data(
                self.store_path, array_name="temperature", storage_options=options
            )

            assert result == mock_lf
            # Check that storage_options were passed correctly
            call_args = mock_reader_class.call_args[1]
            assert call_args["storage_options"] == options

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_streaming_parameter(self, mock_reader_class):
        """Test streaming parameter behavior."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        # Test streaming=True
        scan_climate_data(self.store_path, array_name="temperature", streaming=True)

        call_args = mock_reader.read_array.call_args[1]
        assert call_args["streaming"] is True

        # Test streaming=False
        scan_climate_data(self.store_path, array_name="temperature", streaming=False)

        call_args = mock_reader.read_array.call_args[1]
        assert call_args["streaming"] is False

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_select_dims_types(self, mock_reader_class):
        """Test different types of dimension selections."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        select_dims_tests = [
            # Slice objects
            {"time": slice(0, 10), "lat": slice(None), "lon": slice(100, 200)},
            # Integer indices
            {"time": 5, "lat": 50, "lon": 150},
            # List of indices
            {"time": [0, 5, 10], "lat": [25, 50, 75], "lon": slice(None)},
            # Mixed types
            {"time": slice(0, 10), "lat": [25, 50], "lon": 150},
        ]

        for select_dims in select_dims_tests:
            result = scan_climate_data(
                self.store_path, array_name="temperature", select_dims=select_dims
            )

            assert result == mock_lf
            call_args = mock_reader.read_array.call_args[1]
            assert call_args["select_dims"] == select_dims

    @patch("src.data_access.zarr_scanner.ClimateDataReader")
    def test_consolidated_metadata_parameter(self, mock_reader_class):
        """Test consolidated metadata parameter."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        # Test consolidated=True
        scan_climate_data(self.store_path, array_name="temperature", consolidated=True)

        call_args = mock_reader_class.call_args[1]
        assert call_args["consolidated"] is True

        # Test consolidated=False
        scan_climate_data(self.store_path, array_name="temperature", consolidated=False)

        call_args = mock_reader_class.call_args[1]
        assert call_args["consolidated"] is False
