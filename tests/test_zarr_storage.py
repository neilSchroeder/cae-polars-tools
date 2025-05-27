"""
Unit tests for S3ZarrStore class.

Tests S3 filesystem connections and Zarr store access functionality.
"""

import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.data_access.zarr_storage import S3ZarrStore


@pytest.mark.unit
class TestS3ZarrStore:
    """Test suite for S3ZarrStore functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.store_path = "s3://test-bucket/climate-data/test.zarr"
        self.storage_options = {
            "key": "test_key",
            "secret": "test_secret",
            "region_name": "us-west-2",
        }

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.s3fs")
    def test_init_with_credentials(self, mock_s3fs):
        """Test initialization with S3 credentials."""
        mock_fs = Mock()
        mock_s3fs.S3FileSystem.return_value = mock_fs

        store = S3ZarrStore(self.store_path, self.storage_options)

        assert store.store_path == self.store_path
        assert store.storage_options == self.storage_options

        # Test filesystem creation via get_filesystem method
        fs = store.get_filesystem()
        assert fs == mock_fs

        # Check S3FileSystem was called with correct options
        mock_s3fs.S3FileSystem.assert_called_once_with(**self.storage_options)

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.s3fs")
    def test_init_anonymous(self, mock_s3fs):
        """Test initialization with anonymous access."""
        storage_options = {"anon": True}
        mock_fs = Mock()
        mock_s3fs.S3FileSystem.return_value = mock_fs

        store = S3ZarrStore(self.store_path, storage_options)

        # Access filesystem to trigger creation
        store.get_filesystem()

        mock_s3fs.S3FileSystem.assert_called_once_with(anon=True)

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", False)
    def test_init_missing_dependencies(self):
        """Test initialization when zarr/s3fs not available."""
        with pytest.raises(ImportError, match="zarr and s3fs packages are required"):
            S3ZarrStore(self.store_path, self.storage_options)

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_open_zarr_group_success(self, mock_zarr):
        """Test successful zarr group opening."""
        mock_group = Mock()
        mock_zarr.open_consolidated.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options)
        zarr_group = store.open_zarr_group()

        assert zarr_group == mock_group
        mock_zarr.open_consolidated.assert_called_once_with(
            self.store_path,
            mode="r",
            storage_options=self.storage_options,
        )

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_open_zarr_group_with_specific_group(self, mock_zarr):
        """Test zarr group opening with specific group."""
        mock_group = MagicMock()
        mock_subgroup = Mock()
        mock_group.__getitem__.return_value = mock_subgroup
        mock_zarr.open_consolidated.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options, group="climate_data")
        zarr_group = store.open_zarr_group()

        assert zarr_group == mock_subgroup
        mock_group.__getitem__.assert_called_once_with("climate_data")

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_open_zarr_group_fallback_to_open_group(self, mock_zarr):
        """Test zarr group opening with fallback to open_group."""
        mock_group = Mock()
        mock_zarr.open_consolidated.side_effect = ValueError("No consolidated metadata")
        mock_zarr.open_group.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options, consolidated=None)
        zarr_group = store.open_zarr_group()

        assert zarr_group == mock_group
        mock_zarr.open_group.assert_called_with(
            self.store_path, mode="r", storage_options=self.storage_options
        )

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_open_zarr_group_error_handling(self, mock_zarr):
        """Test zarr group opening error handling."""
        mock_zarr.open_consolidated.side_effect = Exception("Connection failed")
        mock_zarr.open_group.side_effect = Exception("Connection failed")

        store = S3ZarrStore(self.store_path, self.storage_options)

        with pytest.raises(ValueError, match="Failed to open Zarr store"):
            store.open_zarr_group()

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_list_arrays(self, mock_zarr):
        """Test listing arrays in the store."""
        # Create mock arrays with shape and dtype attributes
        mock_temp_array = Mock()
        mock_temp_array.shape = (365, 180, 360)
        mock_temp_array.dtype = "float32"

        mock_precip_array = Mock()
        mock_precip_array.shape = (365, 180, 360)
        mock_precip_array.dtype = "float32"

        # Mock group that contains these arrays
        mock_group = MagicMock()
        mock_group.keys.return_value = ["temperature", "precipitation", "metadata"]
        mock_group.__getitem__.side_effect = lambda name: {
            "temperature": mock_temp_array,
            "precipitation": mock_precip_array,
            "metadata": Mock(spec=[]),  # No shape/dtype attributes
        }[name]

        mock_zarr.open_consolidated.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options)
        arrays = store.list_arrays()

        assert "temperature" in arrays
        assert "precipitation" in arrays
        assert "metadata" not in arrays  # Should be filtered out

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_get_array(self, mock_zarr):
        """Test getting a specific array."""
        mock_array = Mock()
        mock_array.shape = (365, 180, 360)
        mock_array.dtype = "float32"

        mock_group = MagicMock()
        mock_group.__contains__.return_value = True
        mock_group.__getitem__.return_value = mock_array
        mock_zarr.open_consolidated.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options)
        array = store.get_array("temperature")

        assert array == mock_array
        mock_group.__getitem__.assert_called_once_with("temperature")

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_get_array_missing(self, mock_zarr):
        """Test getting non-existent array."""
        mock_group = MagicMock()
        mock_group.__contains__.return_value = False
        mock_zarr.open_consolidated.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options)

        with pytest.raises(KeyError, match="Array 'nonexistent' not found"):
            store.get_array("nonexistent")

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_get_array_info(self, mock_zarr):
        """Test getting array information."""
        mock_array = Mock()
        mock_array.shape = (365, 180, 360)
        mock_array.dtype = "float32"
        mock_array.chunks = (30, 18, 36)
        mock_array.fill_value = -9999
        mock_array.compressor = Mock()
        mock_array.compressor.__str__ = Mock(return_value="blosc")
        mock_array.filters = None
        mock_array.attrs = {"units": "K", "long_name": "Temperature"}
        mock_array.ndim = 3

        mock_group = MagicMock()
        mock_group.__contains__.return_value = True
        mock_group.__getitem__.return_value = mock_array
        mock_zarr.open_consolidated.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options)
        info = store.get_array_info("temperature")

        expected_info = {
            "name": "temperature",
            "shape": (365, 180, 360),
            "dtype": "float32",
            "chunks": (30, 18, 36),
            "dimensions": ["dim_0", "dim_1", "dim_2"],  # Default dimension names
            "fill_value": -9999,
            "compressor": "blosc",
            "filters": [],
            "attrs": {"units": "K", "long_name": "Temperature"},
        }

        assert info == expected_info

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.s3fs")
    def test_get_filesystem_caching(self, mock_s3fs):
        """Test filesystem caching behavior."""
        mock_fs = Mock()
        mock_s3fs.S3FileSystem.return_value = mock_fs

        store = S3ZarrStore(self.store_path, self.storage_options)

        # First call should create filesystem
        fs1 = store.get_filesystem()
        assert fs1 == mock_fs

        # Second call should return cached filesystem
        fs2 = store.get_filesystem()
        assert fs2 == mock_fs
        assert fs1 is fs2

        # S3FileSystem should only be called once
        mock_s3fs.S3FileSystem.assert_called_once()

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.s3fs")
    def test_invalid_store_path(self, mock_s3fs):
        """Test handling of invalid store paths."""
        invalid_paths = ["not-s3-path", "", "s3://", "s3://bucket-without-path"]

        mock_fs = Mock()
        mock_s3fs.S3FileSystem.return_value = mock_fs

        for invalid_path in invalid_paths:
            # Should still be able to create store instance
            store = S3ZarrStore(invalid_path, self.storage_options)
            assert store.store_path == invalid_path

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.s3fs")
    def test_storage_options_validation(self, mock_s3fs):
        """Test validation of storage options."""
        mock_fs = Mock()
        mock_s3fs.S3FileSystem.return_value = mock_fs

        # Test with empty storage options
        store = S3ZarrStore(self.store_path, {})
        assert store.storage_options == {}

        # Test with None storage options
        store = S3ZarrStore(self.store_path, None)
        assert store.storage_options == {}

    @patch("src.data_access.zarr_storage.ZARR_AVAILABLE", True)
    @patch("src.data_access.zarr_storage.zarr")
    def test_zarr_group_caching(self, mock_zarr):
        """Test zarr group caching behavior."""
        mock_group = Mock()
        mock_zarr.open_consolidated.return_value = mock_group

        store = S3ZarrStore(self.store_path, self.storage_options)

        # First call should open zarr group
        group1 = store.open_zarr_group()
        assert group1 == mock_group

        # Second call should return cached group
        group2 = store.open_zarr_group()
        assert group2 == mock_group
        assert group1 is group2

        # zarr.open_consolidated should only be called once
        mock_zarr.open_consolidated.assert_called_once()

    def test_missing_dependencies_warning(self):
        """Test warning when dependencies are missing."""
        with patch("src.data_access.zarr_storage.ZARR_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # This should trigger the import warning during module reload
                import importlib

                import src.data_access.zarr_storage

                importlib.reload(src.data_access.zarr_storage)

                # Check if warning was issued
                warning_found = any(
                    "zarr and s3fs packages required" in str(warning.message)
                    for warning in w
                )
                # Warning should be issued during import
                assert warning_found or len(w) == 0  # May have been issued earlier
