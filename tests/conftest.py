"""
Pytest configuration and shared fixtures for CAE-Polars tests.

This module provides common fixtures and configuration for testing the
CAE-Polars package across unit and integration tests.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def sample_zarr_data():
    """Sample multi-dimensional climate data for testing."""
    return {
        "temperature": {
            "data": np.random.uniform(280, 320, (12, 10, 15)).astype(np.float32),
            "dims": ["time", "lat", "lon"],
            "attrs": {
                "units": "K",
                "long_name": "Air Temperature",
                "standard_name": "air_temperature",
            },
        },
        "precipitation": {
            "data": np.random.uniform(0, 10, (12, 10, 15)).astype(np.float32),
            "dims": ["time", "lat", "lon"],
            "attrs": {
                "units": "mm/day",
                "long_name": "Precipitation",
                "standard_name": "precipitation_flux",
            },
        },
    }


@pytest.fixture
def sample_coordinates():
    """Sample coordinate arrays for testing."""
    return {
        "time": np.arange(12, dtype=np.int32),  # 12 months
        "lat": np.linspace(30.0, 50.0, 10, dtype=np.float32),  # 10 lat points
        "lon": np.linspace(-120.0, -100.0, 15, dtype=np.float32),  # 15 lon points
    }


@pytest.fixture
def mock_s3_credentials():
    """Mock S3 credentials for testing."""
    return {
        "key": "test_access_key",
        "secret": "test_secret_key",
        "token": "test_session_token",
        "region_name": "us-west-2",
    }


@pytest.fixture
def mock_zarr_store():
    """Mock Zarr store with sample data."""
    with patch("zarr.open") as mock_open:
        mock_store = Mock()
        mock_store.keys.return_value = ["temperature", "precipitation"]

        # Mock array access
        mock_temp = Mock()
        mock_temp.shape = (12, 10, 15)
        mock_temp.dtype = np.float32
        mock_temp.chunks = (6, 5, 8)
        mock_temp.attrs = {"units": "K", "long_name": "Air Temperature"}
        mock_temp.__getitem__ = Mock(
            return_value=np.random.uniform(280, 320, (12, 10, 15))
        )

        mock_precip = Mock()
        mock_precip.shape = (12, 10, 15)
        mock_precip.dtype = np.float32
        mock_precip.chunks = (6, 5, 8)
        mock_precip.attrs = {"units": "mm/day", "long_name": "Precipitation"}
        mock_precip.__getitem__ = Mock(
            return_value=np.random.uniform(0, 10, (12, 10, 15))
        )

        mock_store.__getitem__ = Mock(
            side_effect=lambda x: {
                "temperature": mock_temp,
                "precipitation": mock_precip,
            }[x]
        )

        mock_open.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_s3fs():
    """Mock s3fs filesystem for testing."""
    with patch("s3fs.S3FileSystem") as mock_s3fs:
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_s3fs.return_value = mock_fs
        yield mock_fs


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def select_dims_small():
    """Small dimension selection for testing."""
    return {"time": slice(0, 3), "lat": slice(2, 5), "lon": slice(5, 10)}


@pytest.fixture
def select_dims_single():
    """Single point selection for testing."""
    return {"time": 0, "lat": 3, "lon": 7}


@pytest.fixture
def select_dims_list():
    """List-based dimension selection for testing."""
    return {
        "time": [0, 2, 4, 6],
        "lat": [1, 3, 5],
        "lon": slice(None),  # All longitude points
    }


@pytest.fixture
def storage_options(mock_s3_credentials):
    """Default storage options for testing."""
    return mock_s3_credentials


@pytest.fixture
def sample_store_path():
    """Sample S3 store path for testing."""
    return "s3://test-bucket/climate-data/test-dataset.zarr"


@pytest.fixture(autouse=True)
def mock_zarr_dependencies():
    """Auto-mock zarr dependencies to avoid import errors in tests."""
    with patch.dict("sys.modules", {"zarr": Mock(), "s3fs": Mock()}):
        yield


class MockZarrArray:
    """Mock Zarr array for testing."""

    def __init__(self, data, dims, attrs=None, chunks=None):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.dims = dims
        self.attrs = attrs or {}
        self.chunks = chunks or tuple(min(s, 100) for s in self.shape)

    def __getitem__(self, key):
        return self.data[key]

    def __array__(self):
        return self.data


@pytest.fixture
def mock_zarr_arrays(sample_zarr_data, sample_coordinates):
    """Create mock Zarr arrays from sample data."""
    arrays = {}
    for name, info in sample_zarr_data.items():
        arrays[name] = MockZarrArray(
            data=info["data"], dims=info["dims"], attrs=info["attrs"], chunks=(6, 5, 8)
        )

    # Add coordinate arrays
    for coord_name, coord_data in sample_coordinates.items():
        arrays[coord_name] = MockZarrArray(
            data=coord_data,
            dims=[coord_name],
            attrs={"units": "various", "long_name": f"{coord_name} coordinate"},
        )

    return arrays


@pytest.fixture
def sample_polars_data():
    """Sample Polars-compatible data for testing."""
    np.random.seed(42)  # For reproducible tests
    n_points = 100

    return {
        "time": np.random.randint(0, 12, n_points),
        "lat": np.random.uniform(30.0, 50.0, n_points),
        "lon": np.random.uniform(-120.0, -100.0, n_points),
        "value": np.random.uniform(280.0, 320.0, n_points),
    }


# Test markers for different test categories
pytest_plugins = []


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_s3: mark test as requiring S3 access")
