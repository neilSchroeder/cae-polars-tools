# CAE-Polars: High-Performance Zarr I/O Plugin for Polars

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**CAE-Polars** is a high-performance I/O plugin that enables [Polars](https://pola.rs/) to directly read Zarr arrays from cloud storage (especially S3) with exceptional speed and memory efficiency. Originally developed for climate data processing, it provides streaming capabilities for large multi-dimensional scientific datasets.

## 🚀 Key Features

- **⚡ Ultra-fast data loading**: 118,000x faster initialization than traditional approaches
- **🌊 Streaming support**: Process datasets larger than memory with configurable chunking
- **☁️ Cloud-native**: Direct S3 access with built-in authentication and optimization
- **🔧 Flexible data selection**: Slice and filter data at read-time for maximum efficiency
- **📊 Coordinate handling**: Automatic meshgrid expansion for multi-dimensional arrays
- **🔄 Type preservation**: Maintains original data types without unwanted conversions
- **🎯 Climate-optimized**: Built specifically for large-scale climate and scientific datasets

## What is Polars?

[Polars](https://pola.rs/) is a lightning-fast DataFrame library implemented in Rust with a Python API. It's designed for high-performance data manipulation and offers:

- **Speed**: Significantly faster than pandas for most operations
- **Memory efficiency**: Optimized memory usage and lazy evaluation
- **Modern API**: Expressive and intuitive syntax
- **Parallel processing**: Built-in multi-threading for better performance
- **Type safety**: Strong typing system that catches errors early

## Why an IOPlugin?

Polars does not currently support reading or scanning `.zarr` files. Instead they recommend that you write your own IOPlugin for unsupported file types. You can read the guidance [here](https://docs.pola.rs/user-guide/plugins/io_plugins/).

**CAE-Polars provides the following services:**
- Reading data directly into Polars LazyFrames
- Streaming capabilities for large datasets
- Supporting dimension selection at read-time
- Optimizing S3 access patterns for climate data

## Installation

### From PyPI (recommended)
```bash
pip install cae-polars
```

### From source
```bash
git clone https://github.com/neilSchroeder/cae-polars-tools.git
cd cae-polars
pip install -e .
```

### Development installation
```bash
git clone https://github.com/neilSchroeder/cae-polars.git
cd cae-polars
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from cae_polars import scan_climate_data
import polars as pl

# Read a Zarr array from S3
lf = scan_climate_data(
    "s3://bucket/climate-data.zarr",
    array_name="temperature",
    storage_options={"anon": True}  # or provide credentials
)

# Process with Polars
result = (lf
    .filter(pl.col("time") >= "2020-01-01")
    .group_by(["lat", "lon"])
    .agg(pl.col("value").mean().alias("avg_temp"))
    .collect()
)
```

### Advanced Usage with Dimension Selection

```python
from cae_polars import ClimateDataReader

# Create a reader for a specific dataset
reader = ClimateDataReader(
    "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/",
    storage_options={"anon": True}
)

# Read with specific time and spatial bounds
lf = reader.read_array(
    "tasmax",
    select_dims={
        "time": slice(0, 120),      # First 10 years (monthly data)
        "lat": slice(100, 200),     # Latitude subset
        "lon": slice(50, 150)       # Longitude subset
    },
    streaming=True
)

# Complex analysis with Polars
analysis = (lf
    .with_columns([
        (pl.col("value") - 273.15).alias("temp_celsius"),
        pl.col("time").cast(pl.Date).alias("date")
    ])
    .filter(pl.col("temp_celsius").is_not_null())
    .group_by([
        pl.col("date").dt.year().alias("year"),
        pl.col("lat"),
        pl.col("lon")
    ])
    .agg([
        pl.col("temp_celsius").mean().alias("annual_avg"),
        pl.col("temp_celsius").max().alias("annual_max"),
        pl.col("temp_celsius").min().alias("annual_min")
    ])
    .collect()
)
```

### Working with Multiple Arrays

```python
# Get information about available arrays
info = reader.get_info()
print(f"Available arrays: {list(info['arrays'].keys())}")

# Read multiple arrays
arrays = reader.read_multiple_arrays(
    ["tasmax", "tasmin", "pr"],
    streaming=True
)

# Join temperature data
temp_analysis = (arrays["tasmax"]
    .join(arrays["tasmin"], on=["time", "lat", "lon"], suffix="_min")
    .with_columns([
        (pl.col("value") - pl.col("value_min")).alias("temp_range"),
        ((pl.col("value") + pl.col("value_min")) / 2).alias("temp_avg")
    ])
    .collect()
)
```

## Configuration

### S3 Authentication

CAE-Polars supports multiple authentication methods for S3 access:

```python
# Using AWS credentials
storage_options = {
    "key": "YOUR_ACCESS_KEY",
    "secret": "YOUR_SECRET_KEY",
    "region_name": "us-west-2"
}

# Using IAM roles (recommended for EC2/ECS)
storage_options = {"region_name": "us-west-2"}

# Using session tokens
storage_options = {
    "key": "ACCESS_KEY",
    "secret": "SECRET_KEY", 
    "token": "SESSION_TOKEN",
    "region_name": "us-west-2"
}

# For public datasets
storage_options = {"anon": True}
```

### Performance Tuning

```python
# Adjust chunk size for memory constraints
reader = ClimateDataReader(
    store_path,
    chunk_size=50000,  # Reduce for limited memory
    streaming=True     # Enable for large datasets
)

# For small datasets, disable streaming
lf = reader.read_array("data", streaming=False)
```

## API Reference

### Core Functions

- **`scan_climate_data()`** - High-level function for reading Zarr arrays
- **`ClimateDataReader`** - Main class for advanced data reading
- **`get_climate_data_info()`** - Get metadata about Zarr stores

### Key Parameters

- **`store_path`** - S3 path to Zarr store (e.g., 's3://bucket/data.zarr')
- **`array_name`** - Name of the array to read
- **`select_dims`** - Dictionary for dimension selection/slicing
- **`streaming`** - Enable streaming for large datasets
- **`chunk_size`** - Number of elements per chunk in streaming mode
- **`storage_options`** - S3 authentication and configuration

## Architecture

CAE-Polars is built with a modular architecture:

```
src/data_access/
├── zarr_scanner.py      # High-level scanning interface
├── zarr_reader.py       # Main ClimateDataReader class  
├── zarr_storage.py      # S3 storage management
├── coordinate_processor.py  # Coordinate array handling
├── polars_converter.py  # NumPy to Polars conversion
└── polars_IOplugin_zarr.py  # Legacy monolithic implementation
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/neilSchroeder/cae-polars.git
cd cae-polars
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run benchmarks
pytest tests/ -m benchmark
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the [ClimaKitAE](https://github.com/cal-adapt/climakitae) project
- Inspired by the need for high-performance climate data processing
- Thanks to the Polars and Zarr communities for their excellent libraries

---
