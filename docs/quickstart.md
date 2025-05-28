# Quick Start Guide

This guide will get you up and running with CAE Polars in just a few minutes.

## Installation

Install CAE Polars using pip:

```bash
pip install cae-polars
```

For development installation, see the [Installation Guide](installation.md).

## Basic Usage

### 1. Reading Zarr Data

```python
from cae_polars import scan_data

# Scan a Zarr dataset
data_info = scan_data("s3://my-bucket/dataset.zarr")
print(f"Available arrays: {data_info.array_names}")
```

### 2. Loading Data into Polars

```python
from cae_polars.data_access import ZarrDataReader

# Initialize reader
reader = ZarrDataReader("s3://my-bucket/dataset.zarr")

# Read array as Polars LazyFrame
lazy_df = reader.read_array(
    "temperature",
    select_dims={"time": slice(0, 100), "lat": slice(40, 60)}
)

# Execute and collect results
df = lazy_df.collect()
print(df.head())
```

### 3. Command Line Interface

```bash
# Get dataset information
cae-polars info s3://my-bucket/dataset.zarr

# Read specific array with selection
cae-polars read s3://my-bucket/dataset.zarr temperature \
    --select time:0:100 lat:40:60 \
    --output data.parquet
```

## Next Steps

- Check out the [User Guide](user_guide/index.md) for detailed tutorials
- Browse the [API Reference](api_reference/index.md) for complete documentation
- See [Examples](examples/index.md) for real-world usage scenarios
