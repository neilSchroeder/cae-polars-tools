# CAE-Polars Documentation

Welcome to the documentation for **cae-polars**, a high-performance Zarr I/O plugin for Polars with specialized climate data processing capabilities.

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
quickstart
user_guide/index
api_reference/index
examples/index
contributing
changelog
```

## Overview

CAE-Polars provides efficient reading and processing of multi-dimensional climate datasets stored in Zarr format on cloud storage (S3). It combines the performance of Polars DataFrames with the scalability of Zarr arrays and the convenience of cloud storage access.

### Key Features

- **High-Performance I/O**: Optimized reading from S3-based Zarr stores
- **Memory Efficiency**: Intelligent streaming and caching for large datasets
- **Climate Data Focus**: Specialized coordinate handling for time, latitude, longitude
- **Polars Integration**: Seamless conversion to Polars LazyFrames
- **Flexible Selection**: Advanced dimension slicing and indexing
- **Cloud Native**: Built for modern cloud-based climate data workflows

### Quick Example

```python
from cae_polars.data_access import scan_data

# Read climate data from S3
df = scan_data(
    "s3://climate-data/era5.zarr",
    array_name="temperature",
    select_dims={
        "time": slice("2020-01-01", "2020-12-31"),
        "lat": slice(30, 60),
        "lon": slice(-120, -60)
    }
)

# Process with Polars
result = (
    df
    .filter(pl.col("temperature") > 273.15)  # Above freezing
    .group_by("time")
    .agg(pl.col("temperature").mean())
    .collect()
)
```

### Why CAE-Polars?

Climate data analysis often involves:
- **Large multi-dimensional arrays** (time × latitude × longitude × variables)
- **Cloud storage** for data sharing and collaboration
- **Efficient coordinate handling** for spatial and temporal operations
- **Memory-conscious processing** due to dataset sizes

CAE-Polars addresses these challenges by providing:
- Lazy evaluation and streaming capabilities
- Optimized coordinate expansion without full meshgrids
- Intelligent caching based on data characteristics
- Integration with the modern Python data ecosystem

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
