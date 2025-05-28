# Installation

## Requirements

CAE-Polars requires Python 3.9 or later and has the following core dependencies:

- **polars** >= 0.19.0
- **numpy** >= 1.20.0  
- **zarr** >= 2.12.0
- **s3fs** >= 2023.6.0

## Install from PyPI

```bash
pip install cae-polars
```

## Install with Optional Dependencies

### Development Dependencies

For contributing to the project:

```bash
pip install cae-polars[dev]
```

This includes testing, linting, and formatting tools.

### Benchmarking Dependencies  

For running performance benchmarks:

```bash
pip install cae-polars[benchmark]
```

This includes additional packages like xarray, dask, and visualization tools.

### Documentation Dependencies

For building documentation locally:

```bash
pip install cae-polars[docs]
```

### All Dependencies

To install everything:

```bash
pip install cae-polars[all]
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/nschroed/cae-polars.git
cd cae-polars
pip install -e .
```

For development with all dependencies:

```bash
git clone https://github.com/nschroed/cae-polars.git
cd cae-polars
pip install -e .[dev,docs,benchmark]
```

## Verify Installation

Test your installation:

```python
import cae_polars
print(cae_polars.__version__)

# Test basic functionality
from cae_polars.data_access import ZarrDataReader
print("Installation successful!")
```

## Cloud Storage Setup

### AWS Credentials

For accessing private S3 buckets, configure your AWS credentials using one of these methods:

1. **AWS CLI**:
   ```bash
   aws configure
   ```

2. **Environment variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-west-2
   ```

3. **IAM roles** (recommended for EC2/Lambda)

4. **Credential files** in `~/.aws/credentials`

### Anonymous Access

For public datasets:

```python
from cae_polars.data_access import scan_data

# Use anonymous access for public data
df = scan_data(
    "s3://noaa-climate-data/dataset.zarr",
    storage_options={"anon": True}
)
```

## Troubleshooting

### Common Issues

**ImportError with zarr/s3fs**: Ensure all dependencies are installed:
```bash
pip install zarr s3fs
```

**AWS credential errors**: Verify your credentials are configured correctly:
```bash
aws sts get-caller-identity
```

**Memory issues with large datasets**: Use streaming mode and adjust chunk sizes:
```python
df = scan_data(
    path, 
    streaming=True, 
    chunk_size=5000
)
```

### Getting Help

- Check the [User Guide](user_guide/index.md)
- Open an issue on [GitHub](https://github.com/nschroed/cae-polars/issues)
- Review the [examples](examples/index.md)
