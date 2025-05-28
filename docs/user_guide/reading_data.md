# Reading Data

This guide covers how to read and access data using the cae-polars package.

## Quick Start

```python
from src.data_access import ZarrDataReader

# Create a reader for local Zarr store
reader = ZarrDataReader("path/to/zarr/store")

# List available arrays
arrays = reader.list_arrays()
print(f"Available arrays: {arrays}")

# Read a specific array
data = reader.read_array("temperature")
```

## Storage Backends

### Local Filesystem

```python
# Read from local directory
reader = ZarrDataReader("/data/climate/zarr_store")
```

### S3 Storage

```python
from src.data_access import S3ZarrStore

# Create S3 store with credentials
store = S3ZarrStore(
    bucket="my-data-bucket",
    key_prefix="climate/data",
    aws_access_key_id="your_key",
    aws_secret_access_key="your_secret"
)

# Create reader with S3 store
reader = ZarrDataReader(store)
```

### Remote HTTP/HTTPS

```python
# Read from remote Zarr store
reader = ZarrDataReader("https://example.com/data.zarr")
```

## Array Information

Before reading data, you can inspect array properties:

```python
# Get detailed information about an array
info = reader.get_array_info("temperature")
print(f"Shape: {info['shape']}")
print(f"Data type: {info['dtype']}")
print(f"Chunks: {info['chunks']}")
print(f"Dimensions: {info['dims']}")
```

## Reading Options

### Full Array Reading

```python
# Read entire array
data = reader.read_array("temperature")
```

### Coordinate Selection

```python
# Read with coordinate constraints
data = reader.read_array(
    "temperature",
    coordinates={
        "time": slice("2020-01-01", "2020-12-31"),
        "lat": slice(30, 60),
        "lon": slice(-120, -80)
    }
)
```

### Multiple Arrays

```python
# Read multiple related arrays
arrays = reader.read_multiple_arrays(
    ["temperature", "pressure", "humidity"],
    coordinates={"time": slice("2020-01-01", "2020-01-31")}
)
```

## Error Handling

```python
try:
    data = reader.read_array("nonexistent_array")
except KeyError as e:
    print(f"Array not found: {e}")
except Exception as e:
    print(f"Error reading data: {e}")
```

## Performance Tips

1. **Use coordinate selection** to limit data loading:
   ```python
   # Good: Only load needed region
   data = reader.read_array("temp", coordinates={"lat": slice(40, 50)})
   
   # Avoid: Loading entire array then filtering
   data = reader.read_array("temp")  # Loads everything!
   ```

2. **Leverage chunking** for large datasets:
   ```python
   # Process data in chunks for memory efficiency
   info = reader.get_array_info("large_dataset")
   chunk_size = info['chunks'][0]  # Use optimal chunk size
   ```

3. **Reuse readers** when possible:
   ```python
   # Good: Reuse reader for multiple operations
   reader = ZarrDataReader("data.zarr")
   temp = reader.read_array("temperature")
   pressure = reader.read_array("pressure")
   
   # Avoid: Creating new reader each time
   temp = ZarrDataReader("data.zarr").read_array("temperature")
   pressure = ZarrDataReader("data.zarr").read_array("pressure")
   ```
