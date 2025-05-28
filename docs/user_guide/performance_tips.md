# Performance Tips

This guide provides best practices for optimizing performance when working with large datasets.

## Memory Management

### Lazy Loading

Always use lazy operations when possible to minimize memory usage:

```python
from src.data_access import ZarrDataReader, PolarsConverter

reader = ZarrDataReader("large_dataset.zarr")
converter = PolarsConverter()

# Good: Lazy loading
df_lazy = converter.array_to_polars_lazy(data)
result = df_lazy.filter(pl.col("temp") > 273.15).collect()

# Avoid: Eager loading of large datasets
df_eager = converter.array_to_polars_lazy(data).collect()  # Loads everything!
```

### Coordinate Selection

Limit data loading by selecting only needed coordinates:

```python
# Good: Select only needed region and time period
data = reader.read_array(
    "temperature",
    coordinates={
        "time": slice("2020-01-01", "2020-01-31"),  # One month only
        "lat": slice(40, 50),                       # Limited region
        "lon": slice(-80, -70)
    }
)

# Avoid: Loading entire dataset
data = reader.read_array("temperature")  # Loads all data!
```

### Streaming Operations

For very large datasets, use streaming to process data in chunks:

```python
# Enable streaming for large datasets
df_stream = converter.array_to_polars_lazy(
    large_data,
    streaming=True
)

# Process with streaming
result = df_stream.filter(
    pl.col("temperature") > 273.15
).group_by("time").agg([
    pl.col("temperature").mean()
]).collect(streaming=True)
```

## Chunking Strategies

### Optimal Chunk Sizes

Use the dataset's natural chunk structure:

```python
# Get array information
info = reader.get_array_info("temperature")
optimal_chunks = info['chunks']

# Use chunk-aligned coordinate selections
time_chunk = optimal_chunks[0]  # e.g., 30 days
lat_chunk = optimal_chunks[1]   # e.g., 180 degrees
lon_chunk = optimal_chunks[2]   # e.g., 360 degrees
```

### Parallel Processing

Process multiple chunks in parallel:

```python
import concurrent.futures
from itertools import product

def process_chunk(time_slice, lat_slice, lon_slice):
    data = reader.read_array(
        "temperature",
        coordinates={
            "time": time_slice,
            "lat": lat_slice, 
            "lon": lon_slice
        }
    )
    df = converter.array_to_polars_lazy(data)
    return df.filter(pl.col("temperature") > 273.15).collect()

# Define chunk ranges
time_chunks = [slice(f"2020-{m:02d}-01", f"2020-{m:02d}-28") for m in range(1, 13)]
lat_chunks = [slice(i, i+10) for i in range(0, 180, 10)]
lon_chunks = [slice(i, i+10) for i in range(0, 360, 10)]

# Process chunks in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for t_chunk, lat_chunk, lon_chunk in product(time_chunks[:2], lat_chunks[:2], lon_chunks[:2]):
        future = executor.submit(process_chunk, t_chunk, lat_chunk, lon_chunk)
        futures.append(future)
    
    results = [future.result() for future in futures]

# Combine results
final_result = pl.concat(results)
```

## Data Type Optimization

### Choose Appropriate Precision

Use smaller data types when precision allows:

```python
# Convert to smaller types for memory savings
df_optimized = df.with_columns([
    pl.col("temperature").cast(pl.Float32),  # vs Float64
    pl.col("lat").cast(pl.Float32),
    pl.col("lon").cast(pl.Float32),
    pl.col("time").cast(pl.Datetime("ms"))   # vs microsecond precision
])
```

### Categorical Data

Use categorical types for repeated string values:

```python
# Convert string columns to categorical
df_categorical = df.with_columns([
    pl.col("station_id").cast(pl.Categorical),
    pl.col("region").cast(pl.Categorical)
])
```

## Query Optimization

### Filter Early

Apply filters as early as possible in the processing pipeline:

```python
# Good: Filter early
result = df_lazy.filter(
    pl.col("temperature") > 273.15
).filter(
    pl.col("time").is_between("2020-01-01", "2020-12-31")
).select([
    "time", "lat", "lon", "temperature"
]).collect()

# Less efficient: Filter after expensive operations
result = df_lazy.group_by("station").agg([
    pl.col("temperature").mean()
]).filter(
    pl.col("temperature") > 273.15  # Filter after aggregation
).collect()
```

### Projection Pushdown

Select only needed columns early:

```python
# Good: Select columns early
result = df_lazy.select([
    "time", "temperature"  # Only needed columns
]).filter(
    pl.col("temperature") > 273.15
).collect()

# Less efficient: Select columns late
result = df_lazy.filter(
    pl.col("temperature") > 273.15
).select([
    "time", "temperature"  # All columns processed before selection
]).collect()
```

## Storage Optimization

### S3 Configuration

Optimize S3 access for better performance:

```python
from src.data_access import S3ZarrStore

# Configure for better performance
store = S3ZarrStore(
    bucket="my-bucket",
    key_prefix="data/",
    region="us-east-1",  # Use same region as compute
    # Use session token for temporary credentials
    aws_session_token="token"
)

# Use connection pooling for multiple requests
reader = ZarrDataReader(store)
```

### Caching

Implement coordinate caching for repeated access:

```python
# Cache frequently accessed coordinates
coordinate_cache = {}

def get_cached_coordinates(array_name, coord_selection):
    cache_key = (array_name, str(coord_selection))
    if cache_key not in coordinate_cache:
        data = reader.read_array(array_name, coordinates=coord_selection)
        coordinate_cache[cache_key] = data
    return coordinate_cache[cache_key]
```

## Monitoring Performance

### Memory Usage

Monitor memory consumption during processing:

```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Monitor during processing
initial_memory = monitor_memory()
print(f"Initial memory: {initial_memory:.1f} MB")

# Process data
result = df_lazy.collect()

final_memory = monitor_memory()
print(f"Final memory: {final_memory:.1f} MB")
print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
```

### Timing Operations

Profile different approaches:

```python
import time

def time_operation(operation_name, func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{operation_name}: {end_time - start_time:.2f} seconds")
    return result

# Compare different approaches
result1 = time_operation("Lazy approach", lambda: df_lazy.collect())
result2 = time_operation("Eager approach", lambda: df.collect())
```

## Common Pitfalls

### Avoid These Patterns

1. **Loading entire datasets when only subsets are needed**
2. **Creating multiple readers for the same dataset**
3. **Collecting lazy DataFrames too early**
4. **Not using coordinate selection**
5. **Processing data serially when parallel processing is possible**

### Best Practices Summary

1. **Use coordinate selection** to limit data loading
2. **Leverage lazy evaluation** throughout the pipeline
3. **Apply filters early** in the processing chain
4. **Choose appropriate data types** for memory efficiency
5. **Use streaming** for very large datasets
6. **Implement caching** for frequently accessed data
7. **Monitor performance** to identify bottlenecks
