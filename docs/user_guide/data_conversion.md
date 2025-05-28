# Data Conversion

This guide covers converting Zarr arrays to Polars DataFrames for analysis.

## Overview

The `PolarsConverter` class provides efficient conversion from multi-dimensional Zarr arrays to Polars DataFrames, enabling powerful data analysis capabilities.

## Basic Conversion

```python
from src.data_access import ZarrDataReader, PolarsConverter

# Read data
reader = ZarrDataReader("data.zarr")
data = reader.read_array("temperature")

# Convert to Polars DataFrame
converter = PolarsConverter()
df = converter.array_to_polars_lazy(data)

# Collect and inspect
result = df.collect()
print(result.head())
```

## Coordinate Integration

The converter automatically handles coordinate information:

```python
# Data with coordinates will include coordinate columns
data_with_coords = reader.read_array(
    "temperature",
    coordinates={"time": slice("2020-01-01", "2020-12-31")}
)

df = converter.array_to_polars_lazy(data_with_coords)
print(df.columns)  # ['time', 'lat', 'lon', 'temperature']
```

## Streaming Conversion

For large datasets, use streaming conversion to manage memory:

```python
# Enable streaming for large arrays
df_lazy = converter.array_to_polars_lazy(
    large_array,
    streaming=True
)

# Process in chunks
result = df_lazy.select([
    "time",
    "temperature"
]).filter(
    pl.col("temperature") > 273.15
).collect(streaming=True)
```

## Working with Multi-dimensional Data

### Reshaping Data

```python
# 3D array (time, lat, lon) -> DataFrame with coordinate columns
temp_3d = reader.read_array("temperature_3d")
df = converter.array_to_polars_lazy(temp_3d)

# Result has columns: ['time', 'lat', 'lon', 'temperature_3d']
print(df.schema)
```

### Coordinate Selection During Conversion

```python
# Select specific coordinates during conversion
df = converter.array_to_polars_lazy(
    data,
    coordinate_selection={
        "lat": [40.0, 41.0, 42.0],  # Specific latitudes
        "lon": slice(-74, -73)       # Longitude range
    }
)
```

## Advanced Usage

### Custom Column Names

```python
# Specify custom column names
df = converter.array_to_polars_lazy(
    data,
    value_column_name="temp_celsius",
    coordinate_names={"time": "timestamp", "lat": "latitude"}
)
```

### Data Type Optimization

```python
# Convert to appropriate Polars data types
df = converter.array_to_polars_lazy(data).with_columns([
    pl.col("time").cast(pl.Datetime),
    pl.col("temperature").cast(pl.Float32),  # Reduce precision if suitable
    pl.col("lat").cast(pl.Float32),
    pl.col("lon").cast(pl.Float32)
])
```

## Analysis Examples

### Time Series Analysis

```python
# Convert temperature data for time series analysis
temp_df = converter.array_to_polars_lazy(temperature_data)

# Calculate monthly averages
monthly_avg = temp_df.group_by([
    pl.col("time").dt.year().alias("year"),
    pl.col("time").dt.month().alias("month")
]).agg([
    pl.col("temperature").mean().alias("avg_temp"),
    pl.col("temperature").std().alias("temp_std")
]).collect()
```

### Spatial Analysis

```python
# Spatial filtering and aggregation
spatial_analysis = df.filter(
    (pl.col("lat").is_between(40, 50)) &
    (pl.col("lon").is_between(-80, -70))
).group_by(["lat", "lon"]).agg([
    pl.col("temperature").mean().alias("avg_temp"),
    pl.col("temperature").max().alias("max_temp"),
    pl.col("temperature").min().alias("min_temp")
]).collect()
```

### Multi-variable Analysis

```python
# Convert multiple variables for joint analysis
temp_data = reader.read_array("temperature")
pressure_data = reader.read_array("pressure")

temp_df = converter.array_to_polars_lazy(temp_data)
pressure_df = converter.array_to_polars_lazy(pressure_data)

# Join on coordinates
combined = temp_df.join(
    pressure_df,
    on=["time", "lat", "lon"],
    how="inner"
).collect()
```

## Performance Optimization

### Memory Management

```python
# Use lazy evaluation to minimize memory usage
df_lazy = converter.array_to_polars_lazy(large_data)

# Chain operations without materializing intermediate results
result = df_lazy.filter(
    pl.col("temperature") > 273.15
).select([
    "time", "lat", "lon", "temperature"
]).group_by("time").agg([
    pl.col("temperature").mean()
]).collect()
```

### Chunk Processing

```python
# Process data in chunks for very large datasets
def process_chunk(chunk_data):
    df = converter.array_to_polars_lazy(chunk_data)
    return df.filter(pl.col("temperature") > 0).collect()

# Apply to chunks
results = []
for chunk in data_chunks:
    chunk_result = process_chunk(chunk)
    results.append(chunk_result)

# Combine results
final_result = pl.concat(results)
```
