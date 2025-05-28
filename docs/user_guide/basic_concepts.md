# Basic Concepts

This guide covers the fundamental concepts and architecture of the cae-polars package.

## Overview

The cae-polars package provides efficient tools for reading and processing large-scale scientific data stored in Zarr format, with seamless conversion to Polars DataFrames for analysis.

## Key Components

### 1. Zarr Data Access

The package provides several classes for working with Zarr data:

- **ZarrDataReader**: Core class for reading Zarr arrays from various storage backends
- **ZarrScanner**: Tools for discovering and inspecting Zarr datasets
- **S3ZarrStore**: Specialized storage interface for S3-backed Zarr stores

### 2. Data Processing

- **CoordinateProcessor**: Handles coordinate system transformations and selections
- **PolarsConverter**: Converts multi-dimensional arrays to Polars DataFrames

### 3. Storage Backends

The package supports multiple storage backends:

- **Local filesystem**: Direct access to local Zarr stores
- **S3**: Cloud-based storage with AWS S3 integration
- **HTTP/HTTPS**: Remote access to web-hosted Zarr stores

## Data Flow

1. **Discovery**: Use `ZarrScanner` to explore available datasets
2. **Reading**: Use `ZarrDataReader` to load specific arrays
3. **Processing**: Apply coordinate transformations with `CoordinateProcessor`
4. **Conversion**: Convert to Polars DataFrames using `PolarsConverter`
5. **Analysis**: Leverage Polars' powerful query engine for data analysis

## Memory Management

The package is designed for efficient memory usage:

- **Lazy loading**: Data is only loaded when needed
- **Streaming**: Large datasets can be processed in chunks
- **Coordinate caching**: Frequently used coordinates are cached for performance

## Performance Considerations

- Use streaming operations for large datasets (>1GB)
- Leverage coordinate selection to reduce memory usage
- Consider chunking strategies for parallel processing
- Use appropriate data types to minimize memory footprint
