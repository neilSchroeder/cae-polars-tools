# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation with Read the Docs configuration
- NumPy-style docstrings for all modules, classes, and functions
- User guide with practical examples and best practices
- API reference documentation
- Performance optimization guide
- Examples section with real-world use cases

### Changed
- Enhanced error handling with better exception chaining
- Improved code formatting and PEP 8 compliance

### Fixed
- Resolved trailing whitespace issues in source files
- Improved import organization

## [0.1.0] - 2025-01-XX

### Added
- Initial release of cae-polars
- ZarrDataReader for efficient Zarr array reading
- Support for S3, local filesystem, and HTTP storage backends
- PolarsConverter for converting Zarr arrays to Polars DataFrames
- CoordinateProcessor for handling multi-dimensional coordinate systems
- Streaming and lazy evaluation capabilities
- Command-line interface for data inspection and benchmarking
- Comprehensive test suite with high coverage
- Basic documentation and README

### Core Features
- **Data Access**: Read Zarr arrays from various storage backends
- **Data Conversion**: Convert multi-dimensional arrays to Polars DataFrames
- **Coordinate Handling**: Efficient coordinate system processing
- **Memory Management**: Streaming and lazy loading for large datasets
- **Cloud Integration**: Native S3 support with credential management
- **CLI Tools**: Command-line utilities for data exploration

### Dependencies
- polars >= 0.20.0
- zarr >= 2.14.0
- xarray >= 2023.1.0
- numpy >= 1.21.0
- boto3 >= 1.26.0 (for S3 support)
- click >= 8.0.0 (for CLI)

[Unreleased]: https://github.com/your-org/cae-polars/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/cae-polars/releases/tag/v0.1.0
