# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of CAE-Polars
- High-performance Zarr I/O plugin for Polars
- S3 cloud storage support with streaming capabilities
- Multi-dimensional array coordinate handling
- Climate data processing optimizations
- Comprehensive test suite and benchmarking
- Documentation and examples

### Features
- `scan_climate_data()` high-level function
- `ClimateDataReader` class for advanced usage
- Streaming support for large datasets
- Dimension selection and filtering
- S3 authentication support
- Type preservation and memory optimization

## [0.1.0] - 2025-01-XX

### Added
- Initial project structure
- Core Zarr reading functionality
- Polars integration
- S3 storage backend
- Basic documentation

[Unreleased]: https://github.com/nschroed/cae-polars/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nschroed/cae-polars/releases/tag/v0.1.0
