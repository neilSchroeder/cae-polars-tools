# Contributing to CAE-Polars-Tools

Thank you for your interest in contributing to CAE-Polars-Tools! This document provides guidelines and information for contributors.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the [issue tracker](https://github.com/nschroed/cae-polars-tools/issues) to see if the issue has already been reported.

When creating a bug report, please include:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)
- Any relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Include examples if applicable

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Add or update tests as needed
5. Update documentation if required
6. Ensure all tests pass
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Setting up the development environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/cae-polars-tools.git
cd cae-polars

# install uv
pip install uv

# get environment up and running
uv sync
source .venv/bin/activate
```

### Running Tests

```bash
# Run all tests
uv run python -m pytest

# Run tests with coverage
uv run python -m pytest --cov=src --cov-report=html

# Run specific test categories
uv run python pytest -m "not slow"           # Skip slow tests
uv run python pytest -m "integration"       # Run integration tests
uv run python pytest -m "benchmark"         # Run benchmark tests
```

### Code Style

This project uses several tools to maintain code quality:

- **Black** for code formatting

```bash
# Format code
black src tests
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. These run automatically before each commit:

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Project Structure

```
cae-polars-tools/
├── examples/
│   └── basic_usage.py          # Simple usage examples
├── src/
│   └── data_access/                 # Main package
│       ├── __init__.py
│       ├── zarr_scanner.py          # High-level API
│       ├── zarr_reader.py           # ZarrDataReader class
│       ├── zarr_storage.py          # S3 storage management
│       ├── coordinate_processor.py  # Coordinate handling
│       └── polars_converter.py      # Data conversion
├── tests/                     # Test suite
|   ├── conftest.py
|   ├── test_cli.py
|   ├── test_coordinate_processor.py
|   ├── test_polars_converter.py
|   ├── test_zarr_reader.py
|   ├── test_zarr_scanner.py
|   └── test_zarr_storage.py
├── docs/                      # Documentation
├── pyproject.toml             # Project configuration
└── README.md
```

## Coding Guidelines

### General Principles

- Write clear, readable code with meaningful variable names
- Add type hints to all function signatures
- Include docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Follow the existing code style and patterns

### Documentation

- Use numpy-style docstrings
- Include examples in docstrings where helpful
- Update README.md if adding new features
- Add type hints for better IDE support

### Testing

- Write tests for new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Include both unit and integration tests
- Add benchmark tests for performance-critical features

### Performance Considerations

- Profile code changes that might affect performance
- Consider memory usage, especially for large datasets
- Test with realistic data sizes
- Document any performance trade-offs

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release PR
4. Tag the release after merging
5. GitHub Actions will handle publishing to PyPI

## Getting Help

- Check the [documentation](https://cae-polars.readthedocs.io)
- Look through [existing issues](https://github.com/nschroed/cae-polars/issues)
- Ask questions in [discussions](https://github.com/nschroed/cae-polars/discussions)

## Recognition

Contributors will be recognized in:
- The CHANGELOG.md file
- The project's contributors list
- Release notes for significant contributions

Thank you for contributing to CAE-Polars!
