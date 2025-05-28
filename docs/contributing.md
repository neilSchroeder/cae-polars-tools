# Contributing

We welcome contributions to the cae-polars project! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/cae-polars.git
   cd cae-polars
   ```

2. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_zarr_reader.py
```

## Documentation

To build the documentation locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

## Code Style

This project follows:
- **PEP 8** for Python code style
- **NumPy style** for docstrings
- **Black** for code formatting
- **isort** for import sorting

Run formatting tools:
```bash
black src/ tests/
isort src/ tests/
```

## Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run tests and checks**
   ```bash
   pytest
   black --check src/ tests/
   isort --check-only src/ tests/
   ```
7. **Submit a pull request**

## Guidelines

### Code Guidelines
- Write clear, readable code with meaningful variable names
- Add type hints to all public functions
- Follow the existing code structure and patterns
- Keep functions focused and small

### Documentation Guidelines
- Use NumPy-style docstrings for all public functions and classes
- Include examples in docstrings for public APIs
- Update user guides when adding new features
- Ensure documentation builds without warnings

### Testing Guidelines
- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names
- Include both positive and negative test cases
- Test edge cases and error conditions

## Reporting Issues

When reporting bugs or requesting features:

1. **Check existing issues** first
2. **Use the issue templates** provided
3. **Include minimal reproducible examples** for bugs
4. **Provide context** about your use case for feature requests

## Questions?

- Open a discussion on GitHub
- Check the documentation
- Look at existing issues and pull requests

Thank you for contributing to cae-polars!
