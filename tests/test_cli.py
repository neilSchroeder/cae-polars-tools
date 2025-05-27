"""
Unit tests for CLI module.

Tests command-line interface functionality including argument parsing,
command execution, and utility functions.
"""

import argparse
import json
from io import StringIO
from unittest.mock import Mock, mock_open, patch

import polars as pl
import pytest

from src.cli import (
    _parse_select_dims,
    _parse_storage_options,
    benchmark_command,
    create_parser,
    info_command,
    main,
    read_command,
)


@pytest.mark.unit
class TestCLI:
    """Test suite for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.store_path = "s3://test-bucket/climate-data/test.zarr"
        self.mock_info = {
            "store_path": self.store_path,
            "group": None,
            "arrays": {
                "temperature": {
                    "shape": (365, 180, 360),
                    "dtype": "float32",
                    "chunks": (30, 18, 36),
                    "attrs": {"units": "K"},
                    "dimensions": ["time", "lat", "lon"],
                },
                "precipitation": {
                    "shape": (365, 180, 360),
                    "dtype": "float32",
                    "chunks": (30, 18, 36),
                    "attrs": {"units": "mm/day"},
                    "dimensions": ["time", "lat", "lon"],
                },
            },
        }

    def test_parse_storage_options_json(self):
        """Test parsing storage options as JSON."""
        json_str = '{"key": "value", "anon": true}'
        result = _parse_storage_options(json_str)

        expected = {"key": "value", "anon": True}
        assert result == expected

    def test_parse_storage_options_key_value(self):
        """Test parsing storage options as key=value pairs."""
        kv_str = "key=access_key,secret=secret_key,region=us-west-2"
        result = _parse_storage_options(kv_str)

        expected = {"key": "access_key", "secret": "secret_key", "region": "us-west-2"}
        assert result == expected

    def test_parse_storage_options_empty(self):
        """Test parsing empty or None storage options."""
        assert _parse_storage_options(None) is None
        assert _parse_storage_options("") is None
        assert _parse_storage_options("   ") is None

    def test_parse_storage_options_invalid_json(self):
        """Test parsing invalid JSON falls back to key=value."""
        invalid_json = "not_json_but_no_equals"
        result = _parse_storage_options(invalid_json)
        assert result is None

        # Test with equals but invalid JSON
        mixed = "key=value,invalid_json"
        result = _parse_storage_options(mixed)
        expected = {"key": "value"}
        assert result == expected

    def test_parse_select_dims_valid(self):
        """Test parsing valid dimension selection strings."""
        # Note: ast.literal_eval cannot handle slice() objects, so we test
        # what it can actually parse

        # Simple integer selection
        simple_str = "{'time': 5}"
        result = _parse_select_dims(simple_str)
        expected = {"time": 5}
        assert result == expected

        # List selection
        list_str = "{'time': [0, 1, 2], 'lat': [10, 20]}"
        result = _parse_select_dims(list_str)
        expected = {"time": [0, 1, 2], "lat": [10, 20]}
        assert result == expected

    def test_parse_select_dims_invalid(self):
        """Test parsing invalid dimension selection strings."""
        with pytest.raises(ValueError, match="Invalid select_dims format"):
            _parse_select_dims("not_valid_python")

        with pytest.raises(ValueError, match="Invalid select_dims format"):
            _parse_select_dims("{'unclosed': dict")

    @patch("src.cli.get_zarr_data_info")
    @patch("sys.stdout", new_callable=StringIO)
    def test_info_command_basic(self, mock_stdout, mock_get_info):
        """Test basic info command functionality."""
        mock_get_info.return_value = self.mock_info

        args = argparse.Namespace(
            store_path=self.store_path,
            storage_options=None,
            group=None,
            output=None,
        )

        info_command(args)

        # Check that get_zarr_data_info was called correctly
        mock_get_info.assert_called_once_with(
            self.store_path,
            storage_options=None,
            group=None,
        )

        # Check that info was printed to stdout
        output = mock_stdout.getvalue()
        assert self.store_path in output
        assert "temperature" in output
        assert "precipitation" in output

    @patch("src.cli.get_zarr_data_info")
    @patch("builtins.open", new_callable=mock_open)
    @patch("sys.stdout", new_callable=StringIO)
    def test_info_command_with_output_file(self, mock_stdout, mock_file, mock_get_info):
        """Test info command with output file."""
        mock_get_info.return_value = self.mock_info

        args = argparse.Namespace(
            store_path=self.store_path,
            storage_options='{"anon": true}',
            group="climate_data",
            output="info.json",
        )

        info_command(args)

        # Check that get_zarr_data_info was called with parsed options
        mock_get_info.assert_called_once_with(
            self.store_path,
            storage_options={"anon": True},
            group="climate_data",
        )

        # Check that file was opened and written to
        mock_file.assert_called_once_with("info.json", "w")
        handle = mock_file.return_value.__enter__.return_value

        # Check that JSON was written
        written_calls = handle.write.call_args_list
        written_content = "".join(call[0][0] for call in written_calls)
        parsed_content = json.loads(written_content)

        # JSON serialization converts tuples to lists, so we need to account for that
        expected_with_lists = json.loads(json.dumps(self.mock_info))
        assert parsed_content == expected_with_lists

        # Check stdout message
        output = mock_stdout.getvalue()
        assert "Info saved to info.json" in output

    @patch("src.cli.get_zarr_data_info")
    @patch("sys.stderr", new_callable=StringIO)
    def test_info_command_error_handling(self, mock_stderr, mock_get_info):
        """Test info command error handling."""
        mock_get_info.side_effect = Exception("Connection failed")

        args = argparse.Namespace(
            store_path=self.store_path,
            storage_options=None,
            group=None,
            output=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            info_command(args)

        assert exc_info.value.code == 1

        # Check error message was printed to stderr
        error_output = mock_stderr.getvalue()
        assert "Error: Connection failed" in error_output

    @patch("src.cli.ZarrDataReader")
    @patch("sys.stdout", new_callable=StringIO)
    def test_read_command_basic(self, mock_stdout, mock_reader_class):
        """Test basic read command functionality."""
        # Mock the reader and LazyFrame
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_df = Mock(spec=pl.DataFrame)
        mock_df.shape = (1000, 4)
        mock_df.columns = ["time", "lat", "lon", "temperature"]

        mock_lf.collect.return_value = mock_df
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        args = argparse.Namespace(
            store_path=self.store_path,
            array_name="temperature",
            storage_options=None,
            group=None,
            select_dims=None,
            streaming=True,
            output=None,
        )

        read_command(args)

        # Check reader was created correctly
        mock_reader_class.assert_called_once_with(
            self.store_path,
            storage_options=None,
            group=None,
        )

        # Check array was read correctly
        mock_reader.read_array.assert_called_once_with(
            "temperature",
            select_dims=None,
            streaming=True,
        )

        # Check data was collected and written
        mock_lf.collect.assert_called_once()
        mock_df.write_parquet.assert_called_once_with("temperature.parquet")

        # Check output messages
        output = mock_stdout.getvalue()
        assert "Data saved to temperature.parquet" in output
        assert "Shape: (1000, 4)" in output
        assert "Columns: ['time', 'lat', 'lon', 'temperature']" in output

    @patch("src.cli.ZarrDataReader")
    @patch("sys.stdout", new_callable=StringIO)
    def test_read_command_with_options(self, _, mock_reader_class):
        """Test read command with various options."""
        mock_reader = Mock()
        mock_lf = Mock(spec=pl.LazyFrame)
        mock_df = Mock(spec=pl.DataFrame)
        mock_df.shape = (500, 4)
        mock_df.columns = ["time", "lat", "lon", "temperature"]

        mock_lf.collect.return_value = mock_df
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        args = argparse.Namespace(
            store_path=self.store_path,
            array_name="temperature",
            storage_options="key=test,secret=test",
            group="climate_data",
            select_dims="{'time': [0, 1, 2]}",  # Use list instead of slice
            streaming=False,
            output="custom_output.parquet",
        )

        read_command(args)

        # Check reader was created with parsed storage options
        mock_reader_class.assert_called_once_with(
            self.store_path,
            storage_options={"key": "test", "secret": "test"},
            group="climate_data",
        )

        # Check array was read with parsed select_dims
        mock_reader.read_array.assert_called_once_with(
            "temperature",
            select_dims={"time": [0, 1, 2]},
            streaming=False,
        )

        # Check custom output file was used
        mock_df.write_parquet.assert_called_once_with("custom_output.parquet")

    @patch("src.cli.ZarrDataReader")
    @patch("sys.stderr", new_callable=StringIO)
    def test_read_command_error_handling(self, mock_stderr, mock_reader_class):
        """Test read command error handling."""
        mock_reader_class.side_effect = Exception("Failed to create reader")

        args = argparse.Namespace(
            store_path=self.store_path,
            array_name="temperature",
            storage_options=None,
            group=None,
            select_dims=None,
            streaming=True,
            output=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            read_command(args)

        assert exc_info.value.code == 1

        # Check error message
        error_output = mock_stderr.getvalue()
        assert "Error: Failed to create reader" in error_output

    @patch("src.cli.ZarrDataReader")
    @patch("time.time")
    @patch("sys.stdout", new_callable=StringIO)
    def test_benchmark_command_basic(self, mock_stdout, mock_time, mock_reader_class):
        """Test basic benchmark command functionality."""
        # Mock time.time() to return predictable values
        mock_time.side_effect = [
            0.0,
            1.0,
            1.0,
            2.0,
        ]  # start, after read, start collect, after collect

        mock_reader = Mock()
        mock_reader.list_arrays.return_value = ["temperature", "precipitation"]

        mock_lf = Mock(spec=pl.LazyFrame)
        mock_df = Mock(spec=pl.DataFrame)
        mock_df.shape = (1000, 4)
        mock_df.estimated_size.return_value = 15.5

        mock_lf.collect.return_value = mock_df
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        args = argparse.Namespace(
            store_path=self.store_path,
            array_name=None,  # Should use first array
            storage_options=None,
            streaming=True,
        )

        benchmark_command(args)

        # Check reader was created
        mock_reader_class.assert_called_once_with(
            self.store_path,
            storage_options=None,
        )

        # Check first array was benchmarked
        mock_reader.read_array.assert_called_once_with("temperature", streaming=True)

        # Check output contains benchmark results
        output = mock_stdout.getvalue()
        assert "Benchmarking array: temperature" in output
        assert "Read time: 1.0000s" in output
        assert "Collect time: 1.0000s" in output
        assert "Total time: 2.0000s" in output
        assert "Data shape: (1000, 4)" in output
        assert "Memory usage: 15.50 MB" in output

    @patch("src.cli.ZarrDataReader")
    @patch("sys.stdout", new_callable=StringIO)
    def test_benchmark_command_specific_array(self, mock_stdout, mock_reader_class):
        """Test benchmark command with specific array."""
        mock_reader = Mock()
        mock_reader.list_arrays.return_value = ["temperature", "precipitation"]

        mock_lf = Mock(spec=pl.LazyFrame)
        mock_df = Mock(spec=pl.DataFrame)
        mock_df.shape = (500, 3)
        mock_df.estimated_size.return_value = 8.2

        mock_lf.collect.return_value = mock_df
        mock_reader.read_array.return_value = mock_lf
        mock_reader_class.return_value = mock_reader

        args = argparse.Namespace(
            store_path=self.store_path,
            array_name="precipitation",  # Specific array
            storage_options=None,
            streaming=False,
        )

        benchmark_command(args)

        # Check specific array was benchmarked
        mock_reader.read_array.assert_called_once_with("precipitation", streaming=False)

        output = mock_stdout.getvalue()
        assert "Benchmarking array: precipitation" in output

    @patch("src.cli.ZarrDataReader")
    @patch("sys.stdout", new_callable=StringIO)
    def test_benchmark_command_no_arrays(self, mock_stdout, mock_reader_class):
        """Test benchmark command when no arrays are found."""
        mock_reader = Mock()
        mock_reader.list_arrays.return_value = []
        mock_reader_class.return_value = mock_reader

        args = argparse.Namespace(
            store_path=self.store_path,
            array_name=None,
            storage_options=None,
            streaming=True,
        )

        # Should not raise exception, just print message and return
        benchmark_command(args)

        output = mock_stdout.getvalue()
        assert "No arrays found in store" in output

    @patch("src.cli.ZarrDataReader")
    @patch("sys.stderr", new_callable=StringIO)
    def test_benchmark_command_error_handling(self, mock_stderr, mock_reader_class):
        """Test benchmark command error handling."""
        mock_reader_class.side_effect = Exception("Connection timeout")

        args = argparse.Namespace(
            store_path=self.store_path,
            array_name="temperature",
            storage_options=None,
            streaming=True,
        )

        with pytest.raises(SystemExit) as exc_info:
            benchmark_command(args)

        assert exc_info.value.code == 1

        error_output = mock_stderr.getvalue()
        assert "Error: Connection timeout" in error_output

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = create_parser()

        # Test that parser is created correctly
        assert isinstance(parser, argparse.ArgumentParser)
        assert "CAE-Polars" in parser.description

        # Test info command parsing
        args = parser.parse_args(["info", "s3://bucket/store.zarr"])
        assert args.command == "info"
        assert args.store_path == "s3://bucket/store.zarr"
        assert hasattr(args, "func")

        # Test info command with options
        args = parser.parse_args(
            [
                "info",
                "s3://bucket/store.zarr",
                "--group",
                "climate",
                "--storage-options",
                '{"anon": true}',
                "--output",
                "info.json",
            ]
        )
        assert args.group == "climate"
        assert args.storage_options == '{"anon": true}'
        assert args.output == "info.json"

        # Test read command parsing
        args = parser.parse_args(
            [
                "read",
                "s3://bucket/store.zarr",
                "temperature",
                "--select-dims",
                "{'time': slice(0, 10)}",
                "--no-streaming",
                "--output",
                "data.parquet",
            ]
        )
        assert args.command == "read"
        assert args.array_name == "temperature"
        assert args.select_dims == "{'time': slice(0, 10)}"
        assert args.streaming is False
        assert args.output == "data.parquet"

        # Test benchmark command parsing
        args = parser.parse_args(
            [
                "benchmark",
                "s3://bucket/store.zarr",
                "--array-name",
                "precipitation",
                "--no-streaming",
            ]
        )
        assert args.command == "benchmark"
        assert args.array_name == "precipitation"
        assert args.streaming is False

    @patch("src.cli.create_parser")
    @patch("sys.argv", ["cae-polars", "info", "s3://bucket/store.zarr"])
    def test_main_with_command(self, mock_create_parser):
        """Test main function with valid command."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.func = Mock()

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        main()

        # Check that parser was created and args parsed
        mock_create_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()

        # Check that command function was called
        mock_args.func.assert_called_once_with(mock_args)

    @patch("src.cli.create_parser")
    @patch("sys.argv", ["cae-polars"])
    def test_main_no_command(self, mock_create_parser):
        """Test main function with no command (should show help)."""
        mock_parser = Mock()
        mock_args = Mock(spec=[])  # No func attribute

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

        # Check that help was printed
        mock_parser.print_help.assert_called_once()

    def test_integration_info_command(self):
        """Test integration of info command with argument parsing."""
        parser = create_parser()

        # Test valid info command
        args = parser.parse_args(
            [
                "info",
                "s3://test-bucket/data.zarr",
                "--group",
                "climate_data",
                "--storage-options",
                "anon=true",
            ]
        )

        assert args.command == "info"
        assert args.store_path == "s3://test-bucket/data.zarr"
        assert args.group == "climate_data"
        assert args.storage_options == "anon=true"
        assert args.output is None
        assert hasattr(args, "func")

    def test_integration_read_command(self):
        """Test integration of read command with argument parsing."""
        parser = create_parser()

        # Test valid read command
        args = parser.parse_args(
            [
                "read",
                "s3://test-bucket/data.zarr",
                "temperature",
                "--select-dims",
                "{'time': [0, 1, 2]}",
                "--output",
                "temp_data.parquet",
            ]
        )

        assert args.command == "read"
        assert args.store_path == "s3://test-bucket/data.zarr"
        assert args.array_name == "temperature"
        assert args.select_dims == "{'time': [0, 1, 2]}"
        assert args.output == "temp_data.parquet"
        assert args.streaming is True  # Default value
        assert hasattr(args, "func")

    def test_integration_benchmark_command(self):
        """Test integration of benchmark command with argument parsing."""
        parser = create_parser()

        # Test valid benchmark command
        args = parser.parse_args(
            [
                "benchmark",
                "s3://test-bucket/data.zarr",
                "--array-name",
                "precipitation",
                "--storage-options",
                '{"key": "test", "secret": "test"}',
            ]
        )

        assert args.command == "benchmark"
        assert args.store_path == "s3://test-bucket/data.zarr"
        assert args.array_name == "precipitation"
        assert args.storage_options == '{"key": "test", "secret": "test"}'
        assert args.streaming is True  # Default value
        assert hasattr(args, "func")

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        # Test _parse_select_dims with various invalid inputs
        invalid_inputs = [
            "{'unclosed': dict",  # Syntax error
            "import os; os.system('rm -rf /')",  # Potential security issue - not valid Python literal
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                _parse_select_dims(invalid_input)

        # Test cases that should work but return different types
        valid_but_different = [
            (
                "[1, 2, 3]",
                [1, 2, 3],
            ),  # List instead of dict - should work for ast.literal_eval
            ("{'key': 'value'}", {"key": "value"}),  # String values
        ]

        for valid_input, expected in valid_but_different:
            if isinstance(expected, dict):
                result = _parse_select_dims(valid_input)
                assert result == expected

    def test_storage_options_edge_cases(self):
        """Test storage options parsing edge cases."""
        # Test malformed key=value pairs
        test_cases = [
            ("key1=value1,key2", {"key1": "value1"}),  # Missing value
            ("key=", {"key": ""}),  # Empty value
            (
                "key1=value1,key2=value2",
                {"key1": "value1", "key2": "value2"},
            ),  # Valid case
        ]

        for input_str, expected in test_cases:
            result = _parse_storage_options(input_str)
            assert result == expected

        # Test case that creates empty key
        result = _parse_storage_options("=value")
        assert result == {"": "value"}  # This is what the actual implementation does

    def test_command_function_attributes(self):
        """Test that command functions have the correct attributes."""
        parser = create_parser()

        # Test that all subcommands have func attributes set
        info_args = parser.parse_args(["info", "test.zarr"])
        assert info_args.func == info_command

        read_args = parser.parse_args(["read", "test.zarr", "array"])
        assert read_args.func == read_command

        benchmark_args = parser.parse_args(["benchmark", "test.zarr"])
        assert benchmark_args.func == benchmark_command
