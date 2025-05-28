"""
Command-line interface for CAE-Polars.

This module provides a comprehensive command-line interface for working with
Zarr files, including data inspection, reading, and performance benchmarking.
The CLI is designed to be user-friendly while exposing the full power of the
CAE-Polars library for batch processing and automation workflows.

The CLI supports three main commands:
- info: Get detailed information about Zarr stores and arrays
- read: Read Zarr arrays and export to various formats
- benchmark: Performance testing and profiling

Examples
--------
Get information about a Zarr store:
    $ cae-polars info s3://bucket/data.zarr --storage-options '{"anon": true}'

Read an array and save to Parquet:
    $ cae-polars read s3://bucket/data.zarr temperature --output temp.parquet

Benchmark performance:
    $ cae-polars benchmark s3://bucket/data.zarr --array-name temperature

Notes
-----
All commands support S3 storage with flexible authentication options.
Storage options can be provided as JSON or key=value pairs.
Dimension selection uses Python dictionary syntax for flexibility.
"""

import argparse
import json
import sys
from typing import Optional

import polars as pl

from .data_access import ZarrDataReader, get_zarr_data_info


def info_command(args) -> None:
    """
    Get comprehensive information about a Zarr store and its arrays.

    This command provides detailed metadata about Zarr stores, including
    information about available arrays, their shapes, data types, chunking
    schemes, and attributes. The output can be displayed to stdout or saved
    to a JSON file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - store_path : str
            Path to the Zarr store (local or S3)
        - storage_options : str, optional
            Storage authentication options as JSON or key=value pairs
        - group : str, optional
            Specific group within the Zarr store to examine
        - output : str, optional
            Output file path for saving the information as JSON

    Raises
    ------
    SystemExit
        If an error occurs during information retrieval, exits with code 1

    Examples
    --------
    Get info about all arrays in a store:
        $ cae-polars info s3://bucket/data.zarr

    Get info about a specific group and save to file:
        $ cae-polars info s3://bucket/data.zarr --group climate --output info.json

    Get info with S3 credentials:
        $ cae-polars info s3://bucket/data.zarr --storage-options '{"key":"ACCESS_KEY"}'
    """
    try:
        info = get_zarr_data_info(
            args.store_path,
            storage_options=_parse_storage_options(args.storage_options),
            group=args.group,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(info, f, indent=2, default=str)
            print(f"Info saved to {args.output}")
        else:
            print(json.dumps(info, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def read_command(args) -> None:
    """
    Read a Zarr array and export it to Parquet format.

    This command reads a specific array from a Zarr store and saves it as a
    Parquet file. It supports dimension selection for reading subsets of data
    and can handle large datasets through streaming.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - store_path : str
            Path to the Zarr store (local or S3)
        - array_name : str
            Name of the array to read
        - storage_options : str, optional
            Storage authentication options as JSON or key=value pairs
        - group : str, optional
            Specific group within the Zarr store
        - select_dims : str, optional
            Dimension selection as Python dict string
        - streaming : bool
            Whether to use streaming mode (default True)
        - output : str, optional
            Output file path (defaults to {array_name}.parquet)

    Raises
    ------
    SystemExit
        If an error occurs during reading or writing, exits with code 1

    Examples
    --------
    Read an entire array:
        $ cae-polars read s3://bucket/data.zarr temperature

    Read with dimension selection:
        $ cae-polars read s3://bucket/data.zarr temperature \\
            --select-dims "{'time': [0, 1, 2], 'lat': slice(10, 20)}"

    Save to specific file:
        $ cae-polars read s3://bucket/data.zarr temperature --output my_data.parquet
    """
    try:
        reader = ZarrDataReader(
            args.store_path,
            storage_options=_parse_storage_options(args.storage_options),
            group=args.group,
        )

        select_dims = _parse_select_dims(args.select_dims) if args.select_dims else None

        lf = reader.read_array(
            args.array_name,
            select_dims=select_dims,
            streaming=args.streaming,
        )

        # Collect and save
        df = lf.collect()
        output_path = args.output or f"{args.array_name}.parquet"
        df.write_parquet(output_path)

        print(f"Data saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def benchmark_command(args) -> None:
    """
    Run a comprehensive benchmark test on Zarr array reading performance.

    This command performs timing measurements for different stages of the
    data reading pipeline, including array opening, lazy frame creation,
    and data collection. It provides insights into performance characteristics
    for optimization purposes.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - store_path : str
            Path to the Zarr store (local or S3)
        - array_name : str, optional
            Name of the array to benchmark (uses first array if not specified)
        - storage_options : str, optional
            Storage authentication options as JSON or key=value pairs
        - streaming : bool
            Whether to use streaming mode (default True)

    Raises
    ------
    SystemExit
        If an error occurs during benchmarking, exits with code 1

    Examples
    --------
    Benchmark default array:
        $ cae-polars benchmark s3://bucket/data.zarr

    Benchmark specific array:
        $ cae-polars benchmark s3://bucket/data.zarr --array-name temperature

    Benchmark without streaming:
        $ cae-polars benchmark s3://bucket/data.zarr --no-streaming

    Notes
    -----
    The benchmark measures:
    - Read time: Time to create LazyFrame from Zarr array
    - Collect time: Time to materialize LazyFrame into DataFrame
    - Total time: Combined read and collect time
    - Memory usage: Estimated size of the resulting DataFrame
    """
    try:
        import time

        reader = ZarrDataReader(
            args.store_path,
            storage_options=_parse_storage_options(args.storage_options),
        )

        arrays = reader.list_arrays()
        if not arrays:
            print("No arrays found in store")
            return

        array_name = args.array_name or arrays[0]
        print(f"Benchmarking array: {array_name}")

        # Time the read operation
        start_time = time.time()
        lf = reader.read_array(array_name, streaming=args.streaming)
        read_time = time.time() - start_time

        # Time the collection
        start_time = time.time()
        df = lf.collect()
        collect_time = time.time() - start_time

        total_time = read_time + collect_time

        print(f"Read time: {read_time:.4f}s")
        print(f"Collect time: {collect_time:.4f}s")
        print(f"Total time: {total_time:.4f}s")
        print(f"Data shape: {df.shape}")
        print(f"Memory usage: {df.estimated_size('mb'):.2f} MB")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_storage_options(storage_options_str: Optional[str]) -> Optional[dict]:
    """
    Parse storage options from command line string.

    Supports both JSON format and simple key=value pair format for
    specifying S3 credentials and other storage options.

    Parameters
    ----------
    storage_options_str : str or None
        Storage options as either:
        - JSON string: '{"key": "value", "anon": true}'
        - Key=value pairs: "key=access_key,secret=secret_key,region=us-west-2"
        - None or empty string

    Returns
    -------
    dict or None
        Parsed storage options dictionary, or None if input is empty

    Examples
    --------
    >>> _parse_storage_options('{"anon": true}')
    {'anon': True}

    >>> _parse_storage_options('key=access,secret=secret')
    {'key': 'access', 'secret': 'secret'}

    >>> _parse_storage_options(None)
    None
    """
    if not storage_options_str:
        return None

    try:
        return json.loads(storage_options_str)
    except json.JSONDecodeError:
        # Try simple key=value format
        options = {}
        for pair in storage_options_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                options[key.strip()] = value.strip()
        return options if options else None


def _parse_select_dims(select_dims_str: str) -> dict:
    """
    Parse dimension selection from command line string.

    Uses ast.literal_eval to safely parse Python dictionary strings
    containing dimension selection criteria.

    Parameters
    ----------
    select_dims_str : str
        Dimension selection as Python dict string, e.g.:
        "{'time': [0, 1, 2], 'lat': slice(10, 20)}"

    Returns
    -------
    dict
        Parsed dimension selection dictionary

    Raises
    ------
    ValueError
        If the string cannot be parsed as a valid Python dictionary

    Examples
    --------
    >>> _parse_select_dims("{'time': 5}")
    {'time': 5}

    >>> _parse_select_dims("{'time': [0, 1, 2], 'lat': [10, 20]}")
    {'time': [0, 1, 2], 'lat': [10, 20]}

    Notes
    -----
    Only supports literals that can be parsed by ast.literal_eval.
    slice() objects cannot be parsed and should be avoided in the input.
    """
    import ast

    try:
        return ast.literal_eval(select_dims_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid select_dims format: {select_dims_str}") from e


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.

    Constructs a comprehensive argument parser with subcommands for different
    operations (info, read, benchmark) and their respective arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all subcommands and options

    Examples
    --------
    >>> parser = create_parser()
    >>> args = parser.parse_args(['info', 's3://bucket/data.zarr'])
    >>> args.command
    'info'
    """
    parser = argparse.ArgumentParser(
        description="CAE-Polars: High-performance Zarr I/O for Polars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Get information about a Zarr store"
    )
    info_parser.add_argument("store_path", help="Path to Zarr store")
    info_parser.add_argument("--group", help="Group within the store")
    info_parser.add_argument(
        "--storage-options", help="Storage options as JSON or key=value,key=value"
    )
    info_parser.add_argument("--output", "-o", help="Output file for info")
    info_parser.set_defaults(func=info_command)

    # Read command
    read_parser = subparsers.add_parser(
        "read", help="Read a Zarr array and save to Parquet"
    )
    read_parser.add_argument("store_path", help="Path to Zarr store")
    read_parser.add_argument("array_name", help="Name of array to read")
    read_parser.add_argument("--group", help="Group within the store")
    read_parser.add_argument(
        "--storage-options", help="Storage options as JSON or key=value,key=value"
    )
    read_parser.add_argument(
        "--select-dims", help="Dimension selection as Python dict string"
    )
    read_parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming mode",
    )
    read_parser.add_argument("--output", "-o", help="Output file path")
    read_parser.set_defaults(func=read_command)

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run a simple benchmark test"
    )
    benchmark_parser.add_argument("store_path", help="Path to Zarr store")
    benchmark_parser.add_argument("--array-name", help="Array to benchmark")
    benchmark_parser.add_argument(
        "--storage-options", help="Storage options as JSON or key=value,key=value"
    )
    benchmark_parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming mode",
    )
    benchmark_parser.set_defaults(func=benchmark_command)

    return parser


def main() -> None:
    """
    Main CLI entry point.

    Parses command-line arguments and dispatches to the appropriate
    command function. Provides help message if no command is specified.

    Raises
    ------
    SystemExit
        Exits with code 1 if no command is provided, or with the code
        returned by the executed command function.

    Examples
    --------
    Called automatically when the module is run as a script:
        $ python -m cae_polars info s3://bucket/data.zarr
        $ cae-polars read s3://bucket/data.zarr temperature
    """
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
