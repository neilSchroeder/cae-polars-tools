"""
Command-line interface for CAE-Polars.

Provides utilities for working with Zarr files and benchmarking performance.
"""

import argparse
import json
import sys
from typing import Optional

from .data_access import ZarrDataReader, get_zarr_data_info


def info_command(args) -> None:
    """Get information about a Zarr store."""
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
    """Read a Zarr array and save to parquet."""
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
    """Run a simple benchmark test."""
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
    """Parse storage options from command line."""
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
    """Parse dimension selection from command line."""
    import ast

    try:
        return ast.literal_eval(select_dims_str)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid select_dims format: {select_dims_str}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CAE-Polars-Tools: High-performance Zarr I/O for Polars",
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
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
