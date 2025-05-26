#!/usr/bin/env python3
"""
Simplified Performance Comparison: Polars vs XArray for Climate Data

This script provides a focused comparison of Polars vs XArray for basic
climate data operations using our refactored architecture.
"""

import gc
import sys
import time
from typing import Any, Dict

# Add the project to the path
sys.path.append("/home/nschroed/Documents/work/climakitae")

import numpy as np
import polars as pl
import xarray as xr

# Import our refactored modules
from climakitae.new_core import ClimateDataReader


class SimplePerformanceTest:
    """Simple performance comparison between Polars and XArray approaches."""

    def __init__(self, store_path: str):
        self.store_path = store_path

    def time_operation(self, description: str, func, *args, **kwargs):
        """Time an operation and return the result and elapsed time."""
        print(f"  {description}...", end=" ", flush=True)

        gc.collect()  # Clean up before timing
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            print(f"{elapsed:.3f}s ✓")
            return result, elapsed, None
        except Exception as e:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            print(f"{elapsed:.3f}s ✗ ({str(e)[:50]}...)")
            return None, elapsed, str(e)

    def test_polars_approach(self):
        """Test performance using our Polars-based approach."""
        print("\n🚀 Polars Approach:")
        print("-" * 40)

        results = {}

        # Initialize reader
        reader, init_time, error = self.time_operation(
            "Initialize ClimateDataReader", lambda: ClimateDataReader(self.store_path)
        )
        results["init_time"] = init_time
        results["init_error"] = error

        if reader is None:
            return results

        # List arrays
        arrays, list_time, error = self.time_operation(
            "List arrays", reader.list_arrays
        )
        results["list_time"] = list_time
        results["list_error"] = error

        if not arrays:
            return results

        array_name = arrays[0]
        print(f"    Using array: {array_name}")

        # Get array info
        info, info_time, error = self.time_operation(
            "Get array info", reader.get_array_info, array_name
        )
        results["info_time"] = info_time
        results["info_error"] = error

        if info:
            print(f"    Array shape: {info['shape']}")
            print(f"    Array dtype: {info['dtype']}")

        # Test data loading with different sizes
        test_sizes = [
            ("small", {"time": slice(0, 6), "lat": slice(0, 50), "lon": slice(0, 50)}),
            (
                "medium",
                {"time": slice(0, 12), "lat": slice(0, 100), "lon": slice(0, 100)},
            ),
            (
                "large",
                {"time": slice(0, 24), "lat": slice(0, 150), "lon": slice(0, 150)},
            ),
        ]

        for size_name, dims in test_sizes:
            print(f"  Testing {size_name} subset:")

            # Load data
            lf, load_time, error = self.time_operation(
                f"    Load {size_name} data",
                reader.read_array,
                array_name,
                dims,
                False,  # Don't use streaming for timing accuracy
            )
            results[f"{size_name}_load_time"] = load_time
            results[f"{size_name}_load_error"] = error

            if lf is None:
                continue

            # Collect to DataFrame
            df, collect_time, error = self.time_operation(
                f"    Collect {size_name} data", lf.collect
            )
            results[f"{size_name}_collect_time"] = collect_time
            results[f"{size_name}_collect_error"] = error

            if df is None:
                continue

            print(f"    Data shape: {df.shape}")

            # Basic operations
            mean_val, mean_time, error = self.time_operation(
                f"    Calculate {size_name} mean",
                lambda: df.select(pl.col("value").mean()).item(),
            )
            results[f"{size_name}_mean_time"] = mean_time
            results[f"{size_name}_mean_error"] = error

            max_val, max_time, error = self.time_operation(
                f"    Calculate {size_name} max",
                lambda: df.select(pl.col("value").max()).item(),
            )
            results[f"{size_name}_max_time"] = max_time
            results[f"{size_name}_max_error"] = error

            # Filtering operation
            if mean_val is not None:
                filtered_df, filter_time, error = self.time_operation(
                    f"    Filter {size_name} data (>mean)",
                    lambda: df.filter(pl.col("value") > mean_val),
                )
                results[f"{size_name}_filter_time"] = filter_time
                results[f"{size_name}_filter_error"] = error

                if filtered_df is not None:
                    print(f"    Filtered rows: {filtered_df.height}/{df.height}")

        return results

    def test_xarray_approach(self):
        """Test performance using traditional XArray approach."""
        print("\n📊 XArray Approach:")
        print("-" * 40)

        results = {}

        try:
            import s3fs
            import zarr
        except ImportError as e:
            print(f"  Cannot test XArray approach: {e}")
            results["import_error"] = str(e)
            return results

        # Initialize connection
        fs, init_time, error = self.time_operation(
            "Initialize S3 connection", lambda: s3fs.S3FileSystem(anon=True)
        )
        results["init_time"] = init_time
        results["init_error"] = error

        if fs is None:
            return results

        # Open zarr store
        def open_zarr():
            mapper = fs.get_mapper(self.store_path)
            try:
                return zarr.open_consolidated(mapper, mode="r")
            except Exception:
                return zarr.open(mapper, mode="r")

        group, open_time, error = self.time_operation("Open zarr store", open_zarr)
        results["open_time"] = open_time
        results["open_error"] = error

        if group is None:
            return results

        # List arrays
        def list_arrays():
            return [name for name in group.keys() if hasattr(group[name], "shape")]

        arrays, list_time, error = self.time_operation("List arrays", list_arrays)
        results["list_time"] = list_time
        results["list_error"] = error

        if not arrays:
            return results

        array_name = arrays[0]
        print(f"    Using array: {array_name}")

        # Get array
        zarr_array = group[array_name]
        print(f"    Array shape: {zarr_array.shape}")
        print(f"    Array dtype: {zarr_array.dtype}")

        # Test data loading with different sizes
        test_sizes = [
            ("small", (slice(0, 6), slice(0, 50), slice(0, 50))),
            ("medium", (slice(0, 12), slice(0, 100), slice(0, 100))),
            ("large", (slice(0, 24), slice(0, 150), slice(0, 150))),
        ]

        for size_name, slices in test_sizes:
            print(f"  Testing {size_name} subset:")

            # Load data
            data, load_time, error = self.time_operation(
                f"    Load {size_name} data", lambda: zarr_array[slices]
            )
            results[f"{size_name}_load_time"] = load_time
            results[f"{size_name}_load_error"] = error

            if data is None:
                continue

            print(f"    Data shape: {data.shape}")

            # Basic operations
            mean_val, mean_time, error = self.time_operation(
                f"    Calculate {size_name} mean", lambda: np.nanmean(data)
            )
            results[f"{size_name}_mean_time"] = mean_time
            results[f"{size_name}_mean_error"] = error

            max_val, max_time, error = self.time_operation(
                f"    Calculate {size_name} max", lambda: np.nanmax(data)
            )
            results[f"{size_name}_max_time"] = max_time
            results[f"{size_name}_max_error"] = error

            # Filtering operation
            if mean_val is not None and not np.isnan(mean_val):
                filtered_data, filter_time, error = self.time_operation(
                    f"    Filter {size_name} data (>mean)",
                    lambda: data[data > mean_val],
                )
                results[f"{size_name}_filter_time"] = filter_time
                results[f"{size_name}_filter_error"] = error

                if filtered_data is not None:
                    print(f"    Filtered elements: {filtered_data.size}/{data.size}")

        return results

    def compare_results(self, polars_results: Dict, xarray_results: Dict):
        """Compare and summarize the results."""
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)

        # Initialize
        print("\n🔧 Initialization:")
        p_init = polars_results.get("init_time", 0)
        x_init = xarray_results.get("init_time", 0) + xarray_results.get("open_time", 0)
        if p_init > 0 and x_init > 0:
            speedup = x_init / p_init
            print(f"  Polars: {p_init:.3f}s")
            print(f"  XArray: {x_init:.3f}s")
            print(
                f"  Speedup: {speedup:.2f}x ({'Polars faster' if speedup > 1 else 'XArray faster'})"
            )

        # Compare each size
        sizes = ["small", "medium", "large"]
        operations = ["load", "mean", "max", "filter"]

        for size in sizes:
            print(f"\n📏 {size.title()} Dataset Operations:")

            for op in operations:
                p_key = f"{size}_{op}_time"
                x_key = f"{size}_{op}_time"

                p_time = polars_results.get(p_key, 0)
                x_time = xarray_results.get(x_key, 0)

                if p_time > 0 and x_time > 0:
                    speedup = x_time / p_time
                    print(
                        f"  {op.title()}: Polars {p_time:.3f}s vs XArray {x_time:.3f}s "
                        f"({speedup:.2f}x)"
                    )

    def run_comparison(self):
        """Run the full performance comparison."""
        print("Climate Data Performance Test")
        print("Polars vs XArray")
        print("=" * 50)

        # Test Polars approach
        polars_results = self.test_polars_approach()

        # Test XArray approach
        xarray_results = self.test_xarray_approach()

        # Compare results
        self.compare_results(polars_results, xarray_results)

        # Save results
        print(f"\n💾 Saving results...")
        import json

        all_results = {
            "polars": polars_results,
            "xarray": xarray_results,
        }

        with open("simple_benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print("✅ Performance test completed!")
        print("📄 Results saved to simple_benchmark_results.json")


def main():
    """Main function to run the performance test."""

    # Use the same test dataset
    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    # Create and run test
    test = SimplePerformanceTest(store_path)
    test.run_comparison()


if __name__ == "__main__":
    main()
