#!/usr/bin/env python3
"""
Performance Comparison: Polars vs XArray for Climate Data Processing

This script benchmarks the performance of Polars-based climate data processing
against traditional XArray approaches for various operations on Zarr datasets.
"""

import gc
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

# Add the project to the path
sys.path.append("/home/nschroed/Documents/work/climakitae")

import numpy as np
import polars as pl
import xarray as xr

# Import our refactored modules
from climakitae.new_core import ClimateDataReader, scan_climate_data

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark for climate data operations.

    This class compares Polars and XArray performance across various
    climate data processing tasks.
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.results = {}

    def time_operation(self, operation_name: str, func, *args, **kwargs) -> float:
        """Time a single operation and return execution time in seconds."""
        print(f"  Running {operation_name}...", end=" ", flush=True)

        # Force garbage collection before timing
        gc.collect()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        print(f"{elapsed:.3f}s")

        return elapsed, result

    def benchmark_data_loading(self) -> Dict[str, Any]:
        """Benchmark data loading performance."""
        print("\n" + "=" * 60)
        print("BENCHMARK 1: Data Loading Performance")
        print("=" * 60)

        results = {}

        # Test with different data sizes
        test_configs = [
            {
                "name": "small_subset",
                "dims": {
                    "time": slice(0, 12),
                    "lat": slice(0, 50),
                    "lon": slice(0, 50),
                },
                "description": "Small subset (12 months, 50x50 grid)",
            },
            {
                "name": "medium_subset",
                "dims": {
                    "time": slice(0, 60),
                    "lat": slice(0, 100),
                    "lon": slice(0, 100),
                },
                "description": "Medium subset (5 years, 100x100 grid)",
            },
            {
                "name": "large_subset",
                "dims": {
                    "time": slice(0, 120),
                    "lat": slice(0, 200),
                    "lon": slice(0, 200),
                },
                "description": "Large subset (10 years, 200x200 grid)",
            },
        ]

        for config in test_configs:
            print(f"\nTesting {config['description']}")
            print("-" * 40)

            config_results = {}

            # Polars approach
            print("Polars approach:")
            try:
                elapsed, polars_data = self.time_operation(
                    "Data loading + conversion", self._load_with_polars, config["dims"]
                )
                config_results["polars_load_time"] = elapsed
                # Memory estimation for LazyFrame
                try:
                    config_results["polars_memory_mb"] = (
                        polars_data.collect().estimated_size("mb")
                    )
                except AttributeError:
                    config_results["polars_memory_mb"] = "N/A"
                config_results["polars_rows"] = polars_data.height

                # Test basic operation
                elapsed, _ = self.time_operation(
                    "Mean calculation",
                    lambda: polars_data.select(pl.col("value").mean()).collect(),
                )
                config_results["polars_mean_time"] = elapsed

            except Exception as e:
                print(f"    Polars failed: {e}")
                config_results["polars_error"] = str(e)

            # XArray approach
            print("XArray approach:")
            try:
                elapsed, xarray_data = self.time_operation(
                    "Data loading", self._load_with_xarray, config["dims"]
                )
                config_results["xarray_load_time"] = elapsed
                config_results["xarray_memory_mb"] = xarray_data.nbytes / 1024 / 1024
                config_results["xarray_size"] = xarray_data.size

                # Test basic operation
                elapsed, _ = self.time_operation(
                    "Mean calculation", lambda: xarray_data.mean().compute()
                )
                config_results["xarray_mean_time"] = elapsed

                # Clean up xarray data
                xarray_data.close()
                del xarray_data

            except Exception as e:
                print(f"    XArray failed: {e}")
                config_results["xarray_error"] = str(e)

            results[config["name"]] = config_results

            # Clean up
            gc.collect()

        return results

    def benchmark_aggregation_operations(self) -> Dict[str, Any]:
        """Benchmark aggregation operations performance."""
        print("\n" + "=" * 60)
        print("BENCHMARK 2: Aggregation Operations")
        print("=" * 60)

        # Use medium-sized dataset for aggregation tests
        dims = {"time": slice(0, 60), "lat": slice(0, 100), "lon": slice(0, 100)}

        print("Loading test data for aggregation benchmarks...")
        polars_data = self._load_with_polars(dims)
        xarray_data = self._load_with_xarray(dims)

        results = {}

        # Test various aggregation operations
        operations = [
            (
                "mean",
                lambda df: df.select(pl.col("value").mean()),
                lambda da: da.mean(),
            ),
            ("max", lambda df: df.select(pl.col("value").max()), lambda da: da.max()),
            ("min", lambda df: df.select(pl.col("value").min()), lambda da: da.min()),
            ("std", lambda df: df.select(pl.col("value").std()), lambda da: da.std()),
            (
                "quantile_50",
                lambda df: df.select(pl.col("value").quantile(0.5)),
                lambda da: da.quantile(0.5),
            ),
            (
                "quantile_95",
                lambda df: df.select(pl.col("value").quantile(0.95)),
                lambda da: da.quantile(0.95),
            ),
        ]

        for op_name, polars_func, xarray_func in operations:
            print(f"\nTesting {op_name} operation:")
            print("-" * 30)

            op_results = {}

            # Polars
            try:
                elapsed, _ = self.time_operation(
                    f"Polars {op_name}", lambda: polars_func(polars_data).collect()
                )
                op_results["polars_time"] = elapsed
            except Exception as e:
                op_results["polars_error"] = str(e)

            # XArray
            try:
                elapsed, _ = self.time_operation(
                    f"XArray {op_name}", lambda: xarray_func(xarray_data).compute()
                )
                op_results["xarray_time"] = elapsed
            except Exception as e:
                op_results["xarray_error"] = str(e)

            results[op_name] = op_results

        # Clean up
        xarray_data.close()
        del polars_data, xarray_data
        gc.collect()

        return results

    def benchmark_groupby_operations(self) -> Dict[str, Any]:
        """Benchmark group-by operations performance."""
        print("\n" + "=" * 60)
        print("BENCHMARK 3: Group-by Operations")
        print("=" * 60)

        # Use medium-sized dataset
        dims = {"time": slice(0, 60), "lat": slice(0, 100), "lon": slice(0, 100)}

        print("Loading test data for group-by benchmarks...")
        polars_data = self._load_with_polars(dims)
        xarray_data = self._load_with_xarray(dims)

        results = {}

        # Test group-by operations
        operations = [
            ("time_mean", "Group by time and calculate mean"),
            ("spatial_mean", "Group by lat/lon and calculate mean"),
        ]

        for op_name, description in operations:
            print(f"\nTesting {description}:")
            print("-" * 40)

            op_results = {}

            if op_name == "time_mean":
                # Polars: group by time coordinate
                try:
                    elapsed, _ = self.time_operation(
                        "Polars time groupby",
                        lambda: polars_data.group_by("time")
                        .agg(pl.col("value").mean())
                        .collect(),
                    )
                    op_results["polars_time"] = elapsed
                except Exception as e:
                    op_results["polars_error"] = str(e)

                # XArray: group by time
                try:
                    elapsed, _ = self.time_operation(
                        "XArray time groupby",
                        lambda: xarray_data.groupby("time").mean().compute(),
                    )
                    op_results["xarray_time"] = elapsed
                except Exception as e:
                    op_results["xarray_error"] = str(e)

            elif op_name == "spatial_mean":
                # Polars: group by spatial coordinates
                try:
                    elapsed, _ = self.time_operation(
                        "Polars spatial groupby",
                        lambda: polars_data.group_by(["lat", "lon"])
                        .agg(pl.col("value").mean())
                        .collect(),
                    )
                    op_results["polars_time"] = elapsed
                except Exception as e:
                    op_results["polars_error"] = str(e)

                # XArray: spatial mean
                try:
                    elapsed, _ = self.time_operation(
                        "XArray spatial mean",
                        lambda: xarray_data.mean(dim=["lat", "lon"]).compute(),
                    )
                    op_results["xarray_time"] = elapsed
                except Exception as e:
                    op_results["xarray_error"] = str(e)

            results[op_name] = op_results

        # Clean up
        xarray_data.close()
        del polars_data, xarray_data
        gc.collect()

        return results

    def benchmark_filtering_operations(self) -> Dict[str, Any]:
        """Benchmark filtering operations performance."""
        print("\n" + "=" * 60)
        print("BENCHMARK 4: Filtering Operations")
        print("=" * 60)

        # Use medium-sized dataset
        dims = {"time": slice(0, 60), "lat": slice(0, 100), "lon": slice(0, 100)}

        print("Loading test data for filtering benchmarks...")
        polars_data = self._load_with_polars(dims)
        xarray_data = self._load_with_xarray(dims)

        results = {}

        # Get some statistics to define reasonable filters
        polars_stats = polars_data.select(
            [pl.col("value").mean().alias("mean"), pl.col("value").std().alias("std")]
        ).collect()

        mean_val = polars_stats["mean"][0]
        std_val = polars_stats["std"][0]

        # Test filtering operations
        filters = [
            (
                "above_mean",
                f"Values above mean ({mean_val:.2f})",
                lambda x: x > mean_val,
            ),
            (
                "extreme_values",
                f"Values above mean + 2*std ({mean_val + 2*std_val:.2f})",
                lambda x: x > (mean_val + 2 * std_val),
            ),
            (
                "range_filter",
                f"Values between mean ± std",
                lambda x: (x >= (mean_val - std_val)) & (x <= (mean_val + std_val)),
            ),
        ]

        for filter_name, description, filter_func in filters:
            print(f"\nTesting {description}:")
            print("-" * 50)

            filter_results = {}

            # Polars
            try:
                elapsed, filtered_data = self.time_operation(
                    "Polars filtering",
                    lambda: polars_data.filter(filter_func(pl.col("value"))).collect(),
                )
                filter_results["polars_time"] = elapsed
                filter_results["polars_rows_filtered"] = filtered_data.height
            except Exception as e:
                filter_results["polars_error"] = str(e)

            # XArray
            try:
                elapsed, filtered_data = self.time_operation(
                    "XArray filtering",
                    lambda: xarray_data.where(filter_func(xarray_data)).compute(),
                )
                filter_results["xarray_time"] = elapsed
                filter_results["xarray_size_filtered"] = np.count_nonzero(
                    ~np.isnan(filtered_data.values)
                )
            except Exception as e:
                filter_results["xarray_error"] = str(e)

            results[filter_name] = filter_results

        # Clean up
        xarray_data.close()
        del polars_data, xarray_data
        gc.collect()

        return results

    def _load_with_polars(self, dims: Dict[str, slice]) -> pl.LazyFrame:
        """Load data using the Polars-based approach."""
        reader = ClimateDataReader(self.store_path)
        arrays = reader.list_arrays()

        if not arrays:
            raise ValueError("No arrays found in the dataset")

        # Use the first available array
        array_name = arrays[0]
        return reader.read_array(array_name, select_dims=dims, streaming=True)

    def _load_with_xarray(self, dims: Dict[str, slice]) -> xr.DataArray:
        """Load data using the traditional XArray approach."""
        import s3fs
        import zarr

        # Open zarr store
        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=self.store_path, s3=fs, check=False)

        try:
            # Try consolidated first
            group = zarr.open_consolidated(store, mode="r")
        except:
            # Fall back to regular open
            group = zarr.open(store, mode="r")

        # Get the first array
        array_names = [name for name in group.keys() if hasattr(group[name], "shape")]
        if not array_names:
            raise ValueError("No arrays found in the zarr store")

        array_name = array_names[0]
        zarr_array = group[array_name]

        # Create xarray DataArray
        da = xr.DataArray(
            zarr_array,
            dims=["time", "lat", "lon"],  # Assume standard climate dimensions
        )

        # Apply slicing
        selection = {}
        for dim, slice_obj in dims.items():
            if dim in da.dims:
                selection[dim] = slice_obj

        return da.isel(selection)

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("Climate Data Processing Performance Comparison")
        print("Polars vs XArray")
        print("=" * 80)

        all_results = {}

        # Run each benchmark
        benchmarks = [
            ("data_loading", self.benchmark_data_loading),
            ("aggregation", self.benchmark_aggregation_operations),
            ("groupby", self.benchmark_groupby_operations),
            ("filtering", self.benchmark_filtering_operations),
        ]

        for benchmark_name, benchmark_func in benchmarks:
            try:
                results = benchmark_func()
                all_results[benchmark_name] = results
            except Exception as e:
                print(f"Benchmark {benchmark_name} failed: {e}")
                all_results[benchmark_name] = {"error": str(e)}

        return all_results

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)

        # Data loading summary
        if "data_loading" in results:
            print("\n📊 Data Loading Performance:")
            print("-" * 40)
            for subset_name, subset_results in results["data_loading"].items():
                if (
                    "polars_load_time" in subset_results
                    and "xarray_load_time" in subset_results
                ):
                    polars_time = subset_results["polars_load_time"]
                    xarray_time = subset_results["xarray_load_time"]
                    speedup = xarray_time / polars_time if polars_time > 0 else 0

                    print(f"  {subset_name.replace('_', ' ').title()}:")
                    print(f"    Polars: {polars_time:.3f}s")
                    print(f"    XArray: {xarray_time:.3f}s")
                    print(
                        f"    Speedup: {speedup:.2f}x {'(Polars faster)' if speedup > 1 else '(XArray faster)'}"
                    )

        # Aggregation summary
        if "aggregation" in results:
            print("\n🧮 Aggregation Operations:")
            print("-" * 40)
            for op_name, op_results in results["aggregation"].items():
                if "polars_time" in op_results and "xarray_time" in op_results:
                    polars_time = op_results["polars_time"]
                    xarray_time = op_results["xarray_time"]
                    speedup = xarray_time / polars_time if polars_time > 0 else 0

                    print(
                        f"  {op_name}: Polars {polars_time:.3f}s vs XArray {xarray_time:.3f}s "
                        f"({speedup:.2f}x)"
                    )

        # Group-by summary
        if "groupby" in results:
            print("\n📊 Group-by Operations:")
            print("-" * 40)
            for op_name, op_results in results["groupby"].items():
                if "polars_time" in op_results and "xarray_time" in op_results:
                    polars_time = op_results["polars_time"]
                    xarray_time = op_results["xarray_time"]
                    speedup = xarray_time / polars_time if polars_time > 0 else 0

                    print(
                        f"  {op_name.replace('_', ' ').title()}: Polars {polars_time:.3f}s vs XArray {xarray_time:.3f}s "
                        f"({speedup:.2f}x)"
                    )

        # Filtering summary
        if "filtering" in results:
            print("\n🔍 Filtering Operations:")
            print("-" * 40)
            for filter_name, filter_results in results["filtering"].items():
                if "polars_time" in filter_results and "xarray_time" in filter_results:
                    polars_time = filter_results["polars_time"]
                    xarray_time = filter_results["xarray_time"]
                    speedup = xarray_time / polars_time if polars_time > 0 else 0

                    print(
                        f"  {filter_name.replace('_', ' ').title()}: Polars {polars_time:.3f}s vs XArray {xarray_time:.3f}s "
                        f"({speedup:.2f}x)"
                    )


def main():
    """Main function to run the performance comparison."""

    # Test data - using the same dataset from our previous tests
    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    # Create benchmark instance
    benchmark = PerformanceBenchmark(store_path)

    # Run all benchmarks
    results = benchmark.run_all_benchmarks()

    # Print summary
    benchmark.print_summary(results)

    # Save detailed results
    print(f"\n💾 Saving detailed results to benchmark_results.json")
    import json

    # Convert any numpy types to native Python types for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, "item"):
            return obj.item()
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    def clean_results(data):
        if isinstance(data, dict):
            return {k: clean_results(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_results(item) for item in data]
        else:
            return convert_for_json(data)

    clean_results_data = clean_results(results)

    with open("benchmark_results.json", "w") as f:
        json.dump(clean_results_data, f, indent=2)

    print("\n✅ Performance comparison completed!")
    print("\nKey Takeaways:")
    print("- Results saved to benchmark_results.json for detailed analysis")
    print("- Check the summary above for high-level performance comparison")
    print("- Both approaches have their strengths depending on the use case")


if __name__ == "__main__":
    main()
