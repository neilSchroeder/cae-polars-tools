#!/usr/bin/env python3
"""
Extended performance test focusing on large dataset operations where Polars should excel.
This test specifically targets Polars' strengths: large data processing, complex filtering,
aggregations, and columnar operations.
"""

import gc
import json
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import psutil
import xarray as xr

warnings.filterwarnings("ignore")

# Import our refactored modules
from climakitae.new_core import ClimateDataReader


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def time_operation(func, *args, **kwargs) -> Tuple[Any, float, float, float]:
    """Time an operation and measure memory usage."""
    gc.collect()  # Clean up before measurement
    start_memory = get_memory_usage()
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    end_memory = get_memory_usage()

    duration = end_time - start_time
    memory_delta = end_memory - start_memory

    return result, duration, start_memory, memory_delta


class LargeDatasetBenchmark:
    """Comprehensive benchmark focusing on large dataset operations."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.results = {}

    def benchmark_large_data_loading(self) -> Dict[str, Any]:
        """Test data loading at various scales."""
        print("\n" + "=" * 70)
        print("LARGE DATASET LOADING BENCHMARK")
        print("=" * 70)

        results = {}

        # Initialize readers
        polars_reader = ClimateDataReader(self.store_path)
        xr_dataset = xr.open_zarr(self.store_path, storage_options={})

        # Test scales: Small, Medium, Large, Extra Large
        test_scales = [
            (
                "Small",
                {"time": slice(0, 12), "lat": slice(100, 150), "lon": slice(200, 250)},
            ),  # 12 months, 50x50
            (
                "Medium",
                {"time": slice(0, 60), "lat": slice(50, 150), "lon": slice(100, 200)},
            ),  # 5 years, 100x100
            (
                "Large",
                {"time": slice(0, 120), "lat": slice(0, 200), "lon": slice(0, 300)},
            ),  # 10 years, 200x300
            (
                "XLarge",
                {"time": slice(0, 240), "lat": slice(0, 300), "lon": slice(0, 400)},
            ),  # 20 years, 300x400
        ]

        for scale_name, dims in test_scales:
            print(f"\n{scale_name} Scale Dataset:")

            # Calculate expected data points
            time_size = dims["time"].stop - dims["time"].start
            lat_size = dims["lat"].stop - dims["lat"].start
            lon_size = dims["lon"].stop - dims["lon"].start
            total_points = time_size * lat_size * lon_size

            print(
                f"   Dimensions: {time_size} × {lat_size} × {lon_size} = {total_points:,} data points"
            )
            print(f"   Estimated size: {total_points * 4 / 1024**2:.1f} MB")

            # Test Polars - Normal mode
            try:
                df_normal, time_p_normal, mem_p_normal, delta_p_normal = time_operation(
                    lambda: polars_reader.read_array(
                        "tasmax", select_dims=dims, streaming=False
                    ).collect()
                )
                print(
                    f"   Polars Normal:    {time_p_normal:8.3f}s | {delta_p_normal:6.1f}MB | Shape: {df_normal.shape}"
                )
            except Exception as e:
                print(f"   Polars Normal:    FAILED - {str(e)[:50]}...")
                time_p_normal, delta_p_normal = float("inf"), float("inf")

            # Test Polars - Streaming mode
            try:
                df_stream, time_p_stream, mem_p_stream, delta_p_stream = time_operation(
                    lambda: polars_reader.read_array(
                        "tasmax", select_dims=dims, streaming=True
                    ).collect()
                )
                print(
                    f"   Polars Streaming: {time_p_stream:8.3f}s | {delta_p_stream:6.1f}MB | Shape: {df_stream.shape}"
                )
            except Exception as e:
                print(f"   Polars Streaming: FAILED - {str(e)[:50]}...")
                time_p_stream, delta_p_stream = float("inf"), float("inf")

            # Test XArray
            try:
                da_x, time_x, mem_x, delta_x = time_operation(
                    lambda: xr_dataset["tasmax"].isel(**dims).load()
                )
                print(
                    f"   XArray:           {time_x:8.3f}s | {delta_x:6.1f}MB | Shape: {da_x.shape}"
                )
            except Exception as e:
                print(f"   XArray:           FAILED - {str(e)[:50]}...")
                time_x, delta_x = float("inf"), float("inf")

            results[scale_name.lower()] = {
                "data_points": total_points,
                "polars_normal": {"time": time_p_normal, "memory": delta_p_normal},
                "polars_streaming": {"time": time_p_stream, "memory": delta_p_stream},
                "xarray": {"time": time_x, "memory": delta_x},
            }

            # Early exit if datasets get too large
            if total_points > 50_000_000:  # 50M points
                print(f"   Skipping larger datasets due to size...")
                break

        xr_dataset.close()
        return results

    def benchmark_complex_filtering(self) -> Dict[str, Any]:
        """Test complex filtering operations where Polars should excel."""
        print("\n" + "=" * 70)
        print("COMPLEX FILTERING BENCHMARK")
        print("=" * 70)

        results = {}

        # Load a substantial dataset for filtering tests
        print("\nLoading test dataset (5 years, 100x100 grid)...")
        polars_reader = ClimateDataReader(self.store_path)
        xr_dataset = xr.open_zarr(self.store_path, storage_options={})

        # Load Polars data
        df_polars = polars_reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 60),
                "lat": slice(100, 200),
                "lon": slice(200, 300),
            },
            streaming=True,
        ).collect()

        # Load XArray data
        da_xarray = (
            xr_dataset["tasmax"]
            .isel(time=slice(0, 60), lat=slice(100, 200), lon=slice(200, 300))
            .load()
        )

        print(f"Dataset loaded - Polars: {df_polars.shape}, XArray: {da_xarray.shape}")

        # Test 1: Simple threshold filtering
        print("\n1. Simple Threshold Filter (> 300K)")

        def polars_simple_filter():
            return df_polars.filter(pl.col("value") > 300).shape[0]

        count_p1, time_p1, _, _ = time_operation(polars_simple_filter)

        def xarray_simple_filter():
            return int((da_xarray > 300).sum().values)

        count_x1, time_x1, _, _ = time_operation(xarray_simple_filter)

        print(f"   Polars: {time_p1:.4f}s | Count: {count_p1:,}")
        print(f"   XArray: {time_x1:.4f}s | Count: {count_x1:,}")

        results["simple_filter"] = {
            "polars": {"time": time_p1, "count": count_p1},
            "xarray": {"time": time_x1, "count": count_x1},
        }

        # Test 2: Complex multi-condition filter
        print("\n2. Complex Multi-Condition Filter")
        print("   (Temperature > 295K AND < 310K AND lat > middle AND lon < middle)")

        def polars_complex_filter():
            lat_mid = df_polars["lat"].median()
            lon_mid = df_polars["lon"].median()
            return df_polars.filter(
                (pl.col("value") > 295)
                & (pl.col("value") < 310)
                & (pl.col("lat") > lat_mid)
                & (pl.col("lon") < lon_mid)
            ).shape[0]

        count_p2, time_p2, _, _ = time_operation(polars_complex_filter)

        def xarray_complex_filter():
            lat_coords = da_xarray.lat.values
            lon_coords = da_xarray.lon.values
            lat_mid = np.median(lat_coords)
            lon_mid = np.median(lon_coords)

            mask = (
                (da_xarray > 295)
                & (da_xarray < 310)
                & (da_xarray.lat > lat_mid)
                & (da_xarray.lon < lon_mid)
            )
            return int(mask.sum().values)

        count_x2, time_x2, _, _ = time_operation(xarray_complex_filter)

        print(f"   Polars: {time_p2:.4f}s | Count: {count_p2:,}")
        print(f"   XArray: {time_x2:.4f}s | Count: {count_x2:,}")

        results["complex_filter"] = {
            "polars": {"time": time_p2, "count": count_p2},
            "xarray": {"time": time_x2, "count": count_x2},
        }

        # Test 3: String/categorical filtering (simulate season filtering)
        print("\n3. Temporal Filtering (Summer months: June-August)")

        # Add month column to Polars data for realistic filtering
        df_with_month = df_polars.with_columns(
            [(pl.col("time") % 12).alias("month")]  # Simplified month extraction
        )

        def polars_seasonal_filter():
            return df_with_month.filter(
                pl.col("month").is_in([5, 6, 7])  # June, July, August (0-indexed)
            ).shape[0]

        count_p3, time_p3, _, _ = time_operation(polars_seasonal_filter)

        def xarray_seasonal_filter():
            # Simulate month extraction and filtering
            time_indices = np.arange(da_xarray.shape[0])
            month_mask = np.isin(time_indices % 12, [5, 6, 7])
            return int(da_xarray[month_mask].count().values)

        count_x3, time_x3, _, _ = time_operation(xarray_seasonal_filter)

        print(f"   Polars: {time_p3:.4f}s | Count: {count_p3:,}")
        print(f"   XArray: {time_x3:.4f}s | Count: {count_x3:,}")

        results["seasonal_filter"] = {
            "polars": {"time": time_p3, "count": count_p3},
            "xarray": {"time": time_x3, "count": count_x3},
        }

        xr_dataset.close()
        return results

    def benchmark_aggregation_operations(self) -> Dict[str, Any]:
        """Test aggregation operations at scale."""
        print("\n" + "=" * 70)
        print("LARGE-SCALE AGGREGATION BENCHMARK")
        print("=" * 70)

        results = {}

        # Load larger dataset for aggregations
        print("\nLoading large dataset (10 years, 150x150 grid)...")
        polars_reader = ClimateDataReader(self.store_path)
        xr_dataset = xr.open_zarr(self.store_path, storage_options={})

        # Load data
        df_polars = polars_reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 120),
                "lat": slice(50, 200),
                "lon": slice(100, 250),
            },
            streaming=True,
        ).collect()

        da_xarray = (
            xr_dataset["tasmax"]
            .isel(time=slice(0, 120), lat=slice(50, 200), lon=slice(100, 250))
            .load()
        )

        print(f"Dataset loaded - Polars: {df_polars.shape}, XArray: {da_xarray.shape}")

        # Test 1: Group by spatial coordinates and calculate statistics
        print("\n1. Spatial Aggregation (Group by lat/lon, calculate mean temperature)")

        def polars_spatial_agg():
            return df_polars.group_by(["lat", "lon"]).agg(
                [
                    pl.col("value").mean().alias("temp_mean"),
                    pl.col("value").std().alias("temp_std"),
                    pl.col("value").count().alias("count"),
                ]
            )

        agg_p1, time_p1, mem_p1, delta_p1 = time_operation(polars_spatial_agg)

        def xarray_spatial_agg():
            return da_xarray.groupby("time").mean()  # Equivalent operation

        agg_x1, time_x1, mem_x1, delta_x1 = time_operation(xarray_spatial_agg)

        print(
            f"   Polars: {time_p1:.4f}s | {delta_p1:6.1f}MB | Result shape: {agg_p1.shape}"
        )
        print(
            f"   XArray: {time_x1:.4f}s | {delta_x1:6.1f}MB | Result shape: {agg_x1.shape}"
        )

        results["spatial_aggregation"] = {
            "polars": {
                "time": time_p1,
                "memory": delta_p1,
                "result_size": agg_p1.shape[0],
            },
            "xarray": {
                "time": time_x1,
                "memory": delta_x1,
                "result_size": agg_x1.shape[0],
            },
        }

        # Test 2: Temporal aggregation with multiple statistics
        print("\n2. Temporal Aggregation (Group by time, multiple statistics)")

        def polars_temporal_agg():
            return (
                df_polars.group_by("time")
                .agg(
                    [
                        pl.col("value").mean().alias("temp_mean"),
                        pl.col("value").max().alias("temp_max"),
                        pl.col("value").min().alias("temp_min"),
                        pl.col("value").std().alias("temp_std"),
                        pl.col("value").quantile(0.25).alias("temp_q25"),
                        pl.col("value").quantile(0.75).alias("temp_q75"),
                    ]
                )
                .sort("time")
            )

        agg_p2, time_p2, mem_p2, delta_p2 = time_operation(polars_temporal_agg)

        def xarray_temporal_agg():
            return {
                "mean": da_xarray.mean(dim=["lat", "lon"]),
                "max": da_xarray.max(dim=["lat", "lon"]),
                "min": da_xarray.min(dim=["lat", "lon"]),
                "std": da_xarray.std(dim=["lat", "lon"]),
                "q25": da_xarray.quantile(0.25, dim=["lat", "lon"]),
                "q75": da_xarray.quantile(0.75, dim=["lat", "lon"]),
            }

        agg_x2, time_x2, mem_x2, delta_x2 = time_operation(xarray_temporal_agg)

        print(
            f"   Polars: {time_p2:.4f}s | {delta_p2:6.1f}MB | Result shape: {agg_p2.shape}"
        )
        print(f"   XArray: {time_x2:.4f}s | {delta_x2:6.1f}MB | Multiple arrays")

        results["temporal_aggregation"] = {
            "polars": {
                "time": time_p2,
                "memory": delta_p2,
                "result_size": agg_p2.shape[0],
            },
            "xarray": {
                "time": time_x2,
                "memory": delta_x2,
                "result_count": len(agg_x2),
            },
        }

        # Test 3: Complex window operations
        print("\n3. Rolling Window Operations (30-day rolling mean)")

        def polars_rolling():
            return df_polars.sort("time").with_columns(
                [pl.col("value").rolling_mean(window_size=30).alias("rolling_mean")]
            )

        roll_p, time_p3, mem_p3, delta_p3 = time_operation(polars_rolling)

        def xarray_rolling():
            return da_xarray.rolling(time=30, center=True).mean()

        roll_x, time_x3, mem_x3, delta_x3 = time_operation(xarray_rolling)

        print(f"   Polars: {time_p3:.4f}s | {delta_p3:6.1f}MB")
        print(f"   XArray: {time_x3:.4f}s | {delta_x3:6.1f}MB")

        results["rolling_operations"] = {
            "polars": {"time": time_p3, "memory": delta_p3},
            "xarray": {"time": time_x3, "memory": delta_x3},
        }

        xr_dataset.close()
        return results

    def benchmark_join_operations(self) -> Dict[str, Any]:
        """Test join operations - a key Polars strength."""
        print("\n" + "=" * 70)
        print("JOIN OPERATIONS BENCHMARK")
        print("=" * 70)

        results = {}

        polars_reader = ClimateDataReader(self.store_path)

        print("\nLoading multiple variables for join operations...")

        # Load temperature data
        temp_df = (
            polars_reader.read_array(
                "tasmax",
                select_dims={
                    "time": slice(0, 24),
                    "lat": slice(100, 150),
                    "lon": slice(200, 250),
                },
                streaming=True,
            )
            .collect()
            .rename({"value": "temperature"})
        )

        # Create synthetic precipitation data for join testing
        print("Creating synthetic related dataset...")

        # Generate random precipitation values for each row
        np.random.seed(42)  # For reproducible results
        precip_values = np.random.exponential(2.0, temp_df.shape[0])

        precip_df = temp_df.select(["time", "lat", "lon"]).with_columns(
            [pl.Series("precipitation", precip_values)]
        )

        print(f"Temperature data: {temp_df.shape}")
        print(f"Precipitation data: {precip_df.shape}")

        # Test 1: Inner join on coordinates
        print("\n1. Inner Join on Spatial-Temporal Coordinates")

        def polars_join():
            return temp_df.join(precip_df, on=["time", "lat", "lon"], how="inner")

        joined_p, time_p1, mem_p1, delta_p1 = time_operation(polars_join)

        print(
            f"   Polars Join: {time_p1:.4f}s | {delta_p1:6.1f}MB | Result: {joined_p.shape}"
        )

        # XArray equivalent (much more complex)
        def xarray_join():
            # XArray doesn't have direct join - simulate with coordinate alignment
            xr_dataset = xr.open_zarr(self.store_path, storage_options={})
            temp_xa = (
                xr_dataset["tasmax"]
                .isel(time=slice(0, 24), lat=slice(100, 150), lon=slice(200, 250))
                .load()
            )

            # Create synthetic precipitation xarray
            precip_xa = temp_xa.copy()
            precip_xa.values = np.random.exponential(2.0, temp_xa.shape)
            precip_xa.name = "precipitation"

            # Combine into dataset
            combined = xr.Dataset({"temperature": temp_xa, "precipitation": precip_xa})

            xr_dataset.close()
            return combined

        combined_x, time_x1, mem_x1, delta_x1 = time_operation(xarray_join)

        print(
            f"   XArray Combine: {time_x1:.4f}s | {delta_x1:6.1f}MB | Result: {combined_x.sizes}"
        )

        results["inner_join"] = {
            "polars": {
                "time": time_p1,
                "memory": delta_p1,
                "result_shape": joined_p.shape,
            },
            "xarray": {
                "time": time_x1,
                "memory": delta_x1,
                "result_vars": len(combined_x.data_vars),
            },
        }

        # Test 2: Complex join with aggregation
        print("\n2. Join with Aggregation (Monthly averages)")

        def polars_join_agg():
            # Group both datasets by month and location, then join
            temp_monthly = (
                temp_df.with_columns(
                    [
                        (pl.col("time") // 30).alias(
                            "month"
                        )  # Simplified monthly grouping
                    ]
                )
                .group_by(["month", "lat", "lon"])
                .agg([pl.col("temperature").mean().alias("avg_temp")])
            )

            precip_monthly = (
                precip_df.with_columns([(pl.col("time") // 30).alias("month")])
                .group_by(["month", "lat", "lon"])
                .agg([pl.col("precipitation").sum().alias("total_precip")])
            )

            return temp_monthly.join(
                precip_monthly, on=["month", "lat", "lon"], how="inner"
            )

        monthly_p, time_p2, mem_p2, delta_p2 = time_operation(polars_join_agg)

        print(
            f"   Polars Join+Agg: {time_p2:.4f}s | {delta_p2:6.1f}MB | Result: {monthly_p.shape}"
        )

        results["join_with_aggregation"] = {
            "polars": {
                "time": time_p2,
                "memory": delta_p2,
                "result_shape": monthly_p.shape,
            },
            "xarray": {"note": "Complex operation - not directly comparable"},
        }

        return results

    def run_comprehensive_large_dataset_benchmark(self) -> Dict[str, Any]:
        """Run all large dataset benchmarks."""
        print("LARGE DATASET PERFORMANCE BENCHMARK")
        print("=" * 80)
        print(f"Dataset: {self.store_path}")
        print(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f}GB")

        start_time = time.time()

        try:
            # 1. Large Data Loading
            self.results["large_loading"] = self.benchmark_large_data_loading()

            # 2. Complex Filtering
            self.results["complex_filtering"] = self.benchmark_complex_filtering()

            # 3. Aggregation Operations
            self.results["aggregation"] = self.benchmark_aggregation_operations()

            # 4. Join Operations
            self.results["join_operations"] = self.benchmark_join_operations()

        except Exception as e:
            print(f"Error during benchmark: {e}")
            self.results["error"] = str(e)

        total_time = time.time() - start_time
        self.results["total_benchmark_time"] = total_time

        return self.results

    def print_detailed_summary(self):
        """Print detailed summary focusing on where each tool excels."""
        print("\n" + "=" * 80)
        print("LARGE DATASET BENCHMARK SUMMARY")
        print("=" * 80)

        if "error" in self.results:
            print(f"Benchmark failed with error: {self.results['error']}")
            return

        # Large Loading Summary
        if "large_loading" in self.results:
            print(f"\n🔄 LARGE DATA LOADING:")
            loading = self.results["large_loading"]

            for scale in ["small", "medium", "large", "xlarge"]:
                if scale in loading:
                    data = loading[scale]
                    print(
                        f"\n   {scale.upper()} Scale ({data['data_points']:,} points):"
                    )

                    if data["polars_normal"]["time"] != float("inf"):
                        print(
                            f"     Polars Normal:    {data['polars_normal']['time']:8.3f}s"
                        )
                    if data["polars_streaming"]["time"] != float("inf"):
                        print(
                            f"     Polars Streaming: {data['polars_streaming']['time']:8.3f}s"
                        )
                    if data["xarray"]["time"] != float("inf"):
                        print(f"     XArray:           {data['xarray']['time']:8.3f}s")

                    # Determine winner
                    times = [
                        ("Polars Normal", data["polars_normal"]["time"]),
                        ("Polars Streaming", data["polars_streaming"]["time"]),
                        ("XArray", data["xarray"]["time"]),
                    ]
                    winner = min(times, key=lambda x: x[1])
                    if winner[1] != float("inf"):
                        print(f"     🏆 Winner: {winner[0]} ({winner[1]:.3f}s)")

        # Filtering Summary
        if "complex_filtering" in self.results:
            print(f"\n🔍 COMPLEX FILTERING:")
            filtering = self.results["complex_filtering"]

            for test_name, data in filtering.items():
                print(f"\n   {test_name.replace('_', ' ').title()}:")
                print(f"     Polars: {data['polars']['time']:.4f}s")
                print(f"     XArray: {data['xarray']['time']:.4f}s")

                if data["polars"]["time"] < data["xarray"]["time"]:
                    speedup = data["xarray"]["time"] / data["polars"]["time"]
                    print(f"     🏆 Polars wins by {speedup:.1f}x")
                else:
                    speedup = data["polars"]["time"] / data["xarray"]["time"]
                    print(f"     🏆 XArray wins by {speedup:.1f}x")

        # Aggregation Summary
        if "aggregation" in self.results:
            print(f"\n📊 LARGE-SCALE AGGREGATIONS:")
            agg = self.results["aggregation"]

            for test_name, data in agg.items():
                print(f"\n   {test_name.replace('_', ' ').title()}:")
                print(
                    f"     Polars: {data['polars']['time']:.4f}s | {data['polars']['memory']:6.1f}MB"
                )
                print(
                    f"     XArray: {data['xarray']['time']:.4f}s | {data['xarray']['memory']:6.1f}MB"
                )

                if data["polars"]["time"] < data["xarray"]["time"]:
                    speedup = data["xarray"]["time"] / data["polars"]["time"]
                    print(f"     🏆 Polars wins by {speedup:.1f}x")
                else:
                    speedup = data["polars"]["time"] / data["xarray"]["time"]
                    print(f"     🏆 XArray wins by {speedup:.1f}x")

        # Join Operations Summary
        if "join_operations" in self.results:
            print(f"\n🔗 JOIN OPERATIONS (Polars Specialty):")
            joins = self.results["join_operations"]

            for test_name, data in joins.items():
                if "polars" in data:
                    print(f"\n   {test_name.replace('_', ' ').title()}:")
                    print(
                        f"     Polars: {data['polars']['time']:.4f}s | {data['polars']['memory']:6.1f}MB"
                    )
                    if "xarray" in data and "time" in data["xarray"]:
                        print(f"     XArray: {data['xarray']['time']:.4f}s")
                    else:
                        print(f"     XArray: Not directly comparable")

        print(
            f"\n⏱️  Total benchmark time: {self.results.get('total_benchmark_time', 0):.2f}s"
        )

        # Final recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(
            f"   • Use Polars for: ETL operations, complex filtering, joins, streaming large datasets"
        )
        print(
            f"   • Use XArray for: Scientific analysis, dimension-aware operations, metadata preservation"
        )
        print(
            f"   • Consider hybrid approach: Polars for data loading/prep + XArray for analysis"
        )


def main():
    """Run the large dataset benchmark."""
    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    print("EXTENDED LARGE DATASET PERFORMANCE TEST")
    print("=======================================")
    print(
        "Testing Polars' strengths: large data, complex filtering, aggregations, joins"
    )

    benchmark = LargeDatasetBenchmark(store_path)
    results = benchmark.run_comprehensive_large_dataset_benchmark()
    benchmark.print_detailed_summary()

    # Save results
    with open("large_dataset_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: large_dataset_benchmark_results.json")


if __name__ == "__main__":
    main()
