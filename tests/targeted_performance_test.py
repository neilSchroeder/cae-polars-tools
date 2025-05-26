#!/usr/bin/env python3
"""
Targeted performance test focusing on Polars' true strengths vs XArray.
Tests operations where Polars should excel: complex queries, joins, data transformation pipelines.
"""

import gc
import json
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
import psutil
import xarray as xr

# Import our refactored modules
from climakitae.new_core import ClimateDataReader


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def time_operation(func, *args, **kwargs) -> Tuple[Any, float, float, float]:
    """Time an operation and measure memory usage."""
    gc.collect()
    start_memory = get_memory_usage()
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    end_memory = get_memory_usage()

    duration = end_time - start_time
    memory_delta = end_memory - start_memory

    return result, duration, start_memory, memory_delta


class TargetedBenchmark:
    """Benchmark focusing on Polars' strengths."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.results = {}

    def benchmark_data_pipeline_operations(self) -> Dict[str, Any]:
        """Test complex data pipeline operations where Polars should excel."""
        print("\n" + "=" * 70)
        print("DATA PIPELINE OPERATIONS BENCHMARK")
        print("=" * 70)

        # Load a substantial dataset for pipeline testing
        print("\n1. Loading substantial dataset (24 months, 200x200 grid)...")

        reader = ClimateDataReader(self.store_path)

        # Load data with Polars
        def load_polars_pipeline():
            return reader.read_array(
                "tasmax",
                select_dims={
                    "time": slice(0, 24),
                    "lat": slice(100, 300),
                    "lon": slice(200, 400),
                },
                streaming=True,
            ).collect()

        df_p, load_time_p, _, load_mem_p = time_operation(load_polars_pipeline)
        print(
            f"   Polars load: {load_time_p:.4f}s | {load_mem_p:.1f}MB | Shape: {df_p.shape}"
        )

        # Load equivalent with XArray
        def load_xarray_pipeline():
            ds = xr.open_zarr(self.store_path, storage_options={})
            return (
                ds["tasmax"]
                .isel(time=slice(0, 24), lat=slice(100, 300), lon=slice(200, 400))
                .load()
            )

        da_x, load_time_x, _, load_mem_x = time_operation(load_xarray_pipeline)
        print(
            f"   XArray load: {load_time_x:.4f}s | {load_mem_x:.1f}MB | Shape: {da_x.shape}"
        )

        results = {
            "loading": {
                "polars": {
                    "time": load_time_p,
                    "memory": load_mem_p,
                    "shape": df_p.shape,
                },
                "xarray": {
                    "time": load_time_x,
                    "memory": load_mem_x,
                    "shape": da_x.shape,
                },
            }
        }

        # Test 1: Complex chained operations (Polars' strength)
        print("\n2. Complex Chained Operations Pipeline")

        def polars_complex_chain():
            return (
                df_p.filter(pl.col("value").is_not_null())  # Remove nulls
                .with_columns(
                    [
                        (pl.col("value") - 273.15).alias(
                            "temp_celsius"
                        ),  # Convert to Celsius
                        (pl.col("lat").round(1)).alias(
                            "lat_rounded"
                        ),  # Round coordinates
                        (pl.col("lon").round(1)).alias("lon_rounded"),
                        pl.when(pl.col("value") > 300)
                        .then(pl.lit("hot"))
                        .when(pl.col("value") > 285)
                        .then(pl.lit("warm"))
                        .when(pl.col("value") > 270)
                        .then(pl.lit("cool"))
                        .otherwise(pl.lit("cold"))
                        .alias("temp_category"),
                    ]
                )
                .filter(pl.col("temp_celsius") > 10)  # Only above 10°C
                .group_by(["lat_rounded", "lon_rounded", "temp_category"])
                .agg(
                    [
                        pl.col("temp_celsius").mean().alias("mean_temp"),
                        pl.col("temp_celsius").std().alias("std_temp"),
                        pl.count().alias("count"),
                        pl.col("temp_celsius").min().alias("min_temp"),
                        pl.col("temp_celsius").max().alias("max_temp"),
                    ]
                )
                .filter(pl.col("count") > 10)  # Only groups with enough data
                .sort(["lat_rounded", "lon_rounded", "mean_temp"])
            )

        result_p, chain_time_p, _, chain_mem_p = time_operation(polars_complex_chain)
        print(
            f"   Polars chain: {chain_time_p:.4f}s | {chain_mem_p:.1f}MB | Result: {result_p.shape}"
        )

        def xarray_complex_chain():
            # Equivalent operations with XArray (more verbose)
            da_celsius = da_x - 273.15
            da_valid = da_celsius.where(da_celsius.notnull())
            da_filtered = da_valid.where(da_celsius > 10)

            # XArray doesn't have built-in categorization like this, so we'll do basic stats
            result_dict = {
                "mean": da_filtered.mean().values.item(),
                "std": da_filtered.std().values.item(),
                "count": da_filtered.count().values.item(),
                "min": da_filtered.min().values.item(),
                "max": da_filtered.max().values.item(),
            }
            return result_dict

        result_x, chain_time_x, _, chain_mem_x = time_operation(xarray_complex_chain)
        print(
            f"   XArray chain: {chain_time_x:.4f}s | {chain_mem_x:.1f}MB | Result: basic stats"
        )

        results["complex_chain"] = {
            "polars": {
                "time": chain_time_p,
                "memory": chain_mem_p,
                "shape": result_p.shape,
            },
            "xarray": {
                "time": chain_time_x,
                "memory": chain_mem_x,
                "type": "basic_stats",
            },
        }

        return results, df_p, da_x

    def benchmark_analytical_queries(
        self, df_p: pl.DataFrame, da_x: xr.DataArray
    ) -> Dict[str, Any]:
        """Test analytical queries where Polars should excel."""
        print("\n" + "=" * 70)
        print("ANALYTICAL QUERIES BENCHMARK")
        print("=" * 70)

        results = {}

        # Test 1: Complex conditional aggregations
        print("\n1. Complex Conditional Aggregations")

        def polars_conditional_agg():
            return (
                df_p.with_columns((pl.col("value") - 273.15).alias("temp_c"))
                .group_by("time")
                .agg(
                    [
                        pl.col("temp_c")
                        .filter(pl.col("lat") > pl.col("lat").median())
                        .mean()
                        .alias("north_mean"),
                        pl.col("temp_c")
                        .filter(pl.col("lat") <= pl.col("lat").median())
                        .mean()
                        .alias("south_mean"),
                        pl.col("temp_c")
                        .filter(pl.col("lon") > pl.col("lon").median())
                        .mean()
                        .alias("east_mean"),
                        pl.col("temp_c")
                        .filter(pl.col("lon") <= pl.col("lon").median())
                        .mean()
                        .alias("west_mean"),
                        pl.col("temp_c")
                        .filter(pl.col("temp_c") > 25)
                        .count()
                        .alias("hot_count"),
                        pl.col("temp_c").quantile(0.95).alias("p95_temp"),
                    ]
                )
            )

        agg_p, agg_time_p, _, agg_mem_p = time_operation(polars_conditional_agg)
        print(
            f"   Polars conditional agg: {agg_time_p:.4f}s | {agg_mem_p:.1f}MB | Shape: {agg_p.shape}"
        )

        def xarray_conditional_agg():
            # XArray equivalent (more complex for conditional operations)
            temp_c = da_x - 273.15
            lat_median = da_x.lat.median()
            lon_median = da_x.lon.median()

            results = []
            for t in range(da_x.shape[0]):
                time_slice = temp_c[t]
                north_mask = da_x.lat > lat_median
                south_mask = da_x.lat <= lat_median
                east_mask = da_x.lon > lon_median
                west_mask = da_x.lon <= lon_median

                results.append(
                    {
                        "north_mean": time_slice.where(north_mask).mean().values.item(),
                        "south_mean": time_slice.where(south_mask).mean().values.item(),
                        "east_mean": time_slice.where(east_mask).mean().values.item(),
                        "west_mean": time_slice.where(west_mask).mean().values.item(),
                        "hot_count": (time_slice > 25).sum().values.item(),
                        "p95_temp": time_slice.quantile(0.95).values.item(),
                    }
                )

            return pd.DataFrame(results)

        agg_x, agg_time_x, _, agg_mem_x = time_operation(xarray_conditional_agg)
        print(
            f"   XArray conditional agg: {agg_time_x:.4f}s | {agg_mem_x:.1f}MB | Shape: {agg_x.shape}"
        )

        results["conditional_agg"] = {
            "polars": {"time": agg_time_p, "memory": agg_mem_p, "shape": agg_p.shape},
            "xarray": {"time": agg_time_x, "memory": agg_mem_x, "shape": agg_x.shape},
        }

        # Test 2: Window functions and rolling operations (fixed)
        print("\n2. Rolling Window Operations")

        def polars_rolling():
            return (
                df_p.sort("time")
                .with_columns((pl.col("value") - 273.15).alias("temp_c"))
                .group_by(["lat", "lon"], maintain_order=True)
                .agg(
                    [
                        pl.col("time"),
                        pl.col("temp_c")
                        .rolling_mean(window_size=3)
                        .alias("rolling_3mo_mean"),
                        pl.col("temp_c")
                        .rolling_std(window_size=3)
                        .alias("rolling_3mo_std"),
                        pl.col("temp_c")
                        .rolling_max(window_size=6)
                        .alias("rolling_6mo_max"),
                        pl.col("temp_c")
                        .rolling_min(window_size=6)
                        .alias("rolling_6mo_min"),
                    ]
                )
                .explode(
                    [
                        "time",
                        "rolling_3mo_mean",
                        "rolling_3mo_std",
                        "rolling_6mo_max",
                        "rolling_6mo_min",
                    ]
                )
            )

        rolling_p, rolling_time_p, _, rolling_mem_p = time_operation(polars_rolling)
        print(
            f"   Polars rolling: {rolling_time_p:.4f}s | {rolling_mem_p:.1f}MB | Shape: {rolling_p.shape}"
        )

        def xarray_rolling():
            temp_c = da_x - 273.15
            return {
                "rolling_3mo_mean": temp_c.rolling(time=3).mean(),
                "rolling_3mo_std": temp_c.rolling(time=3).std(),
                "rolling_6mo_max": temp_c.rolling(time=6).max(),
                "rolling_6mo_min": temp_c.rolling(time=6).min(),
            }

        rolling_x, rolling_time_x, _, rolling_mem_x = time_operation(xarray_rolling)
        print(
            f"   XArray rolling: {rolling_time_x:.4f}s | {rolling_mem_x:.1f}MB | Multiple arrays"
        )

        results["rolling"] = {
            "polars": {
                "time": rolling_time_p,
                "memory": rolling_mem_p,
                "shape": rolling_p.shape,
            },
            "xarray": {
                "time": rolling_time_x,
                "memory": rolling_mem_x,
                "type": "multiple_arrays",
            },
        }

        return results

    def benchmark_data_joining_operations(self) -> Dict[str, Any]:
        """Test data joining operations where Polars should excel."""
        print("\n" + "=" * 70)
        print("DATA JOINING OPERATIONS BENCHMARK")
        print("=" * 70)

        results = {}

        # Create auxiliary datasets for joining
        reader = ClimateDataReader(self.store_path)

        # Load temperature data
        temp_df = reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 12),
                "lat": slice(150, 200),
                "lon": slice(250, 300),
            },
            streaming=True,
        ).collect()

        # Create synthetic location metadata for joining
        print("\n1. Creating synthetic location metadata...")
        unique_coords = (
            temp_df.select(["lat", "lon"])
            .unique()
            .with_row_index("location_id")
            .with_columns(
                [
                    pl.when(pl.col("lat") > pl.col("lat").median())
                    .then(pl.lit("Northern"))
                    .otherwise(pl.lit("Southern"))
                    .alias("region"),
                    pl.when(pl.col("lon") > pl.col("lon").median())
                    .then(pl.lit("Eastern"))
                    .otherwise(pl.lit("Western"))
                    .alias("zone"),
                    (pl.col("lat") * pl.col("lon") % 100).alias("elevation_class"),
                    pl.when((pl.col("lat") + pl.col("lon")) % 3 == 0)
                    .then(pl.lit("urban"))
                    .otherwise(pl.lit("rural"))
                    .alias("land_use"),
                ]
            )
        )

        print(f"   Location metadata shape: {unique_coords.shape}")

        # Test 1: Complex join with aggregation
        print("\n2. Complex Join with Aggregation")

        def polars_complex_join():
            return (
                temp_df.join(unique_coords, on=["lat", "lon"], how="left")
                .with_columns((pl.col("value") - 273.15).alias("temp_c"))
                .group_by(["time", "region", "zone", "land_use"])
                .agg(
                    [
                        pl.col("temp_c").mean().alias("mean_temp"),
                        pl.col("temp_c").std().alias("std_temp"),
                        pl.col("temp_c").min().alias("min_temp"),
                        pl.col("temp_c").max().alias("max_temp"),
                        pl.count().alias("location_count"),
                        pl.col("elevation_class").mean().alias("avg_elevation_class"),
                    ]
                )
                .filter(pl.col("location_count") > 10)
                .sort(["time", "region", "zone"])
            )

        join_p, join_time_p, _, join_mem_p = time_operation(polars_complex_join)
        print(
            f"   Polars complex join: {join_time_p:.4f}s | {join_mem_p:.1f}MB | Shape: {join_p.shape}"
        )

        # XArray doesn't have native joining capabilities like this
        def xarray_equivalent():
            # This would require manual coordinate matching and is very cumbersome in XArray
            # We'll simulate with basic operations
            ds = xr.open_zarr(self.store_path, storage_options={})
            subset = (
                ds["tasmax"]
                .isel(time=slice(0, 12), lat=slice(150, 200), lon=slice(250, 300))
                .load()
            )

            # Basic regional stats (not a true equivalent)
            temp_c = subset - 273.15
            lat_mid = subset.lat.median()
            lon_mid = subset.lon.median()

            north = temp_c.where(subset.lat > lat_mid)
            south = temp_c.where(subset.lat <= lat_mid)

            return {
                "north_mean": north.mean().values.item(),
                "south_mean": south.mean().values.item(),
                "north_std": north.std().values.item(),
                "south_std": south.std().values.item(),
            }

        equiv_x, equiv_time_x, _, equiv_mem_x = time_operation(xarray_equivalent)
        print(
            f"   XArray equivalent: {equiv_time_x:.4f}s | {equiv_mem_x:.1f}MB | Basic stats only"
        )

        results["complex_join"] = {
            "polars": {
                "time": join_time_p,
                "memory": join_mem_p,
                "shape": join_p.shape,
            },
            "xarray": {
                "time": equiv_time_x,
                "memory": equiv_mem_x,
                "note": "Limited capability",
            },
        }

        return results

    def benchmark_string_and_categorical_operations(self) -> Dict[str, Any]:
        """Test string/categorical operations where Polars excels."""
        print("\n" + "=" * 70)
        print("STRING/CATEGORICAL OPERATIONS BENCHMARK")
        print("=" * 70)

        results = {}

        # Load data and create categorical columns
        reader = ClimateDataReader(self.store_path)
        df = reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 12),
                "lat": slice(100, 150),
                "lon": slice(200, 250),
            },
            streaming=True,
        ).collect()

        print(
            f"\n1. Creating categorical data from {df.shape[0]:,} temperature records..."
        )

        def polars_categorical_ops():
            return (
                df.with_columns(
                    [
                        (pl.col("value") - 273.15).alias("temp_c"),
                        # Create month names from time index
                        (pl.col("time") % 12 + 1)
                        .map_elements(
                            lambda x: [
                                "Jan",
                                "Feb",
                                "Mar",
                                "Apr",
                                "May",
                                "Jun",
                                "Jul",
                                "Aug",
                                "Sep",
                                "Oct",
                                "Nov",
                                "Dec",
                            ][x - 1],
                            return_dtype=pl.Utf8,
                        )
                        .alias("month_name"),
                        # Create temperature categories
                        pl.when(pl.col("value") > 310)
                        .then(pl.lit("Very Hot"))
                        .when(pl.col("value") > 300)
                        .then(pl.lit("Hot"))
                        .when(pl.col("value") > 290)
                        .then(pl.lit("Warm"))
                        .when(pl.col("value") > 280)
                        .then(pl.lit("Mild"))
                        .when(pl.col("value") > 270)
                        .then(pl.lit("Cool"))
                        .otherwise(pl.lit("Cold"))
                        .alias("temp_category"),
                        # Create coordinate-based region strings
                        (
                            pl.lit("Region_")
                            + (pl.col("lat").round(0).cast(pl.Utf8))
                            + pl.lit("_")
                            + (pl.col("lon").round(0).cast(pl.Utf8))
                        ).alias("region_code"),
                    ]
                )
                .group_by(["month_name", "temp_category", "region_code"])
                .agg(
                    [
                        pl.col("temp_c").mean().alias("avg_temp"),
                        pl.count().alias("count"),
                        pl.col("temp_c").std().alias("temp_variability"),
                    ]
                )
                .filter(pl.col("count") > 5)
                .with_columns(
                    [
                        # String operations on results
                        (
                            pl.lit("Summary: ")
                            + pl.col("month_name")
                            + pl.lit(" in ")
                            + pl.col("region_code")
                            + pl.lit(" is typically ")
                            + pl.col("temp_category")
                        ).alias("description"),
                        # Categorical ranking
                        pl.col("avg_temp").rank().alias("temp_rank"),
                    ]
                )
                .sort(["month_name", "temp_rank"])
            )

        cat_p, cat_time_p, _, cat_mem_p = time_operation(polars_categorical_ops)
        print(
            f"   Polars categorical ops: {cat_time_p:.4f}s | {cat_mem_p:.1f}MB | Shape: {cat_p.shape}"
        )
        print(
            f"   Created {cat_p.shape[0]} categorized groups with string descriptions"
        )

        # XArray doesn't handle string/categorical operations well
        def xarray_limited_ops():
            ds = xr.open_zarr(self.store_path, storage_options={})
            subset = (
                ds["tasmax"]
                .isel(time=slice(0, 12), lat=slice(100, 150), lon=slice(200, 250))
                .load()
            )

            # Basic temperature categorization (much more limited)
            temp_c = subset - 273.15
            hot_mask = subset > 300
            cold_mask = subset < 280

            return {
                "hot_locations": hot_mask.sum().values.item(),
                "cold_locations": cold_mask.sum().values.item(),
                "total_points": subset.size,
                "monthly_means": temp_c.groupby("time").mean().values.tolist(),
            }

        limited_x, limited_time_x, _, limited_mem_x = time_operation(xarray_limited_ops)
        print(
            f"   XArray limited ops: {limited_time_x:.4f}s | {limited_mem_x:.1f}MB | Basic categorization only"
        )

        results["categorical"] = {
            "polars": {
                "time": cat_time_p,
                "memory": cat_mem_p,
                "shape": cat_p.shape,
                "groups": cat_p.shape[0],
            },
            "xarray": {
                "time": limited_time_x,
                "memory": limited_mem_x,
                "note": "Very limited string/categorical support",
            },
        }

        return results

    def run_targeted_benchmark(self) -> Dict[str, Any]:
        """Run the targeted benchmark focusing on Polars' strengths."""
        print("TARGETED PERFORMANCE BENCHMARK - POLARS' STRENGTHS")
        print("=" * 80)
        print(f"Dataset: {self.store_path}")
        print(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f}GB")

        start_time = time.time()

        try:
            # 1. Data Pipeline Operations
            pipeline_results, df_p, da_x = self.benchmark_data_pipeline_operations()
            self.results["pipeline"] = pipeline_results

            # 2. Analytical Queries
            query_results = self.benchmark_analytical_queries(df_p, da_x)
            self.results["queries"] = query_results

            # 3. Data Joining Operations
            join_results = self.benchmark_data_joining_operations()
            self.results["joins"] = join_results

            # 4. String/Categorical Operations
            cat_results = self.benchmark_string_and_categorical_operations()
            self.results["categorical"] = cat_results

        except Exception as e:
            print(f"Error during benchmark: {e}")
            self.results["error"] = str(e)
            import traceback

            traceback.print_exc()

        total_time = time.time() - start_time
        self.results["total_benchmark_time"] = total_time

        return self.results

    def print_summary(self):
        """Print a summary of targeted benchmark results."""
        print("\n" + "=" * 80)
        print("TARGETED BENCHMARK SUMMARY - WHERE POLARS EXCELS")
        print("=" * 80)

        if "error" in self.results:
            print(f"Benchmark failed with error: {self.results['error']}")
            return

        # Pipeline operations
        if "pipeline" in self.results:
            pipeline = self.results["pipeline"]
            print(f"\n🔄 DATA PIPELINE OPERATIONS:")

            if "loading" in pipeline:
                load = pipeline["loading"]
                print(f"   Data Loading:")
                print(
                    f"     Polars: {load['polars']['time']:.4f}s | {load['polars']['memory']:.1f}MB"
                )
                print(
                    f"     XArray: {load['xarray']['time']:.4f}s | {load['xarray']['memory']:.1f}MB"
                )

                if load["polars"]["time"] < load["xarray"]["time"]:
                    print(
                        f"     ✅ Polars {load['xarray']['time']/load['polars']['time']:.1f}x faster"
                    )
                else:
                    print(
                        f"     ❌ XArray {load['polars']['time']/load['xarray']['time']:.1f}x faster"
                    )

            if "complex_chain" in pipeline:
                chain = pipeline["complex_chain"]
                print(f"   Complex Chained Operations:")
                print(
                    f"     Polars: {chain['polars']['time']:.4f}s | Result: {chain['polars']['shape']}"
                )
                print(
                    f"     XArray: {chain['xarray']['time']:.4f}s | Result: basic stats only"
                )
                print(f"     ✅ Polars provides much richer analytical capabilities")

        # Query operations
        if "queries" in self.results:
            queries = self.results["queries"]
            print(f"\n🔍 ANALYTICAL QUERIES:")

            if "conditional_agg" in queries:
                agg = queries["conditional_agg"]
                print(f"   Conditional Aggregations:")
                print(f"     Polars: {agg['polars']['time']:.4f}s")
                print(f"     XArray: {agg['xarray']['time']:.4f}s")

                if agg["polars"]["time"] < agg["xarray"]["time"]:
                    print(
                        f"     ✅ Polars {agg['xarray']['time']/agg['polars']['time']:.1f}x faster"
                    )
                else:
                    print(
                        f"     ❌ XArray {agg['polars']['time']/agg['xarray']['time']:.1f}x faster"
                    )

            if "rolling" in queries:
                roll = queries["rolling"]
                print(f"   Rolling Window Operations:")
                print(f"     Polars: {roll['polars']['time']:.4f}s")
                print(f"     XArray: {roll['xarray']['time']:.4f}s")

                if roll["polars"]["time"] < roll["xarray"]["time"]:
                    print(
                        f"     ✅ Polars {roll['xarray']['time']/roll['polars']['time']:.1f}x faster"
                    )
                else:
                    print(
                        f"     ❌ XArray {roll['polars']['time']/roll['xarray']['time']:.1f}x faster"
                    )

        # Join operations
        if "joins" in self.results:
            joins = self.results["joins"]
            print(f"\n🔗 DATA JOINING OPERATIONS:")

            if "complex_join" in joins:
                join = joins["complex_join"]
                print(f"   Complex Joins with Aggregation:")
                print(
                    f"     Polars: {join['polars']['time']:.4f}s | Result: {join['polars']['shape']}"
                )
                print(
                    f"     XArray: {join['xarray']['time']:.4f}s | {join['xarray']['note']}"
                )
                print(f"     ✅ Polars has native, powerful joining capabilities")

        # Categorical operations
        if "categorical" in self.results:
            cat = self.results["categorical"]
            print(f"\n🏷️  STRING/CATEGORICAL OPERATIONS:")

            if "categorical" in cat:
                cat_ops = cat["categorical"]
                print(f"   String/Categorical Processing:")
                print(
                    f"     Polars: {cat_ops['polars']['time']:.4f}s | {cat_ops['polars']['groups']} groups"
                )
                print(
                    f"     XArray: {cat_ops['xarray']['time']:.4f}s | {cat_ops['xarray']['note']}"
                )
                print(f"     ✅ Polars has superior string and categorical handling")

        print(
            f"\n⏱️  Total benchmark time: {self.results.get('total_benchmark_time', 0):.2f}s"
        )

        print(f"\n📊 KEY INSIGHTS:")
        print(f"   • Polars excels at complex data transformation pipelines")
        print(
            f"   • Polars has native support for joins, string ops, and categorical data"
        )
        print(f"   • XArray is better for dimension-aware scientific computing")
        print(f"   • XArray has simpler syntax for spatial/temporal operations")
        print(f"   • Choose Polars for ETL, data engineering, and complex queries")
        print(f"   • Choose XArray for scientific analysis and visualization")


def main():
    """Run the targeted benchmark."""
    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    benchmark = TargetedBenchmark(store_path)
    results = benchmark.run_targeted_benchmark()
    benchmark.print_summary()

    # Save results
    with open("targeted_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: targeted_benchmark_results.json")


if __name__ == "__main__":
    main()
