#!/usr/bin/env python3
"""
Comprehensive speed test comparing Polars vs XArray for climate data operations.
Tests various operations including data loading, filtering, aggregations, and streaming.
"""

import gc
import json
import time
from typing import Any, Dict, Tuple

import numpy as np
import polars as pl
import psutil
import xarray as xr

# Import our refactored modules
from climakitae.new_core import ClimateDataReader, scan_climate_data


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


class PerformanceBenchmark:
    """Comprehensive performance benchmark for Polars vs XArray."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.results = {"polars": {}, "xarray": {}, "comparison": {}}

    def benchmark_data_loading(self) -> Dict[str, Any]:
        """Benchmark basic data loading operations."""
        print("\n" + "=" * 60)
        print("BENCHMARKING DATA LOADING")
        print("=" * 60)

        results = {}

        # Test 1: Initialize readers
        print("\n1. Reader Initialization")

        # Polars reader initialization
        polars_reader, init_time_p, init_mem_p, init_delta_p = time_operation(
            ClimateDataReader, self.store_path
        )
        print(
            f"   Polars reader init: {init_time_p:.4f}s, Memory: {init_delta_p:.1f}MB"
        )

        # XArray initialization (open dataset)
        def open_xarray():
            return xr.open_zarr(self.store_path, storage_options={})

        xr_dataset, init_time_x, init_mem_x, init_delta_x = time_operation(open_xarray)
        print(
            f"   XArray dataset open: {init_time_x:.4f}s, Memory: {init_delta_x:.1f}MB"
        )

        results["initialization"] = {
            "polars": {"time": init_time_p, "memory": init_delta_p},
            "xarray": {"time": init_time_x, "memory": init_delta_x},
        }

        # Test 2: List available arrays/variables
        print("\n2. Listing Arrays/Variables")

        arrays_p, list_time_p, list_mem_p, list_delta_p = time_operation(
            polars_reader.list_arrays
        )
        print(
            f"   Polars list arrays: {list_time_p:.4f}s, Found: {len(arrays_p)} arrays"
        )

        def list_xarray_vars():
            return list(xr_dataset.data_vars.keys()) + list(xr_dataset.coords.keys())

        vars_x, list_time_x, list_mem_x, list_delta_x = time_operation(list_xarray_vars)
        print(
            f"   XArray list vars: {list_time_x:.4f}s, Found: {len(vars_x)} variables"
        )

        results["listing"] = {
            "polars": {"time": list_time_p, "count": len(arrays_p)},
            "xarray": {"time": list_time_x, "count": len(vars_x)},
        }

        # Test 3: Get array/variable info
        print("\n3. Getting Array/Variable Info")

        info_p, info_time_p, info_mem_p, info_delta_p = time_operation(
            polars_reader.get_array_info, "tasmax"
        )
        print(f"   Polars array info: {info_time_p:.4f}s, Shape: {info_p['shape']}")

        def get_xarray_info():
            var = xr_dataset["tasmax"]
            return {
                "shape": var.shape,
                "dtype": str(var.dtype),
                "dims": list(var.dims),
                "attrs": dict(var.attrs),
            }

        info_x, info_time_x, info_mem_x, info_delta_x = time_operation(get_xarray_info)
        print(f"   XArray var info: {info_time_x:.4f}s, Shape: {info_x['shape']}")

        results["info"] = {
            "polars": {"time": info_time_p, "shape": info_p["shape"]},
            "xarray": {"time": info_time_x, "shape": info_x["shape"]},
        }

        return results, polars_reader, xr_dataset

    def benchmark_small_data_reading(self, polars_reader, xr_dataset) -> Dict[str, Any]:
        """Benchmark reading small data subsets."""
        print("\n" + "=" * 60)
        print("BENCHMARKING SMALL DATA READING")
        print("=" * 60)

        results = {}

        # Test 1: Read single time slice
        print("\n1. Single Time Slice (2D spatial data)")

        def read_polars_slice():
            return polars_reader.read_array(
                "tasmax", select_dims={"time": 0}, streaming=False  # Single time step
            ).collect()

        df_p, read_time_p, read_mem_p, read_delta_p = time_operation(read_polars_slice)
        print(
            f"   Polars single slice: {read_time_p:.4f}s, Shape: {df_p.shape}, Memory: {read_delta_p:.1f}MB"
        )

        def read_xarray_slice():
            return xr_dataset["tasmax"].isel(time=0).load()

        da_x, read_time_x, read_mem_x, read_delta_x = time_operation(read_xarray_slice)
        print(
            f"   XArray single slice: {read_time_x:.4f}s, Shape: {da_x.shape}, Memory: {read_delta_x:.1f}MB"
        )

        results["single_slice"] = {
            "polars": {
                "time": read_time_p,
                "memory": read_delta_p,
                "shape": df_p.shape,
            },
            "xarray": {
                "time": read_time_x,
                "memory": read_delta_x,
                "shape": da_x.shape,
            },
        }

        # Test 2: Read time series for single location
        print("\n2. Time Series for Single Location")

        def read_polars_timeseries():
            return polars_reader.read_array(
                "tasmax",
                select_dims={"lat": 100, "lon": 200},  # Single location
                streaming=False,
            ).collect()

        ts_p, ts_time_p, ts_mem_p, ts_delta_p = time_operation(read_polars_timeseries)
        print(
            f"   Polars time series: {ts_time_p:.4f}s, Shape: {ts_p.shape}, Memory: {ts_delta_p:.1f}MB"
        )

        def read_xarray_timeseries():
            return xr_dataset["tasmax"].isel(lat=100, lon=200).load()

        ts_x, ts_time_x, ts_mem_x, ts_delta_x = time_operation(read_xarray_timeseries)
        print(
            f"   XArray time series: {ts_time_x:.4f}s, Shape: {ts_x.shape}, Memory: {ts_delta_x:.1f}MB"
        )

        results["timeseries"] = {
            "polars": {"time": ts_time_p, "memory": ts_delta_p, "shape": ts_p.shape},
            "xarray": {"time": ts_time_x, "memory": ts_delta_x, "shape": ts_x.shape},
        }

        # Test 3: Read small spatial subset
        print("\n3. Small Spatial Subset (50x50 grid, all time)")

        def read_polars_subset():
            return polars_reader.read_array(
                "tasmax",
                select_dims={"lat": slice(100, 150), "lon": slice(200, 250)},
                streaming=False,
            ).collect()

        subset_p, subset_time_p, subset_mem_p, subset_delta_p = time_operation(
            read_polars_subset
        )
        print(
            f"   Polars spatial subset: {subset_time_p:.4f}s, Shape: {subset_p.shape}, Memory: {subset_delta_p:.1f}MB"
        )

        def read_xarray_subset():
            return (
                xr_dataset["tasmax"]
                .isel(lat=slice(100, 150), lon=slice(200, 250))
                .load()
            )

        subset_x, subset_time_x, subset_mem_x, subset_delta_x = time_operation(
            read_xarray_subset
        )
        print(
            f"   XArray spatial subset: {subset_time_x:.4f}s, Shape: {subset_x.shape}, Memory: {subset_delta_x:.1f}MB"
        )

        results["spatial_subset"] = {
            "polars": {
                "time": subset_time_p,
                "memory": subset_delta_p,
                "shape": subset_p.shape,
            },
            "xarray": {
                "time": subset_time_x,
                "memory": subset_delta_x,
                "shape": subset_x.shape,
            },
        }

        return results

    def benchmark_streaming_vs_normal(
        self, polars_reader, xr_dataset
    ) -> Dict[str, Any]:
        """Benchmark streaming vs normal reading for larger datasets."""
        print("\n" + "=" * 60)
        print("BENCHMARKING STREAMING VS NORMAL READING")
        print("=" * 60)

        results = {}

        # Test larger subset - first 12 months, 100x100 spatial region
        print("\n1. Medium Dataset: 12 months, 100x100 spatial")

        # Polars - Normal mode
        def read_polars_normal():
            return polars_reader.read_array(
                "tasmax",
                select_dims={
                    "time": slice(0, 12),
                    "lat": slice(100, 200),
                    "lon": slice(200, 300),
                },
                streaming=False,
            ).collect()

        df_normal, normal_time_p, normal_mem_p, normal_delta_p = time_operation(
            read_polars_normal
        )
        print(
            f"   Polars normal: {normal_time_p:.4f}s, Shape: {df_normal.shape}, Memory: {normal_delta_p:.1f}MB"
        )

        # Polars - Streaming mode
        def read_polars_streaming():
            return polars_reader.read_array(
                "tasmax",
                select_dims={
                    "time": slice(0, 12),
                    "lat": slice(100, 200),
                    "lon": slice(200, 300),
                },
                streaming=True,
            ).collect()

        df_stream, stream_time_p, stream_mem_p, stream_delta_p = time_operation(
            read_polars_streaming
        )
        print(
            f"   Polars streaming: {stream_time_p:.4f}s, Shape: {df_stream.shape}, Memory: {stream_delta_p:.1f}MB"
        )

        # XArray - equivalent operation
        def read_xarray_medium():
            return (
                xr_dataset["tasmax"]
                .isel(time=slice(0, 12), lat=slice(100, 200), lon=slice(200, 300))
                .load()
            )

        da_medium, medium_time_x, medium_mem_x, medium_delta_x = time_operation(
            read_xarray_medium
        )
        print(
            f"   XArray medium: {medium_time_x:.4f}s, Shape: {da_medium.shape}, Memory: {medium_delta_x:.1f}MB"
        )

        results["medium_dataset"] = {
            "polars_normal": {"time": normal_time_p, "memory": normal_delta_p},
            "polars_streaming": {"time": stream_time_p, "memory": stream_delta_p},
            "xarray": {"time": medium_time_x, "memory": medium_delta_x},
            "shape": df_normal.shape,
        }

        return results

    def benchmark_data_operations(self, polars_reader, xr_dataset) -> Dict[str, Any]:
        """Benchmark common data operations."""
        print("\n" + "=" * 60)
        print("BENCHMARKING DATA OPERATIONS")
        print("=" * 60)

        results = {}

        # Load a medium-sized dataset for operations
        print("\n1. Loading test dataset for operations...")

        # Get first 24 months, 50x50 spatial region
        df_p = polars_reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 24),
                "lat": slice(100, 150),
                "lon": slice(200, 250),
            },
            streaming=False,
        ).collect()

        da_x = (
            xr_dataset["tasmax"]
            .isel(time=slice(0, 24), lat=slice(100, 150), lon=slice(200, 250))
            .load()
        )

        print(f"   Test dataset shape: {df_p.shape} (Polars), {da_x.shape} (XArray)")

        # Test 1: Calculate mean temperature
        print("\n2. Calculate Mean Temperature")

        def polars_mean():
            return df_p.select(pl.col("value").mean()).item()

        mean_p, mean_time_p, _, _ = time_operation(polars_mean)
        print(f"   Polars mean: {mean_time_p:.4f}s, Result: {mean_p:.3f}")

        def xarray_mean():
            return float(da_x.mean().values)

        mean_x, mean_time_x, _, _ = time_operation(xarray_mean)
        print(f"   XArray mean: {mean_time_x:.4f}s, Result: {mean_x:.3f}")

        results["mean"] = {
            "polars": {"time": mean_time_p, "result": mean_p},
            "xarray": {"time": mean_time_x, "result": mean_x},
        }

        # Test 2: Filter values above threshold
        print("\n3. Filter Values Above Threshold (300K)")

        def polars_filter():
            return df_p.filter(pl.col("value") > 300).shape[0]

        count_p, filter_time_p, _, _ = time_operation(polars_filter)
        print(f"   Polars filter: {filter_time_p:.4f}s, Count: {count_p}")

        def xarray_filter():
            return int((da_x > 300).sum().values)

        count_x, filter_time_x, _, _ = time_operation(xarray_filter)
        print(f"   XArray filter: {filter_time_x:.4f}s, Count: {count_x}")

        results["filter"] = {
            "polars": {"time": filter_time_p, "count": count_p},
            "xarray": {"time": filter_time_x, "count": count_x},
        }

        # Test 3: Group by time and calculate spatial mean
        print("\n4. Group by Time and Calculate Spatial Mean")

        def polars_groupby():
            return df_p.group_by("time").agg(pl.col("value").mean()).sort("time")

        grouped_p, group_time_p, _, _ = time_operation(polars_groupby)
        print(
            f"   Polars group by time: {group_time_p:.4f}s, Groups: {grouped_p.shape[0]}"
        )

        def xarray_groupby():
            return da_x.mean(dim=["lat", "lon"])

        grouped_x, group_time_x, _, _ = time_operation(xarray_groupby)
        print(
            f"   XArray mean over space: {group_time_x:.4f}s, Shape: {grouped_x.shape}"
        )

        results["groupby"] = {
            "polars": {"time": group_time_p, "groups": grouped_p.shape[0]},
            "xarray": {"time": group_time_x, "shape": grouped_x.shape},
        }

        # Test 4: Calculate percentiles
        print("\n5. Calculate Percentiles (25th, 50th, 75th)")

        def polars_quantiles():
            return df_p.select(
                [
                    pl.col("value").quantile(0.25).alias("q25"),
                    pl.col("value").quantile(0.50).alias("q50"),
                    pl.col("value").quantile(0.75).alias("q75"),
                ]
            ).to_dicts()[0]

        quantiles_p, quant_time_p, _, _ = time_operation(polars_quantiles)
        print(f"   Polars quantiles: {quant_time_p:.4f}s")

        def xarray_quantiles():
            return {
                "q25": float(da_x.quantile(0.25).values),
                "q50": float(da_x.quantile(0.50).values),
                "q75": float(da_x.quantile(0.75).values),
            }

        quantiles_x, quant_time_x, _, _ = time_operation(xarray_quantiles)
        print(f"   XArray quantiles: {quant_time_x:.4f}s")

        results["quantiles"] = {
            "polars": {"time": quant_time_p, "values": quantiles_p},
            "xarray": {"time": quant_time_x, "values": quantiles_x},
        }

        return results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("Starting Comprehensive Performance Benchmark")
        print("=" * 80)

        start_time = time.time()

        try:
            # 1. Data Loading Benchmark
            loading_results, polars_reader, xr_dataset = self.benchmark_data_loading()
            self.results["loading"] = loading_results

            # 2. Small Data Reading Benchmark
            small_data_results = self.benchmark_small_data_reading(
                polars_reader, xr_dataset
            )
            self.results["small_data"] = small_data_results

            # 3. Streaming vs Normal Benchmark
            streaming_results = self.benchmark_streaming_vs_normal(
                polars_reader, xr_dataset
            )
            self.results["streaming"] = streaming_results

            # 4. Data Operations Benchmark
            operations_results = self.benchmark_data_operations(
                polars_reader, xr_dataset
            )
            self.results["operations"] = operations_results

            # Close resources
            xr_dataset.close()

        except Exception as e:
            print(f"Error during benchmark: {e}")
            self.results["error"] = str(e)

        total_time = time.time() - start_time
        self.results["total_benchmark_time"] = total_time

        return self.results

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        if "error" in self.results:
            print(f"Benchmark failed with error: {self.results['error']}")
            return

        # Loading summary
        if "loading" in self.results:
            loading = self.results["loading"]
            print(f"\n📊 LOADING PERFORMANCE:")
            print(
                f"   Initialization - Polars: {loading['initialization']['polars']['time']:.4f}s, XArray: {loading['initialization']['xarray']['time']:.4f}s"
            )
            print(
                f"   Array Listing - Polars: {loading['listing']['polars']['time']:.4f}s, XArray: {loading['listing']['xarray']['time']:.4f}s"
            )
            print(
                f"   Array Info    - Polars: {loading['info']['polars']['time']:.4f}s, XArray: {loading['info']['xarray']['time']:.4f}s"
            )

        # Small data summary
        if "small_data" in self.results:
            small = self.results["small_data"]
            print(f"\n📈 SMALL DATA READING:")
            print(
                f"   Single Slice  - Polars: {small['single_slice']['polars']['time']:.4f}s, XArray: {small['single_slice']['xarray']['time']:.4f}s"
            )
            print(
                f"   Time Series   - Polars: {small['timeseries']['polars']['time']:.4f}s, XArray: {small['timeseries']['xarray']['time']:.4f}s"
            )
            print(
                f"   Spatial Subset- Polars: {small['spatial_subset']['polars']['time']:.4f}s, XArray: {small['spatial_subset']['xarray']['time']:.4f}s"
            )

        # Streaming summary
        if "streaming" in self.results:
            stream = self.results["streaming"]["medium_dataset"]
            print(f"\n🚀 STREAMING vs NORMAL:")
            print(f"   Polars Normal    : {stream['polars_normal']['time']:.4f}s")
            print(f"   Polars Streaming : {stream['polars_streaming']['time']:.4f}s")
            print(f"   XArray           : {stream['xarray']['time']:.4f}s")

        # Operations summary
        if "operations" in self.results:
            ops = self.results["operations"]
            print(f"\n⚡ DATA OPERATIONS:")
            print(
                f"   Mean Calculation - Polars: {ops['mean']['polars']['time']:.4f}s, XArray: {ops['mean']['xarray']['time']:.4f}s"
            )
            print(
                f"   Filtering        - Polars: {ops['filter']['polars']['time']:.4f}s, XArray: {ops['filter']['xarray']['time']:.4f}s"
            )
            print(
                f"   Group By         - Polars: {ops['groupby']['polars']['time']:.4f}s, XArray: {ops['groupby']['xarray']['time']:.4f}s"
            )
            print(
                f"   Quantiles        - Polars: {ops['quantiles']['polars']['time']:.4f}s, XArray: {ops['quantiles']['xarray']['time']:.4f}s"
            )

        print(
            f"\n⏱️  Total benchmark time: {self.results.get('total_benchmark_time', 0):.2f}s"
        )


def main():
    """Run the comprehensive benchmark."""
    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    print("Comprehensive Speed Test: Polars vs XArray")
    print("==========================================")
    print(f"Dataset: {store_path}")
    print(f"System Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f}GB")

    benchmark = PerformanceBenchmark(store_path)
    results = benchmark.run_comprehensive_benchmark()
    benchmark.print_summary()

    # Save results
    with open("comprehensive_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: comprehensive_benchmark_results.json")


if __name__ == "__main__":
    main()
