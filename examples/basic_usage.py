#!/usr/bin/env python3
"""
CAE-Polars Example: Reading Climate Data from Zarr

This example demonstrates how to use CAE-Polars to read climate data
from Zarr files stored on S3 and process it with Polars.

Features demonstrated:
- Basic data reading
- Dimension selection and filtering
- Data transformation and analysis
- Memory-efficient streaming
- Working with multiple arrays
"""

import sys
from pathlib import Path

import polars as pl

# Add the src directory to the path for this example
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_access import ZarrDataReader, get_zarr_data_info, scan_data


def basic_example():
    """Basic example of reading a Zarr array."""
    print("=" * 60)
    print("Basic Example: Reading Temperature Data")
    print("=" * 60)

    # Public climate dataset (LOCA2)
    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    try:
        # Read a larger subset to show more comprehensive data coverage
        lf = scan_data(
            store_path,
            array_name="tasmax",
            storage_options={"anon": True},  # Anonymous access for public data
            select_dims={
                "time": slice(
                    0, 36
                ),  # First 3 years of data for more temporal coverage
                "lat": slice(150, 250),  # Expanded western US region
                "lon": slice(350, 450),  # Expanded western US region
            },
        )

        # Collect and display basic info
        df = lf.collect()
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print(f"Data types: {df.dtypes}")

        # Check for null/NaN values
        null_count = df.filter(pl.col("value").is_null()).height
        nan_count = df.filter(pl.col("value").is_nan()).height
        valid_count = df.filter(
            pl.col("value").is_not_null() & pl.col("value").is_not_nan()
        ).height

        print(f"\nData quality check:")
        print(f"Total rows: {df.height}")
        print(f"Null values: {null_count}")
        print(f"NaN values: {nan_count}")
        print(f"Valid values: {valid_count}")
        print(f"Data completeness: {valid_count/df.height*100:.1f}%")

        # Filter to only valid data for analysis
        df_valid = df.filter(
            pl.col("value").is_not_null() & pl.col("value").is_not_nan()
        )

        print(f"\nFirst 5 rows (valid data only):")
        print(df_valid.head())

        # Convert temperature from Kelvin to Celsius for valid data
        df_celsius = df_valid.with_columns(
            [(pl.col("value") - 273.15).alias("temp_celsius")]
        )

        if df_celsius.height > 0:
            print(f"\nTemperature range (°C) for valid data:")
            print(f"Min: {df_celsius['temp_celsius'].min():.2f}")
            print(f"Max: {df_celsius['temp_celsius'].max():.2f}")
            print(f"Mean: {df_celsius['temp_celsius'].mean():.2f}")
            print(f"Std Dev: {df_celsius['temp_celsius'].std():.2f}")

            # Show coordinate ranges for valid data
            print(f"\nGeographic extent of valid data:")
            print(
                f"Latitude range: {df_celsius['lat'].min():.3f} to {df_celsius['lat'].max():.3f}"
            )
            print(
                f"Longitude range: {df_celsius['lon'].min():.3f} to {df_celsius['lon'].max():.3f}"
            )

            # Show temporal coverage
            print(
                f"Time range: {df_celsius['time'].min()} to {df_celsius['time'].max()} (months)"
            )

            # Show some interesting statistics
            percentiles = df_celsius.select(
                [
                    pl.col("temp_celsius").quantile(0.05).alias("p5"),
                    pl.col("temp_celsius").quantile(0.25).alias("p25"),
                    pl.col("temp_celsius").quantile(0.50).alias("p50"),
                    pl.col("temp_celsius").quantile(0.75).alias("p75"),
                    pl.col("temp_celsius").quantile(0.95).alias("p95"),
                ]
            ).collect()

            print(f"\nTemperature percentiles (°C):")
            print(f"  5th percentile: {percentiles['p5'][0]:.2f}")
            print(f" 25th percentile: {percentiles['p25'][0]:.2f}")
            print(f" 50th percentile: {percentiles['p50'][0]:.2f}")
            print(f" 75th percentile: {percentiles['p75'][0]:.2f}")
            print(f" 95th percentile: {percentiles['p95'][0]:.2f}")

            # Show monthly temperature variation
            monthly_stats = (
                df_celsius.with_columns([(pl.col("time") % 12 + 1).alias("month")])
                .group_by("month")
                .agg(
                    [
                        pl.col("temp_celsius").mean().alias("avg_temp"),
                        pl.col("temp_celsius").std().alias("std_temp"),
                        pl.len().alias("count"),
                    ]
                )
                .sort("month")
            )

            print(f"\nMonthly temperature summary:")
            print(monthly_stats)
        else:
            print(
                "\nNo valid temperature data found in this region. Trying a different area..."
            )

            # Try a different region - central US
            lf2 = scan_data(
                store_path,
                array_name="tasmax",
                storage_options={"anon": True},
                select_dims={
                    "time": slice(0, 12),
                    "lat": slice(300, 350),  # Central US
                    "lon": slice(350, 400),
                },
            )

            df2 = lf2.collect()
            df2_valid = df2.filter(
                pl.col("value").is_not_null() & pl.col("value").is_not_nan()
            )

            if df2_valid.height > 0:
                df2_celsius = df2_valid.with_columns(
                    [(pl.col("value") - 273.15).alias("temp_celsius")]
                )
                print(
                    f"Found {df2_valid.height} valid data points in central US region"
                )
                print(
                    f"Temperature range: {df2_celsius['temp_celsius'].min():.2f}°C to {df2_celsius['temp_celsius'].max():.2f}°C"
                )
            else:
                print(
                    "Still no valid data found. The dataset may have different coordinate conventions."
                )

    except Exception as e:
        print(f"Error in basic example: {e}")
        print("Note: This example requires internet access to public S3 data")


def advanced_analysis_example():
    """Advanced example with complex data analysis."""
    print("\n" + "=" * 60)
    print("Advanced Example: Climate Data Analysis")
    print("=" * 60)

    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    try:
        # Create a reader for more advanced operations
        reader = ZarrDataReader(store_path, storage_options={"anon": True})

        # Read a larger subset with streaming - focus on land areas
        lf = reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 120),  # 10 years of monthly data
                "lat": slice(250, 300),  # US midwest/plains region
                "lon": slice(350, 400),  # US midwest/plains region
            },
            streaming=True,
        )

        # Perform complex analysis with lazy evaluation, properly handling NaN/null values
        analysis = (
            lf.filter(
                pl.col("value").is_not_null() & pl.col("value").is_not_nan()
            )  # Filter invalid data first
            .with_columns(
                [
                    # Convert to Celsius
                    (pl.col("value") - 273.15).alias("temp_celsius"),
                    # Extract year and month
                    (pl.col("time") % 12 + 1).alias("month"),
                    (pl.col("time") // 12 + 1).alias("year"),
                ]
            )
            .group_by(["lat", "lon", "month"])
            .agg(
                [
                    pl.col("temp_celsius").mean().alias("monthly_avg_temp"),
                    pl.col("temp_celsius").max().alias("monthly_max_temp"),
                    pl.col("temp_celsius").min().alias("monthly_min_temp"),
                    pl.col("temp_celsius").std().alias("monthly_temp_std"),
                    pl.len().alias(
                        "data_points"
                    ),  # Use pl.len() instead of deprecated pl.count()
                ]
            )
            .collect()
        )

        print(f"Analysis results shape: {analysis.shape}")
        print("\nMonthly temperature statistics:")
        print(analysis.head(10))

        # Find hottest and coldest locations
        hottest = analysis.filter(
            pl.col("monthly_avg_temp") == pl.col("monthly_avg_temp").max()
        ).select(["lat", "lon", "month", "monthly_avg_temp"])

        coldest = analysis.filter(
            pl.col("monthly_avg_temp") == pl.col("monthly_avg_temp").min()
        ).select(["lat", "lon", "month", "monthly_avg_temp"])

        print("\nHottest location/month:")
        print(hottest)
        print("\nColdest location/month:")
        print(coldest)

        # Seasonal analysis
        seasonal = (
            analysis.with_columns(
                [
                    pl.when(pl.col("month").is_in([12, 1, 2]))
                    .then(pl.lit("Winter"))
                    .when(pl.col("month").is_in([3, 4, 5]))
                    .then(pl.lit("Spring"))
                    .when(pl.col("month").is_in([6, 7, 8]))
                    .then(pl.lit("Summer"))
                    .otherwise(pl.lit("Fall"))
                    .alias("season")
                ]
            )
            .group_by("season")
            .agg(
                [
                    pl.col("monthly_avg_temp").mean().alias("seasonal_avg"),
                    pl.col("monthly_max_temp").max().alias("seasonal_max"),
                    pl.col("monthly_min_temp").min().alias("seasonal_min"),
                ]
            )
            .sort("seasonal_avg", descending=True)
        )

        print("\nSeasonal temperature summary:")
        print(seasonal)

    except Exception as e:
        print(f"Error in advanced example: {e}")
        print("Note: This example requires internet access to public S3 data")


def dataset_info_example():
    """Example of getting dataset information."""
    print("\n" + "=" * 60)
    print("Dataset Information Example")
    print("=" * 60)

    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    try:
        # Get information about the dataset
        info = get_zarr_data_info(store_path, storage_options={"anon": True})

        print(f"Store path: {info['store_path']}")
        print(f"Available arrays: {list(info['arrays'].keys())}")

        # Show details for each array
        for array_name, array_info in info["arrays"].items():
            print(f"\nArray: {array_name}")
            print(f"  Shape: {array_info['shape']}")
            print(f"  Data type: {array_info['dtype']}")
            print(f"  Dimensions: {array_info['dimensions']}")
            print(f"  Chunks: {array_info['chunks']}")

            # Show some attributes
            attrs = array_info.get("attrs", {})
            if attrs:
                print("  Attributes:")
                for key, value in list(attrs.items())[:3]:  # Show first 3 attributes
                    print(f"    {key}: {value}")

    except Exception as e:
        print(f"Error getting dataset info: {e}")
        print("Note: This example requires internet access to public S3 data")


def memory_efficient_example():
    """Example of memory-efficient processing with streaming."""
    print("\n" + "=" * 60)
    print("Memory-Efficient Streaming Example")
    print("=" * 60)

    store_path = "s3://cadcat/loca2/ucsd/access-cm2/historical/r2i1p1f1/mon/tasmax/d03/"

    try:
        # Create reader with custom chunk size for memory control
        reader = ZarrDataReader(
            store_path,
            storage_options={"anon": True},
            chunk_size=10000,  # Smaller chunks for demo
        )

        # Read larger dataset with streaming enabled - focus on continental US
        lf = reader.read_array(
            "tasmax",
            select_dims={
                "time": slice(0, 60),  # 5 years of data
                "lat": slice(200, 350),  # Continental US latitude range
                "lon": slice(300, 450),  # Continental US longitude range
            },
            streaming=True,  # Enable streaming for large data
        )

        # Process in chunks without loading everything into memory
        print("Processing data in streaming mode...")

        # Use lazy operations that will be executed efficiently, filtering out invalid data
        summary = (
            lf.filter(
                pl.col("value").is_not_null() & pl.col("value").is_not_nan()
            )  # Filter invalid data
            .with_columns([(pl.col("value") - 273.15).alias("temp_celsius")])
            .select(
                [
                    pl.col("temp_celsius").mean().alias("global_mean"),
                    pl.col("temp_celsius").max().alias("global_max"),
                    pl.col("temp_celsius").min().alias("global_min"),
                    pl.col("temp_celsius").std().alias("global_std"),
                    pl.len().alias(
                        "total_points"
                    ),  # Use pl.len() instead of deprecated pl.count()
                ]
            )
            .collect()
        )

        print("Streaming processing complete!")
        print("Global temperature statistics:")
        print(summary)

        # Show memory-efficient aggregation by region
        regional_stats = (
            lf.filter(
                pl.col("value").is_not_null() & pl.col("value").is_not_nan()
            )  # Filter invalid data
            .with_columns(
                [
                    (pl.col("value") - 273.15).alias("temp_celsius"),
                    # Create regional bins
                    (pl.col("lat") // 10 * 10).alias("lat_bin"),
                    (pl.col("lon") // 10 * 10).alias("lon_bin"),
                ]
            )
            .group_by(["lat_bin", "lon_bin"])
            .agg(
                [
                    pl.col("temp_celsius").mean().alias("regional_avg"),
                    pl.len().alias(
                        "point_count"
                    ),  # Use pl.len() instead of deprecated pl.count()
                ]
            )
            .sort("regional_avg", descending=True)
            .collect()
        )

        print("\nTop 5 warmest regions (10° x 10° bins):")
        print(regional_stats.head(5))

    except Exception as e:
        print(f"Error in streaming example: {e}")
        print("Note: This example requires internet access to public S3 data")


def local_file_example():
    """Example of reading from a local Zarr file."""
    print("\n" + "=" * 60)
    print("Local File Example")
    print("=" * 60)

    # This would work with a local Zarr file
    local_path = "/path/to/your/local/file.zarr"

    print("To read from a local Zarr file:")
    print("")
    print("from data_access import scan_data")
    print("")
    print("# Read from local file")
    print("lf = scan_data(")
    print(f"    '{local_path}',")
    print("    array_name='your_array_name',")
    print("    # No storage_options needed for local files")
    print(")")
    print("")
    print("df = lf.collect()")
    print("print(df.head())")

    print("\nNote: Replace the path and array name with your actual file details.")


def main():
    """Run all examples."""
    print("CAE-Polars Examples")
    print("==================")
    print("Demonstrating high-performance Zarr I/O with Polars")

    # Run examples
    basic_example()
    dataset_info_example()
    advanced_analysis_example()
    memory_efficient_example()
    local_file_example()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("- README.md for installation and usage")
    print("- CONTRIBUTING.md for development setup")
    print("- The data_access module for API details")


if __name__ == "__main__":
    main()
