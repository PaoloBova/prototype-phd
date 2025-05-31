"""
Utilities for exploring and visualizing cost trend data.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore cost trend data")
    parser.add_argument("--trends", required=True, help="Path to cost trends CSV file")
    parser.add_argument("--forecasts", required=False, help="Path to cost forecasts CSV file (if available)")
    parser.add_argument("--output", default="reports/cost_trend_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--format", default="png", help="Output format (png, pdf, svg)")
    return parser.parse_args()

def plot_doubling_rates(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot doubling rates for different models with confidence intervals when available.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=(12, 7))
    
    # Sort by doubling rate for better visualization
    sorted_df = df.sort_values("doubling_rate")
    
    # Check if confidence intervals are available
    has_ci = ("doubling_rate_ci_lower" in sorted_df.columns and 
              "doubling_rate_ci_upper" in sorted_df.columns)
    
    # Calculate error bars if CIs are available
    if has_ci:
        # Filter out rows with missing or invalid CIs
        valid_ci_mask = (
            sorted_df["doubling_rate_ci_lower"].notna() & 
            sorted_df["doubling_rate_ci_upper"].notna() &
            np.isfinite(sorted_df["doubling_rate_ci_lower"]) & 
            np.isfinite(sorted_df["doubling_rate_ci_upper"])
        )
        
        # For rows with valid CIs, calculate error bar heights
        yerr = np.zeros((2, len(sorted_df)))
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            if valid_ci_mask.iloc[i]:
                yerr[0, i] = row["doubling_rate"] - row["doubling_rate_ci_lower"]
                yerr[1, i] = row["doubling_rate_ci_upper"] - row["doubling_rate"]
            else:
                yerr[:, i] = 0
        
        # Create horizontal bar chart with error bars
        plt.barh(sorted_df["model"], sorted_df["doubling_rate"], 
                xerr=yerr, alpha=0.7, capsize=5)
        
        # Add legend for confidence intervals
        plt.plot([], [], '-', color='black', label='95% Confidence Interval')
        plt.legend(loc='lower right')
    else:
        # Create regular bar chart without error bars
        plt.barh(sorted_df["model"], sorted_df["doubling_rate"], alpha=0.7)
    
    plt.xlabel("Doubling Rate (Difficulty Units)")
    plt.ylabel("Model")
    plt.title("Cost Doubling Rates by Model")
    plt.grid(True, axis='x', alpha=0.3)
    
    # Add text labels to the bars
    for i, v in enumerate(sorted_df["doubling_rate"]):
        plt.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"doubling_rates.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved doubling rates plot to {output_path}")

def plot_doubling_rates_vs_r_squared(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot doubling rates against r-squared values.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df["r_squared"], df["doubling_rate"], alpha=0.7)
    
    # Annotate points with model names
    for i, row in df.iterrows():
        plt.annotate(row["model"], 
                    (row["r_squared"], row["doubling_rate"]),
                    xytext=(5, 5),
                    textcoords="offset points")
    
    plt.xlabel("R-squared Value")
    plt.ylabel("Doubling Rate (Difficulty Units)")
    plt.title("Cost Doubling Rates vs. Goodness of Fit")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"doubling_rates_vs_r_squared.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved doubling rates vs. r-squared plot to {output_path}")

def plot_cost_growth_comparison(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot cost growth curves for different models with confidence intervals when available.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=(12, 8))
    
    # Generate difficulty range
    difficulty_range = np.linspace(0, 15, 1000)
    
    # Check if confidence intervals are available
    has_ci = ("doubling_rate_ci_lower" in df.columns and 
              "doubling_rate_ci_upper" in df.columns)
    
    # Plot cost curve for each model
    for _, row in df.iterrows():
        model = row["model"]
        doubling_rate = row["doubling_rate"]
        intercept = row["intercept"]
        
        # Calculate cost for each difficulty
        log2_costs = intercept + (difficulty_range / doubling_rate)
        costs = 2.0 ** log2_costs
        
        # Normalize costs to start at 1.0 for better comparison
        normalized_costs = costs / costs[0]
        
        # Plot the main trend line
        plt.semilogy(difficulty_range, normalized_costs, label=model)
        
        # Add confidence intervals if available
        if has_ci and pd.notna(row.get("doubling_rate_ci_lower")) and pd.notna(row.get("doubling_rate_ci_upper")):
            # Only plot if CIs are finite
            if np.isfinite(row["doubling_rate_ci_lower"]) and np.isfinite(row["doubling_rate_ci_upper"]):
                # Calculate upper bound using lower doubling rate
                log2_costs_upper = intercept + (difficulty_range / row["doubling_rate_ci_lower"])
                costs_upper = 2.0 ** log2_costs_upper
                norm_costs_upper = costs_upper / costs_upper[0]
                
                # Calculate lower bound using upper doubling rate
                log2_costs_lower = intercept + (difficulty_range / row["doubling_rate_ci_upper"])
                costs_lower = 2.0 ** log2_costs_lower
                norm_costs_lower = costs_lower / costs_lower[0]
                
                # Plot confidence interval as shaded area
                plt.fill_between(difficulty_range, norm_costs_lower, norm_costs_upper, 
                                alpha=0.2, label=f"{model} 95% CI")
    
    plt.xlabel("Task Difficulty")
    plt.ylabel("Relative Cost (log scale)")
    plt.title("Cost Growth Comparison with Confidence Intervals")
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if not any(label == l for l in unique_labels):
            unique_labels.append(label)
            unique_handles.append(handle)
    
    plt.legend(unique_handles, unique_labels, loc='upper left')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"cost_growth_comparison.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved cost growth comparison plot to {output_path}")

def plot_cost_forecasts(forecasts_df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot cost forecasts across difficulty levels with confidence intervals when available.
    
    Args:
        forecasts_df: DataFrame with cost forecast data
        output_dir: Directory to save plots
        fmt: File format for output
    """
    if forecasts_df is None or len(forecasts_df) == 0:
        print("No forecast data available, skipping forecast plots")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Check if confidence intervals are available
    has_ci = ("cost_ci_lower" in forecasts_df.columns and 
              "cost_ci_upper" in forecasts_df.columns)
    
    # Group by model for plotting
    for model, group in forecasts_df.groupby("model"):
        sorted_group = group.sort_values("difficulty")
        
        # Plot the main trend line
        plt.plot(sorted_group["difficulty"], sorted_group["forecasted_cost"], 
                label=f"{model}", alpha=0.7)
        
        # Add scatter points
        plt.scatter(sorted_group["difficulty"], sorted_group["forecasted_cost"], 
                   s=20, alpha=0.5)
        
        # Add confidence intervals if available
        if has_ci:
            valid_ci = (
                sorted_group["cost_ci_lower"].notna() & 
                sorted_group["cost_ci_upper"].notna() &
                np.isfinite(sorted_group["cost_ci_lower"]) & 
                np.isfinite(sorted_group["cost_ci_upper"])
            )
            
            if valid_ci.any():
                ci_group = sorted_group[valid_ci]
                plt.fill_between(
                    ci_group["difficulty"], 
                    ci_group["cost_ci_lower"], 
                    ci_group["cost_ci_upper"],
                    alpha=0.2, label=f"{model} 95% CI"
                )
    
    plt.yscale("log")
    plt.xlabel("Task Difficulty")
    plt.ylabel("Forecasted Cost (log scale)")
    plt.title("Cost vs. Task Difficulty with Confidence Intervals")
    plt.grid(True, alpha=0.3)
    
    # Fix legend (remove duplicate entries)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    plt.legend(unique_handles, unique_labels)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"cost_forecasts.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved cost forecasts plot to {output_path}")

def create_summary_stats(df: pd.DataFrame, output_dir: str):
    """
    Create summary statistics for the cost trend data including confidence intervals.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save summary
    """
    # Calculate basic summary statistics
    summary = {
        "mean_doubling_rate": df["doubling_rate"].mean(),
        "median_doubling_rate": df["doubling_rate"].median(),
        "std_doubling_rate": df["doubling_rate"].std(),
        "min_doubling_rate": df["doubling_rate"].min(),
        "max_doubling_rate": df["doubling_rate"].max(),
        "mean_r_squared": df["r_squared"].mean(),
        "median_r_squared": df["r_squared"].median(),
        "min_intercept": df["intercept"].min(),
        "max_intercept": df["intercept"].max()
    }
    
    # Add confidence interval information if available
    has_ci = ("doubling_rate_ci_lower" in df.columns and 
              "doubling_rate_ci_upper" in df.columns)
    
    if has_ci:
        valid_ci = (
            df["doubling_rate_ci_lower"].notna() & 
            df["doubling_rate_ci_upper"].notna() &
            np.isfinite(df["doubling_rate_ci_lower"]) & 
            np.isfinite(df["doubling_rate_ci_upper"])
        )
        
        if valid_ci.any():
            ci_width = df.loc[valid_ci, "doubling_rate_ci_upper"] - df.loc[valid_ci, "doubling_rate_ci_lower"]
            summary["mean_ci_width"] = float(ci_width.mean())
            summary["median_ci_width"] = float(ci_width.median())
            summary["narrowest_ci"] = float(ci_width.min())
            summary["widest_ci"] = float(ci_width.max())
    
    # Save summary to CSV
    output_path = os.path.join(output_dir, "cost_trend_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)
    
    print(f"Saved summary statistics to {output_path}")
    print("\nSummary of cost trend data:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

def plot_doubling_rates_by_aggregation(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot doubling rates for different aggregation methods.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Filter for aggregation methods (models with "agg_" prefix)
    agg_df = df[df["model"].str.contains("agg_")].copy()
    
    if len(agg_df) == 0:
        print("No aggregation methods found in data, skipping aggregation comparison plot")
        return
    
    # Extract method name from model column (remove "agg_" prefix)
    agg_df["method"] = agg_df["model"].str.replace("agg_", "", regex=False)
    
    # Sort by doubling rate for better visualization
    agg_df = agg_df.sort_values("doubling_rate")
    
    plt.figure(figsize=(10, 6))
    
    # Check if confidence intervals are available
    has_ci = ("doubling_rate_ci_lower" in agg_df.columns and 
              "doubling_rate_ci_upper" in agg_df.columns)
    
    # Calculate error bars if CIs are available
    if has_ci:
        # Filter out rows with missing or invalid CIs
        valid_ci_mask = (
            agg_df["doubling_rate_ci_lower"].notna() & 
            agg_df["doubling_rate_ci_upper"].notna() &
            np.isfinite(agg_df["doubling_rate_ci_lower"]) & 
            np.isfinite(agg_df["doubling_rate_ci_upper"])
        )
        
        # For rows with valid CIs, calculate error bar heights
        yerr = np.zeros((2, len(agg_df)))
        for i, (_, row) in enumerate(agg_df.iterrows()):
            if valid_ci_mask.iloc[i]:
                yerr[0, i] = row["doubling_rate"] - row["doubling_rate_ci_lower"]
                yerr[1, i] = row["doubling_rate_ci_upper"] - row["doubling_rate"]
            else:
                yerr[:, i] = 0
        
        # Create horizontal bar chart with error bars
        plt.barh(agg_df["method"], agg_df["doubling_rate"], 
                xerr=yerr, alpha=0.7, capsize=5)
        
        # Add legend for confidence intervals
        plt.plot([], [], '-', color='black', label='95% Confidence Interval')
        plt.legend(loc='lower right')
    else:
        # Create regular bar chart without error bars
        plt.barh(agg_df["method"], agg_df["doubling_rate"], alpha=0.7)
    
    plt.xlabel("Doubling Rate (Difficulty Units)")
    plt.ylabel("Aggregation Method")
    plt.title("Cost Doubling Rates by Aggregation Method")
    plt.grid(True, axis='x', alpha=0.3)
    
    # Add text labels to the bars
    for i, v in enumerate(agg_df["doubling_rate"]):
        plt.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"doubling_rates_by_aggregation.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved doubling rates by aggregation plot to {output_path}")

def plot_cost_trends_by_aggregation(df: pd.DataFrame, output_dir: str, fmt: str = "png", 
                                   include_ci: bool = True):
    """
    Plot cost trends for different aggregation methods.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save plot
        fmt: File format for output
        include_ci: Whether to include confidence intervals
    """
    # Filter for aggregation and reference methods
    agg_df = df[df["model"].str.contains("agg_")].copy()
    
    # Also include aggregate, recency_weighted for reference
    reference_models = ["aggregate", "recency_weighted"]
    reference_df = df[df["model"].isin(reference_models)]
    
    # Combine aggregation methods with reference models
    plot_df = pd.concat([agg_df, reference_df])
    
    if len(plot_df) == 0:
        print("No aggregation methods found in data, skipping aggregation comparison plot")
        return
    
    # Generate difficulty range
    difficulty_range = np.linspace(0, 15, 1000)
    
    plt.figure(figsize=(12, 8))
    
    # Check if confidence intervals are available
    has_ci = include_ci and ("doubling_rate_ci_lower" in plot_df.columns and 
                            "doubling_rate_ci_upper" in plot_df.columns)
    
    # Plot cost curve for each model
    for _, row in plot_df.iterrows():
        model = row["model"]
        doubling_rate = row["doubling_rate"]
        intercept = row["intercept"]
        
        # Calculate cost for each difficulty
        log2_costs = intercept + (difficulty_range / doubling_rate)
        costs = 2.0 ** log2_costs
        
        # Normalize costs to start at 1.0 for better comparison
        normalized_costs = costs / costs[0]
        
        # Plot the main trend line
        plt.semilogy(difficulty_range, normalized_costs, label=model)
        
        # Add confidence intervals if available and requested
        if has_ci and pd.notna(row.get("doubling_rate_ci_lower")) and pd.notna(row.get("doubling_rate_ci_upper")):
            # Only plot if CIs are finite
            if np.isfinite(row["doubling_rate_ci_lower"]) and np.isfinite(row["doubling_rate_ci_upper"]):
                # Calculate upper bound using lower doubling rate
                log2_costs_upper = intercept + (difficulty_range / row["doubling_rate_ci_lower"])
                costs_upper = 2.0 ** log2_costs_upper
                norm_costs_upper = costs_upper / costs_upper[0]
                
                # Calculate lower bound using upper doubling rate
                log2_costs_lower = intercept + (difficulty_range / row["doubling_rate_ci_upper"])
                costs_lower = 2.0 ** log2_costs_lower
                norm_costs_lower = costs_lower / costs_lower[0]
                
                # Plot confidence interval as shaded area
                plt.fill_between(difficulty_range, norm_costs_lower, norm_costs_upper, 
                                alpha=0.2, label=f"{model} 95% CI")
    
    plt.xlabel("Task Difficulty")
    plt.ylabel("Relative Cost (log scale)")
    
    ci_text = "with Confidence Intervals" if has_ci else "without Confidence Intervals"
    plt.title(f"Cost Growth Comparison by Aggregation Method {ci_text}")
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if not any(label == l for l in unique_labels):
            unique_labels.append(label)
            unique_handles.append(handle)
    
    plt.legend(unique_handles, unique_labels, loc='upper left')
    plt.tight_layout()
    
    # Save figure
    filename = "cost_growth_by_aggregation"
    if include_ci:
        filename += "_with_ci"
    else:
        filename += "_without_ci"
        
    output_path = os.path.join(output_dir, f"{filename}.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved cost growth by aggregation plot to {output_path}")

def plot_forecast_comparison_by_aggregation(forecasts_df: pd.DataFrame, output_dir: str, fmt: str = "png",
                                           include_ci: bool = True):
    """
    Plot cost forecasts for different aggregation methods.
    
    Args:
        forecasts_df: DataFrame with forecast data
        output_dir: Directory to save plot
        fmt: File format for output
        include_ci: Whether to include confidence intervals
    """
    if forecasts_df is None or len(forecasts_df) == 0:
        print("No forecast data available, skipping forecast by aggregation plots")
        return
    
    # Filter for aggregation methods and reference models
    agg_models = [m for m in forecasts_df["model"].unique() if "agg_" in m]
    reference_models = ["aggregate", "recency_weighted"]
    plot_models = agg_models + [m for m in reference_models if m in forecasts_df["model"].unique()]
    
    if len(plot_models) == 0:
        print("No aggregation methods found in forecast data, skipping aggregation comparison plot")
        return
    
    # Filter dataframe to only include selected models
    plot_df = forecasts_df[forecasts_df["model"].isin(plot_models)].copy()
    
    plt.figure(figsize=(12, 8))
    
    # Check if confidence intervals are available
    has_ci = include_ci and ("cost_ci_lower" in plot_df.columns and 
                            "cost_ci_upper" in plot_df.columns)
    
    # Group by model for plotting
    for model, group in plot_df.groupby("model"):
        sorted_group = group.sort_values("difficulty")
        
        # Plot the main trend line
        plt.plot(sorted_group["difficulty"], sorted_group["forecasted_cost"], 
                label=f"{model}", alpha=0.7)
        
        # Add confidence intervals if available and requested
        if has_ci:
            valid_ci = (
                sorted_group["cost_ci_lower"].notna() & 
                sorted_group["cost_ci_upper"].notna() &
                np.isfinite(sorted_group["cost_ci_lower"]) & 
                np.isfinite(sorted_group["cost_ci_upper"])
            )
            
            if valid_ci.any():
                ci_group = sorted_group[valid_ci]
                plt.fill_between(
                    ci_group["difficulty"], 
                    ci_group["cost_ci_lower"], 
                    ci_group["cost_ci_upper"],
                    alpha=0.2, label=f"{model} 95% CI"
                )
    
    plt.yscale("log")
    plt.xlabel("Task Difficulty")
    plt.ylabel("Forecasted Cost (log scale)")
    
    ci_text = "with Confidence Intervals" if has_ci else "without Confidence Intervals"
    plt.title(f"Cost Forecasts by Aggregation Method {ci_text}")
    plt.grid(True, alpha=0.3)
    
    # Fix legend (remove duplicate entries)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    plt.legend(unique_handles, unique_labels)
    plt.tight_layout()
    
    # Save figure
    filename = "cost_forecasts_by_aggregation"
    if include_ci:
        filename += "_with_ci"
    else:
        filename += "_without_ci"
        
    output_path = os.path.join(output_dir, f"{filename}.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved cost forecasts by aggregation plot to {output_path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"Loading cost trends from {args.trends}")
    trends_df = pd.read_csv(args.trends)
    print(f"Loaded {len(trends_df)} cost trend records")
    
    # Load forecasts if available
    forecasts_df = None
    if args.forecasts:
        print(f"Loading cost forecasts from {args.forecasts}")
        forecasts_df = pd.read_csv(args.forecasts)
        print(f"Loaded {len(forecasts_df)} cost forecast records")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations for trend data
    plot_doubling_rates(trends_df, args.output, args.format)
    plot_doubling_rates_vs_r_squared(trends_df, args.output, args.format)
    plot_cost_growth_comparison(trends_df, args.output, args.format)
    create_summary_stats(trends_df, args.output)
    
    # Generate new aggregation comparison plots
    plot_doubling_rates_by_aggregation(trends_df, args.output, args.format)
    plot_cost_trends_by_aggregation(trends_df, args.output, args.format, include_ci=True)
    plot_cost_trends_by_aggregation(trends_df, args.output, args.format, include_ci=False)
    
    # Generate visualizations for forecast data if available
    if forecasts_df is not None:
        plot_cost_forecasts(forecasts_df, args.output, args.format)
        plot_forecast_comparison_by_aggregation(forecasts_df, args.output, args.format, include_ci=True)
        plot_forecast_comparison_by_aggregation(forecasts_df, args.output, args.format, include_ci=False)
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
