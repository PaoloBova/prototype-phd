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
    Plot doubling rates for different models.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by doubling rate for better visualization
    sorted_df = df.sort_values("doubling_rate")
    
    # Create horizontal bar chart
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
    Plot cost growth curves for different models for comparison.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=(12, 8))
    
    # Generate difficulty range
    difficulty_range = np.linspace(0, 15, 1000)
    
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
        
        plt.semilogy(difficulty_range, normalized_costs, label=model)
    
    plt.xlabel("Task Difficulty")
    plt.ylabel("Relative Cost (log scale)")
    plt.title("Cost Growth Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"cost_growth_comparison.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved cost growth comparison plot to {output_path}")

def plot_cost_forecasts(forecasts_df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot cost forecasts across difficulty levels.
    
    Args:
        forecasts_df: DataFrame with cost forecast data
        output_dir: Directory to save plots
        fmt: File format for output
    """
    if forecasts_df is None or len(forecasts_df) == 0:
        print("No forecast data available, skipping forecast plots")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Create seaborn scatter plot with trend lines
    ax = sns.scatterplot(data=forecasts_df, x="difficulty", y="forecasted_cost", 
                         hue="model", alpha=0.5)
    
    # Add trend lines
    for model, group in forecasts_df.groupby("model"):
        sorted_group = group.sort_values("difficulty")
        ax.plot(sorted_group["difficulty"], sorted_group["forecasted_cost"], 
                label=f"{model} trend", alpha=0.7)
    
    plt.yscale("log")
    plt.xlabel("Task Difficulty")
    plt.ylabel("Forecasted Cost (log scale)")
    plt.title("Cost vs. Task Difficulty")
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
    Create summary statistics for the cost trend data.
    
    Args:
        df: DataFrame with cost trend data
        output_dir: Directory to save summary
    """
    # Calculate summary statistics
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
    
    # Save summary to CSV
    output_path = os.path.join(output_dir, "cost_trend_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)
    
    print(f"Saved summary statistics to {output_path}")
    print("\nSummary of cost trend data:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

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
    
    # Generate visualizations for forecast data if available
    if forecasts_df is not None:
        plot_cost_forecasts(forecasts_df, args.output, args.format)
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
