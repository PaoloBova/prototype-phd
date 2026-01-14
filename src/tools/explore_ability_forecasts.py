"""
Utilities for exploring and visualizing ability forecasts.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Global plotting style parameters
FONT_SIZE = 12  # Default font size
TITLE_ENABLED = True  # Whether to show titles in plots

def _style_plots():
    """Apply font size settings and optionally remove titles from plots."""
    try:
        mpl.rcParams.update({
            'axes.labelsize': FONT_SIZE,
            # Force titles to be max 10pt (titles only used for debugging so styling should optimize for long debug titles)
            'axes.titlesize': min(FONT_SIZE, 10),
            'xtick.labelsize': FONT_SIZE * 0.8,
            'ytick.labelsize': FONT_SIZE * 0.8,
            'legend.fontsize': FONT_SIZE * 0.8
        })
        
        if not TITLE_ENABLED:
            # Remove titles from the current figure
            fig = plt.gcf()
            fig.suptitle("")  # Remove figure suptitle
            for ax in fig.axes:
                ax.set_title("")  # Remove axis title
    except Exception as e:
        print(f"Warning: Error applying plot styles: {e}")
        # Continue anyway - don't let styling issues break the plotting

def get_figure_size(base_width: float, base_height: float) -> tuple:
    """
    Get appropriate figure size based on whether titles are enabled.
    When titles are enabled, double the height to accommodate multi-line titles.
    
    Args:
        base_width: Base width in inches
        base_height: Base height in inches
        
    Returns:
        Tuple of (width, height) for figure size
    """
    if TITLE_ENABLED:
        return (base_width * 1.5, base_height * 2.0)
    else:
        return (base_width, base_height)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore ability forecast data")
    parser.add_argument("--input", required=True, help="Path to ability forecasts CSV file")
    parser.add_argument("--output", default="reports/ability_forecast_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--format", default="png", help="Output format (png, pdf, svg)")
    parser.add_argument("--font-size", type=int, default=12,
                        help="Base font size for all plot text")
    parser.add_argument("--disable-titles", action="store_true",
                        help="Strip all titles from plots for publication style")
    return parser.parse_args()

def logistic_function(x: np.ndarray, threshold: float, slope: float) -> np.ndarray:
    """Calculate logistic function values."""
    return 1.0 / (1.0 + np.exp(-slope * (x - threshold)))

def plot_threshold_timeline(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot threshold parameters over time for different forecast scenarios.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=get_figure_size(10, 6))
    
    # Group by scenario and plot each as a separate line
    for scenario, group in df.groupby("scenario"):
        # Sort by date
        group = group.sort_values("date")
        plt.plot(group["date"], group["threshold"], 
                 marker="o", linestyle="-", label=scenario)
        
        # Plot confidence intervals if available
        if "threshold_ci_lower" in group.columns and "threshold_ci_upper" in group.columns:
            has_ci = group["threshold_ci_lower"].notna() & group["threshold_ci_upper"].notna()
            if has_ci.any():
                ci_group = group[has_ci]
                plt.fill_between(ci_group["date"], 
                                ci_group["threshold_ci_lower"], 
                                ci_group["threshold_ci_upper"],
                                alpha=0.2)
    
    plt.xlabel("Date")
    plt.ylabel("Threshold Parameter")
    plt.title("Forecasted Threshold Parameters Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format x-axis dates nicely
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"threshold_timeline.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved threshold timeline to {output_path}")

def plot_slope_timeline(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot slope parameters over time for different forecast scenarios.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=get_figure_size(10, 6))
    
    # Group by scenario and plot each as a separate line
    for scenario, group in df.groupby("scenario"):
        # Sort by date
        group = group.sort_values("date")
        plt.plot(group["date"], group["slope"], 
                 marker="o", linestyle="-", label=scenario)
        
        # Plot confidence intervals if available
        if "slope_ci_lower" in group.columns and "slope_ci_upper" in group.columns:
            has_ci = group["slope_ci_lower"].notna() & group["slope_ci_upper"].notna()
            if has_ci.any():
                ci_group = group[has_ci]
                plt.fill_between(ci_group["date"], 
                                ci_group["slope_ci_lower"], 
                                ci_group["slope_ci_upper"],
                                alpha=0.2)
    
    plt.xlabel("Date")
    plt.ylabel("Slope Parameter")
    plt.title("Forecasted Slope Parameters Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format x-axis dates nicely
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"slope_timeline.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved slope timeline to {output_path}")

def plot_logistic_curves_grid(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot a grid of logistic curves for different dates and scenarios.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save plots
        fmt: File format for output
    """
    # Group by scenario first
    for scenario, scenario_df in df.groupby("scenario"):
        # Sample dates specific to this scenario
        scenario_dates = sorted(scenario_df["date"].unique())
        
        if len(scenario_dates) == 0:
            print(f"No dates found for scenario {scenario}, skipping plot")
            continue
            
        n_dates = min(6, len(scenario_dates))
        
        # Ensure we have enough dates to create a meaningful grid
        if n_dates < 1:
            print(f"Too few dates ({n_dates}) for scenario {scenario}, skipping plot")
            continue
            
        # Select evenly spaced dates from this scenario's date range
        if n_dates == 1:
            sample_dates = scenario_dates
        else:
            # Ensure we don't divide by zero
            step = max(1, len(scenario_dates) // n_dates)
            sample_dates = scenario_dates[::step][:n_dates]
        
        # Create a grid of plots for this scenario
        n_cols = min(3, n_dates)
        n_rows = (n_dates + n_cols - 1) // n_cols
        
        plt.figure(figsize=get_figure_size(5*n_cols, 4*n_rows))
        
        for i, date in enumerate(sample_dates, 1):
            date_df = scenario_df[scenario_df["date"] == date]
            
            if len(date_df) == 0:
                print(f"Warning: No data found for {scenario} on date {date}")
                continue
                
            row = date_df.iloc[0]
            threshold = row["threshold"]
            slope = row["slope"]
            
            # Create subplot
            plt.subplot(n_rows, n_cols, i)
            
            # Generate x values based on threshold
            x = np.linspace(threshold - 5, threshold + 5, 1000)
            y = logistic_function(x, threshold, slope)
            
            # Plot logistic curve
            plt.plot(x, y)
            plt.axvline(threshold, color='r', linestyle='--', alpha=0.7, label=f"Threshold: {threshold:.2f}")
            plt.axhline(0.5, color='g', linestyle=':', alpha=0.7)
            plt.annotate(f"Slope: {slope:.2f}", xy=(0.05, 0.05), xycoords='axes fraction')
            
            plt.title(f"{date.strftime('%Y-%m-%d')}")
            plt.grid(True, alpha=0.3)
            plt.xlabel("Task Difficulty")
            plt.ylabel("Success Probability")
            plt.ylim(0, 1)
        
        # Add overall title
        plt.suptitle(f"Logistic Curves for {scenario}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        output_path = os.path.join(output_dir, f"logistic_curves_{scenario}.{fmt}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _style_plots()  # Apply font and title settings
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved logistic curves for {scenario} to {output_path}")

def plot_logistic_curves_overlay(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot sampled logistic curves for all scenarios on a single plot.
    """
    plt.figure(figsize=get_figure_size(12, 8))
    cmap = plt.cm.get_cmap("tab20")
    idx = 0
    for scenario, scenario_df in df.groupby("scenario"):
        dates = sorted(scenario_df["date"].unique())
        n = min(6, len(dates))
        if n < 1:
            continue
        step = max(1, len(dates) // n) if n > 1 else 1
        sample = dates if n == 1 else dates[::step][:n]
        for dt in sample:
            row = scenario_df[scenario_df["date"] == dt].iloc[0]
            thr, sl = row["threshold"], row["slope"]
            x = np.linspace(thr - 5, thr + 5, 1000)
            y = logistic_function(x, thr, sl)
            plt.plot(x, y, color=cmap(idx), alpha=0.7,
                     label=f"{dt.strftime('%Y-%m-%d')}")
            idx += 1

    plt.xlabel("Task Difficulty")
    plt.ylabel("Success Probability")
    plt.title("Overlayed Logistic Curves for All Scenarios/Dates")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"logistic_curves_overlay.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved overlay logistic curves to {output_path}")

def plot_model_parameters(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot threshold vs slope parameters for each model in the dataset.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save plots
        fmt: File format for output
    """
    plt.figure(figsize=get_figure_size(10, 8))
    
    # Group by scenario for different colors
    for scenario, group in df.groupby("scenario"):
        plt.scatter(group["threshold"], group["slope"], 
                   alpha=0.7, label=scenario)
    
    plt.xlabel("Threshold Parameter")
    plt.ylabel("Slope Parameter")
    plt.title("Distribution of Model Parameters")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, f"model_parameters.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved model parameters plot to {output_path}")

def create_summary_stats(df: pd.DataFrame, output_dir: str):
    """
    Create summary statistics for the ability forecasts.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save summary
    """
    # Group by scenario and generate summary stats
    summary = df.groupby("scenario").agg({
        "threshold": ["mean", "std", "min", "max"],
        "slope": ["mean", "std", "min", "max"],
        "date": ["min", "max", "count"]
    })
    
    # Save summary to CSV
    output_path = os.path.join(output_dir, "ability_forecast_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path)
    
    print(f"Saved summary statistics to {output_path}")
    print("\nSummary of ability forecasts by scenario:")
    print(summary)

def validate_slope_trends(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Create validation plots to identify issues with slope forecasting.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save plot
        fmt: File format for output
    """
    plt.figure(figsize=get_figure_size(12, 8))
    
    # Left subplot: Distribution of slope values
    plt.subplot(2, 2, 1)
    plt.hist(df["slope"], bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel("Slope Value")
    plt.ylabel("Count")
    plt.title("Distribution of Slope Values")
    plt.grid(True, alpha=0.3)
    
    # Right subplot: Slope vs. threshold scatter
    plt.subplot(2, 2, 2)
    for scenario, group in df.groupby("scenario"):
        plt.scatter(group["threshold"], group["slope"], 
                  alpha=0.7, label=scenario)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Threshold")
    plt.ylabel("Slope")
    plt.title("Slope vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom left: Slope histogram by scenario
    plt.subplot(2, 2, 3)
    scenarios = df["scenario"].unique()
    for i, scenario in enumerate(scenarios):
        scenario_slopes = df[df["scenario"] == scenario]["slope"]
        plt.hist(scenario_slopes, bins=15, alpha=0.5, label=scenario)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel("Slope Value")
    plt.ylabel("Count")
    plt.title("Slope Distribution by Scenario")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom right: Example logistic curves with problematic slopes
    plt.subplot(2, 2, 4)
    x = np.linspace(-5, 20, 1000)
    
    # Example curves with different slope signs
    plt.plot(x, logistic_function(x, 5, -1.0), 'g-', label="Negative slope (correct)")
    plt.plot(x, logistic_function(x, 5, 1.0), 'r-', label="Positive slope (incorrect)")
    plt.axvline(x=5, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel("Task Difficulty")
    plt.ylabel("Success Probability")
    plt.title("Effect of Slope Sign")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"slope_validation.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved slope validation plot to {output_path}")

def generate_frequency_comparison(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Generate comparison plots of forecasts across different frequencies.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Extract trend types and frequencies in a more robust way
    df['trend_type'] = df['scenario'].str.extract(r'([^_]+)_')
    df['frequency'] = df['scenario'].str.extract(r'_([^_]+)$')
    
    # Get unique trend types and frequencies
    trend_types = df['trend_type'].unique()
    frequencies = df['frequency'].unique()
    
    # Compare forecasts for each trend type across frequencies
    for trend in trend_types:
        trend_df = df[df['trend_type'] == trend]
        
        # Skip if no data for this trend
        if len(trend_df) == 0:
            print(f"No data for trend type {trend}, skipping comparison plot")
            continue
        
        plt.figure(figsize=get_figure_size(15, 10))
        
        # Create two subplots for threshold and slope comparisons
        plt.subplot(2, 1, 1)
        for freq in frequencies:
            scenario_df = trend_df[trend_df['frequency'] == freq]
            
            if len(scenario_df) == 0:
                continue
                
            scenario_df = scenario_df.sort_values("date")
            plt.plot(scenario_df["date"], scenario_df["threshold"], 
                     marker='o', label=f"{freq}")
        
        plt.ylabel("Threshold")
        plt.title(f"Threshold Forecasts for {trend}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for freq in frequencies:
            scenario_df = trend_df[trend_df['frequency'] == freq]
            
            if len(scenario_df) == 0:
                continue
                
            scenario_df = scenario_df.sort_values("date")
            plt.plot(scenario_df["date"], scenario_df["slope"], 
                     marker='s', label=f"{freq}")
        
        plt.ylabel("Slope")
        plt.title(f"Slope Forecasts for {trend}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"frequency_comparison_{trend}.{fmt}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _style_plots()  # Apply font and title settings
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved frequency comparison for {trend} to {output_path}")
    
    # Create comparison across trend types (one plot per frequency)
    for freq in frequencies:
        plt.figure(figsize=get_figure_size(15, 10))
        
        # Create two subplots for threshold and slope comparisons
        plt.subplot(2, 1, 1)
        for trend in trend_types:
            scenario = f"{trend}_{freq}"
            scenario_df = df[df["scenario"] == scenario]
            
            if len(scenario_df) == 0:
                continue
                
            scenario_df = scenario_df.sort_values("date")
            plt.plot(scenario_df["date"], scenario_df["threshold"], 
                     marker='o', label=f"{trend}")
        
        plt.ylabel("Threshold")
        plt.title(f"Threshold Forecasts for {freq} Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for trend in trend_types:
            scenario = f"{trend}_{freq}"
            scenario_df = df[df["scenario"] == scenario]
            
            if len(scenario_df) == 0:
                continue
                
            scenario_df = scenario_df.sort_values("date")
            plt.plot(scenario_df["date"], scenario_df["slope"], 
                     marker='s', label=f"{trend}")
        
        plt.ylabel("Slope")
        plt.title(f"Slope Forecasts for {freq} Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"trend_comparison_{freq}.{fmt}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _style_plots()  # Apply font and title settings
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved trend comparison for {freq} frequency to {output_path}")

def plot_confidence_intervals(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot confidence intervals for threshold and slope parameters.
    
    Args:
        df: DataFrame with ability forecasts
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Check if confidence intervals exist in the data
    has_threshold_ci = "threshold_ci_lower" in df.columns and "threshold_ci_upper" in df.columns
    has_slope_ci = "slope_ci_lower" in df.columns and "slope_ci_upper" in df.columns
    
    if not (has_threshold_ci or has_slope_ci):
        print("No confidence interval data found, skipping confidence interval plots")
        return
    
    # Plot confidence intervals by scenario type
    scenario_types = df["scenario"].str.split("_", expand=True)[0].unique()
    
    for scenario_type in scenario_types:
        scenario_df = df[df["scenario"].str.startswith(f"{scenario_type}_")]
        
        if len(scenario_df) == 0:
            continue
        
        plt.figure(figsize=get_figure_size(12, 10))
        
        # Plot threshold confidence intervals
        if has_threshold_ci:
            plt.subplot(2, 1, 1)
            
            for freq, group in scenario_df.groupby(scenario_df["scenario"].str.split("_", expand=True)[1]):
                group = group.sort_values("date")
                valid_ci = group["threshold_ci_lower"].notna() & group["threshold_ci_upper"].notna()
                
                if valid_ci.any():
                    ci_group = group[valid_ci]
                    plt.plot(ci_group["date"], ci_group["threshold"], 
                            marker="o", label=f"{freq}", alpha=0.7)
                    plt.fill_between(ci_group["date"], 
                                    ci_group["threshold_ci_lower"], 
                                    ci_group["threshold_ci_upper"],
                                    alpha=0.2)
            
            plt.xlabel("Date")
            plt.ylabel("Threshold")
            plt.title(f"Threshold Forecast with Confidence Intervals - {scenario_type}")
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot slope confidence intervals
        if has_slope_ci:
            plt.subplot(2, 1, 2)
            
            for freq, group in scenario_df.groupby(scenario_df["scenario"].str.split("_", expand=True)[1]):
                group = group.sort_values("date")
                valid_ci = group["slope_ci_lower"].notna() & group["slope_ci_upper"].notna()
                
                if valid_ci.any():
                    ci_group = group[valid_ci]
                    plt.plot(ci_group["date"], ci_group["slope"], 
                            marker="o", label=f"{freq}", alpha=0.7)
                    plt.fill_between(ci_group["date"], 
                                    ci_group["slope_ci_lower"], 
                                    ci_group["slope_ci_upper"],
                                    alpha=0.2)
            
            plt.xlabel("Date")
            plt.ylabel("Slope")
            plt.title(f"Slope Forecast with Confidence Intervals - {scenario_type}")
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"confidence_intervals_{scenario_type}.{fmt}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _style_plots()  # Apply font and title settings
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved confidence interval plot for {scenario_type} to {output_path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set global plotting parameters
    global FONT_SIZE, TITLE_ENABLED
    FONT_SIZE = args.font_size
    TITLE_ENABLED = not args.disable_titles
    
    print(f"Loading ability forecasts from {args.input}")
    df = pd.read_csv(args.input)
    
    # Convert date column to datetime if needed
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    print(f"Loaded {len(df)} ability forecasts")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    plot_threshold_timeline(df, args.output, args.format)
    plot_slope_timeline(df, args.output, args.format)
    plot_logistic_curves_grid(df, args.output, args.format)
    plot_logistic_curves_overlay(df, args.output, args.format)
    plot_model_parameters(df, args.output, args.format)
    create_summary_stats(df, args.output)
    validate_slope_trends(df, args.output, args.format)
    generate_frequency_comparison(df, args.output, args.format)
    plot_confidence_intervals(df, args.output, args.format)  # Add new confidence interval plots
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
