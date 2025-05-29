"""
Utilities for exploring and visualizing evaluation forecast data.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Set

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore evaluation forecast data")
    parser.add_argument("--input", required=True, help="Path to evaluation forecasts CSV file")
    parser.add_argument("--output", default="reports/evaluation_forecast_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--format", default="png", help="Output format (png, pdf, svg)")
    return parser.parse_args()

def get_unique_values(df: pd.DataFrame, column: str) -> List:
    """Get unique values from a column, sorted if possible."""
    values = df[column].unique().tolist()
    try:
        return sorted(values)
    except:
        return values

def plot_evaluation_windows(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot evaluation windows for different budget scenarios.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Get unique values for filtering
    ability_scenarios = get_unique_values(df, "ability_scenario")
    sampler_types = get_unique_values(df, "sampler_type")
    adjustment_methods = get_unique_values(df, "adjustment_method")
    
    # Create visualizations for each combination of ability scenario and sampler type
    for scenario in ability_scenarios:
        for sampler in sampler_types:
            for adj_method in adjustment_methods:
                filtered_df = df[(df["ability_scenario"] == scenario) & 
                                (df["sampler_type"] == sampler) &
                                (df["adjustment_method"] == adj_method)]
                
                if len(filtered_df) == 0:
                    continue
                
                plt.figure(figsize=(12, 6))
                
                # Group by budget scenario and sort by budget fraction
                filtered_df = filtered_df.sort_values("budget_fraction")
                
                # Plot the evaluation window for each budget scenario
                for i, (_, row) in enumerate(filtered_df.iterrows()):
                    budget_fraction = row["budget_fraction"]
                    window_lower = row["window_lower"]
                    window_upper = row["window_upper"]
                    threshold = row["ability_threshold"]
                    
                    # Plot horizontal line representing the evaluation window
                    plt.plot([window_lower, window_upper], [budget_fraction, budget_fraction], 
                            linewidth=2, marker='|', label=f"{budget_fraction*100:.0f}% Budget")
                    
                    # Mark the threshold with a vertical line
                    if i == 0:  # Only add once to avoid cluttering the legend
                        plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
                
                plt.xlabel("Task Difficulty")
                plt.ylabel("Budget Fraction")
                plt.title(f"Evaluation Windows by Budget\nScenario: {scenario}, Sampler: {sampler}, Method: {adj_method}")
                plt.grid(True, axis='x', alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                
                # Save figure
                safe_filename = f"windows_{scenario}_{sampler}_{adj_method}.{fmt}".replace(" ", "_")
                output_path = os.path.join(output_dir, safe_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=150)
                plt.close()
                
                print(f"Saved evaluation windows plot to {output_path}")

def plot_task_density(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot task density across difficulty for different budget scenarios.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Get unique values for filtering
    ability_scenarios = get_unique_values(df, "ability_scenario")
    sampler_types = get_unique_values(df, "sampler_type")
    adjustment_methods = get_unique_values(df, "adjustment_method")
    
    # Create visualizations for each combination of ability scenario and sampler type
    for scenario in ability_scenarios:
        for sampler in sampler_types:
            for adj_method in adjustment_methods:
                filtered_df = df[(df["ability_scenario"] == scenario) & 
                                (df["sampler_type"] == sampler) &
                                (df["adjustment_method"] == adj_method)]
                
                if len(filtered_df) == 0:
                    continue
                
                plt.figure(figsize=(12, 6))
                
                # Group by budget scenario and sort by budget fraction
                filtered_df = filtered_df.sort_values("budget_fraction", ascending=False)
                
                # Plot the task density for each budget scenario
                max_difficulty = filtered_df["window_upper"].max() + 1
                min_difficulty = filtered_df["window_lower"].min() - 1
                x = np.linspace(min_difficulty, max_difficulty, 1000)
                
                for _, row in filtered_df.iterrows():
                    budget_fraction = row["budget_fraction"]
                    window_lower = row["window_lower"]
                    window_upper = row["window_upper"]
                    total_samples = row["total_samples"]
                    
                    # Calculate task density
                    window_width = window_upper - window_lower
                    if window_width <= 0 or total_samples <= 0:
                        continue
                        
                    density = total_samples / window_width
                    
                    # Create density function
                    y = np.zeros_like(x)
                    mask = (x >= window_lower) & (x <= window_upper)
                    
                    if sampler == "uniform":
                        # Uniform density
                        y[mask] = density
                    elif sampler == "normal":
                        # Normal density (approximate)
                        window_center = (window_lower + window_upper) / 2
                        std = window_width * 0.3  # From the code
                        y = density * np.exp(-0.5 * ((x - window_center) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                        y = np.clip(y, 0, None)  # Ensure non-negative
                    
                    plt.plot(x, y, label=f"{budget_fraction*100:.0f}% Budget")
                
                # Mark the threshold
                threshold = filtered_df.iloc[0]["ability_threshold"]
                plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
                
                plt.xlabel("Task Difficulty")
                plt.ylabel("Task Density (samples per difficulty unit)")
                plt.title(f"Task Density by Budget\nScenario: {scenario}, Sampler: {sampler}, Method: {adj_method}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                
                # Save figure
                safe_filename = f"density_{scenario}_{sampler}_{adj_method}.{fmt}".replace(" ", "_")
                output_path = os.path.join(output_dir, safe_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=150)
                plt.close()
                
                print(f"Saved task density plot to {output_path}")

def plot_sample_counts(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot total sample counts for different budget scenarios.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Get unique values for filtering
    ability_scenarios = get_unique_values(df, "ability_scenario")
    
    # Group by ability scenario and adjustment method
    for scenario in ability_scenarios:
        scenario_df = df[df["ability_scenario"] == scenario]
        
        plt.figure(figsize=(10, 6))
        
        # Set up bar plot
        ax = sns.barplot(data=scenario_df, x="budget_fraction", y="total_samples", 
                         hue="adjustment_method", alpha=0.7)
        
        # Customize plot
        plt.xlabel("Budget Fraction")
        plt.ylabel("Total Sample Count")
        plt.title(f"Sample Counts by Budget and Adjustment Method\nScenario: {scenario}")
        plt.grid(True, axis='y', alpha=0.3)
        
        # Format x tick labels as percentages
        plt.xticks(ticks=plt.xticks()[0], 
                   labels=[f"{x*100:.0f}%" for x in sorted(scenario_df["budget_fraction"].unique())])
        
        plt.tight_layout()
        
        # Save figure
        safe_filename = f"sample_counts_{scenario}.{fmt}".replace(" ", "_")
        output_path = os.path.join(output_dir, safe_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved sample counts plot to {output_path}")

def plot_cost_allocation(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot cost allocation for different budget scenarios.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Get unique values for filtering
    ability_scenarios = get_unique_values(df, "ability_scenario")
    
    for scenario in ability_scenarios:
        scenario_df = df[df["ability_scenario"] == scenario]
        
        plt.figure(figsize=(10, 6))
        
        # Set up bar plot
        ax = sns.barplot(data=scenario_df, x="budget_fraction", y="available_budget", 
                         hue="adjustment_method", alpha=0.7)
        
        # Customize plot
        plt.xlabel("Budget Fraction")
        plt.ylabel("Available Budget (cost units)")
        plt.title(f"Cost Allocation by Budget Fraction\nScenario: {scenario}")
        plt.grid(True, axis='y', alpha=0.3)
        
        # Format x tick labels as percentages
        plt.xticks(ticks=plt.xticks()[0], 
                   labels=[f"{x*100:.0f}%" for x in sorted(scenario_df["budget_fraction"].unique())])
        
        plt.tight_layout()
        
        # Save figure
        safe_filename = f"cost_allocation_{scenario}.{fmt}".replace(" ", "_")
        output_path = os.path.join(output_dir, safe_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved cost allocation plot to {output_path}")

def plot_efficiency_metrics(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot efficiency metrics (samples per cost unit).
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Prepare data: calculate efficiency
    plot_df = df.copy()
    
    # Avoid division by zero
    plot_df = plot_df[plot_df["available_budget"] > 0]
    
    # Calculate samples per cost unit
    plot_df["efficiency"] = plot_df["total_samples"] / plot_df["available_budget"]
    
    # Get unique values for filtering
    ability_scenarios = get_unique_values(plot_df, "ability_scenario")
    
    for scenario in ability_scenarios:
        scenario_df = plot_df[plot_df["ability_scenario"] == scenario]
        
        plt.figure(figsize=(10, 6))
        
        # Set up bar plot
        ax = sns.barplot(data=scenario_df, x="budget_fraction", y="efficiency", 
                         hue="adjustment_method", alpha=0.7)
        
        # Customize plot
        plt.xlabel("Budget Fraction")
        plt.ylabel("Efficiency (samples per cost unit)")
        plt.title(f"Evaluation Efficiency by Budget\nScenario: {scenario}")
        plt.grid(True, axis='y', alpha=0.3)
        
        # Format x tick labels as percentages
        plt.xticks(ticks=plt.xticks()[0], 
                   labels=[f"{x*100:.0f}%" for x in sorted(scenario_df["budget_fraction"].unique())])
        
        plt.tight_layout()
        
        # Save figure
        safe_filename = f"efficiency_{scenario}.{fmt}".replace(" ", "_")
        output_path = os.path.join(output_dir, safe_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved efficiency metrics plot to {output_path}")

def create_summary_stats(df: pd.DataFrame, output_dir: str):
    """
    Create summary statistics for the evaluation forecast data.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save summary
    """
    # Group by key attributes and generate summary stats
    summary = df.groupby(["ability_scenario", "budget_scenario", "adjustment_method"]).agg({
        "window_lower": ["mean", "min", "max"],
        "window_upper": ["mean", "min", "max"],
        "total_samples": ["mean", "min", "max", "sum"],
        "available_budget": ["mean", "min", "max", "sum"]
    })
    
    # Add derived metrics
    summary_df = summary.reset_index()
    summary_df["window_width"] = summary_df[("window_upper", "mean")] - summary_df[("window_lower", "mean")]
    
    # Save summary to CSV
    output_path = os.path.join(output_dir, "evaluation_forecast_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path)
    
    print(f"Saved summary statistics to {output_path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"Loading evaluation forecasts from {args.input}")
    df = pd.read_csv(args.input)
    
    # Convert date column to datetime if needed
    if "ability_date" in df.columns:
        df["ability_date"] = pd.to_datetime(df["ability_date"])
    
    print(f"Loaded {len(df)} evaluation forecasts")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    plot_evaluation_windows(df, args.output, args.format)
    plot_task_density(df, args.output, args.format)
    plot_sample_counts(df, args.output, args.format)
    plot_cost_allocation(df, args.output, args.format)
    plot_efficiency_metrics(df, args.output, args.format)
    create_summary_stats(df, args.output)
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
