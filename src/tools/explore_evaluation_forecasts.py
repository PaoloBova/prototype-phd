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
    parser.add_argument("--filter-model", help="Filter by specific ability model")
    parser.add_argument("--filter-scenario", help="Filter by specific scenario")
    parser.add_argument("--filter-cost", help="Filter by specific cost model")
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
    cost_models = get_unique_values(df, "cost_model")
    
    # Create visualizations for each combination
    for scenario in ability_scenarios:
        for cost_model in cost_models[:3]:  # Limit to first 3 cost models to avoid too many plots
            for sampler in sampler_types:
                for adj_method in adjustment_methods:
                    filtered_df = df[(df["ability_scenario"] == scenario) & 
                                    (df["sampler_type"] == sampler) &
                                    (df["adjustment_method"] == adj_method) &
                                    (df["cost_model"] == cost_model)]
                    
                    if len(filtered_df) == 0:
                        continue
                    
                    plt.figure(figsize=(12, 8))
                    
                    # Group by budget scenario and sort by budget fraction
                    filtered_df = filtered_df.sort_values("budget_fraction")
                    
                    # Plot the original and adjusted evaluation windows for each budget scenario
                    for i, (_, row) in enumerate(filtered_df.iterrows()):
                        budget_fraction = row["budget_fraction"]
                        
                        # Original window (semi-transparent)
                        plt.plot([row["original_window_lower"], row["original_window_upper"]], 
                                [budget_fraction, budget_fraction], 
                                linewidth=2, alpha=0.3, color='blue',
                                label="Original Window" if i == 0 else "")
                        
                        # Adjusted window (solid)
                        plt.plot([row["window_lower"], row["window_upper"]], 
                                [budget_fraction, budget_fraction], 
                                linewidth=2, marker='|', color='red',
                                label=f"{budget_fraction*100:.0f}% Budget")
                        
                    # Mark the threshold with a vertical line
                    threshold = filtered_df.iloc[0]["ability_threshold"]
                    plt.axvline(threshold, color='green', linestyle='--', label="Ability Threshold")
                
                    plt.xlabel("Task Difficulty")
                    plt.ylabel("Budget Fraction")
                    plt.title(f"Evaluation Windows by Budget\n"
                             f"Scenario: {scenario}, Cost: {cost_model}\n"
                             f"Sampler: {sampler}, Method: {adj_method}")
                    plt.grid(True, axis='x', alpha=0.3)
                    
                    # Customize legend to avoid duplicates
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
                    
                    # plt.tight_layout()
                    
                    # Save figure
                    safe_filename = f"windows_{scenario}_{cost_model}_{sampler}_{adj_method}.{fmt}".replace(" ", "_")
                    output_path = os.path.join(output_dir, "windows", safe_filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    plt.savefig(output_path, dpi=150)
                    plt.close()
                    
                    print(f"Saved evaluation windows plot to {output_path}")

def plot_window_adjustments(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot how windows are adjusted for different budget fractions.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Get key combinations to plot
    key_combinations = df.groupby(
        ["ability_scenario", "ability_model", "cost_model", "adjustment_method"]
    ).size().reset_index()[["ability_scenario", "ability_model", "cost_model", "adjustment_method"]]
    
    # Limit to a reasonable number of plots
    if len(key_combinations) > 20:
        key_combinations = key_combinations.iloc[:20]
    
    for _, row in key_combinations.iterrows():
        # Filter the data for this combination
        filtered_df = df[
            (df["ability_scenario"] == row["ability_scenario"]) & 
            (df["ability_model"] == row["ability_model"]) & 
            (df["cost_model"] == row["cost_model"]) & 
            (df["adjustment_method"] == row["adjustment_method"])
        ]
        
        # Sort by budget fraction
        filtered_df = filtered_df.sort_values("budget_fraction")
        
        plt.figure(figsize=(10, 6))
        
        # Plot relationship between budget fraction and window width
        plt.scatter(filtered_df["budget_fraction"], filtered_df["window_width"], 
                   label="Window Width", s=50)
        
        # Plot relationship between budget fraction and width ratio
        plt.scatter(filtered_df["budget_fraction"], filtered_df["width_ratio"], 
                   label="Width Ratio", s=50, marker='x')
        
        # Add trend lines
        sns.regplot(x="budget_fraction", y="window_width", data=filtered_df, 
                   scatter=False, label="Width Trend")
        sns.regplot(x="budget_fraction", y="width_ratio", data=filtered_df, 
                   scatter=False, label="Ratio Trend")
        
        plt.xlabel("Budget Fraction")
        plt.ylabel("Window Width / Ratio")
        plt.title(f"Window Adjustment vs Budget\n"
                 f"Scenario: {row['ability_scenario']}, "
                 f"Model: {row['ability_model']}, Cost: {row['cost_model']}\n"
                 f"Method: {row['adjustment_method']}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # plt.tight_layout()
        
        # Save figure
        safe_filename = (f"window_adjustment_{row['ability_scenario']}_{row['ability_model']}_"
                        f"{row['cost_model']}_{row['adjustment_method']}.{fmt}").replace(" ", "_")
        output_path = os.path.join(output_dir, "window_adjustments", safe_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved window adjustment plot to {output_path}")

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
    cost_models = get_unique_values(df, "cost_model")
    
    # Create visualizations for each combination (limit to keep it manageable)
    for scenario in ability_scenarios[:2]:  # Limit to first 2 scenarios
        for cost_model in cost_models[:2]:  # Limit to first 2 cost models
            for sampler in sampler_types:
                for adj_method in adjustment_methods:
                    filtered_df = df[(df["ability_scenario"] == scenario) & 
                                    (df["sampler_type"] == sampler) &
                                    (df["adjustment_method"] == adj_method) &
                                    (df["cost_model"] == cost_model)]
                    
                    if len(filtered_df) == 0:
                        continue
                    
                    plt.figure(figsize=(12, 6))
                    
                    # Group by budget scenario and sort by budget fraction
                    filtered_df = filtered_df.sort_values("budget_fraction", ascending=False)
                    
                    # Get overall min/max for better plotting
                    max_difficulty = max(filtered_df["window_upper"].max(), 
                                         filtered_df["original_window_upper"].max()) + 1
                    min_difficulty = min(filtered_df["window_lower"].min(), 
                                         filtered_df["original_window_lower"].min()) - 1
                    x = np.linspace(min_difficulty, max_difficulty, 1000)
                    
                    # Plot the task density for each budget scenario
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
                    plt.title(f"Task Density by Budget\n"
                             f"Scenario: {scenario}, Cost: {cost_model}\n"
                             f"Sampler: {sampler}, Method: {adj_method}")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # plt.tight_layout()
                    
                    # Save figure
                    safe_filename = f"density_{scenario}_{cost_model}_{sampler}_{adj_method}.{fmt}".replace(" ", "_")
                    output_path = os.path.join(output_dir, "density", safe_filename)
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
    cost_models = get_unique_values(df, "cost_model")
    
    # Group by ability scenario and adjustment method
    for scenario in ability_scenarios[:3]:  # Limit to first 3 scenarios
        for cost_model in cost_models[:3]:  # Limit to first 3 cost models
            scenario_df = df[(df["ability_scenario"] == scenario) & 
                            (df["cost_model"] == cost_model)]
            
            if len(scenario_df) == 0:
                continue
            
            plt.figure(figsize=(12, 6))
            
            # Set up bar plot with adjustment method and sampler type
            scenario_df['method_sampler'] = scenario_df['adjustment_method'] + ' - ' + scenario_df['sampler_type']
            
            ax = sns.barplot(data=scenario_df, x="budget_fraction", y="total_samples", 
                            hue="method_sampler", alpha=0.7)
            
            # Customize plot
            plt.xlabel("Budget Fraction")
            plt.ylabel("Total Sample Count")
            plt.title(f"Sample Counts by Budget and Method\n"
                     f"Scenario: {scenario}, Cost: {cost_model}")
            plt.grid(True, axis='y', alpha=0.3)
            
            # Format x tick labels as percentages
            plt.xticks(ticks=plt.xticks()[0], 
                      labels=[f"{x*100:.0f}%" for x in sorted(scenario_df["budget_fraction"].unique())])
            
            # plt.tight_layout()
            
            # Save figure
            safe_filename = f"sample_counts_{scenario}_{cost_model}.{fmt}".replace(" ", "_")
            output_path = os.path.join(output_dir, "samples", safe_filename)
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
    cost_models = get_unique_values(df, "cost_model")
    
    for scenario in ability_scenarios[:3]:  # Limit to avoid too many plots
        for cost_model in cost_models[:3]:  # Limit to avoid too many plots
            scenario_df = df[(df["ability_scenario"] == scenario) & 
                            (df["cost_model"] == cost_model)]
            
            if len(scenario_df) == 0:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Set up bar plot
            scenario_df['method_sampler'] = scenario_df['adjustment_method'] + ' - ' + scenario_df['sampler_type']
            ax = sns.barplot(data=scenario_df, x="budget_fraction", y="available_budget", 
                            hue="method_sampler", alpha=0.7)
            
            # Customize plot
            plt.xlabel("Budget Fraction")
            plt.ylabel("Available Budget (cost units)")
            plt.title(f"Cost Allocation by Budget Fraction\n"
                     f"Scenario: {scenario}, Cost Model: {cost_model}")
            plt.grid(True, axis='y', alpha=0.3)
            
            # Format x tick labels as percentages
            plt.xticks(ticks=plt.xticks()[0], 
                      labels=[f"{x*100:.0f}%" for x in sorted(scenario_df["budget_fraction"].unique())])
            
            # plt.tight_layout()
            
            # Save figure
            safe_filename = f"cost_allocation_{scenario}_{cost_model}.{fmt}".replace(" ", "_")
            output_path = os.path.join(output_dir, "costs", safe_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"Saved cost allocation plot to {output_path}")

def plot_scenario_comparison(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot comparison between base, lower, and upper CI scenarios.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Make a copy to avoid modifying the original DataFrame
    plot_df = df.copy()
    
    # More flexible pattern matching for scenario variants
    plot_df['scenario_variant'] = 'unknown'
    plot_df.loc[plot_df['budget_scenario'].str.contains('_base'), 'scenario_variant'] = 'base'
    plot_df.loc[plot_df['budget_scenario'].str.contains('_lower'), 'scenario_variant'] = 'lower'
    plot_df.loc[plot_df['budget_scenario'].str.contains('_upper'), 'scenario_variant'] = 'upper'
    
    # Extract the base part of the scenario (remove _base, _lower, _upper suffixes)
    plot_df['base_scenario'] = plot_df['budget_scenario'].str.replace('_lower$|_upper$|_base$', '', regex=True)
    
    # Group by base scenario to identify scenario families
    scenario_counts = plot_df.groupby('base_scenario')['scenario_variant'].nunique()
    complete_scenarios = scenario_counts[scenario_counts >= 3].index.tolist()
    
    print(f"Found {len(complete_scenarios)} complete scenario families with base/lower/upper variants")
    
    if len(complete_scenarios) == 0:
        # Try a more aggressive approach to find any scenario families
        print("Looking for partial scenario families...")
        
        # Extract ability and cost model information from scenario name
        # Format often looks like: model_scenario_costmodel_budgetfraction
        parts = plot_df['budget_scenario'].str.split('_', expand=True)
        if parts.shape[1] >= 3:  # Ensure we have enough parts
            plot_df['scenario_family'] = parts[0] + '_' + parts[1]
            
            # Count variants in each family
            family_counts = plot_df.groupby('scenario_family')['scenario_variant'].nunique()
            candidate_families = family_counts[family_counts >= 2].index.tolist()
            
            print(f"Found {len(candidate_families)} partial scenario families with at least 2 variants")
            
            if len(candidate_families) > 0:
                # Visualize these partial families instead
                complete_scenarios = candidate_families
                # Update the base_scenario field
                for family in candidate_families:
                    related_scenarios = plot_df[plot_df['budget_scenario'].str.contains(family)]
                    unique_variants = related_scenarios['scenario_variant'].unique()
                    print(f"Family {family} has variants: {unique_variants}")
                    plot_df.loc[plot_df['budget_scenario'].str.contains(family), 'base_scenario'] = family
                    
    # Limit to first 10 but ensure we get some if available
    plot_scenarios = complete_scenarios[:10]
    
    # Ensure we have at least one plot if there are any complete scenarios
    if len(plot_scenarios) == 0 and len(complete_scenarios) > 0:
        plot_scenarios = [complete_scenarios[0]]
    
    for base_scenario in plot_scenarios:
        # Get all related scenarios
        scenario_family = plot_df[plot_df['base_scenario'] == base_scenario].copy()  # Create an explicit copy
        
        if len(scenario_family) == 0:
            print(f"No matching scenarios found for base: {base_scenario}")
            continue
            
        print(f"Processing scenario family: {base_scenario} with {len(scenario_family)} variants")
        
        # Group by ability model, cost model and filter to get unique combinations
        key_columns = ['ability_model', 'cost_model', 'adjustment_method', 'sampler_type']
        combinations = scenario_family.groupby(key_columns).size().reset_index()[key_columns]
        
        for _, combo in combinations.iterrows():
            # Use .copy() to create a new DataFrame instead of a view
            filtered_df = scenario_family[
                (scenario_family['ability_model'] == combo['ability_model']) &
                (scenario_family['cost_model'] == combo['cost_model']) &
                (scenario_family['adjustment_method'] == combo['adjustment_method']) &
                (scenario_family['sampler_type'] == combo['sampler_type'])
            ].copy()  # Create an explicit copy to avoid SettingWithCopyWarning
            
            # Sort variants for consistent ordering
            filtered_df['plot_order'] = 0
            filtered_df.loc[filtered_df['scenario_variant'] == 'base', 'plot_order'] = 0
            filtered_df.loc[filtered_df['scenario_variant'] == 'lower', 'plot_order'] = 1
            filtered_df.loc[filtered_df['scenario_variant'] == 'upper', 'plot_order'] = 2
            filtered_df = filtered_df.sort_values('plot_order')
            
            # Get the variants present in this filtered dataset
            variants_present = filtered_df['scenario_variant'].unique()
            if len(variants_present) < 2:
                print(f"Skipping plot for {combo['ability_model']}/{combo['cost_model']} - not enough variants")
                continue
                
            # Plot window comparison
            plt.figure(figsize=(12, 8))
            
            # Calculate number of subplots needed
            n_subplots = len(variants_present)
            
            # Plot each variant
            for i, variant in enumerate(variants_present):
                variant_df = filtered_df[filtered_df['scenario_variant'] == variant].copy()  # Another explicit copy
                variant_df = variant_df.sort_values('budget_fraction')
                
                plt.subplot(n_subplots, 1, i+1)
                
                for _, row in variant_df.iterrows():
                    budget_fraction = row["budget_fraction"]
                    plt.plot([row["window_lower"], row["window_upper"]], 
                            [budget_fraction, budget_fraction], 
                            linewidth=2, marker='|',
                            label=f"{budget_fraction*100:.0f}% Budget")
                
                # Mark threshold
                threshold = variant_df.iloc[0]["ability_threshold"] if len(variant_df) > 0 else 0
                plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
                
                # Set title and labels
                variant_title = "Base Scenario" if variant == "base" else f"{variant.title()} CI Scenario"
                plt.title(variant_title)
                plt.ylabel("Budget Fraction")
                
                # Only show x label on bottom plot
                if i == n_subplots - 1:
                    plt.xlabel("Task Difficulty")
                
                plt.grid(True, alpha=0.3)
                
                # Add legend only to first plot
                if i == 0:
                    plt.legend(loc='upper right')
            
            plt.suptitle(f"Window Comparison Across Scenarios\n"
                        f"Model: {combo['ability_model']}, Cost: {combo['cost_model']}\n"
                        f"Method: {combo['adjustment_method']}, Sampler: {combo['sampler_type']}")
            # plt.tight_layout()
            
            # Save figure
            safe_filename = (f"scenario_comparison_{base_scenario}_{combo['ability_model']}_{combo['cost_model']}_"
                            f"{combo['adjustment_method']}_{combo['sampler_type']}.{fmt}").replace(" ", "_")
            output_path = os.path.join(output_dir, "scenario_comparison", safe_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"Saved scenario comparison plot to {output_path}")

def create_summary_stats(df: pd.DataFrame, output_dir: str):
    """
    Create summary statistics for the evaluation forecast data.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save summary
    """
    # Group by key attributes and generate summary stats
    summary = df.groupby(["ability_scenario", "cost_model", "budget_scenario", "adjustment_method"]).agg({
        "window_lower": ["mean", "min", "max"],
        "window_upper": ["mean", "min", "max"],
        "window_width": ["mean", "min", "max"],
        "original_window_lower": ["mean", "min", "max"],
        "original_window_upper": ["mean", "min", "max"],
        "original_window_width": ["mean", "min", "max"],
        "width_ratio": ["mean", "min", "max"],
        "total_samples": ["mean", "min", "max", "sum"],
        "available_budget": ["mean", "min", "max", "sum"]
    })
    
    # Save summary to CSV
    output_path = os.path.join(output_dir, "evaluation_forecast_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path)
    
    print(f"Saved summary statistics to {output_path}")
    
    # Create a simpler summary for quick reference
    simple_summary = df.groupby(["ability_model", "ability_scenario"]).agg({
        "budget_fraction": "nunique",
        "cost_model": "nunique",
        "window_width": ["mean", "median"],
        "total_samples": ["mean", "sum"],
        "available_budget": ["mean", "sum"]
    }).reset_index()
    
    simple_path = os.path.join(output_dir, "simple_summary.csv")
    simple_summary.to_csv(simple_path, index=False)
    
    print(f"Saved simple summary statistics to {simple_path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"Loading evaluation forecasts from {args.input}")
    df = pd.read_csv(args.input)
    
    # Convert date column to datetime if needed
    if "ability_date" in df.columns:
        df["ability_date"] = pd.to_datetime(df["ability_date"])
    
    print(f"Loaded {len(df)} evaluation forecasts")
    
    # Apply filters if specified
    if args.filter_model:
        df = df[df["ability_model"] == args.filter_model].copy()  # Create copy to avoid warnings
        print(f"Filtered to {len(df)} records for model: {args.filter_model}")
    
    if args.filter_scenario:
        df = df[df["ability_scenario"] == args.filter_scenario].copy()  # Create copy 
        print(f"Filtered to {len(df)} records for scenario: {args.filter_scenario}")
        
    if args.filter_cost:
        df = df[df["cost_model"] == args.filter_cost].copy()  # Create copy
        print(f"Filtered to {len(df)} records for cost model: {args.filter_cost}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    plot_evaluation_windows(df, args.output, args.format)
    plot_window_adjustments(df, args.output, args.format)
    plot_task_density(df, args.output, args.format)
    plot_sample_counts(df, args.output, args.format)
    plot_cost_allocation(df, args.output, args.format)
    plot_scenario_comparison(df, args.output, args.format)
    create_summary_stats(df, args.output)
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
