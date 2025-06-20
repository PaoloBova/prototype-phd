"""
Utilities for exploring and visualizing evaluation forecast data.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Set

# Global plotting style parameters
FONT_SIZE = 12  # Default font size
TITLE_ENABLED = True  # Whether to show titles in plots

def _style_plots():
    """Apply font size settings and optionally remove titles from plots."""
    mpl.rcParams.update({
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore evaluation forecast data")
    parser.add_argument("--input", required=True, help="Path to evaluation forecasts CSV file")
    parser.add_argument("--output", default="reports/evaluation_forecast_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--format", default="png", help="Output format (png, pdf, svg)")
    parser.add_argument("--filter-ability", help="Filter by ability_id")
    parser.add_argument("--filter-cost", help="Filter by cost_id")
    parser.add_argument("--filter-constraint", help="Filter by constraint_id")
    parser.add_argument("--filter-design", help="Filter by design_id")
    parser.add_argument("--filter-model", help="Filter by specific ability model (legacy)")
    parser.add_argument("--filter-scenario", help="Filter by specific scenario (legacy)")
    parser.add_argument("--font-size", type=int, default=12,
                        help="Base font size for all plot text")
    parser.add_argument("--disable-titles", action="store_true",
                        help="Strip all titles from plots for publication style")
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
    # Get unique values for ID-based filtering
    ability_ids = get_unique_values(df, "ability_id")
    cost_ids = get_unique_values(df, "cost_id")
    design_ids = get_unique_values(df, "design_id")
    
    # Limit the number of plots to avoid generating too many
    for ability_id in ability_ids[:3]:  # Limit to first 3 ability IDs
        ability_info = df[df["ability_id"] == ability_id].iloc[0]
        
        for cost_id in cost_ids[:2]:  # Limit to first 2 cost IDs
            cost_info = df[df["cost_id"] == cost_id].iloc[0]
            
            for design_id in design_ids[:2]:  # Limit to first 2 design IDs
                design_info = df[df["design_id"] == design_id].iloc[0]
                
                # Filter data for this combination
                filtered_df = df[
                    (df["ability_id"] == ability_id) & 
                    (df["cost_id"] == cost_id) & 
                    (df["design_id"] == design_id)
                ]
                
                if len(filtered_df) == 0:
                    continue
                
                plt.figure(figsize=(12, 8))
                
                # Group by budget and sort by budget fraction
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
                threshold = ability_info["ability_threshold"]
                plt.axvline(threshold, color='green', linestyle='--', label="Ability Threshold")
            
                plt.xlabel("Task Difficulty")
                plt.ylabel("Budget Fraction")
                plt.title(f"Evaluation Windows by Budget\n"
                         f"Model: {ability_info['ability_model']}, Scenario: {ability_info['ability_scenario']}\n"
                         f"Cost: {cost_info['cost_model']}, Method: {design_info['adjustment_method']}")
                plt.grid(True, axis='x', alpha=0.3)
                
                # Customize legend to avoid duplicates
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                
                plt.tight_layout()
                
                # Save figure with ID-based filename
                safe_filename = f"windows_ability_{ability_id}_cost_{cost_id}_design_{design_id}.{fmt}".replace(" ", "_")
                output_path = os.path.join(output_dir, "windows", safe_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                _style_plots()  # Apply font and title settings
                plt.savefig(output_path, dpi=300)
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
    # Get key combinations to plot using the ID fields
    key_combinations = df.groupby(
        ["ability_id", "cost_id", "design_id"]
    ).size().reset_index()[["ability_id", "cost_id", "design_id"]]
    
    # Limit to a reasonable number of plots
    if len(key_combinations) > 12:
        key_combinations = key_combinations.iloc[:12]
    
    for _, row in key_combinations.iterrows():
        # Get descriptive information for this combination
        combo_info = df[
            (df["ability_id"] == row["ability_id"]) & 
            (df["cost_id"] == row["cost_id"]) & 
            (df["design_id"] == row["design_id"])
        ].iloc[0]
        
        # Filter the data for this combination
        filtered_df = df[
            (df["ability_id"] == row["ability_id"]) & 
            (df["cost_id"] == row["cost_id"]) & 
            (df["design_id"] == row["design_id"])
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
        
        plt.xlabel("Budget Fraction")
        plt.ylabel("Window Width / Ratio")
        plt.title(f"Window Adjustment vs Budget\n"
                 f"Model: {combo_info['ability_model']}, Scenario: {combo_info['ability_scenario']}\n"
                 f"Cost: {combo_info['cost_model']}, Method: {combo_info['adjustment_method']}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure with ID-based filename
        safe_filename = f"window_adjustment_ability_{row['ability_id']}_cost_{row['cost_id']}_design_{row['design_id']}.{fmt}"
        safe_filename = safe_filename.replace(" ", "_")
        output_path = os.path.join(output_dir, "window_adjustments", safe_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
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
    # Get unique values for ID-based filtering
    ability_ids = get_unique_values(df, "ability_id")
    cost_ids = get_unique_values(df, "cost_id")
    design_ids = get_unique_values(df, "design_id")
    
    # Limit the number of plots
    for ability_id in ability_ids[:2]:  # Limit to first 2 ability IDs
        ability_info = df[df["ability_id"] == ability_id].iloc[0]
        
        for cost_id in cost_ids[:1]:  # Limit to first cost ID
            cost_info = df[df["cost_id"] == cost_id].iloc[0]
            
            for design_id in design_ids[:1]:  # Limit to first design ID
                design_info = df[df["design_id"] == design_id].iloc[0]
                
                # Filter data for this combination
                filtered_df = df[
                    (df["ability_id"] == ability_id) & 
                    (df["cost_id"] == cost_id) & 
                    (df["design_id"] == design_id)
                ]
                
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
                    
                    sampler_type = row["sampler_type"]
                    if sampler_type == "uniform":
                        # Uniform density
                        y[mask] = density
                    elif sampler_type == "normal":
                        # Normal density (approximate)
                        window_center = (window_lower + window_upper) / 2
                        std = window_width * 0.3  # From the code
                        y = density * np.exp(-0.5 * ((x - window_center) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                        y = np.clip(y, 0, None)  # Ensure non-negative
                    
                    plt.plot(x, y, label=f"{budget_fraction*100:.0f}% Budget")
                
                # Mark the threshold
                threshold = ability_info["ability_threshold"]
                plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
                
                plt.xlabel("Task Difficulty")
                plt.ylabel("Task Density (samples per difficulty unit)")
                plt.title(f"Task Density by Budget\n"
                         f"Model: {ability_info['ability_model']}, Scenario: {ability_info['ability_scenario']}\n"
                         f"Cost: {cost_info['cost_model']}, Method: {design_info['adjustment_method']}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                
                # Save figure with ID-based filename
                safe_filename = f"density_ability_{ability_id}_cost_{cost_id}_design_{design_id}.{fmt}"
                safe_filename = safe_filename.replace(" ", "_")
                output_path = os.path.join(output_dir, "density", safe_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                print(f"Saved task density plot to {output_path}")

def plot_sample_counts(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot total sample counts for different budget scenarios, grouped by design.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Get unique values for ID-based filtering
    ability_ids = get_unique_values(df, "ability_id")
    cost_ids = get_unique_values(df, "cost_id")
    
    # Limit plots
    for ability_id in ability_ids[:3]:  # Limit to first 3 ability IDs
        ability_info = df[df["ability_id"] == ability_id].iloc[0]
        
        for cost_id in cost_ids[:2]:  # Limit to first 2 cost IDs
            cost_info = df[df["cost_id"] == cost_id].iloc[0]
            
            # Filter data
            scenario_df = df[
                (df["ability_id"] == ability_id) & 
                (df["cost_id"] == cost_id)
            ]
            
            if len(scenario_df) == 0:
                continue
            
            plt.figure(figsize=(12, 6))
            
            # Create a design label combining adjustment method and sampler type
            scenario_df = scenario_df.copy()  # Create copy to avoid SettingWithCopyWarning
            scenario_df['design_label'] = scenario_df['design_id']
            
            # Group by design and plot
            ax = sns.barplot(data=scenario_df, x="budget_fraction", y="total_samples", 
                          hue="design_label", alpha=0.7, errorbar=None)
            
            # Customize plot
            plt.xlabel("Budget Fraction")
            plt.ylabel("Total Sample Count")
            plt.title(f"Sample Counts by Budget and Design Method\n"
                     f"Model: {ability_info['ability_model']}, Scenario: {ability_info['ability_scenario']}\n"
                     f"Cost: {cost_info['cost_model']}")
            plt.grid(True, axis='y', alpha=0.3)
            
            # Format x tick labels as percentages
            plt.xticks(ticks=plt.xticks()[0], 
                      labels=[f"{x*100:.0f}%" for x in sorted(scenario_df["budget_fraction"].unique())])
            
            plt.tight_layout()
            
            # Save figure with ID-based filename
            safe_filename = f"sample_counts_ability_{ability_id}_cost_{cost_id}.{fmt}".replace(" ", "_")
            output_path = os.path.join(output_dir, "samples", safe_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
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
    # Get unique values for ID-based filtering
    ability_ids = get_unique_values(df, "ability_id")
    cost_ids = get_unique_values(df, "cost_id")
    
    # Limit plots
    for ability_id in ability_ids[:3]:  # Limit to first 3 ability IDs
        ability_info = df[df["ability_id"] == ability_id].iloc[0]
        
        for cost_id in cost_ids[:2]:  # Limit to first 2 cost IDs
            cost_info = df[df["cost_id"] == cost_id].iloc[0]
            
            # Filter data
            scenario_df = df[
                (df["ability_id"] == ability_id) & 
                (df["cost_id"] == cost_id)
            ]
            
            if len(scenario_df) == 0:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Create a design label combining adjustment method and sampler type
            scenario_df = scenario_df.copy()  # Create copy to avoid SettingWithCopyWarning
            scenario_df['design_label'] = scenario_df['design_id']
            
            # Group by design and plot
            ax = sns.barplot(data=scenario_df, x="budget_fraction", y="available_budget", 
                          hue="design_label", alpha=0.7, errorbar=None)
            
            # Customize plot
            plt.xlabel("Budget Fraction")
            plt.ylabel("Available Budget (cost units)")
            plt.title(f"Cost Allocation by Budget Fraction\n"
                     f"Model: {ability_info['ability_model']}, Scenario: {ability_info['ability_scenario']}\n"
                     f"Cost: {cost_info['cost_model']}")
            plt.grid(True, axis='y', alpha=0.3)
            
            # Format x tick labels as percentages
            plt.xticks(ticks=plt.xticks()[0], 
                      labels=[f"{x*100:.0f}%" for x in sorted(scenario_df["budget_fraction"].unique())])
            
            plt.tight_layout()
            
            # Save figure with ID-based filename
            safe_filename = f"cost_allocation_ability_{ability_id}_cost_{cost_id}.{fmt}".replace(" ", "_")
            output_path = os.path.join(output_dir, "costs", safe_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved cost allocation plot to {output_path}")

def plot_costs_over_time(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot total evaluation costs over time for each budget constraint in a series.
    For each group of forecasts that share all hyperparameters except date and constraint,
    draw available_budget and gold_standard_cost as lines over time.
    """
    if "ability_date" not in df or "constraint_id" not in df:
        print("Insufficient data for cost-over-time plots")
        return
    df["ability_date"] = pd.to_datetime(df["ability_date"])

    # Determine grouping columns: use ability_model (no date) and other keys
    key_cols = [
        c for c in (
            # "ability_model",
            # "ability_scenario",
            "cost_model",
            "design_id",
            "ability_variant",
            "cost_variant"
        )
        if c in df.columns
    ]
    print(f"Using grouping keys: {key_cols}")
    if not key_cols:
        print("No suitable grouping keys found; cannot plot cost-over-time.")
        return

    groups = df.groupby(key_cols)
    for key, grp in groups:
        if grp["ability_date"].nunique() < 2:
            continue
        grp = grp.sort_values("ability_date")
        plt.figure(figsize=(10,6))
        # plot gold standard
        gold = grp.groupby("ability_date")["gold_standard_cost"].first()
        plt.plot(gold.index, gold.values, "k-o", label="Gold Standard")
        # plot each constraint
        for cid, sub in grp.groupby("constraint_id"):
            sub = sub.sort_values("ability_date")
            bf = sub["budget_fraction"].iloc[0]
            label = f"{cid} ({bf*100:.0f}%)"
            plt.plot(sub["ability_date"], sub["available_budget"], "-o", label=label)
        title_parts = [f"{col}={key[i]}" for i, col in enumerate(key_cols)]
        plt.title("Evaluation Cost Over Time\n" + ", ".join(title_parts))
        plt.xlabel("Date")
        plt.ylabel("Cost")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        outdir = os.path.join(output_dir, "costs_over_time")
        os.makedirs(outdir, exist_ok=True)
        fname = "__".join(str(k) for k in key) + f".{fmt}"
        _style_plots()
        plt.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved cost-over-time plot to {outdir}/{fname}")

def plot_ability_scenario_comparison(plot_df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot comparison between ability scenarios (base, lower, upper CI).
    
    Args:
        plot_df: Pre-processed DataFrame with variant information
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Group by base ability ID to identify scenario families with multiple variants
    scenario_counts = plot_df.groupby(['base_ability_id', 'cost_id'])['ability_variant'].nunique()
    complete_families = scenario_counts[scenario_counts >= 2].reset_index()[['base_ability_id', 'cost_id']]
    
    print(f"Found {len(complete_families)} ability scenario families with at least 2 variants")
    
    if len(complete_families) == 0:
        print("No complete ability scenario families found for comparison plots")
        return
    
    # Limit to first 10 families
    plot_families = complete_families.iloc[:10]
    
    for _, family in plot_families.iterrows():
        base_ability_id = family['base_ability_id']
        cost_id = family['cost_id']
        
        # Get scenarios in this family - fixing to a single cost model
        family_df = plot_df[
            (plot_df['base_ability_id'] == base_ability_id) & 
            (plot_df['cost_id'] == cost_id) &
            (plot_df['cost_variant'] == 'base')  # Use base cost variant
        ].copy()
        
        print(f"Processing ability family: ability={base_ability_id}, cost={cost_id} with {len(family_df)} records")
        
        # Get unique designs for this family
        design_ids = family_df['design_id'].unique()
        
        for design_id in design_ids[:2]:  # Limit to 2 designs per family
            # Filter for this specific design
            filtered_df = family_df[family_df['design_id'] == design_id].copy()
            
            # Get sample information for labels
            if len(filtered_df) > 0:
                sample_row = filtered_df.iloc[0]
                ability_model = sample_row['ability_model']
                cost_model = sample_row['cost_model']
                adjustment_method = sample_row['adjustment_method']
                sampler_type = sample_row['sampler_type']
            else:
                continue
            
            # Sort variants for consistent ordering
            filtered_df['plot_order'] = 0
            filtered_df.loc[filtered_df['ability_variant'] == 'base', 'plot_order'] = 0
            filtered_df.loc[filtered_df['ability_variant'] == 'lower', 'plot_order'] = 1
            filtered_df.loc[filtered_df['ability_variant'] == 'upper', 'plot_order'] = 2
            filtered_df = filtered_df.sort_values('plot_order')
            
            # Get the variants present in this filtered dataset
            variants_present = filtered_df['ability_variant'].unique()
            if len(variants_present) < 2:
                print(f"Skipping ability plot for {ability_model}/{cost_model} - not enough variants")
                continue
                
            # Plot window comparison
            plt.figure(figsize=(12, 8))
            
            # Calculate number of subplots needed
            n_subplots = len(variants_present)
            
            # Plot each variant
            for i, variant in enumerate(variants_present):
                variant_df = filtered_df[filtered_df['ability_variant'] == variant].copy()
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
                variant_title = "Base Ability" if variant == "base" else f"{variant.title()} Ability CI"
                plt.title(variant_title)
                plt.ylabel("Budget Fraction")
                
                # Only show x label on bottom plot
                if i == n_subplots - 1:
                    plt.xlabel("Task Difficulty")
                
                plt.grid(True, alpha=0.3)
                
                # Add legend only to first plot
                if i == 0:
                    plt.legend(loc='upper right')
            
            plt.suptitle(f"Ability Scenario Comparison\n"
                        f"Model: {ability_model}, Cost: {cost_model} (base)\n"
                        f"Method: {adjustment_method}, Sampler: {sampler_type}")
            plt.tight_layout()
            
            # Save figure with ID-based filename
            safe_filename = f"ability_comparison_{base_ability_id}_cost_{cost_id}_design_{design_id}.{fmt}"
            safe_filename = safe_filename.replace(" ", "_")
            output_path = os.path.join(output_dir, "scenario_comparison", safe_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved ability scenario comparison plot to {output_path}")

def plot_cost_scenario_comparison(plot_df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot comparison between cost scenarios (base, lower, upper CI).
    
    Args:
        plot_df: Pre-processed DataFrame with variant information
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Group by base cost ID to identify scenario families with multiple variants
    scenario_counts = plot_df.groupby(['base_cost_id', 'ability_id'])['cost_variant'].nunique()
    complete_families = scenario_counts[scenario_counts >= 2].reset_index()[['base_cost_id', 'ability_id']]
    
    print(f"Found {len(complete_families)} cost scenario families with at least 2 variants")
    
    if len(complete_families) == 0:
        print("No complete cost scenario families found for comparison plots")
        return
    
    # Limit to first 10 families
    plot_families = complete_families.iloc[:10]
    
    for _, family in plot_families.iterrows():
        base_cost_id = family['base_cost_id']
        ability_id = family['ability_id']
        
        # Get scenarios in this family - fixing to a single ability model
        family_df = plot_df[
            (plot_df['base_cost_id'] == base_cost_id) & 
            (plot_df['ability_id'] == ability_id) &
            (plot_df['ability_variant'] == 'base')  # Use base ability variant
        ].copy()
        
        print(f"Processing cost family: ability={ability_id}, cost={base_cost_id} with {len(family_df)} records")
        
        # Get unique designs for this family
        design_ids = family_df['design_id'].unique()
        
        for design_id in design_ids[:2]:  # Limit to 2 designs per family
            # Filter for this specific design
            filtered_df = family_df[family_df['design_id'] == design_id].copy()
            
            # Get sample information for labels
            if len(filtered_df) > 0:
                sample_row = filtered_df.iloc[0]
                ability_model = sample_row['ability_model']
                cost_model = sample_row['cost_model'].replace('_base_ci', '').replace('_lower_ci', '').replace('_upper_ci', '')
                adjustment_method = sample_row['adjustment_method']
                sampler_type = sample_row['sampler_type']
                doubling_rates = {v: filtered_df[filtered_df['cost_variant'] == v]['doubling_rate'].mean() 
                                 for v in filtered_df['cost_variant'].unique() if pd.notna(v)}
            else:
                continue
            
            # Sort variants for consistent ordering
            filtered_df['plot_order'] = 0
            filtered_df.loc[filtered_df['cost_variant'] == 'base', 'plot_order'] = 0
            filtered_df.loc[filtered_df['cost_variant'] == 'lower', 'plot_order'] = 1
            filtered_df.loc[filtered_df['cost_variant'] == 'upper', 'plot_order'] = 2
            filtered_df = filtered_df.sort_values('plot_order')
            
            # Get the variants present in this filtered dataset
            variants_present = filtered_df['cost_variant'].unique()
            if len(variants_present) < 2:
                print(f"Skipping cost plot for {ability_model}/{cost_model} - not enough variants")
                continue
                
            # Plot window comparison
            plt.figure(figsize=(12, 8))
            
            # Calculate number of subplots needed
            n_subplots = len(variants_present)
            
            # Plot each variant
            for i, variant in enumerate(variants_present):
                variant_df = filtered_df[filtered_df['cost_variant'] == variant].copy()
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
                dr = doubling_rates.get(variant, np.nan)
                if variant == "base":
                    variant_title = f"Base Cost (DR = {dr:.2f})"
                elif variant == "lower":
                    variant_title = f"Lower Cost CI (DR = {dr:.2f}, faster growth)"
                else:
                    variant_title = f"Upper Cost CI (DR = {dr:.2f}, slower growth)"
                
                plt.title(variant_title)
                plt.ylabel("Budget Fraction")
                
                # Only show x label on bottom plot
                if i == n_subplots - 1:
                    plt.xlabel("Task Difficulty")
                
                plt.grid(True, alpha=0.3)
                
                # Add legend only to first plot
                if i == 0:
                    plt.legend(loc='upper right')
            
            plt.suptitle(f"Cost Scenario Comparison\n"
                        f"Model: {ability_model} (base), Cost: {cost_model}\n"
                        f"Method: {adjustment_method}, Sampler: {sampler_type}")
            plt.tight_layout()
            
            # Save figure with ID-based filename
            safe_filename = f"cost_comparison_ability_{ability_id}_cost_{base_cost_id}_design_{design_id}.{fmt}"
            safe_filename = safe_filename.replace(" ", "_")
            output_path = os.path.join(output_dir, "scenario_comparison", safe_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved cost scenario comparison plot to {output_path}")

def plot_combined_scenario_comparison(plot_df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot combined comparison showing both ability and cost confidence intervals.
    
    Args:
        plot_df: Pre-processed DataFrame with variant information
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Find combinations with multiple ability and cost variants
    ability_counts = plot_df.groupby(['base_ability_id'])['ability_variant'].nunique()
    cost_counts = plot_df.groupby(['base_cost_id'])['cost_variant'].nunique()
    
    ability_families = ability_counts[ability_counts >= 2].index.tolist()
    cost_families = cost_counts[cost_counts >= 2].index.tolist()
    
    if not ability_families or not cost_families:
        print("Not enough variant families to create combined comparison plots")
        return
    
    print(f"Found {len(ability_families)} ability families and {len(cost_families)} cost families for combined plots")
    
    # Select a few interesting combinations to visualize
    combined_scenarios = []
    for base_ability_id in ability_families[:2]:
        for base_cost_id in cost_families[:2]:
            # Check if this combination exists in the data
            combo_df = plot_df[
                (plot_df['base_ability_id'] == base_ability_id) &
                (plot_df['base_cost_id'] == base_cost_id)
            ]
            if len(combo_df) > 0:
                combined_scenarios.append((base_ability_id, base_cost_id))
    
    if not combined_scenarios:
        print("No viable combinations found for combined scenario plots")
        return
    
    print(f"Creating combined scenario plots for {len(combined_scenarios)} combinations")
    
    # For each combination, create a grid plot
    for base_ability_id, base_cost_id in combined_scenarios:
        # Get all variants for this combination
        combo_df = plot_df[
            (plot_df['base_ability_id'] == base_ability_id) &
            (plot_df['base_cost_id'] == base_cost_id)
        ].copy()
        
        # Get unique designs for this combination
        design_ids = combo_df['design_id'].unique()
        
        for design_id in design_ids[:1]:  # Just use the first design to keep it manageable
            # Filter for this specific design
            filtered_df = combo_df[combo_df['design_id'] == design_id].copy()
            
            # Get variants present
            ability_variants = sorted(filtered_df['ability_variant'].unique())
            cost_variants = sorted(filtered_df['cost_variant'].unique())
            
            # Need at least 2 variants in each dimension
            if len(ability_variants) < 2 or len(cost_variants) < 2:
                print(f"Skipping combined plot for ability={base_ability_id}, cost={base_cost_id} - not enough variants")
                continue
                
            # Get sample information for labels
            if len(filtered_df) > 0:
                sample_row = filtered_df.iloc[0]
                ability_model = sample_row['ability_model'].replace('_base', '').replace('_lower_ci', '').replace('_upper_ci', '')
                cost_model = sample_row['cost_model'].replace('_base', '').replace('_lower_ci', '').replace('_upper_ci', '')
                adjustment_method = sample_row['adjustment_method']
                sampler_type = sample_row['sampler_type']
            else:
                continue
            
            # Create a grid plot showing combinations
            fig = plt.figure(figsize=(16, 12))
            
            # Set up grid dimensions
            n_rows = len(ability_variants)
            n_cols = len(cost_variants)
            
            # Create a grid of subplots
            for i, ability_variant in enumerate(ability_variants):
                for j, cost_variant in enumerate(cost_variants):
                    # Filter for this specific combination
                    cell_df = filtered_df[
                        (filtered_df['ability_variant'] == ability_variant) &
                        (filtered_df['cost_variant'] == cost_variant)
                    ]
                    
                    # Skip if no data
                    if len(cell_df) == 0:
                        continue
                    
                    # Create subplot
                    plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
                    
                    # Plot windows for different budget fractions
                    cell_df = cell_df.sort_values('budget_fraction')
                    for _, row in cell_df.iterrows():
                        budget_fraction = row["budget_fraction"]
                        plt.plot([row["window_lower"], row["window_upper"]], 
                                [budget_fraction, budget_fraction], 
                                linewidth=2, marker='|',
                                label=f"{budget_fraction*100:.0f}% Budget")
                    
                    # Mark threshold
                    if len(cell_df) > 0:
                        threshold = cell_df.iloc[0]["ability_threshold"]
                        doubling_rate = cell_df.iloc[0]["doubling_rate"]
                        plt.axvline(threshold, color='r', linestyle='--')
                    
                    # Consistent axis limits across plots
                    plt.xlim(filtered_df["window_lower"].min() - 1, filtered_df["window_upper"].max() + 1)
                    plt.ylim(filtered_df["budget_fraction"].min() - 0.05, filtered_df["budget_fraction"].max() + 0.05)
                    
                    # Set title for each cell
                    ability_label = f"{ability_variant.title()} Ability"
                    cost_label = f"{cost_variant.title()} Cost (DR={doubling_rate:.2f})"
                    plt.title(f"{ability_label} + {cost_label}")
                    
                    # Only add y labels to leftmost plots
                    if j == 0:
                        plt.ylabel("Budget Fraction")
                    
                    # Only add x labels to bottom plots
                    if i == n_rows - 1:
                        plt.xlabel("Task Difficulty")
                    
                    plt.grid(True, alpha=0.3)
                    
                    # Add legend only to first plot
                    if i == 0 and j == 0:
                        plt.legend(loc='upper right', fontsize='small')
            
            plt.suptitle(f"Combined Scenario Comparison\n"
                        f"Model: {ability_model}, Cost: {cost_model}\n"
                        f"Method: {adjustment_method}, Sampler: {sampler_type}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            
            # Save figure with ID-based filename
            safe_filename = f"combined_comparison_ability_{base_ability_id}_cost_{base_cost_id}_design_{design_id}.{fmt}"
            safe_filename = safe_filename.replace(" ", "_")
            output_path = os.path.join(output_dir, "scenario_comparison", safe_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved combined scenario comparison plot to {output_path}")

def plot_scenario_comparison(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Plot comparison between base, lower, and upper CI scenarios for both ability and cost.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Make a copy to avoid modifying the original DataFrame
    plot_df = df.copy()
    
    # Check if the variant fields are already present in the data
    has_variants = all(col in plot_df.columns for col in ['ability_variant', 'cost_variant', 'base_ability_id', 'base_cost_id'])
    
    if not has_variants:
        # If not present, extract them from IDs (backward compatibility)
        print("Variant fields not found in data, extracting from IDs...")
        # Extract variant type from ability_id (base, lower, upper)
        plot_df['ability_variant'] = 'unknown'
        plot_df.loc[plot_df['ability_id'].str.contains('_base'), 'ability_variant'] = 'base'
        plot_df.loc[plot_df['ability_id'].str.contains('_lower'), 'ability_variant'] = 'lower'
        plot_df.loc[plot_df['ability_id'].str.contains('_upper'), 'ability_variant'] = 'upper'
        
        # Extract variant type from cost_id (base, lower, upper)
        plot_df['cost_variant'] = 'unknown'
        plot_df.loc[plot_df['cost_id'].str.contains('_base'), 'cost_variant'] = 'base'
        plot_df.loc[plot_df['cost_id'].str.contains('_lower'), 'cost_variant'] = 'lower'
        plot_df.loc[plot_df['cost_id'].str.contains('_upper'), 'cost_variant'] = 'upper'
        
        # Extract the base part of the IDs (remove _base, _lower, _upper suffixes)
        plot_df['base_ability_id'] = plot_df['ability_id'].str.replace('_lower$|_upper$|_base$', '', regex=True)
        plot_df['base_cost_id'] = plot_df['cost_id'].str.replace('_lower$|_upper$|_base$', '', regex=True)
    else:
        print("Using pre-computed variant fields from data")
    
    # Print some debug info about the available variants
    ability_variants = plot_df['ability_variant'].value_counts().to_dict()
    cost_variants = plot_df['cost_variant'].value_counts().to_dict()
    print(f"Available ability variants: {ability_variants}")
    print(f"Available cost variants: {cost_variants}")
    
    # Run the different types of comparisons
    plot_ability_scenario_comparison(plot_df, output_dir, fmt)
    plot_cost_scenario_comparison(plot_df, output_dir, fmt)
    plot_combined_scenario_comparison(plot_df, output_dir, fmt)

def create_summary_stats(df: pd.DataFrame, output_dir: str):
    """
    Create summary statistics for the evaluation forecast data.
    
    Args:
        df: DataFrame with evaluation forecast data
        output_dir: Directory to save summary
    """
    # Group by key IDs and generate summary stats
    summary = df.groupby(["ability_id", "cost_id", "constraint_id", "design_id"]).agg({
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
    simple_summary = df.groupby(["ability_id", "ability_model", "ability_scenario"]).agg({
        "budget_fraction": "nunique",
        "cost_id": "nunique",
        "constraint_id": "nunique",
        "design_id": "nunique",
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
    
    # Set global plotting parameters
    global FONT_SIZE, TITLE_ENABLED
    FONT_SIZE = args.font_size
    TITLE_ENABLED = not args.disable_titles
    
    print(f"Loading evaluation forecasts from {args.input}")
    df = pd.read_csv(args.input)
    
    # Convert date column to datetime if needed
    if "ability_date" in df.columns:
        df["ability_date"] = pd.to_datetime(df["ability_date"])
    
    print(f"Loaded {len(df)} evaluation forecasts")
    
    # Apply filters if specified
    original_len = len(df)
    
    # Filter by IDs
    if args.filter_ability:
        df = df[df["ability_id"] == args.filter_ability].copy()
        print(f"Filtered to {len(df)} records for ability_id: {args.filter_ability}")
    
    if args.filter_cost:
        df = df[df["cost_id"] == args.filter_cost].copy()
        print(f"Filtered to {len(df)} records for cost_id: {args.filter_cost}")
        
    if args.filter_constraint:
        df = df[df["constraint_id"] == args.filter_constraint].copy()
        print(f"Filtered to {len(df)} records for constraint_id: {args.filter_constraint}")
        
    if args.filter_design:
        df = df[df["design_id"] == args.filter_design].copy()
        print(f"Filtered to {len(df)} records for design_id: {args.filter_design}")
    
    # Apply legacy filters (backward compatibility)
    if args.filter_model:
        df = df[df["ability_model"] == args.filter_model].copy()
        print(f"Filtered to {len(df)} records for model: {args.filter_model}")
    
    if args.filter_scenario:
        df = df[df["ability_scenario"] == args.filter_scenario].copy()
        print(f"Filtered to {len(df)} records for scenario: {args.filter_scenario}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    # plot_evaluation_windows(df, args.output, args.format)
    # plot_window_adjustments(df, args.output, args.format)
    # plot_task_density(df, args.output, args.format)
    # plot_sample_counts(df, args.output, args.format)
    # plot_cost_allocation(df, args.output, args.format)
    # plot_scenario_comparison(df, args.output, args.format)
    plot_costs_over_time(df, args.output, args.format)
    # create_summary_stats(df, args.output)
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
