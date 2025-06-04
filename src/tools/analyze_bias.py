"""
Analyze bias results from CSV data and generate visualizations 
focusing on bias-variance tradeoffs as budget fraction varies.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from collections import defaultdict

def remove_outliers(df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """Drop extreme bias, variance, mse beyond given quantile."""
    thresh_b = df['bias'].abs().quantile(quantile)
    thresh_v = df['variance'].quantile(quantile)
    thresh_m = df['mse'].quantile(quantile) if 'mse' in df.columns else np.inf
    return df[
        (df['bias'].abs() <= thresh_b) &
        (df['variance'] <= thresh_v) &
        (df.get('mse', df['variance']) <= thresh_m)
    ]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze bias results from CSV data")
    parser.add_argument("--input", required=True, help="Path to bias results CSV file")
    parser.add_argument("--output", default="reports/bias_analysis",
                        help="Output directory for visualizations")
    parser.add_argument("--format", default="png", help="Output format (png, pdf, svg)")
    parser.add_argument("--filter-ability-scenario", help="Filter by ability scenario")
    parser.add_argument("--filter-ability-variant", choices=["base", "lower", "upper"], 
                        help="Filter by ability variant (if available)")
    parser.add_argument("--filter-cost-variant", choices=["base", "lower", "upper"], 
                        help="Filter by cost variant (if available)")
    parser.add_argument("--filter-ability-id", help="Filter by specific ability ID")
    parser.add_argument("--filter-cost-id", help="Filter by specific cost ID")
    parser.add_argument("--filter-base-ability-id", help="Filter by specific base ability ID")
    parser.add_argument("--filter-base-cost-id", help="Filter by specific base cost ID")
    parser.add_argument("--filter-design-id", help="Filter by specific design ID")
    parser.add_argument("--filter-estimator", choices=["threshold", "weighted_score"], 
                        help="Filter by estimator type")
    parser.add_argument("--filter-date", help="Filter by specific date (YYYY-MM-DD)")
    parser.add_argument("--max-lines", type=int, default=8, 
                        help="Maximum number of lines per plot before splitting")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed output")
    parser.add_argument("--remove-outliers", action="store_true",
                        help="Drop extreme outliers before plotting")
    parser.add_argument("--outlier-quantile", type=float, default=0.99,
                        help="Quantile threshold for outlier removal")
    return parser.parse_args()

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load bias results from CSV and preprocess for analysis.
    
    Args:
        csv_path: Path to bias results CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract budget fraction from budget scenario string if not already present
    if 'budget_fraction' not in df.columns and 'budget_scenario' in df.columns:
        # Try to extract budget fraction from scenario name (e.g., "static_75pct" → 0.75)
        def extract_budget(scenario):
            try:
                if 'static_' in scenario:
                    pct = int(scenario.split('static_')[1].split('pct')[0])
                    return pct / 100.0
                return np.nan
            except:
                return np.nan
        
        df['budget_fraction'] = df['budget_scenario'].apply(extract_budget)
    
    # Extract variant information if available in budget_scenario or ability_scenario
    if 'ability_variant' not in df.columns:
        df['ability_variant'] = 'unknown'
        if 'ability_scenario' in df.columns:
            df.loc[df['ability_scenario'].str.contains('_lower_ci', na=False), 'ability_variant'] = 'lower'
            df.loc[df['ability_scenario'].str.contains('_upper_ci', na=False), 'ability_variant'] = 'upper'
            df.loc[~df['ability_scenario'].str.contains('_lower_ci|_upper_ci', na=False), 'ability_variant'] = 'base'
    
    if 'cost_variant' not in df.columns:
        df['cost_variant'] = 'unknown'
        if 'budget_scenario' in df.columns:
            df.loc[df['budget_scenario'].str.contains('_lower', na=False), 'cost_variant'] = 'lower'
            df.loc[df['budget_scenario'].str.contains('_upper', na=False), 'cost_variant'] = 'upper'
            df.loc[~df['budget_scenario'].str.contains('_lower|_upper', na=False), 'cost_variant'] = 'base'
    
    # Create base IDs if not present
    if 'base_ability_id' not in df.columns and 'ability_id' in df.columns:
        df['base_ability_id'] = df['ability_id'].apply(
            lambda x: x[:-6] if x.endswith('_lower') or x.endswith('_upper') else 
                     (x[:-5] if x.endswith('_base') else x)
        )
    
    if 'base_cost_id' not in df.columns and 'cost_id' in df.columns:
        df['base_cost_id'] = df['cost_id'].apply(
            lambda x: x[:-6] if x.endswith('_lower') or x.endswith('_upper') else 
                     (x[:-5] if x.endswith('_base') else x)
        )
    
    # Create scenario keys for various grouping needs
    if all(col in df.columns for col in ['ability_id', 'cost_id', 'design_id']):
        # Full scenario key (unique per scenario)
        df['scenario_key'] = df['ability_id'] + '__|__' + df['cost_id'] + '__|__' + df['design_id']
        
        # Base scenario key (groups all variants of the same base scenario)
        if all(col in df.columns for col in ['base_ability_id', 'base_cost_id']):
            df['base_scenario_key'] = df['base_ability_id'] + '__|__' + df['base_cost_id'] + '__|__' + df['design_id']
            
        # Variant scenario key (groups by ability variant, using base cost)
        df['ability_variant_key'] = df['base_ability_id'] + '__|__' + df['ability_variant'] + '__|__' + df['cost_id'] + '__|__' + df['design_id']
        
        # Variant scenario key (groups by cost variant, using base ability)
        df['cost_variant_key'] = df['ability_id'] + '__|__' + df['base_cost_id'] + '__|__' + df['cost_variant'] + '__|__' + df['design_id']
    else:
        # Fallback to ability_scenario if the ID columns don't exist
        df['scenario_key'] = df['ability_scenario']
        df['base_scenario_key'] = df['ability_scenario']
        df['ability_variant_key'] = df['ability_scenario']
        df['cost_variant_key'] = df['ability_scenario']
    
    # Calculate MSE (bias² + variance)
    df['mse'] = df['bias']**2 + df['variance']
    
    # Calculate coverage ratio
    if 'contains_true' in df.columns:
        df['contains_true_int'] = df['contains_true'].astype(int)
    
    return df

def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Apply command-line filters to the DataFrame.
    
    Args:
        df: DataFrame with bias results
        args: Command-line arguments
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if args.filter_ability_scenario:
        if 'ability_scenario' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['ability_scenario'].str.contains(args.filter_ability_scenario)]
            print(f"Filtered to {len(filtered_df)} records with ability scenario containing '{args.filter_ability_scenario}'")
    
    if args.filter_ability_variant:
        if 'ability_variant' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['ability_variant'] == args.filter_ability_variant]
            print(f"Filtered to {len(filtered_df)} records with ability variant '{args.filter_ability_variant}'")
    
    if args.filter_cost_variant:
        if 'cost_variant' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['cost_variant'] == args.filter_cost_variant]
            print(f"Filtered to {len(filtered_df)} records with cost variant '{args.filter_cost_variant}'")
            
    if args.filter_ability_id:
        if 'ability_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['ability_id'] == args.filter_ability_id]
            print(f"Filtered to {len(filtered_df)} records with ability ID '{args.filter_ability_id}'")
    
    if args.filter_cost_id:
        if 'cost_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['cost_id'] == args.filter_cost_id]
            print(f"Filtered to {len(filtered_df)} records with cost ID '{args.filter_cost_id}'")
    
    if args.filter_base_ability_id:
        if 'base_ability_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['base_ability_id'] == args.filter_base_ability_id]
            print(f"Filtered to {len(filtered_df)} records with base ability ID '{args.filter_base_ability_id}'")
    
    if args.filter_base_cost_id:
        if 'base_cost_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['base_cost_id'] == args.filter_base_cost_id]
            print(f"Filtered to {len(filtered_df)} records with base cost ID '{args.filter_base_cost_id}'")
    
    if args.filter_design_id:
        if 'design_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['design_id'] == args.filter_design_id]
            print(f"Filtered to {len(filtered_df)} records with design ID '{args.filter_design_id}'")
    
    if args.filter_estimator:
        filtered_df = filtered_df[filtered_df['estimator'] == args.filter_estimator]
        print(f"Filtered to {len(filtered_df)} records with estimator '{args.filter_estimator}'")
    
    if args.filter_date:
        if 'date' in filtered_df.columns:
            target_date = pd.to_datetime(args.filter_date)
            filtered_df = filtered_df[filtered_df['date'].dt.date == target_date.date()]
            print(f"Filtered to {len(filtered_df)} records with date '{args.filter_date}'")
    
    return filtered_df

def get_scenario_info(row: pd.Series) -> Dict[str, str]:
    """
    Extract descriptive information for a scenario from a DataFrame row.
    
    Args:
        row: Row from the DataFrame
        
    Returns:
        Dictionary with scenario information
    """
    info = {}
    
    # Extract ability info
    ability_info = row.get('ability_scenario', '')
    if 'ability_variant' in row and row['ability_variant'] != 'unknown':
        ability_info += f" ({row['ability_variant']})"
    info['ability_info'] = ability_info
        
    # Extract cost info
    cost_info = ''
    if 'cost_id' in row:
        cost_info = row['cost_id']
        if 'cost_variant' in row and row['cost_variant'] != 'unknown':
            cost_info += f" ({row['cost_variant']})"
    info['cost_info'] = cost_info
            
    # Extract design info
    design_info = ''
    if 'design_id' in row:
        design_info = row['design_id']
    info['design_info'] = design_info
    
    # Create a display name
    display_name = ability_info
    if 'ability_variant' in row and row['ability_variant'] != 'base':
        display_name += f" ({row['ability_variant']})"
    if 'cost_variant' in row and row['cost_variant'] != 'base':
        display_name += f" | Cost: {row['cost_variant']}"
    info['display_name'] = display_name
    
    return info

def plot_bias_variance_vs_budget(df: pd.DataFrame, output_dir: str, fmt: str = "png", max_lines: int = 8):
    """
    Create plots of bias and variance vs. budget fraction.
    
    Args:
        df: DataFrame with bias results
        output_dir: Directory to save plots
        fmt: File format for output
        max_lines: Maximum number of lines per plot before splitting
    """
    # Ensure budget_fraction is present
    if 'budget_fraction' not in df.columns or df['budget_fraction'].isna().all():
        print("Cannot plot bias/variance vs. budget: missing budget_fraction column")
        return
    
    # Group by estimator for separate plots
    for estimator, estimator_df in df.groupby('estimator'):
        # Create a directory for this estimator
        estimator_dir = os.path.join(output_dir, estimator)
        os.makedirs(estimator_dir, exist_ok=True)
        
        # Create different types of groupings for effective comparison
        
        # 1. Individual scenarios (most detailed)
        individual_plots(estimator_df, 'scenario_key', estimator_dir, fmt, 'bias_variance', max_lines,
                        plot_function=plot_bias_variance_for_scenario, 
                        title_prefix=f"Bias and Variance vs. Budget ({estimator})")
        
        # 2. Base scenarios (ignoring variants)
        if 'base_scenario_key' in estimator_df.columns:
            grouped_variant_plots(estimator_df, 'base_scenario_key', 'ability_variant', estimator_dir, fmt, 
                                 'bias_variance_by_variant', max_lines,
                                 plot_function=plot_bias_variance_for_variant_group,
                                 title_prefix=f"Bias and Variance by Variant ({estimator})")
        
            # 3. Cost variant comparison (fixed ability, varying cost)
            grouped_variant_plots(estimator_df, 'cost_variant_key', 'cost_variant', estimator_dir, fmt, 
                                 'bias_variance_by_cost', max_lines,
                                 plot_function=plot_bias_variance_for_variant_group,
                                 title_prefix=f"Bias and Variance by Cost Variant ({estimator})")

def individual_plots(df: pd.DataFrame, key_column: str, output_dir: str, fmt: str, name_prefix: str, 
                    max_lines: int, plot_function, title_prefix: str):
    """
    Create individual plots for each unique scenario.
    
    Args:
        df: DataFrame with bias results
        key_column: Column to use for grouping scenarios
        output_dir: Directory to save plots
        fmt: File format for output
        name_prefix: Prefix for output filenames
        max_lines: Maximum number of lines per plot before splitting
        plot_function: Function to use for plotting
        title_prefix: Prefix for plot titles
    """
    # Get unique scenario keys
    scenario_keys = df[key_column].unique()
    
    print(f"Creating {len(scenario_keys)} {name_prefix} plots")
    
    # For each unique scenario combination
    for scenario_key in scenario_keys:
        scenario_df = df[df[key_column] == scenario_key].copy()
        
        # Sort by budget fraction for proper line plotting
        scenario_df = scenario_df.sort_values('budget_fraction')
        
        # Skip if we don't have enough budget fractions to make a meaningful plot
        if len(scenario_df['budget_fraction'].unique()) <= 1:
            continue
        
        # Get descriptive info for this scenario
        sample_row = scenario_df.iloc[0]
        scenario_info = get_scenario_info(sample_row)
        
        # Create the plot
        plot_function(scenario_df, scenario_info, title_prefix)
        
        # Generate a safe filename from scenario key
        safe_key = str(scenario_key).replace('/', '_').replace(' ', '_')
        safe_key = "".join(c if c.isalnum() or c in "_-." else "_" for c in safe_key)
        
        # Save the plot
        output_path = os.path.join(output_dir, f"{name_prefix}_{safe_key}.{fmt}")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"  Saved {name_prefix} plot for scenario: {safe_key}")

def plot_bias_variance_for_scenario(df: pd.DataFrame, scenario_info: Dict[str, str], title_prefix: str):
    """
    Plot bias and variance vs budget for a single scenario.
    
    Args:
        df: DataFrame with data for a single scenario
        scenario_info: Dictionary with scenario descriptive information
        title_prefix: Prefix for the plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot bias vs budget
    plt.plot(df['budget_fraction'], df['bias'], 
            marker='o', linestyle='-', color='blue', 
            label="Bias")
    
    # Plot variance vs budget
    plt.plot(df['budget_fraction'], df['variance'], 
            marker='s', linestyle='--', color='red', alpha=0.7,
            label="Variance")
    
    # Calculate MSE for reference
    plt.plot(df['budget_fraction'], df['mse'], 
            marker='x', linestyle=':', color='purple', alpha=0.5,
            label="MSE = Bias² + Variance")
    
    plt.xlabel('Budget Fraction')
    plt.ylabel('Value (bias, variance, or MSE)')
    
    # Build the title
    title = title_prefix
    if scenario_info['ability_info']:
        title += f'\n{scenario_info["ability_info"]}'
    if scenario_info['cost_info']:
        title += f'\nCost: {scenario_info["cost_info"]}'
    if scenario_info['design_info']:
        title += f'\nDesign: {scenario_info["design_info"]}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.legend()
    plt.tight_layout()

def grouped_variant_plots(df: pd.DataFrame, group_column: str, variant_column: str, 
                         output_dir: str, fmt: str, name_prefix: str, max_lines: int,
                         plot_function, title_prefix: str):
    """
    Create grouped plots showing different variants for the same base scenario.
    
    Args:
        df: DataFrame with bias results
        group_column: Column to use for grouping scenarios
        variant_column: Column containing variant information
        output_dir: Directory to save plots
        fmt: File format for output
        name_prefix: Prefix for output filenames
        max_lines: Maximum number of lines per plot before splitting
        plot_function: Function to use for plotting
        title_prefix: Prefix for plot titles
    """
    # Group by the specified column
    grouped = df.groupby(group_column)
    
    print(f"Creating {len(grouped)} {name_prefix} plots")
    
    # For each group, create a plot showing the different variants
    for group_key, group_df in grouped:
        # Check if we have multiple variants
        variants = group_df[variant_column].unique()
        
        if len(variants) <= 1:
            continue  # Skip if only one variant
        
        # Get sample row for group info
        sample_row = group_df.iloc[0]
        group_info = get_scenario_info(sample_row)
        
        # Create the plot
        plot_function(group_df, variant_column, group_info, title_prefix)
        
        # Generate a safe filename
        safe_key = str(group_key).replace('/', '_').replace(' ', '_')
        safe_key = "".join(c if c.isalnum() or c in "_-." else "_" for c in safe_key)
        
        # Save the plot
        output_path = os.path.join(output_dir, f"{name_prefix}_{safe_key}.{fmt}")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"  Saved {name_prefix} plot for group: {safe_key}")

def plot_bias_variance_for_variant_group(df: pd.DataFrame, variant_column: str, 
                                        group_info: Dict[str, str], title_prefix: str):
    """
    Plot bias and variance vs budget for a group of variants.
    
    Args:
        df: DataFrame with data for a variant group
        variant_column: Column containing variant information
        group_info: Dictionary with group descriptive information
        title_prefix: Prefix for the plot title
    """
    plt.figure(figsize=(14, 10))
    
    # Get unique variants
    variants = sorted(df[variant_column].unique())
    
    # Create color palette
    palette = sns.color_palette('husl', len(variants))
    
    # Plot each variant
    for i, variant in enumerate(variants):
        variant_df = df[df[variant_column] == variant].copy()
        variant_df = variant_df.sort_values('budget_fraction')
        
        # Plot bias
        plt.plot(variant_df['budget_fraction'], variant_df['bias'], 
                marker='o', linestyle='-', color=palette[i], 
                label=f"{variant} (bias)")
        
        # Plot variance
        plt.plot(variant_df['budget_fraction'], variant_df['variance'], 
                marker='s', linestyle='--', color=palette[i], alpha=0.5,
                label=f"{variant} (variance)")
    
    plt.xlabel('Budget Fraction')
    plt.ylabel('Value (bias or variance)')
    
    # Build the title
    title = title_prefix
    
    # Remove variant info from ability or cost if the plot is specifically about that variant
    if variant_column == 'ability_variant':
        title += f'\n{group_info["ability_info"].split(" (")[0]}'
    else:
        title += f'\n{group_info["ability_info"]}'
        
    if variant_column == 'cost_variant':
        if group_info['cost_info']:
            title += f'\nCost: {group_info["cost_info"].split(" (")[0]}'
    else:
        if group_info['cost_info']:
            title += f'\nCost: {group_info["cost_info"]}'
            
    if group_info['design_info']:
        title += f'\nDesign: {group_info["design_info"]}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Adjust legend for better readability
    plt.legend(fontsize='small')
    
    plt.tight_layout()

def plot_mse_vs_budget(df: pd.DataFrame, output_dir: str, fmt: str = "png", max_lines: int = 8):
    """
    Create plots of Mean Squared Error vs. budget fraction.
    
    Args:
        df: DataFrame with bias results
        output_dir: Directory to save plots
        fmt: File format for output
        max_lines: Maximum number of lines per plot
    """
    if 'budget_fraction' not in df.columns or df['budget_fraction'].isna().all():
        print("Cannot plot MSE vs. budget: missing budget_fraction column")
        return
    
    # Group by estimator for separate plots
    for estimator, estimator_df in df.groupby('estimator'):
        # Create a directory for this estimator
        estimator_dir = os.path.join(output_dir, estimator)
        os.makedirs(estimator_dir, exist_ok=True)
        
        # Individual scenario plots
        individual_plots(estimator_df, 'scenario_key', estimator_dir, fmt, 'mse', max_lines,
                        plot_function=plot_mse_for_scenario, 
                        title_prefix=f"Mean Squared Error vs. Budget ({estimator})")
        
        # Variant comparison plots
        if 'base_scenario_key' in estimator_df.columns:
            grouped_variant_plots(estimator_df, 'base_scenario_key', 'ability_variant', estimator_dir, fmt, 
                                 'mse_by_ability_variant', max_lines,
                                 plot_function=plot_mse_for_variant_group,
                                 title_prefix=f"MSE by Ability Variant ({estimator})")
            
            grouped_variant_plots(estimator_df, 'cost_variant_key', 'cost_variant', estimator_dir, fmt, 
                                 'mse_by_cost_variant', max_lines,
                                 plot_function=plot_mse_for_variant_group,
                                 title_prefix=f"MSE by Cost Variant ({estimator})")
            
            # Create a comprehensive overview plot showing MSE trends
            plot_mse_overview(estimator_df, estimator_dir, fmt)

def plot_mse_for_scenario(df: pd.DataFrame, scenario_info: Dict[str, str], title_prefix: str):
    """
    Plot MSE vs budget for a single scenario with decomposition.
    
    Args:
        df: DataFrame with data for a single scenario
        scenario_info: Dictionary with scenario descriptive information
        title_prefix: Prefix for the plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by budget fraction
    df = df.sort_values('budget_fraction')
    
    # Plot MSE
    plt.plot(df['budget_fraction'], df['mse'], 
            marker='o', linestyle='-', color='purple', 
            label="MSE")
    
    # Add decomposition showing bias squared component
    bias_squared = df['bias']**2
    plt.fill_between(
        df['budget_fraction'], 
        0, 
        bias_squared,
        color='blue', alpha=0.3, label="Bias²"
    )
    
    # Add decomposition showing variance component
    plt.fill_between(
        df['budget_fraction'], 
        bias_squared, 
        df['mse'],
        color='red', alpha=0.3, label="Variance"
    )
    
    # Add annotations for min and max budget
    if len(df) > 0:
        first_idx = df['budget_fraction'].idxmin()
        last_idx = df['budget_fraction'].idxmax()
        
        for idx, label in [(first_idx, 'Min Budget'), (last_idx, 'Max Budget')]:
            row = df.loc[idx]
            bias_contrib = (row['bias']**2 / row['mse']) * 100 if row['mse'] > 0 else np.nan
            var_contrib = (row['variance'] / row['mse']) * 100 if row['mse'] > 0 else np.nan
            
            if not np.isnan(bias_contrib) and not np.isnan(var_contrib):
                plt.annotate(
                    f"{label}\nBias²: {bias_contrib:.1f}%\nVar: {var_contrib:.1f}%",
                    xy=(row['budget_fraction'], row['mse']),
                    xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
                )
    
    plt.xlabel('Budget Fraction')
    plt.ylabel('Mean Squared Error (MSE)')
    
    # Build the title
    title = title_prefix
    if scenario_info['ability_info']:
        title += f'\n{scenario_info["ability_info"]}'
    if scenario_info['cost_info']:
        title += f'\nCost: {scenario_info["cost_info"]}'
    if scenario_info['design_info']:
        title += f'\nDesign: {scenario_info["design_info"]}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

def plot_mse_for_variant_group(df: pd.DataFrame, variant_column: str, 
                              group_info: Dict[str, str], title_prefix: str):
    """
    Plot MSE vs budget for a group of variants.
    
    Args:
        df: DataFrame with data for a variant group
        variant_column: Column containing variant information
        group_info: Dictionary with group descriptive information
        title_prefix: Prefix for the plot title
    """
    plt.figure(figsize=(14, 10))
    
    # Get unique variants
    variants = sorted(df[variant_column].unique())
    
    # Create color palette
    palette = sns.color_palette('husl', len(variants))
    
    # Plot each variant
    for i, variant in enumerate(variants):
        variant_df = df[df[variant_column] == variant].copy()
        variant_df = variant_df.sort_values('budget_fraction')
        
        # Plot MSE
        plt.plot(variant_df['budget_fraction'], variant_df['mse'], 
                marker='o', linestyle='-', color=palette[i], 
                label=f"{variant}")
        
        # Show bias contribution with shading
        bias_squared = variant_df['bias']**2
        plt.fill_between(
            variant_df['budget_fraction'], 
            0,
            bias_squared, 
            color=palette[i], alpha=0.1
        )
        
        # Add label for last point
        if len(variant_df) > 0:
            last_row = variant_df.iloc[-1]
            bias_contrib = (last_row['bias']**2 / last_row['mse']) * 100 if last_row['mse'] > 0 else np.nan
            var_contrib = (last_row['variance'] / last_row['mse']) * 100 if last_row['mse'] > 0 else np.nan
            
            if not np.isnan(bias_contrib) and not np.isnan(var_contrib):
                plt.annotate(
                    f"{variant}\nBias²: {bias_contrib:.1f}%\nVar: {var_contrib:.1f}%",
                    xy=(last_row['budget_fraction'], last_row['mse']),
                    xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=8
                )
    
    plt.xlabel('Budget Fraction')
    plt.ylabel('Mean Squared Error (MSE)')
    
    # Build the title
    title = title_prefix
    
    # Remove variant info from ability or cost if the plot is specifically about that variant
    if variant_column == 'ability_variant':
        title += f'\n{group_info["ability_info"].split(" (")[0]}'
    else:
        title += f'\n{group_info["ability_info"]}'
        
    if variant_column == 'cost_variant':
        if group_info['cost_info']:
            title += f'\nCost: {group_info["cost_info"].split(" (")[0]}'
    else:
        if group_info['cost_info']:
            title += f'\nCost: {group_info["cost_info"]}'
            
    if group_info['design_info']:
        title += f'\nDesign: {group_info["design_info"]}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Adjust legend for better readability
    plt.legend(fontsize='small')
    
    plt.tight_layout()

def plot_mse_overview(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Create an overview plot showing MSE trends across different scenarios.
    
    Args:
        df: DataFrame with bias results
        output_dir: Directory to save plots
        fmt: File format for output
    """
    # Only plot if we have base IDs
    if not all(col in df.columns for col in ['base_ability_id', 'base_cost_id']):
        return
    
    # Group by base ability ID and get max MSE at highest budget
    overview_data = []
    
    for base_ability_id, ability_group in df.groupby('base_ability_id'):
        # For each ability ID, get the variants
        for ability_variant in sorted(ability_group['ability_variant'].unique()):
            variant_group = ability_group[ability_group['ability_variant'] == ability_variant]
            
            # For each variant, calculate the mean MSE at different budget fractions
            variant_group = variant_group.sort_values('budget_fraction')
            
            # Get budget fractions and corresponding mean MSE values
            budget_fractions = sorted(variant_group['budget_fraction'].unique())
            
            for budget_fraction in budget_fractions:
                budget_data = variant_group[variant_group['budget_fraction'] == budget_fraction]
                
                if len(budget_data) > 0:
                    overview_data.append({
                        'base_ability_id': base_ability_id,
                        'ability_variant': ability_variant,
                        'budget_fraction': budget_fraction,
                        'mse': budget_data['mse'].mean(),
                        'bias': budget_data['bias'].mean(),
                        'variance': budget_data['variance'].mean(),
                        'sample_size': len(budget_data)
                    })
    
    # Create overview DataFrame
    if not overview_data:
        return
        
    overview_df = pd.DataFrame(overview_data)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Group by ability ID and variant
    for (base_ability_id, ability_variant), group in overview_df.groupby(['base_ability_id', 'ability_variant']):
        # Use different line styles for different variants
        linestyle = '-' if ability_variant == 'base' else ('--' if ability_variant == 'lower' else '-.')
        
        # Plot MSE vs budget fraction
        plt.plot(group['budget_fraction'], group['mse'], 
                marker='o' if ability_variant == 'base' else ('s' if ability_variant == 'lower' else '^'), 
                linestyle=linestyle,
                label=f"{base_ability_id} ({ability_variant})")
    
    plt.xlabel('Budget Fraction')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f"MSE Trends Across Ability Models and Variants")
    plt.grid(True, alpha=0.3)
    
    # Adjust legend if too many lines
    if len(overview_df.groupby(['base_ability_id', 'ability_variant'])) > 8:
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3)
    else:
        plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"mse_overview.{fmt}")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved MSE overview plot to {output_path}")

def plot_coverage_vs_budget(df: pd.DataFrame, output_dir: str, fmt: str = "png", max_lines: int = 8):
    """
    Create plots of confidence interval coverage vs. budget fraction.
    """
    if 'budget_fraction' not in df.columns or 'contains_true_int' not in df.columns:
        print("Cannot plot coverage vs. budget: missing required columns")
        return

    for estimator, sub in df.groupby('estimator'):
        out_dir = os.path.join(output_dir, estimator)
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        # average coverage at each budget
        cov = sub.groupby('budget_fraction')['contains_true_int'].mean().reset_index()
        plt.plot(cov['budget_fraction'], cov['contains_true_int'], marker='o')
        plt.axhline(0.95, color='r', linestyle='--', label='95% target')
        plt.xlabel('Budget Fraction')
        plt.ylabel('Coverage Rate')
        plt.title(f'CI Coverage vs Budget ({estimator})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"coverage_vs_budget.{fmt}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved coverage vs. budget plot to {path}")

def plot_bias_variance_tradeoff(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Create a bias vs. variance scatter, colored by budget fraction.
    """
    if not all(c in df.columns for c in ('bias','variance','budget_fraction','estimator')):
        print("Cannot plot bias-variance tradeoff: missing required columns")
        return

    for estimator, sub in df.groupby('estimator'):
        out_dir = os.path.join(output_dir, estimator)
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(sub['bias'], sub['variance'],
                         c=sub['budget_fraction'], cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label='Budget Fraction')
        plt.xlabel('Bias')
        plt.ylabel('Variance')
        plt.title(f'Bias-Variance Tradeoff ({estimator})')
        plt.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"bias_variance_tradeoff.{fmt}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved bias-variance tradeoff plot to {path}")

def plot_ci_width_vs_budget(df: pd.DataFrame, output_dir: str, fmt: str = "png", max_lines: int = 8):
    """
    Create plots of confidence‐interval width vs. budget fraction.
    """
    if not all(c in df.columns for c in ('ci_lower','ci_upper','budget_fraction','estimator')):
        print("Cannot plot CI width vs. budget: missing required columns")
        return

    df['ci_width'] = df['ci_upper'] - df['ci_lower']
    for estimator, sub in df.groupby('estimator'):
        out_dir = os.path.join(output_dir, estimator)
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        agg = sub.groupby('budget_fraction')['ci_width'].mean().reset_index()
        plt.plot(agg['budget_fraction'], agg['ci_width'], marker='o', color='m')
        plt.xlabel('Budget Fraction')
        plt.ylabel('CI Width')
        plt.title(f'CI Width vs Budget ({estimator})')
        plt.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"ci_width_vs_budget.{fmt}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved CI width vs. budget plot to {path}")

def plot_mean_vs_true_scatter(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """Scatter estimated mean vs true value with identity line."""
    if not all(c in df.columns for c in ('mean','true_value','estimator')):
        return
    for est, sub in df.groupby('estimator'):
        plt.figure(figsize=(6,6))
        plt.scatter(sub['true_value'], sub['mean'], c=sub['budget_fraction'],
                    cmap='viridis', alpha=0.7)
        mx = np.nanmax([sub['true_value'].max(), sub['mean'].max()])
        mn = np.nanmin([sub['true_value'].min(), sub['mean'].min()])
        plt.plot([mn,mx], [mn,mx], 'k--', linewidth=1)
        plt.colorbar(label='Budget Fraction')
        plt.xlabel('True Value')
        plt.ylabel('Estimated Mean')
        plt.title(f'Mean vs True ({est})')
        plt.grid(True, alpha=0.3)
        path = os.path.join(output_dir, est, f"mean_vs_true.{fmt}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved mean vs true scatter to {path}")

def plot_bias_variance_by_date(df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """Small multiples of bias vs variance: one subplot per date, one line per budget."""
    if not all(c in df.columns for c in ('date','bias','variance','budget_fraction','estimator')):
        return
    for est, sub in df.groupby('estimator'):
        dates = sorted(sub['date'].unique())
        n = len(dates)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), sharex=True, sharey=True)
        axes = axes.flatten()
        for ax, d in zip(axes, dates):
            ddf = sub[sub['date']==d]
            for bf, bf_df in ddf.groupby('budget_fraction'):
                bf_df = bf_df.sort_values('bias')
                ax.plot(bf_df['bias'], bf_df['variance'], marker='o', label=f"bf={bf}")
            ax.set_title(str(d.date()))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='x-small')
        # turn off unused axes
        for unused in axes[len(dates):]:
            unused.set_visible(False)
        fig.suptitle(f'Bias-Variance by Date ({est})')
        plt.tight_layout(rect=[0,0.03,1,0.95])
        out_dir = os.path.join(output_dir, est)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"bias_variance_by_date.{fmt}")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved bias-variance by date grid to {path}")

def create_summary_table(df: pd.DataFrame, output_dir: str):
    """
    Create CSV summary of bias, variance, MSE and coverage by estimator and budget.
    """
    if 'budget_fraction' not in df.columns:
        print("Cannot create summary: missing budget_fraction")
        return

    summary = df.groupby(['estimator','budget_fraction']).agg(
        bias_mean = ('bias','mean'),
        bias_std  = ('bias','std'),
        var_mean  = ('variance','mean'),
        var_std   = ('variance','std'),
        mse_mean  = (lambda x: np.mean(x['bias']**2 + x['variance']), 'mean'),
        coverage  = ('contains_true_int','mean'),
        n_samples = ('bias','count')
    ).reset_index()

    path = os.path.join(output_dir, "bias_analysis_summary.csv")
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(path, index=False)
    print(f"Saved analysis summary to {path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"Loading bias results from {args.input}")
    df = load_and_preprocess_data(args.input)
    
    print(f"Loaded {len(df)} bias result records")
    print(f"Available estimators: {df['estimator'].unique()}")
    
    # Show breakdown of unique scenarios
    print(f"Available ability scenarios: {df['ability_scenario'].unique()}")
    
    if 'budget_fraction' in df.columns:
        print(f"Budget fractions: {sorted(df['budget_fraction'].unique())}")
    
    # Display scenario breakdown
    if 'scenario_key' in df.columns:
        scenario_counts = df.groupby('scenario_key').size()
        print(f"Found {len(scenario_counts)} unique scenarios")
        if args.debug and len(scenario_counts) < 50:
            for scenario, count in scenario_counts.items():
                print(f"  {scenario}: {count} records")
    
    if 'base_scenario_key' in df.columns:
        base_counts = df.groupby('base_scenario_key').size()
        print(f"Found {len(base_counts)} unique base scenarios")
        
        # Count variants per base scenario
        variant_counts = defaultdict(int)
        for base_id, group in df.groupby('base_scenario_key'):
            variants = group['ability_variant'].nunique()
            variant_counts[variants] += 1
            
        print(f"Variant distribution in scenarios:")
        for n_variants, count in sorted(variant_counts.items()):
            print(f"  {count} scenarios with {n_variants} ability variants")
    
    # Apply filters
    filtered_df = apply_filters(df, args)
    print(f"After filtering: {len(filtered_df)} records")
    if args.remove_outliers:
        filtered_df = remove_outliers(filtered_df, args.outlier_quantile)
        print(f"After outlier removal: {len(filtered_df)} records")
    
    if len(filtered_df) == 0:
        print("No records left after filtering, exiting.")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    plot_bias_variance_vs_budget(filtered_df, args.output, args.format, args.max_lines)
    plot_mse_vs_budget(filtered_df, args.output, args.format, args.max_lines)
    plot_bias_variance_tradeoff(filtered_df, args.output, args.format)
    plot_coverage_vs_budget(filtered_df, args.output, args.format, args.max_lines)
    plot_ci_width_vs_budget(filtered_df, args.output, args.format, args.max_lines)
    
    # --- new plots ---
    plot_mean_vs_true_scatter(filtered_df, args.output, args.format)
    plot_bias_variance_by_date(filtered_df, args.output, args.format)
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
