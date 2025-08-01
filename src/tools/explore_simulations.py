"""
Utilities for exploring and visualizing simulation results from HDF5 files.
"""

import argparse
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import json
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
import seaborn as sns

# Global plotting style parameters
FONT_SIZE = 12  # Default font size
TITLE_ENABLED = True  # Whether to show titles in plots

# Use relevant parameters except those in vary_params
# Only use parameters that are meaningful for grouping plots
ESSENTIAL_KEYS = [
    "estimator", "ability_variant", "cost_variant", "ability_scenario", 
    "elicitation_bias_type", "elicitation_bias_enabled", 
    "alternate_ability_type", "alternate_ability_enabled",
    "coverage_ratio", "sampler_type", "threshold"
]

def _style_plots():
    """Apply font size settings and optionally remove titles from plots."""
    try:
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
    except Exception as e:
        print(f"Warning: Error applying plot styles: {e}")
        # Continue anyway - don't let styling issues break the plotting

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore simulation results")
    parser.add_argument("--raw", required=True, help="Path to raw simulation results HDF5 file")
    parser.add_argument("--output", default="reports/simulation_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--config", help="Path to simulation config file (optional)")
    parser.add_argument("--csv", action="store_true", help="Export summary as CSV instead of generating visualizations")
    parser.add_argument("--filter-ability-variant", choices=["base", "lower", "upper"], 
                        help="Filter by ability variant")
    parser.add_argument("--filter-cost-variant", choices=["base", "lower", "upper"], 
                        help="Filter by cost variant")
    parser.add_argument("--filter-estimator", choices=["threshold", "weighted_score"], 
                        help="Filter by estimator type")
    parser.add_argument("--filter-budget", type=float, help="Filter by specific budget fraction")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--font-size", type=int, default=12,
                        help="Base font size for all plot text")
    parser.add_argument("--disable-titles", action="store_true",
                        help="Strip all titles from plots for publication style")
    return parser.parse_args()

def load_simulation_metadata(h5_file: h5py.File) -> Dict[str, Any]:
    """
    Load metadata about the simulation from the HDF5 file.
    
    Args:
        h5_file: Open HDF5 file
        
    Returns:
        Dictionary of metadata
    """
    metadata = {}
    
    if 'metadata' in h5_file:
        meta_group = h5_file['metadata']
        # Get top-level attributes
        for key, value in meta_group.attrs.items():
            metadata[key] = value
        
        # Get estimator-specific attributes if available
        for estimator in ['threshold', 'weighted_score']:
            if estimator in meta_group:
                estimator_group = meta_group[estimator]
                metadata[estimator] = {}
                for key, value in estimator_group.attrs.items():
                    metadata[estimator][key] = value
    
    return metadata

def recursive_list_groups(h5_file_or_group, prefix=''):
    """
    Recursively list all groups in the HDF5 file for debugging.
    
    Args:
        h5_file_or_group: HDF5 file or group to explore
        prefix: Path prefix for recursive calls
        
    Returns:
        List of all group paths
    """
    paths = []
    for key in h5_file_or_group.keys():
        item = h5_file_or_group[key]
        path = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Group):
            paths.append(path)
            # Recursively explore this group
            paths.extend(recursive_list_groups(item, path))
    
    return paths

def list_available_scenarios(h5_file: h5py.File) -> Dict[str, List[str]]:
    """
    List all available scenarios and their budget options in the HDF5 file.
    
    Args:
        h5_file: Open HDF5 file
        
    Returns:
        Dictionary mapping scenario names to lists of budget options
    """
    scenarios = {}
    
    for scenario_name in h5_file.keys():
        if scenario_name == 'metadata':
            continue
        
        scenario_group = h5_file[scenario_name]
        if not isinstance(scenario_group, h5py.Group):
            continue
            
        scenarios[scenario_name] = []
        for budget_name in scenario_group.keys():
            scenarios[scenario_name].append(budget_name)
    
    return scenarios

def extract_metadata_from_path(path: str) -> Dict[str, str]:
    """
    Extract metadata components from a HDF5 path.
    
    Args:
        path: Path within the HDF5 file
        
    Returns:
        Dictionary with extracted metadata
    """
    parts = path.strip('/').split('/')
    
    metadata = {}
    if len(parts) > 0:
        # Extract ability variant if present
        scenario = parts[0]
        if "_base" in scenario:
            metadata["ability_variant"] = "base"
            metadata["ability_scenario"] = scenario.replace("_base", "")
        elif "_lower" in scenario:
            metadata["ability_variant"] = "lower"
            metadata["ability_scenario"] = scenario.replace("_lower", "")
        elif "_upper" in scenario:
            metadata["ability_variant"] = "upper" 
            metadata["ability_scenario"] = scenario.replace("_upper", "")
        else:
            metadata["ability_scenario"] = scenario
            metadata["ability_variant"] = "unknown"
    
    if len(parts) > 1:
        # Extract cost variant if present
        budget = parts[1]
        if "_base" in budget:
            metadata["cost_variant"] = "base"
            metadata["budget"] = budget.replace("_base", "")
        elif "_lower" in budget:
            metadata["cost_variant"] = "lower"
            metadata["budget"] = budget.replace("_lower", "")
        elif "_upper" in budget:
            metadata["cost_variant"] = "upper"
            metadata["budget"] = budget.replace("_upper", "")
        else:
            metadata["budget"] = budget
            metadata["cost_variant"] = "unknown"
    
    if len(parts) > 2:
        metadata["date"] = parts[2]
    
    if len(parts) > 3:
        metadata["estimator"] = parts[3]
    
    return metadata

def get_simulation_paths(
    h5_file: h5py.File, 
    scenario: Optional[str] = None,
    budget: Optional[str] = None, 
    date: Optional[str] = None,
    estimator: Optional[str] = None,
    ability_variant: Optional[str] = None,
    cost_variant: Optional[str] = None,
    bias_type: Optional[str] = None,
    bias_scaling: Optional[str] = None,
    alt_ability: Optional[str] = None,
    coverage_ratio: Optional[float] = None
) -> List[str]:
    """
    Get paths to simulation results matching the specified criteria.
    
    Args:
        h5_file: Open HDF5 file
        scenario: Optional filter by ability scenario
        budget: Optional filter by budget scenario
        date: Optional filter by date
        estimator: Optional filter by estimator type
        ability_variant: Optional filter by ability variant
        cost_variant: Optional filter by cost variant
        
    Returns:
        List of paths to matching simulation results
    """
    # Get all paths recursively for safety
    all_paths = recursive_list_groups(h5_file)
    
    # Filter out metadata path
    all_paths = [p for p in all_paths if not p.startswith('metadata')]
    
    # Filter paths based on criteria
    filtered_paths = all_paths
    
    # Extract metadata from each path for more detailed filtering
    path_metadata = {}
    
    for path in filtered_paths:
        path_metadata[path] = extract_metadata_from_path(path)
    
    # Apply filters based on metadata
    if scenario:
        filtered_paths = [p for p in filtered_paths if 
                          path_metadata[p].get("ability_scenario", "") == scenario]
        
    if budget:
        filtered_paths = [p for p in filtered_paths if 
                          path_metadata[p].get("budget", "") == budget]
        
    if date:
        filtered_paths = [p for p in filtered_paths if 
                          path_metadata[p].get("date", "") == date]
        
    if estimator:
        filtered_paths = [p for p in filtered_paths if 
                          path_metadata[p].get("estimator", "") == estimator]
    
    if ability_variant:
        filtered_paths = [p for p in filtered_paths if 
                          path_metadata[p].get("ability_variant", "") == ability_variant]
        
    if cost_variant:
        filtered_paths = [p for p in filtered_paths if 
                          path_metadata[p].get("cost_variant", "") == cost_variant]
    
    # Only include paths that have a 'results' dataset
    result_paths = []
    for path in filtered_paths:
        try:
            # Check if path leads to groups that might contain 'results'
            current_group = h5_file[path]
            
            # Look for groups that directly contain a 'results' dataset
            has_results = False
            for key in current_group.keys():
                if isinstance(current_group[key], h5py.Group) and 'results' in current_group[key]:
                    has_results = True
                    result_paths.append(f"{path}/{key}")
                    
            # If we didn't find any results directly, check for results in this group
            if not has_results and 'results' in current_group:
                result_paths.append(path)
                
        except KeyError:
            # Skip paths that don't exist
            continue
    
    return result_paths

def load_simulation_results(h5_file: h5py.File, path: str) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Load simulation results and metadata for a specific path.
    
    Args:
        h5_file: Open HDF5 file
        path: Path to the simulation results
        
    Returns:
        Tuple of (results array, simulation metadata, simulation statistics)
    """
    try:
        group = h5_file[path]
        if 'results' not in group:
            return np.array([]), {}, {}
            
        results = group['results'][:]
        
        # Load metadata
        metadata = {}
        for key, value in group.attrs.items():
            metadata[key] = value
        
        # Add path metadata
        path_metadata = extract_metadata_from_path(path)
        metadata.update(path_metadata)
        
        # Load statistics
        stats = {}
        if 'stats' in group:
            stats_group = group['stats']
            for key, value in stats_group.attrs.items():
                stats[key] = value
        
        return results, metadata, stats
    except KeyError as e:
        print(f"Error accessing path '{path}': {e}")
        return np.array([]), {}, {}

def get_weight_function_description(config: Dict[str, Any]) -> str:
    """
    Generate a description of the weight function from configuration.
    
    Args:
        config: Configuration dictionary with weighted_score section
        
    Returns:
        Text description of weight function
    """
    if "weighted_score" not in config:
        return "Default linear weight function (1.0 + 0.5*x)"
    
    weight_config = config["weighted_score"].get("weight_function", {"type": "linear"})
    weight_type = weight_config.get("type", "linear")
    
    if weight_type == "linear":
        base = weight_config.get("base", 1.0)
        slope = weight_config.get("slope", 0.5)
        return f"Linear weight function: {base} + {slope}*x"
    elif weight_type == "exponential":
        base = weight_config.get("base", 1.0)
        scale = weight_config.get("scale", 0.1)
        return f"Exponential weight function: {base}*exp({scale}*x)"
    elif weight_type == "constant":
        value = weight_config.get("value", 1.0)
        return f"Constant weight function: {value}"
    else:
        return f"Unknown weight function type: {weight_type}"

def visualize_simulation_distribution(
    results: np.ndarray, 
    metadata: Dict[str, Any], 
    stats: Dict[str, Any], 
    output_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create visualizations for a simulation's result distribution.
    
    Args:
        results: Simulation results array
        metadata: Simulation metadata
        stats: Simulation statistics
        output_path: Path to save the visualization
        config: Optional configuration dictionary
    """
    plt.figure(figsize=(12, 8))
    
    # Filter out NaN values for the histogram
    valid_results = results[~np.isnan(results)]
    
    # Plot histogram
    sns.histplot(valid_results, kde=True, bins=30)
    
    # Plot vertical lines for key statistics
    if 'true_value' in metadata:
        plt.axvline(x=metadata['true_value'], color='r', linestyle='-', 
                    label=f"True Value: {metadata['true_value']:.3f}")
    
    if 'mean' in stats:
        plt.axvline(x=stats['mean'], color='g', linestyle='--', 
                    label=f"Mean: {stats['mean']:.3f}")
    
    if 'median' in stats:
        plt.axvline(x=stats['median'], color='b', linestyle=':', 
                    label=f"Median: {stats['median']:.3f}")
    
    if 'lower_ci' in stats and 'upper_ci' in stats:
        plt.axvline(x=stats['lower_ci'], color='m', linestyle='-.', 
                    label=f"95% CI: [{stats['lower_ci']:.3f}, {stats['upper_ci']:.3f}]")
        plt.axvline(x=stats['upper_ci'], color='m', linestyle='-.')
        
        # Shade the confidence interval area
        plt.axvspan(stats['lower_ci'], stats['upper_ci'], alpha=0.1, color='m')
    
    # Get estimator from metadata
    estimator = metadata.get("estimator", "unknown")
    
    # Add labels and title
    plt.xlabel(f"{estimator.capitalize()} Estimate")
    plt.ylabel("Density")
    
    # Create title from metadata
    title_parts = []
    
    # Add variant information if available
    ability_variant = metadata.get("ability_variant", "unknown")
    cost_variant = metadata.get("cost_variant", "unknown")
    variant_info = ""
    
    if ability_variant != "unknown" and cost_variant != "unknown":
        variant_info = f" ({ability_variant} ability, {cost_variant} cost)"
    elif ability_variant != "unknown":
        variant_info = f" ({ability_variant} ability)"
    elif cost_variant != "unknown":
        variant_info = f" ({cost_variant} cost)"
        
    scenario_name = metadata.get("ability_scenario", "")
    if scenario_name:
        title_parts.append(f"Scenario: {scenario_name}{variant_info}")
    
    if 'budget' in metadata:
        title_parts.append(f"Budget: {metadata['budget']}")
    
    if 'window_lower' in metadata and 'window_upper' in metadata:
        title_parts.append(f"Window: [{metadata['window_lower']:.1f}, {metadata['window_upper']:.1f}]")
    
    if 'total_samples' in metadata:
        title_parts.append(f"Samples: {metadata['total_samples']}")
    
    if 'budget_fraction' in metadata:
        title_parts.append(f"Budget: {metadata['budget_fraction']*100:.0f}%")
        
    # Add estimator-specific information
    if estimator == "weighted_score" and config is not None:
        title_parts.append(get_weight_function_description(config))
        
    elif estimator == "threshold" and config is not None and "threshold_estimator" in config:
        threshold_config = config["threshold_estimator"]
        engine = threshold_config.get("engine", "scikit-learn")
        title_parts.append(f"Engine: {engine}")
    
    plt.title("\n".join(title_parts))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = "\n".join([
        f"Bias: {stats.get('bias', 'N/A'):.4f}",
        f"Variance: {stats.get('variance', 'N/A'):.4f}",
        f"Skewness: {stats.get('skewness', 'N/A'):.2f}",
        f"Contains True: {stats.get('contains_true', 'N/A')}",
        f"Valid Results: {len(valid_results)}/{len(results)} ({100*len(valid_results)/len(results) if len(results) > 0 else 0:.1f}%)"
    ])
    plt.figtext(0.02, 0.02, stats_text, fontsize=9)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()

def export_summary_csv(h5_file: h5py.File, output_path: str) -> None:
    """
    Export a CSV summary of all simulation results.
    
    Args:
        h5_file: Open HDF5 file
        output_path: Path to save the CSV file
    """
    # Get all paths with results
    result_paths = []
    
    # Get all paths
    all_paths = recursive_list_groups(h5_file)
    all_paths = [p for p in all_paths if not p.startswith('metadata')]
    
    # Find all paths that contain results
    for path in all_paths:
        try:
            group = h5_file[path]
            if 'results' in group:
                result_paths.append(path)
        except Exception:
            continue
    
    records = []
    
    for path in result_paths:
        try:
            group = h5_file[path]
            
            # Skip groups without results
            if 'results' not in group:
                continue
                
            # Extract metadata from path
            path_metadata = extract_metadata_from_path(path)
            
            # Get results array
            results = group['results'][:]
            
            # Initialize record with path metadata
            record = {
                'path': path,
                'n_results': len(results),
                'n_valid': np.sum(~np.isnan(results))
            }
            record.update(path_metadata)
            
            # Add attributes
            for key, value in group.attrs.items():
                record[f"attr_{key}"] = value
                
            # Add statistics if available
            if 'stats' in group:
                stats_group = group['stats']
                for key, value in stats_group.attrs.items():
                    record[f"stat_{key}"] = value
            
            records.append(record)
        except Exception as e:
            print(f"Error processing path '{path}': {e}")
            continue
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path} with {len(df)} records")

def plot_bias_variance_tradeoff(h5_file: h5py.File, output_dir: str) -> None:
    """
    Create a bias vs. variance plot to visualize the trade-off.
    
    Args:
        h5_file: Open HDF5 file
        output_dir: Directory to save plots
    """
    # Collect bias and variance by estimator, scenario, and budget
    results = []
    
    # Get all paths with results
    all_paths = get_simulation_paths(h5_file)
    
    for path in all_paths:
        _, metadata, stats = load_simulation_results(h5_file, path)
        
        if 'bias' in stats and 'variance' in stats:
            results.append({
                'estimator': metadata.get('estimator', 'unknown'),
                'bias': stats['bias'],
                'variance': stats['variance'],
                'ability_scenario': metadata.get('ability_scenario', 'unknown'),
                'budget': metadata.get('budget', 'unknown'),
                'total_samples': metadata.get('total_samples', 0),
                'budget_fraction': metadata.get('budget_fraction', 0),
                'ability_variant': metadata.get('ability_variant', 'unknown'),
                'cost_variant': metadata.get('cost_variant', 'unknown')
            })
    
    if not results:
        print("No bias/variance data available for plotting")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot bias vs. variance by estimator
    plt.figure(figsize=(12, 8))
    
    # Use different markers for different estimators
    for estimator, group in df.groupby('estimator'):
        scatter = plt.scatter(
            group['bias'], group['variance'],
            label=estimator, alpha=0.7,
            c=group['total_samples'], cmap='viridis',
            s=50, marker='o' if estimator == 'threshold' else 's'
        )
    
    plt.colorbar(scatter, label='Sample Count')
    plt.xlabel('Bias')
    plt.ylabel('Variance')
    plt.title('Bias vs. Variance Trade-off by Estimator')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    output_path = os.path.join(output_dir, "bias_variance_tradeoff.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved bias-variance trade-off plot to {output_path}")
    
    # Plot bias vs. variance by budget fraction
    if len(df['budget_fraction'].unique()) > 1:
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            df['bias'], df['variance'],
            c=df['budget_fraction'], cmap='plasma',
            s=50, alpha=0.7
        )
        
        plt.colorbar(scatter, label='Budget Fraction')
        plt.xlabel('Bias')
        plt.ylabel('Variance')
        plt.title('Bias vs. Variance Trade-off by Budget Fraction')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = os.path.join(output_dir, "bias_variance_by_budget.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        _style_plots()  # Apply font and title settings
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved bias-variance by budget plot to {output_path}")

def calculate_exceedance_probability(results: np.ndarray, threshold: float) -> float:
    """Return fraction of non-NaN estimates above the given threshold."""
    valid = results[~np.isnan(results)]
    if len(valid) == 0:
        return 0.0
    return float((valid > threshold).sum() / len(valid))

def get_plot_colors(n_colors: int, plot_type: str = 'categorical') -> List[str]:
    """
    Get colorblind-friendly colors for plots.
    
    Args:
        n_colors: Number of colors needed
        plot_type: Type of plot ('categorical' or 'sequential')
        
    Returns:
        List of color codes/names
    """
    if plot_type == 'categorical':
        if n_colors <= 6:
            colors = sns.color_palette('colorblind', n_colors=n_colors)
        else:
            colors = sns.color_palette('tab10', n_colors=n_colors)
    elif plot_type == 'sequential':
        colors = sns.color_palette('cividis', n_colors=n_colors)
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")
    
    return colors

def get_line_styles(n_styles: int) -> List[str]:
    """
    Get distinct line styles for additional visual distinction.
    
    Args:
        n_styles: Number of line styles needed
        
    Returns:
        List of line style codes
    """
    # Exclude solid line since that's used for baseline
    base_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
    
    # Repeat styles if we need more than available
    return (base_styles * ((n_styles // len(base_styles)) + 1))[:n_styles]

def create_safe_filename(base_name: str, params: Dict[str, Any]) -> str:
    """
    Create a safe filename from a base name and parameter dictionary.
    
    Args:
        base_name: Base filename
        params: Dictionary of parameters to include in filename
        
    Returns:
        Safe filename string
    """
    def hash_long_value(value: str, max_length: int = 12) -> str:
        """Create a short hash for long parameter values."""
        if len(str(value)) <= max_length:
            return str(value)
        # Create a short hash of the value
        hash_obj = hashlib.md5(str(value).encode())
        return hash_obj.hexdigest()[:8]  # Use first 8 characters of MD5 hash
    
    parts = [base_name]
    
    # Add parameters in a consistent order
    for key in sorted(params.keys()):
        value = params[key]
        # Skip None values and 'unknown' values
        if value is None or value == "unknown":
            continue
        
        # Format values appropriately
        if isinstance(value, float):
            value_str = f"{value:.2f}".rstrip('0').rstrip('.')
        else:
            value_str = str(value)
        
        # Hash long values (especially ability_scenario which can be very long)
        if key == "ability_scenario" or len(value_str) > 15:
            value_str = hash_long_value(value_str)
        
        # Add to parts
        parts.append(f"{key}_{value_str}")
    
    # Join and sanitize
    filename = "_".join(parts)
    safe_filename = "".join(c if c.isalnum() or c in "_-." else "_" for c in filename)
    return safe_filename

def create_title_description(params: Dict[str, Any], max_param_length: int = 20) -> str:
    """
    Create a readable description for plot titles.
    
    Args:
        params: Dictionary of parameters
        max_param_length: Maximum length before abbreviating parameter values
        
    Returns:
        Formatted description string
    """
    def abbreviate_long_value(value: str, max_length: int = 20) -> str:
        """Abbreviate long parameter values for titles."""
        if len(str(value)) <= max_length:
            return str(value)
        # For titles, we can be more descriptive than filenames
        return f"{str(value)[:max_length-3]}..."
    
    param_parts = []
    for key, value in sorted(params.items()):
        if value is None or value == "unknown":
            continue
        
        # Format the key nicely
        nice_key = key.replace('_', ' ').title()
        
        # Format the value
        if isinstance(value, float):
            value_str = f"{value:.2f}".rstrip('0').rstrip('.')
        else:
            value_str = str(value)
        
        # Abbreviate long values for readability
        if len(value_str) > max_param_length:
            value_str = abbreviate_long_value(value_str, max_param_length)
        
        param_parts.append(f"{nice_key}: {value_str}")
    
    return ", ".join(param_parts)

def plot_exceedance_probability(
    h5_file: h5py.File,
    output_dir: str,
    risk_thresholds: List[float],
    paths: List[str],
    group_info: Dict[str, Any],
    group_by: str = "true_value"
) -> None:
    """Plot exceedance probabilities for a specific group of simulation paths."""
    plot_data = []
    
    for path in paths:
        results, metadata, _ = load_simulation_results(h5_file, path)
        group_value = metadata.get(group_by, None)
        bf = metadata.get('budget_fraction', None)
        for threshold in risk_thresholds:
            exceedance_prob = calculate_exceedance_probability(results, threshold)
            plot_data.append({
                'group_value': group_value,
                'threshold': threshold,
                'exceedance_probability': exceedance_prob,
                'budget_fraction': bf,
                'total_samples': len(results)
            })
    
    if not plot_data:
        print(f"No data for exceedance analysis in group {group_info.get('name', 'unknown')}")
        return
    
    df = pd.DataFrame(plot_data)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create one plot per threshold for this parameter group
    for threshold, thresh_group in df.groupby('threshold'):
        plt.figure(figsize=(10, 6))
        
        # Get budget fractions and create distinct visual styles
        budget_fractions = sorted(thresh_group['budget_fraction'].dropna().unique())
        colors = get_plot_colors(len(budget_fractions), 'categorical')
        line_styles = get_line_styles(len(budget_fractions))
        
        # For each budget fraction, create a line
        for i, bf in enumerate(budget_fractions):
            sub = thresh_group[thresh_group['budget_fraction'] == bf].sort_values('group_value')
            if sub.empty:
                continue
            plt.plot(
                sub['group_value'], sub['exceedance_probability'],
                marker='o', color=colors[i], linestyle=line_styles[i],
                label=f"Budget={bf:.2f}"
            )
        
        # Add vertical line for threshold
        plt.axvline(x=threshold, color='black', linestyle='-', linewidth=1,
                  label=f"Threshold={threshold}")
        
        plt.xlabel(f"{group_by.replace('_', ' ').title()}")
        plt.ylabel("Exceedance Probability")
        
        # Create title from group info
        if TITLE_ENABLED:
            title_parts = [f"Exceedance Probability vs {group_by.replace('_', ' ').title()}"]
            # Use the new title description function
            title_description = create_title_description(group_info.get('fixed_params', {}))
            if title_description:
                title_parts.append(title_description)
            title_parts.append(f"Threshold: {threshold}")
            plt.title("\n".join(title_parts))
        
        # Improve legend with better placement
        if len(budget_fractions) > 5:
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add reference line at 50% probability
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # Fix y-axis range for consistency
        plt.ylim(-0.05, 1.05)
        
        # Apply styling before creating filename to ensure all plot elements are set
        _style_plots()  # Apply font and title settings
        
        # Create filename from essential group parameters and threshold
        # Only use the most important parameters to avoid long filenames
        essential_params = {}
        fixed_params = group_info.get('fixed_params', {})
        
        # Include only the most essential parameters for filename
        essential_keys = ESSENTIAL_KEYS
        for key in essential_keys:
            if key == 'threshold':
                essential_params[key] = threshold
            elif key in fixed_params and fixed_params[key] != "unknown":
                essential_params[key] = fixed_params[key]
        
        safe_filename = create_safe_filename('exceedance', essential_params)
        output_path = os.path.join(output_dir, f"{safe_filename}.png")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved exceedance probability plot to {output_path}")

def calculate_detection_lag_via_interpolation(
    df: pd.DataFrame, 
    probability_levels: List[float] = [0.5, 0.9]
) -> pd.DataFrame:
    """
    Calculate detection lag metrics by interpolating the exceedance probability curve.
    
    Args:
        df: DataFrame with 'true_value', 'threshold', and 'exceedance_probability' columns
        probability_levels: Target probability levels for detection
        
    Returns:
        DataFrame with detection lag metrics
    """
    results = []
    
    # Process each threshold separately (no longer grouping by estimator)
    for threshold, group in df.groupby('threshold'):
        # Skip if insufficient data
        if len(group) < 3:
            continue
            
        # Sort by true value for interpolation
        sorted_group = group.sort_values('true_value')
        
        true_values = sorted_group['true_value'].values
        exceedance_probs = sorted_group['exceedance_probability'].values
        
        # Calculate metrics
        max_detection_prob = exceedance_probs.max()
        no_detection_prob = 1.0 - max_detection_prob
        
        # Find threshold index (where true_value = threshold)
        threshold_idx = np.argmin(np.abs(true_values - threshold))
        # Get false/early detection rate
        early_detection_rate = exceedance_probs[threshold_idx]
        
        # Calculate conditional detection metrics
        if max_detection_prob > 0:
            # Normalize exceedance probabilities to create CDF conditional on detection
            norm_probs = exceedance_probs / max_detection_prob
            
            # For each target probability level (conditional)
            for prob in probability_levels:
                try:
                    # Only proceed if we have points on both sides of target probability
                    if min(norm_probs) <= prob <= max(norm_probs):
                        # Since we have a CDF, the function is invertible and so
                        # it makes sense to think about plotting the true value at each probability.
                        # Interpolate to find true value at target conditional probability
                        conditional_value = np.interp(
                            prob,  # target probability 
                            norm_probs,  # x-values (normalized detection probabilities)
                            true_values  # y-values (true values)
                        )
                        
                        # Calculate lag as difference between this value and threshold
                        detection_lag = conditional_value - threshold
                        
                        results.append({
                            'threshold': threshold,
                            'detection_probability': prob,
                            'true_value_at_detection': conditional_value,
                            'detection_lag': detection_lag,
                            'conditional': True,
                            'no_detection_probability': no_detection_prob,
                            'early_detection_rate': early_detection_rate,
                            'max_detection_probability': max_detection_prob
                        })
                except Exception as e:
                    print(f"Error calculating conditional lag for threshold {threshold}: {e}")
        
        # Also calculate unconditional detection values
        for prob in probability_levels:
            try:
                # Only proceed if we have points on both sides of the target probability
                if min(exceedance_probs) <= prob <= max(exceedance_probs):
                    # Interpolate to find true value at target probability (unconditional)
                    unconditional_value = np.interp(
                        prob,  # target probability 
                        exceedance_probs,  # x-values (exceedance probabilities)
                        true_values  # y-values (true values)
                    )
                    
                    # Calculate lag as difference between this value and threshold
                    detection_lag = unconditional_value - threshold
                    
                    results.append({
                        'threshold': threshold,
                        'detection_probability': prob,
                        'true_value_at_detection': unconditional_value,
                        'detection_lag': detection_lag,
                        'conditional': False,
                        'no_detection_probability': no_detection_prob,
                        'early_detection_rate': early_detection_rate,
                        'max_detection_probability': max_detection_prob
                    })
            except Exception as e:
                print(f"Error calculating unconditional lag for threshold {threshold}: {e}")
    
    return pd.DataFrame(results)

def plot_enhanced_detection_metrics(
    h5_file: h5py.File, 
    output_dir: str, 
    thresholds: List[float],
    paths: List[str],
    group_info: Dict[str, Any],
    probability_levels: List[float] = [0.5, 0.9]
) -> None:
    """
    Create enhanced detection metric plots for a specific group of simulation paths.
    
    Args:
        h5_file: Open HDF5 file
        output_dir: Directory to save plots
        thresholds: List of thresholds for this group
        paths: List of simulation paths for this group
        group_info: Information about the parameter group
        probability_levels: Target probability levels for detection metrics
    """
    # Gather exceedance probability data for this group
    exceedance_data = []
    
    for path in paths:
        results, metadata, _ = load_simulation_results(h5_file, path)
        
        # Skip empty results
        if len(results) == 0:
            continue
        
        true_value = metadata.get('true_value', np.nan)
        budget_fraction = metadata.get('budget_fraction', np.nan)
        
        # Skip if key information is missing
        if np.isnan(true_value) or np.isnan(budget_fraction):
            continue
            
        # Calculate exceedance probability for each threshold
        for threshold in thresholds:
            exceedance_prob = calculate_exceedance_probability(results, threshold)
            
            exceedance_data.append({
                'threshold': threshold,
                'true_value': true_value,
                'budget_fraction': budget_fraction,
                'exceedance_probability': exceedance_prob,
            })
    
    if not exceedance_data:
        print(f"No data available for detection metrics analysis in group {group_info.get('name', 'unknown')}")
        return
    
    # Convert to DataFrame
    exceedance_df = pd.DataFrame(exceedance_data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate detection metrics for each budget fraction separately
    all_metrics = []
    
    for budget_fraction, budget_group in exceedance_df.groupby('budget_fraction'):
        # Calculate detection lag metrics
        metrics = calculate_detection_lag_via_interpolation(budget_group, probability_levels)
        
        if not metrics.empty:
            metrics['budget_fraction'] = budget_fraction
            all_metrics.append(metrics)
    
    if not all_metrics:
        print(f"Couldn't compute any detection metrics for group {group_info.get('name', 'unknown')}")
        return
        
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Get estimator name from group info for filename
    estimator = group_info.get('fixed_params', {}).get('estimator', 'unknown')
    
    # 1. Plot conditional median detection lag vs budget
    plt.figure(figsize=(10, 6))
    
    for threshold, thresh_group in metrics_df[metrics_df['conditional'] == True].groupby('threshold'):
        median_group = thresh_group[thresh_group['detection_probability'] == 0.5]
        
        if not median_group.empty:
            plt.plot(
                median_group['budget_fraction'],
                median_group['detection_lag'],
                marker='o',
                label=f"Threshold={threshold}"
            )
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Budget Fraction')
    plt.ylabel('Conditional Median Detection Lag (capability units)')
    
    if TITLE_ENABLED:
        title_parts = ['Conditional Median Detection Lag vs Budget Fraction']
        # Use the new title description function
        title_description = create_title_description(group_info.get('fixed_params', {}))
        if title_description:
            title_parts.append(title_description)
        title_parts.append('True Value at 50% of Successful Detections')
        plt.title('\n'.join(title_parts))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create essential filename from key parameters only
    essential_params = {}
    fixed_params = group_info.get('fixed_params', {})
    
    # Include only the most essential parameters for filename
    essential_keys = ESSENTIAL_KEYS
    for key in essential_keys:
        if key in fixed_params and fixed_params[key] != "unknown":
            essential_params[key] = fixed_params[key]
    
    essential_params["metric"] = "conditional_median_lag"
    filename = create_safe_filename('detection', essential_params)
    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.tight_layout()
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved conditional median detection lag plot to {output_path}")
    
    # 2. Plot probability of no detection vs budget
    plt.figure(figsize=(10, 6))
    
    for threshold, thresh_group in metrics_df.groupby('threshold'):
        # Take the first occurrence for each budget fraction (they should all be the same)
        no_detect_group = thresh_group.drop_duplicates(subset=['budget_fraction'])
        
        plt.plot(
            no_detect_group['budget_fraction'],
            no_detect_group['no_detection_probability'] * 100,  # Convert to percentage
            marker='o',
            label=f"Threshold={threshold}"
        )
    
    plt.xlabel('Budget Fraction')
    plt.ylabel('Probability of No Detection (%)')
    
    if TITLE_ENABLED:
        title_parts = ['Probability of No Detection vs Budget Fraction']
        # Use the new title description function
        title_description = create_title_description(group_info.get('fixed_params', {}))
        if title_description:
            title_parts.append(title_description)
        plt.title('\n'.join(title_parts))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create essential filename from key parameters only  
    essential_params = {}
    fixed_params = group_info.get('fixed_params', {})
    
    # Include only the most essential parameters for filename
    essential_keys = ESSENTIAL_KEYS
    for key in essential_keys:
        if key in fixed_params and fixed_params[key] != "unknown":
            essential_params[key] = fixed_params[key]
    
    essential_params["metric"] = "no_detection_prob"
    filename = create_safe_filename('detection', essential_params)
    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.tight_layout()
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved no detection probability plot to {output_path}")
    
    # 3. Plot early detection rate vs budget
    plt.figure(figsize=(10, 6))
    
    for threshold, thresh_group in metrics_df.groupby('threshold'):
        # Take the first occurrence for each budget fraction (they should all be the same)
        early_detect_group = thresh_group.drop_duplicates(subset=['budget_fraction'])
        
        plt.plot(
            early_detect_group['budget_fraction'],
            early_detect_group['early_detection_rate'] * 100,  # Convert to percentage
            marker='o',
            label=f"Threshold={threshold}"
        )
    
    plt.xlabel('Budget Fraction')
    plt.ylabel('Early Detection Rate (%)')
    
    if TITLE_ENABLED:
        title_parts = ['Detection Rate at Threshold vs Budget Fraction']
        # Use the new title description function
        title_description = create_title_description(group_info.get('fixed_params', {}))
        if title_description:
            title_parts.append(title_description)
        plt.title('\n'.join(title_parts))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create essential filename from key parameters only
    essential_params = {}
    fixed_params = group_info.get('fixed_params', {})
    
    # Include only the most essential parameters for filename
    essential_keys = ESSENTIAL_KEYS
    for key in essential_keys:
        if key in fixed_params and fixed_params[key] != "unknown":
            essential_params[key] = fixed_params[key]
    
    essential_params["metric"] = "early_detection_rate"
    filename = create_safe_filename('detection', essential_params)
    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.tight_layout()
    _style_plots()  # Apply font and title settings
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved early detection rate plot to {output_path}")

def group_simulation_paths_by_fixed_params(
    h5_file: h5py.File, 
    fixed_params: Optional[List[str]] = None,
    vary_params: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Group simulation paths by fixed parameters, allowing specified parameters to vary within groups.
    
    Args:
        h5_file: Open HDF5 file
        fixed_params: Parameters that must be the same within each group. If None and vary_params 
                     is provided, will use all parameters except those in vary_params.
        vary_params: Parameters that are allowed to vary within groups. Used to derive fixed_params
                    if fixed_params is None.
        
    Returns:
        Dictionary mapping group names to group info including paths and parameters
    """
    all_paths = get_simulation_paths(h5_file)
    if not all_paths:
        return {}
    
    # Collect all available parameters from metadata
    all_params = set()
    path_metadata = {}
    
    for path in all_paths:
        _, metadata, _ = load_simulation_results(h5_file, path)
        if not metadata:
            continue
        path_metadata[path] = metadata
        all_params.update(metadata.keys())
    
    # Determine which parameters to group by
    if fixed_params is None and vary_params is not None:
        # Use all parameters except those in vary_params
        fixed_params = [p for p in all_params if p not in vary_params]
    elif fixed_params is None:
        fixed_params = []
    print(f"Grouping by fixed parameters: {fixed_params}")
    
    # Group paths by fixed parameters
    groups = {}
    
    for path in all_paths:
        metadata = path_metadata.get(path, {})
        
        # Create group signature from fixed parameters
        group_signature = {}
        for param in fixed_params:
            value = metadata.get(param, "unknown")
            group_signature[param] = value
        
        # Convert to hashable key
        group_key = tuple(sorted(group_signature.items()))
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(path)
    
    # Convert to readable format with descriptive names
    readable_groups = {}
    for group_key, paths in groups.items():
        group_dict = dict(group_key)
        
        # Create descriptive group name, filtering out 'unknown' values
        name_parts = []
        for key, value in sorted(group_dict.items()):
            if value != "unknown":
                name_parts.append(f"{key}_{value}")
        
        group_name = "_".join(name_parts) if name_parts else "default_group"
        
        readable_groups[group_name] = {
            "paths": paths,
            "fixed_params": group_dict,
            "name": group_name.replace('_', ' ').title(),
            "description": f"Parameter group: {', '.join([f'{k}={v}' for k, v in group_dict.items() if v != 'unknown'])}"
        }
    
    return readable_groups

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set global plotting parameters
    global FONT_SIZE, TITLE_ENABLED
    FONT_SIZE = args.font_size
    TITLE_ENABLED = not args.disable_titles
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    # Open the HDF5 file
    with h5py.File(args.raw, 'r') as h5_file:
        # Load metadata
        metadata = load_simulation_metadata(h5_file)
        print("Simulation Configuration:")
        # Print metadata in a structured way
        if args.debug:
            print("  Metadata:")
            for key, value in metadata.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        # List all groups for debugging if requested
        if args.debug:
            print("\nAll HDF5 Groups:")
            all_groups = recursive_list_groups(h5_file)
            for group in sorted(all_groups):
                print(f"  {group}")
        
        # List available scenarios
        scenarios = list_available_scenarios(h5_file)
        if args.debug:
            print("\nAvailable Scenarios:")
            for scenario, budgets in scenarios.items():
                if scenario == 'metadata':
                    continue
                print(f"  {scenario}:")
                for budget in budgets:
                    print(f"    {budget}")
            
        # If CSV export is requested
        if args.csv:
            csv_path = args.output if args.output.endswith('.csv') else os.path.join(args.output, "simulation_summary.csv")
            export_summary_csv(h5_file, csv_path)
            return
        
        # Get simulation paths, applying filters if provided
        paths = get_simulation_paths(
            h5_file,
            ability_variant=args.filter_ability_variant,
            cost_variant=args.filter_cost_variant,
            estimator=args.filter_estimator
        )
        print(f"\nFound {len(paths)} simulation results")
        
        if args.debug:
            print("\nPaths found:")
            for path in paths:
                print(f"  {path}")
        
        if not paths:
            print("No simulation results found matching the filters.")
            return
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Generate visualizations for each simulation
        for path in paths:
            results, metadata, stats = load_simulation_results(h5_file, path)
            
            # Skip if we couldn't load results
            if len(results) == 0:
                print(f"Skipping path '{path}' - no results found")
                continue
            
            # Create a descriptive output path
            path_metadata = extract_metadata_from_path(path)
            estimator = path_metadata.get("estimator", "unknown")
            ability_scenario = path_metadata.get("ability_scenario", "unknown")
            budget = path_metadata.get("budget", "unknown")
            ability_variant = path_metadata.get("ability_variant", "unknown")
            cost_variant = path_metadata.get("cost_variant", "unknown")
            bias_type = path_metadata.get("bias_type", "unknown")
            bias_scaling = path_metadata.get("bias_scaling", "unknown")
            alt_ability = path_metadata.get("alt_ability", "unknown")
            coverage_ratio = path_metadata.get("coverage_ratio", 0.8)
            
            # Create descriptive filename
            filename = f"{estimator}_{ability_scenario}"
            if ability_variant != "unknown":
                filename += f"_{ability_variant}"
            filename += f"_{budget}"
            if cost_variant != "unknown":
                filename += f"_{cost_variant}"
            if bias_type != "unknown":
                filename += f"_{bias_type}"
            if bias_scaling != "unknown":
                filename += f"_{bias_scaling}"
            if alt_ability != "unknown":
                filename += f"_{alt_ability}"
            if coverage_ratio != 0.8:
                filename += f"_coverage{coverage_ratio:.1f}"
            
            # Clean up filename to remove special characters
            safe_filename = "".join(c if c.isalnum() or c in "_-." else "_" for c in filename)
            output_path = os.path.join(args.output, f"{safe_filename}.png")
            
            if args.debug:
                visualize_simulation_distribution(results, metadata, stats, output_path, config)
                print(f"Created visualization: {output_path}")
        
        # Uncomment to generate additional plots
        # plot_bias_variance_tradeoff(h5_file, args.output)
        
        # Create an exceedance plots directory
        exceedance_dir = os.path.join(args.output, "exceedance_plots")
        os.makedirs(exceedance_dir, exist_ok=True)
        
        # Get risk thresholds from config
        rt_cfg = config.get("risk_thresholds", {}) if config else {}
        
        # Group simulation paths for exceedance plots (vary budget_fraction within groups)
        VARY_PARAMS = ["budget_fraction", "true_value"]
        VARY_PARAMS.extend([# The following parameters are computed per time step so should
                            # vary within groups
                            "slope",
                            "threshold",
                            "date",
                            "budget",
                            "total_samples",
                            "window_lower",
                            "window_upper",
                            ])
        if rt_cfg:
            path_groups = group_simulation_paths_by_fixed_params(
                h5_file, 
                vary_params=VARY_PARAMS
            )
            print(f"Grouped {len(path_groups)} parameter groups for exceedance analysis")
            print(f"Parameter groups: {list(path_groups.keys())}")
            
            for group_name, group_info in path_groups.items():
                # Get estimator for this group to find appropriate thresholds
                estimator = group_info.get('fixed_params', {}).get('estimator', 'unknown')
                thresholds = rt_cfg.get(estimator, [])
                
                if thresholds:
                    plot_exceedance_probability(
                        h5_file, 
                        exceedance_dir, 
                        thresholds,
                        group_info['paths'],
                        group_info
                    )
        
        # Create detection metrics plots
        detection_dir = os.path.join(args.output, "detection_metrics")
        
        if rt_cfg:
            # Group simulation paths for detection metrics (same grouping as exceedance)
            for group_name, group_info in path_groups.items():
                # Get estimator for this group to find appropriate thresholds
                estimator = group_info.get('fixed_params', {}).get('estimator', 'unknown')
                thresholds = rt_cfg.get(estimator, [])
                
                if thresholds:
                    plot_enhanced_detection_metrics(
                        h5_file, 
                        detection_dir, 
                        thresholds,
                        group_info['paths'],
                        group_info
                    )
        
        print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
