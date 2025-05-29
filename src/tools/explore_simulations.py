"""
Utilities for exploring and visualizing simulation results from HDF5 files.
"""

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore simulation results")
    parser.add_argument("--raw", required=True, help="Path to raw simulation results HDF5 file")
    parser.add_argument("--output", default="reports/simulation_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--csv", action="store_true", help="Export summary as CSV instead of generating visualizations")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
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
        for key, value in meta_group.attrs.items():
            metadata[key] = value
    
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

def get_simulation_paths(h5_file: h5py.File, scenario: Optional[str] = None,
                        budget: Optional[str] = None, date: Optional[str] = None,
                        estimator: Optional[str] = None) -> List[str]:
    """
    Get paths to simulation results matching the specified criteria.
    
    Args:
        h5_file: Open HDF5 file
        scenario: Optional filter by ability scenario
        budget: Optional filter by budget scenario
        date: Optional filter by date
        estimator: Optional filter by estimator type
        
    Returns:
        List of paths to matching simulation results
    """
    paths = []
    
    # Get all paths recursively for safety
    all_paths = recursive_list_groups(h5_file)
    
    # Filter out metadata path
    all_paths = [p for p in all_paths if not p.startswith('metadata')]
    
    # Filter paths based on criteria
    filtered_paths = all_paths
    
    if scenario:
        filtered_paths = [p for p in filtered_paths if f"/{scenario}/" in f"/{p}/"]
        
    if budget:
        filtered_paths = [p for p in filtered_paths if f"/{budget}/" in f"/{p}/"]
        
    if date:
        filtered_paths = [p for p in filtered_paths if f"/{date}/" in f"/{p}/"]
        
    if estimator:
        filtered_paths = [p for p in filtered_paths if f"/{estimator}/" in f"/{p}/"]
    
    # Only include paths that have a 'results' dataset
    result_paths = []
    for path in filtered_paths:
        try:
            if 'results' in h5_file[path]:
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

def visualize_simulation_distribution(results: np.ndarray, metadata: Dict[str, Any], 
                                     stats: Dict[str, Any], output_path: str) -> None:
    """
    Create visualizations for a simulation's result distribution.
    
    Args:
        results: Simulation results array
        metadata: Simulation metadata
        stats: Simulation statistics
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Filter out NaN values for the histogram
    valid_results = results[~np.isnan(results)]
    
    # Plot histogram
    plt.hist(valid_results, bins=30, alpha=0.7, density=True)
    
    # Plot vertical lines for key statistics
    if 'true_value' in metadata:
        plt.axvline(x=metadata['true_value'], color='r', linestyle='-', label=f"True Value: {metadata['true_value']:.3f}")
    
    if 'mean' in stats:
        plt.axvline(x=stats['mean'], color='g', linestyle='--', label=f"Mean: {stats['mean']:.3f}")
    
    if 'median' in stats:
        plt.axvline(x=stats['median'], color='b', linestyle=':', label=f"Median: {stats['median']:.3f}")
    
    if 'lower_ci' in stats and 'upper_ci' in stats:
        plt.axvline(x=stats['lower_ci'], color='m', linestyle='-.', label=f"95% CI: [{stats['lower_ci']:.3f}, {stats['upper_ci']:.3f}]")
        plt.axvline(x=stats['upper_ci'], color='m', linestyle='-.')
    
    # Add labels and title
    estimator = os.path.basename(os.path.dirname(os.path.dirname(output_path)))
    plt.xlabel(f"{estimator} Estimate")
    plt.ylabel("Density")
    
    # Create title from metadata
    title_parts = []
    if 'window_lower' in metadata and 'window_upper' in metadata:
        title_parts.append(f"Window: [{metadata['window_lower']:.1f}, {metadata['window_upper']:.1f}]")
    if 'total_samples' in metadata:
        title_parts.append(f"Samples: {metadata['total_samples']}")
    if 'budget_fraction' in metadata:
        title_parts.append(f"Budget: {metadata['budget_fraction']*100:.0f}%")
    
    plt.title(", ".join(title_parts))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    textstr = "\n".join([
        f"Bias: {stats.get('bias', 'N/A'):.4f}",
        f"Variance: {stats.get('variance', 'N/A'):.4f}",
        f"Skewness: {stats.get('skewness', 'N/A'):.2f}",
        f"Valid Results: {len(valid_results)}/{len(results)} ({100*len(valid_results)/len(results):.1f}%)"
    ])
    plt.figtext(0.02, 0.02, textstr, fontsize=9)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def export_summary_csv(h5_file: h5py.File, output_path: str) -> None:
    """
    Export a CSV summary of all simulation results.
    
    Args:
        h5_file: Open HDF5 file
        output_path: Path to save the CSV file
    """
    # Get all paths with results
    paths = recursive_list_groups(h5_file)
    paths = [p for p in paths if 'metadata' not in p]
    
    records = []
    
    for path in paths:
        try:
            group = h5_file[path]
            
            # Skip groups without results
            if 'results' not in group:
                continue
                
            # Extract path components
            path_parts = path.split('/')
            scenario = path_parts[0] if len(path_parts) > 0 else ""
            budget = path_parts[1] if len(path_parts) > 1 else ""
            date = path_parts[2] if len(path_parts) > 2 else ""
            estimator = path_parts[3] if len(path_parts) > 3 else ""
            
            # Get results array
            results = group['results'][:]
            
            # Get metadata from attributes
            record = {
                'scenario': scenario,
                'budget': budget,
                'date': date,
                'estimator': estimator,
                'path': path,
                'n_results': len(results),
                'n_valid': np.sum(~np.isnan(results))
            }
            
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
    print(f"Saved summary to {output_path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Open the HDF5 file
    with h5py.File(args.raw, 'r') as h5_file:
        # Load metadata
        metadata = load_simulation_metadata(h5_file)
        print("Simulation Configuration:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # List all groups for debugging if requested
        if args.debug:
            print("\nAll HDF5 Groups:")
            all_groups = recursive_list_groups(h5_file)
            for group in sorted(all_groups):
                print(f"  {group}")
        
        # List available scenarios
        scenarios = list_available_scenarios(h5_file)
        print("\nAvailable Scenarios:")
        for scenario, budgets in scenarios.items():
            if scenario == 'metadata':
                continue
            print(f"  {scenario}:")
            for budget in budgets:
                print(f"    {budget}")
        
        # If CSV export is requested
        if args.csv:
            export_summary_csv(h5_file, args.output)
            return
        
        # Get all simulation paths
        paths = get_simulation_paths(h5_file)
        print(f"\nFound {len(paths)} simulation results")
        
        if args.debug:
            print("\nPaths found:")
            for path in paths:
                print(f"  {path}")
        
        # Generate visualizations for each simulation
        for path in paths:
            results, metadata, stats = load_simulation_results(h5_file, path)
            
            # Skip if we couldn't load results
            if len(results) == 0:
                print(f"Skipping path '{path}' - no results found")
                continue
            
            # Create a descriptive output path
            parts = path.strip('/').split('/')
            output_dir = os.path.join(args.output, *parts[:-1])
            output_file = os.path.join(output_dir, f"{parts[-1]}.png")
            
            visualize_simulation_distribution(results, metadata, stats, output_file)
        
        print(f"Visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
