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

def list_available_scenarios(h5_file: h5py.File) -> Dict[str, List[str]]:
    """
    List all available scenarios and their budget options in the HDF5 file.
    
    Args:
        h5_file: Open HDF5 file
        
    Returns:
        Dictionary mapping scenario names to lists of budget options
    """
    scenarios = {}
    
    for scenario_name, scenario_group in h5_file.items():
        if scenario_name == 'metadata':
            continue
        
        scenarios[scenario_name] = []
        if isinstance(scenario_group, h5py.Group):
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
    
    # Filter by scenario
    if scenario:
        if scenario in h5_file:
            scenario_groups = [h5_file[scenario]]
        else:
            return []
    else:
        scenario_groups = [h5_file[name] for name in h5_file.keys() if name != 'metadata']
    
    # Filter by budget
    for scenario_group in scenario_groups:
        if budget:
            if budget in scenario_group:
                budget_groups = [scenario_group[budget]]
            else:
                continue
        else:
            budget_groups = [scenario_group[name] for name in scenario_group.keys()]
        
        # Filter by date
        for budget_group in budget_groups:
            if date:
                if date in budget_group:
                    date_groups = [budget_group[date]]
                else:
                    continue
            else:
                date_groups = [budget_group[name] for name in budget_group.keys()]
            
            # Filter by estimator
            for date_group in date_groups:
                if estimator:
                    if estimator in date_group:
                        estimator_groups = [date_group[estimator]]
                    else:
                        continue
                else:
                    estimator_groups = [date_group[name] for name in date_group.keys()]
                
                # Get all simulation groups
                for estimator_group in estimator_groups:
                    for sim_name in estimator_group.keys():
                        path = f"{scenario_group.name}/{budget_group.name}/{date_group.name}/{estimator_group.name}/{sim_name}"
                        paths.append(path)
    
    return paths

def load_simulation_results(h5_file: h5py.File, path: str) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Load simulation results and metadata for a specific path.
    
    Args:
        h5_file: Open HDF5 file
        path: Path to the simulation results
        
    Returns:
        Tuple of (results array, simulation metadata, simulation statistics)
    """
    group = h5_file[path]
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
        
        # List available scenarios
        scenarios = list_available_scenarios(h5_file)
        print("\nAvailable Scenarios:")
        for scenario, budgets in scenarios.items():
            if scenario == 'metadata':
                continue
            print(f"  {scenario}:")
            for budget in budgets:
                print(f"    {budget}")
        
        # Get all simulation paths
        paths = get_simulation_paths(h5_file)
        print(f"\nFound {len(paths)} simulation results")
        
        # Generate visualizations for each simulation
        for path in paths:
            results, metadata, stats = load_simulation_results(h5_file, path)
            
            # Create a descriptive output path
            parts = path.strip('/').split('/')
            output_dir = os.path.join(args.output, *parts[:-1])
            output_file = os.path.join(output_dir, f"{parts[-1]}.png")
            
            visualize_simulation_distribution(results, metadata, stats, output_file)
        
        print(f"Visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
