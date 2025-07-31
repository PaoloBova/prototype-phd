"""
Simplified Elicitation Bias Debug Visualization Script

This script creates focused matplotlib plots for debugging elicitation bias behavior.
Uses config group filtering to organize plots by related parameter variations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import required modules from the project
from prototype_phd.utils import expand_sweep_config
from src.forecast_evaluation import define_elicitation_bias
from src.schemas import (
    EvaluationConfig, EvaluationScenario, AbilityForecast, 
    CalculatedElicitationBias
)


def get_nested_value(config: Dict[str, Any], path: List[str]) -> Any:
    """
    Get a nested value from a configuration dictionary using a path.
    
    Args:
        config: Configuration dictionary
        path: List of keys representing the path to the value
        
    Returns:
        The value at the specified path, or None if not found
    """
    current = config
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return None


def filter_configs(configs: List[Dict[str, Any]], filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter configurations based on filter specifications.
    
    Args:
        configs: List of configuration dictionaries
        filters: List of filter specifications, each with 'path' and 'value' keys
        
    Returns:
        List of configurations that match all filters
    """
    filtered_configs = []
    
    for config in configs:
        matches_all_filters = True
        
        for filter_spec in filters:
            path = filter_spec["path"]
            expected_value = filter_spec["value"]
            actual_value = get_nested_value(config, path)
            
            if actual_value != expected_value:
                matches_all_filters = False
                break
        
        if matches_all_filters:
            filtered_configs.append(config)
    
    return filtered_configs


def create_simple_sweep_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create simple sweep configurations for debugging.
    
    Returns:
        Dictionary of sweep configuration functions
    """
    
    # Logistic ability shift sweep - vary delta parameter
    logistic_ability_shift_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.75, 0.5, 0.25, 0.1]
            },
            "scenario_generation": {
                "include_base_scenarios": True,
                "include_ci_scenarios": False
            },
            "sampler_type": "uniform",
            "adjustment_method": "upper_bound", 
            "repeats_per_unit": 20,
            "sampler_params": {},
            "elicitation_bias_config": {
                "enabled": True,
                "bias_type": "logistic_ability_shift",
                "name": "logistic_ability_shift",
                "source_file": None,
                "parameters": {"delta": 2.0, "elicitation_threshold": 0.0, "sensitivity_rate_after": 0.5},
                "budget_dependent": False,  # Disable budget scaling for cleaner debugging
                "budget_scaling": {}
            },
            "alternate_ability": {
                "enabled": False,
                "function_type": "logistic",
                "args": {}
            }
        },
        "_sweep": [
            {
                "path": ["elicitation_bias_config", "parameters", "delta"],
                "values": [0.5, 1.0, 2.0, 3.0, 4.0]
            }
        ]
    }
    
    # Task filter sweep - vary elicitation_threshold parameter
    task_filter_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.75, 0.5, 0.25, 0.1]
            },
            "scenario_generation": {
                "include_base_scenarios": True,
                "include_ci_scenarios": False
            },
            "sampler_type": "uniform",
            "adjustment_method": "upper_bound", 
            "repeats_per_unit": 20,
            "sampler_params": {},
            "elicitation_bias_config": {
                "enabled": True,
                "bias_type": "task_filter",
                "name": "task_filter",
                "source_file": None,
                "parameters": {"elicitation_threshold": 15.0, "sensitivity_rate_after": 0.5, "delta": 1.0},
                "budget_dependent": False,  # Disable budget scaling for cleaner debugging
                "budget_scaling": {}
            },
            "alternate_ability": {
                "enabled": False,
                "function_type": "logistic",
                "args": {}
            }
        },
        "_sweep": [
            {
                "path": ["elicitation_bias_config", "parameters", "elicitation_threshold"],
                "values": [10.0, 15.0, 20.0, 25.0, 30.0]
            }
        ]
    }
    
    # Budget scaling sweep - test budget dependent behavior
    budget_scaling_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
            },
            "scenario_generation": {
                "include_base_scenarios": True,
                "include_ci_scenarios": False
            },
            "sampler_type": "uniform",
            "adjustment_method": "upper_bound", 
            "repeats_per_unit": 20,
            "sampler_params": {},
            "elicitation_bias_config": {
                "enabled": True,
                "bias_type": "logistic_ability_shift",
                "name": "budget_scaling_test",
                "source_file": None,
                "parameters": {"delta": 3.0, "elicitation_threshold": 0.0, "sensitivity_rate_after": 0.5},
                "budget_dependent": True,
                "budget_scaling": {
                    "delta": {
                        "type": "linear",
                        "params": {"target_value": 0.0}
                    }
                }
            },
            "alternate_ability": {
                "enabled": False,
                "function_type": "logistic",
                "args": {}
            }
        },
        "_sweep": [
            {
                "path": ["elicitation_bias_config", "bias_type"],
                "values": ["logistic_ability_shift", "task_filter"]
            }
        ]
    }
    
    return {
        "logistic_ability_shift": logistic_ability_shift_sweep,
        "task_filter": task_filter_sweep,
        "budget_scaling": budget_scaling_sweep
    }


def define_config_groups() -> Dict[str, Dict[str, Any]]:
    """
    Define config groups for filtering and organizing plots.
    
    Returns:
        Dictionary of config group specifications
    """
    return {
        "logistic_ability_shift_delta_sweep": {
            "name": "Logistic Ability Shift - Delta Variations",
            "description": "Shows how different delta values affect the logistic ability shift bias",
            "filters": [
                {"path": ["elicitation_bias_config", "bias_type"], "value": "logistic_ability_shift"},
                {"path": ["elicitation_bias_config", "budget_dependent"], "value": False}
            ]
        },
        "task_filter_threshold_sweep": {
            "name": "Task Filter - Threshold Variations", 
            "description": "Shows how different elicitation threshold values affect the task filter bias",
            "filters": [
                {"path": ["elicitation_bias_config", "bias_type"], "value": "task_filter"},
                {"path": ["elicitation_bias_config", "budget_dependent"], "value": False}
            ]
        },
        "budget_scaling_logistic_ability_shift": {
            "name": "Budget Scaling - Logistic Ability Shift",
            "description": "Shows how budget scaling affects logistic ability shift bias",
            "filters": [
                {"path": ["elicitation_bias_config", "bias_type"], "value": "logistic_ability_shift"},
                {"path": ["elicitation_bias_config", "budget_dependent"], "value": True}
            ]
        },
        "budget_scaling_task_filter": {
            "name": "Budget Scaling - Task Filter",
            "description": "Shows how budget scaling affects task filter bias",
            "filters": [
                {"path": ["elicitation_bias_config", "bias_type"], "value": "task_filter"},
                {"path": ["elicitation_bias_config", "budget_dependent"], "value": True}
            ]
        }
    }


def create_mock_scenario(budget_fraction: float = 1.0) -> EvaluationScenario:
    """
    Create a mock EvaluationScenario for testing elicitation bias.
    
    Args:
        budget_fraction: Budget fraction (0.0 to 1.0)
        
    Returns:
        Mock EvaluationScenario object
    """
    
    # Create mock ability forecast with logistic parameters
    ability = AbilityForecast(
        date=datetime.now(),
        threshold=20.0,
        slope=-0.6,
        scenario="mock_scenario",
        model="mock_model"
    )
    
    scenario = EvaluationScenario(
        ability=ability,
        doubling_rate=2.0,
        intercept=1.0,
        budget_fraction=budget_fraction,
        ability_id="mock_ability",
        cost_id="mock_cost",
        constraint_id="mock_constraint",
        cost_model="mock_cost_model",
        ability_variant="base",
        cost_variant="base",
        base_ability_id="mock_ability_base",
        base_cost_id="mock_cost_base"
    )
    
    return scenario


def logistic_function(x: np.ndarray, threshold: float, slope: float) -> np.ndarray:
    """
    Calculate logistic function values.
    
    Args:
        x: Input values (task difficulties)
        threshold: Threshold parameter (inflection point)
        slope: Slope parameter (steepness, negative for declining)
    
    Returns:
        Array of logistic function values
    """
    return 1.0 / (1.0 + np.exp(-slope * (x - threshold)))


def plot_sensitivity_curves(configs: List[Dict[str, Any]], group_name: str, group_info: Dict[str, Any], 
                           output_dir: Path, timestamp: str):
    """
    Plot sensitivity curves for a config group.
    
    Args:
        configs: List of filtered configuration dictionaries
        group_name: Name of the config group
        group_info: Config group information
        output_dir: Output directory for plots
        timestamp: Timestamp string for filename
    """
    
    if not configs:
        print(f"No configurations found for group: {group_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Sensitivity Curves: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(10, 30, 200)
    threshold = 20.0  # Mock ability threshold
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    ax.set_title(group_info["description"], fontsize=12)
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Sensitivity Rate')
    ax.grid(True, alpha=0.3)
    
    # Plot each configuration
    for i, config in enumerate(configs):
        try:
            # Create evaluation config and scenario
            eval_config = EvaluationConfig(**config)
            scenario = create_mock_scenario(budget_fraction=1.0)
            
            # Get calculated bias
            bias = define_elicitation_bias(scenario, eval_config)
            
            print(f"Config {i}: Bias type={bias.bias_type}, Args={bias.args}")
            
        except Exception as e:
            print(f"Error processing config {i} for {group_name}: {e}")
            continue
        
        # Calculate sensitivity for this configuration
        if bias.bias_type == "logistic_ability_shift":
            # Ratio of two logistic curves: full elicitation vs reduced elicitation
            base_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            delta = bias.args[1] if len(bias.args) > 1 else 1.0
            slope = bias.args[2] if len(bias.args) > 2 else 1.0
            
            # Calculate numerator (full elicitation) and denominator (reduced elicitation)
            numerator = logistic_function(x, base_threshold, slope)
            denominator = logistic_function(x, base_threshold - delta, slope)
            
            # Handle division by zero
            epsilon = 1e-10
            y = np.where(denominator < epsilon, 1.0, numerator / (denominator + epsilon))
            y = np.clip(y, 0, 1)
            label = f'Delta: {delta:.1f}'
            
        elif bias.bias_type == "task_filter":
            # Step function: full sensitivity before threshold, reduced after
            elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            sensitivity_rate_after = bias.args[1] if len(bias.args) > 1 else 0.5
            y = np.where(x <= elicitation_threshold, 1.0, sensitivity_rate_after)
            label = f'Thresh: {elicitation_threshold:.1f}, Rate: {sensitivity_rate_after:.2f}'
        
        else:
            print(f"Unsupported bias type: {bias.bias_type}")
            continue

        ax.plot(x, y, color=colors[i], linewidth=2, label=label)
    
    # Add vertical line at ability threshold
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Save plot
    filename = f"sensitivity_curves_{group_name}_{timestamp}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sensitivity curves plot saved: {filepath}")


def plot_budget_scaling(configs: List[Dict[str, Any]], group_name: str, group_info: Dict[str, Any], 
                       output_dir: Path, timestamp: str):
    """
    Plot budget scaling effects for a config group.
    
    Args:
        configs: List of filtered configuration dictionaries
        group_name: Name of the config group
        group_info: Config group information
        output_dir: Output directory for plots
        timestamp: Timestamp string for filename
    """
    
    if not configs:
        print(f"No configurations found for group: {group_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Budget Scaling: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(10, 30, 200)
    threshold = 20.0  # Mock ability threshold
    
    # Budget fractions to test
    budget_fractions = [1.0, 0.75, 0.5, 0.25, 0.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(budget_fractions)))
    
    ax.set_title(group_info["description"], fontsize=12)
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Sensitivity Rate')
    ax.grid(True, alpha=0.3)
    
    # Use first configuration (they should be similar for budget scaling)
    if not configs:
        return
    
    config = configs[0]
    
    # Plot for different budget fractions
    for i, budget_fraction in enumerate(budget_fractions):
        try:
            # Create evaluation config and scenario with this budget
            eval_config = EvaluationConfig(**config)
            scenario = create_mock_scenario(budget_fraction=budget_fraction)
            
            # Get calculated bias
            bias = define_elicitation_bias(scenario, eval_config)
            
            print(f"Budget {budget_fraction}: Bias type={bias.bias_type}, Args={bias.args}")
            
        except Exception as e:
            print(f"Error processing budget fraction {budget_fraction} for {group_name}: {e}")
            continue
        
        # Calculate sensitivity for this budget
        if bias.bias_type == "logistic_ability_shift":
            # Ratio of two logistic curves: full elicitation vs reduced elicitation
            base_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            delta = bias.args[1] if len(bias.args) > 1 else 1.0
            slope = bias.args[2] if len(bias.args) > 2 else 1.0
            
            # Calculate numerator (full elicitation) and denominator (reduced elicitation)
            numerator = logistic_function(x, base_threshold, slope)
            denominator = logistic_function(x, base_threshold - delta, slope)
            
            # Handle numeric stability
            epsilon = 1e-10
            y = np.where(denominator < epsilon, 1.0, numerator / (denominator + epsilon))
            y = np.clip(y, 0, 1)
            
        elif bias.bias_type == "task_filter":
            # Step function: full sensitivity before threshold, reduced after
            elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            sensitivity_rate_after = bias.args[1] if len(bias.args) > 1 else 0.5
            y = np.where(x <= elicitation_threshold, 1.0, sensitivity_rate_after)
        
        else:
            print(f"Unsupported bias type: {bias.bias_type}")
            continue
            
        label = f'Budget: {budget_fraction:.2f}'
        ax.plot(x, y, color=colors[i], linewidth=2, label=label)
    
    # Add vertical line at ability threshold
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Save plot
    filename = f"budget_scaling_{group_name}_{timestamp}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Budget scaling plot saved: {filepath}")


def plot_success_probability_comparison(configs: List[Dict[str, Any]], group_name: str, group_info: Dict[str, Any], 
                                      output_dir: Path, timestamp: str):
    """
    Plot success probability comparison for a config group.
    
    Args:
        configs: List of filtered configuration dictionaries
        group_name: Name of the config group
        group_info: Config group information
        output_dir: Output directory for plots
        timestamp: Timestamp string for filename
    """
    
    if not configs:
        print(f"No configurations found for group: {group_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Success Probability Impact: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(10, 30, 200)
    threshold = 20.0  # Mock ability threshold
    slope = -0.6     # Mock ability slope
    
    # Calculate baseline success probabilities (logistic curve)
    baseline_probs = logistic_function(x, threshold, slope)
    
    ax.set_title(group_info["description"], fontsize=12)
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Success Probability')
    ax.grid(True, alpha=0.3)
    
    # Plot baseline curve
    ax.plot(x, baseline_probs, 'k--', linewidth=3, alpha=0.7, label='Original (No Bias)')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    # Plot each configuration
    for i, config in enumerate(configs[:5]):  # Limit to 5 configs for readability
        try:
            # Create evaluation config and scenario
            eval_config = EvaluationConfig(**config)
            scenario = create_mock_scenario(budget_fraction=0.5)  # 50% budget constraint
            
            # Get calculated bias
            bias = define_elicitation_bias(scenario, eval_config)
            
        except Exception as e:
            print(f"Error processing config {i} for {group_name}: {e}")
            continue
        
        # Apply bias to baseline probabilities
        if bias.bias_type == "logistic_ability_shift":
            # Ratio of two logistic curves: full elicitation vs reduced elicitation
            base_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            delta = bias.args[1] if len(bias.args) > 1 else 1.0
            slope_bias = bias.args[2] if len(bias.args) > 2 else 1.0
            
            # Calculate numerator (full elicitation) and denominator (reduced elicitation)
            numerator = logistic_function(x, base_threshold, slope_bias)
            denominator = logistic_function(x, base_threshold - delta, slope_bias)
            
            # Handle division by zero and apply bias
            epsilon = 1e-10
            keep_probs = np.where(denominator < epsilon, 1.0, numerator / (denominator + epsilon))
            keep_probs = np.clip(keep_probs, 0, 1)
            biased_probs = baseline_probs * keep_probs
            
            label = f'Delta: {delta:.1f}'
            
        elif bias.bias_type == "task_filter":
            # Step function: full sensitivity before threshold, reduced after
            elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            sensitivity_rate_after = bias.args[1] if len(bias.args) > 1 else 0.5
            keep_probs = np.where(x <= elicitation_threshold, 1.0, sensitivity_rate_after)
            biased_probs = baseline_probs * keep_probs
            
            label = f'Thresh: {elicitation_threshold:.1f}'
        
        else:
            print(f"Unsupported bias type: {bias.bias_type}")
            continue
        
        ax.plot(x, biased_probs, color=colors[i], linewidth=2, alpha=0.8, label=label)
    
    # Add vertical line at ability threshold
    ax.axvline(x=threshold, color='black', linestyle='-', alpha=0.3, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Save plot
    filename = f"success_probability_{group_name}_{timestamp}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success probability plot saved: {filepath}")


def main():
    """Main execution function."""
    
    print("Starting elicitation bias debug visualization...")
    
    # Create output directory
    output_dir = Path("plots/elicitation_bias_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create simple sweep configurations
    print("Creating sweep configurations...")
    sweep_configs = create_simple_sweep_configs()
    
    # Expand all sweep configurations
    all_configs = []
    for sweep_name, sweep_config in sweep_configs.items():
        expanded = expand_sweep_config(sweep_config)
        all_configs.extend(expanded)
        print(f"Generated {len(expanded)} configs from {sweep_name} sweep")
    
    print(f"Total configurations: {len(all_configs)}")
    
    # Define config groups
    config_groups = define_config_groups()
    
    # Create plots for each config group and plot type
    plot_functions = [
        ("sensitivity_curves", plot_sensitivity_curves),
        ("budget_scaling", plot_budget_scaling),
        ("success_probability", plot_success_probability_comparison)
    ]
    
    for group_name, group_info in config_groups.items():
        print(f"\nProcessing config group: {group_name}")
        
        # Filter configurations for this group
        filtered_configs = filter_configs(all_configs, group_info["filters"])
        print(f"Found {len(filtered_configs)} matching configurations")
        
        if not filtered_configs:
            print(f"No configurations found for group: {group_name}")
            continue
        
        # Create plots for this group
        for plot_type, plot_function in plot_functions:
            print(f"Creating {plot_type} plot for {group_name}...")
            try:
                plot_function(filtered_configs, group_name, group_info, output_dir, timestamp)
            except Exception as e:
                print(f"Error creating {plot_type} plot for {group_name}: {e}")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Debug visualization complete!")


if __name__ == "__main__":
    main()