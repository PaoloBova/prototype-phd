"""
Simplified Elicitation Bias Debug Visualization Script

This script creates focused matplotlib plots for debugging elicitation bias behavior.
Uses config group filtering to organize plots by related parameter variations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import required modules from the project
from prototype_phd.utils import expand_sweep_config
from src.forecast_evaluation import define_elicitation_bias, calculate_evaluation_forecast
from src.simulation import (
    generate_task_samples, generate_success_outcomes, logistic_function,
    calculate_sensitivity_rates, apply_sensitivity_to_probabilities
)
from src.schemas import (
    EvaluationConfig, EvaluationScenario, AbilityForecast, 
    CalculatedElicitationBias
)


# Global plotting style parameters
FONT_SIZE = 20  # Default font size
TITLE_ENABLED = False  # Whether to show titles in plots
FILL_AREA = False  # Whether to fill area under the curve

def _apply_font_styles():
    """Apply font size settings before creating plot elements."""
    mpl.rcParams.update({
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE * 0.8,
        'ytick.labelsize': FONT_SIZE * 0.8,
        'legend.fontsize': FONT_SIZE * 0.8,
        'figure.titlesize': FONT_SIZE * 1.2
    })

def _remove_titles_if_needed():
    """Remove titles from plots if TITLE_ENABLED is False."""
    if not TITLE_ENABLED:
        # Remove titles from the current figure
        fig = plt.gcf()
        fig.suptitle("")  # Remove figure suptitle
        for ax in fig.axes:
            ax.set_title("")  # Remove axis title


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
        # Use seaborn's colorblind-friendly palette for categorical data
        if n_colors <= 6:
            # Use the colorblind palette (6 distinct colors)
            colors = sns.color_palette('colorblind', n_colors=n_colors)
        else:
            # Fall back to tab10 for more colors (still colorblind-friendly)
            colors = sns.color_palette('tab10', n_colors=n_colors)
    
    elif plot_type == 'sequential':
        # Use cividis for sequential data (more colorblind-friendly than viridis)
        colors = sns.color_palette('cividis', n_colors=n_colors)
    
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}. Use 'categorical' or 'sequential'")
    
    return colors


def get_line_styles(n_styles: int) -> List[str]:
    """
    Get distinct line styles for additional visual distinction.
    
    Args:
        n_styles: Number of line styles needed
        
    Returns:
        List of line style codes
    """
    base_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    
    # Repeat styles if we need more than available
    return (base_styles * ((n_styles // len(base_styles)) + 1))[:n_styles]


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
                "parameters":{"elicitation_threshold": 11.0, "sensitivity_rate_after": 0.0, "delta": 1.5},
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
                "values": [0.75, 1.5]
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
                "parameters": {"elicitation_threshold": 11.0, "sensitivity_rate_after": 0.0, "delta": 1.5},
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
                "values": [10.0, 15.0, 20.0]
            }
        ]
    }
    
    # Budget scaling sweep - test budget dependent behavior (linear)
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
                "name": "budget_scaling_linear",
                "source_file": None,
                "parameters": {"delta": 1.5,
                               "elicitation_threshold": 11.0,
                               "sensitivity_rate_after": 0.0},
                "budget_dependent": True,
                "budget_scaling": {
                    "delta": {
                        "type": "linear",
                        "params": {"target_value": 0.0}
                    },
                    "elicitation_threshold": {
                        "type": "linear",
                        "params": {"target_type": "upper_bound"}
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
    
    # Logarithmic budget scaling sweep - test logarithmic scaling behavior
    logarithmic_budget_scaling_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0]
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
                "name": "budget_scaling_logarithmic",
                "source_file": None,
                "parameters": {"delta": 2.0,
                               "elicitation_threshold": 11.0,
                               "sensitivity_rate_after": 0.0},
                "budget_dependent": True,
                "budget_scaling": {
                    "delta": {
                        "type": "log2",
                        "params": {
                            "target_value": 0.0,
                        }
                    },
                    "elicitation_threshold": {
                        "type": "log2",
                        "params": {
                            "target_type": "upper_bound",
                        }
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
        "budget_scaling": budget_scaling_sweep,
        "logarithmic_budget_scaling": logarithmic_budget_scaling_sweep
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
        },
        "logarithmic_budget_scaling_logistic_ability_shift": {
            "name": "Logarithmic Budget Scaling - Logistic Ability Shift",
            "description": "Shows how logarithmic budget scaling affects logistic ability shift bias",
            "filters": [
                {"path": ["elicitation_bias_config", "bias_type"], "value": "logistic_ability_shift"},
                {"path": ["elicitation_bias_config", "budget_dependent"], "value": True},
                {"path": ["elicitation_bias_config", "name"], "value": "budget_scaling_logarithmic"}
            ]
        },
        "logarithmic_budget_scaling_task_filter": {
            "name": "Logarithmic Budget Scaling - Task Filter",
            "description": "Shows how logarithmic budget scaling affects task filter bias",
            "filters": [
                {"path": ["elicitation_bias_config", "bias_type"], "value": "task_filter"},
                {"path": ["elicitation_bias_config", "budget_dependent"], "value": True},
                {"path": ["elicitation_bias_config", "name"], "value": "budget_scaling_logarithmic"}
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
        slope=-0.665,
        scenario="mock_scenario",
        model="mock_model"
    )
    
    scenario = EvaluationScenario(
        ability=ability,
        doubling_rate=1.0,
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
    
    # Apply font styles before creating any plot elements
    _apply_font_styles()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Sensitivity Curves: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(0, 30, 2000)
    colors = get_plot_colors(len(configs), plot_type='categorical')
    line_styles = get_line_styles(len(configs))
    
    ax.set_title(group_info["description"], fontsize=12)
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Sensitivity Rate')
    ax.grid(True, alpha=0.3)
    
    # Plot each configuration
    for i, config in enumerate(configs):
        try:
            # Create evaluation config and scenario
            eval_config = EvaluationConfig(**config)
            scenario = create_mock_scenario(budget_fraction=0)
            threshold = scenario.ability.threshold
            
            # Get calculated bias
            bias = define_elicitation_bias(scenario, eval_config)
            
            
            
            print(f"Config {i}: Bias type={bias.bias_type}, Args={bias.args}")
            
        except Exception as e:
            print(f"Error processing config {i} for {group_name}: {e}")
            continue
        
        # Calculate sensitivity using the modular function from simulation.py
        try:
            # Create a full forecast object for the modular function
            forecast = calculate_evaluation_forecast(scenario, eval_config)
            y = calculate_sensitivity_rates(x, bias.bias_type, bias.args, forecast)
            
            # Create appropriate label based on bias type
            if bias.bias_type == "logistic_ability_shift":
                delta = bias.args[1] if len(bias.args) > 1 else 1.0
                label = f'Delta: {delta:.1f}'
            elif bias.bias_type == "task_filter":
                elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                sensitivity_rate_after = bias.args[1] if len(bias.args) > 1 else 0.5
                label = f'Thresh: {elicitation_threshold:.1f}, Rate: {sensitivity_rate_after:.2f}'
            else:
                label = f'Bias: {bias.bias_type}'
                
        except Exception as e:
            print(f"Error calculating sensitivity for {bias.bias_type}: {e}")
            continue

        # Add filled area under the curve
        if FILL_AREA:
            ax.fill_between(x, 0, y, color=colors[i], alpha=0.3)
        ax.plot(x, y, color=colors[i], linewidth=2, label=label)
    
    # Add vertical line at ability threshold
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Save plot
    filename = f"sensitivity_curves_{group_name}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    _remove_titles_if_needed()
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
    
    # Apply font styles before creating any plot elements
    _apply_font_styles()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Budget Scaling: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(0, 30, 2000)
    
    # Budget fractions to test
    budget_fractions = [1.0, 0.75, 0.5, 0.25]
    colors = get_plot_colors(len(budget_fractions), plot_type='sequential')
    line_styles = get_line_styles(len(budget_fractions))
    
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
            threshold = scenario.ability.threshold
            
            # Get calculated bias
            bias = define_elicitation_bias(scenario, eval_config)
            
            print(f"Budget {budget_fraction}: Bias type={bias.bias_type}, Args={bias.args}")
            
        except Exception as e:
            print(f"Error processing budget fraction {budget_fraction} for {group_name}: {e}")
            continue
        
        # Calculate sensitivity using the modular function from simulation.py
        try:
            # Create a full forecast object for the modular function
            forecast = calculate_evaluation_forecast(scenario, eval_config)
            y = calculate_sensitivity_rates(x, bias.bias_type, bias.args, forecast)
            
        except Exception as e:
            print(f"Error calculating sensitivity for {bias.bias_type} with budget {budget_fraction}: {e}")
            continue
            
        # Create informative label with budget and scaled parameter
        if bias.bias_type == "logistic_ability_shift":
            delta = bias.args[1] if len(bias.args) > 1 else 1.0
            label = f'Budget: {budget_fraction:.2f} (δ={delta:.2f})'
        elif bias.bias_type == "task_filter":
            elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            label = f'Budget: {budget_fraction:.2f} (thresh={elicitation_threshold:.1f})'
        else:
            label = f'Budget: {budget_fraction:.2f}'
        # Add filled area under the curve
        if FILL_AREA:
            ax.fill_between(x, 0, y, color=colors[i], alpha=0.3)
        ax.plot(x, y, color=colors[i], linewidth=2, label=label)
    
    # Add vertical line at ability threshold
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Save plot
    filename = f"budget_scaling_{group_name}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    _remove_titles_if_needed()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Budget scaling plot saved: {filepath}")


def plot_budget_scaling_success_rates(configs: List[Dict[str, Any]], group_name: str, group_info: Dict[str, Any], 
                                     output_dir: Path, timestamp: str):
    """
    Plot success rate curves for budget scaling effects.
    
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
    
    # Apply font styles before creating any plot elements
    _apply_font_styles()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Budget Scaling Success Rates: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(0, 30, 2000)
    
    # Budget fractions to test
    budget_fractions = [1.0, 0.75, 0.5, 0.25]
    colors = get_plot_colors(len(budget_fractions), plot_type='sequential')
    
    ax.set_title(group_info["description"], fontsize=12)
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Success Probability')
    ax.grid(True, alpha=0.3)
    
    # Use first configuration (they should be similar for budget scaling)
    if not configs:
        return
    
    config = configs[0]
    
    # Plot baseline (no bias) curve first
    scenario_baseline = create_mock_scenario(budget_fraction=1.0)
    threshold = scenario_baseline.ability.threshold
    slope = scenario_baseline.ability.slope
    baseline_probs = logistic_function(x, threshold, slope)
    ax.plot(x, baseline_probs, 'k--', linewidth=3, alpha=0.7, label='No Bias (Budget: 1.00)')
    
    # Track previous line for consecutive shading
    previous_probs = baseline_probs.copy()
    
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
        
        # Calculate success probabilities with bias applied
        try:
            # Create a full forecast object for the modular function
            forecast = calculate_evaluation_forecast(scenario, eval_config)
            
            # Calculate sensitivity rates and apply to baseline probabilities
            sensitivity_rates = calculate_sensitivity_rates(x, bias.bias_type, bias.args, forecast)
            biased_probs = apply_sensitivity_to_probabilities(baseline_probs, sensitivity_rates)
            
        except Exception as e:
            print(f"Error calculating success probabilities for {bias.bias_type} with budget {budget_fraction}: {e}")
            continue
            
        # Create informative label with budget and scaled parameter
        if bias.bias_type == "logistic_ability_shift":
            delta = bias.args[1] if len(bias.args) > 1 else 1.0
            label = f'Budget: {budget_fraction:.2f} (δ={delta:.2f})'
        elif bias.bias_type == "task_filter":
            elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
            label = f'Budget: {budget_fraction:.2f} (thresh={elicitation_threshold:.1f})'
        else:
            label = f'Budget: {budget_fraction:.2f}'
            
        # Add filled area under the curve (always for this function)
        ax.fill_between(x, biased_probs, previous_probs, color=colors[i], alpha=0.3)
        ax.plot(x, biased_probs, color=colors[i], linewidth=2, label=label)
        
        # Update previous_probs for next iteration
        previous_probs = biased_probs.copy()
    
    # Add vertical line at ability threshold
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Save plot
    filename = f"budget_scaling_success_rates_{group_name}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    _remove_titles_if_needed()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Budget scaling success rates plot saved: {filepath}")


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
    
    # Apply font styles before creating any plot elements
    _apply_font_styles()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Success Probability Impact: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(0, 30, 2000)
    

    # Create evaluation config and scenario
    scenario = create_mock_scenario(budget_fraction=0.0)
    # Calculate baseline success probabilities (logistic curve)
    threshold = scenario.ability.threshold
    slope = scenario.ability.slope
    baseline_probs = logistic_function(x, threshold, slope)
    
    ax.set_title(group_info["description"], fontsize=12)
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Success Probability')
    ax.grid(True, alpha=0.3)
    
    # Plot baseline curve
    ax.plot(x, baseline_probs, 'k--', linewidth=3, alpha=0.7, label='Original (No Bias)')
    
    colors = get_plot_colors(len(configs), plot_type='categorical')
    line_styles = get_line_styles(len(configs))
    
    # Plot each configuration
    for i, config in enumerate(configs):
        try:
            # Get calculated bias
            eval_config = EvaluationConfig(**config)
            bias = define_elicitation_bias(scenario, eval_config)
            
        except Exception as e:
            print(f"Error processing config {i} for {group_name}: {e}")
            continue
        
        # Apply bias to baseline probabilities using modular functions
        try:
            # Create a full forecast object for the modular function
            forecast = calculate_evaluation_forecast(scenario, eval_config)
            
            # Calculate sensitivity rates and apply to probabilities
            sensitivity_rates = calculate_sensitivity_rates(x, bias.bias_type, bias.args, forecast)
            biased_probs = apply_sensitivity_to_probabilities(baseline_probs, sensitivity_rates)
            
            # Create appropriate label based on bias type
            if bias.bias_type == "logistic_ability_shift":
                delta = bias.args[1] if len(bias.args) > 1 else 1.0
                label = f'Delta: {delta:.1f}'
            elif bias.bias_type == "task_filter":
                elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                label = f'Thresh: {elicitation_threshold:.1f}'
            else:
                label = f'Bias: {bias.bias_type}'
                
        except Exception as e:
            print(f"Error applying bias for {bias.bias_type}: {e}")
            continue
        
        # Add filled area under the curve
        if FILL_AREA:
            ax.fill_between(x, 0, biased_probs, color=colors[i], alpha=0.3)
        ax.plot(x, biased_probs, color=colors[i], linewidth=2, alpha=0.8, label=label)
    
    # Add vertical line at ability threshold
    ax.axvline(x=threshold, color='black', linestyle='-', alpha=0.3, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Save plot
    filename = f"success_probability_{group_name}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    _remove_titles_if_needed()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success probability plot saved: {filepath}")


def main():
    """Main execution function."""
    
    print("Starting elicitation bias debug visualization...")
    
    # Create output directory
    output_dir = Path("elicitation_bias_debug")
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
        
        # Create additional budget scaling success rate plots for budget scaling groups
        if "budget_scaling" in group_name:
            print(f"Creating budget scaling success rates plot for {group_name}...")
            try:
                plot_budget_scaling_success_rates(filtered_configs, group_name, group_info, output_dir, timestamp)
            except Exception as e:
                print(f"Error creating budget scaling success rates plot for {group_name}: {e}")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Debug visualization complete!")


if __name__ == "__main__":
    main()