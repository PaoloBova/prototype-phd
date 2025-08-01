"""
Alternate Ability Debug Visualization Script

This script creates focused matplotlib plots for debugging alternate ability function behavior.
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
from src.alternate_ability import (
    evaluate_alternate_ability,
    calculate_alternate_ability_threshold
)
from src.schemas import (
    EvaluationConfig, EvaluationScenario, AbilityForecast, 
    CalculatedElicitationBias, AlternateAbilityType
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
    # Exclude solid line since that's used for baseline
    base_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
    
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


def create_alternate_ability_sweep_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create sweep configurations for alternate ability functions.
    
    Returns:
        Dictionary of sweep configuration functions
    """
    
    # Richards (Generalized Logistic) function sweep
    richards_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.5, 0.0]
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
                "enabled": False,
                "bias_type": "fall_past_threshold",
                "name": "no_bias",
                "source_file": None,
                "parameters": {},
                "budget_dependent": False,
                "budget_scaling": {}
            },
            "alternate_ability": {
                "enabled": True,
                "function_type": "richards_generalized_logistic",
                "name": "richards_sweep",
                "args": {
                    "threshold": 20.0,
                    "slope": -0.665,
                    "asymmetry": 1.0,
                    "growth_rate": 1.0
                }
            }
        },
        "_sweep": [
            {
                "path": ["alternate_ability", "args", "asymmetry"],
                "values": [0.5, 1.0, 2.0, 5.0]
            }
        ]
    }
    
    # Exponential function sweep
    exponential_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.5, 0.0]
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
                "enabled": False,
                "bias_type": "fall_past_threshold",
                "name": "no_bias",
                "source_file": None,
                "parameters": {},
                "budget_dependent": False,
                "budget_scaling": {}
            },
            "alternate_ability": {
                "enabled": True,
                "function_type": "exponential",
                "name": "exponential_sweep",
                "args": {
                    "threshold": 20.0,
                    "decay_rate": 0.1,
                    "asymptote": 0.0
                }
            }
        },
        "_sweep": [
            {
                "path": ["alternate_ability", "args", "decay_rate"],
                "values": [0.05, 0.1, 0.2, 0.5]
            }
        ]
    }
    
    # Power law function sweep
    power_law_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.5, 0.0]
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
                "enabled": False,
                "bias_type": "fall_past_threshold",
                "name": "no_bias",
                "source_file": None,
                "parameters": {},
                "budget_dependent": False,
                "budget_scaling": {}
            },
            "alternate_ability": {
                "enabled": True,
                "function_type": "power_law",
                "name": "power_law_sweep",
                "args": {
                    "threshold": 20.0,
                    "exponent": -2.0,
                    "scale": 1.0
                }
            }
        },
        "_sweep": [
            {
                "path": ["alternate_ability", "args", "exponent"],
                "values": [-3.0, -2.0, -1.0, -0.5]
            }
        ]
    }
    
    # Logistic function parameter sweep (for comparison)
    logistic_sweep = {
        "base_config": {
            "resource_constraints": {
                "static_budgets": [1.0, 0.5, 0.0]
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
                "enabled": False,
                "bias_type": "fall_past_threshold",
                "name": "no_bias",
                "source_file": None,
                "parameters": {},
                "budget_dependent": False,
                "budget_scaling": {}
            },
            "alternate_ability": {
                "enabled": True,
                "function_type": "logistic",
                "name": "logistic_sweep",
                "args": {
                    "threshold": 20.0,
                    "slope": -0.665
                }
            }
        },
        "_sweep": [
            {
                "path": ["alternate_ability", "args", "slope"],
                "values": [-1.0, -0.665, -0.5, -0.3]
            }
        ]
    }
    
    return {
        "richards": richards_sweep,
        "exponential": exponential_sweep,
        "power_law": power_law_sweep,
        "logistic": logistic_sweep
    }


def define_config_groups() -> Dict[str, Dict[str, Any]]:
    """
    Define config groups for filtering and organizing plots.
    
    Returns:
        Dictionary of config group specifications
    """
    return {
        "richards_asymmetry_sweep": {
            "name": "Richards Function - Asymmetry Variations",
            "description": "Shows how different asymmetry parameters affect the Richards (generalized logistic) function",
            "filters": [
                {"path": ["alternate_ability", "function_type"], "value": "richards_generalized_logistic"},
                {"path": ["alternate_ability", "enabled"], "value": True}
            ]
        },
        "exponential_decay_sweep": {
            "name": "Exponential Function - Decay Rate Variations", 
            "description": "Shows how different decay rates affect the exponential function",
            "filters": [
                {"path": ["alternate_ability", "function_type"], "value": "exponential"},
                {"path": ["alternate_ability", "enabled"], "value": True}
            ]
        },
        "power_law_exponent_sweep": {
            "name": "Power Law Function - Exponent Variations",
            "description": "Shows how different exponents affect the power law function",
            "filters": [
                {"path": ["alternate_ability", "function_type"], "value": "power_law"},
                {"path": ["alternate_ability", "enabled"], "value": True}
            ]
        },
        "logistic_slope_sweep": {
            "name": "Logistic Function - Slope Variations",
            "description": "Shows how different slopes affect the logistic function",
            "filters": [
                {"path": ["alternate_ability", "function_type"], "value": "logistic"},
                {"path": ["alternate_ability", "enabled"], "value": True}
            ]
        }
    }


def create_mock_scenario(budget_fraction: float = 1.0) -> EvaluationScenario:
    """
    Create a mock EvaluationScenario for testing alternate ability functions.
    
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


def evaluate_alternate_ability_local(x: np.ndarray, function_type: str, args: Dict[str, float]) -> np.ndarray:
    """
    Evaluate alternate ability function at given difficulty values using shared functions.
    
    Args:
        x: Array of difficulty values
        function_type: Type of alternate ability function
        args: Function arguments
        
    Returns:
        Array of success probabilities
    """
    return evaluate_alternate_ability(x, function_type, args)


def calculate_true_threshold_50(function_type: str, args: Dict[str, float], 
                               base_threshold: float = 20.0, base_slope: float = -0.665) -> float:
    """
    Calculate the true 50% threshold for alternate ability functions.
    
    Args:
        function_type: Type of alternate ability function
        args: Function arguments
        base_threshold: Base threshold parameter (fallback)
        base_slope: Base slope parameter (fallback)
        
    Returns:
        True 50% threshold value
    """
    alt_config = {
        "function_type": function_type,
        "args": args
    }
    return calculate_alternate_ability_threshold(alt_config, base_threshold, base_slope)


def plot_alternate_ability_curves(configs: List[Dict[str, Any]], group_name: str, group_info: Dict[str, Any], 
                                 output_dir: Path, timestamp: str):
    """
    Plot alternate ability curves for a config group.
    
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
    fig.suptitle(f'Alternate Ability Curves: {group_info["name"]}', fontsize=14, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(5, 35, 2000)
    colors = get_plot_colors(len(configs), plot_type='categorical')
    line_styles = get_line_styles(len(configs))
    
    ax.set_title(group_info["description"], fontsize=12)
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Success Probability')
    ax.grid(True, alpha=0.3)
    
    # Plot baseline logistic curve for comparison
    scenario = create_mock_scenario(budget_fraction=1.0)
    threshold = scenario.ability.threshold
    slope = scenario.ability.slope
    baseline_probs = logistic_function(x, threshold, slope)
    ax.plot(x, baseline_probs, 'k-', linewidth=3, alpha=0.8, label='Baseline Logistic')
    
    # Plot each configuration
    for i, config in enumerate(configs):
        try:
            # Create evaluation config and scenario
            eval_config = EvaluationConfig(**config)
            scenario = create_mock_scenario(budget_fraction=1.0)
            
            # Get alternate ability configuration
            alt_ability = eval_config.alternate_ability
            
            print(f"Config {i}: Function type={alt_ability.function_type}, Args={alt_ability.args}")
            
        except Exception as e:
            print(f"Error processing config {i} for {group_name}: {e}")
            continue
        
        # Calculate success probabilities using alternate ability function
        try:
            y = evaluate_alternate_ability_local(x, alt_ability.function_type.value, alt_ability.args)
            
            # Calculate true 50% threshold for this configuration
            true_threshold_50 = calculate_true_threshold_50(
                alt_ability.function_type.value, alt_ability.args, 
                base_threshold=threshold, base_slope=slope
            )
            
            # Create appropriate label based on function type and varying parameter
            if alt_ability.function_type.value == "richards_generalized_logistic":
                asymmetry = alt_ability.args.get("asymmetry", 1.0)
                label = f'Richards (ν={asymmetry:.1f}, τ₅₀={true_threshold_50:.1f})'
            elif alt_ability.function_type.value == "exponential":
                decay_rate = alt_ability.args.get("decay_rate", 0.1)
                label = f'Exponential (λ={decay_rate:.2f}, τ₅₀={true_threshold_50:.1f})'
            elif alt_ability.function_type.value == "power_law":
                exponent = alt_ability.args.get("exponent", -2.0)
                label = f'Power Law (α={exponent:.1f}, τ₅₀={true_threshold_50:.1f})'
            elif alt_ability.function_type.value == "logistic":
                slope_val = alt_ability.args.get("slope", -0.665)
                label = f'Logistic (β={slope_val:.3f}, τ₅₀={true_threshold_50:.1f})'
            else:
                label = f'{alt_ability.function_type.value} (τ₅₀={true_threshold_50:.1f})'
                
        except Exception as e:
            print(f"Error calculating alternate ability for {alt_ability.function_type}: {e}")
            continue

        # Add filled area under the curve
        if FILL_AREA:
            ax.fill_between(x, 0, y, color=colors[i], alpha=0.3)
        ax.plot(x, y, color=colors[i], linewidth=2, linestyle=line_styles[i], label=label)
        
        # Add vertical line at true 50% threshold for this function
        ax.axvline(x=true_threshold_50, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
    
    # Add vertical line at ability threshold (original baseline)
    ax.axvline(x=threshold, color='black', linestyle=':', alpha=0.7, label=f'Baseline τ₅₀ ({threshold:.1f})')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(5, 35)
    
    # Save plot
    filename = f"alternate_ability_curves_{group_name}.png"
    filepath = output_dir / filename
    plt.tight_layout()
    _remove_titles_if_needed()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Alternate ability curves plot saved: {filepath}")


def main():
    """Main execution function."""
    
    print("Starting alternate ability debug visualization...")
    
    # Create output directory
    output_dir = Path("alternate_ability_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create alternate ability sweep configurations
    print("Creating alternate ability sweep configurations...")
    sweep_configs = create_alternate_ability_sweep_configs()
    
    # Expand all sweep configurations
    all_configs = []
    for sweep_name, sweep_config in sweep_configs.items():
        expanded = expand_sweep_config(sweep_config)
        all_configs.extend(expanded)
        print(f"Generated {len(expanded)} configs from {sweep_name} sweep")
    
    print(f"Total configurations: {len(all_configs)}")
    
    # Define config groups
    config_groups = define_config_groups()
    
    # Create plots for each config group
    for group_name, group_info in config_groups.items():
        print(f"\nProcessing config group: {group_name}")
        
        # Filter configurations for this group
        filtered_configs = filter_configs(all_configs, group_info["filters"])
        print(f"Found {len(filtered_configs)} matching configurations")
        
        if not filtered_configs:
            print(f"No configurations found for group: {group_name}")
            continue
        
        # Create plot for this group
        print(f"Creating alternate ability curves plot for {group_name}...")
        try:
            plot_alternate_ability_curves(filtered_configs, group_name, group_info, output_dir, timestamp)
        except Exception as e:
            print(f"Error creating alternate ability curves plot for {group_name}: {e}")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Alternate ability debug visualization complete!")


if __name__ == "__main__":
    main()
