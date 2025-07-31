"""
Elicitation Bias Visualization Script

This script creates comprehensive matplotlib plots showing different elicitation bias types,
their parameter effects, and impact on success probabilities. It serves as a debugging
tool for understanding elicitation bias behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
import copy

# Import required modules from the project
from prototype_phd.utils import expand_sweep_config
from src.forecast_evaluation import define_elicitation_bias
from src.schemas import (
    EvaluationConfig, EvaluationScenario, AbilityForecast, 
    CalculatedElicitationBias
)


def create_logistic_ability_shift_sweep() -> Dict[str, Any]:
    """
    Create sweep configuration for logistic_ability_shift bias type.
    Focuses on varying delta parameter with budget scaling.
    
    Returns:
        Dictionary with base_config and _sweep specification
    """
    
    base_config = {
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
            "budget_dependent": True,
            "budget_scaling": {
                "delta": {
                    "type": "linear",
                    "params": {"target_value": 1.0}
                }
            }
        },
        
        "alternate_ability": {
            "enabled": False,
            "function_type": "logistic",
            "args": {}
        }
    }
    
    sweep_config = {
        "base_config": base_config,
        "_sweep": [
            # Sweep over delta values - the key parameter for logistic_ability_shift
            {
                "path": ["elicitation_bias_config", "parameters", "delta"],
                "values": [1.0, 2.0, 3.0, 4.0, 5.0]
            }
        ]
    }
    
    return sweep_config


def create_task_filter_sweep() -> Dict[str, Any]:
    """
    Create sweep configuration for task_filter bias type.
    Focuses on varying elicitation_threshold and sensitivity_rate_after parameters.
    
    Returns:
        Dictionary with base_config and _sweep specification
    """
    
    base_config = {
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
            "budget_dependent": True,
            "budget_scaling": {
                "elicitation_threshold": {
                    "type": "linear",
                    "params": {"target_value": 25.0}
                }
            }
        },
        
        "alternate_ability": {
            "enabled": False,
            "function_type": "logistic",
            "args": {}
        }
    }
    
    sweep_config = {
        "base_config": base_config,
        "_sweep": [
            # Primary sweep over elicitation_threshold values - key parameter for task_filter
            {
                "path": ["elicitation_bias_config", "parameters", "elicitation_threshold"],
                "values": [10.0, 15.0, 20.0, 25.0, 30.0]
            }
        ]
    }
    
    return sweep_config


def create_budget_dependent_sweep() -> Dict[str, Any]:
    """
    Create sweep configuration to test budget dependency for new bias types.
    Tests both logistic_ability_shift and task_filter with budget scaling.
    
    Returns:
        Dictionary with base_config and _sweep specification
    """
    
    base_config = {
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
            "name": "budget_dependent",
            "source_file": None,
            "parameters": {"delta": 2.0, "elicitation_threshold": 15.0, "sensitivity_rate_after": 0.5},
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
    }
    
    sweep_config = {
        "base_config": base_config,
        "_sweep": [
            # Sweep over bias types to compare both new types
            {
                "path": ["elicitation_bias_config", "bias_type"],
                "values": ["logistic_ability_shift", "task_filter"]
            }
        ]
    }
    
    return sweep_config


def create_budget_gap_sweep_config() -> Dict[str, Any]:
    """
    Create sweep configuration for budget gap analysis.
    
    Returns:
        Dictionary with configurations varying budget gaps
    """
    
    base_config = {
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
            "bias_type": "task_filter",
            "name": "budget_dependent",
            "source_file": None,
            "parameters": {"elicitation_threshold": 15.0, "sensitivity_rate_after": 1.0, "delta": 1.0},
            "budget_dependent": True,
            "budget_scaling": {
                "sensitivity_rate_after": {
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
    }
    
    sweep_config = {
        "base_config": base_config,
        "_sweep": [
            # Sweep over budget scaling types with appropriate parameters
            {
                "path": ["elicitation_bias_config", "budget_scaling"],
                "values": [
                    # Linear scaling - threshold scales with budget
                    {"elicitation_threshold": {"type": "linear", "params": {"target_value": 25.0}}},
                    
                    # Exponential scaling  
                    {"elicitation_threshold": {"type": "exponential", "params": {"decay_rate": 2.0}}},
                    
                    # Logistic scaling
                    {"elicitation_threshold": {"type": "logistic", "params": {"midpoint": 0.5, "steepness": 4.0, "min_value": 15.0, "max_value": 25.0}}}
                ]
            }
        ]
    }
    
    return sweep_config


def create_budget_scaling_sweep() -> Dict[str, Any]:
    """
    Create sweep configuration focused specifically on budget scaling types and parameters.
    This is the dedicated function for exploring budget scaling behavior.
    
    Returns:
        Dictionary with base_config and _sweep specification
    """
    
    base_config = {
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
            "name": "budget_scaling_study",
            "source_file": None,
            "parameters": {"delta": 2.0, "elicitation_threshold": 15.0, "sensitivity_rate_after": 0.5},
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
    }
    
    sweep_config = {
        "base_config": base_config,
        "_sweep": [
            # Comprehensive sweep over budget scaling types and parameters
            {
                "path": ["elicitation_bias_config", "budget_scaling"],
                "values": [
                    # Linear scaling - delta shrinks toward lower value as budget decreases
                    {"delta": {"type": "linear", "params": {"target_value": 1.0}}},
                    
                    # Exponential scaling  
                    {"delta": {"type": "exponential", "params": {"decay_rate": 1.0}}},
                    
                    # Logistic scaling
                    {"delta": {"type": "logistic", "params": {"midpoint": 0.5, "steepness": 4.0, "min_value": 1.0, "max_value": 3.0}}}
                ]
            }
        ]
    }
    
    return sweep_config


def create_mock_scenario(budget_fraction: float = 1.0) -> EvaluationScenario:
    """
    Create a mock EvaluationScenario for testing elicitation bias.
    
    Args:
        budget_fraction: Budget fraction (0.0 to 1.0)
        
    Returns:
        Mock EvaluationScenario object
    """
    
    # Create mock ability forecast with logistic parameters
    # slope=-0.6, threshold=20 as requested
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


def apply_elicitation_bias(
    task_difficulties: np.ndarray,
    success_probs: np.ndarray, 
    bias: CalculatedElicitationBias,
    threshold: float,
    window_upper: float
) -> np.ndarray:
    """
    Apply elicitation bias to success probabilities.
    
    Args:
        task_difficulties: Array of task difficulties
        success_probs: Original success probabilities
        bias: Calculated elicitation bias object
        threshold: Ability threshold
        window_upper: Upper bound of evaluation window
        
    Returns:
        Success probabilities after applying elicitation bias
    """
    
    if not bias.enabled:
        return success_probs
    
    # Calculate sensitivity rates based on bias type
    if bias.bias_type == "logistic_ability_shift":
        # Ratio of two logistic curves: full elicitation vs reduced elicitation
        base_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
        delta = bias.args[1] if len(bias.args) > 1 else 1.0
        slope = bias.args[2] if len(bias.args) > 2 else 1.0
        
        # Calculate numerator (full elicitation) and denominator (reduced elicitation)
        numerator = logistic_function(task_difficulties, base_threshold, slope)
        denominator = logistic_function(task_difficulties, base_threshold - delta, slope)
        
        # Handle numeric stability: when both are near 0, ratio approaches 1
        # Use small epsilon to avoid division by zero
        epsilon = 1e-10
        keep_probs = np.where(denominator < epsilon, 1.0, numerator / (denominator + epsilon))
        keep_probs = np.clip(keep_probs, 0, 1)
        
    elif bias.bias_type == "task_filter":
        # Step function: full sensitivity before threshold, reduced after
        elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
        sensitivity_rate_after = bias.args[1] if len(bias.args) > 1 else 0.5
        keep_probs = np.where(task_difficulties <= elicitation_threshold, 1.0, sensitivity_rate_after)
        
    else:
        raise ValueError(f"Unsupported elicitation bias type: {bias.bias_type}")
    
    # Apply bias as multiplicative factor on success probabilities
    return success_probs * keep_probs


def plot_elicitation_bias_curves(configs: List[Dict[str, Any]], 
                                save_path: str = "elicitation_bias_curves.png"):
    """
    Plot elicitation bias curves showing how sensitivity varies with task difficulty.
    
    Args:
        configs: List of expanded configuration dictionaries
        save_path: Path to save the plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Elicitation Bias Curves: Sensitivity vs Task Difficulty', fontsize=16, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(10, 30, 200)
    threshold = 20.0  # Mock ability threshold
    window_upper = 30.0
    
    # Separate configurations by bias type
    bias_types = ["logistic_ability_shift", "task_filter"]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, bias_type in enumerate(bias_types):
        if i >= 3:  # Only plot first 3 types
            break
            
        ax = axes[i//2, i%2]
        ax.set_title(f'{bias_type.replace("_", " ").title()} Bias', fontweight='bold')
        ax.set_xlabel('Task Difficulty')
        ax.set_ylabel('Sensitivity Rate')
        ax.grid(True, alpha=0.3)
        
        # Filter configs for this bias type
        type_configs = [c for c in configs if c["elicitation_bias_config"]["bias_type"] == bias_type]
        
        color_idx = 0
        for config in type_configs:  # Limit to 5 configs per type
            
            try:
                # Create evaluation config and scenario
                eval_config = EvaluationConfig(**config)
                scenario = create_mock_scenario(budget_fraction=1.0)
                
                # Get calculated bias
                bias = define_elicitation_bias(scenario, eval_config)
            except Exception as e:
                print(f"Error processing config for {bias_type}: {e}")
                continue
            
            # Calculate sensitivity for this configuration
            if bias_type == "logistic_ability_shift":
                # Ratio of two logistic curves: full elicitation vs reduced elicitation
                base_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                delta = bias.args[1] if len(bias.args) > 1 else 1.0
                slope = bias.args[2] if len(bias.args) > 2 else 1.0
                
                # Calculate numerator (full elicitation) and denominator (reduced elicitation)
                numerator = logistic_function(x, base_threshold, slope)
                denominator = logistic_function(x, base_threshold - delta, slope)
                y = np.clip(numerator / denominator, 0, 1)
                label = f'Delta: {delta:.1f}, Threshold: {base_threshold:.1f}'
                
            elif bias_type == "task_filter":
                # Step function: full sensitivity before threshold, reduced after
                elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                sensitivity_rate_after = bias.args[1] if len(bias.args) > 1 else 0.5
                y = np.where(x <= elicitation_threshold, 1.0, sensitivity_rate_after)
                label = f'Thresh: {elicitation_threshold:.1f}, Rate: {sensitivity_rate_after:.2f}'

            ax.plot(x, y, color=colors[color_idx % len(colors)], linewidth=2, label=label)
            color_idx += 1
        
        # Add vertical line at ability threshold
        ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label='Ability Threshold')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    
    # Remove unused subplot
    if len(bias_types) < 4:
        axes[1, 1].remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Elicitation bias curves saved to {save_path}")


def plot_budget_gap_impact(configs: List[Dict[str, Any]], 
                          save_path: str = "budget_gap_impact.png"):
    """
    Plot how elicitation bias curves change with budget constraints.
    
    Args:
        configs: List of expanded configuration dictionaries
        save_path: Path to save the plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Budget Gap Impact on Elicitation Bias', fontsize=16, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(10, 30, 200)
    threshold = 20.0  # Mock ability threshold
    window_upper = 30.0
    
    # Budget fractions to test
    budget_fractions = [1.0, 0.75, 0.5, 0.25, 0.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(budget_fractions)))
    
    # Group configs by scaling type
    scaling_groups = {"linear": [], "exponential": [], "logistic": []}
    
    for config in configs:
        bias_config = config["elicitation_bias_config"]
        scaling_config = bias_config.get("budget_scaling", {})
        
        # Look for the scaling type in sensitivity_rate parameter
        if "sensitivity_rate" in scaling_config:
            scaling_type = scaling_config["sensitivity_rate"].get("type", "unknown")
            if scaling_type in scaling_groups:
                scaling_groups[scaling_type].append(config)
        # Fallback: also check param_0 format if present
        elif "param_0" in scaling_config:
            scaling_type = scaling_config["param_0"].get("type", "unknown")
            if scaling_type in scaling_groups:
                scaling_groups[scaling_type].append(config)
    
    # Plot each scaling type
    scaling_types = ["linear", "exponential", "logistic"]
    for i, scaling_type in enumerate(scaling_types):
        ax = axes[i//2, i%2]
        ax.set_title(f'{scaling_type.title()} Budget Scaling', fontweight='bold')
        ax.set_xlabel('Task Difficulty')
        ax.set_ylabel('Sensitivity Rate')
        ax.grid(True, alpha=0.3)
        
        # Use first config of this scaling type, or create default
        if not scaling_groups[scaling_type]:
            print(f"No configurations found for scaling type: {scaling_type}")
            continue
        type_config = scaling_groups[scaling_type][0]
        
        # Plot for different budget fractions
        for j, budget_fraction in enumerate(budget_fractions):
            
            try:
                # Create evaluation config and scenario with this budget
                eval_config = EvaluationConfig(**type_config)
                scenario = create_mock_scenario(budget_fraction=budget_fraction)
                
                # Get calculated bias
                bias = define_elicitation_bias(scenario, eval_config)
            except Exception as e:
                print(f"Error processing budget fraction {budget_fraction} for {scaling_type}: {e}")
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
                
            label = f'Budget: {budget_fraction:.2f}'
            ax.plot(x, y, color=colors[j], linewidth=2, label=label)
        
        # Add vertical line at ability threshold
        ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label='Ability Threshold')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    
    # Remove unused subplot
    if len(scaling_types) < 4:
        axes[1, 1].remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Budget gap impact plot saved to {save_path}")


def plot_success_probability_comparison(configs: List[Dict[str, Any]], 
                                      save_path: str = "success_probability_comparison.png"):
    """
    Plot before/after comparison showing impact of elicitation bias on success probabilities.
    
    Args:
        configs: List of expanded configuration dictionaries  
        save_path: Path to save the plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Success Probability Impact: Before vs After Elicitation Bias', 
                 fontsize=16, fontweight='bold')
    
    # Create task difficulty range
    x = np.linspace(10, 30, 200)
    threshold = 20.0  # Mock ability threshold
    slope = -0.6     # Mock ability slope
    window_upper = 30.0
    
    # Calculate baseline success probabilities (logistic curve)
    baseline_probs = logistic_function(x, threshold, slope)
    
    # Select interesting configurations to compare
    bias_types = ["logistic_ability_shift", "task_filter"]
    colors = ['red', 'blue', 'green']
    
    for i, bias_type in enumerate(bias_types):
        if i >= 3:
            break
            
        ax = axes[i//2, i%2]
        ax.set_title(f'{bias_type.replace("_", " ").title()} Bias Impact', fontweight='bold')
        ax.set_xlabel('Task Difficulty')
        ax.set_ylabel('Success Probability')
        ax.grid(True, alpha=0.3)
        
        # Plot baseline curve
        ax.plot(x, baseline_probs, 'k--', linewidth=2, alpha=0.7, label='Original (No Bias)')
        
        # Find a good config for this bias type
        type_configs = [c for c in configs if c["elicitation_bias_config"]["bias_type"] == bias_type]
        
        # Plot a few different parameter settings
        for config in type_configs[:3]:  # Limit to 3 configs
            
            try:
                # Create evaluation config and scenario
                eval_config = EvaluationConfig(**config)
                scenario = create_mock_scenario(budget_fraction=0.5)  # 50% budget constraint
                
                # Get calculated bias
                bias = define_elicitation_bias(scenario, eval_config)
            except Exception as e:
                print(f"Error processing config for {bias_type}: {e}")
                continue
            
            # Apply bias to baseline probabilities
            biased_probs = apply_elicitation_bias(x, baseline_probs, bias, threshold, window_upper)
            
            # Create parameter label
            param_str = f"Params: {[f'{p:.2f}' for p in bias.args[:2]]}"
            label = f'With Bias ({param_str})'
            
            ax.plot(x, biased_probs, color=colors[i], linewidth=2, alpha=0.8, label=label)
            # break  # Just show one example per bias type
        
        # Add vertical line at ability threshold
        ax.axvline(x=threshold, color='black', linestyle='-', alpha=0.3, label='Ability Threshold')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    
    # Create a summary plot in the fourth subplot
    ax = axes[1, 1]
    ax.set_title('All Bias Types Comparison', fontweight='bold')
    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Success Probability')
    ax.grid(True, alpha=0.3)
    
    # Plot baseline
    ax.plot(x, baseline_probs, 'k--', linewidth=3, alpha=0.7, label='Original (No Bias)')
    
    # Plot one example of each bias type
    for i, bias_type in enumerate(bias_types):
        type_configs = [c for c in configs if c["elicitation_bias_config"]["bias_type"] == bias_type]
        if type_configs:
            try:
                config = type_configs[0]
                eval_config = EvaluationConfig(**config)
                scenario = create_mock_scenario(budget_fraction=0.5)
                bias = define_elicitation_bias(scenario, eval_config)
                biased_probs = apply_elicitation_bias(x, baseline_probs, bias, threshold, window_upper)
                
                ax.plot(x, biased_probs, color=colors[i], linewidth=2, 
                       label=f'{bias_type.replace("_", " ").title()} Bias')
            except Exception as e:
                print(f"Error processing summary plot for {bias_type}: {e}")
                continue
    
    ax.axvline(x=threshold, color='black', linestyle='-', alpha=0.3, label='Ability Threshold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Success probability comparison saved to {save_path}")


if __name__ == "__main__":
    print("Generating elicitation bias visualizations...")
    
    # Create and expand sweep configurations for each bias type
    print("Creating bias-type-specific sweep configurations...")
    
    logistic_ability_shift_configs = expand_sweep_config(create_logistic_ability_shift_sweep())
    task_filter_configs = expand_sweep_config(create_task_filter_sweep())
    budget_dependent_configs = expand_sweep_config(create_budget_dependent_sweep())
    
    # Create budget scaling focused configurations
    budget_scaling_configs = expand_sweep_config(create_budget_scaling_sweep())
    
    # Combine all bias configurations
    all_bias_configs = logistic_ability_shift_configs + task_filter_configs + budget_dependent_configs
    
    # Legacy budget gap configurations (kept for compatibility)
    budget_configs = expand_sweep_config(create_budget_gap_sweep_config())
    
    print(f"Generated {len(logistic_ability_shift_configs)} logistic_ability_shift configurations")
    print(f"Generated {len(task_filter_configs)} task_filter configurations")
    print(f"Generated {len(budget_dependent_configs)} budget_dependent configurations")
    print(f"Generated {len(budget_scaling_configs)} budget scaling configurations")
    print(f"Total: {len(all_bias_configs)} bias configurations")
    print(f"Generated {len(budget_configs)} legacy budget configurations")
    
    # Generate all plots
    print("1. Plotting elicitation bias curves...")
    plot_elicitation_bias_curves(all_bias_configs)
    
    print("2. Plotting budget gap impact (using dedicated scaling configs)...")
    plot_budget_gap_impact(budget_scaling_configs)
    
    print("3. Plotting success probability comparison...")
    plot_success_probability_comparison(all_bias_configs)
    
    print("All visualizations complete!")