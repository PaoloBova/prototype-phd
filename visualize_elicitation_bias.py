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


def create_fall_past_threshold_sweep() -> Dict[str, Any]:
    """
    Create sweep configuration for fall_past_threshold bias type.
    Focuses on varying sensitivity_rate parameter with consistent budget scaling.
    
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
            "bias_type": "fall_past_threshold",
            "name": "fall_past_threshold",
            "source_file": None,
            "parameters": {"sensitivity_rate": 1.0, "threshold": 0.0, "slope": 1.0},
            "budget_dependent": True,
            "budget_scaling": {
                "sensitivity_rate": {
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
            # Sweep over sensitivity_rate values - the key parameter for fall_past_threshold
            {
                "path": ["elicitation_bias_config", "parameters", "sensitivity_rate"],
                "values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            }
        ]
    }
    
    return sweep_config


def create_linear_bias_sweep() -> Dict[str, Any]:
    """
    Create sweep configuration for linear bias type.
    Focuses on varying threshold and slope parameters with consistent budget scaling.
    
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
            "bias_type": "linear",
            "name": "linear",
            "source_file": None,
            "parameters": {"sensitivity_rate": 0.5, "threshold": 0.0, "slope": 1.0},
            "budget_dependent": True,
            "budget_scaling": {
                "threshold": {
                    "type": "linear",
                    "params": {"target_value": 0.0}
                },
                "slope": {
                    "type": "linear", 
                    "params": {"target_value": 0.5}
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
            # Sweep over threshold values - key parameter for linear bias
            {
                "path": ["elicitation_bias_config", "parameters", "threshold"],
                "values": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
            },
            
            # Sweep over slope values - second key parameter for linear bias
            {
                "path": ["elicitation_bias_config", "parameters", "slope"],
                "values": [0.2, 0.5, 1.0, 1.5, 2.0]
            }
        ]
    }
    
    return sweep_config


def create_logistic_bias_sweep() -> Dict[str, Any]:
    """
    Create sweep configuration for logistic bias type.
    Focuses on varying threshold and slope parameters with consistent budget scaling.
    
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
            "bias_type": "logistic",
            "name": "logistic",
            "source_file": None,
            "parameters": {"sensitivity_rate": 0.5, "threshold": 0.0, "slope": 1.0},
            "budget_dependent": True,
            "budget_scaling": {
                "threshold": {
                    "type": "linear",
                    "params": {"target_value": 0.0}
                },
                "slope": {
                    "type": "linear",
                    "params": {"target_value": 0.5}
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
            # Sweep over threshold values - key parameter for logistic bias (inflection point)
            {
                "path": ["elicitation_bias_config", "parameters", "threshold"],
                "values": [10.0, 15.0, 20.0, 25.0, 30.0]
            },
            
            # Sweep over slope values - second key parameter for logistic bias (steepness)
            {
                "path": ["elicitation_bias_config", "parameters", "slope"],
                "values": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
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
            "bias_type": "fall_past_threshold",
            "name": "budget_dependent",
            "source_file": None,
            "parameters": {"sensitivity_rate": 1.0, "threshold": 0.0, "slope": 1.0},
            "budget_dependent": True,
            "budget_scaling": {
                "sensitivity_rate": {
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
                    # Linear scaling configurations
                    {"sensitivity_rate": {"type": "linear", "params": {"target_value": 0.0}}},
                    {"sensitivity_rate": {"type": "linear", "params": {"target_value": 0.5}}},
                    
                    # Exponential scaling configurations  
                    {"sensitivity_rate": {"type": "exponential", "params": {"decay_rate": 1.0}}},
                    {"sensitivity_rate": {"type": "exponential", "params": {"decay_rate": 2.0}}},
                    {"sensitivity_rate": {"type": "exponential", "params": {"decay_rate": 3.0}}},
                    
                    # Logistic scaling configurations
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.5, "steepness": 4.0, "min_value": 0.0, "max_value": 1.0}}},
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.25, "steepness": 6.0, "min_value": 0.0, "max_value": 1.0}}},
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.75, "steepness": 2.0, "min_value": 0.0, "max_value": 1.0}}}
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
            "bias_type": "fall_past_threshold",
            "name": "budget_scaling_study",
            "source_file": None,
            "parameters": {"sensitivity_rate": 1.0, "threshold": 0.0, "slope": 1.0},
            "budget_dependent": True,
            "budget_scaling": {
                "sensitivity_rate": {
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
                    # Linear scaling variations
                    {"sensitivity_rate": {"type": "linear", "params": {"target_value": 0.0}}},
                    {"sensitivity_rate": {"type": "linear", "params": {"target_value": 0.25}}},
                    {"sensitivity_rate": {"type": "linear", "params": {"target_value": 0.5}}},
                    {"sensitivity_rate": {"type": "linear", "params": {"target_value": 0.75}}},
                    
                    # Exponential scaling variations  
                    {"sensitivity_rate": {"type": "exponential", "params": {"decay_rate": 0.5}}},
                    {"sensitivity_rate": {"type": "exponential", "params": {"decay_rate": 1.0}}},
                    {"sensitivity_rate": {"type": "exponential", "params": {"decay_rate": 2.0}}},
                    {"sensitivity_rate": {"type": "exponential", "params": {"decay_rate": 3.0}}},
                    
                    # Logistic scaling variations
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.25, "steepness": 4.0, "min_value": 0.0, "max_value": 1.0}}},
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.5, "steepness": 4.0, "min_value": 0.0, "max_value": 1.0}}},
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.75, "steepness": 4.0, "min_value": 0.0, "max_value": 1.0}}},
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.5, "steepness": 2.0, "min_value": 0.0, "max_value": 1.0}}},
                    {"sensitivity_rate": {"type": "logistic", "params": {"midpoint": 0.5, "steepness": 6.0, "min_value": 0.0, "max_value": 1.0}}}
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
    if bias.bias_type == "fall_past_threshold":
        sensitivity_rate = bias.args[0] if len(bias.args) > 0 else 0.0
        keep_probs = np.where(task_difficulties <= threshold, 1.0, sensitivity_rate)
        
    elif bias.bias_type == "linear":
        elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
        elicitation_slope = bias.args[1] if len(bias.args) > 1 else 1.0
        keep_probs = np.clip(
            1 - elicitation_slope * (task_difficulties - elicitation_threshold) / (window_upper - elicitation_threshold), 
            0, 1
        )
        
    elif bias.bias_type == "logistic":
        elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
        elicitation_slope = bias.args[1] if len(bias.args) > 1 else 1.0
        keep_probs = logistic_function(task_difficulties, elicitation_threshold, elicitation_slope)
        
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
    bias_types = ["fall_past_threshold", "linear", "logistic"]
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
        for config in type_configs[:5]:  # Limit to 5 configs per type
            
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
            if bias_type == "fall_past_threshold":
                sensitivity_rate = bias.args[0] if len(bias.args) > 0 else 0.0
                y = np.where(x <= threshold, 1.0, sensitivity_rate)
                label = f'Rate: {sensitivity_rate:.2f}'
                
            elif bias_type == "linear":
                elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                elicitation_slope = bias.args[1] if len(bias.args) > 1 else 1.0
                y = np.clip(
                    1 - elicitation_slope * (x - elicitation_threshold) / (window_upper - elicitation_threshold),
                    0, 1
                )
                label = f'Thresh: {elicitation_threshold:.1f}, Slope: {elicitation_slope:.1f}'
                
            elif bias_type == "logistic":
                elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                elicitation_slope = bias.args[1] if len(bias.args) > 1 else 1.0
                y = logistic_function(x, elicitation_threshold, elicitation_slope)
                label = f'Thresh: {elicitation_threshold:.1f}, Slope: {elicitation_slope:.1f}'
            
            ax.plot(x, y, color=colors[color_idx], linewidth=2, label=label)
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
        if scaling_groups[scaling_type]:
            type_config = scaling_groups[scaling_type][0]
        else:
            # Create a minimal default config for this scaling type
            type_config = {
                "resource_constraints": {"static_budgets": [1.0, 0.75, 0.5, 0.25, 0.0]},
                "scenario_generation": {"include_base_scenarios": True, "include_ci_scenarios": False},
                "sampler_type": "uniform",
                "adjustment_method": "upper_bound",
                "repeats_per_unit": 20,
                "sampler_params": {},
                "elicitation_bias_config": {
                    "enabled": True,
                    "bias_type": "fall_past_threshold",
                    "parameters": {"sensitivity_rate": 1.0, "threshold": 0.0, "slope": 1.0},
                    "budget_dependent": True,
                    "budget_scaling": {
                        "sensitivity_rate": {
                            "type": scaling_type,
                            "params": {"target_value": 0.0} if scaling_type == "linear" else 
                                     {"decay_rate": 2.0} if scaling_type == "exponential" else
                                     {"midpoint": 0.5, "steepness": 4.0, "min_value": 0.0, "max_value": 1.0}
                        }
                    }
                },
                "alternate_ability": {"enabled": False, "function_type": "logistic", "args": {}}
            }
        
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
            if bias.bias_type == "fall_past_threshold":
                sensitivity_rate = bias.args[0] if len(bias.args) > 0 else 0.0
                y = np.where(x <= threshold, 1.0, sensitivity_rate)
                
            elif bias.bias_type == "linear":
                elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                elicitation_slope = bias.args[1] if len(bias.args) > 1 else 1.0
                y = np.clip(
                    1 - elicitation_slope * (x - elicitation_threshold) / (window_upper - elicitation_threshold),
                    0, 1
                )
                
            elif bias.bias_type == "logistic":
                elicitation_threshold = bias.args[0] if len(bias.args) > 0 else 0.0
                elicitation_slope = bias.args[1] if len(bias.args) > 1 else 1.0
                y = logistic_function(x, elicitation_threshold, elicitation_slope)
            
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
    bias_types = ["fall_past_threshold", "linear", "logistic"]
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
    
    fall_past_configs = expand_sweep_config(create_fall_past_threshold_sweep())
    linear_configs = expand_sweep_config(create_linear_bias_sweep())
    logistic_configs = expand_sweep_config(create_logistic_bias_sweep())
    
    # Create budget scaling focused configurations
    budget_scaling_configs = expand_sweep_config(create_budget_scaling_sweep())
    
    # Combine all bias configurations
    all_bias_configs = fall_past_configs + linear_configs + logistic_configs
    
    # Legacy budget gap configurations (kept for compatibility)
    budget_configs = expand_sweep_config(create_budget_gap_sweep_config())
    
    print(f"Generated {len(fall_past_configs)} fall_past_threshold configurations")
    print(f"Generated {len(linear_configs)} linear configurations")
    print(f"Generated {len(logistic_configs)} logistic configurations")
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