"""
Calculate evaluation forecasts under different resource constraints.
"""
import argparse
import json
import logging
import math
import os
import pandas as pd
import numpy as np
import prototype_phd.data_utils as data_utils
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from pydantic import BaseModel, Field
from .schemas import (
    AbilityForecast, 
    TaskSamplerType, 
    TaskSampler, 
    UniformTaskSampler, 
    NormalTaskSampler,
    WindowAdjustmentMethod, 
    ResourceConstraintType, 
    ResourceConstraint,
    EvaluationScenario,
    EvaluationDesign,
    EvaluationForecast,
    ElicitationBiasConfig,
    AlternateAbilityConfig
)

class EvaluationConfig(BaseModel):
    """Configuration for evaluation forecasts."""
    resource_constraints: Dict[str, Any] = Field(
        {
            "static_budgets": [1.0, 0.75, 0.5, 0.25, 0.1, 0.0],
            "include_dynamic_scenarios": False,
            "dynamic_start_date": "2025-01-01T00:00:00",
            "dynamic_end_date": "2030-12-31T00:00:00",
            "dynamic_frequency": "YE"
        },
        description="Resource constraint parameters"
    )
    evaluation_design: Dict[str, Any] = Field(
        {
            "sampler_types": ["uniform", "normal"],
            "adjustment_methods": ["upper_bound", "sample_based"],
            "repeats_per_unit": 20,
            "sampler_params": {
                "normal": {
                    "mean_offset": 0.0,
                    "std_dev_factor": 0.3
                }
            }
        }, 
        description="Evaluation design parameters"
    )
    scenario_generation: Dict[str, bool] = Field(
        {
            "include_base_scenarios": True,
            "include_ci_scenarios": True
        },
        description="Scenario generation options"
    )
    save_detailed_json: bool = Field(
        False,
        description="Whether to save detailed JSON output with all forecast data"
    )
    elicitation_bias: ElicitationBiasConfig = Field(
        default_factory=ElicitationBiasConfig,
        description="Configuration for elicitation bias parameters"
    )
    alternate_ability: AlternateAbilityConfig = Field(
        default_factory=AlternateAbilityConfig,
        description="Configuration for alternate ability function parameters"
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate evaluation forecasts under resource constraints")
    parser.add_argument("--ability", required=True, help="Path to ability forecasts CSV")
    parser.add_argument("--cost", required=True, help="Path to cost trends CSV")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    parser.add_argument("--config", required=False, help="Path to evaluation config JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()

def expand_ability_forecasts(abilities_df: pd.DataFrame, include_ci: bool = True) -> Dict[str, AbilityForecast]:
    """
    Expand ability forecasts to include confidence interval scenarios.
    
    Args:
        abilities_df: DataFrame with ability forecasts including confidence intervals
        include_ci: Whether to include confidence interval scenarios
        
    Returns:
        Dictionary mapping scenario IDs to AbilityForecast objects
    """
    expanded_forecasts = {}
    
    # Process each original forecast
    for _, row in abilities_df.iterrows():
        # Create base scenario from median values
        base_scenario_id = f"{row['model']}_{row['scenario']}_base"
        # Ensure all fields are present
        required_fields = {'date', 'threshold', 'slope', 'scenario', 'model'}
        if not all(field in row for field in required_fields):
            logging.warning(f"Skipping row missing required fields: {row}")
            continue
            
        try:
            base_forecast = AbilityForecast(
                date=pd.to_datetime(row['date']),
                threshold=float(row['threshold']),
                slope=float(row['slope']),
                scenario=row['scenario'],
                model=row['model']
            )
            expanded_forecasts[base_scenario_id] = base_forecast
        except Exception as e:
            logging.error(f"Error creating base forecast: {e}, row: {row}")
            continue
        
        # Create lower/upper bound scenarios if configured and confidence intervals are available
        if include_ci:
            has_threshold_ci = ('threshold_ci_lower' in row and pd.notna(row['threshold_ci_lower']) and 
                               'threshold_ci_upper' in row and pd.notna(row['threshold_ci_upper']))
            has_slope_ci = ('slope_ci_lower' in row and pd.notna(row['slope_ci_lower']) and 
                           'slope_ci_upper' in row and pd.notna(row['slope_ci_upper']))
            
            if has_threshold_ci and has_slope_ci:
                # Lower bound scenario (more pessimistic)
                lower_scenario_id = f"{row['model']}_{row['scenario']}_lower"
                lower_forecast = AbilityForecast(
                    date=pd.to_datetime(row['date']),
                    threshold=float(row['threshold_ci_upper']),  # Higher threshold = harder problems
                    slope=float(row['slope_ci_lower']),  # Flatter slope = less sensitive to difficulty
                    scenario=f"{row['scenario']}_lower_ci",
                    model=row['model']
                )
                expanded_forecasts[lower_scenario_id] = lower_forecast
                
                # Upper bound scenario (more optimistic)
                upper_scenario_id = f"{row['model']}_{row['scenario']}_upper"
                upper_forecast = AbilityForecast(
                    date=pd.to_datetime(row['date']),
                    threshold=float(row['threshold_ci_lower']),  # Lower threshold = easier problems
                    slope=float(row['slope_ci_upper']),  # Steeper slope = more sensitive to difficulty
                    scenario=f"{row['scenario']}_upper_ci",
                    model=row['model']
                )
                expanded_forecasts[upper_scenario_id] = upper_forecast
    
    return expanded_forecasts

def expand_cost_trends(costs_df: pd.DataFrame, include_ci: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Expand cost trends to include confidence interval scenarios.
    
    Args:
        costs_df: DataFrame with cost trends including confidence intervals
        include_ci: Whether to include confidence interval scenarios
        
    Returns:
        Dictionary mapping scenario IDs to cost parameters
    """
    expanded_costs = {}
    
    # Process each cost trend
    for _, row in costs_df.iterrows():
        model = row['model']
        
        # Base scenario with median values
        base_scenario_id = f"{model}_base"
        base_cost = {
            'doubling_rate': float(row['doubling_rate']),
            'intercept': float(row['intercept']),
            'model': model,
        }
        expanded_costs[base_scenario_id] = base_cost

        # Add confidence interval scenarios if configured and available
        if include_ci:
            has_ci = ('doubling_rate_ci_lower' in row and pd.notna(row['doubling_rate_ci_lower']) and
                     'doubling_rate_ci_upper' in row and pd.notna(row['doubling_rate_ci_upper']) and
                     np.isfinite(row['doubling_rate_ci_lower']) and np.isfinite(row['doubling_rate_ci_upper']))
            
            if has_ci:
                # Lower bound scenario (more expensive)
                lower_scenario_id = f"{model}_lower"
                lower_cost = {
                    'doubling_rate': float(row['doubling_rate_ci_lower']),  # Lower doubling rate = costs grow faster
                    'intercept': float(row['intercept']),
                    'model': f"{model}_lower_ci",
                }
                expanded_costs[lower_scenario_id] = lower_cost
                
                # Upper bound scenario (less expensive)
                upper_scenario_id = f"{model}_upper"
                upper_cost = {
                    'doubling_rate': float(row['doubling_rate_ci_upper']),  # Higher doubling rate = costs grow slower
                    'intercept': float(row['intercept']),
                    'model': f"{model}_upper_ci",
                }
                expanded_costs[upper_scenario_id] = upper_cost
    
    return expanded_costs

def define_resource_scenarios(config: EvaluationConfig) -> List[ResourceConstraint]:
    """
    Define resource constraint scenarios based on configuration.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        List of ResourceConstraint objects
    """
    scenarios = []
    
    # Static budget scenarios from config
    budget_fractions = config.resource_constraints["static_budgets"]
    for fraction in budget_fractions:
        scenario = ResourceConstraint(
            type=ResourceConstraintType.STATIC,
            name=f"static_{int(fraction*100)}pct",
            values=fraction
        )
        scenarios.append(scenario)

    # Add dynamic scenarios if configured
    if config.resource_constraints.get("include_dynamic_scenarios", False):
        # Parse dates
        start_date = datetime.fromisoformat(
            config.resource_constraints.get("dynamic_start_date", "2025-01-01T00:00:00").replace('Z', '+00:00')
        )
        end_date = datetime.fromisoformat(
            config.resource_constraints.get("dynamic_end_date", "2030-12-31T00:00:00").replace('Z', '+00:00')
        )
        frequency = config.resource_constraints.get("dynamic_frequency", "YE")
        
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Linear decline over time period
        linear_values = [(d, 1.0 - (i / (len(dates) - 1)) * 0.9) for i, d in enumerate(dates)]
        scenarios.append(ResourceConstraint(
            type=ResourceConstraintType.DYNAMIC,
            name="linear_decline",
            values=linear_values
        ))
        
        # One-step decline scenario
        step_values = []
        for i, d in enumerate(dates):
            if i < 2:
                step_values.append((d, 1.0))
            elif i < 4:
                step_values.append((d, 0.1))
            else:
                step_values.append((d, 0.01))
        scenarios.append(ResourceConstraint(
            type=ResourceConstraintType.DYNAMIC,
            name="step_decline",
            values=step_values
        ))
        
        # Rapid drop then plateau scenario
        plateau_values = []
        for i, d in enumerate(dates):
            if i == 0:
                plateau_values.append((d, 1.0))
            elif i == 1:
                plateau_values.append((d, 0.3))
            else:
                plateau_values.append((d, 0.1))
        scenarios.append(ResourceConstraint(
            type=ResourceConstraintType.DYNAMIC,
            name="plateau_decline",
            values=plateau_values
        ))
    
    return scenarios

def calculate_evaluation_window(threshold: float, slope: float) -> Tuple[float, float]:
    """
    Calculate evaluation window that covers ~80% of the logistic curve.
    
    Args:
        threshold: The 50% threshold parameter
        slope: The slope parameter
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not np.isfinite(threshold) or not np.isfinite(slope) or slope == 0:
        logging.warning(f"Invalid parameters for window calculation: threshold={threshold}, slope={slope}")
        # Return reasonable defaults to avoid crashes
        return (0.0, 10.0)
        
    # Slope should be negative for our logistic model
    abs_slope = abs(slope)
    sign = -1 if slope < 0 else 1
    
    # For logistic curve, we want to cover from p=0.1 to p=0.9
    # Using logit transformation: logit(p) = threshold + slope*difficulty
    # So difficulty = (logit(p) - threshold) / slope
    # logit(0.1) = ln(0.1/0.9) ≈ -2.2
    # logit(0.9) = ln(0.9/0.1) ≈ 2.2
    
    delta = np.log(9)  # ln(0.9/0.1) = ln(9) ≈ 2.2
    scale = 1.0 / abs_slope
    
    # Lower difficulty corresponds to lower performance (p=0.1)
    # Higher difficulty corresponds to higher performance (p=0.9)
    lower_bound = threshold - sign * scale * delta
    upper_bound = threshold + sign * scale * delta
    
    # Ensure lower bound is actually lower than upper bound
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
        
    logging.debug(f"Window calculation: threshold={threshold}, slope={slope}, "
                  f"result: lower={lower_bound}, upper={upper_bound}")
    
    return lower_bound, upper_bound

def calculate_mean_cost(lower_bound: float, upper_bound: float, doubling_rate: float) -> float:
    """
    Calculate the mean cost over an evaluation window.
    
    Args:
        lower_bound: Lower bound of the evaluation window
        upper_bound: Upper bound of the evaluation window
        doubling_rate: Cost doubling rate
    
    Returns:
        Mean cost per task
    """
    # Ensure bounds are correctly ordered
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
        
    # E[c] = (2^(u/d) − 2^(l/d)) · (d / W) / ln 2
    window_width = upper_bound - lower_bound
    if window_width <= 0:
        return 2.0 ** (lower_bound / doubling_rate)
    
    if doubling_rate == 0:
        logging.warning("Doubling rate is zero, using default value")
        doubling_rate = 1.0
        
    term1 = 2.0 ** (upper_bound / doubling_rate)
    term2 = 2.0 ** (lower_bound / doubling_rate)
    
    mean_cost = (term1 - term2) * (doubling_rate / window_width) / np.log(2)
    return mean_cost

def create_task_sampler(sampler_type: TaskSamplerType = TaskSamplerType.UNIFORM, **kwargs) -> TaskSampler:
    """
    Create a task sampler of the specified type.
    
    Args:
        sampler_type: Type of sampler to create
        **kwargs: Additional parameters for the specific sampler type
        
    Returns:
        TaskSampler instance
    """
    if sampler_type == TaskSamplerType.UNIFORM:
        return UniformTaskSampler(sampler_type=sampler_type)
    elif sampler_type == TaskSamplerType.NORMAL:
        return NormalTaskSampler(
            sampler_type=sampler_type,
            mean_offset=kwargs.get("mean_offset", 0.0),
            std_dev_factor=kwargs.get("std_dev_factor", 0.3)
        )
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def _apply_budget_scaling(base_value: float, budget_gap: float, scaling_type: str, scaling_params: Dict[str, float]) -> float:
    """
    Apply budget-dependent scaling to a parameter value.
    
    Args:
        base_value: The base parameter value (when budget_gap = 0)
        budget_gap: Budget gap from 0.0 (full budget) to 1.0 (no budget)
        scaling_type: Type of scaling function ("constant", "linear", "exponential", "power_law", "logistic")
        scaling_params: Named parameters for the scaling function
    
    Returns:
        The scaled parameter value
    """
    if scaling_type == "constant":
        return base_value
    
    elif scaling_type == "linear":
        # Linear interpolation: scaled_value = base_value + (target_value - base_value) * budget_gap
        target_value = scaling_params.get("target_value", base_value)
        return base_value + (target_value - base_value) * budget_gap
    
    elif scaling_type == "exponential":
        # Exponential decay: scaled_value = base_value * exp(-decay_rate * budget_gap)
        decay_rate = scaling_params.get("decay_rate", 1.0)
        import math
        return base_value * math.exp(-decay_rate * budget_gap)
    
    elif scaling_type == "power_law":
        # Power law scaling: scaled_value = base_value * (1 - budget_gap)^exponent
        exponent = scaling_params.get("exponent", 1.0)
        return base_value * ((1.0 - budget_gap) ** exponent)
    
    elif scaling_type == "logistic":
        # Logistic scaling: uses logistic function centered around midpoint
        midpoint = scaling_params.get("midpoint", 0.5)
        steepness = scaling_params.get("steepness", 4.0)
        min_value = scaling_params.get("min_value", 0.0)
        max_value = scaling_params.get("max_value", base_value)
        
        import math
        # Logistic function: L / (1 + exp(-k(x - x0)))
        logistic_val = 1.0 / (1.0 + math.exp(-steepness * (budget_gap - midpoint)))
        # Scale between min and max values
        return min_value + (max_value - min_value) * (1.0 - logistic_val)
    
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}. Must be one of: constant, linear, exponential, power_law, logistic")


def define_elicitation_bias(config: EvaluationConfig, scenario: EvaluationScenario) -> Dict[str, Any]:
    """
    Define elicitation bias parameters based on configuration and budget scenario.
    
    Args:
        config: Evaluation configuration
        scenario: Evaluation scenario containing budget and cost information
        
    Returns:
        Dictionary with elicitation bias parameters
    """
    # Placeholder for actual elicitation bias logic
    
    # Recall that the elicitation bias if a function of the budget fraction,
    # and the ability parameters.
    
    # Past the ability threshold, the sensitivity rate of our test falls.
    # For now, we assume it falls linearly until zero (or until the upper
    # window is reached).
    # The slope of this line is defined by the
    # elicitation_slope parameter which is itself a function of the
    # budget fraction.
    
    # When there is no gap between the resources needed for a gold standard
    # evaluation and the resources available, the slope is 0 such that the
    # the sensitivity rate is always 1.
    # As soon as we have a gap, the slope decreases linearly until it reaches
    # the minimum elicitation slope.
    
    # Ideally, we'd have some way of calibrating this.
    # We know from RE-bench public results that they experimened on two different
    # scaffolds for Claude 3.5 Sonnet (AIDE and MODULAR), where MODULAR achieved
    # an average normalized score 0.2 points higher than AIDE when given between
    # 2 and 8hrs to complete the tasks. 3 iteration stages with many specifications
    # and implementations thrown out. Assume 50% of researcher resources used to
    # create the best scaffold.
    # To finish the calibration, we need to translate the shift in normalized score
    # to a shift in success rates. So, consider that success means scoring above
    # the reference solution which is 1. And assume that scores are normally distributed
    # across tasks with given mean (0.2 and 0.4 respectively for 8hrs with standard
    # errors which are roughly 0.05 in width on both sides) and standard deviation
    # unknown. We will assume it is 0.5 for now, so that we have non-zero success
    # rates.
    # From here, we can compute the success rates for both scaffolds on 8hr tasks.
    # Success rate for AIDE: 0.0548. Success rate for MODULAR: 0.1151.
    # In other words, the sensitivity rate fell from 1 to 0.5.
    # We will treat the 8hr mark for when these experiments were conducted as the
    # upper bound of the evaluation window. We compute the sensitivity at this
    # point, then assume a linear interpolation from the threshold to the upper
    # evaluation window that matches this.
    # Moreover, we also linearly interpolate due to the budget fraction. If
    # the gap in resources is less than 50%, then we linearly interpolate the
    # fall in sensitivity at 8hr mark, and recompute the slope. If the gap
    # in resources is above 50%, then we linearly extrapolate the fall in
    # sensitivity, not letting the sensitivity fall below 0.
    
    # In practise, what we would ideally do, is measure the success rates with
    # various levels of scaffolding across the different task difficulties. We
    # could do this for models with different release dataes. Then, we could
    # fit a functional form to the elicitation bias or sensitivity rate. We
    # would not be constrained to a linear function either.
    
    # For now, we will do something even simpler and just assume that the
    # sensitivity rate is constant past the ability threhsold and depends
    # linearly on the resource gap.
    
    if not config.elicitation_bias.enabled:
        return {
            "elicitation_bias_enabled": False,
            "elicitation_bias_type": None,
            "elicitation_bias_args": None,
        }
    
    # Handle file-based configuration
    if isinstance(config.elicitation_bias.source, str):
        try:
            with open(config.elicitation_bias.source, 'r') as f:
                source_config = json.load(f)
                
            # Validate required fields for file-based config
            if "type" not in source_config:
                raise ValueError(f"Elicitation bias config missing required 'type' field in {config.elicitation_bias.source}")
            if "args" not in source_config:
                raise ValueError(f"Elicitation bias config missing required 'args' field in {config.elicitation_bias.source}")
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load elicitation bias config from {config.elicitation_bias.source}: {e}")
            # Fall back to default
            source_config = {"type": "fall_past_threshold", "args": [0.5]}
    else:
        source_config = config.elicitation_bias.source
    
    # Validate bias type
    from .schemas import ElicitationBiasType
    valid_types = [e.value for e in ElicitationBiasType]
    bias_type = source_config.get("type")
    if bias_type not in valid_types:
        raise ValueError(f"Invalid elicitation bias type: {bias_type}. Must be one of {valid_types}")
    
    # Handle budget-dependent bias scaling
    args = source_config.get("args", [])
    
    if source_config.get("budget_dependent", False):
        # Calculate budget gap: 1.0 = no budget (100% gap), 0.0 = full budget (no gap)
        budget_gap = 1.0 - scenario.budget_fraction
        
        # Get budget scaling configuration
        budget_scaling = source_config.get("budget_scaling", {})
        
        # Apply functional scaling to each parameter
        scaled_args = []
        for i, base_value in enumerate(args):
            param_name = f"param_{i}"
            if param_name in budget_scaling:
                param_config = budget_scaling[param_name]
                scaling_type = param_config.get("type", "constant")
                scaling_params = param_config.get("params", {})
                
                scaled_value = _apply_budget_scaling(base_value, budget_gap, scaling_type, scaling_params)
                scaled_args.append(scaled_value)
            else:
                # No scaling for this parameter, use base value
                scaled_args.append(base_value)
        
        args = scaled_args
    
    # Validate argument counts for each bias type
    if bias_type == "fall_past_threshold" and len(args) != 1:
        raise ValueError(f"fall_past_threshold bias type requires exactly 1 argument, got {len(args)}")
    elif bias_type == "linear" and len(args) != 2:
        raise ValueError(f"linear bias type requires exactly 2 arguments, got {len(args)}")
    elif bias_type == "logistic" and len(args) != 2:
        raise ValueError(f"logistic bias type requires exactly 2 arguments, got {len(args)}")
    
    return {
        "elicitation_bias_enabled": True,
        "elicitation_bias_type": bias_type,
        "elicitation_bias_args": args,
    }

def define_alternate_ability_params(config: EvaluationConfig) -> Dict[str, Any]:
    """
    Define alternate ability function parameters based on configuration.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Dictionary with alternate ability parameters
    """
    if not config.alternate_ability.enabled:
        return {
            "alternate_ability_enabled": False,
            "alternate_ability_type": None,
            "alternate_ability_args": None,
        }
    
    # Validate function type
    from .schemas import AlternateAbilityType
    valid_types = [e.value for e in AlternateAbilityType]
    if config.alternate_ability.function_type not in valid_types:
        raise ValueError(f"Invalid alternate ability function type: {config.alternate_ability.function_type}. Must be one of {valid_types}")
    
    return {
        "alternate_ability_enabled": True,
        "alternate_ability_type": config.alternate_ability.function_type,
        "alternate_ability_args": config.alternate_ability.parameters,
    }

def calculate_evaluation_forecast(
    scenario: EvaluationScenario,
    design: EvaluationDesign,
    config: EvaluationConfig
) -> EvaluationForecast:
    """
    Calculate evaluation forecast parameters under given constraints.
    
    Args:
        scenario: Evaluation scenario with ability, cost, and budget parameters
        design: Evaluation design parameters (sampling, adjustment method, etc.)
        
    Returns:
        EvaluationForecast object with forecast parameters
    """
    # Calculate base evaluation window from ability parameters
    lower_bound, upper_bound = calculate_evaluation_window(
        scenario.ability.threshold, scenario.ability.slope
    )
    
    # Safety check: ensure window bounds are finite and properly ordered
    if not (np.isfinite(lower_bound) and np.isfinite(upper_bound)):
        logging.warning(f"Non-finite window bounds: {lower_bound}, {upper_bound}. "
                       f"Using default values.")
        lower_bound, upper_bound = 0.0, 10.0
        
    if lower_bound > upper_bound:
        logging.warning(f"Window bounds reversed: lower={lower_bound}, upper={upper_bound}. "
                       f"Swapping values.")
        lower_bound, upper_bound = upper_bound, lower_bound
    
    # Calculate costs based on window and doubling rate
    try:
        mean_cost = calculate_mean_cost(lower_bound, upper_bound, scenario.doubling_rate)
    except Exception as e:
        logging.error(f"Error calculating mean cost: {e}. Using default value.")
        mean_cost = 1.0
    
    # Calculate total samples and cost for gold standard
    window_width = upper_bound - lower_bound
    total_samples = math.ceil(window_width) * design.repeats_per_unit
    gold_standard_cost = total_samples * mean_cost
    
    # Calculate available budget
    available_budget = gold_standard_cost * scenario.budget_fraction
    
    # Default to original bounds
    adjusted_lower = lower_bound
    adjusted_upper = upper_bound
    
    # Adjust window based on budget constraints and selected method
    if scenario.budget_fraction >= 1.0:
        # Full budget - use full window
        pass
    elif scenario.budget_fraction <= 0.0:
        # No budget - no window (collapse to lower bound)
        adjusted_upper = lower_bound
    else:
        try:
            # Partial budget - adjust according to method
            if design.adjustment_method == WindowAdjustmentMethod.UPPER_BOUND:
                # Adjust only the upper bound
                if scenario.doubling_rate == 0:
                    # Avoid division by zero
                    adjusted_upper = lower_bound + (window_width * scenario.budget_fraction)
                else:
                    term1 = (available_budget * np.log(2)) / (design.repeats_per_unit * scenario.doubling_rate)
                    term2 = 2.0 ** (lower_bound / scenario.doubling_rate)
                    adjusted_upper = scenario.doubling_rate * np.log2(term1 + term2)
                
                # Ensure we don't exceed the original upper bound
                adjusted_upper = min(adjusted_upper, upper_bound)
                
            elif design.adjustment_method == WindowAdjustmentMethod.BOTH_BOUNDS:
                # Adjust both bounds to maintain the center point
                center_point = (upper_bound + lower_bound) / 2
                affordable_width = (window_width * scenario.budget_fraction)
                
                # Check if we're using cost-based scaling
                if mean_cost > 0:
                    affordable_width = (available_budget * np.log(2)) / (design.repeats_per_unit * mean_cost)
                
                half_width = min(affordable_width / 2, (upper_bound - lower_bound) / 2)
                adjusted_lower = center_point - half_width
                adjusted_upper = center_point + half_width
                
            # For SAMPLE_BASED method, we don't adjust the window but instead will
            # reduce the sampling density (handled later)
        except Exception as e:
            logging.error(f"Error adjusting window: {e}. Using original window.")
            # Fall back to original window or simple scaling
            if design.adjustment_method == WindowAdjustmentMethod.UPPER_BOUND:
                adjusted_upper = lower_bound + (window_width * scenario.budget_fraction)
    
    # Final safety check - ensure window is properly ordered
    if adjusted_lower > adjusted_upper:
        logging.warning(f"Adjusted window bounds reversed: lower={adjusted_lower}, "
                       f"upper={adjusted_upper}. Swapping values.")
        adjusted_lower, adjusted_upper = adjusted_upper, adjusted_lower
    
    # Calculate adjusted width and samples
    adjusted_width = max(0, adjusted_upper - adjusted_lower)
    adjusted_samples = int(design.repeats_per_unit * adjusted_width)
    
    # For SAMPLE_BASED method, scale the samples directly
    if design.adjustment_method == WindowAdjustmentMethod.SAMPLE_BASED and scenario.budget_fraction > 0:
        adjusted_samples = int(total_samples * scenario.budget_fraction)
    
    design_id = f"{design.sampler_type.value}_{design.adjustment_method.value}_repeats_{design.repeats_per_unit}"
    
    # Extract ability and cost variants from their IDs
    ability_variant = "unknown"
    base_ability_id = ""
    if scenario.ability_id.endswith("_base"):
        ability_variant = "base"
        base_ability_id = scenario.ability_id[:-5]  # Remove "_base" suffix
    elif scenario.ability_id.endswith("_lower"):
        ability_variant = "lower"
        base_ability_id = scenario.ability_id[:-6]  # Remove "_lower" suffix
    elif scenario.ability_id.endswith("_upper"):
        ability_variant = "upper"
        base_ability_id = scenario.ability_id[:-6]  # Remove "_upper" suffix
    
    cost_variant = "unknown"
    base_cost_id = ""
    if scenario.cost_id.endswith("_base"):
        cost_variant = "base"
        base_cost_id = scenario.cost_id[:-5]  # Remove "_base" suffix
    elif scenario.cost_id.endswith("_lower"):
        cost_variant = "lower"
        base_cost_id = scenario.cost_id[:-6]  # Remove "_lower" suffix
    elif scenario.cost_id.endswith("_upper"):
        cost_variant = "upper"
        base_cost_id = scenario.cost_id[:-6]  # Remove "_upper" suffix
    
    # Determine elicitation bias parameters
    elicitation_params = define_elicitation_bias(config, scenario)
    
    # Determine if we are using an alternative function to represent the
    # true ability, i.e. even though we forecast a logistic curve, what if
    # the true ability is instead a different function.
    alternate_ability_params = define_alternate_ability_params(config)

    # Create the evaluation forecast with all parameters for complete tracking
    forecast_data = {
        "ability": scenario.ability,
        "budget_fraction": scenario.budget_fraction,
        "budget_scenario": scenario.scenario_id,
        "window_lower": adjusted_lower,
        "window_upper": adjusted_upper,
        "sampler_type": design.sampler_type,
        "total_samples": adjusted_samples,
        "gold_standard_cost": gold_standard_cost,
        "available_budget": available_budget,
        "adjustment_method": design.adjustment_method,
        # Include original window for reference
        "original_window_lower": lower_bound,
        "original_window_upper": upper_bound,
        # Include design parameters
        "repeats_per_unit": design.repeats_per_unit,
        # Include cost model info
        "cost_model": scenario.cost_model,
        "doubling_rate": scenario.doubling_rate,
        # Include IDs for tracking
        "ability_id": scenario.ability_id,
        "cost_id": scenario.cost_id,
        "constraint_id": scenario.constraint_id,
        "design_id": design_id,
        # Add variant information
        "ability_variant": ability_variant,
        "cost_variant": cost_variant,
        "base_ability_id": base_ability_id,
        "base_cost_id": base_cost_id
    }
    
    # Add additional parameters if available
    forecast_data = {**forecast_data,
                     **alternate_ability_params,
                     **elicitation_params}
    
    return EvaluationForecast(**forecast_data)

def generate_task_samples(forecast: EvaluationForecast) -> np.ndarray:
    """
    Generate task difficulty samples for a given forecast.
    
    Args:
        forecast: EvaluationForecast object
        
    Returns:
        Array of task difficulty samples
    """
    if forecast.total_samples <= 0:
        return np.array([])
    
    sampler = create_task_sampler(forecast.sampler_type)
    
    return sampler.sample(
        forecast.total_samples, 
        forecast.window_lower, 
        forecast.window_upper
    )

def discretize_task_allocation(forecast: EvaluationForecast) -> Dict[int, int]:
    """
    Convert continuous evaluation window to discrete task allocations.
    This is an optional utility function that can be used if discrete 
    allocations are needed for visualization or other purposes.
    
    Args:
        forecast: EvaluationForecast object
        
    Returns:
        Dictionary mapping difficulty to task count
    """
    if forecast.total_samples <= 0 or forecast.window_upper <= forecast.window_lower:
        return {}
        
    # Calculate repeats per unit based on total samples and window width
    window_width = forecast.window_upper - forecast.window_lower
    repeats_per_unit = forecast.total_samples / window_width
    
    # Discretize the window into integer difficulty units
    difficulties = np.arange(
        math.floor(forecast.window_lower),
        math.ceil(forecast.window_upper) + 1
    )
    
    # Initialize task allocations
    task_allocations = {}
    
    # Distribute tasks uniformly across difficulty units
    for diff in difficulties:
        # Skip if outside adjusted window
        if diff < forecast.window_lower or diff > forecast.window_upper:
            continue
        
        # Calculate how much of this difficulty unit is within the window
        lower_overlap = max(forecast.window_lower, diff)
        upper_overlap = min(forecast.window_upper, diff + 1)
        overlap = max(0, upper_overlap - lower_overlap)
        
        # Proportionally allocate tasks
        tasks_allocated = int(repeats_per_unit * overlap)
        if tasks_allocated > 0:
            task_allocations[int(diff)] = tasks_allocated
            
    return task_allocations

def generate_evaluation_scenarios(
    abilities_df: pd.DataFrame, 
    costs_df: pd.DataFrame,
    config: EvaluationConfig
) -> List[EvaluationScenario]:
    """
    Generate all combinations of evaluation scenarios.
    
    Args:
        abilities_df: DataFrame with ability forecasts
        costs_df: DataFrame with cost trends
        config: Evaluation configuration
    
    Returns:
        List of EvaluationScenario objects
    """
    # Convert date columns to datetime if needed
    if 'date' in abilities_df.columns:
        abilities_df["date"] = pd.to_datetime(abilities_df["date"])
    
    # Get resource constraints from config
    constraints = define_resource_scenarios(config)
    
    # Expand ability and cost forecasts based on config
    expanded_abilities = expand_ability_forecasts(
        abilities_df, 
        include_ci=config.scenario_generation.get("include_ci_scenarios", True)
    )
    expanded_costs = expand_cost_trends(
        costs_df,
        include_ci=config.scenario_generation.get("include_ci_scenarios", True)
    )
    # Filter for base cost scenarios
    cost_models = [model for model in expanded_costs.keys()]
    
    # Generate all scenario combinations
    scenarios = []
    
    for ability_id, ability_forecast in expanded_abilities.items():
        for cost_id in cost_models:
            cost_params = expanded_costs[cost_id]
            doubling_rate = cost_params['doubling_rate']
            
            for constraint in constraints:
                budget_fraction = float(constraint.values)
                
                # Create unique scenario ID
                scenario_id = f"{ability_id}_{cost_id}_{constraint.name}"
                
                # Create scenario
                scenario = EvaluationScenario(
                    ability=ability_forecast,
                    doubling_rate=doubling_rate,
                    budget_fraction=budget_fraction,
                    scenario_id=scenario_id,
                    ability_id=ability_id,
                    cost_id=cost_id,
                    constraint_id=constraint.name,
                    cost_model=cost_params['model']
                )
                scenarios.append(scenario)
    
    return scenarios

def calculate_forecasts_for_all_combinations(
    abilities_df: pd.DataFrame, 
    costs_df: pd.DataFrame,
    config: EvaluationConfig
) -> List[EvaluationForecast]:
    """
    Calculate evaluation forecasts for all combinations of ability forecasts and cost trends.
    
    Args:
        abilities_df: DataFrame with ability forecasts
        costs_df: DataFrame with cost trends
        config: Evaluation configuration
    
    Returns:
        List of EvaluationForecast objects
    """
    # Generate all evaluation scenarios
    scenarios = generate_evaluation_scenarios(abilities_df, costs_df, config)
    logging.info(f"Generated {len(scenarios)} evaluation scenarios")
    
    # Define evaluation designs based on config
    sampler_types = [
        TaskSamplerType(s) for s in 
        config.evaluation_design.get("sampler_types", ["uniform", "normal"])
    ]
    
    adjustment_methods = [
        WindowAdjustmentMethod(m) for m in 
        config.evaluation_design.get("adjustment_methods", ["upper_bound", "sample_based"])
    ]
    
    logging.info(f"Using sampler types: {sampler_types}")
    logging.info(f"Using adjustment methods: {adjustment_methods}")
    
    repeats_per_unit = config.evaluation_design.get("repeats_per_unit", 20)
    
    # Create all evaluation design combinations
    designs = [
        EvaluationDesign(
            sampler_type=sampler_type,
            adjustment_method=adjustment_method,
            repeats_per_unit=repeats_per_unit,
            sampler_params=config.evaluation_design.get("sampler_params", {}).get(sampler_type.value, {})
        )
        for sampler_type in sampler_types
        for adjustment_method in adjustment_methods
    ]
    
    # Calculate forecasts for each scenario and design combination
    all_forecasts = []
    
    for i, scenario in enumerate(scenarios):
        for design in designs:
            try:
                forecast = calculate_evaluation_forecast(scenario, design, config)
                all_forecasts.append(forecast)
                
                # Log progress periodically
                if i % 100 == 0:
                    logging.debug(f"Processed {i} scenarios out of {len(scenarios)}")
                    
            except Exception as e:
                logging.error(f"Error calculating forecast for scenario {scenario.scenario_id}: {e}")
    
    return all_forecasts

def extract_unnested_dict(record:dict, key:str) -> Dict[str, Any]:
    """Extract an unnested dictionary and flatten."""
    if isinstance(record, dict) and key in record:
        record = record.copy()
        for k, v in record[key].items():
            record[f"{key}_{k}"] = v
        record.pop(key)
    return record
        
def save_forecasts(forecasts: List[EvaluationForecast],
                   output_path: str,
                   config: EvaluationConfig):
    """
    Save evaluation forecasts to CSV file with complete data fields.
    
    Args:
        forecasts: List of evaluation forecast objects
        output_path: Path to output CSV file
        config: Evaluation configuration
    """
    # Flatten nested objects for CSV format
    flat_records = []
    for forecast in forecasts:
        # Convert the forecast to a dictionary
        record = forecast.model_dump()
        # Extract simple nested objects and flatten them
        record = extract_unnested_dict(record, "ability")
        record = extract_unnested_dict(record, "elicitation_bias_args")
        record = extract_unnested_dict(record, "alternative_ability_args")
        flat_records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(flat_records)
    
    # Add sanity check columns
    df["window_width"] = df["window_upper"] - df["window_lower"]
    df["original_window_width"] = df["original_window_upper"] - df["original_window_lower"]
    df["width_ratio"] = df["window_width"] / df["original_window_width"]
    
    # Sort columns for better readability
    priority_cols = [
        "budget_scenario", "ability_id", "ability_model", "ability_scenario", "cost_model", "budget_fraction",
        "ability_threshold", "ability_slope", "constraint_id", "cost_id", "doubling_rate",
        "window_lower", "window_upper", "window_width",
        "original_window_lower", "original_window_upper", "original_window_width",
        "adjustment_method", "sampler_type"
    ]
    
    # Reorder columns, putting priority columns first
    existing_cols = set(df.columns)
    col_order = [col for col in priority_cols if col in existing_cols]
    col_order.extend([col for col in df.columns if col not in col_order])
    df = df[col_order]
    
    # If specified, include diagnostic data
    if config.save_detailed_json:
        # Save a detailed JSON with all data
        json_path = output_path.replace('.csv', '_detailed.json')
        with open(json_path, 'w') as f:
            json.dump([forecast.model_dump() for forecast in forecasts], f, 
                     default=str, indent=2)
        logging.info(f"Saved detailed forecast data to {json_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df)} forecast records to {output_path}")
    
    # Report any potential issues
    n_invalid_width = sum(df["window_width"] <= 0)
    if n_invalid_width > 0:
        logging.warning(f"Found {n_invalid_width} records with invalid window width (≤0)")
        
    n_reversed = sum(df["window_lower"] > df["window_upper"])
    if n_reversed > 0:
        logging.warning(f"Found {n_reversed} records with reversed window bounds")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration if provided, otherwise use defaults
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = EvaluationConfig(**config_data)
    else:
        config = EvaluationConfig()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    data_utils.configure_logging_console(level=log_level)
    
        
    logging.info(f"Using configuration: {config.model_dump_json(indent=2)}")

    logging.info(f"Reading ability forecasts from {args.ability}")
    abilities_df = pd.read_csv(args.ability)
    
    logging.info(f"Reading cost trends from {args.cost}")
    costs_df = pd.read_csv(args.cost)
    
    logging.info("Calculating evaluation forecasts for all combinations")
    forecasts = calculate_forecasts_for_all_combinations(abilities_df, costs_df, config)
    logging.info(f"Calculated {len(forecasts)} evaluation forecasts")
    
    logging.info(f"Saving evaluation forecasts to {args.out}")
    save_forecasts(forecasts, args.out, config)
    
    # Provide summary statistics
    df = pd.DataFrame([f.model_dump() for f in forecasts])
    logging.info(f"Summary statistics:")
    logging.info(f"  Total forecasts: {len(df)}")
    logging.info(f"  Unique forecasts: {df['budget_scenario'].nunique()}")
    logging.info(f"  Number of ability models: {df['ability_id'].nunique()}")
    logging.info(f"  Number of cost models: {df['cost_id'].nunique()}")
    logging.info(f"  Number of constraints: {df['constraint_id'].nunique()}")
    logging.info(f"  Number of designs: {df['design_id'].nunique()}")
    
    logging.info("Complete")

if __name__ == "__main__":
    main()