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
from prototype_phd.utils import expand_sweep_config, get_in
from .schemas import (
    AbilityForecast,
    WindowAdjustmentMethod, 
    ResourceConstraintType, 
    ResourceConstraint,
    EvaluationScenario,
    EvaluationForecast,
    EvaluationConfig,
    EvaluationDesign,
    CalculatedElicitationBias,
    generate_content_hash,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate evaluation forecasts under resource constraints")
    parser.add_argument("--ability", required=True, help="Path to ability forecasts CSV")
    parser.add_argument("--cost", required=True, help="Path to cost trends CSV")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    
    # Configuration options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Path to evaluation config JSON (legacy single-run mode)")
    config_group.add_argument("--sweep-config", help="Path to parameter sweep config JSON (new multi-run mode)")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()

def expand_ability_forecasts(abilities_df: pd.DataFrame, include_ci: bool = True) -> List[Dict[str, Any]]:
    """
    Expand ability forecasts to include confidence interval scenarios with explicit metadata.
    
    Args:
        abilities_df: DataFrame with ability forecasts including confidence intervals
        include_ci: Whether to include confidence interval scenarios
        
    Returns:
        List of dictionaries with forecast, variant_type, base_id, and variant_id
    """
    ability_variants = []
    
    # Process each original forecast
    for _, row in abilities_df.iterrows():
        # Ensure all fields are present
        required_fields = {'date', 'threshold', 'slope', 'scenario', 'model'}
        if not all(field in row for field in required_fields):
            logging.warning(f"Skipping row missing required fields: {row}")
            continue
            
        base_id = f"{row['model']}_{row['scenario']}"
        
        try:
            # Create base scenario from median values
            base_forecast = AbilityForecast(
                date=pd.to_datetime(row['date']),
                threshold=float(row['threshold']),
                slope=float(row['slope']),
                scenario=row['scenario'],
                model=row['model']
            )
            ability_variants.append({
                'forecast': base_forecast,
                'variant_type': 'base',
                'base_id': base_id,
                'variant_id': f"{base_id}_base"
            })
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
                lower_forecast = AbilityForecast(
                    date=pd.to_datetime(row['date']),
                    threshold=float(row['threshold_ci_upper']),  # Higher threshold = harder problems
                    slope=float(row['slope_ci_lower']),  # Flatter slope = less sensitive to difficulty
                    scenario=f"{row['scenario']}_lower_ci",
                    model=row['model']
                )
                ability_variants.append({
                    'forecast': lower_forecast,
                    'variant_type': 'lower',
                    'base_id': base_id,
                    'variant_id': f"{base_id}_lower"
                })
                
                # Upper bound scenario (more optimistic)
                upper_forecast = AbilityForecast(
                    date=pd.to_datetime(row['date']),
                    threshold=float(row['threshold_ci_lower']),  # Lower threshold = easier problems
                    slope=float(row['slope_ci_upper']),  # Steeper slope = more sensitive to difficulty
                    scenario=f"{row['scenario']}_upper_ci",
                    model=row['model']
                )
                ability_variants.append({
                    'forecast': upper_forecast,
                    'variant_type': 'upper',
                    'base_id': base_id,
                    'variant_id': f"{base_id}_upper"
                })
    
    return ability_variants

def expand_cost_trends(costs_df: pd.DataFrame, include_ci: bool = True) -> List[Dict[str, Any]]:
    """
    Expand cost trends to include confidence interval scenarios with explicit metadata.
    
    Args:
        costs_df: DataFrame with cost trends including confidence intervals
        include_ci: Whether to include confidence interval scenarios
        
    Returns:
        List of dictionaries with cost parameters, variant_type, base_id, and variant_id
    """
    cost_variants = []
    
    # Process each cost trend
    for _, row in costs_df.iterrows():
        model = row['model']
        base_id = model
        
        # Base scenario with median values
        base_cost_params = {
            'doubling_rate': float(row['doubling_rate']),
            'intercept': float(row['intercept']),
            'model': model,
        }
        cost_variants.append({
            'cost_params': base_cost_params,
            'variant_type': 'base',
            'base_id': base_id,
            'variant_id': f"{base_id}_base"
        })

        # Add confidence interval scenarios if configured and available
        if include_ci:
            has_ci = ('doubling_rate_ci_lower' in row and pd.notna(row['doubling_rate_ci_lower']) and
                     'doubling_rate_ci_upper' in row and pd.notna(row['doubling_rate_ci_upper']) and
                     np.isfinite(row['doubling_rate_ci_lower']) and np.isfinite(row['doubling_rate_ci_upper']))
            
            if has_ci:
                # Lower bound scenario (more expensive)
                lower_cost_params = {
                    'doubling_rate': float(row['doubling_rate_ci_lower']),  # Lower doubling rate = costs grow faster
                    'intercept': float(row['intercept']),
                    'model': f"{model}_lower_ci",
                }
                cost_variants.append({
                    'cost_params': lower_cost_params,
                    'variant_type': 'lower',
                    'base_id': base_id,
                    'variant_id': f"{base_id}_lower"
                })
                
                # Upper bound scenario (less expensive)
                upper_cost_params = {
                    'doubling_rate': float(row['doubling_rate_ci_upper']),  # Higher doubling rate = costs grow slower
                    'intercept': float(row['intercept']),
                    'model': f"{model}_upper_ci",
                }
                cost_variants.append({
                    'cost_params': upper_cost_params,
                    'variant_type': 'upper',
                    'base_id': base_id,
                    'variant_id': f"{base_id}_upper"
                })
    
    return cost_variants

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
    
    # Expand ability and cost forecasts based on config with explicit metadata
    ability_variants = expand_ability_forecasts(
        abilities_df, 
        include_ci=config.scenario_generation.get("include_ci_scenarios", True)
    )
    cost_variants = expand_cost_trends(
        costs_df,
        include_ci=config.scenario_generation.get("include_ci_scenarios", True)
    )
    
    # Generate all scenario combinations with explicit variant metadata
    scenarios = []
    
    for ability_variant in ability_variants:
        for cost_variant in cost_variants:
            for constraint in constraints:
                budget_fraction = float(constraint.values)
                
                # Create scenario with all fields including variant metadata
                scenario = EvaluationScenario(
                    ability=ability_variant['forecast'],
                    doubling_rate= 2 * cost_variant['cost_params']['doubling_rate'],
                    intercept=cost_variant['cost_params']['intercept'],
                    budget_fraction=budget_fraction,
                    ability_id=ability_variant['variant_id'],
                    cost_id=cost_variant['variant_id'],
                    constraint_id=constraint.name,
                    cost_model=cost_variant['cost_params']['model'],
                    ability_variant=ability_variant['variant_type'],
                    cost_variant=cost_variant['variant_type'],
                    base_ability_id=ability_variant['base_id'],
                    base_cost_id=cost_variant['base_id']
                )
                scenarios.append(scenario)
    
    return scenarios

def calculate_evaluation_window(threshold: float, slope: float, coverage_ratio: float=0.8) -> Tuple[float, float]:
    """
    Calculate evaluation window that covers coverage_ratio*100% of the logistic curve.
    
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
    
    # Example: For coverage_ratio = 0.8, we want to cover 80% of the curve
    # For logistic curve, we want to cover from p=0.1 to p=0.9
    # Using logit transformation: logit(p) = threshold + slope*difficulty
    # So difficulty = (logit(p) - threshold) / slope
    # logit(0.1) = ln(0.1/0.9) ≈ -2.2
    # logit(0.9) = ln(0.9/0.1) ≈ 2.2
    # delta = ln(0.9/0.1) = ln(9) ≈ 2.2
    upper_percentile = (1 + coverage_ratio) / 2
    lower_percentile = (1 - coverage_ratio) / 2
    delta = np.log(upper_percentile / lower_percentile)
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
    
    if lower_bound > upper_bound:
        logging.warning(f"Window bounds reversed: lower={lower_bound}, upper={upper_bound}.")

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
        # Linear interpolation: scaled_value = base_value + (target_value - base_value) * (1 - budget_gap)
        target_value = scaling_params.get("target_value", base_value)
        return base_value + (target_value - base_value) * (1 - budget_gap)
    
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
    elif scaling_type == "logarithmic":
        # Logarithmic scaling from base_value to target_value
        target_value = scaling_params.get("target_value", base_value)
        base = scaling_params.get("base", 2)  # Default to base 2
        scale_factor = scaling_params.get("scale_factor", 1.0)
        
        # For logarithmic interpolation: result = a + (b - a) * log(1 + k*x) / log(1 + k)
        # where x is the input parameter (1 - budget_gap in this case)
        # and k is the scale_factor that controls curvature
        k = scale_factor * (base - 1)  # Convert base to scale factor
        
        if k <= 0:
            # Fall back to linear interpolation if invalid scale factor
            return base_value + (target_value - base_value) * (1 - budget_gap)
        
        # Logarithmic interpolation formula - diminishing returns near budget_gap=0
        log_term = np.log2(1.0 + k * (1 - budget_gap)) / np.log2(1.0 + k)
        return base_value + (target_value - base_value) * log_term
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}. Must be one of: constant, linear, exponential, power_law, logistic, logarithmic")

def define_elicitation_bias(scenario: EvaluationScenario, config: EvaluationConfig) -> CalculatedElicitationBias:
    """
    Define elicitation bias parameters based on configuration and budget scenario.
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
    
    bias_config = config.elicitation_bias_config
    
    if not bias_config.enabled:
        # If bias is not enabled, return empty bias (disabled by default)
        return CalculatedElicitationBias()
    
    # Use the bias type and parameters from the schema directly
    bias_type = bias_config.bias_type.value
    parameters = bias_config.parameters.copy()
    budget_scaling = bias_config.budget_scaling
    
    # Validate bias type
    from .schemas import ElicitationBiasType
    valid_types = [e.value for e in ElicitationBiasType]
    if bias_type not in valid_types:
        raise ValueError(f"Invalid elicitation bias type: {bias_type}. Must be one of {valid_types}")
    
    if bias_type == "task_filter":
        # We need to ensure budget scaling has the correct target value based on the scenario
        _, upper_bound = calculate_evaluation_window(
            scenario.ability.threshold,
            scenario.ability.slope,
            coverage_ratio=config.coverage_ratio
        )
        # Alternatively, set target value to scenario.threshold instead.
        target_type = get_in(bias_config.budget_scaling,
                             ["elicitation_threshold", "params", "target_type"],
                             default="upper_bound")
        if target_type == "upper_bound":
            # Use upper bound as target value
            target_value = upper_bound
        elif target_type == "threshold":
            target_value = scenario.ability.threshold
        else:
            target_value = scenario.ability.threshold  # Default to threshold if unknown
        if "elicitation_threshold" in budget_scaling:
            if "params" not in budget_scaling["elicitation_threshold"]:
                budget_scaling["elicitation_threshold"]["params"] = {"target_value": target_value}
            else:
                budget_scaling["elicitation_threshold"]["params"]["target_value"] = target_value

    if bias_config.budget_dependent:
        # Calculate budget gap: 1.0 = no budget (100% gap), 0.0 = full budget (no gap)
        budget_gap = 1.0 - scenario.budget_fraction

        # Apply functional scaling to named parameters
        for param_name, base_value in parameters.items():
            if param_name in budget_scaling:
                param_config = budget_scaling[param_name]
                scaling_type = param_config.get("type", "constant")
                scaling_params = param_config.get("params", {})
                scaled_value = _apply_budget_scaling(base_value, budget_gap, scaling_type, scaling_params)
                parameters[param_name] = scaled_value
    
    # Extract final args based on bias type
    if bias_type == "fall_past_threshold":
        if "sensitivity_rate" not in parameters:
            raise ValueError(f"fall_past_threshold bias type requires 'sensitivity_rate' parameter")
        args = [parameters["sensitivity_rate"]]
        
    elif bias_type == "linear":
        if "threshold" not in parameters or "slope" not in parameters:
            raise ValueError(f"linear bias type requires 'threshold' and 'slope' parameters")
        args = [parameters["threshold"], parameters["slope"]]
        
    elif bias_type == "logistic":
        if "threshold" not in parameters or "slope" not in parameters:
            raise ValueError(f"logistic bias type requires 'threshold' and 'slope' parameters")
        args = [parameters["threshold"], parameters["slope"]]
        
    elif bias_type == "logistic_ability_shift":
        if "delta" not in parameters:
            raise ValueError(f"logistic_ability_shift bias type requires 'delta' parameter")
        
        # Use scenario ability parameters directly
        base_threshold = scenario.ability.threshold
        slope = scenario.ability.slope
        
        # Use delta parameter directly - budget scaling has already been applied above if enabled
        scaled_delta = parameters["delta"]
        
        args = [base_threshold, scaled_delta, slope]
        
    elif bias_type == "task_filter":
        if "elicitation_threshold" not in parameters or "sensitivity_rate_after" not in parameters:
            raise ValueError(f"task_filter bias type requires 'elicitation_threshold' and 'sensitivity_rate_after' parameters")
        args = [parameters["elicitation_threshold"], parameters["sensitivity_rate_after"]]
    
    return CalculatedElicitationBias(
        enabled=bias_config.enabled,
        bias_type=bias_type,
        name=bias_config.name,
        source_file=bias_config.source_file,
        parameters=bias_config.parameters,
        args=args
    )

def calculate_evaluation_forecast(
    scenario: EvaluationScenario,
    config: EvaluationConfig
) -> EvaluationForecast:
    """
    Calculate evaluation forecast with full design and bias/ability processing.
    
    Args:
        scenario: Evaluation scenario with ability, cost, and budget parameters
        config: Evaluation configuration with design parameters and bias/ability configs
        
    Returns:
        Complete EvaluationForecast object
    """
    # Calculate base evaluation window from ability parameters
    lower_bound, upper_bound = calculate_evaluation_window(
        scenario.ability.threshold,
        scenario.ability.slope,
        coverage_ratio=config.coverage_ratio
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
    total_samples = math.ceil(window_width) * config.repeats_per_unit
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
            if config.adjustment_method == WindowAdjustmentMethod.UPPER_BOUND:
                # Adjust only the upper bound
                if scenario.doubling_rate == 0:
                    # Avoid division by zero
                    adjusted_upper = lower_bound + (window_width * scenario.budget_fraction)
                else:
                    term1 = (available_budget * np.log(2)) / (config.repeats_per_unit * scenario.doubling_rate)
                    term2 = 2.0 ** (lower_bound / scenario.doubling_rate)
                    adjusted_upper = scenario.doubling_rate * np.log2(term1 + term2)
                
                # Ensure we don't exceed the original upper bound
                adjusted_upper = min(adjusted_upper, upper_bound)
                
            elif config.adjustment_method == WindowAdjustmentMethod.BOTH_BOUNDS:
                # Adjust both bounds to maintain the center point
                center_point = (upper_bound + lower_bound) / 2
                affordable_width = (window_width * scenario.budget_fraction)
                
                # Check if we're using cost-based scaling
                if mean_cost > 0:
                    affordable_width = (available_budget * np.log(2)) / (config.repeats_per_unit * mean_cost)
                
                half_width = min(affordable_width / 2, (upper_bound - lower_bound) / 2)
                adjusted_lower = center_point - half_width
                adjusted_upper = center_point + half_width
                
            # For SAMPLE_BASED method, we don't adjust the window but instead will
            # reduce the sampling density (handled later)
        except Exception as e:
            logging.error(f"Error adjusting window: {e}. Using original window.")
            # Fall back to original window or simple scaling
            if config.adjustment_method == WindowAdjustmentMethod.UPPER_BOUND:
                adjusted_upper = lower_bound + (window_width * scenario.budget_fraction)
    
    # Final safety check - ensure window is properly ordered
    if adjusted_lower > adjusted_upper:
        logging.warning(f"Adjusted window bounds reversed: lower={adjusted_lower}, "
                       f"upper={adjusted_upper}. Swapping values.")
        adjusted_lower, adjusted_upper = adjusted_upper, adjusted_lower
    
    # Calculate adjusted width and samples
    adjusted_width = max(0, adjusted_upper - adjusted_lower)
    adjusted_samples = int(config.repeats_per_unit * adjusted_width)
    
    # For SAMPLE_BASED method, scale the samples directly
    if config.adjustment_method == WindowAdjustmentMethod.SAMPLE_BASED and scenario.budget_fraction > 0:
        adjusted_samples = int(total_samples * scenario.budget_fraction)

    # Calculate elicitation bias and alternate ability using existing functions
    calculated_elicitation_bias = define_elicitation_bias(scenario, config)
    
    # Create EvaluationDesign with all calculated results
    design = EvaluationDesign(
        sampler_type=config.sampler_type,
        adjustment_method=config.adjustment_method,
        repeats_per_unit=config.repeats_per_unit,
        sampler_params=config.sampler_params,
        window_lower=adjusted_lower,
        window_upper=adjusted_upper,
        total_samples=adjusted_samples,
        gold_standard_cost=gold_standard_cost,
        available_budget=available_budget,
        original_window_lower=lower_bound,
        original_window_upper=upper_bound,
        elicitation_bias=calculated_elicitation_bias,
        alternate_ability=config.alternate_ability
    )
    
    # Generate hash-based IDs
    scenario_id = generate_content_hash(scenario.model_dump(), "scenario")
    design_id = generate_content_hash(design.model_dump(), "design")
    
    # Create and return complete EvaluationForecast
    return EvaluationForecast(
        scenario_id=scenario_id,
        design_id=design_id,
        scenario=scenario,
        design=design,
        config=config
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

def save_forecasts(forecasts: List[EvaluationForecast],
                   output_path: str):
    """
    Save evaluation forecasts to file with complete data fields.
    
    Args:
        forecasts: List of evaluation forecast objects
        output_path: Path to output file
        config: Evaluation configuration
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([f.model_dump() for f in forecasts], f, cls=data_utils.CombinedEncoder, indent=2)
    logging.info(f"Saved {len(forecasts)} forecast records to {output_path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    data_utils.configure_logging_console(level=log_level)
    
    # Load input data
    logging.info(f"Reading ability forecasts from {args.ability}")
    abilities_df = pd.read_csv(args.ability)
    
    logging.info(f"Reading cost trends from {args.cost}")
    costs_df = pd.read_csv(args.cost)
    
    # Load and expand configurations to list
    if args.sweep_config:
        logging.info("Running in parameter sweep mode")
        with open(args.sweep_config, 'r') as f:
            sweep_config_data = json.load(f)
        run_configs = expand_sweep_config(sweep_config_data)
        logging.info(f"Generated {len(run_configs)} individual run configurations")
    else:
        logging.info("Running in single configuration mode")
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        run_configs = [config_data]
    
    # Process all configurations
    forecasts = []
    for i, run_config_data in enumerate(run_configs):
        logging.debug(f"Processing configuration {i+1}/{len(run_configs)}")
        
        # Clean and create EvaluationConfig (remove _sweep_metadata)
        config_data = {k: v for k, v in run_config_data.items() if k != '_sweep_metadata'}
        config = EvaluationConfig(**config_data)
        
        # Generate scenarios for this config
        scenarios = generate_evaluation_scenarios(abilities_df, costs_df, config)
        logging.debug(f"Generated {len(scenarios)} scenarios for this configuration")
        
        # Process all scenarios
        for scenario in scenarios:
            try:
                forecast = calculate_evaluation_forecast(scenario, config)
                forecasts.append(forecast)
            except Exception as e:
                logging.error(f"Error calculating forecast for scenario {scenario.ability_id}_{scenario.cost_id}_{scenario.constraint_id}: {e}")
        
        if (i + 1) % 10 == 0 or len(run_configs) == 1:
            logging.info(f"Processed {i+1} configurations, generated {len(forecasts)} forecasts so far")
    
    logging.info(f"Calculated {len(forecasts)} total evaluation forecasts")
    
    # Save all results
    logging.info(f"Saving evaluation forecasts to {args.out}")
    save_forecasts(forecasts, args.out)
    
    # Provide summary statistics
    df = pd.DataFrame([f.model_dump() for f in forecasts])
    logging.info(f"Summary statistics:")
    logging.info(f"  Total forecasts: {len(df)}")
    
    if 'budget_scenario' in df.columns:
        logging.info(f"  Unique budget scenarios: {df['budget_scenario'].nunique()}")
    if 'ability_id' in df.columns:
        logging.info(f"  Number of ability models: {df['ability_id'].nunique()}")
    if 'cost_id' in df.columns:
        logging.info(f"  Number of cost models: {df['cost_id'].nunique()}")
    if 'constraint_id' in df.columns:
        logging.info(f"  Number of constraints: {df['constraint_id'].nunique()}")
    if 'design_id' in df.columns:
        logging.info(f"  Number of designs: {df['design_id'].nunique()}")
    
    logging.info("Complete")

if __name__ == "__main__":
    main()