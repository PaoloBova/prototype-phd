"""
Calculate evaluation forecasts under different resource constraints.
"""
import argparse
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
    EvaluationForecast,
    WindowAdjustmentMethod, 
    ResourceConstraintType, 
    ResourceConstraint
)

class EvaluationDesign(BaseModel):
    """Parameters defining how evaluations are designed and sampled."""
    sampler_type: TaskSamplerType = Field(TaskSamplerType.UNIFORM, description="Type of task distribution")
    adjustment_method: WindowAdjustmentMethod = Field(
        WindowAdjustmentMethod.UPPER_BOUND, 
        description="Method used to adjust window based on budget"
    )
    repeats_per_unit: int = Field(20, description="Number of sample repeats per difficulty unit")
    sampler_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the sampler")

class EvaluationScenario(BaseModel):
    """A scenario for evaluation combining ability, cost, and resource constraints."""
    ability: AbilityForecast = Field(..., description="Ability forecast for this scenario")
    doubling_rate: float = Field(..., description="Cost doubling rate in difficulty units")
    budget_fraction: float = Field(..., description="Budget as fraction of gold standard")
    scenario_id: str = Field(..., description="Unique identifier for this scenario")
    
    class Config:
        arbitrary_types_allowed = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate evaluation forecasts under resource constraints")
    parser.add_argument("--ability", required=True, help="Path to ability forecasts CSV")
    parser.add_argument("--cost", required=True, help="Path to cost trends CSV")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    parser.add_argument("--config", required=False, help="Path to evaluation config JSON")
    return parser.parse_args()

def expand_ability_forecasts(abilities_df: pd.DataFrame) -> Dict[str, AbilityForecast]:
    """
    Expand ability forecasts to include confidence interval scenarios.
    
    Args:
        abilities_df: DataFrame with ability forecasts including confidence intervals
        
    Returns:
        Dictionary mapping scenario IDs to AbilityForecast objects
    """
    expanded_forecasts = {}
    
    # Process each original forecast
    for _, row in abilities_df.iterrows():
        # Create base scenario from median values
        base_scenario_id = f"{row['model']}_{row['scenario']}_base"
        base_forecast = AbilityForecast(
            date=pd.to_datetime(row['date']),
            threshold=float(row['threshold']),
            slope=float(row['slope']),
            scenario=row['scenario'],
            model=row['model']
        )
        expanded_forecasts[base_scenario_id] = base_forecast
        
        # Create lower bound scenario if confidence intervals are available
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

def expand_cost_trends(costs_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Expand cost trends to include confidence interval scenarios.
    
    Args:
        costs_df: DataFrame with cost trends including confidence intervals
        
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
            'model': model
        }
        expanded_costs[base_scenario_id] = base_cost
        
        # Check if confidence intervals are available
        has_ci = ('doubling_rate_ci_lower' in row and pd.notna(row['doubling_rate_ci_lower']) and
                 'doubling_rate_ci_upper' in row and pd.notna(row['doubling_rate_ci_upper']) and
                 np.isfinite(row['doubling_rate_ci_lower']) and np.isfinite(row['doubling_rate_ci_upper']))
        
        if has_ci:
            # Lower bound scenario (more expensive)
            lower_scenario_id = f"{model}_lower"
            lower_cost = {
                'doubling_rate': float(row['doubling_rate_ci_lower']),  # Lower doubling rate = costs grow faster
                'intercept': float(row['intercept']),
                'model': f"{model}_lower_ci"
            }
            expanded_costs[lower_scenario_id] = lower_cost
            
            # Upper bound scenario (less expensive)
            upper_scenario_id = f"{model}_upper"
            upper_cost = {
                'doubling_rate': float(row['doubling_rate_ci_upper']),  # Higher doubling rate = costs grow slower
                'intercept': float(row['intercept']),
                'model': f"{model}_upper_ci"
            }
            expanded_costs[upper_scenario_id] = upper_cost
    
    return expanded_costs

def define_resource_scenarios() -> List[ResourceConstraint]:
    """Define resource constraint scenarios."""
    scenarios = []
    
    # Static budget scenarios
    budget_fractions = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    for fraction in budget_fractions:
        scenario = ResourceConstraint(
            type=ResourceConstraintType.STATIC,
            name=f"static_{int(fraction*100)}pct",
            values=fraction
        )
        scenarios.append(scenario)
    
    # Dynamic budget scenarios
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2030, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="YE")
    
    # Linear decline over 5 years
    linear_values = [(d, 1.0 - (i / (len(dates) - 1)) * 0.9) for i, d in enumerate(dates)]
    scenarios.append(ResourceConstraint(
        type=ResourceConstraintType.DYNAMIC,
        name="linear_decline",
        values=linear_values
    ))
    
    # One-step decline (100→10→1)
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
    
    # Rapid drop then plateau
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
    delta = np.log(0.9 / 0.1)
    scale = 1.0 / slope
    
    lower_bound = threshold - (scale * delta)
    upper_bound = threshold + (scale * delta)
    
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
    # E[c] = (2^(u/d) − 2^(b/d)) · (d / W) / ln 2
    window_width = upper_bound - lower_bound
    if window_width <= 0:
        return 2.0 ** (lower_bound / doubling_rate)
    
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

def calculate_evaluation_forecast(
    scenario: EvaluationScenario,
    design: EvaluationDesign
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
    
    # Calculate costs based on window and doubling rate
    mean_cost = calculate_mean_cost(lower_bound, upper_bound, scenario.doubling_rate)
    
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
        # No budget - no window
        adjusted_upper = lower_bound
    else:
        # Partial budget - adjust according to method
        if design.adjustment_method == WindowAdjustmentMethod.UPPER_BOUND:
            # Adjust only the upper bound
            term1 = (available_budget * np.log(2)) / (design.repeats_per_unit * scenario.doubling_rate)
            term2 = 2.0 ** (lower_bound / scenario.doubling_rate)
            adjusted_upper = scenario.doubling_rate * np.log2(term1 + term2)
            # Ensure we don't exceed the original upper bound
            adjusted_upper = min(adjusted_upper, upper_bound)
            
        elif design.adjustment_method == WindowAdjustmentMethod.BOTH_BOUNDS:
            # Adjust both bounds to maintain the center point
            center_point = (upper_bound + lower_bound) / 2
            affordable_width = (available_budget * np.log(2)) / (design.repeats_per_unit * mean_cost)
            half_width = min(affordable_width / 2, (upper_bound - lower_bound) / 2)
            adjusted_lower = center_point - half_width
            adjusted_upper = center_point + half_width
            
        # For SAMPLE_BASED method, we don't adjust the window but instead will
        # reduce the sampling density (handled in sample generation)
    
    # Calculate adjusted width and samples
    adjusted_width = max(0, adjusted_upper - adjusted_lower)
    adjusted_samples = int(design.repeats_per_unit * adjusted_width)
    
    # For SAMPLE_BASED method, scale the samples directly
    if design.adjustment_method == WindowAdjustmentMethod.SAMPLE_BASED and scenario.budget_fraction > 0:
        adjusted_samples = int(total_samples * scenario.budget_fraction)
    
    return EvaluationForecast(
        ability=scenario.ability,
        budget_fraction=scenario.budget_fraction,
        budget_scenario=scenario.scenario_id,
        window_lower=adjusted_lower,
        window_upper=adjusted_upper,
        sampler_type=design.sampler_type,
        total_samples=adjusted_samples,
        gold_standard_cost=gold_standard_cost,
        available_budget=available_budget,
        adjustment_method=design.adjustment_method
    )

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
    costs_df: pd.DataFrame
) -> List[EvaluationScenario]:
    """
    Generate all combinations of evaluation scenarios.
    
    Args:
        abilities_df: DataFrame with ability forecasts
        costs_df: DataFrame with cost trends
    
    Returns:
        List of EvaluationScenario objects
    """
    # Convert date columns to datetime if needed
    if 'date' in abilities_df.columns:
        abilities_df["date"] = pd.to_datetime(abilities_df["date"])
    
    # Get resource constraints (we'll use static ones for now)
    constraints = define_resource_scenarios()
    static_constraints = [s for s in constraints if s.type == ResourceConstraintType.STATIC]
    
    # Expand ability and cost forecasts to include confidence interval scenarios
    expanded_abilities = expand_ability_forecasts(abilities_df)
    expanded_costs = expand_cost_trends(costs_df)
    
    # Filter for base cost scenarios
    cost_models = [model for model in expanded_costs.keys() if model.endswith("_base")]
    
    # Generate all scenario combinations
    scenarios = []
    
    for ability_id, ability_forecast in expanded_abilities.items():
        for cost_id in cost_models:
            cost_params = expanded_costs[cost_id]
            doubling_rate = cost_params['doubling_rate']
            
            for constraint in static_constraints:
                budget_fraction = float(constraint.values)
                
                # Create unique scenario ID
                scenario_id = f"{ability_id}_{cost_id}_{constraint.name}"
                
                # Create scenario
                scenario = EvaluationScenario(
                    ability=ability_forecast,
                    doubling_rate=doubling_rate,
                    budget_fraction=budget_fraction,
                    scenario_id=scenario_id
                )
                
                scenarios.append(scenario)
    
    return scenarios

def calculate_forecasts_for_all_combinations(
    abilities_df: pd.DataFrame, 
    costs_df: pd.DataFrame
) -> List[EvaluationForecast]:
    """
    Calculate evaluation forecasts for all combinations of ability forecasts and cost trends.
    
    Args:
        abilities_df: DataFrame with ability forecasts
        costs_df: DataFrame with cost trends
    
    Returns:
        List of EvaluationForecast objects
    """
    # Generate all evaluation scenarios
    scenarios = generate_evaluation_scenarios(abilities_df, costs_df)
    
    # Define evaluation designs to try
    designs = [
        EvaluationDesign(
            sampler_type=sampler_type,
            adjustment_method=adjustment_method
        )
        for sampler_type in [TaskSamplerType.UNIFORM, TaskSamplerType.NORMAL]
        for adjustment_method in [WindowAdjustmentMethod.UPPER_BOUND, WindowAdjustmentMethod.SAMPLE_BASED]
    ]
    
    # Calculate forecasts for each scenario and design combination
    all_forecasts = []
    
    for scenario in scenarios:
        for design in designs:
            forecast = calculate_evaluation_forecast(scenario, design)
            all_forecasts.append(forecast)
    
    return all_forecasts

def save_forecasts(forecasts: List[EvaluationForecast], output_path: str):
    """Save evaluation forecasts to CSV file."""
    # Flatten nested objects for CSV format
    flat_records = []
    for forecast in forecasts:
        record = forecast.model_dump()
        ability = record.pop("ability")
        # Flatten ability attributes with ability_ prefix
        for key, value in ability.items():
            record[f"ability_{key}"] = value
        flat_records.append(record)
    
    df = pd.DataFrame(flat_records)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Reading ability forecasts from {args.ability}")
    abilities_df = pd.read_csv(args.ability)
    
    logging.info(f"Reading cost trends from {args.cost}")
    costs_df = pd.read_csv(args.cost)
    
    logging.info("Calculating evaluation forecasts for all combinations")
    forecasts = calculate_forecasts_for_all_combinations(abilities_df, costs_df)
    logging.info(f"Calculated {len(forecasts)} evaluation forecasts")
    
    logging.info(f"Saving evaluation forecasts to {args.out}")
    save_forecasts(forecasts, args.out)
    logging.info("Complete")

if __name__ == "__main__":
    main()