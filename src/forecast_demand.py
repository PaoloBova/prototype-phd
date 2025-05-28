"""
Calculate evaluation windows under different resource constraints.
"""
import argparse
import logging
import math
import os
import pandas as pd
import numpy as np
import prototype_phd.data_utils as data_utils
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from .schemas import AbilityForecast, EvaluationWindow, ResourceConstraintType, ResourceConstraint

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate evaluation windows under resource constraints")
    parser.add_argument("--ability", required=True, help="Path to ability forecasts CSV")
    parser.add_argument("--cost", required=True, help="Path to cost trends CSV")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    return parser.parse_args()

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
    dates = pd.date_range(start=start_date, end=end_date, freq="Y")
    
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
    # δ = ln(0.9/0.1) ≈ 2.197
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

def calculate_adjusted_window(
    ability: AbilityForecast,
    doubling_rate: float,
    budget_fraction: float,
    repeats_per_unit: int = 20
) -> EvaluationWindow:
    """
    Calculate evaluation window parameters under given constraints.
    
    Args:
        ability: AbilityForecast object with threshold and slope
        doubling_rate: Cost doubling rate in difficulty units
        budget_fraction: Budget as fraction of gold standard
        repeats_per_unit: Number of repeats per difficulty unit
    
    Returns:
        EvaluationWindow object with window parameters
    """
    # Calculate evaluation window
    lower_bound, upper_bound = calculate_evaluation_window(
        ability.threshold, ability.slope
    )
    
    # Calculate mean cost over the window
    mean_cost = calculate_mean_cost(lower_bound, upper_bound, doubling_rate)
    
    # Calculate window width and total tasks for gold standard
    window_width = upper_bound - lower_bound
    total_samples = math.ceil(window_width) * repeats_per_unit
    
    # Calculate gold standard total cost
    gold_standard_cost = total_samples * mean_cost
    
    # Calculate available budget
    available_budget = gold_standard_cost * budget_fraction
    
    # If budget is sufficient, use full window
    if budget_fraction >= 1.0:
        adjusted_upper = upper_bound
    elif budget_fraction <= 0.0:
        # No budget means no window (set upper = lower)
        adjusted_upper = lower_bound
    else:
        # Solve for new upper bound that fits within budget
        # R = r·(2^(u_new/d) − 2^(b/d))·(d/ln 2)
        # ⇒ 2^(u_new/d) = (R·ln 2)/(r·d) + 2^(b/d)
        # ⇒ u_new = d·log2((R·ln 2)/(r·d) + 2^(b/d))
        term1 = (available_budget * np.log(2)) / (repeats_per_unit * doubling_rate)
        term2 = 2.0 ** (lower_bound / doubling_rate)
        adjusted_upper = doubling_rate * np.log2(term1 + term2)
    
    # Calculate adjusted parameters
    adjusted_width = max(0, adjusted_upper - lower_bound)
    adjusted_samples = int(repeats_per_unit * adjusted_width)
    
    # Limit upper bound to original upper bound
    adjusted_upper = min(adjusted_upper, upper_bound)
    
    return EvaluationWindow(
        ability_scenario=ability.scenario,
        budget_fraction=budget_fraction,
        budget_scenario=f"static_{int(budget_fraction*100)}pct",
        date=ability.date,
        eval_window_lower=lower_bound,
        eval_window_upper=adjusted_upper,
        total_samples=adjusted_samples,
        gold_standard_cost=gold_standard_cost,
        available_budget=available_budget,
        distribution_type="uniform"  # For now we use uniform distribution
    )

def discretize_task_allocation(window: EvaluationWindow) -> Dict[int, int]:
    """
    Convert continuous evaluation window to discrete task allocations.
    This is an optional utility function that can be used if discrete 
    allocations are needed for visualization or other purposes.
    
    Args:
        window: EvaluationWindow object
        
    Returns:
        Dictionary mapping difficulty to task count
    """
    if window.total_samples <= 0 or window.eval_window_upper <= window.eval_window_lower:
        return {}
        
    # Calculate repeats per unit based on total samples and window width
    window_width = window.eval_window_upper - window.eval_window_lower
    repeats_per_unit = window.total_samples / window_width
    
    # Discretize the window into integer difficulty units
    difficulties = np.arange(
        math.floor(window.eval_window_lower),
        math.ceil(window.eval_window_upper) + 1
    )
    
    # Initialize task allocations
    task_allocations = {}
    
    # Distribute tasks uniformly across difficulty units
    for diff in difficulties:
        # Skip if outside adjusted window
        if diff < window.eval_window_lower or diff > window.eval_window_upper:
            continue
        
        # Calculate how much of this difficulty unit is within the window
        lower_overlap = max(window.eval_window_lower, diff)
        upper_overlap = min(window.eval_window_upper, diff + 1)
        overlap = max(0, upper_overlap - lower_overlap)
        
        # Proportionally allocate tasks
        tasks_allocated = int(repeats_per_unit * overlap)
        if tasks_allocated > 0:
            task_allocations[int(diff)] = tasks_allocated
            
    return task_allocations

def calculate_windows_for_all_scenarios(
    abilities_df: pd.DataFrame, 
    costs_df: pd.DataFrame
) -> List[EvaluationWindow]:
    """
    Calculate evaluation windows for all ability forecasts under all resource scenarios.
    
    Args:
        abilities_df: DataFrame with ability forecasts
        costs_df: DataFrame with cost trends
    
    Returns:
        List of EvaluationWindow objects
    """
    scenarios = define_resource_scenarios()
    all_windows = []
    
    # Process only static scenarios for simplicity
    static_scenarios = [s for s in scenarios if s.type == ResourceConstraintType.STATIC]
    
    # Convert date strings to datetime objects
    abilities_df["date"] = pd.to_datetime(abilities_df["date"])
    
    # Process each ability forecast
    for _, row in abilities_df.iterrows():
        model = row["model"]
        
        # Use either model-specific cost trend or aggregate cost trend
        cost_row = costs_df[(costs_df["model"] == model) | (costs_df["model"] == "aggregate")]
        
        # Prefer model-specific cost if available, otherwise use aggregate
        if "aggregate" in cost_row["model"].values and len(cost_row) > 1:
            cost_row = cost_row[cost_row["model"] != "aggregate"]
            
        if len(cost_row) == 0:
            logging.warning(f"No suitable cost trend for model {model}, skipping")
            continue
        
        doubling_rate = cost_row.iloc[0]["doubling_rate"]
        
        # Create AbilityForecast object
        ability = AbilityForecast(
            date=row["date"],
            threshold=row["threshold"],
            slope=row["slope"],
            scenario=row["scenario"],
            model=model
        )
        
        # Calculate evaluation windows for each static resource scenario
        for scenario in static_scenarios:
            budget_fraction = float(scenario.values)
            
            window = calculate_adjusted_window(
                ability=ability,
                doubling_rate=doubling_rate,
                budget_fraction=budget_fraction
            )
            
            all_windows.append(window)
    
    return all_windows

def save_windows(windows: List[EvaluationWindow], output_path: str):
    """Save evaluation windows to CSV file."""
    # Convert to DataFrame
    records = [w.model_dump() for w in windows]
    df = pd.DataFrame(records)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Reading ability forecasts from {args.ability}")
    abilities_df = pd.read_csv(args.ability)
    
    logging.info(f"Reading cost trends from {args.cost}")
    costs_df = pd.read_csv(args.cost)
    
    logging.info("Calculating evaluation windows under resource constraints")
    windows = calculate_windows_for_all_scenarios(abilities_df, costs_df)
    logging.info(f"Calculated {len(windows)} evaluation windows")
    
    logging.info(f"Saving evaluation windows to {args.out}")
    save_windows(windows, args.out)
    logging.info("Complete")

if __name__ == "__main__":
    main()
