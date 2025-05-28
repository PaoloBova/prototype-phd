"""
Simulations for evaluation forecasting.
"""
import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import prototype_phd.data_utils as data_utils
from datetime import datetime
import tqdm
from typing import Dict, List, Optional, Tuple, Any, Callable

from .schemas import (
    EvaluationForecast, 
    AbilityForecast,
    SensitivityResult
)
from .simulation import (
    SimulationConfig, SimulationMethod, CorrelationModel,
    run_simulation, calculate_simulation_statistics,
    threshold_estimator, weighted_score_estimator
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate sensitivity metrics")
    parser.add_argument("--in", dest="input", required=True, help="Path to evaluation forecasts CSV")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    parser.add_argument("--config", default="configs/forecast_sensitivity.json", 
                        help="Path to sensitivity config file")
    return parser.parse_args()

def logistic_function(x: np.ndarray, threshold: float, slope: float) -> np.ndarray:
    """
    Calculate logistic function values.
    
    Args:
        x: Input values
        threshold: Threshold parameter (inflection point)
        slope: Slope parameter (steepness)
    
    Returns:
        Array of logistic function values
    """
    return 1.0 / (1.0 + np.exp(-slope * (x - threshold)))

def create_evaluation_forecast_from_row(row: pd.Series) -> EvaluationForecast:
    """
    Create an EvaluationForecast object from a DataFrame row.
    
    Args:
        row: Row from the evaluation forecasts DataFrame
        
    Returns:
        EvaluationForecast object
    """
    # Extract ability forecast parameters from flattened columns
    ability = AbilityForecast(
        date=row["ability_date"],
        threshold=row["ability_threshold"],
        slope=row["ability_slope"],
        scenario=row["ability_scenario"],
        model=row["ability_model"]
    )
    
    # Create evaluation forecast object
    forecast = EvaluationForecast(
        ability=ability,
        budget_fraction=row["budget_fraction"],
        budget_scenario=row["budget_scenario"],
        window_lower=row["window_lower"],
        window_upper=row["window_upper"],
        sampler_type=row["sampler_type"],
        total_samples=int(row["total_samples"]),
        gold_standard_cost=row["gold_standard_cost"],
        available_budget=row["available_budget"],
        adjustment_method=row["adjustment_method"]
    )
    
    return forecast

def calculate_true_weighted_score(threshold: float, slope: float, 
                                 range_min: float = -5, range_max: float = 20,
                                 weight_fn: Callable[[float], float] = lambda x: 1.0 + 0.5 * x) -> float:
    """
    Calculate the true weighted score for a given logistic curve.
    
    Integrates the product of the logistic function and the weight function
    over the difficulty range, then normalizes by the integral of weights.
    
    Args:
        threshold: Threshold parameter of logistic curve
        slope: Slope parameter of logistic curve
        range_min: Minimum difficulty to consider
        range_max: Maximum difficulty to consider
        weight_fn: Function mapping difficulty to weight
    
    Returns:
        True weighted score
    """
    # Create a fine grid of difficulties
    diff_grid = np.linspace(range_min, range_max, 1000)
    
    # Calculate success probabilities
    probs = logistic_function(diff_grid, threshold, slope)
    
    # Calculate weights
    weights = np.array([weight_fn(x) for x in diff_grid])
    
    # Calculate weighted average
    weighted_score = np.sum(probs * weights) / np.sum(weights)
    
    return weighted_score

def simulate_estimator(
    forecast: EvaluationForecast,
    simulation_config: SimulationConfig,
    estimator: str
) -> Dict[str, float]:
    """
    Simulate an estimator using the specified configuration.
    
    Args:
        forecast: Evaluation forecast to simulate
        simulation_config: Simulation configuration
        estimator: Which estimator to use ("threshold" or "weighted_score")
        
    Returns:
        Dictionary with simulation statistics
    """
    # Set up the analysis function based on the specified estimator
    if estimator == "threshold":
        analysis_fn = threshold_estimator
        # Calculate true value for comparison
        true_value = forecast.ability.threshold    
    elif estimator == "weighted_score":
        analysis_fn = lambda tasks, outcomes: weighted_score_estimator(
            tasks, outcomes, lambda x: 1.0 + 0.5 * x
        )
        # Calculate true value for comparison
        true_value = calculate_true_weighted_score(
            forecast.ability.threshold, forecast.ability.slope
        )
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")
    
    # Run the simulation
    results = run_simulation(
        config=simulation_config,
        n_tasks=forecast.total_samples,
        window_lower=forecast.window_lower,
        window_upper=forecast.window_upper,
        threshold=forecast.ability.threshold,
        slope=forecast.ability.slope,
        sampler_type=forecast.sampler_type,
        analysis_fn=analysis_fn
    )
    
    # Calculate statistics from the simulation results
    stats = calculate_simulation_statistics(results)

    # Add additional metrics
    stats["results_count"] = len(results)
    stats["estimator"] = estimator
    stats["true_value"] = true_value if not np.isnan(true_value) else np.nan
    stats["bias"] = stats["mean"] - true_value if not np.isnan(stats["mean"]) else np.nan
    stats["variance"] = stats["std"] ** 2 if not np.isnan(stats["std"]) else np.nan
    stats["contains_true"] = (
        stats["lower_ci"] <= true_value <= stats["upper_ci"]
        if not np.isnan(stats["lower_ci"]) else False
    )
    
    return stats

def calculate_sensitivity_metrics(
    forecasts_df: pd.DataFrame,
    config_path: str
) -> List[SensitivityResult]:
    """
    Calculate sensitivity metrics for each evaluation forecast using simulation.
    
    Args:
        forecasts_df: DataFrame with evaluation forecasts
        config_path: Path to sensitivity configuration file
        
    Returns:
        List of SensitivityResult objects
    """
    results = []
    
    # Load the simulation configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    simulation_config = SimulationConfig(**config_data["simulation"])
    methods_to_run = config_data.get("methods_to_run", ["threshold", "weighted_score"])
    
    # Process each unique evaluation scenario
    for _, row in tqdm.tqdm(forecasts_df.iterrows()):
        # Skip scenarios with no samples
        if row["total_samples"] <= 0:
            continue
            
        # Create evaluation forecast object from row
        forecast = create_evaluation_forecast_from_row(row)
        
        # Run simulations for each estimator
        for estimator in methods_to_run:
            stats = simulate_estimator(forecast, simulation_config, estimator)
            
            # Create sensitivity result
            result = SensitivityResult(
                ability_scenario=forecast.ability.scenario,
                budget_scenario=forecast.budget_scenario,
                date=forecast.ability.date,
                estimator=estimator,
                bias=float(stats["bias"]),
                variance=float(stats["variance"]),
                ci_lower=float(stats["lower_ci"]),
                ci_upper=float(stats["upper_ci"]),
                contains_true=bool(stats["contains_true"])
            )
            
            results.append(result)
    
    return results

def save_results(results: List[SensitivityResult], output_path: str):
    """Save sensitivity results to CSV file."""
    # Convert to DataFrame
    records = [r.model_dump() for r in results]
    df = pd.DataFrame(records)
    
    # Convert datetime to string for CSV
    if 'date' in df.columns:
        df["date"] = df["date"].apply(lambda x: x.isoformat() if isinstance(x, datetime) else x)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Reading evaluation forecasts from {args.input}")
    forecasts_df = pd.read_csv(args.input)
    
    # For debugging, only use the first 5 rows
    forecasts_df = forecasts_df.head(5)
    
    # Convert date columns back to datetime
    date_cols = [col for col in forecasts_df.columns if 'date' in col]
    for date_col in date_cols:
        forecasts_df[date_col] = pd.to_datetime(forecasts_df[date_col])
        
    logging.info(f"Loaded {len(forecasts_df)} evaluation forecasts")
    
    logging.info(f"Loading simulation configuration from {args.config}")
    
    logging.info("Calculating sensitivity metrics")
    results = calculate_sensitivity_metrics(forecasts_df, args.config)
    logging.info(f"Calculated metrics for {len(results)} scenarios")
    
    logging.info(f"Saving results to {args.out}")
    save_results(results, args.out)
    logging.info("Complete")

if __name__ == "__main__":
    main()
