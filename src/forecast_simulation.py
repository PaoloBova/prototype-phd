"""
Simulations for evaluation forecasting.
"""
import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import h5py
import prototype_phd.data_utils as data_utils
from datetime import datetime
import tqdm
from typing import Dict, List, Optional, Tuple, Any, Callable
import scipy.stats

from .schemas import (
    EvaluationForecast, 
    AbilityForecast,
    SensitivityResult
)
from .simulation import (
    SimulationConfig, SimulationMethod, CorrelationModel,
    run_simulation, calculate_simulation_statistics, weighted_score_estimator
)

import prototype_phd.stats

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate sensitivity metrics")
    parser.add_argument("--in", dest="input", required=True, help="Path to evaluation forecasts CSV")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    parser.add_argument("--raw", required=False, help="Path to raw simulation results HDF5 file")
    parser.add_argument("--config", default="configs/forecast_detection/forecast_simulation.json", 
                        help="Path to sensitivity config file")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited forecasts")
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
    
    # Create evaluation forecast object with all needed fields
    forecast_data = {
        "ability": ability,
        "budget_fraction": row["budget_fraction"],
        "budget_scenario": row["budget_scenario"],
        "window_lower": row["window_lower"],
        "window_upper": row["window_upper"],
        "sampler_type": row["sampler_type"],
        "total_samples": int(row["total_samples"]),
        "gold_standard_cost": row["gold_standard_cost"],
        "available_budget": row["available_budget"],
        "adjustment_method": row["adjustment_method"],
        "original_window_lower": row["original_window_lower"],
        "original_window_upper": row["original_window_upper"],
        "repeats_per_unit": row.get("repeats_per_unit", 20),  # Default if missing
        "cost_model": row["cost_model"],
        "doubling_rate": row["doubling_rate"],
        "ability_id": row["ability_id"],
        "cost_id": row["cost_id"],
        "constraint_id": row["constraint_id"],
        "design_id": row["design_id"]
    }
    
    # Add variant information if available in the DataFrame
    if "ability_variant" in row:
        forecast_data["ability_variant"] = row["ability_variant"]
    if "cost_variant" in row:
        forecast_data["cost_variant"] = row["cost_variant"]
    if "base_ability_id" in row:
        forecast_data["base_ability_id"] = row["base_ability_id"]
    if "base_cost_id" in row:
        forecast_data["base_cost_id"] = row["base_cost_id"]
    
    return EvaluationForecast(**forecast_data)

def create_weight_function(weight_config: Dict[str, Any]) -> Callable[[float], float]:
    """
    Create a weight function based on configuration.
    
    Args:
        weight_config: Weight function configuration
        
    Returns:
        Callable weight function
    """
    weight_type = weight_config.get("type", "linear")
    
    if weight_type == "linear":
        base = weight_config.get("base", 1.0)
        slope = weight_config.get("slope", 0.5)
        return lambda x: base + slope * x
    elif weight_type == "exponential":
        base = weight_config.get("base", 1.0)
        scale = weight_config.get("scale", 0.1)
        return lambda x: base * np.exp(scale * x)
    elif weight_type == "constant":
        value = weight_config.get("value", 1.0)
        return lambda x: value
    else:
        logging.warning(f"Unsupported weight function type: {weight_type}, using default linear")
        return lambda x: 1.0 + 0.5 * x

def calculate_true_weighted_score(threshold: float, slope: float, config: Dict[str, Any]) -> float:
    """
    Calculate the true weighted score for a given logistic curve.
    
    Integrates the product of the logistic function and the weight function
    over the difficulty range, then normalizes by the integral of weights.
    
    Args:
        threshold: Threshold parameter of logistic curve
        slope: Slope parameter of logistic curve
        config: Configuration for the calculation
    
    Returns:
        True weighted score
    """
    # Throw a warning if no range provided
    if "range_min" not in config or "range_max" not in config:
        logging.warning("No range provided in config, using default range [-1000, 1000]")
    range_min = config.get("range_min", -1000)
    range_max = config.get("range_max", 1000)
    weight_fn = create_weight_function(config.get("weight_function", {"type": "linear"}))
    # Create a fine grid of difficulties
    
    diff_grid = np.linspace(range_min, range_max, 1000)
    # Delegates to weighted_score_estimator by treating the logistic
    # probability curve as “outcomes” on a fine difficulty grid.
    probs = logistic_function(diff_grid, threshold, slope)
    # discretize continuous tasks into bins for per‐level estimates
    n_bins = config.get("n_bins", 10)
    bin_edges = np.linspace(range_min, range_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    idx = np.minimum(np.digitize(diff_grid, bin_edges) - 1, n_bins - 1)
    tasks_disc = bin_centers[idx]
    # use weighted_score_estimator to combine levels
    normalize = config.get("normalize", False)
    weighted_score = weighted_score_estimator(tasks_disc,
                                              probs,
                                              level_weight_fn=weight_fn,
                                              normalize=normalize)
    return weighted_score


def simulate_estimator(
    forecast: EvaluationForecast,
    simulation_config: SimulationConfig,
    estimator: str,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Simulate an estimator distribution using the specified configuration.
    
    Args:
        forecast: Evaluation forecast to simulate
        simulation_config: Simulation configuration
        estimator: Which estimator to use ("threshold" or "weighted_score")
        config: Configuration for estimators
        
    Returns:
        Array of simulation results
    """
    # Set up the analysis function based on the specified estimator
    if estimator == "threshold":
        threshold_config = config.get("threshold_estimator", {})
        # if using scikit-learn with warm_start, reuse one classifier for all samples
        if threshold_config.get("engine") == "scikit-learn" and threshold_config.get("warm_start", False):
            from sklearn.linear_model import LogisticRegression
            lr_conf = prototype_phd.stats.LogRegConfig(**threshold_config)
            clf = LogisticRegression(
                C=lr_conf.C,
                solver=lr_conf.solver,
                max_iter=lr_conf.max_iter,
                tol=lr_conf.tol,
                warm_start=lr_conf.warm_start,
                random_state=lr_conf.random_state
            )
            def analysis_fn(tasks, outcomes):
                X = tasks[:, None]
                clf.fit(X, outcomes)                     # warm start uses previous coef
                b0 = clf.intercept_[0]
                b1 = clf.coef_[0][0]
                return -b0 / b1                          # threshold = –intercept/coefficient
        else:
            logreg_config = prototype_phd.stats.LogRegConfig(**threshold_config)
            analysis_fn = lambda tasks, outcomes: prototype_phd.stats.compute_threshold(
                tasks[:, None], outcomes,
                config=logreg_config,
            )
    elif estimator == "weighted_score":
        ws_cfg = config.get("weighted_score", {})
        ws_cfg = {
            **ws_cfg,
            "range_min": forecast.window_lower,
            "range_max": forecast.window_upper,
        }
        weight_fn = create_weight_function(ws_cfg.get("weight_function", {"type": "linear"}))
        
        # discretize continuous tasks into bins for per‐level estimates
        n_bins = ws_cfg.get("n_bins", 10)
        bin_edges = np.linspace(forecast.window_lower, forecast.window_upper, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        def analysis_fn(tasks, outcomes):
            # assign each task to a bin center
            idx = np.minimum(np.digitize(tasks, bin_edges) - 1, n_bins - 1)
            tasks_disc = bin_centers[idx]
            return weighted_score_estimator(tasks_disc,
                                            outcomes,
                                            level_weight_fn=weight_fn,
                                            normalize=ws_cfg.get("normalize", False))
    elif estimator == "max_success":
        # 1) Most difficult single successful task
        def analysis_fn(tasks, outcomes):
            succ = tasks[outcomes == 1]
            return float(np.max(succ)) if succ.size > 0 else np.nan

    elif estimator == "bin_threshold":
        # 2) Most difficult bin with >50% success
        # reuse number of bins from weighted_score config (fallback 10)
        n_bins = config.get("weighted_score", {}).get("n_bins", 10)
        def analysis_fn(tasks, outcomes):
            edges = np.linspace(tasks.min(), tasks.max(), n_bins + 1)
            bins = np.digitize(tasks, edges) - 1
            centers = (edges[:-1] + edges[1:]) / 2
            # compute success rate per bin
            rates = [
                outcomes[bins == i].mean() if np.any(bins == i) else 0.0
                for i in range(n_bins)
            ]
            valid = [centers[i] for i, r in enumerate(rates) if r > 0.5]
            return float(max(valid)) if valid else np.nan

    elif estimator == "logistic_weighted":
        # 3) Fit logistic, then weighted‐sum over its predicted curve
        lr_cfg = prototype_phd.stats.LogRegConfig(**config.get("threshold_estimator", {}))
        ws_cfg = config.get("weighted_score", {})
        weight_fn = create_weight_function(ws_cfg.get("weight_function", {"type":"linear"}))
        normalize = ws_cfg.get("normalize", False)
        def analysis_fn(tasks, outcomes):
            X = tasks.reshape(-1, 1)
            res = prototype_phd.stats.fit_logistic(X, outcomes, lr_cfg)
            b0, b1 = res.coeffs
            # build predicted probs over the full evaluation window, not just the sampled tasks
            grid = np.linspace(forecast.window_lower, forecast.window_upper, 200)
            probs = 1.0 / (1.0 + np.exp(-b1 * (grid - (-b0 / b1))))
            # weighted‐sum under estimated logistic curve
            return weighted_score_estimator(
                grid, probs,
                level_weight_fn=weight_fn,
                normalize=normalize
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

    return results

def calculate_true_value(estimator: str, forecast: EvaluationForecast, config: Dict[str, Any]) -> float:
    """
    Calculate the true value for a given estimator and forecast.
    
    Args:
        estimator: Type of estimator ("threshold" or "weighted_score")
        forecast: Evaluation forecast
        config: Configuration for calculation
        
    Returns:
        True value for the estimator
    """
    if estimator == "threshold":
        return forecast.ability.threshold
    elif estimator == "weighted_score":
        ws_cfg = config.get("weighted_score", {})
        # inject the *same* window used for simulation:
        ws_cfg = {
            **ws_cfg,
            "range_min": forecast.original_window_lower,
            "range_max": forecast.original_window_upper,
        }
        return calculate_true_weighted_score(
            forecast.ability.threshold,
            forecast.ability.slope,
            ws_cfg
        )
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")

def calculate_stats(
    results: np.ndarray,
    estimator: str,
    true_value: float
) -> Dict[str, Any]:
    """
    Calculate statistics from simulation results.
    
    Args:
        results: Array of simulation results
        estimator: Type of estimator ("threshold" or "weighted_score")
        true_value: True value of the parameter
        
    Returns:
        Dictionary of calculated statistics
    """
    # Get basic statistics (mean, std, CI)
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
    
    # Add more advanced statistics
    valid_results = results[~np.isnan(results)]
    if len(valid_results) > 0:
        stats["median"] = float(np.median(valid_results))
        stats["skewness"] = float(scipy.stats.skew(valid_results)) if len(valid_results) > 2 else np.nan
        stats["kurtosis"] = float(scipy.stats.kurtosis(valid_results)) if len(valid_results) > 2 else np.nan
        stats["min"] = float(np.min(valid_results))
        stats["max"] = float(np.max(valid_results))
        stats["q1"] = float(np.percentile(valid_results, 25))
        stats["q3"] = float(np.percentile(valid_results, 75))
        stats["iqr"] = stats["q3"] - stats["q1"]
        stats["valid_ratio"] = len(valid_results) / len(results) if len(results) > 0 else 0.0
    
    return stats

def filter_forecasts(
    forecasts_df: pd.DataFrame, 
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Filter forecasts based on configuration.
    
    Args:
        forecasts_df: DataFrame with evaluation forecasts
        filters: Filter configuration
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = forecasts_df.copy()
    
    # Basic sample size filter
    min_samples = filters.get("min_samples")
    if min_samples is not None:
        filtered_df = filtered_df[filtered_df["total_samples"] >= min_samples]
    
    max_samples = filters.get("max_samples")
    if max_samples is not None:
        filtered_df = filtered_df[filtered_df["total_samples"] <= max_samples]
    
    # Filter by variants
    ability_variants = filters.get("ability_variants")
    if ability_variants is not None and "ability_variant" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["ability_variant"].isin(ability_variants)]
    
    cost_variants = filters.get("cost_variants")
    if cost_variants is not None and "cost_variant" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["cost_variant"].isin(cost_variants)]
    
    # Filter by sampler type
    sampler_types = filters.get("sampler_types")
    if sampler_types is not None:
        filtered_df = filtered_df[filtered_df["sampler_type"].isin(sampler_types)]
    
    logging.info(f"Filtered from {len(forecasts_df)} to {len(filtered_df)} forecasts")
    
    return filtered_df

def generate_simulation_id(forecast: EvaluationForecast, estimator: str, sim_config: SimulationConfig) -> str:
    """
    Generate a unique ID for a simulation run.
    
    Args:
        forecast: Evaluation forecast
        estimator: Estimator type
        sim_config: Simulation configuration
        
    Returns:
        Unique simulation ID
    """
    # Include variant information in ID if available
    variant_info = ""
    if hasattr(forecast, "ability_variant") and hasattr(forecast, "cost_variant"):
        if forecast.ability_variant != "unknown" or forecast.cost_variant != "unknown":
            variant_info = f"{forecast.ability_variant}_{forecast.cost_variant}_"
    
    components = [
        forecast.ability.scenario,
        forecast.budget_scenario,
        forecast.ability.date.strftime("%Y-%m-%d"),
        variant_info + estimator,
        sim_config.method,
        f"{forecast.total_samples}_samples",
        f"{sim_config.n_samples}_sims"
    ]
    return "__".join([str(c).replace(" ", "_") for c in components])

def create_nested_hdf5_structure(
    raw_file: h5py.File,
    forecast: EvaluationForecast,
    estimator: str,
    sim_results: np.ndarray,
    stats_data: Dict[str, Any],
    sim_config: SimulationConfig,
    true_value: float
) -> None:
    """
    Create a nested hierarchical structure in the HDF5 file for storing simulation results.
    
    Args:
        raw_file: Open HDF5 file
        forecast: Evaluation forecast
        estimator: Estimator type
        sim_results: Simulation results array
        stats_data: Statistics calculated from the results
        sim_config: Simulation configuration
    """
    # Create hierarchical structure:
    # /ability_scenario/budget_scenario/date/estimator/simulation_method
    
    # Level 1: Ability scenario (include variant info if available)
    ability_scenario_name = str(forecast.ability.scenario)
    if hasattr(forecast, "ability_variant") and forecast.ability_variant != "unknown":
        ability_scenario_name += f"_{forecast.ability_variant}"
    ability_group_name = ability_scenario_name.replace(" ", "_")
    
    if ability_group_name not in raw_file:
        ability_group = raw_file.create_group(ability_group_name)
    else:
        ability_group = raw_file[ability_group_name]
    
    # Level 2: Budget scenario (include cost variant if available)
    budget_scenario_name = str(forecast.budget_scenario)
    if hasattr(forecast, "cost_variant") and forecast.cost_variant != "unknown":
        budget_scenario_name += f"_{forecast.cost_variant}"
    budget_group_name = budget_scenario_name.replace(" ", "_")
    
    if budget_group_name not in ability_group:
        budget_group = ability_group.create_group(budget_group_name)
    else:
        budget_group = ability_group[budget_group_name]
    
    # Level 3: Date
    date_group_name = forecast.ability.date.strftime("%Y-%m-%d")
    if date_group_name not in budget_group:
        date_group = budget_group.create_group(date_group_name)
    else:
        date_group = budget_group[date_group_name]
    
    # Level 4: Estimator
    estimator_group_name = str(estimator)
    if estimator_group_name not in date_group:
        estimator_group = date_group.create_group(estimator_group_name)
    else:
        estimator_group = date_group[estimator_group_name]
    
    # Level 5: Simulation method
    method_group_name = f"{sim_config.method}_{forecast.total_samples}_samples"
    
    # Ensure uniqueness by adding an index if needed
    base_method_name = method_group_name
    index = 1
    while method_group_name in estimator_group:
        method_group_name = f"{base_method_name}_{index}"
        index += 1
        
    sim_group = estimator_group.create_group(method_group_name)
    
    # Store the results array
    sim_group.create_dataset('results', data=sim_results)
    
    # Store metadata as attributes
    sim_group.attrs['true_value'] = true_value
    sim_group.attrs['window_lower'] = forecast.window_lower
    sim_group.attrs['window_upper'] = forecast.window_upper
    sim_group.attrs['total_samples'] = forecast.total_samples
    sim_group.attrs['threshold'] = forecast.ability.threshold
    sim_group.attrs['slope'] = forecast.ability.slope
    sim_group.attrs['sampler_type'] = str(forecast.sampler_type)
    sim_group.attrs['budget_fraction'] = forecast.budget_fraction
    sim_group.attrs['correlation_model'] = str(sim_config.correlation_model)
    sim_group.attrs['correlation_strength'] = sim_config.correlation_strength
    
    # Store variant information if available
    if hasattr(forecast, "ability_variant"):
        sim_group.attrs['ability_variant'] = forecast.ability_variant
    if hasattr(forecast, "cost_variant"):
        sim_group.attrs['cost_variant'] = forecast.cost_variant
    
    # Store full statistics
    stats_group = sim_group.create_group('stats')
    for stat_name, stat_value in stats_data.items():
        # Handle different types correctly
        if isinstance(stat_value, str):
            stats_group.attrs[stat_name] = stat_value
        elif isinstance(stat_value, (bool, int)):
            stats_group.attrs[stat_name] = stat_value
        elif isinstance(stat_value, float) and not np.isnan(stat_value):
            stats_group.attrs[stat_name] = stat_value

def run_simulations(
    forecasts_df: pd.DataFrame,
    config_path: str,
    raw_output_path: Optional[str] = None
) -> Tuple[List[SensitivityResult], Dict[str, np.ndarray]]:
    """
    Run simulations for each evaluation forecast and estimator.
    
    Args:
        forecasts_df: DataFrame with evaluation forecasts
        config_path: Path to simulation configuration file
        raw_output_path: Optional path to save raw simulation results
        
    Returns:
        Tuple of (list of SensitivityResult objects, dict of raw results)
    """
    results = []
    raw_results = {}
    
    # Load the simulation configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Get filters from config and apply them
    filters = config_data.get("filters", {})
    filtered_df = filter_forecasts(forecasts_df, filters)
    
    # Extract simulation configuration
    simulation_config = SimulationConfig(**config_data["simulation"])
    methods_to_run = config_data.get("methods_to_run", ["threshold", "weighted_score"])
    
    # Set up HDF5 file for raw results if path provided
    if raw_output_path:
        os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
        raw_file = h5py.File(raw_output_path, 'w')
        # Create a metadata group
        meta_group = raw_file.create_group('metadata')
        # Store simulation config as attributes
        for key, value in simulation_config.model_dump().items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
        
        # Store estimator configs if save_individual_simulations is enabled
        if config_data.get("output", {}).get("save_individual_simulations", False):
            for estimator in methods_to_run:
                if estimator in config_data:
                    estimator_group = meta_group.create_group(estimator)
                    for key, value in config_data[estimator].items():
                        if isinstance(value, (str, int, float, bool)):
                            estimator_group.attrs[key] = value
    else:
        raw_file = None
    
    try:
        # Process each evaluation forecast
        for _, row in tqdm.tqdm(filtered_df.iterrows(), total=len(filtered_df)):
            # Skip scenarios with no samples
            if row["total_samples"] <= 0:
                continue
                
            # Create evaluation forecast object from row
            forecast = create_evaluation_forecast_from_row(row)
            
            # Run simulations for each estimator
            for estimator in methods_to_run:
                # Generate a unique simulation ID for the in-memory storage
                sim_id = generate_simulation_id(forecast, estimator, simulation_config)
                
                # Run the simulation and get results array
                sim_results = simulate_estimator(forecast, simulation_config, estimator, config_data)
                
                # Calculate true value
                true_value = calculate_true_value(estimator, forecast, config_data)
                
                # Calculate statistics
                stats = calculate_stats(sim_results, estimator, true_value)
                
                # --- build metadata for CSV output ---
                additional_fields = {
                    "ability_id": forecast.ability_id,
                    "cost_id": forecast.cost_id,
                    "constraint_id": forecast.constraint_id,
                    "design_id": forecast.design_id,
                    "budget_fraction": forecast.budget_fraction,
                    "window_lower": forecast.window_lower,
                    "window_upper": forecast.window_upper,
                    "original_window_lower": forecast.original_window_lower,
                    "original_window_upper": forecast.original_window_upper,
                    "total_samples": forecast.total_samples
                }
                # include variants if present
                if hasattr(forecast, "ability_variant"):
                    additional_fields["ability_variant"] = forecast.ability_variant
                if hasattr(forecast, "cost_variant"):
                    additional_fields["cost_variant"] = forecast.cost_variant
                if hasattr(forecast, "base_ability_id"):
                    additional_fields["base_ability_id"] = forecast.base_ability_id
                if hasattr(forecast, "base_cost_id"):
                    additional_fields["base_cost_id"] = forecast.base_cost_id

                sensitivity_result = SensitivityResult(
                    ability_scenario=forecast.ability.scenario,
                    budget_scenario=forecast.budget_scenario,
                    date=forecast.ability.date,
                    estimator=estimator,
                    bias=float(stats["bias"]),
                    variance=float(stats["variance"]),
                    mean=float(stats["mean"]),               
                    true_value=true_value,                   
                    ci_lower=float(stats["lower_ci"]),
                    ci_upper=float(stats["upper_ci"]),
                    contains_true=bool(stats["contains_true"]),
                    **additional_fields
                )
                results.append(sensitivity_result)

                # Save raw results to HDF5 file if provided and configured
                if raw_file is not None and config_data.get("output", {}).get("save_individual_simulations", False):
                    create_nested_hdf5_structure(
                        raw_file, forecast, estimator, sim_results, stats, simulation_config, true_value
                    )

                # Store in memory
                raw_results[sim_id] = {
                    'results': sim_results,
                    'metadata': {
                        'estimator': estimator,
                        'ability_scenario': forecast.ability.scenario,
                        'budget_scenario': forecast.budget_scenario,
                        'date': forecast.ability.date,
                        'true_value': true_value,
                        'mean': stats["mean"],
                        'window_lower': forecast.window_lower,
                        'window_upper': forecast.window_upper,
                        'total_samples': forecast.total_samples,
                        'threshold': forecast.ability.threshold,
                        'slope': forecast.ability.slope,
                        'sampler_type': forecast.sampler_type,
                        'ability_id': forecast.ability_id,
                        'cost_id': forecast.cost_id,
                        'constraint_id': forecast.constraint_id,
                        'design_id': forecast.design_id,
                        'budget_fraction': forecast.budget_fraction
                    },
                    'stats': stats
                }
                # add variants into raw_results as well
                if hasattr(forecast, "ability_variant"):
                    raw_results[sim_id]['metadata']['ability_variant'] = forecast.ability_variant
                if hasattr(forecast, "cost_variant"):
                    raw_results[sim_id]['metadata']['cost_variant'] = forecast.cost_variant
                if hasattr(forecast, "base_ability_id"):
                    raw_results[sim_id]['metadata']['base_ability_id'] = forecast.base_ability_id
                if hasattr(forecast, "base_cost_id"):
                    raw_results[sim_id]['metadata']['base_cost_id'] = forecast.base_cost_id
    finally:
        # Close the HDF5 file if it was opened
        if raw_file is not None:
            raw_file.close()
    
    return results, raw_results

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
    
    # In debug mode, only use a small subset
    if args.debug:
        logging.info("Running in debug mode with limited forecasts")
        forecasts_df = forecasts_df.head(5)
    
    # Convert date columns back to datetime
    date_cols = [col for col in forecasts_df.columns if 'date' in col]
    for date_col in date_cols:
        forecasts_df[date_col] = pd.to_datetime(forecasts_df[date_col])
        
    logging.info(f"Loaded {len(forecasts_df)} evaluation forecasts")
    
    logging.info(f"Loading simulation configuration from {args.config}")
    
    logging.info("Running simulations for evaluation forecasts")
    results, raw_results = run_simulations(
        forecasts_df, 
        args.config,
        raw_output_path=args.raw if hasattr(args, 'raw') else None
    )
    logging.info(f"Generated results for {len(results)} scenarios")
    
    logging.info(f"Saving summary results to {args.out}")
    save_results(results, args.out)
    logging.info("Complete")

if __name__ == "__main__":
    main()
