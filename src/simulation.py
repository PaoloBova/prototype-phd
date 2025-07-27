"""
Simulation approaches for evaluation forecasting.

This module provides different simulation methods for analyzing model performance
across task distributions, including:

1. Bootstrap sampling: Resampling from a fixed set of task-outcome pairs
2. Monte Carlo sampling: Generating new tasks and outcomes for each sample
3. Correlated outcome sampling: Modeling consistent task successes across models
"""

import numpy as np
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from .schemas import TaskSamplerType, EvaluationForecast
import prototype_phd.stats as stats

class SimulationMethod(str, Enum):
    """Types of simulation methods."""
    BOOTSTRAP = "bootstrap"               # Resampling from a fixed task-outcome set
    MONTE_CARLO = "monte_carlo"           # Generating new tasks and outcomes each time
    CORRELATED_OUTCOMES = "correlated"    # Modeling correlated task successes


class SimulationConfig(BaseModel):
    """Configuration for simulation approaches."""
    method: SimulationMethod = Field(
        SimulationMethod.MONTE_CARLO, 
        description="Simulation method to use"
    )
    n_samples: int = Field(
        1000, 
        description="Number of simulation samples to generate"
    )
    sample_size: Optional[int] = Field(
        None, 
        description="Size of each sample (defaults to original data size)"
    )
    random_seed: Optional[int] = Field(
        None, 
        description="Random seed for reproducibility"
    )
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


def generate_task_samples(
    forecast: EvaluationForecast,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate task difficulty samples.
    
    Args:
        forecast: Evaluation forecast containing parameters for task generation
        seed: Optional random seed for reproducibility
        
    Returns:
        Array of task difficulties
    """
    rng = np.random.RandomState(seed)
    
    n_tasks = forecast.total_samples
    window_lower = forecast.window_lower
    window_upper = forecast.window_upper
    sampler_type = forecast.sampler_type
    
    if n_tasks <= 0:
        return np.array([])
    
    if sampler_type == TaskSamplerType.UNIFORM:
        return rng.uniform(window_lower, window_upper, n_tasks)
    
    elif sampler_type == TaskSamplerType.NORMAL:
        window_width = window_upper - window_lower
        window_center = (window_lower + window_upper) / 2
        mean = window_center
        std = window_width * 0.3
        samples = rng.normal(mean, std, n_tasks)
        # Truncate to window bounds
        return np.clip(samples, window_lower, window_upper)
    
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


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


def generate_success_outcomes(
    task_difficulties: np.ndarray,
    config: SimulationConfig,
    forecast: EvaluationForecast,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate success/failure outcomes for tasks.
    
    Args:
        task_difficulties: Array of task difficulties
        forecast: Evaluation forecast containing parameters for success generation
        seed: Optional random seed for reproducibility
        
    Returns:
        Binary array of success (1) or failure (0) outcomes
    """
    rng = np.random.RandomState(seed)
    n_tasks = len(task_difficulties)
    
    threshold = forecast.ability.threshold
    slope = forecast.ability.slope
    elicitation_bias_enabled = forecast.elicitation_bias_enabled
    elicitation_bias_type = forecast.elicitation_bias_type
    elicitation_bias_args = forecast.elicitation_bias_args
    alternate_ability_enabled = forecast.alternate_ability_enabled
    alternate_ability_type = forecast.alternate_ability_type
    alternate_ability_args = forecast.alternate_ability_args
    
    # Calculate success probabilities
    if alternate_ability_enabled:
        # If alternate ability function is enabled, apply it to modify probabilities
        if alternate_ability_type == "exponential":
            # Exponential survival function: S(x) = exp(-λx) where λ is the rate parameter
            rate_param = alternate_ability_args[0] if len(alternate_ability_args) > 0 else 1.0
            convert_from_log2 = alternate_ability_args[1] if len(alternate_ability_args) > 1 else True
            
            # Convert task difficulties from log2 space to linear space if specified
            if convert_from_log2:
                linear_difficulties = np.exp2(task_difficulties)
            else:
                linear_difficulties = task_difficulties
                
            probs = np.exp(-rate_param * linear_difficulties)
            probs = np.clip(probs, 0, 1)
        
        elif alternate_ability_type == "power_law":
            # Power law function
            exponent = alternate_ability_args[0] if len(alternate_ability_args) > 0 else 1.0
            # Avoid division by zero and ensure positive values
            threshold_safe = max(threshold, 1e-6)
            ratio = np.maximum(task_difficulties, 1e-6) / threshold_safe
            probs = ratio ** (-exponent)
            probs = np.clip(probs, 0, 1)
        elif alternate_ability_type == "tangent":
            # Tangent function
            slope = alternate_ability_args[0] if len(alternate_ability_args) > 0 else 1.0
            intercept = alternate_ability_args[1] if len(alternate_ability_args) > 1 else 0.0
            probs = np.tan(slope * (task_difficulties - threshold)) + intercept
            probs = np.clip(probs, 0, 1)
        elif alternate_ability_type == "logistic":
            # Logistic function with custom parameters
            if len(alternate_ability_args) < 2:
                raise ValueError("Logistic function requires threshold and slope parameters.")
            alt_threshold = alternate_ability_args[0]
            alt_slope = alternate_ability_args[1]
            probs = logistic_function(task_difficulties, alt_threshold, alt_slope)
            probs = np.clip(probs, 0, 1)
        else:
            raise ValueError(f"Unsupported alternate ability function: {alternate_ability_type}")
    else:
        # Default logistic function probabilities
        probs = logistic_function(task_difficulties, threshold, slope)

    # Generate success outcomes using binomial sampling
    success = rng.binomial(1, probs)
    
    # Apply elicitation impact if enabled
    if elicitation_bias_enabled and n_tasks > 0:
        if elicitation_bias_type == "fall_past_threshold":
            # Sensitivity rate falls from 1 to new rate past ability threshold
            sensitivity_rate = elicitation_bias_args[0] if len(elicitation_bias_args) > 0 else 0.0
            keep_probs = np.where(task_difficulties <= threshold, 1.0, sensitivity_rate)
        elif elicitation_bias_type == "linear":
            # Linear decline based on task difficulty
            elicitation_threshold = elicitation_bias_args[0] if len(elicitation_bias_args) > 0 else 0.0
            elicitation_slope = elicitation_bias_args[1] if len(elicitation_bias_args) > 1 else 1.0
            keep_probs = np.clip(1 - elicitation_slope * (task_difficulties - elicitation_threshold) / (forecast.window_upper - elicitation_threshold), 0, 1)
        elif elicitation_bias_type == "logistic":
            # Logistic decline based on task difficulty
            elicitation_threshold = elicitation_bias_args[0] if len(elicitation_bias_args) > 0 else 0.0
            elicitation_slope = elicitation_bias_args[1] if len(elicitation_bias_args) > 1 else 1.0
            keep_probs = logistic_function(task_difficulties, elicitation_threshold, elicitation_slope)
        else:
            raise ValueError(f"Unsupported elicitation bias type: {elicitation_bias_type}")
        
        # Apply elicitation bias as a binary masking based on keep_probs
        keep_mask = rng.binomial(1, keep_probs)
        success = success * keep_mask

    return success


def run_simulation(
    config: SimulationConfig,
    forecast: EvaluationForecast,
    analysis_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """
    Run a simulation using the specified configuration.
    
    Args:
        config: Simulation configuration
        forecast: Evaluation forecast
        analysis_fn: Function that takes (tasks, outcomes) and returns a metric
        
    Returns:
        Array of simulation results
    """
    rng = np.random.RandomState(config.random_seed)
    results = np.zeros(config.n_samples)
    
    if config.method == SimulationMethod.BOOTSTRAP:
        # Generate a single set of tasks and outcomes
        base_tasks = generate_task_samples(forecast, seed=config.random_seed)
        base_outcomes = generate_success_outcomes(base_tasks, config, forecast)
        
        # Bootstrap from this fixed set
        sample_size = config.sample_size if config.sample_size is not None else forecast.total_samples
        
        for i in range(config.n_samples):
            indices = rng.choice(forecast.total_samples, size=sample_size, replace=True)
            tasks = base_tasks[indices]
            outcomes = base_outcomes[indices]
            
            if analysis_fn is not None:
                results[i] = analysis_fn(tasks, outcomes)
    
    elif config.method == SimulationMethod.MONTE_CARLO:
        # Generate new tasks and outcomes for each sample
        for i in range(config.n_samples):
            seed_i = None if config.random_seed is None else config.random_seed + i
            tasks = generate_task_samples(forecast, seed=seed_i)
            outcomes = generate_success_outcomes(tasks, config, forecast, seed=seed_i)
            
            if analysis_fn is not None:
                results[i] = analysis_fn(tasks, outcomes)
    
    else:
        raise ValueError(f"Unsupported simulation method: {config.method}")
    
    return results


def calculate_simulation_statistics(results: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for simulation results.
    
    Args:
        results: Array of simulation results
        
    Returns:
        Dictionary with statistics (mean, std, percentiles)
    """
    if len(results) == 0:
        return {
            "mean": float('nan'),
            "std": float('nan'),
            "lower_ci": float('nan'),
            "upper_ci": float('nan')
        }
    
    return {
        "mean": float(np.mean(results)),
        "std": float(np.std(results)),
        "lower_ci": float(np.percentile(results, 2.5)),
        "upper_ci": float(np.percentile(results, 97.5))
    }

def weighted_score_estimator(
    tasks: np.ndarray,
    outcomes: np.ndarray,
    bin_edges: np.ndarray,
    level_weight_fn: Callable[[float], float] = lambda x: 1.0 + 0.5 * x,
    info_weight_fn: Optional[Callable[[float,int], float]] = None,
    normalize: bool = False
) -> float:
    """
    Compute weighted sum of success rates per difficulty level.

    Args:
        tasks: Array of task difficulties
        outcomes: Binary outcomes array
        bin_edges: array of bin edges
        level_weight_fn: Weight for each difficulty level
        info_weight_fn: Weight based on information (e.g., counts) per level

    Returns:
        Weighted sum of success rates across unique difficulty levels.
    """

    # 1) unique levels, sorted
    if len(bin_edges) < 2:
        return float('nan')
    # assign each task to a bin center
    n_bins = len(bin_edges) - 1
    idx = np.minimum(np.digitize(tasks, bin_edges) - 1, n_bins - 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    tasks_disc = bin_centers[idx]
    levels = bin_centers

    # 2) compute bin‐widths for a trapezoid rule
    bin_widths = np.diff(bin_edges)

    # 3) per‐level success rates
    rates = np.array([outcomes[tasks_disc==lev].mean() for lev in levels])
    # 4) weights
    lvl_w  = np.array([level_weight_fn(lev) for lev in levels])
    info_w = (np.ones_like(rates)
              if info_weight_fn is None
              else np.array([info_weight_fn(lev, (tasks_disc==lev).sum())
                              for lev in levels]))

    # 5) area approximation
    area = np.nansum(rates * lvl_w * info_w * bin_widths)

    if normalize:
        norm = np.nansum(lvl_w * info_w * bin_widths)
        return float(area / (norm or np.nan))
    return float(area)

