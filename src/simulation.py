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

class CorrelationModel(str, Enum):
    """Types of correlation models for task successes."""
    NONE = "none"                         # No correlation (independent successes)
    FIXED_ORDER = "fixed_order"           # Perfect correlation (fixed task order)
    MIXTURE = "mixture"                   # Mixture of correlation and independence

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
    correlation_model: CorrelationModel = Field(
        CorrelationModel.NONE, 
        description="Model for task success correlation"
    )
    correlation_strength: float = Field(
        0.0, 
        description="Strength of correlation (0 = independent, 1 = perfect correlation)",
        ge=0.0,
        le=1.0
    )
    elicitation_enabled: bool = Field(
        False,
        description="Enable elicitation impact on successes"
    )
    elicitation_threshold: float = Field(
        0.0,
        description="Difficulty at which elicitation begins to decline"
    )
    elicitation_slope: float = Field(
        1.0,
        description="Steepness of the elicitation impact curve"
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
    Generate success/failure outcomes for tasks based on a logistic model,
    optional correlation, and then apply elicitation impact.
    
    Args:
        task_difficulties: Array of task difficulties
        forecast: Evaluation forecast containing parameters for success generation
        seed: Optional random seed for reproducibility
        
    Returns:
        Binary array of success (1) or failure (0) outcomes
    """
    rng = np.random.RandomState(seed)
    n_tasks = len(task_difficulties)
    
    threshold = forecast.threshold
    slope = forecast.slope
    
    correlation_model = forecast.correlation_model
    correlation_strength = forecast.correlation_strength
    elicitation_enabled = forecast.elicitation_enabled
    elicitation_threshold = forecast.elicitation_threshold
    elicitation_slope = forecast.elicitation_slope
    
    # Calculate success probabilities using logistic function
    probs = logistic_function(task_difficulties, threshold, slope)
    
    if correlation_model == CorrelationModel.NONE or correlation_strength <= 0.0:
        # Independent successes - standard binomial sampling
        success = rng.binomial(1, probs)
    
    elif correlation_model == CorrelationModel.FIXED_ORDER:
        if correlation_strength >= 1.0:
            # Perfect correlation - deterministic cutoff based on threshold
            success = (task_difficulties <= threshold).astype(int)
        else:
            # Mixture of fixed order and independence
            deterministic = (task_difficulties <= threshold).astype(int)
            independent = rng.binomial(1, probs)
            mask = rng.random(n_tasks) < correlation_strength
            success = np.where(mask, deterministic, independent)
    
    elif correlation_model == CorrelationModel.MIXTURE:
        # Generate a correlated random component
        # Here we use a single random value that shifts the threshold
        # Higher correlation_strength = more threshold shifting
        threshold_shift = rng.normal(0, correlation_strength / slope)
        adjusted_probs = logistic_function(task_difficulties, threshold + threshold_shift, slope)
        success = rng.binomial(1, adjusted_probs)
    
    else:
        raise ValueError(f"Unsupported correlation model: {correlation_model}")
    
    # Apply elicitation impact if enabled
    if elicitation_enabled and n_tasks > 0:
        p_imp = 1.0 / (1.0 + np.exp(elicitation_slope * (task_difficulties - elicitation_threshold)))
        keep = rng.binomial(1, p_imp, size=n_tasks)
        success = success * keep

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
        sample_size = config.sample_size if config.sample_size is not None else forecast.n_tasks
        
        for i in range(config.n_samples):
            indices = rng.choice(forecast.n_tasks, size=sample_size, replace=True)
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

