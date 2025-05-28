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
from .schemas import TaskSamplerType

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
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


def generate_task_samples(
    n_tasks: int,
    window_lower: float,
    window_upper: float,
    sampler_type: TaskSamplerType = TaskSamplerType.UNIFORM,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate task difficulty samples.
    
    Args:
        n_tasks: Number of tasks to generate
        window_lower: Lower bound of difficulty window
        window_upper: Upper bound of difficulty window
        sampler_type: Type of sampling distribution
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Array of task difficulties
    """
    if n_tasks <= 0:
        return np.array([])
        
    rng = np.random.RandomState(random_seed)
    
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
    threshold: float,
    slope: float,
    correlation_model: CorrelationModel = CorrelationModel.NONE,
    correlation_strength: float = 0.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate success/failure outcomes for tasks based on a logistic model.
    
    Args:
        task_difficulties: Array of task difficulties
        threshold: Threshold parameter (difficulty where success rate = 0.5)
        slope: Slope parameter (steepness of the logistic curve)
        correlation_model: Model for task success correlation
        correlation_strength: Strength of correlation (0 = independent, 1 = perfect)
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Binary array of success (1) or failure (0) outcomes
    """
    rng = np.random.RandomState(random_seed)
    n_tasks = len(task_difficulties)
    
    # Calculate success probabilities using logistic function
    probs = logistic_function(task_difficulties, threshold, slope)
    
    if correlation_model == CorrelationModel.NONE or correlation_strength <= 0.0:
        # Independent successes - standard binomial sampling
        return rng.binomial(1, probs)
    
    elif correlation_model == CorrelationModel.FIXED_ORDER:
        if correlation_strength >= 1.0:
            # Perfect correlation - deterministic cutoff based on threshold
            return (task_difficulties <= threshold).astype(int)
        else:
            # Mixture of fixed order and independence
            deterministic = (task_difficulties <= threshold).astype(int)
            independent = rng.binomial(1, probs)
            mask = rng.random(n_tasks) < correlation_strength
            return np.where(mask, deterministic, independent)
    
    elif correlation_model == CorrelationModel.MIXTURE:
        # Generate a correlated random component
        # Here we use a single random value that shifts the threshold
        # Higher correlation_strength = more threshold shifting
        threshold_shift = rng.normal(0, correlation_strength / slope)
        adjusted_probs = logistic_function(task_difficulties, threshold + threshold_shift, slope)
        return rng.binomial(1, adjusted_probs)
    
    else:
        raise ValueError(f"Unsupported correlation model: {correlation_model}")


def run_simulation(
    config: SimulationConfig,
    n_tasks: int,
    window_lower: float,
    window_upper: float,
    threshold: float,
    slope: float,
    sampler_type: TaskSamplerType = TaskSamplerType.UNIFORM,
    analysis_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """
    Run a simulation using the specified configuration.
    
    Args:
        config: Simulation configuration
        n_tasks: Number of tasks in the evaluation
        window_lower: Lower bound of difficulty window
        window_upper: Upper bound of difficulty window
        threshold: Threshold parameter of the logistic curve
        slope: Slope parameter of the logistic curve
        sampler_type: Type of task sampling distribution
        analysis_fn: Function that takes (tasks, outcomes) and returns a metric
        
    Returns:
        Array of simulation results
    """
    rng = np.random.RandomState(config.random_seed)
    results = np.zeros(config.n_samples)
    
    if config.method == SimulationMethod.BOOTSTRAP:
        # Generate a single set of tasks and outcomes
        base_tasks = generate_task_samples(
            n_tasks, window_lower, window_upper, sampler_type, 
            random_seed=config.random_seed
        )
        base_outcomes = generate_success_outcomes(
            base_tasks, threshold, slope, 
            config.correlation_model, config.correlation_strength,
            random_seed=config.random_seed
        )
        
        # Bootstrap from this fixed set
        sample_size = config.sample_size if config.sample_size is not None else n_tasks
        
        for i in range(config.n_samples):
            indices = rng.choice(n_tasks, size=sample_size, replace=True)
            tasks = base_tasks[indices]
            outcomes = base_outcomes[indices]
            
            if analysis_fn is not None:
                results[i] = analysis_fn(tasks, outcomes)
    
    elif config.method == SimulationMethod.MONTE_CARLO:
        # Generate new tasks and outcomes for each sample
        for i in range(config.n_samples):
            seed_i = None if config.random_seed is None else config.random_seed + i
            tasks = generate_task_samples(
                n_tasks, window_lower, window_upper, sampler_type, 
                random_seed=seed_i
            )
            outcomes = generate_success_outcomes(
                tasks, threshold, slope, 
                config.correlation_model, config.correlation_strength,
                random_seed=seed_i
            )
            
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


def threshold_estimator(tasks: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Estimate the threshold parameter from tasks and outcomes.
    
    Args:
        tasks: Array of task difficulties
        outcomes: Array of binary success outcomes
        
    Returns:
        Estimated threshold parameter
    """
    from scipy.optimize import curve_fit
    
    # Skip if all success or all failure
    if len(np.unique(outcomes)) < 2:
        return float('nan')
        
    try:
        # Define logistic function for curve fitting
        def logistic_wrapper(x, threshold, slope):
            return logistic_function(x, threshold, slope)
        
        # Fit the curve
        params, _ = curve_fit(
            logistic_wrapper, 
            tasks, 
            outcomes,
            p0=[np.median(tasks), 1.0],
            bounds=([min(tasks), 0.01], [max(tasks), 10])
        )
        
        return params[0]  # threshold parameter
    except Exception as e:
        return float('nan')


def weighted_score_estimator(
    tasks: np.ndarray, 
    outcomes: np.ndarray, 
    weight_fn: Callable[[float], float] = lambda x: 1.0 + 0.5 * x
) -> float:
    """
    Calculate weighted average of success rates.
    
    Args:
        tasks: Array of task difficulties
        outcomes: Array of binary success outcomes
        weight_fn: Function that maps difficulty to weight
        
    Returns:
        Weighted score
    """
    # Calculate weights
    weights = np.array([weight_fn(diff) for diff in tasks])
    
    # Handle empty samples or all zero weights
    if len(outcomes) == 0 or np.sum(weights) == 0:
        return float('nan')
        
    # Calculate weighted score
    return float(np.sum(outcomes * weights) / np.sum(weights))
