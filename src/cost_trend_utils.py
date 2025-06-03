"""
Utilities for aggregating and analyzing cost trends.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from scipy import stats
from .schemas import CostTrend

def aggregate_doubling_rates(doubling_rates: List[float], 
                             method: str = "geometric_mean",
                             weights: Optional[List[float]] = None) -> float:
    """
    Aggregate doubling rates using statistically sound methods.
    
    Args:
        doubling_rates: List of doubling rates to aggregate
        method: Aggregation method ('arithmetic_mean', 'geometric_mean', 
                'median', 'trimmed_mean', 'weighted')
        weights: Optional weights for weighted aggregation
        
    Returns:
        Aggregated doubling rate
    """
    if len(doubling_rates) == 0:
        raise ValueError("Cannot aggregate empty list of doubling rates")
        
    # Remove any non-finite values
    valid_rates = [r for r in doubling_rates if np.isfinite(r) and r > 0]
    
    if len(valid_rates) == 0:
        raise ValueError("No valid doubling rates to aggregate")
        
    if method == "arithmetic_mean":
        return float(np.mean(valid_rates))
    elif method == "geometric_mean":
        return float(stats.gmean(valid_rates))
    elif method == "median":
        return float(np.median(valid_rates))
    elif method == "trimmed_mean":
        # Remove top and bottom 10%
        return float(stats.trim_mean(valid_rates, 0.1))
    elif method == "weighted":
        if weights is None or len(weights) != len(valid_rates):
            raise ValueError("Weights must be provided and match doubling rates length")
        return float(np.average(valid_rates, weights=weights))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def analyze_doubling_rates(doubling_rates: List[float], 
                           confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Analyze doubling rates with multiple statistical methods and confidence intervals.
    
    Args:
        doubling_rates: List of doubling rates
        confidence_level: Confidence level for intervals (0-1)
        
    Returns:
        Dictionary of statistics and confidence intervals
    """
    valid_rates = [r for r in doubling_rates if np.isfinite(r) and r > 0]
    n = len(valid_rates)
    
    if n < 2:
        return {"warning": "Insufficient data for statistical analysis"}
    
    # Log transform for better statistical properties
    log_rates = np.log(valid_rates)
    
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha/2, n-1)
    
    # Calculate statistics in log space, then transform back
    log_mean = np.mean(log_rates)
    log_std = np.std(log_rates, ddof=1)
    log_se = log_std / np.sqrt(n)
    log_lower = log_mean - t_crit * log_se
    log_upper = log_mean + t_crit * log_se
    
    return {
        "arithmetic_mean": float(np.mean(valid_rates)),
        "geometric_mean": float(stats.gmean(valid_rates)),
        "median": float(np.median(valid_rates)),
        "trimmed_mean": float(stats.trim_mean(valid_rates, 0.1)),
        "min": float(np.min(valid_rates)),
        "max": float(np.max(valid_rates)),
        "std_dev": float(np.std(valid_rates, ddof=1)),
        "geom_mean_ci_lower": float(np.exp(log_lower)),
        "geom_mean_ci_upper": float(np.exp(log_upper)),
        "sample_size": n
    }

def calculate_weighted_doubling_rate(cost_trends: List[CostTrend], 
                                     weighting: str = "r_squared") -> float:
    """
    Calculate weighted doubling rate from multiple cost trends.
    
    Args:
        cost_trends: List of CostTrend objects
        weighting: Method to weight trends ('r_squared', 'uniform', 'inverse_variance')
        
    Returns:
        Weighted doubling rate
    """
    if not cost_trends:
        raise ValueError("No cost trends provided")
        
    doubling_rates = [ct.doubling_rate for ct in cost_trends]
    
    if weighting == "uniform":
        weights = [1.0] * len(doubling_rates)
    elif weighting == "r_squared":
        # Weight by goodness of fit
        weights = [max(0.01, ct.r_squared) for ct in cost_trends]
    elif weighting == "inverse_variance":
        # If you have uncertainty estimates, use inverse variance weighting
        variances = [(1.0 / max(0.01, ct.r_squared)) for ct in cost_trends]
        weights = [1.0 / max(0.01, v) for v in variances]
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")
        
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    return aggregate_doubling_rates(doubling_rates, method="weighted", weights=normalized_weights)
