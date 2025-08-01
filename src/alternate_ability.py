"""
Alternate Ability Functions

This module provides implementations of various alternate ability functions
that can be used in place of the standard logistic function for modeling
task success probabilities.
"""

import numpy as np
import logging
from typing import Dict, Any


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


def richards_function(x: np.ndarray, x0: float, k: float, nu: float, 
                     lower_asymptote: float = 0.0, upper_asymptote: float = 1.0) -> np.ndarray:
    """
    Calculate Richards generalized logistic function values.
    
    Args:
        x: Input values (difficulties)
        x0: Threshold parameter
        k: Slope parameter 
        nu: Asymmetry parameter
        lower_asymptote: Lower asymptote
        upper_asymptote: Upper asymptote
    
    Returns:
        Array of Richards function values
    """
    denom = (1 + nu * np.exp(-k * (x - x0))) ** (1.0 / nu)
    return lower_asymptote + (upper_asymptote - lower_asymptote) / denom


def richards_percentile(p: float, x0: float, k: float, nu: float, 
                       lower: float = 0.0, upper: float = 1.0) -> float:
    """
    Returns x such that Richards(x; x0,k,nu,lower,upper) == lower + p*(upper-lower).
    
    Args:
        p: Percentile (must be between 0 and 1). For 50% point, use p=0.5.
        x0: Threshold parameter
        k: Slope parameter
        nu: Asymmetry parameter
        lower: Lower asymptote
        upper: Upper asymptote
        
    Returns:
        The x value where the Richards function equals the given percentile
    """
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1")

    # Invert the normalized fraction p and apply the closed-form formula:
    # x = x0 - (1/k) * ln((p^(-nu) - 1) / nu)
    numerator = p**(-nu) - 1.0
    
    # Avoid division by zero if nu or numerator are problematic
    if numerator <= 0:
        raise ValueError("Invalid parameters lead to non-positive numerator")

    return x0 - (1.0 / k) * np.log(numerator / nu)


def exponential_function(x: np.ndarray, rate_param: float, convert_from_log2: bool = True) -> np.ndarray:
    """
    Calculate exponential survival function values.
    
    Args:
        x: Input values (difficulties)
        rate_param: Rate parameter (λ)
        convert_from_log2: Whether to convert from log2 space to linear space
    
    Returns:
        Array of exponential function values
    """
    if convert_from_log2:
        linear_difficulties = np.exp2(x)
    else:
        linear_difficulties = x
    return np.exp(-rate_param * linear_difficulties)


def power_law_function(x: np.ndarray, threshold: float, exponent: float) -> np.ndarray:
    """
    Calculate power law function values.
    
    Args:
        x: Input values (difficulties)
        threshold: Threshold parameter
        exponent: Power law exponent
    
    Returns:
        Array of power law function values
    """
    # Avoid division by zero and ensure positive values
    threshold_safe = max(threshold, 1e-6)
    ratio = np.maximum(x, 1e-6) / threshold_safe
    return ratio ** (-exponent)


def calculate_alternate_ability_threshold(alternate_ability_config: Dict[str, Any], 
                                          base_threshold: float,
                                          base_slope: float) -> float:
    """
    Calculate the effective 50% threshold for alternate ability functions.
    
    For non-logistic functions, this may differ from the base threshold parameter.
    
    Args:
        alternate_ability_config: Alternate ability configuration
        base_threshold: Base threshold from ability forecast
        base_slope: Base slope from ability forecast
        
    Returns:
        Effective 50% threshold for the alternate ability function
    """
    function_type = alternate_ability_config.get("function_type", "logistic")
    args = alternate_ability_config.get("args", {})
    
    if function_type == "richards_generalized_logistic":
        # For Richards curve: use the correct percentile formula
        x0 = args.get("threshold", base_threshold)
        k = args.get("slope", base_slope)
        # Handle both parameter names for Richards asymmetry parameter
        nu = args.get("nu", args.get("asymmetry", 1.0))
        lower_asymptote = args.get("lower_asymptote", 0.0)
        upper_asymptote = args.get("upper_asymptote", 1.0)
        
        try:
            # Use the correct Richards percentile formula for 50% point
            threshold_50 = richards_percentile(0.5, x0, k, nu, lower_asymptote, upper_asymptote)
            return threshold_50
            
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logging.warning(f"Could not calculate Richards 50% threshold with ν={nu}, k={k}: {e}")
            return x0
            
    elif function_type == "exponential":
        # Check if this is debug-style exponential (with asymptote parameter)
        if "asymptote" in args:
            # Debug script style: y = asymptote + (1 - asymptote) * exp(-decay_rate * max(0, x - threshold))
            # For 50% threshold: 0.5 = asymptote + (1 - asymptote) * exp(-decay_rate * max(0, x - threshold))
            # Solving: exp(-decay_rate * (x - threshold)) = (0.5 - asymptote) / (1 - asymptote)
            # x - threshold = -ln((0.5 - asymptote) / (1 - asymptote)) / decay_rate
            # x = threshold - ln((0.5 - asymptote) / (1 - asymptote)) / decay_rate
            threshold = args.get("threshold", base_threshold)
            decay_rate = args.get("decay_rate", args.get("rate_param", 0.1))
            asymptote = args.get("asymptote", 0.0)
            
            if decay_rate <= 0 or (1 - asymptote) <= 0:
                logging.warning(f"Invalid parameters for debug exponential: decay_rate={decay_rate}, asymptote={asymptote}")
                return threshold
                
            target_ratio = (0.5 - asymptote) / (1 - asymptote)
            if target_ratio <= 0:
                logging.warning(f"Invalid target ratio for debug exponential: {target_ratio}")
                return threshold
                
            threshold_50 = threshold - np.log(target_ratio) / decay_rate
            return threshold_50
        else:
            # Simulation style: For exponential: S(x) = exp(-λx) = 0.5 => x = ln(2)/λ
            rate_param = args.get("rate_param", args.get("decay_rate", 1.0))
            convert_from_log2 = args.get("convert_from_log2", True)
            
            if rate_param <= 0:
                logging.warning(f"Invalid rate parameter {rate_param} for exponential function")
                return base_threshold
                
            threshold_50_linear = np.log(2) / rate_param
            
            if convert_from_log2:
                # Convert back to log2 space
                threshold_50 = np.log2(threshold_50_linear)
            else:
                threshold_50 = threshold_50_linear
                
            return threshold_50
        
    elif function_type == "power_law":
        # Check if this is debug-style power law (with scale parameter)
        if "scale" in args:
            # Debug script style: For x >= threshold: y = scale * (x - threshold + 1)^exponent
            # For 50% threshold: 0.5 = scale * (x - threshold + 1)^exponent
            # (x - threshold + 1) = (0.5 / scale)^(1/exponent)
            # x = threshold - 1 + (0.5 / scale)^(1/exponent)
            threshold = args.get("threshold", base_threshold)
            exponent = args.get("exponent", -2.0)
            scale = args.get("scale", 1.0)
            
            if scale <= 0 or exponent == 0:
                logging.warning(f"Invalid parameters for debug power law: scale={scale}, exponent={exponent}")
                return threshold
                
            target_base = 0.5 / scale
            if target_base <= 0:
                logging.warning(f"Invalid target base for debug power law: {target_base}")
                return threshold
                
            threshold_50 = threshold - 1 + target_base ** (1.0 / exponent)
            return threshold_50
        else:
            # Simulation style: For power law: (x/x0)^(-α) = 0.5 => x = x0 * 2^(1/α)
            threshold = args.get("threshold", base_threshold)
            exponent = args.get("exponent", 1.0)
            
            if exponent <= 0:
                logging.warning(f"Invalid exponent {exponent} for power law function")
                return threshold
                
            threshold_50 = threshold * (2 ** (1/exponent))
            return threshold_50
        
    elif function_type == "logistic":
        # For logistic, use the provided or base threshold
        return args.get("threshold", base_threshold)
        
    else:
        logging.warning(f"Unknown alternate ability function type: {function_type}")
        return base_threshold


def evaluate_alternate_ability(x: np.ndarray, function_type: str, args: Dict[str, Any],
                              base_threshold: float = 20.0, base_slope: float = -0.665) -> np.ndarray:
    """
    Evaluate alternate ability function at given difficulty values.
    
    Args:
        x: Array of difficulty values
        function_type: Type of alternate ability function
        args: Function arguments
        base_threshold: Base threshold parameter (fallback)
        base_slope: Base slope parameter (fallback)
        
    Returns:
        Array of success probabilities
    """
    if function_type == "logistic":
        threshold = args.get("threshold", base_threshold)
        slope = args.get("slope", base_slope)
        return logistic_function(x, threshold, slope)
    
    elif function_type == "richards_generalized_logistic":
        x0 = args.get("threshold", base_threshold)
        k = args.get("slope", base_slope)
        # Handle both parameter names for Richards asymmetry parameter
        nu = args.get("nu", args.get("asymmetry", 1.0))
        lower_asymptote = args.get("lower_asymptote", 0.0)
        upper_asymptote = args.get("upper_asymptote", 1.0)
        
        return richards_function(x, x0, k, nu, lower_asymptote, upper_asymptote)
    
    elif function_type == "exponential":
        # Check if this is debug-style exponential (with asymptote parameter)
        if "asymptote" in args:
            # Debug script style: y = asymptote + (1 - asymptote) * exp(-decay_rate * max(0, x - threshold))
            threshold = args.get("threshold", base_threshold)
            decay_rate = args.get("decay_rate", args.get("rate_param", 0.1))
            asymptote = args.get("asymptote", 0.0)
            
            return asymptote + (1.0 - asymptote) * np.exp(-decay_rate * np.maximum(0, x - threshold))
        else:
            # Simulation style: survival function S(x) = exp(-λx)
            rate_param = args.get("rate_param", args.get("decay_rate", 1.0))
            convert_from_log2 = args.get("convert_from_log2", True)
            
            return exponential_function(x, rate_param, convert_from_log2)
    
    elif function_type == "power_law":
        # Check if this is debug-style power law (with scale parameter)
        if "scale" in args:
            # Debug script style: y = scale * (x - threshold + 1)^exponent for x >= threshold, 1 otherwise
            threshold = args.get("threshold", base_threshold)
            exponent = args.get("exponent", -2.0)
            scale = args.get("scale", 1.0)
            
            result = np.ones_like(x)
            mask = x >= threshold
            result[mask] = scale * np.power(x[mask] - threshold + 1.0, exponent)
            return np.clip(result, 0.0, 1.0)
        else:
            # Simulation style: (x/threshold)^(-exponent)
            threshold = args.get("threshold", base_threshold)
            exponent = args.get("exponent", 1.0)
            
            return power_law_function(x, threshold, exponent)
    
    else:
        raise ValueError(f"Unknown alternate ability function type: {function_type}")
