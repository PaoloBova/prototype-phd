"""
Forecast ability curves based on historical curve parameters.
"""
import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import prototype_phd.data_utils as data_utils
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from scipy import stats
from .schemas import LogisticFitParams, AbilityForecast

# Enums for configuration
class TrendType(str, Enum):
    """Types of forecasting trends."""
    LINEAR = "linear"
    CYCLIC = "cyclic"
    RANDOM_WALK = "random_walk"
    CONSTANT_SLOPE = "constant_slope"

# Data models for validation
class TrendEstimate(BaseModel):
    """Results of parameter trend estimation."""
    # Parameter estimates
    threshold_intercept: float = Field(..., description="Intercept for threshold trend")
    threshold_slope: float = Field(..., description="Slope for threshold trend")
    slope_intercept: float = Field(..., description="Intercept for slope trend")
    slope_slope: float = Field(..., description="Slope for slope trend")
    
    # Confidence intervals
    threshold_intercept_ci: Optional[Tuple[float, float]] = Field(None, description="95% CI for threshold intercept")
    threshold_slope_ci: Optional[Tuple[float, float]] = Field(None, description="95% CI for threshold slope")
    slope_intercept_ci: Optional[Tuple[float, float]] = Field(None, description="95% CI for slope intercept")
    slope_slope_ci: Optional[Tuple[float, float]] = Field(None, description="95% CI for slope slope")
    
    # Reference data
    base_date: datetime = Field(..., description="Base date for the trend")
    slope_scenarios: Dict[str, float] = Field(default_factory=dict, description="Different slope values")

class ForecastConfig(BaseModel):
    """Configuration for ability forecasts."""
    trend_type: TrendType = Field(TrendType.LINEAR, description="Type of trend to apply")
    start_date: datetime = Field(..., description="Start date for forecasts")
    end_date: datetime = Field(..., description="End date for forecasts")
    frequency: str = Field("QE", description="Frequency for forecast dates (QE=quarterly, ME=monthly, YE=yearly)")
    cycle_period: float = Field(3.0, description="Period in years for cyclic trends")
    cycle_amplitude: float = Field(0.5, description="Amplitude for cyclic trends")
    random_walk_std: float = Field(0.1, description="Standard deviation for random walk innovations")
    random_seed: int = Field(42, description="Random seed for reproducible forecasts")
    constant_slope_type: Optional[str] = Field("mean", description="Type of constant slope to use (mean, median, min, max, etc.)")
    
    class Config:
        arbitrary_types_allowed = True

# Multimethod implementation
def multi(dispatch_fn):
    """Create a multimethod dispatcher."""
    def _inner(*args, **kwargs):
        key = dispatch_fn(*args, **kwargs)
        fn = _inner.__multi__.get(key, _inner.__multi_default__)
        return fn(*args, **kwargs)
    _inner.__dispatch_fn__ = dispatch_fn
    _inner.__multi__ = {}
    _inner.__multi_default__ = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError(f"Unsupported type: {dispatch_fn(*args, **kwargs)}"))
    return _inner

def method(dispatch_fn, dispatch_key=None):
    """Register a function with a multimethod."""
    def apply_decorator(fn):
        if dispatch_key is None:
            dispatch_fn.__multi_default__ = fn
        else:
            dispatch_fn.__multi__[dispatch_key] = fn
        return dispatch_fn
    return apply_decorator

# Multimethod for trend estimation
def trend_type_dispatch(config: ForecastConfig, *args, **kwargs):
    """Dispatch based on trend type from config."""
    return config.trend_type

forecast_trend = multi(trend_type_dispatch)

# Common utility functions
def calculate_slope_scenarios(slopes: np.ndarray) -> Dict[str, float]:
    """Calculate various slope scenarios from historical data."""
    return {
        "mean": float(np.mean(slopes)),
        "median": float(np.median(slopes)),
        "min": float(np.min(slopes)),
        "max": float(np.max(slopes)),
        "p25": float(np.percentile(slopes, 25)) if len(slopes) >= 4 else float(np.min(slopes)),
        "p75": float(np.percentile(slopes, 75)) if len(slopes) >= 4 else float(np.max(slopes))
    }

def estimate_confidence_intervals(
    years: np.ndarray, 
    values: np.ndarray, 
    model: LinearRegression
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Calculate confidence intervals for intercept and slope if enough data points."""
    n = len(years)
    if n <= 2:  # Need at least 3 points for confidence intervals
        return None, None
        
    t_value = stats.t.ppf(0.975, n-2)
    
    # Calculate standard errors
    y_pred = model.predict(years)
    residuals = values - y_pred
    mse = np.sum(residuals**2) / (n - 2)
    intercept_se = np.sqrt(mse * (1/n + np.mean(years)**2 / np.sum((years - np.mean(years))**2)))
    slope_se = np.sqrt(mse / np.sum((years - np.mean(years))**2))
    
    intercept = model.intercept_
    slope = model.coef_[0]
    
    # Calculate confidence intervals
    intercept_ci = (
        float(intercept - t_value * intercept_se),
        float(intercept + t_value * intercept_se)
    )
    slope_ci = (
        float(slope - t_value * slope_se),
        float(slope + t_value * slope_se)
    )
    
    return intercept_ci, slope_ci

def estimate_linear_trend_core(
    dates: List[datetime], 
    thresholds: List[float], 
    slopes: List[float]
) -> TrendEstimate:
    """Core implementation of linear trend estimation."""
    # Extract dates as years since earliest date
    base_date = min(dates)
    years = np.array([(d - base_date).total_seconds() / (365.25 * 24 * 3600) 
                     for d in dates]).reshape(-1, 1)
    
    # Fit linear regression for threshold trend
    threshold_model = LinearRegression()
    threshold_model.fit(years, thresholds)
    threshold_intercept = float(threshold_model.intercept_)
    threshold_slope = float(threshold_model.coef_[0])
    
    # Fit linear regression for slope trend
    slope_model = LinearRegression()
    slope_model.fit(years, slopes)
    slope_intercept = float(slope_model.intercept_)
    slope_slope = float(slope_model.coef_[0])
    
    # Calculate confidence intervals
    threshold_intercept_ci, threshold_slope_ci = estimate_confidence_intervals(
        years, thresholds, threshold_model
    )
    
    slope_intercept_ci, slope_slope_ci = estimate_confidence_intervals(
        years, slopes, slope_model
    )
    
    # Calculate slope scenarios
    slope_scenarios = calculate_slope_scenarios(np.array(slopes))
    
    return TrendEstimate(
        threshold_intercept=threshold_intercept,
        threshold_slope=threshold_slope,
        slope_intercept=slope_intercept,
        slope_slope=slope_slope,
        threshold_intercept_ci=threshold_intercept_ci,
        threshold_slope_ci=threshold_slope_ci,
        slope_intercept_ci=slope_intercept_ci,
        slope_slope_ci=slope_slope_ci,
        base_date=base_date,
        slope_scenarios=slope_scenarios
    )

# Multimethod implementations for different trend types
@method(forecast_trend, TrendType.LINEAR)
def forecast_linear_trend(
    config: ForecastConfig, 
    historical_params: Dict[str, LogisticFitParams],
    forecast_dates: List[datetime]
) -> List[AbilityForecast]:
    """Generate forecasts using linear trends for both threshold and slope."""
    # Extract historical parameters
    params_list = list(historical_params.values())
    params_list.sort(key=lambda p: p.date)
    dates = [p.date for p in params_list]
    thresholds = [p.threshold for p in params_list]
    slopes = [p.slope for p in params_list]
    
    # Estimate trends
    trend = estimate_linear_trend_core(dates, thresholds, slopes)
    
    # Generate forecasts for each date
    forecasts = []
    for date in forecast_dates:
        years_delta = (date - trend.base_date).total_seconds() / (365.25 * 24 * 3600)
        
        # Calculate forecasted parameters
        new_threshold = trend.threshold_intercept + trend.threshold_slope * years_delta
        new_slope = trend.slope_intercept + trend.slope_slope * years_delta
        
        # Calculate confidence intervals if available
        threshold_ci = None
        if trend.threshold_intercept_ci and trend.threshold_slope_ci:
            threshold_ci_lower = trend.threshold_intercept_ci[0] + trend.threshold_slope_ci[0] * years_delta
            threshold_ci_upper = trend.threshold_intercept_ci[1] + trend.threshold_slope_ci[1] * years_delta
            threshold_ci = (float(threshold_ci_lower), float(threshold_ci_upper))
        
        slope_ci = None
        if trend.slope_intercept_ci and trend.slope_slope_ci:
            slope_ci_lower = trend.slope_intercept_ci[0] + trend.slope_slope_ci[0] * years_delta
            slope_ci_upper = trend.slope_intercept_ci[1] + trend.slope_slope_ci[1] * years_delta
            slope_ci = (float(slope_ci_lower), float(slope_ci_upper))
        
        # Ensure slope is negative
        avg_slope = np.mean(slopes)
        if new_slope >= 0:
            new_slope = avg_slope if avg_slope < 0 else -abs(avg_slope)
            
        # Create forecast
        forecast_data = {
            "date": date,
            "threshold": float(new_threshold),
            "slope": float(new_slope),
            "scenario": f"{config.trend_type.value}_{config.frequency}",
            "model": f"future_model_{date.strftime('%Y%m%d')}"
        }
        
        # Add confidence intervals if available
        if threshold_ci:
            forecast_data["threshold_ci_lower"] = threshold_ci[0]
            forecast_data["threshold_ci_upper"] = threshold_ci[1]
        
        if slope_ci:
            forecast_data["slope_ci_lower"] = slope_ci[0]
            forecast_data["slope_ci_upper"] = slope_ci[1]
            
        forecast = AbilityForecast(**forecast_data)
        forecasts.append(forecast)
    
    return forecasts

@method(forecast_trend, TrendType.CONSTANT_SLOPE)
def forecast_constant_slope_trend(
    config: ForecastConfig, 
    historical_params: Dict[str, LogisticFitParams],
    forecast_dates: List[datetime]
) -> List[AbilityForecast]:
    """Generate forecasts using linear trend for threshold but constant slope."""
    # Extract historical parameters
    params_list = list(historical_params.values())
    params_list.sort(key=lambda p: p.date)
    dates = [p.date for p in params_list]
    thresholds = [p.threshold for p in params_list]
    slopes = [p.slope for p in params_list]
    
    # Estimate trends (using same function as linear, but we'll use a constant slope)
    trend = estimate_linear_trend_core(dates, thresholds, slopes)
    
    # Get the constant slope value based on specified type
    slope_type = config.constant_slope_type or "mean"
    constant_slope = trend.slope_scenarios.get(slope_type, trend.slope_scenarios["mean"])
    
    # Generate forecasts for each date
    forecasts = []
    for date in forecast_dates:
        years_delta = (date - trend.base_date).total_seconds() / (365.25 * 24 * 3600)
        
        # Calculate forecasted threshold with linear trend
        new_threshold = trend.threshold_intercept + trend.threshold_slope * years_delta
        
        # Use constant slope
        new_slope = constant_slope
            
        # Create forecast
        forecast = AbilityForecast(
            date=date,
            threshold=float(new_threshold),
            slope=float(new_slope),
            scenario=f"{config.trend_type.value}_{slope_type}_{config.frequency}",
            model=f"future_model_{date.strftime('%Y%m%d')}"
        )
        forecasts.append(forecast)
    
    return forecasts

@method(forecast_trend, TrendType.CYCLIC)
def forecast_cyclic_trend(
    config: ForecastConfig, 
    historical_params: Dict[str, LogisticFitParams],
    forecast_dates: List[datetime]
) -> List[AbilityForecast]:
    """Generate forecasts using cyclic trends for threshold."""
    # Extract historical parameters
    params_list = list(historical_params.values())
    params_list.sort(key=lambda p: p.date)
    dates = [p.date for p in params_list]
    thresholds = [p.threshold for p in params_list]
    slopes = [p.slope for p in params_list]
    
    # Estimate base linear trends
    trend = estimate_linear_trend_core(dates, thresholds, slopes)
    
    # Generate forecasts for each date
    forecasts = []
    for date in forecast_dates:
        years_delta = (date - trend.base_date).total_seconds() / (365.25 * 24 * 3600)
        
        # Calculate cyclic component
        cycle_component = config.cycle_amplitude * np.sin(2 * np.pi * years_delta / config.cycle_period)
        
        # Calculate forecasted parameters
        new_threshold = trend.threshold_intercept + trend.threshold_slope * years_delta + cycle_component
        new_slope = trend.slope_intercept + trend.slope_slope * years_delta
        
        # Ensure slope is negative
        avg_slope = np.mean(slopes)
        if new_slope >= 0:
            new_slope = avg_slope if avg_slope < 0 else -abs(avg_slope)
            
        # Create forecast
        forecast = AbilityForecast(
            date=date,
            threshold=float(new_threshold),
            slope=float(new_slope),
            scenario=f"{config.trend_type.value}_{config.frequency}",
            model=f"future_model_{date.strftime('%Y%m%d')}"
        )
        forecasts.append(forecast)
    
    return forecasts

@method(forecast_trend, TrendType.RANDOM_WALK)
def forecast_random_walk_trend(
    config: ForecastConfig, 
    historical_params: Dict[str, LogisticFitParams],
    forecast_dates: List[datetime]
) -> List[AbilityForecast]:
    """Generate forecasts using random walk with drift."""
    # Extract historical parameters
    params_list = list(historical_params.values())
    params_list.sort(key=lambda p: p.date)
    dates = [p.date for p in params_list]
    thresholds = [p.threshold for p in params_list]
    slopes = [p.slope for p in params_list]
    
    # Estimate base linear trends
    trend = estimate_linear_trend_core(dates, thresholds, slopes)
    
    # Set random seed for reproducibility
    np.random.seed(config.random_seed)
    
    # Generate forecasts iteratively (random walk depends on prior forecasts)
    forecasts = []
    last_threshold = None
    last_slope = None
    last_date = None
    
    for date in forecast_dates:
        years_delta = (date - trend.base_date).total_seconds() / (365.25 * 24 * 3600)
        
        if last_threshold is None:
            # First forecast - start from linear trend
            new_threshold = trend.threshold_intercept + trend.threshold_slope * years_delta + \
                np.random.normal(0, config.random_walk_std)
            new_slope = trend.slope_intercept + trend.slope_slope * years_delta + \
                np.random.normal(0, config.random_walk_std)
        else:
            # Continue from last forecast
            years_since_last = (date - last_date).total_seconds() / (365.25 * 24 * 3600)
            
            # Drift + random innovation
            new_threshold = last_threshold + trend.threshold_slope * years_since_last + \
                np.random.normal(0, config.random_walk_std * np.sqrt(years_since_last))
            new_slope = last_slope + trend.slope_slope * years_since_last + \
                np.random.normal(0, config.random_walk_std * np.sqrt(years_since_last))
        
        # Ensure slope is negative
        avg_slope = np.mean(slopes)
        if new_slope >= 0:
            new_slope = avg_slope if avg_slope < 0 else -abs(avg_slope)
            
        # Create forecast
        forecast = AbilityForecast(
            date=date,
            threshold=float(new_threshold),
            slope=float(new_slope),
            scenario=f"{config.trend_type.value}_{config.frequency}",
            model=f"future_model_{date.strftime('%Y%m%d')}"
        )
        forecasts.append(forecast)
        
        # Update last values for next iteration
        last_threshold = new_threshold
        last_slope = new_slope
        last_date = date
    
    return forecasts

# Main functions
def generate_forecasts(
    historical_params: Dict[str, LogisticFitParams], 
    config: ForecastConfig
) -> List[AbilityForecast]:
    """
    Generate forecasts based on historical parameters and configuration.
    
    Args:
        historical_params: Dictionary mapping model names to LogisticFitParams
        config: Forecast configuration
    
    Returns:
        List of AbilityForecast objects for future models
    """
    # Create date range for forecast
    forecast_dates = pd.date_range(start=config.start_date, end=config.end_date, freq=config.frequency)
    
    # Use the multimethod to dispatch based on trend type
    return forecast_trend(config, historical_params, forecast_dates)

def read_curve_params(file_path: str) -> Dict[str, List[LogisticFitParams]]:
    """Read curve parameters from JSON file."""
    with open(file_path, "r") as f:
        params_dict = json.load(f)
    
    # Convert to LogisticFitParams objects
    params = {}
    for model, model_params in params_dict.items():
        try:
            # Parse date string to datetime
            model_params["date"] = datetime.fromisoformat(model_params["date"].split("+")[0])
            params[model] = LogisticFitParams(**model_params)
        except Exception as e:
            logging.error(f"Error parsing parameters for model {model}: {e}")
    
    return params

def save_forecasts(forecasts: List[AbilityForecast], output_path: str):
    """Save forecasts to CSV file."""
    # Convert to DataFrame
    records = [f.model_dump() for f in forecasts]
    df = pd.DataFrame(records)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forecast ability curves")
    parser.add_argument("--params", required=True, help="Path to curve parameters JSON")
    parser.add_argument("--config", required=True, help="Path to forecast config JSON")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    return parser.parse_args()

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Reading curve parameters from {args.params}")
    curve_params = read_curve_params(args.params)
    logging.info(f"Read parameters for {len(curve_params)} models")
    
    logging.info(f"Reading forecast config from {args.config}")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    # Convert date strings to datetime objects
    for date_field in ['start_date', 'end_date']:
        if date_field in config_dict and isinstance(config_dict[date_field], str):
            config_dict[date_field] = datetime.fromisoformat(config_dict[date_field].replace('Z', '+00:00'))
    
    # Generate forecasts for each trend type and frequency
    all_forecasts = []
    
    for trend_type in TrendType:
        for freq in ['YE', 'QE', 'ME']:
            # Create a config for this combination
            trend_config_args = {**config_dict, 'trend_type': trend_type, 'frequency': freq}
            
            if trend_type == TrendType.CONSTANT_SLOPE:
                # Generate different constant slope scenarios
                for slope_type in ["mean", "median", "min", "max", "p25", "p75"]:
                    slope_config_args = {**trend_config_args, 'constant_slope_type': slope_type}
                    trend_config = ForecastConfig(**slope_config_args)
                    
                    logging.info(f"Generating {trend_type.value} ({slope_type}) forecasts at {freq} frequency")
                    forecasts = generate_forecasts(curve_params, trend_config)
                    all_forecasts.extend(forecasts)
            else:
                trend_config = ForecastConfig(**trend_config_args)
                
                logging.info(f"Generating {trend_type.value} forecasts at {freq} frequency")
                forecasts = generate_forecasts(curve_params, trend_config)
                all_forecasts.extend(forecasts)
    
    logging.info(f"Generated {len(all_forecasts)} forecasts")
    
    logging.info(f"Saving forecasts to {args.out}")
    save_forecasts(all_forecasts, args.out)
    logging.info("Complete")

if __name__ == "__main__":
    main()
