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
from typing import Dict, List, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from .schemas import LogisticFitParams, AbilityForecast

class TrendType(str, Enum):
    LINEAR = "linear"
    CYCLIC = "cyclic"
    RANDOM_WALK = "random_walk"

class ForecastConfig(BaseModel):
    """Configuration for ability forecasts."""
    trend_type: TrendType = Field(TrendType.LINEAR, description="Type of trend to apply")
    start_date: datetime = Field(..., description="Start date for forecasts")
    end_date: datetime = Field(..., description="End date for forecasts")
    frequency: str = Field("Q", description="Frequency for forecast dates (Q=quarterly, M=monthly, Y=yearly)")
    cycle_period: float = Field(3.0, description="Period in years for cyclic trends")
    cycle_amplitude: float = Field(0.5, description="Amplitude for cyclic trends")
    random_walk_std: float = Field(0.1, description="Standard deviation for random walk innovations")
    random_seed: int = Field(42, description="Random seed for reproducible forecasts")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forecast ability curves")
    parser.add_argument("--params", required=True, help="Path to curve parameters JSON")
    parser.add_argument("--config", required=True, help="Path to forecast config JSON")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    return parser.parse_args()

def read_curve_params(file_path: str) -> Dict[str, List[LogisticFitParams]]:
    """
    Read curve parameters from JSON file.
    
    Returns a dictionary where keys are model names and values are 
    LogisticFitParams objects.
    """
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

def estimate_parameter_trends(all_params: Dict[str, LogisticFitParams]) -> Tuple[float, float, float, float]:
    """
    Estimate linear trends in threshold and slope parameters across all models.
    
    Args:
        all_params: Dictionary mapping model names to their LogisticFitParams
    
    Returns:
        Tuple of (threshold_intercept, threshold_slope, slope_intercept, slope_slope)
    """
    # Convert dictionary to flat list of parameters
    params_list = list(all_params.values())
    
    if len(params_list) < 2:
        logging.warning("Not enough data points to estimate trends, using default values")
        return 0.0, 0.5, 1.0, 0.0  # Default values if not enough data
    
    # Sort by date for cleaner logging
    params_list.sort(key=lambda p: p.date)
    
    # Extract dates (as years since earliest date), thresholds, and slopes
    base_date = min(p.date for p in params_list)
    dates = np.array([(p.date - base_date).total_seconds() / (365.25 * 24 * 3600) 
                     for p in params_list]).reshape(-1, 1)  # Convert to years from first date
    thresholds = np.array([p.threshold for p in params_list])
    slopes = np.array([p.slope for p in params_list])
    
    # Fit linear regression for threshold trend
    threshold_model = LinearRegression()
    threshold_model.fit(dates, thresholds)
    threshold_intercept = threshold_model.intercept_
    threshold_slope = threshold_model.coef_[0]  # Annual change in threshold
    
    # Fit linear regression for slope trend
    slope_model = LinearRegression()
    slope_model.fit(dates, slopes)
    slope_intercept = slope_model.intercept_
    slope_slope = slope_model.coef_[0]  # Annual change in slope
    
    logging.info(f"Models analyzed: {len(params_list)}")
    logging.info(f"Date range: {params_list[0].date} to {params_list[-1].date}")
    logging.info(f"Threshold range: {min(thresholds):.2f} to {max(thresholds):.2f}")
    logging.info(f"Estimated threshold trend: {threshold_intercept:.2f} + {threshold_slope:.2f}t per year")
    logging.info(f"Estimated slope trend: {slope_intercept:.2f} + {slope_slope:.2f}t per year")
    
    return threshold_intercept, threshold_slope, slope_intercept, slope_slope

def generate_forecasts(
    all_params: Dict[str, LogisticFitParams], 
    config: ForecastConfig
) -> List[AbilityForecast]:
    """
    Generate forecasts based on parameter trends and forecast configuration.
    
    Args:
        all_params: Dictionary mapping model names to their LogisticFitParams
        config: Forecast configuration
    
    Returns:
        List of AbilityForecast objects for future hypothetical models
    """
    # Estimate parameter trends across all models
    threshold_intercept, threshold_slope, slope_intercept, slope_slope = estimate_parameter_trends(all_params)
    
    # Base date for calculating years_delta
    base_date = min(p.date for p in all_params.values())
    logging.info(f"Base date for forecasting: {base_date}")
    
    # Create date range for forecast
    forecast_dates = pd.date_range(start=config.start_date, end=config.end_date, freq=config.frequency)
    
    # Set random seed for reproducibility
    np.random.seed(config.random_seed)
    
    forecasts = []
    
    # Generate forecasts for each date
    for date in forecast_dates:
        # Calculate years since base date
        years_delta = (date - base_date).total_seconds() / (365.25 * 24 * 3600)
        
        # Calculate new parameters based on trend type
        if config.trend_type == TrendType.LINEAR:
            # Simple linear trend
            new_threshold = threshold_intercept + threshold_slope * years_delta
            new_slope = slope_intercept + slope_slope * years_delta
            
        elif config.trend_type == TrendType.CYCLIC:
            # Linear trend with cyclic component
            cycle_component = config.cycle_amplitude * np.sin(2 * np.pi * years_delta / config.cycle_period)
            new_threshold = threshold_intercept + threshold_slope * years_delta + cycle_component
            new_slope = slope_intercept + slope_slope * years_delta
            
        elif config.trend_type == TrendType.RANDOM_WALK:
            # Random walk with drift
            # Start from the latest predicted values
            prior_forecasts = [f for f in forecasts if f.scenario == f"{config.trend_type.value}_{config.frequency}"]
            
            if prior_forecasts:
                # Get the most recent forecast
                latest = max(prior_forecasts, key=lambda f: f.date)
                years_since_latest = (date - latest.date).total_seconds() / (365.25 * 24 * 3600)
                
                # Add drift component plus random noise
                new_threshold = latest.threshold + threshold_slope * years_since_latest + \
                    np.random.normal(0, config.random_walk_std * np.sqrt(years_since_latest))
                new_slope = latest.slope + slope_slope * years_since_latest + \
                    np.random.normal(0, config.random_walk_std * np.sqrt(years_since_latest))
            else:
                # First point, start from the trend
                new_threshold = threshold_intercept + threshold_slope * years_delta + \
                    np.random.normal(0, config.random_walk_std)
                new_slope = slope_intercept + slope_slope * years_delta + \
                    np.random.normal(0, config.random_walk_std)
        
        # Ensure slope is positive (logistic curve requirement)
        new_slope = max(0.01, new_slope)
        
        # Create forecast object
        forecast = AbilityForecast(
            date=date,
            threshold=float(new_threshold),
            slope=float(new_slope),
            scenario=f"{config.trend_type.value}_{config.frequency}",
            model=f"future_model_{date.strftime('%Y%m%d')}" # Name the hypothetical future model by its release date
        )
        
        forecasts.append(forecast)
    
    return forecasts

def save_forecasts(forecasts: List[AbilityForecast], output_path: str):
    """Save forecasts to CSV file."""
    # Convert to DataFrame
    records = [f.model_dump() for f in forecasts]
    df = pd.DataFrame(records)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

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
