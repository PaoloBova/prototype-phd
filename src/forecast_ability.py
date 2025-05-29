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
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from scipy import stats
from .schemas import LogisticFitParams, AbilityForecast

# Define trend types for different parameters
class ParameterTrendType(str, Enum):
    """Base enum for parameter trend types."""
    LINEAR = "linear"
    CONSTANT = "constant"
    CYCLIC = "cyclic"
    RANDOM_WALK = "random_walk"

# Define the parameter trend data models
class ParameterTrend(BaseModel):
    """Base class for parameter trends."""
    param_name: str = Field(..., description="Name of the parameter")
    trend_type: ParameterTrendType = Field(..., description="Type of trend")
    
    # Methods each trend must implement
    def forecast(self, base_date: datetime, target_date: datetime, 
                 random_state: Optional[np.random.RandomState] = None) -> float:
        """Generate a forecast for the parameter at the target date."""
        raise NotImplementedError("Subclasses must implement forecast method")

class LinearTrend(ParameterTrend):
    """Linear trend for a parameter."""
    intercept: float = Field(..., description="Intercept of the linear trend")
    slope: float = Field(..., description="Slope of the linear trend (change per year)")
    intercept_ci: Optional[Tuple[float, float]] = Field(None, description="Confidence interval for intercept")
    slope_ci: Optional[Tuple[float, float]] = Field(None, description="Confidence interval for slope")
    
    def forecast(self, base_date: datetime, target_date: datetime, 
                 random_state: Optional[np.random.RandomState] = None) -> float:
        """Generate a forecast based on linear trend."""
        years_delta = (target_date - base_date).total_seconds() / (365.25 * 24 * 3600)
        return self.intercept + self.slope * years_delta
    
    def forecast_ci(self, base_date: datetime, target_date: datetime) -> Optional[Tuple[float, float]]:
        """Generate confidence interval for the forecast if available."""
        if self.intercept_ci is None or self.slope_ci is None:
            return None
            
        years_delta = (target_date - base_date).total_seconds() / (365.25 * 24 * 3600)
        lower = self.intercept_ci[0] + self.slope_ci[0] * years_delta
        upper = self.intercept_ci[1] + self.slope_ci[1] * years_delta
        return (lower, upper)

class ConstantTrend(ParameterTrend):
    """Constant trend for a parameter."""
    value: float = Field(..., description="Constant value for the parameter")
    variance: Optional[float] = Field(None, description="Variance of the parameter value")
    
    def forecast(self, base_date: datetime, target_date: datetime, 
                 random_state: Optional[np.random.RandomState] = None) -> float:
        """Generate a forecast with constant value."""
        return self.value

class CyclicTrend(ParameterTrend):
    """Cyclic trend with underlying linear trend for a parameter."""
    base_trend: LinearTrend = Field(..., description="Underlying linear trend")
    period: float = Field(..., description="Period of the cycle in years")
    amplitude: float = Field(..., description="Amplitude of the cycle")
    
    def forecast(self, base_date: datetime, target_date: datetime, 
                 random_state: Optional[np.random.RandomState] = None) -> float:
        """Generate a forecast based on linear trend with cyclic component."""
        years_delta = (target_date - base_date).total_seconds() / (365.25 * 24 * 3600)
        linear_component = self.base_trend.forecast(base_date, target_date)
        cyclic_component = self.amplitude * np.sin(2 * np.pi * years_delta / self.period)
        return linear_component + cyclic_component

class RandomWalkTrend(ParameterTrend):
    """Random walk trend for a parameter."""
    base_trend: LinearTrend = Field(..., description="Underlying drift trend")
    std_dev: float = Field(..., description="Standard deviation of random innovations")
    last_value: Optional[float] = Field(None, description="Last forecasted value")
    last_date: Optional[datetime] = Field(None, description="Date of last forecast")
    
    def forecast(self, base_date: datetime, target_date: datetime, 
                 random_state: Optional[np.random.RandomState] = None) -> float:
        """Generate a forecast based on random walk with drift."""
        if random_state is None:
            random_state = np.random.RandomState()
            
        if self.last_value is None or self.last_date is None:
            # First forecast - start from base trend
            value = self.base_trend.forecast(base_date, target_date)
            innovation = random_state.normal(0, self.std_dev)
            forecast_value = value + innovation
        else:
            # Continue from last forecast
            years_delta = (target_date - self.last_date).total_seconds() / (365.25 * 24 * 3600)
            drift = self.base_trend.slope * years_delta
            innovation = random_state.normal(0, self.std_dev * np.sqrt(years_delta))
            forecast_value = self.last_value + drift + innovation
            
        # Update state for next forecast
        self._update_state(target_date, forecast_value)
        return forecast_value
    
    def _update_state(self, date: datetime, value: float):
        """Update the internal state with the new forecast."""
        self.last_date = date
        self.last_value = value

class AbilityTrend(BaseModel):
    """Container for threshold and slope trends."""
    threshold_trend: Union[LinearTrend, ConstantTrend, CyclicTrend, RandomWalkTrend] = Field(
        ..., description="Trend for threshold parameter")
    slope_trend: Union[LinearTrend, ConstantTrend, CyclicTrend, RandomWalkTrend] = Field(
        ..., description="Trend for slope parameter")
    base_date: datetime = Field(..., description="Base date for the trends")
    historic_dates: List[datetime] = Field(default_factory=list, description="Dates of historical data points")
    historic_thresholds: List[float] = Field(default_factory=list, description="Historical threshold values")
    historic_slopes: List[float] = Field(default_factory=list, description="Historical slope values")

class ForecastConfig(BaseModel):
    """Configuration for ability forecasts."""
    threshold_trend_type: ParameterTrendType = Field(ParameterTrendType.LINEAR, description="Trend type for threshold")
    slope_trend_type: ParameterTrendType = Field(ParameterTrendType.CONSTANT, description="Trend type for slope")
    slope_constant_type: Optional[str] = Field("mean", description="Type of constant slope value if using CONSTANT trend")
    start_date: datetime = Field(..., description="Start date for forecasts")
    end_date: datetime = Field(..., description="End date for forecasts")
    frequency: str = Field("QE", description="Frequency for forecast dates")
    cycle_period: float = Field(3.0, description="Period in years for cyclic trends")
    cycle_amplitude: float = Field(0.5, description="Amplitude for cyclic trends")
    random_walk_std: float = Field(0.1, description="Standard deviation for random walk innovations")
    random_seed: int = Field(42, description="Random seed for reproducibility")

def estimate_linear_trend(dates: List[datetime], values: List[float]) -> Tuple[LinearTrend, Dict[str, Any]]:
    """
    Estimate a linear trend from historical data.
    
    Args:
        dates: List of dates for historical data points
        values: List of parameter values corresponding to the dates
    
    Returns:
        Tuple of (LinearTrend object, metadata dictionary)
    """
    if len(dates) < 2:
        return LinearTrend(
            param_name="unknown",
            trend_type=ParameterTrendType.LINEAR,
            intercept=values[0] if values else 0.0,
            slope=0.0
        ), {}
    
    # Convert dates to years since earliest date
    base_date = min(dates)
    years = np.array([(d - base_date).total_seconds() / (365.25 * 24 * 3600) 
                    for d in dates]).reshape(-1, 1)
    vals = np.array(values)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(years, vals)
    intercept = float(model.intercept_)
    slope = float(model.coef_[0])  # Annual change
    
    # Calculate confidence intervals if enough data points
    intercept_ci = None
    slope_ci = None
    metadata = {}
    
    n = len(years)
    if n > 2:  # Need at least 3 points for confidence intervals
        t_value = stats.t.ppf(0.975, n-2)
        
        # Calculate standard errors
        residuals = vals - model.predict(years)
        mse = np.sum(residuals**2) / (n - 2)
        intercept_se = np.sqrt(mse * (1/n + np.mean(years)**2 / np.sum((years - np.mean(years))**2)))
        slope_se = np.sqrt(mse / np.sum((years - np.mean(years))**2))
        
        intercept_ci = (
            float(intercept - t_value * intercept_se[0]),
            float(intercept + t_value * intercept_se[0])
        )
        slope_ci = (
            float(slope - t_value * slope_se),
            float(slope + t_value * slope_se)
        )
        
        metadata = {
            "r_squared": float(model.score(years, vals)),
            "mse": float(mse),
            "residuals": residuals.tolist()
        }
    
    return LinearTrend(
        param_name="unknown",
        trend_type=ParameterTrendType.LINEAR,
        intercept=intercept,
        slope=slope,
        intercept_ci=intercept_ci,
        slope_ci=slope_ci
    ), metadata

def estimate_constant_trend(values: List[float], method: str = "mean") -> ConstantTrend:
    """
    Estimate a constant trend using various methods.
    
    Args:
        values: List of parameter values
        method: Method to compute the constant value ("mean", "median", "min", "max", "p25", "p75")
        
    Returns:
        ConstantTrend object
    """
    if not values:
        return ConstantTrend(
            param_name="unknown",
            trend_type=ParameterTrendType.CONSTANT,
            value=0.0
        )
    
    if method == "mean":
        value = float(np.mean(values))
    elif method == "median":
        value = float(np.median(values))
    elif method == "min":
        value = float(np.min(values))
    elif method == "max":
        value = float(np.max(values))
    elif method == "p25" and len(values) >= 4:
        value = float(np.percentile(values, 25))
    elif method == "p75" and len(values) >= 4:
        value = float(np.percentile(values, 75))
    else:
        value = float(np.mean(values))
        
    variance = float(np.var(values)) if len(values) > 1 else None
    
    return ConstantTrend(
        param_name="unknown",
        trend_type=ParameterTrendType.CONSTANT,
        value=value,
        variance=variance
    )

def estimate_parameter_trends(all_params: Dict[str, LogisticFitParams], 
                              config: ForecastConfig) -> AbilityTrend:
    """
    Estimate parameter trends based on historical data and configuration.
    
    Args:
        all_params: Dictionary mapping model names to their LogisticFitParams
        config: Forecast configuration
        
    Returns:
        AbilityTrend object with estimated trends for threshold and slope
    """
    # Extract historical parameter values
    params_list = list(all_params.values())
    params_list.sort(key=lambda p: p.date)
    
    dates = [p.date for p in params_list]
    thresholds = [p.threshold for p in params_list]
    slopes = [p.slope for p in params_list]
    
    base_date = min(dates) if dates else datetime.now()
    
    # Log the historical data
    logging.info(f"Estimating trends from {len(params_list)} models")
    logging.info(f"Date range: {min(dates)} to {max(dates)}")
    logging.info(f"Threshold range: {min(thresholds):.2f} to {max(thresholds):.2f}")
    logging.info(f"Slope range: {min(slopes):.2f} to {max(slopes):.2f}")
    logging.info(f"Slope values: {', '.join([f'{s:.2f}' for s in slopes])}")
    
    # Check if slopes are consistently negative
    if not all(s < 0 for s in slopes):
        logging.warning("Some slope values are non-negative, which contradicts expectations")
        logging.warning("Models with non-negative slopes: " + 
                       ", ".join([p.model for p in params_list if p.slope >= 0]))
    
    # Estimate threshold trend based on configuration
    if config.threshold_trend_type == ParameterTrendType.LINEAR:
        threshold_trend, t_meta = estimate_linear_trend(dates, thresholds)
        threshold_trend.param_name = "threshold"
        logging.info(f"Estimated linear threshold trend: "
                    f"{threshold_trend.intercept:.2f} + {threshold_trend.slope:.2f}t per year")
        if threshold_trend.intercept_ci:
            logging.info(f"Threshold intercept 95% CI: "
                        f"({threshold_trend.intercept_ci[0]:.2f}, {threshold_trend.intercept_ci[1]:.2f})")
            logging.info(f"Threshold slope 95% CI: "
                        f"({threshold_trend.slope_ci[0]:.2f}, {threshold_trend.slope_ci[1]:.2f})")
    
    elif config.threshold_trend_type == ParameterTrendType.CONSTANT:
        threshold_trend = estimate_constant_trend(thresholds, "mean")
        threshold_trend.param_name = "threshold"
        logging.info(f"Using constant threshold value: {threshold_trend.value:.2f}")
    
    elif config.threshold_trend_type == ParameterTrendType.CYCLIC:
        base_trend, _ = estimate_linear_trend(dates, thresholds)
        threshold_trend = CyclicTrend(
            param_name="threshold",
            trend_type=ParameterTrendType.CYCLIC,
            base_trend=base_trend,
            period=config.cycle_period,
            amplitude=config.cycle_amplitude
        )
        logging.info(f"Using cyclic threshold trend with period {config.cycle_period} years "
                    f"and amplitude {config.cycle_amplitude}")
    
    elif config.threshold_trend_type == ParameterTrendType.RANDOM_WALK:
        base_trend, _ = estimate_linear_trend(dates, thresholds)
        threshold_trend = RandomWalkTrend(
            param_name="threshold",
            trend_type=ParameterTrendType.RANDOM_WALK,
            base_trend=base_trend,
            std_dev=config.random_walk_std
        )
        logging.info(f"Using random walk threshold trend with std_dev {config.random_walk_std}")
    
    # Estimate slope trend based on configuration
    if config.slope_trend_type == ParameterTrendType.LINEAR:
        slope_trend, s_meta = estimate_linear_trend(dates, slopes)
        slope_trend.param_name = "slope"
        logging.info(f"Estimated linear slope trend: "
                   f"{slope_trend.intercept:.2f} + {slope_trend.slope:.2f}t per year")
        if slope_trend.intercept_ci:
            logging.info(f"Slope intercept 95% CI: "
                        f"({slope_trend.intercept_ci[0]:.2f}, {slope_trend.intercept_ci[1]:.2f})")
            logging.info(f"Slope slope 95% CI: "
                        f"({slope_trend.slope_ci[0]:.2f}, {slope_trend.slope_ci[1]:.2f})")
    
    elif config.slope_trend_type == ParameterTrendType.CONSTANT:
        slope_trend = estimate_constant_trend(slopes, config.slope_constant_type)
        slope_trend.param_name = "slope"
        logging.info(f"Using constant slope value ({config.slope_constant_type}): {slope_trend.value:.2f}")
    
    elif config.slope_trend_type == ParameterTrendType.CYCLIC:
        base_trend, _ = estimate_linear_trend(dates, slopes)
        slope_trend = CyclicTrend(
            param_name="slope",
            trend_type=ParameterTrendType.CYCLIC,
            base_trend=base_trend,
            period=config.cycle_period,
            amplitude=config.cycle_amplitude / 2  # Usually smaller amplitude for slope
        )
        logging.info(f"Using cyclic slope trend with period {config.cycle_period} years "
                   f"and amplitude {config.cycle_amplitude/2}")
    
    elif config.slope_trend_type == ParameterTrendType.RANDOM_WALK:
        base_trend, _ = estimate_linear_trend(dates, slopes)
        slope_trend = RandomWalkTrend(
            param_name="slope",
            trend_type=ParameterTrendType.RANDOM_WALK,
            base_trend=base_trend,
            std_dev=config.random_walk_std / 2  # Usually smaller std_dev for slope
        )
        logging.info(f"Using random walk slope trend with std_dev {config.random_walk_std/2}")
    
    return AbilityTrend(
        threshold_trend=threshold_trend,
        slope_trend=slope_trend,
        base_date=base_date,
        historic_dates=dates,
        historic_thresholds=thresholds,
        historic_slopes=slopes
    )

def generate_forecasts(
    ability_trend: AbilityTrend, 
    config: ForecastConfig
) -> List[AbilityForecast]:
    """
    Generate forecasts based on parameter trends and forecast configuration.
    
    Args:
        ability_trend: AbilityTrend object with estimated trends
        config: Forecast configuration
    
    Returns:
        List of AbilityForecast objects for future hypothetical models
    """
    # Create date range for forecast
    forecast_dates = pd.date_range(start=config.start_date, end=config.end_date, freq=config.frequency)
    
    # Set random seed for reproducibility
    random_state = np.random.RandomState(config.random_seed)
    
    # Get mean slope for validation
    avg_slope = np.mean(ability_trend.historic_slopes)
    
    forecasts = []
    
    # Generate forecasts for each date
    for date in forecast_dates:
        # Forecast threshold parameter
        threshold = ability_trend.threshold_trend.forecast(
            ability_trend.base_date, date, random_state)
            
        # Forecast slope parameter
        slope = ability_trend.slope_trend.forecast(
            ability_trend.base_date, date, random_state)
        
        # Ensure slope is negative (logistic curve requirement)
        if slope >= 0:
            logging.warning(f"Forecast produced invalid positive slope: {slope}. Fixing to negative value.")
            # Use the average slope from historical data, making it negative if needed
            slope = avg_slope if avg_slope < 0 else -abs(avg_slope)
        
        # Create scenario name based on trend types
        scenario = f"{config.threshold_trend_type.value}_threshold"
        
        if config.slope_trend_type == ParameterTrendType.CONSTANT:
            scenario += f"_{config.slope_trend_type.value}_{config.slope_constant_type}_slope"
        else:
            scenario += f"_{config.slope_trend_type.value}_slope"
            
        scenario += f"_{config.frequency}"
        
        # Create forecast data dictionary
        forecast_data = {
            "date": date,
            "threshold": float(threshold),
            "slope": float(slope),
            "scenario": scenario,
            "model": f"future_model_{date.strftime('%Y%m%d')}"
        }
        
        # Add confidence intervals if available for linear trends
        if isinstance(ability_trend.threshold_trend, LinearTrend):
            threshold_ci = ability_trend.threshold_trend.forecast_ci(ability_trend.base_date, date)
            if threshold_ci:
                forecast_data["threshold_ci_lower"] = threshold_ci[0]
                forecast_data["threshold_ci_upper"] = threshold_ci[1]
                
        if isinstance(ability_trend.slope_trend, LinearTrend):
            slope_ci = ability_trend.slope_trend.forecast_ci(ability_trend.base_date, date)
            if slope_ci:
                forecast_data["slope_ci_lower"] = slope_ci[0]
                forecast_data["slope_ci_upper"] = slope_ci[1]
        
        # Create AbilityForecast object
        forecast = AbilityForecast(**forecast_data)
        forecasts.append(forecast)
    
    return forecasts

def read_curve_params(file_path: str) -> Dict[str, LogisticFitParams]:
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
    
    # Generate forecasts for various trend combinations
    all_forecasts = []
    
    # Define the trend combinations to generate
    trend_combinations = [
        # Linear threshold with various slope scenarios
        {"threshold_trend_type": ParameterTrendType.LINEAR, "slope_trend_type": ParameterTrendType.CONSTANT, "slope_constant_type": "mean"},
        {"threshold_trend_type": ParameterTrendType.LINEAR, "slope_trend_type": ParameterTrendType.CONSTANT, "slope_constant_type": "median"},
        {"threshold_trend_type": ParameterTrendType.LINEAR, "slope_trend_type": ParameterTrendType.CONSTANT, "slope_constant_type": "min"},
        {"threshold_trend_type": ParameterTrendType.LINEAR, "slope_trend_type": ParameterTrendType.CONSTANT, "slope_constant_type": "max"},
        {"threshold_trend_type": ParameterTrendType.LINEAR, "slope_trend_type": ParameterTrendType.LINEAR},
        
        # Cyclic threshold with constant slope
        {"threshold_trend_type": ParameterTrendType.CYCLIC, "slope_trend_type": ParameterTrendType.CONSTANT, "slope_constant_type": "mean"},
        
        # Random walk for both parameters
        {"threshold_trend_type": ParameterTrendType.RANDOM_WALK, "slope_trend_type": ParameterTrendType.CONSTANT, "slope_constant_type": "mean"},
    ]
    
    for trend_combo in trend_combinations:
        for freq in ['YE', 'QE', 'ME']:
            # Create a config for this combination
            combo_config_args = {**config_dict, **trend_combo, 'frequency': freq}
            combo_config = ForecastConfig(**combo_config_args)
            
            # Generate the trend estimates
            ability_trend = estimate_parameter_trends(curve_params, combo_config)
            
            # Generate forecasts using the estimated trends
            logging.info(f"Generating forecasts with {combo_config.threshold_trend_type.value} threshold and "
                       f"{combo_config.slope_trend_type.value} slope at {freq} frequency")
            forecasts = generate_forecasts(ability_trend, combo_config)
            all_forecasts.extend(forecasts)
    
    logging.info(f"Generated {len(all_forecasts)} forecasts")
    
    logging.info(f"Saving forecasts to {args.out}")
    save_forecasts(all_forecasts, args.out)
    logging.info("Complete")

if __name__ == "__main__":
    main()
